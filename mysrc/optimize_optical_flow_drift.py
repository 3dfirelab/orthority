#!/usr/bin/env python3
"""Generate robust optical-flow links and smooth per-image drift targets.

The resulting map-shift targets are measurements for the next, camera-model
stage that converts them into time-varying roll/pitch/yaw corrections.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml
from rasterio.warp import reproject
from scipy.optimize import least_squares, minimize
from shapely.geometry import Point


@dataclass(frozen=True)
class Frame:
    path: Path
    time: pd.Timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Optical-flow drift YAML")
    parser.add_argument("--max-frames", type=int, help="Limit frames for a test")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-opk", action="store_true", help="Only create and fit optical-flow links")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    required = {"reference_dir", "output_dir", "link_offsets_s", "spline_knot_spacing_s"}
    if not isinstance(config, dict) or required - set(config):
        raise ValueError(f"Invalid optical-flow configuration: {path}")
    return config


def discover_frames(directory: Path) -> list[Frame]:
    frames = []
    for path in directory.glob("*_ORTHO.tif"):
        with rasterio.open(path) as dataset:
            acquisition_time = dataset.tags().get("Time")
        if acquisition_time:
            frames.append(Frame(path, pd.Timestamp(acquisition_time, tz="UTC")))
    frames.sort(key=lambda frame: frame.time)
    if len(frames) < 2:
        raise ValueError(f"Fewer than two timestamped ortho TIFFs in {directory}")
    return frames


def common_grid_images(reference: Frame, candidate: Frame) -> tuple[np.ndarray, np.ndarray, rasterio.Affine]:
    with rasterio.open(reference.path) as ref, rasterio.open(candidate.path) as src:
        reference_image = ref.read(1, out_dtype="float32")
        reference_valid = (ref.read_masks(1) > 0) & np.isfinite(reference_image)
        candidate_image = np.full(reference_image.shape, np.nan, dtype=np.float32)
        reproject(rasterio.band(src, 1), candidate_image, src_transform=src.transform,
                  src_crs=src.crs, src_nodata=src.nodata, dst_transform=ref.transform,
                  dst_crs=ref.crs, dst_nodata=np.nan)
    return np.stack((reference_image, candidate_image)), reference_valid & np.isfinite(candidate_image), ref.transform


def to_uint8(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    low, high = np.percentile(image[valid], (2, 98))
    if not np.isfinite(low) or high <= low:
        raise ValueError("Image has insufficient contrast")
    result = np.zeros(image.shape, dtype=np.uint8)
    result[valid] = np.clip((image[valid] - low) * 255 / (high - low), 0, 255).astype(np.uint8)
    return result


def track_link(reference: Frame, candidate: Frame, config: dict) -> dict | None:
    images, valid, transform = common_grid_images(reference, candidate)
    margin = int(config["border_margin_px"])
    mask = cv2.erode(valid.astype(np.uint8) * 255, np.ones((2 * margin + 1, 2 * margin + 1), np.uint8))
    source, destination = to_uint8(images[0], valid), to_uint8(images[1], valid)
    points = cv2.goodFeaturesToTrack(source, maxCorners=int(config["max_features_per_image"]),
        qualityLevel=float(config["feature_quality_level"]), minDistance=float(config["feature_min_distance_px"]),
        mask=mask, blockSize=int(config["feature_block_size_px"]))
    if points is None:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(config["flow_max_iterations"]), float(config["flow_epsilon"]))
    kwargs = {"winSize": (int(config["flow_window_px"]),) * 2, "maxLevel": int(config["flow_pyramid_levels"]), "criteria": criteria}
    forward, status, error = cv2.calcOpticalFlowPyrLK(source, destination, points, None, **kwargs)
    backward, reverse_status, _ = cv2.calcOpticalFlowPyrLK(destination, source, forward, None, **kwargs)
    p0, p1, pback = points.reshape(-1, 2), forward.reshape(-1, 2), backward.reshape(-1, 2)
    displacement = p1 - p0
    keep = (status.ravel() > 0) & (reverse_status.ravel() > 0) & np.isfinite(displacement).all(axis=1)
    keep &= error.ravel() <= float(config["max_lk_error_px"])
    keep &= np.linalg.norm(pback - p0, axis=1) <= float(config["max_forward_backward_error_px"])
    keep &= np.linalg.norm(displacement, axis=1) <= float(config["max_displacement_px"])
    p0, displacement = p0[keep], displacement[keep]
    minimum = int(config["min_link_observations"])
    if len(displacement) < minimum:
        return None
    rows, cols = int(config["grid_rows"]), int(config["grid_cols"])
    cell = np.minimum((p0[:, 1] * rows / source.shape[0]).astype(int), rows - 1) * cols + np.minimum((p0[:, 0] * cols / source.shape[1]).astype(int), cols - 1)
    occupied = sum(np.count_nonzero(cell == number) >= int(config["min_features_per_occupied_cell"]) for number in range(rows * cols))
    if occupied < int(config["min_occupied_cells_per_link"]):
        return None
    offset = np.median(displacement, axis=0)
    keep = np.linalg.norm(displacement - offset, axis=1) <= float(config["max_final_feature_residual_px"])
    if np.count_nonzero(keep) < minimum:
        return None
    offset = np.median(displacement[keep], axis=0)
    apply = -offset
    feature_rows = []
    for feature_id, (point, motion) in enumerate(zip(p0[keep], displacement[keep])):
        estimated = point + motion
        map_x, map_y = transform * (float(estimated[0]), float(estimated[1]))
        feature_rows.append({"feature_id": feature_id, "image_x": float(point[0]),
            "image_y": float(point[1]), "estimated_x": float(estimated[0]),
            "estimated_y": float(estimated[1]), "estimated_map_x": float(map_x),
            "estimated_map_y": float(map_y), "residual_px": float(np.linalg.norm(motion - offset))})
    return {"reference_time": reference.time.isoformat(), "time": candidate.time.isoformat(),
            "offset_s": (candidate.time - reference.time).total_seconds(), "features_detected": len(points),
            "features_retained": int(np.count_nonzero(keep)), "occupied_cells": int(occupied),
            "flow_dx_px": float(offset[0]), "flow_dy_px": float(offset[1]),
            "apply_dx_px": float(apply[0]), "apply_dy_px": float(apply[1]), "apply_norm_px": float(np.hypot(*apply)),
            "apply_map_x": float(transform.a * apply[0] + transform.b * apply[1]),
            "apply_map_y": float(transform.d * apply[0] + transform.e * apply[1]),
            "feature_rows": feature_rows}


def nearest_index(times: np.ndarray, target: int) -> int | None:
    index = int(np.argmin(np.abs(times - target)))
    return index if abs(times[index] - target) <= 60_000_000 else None


def fit_spline(frames: list[Frame], links: pd.DataFrame, spacing_s: float, curvature: float) -> pd.DataFrame:
    times = np.array([frame.time.value for frame in frames], dtype=np.int64)
    elapsed = (times - times[0]) / 1e9
    knots = np.arange(0, elapsed[-1] + spacing_s, spacing_s)
    def basis(value: float) -> np.ndarray:
        index = min(int(value // spacing_s), len(knots) - 2)
        fraction = (value - knots[index]) / spacing_s
        row = np.zeros(len(knots)); row[index:index + 2] = (1 - fraction, fraction)
        return row
    B = np.vstack([basis(value) for value in elapsed])
    rows, x, y = [], [], []
    for link in links.itertuples():
        first = nearest_index(times, pd.Timestamp(link.reference_time).value)
        second = nearest_index(times, pd.Timestamp(link.time).value)
        if first is not None and second is not None:
            rows.append(B[second] - B[first]); x.append(link.apply_map_x); y.append(link.apply_map_y)
    A = np.vstack(rows)
    smooth = np.diff(np.eye(len(knots)), n=2, axis=0) * np.sqrt(curvature)
    anchor = np.zeros((1, len(knots))); anchor[0, 0] = 1000.0
    system = np.vstack((A, smooth, anchor))
    def solve(values: list[float]) -> np.ndarray:
        target = np.r_[values, np.zeros(len(smooth) + 1)]
        return least_squares(lambda result: system @ result - target, np.zeros(len(knots)), loss="huber").x
    return pd.DataFrame({"time": [frame.time.isoformat() for frame in frames],
        "target_map_x": B @ solve(x), "target_map_y": B @ solve(y), "spline_knot_spacing_s": spacing_s})


def frame_id(path: Path) -> int | None:
    numbers = re.findall(r"\d+", path.stem)
    return int(numbers[-1]) if numbers else None


def solve_opk_knots(frames: list[Frame], config: dict, output: Path) -> pd.DataFrame:
    """Reuse estimate_imu_drift's calibrated local OPK objective at each knot."""
    # These imports pull in Orthority and are deliberately delayed: feature-link
    # extraction can run in the lighter tracking environment.
    from calibrate_orthority_imu import (
        CalibrationObjective, ImagePair, _initial_simplex, _load_calibration,
        _load_imu, _prepare_reference,
    )
    from estimate_imu_drift import (
        OPK_INDICES, RAW_PATTERN, REFERENCE_PATTERN, _indexed_files,
        _local_bounds,
    )
    raw_files = _indexed_files(Path(config["input_dir"]), RAW_PATTERN, int(config.get("filter", 1)))
    reference_files = _indexed_files(Path(config["reference_dir"]), REFERENCE_PATTERN, int(config.get("filter", 1)))
    baseline = _load_calibration(Path(config["calibration"]))
    opk_range = np.asarray(config.get("opk_range_deg", [2.0, 2.0, 2.0]), dtype=float)
    lower, upper = _local_bounds(baseline, np.ones(3), opk_range)
    imu = _load_imu(Path(config["imu"]))
    with Path(config["calibration"]).open() as source:
        pose_model = json.load(source)["pose_model"]
    first_time = frames[0].time
    spacing = float(config["spline_knot_spacing_s"])
    offset = pd.Timedelta(seconds=float(config.get("opk_pair_offset_s", 1.0)))
    rows = []
    for knot_seconds in np.arange(spacing, (frames[-1].time - first_time).total_seconds(), spacing):
        candidate = min(frames, key=lambda item: abs((item.time - (first_time + pd.Timedelta(seconds=float(knot_seconds)))).total_seconds()))
        reference = min(frames, key=lambda item: abs((item.time - (candidate.time - offset)).total_seconds()))
        candidate_id, reference_id = frame_id(candidate.path), frame_id(reference.path)
        if candidate_id not in raw_files or reference_id not in reference_files:
            continue
        pair = ImagePair(image=raw_files[candidate_id], reference=reference_files[reference_id])
        prepared = _prepare_reference(pair, int(config.get("max_dimension", 1024)))
        objective_config = {
            "flight_name": config["flightname"], "pose_model": pose_model,
            "int_param": Path(config["int_param"]), "dem": Path(config["dem"]),
            "work_dir": output / "opk_work" / f"{reference_id:09d}_{candidate_id:09d}",
            "overlap_penalty": float(config.get("overlap_penalty", 0.5)),
            "initial_step_xyz": [0.25, 0.25, 0.25],
            "initial_step_opk": [0.1, 0.1, 0.1],
        }
        objective = CalibrationObjective(objective_config, imu, [prepared], lower, upper,
            fixed_parameters=baseline, active_indices=OPK_INDICES)
        initial = (baseline[OPK_INDICES] - lower[OPK_INDICES]) / (upper[OPK_INDICES] - lower[OPK_INDICES])
        result = minimize(objective, initial, method="Nelder-Mead", bounds=[(0.0, 1.0)] * 3,
            options={"initial_simplex": _initial_simplex(initial, lower, upper, objective_config, active_indices=OPK_INDICES),
                     "maxiter": int(config.get("opk_max_iterations", 150)),
                     "xatol": float(config.get("opk_parameter_tolerance", 0.001)),
                     "fatol": float(config.get("opk_cost_tolerance", 0.0001)), "adaptive": True})
        local = objective.physical_parameters(result.x)
        rows.append({"time": candidate.time.isoformat(), "reference_time": reference.time.isoformat(),
            "reference_frame_id": reference_id, "image_frame_id": candidate_id, "cost": float(result.fun),
            "success": bool(result.success), "evaluations": int(result.nfev),
            "extra_omega": float(local[3] - baseline[3]), "extra_phi": float(local[4] - baseline[4]),
            "extra_kappa": float(local[5] - baseline[5])})
        print(f"OPK knot {len(rows)}: {reference_id} -> {candidate_id}, cost={result.fun:.5f}", flush=True)
    if not rows:
        raise ValueError("No valid raw/reference pairs were available for OPK knots")
    return pd.DataFrame(rows)


def interpolate_opk(frames: list[Frame], knots: pd.DataFrame) -> pd.DataFrame:
    frame_times = np.array([frame.time.value for frame in frames], dtype=np.int64)
    knot_times = pd.to_datetime(knots["time"], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    result = {"time": [frame.time.isoformat() for frame in frames]}
    for name in ("omega", "phi", "kappa"):
        result[f"extra_{name}"] = np.interp(frame_times, knot_times, knots[f"extra_{name}"].to_numpy(float))
    return pd.DataFrame(result)


def save_per_knot_gpkgs(output: Path, feature_rows: list[dict], frames: list[Frame], spacing_s: float) -> None:
    by_knot: dict[int, list[dict]] = {}
    for row in feature_rows:
        by_knot.setdefault(int(round(pd.Timestamp(row["knot_time"]).value / 1e9)), []).append(row)
    for knot_seconds, rows in by_knot.items():
        frame = min(frames, key=lambda item: abs(item.time.value / 1e9 - knot_seconds))
        name = frame.path.stem.replace("_expcorr_ORTHO", "") + "_features.gpkg"
        path = output / name
        for row in rows:
            row["feature_count"] = len(rows)
        gdf = gpd.GeoDataFrame(rows, geometry=[Point(r["estimated_map_x"], r["estimated_map_y"]) for r in rows], crs="EPSG:4326")
        gdf.to_file(path, layer="features", driver="GPKG")
    print(f"Wrote {len(by_knot)} per-knot feature GeoPackages")


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    frames = discover_frames(Path(config["reference_dir"]))
    if args.max_frames:
        frames = frames[:args.max_frames]
    output = Path(config["output_dir"])
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"Output directory is non-empty: {output}; pass --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    times = np.array([frame.time.value for frame in frames], dtype=np.int64)
    links = []
    feature_rows = []
    knot_indices = set()
    spacing = float(config["spline_knot_spacing_s"])
    for elapsed in np.arange(0, (times[-1] - times[0]) / 1e9 + spacing, spacing):
        knot_indices.add(int(np.argmin(np.abs(times - (times[0] + int(elapsed * 1e9))))))
    for index, reference in enumerate(frames):
        for offset_s in config["link_offsets_s"]:
            candidate_index = nearest_index(times, reference.time.value + int(float(offset_s) * 1e9))
            if candidate_index is not None and index < candidate_index:
                link = track_link(reference, frames[candidate_index], config)
                if link:
                    details = link.pop("feature_rows", [])
                    if index in knot_indices:
                        for detail in details:
                            detail.update({"knot_time": reference.time.isoformat(), "offset_s": link["offset_s"]})
                            feature_rows.append(detail)
                    links.append(link)
        if index % 25 == 0 or index + 1 == len(frames):
            print(f"Tracked {index + 1}/{len(frames)}; accepted links: {len(links)}", flush=True)
    if not links:
        raise ValueError("No links passed the configured feature rejection checks")
    link_table = pd.DataFrame(links)
    targets = fit_spline(frames, link_table, float(config["spline_knot_spacing_s"]), float(config["spline_curvature_weight"]))
    link_table.to_csv(output / "optical_flow_links.csv", index=False)
    targets.to_csv(output / "map_shift_targets.csv", index=False)
    save_per_knot_gpkgs(output, feature_rows, frames, spacing)
    if not args.skip_opk:
        opk_knots = solve_opk_knots(frames, config, output)
        opk_knots.to_csv(output / "opk_knot_corrections.csv", index=False)
        interpolate_opk(frames, opk_knots).to_csv(output / "opk_corrections.csv", index=False)
    (output / "run_summary.json").write_text(json.dumps({"frames": len(frames), "accepted_links": len(links),
        "median_link_norm_px": float(link_table.apply_norm_px.median()), "opk_solver": "estimate_imu_drift_local_knots"}, indent=2) + "\n")
    print(f"Wrote {len(links)} links and {len(targets)} map-shift targets to {output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, cv2.error, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
