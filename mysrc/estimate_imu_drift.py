#!/usr/bin/env python3
"""Estimate time-varying IMU XYZ/OPK drift from pairs of Orthority images.

For each window, a later raw image is orthorectified repeatedly and compared
with an earlier fixed image from ``reference_dir``. Their frame IDs differ by
``pair_separation``.

The optimizer starts from the global correction produced by
``calibrate_orthority_imu.py``. Reported drift values are:

    local correction - global correction

By default, windows advance by the same number of IDs as the pair separation:
reference 1 -> raw 11, reference 11 -> raw 21, ...
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.optimize import minimize

import imuNcOoGeojson
from calibrate_orthority_imu import (
    CalibrationObjective,
    ImagePair,
    _initial_simplex,
    _candidate_on_reference,
    _load_calibration,
    _load_imu,
    _prepare_reference,
)


RAW_PATTERN = re.compile(r"^f(?P<filter>\d+)-(?P<id>\d+)\.tif$")
REFERENCE_PATTERN = re.compile(
    r"^f(?P<filter>\d+)-(?P<id>\d+)(?:_expcorr)?_ORTHO\.tif$"
)
PARAMETER_NAMES = ("x", "y", "z", "omega", "phi", "kappa")


def _config_path(config_file: Path, config: dict, key: str) -> Path:
    if key not in config:
        raise ValueError(f"Missing required configuration key: {key}")
    path = Path(config[key]).expanduser()
    if not path.is_absolute():
        path = config_file.resolve().parent / path
    return path.resolve()


def _load_dataset_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError("The dataset configuration must be a YAML mapping.")

    data_root = Path(config["dirTelops"]).expanduser().resolve().parent
    transect = config["extractionName"]
    flight = config["flightname"]
    imu_name = config.get("imufile_name", "safire")
    filter_id = int(config.get("filter", 1))

    calibration_path = _config_path(path, config, "calibration")
    with calibration_path.open("r", encoding="utf-8") as calibration_file:
        calibration_metadata = json.load(calibration_file)
    pose_model = calibration_metadata.get("pose_model")
    if pose_model not in imuNcOoGeojson.SUPPORTED_POSE_MODELS:
        raise ValueError(
            f"{calibration_path} has no supported pose_model."
        )

    return {
        "flight_name": flight,
        "filter": filter_id,
        "input_dir": _config_path(path, config, "drift_input_dir"),
        "reference_dir": _config_path(path, config, "drift_reference_dir"),
        "output_dir": _config_path(path, config, "drift_output_dir"),
        "calibration": calibration_path,
        "pose_model": pose_model,
        "imu": data_root / "safire" / f"{flight}_{imu_name}.gpkg",
        "int_param": (
            data_root / "Transects" / transect / "io" / f"{flight}_int_param.yaml"
        ),
        "dem": data_root / "dem" / f"{flight}_dem_1m.tif",
        "pair_separation": int(config.get("drift_pair_separation", 10)),
        "window_step": int(config.get("drift_window_step", 10)),
        "max_iterations": int(config.get("drift_max_iterations", 150)),
        "max_dimension": int(config.get("drift_max_dimension", 1024)),
        "xyz_range": np.asarray(
            config.get("drift_xyz_range", [5.0, 5.0, 5.0]), dtype=float
        ),
        "opk_range": np.asarray(
            config.get("drift_opk_range", [2.0, 2.0, 2.0]), dtype=float
        ),
        "initial_step_xyz": config.get(
            "drift_initial_step_xyz", [0.25, 0.25, 0.25]
        ),
        "initial_step_opk": config.get(
            "drift_initial_step_opk", [0.1, 0.1, 0.1]
        ),
        "parameter_tolerance": float(
            config.get("drift_parameter_tolerance", 1.0e-3)
        ),
        "cost_tolerance": float(config.get("drift_cost_tolerance", 1.0e-4)),
        "overlap_penalty": float(config.get("drift_overlap_penalty", 0.5)),
        "ecc_iterations": int(config.get("drift_ecc_iterations", 200)),
        "ecc_epsilon": float(config.get("drift_ecc_epsilon", 1.0e-6)),
        "flow_max_corners": int(config.get("drift_flow_max_corners", 1000)),
        "flow_quality_level": float(
            config.get("drift_flow_quality_level", 0.01)
        ),
        "flow_min_distance": float(
            config.get("drift_flow_min_distance", 7.0)
        ),
    }


def _indexed_files(directory: Path, pattern: re.Pattern, filter_id: int) -> dict:
    indexed = {}
    for path in directory.glob("*.tif"):
        match = pattern.match(path.name)
        if match and int(match.group("filter")) == filter_id:
            indexed[int(match.group("id"))] = path
    return indexed


def _build_windows(
    raw_files: dict[int, Path],
    reference_files: dict[int, Path],
    separation: int,
    step: int,
) -> list[tuple[int, int]]:
    if separation <= 0 or step <= 0:
        raise ValueError("Pair separation and window step must be positive.")
    available = sorted(set(raw_files) & set(reference_files))
    if not available:
        raise ValueError("No matching raw/reference frame IDs were found.")

    first = available[0]
    starts = range(first, available[-1] - separation + 1, step)
    windows = [
        (start, start + separation)
        for start in starts
        if start in raw_files
        and start in reference_files
        and start + separation in raw_files
        and start + separation in reference_files
    ]
    if not windows:
        raise ValueError("No complete frame pairs match the requested separation.")
    return windows


def _frame_time(path: Path) -> dt.datetime | None:
    with rasterio.open(path) as dataset:
        tags = dataset.tags()
    candidates = [tags.get("Time"), tags.get("TIFFTAG_DATETIME")]
    description = tags.get("TIFFTAG_IMAGEDESCRIPTION")
    if description:
        try:
            candidates.insert(0, json.loads(description).get("Time"))
        except (TypeError, json.JSONDecodeError):
            pass
    for value in candidates:
        if not value:
            continue
        for format_string in (
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
            "%y%m%d %H%M%S%f",
        ):
            try:
                return dt.datetime.strptime(value, format_string)
            except ValueError:
                continue
    return None


def _local_bounds(
    baseline: np.ndarray, xyz_range: np.ndarray, opk_range: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if xyz_range.shape != (3,) or opk_range.shape != (3,):
        raise ValueError("Drift XYZ and OPK ranges must each contain 3 values.")
    ranges = np.concatenate((xyz_range, opk_range))
    if np.any(ranges <= 0):
        raise ValueError("Drift XYZ and OPK ranges must be positive.")
    return baseline - ranges, baseline + ranges


def _write_checkpoint(rows: list[dict], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "imu_drift_timeseries.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    return csv_path


def _ecc_translation(
    reference,
    candidate_path: Path,
    iterations: int,
    epsilon: float,
) -> dict:
    """Return ECC translation for an existing RunOrtho image."""
    _, candidate_gradient, candidate_valid = _candidate_on_reference(
        candidate_path, reference
    )
    common = reference.valid & candidate_valid
    if np.count_nonzero(common) < 1000:
        raise ValueError("Too few common pixels for ECC image registration.")

    mask = common.astype(np.uint8) * 255
    mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
    template = reference.gradient.astype(np.float32)
    candidate = candidate_gradient.astype(np.float32)
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        iterations,
        epsilon,
    )
    correlation, warp = cv2.findTransformECC(
        template,
        candidate,
        warp,
        cv2.MOTION_TRANSLATION,
        criteria,
        inputMask=mask,
        gaussFiltSize=5,
    )

    offset_col = float(warp[0, 2])
    offset_row = float(warp[1, 2])
    apply_col = -offset_col
    apply_row = -offset_row
    apply_norm = float(np.hypot(apply_col, apply_row))
    transform = reference.transform
    apply_map_x = transform.a * apply_col + transform.b * apply_row
    apply_map_y = transform.d * apply_col + transform.e * apply_row
    return {
        "runortho_ecc_correlation": float(correlation),
        "runortho_offset_dx_px": offset_col,
        "runortho_offset_dy_px": offset_row,
        "runortho_shift_apply_dx_px": apply_col,
        "runortho_shift_apply_dy_px": apply_row,
        "runortho_shift_apply_norm_px": apply_norm,
        "runortho_shift_apply_map_x": float(apply_map_x),
        "runortho_shift_apply_map_y": float(apply_map_y),
        "runortho_shift_map_units": reference.crs.to_string(),
    }


def _normalize_uint8(image: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = image[valid]
    if values.size < 1000:
        raise ValueError("Too few valid pixels for optical flow.")
    low, high = np.percentile(values, (2, 98))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Insufficient image contrast for optical flow.")
    result = np.zeros(image.shape, dtype=np.uint8)
    result[valid] = np.clip(
        (image[valid] - low) * 255.0 / (high - low), 0, 255
    ).astype(np.uint8)
    return result


def _optical_flow_translation(
    reference,
    candidate_path: Path,
    max_corners: int,
    quality_level: float,
    min_distance: float,
) -> dict:
    """Estimate translation from the nearest 90% of LK feature distances."""
    candidate_image, _, candidate_valid = _candidate_on_reference(
        candidate_path, reference
    )
    common = reference.valid & candidate_valid
    if np.count_nonzero(common) < 1000:
        raise ValueError("Too few common pixels for optical-flow registration.")

    mask = cv2.erode(
        common.astype(np.uint8) * 255, np.ones((7, 7), np.uint8)
    )
    reference_image = _normalize_uint8(reference.image, common)
    candidate = _normalize_uint8(candidate_image, common)
    reference_image = cv2.GaussianBlur(reference_image, (5, 5), 0)
    candidate = cv2.GaussianBlur(candidate, (5, 5), 0)

    points = cv2.goodFeaturesToTrack(
        reference_image,
        maxCorners=max_corners,
        qualityLevel=quality_level,
        minDistance=min_distance,
        mask=mask,
        blockSize=7,
    )
    if points is None or len(points) < 3:
        raise ValueError("Fewer than three features were detected.")

    tracked, status, _ = cv2.calcOpticalFlowPyrLK(
        reference_image,
        candidate,
        points,
        None,
        winSize=(31, 31),
        maxLevel=4,
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            50,
            0.001,
        ),
    )
    valid = status.reshape(-1).astype(bool)
    source_points = points.reshape(-1, 2)[valid]
    tracked_points = tracked.reshape(-1, 2)[valid]
    displacements = tracked_points - source_points
    finite = np.all(np.isfinite(displacements), axis=1)
    displacements = displacements[finite]
    if len(displacements) < 3:
        raise ValueError("Fewer than three valid optical-flow tracks remain.")

    distances = np.linalg.norm(displacements, axis=1)
    distance_p90 = float(np.percentile(distances, 90))
    retained = displacements[distances <= distance_p90]
    if len(retained) < 3:
        raise ValueError("Fewer than three tracks remain after 90% filtering.")

    offset_col, offset_row = np.median(retained, axis=0)
    apply_col = -float(offset_col)
    apply_row = -float(offset_row)
    apply_norm = float(np.hypot(apply_col, apply_row))
    transform = reference.transform
    apply_map_x = transform.a * apply_col + transform.b * apply_row
    apply_map_y = transform.d * apply_col + transform.e * apply_row
    return {
        "runortho_flow_feature_count": int(len(displacements)),
        "runortho_flow_retained_count": int(len(retained)),
        "runortho_flow_distance_p90_px": distance_p90,
        "runortho_flow_offset_dx_px": float(offset_col),
        "runortho_flow_offset_dy_px": float(offset_row),
        "runortho_flow_shift_apply_dx_px": apply_col,
        "runortho_flow_shift_apply_dy_px": apply_row,
        "runortho_flow_shift_apply_norm_px": apply_norm,
        "runortho_flow_shift_apply_map_x": float(apply_map_x),
        "runortho_flow_shift_apply_map_y": float(apply_map_y),
    }


def _plot_timeseries(table: pd.DataFrame, output_dir: Path) -> Path:
    if table.empty:
        raise ValueError("No successful drift estimates are available to plot.")
    if table["time"].notna().any():
        x_values = pd.to_datetime(table["time"])
        x_label = "Image time"
    else:
        x_values = table["mid_frame_id"]
        x_label = "Midpoint frame ID"

    figure, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    for name in ("x", "y", "z"):
        axes[0].plot(x_values, table[f"extra_{name}"], marker="o", label=name)
    axes[0].axhline(0, color="black", linewidth=0.8)
    axes[0].set_ylabel("Additional XYZ correction (m)")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.3)

    for name in ("omega", "phi", "kappa"):
        axes[1].plot(x_values, table[f"extra_{name}"], marker="o", label=name)
    axes[1].axhline(0, color="black", linewidth=0.8)
    axes[1].set_ylabel("Additional OPK correction (degrees)")
    axes[1].legend(ncol=3)
    axes[1].grid(alpha=0.3)

    axes[2].plot(
        x_values,
        table["runortho_shift_apply_dx_px"],
        marker="o",
        label="ECC dx",
    )
    axes[2].plot(
        x_values,
        table["runortho_shift_apply_dy_px"],
        marker="o",
        label="ECC dy",
    )
    axes[2].plot(
        x_values,
        table["runortho_shift_apply_norm_px"],
        marker="o",
        linewidth=2,
        label="ECC norm",
    )
    axes[2].plot(
        x_values,
        table["runortho_flow_shift_apply_dx_px"],
        marker=".",
        linestyle="--",
        label="flow dx",
    )
    axes[2].plot(
        x_values,
        table["runortho_flow_shift_apply_dy_px"],
        marker=".",
        linestyle="--",
        label="flow dy",
    )
    axes[2].plot(
        x_values,
        table["runortho_flow_shift_apply_norm_px"],
        marker=".",
        linestyle="--",
        linewidth=2,
        label="flow norm",
    )
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].set_ylabel("Current RunOrtho shift to apply (pixels)")
    axes[2].set_xlabel(x_label)
    axes[2].legend(ncol=3)
    axes[2].grid(alpha=0.3)
    figure.suptitle("IMU correction drift relative to global calibration")
    figure.autofmt_xdate()
    figure.tight_layout()

    plot_path = output_dir / "imu_drift_timeseries.png"
    figure.savefig(plot_path, dpi=180)
    plt.close(figure)
    return plot_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate additional XYZ/OPK correction over time from pairs of "
            "raw images and fixed full-ortho references."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--max-windows",
        type=int,
        help="Process only the first N windows for testing.",
    )
    parser.add_argument(
        "--maxiter",
        type=int,
        help="Override drift_max_iterations from the YAML file.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = _load_dataset_config(args.config)
        for key in (
            "input_dir",
            "reference_dir",
            "calibration",
            "imu",
            "int_param",
            "dem",
        ):
            if not Path(config[key]).exists():
                raise ValueError(f"{key} not found: {config[key]}")

        baseline = _load_calibration(config["calibration"])
        lower, upper = _local_bounds(
            baseline, config["xyz_range"], config["opk_range"]
        )
        raw_files = _indexed_files(
            config["input_dir"], RAW_PATTERN, config["filter"]
        )
        reference_files = _indexed_files(
            config["reference_dir"], REFERENCE_PATTERN, config["filter"]
        )
        windows = _build_windows(
            raw_files,
            reference_files,
            config["pair_separation"],
            config["window_step"],
        )
        if args.max_windows is not None:
            windows = windows[: args.max_windows]

        imu = _load_imu(config["imu"])
        output_dir = config["output_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        max_iterations = args.maxiter or config["max_iterations"]

        print(f"Global correction: {baseline.tolist()}")
        print(f"Processing {len(windows)} frame-pair windows.")
        for window_index, (first_id, second_id) in enumerate(windows, start=1):
            print(
                f"\nWindow {window_index}/{len(windows)}: "
                f"reference {first_id} -> raw {second_id}"
            )
            pairs = [
                ImagePair(
                    image=raw_files[second_id],
                    reference=reference_files[first_id],
                )
            ]
            references = [
                _prepare_reference(pair, config["max_dimension"]) for pair in pairs
            ]
            try:
                image_shift = _ecc_translation(
                    references[0],
                    reference_files[second_id],
                    config["ecc_iterations"],
                    config["ecc_epsilon"],
                )
            except (
                OSError,
                ValueError,
                cv2.error,
                rasterio.errors.RasterioError,
            ) as error:
                print(f"ECC registration failed: {error}", file=sys.stderr)
                image_shift = {
                    "runortho_ecc_correlation": float("nan"),
                    "runortho_offset_dx_px": float("nan"),
                    "runortho_offset_dy_px": float("nan"),
                    "runortho_shift_apply_dx_px": float("nan"),
                    "runortho_shift_apply_dy_px": float("nan"),
                    "runortho_shift_apply_norm_px": float("nan"),
                    "runortho_shift_apply_map_x": float("nan"),
                    "runortho_shift_apply_map_y": float("nan"),
                    "runortho_shift_map_units": references[0].crs.to_string(),
                }
            try:
                optical_flow_shift = _optical_flow_translation(
                    references[0],
                    reference_files[second_id],
                    config["flow_max_corners"],
                    config["flow_quality_level"],
                    config["flow_min_distance"],
                )
            except (
                OSError,
                ValueError,
                cv2.error,
                rasterio.errors.RasterioError,
            ) as error:
                print(
                    f"Optical-flow registration failed: {error}",
                    file=sys.stderr,
                )
                optical_flow_shift = {
                    "runortho_flow_feature_count": 0,
                    "runortho_flow_retained_count": 0,
                    "runortho_flow_distance_p90_px": float("nan"),
                    "runortho_flow_offset_dx_px": float("nan"),
                    "runortho_flow_offset_dy_px": float("nan"),
                    "runortho_flow_shift_apply_dx_px": float("nan"),
                    "runortho_flow_shift_apply_dy_px": float("nan"),
                    "runortho_flow_shift_apply_norm_px": float("nan"),
                    "runortho_flow_shift_apply_map_x": float("nan"),
                    "runortho_flow_shift_apply_map_y": float("nan"),
                }
            objective_config = {
                "flight_name": config["flight_name"],
                "pose_model": config["pose_model"],
                "int_param": config["int_param"],
                "dem": config["dem"],
                "work_dir": output_dir / "work" / f"{first_id:09d}_{second_id:09d}",
                "overlap_penalty": config["overlap_penalty"],
                "initial_step_xyz": config["initial_step_xyz"],
                "initial_step_opk": config["initial_step_opk"],
            }
            objective = CalibrationObjective(
                objective_config, imu, references, lower, upper
            )
            initial_normalized = (baseline - lower) / (upper - lower)
            result = minimize(
                objective,
                initial_normalized,
                method="Nelder-Mead",
                bounds=[(0.0, 1.0)] * 6,
                options={
                    "initial_simplex": _initial_simplex(
                        initial_normalized, lower, upper, objective_config
                    ),
                    "maxiter": max_iterations,
                    "xatol": config["parameter_tolerance"],
                    "fatol": config["cost_tolerance"],
                    "adaptive": True,
                    "disp": True,
                },
            )
            local = objective.physical_parameters(result.x)
            extra = local - baseline
            reference_time = _frame_time(raw_files[first_id])
            image_time = _frame_time(raw_files[second_id])
            row = {
                "window": window_index,
                "reference_frame_id": first_id,
                "image_frame_id": second_id,
                "mid_frame_id": second_id,
                "reference_time": (
                    reference_time.isoformat(sep=" ") if reference_time else None
                ),
                "time": image_time.isoformat(sep=" ") if image_time else None,
                "cost": float(result.fun),
                "success": bool(result.success),
                "message": str(result.message),
                "evaluations": int(result.nfev),
            }
            row.update(image_shift)
            row.update(optical_flow_shift)
            for index, name in enumerate(PARAMETER_NAMES):
                row[f"baseline_{name}"] = float(baseline[index])
                row[f"total_{name}"] = float(local[index])
                row[f"extra_{name}"] = float(extra[index])
            rows.append(row)
            checkpoint = _write_checkpoint(rows, output_dir)
            print(f"Additional correction: {extra.tolist()}")
            print(
                "Current RunOrtho shift to apply: "
                f"dx={row['runortho_shift_apply_dx_px']:.3f}px, "
                f"dy={row['runortho_shift_apply_dy_px']:.3f}px, "
                f"norm={row['runortho_shift_apply_norm_px']:.3f}px"
            )
            print(
                "Optical-flow shift to apply: "
                f"dx={row['runortho_flow_shift_apply_dx_px']:.3f}px, "
                f"dy={row['runortho_flow_shift_apply_dy_px']:.3f}px, "
                f"norm={row['runortho_flow_shift_apply_norm_px']:.3f}px "
                f"({row['runortho_flow_retained_count']}/"
                f"{row['runortho_flow_feature_count']} tracks)"
            )
            print(f"Checkpoint: {checkpoint}")

        table = pd.DataFrame(rows)
        plot_path = _plot_timeseries(table, output_dir)
        json_path = output_dir / "imu_drift_timeseries.json"
        json_path.write_text(
            json.dumps(rows, indent=2, default=str) + "\n", encoding="utf-8"
        )
        print(f"Time-series plot: {plot_path}")
        print(f"Time-series JSON: {json_path}")
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
