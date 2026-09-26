#!/usr/bin/env python3
"""Estimate consecutive-image optical-flow translation and compare it with IMU attitude changes."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import xarray as xr


def image_time(path: Path) -> datetime:
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        tag = page.tags.get("ImageDescription")
        if tag is not None:
            try:
                value = json.loads(str(tag.value)).get("Time")
            except json.JSONDecodeError:
                value = None
            if value:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        result = datetime.strptime(str(value), fmt)
                        return result.replace(tzinfo=timezone.utc)
                    except ValueError:
                        pass
        tag = page.tags.get("DateTime")
        if tag is not None:
            try:
                result = datetime.strptime(str(tag.value).strip(), "%y%m%d %H%M%S%f")
                return result.replace(tzinfo=timezone.utc)
            except ValueError:
                pass
    raise ValueError(f"No usable acquisition time metadata in {path}")


def gray_image(path: Path, max_size: int = 1600) -> tuple[np.ndarray, float]:
    data = tifffile.imread(path)
    if data.ndim == 3:
        if data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4):
            data = np.moveaxis(data, 0, -1)
        if data.shape[-1] >= 3:
            data = (0.299 * data[..., 0] + 0.587 * data[..., 1] + 0.114 * data[..., 2])
        else:
            data = data[..., 0]
    data = np.asarray(data, dtype=np.float32)
    scale = min(1.0, max_size / max(data.shape))
    if scale < 1.0:
        data = cv2.resize(data, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError(f"Image has no finite pixels: {path}")
    fill = float(np.nanmedian(data[finite]))
    data = np.nan_to_num(data, nan=fill, posinf=fill, neginf=fill)
    data -= np.mean(data)
    std = np.std(data)
    if std > 0:
        data /= std
    return data, scale




def optical_flow_translation(first: np.ndarray, second: np.ndarray):
    """Track many points and return all vectors plus their arithmetic mean.

    Confidence is based on the standard deviation of all valid tracked
    vectors. RANSAC is retained only to mark diagnostic inliers/outliers.
    """
    first_u8 = cv2.normalize(first, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    second_u8 = cv2.normalize(second, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    points0 = cv2.goodFeaturesToTrack(
        first_u8, maxCorners=2000, qualityLevel=0.005, minDistance=25, blockSize=7
    )
    empty = (np.nan, np.nan, 0.0, 0, np.nan, np.nan, 0.0,
             np.empty((0, 2)), np.empty((0, 2)), np.empty(0, dtype=bool))
    if points0 is None or len(points0) < 4:
        return empty
    points1, status, _ = cv2.calcOpticalFlowPyrLK(
        first_u8, second_u8, points0, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if points1 is None:
        return empty
    back, back_status, _ = cv2.calcOpticalFlowPyrLK(
        second_u8, first_u8, points1, None,
        winSize=(31, 31), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    p0_all = points0.reshape(-1, 2)
    p1_all = points1.reshape(-1, 2)
    pb = back.reshape(-1, 2) if back is not None else np.full_like(p0_all, np.nan)
    valid = (status.reshape(-1).astype(bool) & back_status.reshape(-1).astype(bool)
             & np.isfinite(p1_all).all(axis=1) & (np.linalg.norm(pb - p0_all, axis=1) < 1.5))
    p0 = p0_all[valid]
    flow = p1_all[valid] - p0
    if len(flow) < 4:
        return (np.nan, np.nan, 0.0, len(flow), np.nan, np.nan, 0.0, p0, flow,
                np.zeros(len(flow), dtype=bool))
    affine, inlier_mask = cv2.estimateAffinePartial2D(
        p0, p0 + flow, method=cv2.RANSAC,
        ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99,
    )
    inliers = (inlier_mask.reshape(-1).astype(bool)
               if affine is not None and inlier_mask is not None
               else np.ones(len(flow), dtype=bool))
    inlier_flow = flow[inliers]
    if len(inlier_flow) == 0:
        return (np.nan, np.nan, 0.0, 0, np.nan, np.nan, 0.0, p0, flow, inliers)
    # The center arrow must represent all displayed valid vectors, including
    # vectors that RANSAC marks as outliers.
    vector = np.mean(flow, axis=0)
    spread = np.std(flow, axis=0)
    # Directional coherence is one for parallel vectors and approaches zero
    # when vectors point in many directions. Ignore subpixel vectors when
    # estimating direction because their angle is dominated by noise.
    magnitudes = np.linalg.norm(flow, axis=1)
    directional = magnitudes > 0.5
    if directional.sum() < 4:
        direction_coherence = 0.0
    else:
        unit_vectors = flow[directional] / magnitudes[directional, None]
        direction_coherence = float(np.linalg.norm(np.mean(unit_vectors, axis=0)))
    # Direction is the confidence criterion; displacement spread remains a
    # separate diagnostic and does not dominate the score.
    confidence = direction_coherence
    # Penalize spatially incomplete tracking: all four image quadrants must
    # contain at least one valid vector, otherwise confidence is reduced 10x.
    height, width = first.shape[:2]
    cx, cy = width / 2.0, height / 2.0
    quadrant_counts = [
        np.sum((p0[:, 0] < cx) & (p0[:, 1] < cy)),
        np.sum((p0[:, 0] >= cx) & (p0[:, 1] < cy)),
        np.sum((p0[:, 0] < cx) & (p0[:, 1] >= cy)),
        np.sum((p0[:, 0] >= cx) & (p0[:, 1] >= cy)),
    ]
    if any(count == 0 for count in quadrant_counts):
        confidence *= 0.1
    return (float(vector[0]), float(vector[1]), confidence, int(len(inlier_flow)),
            float(spread[0]), float(spread[1]), direction_coherence, p0, flow, inliers)


def save_vector_image(path: Path, output_dir: Path, dx: float, dy: float,
                      confidence: float, max_size: int, points: np.ndarray,
                      flows: np.ndarray, inliers: np.ndarray) -> None:
    """Save all tracked vectors and the mean vector from the image center."""
    image, scale = gray_image(path, max_size)
    display = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    for point, flow, is_inlier in zip(points, flows, inliers):
        start = tuple(np.rint(point).astype(int))
        end = tuple(np.rint(point + flow).astype(int))
        color = (0, 255, 0) if is_inlier else (0, 165, 255)
        cv2.arrowedLine(display, start, end, color, 1, cv2.LINE_AA, tipLength=0.2)
        cv2.circle(display, start, 2, color, -1, cv2.LINE_AA)
    height, width = display.shape[:2]
    center = (width // 2, height // 2)
    if np.isfinite(dx) and np.isfinite(dy):
        end = (int(round(center[0] + dx * scale)), int(round(center[1] + dy * scale)))
        cv2.arrowedLine(display, center, end, (0, 0, 255), 4, cv2.LINE_AA, tipLength=0.15)
    label = f"mean dx={dx:+.2f} px  dy={dy:+.2f} px  confidence={confidence:.3f}"
    cv2.putText(display, label, (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 0, 255), 2, cv2.LINE_AA)
    output = output_dir / f"vector_{path.stem}.png"
    cv2.imwrite(str(output), display)


def wrap_delta(a: float, b: float) -> float:
    return float((b - a + 180.0) % 360.0 - 180.0)


def nearest_attitudes(times: list[datetime], imu_path: Path) -> list[dict]:
    with xr.open_dataset(imu_path) as ds:
        if "time" not in ds:
            raise ValueError("IMU has no time coordinate")
        imu_times = ds.time.values.astype("datetime64[ns]")
        result = []
        for timestamp in times:
            target = np.datetime64(timestamp.astimezone(timezone.utc).replace(tzinfo=None), "ns")
            right = int(np.searchsorted(imu_times, target, side="left"))
            candidates = [i for i in (right - 1, right) if 0 <= i < len(imu_times)]
            if not candidates:
                result.append({"roll": np.nan, "pitch": np.nan, "heading": np.nan})
                continue
            i = min(candidates, key=lambda j: abs(imu_times[j] - target))
            def value(name):
                return float(ds[name].values[i]) if name in ds else np.nan
            result.append({"roll": value("ROLL"), "pitch": value("PITCH"), "heading": value("THEAD")})
    return result



def estimate_lag(start_times, end_times, dx_values, weights, imu_path, max_lag_seconds=10.0):
    """Search nearest 200 Hz IMU roll samples at 0.005 s increments.

    The correlation objective is weighted by optical-flow confidence.
    """
    dx = np.asarray(dx_values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if len(dx) < 5: return np.nan, np.nan
    with xr.open_dataset(imu_path) as ds:
        if "time" not in ds or "ROLL" not in ds: return np.nan, np.nan
        it = ds.time.values.astype("datetime64[ns]").astype("int64").astype(float) / 1e9
        roll = np.asarray(ds["ROLL"].values, dtype=float)
    ok = np.isfinite(roll); it, roll = it[ok], roll[ok]
    if len(roll) < 2: return np.nan, np.nan
    t1 = np.array([t.timestamp() for t in start_times]); t2 = np.array([t.timestamp() for t in end_times])
    def nearest(q):
        right = np.clip(np.searchsorted(it, q, side="left"), 0, len(it)-1)
        left = np.clip(right-1, 0, len(it)-1)
        return roll[np.where(np.abs(it[left]-q) <= np.abs(it[right]-q), left, right)]
    best_lag, best_r = np.nan, np.nan
    for lag in np.arange(-max_lag_seconds, max_lag_seconds + 0.0025, 0.005):
        inside = (t1 + lag >= it[0]) & (t2 + lag <= it[-1])
        if inside.sum() < 5: continue
        dr = nearest(t2[inside] + lag) - nearest(t1[inside] + lag)
        x = dx[inside]; w = np.clip(weights[inside], 0.0, 1.0)
        use = np.isfinite(x) & np.isfinite(dr) & np.isfinite(w) & (w > 0)
        if use.sum() < 5: continue
        x, dr, w = x[use], dr[use], w[use]
        wsum = np.sum(w)
        if wsum <= 0: continue
        x_mean, dr_mean = np.sum(w * x) / wsum, np.sum(w * dr) / wsum
        covariance = np.sum(w * (x - x_mean) * (dr - dr_mean))
        variance_x = np.sum(w * (x - x_mean) ** 2)
        variance_dr = np.sum(w * (dr - dr_mean) ** 2)
        if variance_x == 0 or variance_dr == 0: continue
        r = float(covariance / np.sqrt(variance_x * variance_dr))
        if not np.isfinite(best_r) or abs(r) > abs(best_r): best_lag, best_r = float(lag), r
    return best_lag, best_r

def shifted_roll_delta(start_times, end_times, imu_path, lag):
    with xr.open_dataset(imu_path) as ds:
        it = ds.time.values.astype("datetime64[ns]").astype("int64").astype(float) / 1e9
        roll = np.asarray(ds["ROLL"].values, dtype=float)
    ok = np.isfinite(roll); it, roll = it[ok], roll[ok]
    def sample(times):
        q = np.array([t.timestamp() + lag for t in times])
        right = np.clip(np.searchsorted(it, q, side="left"), 0, len(it)-1)
        left = np.clip(right - 1, 0, len(it)-1)
        return roll[np.where(np.abs(it[left]-q) <= np.abs(it[right]-q), left, right)]
    return sample(end_times) - sample(start_times)

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--imu", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument("--max-size", type=int, default=1600)
    args = parser.parse_args()
    output_dir = args.output_dir or args.images_dir / "image_sequence_motion"
    output_dir.mkdir(parents=True, exist_ok=True)

    files = []
    for path in sorted(args.images_dir.glob(args.glob)):
        try:
            files.append((path, image_time(path)))
        except ValueError as exc:
            print(f"warning: {exc}; skipping")
    files.sort(key=lambda item: item[1])
    if len(files) < 2:
        raise ValueError("Need at least two timestamped images")
    times = [t for _, t in files]
    attitudes = nearest_attitudes(times, args.imu)

    rows = []
    for index, ((path1, time1), (path2, time2)) in enumerate(zip(files, files[1:]), 1):
        first, scale1 = gray_image(path1, args.max_size)
        second, scale2 = gray_image(path2, args.max_size)
        if first.shape != second.shape:
            second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
            scale2 = scale1
        (dx_scaled, dy_scaled, confidence, flow_points, flow_std_x, flow_std_y,
         flow_direction_coherence, tracked_points, tracked_flows, flow_inliers) = optical_flow_translation(first, second)
        # Optical-flow vectors are measured on the resized images. Convert
        # them back to the original image pixel units.
        dx, dy = float(dx_scaled / scale1), float(dy_scaled / scale1)
        dt = (time2 - time1).total_seconds()
        a1, a2 = attitudes[index - 1], attitudes[index]
        save_vector_image(
            path1, output_dir, dx, dy, float(confidence), args.max_size,
            tracked_points, tracked_flows, flow_inliers,
        )
        rows.append({
            "index": index, "time_1": time1.isoformat(), "time_2": time2.isoformat(),
            "dt_seconds": dt, "image_1": path1.name, "image_2": path2.name,
            "dx_pixels": dx, "dy_pixels": dy,
            "translation_pixels": float(np.hypot(dx, dy)),
            "flow_confidence": float(confidence), "flow_inlier_points": flow_points,
            "flow_std_dx_pixels": flow_std_x / scale1, "flow_std_dy_pixels": flow_std_y / scale1,
            "flow_direction_coherence": flow_direction_coherence,
            "roll_1_deg": a1["roll"], "roll_2_deg": a2["roll"],
            "pitch_1_deg": a1["pitch"], "pitch_2_deg": a2["pitch"],
            "heading_1_deg": a1["heading"], "heading_2_deg": a2["heading"],
            "delta_roll_deg": a2["roll"] - a1["roll"],
            "delta_pitch_deg": a2["pitch"] - a1["pitch"],
            "delta_heading_deg": wrap_delta(a1["heading"], a2["heading"]),
        })
        print(f"{index:04d} {path1.name} -> {path2.name}: dx={dx:.2f} dy={dy:.2f} confidence={confidence:.3f}")

    fields = list(rows[0])
    csv_path = output_dir / "image_sequence_motion.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    mid = [datetime.fromisoformat(r["time_1"]) + (datetime.fromisoformat(r["time_2"]) - datetime.fromisoformat(r["time_1"])) / 2 for r in rows]
    dt = np.array([r["dt_seconds"] for r in rows])
    pair_start_times = [datetime.fromisoformat(r["time_1"]) for r in rows]
    pair_end_times = [datetime.fromisoformat(r["time_2"]) for r in rows]
    lag_seconds, lag_correlation = estimate_lag(
        pair_start_times, pair_end_times, [r["dx_pixels"] for r in rows],
        [r["flow_confidence"] for r in rows], args.imu
    )
    lag_file = output_dir / "best_time_lag.txt"
    if np.isfinite(lag_seconds):
        lag_file.write_text(
            f"time_lag_seconds={lag_seconds:.3f}\n"
            f"weighted_correlation={lag_correlation:.6f}\n"
            "positive_lag_means_roll_occurs_later=True\n",
            encoding="utf-8",
        )
        print(
            f"Best dx/delta-roll lag: {lag_seconds:+.3f} s "
            f"(positive means roll occurs later; r={lag_correlation:+.3f})"
        )
    else:
        lag_file.write_text("time_lag_seconds=nan\nweighted_correlation=nan\n", encoding="utf-8")
    print(f"Wrote {lag_file}")
    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)
    axes[0].plot(mid, [r["dx_pixels"] for r in rows], label="ΔX pixels")
    axes[0].plot(mid, [r["dy_pixels"] for r in rows], label="ΔY pixels")
    axes[0].plot(mid, [r["translation_pixels"] for r in rows], "k--", linewidth=0.8, label="magnitude")
    axes[0].set_ylabel("Image displacement (pixels)")
    axes[0].legend(loc="best")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(mid, [r["flow_confidence"] for r in rows], color="tab:red", label="Optical-flow directional confidence")
    axes[1].axhline(0.5, color="0.4", linestyle="--", linewidth=0.8, label="confidence = 0.5")
    axes[1].set_ylabel("Confidence")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].legend(loc="best")
    axes[1].grid(True, alpha=0.3)
    axes[2].plot(mid, [r["delta_roll_deg"] for r in rows], label="Δ roll")
    axes[2].plot(mid, [r["delta_pitch_deg"] for r in rows], label="Δ pitch")
    axes[2].plot(mid, [r["delta_heading_deg"] for r in rows], label="Δ heading")
    axes[2].set_ylabel("Attitude change (deg)")
    axes[2].legend(loc="best")
    axes[2].grid(True, alpha=0.3)

    # Direct comparison of horizontal image motion and roll change.
    dx_axis = axes[3]
    roll_axis = dx_axis.twinx()
    dx_line, = dx_axis.plot(mid, [r["dx_pixels"] for r in rows],
                            color="tab:blue", label="Optical-flow ΔX")
    if np.isfinite(lag_seconds):
        aligned_roll = shifted_roll_delta(pair_start_times, pair_end_times, args.imu, lag_seconds)
        roll_label = "Δ roll (reassigned at best lag)"
    else:
        aligned_roll = np.asarray([r["delta_roll_deg"] for r in rows], dtype=float)
        roll_label = "Δ roll"
    roll_line, = roll_axis.plot(mid, aligned_roll,
                                color="tab:orange", label=roll_label)
    dx_axis.axhline(0.0, color="tab:blue", linewidth=0.6, alpha=0.5)
    roll_axis.axhline(0.0, color="tab:orange", linewidth=0.6, alpha=0.5)
    dx_axis.set_ylabel("Optical-flow ΔX (pixels)", color="tab:blue")
    roll_axis.set_ylabel("Δ roll (deg)", color="tab:orange")
    dx_axis.tick_params(axis="y", labelcolor="tab:blue")
    roll_axis.tick_params(axis="y", labelcolor="tab:orange")
    dx_axis.grid(True, alpha=0.3)
    dx_axis.legend([dx_line, roll_line], ["Optical-flow ΔX", "Δ roll"], loc="best")
    dx_axis.set_xlabel("Pair midpoint time (UTC)")
    lag_text = (f", best dx/roll lag={lag_seconds:+.3f} s, r={lag_correlation:+.3f}"
                if np.isfinite(lag_seconds) else "")
    fig.suptitle(f"Consecutive image motion and AIRINS attitude changes (mean dt={np.mean(dt):.3f} s{lag_text})")
    fig.autofmt_xdate()
    fig.tight_layout()
    png_path = output_dir / "image_sequence_motion.png"
    fig.savefig(png_path, dpi=180)
    plt.close(fig)
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
