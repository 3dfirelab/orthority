#!/usr/bin/env python3
"""Compute consecutive GeoTIFF ortho-image correlation as a time series.

The comparison is made in the geographic intersection of each consecutive
pair.  The second ortho is resampled onto the first ortho's pixel grid before
calculating Pearson correlation, so different output footprints and raster
sizes are handled correctly.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio
import xarray as xr
from rasterio.enums import Resampling
from rasterio.transform import array_bounds
from rasterio.warp import reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from shapely.geometry import Point


def parse_time(ds: rasterio.DatasetReader, path: Path) -> datetime:
    value = ds.tags().get("Time")
    if not value:
        raise ValueError(f"{path} has no GeoTIFF Time tag")
    value = value.replace("Z", "+00:00")
    result = datetime.fromisoformat(value)
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result


def intersection_window(ds1, ds2) -> Window | None:
    # Transform the second footprint into the first CRS before intersecting.
    b1 = array_bounds(ds1.height, ds1.width, ds1.transform)
    b2 = transform_bounds(ds2.crs, ds1.crs, *ds2.bounds, densify_pts=21)
    left = max(b1[0], b2[0])
    bottom = max(b1[1], b2[1])
    right = min(b1[2], b2[2])
    top = min(b1[3], b2[3])
    if right <= left or top <= bottom:
        return None
    window = from_bounds(left, bottom, right, top, ds1.transform)
    # Round outward so a narrow valid intersection is not lost to rounding.
    row0 = max(0, int(np.floor(window.row_off)))
    col0 = max(0, int(np.floor(window.col_off)))
    row1 = min(ds1.height, int(np.ceil(window.row_off + window.height)))
    col1 = min(ds1.width, int(np.ceil(window.col_off + window.width)))
    if row1 <= row0 or col1 <= col0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def read_pair(ds1, ds2, window: Window, band: int) -> tuple[np.ndarray, np.ndarray, int]:
    a = ds1.read(band, window=window, masked=True).astype(np.float32)
    destination = np.full((int(window.height), int(window.width)), np.nan, dtype=np.float32)
    destination_transform = ds1.window_transform(window)
    reproject(
        source=rasterio.band(ds2, band),
        destination=destination,
        src_transform=ds2.transform,
        src_crs=ds2.crs,
        src_nodata=ds2.nodata,
        dst_transform=destination_transform,
        dst_crs=ds1.crs,
        dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    a_data = np.asarray(a.filled(np.nan), dtype=np.float32)
    valid = np.isfinite(a_data) & np.isfinite(destination)
    return a_data, destination, int(valid.sum())


def correlation(a: np.ndarray, b: np.ndarray) -> tuple[float, float, int]:
    valid_a = np.isfinite(a)
    valid_b = np.isfinite(b)
    valid = valid_a & valid_b
    n = int(valid.sum())
    # Ignore empty/nodata edges: overlap is valid-footprint IoU, consistent
    # with the calibration metric used for same-image reference pairs.
    union = int((valid_a | valid_b).sum())
    overlap = n / float(union) if union else float("nan")
    if n < 2:
        return float("nan"), overlap, n
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    sx = x.std()
    sy = y.std()
    if sx == 0.0 or sy == 0.0:
        return float("nan"), overlap, n
    r = float(np.corrcoef(x, y)[0, 1])
    return r, overlap, n


def nearest_imu_attitude(files: list[tuple[Path, datetime]], imu_path: Path) -> list[dict]:
    """Return the nearest IMU attitude sample for each image timestamp."""
    with xr.open_dataset(imu_path) as imu:
        if "time" not in imu:
            raise ValueError("IMU must contain a time variable")
        imu_times = imu["time"].values.astype("datetime64[ns]")
        result = []
        for _, image_time in files:
            target = np.datetime64(image_time.astimezone(timezone.utc).replace(tzinfo=None), "ns")
            right = int(np.searchsorted(imu_times, target, side="left"))
            candidates = [idx for idx in (right - 1, right) if 0 <= idx < len(imu_times)]
            if not candidates:
                result.append({
                    "roll": np.nan, "pitch": np.nan, "heading": np.nan,
                    "latitude": np.nan, "longitude": np.nan, "height": np.nan,
                })
                continue
            idx = min(candidates, key=lambda item: abs(imu_times[item] - target))
            def value(name):
                return float(imu[name].values[idx]) if name in imu else np.nan
            result.append({
                "roll": value("ROLL"), "pitch": value("PITCH"), "heading": value("THEAD"),
                "latitude": value("LATITUDE"), "longitude": value("LONGITUDE"),
                "height": value("HEIGHT_WGS84"),
            })
    return result


def local_position_and_speed(attitudes: list[dict], files: list[tuple[Path, datetime]]):
    """Return local east/north positions and pairwise ground speeds.

    Positions are relative to the first image, in metres. Speeds use the
    horizontal distance between consecutive nearest-IMU positions divided by
    the measured image time interval.
    """
    lat0 = next((a["latitude"] for a in attitudes if np.isfinite(a["latitude"])), np.nan)
    lon0 = next((a["longitude"] for a in attitudes if np.isfinite(a["longitude"])), np.nan)
    earth_lat_m = 110540.0
    earth_lon_m = 111320.0 * np.cos(np.deg2rad(lat0)) if np.isfinite(lat0) else np.nan
    positions = []
    for attitude in attitudes:
        lat, lon = attitude["latitude"], attitude["longitude"]
        if np.isfinite(lat) and np.isfinite(lon) and np.isfinite(lat0) and np.isfinite(lon0):
            x = (lon - lon0) * earth_lon_m
            y = (lat - lat0) * earth_lat_m
        else:
            x = y = np.nan
        positions.append((float(x), float(y)))
    speeds = []
    for (x1, y1), (x2, y2), ((_, t1), (_, t2)) in zip(positions, positions[1:], zip(files, files[1:])):
        dt = (t2 - t1).total_seconds()
        speeds.append(float(np.hypot(x2 - x1, y2 - y1) / dt) if dt > 0 and np.isfinite(x1 + y1 + x2 + y2) else np.nan)
    return positions, speeds


def wrapped_angle_difference(first: float, second: float) -> float:
    """Return second-first in degrees, wrapped to [-180, 180)."""
    return float((second - first + 180.0) % 360.0 - 180.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input_dir", nargs="?", type=Path,
        default=Path("/data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Output directory (default: INPUT_DIR/correlation_timeseries)",
    )
    parser.add_argument("--band", type=int, default=1, help="1-based band to compare (default: 1)")
    parser.add_argument("--glob", default="*.tif")
    parser.add_argument(
        "--imu", type=Path, required=True,
        help="One IMU NetCDF used for this correlation run and image-frequency subset",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.input_dir / "correlation_timeseries"
    imu_label = next((label for label in ("AIRINS", "ATLANS") if label in args.imu.name.upper()), args.imu.stem)
    imu_label = "".join(char if char.isalnum() or char == "_" else "_" for char in imu_label)
    output = output_dir / f"ortho_correlation_timeseries_{imu_label}.csv"

    files: list[tuple[Path, datetime]] = []
    for path in sorted(args.input_dir.glob(args.glob)):
        with rasterio.open(path) as ds:
            try:
                acquired = parse_time(ds, path)
            except ValueError as exc:
                print(f"warning: {exc}; skipping file")
                continue
            files.append((path, acquired))
            if not 1 <= args.band <= ds.count:
                raise ValueError(f"Band {args.band} is not available in {path} (count={ds.count})")
    files.sort(key=lambda item: item[1])
    if len(files) < 2:
        raise ValueError(f"Need at least two timestamped TIFFs in {args.input_dir}")

    image_attitudes = nearest_imu_attitude(files, args.imu)
    image_positions, pair_speeds = local_position_and_speed(image_attitudes, files)
    output_dir.mkdir(parents=True, exist_ok=True)
    fields = [
        "index", "time_1", "time_2", "dt_seconds", "image_1", "image_2",
        "pearson_r", "overlap_fraction", "valid_pixels",
        "roll_1_deg", "pitch_1_deg", "heading_1_deg",
        "roll_2_deg", "pitch_2_deg", "heading_2_deg",
        "delta_roll_deg", "delta_pitch_deg", "delta_heading_deg",
        "x_position_m_1", "y_position_m_1", "x_position_m_2", "y_position_m_2",
        "speed_m_s",
    ]
    records = []
    image_points = []
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, ((path1, time1), (path2, time2)) in enumerate(zip(files, files[1:]), 1):
            with rasterio.open(path1) as ds1, rasterio.open(path2) as ds2:
                if ds1.crs is None or ds2.crs is None:
                    raise ValueError(f"Both images need a CRS: {path1.name}, {path2.name}")
                window = intersection_window(ds1, ds2)
                if window is None:
                    r = overlap = float("nan")
                    n = 0
                else:
                    a, b, _ = read_pair(ds1, ds2, window, args.band)
                    r, overlap, n = correlation(a, b)
                # The centre of the georeferenced ortho footprint is used as
                # the aircraft/nadir point for this image.
                x1, y1 = ds1.xy((ds1.height - 1) / 2, (ds1.width - 1) / 2)
            attitude1 = image_attitudes[index - 1]
            attitude2 = image_attitudes[index]
            image_points.append({
                    "image_index": index, "image_name": path1.name, "imu_name": imu_label,
                    "time": time1.isoformat(), "dt_seconds": (time2 - time1).total_seconds(),
                    "pearson_r": r, "correlation_with": path2.name, "correlation_role": "next",
                    "roll_deg": attitude1["roll"], "pitch_deg": attitude1["pitch"],
                    "heading_deg": attitude1["heading"], "delta_roll_deg": attitude2["roll"] - attitude1["roll"],
                    "delta_pitch_deg": attitude2["pitch"] - attitude1["pitch"],
                    "delta_heading_deg": wrapped_angle_difference(attitude1["heading"], attitude2["heading"]),
                    "x_position_m": image_positions[index - 1][0], "y_position_m": image_positions[index - 1][1],
                    "speed_m_s": pair_speeds[index - 1],
                    "geometry": Point(x1, y1), "crs": ds1.crs,
                })
            row = {
                "index": index, "time_1": time1.isoformat(), "time_2": time2.isoformat(),
                "dt_seconds": (time2 - time1).total_seconds(), "image_1": path1.name,
                "image_2": path2.name, "pearson_r": r, "overlap_fraction": overlap,
                "valid_pixels": n,
                "roll_1_deg": attitude1["roll"], "pitch_1_deg": attitude1["pitch"],
                "heading_1_deg": attitude1["heading"], "roll_2_deg": attitude2["roll"],
                "pitch_2_deg": attitude2["pitch"], "heading_2_deg": attitude2["heading"],
                "delta_roll_deg": attitude2["roll"] - attitude1["roll"],
                "delta_pitch_deg": attitude2["pitch"] - attitude1["pitch"],
                "delta_heading_deg": wrapped_angle_difference(attitude1["heading"], attitude2["heading"]),
                "x_position_m_1": image_positions[index - 1][0], "y_position_m_1": image_positions[index - 1][1],
                "x_position_m_2": image_positions[index][0], "y_position_m_2": image_positions[index][1],
                "speed_m_s": pair_speeds[index - 1],
            }
            writer.writerow(row)
            records.append(row)
            print(f"{index:04d} {path1.name} -> {path2.name}: r={r:.4f} overlap={overlap:.3f} n={n}")

            if index == len(files) - 1:
                with rasterio.open(path2) as final_ds:
                    x2, y2 = final_ds.xy((final_ds.height - 1) / 2, (final_ds.width - 1) / 2)
                    image_points.append({
                        "image_index": index + 1, "image_name": path2.name, "imu_name": imu_label,
                        "time": time2.isoformat(), "dt_seconds": (time2 - time1).total_seconds(),
                        "pearson_r": r, "correlation_with": path1.name, "correlation_role": "previous",
                        "roll_deg": attitude2["roll"], "pitch_deg": attitude2["pitch"],
                        "heading_deg": attitude2["heading"], "delta_roll_deg": attitude2["roll"] - attitude1["roll"],
                        "delta_pitch_deg": attitude2["pitch"] - attitude1["pitch"],
                        "delta_heading_deg": wrapped_angle_difference(attitude1["heading"], attitude2["heading"]),
                        "x_position_m": image_positions[index][0], "y_position_m": image_positions[index][1],
                        "speed_m_s": pair_speeds[index - 1],
                        "geometry": Point(x2, y2), "crs": final_ds.crs,
                    })

    if image_points:
        crs_values = {str(record["crs"]) for record in image_points}
        if len(crs_values) != 1:
            raise ValueError("Input orthos use multiple CRSs; cannot write one GeoPackage layer")
        frame = gpd.GeoDataFrame(
            [{key: value for key, value in record.items() if key not in ("geometry", "crs")}
             for record in image_points],
            geometry=[record["geometry"] for record in image_points],
            crs=image_points[0]["crs"],
        )
        gpkg = output_dir / f"ortho_correlation_{imu_label}.gpkg"
        frame.to_file(gpkg, layer=f"correlation_points_{imu_label.lower()}", driver="GPKG", index=False)
        print(f"Wrote {gpkg}")

    values = np.asarray([record["pearson_r"] for record in records], dtype=float)
    mid_times = []
    for record in records:
        t1 = datetime.fromisoformat(record["time_1"])
        t2 = datetime.fromisoformat(record["time_2"])
        mid_times.append(t1 + (t2 - t1) / 2)
    intervals = np.asarray([record["dt_seconds"] for record in records], dtype=float)
    finite_values = values[np.isfinite(values)]
    mean_correlation = float(np.mean(finite_values)) if finite_values.size else float("nan")
    if finite_values.size:
        value_min = float(finite_values.min())
        value_max = float(finite_values.max())
        margin = max(0.02, 0.08 * (value_max - value_min))
        correlation_limits = (max(-1.0, value_min - margin), min(1.0, value_max + margin))
    else:
        correlation_limits = (-1.0, 1.0)
    figure, axis = plt.subplots(figsize=(13, 5))
    correlation_line, = axis.plot(
        mid_times, values, ".-", linewidth=0.8, markersize=3,
        color="tab:blue", label="Pearson correlation",
    )
    axis.axhline(0.6, color="tab:red", linestyle="--", linewidth=1, label="r = 0.6")
    mean_line = axis.axhline(
        mean_correlation, color="tab:blue", linestyle=":", linewidth=1,
        label=f"mean r = {mean_correlation:.3f}",
    ) if np.isfinite(mean_correlation) else None
    axis.set_ylim(*correlation_limits)
    axis.set_xlim(min(mid_times), max(mid_times))
    axis.set_xlabel("Pair midpoint time (UTC)")
    axis.set_ylabel("Pixel-to-pixel Pearson correlation", color="tab:blue")
    axis.tick_params(axis="y", labelcolor="tab:blue")
    interval_axis = axis.twinx()
    interval_line, = interval_axis.plot(
        mid_times, intervals, "--", color="tab:orange", linewidth=1,
        label="Image interval (s)",
    )
    interval_axis.set_ylabel("Time between images (s)", color="tab:orange")
    interval_axis.tick_params(axis="y", labelcolor="tab:orange")
    axis.set_title(
        f"{imu_label}: consecutive ortho correlation and image interval "
        f"(band {args.band}, mean r={mean_correlation:.3f})"
    )
    axis.grid(True, alpha=0.3)
    legend_lines = [correlation_line, interval_line]
    legend_labels = ["Pearson correlation", "Image interval (s)"]
    if mean_line is not None:
        legend_lines.append(mean_line)
        legend_labels.append(f"Mean correlation ({mean_correlation:.3f})")
    axis.legend(legend_lines, legend_labels, loc="best")
    figure.autofmt_xdate()
    figure.tight_layout()
    plot = output_dir / f"ortho_correlation_timeseries_{imu_label}.png"
    figure.savefig(plot, dpi=180)
    plt.close(figure)
    print(f"Wrote {plot}")
    print(f"Mean Pearson correlation ({imu_label}): {mean_correlation:.6f}")

    # Diagnostic plot for correlation versus frame-to-frame attitude changes.
    delta_roll = np.asarray([record["delta_roll_deg"] for record in records], dtype=float)
    delta_pitch = np.asarray([record["delta_pitch_deg"] for record in records], dtype=float)
    delta_heading = np.asarray([record["delta_heading_deg"] for record in records], dtype=float)
    diagnostic, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    axes[0].plot(mid_times, values, ".-", color="tab:blue", markersize=3, linewidth=0.8)
    axes[0].axhline(0.6, color="tab:red", linestyle="--", linewidth=1)
    axes[0].axhline(mean_correlation, color="tab:blue", linestyle=":", linewidth=1)
    axes[0].set_ylabel("Correlation")
    axes[0].set_ylim(*correlation_limits)
    for axis, data, label, color in zip(
        axes[1:], (delta_roll, delta_pitch, delta_heading),
        ("Delta roll (deg)", "Delta pitch (deg)", "Delta heading (deg)"),
        ("tab:orange", "tab:green", "tab:purple"),
    ):
        axis.plot(mid_times, data, ".-", color=color, markersize=3, linewidth=0.8)
        axis.axhline(0.0, color="0.4", linewidth=0.6)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_title(f"{imu_label}: correlation and frame-to-frame attitude changes")
    axes[-1].set_xlabel("Pair midpoint time (UTC)")
    diagnostic.autofmt_xdate()
    diagnostic.tight_layout()
    diagnostic_plot = output_dir / f"ortho_correlation_attitude_{imu_label}.png"
    diagnostic.savefig(diagnostic_plot, dpi=180)
    plt.close(diagnostic)
    print(f"Wrote {diagnostic_plot}")

    # Export one nearest IMU sample per timestamped image.  This is an
    # image-frequency subset, not a resampling of the IMU trajectory.
    imu_output = output_dir / f"imu_image_frequency_{imu_label}.gpkg"
    with xr.open_dataset(args.imu) as imu:
        if "time" not in imu or "LATITUDE" not in imu or "LONGITUDE" not in imu:
            raise ValueError("IMU must contain time, LATITUDE, and LONGITUDE variables")
        imu_times = imu["time"].values.astype("datetime64[ns]")
        imu_records = []
        for image_index, (path, image_time) in enumerate(files, 1):
            target = np.datetime64(image_time.astimezone(timezone.utc).replace(tzinfo=None), "ns")
            right = int(np.searchsorted(imu_times, target, side="left"))
            candidates = [idx for idx in (right - 1, right) if 0 <= idx < len(imu_times)]
            if not candidates:
                continue
            imu_index = min(candidates, key=lambda idx: abs(imu_times[idx] - target))
            sample_time = imu_times[imu_index].astype("datetime64[us]").tolist()
            sample_dt = (sample_time - image_time.astimezone(timezone.utc).replace(tzinfo=None)).total_seconds()
            def value(name):
                return float(imu[name].values[imu_index]) if name in imu else None
            imu_records.append({
                "image_index": image_index, "image_name": path.name, "imu_name": imu_label,
                "image_time": image_time.isoformat(), "imu_time": sample_time.isoformat() + "+00:00",
                "time_offset_s": sample_dt, "latitude": value("LATITUDE"),
                "longitude": value("LONGITUDE"), "height_wgs84_m": value("HEIGHT_WGS84"),
                "roll_deg": value("ROLL"), "pitch_deg": value("PITCH"),
                "heading_deg": value("THEAD"), "course_deg": value("COURSE"),
                "vn_m_s": value("VN"), "ve_m_s": value("VE"), "vv_m_s": value("VV"),
                "geometry": Point(value("LONGITUDE"), value("LATITUDE")),
            })
    imu_frame = gpd.GeoDataFrame(imu_records, geometry="geometry", crs="EPSG:4326")
    imu_frame.to_file(imu_output, layer=f"imu_image_frequency_{imu_label.lower()}", driver="GPKG", index=False)
    print(f"Wrote {imu_output}")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
