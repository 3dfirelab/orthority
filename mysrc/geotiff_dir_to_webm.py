#!/usr/bin/env python3
"""Create a cumulative, full-track VP9 WebM from georeferenced TIFF frames.

Frames are ordered by their ``Time`` GeoTIFF metadata.  Each video frame shows
the complete track extent, with all orthos acquired up to that time merged onto
it.  The script requires rasterio and numpy from the ``tracking`` Mamba
environment, plus ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
import yaml
import xarray as xr
from affine import Affine
from rasterio.enums import Resampling
from rasterio.transform import array_bounds, from_origin
from rasterio.warp import reproject, transform, transform_bounds
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class Frame:
    path: Path
    acquired: str
    frame_id: int
    attitude: tuple[float, float, float] | None = None  # roll, pitch, yaw
    ecc_norm: float = float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tif_dir", type=Path, help="Directory containing GeoTIFF frames")
    parser.add_argument("config", type=Path, help="Dataset YAML containing the raw IMU path")
    parser.add_argument("--output", type=Path,
                        help="Output path (default: <tif_dir>/<directory-name>_track.webm)")
    parser.add_argument("--fps", type=float, default=10, help="Video frame rate (default: 10)")
    parser.add_argument("--width", type=int, default=960,
                        help="Full-track canvas width in pixels (default: 960)")
    parser.add_argument("--crf", type=int, default=32, help="VP9 quality; lower is better/larger (default: 32)")
    parser.add_argument("--percentiles", type=float, nargs=2, default=(2, 98), metavar=("LOW", "HIGH"),
                        help="Global display-stretch percentiles (default: 2 98)")
    parser.add_argument("--sample-stride", type=int, default=10,
                        help="Use every Nth TIFF to calculate the display stretch (default: 10)")
    parser.add_argument("--max-frames", type=int, help="Encode only the first N frames (preview/testing)")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output")
    return parser.parse_args()


def even(value: int) -> int:
    return value if value % 2 == 0 else value + 1


def discover_frames(tif_dir: Path) -> list[Frame]:
    paths = sorted([*tif_dir.glob("*.tif"), *tif_dir.glob("*.tiff")])
    if not paths:
        raise ValueError(f"No .tif or .tiff files found in {tif_dir}")
    frames = []
    for path in paths:
        with rasterio.open(path) as dataset:
            acquired = dataset.tags().get("Time")
        if acquired is None:
            raise ValueError(f"{path.name} has no Time metadata; cannot determine acquisition order")
        numbers = re.findall(r"\d+", path.stem)
        if not numbers:
            raise ValueError(f"Could not obtain a frame number from {path.name}")
        frames.append(Frame(path, acquired, int(numbers[-1])))
    return sorted(frames, key=lambda frame: frame.acquired)


def load_config_and_source(tif_dir: Path, config_path: Path) -> tuple[dict, str | None]:
    with config_path.open() as source:
        config = yaml.safe_load(source)
    raw_imu = config.get("imu")
    if not raw_imu:
        raise ValueError(f"No 'imu' entry in {config_path}")
    if not isinstance(raw_imu, dict):
        return config, None
    transect_name = tif_dir.parent.name
    sources = sorted(raw_imu, key=len, reverse=True)
    source = next((name for name in sources if transect_name.endswith(f"_{name}")), None)
    if source is None:
        raise ValueError(f"Could not infer an IMU source from '{transect_name}'; configured: {', '.join(sources)}")
    return config, source


def add_attitude(frames: list[Frame], config_path: Path, config: dict, imu_source: str | None) -> list[Frame]:
    """Interpolate roll, pitch, and yaw from the raw IMU selected by the YAML."""
    raw_imu = config["imu"]
    if isinstance(raw_imu, dict):
        assert imu_source is not None
        imu_path = Path(raw_imu[imu_source]).expanduser()
    else:
        imu_source = "default"
        imu_path = Path(raw_imu).expanduser()
    if not imu_path.is_absolute():
        imu_path = config_path.resolve().parent / imu_path
    if not imu_path.is_file():
        raise ValueError(f"Raw IMU file does not exist: {imu_path}")
    if imu_path.suffix.lower() == ".nc":
        imu = xr.open_dataset(imu_path)
        imu = imu.rename({
            "ROLL": "ROLL_smooth",
            "PITCH": "PITCH_smooth",
            "THEAD": "THEAD_smooth",
        })
        required = {"time", "ROLL_smooth", "PITCH_smooth", "THEAD_smooth"}
        missing = required - set(imu.variables)
        if missing:
            raise ValueError(f"NetCDF IMU is missing variables: {', '.join(sorted(missing))}")
        imu_times = pd.to_datetime(imu["time"].values, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        roll_values = np.asarray(imu["ROLL_smooth"].values, dtype=float)
        pitch_values = np.asarray(imu["PITCH_smooth"].values, dtype=float)
        yaw_values = np.asarray(imu["THEAD_smooth"].values, dtype=float)
    else:
        imu = gpd.read_file(imu_path)
        required = {"datetime_utc", "roll_deg", "pitch_deg", "heading_deg"}
        missing = required - set(imu.columns)
        if missing:
            raise ValueError(f"IMU file is missing columns: {', '.join(sorted(missing))}")
        # GeoPackage timestamps are millisecond precision while TIFF tags parse as
        # nanoseconds; explicitly normalise both before interpolation.
        imu_times = pd.to_datetime(imu["datetime_utc"], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
        roll_values = imu["roll_deg"].to_numpy(float)
        pitch_values = imu["pitch_deg"].to_numpy(float)
        yaw_values = imu["heading_deg"].to_numpy(float)
    frame_times = pd.to_datetime([frame.acquired for frame in frames], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    order = np.argsort(imu_times)
    imu_times = imu_times[order]
    if frame_times[0] < imu_times[0] or frame_times[-1] > imu_times[-1]:
        raise ValueError("GeoTIFF acquisition times lie outside the supplied IMU time range")
    roll = np.interp(frame_times, imu_times, roll_values[order])
    pitch = np.interp(frame_times, imu_times, pitch_values[order])
    # Unwrapping preserves continuity across a 0/360-degree heading crossing.
    yaw = np.rad2deg(np.interp(frame_times, imu_times, np.unwrap(np.deg2rad(yaw_values[order]))))
    print(f"Using raw IMU ({imu_source}): {imu_path}")
    return [Frame(frame.path, frame.acquired, frame.frame_id, (roll[index], pitch[index], yaw[index]), frame.ecc_norm)
            for index, frame in enumerate(frames)]


def add_ecc_norm(frames: list[Frame], config_path: Path, config: dict, imu_source: str | None) -> tuple[list[Frame], float | None]:
    raw_dir = str(config.get("drift_output_dir", ""))
    if not raw_dir:
        print("No drift_output_dir configured; continuing without ECC data.")
        return frames, None
    if "<imu>" in raw_dir:
        if imu_source is None:
            raise ValueError("drift_output_dir uses <imu>, but the IMU source could not be inferred")
        raw_dir = raw_dir.replace("<imu>", imu_source)
    drift_csv = Path(raw_dir).expanduser() / "imu_drift_timeseries.csv"
    if not drift_csv.is_absolute():
        drift_csv = config_path.resolve().parent / drift_csv
    if not drift_csv.is_file():
        print(f"ECC time series not found; continuing without ECC data: {drift_csv}")
        return frames, None
    table = pd.read_csv(drift_csv)
    required = {"reference_time", "time", "runortho_shift_apply_norm_px"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"ECC CSV is missing columns: {', '.join(sorted(missing))}")
    table["time"] = pd.to_datetime(table["time"], utc=True)
    table["reference_time"] = pd.to_datetime(table["reference_time"], utc=True)
    table = table[np.isfinite(table["runortho_shift_apply_norm_px"])].sort_values("time")
    ecc_times = table["time"].astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    ecc_values = table["runortho_shift_apply_norm_px"].to_numpy(float)
    frame_times = pd.to_datetime([frame.acquired for frame in frames], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    values = np.interp(frame_times, ecc_times, ecc_values, left=np.nan, right=np.nan)
    shift_seconds = float(np.median((table["time"] - table["reference_time"]).dt.total_seconds()))
    print(f"Using ECC norm: {drift_csv} (image-pair offset {shift_seconds:g}s)")
    updated = [Frame(frame.path, frame.acquired, frame.frame_id, frame.attitude, values[index])
               for index, frame in enumerate(frames)]
    return updated, shift_seconds


def attitude_chart(frames: list[Frame], width: int, ecc_shift_seconds: float | None, panel_height: int = 280) -> tuple[Image.Image, list[int]]:
    chart = Image.new("RGB", (width, panel_height), (18, 18, 22))
    draw = ImageDraw.Draw(chart)
    font = ImageFont.load_default()
    left, right, top, bottom = 72, width - 18, 14, panel_height - 18
    names = ["Roll", "Pitch", "Yaw"]
    colors = [(80, 210, 255), (90, 235, 120), (255, 105, 100)]
    attitude_values = np.asarray([frame.attitude for frame in frames], dtype=float)
    values = attitude_values
    if ecc_shift_seconds is not None:
        names.append(f"ECC norm (px; dt={ecc_shift_seconds:g}s)")
        colors.append((250, 205, 70))
        values = np.column_stack((attitude_values,
                                  np.asarray([frame.ecc_norm for frame in frames], dtype=float)))
    marker_x: list[int] = []
    row_height = (bottom - top) / len(names)
    for row, (name, color) in enumerate(zip(names, colors)):
        y0, y1 = int(top + row * row_height), int(top + (row + 1) * row_height - 4)
        series = values[:, row]
        finite = series[np.isfinite(series)]
        low, high = np.percentile(finite, (1, 99))
        padding = max((high - low) * 0.1, 0.1)
        low, high = low - padding, high + padding
        draw.rectangle((left, y0, right, y1), outline=(75, 75, 80))
        draw.text((5, y0 + 3), f"{name} ({low:.1f}..{high:.1f}°)", fill=color, font=font)
        points = []
        for index, value in enumerate(series):
            x = round(left + (right - left) * index / max(len(series) - 1, 1))
            if np.isfinite(value):
                y = round(y1 - (value - low) * (y1 - y0) / (high - low))
                points.append((x, y))
            if row == 0:
                marker_x.append(x)
        if points:
            draw.line(points, fill=color, width=1)
    return chart, marker_x


def track_grid(frames: list[Frame], width: int) -> tuple[Affine, int, int, rasterio.crs.CRS]:
    target_crs = rasterio.crs.CRS.from_epsg(3857)
    bounds = []
    for frame in frames:
        with rasterio.open(frame.path) as dataset:
            if dataset.crs is None:
                raise ValueError(f"The frame has no CRS: {frame.path}")
            bounds.append(transform_bounds(dataset.crs, target_crs, *dataset.bounds, densify_pts=21))
    left = min(bound[0] for bound in bounds)
    bottom = min(bound[1] for bound in bounds)
    right = max(bound[2] for bound in bounds)
    top = max(bound[3] for bound in bounds)
    resolution = (right - left) / width
    height = even(int(np.ceil((top - bottom) / resolution)))
    return from_origin(left, top, resolution, resolution), even(width), height, target_crs

def display_range(frames: list[Frame], percentiles: tuple[float, float], stride: int) -> tuple[float, float]:
    samples = []
    for frame in frames[::stride]:
        with rasterio.open(frame.path) as dataset:
            image = dataset.read(1, out_dtype="float32")[::4, ::4]
            mask = dataset.read_masks(1)[::4, ::4] != 0
        samples.append(image[mask & np.isfinite(image)])
    values = np.concatenate(samples)
    low, high = np.percentile(values, percentiles)
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        raise ValueError("Could not calculate a valid display stretch")
    return float(low), float(high)


def merge_frame(mosaic: np.ndarray, frame: Frame, transform: Affine, crs: rasterio.crs.CRS) -> None:
    with rasterio.open(frame.path) as dataset:
        source = dataset.read(1, out_dtype="float32")
        source[dataset.read_masks(1) == 0] = np.nan
        projected = np.full(mosaic.shape, np.nan, dtype=np.float32)
        reproject(source=source, destination=projected, src_transform=dataset.transform, src_crs=dataset.crs,
                  src_nodata=np.nan, dst_transform=transform, dst_crs=crs, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    valid = np.isfinite(projected)
    mosaic[valid] = projected[valid]


def encode(frames: list[Frame], output: Path, transform: Affine, width: int, height: int,
           crs: rasterio.crs.CRS, display: tuple[float, float], fps: float, crf: int, ecc_shift_seconds: float | None) -> None:
    temporary = output.with_name(f".{output.stem}.part.webm")
    video_height = height
    command = ["ffmpeg", "-hide_banner", "-loglevel", "warning", "-f", "rawvideo", "-pixel_format", "rgba",
               "-video_size", f"{width}x{video_height}", "-framerate", str(fps), "-i", "-", "-c:v", "libvpx-vp9",
               "-crf", str(crf), "-b:v", "0", "-row-mt", "1", "-threads", "8", "-auto-alt-ref", "0",
               "-pix_fmt", "yuva420p", "-y", str(temporary)]
    mosaic = np.full((height, width), np.nan, dtype=np.float32)
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    try:
        # Video frame 1 corresponds directly to the first acquired ortho.
        for index, frame in enumerate(frames, start=1):
            merge_frame(mosaic, frame, transform, crs)
            low, high = display
            valid = np.isfinite(mosaic)
            image = np.clip((mosaic - low) * 255 / (high - low), 0, 255)
            image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0).astype(np.uint8)
            rgb = np.repeat(image[:, :, None], 3, axis=2)
            alpha = np.where(valid, 255, 0).astype(np.uint8)
            image = np.dstack((rgb, alpha))
            assert process.stdin is not None
            process.stdin.write(image.tobytes())
            if index == 1 or index % 100 == 0 or index == len(frames):
                print(f"Rendered {index}/{len(frames)}: {frame.acquired}", flush=True)
        assert process.stdin is not None
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg failed while encoding the WebM")
        shutil.move(temporary, output)
    finally:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            process.kill()
        if temporary.exists():
            temporary.unlink()


def write_manifest(output: Path, frames: list[Frame], transform_: Affine, width: int, height: int,
                   crs: rasterio.crs.CRS, fps: float) -> Path:
    raster_bounds = array_bounds(height, width, transform_)
    wgs84 = rasterio.crs.CRS.from_epsg(4326)
    west, south, east, north = transform_bounds(crs, wgs84, *raster_bounds, densify_pts=21)
    leaflet_bounds = [[south, west], [north, east]]
    frame_entries = []
    for index, frame in enumerate(frames):
        with rasterio.open(frame.path) as dataset:
            frame_bounds = tuple(dataset.bounds)
            if dataset.crs != wgs84:
                frame_bounds = transform_bounds(dataset.crs, wgs84, *frame_bounds, densify_pts=21)
                center_x, center_y = dataset.xy((dataset.height - 1) / 2, (dataset.width - 1) / 2)
                center_x, center_y = transform(dataset.crs, wgs84, [center_x], [center_y])
                center = [float(center_y[0]), float(center_x[0])]
            else:
                center_x, center_y = dataset.xy((dataset.height - 1) / 2, (dataset.width - 1) / 2)
                center = [float(center_y), float(center_x)]
        frame_entries.append({
            "video_frame": index + 1,
            "file": frame.path.name,
            "time": frame.acquired,
            "frame_id": frame.frame_id,
            "attitude": list(frame.attitude) if frame.attitude is not None else None,
            "ecc_norm": frame.ecc_norm if np.isfinite(frame.ecc_norm) else None,
            "center": {"lat": center[0], "lon": center[1]},
            "bbox": [float(frame_bounds[0]), float(frame_bounds[1]),
                      float(frame_bounds[2]), float(frame_bounds[3])],
            "bounds": {
                "west": float(frame_bounds[0]), "south": float(frame_bounds[1]),
                "east": float(frame_bounds[2]), "north": float(frame_bounds[3]),
                "leaflet": [[float(frame_bounds[1]), float(frame_bounds[0])],
                             [float(frame_bounds[3]), float(frame_bounds[2])]],
            },
        })
    manifest_path = output.with_suffix(".manifest.json")
    payload = {
        "type": "ortho_track_manifest",
        "version": 2,
        "webm": output.name,
        "transparent_background": True,
        "initial_blank_video_frame": False,
        "first_ortho_video_frame": 1,
        "video_frame_count": len(frames),
        "crs": crs.to_string(),
        "bbox": [float(west), float(south), float(east), float(north)],
        "bbox_crs": "EPSG:4326",
        "canvas": {"width": width, "height": height,
                   "bounds": [float(west), float(south), float(east), float(north)],
                   "bounds_order": "west,south,east,north", "leaflet_bounds": leaflet_bounds,
                   "raster_crs": crs.to_string(), "raster_bounds": list(raster_bounds),
                   "transform": list(transform_), "fps": fps},
        "frames": frame_entries,
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path

def main() -> int:
    args = parse_args()
    if not args.tif_dir.is_dir():
        raise ValueError(f"Not a directory: {args.tif_dir}")
    if args.fps <= 0 or args.width < 2 or args.sample_stride < 1:
        raise ValueError("--fps, --width, and --sample-stride must be positive")
    if not 0 <= args.percentiles[0] < args.percentiles[1] <= 100:
        raise ValueError("--percentiles must be ordered values from 0 to 100")
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg was not found on PATH")
    if not args.config.is_file():
        raise ValueError(f"Config file does not exist: {args.config}")
    frames = discover_frames(args.tif_dir)
    config, imu_source = load_config_and_source(args.tif_dir, args.config)
    frames = add_attitude(frames, args.config, config, imu_source)
    frames, ecc_shift_seconds = add_ecc_norm(frames, args.config, config, imu_source)
    if args.max_frames:
        frames = frames[:args.max_frames]
    output = args.output or args.tif_dir / f"{args.tif_dir.name}_track.webm"
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output exists: {output} (pass --overwrite to replace it)")
    transform, width, height, crs = track_grid(frames, args.width)
    display = display_range(frames, tuple(args.percentiles), args.sample_stride)
    print(f"Encoding {len(frames)} cumulative frames at {width}x{height}; display range={display}")
    encode(frames, output, transform, width, height, crs, display, args.fps, args.crf, ecc_shift_seconds)
    manifest = write_manifest(output, frames, transform, width, height, crs, args.fps)
    print(f"Wrote {output}")
    print(f"Wrote manifest {manifest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileExistsError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
