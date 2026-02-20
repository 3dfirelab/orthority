import os
import shutil
import subprocess
from datetime import datetime, timedelta
import tempfile
from pathlib import Path
import json
import pdb 
import sys 

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 (registers .rio accessor)
from rasterio.enums import Resampling
from pyproj import Transformer
import pandas as pd
from PIL import Image
from matplotlib import colormaps

SRC_NC = Path('f5_merged_time_Sijean01.nc')
OUT_DIR = Path('.')
OUT_BASE = OUT_DIR / 'ir38'
BOUNDS_FILE = OUT_DIR / 'ir38_bounds.json'
TIME_FILE = OUT_DIR / 'ir38_time.json'

# Controls
CRF = 30
CPU_USED = 2

# Use native resolution (no downsampling)
TARGET_SIZE = None

# Color mapping
CMAP = colormaps.get_cmap("inferno")
VMIN = 47.0
VMAX = 250.0


def parse_datetime_from_filename(fname):
    try:
        date_str = ''.join(fname.split('-')[-1].split('.')[:2])
        return datetime.strptime(date_str, '%Y%j%H%M%S%f')
    except Exception:
        return None


def convert_to_vp9(temp_dir, output_path_base, framerate, duration_sec, frame_size):
    pngs = sorted([f for f in os.listdir(temp_dir) if f.endswith('.png')])
    if not pngs:
        print(f"No images to convert in {temp_dir}")
        return

    timestamps = [parse_datetime_from_filename(f) for f in pngs]
    timestamps = [t for t in timestamps if t is not None]
    if not timestamps:
        print(f"No valid timestamps found in {temp_dir}")
        return

    timestamps.sort()
    start_time = timestamps[0]

    for i, fname in enumerate(pngs):
        old_path = os.path.join(temp_dir, fname)
        new_name = f"frame_{i:04d}.png"
        new_path = os.path.join(temp_dir, new_name)
        os.rename(old_path, new_path)

    start_str = start_time.strftime('%Y%jT%H%M%S.%fZ')
    output_path = f"{output_path_base}_new.webm"

    width, height = frame_size
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(framerate),
        '-i', os.path.join(temp_dir, 'frame_%04d.png'),
        '-vf', f'scale={width}:{height}:flags=neighbor',
        '-c:v', 'libvpx-vp9',
        '-crf', str(CRF),
        '-b:v', '0',
        '-r', str(framerate),
        '-g', '30',
        '-keyint_min', '30',
        '-sc_threshold', '0',
        '-cpu-used', str(CPU_USED),
        '-deadline', 'good',
        '-threads', '8',
        '-row-mt', '1',
        '-auto-alt-ref', '1',
        '-metadata', f'real_start_time={start_str}',
        '-metadata', f'real_duration={int(duration_sec)}',
        output_path,
    ]

    subprocess.run(cmd, check=True)
    print(f"Created video: {output_path}")


def generate_placeholder(dst_path, size):
    img = Image.new('RGBA', size, color=(0, 0, 0, 255))
    img.save(dst_path)


def main(transectname):

    root_data = '/data/shared/ATR42/as250026/Transects'
    SRC_NC = Path(f'{root_data}/{transectname}/full_ortho_f5/f5_merged_time_{transectname}.nc')
    OUT_DIR = Path(f'{root_data}/viewer/webm')
    OUT_BASE = OUT_DIR / f'ir38_{transectname}'
    BOUNDS_FILE = OUT_DIR / f'ir38_{transectname}_bounds.json'
    TIME_FILE = OUT_DIR / f'ir38_{transectname}_time.json'


    with xr.open_dataset(SRC_NC, decode_times=True) as ds, tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = temp_dir
        ir38 = ds["ir38"]
        if ir38.rio.crs is None:
            ir38 = ir38.rio.write_crs("EPSG:4326")

        # Reproject a single frame to establish the target grid in EPSG:3857.
        ref = ir38.isel(time=0).rio.reproject(
            "EPSG:3857",
            resampling=Resampling.nearest
        )

        t_len = ir38.sizes["time"]
        y_len, x_len = ref.shape

        stride_x = 1
        stride_y = 1

        x = ref["x"].values.astype("float64")
        y = ref["y"].values.astype("float64")
        x_step = x[1] - x[0]
        y_step = y[1] - y[0]
        if x_step == 0 or y_step == 0:
            raise RuntimeError("Invalid source grid step; check x/y arrays.")

        frame_size = (x_len, y_len)

        # Compute bounds from 3857 grid edges, then convert to EPSG:4326.
        dx = abs(x_step)
        dy = abs(y_step)
        min_x = float(np.min(x)) - dx * 0.5
        max_x = float(np.max(x)) + dx * 0.5
        min_y = float(np.min(y)) - dy * 0.5
        max_y = float(np.max(y)) + dy * 0.5

        to_4326 = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        min_lon, min_lat = to_4326.transform(min_x, min_y)
        max_lon, max_lat = to_4326.transform(max_x, max_y)

        bounds_payload = {
            "minX": min_lon,
            "minY": min_lat,
            "maxX": max_lon,
            "maxY": max_lat,
            "crs": "EPSG:4326"
        }
        with open(BOUNDS_FILE, "w", encoding="utf-8") as f:
            json.dump(bounds_payload, f)
        print(f"Wrote bounds to {BOUNDS_FILE}")

        t_vals = ds["time"].values
        dts = [pd.to_datetime(t).to_pydatetime() for t in t_vals]
        if t_len > 1:
            diffs = np.diff([t.timestamp() for t in dts])
            step_sec = float(np.median(diffs))
            if step_sec <= 0:
                step_sec = 0.02
        else:
            step_sec = 1.0

        fps = int(round(1.0 / step_sec))
        fps = max(1, min(fps, 60))
        duration_sec = (dts[-1] - dts[0]).total_seconds() + step_sec

        print(f"Frames: {t_len}")
        print(f"Step: {step_sec:.3f}s, fps={fps}")
        print(f"Using stride_x={stride_x}, stride_y={stride_y}")

        fill = ir38.rio.nodata if ir38.rio.nodata is not None else 0.0
        prev_dt = None
        expected_delta = step_sec
  
        nbre_missing_frame = 0 

        for i, dt in enumerate(dts):
            if prev_dt is not None:
                while (dt - prev_dt).total_seconds() > expected_delta * 1.5:
                    prev_dt = prev_dt + timedelta(seconds=expected_delta)
                    missing_name = f"ir38-{prev_dt.strftime('%Y%j.%H%M%S%f')}.png"
                    missing_path = os.path.join(frames_dir, missing_name)
                    generate_placeholder(missing_path, frame_size)
                    nbre_missing_frame +=1 

            frame_da = ir38.isel(time=i).rio.reproject(
                "EPSG:3857",
                transform=ref.rio.transform(),
                shape=ref.shape,
                resampling=Resampling.nearest,
                nodata=fill
            )
            frame = frame_da.values.astype('float32')
            mask = (frame == fill) | ~np.isfinite(frame)

            # Normalize to fixed range and apply inferno colormap.
            norm = (frame - VMIN) / (VMAX - VMIN)
            norm = np.clip(norm, 0, 1)
            rgba = (CMAP(norm) * 255).astype('uint8')

            rgba[mask, 3] = 0

            img = Image.fromarray(rgba, mode='RGBA')

            fname = f"ir38-{dt.strftime('%Y%j.%H%M%S%f')}.png"
            img.save(os.path.join(frames_dir, fname))

            prev_dt = dt
            if i % 100 == 0:
                print(f"Wrote frame {i+1}/{t_len}")
           
        print(f' nbfre of missing frame = {nbre_missing_frame}')

        # Write time metadata for viewer.
        time_payload = {
            "start_time_iso": dts[0].isoformat(),
            "end_time_iso": dts[-1].isoformat(),
            "duration_seconds": duration_sec,
            "fps": fps,
            "crs": "EPSG:4326"
        }
        with open(TIME_FILE, "w", encoding="utf-8") as f:
            json.dump(time_payload, f)
        print(f"Wrote time metadata to {TIME_FILE}")

        convert_to_vp9(frames_dir, OUT_BASE, framerate=fps, duration_sec=duration_sec, frame_size=frame_size)
        old_path = f"{OUT_BASE}_new.webm"
        new_path = f"{OUT_BASE}.webm"
        if os.path.exists(old_path):
            shutil.move(old_path, new_path)
            print(f"Moved {old_path} -> {new_path}")
        else:
            print(f"Source file {old_path} does not exist")


if __name__ == '__main__':
    
    if sys.argv[1] is None:
        print('missing transect name input')
        sys.exit()

    main(sys.argv[1])
