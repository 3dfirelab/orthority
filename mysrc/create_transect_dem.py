#!/usr/bin/env python3
"""Build a buffered RGE ALTI DEM from image timing and the matching IMU track."""
import argparse, re, subprocess, tempfile
from datetime import datetime
from pathlib import Path
import numpy as np
import xarray as xr
from PIL import Image
from pyproj import Transformer

def image_time(path):
    with Image.open(path) as image:
        description = str(image.tag_v2.get(270, ""))
        match = re.search(r'"Time"\s*:\s*"([^"]+)"', description)
        if match:
            return np.datetime64(match.group(1).replace(" ", "T"))
        tag = str(image.tag_v2.get(306, ""))
    match = re.search(r'(\d{6})[ _](\d{12})', tag)
    if match:
        day, clock = match.groups()
        return np.datetime64(datetime.strptime(day + clock, "%y%m%d%H%M%S%f"))
    raise ValueError(f"No usable acquisition time in {path}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("transect_dir", type=Path)
    p.add_argument("--dem-root", type=Path, default=Path("/data/shared/RGEALTI_IGN"))
    p.add_argument("--imu-dir", type=Path, default=None)
    p.add_argument("--image-dir", type=Path, default=None)
    p.add_argument("--buffer-km", type=float, default=5.0)
    a = p.parse_args()
    transect = a.transect_dir.resolve()
    image_dir = a.image_dir or transect / "tif_f1"
    imu_dir = a.imu_dir or transect.parent.parent / "imu"
    images = sorted(image_dir.glob("*.tif"))
    if not images:
        raise FileNotFoundError(f"No TIFF images found in {image_dir}")
    times = np.array([image_time(path) for path in images], dtype="datetime64[ns]")
    start, end = times.min(), times.max()
    imu_files = sorted(imu_dir.glob("*.nc"))
    if not imu_files:
        raise FileNotFoundError(f"No NetCDF IMU file found in {imu_dir}")
    with xr.open_dataset(imu_files[0]) as ds:
        imu_time = ds["time"].values.astype("datetime64[ns]")
        inside = (imu_time >= start) & (imu_time <= end)
        if not inside.any():
            raise ValueError(f"Image interval {start} to {end} does not overlap IMU data")
        lat = np.asarray(ds["LATITUDE"].values, float)[inside]
        lon = np.asarray(ds["LONGITUDE"].values, float)[inside]
    good = np.isfinite(lat) & np.isfinite(lon)
    if not good.any():
        raise ValueError("No finite IMU positions in the image time interval")
    to_l93 = Transformer.from_crs("EPSG:4326", "EPSG:2154", always_xy=True)
    x, y = to_l93.transform(lon[good], lat[good])
    b = a.buffer_km * 1000.0
    x1, x2 = float(np.min(x)-b), float(np.max(x)+b)
    y1, y2 = float(np.min(y)-b), float(np.max(y)+b)
    outdir = transect / "dem"
    outdir.mkdir(parents=True, exist_ok=True)
    transect_name = transect.name.split("-", 1)[0]
    vrt, dem = outdir / "rgealti_source.vrt", outdir / f"{transect_name}_rgealti.tif"
    tiles = sorted(a.dem_root.rglob("*.asc"))
    if not tiles:
        raise FileNotFoundError(f"No RGE ALTI .asc tiles found below {a.dem_root}")
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(str(tile) for tile in tiles) + "\n")
        tile_list = f.name
    try:
        subprocess.run(["gdalbuildvrt", "-overwrite", "-a_srs", "EPSG:2154", "-input_file_list", tile_list, str(vrt)], check=True)
    finally:
        Path(tile_list).unlink(missing_ok=True)
    subprocess.run(["gdalwarp", "-overwrite", "-s_srs", "EPSG:2154", "-t_srs", "EPSG:2154",
                        "-te", str(x1), str(y1), str(x2), str(y2), "-tr", "1", "1", "-r", "bilinear",
                        "-co", "TILED=YES", "-co", "COMPRESS=DEFLATE", str(vrt), str(dem)], check=True)
    print(f"Transect: {transect.name}")
    print(f"Image time range: {start} to {end} ({len(images)} images)")
    print(f"IMU source: {imu_files[0]}")
    print(f"Lambert-93 extent with {a.buffer_km:g} km buffer: {x1},{y1},{x2},{y2}")
    print(f"Wrote: {dem}")

if __name__ == "__main__":
    main()
