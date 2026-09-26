#!/usr/bin/env python3
"""Center-crop tif_f1 images to 320 x 241 while preserving TIFF metadata."""
import argparse
import datetime
import json
import shutil

import numpy as np
import subprocess
from pathlib import Path

import rasterio
from rasterio.windows import Window


def copy_metadata(source: Path, destination: Path, width: int, height: int) -> None:
    """Copy non-structural metadata, including acquisition EXIF, to a crop."""
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        print("WARNING: exiftool not found; EXIF metadata was not copied.")
        return
    command = [
        exiftool, "-overwrite_original", "-TagsFromFile", str(source),
        "-EXIF:all", "-XMP:all", "-IPTC:all", "-MakerNotes:all",
        f"-EXIF:PixelXDimension={width}",
        f"-EXIF:PixelYDimension={height}",
        str(destination),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL,
                   stderr=subprocess.PIPE, text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input_dir", type=Path, help="Directory containing source TIFFs, normally .../tif_f1")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=241)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()

    if not a.input_dir.is_dir():
        raise FileNotFoundError(a.input_dir)
    output_dir = a.output_dir or a.input_dir.parent / "tif_f1_320"
    output_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(set(a.input_dir.glob("*.tif")) | set(a.input_dir.glob("*.TIF")))
    if not images:
        raise FileNotFoundError(f"No TIFF images found in {a.input_dir}")

    count = 0
    for source in images:
        destination = output_dir / source.name
        if destination.exists() and not a.overwrite:
            continue
        with rasterio.open(source) as src:
            if a.width > src.width or a.height > src.height:
                raise ValueError(f"Crop {a.width}x{a.height} exceeds {source.name} ({src.width}x{src.height})")
            col = (src.width - a.width) // 2
            row = (src.height - a.height) // 2
            window = Window(col, row, a.width, a.height)
            profile = src.profile.copy()
            profile.update(
                width=a.width,
                height=a.height,
                count=4,
                dtype="uint8",
                photometric="RGB",
                transform=src.window_transform(window),
            )
            tags = src.tags().copy()
            # Orthority reads the acquisition time from JSON ImageDescription.
            # Telops files may instead expose it as TIFFTAG_DATETIME, e.g.
            # ``260910 155428440326``. Convert that tag to the same JSON
            # representation used by already-corrected crops.
            description = tags.get("TIFFTAG_IMAGEDESCRIPTION")
            metadata = {}
            if description:
                try:
                    metadata = json.loads(description)
                except json.JSONDecodeError:
                    metadata = {}
            if "Time" not in metadata:
                date_value = tags.get("TIFFTAG_DATETIME") or tags.get("DateTime")
                if date_value:
                    try:
                        parsed_time = datetime.datetime.strptime(
                            str(date_value).strip(), "%y%m%d %H%M%S%f"
                        )
                    except ValueError as exc:
                        raise ValueError(
                            f"Unsupported Telops TIFF datetime in {source}: {date_value!r}"
                        ) from exc
                    metadata["Time"] = parsed_time.strftime("%Y-%m-%d %H:%M:%S.%f")
            if "Time" not in metadata:
                raise ValueError(
                    f"No acquisition time found in source TIFF: {source}. "
                    "The crop must contain TIFFTAG_DATETIME or JSON Time."
                )
            tags["TIFFTAG_IMAGEDESCRIPTION"] = json.dumps(metadata, separators=(",", ":"))
            band_tags = [src.tags(index) for index in range(1, src.count + 1)]
            source_data = src.read(window=window)
            # Store monochrome data as RGB+A: the grayscale plane is repeated
            # in R, G and B, and alpha is opaque.
            if source_data.shape[0] >= 3:
                gray = (
                    0.299 * source_data[0].astype(np.float32)
                    + 0.587 * source_data[1].astype(np.float32)
                    + 0.114 * source_data[2].astype(np.float32)
                )
            else:
                gray = source_data[0].astype(np.float32)
            gray = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
            data = np.concatenate(
                (np.repeat(gray[None, ...], 3, axis=0),
                 np.full((1, a.height, a.width), 255, dtype=np.uint8)),
                axis=0,
            )
            with rasterio.open(destination, "w", **profile) as dst:
                dst.write(data)
                if tags:
                    dst.update_tags(**tags)
                for index, values in enumerate(band_tags, start=1):
                    if values:
                        dst.update_tags(index, **values)
                if src.nodata is not None:
                    dst.nodata = src.nodata
        # Rasterio preserves GDAL/raster tags above, but normally drops the
        # source EXIF block. Copy acquisition metadata after closing the TIFF.
        copy_metadata(source, destination, a.width, a.height)
        count += 1

    print(f"Created {count} cropped images in {output_dir}")
    print(f"Output dimensions: {a.width} x {a.height} pixels")


if __name__ == "__main__":
    main()
