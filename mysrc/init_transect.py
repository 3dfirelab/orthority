#!/usr/bin/env python3
"""Initialise a transect directory and link its report-time raw images."""
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import tifffile


CAMERA_YAML = '''Proselica_4mm:
    type: opencv
    im_size: [1936, 1456]
    focal_len: [0.688510, 0.689796]
    cx: 0.0
    cy: 0.0
    k1: -0.1162685
    k2: 0.0825623
    p1: 0.0
    p2: 0.0
    k3: -0.02610017
'''


def read_image_time(path: Path):
    """Read the acquisition time embedded in a Telops TIFF."""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        description_tag = page.tags.get("ImageDescription")
        if description_tag is not None:
            try:
                metadata = json.loads(str(description_tag.value))
            except json.JSONDecodeError:
                metadata = {}
            value = metadata.get("Time")
            if value:
                for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
                    try:
                        return datetime.strptime(str(value), fmt)
                    except ValueError:
                        pass
        date_tag = page.tags.get("DateTime")
        if date_tag is not None:
            value = str(date_tag.value).strip()
            try:
                return datetime.strptime(value, "%y%m%d %H%M%S%f")
            except ValueError:
                pass
    return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--transects-dir', type=Path, required=True)
    p.add_argument('--prefix', choices=('bas', 'avant'), required=True)
    p.add_argument('--number', type=int, required=True)
    p.add_argument('--output-name', default=None, help='Optional output directory name')
    p.add_argument('--manifest', type=Path, required=True, help='CSV manifest containing transect_name,start_utc,end_utc')
    p.add_argument('--raw-dir', type=Path, required=True)
    a = p.parse_args()
    root = a.transects_dir / (a.output_name or f'{a.prefix}{a.number:05d}')
    for name in ('tif_f1', 'io', 'tif_f1_ortho', 'full_ortho_f1'):
        (root / name).mkdir(parents=True, exist_ok=True)
    transect_name = f'{a.prefix}{a.number:05d}'
    with a.manifest.open(newline='', encoding='utf-8') as stream:
        manifest = {row['transect_name']: row for row in csv.DictReader(stream)}
    if transect_name not in manifest:
        raise KeyError(f'{transect_name} not found in {a.manifest}')
    start = datetime.fromisoformat(manifest[transect_name]['start_utc'])
    end = datetime.fromisoformat(manifest[transect_name]['end_utc'])
    linked = []
    for image in a.raw_dir.glob('*.tif'):
        image_time = read_image_time(image)
        if image_time is None:
            raise ValueError(f"No acquisition time metadata found in {image}")
        if start <= image_time <= end:
            link = root / 'tif_f1' / image.name
            if not link.exists():
                link.symlink_to(image)
            linked.append(image)
    (root / 'io' / 'camera.yaml').write_text(CAMERA_YAML, encoding='utf-8')
    (root / 'io' / 'README.txt').write_text(
        'Orthority camera parameters for raw distorted images in ../tif_f1.\n',
        encoding='utf-8')
    print(f'Created {root}')
    print(f'Linked {len(linked)} raw images into {root / "tif_f1"}')
    print(f'Run undistort_one_image.py on any tif_f1 image; output goes to tif_f1_undistorted.')


if __name__ == '__main__':
    main()
