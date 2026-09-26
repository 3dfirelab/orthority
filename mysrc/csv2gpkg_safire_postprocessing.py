#!/usr/bin/env python3
"""Convert a SAFIRE postprocessing CSV to a gpkg with the 'aipov' schema.

The CSV columns are mapped positionally onto the same field names/order as
the existing *_safire.gpkg files in the same directory (see CSV_TO_GPKG_FIELDS
below). The CSV has no navigation 'status' code, so that field is left NULL
in the output rather than guessed.

Example:

    python3 csv2gpkg_safire_postprocessing.py \\
        /data/shared/PIPER/az260002/safire/az260002_PH-579_postprocessing.csv \\
        /data/shared/PIPER/az260002/safire/az260002_safire_postprocessed.gpkg
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
from pathlib import Path

from osgeo import ogr, osr

LAYER_NAME = "aipov"

# (csv_column, gpkg_field, ogr_field_type)
CSV_TO_GPKG_FIELDS = [
    ("datetime", "datetime_utc", ogr.OFTDateTime),
    ("true_heading(degree)", "heading_deg", ogr.OFTReal),
    ("roll(degree)", "roll_deg", ogr.OFTReal),
    ("pitch(degree)", "pitch_deg", ogr.OFTReal),
    ("angular_rate_X(degree/s)", "rot_rate_xv1", ogr.OFTReal),
    ("angular_rate_Y(degree/s)", "rot_rate_xv2", ogr.OFTReal),
    ("angular_rate_Z(degree/s)", "rot_rate_xv3", ogr.OFTReal),
    ("acceleration_X(m/s2)", "lin_acc_xv1", ogr.OFTReal),
    ("acceleration_Y(m/s2)", "lin_acc_xv2", ogr.OFTReal),
    ("acceleration_Z(m/s2)", "lin_acc_xv3", ogr.OFTReal),
    ("latitude(degree_north)", "latitude", ogr.OFTReal),
    ("longitude(degree_east)", "longitude", ogr.OFTReal),
    ("altitude(m)", "altitude_m", ogr.OFTReal),
    ("velocity_N(m/s)", "north_velocity", ogr.OFTReal),
    ("velocity_E(m/s)", "east_velocity", ogr.OFTReal),
    ("velocity_W(m/s)", "vertical_velocity", ogr.OFTReal),
    ("velocity_X(m/s)", "along_velocity_xv1", ogr.OFTReal),
    ("velocity_Y(m/s)", "across_velocity_xv2", ogr.OFTReal),
    ("velocity_Z(m/s)", "down_velocity_xv3", ogr.OFTReal),
    ("track(degree)", "true_course", ogr.OFTReal),
]
# Present in the reference schema but not derivable from this CSV.
EXTRA_NULL_FIELDS = [("status", ogr.OFTInteger64)]


def _parse_datetime(value: str) -> str:
    # CSV format: 20260520T083550.011600 -> gpkg format: 2026-05-20T08:35:50.011Z
    parsed = dt.datetime.strptime(value, "%Y%m%dT%H%M%S.%f")
    return parsed.strftime("%Y-%m-%dT%H:%M:%S.") + f"{parsed.microsecond // 1000:03d}Z"


def convert(csv_path: Path, out_path: Path) -> int:
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    driver = ogr.GetDriverByName("GPKG")
    dataset = driver.CreateDataSource(str(out_path))
    layer = dataset.CreateLayer(LAYER_NAME, srs, ogr.wkbPoint)

    # latitude/longitude are stored both as plain attribute fields and as
    # the point geometry in the reference *_safire.gpkg files.
    for _, field_name, field_type in CSV_TO_GPKG_FIELDS:
        layer.CreateField(ogr.FieldDefn(field_name, field_type))
    for field_name, field_type in EXTRA_NULL_FIELDS:
        layer.CreateField(ogr.FieldDefn(field_name, field_type))

    layer_defn = layer.GetLayerDefn()
    count = 0
    with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [name for name, _, _ in CSV_TO_GPKG_FIELDS if name not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing expected columns: {missing}")

        for row in reader:
            feature = ogr.Feature(layer_defn)
            for csv_col, field_name, field_type in CSV_TO_GPKG_FIELDS:
                value = row[csv_col]
                if field_type == ogr.OFTDateTime:
                    feature.SetField(field_name, _parse_datetime(value))
                else:
                    feature.SetField(field_name, float(value))

            longitude = float(row["longitude(degree_east)"])
            latitude = float(row["latitude(degree_north)"])
            point = ogr.Geometry(ogr.wkbPoint)
            point.AddPoint(longitude, latitude)
            feature.SetGeometry(point)

            layer.CreateFeature(feature)
            feature = None
            count += 1

    dataset = None
    return count


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="SAFIRE postprocessing CSV.")
    parser.add_argument("out", type=Path, help="Output .gpkg path.")
    return parser


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        count = convert(args.csv, args.out)
    except (OSError, ValueError, FileExistsError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(f"Wrote {count} features to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
