#!/usr/bin/env python3
"""Generate an orthorectification YAML from the angle-bracket template."""
from __future__ import annotations

import argparse
import re
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flightname", required=True)
    parser.add_argument("--transectname", required=True)
    parser.add_argument("--imuname", required=True)
    parser.add_argument("--imufilename", required=True)
    parser.add_argument("--flightdate", required=True, help="Flight date, e.g. 20260910")
    parser.add_argument("--defaultcalibration", required=True, help="Fallback calibration JSON path")
    parser.add_argument("--calib-transect", type=int, required=True, help="Calibration transect number")
    parser.add_argument("--note", required=True, help="Description of the flight or transect")
    parser.add_argument(
        "--time-shift", type=float, default=0.0,
        help="Seconds to add to image timestamps (default: 0)",
    )
    parser.add_argument(
        "--root-data-dir", default="/data/shared/PIPER",
        help="Root data directory substituted for <root_data_dir> in the template",
    )
    parser.add_argument(
        "--template", type=Path,
        default=Path("config/config-flightname-transectname-imuname.yaml"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("config"))
    args = parser.parse_args()

    calibration_suffix = f"{args.imuname}-calib{args.calib_transect:05d}"
    output_transect = args.transectname
    if not re.search(r"-calib\d{5}$", output_transect):
        output_transect = f"{output_transect}-{calibration_suffix}"

    calibration = Path(args.defaultcalibration).expanduser()
    if not calibration.is_absolute():
        calibration = (Path.cwd() / calibration).resolve()
    if not calibration.is_file():
        raise FileNotFoundError(f"Calibration file not found: {calibration}")
    print(f"Using calibration: {calibration}")

    calibration_payload = json.loads(calibration.read_text(encoding="utf-8"))
    calibration_pairs = calibration_payload.get("pairs") or []
    calibration_image = calibration_pairs[0].get("image", "") if calibration_pairs else ""
    input_image_dir = "tif_f1_320" if "tif_f1_320" in calibration_image else "tif_f1"
    camera_io_suffix = "-320" if input_image_dir == "tif_f1_320" else ""
    print(f"Calibration image directory: {input_image_dir} (from {calibration_image or 'no pairs recorded'})")

    values = {
        "flightname": args.flightname,
        "transectname": output_transect,
        "demname": output_transect.split("-", 1)[0],
        "imuname": args.imuname,
        "imufilename": args.imufilename,
        "flightdate": args.flightdate,
        "timelagcamera": f"{args.time_shift:g}",
        "calibration": str(calibration),
        "note": json.dumps(args.note),
        "root_data_dir": args.root_data_dir,
        "input_image_dir": input_image_dir,
        "camera_io_suffix": camera_io_suffix,
    }
    template = args.template.read_text(encoding="utf-8")

    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        if name not in values:
            raise ValueError(f"Unknown template placeholder: <{name}>")
        return values[name]

    rendered = re.sub(r"<([^<>]+)>", replace, template)
    unresolved = re.findall(r"<([^<>]+)>", rendered)
    if unresolved:
        raise ValueError(f"Unresolved template placeholders: {sorted(set(unresolved))}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"config-{args.flightname}-{output_transect}.yaml"
    output.write_text(rendered, encoding="utf-8")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
