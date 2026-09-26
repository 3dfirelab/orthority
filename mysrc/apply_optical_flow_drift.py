#!/usr/bin/env python3
"""Apply a time-varying OPK model to an IMU GeoPackage and measure ECC quality.

Run this once per iteration after ``opk_corrections.csv`` has been produced.
It writes a corrected IMU GeoPackage and a metrics JSON/CSV.  Supplying the
newly orthorectified directory with ``--ortho-dir`` makes the metrics directly
comparable between iterations.
"""
from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import yaml

from calibrate_orthority_imu import ImagePair, _prepare_reference
from estimate_imu_drift import _ecc_translation


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Optical-flow drift YAML")
    parser.add_argument("--ortho-dir", type=Path, help="Corrected ortho directory to validate")
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-orthorectify", action="store_true", help="Only write corrected IMU and evaluate --ortho-dir")
    return parser.parse_args()


def time_frames(directory: Path) -> list[tuple[Path, pd.Timestamp]]:
    frames = []
    for path in directory.glob("*_ORTHO.tif"):
        with rasterio.open(path) as dataset:
            value = dataset.tags().get("Time")
        if value:
            frames.append((path, pd.Timestamp(value, tz="UTC")))
    return sorted(frames, key=lambda item: item[1])


def interpolate_model(model: pd.DataFrame, values: pd.Series) -> np.ndarray:
    model_time = pd.to_datetime(model["time"], utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    source_time = pd.to_datetime(values, utc=True).astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    return np.interp(source_time, model_time, values=model.iloc[:, 0].to_numpy(float), left=0.0, right=0.0)


def corrected_imu(config: dict, output: Path) -> Path:
    """Copy the delivered IMU unchanged; camera-frame drift stays external."""
    imu = gpd.read_file(config["imu"])
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    imu.to_file(output, driver="GPKG")
    return output


def orthorectify_iteration(config: dict, config_path: Path, corrected_imu: Path, output: Path) -> Path:
    """Create an isolated RunOrtho configuration and generate corrected orthos."""
    dataset_config = Path(config["dataset_config"])
    if not dataset_config.is_absolute():
        dataset_config = config_path.resolve().parent.parent / dataset_config
    run_config = yaml.safe_load(dataset_config.read_text())
    corrected_dir = output / "full_ortho_corrected"
    run_config["extractionName"] = f"{config['transect']}_{config['imu_source']}"
    run_config["input_dir"] = str(config["input_dir"])
    run_config["output_dir"] = str(corrected_dir)
    run_config["imu"] = str(corrected_imu)
    run_config["calibration"] = str(config["calibration"])
    generated = output / "runOrtho_corrected.yaml"
    generated.write_text(yaml.safe_dump(run_config, sort_keys=False))
    model_path = Path(config["output_dir"]) / "opk_corrections.csv"
    subprocess.run([sys.executable, "runOrtho.py", "--config", str(generated),
                    "--drift-model", str(model_path)], check=True)
    return corrected_dir


def ecc_metrics(directory: Path, offsets: list[float]) -> tuple[pd.DataFrame, dict]:
    frames = time_frames(directory)
    if len(frames) < 2:
        raise ValueError(f"Fewer than two timestamped orthos in {directory}")
    times = np.asarray([time.value for _, time in frames], dtype=np.int64)
    rows = []
    for reference_path, reference_time in frames:
        for offset in offsets:
            target = reference_time.value + int(offset * 1e9)
            index = int(np.argmin(np.abs(times - target)))
            if abs(times[index] - target) > 60_000_000:
                continue
            candidate_path, candidate_time = frames[index]
            if candidate_time <= reference_time:
                continue
            try:
                prepared = _prepare_reference(ImagePair(image=candidate_path, reference=reference_path), 1024)
                result = _ecc_translation(prepared, candidate_path, 200, 1.0e-6)
                rows.append({"reference_time": reference_time.isoformat(), "time": candidate_time.isoformat(),
                             "offset_s": (candidate_time - reference_time).total_seconds(), **result, "success": True})
            except (OSError, ValueError, cv2.error, rasterio.errors.RasterioError) as error:
                rows.append({"reference_time": reference_time.isoformat(), "time": candidate_time.isoformat(),
                             "offset_s": (candidate_time - reference_time).total_seconds(), "success": False, "error": str(error)})
    table = pd.DataFrame(rows)
    values = table.loc[table.success, "runortho_shift_apply_norm_px"].dropna()
    metrics = {"pair_count": int(len(table)), "success_count": int(table.success.sum()),
               "success_rate": float(table.success.mean()) if len(table) else 0.0,
               "ecc_norm_mean_px": float(values.mean()) if len(values) else None,
               "ecc_norm_median_px": float(values.median()) if len(values) else None,
               "ecc_norm_p90_px": float(values.quantile(.90)) if len(values) else None,
               "ecc_norm_p95_px": float(values.quantile(.95)) if len(values) else None,
               "ecc_norm_max_px": float(values.max()) if len(values) else None}
    return table, metrics


def main() -> int:
    args = arguments()
    config = yaml.safe_load(args.config.read_text())
    output = Path(config["output_dir"]) / f"iteration_{args.iteration:02d}"
    if output.exists() and any(output.iterdir()) and not args.overwrite:
        raise ValueError(f"Iteration output exists: {output}; pass --overwrite")
    imu_path = corrected_imu(config, output / "corrected_imu.gpkg")
    ortho_dir = args.ortho_dir
    if ortho_dir is None and not args.no_orthorectify:
        ortho_dir = orthorectify_iteration(config, args.config, imu_path, output)
    ortho_dir = ortho_dir or Path(config["reference_dir"])
    offsets = sorted({abs(float(value)) for value in config["validation_link_offsets_s"]})
    links, metrics = ecc_metrics(ortho_dir, offsets)
    metrics.update({"iteration": args.iteration, "imu": str(imu_path), "ortho_dir": str(ortho_dir)})
    links.to_csv(output / "ecc_pairs.csv", index=False)
    (output / "ecc_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2)
