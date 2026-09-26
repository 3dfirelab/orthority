#!/usr/bin/env python3
"""Search image/IMU time shift by repeatedly running calibration."""
from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PAIR_RE = re.compile(r"pair\d+\[r=([+-]?[0-9]*\.?[0-9]+),o=([+-]?[0-9]*\.?[0-9]+)\]")


def values(start: float, stop: float, step: float) -> list[float]:
    count = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(count + 1)]


def parse_final_pairs(text: str) -> tuple[float, float, int]:
    lines = [line for line in text.splitlines() if "pair" in line and "r=" in line and "o=" in line]
    if not lines:
        raise ValueError("No pair diagnostics found in calibration output")
    matches = PAIR_RE.findall(lines[-1])
    if not matches:
        raise ValueError("Could not parse final pair diagnostics")
    correlations = np.asarray([float(r) for r, _ in matches], dtype=float)
    overlaps = np.asarray([float(o) for _, o in matches], dtype=float)
    return float(correlations.mean()), float(overlaps.mean()), len(matches)


def existing_results(search_dir: Path) -> list[dict]:
    results = []
    for log_path in sorted(search_dir.glob("shift_*/calibration.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        shift_match = re.search(r"time_shift_to_add_to_image:\s*([+-]?[0-9.]+)", text)
        if not shift_match:
            continue
        lines = [line for line in text.splitlines() if "pair" in line and "r=" in line and "o=" in line]
        if not lines:
            continue
        try:
            mean_r, mean_overlap, pairs = parse_final_pairs(text)
        except ValueError:
            continue
        shift = float(shift_match.group(1))
        results.append({
            "time_shift_s": shift, "status": "ok", "mean_r": mean_r,
            "mean_overlap": mean_overlap, "score": (mean_r + mean_overlap) / 2.0,
            "pairs": pairs, "return_code": 0, "run_dir": str(log_path.parent),
        })
    return results


def run_one(base: dict, config_path: Path, output_dir: Path, shift: float) -> dict:
    label = f"{shift:+.2f}".replace(".", "p").replace("+", "plus").replace("-", "minus")
    run_dir = output_dir / f"shift_{label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_config = dict(base)
    run_config["output"] = str(run_dir / "imu_camera_calibration.json")
    run_config["work_dir"] = str(run_dir / "work")
    run_config["time_shift_to_add_to_image"] = shift
    run_yaml = run_dir / "calibration.yaml"
    run_yaml.write_text(yaml.safe_dump(run_config, sort_keys=False), encoding="utf-8")
    log_path = run_dir / "calibration.log"
    command = [
        sys.executable, str(Path(__file__).with_name("calibrate_orthority_imu.py")),
        str(run_yaml), "--time-shift-to-add-to-image", str(shift),
    ]
    print(f"\n===== time shift {shift:+.1f} s =====", flush=True)
    completed = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log_path.write_text(completed.stdout, encoding="utf-8")
    print(completed.stdout, end="", flush=True)
    result = {"time_shift_s": shift, "status": "ok" if completed.returncode == 0 else "failed",
              "mean_r": np.nan, "mean_overlap": np.nan, "score": np.nan, "pairs": 0,
              "return_code": completed.returncode, "run_dir": str(run_dir)}
    if completed.returncode == 0:
        try:
            mean_r, mean_overlap, pairs = parse_final_pairs(completed.stdout)
            result.update(mean_r=mean_r, mean_overlap=mean_overlap,
                          score=(mean_r + mean_overlap) / 2.0, pairs=pairs)
        except ValueError as error:
            result["status"] = f"parse_failed: {error}"
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--search-dir", type=Path, default=None)
    parser.add_argument("--resume-dir", type=Path, default=None,
                        help="Read completed calibration logs from this directory and skip those shifts.")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=3.5)
    parser.add_argument("--coarse-step", type=float, default=0.5)
    parser.add_argument("--refine-step", type=float, default=0.1)
    args = parser.parse_args()
    base = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("Calibration config must be a YAML mapping")
    search_dir = args.search_dir or args.config.parent / "time_shift_search"
    search_dir.mkdir(parents=True, exist_ok=True)
    coarse = values(args.start, args.stop, args.coarse_step)
    results = existing_results(args.resume_dir) if args.resume_dir else []
    completed = {round(row["time_shift_s"], 10) for row in results}
    for shift in coarse:
        if round(shift, 10) in completed:
            print(f"Skipping existing coarse shift {shift:+.1f} s")
            continue
        results.append(run_one(base, args.config, search_dir, shift))
        completed.add(round(shift, 10))
    valid = [row for row in results if np.isfinite(row["score"])]
    if not valid:
        raise RuntimeError("No coarse time-shift calibration completed successfully")
    best = max(valid, key=lambda row: row["score"])
    # Refine one coarse-step on either side using a 0.1-second grid.
    refine_start = max(args.start, best["time_shift_s"] - args.coarse_step)
    refine_stop = min(args.stop, best["time_shift_s"] + args.coarse_step)
    refine = values(refine_start, refine_stop, args.refine_step)
    already = {round(row["time_shift_s"], 10) for row in results}
    for shift in refine:
        if round(shift, 10) not in already:
            results.append(run_one(base, args.config, search_dir, shift))
    fields = ["phase", "time_shift_s", "mean_r", "mean_overlap", "score", "pairs", "status", "return_code", "run_dir"]
    summary = search_dir / "time_shift_search.csv"
    with summary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in results:
            row = dict(row)
            row["phase"] = "coarse" if row["time_shift_s"] in coarse else "refine"
            writer.writerow(row)
    valid = [row for row in results if np.isfinite(row["score"])]
    best = max(valid, key=lambda row: row["score"])
    print("\n===== BEST TIME SHIFT =====")
    print(f"shift={best['time_shift_s']:+.1f} s mean_r={best['mean_r']:.4f} "
          f"mean_overlap={best['mean_overlap']:.4f} score={best['score']:.4f}")
    print(f"Summary: {summary}")
    print(f"Calibration directory: {best['run_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
