#!/bin/bash
#SBATCH --job-name=az260007_time_shift
#SBATCH --partition=prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-time-shift-%A_%a.out
#SBATCH --error=slurm-time-shift-%A_%a.err

set -euo pipefail
source "$HOME/miniforge3/bin/activate" orthoriry
cd "$SLURM_SUBMIT_DIR"

BASE_CONFIG="config/calibration-az260007-bas00001.yaml"
SEARCH_DIR="/data/shared/PIPER/az260007/Transects/bas00001/calib/time_shift_search_slurm_20260916"
mkdir -p "$SEARCH_DIR"

make_config() {
    local shift="$1"
    local run_dir="$2"
    python - "$BASE_CONFIG" "$shift" "$run_dir" <<'PY'
import sys
from pathlib import Path
import yaml

base = yaml.safe_load(Path(sys.argv[1]).read_text())
shift = float(sys.argv[2])
run_dir = Path(sys.argv[3])
run_dir.mkdir(parents=True, exist_ok=True)
base["output"] = str(run_dir / "imu_camera_calibration.json")
base["work_dir"] = str(run_dir / "work")
base["time_shift_to_add_to_image"] = shift
(run_dir / "calibration.yaml").write_text(yaml.safe_dump(base, sort_keys=False))
PY
}

run_shift() {
    local shift="$1"
    local label
    label=$(printf '%+.1f' "$shift" | sed 's/+//; s/-/minus/; s/\./p/')
    local run_dir="$SEARCH_DIR/shift_${label}"
    if [[ -s "$run_dir/calibration.log" ]]; then
        echo "Skipping existing shift ${shift}s"
        return
    fi
    make_config "$shift" "$run_dir"
    python calibrate_orthority_imu.py "$run_dir/calibration.yaml" \
        --time-shift-to-add-to-image "$shift" \
        > "$run_dir/calibration.log" 2>&1
}

phase="${1:-submit}"
script_path="/home/paugam/Src/orthority/mysrc/slurm_time_shift_calibration.sh"

if [[ "$phase" == "submit" ]]; then
    coarse_job=$(sbatch --parsable --array=0-7%16 "$script_path" coarse)
    prepare_job=$(sbatch --parsable --dependency="afterok:${coarse_job}" "$script_path" prepare)
    refine_job=$(sbatch --parsable --dependency="afterok:${prepare_job}" --array=0-10%16 "$script_path" refine)
    echo "Submitted coarse array: ${coarse_job}"
    echo "Submitted preparation job: ${prepare_job}"
    echo "Submitted refinement array: ${refine_job}"
elif [[ "$phase" == "coarse" ]]; then
    shifts=(0.0 0.5 1.0 1.5 2.0 2.5 3.0 3.5)
    run_shift "${shifts[$SLURM_ARRAY_TASK_ID]}"
elif [[ "$phase" == "prepare" ]]; then
    # Select the best coarse shift and create the 0.1 s refinement list.
    python - "$SEARCH_DIR" <<'PY2'
import re, sys
from pathlib import Path
root = Path(sys.argv[1])
rows = []
for log in root.glob("shift_*/calibration.log"):
    text = log.read_text(errors="replace")
    times = re.findall(r"time_shift_to_add_to_image:\s*([+-]?[0-9.]+)", text)
    lines = [line for line in text.splitlines() if "pair" in line and "r=" in line and "o=" in line]
    pairs = re.findall(r"pair\d+\[r=([+-]?[0-9.]+),o=([+-]?[0-9.]+)\]", lines[-1]) if lines else []
    if times and pairs:
        r = sum(float(x) for x, _ in pairs) / len(pairs)
        o = sum(float(y) for _, y in pairs) / len(pairs)
        rows.append((float(times[-1]), r, o, (r + o) / 2))
if len(rows) < 8:
    raise SystemExit(f"Only {len(rows)}/8 coarse calibrations completed")
best = max(rows, key=lambda row: row[3])
print(f"Best coarse shift: {best[0]:+.1f}s, r={best[1]:.4f}, o={best[2]:.4f}")
start = max(0.0, best[0] - 0.5)
stop = min(3.5, best[0] + 0.5)
shifts = [round(start + i * 0.1, 1) for i in range(round((stop-start)/0.1)+1)]
(root / "refine_shifts.txt").write_text("".join(f"{x:.1f}\n" for x in shifts))
print("Refinement shifts:", " ".join(f"{x:.1f}" for x in shifts))
PY2
elif [[ "$phase" == "refine" ]]; then
    shift=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$SEARCH_DIR/refine_shifts.txt")
    if [[ -z "$shift" ]]; then
        echo "No refinement shift for task ${SLURM_ARRAY_TASK_ID}; exiting"
        exit 0
    fi
    run_shift "$shift"
else
    echo "Unknown phase: $phase" >&2
    exit 2
fi
