#!/bin/bash
#SBATCH --job-name=az260007_ts_adapt
#SBATCH --partition=prod
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=slurm-ts-adapt-%A_%a.out
#SBATCH --error=slurm-ts-adapt-%A_%a.err

set -euo pipefail
source /home/paugam/miniforge3/bin/activate orthoriry
cd "$SLURM_SUBMIT_DIR"
BASE_CONFIG="config/calibration-az260007-bas00001.yaml"
SCRIPT_PATH="/home/paugam/Src/orthority/mysrc/slurm_time_shift_adaptive_search.sh"
SEARCH_DIR="/data/shared/PIPER/az260007/Transects/bas00001/calib/time_shift_search_slurm_adaptive_20260916"
MIN_SHIFT=2.3
MAX_SHIFT=2.5
mkdir -p "$SEARCH_DIR"

make_config() {
    local shift="$1" run_dir="$2"
    python - "$BASE_CONFIG" "$shift" "$run_dir" <<'PYCFG'
import sys
from pathlib import Path
import yaml
base=yaml.safe_load(Path(sys.argv[1]).read_text())
run_dir=Path(sys.argv[3]); run_dir.mkdir(parents=True,exist_ok=True)
base["output"]=str(run_dir/"imu_camera_calibration.json")
base["work_dir"]=str(run_dir/"work")
base["time_shift_to_add_to_image"]=float(sys.argv[2])
(run_dir/"calibration.yaml").write_text(yaml.safe_dump(base,sort_keys=False))
PYCFG
}

run_shift() {
    local shift="$1" label run_dir
    label=$(printf '%+.6f' "$shift" | sed 's/+//; s/-/minus/; s/\./p/')
    run_dir="$SEARCH_DIR/shift_${label}"
    if [[ -s "$run_dir/calibration.log" && -f "$run_dir/imu_camera_calibration.json" ]]; then
        echo "Skipping completed shift ${shift}s"; return
    fi
    make_config "$shift" "$run_dir"
    python calibrate_orthority_imu.py "$run_dir/calibration.yaml" \
      --time-shift-to-add-to-image "$shift" > "$run_dir/calibration.log" 2>&1
}

prepare_next() {
    local stage="$1" step="$2"
    python - "$SEARCH_DIR" "$stage" "$step" "$MIN_SHIFT" "$MAX_SHIFT" <<'PYPREP'
import csv,re,sys
from pathlib import Path
root=Path(sys.argv[1]); stage=int(sys.argv[2]); step=float(sys.argv[3]); lo=float(sys.argv[4]); hi=float(sys.argv[5])
requested=[float(x) for x in (root/f"stage_{stage}_shifts.txt").read_text().split()]
rows=[]
for shift in requested:
    label=f"{shift:+.6f}".replace("+","").replace("-","minus").replace(".","p")
    log=root/f"shift_{label}"/"calibration.log"
    text=log.read_text(errors="replace") if log.is_file() else ""
    costs=re.findall(r"Selected full optimization .*?cost[= ]([0-9]+\.[0-9]+)",text)
    lines=[x for x in text.splitlines() if "pair" in x and "r=" in x and "o=" in x]
    pairs=re.findall(r"pair\d+\[r=([+-]?[0-9.]+),o=([+-]?[0-9.]+)\]",lines[-1]) if lines else []
    if costs and pairs:
        rows.append({"time_shift_s":shift,"cost":float(costs[-1]),"mean_r":sum(float(x) for x,_ in pairs)/len(pairs),"mean_overlap":sum(float(y) for _,y in pairs)/len(pairs),"pairs":len(pairs),"run_dir":str(log.parent)})
if len(rows)!=len(requested): raise SystemExit(f"Only {len(rows)}/{len(requested)} shifts completed in stage {stage}")
allrows=[]
for log in root.glob("shift_*/calibration.log"):
    text=log.read_text(errors="replace"); ts=re.findall(r"time_shift_to_add_to_image:\s*([+-]?[0-9.]+)",text); costs=re.findall(r"Selected full optimization .*?cost[= ]([0-9]+\.[0-9]+)",text)
    lines=[x for x in text.splitlines() if "pair" in x and "r=" in x and "o=" in x]; pairs=re.findall(r"pair\d+\[r=([+-]?[0-9.]+),o=([+-]?[0-9.]+)\]",lines[-1]) if lines else []
    if ts and costs and pairs: allrows.append({"time_shift_s":float(ts[-1]),"cost":float(costs[-1]),"mean_r":sum(float(x) for x,_ in pairs)/len(pairs),"mean_overlap":sum(float(y) for _,y in pairs)/len(pairs),"pairs":len(pairs),"run_dir":str(log.parent)})
with (root/"time_shift_search.csv").open("w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["time_shift_s","cost","mean_r","mean_overlap","pairs","run_dir"]); w.writeheader(); w.writerows(sorted(allrows,key=lambda x:x["time_shift_s"]))
best=min(rows,key=lambda x:x["cost"]); print(f"Stage {stage} best: shift={best['time_shift_s']:+.6f}s cost={best['cost']:.6f} r={best['mean_r']:.4f} o={best['mean_overlap']:.4f}")
if step<=0.005000001: print(f"FINAL SHIFT {best['time_shift_s']:+.6f}s"); raise SystemExit
next_step=max(step/2,0.005); next_stage=stage+1
shifts=list(dict.fromkeys(round(min(hi,max(lo,best["time_shift_s"]+(i-2)*next_step)),6) for i in range(5)))
(root/f"stage_{next_stage}_shifts.txt").write_text("".join(f"{x:.6f}\n" for x in shifts)); (root/f"stage_{next_stage}_step.txt").write_text(f"{next_step:.6f}\n")
print(f"Next stage {next_stage}: step={next_step:.6f}s shifts={shifts}")
PYPREP
}

phase="${1:-submit}"
if [[ "$phase" == "submit" ]]; then
    if [[ -f "$SEARCH_DIR/stage_0_shifts.txt" ]]; then echo "Search already exists: $SEARCH_DIR" >&2; exit 2; fi
    printf '2.300000\n2.350000\n2.400000\n2.450000\n2.500000\n' > "$SEARCH_DIR/stage_0_shifts.txt"
    coarse=$(sbatch --parsable --array=0-4%16 "$SCRIPT_PATH" stage 0)
    sbatch --parsable --dependency="afterok:${coarse}" "$SCRIPT_PATH" prepare 0 0.05
    echo "Submitted coarse array: $coarse"
elif [[ "$phase" == "stage" ]]; then
    stage="$2"; shift_value=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$SEARCH_DIR/stage_${stage}_shifts.txt")
    [[ -n "$shift_value" ]] && run_shift "$shift_value"
elif [[ "$phase" == "prepare" ]]; then
    stage="$2"; step="$3"; prepare_next "$stage" "$step"
    if awk "BEGIN {exit !($step <= 0.005000001)}"; then exit 0; fi
    next_stage=$((stage+1)); next_step=$(cat "$SEARCH_DIR/stage_${next_stage}_step.txt")
    array=$(sbatch --parsable --dependency="afterok:${SLURM_JOB_ID}" --array=0-4%16 "$SCRIPT_PATH" stage "$next_stage")
    sbatch --parsable --dependency="afterok:${array}" "$SCRIPT_PATH" prepare "$next_stage" "$next_step"
    echo "Submitted stage ${next_stage} array: ${array}"
else
    echo "Usage: $0 submit" >&2; exit 2
fi
