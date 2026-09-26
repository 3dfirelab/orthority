#!/bin/bash
#SBATCH --job-name=piper03_ortho_drift
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=10
#SBATCH --mem=20G
#SBATCH --partition=prod
##SBATCH --dependency=afterok:97402

# Runs orthorectification -> drift estimation for the piper03 transect
# (003_320256_safire), once per IMU source, in sequence (safire, then loa,
# then safire_post). Transect 003 has no manual-reference images of its own,
# so config-piper03.yaml's 'calibration' entries reuse the calibration JSONs
# already produced for piper01 (transect 001) -- run
# slurm_calib_ortho_drift_piper01.sh first so those exist. A failure in any
# step aborts the job (set -e); drop -e if you'd rather let later sources
# run even if an earlier one fails.
set -euo pipefail

source ~/miniforge3/bin/activate orthoriry

# Not "cd $(dirname ${BASH_SOURCE[0]})": slurmd stages the submitted script
# under /var/spool/slurmd/job<ID>/ on the exec node, so BASH_SOURCE would
# point there instead of the real mysrc/ directory. SLURM sets
# SLURM_SUBMIT_DIR to wherever sbatch was actually invoked from.
cd "$SLURM_SUBMIT_DIR"

DATASET_CFG=config/config-piper03.yaml

for imu_source in safire loa safire_post; do
    echo "================================================================"
    echo "[$(date)] ${imu_source}: orthorectification"
    echo "================================================================"
    python runOrtho.py --config "${DATASET_CFG}" --imu-source "${imu_source}"

    echo "================================================================"
    echo "[$(date)] ${imu_source}: drift estimation"
    echo "================================================================"
    python estimate_imu_drift.py --config "${DATASET_CFG}" --imu-source "${imu_source}"
done

echo "[$(date)] done."
