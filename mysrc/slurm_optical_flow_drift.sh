#!/usr/bin/env bash
#SBATCH --job-name=optical-flow-drift
#SBATCH --partition=prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=optical-flow-drift-%j.out
#SBATCH --error=optical-flow-drift-%j.err

# Usage: sbatch slurm_optical_flow_drift.sh path/to/optical-flow-drift.yaml
set -euo pipefail

if [[ $# -ne 1 ]]; then
    echo "Usage: sbatch $0 path/to/optical-flow-drift.yaml" >&2
    exit 2
fi

# Slurm stages this batch file under /var/spool/slurmd, so BASH_SOURCE
# does not identify the workspace.  SLURM_SUBMIT_DIR does.
cd "${SLURM_SUBMIT_DIR:?submit this job with sbatch from the project directory}"

# Iteration 0: establish the uncorrected full-track ECC baseline before any
# optical-flow model is estimated or applied.
/home/paugam/miniforge3/bin/mamba run -n orthoriry python \
    apply_optical_flow_drift.py "$1" --iteration 0 --no-orthorectify --overwrite

# Iteration 1: estimate the optical-flow OPK model from the baseline orthos.
/home/paugam/miniforge3/bin/mamba run -n orthoriry python \
    optimize_optical_flow_drift.py "$1" --overwrite

# Apply the interpolated OPK model, re-orthorectify, then write comparable ECC
# diagnostics to output_dir/iteration_01/.
exec /home/paugam/miniforge3/bin/mamba run -n orthoriry python \
    apply_optical_flow_drift.py "$1" --iteration 1 --overwrite
