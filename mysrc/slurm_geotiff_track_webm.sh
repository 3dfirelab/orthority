#!/usr/bin/env bash
#SBATCH --job-name=geotiff-track-webm
#SBATCH --partition=prod
#SBATCH --cpus-per-task=8
#SBATCH --mem=12G
#SBATCH --time=02:00:00
#SBATCH --output=geotiff-track-webm-%j.out
#SBATCH --error=geotiff-track-webm-%j.err

# Usage: sbatch slurm_geotiff_track_webm.sh /path/to/tif_dir /path/to/config.yaml
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: sbatch $0 /path/to/tif_dir /path/to/config.yaml" >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec /home/paugam/miniforge3/bin/mamba run -n tracking python \
    "${script_dir}/geotiff_dir_to_webm.py" "$1" "$2" --overwrite
