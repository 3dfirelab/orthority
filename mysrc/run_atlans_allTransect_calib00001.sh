#!/usr/bin/env bash
#SBATCH --partition=prod
#SBATCH --cpus-per-task=20
#SBATCH --mem=20G
#SBATCH --job-name=atlans_az260007

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is not set}"

./run_transect.sh \
    --flightname az260007 \
    --flight-date 20260910 \
    --transect-prefix bas \
    --transect-number 1 \
    --imu-name atlans \
    --imu-file /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc\
    --note "smooth flight" \
    --calib-transect 1 \
    --overwrite true 


./run_transect.sh \
    --flightname az260007 \
    --flight-date 20260910 \
    --transect-prefix bas \
    --transect-number 2 \
    --imu-name atlans \
    --imu-file /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc\
    --note "smooth flight" \
    --calib-transect 1 \
    --calib-file /data/shared/PIPER/az260007/Transects/bas00001-atlans-calib00001/calib/imu_camera_calibration_atlans.json\
    --overwrite true   


./run_transect.sh \
    --flightname az260007 \
    --flight-date 20260910 \
    --transect-prefix bas \
    --transect-number 3 \
    --imu-name atlans \
    --imu-file /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc\
    --note "smooth flight" \
    --calib-transect 1 \
    --calib-file /data/shared/PIPER/az260007/Transects/bas00001-atlans-calib00001/calib/imu_camera_calibration_atlans.json\
    --overwrite true   


./run_transect.sh \
    --flightname az260007 \
    --flight-date 20260910 \
    --transect-prefix bas \
    --transect-number 4 \
    --imu-name atlans \
    --imu-file /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc\
    --note "turbulent flight flight" \
    --calib-transect 1 \
    --calib-file /data/shared/PIPER/az260007/Transects/bas00001-atlans-calib00001/calib/imu_camera_calibration_atlans.json\
    --overwrite true   
