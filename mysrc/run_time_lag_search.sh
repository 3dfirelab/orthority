#!/bin/bash
#
#
python image_sequence_motion.py     --images-dir /data/shared/PIPER/az260007/Transects/bas00001/tif_f1                                                         \
                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc     \
                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00001/image_sequence_motion_atlans

python image_sequence_motion.py     --images-dir /data/shared/PIPER/az260007/Transects/bas00002/tif_f1                                                         \
                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc     \
                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00002/image_sequence_motion_atlans


python image_sequence_motion.py     --images-dir /data/shared/PIPER/az260007/Transects/bas00003/tif_f1                                                         \
                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc     \
                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00003/image_sequence_motion_atlans

python image_sequence_motion.py     --images-dir /data/shared/PIPER/az260007/Transects/bas00004/tif_f1                                                         \
                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc     \
                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00004/image_sequence_motion_atlans


cat /data/shared/PIPER/az260007/Transects/bas00001/image_sequence_motion_atlans/best_time_lag.txt
cat /data/shared/PIPER/az260007/Transects/bas00002/image_sequence_motion_atlans/best_time_lag.txt
cat /data/shared/PIPER/az260007/Transects/bas00003/image_sequence_motion_atlans/best_time_lag.txt
cat /data/shared/PIPER/az260007/Transects/bas00004/image_sequence_motion_atlans/best_time_lag.txt
