#!/bin/bash
#

source ~/miniforge3/bin/activate orthoriry

#python crop_tif_f1_320.py \
#    /data/shared/PIPER/az260007/Transects/bas00001/tif_f1 \
#    --output-dir /data/shared/PIPER/az260007/Transects/bas00001/tif_f1_320 \
#    --overwrite


#python create_transect_dem.py \
#    /data/shared/PIPER/az260007/Transects/bas00001 \
#    --dem-root /data/shared/RGEALTI_IGN \
#    --image-dir /data/shared/PIPER/az260007/Transects/bas00001/tif_f1 \
#    --imu-dir /data/shared/PIPER/az260007/imu \
#    --buffer-km 5

#python calibrate_orthority_imu.py     config/calibration-az260007-bas00001-airins.yaml
#python calibrate_orthority_imu.py     config/calibration-az260007-bas00001-atlans.yaml
#python calibrate_orthority_imu.py     config/calibration-az260007-bas00001-atlans_p.yaml

#python runOrtho.py     --config config/config-az260007-bas00001-airins.yaml
#python runOrtho.py     --config config/config-az260007-bas00001-atlans.yaml
#python runOrtho.py     --config config/config-az260007-bas00001-atlans_p.yaml


#python correlate_ortho_sequence.py  /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_airins     \
#                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_AIRINS_100HZ_20260910_az260007_L1_V1.nc \
#                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00001/airins_correlation_timeseries

#python correlate_ortho_sequence.py  /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans     \
#                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_200HZ_20260910_az260007_L1_V1.nc \
#                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00001/atlans_correlation_timeseries

#python correlate_ortho_sequence.py  /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans_p     \
#                                    --imu /data/shared/PIPER/az260007/imu/TEST-ATLANS-2026_SAFIRE-PA23_SAFIRE_CORE_ATLANS_POSTPROCESSING_200HZ_20260910_az260007_L2_V1.nc \
#                                    --output-dir /data/shared/PIPER/az260007/Transects/bas00001/atlans_p_correlation_timeseries


#python geotiff_dir_to_webm.py     /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_airins                                              \
#                                  config/config-az260007-bas00001-airins.yaml                                                                      \
#                                  --output /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_airins/full_ortho_f1_track_airins.webm     \
#                                  --overwrite

#python geotiff_dir_to_webm.py     /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans                                              \
#                                  config/config-az260007-bas00001-atlans.yaml                                                                      \
#                                  --output /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans/full_ortho_f1_track_atlans.webm     \
#                                  --overwrite

#python geotiff_dir_to_webm.py     /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans_p                                             \
#                                  config/config-az260007-bas00001-atlans_p.yaml                                                                     \
#                                  --output /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_atlans_p/full_ortho_f1_track_atlans_p.webm  \
#                                  --overwrite


export simu_name=airins
mkdir -p /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_$simu_name/full_ortho_f1_track_"$simu_name".webm                                       /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_$simu_name/full_ortho_f1_track_"$simu_name".manifest.json                              /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/"$simu_name"_correlation_timeseries/ortho_correlation_timeseries_AIRINS.csv                         /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/

export simu_name=atlans
mkdir -p /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_$simu_name/full_ortho_f1_track_"$simu_name".webm                                       /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_$simu_name/full_ortho_f1_track_"$simu_name".manifest.json                              /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/"$simu_name"_correlation_timeseries/ortho_correlation_timeseries_ATLANS.csv                         /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/

export simu_name=atlans_p
mkdir -p /data/shared/PIPER/website/data/az2600007-bas00001-"$simu_name"
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_"$simu_name"/full_ortho_f1_track_"$simu_name".webm                                       /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/full_ortho_f1_"$simu_name"/full_ortho_f1_track_"$simu_name".manifest.json                              /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
cp /data/shared/PIPER/az260007/Transects/bas00001/"$simu_name"_correlation_timeseries/ortho_correlation_timeseries_ATLANS.csv                         /data/shared/PIPER/website/data/az2600007-bas00001-$simu_name/
