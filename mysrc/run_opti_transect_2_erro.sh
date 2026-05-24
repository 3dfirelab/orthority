source ~/miniforge3/bin/activate orthoriry

python optimizeAlignement_telops_on_transect.py --imufile_name safire --transectName 001_320256
python optimizeAlignement_telops_on_transect.py --imufile_name loa --transectName 001_320256

ln -fs resbrute1_xycopk_minimize4_001_320256_safire.npy resbrute1_xycopk_minimize4_003_320256_safire.npy
ln -fs resbrute1_xycopk_minimize4_001_320256_loa.npy resbrute1_xycopk_minimize4_003_320256_loa.npy

python runOrtho.py --imufile_name safire --minimizeID 4 --transectName 001_320256
python runOrtho.py --imufile_name loa --minimizeID 4 --transectName 001_320256
python runOrtho.py --imufile_name safire --minimizeID 4 --transectName 003_320256
python runOrtho.py --imufile_name loa --minimizeID 4 --transectName 003_320256

python error_imu_sequence_telops.py --imufile_name safire --minimizeID 4  --transectName 003_320256
python error_imu_sequence_telops.py --imufile_name loa --minimizeID 4  --transectName 003_320256
