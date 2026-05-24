source ~/miniforge3/bin/activate orthoriry

python optimizeAlignement_telops_against_manual.py --imufile_name safire --transectName 001_320256
python optimizeAlignement_telops_against_manual.py --imufile_name loa --transectName 001_320256

ln -fs resbrute1_xycopk_minimize3_001_320256_safire.npy resbrute1_xycopk_minimize3_003_320256_safire.npy
ln -fs resbrute1_xycopk_minimize3_001_320256_loa.npy resbrute1_xycopk_minimize3_003_320256_loa.npy

python runOrtho.py --imufile_name safire --minimizeID 3 --transectName 001_320256
python runOrtho.py --imufile_name loa --minimizeID 3 --transectName 001_320256
python runOrtho.py --imufile_name safire --minimizeID 3 --transectName 003_320256
python runOrtho.py --imufile_name loa --minimizeID 3 --transectName 003_320256

python error_imu_sequence_telops.py --imufile_name safire --minimizeID 3  --transectName 003_320256
python error_imu_sequence_telops.py --imufile_name loa --minimizeID 3  --transectName 003_320256
