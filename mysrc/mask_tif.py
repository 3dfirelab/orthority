import rasterio
import numpy as np 
import matplotlib.pyplot as plt 
import glob
import sys
import os 

indir = '/mnt/data/ATR42/as240051/'
imgdirname = 'img_manualOrtho'

input_tiffs = sorted(glob.glob(indir+'/{:s}/*.tif'.format(imgdirname)))
nodata_value = np.nan

os.makedirs(indir+'{:s}_masked'.format(imgdirname), exist_ok=True)

for input_tiff in input_tiffs:
    print(input_tiff)
    with rasterio.open(input_tiff) as src:
        profile = src.profile
        data = src.read().astype(np.float32)  # Read all bands into a 3D array

        # Apply a mask to all bands
        data[:,:32,:252] = nodata_value

        profile.update(nodata=nodata_value, dtype=rasterio.float32)


    output_tiff = input_tiff.replace('/{:s}/'.format(imgdirname),'/{:s}_masked/'.format(imgdirname))
    # Write the modified raster
    with rasterio.open(output_tiff, "w", **profile) as dst:
        dst.write(data)
    

