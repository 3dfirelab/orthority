import numpy as np 
import cv2
import xarray as xr 
import matplotlib.pyplot as plt
import subprocess
import os
from scipy import optimize
import warnings
import pdb 
import geopandas as gpd
import shutil
import importlib
import orthority as oty
import glob 

from pathlib import Path
import tempfile
import rioxarray  # ensures .rio accessor is registered
from rasterio.enums import Resampling
from osgeo import gdal
import subprocess
import rasterio

#homebrewed
import imuNcOoGeojson 
importlib.reload(imuNcOoGeojson)


#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    im = np.array(im,dtype=np.float32)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    
    mag, angle = cv2.cartToPolar(grad_x,grad_y)
    return grad, mag, angle

#################################################
def orthro(args):
   
    x, y, z = args[:3]
    o, p, k = args[3:]

    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([x,y,z])
    src_files =  sorted(glob.glob(f"{indirimg}/f1*.tif" ))

    print('process imu ...')
    imu = xr.open_dataset(indir+imufile)    
    imuNcOoGeojson.imutogeojson(imu, wkdir, indirimg, flightname, correction_xyz, correction_opk, src_files) 
    print('done                ') 

   
    '''
    command = [
            "oty", "frame",
            "--dem", '{:s}/dem/dem.tif'.format(indir),
            "--int-param", "{:s}/as240051_int_param.yaml".format(outdirIO),
            "--ext-param", "{:s}/as240051_ext_param.geojson".format(outdirIO),
            "--out-dir", outdir,
            "-o", 
            "/{:s}as240051_20241113_103254-*.tif".format(indirimg), 
            ]
    print(' '.join(command))
    #run orthorectification
    result = subprocess.run(command, capture_output=True, text=True)
    '''
    
    str_tag = ''
    extparamFile =  f"{wkdir}/{flightname}_ext_param{str_tag}.geojson".format(wkdir,str_tag)
    #create a camera model for src_file from interior & exterior parameters
    cameras = oty.FrameCameras(intparamFile, extparamFile)

    #for idimg in idimgs[:1]:
    src_files =  sorted(glob.glob(f"{indirimg}/f1*.tif" ))
    
    for src_file in src_files:
        print(os.path.basename(src_file))
        camera = cameras.get(src_file)
        # create Ortho object and orthorectify
        ortho = oty.Ortho(src_file, demFile, camera=camera, crs=cameras.crs)
        ortho.process(outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif'), overwrite=True)
        del ortho, camera
        
        to_epsg4326_inplace( outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif') )
        
        #remove_halo(         outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif') )
        
    
    del cameras

    return 'done'


####################
import os
import numpy as np
import rasterio

def remove_halo(input_path, white_thresh=220, chroma_thresh=8):
    """
    Detect near-white, low-chroma 'halo' pixels and store them in the dataset mask.
    Overwrites the GeoTIFF in place (atomic replace via temp file).

    Parameters
    ----------
    input_path : str
        Path to a GeoTIFF (expects >=3 bands interpreted as RGB).
    white_thresh : int
        Threshold (0–255) per-channel to be considered 'white-ish'.
    chroma_thresh : int
        Max channel spread to be considered low chroma (greyish).
    """
    tmp_path = input_path + ".tmp"

    with rasterio.open(input_path) as src:
        arr = src.read()            # (bands, H, W)
        profile = src.profile

    if arr.shape[0] < 3:
        raise ValueError("Expected at least 3 bands (RGB).")

    # Compute background mask from first three bands
    R, G, B = arr[:3].astype(np.uint16)
    #near_white = (R > white_thresh) & (G > white_thresh) & (B > white_thresh)
    alpha = arr[3]
    mm = alpha < white_thresh
    low_chroma = (np.maximum.reduce([R, G, B]) - np.minimum.reduce([R, G, B]) < chroma_thresh)
    #bg = near_white & low_chroma  # True = halo/background to hide
    bg = mm #& low_chroma  # True = halo/background to hide

    # Create 8-bit dataset mask: 0 = masked (transparent), 255 = valid
    ds_mask = np.where(bg, 0, 255).astype(np.uint8)

    arr[:,bg]=0
    # Write a new file with original bands and the computed mask
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(arr)                 # write all original bands
        #dst.write_mask(ds_mask)        # set dataset mask

    os.replace(tmp_path, input_path)

##################################
def to_epsg4326_inplace(path, compress="LZW"):
    
    # Reproject to EPSG:4326
    gdal.Warp(
        path,
        path,
        dstSRS='EPSG:4326',
        resampleAlg='near',
        srcNodata=0,      # or the actual nodata in your file
        dstNodata=0
        )

    return None

##################################
if __name__ == "__main__":
##################################
    #indir = '/mnt/dataEstrella2/SILEX/ATR42/as240051/'
    #indir = '/home/paugam/Data/ATR42/as250018/visible/bas/'
    #indir = '/home/paugam/Data/ATR42/as250018/imgRef/'
    transectname = 'Sijean10'
    indir = f'/home/paugam/Data/ATR42/as250026/{transectname}/'
    
    indirimg = indir + 'tif_f1/'
    outdir   = indir + 'ortho/'
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    
    wkdir = '/tmp/paugam/orthority_wkdir/'
    #if os.path.isdir(wkdir): shutil.rmtree(wkdir)
    os.makedirs(wkdir, exist_ok=True)

    #imufile = '../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_CORE_NAV_200HZ_20250710_as250018_L1_V1_smooth.nc'
    imufile = '../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_NAV_ATLANS_200HZ_20250726_as250026_L1_V1_smooth.nc'
    
    #flightname = 'as250018'
    flightname = 'as250026'
    
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")
    #demFile =  '{:s}/../dem/dem_as250018.tif'.format(indir)
    demFile =  '{:s}/../dem/as250026_dem_1m.tif'.format(indir)
    #demFile =  '{:s}/../dem/as250018_dem_1m.tif'.format(indir)
    
    intparamFile = f"{indir}/io/{flightname}_int_param.yaml"
    #from popt = optimize.fmin(residual, tuple([-1, 3.5, -15]), args=tuple(params), xtol=5, ftol=1.e-4, disp=False
    #correction_opk = [-1.69183706,   3.06752315, -14.98225642] 
    #correction_xyz = [2.10512566e-04,  1.70885594e-04,  2.61422540e-04, ]
    #correction_opk = [-5.52140251e-01,3.34409679e+00, -1.62949875e+01]
    
    #correction_xyz = [ 2.42538810e-05, 2.04130933e-04,  4.58924207e-05]
    #correction_opk = [-8.88472742e-01, 5.01270986e-01, -3.32808844e-05]
     
    #correction_xyz = [-4.54166272e-05,  1.46991992e-04,  3.92582905e-04]
    #correction_opk = [ -4.62240706e-01,2.50020186e+00,  1.76677744e-04]
    offset = [ np.array([.5,.5,.5]),    np.array([5,5,5]) ]
    scale = [ np.array([1,1,1]), np.array([10,10,10,]) ]   
    
    #offset = [ np.array([479,24.5,20]),    np.array([3.5,3.5,3.5]) ]
    #scale = [ np.array([10,10,10]), np.array([7,7,7,]) ]
    #popt = np.load('resbrute1_xycopk_minimize.npy',allow_pickle=True)
    #xc,yc,zc,o,p,k = popt.item().x 
    #popt = np.load('resbrute1_xycopk_minimize.npy',allow_pickle=True)
    #xc,yc,zc,o,p,k = [0,0,0,0,0,0] #popt.item().x 
    oc,pc,kc = np.load(f'resbrute1_xycopk_minimize2_{transectname}.npy',allow_pickle=True).item().x 
    
    #xc,yc,zc,oc,pc,kc = 0.5,0.5,0.5,  0.5,0.5,0.5
    xc,yc,zc = 0.5,0.5,0.5
    
    correction_xyz = (np.array([xc,yc,zc]) * scale[0]) - offset[0]
    correction_opk = (np.array([oc,pc,kc]) * scale[1]) - offset[1]


    orthro([*correction_xyz,*correction_opk])
    
    #if os.path.isdir(wkdir): shutil.rmtree(wkdir)


