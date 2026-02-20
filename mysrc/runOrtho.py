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
import tifffile
import json 
from pathlib import Path
import tempfile
import rioxarray  # ensures .rio accessor is registered
from rasterio.enums import Resampling
from osgeo import gdal
import subprocess
import rasterio
from orthority.ortho import OrthorityWarning
import pandas as pd 


#homebrewed
import imuNcOoGeojson 
#import optimizeAlignement_telops_function 
#importlib.reload(optimizeAlignement_telops_function)
import optimizeAlignement_telops_f1


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


#########################################
def radiance_um_to_wavenumber(lambda_um, L_lambda):
    """
    Convert spectral radiance from per micrometre to per wavenumber.

    Parameters:
    ----------
    lambda_um : float or np.ndarray
        Wavelength(s) in micrometres (μm).
    L_lambda : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·μm⁻¹.

    Returns:
    -------
    L_wavenumber : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·(cm⁻¹)⁻¹.
    """
    lambda_um = np.asarray(lambda_um)
    L_lambda = np.asarray(L_lambda)
    return L_lambda * (lambda_um ** 2) / 1e4


def radiance_wavenumber_to_um(lambda_um, L_wavenumber):
    """
    Convert spectral radiance from per wavenumber to per micrometre.

    Parameters:
    ----------
    lambda_um : float or np.ndarray
        Wavelength(s) in micrometres (μm).
    L_wavenumber : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·(cm⁻¹)⁻¹.

    Returns:
    -------
    L_lambda : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·μm⁻¹.
    """
    lambda_um = np.asarray(lambda_um)
    L_wavenumber = np.asarray(L_wavenumber)
    return L_wavenumber * 1e4 / (lambda_um ** 2)


#########################################
# Define affine model
def affine(x, m, p):
    return m * x + p


#################################################
def orthro(args, transectname, flightname, flightdate):
   
    x, y, z = args[:3]
    o, p, k = args[3:]

    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([x,y,z])
    src_files =  sorted(glob.glob(f"{indirimg}/f{filtre}*.tif" ))


   
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
    
    imu = xr.open_dataset(indir+imufile)
    #for idimg in idimgs[:1]:
    src_files =  sorted(glob.glob(f"{indirimg}/f{filtre}*.tif" ))
    df_calib = pd.read_csv('/data/shared/ATR42/TelposDLCalib/SILEX_telops_filtre_DL_fit.csv')
    
    df_calib_f =    df_calib[df_calib.filtre == filtre]

    for src_file in src_files:
        if os.path.isfile(outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif')): continue
        
        print(os.path.basename(src_file))
        base = os.path.basename(src_file)       # "f1-000000001.tif"
        id_str = base.replace(f"f{filtre}-", "").replace(".tif", "")
        frame_id = int(id_str)    
    
        with tifffile.TiffFile(src_file) as tif:
            # Get ImageDescription tag
            description = tif.pages[0].tags["ImageDescription"].value
            # Convert JSON string to dictionary
            metadata = json.loads(description)
            # Access ExposureTime
            exposure_time = metadata["ExposureTime"]
            # Read image data
            data = tif.asarray()
        
        if filtre>=3:
            #convert DL to Radiance
            #-----------------
            rad_cm = affine( data/exposure_time, df_calib_f.m.values, df_calib_f.p.values)
            rad_lambda = radiance_wavenumber_to_um( df_calib_f['lambda'] , rad_cm)
            
            # sace to tmp before ortho
            #-----------------
            tif_path_ = f"{wkdir}/{base}".replace('.tif','_rad.tif')
            # Save as float32 TIFF with metadata
            tifffile.imwrite(
                tif_path_,
                rad_lambda.astype('float32'),
                metadata=metadata
            )
            src_file_ = tif_path_

        else: 
            data = data/exposure_time
            tif_path_ = f"{wkdir}/{base}".replace('.tif','_expcorr.tif')
            # Save as float32 TIFF with metadata
            tifffile.imwrite(
                tif_path_,
                data.astype('float32'),
                metadata=metadata
            )
            src_file_ = tif_path_

        if (frame_id >= 100) & (frame_id % 100 == 0 ):
            print('###############')
            print(src_file_)
            print('ref id:', frame_id-99) 
            correction_opk_copy = correction_opk.copy()
            oc, pc, kc =  ( np.array(correction_opk) + offset[1] ) / scale[1]
            result, params_ = optimizeAlignement_telops_f1.run_opt_correction(frame_id-99, src_file_, transectname, flightname, flightdate, oc, pc, kc)

            [oc,pc,kc] = result.x
            #xc,yc,zc = 0.5,0.5,0.5
            #correction_xyz = (np.array([xc,yc,zc]) * scale[0]) - offset[0]
            correction_opk = (np.array([oc,pc,kc]) * scale[1]) - offset[1]
            print('modif of correction:')
            print(np.array(correction_opk)-np.array(correction_opk_copy))
            print('###############')
            ##plot
            #params_['flag_plot'] = True
            #optimizeAlignement_telops_f1.residual( result.x , params_ )

        print(correction_opk)
        print(src_file_)
        print('process imu ...')
        imu = xr.open_dataset(indir+imufile)    
        imuNcOoGeojson.imutogeojson(imu, wkdir, indirimg, flightname, correction_xyz, correction_opk, [src_file_]) 
        print('done                ') 

        str_tag = ''
        extparamFile =  f"{wkdir}/{flightname}_ext_param{str_tag}.geojson".format(wkdir,str_tag)
        #create a camera model for src_file from interior & exterior parameters
        cameras = oty.FrameCameras(intparamFile, extparamFile)

        camera = cameras.get(src_file_)
        # create Ortho object and orthorectify
        ortho = oty.Ortho(src_file_, demFile, camera=camera, crs=cameras.crs)
        out_file_ = outdir+os.path.basename(src_file_).replace('.tif','_ORTHO.tif')
        ortho.process(out_file_, overwrite=True)
        del ortho, camera
        
        to_epsg4326_inplace( outdir+os.path.basename(src_file_).replace('.tif','_ORTHO.tif') )
       
        
        #remove_halo(         outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif') )
        
        #add meta data to ortho tif

        # read existing image and metadata
        with rasterio.open(out_file_, "r") as src:
            img = src.read()
            profile = src.profile
            existing_meta = src.tags()

        # merge metadata
        existing_meta.update(metadata)

        # write back preserving CRS and transform
        with rasterio.open(out_file_, "w", **profile) as dst:
            dst.write(img)
            dst.update_tags(**existing_meta)


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
    importlib.reload(optimizeAlignement_telops_f1)
    importlib.reload(imuNcOoGeojson)
    
    flightname = 'as250026'
    flightdate = '20250726'
    transectname = 'Sijean01'
    filtre=1
    
    indir = f'/data/shared/ATR42/as250026/Transects/{transectname}'
   
    indirimg = indir + f'/tif_f{filtre}/'
    outdir   = indir + f'/full_ortho_f{filtre}/'
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    
    wkdir = '/tmp/paugam/orthority_wkdir/'
    if os.path.isdir(wkdir): shutil.rmtree(wkdir)
    os.makedirs(wkdir, exist_ok=True)

    imufile = '/../../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_NAV_ATLANS_200HZ_20250726_as250026_L1_V1_smooth.nc'
    
    
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")
    warnings.filterwarnings("ignore", category=OrthorityWarning)  # show once per message

    demFile =  '{:s}/../../dem/as250026_dem_1m.tif'.format(indir)
    
    intparamFile = f"{indir}/io/{flightname}_int_param.yaml"
    offset = [ np.array([.5,.5,.5]),    np.array([5,5,5]) ]
    scale = [ np.array([1,1,1]), np.array([10,10,10,]) ]   
    
    oc,pc,kc = np.load(f'resbrute1_xycopk_minimize2_{transectname}.npy',allow_pickle=True).item().x 
    
    xc,yc,zc = 0.5,0.5,0.5
    
    correction_xyz = (np.array([xc,yc,zc]) * scale[0]) - offset[0]
    correction_opk = (np.array([oc,pc,kc]) * scale[1]) - offset[1]


    orthro([*correction_xyz,*correction_opk], transectname, flightname, flightdate)
    
