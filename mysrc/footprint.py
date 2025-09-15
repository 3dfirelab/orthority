import numpy as np 
import cv2
import xarray as xr 
import matplotlib.pyplot as plt
import subprocess
import os
from scipy import optimize
import warnings
import pdb 
import pandas as pd
import geopandas as gpd
import shutil
import importlib
import orthority as oty
import glob 
import pyproj
from PIL import Image
import json 
import sys
import rioxarray
from shapely.geometry import box
from shapely.geometry import shape
from rasterio.features import shapes

###################################################
def transform_point(x, y, z, X_IMU, Y_IMU, Z_IMU, o, p, k):
    # Rotation matrices
    R_roll = np.array([
        [1, 0, 0],
        [0, np.cos(o), -np.sin(o)],
        [0, np.sin(o), np.cos(o)]
    ])

    R_pitch = np.array([
        [np.cos(p), 0, np.sin(p)],
        [0, 1, 0],
        [-np.sin(p), 0, np.cos(p)]
    ])

    R_yaw = np.array([
        [np.cos(k), -np.sin(k), 0],
        [np.sin(k), np.cos(k), 0],
        [0, 0, 1]
    ])

    # Combined rotation matrix
    R = R_yaw @ R_pitch @ R_roll

    # Local to global transformation
    local_point = np.array([x, y, z])
    global_point = R @ local_point + np.array([X_IMU, Y_IMU, Z_IMU])

    return global_point


############################################
def rotation_matrix_to_opk(R):
    """Extract Omega, Phi, Kappa from a rotation matrix."""
    omega = np.arctan2(R[0, 2], R[2, 2])  # Rotation around X-axis
    phi = np.arcsin(-R[1, 2])               # Rotation around Y-axis
    kappa = -1*np.arctan2(R[1, 0], R[1, 1])  # Rotation around Z-axis
    return np.degrees(omega), np.degrees(phi), np.degrees(kappa)


############################################
def rpy_to_rotation_matrix(roll, pitch, yaw):
    """
    Converts euler angles to a rotation matrix
    """
    roll, pitch, yaw = np.radians(roll), np.radians(pitch), np.radians(yaw)
    size = (3,3)
    RX = np.zeros(size)
    RY = np.zeros(size)
    RZ = np.zeros(size)
    
    # convert from array to matrix
    RX = np.matrix(RX)
    RY = np.matrix(RY)
    RZ = np.matrix(RZ)
    
    c = np.cos(yaw)
    s = np.sin(yaw)
    RZ[0,0] = c
    RZ[0,1] = -s
    RZ[1,0] = s
    RZ[1,1] = c
    RZ[2,2] = 1

    c = np.cos(pitch)
    s = np.sin(pitch)
    RY[0,0] = c
    RY[0,2] = s
    RY[2,0] = -s
    RY[2,2] = c
    RY[1,1] = 1
    
    c = np.cos(roll)
    s = np.sin(roll)
    RX[1,1] = c 
    RX[1,2] = -s
    RX[2,1] = s
    RX[2,2] = c
    RX[0,0] = 1

    # combine to final rotation matrix
    return RZ@RY@RX

##################################
def append_to_dict(file_name, xyz, opk, latlonalt, data_dict):
    # Create the feature data for the file
    feature = {
        "type": "Feature",
        "properties": {
            "filename": file_name,
            "camera": "Integraph DMC",
            "xyz": xyz,  # Example: [-55094.50448, -3727407.03748, 5258.30793]
            "opk": opk  # Example: [-0.006094969000644519, 0.005209528564522755, -3.1256525964379143]
        },
        "geometry": {
            "type": "Point",
            "coordinates": latlonalt  # Only the first two values for latitude/longitude
        }
    }
    
    # Append the feature to the dictionary
    data_dict["features"].append(feature)


##################################################
def imu2ext_param(lat,lon,alt,roll,pitch,yaw, correction_xyz,correction_opk, crs_code):
    
    crs = pyproj.CRS.from_epsg(crs_code)
    data_dict = {
                "type": "FeatureCollection",
                "world_crs":  crs.to_proj4(),
                "features": []
                }
    
    R = rpy_to_rotation_matrix(roll, pitch, yaw)
    omega, phi, kappa = rotation_matrix_to_opk(R)
    #print( omega, phi, kappa)
   
    position = phi
    orientation = omega
    #kinematics = (float(imu.THEAD[idx].data) - 180) % 360
    kinematics = kappa
    #xyz
    Ximu, Yimu = transformer.transform(lon, lat)
    Zimu = alt
    
    #apply correction 
    xavion = -479.e-2    + correction_xyz[0]
    yavion = 24.5e-2   + correction_xyz[1]
    zavion = 20e-2      + correction_xyz[2]
    opk = np.array([orientation, position, kinematics])+correction_opk
    
    #transformation from 
    xyz = transform_point(xavion, yavion, zavion, Ximu, Yimu, Zimu, opk[0], opk[1], opk[2])

    lon, alt = transformer_inv.transform(*xyz[:2])
    alt = xyz[-1]

    #print('{:.1f}   {:.1f}  {:.1f}'.format(*xyz) + 
    #   '   {:.1f}   {:.1f}  {:.1f}'.format(*opk ))
    # Append a new feature to the dict
    append_to_dict(
        'now_visible_atr42',
        list(xyz),  # Example xyz coordinates
        list(3.14/180*opk),  # Example opk # conversion to radian
        [lon,lat,alt],  # Example xyz coordinates
        data_dict
    )

    return data_dict


#################################################
def orthro(tiffile, time, lat,lon,alt,roll,pitch,yaw,correction_xyz, correction_opk ):
   
    ext_param = imu2ext_param(lat,lon,alt,roll,pitch,yaw, correction_xyz,correction_opk, crs_code)
    str_tag = 'mm'
    extparamFile =  "{:s}/mm_ext_param{:s}.geojson".format(wkdir,str_tag)
    with open(extparamFile, 'w') as f:
        json.dump(ext_param, f, indent=4)
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
    
    demFile =  '{:s}/dem/dem.tif'.format(indir)
    #create a camera model for src_file from interior & exterior parameters
    cameras = oty.FrameCameras(intparamFile, extparamFile)

    #for idimg in idimgs[:1]:
    src_files =  [tiffile] 
    for src_file in src_files:
        #print(os.path.basename(src_file))
        camera = cameras.get(src_file)
        # create Ortho object and orthorectify
        ortho = oty.Ortho(src_file, demFile, camera=camera, crs=cameras.crs)
        ortho.process(wkdir+os.path.basename(src_file), overwrite=True)
        del ortho, camera
    del cameras
    os.remove(extparamFile)

    
    gdf = tiff_bounds_to_gdf(wkdir+os.path.basename(src_file), time)

    os.remove(wkdir+os.path.basename(src_file))

    return gdf


#############################
def tiff_bounds_to_gdf(tiff_path, time):
    '''
    # Open TIFF with rioxarray
    da = rioxarray.open_rasterio(tiff_path, masked=True)
    
    # Get bounds (left, bottom, right, top)
    bounds = da.rio.bounds()

    # Create a shapely box polygon from bounds
    geom = box(*bounds)
    '''
    # Open TIFF with rioxarray
    da = rioxarray.open_rasterio(tiff_path, masked=True).squeeze()  # remove extra dims

    # Create mask where data == 1
    mask = (da.data == 1)

    # Extract shapes (vectorize)
    results = shapes(da.data, mask=mask, transform=da.rio.transform())

    # Convert to geometries
    geoms = [shape(geom) for geom, value in results if value == 1]
    
 
    # Build GeoDataFrame
    gdf = gpd.GeoDataFrame(
        [{"geometry": geom, "time": time} for geom in geoms],
        crs=da.rio.crs
    )
    
    return gdf



#############################
def zero_out_image_and_update_time(filepath, mytime, output_path=None):
    # Open original image to get size and mode
    with Image.open(filepath) as img:
        width, height = img.size
        mode = img.mode  # should be 'RGB'

    # Create black (zero) image
    zero_array = np.ones((height, width, 3), dtype=np.uint8)  # 3 channels for RGB
    zero_image = Image.fromarray(zero_array, mode='RGB')

    # Set output path
    if output_path is None:
        output_path = filepath  # overwrite

    # Save image with same metadata format (TIFF)
    zero_image.save(output_path, format='TIFF')

    # Update the ModifyDate tag using exiftool
    subprocess.run([
        "exiftool",
        f"-ModifyDate={mytime}",
        "-overwrite_original",
        output_path
    ], check=True)


##################################
if __name__ == "__main__":
##################################
    

    indir = '/home/paugam/Data/ATR42/as240051/'
    #indirimg = indir + 'img/'
    #outdir   = indir + 'ortho/'
    #if os.path.isdir(outdir):
    #    shutil.rmtree(outdir)
    #os.makedirs(outdir, exist_ok=True)


    crs_code = 32631

    wkdir = '/tmp/paugam/footprint_wkdir/'
    if os.path.isdir(wkdir): shutil.rmtree(wkdir)
    os.makedirs(wkdir, exist_ok=True)

    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    flightname = 'as240051'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")

    intparamFile = "{:s}/io/as240051_int_param.yaml".format(indir)
     
    correction_xyz = [-4.54166272e-05,  1.46991992e-04,  3.92582905e-04]
    correction_opk = [ -4.62240706e-01,2.50020186e+00,  1.76677744e-04]
        
    crs_code=32631
    crs = pyproj.CRS.from_epsg(crs_code)
    # Define the coordinate systems
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
    # Initialize the transformer
    transformer     = pyproj.Transformer.from_crs(wgs84, crs, always_xy=True)
    transformer_inv = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True)
    
    with xr.open_dataset(indir+imufile) as imu:
        dfimu = imu.to_dataframe()

    gdf_footprint = None
    it = -1
    time_prev = None
    for _, row in dfimu.iterrows():
        it +=1 
        if row.ALTITUDE < 200: continue 
        if time_prev is not None: 
            if (row.time_bnds - time_prev).seconds < 30: continue 

        print(it,  row.time_bnds )
        file_path = "../data_static/template_atr42_visible.tif"
        new_time = row.time_bnds.strftime("%Y-%m-%d %H:%M:%S")  # ExifTool format: YYYY:MM:DD HH:MM:SS
        row_dummy_file = wkdir+'now_visible_atr42.tif'
        zero_out_image_and_update_time(file_path, new_time, row_dummy_file)
        
        lat = float(row.LATITUDE)
        lon = float(row.LONGITUDE)
        alt = float(row.ALTITUDE)
        roll  = float(row.ROLL)
        pitch = float(row.PITCH)
        yaw   = float(row.THEAD)  # Example input angles in degrees
    
        gdf_ = orthro(row_dummy_file, row.time_bnds, lat,lon,alt,roll,pitch,yaw, correction_xyz,correction_opk)
        
        if gdf_footprint is None:
            gdf_footprint = gdf_
        else:
            gdf_footprint = pd.concat([gdf_,gdf_footprint])

        
        time_prev = row.time_bnds


    # Save to GeoJSON
    geojson_path = './footprint_as240051.geojson'
    gdf_footprint.to_file(geojson_path, driver="GeoJSON")

