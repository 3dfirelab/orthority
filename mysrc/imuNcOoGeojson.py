import numpy as np 
import xarray as xr 
import rasterio
import sys
import os
import json
import glob 
import datetime 
import pyproj
import warnings
from rasterio.errors import NotGeoreferencedWarning
warnings.simplefilter('ignore', NotGeoreferencedWarning)

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

#####################################
def imutogeojson( indir, outdir, imufile, indirimg, flightname, correction_xyz, correction_opk):
    
    imu = xr.open_dataset(indir+imufile)
    frames = sorted(glob.glob(indirimg+'*.tif'))


    crs = pyproj.CRS.from_epsg(27563)
    data_dict = {
                "type": "FeatureCollection",
                "world_crs":  crs.to_proj4(),
                "features": []
                }
    # Define the coordinate systems
    wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
    utm = pyproj.CRS('EPSG:27563')   # EPSG:27563 (projected CRS)
    # Initialize the transformer
    transformer = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)


    for frame in frames:
        
        with rasterio.open(frame) as src:
            # Read metadata
            metadata = src.tags()
            
            # Check for time-related metadata (if available)
            time = datetime.datetime.strptime(metadata.get('TIFFTAG_DATETIME', 'Time not available'),
                                               "%Y:%m:%d %H:%M:%S")
            
            idx = np.abs(imu.time-np.datetime64(time)).argmin()
            #print(time, idx.data, end=' | ')

            #latlonZ
            lat = float(imu.LATITUDE[idx].data)
            lon = float(imu.LONGITUDE[idx].data)
            alt = float(imu.ALTITUDE[idx].data)
            #opk
            position = float(imu.PITCH[idx].data)
            orientation = float(imu.ROLL[idx].data)
            kinematics = (float(imu.THEAD[idx].data) - 180) % 360
            #xyz
            Ximu, Yimu = transformer.transform(lon, lat)
            Zimu = alt
    
            #correction in the arcraft referentiel
            #correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
            #correction_opk = np.array([-1.,3.,-16]) # degree
            
            #apply correction 
            xavion = -479.e-2    + correction_xyz[0]
            yavion = 24.5e-2   + correction_xyz[1]
            zavion = 20e-2      + correction_xyz[2]
            opk = np.array([orientation, position, kinematics])+correction_opk
            
            #transformation from 
            xyz = transform_point(xavion, yavion, zavion, Ximu, Yimu, Zimu, opk[0], opk[1], opk[2])
   

            #print('{:.1f}   {:.1f}  {:.1f}'.format(*xyz) + 
            #   '   {:.1f}   {:.1f}  {:.1f}'.format(*opk ))
            # Append a new feature to the dict
            append_to_dict(
                os.path.basename(frame).split('.ti')[0],
                list(xyz),  # Example xyz coordinates
                list(3.14/180*opk),  # Example opk # conversion to radian
                [lon,lat,alt],  # Example xyz coordinates
                data_dict
            )

    # Now, data_dict contains the added feature, and you can dump it to a JSON file
    with open('{:s}/{:s}_ext_param.geojson'.format(outdir,flightname), 'w') as f:
        json.dump(data_dict, f, indent=4)

##################################
if __name__ == "__main__":
##################################
    indir = '/home/paugam/Data/ATR42/as240051/'
    outdir = indir + 'io/'
    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    indirimg = indir + 'img/'
    flightname = 'as240051'
            
    correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    correction_opk = np.array([-1.,3.,-16]) # degree

    imutogeojson( indir, outdir, imufile, indirimg, flightname, correction_xyz, correction_opk)
    

