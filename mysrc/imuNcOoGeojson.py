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
import pdb 
import pandas as pd 
import math 

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

'''
def rpy_to_rotation_matrix(roll, pitch, yaw):
    """Generate a rotation matrix from RPY angles."""
    Rz_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    Ry_pitch = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    Rx_roll = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    return Rz_yaw @ Ry_pitch @ Rx_roll
'''

def rotation_matrix_to_opk(R):
    """Extract Omega, Phi, Kappa from a rotation matrix."""
    omega = np.arctan2(R[0, 2], R[2, 2])  # Rotation around X-axis
    phi = np.arcsin(-R[1, 2])               # Rotation around Y-axis
    kappa = -1*np.arctan2(R[1, 0], R[1, 1])  # Rotation around Z-axis
    return np.degrees(omega), np.degrees(phi), np.degrees(kappa)



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

#####################################
def imutogeojson( indir, outdir, imufile, indirimg, flightname, correction_xyz, correction_opk, frames=None, str_tag=''):
  


    with xr.open_dataset(indir+imufile) as imu:
        if frames is None:
            frames = sorted(glob.glob(indirimg+'*.tif'))


        crs = pyproj.CRS.from_epsg(32631)
        data_dict = {
                    "type": "FeatureCollection",
                    "world_crs":  crs.to_proj4(),
                    "features": []
                    }
        # Define the coordinate systems
        wgs84 = pyproj.CRS('EPSG:4326')  # WGS84 (lat/lon)
        # Initialize the transformer
        transformer     = pyproj.Transformer.from_crs(wgs84, crs, always_xy=True)
        transformer_inv = pyproj.Transformer.from_crs(crs, wgs84, always_xy=True)


        df = None #pd.DataFrame(columns=['Photo','X','Y','Z','Yaw','Pitch','Roll'])
        
        for iframe, frame in enumerate(frames):
           
            print('imu {:.1f} %-- '.format(iframe+1/len(frames)*100), os.path.basename(frame),  end='\r' )

            with rasterio.open(frame.replace('_masked','')) as src:
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

                # Example Usage
                roll, pitch, yaw = float(imu.ROLL[idx].data), float(imu.PITCH[idx].data), float(imu.THEAD[idx].data)  # Example input angles in degrees
                #print('------')
                #print( roll, pitch, yaw)
                
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
      
                lon, alt = transformer_inv.transform(*xyz[:2])
                alt = xyz[-1]

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
                
                
                #for csv file
                new_row = {'Photo': os.path.basename(frame), 
                           'X':lon, 
                           'Y':lat, 
                           'Z':alt, 
                           'Roll':roll, 
                           'Pitch':pitch,
                           'Yaw': yaw,
                           }
                new_row_df = pd.DataFrame([new_row])
                
                if df is None:
                    df = new_row_df
                else:
                    df = pd.concat([df, new_row_df], ignore_index=True)
                
        # Now, data_dict contains the added feature, and you can dump it to a JSON file
        with open('{:s}/{:s}_ext_param{:s}.geojson'.format(outdir,flightname,str_tag), 'w') as f:
            json.dump(data_dict, f, indent=4)

        
        df.to_csv('{:s}/{:s}_ext_param.csv'.format(outdir,flightname), index=False)

    
    return 

##################################
if __name__ == "__main__":
##################################
    indir = '/mnt/data/ATR42/as240051/'
    outdir = indir + 'io2/'
    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    indirimg = indir + 'img2/'
    flightname = 'as240051'
            
    correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    correction_opk = np.array([0.,-0.1,0]) # degree

    imutogeojson( indir, outdir, imufile, indirimg, flightname, correction_xyz, correction_opk)
    

