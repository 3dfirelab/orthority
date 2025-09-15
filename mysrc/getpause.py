import numpy as np 
import cv2
import sys
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import rioxarray as xr 
import matplotlib.pyplot as plt 
from io import StringIO


import transformation

#################################################
def get_cam_loc_angle(rvec,tvec):
    
    rotM_cam = cv2.Rodrigues(rvec)[0]
    cameraPosition = np.array(-(np.matrix(rotM_cam).T * np.matrix(tvec)))
   
    tmp = transformation.euler_from_matrix(rotM_cam,'rzyz')
    cameraAngle = [180 + 180/3.14*tmp[1], -tmp[2]*180/3.14, 180+(tmp[0]*180/3.14)]
    
    return cameraPosition, cameraAngle




# Skip the first line with CRS info
dirin = '/home/paugam/Data/ATR42/as250018/visible/bas/img/'
file_path = dirin+"as250018_20250710_071751-7452.tif.points"

with open(file_path, 'r') as f:
    lines = f.readlines()

# Extract CRS from the header
crs_line = lines[0].strip()
crs_wkt = crs_line.replace("#CRS: ", "")

# Read the rest of the lines into a DataFrame
data = ''.join(lines[1:])  # Skip header
df = pd.read_csv(StringIO(data))

print(f"number of gcp: {len(df)}")

# Create GeoDataFrame for map coordinates (in EPSG:3857)
gdf_map = gpd.GeoDataFrame(
    df,
    geometry=[Point(xy) for xy in zip(df['mapX'], df['mapY'])],
    crs=crs_wkt  # Or just "EPSG:3857" if you prefer 
    )

file_path = dirin+"as250018_20250710_071751-7452.tif"
img = xr.open_rasterio(file_path)  # remove band dim if 1 band

demFile = '/home/paugam/Data/ATR42/as250018/dem/dem_as250018.tif'
dem = xr.open_rasterio(demFile, masked=True).squeeze()  # remove band dim if 1 band
dem = dem.rio.reproject(gdf_map.crs)

coords_df = gdf_map.geometry.apply(lambda p: {"x": p.x, "y": p.y})
sampled_vals = [dem.sel(x=pt["x"], y=pt["y"], method="nearest").item() for pt in coords_df]
gdf_map["dem"] = sampled_vals

gcps_world = gdf_map[['mapX','mapY','dem']].values
gcps_cam   = gdf_map[['sourceX','sourceY']].values

gcps_cam[:,1] *= -1
#plt.imshow(img.T, origin='lower')
img_rgb = np.transpose(np.array(img), (1, 2, 0))
plt.imshow(img_rgb)
plt.scatter(gcps_cam[:,0], gcps_cam[:,1])
plt.show()

mtx, dist = np.load('/home/paugam/Data/ATR42/CameraCalib/ATRvis_72mm_cameraMatrix_DistortionCoeff.npy', allow_pickle=True)

flag, rvec, tvec = cv2.solvePnP(gcps_world, gcps_cam, mtx, dist)

loc, angle =  get_cam_loc_angle(rvec,tvec)

