
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
import geopandas as gpd 
import math
from shapely.geometry import Point
import matplotlib.pyplot as plt
import shutil 


if __name__ == '__main__':

    #dirInOut = '/home/paugam/Data/ATR42/as250018/safire'
    #filename = 'SILEX-2025_SAFIRE-ATR42_SAFIRE_CORE_NAV_200HZ_20250710_as250018_L1_V1.nc'
    dirInOut = '/home/paugam/Data/ATR42/as250026/safire'
    filename = 'SILEX-2025_SAFIRE-ATR42_SAFIRE_NAV_ATLANS_200HZ_20250726_as250026_L1_V1.nc'
    
    imu = xr.open_dataset(f'{dirInOut}/{filename}')

    # Convert xarray Dataset to pandas DataFrame
    df = imu[["LATITUDE", "LONGITUDE", "ALTITUDE", "THEAD", "ROLL", "PITCH"]].to_dataframe().reset_index()

    # Create geometry column (POINT from lat/lon)
    geometry = gpd.points_from_xy(df["LONGITUDE"], df["LATITUDE"], crs="EPSG:4326")

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry)

    gdf = gdf.sort_values("time")

    for var in ['ROLL','PITCH','THEAD']:
        # Apply running average over 0.3 seconds
        gdf[f"{var}_smooth"] = (
            gdf.rolling("300ms", on="time", center=True)[var].mean()
            )
    
    print(gdf[["time", "ROLL", "ROLL_smooth"]].head(10)) 
    
    gdf_no_geom = gdf.drop(columns="geometry")

    # Ensure time is datetime type
    gdf_no_geom["time"] = pd.to_datetime(gdf_no_geom["time"])

    # Convert to xarray.Dataset
    ds_out = gdf_no_geom.set_index("time").to_xarray()

    # Save to NetCDF
    out_file = filename.replace('.nc','_smooth.nc')
    ds_out.to_netcdf(out_file)

    shutil.move(filename.replace('.nc','_smooth.nc'), f"{dirInOut}/{filename.replace('.nc','_smooth.nc')}")
    

    #ax = plt.subplot(111)
    #gdf.ROLL.plot(ax=ax)
    #gdf.ROLL_smooth.plot(ax=ax)
    #plt.show()
