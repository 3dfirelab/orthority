import os
import sys
import glob
import warnings
import numpy as np 
import rioxarray as rxr
import xarray as xr
from rasterio.enums import Resampling
from shapely.geometry import box
import geopandas as gpd
from osgeo import gdal
import cv2


#################################################
def get_gradient(im) :
    # Calculate the x and y gradients using Sobel operator
    im = np.array(im,dtype=np.float32)
    grad_x = cv2.Sobel(im,cv2.CV_32F,1,0,ksize=3)
    grad_y = cv2.Sobel(im,cv2.CV_32F,0,1,ksize=3)
 
    # Combine the two gradients
    grad = cv2.addWeighted(np.absolute(grad_x), 0.5, np.absolute(grad_y), 0.5, 0)
    
    mag, angle = cv2.cartToPolar(grad_x,grad_y)
    return grad


#################################################
def img2da4residu(rrh, atr, da1Ref, threshold_fire=8000):
   
    #with  xr.open_dataset(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimgs_)) as atr: 
        #atr = atr.rio.reproject(daRef.rio.crs)
    
    da1 = atr#.band_data.isel(band=0)
    da1 = da1.coarsen(dim={'x': 2, 'y': 2}, boundary="trim").mean()
    
    #remove a buffer around the image
    buffer = 80
    mask = np.where(np.isnan(da1.values), 0, 1).astype(np.uint8) 
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones([buffer,buffer]))
    da1 = da1.where(mask==1)
    

    da1 = xr.DataArray(get_gradient(da1), dims=["y", "x"], coords={"y": da1.y, "x": da1.x})
    da1 = da1.interp(x=da1Ref.x, y=da1Ref.y)
   

    '''
    # Ensure CRS match
    gdf = gpd.read_file('/home/paugam/Data/ATR42/as250026/mask/mine.gpkg')
    gdf = gdf.to_crs(da1.rio.crs)
    # Step 1: Check intersection with da extent
    da1_bounds = da1.rio.bounds()  # (minx, miny, maxx, maxy)
    da1_box = gpd.GeoSeries([box(*da1_bounds)], crs=da1.rio.crs)
    gdf_inter = gdf[gdf.intersects(da1_box.geometry[0])]

    if not gdf_inter.empty:
        # Step 2: Create mask from polygons (True inside polygon)
        mask = features.geometry_mask(
            gdf_inter.geometry,
            out_shape=da1.rio.shape,
            transform=da1.rio.transform(),
            invert=True  # True inside polygons
        )

        # Step 3: Apply mask (mask INSIDE polygon -> invert again)
        da1_masked = da1.where(~mask)
    else:
        da1_masked = da1.copy()
    '''
    da1_masked = da1.copy()


    ''' 
    # Convert to Dask array and chunk it
    darr = da1_masked.chunk({'y': da1_masked.shape[0], 'x': da1_masked.shape[1]}).data

    # Apply map_overlap with buffer
    darr_eq = da.map_overlap(
        normalization.clahe_block_numpy,
        darr,
        depth=16,
        boundary='reflect',
         tile_grid_size=(11,11),   #(512+32(=depthx2))/32(=disksize) = 17 
        dtype=np.float32
    )
    
    # Wrap back in xarray
    da1 = xr.DataArray(darr_eq, dims=da1.dims, coords=da1.coords, attrs=da1.attrs)
    '''
   
    da1= da1.where(da1 <= threshold_fire) # mask fire

    #if np.nanpercentile(da1,80) > 1000:
    #    da1= da1.where(da1 <= np.nanpercentile(da1,80)) # this is to mask house or small structure with high temperature 

    return da1



BDORTHO_DIR = "/home/paugam/Data/BDORTHO_IGN/BDORTHO_2-0_RVB-0M20_JP2-E080_LAMB93_D011_2024-01-01/"
ATR_TIF     = "/home/paugam/Data/ATR42/as250026/Sijean06/ortho/f1-000000260_ORTHO.tif"
BD_VRT      = "/home/paugam/Data/BDORTHO_IGN/bdortho_2024.vrt"  # will be created if missing
#ALIGNED_OUT = "/home/paugam/Data/ATR42/as250026/Sijean06/ortho/f1-000000560_ORTHO_toBDORTHO.tif"


# 1) Build a VRT for BD ORTHO if you don't already have one
if not os.path.exists(BD_VRT):
    jp2_list = sorted(glob.glob(os.path.join(BDORTHO_DIR, "**", "*.jp2"), recursive=True))
    if not jp2_list:
        raise FileNotFoundError("No .jp2 files found under BDORTHO_DIR; check the path.")
    vrt_opts = gdal.BuildVRTOptions(resolution="highest", addAlpha=True)
    gdal.BuildVRT(BD_VRT, jp2_list, options=vrt_opts)

# 2) Open BD ORTHO as a DataArray (this is the target grid)
#    rioxarray.open_rasterio returns: dims=("band","y","x") for multiband rasters
bd = rxr.open_rasterio(BD_VRT, masked=True, chunks={"x": 4096, "y": 4096})

# 3) Open your ATR image as a DataArray ("make it a da")
atr = rxr.open_rasterio(ATR_TIF, masked=True)  # dims=("band","y","x") or ("y","x") if single-band
# If single-band and you prefer ("y","x"), you can squeeze:
if "band" in atr.dims and atr.sizes.get("band", 1) == 1:
    atr = atr.squeeze("band", drop=True)  # now ("y","x"); metadata still attached via .rio

atr = atr.rio.reproject(
    bd.rio.crs,
    resampling=Resampling.bilinear,  # use nearest for categorical data
)


# 4) (optional but recommended) Limit processing to the spatial intersection to save time
#    Compute an intersection bbox in the BD ORTHO CRS
#try:
# ensure both have CRS written
atr_crs = atr.rio.crs
if atr_crs is None:
    raise ValueError("ATR image has no CRS. Set it with atr.rio.write_crs(EPSG_CODE) if known.")
bd_bounds = bd.rio.bounds()
# reproject ATR bounds to BD CRS and intersect

# bounding boxes
atr_bounds = atr.rio.bounds()
bd_bounds  = bd.rio.bounds()

# ✅ GeoDataFrame, not GeoSeries
atr_poly = gpd.GeoDataFrame(geometry=[box(*atr_bounds)], crs=atr.rio.crs)
bd_poly  = gpd.GeoDataFrame(geometry=[box(*bd_bounds)],  crs=bd.rio.crs)

# reproject ATR polygon to BD CRS
atr_in_bd = atr_poly.to_crs(bd.rio.crs)

# intersection as GeoDataFrame
inter = gpd.overlay(atr_in_bd, bd_poly, how="intersection")

if not inter.empty:
    minx, miny, maxx, maxy = inter.geometry.iloc[0].bounds
    # Clip BD ORTHO to intersection before matching (faster)
    bd_clip = bd.rio.clip_box(minx=minx, miny=miny, maxx=maxx, maxy=maxy)
else:
    warnings.warn("No spatial intersection found; proceeding with full extent reproject.")
    bd_clip = bd
#except Exception as e:
#    warnings.warn(f"Skipping intersection clip: {e}")
#    bd_clip = bd
bd_clip_1m = bd_clip.rio.reproject(
    dst_crs=bd_clip.rio.crs,
    resolution=1.0,   # 1 metre
    resampling=Resampling.average  # for downsampling imagery
)
# bd_clip_1m : (band, y, x) with bands [R,G,B,(A)]
# 1) pull RGB
R = bd_clip_1m.sel(band=1)
G = bd_clip_1m.sel(band=2)
B = bd_clip_1m.sel(band=3)

# 2) grayscale (BT.601 luma)
bd_clip_1m = (0.2989 * R + 0.5870 * G + 0.1140 * B).astype("float32")

# 3) (optional) apply alpha as mask if band 4 exists (assumes 0=transparent, 255=opaque)
if 4 in bd_clip_1m.coords["band"].values:
    A = bd_clip_1m.sel(band=4)
    bd_clip_1m = xr.where(bd_clip_1m > 0, bd_clip_1m, np.nan)

# 4) carry CRS/transform; bd_clip_1m already has x/y coords and CRS via rioxarray
bd_clip_1m = bd_clip_1m.rio.write_crs(bd_clip_1m.rio.crs, inplace=False)

# 5) Align ATR to BD ORTHO grid
#    - reproject_match() copies BD grid (CRS, resolution, transform, shape)
#    - choose resampling as appropriate; bilinear (continuous) or nearest/cubic, etc.
resampling = Resampling.bilinear
atr_to_bd = atr.rio.reproject_match(bd_clip_1m, resampling=resampling)

rr=3
da1 = img2da4residu(rr,atr_to_bd,bd_clip_1m)



sys.exit()
# 6) Persist the aligned raster (GeoTIFF with tiling & compression)
atr_to_bd.rio.to_raster(
    ALIGNED_OUT,
    compress="deflate",
    tiled=True,
    blockxsize=512,
    blockysize=512,
    dtype=str(atr_to_bd.dtype)
)
print(f"Saved: {ALIGNED_OUT}")

# 7) If you also want a stacked, co-registered xarray object for analysis:
#    Ensure both have matching dims; convert BD to same band handling as ATR if needed
if "band" in bd_clip.dims and "band" not in atr_to_bd.dims:
    # expand ATR to have a fake band dim for stacking consistency
    atr_to_bd = atr_to_bd.expand_dims({"band":[1]})
stack = xr.align(bd_clip, atr_to_bd, join="exact")  # ensures identical coords
