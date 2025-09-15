import rasterio
from rasterio.windows import from_bounds
from pyproj import Transformer
import xarray as xr
import numpy as np 
import os 

dirOut = '/home/paugam/Data/ATR42/as250026'
    
#ds = xr.open_dataset(f"{dirIn}/safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_CORE_NAV_200HZ_20250710_as250018_L1_V1.nc")
#lonmin = float(np.nanmin(ds["LONGITUDE"].values)) - 0.01
#lonmax = float(np.nanmax(ds["LONGITUDE"].values)) + 0.01
#latmin = float(np.nanmin(ds["LATITUDE"].values)) - 0.01
#latmax = float(np.nanmax(ds["LATITUDE"].values)) + 0.01

lonmin = 2.8 
lonmax = 3.1
latmin = 42.8
latmax = 43.1


print(lonmin, latmin, lonmax, latmax)
vrt = f"/home/paugam/Data/RGEALTI_IGN/RGEALTI_2-0_1M_ASC_LAMB93-IGN69_D011_2024-11-26/RGEALTI/3_SUPPLEMENTS_LIVRAISON_2024-12-00020/RGEALTI_MNT_1M_ASC_LAMB93_IGN69_D011_20241209/mosaique_POSIX_MNT.vrt"



with rasterio.open(vrt) as src:
    to2154 = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True).transform
    xmin, ymin = to2154(lonmin, latmin)
    xmax, ymax = to2154(lonmax, latmax)
    win = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
    subset = src.read(1, window=win, masked=True)
    subset_transform = src.window_transform(win)

    if subset.size == 0:
        raise ValueError("Subset is empty; check your bounds.")

    nodata = src.nodata if src.nodata is not None else -99999.0
    data_to_write = subset.filled(nodata)

    profile = src.profile.copy()
    profile.update(
        driver="GTiff",
        height=subset.shape[0],
        width=subset.shape[1],
        count=1,
        dtype=data_to_write.dtype,
        crs=src.crs,
        transform=subset_transform,
        nodata=nodata,
        tiled=True,
        compress="ZSTD",
        predictor=2,
        bigtiff="IF_SAFER",
    )

    os.makedirs(f"{dirOut}/dem/",exist_ok=True)
    out_path = f"{dirOut}/dem/{dirOut.split('/')[-1]}_dem_1m.tif"
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data_to_write, 1)

