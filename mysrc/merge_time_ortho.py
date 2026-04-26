from pathlib import Path
import re, math, json, datetime
import numpy as np
import xarray as xr
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from affine import Affine
import pdb 
import warnings
from rasterio.errors import NotGeoreferencedWarning
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import os 

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

#folder = Path("/home/paugam/Data/ATR42/as250026/Sijean06_short/ortho")
#out_path = folder / "f1_merged_Sijean06_short.tif"


def numeric_id(path: Path) -> int:
    m = re.search(r'-(\d+)_rad_ORTHO\.tif$', path.name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', path.name)
    return int(max(nums, key=len)) if nums else 10**12

def open_single_band(fp: Path):
    da = rxr.open_rasterio(fp, masked=True)
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    if da.rio.nodata is None:
        da = da.rio.write_nodata(default_nodata)
    return da

def get_time(fp: Path):
    """Extract datetime from TIFFTAG_IMAGEDESCRIPTION if present."""
    with rxr.open_rasterio(str(fp.absolute()).replace(f'/full_ortho_f{filterCam}/',f'/tif_f{filterCam}/').replace('_rad_ORTHO','')) as da:
        desc = da.attrs.get("TIFFTAG_IMAGEDESCRIPTION")
    if not desc:
        return None
    try:
        meta = json.loads(desc)
        time_str = meta.get("Time")
        if time_str:
            return pd.to_datetime(time_str) #datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return None
    return None


################################
if __name__ == '__main__':
################################

    transectname = 'Sijean01'
    filterCam=1

    indir = f'/data/shared/ATR42/as250026/Transects/{transectname}'
    
    folder = Path(f"{indir}/full_ortho_f{filterCam}")
    out_path = folder / f"f{filterCam}_merged_time_{transectname}.nc"
    if os.path.isfile(out_path): os.remove(out_path)
    default_nodata = 0

    df_calib = pd.read_csv('/data/shared/ATR42/TelposDLCalib/SILEX_telops_filtre_DL_fit.csv')
    df_calib_f =    df_calib[df_calib.filtre == filterCam]

    # --- collect and sort
    tifs = sorted(folder.glob(f"f{filterCam}-*.tif"), key=numeric_id)
    if not tifs:
        raise FileNotFoundError("No tifs found")

    #-- select only tif where rad_max > 100
    selected_tifs = []
    for fp in tifs:
        da = open_single_band(fp)
        if da.max() > 100: 
            selected_tifs.append(fp)

    print(f'{len(selected_tifs)} tif selected out of {len(tifs)}')
    tifs = selected_tifs

    # --- reference grid (first file)
    ref = open_single_band(tifs[0])
    first_time = get_time(tifs[0])
    crs = ref.rio.crs
    xres, yres = ref.rio.resolution()  # yres is negative
    nodata = ref.rio.nodata
    x0, y0 = ref.rio.transform().c, ref.rio.transform().f  # top-left origin

    # --- compute union bounds in ref CRS
    uminx, uminy, umaxx, umaxy = ref.rio.bounds()
    print('first loop to get the extent of the domain')
    for fp in tifs[1:]:
        #print(fp)
        da = open_single_band(fp)
        if da.rio.crs != crs:
            da = da.rio.reproject(crs)
        bminx, bminy, bmaxx, bmaxy = da.rio.bounds()
        uminx, uminy, umaxx, umaxy = min(uminx,bminx), min(uminy,bminy), max(umaxx,bmaxx), max(umaxy,bmaxy)

    # keep reference origin for grid alignment
    last_time = get_time(tifs[-1])

    # --- snap union to ref grid
    col_min = math.floor((uminx - x0) / xres)
    col_max = math.ceil((umaxx - x0) / xres)
    row_min = math.floor((y0 - umaxy) / (-yres))
    row_max = math.ceil((y0 - uminy) / (-yres))

    print( uminx, uminy, umaxx, umaxy)

    width  = col_max - col_min
    height = row_max - row_min

    snapped_minx = x0 + col_min * xres
    snapped_maxy = y0 - row_min * (-yres)

    final_transform = Affine(xres, 0, snapped_minx, 0, yres, snapped_maxy)

    print('Reproject and stack')
    print('---------')
    # --- reproject all into the snapped union grid and merge
    das = []
    times = []
    ref_da = None

    for fp in tqdm(tifs, desc="stack tif", unit="file"):

        da = open_single_band(fp)

        # remove band dimension if present
        if "band" in da.dims:
            da = da.squeeze("band", drop=True)

        # ensure CRS
        if da.rio.crs != crs:
            da = da.rio.reproject(crs)

        # reproject to union grid
        da = da.rio.reproject(
            crs,
            transform=final_transform,
            shape=(height, width),
            resampling=Resampling.nearest,
            nodata=nodata
        )

        #enforce identical grid
        if ref_da is None:
            ref_da = da.copy()
        else:
            da = da.rio.reproject_match(ref_da)
        #else:
        #    da = da.assign_coords({
        #        "x": ref_da.x,
        #        "y": ref_da.y
        #    })
        
        # enforce variable name
        da = da.rename(f"ir{df_calib_f['lambda'].values[0] * 10:.0f}")

        # get time and convert immediately
        #t = np.datetime64(pd.to_datetime(get_time(fp)))
        t = np.datetime64(pd.to_datetime(get_time(fp)), "ns")

        # assign time coordinate properly
        da = da.expand_dims(time=[t])

        das.append(da)

    # concat
    da_time = xr.concat(das, dim="time")

    # enforce dimension order
    da_time = da_time.transpose("time", "y", "x")

    # convert to Dataset
    ds = da_time.to_dataset()


    ds.to_netcdf(out_path)







