from pathlib import Path
import re, math, json, datetime
import numpy as np
import rioxarray as rxr
from rioxarray.merge import merge_arrays
from rasterio.enums import Resampling
from affine import Affine
import pdb 
import warnings
from rasterio.errors import NotGeoreferencedWarning

warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)

#folder = Path("/home/paugam/Data/ATR42/as250026/Sijean06_short/ortho")
#out_path = folder / "f1_merged_Sijean06_short.tif"
folder = Path("/home/paugam/Data/ATR42/as250026/Sijean10/ortho")
out_path = folder / "f1_merged_Sijean10.tif"
default_nodata = 0

def numeric_id(path: Path) -> int:
    m = re.search(r'-(\d+)_ORTHO\.tif$', path.name, re.IGNORECASE)
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
    with rxr.open_rasterio(str(fp.absolute()).replace('/ortho/','/tif_f1/').replace('_ORTHO','')) as da:
        desc = da.attrs.get("TIFFTAG_IMAGEDESCRIPTION")
    if not desc:
        return None
    try:
        meta = json.loads(desc)
        time_str = meta.get("Time")
        if time_str:
            return datetime.datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S.%f")
    except Exception:
        return None
    return None

# --- collect and sort
tifs = sorted(folder.glob("*.tif"), key=numeric_id)
if not tifs:
    raise FileNotFoundError("No tifs found")

# --- reference grid (first file)
ref = open_single_band(tifs[0])
first_time = get_time(tifs[0])
crs = ref.rio.crs
xres, yres = ref.rio.resolution()  # yres is negative
nodata = ref.rio.nodata
x0, y0 = ref.rio.transform().c, ref.rio.transform().f  # top-left origin

# --- compute union bounds in ref CRS
uminx, uminy, umaxx, umaxy = ref.rio.bounds()
for fp in tifs[1:]:
    print(fp)
    da = open_single_band(fp)
    if da.rio.crs != crs:
        da = da.rio.reproject(crs)
    bminx, bminy, bmaxx, bmaxy = da.rio.bounds()
    uminx, uminy, umaxx, umaxy = min(uminx,bminx), min(uminy,bminy), max(umaxx,bmaxx), max(umaxy,bmaxy)

x0, y0 =uminx, uminy
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

print('Reproject')
print('---------')
# --- reproject all into the snapped union grid and merge
merged = None
for fp in tifs:
    print(fp)
    da = open_single_band(fp)
    if da.rio.crs != crs:
        da = da.rio.reproject(crs)

    da = da.rio.reproject(
        crs,
        transform=final_transform,
        shape=(height, width),
        resampling=Resampling.nearest,
        nodata=nodata
    )
    if merged is None:
        merged = da
    else:
        merged = merge_arrays([merged, da], method="last", nodata=nodata)
    
merged.attrs["start_time"] = first_time.strftime("%Y-%m-%d %H:%M:%S.%f")
merged.attrs["end_time"]   = last_time.strftime("%Y-%m-%d %H:%M:%S.%f")

# --- save
merged.rio.to_raster(
    out_path,
    compress="LZW",
    tiled=True,
    dtype=str(merged.dtype)
)

print(f"Mosaic written to {out_path}")

