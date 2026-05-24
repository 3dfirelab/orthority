
import rasterio
from rasterio.merge import merge
from rasterio.fill import fillnodata
import numpy as np


dirData = "/home/paugam/Data/ATR_test/flight01/dem"
files = [
    f"{dirData}/piper01_D031_dem_1m.tif",
    f"{dirData}/piper01_D032_dem_1m.tif",
    f"{dirData}/piper01_D065_dem_1m.tif",
]

srcs = [rasterio.open(f) for f in files]
mosaic, transform = merge(srcs)

profile = srcs[0].profile.copy()
profile.update(
    height=mosaic.shape[1],
    width=mosaic.shape[2],
    transform=transform,
    compress="lzw",
    tiled=True,
    nodata=np.nan,
)

dem = mosaic[0].astype("float32")
valid = np.isfinite(dem)

filled = fillnodata(
    dem,
    mask=valid,
    max_search_distance=100,
    smoothing_iterations=2,
)

with rasterio.open(f"{dirData}/piper01_dem_1m.tif", "w", **profile) as dst:
    dst.write(filled.astype("float32"), 1)

for s in srcs:
    s.close()
