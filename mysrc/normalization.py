import numpy as np
import cv2
import xarray as xr
import pdb 
import matplotlib.pyplot as plt 

def clahe_local_normalization(img, clip_limit=2.0, tile_grid_size=(8, 8), pmin=20, pmax=80):
    """
    Perform local contrast normalization using percentile clipping + CLAHE.
    """
    mask = ~np.isnan(img)
    if not np.any(mask):
        return np.full_like(img, np.nan, dtype=np.float32)

    valid_vals = img[mask]
    vmin, vmax = np.percentile(valid_vals, [pmin, pmax])
    if vmax == vmin:
        return np.full_like(img, np.nan, dtype=np.float32)

    # Clip and normalize to [0, 1]
    img_clipped = np.clip(img, vmin, vmax)
    img_norm = (img_clipped - vmin) / (vmax - vmin)

    # Fill NaNs before CLAHE
    img_norm_filled = img_norm.copy()
    img_norm_filled[~mask] = np.nanmean(img_norm)

    # Convert to uint8
    img_uint8 = (img_norm_filled * 255).astype(np.uint8)

    # Apply CLAHE
    #print(tile_grid_size)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_eq = clahe.apply(img_uint8).astype(np.float32) / 255.0

    # Restore NaNs
    img_eq[~mask] = np.nan
    return img_eq


def clahe_block(block: xr.DataArray, **kwargs) -> xr.DataArray:
    img = block.data.astype(np.float32)
    img_eq = clahe_local_normalization(img, **kwargs)
    return xr.DataArray(img_eq, dims=block.dims, coords=block.coords)

def clahe_cpu(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a 2D float image.
    """
    mask = ~np.isnan(img)
    if not np.any(mask):
        return np.full_like(img, np.nan, dtype=np.float32)

    img_filled = img.copy()
    img_filled[~mask] = np.nanmean(img_filled)

    # Normalize to 0–255 and convert to uint8
    img_uint8 = cv2.normalize(img_filled, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    img_eq = clahe.apply(img_uint8).astype(np.float32) / 255.0

    img_eq[~mask] = np.nan
    return img_eq

def apply_clahe_dask(da: xr.DataArray, **kwargs) -> xr.DataArray:
    if not da.chunks:
        da = da.chunk({'y': 512, 'x': 512})

    da_eq = da.map_blocks(
        clahe_block,
        kwargs=kwargs,
        template=da.astype(np.float32)
    )
    return da_eq

def clahe_block_numpy(img_block, **kwargs):
    return clahe_local_normalization(img_block, **kwargs)
