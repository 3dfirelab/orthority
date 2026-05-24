import numpy as np 
import cv2
import xarray as xr 
import matplotlib.pyplot as plt
import subprocess
import os
from scipy import optimize
import warnings
import pdb 
import geopandas as gpd
import skimage
from scipy.signal import convolve
import psutil
import tempfile
import orthority as oty
import importlib 
import sys
import gc
import functools
from scipy.optimize import minimize
import cv2
import pandas as pd
from shapely.geometry import box
import argparse

#homebrewed
import imuNcOoGeojson  
importlib.reload(imuNcOoGeojson)
import normalization
importlib.reload(normalization)
from normalization import clahe_block_numpy
import numpy as np
import xarray as xr
import skimage
from skimage.morphology import disk
from skimage.filters.rank import equalize
import warnings

import dask
from dask.diagnostics import ProgressBar

import numpy as np
import xarray as xr
from skimage.morphology import disk
from skimage.filters.rank import equalize
from skimage.util import img_as_uint
import warnings
import dask.array as da


iii = 0

def local_equalize_block_numpy(arr, diskSize=30):
    mask = ~np.isnan(arr)
    if not np.any(mask):
        return np.full_like(arr, np.nan, dtype=np.float32)

    valid_vals = arr[mask]
    p20, p80 = np.percentile(valid_vals, [20, 80])
    if p80 == p20:
        return np.full_like(arr, np.nan, dtype=np.float32)

    arr[~mask] = p80
    clipped = np.clip(arr, p20, p80)
    norm = (clipped - p20) / (p80 - p20)
    norm_uint16 = skimage.img_as_uint(norm)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        eq = equalize(norm_uint16, footprint=disk(diskSize), mask=mask.astype(np.uint8))

    result = eq.astype(np.float32) / 65535.0
    result[~mask] = np.nan
    return result


def local_normalization_block(block: xr.DataArray, diskSize=30) -> xr.DataArray:
    input_float = block.data.astype(np.float32)  # raw NumPy array
    mask = ~np.isnan(input_float)

    if not np.any(mask):
        result = np.full_like(input_float, np.nan, dtype=np.float32)
        return xr.DataArray(result, dims=block.dims, coords=block.coords)

    valid_vals = input_float[mask]
    p20, p80 = np.percentile(valid_vals, [20, 80])
    if p80 == p20:
        result = np.full_like(input_float, np.nan, dtype=np.float32)
        return xr.DataArray(result, dims=block.dims, coords=block.coords)

    trange = np.float32([p20, p80])

    input_float[~mask] = trange[1]
    clipped = np.clip(input_float, trange[0], trange[1])
    norm = (clipped - trange[0]) / (trange[1] - trange[0])
    norm_uint16 = skimage.img_as_uint(norm)

    selem = disk(diskSize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_eq = equalize(norm_uint16, footprint=selem, mask=mask.astype(np.uint8))

    result = img_eq.astype(np.float32) / 65535.0
    result[~mask] = np.nan

    return xr.DataArray(result, dims=block.dims, coords=block.coords)
def apply_local_normalization_dask(da, diskSize=30):
    """Wrapper to apply local normalization using Dask over 2D chunks."""
    if not da.chunks:
        # If not already chunked, choose a reasonable chunk size
        da = da.chunk({'y': 512, 'x': 512})

    # Apply the function over 2D chunks
    normed = da.map_blocks(
        local_normalization_block,
        kwargs={'diskSize': diskSize},
        template=da,
    )
    return normed


############################
def local_normalization3(da: xr.DataArray, diskSize=30) -> xr.DataArray:
    input_float = da.data.astype(np.float32)
    mask = ~np.isnan(input_float)

    valid_vals = input_float[mask]
    if valid_vals.size == 0:
        return da

    p20, p80 = np.percentile(valid_vals, [20, 80])
    trange = np.float32([p20, p80])

    # Clip values and replace NaNs temporarily
    clipped = np.clip(np.nan_to_num(input_float, nan=trange[1]), trange[0], trange[1])

    # Normalize to [0, 1]
    norm = (clipped - trange[0]) / (trange[1] - trange[0])
    norm_uint16 = skimage.img_as_uint(norm)

    selem = disk(diskSize)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img_eq = equalize(norm_uint16, footprint=selem, mask=mask.astype(np.uint8))

    # Rescale back to float32 in original range
    da.data = (img_eq / 65535.0).astype(np.float32)
    da.data[~mask] = np.nan
    return da

#########################################
def local_normalization_OLD(da, diskSize=30,):

    #idx = np.where(mask==1)
    #trange_ = [input_float[idx].min(), input_float[idx].max()]
    #img = convert_2_uint16(input_float, trange_ )
    input_float = np.array(da.data)#,dtype=np.float32)
    idx = np.where(~np.isnan(input_float))
    idxnan = np.where(np.isnan(input_float))
    mask = np.zeros_like(input_float)
    mask[idx] = 1
    try: 
        trange = np.array([np.percentile(input_float[idx],20), np.percentile(input_float[idx],80)]).astype(np.float32)
    except: 
        pdb.set_trace()
    input_float[idxnan] = trange[1]
    input_float = np.where(input_float<trange[0],trange[0],input_float)
    input_float = np.where(input_float>trange[1],trange[1],input_float)
    input_float = input_float.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            img = skimage.img_as_uint((input_float-trange[0])/(trange[1]-trange[0]))
        except: 
            pdb.set_trace()

        selem = skimage.morphology.disk(diskSize)
        #img_eq = skimage.filters.rank.equalize(img, selem=selem, mask=mask)
        img_eq = skimage.filters.rank.equalize(img, footprint=selem, mask=mask)
    
    da.data = np.float32((img_eq/6.5535e4) )
    da.data[idxnan] = np.nan
    return da

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
def img2da4residu(rrh, atr, da1Ref, flag_ref=False):
   
    #with  xr.open_dataset(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimgs_)) as atr: 
        #atr = atr.rio.reproject(daRef.rio.crs)
        
    da1 = atr.band_data.isel(band=0)
    da1 = da1.coarsen(dim={'x': 2, 'y': 2}, boundary="trim").mean()
    da1 = xr.DataArray(get_gradient(da1), dims=["y", "x"], coords={"y": da1.y, "x": da1.x})
    da1 = da1.interp(x=da1Ref.x, y=da1Ref.y)
  
    if flag_ref:
        buffer = 100
    else:
        buffer = 400

    mask = np.where(np.isnan(da1.values), 0, 1).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones([buffer,buffer]))
    #da1 = da1.where(mask==1)

    
    #da_mask = dagradAtr_inter.where(~dagradAtr_inter.isnull(), -9999)
    #dagradAtr_inter = dagradAtr_inter.where(dagradAtr_inter<1,1)
    #dagradAtr_inter = dagradAtr_inter.where(da_mask!=-9999,np.nan)

    #da1 = da1.rio.write_crs(daRef.rio.crs)
    #da1 = da_coarse.interp(x=dagradS2.x, y=dagradS2.y)
    #da1 = da1.rio.write_crs(s2.rio.crs)
    
    #da1 = da1.rio.clip(mask.geometry.values, mask.crs)
    #MERDE
    ##da1 = da1.rolling(x=rrh, y=rrh, center=True).mean()
    
    #nan_count = da1.isnull().sum()  # Count of NaN values
    #total_count = da1.size          # Total number of elements
    #nan_coverage = nan_count / total_count  # Fraction of NaN values
    #if nan_coverage > 0.9: 
    #    return da1 
    #da1 = local_normalization(da1, diskSize=100,)

    #da1 = da1.fillna(0) 
    #da1 = local_normalization(da1, diskSize=50,)
    
    '''
    da_chunked = da1.chunk({'y': 512, 'x': 512})

    # Apply normalization
    da_norm= apply_local_normalization_dask(da_chunked)
    
    with dask.config.set(scheduler='threads', num_workers=20):
        #with ProgressBar():
        da1 = da_norm.compute()

    '''
    #merde
    '''
    # Convert xarray to dask array
    darr = da.from_array(da1.data, chunks=(512, 512))

    # Overlap of diskSize on each edge
    diskSize = 50
    overlap = diskSize

    # Apply with overlap
    result = da.map_overlap(
        local_equalize_block_numpy,
        darr,
        depth=overlap,
        boundary='reflect',
        dtype=np.float32,
        diskSize=diskSize
    )

    # Convert back to xarray
    da1 = xr.DataArray(result, dims=da1.dims, coords=da1.coords)
    '''
    #def apply_local_clahe_2d(da: xr.DataArray, block_size=(64, 64), clip_limit=0.01) -> xr.DataArray:
    #    result = clahe_block_numpy(da.values, block_size=block_size, clip_limit=clip_limit)
    #    return xr.DataArray(result, coords=da.coords, dims=da.dims, attrs=da.attrs)

    # Convert to Dask array and chunk it
    darr = da1.chunk({'y': da1.shape[0], 'x': da1.shape[1]}).data

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
    da_norm = normalization.apply_clahe_dask(
        da1.chunk({'y': 512, 'x': 512}),
        clip_limit=2.0,
        tile_grid_size=(16, 16),
        pmin=20,
        pmax=80
    )
    da1 = da_norm.compute()
    '''

    
    #ax = plt.subplot(111) 
    #(da1).plot(ax=ax)
    #(da1).plot.contour(ax=ax,colors='k')
    #plt.show()
    #pdb.set_trace() 
    
    return da1

#################################################
def residual(args, *params):
    
    if len(params)==1: 
        params= params[0]
    flag_plot = params[1]
    global iii
    
    #if (iii) % 50 == 0: flag_plot = True

    flag_resi = params[5]

    #mem0 = psutil.virtual_memory()
    #rrh = 3
    if len(params) == 1: 
        params = params[0]

    if params[2] == 'opk':
        o, p, k = args[:]
        xc, yc, zc = params[4]
    if params[2] == 'xyz':
        xc, yc, zc = args[:]
        o, p, k = params[3:]
    if params[2] == 'xyzopk':
        o, p, k   = args[3:]
        xc, yc, zc = args[:3]
    
    offset = np.array(params[0]['offset'])
    scale = np.array(params[0]['scale'])

    correction_opk = (np.array([o,p,k])*scale[1])    - offset[1]
    correction_xyz = (np.array([xc,yc,zc])*scale[0]) - offset[0]

    #atr
    #correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    #correction_opk = np.array([0.,3.,-16]) # degree
    #src_files = [f"/{indir}/{imgdirname}/{flightname}_{flightdate}_{startdate}-{idimg}.tif" for idimg in idimgs] 
    src_files = [f"/{indir}/{imgdirname}/f1-{idimg:09d}.tif" for idimg in idimgs] 

    str_tag = '_{:.1f}{:.1f}{:.1f}_{:.1f}{:.1f}{:.1f}'.format(*correction_xyz,*correction_opk).replace('.','p')
    imuNcOoGeojson.imutogeojson( params[3], wkdir, indirimg, flightname, correction_xyz, correction_opk, src_files,str_tag=str_tag)

    #for idimg in idimgs:
    #    if os.path.isfile(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimg)):
    #        os.remove(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimg))
  
    '''
    command = [
            "oty", "frame",
            "--dem", '{:s}/dem/dem.tif'.format(indir),
            "--int-param", "{:s}/io/as240051_int_param.yaml".format(indir),
            "--ext-param", "{:s}/io/as240051_ext_param.geojson".format(indir),
            "--out-dir", indir,
            "-o", 
            "/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimgs[0]), 
            "/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimgs[1]),
            #"/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimgs[2]),
            #"/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimgs[3]),
            ]
    '''
    #mask = gpd.read_file(indir+'mask.shp')

    #run orthorectification
    #result = subprocess.run(command, capture_output=True, text=True)
    #subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    #process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    #stdout, stderr = process.communicate()

    #if process.info['status'] == psutil.STATUS_ZOMBIE:
    #     print(f"Zombie Process Found: PID={proc.info['pid']}, Name={proc.info['name']}")

            
    extparamFile =  "{:s}/{:s}_ext_param{:s}.geojson".format(wkdir,flightname,str_tag)
    #create a camera model for src_file from interior & exterior parameters
    cameras = oty.FrameCameras(intparamFile, extparamFile)

    for idimg, src_file in zip(idimgs,src_files):
        #src_file =  "/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimg) 
        camera = cameras.get(src_file)
        
        # create Ortho object and orthorectify
        ortho = oty.Ortho(src_file, demFile, camera=camera, crs=cameras.crs)
        ortho.process( f"{wkdir}/{flightname}_{flightdate}-{idimg}_ORTHO{str_tag}.tif", overwrite=True)
        del ortho, camera
    del cameras
    
    da1_res = [] 
    resi = float(0.0)
    penalty_factor = 1.
    for idimg,da1Ref in zip(idimgs,da1Refs):
        #mem1 = psutil.virtual_memory()
        #atr = xr.open_dataset(wkdir+'as240051_20241113_103254-{:d}_ORTHO{:s}.tif'.format(idimg,str_tag))
        atr = xr.open_dataset(f"{wkdir}/{flightname}_{flightdate}-{idimg}_ORTHO{str_tag}.tif")
        #atr2 = atr.rio.reproject(daRef.rio.crs)
        #del atr
        da1 = img2da4residu(rr,atr,da1Ref)

        if flag_resi == 'resi2':
            resi += float(abs((da1-da1Ref)).sum()) #+ abs((da2-da2Ref)).sum() )#+  abs((da3-daRef)).sum() + abs((da4-daRef)).sum()
       
        overlap_mask = (~da1Ref.isnull())  # or any valid domain definition
        #resi += float(abs((da1 - da1Ref)).where(overlap_mask,0.0).sum())
        #print(resi, end=' | ')
        resi1 = resi 
        da1_mask = (~da1.isnull())  # or any valid domain definition
        if flag_resi == 'resi1':
            resi += float(np.abs(da1_mask.values.astype(int)-overlap_mask.values.astype(int)).sum()) 
     
        #print(resi-resi1, end=' | ')
        if resi<1: pdb.set_trace()

        da1_res.append(da1)

    #del da2 
    #gc.collect()
    #mem2 = psutil.virtual_memory()
    #print(mem2.used/mem1.used)
    
    #mem = psutil.virtual_memory()
    #print('{:.2f},{:.2f},{:.2f}  {:.2f},{:.2f},{:.2f} | {:.1f} || {:.1f} || {:.3f},{:.3f},{:.3f} {:d} '.format(*correction_xyz,*correction_opk, resi, mem.used/(1024 ** 3), o,p,k, iii))
    
    '''snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:3]:
        print(stat)
    '''
  
  
    os.remove(extparamFile)
    for idimg in idimgs:
        os.remove(wkdir+f'{flightname}_{flightdate}-{idimg}_ORTHO{str_tag}.tif')
    
    del atr,da1
    iii += 1
    flag_plot = False


    return float(resi)

def get_shift(da1Refs, da10ris):
    
    ref = da1Refs[0]
    img = da10ris[0]

    ref_bounds = ref.rio.bounds()
    img_bounds = img.rio.bounds()

    ref_box = box(*ref_bounds)
    img_box = box(*img_bounds)

    overlap = ref_box.intersection(img_box)
    if overlap.is_empty:
        rel_overlap = 0.0
    else:
        rel_overlap = overlap.area / img_box.area

    if rel_overlap < 0.5: 
        return None, None 

    # reproject img onto ref grid
    img_match = img.rio.reproject_match(ref)

    # now both have same:
    # - CRS
    # - resolution
    # - extent
    # - shape

    # convert to numpy
    ref_np = ref.compute().values.astype(np.float32)
    img_np = img_match.compute().values.astype(np.float32)

    # NaN handling
    mask = np.isfinite(ref_np) & np.isfinite(img_np)

    ref_np = np.where(mask, ref_np, 0)
    img_np = np.where(mask, img_np, 0)

    # normalize
    ref_np = cv2.normalize(ref_np, None, 0, 1, cv2.NORM_MINMAX)
    img_np = cv2.normalize(img_np, None, 0, 1, cv2.NORM_MINMAX)

    warp = np.eye(2, 3, dtype=np.float32)

    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        500,
        1e-7
    )

    try:
        cc, warp = cv2.findTransformECC( ref_np, img_np, warp, cv2.MOTION_TRANSLATION, criteria )
    except: 
        pdb.set_trace()

    dx_pix = warp[0, 2]
    dy_pix = warp[1, 2]


    return [dx_pix, dy_pix], cc


##################################
if __name__ == "__main__":
##################################
    import tracemalloc
    tracemalloc.start()
       parser = argparse.ArgumentParser()

    parser.add_argument(
        "--imufile_name",
        type=str,
        default="safire",
        help="IMU file name"
    )

    parser.add_argument(
        "--transectName",
        type=str,
        default='001_320256',
        help="transect name"
    )
    parser.add_argument(
        "--minimizeID",
        type=int,
        default=3,
        help="transect name"
    )
    args = parser.parse_args()
     
    imufile_name = args.imufile_name
    transectname = args.transectName
    minimizeID = args.minimizeID
    
    flightname = 'piper01'
    flightdate = '20260520'
    delta_img = 25
    #imufile_name = 'loa'
    #minimizeID = 3 # 2, 3, 4
    #imufile_name = 'safire'
    #minimizeID = 4 # 2, 3, 4

    indir = f'/home/paugam/Data/ATR_test/flight01/Transects/{transectname}_{imufile_name}'
    
    outdir = indir + 'io/'
    wkdir = '/tmp/orthority3/'
    os.makedirs(wkdir, exist_ok=True)

    #imufile = '/../../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_NAV_ATLANS_200HZ_20250726_as250026_L1_V1_smooth.nc'
    
    if imufile_name == 'loa':
        imufile = '/../../safire/piper01_psbga_gpgga_sync.gpkg'
    if imufile_name == 'safire':
        imufile = '/../../safire/piper01_safire.gpkg'
    

    imgdirname = 'tif_f1/'
    demFile =  '{:s}/../../dem/{:s}_dem_1m.tif'.format(indir,flightname)

    indirimg = indir + '{:s}/'.format(imgdirname)
    
    
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")         

    intparamFile = f"{indir}/io/{flightname}_int_param.yaml"
        
    offset = [ np.array([.5,.5,.5]),    np.array([2,2,2]) ]
    scale = [ np.array([1,1,1]), np.array([4,4,4]) ]

    #imu = xr.open_dataset(indir+imufile)
    imu = gpd.read_file(indir+imufile)
    imu = imu.dropna(subset=['latitude'])

    imu = imu.rename(columns={'datetime_utc': 'time'})
    imu = imu.rename(columns={'latitude': 'LATITUDE'})
    imu = imu.rename(columns={'longitude': 'LONGITUDE'})
    imu = imu.rename(columns={'altitude_m': 'ALTITUDE'})
    imu = imu.rename(columns={'roll_deg': 'ROLL_smooth'})
    imu = imu.rename(columns={'pitch_deg': 'PITCH_smooth'})
    imu = imu.rename(columns={'heading_deg': 'THEAD_smooth'})


    popt = np.load(f'resbrute1_xycopk_minimize{minimizeID}_{transectname}_{imufile_name}.npy', allow_pickle=True).item().x
   
    arr = []
    for idimg_ in range(52,912,20):  

        idimgs = [idimg_]
        rr = 3
        idimg_ref = idimgs[0] - delta_img
        da1Rs =  [xr.open_dataset(f"{indir}/full_ortho_f1/f1-{idimg_ref:09d}_expcorr_ORTHO.tif").rio.reproject(32631) ]
        da1Refs = [img2da4residu(rr,da1R,da1R,flag_ref=True) for da1R in da1Rs ]
        
        da10s =  [xr.open_dataset(f"{indir}/full_ortho_f1/f1-{idimg_:09d}_expcorr_ORTHO.tif").rio.reproject(32631) ]
        da10ris = [img2da4residu(rr,da10,da10,flag_ref=True) for da10 in da10s ]

        shift, cc = get_shift(da1Refs, da10ris)
        if shift is None: 
            arr.append( [idimgs[0], idimg_ref, None, None] )
            continue
        
        params = [ {'offset':offset,'scale':scale}, False,'xyzopk', imu, [], 'resi2' ]

        # Your initial parameter guess (6 elements for x, y, z, omega, phi, kappa)
        x0 = np.array([*popt])   # assuming resbrutef is a list or array of length 6

        # Construct an initial simplex: one vertex is x0, others are small perturbations
        step_size = 0.05  # Increase this if your function is too flat at the start
        n = len(x0)
        initial_simplex = np.vstack([x0] + [x0 + step_size * np.eye(n)[i] for i in range(n)])

        # Perform optimization using Nelder-Mead with the custom simplex
        result = minimize(
            fun=residual,
            x0=x0,
            args=tuple(params),  # your additional parameters to residual()
            method='Nelder-Mead',
            options={
                'initial_simplex': initial_simplex,
                'xatol': 0.01,
                'fatol': 0.1,
                'disp': False,
                'maxiter': 1000  # optional: increase if needed
            }
        )

        print(idimg_)
        print(x0)
        print(result.x)
        print(shift)
        print('-----')
        arr.append( [idimgs[0], idimg_ref, result.x-x0, shift] )
        

    df = pd.DataFrame(
        arr,
        columns=[
            "id_img",
            "id_img_ref",
            "diff_6d",
            "pixel_shift"
        ]
    )
    df = df[df[pixel_shift].notnull()]

    df[["diff_x", "diff_y", "diff_z","diff_o","diff_p","diff_k", ]] = pd.DataFrame(
        df["diff_6d"].tolist(),
        index=df.index
    )
    df = df.drop(columns=["diff_6d"])
    
    df.to_csv(f"error_imu_sequence_piper01_{transectname}_{imufile_name}_deltaImg{delta_img}_mID{minimizeID}.csv", index=False)

    fig = plt.figure()
    ax = plt.subplot(121)
    df.diff_x.plot(ax=ax, label='diff_x')
    df.diff_y.plot(ax=ax, label='diff_y')
    df.diff_z.plot(ax=ax, label='diff_z')
    ax = plt.subplot(122)
    df.diff_o.plot(ax=ax, label='diff_o')
    df.diff_p.plot(ax=ax, label='diff_p')
    df.diff_k.plot(ax=ax, label='diff_k')
    fig.savefig(f"error_imu_sequence_piper01_{transectname}_{imufile_name}_deltaImg{delta_img}_mID{minimizeID}.png")
    plt.close(fig)
