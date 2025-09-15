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
        
    da1 = atr.band_data.isel(band=1)
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
    if (iii) % 50 == 0: flag_plot = True

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
    src_files = [f"/{indir}/{imgdirname}/f1-{idimg:09d}.png" for idimg in idimgs] 

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
        ortho.process( f"{wkdir}/{flightname}_{flightdate}_{startdate}-{idimg}_ORTHO{str_tag}.tif", overwrite=True)
        del ortho, camera
    del cameras
    
    da1_res = [] 
    resi = float(0.0)
    penalty_factor = 1.
    atr_arr = []
    for idimg in idimgs:
        #mem1 = psutil.virtual_memory()
        #atr = xr.open_dataset(wkdir+'as240051_20241113_103254-{:d}_ORTHO{:s}.tif'.format(idimg,str_tag))
        atr = xr.open_dataset(f"{wkdir}/{flightname}_{flightdate}_{startdate}-{idimg}_ORTHO{str_tag}.tif")
        #atr2 = atr.rio.reproject(daRef.rio.crs)
        #del atr
        #da1 = img2da4residu(rr,atr,da1Ref)
        #da2 = img2da4residu(idimgs[1],rrh)
        
        atr_arr.append( atr )


        #get a metric to penalize when da1 non nan are outsie da1Ref value

    diffs = []
    resi = 0
    from itertools import combinations
    for da1, da2 in zip(atr_arr[:-1], atr_arr[1:]): # combinations(atr_arr, 2):
        #da2_on_da1 = da2.interp(x=da1.x, y=da1.y, method="linear")
        #da1r, da2r = xr.align(da1.band_data.isel(band=1), da2_on_da1.band_data.isel(band=1), join="inner")  # keep only overlap
        #if da1r.sizes['x'] * da1r.sizes['y'] == 0 : continue
        #diffs.append(abs(da1r - da2r))

        # 1) Select the band and normalize axis order/orientation
        a = da1.band_data.isel(band=1).sortby("x").sortby("y")
        b = da2.band_data.isel(band=1).sortby("x").sortby("y")

        # 2) Put b on a's grid (same coords). Use "linear" for continuous fields, "nearest" for categorical.
        b_on_a = b.interp(x=a.x, y=a.y, method="linear")


        # 3) Valid overlap where both have finite data
        valid = np.isfinite(a) & np.isfinite(b_on_a)

        # 4) Sum and mean absolute differences over the overlap
        diff = (a - b_on_a).where(valid)
        total_abs_diff = np.abs(diff).sum().item()
        overlap_size = valid.sum().item()

        if overlap_size > 0:
            mean_abs_diff = total_abs_diff / overlap_size
        else:
            mean_abs_diff = np.nan

        resi += mean_abs_diff
        diffs.append(diff)  

        
    #resi_da = xr.concat(diffs, dim="pairs")
    #resi = resi_da.sum(dim=("pairs", "x", "y")).values

    if flag_plot:
        print(correction_xyz)
        print(correction_opk)
        fig = plt.figure()
        ax = plt.subplot(211)
        alpha = 1 
        for ida, da1 in enumerate(atr_arr): # combinations(atr_arr, 2):
            if ida>0: alpha=0.9
            da1.band_data.isel(band=1).plot(ax=ax, add_colorbar=False,alpha=alpha) 
        ax = plt.subplot(212)
        for da_slice in diffs: #(resi_da.sizes["pairs"]):
            (da_slice).plot(ax=ax, add_colorbar=False, vmax=20, vmin=-20)
        
        plt.show()
        pdb.set_trace()
  
    if not(flag_plot):
        os.remove(extparamFile)
        for idimg in idimgs:
            os.remove(wkdir+f'{flightname}_{flightdate}_{startdate}-{idimg}_ORTHO{str_tag}.tif')
    
    #del atr,da1
    iii += 1
    flag_plot = False

    print(f"{o:2.6f}, {p:2.6f}, {k:2.6f}, |  {float(resi)}")
    return float(resi)


##################################
if __name__ == "__main__":
##################################
    import tracemalloc
    tracemalloc.start()
    

    #indir = '/home/paugam/Data/ATR42/as250018/visible/bas/'
    #indir = '/home/paugam/Data/ATR42/as250018/imgRef/'
    indir = '/home/paugam/Data/ATR42/as250018/imgRef/'
    outdir = indir + 'io/'
    wkdir = '/tmp/orthority3/'
    os.makedirs(wkdir, exist_ok=True)

    #imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    imufile = '../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_CORE_NAV_200HZ_20250710_as250018_L1_V1_smooth.nc'
    imgdirname = 'png_f1/'
    #idimgs = [0]   
    idimgs = [2,3,4,5]#,3,4]#,5,6,7]   
    #imgdirname = 'img2'
    #idimgs = [65]   
    demFile =  '{:s}/../dem/dem_as250018.tif'.format(indir)

    indirimg = indir + '{:s}/'.format(imgdirname)
    flightname = 'as250018'
    flightdate = '20250710'
    startdate = '071751'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")         

    intparamFile = f"{indir}/io/{flightname}_int_param.yaml"

    #sentinel
    #s2 = xr.open_dataset(indir+'sentinel_background_test_cropped_{:s}.tif'.format(imgdirname))
    #gradS2 = get_gradient(s2.band_data.isel(band=1))
    #dagradS2 = xr.DataArray(gradS2, dims=["y", "x"], coords={"y": s2.y, "x": s2.x})

    #rr = 3
    #dagradS2 = dagradS2/1000.
    #dagradS2 = dagradS2.where(dagradS2<1,1)
    #daRef = dagradS2
    #daRef = daRef.rolling(x=rr, y=rr, center=True).mean()
    #daRef = local_normalization(daRef, diskSize=200,)
    #daRef = daRef.rio.write_crs(s2.rio.crs)
   

    rr = 3
    da1Rs =  [xr.open_dataset(f"{indir}/ortho/f1-{idimg:09d}_ORTHO.tif").rio.reproject(32631) for idimg in idimgs]
    #da1R = da1R.band_data.isel(band=1)
    #da1R = da1R.drop_vars(['band'])
    #da1R = da1R.rio.reproject(27563)
    #da1R = da1R.coarsen(dim={'x': 2, 'y': 2}, boundary="trim").mean()
    #gradda1R = get_gradient(da1R)
    #dagradda1R = xr.DataArray(gradda1R, dims=["y", "x"], coords={"y": da1R.y, "x": da1R.x})
    #dagradda1R = dagradda1R.interp(x=daRef.x, y=daRef.y)
    #da1Ref = dagradda1R.rolling(x=rr, y=rr, center=True).mean()
    #da1Ref = local_normalization(da1Ref, diskSize=200,)
    #da1Ref = da1Ref.rio.write_crs(da1R.rio.crs)
    #da1Ref = da1Ref.fillna(0)
    
    da1Refs = [img2da4residu(rr,da1R,da1R,flag_ref=True) for da1R in da1Rs ]
     
    #da1Refs[0].plot()
    #plt.show()
    #sys.exit()failed

    ''' 
    da2R =  xr.open_dataset(indir+'img_manualOrtho/as240051_20241113_103254-99.tif')
    da2R = da2R.coarsen(dim={'x': 2, 'y': 2}, boundary="trim").mean()
    gradda2R = get_gradient(da2R.band_data.isel(band=1))
    dagradda2R = xr.DataArray(gradda2R, dims=["y", "x"], coords={"y": da2R.y, "x": da2R.x})
    dagradda2R = dagradda2R.interp(x=daRef.x, y=daRef.y)
    da2Ref = dagradda2R.rolling(x=rr, y=rr, center=True).mean()
    da2Ref = local_normalization(da2Ref, diskSize=200,)
    da2Ret = da2Ref.rio.write_crs(da2R.rio.crs)
    da2Ref = da2Ref.fillna(1)
    '''
    #loc camera to central 479cm, 24.5 cm et -20c

    offset = [ np.array([.5,.5,.5]),    np.array([5,5,5]) ]
    scale = [ np.array([1,1,1]), np.array([10,10,10,]) ]
    #offset = [ np.array([.5,.5,.5]),    np.array([2,2,2]) ]
    #scale = [ np.array([1,1,1]), np.array([4,4,4,]) ]

    imu = xr.open_dataset(indir+imufile)
     

    if False: 
        #popt = np.array([ -0.93871095,  -1.06893665,  -1.03742455,  -1.56363453, 2.64046062, -15.92574779])
        #popt = np.array([-0.96509656,  -1.02242933,  -1.02461382,  -1.54239457, 2.6027513 , -15.99057932])
        #popt = np.array([0.,0.,0.,0., 2.,0 ])
        popt = np.load('resbrute1_xycopk_minimize2.npy',allow_pickle=True).item().x
        #xc,yc,zc,o,p,k = popt.item().x
        #correction_opk = (np.array([o,p,k])-offset[1])    / scale[1]
        #correction_xyz = (np.array([xc,yc,zc])-offset[0]) / scale[0]
        #popt = [*correction_xyz, *correction_xyz]
        params = [ {'offset':offset,'scale':scale}, True,'opk', imu, [0.5,0.5,0.5] , 'resi2']
        residual( popt , params )
        #residual( popt.item().x , params)
        sys.exit()
    '''
    rranges = (slice(-2,2,1), slice(-2,2,1), commentslice(-2,2,1), 
               slice(-2, 2, .5), slice(-2, 2, .5), slice(-2, 2, 0.5))

    if not(os.path.isfile('resbrute1.npy')):
        print('start first opt')
        params = [False]
        resbrute1 = optimize.brute(residual, rranges, args=params, full_output=False,
                                  finish=None,)# workers=16 )
        
        np.save('resbrute1.npy',resbrute1)
    else:
        resbrute1 = np.load('resbrute1.npy')

    print(resbrute1)
    xc,yc,zc,oc,pc,kc = resbrute1
    '''
    #xc,yc,zc,oc,pc,kc = 0.5,0.5,0.5,  0.35,0.7,0.5

    
    import cma
    import numpy as np
    from scipy.optimize import minimize  # optional polish

# --- build the same params tuple you already pass to `residual` ---
    params = [ {'offset':offset,'scale':scale}, False, 'opk', imu, [0.5,0.5,0.5], 'resi1' ]

# We optimize u = [o,p,k] in the latent [0,1]^3 space (as your residual expects).
    def f_latent(u):
        WARN_MSG = r".*You will likely lose important projection information when converting to a PROJ string.*"
        import warnings
        warnings.filterwarnings(
            "ignore",
            category=UserWarning,
            message=WARN_MSG,
            module=r"pyproj\.crs\.crs",
        )
        u = np.asarray(u, dtype=float)
        # (Optional) clamp just in case; CMA-ES should respect bounds though
        u = np.clip(u, 0.0, 1.0)
        return float(residual(u, *params))

# Initial guess and step size (fraction of the box)
    x0      = [0.5, 0.5, 0.5]   # your current starting point
    sigma0  = 0.5               # ~20% of the box; enlarge if you want broader exploration

# CMA-ES options: box bounds [0,1]^3, popsize controls exploration
    opts = {
        'bounds': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        'popsize': 32,        # try 8–32; larger = more global, slower per iteration
        'seed': 54,
        'maxiter': 2000,       # or use 'maxfevals': e.g. 2000
        'verb_disp': 1,
    }

# --- Simple API with restarts (covers larger latent space) ---
    best_u, es = cma.fmin2(f_latent, x0, sigma0, options=opts,
                           restarts=6, incpopsize=2)  # restarts broaden search
    print("CMA-ES best latent [o,p,k]:", best_u)
    
#    from joblib import Parallel, delayed
#
#    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
#    while not es.stop():
#        U = es.ask()
#        # parallel evaluation of the whole population
#        F = Parallel(n_jobs=-1, prefer='processes')(
#                delayed(f_latent)(u) for u in U
#            )
#        es.tell(U, F)
#        es.disp()

#    best_u = es.result.xbest
#    print("CMA-ES best latent [o,p,k]:", best_u)

# Optional: local polish with Nelder–Mead (still in latent space)
    res_nm = minimize(f_latent, best_u, method='Nelder-Mead',
                      options={'xatol': 0.01, 'fatol': 0.1, 'maxiter': 300, 'disp': True})
    final_u = res_nm.x
    print("Final latent [o,p,k]:", final_u)

# If you want the physical angle corrections (degrees), apply your mapping explicitly:
    phys = (np.array(final_u) * scale[1]) - offset[1]
    print("Final physical [o,p,k] degrees:", phys)




    sys.exit()
    

    xc,yc,zc,oc,pc,kc = 0.5,0.5,0.5,  0.5,0.5,0.5
    
    #resbrutef =  [xc,yc,zc] + [oc,pc,kc]
    resbrutef =  [oc,pc,kc]
    
    from scipy.optimize import minimize
    params = [ {'offset':offset,'scale':scale}, False,'opk', imu, [0.5,0.5,0.5], 'resi1' ]

    # Your initial parameter guess (6 elements for x, y, z, omega, phi, kappa)
    x0 = np.array(resbrutef)  # assuming resbrutef is a list or array of length 6

    # Construct an initial simplex: one vertex is x0, others are small perturbations
    step_size = 0.9  # Increase this if your function is too flat at the start
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
            'disp': True,
            'maxiter': 1000  # optional: increase if needed
        }
    )
    np.save('resbrute1_xycopk_minimize1.npy',result)


    popt = result.x
    params = [ {'offset':offset,'scale':scale}, False,'opk', imu, [0.5,0.5,0.5], 'resi2' ]

    # Your initial parameter guess (6 elements for x, y, z, omega, phi, kappa)
    x0 = np.array(popt)  # assuming resbrutef is a list or array of length 6

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
            'disp': True,
            'maxiter': 1000  # optional: increase if needed
        }
    )

    np.save('resbrute1_xycopk_minimize2.npy',result)
    sys.exit()


# Resulting optimal parameters
    popt = result.x

    print('start last opt')
    params = [ {'offset':offset,'scale':scale}, False,'opk', imu, [0,0,0]]
    popt2 = optimize.fmin(residual, tuple(popt), args=tuple(params), xtol=0.01, ftol=0.01, disp=False)
    np.save('resbrute2_xycopk_minimize.npy',popt2)
    sys.exit()




    print('start opk opt')
    rranges = (slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7))
    params = [ {'offset':offset,'scale':scale}, False,'xyzopk', imu]
    resbrute2 = optimize.brute(residual, rranges, args=params, full_output=False,
                              finish=None)#, workers=16 )
    oc,pc,kc = resbrute2
    np.save('resbrute1_opk.npy',resbrute2)
    print(xc,yc,zc,oc,pc,kc)

    #rranges = (slice(oc-3,oc+3,.5), slice(pc-3,pc+3,.5), slice(kc-3,  kc+3,  .5))
    '''
    print('start opk opt')
    rranges = (slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7))
    params = [ {'offset':offset,'scale':scale}, False,'opk',xc,yc,zc, imu]
    resbrute2 = optimize.brute(residual, rranges, args=params, full_output=False,
                              finish=None)#, workers=16 )
    oc,pc,kc = resbrute2
    np.save('resbrute1_opk.npy',resbrute2)
    print(xc,yc,zc,oc,pc,kc)

    #rranges = (slice(xc-20,xc+20,3), slice(yc-20,yc+20,3), slice(zc-20,  zc+20,  3))
    print('start xyz opt')
    rranges = (slice(0,1,1./7), slice(0,1,1./7), slice(0,1,1./7))
    params = [ {'offset':offset,'scale':scale}, False,'xyz',oc,pc,kc, imu]
    resbrute2 = optimize.brute(residual, rranges, args=params, full_output=False,
                              finish=None)#, workers=16 )
    xc,yc,zc = resbrute2
    np.save('resbrute1_xyc.npy',resbrute2)
    '''

    print(xc,yc,zc,oc,pc,kc)
    
    resbrutef =  [xc,yc,zc] + [oc,pc,kc]

    print('start last opt')
    params = [ {'offset':offset,'scale':scale}, False,'xyzopk', imu]
    popt = optimize.fmin(residual, tuple(resbrutef), args=tuple(params), xtol=0.1, ftol=0.1, disp=False)

    np.save('resbrute22.npy',popt)
    print(popt)
    #resbrute.append([-1.69183706,   3.06752315, -14.98225642])
    #resbrute = []
    #resbrute.append([0,0,0, -0.63767196,   3.36609474, -16.31209226])


