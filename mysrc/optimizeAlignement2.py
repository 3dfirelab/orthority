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
from scipy.optimize import minimize

#homebrewed
import imuNcOoGeojson  
importlib.reload(imuNcOoGeojson)
import normalization
importlib.reload(normalization)
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
def img2da4residu(da1, da2, flag_ref=False):
   
    #with  xr.open_dataset(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimgs_)) as atr: 
        #atr = atr.rio.reproject(daRef.rio.crs)
    da2 = da2.band_data.isel(band=1)
    #da2 = da2.coarsen(dim={'x': 8, 'y': 8}, boundary="trim").mean()
    da2 = xr.DataArray(get_gradient(da2), dims=["y", "x"], coords={"y": da2.y, "x": da2.x})
        
    da1 = da1.band_data.isel(band=1)
    #da1 = da1.coarsen(dim={'x': 8, 'y': 8}, boundary="trim").mean()
    da1 = xr.DataArray(get_gradient(da1), dims=["y", "x"], coords={"y": da1.y, "x": da1.x})
    da1 = da1.interp(x=da2.x, y=da2.y)
  
    if flag_ref:
        buffer = 50
    else:
        buffer = 100

    mask = np.where(np.isnan(da1.values), 0, 1).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, np.ones([buffer,buffer]))
    da1 = da1.where(mask==1)

    
    # Convert to Dask array and chunk it
    darr = da1.chunk({'y': 512, 'x': 512}).data

    # Apply map_overlap with buffer
    darr_eq = da.map_overlap(
        normalization.clahe_block_numpy,
        darr,
        depth=16,
        boundary='reflect',
         tile_grid_size=(17,17),   #(512+32(=depthx2))/32(=disksize) = 17 
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
def residual2(args, *params):
   
    #mem0 = psutil.virtual_memory()
    #rrh = 3
    if len(params) == 1: 
        params = params[0]

    if params[2] == 'opk':
        o, p, k = args[:]
        xc, yc, zc = params[3:]
    if params[2] == 'xyz':
        xc, yc, zc = args[:]
        o, p, k = params[3:]
    if params[2] == 'xyzopk':
        o, p, k   = args[3:]
        xc, yc, zc = args[:3]
    
    flag_plot = params[1]

    offset = np.array(params[0]['offset'])
    scale = np.array(params[0]['scale'])

    correction_opk = (np.array([o,p,k])*scale[1])    - offset[1]
    correction_xyz = (np.array([xc,yc,zc])*scale[0]) - offset[0]

    #atr
    #correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    #correction_opk = np.array([0.,3.,-16]) # degree
    src_files = [f"/{indir}/{imgdirname}_corrected_masked/{flightname}_{flightdate}_{startdate}-{idimg}.tif" for idimg in idimgs] 

    str_tag = '_{:.1f}{:.1f}{:.1f}_{:.1f}{:.1f}{:.1f}'.format(*correction_xyz,*correction_opk).replace('.','p')
    imuNcOoGeojson.imutogeojson( params[-1], wkdir, indirimg, flightname, correction_xyz, correction_opk, src_files,str_tag=str_tag)

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
    penalty_factor = 10
    
    idimg1 = idimgs[0]
    da1 = xr.open_dataset(f"{wkdir}/{flightname}_{flightdate}_{startdate}-{idimg1}_ORTHO{str_tag}.tif")
    idimg2 = idimgs[1]
    da2 = xr.open_dataset(f"{wkdir}/{flightname}_{flightdate}_{startdate}-{idimg2}_ORTHO{str_tag}.tif")
  
    da1 = img2da4residu(da1,da2)
    da2 = img2da4residu(da2,da2)
    
    overlap_mask = (da2 > 0)  # or any valid domain definition
    resi += float(abs((da1 - da2)).where(overlap_mask,0.0).sum())
    
    resi += float(np.abs(da1.where(~overlap_mask,0.0).sum())) * penalty_factor
    
    
    mem = psutil.virtual_memory()
    print('{:.2f},{:.2f},{:.2f}  {:.2f},{:.2f},{:.2f} | {:.1f} || {:.1f} '.format(*correction_xyz,*correction_opk, resi, mem.used/(1024 ** 3)))
    
    '''snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:3]:
        print(stat)
    '''
  
    if flag_plot:
        print(correction_xyz)
        print(correction_opk)
        
        fig = plt.figure()
        ax = plt.subplot(121)
        (da1-da2).plot(ax=ax)
        #ax = plt.subplot(122)
        #(da2-da2Ref).plot(ax=ax)
        ax = plt.subplot(122)
        (da2).plot(ax=ax,alpha=.5)
        (da1).plot.contour(ax=ax,colors='k')

        plt.show()
        pdb.set_trace()
  
    if not(flag_plot):
        os.remove(extparamFile)
        for idimg in idimgs:
            os.remove(wkdir+f'{flightname}_{flightdate}_{startdate}-{idimg}_ORTHO{str_tag}.tif')
    
    del da2, da1

    return float(resi)


##################################
if __name__ == "__main__":
##################################
    import tracemalloc
    tracemalloc.start()
    

    indir = '/home/paugam/Data/ATR42/as250018/visible/bas/'
    outdir = indir + 'io/'
    wkdir = '/tmp/orthority2/'
    os.makedirs(wkdir, exist_ok=True)

    #imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    imufile = '../../safire/SILEX-2025_SAFIRE-ATR42_SAFIRE_CORE_NAV_200HZ_20250710_as250018_L1_V1.nc'
    imgdirname = 'img'
    idimgs = [7450,  7459]   
    #imgdirname = 'img2'
    #idimgs = [65]   
    demFile =  '{:s}/../../dem/dem_7452.tif'.format(indir)

    indirimg = indir + '{:s}/'.format(imgdirname)
    flightname = 'as250018'
    flightdate = '20250710'
    startdate = '071751'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")         

    intparamFile = f"{indir}/io/{flightname}_int_param.yaml"

    '''
    rr = 3
    da1Rs =  [xr.open_dataset(f"{indir}/img_corrected_masked_manualOrtho/{flightname}_{flightdate}_{startdate}-{idimg}_modified.tif").rio.reproject(32631) for idimg in idimgs]
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
    
    '''
    #da1Ref.plot()
    #plt.show()
    #sys.exit()

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
    #offset = [ np.array([25,25,25]),    np.array([3.5,3.5,3.5]) ]
    #scale = [ np.array([50,50,50]), np.array([7,7,7,]) ]
    offset = [ np.array([.5,.5,.5]),    np.array([3.5,3.5,3.5]) ]
    scale = [ np.array([1,1,1]), np.array([7,7,7,]) ]
    imu = xr.open_dataset(indir+imufile)
    
    if True: 
        #popt = np.array([ -0.93871095,  -1.06893665,  -1.03742455,  -1.56363453, 2.64046062, -15.92574779])
        #popt = np.array([-0.96509656,  -1.02242933,  -1.02461382,  -1.54239457, 2.6027513 , -15.99057932])
        #popt = np.array([0.,0.,0.,0., 2.,0 ])
        popt = np.load('opt2_resbrute1_xycopk_minimize.npy',allow_pickle=True).item().x
        #popt = np.load('opt2_resbrute2_xycopk_minimize.npy',allow_pickle=True)
        #xc,yc,zc,o,p,k = popt.item().x
        #correction_opk = (np.array([o,p,k])-offset[1])    / scale[1]
        #correction_xyz = (np.array([xc,yc,zc])-offset[0]) / scale[0]
        #popt = [*correction_xyz, *correction_xyz]
        params = [ {'offset':offset,'scale':scale}, True,'xyzopk', imu]
        residual2( popt , params)
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
    popt = np.load('resbrute2_xycopk_minimize.npy',allow_pickle=True)
    xc,yc,zc,oc,pc,kc = [ 0.5, 0.5, 0.5,0.5, 0.5, 0.5 ] #popt
    
    resbrutef =  [xc,yc,zc] + [oc,pc,kc]
    
    params = [ {'offset':offset,'scale':scale}, False,'xyzopk', imu]

    # Your initial parameter guess (6 elements for x, y, z, omega, phi, kappa)
    x0 = np.array(resbrutef)  # assuming resbrutef is a list or array of length 6

    # Construct an initial simplex: one vertex is x0, others are small perturbations
    step_size = 0.1  # Increase this if your function is too flat at the start
    n = len(x0)
    initial_simplex = np.vstack([x0] + [x0 + step_size * np.eye(n)[i] for i in range(n)])

    # Perform optimization using Nelder-Mead with the custom simplex
    result = minimize(
        fun=residual2,
        x0=x0,
        args=tuple(params),  # your additional parameters to residual()
        method='Nelder-Mead',
        options={
            'initial_simplex': initial_simplex,
            'xatol': 0.01,
            'fatol': 0.01,
            'disp': True,
            'maxiter': 1000  # optional: increase if needed
        }
    )
    np.save('opt2_resbrute1_xycopk_minimize.npy',result)

# Resulting optimal parametersre
    popt = result.x

    print('start last opt')
    params = [ {'offset':offset,'scale':scale}, False,'xyzopk', imu]
    popt2 = optimize.fmin(residual2, tuple(popt), args=tuple(params), xtol=0.01, ftol=0.01, disp=False)
    np.save('opt2_resbrute2_xycopk_minimize.npy',popt2)
    sys.exit()


