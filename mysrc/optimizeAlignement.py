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

#homebrewed
import imuNcOoGeojson  
importlib.reload(imuNcOoGeojson)


#########################################
def local_normalization(da, diskSize=30,):

    #idx = np.where(mask==1)
    #trange_ = [input_float[idx].min(), input_float[idx].max()]
    #img = convert_2_uint16(input_float, trange_ )
    input_float = np.array(da.data,dtype=np.float32)
    idx = np.where(~np.isnan(input_float))
    idxnan = np.where(np.isnan(input_float))
    mask = np.zeros_like(input_float)
    mask[idx] = 1
    try: 
        trange = [np.percentile(input_float[idx],20), np.percentile(input_float[idx],80)]
    except: 
        pdb.set_trace()
    input_float = np.where(input_float<trange[0],trange[0],input_float)
    input_float = np.where(input_float>trange[1],trange[1],input_float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        input_float[idxnan] = trange[1]
        img = skimage.img_as_uint((input_float-trange[0])/(trange[1]-trange[0]))

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
def img2da4residu(rrh, atr, da1Ref):
   
    #with  xr.open_dataset(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimgs_)) as atr: 
        #atr = atr.rio.reproject(daRef.rio.crs)
        
    da1 = atr.band_data.isel(band=1)
    da1 = da1.coarsen(dim={'x': 4, 'y': 4}, boundary="trim").mean()
    da1 = xr.DataArray(get_gradient(da1), dims=["y", "x"], coords={"y": da1.y, "x": da1.x})
    da1 = da1.interp(x=da1Ref.x, y=da1Ref.y)
    #dagradAtr_inter = dagradAtr_inter/80
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

    da1 = da1.fillna(0) 

    return da1

#################################################
def residual(args, *params):
   
    #mem0 = psutil.virtual_memory()
    #rrh = 3
    
    if len(args)==6: 
        o, p, k = args[3:]
        xc, yc, zc = args[:3]
    elif len(args)==3: 
        o, p, k = args[:]
        xc, yc, zc = params[1:4]

    flag_plot = params[0]
    demFile =  '{:s}/dem/dem.tif'.format(indir)

    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([xc,yc,zc])
    #atr
    #correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    #correction_opk = np.array([0.,3.,-16]) # degree
    src_files = ["/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimg) for idimg in idimgs] 

    str_tag = '_{:.1f}{:.1f}{:.1f}_{:.1f}{:.1f}{:.1f}'.format(*correction_xyz,*correction_opk).replace('.','p')
    imuNcOoGeojson.imutogeojson( indir, wkdir, imufile, indirimg, flightname, correction_xyz, correction_opk, src_files,str_tag=str_tag)

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

            
    extparamFile =  "{:s}/as240051_ext_param{:s}.geojson".format(wkdir,str_tag)
    #create a camera model for src_file from interior & exterior parameters
    cameras = oty.FrameCameras(intparamFile, extparamFile)

    #for idimg in idimgs[:1]:
    idimg = idimgs[0]
    src_file =  "/{:s}/{:s}_masked/as240051_20241113_103254-{:d}.tif".format(indir,imgdirname,idimg) 
    camera = cameras.get(src_file)
    
    # create Ortho object and orthorectify
    ortho = oty.Ortho(src_file, demFile, camera=camera, crs=cameras.crs)
    ortho.process( wkdir+'as240051_20241113_103254-{:d}_ORTHO{:s}.tif'.format(idimg,str_tag), overwrite=True)
    del ortho, camera
    del cameras
    
    
    resi = float(1.0)
    #mem1 = psutil.virtual_memory()
    atr = xr.open_dataset(wkdir+'as240051_20241113_103254-{:d}_ORTHO{:s}.tif'.format(idimgs[0],str_tag))
    #atr2 = atr.rio.reproject(daRef.rio.crs)
    #del atr
    da1 = img2da4residu(rr,atr,da1Ref)
    #da2 = img2da4residu(idimgs[1],rrh)
    resi = float(abs((da1-da1Ref)).sum()) #+ abs((da2-da2Ref)).sum() )#+  abs((da3-daRef)).sum() + abs((da4-daRef)).sum()
  

    #del da2 
    #gc.collect()
    #mem2 = psutil.virtual_memory()
    #print(mem2.used/mem1.used)
    
    mem = psutil.virtual_memory()
    print('{:.1f},{:.1f},{:.1f}  {:.1f},{:.1f},{:.1f} | {:.1f} || {:.1f} '.format(xc, yc, zc, o, p, k,resi, mem.used/(1024 ** 3)))
    
    '''snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')

    for stat in top_stats[:3]:
        print(stat)
    '''
    
    if flag_plot:
        ax = plt.subplot(111)
        (da1-da1Ref).plot(ax=ax)
        #ax = plt.subplot(122)
        #(da2-da2Ref).plot(ax=ax)
        plt.figure()
        ax = plt.subplot(111)
        (da1Ref).plot(ax=ax,alpha=.5)
        (da1).plot.contour(ax=ax,colors='k')
        #plt.figure()
        #ax = plt.subplot(111)
        #(da2Ref).plot(ax=ax,alpha=.5)
        #(da2).plot.contour(ax=ax,colors='k')

        plt.show()
        pdb.set_trace()
  
    if not(flag_plot):
        os.remove(extparamFile)
        os.remove(wkdir+'as240051_20241113_103254-{:d}_ORTHO{:s}.tif'.format(idimgs[0],str_tag))
    
    del atr,da1

    return float(resi)


##################################
if __name__ == "__main__":
##################################
    import tracemalloc
    tracemalloc.start()
    

    indir = '/mnt/data/ATR42/as240051/' #'/home/paugam/Data/ATR42/as240051/'
    outdir = indir + 'io/'
    wkdir = '/tmp/orthority2/'
    os.makedirs(wkdir, exist_ok=True)

    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    #imgdirname = 'img'
    #idimgs = [93]   
    imgdirname = 'img2'
    idimgs = [65]   

    indirimg = indir + '{:s}/'.format(imgdirname)
    flightname = 'as240051'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")         

    intparamFile = "{:s}/io/as240051_int_param.yaml".format(indir)

    #sentinel
    s2 = xr.open_dataset(indir+'sentinel_background_test_cropped_{:s}.tif'.format(imgdirname))
    gradS2 = get_gradient(s2.band_data.isel(band=1))
    dagradS2 = xr.DataArray(gradS2, dims=["y", "x"], coords={"y": s2.y, "x": s2.x})

    rr = 3
    #dagradS2 = dagradS2/1000.
    #dagradS2 = dagradS2.where(dagradS2<1,1)
    daRef = dagradS2
    daRef = daRef.rolling(x=rr, y=rr, center=True).mean()
    daRef = local_normalization(daRef, diskSize=200,)
    daRef = daRef.rio.write_crs(s2.rio.crs)
   

    rr = 3
    da1R =  xr.open_dataset(indir+'img_manualOrtho/as240051_20241113_103254-93_masked.tif')
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
    
    da1Ref = img2da4residu(rr,da1R,da1R)
    
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

    if True: 
        #popt = np.array([ -0.93871095,  -1.06893665,  -1.03742455,  -1.56363453, 2.64046062, -15.92574779])
        #popt = np.array([-0.96509656,  -1.02242933,  -1.02461382,  -1.54239457, 2.6027513 , -15.99057932])
        #popt = np.array([0.,0.,0.,0, 2.,0 ])
        popt = np.load('resbrute22.npy')
        residual( popt , True)
        sys.exit()
    '''
    rranges = (slice(-2,2,1), slice(-2,2,1), slice(-2,2,1), 
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
    xc,yc,zc,oc,pc,kc = 0,0,0, 0,0,0

    rranges = (slice(oc-3,oc+3,.5), slice(pc-3,pc+3,.5), slice(kc-3,  kc+3,  .5))
    params = [False,xc,yc,zc]
    print('start second opt')
    resbrute2 = optimize.brute(residual, rranges, args=params, full_output=False,
                              finish=None)#, workers=16 )
    
    print(resbrute2)
    np.save('resbrute21.npy',resbrute2)
    resbrutef =  [xc,yc,zc] + list(resbrute2)

    print('start last opt')
    popt = optimize.fmin(residual, tuple(resbrutef), args=tuple(params), xtol=0.1, ftol=1.e0, disp=False)

    np.save('resbrute22.npy',popt)

    #resbrute.append([-1.69183706,   3.06752315, -14.98225642])
    #resbrute = []
    #resbrute.append([0,0,0, -0.63767196,   3.36609474, -16.31209226])


