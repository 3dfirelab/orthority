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
#homebrewed
from imuNcOoGeojson import imutogeojson
import tempfile
import orthority as oty


#########################################
def local_normalization(da, diskSize=30,):

    #idx = np.where(mask==1)
    #trange_ = [input_float[idx].min(), input_float[idx].max()]
    #img = convert_2_uint16(input_float, trange_ )
    input_float = np.array(da.data)
    idx = np.where(~np.isnan(input_float))
    idxnan = np.where(np.isnan(input_float))
    trange = [np.percentile(input_float[idx],20), np.percentile(input_float[idx],80)]
    input_float = np.where(input_float<trange[0],trange[0],input_float)
    input_float = np.where(input_float>trange[1],trange[1],input_float)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        img = skimage.img_as_uint((input_float-trange[0])/(trange[1]-trange[0]),force_copy=True)

        selem = skimage.morphology.disk(diskSize)
        #img_eq = skimage.filters.rank.equalize(img, selem=selem, mask=mask)
        img_eq = skimage.filters.rank.equalize(img, footprint=selem)
    
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
    return grad, mag, angle

#################################################
def img2da4residu(idimgs_,rrh):
    atr = xr.open_dataset(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimgs_))
    atr = atr.rio.reproject(daRef.rio.crs)
    da_ = atr.band_data.isel(band=1)
    da_coarse = da_.coarsen(dim={'x': 3, 'y': 3}, boundary="trim").mean()
    gradAtr,_,_ = get_gradient(da_coarse)
    dagradAtr = xr.DataArray(gradAtr, dims=["y", "x"], coords={"y": da_coarse.y, "x": da_coarse.x})

    dagradAtr_inter = dagradAtr.interp(x=daRef.x, y=daRef.y)
    #dagradAtr_inter = dagradAtr_inter/80
    #da_mask = dagradAtr_inter.where(~dagradAtr_inter.isnull(), -9999)
    #dagradAtr_inter = dagradAtr_inter.where(dagradAtr_inter<1,1)
    #dagradAtr_inter = dagradAtr_inter.where(da_mask!=-9999,np.nan)

    da1 = dagradAtr_inter.rio.write_crs(daRef.rio.crs)
    #da1 = da_coarse.interp(x=dagradS2.x, y=dagradS2.y)
    #da1 = da1.rio.write_crs(s2.rio.crs)
    
    #da1 = da1.rio.clip(mask.geometry.values, mask.crs)
    da1 = da1.rolling(x=rrh, y=rrh, center=True).mean()
    da1 = local_normalization(da1, diskSize=200,)

    atr = None
    da = None
    da_coarse = None
    dagradAtr = None
    dagradAtr_inter = None

    return da1

#################################################
def residual(args, *params):
   
    rrh = 3
    o, p, k = args[3:]
    xc, yc, zc = args[:3]
    flag_plot = params[0]
    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([xc,yc,zc])
    #atr
    #correction_xyz = np.array([0.,0.,0.]) # correction aricraft ref
    #correction_opk = np.array([0.,3.,-16]) # degree

    imutogeojson( indir, outdir, imufile, indirimg, flightname, correction_xyz, correction_opk)
 
    for idimg in idimgs:
        if os.path.isfile(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimg)):
            os.remove(indir+'as240051_20241113_103254-{:d}_ORTHO.tif'.format(idimg))
   
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
    
    #mask = gpd.read_file(indir+'mask.shp')

    #run orthorectification
    #result = subprocess.run(command, capture_output=True, text=True)
    #subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    #if process.info['status'] == psutil.STATUS_ZOMBIE:
    #     print(f"Zombie Process Found: PID={proc.info['pid']}, Name={proc.info['name']}")



    
    with tempfile.TemporaryFile() as tempf:
        process = subprocess.Popen(command, stdout=tempf)
        process.communicate()



    da1 = img2da4residu(idimgs[0],rrh)
    da2 = img2da4residu(idimgs[1],rrh)
    #da3 = img2da4residu(idimgs[2],rrh)
    #da4 = img2da4residu(idimgs[3],rrh)
    
    resi = abs((da1-da1Ref)).sum() + abs((da2-da2Ref)).sum() #+  abs((da3-daRef)).sum() + abs((da4-daRef)).sum()
 
    print('{:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f} | {:f}'.format(xc, yc, zc, o,p,k,resi) )
    
    if flag_plot:
        ax = plt.subplot(121)
        (da1-da1Ref).plot(ax=ax)
        ax = plt.subplot(122)
        (da2-da2Ref).plot(ax=ax)
        plt.figure()
        ax = plt.subplot(111)
        (da1Ref).plot(ax=ax,alpha=.5)
        (da1).plot.contour(ax=ax,colors='k')
        plt.figure()
        ax = plt.subplot(111)
        (da2Ref).plot(ax=ax,alpha=.5)
        (da2).plot.contour(ax=ax,colors='k')

        plt.show()
        pdb.set_trace()
    
    da1 = None
    da2 = None

    return resi


##################################
if __name__ == "__main__":
##################################
    indir = '/mnt/data/ATR42/as240051/' #'/home/paugam/Data/ATR42/as240051/'
    outdir = indir + 'io/'
    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    imgdirname = 'img'
    idimgs = [93,99]   
    #imgdirname = 'img2'
    #idimgs = [50,55,60,65]   

    indirimg = indir + '{:s}/'.format(imgdirname)
    flightname = 'as240051'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")         

    #sentinel
    
    s2 = xr.open_dataset(indir+'sentinel_background_test_cropped_{:s}.tif'.format(imgdirname))
    gradS2,_,_ = get_gradient(s2.band_data.isel(band=1))
    dagradS2 = xr.DataArray(gradS2, dims=["y", "x"], coords={"y": s2.y, "x": s2.x})

    rr = 3
    #dagradS2 = dagradS2/1000.
    #dagradS2 = dagradS2.where(dagradS2<1,1)
    daRef = dagradS2
    daRef = daRef.rolling(x=rr, y=rr, center=True).mean()
    daRef = local_normalization(daRef, diskSize=200,)
    daRef = daRef.rio.write_crs(s2.rio.crs)
   

    rr = 3
    da1R =  xr.open_dataset(indir+'img_manualOrtho/as240051_20241113_103254-93.tif')
    da1R = da1R.coarsen(dim={'x': 3, 'y': 3}, boundary="trim").mean()
    gradda1R,_,_ = get_gradient(da1R.band_data.isel(band=1))
    dagradda1R = xr.DataArray(gradda1R, dims=["y", "x"], coords={"y": da1R.y, "x": da1R.x})
    dagradda1R = dagradda1R.interp(x=daRef.x, y=daRef.y)
    da1Ref = dagradda1R.rolling(x=rr, y=rr, center=True).mean()
    da1Ref = local_normalization(da1Ref, diskSize=200,)
    da1Ref = da1Ref.rio.write_crs(da1R.rio.crs)
    da1Ref = da1Ref.fillna(1)

    da2R =  xr.open_dataset(indir+'img_manualOrtho/as240051_20241113_103254-99.tif')
    da2R = da2R.coarsen(dim={'x': 3, 'y': 3}, boundary="trim").mean()
    gradda2R,_,_ = get_gradient(da2R.band_data.isel(band=1))
    dagradda2R = xr.DataArray(gradda2R, dims=["y", "x"], coords={"y": da2R.y, "x": da2R.x})
    dagradda2R = dagradda2R.interp(x=daRef.x, y=daRef.y)
    da2Ref = dagradda2R.rolling(x=rr, y=rr, center=True).mean()
    da2Ref = local_normalization(da2Ref, diskSize=200,)
    da2Ret = da2Ref.rio.write_crs(da2R.rio.crs)
    da2Ref = da2Ref.fillna(1)

    rranges = (slice(-2,2,.5), slice(-2,2,.5), slice(-2,2,.5), 
               slice(-2, -1, .2), slice(2, 4, .5), slice(-19, -13, 1))

    params = [False]
    resbrute = optimize.brute(residual, rranges, args=params, full_output=False,
                              finish=None)

    #residual([ 2.10512566e-04,  1.70885594e-04,  2.61422540e-04, 
    #          -5.52140251e-01, 3.34409679e+00, -1.62949875e+01], [True])
    #sys.exit()

    #resbrute.append([-1.69183706,   3.06752315, -14.98225642])
    #resbrute = []
    #resbrute.append([0,0,0, -0.63767196,   3.36609474, -16.31209226])
    popt = optimize.fmin(residual, tuple(resbrute[0]), args=tuple(params), xtol=0.1, ftol=1.e0, disp=False)


