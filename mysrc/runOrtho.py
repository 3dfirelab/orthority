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
import shutil


#homebrewed
from imuNcOoGeojson import imutogeojson

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
def orthro(args):
   
    x, y, z = args[:3]
    o, p, k = args[3:]

    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([x,y,z])

    imutogeojson( indir, outdirIO, imufile, indirimg, flightname, correction_xyz, correction_opk)
 
    if os.path.isdir(outdir):
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
   
    command = [
            "oty", "frame",
            "--dem", '{:s}/dem/dem.tif'.format(indir),
            "--int-param", "{:s}/as240051_int_param.yaml".format(outdirIO),
            "--ext-param", "{:s}/as240051_ext_param.geojson".format(outdirIO),
            "--out-dir", outdir,
            "-o", 
            "/{:s}as240051_20241113_103254-*.tif".format(indirimg), 
            ]
    pdb.set_trace() 
    #run orthorectification
    result = subprocess.run(command, capture_output=True, text=True)

    return 'done'


##################################
if __name__ == "__main__":
##################################
    indir = '/mnt/data/ATR42/as240051/'
    indirimg = indir + 'img2/'
    outdirIO = indir + 'io2/'
    outdir = indir+'ortho2/'

    imufile = 'SCALE-2024_SAFIRE-ATR42_SAFIRE_CORE_NAV_100HZ_20241113_as240051_L1_V1.nc'
    flightname = 'as240051'
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")

    #from popt = optimize.fmin(residual, tuple([-1, 3.5, -15]), args=tuple(params), xtol=5, ftol=1.e-4, disp=False
    #correction_opk = [-1.69183706,   3.06752315, -14.98225642] 
    correction_xyz = [2.10512566e-04,  1.70885594e-04,  2.61422540e-04, ]
    correction_opk = [-5.52140251e-01,3.34409679e+00, -1.62949875e+01]
    
    orthro([*correction_xyz,*correction_opk])


