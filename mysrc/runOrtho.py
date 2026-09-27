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
import importlib
import orthority as oty
import glob 
import tifffile
import json 
import re
from pathlib import Path
import tempfile
import rioxarray  # ensures .rio accessor is registered
from rasterio.enums import Resampling
from osgeo import gdal
import subprocess
import rasterio
from orthority.ortho import OrthorityWarning
import pandas as pd 
import datetime
import argparse
import importlib
import socket
import yaml

#homebrewed
import imuNcOoGeojson 
#import optimizeAlignement_telops_function 
#importlib.reload(optimizeAlignement_telops_function)
#import optimizeAlignement_telops_f1


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


#########################################
def radiance_um_to_wavenumber(lambda_um, L_lambda):
    """
    Convert spectral radiance from per micrometre to per wavenumber.

    Parameters:
    ----------
    lambda_um : float or np.ndarray
        Wavelength(s) in micrometres (μm).
    L_lambda : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·μm⁻¹.

    Returns:
    -------
    L_wavenumber : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·(cm⁻¹)⁻¹.
    """
    lambda_um = np.asarray(lambda_um)
    L_lambda = np.asarray(L_lambda)
    return L_lambda * (lambda_um ** 2) / 1e4


def radiance_wavenumber_to_um(lambda_um, L_wavenumber):
    """
    Convert spectral radiance from per wavenumber to per micrometre.

    Parameters:
    ----------
    lambda_um : float or np.ndarray
        Wavelength(s) in micrometres (μm).
    L_wavenumber : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·(cm⁻¹)⁻¹.

    Returns:
    -------
    L_lambda : float or np.ndarray
        Radiance in W·m⁻²·sr⁻¹·μm⁻¹.
    """
    lambda_um = np.asarray(lambda_um)
    L_wavenumber = np.asarray(L_wavenumber)
    return L_wavenumber * 1e4 / (lambda_um ** 2)


#########################################
# Define affine model
def affine(x, m, p):
    return m * x + p


#################################################
def copy_source_exif_metadata(source_path, output_path):
    """Copy acquisition EXIF tags into the final ortho TIFF."""
    exiftool = shutil.which("exiftool")
    if exiftool is None:
        print("WARNING: exiftool not found; Date/Time Original was not written.")
        return
    command = [
        exiftool, "-overwrite_original", "-TagsFromFile", str(source_path),
        "-EXIF:DateTimeOriginal", "-EXIF:SubSecTimeOriginal",
        "-EXIF:ExposureTime", "-EXIF:ExifVersion", str(output_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)


def read_source_metadata(path):
    """Read acquisition metadata that should survive ortho raster rewriting."""
    preserved = {}
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        date_tag = page.tags.get("DateTime")
        if date_tag is not None:
            preserved["TIFFTAG_DATETIME"] = str(date_tag.value)
        exif_tag = page.tags.get("ExifTag")
        exif = exif_tag.value if exif_tag is not None and isinstance(exif_tag.value, dict) else {}
        for source_name, output_name in (("DateTimeOriginal", "DateTimeOriginal"),
                                         ("SubsecTimeOriginal", "SubSecTimeOriginal"),
                                         ("ExposureTime", "ExposureTime"),
                                         ("ExifVersion", "ExifVersion")):
            if source_name in exif:
                value = exif[source_name]
                if isinstance(value, tuple) and len(value) == 2:
                    value = value[0] / value[1] if value[1] else value[0]
                preserved[output_name] = str(value)
    return preserved


def orthro(
    args,
    transectname,
    flightname,
    flightdate,
    indir,
    indirimg,
    outdir,
    wkdir,
    imufile,
    demFile,
    intparamFile,
    filtre=1,
    image_glob=None,
    pose_model=imuNcOoGeojson.DEFAULT_POSE_MODEL,
    drift_model=None,
    time_shift_to_add_to_image=0.0,
):
   
    x, y, z = args[:3]
    o, p, k = args[3:]

    correction_opk = np.array([o,p,k])
    correction_xyz = np.array([x,y,z])
    image_pattern = image_glob or f"f{filtre}*.tif"
    src_files = sorted(glob.glob(f"{indirimg}/{image_pattern}"))
    if not src_files:
        raise ValueError(f"No input images found in {indirimg}")


   
    '''
    command = [
            "oty", "frame",
            "--dem", '{:s}/dem/dem.tif'.format(indir),
            "--int-param", "{:s}/as240051_int_param.yaml".format(outdirIO),
            "--ext-param", "{:s}/as240051_ext_param.geojson".format(outdirIO),
            "--out-dir", outdir,
            "-o", 
            "/{:s}as240051_20241113_103254-*.tif".format(indirimg), 
            ]
    print(' '.join(command))
    #run orthorectification
    result = subprocess.run(command, capture_output=True, text=True)
    '''
    
    #imu = xr.open_dataset(indir+imufile)
    if Path(imufile).suffix.lower() == ".nc":
        # SAFIRE NetCDF navigation product. Keep it as xarray because
        # imutogeojson supports xarray interpolation directly.
        imu = xr.open_dataset(imufile)
        rename = {
            "HEIGHT_WGS84": "ALTITUDE",
            "ROLL": "ROLL_smooth",
            "PITCH": "PITCH_smooth",
            "THEAD": "THEAD_smooth",
        }
        imu = imu.rename({key: value for key, value in rename.items() if key in imu})
    else:
        imu = gpd.read_file(imufile)
        imu = imu.dropna(subset=["latitude"])
        imu = imu.rename(columns={
            "datetime_utc": "time",
            "latitude": "LATITUDE",
            "longitude": "LONGITUDE",
            "altitude_m": "ALTITUDE",
            "roll_deg": "ROLL_smooth",
            "pitch_deg": "PITCH_smooth",
            "heading_deg": "THEAD_smooth",
        })

    #for idimg in idimgs[:1]:
    src_files = sorted(glob.glob(f"{indirimg}/{image_pattern}"))
    df_calib_f = None
    if filtre >= 3:
        calibration_csv = (
            Path(indir).parents[1]
            / "TelposDLCalib"
            / "SILEX_telops_filtre_DL_fit.csv"
        )
        df_calib = pd.read_csv(calibration_csv)
        df_calib_f = df_calib[df_calib.filtre == filtre]
    correction_opk_arr = []
    correction_opk_time_arr = []

    #maskPlume_file = f'/data/shared/ATR42/{flightname}/mask/plumeMask_{transectname}.gpkg'
    #maskPlume = gpd.read_file(maskPlume_file)
    maskPlume = None
    for src_file in src_files:
        if os.path.isfile(outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif')): continue
        
        print(os.path.basename(src_file))
        base = os.path.basename(src_file)       # "f1-000000001.tif"
        id_str = base.replace(f"f{filtre}-", "").replace(".tif", "")
        frame_match = re.search(r"-(\d+)$", id_str)
        frame_id = int(frame_match.group(1)) if frame_match else int(id_str)    
    
        source_metadata = read_source_metadata(src_file)
        with tifffile.TiffFile(src_file) as tif:
            # Get ImageDescription tag
            description_tag = tif.pages[0].tags.get("ImageDescription")
            description = description_tag.value if description_tag is not None else None
            # Some acquisition TIFFs have no JSON ImageDescription.
            metadata = json.loads(description) if description else {}
            # Raw frames without exposure metadata are already treated as
            # linear image values by this workflow.
            exposure_time = float(metadata.get("ExposureTime", 1.0))
            # Read image data
            data = tif.asarray()
        
        if filtre>=3:
            #convert DL to Radiance
            #-----------------
            rad_cm = affine( data/exposure_time, df_calib_f.m.values, df_calib_f.p.values)
            rad_lambda = radiance_wavenumber_to_um( df_calib_f['lambda'] , rad_cm)
            
            # sace to tmp before ortho
            #-----------------
            tif_path_ = f"{wkdir}/{base}".replace('.tif','_rad.tif')
            # Save as float32 TIFF with metadata
            tifffile.imwrite(
                tif_path_,
                rad_lambda.astype('float32'),
                metadata=metadata
            )
            src_file_ = tif_path_

        else: 
            data = data/exposure_time
            tif_path_ = f"{wkdir}/{base}".replace('.tif','_expcorr.tif')
            # Save as float32 TIFF with metadata
            tifffile.imwrite(
                tif_path_,
                data.astype('float32'),
                metadata=metadata
            )
            src_file_ = tif_path_

        d_id_update = 50 #100
        '''
        if False: #(frame_id >= d_id_update) & (frame_id % d_id_update == 0 ):
            print('###############')
            print(src_file_)
            print('ref id:', frame_id- (d_id_update-1)) 
            correction_opk_copy = correction_opk.copy()
            oc, pc, kc =  ( np.array(correction_opk) + offset[1] ) / scale[1]
            result, params_ = optimizeAlignement_telops_f1.run_opt_correction(frame_id-(d_id_update-1), src_file_, transectname, flightname, flightdate, oc, pc, kc, maskPlume=maskPlume)

            [oc,pc,kc] = result.x
            #xc,yc,zc = 0.5,0.5,0.5
            #correction_xyz = (np.array([xc,yc,zc]) * scale[0]) - offset[0]
            correction_opk = (np.array([oc,pc,kc]) * scale[1]) - offset[1]
            print('modif of correction:')
            print(np.array(correction_opk)-np.array(correction_opk_copy))
            correction_opk_arr.append(correction_opk)
            try:
                time_ =datetime.datetime.strptime(metadata['Time'], "%Y-%m-%d %H:%M:%S.%f")
            except: 
                time_ = datetime.datetime.strptime( metadata['Time'], "%Y-%m-%d %H:%M:%S")
            correction_opk_time_arr.append(time_)
            print('###############')
            ##plot
            #params_['flag_plot'] = True
            #optimizeAlignement_telops_f1.residual( result.x , params_ )
        '''
        print(
            f"correction_xyz={np.asarray(correction_xyz, dtype=float).tolist()} "
            f"correction_opk={np.asarray(correction_opk, dtype=float).tolist()}"
        )
        print(src_file_)
        print('process imu ...')
       
        imuNcOoGeojson.imutogeojson(
            imu, wkdir, indirimg, flightname,
            correction_xyz, correction_opk, [src_file_],
            pose_model=pose_model,
            drift_model=drift_model,
            time_shift_to_add_to_image=time_shift_to_add_to_image,
        )
        print('done                ') 

        str_tag = ''
        extparamFile =  f"{wkdir}/{flightname}_ext_param{str_tag}.geojson".format(wkdir,str_tag)
        #create a camera model for src_file from interior & exterior parameters
        cameras = oty.FrameCameras(intparamFile, extparamFile)

        camera = cameras.get(src_file_)
        # create Ortho object and orthorectify
        ortho = oty.Ortho(src_file_, demFile, camera=camera, crs=cameras.crs)
        out_file_ = outdir+os.path.basename(src_file_).replace('.tif','_ORTHO.tif')
        ortho.process(out_file_, overwrite=True)
        del ortho, camera
        
        to_epsg4326_inplace( outdir+os.path.basename(src_file_).replace('.tif','_ORTHO.tif') )
       
        
        #remove_halo(         outdir+os.path.basename(src_file).replace('.tif','_ORTHO.tif') )
        
        #add meta data to ortho tif

        # read existing image and metadata
        with rasterio.open(out_file_, "r") as src:
            img = src.read()
            profile = src.profile
            existing_meta = src.tags()

        # Merge output metadata, Telops JSON metadata, and source acquisition
        # tags after orthorectification has created a new raster.
        existing_meta.update(source_metadata)
        existing_meta.update(metadata)

        # write back preserving CRS and transform
        with rasterio.open(out_file_, "w", **profile) as dst:
            dst.write(img)
            dst.update_tags(**existing_meta)

        # Rasterio/GDAL does not write EXIF DateTimeOriginal. Copy the actual
        # EXIF tags after the ortho GeoTIFF has been written.
        copy_source_exif_metadata(src_file, out_file_)

    #save correction history
    df = pd.DataFrame({
                    "time": correction_opk_time_arr,
                    "correction_opk": correction_opk_arr
                     })                                    
    df.to_csv(f"{indir}/correction_opk_{transectname}.csv", index=False)

    del cameras

    return 'done'


####################
import os
import numpy as np
import rasterio

def remove_halo(input_path, white_thresh=220, chroma_thresh=8):
    """
    Detect near-white, low-chroma 'halo' pixels and store them in the dataset mask.
    Overwrites the GeoTIFF in place (atomic replace via temp file).

    Parameters
    ----------
    input_path : str
        Path to a GeoTIFF (expects >=3 bands interpreted as RGB).
    white_thresh : int
        Threshold (0–255) per-channel to be considered 'white-ish'.
    chroma_thresh : int
        Max channel spread to be considered low chroma (greyish).
    """
    tmp_path = input_path + ".tmp"

    with rasterio.open(input_path) as src:
        arr = src.read()            # (bands, H, W)
        profile = src.profile

    if arr.shape[0] < 3:
        raise ValueError("Expected at least 3 bands (RGB).")

    # Compute background mask from first three bands
    R, G, B = arr[:3].astype(np.uint16)
    #near_white = (R > white_thresh) & (G > white_thresh) & (B > white_thresh)
    alpha = arr[3]
    mm = alpha < white_thresh
    low_chroma = (np.maximum.reduce([R, G, B]) - np.minimum.reduce([R, G, B]) < chroma_thresh)
    #bg = near_white & low_chroma  # True = halo/background to hide
    bg = mm #& low_chroma  # True = halo/background to hide

    # Create 8-bit dataset mask: 0 = masked (transparent), 255 = valid
    ds_mask = np.where(bg, 0, 255).astype(np.uint8)

    arr[:,bg]=0
    # Write a new file with original bands and the computed mask
    with rasterio.open(tmp_path, "w", **profile) as dst:
        dst.write(arr)                 # write all original bands
        #dst.write_mask(ds_mask)        # set dataset mask

    os.replace(tmp_path, input_path)

##################################
def to_epsg4326_inplace(path, compress="LZW"):
    
    # Reproject to EPSG:4326
    gdal.Warp(
        path,
        path,
        dstSRS='EPSG:4326',
        resampleAlg='near',
        srcNodata=0,      # or the actual nodata in your file
        dstNodata=0
        )

    return None

##################################
if __name__ == "__main__":
##################################
    #importlib.reload(optimizeAlignement_telops_f1)
    importlib.reload(imuNcOoGeojson)
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Dataset YAML, for example config/config-cimenterie-01.yaml.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=None,
        help=(
            "JSON produced by calibrate_orthority_imu.py. When supplied, its "
            "correction_xyz and correction_opk override the YAML calibration."
        ),
    )
    parser.add_argument(
        "--drift-model",
        type=Path,
        default=None,
        help="CSV with time, extra_omega, extra_phi, and extra_kappa columns.",
    )
    parser.add_argument(
        "--time-shift-to-add-to-image",
        type=float,
        default=None,
        help="Seconds added to each image timestamp before IMU interpolation (e.g. 2).",
    )
    parser.add_argument(
        "--imu-source",
        type=str,
        default=None,
        help=(
            "Select an entry when 'imu' in the dataset YAML is a mapping of "
            "source name to path, e.g. 'safire' or 'loa'. Not needed when "
            "'imu' is a single path."
        ),
    )
    args = parser.parse_args()

    with args.config.open("r", encoding="utf-8") as config_file:
        cfg = yaml.safe_load(config_file)

    flightname = cfg["flightname"]
    flightdate = cfg["flightdate"]
    transectname = cfg["extractionName"]
    filtre = int(cfg.get("filter", 1))
    time_shift_to_add_to_image = (
        args.time_shift_to_add_to_image
        if args.time_shift_to_add_to_image is not None
        else float(cfg.get("time_shift_to_add_to_image", 0.0))
    )
    data_root = Path(cfg["dirTelops"]).expanduser().resolve().parent
    # A YAML label is useful even when ``imu`` is a single direct path: it
    # names temporary/intermediate products without selecting another file.
    imu_selector = args.imu_source
    if imu_selector is None and isinstance(cfg.get("imu"), dict):
        imu_selector = cfg.get("imu_source")
    imufile, resolved_source = imuNcOoGeojson.resolve_imu_path(
        cfg, args.config.resolve().parent, imu_selector
    )
    imu_source = resolved_source or cfg.get("imu_source")
    if "<imu>" in transectname:
        transectname = transectname.replace("<imu>", imu_source or "")
    indir = data_root / "Transects" / transectname

    def config_path(key):
        if key not in cfg:
            raise ValueError(f"Missing required configuration key: {key}")
        raw_path = str(cfg[key])
        if "<imu>" in raw_path:
            if imu_source is None:
                raise ValueError(f"{key} uses <imu>; pass --imu-source.")
            raw_path = raw_path.replace("<imu>", imu_source)
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = args.config.resolve().parent / path
        return path.resolve()

    indirimg = config_path("input_dir")
    outdir = config_path("output_dir")
    if imu_selector is not None and "<imu>" not in str(cfg["output_dir"]):
        # Put each source in its sibling transect, rather than retaining the
        # source named in extractionName for every run.
        transect_prefix = transectname.rsplit("_", 1)[0]
        source_transect = f"{transect_prefix}_{imu_source}"
        outdir = outdir.parent.with_name(source_transect) / (
            f"{outdir.name}_{imu_source}"
        )
    # Prefer explicit paths from the dataset configuration. The fallback
    # keeps compatibility with the older PIPER directory layout.
    demFile = (
        config_path("dem")
        if "dem" in cfg
        else data_root / "dem" / f"{flightname}_dem_1m.tif"
    )
    intparamFile = (
        config_path("int_param")
        if "int_param" in cfg
        else indir / "io" / f"{flightname}_int_param.yaml"
    )
    calibration_path = args.calibration
    if calibration_path is None and cfg.get("calibration"):
        calibration_path, _ = imuNcOoGeojson.resolve_imu_path(
            cfg, args.config.resolve().parent, imu_selector, key="calibration"
        )

    for label, path in (
        ("input directory", indirimg),
        ("IMU", imufile),
        ("DEM", demFile),
        ("interior parameters", intparamFile),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    if os.path.isdir(outdir):
        print("Removing existing ortho output directory:", outdir)
        shutil.rmtree(outdir)
    os.makedirs(outdir, exist_ok=True)
    print("Created fresh ortho output directory:", outdir)

    wkdir = Path(f"/tmp/orthority_wkdir_ortho_{transectname}_{imu_source or 'default'}")
    if os.path.isdir(wkdir): shutil.rmtree(wkdir)
    os.makedirs(wkdir, exist_ok=True)
    
    warnings.filterwarnings("ignore", category=UserWarning, module="pyproj")
    warnings.filterwarnings("ignore", category=OrthorityWarning)  # show once per message

    if calibration_path:
        with calibration_path.open("r", encoding="utf-8") as calibration_file:
            calibration = json.load(calibration_file)
        pose_model = calibration.get("pose_model")
        if pose_model not in imuNcOoGeojson.SUPPORTED_POSE_MODELS:
            raise ValueError(
                f"{calibration_path} has no supported pose_model. "
                "Re-run the calibration."
            )
        try:
            correction_xyz = np.asarray(calibration["correction_xyz"], dtype=float)
            correction_opk = np.asarray(calibration["correction_opk"], dtype=float)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                f"{calibration_path} has no valid correction_xyz/correction_opk."
            ) from error
        if correction_xyz.shape != (3,) or correction_opk.shape != (3,):
            raise ValueError("Calibration corrections must each contain 3 values.")
    else:
        pose_model = imuNcOoGeojson.DEFAULT_POSE_MODEL
        correction_xyz = np.asarray(cfg.get("correction_xyz", [0.0, 0.0, 0.0]), dtype=float)
        correction_opk = np.asarray(cfg.get("correction_opk", [0.0, 0.0, 0.0]), dtype=float)
        if correction_xyz.shape != (3,) or correction_opk.shape != (3,):
            raise ValueError("Configured correction_xyz and correction_opk must each contain 3 values.")
        print("No calibration specified; using corrections from YAML.")

    print("input:", indirimg)
    print("output:", outdir)
    print("imu:", imufile, f"(source={imu_source})" if imu_source else "")
    print("correction_xyz:", correction_xyz)
    print("correction_opk:", correction_opk)
    print("pose_model:", pose_model)
    print("time_shift_to_add_to_image:", time_shift_to_add_to_image, "s")
    if time_shift_to_add_to_image != 0.0:
        print("###############")
        print(
            "WARNING: IMAGE TIME SHIFT APPLIED: "
            f"image timestamps will be shifted by {time_shift_to_add_to_image:+.6f} s "
            "before IMU interpolation."
        )
        print("###############")
    orthro(
        [*correction_xyz, *correction_opk],
        transectname,
        flightname,
        flightdate,
        str(indir),
        str(indirimg),
        str(outdir) + os.sep,
        str(wkdir),
        str(imufile),
        str(demFile),
        str(intparamFile),
        filtre=filtre,
        image_glob=cfg.get("image_glob"),
        pose_model=pose_model,
        drift_model=args.drift_model,
        time_shift_to_add_to_image=time_shift_to_add_to_image,
    )
    
