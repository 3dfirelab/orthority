"""
   Copyright 2023 Dugal Harris - dugalh@gmail.com

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

# Adapted from the OpenSFM exif module https://github.com/mapillary/OpenSfM/blob/main/opensfm/exif.py

import logging
from pathlib import Path
from typing import Dict, Tuple, Union
from xml.etree import cElementTree as ET

import numpy as np
import rasterio as rio

logger = logging.getLogger(__name__)

# Create a schema of known RPY XMP keys. Use xml namspace qualified keys which are unique, rather than xmltodict
# type prefix qualified keys which can have different prefixes referring to the same namepace.
rpy_schemas = dict(
    dji=dict(
        rpy_keys=[
            '{http://www.dji.com/drone-dji/1.0/}GimbalRollDegree',
            '{http://www.dji.com/drone-dji/1.0/}GimbalPitchDegree',
            '{http://www.dji.com/drone-dji/1.0/}GimbalYawDegree'
        ],
        rpy_offsets=(0., 90., 0.)
    ),
    # these Sony DSC keys may refer to RPY of the drone, not camera, but am including for now
    sensefly=dict(
        rpy_keys=[
            '{http://ns.sensefly.com/Camera/1.0/}Roll',
            '{http://ns.sensefly.com/Camera/1.0/}Pitch',
            '{http://ns.sensefly.com/Camera/1.0/}Yaw'
        ],
        rpy_offsets=(0., 0., 0.)
    ),
)


def xml_to_flat_dict(xmp_str: str) -> Dict[str, str]:
    """ Convert the given XML string to a flat dictionary.  """
    etree = ET.fromstring(xmp_str)
    flat_dict = {}

    def traverse_etree(etree):
        """
        Traverse the given XML tree, populating flat_dict with xml element (tag, text) and attribute (name, value)
        pairs.
        """
        flat_dict[etree.tag] = etree.text
        if etree.attrib:
            flat_dict.update(**etree.attrib)
        for child in etree.findall("./*"):
            traverse_etree(child)

    traverse_etree(etree)
    return flat_dict


class Exif:
    def __init__(self, filename: Union[str, Path]):
        """
        Class to extract camera model relevant EXIF and XMP values from an image.

        Parameters
        ----------
        filename: str, Path
            Path to the image file.
        """
        file_path = Path(filename)
        if not file_path.exists():
            raise FileNotFoundError(f'File does not exist: {file_path}')
        with rio.open(filename, 'r') as ds:
            self._exif_dict = ds.tags()
            self._image_size = ds.shape[::-1]

            namespaces = ds.tag_namespaces()
            if 'xml:XMP' in ds.tag_namespaces():
                xmp_str = ds.tags(ns='xml:XMP')['xml:XMP']
                self._xmp_dict = xml_to_flat_dict(xmp_str)
            else:
                logger.warning(f'{file_path.name} contains no XMP metadata')
                self._xmp_dict = None

        self._sensor_size = self._get_sensor_size(self._exif_dict, self._image_size)
        self._focal, self._focal_35 = self._get_focal(self._exif_dict)
        self._lla = self._get_lla(self._exif_dict)
        self._rpy = self._get_rpy(self._xmp_dict)

    @property
    def image_size(self) -> Union[None, Tuple[int, int]]:
        """ Image (width, height) in pixels. """
        return self._image_size

    @property
    def sensor_size(self) -> Union[None, Tuple[float, float]]:
        """ Sensor (width, height) in mm. """
        return self._sensor_size

    @property
    def focal(self) -> Union[None, float]:
        """ Focal length in mm. """
        return self._focal

    @property
    def focal_35(self) -> Union[None, float]:
        """ 35mm equivalent focal length in mm. """
        return self._focal_35

    @property
    def lla(self) -> Union[None, float]:
        """
        (Latitude, longitude, altitude) co-ordinates with latitude and longitude in decimal degrees, and altitude in
        meters.
        """
        return self._lla

    @property
    def rpy(self) -> Union[None, float]:
        """ (Roll, pitch, yaw) camera/gimbal angles in degrees. """
        return self._rpy

    @staticmethod
    def _get_exif_float(exif_dict: Dict[str, str], key: str) -> Union[None, float, Tuple[float]]:
        """ Convert numeric EXIF tag to float(s). """
        if not key in exif_dict:
            return None
        val_list = [
            float(val_str.strip(' (')) for val_str in exif_dict[key].split(')')
            if len(val_str) > 0
        ]
        return val_list[0] if len(val_list) == 1 else tuple(val_list)

    @staticmethod
    def _get_sensor_size(
        exif_dict: Dict[str, str], im_size: Union[Tuple[int, int], np.ndarray]
    ) -> Union[None, Tuple[float, float]]:
        """ Return the sensor (width, height) in mm. """

        unit_key = 'EXIF_FocalPlaneResolutionUnit'
        xres_key = 'EXIF_FocalPlaneXResolution'
        yres_key = 'EXIF_FocalPlaneYResolution'

        if unit_key not in exif_dict or xres_key not in exif_dict or yres_key not in exif_dict:
            return None

        # find mm per resolution unit
        unit_code = int(exif_dict["EXIF_FocalPlaneResolutionUnit"])
        mm_per_unit_dict = {
            # https://www.sno.phy.queensu.ca/~phil/exiftool/TagNames/EXIF.html
            2: 25.4,  # inches
            3: 10.,  # cm
            4: 1.,  # mm
            5: 0.001,  # um
        }
        mm_per_unit = mm_per_unit_dict.get(unit_code, None)
        if not mm_per_unit:
            logger.warning(f'Unknown focal plane resolution unit: {unit_code}')
            return None

        # return sensor size in mm
        pixels_per_unit = np.array(
            [Exif._get_exif_float(exif_dict, xres_key), Exif._get_exif_float(exif_dict, yres_key)]
        )
        return tuple(mm_per_unit * np.array(im_size) / pixels_per_unit)

    @staticmethod
    def _get_focal(exif_dict: Dict[str, str]) -> Tuple[float, float]:
        """ Return the actual and 35mm equivalent focal lengths in mm. """
        focal_35 = Exif._get_exif_float(exif_dict, 'EXIF_FocalLengthIn35mmFilm')
        focal = Exif._get_exif_float(exif_dict, 'EXIF_FocalLength')
        return focal, focal_35

    @staticmethod
    def _get_lla(exif_dict: Dict[str, str]) -> Union[None, Tuple[float, float, float]]:
        """
        Return the (latitutde, longitude, altitude) image location with latitude, longitude in decimal degrees, and
        altitude in meters.
        """
        lat_ref_key = 'EXIF_GPSLatitudeRef'
        lon_ref_key = 'EXIF_GPSLongitudeRef'
        lat_key = 'EXIF_GPSLatitude'
        lon_key = 'EXIF_GPSLongitude'
        if any([key not in exif_dict for key in [lat_ref_key, lon_ref_key, lat_key, lon_key]]):
            return None

        # get latitude, longitude
        def dms_to_decimal(dms: Tuple[float, float, float], ref: str):
            """ Convert (degrees, minutes, seconds) tuple to decimal degrees, applying reference sign. """
            sign = 1 if ref in 'NE' else -1
            return ((dms[2] / 60 + dms[1]) / 60 + dms[0]) * sign

        lat = dms_to_decimal(Exif._get_exif_float(exif_dict, lat_key), exif_dict[lat_ref_key])
        lon = dms_to_decimal(Exif._get_exif_float(exif_dict, lon_key), exif_dict[lon_ref_key])

        # get altitude
        alt = Exif._get_exif_float(exif_dict, 'EXIF_GPSAltitude') or 0.
        alt_ref = int(exif_dict.get('EXIF_GPSAltitudeRef', '0x00'), 0)
        if alt_ref == 1:
            alt *= -1

        return lat, lon, alt

    @staticmethod
    def _get_rpy(xmp_dict: Dict[str, str]) -> Union[None, Tuple[float, float, float]]:
        """ Return the camera / gimbal (roll, pitch, yaw) angles in degrees if they exist. """
        # extract RPY from XMP if the values exist
        rpy = None
        for schema_name, rpy_schema in rpy_schemas.items():
            if all([rpy_key in xmp_dict for rpy_key in rpy_schema['rpy_keys']]):
                rpy = np.array([float(xmp_dict[rpy_key]) for rpy_key in rpy_schema['rpy_keys']])
                rpy += np.array(rpy_schema['rpy_offsets'])
                break

        return tuple(rpy)
