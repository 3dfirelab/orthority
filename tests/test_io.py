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
import csv
import json
import logging
from pathlib import Path
from typing import Tuple, Dict, List

import numpy as np
import pytest
import rasterio as rio
from simple_ortho import io
from simple_ortho.utils import validate_collection
from simple_ortho.enums import CameraType, Interp, CsvFormat
from simple_ortho.errors import ParamFileError, CrsMissingError
from tests.conftest import oty_to_osfm_int_param


def _validate_ext_param_dict(ext_param_dict: Dict, camera: str=None):
    """ Basic validation of an external parameter dictionary. """
    for filename, ext_params in ext_param_dict.items():
        assert set(ext_params.keys()) == {'opk', 'xyz', 'camera'}
        opk, xyz = np.array(ext_params['opk']), np.array(ext_params['xyz'])
        assert len(opk) == 3 and len(xyz) == 3
        # rough check for radians
        assert all(np.abs(opk) <= 2 * np.pi) and any(opk != 0.)
        # rough check for not latitude, longitude, & altitude > 0
        assert all(xyz != 0) and np.abs(xyz[0]) > 180. and np.abs(xyz[1]) > 90. and xyz[2] > 0
        assert ext_params['camera'] == camera


def test_rw_oty_int_param(mult_int_param_dict: Dict, tmp_path: Path):
    """ Test interior parameter read / write from / to orthority yaml format. """
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, mult_int_param_dict)
    test_dict = io.read_oty_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['focal_len', 'sensor_size', 'im_size'])
def test_read_oty_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params.pop(missing_key)
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert missing_key in str(ex)


def test_read_oty_int_param_unknown_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when unknown keys are present. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['other'] = 0.
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert 'other' in str(ex)


def test_read_oty_int_param_cam_type_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error the camera type is unknown. """
    int_params = next(iter(pinhole_int_param_dict.values())).copy()
    int_params['cam_type'] = Interp.cubic
    filename = tmp_path.joinpath('int_param.yaml')
    io.write_int_param(filename, dict(default=int_params))
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_oty_int_param(filename)
    assert 'camera type' in str(ex)


@pytest.mark.parametrize('filename', ['osfm_int_param_file', 'odm_int_param_file'])
def test_read_osfm_int_param(filename: str, mult_int_param_dict: Dict, request: pytest.FixtureRequest):
    """ Test reading interior parameters from ODM / OpenSfM format files. """
    filename: Path = request.getfixturevalue(filename)
    test_dict = io.read_osfm_int_param(filename)
    assert test_dict == mult_int_param_dict


@pytest.mark.parametrize('missing_key', ['projection_type', 'width', 'height', 'focal_x', 'focal_y'])
def test_read_osfm_int_param_missing_error(pinhole_int_param_dict: Dict, missing_key: str, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when required keys are missing. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params.pop(missing_key)
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert missing_key in str(ex)


def test_read_osfm_int_param_unknown_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error when unknown keys are present. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['other'] = 0.
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'other' in str(ex)


def test_read_osfm_int_param_proj_type_error(pinhole_int_param_dict: Dict, tmp_path: Path):
    """ Test reading orthority format interior parameters raises an error the projection type is unsupported. """
    osfm_dict = oty_to_osfm_int_param(pinhole_int_param_dict)
    int_params = next(iter(osfm_dict.values()))
    int_params['projection_type'] = 'other'
    with pytest.raises(ParamFileError) as ex:
        _ = io._read_osfm_int_param(osfm_dict)
    assert 'projection type' in str(ex)


def test_read_exif_int_param_dewarp(odm_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image with the `DewarpData` XMP tag. """
    int_param_dict = io.read_exif_int_param(odm_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'brown'
    assert {'k1', 'k2', 'p1', 'p2', 'k3'}.issubset(int_params.keys())


def test_read_exif_int_param_no_dewarp(exif_image_file: Path):
    """ Testing reading EXIF / XMP tag interior parameters from an image without the `DewarpData` XMP tag. """
    int_param_dict = io.read_exif_int_param(exif_image_file)
    int_params = next(iter(int_param_dict.values()))
    assert int_params.get('cam_type', None) == 'pinhole'


def test_read_exif_int_param_error(ngi_image_file: Path):
    """ Test reading EXIF tag interior parameters from a non EXIF image raises an error. """
    with pytest.raises(ParamFileError) as ex:
        _ = io.read_exif_int_param(ngi_image_file)
    assert 'focal length' in str(ex)


def test_csv_reader_legacy(ngi_legacy_csv_file: Path, ngi_crs: str, ngi_image_files: Tuple[Path, ...]):
    """ Test reading exterior parameters from a legacy format CSV file. """
    reader = io.CsvReader(ngi_legacy_csv_file, crs=ngi_crs)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    assert reader.crs == rio.CRS.from_string(ngi_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict)


def test_csv_reader_xyz_opk(ngi_xyz_opk_csv_file: Path, ngi_crs: str, ngi_image_files: Tuple[Path, ...]):
    """ Test reading exterior parameters from an xyz_opk format CSV file with a header. """
    reader = io.CsvReader(ngi_xyz_opk_csv_file, crs=ngi_crs)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    assert reader._format is CsvFormat.xyz_opk
    with open(ngi_xyz_opk_csv_file.with_suffix('.prj')) as f:
        assert reader.crs == rio.CRS.from_string(f.read())

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.stem for filename in ngi_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    _validate_ext_param_dict(ext_param_dict)


def test_csv_reader_lla_rpy(
    odm_lla_rpy_csv_file: Path, odm_crs: str, odm_image_files: Tuple[Path, ...], osfm_reconstruction_file: Path
):
    """ Test reading exterior parameters from an lla_rpy format CSV file with a header. """
    reader = io.CsvReader(odm_lla_rpy_csv_file, crs=odm_crs)
    assert set(reader._fieldnames) == {
        'filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'camera', 'other'
    }
    assert reader._format is CsvFormat.lla_rpy
    assert reader.crs == rio.CRS.from_string(odm_crs)

    ext_param_dict = reader.read_ext_param()
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)

    with open(osfm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, camera=cam_id)


def test_csv_reader_xyz_opk_prj_crs(ngi_xyz_opk_csv_file: Path):
    """ Test CsvReader initialised with a xyz_* format CSV file and no CRS, reads the CRS from a .prj file. """
    reader = io.CsvReader(ngi_xyz_opk_csv_file, crs=None)
    assert reader._fieldnames == io.CsvReader._legacy_fieldnames
    with open(ngi_xyz_opk_csv_file.with_suffix('.prj')) as f:
        assert reader.crs == rio.CRS.from_string(f.read())


def test_csv_reader_lla_rpy_auto_crs(odm_lla_rpy_csv_file: Path, odm_crs: str):
    """ Test CsvReader initialised with a lla_rpy format CSV file and no CRS generates an auto UTM CRS. """
    reader = io.CsvReader(odm_lla_rpy_csv_file, crs=None)
    assert reader.crs == rio.CRS.from_string(odm_crs)


def test_csv_reader_fieldnames(odm_lla_rpy_csv_file: Path):
    """ Test reading exterior parameters from a CSV file with `fieldnames` argument. """
    fieldnames = ['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw', 'camera', 'custom']
    reader = io.CsvReader(odm_lla_rpy_csv_file, fieldnames=fieldnames)
    assert set(reader._fieldnames) == set(fieldnames)
    _ = reader.read_ext_param()


@pytest.mark.parametrize('filename, crs, fieldnames, exp_format', [
    (
        'ngi_xyz_opk_csv_file', 'ngi_crs', ['filename', 'easting', 'northing', 'altitude', 'omega', 'phi', 'kappa'],
        CsvFormat.xyz_opk
    ),
    (
        'ngi_xyz_opk_csv_file', 'ngi_crs', ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw'],
        CsvFormat.xyz_rpy
    ),
    (
        'odm_lla_rpy_csv_file', 'odm_crs', ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
        CsvFormat.lla_opk
    ),
    (
        'odm_lla_rpy_csv_file', 'odm_crs', ['filename', 'latitude', 'longitude', 'altitude', 'roll', 'pitch', 'yaw'],
        CsvFormat.lla_rpy
    ),
])  # yapf: disable
def test_csv_reader_format(
    filename: str, crs: str, fieldnames: List, exp_format:CsvFormat, request: pytest.FixtureRequest
):
    """ Test reading exterior parameters from a CSV file in different (simlated) position / orientation formats. """
    filename: Path = request.getfixturevalue(filename)
    crs: str = request.getfixturevalue(crs)

    reader = io.CsvReader(filename, crs=crs, fieldnames=fieldnames)
    assert reader._format == exp_format
    assert reader.crs == rio.CRS.from_string(crs)

    ext_param_dict = reader.read_ext_param()
    assert len(ext_param_dict) > 0
    _validate_ext_param_dict(ext_param_dict)


@pytest.mark.parametrize('dialect', [
    dict(delimiter=' ', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\n', quotechar="'", quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_ALL),
    dict(delimiter=' ', lineterminator='\n', quotechar="'", quoting=csv.QUOTE_ALL),
    dict(delimiter=' ', lineterminator='\r', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=' ', lineterminator='\r\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=',', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=';', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter=':', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
    dict(delimiter='\t', lineterminator='\n', quotechar='"', quoting=csv.QUOTE_MINIMAL),
])  # yapf: disable
def test_csv_reader_dialect(
    odm_lla_rpy_csv_file: Path, odm_crs: str, odm_image_files: Tuple[Path, ...], osfm_reconstruction_file: Path,
    dialect: Dict, tmp_path: Path
):
    """ Test reading exterior parameters from a CSV files in different dialects. """
    # create test CSV file
    test_filename = tmp_path.joinpath('ext-param-test.csv')
    with open(odm_lla_rpy_csv_file, 'r') as fin:
        with open(test_filename, 'w', newline='') as fout:
            reader = csv.reader(fin, delimiter=' ', quotechar='"')
            writer = csv.writer(fout, **dialect)
            for row in reader:
                writer.writerow(row)

    # read test file
    reader = io.CsvReader(test_filename, crs=odm_crs)
    for attr in ['delimiter', 'quotechar']:
        assert getattr(reader._dialect, attr) == dialect[attr]
    ext_param_dict = reader.read_ext_param()

    # validate dict
    file_keys = [filename.name for filename in odm_image_files]
    assert set(ext_param_dict.keys()).issubset(file_keys)
    with open(osfm_reconstruction_file, 'r') as f:
        json_obj = json.load(f)
    cam_id = next(iter(json_obj[0]['cameras'].keys())).strip('v2 ')
    _validate_ext_param_dict(ext_param_dict, camera=cam_id)


@pytest.mark.parametrize('fieldnames', [
    ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw'],
    ['filename', 'latitude', 'longitude', 'altitude', 'omega', 'phi', 'kappa'],
])  # yapf: disable
def test_csv_reader_crs_error(ngi_legacy_csv_file: Path, fieldnames: List):
    """ Test that CsvReader initialised with a xyz_rpy or lla_opk format file and no CRS raises an error. """
    fieldnames = ['filename', 'easting', 'northing', 'altitude', 'roll', 'pitch', 'yaw']
    with pytest.raises(CrsMissingError) as ex:
        reader = io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert 'crs' in str(ex).lower()


@pytest.mark.parametrize('missing_field', io.CsvReader._legacy_fieldnames)
def test_csv_reader_missing_fieldname_error(ngi_legacy_csv_file: Path, missing_field):
    """ Test that CsvReader intialised with a missing fieldname raises an error. """
    fieldnames = io.CsvReader._legacy_fieldnames.copy()
    fieldnames.remove(missing_field)
    with pytest.raises(ParamFileError) as ex:
        reader = io.CsvReader(ngi_legacy_csv_file, fieldnames=fieldnames)
    assert missing_field in str(ex)



# TODO:
# - Interior:
#   - Test reading different formats is done correctly
#   - Test error conditions reported meaningfully (missing keys?)
#   - Have a test int-param dict / fixture
#   - Fixtures that are the above written to oty, cameras.json & reconstruction.json files
#   - For now, I think just use existing test ngi/odm images for testing exif.  It is not complete, but seems overkill
#     to make code for generating exif test data.
#   - Can there be a single set of interior params that is used for testing io and testing camera & ortho?  We would
#   probably need to change how camera is initialised.
#   - Writing oty & then reading oty params.
#   - We still haven't done checking of distortion params for different cam types...  Use a schema with validate_collection?
#   - Test multiple camera config & legacy format(s)
# Exterior & readers:
#   - Maybe have an exterior param dict fixture, that is used to create different formats, and test reading against.
#   - The "create different formats" is not that trivial though...
#   CSV
#   - Different angle / position formats
#   - With / without header, different delimiters, additional columns, with / without camera id
#   - With / without proj CRS file
#   - Error conditions
