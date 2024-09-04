# TODO: change URL to main branch
# code for getting started->api->pan sharpening
# [pan_sharpen]
import orthority as oty

# URLs of required files
url_root = 'https://raw.githubusercontent.com/leftfield-geospatial/orthority/feature_refine_gcp/tests/data/'
pan_file = url_root + 'pan_sharp/pan.tif'  # panchromatic drone image
ms_file = url_root + 'pan_sharp/ms.tif'  # multispectral (RGB) drone image

# create PanSharpen object and pan sharpen
pan_sharp = oty.PanSharpen(pan_file, ms_file)
pan_sharp.process('pan_sharp.tif')
# [end pan_sharpen]
