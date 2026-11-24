"""
crop_goes.py
Author: Rachael Stephan

Purpose: to crop the raw .nc goes files into a .tiff file spanning only the study domain.
"""

# libraries
import os
from osgeo import gdal
import glob
from pathlib import Path

# constants
GOES_DIR_RAW = "/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES/raw"
GOES_DIR_CLIP = "/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES/processed"

DOMAIN_TOP = 35
DOMAIN_BOTTOM = 34
DOMAIN_LEFT = -88
DOMAIN_RIGHT = -87

# clear processed folder
for path in Path(GOES_DIR_CLIP).iterdir():
    if path.is_file():
        path.unlink()


# set gdal warping options
gdal.UseExceptions()

options = gdal.WarpOptions(
    format = 'GTiff',
    outputBounds = (DOMAIN_LEFT,DOMAIN_BOTTOM,DOMAIN_RIGHT,DOMAIN_TOP)
) 

# loop over all files in the GOES directory

for file in os.listdir(GOES_DIR_RAW):
    
    old_raster = os.path.join(GOES_DIR_RAW, file)
    new_raster = os.path.join(GOES_DIR_CLIP, file)

    gdal.Warp(new_raster, old_raster, options=options)

