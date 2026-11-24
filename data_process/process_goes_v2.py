"""
process_goes.py

Purpose: Convert downloaded GOES-East ABI Land Surface Temperature (LST)
NetCDF files into the GeoTIFFs that the lidar processing script reads.

Each output file:
    * has one band of LST in Kelvin (float32),
    * has NaN wherever a pixel is cloudy, low quality or out of range,
    * is on a regular longitude/latitude grid (EPSG:4326) that covers the
      BNF lidar sites, and
    * is named YYYYMMDDHHMM_<satellite>_LST.tif, so the lidar script can
      read the date from characters 0-7 and the UTC hour from 8-9.

Steps for each input file:
    1. Read the start time from the file name (UTC).
    2. Skip the file unless its hour is in PROCESS_HOURS.
    3. Read LST and the quality flag (DQF); set bad pixels to NaN.
    4. Rebuild the geostationary projection from the file's metadata.
    5. Resample onto the output lon/lat grid.
    6. Write the GeoTIFF and record its valid-pixel fraction.

Run again at any time: files that already exist are skipped.
"""
# standard-library imports
import csv
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

# third-party imports
import netCDF4 as nc
import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

# module level set-up
LOGGER = logging.getLogger(__name__)

# constants: folders
GOES_ROOT = Path("/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES")
INPUT_DIR = GOES_ROOT / "raw"             # downloaded ABI .nc files
OUTPUT_DIR = GOES_ROOT / "processed"      # GeoTIFFs for the lidar script
SUMMARY_CSV = OUTPUT_DIR / "goes_processed_summary.csv"

# constants: which files to process
# The lidar script keeps only the 15 and 16 UTC files (its GOES_HOURS).
# Use tuple(range(24)) to convert every hour instead.
PROCESS_HOURS = (15, 16)
OVERWRITE = False          # True: rebuild files that already exist

# constants: quality control
# DQF values treated as good. Confirm against the flag_meanings that the
# script prints for your files before trusting this setting.
KEEP_DQF = (0,)
MIN_VALID_K = 200.0        # LST outside this range is set to NaN
MAX_VALID_K = 350.0

# constants: output grid (longitude/latitude, EPSG:4326)
# The lidar script assumes roughly 2 km pixels and a domain at least
# about 100 km across. A larger domain contains more heterogeneity but
# is more likely to fail the lidar script's clear-sky test.
WEST, EAST = -88.2, -86.5
SOUTH, NORTH = 33.7, 35.1
PIXEL_SIZE_DEG = 0.02      # about 2.2 km north-south, 1.8 km east-west
RESAMPLING = Resampling.nearest   # keeps original values and NaN gaps

# BNF lidar sites, [longitude, latitude]; keep in sync with the lidar
# script. Used only to check that the sites fall inside the grid.
SITE_LONLAT = {"M1": [-87.33786, 34.34234],
               "S20": [-87.29247, 34.65400],
               "S40": [-87.45347, 34.17954]}

# derived values
N_COLS = int(round((EAST - WEST) / PIXEL_SIZE_DEG))
N_ROWS = int(round((NORTH - SOUTH) / PIXEL_SIZE_DEG))
DST_TRANSFORM = from_origin(WEST, NORTH, PIXEL_SIZE_DEG, PIXEL_SIZE_DEG)
DST_CRS = "EPSG:4326"

# ABI start time: ..._G16_s20250910001170_e... = year, day of year, hour,
# minute, second, tenth of a second
START_PATTERN = re.compile(r"_(G\d+)_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d_e")


# functions
def parse_start_time(filename):
    """Return the satellite and start time encoded in an ABI file name.

    Parameters
    ----------
    filename : str
        File name such as ``OR_ABI-L2-LSTC-M6_G16_s20250910001170_e...nc``.

    Returns
    -------
    tuple of (str, datetime.datetime) or None
        Satellite label (for example ``G16``) and the UTC start time, or
        None if the name does not follow the ABI pattern.
    """
    match = START_PATTERN.search(filename)
    if match is None:
        return None
    satellite = match.group(1)
    year, doy, hour, minute, second = (int(g) for g in match.groups()[1:])
    start = datetime(year, 1, 1) + timedelta(
        days=doy - 1, hours=hour, minutes=minute, seconds=second)
    return satellite, start


def read_source_grid(fp):
    """Build the projection and affine transform of the ABI grid.

    The x and y coordinates in ABI files are scan angles in radians.
    Multiplying by the satellite height gives metres in the geostationary
    projection, which is what GDAL expects.

    Parameters
    ----------
    fp : netCDF4.Dataset
        Open ABI LST file.

    Returns
    -------
    tuple of (rasterio.crs.CRS, affine.Affine)
        Source coordinate system and the transform for the full grid.
    """
    name = getattr(fp["LST"], "grid_mapping", "goes_imager_projection")
    proj = fp[name]
    height = float(proj.perspective_point_height)
    crs = CRS.from_string(
        "+proj=geos "
        f"+h={height} "
        f"+lon_0={float(proj.longitude_of_projection_origin)} "
        f"+sweep={proj.sweep_angle_axis} "
        f"+a={float(proj.semi_major_axis)} "
        f"+b={float(proj.semi_minor_axis)} "
        "+units=m +no_defs")

    x = np.asarray(fp["x"][:], dtype="float64") * height
    y = np.asarray(fp["y"][:], dtype="float64") * height
    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)      # negative: y runs north to south
    transform = Affine(dx, 0.0, x[0] - dx / 2, 0.0, dy, y[0] - dy / 2)
    return crs, transform


def read_lst(fp):
    """Read LST in Kelvin, with NaN for every pixel that fails quality control.

    Parameters
    ----------
    fp : netCDF4.Dataset
        Open ABI LST file.

    Returns
    -------
    numpy.ndarray
        float32 array on the ABI grid.
    """
    # netCDF4 applies scale_factor, add_offset and the fill value.
    lst = np.ma.asarray(fp["LST"][:]).astype("float32").filled(np.nan)
    dqf = np.ma.asarray(fp["DQF"][:]).filled(-1)   # masked flag counts as bad
    bad = ~np.isin(dqf, KEEP_DQF)
    bad |= ~np.isfinite(lst) | (lst < MIN_VALID_K) | (lst > MAX_VALID_K)
    lst[bad] = np.nan
    return lst


def log_flag_info(path):
    """Log the units and quality-flag definitions of one input file."""
    with nc.Dataset(path) as fp:
        LOGGER.info("LST units: %s", getattr(fp["LST"], "units", "unknown"))
        dqf = fp["DQF"]
        LOGGER.info("DQF flag_values: %s", getattr(dqf, "flag_values", "n/a"))
        LOGGER.info("DQF flag_meanings: %s",
                    getattr(dqf, "flag_meanings", "n/a"))
        LOGGER.info("Keeping DQF values: %s", KEEP_DQF)


def process_file(path):
    """Convert one ABI LST file to a GeoTIFF on the output grid.

    Parameters
    ----------
    path : pathlib.Path
        Input ABI LST NetCDF file.

    Returns
    -------
    tuple of (str, dict or None)
        Status (``done``, ``skipped_hour``, ``exists`` or ``bad_name``)
        and, for ``done``, a summary row for the CSV.
    """
    parsed = parse_start_time(path.name)
    if parsed is None:
        return "bad_name", None
    satellite, start = parsed
    if start.hour not in PROCESS_HOURS:
        return "skipped_hour", None

    out_path = OUTPUT_DIR / f"{start:%Y%m%d%H%M}_{satellite}_LST.tif"
    if out_path.exists() and not OVERWRITE:
        return "exists", None

    with nc.Dataset(path) as fp:
        src_crs, src_transform = read_source_grid(fp)
        lst = read_lst(fp)

    dst = np.full((N_ROWS, N_COLS), np.nan, dtype="float32")
    reproject(source=lst, destination=dst,
              src_transform=src_transform, src_crs=src_crs,
              dst_transform=DST_TRANSFORM, dst_crs=DST_CRS,
              src_nodata=np.nan, dst_nodata=np.nan,
              resampling=RESAMPLING)

    with rasterio.open(out_path, "w", driver="GTiff",
                       height=N_ROWS, width=N_COLS, count=1,
                       dtype="float32", crs=DST_CRS,
                       transform=DST_TRANSFORM, nodata=np.nan,
                       compress="deflate") as dataset:
        dataset.write(dst, 1)

    row = {"output": out_path.name,
           "source": path.name,
           "valid_fraction": round(float(np.isfinite(dst).mean()), 4)}
    return "done", row


def check_sites_in_grid():
    """Warn if any lidar site lies outside the output grid."""
    for site, (lon, lat) in SITE_LONLAT.items():
        if not (WEST < lon < EAST and SOUTH < lat < NORTH):
            LOGGER.warning("Site %s (%.4f, %.4f) is outside the output grid",
                           site, lon, lat)


def write_summary(rows):
    """Append summary rows to the CSV, writing a header for a new file."""
    if not rows:
        return
    is_new = not SUMMARY_CSV.exists()
    with open(SUMMARY_CSV, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def main():
    """Process every file in INPUT_DIR and report the outcome."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_sites_in_grid()

    files = sorted(INPUT_DIR.glob("*.nc"))
    LOGGER.info("Found %d input files in %s", len(files), INPUT_DIR)

    counts = {"done": 0, "skipped_hour": 0, "exists": 0, "bad_name": 0,
              "failed": 0}
    rows = []
    logged_flags = False
    for number, path in enumerate(files, start=1):
        try:
            if not logged_flags and path.name:
                parsed = parse_start_time(path.name)
                if parsed and parsed[1].hour in PROCESS_HOURS:
                    log_flag_info(path)
                    logged_flags = True
            status, row = process_file(path)
        except (OSError, KeyError, ValueError, AttributeError) as error:
            LOGGER.warning("Failed on %s: %s", path.name, error)
            status, row = "failed", None
        counts[status] += 1
        if row is not None:
            rows.append(row)
        if number % 200 == 0:
            LOGGER.info("Checked %d of %d files", number, len(files))

    write_summary(rows)
    LOGGER.info("Finished: %s", counts)
    if rows:
        fractions = [r["valid_fraction"] for r in rows]
        LOGGER.info("Valid-pixel fraction of new files: min %.2f, mean %.2f",
                    min(fractions), float(np.mean(fractions)))


# run
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    main()
