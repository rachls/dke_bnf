"""
process_goes.py

Purpose: Convert downloaded GOES-East ABI Land Surface Temperature (LST)
NetCDF files into GeoTIFFs that the lidar processing script reads.

Each output file:
    1. has one band of LST in Kelvin (float32),
    2. has NaN wherever a pixel is cloudy, low quality or out of range,
    3. is on a regular longitude/latitude grid (EPSG:4326) covering the
      BNF lidar sites, and
    4. is named YYYYMMDDHHMM_<satellite>_LST.tif.

GDAL Warp performs the reprojection, resampling, cropping and GeoTIFF
creation. Existing outputs are skipped when OVERWRITE is False.
"""

# standard-library imports
import csv
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path

# third-party imports
import netCDF4 as nc
import numpy as np
from osgeo import gdal, osr

# module-level set-up
LOGGER = logging.getLogger(__name__)
gdal.UseExceptions()

# constants: folders
GOES_ROOT = Path(
    "/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES"
)
INPUT_DIR = GOES_ROOT / "raw"
OUTPUT_DIR = GOES_ROOT / "processed"
SUMMARY_CSV = OUTPUT_DIR / "goes_processed_summary.csv"

# constants: which files to process
PROCESS_HOURS = (15, 16)
OVERWRITE = False

# constants: quality control
# Confirm flag_meanings in the log before treating DQF == 0 as sufficient.
KEEP_DQF = (0,)
MIN_VALID_K = 200.0
MAX_VALID_K = 350.0

# constants: output grid (longitude/latitude, EPSG:4326)
WEST, EAST = -88.2, -86.5
SOUTH, NORTH = 33.7, 35.1
PIXEL_SIZE_DEG = 0.02
RESAMPLING = "near"

# BNF lidar sites, [longitude, latitude]
SITE_LONLAT = {
    "M1": [-87.33786, 34.34234],
    "S20": [-87.29247, 34.65400],
    "S40": [-87.45347, 34.17954],
}

# derived values
N_COLS = int(round((EAST - WEST) / PIXEL_SIZE_DEG))
N_ROWS = int(round((NORTH - SOUTH) / PIXEL_SIZE_DEG))
DST_CRS = "EPSG:4326"

# ABI start time:
# ..._G16_s20250910001170_e... = year, day of year, hour, minute,
# second and tenths of a second
START_PATTERN = re.compile(
    r"_(G\d+)_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})\d_e"
)


def parse_start_time(filename):
    """Return satellite and UTC start time encoded in an ABI filename."""
    match = START_PATTERN.search(filename)
    if match is None:
        return None

    satellite = match.group(1)
    year, doy, hour, minute, second = (
        int(value) for value in match.groups()[1:]
    )
    start = datetime(year, 1, 1) + timedelta(
        days=doy - 1,
        hours=hour,
        minutes=minute,
        seconds=second,
    )
    return satellite, start


def read_source_grid(fp):
    """Return the source geostationary projection and GDAL geotransform."""
    mapping_name = getattr(
        fp["LST"], "grid_mapping", "goes_imager_projection"
    )
    projection = fp[mapping_name]

    height = float(projection.perspective_point_height)
    longitude = float(projection.longitude_of_projection_origin)
    sweep = str(projection.sweep_angle_axis)
    semi_major = float(projection.semi_major_axis)
    semi_minor = float(projection.semi_minor_axis)

    proj4 = (
        "+proj=geos "
        f"+h={height} "
        f"+lon_0={longitude} "
        f"+sweep={sweep} "
        f"+a={semi_major} "
        f"+b={semi_minor} "
        "+units=m +no_defs"
    )

    source_srs = osr.SpatialReference()
    source_srs.ImportFromProj4(proj4)

    # ABI x/y coordinates are scan angles in radians. Convert to metres.
    x = np.asarray(fp["x"][:], dtype="float64") * height
    y = np.asarray(fp["y"][:], dtype="float64") * height

    if len(x) < 2 or len(y) < 2:
        raise ValueError("GOES x/y coordinate arrays are too short")

    dx = (x[-1] - x[0]) / (len(x) - 1)
    dy = (y[-1] - y[0]) / (len(y) - 1)

    # GDAL geotransform: upper-left pixel corner, not pixel centre.
    geotransform = (
        x[0] - dx / 2.0,
        dx,
        0.0,
        y[0] - dy / 2.0,
        0.0,
        dy,
    )

    return source_srs, geotransform


def read_lst(fp):
    """Read scaled LST in Kelvin and mask pixels failing quality control."""
    # netCDF4 normally applies scale_factor/add_offset and masks fill values.
    lst = np.ma.asarray(fp["LST"][:]).astype("float32").filled(np.nan)
    dqf = np.ma.asarray(fp["DQF"][:]).astype(np.int32).filled(-1)

    bad = ~np.isin(dqf, KEEP_DQF)
    bad |= ~np.isfinite(lst)
    bad |= (lst < MIN_VALID_K) | (lst > MAX_VALID_K)
    lst[bad] = np.nan

    return lst


def log_flag_info(path):
    """Log LST units and DQF definitions from the first eligible file."""
    with nc.Dataset(path) as fp:
        LOGGER.info(
            "LST units: %s",
            getattr(fp["LST"], "units", "unknown"),
        )
        dqf = fp["DQF"]
        LOGGER.info(
            "DQF flag_values: %s",
            getattr(dqf, "flag_values", "n/a"),
        )
        LOGGER.info(
            "DQF flag_meanings: %s",
            getattr(dqf, "flag_meanings", "n/a"),
        )
        LOGGER.info("Keeping DQF values: %s", KEEP_DQF)


def process_file(path):
    """Convert one ABI LST NetCDF file to a cropped EPSG:4326 GeoTIFF."""
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
        source_srs, source_geotransform = read_source_grid(fp)
        lst = read_lst(fp)

    # Create an in-memory GDAL source raster containing the QC-screened LST.
    # GDAL Warp will reproject/crop this source to the requested output grid.
    source_driver = gdal.GetDriverByName("MEM")
    source_ds = source_driver.Create(
        "",
        int(lst.shape[1]),
        int(lst.shape[0]),
        1,
        gdal.GDT_Float32,
    )
    if source_ds is None:
        raise RuntimeError("Could not create in-memory GDAL source dataset")

    source_ds.SetGeoTransform(source_geotransform)
    source_ds.SetProjection(source_srs.ExportToWkt())

    source_band = source_ds.GetRasterBand(1)
    source_band.SetNoDataValue(float("nan"))
    source_band.WriteArray(lst)
    source_band.FlushCache()

    # GDAL Warp performs reprojection, resampling, cropping and output writing.
    warp_options = gdal.WarpOptions(
        format="GTiff",
        dstSRS=DST_CRS,
        outputBounds=(WEST, SOUTH, EAST, NORTH),
        outputBoundsSRS=DST_CRS,
        width=N_COLS,
        height=N_ROWS,
        resampleAlg=RESAMPLING,
        srcNodata=float("nan"),
        dstNodata=float("nan"),
        outputType=gdal.GDT_Float32,
        multithread=True,
        creationOptions=[
            "COMPRESS=DEFLATE",
            "PREDICTOR=3",
            "TILED=YES",
        ],
    )

    warped_ds = gdal.Warp(
        str(out_path),
        source_ds,
        options=warp_options,
    )

    # Release the in-memory source after Warp has completed.
    source_ds = None

    if warped_ds is None:
        raise RuntimeError(f"GDAL Warp returned no dataset for {path.name}")

    result = warped_ds.GetRasterBand(1).ReadAsArray()
    valid_fraction = float(np.isfinite(result).mean())
    warped_ds = None

    row = {
        "output": out_path.name,
        "source": path.name,
        "valid_fraction": round(valid_fraction, 4),
    }
    return "done", row


def check_sites_in_grid():
    """Warn if any lidar site lies outside the output grid."""
    for site, (longitude, latitude) in SITE_LONLAT.items():
        if not (
            WEST < longitude < EAST
            and SOUTH < latitude < NORTH
        ):
            LOGGER.warning(
                "Site %s (%.4f, %.4f) is outside the output grid",
                site,
                longitude,
                latitude,
            )


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
    """Process all input files and report progress and elapsed time."""
    start_time = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    check_sites_in_grid()

    files = sorted(INPUT_DIR.glob("*.nc"))
    total_files = len(files)
    LOGGER.info("Found %d input files in %s", total_files, INPUT_DIR)

    counts = {
        "done": 0,
        "skipped_hour": 0,
        "exists": 0,
        "bad_name": 0,
        "failed": 0,
    }
    rows = []
    logged_flags = False

    for number, path in enumerate(files, start=1):
        try:
            if not logged_flags:
                parsed = parse_start_time(path.name)
                if parsed and parsed[1].hour in PROCESS_HOURS:
                    log_flag_info(path)
                    logged_flags = True

            status, row = process_file(path)

        except Exception:
            LOGGER.exception("Failed on %s", path.name)
            status, row = "failed", None

        counts[status] += 1
        if row is not None:
            rows.append(row)

        if number % 200 == 0 or number == total_files:
            elapsed = time.perf_counter() - start_time
            rate = number / elapsed if elapsed > 0 else float("nan")
            LOGGER.info(
                "Checked %d/%d | done=%d skipped_hour=%d exists=%d "
                "failed=%d | elapsed=%.1f min | check_rate=%.2f files/s",
                number,
                total_files,
                counts["done"],
                counts["skipped_hour"],
                counts["exists"],
                counts["failed"],
                elapsed / 60.0,
                rate,
            )

    write_summary(rows)

    elapsed = time.perf_counter() - start_time
    LOGGER.info("Finished: %s", counts)
    LOGGER.info("Total elapsed time: %.1f minutes", elapsed / 60.0)

    if rows:
        fractions = [row["valid_fraction"] for row in rows]
        LOGGER.info(
            "Valid-pixel fraction of new files: min %.2f, mean %.2f",
            min(fractions),
            float(np.mean(fractions)),
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    main()

