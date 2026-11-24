"""
download_goes.py
Author: R. Stephan

Purpose: Download GOES-East ABI Land Surface Temperature (LST) files from AWS S3.

GOES-16 is used before the April 2025 operational transition and GOES-19 after it. The transition occured on April 7th, 2025 at ~15:10 UTC.

The list of dates to download is built from the filenames in a folder of ARM files.

No credentials are required.
"""
# standard-library imports
import datetime as dt
import logging              # report progress and warnings with timestamps instead of printing
import os
import sys
import re
from pathlib import Path

# third-party imports
import s3fs                 # third-party library that makes Amazon S3 buckets behave like a file system


# module level set-up
LOGGER = logging.getLogger(__name__)

# constants
SAVE_DIR = "/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES/raw"
TRANSITION_DATE = dt.date(2025, 4, 7)
TRANSITION_HOUR = 15

# dates to download, inclusive (edit these two lines)
START_DATE = dt.date(2024, 9, 6)
END_DATE = dt.date(2026, 8, 29)

# UTC hours to download: all 24
HOURS = list(range(24))

CLEAR_RAW_DIR = False  # True deletes everything in SAVE_DIR before running

def build_date_list(start=START_DATE, end=END_DATE):
    """Return every date from start to end, inclusive."""
    
    if end < start:
        raise ValueError(f"END_DATE {end} is before START_DATE {start}")
    n_days = (end - start).days + 1
    return [start + dt.timedelta(days=n) for n in range(n_days)]

def pick_bucket(date, hour):
    """Return the correct GOES-East bucket for a given UTC date and hour (Make sure correct satellite is used)."""

    if (date, hour) >= (TRANSITION_DATE, TRANSITION_HOUR):
        return "noaa-goes19"
    return "noaa-goes16"

def list_hour_files(fs, bucket, product, date, hour):
    """Return the S3 keys for one product, date, and UTC hour. fs is a filesystem object."""

    # get day of year
    doy = date.timetuple().tm_yday

    # builds S3 folder path based on date
    prefix = f"{bucket}/{product}/{date.year}/{doy:03d}/{hour:02d}/"

    # trys to retrieve the file, otherwise logs a warning
    try:
        return fs.ls(prefix)
    except FileNotFoundError:
        LOGGER.warning("No data found at %s", prefix)
        return []

def download_goes_lst(dates, outdir=SAVE_DIR, product="ABI-L2-LSTC", hours= HOURS):
    """Download GOES-East LST files for each date.

    Parameters
    ----------
    dates : iterable of datetime.date
        Days to download.
    
    outdir : str, optional
        Destination directory (created if needed). Defaults to SAVE_DIR.
    
    product : str, optional
        ABI product name. ABI-L2-LSTC is CONUS and ABI-L2-LSTF is full disk.
    
    hours : iterable of int, optional
        UTC hours to download. Defaults to all 24.

    Returns
    -------
    list of str
        Local paths of all files present after the run.
    """

    # create list of hours to download
    hours = list(range(24)) if hours is None else list(hours)
    
    # check if directory exists
    os.makedirs(outdir, exist_ok=True)

    # opens a connection to S3
    fs = s3fs.S3FileSystem(anon=True)

    local_paths = []
    
    # finds files and downloads them
    for date in dates:
        for hour in hours:
            # pick correct satellite bucket to grab data from
            bucket = pick_bucket(date, hour)

            # gets file paths and adds to download list if they don't already exist locally
            for key in list_hour_files(fs, bucket, product, date, hour):
                local = os.path.join(outdir, os.path.basename(key))
                local_paths.append(local)
                already_done = (os.path.exists(local)
                                and os.path.getsize(local) == fs.size(key))
                if already_done:
                    continue
                LOGGER.info("Downloading %s", key)
                fs.get(key, local)

    return local_paths

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # clear raw directory (only if explicitly requested)
    if CLEAR_RAW_DIR:
        for path in Path(SAVE_DIR).iterdir():
            if path.is_file():
                path.unlink()

    DATES = build_date_list(START_DATE, END_DATE)
    LOGGER.info("Downloading %d days, %s to %s, %d hours per day",
                len(DATES), DATES[0], DATES[-1], len(HOURS))
    download_goes_lst(dates=DATES, product="ABI-L2-LSTC", hours=HOURS)
