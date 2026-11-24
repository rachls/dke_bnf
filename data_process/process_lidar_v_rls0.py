"""
process_lidar_v_rls.py
Code adapted from process_lidar_v2.py (located in FILL IN OLD PATH) 

Relate GOES land-surface-temperature (LST) heterogeneity to lidar winds from ARM data.

For each usable day over the user designated months this script:

1. Finds days that have both GOES LST files and Doppler-lidar (ARM) files.
2. Keeps only clear-sky days (few missing LST pixels).
3. Converts each site's lidar winds to hourly means (13-23 UTC) and computes, per hour and height, the DKE and MKE.
4. Characterises the surface temperature field (spread, length scale, gradient direction, variability along and across the wind).
5. Optionally adds large-scale controls from HRRR (vorticity and pressure gradient).
6. Writes everything to one NetCDF file indexed by date.

Comment tags used below:
    BNF CHANGE: changes from the original (Oklahoma) version of the script.
    FIX: edits from the original.
    NOTE: things to verify or change from the original.
"""

import glob
import os
import warnings
from datetime import datetime, timedelta

import gstools as gs
import netCDF4 as nc
import numpy as np
import rasterio
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.stats import circmean

warnings.simplefilter('ignore', category=FutureWarning)
# Statistics over empty slices (for example a day with a single site)
# legitimately give NaN; silence the accompanying RuntimeWarnings.
warnings.simplefilter('ignore', category=RuntimeWarning)

# CONFIGURATION 

ARM_DIR = '/stor/soteria/hydro/private/rls134/data/dke_bnf/ARM/raw/'
GOES_DIR = '/stor/soteria/hydro/private/rls134/data/dke_bnf/GOES/processed/'
OUT_DIR = '/stor/soteria/hydro/private/rls134/data/dke_bnf/'
OUT_FILE = OUT_DIR + 'lidar_lst_bnf.nc'

DEBUG = False               # print extra diagnostics
NEW_FILE = True             # True: create a new output file; False: open 'r+'
SKIP_HERBIE = True          # True: skip the HRRR vorticity / pressure step
MAX_DAYS = None             # for example 3 for a quick test; None = all days
ALLOW_PARTIAL_DAYS = False  # only keeps days with at least one sample per hour in the window  

# Bankhead sites
SITES = ['M1', 'S20', 'S40']
LONLAT = {'M1': [-87.33786, 34.34234],    # [longitude, latitude]
          'S20': [-87.29247, 34.65400],
          'S40': [-87.45347, 34.17954]}

# Day selection
GOES_HOURS = ('15', '16')      # keep GOES files from these UTC hours
MIN_SITES = 2                  # a day needs at least this many ARM files
FIRST_MONTH, LAST_MONTH = 5, 9
# Dates flagged as bad by the original author
EXCLUDED_DAYS = {'20210730'}
CLEAR_SKY_MAX_BAD = 0.1        # reject a GOES file if >10% of pixels bad

# Lidar window and geometry
FIRST_HOUR = 13      # first UTC hour kept (13 UTC = 8 am CDT)
N_HOURS = 11         # 13-23 UTC -> 11 hourly bins
N_LEVELS = 164       # number of heights in an ARM lidar file
N_HEIGHT = 36        # lowest levels kept (up to about 1000 m)
MISSING = -9999      # ARM missing-value flag

# HRRR sub-grid indices. NOTE: these are for the Oklahoma domain - must be recomputed for Alabama before SKIP_HERBIE is set to False.
HRRR_Y = slice(591, 630)
HRRR_X = slice(887, 916)



# HELPER FUNCTIONS



def exponential(x, a, b):
    """Exponential decay a * exp(-x / b), used to fit the covariance."""
    return a * np.exp(-x / b)


def nans(*shape):
    """Return an array of the given shape filled with NaN."""
    return np.full(shape, np.nan)


def dims_for(name):
    """Return the NetCDF dimension names for an output variable.

    FIX: the original inferred dimension names from array sizes, this avoids any confusion.
    """
    profile_vars = ('DKE_xy', 'DKE_z', 'MKE_xy', 'MKE_z', 'wind_speed',
                    'wind_a', 'weighting')
    hourly_vars = ('wind_a_diff', 'wind_speed_diff', 'precip')
    if name in ('date', 'hour', 'height'):
        return (name,)
    if name in ('lon', 'lat'):
        return ('sites',)
    if name == 'sites_repo':
        return ('date', 'hour', 'height', 'sites')
    if name in profile_vars:
        return ('date', 'hour', 'height')
    if name in hourly_vars:
        return ('date', 'hour')
    return ('date',)  # everything else is one value per day


def site_of(path):
    """Return the site code (M1, S20 or S40) for an ARM file path."""
    base = os.path.basename(path)
    for site in SITES:
        if f'news{site}.' in base:
            return site
    return None


def calculate_length_scale(z, undef=-9999):
    """Estimate the heterogeneity length scale (m) of a 2-D field.

    Samples 1000 random pixels, builds an empirical variogram out to 40 km, converts it to a covariance and fits an exponential decay.
    The e-folding distance of that fit is the length scale.
    """
    np.random.seed(1)  # fixed seed: repeated runs give identical results
    # FIX: columns are drawn from shape[1] and rows from shape[0]. This was swapped in the original.
    ix = np.random.randint(0, z.shape[1], 1000)
    iy = np.random.randint(0, z.shape[0], 1000)
    x = 2000 * ix  # assumes 2 km pixels
    y = 2000 * iy
    field = z[iy, ix]
    # FIX: mask on the sampled values
    valid = field != undef
    x, y, field = x[valid], y[valid], field[valid]
    bins = np.linspace(0, 40000, 10)  # lag-distance bins, 0-40 km
    bin_center, gamma = gs.vario_estimate((x, y), field, bins)
    cov = np.max(gamma) - gamma  # covariance = sill - semivariance
    # Fit a * exp(-h / b): amplitude pinned to the sill, b <= 40 km.
    opt, _ = curve_fit(
        exponential, bin_center, cov,
        bounds=([0.9999 * np.max(gamma), 0], [np.max(gamma), 40000]))
    return opt[1]


def calculate_air_density(z, min_z, max_z):
    """Approximate air density (kg m-3) from a standard atmosphere.

    Density is set to zero outside [min_z, max_z] so the layer weights used for DKE and MKE ignore those heights.
    """
    temp = 20 - 0.00649 * z                                # deg C
    pres = 101.29 * ((temp + 273.15) / 288.08) ** 5.256    # kPa
    rho = pres / (0.2869 * (temp + 273.15))                # ideal gas law
    rho[z < min_z] = 0.0
    rho[z > max_z] = 0.0
    return rho


def hourly_mean(values, hours, first_hour=FIRST_HOUR, n_hours=N_HOURS,
                is_angle=False):
    """Average samples into clock-hour bins; empty bins are NaN.

    BNF CHANGE: replaces the original 'upscale', which assumed exactly 144 or 96 samples per file. Binning by timestamp works for any
    sampling interval (the BNF files mix 5- and 15-minute data) and for partial days, if allowed.
    
    Partial days are controlled by ALLOW_PARTIAL_DAYS. A day counts as partial when at least one of the n_hours clock hours has no sample
    at all. A file with a few missing records inside otherwise covered hours is not partial. If partial days are not allowed, every bin
    is returned as NaN for such a file.

    values: array (n_samples, n_columns); MISSING or NaN are ignored.
    hours:  integer array (n_samples,) with the UTC hour of each sample.
    """
    out = np.full((n_hours, values.shape[1]), np.nan)
    if not ALLOW_PARTIAL_DAYS:
        expected_hours = np.arange(first_hour, first_hour + n_hours)
        if not np.isin(expected_hours, hours).all():
            return out  # incomplete day: every bin stays NaN
    for i in range(n_hours):
        rows = values[hours == first_hour + i]
        for j in range(values.shape[1]):
            col = rows[:, j]
            col = col[(col != MISSING) & ~np.isnan(col)]
            if col.size == 0:
                continue
            if is_angle:
                # Circular mean so that 359 and 1 degrees average to 0.
                out[i, j] = np.rad2deg(circmean(np.deg2rad(col)))
            else:
                out[i, j] = np.mean(col)
    return out

def read_hourly(fp, name, in_window, hours, is_angle=False):
    """Read one variable, keep the time window, average to hourly values.

    Returns an (N_HOURS, n_columns) array. One-dimensional variables (surface met data) come back with a single column.
    """
    # Masked (flagged) values become NaN, so any declared fill value is handled, not only -9999.
    data = np.ma.asarray(fp[name][in_window]).astype(float)
    values = data.filled(np.nan)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return hourly_mean(values, hours, is_angle=is_angle)


def unit_vector(vector):
    """Return the vector scaled to unit length."""
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """Angle in degrees between two 2-D vectors (NaN if either is NaN)."""
    v1 = np.asarray(v1, dtype=float)
    v2 = np.asarray(v2, dtype=float)
    if np.isnan(v1).any() or np.isnan(v2).any():
        return float('nan')
    cosine = np.dot(unit_vector(v1), unit_vector(v2))
    return np.arccos(np.clip(cosine, -1.0, 1.0)) * 180 / np.pi



# STEP 1: Index the input files by date



# flist maps 'YYYYMMDD' to the files available that day:
#   {'20250715': {'GOES': [tif paths], 'ARM': [nc paths]}}
flist = {}


def add_file(day, kind, path):
    """Register one file under its date, creating the entry if needed."""
    if day not in flist:
        flist[day] = {'GOES': [], 'ARM': []}
    flist[day][kind].append(path)

# GOES files: all in GOES_DIR, with names that start with YYYYMMDDHH (date followed by the UTC hour). Only the hours in GOES_HOURS are kept.
for name in sorted(os.listdir(GOES_DIR)):
    if name[8:10] not in GOES_HOURS:
        continue
    add_file(name[0:8], 'GOES', os.path.join(GOES_DIR, name))

# BNF CHANGE: ARM files sit in one sub-folder per datastream
arm_pattern = os.path.join(ARM_DIR, 'bnfdlprofwind4news*.c1', '*.nc')
for path in sorted(glob.glob(arm_pattern)):
    add_file(os.path.basename(path).split('.')[2], 'ARM', path)



# STEP 2: Drop days that cannot be analysed



for day in list(flist):
    month = int(day[4:6])
    drop = (len(flist[day]['ARM']) < MIN_SITES       # too few sites
            or len(flist[day]['GOES']) < 1           # no satellite data
            or month < FIRST_MONTH or month > LAST_MONTH
            or day in EXCLUDED_DAYS)
    if drop:
        del flist[day]



# STEP 3: Clear-sky screen



# A day is kept if at least one of its GOES files has few bad pixels. Cloudy pixels must already be NaN (or below 1) in the GeoTIFFs for this
# test to work; the script does not read the GOES quality flags itself.
klist = []
for day in sorted(flist):
    is_bad = True
    for path in flist[day]['GOES']:
        with rasterio.open(path) as src:
            data = src.read(1)
        n_bad = np.sum((data < 1) | np.isnan(data))
        if n_bad <= CLEAR_SKY_MAX_BAD * data.size:
            is_bad = False
    if not is_bad:
        klist.append(day)

if MAX_DAYS:
    klist = klist[:MAX_DAYS]
n_days = len(klist)
n_sites = len(SITES)
print(n_days)
if DEBUG:
    print(klist)



# STEP 4: Allocate the output arrays (everything starts as NaN)



out = {
    'sites': SITES,
    'height': np.zeros(N_HEIGHT),          # filled from the first file
    'hour': list(range(8, 19)),            # local (CDT) hour labels
    'date': [float(day) for day in klist],  # YYYYMMDD as a number
    'lon': np.array([LONLAT[s][0] for s in SITES]),
    'lat': np.array([LONLAT[s][1] for s in SITES]),
}

# One value per day: surface temperature (LST) descriptors
out['lst_std'] = nans(n_days)           # std of LST over the domain
out['lst_std_parallel'] = nans(n_days)  # mean std along the wind
out['lst_std_perpendicular'] = nans(n_days)  # mean std across the wind
out['lst_mean'] = nans(n_days)          # domain-mean LST
out['lst_std_site'] = nans(n_days)      # std of LST between the sites
out['lst_lhet'] = nans(n_days)          # heterogeneity length scale
out['lst_a'] = nans(n_days)             # direction of the LST gradient
out['lst_a2'] = nans(n_days)            # direction of max variability

# Per day, hour and height: lidar wind energy and winds
out['DKE_xy'] = nans(n_days, N_HOURS, N_HEIGHT)  # between-site variance
out['DKE_z'] = nans(n_days, N_HOURS, N_HEIGHT)
out['MKE_xy'] = nans(n_days, N_HOURS, N_HEIGHT)  # mean-flow energy
out['MKE_z'] = nans(n_days, N_HOURS, N_HEIGHT)
out['wind_speed'] = nans(n_days, N_HOURS, N_HEIGHT)
out['wind_a'] = nans(n_days, N_HOURS, N_HEIGHT)
out['weighting'] = nans(n_days, N_HOURS, N_HEIGHT)  # density * dz

# Per day and hour: quality-control and met values
out['wind_a_diff'] = nans(n_days, N_HOURS)      # met vs lidar direction
out['wind_speed_diff'] = nans(n_days, N_HOURS)  # met vs lidar speed
out['precip'] = nans(n_days, N_HOURS)           # daily mean, repeated

# 1 where a site reported a valid wind at (day, hour, height, site)
out['sites_repo'] = np.zeros((n_days, N_HOURS, N_HEIGHT, n_sites))

if not SKIP_HERBIE:
    # Imported here so the packages are only needed when HRRR is used.
    import metpy.calc as mpcalc
    from herbie import Herbie
    from metpy.units import units
    out['vort'] = nans(n_days)    # maximum vorticity magnitude (HRRR)
    out['pgrad'] = nans(n_days)   # mean pressure-gradient magnitude

print('SETUP and PREPROCESSING COMPLETE')
print('TOTAL days: ' + str(n_days))
print('Beginning processing...', flush=True)




# STEP 5: Process each day



height_set = False
for di, day in enumerate(klist):
    print('::::' + day + ':::::')
    date = datetime(int(day[0:4]), int(day[4:6]), int(day[6:8]), 1)
    window_start = datetime(date.year, date.month, date.day, FIRST_HOUR, 0)
    window_end = window_start + timedelta(hours=N_HOURS)
    arm_paths = flist[day]['ARM']

    # LIDAR: one hourly profile per site
    # Each list below gets one entry per site, shaped (11 hours, levels).
    u, v, w = [], [], []   # wind components (east, north, vertical)
    ug = []                # horizontal speed sqrt(u^2 + v^2)
    ua = []                # lidar wind direction (degrees)
    precip = []            # surface 'met_spr' value (see NOTE below)
    ma = []                # met minus lidar wind direction
    mws = []               # met minus lidar wind speed

    print('   LIDAR processing', end='', flush=True)
    for path in arm_paths:
        print('.', end='', flush=True)
        fp = nc.Dataset(path, 'r')
        dates = nc.num2date(fp['time'][:], units=fp['time'].units,
                            only_use_cftime_datetimes=False,
                            only_use_python_datetimes=True)
        # BNF CHANGE: keep only 13:00-24:00 UTC, so later data in the file cannot leak into the hourly bins.
        in_window = (dates >= window_start) & (dates < window_end)
        if fp['u'].shape[1] != N_LEVELS or not np.any(in_window):
            fp.close()
            continue
        hours = np.array([d.hour for d in dates[in_window]])

        # Average every variable to 11 hourly values (13-23 UTC).
        tmp_u = read_hourly(fp, 'u', in_window, hours)
        tmp_v = read_hourly(fp, 'v', in_window, hours)
        tmp_w = read_hourly(fp, 'w', in_window, hours)
        tmp_ws = read_hourly(fp, 'wind_speed', in_window, hours)
        tmp_a = read_hourly(fp, 'wind_direction', in_window, hours,
                            is_angle=True)
        tmp_met_dir = read_hourly(fp, 'met_wdir', in_window, hours,
                                  is_angle=True)
        tmp_met_spd = read_hourly(fp, 'met_wspd', in_window, hours)
        # NOTE: the original stores 'met_spr' as precipitation. Check the variable's long_name in the BNF files before interpreting it.
        tmp_met_spr = read_hourly(fp, 'met_spr', in_window, hours)
        # NOTE: assumed to be metres above ground; confirm the 'height' attributes, since the density weights depend on it.
        z = np.asarray(np.ma.getdata(fp['height'][:]), dtype=float)
        fp.close()

        # Heights are taken from the first file and compared afterwards.
        if not height_set:
            out['height'][:] = z[:N_HEIGHT]
            height_set = True
        elif abs(out['height'][25] - z[25]) > 1:
            print(out['height'][25] - z[25])
            print('ERROR! Height Incorrect ', flush=True)

        rho = calculate_air_density(z, 0, 1000)  # zero above 1000 m
        ug.append(np.sqrt(tmp_u ** 2 + tmp_v ** 2))
        u.append(tmp_u)
        v.append(tmp_v)
        w.append(tmp_w)
        ua.append(tmp_a)
        precip.append(tmp_met_spr)
        # Compare the surface met station with lidar level index 9 (see out['height'][9] for its altitude).
        # NOTE: the direction difference is not wrapped to +/-180 degrees, so values near north can look large.
        ma.append(tmp_met_dir[:, 0] - tmp_a[:, 9])
        mws.append(tmp_met_spd[:, 0] - tmp_ws[:, 9])

        # Record which hours and heights this site reported valid winds.
        site_u = tmp_u[:, :N_HEIGHT]
        valid = np.isfinite(site_u) & (site_u >= -100)
        s = SITES.index(site_of(path))
        out['sites_repo'][di, :, :, s] = np.where(valid, 1.0, 0.0)

    print('COMPLETE')
    if not u:
        # The day stays all-NaN in the output file.
        print('   No usable lidar files; skipping this day')
        continue

    # Stack the sites: arrays shaped (sites, 11 hours, 164 levels).
    u = np.array(u)[:, :, :N_HEIGHT]
    v = np.array(v)[:, :, :N_HEIGHT]
    w = np.array(w)[:, :, :N_HEIGHT]
    ug = np.array(ug)[:, :, :N_HEIGHT]
    ua = np.array(ua)[:, :, :N_HEIGHT]
    precip = np.array(precip)
    ma = np.array(ma)
    mws = np.array(mws)
    # (Missing values are already NaN from hourly_mean.)

    # Largest met-vs-lidar disagreement across sites, per hour (QC).
    out['wind_speed_diff'][di, :] = np.nanmax(np.abs(mws), axis=0)
    out['wind_a_diff'][di, :] = np.nanmax(np.abs(ma), axis=0)
    # Circular mean direction and mean speed across sites.
    out['wind_a'][di, :, :] = np.rad2deg(
        circmean(np.deg2rad(ua), axis=0, nan_policy='omit'))
    out['wind_speed'][di, :, :] = np.nanmean(ug, axis=0)
    # Averaged over sites AND hours: one daily value, repeated per hour.
    out['precip'][di, :] = np.nanmean(precip, axis=(0, 1))

    # Dispersive part: variance of the hourly winds ACROSS sites (spatial disagreement). Mean part: square of the cross-site mean.
    up2 = np.nanvar(u, axis=0, ddof=1)
    vp2 = np.nanvar(v, axis=0, ddof=1)
    wp2 = np.nanvar(w, axis=0, ddof=1)
    u2 = np.nanmean(u, axis=0) ** 2
    v2 = np.nanmean(v, axis=0) ** 2
    w2 = np.nanmean(w, axis=0) ** 2

    # Layer weight: density times layer thickness (heights zero density above 1000 m), so the energies integrate over the lowest 1 km.
    weight = rho[:N_HEIGHT] * (z[1:N_HEIGHT + 1] - z[:N_HEIGHT])
    out['DKE_xy'][di, :, :] = weight * 0.5 * (up2 + vp2)  # horizontal
    out['DKE_z'][di, :, :] = weight * 0.5 * wp2           # vertical
    out['MKE_xy'][di, :, :] = weight * 0.5 * (u2 + v2)
    out['MKE_z'][di, :, :] = weight * 0.5 * w2
    out['weighting'][di, :, :] = weight

    # LST: surface temperature descriptor
    std, lhet, tmean, std_site = [], [], [], []
    std_par, std_perp = [], []
    xgrad, ygrad, xgrad2, ygrad2 = [], [], [], []
    # Sites with an ARM file today (used to sample the LST field).
    sites_today = sorted({site_of(p) for p in arm_paths})
    print('   LST processing...', end='', flush=True)

    # One or two GOES files per day (15 and 16 UTC).
    for path in flist[day]['GOES']:
        with rasterio.open(path) as src:
            data = src.read(1)
            # Pixel (row, col) of each lidar site; index takes (lon, lat) in the raster's own coordinate system. NOTE: the raster
            # must be in lon/lat, or the site coordinates need converting.
            site_rc = {s: src.index(*LONLAT[s]) for s in sites_today}

        # Skip hazy files (same test as the clear-sky screen).
        n_bad = np.sum((data < 1) | np.isnan(data))
        if n_bad > CLEAR_SKY_MAX_BAD * data.size:
            continue

        # Fill gaps with the field mean, then smooth to isolate the mesoscale pattern (Gaussian filter, sigma = 3 pixels).
        tmp = np.copy(data)
        tmp[np.isnan(tmp)] = np.mean(tmp[~np.isnan(tmp)])
        tmp[tmp == 0] = np.mean(tmp[tmp != 0])
        lst_meso = ndimage.gaussian_filter(tmp, sigma=3, mode='reflect')

        # Descriptors of the smoothed field.
        std.append(np.nanstd(lst_meso))
        lhet.append(calculate_length_scale(lst_meso, undef=0))
        tmean.append(np.nanmean(lst_meso))
        # NOTE: np.gradient returns the derivative along rows first, so 'xgrad' (grad[0]) is the row-direction derivative despite its
        # name. The original naming is kept.
        grad = np.gradient(lst_meso)
        xgrad.append(np.nanmean(grad[0]))
        ygrad.append(np.nanmean(grad[1]))
        xgrad2.append(np.nanmean(np.abs(grad[0])))
        ygrad2.append(np.nanmean(np.abs(grad[1])))

        # Rotate the field to the mean lidar wind direction (circular mean over 15-20 UTC and the lowest five levels).
        wind_dir = np.rad2deg(circmean(
            np.deg2rad(out['wind_a'][di, 2:8, 0:5]), nan_policy='omit'))
        lst_rot = ndimage.rotate(lst_meso, wind_dir, cval=np.nan)

        # Std along each row, weighted by the number of valid pixels in that row, then summed (a pixel-weighted mean of row stds).
        row_std = np.nanstd(lst_rot, axis=1)
        row_cnt = np.nansum(~np.isnan(lst_rot), axis=1)
        row_std = row_std * row_cnt / np.nansum(row_cnt)
        std_perp.append(np.nansum(row_std))

        # Same along each column.
        col_std = np.nanstd(lst_rot, axis=0)
        col_cnt = np.nansum(~np.isnan(lst_rot), axis=0)
        col_std = col_std * col_cnt / np.nansum(col_cnt)
        std_par.append(np.nansum(col_std))

        # LST at each lidar site. NOTE: rasterio returns (row, col) but the original indexes [col, row]. Verify this on a known pixel.
        t_sites = [lst_meso[rc[1], rc[0]] for rc in site_rc.values()]
        std_site.append(np.nanstd(t_sites))
    print('COMPLETE', flush=True)

    # Combine the one or two GOES files into one daily value each.
    out['lst_lhet'][di] = np.nanmean(lhet)
    out['lst_std'][di] = np.nanmean(std)
    out['lst_mean'][di] = np.nanmean(tmean)
    out['lst_std_site'][di] = np.nanmean(std_site)
    out['lst_std_parallel'][di] = np.nanmean(std_par)
    out['lst_std_perpendicular'][di] = np.nanmean(std_perp)
    # Angle of the mean LST gradient vector relative to the x axis.
    out['lst_a'][di] = angle_between(
        [-np.nanmean(xgrad), np.nanmean(ygrad)], [1, 0])
    out['lst_a2'][di] = angle_between(
        [np.nanmean(xgrad2), np.nanmean(ygrad2)], [1, 0])

    # ---- 5c. HRRR (optional): large-scale vorticity and pressure ------
    if not SKIP_HERBIE:
        print('   HERBIE processing...', end='', flush=True)
        forecast = f'{date.year:04d}-{date.month:02d}-{date.day:02d} 15:00'
        hrrr = Herbie(forecast, model='hrrr', product='sfc', fxx=0,
                      save_dir=OUT_DIR + 'herbie')
        vort = 0
        gout = np.nan  # FIX: the original left this undefined on errors
        try:
            # NOTE: the list positions below depend on the Herbie and
            # cfgrib versions; confirm they select the intended fields.
            ds_uv = hrrr.xarray(':[UV]GRD:', remove_grib=False)
            ds_sp = hrrr.xarray(':PRES', remove_grib=False)
            pres = ds_sp[4]['sp'].values[HRRR_Y, HRRR_X]
            grad = np.gradient(pres)
            gout = max(np.abs(np.nanmean(grad[0])),
                       np.abs(np.nanmean(grad[1])))

            winds = ds_uv[2]
            uvort = (winds['u'].values[2, HRRR_Y, HRRR_X]
                     * (units.meter / units.second))
            vvort = (winds['v'].values[2, HRRR_Y, HRRR_X]
                     * (units.meter / units.second))
            lat = winds['latitude'].values[HRRR_Y, 0]
            lon = winds['longitude'].values[0, HRRR_X]
            dx, dy = mpcalc.lat_lon_grid_deltas(lon, lat)
            vort = mpcalc.vorticity(uvort, vvort, dx=dx, dy=dy)
            for dataset in ds_uv:
                dataset.close()
        except (FileNotFoundError, ValueError):
            vort = 0
        except Exception:
            vort = 0
            gout = -0.0001
        out['vort'][di] = np.nanmax(np.abs(np.array(vort)))
        out['pgrad'][di] = gout
        print('COMPLETE')

    print('   Day complete')



# STEP 6: Write the NetCDF output file



if NEW_FILE:
    fp_out = nc.Dataset(OUT_FILE, 'w')
    fp_out.createDimension('date', size=n_days)
    fp_out.createDimension('hour', size=N_HOURS)
    fp_out.createDimension('height', size=N_HEIGHT)
    fp_out.createDimension('sites', size=n_sites)
else:
    fp_out = nc.Dataset(OUT_FILE, 'r+')

for name, value in out.items():
    if name == 'sites':
        # Site names are stored as a global attribute, not a variable.
        fp_out.sites = ', '.join(value)
        continue
    if name not in fp_out.variables:
        fp_out.createVariable(name, 'f8', dimensions=dims_for(name))
    fp_out[name][:] = np.array(value)

if DEBUG:
    print(out['date'])
    print(fp_out['date'][:])

fp_out.close()
