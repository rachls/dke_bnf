"""Inspect one ARM file per datastream to verify the expected variables."""

import glob
import os
import sys

import netCDF4 as nc

# Set this to the directory containing the three bnfdlprofwind4news*.c1 folders.
RAW_DIR = '/stor/soteria/hydro/private/rls134/data/dke_bnf/ARM/raw'

REQUIRED = ('time', 'height', 'u', 'v', 'w', 'wind_speed', 'wind_direction',
            'met_wdir', 'met_wspd', 'met_spr')
STREAMS = ('M1', 'S20', 'S40')


def inspect_stream(raw_dir, stream):
    """Print the shape of each required variable for one datastream."""
    pattern = os.path.join(raw_dir, f'bnfdlprofwind4news{stream}.c1', '*.nc')
    files = sorted(glob.glob(pattern))
    if not files:
        print(f'{stream}: no files found matching {pattern}')
        return

    path = files[0]
    print(f'{stream}: {len(files)} files, inspecting {path}')
    with nc.Dataset(path) as fp:
        for name in REQUIRED:
            if name in fp.variables:
                print(f'  {name}: {fp[name].shape}')
            else:
                print(f'  {name}: MISSING')
        for key in ('lat', 'lon'):
            if key in fp.variables:
                print(f'  {key} = {float(fp[key][:]):.5f}')


def main():
    if not os.path.isdir(RAW_DIR):
        sys.exit(f'Directory not found: {RAW_DIR}')

    for stream in STREAMS:
        inspect_stream(RAW_DIR, stream)


if __name__ == '__main__':
    main()
