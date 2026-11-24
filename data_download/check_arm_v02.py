"""Summarize the time axis of every ARM lidar file."""

import glob
import os

import netCDF4 as nc
import numpy as np
import pandas as pd

RAW_DIR = '/stor/soteria/hydro/private/rls134/data/dke_bnf/ARM/raw'
STREAMS = ('M1', 'S20', 'S40')
FILE_CSV = 'arm_file_summary.csv'
DAY_CSV = 'arm_samples_by_day.csv'


def summarize_file(path):
    """Return a dict describing the time axis of one file."""
    name = os.path.basename(path)
    row = {'file': name, 'date': name.split('.')[2]}
    try:
        with nc.Dataset(path) as fp:
            dates = nc.num2date(fp['time'][:], units=fp['time'].units)
    except (OSError, KeyError, IndexError) as err:
        row['error'] = str(err)
        return row

    row['n_samples'] = len(dates)
    if len(dates) > 1:
        steps = [(b - a).total_seconds()
                 for a, b in zip(dates[:-1], dates[1:])]
        row['step_min'] = round(float(np.median(steps)) / 60, 1)
    if len(dates) > 0:
        row['start_utc'] = dates[0].strftime('%H:%M')
        row['end_utc'] = dates[-1].strftime('%H:%M')
    return row


def main():
    frames = []
    for stream in STREAMS:
        pattern = os.path.join(
            RAW_DIR, f'bnfdlprofwind4news{stream}.c1', '*.nc')
        rows = [summarize_file(p) for p in sorted(glob.glob(pattern))]
        df = pd.DataFrame(rows)
        df.insert(0, 'stream', stream)
        frames.append(df)

    table = pd.concat(frames, ignore_index=True)
    table.to_csv(FILE_CSV, index=False)

    by_day = table.pivot_table(index='date', columns='stream',
                               values='n_samples', aggfunc='sum')
    by_day.to_csv(DAY_CSV)

    print(f'Wrote {FILE_CSV} ({len(table)} files) and {DAY_CSV} '
          f'({len(by_day)} days)')
    n_dup = table.duplicated(['stream', 'date']).sum()
    print(f'Days with more than one file for a stream: {n_dup}\n')

    for stream, df in table.groupby('stream'):
        print(f'=== {stream}: {len(df)} files ===')
        if 'error' in df:
            print(f'Unreadable files: {df["error"].notna().sum()}')
        print('Sampling step (minutes):')
        print(df['step_min'].value_counts(dropna=False).to_string())
        print('Most common samples per file:')
        print(df['n_samples'].value_counts().head(8).to_string())
        print('Shortest files:')
        shortest = df.nsmallest(5, 'n_samples')
        print(shortest[['date', 'n_samples', 'start_utc',
                        'end_utc']].to_string(index=False))
        print()


if __name__ == '__main__':
    main()
