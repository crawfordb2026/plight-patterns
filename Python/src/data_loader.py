#!/usr/bin/env python3
"""
Load and parse DAM monitor files and metadata for TTD analysis.
"""
import numpy as np
import pandas as pd
from pathlib import Path


def _extract_monitor_num(monitor_str):
    """Extract integer monitor number from '51_06_20_25' or '51'."""
    return int(str(monitor_str).split('_')[0])


def parse_metadata(filepath):
    """
    Parse metadata.txt. Returns DataFrame with columns:
      monitor (int), channel (int), fly_id (str), genotype (str).

    Handles Monitor column in '51_06_20_25' or plain '51' format.
    Skips rows where genotype is 'na' / 'nan' / empty.
    """
    rows = []
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines[1:]:
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        monitor_str, channel_str, genotype = parts[0], parts[1], parts[2]
        if genotype.lower() in ('na', 'nan', ''):
            continue
        monitor_num = _extract_monitor_num(monitor_str)
        channel_num = int(channel_str.lower().replace('ch', ''))
        fly_id = f"M{monitor_num}_Ch{channel_num:02d}"
        rows.append({
            'monitor': monitor_num,
            'channel': channel_num,
            'fly_id': fly_id,
            'genotype': genotype,
        })

    if not rows:
        raise ValueError(f"No valid fly entries found in {filepath}")
    return pd.DataFrame(rows)


def _parse_monitor_file_mt(filepath):
    """
    Parse a Monitor*.txt file and return MT readings only.
    Returns DataFrame with columns: datetime, channel, mt.
    """
    df = pd.read_csv(filepath, sep='\t', header=None)

    cols = ['id', 'date', 'time', 'port', 'u1', 'u2', 'u3', 'movement_type', 'z1', 'z2']
    for i in range(1, 33):
        cols.append(f'ch{i}')
    df.columns = cols

    df['datetime'] = pd.to_datetime(
        df['date'] + ' ' + df['time'], format='%d %b %y %H:%M:%S'
    )
    mt_df = df[df['movement_type'] == 'MT'].copy()

    channel_cols = [f'ch{i}' for i in range(1, 33)]
    mt_long = mt_df.melt(
        id_vars=['datetime'], value_vars=channel_cols,
        var_name='ch_col', value_name='mt'
    )
    mt_long['channel'] = mt_long['ch_col'].str.replace('ch', '', regex=False).astype(int)
    mt_long = mt_long[['datetime', 'channel', 'mt']].copy()
    mt_long['mt'] = pd.to_numeric(mt_long['mt'], errors='coerce').fillna(0).astype(int)
    return mt_long


def load_fly_series(monitors_dir, metadata_path):
    """
    Load all Monitor*.txt files and return per-fly MT time series.

    Args:
        monitors_dir: directory containing Monitor*.txt files
        metadata_path: path to metadata.txt

    Returns:
        fly_meta: DataFrame (fly_id, monitor, channel, genotype)
        fly_series: dict mapping fly_id -> pd.Series (DatetimeIndex, MT values)
    """
    monitors_dir = Path(monitors_dir)
    monitor_files = sorted(monitors_dir.glob('Monitor*.txt'))
    if not monitor_files:
        raise FileNotFoundError(f"No Monitor*.txt files found in {monitors_dir}")

    fly_meta = parse_metadata(metadata_path)

    frames = []
    for f in monitor_files:
        stem = f.stem  # e.g. "Monitor51_06_20_25"
        num_str = stem[7:].split('_')[0]   # "51"
        monitor_num = int(num_str)
        mt_df = _parse_monitor_file_mt(f)
        mt_df['monitor'] = monitor_num
        frames.append(mt_df)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['monitor', 'channel', 'datetime']).reset_index(drop=True)

    merged = combined.merge(
        fly_meta[['monitor', 'channel', 'fly_id', 'genotype']],
        on=['monitor', 'channel'],
        how='inner'
    )

    fly_series = {}
    for fly_id, group in merged.groupby('fly_id'):
        s = group.set_index('datetime')['mt'].sort_index()
        s = s[~s.index.duplicated(keep='first')]
        fly_series[fly_id] = s

    return fly_meta, fly_series
