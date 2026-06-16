#!/usr/bin/env python3
"""
Death detection and backward window generation for TTD prediction.
"""
import numpy as np
import pandas as pd


def detect_death(mt_series):
    """
    Find time of death: the last minute with MT > 0, confirmed by 48 subsequent
    hours in which all MT values are zero.

    Args:
        mt_series: pd.Series with DatetimeIndex, 1-min MT values

    Returns:
        pd.Timestamp of last movement, or None if death cannot be confirmed.
    """
    nonzero = mt_series[mt_series > 0]
    if len(nonzero) == 0:
        return None

    last_activity = nonzero.index[-1]
    post = mt_series[mt_series.index > last_activity]

    if len(post) == 0:
        return None

    # Need at least 48 hours of subsequent recording
    hours_after = (post.index[-1] - last_activity).total_seconds() / 3600
    if hours_after < 48:
        return None

    # All minutes within 48h of last activity must be zero
    confirm_end = last_activity + pd.Timedelta(hours=48)
    post_48h = post[post.index <= confirm_end]
    if not (post_48h == 0).all():
        return None

    return last_activity


def generate_windows(mt_series, death_time, lights_on=9, min_data_minutes=55):
    """
    Yield non-overlapping 1-hour windows anchored backward from death.

    Window TTD=1 spans [death_time - 1h, death_time).
    Window TTD=k spans [death_time - k*h, death_time - (k-1)*h).

    Constraints:
      - window_end > recording_start + 24h  (circadian lookback requires full 24h)
      - window has >= min_data_minutes of actual data

    Args:
        mt_series: pd.Series with DatetimeIndex, MT values
        death_time: pd.Timestamp of last movement
        lights_on: ZT0 clock hour (default 9)
        min_data_minutes: minimum data points required in window

    Yields dicts with: window_start, window_end, ttd, age_hours, zt_sin, zt_cos
    """
    recording_start = mt_series.index[0]
    earliest_valid_end = recording_start + pd.Timedelta(hours=24)

    ttd = 1
    while True:
        window_end = death_time - pd.Timedelta(hours=ttd - 1)
        window_start = window_end - pd.Timedelta(hours=1)

        if window_end <= earliest_valid_end:
            break
        if window_start < recording_start:
            break

        window_data = mt_series[
            (mt_series.index >= window_start) & (mt_series.index < window_end)
        ]
        if len(window_data) < min_data_minutes:
            ttd += 1
            continue

        zt_raw = ((window_start.hour + window_start.minute / 60) - lights_on) % 24
        zt_sin = float(np.sin(2 * np.pi * zt_raw / 24))
        zt_cos = float(np.cos(2 * np.pi * zt_raw / 24))
        age_hours = float((window_start - recording_start).total_seconds() / 3600)

        yield {
            'window_start': window_start,
            'window_end': window_end,
            'ttd': ttd,
            'age_hours': age_hours,
            'zt_sin': zt_sin,
            'zt_cos': zt_cos,
        }

        ttd += 1
