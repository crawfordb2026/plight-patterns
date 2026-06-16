#!/usr/bin/env python3
"""
Feature computation for TTD prediction.

Sleep features: computed from the 1-hour prediction window.
Circadian features: computed from the 24-hour lookback ending at window end.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


SLEEP_FEATURE_NAMES = [
    'total_sleep_min', 'sleep_bout_count', 'mean_bout_min',
    'max_bout_min', 'p_wake', 'p_doze', 'waso_min',
]

CIRCADIAN_FEATURE_NAMES = [
    'mesor', 'amplitude', 'phase', 'iv', 'ra', 'l5', 'm10',
]


def rle(seq):
    """Run-length encoding. Returns (values, lengths)."""
    arr = np.asarray(seq)
    if len(arr) == 0:
        return np.array([]), np.array([], dtype=int)
    changes = np.concatenate(([True], np.diff(arr) != 0))
    indices = np.concatenate((np.where(changes)[0], [len(arr)]))
    lengths = np.diff(indices).astype(int)
    values = arr[np.where(changes)[0]]
    return values, lengths


def compute_sleep_features(mt_window, sleep_threshold_min=5, bin_min=1):
    """
    Compute sleep features from a 1-hour window of 1-min MT data.

    Args:
        mt_window: pd.Series of MT values (up to 60 values)
        sleep_threshold_min: consecutive inactive minutes that count as sleep
        bin_min: bin duration in minutes

    Returns dict of sleep features.
    """
    nan_result = {k: np.nan for k in SLEEP_FEATURE_NAMES}
    if len(mt_window) == 0:
        return nan_result

    inactive = (mt_window.values == 0)
    vals, lens = rle(inactive)

    # Reconstruct per-minute sleep boolean
    sleep_vec = np.zeros(len(inactive), dtype=bool)
    pos = 0
    for v, l in zip(vals, lens):
        if v and l >= sleep_threshold_min:
            sleep_vec[pos:pos + l] = True
        pos += l

    sleep_run_lengths = lens[(vals == True) & (lens >= sleep_threshold_min)]

    total_sleep_min = float(sleep_vec.sum() * bin_min)
    bout_count = int(len(sleep_run_lengths))
    mean_bout = float(sleep_run_lengths.mean() * bin_min) if bout_count > 0 else 0.0
    max_bout = float(sleep_run_lengths.max() * bin_min) if bout_count > 0 else 0.0

    N = len(sleep_vec)
    N_S = int(sleep_vec.sum())
    N_W = N - N_S

    if N > 1:
        N_S_to_W = int(((~sleep_vec[1:]) & sleep_vec[:-1]).sum())
        N_W_to_S = int((sleep_vec[1:] & (~sleep_vec[:-1])).sum())
    else:
        N_S_to_W = N_W_to_S = 0

    p_wake = N_S_to_W / N_S if N_S > 0 else 0.0
    p_doze = N_W_to_S / N_W if N_W > 0 else 0.0

    if N_S > 0:
        first_sleep = int(np.where(sleep_vec)[0][0])
        waso_min = float((~sleep_vec[first_sleep:]).sum() * bin_min)
    else:
        waso_min = 0.0

    return {
        'total_sleep_min': total_sleep_min,
        'sleep_bout_count': float(bout_count),
        'mean_bout_min': mean_bout,
        'max_bout_min': max_bout,
        'p_wake': p_wake,
        'p_doze': p_doze,
        'waso_min': waso_min,
    }


def _hourly_zt_totals(mt_24h, lights_on=9):
    """
    Compute total MT per ZT hour (0–23) from a minute-level Series.
    Returns pd.Series indexed 0–23.
    """
    zt_raw = ((mt_24h.index.hour + mt_24h.index.minute / 60) - lights_on) % 24
    zt_floor = np.floor(zt_raw).astype(int)
    totals = mt_24h.groupby(zt_floor).sum()
    return totals.reindex(range(24), fill_value=0).astype(float)


def _cosinor(hourly_totals, period=24):
    """
    Fit cosinor model to 24 hourly ZT totals.
    Returns (mesor, amplitude, phase_hours).
    """
    x = hourly_totals.values.astype(float)
    zt = np.arange(24)
    cos_t = np.cos(2 * np.pi * zt / period)
    sin_t = np.sin(2 * np.pi * zt / period)
    X = np.column_stack([cos_t, sin_t])

    model = LinearRegression().fit(X, x)
    mesor = float(model.intercept_)
    a, b = float(model.coef_[0]), float(model.coef_[1])
    amplitude = float(np.sqrt(a ** 2 + b ** 2))
    phase_rad = np.arctan2(-b, a)
    phase_hours = float((period * phase_rad / (2 * np.pi)) % period)
    return mesor, amplitude, phase_hours


def _intradaily_variability(hourly_totals):
    """
    IV = [n * sum(x_i - x_{i-1})^2] / [(n-1) * sum(x_i - x_bar)^2]
    Measures within-day fragmentation of the rest-activity rhythm.
    """
    x = hourly_totals.values.astype(float)
    n = len(x)
    if n < 2:
        return np.nan
    x_bar = x.mean()
    denom = np.sum((x - x_bar) ** 2)
    if denom == 0:
        return np.nan
    return float(n * np.sum(np.diff(x) ** 2) / ((n - 1) * denom))


def _l5_m10(hourly_totals):
    """
    L5: mean activity of the least active 5 consecutive hours (circular).
    M10: mean activity of the most active 10 consecutive hours (circular).
    """
    x = hourly_totals.values.astype(float)
    x2 = np.concatenate([x, x])

    l5 = float(min(x2[i:i + 5].mean() for i in range(24)))
    m10 = float(max(x2[i:i + 10].mean() for i in range(24)))
    return l5, m10


def compute_circadian_features(mt_24h, lights_on=9):
    """
    Compute circadian features from a 24-hour lookback window of 1-min MT data.

    Args:
        mt_24h: pd.Series with DatetimeIndex, MT values (~24*60 entries)
        lights_on: ZT0 clock hour (default 9 = 9am)

    Returns dict of circadian features.
    """
    nan_result = {k: np.nan for k in CIRCADIAN_FEATURE_NAMES}
    if len(mt_24h) < 12:
        return nan_result

    hourly = _hourly_zt_totals(mt_24h, lights_on)

    mesor, amplitude, phase = _cosinor(hourly)
    iv = _intradaily_variability(hourly)
    l5, m10 = _l5_m10(hourly)
    total = m10 + l5
    ra = float((m10 - l5) / total) if total > 0 else np.nan

    return {
        'mesor': mesor,
        'amplitude': amplitude,
        'phase': phase,
        'iv': iv,
        'ra': ra,
        'l5': l5,
        'm10': m10,
    }
