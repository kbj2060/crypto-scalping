"""Causal CUSUM event detection and symmetric directional triple-barrier labels."""
from __future__ import annotations

import numpy as np


def causal_cusum_events(close: np.ndarray, volatility: np.ndarray, multiplier: float) -> np.ndarray:
    if multiplier <= 0:
        raise ValueError("multiplier must be positive")
    close, volatility = np.asarray(close, float), np.asarray(volatility, float)
    if len(close) != len(volatility):
        raise ValueError("close and volatility must align")
    logret = np.diff(np.log(close), prepend=np.log(close[0]))
    up = down = 0.0; events = []
    for i in range(1, len(close)):
        threshold = multiplier * max(float(volatility[i]), 1e-3)
        up, down = max(0.0, up + logret[i]), min(0.0, down + logret[i])
        if up >= threshold or down <= -threshold:
            events.append(i); up = down = 0.0
    return np.asarray(events, dtype=np.int64)


def triple_barrier_direction(
    *, entry: float, high: np.ndarray, low: np.ndarray, close: np.ndarray, move: float
) -> int:
    """Return 2=LONG, 0=FLAT, 1=SHORT; intrabar dual touches are conservatively FLAT."""
    upper, lower = entry * (1.0 + move), entry * (1.0 - move)
    for hi, lo in zip(np.asarray(high, float), np.asarray(low, float)):
        hit_up, hit_down = hi >= upper, lo <= lower
        if hit_up and hit_down:
            return 0
        if hit_up:
            return 2
        if hit_down:
            return 1
    return 0
