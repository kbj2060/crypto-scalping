"""Build the ETH zigzag exit-layer dataset (2026-08-10).

Mirrors scripts/build_btc_zigzag_exit_layer_labels_20260810.py -- see that file's docstring
for the full design rationale (causal bars_since_last_pivot + trailing efficiency-ratio
features, oracle exit-now-vs-hold cost-aware target).

theta/eff_window/horizon are NOT assumed to transfer from BTC unchanged: this project has
repeatedly found BTC-tuned parameters fail to transfer to ETH (regime sizing, multislot
capacity, exit-logic axis -- see memory). main() prints the same wave-duration and class-
balance diagnostics used to choose the BTC values so a mismatch is visible before this
dataset is used for anything downstream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
OUT_DIR = ROOT / "tmp/eth_zigzag_exit_layer_20260810"

THETA = 0.02          # start from the BTC value; diagnostics below confirm or refute it for ETH
EFF_WINDOW = 24
HORIZON_BARS = 48
FEE, SLIP, COST_MULT = 0.0005, 0.0002, 3.0
COST = 2 * (FEE + SLIP) * COST_MULT

TRAIN_END = pd.Timestamp("2025-08-31 23:59:59")
VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")


def zigzag_oracle(close: np.ndarray, threshold: float = 0.04) -> tuple[np.ndarray, list[int]]:
    """Copied verbatim from scripts/test_statistical_jump_model_regimes_20260808.py:140.
    Look-ahead by design -- feeds the y target only, never a live feature."""
    n = len(close)
    hi_i = lo_i = 0
    up: bool | None = None
    ext_i = 0
    pivots: list[int] = []
    for t in range(1, n):
        if close[t] > close[hi_i]:
            hi_i = t
        if close[t] < close[lo_i]:
            lo_i = t
        if up is None:
            if close[t] >= close[lo_i] * (1 + threshold):
                up, ext_i = True, t
                pivots.append(lo_i)
            elif close[t] <= close[hi_i] * (1 - threshold):
                up, ext_i = False, t
                pivots.append(hi_i)
        elif up:
            if close[t] > close[ext_i]:
                ext_i = t
            elif close[t] <= close[ext_i] * (1 - threshold):
                pivots.append(ext_i)
                up, ext_i = False, t
        else:
            if close[t] < close[ext_i]:
                ext_i = t
            elif close[t] >= close[ext_i] * (1 + threshold):
                pivots.append(ext_i)
                up, ext_i = True, t
    direction = np.zeros(n, dtype=np.int8)
    if len(pivots) >= 2:
        first_up = close[pivots[1]] > close[pivots[0]]
        bounds = pivots + [n - 1]
        d = 1 if first_up else -1
        for i in range(len(bounds) - 1):
            direction[bounds[i]: bounds[i + 1] + 1] = d
            d = -d
    return direction, pivots


def causal_features(close: np.ndarray, pivots: list[int], eff_window: int) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    piv = np.asarray(pivots, dtype=np.int64)
    t_idx = np.arange(n)

    last_pos = np.searchsorted(piv, t_idx, side="right") - 1
    bars_since_pivot = np.full(n, np.nan)
    has_past_pivot = last_pos >= 0
    bars_since_pivot[has_past_pivot] = t_idx[has_past_pivot] - piv[last_pos[has_past_pivot]]

    logp = np.log(close)
    d = np.abs(np.diff(logp))
    cs = np.concatenate([[0.0], np.cumsum(d)])
    net_back = np.full(n, np.nan)
    gross_back = np.full(n, np.nan)
    net_back[eff_window:] = logp[eff_window:] - logp[:-eff_window]
    gross_back[eff_window:] = cs[eff_window:] - cs[:-eff_window]
    eff_ratio_trailing = net_back / np.where(gross_back > 1e-12, gross_back, np.nan)

    return bars_since_pivot, eff_ratio_trailing


def exit_now_vs_hold_target(close: np.ndarray, direction: np.ndarray, horizon: int, cost: float) -> np.ndarray:
    n = len(close)
    side = direction.astype(float)
    fwd_ret = np.full(n, np.nan)
    fwd_ret[: n - horizon] = (close[horizon:] - close[: n - horizon]) / close[: n - horizon]
    aligned = side * fwd_ret
    y = np.where(aligned <= cost, 1.0, 0.0)
    y[~np.isfinite(aligned)] = np.nan
    y[side == 0] = np.nan
    return y


def split_label(ts: pd.Series) -> np.ndarray:
    out = np.full(len(ts), "other", dtype=object)
    out[(ts <= TRAIN_END).to_numpy()] = "train"
    out[((ts >= VAL_START) & (ts <= VAL_END)).to_numpy()] = "val"
    out[((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()] = "oos"
    return out


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(PANEL_PATH, usecols=["timestamp", "close"])
    panel["timestamp"] = pd.to_datetime(panel["timestamp"])
    panel = panel.sort_values("timestamp").reset_index(drop=True)
    close = panel["close"].to_numpy(dtype=np.float64)
    n = len(close)

    direction, pivots = zigzag_oracle(close, threshold=THETA)
    bars_since_pivot, eff_ratio_trailing = causal_features(close, pivots, EFF_WINDOW)
    y = exit_now_vs_hold_target(close, direction, HORIZON_BARS, COST)

    out = panel.copy()
    out["bars_since_last_pivot"] = bars_since_pivot
    out["eff_ratio_2h_trailing"] = eff_ratio_trailing
    out["y_exit_now"] = y
    out["split"] = split_label(panel["timestamp"])
    dataset_path = OUT_DIR / "dataset.csv"
    out.to_csv(dataset_path, index=False)

    durations = np.diff(np.asarray(pivots, dtype=np.int64))
    valid_feat = np.isfinite(bars_since_pivot) & np.isfinite(eff_ratio_trailing)
    valid_y = np.isfinite(y)

    per_split = {}
    for name in ("train", "val", "oos"):
        m = (out["split"] == name).to_numpy() & valid_y
        if m.sum() > 0:
            per_split[name] = {"n": int(m.sum()), "exit_now_frac": round(float(out["y_exit_now"][m].mean()), 4)}

    summary = {
        "asset": "ETH",
        "theta": THETA,
        "eff_window_bars": EFF_WINDOW,
        "horizon_bars": HORIZON_BARS,
        "round_trip_cost_pct": round(COST * 100, 3),
        "n_bars": n,
        "n_pivots": len(pivots),
        "wave_duration_bars": {
            "p10": float(np.percentile(durations, 10)),
            "p50": float(np.percentile(durations, 50)),
            "p90": float(np.percentile(durations, 90)),
            "mean": float(durations.mean()),
        },
        "causal_feature_coverage": {"n_valid": int(valid_feat.sum()), "n_total": n},
        "feature_correlation_bars_since_pivot_vs_eff_ratio": round(
            float(np.corrcoef(bars_since_pivot[valid_feat], eff_ratio_trailing[valid_feat])[0, 1]), 4),
        "y_class_balance_overall": {
            "exit_now_1": round(float(np.mean(y[valid_y] == 1)), 4),
            "keep_holding_0": round(float(np.mean(y[valid_y] == 0)), 4),
        },
        "y_class_balance_by_split": per_split,
        "dataset_path": str(dataset_path.relative_to(ROOT)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {dataset_path}", flush=True)
    print(f"wrote {OUT_DIR / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
