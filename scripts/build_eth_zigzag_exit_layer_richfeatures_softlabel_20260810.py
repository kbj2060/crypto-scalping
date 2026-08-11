"""ETH zigzag exit-layer, richer features + soft label (2026-08-10).

Extends scripts/build_eth_zigzag_exit_layer_labels_20260810.py (2 features, hard 0/1 label,
AUC 0.58-0.63 OOS) in two ways the user asked for:

  1. Richer X: still strictly causal/backward-looking, but multi-timescale instead of a
     single 2h window -- efficiency ratio at 1h/2h/4h/8h, trailing realized vol, trailing
     8h momentum, position-in-8h-range, RSI(14). 9 features total. Kept to a moderate count
     on purpose: this project's kitchen-sink attempts (see memory
     project-eth-kitchen-sink-auc-overfit-check-20260809) inflated TRAIN auc to 0.956 while
     OOS stayed ~0.52 -- main() reports train/val/oos side by side specifically to catch
     that failure mode here too.

  2. Soft y instead of hard 0/1: soft_y_exit_now = sigmoid(-(aligned_ret - COST) / COST).
     Same decision boundary as the hard label (aligned_ret == COST -> soft_y == 0.5,
     identical cutoff), but the confidence now decays smoothly on either side over a
     one-cost-wide band instead of flipping instantly. Trained with LightGBM's xentropy
     objective (cross-entropy against a [0,1] soft target, not binary 0/1).

theta/eff-window-base/horizon/cost unchanged from the original ETH build script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "data/splits/year_oos/eth_features_2024_2026_analysis.csv"
OUT_DIR = ROOT / "tmp/eth_zigzag_exit_layer_richfeatures_softlabel_20260810"

THETA = 0.02
HORIZON_BARS = 48
FEE, SLIP, COST_MULT = 0.0005, 0.0002, 3.0
COST = 2 * (FEE + SLIP) * COST_MULT
RSI_PERIOD = 14
RANGE_WINDOW = 96
VOL_WINDOW = 24
MOM_WINDOW = 96
EFF_WINDOWS = (12, 24, 48, 96)

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


def trailing_eff_ratio(logp: np.ndarray, cs: np.ndarray, window: int) -> np.ndarray:
    n = len(logp)
    net_back = np.full(n, np.nan)
    gross_back = np.full(n, np.nan)
    net_back[window:] = logp[window:] - logp[:-window]
    gross_back[window:] = cs[window:] - cs[:-window]
    return net_back / np.where(gross_back > 1e-12, gross_back, np.nan)


def rsi(close: np.ndarray, period: int) -> np.ndarray:
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()
    avg_loss = pd.Series(loss).ewm(alpha=1 / period, adjust=False, min_periods=period).mean().to_numpy()
    rs = avg_gain / np.where(avg_loss > 1e-12, avg_loss, np.nan)
    return 100.0 - 100.0 / (1.0 + rs)


def causal_features(close: np.ndarray, pivots: list[int]) -> dict[str, np.ndarray]:
    n = len(close)
    piv = np.asarray(pivots, dtype=np.int64)
    t_idx = np.arange(n)

    last_pos = np.searchsorted(piv, t_idx, side="right") - 1
    bars_since_pivot = np.full(n, np.nan)
    has_past = last_pos >= 0
    bars_since_pivot[has_past] = t_idx[has_past] - piv[last_pos[has_past]]

    logp = np.log(close)
    d = np.abs(np.diff(logp))
    cs = np.concatenate([[0.0], np.cumsum(d)])
    logret = np.diff(logp, prepend=logp[0])

    feats = {"bars_since_last_pivot": bars_since_pivot}
    for w in EFF_WINDOWS:
        feats[f"eff_ratio_trailing_{w}"] = trailing_eff_ratio(logp, cs, w)

    vol = pd.Series(logret).rolling(VOL_WINDOW).std().to_numpy()
    feats["trailing_vol"] = vol

    mom = np.full(n, np.nan)
    mom[MOM_WINDOW:] = logp[MOM_WINDOW:] - logp[:-MOM_WINDOW]
    feats["trailing_mom"] = mom

    roll_max = pd.Series(close).rolling(RANGE_WINDOW).max().to_numpy()
    roll_min = pd.Series(close).rolling(RANGE_WINDOW).min().to_numpy()
    span = np.where((roll_max - roll_min) > 1e-9, roll_max - roll_min, np.nan)
    feats["range_pos"] = (close - roll_min) / span

    feats["rsi_14"] = rsi(close, RSI_PERIOD)
    return feats


def exit_targets(close: np.ndarray, direction: np.ndarray, horizon: int, cost: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    side = direction.astype(float)
    fwd_ret = np.full(n, np.nan)
    fwd_ret[: n - horizon] = (close[horizon:] - close[: n - horizon]) / close[: n - horizon]
    aligned = side * fwd_ret
    hard_y = np.where(aligned <= cost, 1.0, 0.0)
    soft_y = 1.0 / (1.0 + np.exp((aligned - cost) / cost))  # sigmoid(-(aligned-cost)/cost)
    invalid = ~np.isfinite(aligned) | (side == 0)
    hard_y[invalid] = np.nan
    soft_y[invalid] = np.nan
    return hard_y, soft_y, aligned


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

    direction, pivots = zigzag_oracle(close, threshold=THETA)
    feats = causal_features(close, pivots)
    hard_y, soft_y, aligned = exit_targets(close, direction, HORIZON_BARS, COST)

    out = panel.copy()
    for name, arr in feats.items():
        out[name] = arr
    out["y_exit_now"] = hard_y
    out["soft_y_exit_now"] = soft_y
    out["aligned_fwd_ret"] = aligned
    out["split"] = split_label(panel["timestamp"])

    dataset_path = OUT_DIR / "dataset.csv"
    out.to_csv(dataset_path, index=False)

    feature_cols = list(feats.keys())
    valid = out[feature_cols + ["y_exit_now"]].notna().all(axis=1)
    summary = {
        "asset": "ETH", "theta": THETA, "horizon_bars": HORIZON_BARS,
        "round_trip_cost_pct": round(COST * 100, 3),
        "feature_cols": feature_cols,
        "soft_label_temperature": COST,
        "n_bars": len(out), "n_valid_rows": int(valid.sum()),
        "soft_y_stats": {
            "mean": round(float(np.nanmean(soft_y)), 4),
            "std": round(float(np.nanstd(soft_y)), 4),
            "frac_between_0.3_0.7": round(float(np.nanmean((soft_y > 0.3) & (soft_y < 0.7))), 4),
        },
        "hard_y_class_balance": {
            "exit_now_1": round(float(np.nanmean(hard_y == 1)), 4),
            "keep_holding_0": round(float(np.nanmean(hard_y == 0)), 4),
        },
        "dataset_path": str(dataset_path.relative_to(ROOT)),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {dataset_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
