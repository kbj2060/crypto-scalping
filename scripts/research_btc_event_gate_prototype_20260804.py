#!/usr/bin/env python3
"""Stage-1 prototype: BTC "Volatility-Aware Extreme Event Gate".

Purpose (see docs/model_contracts/... design discussion 2026-08-04): causalfix_final's
114-col per-bar feature set has been repeatedly confirmed to carry no extractable
directional signal for BTC (see project-btc-cusum-architecture-structural-redesign-
closed-20260804). The only surviving real-but-thin BTC signals are the standalone
GMM volatility (r+0.31~0.33) and Isolation Forest anomaly (r+0.18) detectors, both
data-starved when used directly as h48qual entry filters.

This script does NOT retrain a new classifier. It only evaluates whether combining
the two EXISTING pretrained detectors into a rare-event gate -- multi-timescale
agreement + online conformal-style threshold recalibration -- produces a bar-firing
rate and hit-rate that is dense enough to be tradeable, before any downstream
prediction head (stage 2) is built. If this gate itself shows no lift over base
rate on fresh-forward VAL/OOS, stage 2 should not be built.

Fresh-Forward compliance: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. Gate inputs
(gmm_cluster_rank, gmm_confidence, if_score, rolling threshold) at bar i use only data
available up to and including bar i. The event LABEL used to score gate quality is
necessarily forward-looking (it asks "did a big move happen after this bar") -- it is
an evaluation target only, never fed back into the gate's own decision at bar i.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OHLC_CSVS = [
    ROOT / "data/splits/year_oos/btc_features_2025.csv",
    ROOT / "data/splits/year_oos/btc_features_2026.csv",
]
GMM_SCORES = ROOT / "tmp/research_20260802/btc_gmm_volatility_signal_check/gmm_score_series_full.csv"
IF_SCORES = ROOT / "tmp/research_20260802/btc_isolation_forest_signal_check/if_score_series_full.csv"
OUT_DIR = ROOT / "tmp/research_20260804/btc_event_gate_prototype"

VAL_START, VAL_END = pd.Timestamp("2025-09-01"), pd.Timestamp("2025-12-31 23:59:59")
OOS_START, OOS_END = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-03-31 23:59:59")

ATR_WINDOW = 96          # bars, matches build_omega1_2_triple_barrier_labels_btc_20260708.py
EVENT_HORIZON = 48       # bars (~4h), matches h48qual horizon for comparability
# max_abs_move / atr over EVENT_HORIZON bars has median ~5.3x, 90th pct ~12.6x, 95th pct
# ~16.3x on 2024-2026 BTC 5m data (single-bar ATR vs 4h max excursion) -- thresholds below
# are picked off that empirical distribution, not arbitrary round numbers.
EVENT_ATR_MULT = 13.0     # "big move" alone (~90th percentile) = genuinely extreme, not typical
HIGH_VOL_RANK_MIN = 4     # gmm_cluster_rank >= this (0..5, 5=highest vol) counts as high-vol regime
HIGH_VOL_EVENT_ATR_MULT = 8.0  # smaller move threshold (~75th pct) allowed when already in high-vol regime

ROLL_1H = 12             # bars
ROLL_4H = 48             # bars
AGREEMENT_MIN = 2        # of 3 timescales (5m/1h/4h) must agree for gate to fire

RECAL_WINDOW = 4032      # bars (~2 weeks), trailing window for online threshold recalibration
TARGET_FIRE_RATE = 0.03  # top ~3% of bars, causal rolling quantile


def _load_ohlc() -> pd.DataFrame:
    frames = [pd.read_csv(p, usecols=["timestamp", "open", "high", "low", "close"]) for p in OHLC_CSVS]
    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    return df


def _causal_atr(frame: pd.DataFrame) -> pd.Series:
    high, low, close = frame["high"], frame["low"], frame["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    atr = (tr / close.replace(0.0, np.nan)).rolling(ATR_WINDOW, min_periods=24).mean().shift(1)
    return atr.replace([np.inf, -np.inf], np.nan)


def _event_label(frame: pd.DataFrame, atr: pd.Series, gmm_rank: pd.Series) -> pd.Series:
    """Volatility-aware extreme-event label (paper: Volatility-Aware Extreme Event
    Detection, arXiv 2607.17555) -- large future move OR (high-vol regime AND a
    smaller future move), evaluated at bar i using bars i+1..i+EVENT_HORIZON only."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    entry = frame["close"].to_numpy()
    n = len(frame)
    max_abs_move = np.full(n, np.nan)
    for i in range(n - EVENT_HORIZON - 1):
        e = entry[i]
        if e <= 0:
            continue
        fut_high = high[i + 1 : i + 1 + EVENT_HORIZON]
        fut_low = low[i + 1 : i + 1 + EVENT_HORIZON]
        up_move = fut_high.max() / e - 1.0
        down_move = 1.0 - fut_low.min() / e
        max_abs_move[i] = max(up_move, down_move)
    max_abs_move = pd.Series(max_abs_move, index=frame.index)
    big_move = max_abs_move >= EVENT_ATR_MULT * atr
    high_vol_regime = gmm_rank >= HIGH_VOL_RANK_MIN
    small_move_in_high_vol = high_vol_regime & (max_abs_move >= HIGH_VOL_EVENT_ATR_MULT * atr)
    label = (big_move | small_move_in_high_vol).astype(float)
    label[atr.isna() | max_abs_move.isna()] = np.nan
    return label


def _zscore_causal(s: pd.Series, window: int) -> pd.Series:
    roll = s.rolling(window, min_periods=max(8, window // 4))
    mu, sd = roll.mean(), roll.std()
    return (s - mu) / sd.replace(0.0, np.nan)


def _multi_timescale_gate(gmm_rank: pd.Series, gmm_conf: pd.Series, if_score: pd.Series) -> pd.DataFrame:
    """Creative idea 1: same detectors resampled to 5m/1h/4h via causal rolling means;
    a bar counts as 'flagged' at a given timescale if that timescale's z-score is in
    the extreme tail. Gate fires only when >=AGREEMENT_MIN of 3 timescales agree."""
    raw = gmm_rank.astype(float) * gmm_conf - if_score  # if_score: lower = more anomalous, so subtract
    scales = {"5m": 1, "1h": ROLL_1H, "4h": ROLL_4H}
    flags = {}
    for name, w in scales.items():
        smoothed = raw.rolling(w, min_periods=max(1, w // 2)).mean()
        z = _zscore_causal(smoothed, RECAL_WINDOW)
        flags[f"flag_{name}"] = z >= z.rolling(RECAL_WINDOW, min_periods=RECAL_WINDOW // 4).quantile(1 - TARGET_FIRE_RATE)
    out = pd.DataFrame(flags, index=raw.index)
    out["raw_score"] = raw
    out["agreement"] = out[[f"flag_{n}" for n in scales]].sum(axis=1)
    return out


def _online_conformal_threshold(raw_score: pd.Series) -> pd.Series:
    """Creative idea 2: rolling trailing-window quantile threshold (causal, uses only
    past RECAL_WINDOW bars) recalibrated every bar to keep the target fire rate stable
    as BTC's volatility regime drifts, instead of a fixed hyperparameter threshold."""
    return raw_score.rolling(RECAL_WINDOW, min_periods=RECAL_WINDOW // 4).quantile(1 - TARGET_FIRE_RATE)


def _evaluate_window(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end) & df["event_label"].notna()
    win = df.loc[mask]
    n = len(win)
    base_rate = float(win["event_label"].mean()) if n else float("nan")
    fired = win[win["gate_fired"]]
    n_fired = len(fired)
    precision = float(fired["event_label"].mean()) if n_fired else float("nan")
    lift = precision / base_rate if n_fired and base_rate > 0 else float("nan")
    return {
        "start": str(start.date()),
        "end": str(end.date()),
        "n_bars": int(n),
        "base_event_rate": base_rate,
        "n_fired": int(n_fired),
        "fire_rate": float(n_fired / n) if n else float("nan"),
        "precision_given_fired": precision,
        "lift_vs_base_rate": lift,
    }


def main() -> int:
    ohlc = _load_ohlc()
    gmm = pd.read_csv(GMM_SCORES, usecols=["timestamp", "gmm_cluster_rank", "gmm_confidence"])
    ifs = pd.read_csv(IF_SCORES, usecols=["timestamp", "if_score"])
    gmm["timestamp"] = pd.to_datetime(gmm["timestamp"])
    ifs["timestamp"] = pd.to_datetime(ifs["timestamp"])

    df = ohlc.merge(gmm, on="timestamp", how="inner").merge(ifs, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    atr = _causal_atr(df)
    df["event_label"] = _event_label(df, atr, df["gmm_cluster_rank"])

    gate = _multi_timescale_gate(df["gmm_cluster_rank"], df["gmm_confidence"], df["if_score"])
    df = pd.concat([df, gate], axis=1)
    df["threshold"] = _online_conformal_threshold(df["raw_score"])
    df["gate_fired"] = (df["raw_score"] >= df["threshold"]) & (df["agreement"] >= AGREEMENT_MIN)

    val = _evaluate_window(df, VAL_START, VAL_END)
    oos = _evaluate_window(df, OOS_START, OOS_END)

    result = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "config": {
            "event_horizon_bars": EVENT_HORIZON,
            "event_atr_mult": EVENT_ATR_MULT,
            "high_vol_rank_min": HIGH_VOL_RANK_MIN,
            "high_vol_event_atr_mult": HIGH_VOL_EVENT_ATR_MULT,
            "agreement_min": AGREEMENT_MIN,
            "recal_window_bars": RECAL_WINDOW,
            "target_fire_rate": TARGET_FIRE_RATE,
        },
        "validation_2025_09_to_12": val,
        "oos_2026_01_to_03": oos,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "gate_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)
    df[["timestamp", "raw_score", "threshold", "agreement", "gate_fired", "event_label"]].to_csv(
        OUT_DIR / "gate_series.csv", index=False
    )

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
