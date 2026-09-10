#!/usr/bin/env python3
"""Phase 1 of the scalping-scale regime LABEL redesign -- user request 2026-09-02 ("레짐은 gbm3
모델로 유지하되 라벨을 좀 바꾸고 싶어. 우린 스캘핑이니까 작은 범위에서 레짐 전환이 잘되는 라벨").

MODEL IS HELD FIXED (GBM3). Only the label changes. This phase measures LABEL GEOMETRY ONLY --
no classifier is trained and **the OOS window is never touched** (TRAIN+VAL 2024-01-01~2026-06-30
only). Learnability costs an OOS look, so it is deliberately deferred until a candidate is chosen.

WHY THE CURRENT LABEL IS SWING-SCALE, NOT SCALPING-SCALE

  features/elite.py::RegimeEngine.compute()'s shortest window is er_24 (24 bars = 2h) and its
  DIRECTION anchor is net_change_48 / ret_48 (4h), with mtf_trend_1h (1h EMA slope) as the
  confirming slope. So "bull" means "the last 2-4 hours have been efficiently up" -- backward-looking
  at swing scale. A scalper holding minutes-to-an-hour is being told about a window that already
  closed.

THE TENSION THIS SWEEP EXISTS TO MEASURE

  Shortening the window raises responsiveness AND flicker together. The raw current label already
  has flip_rate 0.1877, which train_eth_regime_gbm2_trend_chop_20260827.py documented as "visibly
  flickery" against price -- that whole GBM2 project existed to damp it (debounce K=12). So the
  question is not "can we make it faster" (trivially yes) but "is there a scale where it is faster
  AND the transitions still mean something".

SCALE-PARAMETERIZED LABEL FAMILY (generalizes RegimeEngine's own structure, S = scale in bars)

    er_S    = |close - close[-S]|  / sum(|diff|, S)          (efficiency ratio, short leg)
    er_2S   = |close - close[-2S]| / sum(|diff|, 2S)         (long leg)
    net_2S  = close - close[-2S]                             (direction anchor)
    slope_S = EMA(close, S).pct_change()                     (generalizes mtf_trend_1h, which is S=12)
    trend   = (er_S >= T1_S) | (er_2S >= T2_S)
    bull    = trend & net_2S > 0 & slope_S > 0   /  bear = mirror  /  chop = rest

  S=24 reproduces the current label's scale (er_24 / er_48 / net_48 / 1h slope).

  ⭐THRESHOLD CALIBRATION IS MANDATORY, NOT COSMETIC. The efficiency ratio's distribution is
  scale-dependent (for a random walk E[ER] ~ 1/sqrt(N)), so reusing the current 0.20/0.16 thresholds
  at S=6 would inflate trend_share purely mechanically and make every comparison meaningless. T1_S
  and T2_S are therefore percentile-matched ON TRAIN to the CURRENT label's own firing rates
  P(er_24>=0.20) and P(er_48>=0.16), holding base rates comparable across S. Same discipline as
  README ss5.13 (per-horizon K recalibration for the unsigned-magnitude target).

METRICS (all label-only)

  1. class shares            -- degenerate guard. GBM2 found K=48 debounce gives flip_rate 0.0001 but
                                trend_share collapses 0.45->0.12: lock-up, a red flag not a win.
  2. run-length distribution -- THE scalping-relevance metric. A "regime" whose median state lasts
                                2 bars is not a regime, it is a noisy indicator.
  3. flip_rate               -- flicker.
  4. ⭐transition edge       -- what "레짐 전환이 잘된다" actually means: on the bar a FRESH transition
                                into bull (bear) fires, what is the forward return over the next
                                6/12/24 bars, versus the unconditional baseline over the same window?
                                A label whose transitions carry no forward edge is useless no matter
                                how learnable it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features.elite import RegimeEngine  # noqa: E402

# --- GBM3 basis (user directive 2026-09-02: "GBM2가 아니라 GBM3를 기반으로") -------------------
# TRAIN range is read from the DEPLOYED GBM3 artifact itself rather than copied from a sibling
# script, and the two state-machine helpers below are inlined so this study has NO import
# dependency on the GBM2 (2-class trend/chop) lineage at all. The reference label is GBM3's own:
# RegimeEngine 3-class, RAW (no debounce) -- debounce was a GBM2 invention and is treated here as
# an explicitly-flagged optional axis that must earn its place, not as a default.
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
TRAIN_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]
_gbm3_train_range = joblib.load(GBM3_MODEL_PATH)["train_range"]          # "2024-01-01T.. ~ 2026-06-30T.."
TRAIN_START, TRAIN_END = (pd.Timestamp(x.strip()) for x in _gbm3_train_range.split("~"))


def _debounce(raw: np.ndarray, k_bars: int) -> np.ndarray:
    """K-consecutive-bar confirm. k=1 is a no-op (== raw, i.e. GBM3's own convention)."""
    n = len(raw)
    confirmed = np.empty(n, dtype=int)
    confirmed[0] = raw[0]
    candidate, streak = raw[0], 0
    for t in range(1, n):
        if raw[t] == confirmed[t - 1]:
            candidate, streak = confirmed[t - 1], 0
        else:
            streak = streak + 1 if raw[t] == candidate else 1
            candidate = raw[t]
        confirmed[t] = candidate if streak >= k_bars else confirmed[t - 1]
    return confirmed


def _run_lengths(pred: np.ndarray) -> list[int]:
    if len(pred) == 0:
        return []
    lengths, start = [], 0
    for i in range(1, len(pred)):
        if pred[i] != pred[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(pred) - start)
    return lengths

SCALES = (6, 12, 24, 48)          # 30min / 1h / 2h / 4h ; S=24 == the current label's scale
DEBOUNCES = (1, 3, 6, 12)         # 1 = raw (no confirm)
FWD_HORIZONS = (6, 12, 24)        # 30min / 1h / 2h forward, scalping-relevant
OUT_DIR = ROOT / "tmp/eth_regime_scalping_label_geometry_20260902"


def load_train() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"],
                          usecols=["timestamp", "open", "high", "low", "close", "volume"])
              for p in TRAIN_CSVS]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last").reset_index(drop=True)
    return df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)].reset_index(drop=True)


def efficiency_ratio(close: pd.Series, n: int) -> pd.Series:
    diff_abs = close.diff().abs()
    net = (close - close.shift(n)).abs()
    return (net / (diff_abs.rolling(n, min_periods=max(2, n // 6)).sum() + 1e-12)).fillna(0.0)


def scaled_label(close: pd.Series, s: int, t1: float, t2: float) -> np.ndarray:
    """0=bull, 1=bear, 2=chop -- same class coding as the deployed GBM3."""
    er_s, er_2s = efficiency_ratio(close, s), efficiency_ratio(close, 2 * s)
    net_2s = close - close.shift(2 * s)
    slope = close.ewm(span=s, adjust=False).mean().pct_change().fillna(0.0)
    trend = (er_s >= t1) | (er_2s >= t2)
    y = np.full(len(close), 2, dtype=int)
    y[(trend & (net_2s > 0) & (slope > 0)).to_numpy()] = 0
    y[(trend & (net_2s < 0) & (slope < 0)).to_numpy()] = 1
    return y


def transition_edge(y: np.ndarray, close: np.ndarray, h: int) -> dict:
    """Forward return after a FRESH transition into bull/bear, vs the unconditional baseline.

    Signed by the regime's own direction (bull -> +ret, bear -> -ret) so both sides are 'edge in the
    direction the label just claimed'. Baseline is the same-horizon mean |signed| move over all bars,
    computed identically, so the comparison is like-for-like."""
    fwd = np.full(len(close), np.nan)
    fwd[:-h] = close[h:] / close[:-h] - 1.0
    fresh = np.zeros(len(y), dtype=bool)
    fresh[1:] = y[1:] != y[:-1]
    out = {}
    for cls, name, sign in ((0, "bull", 1.0), (1, "bear", -1.0)):
        m = fresh & (y == cls) & np.isfinite(fwd)
        out[f"{name}_n"] = int(m.sum())
        out[f"{name}_fwd_bp"] = float(np.mean(sign * fwd[m]) * 1e4) if m.any() else float("nan")
    base = np.isfinite(fwd)
    out["baseline_abs_bp"] = float(np.mean(np.abs(fwd[base])) * 1e4)

    # pooled signed forward move over all fresh transitions, with a block bootstrap CI.
    # BLOCK, not iid: overlapping h-bar forward windows on adjacent transitions are strongly
    # autocorrelated, so an iid bootstrap would understate the interval badly.
    sel = fresh & np.isin(y, (0, 1)) & np.isfinite(fwd)
    idx = np.flatnonzero(sel)
    signed = np.where(y[idx] == 0, 1.0, -1.0) * fwd[idx]
    out["edge_bp"] = float(np.mean(signed) * 1e4) if len(signed) else float("nan")
    if len(signed) >= 200:
        rng = np.random.default_rng(20260902)
        block = max(1, int(np.ceil(h / 1.0)))          # one block ~ one forward-window length
        nblocks = int(np.ceil(len(signed) / block))
        starts = np.arange(0, len(signed), block)
        boot = np.empty(400)
        for b in range(400):
            pick = rng.choice(starts, size=nblocks, replace=True)
            samp = np.concatenate([signed[p:p + block] for p in pick])
            boot[b] = samp.mean()
        out["edge_ci_lo_bp"] = float(np.percentile(boot, 2.5) * 1e4)
        out["edge_ci_hi_bp"] = float(np.percentile(boot, 97.5) * 1e4)
    else:
        out["edge_ci_lo_bp"] = out["edge_ci_hi_bp"] = float("nan")
    return out


def main() -> None:
    df = load_train()
    close = df["close"]
    close_np = close.to_numpy()
    print(f"TRAIN {len(df):,} bars  {df['timestamp'].min().date()} ~ {df['timestamp'].max().date()}"
          "  (OOS deliberately NOT touched in this phase)")

    # --- reference: the deployed label, same window ---
    ref_df = df.copy()
    ref_df["mtf_trend_1h"] = close.ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(ref_df)
    y_ref = np.full(len(df), 2, dtype=int)
    y_ref[lab["regime_bull"].to_numpy() > 0] = 0
    y_ref[lab["regime_bear"].to_numpy() > 0] = 1
    runs_ref = _run_lengths(y_ref)
    print(f"\nREFERENCE (deployed RegimeEngine, S=24-equivalent): "
          f"shares bull={np.mean(y_ref==0):.3f} bear={np.mean(y_ref==1):.3f} chop={np.mean(y_ref==2):.3f} "
          f"| flip={np.mean(y_ref[1:]!=y_ref[:-1]):.4f} | run median={np.median(runs_ref):.0f} "
          f"mean={np.mean(runs_ref):.1f} bars")
    for h in FWD_HORIZONS:
        te = transition_edge(y_ref, close_np, h)
        print(f"    h={h:2d} transition edge {te['edge_bp']:+.2f}bp "
              f"[95% {te['edge_ci_lo_bp']:+.2f},{te['edge_ci_hi_bp']:+.2f}] "
              f"(bull {te['bull_fwd_bp']:+.2f} n={te['bull_n']}, bear {te['bear_fwd_bp']:+.2f} "
              f"n={te['bear_n']}) | baseline |move| {te['baseline_abs_bp']:.1f}bp")

    # --- per-scale threshold calibration to the CURRENT label's firing rates ---
    rate1 = float((efficiency_ratio(close, 24) >= 0.20).mean())
    rate2 = float((efficiency_ratio(close, 48) >= 0.16).mean())
    print(f"\nCalibration targets from the deployed label on TRAIN: "
          f"P(er_24>=0.20)={rate1:.4f}, P(er_48>=0.16)={rate2:.4f}")

    rows = []
    for s in SCALES:
        er_s, er_2s = efficiency_ratio(close, s), efficiency_ratio(close, 2 * s)
        t1 = float(er_s.quantile(1.0 - rate1))
        t2 = float(er_2s.quantile(1.0 - rate2))
        y_raw = scaled_label(close, s, t1, t2)
        for k in DEBOUNCES:
            y = y_raw if k == 1 else _debounce(y_raw, k)
            runs = _run_lengths(y)
            row = {"scale_bars": s, "scale_min": s * 5, "debounce_k": k,
                   "T1": round(t1, 4), "T2": round(t2, 4),
                   "bull": round(float(np.mean(y == 0)), 3), "bear": round(float(np.mean(y == 1)), 3),
                   "chop": round(float(np.mean(y == 2)), 3),
                   "flip_rate": round(float(np.mean(y[1:] != y[:-1])), 4),
                   "run_median": float(np.median(runs)), "run_mean": round(float(np.mean(runs)), 1)}
            for h in FWD_HORIZONS:
                te = transition_edge(y, close_np, h)
                row[f"edge_h{h}_bp"] = round(te["edge_bp"], 2)
                row[f"ci_h{h}"] = f"[{te['edge_ci_lo_bp']:+.2f},{te['edge_ci_hi_bp']:+.2f}]"
                row[f"sig_h{h}"] = bool(te["edge_ci_lo_bp"] > 0)
                row[f"n_trans_h{h}"] = te["bull_n"] + te["bear_n"]
            rows.append(row)

    out = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_DIR / "label_geometry.csv", index=False)
    pd.set_option("display.width", 240)
    pd.set_option("display.max_rows", 100)
    print("\n=== label geometry sweep (TRAIN only) ===")
    print(out[["scale_bars", "scale_min", "debounce_k", "chop", "flip_rate", "run_median",
               "edge_h6_bp", "ci_h6", "sig_h6", "edge_h12_bp", "ci_h12", "sig_h12",
               "n_trans_h12"]].to_string(index=False))
    sig = out[out["sig_h6"] | out["sig_h12"]]
    print(f"\ncells with a 95% CI strictly above 0 at h=6 or h=12: {len(sig)} / {len(out)}")
    print(f"\nWrote {OUT_DIR / 'label_geometry.csv'}")


if __name__ == "__main__":
    main()
