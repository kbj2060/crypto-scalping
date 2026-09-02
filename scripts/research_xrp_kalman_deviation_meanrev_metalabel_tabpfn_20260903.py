#!/usr/bin/env python3
"""BTC port of `kalman_deviation_meanrev` -- one of ETH's live dashboard evidence signals
(`compute_signals()`'s SIGNAL_ORDER in live_evidence_signal_dashboard_20260823.py, added
2026-08-31 from the Homer candidate pool, ETH result VAL/OOS/HOLDOUT AUC 0.6569/0.6311/0.6284).
Combines a quick HORIZON x K grid confirm/refine (CPU, no GPU needed) with the full TabPFN
metalabel pipeline (GPU) in one script, since ETH's own reference point for this exact signal is
already well-established (docs/homer/README.md "후보 풀" section, 2026-08-31).

Trigger formula (verbatim from live_evidence_signal_dashboard_20260823.py lines ~531-561,
confirmed by direct read before writing this): constant-velocity 2-state Kalman filter on close
(F=[[1,1],[0,1]], H=[[1,0]], Q=I*1e-5, R=[[1e-3]]) -> kalman_dev = (close-level)/level ->
kalman_dev_z = rolling zscore (window=288, same ZSCORE_WINDOW convention used elsewhere in this
project) -> bottom fires when kalman_dev_z<=-2.0, top when >=2.0. Pure price (close only), no
cross-asset/funding dependency -- computed fresh here from the BTC Tier0 CSV's own `close` column
(itself sourced from binance_data/klines/BTCUSDT/BTCUSDT-5m-api.csv, 2024-01-01 onward) rather than
imported from any other script, since the Tier0 build
(scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py) explicitly EXCLUDED this
signal (still under independent ETH validation at the time that script was written). A local CPU
dry run (quant_ai env, no GPU) confirmed the recursive loop runs in ~4s for all 277,191 BTC bars --
no numba/vectorization needed. Warmup note: the Tier0 CSV's `close` column already starts at
2024-01-01 (vs BTC's full klines history starting 2023-12-31) -- under 1 day / ~288 bars of
difference, which is irrelevant here since kalman_dev_z itself is undefined (NaN) until bar 288
anyway (rolling zscore min_periods=288), by which point any cold-start transient from this fast
2-state filter (fixed small Q/R) is long gone regardless of the exact start date.

Methodology, ported/adapted from this project's established lineage (all read in full before
writing this script):
  - build_indicator_frame's FEATURE_COLUMNS (23 features) verbatim from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py, applied via BTC's own Tier0 CSV
    (which already carries the equivalent columns) plus the 4 genuinely-missing ones
    (atr_pct/nyse_open_flag/er_24/realized_vol_ratio) -- exact same shortcut this signal's BTC
    sibling used (research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py::
    add_missing_features, ported verbatim here).
  - kalman_dev_z itself is added as a 24th feature -- this is NOT a free choice, it is ETH's own
    established convention for this exact signal (research_eth_kalman_demarker_gridscreen_
    20260831.py / _tabpfn_confirm_20260831.py / _metalabel_holdout_20260831.py all use
    FEATURE_COLUMNS + ["kalman_dev_z"], the same pattern demarker_extreme uses with its own `dem`
    value, and taker_delta_z_climax/short_term_return_z use with their own delta_z/ret3_z).
  - Hit definition: touch-based MFE using intrabar high/low over bars[fire+1:fire+HORIZON+1],
    hit = move_pct >= K * atr_pct. Confirmed (by reading research_eth_kalman_demarker_gridscreen_
    20260831.py::build_fires directly) that ETH's own kalman pipeline compares against atr_pct
    (dimensionless), NOT raw price-scale atr -- `peak = move_pct / atr_pct; hit = peak >= K`. This
    disambiguates the task's shorthand "K*atr" formula, and deliberately does NOT reuse the BTC
    taker_delta_z_climax sibling's own close_at_h/raw-atr convention (that was a signal-specific
    grid-screen finding for THAT signal, not this project's general convention).
  - cluster_dedup: verbatim pattern from research_eth_kalman_demarker_gridscreen_20260831.py /
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py's v4 dedup (collapse same-side
    fires within GAP bars into one cluster, keep the single most-extreme-kalman_dev_z bar per
    cluster -- causal, uses only kalman_dev_z itself, never future price). GAP is FIXED at 6 bars
    per explicit task instruction (NOT re-screened here) -- this differs from ETH's own grid-
    selected GAP=12 for this signal; the comparison against ETH's headline numbers below is
    therefore directional/methodological, not apples-to-apples on GAP.
  - run_tabpfn_panel / evaluate / compute_permutation_importance: ported VERBATIM (identical code)
    from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py.

Grid screen (HORIZON x K, CPU-only, no GPU): "quick grid confirm/refine", not a full re-run of
ETH's own 3-stage GBM+K-sweep+TabPFN screen -- justified because ETH's own reference point for
this exact signal is already well-established. HORIZON_GRID=[8,10,12,16,20,24], K_GRID=
[1.5,2.0,2.5,3.0,3.5,4.0] (both centered on ETH's own H=12/K=2.5, "wide enough to not hit a
boundary" per task instruction -- verified empirically: raw trigger counts are ~5,000/side in
TRAIN and stay >=700/side even in the smallest split (OOS), so no cell is sample-starved).

⚠️ DEVIATION FROM THE ORIGINAL "lift vs random-bar baseline" PLAN, discovered and fixed during
development (documented here rather than silently changed, per this project's own standing
practice of investigating anomalies instead of shipping them -- see e.g. the K-recalibration
story in docs/homer/README.md's "HORIZON x GAP x K 그리드 확정" section):
A first implementation (mirroring research_btc_taker_delta_climax_gridscreen_20260901.py's
random_baseline_hit -- draw N random non-triggered bars, apply the same mirrored-direction
K*atr_pct touch check, hit threshold self-normalized by EACH bar's own contemporaneous atr_pct)
found lift < 1.0x in EVERY one of the 36 (H,K) cells (range ~0.73x-0.95x), with the argmax sitting
at K=1.5 -- a literal grid boundary, exactly the failure mode this project's own methodology
explicitly warns against trusting. Diagnosis (scratchpad/diagnose_lift_sign.py, reran against this
module's real functions before accepting the conclusion): kalman_dev_z extremes co-occur with a
LOCALLY ELEVATED atr_pct at the fire bar itself (TRAIN fire-bar atr_pct mean ~0.0030 vs ~0.0019 for
the non-fired pool, ~1.6-1.7x) -- both quantities react to the same recent-volatility burst (a
14-bar rolling ATR and a fast-reacting deviation-from-a-slow-trend measure). Self-normalizing the
hit threshold by each bar's OWN atr_pct therefore imposes a mechanically HARDER hurdle specifically
on fired bars, masking a genuinely LARGER raw forward move: fired bars' mean forward MFE is ~53-56%
LARGER in absolute terms than the pool's (bottom 0.516% vs 0.337%; top 0.544% vs 0.348%, H=12). A
cross-check using a FIXED threshold (K times the TRAIN-pool's median atr_pct, applied uniformly
instead of each bar's own) flips the sign decisively: lift 1.61x (bottom) / 1.59x (top) at H=12
K=2.5 -- squarely in the ballpark of ETH's own reported raw lift for this signal (2.16x/2.36x,
docs/homer/README.md). A GBM sanity fit (HistGradientBoostingClassifier, TRAIN-fit, VAL AUC) using
the SAME self-normalized hit label used throughout this script got 0.6400 at ETH's own H=12/K=2.5
-- genuine, comparable to ETH's own VAL AUC (0.6569). Conclusion: the self-normalized hit LABEL
itself is a sound classification target (confirmed downstream by both the GBM check here and the
eventual TabPFN numbers below); the raw-lift-vs-random-baseline check specifically, as a
SELECTION metric, is confounded by this atr_pct co-movement effect and does not discriminate
genuine signal from noise across the grid for THIS signal.
Fix: the grid screen below uses GBM (HistGradientBoostingClassifier) VAL AUC, TRAIN-fit, as its
selection metric instead -- this is not an ad hoc substitute, it is ETH's OWN actual established
methodology for THIS exact signal (research_eth_kalman_demarker_gridscreen_20260831.py::
screen_signal), which this run had initially deviated from in favor of a different sibling
script's pattern. GBM AUC sidesteps the confound because atr_pct is itself one of the 24 input
features, not a hand-picked normalization constant imposed on top of the model. The self-
normalized raw-lift check is KEPT in the grid table below as a diagnostic column (not for
selection), and a fixed-threshold cross-check at ETH's own H=12/K=2.5 center is also computed and
reported, for direct comparability with ETH's reported raw-lift figures. Selection: among cells
with enough pooled TRAIN fires (>=400), argmax of GBM VAL AUC (TRAIN-fit) -- OOS/HOLDOUT are still
never touched at this stage, matching the task's original constraint.

⚠️ Per docs/homer/README.md's 2026-08-31 finding for THIS signal on ETH: a mean-reversion
autocorrelation regime gate was tested and found momentum-regime fires predict BETTER than
mean-reversion-regime fires (opposite of the natural hypothesis for a mean-reversion signal) --
that gate is deliberately NOT applied here.

Splits: this repo's Fresh-Forward default (CLAUDE.md) -- TRAIN < 2025-09-01, VAL 2025-09-01..
2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT 2026-04-01..latest (single-touch, evaluated once).

Runs entirely on the GPU server (quant_ai env, CUDA required for TabPFN) under a system-wide flock
(shared 8GB GPU across concurrently running signal-research agents this session) -- see
scripts/ops/handoff.sh push before executing remotely. The grid-screen portion is CPU-only and
fast, but is combined into this one script (not split into a separate local pre-pass) per task
instruction, since ETH's own reference point makes a full separate local dry-run pass unnecessary
beyond the sanity check already done in scratchpad during development.

## ⚠️XRP 포팅 (2026-09-03)

`research_btc_<signal>_metalabel_tabpfn_20260901.py`의 **자산 상수만** 바꾼 포팅.
격자(HORIZON x K)와 TabPFN 절차는 그대로 재탐색한다 -- 자산이 바뀌면 최적 셀도 바뀐다.
이 두 신호는 ETH·BTC 모두 **plain touch**(`peak >= K`) HIT 정의를 쓰므로 HIT_TYPE은 안 쓴다.
절차: `docs/homer/evidence_signal_new_coin_port_protocol.md`
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

TIER0_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
REPORT_PATH = OUT_DIR / "kalman_deviation_meanrev_tabpfn_report.json"
FEATURES_CSV_PATH = OUT_DIR / "btc_5m_kalman_deviation_meanrev_metalabel_features.csv"

START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

ZSCORE_WINDOW = 288  # matches live_evidence_signal_dashboard_20260823.py's ZSCORE_WINDOW

HORIZON_GRID = [8, 10, 12, 16, 20, 24]
K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
RNG_SEED = 20260901

CLUSTER_GAP = 6  # fixed per task instruction (ETH's own grid-selected value for this signal was
                 # GAP=12 -- not re-screened here, see module docstring)

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds used throughout this project's lineage

ETH_OWN_RESULT = {
    "note": ("same signal (kalman_deviation_meanrev), ETH, H=12 GAP=12 K=2.5 -- GAP differs from "
             "this BTC run's fixed GAP=6, so the comparison below is directional/methodological, "
             "not apples-to-apples on GAP."),
    "horizon": 12, "gap": 12, "k": 2.5,
    "val_auc": 0.6569, "oos_auc": 0.6311, "holdout_auc": 0.6284,
}

# ETH's own canonical 23-feature Tier0 set, verbatim from research_eth_taker_delta_climax_
# metalabel_tabpfn_20260829.py::FEATURE_COLUMNS, plus this signal's own trigger value
# kalman_dev_z as a 24th feature (ETH's own established convention for this signal, see docstring).
FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
    "kalman_dev_z",
]


def log(msg: str) -> None:
    print(f"[btc_kalman_deviation_meanrev_tabpfn] {msg}", flush=True)


def load_tier0() -> pd.DataFrame:
    usecols = ["timestamp", "high", "low", "close", "atr", "atr_percentile_864",
               "hour_utc", "weekday", "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
               "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
               "adx14", "pdi", "ndi", "bb_width_pctile", "rsi"]
    df = pd.read_csv(TIER0_PATH, usecols=usecols, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # tz-aware UTC -> naive UTC, matches
                                                              # this BTC lineage's own convention
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in BTC Tier0 rows"
    return df


def compute_kalman_dev_z(close: np.ndarray) -> np.ndarray:
    """Verbatim port of live_evidence_signal_dashboard_20260823.py's kalman_deviation_meanrev
    block (read directly before writing this, lines ~531-561) -- constant-velocity 2-state Kalman
    filter on close, genuine per-bar recursive loop. Confirmed by local CPU dry run: ~4s for
    277,191 BTC bars, no numba/JIT needed."""
    n = len(close)
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.eye(2) * 1e-5
    R = np.array([[1e-3]])
    x = np.array([close[0], 0.0])
    P = np.eye(2)
    levels = np.empty(n)
    for i in range(n):
        x = F @ x
        P = F @ P @ F.T + Q
        S = (H @ P @ H.T + R)[0, 0]
        K = (P @ H.T).flatten() / S
        innovation = close[i] - (H @ x)[0]
        x = x + K * innovation
        P = (np.eye(2) - np.outer(K, H)) @ P
        levels[i] = x[0]
    kalman_dev = pd.Series((close - levels) / levels)
    mean = kalman_dev.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).mean()
    std = kalman_dev.rolling(ZSCORE_WINDOW, min_periods=ZSCORE_WINDOW).std().replace(0.0, np.nan)
    return ((kalman_dev - mean) / std).to_numpy()


def add_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port from research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py::
    add_missing_features (itself ported from research_eth_taker_delta_climax_metalabel_tabpfn_
    20260829.py::build_indicator_frame) -- adds the handful of FEATURE_COLUMNS not already
    present in the BTC Tier0 CSV."""
    close = frame["close"]
    frame["atr_pct"] = frame["atr"] / close.clip(lower=1e-12)

    tmin = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    is_weekday = frame["timestamp"].dt.dayofweek < 5
    frame["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    frame["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)

    log_ret = np.log(close / close.shift(1))
    frame["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()

    return frame


def forward_extremes(high: np.ndarray, low: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    fwd_high_max = pd.Series(high).rolling(horizon, min_periods=horizon).max().shift(-horizon).to_numpy()
    fwd_low_min = pd.Series(low).rolling(horizon, min_periods=horizon).min().shift(-horizon).to_numpy()
    return fwd_high_max, fwd_low_min


def random_baseline_hit(rng: np.random.Generator, pool_idx: np.ndarray, n_draw: int,
                         fwd_ext: np.ndarray, close: np.ndarray, atr_pct: np.ndarray,
                         k: float, direction: str) -> float:
    if n_draw <= 0 or len(pool_idx) < n_draw:
        return float("nan")
    samp = rng.choice(pool_idx, size=n_draw, replace=False)
    if direction == "up":
        hit = (fwd_ext[samp] - close[samp]) / close[samp] >= k * atr_pct[samp]
    else:
        hit = (close[samp] - fwd_ext[samp]) / close[samp] >= k * atr_pct[samp]
    return float(hit.mean())


def raw_lift_diagnostic(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray,
                         horizon: int, k: float, rng: np.random.Generator) -> dict:
    """Self-normalized (each bar's own atr_pct) lift-vs-random-baseline, TRAIN+VAL only -- the
    ORIGINAL plan for grid-screen selection, kept as a DIAGNOSTIC column only (see module
    docstring: this metric is confounded by fire bars' locally-elevated atr_pct and does not
    discriminate well across the grid for this signal). No clustering (raw triggers)."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr_pct = frame["atr_pct"].to_numpy()
    ts_col = frame["timestamp"]
    any_trig = bottom_trig | top_trig
    train_mask = (ts_col < VAL_START).to_numpy()
    val_mask = ((ts_col >= VAL_START) & (ts_col < OOS_START)).to_numpy()
    finite_ok = np.isfinite(close) & np.isfinite(atr_pct)

    fwd_high_max, fwd_low_min = forward_extremes(high, low, horizon)
    valid_fwd = np.isfinite(fwd_high_max) & np.isfinite(fwd_low_min) & finite_ok
    bottom_idx_all = np.flatnonzero(bottom_trig & valid_fwd)
    top_idx_all = np.flatnonzero(top_trig & valid_fwd)
    pool_idx_train = np.flatnonzero(train_mask & (~any_trig) & valid_fwd)
    pool_idx_val = np.flatnonzero(val_mask & (~any_trig) & valid_fwd)

    bottom_hit_all = (fwd_high_max[bottom_idx_all] - close[bottom_idx_all]) / close[bottom_idx_all] >= k * atr_pct[bottom_idx_all]
    top_hit_all = (close[top_idx_all] - fwd_low_min[top_idx_all]) / close[top_idx_all] >= k * atr_pct[top_idx_all]

    b_train_m, b_val_m = train_mask[bottom_idx_all], val_mask[bottom_idx_all]
    t_train_m, t_val_m = train_mask[top_idx_all], val_mask[top_idx_all]
    n_b_train, n_t_train = int(b_train_m.sum()), int(t_train_m.sum())
    n_b_val, n_t_val = int(b_val_m.sum()), int(t_val_m.sum())

    pooled_train_n = n_b_train + n_t_train
    pooled_val_n = n_b_val + n_t_val
    pooled_train_hit = float((bottom_hit_all[b_train_m].sum() + top_hit_all[t_train_m].sum()) / max(pooled_train_n, 1))
    pooled_val_hit = float((bottom_hit_all[b_val_m].sum() + top_hit_all[t_val_m].sum()) / max(pooled_val_n, 1))

    b_base = random_baseline_hit(rng, pool_idx_train, n_b_train, fwd_high_max, close, atr_pct, k, "up")
    t_base = random_baseline_hit(rng, pool_idx_train, n_t_train, fwd_low_min, close, atr_pct, k, "down")
    pooled_base = (b_base * n_b_train + t_base * n_t_train) / pooled_train_n if pooled_train_n and np.isfinite(b_base) and np.isfinite(t_base) else float("nan")
    lift_train = pooled_train_hit / pooled_base if np.isfinite(pooled_base) and pooled_base > 0 else float("nan")

    b_base_val = random_baseline_hit(rng, pool_idx_val, n_b_val, fwd_high_max, close, atr_pct, k, "up")
    t_base_val = random_baseline_hit(rng, pool_idx_val, n_t_val, fwd_low_min, close, atr_pct, k, "down")
    pooled_base_val = (b_base_val * n_b_val + t_base_val * n_t_val) / pooled_val_n if pooled_val_n and np.isfinite(b_base_val) and np.isfinite(t_base_val) else float("nan")
    lift_val = pooled_val_hit / pooled_base_val if np.isfinite(pooled_base_val) and pooled_base_val > 0 else float("nan")

    return {
        "train_hitrate_pooled": round(pooled_train_hit, 4),
        "train_baseline_pooled": round(pooled_base, 4) if np.isfinite(pooled_base) else None,
        "lift_train_pooled_self_normalized": round(lift_train, 4) if np.isfinite(lift_train) else None,
        "val_hitrate_pooled": round(pooled_val_hit, 4),
        "val_baseline_pooled": round(pooled_base_val, 4) if np.isfinite(pooled_base_val) else None,
        "lift_val_pooled_self_normalized": round(lift_val, 4) if np.isfinite(lift_val) else None,
    }


def eth_center_fixed_threshold_crosscheck(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray,
                                           horizon: int = 12, k: float = 2.5) -> dict:
    """Cross-check at ETH's own H=12/K=2.5 center using a FIXED threshold (K times the TRAIN-pool's
    median atr_pct, applied uniformly to fired AND pool populations) instead of each bar's own
    atr_pct -- isolates the atr_pct co-movement confound described in the module docstring and
    gives a lift number directly comparable to ETH's own reported raw lift (2.16x/2.36x, top/
    bottom, docs/homer/README.md)."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr_pct = frame["atr_pct"].to_numpy()
    ts_col = frame["timestamp"]
    any_trig = bottom_trig | top_trig
    train_mask = (ts_col < VAL_START).to_numpy()
    finite_ok = np.isfinite(close) & np.isfinite(atr_pct)
    fwd_high_max, fwd_low_min = forward_extremes(high, low, horizon)
    valid_fwd = np.isfinite(fwd_high_max) & np.isfinite(fwd_low_min) & finite_ok

    med_atr_pct = float(np.nanmedian(atr_pct[train_mask & valid_fwd]))
    bottom_idx = np.flatnonzero(bottom_trig & train_mask & valid_fwd)
    top_idx = np.flatnonzero(top_trig & train_mask & valid_fwd)
    pool_idx = np.flatnonzero(train_mask & (~any_trig) & valid_fwd)

    b_move = (fwd_high_max[bottom_idx] - close[bottom_idx]) / close[bottom_idx]
    t_move = (close[top_idx] - fwd_low_min[top_idx]) / close[top_idx]
    pb_move = (fwd_high_max[pool_idx] - close[pool_idx]) / close[pool_idx]
    pt_move = (close[pool_idx] - fwd_low_min[pool_idx]) / close[pool_idx]

    b_hit = float((b_move >= k * med_atr_pct).mean())
    t_hit = float((t_move >= k * med_atr_pct).mean())
    pb_hit = float((pb_move >= k * med_atr_pct).mean())
    pt_hit = float((pt_move >= k * med_atr_pct).mean())

    return {
        "horizon": horizon, "k": k, "train_pool_median_atr_pct": round(med_atr_pct, 6),
        "fire_bar_mean_atr_pct": {"bottom": round(float(atr_pct[bottom_idx].mean()), 6),
                                   "top": round(float(atr_pct[top_idx].mean()), 6)},
        "pool_mean_atr_pct": round(float(atr_pct[pool_idx].mean()), 6),
        "bottom_hit_rate": round(b_hit, 4), "bottom_pool_hit_rate": round(pb_hit, 4),
        "lift_bottom_fixed_threshold": round(b_hit / pb_hit, 4) if pb_hit > 0 else None,
        "top_hit_rate": round(t_hit, 4), "top_pool_hit_rate": round(pt_hit, 4),
        "lift_top_fixed_threshold": round(t_hit / pt_hit, 4) if pt_hit > 0 else None,
        "mean_forward_move_pct": {
            "bottom_fired": round(float(b_move.mean()), 6), "bottom_pool": round(float(pb_move.mean()), 6),
            "top_fired": round(float(t_move.mean()), 6), "top_pool": round(float(pt_move.mean()), 6),
        },
    }


GBM_SEED = 20260901
MIN_TRAIN_TOTAL = 400  # pooled bottom+top, post-dedup; actual counts are ~5,000+, generous headroom
# Only used if the primary HORIZON_GRID's winner sits at HORIZON_GRID[0]=8 (a boundary) -- a
# one-shot supplementary check, not an open-ended search (see finalize_horizon_choice).
BOUNDARY_EXTENSION_HORIZON_GRID = [4, 5, 6, 7]
BOUNDARY_EXTENSION_MIN_AUC_GAIN = 0.02  # extension must beat the original non-boundary best by
                                        # more than this (well above single-seed GBM noise, see
                                        # finalize_horizon_choice) to be trusted over it


def _screen_one_cell(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray,
                      horizon: int, k: float, rng: np.random.Generator) -> dict:
    """One (H,K) cell: self-normalized raw-lift diagnostic + GBM (HistGradientBoostingClassifier)
    TRAIN-fit/VAL-eval AUC on the final clustered (GAP=6) 24-feature set. Shared by run_grid_screen
    and the horizon-boundary extension check (finalize_horizon_choice) so both use identical
    per-cell logic."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    diag = raw_lift_diagnostic(frame, bottom_trig, top_trig, horizon, k, rng)

    fires, _ = build_fires_and_features(frame, bottom_trig, top_trig, horizon, k, CLUSTER_GAP)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    ts = fires["timestamp"]
    train_mask = (ts < VAL_START).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    n_train, n_val = int(train_mask.sum()), int(val_mask.sum())
    train_y = fires.loc[train_mask, "hit"].to_numpy().astype(int)
    val_y = fires.loc[val_mask, "hit"].to_numpy().astype(int)

    row = {"horizon": horizon, "k": k, "n_train": n_train, "n_val": n_val, **diag}
    if n_train < MIN_TRAIN_TOTAL or len(np.unique(train_y)) < 2 or len(np.unique(val_y)) < 2:
        row["gbm_train_auc"] = None
        row["gbm_val_auc"] = None
        log(f"H={horizon:>3d} K={k:.1f}  n_train={n_train} n_val={n_val}  GBM: skipped (degenerate/too few) "
            f"|  raw-lift-diag: train={diag['lift_train_pooled_self_normalized']} val={diag['lift_val_pooled_self_normalized']}")
        return row

    clf = HistGradientBoostingClassifier(random_state=GBM_SEED)
    clf.fit(fires.loc[train_mask, FEATURE_COLUMNS], train_y)
    train_auc = float(roc_auc_score(train_y, clf.predict_proba(fires.loc[train_mask, FEATURE_COLUMNS])[:, 1]))
    val_auc = float(roc_auc_score(val_y, clf.predict_proba(fires.loc[val_mask, FEATURE_COLUMNS])[:, 1]))
    row["gbm_train_auc"] = round(train_auc, 4)
    row["gbm_val_auc"] = round(val_auc, 4)
    row["train_hit_rate_deduped"] = round(float(train_y.mean()), 4)
    row["val_hit_rate_deduped"] = round(float(val_y.mean()), 4)
    log(f"H={horizon:>3d} K={k:.1f}  n_train={n_train} n_val={n_val}  "
        f"GBM: train_auc={train_auc:.4f} val_auc={val_auc:.4f}  "
        f"|  raw-lift-diag (self-normalized, NOT used for selection): "
        f"train={diag['lift_train_pooled_self_normalized']} val={diag['lift_val_pooled_self_normalized']}")
    return row


def _row_to_chosen(best_row: pd.Series) -> dict:
    return {
        "horizon": int(best_row["horizon"]), "k": float(best_row["k"]),
        "gbm_train_auc": float(best_row["gbm_train_auc"]), "gbm_val_auc": float(best_row["gbm_val_auc"]),
        "n_train": int(best_row["n_train"]), "n_val": int(best_row["n_val"]),
        "raw_lift_diagnostic_self_normalized": {
            "train": best_row["lift_train_pooled_self_normalized"], "val": best_row["lift_val_pooled_self_normalized"],
        },
        "at_horizon_boundary": bool(best_row["horizon"] in (HORIZON_GRID[0], HORIZON_GRID[-1])),
        "at_k_boundary": bool(best_row["k"] in (K_GRID[0], K_GRID[-1])),
    }


def run_grid_screen(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray) -> tuple[list[dict], dict]:
    """H x K screen, TRAIN-fit + VAL evaluation only -- no OOS/HOLDOUT access at this stage.
    Selection metric: GBM (HistGradientBoostingClassifier) VAL AUC on the final clustered
    (GAP=6) fires+24-feature set -- ETH's OWN actual methodology for this exact signal
    (research_eth_kalman_demarker_gridscreen_20260831.py::screen_signal), NOT a raw lift-vs-
    random-baseline check (see module docstring for why that was tried first and replaced).
    The self-normalized raw-lift diagnostic is still computed and attached to each row, for
    transparency/comparison. Returns the RAW mechanical argmax as `chosen` -- boundary handling
    happens one level up, in finalize_horizon_choice()."""
    ts_col = frame["timestamp"]
    train_mask_bar = (ts_col < VAL_START).to_numpy()
    val_mask_bar = ((ts_col >= VAL_START) & (ts_col < OOS_START)).to_numpy()
    log(f"raw trigger counts: bottom={int(bottom_trig.sum())} top={int(top_trig.sum())} "
        f"(TRAIN bottom={int((bottom_trig & train_mask_bar).sum())} top={int((top_trig & train_mask_bar).sum())}, "
        f"VAL bottom={int((bottom_trig & val_mask_bar).sum())} top={int((top_trig & val_mask_bar).sum())})")

    rng = np.random.default_rng(RNG_SEED)
    grid_rows = [_screen_one_cell(frame, bottom_trig, top_trig, horizon, k, rng)
                 for horizon in HORIZON_GRID for k in K_GRID]

    grid_df = pd.DataFrame(grid_rows)
    eligible = grid_df.dropna(subset=["gbm_val_auc"]).copy()
    assert len(eligible) > 0, "no grid cell survived the floor+degenerate-split filters"
    chosen = _row_to_chosen(eligible.loc[eligible["gbm_val_auc"].idxmax()])
    log(f"RAW ARGMAX: HORIZON={chosen['horizon']} K={chosen['k']} (GBM val_auc={chosen['gbm_val_auc']:.4f}, "
        f"boundary: H={chosen['at_horizon_boundary']} K={chosen['at_k_boundary']})")
    return grid_rows, chosen


def finalize_horizon_choice(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray,
                             grid_rows: list[dict], raw_chosen: dict) -> tuple[dict, dict | None]:
    """If the primary grid's raw argmax sits at the HORIZON_GRID boundary (H=8), this project's
    own established caution ("don't trust a grid boundary -- extend the range", docs/homer/
    README.md) applies. Runs ONE supplementary extension check (H in BOUNDARY_EXTENSION_
    HORIZON_GRID=[4,5,6,7], same K_GRID) rather than either blindly trusting the boundary or
    blindly ignoring it.

    Decision rule (documented, not silent): adopt the extension's own best cell ONLY if it (a) is
    NOT itself sitting at the new boundary (H=4) -- otherwise we'd be chasing an open-ended
    regress toward H=0 -- AND (b) beats the best NON-boundary cell in the ORIGINAL grid by more
    than BOUNDARY_EXTENSION_MIN_AUC_GAIN (0.02 AUC), a margin well above the noise this project
    observed in practice (K=3.0->3.5->4.0 at fixed H swinging by 0.05-0.09 AUC from a single GBM
    seed/fit with n_val~1,095 -- see development note below). Otherwise, fall back to the best
    NON-boundary cell already in the original grid.

    Development note (why this function exists at all): during development, the raw argmax landed
    at H=8/K=3.5 (VAL AUC 0.7017), with H=10/K=3.5 essentially tied (0.7013) -- a difference far
    smaller than the noise visible elsewhere in the same grid. Running this exact extension check
    found H=4/K=4.5 nominally highest (0.7396) but ITSELF sitting at the new boundary, with a
    jagged, non-monotonic K-pattern at neighboring cells (e.g. H=8: K=4.0 -> 0.6519 but
    K=4.5 -> 0.7396) inconsistent with a genuine smooth signal -- i.e., extending the range did
    NOT resolve the boundary concern, it reproduced it one step further out. This matches a single-
    seed unregularized-GBM selection metric being too noisy at this sample size to trust a literal
    argmax; the fix here is to require a clear, one-shot-verified margin before trusting a boundary
    cell, and otherwise prefer the nearest robust interior alternative -- not to keep extending
    indefinitely chasing noise."""
    if not raw_chosen["at_horizon_boundary"]:
        return raw_chosen, None

    log(f"raw argmax HORIZON={raw_chosen['horizon']} sits at the grid boundary -- running the "
        f"one-shot extension check (H in {BOUNDARY_EXTENSION_HORIZON_GRID}) before trusting it...")
    rng = np.random.default_rng(RNG_SEED + 1)
    ext_rows = [_screen_one_cell(frame, bottom_trig, top_trig, horizon, k, rng)
                for horizon in BOUNDARY_EXTENSION_HORIZON_GRID for k in K_GRID]
    ext_df = pd.DataFrame(ext_rows).dropna(subset=["gbm_val_auc"])
    ext_best = _row_to_chosen(ext_df.loc[ext_df["gbm_val_auc"].idxmax()]) if len(ext_df) else None

    grid_df = pd.DataFrame(grid_rows).dropna(subset=["gbm_val_auc"])
    non_boundary = grid_df[~grid_df["horizon"].isin([HORIZON_GRID[0], HORIZON_GRID[-1]])]
    fallback = _row_to_chosen(non_boundary.loc[non_boundary["gbm_val_auc"].idxmax()])

    outcome = {
        "extension_grid": ext_rows,
        "extension_best": ext_best,
        "original_best_non_boundary": fallback,
        "extension_best_itself_at_new_boundary": bool(ext_best and ext_best["horizon"] == BOUNDARY_EXTENSION_HORIZON_GRID[0]),
    }
    if ext_best is not None and not outcome["extension_best_itself_at_new_boundary"] and \
            ext_best["gbm_val_auc"] > fallback["gbm_val_auc"] + BOUNDARY_EXTENSION_MIN_AUC_GAIN:
        outcome["decision"] = "adopted extension's own best cell (clear, non-boundary improvement)"
        final = ext_best
    else:
        outcome["decision"] = (
            "fell back to the best NON-boundary cell in the original grid -- extension's own best "
            "either sat at its own new boundary, or did not clearly beat the fallback"
        )
        final = fallback
    log(f"boundary check outcome: {outcome['decision']}. FINAL: HORIZON={final['horizon']} K={final['k']} "
        f"(GBM val_auc={final['gbm_val_auc']:.4f})")
    return final, outcome


def cluster_dedup(idx: np.ndarray, extremeness: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    """Verbatim pattern from research_eth_kalman_demarker_gridscreen_20260831.py::cluster_dedup
    (matching research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py's v4 dedup),
    parameterized by gap (fixed to CLUSTER_GAP=6 for this run, see module docstring)."""
    order = np.argsort(idx)
    idx_sorted, ex_sorted = idx[order], extremeness[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "ex": ex_sorted})
    keep = df.loc[df.groupby("cluster")["ex"].idxmin()] if most_negative else df.loc[df.groupby("cluster")["ex"].idxmax()]
    return np.sort(keep["idx"].to_numpy())


def build_fires_and_features(frame: pd.DataFrame, bottom_trig: np.ndarray, top_trig: np.ndarray,
                              horizon: int, k: float, gap: int) -> tuple[pd.DataFrame, dict]:
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    atr_pct = frame["atr_pct"].to_numpy()
    kalman_dev_z_all = frame["kalman_dev_z"].to_numpy()
    ts_all = frame["timestamp"].to_numpy()
    n = len(frame)
    rows = []
    dedup_stats = {}
    for side, trig in [("bottom", bottom_trig), ("top", top_trig)]:
        idx = np.flatnonzero(trig)
        idx = idx[(idx < n - horizon) & (ts_all[idx] >= np.datetime64(START))]
        idx_before_dedup = len(idx)
        idx = cluster_dedup(idx, kalman_dev_z_all[idx], most_negative=(side == "bottom"), gap=gap)
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after cluster-anchor dedup (gap={gap})")
        dedup_stats[side] = {"raw": int(idx_before_dedup), "deduped": int(len(idx))}

        entry = close[idx]
        a = atr_pct[idx]
        # touch-based MFE, intrabar high/low over bars[fire+1:fire+horizon+1] -- see module
        # docstring for the atr_pct-vs-raw-atr disambiguation.
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
            move_pct = (fut_ext - entry) / entry
        else:
            fut_ext = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
            move_pct = (entry - fut_ext) / entry
        hit = (move_pct >= k * a).astype(float)

        feat_rows = frame.iloc[idx]
        out = pd.DataFrame({
            "pos": idx, "timestamp": feat_rows["timestamp"].to_numpy(), "side": side,
            "hit": hit, "is_bottom": 1 if side == "bottom" else 0,
        })
        for col_name in FEATURE_COLUMNS:
            if col_name == "is_bottom":
                continue
            out[col_name] = feat_rows[col_name].to_numpy()
        rows.append(out)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    return fires, dedup_stats


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    """VERBATIM port from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py."""
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["hit"].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
            f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    return {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }


def compute_permutation_importance(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                                    seed: int = SEEDS[0], n_repeats: int = 5) -> dict:
    """VERBATIM port from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py -- single-
    seed, hand-rolled permutation importance (AUC-scored) on the VAL set."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(train[feature_cols], train["hit"].to_numpy().astype(int))
    y = eval_df["hit"].to_numpy().astype(int)
    X = eval_df[feature_cols].to_numpy()
    baseline_auc = roc_auc_score(y, clf.predict_proba(X)[:, 1])

    rng = np.random.default_rng(seed)
    rows = []
    for j, feat in enumerate(feature_cols):
        shuffled_aucs = []
        for _ in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            shuffled_aucs.append(roc_auc_score(y, clf.predict_proba(X_perm)[:, 1]))
        importance = baseline_auc - np.mean(shuffled_aucs)
        rows.append({"feature": feat, "importance_mean": round(float(importance), 5),
                     "importance_std": round(float(np.std(shuffled_aucs, ddof=1)), 5)})
    rows.sort(key=lambda r: -r["importance_mean"])
    return {"baseline_auc": round(float(baseline_auc), 4), "n_repeats": n_repeats, "seed": seed, "importances": rows}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    log("loading BTC Tier0 CSV...")
    frame = load_tier0()
    log(f"{len(frame)} bars loaded, {frame['timestamp'].min()} .. {frame['timestamp'].max()}")

    log("adding missing features (atr_pct/nyse_open_flag/er_24/realized_vol_ratio)...")
    frame = add_missing_features(frame)

    log("computing kalman_dev_z (per-bar recursive Kalman filter + rolling zscore, window=288)...")
    t0 = time.time()
    frame["kalman_dev_z"] = compute_kalman_dev_z(frame["close"].to_numpy())
    log(f"kalman_dev_z computed in {time.time() - t0:.1f}s")

    bottom_trig = (frame["kalman_dev_z"] <= -2.0).fillna(False).to_numpy()
    top_trig = (frame["kalman_dev_z"] >= 2.0).fillna(False).to_numpy()

    log("=== ETH-center (H=12,K=2.5) fixed-threshold raw-lift cross-check (see module docstring) ===")
    eth_center_crosscheck = eth_center_fixed_threshold_crosscheck(frame, bottom_trig, top_trig)
    log(f"  fixed-threshold lift: bottom={eth_center_crosscheck['lift_bottom_fixed_threshold']}x "
        f"top={eth_center_crosscheck['lift_top_fixed_threshold']}x  "
        f"(ETH's own reported raw lift: bottom 2.36x / top 2.16x)")
    log(f"  fire-bar mean atr_pct={eth_center_crosscheck['fire_bar_mean_atr_pct']} vs "
        f"pool mean atr_pct={eth_center_crosscheck['pool_mean_atr_pct']}")

    log("=== HORIZON x K grid screen (GBM VAL AUC selection, TRAIN-fit, no OOS/HOLDOUT access) ===")
    grid_rows, raw_chosen = run_grid_screen(frame, bottom_trig, top_trig)
    chosen, boundary_check = finalize_horizon_choice(frame, bottom_trig, top_trig, grid_rows, raw_chosen)
    HORIZON, K = chosen["horizon"], chosen["k"]

    log(f"=== building final fires+features at chosen HORIZON={HORIZON} K={K} GAP={CLUSTER_GAP} ===")
    fires, dedup_stats = build_fires_and_features(frame, bottom_trig, top_trig, HORIZON, K, CLUSTER_GAP)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side'] == 'bottom').sum())}, top={int((fires['side'] == 'top').sum())})")

    fire_hit_rate = float(fires["hit"].mean())
    log(f"pooled hit rate (deduped fires): {fire_hit_rate:.4f}")

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, "
        f"HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")

    fires.to_csv(FEATURES_CSV_PATH, index=False)
    log(f"features CSV saved -> {FEATURES_CSV_PATH}")

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"acc {val_result['accuracy_mean']:.4f}  bal_acc {val_result['balanced_accuracy_mean']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"acc {oos_result['accuracy_mean']:.4f}  bal_acc {oos_result['balanced_accuracy_mean']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (2026-04-01~latest, single-touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout fires"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}  "
            f"acc {holdout_result['accuracy_mean']:.4f}  bal_acc {holdout_result['balanced_accuracy_mean']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "kalman_deviation_meanrev",
        "asset": "BTCUSDT",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "label_definition": (
            f"touch-based MFE (intrabar high/low over bars[fire+1:fire+{HORIZON}+1]) "
            f">= K*atr_pct, K={K}, HORIZON={HORIZON}, cluster-anchor deduped (gap={CLUSTER_GAP}, "
            "keep most-extreme-kalman_dev_z bar per same-side cluster)"
        ),
        "provenance": (
            "BTC port of ETH's kalman_deviation_meanrev Homer candidate-pool signal "
            "(docs/homer/README.md '후보 풀' section, 2026-08-31; ETH VAL/OOS/HOLDOUT AUC "
            "0.6569/0.6311/0.6284 at H=12/GAP=12/K=2.5). This script: (1) computes BTC's own "
            "kalman_dev_z from the Tier0 CSV's close column (verbatim Kalman filter formula from "
            "live_evidence_signal_dashboard_20260823.py), (2) HORIZON x K grid confirm/refine "
            "(CPU, GBM VAL AUC selection -- see grid_screen.selection_metric_note for why this "
            "replaced an originally-planned raw lift-vs-random-baseline metric), (3) final TabPFN "
            "pipeline (run_tabpfn_panel/evaluate/compute_permutation_importance ported verbatim "
            "from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py) at the chosen "
            "(H,K), GAP fixed at 6 per task instruction (not re-screened)."
        ),
        "splits": {
            "train": f"<{VAL_START.date()}", "val": f"{VAL_START.date()}..{(OOS_START - pd.Timedelta(days=1)).date()}",
            "oos": f"{OOS_START.date()}..{(HOLDOUT_START - pd.Timedelta(days=1)).date()}",
            "holdout_single_touch": f">={HOLDOUT_START.date()}",
        },
        "eth_center_fixed_threshold_crosscheck": eth_center_crosscheck,
        "grid_screen": {
            "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
            "selection_metric": "argmax over eligible cells (n_train>=400, non-degenerate class split) of GBM (HistGradientBoostingClassifier) VAL AUC, TRAIN-fit, on the final clustered (GAP=6) 24-feature set",
            "selection_metric_note": (
                "Originally planned as 'lift vs random-bar baseline' (self-normalized by each "
                "bar's own atr_pct). That metric was tried first and found <1.0x in EVERY one of "
                "the 36 (H,K) cells, with the argmax sitting at a grid boundary (K=1.5) -- "
                "diagnosed as confounded by kalman_dev_z extremes co-occurring with locally "
                "elevated atr_pct at the fire bar (fire-bar atr_pct ~1.6-1.7x the non-fired "
                "pool's), which self-normalization punishes. A fixed-threshold cross-check (see "
                "eth_center_fixed_threshold_crosscheck) confirms genuine positive raw lift once "
                "this confound is removed, in the same ballpark as ETH's own reported figures. "
                "GBM VAL AUC was substituted as the selection metric -- ETH's own actual "
                "established methodology for this exact signal -- because atr_pct is one of the "
                "24 input features there rather than a hand-picked normalization constant, so it "
                "cannot silently invert the ranking the way it did in the raw-lift check. The "
                "self-normalized raw-lift diagnostic is still reported per cell (lift_*_pooled_"
                "self_normalized fields) for transparency, but was NOT used for selection."
            ),
            "grid": grid_rows,
            "raw_argmax": raw_chosen,
            "boundary_check": boundary_check,
            "boundary_check_note": (
                "raw_argmax is the mechanical argmax over the 36-cell grid; if it sat at the "
                "HORIZON_GRID boundary (H=8), boundary_check records a one-shot supplementary "
                "extension (H in [4,5,6,7]) and the decision rule that resolved it (see "
                "finalize_horizon_choice() docstring) -- null if raw_argmax was not at a boundary."
            ),
            "chosen": chosen,
        },
        "cluster_gap": CLUSTER_GAP,
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "dedup_stats": dedup_stats,
        "fired_signal_hit_rate_pooled": fire_hit_rate,
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "eth_own_result_for_comparison": ETH_OWN_RESULT,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
