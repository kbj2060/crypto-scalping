#!/usr/bin/env python3
"""BTC taker_delta_z_climax: HIT_TYPE x HORIZON x K grid-screen (round 2) + Tier0 feature analysis.

User request (2026-09-01, follow-up to round 1's research_btc_taker_delta_climax_gridscreen_
20260901.py): round 1 grid-screened HORIZON x K with ONE fixed hit definition (pure touch-based
MFE). The user asked a methodological question -- why assume touch-based-MFE is the right HIT
DEFINITION at all, shouldn't the hit definition itself also be grid-searched per signal? This
script adds HIT_TYPE as a THIRD search dimension: 4 hit_type x 6 HORIZON x 5 K = 120 cells.

Round 1's key finding this script must respect: BTC's taker_delta_z_climax wants a SHORT horizon
(lift peaked at H=12, monotonically decayed toward 1.0x by H=36) -- the OPPOSITE of ETH's own
taker_delta_z_climax (which needed H~24, 2 hours). HORIZON_GRID here is centered shorter than
round 1's [12,18,24,30,36] to give the screen room to find an even-shorter peak if one exists:
[6,9,12,18,24,30].

FOUR HIT_TYPE DEFINITIONS (entry=close[i], atr=atr[i] (CSV's raw price-scale ATR column, verified
== atr_price in scale, NOT atr_pct -- see round 1's docstring), candidate at row i, "bottom" =
bottom_taker_delta_z_climax candidate (predicts UP), "top" = top_taker_delta_z_climax candidate
(predicts DOWN)):

  1. touch_mfe (round 1's method, kept as this round's baseline): intrabar high/low touch anywhere
     in [i+1, i+H]. bottom: high[i+1:i+H+1].max() >= entry+K*atr. top: mirror on low/-K*atr.

  2. close_at_h (stricter -- no credit for touch-then-revert): only the bar-H CLOSE counts.
     bottom: close[i+H] >= entry+K*atr. top: close[i+H] <= entry-K*atr.

  3. touch_mae_capped (touch_mfe, disqualified if adverse excursion BEFORE the touch was too big):
     K_LOSS_MULT=2.0 fixed (this project's fib_extension_exhaustion MAE-cap constant -- NOT that
     script's exact window convention, which is whole-window/order-blind; this hit_type's MAE
     window is explicitly ORDER-DEPENDENT / path-dependent per this task's literal spec: only
     bars up to and including the first touch bar count toward MAE, not the full H window).
     bottom: touch_bar = first j in [i+1,i+H] with high[j]>=entry+K*atr (no touch -> hit=0);
     MAE = entry - low[i+1:touch_bar+1].min(); hit = touch AND MAE<=K_LOSS_MULT*atr. top: mirror.

  4. touch_giveback_sustained (V_REBOUND-style persistence check adapted as a candidate hit_type
     for THIS signal -- NOT literally V_REBOUND, which anchors off the trigger bar's own high/low
     extreme; this hit_type anchors off entry=close[i] like the other 3, for consistency across
     the grid): FAST_WINDOW=H, FULL_WINDOW=2*H (fixed multiple, not swept), giveback ceiling=0.20
     fixed (V_REBOUND's convention, see build_eth_5m_v_rebound_multitrigger_labels_20260831.py /
     research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome, whose
     FULL_BARS=2*FAST_BARS and T_SUSTAIN=0.20 match). bottom: fast_move=close[i+1:i+H+1].max()-
     entry; fast_mult=fast_move/atr; peak=high[i+1:i+2H+1].max(); end_price=close[i+2H];
     denom=peak-entry; giveback=(peak-end_price)/denom (NaN if |denom|<1e-12); hit = fast_mult>=K
     AND giveback<=0.20. top: mirror (fast_move on close.min(), peak on low.min(),
     denom=entry-peak, giveback=(end_price-peak)/denom).

VECTORIZATION: touch_mfe/close_at_h/touch_giveback_sustained are fully vectorized with pandas
rolling(window).max()/.min().shift(-window) (same trick as round 1's forward_extremes) computed
ONCE per HORIZON across the full 277K-row array, then indexed by candidate/baseline-draw position
-- cheap since none of their quantities depend on K. touch_mae_capped's "first touch bar" genuinely
depends on K (the threshold), so it cannot be reduced to a single global rolling op; it is instead
computed via a vectorized numpy gather (candidate_idx[:,None] + offsets[None,:], shape
[n_candidates, H]) run ONLY on the actual candidate/baseline-draw index arrays (thousands of rows,
H<=30 columns -- at most ~132K element matrices, not 277K bars), which is mathematically identical
to a per-candidate loop but avoids Python-level iteration.

METHODOLOGY:
  1. Fresh-Forward split by date: TRAIN<2025-09-01, VAL=2025-09-01..2025-12-31,
     OOS=2026-01-01..2026-03-31 (evaluated this round as a bonus check, NOT used for selection),
     HOLDOUT(>=2026-04-01) NEVER read/filtered/computed on.
  2. For every (hit_type, H, K): compute hit labels for bottom/top candidates on TRAIN, compute
     lift vs a same-count random-non-trigger-TRAIN-bar baseline evaluated under the IDENTICAL
     (hit_type,H,K) formula (bottom-side draws checked with the bottom/"up" formula, top-side with
     the top/"down" formula) -- isolates whether the |delta_z|>=2 trigger itself adds lift over an
     unconditional base rate, same isolation logic as round 1, now repeated per hit_type. Gate:
     n_train_bottom>=300 AND n_train_top>=300 AND n_train_bottom_hits>=30 AND n_train_top_hits>=30.
     Fixed RNG seed (20260901, single continuing stream in H->hit_type->K iteration order) for
     reproducibility.
  3. Selection metric (per task spec, NOT round 1's count-weighted pooled lift): min(train_lift_
     bottom, train_lift_top) -- rewards a hit_type/H/K that shows lift on BOTH sides, not one that
     wins on a count-weighted average while one side is flat/negative. argmax over all 120
     gate-passing cells = the global winner. VAL is then read at that exact point (no re-search);
     OOS is an additional bonus read (also no re-search, never used to pick the winner).
  4. Per-hit_type leaderboard: within each of the 4 hit_type families (30 cells each: 6H x 5K),
     the best gate-passing cell by the same min(lift_bottom,lift_top) metric, so the 4 families can
     be compared head-to-head, not just the single global argmax.
  5. Feature analysis at the GLOBAL winning (hit_type,H,K): pooled bottom+top TRAIN/VAL candidates,
     literal Tier0 21+rsi columns + is_bottom (extra structural column, not in Tier0, labeled
     separately) -- (a) TRAIN point-biserial correlation vs hit, (b) HistGradientBoostingClassifier
     TRAIN-fit -> permutation importance on VAL (roc_auc scoring). Sanity-only, not a promotion
     model -- identical caveat to round 1.

HOLDOUT (>=2026-04-01) is never read, filtered, or computed on in this script.

Run: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_taker_delta_climax_gridscreen_hittype_20260901.py

## ⚠️XRP 포팅 (2026-09-03)

`research_btc_<signal>_gridscreen_hittype_20260901.py`의 **자산 상수만** 바꾼 포팅이다.
선정 절차(클러스터 디둡 / 기간별 무작위 기준선 매칭 / 게이트+안정성 가드 / joint lift)는
한 줄도 바꾸지 않았다 -- 바꾸면 자산 간 비교가 무의미해진다.

⭐**HIT_TYPE을 반드시 재탐색하는 이유**: BTC 포팅에서 같은 이름의 신호가 ETH와
HIT_TYPE·H·K가 전부 달랐다(8종 중 완전히 같은 건 demarker 하나뿐). 서빙 코드가 원본을
따라가려는 관성 때문에 라이브 hit률이 2.6배 과대평가된 사고가 났다.
근거/절차: `docs/homer/evidence_signal_new_coin_port_protocol.md`

데이터: `data/labels/xrp_5m_evidence_signal_candidates_20260903/` (272,490행, 2024-01-01~2026-08-04)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/taker_delta_climax_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
OOS_END = pd.Timestamp("2026-04-01", tz="UTC")  # == HOLDOUT_START; HOLDOUT itself never touched
HOLDOUT_START = OOS_END

HIT_TYPES = ["touch_mfe", "close_at_h", "touch_mae_capped", "touch_giveback_sustained"]
HORIZON_GRID = [6, 9, 12, 18, 24, 30]
K_GRID = [1.5, 2.0, 2.4, 2.8, 3.2]

K_LOSS_MULT = 2.0        # touch_mae_capped, fixed (fib_extension_exhaustion project convention)
GIVEBACK_CEIL = 0.20     # touch_giveback_sustained, fixed (V_REBOUND project convention)
FULL_WINDOW_MULT = 2     # touch_giveback_sustained FULL_WINDOW = FULL_WINDOW_MULT * H, fixed

# Literal "21+rsi" Tier0 feature set (identical to round 1).
TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]
assert len(TIER0_FEATURES) == 22  # 21 + rsi

RNG_SEED = 20260901
MIN_TRAIN_CANDIDATES = 300
MIN_TRAIN_HITS = 30


def log(msg: str) -> None:
    print(f"[btc_taker_hittype_gridscreen] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    usecols = list(dict.fromkeys(
        ["timestamp", "high", "low", "close", "atr",
         "bottom_taker_delta_z_climax", "top_taker_delta_z_climax"] + TIER0_FEATURES
    ))
    df = pd.read_csv(CSV_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def fwd_roll_max(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).max().shift(-window).to_numpy()


def fwd_roll_min(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).min().shift(-window).to_numpy()


def fwd_shift(arr: np.ndarray, offset: int) -> np.ndarray:
    return pd.Series(arr).shift(-offset).to_numpy()


def build_horizon_bundle(h: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> dict:
    """Precompute every H-dependent (but K-independent) forward-looking array ONCE per horizon,
    shared across all 5 K values and (for touch_mfe/close_at_h/touch_giveback_sustained) all 4
    hit_types' threshold checks. touch_mae_capped is NOT precomputable this way (its "first touch
    bar" depends on K), so it is handled separately via a vectorized gather in hit_mae_capped()."""
    full = FULL_WINDOW_MULT * h

    fwd_high_max_H = fwd_roll_max(high, h)
    fwd_low_min_H = fwd_roll_min(low, h)
    fwd_close_H = fwd_shift(close, h)

    fwd_close_fast_max = fwd_roll_max(close, h)
    fwd_close_fast_min = fwd_roll_min(close, h)
    fwd_high_full_max = fwd_roll_max(high, full)
    fwd_low_full_min = fwd_roll_min(low, full)
    fwd_close_full = fwd_shift(close, full)

    entry = close
    fast_mult_bottom = (fwd_close_fast_max - entry) / atr
    peak_bottom = fwd_high_full_max
    denom_bottom = peak_bottom - entry
    fast_mult_top = (entry - fwd_close_fast_min) / atr
    peak_top = fwd_low_full_min
    denom_top = entry - peak_top
    # np.where evaluates both branches eagerly, so the true-branch division below computes a
    # (harmless, immediately discarded) inf/nan at the handful of NaN-edge/zero-denom rows that
    # the false-branch np.nan replaces -- np.errstate just silences the resulting RuntimeWarning;
    # the selected output values are unaffected (verified via the get_hit() np.isfinite(giveback)
    # gate, which correctly treats any inf/nan giveback as hit=False either way).
    with np.errstate(divide="ignore", invalid="ignore"):
        giveback_bottom = np.where(np.abs(denom_bottom) > 1e-12, (peak_bottom - fwd_close_full) / denom_bottom, np.nan)
        giveback_top = np.where(np.abs(denom_top) > 1e-12, (fwd_close_full - peak_top) / denom_top, np.nan)

    gb_valid = (np.isfinite(fwd_close_full) & np.isfinite(fwd_high_full_max) & np.isfinite(fwd_low_full_min)
                & np.isfinite(fwd_close_fast_max) & np.isfinite(fwd_close_fast_min))

    return {
        "fwd_high_max_H": fwd_high_max_H, "fwd_low_min_H": fwd_low_min_H,
        "mfe_valid": np.isfinite(fwd_high_max_H) & np.isfinite(fwd_low_min_H),
        "fwd_close_H": fwd_close_H, "close_valid": np.isfinite(fwd_close_H),
        "gb_fast_mult_bottom": fast_mult_bottom, "gb_giveback_bottom": giveback_bottom,
        "gb_fast_mult_top": fast_mult_top, "gb_giveback_top": giveback_top,
        "gb_valid": gb_valid,
    }


def get_valid(hit_type: str, idx_arr: np.ndarray, h: int, bundle: dict, n: int) -> np.ndarray:
    if hit_type == "touch_mfe":
        return bundle["mfe_valid"][idx_arr]
    if hit_type == "close_at_h":
        return bundle["close_valid"][idx_arr]
    if hit_type == "touch_mae_capped":
        return (idx_arr + h) <= (n - 1)
    if hit_type == "touch_giveback_sustained":
        return bundle["gb_valid"][idx_arr]
    raise ValueError(hit_type)


def hit_mae_capped(idx_v: np.ndarray, is_bottom: bool, h: int, k: float,
                    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """idx_v assumed already bounds-valid (idx_v+h <= n-1). Vectorized gather over
    shape [len(idx_v), h] -- equivalent to a per-candidate loop finding the first touch bar,
    then MAE up to (and including) that bar, per this task's literal path-dependent spec."""
    entry = close[idx_v]
    atr_v = atr[idx_v]
    offsets = np.arange(1, h + 1)
    window_idx = idx_v[:, None] + offsets[None, :]
    col_idx = np.arange(h)[None, :]

    if is_bottom:
        thr = entry + k * atr_v
        touch_mask = high[window_idx] >= thr[:, None]
        touch_exists = touch_mask.any(axis=1)
        first_pos = np.argmax(touch_mask, axis=1)
        mae_mask = col_idx <= first_pos[:, None]
        masked_low = np.where(mae_mask, low[window_idx], np.inf)
        mae = entry - masked_low.min(axis=1)
    else:
        thr = entry - k * atr_v
        touch_mask = low[window_idx] <= thr[:, None]
        touch_exists = touch_mask.any(axis=1)
        first_pos = np.argmax(touch_mask, axis=1)
        mae_mask = col_idx <= first_pos[:, None]
        masked_high = np.where(mae_mask, high[window_idx], -np.inf)
        mae = masked_high.max(axis=1) - entry

    return touch_exists & (mae <= K_LOSS_MULT * atr_v)


def get_hit(hit_type: str, idx_v: np.ndarray, is_bottom: bool, h: int, k: float, bundle: dict,
            high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """idx_v assumed already filtered to get_valid()==True for this (hit_type,h)."""
    if hit_type == "touch_mfe":
        entry, atr_v = close[idx_v], atr[idx_v]
        if is_bottom:
            return bundle["fwd_high_max_H"][idx_v] >= entry + k * atr_v
        return bundle["fwd_low_min_H"][idx_v] <= entry - k * atr_v
    if hit_type == "close_at_h":
        entry, atr_v = close[idx_v], atr[idx_v]
        fwd_close = bundle["fwd_close_H"][idx_v]
        if is_bottom:
            return fwd_close >= entry + k * atr_v
        return fwd_close <= entry - k * atr_v
    if hit_type == "touch_mae_capped":
        return hit_mae_capped(idx_v, is_bottom, h, k, high, low, close, atr)
    if hit_type == "touch_giveback_sustained":
        if is_bottom:
            fast_mult, giveback = bundle["gb_fast_mult_bottom"][idx_v], bundle["gb_giveback_bottom"][idx_v]
        else:
            fast_mult, giveback = bundle["gb_fast_mult_top"][idx_v], bundle["gb_giveback_top"][idx_v]
        return (fast_mult >= k) & np.isfinite(giveback) & (giveback <= GIVEBACK_CEIL)
    raise ValueError(hit_type)


def draw_baseline_hitrate(rng: np.random.Generator, pool_idx_v: np.ndarray, n_draw: int, hit_type: str,
                           is_bottom: bool, h: int, k: float, bundle: dict,
                           high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> float:
    if n_draw <= 0 or len(pool_idx_v) < n_draw:
        return float("nan")
    samp = rng.choice(pool_idx_v, size=n_draw, replace=False)
    hit = get_hit(hit_type, samp, is_bottom, h, k, bundle, high, low, close, atr)
    return float(hit.mean())


def confirm_val_oos(hit_type: str, h: int, k: float, bundle: dict,
                     bottom_idx_all: np.ndarray, top_idx_all: np.ndarray,
                     val_mask: np.ndarray, oos_mask: np.ndarray, any_trig: np.ndarray, finite_ok: np.ndarray,
                     n: int, high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
                     rng: np.random.Generator) -> dict:
    """VAL + OOS baseline confirmation at one (hit_type,h,k) cell -- same baseline-draw isolation
    logic as the TRAIN grid, just re-pooled from VAL-period / OOS-period non-trigger bars. Used
    both for the single global-winner confirmation and for each per-hit_type leaderboard entry, so
    the leaderboard shows whether each family's TRAIN pick actually holds out of sample, not just
    its TRAIN number."""
    valid_bottom = get_valid(hit_type, bottom_idx_all, h, bundle, n)
    valid_top = get_valid(hit_type, top_idx_all, h, bundle, n)
    bidx_v = bottom_idx_all[valid_bottom]
    tidx_v = top_idx_all[valid_top]

    pool_idx_val = np.flatnonzero(val_mask & (~any_trig) & finite_ok)
    pool_idx_val_v = pool_idx_val[get_valid(hit_type, pool_idx_val, h, bundle, n)]
    pool_idx_oos = np.flatnonzero(oos_mask & (~any_trig) & finite_ok)
    pool_idx_oos_v = pool_idx_oos[get_valid(hit_type, pool_idx_oos, h, bundle, n)]

    hit_b = get_hit(hit_type, bidx_v, True, h, k, bundle, high, low, close, atr)
    hit_t = get_hit(hit_type, tidx_v, False, h, k, bundle, high, low, close, atr)
    b_val_sel, t_val_sel = val_mask[bidx_v], val_mask[tidx_v]
    b_oos_sel, t_oos_sel = oos_mask[bidx_v], oos_mask[tidx_v]
    n_b_val, n_t_val = int(b_val_sel.sum()), int(t_val_sel.sum())
    n_b_oos, n_t_oos = int(b_oos_sel.sum()), int(t_oos_sel.sum())
    b_val_hr = float(hit_b[b_val_sel].mean()) if n_b_val else float("nan")
    t_val_hr = float(hit_t[t_val_sel].mean()) if n_t_val else float("nan")
    b_oos_hr = float(hit_b[b_oos_sel].mean()) if n_b_oos else float("nan")
    t_oos_hr = float(hit_t[t_oos_sel].mean()) if n_t_oos else float("nan")

    b_base_val = draw_baseline_hitrate(rng, pool_idx_val_v, n_b_val, hit_type, True, h, k, bundle, high, low, close, atr)
    t_base_val = draw_baseline_hitrate(rng, pool_idx_val_v, n_t_val, hit_type, False, h, k, bundle, high, low, close, atr)
    b_base_oos = draw_baseline_hitrate(rng, pool_idx_oos_v, n_b_oos, hit_type, True, h, k, bundle, high, low, close, atr)
    t_base_oos = draw_baseline_hitrate(rng, pool_idx_oos_v, n_t_oos, hit_type, False, h, k, bundle, high, low, close, atr)

    lift_val_bottom = b_val_hr / b_base_val if np.isfinite(b_base_val) and b_base_val > 0 else float("nan")
    lift_val_top = t_val_hr / t_base_val if np.isfinite(t_base_val) and t_base_val > 0 else float("nan")
    lift_oos_bottom = b_oos_hr / b_base_oos if np.isfinite(b_base_oos) and b_base_oos > 0 else float("nan")
    lift_oos_top = t_oos_hr / t_base_oos if np.isfinite(t_base_oos) and t_base_oos > 0 else float("nan")

    return {
        "bidx_v": bidx_v, "tidx_v": tidx_v, "hit_b": hit_b, "hit_t": hit_t,
        "n_val_bottom": n_b_val, "n_val_top": n_t_val, "n_oos_bottom": n_b_oos, "n_oos_top": n_t_oos,
        "val_hitrate_bottom": b_val_hr, "val_hitrate_top": t_val_hr,
        "val_baseline_bottom": b_base_val, "val_baseline_top": t_base_val,
        "val_lift_bottom": lift_val_bottom, "val_lift_top": lift_val_top,
        "oos_hitrate_bottom": b_oos_hr, "oos_hitrate_top": t_oos_hr,
        "oos_baseline_bottom": b_base_oos, "oos_baseline_top": t_base_oos,
        "oos_lift_bottom": lift_oos_bottom, "oos_lift_top": lift_oos_top,
    }


def rnd(x: float, nd: int = 4):
    return round(float(x), nd) if np.isfinite(x) else None


def main() -> int:
    log("loading BTC Tier0 candidates CSV...")
    df = load_data()
    n = len(df)
    log(f"{n} rows loaded, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)
    ts_col = df["timestamp"]

    bottom_trig = df["bottom_taker_delta_z_climax"].fillna(False).to_numpy(dtype=bool)
    top_trig = df["top_taker_delta_z_climax"].fillna(False).to_numpy(dtype=bool)
    any_trig = bottom_trig | top_trig
    assert not (bottom_trig & top_trig).any(), "bottom/top should be mutually exclusive"

    train_mask = (ts_col < VAL_START).to_numpy()
    val_mask = ((ts_col >= VAL_START) & (ts_col < OOS_START)).to_numpy()
    oos_mask = ((ts_col >= OOS_START) & (ts_col < OOS_END)).to_numpy()
    finite_ok = np.isfinite(close) & np.isfinite(atr) & np.isfinite(high) & np.isfinite(low)

    bottom_idx_all = np.flatnonzero(bottom_trig & finite_ok)
    top_idx_all = np.flatnonzero(top_trig & finite_ok)
    pool_idx_train_all = np.flatnonzero(train_mask & (~any_trig) & finite_ok)

    log(f"raw trigger counts: bottom={len(bottom_idx_all)} top={len(top_idx_all)} "
        f"(TRAIN pool for baseline draws: {len(pool_idx_train_all)})")

    master_rng = np.random.default_rng(RNG_SEED)
    grid_rows: list[dict] = []
    bundle_cache: dict[int, dict] = {}

    for h in HORIZON_GRID:
        bundle = build_horizon_bundle(h, high, low, close, atr)
        bundle_cache[h] = bundle

        for hit_type in HIT_TYPES:
            valid_bottom = get_valid(hit_type, bottom_idx_all, h, bundle, n)
            valid_top = get_valid(hit_type, top_idx_all, h, bundle, n)
            valid_pool = get_valid(hit_type, pool_idx_train_all, h, bundle, n)
            bidx_v = bottom_idx_all[valid_bottom]
            tidx_v = top_idx_all[valid_top]
            pool_idx_v = pool_idx_train_all[valid_pool]

            b_train_sel, b_val_sel, b_oos_sel = train_mask[bidx_v], val_mask[bidx_v], oos_mask[bidx_v]
            t_train_sel, t_val_sel, t_oos_sel = train_mask[tidx_v], val_mask[tidx_v], oos_mask[tidx_v]
            n_b_train, n_t_train = int(b_train_sel.sum()), int(t_train_sel.sum())
            n_b_val, n_t_val = int(b_val_sel.sum()), int(t_val_sel.sum())
            n_b_oos, n_t_oos = int(b_oos_sel.sum()), int(t_oos_sel.sum())

            for k in K_GRID:
                hit_b = get_hit(hit_type, bidx_v, True, h, k, bundle, high, low, close, atr)
                hit_t = get_hit(hit_type, tidx_v, False, h, k, bundle, high, low, close, atr)

                n_b_train_hits = int(hit_b[b_train_sel].sum())
                n_t_train_hits = int(hit_t[t_train_sel].sum())
                b_train_hr = float(hit_b[b_train_sel].mean()) if n_b_train else float("nan")
                t_train_hr = float(hit_t[t_train_sel].mean()) if n_t_train else float("nan")
                b_val_hr = float(hit_b[b_val_sel].mean()) if n_b_val else float("nan")
                t_val_hr = float(hit_t[t_val_sel].mean()) if n_t_val else float("nan")
                b_oos_hr = float(hit_b[b_oos_sel].mean()) if n_b_oos else float("nan")
                t_oos_hr = float(hit_t[t_oos_sel].mean()) if n_t_oos else float("nan")

                b_base = draw_baseline_hitrate(master_rng, pool_idx_v, n_b_train, hit_type, True, h, k, bundle,
                                                high, low, close, atr)
                t_base = draw_baseline_hitrate(master_rng, pool_idx_v, n_t_train, hit_type, False, h, k, bundle,
                                                high, low, close, atr)
                lift_bottom = b_train_hr / b_base if np.isfinite(b_base) and b_base > 0 else float("nan")
                lift_top = t_train_hr / t_base if np.isfinite(t_base) and t_base > 0 else float("nan")
                min_lift = min(lift_bottom, lift_top) if np.isfinite(lift_bottom) and np.isfinite(lift_top) else float("nan")

                eligible = (n_b_train >= MIN_TRAIN_CANDIDATES and n_t_train >= MIN_TRAIN_CANDIDATES
                            and n_b_train_hits >= MIN_TRAIN_HITS and n_t_train_hits >= MIN_TRAIN_HITS)

                grid_rows.append({
                    "hit_type": hit_type, "horizon": h, "k": k,
                    "n_train_bottom": n_b_train, "n_train_top": n_t_train,
                    "n_train_bottom_hits": n_b_train_hits, "n_train_top_hits": n_t_train_hits,
                    "n_val_bottom": n_b_val, "n_val_top": n_t_val,
                    "n_oos_bottom": n_b_oos, "n_oos_top": n_t_oos,
                    "train_hitrate_bottom": round(b_train_hr, 4) if np.isfinite(b_train_hr) else None,
                    "train_hitrate_top": round(t_train_hr, 4) if np.isfinite(t_train_hr) else None,
                    "train_baseline_bottom": round(b_base, 4) if np.isfinite(b_base) else None,
                    "train_baseline_top": round(t_base, 4) if np.isfinite(t_base) else None,
                    "lift_bottom": round(lift_bottom, 4) if np.isfinite(lift_bottom) else None,
                    "lift_top": round(lift_top, 4) if np.isfinite(lift_top) else None,
                    "min_lift": round(min_lift, 4) if np.isfinite(min_lift) else None,
                    "val_hitrate_bottom": round(b_val_hr, 4) if np.isfinite(b_val_hr) else None,
                    "val_hitrate_top": round(t_val_hr, 4) if np.isfinite(t_val_hr) else None,
                    "oos_hitrate_bottom": round(b_oos_hr, 4) if np.isfinite(b_oos_hr) else None,
                    "oos_hitrate_top": round(t_oos_hr, 4) if np.isfinite(t_oos_hr) else None,
                    "eligible": eligible,
                })
            just_added = grid_rows[-len(K_GRID):]
            lifts_this_row = [r["min_lift"] for r in just_added if r["min_lift"] is not None]
            lift_range = f"[{min(lifts_this_row):.3f}, {max(lifts_this_row):.3f}]" if lifts_this_row else "[no eligible K]"
            log(f"H={h:>2d} {hit_type:<24s} TRAIN n(bot/top)={n_b_train}/{n_t_train}  K-sweep min_lift range={lift_range}")

    grid_df = pd.DataFrame(grid_rows)
    eligible_df = grid_df[grid_df["eligible"]].dropna(subset=["min_lift"])
    if eligible_df.empty:
        raise RuntimeError("no grid cell passed the eligibility gate -- cannot select a winner")
    best_row = eligible_df.loc[eligible_df["min_lift"].idxmax()]
    chosen_hit_type = str(best_row["hit_type"])
    chosen_h = int(best_row["horizon"])
    chosen_k = float(best_row["k"])
    log(f"GLOBAL WINNER: hit_type={chosen_hit_type} H={chosen_h} K={chosen_k}  "
        f"min_lift={best_row['min_lift']:.3f}x (lift_bottom={best_row['lift_bottom']:.3f}x, "
        f"lift_top={best_row['lift_top']:.3f}x)")

    # ---- per-hit_type leaderboard, each family-best ALSO confirmed on VAL+OOS (not just TRAIN) ----
    # confs_by_hit_type is cached here and REUSED for the "chosen" section below (rather than
    # calling confirm_val_oos a second time) -- the global winner's (H,K) always exactly equals its
    # own family's leaderboard entry, so recomputing would just burn two more RNG draws and report
    # two slightly different numbers for the identical cell within the same run.
    leaderboard = {}
    confs_by_hit_type = {}
    for ht in HIT_TYPES:
        sub = eligible_df[eligible_df["hit_type"] == ht]
        if sub.empty:
            leaderboard[ht] = None
            log(f"leaderboard[{ht}]: NO eligible cells")
            continue
        row = sub.loc[sub["min_lift"].idxmax()]
        ht_h, ht_k = int(row["horizon"]), float(row["k"])
        conf = confirm_val_oos(ht, ht_h, ht_k, bundle_cache[ht_h], bottom_idx_all, top_idx_all,
                                val_mask, oos_mask, any_trig, finite_ok, n, high, low, close, atr, master_rng)
        confs_by_hit_type[ht] = (ht_h, ht_k, conf)
        # .item() converts numpy scalars (int64/float64/bool_) to native Python types so json.dumps
        # emits real numbers/booleans instead of falling back to default=str stringification.
        entry = {kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in row.to_dict().items()}
        entry.update({
            "val_baseline_bottom": rnd(conf["val_baseline_bottom"]), "val_baseline_top": rnd(conf["val_baseline_top"]),
            "val_lift_bottom": rnd(conf["val_lift_bottom"]), "val_lift_top": rnd(conf["val_lift_top"]),
            "val_min_lift": rnd(min(conf["val_lift_bottom"], conf["val_lift_top"])) if np.isfinite(conf["val_lift_bottom"]) and np.isfinite(conf["val_lift_top"]) else None,
            "oos_hitrate_bottom": rnd(conf["oos_hitrate_bottom"]), "oos_hitrate_top": rnd(conf["oos_hitrate_top"]),
            "oos_lift_bottom": rnd(conf["oos_lift_bottom"]), "oos_lift_top": rnd(conf["oos_lift_top"]),
        })
        leaderboard[ht] = entry
        log(f"leaderboard[{ht}]: H={ht_h} K={ht_k}  TRAIN min_lift={row['min_lift']:.3f}x "
            f"(bot={row['lift_bottom']:.3f}x top={row['lift_top']:.3f}x, n_hits bot/top={int(row['n_train_bottom_hits'])}/{int(row['n_train_top_hits'])})  "
            f"VAL lift bot/top={conf['val_lift_bottom']:.3f}x/{conf['val_lift_top']:.3f}x "
            f"(hitrate {conf['val_hitrate_bottom']:.4f}/{conf['val_hitrate_top']:.4f}, n={conf['n_val_bottom']}/{conf['n_val_top']})")

    # ---- round-1 cross-check: touch_mfe @ H=12,K=2.0 should reproduce round 1's numbers exactly ----
    r1_row = grid_df[(grid_df["hit_type"] == "touch_mfe") & (grid_df["horizon"] == 12) & (grid_df["k"] == 2.0)]
    if not r1_row.empty:
        r1 = r1_row.iloc[0]
        log(f"round-1 cross-check (touch_mfe H=12 K=2.0): lift_bottom={r1['lift_bottom']} lift_top={r1['lift_top']} "
            f"(round 1 reported TRAIN bottom 1.103x / top 1.123x, VAL hit pooled 40.00%) "
            f"-- this round's VAL hit(bot/top)={r1['val_hitrate_bottom']}/{r1['val_hitrate_top']}")

    # ---- VAL + OOS baseline confirmation at the GLOBAL chosen cell (reused from the leaderboard
    # pass above -- the global winner's (H,K) always equals its own family's leaderboard entry) ----
    conf_h, conf_k, chosen_conf = confs_by_hit_type[chosen_hit_type]
    assert (conf_h, conf_k) == (chosen_h, chosen_k), "global winner should equal its own family's leaderboard entry"
    bidx_v, tidx_v = chosen_conf["bidx_v"], chosen_conf["tidx_v"]
    hit_b, hit_t = chosen_conf["hit_b"], chosen_conf["hit_t"]
    n_b_val, n_t_val = chosen_conf["n_val_bottom"], chosen_conf["n_val_top"]
    n_b_oos, n_t_oos = chosen_conf["n_oos_bottom"], chosen_conf["n_oos_top"]
    b_val_hr, t_val_hr = chosen_conf["val_hitrate_bottom"], chosen_conf["val_hitrate_top"]
    b_oos_hr, t_oos_hr = chosen_conf["oos_hitrate_bottom"], chosen_conf["oos_hitrate_top"]
    b_base_val, t_base_val = chosen_conf["val_baseline_bottom"], chosen_conf["val_baseline_top"]
    b_base_oos, t_base_oos = chosen_conf["oos_baseline_bottom"], chosen_conf["oos_baseline_top"]
    lift_val_bottom, lift_val_top = chosen_conf["val_lift_bottom"], chosen_conf["val_lift_top"]
    lift_oos_bottom, lift_oos_top = chosen_conf["oos_lift_bottom"], chosen_conf["oos_lift_top"]
    log(f"VAL confirm @ GLOBAL chosen: lift_bottom={lift_val_bottom:.3f}x lift_top={lift_val_top:.3f}x "
        f"(TRAIN was {best_row['lift_bottom']:.3f}x / {best_row['lift_top']:.3f}x)  "
        f"n_train_hits bot/top={int(best_row['n_train_bottom_hits'])}/{int(best_row['n_train_top_hits'])}")
    log(f"OOS bonus @ GLOBAL chosen:   lift_bottom={lift_oos_bottom:.3f}x lift_top={lift_oos_top:.3f}x")
    if lift_val_top < 1.0 or lift_val_bottom < 1.0:
        log("WARNING: global TRAIN-argmax winner does NOT hold on VAL for at least one side "
            "(lift<1.0x = worse than random baseline) -- likely a thin/overfit grid corner, see leaderboard for a more robust alternative")

    # ---- feature analysis at chosen (hit_type, H, K) ----
    feat_bottom = df.iloc[bidx_v][["timestamp"] + TIER0_FEATURES].copy()
    feat_bottom["hit"] = hit_b.astype(int)
    feat_bottom["is_bottom"] = 1
    feat_top = df.iloc[tidx_v][["timestamp"] + TIER0_FEATURES].copy()
    feat_top["hit"] = hit_t.astype(int)
    feat_top["is_bottom"] = 0

    feat_all = pd.concat([feat_bottom, feat_top], ignore_index=True)
    n_before_dropna = len(feat_all)
    feat_all = feat_all.dropna(subset=TIER0_FEATURES + ["hit"]).reset_index(drop=True)
    log(f"feature-analysis candidates: {len(feat_all)}/{n_before_dropna} after dropna on Tier0 set")

    feat_cols = TIER0_FEATURES + ["is_bottom"]
    train_feat = feat_all[feat_all["timestamp"] < VAL_START].reset_index(drop=True)
    val_feat = feat_all[(feat_all["timestamp"] >= VAL_START) & (feat_all["timestamp"] < OOS_START)].reset_index(drop=True)
    log(f"feature-analysis split: TRAIN={len(train_feat)} VAL={len(val_feat)}")

    y_train = train_feat["hit"].to_numpy()
    corr_rows = []
    for col in feat_cols:
        x = train_feat[col].to_numpy(dtype=float)
        r, p = pearsonr(y_train.astype(float), x)
        corr_rows.append({"feature": col, "point_biserial_r": round(float(r), 4), "p_value": round(float(p), 6)})
    corr_rows.sort(key=lambda row: -abs(row["point_biserial_r"]))
    log("=== point-biserial correlation vs hit (TRAIN) ===")
    for row in corr_rows[:10]:
        log(f"  {row['feature']:<22s} r={row['point_biserial_r']:+.4f} (p={row['p_value']:.4g})")

    clf = HistGradientBoostingClassifier(random_state=RNG_SEED, max_iter=200)
    clf.fit(train_feat[feat_cols], y_train)
    val_proba = clf.predict_proba(val_feat[feat_cols])[:, 1]
    val_auc = float(roc_auc_score(val_feat["hit"], val_proba))
    log(f"HistGBM sanity fit (NOT promotion-grade): VAL AUC={val_auc:.4f}")

    perm = permutation_importance(clf, val_feat[feat_cols], val_feat["hit"], scoring="roc_auc",
                                   n_repeats=20, random_state=RNG_SEED)
    perm_rows = [
        {"feature": feat_cols[i], "importance_mean": round(float(perm.importances_mean[i]), 5),
         "importance_std": round(float(perm.importances_std[i]), 5)}
        for i in range(len(feat_cols))
    ]
    perm_rows.sort(key=lambda row: -row["importance_mean"])
    log("=== permutation importance (VAL, roc_auc scoring, 20 repeats) ===")
    for row in perm_rows[:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "taker_delta_z_climax", "asset": "BTC",
        "stage": "gridscreen_hittype_featureanalysis_only",
        "round": 2, "round1_script": "scripts/research_btc_taker_delta_climax_gridscreen_20260901.py",
        "tabpfn_trained": False, "economic_cost_gate_run": False, "holdout_touched": False,
        "clustering_dedup_applied": False,
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "data_source": str(CSV_PATH),
        "splits": {"train": f"<{VAL_START.date()}", "val": f"{VAL_START.date()}..{(OOS_START - pd.Timedelta(days=1)).date()}",
                   "oos_bonus_not_used_for_selection": f"{OOS_START.date()}..{(OOS_END - pd.Timedelta(days=1)).date()}",
                   "holdout_not_touched": f">={HOLDOUT_START.date()}"},
        "hit_types": HIT_TYPES, "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
        "k_loss_mult": K_LOSS_MULT, "giveback_ceiling": GIVEBACK_CEIL, "full_window_mult": FULL_WINDOW_MULT,
        "gate_min_train_candidates": MIN_TRAIN_CANDIDATES, "gate_min_train_hits": MIN_TRAIN_HITS,
        "selection_metric": "argmax over gate-passing cells of min(train_lift_bottom, train_lift_top)",
        "raw_trigger_counts": {"bottom_total": int(len(bottom_idx_all)), "top_total": int(len(top_idx_all))},
        "grid": grid_rows,
        "leaderboard_by_hit_type": leaderboard,
        "chosen": {
            "hit_type": chosen_hit_type, "horizon": chosen_h, "k": chosen_k,
            "train_lift_bottom": float(best_row["lift_bottom"]), "train_lift_top": float(best_row["lift_top"]),
            "train_min_lift": float(best_row["min_lift"]),
            "train_hitrate_bottom": float(best_row["train_hitrate_bottom"]), "train_hitrate_top": float(best_row["train_hitrate_top"]),
            "train_baseline_bottom": float(best_row["train_baseline_bottom"]), "train_baseline_top": float(best_row["train_baseline_top"]),
            "n_train_bottom": int(best_row["n_train_bottom"]), "n_train_top": int(best_row["n_train_top"]),
            "val_hitrate_bottom": round(b_val_hr, 4) if np.isfinite(b_val_hr) else None,
            "val_hitrate_top": round(t_val_hr, 4) if np.isfinite(t_val_hr) else None,
            "val_baseline_bottom": round(b_base_val, 4) if np.isfinite(b_base_val) else None,
            "val_baseline_top": round(t_base_val, 4) if np.isfinite(t_base_val) else None,
            "val_lift_bottom": round(lift_val_bottom, 4) if np.isfinite(lift_val_bottom) else None,
            "val_lift_top": round(lift_val_top, 4) if np.isfinite(lift_val_top) else None,
            "n_val_bottom": n_b_val, "n_val_top": n_t_val,
            "oos_hitrate_bottom": round(b_oos_hr, 4) if np.isfinite(b_oos_hr) else None,
            "oos_hitrate_top": round(t_oos_hr, 4) if np.isfinite(t_oos_hr) else None,
            "oos_baseline_bottom": round(b_base_oos, 4) if np.isfinite(b_base_oos) else None,
            "oos_baseline_top": round(t_base_oos, 4) if np.isfinite(t_base_oos) else None,
            "oos_lift_bottom": round(lift_oos_bottom, 4) if np.isfinite(lift_oos_bottom) else None,
            "oos_lift_top": round(lift_oos_top, 4) if np.isfinite(lift_oos_top) else None,
            "n_oos_bottom": n_b_oos, "n_oos_top": n_t_oos,
        },
        "round1_touch_mfe_h12_k2_crosscheck": (
            {"lift_bottom": r1_row.iloc[0]["lift_bottom"], "lift_top": r1_row.iloc[0]["lift_top"],
             "val_hitrate_bottom": r1_row.iloc[0]["val_hitrate_bottom"], "val_hitrate_top": r1_row.iloc[0]["val_hitrate_top"]}
            if not r1_row.empty else None
        ),
        "feature_analysis": {
            "tier0_feature_columns_literal_21_plus_rsi": TIER0_FEATURES,
            "extra_structural_column_not_in_tier0": "is_bottom",
            "n_train": int(len(train_feat)), "n_val": int(len(val_feat)),
            "n_before_dropna": n_before_dropna,
            "point_biserial_train": corr_rows,
            "histgbm_val_auc_sanity_only": round(val_auc, 4),
            "permutation_importance_val": perm_rows,
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str, ensure_ascii=False))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
