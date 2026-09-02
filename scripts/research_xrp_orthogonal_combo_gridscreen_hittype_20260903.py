#!/usr/bin/env python3
"""Round 2 of BTC orthogonal_combo grid screening: HIT_TYPE added as a THIRD search dimension
alongside HORIZON and K (round 1: scripts/research_btc_orthogonal_combo_gridscreen_20260901.py,
docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md).

Motivation (user question, 2026-09-01): round 1 used ONE fixed HIT definition (pure touch-based
MFE, no persistence check) and flagged a real concern -- OOS lift dropped to 0.929 (below 1.0) at
its selected point (H=12,K=3.0), reseeded 5x, consistently <1. The user asked: why assume
touch-based-MFE is the right HIT definition at all -- shouldn't the HIT DEFINITION ITSELF also be
grid-searched per signal, not just H and K? This script answers that by sweeping 4 HIT_TYPE
families x HORIZON x K jointly, and checks whether a stricter/persistence-aware HIT_TYPE is more
OOS-robust than plain touch.

Data (unchanged, already built, NOT recomputed here): data/labels/btc_5m_evidence_signal_
candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv (277,191 rows, 2024-01-01 to
2026-08-20, BTCUSDT 5m). bottom_orthogonal_combo/top_orthogonal_combo triggers already computed.

HIT_TYPE families (entry=close[i], atr=atr[i] at the candidate row; all touch/close conditions use
intrabar high/low except where noted):
  1. touch_mfe (round 1's method, kept as baseline for comparison):
       bottom: hit=1 if high[i+1:i+H+1].max() >= entry+K*atr
       top:    hit=1 if low[i+1:i+H+1].min()  <= entry-K*atr
  2. close_at_h (stricter -- only the bar-H CLOSE counts, no credit for touch-then-revert):
       bottom: hit=1 if close[i+H] >= entry+K*atr
       top:    hit=1 if close[i+H] <= entry-K*atr
  3. touch_mae_capped (touch_mfe, disqualified if price ran too far against the position FIRST):
       K_LOSS_MULT=2.0 fixed (this project's fib_extension_exhaustion MAE-cap convention, not swept).
       bottom: touch_bar = first bar in [i+1,i+H] where high>=entry+K*atr (no touch -> not a hit);
               MAE = entry - low[i+1:touch_bar+1].min(); hit=1 if touched AND MAE<=K_LOSS_MULT*atr.
       top: mirror (MAE = high[i+1:touch_bar+1].max() - entry).
  4. touch_giveback_sustained (V_REBOUND-style persistence check, ported as a candidate HIT_TYPE
     for THIS signal -- not literally V_REBOUND itself):
       FAST_WINDOW=H, FULL_WINDOW=2*H (fixed multiple, not swept), giveback ceiling=0.20 fixed
       (this project's V_REBOUND convention).
       bottom: fast_move=close[i+1:i+FAST_WINDOW+1].max()-entry (NOTE: close-based, not high --
               deliberately less wick-noisy than the MFE/peak measurement below);
               fast_mult=fast_move/atr; peak=high[i+1:i+FULL_WINDOW+1].max();
               end_price=close[i+FULL_WINDOW]; denom=peak-entry;
               giveback=(peak-end_price)/denom (denom~0 -> giveback=0, NaN-safe);
               hit=1 if fast_mult>=K AND giveback<=0.20.
       top: mirror (fast_move off close.min(), trough=low[...].min(), denom=entry-trough,
               giveback=(end_price-trough)/denom).

Grid: HIT_TYPE in the 4 above x HORIZON in [8,12,18,24,30,36] (round 1's TRAIN lift decreased
monotonically with horizon, peaking at the grid's lower boundary H=12 -- extended down to H=8 this
round) x K in [2.0,2.5,3.0,3.57,4.0,4.5] (added K=2.0 vs round 1's [2.5..4.5]). 144 (HIT_TYPE,H,K)
combos total.

Methodology assumption stated explicitly (task spec is silent on this): GAP=12 cluster-dedup
embargo is KEPT, verbatim from round 1 (itself ported from ETH's own orthogonal_combo screen) --
consecutive raw trigger fires within 12 bars are merged into one cluster, keeping only the most
oscillator-extreme row. The new task's methodology section is a complete rewrite of round 1's (every
other fixed parameter -- MAE cap ratio, giveback window/ceiling -- is explicitly restated as
"fixed, don't sweep"), but says nothing about removing dedup. Silently dropping it would treat
bursty, autocorrelated consecutive fires (same underlying extreme event) as independent samples,
which would be a real methodological regression, not something the task asked for. Verified impact:
TRAIN raw (non-deduped) fires = 1420 bottom / 1066 top vs deduped = 831 bottom / 710 top at H=12 --
dedup materially thins and decorrelates the candidate pool, so this choice matters and is flagged
here plus in the output doc rather than picked silently.

Per-HIT_TYPE candidate validity cutoff: touch_mfe / close_at_h / touch_mae_capped all need bar i+H
to exist (idx < n-H); touch_giveback_sustained needs bar i+2H to exist (idx < n-2H, its FULL_WINDOW).
Two dedup passes are built per horizon (cutoff=H and cutoff=2H) to serve these correctly -- see
build_candidate_idx()/idx_cache_for().

Methodology (per task spec):
  1. Fresh-forward split by DATE: TRAIN<2025-09-01, VAL=[2025-09-01,2026-01-01),
     OOS=[2026-01-01,2026-04-01). HOLDOUT(>=2026-04-01) dropped from the frame immediately after
     load -- never touched, never read past that point.
  2. For every (HIT_TYPE,H,K): compute hit labels for bottom_orthogonal_combo/top_orthogonal_combo
     candidates on TRAIN; compute lift vs a same-count random-non-trigger-bar baseline at the SAME
     (HIT_TYPE,H,K), drawn separately per side from the TRAIN pool, tested with the same directional
     formula. Gate on n_cand>=300 AND n_hits>=30, BOTH sides independently.
  3. Selection = argmax over the full (HIT_TYPE,H,K) grid of min(train_lift_bottom, train_lift_top)
     among gate-passing cells (matches this project's fib_extension_exhaustion BTC screen's
     per-side-lift + min() selection convention -- NOT round 1's pooled-both-sides single lift).
     Confirm on VAL (same point, no re-search); report OOS at that same point too (prominent, not
     for selection, given round 1's OOS concern for this specific signal).
  4. Per-HIT_TYPE leaderboard: within each of the 4 families' own 36-cell sub-grid, the same
     selection rule picks that family's own best (H,K); VAL+OOS reported at each family's point.
     The global winner (step 3) is necessarily one of these 4 rows.
  5. Feature analysis (Tier0 22 features incl. rsi) at the GLOBAL winning (HIT_TYPE,H,K):
     point-biserial correlation (TRAIN) + HistGradientBoostingClassifier (TRAIN-fit) permutation
     importance (VAL-scored) -- same method as round 1.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_orthogonal_combo_gridscreen_hittype_20260901.py

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
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import pointbiserialr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/orthogonal_combo_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

GAP = 12  # kept from round 1, see docstring assumption note above -- fixed, not swept
HIT_TYPES = ["touch_mfe", "close_at_h", "touch_mae_capped", "touch_giveback_sustained"]
HORIZON_GRID = [8, 12, 18, 24, 30, 36]
K_GRID = [2.0, 2.5, 3.0, 3.57, 4.0, 4.5]
MIN_TRAIN_CANDIDATES = 300  # per side
MIN_TRAIN_HITS = 30         # per side
K_LOSS_MULT = 2.0           # touch_mae_capped MAE cap, fixed per task spec
GIVEBACK_CEIL = 0.20        # touch_giveback_sustained giveback ratio ceiling, fixed per task spec
RNG_SEED = 20260901

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high", "range_width_pct",
    "hour_utc", "weekday", "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
    "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def log(msg: str) -> None:
    print(f"[btc_orthogonal_hittype] {msg}", flush=True)


# ---------------------------------------------------------------------------
# candidate dedup (verbatim from round 1 / ETH orthogonal_combo convention)
# ---------------------------------------------------------------------------

def cluster_dedup(idx: np.ndarray, p_fast: np.ndarray, p_slow: np.ndarray, side: str, gap: int) -> np.ndarray:
    if len(idx) == 0:
        return idx
    score = -(p_fast[idx] + p_slow[idx]) if side == "bottom" else (p_fast[idx] + p_slow[idx])
    cluster_id = np.zeros(len(idx), dtype=int)
    cid = 0
    for i in range(1, len(idx)):
        if idx[i] - idx[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx, "cluster": cluster_id, "s": score})
    return np.sort(df.loc[df.groupby("cluster")["s"].idxmax(), "idx"].to_numpy())


def build_candidate_idx(frame: pd.DataFrame, side: str, col: str, cutoff: int, gap: int) -> np.ndarray:
    """Deduped candidate row-positions for one side at a given lookahead cutoff (H for
    touch_mfe/close_at_h/touch_mae_capped; 2H for touch_giveback_sustained's FULL_WINDOW)."""
    n = len(frame)
    atr = frame["atr"].to_numpy()
    close = frame["close"].to_numpy()
    p_fast = frame["p_fast"].to_numpy()
    p_slow = frame["p_slow"].to_numpy()
    idx = np.flatnonzero(frame[col].to_numpy())
    idx = idx[(idx < n - cutoff) & np.isfinite(atr[idx]) & (atr[idx] > 0) & np.isfinite(close[idx])]
    return cluster_dedup(idx, p_fast, p_slow, side, gap)


# ---------------------------------------------------------------------------
# HIT_TYPE formulas
# ---------------------------------------------------------------------------

def hits_touch_mfe(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=int)
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy()
    close = frame["close"].to_numpy(); atr = frame["atr"].to_numpy()
    entry, a = close[idx], atr[idx]
    if side == "bottom":
        target = entry + k * a
        fut = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
        return (fut >= target).astype(int)
    target = entry - k * a
    fut = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
    return (fut <= target).astype(int)


def hits_close_at_h(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=int)
    close = frame["close"].to_numpy(); atr = frame["atr"].to_numpy()
    entry, a = close[idx], atr[idx]
    close_h = close[idx + horizon]
    if side == "bottom":
        return (close_h >= entry + k * a).astype(int)
    return (close_h <= entry - k * a).astype(int)


def hits_touch_mae_capped(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float,
                           k_loss_mult: float = K_LOSS_MULT) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=int)
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy()
    close = frame["close"].to_numpy(); atr = frame["atr"].to_numpy()
    out = np.zeros(len(idx), dtype=int)
    for n_i, i in enumerate(idx):
        entry = close[i]; a = atr[i]
        if side == "bottom":
            target = entry + k * a
            touched = np.flatnonzero(high[i + 1:i + horizon + 1] >= target)
            if touched.size == 0:
                continue
            t_rel = int(touched[0])
            mae = entry - low[i + 1:i + 2 + t_rel].min()
            if mae <= k_loss_mult * a:
                out[n_i] = 1
        else:
            target = entry - k * a
            touched = np.flatnonzero(low[i + 1:i + horizon + 1] <= target)
            if touched.size == 0:
                continue
            t_rel = int(touched[0])
            mae = high[i + 1:i + 2 + t_rel].max() - entry
            if mae <= k_loss_mult * a:
                out[n_i] = 1
    return out


def giveback_arrays(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """K-independent fast_mult/giveback arrays for touch_giveback_sustained (only the final
    fast_mult>=K threshold depends on K -- computed once per (side,horizon), reused across K)."""
    if len(idx) == 0:
        return np.array([]), np.array([])
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy()
    close = frame["close"].to_numpy(); atr = frame["atr"].to_numpy()
    full_window = 2 * horizon
    entry = close[idx]; a = atr[idx]
    end_price = close[idx + full_window]
    fast_mult = np.empty(len(idx)); giveback = np.empty(len(idx))
    for n_i, i in enumerate(idx):
        if side == "bottom":
            fast_move = close[i + 1:i + horizon + 1].max() - entry[n_i]
            peak = high[i + 1:i + full_window + 1].max()
            denom = peak - entry[n_i]
            gb = (peak - end_price[n_i]) / denom if denom > 1e-12 else 0.0
        else:
            fast_move = entry[n_i] - close[i + 1:i + horizon + 1].min()
            trough = low[i + 1:i + full_window + 1].min()
            denom = entry[n_i] - trough
            gb = (end_price[n_i] - trough) / denom if denom > 1e-12 else 0.0
        fast_mult[n_i] = fast_move / a[n_i]
        giveback[n_i] = gb
    return fast_mult, giveback


_GIVEBACK_CACHE: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray]] = {}


def hits_touch_giveback_sustained(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float) -> np.ndarray:
    if len(idx) == 0:
        return np.array([], dtype=int)
    # fast_mult/giveback are K-independent (see giveback_arrays docstring) -- cache by the exact
    # idx array's content so the same (horizon,side,scope) draw isn't recomputed for every K.
    key = (horizon, side, hash(idx.tobytes()))
    if key not in _GIVEBACK_CACHE:
        _GIVEBACK_CACHE[key] = giveback_arrays(frame, idx, side, horizon)
    fast_mult, giveback = _GIVEBACK_CACHE[key]
    return ((fast_mult >= k) & (giveback <= GIVEBACK_CEIL)).astype(int)


def compute_hits(frame: pd.DataFrame, idx: np.ndarray, side: str, horizon: int, k: float, hit_type: str) -> np.ndarray:
    if hit_type == "touch_mfe":
        return hits_touch_mfe(frame, idx, side, horizon, k)
    if hit_type == "close_at_h":
        return hits_close_at_h(frame, idx, side, horizon, k)
    if hit_type == "touch_mae_capped":
        return hits_touch_mae_capped(frame, idx, side, horizon, k)
    if hit_type == "touch_giveback_sustained":
        return hits_touch_giveback_sustained(frame, idx, side, horizon, k)
    raise ValueError(f"unknown hit_type {hit_type!r}")


# ---------------------------------------------------------------------------
# baseline + per-side / grid-row metrics
# ---------------------------------------------------------------------------

def random_baseline(frame: pd.DataFrame, side: str, count: int, horizon: int, k: float, hit_type: str,
                     scope_mask: np.ndarray, seed: int) -> tuple[float, int]:
    """Same-count random NON-trigger bars (neither bottom nor top orthogonal_combo fired) from the
    given scope, tested with the SAME (side,horizon,k,hit_type) directional formula."""
    n = len(frame)
    cutoff = 2 * horizon if hit_type == "touch_giveback_sustained" else horizon
    atr = frame["atr"].to_numpy(); close = frame["close"].to_numpy()
    not_trigger = ~(frame["bottom_orthogonal_combo"].to_numpy() | frame["top_orthogonal_combo"].to_numpy())
    valid_pos = np.arange(n) < (n - cutoff)
    eligible_mask = not_trigger & valid_pos & scope_mask & np.isfinite(atr) & (atr > 0) & np.isfinite(close)
    eligible_idx = np.flatnonzero(eligible_mask)
    if count == 0 or len(eligible_idx) == 0:
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    replace = count > len(eligible_idx)
    sampled = rng.choice(eligible_idx, size=count, replace=replace)
    hits = compute_hits(frame, sampled, side, horizon, k, hit_type)
    return float(hits.mean()), int(len(hits))


def side_metrics(frame: pd.DataFrame, idx_side: np.ndarray, side: str, scope_mask: np.ndarray,
                  horizon: int, k: float, hit_type: str, baseline_seed: int) -> dict:
    idx_scope = idx_side[scope_mask[idx_side]]
    hits = compute_hits(frame, idx_scope, side, horizon, k, hit_type)
    n = len(idx_scope)
    n_hits = int(hits.sum()) if n else 0
    hit_rate = n_hits / n if n else float("nan")
    baseline_rate, n_baseline = random_baseline(frame, side, n, horizon, k, hit_type, scope_mask, baseline_seed)
    lift = hit_rate / baseline_rate if n and baseline_rate and baseline_rate > 0 else float("nan")
    return {"n": n, "n_hits": n_hits, "hit_rate": hit_rate,
            "baseline_hit_rate": baseline_rate, "n_baseline": n_baseline, "lift": lift}


def idx_cache_for(hit_type: str, horizon: int, idx_h_cache: dict, idx_2h_cache: dict) -> dict:
    return idx_2h_cache[horizon] if hit_type == "touch_giveback_sustained" else idx_h_cache[horizon]


def build_grid_row(frame: pd.DataFrame, idx_h_cache: dict, idx_2h_cache: dict, hit_type: str, horizon: int,
                    k: float, scope_mask: np.ndarray, baseline_seed: int) -> dict:
    cache = idx_cache_for(hit_type, horizon, idx_h_cache, idx_2h_cache)
    bottom = side_metrics(frame, cache["bottom"], "bottom", scope_mask, horizon, k, hit_type, baseline_seed)
    top = side_metrics(frame, cache["top"], "top", scope_mask, horizon, k, hit_type, baseline_seed + 500)
    both_finite = np.isfinite(bottom["lift"]) and np.isfinite(top["lift"])
    joint = min(bottom["lift"], top["lift"]) if both_finite else float("nan")
    gate_passed = (bottom["n"] >= MIN_TRAIN_CANDIDATES and top["n"] >= MIN_TRAIN_CANDIDATES and
                   bottom["n_hits"] >= MIN_TRAIN_HITS and top["n_hits"] >= MIN_TRAIN_HITS)
    return {
        "hit_type": hit_type, "horizon": horizon, "k": k,
        "n_bottom": bottom["n"], "n_hits_bottom": bottom["n_hits"],
        "hit_rate_bottom": round(bottom["hit_rate"], 4) if bottom["n"] else None,
        "baseline_hit_rate_bottom": round(bottom["baseline_hit_rate"], 4) if np.isfinite(bottom["baseline_hit_rate"]) else None,
        "lift_bottom": round(bottom["lift"], 4) if np.isfinite(bottom["lift"]) else None,
        "n_top": top["n"], "n_hits_top": top["n_hits"],
        "hit_rate_top": round(top["hit_rate"], 4) if top["n"] else None,
        "baseline_hit_rate_top": round(top["baseline_hit_rate"], 4) if np.isfinite(top["baseline_hit_rate"]) else None,
        "lift_top": round(top["lift"], 4) if np.isfinite(top["lift"]) else None,
        "joint_min_lift": round(joint, 4) if np.isfinite(joint) else None,
        "gate_passed": bool(gate_passed),
    }


def build_candidate_features_df(frame: pd.DataFrame, cache: dict, scope_mask: np.ndarray,
                                 horizon: int, k: float, hit_type: str) -> pd.DataFrame:
    rows = []
    for side in ("bottom", "top"):
        idx = cache[side]
        idx_scope = idx[scope_mask[idx]]
        if len(idx_scope) == 0:
            continue
        hits = compute_hits(frame, idx_scope, side, horizon, k, hit_type)
        sub = frame.iloc[idx_scope][["timestamp"] + TIER0_FEATURES].copy()
        sub["side"] = side
        sub["hit"] = hits
        rows.append(sub)
    return pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    log("loading CSV...")
    usecols = sorted(set(
        ["timestamp", "high", "low", "close", "atr", "p_fast", "p_slow",
         "bottom_orthogonal_combo", "top_orthogonal_combo"] + TIER0_FEATURES
    ))
    frame = pd.read_csv(CSV_PATH, usecols=usecols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    n_raw = len(frame)
    frame = frame.loc[frame["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"loaded {n_raw} rows; HOLDOUT (>= {HOLDOUT_START.date()}) dropped -> working frame n={len(frame)}, "
        f"range {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")

    ts = frame["timestamp"]
    train_mask = (ts < VAL_START).to_numpy()
    val_mask = ((ts >= VAL_START) & (ts < OOS_START)).to_numpy()
    oos_mask = ((ts >= OOS_START) & (ts < HOLDOUT_START)).to_numpy()
    log(f"TRAIN(<{VAL_START.date()}) n={train_mask.sum()}, VAL n={val_mask.sum()}, OOS n={oos_mask.sum()}")

    log(f"=== building candidate idx: HORIZON in {HORIZON_GRID}, GAP={GAP} fixed dedup, "
        f"2 cutoffs per horizon (H for touch_mfe/close_at_h/touch_mae_capped, 2H for touch_giveback_sustained) ===")
    idx_h_cache: dict[int, dict[str, np.ndarray]] = {}
    idx_2h_cache: dict[int, dict[str, np.ndarray]] = {}
    for horizon in HORIZON_GRID:
        idx_h_cache[horizon] = {
            "bottom": build_candidate_idx(frame, "bottom", "bottom_orthogonal_combo", horizon, GAP),
            "top": build_candidate_idx(frame, "top", "top_orthogonal_combo", horizon, GAP),
        }
        idx_2h_cache[horizon] = {
            "bottom": build_candidate_idx(frame, "bottom", "bottom_orthogonal_combo", 2 * horizon, GAP),
            "top": build_candidate_idx(frame, "top", "top_orthogonal_combo", 2 * horizon, GAP),
        }
        log(f"  H={horizon:>2d}: dedup(cutoff=H) bottom={len(idx_h_cache[horizon]['bottom'])} "
            f"top={len(idx_h_cache[horizon]['top'])}; dedup(cutoff=2H) bottom={len(idx_2h_cache[horizon]['bottom'])} "
            f"top={len(idx_2h_cache[horizon]['top'])}")

    log(f"=== TRAIN grid screen: {len(HIT_TYPES)} HIT_TYPEs x {len(HORIZON_GRID)} HORIZONs x "
        f"{len(K_GRID)} Ks = {len(HIT_TYPES)*len(HORIZON_GRID)*len(K_GRID)} combos ===")
    grid_rows: list[dict] = []
    for hit_type in HIT_TYPES:
        for horizon in HORIZON_GRID:
            for k in K_GRID:
                row = build_grid_row(frame, idx_h_cache, idx_2h_cache, hit_type, horizon, k, train_mask, RNG_SEED)
                grid_rows.append(row)
                log(f"  [{hit_type:<24s}] H={horizon:>2d} K={k:.2f}: "
                    f"bottom(n={row['n_bottom']},hits={row['n_hits_bottom']},lift={row['lift_bottom']}) "
                    f"top(n={row['n_top']},hits={row['n_hits_top']},lift={row['lift_top']}) "
                    f"joint={row['joint_min_lift']} gate={row['gate_passed']}")

    # ---- global selection: argmax(joint_min_lift) among gate-passing cells ----
    eligible = [r for r in grid_rows if r["gate_passed"] and r["joint_min_lift"] is not None]
    global_gate_relaxed = False
    if not eligible:
        global_gate_relaxed = True
        log("WARNING: no cell passed gates across the WHOLE grid; relaxing to any finite joint_min_lift")
        eligible = [r for r in grid_rows if r["joint_min_lift"] is not None]
    assert eligible, "no (HIT_TYPE,H,K) cell produced a finite joint lift at all"
    global_best = max(eligible, key=lambda r: r["joint_min_lift"])
    log(f"\n=== GLOBAL SELECTED: HIT_TYPE={global_best['hit_type']} H={global_best['horizon']} "
        f"K={global_best['k']} (TRAIN joint_min_lift={global_best['joint_min_lift']}, "
        f"gate_relaxed={global_gate_relaxed}) ===")

    # ---- per-HIT_TYPE leaderboard: best own (H,K) within each family, then VAL+OOS at that point ----
    leaderboard: dict[str, dict] = {}
    for hit_type in HIT_TYPES:
        subgrid = [r for r in grid_rows if r["hit_type"] == hit_type]
        gated = [r for r in subgrid if r["gate_passed"] and r["joint_min_lift"] is not None]
        family_gate_relaxed = not gated
        pool = gated if gated else [r for r in subgrid if r["joint_min_lift"] is not None]
        best = max(pool, key=lambda r: r["joint_min_lift"])
        val_row = build_grid_row(frame, idx_h_cache, idx_2h_cache, hit_type, best["horizon"], best["k"],
                                  val_mask, RNG_SEED + 1)
        oos_row = build_grid_row(frame, idx_h_cache, idx_2h_cache, hit_type, best["horizon"], best["k"],
                                  oos_mask, RNG_SEED + 2)
        leaderboard[hit_type] = {
            "chosen_horizon": best["horizon"], "chosen_k": best["k"],
            "family_gate_relaxed": family_gate_relaxed,
            "train": best, "val": val_row, "oos": oos_row,
        }
        log(f"\n-- {hit_type} leaderboard point: H={best['horizon']} K={best['k']} "
            f"(family_gate_relaxed={family_gate_relaxed}) --")
        log(f"   TRAIN joint_min_lift={best['joint_min_lift']} (bottom={best['lift_bottom']}, top={best['lift_top']})")
        log(f"   VAL   joint_min_lift={val_row['joint_min_lift']} (bottom={val_row['lift_bottom']}, top={val_row['lift_top']})")
        log(f"   OOS   joint_min_lift={oos_row['joint_min_lift']} (bottom={oos_row['lift_bottom']}, top={oos_row['lift_top']})")

    global_val = leaderboard[global_best["hit_type"]]["val"]
    global_oos = leaderboard[global_best["hit_type"]]["oos"]

    # ---- OOS degradation comparison across families ----
    oos_degradation = {}
    for hit_type, entry in leaderboard.items():
        t, v, o = entry["train"]["joint_min_lift"], entry["val"]["joint_min_lift"], entry["oos"]["joint_min_lift"]
        oos_degradation[hit_type] = {
            "train_joint_lift": t, "val_joint_lift": v, "oos_joint_lift": o,
            "train_minus_oos": round(t - o, 4) if (t is not None and o is not None) else None,
            "oos_below_1": (o is not None and o < 1.0),
        }
    log("\n=== OOS degradation by HIT_TYPE family (train_joint_lift - oos_joint_lift; lower = more OOS-robust) ===")
    for hit_type, d in oos_degradation.items():
        log(f"  {hit_type:<24s} train={d['train_joint_lift']} val={d['val_joint_lift']} oos={d['oos_joint_lift']} "
            f"degradation={d['train_minus_oos']} oos<1.0={d['oos_below_1']}")

    # ---- feature analysis at the GLOBAL winning point ----
    win_hit_type, win_h, win_k = global_best["hit_type"], global_best["horizon"], global_best["k"]
    log(f"\n=== feature analysis @ HIT_TYPE={win_hit_type} H={win_h} K={win_k} (Tier0 {len(TIER0_FEATURES)} features) ===")
    win_cache = idx_cache_for(win_hit_type, win_h, idx_h_cache, idx_2h_cache)
    train_df = build_candidate_features_df(frame, win_cache, train_mask, win_h, win_k, win_hit_type)
    val_df = build_candidate_features_df(frame, win_cache, val_mask, win_h, win_k, win_hit_type)
    n_train_before_dropna = len(train_df)
    train_df = train_df.dropna(subset=TIER0_FEATURES).reset_index(drop=True)
    n_val_before_dropna = len(val_df)
    val_df = val_df.dropna(subset=TIER0_FEATURES).reset_index(drop=True)
    log(f"TRAIN candidates: {n_train_before_dropna} -> {len(train_df)} after dropna(Tier0 features); "
        f"VAL: {n_val_before_dropna} -> {len(val_df)}")

    corr_rows = []
    for feat in TIER0_FEATURES:
        r, p = pointbiserialr(train_df["hit"].to_numpy(), train_df[feat].to_numpy())
        corr_rows.append({"feature": feat, "point_biserial_r": round(float(r), 4), "p_value": round(float(p), 5)})
    corr_rows.sort(key=lambda r: abs(r["point_biserial_r"]), reverse=True)
    log("-- point-biserial correlation vs hit (TRAIN), ranked by |r| --")
    for r in corr_rows:
        log(f"  {r['feature']:<20s} r={r['point_biserial_r']:+.4f}  p={r['p_value']:.5f}")

    val_auc = float("nan")
    perm_rows: list[dict] = []
    if train_df["hit"].nunique() > 1 and len(val_df) > 0 and val_df["hit"].nunique() > 1:
        clf = HistGradientBoostingClassifier(random_state=RNG_SEED)
        clf.fit(train_df[TIER0_FEATURES], train_df["hit"].to_numpy().astype(int))
        val_proba = clf.predict_proba(val_df[TIER0_FEATURES])[:, 1]
        val_auc = roc_auc_score(val_df["hit"].to_numpy().astype(int), val_proba)
        log(f"HistGradientBoostingClassifier TRAIN-fit -> VAL AUC = {val_auc:.4f} (sanity check only, not a promotion metric)")

        perm = permutation_importance(clf, val_df[TIER0_FEATURES], val_df["hit"].to_numpy().astype(int),
                                       n_repeats=20, random_state=RNG_SEED, scoring="roc_auc")
        perm_rows = [
            {"feature": feat, "importance_mean": round(float(perm.importances_mean[i]), 5),
             "importance_std": round(float(perm.importances_std[i]), 5)}
            for i, feat in enumerate(TIER0_FEATURES)
        ]
        perm_rows.sort(key=lambda r: r["importance_mean"], reverse=True)
        log("-- permutation importance (VAL, HistGBM TRAIN-fit, 20 repeats), ranked --")
        for r in perm_rows:
            log(f"  {r['feature']:<20s} importance={r['importance_mean']:+.5f} (+/-{r['importance_std']:.5f})")
    else:
        log("WARNING: TRAIN or VAL hit label has < 2 classes at the winning point; skipping HistGBM/permutation importance")

    report = {
        "signal": "orthogonal_combo", "asset": "BTC", "round": 2,
        "status": "gridscreen_hittype_and_feature_analysis_only",
        "not_done_this_round": ["TabPFN training", "economic/cost-gate backtest", "holdout exposure"],
        "holdout_touched": False, "holdout_start": str(HOLDOUT_START),
        "round1_reference": {
            "script": "scripts/research_btc_orthogonal_combo_gridscreen_20260901.py",
            "doc": "docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md",
            "hit_type_used": "touch_mfe (only one tested)",
            "selected_horizon": 12, "selected_k": 3.0,
            "train_lift_pooled_both_sides": 1.495, "val_lift_pooled_both_sides": 1.352,
            "oos_lift_pooled_both_sides": 0.929,
            "note": "round 1 used a POOLED both-sides lift (not separate lift_bottom/lift_top), and "
                    "K_GRID/HORIZON_GRID were narrower ([2.5..4.5] x [12..36], no K=2.0 or H=8). "
                    "OOS lift <1.0, reseeded 5x baseline consistently 0.83-0.95, flagged as a real concern.",
        },
        "methodology_assumption": "GAP=12 cluster-dedup embargo (round 1 / ETH orthogonal_combo convention) "
                                   "kept even though this round's task spec does not mention it explicitly -- "
                                   "see script docstring for the reasoning and the raw-vs-deduped count check.",
        "gap_fixed": GAP, "hit_types": HIT_TYPES, "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
        "min_train_candidates": MIN_TRAIN_CANDIDATES, "min_train_hits": MIN_TRAIN_HITS,
        "mae_k_loss_mult": K_LOSS_MULT, "giveback_full_window_mult_of_h": 2, "giveback_ceiling": GIVEBACK_CEIL,
        "selection_rule": "argmax over (HIT_TYPE,H,K) of min(train_lift_bottom, train_lift_top) among "
                           "gate-passing cells (n_cand>=300 AND n_hits>=30, both sides independently)",
        "full_train_grid": grid_rows,
        "global_selection": {
            "hit_type": win_hit_type, "horizon": win_h, "k": win_k, "gate_relaxed": global_gate_relaxed,
            "train": global_best, "val": global_val, "oos": global_oos,
        },
        "per_hittype_leaderboard": leaderboard,
        "oos_degradation_by_hittype": oos_degradation,
        "feature_analysis": {
            "tier0_features": TIER0_FEATURES,
            "n_train_candidates_before_dropna": n_train_before_dropna, "n_train_candidates": len(train_df),
            "n_val_candidates_before_dropna": n_val_before_dropna, "n_val_candidates": len(val_df),
            "point_biserial_correlation_train": corr_rows,
            "histgbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "permutation_importance_val": perm_rows,
        },
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_fresh_forward": "Grid-screen/feature-analysis pass (label separability check across HIT_TYPE "
                               "definitions), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar is "
                               "N/A=False by construction, no trade ledger exists yet.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
