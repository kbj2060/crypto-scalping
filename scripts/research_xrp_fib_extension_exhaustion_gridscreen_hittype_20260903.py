#!/usr/bin/env python3
"""HIT_TYPE x HORIZON x K 3-D grid screen for BTC's `fib_extension_exhaustion` evidence signal --
round 2, redone after round 1 (scripts/research_btc_fib_extension_exhaustion_gridscreen_20260901.py)
used a single fixed hit definition (touch-based MFE) and produced this project's weakest BTC
signal so far (VAL AUC 0.57-0.60, joint TRAIN lift ~1.34x on a thin ~600-candidate/side sample).
The user asked a pointed question: why assume touch-based-MFE is the right HIT definition at all --
shouldn't HIT_TYPE itself be grid-searched per signal, not just H and K? This script adds HIT_TYPE
as a third grid axis to answer that directly: is this signal weak regardless of hit definition, or
was round 1's fixed choice hiding a better result?

Data: data/labels/xrp_5m_evidence_signal_candidates_20260903/btc_5m_evidence_signal_candidates_
tier0.csv (277,191 rows, 2024-01-01 to 2026-08-20, BTCUSDT 5m). `bottom_fib_extension_exhaustion`/
`top_fib_extension_exhaustion` triggers and all Tier0 features are read as-is, NOT recomputed here
(same convention as round 1).

Four HIT_TYPE families (all use entry=close[i], atr=atr[i] absolute-price ATR14, candidate at row i):

  1. touch_mfe (round 1's method, kept as baseline for comparison):
       bottom: hit=1 if high[i+1:i+H+1].max() >= entry + K*atr
       top:    hit=1 if low[i+1:i+H+1].min()  <= entry - K*atr

  2. close_at_h (stricter -- only the bar-H close counts, no credit for touch-then-revert):
       bottom: hit=1 if close[i+H] >= entry + K*atr
       top:    hit=1 if close[i+H] <= entry - K*atr

  3. touch_mae_capped (touch_mfe, disqualified if price first went too far against the position
     before reaching target -- ORDER-AWARE: MAE is measured only up to the first touch bar, not
     over the whole window; this differs from round 1's bonus mae_cap_bonus_check(), which used a
     whole-window, order-blind MAE -- so results here are expected to ROUGHLY match round 1's bonus
     check, not exactly, per this task's own framing "use as a cross-check"):
       K_LOSS_MULT = 2.0 (fixed, matches round 1's bonus-check value)
       bottom: touch_bar = first bar in [i+1,i+H] with high>=entry+K*atr (none -> not a hit);
               MAE = entry - low[i+1:touch_bar+1].min(); hit = touch found AND MAE <= K_LOSS_MULT*atr
       top: mirror (MAE = high[i+1:touch_bar+1].max() - entry)

  4. touch_giveback_sustained (V_REBOUND-style persistence check, used here only as a CANDIDATE hit
     definition for fib_extension_exhaustion, not literally the V_REBOUND label):
       FAST_WINDOW = H, FULL_WINDOW = 2*H (fixed multiple, not swept separately)
       giveback ceiling = 0.20 (fixed, matches this project's V_REBOUND convention)
       bottom: fast_move = close[i+1:i+FAST_WINDOW+1].max() - entry; fast_mult = fast_move/atr;
               peak = high[i+1:i+FULL_WINDOW+1].max(); end_price = close[i+FULL_WINDOW];
               denom = peak - entry; giveback = (peak-end_price)/denom (NaN-safe if denom<=0);
               hit = fast_mult >= K AND giveback <= 0.20
       top: mirror

Grid: HIT_TYPE(4) x HORIZON in [10,16,20,24,30] x K in [1.5,2.0,2.35,2.75,3.25] x side(2) = 200
cells. Given this signal's already-thin round-1 sample (~600 candidates/side), the frequency gate
is RELAXED vs round 1 (300/30 -> 150/15 per side) per this task's explicit instruction -- every
reported number still carries its raw sample size so thin cells are never silently over-read.

Split (Fresh-Forward, matches round 1 and this repo's contract): TRAIN <2025-09-01, VAL 2025-09-01
to 2026-01-01, OOS 2026-01-01 to 2026-04-01 (bonus check this round, round 1 left it unscored),
HOLDOUT >=2026-04-01 (dropped at load time, never read). Selection is TRAIN-only (argmax over the
full 3-D grid of min(lift_bottom,lift_top) among gate-passing cells); VAL/OOS confirm the ONE chosen
point, no re-search.

Run with: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_fib_extension_exhaustion_gridscreen_hittype_20260901.py

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/fib_extension_exhaustion_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")   # never touched past this point

HORIZON_GRID = [10, 16, 20, 24, 30]
K_GRID = [1.5, 2.0, 2.35, 2.75, 3.25]
HIT_TYPES = ["touch_mfe", "close_at_h", "touch_mae_capped", "touch_giveback_sustained"]

MAE_K_LOSS_MULT = 2.0     # touch_mae_capped, fixed per task spec
GIVEBACK_CEIL = 0.20      # touch_giveback_sustained, fixed per task spec (V_REBOUND convention)
FULL_WINDOW_MULT = 2      # touch_giveback_sustained FULL_WINDOW = 2*H, fixed per task spec

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

MIN_TRAIN_CANDIDATES = 150  # per side -- RELAXED vs round 1's 300 (task step: thin-sample signal)
MIN_TRAIN_HITS = 15         # per side -- RELAXED vs round 1's 30
MIN_VAL_FOR_PERMUTATION = 30
MIN_VAL_MINORITY_CLASS = 10
MIN_OOS_CANDIDATES_FOR_RATIO = 20  # below this, report raw counts only, no lift ratio (task step 1)
RNG_SEED = 20260901
Z_95 = 1.959963984540054


def log(msg: str) -> None:
    print(f"[xrp_fib_ext_hittype_gridscreen] {msg}", flush=True)


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_data() -> pd.DataFrame:
    usecols = sorted(set(
        ["timestamp", "high", "low", "close", "atr",
         "bottom_fib_extension_exhaustion", "top_fib_extension_exhaustion"] + TIER0_FEATURES
    ))
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    assert df["timestamp"].max() < HOLDOUT_START, "HOLDOUT row leaked past truncation"
    return df


# ---------------------------------------------------------------------------
# HIT_TYPE implementations. Each family exposes compute_hit_dict(...) which
# returns {K: bool_array} for a given (idx, horizon, side) -- pool eligibility
# (bound-checked against the array length so a forward window never reaches
# into un-loaded HOLDOUT rows) is handled by the caller (get_pool below), not
# here.
# ---------------------------------------------------------------------------

def bound_for(hit_type: str, horizon: int) -> int:
    """Bars beyond i that must exist in the loaded (HOLDOUT-truncated) frame."""
    if hit_type == "touch_giveback_sustained":
        return FULL_WINDOW_MULT * horizon
    return horizon


def fwd_extreme(pos_idx: np.ndarray, arr: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """Extreme of arr over (i+1..i+horizon] for each i in pos_idx. Caller guarantees i+horizon<n."""
    out = np.empty(len(pos_idx), dtype=float)
    for k, i in enumerate(pos_idx):
        window = arr[i + 1:i + horizon + 1]
        out[k] = window.max() if mode == "max" else window.min()
    return out


def hit_from_ext(ext: np.ndarray, idx: np.ndarray, close: np.ndarray, atr: np.ndarray, k: float, side: str) -> np.ndarray:
    entry = close[idx]
    a = atr[idx]
    if side == "bottom":
        return (ext - entry) >= k * a
    return (entry - ext) >= k * a


def hit_dict_touch_mfe(idx: np.ndarray, horizon: int, side: str, high, low, close, atr) -> dict[float, np.ndarray]:
    ext = fwd_extreme(idx, high if side == "bottom" else low, horizon, "max" if side == "bottom" else "min")
    return {k: hit_from_ext(ext, idx, close, atr, k, side) for k in K_GRID}


def hit_dict_close_at_h(idx: np.ndarray, horizon: int, side: str, high, low, close, atr) -> dict[float, np.ndarray]:
    ext = close[idx + horizon]  # caller guarantees idx+horizon < n
    return {k: hit_from_ext(ext, idx, close, atr, k, side) for k in K_GRID}


def hit_dict_touch_mae_capped(idx: np.ndarray, horizon: int, side: str, high, low, close, atr) -> dict[float, np.ndarray]:
    """Order-aware: for each K, single forward pass per candidate tracking the running adverse
    extreme; MAE is whatever that running extreme is AT the first bar the touch condition fires
    (i.e. over [i+1, touch_bar] inclusive), not the whole [i+1,i+H] window."""
    out = {k: np.zeros(len(idx), dtype=bool) for k in K_GRID}
    entries = close[idx]
    atrs = atr[idx]
    for n_i, i in enumerate(idx):
        entry = entries[n_i]
        a = atrs[n_i]
        hi_win = high[i + 1:i + horizon + 1]
        lo_win = low[i + 1:i + horizon + 1]
        for k in K_GRID:
            running_adverse = entry
            touched = False
            if side == "bottom":
                thresh = entry + k * a
                for h, l in zip(hi_win, lo_win):
                    if l < running_adverse:
                        running_adverse = l
                    if h >= thresh:
                        touched = True
                        break
                if touched:
                    mae = entry - running_adverse
                    out[k][n_i] = mae <= MAE_K_LOSS_MULT * a
            else:
                thresh = entry - k * a
                for h, l in zip(hi_win, lo_win):
                    if h > running_adverse:
                        running_adverse = h
                    if l <= thresh:
                        touched = True
                        break
                if touched:
                    mae = running_adverse - entry
                    out[k][n_i] = mae <= MAE_K_LOSS_MULT * a
    return out


def hit_dict_touch_giveback_sustained(idx: np.ndarray, horizon: int, side: str, high, low, close, atr) -> dict[float, np.ndarray]:
    full = FULL_WINDOW_MULT * horizon
    fast_mult = np.full(len(idx), np.nan)
    giveback = np.full(len(idx), np.nan)
    entries = close[idx]
    atrs = atr[idx]
    for n_i, i in enumerate(idx):
        entry = entries[n_i]
        a = atrs[n_i]
        if side == "bottom":
            fast_move = close[i + 1:i + horizon + 1].max() - entry
            peak = high[i + 1:i + full + 1].max()
            end_price = close[i + full]
            denom = peak - entry
            gb = (peak - end_price) / denom if denom > 0 else np.nan
        else:
            fast_move = entry - close[i + 1:i + horizon + 1].min()
            peak = low[i + 1:i + full + 1].min()
            end_price = close[i + full]
            denom = entry - peak
            gb = (end_price - peak) / denom if denom > 0 else np.nan
        fast_mult[n_i] = fast_move / a
        giveback[n_i] = gb
    with np.errstate(invalid="ignore"):
        return {k: (fast_mult >= k) & (giveback <= GIVEBACK_CEIL) for k in K_GRID}


HIT_DICT_FN = {
    "touch_mfe": hit_dict_touch_mfe,
    "close_at_h": hit_dict_close_at_h,
    "touch_mae_capped": hit_dict_touch_mae_capped,
    "touch_giveback_sustained": hit_dict_touch_giveback_sustained,
}


def compute_hit_dict(hit_type: str, idx: np.ndarray, horizon: int, side: str, high, low, close, atr) -> dict[float, np.ndarray]:
    if len(idx) == 0:
        return {k: np.array([], dtype=bool) for k in K_GRID}
    return HIT_DICT_FN[hit_type](idx, horizon, side, high, low, close, atr)


def get_pool(hit_type: str, horizon: int, split_mask: np.ndarray, trig_this_side: np.ndarray,
             any_trig: np.ndarray, atr: np.ndarray, close: np.ndarray, n: int,
             rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    bound = bound_for(hit_type, horizon)
    elig = split_mask & ~np.isnan(atr) & (atr > 0) & ~np.isnan(close)
    cand_pool = np.flatnonzero(elig & trig_this_side)
    cand_pool = cand_pool[cand_pool + bound < n]
    noncand_pool = np.flatnonzero(elig & ~any_trig)
    noncand_pool = noncand_pool[noncand_pool + bound < n]
    n_base = min(len(cand_pool), len(noncand_pool))
    base_idx = rng.choice(noncand_pool, size=n_base, replace=False) if n_base > 0 else np.array([], dtype=int)
    return cand_pool, base_idx


def build_grid_rows(hit_type: str, side: str, horizon: int, cand_idx: np.ndarray, base_idx: np.ndarray,
                     cand_hits: dict[float, np.ndarray], base_hits: dict[float, np.ndarray]) -> list[dict]:
    rows = []
    n_cand = len(cand_idx)
    n_base = len(base_idx)
    for k in K_GRID:
        ch = cand_hits[k]
        bh = base_hits[k]
        n_cand_hits = int(ch.sum()) if n_cand else 0
        n_base_hits = int(bh.sum()) if n_base else 0
        cand_rate = float(ch.mean()) if n_cand else float("nan")
        base_rate = float(bh.mean()) if n_base else float("nan")
        lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")
        ci_lo, ci_hi = wilson_ci(n_cand_hits, n_cand)
        rows.append({
            "hit_type": hit_type, "side": side, "horizon": horizon, "k": k,
            "n_cand": n_cand, "n_cand_hits": n_cand_hits,
            "n_base": n_base, "n_base_hits": n_base_hits,
            "cand_hit_rate": round(cand_rate, 4) if np.isfinite(cand_rate) else None,
            "cand_hit_rate_ci_lo": round(ci_lo, 4) if np.isfinite(ci_lo) else None,
            "cand_hit_rate_ci_hi": round(ci_hi, 4) if np.isfinite(ci_hi) else None,
            "base_hit_rate": round(base_rate, 4) if np.isfinite(base_rate) else None,
            "lift": round(lift, 4) if np.isfinite(lift) else None,
        })
    return rows


def train_cv_permutation_importance(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str],
                                     seed: int, n_splits: int = 5) -> tuple[dict, dict]:
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_feat = {c: [] for c in feature_cols}
    for tr_idx, te_idx in kf.split(X):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X.iloc[tr_idx], y_tr)
        perm = permutation_importance(clf, X.iloc[te_idx], y_te, scoring="roc_auc", n_repeats=15, random_state=seed)
        for i, c in enumerate(feature_cols):
            per_feat[c].append(perm.importances_mean[i])
    mean_out = {c: float(np.mean(v)) for c, v in per_feat.items() if v}
    std_out = {c: float(np.std(v)) for c, v in per_feat.items() if v}
    return mean_out, std_out


def main() -> int:
    log("loading XRP Tier0 candidate CSV...")
    df = load_data()
    n = len(df)
    log(f"{n} rows loaded, {df['timestamp'].min()} -> {df['timestamp'].max()} (HOLDOUT never loaded)")

    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    close = df["close"].to_numpy(dtype=float)
    atr = df["atr"].to_numpy(dtype=float)

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] < HOLDOUT_START)).to_numpy()
    log(f"TRAIN rows={train_mask.sum()} VAL rows={val_mask.sum()} OOS rows={oos_mask.sum()}")

    any_trig = df["bottom_fib_extension_exhaustion"].fillna(False).to_numpy() | df["top_fib_extension_exhaustion"].fillna(False).to_numpy()
    log(f"raw trigger fires (pre-eligibility, whole loaded frame): "
        f"bottom={int(df['bottom_fib_extension_exhaustion'].sum())} top={int(df['top_fib_extension_exhaustion'].sum())}")

    rng = np.random.default_rng(RNG_SEED)

    # ---- TRAIN 3-D grid screen: HIT_TYPE x side x HORIZON x K ----
    grid_rows: list[dict] = []
    train_pools: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]] = {}
    for hit_type in HIT_TYPES:
        for side in ("bottom", "top"):
            trig_this_side = df[f"{side}_fib_extension_exhaustion"].fillna(False).to_numpy()
            for horizon in HORIZON_GRID:
                cand_idx, base_idx = get_pool(hit_type, horizon, train_mask, trig_this_side, any_trig, atr, close, n, rng)
                train_pools[(hit_type, side, horizon)] = (cand_idx, base_idx)
                cand_hits = compute_hit_dict(hit_type, cand_idx, horizon, side, high, low, close, atr)
                base_hits = compute_hit_dict(hit_type, base_idx, horizon, side, high, low, close, atr)
                rows = build_grid_rows(hit_type, side, horizon, cand_idx, base_idx, cand_hits, base_hits)
                grid_rows.extend(rows)
                best_k_row = max(rows, key=lambda r: (r["lift"] if r["lift"] is not None else -1))
                log(f"  TRAIN hit_type={hit_type:24s} side={side:6s} H={horizon:>3d}: n_cand={len(cand_idx):>5d} "
                    f"best_lift={best_k_row['lift']} @K={best_k_row['k']}")

    grid_df = pd.DataFrame(grid_rows)

    # ---- selection: argmax over (HIT_TYPE,H,K) of min(lift_bottom,lift_top), gated on
    # n_cand>=150 and n_hits>=15 BOTH sides (relaxed vs round 1's 300/30 per task instruction) ----
    pivot_bottom = grid_df[grid_df["side"] == "bottom"].set_index(["hit_type", "horizon", "k"])
    pivot_top = grid_df[grid_df["side"] == "top"].set_index(["hit_type", "horizon", "k"])

    all_cells = []
    for hit_type in HIT_TYPES:
        for horizon in HORIZON_GRID:
            for k in K_GRID:
                b = pivot_bottom.loc[(hit_type, horizon, k)]
                t = pivot_top.loc[(hit_type, horizon, k)]
                if b["lift"] is None or t["lift"] is None:
                    continue
                gate_pass = (b["n_cand"] >= MIN_TRAIN_CANDIDATES and t["n_cand"] >= MIN_TRAIN_CANDIDATES
                             and b["n_cand_hits"] >= MIN_TRAIN_HITS and t["n_cand_hits"] >= MIN_TRAIN_HITS)
                joint = min(b["lift"], t["lift"])
                all_cells.append({
                    "hit_type": hit_type, "horizon": horizon, "k": k,
                    "lift_bottom": b["lift"], "lift_top": t["lift"], "joint_min": joint,
                    "n_cand_bottom": int(b["n_cand"]), "n_cand_top": int(t["n_cand"]),
                    "n_hits_bottom": int(b["n_cand_hits"]), "n_hits_top": int(t["n_cand_hits"]),
                    "gate_pass": bool(gate_pass),
                })

    gate_passing = [c for c in all_cells if c["gate_pass"]]
    gate_relaxed = False
    pool_for_choice = gate_passing
    if not pool_for_choice:
        gate_relaxed = True
        log(f"WARNING: no (HIT_TYPE,H,K) combo passed gates (n_cand>={MIN_TRAIN_CANDIDATES}, "
            f"n_hits>={MIN_TRAIN_HITS} both sides); relaxing to ungated full grid")
        pool_for_choice = all_cells

    assert pool_for_choice, "no (HIT_TYPE,H,K) combo produced a finite joint lift at all"
    pool_for_choice_sorted = sorted(pool_for_choice, key=lambda c: c["joint_min"], reverse=True)
    chosen = pool_for_choice_sorted[0]
    CHOSEN_HIT_TYPE = chosen["hit_type"]
    CHOSEN_H = chosen["horizon"]
    CHOSEN_K = chosen["k"]
    log(f"\n=== GLOBAL CHOSEN: HIT_TYPE={CHOSEN_HIT_TYPE} HORIZON={CHOSEN_H} K={CHOSEN_K}: "
        f"TRAIN lift bottom={chosen['lift_bottom']} top={chosen['lift_top']} joint(min)={chosen['joint_min']} "
        f"(gate_relaxed={gate_relaxed}) ===")

    log("\nTop 10 (HIT_TYPE,H,K) combos overall by joint(min) TRAIN lift (gate-passing pool):")
    for c in pool_for_choice_sorted[:10]:
        log(f"  {c['hit_type']:24s} H={c['horizon']:>3d} K={c['k']:.2f}: bottom={c['lift_bottom']:.3f} "
            f"top={c['lift_top']:.3f} joint={c['joint_min']:.3f} (n_cand b/t={c['n_cand_bottom']}/{c['n_cand_top']})")

    # ---- per-HIT_TYPE leaderboard: best (and top-3) cell within each family ----
    per_hit_type_leaderboard = {}
    for hit_type in HIT_TYPES:
        family_gate = [c for c in gate_passing if c["hit_type"] == hit_type]
        family_all = [c for c in all_cells if c["hit_type"] == hit_type]
        family_relaxed = False
        family_pool = family_gate
        if not family_pool:
            family_relaxed = True
            family_pool = family_all
        family_sorted = sorted(family_pool, key=lambda c: c["joint_min"], reverse=True)
        per_hit_type_leaderboard[hit_type] = {
            "gate_relaxed_within_family": family_relaxed,
            "n_gate_passing_cells": len(family_gate),
            "n_total_cells": len(family_all),
            "top3": family_sorted[:3],
        }
        best = family_sorted[0] if family_sorted else None
        if best:
            log(f"  FAMILY BEST {hit_type:24s}: H={best['horizon']} K={best['k']} "
                f"joint={best['joint_min']:.3f} (bottom={best['lift_bottom']:.3f} top={best['lift_top']:.3f}, "
                f"n_cand b/t={best['n_cand_bottom']}/{best['n_cand_top']}, family_gate_relaxed={family_relaxed})")

    # ---- VAL confirmation at GLOBAL chosen (HIT_TYPE,H,K) only (no re-search) ----
    val_rows = []
    val_pools = {}
    val_thin_flags = {}
    for side in ("bottom", "top"):
        trig_this_side = df[f"{side}_fib_extension_exhaustion"].fillna(False).to_numpy()
        cand_idx, base_idx = get_pool(CHOSEN_HIT_TYPE, CHOSEN_H, val_mask, trig_this_side, any_trig, atr, close, n, rng)
        val_pools[side] = (cand_idx, base_idx)
        cand_hits = compute_hit_dict(CHOSEN_HIT_TYPE, cand_idx, CHOSEN_H, side, high, low, close, atr)
        base_hits = compute_hit_dict(CHOSEN_HIT_TYPE, base_idx, CHOSEN_H, side, high, low, close, atr)
        rows = build_grid_rows(CHOSEN_HIT_TYPE, side, CHOSEN_H, cand_idx, base_idx, cand_hits, base_hits)
        row_at_k = next(r for r in rows if r["k"] == CHOSEN_K)
        val_rows.append(row_at_k)
        val_thin_flags[side] = row_at_k["n_cand"] < 30
        log(f"  VAL side={side:6s} hit_type={CHOSEN_HIT_TYPE} H={CHOSEN_H} K={CHOSEN_K}: n_cand={row_at_k['n_cand']} "
            f"lift={row_at_k['lift']} cand_hit_rate={row_at_k['cand_hit_rate']} base_hit_rate={row_at_k['base_hit_rate']}"
            f"{'  *** THIN VAL (<30) ***' if val_thin_flags[side] else ''}")

    # ---- OOS bonus check at GLOBAL chosen (HIT_TYPE,H,K), same rule (no re-search) ----
    oos_rows = []
    oos_thin_flags = {}
    for side in ("bottom", "top"):
        trig_this_side = df[f"{side}_fib_extension_exhaustion"].fillna(False).to_numpy()
        cand_idx, base_idx = get_pool(CHOSEN_HIT_TYPE, CHOSEN_H, oos_mask, trig_this_side, any_trig, atr, close, n, rng)
        cand_hits = compute_hit_dict(CHOSEN_HIT_TYPE, cand_idx, CHOSEN_H, side, high, low, close, atr)
        base_hits = compute_hit_dict(CHOSEN_HIT_TYPE, base_idx, CHOSEN_H, side, high, low, close, atr)
        rows = build_grid_rows(CHOSEN_HIT_TYPE, side, CHOSEN_H, cand_idx, base_idx, cand_hits, base_hits)
        row_at_k = next(r for r in rows if r["k"] == CHOSEN_K)
        oos_thin_flags[side] = row_at_k["n_cand"] < MIN_OOS_CANDIDATES_FOR_RATIO
        if oos_thin_flags[side]:
            row_at_k = dict(row_at_k)
            row_at_k["lift_suppressed_too_thin"] = True
            note = f"n_cand={row_at_k['n_cand']} < {MIN_OOS_CANDIDATES_FOR_RATIO}: lift ratio not meaningful, reporting raw counts only"
            row_at_k["thin_note"] = note
            log(f"  OOS side={side:6s}: {note}")
        else:
            log(f"  OOS side={side:6s} hit_type={CHOSEN_HIT_TYPE} H={CHOSEN_H} K={CHOSEN_K}: n_cand={row_at_k['n_cand']} "
                f"lift={row_at_k['lift']} cand_hit_rate={row_at_k['cand_hit_rate']} base_hit_rate={row_at_k['base_hit_rate']}")
        oos_rows.append(row_at_k)

    train_rows_chosen = [r for r in grid_rows if r["hit_type"] == CHOSEN_HIT_TYPE and r["horizon"] == CHOSEN_H and r["k"] == CHOSEN_K]

    # ---- feature analysis at GLOBAL chosen (HIT_TYPE,H,K): TRAIN candidates -> hit label ----
    feature_analysis = {}
    for side in ("bottom", "top"):
        cand_idx, _base_idx = train_pools[(CHOSEN_HIT_TYPE, side, CHOSEN_H)]
        hit = compute_hit_dict(CHOSEN_HIT_TYPE, cand_idx, CHOSEN_H, side, high, low, close, atr)[CHOSEN_K].astype(int)
        feat_df = df.loc[cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        feat_df["hit"] = hit

        corr = feat_df.corr(numeric_only=True)["hit"].drop("hit").sort_values(key=lambda s: s.abs(), ascending=False)
        mean_hit1 = feat_df.loc[feat_df["hit"] == 1, TIER0_FEATURES].mean()
        mean_hit0 = feat_df.loc[feat_df["hit"] == 0, TIER0_FEATURES].mean()

        clf = HistGradientBoostingClassifier(random_state=RNG_SEED)
        X_train = feat_df[TIER0_FEATURES]
        y_train = feat_df["hit"].to_numpy()
        clf.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1]) if len(np.unique(y_train)) > 1 else float("nan")

        val_cand_idx, _val_base_idx = val_pools[side]
        val_hit = compute_hit_dict(CHOSEN_HIT_TYPE, val_cand_idx, CHOSEN_H, side, high, low, close, atr)[CHOSEN_K].astype(int)
        val_feat_df = df.loc[val_cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        X_val = val_feat_df[TIER0_FEATURES]
        y_val = val_hit
        minority_val = min((y_val == 0).sum(), (y_val == 1).sum()) if len(y_val) else 0
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")

        use_val_for_importance = len(y_val) >= MIN_VAL_FOR_PERMUTATION and minority_val >= MIN_VAL_MINORITY_CLASS
        if use_val_for_importance:
            perm = permutation_importance(clf, X_val, y_val, scoring="roc_auc", n_repeats=30, random_state=RNG_SEED)
            perm_mean = dict(zip(TIER0_FEATURES, perm.importances_mean))
            perm_std = dict(zip(TIER0_FEATURES, perm.importances_std))
            importance_method = "VAL"
        else:
            log(f"  side={side}: VAL too thin for permutation importance (n={len(y_val)}, minority={minority_val}) "
                f"-> falling back to TRAIN 5-fold CV permutation importance")
            perm_mean, perm_std = train_cv_permutation_importance(X_train, y_train, TIER0_FEATURES, RNG_SEED)
            importance_method = "TRAIN_5FOLD_CV"

        perm_series = pd.Series(perm_mean).sort_values(key=np.abs, ascending=False)

        log(f"\n=== Feature analysis side={side} hit_type={CHOSEN_HIT_TYPE} H={CHOSEN_H} K={CHOSEN_K} "
            f"n_train_cand={len(cand_idx)} n_val_cand={len(val_cand_idx)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} importance_method={importance_method} ===")
        log("  top |corr| (TRAIN, desc): " + ", ".join(f"{f}={corr[f]:+.3f}" for f in corr.index[:8]))
        log("  top |perm-importance| (desc): " + ", ".join(
            f"{f}={perm_series[f]:+.4f}(+-{perm_std.get(f, float('nan')):.4f})" for f in perm_series.index[:8]))

        feature_analysis[side] = {
            "n_train_candidates": int(len(cand_idx)),
            "n_val_candidates": int(len(val_cand_idx)),
            "train_hit_rate": round(float(y_train.mean()), 4) if len(y_train) else None,
            "val_hit_rate": round(float(np.mean(y_val)), 4) if len(y_val) else None,
            "val_minority_class_n": int(minority_val),
            "gbm_train_auc": round(float(train_auc), 4) if np.isfinite(train_auc) else None,
            "gbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "permutation_importance_method": importance_method,
            "point_biserial_corr_train": {f: round(float(corr[f]), 4) for f in corr.index},
            "mean_feature_hit1_train": {f: round(float(mean_hit1[f]), 4) for f in TIER0_FEATURES},
            "mean_feature_hit0_train": {f: round(float(mean_hit0[f]), 4) for f in TIER0_FEATURES},
            "permutation_importance_mean": {f: round(float(perm_series[f]), 5) for f in perm_series.index},
            "permutation_importance_std": {f: round(float(perm_std.get(f, float("nan"))), 5) for f in perm_series.index},
        }

    report = {
        "asset": "BTCUSDT", "signal": "fib_extension_exhaustion", "bar": "5m",
        "round": 2, "round_description": "HIT_TYPE x HORIZON x K 3-D grid (round 1 fixed HIT_TYPE=touch_mfe only)",
        "prior_round_script": "scripts/research_btc_fib_extension_exhaustion_gridscreen_20260901.py",
        "prior_round_report": "docs/experiments/btc_5m_fib_extension_exhaustion_gridscreen_featureanalysis_20260901.md (pre-overwrite content)",
        "data_path": str(DATA_PATH),
        "rows_loaded": int(n),
        "date_range_used": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "holdout_start_never_touched": str(HOLDOUT_START),
        "split": {"train_end_excl": str(VAL_START), "val_start_incl": str(VAL_START), "val_end_excl": str(OOS_START),
                  "oos_start_incl": str(OOS_START), "oos_end_excl": str(HOLDOUT_START)},
        "horizon_grid": HORIZON_GRID, "k_grid": K_GRID, "hit_types": HIT_TYPES,
        "hit_type_formulas": {
            "touch_mfe": "bottom: high[i+1:i+H+1].max()>=entry+K*atr; top: low[i+1:i+H+1].min()<=entry-K*atr",
            "close_at_h": "bottom: close[i+H]>=entry+K*atr; top: close[i+H]<=entry-K*atr",
            "touch_mae_capped": f"touch_mfe AND order-aware MAE (up to first touch bar) <= {MAE_K_LOSS_MULT}*K_loss_mult*atr (K_LOSS_MULT={MAE_K_LOSS_MULT} fixed)",
            "touch_giveback_sustained": f"fast_mult=(close-based fast move over FAST_WINDOW=H)/atr >= K AND giveback (peak-to-end retracement over FULL_WINDOW=2H) <= {GIVEBACK_CEIL} fixed",
        },
        "selection_rule": (f"argmax over (HIT_TYPE,H,K) of min(train_lift_bottom, train_lift_top), gated on "
                            f"n_cand>={MIN_TRAIN_CANDIDATES} and n_hits>={MIN_TRAIN_HITS} both sides "
                            f"(RELAXED vs round 1's 300/30 gate per task instruction -- this signal fires sparsely)"),
        "selection_gate_relaxed_globally": gate_relaxed,
        "chosen_hit_type": CHOSEN_HIT_TYPE, "chosen_horizon": CHOSEN_H, "chosen_k": CHOSEN_K,
        "chosen_train_lift": {"bottom": chosen["lift_bottom"], "top": chosen["lift_top"], "joint_min": chosen["joint_min"]},
        "chosen_val_confirmation": val_rows,
        "val_thin_sample_flag": val_thin_flags,
        "chosen_oos_bonus_check": oos_rows,
        "oos_thin_sample_flag": oos_thin_flags,
        "oos_min_candidates_for_ratio": MIN_OOS_CANDIDATES_FOR_RATIO,
        "chosen_train_rows": train_rows_chosen,
        "top10_combos_overall_by_joint_lift": pool_for_choice_sorted[:10],
        "per_hit_type_leaderboard": per_hit_type_leaderboard,
        "full_train_grid": grid_rows,
        "feature_analysis": feature_analysis,
        "tier0_features": TIER0_FEATURES,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_fresh_forward": "This is a grid-screen/feature-analysis pass (label separability check), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar is N/A=False by construction, no trade ledger exists yet.",
        "cross_asset_info_used": False,
        "cross_asset_note": "fib_extension_exhaustion is a single-asset (BTC-only OHLC) leg/zone signal by definition, no BTC-ETH cross-asset info used.",
        "tabpfn_training_done": False,
        "economic_cost_gate_done": False,
        "holdout_exposure_done": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
