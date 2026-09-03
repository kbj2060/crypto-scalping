#!/usr/bin/env python3
"""BTC grid-screen for short_term_return_z, ROUND 2: HIT_TYPE as a third search dimension.

User's question that motivated this round (2026-09-01): round 1
(research_btc_short_term_return_z_gridscreen_20260901.py) grid-searched HORIZON x K using ONE
fixed hit definition (pure touch-based MFE) and found an unresolved tradeoff -- the mechanical
selection (HORIZON=2, K=1.75) had strong TRAIN/VAL lift (2.42x/2.29x) but was fragile (OOS drifted
to 1.47x, small hit counts), while HORIZON=6 was much flatter/more stable (TRAIN/VAL/OOS all
1.50-1.56x) with far larger sample -- left as a human decision. The user then asked: why assume
touch-based-MFE is the right HIT definition at all? Shouldn't the HIT DEFINITION ITSELF be
grid-searched too, not just HORIZON and K? This script adds HIT_TYPE as a third grid dimension and
specifically checks whether a different HIT_TYPE resolves the H2-vs-H6 tradeoff more cleanly.

Four HIT_TYPE families (candidate at row i, entry=close[i], atr=atr[i], is_down=bottom-side
candidate; all touch/close conditions use intrabar high/low except where noted):

  1. touch_mfe (round 1's method, kept as baseline for direct comparison):
     bottom: hit=1 if high[i+1:i+H+1].max() >= entry + K*atr
     top:    hit=1 if low[i+1:i+H+1].min()  <= entry - K*atr

  2. close_at_h (stricter -- only the bar-H CLOSE counts, no credit for touch-then-revert):
     bottom: hit=1 if close[i+H] >= entry + K*atr
     top:    hit=1 if close[i+H] <= entry - K*atr

  3. touch_mae_capped (touch_mfe, but disqualified if price went too far against the position
     FIRST -- i.e. before the profit target was touched). K_LOSS_MULT=2.0 fixed (matches this
     project's fib_extension_exhaustion MAE-cap convention's constant -- NOT its k*K_LOSS_MULT
     scaling; here the cap is a flat K_LOSS_MULT*atr regardless of K, per this task's spec):
     bottom: touch_bar = first bar in [i+1,i+H] where high>=entry+K*atr (no touch -> not a hit);
             MAE = entry - low[i+1:touch_bar+1].min(); hit=1 if touched AND MAE<=K_LOSS_MULT*atr
     top:    mirror

  4. touch_giveback_sustained (V_REBOUND-style persistence check, ported as a candidate HIT_TYPE
     for THIS signal -- not literally V_REBOUND itself). FAST_WINDOW=H, FULL_WINDOW=2*H (fixed
     multiple of H, not swept separately). giveback ceiling = 0.20 fixed (matches this project's
     V_REBOUND convention):
     bottom: fast_move=close[i+1:i+H+1].max()-entry; fast_mult=fast_move/atr;
             peak=high[i+1:i+2H+1].max(); end_price=close[i+2H]; denom=peak-entry;
             giveback=(peak-end_price)/denom (NaN-safe if denom~0);
             hit=1 if fast_mult>=K AND giveback<=0.20
     top:    mirror (fast_move=entry-close.min(), peak=low.min(), denom=entry-peak,
             giveback=(end_price-peak)/denom)

Grid: HIT_TYPE in the 4 above x HORIZON in [2,3,6,9,12,18] x K in [1.0,1.5,1.75,2.0,2.5] = 120
cells. HORIZON keeps 2 (round 1's mechanical pick) and effectively brackets 6 (round 1's stability
recommendation) so this round can be compared directly against round 1's numbers.

Candidate pool: same cluster-anchor dedup as round 1 (GAP=12 bars, fixed project-wide convention,
same-side fires within GAP collapsed to the single most-extreme-ret3_z bar, causal). Because
touch_giveback_sustained needs 2*HORIZON forward bars (vs HORIZON for the other three), dedup is
computed per REQUIRED FORWARD WINDOW (cached by window size, not by nominal horizon) -- a horizon=3
touch_giveback_sustained cell and a horizon=6 touch_mfe/close_at_h/touch_mae_capped cell both need a
6-bar-forward-room candidate pool and correctly share one dedup computation.

Baseline: same-count random non-trigger bars from the SAME period as the candidates being compared
(TRAIN baseline from TRAIN-period eligible bars, etc.), same direction-hit rule as the matching
side, fixed seed per (window_needed, period, side). Reused across all HIT_TYPEs/K values that share
the same window_needed (the baseline BAR SET doesn't depend on K or on which of the 3 HIT_TYPEs in
the H-window group is being evaluated -- only the hit RULE applied to those bars does).

Selection methodology (UPGRADED from round 1): round 1 selected on the POOLED (bottom+top combined)
TRAIN lift. This round selects on min(train_lift_bottom, train_lift_top) instead -- the weaker of
the two sides -- per explicit task instruction, so a cell can't win by having one strong side mask a
weak/negative other side. Gates (TRAIN only, evaluated per side): n_cand>=300 AND n_hits>=30 for
BOTH bottom and top (stricter than round 1's single combined n>=300). Stability guard (unchanged
mechanism from round 1, delta-method relative SE of the POOLED train_lift ratio, threshold raised
to 15% per this round's task spec vs round 1's empirically-tuned 10%): cells with
train_lift_rel_se > MAX_REL_SE_LIFT are excluded from selection (kept in the reported grid).
Applied uniformly to all 4 HIT_TYPEs (not just touch_mfe/close_at_h) since the delta-method formula
is generic to any hit/baseline binomial-proportion ratio, and touch_mae_capped/
touch_giveback_sustained's extra AND conditions make their hit rates <= touch_mfe's at the same
(H,K) -- i.e. they are, if anything, MORE exposed to rare-event ratio noise, not less.

Recommendation logic: "mechanical strongest" = argmax selection_score among gate+stability-passing
cells (whole grid). "most stable" = argmin flatness_spread_pooled ((max-min)/min of pooled
TRAIN/VAL/OOS lift) among the same eligible pool. If the mechanical-strongest cell's own
flatness_spread_pooled is already <= MAX_REL_SE_LIFT (0.15), the H2-vs-H6-style tradeoff is
considered RESOLVED at that point and it is used as the final recommendation. Otherwise, falls back
to the most-stable cell (provided its TRAIN lift clears a minimal MIN_MEANINGFUL_LIFT=1.2x bar,
matching round 1's own judgment call preferring flat generalization over raw TRAIN lift) and
reports both "strongest" and "most stable" explicitly, same as round 1 did.

HOLDOUT (timestamp >= 2026-04-01) is dropped from the working frame immediately after load and
never referenced again -- no candidate, label, baseline draw, or feature stat in this run ever
reads a HOLDOUT row. TabPFN training / economic backtest / HOLDOUT exposure remain future work,
same scope boundary as round 1.

Run: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_short_term_return_z_gridscreen_hittype_20260901.py

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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/short_term_return_z_gridscreen_report.json"

# Identical Tier0 22-feature set to round 1 (21 + rsi) -- see
# build_btc_5m_evidence_signal_candidates_tier0_20260901.py module docstring for why this BTC set
# differs from ETH's own FEATURE_COLUMNS.
TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high", "range_width_pct",
    "hour_utc", "weekday", "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
    "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

GAP = 12  # fixed cluster-dedup convention (project-wide), NOT swept -- same as round 1
HIT_TYPES = ["touch_mfe", "close_at_h", "touch_mae_capped", "touch_giveback_sustained"]
HORIZONS = [2, 3, 6, 9, 12, 18]
KS = [1.0, 1.5, 1.75, 2.0, 2.5]

MAE_K_LOSS_MULT = 2.0       # fixed, project fib_extension_exhaustion MAE-cap constant -- not swept
GIVEBACK_FULL_MULT = 2      # FULL_WINDOW = GIVEBACK_FULL_MULT * horizon -- fixed, not swept
GIVEBACK_CEIL = 0.20        # fixed, project V_REBOUND convention -- not swept

MIN_TRAIN_N_SIDE = 300      # per-side gate (stricter than round 1's combined n>=300)
MIN_TRAIN_HITS_SIDE = 30    # per-side gate, NEW this round -- protects against rare-hit-count noise
MAX_REL_SE_LIFT = 0.15      # this round's spec value (round 1 used an empirically-tuned 0.10)
MIN_MEANINGFUL_LIFT = 1.2   # floor for the "most stable" fallback recommendation to still count

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
PERIODS = [("train", (None, VAL_START)), ("val", (VAL_START, OOS_START)), ("oos", (OOS_START, HOLDOUT_START))]

BASE_SEED = 20260901


def log(msg: str) -> None:
    print(f"[xrp_str_z_hittype] {msg}", flush=True)


def load_frame() -> pd.DataFrame:
    usecols = list(dict.fromkeys(
        ["timestamp", "high", "low", "close"] + TIER0_FEATURES
        + ["bottom_short_term_return_z", "top_short_term_return_z"]
    ))
    df = pd.read_csv(CSV_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_full = len(df)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"loaded {n_full} rows, truncated to {len(df)} rows before HOLDOUT_START={HOLDOUT_START.date()} "
        f"(HOLDOUT rows dropped immediately -- never read again below)")
    return df


def cluster_dedup_gap(idx: np.ndarray, anchor_val: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    """Collapse same-side fires within `gap` bars into one cluster, keep only the bar with the most
    extreme anchor_val (ret3_z) per cluster. Causal. Ported verbatim from round 1 /
    research_eth_short_term_return_z_metalabel_v2_gap_sweep_20260830.py::cluster_dedup_gap."""
    if len(idx) == 0:
        return idx
    order = np.argsort(idx)
    idx_sorted = idx[order]
    val_sorted = anchor_val[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    tmp = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "val": val_sorted})
    keep = (tmp.loc[tmp.groupby("cluster")["val"].idxmin()] if most_negative
            else tmp.loc[tmp.groupby("cluster")["val"].idxmax()])
    return np.sort(keep["idx"].to_numpy())


def period_mask(ts: np.ndarray, start: pd.Timestamp | None, end: pd.Timestamp | None) -> np.ndarray:
    m = np.ones(len(ts), dtype=bool)
    if start is not None:
        m &= ts >= np.datetime64(start)
    if end is not None:
        m &= ts < np.datetime64(end)
    return m


def window_needed_for(hit_type: str, horizon: int) -> int:
    return GIVEBACK_FULL_MULT * horizon if hit_type == "touch_giveback_sustained" else horizon


# ---------------------------------------------------------------------------
# Hit-rule implementations. Each takes an idx array (row positions), returns a 0/1 int array.
# ---------------------------------------------------------------------------

def hit_touch_mfe(high, low, close, atr, idx, horizon, k, side):
    if len(idx) == 0:
        return np.array([], dtype=int)
    entry, atr_i = close[idx], atr[idx]
    if side == "bottom":
        fut = np.array([high[i + 1:i + horizon + 1].max() for i in idx])
        return (fut >= entry + k * atr_i).astype(int)
    fut = np.array([low[i + 1:i + horizon + 1].min() for i in idx])
    return (fut <= entry - k * atr_i).astype(int)


def hit_close_at_h(close, atr, idx, horizon, k, side):
    if len(idx) == 0:
        return np.array([], dtype=int)
    entry, atr_i, end = close[idx], atr[idx], close[idx + horizon]
    if side == "bottom":
        return (end >= entry + k * atr_i).astype(int)
    return (end <= entry - k * atr_i).astype(int)


def hit_touch_mae_capped(high, low, close, atr, idx, horizon, k, side, k_loss_mult=MAE_K_LOSS_MULT):
    if len(idx) == 0:
        return np.array([], dtype=int)
    out = np.zeros(len(idx), dtype=int)
    for j, i in enumerate(idx):
        entry, a = close[i], atr[i]
        thresh = k * a
        if side == "bottom":
            cond = high[i + 1:i + horizon + 1] >= entry + thresh
            if not cond.any():
                continue
            touch_bar = i + 1 + int(cond.argmax())
            mae = entry - low[i + 1:touch_bar + 1].min()
        else:
            cond = low[i + 1:i + horizon + 1] <= entry - thresh
            if not cond.any():
                continue
            touch_bar = i + 1 + int(cond.argmax())
            mae = high[i + 1:touch_bar + 1].max() - entry
        if mae <= k_loss_mult * a:
            out[j] = 1
    return out


def hit_touch_giveback_sustained(high, low, close, atr, idx, horizon, k, side,
                                  full_mult=GIVEBACK_FULL_MULT, ceil=GIVEBACK_CEIL):
    if len(idx) == 0:
        return np.array([], dtype=int)
    full_w = full_mult * horizon
    out = np.zeros(len(idx), dtype=int)
    for j, i in enumerate(idx):
        entry, a = close[i], atr[i]
        if not (a > 0) or not np.isfinite(a):
            continue
        if side == "bottom":
            fast_move = close[i + 1:i + horizon + 1].max() - entry
            fast_mult = fast_move / a
            if fast_mult < k:
                continue
            peak = high[i + 1:i + full_w + 1].max()
            end_price = close[i + full_w]
            denom = peak - entry
            giveback = (peak - end_price) / denom if abs(denom) > 1e-12 else np.nan
        else:
            fast_move = entry - close[i + 1:i + horizon + 1].min()
            fast_mult = fast_move / a
            if fast_mult < k:
                continue
            peak = low[i + 1:i + full_w + 1].min()
            end_price = close[i + full_w]
            denom = entry - peak
            giveback = (end_price - peak) / denom if abs(denom) > 1e-12 else np.nan
        if np.isfinite(giveback) and giveback <= ceil:
            out[j] = 1
    return out


def compute_hit(hit_type, high, low, close, atr, idx, horizon, k, side):
    if hit_type == "touch_mfe":
        return hit_touch_mfe(high, low, close, atr, idx, horizon, k, side)
    if hit_type == "close_at_h":
        return hit_close_at_h(close, atr, idx, horizon, k, side)
    if hit_type == "touch_mae_capped":
        return hit_touch_mae_capped(high, low, close, atr, idx, horizon, k, side)
    if hit_type == "touch_giveback_sustained":
        return hit_touch_giveback_sustained(high, low, close, atr, idx, horizon, k, side)
    raise ValueError(f"unknown hit_type: {hit_type}")


def eligible_with_fallback(rows: list[dict], label: str) -> tuple[list[dict], str]:
    """gate+stability -> gate-only -> any-scored fallback chain, logging a WARNING on each fallback."""
    elig = [r for r in rows if r["gate_pass"] and r["stability_pass"] and r["selection_score_min_bottom_top"] is not None]
    if elig:
        return elig, "gate+stability"
    elig = [r for r in rows if r["gate_pass"] and r["selection_score_min_bottom_top"] is not None]
    if elig:
        log(f"WARNING ({label}): no cell passes the stability guard, falling back to gate-only")
        return elig, "gate_only"
    elig = [r for r in rows if r["selection_score_min_bottom_top"] is not None]
    log(f"WARNING ({label}): no cell passes gates at all, falling back to any scored cell")
    return elig, "no_gate"


def main() -> int:
    df = load_frame()
    n = len(df)
    ts = df["timestamp"].to_numpy()
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    ret3z = df["ret3_z"].to_numpy()
    bottom_trig = df["bottom_short_term_return_z"].fillna(False).to_numpy()
    top_trig = df["top_short_term_return_z"].fillna(False).to_numpy()
    any_trigger = bottom_trig | top_trig

    log(f"TRAIN rows={period_mask(ts, *PERIODS[0][1]).sum()}, VAL rows={period_mask(ts, *PERIODS[1][1]).sum()}, "
        f"OOS rows={period_mask(ts, *PERIODS[2][1]).sum()}")

    window_cache: dict[int, dict] = {}

    def get_window(window: int) -> dict:
        if window not in window_cache:
            idx_map = {}
            for side, trig, most_neg in [("bottom", bottom_trig, True), ("top", top_trig, False)]:
                raw_idx = np.flatnonzero(trig)
                raw_idx = raw_idx[(raw_idx < n - window) & np.isfinite(atr[raw_idx]) & np.isfinite(ret3z[raw_idx])]
                idx_map[side] = cluster_dedup_gap(raw_idx, ret3z[raw_idx], most_negative=most_neg, gap=GAP)
            eligible = np.flatnonzero((~any_trigger) & (np.arange(n) < n - window) & np.isfinite(atr))
            baseline_idx = {}
            for period_name, bounds in PERIODS:
                pm = period_mask(ts, *bounds)
                eligible_p = eligible[pm[eligible]]
                for side in ("bottom", "top"):
                    idx_side = idx_map[side]
                    idx_p = idx_side[pm[idx_side]]
                    n_need = len(idx_p)
                    seed = BASE_SEED + window * 1000 + (1 if side == "bottom" else 2) + \
                        {"train": 0, "val": 10000, "oos": 20000}[period_name]
                    rng = np.random.default_rng(seed)
                    n_draw = min(n_need, len(eligible_p))
                    baseline_idx[(period_name, side)] = (
                        rng.choice(eligible_p, size=n_draw, replace=False) if n_draw > 0 else np.array([], dtype=int)
                    )
            window_cache[window] = {"idx_map": idx_map, "baseline_idx": baseline_idx}
            for side in ("bottom", "top"):
                counts = {pn: int(period_mask(ts, *b)[idx_map[side]].sum()) for pn, b in PERIODS}
                log(f"window={window}: {side} deduped candidates train={counts['train']} "
                    f"val={counts['val']} oos={counts['oos']}")
        return window_cache[window]

    grid_rows: list[dict] = []
    for hit_type in HIT_TYPES:
        for horizon in HORIZONS:
            window = window_needed_for(hit_type, horizon)
            wd = get_window(window)
            idx_map, baseline_idx = wd["idx_map"], wd["baseline_idx"]

            for k in KS:
                row = {"hit_type": hit_type, "horizon": horizon, "k": k, "window_needed": window}
                side_train_lift = {}
                pooled_train = {}
                for period_name, bounds in PERIODS:
                    pm = period_mask(ts, *bounds)
                    cand_hits_all, base_hits_all = [], []
                    for side in ("bottom", "top"):
                        idx_side = idx_map[side]
                        idx_p = idx_side[pm[idx_side]]
                        b_idx = baseline_idx[(period_name, side)]

                        cand_hit = compute_hit(hit_type, high, low, close, atr, idx_p, horizon, k, side)
                        base_hit = compute_hit(hit_type, high, low, close, atr, b_idx, horizon, k, side)

                        n_cand, n_base = len(idx_p), len(b_idx)
                        n_hits = int(cand_hit.sum()) if n_cand else 0
                        n_base_hits = int(base_hit.sum()) if n_base else 0
                        cand_rate = float(cand_hit.mean()) if n_cand else float("nan")
                        base_rate = float(base_hit.mean()) if n_base else float("nan")
                        lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")

                        row[f"{period_name}_n_{side}"] = n_cand
                        row[f"{period_name}_n_hits_{side}"] = n_hits
                        row[f"{period_name}_n_base_{side}"] = n_base
                        row[f"{period_name}_n_base_hits_{side}"] = n_base_hits
                        row[f"{period_name}_hit_rate_{side}"] = round(cand_rate, 4) if cand_rate == cand_rate else None
                        row[f"{period_name}_baseline_hit_rate_{side}"] = round(base_rate, 4) if base_rate == base_rate else None
                        row[f"{period_name}_lift_{side}"] = round(lift, 4) if lift == lift else None

                        if period_name == "train":
                            side_train_lift[side] = lift

                        cand_hits_all.append(cand_hit)
                        base_hits_all.append(base_hit)

                    all_cand = np.concatenate(cand_hits_all) if sum(len(x) for x in cand_hits_all) else np.array([])
                    all_base = np.concatenate(base_hits_all) if sum(len(x) for x in base_hits_all) else np.array([])
                    pooled_cand_rate = float(all_cand.mean()) if len(all_cand) else float("nan")
                    pooled_base_rate = float(all_base.mean()) if len(all_base) else float("nan")
                    pooled_lift = pooled_cand_rate / pooled_base_rate if pooled_base_rate and pooled_base_rate > 0 else float("nan")
                    pooled_n = row[f"{period_name}_n_bottom"] + row[f"{period_name}_n_top"]

                    row[f"{period_name}_n"] = pooled_n
                    row[f"{period_name}_hit_rate"] = round(pooled_cand_rate, 4) if pooled_cand_rate == pooled_cand_rate else None
                    row[f"{period_name}_baseline_hit_rate"] = round(pooled_base_rate, 4) if pooled_base_rate == pooled_base_rate else None
                    row[f"{period_name}_lift"] = round(pooled_lift, 4) if pooled_lift == pooled_lift else None

                    if period_name == "train":
                        pooled_train = {"cand_rate": pooled_cand_rate, "base_rate": pooled_base_rate, "n": pooled_n}

                p_hit, p_base, n_tr = pooled_train["cand_rate"], pooled_train["base_rate"], pooled_train["n"]
                if p_hit and p_base and p_hit > 0 and p_base > 0 and n_tr > 0:
                    rel_se = float(np.sqrt((1 - p_hit) / (p_hit * n_tr) + (1 - p_base) / (p_base * n_tr)))
                else:
                    rel_se = float("inf")
                row["train_lift_rel_se"] = round(rel_se, 4) if np.isfinite(rel_se) else None

                gate_pass = bool(
                    row["train_n_bottom"] >= MIN_TRAIN_N_SIDE and row["train_n_top"] >= MIN_TRAIN_N_SIDE and
                    row["train_n_hits_bottom"] >= MIN_TRAIN_HITS_SIDE and row["train_n_hits_top"] >= MIN_TRAIN_HITS_SIDE
                )
                stability_pass = bool(np.isfinite(rel_se) and rel_se <= MAX_REL_SE_LIFT)
                row["gate_pass"] = gate_pass
                row["stability_pass"] = stability_pass

                lb, lt = side_train_lift.get("bottom"), side_train_lift.get("top")
                valid_score = lb is not None and lt is not None and lb == lb and lt == lt
                row["selection_score_min_bottom_top"] = round(min(lb, lt), 4) if valid_score else None

                grid_rows.append(row)
                log(f"{hit_type:<24s} H={horizon:>2} K={k:.2f} W={window:>2} "
                    f"TRAIN n=(b{row['train_n_bottom']},t{row['train_n_top']}) "
                    f"hits=(b{row['train_n_hits_bottom']},t{row['train_n_hits_top']}) "
                    f"lift(b/t/pooled)=({row['train_lift_bottom']},{row['train_lift_top']},{row['train_lift']}) "
                    f"relSE={row['train_lift_rel_se']} gate={gate_pass} stable={stability_pass} "
                    f"score={row['selection_score_min_bottom_top']} "
                    f"| VAL lift(b/t/pooled)=({row['val_lift_bottom']},{row['val_lift_top']},{row['val_lift']}) "
                    f"| OOS lift(b/t/pooled)=({row['oos_lift_bottom']},{row['oos_lift_top']},{row['oos_lift']})")

    # flatness across TRAIN/VAL/OOS pooled lift -- lower = more stable (matches round 1's manual
    # "1.50~1.56 범위" flat-band framing, computed programmatically this round)
    for row in grid_rows:
        vals = [row.get("train_lift"), row.get("val_lift"), row.get("oos_lift")]
        if any(v is None for v in vals) or any(v <= 0 for v in vals):
            row["flatness_spread_pooled"] = None
        else:
            row["flatness_spread_pooled"] = round((max(vals) - min(vals)) / min(vals), 4)

    # ---- overall (whole grid) selection ----
    overall_eligible, overall_basis = eligible_with_fallback(grid_rows, "overall")
    chosen = dict(max(overall_eligible, key=lambda r: r["selection_score_min_bottom_top"]))
    chosen["selection_rule"] = (
        f"max(min(train_lift_bottom, train_lift_top)) subject to train_n_bottom/top>={MIN_TRAIN_N_SIDE}, "
        f"train_n_hits_bottom/top>={MIN_TRAIN_HITS_SIDE}, train_lift_rel_se<={MAX_REL_SE_LIFT} (basis={overall_basis})"
    )
    log(f"OVERALL STRONGEST (mechanical): hit_type={chosen['hit_type']} H={chosen['horizon']} K={chosen['k']} "
        f"score={chosen['selection_score_min_bottom_top']} VAL lift={chosen['val_lift']} OOS lift={chosen['oos_lift']} "
        f"flatness={chosen['flatness_spread_pooled']}")

    stable_pool = [r for r in overall_eligible if r["flatness_spread_pooled"] is not None]
    most_stable = None
    if stable_pool:
        most_stable = dict(min(stable_pool, key=lambda r: (r["flatness_spread_pooled"], -r["selection_score_min_bottom_top"])))
        log(f"OVERALL MOST STABLE: hit_type={most_stable['hit_type']} H={most_stable['horizon']} K={most_stable['k']} "
            f"flatness={most_stable['flatness_spread_pooled']} score={most_stable['selection_score_min_bottom_top']} "
            f"TRAIN/VAL/OOS lift={most_stable['train_lift']}/{most_stable['val_lift']}/{most_stable['oos_lift']}")

    # ---- per-HIT_TYPE family leaderboard ----
    family_leaderboard = {}
    for ht in HIT_TYPES:
        rows_ht = [r for r in grid_rows if r["hit_type"] == ht]
        elig_ht, basis_ht = eligible_with_fallback(rows_ht, ht)
        best_ht = dict(max(elig_ht, key=lambda r: r["selection_score_min_bottom_top"])) if elig_ht else None
        stable_ht_pool = [r for r in elig_ht if r["flatness_spread_pooled"] is not None]
        most_stable_ht = (
            dict(min(stable_ht_pool, key=lambda r: (r["flatness_spread_pooled"], -r["selection_score_min_bottom_top"])))
            if stable_ht_pool else None
        )
        family_leaderboard[ht] = {"strongest": best_ht, "most_stable": most_stable_ht, "selection_basis": basis_ht}
        if best_ht:
            log(f"FAMILY {ht}: strongest H={best_ht['horizon']} K={best_ht['k']} score={best_ht['selection_score_min_bottom_top']} "
                f"TRAIN/VAL/OOS={best_ht['train_lift']}/{best_ht['val_lift']}/{best_ht['oos_lift']} "
                f"flat={best_ht['flatness_spread_pooled']}")
        if most_stable_ht:
            log(f"FAMILY {ht}: most_stable H={most_stable_ht['horizon']} K={most_stable_ht['k']} "
                f"flat={most_stable_ht['flatness_spread_pooled']} score={most_stable_ht['selection_score_min_bottom_top']} "
                f"TRAIN/VAL/OOS={most_stable_ht['train_lift']}/{most_stable_ht['val_lift']}/{most_stable_ht['oos_lift']}")

    # ---- recommendation logic ----
    tradeoff_resolved = chosen.get("flatness_spread_pooled") is not None and chosen["flatness_spread_pooled"] <= MAX_REL_SE_LIFT
    if tradeoff_resolved:
        recommended = chosen
        recommendation_note = (
            "The mechanical strongest cell is ALSO flat across TRAIN/VAL/OOS pooled lift "
            f"(flatness_spread_pooled={chosen['flatness_spread_pooled']} <= {MAX_REL_SE_LIFT}) -- "
            "the round-1 H2-vs-H6-style tradeoff is RESOLVED at this point, no separate stability pick needed."
        )
    else:
        ms_lift = most_stable.get("train_lift") if most_stable else None
        if most_stable is not None and ms_lift is not None and ms_lift >= MIN_MEANINGFUL_LIFT:
            recommended = most_stable
            recommendation_note = (
                f"The mechanical strongest cell (hit_type={chosen['hit_type']} H={chosen['horizon']} K={chosen['k']}) "
                f"has flatness_spread_pooled={chosen.get('flatness_spread_pooled')} > {MAX_REL_SE_LIFT} -- NOT resolved "
                "by that cell alone. Falling back to the most-stable gate+stability-passing cell (TRAIN lift "
                f">= {MIN_MEANINGFUL_LIFT}x meaningful-lift floor), matching round 1's own precedent of preferring "
                "flat TRAIN/VAL/OOS generalization over raw TRAIN lift."
            )
        else:
            recommended = chosen
            recommendation_note = (
                "Neither the strongest nor the most-stable cell cleanly resolves the tradeoff (the most-stable "
                f"cell's TRAIN lift is below the {MIN_MEANINGFUL_LIFT}x floor or unavailable). Defaulting to the "
                "mechanical strongest cell for this report's feature analysis; human review recommended before "
                "treating either point as a live candidate."
            )
    log(f"RECOMMENDATION: {recommendation_note}")
    log(f"RECOMMENDED POINT: hit_type={recommended['hit_type']} H={recommended['horizon']} K={recommended['k']}")

    # ---- feature analysis at the recommended (hit_type, horizon, k) ----
    horizon, k, hit_type_rec = recommended["horizon"], recommended["k"], recommended["hit_type"]
    window = window_needed_for(hit_type_rec, horizon)
    idx_map = get_window(window)["idx_map"]

    rows = []
    for side in ("bottom", "top"):
        idx_side = idx_map[side]
        hit = compute_hit(hit_type_rec, high, low, close, atr, idx_side, horizon, k, side)
        sub = df.iloc[idx_side][["timestamp"] + TIER0_FEATURES].copy()
        sub["side"] = side
        sub["hit"] = hit
        rows.append(sub)
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    n_before = len(fires)
    fires = fires.dropna(subset=TIER0_FEATURES + ["hit"]).reset_index(drop=True)
    log(f"feature-analysis frame: {len(fires)}/{n_before} usable after dropna")

    fts = fires["timestamp"]
    train_f = fires.loc[fts < VAL_START].reset_index(drop=True)
    val_f = fires.loc[(fts >= VAL_START) & (fts < OOS_START)].reset_index(drop=True)
    oos_f = fires.loc[(fts >= OOS_START) & (fts < HOLDOUT_START)].reset_index(drop=True)
    log(f"feature-analysis split: TRAIN={len(train_f)} VAL={len(val_f)} OOS={len(oos_f)}")

    corr_rows = []
    for feat in TIER0_FEATURES:
        c = train_f[feat].astype(float).corr(train_f["hit"].astype(float))
        corr_rows.append({"feature": feat, "point_biserial_corr": round(float(c), 4) if c == c else None})
    corr_rows.sort(key=lambda r: -abs(r["point_biserial_corr"]) if r["point_biserial_corr"] is not None else 0)
    log("top point-biserial |corr| features:")
    for r in corr_rows[:6]:
        log(f"  {r['feature']:<22s} corr={r['point_biserial_corr']:+.4f}")

    X_train, y_train = train_f[TIER0_FEATURES], train_f["hit"].astype(int)
    X_val, y_val = val_f[TIER0_FEATURES], val_f["hit"].astype(int)
    clf = HistGradientBoostingClassifier(random_state=BASE_SEED)
    clf.fit(X_train, y_train)
    val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = float(roc_auc_score(y_val, val_proba))
    log(f"HGB TRAIN-fit VAL AUC={val_auc:.4f}")

    perm = permutation_importance(clf, X_val, y_val, scoring="roc_auc", n_repeats=10,
                                   random_state=BASE_SEED, n_jobs=1)
    perm_rows = [
        {"feature": feat, "importance_mean": round(float(m), 5), "importance_std": round(float(s), 5)}
        for feat, m, s in zip(TIER0_FEATURES, perm.importances_mean, perm.importances_std)
    ]
    perm_rows.sort(key=lambda r: -r["importance_mean"])
    log("top permutation-importance features (VAL, AUC-scored):")
    for r in perm_rows[:6]:
        log(f"  {r['feature']:<22s} importance={r['importance_mean']:+.5f} (std={r['importance_std']:.5f})")

    report = {
        "signal": "short_term_return_z",
        "asset": "BTCUSDT",
        "stage": "grid_screen_hit_type_dimension_and_feature_analysis",
        "round": 2,
        "prior_round": "research_btc_short_term_return_z_gridscreen_20260901.py (touch_mfe only, HORIZON x K grid)",
        "not_done_this_round": ["TabPFN training", "economic/cost-gate backtest", "HOLDOUT exposure"],
        "holdout_touched": False,
        "holdout_start": str(HOLDOUT_START),
        "gap_fixed": GAP,
        "hit_types_tried": HIT_TYPES,
        "horizons_tried": HORIZONS,
        "ks_tried": KS,
        "mae_k_loss_mult_fixed": MAE_K_LOSS_MULT,
        "giveback_full_window_mult_fixed": GIVEBACK_FULL_MULT,
        "giveback_ceiling_fixed": GIVEBACK_CEIL,
        "min_train_n_side": MIN_TRAIN_N_SIDE,
        "min_train_hits_side": MIN_TRAIN_HITS_SIDE,
        "max_rel_se_lift": MAX_REL_SE_LIFT,
        "min_meaningful_lift_for_stable_fallback": MIN_MEANINGFUL_LIFT,
        "selection_methodology_note": (
            "Selection score is min(train_lift_bottom, train_lift_top) -- the WEAKER side -- not the "
            "pooled lift round 1 used, so a cell can't win on the strength of one side alone. Gates "
            "(TRAIN only): n_cand>=300 AND n_hits>=30 required independently for BOTH bottom and top. "
            "Stability guard (delta-method relative SE of the pooled TRAIN hit/baseline ratio, same "
            "mechanism as round 1, threshold raised to 0.15 per this round's spec) applied uniformly "
            "across all 4 HIT_TYPEs."
        ),
        "tier0_features": TIER0_FEATURES,
        "grid": grid_rows,
        "overall_strongest": chosen,
        "overall_most_stable": most_stable,
        "overall_selection_basis": overall_basis,
        "family_leaderboard": family_leaderboard,
        "tradeoff_resolved": tradeoff_resolved,
        "recommendation_note": recommendation_note,
        "recommended": recommended,
        "feature_analysis": {
            "hit_type": hit_type_rec, "horizon": horizon, "k": k,
            "n_train": len(train_f), "n_val": len(val_f), "n_oos": len(oos_f),
            "point_biserial_corr_train": corr_rows,
            "hgb_val_auc": round(val_auc, 4),
            "permutation_importance_val": perm_rows,
        },
        "splits": {
            "train": f"< {VAL_START.date()}",
            "val": f"{VAL_START.date()} ~ {OOS_START.date()}",
            "oos": f"{OOS_START.date()} ~ {HOLDOUT_START.date()}",
            "holdout": f">= {HOLDOUT_START.date()} (NOT TOUCHED THIS ROUND)",
        },
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
