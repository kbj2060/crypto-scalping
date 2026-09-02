#!/usr/bin/env python3
"""HIT_TYPE x HORIZON x K 3-D grid screen for BTC's `liquidity_sweep` evidence signal -- round 2,
redone after round 1 (scripts/research_btc_liquidity_sweep_gridscreen_20260901.py) fixed HIT_TYPE
to touch-based MFE only. The user asked a pointed question: why assume touch-based-MFE is the right
HIT definition at all -- shouldn't the HIT DEFINITION ITSELF be grid-searched per signal, not just H
and K? This script adds HIT_TYPE as a third grid axis to answer that directly.

Round 1 found the raw touch_mfe grid weak overall (max lift 1.058x across 42 cells, K>=3.0 mostly
<1.0 on the bottom side) and chose H=15/K=2.0, but TRAIN lift 1.056 on the bottom side collapsed to
VAL 1.013 (near-noise) while top strengthened to 1.152. This round checks whether a different hit
definition (stricter close-only, MAE-capped, or persistence/giveback-checked) reveals a stronger,
more stable signal, or confirms the weakness is intrinsic to the raw trigger regardless of hit
definition.

Data: data/labels/xrp_5m_evidence_signal_candidates_20260903/btc_5m_evidence_signal_candidates_
tier0.csv (277,191 rows, 2024-01-01 to 2026-08-20, BTCUSDT 5m). `bottom_liquidity_sweep`/
`top_liquidity_sweep` triggers and all Tier0 features are read as-is, NOT recomputed (same
convention as round 1).

Four HIT_TYPE families (entry=close[i], atr=atr[i] absolute-price ATR14, candidate at row i):

  1. touch_mfe (round 1's method, kept as baseline for comparison):
       bottom: hit=1 if high[i+1:i+H+1].max() >= entry + K*atr
       top:    hit=1 if low[i+1:i+H+1].min()  <= entry - K*atr

  2. close_at_h (stricter -- only the bar-H close counts, no credit for touch-then-revert):
       bottom: hit=1 if close[i+H] >= entry + K*atr
       top:    hit=1 if close[i+H] <= entry - K*atr

  3. touch_mae_capped (touch_mfe, disqualified if price first went too far against the position
     before reaching target; MAE measured only over [i+1, touch_bar], the first bar the touch
     condition fires -- order-aware, not whole-window):
       K_LOSS_MULT = 2.0 (fixed, matches this project's fib_extension_exhaustion MAE-cap convention)
       bottom: touch_bar = first bar in [i+1,i+H] with high>=entry+K*atr (none -> not a hit);
               MAE = entry - low[i+1:touch_bar+1].min(); hit = touch found AND MAE <= K_LOSS_MULT*atr
       top: mirror (MAE = high[i+1:touch_bar+1].max() - entry)

  4. touch_giveback_sustained (V_REBOUND-style persistence check, used here only as a CANDIDATE hit
     definition for liquidity_sweep, not literally the V_REBOUND label):
       FAST_WINDOW = H, FULL_WINDOW = 2*H (fixed multiple, not swept separately)
       giveback ceiling = 0.20 (fixed, matches this project's V_REBOUND convention)
       bottom: fast_move = close[i+1:i+FAST_WINDOW+1].max() - entry; fast_mult = fast_move/atr;
               peak = high[i+1:i+FULL_WINDOW+1].max(); end_price = close[i+FULL_WINDOW];
               denom = peak - entry; giveback = (peak-end_price)/denom, NaN if denom<=0.
               (Note: denom<=0 can only happen when fast_mult already fails the K>=1.5 gate --
               peak is a high-based max over a window that is a superset of the fast window, and
               high[t]>=close[t] always, so peak >= entry+fast_move whenever fast_move>0; the NaN
               path is therefore inert but is still handled explicitly for safety, matching the
               task's "NaN-safe" spec.)
               hit = fast_mult >= K AND giveback <= 0.20
       top: mirror (trough instead of peak, signs flipped)

Grid: HIT_TYPE(4) x HORIZON in [10,15,20,25,30,40] x K in [1.5,2.0,2.5,3.0,3.5,4.0] x side(2) = 288
cells. Gate UNCHANGED from round 1: n_cand>=300 and n_hits>=30 BOTH sides -- this signal's sample is
healthy (round 1 had ~4,900-5,000 TRAIN candidates/side at every cell, essentially flat across H),
so unlike this project's fib_extension_exhaustion round-2 sibling screen, no gate relaxation is
needed or applied here.

Split (Fresh-Forward, matches round 1 and this repo's contract): TRAIN <2025-09-01, VAL 2025-09-01
to 2026-01-01, OOS 2026-01-01 to 2026-04-01 (bonus check on the GLOBAL chosen point only, never
selected on), HOLDOUT >=2026-04-01 (dropped at load time, never read past that point).

Selection is TRAIN-only: for each HIT_TYPE, argmax within that family of min(train_lift_bottom,
train_lift_top) among gate-passing (H,K) cells (-> per-HIT_TYPE leaderboard, 4 entries); the GLOBAL
winner is simply the best of those 4 family winners (mathematically identical to one flat argmax
over the full 288-cell union -- asserted in code). VAL confirms the GLOBAL chosen point AND,
additionally, each of the 4 leaderboard points -- this is a small, deliberate addition beyond the
letter of the task spec (which only required VAL confirmation of the single global point), because
the task's own step-4 goal is "see how the 4 definitions actually compare", and round 1 already
showed TRAIN lift alone can be misleading (bottom lift 1.056 -> VAL 1.013). OOS bonus stays scoped
to the single global point only, per spec, to limit multiple-comparison exposure.

Run with: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_liquidity_sweep_gridscreen_hittype_20260901.py

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

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/liquidity_sweep_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # never touched past this point

HORIZON_GRID = [10, 15, 20, 25, 30, 40]
K_GRID = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
HIT_TYPES = ["touch_mfe", "close_at_h", "touch_mae_capped", "touch_giveback_sustained"]

MAE_K_LOSS_MULT = 2.0     # touch_mae_capped, fixed per task spec (not swept)
GIVEBACK_CEIL = 0.20      # touch_giveback_sustained, fixed per task spec (V_REBOUND convention)
FULL_WINDOW_MULT = 2      # touch_giveback_sustained FULL_WINDOW = 2*H, fixed per task spec

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

MIN_TRAIN_CANDIDATES = 300  # per side, same as round 1 (unrelaxed -- this signal's sample is healthy)
MIN_TRAIN_HITS = 30         # per side, same as round 1
RNG_SEED = 20260901


def log(msg: str) -> None:
    print(f"[btc_liq_sweep_hittype_gridscreen] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    usecols = sorted(set(
        ["timestamp", "open", "high", "low", "close", "atr",
         "bottom_liquidity_sweep", "top_liquidity_sweep"] + TIER0_FEATURES
    ))
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    assert df["timestamp"].max() < HOLDOUT_START, "HOLDOUT row leaked past truncation"
    return df


# ---------------------------------------------------------------------------
# HIT_TYPE raw-material + hit computation.
#
# "Raw materials" are the parts of each hit formula that do NOT depend on K --
# computed once per (side, hit_type, horizon) and reused across the whole
# K_GRID sweep (touch_mfe/close_at_h/touch_giveback_sustained all qualify: the
# window extremes and end-price lookups are pure functions of the forward bars,
# not of K). Only touch_mae_capped's touch-bar search genuinely depends on K
# (the touch threshold itself IS K*atr, so the first-touch index moves with K)
# -- it alone is recomputed fresh per K via a per-candidate loop. Candidate
# pools are in the thousands, not the 277K full bar count, so this stays fast.
# ---------------------------------------------------------------------------

def required_fwd_bars(hit_type: str, horizon: int) -> int:
    return FULL_WINDOW_MULT * horizon if hit_type == "touch_giveback_sustained" else horizon


def fwd_extreme(pos_idx: np.ndarray, arr: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """max/min(arr[i+1:i+horizon+1]) per i in pos_idx. Caller guarantees i+horizon < len(arr)."""
    out = np.empty(len(pos_idx), dtype=float)
    for out_i, i in enumerate(pos_idx):
        window = arr[i + 1:i + horizon + 1]
        out[out_i] = window.max() if mode == "max" else window.min()
    return out


def raw_materials(hit_type: str, side: str, pos_idx: np.ndarray, horizon: int,
                   high: np.ndarray, low: np.ndarray, close: np.ndarray) -> dict:
    if len(pos_idx) == 0:
        return {}
    if hit_type == "touch_mfe":
        mode = "max" if side == "bottom" else "min"
        return {"ext": fwd_extreme(pos_idx, high if side == "bottom" else low, horizon, mode)}
    if hit_type == "close_at_h":
        return {"ext": close[pos_idx + horizon]}  # caller guarantees pos_idx+horizon < n
    if hit_type == "touch_giveback_sustained":
        full = FULL_WINDOW_MULT * horizon
        mode = "max" if side == "bottom" else "min"
        fast_ext = fwd_extreme(pos_idx, close, horizon, mode)
        full_ext = fwd_extreme(pos_idx, high if side == "bottom" else low, full, mode)
        end_price = close[pos_idx + full]
        return {"fast_ext": fast_ext, "full_ext": full_ext, "end_price": end_price}
    if hit_type == "touch_mae_capped":
        return {}  # K-dependent, computed fresh per K in compute_hits()
    raise ValueError(hit_type)


def simple_hit(side: str, k: float, ext: np.ndarray, entry: np.ndarray, atr: np.ndarray) -> np.ndarray:
    return (ext - entry >= k * atr) if side == "bottom" else (entry - ext >= k * atr)


def giveback_hit(side: str, k: float, raw: dict, entry: np.ndarray, atr: np.ndarray) -> np.ndarray:
    fast_ext, full_ext, end_price = raw["fast_ext"], raw["full_ext"], raw["end_price"]
    if side == "bottom":
        fast_move = fast_ext - entry
        denom = full_ext - entry             # peak - entry
        giveback_raw = full_ext - end_price   # peak - end_price
    else:
        fast_move = entry - fast_ext
        denom = entry - full_ext              # entry - trough
        giveback_raw = end_price - full_ext   # end_price - trough
    fast_mult = fast_move / atr
    with np.errstate(invalid="ignore", divide="ignore"):
        giveback = np.where(denom > 0, giveback_raw / np.where(denom > 0, denom, 1.0), np.nan)
        return (fast_mult >= k) & (giveback <= GIVEBACK_CEIL)


def mae_capped_hit(pos_idx: np.ndarray, side: str, horizon: int, k: float,
                    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray) -> np.ndarray:
    """Order-aware: touch_bar = first bar in [i+1,i+H] crossing K*atr; MAE = worst excursion over
    [i+1, touch_bar] only. Genuinely K-dependent (touch_bar moves with K)."""
    hit = np.zeros(len(pos_idx), dtype=bool)
    for out_i, i in enumerate(pos_idx):
        entry = close[i]
        a = atr[i]
        if side == "bottom":
            seg_high = high[i + 1:i + horizon + 1]
            touches = np.flatnonzero(seg_high >= entry + k * a)
            if touches.size == 0:
                continue
            tb = int(touches[0])  # index relative to i+1
            mae = entry - low[i + 1:i + 2 + tb].min()
            hit[out_i] = mae <= MAE_K_LOSS_MULT * a
        else:
            seg_low = low[i + 1:i + horizon + 1]
            touches = np.flatnonzero(seg_low <= entry - k * a)
            if touches.size == 0:
                continue
            tb = int(touches[0])
            mae = high[i + 1:i + 2 + tb].max() - entry
            hit[out_i] = mae <= MAE_K_LOSS_MULT * a
    return hit


def compute_hits(hit_type: str, side: str, k: float, pos_idx: np.ndarray, entry: np.ndarray,
                  atr_g: np.ndarray, raw: dict, horizon: int,
                  high: np.ndarray, low: np.ndarray, close: np.ndarray, atr_full: np.ndarray) -> np.ndarray:
    if len(pos_idx) == 0:
        return np.array([], dtype=bool)
    if hit_type == "touch_mae_capped":
        return mae_capped_hit(pos_idx, side, horizon, k, high, low, close, atr_full)
    if hit_type == "touch_giveback_sustained":
        return giveback_hit(side, k, raw, entry, atr_g)
    return simple_hit(side, k, raw["ext"], entry, atr_g)


def base_pools(df: pd.DataFrame, split_mask: np.ndarray, atr: np.ndarray) -> dict:
    elig = split_mask & df["atr"].notna().to_numpy() & (atr > 0) & df["close"].notna().to_numpy()
    pools = {}
    for side in ("bottom", "top"):
        trig = df[f"{side}_liquidity_sweep"].fillna(False).to_numpy()
        pools[side] = {"trig": np.flatnonzero(elig & trig), "nontrig": np.flatnonzero(elig & ~trig)}
    return pools


def filtered_pool(pool_side: dict, hit_type: str, horizon: int, n: int,
                   rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    req = required_fwd_bars(hit_type, horizon)
    trig = pool_side["trig"]
    nontrig = pool_side["nontrig"]
    cand_idx = trig[trig + req < n]
    noncand_idx = nontrig[nontrig + req < n]
    n_base = min(len(cand_idx), len(noncand_idx))
    base_idx = rng.choice(noncand_idx, size=n_base, replace=False) if n_base > 0 else np.array([], dtype=int)
    return cand_idx, base_idx


def grid_row(hit_type: str, side: str, horizon: int, k: float, cand_idx: np.ndarray, cand_hit: np.ndarray,
             base_idx: np.ndarray, base_hit: np.ndarray) -> dict:
    n_cand, n_base = len(cand_idx), len(base_idx)
    n_cand_hits = int(cand_hit.sum()) if n_cand else 0
    n_base_hits = int(base_hit.sum()) if n_base else 0
    cand_rate = float(cand_hit.mean()) if n_cand else float("nan")
    base_rate = float(base_hit.mean()) if n_base else float("nan")
    lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")
    return {
        "hit_type": hit_type, "side": side, "horizon": horizon, "k": k,
        "n_cand": n_cand, "n_cand_hits": n_cand_hits,
        "n_base": n_base, "n_base_hits": n_base_hits,
        "cand_hit_rate": round(cand_rate, 4) if np.isfinite(cand_rate) else None,
        "base_hit_rate": round(base_rate, 4) if np.isfinite(base_rate) else None,
        "lift": round(lift, 4) if np.isfinite(lift) else None,
    }


def eval_point(side: str, hit_type: str, horizon: int, k: float, pool_side: dict,
               high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray, n: int,
               rng: np.random.Generator):
    cand_idx, base_idx = filtered_pool(pool_side, hit_type, horizon, n, rng)
    cand_entry, cand_atr = close[cand_idx], atr[cand_idx]
    base_entry, base_atr = close[base_idx], atr[base_idx]
    cand_raw = raw_materials(hit_type, side, cand_idx, horizon, high, low, close)
    base_raw = raw_materials(hit_type, side, base_idx, horizon, high, low, close)
    cand_hit = compute_hits(hit_type, side, k, cand_idx, cand_entry, cand_atr, cand_raw, horizon, high, low, close, atr)
    base_hit = compute_hits(hit_type, side, k, base_idx, base_entry, base_atr, base_raw, horizon, high, low, close, atr)
    row = grid_row(hit_type, side, horizon, k, cand_idx, cand_hit, base_idx, base_hit)
    return row, cand_idx, cand_hit


def main() -> int:
    log("loading BTC Tier0 candidate CSV...")
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

    train_pools = base_pools(df, train_mask, atr)
    val_pools = base_pools(df, val_mask, atr)
    oos_pools = base_pools(df, oos_mask, atr)

    rng = np.random.default_rng(RNG_SEED)

    # ---- TRAIN 3-D grid: HIT_TYPE x side x HORIZON x K, raw materials cached per
    # (side,hit_type,horizon) and reused across the K sweep ----
    grid_rows: list[dict] = []
    train_pool_cache: dict[tuple[str, str, int], dict] = {}
    for side in ("bottom", "top"):
        for hit_type in HIT_TYPES:
            for horizon in HORIZON_GRID:
                cand_idx, base_idx = filtered_pool(train_pools[side], hit_type, horizon, n, rng)
                cand_entry, cand_atr = close[cand_idx], atr[cand_idx]
                base_entry, base_atr = close[base_idx], atr[base_idx]
                cand_raw = raw_materials(hit_type, side, cand_idx, horizon, high, low, close)
                base_raw = raw_materials(hit_type, side, base_idx, horizon, high, low, close)
                train_pool_cache[(side, hit_type, horizon)] = dict(
                    cand_idx=cand_idx, cand_entry=cand_entry, cand_atr=cand_atr, cand_raw=cand_raw,
                    base_idx=base_idx, base_entry=base_entry, base_atr=base_atr, base_raw=base_raw,
                )
                rows_here = []
                for k in K_GRID:
                    cand_hit = compute_hits(hit_type, side, k, cand_idx, cand_entry, cand_atr, cand_raw,
                                             horizon, high, low, close, atr)
                    base_hit = compute_hits(hit_type, side, k, base_idx, base_entry, base_atr, base_raw,
                                             horizon, high, low, close, atr)
                    rows_here.append(grid_row(hit_type, side, horizon, k, cand_idx, cand_hit, base_idx, base_hit))
                grid_rows.extend(rows_here)
                best = max(rows_here, key=lambda r: (r["lift"] if r["lift"] is not None else -1))
                log(f"  TRAIN hit_type={hit_type:24s} side={side:6s} H={horizon:>3d}: n_cand={len(cand_idx):>5d} "
                    f"best_lift={best['lift']} @K={best['k']}")

    expected_rows = len(HIT_TYPES) * 2 * len(HORIZON_GRID) * len(K_GRID)
    log(f"\nfull TRAIN grid: {len(grid_rows)} rows (expected {expected_rows})")
    assert len(grid_rows) == expected_rows

    # ---- selection: per-HIT_TYPE leaderboard first (argmax within each family, gated); GLOBAL
    # winner = best of the 4 family winners (== flat argmax over the 288-cell union, asserted) ----
    row_lookup = {(r["hit_type"], r["side"], r["horizon"], r["k"]): r for r in grid_rows}

    all_gate_passing: list[dict] = []
    per_hit_type_best: dict[str, dict] = {}
    for hit_type in HIT_TYPES:
        family_cells = []
        for horizon in HORIZON_GRID:
            for k in K_GRID:
                b = row_lookup[(hit_type, "bottom", horizon, k)]
                t = row_lookup[(hit_type, "top", horizon, k)]
                if b["lift"] is None or t["lift"] is None:
                    continue
                gate = (b["n_cand"] >= MIN_TRAIN_CANDIDATES and t["n_cand"] >= MIN_TRAIN_CANDIDATES
                        and b["n_cand_hits"] >= MIN_TRAIN_HITS and t["n_cand_hits"] >= MIN_TRAIN_HITS)
                if not gate:
                    continue
                cell = {
                    "hit_type": hit_type, "horizon": horizon, "k": k,
                    "lift_bottom": b["lift"], "lift_top": t["lift"], "joint_min": min(b["lift"], t["lift"]),
                    "n_cand_bottom": b["n_cand"], "n_cand_top": t["n_cand"],
                    "n_hits_bottom": b["n_cand_hits"], "n_hits_top": t["n_cand_hits"],
                }
                family_cells.append(cell)
                all_gate_passing.append(cell)
        assert family_cells, f"no gate-passing (H,K) cell for hit_type={hit_type} -- unexpected given round 1's sample size"
        family_sorted = sorted(family_cells, key=lambda c: c["joint_min"], reverse=True)
        per_hit_type_best[hit_type] = family_sorted[0]
        log(f"  FAMILY BEST {hit_type:24s}: H={family_sorted[0]['horizon']} K={family_sorted[0]['k']} "
            f"joint={family_sorted[0]['joint_min']:.3f} (bottom={family_sorted[0]['lift_bottom']:.3f} "
            f"top={family_sorted[0]['lift_top']:.3f}, n_gate_passing_in_family={len(family_cells)}/{len(HORIZON_GRID)*len(K_GRID)})")

    assert all_gate_passing, "no (HIT_TYPE,H,K) combo passed the gates at all"
    all_sorted = sorted(all_gate_passing, key=lambda c: c["joint_min"], reverse=True)
    global_chosen = all_sorted[0]
    CHOSEN_HIT_TYPE = global_chosen["hit_type"]
    CHOSEN_H = global_chosen["horizon"]
    CHOSEN_K = global_chosen["k"]
    assert per_hit_type_best[CHOSEN_HIT_TYPE] is global_chosen, "global/family-best inconsistency (bug)"

    log(f"\n=== GLOBAL CHOSEN: HIT_TYPE={CHOSEN_HIT_TYPE} HORIZON={CHOSEN_H} K={CHOSEN_K}: "
        f"TRAIN lift bottom={global_chosen['lift_bottom']} top={global_chosen['lift_top']} "
        f"joint(min)={global_chosen['joint_min']} ===")
    log("\nTop 10 combos overall by joint(min) TRAIN lift:")
    for c in all_sorted[:10]:
        log(f"  {c['hit_type']:24s} H={c['horizon']:>3d} K={c['k']:.1f}: bottom={c['lift_bottom']:.3f} "
            f"top={c['lift_top']:.3f} joint={c['joint_min']:.3f}")

    # ---- VAL confirmation: GLOBAL point (mandatory) + each of the 4 per-HIT_TYPE leaderboard
    # points (extra, see module docstring) ----
    def confirm_on_val(hit_type: str, horizon: int, k: float):
        rows = {}
        for side in ("bottom", "top"):
            row, _cand_idx, _cand_hit = eval_point(side, hit_type, horizon, k, val_pools[side],
                                                     high, low, close, atr, n, rng)
            rows[side] = row
        lift_b, lift_t = rows["bottom"]["lift"], rows["top"]["lift"]
        joint = min(lift_b, lift_t) if lift_b is not None and lift_t is not None else None
        return rows, joint

    val_rows_global, val_joint_global = confirm_on_val(CHOSEN_HIT_TYPE, CHOSEN_H, CHOSEN_K)
    for side in ("bottom", "top"):
        r = val_rows_global[side]
        log(f"  VAL(GLOBAL) side={side:6s} hit_type={CHOSEN_HIT_TYPE} H={CHOSEN_H} K={CHOSEN_K}: "
            f"n_cand={r['n_cand']} lift={r['lift']} cand_hit_rate={r['cand_hit_rate']} base_hit_rate={r['base_hit_rate']}")

    per_hit_type_leaderboard = []
    for hit_type in HIT_TYPES:
        best = per_hit_type_best[hit_type]
        if hit_type == CHOSEN_HIT_TYPE:
            val_rows_h, val_joint_h = val_rows_global, val_joint_global
        else:
            val_rows_h, val_joint_h = confirm_on_val(hit_type, best["horizon"], best["k"])
        entry = dict(best)
        entry["val_lift_bottom"] = val_rows_h["bottom"]["lift"]
        entry["val_lift_top"] = val_rows_h["top"]["lift"]
        entry["val_joint_min"] = round(val_joint_h, 4) if val_joint_h is not None else None
        entry["val_n_cand_bottom"] = val_rows_h["bottom"]["n_cand"]
        entry["val_n_cand_top"] = val_rows_h["top"]["n_cand"]
        entry["is_global_winner"] = (hit_type == CHOSEN_HIT_TYPE)
        per_hit_type_leaderboard.append(entry)
        log(f"  VAL(family) {hit_type:24s} H={best['horizon']} K={best['k']}: "
            f"val_lift bottom={entry['val_lift_bottom']} top={entry['val_lift_top']} joint={entry['val_joint_min']}")

    # ---- OOS bonus check at GLOBAL chosen point only (never selected on) ----
    oos_rows_global = {}
    for side in ("bottom", "top"):
        row, _ci, _ch = eval_point(side, CHOSEN_HIT_TYPE, CHOSEN_H, CHOSEN_K, oos_pools[side],
                                    high, low, close, atr, n, rng)
        oos_rows_global[side] = row
        log(f"  OOS(bonus,GLOBAL) side={side:6s}: n_cand={row['n_cand']} lift={row['lift']} "
            f"cand_hit_rate={row['cand_hit_rate']} base_hit_rate={row['base_hit_rate']}")

    # ---- feature analysis at GLOBAL chosen (HIT_TYPE,H,K): TRAIN candidates -> hit label,
    # VAL candidates -> AUC + permutation importance (same methodology as round 1) ----
    feature_analysis = {}
    for side in ("bottom", "top"):
        pool = train_pool_cache[(side, CHOSEN_HIT_TYPE, CHOSEN_H)]
        cand_idx = pool["cand_idx"]
        hit = compute_hits(CHOSEN_HIT_TYPE, side, CHOSEN_K, cand_idx, pool["cand_entry"], pool["cand_atr"],
                            pool["cand_raw"], CHOSEN_H, high, low, close, atr).astype(int)
        feat_df = df.loc[cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        feat_df["hit"] = hit

        corr = feat_df.corr(numeric_only=True)["hit"].drop("hit").sort_values(key=lambda s: s.abs(), ascending=False)

        clf = HistGradientBoostingClassifier(random_state=RNG_SEED)
        X_train = feat_df[TIER0_FEATURES]
        y_train = feat_df["hit"].to_numpy()
        clf.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])

        _val_row, val_cand_idx, val_cand_hit = eval_point(side, CHOSEN_HIT_TYPE, CHOSEN_H, CHOSEN_K,
                                                            val_pools[side], high, low, close, atr, n, rng)
        val_feat_df = df.loc[val_cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        X_val = val_feat_df[TIER0_FEATURES]
        y_val = val_cand_hit.astype(int)
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")

        perm = permutation_importance(clf, X_val, y_val, scoring="roc_auc", n_repeats=20,
                                       random_state=RNG_SEED) if len(np.unique(y_val)) > 1 else None
        if perm is not None:
            perm_series = pd.Series(perm.importances_mean, index=TIER0_FEATURES).sort_values(key=np.abs, ascending=False)
            perm_std = pd.Series(perm.importances_std, index=TIER0_FEATURES)
        else:
            perm_series = pd.Series(dtype=float)
            perm_std = pd.Series(dtype=float)

        log(f"\n=== Feature analysis side={side} hit_type={CHOSEN_HIT_TYPE} H={CHOSEN_H} K={CHOSEN_K} "
            f"n_train_cand={len(cand_idx)} n_val_cand={len(val_cand_idx)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} ===")
        log("  top corr (abs, desc): " + ", ".join(f"{f}={corr[f]:+.3f}" for f in corr.index[:8]))
        if len(perm_series):
            log("  top perm-importance (VAL, desc): " + ", ".join(
                f"{f}={perm_series[f]:+.4f}(+-{perm_std[f]:.4f})" for f in perm_series.index[:8]))

        feature_analysis[side] = {
            "n_train_candidates": int(len(cand_idx)),
            "n_val_candidates": int(len(val_cand_idx)),
            "train_hit_rate": round(float(y_train.mean()), 4),
            "val_hit_rate": round(float(np.mean(y_val)), 4) if len(y_val) else None,
            "gbm_train_auc": round(float(train_auc), 4),
            "gbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "point_biserial_corr_train": {f: round(float(corr[f]), 4) for f in corr.index},
            "permutation_importance_val_mean": {f: round(float(perm_series[f]), 5) for f in perm_series.index} if len(perm_series) else {},
            "permutation_importance_val_std": {f: round(float(perm_std[f]), 5) for f in perm_series.index} if len(perm_series) else {},
        }

    report = {
        "asset": "BTCUSDT", "signal": "liquidity_sweep", "bar": "5m",
        "round": 2,
        "round_description": "HIT_TYPE x HORIZON x K 3-D grid (round 1 fixed HIT_TYPE=touch_mfe only)",
        "prior_round_script": "scripts/research_btc_liquidity_sweep_gridscreen_20260901.py",
        "prior_round_summary": {
            "hit_type_tried": "touch_mfe (only option)",
            "chosen_horizon": 15, "chosen_k": 2.0,
            "train_lift": {"bottom": 1.056, "top": 1.057},
            "val_lift": {"bottom": 1.013, "top": 1.152},
            "verdict": "grid overall weak (max lift 1.058x across all 42 touch_mfe cells; K>=3.0 mostly <1.0 on bottom); bottom TRAIN->VAL lift collapsed near 1.0 while top strengthened",
        },
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
            "touch_mae_capped": f"touch_mfe AND order-aware MAE over [i+1,touch_bar] <= {MAE_K_LOSS_MULT}*atr (K_LOSS_MULT={MAE_K_LOSS_MULT} fixed)",
            "touch_giveback_sustained": f"fast_mult=(close-based max over FAST_WINDOW=H)/atr >= K AND giveback (peak/trough-to-end retracement over FULL_WINDOW=2H) <= {GIVEBACK_CEIL} (fixed)",
        },
        "gate": {"min_train_candidates": MIN_TRAIN_CANDIDATES, "min_train_hits": MIN_TRAIN_HITS, "relaxed": False},
        "selection_rule": (f"per-HIT_TYPE argmax of min(train_lift_bottom, train_lift_top) among gate-passing "
                            f"(H,K) cells (n_cand>={MIN_TRAIN_CANDIDATES}, n_hits>={MIN_TRAIN_HITS} both sides, "
                            f"unrelaxed -- matches round 1's gate exactly); GLOBAL winner = best of the 4 family "
                            f"winners (equivalent to a flat argmax over the full 288-cell grid)"),
        "chosen_hit_type": CHOSEN_HIT_TYPE, "chosen_horizon": CHOSEN_H, "chosen_k": CHOSEN_K,
        "chosen_train_lift": {"bottom": global_chosen["lift_bottom"], "top": global_chosen["lift_top"],
                               "joint_min": global_chosen["joint_min"]},
        "chosen_val_confirmation": [val_rows_global["bottom"], val_rows_global["top"]],
        "chosen_val_joint_min": round(val_joint_global, 4) if val_joint_global is not None else None,
        "chosen_oos_bonus_check": [oos_rows_global["bottom"], oos_rows_global["top"]],
        "chosen_oos_note": ("bonus check only, NOT used for selection; scoped to the single global-chosen point "
                             "only (not repeated across the 4 leaderboard points) to limit multiple-comparison exposure"),
        "per_hit_type_leaderboard": per_hit_type_leaderboard,
        "top10_combos_overall_by_train_joint_lift": all_sorted[:10],
        "full_train_grid": grid_rows,
        "feature_analysis": feature_analysis,
        "tier0_features": TIER0_FEATURES,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_fresh_forward": "Grid-screen/feature-analysis pass (label separability check), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar is N/A=False by construction, no trade ledger exists yet.",
        "cross_asset_info_used": False,
        "cross_asset_note": "liquidity_sweep is a single-asset (BTC-only OHLC) signal by definition, no BTC-ETH cross-asset info used (matches round 1).",
        "tabpfn_training_done": False,
        "economic_cost_gate_done": False,
        "holdout_exposure_done": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str, allow_nan=False))
    log(f"\nreport saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
