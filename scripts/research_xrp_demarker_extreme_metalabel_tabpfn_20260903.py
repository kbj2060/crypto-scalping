#!/usr/bin/env python3
"""BTC port of the `demarker_extreme` evidence signal: HORIZON x K grid-screen (TRAIN lift vs
random-bar baseline, confirmed on VAL) + TabPFN meta-label panel, combined into ONE script per
user instruction -- ETH's own reference point for this exact signal is already well-established
(H=8/GAP=12/K=0.70, VAL/OOS/HOLDOUT AUC 0.7527/0.7157/0.7464, this project's all-time-best
classification result, docs/homer/README.md "후보 풀" section), so this BTC port does not need the
full 2-round grid-then-TabPFN back-and-forth the other 5 BTC signals went through.

Trigger (verbatim from live_evidence_signal_dashboard_20260823.py lines ~519-529, pure OHLC,
DeMarker-14):
    dem_up_move = high.diff()
    dem_down_move = low.shift(1) - low
    dem_de_max = dem_up_move.clip(lower=0.0).fillna(0.0)
    dem_de_min = dem_down_move.clip(lower=0.0).fillna(0.0)
    dem_sma_max = dem_de_max.rolling(14, min_periods=14).mean()
    dem_sma_min = dem_de_min.rolling(14, min_periods=14).mean()
    dem = dem_sma_max / (dem_sma_max + dem_sma_min)   # NaN-safe divide
    bottom_demarker_extreme = dem <= 0.10
    top_demarker_extreme = dem >= 0.90
`compute_demarker()` is imported verbatim (not reimplemented) from
research_eth_demarker_evidence_signal_lift_check_20260831.py, which live_evidence_signal_
dashboard_20260823.py's own inline copy cites as its source of truth -- confirmed byte-identical
formula by reading both files directly.

Data: data/labels/xrp_5m_evidence_signal_candidates_20260903/btc_5m_evidence_signal_candidates_
tier0.csv (built by scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py, 277,191
rows, 2024-01-01..2026-08-20, 5m BTCUSDT) -- read-only here (NOT extended/overwritten: this file
is shared with concurrently-running sibling BTC signal-porting agents this session, confirmed via
`git hash-object` that the server's copy is already byte-identical to this local one, so it does
not need pushing either). `dem` plus the handful of genuinely-missing canonical features
(atr_pct/nyse_open_flag/er_24/realized_vol_ratio) are computed in-memory here, formulas ported
verbatim from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame
-- same pattern research_btc_taker_delta_climax_metalabel_tabpfn_20260901.py already established
for this Tier0 CSV.

Methodology, 2 phases:

  Phase A -- HORIZON x K grid screen (step 2 of the task), RAW (non-deduped) fires, OOS/HOLDOUT
  never touched here:
    # ⚠️2026-09-03 2차: K를 4.0까지 넓혔더니 이번엔 **HORIZON=6이 하단 경계**에서 선택됐다.
# 경계 경고가 뜨면 그 방향으로 넓히는 게 이 저장소의 규칙이다(README 5.6). 아래로 확장한다.
HORIZON_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20], # ⚠️2026-09-03 XRP 1차 실행에서 **K=2.0이 격자 상단 경계**에서 선택돼 스크립트 자신이 경고를
# 냈다("K=2.0 sits at the grid EDGE -- extend the grid before treating this as final").
# 이 저장소는 같은 실수로 ETH demarker의 진짜 최적값을 놓친 전례가 있다(README 5.6).
# 위쪽으로 확장해 재탐색한다. 아래쪽은 1차에서 이미 탐색됐고 선택되지 않았다.
K_GRID = [0.40, 0.55, 0.70, 0.85, 1.00, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0] --
    both centered on ETH's own H=8/K=0.70 with room on every side (this project has a standing
    "don't trust a grid boundary" lesson: ETH's own K search for this exact signal first wrongly
    converged near K=2.0 by assuming a hit-rate-far-from-50/50 split must be unstable, and only
    found the true optimum K=0.70 by extending the grid downward -- docs/homer/README.md 2026-08-31
    HORIZON x GAP x K entry). hit = touch-based MFE, intrabar high/low within bars[fire+1:fire+H+1]
    reaching K*atr_pct (fractional ATR-at-fire, NOT raw price ATR -- matches
    research_eth_kalman_demarker_gridscreen_20260831.py::build_fires's own `peak = move_pct/atr_pct
    >= K` convention for this same signal on ETH). For each (H,K): TRAIN pooled lift = fired hit
    rate / random-bar baseline hit rate (baseline draws the same COUNT of random non-trigger TRAIN
    bars, same K/H/direction rule, isolates whether the dem<=0.10/>=0.90 extremity threshold itself
    adds value); VAL pooled hit rate + baseline + lift computed the same way for every cell (not
    just a chosen-cell-only check).
    Eligibility gate: (a) n_train >= 200 per side (this project's established floor, trivially
    satisfied here -- raw TRAIN fires are ~2,400-2,750/side, see module-level constants) and (b) VAL
    pooled minority-hit-class count >= 20 (this project's OTHER hard-won lesson for this exact
    signal: judge K by absolute minority-class sample count in VAL, not by closeness to a 50/50
    split). Selection metric: min(TRAIN lift, VAL lift), NOT a TRAIN-only argmax -- see
    select_horizon_k()'s own docstring for why (a dry run of pure TRAIN-argmax on this exact grid
    picked a cell VAL flatly contradicted, the same single-split-max overfitting this project's own
    `min(VAL,OOS)` convention elsewhere already guards against), plus a documented, non-blind
    tie-break toward ETH's own H=8/K=0.70 center when a candidate is statistically indistinguishable
    from the mechanical argmax. Grid-edge check: if the chosen H or K sits at K_GRID[0]/[-1] or
    HORIZON_GRID[0]/[-1], this is flagged loudly (not auto-extended -- same "flagged for a human to
    notice" convention as research_eth_kalman_demarker_ksweep_20260831.py).

  Phase B -- canonical feature build at the chosen (H,K) (step 3), cluster-dedup GAP=6 (fixed,
  given directly by the task -- NOT re-derived from ETH's own GAP=12 for this signal, and GAP
  itself is not grid-searched here, only H and K are), keep the bar with the most extreme `dem`
  value per cluster (closest to 0 for bottom, closest to 1 for top) -- `cluster_dedup()` is the
  same generalized-extremeness form research_eth_kalman_demarker_gridscreen_20260831.py already
  uses for this exact signal (not taker's delta_z-specific version). Canonical features = ETH's own
  exact Tier0-23 FEATURE_COLUMNS (research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py) plus
  `dem` itself as a 24th feature -- matches research_eth_kalman_demarker_gridscreen_20260831.py /
  _ksweep_ / _tabpfn_confirm_'s own `FEATURE_COLUMNS + ["dem"]` convention for this identical
  signal on ETH.

  Phase C -- TabPFN (step 4): run_tabpfn_panel / evaluate / compute_permutation_importance ported
  VERBATIM (identical code) from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py.
  SEEDS = [20260829, 141592, 271828, 577215] (same 4 seeds used throughout this project's Homer
  lineage). Splits = this repo's Fresh-Forward default (CLAUDE.md): TRAIN<2025-09-01,
  VAL 2025-09-01..2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT>=2026-04-01 (single-touch,
  evaluated once, only after H/K are already fixed from Phase A/TRAIN+VAL-only screening).

NOT done here (explicitly out of scope per task): no autocorrelation mean-reversion regime gate
(ETH tested this exact idea for this exact signal and found momentum-regime fires predict BETTER
than mean-reversion-regime fires, the opposite of the intuitive hypothesis -- docs/homer/README.md
2026-08-31 "자기상관 레짐게이트" entry -- so deliberately not reproduced here), no economics/cost-gate
backtest, no dashboard wiring, no HOLDOUT re-touch.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) under a system-wide flock (shared
8GB GPU across concurrently-running sibling signal-research agents this session) -- see
scripts/ops/handoff.sh push before executing remotely.

## ⚠️XRP 포팅 (2026-09-03)

`research_btc_<signal>_metalabel_tabpfn_20260901.py`의 **자산 상수만** 바꾼 포팅.
격자(HORIZON x K)와 TabPFN 절차는 그대로 재탐색한다 -- 자산이 바뀌면 최적 셀도 바뀐다.
이 두 신호는 ETH·BTC 모두 **plain touch**(`peak >= K`) HIT 정의를 쓰므로 HIT_TYPE은 안 쓴다.
절차: `docs/homer/evidence_signal_new_coin_port_protocol.md`
"""
from __future__ import annotations

import json
import sys
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

from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402

TIER0_PATH = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903/xrp_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/xrp_5m_evidence_signal_candidates_20260903"
REPORT_PATH = OUT_DIR / "demarker_extreme_tabpfn_report.json"
FEATURES_CSV_PATH = OUT_DIR / "btc_5m_demarker_extreme_metalabel_features.csv"

START = pd.Timestamp("2024-01-01")
VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")

# ⚠️2026-09-03 2차: K를 4.0까지 넓혔더니 이번엔 HORIZON=6이 **하단 경계**였다.
# 경계 경고가 뜨면 그 방향으로 넓히는 게 이 저장소 규칙이다(README 5.6).
HORIZON_GRID = [2, 3, 4, 5, 6, 8, 10, 12, 16, 20]
K_GRID = [0.40, 0.55, 0.70, 0.85, 1.00, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
CLUSTER_GAP = 6  # fixed by task instruction (ETH's own final choice for this signal was GAP=12,
                 # not re-derived here -- GAP is not part of this script's grid search)

MIN_TRAIN_CANDIDATES = 200  # per side, this project's established floor (BTC taker round-1 gridscreen)
MIN_VAL_MINORITY = 20       # pooled minority-hit-class count in VAL -- the "judge by absolute count,
                             # not 50/50 closeness" lesson from this signal's own ETH K-search history

ETH_REFERENCE_HORIZON = 8   # ETH's own H=8/GAP=12/K=0.70 (docs/homer/README.md 후보 풀 section) --
ETH_REFERENCE_K = 0.70      # used only as a tie-break preference below, never forced blindly
NEAR_BEST_TOLERANCE = 0.02  # absolute min(TRAIN,VAL)-lift tolerance for the tie-break, see
                             # select_horizon_k()'s docstring

RNG_SEED = 20260901
SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as every other TabPFN run in this lineage

# ETH's exact Tier0-23 FEATURE_COLUMNS (research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py)
# plus `dem` itself as a 24th feature (matches ETH's own demarker gridscreen/ksweep/confirm convention).
FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi", "dem",
]


def log(msg: str) -> None:
    print(f"[xrp_demarker_extreme_tabpfn] {msg}", flush=True)


def load_tier0() -> pd.DataFrame:
    df = pd.read_csv(TIER0_PATH, parse_dates=["timestamp"])
    df["timestamp"] = df["timestamp"].dt.tz_localize(None)  # tz-aware UTC -> naive UTC, matches
                                                              # research_btc_taker_delta_climax_
                                                              # metalabel_tabpfn_20260901.py's own convention
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in XRP Tier0 rows"
    return df


def add_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add dem + the handful of columns not already in Tier0. Formulas ported verbatim from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame
    (is_bottom is added later, per-fire, not per-bar)."""
    frame["dem"] = compute_demarker(frame["high"], frame["low"]).to_numpy()

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


# ---------------------------------------------------------------------------
# Phase A: HORIZON x K grid screen (raw, non-deduped fires; TRAIN lift selection, VAL confirm-only)
# ---------------------------------------------------------------------------

def forward_extremes(high: np.ndarray, low: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """fwd_high_max[i] = max(high[i+1:i+horizon+1]); fwd_low_min[i] = min(low[i+1:i+horizon+1]).
    Vectorized (verbatim idiom from research_btc_taker_delta_climax_gridscreen_20260901.py)."""
    fwd_high_max = pd.Series(high).rolling(horizon, min_periods=horizon).max().shift(-horizon).to_numpy()
    fwd_low_min = pd.Series(low).rolling(horizon, min_periods=horizon).min().shift(-horizon).to_numpy()
    return fwd_high_max, fwd_low_min


def random_baseline_hit(rng: np.random.Generator, pool_idx: np.ndarray, n_draw: int,
                         fwd_ext: np.ndarray, close: np.ndarray, atr_pct: np.ndarray,
                         k: float, direction: str) -> float:
    """Same COUNT of random non-trigger TRAIN bars, same mirrored direction/K*atr_pct threshold --
    isolates whether the dem extremity threshold itself adds lift over an unconditional base rate."""
    if n_draw <= 0 or len(pool_idx) < n_draw:
        return float("nan")
    samp = rng.choice(pool_idx, size=n_draw, replace=False)
    if direction == "up":
        move_pct = (fwd_ext[samp] - close[samp]) / close[samp]
    else:
        move_pct = (close[samp] - fwd_ext[samp]) / close[samp]
    hit = move_pct >= k * atr_pct[samp]
    return float(hit.mean())


def run_grid_screen(frame: pd.DataFrame) -> pd.DataFrame:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr_pct = frame["atr_pct"].to_numpy(dtype=float)
    ts_col = frame["timestamp"]

    bottom_trig = (frame["dem"] <= 0.10).to_numpy()
    top_trig = (frame["dem"] >= 0.90).to_numpy()
    any_trig = bottom_trig | top_trig
    assert not (bottom_trig & top_trig).any(), "bottom/top should be mutually exclusive"

    train_mask = (ts_col < VAL_START).to_numpy()
    val_mask = ((ts_col >= VAL_START) & (ts_col < OOS_START)).to_numpy()
    started = (ts_col >= START).to_numpy()
    finite_ok = np.isfinite(close) & np.isfinite(atr_pct) & (atr_pct > 0)

    log(f"raw trigger counts: bottom={int(bottom_trig.sum())} top={int(top_trig.sum())} "
        f"(TRAIN bottom={int((bottom_trig & train_mask).sum())} top={int((top_trig & train_mask).sum())}, "
        f"VAL bottom={int((bottom_trig & val_mask).sum())} top={int((top_trig & val_mask).sum())})")

    grid_rows = []
    master_rng = np.random.default_rng(RNG_SEED)

    for horizon in HORIZON_GRID:
        fwd_high_max, fwd_low_min = forward_extremes(high, low, horizon)
        valid_fwd = np.isfinite(fwd_high_max) & np.isfinite(fwd_low_min) & finite_ok & started

        bottom_idx_all = np.flatnonzero(bottom_trig & valid_fwd)
        top_idx_all = np.flatnonzero(top_trig & valid_fwd)
        pool_idx_train = np.flatnonzero(train_mask & (~any_trig) & valid_fwd)

        for k in K_GRID:
            bottom_move = (fwd_high_max[bottom_idx_all] - close[bottom_idx_all]) / close[bottom_idx_all]
            top_move = (close[top_idx_all] - fwd_low_min[top_idx_all]) / close[top_idx_all]
            bottom_hit_all = bottom_move >= k * atr_pct[bottom_idx_all]
            top_hit_all = top_move >= k * atr_pct[top_idx_all]

            b_train_m, b_val_m = train_mask[bottom_idx_all], val_mask[bottom_idx_all]
            t_train_m, t_val_m = train_mask[top_idx_all], val_mask[top_idx_all]
            n_b_train, n_t_train = int(b_train_m.sum()), int(t_train_m.sum())
            n_b_val, n_t_val = int(b_val_m.sum()), int(t_val_m.sum())

            b_train_hitrate = float(bottom_hit_all[b_train_m].mean()) if n_b_train else float("nan")
            t_train_hitrate = float(top_hit_all[t_train_m].mean()) if n_t_train else float("nan")
            b_val_hitrate = float(bottom_hit_all[b_val_m].mean()) if n_b_val else float("nan")
            t_val_hitrate = float(top_hit_all[t_val_m].mean()) if n_t_val else float("nan")

            pooled_train_n = n_b_train + n_t_train
            pooled_val_n = n_b_val + n_t_val
            n_train_hits = int(bottom_hit_all[b_train_m].sum() + top_hit_all[t_train_m].sum())
            n_val_hits = int(bottom_hit_all[b_val_m].sum() + top_hit_all[t_val_m].sum())
            pooled_train_hit = n_train_hits / max(pooled_train_n, 1)
            pooled_val_hit = n_val_hits / max(pooled_val_n, 1)
            val_minority = min(n_val_hits, pooled_val_n - n_val_hits) if pooled_val_n else 0

            b_base = random_baseline_hit(master_rng, pool_idx_train, n_b_train, fwd_high_max, close, atr_pct, k, "up")
            t_base = random_baseline_hit(master_rng, pool_idx_train, n_t_train, fwd_low_min, close, atr_pct, k, "down")
            pooled_base = ((b_base * n_b_train + t_base * n_t_train) / pooled_train_n
                           if pooled_train_n and np.isfinite(b_base) and np.isfinite(t_base) else float("nan"))
            lift_pooled = pooled_train_hit / pooled_base if np.isfinite(pooled_base) and pooled_base > 0 else float("nan")

            # VAL confirmation baseline: same per-side draw-and-weighted-average as the TRAIN
            # baseline above, just drawn from the VAL non-trigger pool -- descriptive only, never
            # used for cell selection (selection is TRAIN-lift-only per task).
            val_pool_idx = np.flatnonzero(val_mask & (~any_trig) & valid_fwd)
            b_base_val = random_baseline_hit(master_rng, val_pool_idx, n_b_val, fwd_high_max, close, atr_pct, k, "up")
            t_base_val = random_baseline_hit(master_rng, val_pool_idx, n_t_val, fwd_low_min, close, atr_pct, k, "down")
            val_base = ((b_base_val * n_b_val + t_base_val * n_t_val) / pooled_val_n
                        if pooled_val_n and np.isfinite(b_base_val) and np.isfinite(t_base_val) else float("nan"))
            lift_val = pooled_val_hit / val_base if np.isfinite(val_base) and val_base > 0 else float("nan")

            grid_rows.append({
                "horizon": horizon, "k": k,
                "n_train_bottom": n_b_train, "n_train_top": n_t_train,
                "n_val_bottom": n_b_val, "n_val_top": n_t_val,
                "val_minority_count": val_minority,
                "train_hitrate_pooled": round(pooled_train_hit, 4),
                "train_baseline_pooled": round(pooled_base, 4) if np.isfinite(pooled_base) else None,
                "lift_train": round(lift_pooled, 4) if np.isfinite(lift_pooled) else None,
                "val_hitrate_pooled": round(pooled_val_hit, 4),
                "val_baseline_pooled": round(val_base, 4) if np.isfinite(val_base) else None,
                "lift_val": round(lift_val, 4) if np.isfinite(lift_val) else None,
                "eligible": bool(n_b_train >= MIN_TRAIN_CANDIDATES and n_t_train >= MIN_TRAIN_CANDIDATES
                                 and val_minority >= MIN_VAL_MINORITY),
            })
            log(f"H={horizon:>3d} K={k:.2f}  TRAIN n(bot/top)={n_b_train}/{n_t_train} hit={pooled_train_hit:.4f} "
                f"base={pooled_base:.4f} lift={lift_pooled:.3f}x  |  VAL n={pooled_val_n}(minority={val_minority}) "
                f"hit={pooled_val_hit:.4f} base={val_base:.4f} lift={lift_val:.3f}x")

    return pd.DataFrame(grid_rows)


def select_horizon_k(grid_df: pd.DataFrame) -> tuple[int, float, dict]:
    """Selection metric: min(TRAIN lift, VAL lift), NOT a pure TRAIN-only argmax. This project has
    a repeated, explicit precedent against single-split-max selection (e.g.
    research_eth_kalman_demarker_gridscreen_20260831.py's own `min(VAL,OOS)` rule, "the
    volume_wick_climax lesson"). A dry run of pure TRAIN-lift argmax on this exact grid picked
    H=8/K=1.5 (TRAIN lift 1.135x) which VAL then flatly contradicted (VAL lift 0.866x, WORSE than
    random) -- exactly the TRAIN-only overfitting this stability metric guards against. Applied
    here for the same reason, even though the task's own phrasing ("grid search on TRAIN lift,
    confirm on VAL") could be read as TRAIN-only -- "confirm on VAL" is read here as VAL needing to
    actually agree, not just be reported alongside a TRAIN-only pick it contradicts.

    Tie-break: this quick lift-vs-random-baseline screen is noisy (single-draw baselines) and only
    tests dem's own marginal extremity effect, not the full 24-feature interaction TabPFN actually
    uses downstream -- not powerful enough to justify chasing a noise-level "improvement" away from
    ETH's own extensively-validated center for this identical signal (H=8/K=0.70). Among cells
    statistically indistinguishable from the mechanical argmax (within NEAR_BEST_TOLERANCE), prefer
    ETH's own center if it is one of them -- maximizes cross-asset comparability without discarding
    the screen's actual job (it still vetoes ETH's center if that cell is NOT near-best)."""
    eligible = grid_df[grid_df["eligible"]].dropna(subset=["lift_train", "lift_val"]).copy()
    if eligible.empty:
        log("  WARNING: no grid cell satisfied both eligibility floors -- falling back to full grid")
        eligible = grid_df.dropna(subset=["lift_train", "lift_val"]).copy()

    eligible["min_lift"] = eligible[["lift_train", "lift_val"]].min(axis=1)
    best_min_lift = float(eligible["min_lift"].max())
    near_best = eligible[eligible["min_lift"] >= best_min_lift - NEAR_BEST_TOLERANCE]
    eth_center_row = near_best[(near_best["horizon"] == ETH_REFERENCE_HORIZON) & (near_best["k"] == ETH_REFERENCE_K)]
    if not eth_center_row.empty:
        best_row = eth_center_row.iloc[0]
        log(f"  ETH's own center H={ETH_REFERENCE_HORIZON}/K={ETH_REFERENCE_K} is within "
            f"{NEAR_BEST_TOLERANCE} of the mechanical min(TRAIN,VAL)-lift argmax ({best_min_lift:.4f}) "
            f"-- adopted over chasing a noise-level 'improvement'")
    else:
        best_row = eligible.loc[eligible["min_lift"].idxmax()]
        log("  ETH's own center is NOT near the mechanical argmax -- using the argmax cell as-is")
    chosen_horizon, chosen_k = int(best_row["horizon"]), float(best_row["k"])

    edge_flags = []
    if chosen_horizon in (HORIZON_GRID[0], HORIZON_GRID[-1]):
        edge_flags.append(f"HORIZON={chosen_horizon} sits at the grid EDGE ({HORIZON_GRID})")
    if chosen_k in (K_GRID[0], K_GRID[-1]):
        edge_flags.append(f"K={chosen_k} sits at the grid EDGE ({K_GRID})")
    for msg in edge_flags:
        log(f"  WARNING: {msg} -- extend the grid before treating this as final if revisited")

    log(f"CHOSEN: HORIZON={chosen_horizon} K={chosen_k} (TRAIN lift={best_row['lift_train']:.3f}x, "
        f"VAL lift={best_row['lift_val']:.3f}x, min={best_row['min_lift']:.3f}x, "
        f"VAL minority n={int(best_row['val_minority_count'])})")
    return chosen_horizon, chosen_k, {"edge_flags": edge_flags, "best_row": best_row.to_dict()}


# ---------------------------------------------------------------------------
# Phase B: canonical fires at chosen (H,K), cluster-dedup GAP=6
# ---------------------------------------------------------------------------

def cluster_dedup(idx: np.ndarray, extremeness: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    """Collapse same-side fires within `gap` bars into one cluster, keep only the bar with the most
    extreme `dem` per cluster (closest to 0 for bottom [most_negative=True -> idxmin], closest to 1
    for top [most_negative=False -> idxmax]). Causal (uses only dem, never future price). Same
    generalized-extremeness form as research_eth_kalman_demarker_gridscreen_20260831.py::cluster_dedup."""
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


def build_final_fires(frame: pd.DataFrame, horizon: int, k: float, gap: int) -> tuple[pd.DataFrame, dict]:
    high = frame["high"].to_numpy(dtype=float)
    low = frame["low"].to_numpy(dtype=float)
    close = frame["close"].to_numpy(dtype=float)
    atr_pct = frame["atr_pct"].to_numpy(dtype=float)
    dem = frame["dem"].to_numpy(dtype=float)
    ts_all = frame["timestamp"].to_numpy()
    n = len(frame)

    bottom_trig = (frame["dem"] <= 0.10).to_numpy()
    top_trig = (frame["dem"] >= 0.90).to_numpy()

    rows = []
    dedup_stats = {}
    for side, trig in [("bottom", bottom_trig), ("top", top_trig)]:
        idx = np.flatnonzero(trig)
        idx = idx[(idx < n - horizon) & (ts_all[idx] >= np.datetime64(START))]
        idx_before = len(idx)
        idx = cluster_dedup(idx, dem[idx], most_negative=(side == "bottom"), gap=gap)
        log(f"  {side}: {idx_before} raw fires -> {len(idx)} after cluster-anchor dedup (GAP={gap})")
        dedup_stats[side] = {"raw": int(idx_before), "deduped": int(len(idx))}

        entry = close[idx]
        a = atr_pct[idx]
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


# ---------------------------------------------------------------------------
# Phase C: TabPFN (verbatim port from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py)
# ---------------------------------------------------------------------------

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

    log("loading XRP Tier0 CSV...")
    frame = load_tier0()
    log(f"{len(frame)} bars loaded, {frame['timestamp'].min()}..{frame['timestamp'].max()}")

    log("computing dem (DeMarker-14) + adding missing canonical features "
        "(atr_pct/nyse_open_flag/er_24/realized_vol_ratio)...")
    frame = add_missing_features(frame)

    log("=== Phase A: HORIZON x K grid screen (TRAIN lift vs random-bar baseline, VAL confirm) ===")
    grid_df = run_grid_screen(frame)
    chosen_horizon, chosen_k, selection_meta = select_horizon_k(grid_df)

    log(f"=== Phase B: canonical fires @ H={chosen_horizon} K={chosen_k} GAP={CLUSTER_GAP} ===")
    fires, dedup_stats = build_final_fires(frame, chosen_horizon, chosen_k, CLUSTER_GAP)
    n_before = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)
    log(f"{len(fires)}/{n_before} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

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

    log("=== Phase C: TabPFN -- VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}  "
        f"acc {val_result['accuracy_mean']:.4f}  bal_acc {val_result['balanced_accuracy_mean']:.4f}")

    log("=== TabPFN -- OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}  "
        f"acc {oos_result['accuracy_mean']:.4f}  bal_acc {oos_result['balanced_accuracy_mean']:.4f}")

    log("=== TabPFN -- RESERVED HOLDOUT evaluation (2026-04-01~latest, single-touch, TRAIN-fit, 4 seeds) ===")
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
        "signal": "demarker_extreme",
        "asset": "BTCUSDT",
        "trigger_formula": "DeMarker-14, bottom: dem<=0.10, top: dem>=0.90 (verbatim from "
                            "live_evidence_signal_dashboard_20260823.py, pure OHLC)",
        "grid_screen": {
            "horizon_grid": HORIZON_GRID, "k_grid": K_GRID,
            "min_train_candidates_per_side": MIN_TRAIN_CANDIDATES,
            "min_val_minority_count": MIN_VAL_MINORITY,
            "chosen_horizon": chosen_horizon, "chosen_k": chosen_k,
            "cluster_gap_final_build": CLUSTER_GAP,
            "edge_flags": selection_meta["edge_flags"],
            "full_grid": grid_df.to_dict(orient="records"),
        },
        "label_definition": (
            f"touch-based MFE, intrabar high/low within bars[fire+1:fire+{chosen_horizon}+1] "
            f"reaching {chosen_k}*atr_pct_at_fire, cluster-anchor deduped (GAP={CLUSTER_GAP} bars, "
            "keep most-extreme-dem bar per cluster)"
        ),
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total": int(len(fires)),
        "dedup_stats": dedup_stats,
        "fired_signal_hit_rate_pooled": fire_hit_rate,
        "n_train": len(train), "n_val": len(val), "n_oos": len(oos), "n_holdout": len(holdout),
        "val": val_result,
        "oos": oos_result,
        "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "eth_own_result_for_comparison": {
            "note": "same signal (demarker_extreme), ETH's own H=8/GAP=12/K=0.70 result, this "
                    "project's all-time-best classification AUC -- docs/homer/README.md 후보 풀 section",
            "horizon": 8, "gap": 12, "k": 0.70,
            "val_auc": 0.7527, "oos_auc": 0.7157, "holdout_auc": 0.7464,
        },
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
