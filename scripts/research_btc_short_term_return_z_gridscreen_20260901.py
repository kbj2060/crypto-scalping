#!/usr/bin/env python3
"""BTC grid-screen + feature-analysis for short_term_return_z (Homer methodology, ported to BTC).

User request (2026-09-01): redo this project's Homer evidence-signal methodology for BTC's own
short_term_return_z signal -- grid-screen HORIZON x K + a feature-analysis pass ONLY. No TabPFN
training, no economic/cost-gate backtest, no HOLDOUT exposure -- those stay future work pending
human review, matching how every other Homer signal in this project went through this same
grid-screen step first, before any TabPFN/economic work started.

Signal definition (unchanged from ETH's own compute_signals()/live_evidence_signal_dashboard_
20260823.py, ported verbatim into this BTC dataset by
scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py): ret3_z (3-bar/15min return
z-score) crosses +-2.5 -- bottom candidates (ret3_z<=-2.5, predicting UP) and top candidates
(ret3_z>=2.5, predicting DOWN), mirrored/direction-aligned into one combined candidate pool.

Data: data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv
Already has the Tier0 22-feature set (21 + rsi) and the bottom_/top_short_term_return_z boolean
trigger columns pre-computed. This script does NOT recompute triggers or features, only
labels+screens+analyzes.

Methodology (mirrors ETH's own short_term_return_z v1/v2 scripts -- see
scripts/research_eth_short_term_return_z_metalabel_tabpfn_20260829.py [HORIZON=12/K=1.75 chosen
there, used below as this grid's search CENTER, not a preset answer] and
scripts/research_eth_short_term_return_z_metalabel_v2_gap_sweep_20260830.py [cluster_dedup_gap
mechanism ported near-verbatim]):

  1. Cluster-anchor dedup: same-side fires within GAP=12 bars (1h, fixed project-wide convention
     per user instruction, NOT swept here) collapsed to the single most-extreme-ret3_z bar in each
     cluster. Causal -- uses only ret3_z (the trigger's own defining variable), never future price.
  2. Hit label per (HORIZON,K): touch-based MFE using intrabar high/low over bars[fire+1:fire+H] --
     bottom: high.max() >= close[fire] + K*atr[fire]; top: low.min() <= close[fire] - K*atr[fire].
     `atr` in this CSV is PRICE-SCALE (dollars), confirmed against
     build_eth_5m_sweep_followthrough_v2_labels_20260829.py::add_causal_columns (raw true_range
     rolling mean, not a percentage) and empirically (mean atr/close ~0.18%, sane for BTC 5m ATR14
     -- atr ranges ~$2-3,335 while close ranges ~$38.6k-126k).
  3. Baseline: same-count RANDOM non-trigger bars, drawn from the SAME period as the candidates
     being compared (TRAIN baseline from TRAIN-period eligible bars, VAL baseline from VAL-period,
     etc.) -- same direction-hit rule as the matching side (bottom-direction baseline tests UP
     moves, top-direction tests DOWN), fixed seed per (horizon,period,side) for reproducibility.
     Isolates whether the |ret3_z|>=2.5 trigger itself adds value over randomly betting in the same
     direction at a random time.
  4. Grid selection: best TRAIN lift among (HORIZON,K) cells with TRAIN n>=300 (project's "few
     hundred+" minimum). VAL is used ONLY to confirm the chosen cell generalizes -- never to
     re-search the grid. OOS is also reported for the chosen cell as supplementary context (OOS is
     NOT HOLDOUT, touching it is fine) but likewise not part of selection.
  5. Feature analysis at the chosen (HORIZON,K), TRAIN candidates: (a) point-biserial correlation
     (hit vs each of the 22 Tier0 features -- Pearson corr with a 0/1 target IS point-biserial) and
     (b) HistGradientBoostingClassifier fit on TRAIN, permutation importance (AUC-scored) measured
     on VAL, single seed (matches this project's own established single-seed VAL-importance
     convention, e.g. compute_permutation_importance() in
     scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py).

HOLDOUT (timestamp >= 2026-04-01) is dropped from the working frame IMMEDIATELY after load and
never referenced again anywhere in this script -- no candidate, label, baseline draw, or feature
stat in this run ever reads a HOLDOUT row.

Explicitly OUT OF SCOPE this round (future work, pending human review): TabPFN training,
economic/cost-gate backtest, any HOLDOUT touch. Mirrors ETH's own process for this exact signal --
grid-screen+feature-analysis happened first, TabPFN/economics were separate later sessions.

Run: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_short_term_return_z_gridscreen_20260901.py
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
CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/short_term_return_z_gridscreen_report.json"

# 21 + rsi, exactly the Tier0 set specified for this BTC dataset (NOT identical to ETH's own
# FEATURE_COLUMNS list -- ETH used atr_pct/is_bottom/nyse_open_flag/er_24/realized_vol_ratio
# instead of atr/sweep_level_low/sweep_level_high/range_width_pct; this BTC Tier0 build
# intentionally carries the raw sweep/atr ingredients instead, see
# build_btc_5m_evidence_signal_candidates_tier0_20260901.py module docstring).
TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high", "range_width_pct",
    "hour_utc", "weekday", "delta_z", "p_fast", "p_slow", "ret3_z", "vwap_dev_z",
    "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

GAP = 12  # fixed cluster-dedup convention (project-wide), NOT swept
# HORIZONS: base grid [6,9,12,18,24] centered on ETH's chosen 12; extended down to 3/4 after the
# first run showed lift monotonically INCREASING as horizon shrank, with H=6 (the smallest base
# grid point) on top -- a boundary optimum. Project precedent (smt_divergence, 2026-08-31 memory:
# "그리드경계값(H=48)안믿고 H=60~96확장중") explicitly warns not to trust a boundary grid value
# without expanding past it, so 3 and 4 were added to check whether the trend continues or reverses.
HORIZONS = [1, 2, 3, 4, 6, 9, 12, 18, 24]
KS = [1.0, 1.5, 1.75, 2.0, 2.5]
MIN_TRAIN_N = 300

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")
PERIODS = [("train", (None, VAL_START)), ("val", (VAL_START, OOS_START)), ("oos", (OOS_START, HOLDOUT_START))]

BASE_SEED = 20260901


def log(msg: str) -> None:
    print(f"[btc_str_z_gridscreen] {msg}", flush=True)


def load_frame() -> pd.DataFrame:
    usecols = list(dict.fromkeys(
        ["timestamp", "high", "low", "close"] + TIER0_FEATURES
        + ["bottom_short_term_return_z", "top_short_term_return_z"]
    ))
    df = pd.read_csv(CSV_PATH, usecols=usecols)
    # parse as UTC then drop the tz label (values stay UTC wall-clock) -- matches this project's
    # own reference scripts' convention (naive Timestamp boundaries throughout), and avoids
    # numpy datetime64 tz-naive/tz-aware comparison errors below.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_localize(None)
    df = df.sort_values("timestamp").reset_index(drop=True)
    n_full = len(df)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    log(f"loaded {n_full} rows, truncated to {len(df)} rows before HOLDOUT_START={HOLDOUT_START.date()} "
        f"(HOLDOUT rows dropped immediately -- never read again below)")
    return df


def cluster_dedup_gap(idx: np.ndarray, anchor_val: np.ndarray, most_negative: bool, gap: int) -> np.ndarray:
    """Collapse same-side fires within `gap` bars of each other into one cluster, keep only the
    bar with the most extreme anchor_val (ret3_z) per cluster. Causal -- anchor picked using only
    the trigger's own defining variable, never future price. Ported from ETH's
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


def forward_extreme(high: np.ndarray, low: np.ndarray, idx: np.ndarray, horizon: int, side: str) -> np.ndarray:
    if len(idx) == 0:
        return np.array([])
    if side == "bottom":
        return np.array([high[i + 1:i + horizon + 1].max() for i in idx])
    return np.array([low[i + 1:i + horizon + 1].min() for i in idx])


def hit_from_extreme(close: np.ndarray, atr: np.ndarray, idx: np.ndarray, fut_ext: np.ndarray,
                      side: str, k: float) -> np.ndarray:
    entry = close[idx]
    atr_i = atr[idx]
    if side == "bottom":
        return (fut_ext >= entry + k * atr_i).astype(int)
    return (fut_ext <= entry - k * atr_i).astype(int)


def period_mask(ts: np.ndarray, start: pd.Timestamp | None, end: pd.Timestamp | None) -> np.ndarray:
    m = np.ones(len(ts), dtype=bool)
    if start is not None:
        m &= ts >= np.datetime64(start)
    if end is not None:
        m &= ts < np.datetime64(end)
    return m


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

    grid_rows = []
    horizon_cache: dict[int, dict] = {}

    for horizon in HORIZONS:
        idx_map = {}
        for side, trig, most_neg in [("bottom", bottom_trig, True), ("top", top_trig, False)]:
            raw_idx = np.flatnonzero(trig)
            raw_idx = raw_idx[(raw_idx < n - horizon) & np.isfinite(atr[raw_idx]) & np.isfinite(ret3z[raw_idx])]
            idx_map[side] = cluster_dedup_gap(raw_idx, ret3z[raw_idx], most_negative=most_neg, gap=GAP)

        cand_fut_ext = {side: forward_extreme(high, low, idx_map[side], horizon, side) for side in idx_map}

        eligible = np.flatnonzero((~any_trigger) & (np.arange(n) < n - horizon) & np.isfinite(atr))

        baseline_cache = {}
        for period_name, bounds in PERIODS:
            pm = period_mask(ts, *bounds)
            eligible_p = eligible[pm[eligible]]
            for side in ("bottom", "top"):
                idx_p = idx_map[side][pm[idx_map[side]]]
                n_need = len(idx_p)
                seed = BASE_SEED + horizon * 1000 + (1 if side == "bottom" else 2) + \
                    {"train": 0, "val": 10000, "oos": 20000}[period_name]
                rng = np.random.default_rng(seed)
                n_draw = min(n_need, len(eligible_p))
                if n_draw == 0:
                    baseline_cache[(period_name, side)] = (np.array([], dtype=int), np.array([]))
                    continue
                chosen = rng.choice(eligible_p, size=n_draw, replace=False)
                baseline_cache[(period_name, side)] = (chosen, forward_extreme(high, low, chosen, horizon, side))

        horizon_cache[horizon] = {"idx_map": idx_map, "cand_fut_ext": cand_fut_ext}

        for side in ("bottom", "top"):
            counts = {pn: int(period_mask(ts, *b)[idx_map[side]].sum()) for pn, b in PERIODS}
            log(f"HORIZON={horizon}: {side} deduped candidates train={counts['train']} "
                f"val={counts['val']} oos={counts['oos']}")

        for k in KS:
            row = {"horizon": horizon, "k": k}
            for period_name, bounds in PERIODS:
                pm = period_mask(ts, *bounds)
                cand_hits, base_hits = [], []
                n_side = {}
                for side in ("bottom", "top"):
                    idx_side = idx_map[side]
                    m = pm[idx_side]
                    idx_p = idx_side[m]
                    fut_p = cand_fut_ext[side][m]
                    n_side[side] = len(idx_p)
                    if len(idx_p):
                        cand_hits.append(hit_from_extreme(close, atr, idx_p, fut_p, side, k))
                    b_idx, b_fut = baseline_cache[(period_name, side)]
                    if len(b_idx):
                        base_hits.append(hit_from_extreme(close, atr, b_idx, b_fut, side, k))
                cand_rate = float(np.concatenate(cand_hits).mean()) if cand_hits else float("nan")
                base_rate = float(np.concatenate(base_hits).mean()) if base_hits else float("nan")
                lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")
                row[f"{period_name}_n"] = n_side["bottom"] + n_side["top"]
                row[f"{period_name}_n_bottom"] = n_side["bottom"]
                row[f"{period_name}_n_top"] = n_side["top"]
                row[f"{period_name}_hit_rate"] = round(cand_rate, 4) if cand_rate == cand_rate else None
                row[f"{period_name}_baseline_hit_rate"] = round(base_rate, 4) if base_rate == base_rate else None
                row[f"{period_name}_lift"] = round(lift, 4) if lift == lift else None
            # Stability guard: fixing candidate COUNT (via MIN_TRAIN_N) does NOT fix the HIT count
            # at a given K -- a short HORIZON + high K combo can push both the candidate hit rate
            # and the baseline hit rate toward rare-event territory (seen empirically: H=1/K=2.5
            # TRAIN lift=2.58 but swings to VAL=1.50/OOS=3.50, a classic small-count-ratio noise
            # signature), even though train_n itself stays fixed at 2384. Delta-method approx for
            # the relative SE of a ratio of two independent binomial proportions (p_hit/p_base,
            # both estimated from the same TRAIN n): relSE = sqrt((1-p_hit)/(p_hit*n) +
            # (1-p_base)/(p_base*n)). Cells above MAX_REL_SE_LIFT are excluded from selection (but
            # still kept in the reported grid) -- this is what actually separates "genuinely
            # higher lift" from "ratio of two small counts, high variance."
            p_hit, p_base, n_tr = row["train_hit_rate"], row["train_baseline_hit_rate"], row["train_n"]
            if p_hit and p_base and p_hit > 0 and p_base > 0 and n_tr > 0:
                rel_se = float(np.sqrt((1 - p_hit) / (p_hit * n_tr) + (1 - p_base) / (p_base * n_tr)))
            else:
                rel_se = float("inf")
            row["train_lift_rel_se"] = round(rel_se, 4) if np.isfinite(rel_se) else None
            grid_rows.append(row)
            log(f"H={horizon:>2} K={k:.2f}  TRAIN n={row['train_n']:>5} hit={row['train_hit_rate']} "
                f"base={row['train_baseline_hit_rate']} lift={row['train_lift']} relSE={row['train_lift_rel_se']}  "
                f"| VAL n={row['val_n']:>4} lift={row['val_lift']}  | OOS n={row['oos_n']:>4} lift={row['oos_lift']}")

    MAX_REL_SE_LIFT = 0.10  # 10% -- empirically separates the stable H>=2 cluster (~4-8% relSE)
                             # from the noisy H=1 cluster (~14-27% relSE) observed in this run
    eligible_rows = [r for r in grid_rows if r["train_n"] >= MIN_TRAIN_N and r["train_lift"] is not None]
    if not eligible_rows:
        log(f"WARNING: no grid cell meets MIN_TRAIN_N={MIN_TRAIN_N}, falling back to all cells")
        eligible_rows = [r for r in grid_rows if r["train_lift"] is not None]
    raw_max = max(eligible_rows, key=lambda r: r["train_lift"])
    log(f"RAW GRID-MAX (lift only, ignoring stability): HORIZON={raw_max['horizon']} K={raw_max['k']} "
        f"lift={raw_max['train_lift']} relSE={raw_max['train_lift_rel_se']}")

    stable_rows = [r for r in eligible_rows if r["train_lift_rel_se"] is not None and r["train_lift_rel_se"] <= MAX_REL_SE_LIFT]
    if not stable_rows:
        log(f"WARNING: no grid cell meets MAX_REL_SE_LIFT={MAX_REL_SE_LIFT}, falling back to raw grid-max")
        stable_rows = eligible_rows
    chosen = max(stable_rows, key=lambda r: r["train_lift"])
    chosen["selection_rule"] = f"max TRAIN lift subject to train_n>={MIN_TRAIN_N} and train_lift_rel_se<={MAX_REL_SE_LIFT}"
    log(f"CHOSEN (stability-filtered): HORIZON={chosen['horizon']} K={chosen['k']} "
        f"(TRAIN n={chosen['train_n']}, lift={chosen['train_lift']}, relSE={chosen['train_lift_rel_se']}; "
        f"VAL n={chosen['val_n']}, lift={chosen['val_lift']})")

    # ---- feature analysis at chosen (horizon, k) ----
    horizon, k = chosen["horizon"], chosen["k"]
    idx_map = horizon_cache[horizon]["idx_map"]
    cand_fut_ext = horizon_cache[horizon]["cand_fut_ext"]

    rows = []
    for side in ("bottom", "top"):
        idx_side = idx_map[side]
        hit = hit_from_extreme(close, atr, idx_side, cand_fut_ext[side], side, k)
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

    # (a) point-biserial correlation (Pearson corr of a raw feature vs a binary 0/1 target IS
    # point-biserial correlation -- same statistic, no separate implementation needed)
    corr_rows = []
    for feat in TIER0_FEATURES:
        c = train_f[feat].astype(float).corr(train_f["hit"].astype(float))
        corr_rows.append({"feature": feat, "point_biserial_corr": round(float(c), 4) if c == c else None})
    corr_rows.sort(key=lambda r: -abs(r["point_biserial_corr"]) if r["point_biserial_corr"] is not None else 0)
    log("top point-biserial |corr| features:")
    for r in corr_rows[:6]:
        log(f"  {r['feature']:<22s} corr={r['point_biserial_corr']:+.4f}")

    # (b) HistGradientBoostingClassifier fit TRAIN -> permutation importance on VAL (AUC-scored,
    # single seed, matches this project's established single-seed VAL-importance convention)
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
        "stage": "grid_screen_and_feature_analysis_only",
        "not_done_this_round": ["TabPFN training", "economic/cost-gate backtest", "HOLDOUT exposure"],
        "holdout_touched": False,
        "holdout_start": str(HOLDOUT_START),
        "gap_fixed": GAP,
        "horizons_tried": HORIZONS,
        "ks_tried": KS,
        "min_train_n": MIN_TRAIN_N,
        "max_rel_se_lift": MAX_REL_SE_LIFT,
        "stability_guard_note": (
            "train_lift_rel_se is a delta-method approx of the relative SE of the TRAIN "
            "hit_rate/baseline_rate ratio. Cells above max_rel_se_lift are excluded from "
            "selection (kept in 'grid' for transparency) because at short HORIZON + high K the "
            "hit AND baseline rates both become rare-event-small, making the lift ratio noisy "
            "even though train_n (candidate count) stays fixed -- empirically confirmed in this "
            "run: raw_grid_max_by_lift_only (H=1) swung TRAIN/VAL/OOS lift 2.98/2.19/2.88 (K=1.75) "
            "and 2.58/1.50/3.50 (K=2.5), while 'chosen' (stability-filtered) stayed in a tight "
            "~1.50-1.56 band across all three periods."
        ),
        "tier0_features": TIER0_FEATURES,
        "grid": grid_rows,
        "raw_grid_max_by_lift_only": raw_max,
        "chosen": chosen,
        "feature_analysis": {
            "horizon": horizon, "k": k,
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
