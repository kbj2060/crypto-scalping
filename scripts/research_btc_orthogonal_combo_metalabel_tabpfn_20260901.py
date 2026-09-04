#!/usr/bin/env python3
"""Meta-labeling for orthogonal_combo -- BTC port, FINAL TabPFN phase (2026-09-01).

Project Homer flagship signal (#6, historically ETH's #1-lift signal), BTC port. Two prior rounds
already ran on this repo's devmachine (no GPU needed, plain HistGBM/lift screening):
  - Round 1 (scripts/research_btc_orthogonal_combo_gridscreen_20260901.py): HORIZON x K grid,
    single touch-based-MFE HIT definition, flagged OOS lift dropping below 1.0 as a concern.
  - Round 2 (scripts/research_btc_orthogonal_combo_gridscreen_hittype_20260901.py, doc:
    docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md): added
    HIT_TYPE (4 families) as a 3rd grid dimension to test whether a stricter/persistence-aware hit
    definition fixes the OOS instability. It does NOT -- OOS lift bounces 0.556-1.148x across
    adjacent (HIT_TYPE,H,K) cells with no clean pattern; the best-supported explanation is BTC's
    small candidate count (76-131/side in OOS), not a flaw in the HIT definition itself. The
    round-2 "global winner" (touch_giveback_sustained, H=8,K=3.0) was explicitly distrusted --
    only 2-5 OOS hits, TRAIN gate (n_hits>=30) not remotely met OOS.

THIS ROUND uses the most ROBUST (largest-sample) touch-based point instead of that distrusted
winner: touch_mfe (pure touch, no persistence check), H=8, K=2.0 -- round 2's own per-HIT_TYPE
leaderboard entry for touch_mfe (TRAIN lift 1.505/VAL 1.318/OOS 1.148, OOS hit sample 49
bottom/31 top, by far the best-supported OOS>1 point in that round). Label formula (entry=close[i],
atr=atr[i] the CSV's own dollar-ATR column, same convention round 1/2 used for touch_mfe):
    bottom: hit=1 if high[i+1:i+9].max() >= entry + 2.0*atr else 0
    top:    hit=1 if low[i+1:i+9].min()  <= entry - 2.0*atr else 0
Plain binary hit/miss -- NOT the ETH orthogonal_combo script's "exclude-middle" K_lo/K_hi
refinement (that was a fix for a documented ETH-ONLY label pathology, a NO_HIT population
concentrated just below one threshold; round 2's clean touch_mfe binary label has no such
documented problem here and this round's task spec explicitly says not to port that fix).

Methodology template (ported as closely as possible, per this project's established convention):
  - scripts/research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py -- this signal's own ETH
    TabPFN script. cluster_dedup_oscillator() is ported VERBATIM (dedup anchor = most
    oscillator-extreme row per cluster, i.e. p_fast+p_slow most negative for bottom / most
    positive for top), called with gap=6 (a reasonable middle of ETH's own [3,6,12] GAP_GRID --
    round 2 never screened a BTC-specific GAP, it kept GAP=12 fixed throughout).
  - scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py -- run_tabpfn_panel()
    and compute_permutation_importance() (which itself uses evaluate()) are imported and used
    UNCHANGED, including that module's own SEEDS=[20260829,141592,271828,577215] and its
    n_repeats=5 permutation-importance default -- both already match this round's spec exactly.

Data: data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_
tier0.csv (277,191 rows, 2024-01-01..2026-08-20, BTCUSDT 5m). bottom_orthogonal_combo/
top_orthogonal_combo triggers are already computed using BTC's own funding_z internally (see
scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py) -- funding is NOT recomputed
or re-loaded here.

Features (24, ETH's canonical 23-feature set from build_indicator_frame + this Tier0's own
delta_z/range_width_pct, informative for this signal per round 1/2): the CSV already carries 19 of
these as-is (atr_percentile_864, hour_utc, weekday, p_fast, p_slow, ret3_z, vwap_dev_z,
cvd_roll_roc_48, vol_z, lower_wick_ratio, upper_wick_ratio, bb_pctb, adx14, pdi, ndi,
bb_width_pctile, rsi, delta_z, range_width_pct) -- loaded unchanged, NOT recomputed. The other 5
(is_bottom, atr_pct, nyse_open_flag, er_24, realized_vol_ratio) are computed fresh here, formulas
ported VERBATIM from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::
build_indicator_frame -- note atr_pct here is a FRESH 14-period-SMA-true-range/close calc (matches
ETH's own feature construction exactly), deliberately NOT the same as either the CSV's own
`atr_pct` column (from compute_indicators(), different smoothing) or the CSV's `atr` column (a
different causal dollar-ATR used only for the label threshold above, per this round's task spec).
These are three legitimately different ATR measures serving three different roles in this script;
that is intentional, not an inconsistency to fix.

Splits: TRAIN < 2025-09-01, VAL 2025-09-01..2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT >=
2026-04-01 (single-touch, evaluated once here for the first time this signal touches it).

Runs on the GPU server (quant_ai env, CUDA required for TabPFN) -- must be flock-wrapped, this
devmachine has no GPU and the server's single 8GB GPU is shared across concurrently-running Homer
signal agents.
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

from research_eth_taker_delta_climax_metalabel_tabpfn_20260829 import (
    compute_permutation_importance, run_tabpfn_panel,
)

CSV_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_DIR = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901"
REPORT_PATH = OUT_DIR / "orthogonal_combo_tabpfn_report.json"
FEATURES_CSV_PATH = OUT_DIR / "orthogonal_combo_tabpfn_features.csv"

START = pd.Timestamp("2024-01-01", tz="UTC")
HORIZON = 8   # bars forward -- round 2's touch_mfe leaderboard point (most-robust-sample-size)
K = 2.0       # ATR multiple for the hit threshold -- round 2's touch_mfe leaderboard point
GAP = 6       # cluster-dedup embargo bars -- middle of ETH's own [3,6,12] GAP_GRID, see docstring

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

SEEDS = [20260829, 141592, 271828, 577215]  # matches the imported taker script's own module-level
                                             # SEEDS (used internally by run_tabpfn_panel /
                                             # compute_permutation_importance) -- redeclared here
                                             # only so the report JSON documents the exact seeds.

FEATURE_COLUMNS = [
    "is_bottom", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi", "delta_z", "range_width_pct",
]
_NEW_FEATURES = {"is_bottom", "atr_pct", "nyse_open_flag", "er_24", "realized_vol_ratio"}
EXISTING_TIER0_COLUMNS = [c for c in FEATURE_COLUMNS if c not in _NEW_FEATURES]


def log(msg: str) -> None:
    print(f"[btc_orthogonal_combo_tabpfn] {msg}", flush=True)


def cluster_dedup_oscillator(idx: np.ndarray, p_fast: np.ndarray, p_slow: np.ndarray, side: str, gap: int) -> np.ndarray:
    """Verbatim port of research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py::cluster_dedup_oscillator."""
    score = -(p_fast[idx] + p_slow[idx]) if side == "bottom" else (p_fast[idx] + p_slow[idx])
    order = np.argsort(idx)
    idx_sorted, s_sorted = idx[order], score[order]
    cluster_id = np.zeros(len(idx_sorted), dtype=int)
    cid = 0
    for i in range(1, len(idx_sorted)):
        if idx_sorted[i] - idx_sorted[i - 1] > gap:
            cid += 1
        cluster_id[i] = cid
    df = pd.DataFrame({"idx": idx_sorted, "cluster": cluster_id, "s": s_sorted})
    return np.sort(df.loc[df.groupby("cluster")["s"].idxmax()]["idx"].to_numpy())


def load_tier0() -> pd.DataFrame:
    usecols = sorted(set(
        ["timestamp", "high", "low", "close", "atr",
         "bottom_orthogonal_combo", "top_orthogonal_combo"] + EXISTING_TIER0_COLUMNS
    ))
    frame = pd.read_csv(CSV_PATH, usecols=usecols)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    return frame


def add_missing_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Ports atr_pct / nyse_open_flag / er_24 / realized_vol_ratio EXACT formulas from
    research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::build_indicator_frame. The other
    19 of the 24 final features already exist in the BTC Tier0 CSV as-is (loaded, not recomputed)."""
    close, high, low = frame["close"], frame["high"], frame["low"]
    prev_close = close.shift(1)
    prev_close.iloc[0] = close.iloc[0]
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_raw = tr.rolling(14, min_periods=14).mean()
    frame["atr_pct"] = atr_raw / close.clip(lower=1e-12)

    tmin = frame["timestamp"].dt.hour * 60 + frame["timestamp"].dt.minute
    is_weekday = frame["timestamp"].dt.dayofweek < 5
    frame["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    net_change_24 = close - close.shift(24)
    diff_abs = close.diff().abs()
    frame["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)
    log_ret = np.log(close / close.shift(1))
    frame["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()
    return frame


def build_fires(frame: pd.DataFrame) -> pd.DataFrame:
    """Deduped bottom/top orthogonal_combo candidates with the fixed touch_mfe H=8/K=2.0 label
    (round 2's most-robust-sample-size point, see module docstring) -- clean binary hit/miss, no
    exclude-middle (that was an ETH-only fix, not replicated here per this round's task spec)."""
    high = frame["high"].to_numpy(); low = frame["low"].to_numpy(); close = frame["close"].to_numpy()
    atr = frame["atr"].to_numpy()
    p_fast = frame["p_fast"].to_numpy(); p_slow = frame["p_slow"].to_numpy()
    ts = frame["timestamp"].to_numpy()
    n = len(frame)
    valid_atr = np.isfinite(atr) & (atr > 0) & np.isfinite(close)
    rows = []
    for side, col in [("bottom", "bottom_orthogonal_combo"), ("top", "top_orthogonal_combo")]:
        idx = np.flatnonzero(frame[col].fillna(False).to_numpy())
        idx = idx[(idx < n - HORIZON) & (ts[idx] >= START) & valid_atr[idx]]
        idx_before_dedup = len(idx)
        idx = cluster_dedup_oscillator(idx, p_fast, p_slow, side, GAP)
        log(f"  {side}: {idx_before_dedup} raw fires -> {len(idx)} after GAP={GAP} cluster-anchor dedup")
        entry = close[idx]; a = atr[idx]
        if side == "bottom":
            fut_ext = np.array([high[i + 1:i + HORIZON + 1].max() for i in idx])
            hit = (fut_ext >= entry + K * a).astype(float)
        else:
            fut_ext = np.array([low[i + 1:i + HORIZON + 1].min() for i in idx])
            hit = (fut_ext <= entry - K * a).astype(float)
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
    return fires


def split_train_val_oos_holdout(fires: pd.DataFrame):
    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    holdout = fires.loc[ts >= HOLDOUT_START].reset_index(drop=True)
    return train, val, oos, holdout


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    log(f"loading BTC Tier0 CSV ({CSV_PATH})...")
    frame = load_tier0()
    log(f"{len(frame)} rows loaded, range {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")

    log("adding 4 missing bar-wide features (atr_pct/nyse_open_flag/er_24/realized_vol_ratio, ported verbatim)...")
    frame = add_missing_features(frame)

    log(f"building orthogonal_combo fires: HORIZON={HORIZON} K={K} GAP={GAP} (touch_mfe, binary hit/miss, no exclude-middle)...")
    fires = build_fires(frame)
    n_before_dropna = len(fires)
    fires = fires.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)
    log(f"{len(fires)}/{n_before_dropna} usable fires after dropna(FEATURE_COLUMNS) "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    train, val, oos, holdout = split_train_val_oos_holdout(fires)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, "
        f"HOLDOUT(>={HOLDOUT_START.date()}) n={len(holdout)}")
    fire_hit_rate = float(fires["hit"].mean())
    holdout_hr = f"{holdout['hit'].mean():.4f}" if len(holdout) else "n/a"
    log(f"hit-rate-of-kept: overall={fire_hit_rate:.4f} TRAIN={train['hit'].mean():.4f} "
        f"VAL={val['hit'].mean():.4f} OOS={oos['hit'].mean():.4f} HOLDOUT={holdout_hr}")

    fires.to_csv(FEATURES_CSV_PATH, index=False)
    log(f"features CSV saved -> {FEATURES_CSV_PATH}")

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f} (n_eval={val_result['n_eval']})")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f} (n_eval={oos_result['n_eval']})")

    log("=== RESERVED HOLDOUT evaluation (single-touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = (
        run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30
        else {"note": "too few holdout fires (<30)", "n_holdout": int(len(holdout))}
    )
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f} (n_eval={holdout_result['n_eval']})")

    log("=== permutation feature importance (VAL, single seed, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "orthogonal_combo", "asset": "BTC",
        "status": "exploratory_single_signal_below_promotion_bar",
        "summary_for_future_sessions": (
            f"Final TabPFN phase of the BTC orthogonal_combo port. Label fixed at round 2's "
            f"most-robust-sample-size touch_mfe point (H={HORIZON}, K={K}, GAP={GAP} cluster-dedup) "
            "-- NOT the round-2 'global winner' (touch_giveback_sustained H=8/K=3.0), which was "
            "explicitly distrusted for having only 2-5 OOS hits, far short of the TRAIN gate. "
            "touch_mfe here is a plain binary hit/miss label, no exclude-middle refinement (that "
            "was an ETH-only fix for a documented ETH label pathology, not replicated here per "
            f"task spec). VAL AUC {val_result.get('auc_mean')}, OOS AUC {oos_result.get('auc_mean')}, "
            f"HOLDOUT AUC {holdout_result.get('auc_mean', 'N/A')} (TabPFN, {len(FEATURE_COLUMNS)} "
            "features, 4 seeds each). See docs/experiments/"
            "btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md for the round-1/round-2 "
            "grid-screen history and the raw-lift OOS-instability diagnosis this run is checking "
            "against, and docs/experiments/btc_5m_orthogonal_combo_metalabel_tabpfn_20260901.md for "
            "this round's own writeup."
        ),
        "round2_reference": {
            "doc": "docs/experiments/btc_5m_orthogonal_combo_gridscreen_featureanalysis_20260901.md",
            "chosen_point": "touch_mfe, H=8, K=2.0 (most-robust-sample-size touch-based point, "
                             "NOT the distrusted global winner touch_giveback_sustained H=8/K=3.0)",
            "gap_used_round2": 12, "gap_used_here": GAP,
            "raw_lift_train": 1.505, "raw_lift_val": 1.318, "raw_lift_oos": 1.148,
            "raw_lift_oos_hit_sample_bottom_top": "49/31",
            "diagnosis": "OOS lift bounced 0.556-1.148x across adjacent grid cells with no clean "
                         "pattern by HIT_TYPE, H, or K -- best-supported explanation is BTC's small "
                         "candidate count (76-131 per side in OOS), not a HIT-definition flaw.",
        },
        "eth_orthogonal_combo_tabpfn_reference": {
            "script": "scripts/research_eth_orthogonal_combo_metalabel_tabpfn_20260830.py",
            "report": "tmp/eth_orthogonal_combo_metalabel_tabpfn_20260830/report.json",
            "label": "v2_exclude_middle (K_lo/K_hi around auto-calibrated K_center=2.5), HORIZON=24, GAP=12",
            "val_auc_mean": 0.723, "oos_auc_mean": 0.7162, "holdout_auc_mean": 0.7076,
            "n_train": 956, "n_val": 181, "n_oos": 128, "n_holdout": 228,
            "note": "ETH's headline AUC was later found to be inflated by the exclude-middle "
                    "'kept-only' population itself (2026-08-30 deep-dive re-evaluation on the full "
                    "population found ~0.665-0.680 AUC) -- not directly comparable to this BTC "
                    "run's plain-binary label without that caveat; included for rough context only.",
        },
        "horizon": HORIZON, "k": K, "gap": GAP, "hit_type": "touch_mfe",
        "feature_columns": FEATURE_COLUMNS,
        "features_newly_computed_this_script": sorted(_NEW_FEATURES),
        "features_reused_from_tier0_csv_asis": EXISTING_TIER0_COLUMNS,
        "n_fires_before_dropna": n_before_dropna, "n_fires_total": int(len(fires)),
        "fired_signal_hit_rate_of_kept": fire_hit_rate,
        "n_train": int(len(train)), "n_val": int(len(val)), "n_oos": int(len(oos)), "n_holdout": int(len(holdout)),
        "val": val_result, "oos": oos_result, "reserved_holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "holdout_touched": bool(len(holdout) > 0),
        "note_fresh_forward": "Label-separability / TabPFN classification pass (touch-based MFE hit "
                               "vs miss), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar "
                               "is N/A=False by construction, no trade ledger exists yet.",
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
