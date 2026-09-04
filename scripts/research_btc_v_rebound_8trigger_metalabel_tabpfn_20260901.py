#!/usr/bin/env python3
"""BTC V자반등(V_REBOUND) metalabel -- 8-TRIGGER candidate-pool widening, TabPFN TRAIN/VAL/OOS/HOLDOUT.

Follow-up to scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py (the 6-trigger round,
PRESERVED UNCHANGED as the historical record -- this is a NEW file, not an edit of that one, per
this project's convention of keeping each round's script). User explicitly approved widening the
candidate pool from 6 to 8 triggers now that demarker_extreme and kalman_deviation_meanrev have
each independently confirmed as real BTC signals in their own TabPFN confirmation runs this same
session (demarker_extreme VAL/OOS/HOLDOUT AUC 0.690/0.666/0.729; kalman_deviation_meanrev
0.729/0.624/0.671 -- both from this session's demarker_extreme_tabpfn_report.json /
kalman_deviation_meanrev_tabpfn_report.json). smt_divergence stays OUT of scope (its cross-asset
partner asset is still undecided for BTC) -- so this is 8 triggers, NOT full 9-trigger parity with
ETH's current live union.

EVERYTHING about the label formula, split boundaries, feature set, and TabPFN infra is reused
BYTE-IDENTICAL from the 6-trigger script (FAST_BARS=6, FULL_BARS=12, ATR_MULT=1.5, CHOP_MULT=1.0,
T_SUSTAIN=0.20, label_side()/compute_outcome_fields()/build_side_frame()/summarize_split()/
evaluate()/run_tabpfn_panel()/compute_permutation_importance() all copied verbatim, unmodified).
The ONLY change is the candidate-pool INPUT: instead of reading any_bottom_trigger/any_top_trigger
directly from the Tier0 CSV (encodes only the original 6-trigger union: liquidity_sweep/
taker_delta_z_climax/short_term_return_z/orthogonal_combo/fib_extension_exhaustion/local_extreme),
two more triggers are computed FRESH here and OR'd in:

  demarker_extreme: `compute_demarker(high, low)` imported VERBATIM (not reimplemented) from
  research_eth_demarker_evidence_signal_lift_check_20260831.py -- the same import this project's
  own research_btc_demarker_extreme_metalabel_tabpfn_20260901.py already uses for the identical
  function. bottom fires when dem<=0.10, top when dem>=0.90 -- the RAW (non-deduped) per-bar
  trigger from that script's Phase A (`bottom_trig`/`top_trig` in run_grid_screen), matching the
  task instruction exactly (not the Phase B cluster-deduped version).

  kalman_deviation_meanrev: `compute_kalman_dev_z(close)` imported VERBATIM (not reimplemented)
  from research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py -- itself a verbatim port
  of live_evidence_signal_dashboard_20260823.py's kalman filter (F=[[1,1],[0,1]], H=[[1,0]],
  Q=I*1e-5, R=[[1e-3]], rolling z-score window=288). bottom fires when kalman_dev_z<=-2.0, top when
  >=2.0 -- again the RAW per-bar trigger, no clustering. Both source modules are import-safe
  (function/constant definitions only at module scope, GPU/tabpfn imports deferred inside their own
  functions, main() gated behind `if __name__ == "__main__"`), matching the demarker BTC script's
  own established import pattern -- confirmed safe by direct read of both files before writing this.

  any_bottom_trigger_8bit = any_bottom_trigger | bottom_demarker_extreme | bottom_kalman_deviation_meanrev
  any_top_trigger_8bit    = any_top_trigger    | top_demarker_extreme    | top_kalman_deviation_meanrev

This 8-bit union REPLACES the original 2 columns as build_candidate_pool()'s mask input; the
labeling logic downstream of that mask (label_side, build_side_frame, the 23-feature set) is
completely unchanged -- dem/kalman_dev_z are used ONLY as pool triggers here, NOT added as new
features (FEATURE_COLUMNS stays the same 23 columns as the 6-trigger script). The original
6-trigger candidate pool is ALSO rebuilt here (unchanged trigger columns, unchanged logic) purely
for side-by-side before/after logging -- see build_dataset()/log_pool_comparison() -- so the
pool-size and label-rate change from adding the 2 new triggers is visible and auditable in this
run's own log + report JSON, not just asserted from the prior script's separately-saved report.

Tier0 CSV, split boundaries, TabPFN SEEDS, sweep_penetration_atr sign convention: all identical to
the 6-trigger script -- see that script's module docstring for the full history (grid-screening
round, the sign-note resolution, etc.), not repeated here.

HOLDOUT is evaluated ONCE in this script (single-touch discipline) -- this is the first time this
SPECIFIC 8-trigger candidate pool touches HOLDOUT (a distinct pool definition from the 6-trigger
run's own, separately single-touched, HOLDOUT exposure).

Must run on the GPU server under the same system-wide flock as every other TabPFN script this
session (single shared 8GB GPU): /home/llewyn/crypto-scalping/.tabpfn_gpu.lock. Not runnable
locally for the TabPFN portion (no GPU, no tabpfn package) -- build_dataset() itself is CPU-only
(candidate-pool assembly, importable/testable without GPU or tabpfn), same convention as the
6-trigger script.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research_eth_demarker_evidence_signal_lift_check_20260831 import compute_demarker  # noqa: E402
from research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901 import compute_kalman_dev_z  # noqa: E402

DATA_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_8trigger_tabpfn_report.json"
FEATURES_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_v_rebound_8trigger_metalabel_features_20260901.csv"

# --- outcome formula constants, reused verbatim from the 6-trigger script (see module docstring) ---
FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20

# --- fresh-forward split boundaries (CLAUDE.md default), identical to the 6-trigger script ---
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")   # == VAL start
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

TRIGGERS_6 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
              "orthogonal_combo", "fib_extension_exhaustion", "local_extreme"]
NEW_TRIGGERS = ["demarker_extreme", "kalman_deviation_meanrev"]
TRIGGERS = TRIGGERS_6 + NEW_TRIGGERS  # 8 -- used for the trig_{name} audit columns in build_side_frame

FEATURE_COLUMNS = [
    "is_bottom", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]  # 23 -- UNCHANGED from the 6-trigger script; only the candidate-pool input changes this round,
   # NOT the feature set (dem/kalman_dev_z are pool triggers only here, not added as features)

SEEDS = [20260829, 141592, 271828, 577215]
RANDOM_STATE = 20260901

BTC_6TRIGGER_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py (this project's own "
              "immediately-preceding round, preserved unchanged) via its server-side "
              "v_rebound_tabpfn_report.json",
    "n_triggers": 6,
    "n_train": 13185,
    "val_auc_mean": 0.8351, "oos_auc_mean": 0.8202, "holdout_auc_mean": 0.8277,
}

ETH_9TRIGGER_BENCHMARK = {
    "source": "live_eth_sweep_v_rebound_signal_20260829.py docstring + docs/homer/README.md "
               "(4-seed VAL/OOS stability -> reserved HOLDOUT, single-touch)",
    "n_triggers": 9,
    "val_auc_mean": 0.8292, "oos_auc_mean": 0.8127, "holdout_auc_mean": 0.8465,
}


def log(msg: str) -> None:
    print(f"[btc_v_rebound_8trigger_tabpfn] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def add_new_triggers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute demarker_extreme and kalman_deviation_meanrev triggers fresh on this Tier0
    dataframe's own high/low/close columns, and OR them into the existing 6-trigger union to build
    the 8-trigger pool. Both formulas reused VERBATIM via import (see module docstring) -- NOT
    re-derived, and RAW (non-deduped) per task instruction. Writes bottom_/top_ columns for each
    new trigger (so build_side_frame's existing `trig_{trig}` audit-column loop picks them up
    unchanged) plus the any_*_trigger_8bit union columns. Does NOT modify the Tier0 CSV on disk --
    these columns exist only in this in-memory df."""
    dem = compute_demarker(df["high"], df["low"])
    df["bottom_demarker_extreme"] = (dem <= 0.10).to_numpy()
    df["top_demarker_extreme"] = (dem >= 0.90).to_numpy()

    kalman_dev_z = pd.Series(compute_kalman_dev_z(df["close"].to_numpy()))
    df["bottom_kalman_deviation_meanrev"] = (kalman_dev_z <= -2.0).fillna(False).to_numpy()
    df["top_kalman_deviation_meanrev"] = (kalman_dev_z >= 2.0).fillna(False).to_numpy()

    log(f"  fresh trigger raw fire counts (whole history, pre-union): "
        f"demarker_extreme bottom={int(df['bottom_demarker_extreme'].sum())} "
        f"top={int(df['top_demarker_extreme'].sum())}, "
        f"kalman_deviation_meanrev bottom={int(df['bottom_kalman_deviation_meanrev'].sum())} "
        f"top={int(df['top_kalman_deviation_meanrev'].sum())}")

    df["any_bottom_trigger_8bit"] = (
        df["any_bottom_trigger"] | df["bottom_demarker_extreme"] | df["bottom_kalman_deviation_meanrev"]
    )
    df["any_top_trigger_8bit"] = (
        df["any_top_trigger"] | df["top_demarker_extreme"] | df["top_kalman_deviation_meanrev"]
    )
    log(f"  any_bottom_trigger: {int(df['any_bottom_trigger'].sum())} -> "
        f"any_bottom_trigger_8bit: {int(df['any_bottom_trigger_8bit'].sum())}  |  "
        f"any_top_trigger: {int(df['any_top_trigger'].sum())} -> "
        f"any_top_trigger_8bit: {int(df['any_top_trigger_8bit'].sum())}")
    return df


def compute_outcome_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::compute_outcome_fields,
    unchanged from the 6-trigger script. Computed once over the FULL (untruncated) frame --
    forward windows may read past a split boundary, see module docstring."""
    close, high, low, atr = df["close"], df["high"], df["low"], df["atr"]

    fast_close_max = close[::-1].rolling(FAST_BARS, min_periods=FAST_BARS).max()[::-1].shift(-1)
    fast_close_min = close[::-1].rolling(FAST_BARS, min_periods=FAST_BARS).min()[::-1].shift(-1)
    full_high_max = high[::-1].rolling(FULL_BARS, min_periods=FULL_BARS).max()[::-1].shift(-1)
    full_low_min = low[::-1].rolling(FULL_BARS, min_periods=FULL_BARS).min()[::-1].shift(-1)
    end_price = close.shift(-FULL_BARS)

    return pd.DataFrame({
        "pre_atr": atr.shift(1),
        "low": low, "high": high,
        "fast_close_max": fast_close_max, "fast_close_min": fast_close_min,
        "full_high_max": full_high_max, "full_low_min": full_low_min,
        "end_price": end_price,
    }, index=df.index)


def label_side(fields: pd.DataFrame, is_down: bool, atr_mult: float = ATR_MULT) -> pd.DataFrame:
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::label_side, unchanged from
    the 6-trigger script."""
    pre_atr = fields["pre_atr"]
    if is_down:
        extreme = fields["low"]
        fast_move = fields["fast_close_max"] - extreme
        peak = fields["full_high_max"]
    else:
        extreme = fields["high"]
        fast_move = extreme - fields["fast_close_min"]
        peak = fields["full_low_min"]
    end_price = fields["end_price"]

    valid = (
        pre_atr.notna() & (pre_atr > 0)
        & fields["full_high_max"].notna() & fields["full_low_min"].notna() & end_price.notna()
    )

    fast_mult = fast_move / pre_atr
    denom = (peak - extreme) if is_down else (extreme - peak)
    denom_ok = denom.abs() >= 1e-12
    giveback = ((peak - end_price) / denom) if is_down else ((end_price - peak) / denom)
    giveback = giveback.where(denom_ok, other=np.nan)

    is_v = (fast_mult >= atr_mult) & giveback.notna() & (giveback <= T_SUSTAIN)
    is_chop = fast_mult < CHOP_MULT
    label_raw = np.where(is_v, 1, np.where(is_chop, 0, -1))  # -1 = ambiguous middle

    return pd.DataFrame(
        {"fast_mult": fast_mult, "giveback": giveback, "label_raw": label_raw, "valid": valid},
        index=fields.index,
    )


def assign_split(ts: pd.Series) -> np.ndarray:
    return np.select(
        [ts < TRAIN_END, ts < OOS_START, ts < HOLDOUT_START],
        ["TRAIN", "VAL", "OOS"],
        default="HOLDOUT",
    )


def build_side_frame(df: pd.DataFrame, labels: pd.DataFrame, mask: np.ndarray, side: str) -> pd.DataFrame:
    """Unchanged from the 6-trigger script, except TRIGGERS now has 8 entries -- the trig_ loop
    below reads the 2 new bottom_/top_ columns add_new_triggers() already wrote onto df."""
    idx = np.flatnonzero(mask)
    sub = df.iloc[idx].copy()
    lab = labels.iloc[idx]
    sub["side"] = side
    sub["is_bottom"] = 1 if side == "bottom" else 0
    sub["fast_mult"] = lab["fast_mult"].to_numpy()
    sub["giveback"] = lab["giveback"].to_numpy()
    valid = lab["valid"].to_numpy()
    label_raw = lab["label_raw"].to_numpy()
    status = np.where(~valid, "invalid_insufficient_data",
              np.where(label_raw == 1, "v_rebound",
              np.where(label_raw == 0, "chop_support", "ambiguous_excluded")))
    sub["status"] = status
    sub["label"] = np.where(status == "v_rebound", 1.0, np.where(status == "chop_support", 0.0, np.nan))
    sub["split"] = assign_split(sub["timestamp"])

    # direction-relative features -- verbatim formula match to live_eth_sweep_v_rebound_signal_
    # 20260829.py::_multitrigger_rows() (see the 6-trigger script's module docstring for the sign
    # discrepancy note vs the task's inline prose). Same-bar atr[i], NOT pre_atr.
    atr_i = df["atr"].to_numpy()[idx]
    if side == "bottom":
        level = df["sweep_level_low"].to_numpy()[idx]
        extreme = df["low"].to_numpy()[idx]
        penetration = level - extreme
        sub["flow_aligned_delta_z"] = df["delta_z"].to_numpy()[idx]
    else:
        level = df["sweep_level_high"].to_numpy()[idx]
        extreme = df["high"].to_numpy()[idx]
        penetration = extreme - level
        sub["flow_aligned_delta_z"] = -df["delta_z"].to_numpy()[idx]
    with np.errstate(divide="ignore", invalid="ignore"):
        sub["sweep_penetration_atr"] = np.where(np.isfinite(atr_i) & (atr_i > 0), penetration / atr_i, np.nan)

    for trig in TRIGGERS:
        sub[f"trig_{trig}"] = df[f"{side}_{trig}"].to_numpy()[idx]
    return sub


def build_candidate_pool(df: pd.DataFrame, fields: pd.DataFrame, bottom_col: str, top_col: str,
                          atr_mult: float = ATR_MULT) -> pd.DataFrame:
    """Same structure as the 6-trigger script's build_candidate_pool, parameterized by which
    bottom/top mask column to union on -- lets this script build BOTH the 6-trigger pool (for
    before/after comparison logging) and the 8-trigger pool (the one actually used downstream)
    from the identical labeling logic."""
    down_labels = label_side(fields, is_down=True, atr_mult=atr_mult)
    up_labels = label_side(fields, is_down=False, atr_mult=atr_mult)
    bottom_mask = df[bottom_col].to_numpy()
    top_mask = df[top_col].to_numpy()
    bottom_cand = build_side_frame(df, down_labels, bottom_mask, side="bottom")
    top_cand = build_side_frame(df, up_labels, top_mask, side="top")
    return pd.concat([bottom_cand, top_cand], ignore_index=True)


def summarize_split(cand: pd.DataFrame, split: str) -> dict:
    """Unchanged from the 6-trigger script."""
    pool = cand.loc[cand["split"] == split]
    n_total = len(pool)
    n_invalid = int((pool["status"] == "invalid_insufficient_data").sum())
    n_valid = n_total - n_invalid
    n_v = int((pool["status"] == "v_rebound").sum())
    n_chop = int((pool["status"] == "chop_support").sum())
    n_amb = int((pool["status"] == "ambiguous_excluded").sum())
    n_labeled = n_v + n_chop
    return {
        "candidate_count_total": int(n_total),
        "invalid_insufficient_data": n_invalid,
        "valid_count": int(n_valid),
        "v_rebound": n_v,
        "chop_support": n_chop,
        "ambiguous_excluded": n_amb,
        "excluded_rate_of_valid": round(n_amb / n_valid, 4) if n_valid else None,
        "label_rate_v_rebound_of_labeled": round(n_v / n_labeled, 4) if n_labeled else None,
        "n_labeled": n_labeled,
    }


def log_pool_comparison(cand_6: pd.DataFrame, cand_8: pd.DataFrame) -> dict:
    """Before(6-trigger)/after(8-trigger) candidate-count and label-rate comparison, computed
    within THIS run (not just quoting the prior script's saved report) so the pool-size change from
    adding demarker_extreme/kalman_deviation_meanrev is directly visible and auditable here."""
    log("=== candidate pool size comparison: 6-trigger (baseline, recomputed here) vs 8-trigger (this run) ===")
    comparison = {}
    for split in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        c6 = summarize_split(cand_6, split)
        c8 = summarize_split(cand_8, split)
        d_total = c8["candidate_count_total"] - c6["candidate_count_total"]
        d_labeled = c8["n_labeled"] - c6["n_labeled"]
        pct = (d_total / c6["candidate_count_total"] * 100) if c6["candidate_count_total"] else float("nan")
        log(f"  {split}: candidates {c6['candidate_count_total']} -> {c8['candidate_count_total']} "
            f"({d_total:+d}, {pct:+.1f}%)  labeled {c6['n_labeled']} -> {c8['n_labeled']} ({d_labeled:+d})  "
            f"label_rate {c6['label_rate_v_rebound_of_labeled']} -> {c8['label_rate_v_rebound_of_labeled']}")
        comparison[split] = {
            "6trigger": c6, "8trigger": c8,
            "delta_candidate_count_total": d_total, "delta_candidate_count_pct": round(pct, 2),
            "delta_n_labeled": d_labeled,
        }
    return comparison


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    """Unchanged from the 6-trigger script."""
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    """Unchanged from the 6-trigger script (ported from research_eth_taker_delta_climax_metalabel_
    tabpfn_20260829.py::run_tabpfn_panel)."""
    from tabpfn import TabPFNClassifier
    seed_rows = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train["label"].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        r = evaluate(proba, eval_df["label"].to_numpy().astype(int))
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
    """Unchanged from the 6-trigger script. Single-seed, hand-rolled permutation importance
    (AUC-scored) on VAL -- model-agnostic (TabPFN has no native .feature_importances_)."""
    from tabpfn import TabPFNClassifier

    clf = TabPFNClassifier(device="cuda", random_state=seed)
    clf.fit(train[feature_cols], train["label"].to_numpy().astype(int))
    y = eval_df["label"].to_numpy().astype(int)
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


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """All CPU-only prep (no GPU/tabpfn needed) -- data load through feature-ready 8-trigger
    candidate pool with split assignment, PLUS the 6-trigger pool (unchanged trigger columns) for
    before/after comparison. Separated from main() so this can be imported and sanity-checked
    locally (no GPU/tabpfn) without triggering TabPFN, same convention as the 6-trigger script."""
    log("loading BTC Tier0 candidate CSV (full range, TRAIN..HOLDOUT)...")
    df = load_data()
    log(f"  {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    log("computing fresh demarker_extreme/kalman_deviation_meanrev triggers and 8-trigger union...")
    df = add_new_triggers(df)

    fields = compute_outcome_fields(df)
    cand_6 = build_candidate_pool(df, fields, "any_bottom_trigger", "any_top_trigger", atr_mult=ATR_MULT)
    cand_8 = build_candidate_pool(df, fields, "any_bottom_trigger_8bit", "any_top_trigger_8bit", atr_mult=ATR_MULT)
    pool_comparison = log_pool_comparison(cand_6, cand_8)
    return cand_6, cand_8, pool_comparison


def main() -> int:
    t0 = time.time()
    cand_6, cand, pool_comparison = build_dataset()

    log("label/exclusion summary by split (8-trigger pool, used for training below)...")
    splits = ("TRAIN", "VAL", "OOS", "HOLDOUT")
    counts = {split: summarize_split(cand, split) for split in splits}
    for split in splits:
        c = counts[split]
        log(f"  {split}: total={c['candidate_count_total']} invalid={c['invalid_insufficient_data']} "
            f"v_rebound={c['v_rebound']} chop={c['chop_support']} ambiguous={c['ambiguous_excluded']} "
            f"excl_rate={c['excluded_rate_of_valid']} label_rate={c['label_rate_v_rebound_of_labeled']}")

    labeled = cand.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)
    n_dropped_extra = int((cand["status"].isin(["v_rebound", "chop_support"])).sum()) - len(labeled)
    log(f"labeled+feature-complete rows: {len(labeled)} "
        f"(additional NaN-feature drops beyond ambiguous/invalid: {n_dropped_extra})")

    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_csv(FEATURES_CSV, index=False)
    log(f"features CSV saved -> {FEATURES_CSV}")

    train = labeled.loc[labeled["split"] == "TRAIN"].reset_index(drop=True)
    val = labeled.loc[labeled["split"] == "VAL"].reset_index(drop=True)
    oos = labeled.loc[labeled["split"] == "OOS"].reset_index(drop=True)
    holdout = labeled.loc[labeled["split"] == "HOLDOUT"].reset_index(drop=True)
    log(f"TRAIN n={len(train)}, VAL n={len(val)}, OOS n={len(oos)}, HOLDOUT n={len(holdout)}")
    log(f"TRAIN label rate={train['label'].mean():.4f}, VAL={val['label'].mean():.4f}, "
        f"OOS={oos['label'].mean():.4f}, HOLDOUT={holdout['label'].mean():.4f}")

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== RESERVED HOLDOUT evaluation (>=2026-04-01, single-touch, TRAIN-fit, 4 seeds) ===")
    holdout_result = run_tabpfn_panel(train, holdout, FEATURE_COLUMNS, "HOLDOUT") if len(holdout) >= 30 else {"note": "too few holdout rows"}
    if "auc_mean" in holdout_result:
        log(f"HOLDOUT -> AUC {holdout_result['auc_mean']:.4f}+/-{holdout_result['auc_std']:.4f}")

    log("=== permutation feature importance (VAL, single seed, AUC-scored, 5 repeats) ===")
    perm_importance = compute_permutation_importance(train, val, FEATURE_COLUMNS)
    log(f"baseline VAL AUC (single seed {perm_importance['seed']}): {perm_importance['baseline_auc']:.4f}")
    for row in perm_importance["importances"][:10]:
        log(f"  {row['feature']:<22s} importance={row['importance_mean']:+.5f} (std={row['importance_std']:.5f})")

    report = {
        "signal": "v_rebound",
        "asset": "BTCUSDT",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "scope": {
            "candidate_pool_triggers": TRIGGERS,
            "n_triggers_this_run": len(TRIGGERS),
            "new_triggers_added_this_round": NEW_TRIGGERS,
            "live_eth_signal_n_triggers": 9,
            "triggers_excluded_vs_live_eth": ["smt_divergence"],
            "note": "8-trigger widening of scripts/research_btc_v_rebound_metalabel_tabpfn_"
                     "20260901.py's 6-trigger pool -- demarker_extreme and kalman_deviation_meanrev "
                     "added now that both are independently confirmed as real BTC signals (see "
                     "btc_6trigger_benchmark_for_comparison / eth_9trigger_benchmark_for_comparison "
                     "below). smt_divergence stays OUT of scope (cross-asset partner asset still "
                     "undecided for BTC) -- this is 8 triggers, NOT full 9-trigger parity with "
                     "ETH's current live union.",
        },
        "formula": {
            "fast_bars": FAST_BARS, "full_bars": FULL_BARS,
            "atr_mult": ATR_MULT, "chop_mult": CHOP_MULT, "t_sustain": T_SUSTAIN,
            "source": "research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome "
                       "(verbatim, unchanged from the 6-trigger script -- not re-derived this round)",
        },
        "new_trigger_formulas": {
            "demarker_extreme": "compute_demarker(high, low) imported verbatim from "
                "research_eth_demarker_evidence_signal_lift_check_20260831.py; RAW per-bar trigger, "
                "bottom: dem<=0.10, top: dem>=0.90",
            "kalman_deviation_meanrev": "compute_kalman_dev_z(close) imported verbatim from "
                "research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py; RAW per-bar "
                "trigger, bottom: kalman_dev_z<=-2.0, top: kalman_dev_z>=2.0",
        },
        "sweep_penetration_atr_sign_note": (
            "Computed to match live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() "
            "exactly (level-minus-extreme for bottom / extreme-minus-level for top, same-bar atr[i]) "
            "-- unchanged from the 6-trigger script, see that script's module docstring for detail."
        ),
        "split_boundaries": {
            "train_end": str(TRAIN_END), "oos_start": str(OOS_START), "holdout_start": str(HOLDOUT_START),
        },
        "candidate_pool_comparison_6trigger_vs_8trigger": pool_comparison,
        "label_summary_by_split": counts,
        "n_dropped_extra_nan_features": n_dropped_extra,
        "feature_columns": FEATURE_COLUMNS,
        "val": val_result,
        "oos": oos_result,
        "holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "btc_6trigger_benchmark_for_comparison": BTC_6TRIGGER_BENCHMARK,
        "eth_9trigger_benchmark_for_comparison": ETH_9TRIGGER_BENCHMARK,
        "runtime_sec": round(time.time() - t0, 1),
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    log(f"total runtime: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
