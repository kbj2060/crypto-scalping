#!/usr/bin/env python3
"""BTC V자반등(V_REBOUND) metalabel -- 7-TRIGGER candidate pool (DeMarker ONLY, kalman DROPPED),
TabPFN TRAIN/VAL/OOS/HOLDOUT.

4th round of this V_REBOUND BTC candidate-pool-widening lineage:
  (1) scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py -- 6-trigger baseline
      (VAL/OOS/HOLDOUT AUC 0.8351/0.8202/0.8277, preserved unchanged). Standing best entering this
      round.
  (2) scripts/research_btc_v_rebound_8trigger_metalabel_tabpfn_20260901.py -- widened to 8 triggers
      (added demarker_extreme/kalman_deviation_meanrev) using RAW (non-deduped) per-bar trigger
      booleans OR'd straight into the union. REJECTED: regressed all 3 splits (VAL/OOS/HOLDOUT AUC
      0.8069/0.8067/0.8189). Preserved unchanged (read-only reference).
  (3) scripts/research_btc_v_rebound_8trigger_deduped_metalabel_tabpfn_20260901.py -- same 8
      triggers, but demarker_extreme/kalman_deviation_meanrev cluster-deduped (GAP=6, each signal's
      own already-validated solo-BTC-script dedup reused verbatim) before the union. Partial fix:
      VAL 0.8198 (real recovery), OOS 0.8074 (~noise), HOLDOUT 0.8174 (slightly worse) -- still below
      the 6-trigger baseline on all 3 splits. Preserved unchanged (read-only reference).
  (4) THIS SCRIPT -- root-cause follow-up (docs/experiments/btc_v_rebound_feeder_gap_threshold_
      screen_20260901.md, orchestrating session, local CPU-only grid screen, not a subagent) found
      GAP was NOT the limiting factor: sweeping GAP 6->96 bars left net-new-candidate quality nearly
      flat for both signals. What actually differs: demarker_extreme's net-new contribution to the
      V_REBOUND pool is a real, if weaker-than-baseline, signal (net-new candidates succeed at the
      V-rebound outcome ~27-48% of the time depending on split/side/threshold, vs the 6-trigger
      baseline's ~42-47%, and this is STABLE as its threshold is tightened). kalman_deviation_
      meanrev's net-new contribution is much weaker (~13-16% on the bottom side) and, on the TOP
      side, gets monotonically WORSE the more strictly it is thresholded (10.7%->9.3%->7.5%->6.1%->
      4.3% as the z-cutoff tightens 2.0->4.0). Kalman was the one dragging the 8-trigger ensemble
      down, not DeMarker. Per explicit user instruction: kalman_deviation_meanrev is DROPPED
      ENTIRELY this round (not imported, not computed, no contribution of any kind) -- this is NOT
      "drop it from the union formula only," it is absent from this script altogether. demarker_
      extreme is kept, computed and cluster-deduped (GAP=6) EXACTLY as round (3) already does it --
      verbatim reuse of compute_demarker() + cluster_dedup(), neither re-derived nor re-tuned here.

      any_bottom_trigger_7trigger = any_bottom_trigger (original raw 6-signal OR, unchanged)
                                     | bottom_demarker_extreme_deduped
      any_top_trigger_7trigger    = any_top_trigger (unchanged)
                                     | top_demarker_extreme_deduped
      (mirrors round (3)'s any_bottom_trigger_8dedup formula with the "| bottom_kalman_deviation_
      meanrev_deduped" term simply absent.)

EVERYTHING about the label formula, split boundaries, feature set, and TabPFN infra is reused
BYTE-IDENTICAL from round (3) (FAST_BARS=6, FULL_BARS=12, ATR_MULT=1.5, CHOP_MULT=1.0,
T_SUSTAIN=0.20, label_side()/compute_outcome_fields()/build_side_frame()/build_candidate_pool()/
summarize_split()/evaluate()/run_tabpfn_panel()/compute_permutation_importance() all copied
verbatim, unmodified). The TRIGGERS audit-column loop in build_side_frame() is unchanged in
mechanism but now iterates a 7-item list (6 original + demarker_extreme only -- no
trig_kalman_deviation_meanrev column this round, since kalman is not computed at all).

Per task instruction, this script does NOT recompute the 6-trigger / raw-8-trigger / deduped-8-
trigger candidate pools fresh in-run for the comparison table (unlike round (3), which recomputed
rounds (1)-(2) fresh alongside its own for direct in-run comparability). None of those three pool
definitions change this round, so their already-saved numbers are pulled directly from their own
report JSONs on disk (data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound{,_8trigger,
_8trigger_deduped}_tabpfn_report.json) and embedded below as BTC_6TRIGGER_BENCHMARK /
BTC_8TRIGGER_RAW_BENCHMARK / BTC_8TRIGGER_DEDUPED_BENCHMARK (including each split's candidate_
count_total / label_rate_v_rebound_of_labeled, not just headline AUCs, so the 4-way comparison
table needs no separate file reads). Only this run's own 7-trigger pool is built fresh.

Tier0 CSV, split boundaries, TabPFN SEEDS, sweep_penetration_atr sign convention: all identical to
the prior three scripts -- see the 6-trigger script's module docstring for the full history (grid-
screening round, the sign-note resolution, etc.), not repeated here.

HOLDOUT is evaluated ONCE in this script (single-touch discipline) -- this is the first time this
SPECIFIC 7-trigger (demarker-only) candidate pool definition touches HOLDOUT (a distinct pool
definition from all three prior rounds' own, separately single-touched, HOLDOUT exposures).

Must run on the GPU server under the same system-wide flock as every other TabPFN script this
session (single shared 8GB GPU): /home/llewyn/crypto-scalping/.tabpfn_gpu.lock. Not runnable
locally for the TabPFN portion (no GPU, no tabpfn package) -- build_dataset() itself is CPU-only
(candidate-pool assembly, importable/testable without GPU or tabpfn), same convention as the prior
three scripts.
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
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402
# ^ kalman_deviation_meanrev is intentionally NOT imported anywhere in this script -- dropped
# entirely this round, not just from the union formula. See module docstring.

DATA_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_7trigger_demarker_only_tabpfn_report.json"
FEATURES_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_v_rebound_7trigger_demarker_only_metalabel_features_20260901.csv"

# --- outcome formula constants, reused verbatim from all three prior scripts ---
FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20

# --- fresh-forward split boundaries (CLAUDE.md default), identical to prior scripts ---
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")   # == VAL start
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

TRIGGERS_6 = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
              "orthogonal_combo", "fib_extension_exhaustion", "local_extreme"]
NEW_TRIGGERS = ["demarker_extreme"]  # kalman_deviation_meanrev dropped entirely this round
TRIGGERS = TRIGGERS_6 + NEW_TRIGGERS  # 7 -- used for the trig_{name} audit columns in build_side_frame

CLUSTER_GAP = 6  # fixed, reused verbatim from demarker_extreme's own already-validated BTC solo
                 # script -- NOT re-tuned here. The orchestrating session's own GAP sweep (6->96
                 # bars, docs/experiments/btc_v_rebound_feeder_gap_threshold_screen_20260901.md)
                 # confirmed GAP was not the limiting factor, so no need to revisit it here either.

FEATURE_COLUMNS = [
    "is_bottom", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]  # 23 -- UNCHANGED from all three prior scripts; only the candidate-pool input changes this round,
   # NOT the feature set (dem is a pool trigger only, not added as a feature)

SEEDS = [20260829, 141592, 271828, 577215]
RANDOM_STATE = 20260901

# --- prior rounds' ALREADY-SAVED numbers, pulled verbatim from their own report JSONs on disk
# (data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound{,_8trigger,_8trigger_deduped}_
# tabpfn_report.json) -- NOT recomputed in this run, per task instruction (none of these three pool
# definitions change this round). label_summary_by_split entries are copied field-for-field from
# each script's own summarize_split() output so the 4-way comparison table below needs no separate
# file reads. ---
BTC_6TRIGGER_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py (6-trigger baseline, "
              "preserved unchanged) via its server-side v_rebound_tabpfn_report.json",
    "n_triggers": 6,
    "n_train": 13185,
    "val_auc_mean": 0.8351, "val_auc_std": 0.0003,
    "oos_auc_mean": 0.8202, "oos_auc_std": 0.0011,
    "holdout_auc_mean": 0.8277, "holdout_auc_std": 0.0005,
    "label_summary_by_split": {
        "TRAIN": {"candidate_count_total": 34117, "invalid_insufficient_data": 1, "valid_count": 34116,
                  "v_rebound": 5898, "chop_support": 7356, "ambiguous_excluded": 20862,
                  "excluded_rate_of_valid": 0.6115, "label_rate_v_rebound_of_labeled": 0.445, "n_labeled": 13254},
        "VAL": {"candidate_count_total": 6828, "invalid_insufficient_data": 0, "valid_count": 6828,
                "v_rebound": 1179, "chop_support": 1512, "ambiguous_excluded": 4137,
                "excluded_rate_of_valid": 0.6059, "label_rate_v_rebound_of_labeled": 0.4381, "n_labeled": 2691},
        "OOS": {"candidate_count_total": 4926, "invalid_insufficient_data": 0, "valid_count": 4926,
                "v_rebound": 858, "chop_support": 986, "ambiguous_excluded": 3082,
                "excluded_rate_of_valid": 0.6257, "label_rate_v_rebound_of_labeled": 0.4653, "n_labeled": 1844},
        "HOLDOUT": {"candidate_count_total": 7593, "invalid_insufficient_data": 2, "valid_count": 7591,
                    "v_rebound": 1399, "chop_support": 1590, "ambiguous_excluded": 4602,
                    "excluded_rate_of_valid": 0.6062, "label_rate_v_rebound_of_labeled": 0.468, "n_labeled": 2989},
    },
}

BTC_8TRIGGER_RAW_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_8trigger_metalabel_tabpfn_20260901.py (REJECTED round "
              "-- raw/non-deduped per-bar OR union, preserved unchanged) via its server-side "
              "v_rebound_8trigger_tabpfn_report.json",
    "n_triggers": 8,
    "dedup": "NONE (raw per-bar OR union) -- REGRESSED all 3 splits vs the 6-trigger baseline",
    "n_train": 17548,
    "val_auc_mean": 0.8069, "val_auc_std": 0.0011,
    "oos_auc_mean": 0.8067, "oos_auc_std": 0.0015,
    "holdout_auc_mean": 0.8189, "holdout_auc_std": 0.0007,
    "label_summary_by_split": {
        "TRAIN": {"candidate_count_total": 41339, "invalid_insufficient_data": 1, "valid_count": 41338,
                  "v_rebound": 6407, "chop_support": 11258, "ambiguous_excluded": 23673,
                  "excluded_rate_of_valid": 0.5727, "label_rate_v_rebound_of_labeled": 0.3627, "n_labeled": 17665},
        "VAL": {"candidate_count_total": 8439, "invalid_insufficient_data": 0, "valid_count": 8439,
                "v_rebound": 1293, "chop_support": 2387, "ambiguous_excluded": 4759,
                "excluded_rate_of_valid": 0.5639, "label_rate_v_rebound_of_labeled": 0.3514, "n_labeled": 3680},
        "OOS": {"candidate_count_total": 6063, "invalid_insufficient_data": 0, "valid_count": 6063,
                "v_rebound": 953, "chop_support": 1582, "ambiguous_excluded": 3528,
                "excluded_rate_of_valid": 0.5819, "label_rate_v_rebound_of_labeled": 0.3759, "n_labeled": 2535},
        "HOLDOUT": {"candidate_count_total": 9609, "invalid_insufficient_data": 3, "valid_count": 9606,
                    "v_rebound": 1553, "chop_support": 2667, "ambiguous_excluded": 5386,
                    "excluded_rate_of_valid": 0.5607, "label_rate_v_rebound_of_labeled": 0.368, "n_labeled": 4220},
    },
}

BTC_8TRIGGER_DEDUPED_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_8trigger_deduped_metalabel_tabpfn_20260901.py "
              "(demarker_extreme+kalman_deviation_meanrev both cluster-deduped GAP=6 before union, "
              "preserved unchanged) via its server-side v_rebound_8trigger_deduped_tabpfn_report.json",
    "n_triggers": 8,
    "dedup": "cluster_dedup(GAP=6) applied to BOTH demarker_extreme and kalman_deviation_meanrev -- "
             "partial fix, still BELOW the 6-trigger baseline on all 3 splits",
    "n_train": 14567,
    "val_auc_mean": 0.8198, "val_auc_std": 0.0008,
    "oos_auc_mean": 0.8074, "oos_auc_std": 0.0014,
    "holdout_auc_mean": 0.8174, "holdout_auc_std": 0.0004,
    "label_summary_by_split": {
        "TRAIN": {"candidate_count_total": 36744, "invalid_insufficient_data": 1, "valid_count": 36743,
                  "v_rebound": 6133, "chop_support": 8512, "ambiguous_excluded": 22098,
                  "excluded_rate_of_valid": 0.6014, "label_rate_v_rebound_of_labeled": 0.4188, "n_labeled": 14645},
        "VAL": {"candidate_count_total": 7430, "invalid_insufficient_data": 0, "valid_count": 7430,
                "v_rebound": 1227, "chop_support": 1784, "ambiguous_excluded": 4419,
                "excluded_rate_of_valid": 0.5948, "label_rate_v_rebound_of_labeled": 0.4075, "n_labeled": 3011},
        "OOS": {"candidate_count_total": 5321, "invalid_insufficient_data": 0, "valid_count": 5321,
                "v_rebound": 893, "chop_support": 1153, "ambiguous_excluded": 3275,
                "excluded_rate_of_valid": 0.6155, "label_rate_v_rebound_of_labeled": 0.4365, "n_labeled": 2046},
        "HOLDOUT": {"candidate_count_total": 8331, "invalid_insufficient_data": 3, "valid_count": 8328,
                    "v_rebound": 1463, "chop_support": 1916, "ambiguous_excluded": 4949,
                    "excluded_rate_of_valid": 0.5943, "label_rate_v_rebound_of_labeled": 0.433, "n_labeled": 3379},
    },
}

ETH_9TRIGGER_BENCHMARK = {
    "source": "live_eth_sweep_v_rebound_signal_20260829.py docstring + docs/homer/README.md "
               "(4-seed VAL/OOS stability -> reserved HOLDOUT, single-touch)",
    "n_triggers": 9,
    "val_auc_mean": 0.8292, "oos_auc_mean": 0.8127, "holdout_auc_mean": 0.8465,
}


def log(msg: str) -> None:
    print(f"[btc_v_rebound_7trigger_demarker_only_tabpfn] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def dedup_trigger(raw_trig: np.ndarray, extremeness: np.ndarray, most_negative: bool,
                   gap: int = CLUSTER_GAP) -> tuple[np.ndarray, dict]:
    """Collapse a raw per-bar boolean trigger into cluster-deduped form via cluster_dedup()
    (imported verbatim, see module docstring): same-side raw fires within `gap` bars collapse into
    one cluster, keeping only the single most-extreme-`extremeness` bar per cluster. Returns
    (deduped_bool_array, {"raw": n_raw_fires, "deduped": n_kept_fires}) -- deduped_bool_array is
    True only at the kept cluster-representative bars, same length as raw_trig. Unchanged from
    round (3)."""
    idx = np.flatnonzero(raw_trig)
    idx_before = len(idx)
    kept_idx = cluster_dedup(idx, extremeness[idx], most_negative=most_negative, gap=gap)
    out = np.zeros(len(raw_trig), dtype=bool)
    out[kept_idx] = True
    return out, {"raw": int(idx_before), "deduped": int(len(kept_idx))}


def add_new_triggers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute demarker_extreme fresh on this Tier0 dataframe's own high/low columns, in BOTH raw
    (whole-history per-bar, unchanged formula from all three prior scripts) and cluster-deduped form
    (the union input used for training, GAP=6 -- see module docstring). kalman_deviation_meanrev is
    NOT computed at all this round (dropped entirely, not just from the union). Writes:
      any_bottom_trigger          original 6-trigger union, already in the Tier0 CSV, untouched.
      any_bottom_trigger_7trigger 6-trigger | DEDUPED demarker_extreme -- USED for training.
    (top_* mirror each.) Does NOT modify the Tier0 CSV on disk -- these columns exist only in this
    in-memory df. Returns (df, dedup_stats) where dedup_stats carries raw->deduped fire counts for
    demarker_extreme/bottom+top, for the report JSON."""
    dem = compute_demarker(df["high"], df["low"]).to_numpy()
    df["bottom_demarker_extreme"] = dem <= 0.10
    df["top_demarker_extreme"] = dem >= 0.90

    log(f"  fresh demarker_extreme RAW fire counts (whole history, pre-dedup, pre-union): "
        f"bottom={int(df['bottom_demarker_extreme'].sum())} top={int(df['top_demarker_extreme'].sum())} "
        f"(kalman_deviation_meanrev NOT computed this round -- dropped entirely)")

    dem_bottom_dd, dem_bottom_stats = dedup_trigger(
        df["bottom_demarker_extreme"].to_numpy(), dem, most_negative=True, gap=CLUSTER_GAP)
    dem_top_dd, dem_top_stats = dedup_trigger(
        df["top_demarker_extreme"].to_numpy(), dem, most_negative=False, gap=CLUSTER_GAP)

    df["bottom_demarker_extreme_deduped"] = dem_bottom_dd
    df["top_demarker_extreme_deduped"] = dem_top_dd

    dedup_stats = {
        "demarker_extreme": {"bottom": dem_bottom_stats, "top": dem_top_stats},
        "cluster_gap": CLUSTER_GAP,
    }
    log(f"  demarker_extreme cluster-dedup (GAP={CLUSTER_GAP}): "
        f"bottom {dem_bottom_stats['raw']} -> {dem_bottom_stats['deduped']}, "
        f"top {dem_top_stats['raw']} -> {dem_top_stats['deduped']}")

    # 7-trigger union -- THE candidate pool used for training below (kalman term simply absent,
    # vs round (3)'s any_bottom_trigger_8dedup = any_bottom_trigger | dem_dedup | kalman_dedup)
    df["any_bottom_trigger_7trigger"] = df["any_bottom_trigger"] | df["bottom_demarker_extreme_deduped"]
    df["any_top_trigger_7trigger"] = df["any_top_trigger"] | df["top_demarker_extreme_deduped"]

    log(f"  any_bottom_trigger: {int(df['any_bottom_trigger'].sum())}  ->  "
        f"7trigger(6trig|demarker_dedup)={int(df['any_bottom_trigger_7trigger'].sum())}  |  "
        f"any_top_trigger: {int(df['any_top_trigger'].sum())}  ->  "
        f"7trigger(6trig|demarker_dedup)={int(df['any_top_trigger_7trigger'].sum())}")

    return df, dedup_stats


def compute_outcome_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port, unchanged from all three prior scripts. Computed once over the FULL
    (untruncated) frame -- forward windows may read past a split boundary, see module docstring."""
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
    """Verbatim port, unchanged from all three prior scripts."""
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
    """Unchanged mechanism from all three prior scripts. The trig_ loop below now iterates a
    7-item TRIGGERS list (6 original + demarker_extreme only -- no trig_kalman_deviation_meanrev
    column this round, since kalman is not computed at all in this script)."""
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
    """Unchanged from all three prior scripts."""
    down_labels = label_side(fields, is_down=True, atr_mult=atr_mult)
    up_labels = label_side(fields, is_down=False, atr_mult=atr_mult)
    bottom_mask = df[bottom_col].to_numpy()
    top_mask = df[top_col].to_numpy()
    bottom_cand = build_side_frame(df, down_labels, bottom_mask, side="bottom")
    top_cand = build_side_frame(df, up_labels, top_mask, side="top")
    return pd.concat([bottom_cand, top_cand], ignore_index=True)


def summarize_split(cand: pd.DataFrame, split: str) -> dict:
    """Unchanged from all three prior scripts."""
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


def log_pool_comparison_4way(cand_7: pd.DataFrame) -> dict:
    """4-way candidate-pool size / label-rate comparison: this run's freshly-computed 7-trigger
    (demarker-only) pool against the three prior rounds' ALREADY-SAVED numbers (6-trigger baseline,
    rejected raw-8-trigger, deduped-8-trigger -- BTC_*_BENCHMARK constants above, pulled from their
    own saved report JSONs, NOT recomputed in this run per task instruction since none of those
    three pool definitions change this round)."""
    log("=== candidate pool comparison: 6trig vs 8trig-RAW vs 8trig-DEDUPED vs 7trig-DEMARKER-ONLY (this run) ===")
    comparison = {}
    for split in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        c7 = summarize_split(cand_7, split)
        b6 = BTC_6TRIGGER_BENCHMARK["label_summary_by_split"][split]
        b8r = BTC_8TRIGGER_RAW_BENCHMARK["label_summary_by_split"][split]
        b8d = BTC_8TRIGGER_DEDUPED_BENCHMARK["label_summary_by_split"][split]
        log(f"  {split}: candidates  6trig={b6['candidate_count_total']}  "
            f"8trig_raw={b8r['candidate_count_total']}  8trig_dedup={b8d['candidate_count_total']}  "
            f"7trig_demarker_only={c7['candidate_count_total']}   |   "
            f"label_rate  6trig={b6['label_rate_v_rebound_of_labeled']}  "
            f"8trig_raw={b8r['label_rate_v_rebound_of_labeled']}  "
            f"8trig_dedup={b8d['label_rate_v_rebound_of_labeled']}  "
            f"7trig_demarker_only={c7['label_rate_v_rebound_of_labeled']}")
        r6 = b6["label_rate_v_rebound_of_labeled"]
        r8d = b8d["label_rate_v_rebound_of_labeled"]
        r7 = c7["label_rate_v_rebound_of_labeled"]
        comparison[split] = {
            "6trigger": b6, "8trigger_raw_rejected": b8r, "8trigger_deduped": b8d,
            "7trigger_demarker_only": c7,
            "delta_candidate_count_7trig_vs_6trigger": c7["candidate_count_total"] - b6["candidate_count_total"],
            "delta_candidate_count_7trig_vs_8dedup": c7["candidate_count_total"] - b8d["candidate_count_total"],
            "delta_label_rate_7trig_vs_6trigger": round(r7 - r6, 4) if r7 is not None and r6 is not None else None,
            "delta_label_rate_7trig_vs_8dedup": round(r7 - r8d, 4) if r7 is not None and r8d is not None else None,
        }
    return comparison


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    """Unchanged from all three prior scripts."""
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    """Unchanged from all three prior scripts."""
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
    """Unchanged from all three prior scripts. Single-seed, hand-rolled permutation importance
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


def build_dataset() -> tuple[pd.DataFrame, dict, dict]:
    """All CPU-only prep (no GPU/tabpfn needed) -- data load through feature-ready 7-trigger
    (demarker-only) candidate pool with split assignment. Separated from main() so this can be
    imported and sanity-checked locally (no GPU/tabpfn) without triggering TabPFN, same convention
    as the prior three scripts. Unlike round (3), only ONE pool is built here (the 6trig/8raw/
    8dedup comparators are pulled from disk, see module docstring)."""
    log("loading BTC Tier0 candidate CSV (full range, TRAIN..HOLDOUT)...")
    df = load_data()
    log(f"  {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    log("computing fresh demarker_extreme trigger (raw + cluster-deduped GAP=6, reused verbatim "
        "from its own already-validated solo BTC script) -- kalman_deviation_meanrev dropped "
        "entirely this round (not imported, not computed)...")
    df, dedup_stats = add_new_triggers(df)

    fields = compute_outcome_fields(df)
    cand_7 = build_candidate_pool(df, fields, "any_bottom_trigger_7trigger", "any_top_trigger_7trigger",
                                   atr_mult=ATR_MULT)
    pool_comparison = log_pool_comparison_4way(cand_7)
    return cand_7, dedup_stats, pool_comparison


def main() -> int:
    t0 = time.time()
    cand, dedup_stats, pool_comparison = build_dataset()

    log("label/exclusion summary by split (7-trigger demarker-only pool, used for training below)...")
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
            "trigger_dropped_this_round": "kalman_deviation_meanrev",
            "live_eth_signal_n_triggers": 9,
            "triggers_excluded_vs_live_eth": ["smt_divergence", "kalman_deviation_meanrev"],
            "round": (
                "4th round of this V_REBOUND BTC candidate-pool-widening lineage: (1) 6-trigger "
                "baseline, (2) raw/non-deduped 8-trigger union -- REJECTED, regressed all 3 splits, "
                "(3) 8-trigger with BOTH new signals cluster-deduped (GAP=6) -- partial fix, still "
                "below baseline all 3 splits, (4) THIS RUN -- kalman_deviation_meanrev dropped "
                "ENTIRELY (not computed at all), demarker_extreme kept and cluster-deduped exactly "
                "as round (3)."
            ),
            "note": (
                "Follow-up to round (3)'s partial fix. The orchestrating session's own root-cause "
                "screen (docs/experiments/btc_v_rebound_feeder_gap_threshold_screen_20260901.md, "
                "local CPU-only GAP x threshold grid, no GPU needed) found GAP was NOT the limiting "
                "factor (swept 6->96 bars, net-new-candidate quality nearly flat). Instead: "
                "demarker_extreme's net-new candidates succeed at the V-rebound outcome ~27-48% of "
                "the time (stable across thresholds), vs kalman_deviation_meanrev's ~13-16% on the "
                "bottom side and a MONOTONIC WORSENING on the top side (10.7%->9.3%->7.5%->6.1%->"
                "4.3% as the z-cutoff tightens 2.0->4.0). kalman was diagnosed as the primary drag "
                "on the 8-trigger ensemble, not demarker. This run tests whether dropping kalman "
                "entirely lets the demarker-only 7-trigger pool match or beat the 6-trigger baseline."
            ),
        },
        "dedup_method": {
            "cluster_gap": CLUSTER_GAP,
            "algorithm": (
                "cluster_dedup() imported verbatim from research_btc_demarker_extreme_metalabel_"
                "tabpfn_20260901.py, applied to demarker_extreme ONLY this round (kalman_deviation_"
                "meanrev is not computed at all, so no dedup is applied to it either). Collapses "
                "same-side raw fires within GAP bars into one cluster, keeps only the single most-"
                "extreme bar per cluster (closest to 0/1 for dem). NOT re-tuned here -- GAP=6 reused "
                "exactly as already fixed in demarker_extreme's own solo BTC script, and confirmed "
                "not to be the limiting factor by the orchestrating session's own GAP sweep (6->96 "
                "bars, see docs/experiments/btc_v_rebound_feeder_gap_threshold_screen_20260901.md)."
            ),
            "dedup_stats": dedup_stats,
        },
        "formula": {
            "fast_bars": FAST_BARS, "full_bars": FULL_BARS,
            "atr_mult": ATR_MULT, "chop_mult": CHOP_MULT, "t_sustain": T_SUSTAIN,
            "source": "research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome "
                       "(verbatim, unchanged from all three prior scripts -- not re-derived this round)",
        },
        "new_trigger_formulas": {
            "demarker_extreme": (
                "compute_demarker(high, low) imported verbatim from research_eth_demarker_evidence_"
                "signal_lift_check_20260831.py; bottom: dem<=0.10, top: dem>=0.90, THEN "
                "cluster_dedup(GAP=6) (see dedup_method above) -- unchanged from round (3)."
            ),
            "kalman_deviation_meanrev": (
                "DROPPED ENTIRELY this round -- not imported, not computed, zero contribution to "
                "the candidate pool. See scope.note above for the root-cause finding that motivated "
                "this."
            ),
        },
        "sweep_penetration_atr_sign_note": (
            "Computed to match live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() "
            "exactly (level-minus-extreme for bottom / extreme-minus-level for top, same-bar atr[i]) "
            "-- unchanged from all three prior scripts."
        ),
        "split_boundaries": {
            "train_end": str(TRAIN_END), "oos_start": str(OOS_START), "holdout_start": str(HOLDOUT_START),
        },
        "candidate_pool_comparison_4way_6trigger_vs_8trigger_raw_vs_8trigger_deduped_vs_7trigger_demarker_only": pool_comparison,
        "label_summary_by_split": counts,
        "n_dropped_extra_nan_features": n_dropped_extra,
        "feature_columns": FEATURE_COLUMNS,
        "val": val_result,
        "oos": oos_result,
        "holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "btc_6trigger_benchmark_for_comparison": BTC_6TRIGGER_BENCHMARK,
        "btc_8trigger_raw_rejected_benchmark_for_comparison": BTC_8TRIGGER_RAW_BENCHMARK,
        "btc_8trigger_deduped_benchmark_for_comparison": BTC_8TRIGGER_DEDUPED_BENCHMARK,
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
