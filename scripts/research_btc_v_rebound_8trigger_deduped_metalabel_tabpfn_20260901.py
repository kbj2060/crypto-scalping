#!/usr/bin/env python3
"""BTC V자반등(V_REBOUND) metalabel -- 8-TRIGGER candidate pool, DEDUPED fix, TabPFN TRAIN/VAL/OOS/HOLDOUT.

3rd round of this V_REBOUND BTC candidate-pool-widening lineage:
  (1) scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py -- 6-trigger baseline
      (VAL/OOS/HOLDOUT AUC 0.8351/0.8202/0.8277, preserved unchanged).
  (2) scripts/research_btc_v_rebound_8trigger_metalabel_tabpfn_20260901.py -- widened to 8 triggers
      (added demarker_extreme/kalman_deviation_meanrev) using their RAW (non-deduped) per-bar
      trigger booleans OR'd straight into the union. REJECTED: regressed all 3 splits (VAL/OOS/
      HOLDOUT AUC 0.8069/0.8067/0.8189 -- 6-40x the 4-seed noise floor below the 6-trigger baseline,
      see docs/experiments/btc_5m_v_rebound_8trigger_metalabel_tabpfn_20260901.md), despite the
      candidate pool growing ~21-27%. Root cause (confirmed by inspecting the label-rate breakdown):
      demarker/kalman are STATE indicators, not EVENT indicators -- during a sustained trend, `dem`
      can stay pinned <=0.10 (or kalman_dev_z beyond +/-2.0) for many consecutive bars, so raw
      per-bar OR-ing counted one prolonged excursion as 10-15+ separate "candidates," most of them
      low-quality (still mid-excursion, not yet resolved). This diluted the pool's average quality
      and dropped the V자반등 label rate ~8-10pp across every split. That script and its report/doc
      are PRESERVED UNCHANGED (read-only reference) -- this is a NEW file, not an edit of that one,
      per this project's standing convention of keeping every round's script.
  (3) THIS SCRIPT -- the fix the user explicitly asked for: don't raw-union, properly
      discriminate/deduplicate the new signals' fires first. This is NOT a novel technique invented
      here -- both demarker_extreme's and kalman_deviation_meanrev's own standalone BTC research
      scripts (research_btc_demarker_extreme_metalabel_tabpfn_20260901.py,
      research_btc_kalman_deviation_meanrev_metalabel_tabpfn_20260901.py) already solved exactly
      this problem for their own solo candidate pools, and both scored well doing it (DeMarker
      VAL/OOS/HOLDOUT AUC 0.6902/0.6659/0.7286; Kalman 0.7288/0.6242/0.6709). Their `cluster_dedup()`
      function -- collapse same-side raw fires within GAP=6 bars into one cluster, keep only the
      single most-extreme bar per cluster (closest to 0/1 for dem, most negative/positive for
      kalman_dev_z) -- is reused VERBATIM here (imported, not re-derived). Both signals' own scripts
      already fixed GAP=6 for this by earlier task instruction, so GAP=6 is reused as-is here too,
      not re-tuned.

      cluster_dedup() is defined identically (confirmed by direct byte-for-byte comparison of both
      function bodies before writing this) in BOTH the demarker_extreme and kalman_deviation_meanrev
      BTC scripts -- so this script imports it ONCE, from research_btc_demarker_extreme_metalabel_
      tabpfn_20260901.py, and reuses it for both signals rather than duplicating identical code.
      compute_demarker() and compute_kalman_dev_z() are imported verbatim from their respective
      source modules exactly as the raw-8-trigger script already did. All three imported modules are
      import-safe (function/constant definitions only at module scope, GPU/tabpfn imports deferred
      inside their own functions, main() gated behind `if __name__ == "__main__"`) -- confirmed by
      direct read of all three files before writing this, and already proven in practice since the
      raw-8-trigger script imports two of them the same way and ran successfully on the server.

EVERYTHING about the label formula, split boundaries, feature set, and TabPFN infra is reused
BYTE-IDENTICAL from the raw-8-trigger script, which itself reused it byte-identical from the
6-trigger script (FAST_BARS=6, FULL_BARS=12, ATR_MULT=1.5, CHOP_MULT=1.0, T_SUSTAIN=0.20,
label_side()/compute_outcome_fields()/build_side_frame()/summarize_split()/evaluate()/
run_tabpfn_panel()/compute_permutation_importance() all copied verbatim, unmodified). The TRIGGERS
audit-column loop in build_side_frame() is also unchanged -- it still reads RAW per-bar
bottom_/top_demarker_extreme and bottom_/top_kalman_deviation_meanrev columns (same meaning as the
rejected round), so a candidate row's `trig_demarker_extreme`/`trig_kalman_deviation_meanrev` audit
flags stay comparable across all three rounds. The ONLY change is which UNION column feeds
build_candidate_pool(): this run adds two new DEDUPED boolean columns
(bottom_/top_demarker_extreme_deduped, bottom_/top_kalman_deviation_meanrev_deduped -- True only at
the kept cluster-representative bar) and unions THOSE instead of the raw per-bar columns:

  any_bottom_trigger_8dedup = any_bottom_trigger | bottom_demarker_extreme_deduped | bottom_kalman_deviation_meanrev_deduped
  any_top_trigger_8dedup    = any_top_trigger    | top_demarker_extreme_deduped    | top_kalman_deviation_meanrev_deduped

The raw (rejected-round) any_bottom_trigger_8bit/any_top_trigger_8bit union is ALSO recomputed here
(unchanged formula) purely for a 3-way before/after/fixed comparison (6-trigger vs 8-trigger RAW vs
8-trigger DEDUPED), all computed fresh within this one run for direct in-run comparability -- not
just quoting the two prior scripts' separately-saved reports (those numbers are also embedded below
as BTC_6TRIGGER_BENCHMARK / BTC_8TRIGGER_RAW_BENCHMARK for a sanity cross-check against this run's
own from-scratch recomputation).

Additional diagnostic (NOT a further dedup pass, report-only per task instruction): after deduping
demarker_extreme and kalman_deviation_meanrev individually, log_residual_cross_signal_overlap()
checks how many of their kept (cluster-representative) bars still land within GAP=6 bars of EACH
OTHER, or of an original-6-signal raw trigger bar -- pure index-proximity count/rate, does not
change the candidate pool built above.

Tier0 CSV, split boundaries, TabPFN SEEDS, sweep_penetration_atr sign convention: all identical to
the 6-trigger and raw-8-trigger scripts -- see the 6-trigger script's module docstring for the full
history (grid-screening round, the sign-note resolution, etc.), not repeated here.

HOLDOUT is evaluated ONCE in this script (single-touch discipline) -- this is the first time this
SPECIFIC deduped-8-trigger candidate pool touches HOLDOUT (a distinct pool definition from both the
6-trigger run's and the raw-8-trigger run's own, separately single-touched, HOLDOUT exposures).

Must run on the GPU server under the same system-wide flock as every other TabPFN script this
session (single shared 8GB GPU): /home/llewyn/crypto-scalping/.tabpfn_gpu.lock. Not runnable
locally for the TabPFN portion (no GPU, no tabpfn package) -- build_dataset() itself is CPU-only
(candidate-pool assembly, importable/testable without GPU or tabpfn), same convention as the prior
two scripts.
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
from research_btc_demarker_extreme_metalabel_tabpfn_20260901 import cluster_dedup  # noqa: E402
# ^ cluster_dedup is defined identically in research_btc_kalman_deviation_meanrev_metalabel_tabpfn_
# 20260901.py (confirmed by direct comparison before writing this) -- imported once here, reused for
# both signals, rather than duplicating the same function body twice.

DATA_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_8trigger_deduped_tabpfn_report.json"
FEATURES_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_v_rebound_8trigger_deduped_metalabel_features_20260901.csv"

# --- outcome formula constants, reused verbatim from the 6-trigger/raw-8-trigger scripts ---
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
NEW_TRIGGERS = ["demarker_extreme", "kalman_deviation_meanrev"]
TRIGGERS = TRIGGERS_6 + NEW_TRIGGERS  # 8 -- used for the trig_{name} audit columns in build_side_frame
                                       # (unchanged meaning vs the raw-8-trigger script: these audit
                                       # flags stay RAW per-bar, not deduped -- see module docstring)

CLUSTER_GAP = 6  # fixed, reused verbatim from both demarker_extreme's and kalman_deviation_meanrev's
                 # own already-validated BTC solo scripts -- NOT re-tuned here (task instruction).

FEATURE_COLUMNS = [
    "is_bottom", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]  # 23 -- UNCHANGED from both prior scripts; only the candidate-pool input changes this round,
   # NOT the feature set (dem/kalman_dev_z are pool triggers only, not added as features)

SEEDS = [20260829, 141592, 271828, 577215]
RANDOM_STATE = 20260901

BTC_6TRIGGER_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_metalabel_tabpfn_20260901.py (this project's own "
              "6-trigger baseline round, preserved unchanged) via its server-side "
              "v_rebound_tabpfn_report.json",
    "n_triggers": 6,
    "n_train": 13185,
    "val_auc_mean": 0.8351, "oos_auc_mean": 0.8202, "holdout_auc_mean": 0.8277,
}

BTC_8TRIGGER_RAW_BENCHMARK = {
    "source": "scripts/research_btc_v_rebound_8trigger_metalabel_tabpfn_20260901.py (this project's "
              "own immediately-preceding, REJECTED round -- raw/non-deduped per-bar OR union, "
              "preserved unchanged) via its server-side v_rebound_8trigger_tabpfn_report.json",
    "n_triggers": 8,
    "dedup": "NONE (raw per-bar OR union) -- docs/experiments/btc_5m_v_rebound_8trigger_metalabel_"
             "tabpfn_20260901.md found this round REGRESSED all 3 splits vs the 6-trigger baseline",
    "n_train": 17548,
    "val_auc_mean": 0.8069, "oos_auc_mean": 0.8067, "holdout_auc_mean": 0.8189,
}

ETH_9TRIGGER_BENCHMARK = {
    "source": "live_eth_sweep_v_rebound_signal_20260829.py docstring + docs/homer/README.md "
               "(4-seed VAL/OOS stability -> reserved HOLDOUT, single-touch)",
    "n_triggers": 9,
    "val_auc_mean": 0.8292, "oos_auc_mean": 0.8127, "holdout_auc_mean": 0.8465,
}


def log(msg: str) -> None:
    print(f"[btc_v_rebound_8trigger_deduped_tabpfn] {msg}", flush=True)


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
    True only at the kept cluster-representative bars, same length as raw_trig."""
    idx = np.flatnonzero(raw_trig)
    idx_before = len(idx)
    kept_idx = cluster_dedup(idx, extremeness[idx], most_negative=most_negative, gap=gap)
    out = np.zeros(len(raw_trig), dtype=bool)
    out[kept_idx] = True
    return out, {"raw": int(idx_before), "deduped": int(len(kept_idx))}


def add_new_triggers(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Compute demarker_extreme and kalman_deviation_meanrev triggers fresh on this Tier0
    dataframe's own high/low/close columns, in BOTH raw (whole-history per-bar, unchanged formula
    from the rejected raw-8-trigger script) and cluster-deduped form (the fix, GAP=6 -- see module
    docstring). Writes three union pairs onto df:
      any_bottom_trigger        original 6-trigger union, already in the Tier0 CSV, untouched.
      any_bottom_trigger_8bit   6-trigger | RAW demarker | RAW kalman -- recomputed here (byte-
                                 identical formula to the rejected round) purely for the 3-way
                                 pool-comparison log below.
      any_bottom_trigger_8dedup 6-trigger | DEDUPED demarker | DEDUPED kalman -- USED for training.
    (top_* mirror each.) Does NOT modify the Tier0 CSV on disk -- these columns exist only in this
    in-memory df. Returns (df, dedup_stats) where dedup_stats carries raw->deduped fire counts for
    both signals/sides, for the report JSON."""
    dem = compute_demarker(df["high"], df["low"]).to_numpy()
    df["bottom_demarker_extreme"] = dem <= 0.10
    df["top_demarker_extreme"] = dem >= 0.90

    kalman_dev_z = pd.Series(compute_kalman_dev_z(df["close"].to_numpy()))
    df["bottom_kalman_deviation_meanrev"] = (kalman_dev_z <= -2.0).fillna(False).to_numpy()
    df["top_kalman_deviation_meanrev"] = (kalman_dev_z >= 2.0).fillna(False).to_numpy()
    kalman_dev_z = kalman_dev_z.to_numpy()

    log(f"  fresh trigger RAW fire counts (whole history, pre-dedup, pre-union): "
        f"demarker_extreme bottom={int(df['bottom_demarker_extreme'].sum())} "
        f"top={int(df['top_demarker_extreme'].sum())}, "
        f"kalman_deviation_meanrev bottom={int(df['bottom_kalman_deviation_meanrev'].sum())} "
        f"top={int(df['top_kalman_deviation_meanrev'].sum())}")

    dem_bottom_dd, dem_bottom_stats = dedup_trigger(
        df["bottom_demarker_extreme"].to_numpy(), dem, most_negative=True, gap=CLUSTER_GAP)
    dem_top_dd, dem_top_stats = dedup_trigger(
        df["top_demarker_extreme"].to_numpy(), dem, most_negative=False, gap=CLUSTER_GAP)
    kal_bottom_dd, kal_bottom_stats = dedup_trigger(
        df["bottom_kalman_deviation_meanrev"].to_numpy(), kalman_dev_z, most_negative=True, gap=CLUSTER_GAP)
    kal_top_dd, kal_top_stats = dedup_trigger(
        df["top_kalman_deviation_meanrev"].to_numpy(), kalman_dev_z, most_negative=False, gap=CLUSTER_GAP)

    df["bottom_demarker_extreme_deduped"] = dem_bottom_dd
    df["top_demarker_extreme_deduped"] = dem_top_dd
    df["bottom_kalman_deviation_meanrev_deduped"] = kal_bottom_dd
    df["top_kalman_deviation_meanrev_deduped"] = kal_top_dd

    dedup_stats = {
        "demarker_extreme": {"bottom": dem_bottom_stats, "top": dem_top_stats},
        "kalman_deviation_meanrev": {"bottom": kal_bottom_stats, "top": kal_top_stats},
        "cluster_gap": CLUSTER_GAP,
    }
    log(f"  demarker_extreme cluster-dedup (GAP={CLUSTER_GAP}): "
        f"bottom {dem_bottom_stats['raw']} -> {dem_bottom_stats['deduped']}, "
        f"top {dem_top_stats['raw']} -> {dem_top_stats['deduped']}")
    log(f"  kalman_deviation_meanrev cluster-dedup (GAP={CLUSTER_GAP}): "
        f"bottom {kal_bottom_stats['raw']} -> {kal_bottom_stats['deduped']}, "
        f"top {kal_top_stats['raw']} -> {kal_top_stats['deduped']}")

    # raw (rejected-round) union -- recomputed here for the 3-way pool comparison, byte-identical
    # formula to research_btc_v_rebound_8trigger_metalabel_tabpfn_20260901.py::add_new_triggers
    df["any_bottom_trigger_8bit"] = (
        df["any_bottom_trigger"] | df["bottom_demarker_extreme"] | df["bottom_kalman_deviation_meanrev"]
    )
    df["any_top_trigger_8bit"] = (
        df["any_top_trigger"] | df["top_demarker_extreme"] | df["top_kalman_deviation_meanrev"]
    )
    # deduped union -- THE FIX, used for training below
    df["any_bottom_trigger_8dedup"] = (
        df["any_bottom_trigger"] | df["bottom_demarker_extreme_deduped"] | df["bottom_kalman_deviation_meanrev_deduped"]
    )
    df["any_top_trigger_8dedup"] = (
        df["any_top_trigger"] | df["top_demarker_extreme_deduped"] | df["top_kalman_deviation_meanrev_deduped"]
    )
    log(f"  any_bottom_trigger: {int(df['any_bottom_trigger'].sum())}  ->  "
        f"8bit(raw)={int(df['any_bottom_trigger_8bit'].sum())}  ->  "
        f"8dedup={int(df['any_bottom_trigger_8dedup'].sum())}  |  "
        f"any_top_trigger: {int(df['any_top_trigger'].sum())}  ->  "
        f"8bit(raw)={int(df['any_top_trigger_8bit'].sum())}  ->  "
        f"8dedup={int(df['any_top_trigger_8dedup'].sum())}")

    return df, dedup_stats


def frac_within_gap(query_idx: np.ndarray, ref_idx: np.ndarray, gap: int) -> dict:
    """For each element of query_idx, is there an element of ref_idx within `gap` bars (inclusive,
    either direction)? Pure index-proximity diagnostic (task point 4c) -- not a further dedup pass,
    does not modify any candidate pool."""
    if len(query_idx) == 0:
        return {"n_query": 0, "n_within_gap": 0, "rate": None}
    if len(ref_idx) == 0:
        return {"n_query": int(len(query_idx)), "n_within_gap": 0, "rate": 0.0}
    ref_sorted = np.sort(ref_idx)
    pos = np.searchsorted(ref_sorted, query_idx)
    pos_right = np.clip(pos, 0, len(ref_sorted) - 1)
    pos_left = np.clip(pos - 1, 0, len(ref_sorted) - 1)
    dist_right = np.abs(ref_sorted[pos_right] - query_idx)
    dist_left = np.abs(ref_sorted[pos_left] - query_idx)
    min_dist = np.minimum(dist_right, dist_left)
    n_within = int((min_dist <= gap).sum())
    return {"n_query": int(len(query_idx)), "n_within_gap": n_within,
            "rate": round(n_within / len(query_idx), 4)}


def log_residual_cross_signal_overlap(df: pd.DataFrame) -> dict:
    """Diagnostic only (task point 4c): after deduping demarker_extreme/kalman_deviation_meanrev
    individually (each against ITSELF, same-signal same-side), how many of their kept
    cluster-representative bars still land within GAP=6 bars of EACH OTHER, or of an
    original-6-signal raw trigger bar? Reported per side (bottom/top) since bottom/top are
    structurally separate candidate pools. NOT used to further filter/change the candidate pool
    built in build_dataset() -- report/log only, per task instruction."""
    orig6_bottom_idx = np.flatnonzero(df["any_bottom_trigger"].to_numpy())
    orig6_top_idx = np.flatnonzero(df["any_top_trigger"].to_numpy())
    dem_bottom_idx = np.flatnonzero(df["bottom_demarker_extreme_deduped"].to_numpy())
    dem_top_idx = np.flatnonzero(df["top_demarker_extreme_deduped"].to_numpy())
    kal_bottom_idx = np.flatnonzero(df["bottom_kalman_deviation_meanrev_deduped"].to_numpy())
    kal_top_idx = np.flatnonzero(df["top_kalman_deviation_meanrev_deduped"].to_numpy())

    result = {}
    for side, orig_idx, dem_idx, kal_idx in (
        ("bottom", orig6_bottom_idx, dem_bottom_idx, kal_bottom_idx),
        ("top", orig6_top_idx, dem_top_idx, kal_top_idx),
    ):
        result[side] = {
            "demarker_deduped_within_gap_of_kalman_deduped": frac_within_gap(dem_idx, kal_idx, CLUSTER_GAP),
            "kalman_deduped_within_gap_of_demarker_deduped": frac_within_gap(kal_idx, dem_idx, CLUSTER_GAP),
            "demarker_deduped_within_gap_of_orig6": frac_within_gap(dem_idx, orig_idx, CLUSTER_GAP),
            "kalman_deduped_within_gap_of_orig6": frac_within_gap(kal_idx, orig_idx, CLUSTER_GAP),
        }
        log(f"  [residual overlap diag, {side}] demarker(dedup)~kalman(dedup): "
            f"{result[side]['demarker_deduped_within_gap_of_kalman_deduped']}")
        log(f"  [residual overlap diag, {side}] kalman(dedup)~demarker(dedup): "
            f"{result[side]['kalman_deduped_within_gap_of_demarker_deduped']}")
        log(f"  [residual overlap diag, {side}] demarker(dedup)~orig6(raw): "
            f"{result[side]['demarker_deduped_within_gap_of_orig6']}")
        log(f"  [residual overlap diag, {side}] kalman(dedup)~orig6(raw): "
            f"{result[side]['kalman_deduped_within_gap_of_orig6']}")
    return result


def compute_outcome_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::compute_outcome_fields,
    unchanged from the 6-trigger and raw-8-trigger scripts. Computed once over the FULL
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
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::label_side, unchanged from
    the 6-trigger and raw-8-trigger scripts."""
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
    """Unchanged from the 6-trigger/raw-8-trigger scripts. The trig_ loop below reads the RAW
    (non-deduped) bottom_/top_ columns add_new_triggers() writes onto df -- same meaning as the
    rejected round, kept that way deliberately for cross-round audit-column comparability (see
    module docstring)."""
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
    """Same structure as the prior scripts' build_candidate_pool, parameterized by which bottom/top
    mask column to union on -- lets this script build the 6-trigger pool, the raw-8-trigger pool,
    and the deduped-8-trigger pool (the one actually used downstream) from identical labeling
    logic, for a true apples-to-apples 3-way comparison."""
    down_labels = label_side(fields, is_down=True, atr_mult=atr_mult)
    up_labels = label_side(fields, is_down=False, atr_mult=atr_mult)
    bottom_mask = df[bottom_col].to_numpy()
    top_mask = df[top_col].to_numpy()
    bottom_cand = build_side_frame(df, down_labels, bottom_mask, side="bottom")
    top_cand = build_side_frame(df, up_labels, top_mask, side="top")
    return pd.concat([bottom_cand, top_cand], ignore_index=True)


def summarize_split(cand: pd.DataFrame, split: str) -> dict:
    """Unchanged from the prior scripts."""
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


def log_pool_comparison_3way(cand_6: pd.DataFrame, cand_8raw: pd.DataFrame, cand_8dedup: pd.DataFrame) -> dict:
    """3-way candidate-pool size / label-rate comparison, ALL computed fresh within THIS run: the
    6-trigger baseline (unchanged trigger columns), the REJECTED raw-8-trigger union (recomputed
    here, byte-identical formula, for direct in-run comparability with the already-saved
    v_rebound_8trigger_tabpfn_report.json), and the deduped-8-trigger union actually used for
    training below."""
    log("=== candidate pool size comparison: 6-trigger vs 8-trigger RAW (rejected) vs 8-trigger DEDUPED (this run) ===")
    comparison = {}
    for split in ("TRAIN", "VAL", "OOS", "HOLDOUT"):
        c6 = summarize_split(cand_6, split)
        c8r = summarize_split(cand_8raw, split)
        c8d = summarize_split(cand_8dedup, split)
        log(f"  {split}: candidates  6trig={c6['candidate_count_total']}  "
            f"8trig_raw={c8r['candidate_count_total']}  8trig_dedup={c8d['candidate_count_total']}   |   "
            f"label_rate  6trig={c6['label_rate_v_rebound_of_labeled']}  "
            f"8trig_raw={c8r['label_rate_v_rebound_of_labeled']}  "
            f"8trig_dedup={c8d['label_rate_v_rebound_of_labeled']}")
        r6 = c6["label_rate_v_rebound_of_labeled"]
        r8r = c8r["label_rate_v_rebound_of_labeled"]
        r8d = c8d["label_rate_v_rebound_of_labeled"]
        comparison[split] = {
            "6trigger": c6, "8trigger_raw_rejected": c8r, "8trigger_deduped": c8d,
            "delta_candidate_count_dedup_vs_6trigger": c8d["candidate_count_total"] - c6["candidate_count_total"],
            "delta_candidate_count_dedup_vs_8raw": c8d["candidate_count_total"] - c8r["candidate_count_total"],
            "delta_label_rate_dedup_vs_6trigger": round(r8d - r6, 4) if r8d is not None and r6 is not None else None,
            "delta_label_rate_dedup_vs_8raw": round(r8d - r8r, 4) if r8d is not None and r8r is not None else None,
        }
    return comparison


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    """Unchanged from the prior scripts."""
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str], tag: str) -> dict:
    """Unchanged from the prior scripts (ported from research_eth_taker_delta_climax_metalabel_
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
    """Unchanged from the prior scripts. Single-seed, hand-rolled permutation importance (AUC-scored)
    on VAL -- model-agnostic (TabPFN has no native .feature_importances_)."""
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


def build_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, dict, dict]:
    """All CPU-only prep (no GPU/tabpfn needed) -- data load through feature-ready deduped-8-trigger
    candidate pool with split assignment, PLUS the 6-trigger and raw-8-trigger pools for the 3-way
    before/after/fixed comparison. Separated from main() so this can be imported and sanity-checked
    locally (no GPU/tabpfn) without triggering TabPFN, same convention as the prior two scripts."""
    log("loading BTC Tier0 candidate CSV (full range, TRAIN..HOLDOUT)...")
    df = load_data()
    log(f"  {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    log("computing fresh demarker_extreme/kalman_deviation_meanrev triggers "
        "(raw + cluster-deduped GAP=6, reused verbatim from each signal's own solo BTC script)...")
    df, dedup_stats = add_new_triggers(df)

    log("=== residual cross-signal overlap diagnostic (report-only, NOT a further dedup pass) ===")
    residual_overlap = log_residual_cross_signal_overlap(df)

    fields = compute_outcome_fields(df)
    cand_6 = build_candidate_pool(df, fields, "any_bottom_trigger", "any_top_trigger", atr_mult=ATR_MULT)
    cand_8raw = build_candidate_pool(df, fields, "any_bottom_trigger_8bit", "any_top_trigger_8bit", atr_mult=ATR_MULT)
    cand_8dedup = build_candidate_pool(df, fields, "any_bottom_trigger_8dedup", "any_top_trigger_8dedup", atr_mult=ATR_MULT)
    pool_comparison = log_pool_comparison_3way(cand_6, cand_8raw, cand_8dedup)
    return cand_6, cand_8raw, cand_8dedup, dedup_stats, residual_overlap, pool_comparison


def main() -> int:
    t0 = time.time()
    cand_6, cand_8raw, cand, dedup_stats, residual_overlap, pool_comparison = build_dataset()

    log("label/exclusion summary by split (DEDUPED 8-trigger pool, used for training below)...")
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
            "round": (
                "3rd round of this V_REBOUND BTC candidate-pool-widening lineage: (1) 6-trigger "
                "baseline, (2) raw/non-deduped 8-trigger union -- REJECTED, regressed all 3 splits, "
                "(3) THIS RUN -- same 8 triggers, but demarker_extreme and kalman_deviation_meanrev "
                "are now cluster-deduped (GAP=6) before the union, reusing each signal's own "
                "already-validated solo-BTC-script dedup verbatim."
            ),
            "note": (
                "Fix attempt for the raw-8-trigger regression documented in docs/experiments/"
                "btc_5m_v_rebound_8trigger_metalabel_tabpfn_20260901.md -- root cause there was "
                "diagnosed as demarker/kalman being STATE indicators (can stay pinned past their "
                "threshold for many consecutive bars during a sustained trend) rather than EVENT "
                "indicators, so raw per-bar OR-union counted one prolonged excursion as 10-15+ "
                "separate low-quality 'candidates'. This run replaces the raw per-bar trigger with "
                "each signal's own cluster-deduped (GAP=6, keep only the most-extreme bar per "
                "same-side cluster) fires before unioning into the pool."
            ),
        },
        "dedup_method": {
            "cluster_gap": CLUSTER_GAP,
            "algorithm": (
                "cluster_dedup() imported verbatim from research_btc_demarker_extreme_metalabel_"
                "tabpfn_20260901.py (confirmed byte-identical, by direct comparison, to research_btc_"
                "kalman_deviation_meanrev_metalabel_tabpfn_20260901.py's own copy of the same "
                "function -- so importing once serves both signals rather than duplicating identical "
                "code). Collapses same-side raw fires within GAP bars into one cluster, keeps only "
                "the single most-extreme bar per cluster (closest to 0/1 for dem, most negative/"
                "positive for kalman_dev_z). NOT re-tuned here -- GAP=6 reused exactly as already "
                "fixed by task instruction in both signals' own solo BTC scripts."
            ),
            "dedup_stats": dedup_stats,
        },
        "residual_cross_signal_overlap_diagnostic": residual_overlap,
        "formula": {
            "fast_bars": FAST_BARS, "full_bars": FULL_BARS,
            "atr_mult": ATR_MULT, "chop_mult": CHOP_MULT, "t_sustain": T_SUSTAIN,
            "source": "research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome "
                       "(verbatim, unchanged from the 6-trigger and raw-8-trigger scripts -- not "
                       "re-derived this round)",
        },
        "new_trigger_formulas": {
            "demarker_extreme": (
                "compute_demarker(high, low) imported verbatim from research_eth_demarker_evidence_"
                "signal_lift_check_20260831.py; bottom: dem<=0.10, top: dem>=0.90, THEN "
                "cluster_dedup(GAP=6) (see dedup_method above) -- raw threshold unchanged from the "
                "rejected round, only the union input changed."
            ),
            "kalman_deviation_meanrev": (
                "compute_kalman_dev_z(close) imported verbatim from research_btc_kalman_deviation_"
                "meanrev_metalabel_tabpfn_20260901.py; bottom: kalman_dev_z<=-2.0, top: "
                "kalman_dev_z>=2.0, THEN cluster_dedup(GAP=6)."
            ),
        },
        "sweep_penetration_atr_sign_note": (
            "Computed to match live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() "
            "exactly (level-minus-extreme for bottom / extreme-minus-level for top, same-bar atr[i]) "
            "-- unchanged from the 6-trigger and raw-8-trigger scripts."
        ),
        "split_boundaries": {
            "train_end": str(TRAIN_END), "oos_start": str(OOS_START), "holdout_start": str(HOLDOUT_START),
        },
        "candidate_pool_comparison_3way_6trigger_vs_8trigger_raw_vs_8trigger_deduped": pool_comparison,
        "label_summary_by_split": counts,
        "n_dropped_extra_nan_features": n_dropped_extra,
        "feature_columns": FEATURE_COLUMNS,
        "val": val_result,
        "oos": oos_result,
        "holdout": holdout_result,
        "permutation_importance_val": perm_importance,
        "btc_6trigger_benchmark_for_comparison": BTC_6TRIGGER_BENCHMARK,
        "btc_8trigger_raw_rejected_benchmark_for_comparison": BTC_8TRIGGER_RAW_BENCHMARK,
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
