#!/usr/bin/env python3
"""BTC V자반등(V_REBOUND) metalabel -- final TabPFN training/eval, TRAIN/VAL/OOS/HOLDOUT.

Final phase of the BTC port of this dashboard's "V자 반등락" specialized-detector signal
(V_REBOUND). Unlike this batch's other 5 evidence signals, V_REBOUND's label formula and
candidate pool were finalized in an EARLIER round and are reused here UNCHANGED (not
re-screened) -- see docs/experiments/btc_5m_v_rebound_gridscreen_featureanalysis_20260901.md
for the grid-screening + feature-analysis round that established: label rate TRAIN/VAL
44.50%/43.81%, excluded-middle rate 61.15%/60.58%, ATR_MULT=1.5 kept as-is, trigger-quality
ranking matches ETH's own order (local_extreme > fib_ext > liquidity_sweep > taker > str_z >
orthogonal_combo).

Candidate pool: 6-trigger union (liquidity_sweep / taker_delta_z_climax / short_term_return_z /
orthogonal_combo / fib_extension_exhaustion / local_extreme) via any_bottom_trigger/
any_top_trigger in the Tier0 CSV -- SMALLER than ETH's current live 9-trigger union, which
additionally ORs smt_divergence (cross-asset, partner asset unresolved for BTC),
demarker_extreme, kalman_deviation_meanrev (both still mid-validation on ETH itself). Stated
explicitly, not glossed over, same as the grid-screening round did.

Label formula: reused VERBATIM (do not re-derive) from research_eth_v_rebound_sweep_gate_
recall_check_90d_20260831.py::realized_outcome, via the same vectorized port that round-1's
research_btc_v_rebound_gridscreen_20260901.py used and self-checked (500-sample cross-validation
against the original loop implementation, 0 mismatches, tolerance 6e-4 to absorb the reference's
3-decimal rounding). That vectorized formula is copied here unmodified (compute_outcome_fields/
label_side) -- only the split boundaries (now extended to OOS/HOLDOUT) and the feature set (now
the 23-feature live-serving convention, see below) are new in this script. Re-verified locally
(anaconda3 env, CPU-only, no GPU needed) against the same reference loop on a fresh random sample
spanning ALL FOUR splits (round-1 only ever touched TRAIN/VAL) before this script was ever pushed
to the GPU server -- see the session's ad hoc verification note in the results doc.

FAST_BARS=6, FULL_BARS=12, ATR_MULT=1.5, CHOP_MULT=1.0, T_SUSTAIN=0.20. label=1 (V자반등) if
fast_mult>=ATR_MULT AND giveback<=T_SUSTAIN; label=0 (지지/횡보) if fast_mult<CHOP_MULT; else
excluded (ambiguous middle, dropped from train/eval entirely -- same ~61%/60.6% exclusion regime
round-1 measured).

Features (23, matches scripts/live_eth_sweep_v_rebound_signal_20260829.py::FEATURES exactly,
is_downside renamed is_bottom): is_bottom, sweep_penetration_atr, atr, atr_percentile_864,
range_width_pct, hour_utc, weekday, delta_z, flow_aligned_delta_z, p_fast, p_slow, ret3_z,
vwap_dev_z, cvd_roll_roc_48, vol_z, lower_wick_ratio, upper_wick_ratio, bb_pctb, adx14, pdi, ndi,
bb_width_pctile, rsi. is_bottom/sweep_penetration_atr/flow_aligned_delta_z are computed the same
way live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() does for ANY firing
candidate regardless of which of the 6 triggers actually fired (sweep_level_low/high treated as a
generic reference level, not sweep-specific) -- verbatim formula:
    penetration = (sweep_level_low[i] - low[i])   if bottom   (POSITIVE = genuine penetration)
                = (high[i] - sweep_level_high[i])  if top
    sweep_penetration_atr = penetration / atr[i]   (same-bar atr, NOT pre_atr/atr[i-1])
    flow_aligned_delta_z = delta_z[i] if bottom else -delta_z[i]
NOTE on a discrepancy found and resolved: the task instructions' inline prose formula for
sweep_penetration_atr ("(extreme - sweep_level_low[i])/atr[i]" for bottom) is the SIGN-FLIPPED
mirror of what _multitrigger_rows() actually computes (which is level-minus-extreme, i.e.
positive for genuine penetration). Since the task explicitly names _multitrigger_rows() as the
ground truth to mirror ("the way the ETH multitrigger script's own _multitrigger_rows() function
does"), this script follows the ACTUAL FUNCTION CODE (verified by direct reading), not the
inline prose -- both for train/serve parity with any future BTC live port of that function, and
because "penetration" should read positive for a real sweep-like breach. This is flagged here and
in the results doc per CLAUDE.md's "state tradeoffs, don't silently pick an interpretation" rule.

Splits (this repo's Fresh-Forward default, CLAUDE.md): TRAIN < 2025-09-01, VAL 2025-09-01..
2026-01-01, OOS 2026-01-01..2026-04-01, HOLDOUT >= 2026-04-01. Full CSV is loaded (unlike round-1,
which deliberately truncated before VAL_END to avoid any OOS/HOLDOUT exposure during screening) --
this run's whole point is to finally touch OOS and, once, HOLDOUT. Forward-outcome windows (up to
FULL_BARS=12 bars ahead) are allowed to read across a split boundary into the next split's price
bars where needed (e.g. a late-VAL candidate's outcome window reads a few OOS bars) -- this is
label construction using already-realized history at build time, not a live-serving causality
violation, and is the same convention this project's other metalabel builds (taker_delta_climax,
ETH's own v_rebound multitrigger) already use unguarded.

TabPFN: SEEDS=[20260829, 141592, 271828, 577215], 4-seed VAL/OOS/HOLDOUT panels (run_tabpfn_panel)
+ single-seed (SEEDS[0]) 5-repeat permutation importance on VAL (compute_permutation_importance).
Both ported near-verbatim from scripts/research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py
(this project's established reusable multi-seed TabPFN panel infrastructure), adapted only for the
"label" column name (that script's fires used "hit").

HOLDOUT is evaluated ONCE in this script (single-touch discipline) -- this is the first time this
signal's BTC work touches HOLDOUT. Nothing about VAL/OOS informs a second HOLDOUT run.

Must run on the GPU server under a system-wide flock (single shared 8GB GPU, up to 6 concurrent
signal agents this session) -- see handoff.sh / the session's own remote invocation for the exact
flock-wrapped command. Not runnable locally (no GPU here, no tabpfn package).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_tabpfn_report.json"
FEATURES_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_v_rebound_metalabel_features_20260901.csv"

# --- outcome formula constants, reused verbatim from research_eth_v_rebound_sweep_gate_recall_
# check_90d_20260831.py::realized_outcome (see module docstring) ---
FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20

# --- fresh-forward split boundaries (CLAUDE.md default) ---
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")   # == VAL start
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")

TRIGGERS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
            "orthogonal_combo", "fib_extension_exhaustion", "local_extreme"]

FEATURE_COLUMNS = [
    "is_bottom", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]  # 23 -- matches live_eth_sweep_v_rebound_signal_20260829.py::FEATURES exactly (is_downside->is_bottom)

SEEDS = [20260829, 141592, 271828, 577215]
RANDOM_STATE = 20260901

ETH_9TRIGGER_BENCHMARK = {
    "source": "live_eth_sweep_v_rebound_signal_20260829.py docstring + docs/homer/README.md "
               "(4-seed VAL/OOS stability -> reserved HOLDOUT, single-touch)",
    "n_triggers": 9,
    "val_auc_mean": 0.8292, "oos_auc_mean": 0.8127, "holdout_auc_mean": 0.8465,
}


def log(msg: str) -> None:
    print(f"[btc_v_rebound_tabpfn] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def compute_outcome_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::compute_outcome_fields.
    Computed once over the FULL (untruncated) frame -- forward windows may read past a split
    boundary, see module docstring."""
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
    """Verbatim port of research_btc_v_rebound_gridscreen_20260901.py::label_side."""
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
    # 20260829.py::_multitrigger_rows() (see module docstring for the sign-discrepancy note vs the
    # task's inline prose). Same-bar atr[i], NOT pre_atr.
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


def build_candidate_pool(df: pd.DataFrame, fields: pd.DataFrame, atr_mult: float = ATR_MULT) -> pd.DataFrame:
    down_labels = label_side(fields, is_down=True, atr_mult=atr_mult)
    up_labels = label_side(fields, is_down=False, atr_mult=atr_mult)
    bottom_mask = df["any_bottom_trigger"].to_numpy()
    top_mask = df["any_top_trigger"].to_numpy()
    bottom_cand = build_side_frame(df, down_labels, bottom_mask, side="bottom")
    top_cand = build_side_frame(df, up_labels, top_mask, side="top")
    return pd.concat([bottom_cand, top_cand], ignore_index=True)


def summarize_split(cand: pd.DataFrame, split: str) -> dict:
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
    """Ported from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::run_tabpfn_panel
    (adapted: "hit" column -> "label")."""
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
    """Ported from research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py::
    compute_permutation_importance (adapted: "hit" column -> "label"). Single-seed, hand-rolled
    permutation importance (AUC-scored) on VAL -- model-agnostic (TabPFN has no native
    .feature_importances_)."""
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


def build_dataset() -> pd.DataFrame:
    """All CPU-only prep (no GPU/tabpfn needed) -- data load through feature-ready candidate pool
    with split assignment. Separated from main() so this can be imported and sanity-checked
    (e.g. locally, without a GPU or the tabpfn package installed) without triggering TabPFN."""
    log("loading BTC Tier0 candidate CSV (full range, TRAIN..HOLDOUT)...")
    df = load_data()
    log(f"  {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    fields = compute_outcome_fields(df)
    cand = build_candidate_pool(df, fields, atr_mult=ATR_MULT)
    return cand


def main() -> int:
    t0 = time.time()
    cand = build_dataset()

    log("label/exclusion summary by split...")
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
        "scope": {
            "candidate_pool_triggers": TRIGGERS,
            "n_triggers_this_run": len(TRIGGERS),
            "live_eth_signal_n_triggers": 9,
            "triggers_excluded_vs_live_eth": ["smt_divergence", "demarker_extreme", "kalman_deviation_meanrev"],
            "note": "BTC candidate pool is structurally smaller than ETH's current live 9-trigger "
                     "union -- stated explicitly, not glossed over, per docs/experiments/"
                     "btc_5m_v_rebound_gridscreen_featureanalysis_20260901.md's own convention.",
        },
        "formula": {
            "fast_bars": FAST_BARS, "full_bars": FULL_BARS,
            "atr_mult": ATR_MULT, "chop_mult": CHOP_MULT, "t_sustain": T_SUSTAIN,
            "source": "research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome "
                       "(verbatim, reused unchanged per task instruction -- not re-derived this round)",
        },
        "sweep_penetration_atr_sign_note": (
            "Computed to match live_eth_sweep_v_rebound_signal_20260829.py::_multitrigger_rows() "
            "exactly (level-minus-extreme for bottom / extreme-minus-level for top, same-bar atr[i]) "
            "-- POSITIVE for genuine penetration. This is the sign-flipped mirror of the task "
            "instructions' inline prose formula; the actual _multitrigger_rows() function (named as "
            "the ground truth to mirror) was followed instead. See module docstring for detail."
        ),
        "split_boundaries": {
            "train_end": str(TRAIN_END), "oos_start": str(OOS_START), "holdout_start": str(HOLDOUT_START),
        },
        "label_summary_by_split": counts,
        "n_dropped_extra_nan_features": n_dropped_extra,
        "feature_columns": FEATURE_COLUMNS,
        "val": val_result,
        "oos": oos_result,
        "holdout": holdout_result,
        "permutation_importance_val": perm_importance,
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
