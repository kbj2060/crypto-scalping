#!/usr/bin/env python3
"""BTC V자반등(V_REBOUND) grid-screening + feature analysis -- SCREENING ONLY.

User request 2026-09-01: grid-screen + feature-analyze BTC's own version of the "V자 반등락"
specialized detector. Explicitly SCOPED to grid-screening + feature analysis only -- no TabPFN
training, no economic/cost-gate backtest, no HOLDOUT exposure. Those are future work pending
human review of this screening pass.

Candidate pool (6-trigger union, NOT the 9-trigger union the live ETH signal currently uses):
  any_bottom_trigger / any_top_trigger in data/labels/btc_5m_evidence_signal_candidates_20260901/
  btc_5m_evidence_signal_candidates_tier0.csv -- union of liquidity_sweep, taker_delta_z_climax,
  short_term_return_z, orthogonal_combo, fib_extension_exhaustion, local_extreme. The live ETH
  V_REBOUND signal additionally ORs smt_divergence, demarker_extreme, kalman_deviation_meanrev
  (see build_btc_5m_evidence_signal_candidates_tier0_20260901.py's docstring for why those 3 are
  excluded here -- cross-asset-ambiguous / not-yet-validated-for-BTC / out of this round's scope).
  This is a smaller candidate pool than ETH's current live one -- stated explicitly, not glossed
  over, per user instruction.

Outcome/label formula: reused VERBATIM from research_eth_v_rebound_sweep_gate_recall_check_90d_
20260831.py::realized_outcome (imported directly below and used for a self-check cross-validation
against this script's own vectorized re-implementation -- NOT redesigned; only the candidate pool
is BTC-specific, formula and candidate-selection are deliberately decoupled axes in this project's
convention). FAST_BARS/FULL_BARS/ATR_MULT/T_SUSTAIN are fixed, reused as-is (ATR_MULT=1.5 is the
primary/recommended setting; a light sensitivity check at 1.25/1.75 is reported separately, not a
re-optimization).

Fresh-forward split (CLAUDE.md default boundaries): TRAIN < 2025-09-01, VAL 2025-09-01..2025-12-31,
OOS 2026-01-01..2026-03-31, HOLDOUT >= 2026-04-01. This script NEVER loads OOS or HOLDOUT rows --
the raw frame is truncated to timestamp < VAL_END (== OOS start) immediately after load, before any
computation. This means the outcome-window lookforward for the last few VAL-dated candidates (within
1h / 12 bars of 2025-12-31 23:55) has no data to look into and is correctly excluded as "insufficient
forward data" rather than reading into OOS -- a deliberate, conservative tradeoff (a small, expected
handful of late-VAL candidates lost) in exchange for a structural (not just conventional) guarantee
that OOS/HOLDOUT are untouched this round.

Run with the quant_ai conda env:
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_v_rebound_gridscreen_20260901.py
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA_CSV = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/v_rebound_gridscreen_report.json"
REF_SCRIPT = ROOT / "scripts/research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py"

# --- outcome formula constants, reused verbatim (see docstring) ---
FAST_BARS = 6
FULL_BARS = 12
ATR_MULT = 1.5
CHOP_MULT = 1.0
T_SUSTAIN = 0.20
ATR_MULT_SENSITIVITY = [1.25, 1.75]

# --- fresh-forward split boundaries ---
TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")  # == OOS start. Raw frame truncated here; OOS/HOLDOUT never loaded.

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]  # 21 Tier0 + rsi = 22
TRIGGERS = ["liquidity_sweep", "taker_delta_z_climax", "short_term_return_z",
            "orthogonal_combo", "fib_extension_exhaustion", "local_extreme"]
RANDOM_STATE = 20260901


def load_ref_impl():
    """Loads research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py as a module purely to
    reuse its realized_outcome() for a self-check cross-validation. exec_module only runs top-level
    (import/def/const) code -- main()'s network fetch is guarded by if __name__=='__main__' and is
    never triggered."""
    spec = importlib.util.spec_from_file_location("v_rebound_ref_impl_20260901", REF_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.loc[df["timestamp"] < VAL_END].sort_values("timestamp").reset_index(drop=True)
    return df


def compute_outcome_fields(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized port of realized_outcome()'s arithmetic, computed once for the whole (truncated)
    frame; candidate rows are selected afterward. Forward-window pattern (reverse -> rolling -> reverse
    -> shift(-1)) matches this project's own established idiom, e.g. docs/experiments/
    eth_short_term_return_z_metalabel_20260829.md's fwd_high_max."""
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


def build_side_frame(df: pd.DataFrame, labels: pd.DataFrame, mask: np.ndarray, side: str) -> pd.DataFrame:
    idx = np.flatnonzero(mask)
    sub = df.iloc[idx].copy()
    lab = labels.iloc[idx]
    sub["side"] = side
    sub["fast_mult"] = lab["fast_mult"].to_numpy()
    sub["giveback"] = lab["giveback"].to_numpy()
    valid = lab["valid"].to_numpy()
    label_raw = lab["label_raw"].to_numpy()
    status = np.where(~valid, "invalid_insufficient_data",
              np.where(label_raw == 1, "v_rebound",
              np.where(label_raw == 0, "chop_support", "ambiguous_excluded")))
    sub["status"] = status
    sub["label"] = np.where(status == "v_rebound", 1.0, np.where(status == "chop_support", 0.0, np.nan))
    sub["split"] = np.where(sub["timestamp"] < TRAIN_END, "TRAIN", "VAL")  # sub is already truncated to < VAL_END
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
        "by_side": {
            side: {
                "candidate_count": int((pool["side"] == side).sum()),
                "v_rebound": int(((pool["side"] == side) & (pool["status"] == "v_rebound")).sum()),
                "chop_support": int(((pool["side"] == side) & (pool["status"] == "chop_support")).sum()),
                "ambiguous_excluded": int(((pool["side"] == side) & (pool["status"] == "ambiguous_excluded")).sum()),
            }
            for side in ("bottom", "top")
        },
    }


def self_check(df: pd.DataFrame, cand: pd.DataFrame, ref, n_sample: int = 500) -> dict:
    """Cross-validate this script's vectorized labels against the reference script's own
    realized_outcome() loop implementation, called directly (not re-derived) on a random sample."""
    checkable = cand.loc[cand["status"] != "invalid_insufficient_data"]
    # exclude idx==0 candidates defensively (ref fn does frame['atr'].iloc[idx-1], which wraps at idx=0)
    sample = checkable.sample(n=min(n_sample, len(checkable)), random_state=RANDOM_STATE)
    mismatches = []
    n_checked = 0
    for row_pos, row in sample.iterrows():
        # recover original df row index by matching timestamp (unique, 5-min bars)
        orig_idx = df.index[df["timestamp"] == row["timestamp"]][0]
        if orig_idx == 0:
            continue
        is_down = row["side"] == "bottom"
        ref_out = ref.realized_outcome(df, orig_idx, is_down)
        n_checked += 1
        if ref_out is None or ref_out["partial_window"]:
            mismatches.append({"idx": int(orig_idx), "reason": "ref_none_or_partial", "ref": ref_out})
            continue
        ref_label = {"V자반등": "v_rebound", "지지/횡보": "chop_support", "애매(제외권)": "ambiguous_excluded"}[ref_out["outcome"]]
        ok_status = ref_label == row["status"]
        # ref_out's floats are round(x, 3) -- tolerance must absorb that rounding, not just fp noise
        ok_fast = abs(ref_out["fast_move_atr_mult"] - row["fast_mult"]) < 6e-4
        ok_gb = (ref_out["giveback_ratio"] is None and pd.isna(row["giveback"])) or (
            ref_out["giveback_ratio"] is not None and not pd.isna(row["giveback"])
            and abs(ref_out["giveback_ratio"] - row["giveback"]) < 6e-4
        )
        if not (ok_status and ok_fast and ok_gb):
            mismatches.append({
                "idx": int(orig_idx), "side": row["side"],
                "mine": {"status": row["status"], "fast_mult": row["fast_mult"], "giveback": row["giveback"]},
                "ref": ref_out,
            })
    return {"n_checked": n_checked, "n_mismatches": len(mismatches), "mismatches_sample": mismatches[:10]}


def atr_mult_sensitivity(df: pd.DataFrame, fields: pd.DataFrame) -> dict:
    out = {}
    for mult in [ATR_MULT] + ATR_MULT_SENSITIVITY:
        cand = build_candidate_pool(df, fields, atr_mult=mult)
        out[str(mult)] = {split: summarize_split(cand, split) for split in ("TRAIN", "VAL")}
    return out


def trigger_quality(cand: pd.DataFrame, split: str) -> dict:
    """Per-trigger V자반등 rate, reported two ways:
    - v_rebound_rate_of_labeled: among that trigger's labeled (non-ambiguous) candidates only.
    - v_rebound_rate_of_total_fired: V자반등 / ALL candidates that fired that trigger (ambiguous
      counted as a miss in the denominator) -- this is the SAME convention docs/homer/README.md's
      9-trigger ETH multitrigger build uses for its own "트리거별 적중률" table (line ~414-418:
      sweep 15.7%/taker 13.2%/str_z 13.6%/orthogonal_combo 12.2%/fib_ext 19.9%/local_extreme 22.2%,
      computed against each trigger's full fired population) -- reported here for a direct
      apples-to-apples comparison against that established ETH benchmark.
    """
    pool = cand.loc[(cand["split"] == split) & (cand["status"] != "invalid_insufficient_data")]
    out = {}
    for trig in TRIGGERS:
        fired = pool.loc[pool[f"trig_{trig}"].astype(bool)]
        labeled = fired.loc[fired["status"] != "ambiguous_excluded"]
        n_v = int((labeled["status"] == "v_rebound").sum())
        n_labeled = len(labeled)
        n_fired = len(fired)
        out[trig] = {
            "n_fired": int(n_fired),
            "n_labeled": n_labeled,
            "n_ambiguous_excluded": int(n_fired - n_labeled),
            "v_rebound_rate_of_labeled": round(n_v / n_labeled, 4) if n_labeled else None,
            "v_rebound_rate_of_total_fired": round(n_v / n_fired, 4) if n_fired else None,
        }
    return dict(sorted(out.items(), key=lambda kv: (kv[1]["v_rebound_rate_of_total_fired"] is None,
                                                      -(kv[1]["v_rebound_rate_of_total_fired"] or 0))))


def feature_analysis(cand: pd.DataFrame) -> dict:
    train = cand.loc[(cand["split"] == "TRAIN") & (cand["status"] != "invalid_insufficient_data")
                      & (cand["status"] != "ambiguous_excluded")].copy()
    val = cand.loc[(cand["split"] == "VAL") & (cand["status"] != "invalid_insufficient_data")
                    & (cand["status"] != "ambiguous_excluded")].copy()

    # --- 1) per-feature mean/std separation + correlation, pooled bottom+top, TRAIN ---
    corr_rows = []
    for feat in TIER0_FEATURES:
        s = train[[feat, "label"]].dropna()
        if len(s) < 30:
            corr_rows.append({"feature": feat, "n": len(s), "corr": None, "mean_label1": None,
                               "mean_label0": None, "std_label1": None, "std_label0": None})
            continue
        corr = float(s[feat].corr(s["label"]))
        g1 = s.loc[s["label"] == 1, feat]
        g0 = s.loc[s["label"] == 0, feat]
        corr_rows.append({
            "feature": feat, "n": int(len(s)), "corr": round(corr, 4),
            "mean_label1": round(float(g1.mean()), 5), "mean_label0": round(float(g0.mean()), 5),
            "std_label1": round(float(g1.std()), 5), "std_label0": round(float(g0.std()), 5),
        })
    corr_rows.sort(key=lambda r: (r["corr"] is None, -abs(r["corr"]) if r["corr"] is not None else 0))

    # --- 2) HistGradientBoostingClassifier fit TRAIN -> permutation importance on VAL ---
    X_train, y_train = train[TIER0_FEATURES], train["label"].astype(int)
    X_val, y_val = val[TIER0_FEATURES], val["label"].astype(int)
    clf = HistGradientBoostingClassifier(random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    train_acc = float(clf.score(X_train, y_train))
    val_acc = float(clf.score(X_val, y_val))
    from sklearn.metrics import roc_auc_score
    val_auc = float(roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]))

    perm = permutation_importance(clf, X_val, y_val, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    perm_rows = sorted(
        [{"feature": f, "importance_mean": round(float(m), 5), "importance_std": round(float(s), 5)}
         for f, m, s in zip(TIER0_FEATURES, perm.importances_mean, perm.importances_std)],
        key=lambda r: -r["importance_mean"],
    )

    return {
        "train_n": int(len(train)), "val_n": int(len(val)),
        "train_label1_rate": round(float(y_train.mean()), 4),
        "val_label1_rate": round(float(y_val.mean()), 4),
        "gbm_train_accuracy": round(train_acc, 4),
        "gbm_val_accuracy": round(val_acc, 4),
        "gbm_val_auc": round(val_auc, 4),
        "correlation_train_pooled": corr_rows,
        "permutation_importance_val": perm_rows,
    }


def main() -> int:
    print("Loading BTC Tier0 candidate CSV (truncated to timestamp < VAL_END, OOS/HOLDOUT never loaded)...")
    df = load_data()
    print(f"  {len(df)} rows, {df['timestamp'].min()} .. {df['timestamp'].max()}")

    fields = compute_outcome_fields(df)
    cand = build_candidate_pool(df, fields, atr_mult=ATR_MULT)

    print("Running self-check against reference realized_outcome() implementation...")
    ref = load_ref_impl()
    assert ref.FAST_BARS == FAST_BARS and ref.FULL_BARS == FULL_BARS
    assert ref.ATR_MULT == ATR_MULT and ref.CHOP_MULT == CHOP_MULT and ref.T_SUSTAIN == T_SUSTAIN
    sc = self_check(df, cand, ref, n_sample=500)
    print(f"  self-check: {sc['n_checked']} checked, {sc['n_mismatches']} mismatches")
    if sc["n_mismatches"]:
        print("  MISMATCH SAMPLE:", json.dumps(sc["mismatches_sample"][:3], default=str, ensure_ascii=False, indent=2))

    print("Summarizing TRAIN/VAL label-rate & excluded-rate...")
    counts = {split: summarize_split(cand, split) for split in ("TRAIN", "VAL")}

    print("ATR_MULT sensitivity check (1.25 / 1.5 / 1.75)...")
    sensitivity = atr_mult_sensitivity(df, fields)

    print("Trigger quality breakdown (TRAIN primary, VAL cross-check)...")
    trig_train = trigger_quality(cand, "TRAIN")
    trig_val = trigger_quality(cand, "VAL")

    print("Feature analysis (TRAIN correlation + GBM->VAL permutation importance)...")
    feat = feature_analysis(cand)

    report = {
        "scope": {
            "grid_screening_and_feature_analysis_only": True,
            "tabpfn_training_done": False,
            "economic_cost_gate_done": False,
            "holdout_touched": False,
            "holdout_definition": "timestamp >= 2026-04-01",
            "oos_touched": False,
            "oos_definition": "2026-01-01 <= timestamp < 2026-04-01",
            "raw_frame_truncated_at": str(VAL_END),
            "candidate_pool_triggers": TRIGGERS,
            "n_triggers_this_run": len(TRIGGERS),
            "live_eth_signal_n_triggers": 9,
            "triggers_excluded_vs_live_eth": ["smt_divergence", "demarker_extreme", "kalman_deviation_meanrev"],
        },
        "formula": {
            "fast_bars": FAST_BARS, "full_bars": FULL_BARS,
            "atr_mult_primary": ATR_MULT, "chop_mult": CHOP_MULT, "t_sustain": T_SUSTAIN,
            "source": "research_eth_v_rebound_sweep_gate_recall_check_90d_20260831.py::realized_outcome (verbatim)",
        },
        "self_check_vs_reference_impl": sc,
        "label_excluded_rate_by_split": counts,
        "atr_mult_sensitivity": sensitivity,
        "trigger_quality_v_rebound_rate": {"TRAIN": trig_train, "VAL": trig_val},
        "feature_analysis": feat,
        "eth_multitrigger_benchmark_for_comparison": {
            "source": "docs/homer/README.md lines ~412-420 (9-trigger ETH multitrigger V자반등 build, "
                       "scripts/build_eth_5m_v_rebound_multitrigger_labels_20260831.py, full history "
                       "2023-12-31..2026-08-28, same v7b outcome formula)",
            "note": "This is the apples-to-apples multi-trigger benchmark. It differs from the "
                    "single-trigger (liquidity_sweep-only) v7b figures reported elsewhere in this "
                    "project's history (e.g. 43.9%/56.1% split), which used a narrower candidate pool.",
            "n_candidates_total": 66395,
            "v_rebound_count": 9315, "v_rebound_pct_of_total": 0.140,
            "chop_support_count": 18870, "chop_support_pct_of_total": 0.284,
            "ambiguous_excluded_count": 38210, "ambiguous_excluded_pct_of_total": 0.575,
            "v_rebound_rate_of_labeled": round(9315 / (9315 + 18870), 4),
            "per_trigger_v_rebound_rate_of_total_fired": {
                "liquidity_sweep": 0.157, "taker_delta_z_climax": 0.132, "short_term_return_z": 0.136,
                "orthogonal_combo": 0.122, "smt_divergence": 0.133, "fib_extension_exhaustion": 0.199,
                "demarker_extreme": 0.129, "kalman_deviation_meanrev": 0.095, "local_extreme": 0.222,
            },
        },
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")
    print(json.dumps({"counts": counts, "self_check": sc}, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
