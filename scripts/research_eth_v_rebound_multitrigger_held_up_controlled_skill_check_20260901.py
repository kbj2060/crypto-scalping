#!/usr/bin/env python3
"""Does the ALREADY-DEPLOYED 9-trigger V자반등 TabPFN model have genuine skill beyond the
"held_up" tautology found in research_eth_v_rebound_local_extreme_circularity_check_20260901.py?

That check proved: any candidate (any trigger, or none) satisfying held_up[i] (price never made a
new low/high in the SAME W=6-bar window fast_move is computed over) has a ~4-5x higher V자반등
rate than one that doesn't -- a near-tautological consequence of computing fast_move from a FIXED
reference point low[i]/high[i], not new information. This does NOT by itself mean the deployed
model is "broken" -- held_up is not a Tier0 FEATURE the model ever sees directly. The open question
this script answers: does the model's proba carry genuine discriminative skill WITHIN each held_up
stratum (real signal), or is its overall AUC mostly just an implicit proxy for predicting held_up
itself (in which case its live trading value is closer to "avoids picking already-doomed-to-be-
undercut candidates" than "predicts genuine V-shaped reversals")?

Method (VAL+OOS only -- does NOT touch HOLDOUT, which is already spent twice today for this model
family and must not be re-touched for a diagnostic):
  1. Reproduce the DEPLOYED model's own methodology exactly (embargoed split, FEATURE_COLUMNS,
     4-seed average) on the ORIGINAL 9-trigger pool (data/labels/eth_5m_v_rebound_multitrigger_
     20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv) -- the pool actually serving
     live_eth_sweep_v_rebound_signal_20260829.py today.
  2. Recompute held_up[i] for every VAL+OOS candidate directly from raw klines via each row's own
     `idx` + `direction` column (same W=6 forward-window formula, self-checked already).
  3. Report: (a) overall AUC (sanity vs the known ~0.829/~0.813 4-seed figures), (b) AUC of proba
     AGAINST held_up itself (is proba secretly a held_up detector?), (c) AUC of proba against the
     TRUE label COMPUTED SEPARATELY within held_up=True and held_up=False strata -- the decisive
     test. If within-stratum AUCs collapse toward 0.5 while the pooled AUC stays high, the model's
     apparent skill is mostly "sorting candidates by held_up-likelihood," not genuine reversal
     prediction. If within-stratum AUCs stay well above 0.5, the model has real skill on top of
     that tautology.

Must run in the quant_ai conda env on the SERVER (GPU required):
  ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_eth_v_rebound_multitrigger_held_up_controlled_skill_check_20260901.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/research/eth_v_rebound_multitrigger_held_up_controlled_skill_20260901"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # NEVER loaded/touched by this script
LABEL_WINDOW = pd.Timedelta(minutes=60)
SEEDS = [20260829, 141592, 271828, 577215]
W = 6  # == FAST_BARS, the held_up overlap window

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def log(msg: str) -> None:
    print(f"[held_up_controlled_skill] {msg}", flush=True)


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
    }


def compute_held_up(df: pd.DataFrame, kl_low: np.ndarray, kl_high: np.ndarray, n_kl: int) -> np.ndarray:
    """held_up[i] per this candidate's own direction: bottom(downside)->low never undercut in next W
    bars; top(upside)->high never exceeded in next W bars. Uses the candidate's `idx` into the FULL
    klines frame (same convention as build_eth_5m_v_rebound_multitrigger_features_tier0_20260831.py
    -- idx is a positional index into the raw ETH klines, unaffected by this features CSV's own
    row order)."""
    out = np.full(len(df), False)
    idx_arr = df["idx"].to_numpy()
    is_down = (df["direction"] == "downside").to_numpy()
    for row_i, (idx, down) in enumerate(zip(idx_arr, is_down)):
        idx = int(idx)
        a, b = idx + 1, idx + W
        if b >= n_kl:
            continue
        if down:
            out[row_i] = kl_low[a:b + 1].min() >= kl_low[idx]
        else:
            out[row_i] = kl_high[a:b + 1].max() <= kl_high[idx]
    return out


def stratified_auc(proba: np.ndarray, y: np.ndarray, held_up: np.ndarray) -> dict:
    result = {"overall": {"n": int(len(y)), "auc": round(float(roc_auc_score(y, proba)), 4)}}
    result["proba_vs_held_up_itself"] = {
        "n": int(len(y)), "auc": round(float(roc_auc_score(held_up.astype(int), proba)), 4),
    }
    for name, mask in (("held_up_true", held_up), ("held_up_false", ~held_up)):
        yy, pp = y[mask], proba[mask]
        if len(yy) < 30 or yy.min() == yy.max():
            result[name] = {"n": int(len(yy)), "auc": None, "label_rate": round(float(yy.mean()), 4) if len(yy) else None}
            continue
        result[name] = {"n": int(len(yy)), "auc": round(float(roc_auc_score(yy, pp)), 4),
                         "label_rate": round(float(yy.mean()), 4)}
    return result


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        log(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")
    over_limit = len(parts["train"]) > 10000

    log("loading raw klines for held_up recomputation...")
    kl = pd.read_csv(KLINES, usecols=["low", "high"])
    kl_low, kl_high, n_kl = kl["low"].to_numpy(), kl["high"].to_numpy(), len(kl)

    log("4-seed fit on TRAIN, scoring VAL+OOS (HOLDOUT NOT loaded/touched)...")
    val_probas, oos_probas = [], []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        val_probas.append(clf.predict_proba(parts["val"][FEATURE_COLUMNS])[:, 1])
        oos_probas.append(clf.predict_proba(parts["oos"][FEATURE_COLUMNS])[:, 1])
        log(f"  seed={seed} done")

    report = {}
    for name, part, probas in (("val", parts["val"], val_probas), ("oos", parts["oos"], oos_probas)):
        proba = np.mean(probas, axis=0)
        y = part["label"].to_numpy()
        held_up = compute_held_up(part, kl_low, kl_high, n_kl)
        held_up_incidence = round(float(held_up.mean()), 4)
        r = stratified_auc(proba, y, held_up)
        r["held_up_incidence"] = held_up_incidence
        report[name] = r
        log(f"[{name}] overall AUC={r['overall']['auc']}  proba_vs_held_up_itself AUC={r['proba_vs_held_up_itself']['auc']}  "
            f"held_up incidence={held_up_incidence:.1%}")
        log(f"[{name}] WITHIN held_up=True  (n={r['held_up_true']['n']}, base_rate={r['held_up_true'].get('label_rate')}): AUC={r['held_up_true']['auc']}")
        log(f"[{name}] WITHIN held_up=False (n={r['held_up_false']['n']}, base_rate={r['held_up_false'].get('label_rate')}): AUC={r['held_up_false']['auc']}")

    combined_val_oos_note = (
        "If held_up=True/False within-stratum AUCs are both well above 0.5 and comparable to the "
        "overall AUC, the model has genuine skill beyond the held_up tautology. If they collapse "
        "toward 0.5 while overall AUC stays high (~0.8+), the model's apparent skill is "
        "substantially an implicit held_up detector, not a genuine V-shape-reversal predictor."
    )
    report["interpretation_guide"] = combined_val_oos_note
    report["holdout_touched"] = False
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"Wrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
