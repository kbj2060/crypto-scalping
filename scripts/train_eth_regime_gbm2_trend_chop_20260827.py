#!/usr/bin/env python3
"""Train a 2-class (trend vs chop) regime GBM to replace GBM3 (bull/bear/chop) on the Snapshot
tab's liquidation-map chart ribbon, 2026-08-27.

Why 2-class instead of adding a 3rd "transition" class: this project already tried and explicitly
rejected treating whipsaw/instability as an independent discrete class twice --
docs/model_contracts/regime3_whipsaw_risk_policy_20260529.md ("do not use whipsaw as an independent
direction/state class for new action classifiers" -- OOS inspection showed frequent flips that
destabilize the classifier and conflict with chop) and docs/active_live/regime3_policy_20260530.md
(a continuous transition-risk score reached only 2026 OOS AUC=0.676/bal_acc=0.587, "not reliable
enough to own future class direction"). The 2026-08-26 GBM3 whipsaw-hierarchical research
(memory eth_regime_hierarchical_whipsaw_circularity_rejected_20260826) independently re-discovered
the same conclusion after 6 rounds. Merging bull+bear into a single "trend" class (this script) is
a genuinely new combination that was never tried in any of those three rounds, and sidesteps the
policy conflict entirely -- there is still no discrete instability/transition/whipsaw class here.

Label construction -- two steps, both causal/vectorized over the full history before any split:
  1. features.elite.RegimeEngine.compute() (unchanged, the same rule engine trading_bot.py's live
     owner-routing already consumes) -> is_trend_raw = regime_bull | regime_bear.
     chop = regime_chop | regime_whipsaw | regime_normal (verified to match GBM3's own merge by
     comparing GBM3's in-sample predicted class distribution against this raw split on TRAIN --
     see eth_regime_gbm2_trend_chop_20260827_report.json's "label_merge_verification" field).
  2. _debounce(is_trend_raw, k_bars=K_BARS_LABEL): a discrete K-consecutive-bar confirm. The raw
     label's own OOS flip_rate is 0.1877 -- visibly flickery when plotted against price (a user
     sanity-check chart showed genuine multi-hour trends captured cleanly, but short-lived
     ripples inside chop constantly re-triggering "trend"). The user asked for the *label itself*
     (not just a serving-side smoothing pass) to be more stable, so this debounced sequence -- not
     the raw per-bar label -- is what the model is actually trained to predict.

     K_BARS_LABEL=12 (1h) was chosen after comparing k in {6,12,24,48} directly on the ground-truth
     label plotted against real OOS price action: k=48 (4h) visibly locks up (the debounce counter
     resets on any single contrary bar, so at high k it tends to get stuck in whichever state
     accumulated 48 consecutive bars first, rather than genuinely stabilizing -- flip_rate=0.0001
     but trend_share collapses from ~0.45 to ~0.12, a red flag, not an improvement). k=24 (2h) was
     initially picked from the numeric grid alone (flip_rate=0.0023) but the user changed to k=12
     after seeing the actual side-by-side chart, preferring its finer responsiveness (still a 14.6x
     flip-rate reduction over raw: 0.1877 -> 0.0128) with no lock-up symptoms.

Everything else deliberately mirrors GBM3 for direct comparability: same 136 feature_cols (sourced
from the GBM3 joblib payload -- already excludes the 5 columns confirmed circular with the
RegimeEngine label formula), same HistGradientBoostingClassifier family, same TRAIN
(2024-01-01~2026-06-30) / internal causal VAL (2026-04~06, selection only) / OOS
(2026-07-01~08-19) split. NOTE: this exact OOS window has already been touched by ~8 prior rounds
of regime-classifier research in this repo (wide24 grid sweeps, whipsaw's 6 rounds, GBM3 itself) --
report it honestly, do not claim single-touch purity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from features.elite import RegimeEngine  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

CLASSES2 = ["chop", "trend"]  # index 0=chop, 1=trend -- matches sklearn's ascending int-label order
K_BARS_LABEL = 12  # 1h at 5min bars -- see module docstring for how this was chosen
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
MODEL_ID = "eth_regime_gbm2_trend_chop_20260827"
MODEL_OUT_DIR = ROOT / f"tmp/{MODEL_ID}"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}_report.json"
TRAIN_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]

TRAIN_START = pd.Timestamp("2024-01-01T00:00:00")
SEL_END = pd.Timestamp("2026-03-31T23:55:00")       # TRAIN minus VAL, for leak-free HP selection
VAL_START = pd.Timestamp("2026-04-01T00:00:00")
TRAIN_END = pd.Timestamp("2026-06-30T23:55:00")     # == VAL_END; full TRAIN (incl VAL) for final fit
OOS_START = pd.Timestamp("2026-07-01T00:00:00")
OOS_END = pd.Timestamp("2026-08-19T23:55:00")

HP_CANDIDATES = {
    "default_gbm3": dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0),
    "shallower": dict(max_depth=6, learning_rate=0.04, max_iter=400, l2_regularization=2.0),
    "more_iters": dict(max_depth=10, learning_rate=0.04, max_iter=600, l2_regularization=2.0),
}
HP_SELECTION_MARGIN = 0.005  # keep the GBM3-identical default unless another config beats it by >0.5pp VAL bal_acc


def _debounce(raw: np.ndarray, k_bars: int) -> np.ndarray:
    """Discrete K-bar confirm: the confirmed state only flips once `raw` has been constant at a new
    value for k_bars consecutive bars. k_bars=1 is a no-op (== raw). Verified against the
    scratchpad sanity-check script the user reviewed directly (regime_label_debounce_compare.py)."""
    n = len(raw)
    confirmed = np.empty(n, dtype=int)
    confirmed[0] = raw[0]
    candidate, streak = raw[0], 0
    for t in range(1, n):
        if raw[t] == confirmed[t - 1]:
            candidate, streak = confirmed[t - 1], 0
        else:
            streak = streak + 1 if raw[t] == candidate else 1
            candidate = raw[t]
        confirmed[t] = candidate if streak >= k_bars else confirmed[t - 1]
    return confirmed


def _apply_hysteresis(trend_prob: np.ndarray, k_bars: int, band: float = 0.0) -> np.ndarray:
    """Serving-side *secondary* smoothing on the model's own probability output -- the primary
    stabilization already happened at the label level (K_BARS_LABEL debounce above). raw[t]: an
    optional dead-zone around 0.5 (Schmitt trigger) that freezes on the previous raw value while
    trend_prob sits inside [0.5-band, 0.5+band]. confirmed[t]: raw must hold k_bars consecutive
    bars before the displayed state flips. k_bars=1, band=0.0 must reduce to plain argmax."""
    n = len(trend_prob)
    raw = np.empty(n, dtype=int)
    raw[0] = int(trend_prob[0] >= 0.5)
    for t in range(1, n):
        if trend_prob[t] >= 0.5 + band:
            raw[t] = 1
        elif trend_prob[t] <= 0.5 - band:
            raw[t] = 0
        else:
            raw[t] = raw[t - 1]
    return _debounce(raw, k_bars)


def _run_lengths(pred: np.ndarray) -> list[int]:
    if len(pred) == 0:
        return []
    lengths, start = [], 0
    for i in range(1, len(pred)):
        if pred[i] != pred[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(pred) - start)
    return lengths


def _eval(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1])
    runs = _run_lengths(pred)
    recall = {}
    for i, name in enumerate(CLASSES2):
        recall[name] = None if cm[i].sum() == 0 else float(cm[i, i] / cm[i].sum())
    return {
        "rows": int(len(y)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall": recall,
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "trend_share": float(pred.mean()),
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
        "median_state_duration_bars": float(np.median(runs)) if runs else 0.0,
    }


def load_data() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Computed once over the full continuous 2024-01-01~2026-08-19 series (both RegimeEngine's
    rolling windows and the debounce state machine need an unbroken sequence) -- callers slice the
    returned frame into TRAIN/VAL/OOS afterward, never re-run this per-split."""
    labeled = RegimeEngine().compute(df.copy())
    is_trend_raw = ((labeled["regime_bull"] + labeled["regime_bear"]) > 0).astype(int).to_numpy()
    df = df.copy()
    for col in RegimeEngine.COLS:
        df[col] = labeled[col]
    df["is_trend_raw"] = is_trend_raw
    df["is_trend_confirmed"] = _debounce(is_trend_raw, K_BARS_LABEL)
    return df


def _fit(x: pd.DataFrame, y: np.ndarray, hp: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=7529, **hp)
    model.fit(x, y)
    return model


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"  rows: {len(df)}  range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    print("Adding state7/state12 columns...")
    df = _with_raw_state12(df)

    print("Building labels (RegimeEngine + k_bars=%d debounce)..." % K_BARS_LABEL)
    df = build_labels(df)
    print("  is_trend_raw flip_rate (full range):",
          float(np.mean(df["is_trend_raw"].to_numpy()[1:] != df["is_trend_raw"].to_numpy()[:-1])))
    print("  is_trend_confirmed flip_rate (full range):",
          float(np.mean(df["is_trend_confirmed"].to_numpy()[1:] != df["is_trend_confirmed"].to_numpy()[:-1])))

    gbm3_payload = joblib.load(GBM3_MODEL_PATH)
    feature_cols = list(gbm3_payload["feature_cols"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature columns after _with_raw_state12: {missing}")

    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_full_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)
    medians = x_all[train_full_mask].median(numeric_only=True).fillna(0.0)
    x_all = x_all.fillna(medians).fillna(0.0)

    y_confirmed = df["is_trend_confirmed"].to_numpy()
    y_raw = df["is_trend_raw"].to_numpy()

    sel_mask = (df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= SEL_END)
    val_mask = (df["timestamp"] >= VAL_START) & (df["timestamp"] <= TRAIN_END)
    oos_mask = (df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)
    print(f"  TRAIN_SEL={sel_mask.sum()}  VAL={val_mask.sum()}  TRAIN_FULL={train_full_mask.sum()}  OOS={oos_mask.sum()}")

    # --- cheap HP sanity check on VAL only (fit on TRAIN_SEL, never touches VAL) ---
    print("HP sanity check (fit on TRAIN_SEL, eval bal_acc on VAL)...")
    hp_results = {}
    for name, hp in HP_CANDIDATES.items():
        m = _fit(x_all[sel_mask], y_confirmed[sel_mask], hp)
        pred = m.predict(x_all[val_mask])
        bal_acc = balanced_accuracy_score(y_confirmed[val_mask], pred)
        hp_results[name] = bal_acc
        print(f"  {name}: VAL bal_acc={bal_acc:.4f}  {hp}")
    best_name = max(hp_results, key=hp_results.get)
    if hp_results[best_name] - hp_results["default_gbm3"] > HP_SELECTION_MARGIN:
        chosen_hp_name, chosen_hp = best_name, HP_CANDIDATES[best_name]
    else:
        chosen_hp_name, chosen_hp = "default_gbm3", HP_CANDIDATES["default_gbm3"]
    print(f"  chosen: {chosen_hp_name} {chosen_hp}")

    # --- final fit on full TRAIN (incl VAL), matching GBM3's own convention ---
    print("Final fit on TRAIN_FULL (2024-01-01~2026-06-30, incl VAL)...")
    final_model = _fit(x_all[train_full_mask], y_confirmed[train_full_mask], chosen_hp)
    assert (final_model.classes_ == np.arange(len(CLASSES2))).all(), \
        f"unexpected classes_ order: {final_model.classes_} (must be int-coded [0,1], not sorted strings)"

    # --- evaluate raw argmax on VAL/OOS against the CONFIRMED target (the actual training target) ---
    metrics_confirmed_target = {}
    for split_name, mask in (("val", val_mask), ("oos", oos_mask)):
        pred = final_model.predict(x_all[mask])
        metrics_confirmed_target[split_name] = _eval(y_confirmed[mask], pred)
        print(f"  [{split_name}] argmax vs confirmed-target: {metrics_confirmed_target[split_name]}")

    # --- also report against the raw (undebounced) label, for transparency on what was traded away ---
    metrics_raw_target = {}
    for split_name, mask in (("val", val_mask), ("oos", oos_mask)):
        pred = final_model.predict(x_all[mask])
        metrics_raw_target[split_name] = _eval(y_raw[mask], pred)
        print(f"  [{split_name}] argmax vs raw (undebounced) label: {metrics_raw_target[split_name]}")

    # --- light serving-side hysteresis grid on VAL (secondary smoothing on top of an already-smooth model) ---
    print("Serving-side hysteresis grid (VAL only, secondary smoothing)...")
    proba_val = final_model.predict_proba(x_all[val_mask])[:, 1]
    hysteresis_grid = {}
    for k_bars in (1, 3, 6):
        for band in (0.0, 0.05):
            key = f"k{k_bars}_b{band}"
            confirmed = _apply_hysteresis(proba_val, k_bars, band)
            hysteresis_grid[key] = _eval(y_confirmed[val_mask], confirmed)
            print(f"  {key}: {hysteresis_grid[key]}")
    # sanity: k_bars=1, band=0.0 must equal plain argmax bit-for-bit
    plain_argmax = (proba_val >= 0.5).astype(int)
    hysteresis_identity_check = bool(np.array_equal(_apply_hysteresis(proba_val, 1, 0.0), plain_argmax))
    print(f"  hysteresis(k=1,band=0) == plain argmax: {hysteresis_identity_check}")

    # Among configs that also improve (lower) flip_rate vs the k1_b0.0 baseline, pick the one with the
    # best bal_acc -- not just the first one to clear a threshold (an earlier version of this script
    # stopped at the first "good enough" match and missed a strictly-dominant option later in the grid).
    baseline_flip = hysteresis_grid["k1_b0.0"]["flip_rate"]
    improved = {k: m for k, m in hysteresis_grid.items() if k != "k1_b0.0" and m["flip_rate"] < baseline_flip}
    chosen_serving = max(improved, key=lambda k: improved[k]["balanced_accuracy"]) if improved else "k1_b0.0"
    serving_k_bars, serving_band = (int(chosen_serving.split("_")[0][1:]), float(chosen_serving.split("_b")[1]))
    print(f"  chosen serving-side hysteresis: {chosen_serving} (k_bars={serving_k_bars}, band={serving_band})")

    # Confirmatory single application on OOS with the VAL-chosen config (grid search itself stays VAL-only).
    proba_oos = final_model.predict_proba(x_all[oos_mask])[:, 1]
    oos_confirmed_hysteresis = _eval(y_confirmed[oos_mask], _apply_hysteresis(proba_oos, serving_k_bars, serving_band))
    print(f"  [oos] confirmatory, hysteresis={chosen_serving} applied: {oos_confirmed_hysteresis}")

    # --- label-merge verification (cross-check against GBM3's in-sample predicted distribution) ---
    gbm3_model = gbm3_payload["model"]
    gbm3_x = x_all[train_full_mask][gbm3_payload["feature_cols"]]
    gbm3_pred = gbm3_model.predict(gbm3_x)
    gbm3_dist = {gbm3_payload["classes"][i]: float((gbm3_pred == i).mean()) for i in range(3)}
    regime_raw_dist = {
        "bull": float(df.loc[train_full_mask, "regime_bull"].mean()),
        "bear": float(df.loc[train_full_mask, "regime_bear"].mean()),
        "chop_merged": float((df.loc[train_full_mask, ["regime_chop", "regime_whipsaw", "regime_normal"]].sum(axis=1) > 0).mean()),
    }
    print("Label-merge cross-check -- GBM3 in-sample predicted dist:", gbm3_dist,
          " vs RegimeEngine raw dist (whipsaw+normal->chop):", regime_raw_dist)

    # --- feature importances (top 15, sanity check against known circular features) ---
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(final_model, x_all[oos_mask], y_confirmed[oos_mask],
                                       n_repeats=5, random_state=7529, n_jobs=-1)
        top15 = sorted(zip(feature_cols, perm.importances_mean), key=lambda t: -t[1])[:15]
    except Exception as e:  # noqa: BLE001 -- diagnostic only, never block training on this
        top15 = [("permutation_importance_failed", str(e))]
    print("Top-15 permutation importances (OOS):", top15)

    # --- save artifact ---
    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "classes": CLASSES2,
        "feature_cols": feature_cols,
        "feature_medians": medians.to_dict(),
        "model": final_model,
        "config": chosen_hp,
        "config_name": chosen_hp_name,
        "train_range": f"{TRAIN_START.isoformat()} ~ {TRAIN_END.isoformat()}",
        "val_range": f"{VAL_START.isoformat()} ~ {TRAIN_END.isoformat()} (internal causal, subset of TRAIN, selection only)",
        "oos_range": f"{OOS_START.isoformat()} ~ {OOS_END.isoformat()}",
        "oos_reuse_caveat": (
            "This OOS window has been reused across ~8 prior regime-classifier rounds in this repo "
            "(wide24 grid sweeps, whipsaw's 6 rounds, GBM3 itself) -- reported honestly, not claimed "
            "single-touch-pure."
        ),
        "label_logic": (
            "is_trend_raw = features.elite.RegimeEngine.compute()['regime_bull'|'regime_bear']; "
            "chop = regime_chop|regime_whipsaw|regime_normal (matches GBM3's own merge, verified by "
            "distribution cross-check -- see label_merge_verification). Training target "
            f"is_trend_confirmed = debounce(is_trend_raw, k_bars={K_BARS_LABEL}) -- NOT the raw label; "
            "the user asked for the label itself to be stable, not just a serving-side smoothing pass."
        ),
        "label_debounce_k_bars": K_BARS_LABEL,
        "label_merge_verification": {"gbm3_in_sample_predicted_dist": gbm3_dist, "regime_engine_raw_dist": regime_raw_dist},
        "excluded_circular_features": [
            "mtf_trend_1h", "state7_trend_efficiency_48", "state7_directional_return_48",
            "state7_volatility_state", "state7_sign_flip_rate_24",
        ],
        "base_lineage": str(GBM3_MODEL_PATH.relative_to(ROOT)),
        "training_script": "scripts/train_eth_regime_gbm2_trend_chop_20260827.py",
        "hp_selection": {"candidates_val_bal_acc": hp_results, "chosen": chosen_hp_name},
        "hysteresis_config": {
            "k_bars": serving_k_bars, "band": serving_band,
            "selected_on": "VAL 2026-04~06 light grid (k_bars in {1,3,6} x band in {0.0,0.05}), "
                           "secondary smoothing only -- primary stability comes from the label debounce",
        },
        "hysteresis_grid_val": hysteresis_grid,
        "hysteresis_identity_check_passed": hysteresis_identity_check,
        "hysteresis_oos_confirmatory": oos_confirmed_hysteresis,
        "metrics": {"confirmed_target": metrics_confirmed_target, "raw_label_target": metrics_raw_target},
        "top15_permutation_importance_oos": top15,
        "notes": (
            f"2-class (trend=bull|bear merged, chop=chop|whipsaw|normal) HistGradientBoostingClassifier, "
            f"trained on a k_bars={K_BARS_LABEL}-debounced RegimeEngine label (not the raw per-bar label) "
            "specifically to give a low-flip signal for the Snapshot-tab regime ribbon. Replaces GBM3 "
            "(bull/bear/chop) as the dashboard's regime source; does not touch trading_bot.py's live "
            "RegimeEngine-based owner routing."
        ),
    }
    joblib.dump(payload, MODEL_OUT_DIR / "model.joblib")
    print(f"Saved model to {MODEL_OUT_DIR / 'model.joblib'}")

    report = {k: v for k, v in payload.items() if k != "model"}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
