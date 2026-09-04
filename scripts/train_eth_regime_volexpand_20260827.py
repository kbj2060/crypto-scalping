#!/usr/bin/env python3
"""Train a 2-class (expanding vs not) volatility-trend regime classifier, 2026-08-27, per the plan
at /home/kbj20/.claude/plans/pure-hugging-book.md (approved same day).

Motivation: GBM2's efficiency-ratio label doesn't fire on retracement-heavy "grinding" declines with
real net displacement (concrete case: ETH 2528->2417 over 17h, 2026-08-25/26, mostly classified
chop). Hypothesis: a magnitude-based volatility signal isn't reset by contrary bars the way
efficiency ratio is, so it can stabilize (need less debounce) and confirm earlier than GBM2 WITHOUT
any forward-looking forecasting -- a same-time state classifier, exactly like GBM2 itself, not a
"predict the future" model. See scripts/research_eth_volexpand_regime_label_design_20260827.py
(Phase 0+1) for the diagnostic chart and label-design grid that motivated the choices locked below:
cutoff=top 20% of realized_vol_ratio, K_BARS_LABEL=12 (matches GBM2's own K for direct comparability
-- Phase 1 showed this reaches confirmed flip_rate=0.0129, essentially tying GBM2's own 0.0122 at
the SAME K, while the raw undebounced label is already ~4-5x more stable than GBM2's raw OR GBM3's
live (no-debounce) flip_rate, ~0.20 for both).

Label: realized_vol_ratio (features/engineering.py:296-298, rv_short(12bar)/rv_long(288bar),
already computed, no new formula) >= a TRAIN-fit (2024-01-01~2026-06-30, matches GBM2's TRAIN_FULL)
top-20th-percentile threshold -> is_expand_raw -> debounce(K=12) -> is_expand_confirmed (the
training target). NOT a slope/derivative of the ratio -- a level comparison (is short-term realized
vol currently elevated vs its own 288-bar baseline), stated explicitly to avoid the name misleading
a future reader.

Features: GBM3's 136 feature_cols minus a ~14-column near-tautological volatility-proxy exclusion
list (see EXCLUDED_VOL_PROXY_FEATURES below) -- NOT GBM2's own 5-column exclusion list, which is
circular w.r.t. a *directional/efficiency* label, not this *magnitude* label (state7_volatility_state
is the only overlap). compression_release_up/down are ablated (included vs excluded) as a separate
arm since they're conceptually close to "volatility just started expanding" without being formula-
identical to any excluded column.

Evaluation adds, beyond standard classification metrics (mirroring GBM2's _eval/HP-selection/
hysteresis-grid structure verbatim): an early-warning event study (reusing event_study() from
analyze_eth_confluence_oscillator_bottom_top_evidence_20260814.py, not reinventing lead-time logic)
against GBM2's own chop->trend CONFIRMED transitions (0->1 rising edges only, VAL+OOS range only),
comparing the TRAINED MODEL's predicted-state rising edges vs a TRIVIAL BASELINE (the raw
is_expand_confirmed label itself, i.e. what you'd get just watching realized_vol_ratio directly with
no model) -- if the trivial baseline matches the model, that's honest evidence the other ~120
features aren't earning their keep. Also stratifies pivots by RegimeEngine's own er_24 at
confirmation (clean vs grinding transitions, features/elite.py:509-514 formula) since the retracement-
heavy case is specifically what motivated this whole model, not clean momentum bursts GBM2 already
catches fine.
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
from train_eth_regime_gbm2_trend_chop_20260827 import _apply_hysteresis, _debounce  # noqa: E402
from analyze_eth_confluence_oscillator_bottom_top_evidence_20260814 import event_study  # noqa: E402

CLASSES2 = ["not_expanding", "expanding"]  # index 0/1 -- matches sklearn ascending int-label order
CUTOFF_TOP_PCT = 0.20   # locked from Phase 1 grid (research_eth_volexpand_regime_label_design_20260827.py)
K_BARS_LABEL = 12       # locked from Phase 1 grid -- matches GBM2's own K_BARS_LABEL for comparability
LEAD_WINDOW_K_GRID = [12, 24, 48, 96]  # event-study lead window (bars) -- DISTINCT from K_BARS_LABEL
GBM2_K_BARS_LABEL = 12  # GBM2's own label debounce, reproduced here (not imported: GBM2 hardcodes it as a module constant, not exported as a named value the training pipeline expects to reuse as an object -- reproduced verbatim, matches scripts/train_eth_regime_gbm2_trend_chop_20260827.py:69)

MODEL_ID = "eth_regime_volexpand_20260827"
MODEL_OUT_DIR = ROOT / f"tmp/{MODEL_ID}"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}_report.json"
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
TRAIN_CSVS = [
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
    ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
]

TRAIN_START = pd.Timestamp("2024-01-01T00:00:00")
SEL_END = pd.Timestamp("2026-03-31T23:55:00")
VAL_START = pd.Timestamp("2026-04-01T00:00:00")
TRAIN_END = pd.Timestamp("2026-06-30T23:55:00")
OOS_START = pd.Timestamp("2026-07-01T00:00:00")
OOS_END = pd.Timestamp("2026-08-19T23:55:00")

HP_CANDIDATES = {
    "default_gbm3": dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0),
    "shallower": dict(max_depth=6, learning_rate=0.04, max_iter=400, l2_regularization=2.0),
    "more_iters": dict(max_depth=10, learning_rate=0.04, max_iter=600, l2_regularization=2.0),
}
HP_SELECTION_MARGIN = 0.005

# Near-tautological volatility-LEVEL proxies -- excluded so the model must find genuine precursor
# information rather than just re-deriving today's already-visible vol level through a side door.
# Confirmed present in GBM3's 136-col feature set by direct joblib inspection. state12_garman_klass_vol
# = tanh(garman_klass_vol/0.00002) (a monotonic transform of an already-excluded column) and
# state7_range_compression (35% = bb_width_z, already excluded) added on top of the raw ~11-item set
# a prior (abandoned) BTC 1h volregime line flagged (eval_btc_1h_volregime_predictability_ablation_
# 20260805.py:57-61) -- both would let the model recover excluded information through a side door.
EXCLUDED_VOL_PROXY_FEATURES = [
    "realized_vol_ratio",  # the label's own raw signal
    "volatility_z", "garman_klass_vol", "rogers_satchell_vol", "parkinson_vol",
    "bb_width", "bb_width_z", "bb_width_pct_rank_288", "atr_pct_rank_288",
    "compression_score", "garch_vol", "garch_vol_z",
    "state7_volatility_state", "state12_garman_klass_vol", "state7_range_compression",
]
# Ablated separately (include vs exclude arm) -- conceptually close to "vol just started expanding"
# but not formula-identical to any excluded column (single-bar impulse x lagged compression state,
# not a 12/288-bar rolling ratio) -- see features/engineering.py:728-732.
ABLATION_FEATURES = ["compression_release_up", "compression_release_down"]

# NOT GBM2's excluded_circular_features ([mtf_trend_1h, state7_trend_efficiency_48,
# state7_directional_return_48, state7_volatility_state, state7_sign_flip_rate_24]) -- those are
# circular w.r.t. a DIRECTIONAL/efficiency-ratio label. Only state7_volatility_state overlaps (already
# in EXCLUDED_VOL_PROXY_FEATURES above); the other four measure directional cleanness, not magnitude,
# and are plausibly legitimate features for THIS model -- left in, let permutation importance judge.


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
        "rows": int(len(y)), "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "recall": recall, "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "expand_share": float(pred.mean()),
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
        "median_state_duration_bars": float(np.median(runs)) if runs else 0.0,
    }


def _compute_er24(close: pd.Series) -> np.ndarray:
    """Kaufman efficiency ratio, 24-bar window -- verbatim formula from features/elite.py:509-514,
    recomputed here (RegimeEngine.compute() only returns the 5 one-hot COLS, not er_24/er_48
    themselves) purely for the clean-vs-grinding stratification below."""
    c = pd.to_numeric(close, errors="coerce").ffill()
    diff_abs = c.diff().abs()
    net_change_24 = c - c.shift(24)
    er_24 = net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)
    return er_24.fillna(0.0).to_numpy()


def load_data() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    return df


def build_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Both this model's vol-expansion label AND GBM2's own trend label (needed as event-study
    pivots) are built here, over the full continuous series before any split."""
    assert "realized_vol_ratio" in df.columns, "realized_vol_ratio missing from offline TRAIN csv"
    ratio = df["realized_vol_ratio"].to_numpy()
    train_full_mask = ((df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)).to_numpy()
    thresh = float(np.nanquantile(ratio[train_full_mask], 1.0 - CUTOFF_TOP_PCT))
    is_expand_raw = np.where(np.isfinite(ratio), (ratio >= thresh).astype(int), 0)
    df["is_expand_raw"] = is_expand_raw
    df["is_expand_confirmed"] = _debounce(is_expand_raw, K_BARS_LABEL)

    labeled = RegimeEngine().compute(df.copy())
    gbm2_is_trend_raw = ((labeled["regime_bull"] + labeled["regime_bear"]) > 0).astype(int).to_numpy()
    df["gbm2_is_trend_confirmed"] = _debounce(gbm2_is_trend_raw, GBM2_K_BARS_LABEL)
    df["er_24"] = _compute_er24(df["close"])
    return df, thresh


def _fit(x: pd.DataFrame, y: np.ndarray, hp: dict) -> HistGradientBoostingClassifier:
    model = HistGradientBoostingClassifier(random_state=7529, **hp)
    model.fit(x, y)
    return model


def rising_edges(codes: np.ndarray) -> np.ndarray:
    """Bar positions where codes transitions 0->1 (not every bar the state holds 1) -- using every
    held bar would place a trigger immediately adjacent to nearly every pivot and understate true
    lead time."""
    codes = np.asarray(codes)
    if len(codes) < 2:
        return np.array([], dtype=np.int64)
    return np.flatnonzero((codes[1:] == 1) & (codes[:-1] == 0)) + 1


def run_event_study(pivot_pos: np.ndarray, trigger_pos: np.ndarray, all_pos: np.ndarray) -> dict[str, dict]:
    return {str(K): event_study(trigger_pos, pivot_pos, all_pos, K) for K in LEAD_WINDOW_K_GRID}


def main() -> None:
    print("Loading data...")
    df = load_data()
    print(f"  rows: {len(df)}  range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    print("Adding state7/state12 columns...")
    df = _with_raw_state12(df)

    print(f"Building labels (realized_vol_ratio top-{CUTOFF_TOP_PCT:.0%} threshold, K={K_BARS_LABEL} debounce)...")
    df, vol_threshold = build_labels(df)
    y_raw_all = df["is_expand_raw"].to_numpy()
    y_confirmed_all = df["is_expand_confirmed"].to_numpy()
    print(f"  threshold={vol_threshold:.4f}")
    print(f"  is_expand_raw flip_rate={np.mean(y_raw_all[1:] != y_raw_all[:-1]):.4f}, "
          f"is_expand_confirmed flip_rate={np.mean(y_confirmed_all[1:] != y_confirmed_all[:-1]):.4f}")

    gbm3_payload = joblib.load(GBM3_MODEL_PATH)
    all_feature_cols = list(gbm3_payload["feature_cols"])
    missing = [c for c in all_feature_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"missing feature columns after _with_raw_state12: {missing}")
    feature_cols = [c for c in all_feature_cols
                    if c not in EXCLUDED_VOL_PROXY_FEATURES and c not in ABLATION_FEATURES]
    print(f"  feature_cols: {len(all_feature_cols)} -> {len(feature_cols)} after excluding "
          f"{len(EXCLUDED_VOL_PROXY_FEATURES)} near-tautological vol-proxy columns + "
          f"{len(ABLATION_FEATURES)} ablation columns (default-excluded, tested as a separate arm below)")

    x_all = df[feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    train_full_mask = ((df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= TRAIN_END)).to_numpy()
    medians = x_all[train_full_mask].median(numeric_only=True).fillna(0.0)
    x_all = x_all.fillna(medians).fillna(0.0)

    sel_mask = ((df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= SEL_END)).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] <= TRAIN_END)).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)).to_numpy()
    print(f"  TRAIN_SEL={sel_mask.sum()}  VAL={val_mask.sum()}  TRAIN_FULL={train_full_mask.sum()}  OOS={oos_mask.sum()}")

    print("HP sanity check (fit on TRAIN_SEL, eval bal_acc on VAL)...")
    hp_results = {}
    for name, hp in HP_CANDIDATES.items():
        m = _fit(x_all[sel_mask], y_confirmed_all[sel_mask], hp)
        pred = m.predict(x_all[val_mask])
        bal_acc = balanced_accuracy_score(y_confirmed_all[val_mask], pred)
        hp_results[name] = bal_acc
        print(f"  {name}: VAL bal_acc={bal_acc:.4f}  {hp}")
    best_name = max(hp_results, key=hp_results.get)
    if hp_results[best_name] - hp_results["default_gbm3"] > HP_SELECTION_MARGIN:
        chosen_hp_name, chosen_hp = best_name, HP_CANDIDATES[best_name]
    else:
        chosen_hp_name, chosen_hp = "default_gbm3", HP_CANDIDATES["default_gbm3"]
    print(f"  chosen: {chosen_hp_name} {chosen_hp}")

    print("Final fit on TRAIN_FULL...")
    final_model = _fit(x_all[train_full_mask], y_confirmed_all[train_full_mask], chosen_hp)
    assert (final_model.classes_ == np.arange(len(CLASSES2))).all()

    metrics_confirmed_target = {}
    for split_name, mask in (("val", val_mask), ("oos", oos_mask)):
        pred = final_model.predict(x_all[mask])
        metrics_confirmed_target[split_name] = _eval(y_confirmed_all[mask], pred)
        print(f"  [{split_name}] argmax vs confirmed-target: {metrics_confirmed_target[split_name]}")

    metrics_raw_target = {}
    for split_name, mask in (("val", val_mask), ("oos", oos_mask)):
        pred = final_model.predict(x_all[mask])
        metrics_raw_target[split_name] = _eval(y_raw_all[mask], pred)
        print(f"  [{split_name}] argmax vs raw (undebounced) label: {metrics_raw_target[split_name]}")

    # --- ablation arm: compression_release_up/down included ---
    feature_cols_ablation = feature_cols + [c for c in ABLATION_FEATURES if c not in feature_cols]
    x_ablation = df[feature_cols_ablation].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    med_ablation = x_ablation[train_full_mask].median(numeric_only=True).fillna(0.0)
    x_ablation = x_ablation.fillna(med_ablation).fillna(0.0)
    model_ablation = _fit(x_ablation[train_full_mask], y_confirmed_all[train_full_mask], chosen_hp)
    ablation_results = {}
    for split_name, mask in (("val", val_mask), ("oos", oos_mask)):
        pred = model_ablation.predict(x_ablation[mask])
        ablation_results[split_name] = balanced_accuracy_score(y_confirmed_all[mask], pred)
    print(f"  ablation (compression_release_up/down INCLUDED): VAL/OOS bal_acc = "
          f"{ablation_results['val']:.4f} / {ablation_results['oos']:.4f}  "
          f"vs primary (excluded) {metrics_confirmed_target['val']['balanced_accuracy']:.4f} / "
          f"{metrics_confirmed_target['oos']['balanced_accuracy']:.4f}")

    # --- serving-side hysteresis grid on VAL (secondary smoothing, mirrors GBM2's pattern) ---
    print("Serving-side hysteresis grid (VAL only)...")
    proba_val = final_model.predict_proba(x_all[val_mask])[:, 1]
    hysteresis_grid = {}
    for k_bars in (1, 3, 6):
        for band in (0.0, 0.05):
            key = f"k{k_bars}_b{band}"
            confirmed = _apply_hysteresis(proba_val, k_bars, band)
            hysteresis_grid[key] = _eval(y_confirmed_all[val_mask], confirmed)
    baseline_flip = hysteresis_grid["k1_b0.0"]["flip_rate"]
    improved = {k: m for k, m in hysteresis_grid.items() if k != "k1_b0.0" and m["flip_rate"] < baseline_flip}
    chosen_serving = max(improved, key=lambda k: improved[k]["balanced_accuracy"]) if improved else "k1_b0.0"
    serving_k_bars, serving_band = (int(chosen_serving.split("_")[0][1:]), float(chosen_serving.split("_b")[1]))
    print(f"  chosen serving-side hysteresis: {chosen_serving}")

    # --- permutation importance ---
    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(final_model, x_all[oos_mask], y_confirmed_all[oos_mask],
                                       n_repeats=5, random_state=7529, n_jobs=-1)
        top15 = sorted(zip(feature_cols, perm.importances_mean), key=lambda t: -t[1])[:15]
    except Exception as e:  # noqa: BLE001 -- diagnostic only
        top15 = [("permutation_importance_failed", str(e))]
    print("Top-15 permutation importances (OOS):", top15)

    # =============================================================================================
    # Early-warning event study: model vs trivial baseline, against GBM2 chop->trend pivots
    # =============================================================================================
    print("\n=== Early-warning event study (VAL+OOS only) ===")
    eval_mask = val_mask | oos_mask
    eval_idx = np.flatnonzero(eval_mask)
    gbm2_confirmed_eval = df["gbm2_is_trend_confirmed"].to_numpy()[eval_mask]
    er24_eval = df["er_24"].to_numpy()[eval_mask]

    pivot_pos_local = rising_edges(gbm2_confirmed_eval)  # chop->trend starts only, local (0-based within eval slice) indices
    all_pos_local = np.arange(eval_mask.sum())
    print(f"  GBM2 chop->trend pivots in VAL+OOS: {len(pivot_pos_local)}")

    model_pred_confirmed = np.concatenate([final_model.predict(x_all[val_mask]), final_model.predict(x_all[oos_mask])])
    trivial_confirmed = y_confirmed_all[eval_mask]  # the label itself, i.e. thresholding realized_vol_ratio directly, no model
    model_trigger_local = rising_edges(model_pred_confirmed)
    trivial_trigger_local = rising_edges(trivial_confirmed)
    print(f"  model expand-rising-edges: {len(model_trigger_local)}, trivial-baseline expand-rising-edges: {len(trivial_trigger_local)}")

    event_study_model = run_event_study(pivot_pos_local, model_trigger_local, all_pos_local)
    event_study_trivial = run_event_study(pivot_pos_local, trivial_trigger_local, all_pos_local)
    for K in LEAD_WINDOW_K_GRID:
        m, t = event_study_model[str(K)], event_study_trivial[str(K)]
        print(f"  K={K:>3}: model recall={m['recall']:.3f} precision={m['precision']:.3f} lift={m['lift']:.2f} "
              f"lead={m['median_lead_bars']}  |  trivial recall={t['recall']:.3f} precision={t['precision']:.3f} "
              f"lift={t['lift']:.2f} lead={t['median_lead_bars']}")

    # ER-stratified: clean (er_24>=0.20, RegimeEngine's own trend threshold) vs grinding (<0.20) pivots
    pivot_er24 = er24_eval[pivot_pos_local]
    clean_pivots = pivot_pos_local[pivot_er24 >= 0.20]
    grinding_pivots = pivot_pos_local[pivot_er24 < 0.20]
    print(f"\n  pivot ER stratification: clean(er_24>=0.20)={len(clean_pivots)}, grinding(<0.20)={len(grinding_pivots)}")
    event_study_model_clean = run_event_study(clean_pivots, model_trigger_local, all_pos_local)
    event_study_model_grinding = run_event_study(grinding_pivots, model_trigger_local, all_pos_local)
    event_study_trivial_clean = run_event_study(clean_pivots, trivial_trigger_local, all_pos_local)
    event_study_trivial_grinding = run_event_study(grinding_pivots, trivial_trigger_local, all_pos_local)
    for K in LEAD_WINDOW_K_GRID:
        mc, mg = event_study_model_clean[str(K)], event_study_model_grinding[str(K)]
        tc, tg = event_study_trivial_clean[str(K)], event_study_trivial_grinding[str(K)]
        print(f"  K={K:>3}: model  clean lift={mc['lift']:.2f}  grinding lift={mg['lift']:.2f}  |  "
              f"trivial clean lift={tc['lift']:.2f}  grinding lift={tg['lift']:.2f}")

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
        "label_logic": (
            f"is_expand_raw = realized_vol_ratio(rv_short12/rv_long288) >= TRAIN-fit top-"
            f"{CUTOFF_TOP_PCT:.0%} threshold ({vol_threshold:.4f}); is_expand_confirmed = "
            f"debounce(is_expand_raw, k_bars={K_BARS_LABEL}). A same-time LEVEL state (is current "
            "realized vol elevated vs its own 288-bar baseline), NOT a slope/derivative of the ratio."
        ),
        "label_debounce_k_bars": K_BARS_LABEL,
        "vol_ratio_threshold": vol_threshold,
        "excluded_vol_proxy_features": EXCLUDED_VOL_PROXY_FEATURES,
        "ablation_features_excluded_by_default": ABLATION_FEATURES,
        "ablation_result_bal_acc_val_oos_with_features_included": ablation_results,
        "base_lineage": str(GBM3_MODEL_PATH.relative_to(ROOT)),
        "training_script": "scripts/train_eth_regime_volexpand_20260827.py",
        "hp_selection": {"candidates_val_bal_acc": hp_results, "chosen": chosen_hp_name},
        "hysteresis_config": {"k_bars": serving_k_bars, "band": serving_band},
        "hysteresis_grid_val": hysteresis_grid,
        "metrics": {"confirmed_target": metrics_confirmed_target, "raw_label_target": metrics_raw_target},
        "top15_permutation_importance_oos": top15,
        "early_warning_event_study": {
            "gbm2_pivot_source": "gbm2_is_trend_confirmed (RegimeEngine + K=12 debounce), rising edges only, VAL+OOS range",
            "lead_window_k_grid": LEAD_WINDOW_K_GRID,
            "n_pivots_total": int(len(pivot_pos_local)), "n_pivots_clean": int(len(clean_pivots)), "n_pivots_grinding": int(len(grinding_pivots)),
            "model": event_study_model, "trivial_baseline": event_study_trivial,
            "model_clean": event_study_model_clean, "model_grinding": event_study_model_grinding,
            "trivial_clean": event_study_trivial_clean, "trivial_grinding": event_study_trivial_grinding,
        },
        "notes": (
            "2-class (expanding vs not) HistGradientBoostingClassifier on a k_bars=12-debounced "
            "realized_vol_ratio threshold label. NOT deployed -- research/diagnostic only, see "
            "docs/model_reports/eth_regime_volexpand_20260827.md. dashboard/server.py and app.js "
            "are untouched by this script."
        ),
    }
    joblib.dump(payload, MODEL_OUT_DIR / "model.joblib")
    print(f"\nSaved model to {MODEL_OUT_DIR / 'model.joblib'}")
    report = {k: v for k, v in payload.items() if k != "model"}
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"Saved report to {REPORT_PATH}")


if __name__ == "__main__":
    main()
