#!/usr/bin/env python3
"""Follow-up to research_eth_wide24_tailrisk_feature_augmentation_20260826.py (tail_risk features
REJECTED). This round tests the OTHER "newly discovered signal families" the user asked about --
liquidity sweep, funding/OI divergence, whale/retail flow, CVD, volume-profile POC, ICT fair-value
gap -- all of which turn out to already be columns in the canonical data/splits/year_oos/
training_features_2026_rebuilt.csv (142 cols total), so no fresh data collection needed and the
test can use the FULL 2026-01-01..2026-08-19 span (much longer than tail_risk's 3.8mo).

TRAIN=2026-01-01..2026-06-30, EVAL=2026-07-01..2026-08-19 -- EVAL window deliberately matches the
CONFIRMED wide24 baseline's own official OOS window (docs/model_contracts/ilias_eth_human_
direction_risk_management_contract_20260817.md), though TRAIN here is 2026-only (6 months) vs the
baseline's full 2024-01..2026-06-30 anchored TRAIN -- still not apples-to-apples with the
headline 0.7691, but a fair same-window baseline-vs-augmented comparison.

Same states=24/sticky=0.90/seed=7529/balancedish_adx16_slope03_bb006 label as before. Everything
else (helper functions) copied verbatim from the tail_risk script / experiment_regime3_current_
hmm_wide24_20260529.py, same rationale (mamba_ssm import chain avoided).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel  # noqa: E402
from scripts.retrain_clean_regime_hmm_raw_state12_20260517 import STATE12_COLS, _with_raw_state12  # noqa: E402

CLASSES3 = ["bull", "bear", "chop"]
LABEL_CFG = {"trend_adx_min": 16.0, "weak_adx_max": 12.0, "slope_min": 0.00003, "tight_bb_max": 0.006}
WIDE24_EXTRA_COLS = ["volatility_z", "rsi", "macd_hist", "bb_width_z", "hma_slope", "wick_ratio",
                      "mtf_trend_1h", "mtf_trend_4h", "breakout_strength", "mean_reversion_z",
                      "ofi_acceleration", "taker_acceleration"]
WIDE24_COLS = list(STATE12_COLS) + WIDE24_EXTRA_COLS

# New candidate columns, grouped by the signal family they represent (all already in the
# canonical 142-col file -- none overlap with WIDE24_COLS):
EVIDENCE_EXTRA_COLS = [
    "sweep_prev_high_reclaim", "sweep_prev_low_reclaim",   # liquidity sweep (strongest standalone evidence signal, 3.51x lift)
    "funding_oi_divergence", "funding_z_score",             # funding+OI combo (9th evidence signal, IC real / econ gate failed)
    "whale_conviction", "whale_retail_ratio",                # whale/retail flow family
    "cvd_12", "cvd_slope_48", "price_cvd_divergence",        # CVD orderflow family
    "cvp_poc_dist", "cvp_volume_imbalance",                  # volume profile POC/VAH-VAL family
    "fvg_dist",                                               # ICT fair-value-gap distance
]
FEATURE_SETS = {"wide24": WIDE24_COLS, "wide24_evidence": WIDE24_COLS + EVIDENCE_EXTRA_COLS}
STATES, N_ITER, SEED, STICKY = 24, 22, 7529, 0.90
TRAIN_START = pd.Timestamp("2026-01-01")
EVAL_START = pd.Timestamp("2026-07-01")


def _num(frame, col, default=0.0):
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _adx(high, low, close, period=14):
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=period, adjust=False).mean()
    up = high.diff()
    down = -low.diff()
    pdm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=high.index)
    ndm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=high.index)
    pdi = 100.0 * pdm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    ndi = 100.0 * ndm.ewm(span=period, adjust=False).mean() / (atr + 1e-12)
    dx = 100.0 * (pdi - ndi).abs() / (pdi + ndi + 1e-12)
    return dx.ewm(span=period, adjust=False).mean()


def _current_labels3_thresholded(frame, cfg):
    close = _num(frame, "close")
    high = _num(frame, "high")
    low = _num(frame, "low")
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema_slope = (ema21 - ema21.shift(5)) / (close * 5.0 + 1e-12)
    adx = _num(frame, "adx_14", np.nan)
    if adx.isna().all():
        adx = _adx(high, low, close)
    bb_width = _num(frame, "bb_width", np.nan)
    if bb_width.isna().all():
        sma20 = close.rolling(20, min_periods=5).mean()
        bb_width = 2.0 * close.rolling(20, min_periods=5).std() / (sma20 + 1e-12)
    labels = np.full(len(frame), 2, dtype=np.int64)
    trending = adx.fillna(0.0).to_numpy() >= float(cfg["trend_adx_min"])
    slope = ema_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    slope_min = float(cfg["slope_min"])
    labels[trending & (slope > slope_min)] = 0
    labels[trending & (slope < -slope_min)] = 1
    labels[(adx.fillna(0.0).to_numpy() < float(cfg["weak_adx_max"])) | (bb_width.fillna(0.0).to_numpy() < float(cfg["tight_bb_max"]))] = 2
    return labels


def _with_features(frame, cols):
    out = _with_raw_state12(frame.copy())
    for col in cols:
        if col not in out.columns:
            raise ValueError(f"missing feature column: {col}")
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _fit_obs(train, pred, cols):
    x_train_raw = train[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x_train_raw.median(numeric_only=True).fillna(0.0)
    x_train = x_train_raw.fillna(medians).fillna(0.0)
    x_pred = pred[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(medians).fillna(0.0)
    scaler = RobustScaler(quantile_range=(5.0, 95.0))
    return scaler.fit_transform(x_train), scaler.transform(x_pred), scaler, medians


def _state_class_matrix(state_prob, y, smoothing=0.02):
    mat = np.full((state_prob.shape[1], len(CLASSES3)), smoothing, dtype=np.float64)
    for cls in range(len(CLASSES3)):
        mat[:, cls] += state_prob[y == cls].sum(axis=0) / max(int((y == cls).sum()), 1)
    mat /= np.clip(mat.sum(axis=1, keepdims=True), 1e-300, None)
    return mat


def _class_proba(state_prob, state_class):
    proba = state_prob @ state_class
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-300, None)
    return proba


def _run_lengths(pred):
    if len(pred) == 0:
        return []
    lengths, start = [], 0
    for i in range(1, len(pred)):
        if pred[i] != pred[i - 1]:
            lengths.append(i - start)
            start = i
    lengths.append(len(pred) - start)
    return lengths


def _eval(y, proba):
    proba = np.asarray(proba, dtype=np.float64)
    proba /= np.clip(proba.sum(axis=1, keepdims=True), 1e-12, None)
    pred = np.argmax(proba, axis=1)
    cm = confusion_matrix(y, pred, labels=list(range(len(CLASSES3))))
    recalls = {}
    for i, name in enumerate(CLASSES3):
        denom = cm[i].sum()
        recalls[name] = None if denom == 0 else float(cm[i, i] / denom)
    runs = _run_lengths(pred)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, proba, labels=list(range(len(CLASSES3))))),
        "recall": recalls,
        "true_counts": {CLASSES3[i]: int((y == i).sum()) for i in range(len(CLASSES3))},
        "flip_rate": float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else 0.0,
        "mean_state_duration_bars": float(np.mean(runs)) if runs else 0.0,
    }


def train_one(work_raw, feature_set_name, cols):
    work = _with_features(work_raw, cols)
    train_mask = (work["timestamp"] >= TRAIN_START) & (work["timestamp"] < EVAL_START)
    eval_mask = work["timestamp"] >= EVAL_START
    train_part = work.loc[train_mask].copy()
    eval_part = work.loc[eval_mask].copy()
    train_obs, eval_obs, _, _ = _fit_obs(train_part, eval_part, cols)
    model = GaussianStateModel(STATES, N_ITER, SEED, sticky=STICKY).fit(train_obs)
    y_train = _current_labels3_thresholded(train_part, LABEL_CFG)
    y_eval = _current_labels3_thresholded(eval_part, LABEL_CFG)
    state_class = _state_class_matrix(model.filter_proba(train_obs), y_train)
    eval_proba = _class_proba(model.filter_proba(eval_obs), state_class)
    report = _eval(y_eval, eval_proba)
    report["feature_set"] = feature_set_name
    report["n_features"] = len(cols)
    report["train_rows"] = int(len(train_part))
    report["train_range"] = [str(train_part["timestamp"].min()), str(train_part["timestamp"].max())]
    report["eval_range"] = [str(eval_part["timestamp"].min()), str(eval_part["timestamp"].max())]
    return report


def main():
    print("loading training_features_2026_rebuilt.csv (UTC) ...", flush=True)
    tf = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv", low_memory=False)
    tf["timestamp"] = pd.to_datetime(tf["timestamp"])
    tf = tf.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    print(f"  {len(tf)} rows, {tf['timestamp'].min()} .. {tf['timestamp'].max()} (UTC)")

    eval_window_mask = tf["timestamp"] >= TRAIN_START
    for c in EVIDENCE_EXTRA_COLS:
        n_missing = int(tf.loc[eval_window_mask, c].isna().sum()) if c in tf.columns else -1
        n_total = int(eval_window_mask.sum())
        print(f"  {c}: {n_missing}/{n_total} missing (or -1 = column absent)")

    results = {}
    for name, cols in FEATURE_SETS.items():
        print(f"\n=== training {name} ({len(cols)} features) ===", flush=True)
        r = train_one(tf, name, cols)
        results[name] = r
        print({k: v for k, v in r.items() if k not in ("train_range", "eval_range")})

    print("\n\n==== SUMMARY ====")
    for name, r in results.items():
        print(f"{name:20s} balanced_accuracy={r['balanced_accuracy']:.4f}  accuracy={r['accuracy']:.4f}  "
              f"log_loss={r['log_loss']:.4f}  flip_rate={r['flip_rate']:.4f}  "
              f"recall={ {k: (round(v,3) if v is not None else None) for k,v in r['recall'].items()} }  "
              f"train_rows={r['train_rows']}  eval_rows={r['rows']}")

    import json
    out_path = ROOT / "tmp/eth_wide24_evidence_augmented_regime_test_20260826.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "train_start": str(TRAIN_START), "eval_start": str(EVAL_START),
        "hyperparams": {"states": STATES, "n_iter": N_ITER, "seed": SEED, "sticky": STICKY},
        "label_config": LABEL_CFG,
        "new_columns": EVIDENCE_EXTRA_COLS,
        "results": results,
    }, indent=2, default=str))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
