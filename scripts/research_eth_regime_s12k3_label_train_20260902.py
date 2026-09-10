#!/usr/bin/env python3
"""Phase 3 of the scalping regime LABEL redesign: is the S12_K3 candidate label LEARNABLE by the
deployed GBM3 setup? -- user directive 2026-09-02 ("Phase 3 진행해줘").

LINEAGE
  Phase 1 (eth_regime_scalping_label_geometry_20260902.md): closed the "faster transitions" framing --
    no scale/debounce combination had a significant transition edge, and shrinking scale made it worse.
  Phase 2 (eth_regime_label_conditional_lift_20260902.md): reframed to conditioning value and picked
    S12_K3 -- 10/16 signal-side cells positive in BOTH windows (10/11 among adequately-sampled cells)
    vs the deployed label's 3/14. Circularity pre-check passed (er_12/er_24 have no >=0.80 proxy in
    the 136 features; slope_12 is literally mtf_trend_1h, already excluded by GBM3).

⚠️ THIS PHASE SPENDS ONE OOS LOOK on 2026-07-01~08-19, a window already consumed by ~8+ prior
regime-classifier rounds. Research/dev score, not promotion evidence. Fresh-Forward not satisfied.

MODEL HELD FIXED: GBM3's exact HistGradientBoostingClassifier config and 136 feature_cols, read from
the deployed artifact. Only the LABEL changes. No GBM2-lineage import.

THREE CHECKS THE USER ASKED FOR

  1. ⭐CHOP-CLASS METRICS ARE PRIMARY, NOT bal_acc. Phase 2's circularity pre-check showed the
     direction components (net_24, slope_12) have 0.80-0.91 correlates among the features
     (vwap_dist_24, kalman_velocity, rsi, hma_slope, mtf_trend_4h), so overall bal_acc can be
     inflated by "get the direction right from momentum features". This label's value lives on the
     chop boundary, so chop recall AND chop precision are reported first. Chop PRECISION matters most
     for the actual use case: the evidence-signal gate fires on PREDICTED chop, so what we need is
     "when the model says chop, is it really chop".
  2. DIRECTION-PROXY ABLATION: every arm is run twice, with and without those 5 correlates, so the
     share of bal_acc that is really "direction from momentum" is measurable.
  3. LIKE-FOR-LIKE BASELINE: the deployed RegimeEngine label is re-trained and re-scored by this
     same script (not quoted from the artifact), so both labels face identical data/config/seed.

LEAKAGE CARE: T1/T2 are calibrated on TRAIN ONLY (Phase 2 calibrated on the full evidence frame,
which is fine for a label-geometry study but would leak here). Labels are computed once over the
full continuous series -- both the efficiency-ratio rollings and the debounce state machine need an
unbroken sequence -- then sliced; every component is backward-looking so this is causal.
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
from research_eth_regime_scalping_label_geometry_20260902 import (  # noqa: E402
    TRAIN_CSVS,
    TRAIN_END,
    TRAIN_START,
    _debounce,
    _run_lengths,
    efficiency_ratio,
    scaled_label,
)

GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
GBM3_HP = dict(max_depth=10, learning_rate=0.04, max_iter=400, l2_regularization=2.0)
SEED = 7529
OOS_START = pd.Timestamp("2026-07-01T00:00:00")
OOS_END = pd.Timestamp("2026-08-19T23:55:00")
CLASSES3 = ["bull", "bear", "chop"]
SCALE, DEBOUNCE_K = 12, 3
# 5 features Phase 2's pre-check found correlated >=0.80 with the NEW label's direction components
DIRECTION_PROXIES = ["vwap_dist_24", "kalman_velocity", "rsi", "hma_slope", "mtf_trend_4h"]
OUT_DIR = ROOT / "tmp/eth_regime_s12k3_label_train_20260902"


def load_frame() -> pd.DataFrame:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    df = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last").reset_index(drop=True)
    df = df[(df["timestamp"] >= TRAIN_START) & (df["timestamp"] <= OOS_END)].reset_index(drop=True)
    return _with_raw_state12(df)


def deployed_label(df: pd.DataFrame) -> np.ndarray:
    d = df.copy()
    if "mtf_trend_1h" not in d.columns:
        d["mtf_trend_1h"] = d["close"].ewm(span=12, adjust=False).mean().pct_change().fillna(0.0)
    lab = RegimeEngine().compute(d)
    y = np.full(len(df), 2, dtype=int)
    y[lab["regime_bull"].to_numpy() > 0] = 0
    y[lab["regime_bear"].to_numpy() > 0] = 1
    return y


def s12k3_label(df: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Thresholds calibrated on TRAIN ONLY (percentile-matched to the deployed label's own firing
    rates there), then applied to the whole series -- no OOS information enters the label."""
    close = df["close"]
    er24_tr = efficiency_ratio(close, 24)[train_mask]
    er48_tr = efficiency_ratio(close, 48)[train_mask]
    rate1, rate2 = float((er24_tr >= 0.20).mean()), float((er48_tr >= 0.16).mean())
    t1 = float(efficiency_ratio(close, SCALE)[train_mask].quantile(1.0 - rate1))
    t2 = float(efficiency_ratio(close, 2 * SCALE)[train_mask].quantile(1.0 - rate2))
    return _debounce(scaled_label(close, SCALE, t1, t2), DEBOUNCE_K), t1, t2


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    cm = confusion_matrix(y, pred, labels=[0, 1, 2])
    runs = _run_lengths(pred)
    out = {"balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
           "flip_rate": round(float(np.mean(pred[1:] != pred[:-1])), 4),
           "median_run_bars": float(np.median(runs)) if runs else 0.0}
    for i, n in enumerate(CLASSES3):
        sup, prd = cm[i].sum(), cm[:, i].sum()
        out[f"{n}_recall"] = round(float(cm[i, i] / sup), 4) if sup else None
        out[f"{n}_precision"] = round(float(cm[i, i] / prd), 4) if prd else None
        out[f"{n}_label_share"] = round(float((y == i).mean()), 4)
        out[f"{n}_pred_share"] = round(float((pred == i).mean()), 4)
    return out


def main() -> None:
    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols = payload["feature_cols"]
    medians = payload["feature_medians"]
    print(f"GBM3 artifact: {len(feat_cols)} features | deployed OOS bal_acc="
          f"{payload['oos_validated_bal_acc']} (artifact record)")
    print("⚠️ This phase spends ONE OOS look on 2026-07-01~08-19 (already ~8+ consumed).\n")

    df = load_frame()
    ts = df["timestamp"]
    tr_m = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    oos_m = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()

    labels = {"DEPLOYED_RegimeEngine": deployed_label(df)}
    y_new, t1, t2 = s12k3_label(df, tr_m)
    labels[f"S{SCALE}_K{DEBOUNCE_K}"] = y_new
    print(f"S12_K3 thresholds calibrated on TRAIN only: T1={t1:.4f} T2={t2:.4f}")

    x_full = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x_full[c] = x_full[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    feature_sets = {
        "full136": feat_cols,
        "ablated": [c for c in feat_cols if c not in DIRECTION_PROXIES],
    }
    print(f"ablation drops {len([c for c in DIRECTION_PROXIES if c in feat_cols])} of "
          f"{len(DIRECTION_PROXIES)} direction proxies: "
          f"{[c for c in DIRECTION_PROXIES if c in feat_cols]}\n")

    results = {}
    for lname, y in labels.items():
        for fname, cols in feature_sets.items():
            x = x_full[cols]
            model = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP)
            model.fit(x[tr_m], y[tr_m])
            r = evaluate(y[oos_m], model.predict(x[oos_m]))
            r["train_label_shares"] = {n: round(float((y[tr_m] == i).mean()), 4)
                                       for i, n in enumerate(CLASSES3)}
            r["oos_label_flip_rate"] = round(float(np.mean(y[oos_m][1:] != y[oos_m][:-1])), 4)
            results[f"{lname}__{fname}"] = r
            print(f"{lname:22s} {fname:8s}  bal_acc={r['balanced_accuracy']:.4f}  "
                  f"chop_R={r['chop_recall']:.4f} chop_P={r['chop_precision']:.4f}  "
                  f"bull_R={r['bull_recall']:.4f} bear_R={r['bear_recall']:.4f}  "
                  f"pred_flip={r['flip_rate']:.4f} (label_flip={r['oos_label_flip_rate']:.4f})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps({
        "config": {"scale": SCALE, "debounce_k": DEBOUNCE_K, "T1": t1, "T2": t2,
                   "gbm_hp": GBM3_HP, "seed": SEED, "n_features": len(feat_cols),
                   "direction_proxies_dropped": DIRECTION_PROXIES,
                   "train": f"{TRAIN_START}~{TRAIN_END}", "oos": f"{OOS_START}~{OOS_END}"},
        "oos_purity_disclosure": ("2026-07-01~08-19 already consumed by ~8+ prior regime rounds; "
                                  "research/dev only, NOT promotion evidence, this run adds one more touch."),
        "primary_metric_note": ("chop_precision/chop_recall are primary, not balanced_accuracy -- "
                                "direction components have 0.80-0.91 feature correlates so bal_acc "
                                "can be carried by momentum features (see the ablation arms)."),
        "results": results}, indent=2, ensure_ascii=False))

    print("\n=== direction-proxy ablation delta (full136 -> ablated) ===")
    for lname in labels:
        a, b = results[f"{lname}__full136"], results[f"{lname}__ablated"]
        print(f"  {lname:22s} bal_acc {a['balanced_accuracy']:.4f} -> {b['balanced_accuracy']:.4f} "
              f"({b['balanced_accuracy']-a['balanced_accuracy']:+.4f}) | "
              f"chop_P {a['chop_precision']:.4f} -> {b['chop_precision']:.4f} "
              f"({b['chop_precision']-a['chop_precision']:+.4f})")
    print(f"\nWrote {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
