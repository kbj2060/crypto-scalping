#!/usr/bin/env python3
"""Train the deployable S12_K3 regime classifier -- 2026-09-02, user directive "대시보드에 배포해줘".

Replaces the label of the deployed GBM3 (scripts/live_regime_gbm3_signal_20260826.py,
tmp/eth_regime_gbm3_independent_20260826/model.joblib) while holding the MODEL fixed: identical
HistGradientBoostingClassifier config, identical 136 feature_cols, identical TRAIN range. Only the
target changes, from RegimeEngine's 2h/4h-scale label to the S=12 (1h) / K=3 (15min confirm) label
selected in the Phase 1-3b study chain:

  docs/experiments/eth_regime_scalping_label_geometry_20260902.md      (Phase 1: transition-edge axis closed)
  docs/experiments/eth_regime_label_conditional_lift_20260902.md       (Phase 2: S12_K3 selected)
  docs/experiments/eth_regime_s12k3_label_train_20260902.md            (Phase 3/3b: adopt recommendation)

WHY THIS SHIPS DESPITE LOWER ACCURACY (user decision, made with the tradeoff stated):
  classification accuracy REGRESSES (OOS bal_acc 0.8550 vs 0.9108, chop precision 0.8670 vs 0.9202)
  but the two things the label is actually for both improve --
    * evidence-signal gate quality (predicted-chop conditional lift, pooled +9.8% / 14 of 16 cells
      positive, vs the deployed label's -0.8% / 6 of 14);
    * display stability (predicted flip_rate 0.0965 vs 0.1803 -- the "visibly flickery" complaint
      that spawned the GBM2 project, halved by the label change alone).

The regime endpoint is DISPLAY-ONLY (dashboard/live/app.js ribbon + the Snapshot tab's
liquidation-map overlay); no trading path consumes it. Verified before deploying.

Artifact payload schema is byte-compatible with the GBM3 one that live_regime_gbm3_signal_20260826.py
already reads (model_id/classes/feature_cols/feature_medians/model/config/train_range/notes), so the
live scorer needs a one-line MODEL_PATH change and nothing else.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from research_eth_regime_s12k3_label_train_20260902 import (  # noqa: E402
    GBM3_HP, GBM3_MODEL_PATH, SEED, load_frame, s12k3_label,
)
from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_END, TRAIN_START  # noqa: E402

MODEL_ID = "eth_regime_s12k3_20260902"
OUT_DIR = ROOT / f"tmp/{MODEL_ID}"
CLASSES3 = ["bull", "bear", "chop"]     # label ints 0/1/2 -- same coding/order as the GBM3 artifact


def main() -> None:
    src = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = src["feature_cols"], src["feature_medians"]

    df = load_frame()
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    y, t1, t2 = s12k3_label(df, tr)

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    shares = {n: round(float((y[tr] == i).mean()), 4) for i, n in enumerate(CLASSES3)}
    print(f"TRAIN {int(tr.sum()):,} bars {TRAIN_START.date()}~{TRAIN_END.date()} | shares {shares}")
    print(f"label thresholds calibrated on TRAIN: T1={t1:.6f} T2={t2:.6f}")

    model = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[tr], y[tr])
    assert list(model.classes_) == [0, 1, 2], f"unexpected class order {model.classes_}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID,
        "classes": CLASSES3,
        "feature_cols": feat_cols,
        "feature_medians": medians,
        "model": model,
        "config": GBM3_HP,
        "train_range": f"{TRAIN_START.isoformat()} ~ {TRAIN_END.isoformat()}",
        "oos_validated_bal_acc": 0.8550,
        "oos_validated_range": "2026-07-01 ~ 2026-08-19",
        "label_spec": {"family": "scale-parameterized RegimeEngine-style 3-class",
                       "scale_bars": 12, "debounce_k": 3, "T1_er12": t1, "T2_er24": t2,
                       "definition": ("er_12=|c-c[-12]|/sum|diff|(12); er_24 likewise over 24; "
                                      "net_24=c-c[-24]; slope_12=EMA(c,12).pct_change(); "
                                      "trend=(er_12>=T1)|(er_24>=T2); bull=trend&net_24>0&slope_12>0; "
                                      "bear=mirror; chop=rest; then K=3 consecutive-bar confirm"),
                       "thresholds_calibrated_on": "TRAIN only, percentile-matched to the deployed "
                                                   "label's P(er_24>=0.20) and P(er_48>=0.16)"},
        "notes": ("Label-only replacement for eth_regime_gbm3_independent_20260826 (model config and "
                  "the same 136 feature_cols held fixed, incl. its exclusion of the 5 columns that "
                  "are literal/monotonic proxies of RegimeEngine's label formula). Accuracy "
                  "regresses (bal_acc 0.8550 vs 0.9108, chop_P 0.8670 vs 0.9202) but predicted-chop "
                  "gate quality (+9.8%/14-of-16 vs -0.8%/6-of-14) and display stability "
                  "(pred flip 0.0965 vs 0.1803) both improve. OOS window 2026-07-01~08-19 was "
                  "already consumed by ~8+ prior regime rounds -- research/dev score, not a "
                  "single-touch Fresh-Forward result. See docs/experiments/"
                  "eth_regime_s12k3_label_train_20260902.md."),
    }
    joblib.dump(payload, OUT_DIR / "model.joblib")
    (OUT_DIR / "train_report.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ("model", "feature_medians", "feature_cols")}
        | {"n_features": len(feat_cols), "train_class_shares": shares}, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
