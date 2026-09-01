#!/usr/bin/env python3
"""Train the deployable BTC regime classifier (S24_K3) -- 2026-09-02.

BTC's FIRST dashboard regime classifier. Until now the BTC snapshot ribbon was a hard-coded grey
"model not available" band (see memory eth-dashboard-btc-regime-classifier-not-trained-todo-20260831).

Label selected by a BTC-native re-screen, NOT ported from ETH:
  Phase 1 docs/experiments/btc_regime_scalping_label_geometry_20260902.md    (transition axis closed, 0/16)
  Phase 2 docs/experiments/btc_regime_label_conditional_lift_20260902.md     (S24_K3 selected)
  Phase 3/3b docs/experiments/btc_regime_s24k3_label_train_20260902.md       (learnability + real gate)
ETH's winner S12_K3 scores only 3/10 on BTC; BTC's winner is the ORIGINAL RegimeEngine SCALE (S=24)
plus a K=3 (15min) confirm. Notably the TRAIN percentile-matching lands T1=0.2000 / T2=0.1600 --
RegimeEngine's own thresholds -- because S=24 *is* its scale, so this label is best read as
"RegimeEngine's trend/direction test, simplified and debounced", not as a different scale.

Model config, the 136 feature_cols and the feature medians are taken from the ETH GBM3 artifact so
the two assets' scorers stay structurally identical; the BTC canonical feature file carries the same
column set (128 present + the 8 that _with_raw_state12 derives).

⚠️ The quoted OOS figures come from BTC's first (and only) OOS look, 2026-07-01~2026-08-01 (9,141
bars, ~32d -- the canonical BTC feature file ends 2026-08-01 17:40). Research/dev score.
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

from research_btc_regime_s24k3_label_train_20260902 import (  # noqa: E402
    CLASSES3, DEBOUNCE_K, GBM3_HP, GBM3_MODEL_PATH, SCALE, SEED, TRAIN_END, TRAIN_START,
    load_btc_frame, s24k3_label,
)

MODEL_ID = "btc_regime_s24k3_20260902"
OUT_DIR = ROOT / f"tmp/{MODEL_ID}"


def main() -> None:
    src = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = src["feature_cols"], src["feature_medians"]

    df = load_btc_frame(feat_cols)
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    y, t1, t2 = s24k3_label(df, tr)

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    shares = {n: round(float((y[tr] == i).mean()), 4) for i, n in enumerate(CLASSES3)}
    print(f"BTC TRAIN {int(tr.sum()):,} bars {TRAIN_START.date()}~{TRAIN_END.date()} | shares {shares}")
    print(f"thresholds (TRAIN-only percentile match): T1={t1:.6f} T2={t2:.6f}")

    model = HistGradientBoostingClassifier(random_state=SEED, **GBM3_HP).fit(x[tr], y[tr])
    assert list(model.classes_) == [0, 1, 2], f"unexpected class order {model.classes_}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": MODEL_ID, "classes": CLASSES3, "feature_cols": feat_cols,
        "feature_medians": medians, "model": model, "config": GBM3_HP,
        "train_range": f"{TRAIN_START.isoformat()} ~ {TRAIN_END.isoformat()}",
        "oos_validated_bal_acc": 0.8687,
        "oos_validated_range": "2026-07-01 ~ 2026-08-01",
        "asset": "BTCUSDT",
        "cross_asset": "ETHUSDT",
        "label_spec": {"family": "scale-parameterized RegimeEngine-style 3-class",
                       "scale_bars": SCALE, "debounce_k": DEBOUNCE_K, "T1_er24": t1, "T2_er48": t2,
                       "definition": ("er_24=|c-c[-24]|/sum|diff|(24); er_48 likewise over 48; "
                                      "net_48=c-c[-48]; slope_24=EMA(c,24).pct_change(); "
                                      "trend=(er_24>=T1)|(er_48>=T2); bull=trend&net_48>0&slope_24>0; "
                                      "bear=mirror; chop=rest; then K=3 consecutive-bar confirm"),
                       "note": ("T1/T2 land on RegimeEngine's own 0.20/0.16 because S=24 IS its "
                                "scale -- the label's novelty is the debounce, not the scale.")},
        "notes": ("BTC's first dashboard regime classifier. Selected by a BTC-native re-screen "
                  "(ETH's S12_K3 scores 3/10 on BTC). vs RegimeEngine on BTC: classification "
                  "regresses (OOS bal_acc 0.8687 vs 0.9088, chop_P 0.8827 vs 0.9219, chop_R 0.9025 "
                  "vs 0.9208) but the predicted-chop evidence-signal gate improves (pooled +14.0% "
                  "and 13/16 cells positive vs +9.2% and 9/13; both-window-positive 7/16 vs 4/13, "
                  "and RegimeEngine's VAL mean is ~0 so its bigger OOS number is uncorroborated) "
                  "and the ribbon is far more stable (predicted flip_rate 0.0777 vs 0.1748). "
                  "Direction-proxy dependence is comparable (ablation -0.033 vs -0.030). "
                  "⚠️ Cross-asset naming trap: FeatureEngineer's *_btc columns hold the CROSS asset; "
                  "for this BTC-subject model they carry ETH -- see live_regime_btc_signal_20260902.py."),
    }
    joblib.dump(payload, OUT_DIR / "model.joblib")
    (OUT_DIR / "train_report.json").write_text(json.dumps(
        {k: v for k, v in payload.items() if k not in ("model", "feature_medians", "feature_cols")}
        | {"n_features": len(feat_cols), "train_class_shares": shares}, indent=2, ensure_ascii=False))
    print(f"wrote {OUT_DIR / 'model.joblib'}")


if __name__ == "__main__":
    main()
