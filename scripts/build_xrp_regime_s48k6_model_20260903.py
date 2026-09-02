#!/usr/bin/env python3
"""XRP 레짐 분류기 **배포 아티팩트** 학습 — S48_K6, 대시보드 리본용.

Phase 3(`research_xrp_regime_s48k6_label_train_20260903.py`)은 학습 가능성과 게이트 개선을
**평가**만 했고 모델을 저장하지 않았다. 이 스크립트가 배포용 `model.joblib`을 만든다.

## 아티팩트 계약 (BTC와 동일)

`live_regime_btc_signal_20260902.py`가 읽는 키를 그대로 맞춘다:
`model` / `feature_cols` / `feature_medians`. 그래야 라이브 스코어러를 같은 모양으로 포팅할 수 있다.

## ⭐교차자산 슬롯 (학습/추론 파리티)

`FeatureEngineer`는 교차자산 컬럼을 `close_btc`/`volume_btc`/`quote_volume_btc`로 **하드코딩**한다.
XRP 캐노니컬(`build_xrp_raw_frame_20260903.py`)은 그 슬롯에 **BTC**를 넣어 만들었으므로,
라이브 스코어러도 반드시 BTC를 그 슬롯에 넣어야 한다.
(BTC 캐노니컬은 ETH가 들어있다 — 자산마다 파트너가 다르다.)

## 근거

Phase 2에서 S48_K6 선택(10/16, ETH S12_K3은 5/16, BTC S24_K3은 3/16).
Phase 3에서 REF 대비 배포 형태 우위 확인:
  S48_K6  bal_acc 0.8644  pred_flip 0.0458  게이트 8/16  OOS +0.0306
  REF     bal_acc 0.9113  pred_flip 0.1787  게이트 2/13  OOS **-0.0756**

⚠️Phase 3은 1차 실행이 BTC 캐노니컬을 읽어 무효였다(변수명만 XRP였다). 위 수치는 정정 후 값이다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _m(n, rel):
    sp = importlib.util.spec_from_file_location(n, ROOT / rel)
    m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m


P3 = _m("xrp_p3", "scripts/research_xrp_regime_s48k6_label_train_20260903.py")

CANON = ROOT / "data/splits/year_oos/xrp_features_2024_2026.csv"
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
OUT_DIR = ROOT / "tmp/xrp_regime_s48k6_20260903"
EXPECTED_ROWS = 272_490
TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30 23:55")
OOS_START, OOS_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-01 23:55")


def log(m): print(f"[xrp-regime-model] {m}", flush=True)


def main() -> int:
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import balanced_accuracy_score

    n = sum(1 for _ in open(CANON)) - 1
    if abs(n - EXPECTED_ROWS) > 200:                       # ⭐자산 오염 가드
        raise RuntimeError(f"{CANON.name}: {n:,}행 != XRP 기대치 {EXPECTED_ROWS:,}")
    log(f"캐노니컬 {n:,}행 (자산 가드 통과)")

    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = P3.load_btc_frame(feat_cols)                      # XRP_CANON을 읽는다(정정 완료)
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    oos = ((ts >= OOS_START) & (ts <= OOS_END)).to_numpy()
    log(f"TRAIN {int(tr.sum()):,} / OOS {int(oos.sum()):,}")

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))
    y, t1, t2 = P3.s24k3_label(df, tr)                     # 함수명은 유산, SCALE/K는 48/6
    log(f"S48_K6 T1={t1:.6f} T2={t2:.6f}  TRAIN shares " +
        " ".join(f"{c}={np.mean(y[tr]==i):.3f}" for i, c in enumerate(P3.CLASSES3)))

    model = HistGradientBoostingClassifier(random_state=P3.SEED, **P3.GBM3_HP).fit(x.loc[tr], y[tr])
    ba = float(balanced_accuracy_score(y[oos], model.predict(x.loc[oos]))) if oos.sum() else None
    pred = model.predict(x.loc[oos]) if oos.sum() else np.array([])
    flip = float(np.mean(pred[1:] != pred[:-1])) if len(pred) > 1 else None
    log(f"OOS bal_acc={ba:.4f}  pred_flip={flip:.4f}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feat_cols, "feature_medians": medians},
                OUT_DIR / "model.joblib")
    (OUT_DIR / "train_report.json").write_text(json.dumps({
        "model_id": "xrp_regime_s48k6_20260903", "classes": list(P3.CLASSES3),
        "asset": "XRPUSDT", "cross_asset": "BTCUSDT",
        "train_range": f"{TRAIN_START} ~ {TRAIN_END}", "oos_validated_bal_acc": round(ba, 4) if ba else None,
        "oos_validated_range": f"{OOS_START.date()} ~ {OOS_END.date()}",
        "oos_pred_flip_rate": round(flip, 4) if flip else None,
        "n_features": len(feat_cols),
        "train_class_shares": {c: round(float(np.mean(y[tr] == i)), 4) for i, c in enumerate(P3.CLASSES3)},
        "label_spec": {"family": "scale-parameterized RegimeEngine-style 3-class",
                       "scale_bars": P3.SCALE, "debounce_k": P3.DEBOUNCE_K, "T1_er24": t1, "T2_er48": t2},
        "notes": ("XRP 첫 대시보드 레짐 분류기. XRP 자체 재스크리닝으로 S48_K6 선택 "
                  "(ETH S12_K3은 5/16, BTC S24_K3은 3/16으로 거의 최하위). "
                  "REF 대비 분류는 후퇴하나(bal_acc 0.8644 vs 0.9113) 예측-chop 게이트는 압승 "
                  "(8/16 vs 2/13, OOS +0.0306 vs -0.0756)이고 플리커는 3.9배 낮다. "
                  "⚠️교차자산 슬롯(close_btc 등)에는 BTC가 들어간다 -- 라이브도 동일해야 파리티."),
    }, ensure_ascii=False, indent=2))
    log(f"저장 -> {OUT_DIR}/model.joblib")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
