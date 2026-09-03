#!/usr/bin/env python3
"""XRP 3-class 레짐 분류기 **S96_K9** 프로덕션 아티팩트 빌드 (S48_K6 교체본).

## 왜 교체하나

2026-09-03 튜닝 감사에서 기존 `S48_K6`이 **격자 상단 경계에서 뽑힌 값**임이 드러났다
(`SCALES=(6,12,24,48)`→S48, `DEBOUNCES=(1,3,6)`→K6, 둘 다 격자 끝. 게다가 `DEBOUNCES`는
ETH 스크립트에서 import한 것이고 K=12 배제 근거도 ETH의 lock-up 관측이었다).
격자를 S→192 / K→12로 넓히니 `S96_K9`가 네 축 전부에서 이겼다.

    Phase 2 (조건화 가치)   15/16  vs  S48_K6 10/16
    Phase 3 (학습가능성)    0.8676 vs  0.8364     <- 트레이드오프 없음(이 저장소에서 드묾)
    Phase 3b (실배포형태)   13/16 OOS +0.1437 vs 10/16 +0.0466
    표시 플리커             0.0358 vs 0.0476

## ⭐승격 근거 -- 미사용 창 단일 노출 (2026-04-01~06-30)

`scripts/confirm_xrp_regime_s96k9_unspent_window_20260903.py`, 사전등록 기준 3개 전부 통과:

    bal_acc     0.8439 >= 0.8219   (+0.0220)  ✅
    pred_flip   0.0339 <= 0.0513   (-0.0174)  ✅
    chop_recall 0.9173 >= 0.85                ✅

⚠️그 창은 이 확인으로 **소진**됐다. 재실행은 근거로 쓸 수 없다.

## ⚠️창 규율 -- 2026-07-01 이후를 읽지 않는다

기존 `build_xrp_regime_s48k6_model_20260903.py`는 2026-07-01~08-01에서 `oos_validated_bal_acc`를
찍었는데 **그 창은 S48_K6 채택으로 이미 소진**됐다. 여기서는 그 창을 평가에 쓰지 않고,
검증 수치로 위 확인 창(2026-04~06) 결과를 기록한다.
학습 구간은 기존과 동일하게 2026-06-30까지 쓴다(선택이 끝난 뒤 전 데이터로 적합하는 건 표준).

## 의미론

XRP 레짐 스케일이 4시간 -> **8시간**이 된다(ETH S12=1h, BTC S24=2h 중 가장 느림).
"XRP는 느릴수록 좋다"가 스케일·디바운스 두 축에서 일관되게 나온 결과다.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
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
CLEAN = _m("xrp_clean", "scripts/research_xrp_regime_extended_label_phase3_clean_20260903.py")

CANON = ROOT / "data/splits/year_oos/xrp_features_2024_2026.csv"
GBM3_MODEL_PATH = ROOT / "tmp/eth_regime_gbm3_independent_20260826/model.joblib"
OUT_DIR = ROOT / "tmp/xrp_regime_s96k9_20260903"
EXPECTED_ROWS = 272_490
TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30 23:55")
NEVER_READ_FROM = pd.Timestamp("2026-07-01")        # 소진된 원본 Phase3 창

SCALE, DEBOUNCE_K = 96, 9                           # ⭐확장 격자 승자
CONFIRMATION = {                                     # 미사용 창 단일 노출 결과 (소진됨)
    "window": "2026-04-01 ~ 2026-06-30",
    "fit_range": "2024-01-01 ~ 2026-03-31",
    "bal_acc": 0.8439, "chop_recall": 0.9173, "chop_precision": 0.8472,
    "bull_recall": 0.7828, "bear_recall": 0.8316, "pred_flip": 0.0339,
    "vs_deployed_s48k6": {"bal_acc": 0.8219, "chop_recall": 0.8875, "pred_flip": 0.0513},
    "preregistered_criteria_passed": 3,
    "single_exposure_spent": True,
}


def log(m): print(f"[xrp-s96k9] {m}", flush=True)


def main() -> int:
    t0 = time.time()
    from sklearn.ensemble import HistGradientBoostingClassifier

    n = sum(1 for _ in open(CANON)) - 1
    if abs(n - EXPECTED_ROWS) > 200:                       # ⭐자산 오염 가드
        raise RuntimeError(f"{CANON.name}: {n:,}행 != XRP 기대치 {EXPECTED_ROWS:,}")
    log(f"캐노니컬 {n:,}행 (자산 가드 통과)")

    payload = joblib.load(GBM3_MODEL_PATH)
    feat_cols, medians = payload["feature_cols"], payload["feature_medians"]
    df = P3.load_btc_frame(feat_cols)                      # 이름만 btc -- XRP canonical을 읽는다
    ts = df["timestamp"]
    tr = ((ts >= TRAIN_START) & (ts <= TRAIN_END)).to_numpy()
    log(f"TRAIN {int(tr.sum()):,}봉 ({TRAIN_START.date()} ~ {TRAIN_END.date()})")
    log(f"⚠️{NEVER_READ_FROM.date()} 이후 {int((ts >= NEVER_READ_FROM).sum()):,}봉은 평가에 쓰지 않는다"
        " (소진된 원본 Phase3 창)")

    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    # 라벨: 확인 스크립트와 **완전히 같은** 함수. 임계는 학습구간에서만 뽑는다.
    y = CLEAN.make_label(df["close"], tr, SCALE, DEBOUNCE_K)
    shares = {c: float(np.mean(y[tr] == i)) for i, c in enumerate(P3.CLASSES3)}
    log(f"S{SCALE}_K{DEBOUNCE_K} TRAIN shares " +
        " ".join(f"{c}={v:.3f}" for c, v in shares.items()))

    model = HistGradientBoostingClassifier(random_state=P3.SEED, **P3.GBM3_HP).fit(x.loc[tr], y[tr])
    log("적합 완료 (검증 수치는 소진된 확인 창의 기록값을 쓴다 -- 새 창을 열지 않는다)")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_cols": feat_cols, "feature_medians": medians},
                OUT_DIR / "model.joblib")
    report = {
        "model_id": "xrp_regime_s96k9_20260903", "classes": list(P3.CLASSES3),
        "asset": "XRPUSDT", "cross_asset": "BTCUSDT",
        "label": {"scale": SCALE, "debounce_k": DEBOUNCE_K,
                  "note": "확장 격자(S->192, K->12) 승자. 기존 S48_K6은 두 축 모두 격자 상단 경계였다."},
        "train_range": f"{TRAIN_START} ~ {TRAIN_END}",
        "train_class_shares": shares,
        "confirmation": CONFIRMATION,
        "replaces": "xrp_regime_s48k6_20260903",
        "spent_phase3_window_touched": False,
        "audit": "docs/experiments/xrp_tuning_gap_grid_boundary_audit_20260903.md",
        "runtime_sec": round(time.time() - t0, 1),
    }
    (OUT_DIR / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    log(f"artifact -> {OUT_DIR}/model.joblib  ({report['runtime_sec']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
