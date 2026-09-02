#!/usr/bin/env python3
"""레짐 분류기 재훈련 -- 학습 종료를 VAL 시작 이전으로 잘라 게이트 오염을 제거한다 (2026-09-02).

WHY
---
2026-09-02 게이트 앙상블 연구(docs/experiments/eth_regime_gated_costgate_and_ensemble_pnl_20260902.md)
에서 top1_btcchop이 OOS mean +23.94bp / PF 3.557이라는 이 저장소 최고 수치를 냈으나, 그 게이트를
만든 두 레짐 모델의 train_range가 2024-01-01 ~ 2026-06-30 이라 **평가창 VAL(2025-09~12) /
OOS(2026-01~03)를 통째로 포함**한다. 즉 그 구간의 chop/trend 판정은 예측이 아니라 적합이고,
라이브보다 정확한 게이트는 라이브보다 좋은 게이트 성과를 만든다. 원 문서도 "절대치는 라이브
추정이 아니다"라고 명시했다.

경고 신호도 같이 있었다: top1_btcchop의 개선폭이 VAL +1.06 / OOS +8.24로 8배 비대칭인데, 효과가
안정적이라면 두 창에서 비슷해야 한다. in-sample 게이트 + 작은 n의 노이즈가 만드는 전형이다.

WHAT
----
라벨·피처·하이퍼파라미터·시드를 **전부 고정**하고 학습 종료일만 2026-06-30 -> 2025-08-31로
당긴다(VAL 시작 2025-09-01 직전). 그러면 VAL/OOS 양 창에서 게이트가 진짜 out-of-sample 예측이
된다. 배포 모델은 손대지 않는다 -- 이건 연구 전용 변형이다.

라벨 임계값(T1/T2)은 두 스크립트 모두 train_mask 안에서만 백분위 보정하므로, 학습창이 줄면
임계값도 그 창 기준으로 다시 잡힌다. 이게 올바른 동작이다(OOS 정보 유입 없음).

주의: 학습창이 30개월 -> 20개월로 줄어 분류 정확도 자체는 다소 떨어질 수 있다. 그건 대가가
아니라 **정직한 수치**다 -- 라이브 게이트도 미래를 못 보고 돌기 때문이다.
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

CLEAN_TRAIN_END = pd.Timestamp("2025-08-31T23:55:00")   # VAL 시작(2025-09-01) 직전
OUT_ETH = ROOT / "tmp/eth_regime_s12k3_clean_20260902"
OUT_BTC = ROOT / "tmp/btc_regime_s24k3_clean_20260902"


def _fit_and_save(*, tag, df, feat_cols, medians, y, t1, t2, train_start, hp, seed,
                  classes, out_dir, deployed_ref) -> dict:
    ts = df["timestamp"]
    tr = ((ts >= train_start) & (ts <= CLEAN_TRAIN_END)).to_numpy()
    x = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    for c in feat_cols:
        x[c] = x[c].replace([np.inf, -np.inf], np.nan).fillna(medians.get(c, 0.0))

    shares = {n: round(float((y[tr] == i).mean()), 4) for i, n in enumerate(classes)}
    print(f"[{tag}] TRAIN {int(tr.sum()):,} bars {train_start.date()}~{CLEAN_TRAIN_END.date()} "
          f"| shares {shares} | T1={t1:.6f} T2={t2:.6f}")

    model = HistGradientBoostingClassifier(random_state=seed, **hp).fit(x[tr], y[tr])
    assert list(model.classes_) == [0, 1, 2], f"unexpected class order {model.classes_}"

    # 학습창 밖(=VAL/OOS 포함) 전체 구간에 대한 예측을 그대로 저장한다 -- 이게 게이트 재평가 입력.
    pred = model.predict(x)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"timestamp": ts, "regime": pred}).to_parquet(out_dir / "predictions.parquet",
                                                               index=False)
    joblib.dump({"model_id": out_dir.name, "classes": classes, "feature_cols": feat_cols,
                 "feature_medians": medians, "model": model,
                 "train_range": f"{train_start.isoformat()} ~ {CLEAN_TRAIN_END.isoformat()}",
                 "notes": "clean-cutoff research variant; NOT for deployment"},
                out_dir / "model.joblib")

    post = ~tr
    rep = {"tag": tag, "train_bars": int(tr.sum()), "train_shares": shares,
           "train_range": f"{train_start.isoformat()} ~ {CLEAN_TRAIN_END.isoformat()}",
           "deployed_train_range": deployed_ref,
           "chop_share_train": round(float((pred[tr] == 2).mean()), 4),
           "chop_share_post_train": round(float((pred[post] == 2).mean()), 4),
           "label_agreement_post_train": round(float((pred[post] == y[post]).mean()), 4),
           "thresholds": {"t1": t1, "t2": t2}}
    print(f"[{tag}] 학습창 밖 chop 비중 {rep['chop_share_post_train']:.3f} "
          f"(학습창 안 {rep['chop_share_train']:.3f}), 라벨일치 {rep['label_agreement_post_train']:.3f}")
    (out_dir / "train_report.json").write_text(json.dumps(rep, indent=2))
    return rep


def main() -> int:
    reports = []

    # ---- ETH S12_K3 ----
    from research_eth_regime_s12k3_label_train_20260902 import (
        GBM3_HP, GBM3_MODEL_PATH, SEED, load_frame, s12k3_label)
    from research_eth_regime_scalping_label_geometry_20260902 import TRAIN_START as ETH_TS
    src = joblib.load(GBM3_MODEL_PATH)
    df = load_frame()
    tr_clean = ((df["timestamp"] >= ETH_TS) & (df["timestamp"] <= CLEAN_TRAIN_END)).to_numpy()
    y, t1, t2 = s12k3_label(df, tr_clean)
    reports.append(_fit_and_save(
        tag="ETH_S12_K3_clean", df=df, feat_cols=src["feature_cols"],
        medians=src["feature_medians"], y=y, t1=t1, t2=t2, train_start=ETH_TS,
        hp=GBM3_HP, seed=SEED, classes=["bull", "bear", "chop"], out_dir=OUT_ETH,
        deployed_ref="2024-01-01 ~ 2026-06-30"))

    # ---- BTC S24_K3 ----
    from research_btc_regime_s24k3_label_train_20260902 import (
        CLASSES3 as BC, GBM3_HP as BHP, GBM3_MODEL_PATH as BMP, SEED as BSEED,
        TRAIN_START as BTC_TS, load_btc_frame, s24k3_label)
    bsrc = joblib.load(BMP)
    bdf = load_btc_frame(bsrc["feature_cols"])
    btr = ((bdf["timestamp"] >= BTC_TS) & (bdf["timestamp"] <= CLEAN_TRAIN_END)).to_numpy()
    by, bt1, bt2 = s24k3_label(bdf, btr)
    reports.append(_fit_and_save(
        tag="BTC_S24_K3_clean", df=bdf, feat_cols=bsrc["feature_cols"],
        medians=bsrc["feature_medians"], y=by, t1=bt1, t2=bt2, train_start=BTC_TS,
        hp=BHP, seed=BSEED, classes=BC, out_dir=OUT_BTC,
        deployed_ref="2024-01-01 ~ 2026-06-30"))

    print("\n=== 요약 ===")
    for r in reports:
        print(f"  {r['tag']}: TRAIN {r['train_bars']:,}봉, 학습창밖 chop {r['chop_share_post_train']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
