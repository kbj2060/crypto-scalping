"""전신(Omega4.6.1) wide24 HMM 레짐 vs 대시보드 S12_K3 GBM 레짐 — 같은 봉 위 실측 비교.

하위 프로젝트 `omega461_regimegbm_rebuild_20260909`의 Phase 0 근거 문서용.
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md

두 분류기는 이름만 3-class(bull/bear/chop)를 공유하고 계보가 다르다:
  · 전신   : 12-state GaussianStateModel(HMM) + RobustScaler, 24피쳐(wide24), 2024년만 학습
  · 대시보드: HistGradientBoostingClassifier, 136피쳐(wide24의 상위집합), 2024-01~2026-06 학습

전신 쪽 출력은 이미 사이드카 CSV 에 계산돼 있으므로 그대로 읽고, 대시보드 쪽은 라이브
스코어러와 **동일한 경로**(`_with_raw_state12` → feature_medians 대체 → predict_proba)로
같은 프레임 위에서 채점한다. `_with_raw_state12()` 를 빼먹으면 8개 state7_*/state12_* 컬럼이
조용히 median 으로 대체된다(2026-08-26 에 실제로 발생했던 버그) — 반드시 호출한다.

⚠️ 이 비교는 **분류기 출력의 일치도**를 재는 것이지 어느 쪽이 더 정확한지를 재는 게 아니다.
두 모델은 서로 다른 라벨(balancedish_adx16 vs S12_K3)을 학습했으므로 공통 정답이 없다.
각자의 검증 bal_acc(0.7638 vs 0.8550)는 서로 다른 타깃에 대한 값이라 직접 비교 불가다.

⚠️ 대시보드 GBM 의 학습구간(2024-01-01~2026-06-30)은 이 비교 구간을 포함한다. 즉 아래 수치의
대시보드 쪽은 부분적으로 in-sample 이다. 이 비교의 목적(두 분류기가 얼마나 다르게 판정하는가)에는
영향이 적지만, **이 GBM 을 그대로 오메가 피쳐로 쓰면 미래참조**라는 게 계약의 최우선 이슈다.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402

BASE = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
SIDECAR = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                / "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv")
DASH_MODEL = ROOT / "tmp/eth_regime_s12k3_20260902/model.joblib"
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"

CLASSES = ("bull", "bear", "chop")
WIDE24_PROB_COLS = [f"regime3_current_sensitive_wide24_{c}_prob" for c in CLASSES]


def flip_rate(a: np.ndarray) -> float:
    return float((a[1:] != a[:-1]).mean())


def median_run(a: np.ndarray) -> float:
    runs, c = [], 1
    for i in range(1, len(a)):
        if a[i] == a[i - 1]:
            c += 1
        else:
            runs.append(c)
            c = 1
    runs.append(c)
    return float(np.median(runs))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(BASE, low_memory=False)
    base["timestamp"] = pd.to_datetime(base["timestamp"])
    side = pd.read_csv(SIDECAR, low_memory=False)
    side["timestamp"] = pd.to_datetime(side["timestamp"])
    df = base.merge(side, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    print(f"공통 봉 {len(df)}  {df.timestamp.iloc[0]} ~ {df.timestamp.iloc[-1]}", flush=True)

    omega = np.array(CLASSES)[df[WIDE24_PROB_COLS].to_numpy().argmax(1)]

    pay = joblib.load(DASH_MODEL)
    cols, med = pay["feature_cols"], pd.Series(pay["feature_medians"])
    feats = _with_raw_state12(df.copy())
    missing = [c for c in cols if c not in feats.columns]
    print(f"median 으로 대체된 피쳐: {len(missing)}개 {missing[:8]}", flush=True)
    X = (feats.reindex(columns=cols).apply(pd.to_numeric, errors="coerce")
              .replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0))
    dash = np.array(pay["classes"])[pay["model"].predict_proba(X).argmax(1)]

    rows = {}
    print(f"\n{'':14s}{'bull':>8s}{'bear':>8s}{'chop':>8s}{'flip율':>10s}{'중앙지속':>10s}", flush=True)
    for name, arr in (("omega_wide24_hmm", omega), ("dashboard_s12k3_gbm", dash)):
        shares = {c: float((arr == c).mean()) for c in CLASSES}
        rows[name] = {"shares": shares, "flip_rate": flip_rate(arr), "median_run_bars": median_run(arr)}
        print(f"{name:14s}{shares['bull']:8.3f}{shares['bear']:8.3f}{shares['chop']:8.3f}"
              f"{rows[name]['flip_rate']:10.4f}{rows[name]['median_run_bars']:10.1f}", flush=True)

    agree = float((omega == dash).mean())
    ct = pd.crosstab(pd.Series(omega, name="omega"), pd.Series(dash, name="dash"), normalize="all") * 100
    head_on = ((omega == "bull") & (dash == "bear")) | ((omega == "bear") & (dash == "bull"))
    trend_chop = ((omega != "chop") & (dash == "chop")) | ((omega == "chop") & (dash != "chop"))

    print(f"\n일치율 {agree*100:.1f}%", flush=True)
    print("\n교차표 (행=omega, 열=dash, %):\n", ct.round(2), flush=True)
    print(f"\n정면충돌(bull<->bear) {int(head_on.sum())}봉 ({head_on.mean()*100:.2f}%)", flush=True)
    print(f"추세<->횡보 경계 불일치 {int(trend_chop.sum())}봉 ({trend_chop.mean()*100:.2f}%)", flush=True)

    report = {
        "window": {"start": str(df.timestamp.iloc[0]), "end": str(df.timestamp.iloc[-1]), "bars": int(len(df))},
        "classifiers": rows,
        "agreement_rate": agree,
        "crosstab_pct": json.loads(ct.round(4).to_json()),
        "head_on_conflict": {"bars": int(head_on.sum()), "share": float(head_on.mean())},
        "trend_chop_boundary": {"bars": int(trend_chop.sum()), "share": float(trend_chop.mean())},
        "caveats": [
            "두 모델은 서로 다른 라벨을 학습했으므로 공통 정답이 없다 -- 정확도 직접 비교 불가.",
            "대시보드 GBM 학습구간(2024-01-01~2026-06-30)이 이 비교 구간을 포함한다(부분 in-sample).",
        ],
    }
    (OUT / "regime_classifier_comparison.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame({"timestamp": df["timestamp"], "omega": omega, "dash": dash}).to_csv(
        OUT / "regime_labels_side_by_side.csv", index=False)
    print(f"\n산출물: {OUT}/regime_classifier_comparison.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
