"""진단 — "상승인데 bear, 하락인데 bull" 이 라벨 탓인가 코드 버그인가.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계기: 사용자 지적(2026-09-09) "차트를 보니까 상승하는데 bear이고 하락하는데 bull 인 경우가
많은데 라벨 자체가 그래?"

두 가지 가능성을 분리해서 검증한다
--------------------------------
**(A) 코드 버그** — 클래스 인덱스 매핑이 뒤집혔을 가능성.
    · 라벨 함수는 0=bull / 1=bear / 2=chop
    · `HistGradientBoostingClassifier.predict_proba` 는 `classes_` 순서(=정렬된 [0,1,2])로 낸다
    · 사이드카는 `{prefix}bull_prob = p[:,0]` 으로 쓴다
    이 사슬이 실제로 맞는지 **모델의 `classes_` 를 직접 읽어** 확인한다. 추가로 라벨 자체가
    과거 수익과 맞는 부호인지(= bull 구간의 직전 수익이 양수인지) 검정한다. 이게 뒤집혀 있으면
    코드 버그다.

**(B) 라벨의 후행성** — balancedish 는 후행 지표로만 만들어진다:
        ema21 = EMA(close,21);  slope = (ema21 − ema21.shift(5)) / (close*5)
        adx   = ADX(14)
    둘 다 **과거만** 본다. 그래서 고점 직후에는 가격이 내려가는데 slope 는 아직 양수라 bull 이,
    저점 직후에는 가격이 오르는데 slope 가 아직 음수라 bear 가 유지된다. **전환점에서 부호가
    반대로 보이는 것은 정의상 필연**이다. debounce(K) 를 걸면 그 지연이 더 커진다.

무엇을 재는가
------------
1. **매핑 검증**: 모델 `classes_` 순서, 그리고 라벨/예측별로 "직전 수익 부호와 일치하는 비율".
   bull 태그의 직전 수익이 압도적으로 양수면 매핑은 정상이다.
2. **리드-랙 상관**: 방향신호 s_t ∈ {+1 bull, −1 bear, 0 chop} 와 수익률 r_{t+h} 의 상관을
   h ∈ [−288, +288] 에서 훑어 **상관이 최대가 되는 h** 를 찾는다. h<0 이면 그 신호는 **과거를
   설명**하는 것이고(후행), h>0 이면 미래를 예측하는 것이다.
3. **단계별 지연 분해**: 라벨 → 모델 예측(K=0) → debounce(K=3/6) 로 가면서 지연이 얼마나
   커지는지. 사용자가 본 현상이 라벨 탓인지 debounce 탓인지 가른다.

준수: 신규 학습 없음. 라이브 파일 미변경.
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

from experiment_regime3_current_hmm_wide24_20260529 import _current_labels3_thresholded  # noqa: E402
from research_eth_regime_scalping_label_geometry_20260902 import _debounce  # noqa: E402

HMM_MODEL = (ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
                  / "regime3_current_sensitive_hmm_wide24_2024.joblib")
BAL = ("balgbm", ROOT / "data/ensemble/supervised/omega461_balgbm_cut2509_20260909",
       "regime3_balgbm_cut2509_", ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_balgbm_cut2509_model.joblib", 3)
NOBB = ("balnobb", ROOT / "data/ensemble/supervised/omega461_balnobb_cut2509_20260909",
        "regime3_balnobb_cut2509_", ROOT / "tmp/omega461_regimegbm_rebuild_20260909/regime_balnobb_cut2509_model.joblib", 6)
ARMS = [BAL, NOBB]
CLASSES = ("bull", "bear", "chop")
OOS = ("2026-01-01", "2026-02-28 23:55:00")
LAGS = np.arange(-288, 289, 6)
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909"


def load() -> pd.DataFrame:
    b = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
                    low_memory=False, parse_dates=["timestamp"])
    for _, d, pref, _, _ in ARMS:
        s = pd.read_csv(d / f"training_features_2026_rebuilt_{pref}sidecar.csv",
                        parse_dates=["timestamp"],
                        usecols=["timestamp"] + [f"{pref}{c}_prob" for c in CLASSES])
        b = b.merge(s, on="timestamp", how="inner")
    b = b.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return b[(b.timestamp >= OOS[0]) & (b.timestamp <= OOS[1])].reset_index(drop=True)


def past_future_agreement(pred: np.ndarray, close: np.ndarray, h: int) -> dict:
    """bull/bear 태그가 '직전 h봉 수익'/'이후 h봉 수익' 부호와 맞는 비율."""
    past = np.full(len(close), np.nan); past[h:] = (close[h:] - close[:-h]) / close[:-h]
    fut = np.full(len(close), np.nan); fut[:-h] = (close[h:] - close[:-h]) / close[:-h]
    out = {}
    for tag, idx, sgn in (("bull", 0, +1), ("bear", 1, -1)):
        m = pred == idx
        for nm, arr in (("past", past), ("fut", fut)):
            v = arr[m & np.isfinite(arr)]
            out[f"{tag}_{nm}_agree"] = round(float((np.sign(v) == sgn).mean()), 4) if len(v) else None
            out[f"{tag}_{nm}_mean_bp"] = round(float(v.mean() * 1e4), 1) if len(v) else None
    return out


def lead_lag_peak(pred: np.ndarray, close: np.ndarray, step: int = 12) -> dict:
    """s_t(+1/0/-1) 와 r_{t+h}(step봉 수익)의 상관을 h 스윕. 최대 상관의 h 를 반환."""
    s = np.where(pred == 0, 1.0, np.where(pred == 1, -1.0, 0.0))
    r = np.full(len(close), np.nan)
    r[:-step] = (close[step:] - close[:-step]) / close[:-step]
    cors = []
    for h in LAGS:
        if h >= 0:
            a, b = s[:len(s) - h], r[h:]
        else:
            a, b = s[-h:], r[:len(r) + h]
        m = np.isfinite(a) & np.isfinite(b)
        cors.append(float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 100 else np.nan)
    cors = np.array(cors)
    i = int(np.nanargmax(cors))
    return {"peak_lag_bars": int(LAGS[i]), "peak_corr": round(float(cors[i]), 4),
            "corr_at_0": round(float(cors[np.where(LAGS == 0)[0][0]]), 4),
            "curve": {int(l): (round(float(c), 4) if np.isfinite(c) else None)
                      for l, c in zip(LAGS, cors)}}


def main() -> int:
    df = load()
    close = pd.to_numeric(df["close"], errors="raise").to_numpy(np.float64)
    cfg = joblib.load(HMM_MODEL)["label_config"]
    y_bal = _current_labels3_thresholded(df, cfg)

    print("=" * 90)
    print("[A] 매핑 검증 — 모델 classes_ 순서")
    for name, _, _, mp, _ in ARMS:
        pay = joblib.load(mp)
        print(f"  {name:9s} model.classes_ = {list(pay['model'].classes_)}   "
              f"payload['classes'] = {pay['classes']}   → 사이드카 컬럼 순서와 일치해야 함", flush=True)

    series = {"balancedish 라벨(원본)": y_bal}
    for name, _, pref, _, k in ARMS:
        p = df[[f"{pref}{c}_prob" for c in CLASSES]].to_numpy(np.float64).argmax(1)
        series[f"{name} 예측 K=0"] = p
        series[f"{name} 예측 K={k}"] = _debounce(p, k)

    rep = {"classes_check": {}, "agreement": {}, "lead_lag": {}}
    for name, _, _, mp, _ in ARMS:
        pay = joblib.load(mp)
        rep["classes_check"][name] = {"model_classes_": [int(c) for c in pay["model"].classes_],
                                      "payload_classes": pay["classes"]}

    print("\n" + "=" * 90)
    print("[A-2] 태그 부호 검증 — bull 은 직전수익이 +, bear 는 − 여야 정상 매핑")
    print(f"\n{'시리즈':26s}{'bull직전+':>11s}{'bear직전−':>11s}{'bull이후+':>11s}{'bear이후−':>11s}"
          f"{'bull직전bp':>12s}{'bear직전bp':>12s}", flush=True)
    for nm, pr in series.items():
        a = past_future_agreement(pr, close, 48)
        rep["agreement"][nm] = a
        print(f"{nm:26s}{a['bull_past_agree']*100:10.1f}%{a['bear_past_agree']*100:10.1f}%"
              f"{a['bull_fut_agree']*100:10.1f}%{a['bear_fut_agree']*100:10.1f}%"
              f"{a['bull_past_mean_bp']:12.1f}{a['bear_past_mean_bp']:12.1f}", flush=True)

    print("\n" + "=" * 90)
    print("[B] 리드-랙 상관 — 방향신호 vs 이후 h봉 수익 (h<0 = 과거를 설명 = 후행)")
    print(f"\n{'시리즈':26s}{'최대상관 h':>12s}{'그때 상관':>11s}{'h=0 상관':>11s}", flush=True)
    for nm, pr in series.items():
        ll = lead_lag_peak(pr, close)
        rep["lead_lag"][nm] = ll
        print(f"{nm:26s}{ll['peak_lag_bars']:10d}봉{ll['peak_corr']:11.4f}{ll['corr_at_0']:11.4f}",
              flush=True)

    (OUT / "regime_lag_and_mapping_diagnosis.json").write_text(
        json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n산출물: {OUT}/regime_lag_and_mapping_diagnosis.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
