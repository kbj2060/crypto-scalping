"""순자산 대비 상한 배수를 «실제 보유시간에 대입한 청산 확률»로 고른다 (2026-09-13).

왜 쟀나: 기존 12.5배(=청산거리 8%)는 역행폭 **95분위**로 골랐는데 그게 틀린 잣대다.
청산은 꼬리 사건이라 95분위가 아니라 꼬리를 봐야 하고, 보유시간을 «24시간»으로 가정했지
실제 분포를 보지 않았다(실측: 중앙 1.12시간인데 **최대 9일**, 1일 초과 7.4%).

🔴같은 날 초판에서 **두 가지를 틀렸다**. 재현자가 다시 밟지 않도록 남긴다:
  ① 저변동을 **후행 `atr_pct`** 로 정의했다. 배포 경로는 `live_eth_sizing_vol_model_20260912`
     의 **전방 변동성 예측**이다. 후행으로 재면 «저변동일수록 꼬리가 두껍다»는 결과가 나오는데
     (1일 MAE 99분위 21.66% vs 전구간 15.62%) 그건 **후행 변동성의 성질**이지 모델의 성질이
     아니다 -- 모델 기준 저변동에서는 99분위가 7.83% 로 오히려 얇다. 모델이 고치려던 게 그거다.
  ② 조건을 «예측변동성 하위 5.68%» 로 걸었다. 맞는 조건은 **«상한이 실제로 묶이는 상태»**
     (모델 권고 > 상한)다. 안 묶이는 봉은 포지션이 상한보다 작아 청산거리가 더 멀기 때문에
     위험 평가에서 빼야 한다. 6.7배에서 묶임 비율은 55% 로 하위 5.68% 보다 훨씬 넓다.
  ①은 위험을 부풀리고 ②는 줄이는데 ②가 더 커서 **합치면 과소평가**였다(6.7배 19.0% -> 33.0%).

무작위 진입 기준이다. 이 저장소의 반복 결론이 «방향 실력 ≈ 0» 이므로 그 기준이 맞다.
독립 가정으로 «최소 1회»를 합치므로 군집이 있으면 과대평가 쪽이다.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import live_eth_sizing_vol_model_20260912 as svm  # noqa: E402

KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
BARS = [3, 6, 12, 24, 48, 96, 192, 288, 576, 1152, 2304, 4608]   # 5분봉 개수
CAPS = [12.5, 10.0, 8.0, 6.7, 5.0]
BASE_QTY = 2.727          # 워커의 BASE_QTY_DEFAULT 와 같아야 한다
EQUITY = 1089.45          # 실측 순자산. 바뀌면 절대값만 움직이고 순위는 안 변한다.


def predicted_vol(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    ts = pd.to_datetime(df.timestamp, unit="ms", errors="coerce")
    if ts.isna().all():   # timestamp 가 epoch ms 가 아니라 문자열인 판본이 있다
        ts = pd.to_datetime(df.timestamp, errors="coerce")
    assert not ts.isna().all(), "timestamp 파싱 실패 -- 피쳐가 전부 NaN 이 되어 조용히 표본 0 이 된다"
    X = svm.build_features(ts, df.close.to_numpy(float), df.quote_volume.to_numpy(float),
                           df.trades.to_numpy(float), df.high.to_numpy(float),
                           df.low.to_numpy(float))
    art = svm.load_model()
    idx = np.flatnonzero(np.asarray(np.isfinite(X.to_numpy(dtype=float)).all(axis=1)).ravel())
    assert len(idx) > 1000, f"유효 피쳐 행이 {len(idx)} 개뿐 -- 위 timestamp 함정을 의심할 것"
    out = np.full(len(df), np.nan)
    out[idx] = svm.predict_vol(art["models"], X.iloc[idx])
    return out, float(art["ref_pred"])


def holding_minutes() -> np.ndarray:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    return np.array(sorted(
        (r["exit_time"] - r["entry_time"]) / 60000.0 for r in rows
        if r.get("entry_time") and r.get("exit_time") and r["exit_time"] >= r["entry_time"]))


def main() -> int:
    df = pd.read_csv(KLINES)
    pred, ref = predicted_vol(df)
    c, hi, lo = df.close.to_numpy(float), df.high.to_numpy(float), df.low.to_numpy(float)
    qty = BASE_QTY * ref / pred                      # 배포 경로와 같은 식

    mae = {}
    for h in BARS:
        fmin = pd.Series(lo).iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1).values
        fmax = pd.Series(hi).iloc[::-1].rolling(h, min_periods=h).max().iloc[::-1].shift(-1).values
        mae[h] = np.maximum((c - fmin) / c * 100, (fmax - c) / c * 100)

    holds = holding_minutes()
    near = lambda m: min(BARS, key=lambda h: abs(np.log(h * 5) - np.log(max(m, 1))))
    print(f"5분봉 {len(df):,} · 예측 {np.isfinite(pred).sum():,} · 원장 {len(holds)}왕복 "
          f"(보유 중앙 {np.median(holds)/60:.2f}h · 최대 {holds.max()/1440:.1f}일)")
    print(f"\n{'상한':>7} {'청산거리':>8} {'묶임비율':>9} {'건당':>7} {'최소1회':>8}")
    results = {}
    for x in CAPS:
        bind = np.isfinite(qty) & (qty > EQUITY * x / c)
        thr = 100.0 / x
        ps = np.array([(mae[near(m)][bind & np.isfinite(mae[near(m)])] >= thr).mean()
                       for m in holds])
        results[x] = 1 - np.prod(1 - ps)
        print(f"{x:>6}배 {thr:>7.1f}% {100*bind.mean():>8.1f}% "
              f"{100*ps.mean():>6.2f}% {100*results[x]:>7.1f}%")

    # 이 분석이 결정에 쓰이는 방식 자체를 고정한다 -- 상한을 조일수록 위험이 내려가야 한다.
    order = [results[x] for x in CAPS]
    assert all(a >= b for a, b in zip(order, order[1:])), \
        f"상한을 조였는데 위험이 안 내려간다 -- 계산이 깨졌다는 뜻이다: {order}"
    # ①의 함정 재발 방지: 모델 기준 저변동의 꼬리는 후행 atr 기준보다 **얇아야** 한다.
    atr = pd.Series(np.abs(np.diff(c, prepend=c[0]))).rolling(288, min_periods=200).mean().values / c
    m1 = np.isfinite(mae[288])
    lo_m = m1 & np.isfinite(pred) & (pred <= np.nanquantile(pred, 0.0568))
    lo_a = m1 & np.isfinite(atr) & (atr <= np.nanquantile(atr, 0.0568))
    q_m, q_a = np.quantile(mae[288][lo_m], .99), np.quantile(mae[288][lo_a], .99)
    print(f"\n1일 MAE 99분위 — 모델 저변동 {q_m:.2f}% · 후행 atr 저변동 {q_a:.2f}% "
          f"· 전구간 {np.quantile(mae[288][m1], .99):.2f}%")
    assert q_m < q_a, "모델 기준 저변동의 꼬리가 후행 atr 기준보다 두껍다 -- 모델이 퇴화했다는 신호다"
    print("확인: 조임에 따라 위험 단조 감소 · 모델 저변동 꼬리가 후행 atr 보다 얇음")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
