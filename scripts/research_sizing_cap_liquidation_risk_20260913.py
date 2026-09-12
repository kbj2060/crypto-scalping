"""순자산 대비 상한 배수를 «실제 보유시간에 대입한 청산 확률»로 고른다 (2026-09-13).

왜 다시 쟀나: 기존 12.5배(=청산거리 8%)는 역행폭 **95분위**로 골랐는데 그게 틀린 잣대다.
청산은 꼬리 사건이라 95분위가 아니라 꼬리를 봐야 하고, 보유시간을 «24시간»으로 가정했지
실제 분포를 보지 않았다(실측: 중앙 1.12시간인데 **최대 9일**, 1일 초과 7.4%).

방법: ETH 1분봉 869일에서 보유창 H 의 최대 역행폭(MAE)을 롱/숏 양쪽으로 구하고,
원장 68왕복의 **실제 보유시간**을 각각 가장 가까운 H 에 매핑해 건당 청산확률을 낸 뒤
독립 가정으로 «최소 1회 이상» 확률을 합친다.

⚠️저변동 구간 조건부를 **반드시 같이** 본다. 역변동성 사이징은 변동성이 낮을 때만 상한까지
가므로 전 구간 평균으로 재면 과대평가다. 그런데 조건부로 재도 8% 는 49.5% 로 거의 안
내려간다 -- 저변동 구간의 1일 MAE 는 중앙이 낮지만(2.79% vs 3.43%) **99분위가 더
높기 때문이다(21.82% vs 16.10%)**. 변동성 압축이 확장에 선행한다.

무작위 진입 기준이다. 이 저장소의 반복 결론이 «방향 실력 ≈ 0» 이므로 그 기준이 맞다.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

TAPE = pathlib.Path("data/research/eth_tape_1m_20260906.parquet")
TRIPS = pathlib.Path("data/live/account_round_trips.jsonl")
HORIZONS = [15, 30, 60, 120, 240, 480, 960, 1440, 2880, 5760, 11520, 23040]
THRESHOLDS = [4, 6, 8, 10, 15, 20]
ATR_BARS = 1440          # 24시간. 사이징 워커(5분봉 288)와 같은 창을 1분봉으로 환산.
LOW_VOL_Q = 0.0568       # 상한이 실제로 묶이는 구간 = atr_pct 하위 5.68%


def holding_minutes() -> np.ndarray:
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    return np.array(sorted(
        (r["exit_time"] - r["entry_time"]) / 60000.0 for r in rows
        if r.get("entry_time") and r.get("exit_time") and r["exit_time"] >= r["entry_time"]))


def mae_tables(px, hi, lo, low_mask):
    """보유창별 «역행폭 >= t» 비율. 전 구간과 저변동 구간을 따로 낸다."""
    rev = lambda a, w, f: getattr(pd.Series(a).iloc[::-1].rolling(w, min_periods=w), f)().iloc[::-1].shift(-1).values
    allt, lowt = {}, {}
    for h in HORIZONS:
        dl = (px - rev(lo, h, "min")) / px * 100
        ds = (rev(hi, h, "max") - px) / px * 100
        ok = np.isfinite(dl) & np.isfinite(ds)
        allt[h] = {t: 0.5 * ((dl[ok] >= t).mean() + (ds[ok] >= t).mean()) for t in THRESHOLDS}
        s = ok & low_mask
        lowt[h] = ({t: 0.5 * ((dl[s] >= t).mean() + (ds[s] >= t).mean()) for t in THRESHOLDS}
                   if s.sum() > 500 else allt[h])
    return allt, lowt


def main() -> int:
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    px, hi, lo = d.px_last.values, d.px_max.values, d.px_min.values
    atr_pct = (pd.Series(np.abs(np.diff(px, prepend=px[0])))
               .rolling(ATR_BARS, min_periods=1000).mean().values) / px
    low = np.isfinite(atr_pct) & (atr_pct <= np.nanquantile(atr_pct, LOW_VOL_Q))
    allt, lowt = mae_tables(px, hi, lo, low)

    holds = holding_minutes()
    near = lambda x: min(HORIZONS, key=lambda h: abs(np.log(h) - np.log(max(x, 1))))
    print(f"1분봉 {len(px):,} · 원장 {len(holds)}왕복 "
          f"(보유 중앙 {np.median(holds)/60:.2f}h · 최대 {holds.max()/1440:.1f}일)")
    print(f"\n{'청산거리':>8} {'레버리지':>9} {'건당':>7} {'최소1회(전구간)':>15} {'최소1회(저변동)':>15}")
    for t in THRESHOLDS:
        pa = np.array([allt[near(h)][t] for h in holds])
        pl = np.array([lowt[near(h)][t] for h in holds])
        print(f"{t:>7}% {100/t:>8.1f}배 {100*pa.mean():>6.2f}% "
              f"{100*(1-np.prod(1-pa)):>14.1f}% {100*(1-np.prod(1-pl)):>14.1f}%")

    # 이 스크립트의 핵심 주장 -- 저변동이 «안전»하지 않다는 것 -- 을 숫자로 남긴다.
    h = 1440
    rev = lambda a, f: getattr(pd.Series(a).iloc[::-1].rolling(h, min_periods=h), f)().iloc[::-1].shift(-1).values
    dl = (px - rev(lo, "min")) / px * 100
    ds = (rev(hi, "max") - px) / px * 100
    ok = np.isfinite(dl) & np.isfinite(ds)
    b = np.maximum(dl, ds)
    print(f"\n1일 MAE  전구간 중앙 {np.median(b[ok]):.2f}% / 99% {np.quantile(b[ok],.99):.2f}%"
          f"   저변동 중앙 {np.median(b[ok&low]):.2f}% / 99% {np.quantile(b[ok&low],.99):.2f}%")
    assert np.quantile(b[ok & low], .99) > np.quantile(b[ok], .99), \
        "저변동 구간의 꼬리가 더 두껍다는 게 이 분석의 핵심 -- 뒤집히면 결론을 다시 봐야 한다"
    print("확인: 저변동 구간의 99분위가 전 구간보다 크다(꼬리가 더 두껍다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
