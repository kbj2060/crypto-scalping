"""진입 사이징을 **습관 앵커가 아니라 파산제약**으로 다시 세운다 (2026-09-13, 사용자 요청).

사용자: *"앵커나 내 습관으로 만들지 말고 어떤 전략이 제일 효과적인지 분석해서 모델을 만들어.
분할 매수 계획과 매수 비율 등 논문과 외부 문헌을 참고해서라도. 켈리 공식 같은 것도 참고해"*

## 결론 한 줄
**켈리는 구속조건이 아니다. 파산이 구속조건이고, 파산을 정하는 건 레버리지가 아니라 보유시간이다.**

## 왜 켈리가 안 쓰이나
실계좌 68왕복의 단위당 수익은 μ=18.37bp · σ=49.13bp · t=3.08 로 **엣지가 실재한다**
(실제 손익 t 는 0.43 -- 크기가 먹는다). 켈리 f* = μ/σ² = **76배**.
추정오차 축소(1-1/t² = 0.89)를 먹여도 68배, 관행적 half-Kelly 까지 가도 34배다.
그런데 76배는 청산거리 1.31% 로 **1시간 MAE 중앙값(0.59%)보다도 작다** -- 즉시 청산이다.
켈리는 파산 장벽이 없는 모형이라 여기서는 상한으로 기능하지 못한다.
문헌도 같은 말을 한다: drawdown/ruin 제약을 넣으면 최적 비율이 크게 줄어든다
(MacLean·Thorp·Ziemba, *Capital Growth with Drawdown Constraints*, 2011;
estimation risk 하의 fractional Kelly 는 2014~2026 다수).

## 무엇이 실제로 크기를 정하나
55왕복의 **실제 MAE** 는 중앙 0.27% · 95% 2.68% · 최대 3.31% 로 매우 작다. 그대로 믿으면
30배까지 «청산 0건»이 나온다. 🔴그러나 그 55건은 **2026-08~09**, 30개월 중 MAE 중앙이
**가장 낮았던 달**이다(0.21% vs 전체 0.34%). 같은 보유시간 분포를 30개월에 적용하면
30배의 «55건 중 최소 1회 청산»은 **96.2%** 다. 표본내 30배는 잔잔한 한 달의 운이었다.

## ⭐지렛대 순위 (55건 중 «최소 1회 청산» ≤ 5% 를 만족하는 최대 레버리지)
    보유상한 없음 ->  3.7배 | 1일 -> 6.2배 | 8시간 -> 9.7배 | 4시간 -> 12.6배 | 1시간 -> 18.6배
보유시간 상한이 레버리지 상한보다 훨씬 강하다. 중앙 보유(1.12h)는 안 변하고 **꼬리만 잘린다** --
그 꼬리(1일 초과 7.4%)가 위험의 거의 전부다. Moreira-Muir 의 변동성 타게팅이 노리는 것과
같은 구조지만, 여기서 조절하는 축은 변동성이 아니라 **노출 시간**이다.

## 분할은 부수적이다
4시간 상한·최종 12.6배에서 일괄 5.2% -> 3분할/30분 4.5%. 20배에서는 23.0% -> 17.4%.
도움은 되지만 보유상한(3.7->12.6배)에 비하면 2차 효과다. 2026-09-06 물타기 감사에서
«부분 투입만 살아남았다»던 것과 같은 크기의 효과다.

⚠️한계: 무작위 방향(진입 실력 미반영, 보수적) · 독립 가정(군집 있으면 과대) ·
MAE 는 1분봉 고가/저가(그 안의 경로는 모름) · 테이프 2026-09-05 까지라 최근 13건 제외.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
TAPE = ROOT / "data/research/eth_tape_1m_20260906.parquet"
TRIPS = ROOT / "data/live/account_round_trips.jsonl"
EQUITY = 1089.45
CAPS = [(None, "없음"), (1440, "1일"), (480, "8시간"), (240, "4시간"), (60, "1시간")]
SEED = 20260913


def load():
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    rows = [json.loads(l) for l in TRIPS.read_text().splitlines() if l.strip()]
    tr = [x for x in rows if x.get("closed") and x.get("side") in ("LONG", "SHORT")]
    holds = np.array([(x["exit_time"] - x["entry_time"]) / 60000.0 for x in tr])
    ret = np.array([x["net_pnl"] / (x["max_qty"] * x["entry_price"]) for x in tr])
    return (d.px_last.to_numpy(float), d.px_max.to_numpy(float), d.px_min.to_numpy(float),
            holds, ret)


def mae_sample(px, hi, lo, holds, hcap, rng, n=60000):
    """보유상한 hcap 을 걸었을 때의 최대 역행폭(%) 표본. 방향은 무작위(보수적)."""
    m = len(px)
    h = np.minimum(holds, hcap) if hcap else holds
    start = rng.integers(0, m - 1, n)
    dur = np.maximum(1, rng.choice(h, n, replace=True).astype(int))
    side = rng.integers(0, 2, n)
    out = np.empty(n)
    for k in range(n):
        a = start[k]; b = min(m, a + dur[k]); e = px[a]
        out[k] = (e - lo[a:b].min()) / e if side[k] else (hi[a:b].max() - e) / e
    return np.maximum(out, 0) * 100


def max_leverage(mae: np.ndarray, trades: int, target: float) -> float:
    """«trades 건 중 최소 1회 청산» 확률을 target 이하로 두는 최대 레버리지."""
    per = 1 - (1 - target) ** (1 / trades)
    return 100.0 / np.quantile(mae, 1 - per)


def main() -> int:
    px, hi, lo, holds, ret = load()
    rng = np.random.default_rng(SEED)
    mu, sd = ret.mean(), ret.std(ddof=1)
    t = mu / (sd / math.sqrt(len(ret)))
    kelly = mu / sd ** 2
    shrink = max(0.0, 1 - 1 / t ** 2)
    print(f"엣지 μ={mu*1e4:.2f}bp σ={sd*1e4:.2f}bp n={len(ret)} t={t:.2f}")
    print(f"켈리 {kelly:.1f}배 · 추정오차축소 {shrink:.2f} -> {kelly*shrink:.1f}배 "
          f"· half-Kelly {kelly*shrink/2:.1f}배  (전부 파산제약보다 크다)\n")
    print(f"{'보유상한':>9} {'MAE95':>7} {'MAE99.9':>8} {'L(5%)':>7} {'명목(5%)':>11}")
    table = {}
    for cap, lab in CAPS:
        m = mae_sample(px, hi, lo, holds, cap, rng)
        L = max_leverage(m, len(ret), 0.05)
        table[lab] = L
        print(f"{lab:>9} {np.quantile(m,.95):>6.2f}% {np.quantile(m,.999):>7.2f}% "
              f"{L:>6.1f}배 {EQUITY*L:>10,.0f}")
    # 이 분석의 핵심 주장을 고정한다 -- 보유상한이 레버리지 여력을 단조 증가시킨다.
    order = [table[lab] for _, lab in CAPS]
    assert all(a <= b for a, b in zip(order, order[1:])), \
        f"보유상한을 조였는데 허용 레버리지가 안 늘었다 -- 계산이 깨졌다: {order}"
    assert kelly > table["없음"] * 5, \
        "켈리가 파산제약과 비슷해졌다 -- 엣지 추정이 바뀌었으니 결론을 다시 볼 것"
    print("\n확인: 보유상한이 조일수록 허용 레버리지 단조 증가 · 켈리는 파산제약보다 훨씬 큼")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
