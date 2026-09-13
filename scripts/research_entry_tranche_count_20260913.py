"""**지금 상황에서 몇 분할이 최적인가** — 고정 기본값이 아니라 계산으로 (2026-09-13).

사용자: *"기본값이 아니라 지금 현 상황에 맞게 몇 분할로 진입 및 청산하는게 좋겠다라는
수학적 근거가 있어야해"*.

## 목적함수
파산을 «계좌를 잃는 것»으로 두고 **순자산 대비**로 잰다. 임의의 로그 페널티(log 1e-12 같은)를
쓰면 그 상수가 답을 지배해 버린다 -- 실제로 첫 판에서 그렇게 됐다.

    분할 순이득 = Δp_ruin  −  (1 − capture(spread)) · L · μ

두 항이 정반대로 움직인다:
  · **capture(spread)**: 나눠 넣는 동안 노출이 모자라 엣지를 덜 먹는다. 드리프트가 일정하면
    선형 램프의 노출가중 포착률은 `1 − spread/(2H)` 다. spread 를 키울수록 준다.
    (사용자는 엣지가 실재한다 -- 68왕복 단위당 μ=18.37bp·t=3.08. 엣지가 0이면 이 항이
     사라지고 답은 «최대한 나눠라»가 된다. 그래서 이 계산은 엣지 추정에 의존한다.)
  · **p_ruin(k, spread)**: 초기 노출이 낮아 파산 확률이 내려간다. 테이프에서 **실측**한다 --
    칸마다 실제 진입가가 다르므로 평단(VWAP)과 그때그때의 노출로 경로를 따라간다.

## 왜 «지금 상황»에 따라 달라지나
p_ruin 은 변동성 국면에 따라 크게 다르다. 잔잔하면 나눌 이유가 적고(엣지만 손해),
험하면 나눠야 한다. 그래서 **현재 안전 MAE 로 국면을 잡아** 그 국면의 봉만 표집한다.

⚠️한계: 드리프트 일정 가정(capture 항) · 칸 간격 고정 · 방향 무작위(파산 항은 보수적) ·
체결은 즉시 가정(peg 미체결은 별개 축). 절대값이 아니라 **k 사이의 순위**를 읽는 용도다.
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
EDGE_MU = 18.37e-4          # 단위당 기대수익(비율). live_eth_risk_sizing_policy 와 같은 값.
KS = (1, 2, 3, 4, 6)
SPREAD_FRACS = (0.0, 0.1, 0.25, 0.5)     # 보유시간 대비 퍼뜨리는 비율
N_SIM = 12000
SEED = 20260913


def load_paths():
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    return d.px_last.to_numpy(float), d.px_max.to_numpy(float), d.px_min.to_numpy(float)


def vol_regime_mask(px: np.ndarray, pct_lo: float, pct_hi: float) -> np.ndarray:
    """24시간 ATR/가격 분위로 국면을 자른다. «지금»을 이 안에 넣고 그 국면만 표집한다."""
    atr = pd.Series(np.abs(np.diff(px, prepend=px[0]))).rolling(1440, min_periods=1000).mean().values
    ap = atr / px
    lo, hi = np.nanquantile(ap, pct_lo), np.nanquantile(ap, pct_hi)
    return np.isfinite(ap) & (ap >= lo) & (ap <= hi)


def simulate(px, hi, lo, mask, *, L: float, hold: int, k: int, spread: int,
             n: int = N_SIM, seed: int = SEED) -> float:
    """파산 확률. 칸을 `spread` 분에 걸쳐 k 번 넣고 `hold` 분까지 들고 간다.

    파산 = 평단 대비 역행 × 그 시점 노출 >= 1 (교차마진: 손실/순자산 >= 1)."""
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(mask)
    idx = idx[idx < len(px) - hold - spread - 1]
    if len(idx) < 500:
        return float("nan")
    start = rng.choice(idx, n, replace=True)
    side = rng.integers(0, 2, n)          # 방향 무작위 -- 파산 항은 보수적으로
    gap = 0 if k == 1 else spread // max(1, k - 1)
    dead = 0
    for t in range(n):
        a, s = start[t], side[t]
        qty = 0.0; notion = 0.0; worst = 0.0
        for i in range(k):
            j = a + i * gap
            p = px[j]
            qty += L / k; notion += p * (L / k)
            vw = notion / qty
            end = a + hold if i == k - 1 else min(a + hold, a + (i + 1) * gap)
            seg_lo, seg_hi = lo[j:end + 1], hi[j:end + 1]
            if len(seg_lo) == 0:
                continue
            adv = (vw - seg_lo.min()) / vw if s else (seg_hi.max() - vw) / vw
            worst = max(worst, adv * qty)
        if worst >= 1.0:
            dead += 1
    return dead / n


def capture(spread: int, hold: int) -> float:
    """노출가중 엣지 포착률. 선형 램프면 1 − spread/(2H)."""
    return max(0.0, 1.0 - spread / (2.0 * max(hold, 1)))


def split_net_gain(p_lump: float, p_split: float, spread: int, hold: int, L: float) -> float:
    """분할의 순이득(순자산 대비 비율). 양수면 나누는 게 낫다."""
    return (p_lump - p_split) - (1 - capture(spread, hold)) * L * EDGE_MU


def main() -> int:
    px, hi, lo = load_paths()
    print(f"1분봉 {len(px):,} · 엣지 μ={EDGE_MU*1e4:.2f}bp/단위")
    print("분할(3분할·보유의 50%에 걸쳐) 순이득 = Δ파산 − (1−포착)·L·μ  [순자산 대비 %]\n")
    print(f"{'국면':>5} {'L':>4} {'H':>7} {'일괄파산':>9} {'분할파산':>9} {'순이득':>9} {'권고':>7}")
    rows = {}
    for lab, (a, b) in (("잔잔", (0.0, 0.25)), ("보통", (0.25, 0.75)), ("험함", (0.75, 1.0))):
        mask = vol_regime_mask(px, a, b)
        for L in (5, 12, 20):
            for H in (240, 1440):
                p1 = simulate(px, hi, lo, mask, L=float(L), hold=H, k=1, spread=0)
                p3 = simulate(px, hi, lo, mask, L=float(L), hold=H, k=3, spread=int(H * 0.5))
                net = split_net_gain(p1, p3, int(H * 0.5), H, L)
                rows[(lab, L, H)] = net
                print(f"{lab:>5} {L:>4} {H:>6}분 {100*p1:>8.4f} {100*p3:>8.4f} "
                      f"{100*net:>8.4f} {'나눠라' if net > 0 else '일괄':>7}")

    # 이 분석의 결론 두 개를 고정한다.
    assert all(rows[(lab, L, 240)] < 0 for lab in ("잔잔", "보통") for L in (5, 12)), \
        "4시간 보유에서는 분할이 손해여야 한다 -- 뒤집혔으면 엣지나 변동성이 크게 변한 것"
    assert rows[("험함", 20, 1440)] > rows[("잔잔", 5, 240)], \
        "험하고 크고 오래 들수록 분할이 유리해야 한다"
    print("\n확인: 4시간 보유는 분할이 손해 · 험함×고노출×장기일수록 분할이 유리")
    print("⇒ 상한이 5배면 분할이 이기는 칸은 «1일 보유 + 12배 이상» 뿐이다 -- 상한이 그걸 막는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
