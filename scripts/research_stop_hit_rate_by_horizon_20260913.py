#!/usr/bin/env python3
"""**보유시간별 손절 발동률** 실측 (2026-09-13).

왜: `live_eth_trade_plan_20260913.stop_risk` 가 4시간 실측치 0.043 을 **모든 지평에**
그대로 쓰고 있었다(`hold_min` 인자를 받고도 안 썼다). 24시간 보유의 손절 확률이 4시간과
같을 리 없다 -- 실측하니 1440분은 **0.32** 로 7.4배 차이였다. 위험을 그만큼 과소표시한다.

방법: 869일 1분봉 테이프에서 매 분을 진입점으로 보고, 다음 H 분 안에 **봉내 고가/저가**가
평단 -3%(롱) / +3%(숏)를 지나는 비율. 봉내 기준인 이유는 실제 주문이 STOP_MARKET 이라
닿는 즉시 발동하기 때문이다(종가 기준으로 재면 과소).

겹치는 창을 쓰므로 **비율의 점추정에는 편향이 없고 신뢰구간만 좁게 나온다** -- 여기서 쓰는
값은 점추정뿐이라 문제되지 않는다.
"""
import pathlib
import sys

import numpy as np
import pandas as pd

TAPE = next((p for p in (
    pathlib.Path(__file__).resolve().parents[1] / "data/research/eth_tape_1m_20260906.parquet",
    pathlib.Path("/home/kbj20/crypto-scalping/data/research/eth_tape_1m_20260906.parquet"))
    if p.exists()), None)
HOLDS = (60, 120, 240, 480, 1440)
STOP_PCT = 0.03


def hit_rates(c, hi, lo, hold_min: int, stop_pct: float = STOP_PCT) -> tuple[float, float, int]:
    """(롱 발동률, 숏 발동률, 표본). 진입 봉 **다음** 분부터 H 분을 본다."""
    n, H = len(c), hold_min
    rev_lo = pd.Series(lo[::-1]).rolling(H, min_periods=H).min().to_numpy()[::-1]
    rev_hi = pd.Series(hi[::-1]).rolling(H, min_periods=H).max().to_numpy()[::-1]
    fwd_lo, fwd_hi = np.full(n, np.nan), np.full(n, np.nan)
    fwd_lo[:-H], fwd_hi[:-H] = rev_lo[1:n - H + 1], rev_hi[1:n - H + 1]
    ok = ~np.isnan(fwd_lo)
    return (float((fwd_lo[ok] <= c[ok] * (1 - stop_pct)).mean()),
            float((fwd_hi[ok] >= c[ok] * (1 + stop_pct)).mean()), int(ok.sum()))


def main() -> int:
    if TAPE is None:
        print("테이프 없음 -- 건너뜀", file=sys.stderr)
        return 0
    d = pd.read_parquet(TAPE, columns=["px_last", "px_max", "px_min"])
    c, hi, lo = (d[k].to_numpy(float) for k in ("px_last", "px_max", "px_min"))
    print(f"표본 {len(c):,}분 = {len(c)/1440:.0f}일 · 손절 {STOP_PCT:.0%}")
    print(f"{'H(분)':>7} {'LONG':>8} {'SHORT':>8} {'표본':>12}")
    table = {}
    for H in HOLDS:
        L, S, n = hit_rates(c, hi, lo, H)
        table[H] = (round(L, 4), round(S, 4))
        print(f"{H:>7} {L:>8.4f} {S:>8.4f} {n:>12,}")
    # 단조성: 오래 들수록 손절이 더 자주 걸린다. 깨지면 계산이 틀린 것이다.
    for side in (0, 1):
        vals = [table[H][side] for H in HOLDS]
        assert all(a < b for a, b in zip(vals, vals[1:])), vals
    assert 0.04 < table[240][0] < 0.07, "240분 롱이 기존 4시간 실측 0.043 근방이어야 한다"
    print("\nSTOP_HIT_RATE_BY_HOLD =", {H: table[H] for H in HOLDS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
