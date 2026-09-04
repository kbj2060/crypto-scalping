#!/usr/bin/env python3
"""ZDC costgate 최적조합(SL=5.0/ARM=0.1/Trail=0.1, 승률97.6%/98.3%)이 진짜 방향예측 실력인지,
아니면 노이즈만으로도 거의 항상 트리거되는 백테스트 메커니즘 아티팩트인지 판별하는 대조군.

방법: 같은 후보(같은 entry_price/atr/forward OHLC)를 그대로 두고 side(long/short)만 전부
뒤집는다(일부러 틀린 방향). 진짜 방향예측 실력 때문이라면 뒤집었을 때 승률이 폭락해야 한다.
메커니즘 아티팩트(노이즈수확)라면 뒤집어도 승률이 비슷하게 높게 나와야 한다 -- 이 저장소의
비-동어반복 베이스라인 원칙과 동일한 정신(신호 자체가 아니라 채점 메커니즘의 착시인지 분리).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "data/research/eth_v_rebound_multitrigger_zigzag_direction_costgate_20260901/zdc_costgate_candidates.pkl"
STANDARD_COST_BP = 10.0


def simulate_trailing(row: pd.Series, sl_mult: float, arm_mult: float, trail_mult: float,
                       pessimistic: bool) -> float:
    atr = row["atr"]
    entry = row["entry_price"]
    side = row["side"]
    opens, highs, lows, closes = row["fwd_open"], row["fwd_high"], row["fwd_low"], row["fwd_close"]
    sign = 1.0 if side == "long" else -1.0
    stop = entry - sign * sl_mult * atr
    armed = False
    best = entry
    for o, h, l, c in zip(opens, highs, lows, closes):
        fav_extreme = h if side == "long" else l
        adv_extreme = l if side == "long" else h

        def stop_hit() -> bool:
            return (adv_extreme <= stop) if side == "long" else (adv_extreme >= stop)

        def update_trailing() -> None:
            nonlocal armed, stop, best
            if sign * (fav_extreme - best) > 0:
                best = fav_extreme
            if not armed and sign * (best - entry) >= arm_mult * atr:
                armed = True
            if armed:
                new_stop = best - sign * trail_mult * atr
                if sign * (new_stop - stop) > 0:
                    stop = new_stop

        if pessimistic:
            if stop_hit():
                return sign * (stop - entry) / entry
            update_trailing()
        else:
            update_trailing()
            if stop_hit():
                return sign * (stop - entry) / entry
    return sign * (closes[-1] - entry) / entry


def report(df: pd.DataFrame, label: str, sl: float, arm: float, trail: float) -> None:
    print(f"\n=== {label} (SL={sl} ARM={arm} Trail={trail}) ===")
    for split_name in ("val", "oos"):
        sub = df[df["split"] == split_name]
        opt = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, False), axis=1)
        pess = sub.apply(lambda r: simulate_trailing(r, sl, arm, trail, True), axis=1)
        opt_bp = float(opt.mean() * 1e4 - STANDARD_COST_BP)
        pess_bp = float(pess.mean() * 1e4 - STANDARD_COST_BP)
        win_rate = float((opt > 0).mean())
        print(f"  {split_name}: n={len(sub)}  opt={opt_bp:+.2f}bp  pess={pess_bp:+.2f}bp  win_rate={win_rate:.1%}")


def main() -> int:
    df = pd.read_pickle(CANDIDATES)
    sl, arm, trail = 5.0, 0.10, 0.10  # the grid's top-sorted config

    report(df, "REAL direction (as predicted)", sl, arm, trail)

    flipped = df.copy()
    flipped["side"] = flipped["side"].map({"long": "short", "short": "long"})
    report(flipped, "FLIPPED direction (deliberately wrong control)", sl, arm, trail)

    # also check a saner, giveback-like ARM to see if the gap between real/flipped opens up there
    sl2, arm2, trail2 = 4.0, 1.5, 0.1
    report(df, "REAL direction, giveback-like ARM=1.5", sl2, arm2, trail2)
    report(flipped, "FLIPPED direction, giveback-like ARM=1.5", sl2, arm2, trail2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
