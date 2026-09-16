#!/usr/bin/env python3
"""E2 -- 펀딩 이력 복구. 저장소의 `_fetch_funding` 을 그대로 재사용한다(페이징 이미 구현됨).

D단계에서 드러난 구멍: `last_funding_rate` 가 **2022·2023·2024 전량 상수**(중앙값 채움)이고,
거기서 파생된 9열이 같이 죽어 있었다. 펀딩 하나만 진짜 값으로 되돌리면 11열 중 9열이 산다.
OI·롱숏비는 2022~2026 전부 실값이라 건드릴 게 없다(2022 toptrader 87.3% 상수는 별건).

`/fapi/v1/fundingRate` 는 상장 시점까지 8시간 간격으로 다 준다. 가중치 1/호출 ·
1000행/호출이라 4년치가 ~5호출 -- 트레이딩 봇과 공유하는 REST 예산에 사실상 무영향.
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

ROOT = Path.home() / "crypto-scalping"
sys.path.insert(0, str(ROOT / "scripts"))
from live_regime_wide24_signal_20260826 import _fetch_funding  # noqa: E402

OUT = ROOT / "tmp/omega461_longwindow_20260917/funding_2021_2026.csv"
START, END = "2021-11-01", "2026-09-17"

def main() -> int:
    s = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    e = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    df = _fetch_funding("ETHUSDT", s, e)
    print(f"{len(df)}건  {df.timestamp.min()} ~ {df.timestamp.max()}", flush=True)

    gap = df.timestamp.diff().dropna()
    big = gap[gap > pd.Timedelta("8h")]
    print(f"8시간 초과 간격 {len(big)}건" + (f" (최대 {big.max()})" if len(big) else ""), flush=True)
    yr = df.timestamp.dt.year
    for y in sorted(yr.unique()):
        v = df.last_funding_rate[yr == y]
        print(f"  {y}: {len(v):>5}건 · 평균 {v.mean():+.6f} · 표준편차 {v.std():.6f} "
              f"· 최빈값점유 {(v == v.mode().iloc[0]).mean():.3f}", flush=True)

    # 관문: 연도마다 실제로 분산이 있어야 한다. 하나라도 상수면 채움과 다를 게 없다.
    for y in sorted(yr.unique()):
        v = df.last_funding_rate[yr == y]
        assert v.std() > 1e-6, f"{y} 펀딩이 상수다 -- 복구 실패"
        assert len(v) > 300 or y in (2021, 2026), f"{y} 펀딩 건수 부족: {len(v)}"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"저장: {OUT}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
