#!/usr/bin/env python3
"""D단계 대조군 2개.

C1 데이터품질: A단계 중앙값 채움(펀딩 65.6% 등)이 **연도별로 쏠려 있는가**.
   2024 에 몰려 있다면 m21 의 열세는 「옛 레짐」이 아니라 내가 만든 결손일 수 있다.
C2 노후 대조군: 같은 길이(9개월)의 **옛 창**(2024-01~09)을 같은 VAL 로 채점한다.
   깊이가 아니라 「데이터가 오래됐다」가 원인인지 가른다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_omega461_longwindow_depth_ladder_20260917 as D  # noqa: E402

def main() -> int:
    df, base_cols = D.load()
    print("\n=== C1 중앙값 채움의 연도 분포 (연속 중복값 = 채움 흔적) ===", flush=True)
    yr = df.timestamp.dt.year
    for c in ["last_funding_rate", "funding_abs", "funding_pressure",
              "sum_toptrader_long_short_ratio", "count_long_short_ratio"]:
        if c not in df.columns:
            continue
        v = pd.to_numeric(df[c], errors="coerce")
        mode = v.mode()
        if mode.empty:
            continue
        hit = (v == mode.iloc[0])
        print(f"  {c:<34} 최빈값점유 전체 {hit.mean():.3f} | " +
              " ".join(f"{y}:{hit[yr == y].mean():.3f}" for y in sorted(yr.unique())), flush=True)

    print("\n=== C2 같은 길이·옛 창 (2024-01-01~2024-09-30) ===", flush=True)
    D.TAG = "_control_old09"
    D.ARMS.clear()
    D.ARMS["old09"] = ("2024-01-01", "2024-09-30")
    D.main()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
