#!/usr/bin/env python3
"""D단계 재실행 -- 「2024 에서 상수인 열」을 전 판에서 제거하고 사다리를 다시 올린다.

C1 에서 드러난 문제: A단계 중앙값 채움이 **2024 에 100% 몰려 있다**(2025/2026 은 0%).
즉 m21 이 더 받은 데이터는 「깊은 학습 데이터」가 아니라 「펀딩이 상수인 데이터」였다.
그 상태의 m21 열세를 «옛 레짐» 이라 부르면 내 채움을 시장 성질로 오인하는 것이다.

그래서 이름으로 고르지 않고 **실측으로** 고른다: 2024 최빈값 점유율이 높고 2025 에선
낮은 열 = 내가 상수로 만든 열. 세 판 전부에서 같이 빼므로 비교는 공정하다.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_omega461_longwindow_depth_ladder_20260917 as D  # noqa: E402

DEGEN_2024, CLEAN_2025 = 0.50, 0.10

def main() -> int:
    df, base_cols = D.load()
    yr = df.timestamp.dt.year
    m24, m25 = (yr == 2024), (yr == 2025)
    drop = []
    for c in base_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        mo = v[m24].mode()
        if mo.empty:
            continue
        o24 = float((v[m24] == mo.iloc[0]).mean())
        o25 = float((v[m25] == mo.iloc[0]).mean())
        if o24 >= DEGEN_2024 and o25 <= CLEAN_2025:
            drop.append((c, o24, o25))
    print(f"\n=== 2024 에서 상수인 열 {len(drop)}개 (전 판에서 제거) ===", flush=True)
    for c, a, b in drop:
        print(f"  {c:<34} 2024 {a:.3f} · 2025 {b:.3f}", flush=True)
    names = [c for c, _, _ in drop]
    assert names, "제거 대상이 없다 -- C1 과 모순, 멈춘다"

    kept = [c for c in base_cols if c not in names]
    print(f"base_cols {len(base_cols)} -> {len(kept)}", flush=True)
    orig = D.make_x
    D.make_x = lambda frame, _bc, _k=kept, _o=orig: _o(frame, _k)
    D.TAG = "_nofill"
    D.main()
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
