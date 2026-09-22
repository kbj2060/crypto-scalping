"""RVOL 계산 검증 (2026-09-23). `python test/test_breakout_rvol_20260923.py`

무엇을 지키는가: ①선(1시간 롤링)이 «평소 대비 몇 배»인가 ②기준선이 **자기 자신을 안 본다**
(미래참조) ③세션 누적이 UTC 00:00 에서 리셋되는가 ④웜업 구간은 None 인가
⑤중앙값이라 과거 스파이크 하나가 «평소»를 못 들어올리는가.

🔴2026-09-23 5분 계열(`bar5`)은 제거됐다 -- 유일한 쓸모였던 «흡수» 판독이 측정에서 무너졌다
  (같은 거래량 수준에서 델타 低/高의 앞 30분 되돌림이 50.3~51.6% 대 52.4~55.3% 로 둘 다 동전).
  그래서 여기서도 5분 계열을 검사하지 않는다 -- 없는 것을 검사하면 테스트가 먼저 죽는다.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from live_eth_breakout_detector_20260911 import (  # noqa: E402
    RVOL_BASE_DAYS, RVOL_LINE_BARS, RVOL_SESSION_HI, RVOL_SESSION_LO, _rvol)

BARS_PER_DAY = 288


def frame(days: int, qv) -> pd.DataFrame:
    n = days * BARS_PER_DAY
    ts = pd.date_range("2026-01-01", periods=n, freq="5min")
    return pd.DataFrame({"timestamp": ts, "qv": np.asarray(qv, float)})


def main() -> int:
    fail = 0

    def ck(cond, what):
        nonlocal fail
        print(("  ok " if cond else "🔴 ") + what)
        if not cond:
            fail += 1

    days = RVOL_BASE_DAYS + 2
    qv = np.full(days * BARS_PER_DAY, 100.0)
    spike = (days - 1) * BARS_PER_DAY + 137          # 마지막 날의 슬롯 137
    qv[spike] = 300.0
    line, sess = _rvol(frame(days, qv))

    # ① 3배 봉 하나는 1시간 창(12봉)에 퍼진다 -> 1 + 2/12
    want = 1 + 2 / RVOL_LINE_BARS
    ck(abs(line[spike] - want) < 1e-9, f"3배 봉 하나 -> 선 {want:.4f}배 (실제 {line[spike]:.4f})")
    ck(abs(line[spike - 1] - 1.0) < 1e-9, f"스파이크 직전은 1.0 (실제 {line[spike - 1]:.4f})")
    #    그리고 정확히 12봉 뒤 창에서 빠진다
    ck(abs(line[spike + RVOL_LINE_BARS - 1] - want) < 1e-9
       and abs(line[spike + RVOL_LINE_BARS] - 1.0) < 1e-9,
       f"스파이크가 정확히 {RVOL_LINE_BARS}봉 뒤 선에서 빠진다")

    # ② 미래참조 없음 -- 스파이크 **뒤** 값을 바꿔도 그 시점 값은 안 변한다
    qv2 = qv.copy()
    qv2[spike + RVOL_LINE_BARS:] = 9999.0
    line2, _ = _rvol(frame(days, qv2))
    ck(abs(line2[spike] - line[spike]) < 1e-12, "미래 봉을 바꿔도 과거 값 불변")

    # ③ 세션 누적은 UTC 00:00 리셋
    ck(abs(sess[spike - 1] - 1.0) < 1e-9, f"평소 누적비 = 1.0 (실제 {sess[spike - 1]:.4f})")
    ck(sess[spike] > 1.0 and sess[spike] > sess[-1],
       f"스파이크 후 누적비 > 1 이고 이후 희석 ({sess[spike]:.4f} -> {sess[-1]:.4f})")

    # ④ 웜업 -- min_periods=5 라 5일치 같은 슬롯이 모이기 전에는 NaN
    l3, _ = _rvol(frame(3, np.full(3 * BARS_PER_DAY, 100.0)))
    ck(not np.isfinite(l3).any(), "3일치로는 전부 NaN(웜업)")
    l7, _ = _rvol(frame(7, np.full(7 * BARS_PER_DAY, 100.0)))
    ck(np.isfinite(l7[-1]), "7일치면 마지막 봉은 유한")

    # ⑤ 중앙값이라 과거 스파이크 하나가 «평소»를 못 들어올린다
    qv5 = np.full(days * BARS_PER_DAY, 100.0)
    qv5[137] = 1e6                                   # 첫날 같은 슬롯에 거대한 값
    l5, _ = _rvol(frame(days, qv5))
    last = (days - 1) * BARS_PER_DAY + 137
    ck(abs(l5[last] - 1.0) < 1e-9, f"과거 스파이크가 기준선을 안 들어올림 (실제 {l5[last]:.4f})")

    # ⑥ 세션 라벨 경계는 분위 상수이고 1.0 을 사이에 둔다
    ck(RVOL_SESSION_LO < 1.0 < RVOL_SESSION_HI, "세션 경계가 1.0 을 사이에 둔다")

    # ⑦ 🔴5분 계열이 되살아나면 알린다 -- 되살리려면 흡수를 먼저 다시 재야 한다
    ck(len(_rvol(frame(days, qv))) == 2, "_rvol 은 (선, 세션) 둘만 돌려준다")

    return fail


if __name__ == "__main__":
    raise SystemExit(main())
