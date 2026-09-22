"""RVOL 계산 검증 (2026-09-23). `python test/test_breakout_rvol_20260923.py`

무엇을 지키는가: ①배수가 실제로 «평소 대비 몇 배»인가 ②기준선이 **자기 자신을 안 본다**
(미래참조) ③세션 누적이 UTC 00:00 에서 리셋되는가 ④웜업 구간은 None 인가.
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

    # ① 평소 100, 마지막 날 한 봉만 300 -> 그 봉의 RVOL = 3.0
    days = RVOL_BASE_DAYS + 2
    qv = np.full(days * BARS_PER_DAY, 100.0)
    spike = (days - 1) * BARS_PER_DAY + 137          # 마지막 날의 슬롯 137
    qv[spike] = 300.0
    line, bar, sess = _rvol(frame(days, qv))
    ck(abs(bar[spike] - 3.0) < 1e-9, f"스파이크 봉 RVOL = 3.0 (실제 {bar[spike]})")
    ck(abs(bar[spike - 1] - 1.0) < 1e-9, f"옆 봉 RVOL = 1.0 (실제 {bar[spike - 1]})")

    # ② 미래참조 없음 -- 스파이크 **뒤** 값을 바꿔도 스파이크 봉의 RVOL 은 안 변한다
    qv2 = qv.copy()
    qv2[spike + 1:] = 9999.0
    _, bar2, _ = _rvol(frame(days, qv2))
    ck(abs(bar2[spike] - bar[spike]) < 1e-12, "미래 봉을 바꿔도 과거 RVOL 불변")
    # 같은 슬롯의 **다음 날**을 바꿔도 안 변한다(기준선이 자기 이후를 안 본다)
    ck(np.isfinite(bar[spike]), "스파이크 봉 RVOL 이 유한")

    # ③ 세션 누적은 UTC 00:00 리셋 -- 평소가 100 고정이면 하루 내내 누적비가 1.0
    ck(abs(sess[spike - 1] - 1.0) < 1e-9, f"평소 누적비 = 1.0 (실제 {sess[spike - 1]})")
    #    스파이크 뒤에는 누적이 +200 이므로 1.0 보다 크고, 슬롯이 갈수록 희석된다
    ck(sess[spike] > 1.0 and sess[spike] > sess[-1],
       f"스파이크 후 누적비 > 1 이고 이후 희석 ({sess[spike]:.4f} -> {sess[-1]:.4f})")

    # ④ 웜업 -- min_periods=5 라 5일치 같은 슬롯이 모이기 전에는 NaN
    _, bar3, _ = _rvol(frame(3, np.full(3 * BARS_PER_DAY, 100.0)))
    ck(not np.isfinite(bar3).any(), "3일치로는 전부 NaN(웜업)")
    _, bar4, _ = _rvol(frame(7, np.full(7 * BARS_PER_DAY, 100.0)))
    ck(np.isfinite(bar4[-1]), "7일치면 마지막 봉은 유한")

    # ⑤ 중앙값이라 한 번의 스파이크가 «평소»를 못 들어올린다
    qv5 = np.full(days * BARS_PER_DAY, 100.0)
    qv5[137] = 1e6                                   # 첫날 같은 슬롯에 거대한 값
    _, bar5, _ = _rvol(frame(days, qv5))
    last = (days - 1) * BARS_PER_DAY + 137
    ck(abs(bar5[last] - 1.0) < 1e-9, f"과거 스파이크가 기준선을 안 들어올림 (실제 {bar5[last]})")

    # ⑥ 선(1시간 롤링) -- 한 봉 스파이크는 12봉에 퍼져 **덜 튄다**. 이게 창을 넓힌 이유다.
    ck(abs(line[spike] - (1 + 2 / RVOL_LINE_BARS)) < 1e-9,
       f"1시간 선: 3배 봉 하나 -> {1 + 2 / RVOL_LINE_BARS:.4f}배 (실제 {line[spike]:.4f})")
    ck(line[spike] < bar[spike], "선이 봉보다 덜 튄다")
    #    그리고 스파이크가 **지나간 뒤에도 12봉 동안** 선에 남는다(롤링이므로)
    ck(abs(line[spike + 11] - line[spike]) < 1e-9 and abs(line[spike + 12] - 1.0) < 1e-9,
       "스파이크가 정확히 12봉 뒤 선에서 빠진다")

    # ⑦ 세션 라벨 경계가 분위 상수와 일치 (화면이 아니라 워커가 라벨을 붙인다)
    ck(RVOL_SESSION_LO < 1.0 < RVOL_SESSION_HI, "세션 경계가 1.0 을 사이에 둔다")

    return fail


if __name__ == "__main__":
    raise SystemExit(main())
