"""V자반등 섀도우 러너의 bars_held 회계 테스트, 2026-09-04.

사용자 신고로 시작: 대시보드에 "3봉"이어야 할 보유가 "1봉"으로 나왔다. 실측하니 원장 17건이
**전부** 어긋나 있었고, 원인이 둘로 갈렸다.

1. 스톱 청산 봉이 안 세졌다 -- 루프가 bars_held를 올리기 전에 break했다. 타임아웃 경로는
   올린 뒤 청산해서 세고 있었으니, 두 청산 경로의 회계가 서로 달랐다. 이게 상수 -1 성분.
2. 신호봉(entry_utc)과 배리어 평가 시작봉 사이가 비어 있었다. SCORE_TAIL_BARS=3이라 신호는
   최대 3봉 묵은 채 처리될 수 있다. 이건 카운터 버그가 아니라 알려진 진입 지연이므로,
   고치지 않고 `entry_bar_utc`/`signal_lag_bars`로 **기록**한다.

그래서 이 테스트가 고정하는 계약은 "bars_held는 평가 시작봉부터 청산봉까지의 개수"다:
    bars_held == (exit_utc - entry_bar_utc) / 5분 + 1
이 검산이 성립하면 bars_held는 원장 안에서 스스로 증명된다.
"""
from __future__ import annotations

import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import live_eth_v_rebound_econ_shadow_runner_20260902 as runner  # noqa: E402

T0 = datetime(2026, 9, 4, 0, 0, tzinfo=timezone.utc)


def bar(i: int, *, high: float, low: float, close: float) -> dict:
    return {"timestamp_utc": (T0 + timedelta(minutes=5 * i)).isoformat(),
            "high": high, "low": low, "close": close, "open": close}


def position(**kw) -> dict:
    base = {"entry_utc": T0.isoformat(), "side": "long", "entry": 100.0, "atr": 1.0,
            "stop": 95.0, "best": 100.0, "armed": False, "bars_held": 0,
            "last_bar_utc": None, "proba": 0.9, "opened_utc": T0.isoformat()}
    base.update(kw)
    return base


def held_from_timestamps(trade: dict) -> int:
    a = datetime.fromisoformat(trade["entry_bar_utc"])
    b = datetime.fromisoformat(trade["exit_utc"])
    return int((b - a).total_seconds() // 300) + 1


class BarsHeldAccountingTests(unittest.TestCase):
    def _run(self, bars: list[dict], **pos_kw) -> dict:
        s = {"positions": [position(**pos_kw)], "ledger": [], "consec_loss": 0}
        runner.manage(s, bars)
        self.assertEqual(len(s["ledger"]), 1, "포지션이 청산되지 않았습니다")
        return s["ledger"][0]

    def test_stop_exit_counts_its_own_bar(self) -> None:
        """이 테스트가 사용자가 신고한 증상을 직접 고정한다 -- 예전에는 1이 나왔다."""
        bars = [bar(0, high=101, low=99, close=100),
                bar(1, high=102, low=99, close=101),
                bar(2, high=101, low=94, close=95)]      # 3번째 봉에서 스톱(95.0)
        t = self._run(bars)
        self.assertEqual(t["reason"], "stop")
        self.assertEqual(t["bars_held"], 3)

    def test_bars_held_reconciles_with_timestamps(self) -> None:
        """원장 안에서 스스로 검산된다: (청산봉 - 평가시작봉)/5분 + 1."""
        bars = [bar(i, high=101, low=99, close=100) for i in range(6)]
        bars.append(bar(6, high=100, low=90, close=94))   # 7번째 봉에서 스톱
        t = self._run(bars)
        self.assertEqual(t["bars_held"], 7)
        self.assertEqual(t["bars_held"], held_from_timestamps(t))

    def test_single_bar_stop_is_one_not_zero(self) -> None:
        bars = [bar(0, high=100, low=90, close=94)]
        t = self._run(bars)
        self.assertEqual(t["bars_held"], 1)
        self.assertEqual(t["bars_held"], held_from_timestamps(t))

    def test_timeout_path_unchanged_and_also_reconciles(self) -> None:
        """타임아웃 경로는 원래 맞았다 -- 고치면서 깨뜨리지 않았는지 확인한다."""
        original = runner.MAX_HOLD_BARS
        runner.MAX_HOLD_BARS = 4
        try:
            bars = [bar(i, high=101, low=99, close=100) for i in range(10)]
            t = self._run(bars)
            self.assertEqual(t["reason"], "timeout")
            self.assertEqual(t["bars_held"], 4)
            self.assertEqual(t["bars_held"], held_from_timestamps(t))
        finally:
            runner.MAX_HOLD_BARS = original

    def test_signal_lag_is_recorded_not_silently_absorbed(self) -> None:
        """신호가 3봉 묵은 뒤 처리된 경우 -- 그 지연이 원장에 남아야 한다.
        bars_held는 여전히 평가 시작봉 기준이고, 검산도 그 기준으로 성립한다."""
        bars = [bar(i, high=101, low=99, close=100) for i in range(6)]
        bars.append(bar(6, high=100, low=90, close=94))
        # 신호는 봉0, 평가는 봉3부터(진입 처리 시점의 마지막 완결봉이 봉2였다)
        t = self._run(bars, entry_utc=T0.isoformat(), last_bar_utc=bars[2]["timestamp_utc"])
        self.assertEqual(t["signal_lag_bars"], 3)
        self.assertEqual(t["entry_bar_utc"], bars[3]["timestamp_utc"])
        self.assertEqual(t["bars_held"], 4)              # 봉3~봉6
        self.assertEqual(t["bars_held"], held_from_timestamps(t))

    def test_no_lag_records_zero_not_none(self) -> None:
        bars = [bar(0, high=100, low=90, close=94)]
        t = self._run(bars)
        self.assertEqual(t["signal_lag_bars"], 0)


if __name__ == "__main__":
    unittest.main()
