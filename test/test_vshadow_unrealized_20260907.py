"""V자반등 섀도우 패널의 보유분 미실현 손익, 2026-09-07.

지키려는 계약은 하나다 -- **미실현 값이 원장 `pnl_bp`와 같은 규약이어야 한다.** 방향 부호와
왕복비용 10bp 차감이 어긋나면, 화면의 "지금 청산 시 +5.5bp"가 실제로 그때 청산됐을 때 원장에
남는 값과 달라진다(그러면 마감분 `total_bp`와 더할 수도 없다).

`dashboard/server.py`는 torch를 끌어오는 모듈들을 import하므로 통째로 import하지 않고 함수
원문만 떼어다 실행한다 -- 그래도 검사 대상은 실제 배포되는 코드 그대로다.
"""
from __future__ import annotations

import re
import unittest
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "dashboard/server.py"


def _load(name: str):
    src = SERVER.read_text()
    m = re.search(rf"^def {re.escape(name)}\(.*?^(?=\S)", src, re.S | re.M)
    if not m:
        raise AssertionError(f"{name}() 를 {SERVER} 에서 찾지 못했다 -- 이름이 바뀌었나?")
    ns: dict[str, Any] = {"Any": Any}
    exec(compile(m.group(0), str(SERVER), "exec"), ns)
    return ns[name]


class UnrealizedBpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.f = staticmethod(_load("_unrealized_bp"))
        cls.locked = staticmethod(_load("_locked_bp"))

    def test_matches_ledger_pnl_convention(self) -> None:
        """실제 원장 행 재현: 숏 2458.00 -> 2467.857142857 이 pnl_bp -50.1 로 기록돼 있다.
        같은 가격을 '지금 시장가'로 넣으면 같은 값이 나와야 한다."""
        got = self.f({"side": "short", "entry": 2458.0}, 2467.8571428571427)
        self.assertAlmostEqual(got, -50.1, places=1)

    def test_long_and_short_signs(self) -> None:
        up = self.f({"side": "long", "entry": 2000.0}, 2020.0)     # +100bp - 10
        dn = self.f({"side": "short", "entry": 2000.0}, 2020.0)    # -100bp - 10
        self.assertAlmostEqual(up, 90.0, places=2)
        self.assertAlmostEqual(dn, -110.0, places=2)

    def test_cost_is_always_subtracted(self) -> None:
        """가격이 진입가 그대로여도 왕복 수수료만큼은 손실이다 -- 0이 아니어야 한다."""
        for side in ("long", "short"):
            with self.subTest(side=side):
                self.assertAlmostEqual(self.f({"side": side, "entry": 2500.0}, 2500.0), -10.0, places=2)

    def test_no_price_returns_none(self) -> None:
        """기준가를 못 읽었을 때 0이 아니라 None -- 화면이 '이익도 손실도 아님'으로 오독하면 안 된다."""
        self.assertIsNone(self.f({"side": "long", "entry": 2000.0}, None))

    def test_malformed_position_returns_none(self) -> None:
        for bad in ({}, {"side": "long"}, {"side": "long", "entry": "x"}, {"side": "long", "entry": 0.0}):
            with self.subTest(bad=bad):
                self.assertIsNone(self.f(bad, 2000.0))

    def test_agrees_with_locked_bp_when_price_is_the_stop(self) -> None:
        """현재가가 정확히 손절선이면 미실현 = locked_bp. 두 값이 같은 자로 재는지 확인."""
        pos = {"side": "short", "entry": 2498.47, "stop": 2514.087857142857}
        self.assertAlmostEqual(self.f(pos, pos["stop"]), self.locked(pos), places=2)


if __name__ == "__main__":
    unittest.main()
