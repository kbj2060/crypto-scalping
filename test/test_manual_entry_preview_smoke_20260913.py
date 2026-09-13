"""진입·청산 미리보기가 **죽지 않는지**만 본다 (2026-09-13).

왜 필요한가: 2026-09-13 에 분할 권고 블록을 `binding` 정의 **전에** 넣어
`/api/manual-entry/preview` 가 통째로 UnboundLocalError 로 502 가 됐다. 화면에는
«실패: UnboundLocalError» 만 떴다. 그런데 **기존 대시보드 테스트 4/4 는 전부 통과했다** --
이 경로를 아무도 안 지나갔기 때문이다.

여기서 고정하는 건 «숫자가 맞다»가 아니라 **«핸들러가 끝까지 실행된다»** 이다.
파이썬 예외 이름이 본문에 실려 나오면 실패로 본다 -- 그게 위 사고의 정확한 증상이다.

🔴첫 판은 **버그를 못 잡았다**. 사이징 워커 상태가 없어 진입 미리보기가 503 으로 일찍
끝나 문제의 줄에 도달조차 안 했고, 게다가 청산 쪽은 **실계좌와 바이낸스를 그대로 때렸다**.
그래서 여기서는 워커 상태파일을 심고 네트워크·계좌를 전부 갈아끼워 **상한 계산 구간까지
실제로 지나가게** 한다. 음성 대조(버그 재주입)로 이 테스트가 정말 잡는지 확인했다.
"""
from __future__ import annotations

import asyncio
import contextlib
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aiohttp.test_utils import TestClient, TestServer  # noqa: E402

from dashboard import server  # noqa: E402

# 핸들러가 중간에 터지면 이런 이름이 본문에 실린다(assemble_*_plan 의 except 가 문자열화한다).
PY_ERRORS = re.compile(
    r"UnboundLocalError|NameError|AttributeError|TypeError|KeyError|IndexError|ZeroDivisionError")


HOLDS = (60, 120, 240, 480, 1440)
FAKE_STATE = {
    "ok": True, "generated_at": "2099-01-01T00:00:00+00:00",
    "price": 2500.0, "atr_pct": 0.001, "atr_pct_percentile": 0.5, "atr_pct_ref": 0.0012,
    "base_qty": 2.7, "vol_equivalent_qty": 3.0, "vol_equivalent_qty_formula": 3.0,
    "sizing_model": {"used": True}, "horizons": {}, "touch_prob": {},
    # 상한 계산·분할 권고가 실제로 도는 데 필요한 부분
    "risk_mae": {str(h): {s: {"safe_mae_pct": v, "max_leverage": round(100.0 / v, 2)}
                          for s in ("LONG", "SHORT")}
                 for h, v in zip(HOLDS, (1.5, 2.0, 3.0, 7.4, 23.9))},
}
FAKE_ACCOUNT = {
    "ok": True,
    "balance": {"margin": 1000.0, "initial_margin": 0.0, "available": 1000.0, "unrealized": 0.0},
    "positions": [{"symbol": "ETHUSDT", "side": "LONG", "qty": 1.0, "entry_price": 2500.0,
                   "mark_price": 2500.0, "liquidation_price": 2000.0, "leverage": 30.0,
                   "notional": 2500.0, "unrealized_pnl": 0.0, "updated_at": None}],
    "leverage_by_symbol": {"ETHUSDT": 30.0}, "trips": [],
}


async def _fake_binance(url, params, *, timeout=10.0, error_reason=None):
    """bookTicker/klines 만 흉내낸다. **테스트가 바깥 세상을 건드리면 안 된다.**"""
    if "bookTicker" in url:
        return {"symbol": "ETHUSDT", "bidPrice": "2500.00", "askPrice": "2500.01"}
    if "klines" in url:
        n = int(params.get("limit") or 500)
        return [[1700000000000 + i * 300000, "2500", "2501", "2499", "2500",
                 "10", 0, "25000", 100, "5", "12500", "0"] for i in range(n)]
    return None


@contextlib.contextmanager
def _isolated_dirs():
    """디렉터리 격리 + 워커 상태 심기 + 네트워크/계좌 차단."""
    import json as _json
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        live, dash = root / "live", root / "dashboard"
        live.mkdir(); dash.mkdir()
        (dash / "index.html").write_text("<html></html>", encoding="utf-8")
        (live / "eth_position_sizing_state.json").write_text(
            _json.dumps(FAKE_STATE), encoding="utf-8")

        async def fake_account():
            return FAKE_ACCOUNT

        with mock.patch.object(server, "LIVE_DIR", live), \
             mock.patch.object(server, "DASHBOARD_DIR", dash), \
             mock.patch.object(server, "POSITION_SIZING_STATE_PATH",
                               live / "eth_position_sizing_state.json"), \
             mock.patch.object(server, "POSITION_SIZING_MAX_AGE_MIN", 10 ** 9), \
             mock.patch.object(server, "fetch_binance_json", _fake_binance, create=True), \
             mock.patch.object(server, "produce_account", fake_account, create=True), \
             mock.patch.object(server, "load_filters", mock.AsyncMock(return_value={
                 "step": 0.001, "tick": 0.01, "min_qty": 0.001, "min_notional": 20.0})):
            yield


class ManualPreviewSmokeTest(unittest.TestCase):
    def _get_all(self, checks) -> None:
        """checks: [(path, expected_status|None)]. status 가 None 이면 아무 값이나 좋다."""
        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                for path, expect in checks:
                    resp = await client.get(path)
                    body = await resp.text()
                    self.assertIsNone(
                        PY_ERRORS.search(body),
                        f"{path} 응답에 파이썬 예외가 실렸다 ({resp.status}): {body[:200]}")
                    if expect is not None:
                        self.assertEqual(resp.status, expect, f"{path} -> {resp.status}")
            finally:
                await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def _run(self, paths: list[str]) -> None:
        self._get_all([(p, None) for p in paths])

    def test_entry_preview_handler_completes(self) -> None:
        """보유시간·비율을 바꿔가며 진입 미리보기가 예외 없이 끝난다."""
        self._run([
            "/api/manual-entry/preview?side=LONG",
            "/api/manual-entry/preview?side=SHORT",
            "/api/manual-entry/preview?side=LONG&hold=60&pct=30",
            "/api/manual-entry/preview?side=LONG&hold=1440&pct=100",
        ])

    def test_exit_preview_handler_completes(self) -> None:
        self._run([
            "/api/manual-exit/preview?side=LONG",
            "/api/manual-exit/preview?side=SHORT&hold=240&pct=50",
        ])

    def test_bad_inputs_are_rejected_not_crashed(self) -> None:
        """잘못된 입력은 **검증**으로 막혀야지 예외로 죽으면 안 된다."""
        self._get_all([("/api/manual-entry/preview?side=NOPE", 400),
                       ("/api/manual-entry/preview?side=LONG&pct=0", 400),
                       ("/api/manual-entry/preview?side=LONG&pct=abc", 400),
                       ("/api/manual-exit/preview?side=LONG&pct=101", 400)])


if __name__ == "__main__":
    unittest.main()
