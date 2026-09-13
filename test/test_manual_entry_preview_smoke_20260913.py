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
🔴2026-09-13 둘째 판: `produce_account`/`fetch_binance_json` 은 make_app **클로저**라 모듈 패치가
닿지 않았다(청산 미리보기가 no_position 400 으로 끝나 내용 검사가 불가능했다). 계좌는 모듈 수준
`fetch_account` 를 갈아끼워 해결했다. 호가/클라인은 아직 클로저라 실제 공개 엔드포인트를 친다.
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

        async def fake_account(*_a, **_k):
            return FAKE_ACCOUNT

        with mock.patch.object(server, "LIVE_DIR", live), \
             mock.patch.object(server, "DASHBOARD_DIR", dash), \
             mock.patch.object(server, "POSITION_SIZING_STATE_PATH",
                               live / "eth_position_sizing_state.json"), \
             mock.patch.object(server, "POSITION_SIZING_MAX_AGE_MIN", 10 ** 9), \
             mock.patch.object(server, "fetch_binance_json", _fake_binance, create=True), \
             mock.patch.object(server, "produce_account", fake_account, create=True), \
             mock.patch.object(server, "fetch_account", fake_account), \
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

    def test_trade_plan_is_attached(self) -> None:
        """2026-09-13 «지금 상황» 플랜이 진입·청산 미리보기에 실려 온다(보유 권고·사다리)."""
        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                for path in ("/api/manual-entry/preview?side=LONG&hold=240",
                             "/api/manual-exit/preview?side=LONG&hold=240"):
                    body = await (await client.get(path)).json()
                    tp = (body.get("plan") or {}).get("trade_plan") or {}
                    self.assertEqual(tp.get("hold_min"), 240, path)
                    self.assertTrue(tp["hold"]["available"], path)
                    self.assertIn(tp["hold"]["recommended_min"], HOLDS, path)
                    self.assertTrue(tp["exit_ladder"]["ladder"], path)
                    self.assertIn(tp["execution"]["exit"]["mode"], ("peg_repeg", "market"), path)
            finally:
                await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_unrealized_loss_does_not_block_entry(self) -> None:
        """🔴2026-09-13 철회 고정: 같은 방향이 평가손실 중이어도 진입을 **막지 않는다**.

        같은 날 오전에 «역행 중 추가 금지»를 submit 게이트로 올렸다가 사용자 실계좌 69왕복으로
        되재고 철회했다(분할 자체는 건당 수익률과 무관, 상관 −0.04 · 95%CI 0 포함).
        근거: research_user_scaling_in_pnl_attribution_20260913. 이 테스트는 그 게이트가 다시
        기어들어오면 실패한다 -- 게이트를 되살리려면 사용자 데이터로 근거를 먼저 만들어야 한다.

        `run_entry` 를 갈아끼워 **주문이 절대 못 나가게** 한 뒤 계획까지만 확인한다.
        """
        losing = dict(FAKE_ACCOUNT)
        losing["positions"] = [{**FAKE_ACCOUNT["positions"][0], "unrealized_pnl": -120.0}]
        fired = []

        async def never_run(_session, plan, state):
            fired.append(plan)
            state.update(phase="filled_maker")
            return state

        async def fake_account(*_a, **_k):
            return losing

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "exec_enabled", lambda: True), \
                 mock.patch.object(server, "run_entry", never_run):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    prev = await (await client.get(
                        "/api/manual-entry/preview?side=LONG&hold=240&pct=50")).json()
                    self.assertTrue(prev.get("ok"), prev)
                    self.assertNotIn("averaging_down", prev["plan"],
                                     "철회한 물타기 판정이 계획에 다시 붙었다")
                    self.assertNotIn("add_allowed", prev["plan"]["trade_plan"]["entry_split"])
                    resp = await client.post(
                        "/api/manual-entry/submit?side=LONG&confirm=1&hold=240&pct=50")
                    body = await resp.json()
                    self.assertTrue(body.get("ok"),
                                    f"평가손실 중 진입이 막혔다(철회한 규칙이 살아있다): {body}")
                    self.assertEqual(len(fired), 1, "통과했으면 주문 경로가 불려야 한다")
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_entry_and_exit_use_the_same_cap(self) -> None:
        """🔴진입과 청산이 **같은 상한**을 본다(2026-09-13).

        예전에는 진입만 세 상한(원장·순자산·모델)의 최솟값을 쓰고 청산의 «최소 청산 비율»은
        정책상한 25배 기준이었다. 그래서 20배 포지션에서 같은 카드가 «최소 청산 0%»와
        «예산 사다리 60%»를 나란히 띄웠다. 두 숫자가 같은 상한에서 나오는지 고정한다.
        """
        big = dict(FAKE_ACCOUNT)
        # 순자산 1,000 · 명목 20,000 = 20배. 순자산 상한 8배를 크게 넘는다.
        big["positions"] = [{**FAKE_ACCOUNT["positions"][0], "qty": 8.0, "notional": 20000.0}]

        async def fake_account(*_a, **_k):
            return big

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    body = await (await client.get(
                        "/api/manual-exit/preview?side=LONG&hold=60")).json()
                    plan = body["plan"]
                    risk = plan["risk"]
                    ladder = plan["trade_plan"]["exit_ladder"]["ladder"]
                    same = [r for r in ladder if r["hold_min"] == risk["hold_min"]][0]
                    self.assertGreater(risk["required_fraction"], 0.0,
                                       "20배 포지션인데 닫을 필요가 없다고 한다")
                    self.assertAlmostEqual(risk["required_fraction"],
                                           same["required_fraction"], places=3,
                                           msg=f"최소 청산과 사다리가 다른 상한을 쓴다: {risk} / {same}")
                    self.assertIn(risk.get("applied_binding"),
                                  ("ledger", "equity", "model"), risk)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_prescription_is_attached_and_always_lump(self) -> None:
        """처방(배수·보유·분할)이 진입·청산 미리보기에 실리고, 분할은 **항상 1** 이다.

        분할이 1 이 아니게 되면 869일 크기매칭 실험(6/6 전패)을 다시 돌려야 한다는 뜻이므로
        여기서 잡는다. 배수는 방향 가정과 무관해야 하므로 그 사실도 같이 고정한다.
        """
        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                for path in ("/api/manual-entry/preview?side=LONG&hold=240",
                             "/api/manual-exit/preview?side=LONG&hold=240"):
                    body = await (await client.get(path)).json()
                    rx = (body.get("plan") or {}).get("trade_plan", {}).get("prescription") or {}
                    self.assertTrue(rx.get("available"), f"{path}: {rx}")
                    self.assertEqual(rx["tranches"], 1, "분할은 자유 변수가 아니다")
                    self.assertIn(rx["hold_min"], HOLDS, rx)
                    self.assertGreater(rx["leverage"], 0, rx)
                    self.assertAlmostEqual(rx["liq_distance_pct"], 100.0 / rx["leverage"],
                                           delta=0.1, msg=str(rx))
                    # 거래소 레버리지: 정책 상한만큼은 반드시 열려야 한다
                    lv = rx["exchange_leverage"]
                    self.assertTrue(lv["available"], lv)
                    self.assertGreaterEqual(lv["setting"], lv["min_feasible"], lv)
                    self.assertLessEqual(lv["margin_pct_of_equity"], 100.0, lv)
            finally:
                await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_exchange_leverage_is_stable_across_hold(self) -> None:
        """🔴거래소 레버리지는 보유시간 선택에 흔들리면 안 된다(2026-09-13 라이브 회귀).

        한 번 걸어 두는 값인데 지평별 모델 상한을 따라가고 있었다. 기준은 정책 천장
        (원장·순자산의 작은 쪽)이어야 한다. 로컬에서는 순자산 상한이 묶여 증상이 안 보였다.
        """
        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                seen = set()
                for h in HOLDS:
                    body = await (await client.get(
                        f"/api/manual-entry/preview?side=LONG&hold={h}")).json()
                    rx = body["plan"]["trade_plan"]["prescription"]
                    seen.add(rx["exchange_leverage"]["setting"])
                self.assertEqual(len(seen), 1,
                                 f"보유시간에 따라 거래소 설정이 흔들린다: {sorted(seen)}")
            finally:
                await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_bad_inputs_are_rejected_not_crashed(self) -> None:
        """잘못된 입력은 **검증**으로 막혀야지 예외로 죽으면 안 된다."""
        self._get_all([("/api/manual-entry/preview?side=NOPE", 400),
                       ("/api/manual-entry/preview?side=LONG&pct=0", 400),
                       ("/api/manual-entry/preview?side=LONG&pct=abc", 400),
                       ("/api/manual-exit/preview?side=LONG&pct=101", 400)])


if __name__ == "__main__":
    unittest.main()
