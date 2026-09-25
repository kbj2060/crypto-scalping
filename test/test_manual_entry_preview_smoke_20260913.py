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
from datetime import datetime, timedelta, timezone
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

        # 🔴옛 정책(원장∧순자산∧모델)을 고정한다 -- 2026-09-25 «위험모델만» 스위치는 나중에
        #   되돌릴 한시 설정이라, 되돌아올 규칙을 여기서 계속 지킨다. 새 동작은 test_model_only_*.
        with mock.patch.object(server, "SIZING_CAP_MODEL_ONLY", False), \
             mock.patch.object(server, "SIZING_MARGIN_CAP_PCT", 0.0), \
             mock.patch.object(server, "LIVE_DIR", live), \
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
                    # 🔴손절이 있으면 위험 지표가 손절 기준이어야 한다(2026-09-13)
                    sr = rx["stop_risk"]
                    self.assertTrue(sr["available"], sr)
                    self.assertTrue(sr["liq_unreachable"],
                                    f"3% 손절이 청산선 밖이면 보호가 없다: {sr}")
                    self.assertAlmostEqual(sr["per_stop_pct"], 3.0 * rx["leverage"],
                                           places=1, msg=str(sr))
                    self.assertGreaterEqual(sr["consecutive_to_half"], 1, sr)
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

    def test_leverage_gauge_overrides_model(self) -> None:
        """게이지가 값을 주면 그걸 쓰고, 없으면 모델 추천을 쓴다. 범위 밖이면 모델로 떨어진다.

        `target_leverage` 가 집행기로 넘어가는 유일한 값이라 여기서 계약을 고정한다 --
        이 값이 틀리면 실계좌 레버리지가 틀리게 걸린다.
        """
        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                base = await (await client.get(
                    "/api/manual-entry/preview?side=LONG&hold=240")).json()
                p0 = base["plan"]
                self.assertEqual(p0["leverage_source"], "model", p0)
                self.assertEqual(p0["target_leverage"], p0["leverage_model"], p0)
                self.assertTrue(p0["leverage_steps"], p0)

                man = await (await client.get(
                    "/api/manual-entry/preview?side=LONG&hold=240&lev=25")).json()
                p1 = man["plan"]
                self.assertEqual(p1["target_leverage"], 25, p1)
                self.assertEqual(p1["leverage_source"], "manual", p1)
                # 모델 추천은 게이지와 무관하게 그대로여야 한다(화면이 둘을 나란히 보여준다)
                self.assertEqual(p1["leverage_model"], p0["leverage_model"], p1)

                for bad in ("0", "999", "abc", "-3"):
                    r = await (await client.get(
                        f"/api/manual-entry/preview?side=LONG&hold=240&lev={bad}")).json()
                    self.assertEqual(r["plan"]["leverage_source"], "model",
                                     f"lev={bad} 가 조용히 먹혔다: {r['plan']}")
            finally:
                await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_leverage_floor_respects_open_position(self) -> None:
        """🔴열린 포지션이 큰 상태에서 모델이 «내릴 수 없는 값»을 권하면 안 된다.

        레버리지를 내리면 기존 포지션의 초기증거금이 올라가고, 순자산을 넘으면 거래소가
        -2028(MIN_LEVERAGE_RATIO)로 거부한다. 그 바닥 위를 권해야 실제로 걸린다.
        (격리였다면 -4161 로 아예 막힌다 -- 이 계좌는 cross 라 해당 없다.)
        """
        big = dict(FAKE_ACCOUNT)
        # 순자산 1,000 · 명목 11,000 = 11배. 10배로 내리면 증거금 1,100 > 1,000 이라 거부된다.
        big["positions"] = [{**FAKE_ACCOUNT["positions"][0], "qty": 4.4, "notional": 11000.0}]

        async def fake_account(*_a, **_k):
            return big

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    body = await (await client.get(
                        "/api/manual-entry/preview?side=LONG&hold=240")).json()
                    p = body["plan"]
                    lv = p["trade_plan"]["prescription"]["exchange_leverage"]
                    self.assertTrue(lv["forced_by_position"], lv)
                    self.assertGreaterEqual(lv["setting"] * 1000.0, 11000.0,
                                            f"이 설정으로는 기존 포지션을 못 버텨 거부된다: {lv}")
                    self.assertEqual(p["target_leverage"], lv["setting"], p)
                    self.assertIsNotNone(p["leverage_position_floor"], p)
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_hold_ignores_query_and_does_not_extend_on_scale_in(self) -> None:
        """🔴지평은 **서버가 정하고**(planning_hold), 물타기해도 시계가 안 늘어난다.

        ① 쿼리로 다른 보유시간을 주어도 서버가 무시한다 -- 2026-09-14 부터는 읽는 코드
           자체가 없다(`query_hold` 삭제). 쿼리로만 바꿀 수 있으면 «화면과 다른 크기»가 나간다.
        ② 포지션이 이미 2시간 묵었으면 남은 시간은 계획 지평이 아니라 그 나머지다.
           `entry_at` 은 첫 체결 시각이라 추가 진입에도 안 움직인다.
        이 픽스처에서는 planning_hold 가 240 을 고른다 -- 240 이라는 **상수**를 고정하는 게
        아니라 «서버가 고른 값이 그대로 내려온다»를 고정한다(가변 지평은 아래 테스트).
        """
        aged = dict(FAKE_ACCOUNT)
        old_entry = (datetime.now(timezone.utc) - timedelta(minutes=125)).isoformat()
        aged["positions"] = [{**FAKE_ACCOUNT["positions"][0], "entry_at": old_entry}]

        async def fake_account(*_a, **_k):
            return aged

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    for q in ("", "&hold=60", "&hold=1440", "&hold=abc"):
                        b = await (await client.get(
                            f"/api/manual-entry/preview?side=LONG{q}")).json()
                        p = b["plan"]
                        self.assertEqual(p["hold_planned_min"], 240, f"{q}: {p['hold_planned_min']}")
                        # 125분 묵었으니 남은 115분 -> 모델 지평으로 올림하면 120
                        self.assertEqual(p["hold_remaining_min"], 120,
                                         f"{q}: 물타기로 시계가 늘어났다 {p['hold_remaining_min']}")
                    x = await (await client.get("/api/manual-exit/preview?side=LONG")).json()
                    self.assertEqual(x["plan"]["hold_remaining_min"], 120, x["plan"])
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_bracket_uses_liquidation_levels_or_says_why(self) -> None:
        """2026-09-25 청산맵 TP/SL(고정 3% 손절 대체). 롱: TP 저항2 · SL 지지1 · 비상 스탑은 SL 너머.
        숏은 거울. 레벨이 없으면 **이유를 싣는다** -- 옛 손절은 조용히 안 걸려 있었다(-4120)."""
        # 🔴이 픽스처의 가짜 호가는 서버 안쪽 함수까지 못 닿아 미리보기가 **실시세**를 쓴다 --
        #   레벨을 고정값으로 심으면 시세 쪽에 따라 전부 «지나간 레벨»이 된다. 시세 기준 ±3%/±6%.
        seen = {}

        def levels(_df, price):
            seen["p"] = price
            return {"warmed_up": True,
                    "support_levels": [{"price": price * 0.97}, {"price": price * 0.94}],
                    "resistance_levels": [{"price": price * 1.03}, {"price": price * 1.06}]}

        async def exercise() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                for side, tp_x, sl_x in (("LONG", 1.06, 0.97), ("SHORT", 0.94, 1.03)):
                    b = (await (await client.get(
                        f"/api/manual-entry/preview?side={side}&pct=5")).json())["plan"]["bracket"]
                    self.assertTrue(b["available"], b)
                    tp, sl = seen["p"] * tp_x * b["basis"], seen["p"] * sl_x * b["basis"]
                    self.assertIsNotNone(b["tp_price"], b)
                    self.assertAlmostEqual(b["tp_price"], tp, delta=0.02)
                    self.assertAlmostEqual(b["sl_price"], sl, delta=0.02)
                    far = b["backstop_price"] < sl if side == "LONG" else b["backstop_price"] > sl
                    self.assertTrue(far, f"비상 스탑이 SL 보다 먼저 걸린다: {b}")
            finally:
                await client.close()

        with _isolated_dirs(), mock.patch.object(server, "compute_spliced_levels", levels):
            asyncio.run(exercise())

        async def no_levels() -> None:
            client = TestClient(TestServer(server.make_app()))
            await client.start_server()
            try:
                b = (await (await client.get(
                    "/api/manual-entry/preview?side=LONG&pct=5")).json())["plan"]["bracket"]
                self.assertFalse(b["available"], b)
                self.assertTrue(b.get("reason"), f"못 거는 이유가 없다: {b}")
            finally:
                await client.close()

        with _isolated_dirs(), mock.patch.object(server, "compute_spliced_levels",
                                                 lambda _d, _p: {"warmed_up": True}):
            asyncio.run(no_levels())

    def test_entry_and_exit_share_one_horizon_and_one_cap(self) -> None:
        """🔴진입과 청산이 **같은 지평·같은 상한**을 쓴다(2026-09-14 병합 감사).

        두 가지가 갈라져 있었다:
        ① 청산 카드의 `trade_plan` 이 상한을 상수 6.0 배로 계산했다. 같은 카드의 `risk` 는
           `effective_cap`(원장∧순자산∧모델)을 쓰므로, 원장이 낮게 묶으면 한 카드가
           «상한 넘었으니 닫아라»와 «더 넣을 여유 있음»을 나란히 띄웠다.
        ② 청산·추가진입의 시계 길이가 240분 상수였다. 진입이 1440분으로 처방하고 크기까지
           그 셀로 정했는데 청산은 4시간 예산으로 쟀고, 4시간이 지나면 60분 셀로 떨어져
           «여유가 더 생겼다»고 말했다(지평이 짧을수록 허용 배수가 커지기 때문).

        원장 상한 4.0배(순자산 6.0배보다 낮다)를 심고 planning_hold 를 1440분으로 고정해
        두 결함을 동시에 드러낸다. 수정 전 값: 청산 배수 4.18(상수 6.0 기준) · 남은 시계 120분.
        """
        aged = dict(FAKE_ACCOUNT)
        old_entry = (datetime.now(timezone.utc) - timedelta(minutes=125)).isoformat()
        aged["positions"] = [{**FAKE_ACCOUNT["positions"][0], "entry_at": old_entry}]

        async def fake_account(*_a, **_k):
            return aged

        real_payload = server.position_sizing_payload

        def with_ledger_cap():
            # 순자산 상한(1000 x 6 = 6000)보다 **낮은** 원장 상한. 이게 있어야 두 상한이 갈린다.
            pay = real_payload()
            pay["cap"] = {"available": True, "cap_notional_usdt": 4000.0,
                          "trips": 20, "need": 10}
            return pay

        async def exercise() -> None:
            with mock.patch.object(server, "fetch_account", fake_account), \
                 mock.patch.object(server, "position_sizing_payload", with_ledger_cap), \
                 mock.patch.object(server, "recommend_hold", lambda *a, **k: {
                     "available": True, "recommended_min": 1440}):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    e = (await (await client.get(
                        "/api/manual-entry/preview?side=LONG")).json())["plan"]
                    x = (await (await client.get(
                        "/api/manual-exit/preview?side=LONG")).json())["plan"]

                    # ② 지평 단일 출처: 두 카드가 같은 계획 지평을 말한다.
                    self.assertEqual(e["hold_planned_min"], 1440, e["hold_planned_min"])
                    self.assertEqual(x["hold_planned_min"], e["hold_planned_min"],
                                     f"진입 {e['hold_planned_min']} vs 청산 {x['hold_planned_min']}")
                    # 125분 묵었고 계획이 1440분이면 남은 1315 -> 모델 지평으로 올림하면 1440.
                    # 240분 상수를 쓰던 시절에는 120 이 나왔다.
                    self.assertEqual(x["hold_remaining_min"], 1440,
                                     f"청산 시계가 계획 지평을 안 따른다: {x['hold_remaining_min']}")

                    # ① 상한 단일 출처: 원장 4.0배가 묶었으니 양쪽 다 4.0 이어야 한다.
                    self.assertEqual(x["risk"]["effective_x"], 4.0, x["risk"])
                    for who, p in (("진입", e), ("청산", x)):
                        lev = p["trade_plan"]["size"]["leverage"]
                        self.assertEqual(lev, 4.0,
                                         f"{who} trade_plan 이 실효 상한을 안 쓴다: {lev}")
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_model_only_cap_drops_ledger_and_equity(self) -> None:
        """2026-09-25 사용자 «위험모델 상한만, 나머지 해제». 원장 4.0배·순자산 6배가 모델보다
        낮아도 진입·청산 둘 다 **모델**이 묶고, 거래소 레버리지는 그 크기를 열 만큼 올라간다
        (정책 천장이 6배로 남으면 7배로 걸려 거래소가 주문을 거부한다). 모델이 없으면 옛 상한."""
        real_payload = server.position_sizing_payload

        def with_ledger_cap(drop_model=False):
            def f():
                pay = real_payload()
                pay["cap"] = {"available": True, "cap_notional_usdt": 4000.0, "trips": 20, "need": 10}
                if drop_model:
                    pay["risk_mae"] = {}
                return pay
            return f

        async def exercise() -> None:
            with mock.patch.object(server, "SIZING_CAP_MODEL_ONLY", True), \
                 mock.patch.object(server, "position_sizing_payload", with_ledger_cap()):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    b = await (await client.get("/api/manual-entry/preview?side=LONG")).json()
                    cap, plan = b["cap"], b["plan"]
                    self.assertEqual(cap["binding"], "model", cap)
                    self.assertEqual(cap["cap_notional_usdt"], cap["cap_model_usdt"], cap)
                    self.assertGreater(cap["cap_notional_usdt"], 4000.0, "원장 4.0배가 아직 묶는다")
                    lv = plan["trade_plan"]["prescription"]["exchange_leverage"]
                    self.assertGreaterEqual(lv["setting"] * 1000.0, cap["cap_notional_usdt"], lv)
                    x = (await (await client.get("/api/manual-exit/preview?side=LONG")).json())["plan"]
                    self.assertEqual(x["risk"]["applied_binding"], "model", x["risk"])
                finally:
                    await client.close()
            with mock.patch.object(server, "SIZING_CAP_MODEL_ONLY", True), \
                 mock.patch.object(server, "position_sizing_payload", with_ledger_cap(True)):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                try:
                    b = await (await client.get("/api/manual-entry/preview?side=LONG")).json()
                    self.assertEqual(b["cap"]["binding"], "ledger",
                                     f"모델이 없으면 옛 상한으로 떨어져야 한다: {b['cap']}")
                finally:
                    await client.close()

        with _isolated_dirs():
            asyncio.run(exercise())

    def test_margin_cap_50pct_follows_order_leverage(self) -> None:
        """2026-09-25 사용자 «증거금 50% 상한» + «비율 10% = 순자산의 10% 를 증거금으로, 넘으면 진입 불가».
        순자산 1000 · 기존 2500. 게이지 6배면 상한 = 증거금 50% = 명목 3000(모델 4184 보다 낮다).
          5%  → 증거금 50 = 명목 300 → 2800 ≤ 3000 통과(자르지 않은 그 크기).
          100% → 명목 6000 → 넘는다 → **자르지 않고** 막는다.
        «자동»(처방 30배)에서 10% → 명목 3000 → 5500 > 모델 4184 → «위험모델 상한» 으로 막는다."""
        async def exercise() -> None:
            with mock.patch.object(server, "SIZING_CAP_MODEL_ONLY", True), \
                 mock.patch.object(server, "SIZING_MARGIN_CAP_PCT", 50.0):
                client = TestClient(TestServer(server.make_app()))
                await client.start_server()
                get = lambda q: client.get("/api/manual-entry/preview?side=LONG" + q)
                try:
                    b = await (await get("&lev=6&pct=5")).json()
                    cap, plan = b["cap"], b["plan"]
                    self.assertEqual(cap["binding"], "margin", cap)
                    self.assertAlmostEqual(cap["cap_notional_usdt"], 3000.0, places=2)
                    self.assertIsNone(plan["blocked"], plan["blocked"])
                    self.assertAlmostEqual(plan["notional_usdt"], 300.0, delta=plan["price"] * 0.001)
                    b = await (await get("&lev=6&pct=100")).json()
                    self.assertTrue(str(b["plan"]["blocked"]).startswith("증거금 상한 50% 초과"),
                                    b["plan"]["blocked"])
                    self.assertAlmostEqual(b["plan"]["notional_usdt"], 6000.0, delta=3.0)
                    b = await (await get("&pct=10")).json()
                    self.assertEqual(b["cap"]["binding"], "model", b["cap"])
                    self.assertTrue(str(b["plan"]["blocked"]).startswith("위험모델 상한 초과"),
                                    b["plan"]["blocked"])
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
