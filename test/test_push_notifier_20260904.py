"""웹푸시 알림 스택 테스트, 2026-09-04.

중점은 두 가지다.
1. 손으로 구현한 RFC 8291 암호화가 맞는가 -- 틀리면 푸시 서비스는 201을 주고 브라우저만 조용히
   복호화에 실패하므로, 눈으로는 "알림이 안 온다"와 구분되지 않는다. RFC의 고정 벡터로 못박는다.
2. 감지기(2026-10-05 재배선) -- 선을 «넘어 들어설 때» 한 번만 · 다시 무장(히스테리시스) · 보유 코인만.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import live_push_notifier_20260904 as notifier  # noqa: E402
from scripts import push_webpush_20260904 as webpush  # noqa: E402


class WebPushCryptoTests(unittest.TestCase):
    def test_rfc8291_section5_vector(self) -> None:
        """RFC 8291 5절 예제를 바이트 단위로 재현. 이게 깨지면 암호화가 틀린 것이다."""
        webpush.selftest_rfc8291()

    def test_vapid_header_audience_is_origin_not_full_endpoint(self) -> None:
        """`aud`에 전체 엔드포인트를 넣는 것은 흔한 실수다 -- FCM은 관대하지만 Mozilla는 401."""
        private, _ = webpush.generate_vapid_keys()
        header = webpush.vapid_authorization(
            "https://updates.push.services.mozilla.com/wpush/v2/gAAAA-long-token",
            private, "mailto:x@y.z")
        self.assertTrue(header.startswith("vapid t="))
        jwt = header.split("t=", 1)[1].split(",", 1)[0]
        claims = json.loads(webpush.b64u_decode(jwt.split(".")[1]))
        self.assertEqual(claims["aud"], "https://updates.push.services.mozilla.com")
        self.assertNotIn("/wpush", claims["aud"])

    def test_vapid_signature_is_raw_64_bytes_not_der(self) -> None:
        """ES256은 r||s 64바이트를 요구한다. cryptography가 주는 DER을 그대로 넘기면 조용한 401."""
        private, _ = webpush.generate_vapid_keys()
        header = webpush.vapid_authorization("https://fcm.googleapis.com/fcm/send/abc",
                                             private, "mailto:x@y.z")
        jwt = header.split("t=", 1)[1].split(",", 1)[0]
        self.assertEqual(len(webpush.b64u_decode(jwt.split(".")[2])), 64)

    def test_public_key_derivation_matches_generated_pair(self) -> None:
        private, public = webpush.generate_vapid_keys()
        self.assertEqual(webpush.vapid_public_key_from_private(private), public)
        self.assertEqual(len(webpush.b64u_decode(public)), 65)  # 비압축 P-256 점


class SubscriptionStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp()) / "subs.json"

    def test_add_is_idempotent_by_endpoint(self) -> None:
        sub = {"endpoint": "https://push.example/abc", "keys": {"p256dh": "p", "auth": "a"}}
        first = webpush.add_subscription(sub, label="데스크톱", path=self.tmp)
        second = webpush.add_subscription(sub, label="데스크톱", path=self.tmp)
        self.assertEqual(first, second)
        self.assertEqual(len(webpush.load_subscriptions(self.tmp)), 1)

    def test_remove_and_missing_file_degrade_cleanly(self) -> None:
        self.assertEqual(webpush.load_subscriptions(self.tmp), {})
        sub = {"endpoint": "https://push.example/abc", "keys": {"p256dh": "p", "auth": "a"}}
        sid = webpush.add_subscription(sub, path=self.tmp)
        self.assertTrue(webpush.remove_subscription(sid, path=self.tmp))
        self.assertFalse(webpush.remove_subscription(sid, path=self.tmp))

    def test_corrupt_store_reads_as_empty_not_crash(self) -> None:
        """찢어진 JSON에 데몬이 죽으면 알림 전체가 조용히 멈춘다."""
        self.tmp.parent.mkdir(parents=True, exist_ok=True)
        self.tmp.write_text("{not json", encoding="utf-8")
        self.assertEqual(webpush.load_subscriptions(self.tmp), {})


def _acct(*positions, margin: float = 1000.0) -> dict:
    return {"ok": True, "balance": {"margin": margin}, "positions": list(positions)}


def _pos(symbol="ETHUSDC", side="LONG", qty=1.0, mark=2500.0, liq=1500.0, notional=2500.0) -> dict:
    return {"symbol": symbol, "side": side, "qty": qty, "mark_price": mark,
            "liquidation_price": liq, "notional": notional}


class EnabledDetectorsTests(unittest.TestCase):
    """운영 스위치를 고정한다. 바꾸면 이 테스트가 먼저 깨져 «의도한 변경»임을 확인하게 된다.
    2026-10-05: 켜져 있던 net_score·v_rebound 의 원천 API 가 지워져 09-24 이후 발송 0건이었다."""

    def test_production_switch_is_what_we_think_it_is(self) -> None:
        self.assertEqual(notifier.ENABLED_DETECTORS, {"verdict", "risk", "burst_hold", "prewarn_hold"})
        self.assertEqual(set(notifier.PUSH_KINDS), notifier.ENABLED_DETECTORS)   # 알림 센터 표 = 실제 발송 종류

    def test_calendar_file_parses_and_has_dates(self) -> None:
        items = notifier.load_calendar().get("items") or []
        self.assertTrue(items)
        for it in items:
            time.strptime(it["date"], "%Y-%m-%d")
            self.assertTrue(it["id"] and it["title"] and it["how"])


class DetectorTests(unittest.TestCase):
    def test_verdict_only_on_the_day_after_9_kst(self) -> None:
        cal = {"items": [{"id": "x", "date": "2026-11-10", "title": "T", "what": "W", "how": "H"},
                         {"id": "y", "date": "2026-11-10", "title": "건수", "how": "H", "approx": True}]}
        at = lambda s: time.mktime(time.strptime(s, "%Y-%m-%dT%H:%M")) - time.timezone - 9 * 3600   # noqa: E731 -- KST
        self.assertEqual(notifier.detect_verdicts(cal, at("2026-11-10T08:59")), [])
        notes = notifier.detect_verdicts(cal, at("2026-11-10T09:00"))
        self.assertEqual([n.key for n in notes], ["verdict:x:2026-11-10"])     # 건수 기준(approx)은 날짜가 추정이라 안 보낸다
        self.assertEqual(notifier.detect_verdicts(cal, at("2026-11-11T10:00")), [])

    def test_risk_fires_once_on_entry_and_rearms_with_hysteresis(self) -> None:
        st: dict = {}
        far = _acct(_pos(mark=2500, liq=1500))                       # 40%
        near = _acct(_pos(mark=2500, liq=2400))                      # 4%
        mid = _acct(_pos(mark=2500, liq=2360))                       # 5.6% -- 무장선(7%) 안
        self.assertEqual(notifier.detect_risk(far, st, 1), [])
        self.assertEqual(len(notifier.detect_risk(near, st, 2)), 1)
        self.assertEqual(notifier.detect_risk(mid, st, 3), [])
        self.assertEqual(notifier.detect_risk(near, st, 4), [])      # 7% 밖으로 안 나갔다 → 아직 무장 전
        notifier.detect_risk(far, st, 5)
        self.assertEqual(len(notifier.detect_risk(near, st, 6)), 1)  # 다시 무장된 뒤 재진입
        self.assertEqual(notifier.detect_risk({"ok": False}, st, 7), [])

    def test_exposure_over_6x(self) -> None:
        st: dict = {}
        big = _acct(_pos(notional=6500.0), margin=1000.0)
        notes = notifier.detect_risk(big, st, 1)
        self.assertEqual([n.key.split(":")[1] for n in notes], ["expo"])
        self.assertEqual(notifier.detect_risk(big, st, 2), [])
        notifier.detect_risk(_acct(_pos(notional=4000.0)), st, 3)   # 5배 밑 → 다시 무장
        self.assertEqual(len(notifier.detect_risk(big, st, 4)), 1)

    def test_burst_needs_eth_position_and_fires_on_rising_edge(self) -> None:
        st: dict = {}
        mc = {"burst": {"long_usd_60s": 400_000.0, "short_usd_60s": 0.0, "thr": [338_000, 546_000]}}
        self.assertEqual(notifier.detect_burst_hold(mc, _acct(), st, 1), [])           # 보유 없음
        st.clear()
        self.assertEqual(len(notifier.detect_burst_hold(mc, _acct(_pos()), st, 2)), 1)
        self.assertEqual(notifier.detect_burst_hold(mc, _acct(_pos()), st, 3), [])      # 켜진 채 -- 다시 안 보냄
        self.assertEqual(notifier.detect_burst_hold({}, _acct(_pos()), st, 4), [])

    def test_prewarn_for_held_coin_only(self) -> None:
        st: dict = {}
        bo = {"sol": {"available": True, "timestamp": "T1", "prewarn": {"on": True}},
              "eth": {"available": True, "timestamp": "T1", "prewarn": {"on": True}}}
        notes = notifier.detect_prewarn_hold(bo, _acct(_pos(symbol="SOLUSDC")), st)
        self.assertEqual([n.key for n in notes], ["prewarn:sol:T1"])
        self.assertEqual(notifier.detect_prewarn_hold(bo, _acct(_pos(symbol="SOLUSDC")), st), [])


class RunCycleTests(unittest.IsolatedAsyncioTestCase):
    """run_cycle 자체(기준선·중복제거·보낸 알림 기록)를 검사한다."""

    def setUp(self) -> None:
        self.sent: list[dict] = []
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig = (notifier.STATE_PATH, notifier.SENT_LOG_PATH, notifier.broadcast)
        notifier.STATE_PATH = self.tmpdir / "state.json"
        notifier.SENT_LOG_PATH = self.tmpdir / "sent.jsonl"

        async def fake_broadcast(payload, **kwargs):
            self.sent.append(payload)
            return {"sent": 1, "pruned": 0, "failed": 0}

        notifier.broadcast = fake_broadcast

    def tearDown(self) -> None:
        notifier.STATE_PATH, notifier.SENT_LOG_PATH, notifier.broadcast = self._orig

    async def _cycle(self, state, data):
        async def fake_fetch(_session, _base):
            return data
        orig = notifier.fetch_all
        notifier.fetch_all = fake_fetch
        try:
            await notifier.run_cycle(None, "", state, private="k", subject="mailto:x@y.z", dry_run=False)
        finally:
            notifier.fetch_all = orig

    async def test_first_run_sends_nothing_and_records_baseline(self) -> None:
        """재시작 폭주 방지. 이게 없으면 데몬을 껐다 켤 때마다 현재 상태 전부가 쏟아진다."""
        state = notifier.load_state()
        await self._cycle(state, {"account": _acct(_pos(liq=2450))})
        self.assertEqual(self.sent, [])
        self.assertTrue(state["baseline_done"] and state["seen"])

    async def test_second_run_sends_and_logs_once(self) -> None:
        state = notifier.load_state()
        await self._cycle(state, {"account": _acct(_pos())})
        await self._cycle(state, {"account": _acct(_pos(liq=2450))})
        await self._cycle(state, {"account": _acct(_pos(liq=2450))})
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["tier"], "t1")
        rows = [json.loads(x) for x in notifier.SENT_LOG_PATH.read_text().splitlines()]
        self.assertEqual([(r["kind"], r["sent"]) for r in rows], [("risk", 1)])


if __name__ == "__main__":
    unittest.main()
