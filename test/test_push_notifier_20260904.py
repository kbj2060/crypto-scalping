"""웹푸시 알림 스택 테스트, 2026-09-04.

중점은 두 가지다.
1. 손으로 구현한 RFC 8291 암호화가 맞는가 -- 틀리면 푸시 서비스는 201을 주고 브라우저만 조용히
   복호화에 실패하므로, 눈으로는 "알림이 안 온다"와 구분되지 않는다. RFC의 고정 벡터로 못박는다.
2. sustain window 함정 -- payload의 `*_fired`는 신호별 8~72봉 동안 계속 True다. 그걸로 알림 key를
   만들면 사건 하나에 최대 72번 발송된다. 발동 감지는 반드시 `*_last_fired_ts`를 써야 한다.
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


def _evidence(latest_bar: str, *, net: int, signals: list[dict]) -> dict:
    return {"warmed_up": True, "latest_bar_utc": latest_bar, "price": 4400.0,
            "net_score": net, "signals": signals}


class SustainWindowTrapTests(unittest.TestCase):
    """이 클래스가 이 기능의 핵심 버그를 막는다 -- 모듈 docstring 2번 참고."""

    def test_key_is_stable_across_cycles_for_the_same_firing_bar(self) -> None:
        """sustain window 동안 매 폴링마다 감지돼도 key가 같아야 seen이 중복을 막는다."""
        signals = [{"name": "demarker_extreme", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T01:00:00+00:00"},
                   {"name": "orthogonal_combo", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T01:00:00+00:00"},
                   {"name": "liquidity_sweep", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T01:00:00+00:00"}]
        payload = _evidence("2026-09-04T01:00:00+00:00", net=3, signals=signals)
        keys = {notifier.detect_net_score(payload)[0].key for _ in range(5)}
        self.assertEqual(len(keys), 1)

    def test_only_signals_firing_on_the_latest_bar_are_named(self) -> None:
        """`bottom_fired`(sustain)가 True여도 발동봉이 과거면 이번 사건의 구성원이 아니다."""
        signals = [{"name": "지금발동", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T01:00:00+00:00"},
                   {"name": "옛날발동", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T00:10:00+00:00"},
                   {"name": "또지금", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-04T01:00:00+00:00"},
                   {"name": "또옛날", "bottom_fired": True,
                    "bottom_last_fired_ts": "2026-09-03T22:00:00+00:00"}]
        note = notifier.detect_net_score(_evidence("2026-09-04T01:00:00+00:00", net=4,
                                                   signals=signals))[0]
        self.assertIn("지금발동", note.body)
        self.assertIn("또지금", note.body)
        self.assertNotIn("옛날발동", note.body)
        self.assertNotIn("또옛날", note.body)

    def test_new_firing_bar_produces_a_new_key(self) -> None:
        signals = [{"name": "a", "bottom_fired": True, "bottom_last_fired_ts": "T1"}]
        first = notifier.detect_net_score(_evidence("T1", net=3, signals=signals))[0]
        second = notifier.detect_net_score(_evidence("T2", net=3, signals=signals))[0]
        self.assertNotEqual(first.key, second.key)

    def test_below_threshold_emits_nothing(self) -> None:
        self.assertEqual(notifier.detect_net_score(_evidence("T1", net=2, signals=[])), [])
        self.assertEqual(notifier.detect_net_score(_evidence("T1", net=-2, signals=[])), [])
        self.assertEqual(len(notifier.detect_net_score(_evidence("T1", net=-3, signals=[]))), 1)


class SessionAndBurstDedupTests(unittest.TestCase):
    def test_session_window_keys_once_per_market_per_day(self) -> None:
        """창 안에 있는 동안 매 폴링마다 active로 보이므로 key가 하루 단위로 고정돼야 한다."""
        alerts = {"session_volatility_alert": {"active": [
            {"code": "NYSE", "label": "미국장", "minutes_from_open": 5.0}]}}
        a = notifier.detect_session_window(alerts)[0].key
        alerts["session_volatility_alert"]["active"][0]["minutes_from_open"] = 41.0
        b = notifier.detect_session_window(alerts)[0].key
        self.assertEqual(a, b)

    def test_liq_burst_inactive_emits_nothing(self) -> None:
        self.assertEqual(notifier.detect_liq_burst({"available": True, "hawkes_active": False}), [])
        self.assertEqual(notifier.detect_liq_burst({"available": False}), [])


class RunCycleTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.sent: list[dict] = []
        self.tmpdir = Path(tempfile.mkdtemp())
        self._orig_state = notifier.STATE_PATH
        self._orig_broadcast = notifier.broadcast
        notifier.STATE_PATH = self.tmpdir / "state.json"

        async def fake_broadcast(payload, **kwargs):
            self.sent.append(payload)
            return {"sent": 1, "pruned": 0, "failed": 0}

        notifier.broadcast = fake_broadcast

    def tearDown(self) -> None:
        notifier.STATE_PATH = self._orig_state
        notifier.broadcast = self._orig_broadcast

    async def _cycle(self, state, data):
        async def fake_fetch(_session, _base):
            return data
        orig = notifier.fetch_all
        notifier.fetch_all = fake_fetch
        try:
            await notifier.run_cycle(None, "", state, private="k", subject="mailto:x@y.z",
                                     dry_run=False)
        finally:
            notifier.fetch_all = orig

    def _live_shadow(self):
        """지금 막 열린 포지션 -- EVENT_MAX_AGE_SEC 안에 들어오도록 현재 시각으로 만든다."""
        now = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime())
        return {"shadow": {"open_positions": [{"side": "long", "entry": 4400.0,
                                               "opened_utc": now, "proba": 0.71}],
                           "recent_trades": [], "n_open": 1}}

    async def test_first_run_sends_nothing_and_records_baseline(self) -> None:
        """재시작 폭주 방지 1단계. 이게 없으면 데몬을 껐다 켤 때마다 현재 상태 전부가 쏟아진다."""
        state = notifier.load_state()
        await self._cycle(state, self._live_shadow())
        self.assertEqual(self.sent, [])
        self.assertTrue(state["baseline_done"])
        self.assertTrue(state["seen"])

    async def test_second_run_sends_new_event(self) -> None:
        state = notifier.load_state()
        await self._cycle(state, {"shadow": {"open_positions": [], "recent_trades": []}})
        self.sent.clear()
        await self._cycle(state, self._live_shadow())
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["tier"], "t1")
        self.assertIn("진입", self.sent[0]["title"])

    async def test_same_event_is_not_resent_next_cycle(self) -> None:
        state = notifier.load_state()
        await self._cycle(state, {"shadow": {"open_positions": [], "recent_trades": []}})
        data = self._live_shadow()
        await self._cycle(state, data)
        self.sent.clear()
        await self._cycle(state, data)
        self.assertEqual(self.sent, [])

    async def test_stale_event_is_marked_seen_but_not_sent(self) -> None:
        """재시작 폭주 방지 2단계 -- 6시간 전에 끝난 일은 지금 알릴 가치가 없다."""
        state = notifier.load_state()
        await self._cycle(state, {"shadow": {"open_positions": [], "recent_trades": []}})
        self.sent.clear()
        old = time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(time.time() - 6 * 3600))
        await self._cycle(state, {"shadow": {"open_positions": [
            {"side": "long", "entry": 1.0, "opened_utc": old}], "recent_trades": []}})
        self.assertEqual(self.sent, [])
        self.assertIn(f"shadow_open:{old}", state["seen"])

    async def test_digest_sends_on_change_then_suppresses_when_unchanged(self) -> None:
        signals = [{"name": "demarker_extreme", "bottom_fired": True, "top_fired": False}]
        data = {"evidence": _evidence("T1", net=1, signals=signals),
                "regime": {"warmed_up": True, "bull_prob": 0.2, "bear_prob": 0.1, "chop_prob": 0.7},
                "shadow": {"n_open": 0}}
        state = notifier.load_state()
        await self._cycle(state, data)          # 기준선
        state["digest_sent_at"] = 0             # 최소 간격 통과시킴
        state["digest_fingerprint"] = None
        self.sent.clear()
        await self._cycle(state, data)
        self.assertEqual(len(self.sent), 1)
        self.assertEqual(self.sent[0]["tier"], "digest")
        self.assertIn("횡보", self.sent[0]["body"])
        self.sent.clear()
        state["digest_sent_at"] = 0             # 간격이 아니라 '변화 없음'으로 막히는지 확인
        await self._cycle(state, data)
        self.assertEqual(self.sent, [])

    async def test_digest_change_within_min_interval_is_deferred_not_dropped(self) -> None:
        """최소 간격 안의 변화는 버리지 않고, 간격이 지난 뒤 그때의 최신 상태로 나가야 한다."""
        base = {"regime": {"warmed_up": False}, "shadow": {"n_open": 0}}
        state = notifier.load_state()
        state["baseline_done"] = True
        state["digest_sent_at"] = time.time()   # 방금 보낸 상태
        await self._cycle(state, {**base, "evidence": _evidence(
            "T1", net=1, signals=[{"name": "a", "bottom_fired": True, "top_fired": False}])})
        self.assertEqual(self.sent, [])
        self.assertIsNone(state["digest_fingerprint"])  # 지문을 삼키지 않았다


if __name__ == "__main__":
    unittest.main()
