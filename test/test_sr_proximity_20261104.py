"""지지·저항 «가까워짐» 알림 판정(scripts/live_sr_proximity_alert_20261104.py::decide) + 서버 sr_watch_view. 네트워크·텔레그램 0.
python -m pytest -q test/test_sr_proximity_20261104.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dashboard.server import sr_watch_view  # noqa: E402
from scripts import live_sr_proximity_alert_20261104 as w  # noqa: E402

T0 = 1_000_000.0


def V(px, sup=(), res=(), t=T0):   # ETH: 기준 20bp = 현재가 ±0.20%
    return {"eth": {"px": px, "t": t, "dp": 2, "sup": list(sup), "res": list(res)}}


class Proximity(unittest.TestCase):
    def test_alerts_once_when_far_line_gets_close_then_rearms_after_moving_away(self):
        st = {}
        self.assertEqual(w.decide(st, V(2300, sup=[2273.85]), T0), [])                 # 처음 본 레벨(−1.1%, 멀다) = 무장만
        a = w.decide(st, V(2280, sup=[2273.85]), T0 + 10)                               # −0.27% = 아직 기준(0.20%) 밖
        self.assertEqual(a, [])
        a = w.decide(st, V(2277, sup=[2273.85]), T0 + 20)                               # −0.14% = 기준 안 → 알림
        self.assertEqual([(x["name"], x["level"]) for x in a], [("지지1", 2273.85)])
        self.assertEqual(w.decide(st, V(2276, sup=[2273.85]), T0 + 30), [])             # 계속 가까워도 반복 없음
        self.assertEqual(w.decide(st, V(2330, sup=[2273.85]), T0 + 40), [])             # 멀어짐(−2.4% ≥ 2×기준) = 다시 무장
        self.assertEqual(w.decide(st, V(2277, sup=[2273.85]), T0 + 50), [])             # 다시 가까워졌지만 쿨다운(30분) 안
        a = w.decide(st, V(2277, sup=[2273.85], t=T0 + w.COOLDOWN_S + 60), T0 + w.COOLDOWN_S + 60)
        self.assertEqual(len(a), 1)                                                     # 쿨다운 뒤엔 다시 알림

    def test_first_sight_already_close_is_silent_and_resistance_names(self):
        st = {}
        self.assertEqual(w.decide(st, V(2300, res=[2301.0]), T0), [])                   # 재시작·새 청산맵에서 이미 가까운 레벨 = 조용히
        self.assertEqual(w.decide(st, V(2300, res=[2301.0]), T0 + 10), [])
        w.decide(st, V(2200, res=[2301.0]), T0 + 20)                                    # 멀어졌다가(−4.6%)
        a = w.decide(st, V(2299, res=[2301.0], t=T0 + w.COOLDOWN_S + 30), T0 + w.COOLDOWN_S + 30)
        self.assertEqual([x["name"] for x in a], ["저항1"])

    def test_stale_price_or_unknown_asset_threshold(self):
        st = {}
        w.decide(st, V(2300, sup=[2200]), T0)
        self.assertEqual(w.decide(st, V(2201, sup=[2200], t=T0 - 10_000), T0), [])      # 1초 체결가가 멈춤 = 알리지 않는다
        v = {"zzz": {"px": 100.0, "t": T0, "dp": 2, "sup": [90.0]}}
        w.decide(st, v, T0)
        v["zzz"]["px"] = 90.2                                                            # 기본 기준 25bp: −0.22% → 알림
        self.assertEqual(len(w.decide(st, v, T0 + 10)), 1)

    def test_forgets_levels_and_message_is_descriptive(self):
        st = {}
        w.decide(st, V(2300, sup=[2200]), T0)
        w.decide(st, {}, T0 + w.FORGET_S + 1)
        self.assertEqual(st, {})
        a = w.decide({}, V(2300, sup=[2200]), T0)
        self.assertEqual(a, [])
        st = {}
        w.decide(st, V(2300, sup=[2273.85]), T0)
        a = w.decide(st, V(2277, sup=[2273.85]), T0 + 10)
        m = w.message("eth", a)
        self.assertIn("ETH 지지1 가까워짐", m)
        self.assertIn("2,273.85", m)
        self.assertNotIn("매수", m)   # 알림은 매매 지시가 아니다

    def test_server_view_matches_screen_rule(self):
        # 현재가 쪽 · 안 쓸린 · 가까운 순 n개
        self.assertEqual(sr_watch_view([2273.0, 2257.0, 2240.0, 2211.0], [2317.0, 2319.0, 2335.0, 2358.0], 2290.0, {2257.0}),
                         {"sup": [2273.0, 2240.0, 2211.0], "res": [2317.0, 2319.0, 2335.0]})
        self.assertEqual(sr_watch_view([2300.0], [2280.0], 2290.0, set()), {"sup": [], "res": []})   # 가격이 넘어간 레벨은 그 쪽 목록에서 빠진다


class MovedLoops(unittest.TestCase):
    """2026-11-28 대시보드에서 옮겨 온 sr_break_loop·chip_score_loop 한 바퀴(가짜 응답 서버 · 텔레그램 0)."""

    def test_sr_tick_baseline_then_break_message_and_watch_file(self):
        import tempfile
        from dashboard import server as srv
        lm = {"warmed_up": True, "heatmap_history": [{"ts_utc": "2026-11-28T10:00:00Z"}],
              "support_levels": [{"price": 2273.0}, {"price": 2257.0}], "resistance_levels": [{"price": 2317.0}]}
        secs = [[[100, 0, 0, 0, 0, 0, 0, 2290.0]], [[100, 0, 0, 0, 0, 0, 0, 2290.0], [101, 0, 0, 0, 0, 0, 0, 2270.0], [102, 0, 0, 0, 0, 0, 0, 2275.0]]]
        calls, sent = [], []

        def get(path):
            calls.append(path)
            return lm if path.startswith("/api/liquidation-map") else {"seconds": secs[0] if len(sent) == 0 and len(calls) <= 2 else secs[1]}
        with tempfile.TemporaryDirectory() as d:
            w.SR_WATCH_PATH = Path(d) / "sr_watch.json"
            st, watch = {}, {}
            w.sr_tick(srv, ["eth"], st, watch, get=get, send=lambda m: sent.append(m) or True)
            self.assertEqual(sent, [])                                   # 첫 바퀴 = 기준만(원본과 같다)
            self.assertIn("since=0", calls[1])
            w.sr_tick(srv, ["eth"], st, watch, get=get, send=lambda m: sent.append(m) or True)
            self.assertIn("since=100", calls[3])                         # 커서 = 지난번 마지막 초
            self.assertEqual(sent, ["🔻 <b>ETH 지지1 뚫림</b> 2,273.00 · 다음 지지 2,257.00 (-0.79%)\n지금 2,275.00"])
            import json
            self.assertEqual(json.loads(w.SR_WATCH_PATH.read_text())["eth"],
                             {"px": 2275.0, "t": 102, "dp": 2, "ku": "2026-11-28T10", "sup": [2257.0], "res": [2317.0]})

    def test_chip_tick_burst_row_after_15min(self):
        import json
        import tempfile
        from dashboard import server as srv
        thr = srv.LIQ_BURST_60S_USD[0]
        mc = {"available": True, "mid": 2000.0, "burst": {"long_usd_60s": thr + 1, "short_usd_60s": 0}}
        with tempfile.TemporaryDirectory() as d:
            orig = srv.chip_score_path
            srv.chip_score_path = lambda ts: Path(d) / "c.jsonl"
            try:
                cs = {}
                w.chip_tick(srv, cs, 1000.0, get=lambda p: mc)
                mc = {"available": True, "mid": 2010.0, "burst": {"long_usd_60s": 0, "short_usd_60s": 0}}
                w.chip_tick(srv, cs, 1899.0, get=lambda p: mc)
                self.assertFalse((Path(d) / "c.jsonl").exists())          # 15분 전엔 안 쓴다
                w.chip_tick(srv, cs, 1900.0, get=lambda p: mc)
                row = json.loads((Path(d) / "c.jsonl").read_text())
                self.assertEqual(row, {"k": "burst", "a": "eth", "t": 1000, "side": "long", "usd": round(thr + 1), "rev_bp": 50.0})
            finally:
                srv.chip_score_path = orig


if __name__ == "__main__":
    unittest.main()
