"""«Macro Trend» 카드 -- dashboard/macro_trend.py 와 /api/macro-trend (2026-10-28 사용자 «대시보드 카드 하나 만들어서 모두 추가»).

🔴바깥 요청 0: 원천(FRED·DefiLlama·바이낸스)은 전부 가짜로 막는다.
검사:
  1. 판정표 로드 -- «사용 가능»·«관찰»만, KOSPI(원천 없음) 제외, USD/JPY 선행은 BTC·ETH 둘 다 «사용 가능», 연구 숫자가 표와 같다
  2. 계산 -- 선행 관계를 심은 합성 데이터에서 감시 «유효», 반대 부호를 심으면 «이탈», 고정된 지표는 «변동 없음», 미완결 주 제외
  3. 응답 -- NaN 없음(브라우저 JSON.parse), 원천 하나 실패해도 나머지로 ok, 요약은 «사용 가능 · 유효 · 변동 있음»만 센다
  4. 라우트 -- 200 · 1시간 캐시(두 번째 호출은 다시 계산 안 함) · 계산이 예외를 내도 200 + ok False
실행: python -m pytest -q test/test_macro_trend_card_20261028.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard import macro_trend as mt  # noqa: E402

NOW = pd.Timestamp("2026-10-25").timestamp()   # 일요일 -- 10-23(금) 주가 마지막 완결 주
DAYS = pd.date_range("2022-01-03", "2026-10-23", freq="D")


def synth(seed: int = 3, lead_sign: float = 1.0):
    """BTC 가 지표 L 을 13주(91일) 늦게 따라간다(lead_sign 부호로). F = 마지막 300일 고정."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0, 1, len(DAYS)))
    btc = pd.Series(np.exp(10 + lead_sign * 0.02 * np.r_[np.zeros(91), f[:-91]] + 0.002 * np.cumsum(rng.normal(0, 1, len(DAYS)))), index=DAYS)
    walk = np.exp(5 + 0.01 * np.cumsum(rng.normal(0, 1, len(DAYS) - 300)))
    return {"BTC": btc, "ETH": btc * 0.05, "L": pd.Series(np.exp(4 + 0.02 * f), index=DAYS),
            "F": pd.Series(np.r_[walk, np.full(300, walk[-1])], index=DAYS)}


def spec_row(key, kind, src, r):
    return {"key": key, "name": key, "kind": kind, "source": f"fred:{src}", "transform": "logpct", "unit": "%", "realtime": "free",
            "realtime_note": "", "note": "", "assets": {"BTC": {"verdict": "사용 가능", "r_p3": r}}}


class SpecTest(unittest.TestCase):
    def test_loads_usable_and_watch_rows_only(self):
        spec = mt.load_spec()
        names = {(g["name"], g["kind"]) for g in spec}
        self.assertIn(("USD/JPY", "lead"), names)
        self.assertNotIn(("KOSPI", "co"), names)          # FRED 에 없어 뺀다
        jpy = next(g for g in spec if g["name"] == "USD/JPY" and g["kind"] == "lead")
        self.assertEqual({a: v["verdict"] for a, v in jpy["assets"].items()}, {"BTC": "사용 가능", "ETH": "사용 가능"})
        with open(mt.VERDICT_CSV, newline="", encoding="utf-8") as f:
            src = next(r for r in csv.DictReader(f) if r["asset"] == "BTC" and r["kind"] == "선행13" and r["indicator"] == "USD/JPY")
        self.assertAlmostEqual(jpy["assets"]["BTC"]["r_p3"], float(src["r_p3"]), places=6)   # 연구 숫자 = 표 그대로
        for g in spec:
            for v in g["assets"].values():
                self.assertIn(v["verdict"], ("사용 가능", "관찰"))
        hy = next(g for g in spec if g["name"] == "HY스프레드")
        self.assertIsNone(hy["assets"]["BTC"]["r_2020_23"])   # 2020–23 데이터 없음 → None(빈 칸이 float 오류를 내던 자리)


class ComputeTest(unittest.TestCase):
    def test_watch_valid_breaks_and_stale(self):
        spec = [spec_row("L", "lead", "L", 0.5), spec_row("F", "co", "F", -0.4)]
        p = mt.build_payload(spec, synth(), {}, NOW)
        lead, co = p["rows"][0]["live"], p["rows"][1]["live"]
        self.assertEqual(lead["week"], "2026-10-23")
        self.assertEqual(lead["assets"]["BTC"]["watch"], "유효")
        self.assertTrue(co["stale"])
        # 같은 데이터인데 연구 부호가 반대라고 하면 «이탈»
        p2 = mt.build_payload([spec_row("L", "lead", "L", -0.5)], synth(), {}, NOW)
        self.assertEqual(p2["rows"][0]["live"]["assets"]["BTC"]["watch"], "이탈")
        # 편향 = 변화 부호 × 연구 부호 (|z|<0.5 중립)
        z, chg = lead["z"], lead["chg13"]
        self.assertEqual(lead["assets"]["BTC"]["bias"], 0 if abs(z) < mt.Z_NEUTRAL else int(np.sign(chg)))

    def test_incomplete_week_dropped(self):
        w = mt.weekly(synth()["BTC"], pd.Timestamp("2026-10-22"))   # 목요일 -- 10-23 주는 아직 안 끝났다
        self.assertEqual(w.index[-1], pd.Timestamp("2026-10-16"))

    def test_fred_csv_parser(self):
        s = mt.parse_fred_csv("DATE,X\n2024-01-01,1.5\n2024-01-02,.\n2024-01-03,2\n")
        self.assertEqual(list(s.values), [1.5, 2.0])

    def test_payload_json_safe_and_partial_failure(self):
        spec = [spec_row("L", "lead", "L", 0.5), spec_row("G", "co", "G", 0.3)]   # G 원천 없음
        p = mt.json_safe(mt.build_payload(spec, synth(), {"G": "HTTPError: 503"}, NOW))
        json.dumps(p, allow_nan=False)
        self.assertTrue(p["ok"])
        self.assertIsNone(p["rows"][1]["live"])
        self.assertEqual(p["errors"], {"G": "HTTPError: 503"})

    def test_summary_counts_only_valid(self):
        spec = [spec_row("L", "lead", "L", 0.5), spec_row("L2", "lead", "L", -0.5), spec_row("F", "co", "F", -0.4)]
        p = mt.build_payload(spec, synth(), {}, NOW)
        s = p["summary"]["BTC"]
        self.assertEqual(s["excluded"], 2)                 # 이탈 1 + 변동 없음 1
        self.assertEqual(s["lead_up"] + s["lead_dn"] + s["neutral"], 1)

    def test_fetch_all_no_network_and_failure_isolated(self):
        spec = mt.load_spec()
        calls = []

        def fake_get(url, params=None, timeout=None, headers=None):
            calls.append(url)
            r = mock.Mock()
            r.raise_for_status = mock.Mock()
            if "fredgraph" in url:
                if "DEXJPUS" in url:
                    raise ConnectionError("down")
                r.text = "observation_date,X\n" + "\n".join(f"{d.date()},{1 + i * 0.001}" for i, d in enumerate(DAYS))
            elif "llama" in url:
                r.json = mock.Mock(return_value=[{"date": str(int(d.timestamp())), "totalCirculatingUSD": {"peggedUSD": 1e11 + i}} for i, d in enumerate(DAYS)])
            else:
                r.json = mock.Mock(return_value=[[int(d.timestamp() * 1000), 0, 0, 0, str(100 + i), 0, int(d.timestamp() * 1000) + 86399999]
                                                 for i, d in enumerate(DAYS)])
            return r

        with mock.patch.object(mt.requests, "get", side_effect=fake_get):
            raw, errors = mt.fetch_all(spec)
        self.assertIn("DEXJPUS", errors)
        self.assertIn("BTC", raw)
        self.assertIn("stables", raw)
        self.assertTrue(all(("stlouisfed" in u) or ("llama" in u) or ("binance" in u) for u in calls))


class RouteTest(unittest.TestCase):
    def test_route_cache_and_failure(self):
        from aiohttp.test_utils import TestClient, TestServer
        from offline_app import offline_app, patch_live_dir
        from dashboard import server

        n = {"calls": 0}

        def fake_payload():
            n["calls"] += 1
            return {"ok": True, "rows": [], "summary": {}, "errors": {}, "asof": 1}

        async def run(fn):
            with tempfile.TemporaryDirectory() as d, patch_live_dir(server, Path(d)), mock.patch.object(server.macro_trend, "payload", fn):
                async with TestClient(TestServer(offline_app(server))) as c:
                    r1 = await c.get("/api/macro-trend")
                    j1 = await r1.json()
                    r2 = await c.get("/api/macro-trend")
                    return r1.status, j1, r2.status

        s1, j1, s2 = asyncio.run(run(fake_payload))
        self.assertEqual((s1, s2), (200, 200))
        self.assertTrue(j1["ok"])
        self.assertEqual(n["calls"], 1)                  # 1시간 캐시

        def boom():
            raise FileNotFoundError("L_p3_trend_verdict.csv")
        s1, j1, _ = asyncio.run(run(boom))
        self.assertEqual(s1, 200)
        self.assertFalse(j1["ok"])
        self.assertIn("FileNotFoundError", j1["error"])


if __name__ == "__main__":
    unittest.main()
