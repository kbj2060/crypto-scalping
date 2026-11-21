#!/usr/bin/env python3
"""rtserve 셋째 조각 정답지(2026-11-21): /api/session-alerts · /api/liquidation-5m-signal.

방식은 gen_http_fixture.py 와 같다(가짜 전송·시계 고정·미들웨어를 가져다 쓴다 · 서버를 띄우지 않는다). 원본 핸들러는 옮겨 적고(행 주석),
계산 함수는 모듈에서 그대로 부른다: scripts/live_macro_calendar_20260826.compute_macro_event_alert ·
scripts/live_liquidation_5m_signal_20260825.compute_liquidation_5m_signal(COIN_CONFIG 의 tail_risk 경로만 합성 트리로 바꿔 끼운다).
두 모듈 모두 import 때 .env·네트워크를 건드리지 않는다(확인함 -- 앞 모듈의 .env 읽기는 __main__ 안). tail_risk DuckDB 는 파이썬 duckdb 로 만든다.

    python3 rtserve/tools/gen_http3_fixture.py     # 저장소 루트에서 · rtserve/tests/fixtures/http3.json
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_http_fixture as g1  # noqa: E402
from aiohttp import web  # noqa: E402

MTIME_NS, ROOT = g1.MTIME_NS, g1.ROOT
NOW = g1.NOW + 1200   # 11:28:43 UTC -- 30분 봉(11:00)과 15분 봉(11:15)이 갈리는 자리(봉 길이를 지키는지)
import time  # noqa: E402
time.time = lambda: NOW
OUT = ROOT / "rtserve" / "tests" / "fixtures" / "http3.json"
sys.path.insert(0, str(ROOT / "scripts"))
mcal = g1.load("live_macro_calendar", "scripts/live_macro_calendar_20260826.py")
liq = g1.load("live_liq5m", "scripts/live_liquidation_5m_signal_20260825.py")
import duckdb  # noqa: E402


class _FixedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return datetime.fromtimestamp(NOW, tz)


mcal.datetime = _FixedDT
liq.datetime = _FixedDT
g1.worker_state.datetime = _FixedDT   # 달력 신선도도 같은 시계
NOCACHE = g1.NOCACHE
COIN_CONFIG = list(liq.COIN_CONFIG)


def make_app(R: Path) -> web.Application:
    LIVE_DIR = R / "data" / "live"
    for c, cfg in liq.COIN_CONFIG.items():            # 원본 경로(저장소 data/live) → 합성 트리 -- 파일 이름·표 이름은 그대로
        cfg["tail_risk_db_path"] = LIVE_DIR / Path(cfg["tail_risk_db_path"]).name
    app = web.Application(middlewares=[g1.static_asset_headers, g1.json_compress_etag])

    def load_macro_calendar():                                               # 3491 (swr 빼고)
        return g1.worker_state.worker_payload(LIVE_DIR / "macro_calendar_state.json", 90.0, ts_field="generated_at", extra_missing={"events": []})

    async def api_session_alerts(request):                                   # 4840
        macro_cal = load_macro_calendar()
        payload = {"macro_event_alert": mcal.compute_macro_event_alert(macro_cal.get("events", []))}
        return web.json_response(payload, headers=NOCACHE)

    def _query_coin_asset(request):                                          # 4742
        asset = request.query.get("asset", "eth").lower()
        if asset not in COIN_CONFIG:
            raise web.HTTPBadRequest(reason="unsupported_asset")
        return asset

    async def api_liquidation_5m_signal(request):                            # 4751 (swr 빼고)
        payload = liq.compute_liquidation_5m_signal(coin=_query_coin_asset(request))
        return web.json_response(payload, headers=NOCACHE)

    app.router.add_get("/api/session-alerts", api_session_alerts)
    app.router.add_get("/api/liquidation-5m-signal", api_liquidation_5m_signal)
    app.router.add_static("/dashboard/live/", R / "dashboard" / "live", show_index=False)
    return app


def duck(path: Path, tables: dict[str, list[tuple]]) -> None:
    con = duckdb.connect(str(path))
    for t, rows in tables.items():
        con.execute(f"CREATE TABLE {t} (ts TIMESTAMPTZ, long_usd_1m DOUBLE, short_usd_1m DOUBLE)")
        if rows:
            con.executemany(f"INSERT INTO {t} VALUES (to_timestamp(?), ?, ?)", rows)
    con.close()


def utc(t):
    return datetime.fromtimestamp(t, timezone.utc)


def tree(td: Path, kind: str) -> dict[str, bytes]:
    """DuckDB 는 파일로 만들어 바이트를 싣는다."""
    out: dict[str, bytes] = {}
    bar = int(NOW) // 1800 * 1800
    if kind == "main":
        tmp = td / "x"
        tmp.mkdir()
        # ETH: 지금 30분봉 안 8분 + 앞 봉 · 빈 값(NULL) 섞임 · 마이크로초 붙은 시각
        eth = [(bar - 600 + 60 * k + (0.123456 if k in (3, 17) else 0), 1234.5 * k if k % 3 else None, 77.25 + k / 7) for k in range(18)]
        duck(tmp / "tail_risk.duckdb", {"tail_risk_1m": eth})
        # BTC·SOL 한 파일 두 표 · SOL 은 지금 봉에 행 없음
        duck(tmp / "tail_risk_btc_sol.duckdb", {"tail_risk_1m_btc": [(bar + 60 * k, 0.1 * k, 1e6 / (k + 1)) for k in range(9)],
                                                "tail_risk_1m_sol": [(bar - 120, 5.0, 6.0), (bar - 60, 7.0, None)]})
        duck(tmp / "tail_risk_hype.duckdb", {"tail_risk_1m_hype": []})   # 빈 표 → no_rows
        for f in tmp.iterdir():
            out[f"data/live/{f.name}"] = f.read_bytes()
        cal = {"generated_at": utc(NOW - 600).strftime("%Y-%m-%dT%H:%M:%S"), "ok": True, "events": [
            {"importance": "high", "time_utc": utc(NOW + 600).isoformat(), "title_ko": "미국 CPI"},
            {"importance": "high", "time_utc": utc(NOW - 1799).strftime("%Y-%m-%dT%H:%M:%S+00:00"), "title_ko": "FOMC 의사록"},
            {"importance": "high", "time_utc": utc(NOW - 1801).isoformat(), "title_ko": "창 밖"},
            {"importance": "medium", "time_utc": utc(NOW).isoformat(), "title_ko": "EIA"},
            {"importance": "high", "time_utc": "엉터리", "title_ko": "x"},
            {"importance": "high", "title_ko": "시각 없음"},
            {"importance": "high", "time_utc": (utc(NOW + 30 * 60)).strftime("%Y-%m-%dT%H:%M:%S.%f") + "Z", "title_ko": "경계 +30분"}]}
        out["data/live/macro_calendar_state.json"] = json.dumps(cal, ensure_ascii=False).encode()
    else:
        cal = {"generated_at": utc(NOW - 9000).isoformat(), "events": [{"importance": "high", "time_utc": "2026-11-18T11:00:00", "title_ko": "시간대 없음"}]}
        out["data/live/macro_calendar_state.json"] = json.dumps(cal, ensure_ascii=False).encode()   # 낡음 + 시간대 없는 시각 → 원본은 TypeError(500)
    return out


REQ_MAIN = [("GET", p, {}) for p in ("/api/session-alerts", "/api/liquidation-5m-signal", "/api/liquidation-5m-signal?asset=BTC",
                                     "/api/liquidation-5m-signal?asset=sol", "/api/liquidation-5m-signal?asset=xrp",
                                     "/api/liquidation-5m-signal?asset=hype", "/api/liquidation-5m-signal?asset=doge")]
REQ_VAR = [("GET", "/api/session-alerts", {}), ("GET", "/api/liquidation-5m-signal", {})]


async def scenario(name, kind, reqs):
    with tempfile.TemporaryDirectory() as td:
        files = tree(Path(td), kind)
        R = Path(td) / "root"
        (R / "dashboard" / "live").mkdir(parents=True)
        for rel, data in files.items():
            p = R / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            os.utime(p, ns=(MTIME_NS, MTIME_NS))
        app = make_app(R)
        app.freeze()
        cases = []
        for method, path, headers in reqs:
            out = await g1.run_one(app, method, path, headers)
            cases.append({"method": method, "path": path, "req_headers": headers, **out})
    return {"name": name, "files": {k: base64.b64encode(v).decode() for k, v in files.items()}, "cases": cases}


async def main():
    out = {"now": NOW, "mtime_ns": MTIME_NS, "dashboard_assets": ",".join(g1.DASHBOARD_ASSETS), "duckdb": duckdb.__version__,
           "scenarios": [await scenario("calc", "main", REQ_MAIN), await scenario("calc_variants", "var", REQ_VAR)]}
    OUT.write_text(json.dumps(out, ensure_ascii=False) + "\n")
    for s in out["scenarios"]:
        for c in s["cases"]:
            print(f"  {s['name']:14} {c['path'][:52]:52} {c['status']} {base64.b64decode(c['body_b64'])[:220]}")


if __name__ == "__main__":
    asyncio.run(main())
