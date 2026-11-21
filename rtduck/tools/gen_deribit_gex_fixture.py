"""Deribit ②(10분 체인·옵션 요약·deribit_gex_state.json) 정답지: 원본 collect_deribit_option_gex_20260815 의 poll_once 를 가짜 requests 로 두 번(10분 간격).

체인 = 코인 넷(ETH·BTC 역 · SOL·XRP USDC 선형 · USDC 목록에 AVAX 섞임) · 끝난 만기 · iv 0 · 한쪽 호가 · 지수 실패(XRP → 체인 중앙값) · DVOL 빈 응답(BTC) ·
실현 7일 실패(BTC)·짧음(SOL)·0 값 · 두 번째 폴링은 실현 7일 캐시(6시간) · USDC 목록 캐시(60초 -- 같은 폴링 안 SOL·XRP) · OKX·Bybit 옵션(BTC OKX 실패).
상태 = vov24(지난 25시간 시간별 dvol/iv30) · atm_oi_pct(30일 07:00~07:10 UTC) -- 미리 넣은 option_summary 이력으로.
🔴DuckDB now() 는 진짜 시계다 -- 정답지 시각 NOW 를 «지금»으로 잡고 경계에서 멀리 둔다. 서버처럼 TZ=Asia/Seoul 로 돌린다(recorded_at_utc 글자의 +09:00).
  TZ=Asia/Seoul $PY rtduck/tools/gen_deribit_gex_fixture.py
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import random
import sys
import tempfile
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
fake_requests = types.ModuleType("requests")
sys.modules["requests"] = fake_requests
import collect_deribit_option_gex_20260815 as G  # noqa: E402
import duckdb  # noqa: E402

FIX = ROOT / "rtduck/tests/fixtures"
assert os.environ.get("TZ") == "Asia/Seoul", "TZ=Asia/Seoul 로 돌린다(서버와 같은 지역 시각)"
NOW = dt.datetime.fromtimestamp(int(time.time()) + 0.654321, dt.timezone.utc)
rnd = random.Random(1121)
MON = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]


def code(d: dt.date) -> str:
    return f"{d.day}{MON[d.month - 1]}{d.year % 100:02d}"


def chain_rows(prefix: str, spot: float, step: float, inverse: bool, fmt=lambda k: str(int(k))) -> list:
    rows = []
    days = [-1, 1, 3, 7, 14, 30, 61, 92]
    for dd in days:
        d = (NOW + dt.timedelta(days=dd)).date()
        fwd = spot * (1 + 0.0003 * dd)
        for j in range(-8, 9):
            k = round(spot + j * step, 6)
            for cp in ("C", "P"):
                iv = 0.0 if (j == 8 and cp == "C") else round(55 + 0.4 * j * j - 3 * j + rnd.uniform(-1, 1), 2)
                t = max(dd, 0.5) / 365
                sig = max(iv, 1) / 100
                d1 = (math.log(fwd / k) + 0.5 * sig * sig * t) / (sig * math.sqrt(t))
                n = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))   # noqa: E731
                c = fwd * n(d1) - k * n(d1 - sig * math.sqrt(t))
                px = c if cp == "C" else c - (fwd - k)
                px = max(px, 0.0) / (fwd if inverse else 1)
                bid = round(px * 0.97, 6) if j % 5 else 0.0
                rows.append({"instrument_name": f"{prefix}{code(d)}-{fmt(k)}-{cp}", "underlying_price": round(fwd, 4), "mark_iv": iv,
                             "open_interest": round(rnd.uniform(0, 900), 1) if j != 3 else 0, "mark_price": round(px, 6), "volume": round(rnd.uniform(0, 50), 1),
                             "bid_price": bid, "ask_price": round(px * 1.03, 6) if j % 7 else None})
    return rows


BOOK = {"ETH": chain_rows("ETH-", 2600.0, 100.0, True), "BTC": chain_rows("BTC-", 95000.0, 2500.0, True),
        "USDC": chain_rows("SOL_USDC-", 190.0, 5.0, False) + chain_rows("XRP_USDC-", 2.1, 0.05, False, fmt=lambda k: f"{k:.2f}".replace(".", "d").rstrip("0").rstrip("d"))
        + chain_rows("AVAX_USDC-", 30.0, 1.0, False)[:6]}


class Resp:
    def __init__(self, body, status=200):
        self.b, self.status_code = body, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTPError {self.status_code}")

    def json(self):
        return self.b


CALLS: list = []
T = {"now": NOW.timestamp()}


def get(url, params=None, timeout=None, headers=None):
    r = _get(url, params, timeout, headers)
    CALLS.append([url, dict(params or {}), timeout, dict(headers or {}), r.status_code, r.b])
    return r


def _get(url, params=None, timeout=None, headers=None):
    p = params or {}
    if url == G.BASE_URL:
        return Resp({"result": BOOK[p["currency"]]})
    if url.endswith("get_index_price"):
        if p["index_name"] == "xrp_usdc":
            return Resp({}, 500)
        return Resp({"result": {"index_price": {"eth_usd": 2601.25, "btc_usd": 95010.5, "sol_usdc": 190.3}[p["index_name"]]}})
    if url.endswith("get_volatility_index_data"):
        data = [] if p["currency"] == "BTC" else [[p["start_timestamp"] + i * 3_600_000, 50, 52, 49, 51.25 + i] for i in range(2)]
        return Resp({"result": {"data": data}})
    if url.endswith("get_tradingview_chart_data"):
        inst = p["instrument_name"]
        if inst.startswith("BTC"):
            return Resp({}, 503)
        n = 20 if inst.startswith("SOL") else 169
        cl = [round(100 * math.exp(0.01 * math.sin(i / 3) + 0.002 * i), 4) for i in range(n)]
        cl[5] = 0
        cl[9] = None
        return Resp({"result": {"close": cl}})
    if "okx.com" in url and "open-interest" in url:
        if p["instFamily"] == "BTC-USD":
            return Resp({}, 500)
        d1, d2 = (NOW + dt.timedelta(days=7)).date(), (NOW - dt.timedelta(days=1)).date()
        return Resp({"data": [{"instId": f"ETH-USD-{d1:%y%m%d}-3000-C", "oiCcy": "12.5"}, {"instId": f"ETH-USD-{d1:%y%m%d}-2500.5-P", "oiCcy": "3"},
                              {"instId": f"ETH-USD-{d2:%y%m%d}-2600-C", "oiCcy": "9"}, {"instId": f"ETH-USD-{d1:%y%m%d}-2700-C", "oiCcy": "0"}, {"instId": "BAD", "oiCcy": "1"}]})
    if "okx.com" in url:
        d1 = (NOW + dt.timedelta(days=7)).date()
        return Resp({"data": [{"instId": f"ETH-USD-{d1:%y%m%d}-3000-C", "markVol": "0.5512"}, {"instId": "x", "markVol": ""}]})
    if "bybit" in url:
        d1 = (NOW + dt.timedelta(days=14)).date()
        return Resp({"result": {"list": [{"symbol": f"{p['baseCoin']}-{code(d1)}-3000-C-USDT", "openInterest": "4.5", "markIv": "0.61", "volume24h": "12"},
                                         {"symbol": f"{p['baseCoin']}-{code(d1)}-2800-P", "openInterest": "1", "markIv": "", "volume24h": None},
                                         {"symbol": f"{p['baseCoin']}-{code(d1)}-2900-P", "openInterest": "0", "markIv": "0.5"}]}})
    raise AssertionError(url)


fake_requests.get = get


class FakeDT(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return dt.datetime.fromtimestamp(T["now"], dt.timezone.utc)


G.datetime = FakeDT
G.time = types.SimpleNamespace(time=lambda: T["now"], sleep=lambda s: None)


def seed(con) -> list:
    """지난 30시간 시간별(ETH dvol · SOL iv30) + 지난 10일 07:05 UTC(front_atm_oi_usd) 이력."""
    rows = []
    for k in range(1, 31):
        t = NOW - dt.timedelta(seconds=k * 3600 + 1234)
        rows.append((t, "ETH", 2600.0, 50 + 2 * math.sin(k / 2), None, None, None, None, None, json.dumps({"iv30": 60.0 + k, "front_atm_oi_usd": 1e6 + k})))
        rows.append((t, "SOL", 190.0, None, None, None, None, None, None, json.dumps({"iv30": 70.0 + math.cos(k), "front_atm_oi_usd": None})))
    for k in range(1, 11):
        day = (NOW - dt.timedelta(days=k)).date()
        t = dt.datetime(day.year, day.month, day.day, 7, 5, k, tzinfo=dt.timezone.utc)
        rows.append((t, "ETH", 2600.0, 48.0, None, None, None, None, None, json.dumps({"iv30": 61.0, "front_atm_oi_usd": 5e5 * k})))
    con.executemany("INSERT INTO option_summary VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    return [[int(r[0].timestamp() * 1_000_000), *r[1:]] for r in rows]


def main() -> None:
    with tempfile.TemporaryDirectory() as td:
        G.STATE_PATH = Path(td) / "deribit_gex_state.json"
        con = duckdb.connect(str(Path(td) / "d.duckdb"))
        G.ensure_tables(con)
        hist = seed(con)
        out = []
        for k in range(2):
            T["now"] = NOW.timestamp() + 600 * k
            CALLS.clear()
            G.poll_once(con)
            db_now = con.execute("SELECT epoch_us(now())").fetchone()[0]
            out.append({"calls": list(CALLS), "state": G.STATE_PATH.read_text(), "db_now_us": db_now})
        dump = {}
        for t in ("option_chain_snapshot", "option_summary", "option_oi_other"):
            cols = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
            types_ = [r[1] for r in con.execute(f"DESCRIBE {t}").fetchall()]
            sel = ", ".join(f"epoch_us({c})" if "TIMESTAMP" in ty else (f"CASE WHEN isnan({c}) THEN 'NaN' ELSE {c}::VARCHAR END" if ty == "DOUBLE" else c) for c, ty in zip(cols, types_))
            dump[t] = [list(r) for r in con.execute(f"SELECT {sel} FROM {t} ORDER BY ALL").fetchall()]
        con.close()
    (FIX / "deribit_gex_input.json").write_text(json.dumps({"now_s": NOW.timestamp(), "book": BOOK, "hist": hist, "tz_offset_s": 9 * 3600}))
    (FIX / "deribit_gex_golden.json").write_text(json.dumps({"polls": out, "dump": dump}, ensure_ascii=False))
    st = json.loads(out[-1]["state"])
    print("요청", [len(o["calls"]) for o in out], "· 코인", list(st["currencies"]), "· opt_hist", {c: v["opt_hist"] for c, v in st["currencies"].items()},
          "· 표", {k: len(v) for k, v in dump.items()}, "· db now 차이 초", [(o["db_now_us"] / 1e6 - NOW.timestamp()) for o in out])


if __name__ == "__main__":
    main()
