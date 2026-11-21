#!/usr/bin/env python3
"""미시구조(microstructure_scanner.py) 일치 시험 (2026-11-21).

원본 `MicrostructureScanner` 의 `_db_init`(부트스트랩)·`compute_signal`·`_db_insert`·`_poll_recent_agg_trades` 는 그대로 부르고, 루프 몸통
(_depth_loop 메시지 :294-301 · _trade_loop 메시지 :331-341 · _poll_loop 한 바퀴 :402-442 · _scan_loop 한 바퀴 :836-854)만 글자 그대로 옮겨
사건 열로 돈다(가짜 시계 · 가짜 aiohttp 세션 = 요청 URL 기록 + 정해 둔 응답). 10초 틱 = 경계 + 0.05초.
🔴서버와 같게: TZ=Asia/Seoul(분 저장 = 지역 시각 datetime.now() · DuckDB 가 지역 시각을 TIMESTAMPTZ 로 바꾼다) · NPY_DISABLE_CPU_FEATURES=X86_V4.
부트스트랩: DuckDB now() 를 쓰므로 T0 = 만들 때의 실제 시각(초 내림) -- 씨앗 행은 T0 기준 60분·5분 경계에서 20초 넘게 떨어져 있다. Rust 는 T0 를 주입.
종목 ETH(빈 표 → 씨앗 → 부트스트랩) · BTC(빈 표 부트스트랩). 합성 자료만 · 네트워크 없음.

    TZ=Asia/Seoul NPY_DISABLE_CPU_FEATURES=X86_V4 python3 rtduck/tools/gen_micro_fixture.py
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import math
import os
import random
import sys
import tempfile
import time
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "rtduck" / "tests" / "fixtures"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "rtcollect" / "tools"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
assert "X86_V4" in os.environ.get("NPY_DISABLE_CPU_FEATURES", ""), "NPY_DISABLE_CPU_FEATURES=X86_V4"
assert os.environ.get("TZ") == "Asia/Seoul", "TZ=Asia/Seoul(서버 시간대)"
time.tzset()
assert time.localtime().tm_gmtoff == 9 * 3600
import gen_venue_fixture as gv  # noqa: E402
import duckdb  # noqa: E402
from duckdump import dump  # noqa: E402

PX = {"ethusdt": 2500.0, "btcusdt": 90000.0}


def gen(rng, t0: float) -> dict:
    seeds = []                                            # ETH 씨앗 행(분 정각 · T0 기준 -75분 ~ -2분)
    for k in range(75, 1, -1):
        if k in (60, 5):
            continue
        ts = (int(t0) // 60 - k) * 60
        if abs((t0 - ts) - 3600) < 20 or abs((t0 - ts) - 300) < 20:
            continue
        seeds.append([ts * 1_000_000, rng.uniform(-0.6, 0.6), rng.gauss(0, 3e-4), rng.random(), rng.random(), rng.choice((-1, 0, 1)),
                      rng.uniform(0, 4), None if k % 17 == 0 else PX["ethusdt"] + rng.gauss(0, 5)])
    ev = []
    for sym in PX:
        mid, t = PX[sym], t0
        oi = 1.5e6 if sym == "ethusdt" else 9e4
        ev += [[t0 + 0.1, "do", sym, ""], [t0 + 0.2, "to", sym, ""]]
        tid = 1000
        for s in range(0, 1500 if sym == "ethusdt" else 300, 2):   # ETH 25분 · BTC 5분
            t = t0 + 1 + s
            mid += rng.gauss(0, mid * 2e-4)
            if s % 6 == 0:                                # 호가 12단(원본은 20단 -- 앞 10단만 쓴다)
                bids = [[f"{mid - 0.1 * (k + 1):.2f}", f"{rng.uniform(0.1, 60) * (3 if s % 400 < 40 else 1):.2f}"] for k in range(12)]
                asks = [[f"{mid + 0.1 * (k + 1):.2f}", f"{rng.uniform(0.1, 60):.2f}"] for k in range(12)]
                ev.append([t + 0.01, "D", sym, json.dumps({"e": "depthUpdate", "b": bids, "a": asks})])
            if not (600 <= s < 700):                    # 100초 체결 웹소켓이 조용 → 폴링이 aggTrades 로 메운다
                for _ in range(rng.randint(0, 2)):
                    big = rng.random() < 0.08
                    usd = rng.uniform(1.2e5, 6e5) if big else rng.lognormvariate(8, 1.5)
                    tid += 1
                    ev.append([t + rng.uniform(0.02, 1.9), "T", sym, json.dumps({"e": "aggTrade", "a": tid, "p": f"{mid:.2f}", "q": f"{usd / mid:.4f}",
                                                                                  "T": int((t + 0.01) * 1000), "m": rng.random() < (0.3 if s < 700 else 0.7)})])
            if s % 10 == 4:                               # 폴링 10초
                oi *= 1 + rng.gauss(0, 4e-4) + (3e-3 if 800 <= s < 900 else 0)
                fr = 0.0009 if 800 <= s < 900 else 0.0012 if 900 <= s < 1000 else rng.gauss(1e-4, 5e-5)   # 0.0009 = squeeze 문턱(0.0008) 바로 위
                resp = {"oi": {"symbol": sym.upper(), "openInterest": f"{oi:.3f}", "time": int(t * 1000)},
                        "fund": {"symbol": sym.upper(), "markPrice": f"{mid:.2f}", "lastFundingRate": f"{fr:.8f}"},
                        "agg": [{"a": tid + j, "p": f"{mid:.2f}", "q": f"{rng.uniform(0.5, 90):.3f}", "T": int((t - 5 + j) * 1000), "m": j % 2 == 0} for j in range(5)]}
                if s == 204:
                    resp["oi"] = {"code": -1003, "msg": "Too many requests"}          # 상태 코드는 안 본다 -- openInterest 없음 = 0
                if s == 404:
                    resp["oi"] = "<html>"                                            # JSON 아님 = 그 바퀴 예외
                if s == 1104:
                    resp["fund"] = {"lastFundingRate": "0.0001"}                     # markPrice 없음 = 0 → 가격 안 쌓음
                ev.append([t + 0.3, "P", sym, json.dumps(resp)])
            if s == 900:
                ev.append([t + 0.4, "de", sym, ""])
                ev.append([t + 8.4, "do", sym, ""])
    ev.sort(key=lambda e: (e[0], e[2]))
    return {"t0": t0, "seeds": seeds, "events": ev}


class FakeSession:
    def __init__(self, resp, sent):
        self.resp, self.sent = resp, sent

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    def get(self, url, timeout=None):
        self.sent.append(url)
        key = "oi" if "openInterest" in url else "fund" if "premiumIndex" in url else "agg"
        body = self.resp[key]

        class R:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                return False

            async def json(self, content_type=None):
                if isinstance(body, str):
                    raise ValueError("JSON 아님")
                return body
        return R()


def seed_sql(table):
    return f"""INSERT INTO {table} (ts, nif_whale, oi_delta_pct, shadow_absorption_score, shadow_toxicity_score, shadow_queue_bias, eai, mark_price)
               VALUES (make_timestamptz(?),?,?,?,?,?,?,?)"""


def python_reference(inp, d: Path) -> dict:
    m = __import__("microstructure_scanner")
    clock = [inp["t0"]]

    class FakeDT(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return dt.datetime.fromtimestamp(clock[0], tz=tz)

    m.time = gv._ns(time, time=lambda: clock[0])
    m.datetime = FakeDT
    db = d / "microstructure.duckdb"
    os.environ["QUANT_MICRO_DB_PATH"] = str(db)
    first = m.MicrostructureScanner("ethusdt")
    first._db_init()                                     # 빈 표를 만든다
    con = duckdb.connect(str(db))
    con.executemany(seed_sql("microstructure_1m"), inp["seeds"])
    con.close()
    sc = {s: m.MicrostructureScanner(s) for s in PX}
    boot = {}
    for s, c in sc.items():
        c._db_init()
        boot[s] = c._bootstrap_loaded_rows
    sent, sigs, last_min, full = [], [], {s: -1 for s in PX}, []

    def depth(c, raw):                                   # :294-301
        self = c
        try:
            msg = json.loads(raw)
            self._depth_last_msg_ts = m.time.time()
            bids = [(float(p), float(q)) for p, q in msg.get("b", [])]
            asks = [(float(p), float(q)) for p, q in msg.get("a", [])]
            self._bids, self._asks = bids, asks
        except Exception:
            pass

    def trade(c, raw):                                   # :331-341
        self = c
        try:
            msg = json.loads(raw)
            self._trade_last_msg_ts = m.time.time()
            is_buyer_maker = bool(msg.get("m", False))
            qty_usd = float(msg.get("q", 0.0)) * float(msg.get("p", 0.0))
            ts_ms = int(msg.get("T", 0))
            if qty_usd > 0:
                self._trades.append((ts_ms, is_buyer_maker, qty_usd))
                self._trade_notional_qty.append((ts_ms, qty_usd, float(msg.get("q", 0.0))))
        except Exception:
            pass

    async def poll(c, resp):                             # :402-442 (세션 = 가짜 · 잠·로그 뺌)
        self = c
        oi_url = m._OI_URL.format(SYMBOL=self.symbol.upper())
        fund_url = m._FUND_URL.format(SYMBOL=self.symbol.upper())
        timeout = None
        try:
            async with FakeSession(resp, sent) as session:
                self._poll_connected = True
                async with session.get(oi_url, timeout=timeout) as r:
                    data = await r.json(content_type=None)
                    oi = float(data.get("openInterest", 0.0))
                    now_ms = int(m.time.time() * 1000)
                    now_s = m.time.time()
                    prev_tick = self._last_oi_tick if self._last_oi_tick is not None else oi
                    self._last_oi_tick = oi
                    self._oi_obs.append((now_ms, oi))
                    oi_delta_tick = float((oi - prev_tick) / (abs(prev_tick) + 1e-8))
                    self._cached["oi_delta_pct"] = oi_delta_tick
                    bucket_5m = int(now_s // 300)
                    if bucket_5m != self._oi_delta_5m_bucket:
                        self._oi_delta_5m_bucket = bucket_5m
                        self._oi_delta_cum_5m = 0.0
                    self._oi_delta_cum_5m = (1.0 + self._oi_delta_cum_5m) * (1.0 + oi_delta_tick) - 1.0
                    self._cached["oi_delta_cum_5m"] = float(self._oi_delta_cum_5m)
                    self._cached["oi_delta_cum_5m_bucket_start_ts"] = int(bucket_5m * 300)
                async with session.get(fund_url, timeout=timeout) as r:
                    data = await r.json(content_type=None)
                    self._fund_rate = float(data.get("lastFundingRate", 0.0))
                    mark = float(data.get("markPrice", 0.0))
                    if mark > 0:
                        self._price_obs.append((now_ms, mark))
                    bucket_min = int(m.time.time() // 60)
                    if self._last_hist_bucket_min != bucket_min:
                        self._last_hist_bucket_min = bucket_min
                        if oi > 0:
                            self._oi_hist.append(oi)
                        if mark > 0:
                            self._price_hist.append(mark)
                if m.time.time() - self._trade_last_msg_ts > 30.0:
                    await self._poll_recent_agg_trades(session)
                self._poll_last_msg_ts = m.time.time()
        except Exception:
            self._poll_connected = False

    def scan(c, s):                                      # :836-854
        self = c
        sig = self.compute_signal()
        self._cached = sig
        txt = json.dumps(sig)
        full.append(txt)
        sigs.append([s, zlib.crc32(txt.encode()), len(txt), txt if len(sigs) < 12 or len(sigs) % 50 == 0 else ""])   # 글자 = 앞 12개·50번째마다(크기)
        dt_now = m.datetime.now()
        if dt_now.minute != last_min[s] and dt_now.second < 15:
            last_min[s] = dt_now.minute
            self._nif_hist.append(sig["nif_whale"])
            self._oi_delta_hist.append(sig["oi_delta_pct"])
            self._abs_hist.append(sig["shadow_absorption_score"])
            self._tox_hist.append(sig["shadow_toxicity_score"])
            self._bias_hist.append(sig["shadow_queue_bias"])
            self._eai_hist.append(sig["eai"])
            bucket_ts = (dt_now - dt.timedelta(minutes=1)).replace(second=0, microsecond=0)
            self._db_insert(bucket_ts, sig)

    nxt = (inp["t0"] // 10) * 10 + 10 + 0.05
    with gv._quiet():
        for t, kind, s, body in inp["events"]:
            while nxt <= t:
                clock[0] = nxt
                for sym in PX:
                    scan(sc[sym], sym)
                nxt = (math.floor(nxt / 10) + 1) * 10 + 0.05
            clock[0] = t
            c = sc[s]
            if kind == "D":
                depth(c, body)
            elif kind == "T":
                trade(c, body)
            elif kind == "P":
                asyncio.run(poll(c, json.loads(body)))
            elif kind == "do":
                c._depth_connected = True
            elif kind == "de":
                c._depth_connected = False
            elif kind == "to":
                c._trade_connected = True
    return {"boot": boot, "requests": sent, "sigs": sigs, "full": full}


def check_watchdog() -> None:
    """감시 ops_watchdog.check_data_sources(:236-279)를 파이썬이 쓴 파일과 Rust 가 쓴 파일(같은 새 입력 · 미시구조 + 요격기 liq_burst_state)에 돌려 같은 판정인지.
    네트워크 없이 함수만. 신선도는 실제 시계로 재므로 두 쪽 다 같은 «지금» 에서 본다."""
    import subprocess
    import gen_market_fixture as gm
    t0 = float(int(time.time())) - 1500                  # 끝이 지금 언저리에 오게(미시구조 25분)
    inp = gen(random.Random(7), t0)
    minp = gm.gen(random.Random(8), float(int(time.time())) - 1200)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for side in ("py", "rs"):
            (td / side).mkdir()
        (td / "mi.json").write_text(json.dumps(inp))
        (td / "ma.json").write_text(json.dumps(minp))
        python_reference(inp, td / "py")
        gm.python_reference(minp, td / "py")
        env = dict(os.environ, PATH=f"{Path.home()}/.cargo/bin:{os.environ.get('PATH', '')}")
        for ex, a1, a2 in (("replay_micro", td / "mi.json", td / "rs" / "microstructure.duckdb"), ("replay_market", td / "ma.json", td / "rs")):
            r = subprocess.run(["bash", "-c", 'ulimit -v 4000000; exec timeout 300 "$@"', "_", "cargo", "run", "-q", "-j", "2", "--release", "--example", ex, str(a1), str(a2)],
                               cwd=ROOT / "rtduck", env=env, capture_output=True, text=True)
            assert r.returncode == 0, r.stderr
        import scripts.ops_watchdog as wd
        out = {}
        for side in ("py", "rs"):
            wd.LIVE = td / side
            wd._DUCKDB_CHECK_CACHE.clear()
            c = wd.check_data_sources(td / side / "microstructure.duckdb")
            out[side] = (c.status, json.dumps(c.details, sort_keys=True, default=str))
            print(f"  {side}: {c.status} · {c.details.get('sources')} · stale {c.details.get('data_stale')}")
        assert out["py"] == out["rs"], out
    print("check-watchdog OK")


def main() -> None:
    if "--check-watchdog" in sys.argv:
        return check_watchdog()
    t0 = float(int(time.time()))
    inp = gen(random.Random(20261132), t0)
    with tempfile.TemporaryDirectory() as td:
        ref = python_reference(inp, Path(td))
        golden = dict(ref, tables=dump(Path(td) / "microstructure.duckdb"))
    if os.environ.get("MICRO_FULL"):                     # 고치는 중에만: 신호 글자 전부를 따로
        Path(os.environ["MICRO_FULL"]).write_text(json.dumps(ref.pop("full")))
    ref.pop("full", None)
    golden.pop("full", None)
    (FIX / "micro_input.json").write_text(json.dumps(inp, separators=(",", ":")))
    (FIX / "micro_golden.json").write_text(json.dumps(golden, separators=(",", ":"), ensure_ascii=False))
    print({k: len(v["rows"]) for k, v in golden["tables"].items()}, "부트스트랩", ref["boot"], f"신호 {len(ref['sigs'])} · 요청 {len(ref['requests'])}",
          f"{(FIX / 'micro_input.json').stat().st_size // 1024}+{(FIX / 'micro_golden.json').stat().st_size // 1024}KB")


if __name__ == "__main__":
    main()
