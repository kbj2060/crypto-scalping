#!/usr/bin/env python3
"""바이낸스 체결 테이프 REST 검증·복구 일치 시험의 입력·정답지 (2026-11-20).

원본 scripts/live_trade_tape_collector_20260916.py 의 `backfill_loop`·`binance_minute_fetcher`·`verify_recent`·`_bn_pace`·
`_bn_note_weight` 와 `TapeStore` 의 복구 함수들을 **그대로** 돌린다. 바뀌는 것은 바깥뿐:
  · REST = 가짜 aiohttp 세션: 요청(경로?쿼리)을 열쇠로 정해 둔 응답을 차례로 꺼낸다(정상 · 빈 응답 · 쪽 넘김 · 1분봉과 안 맞음 ·
    400(-4166) · 418 · 429 · 시간초과 · 500). 나간 요청은 (단조 시각, 벽시각, 경로?쿼리)로 적는다 = 정답 «요청 열».
  · 시계 = 모듈의 `time.time`·`time.monotonic` 을 가짜로, `asyncio.sleep` 은 가짜 시계를 민다(속도 제한의 간격이 요청 시각에 남는다).
    `backfill_loop` 는 한 주기 = 잠 하나 + 몸통 한 번(두 번째 잠에서 멈춘다). `asyncio.to_thread` 는 바로 부른다.
  · 시작 DB = 원본 TapeStore 로 만든 표에 정해 둔 행·공백·검사 기록을 넣는다(Rust 시험도 같은 순서로 넣는다).
결과 DB(테이프·gaps·verify_1m·backfill_1m·meta, checked_at·done_at 은 CURRENT_TIMESTAMP 라 모양만)와 요청 열이 정답지다.
rtcollect/tests/rest_parity.rs 가 같은 입력을 127.0.0.1 가짜 HTTP 서버로 Rust 에 넣어 비교한다. 네트워크 없음(소켓 connect 를 막는다).
합성 자료만(실제 장과 무관한 시각·가격).

    python3 rtcollect/tools/gen_rest_fixture.py        # tests/fixtures/rest_{input,golden}.json
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from urllib.parse import urlencode, urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_venue_fixture as gv  # noqa: E402 -- 소켓 막기 · _ns · dump · close_shared

FIX = gv.FIX
SEED = 20261122
W0 = 1_763_683_200.0 + 3 * 3600 + 0.25     # 2025-11-21 03:00:00.25 UTC(합성). 분 경계 0.25초 뒤
M0 = 50_000.0
NOW = int(W0)
LO = NOW - 47 * 3600                        # 원본 BINANCE_BACKFILL_LOOKBACK_S
URLS = {"futures": ("http://127.0.0.1/fapi/v1/klines", "http://127.0.0.1/fapi/v1/aggTrades"),
        "spot": ("http://127.0.0.1/api/v3/klines", "http://127.0.0.1/api/v3/aggTrades")}
TABLES = gv.TAPE_TABLES


def _mod():
    return __import__("scripts.live_trade_tape_collector_20260916", fromlist=["x"])


# ── 입력 ─────────────────────────────────────────────────────────────────────
class Gen:
    def __init__(self, rng: random.Random, symbol: str, px: float, step: float, dec: int):
        self.rng, self.sym, self.px, self.step, self.dec, self.aid, self.fid = rng, symbol.upper(), px, step, dec, 7_000_000, 90_000_000
        self.resp: dict[str, list] = {}

    def trades(self, m: int, n: int, secs=(0, 60), beyond: int = 0) -> list[dict]:
        """그 분의 aggTrades(시각 순). beyond = 분 끝 뒤 체결 몇 개(걸러져야 한다)."""
        ts = sorted(m * 1000 + self.rng.randrange(secs[0] * 1000, secs[1] * 1000) for _ in range(n))
        ts += sorted((m + 60) * 1000 + self.rng.randrange(0, 3000) for _ in range(beyond))
        out = []
        for t in ts:
            self.aid += 1
            f = self.fid + 1
            self.fid += 1 + self.rng.randrange(3)
            q = max(0.001, round(self.rng.lognormvariate(-1.5, 1.3), 3))
            out.append({"a": self.aid, "p": f"{self.px + self.step * self.rng.randrange(3):.{self.dec}f}", "q": f"{q:.3f}",
                        "f": f, "l": self.fid, "T": t, "m": self.rng.random() < 0.5})
        return out

    @staticmethod
    def vol(trades: list[dict], extra: float = 0.0, scale: float = 1.0) -> str:
        return f"{(sum(float(a['q']) for a in trades) + extra) * scale:.3f}"

    def put(self, path: str, params: dict, *responses) -> None:
        self.resp.setdefault(f"{path}?{urlencode(params)}", []).extend(responses)

    def kline(self, path: str, m: int, vol: str, close_ms: int | None = None, **r) -> dict:
        row = [m * 1000, "1", "1", "1", "1", vol, close_ms if close_ms is not None else m * 1000 + 59_999, "0", 1, "0", "0", "0"]
        self.put(path, {"symbol": self.sym, "interval": "1m", "startTime": m * 1000, "limit": 1}, {"status": 200, "body": [row], **r})
        return row

    def agg(self, path: str, m: int, *responses, from_id: int | None = None) -> None:
        p = ({"symbol": self.sym, "startTime": m * 1000, "endTime": m * 1000 + 59_999, "limit": 1000} if from_id is None
             else {"symbol": self.sym, "fromId": from_id, "limit": 1000})
        self.put(path, p, *responses)


def live_rows(trades: list[dict], symbol: str, skip_secs=()) -> list[list]:
    """라이브 행(원본 TapeBuffer.add_agg 로 만든 것) -- skip_secs 초는 뺀다."""
    m = _mod()
    b = m.TapeBuffer(m.BUCKETS.get(symbol, 0.01), symbol)
    for a in trades:
        if (a["T"] // 1000) % 60 not in skip_secs:
            b.add_agg(a["T"], float(a["p"]), float(a["q"]), a["m"], a["l"] - a["f"] + 1)
    return [list(r) for r in b.take_closed(everything=True)]


def mixed(rng) -> dict:
    """혼합: 공백 분(라이브+REST) · 불일치 분(REST 만) · 확정 전 1분봉 · 영구 거부(-4166) · 1분봉과 안 맞음(시도 3번째) · 빈 1분봉 ·
    물량 0 1분봉 · 분 경계를 건너는 공백 · 47시간 경계 · 이미 맞은 분·교체한 분·3번 실패한 분·너무 새 분(빼야 한다) · 가중치 양보."""
    g = Gen(rng, "ethusdt", 2500.0, 0.1, 2)
    ku, au = (urlparse(u).path for u in URLS["futures"])
    setup: list = []
    A, B, C, D, E, H, I, J, K = NOW - 600, NOW - 540, NOW - 480, NOW - 60, NOW - 180, NOW - 660, NOW - 720, NOW - 780, NOW - 900
    # A: 20~35초 공백. 라이브는 공백 밖 초 + REST 에 없는 체결 하나(5초, 시장 체결 아님) · 공백 안 22초의 반쪽 행(덮여야 한다)
    ta = g.trades(A, 40)
    rows = live_rows(ta, "ethusdt", skip_secs=range(20, 35))
    rows.append([A + 5, 25009, 0.5, 0.0, 1, 0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0, 0, 1, 0, 1, 0])
    rows.append([A + 22, 25000, 0.25, 0.0, 1, 0, 0.25, 0.0, 0.25, 0.0, 0.0, 0.0, 0, 0, 1, 0, 1, 0])
    rows = sorted({(r[0], r[1]): r for r in rows}.values())
    setup += [["write", rows], ["gap", A * 1000 + 20_000, A * 1000 + 35_000, "ws_reconnect"]]
    g.kline(ku, A, Gen.vol(ta, extra=0.5))
    g.agg(au, A, {"status": 200, "body": ta})
    # B: 1분봉 불일치(-20%) 기록, 공백 없음. 라이브 한 행이 반쪽 -> 라이브+REST 는 안 맞고 REST 만이 맞는다
    tb = g.trades(B, 30)
    rows = live_rows(tb, "ethusdt")
    rows[3][2], rows[3][3] = rows[3][2] * 0.5, rows[3][3] * 0.5
    setup += [["write", rows], ["verify", B, 1.0, 1.25, -0.2]]
    g.kline(ku, B, Gen.vol(tb))
    g.agg(au, B, {"status": 200, "headers": {"x-mbx-used-weight-1m": "1500"}, "body": tb})   # 가중치 1500 -> 다음 분까지 양보
    # C: 맞는 검사 + 공백 -> 빼야 한다 · H: 교체한 분 · I: 3번 실패 · D: 너무 새 분(첫 주기엔 빼고 뒤 주기엔 후보)
    setup += [["write", live_rows(g.trades(C, 10), "ethusdt")], ["verify", C, 1.0, 1.0, 0.0],
              ["gap", C * 1000 + 10_000, C * 1000 + 15_000, "ws_reconnect"],
              ["gap", H * 1000 + 1_000, H * 1000 + 9_000, "ws_reconnect"], ["backfill", H, 1, 1],
              ["verify", I, 1.0, 2.0, -0.5], ["backfill", I, 0, 3],
              ["gap", D * 1000 + 30_000, D * 1000 + 40_000, "ws_reconnect"]]
    td = g.trades(D, 12)
    g.kline(ku, D, Gen.vol(td))
    g.agg(au, D, {"status": 200, "body": td})
    # E: 끝난 지 3분 -- 첫 주기엔 1분봉이 확정 전(closeTime 이 미래) -> 건너뜀, 다음 주기에 교체
    te = g.trades(E, 15, secs=(30, 50))
    setup += [["gap", E * 1000 + 30_000, E * 1000 + 50_000, "ws_reconnect"]]
    g.kline(ku, E, Gen.vol(te), close_ms=int((W0 + 600) * 1000))
    g.kline(ku, E, Gen.vol(te))
    g.agg(au, E, {"status": 200, "body": te})
    # J: 2번 실패한 불일치 분 -- 또 안 맞음(1분봉 +5%) -> 시도 3, 다음 주기엔 빠진다
    tj = g.trades(J, 20)
    setup += [["verify", J, 1.0, 2.0, -0.5], ["backfill", J, 0, 2]]
    g.kline(ku, J, Gen.vol(tj, scale=1.05))
    g.agg(au, J, {"status": 200, "body": tj})
    # K..K+60: 분 경계를 건너는 공백. K = 빈 1분봉(None) · K+60 = 물량 0 1분봉 + 빈 aggTrades -> 공백 밖 라이브 행만으로 «맞음»(1분봉 0 이면 rel 0)
    setup += [["write", [[K + 90, 25003, 0.75, 0.25, 1, 1, 0.75, 0.25, 0.75, 0.25, 0.0, 0.0, 0, 0, 1, 1, 1, 1]]],
              ["gap", K * 1000 + 50_000, (K + 60) * 1000 + 10_000, "ws_reconnect"]]
    g.put(ku, {"symbol": "ETHUSDT", "interval": "1m", "startTime": K * 1000, "limit": 1}, *[{"status": 200, "body": []}] * 3)   # 주기마다 다시 묻는다
    g.kline(ku, K + 60, "0")
    g.agg(au, K + 60, {"status": 200, "body": []})
    # 47시간 경계(첫 주기의 now = NOW + 60 기준): 그 앞 공백은 통째로 밖 · 경계를 걸친 공백은 경계 분만 -> 영구 거부 400(-4166) -> 실패 1회
    L = LO + 60
    setup += [["gap", (L - 600) * 1000, (L - 500) * 1000, "ws_reconnect"], ["gap", (L - 90) * 1000, (L + 30) * 1000, "startup"]]
    tl = g.trades(L, 3)
    g.kline(ku, L, Gen.vol(tl))
    g.agg(au, L, {"status": 400, "body": {"code": -4166, "msg": "Search window is restricted to recent 2 days only."}})
    return {"name": "혼합", "symbol": "ethusdt", "market": "futures", "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["cycle", "cycle", "cycle"], "responses": g.resp}


def paging(rng) -> dict:
    """쪽 넘김: 1000건 쪽 → fromId 로 다음 쪽(마지막 체결이 분 끝 뒤 -> 멈춤, 분 밖 체결은 걸러진다)."""
    g = Gen(rng, "ethusdt", 2500.0, 0.1, 2)
    ku, au = (urlparse(u).path for u in URLS["futures"])
    P = NOW - 300
    t = g.trades(P, 1990, beyond=10)
    g.kline(ku, P, Gen.vol(t[:1990]))
    g.agg(au, P, {"status": 200, "body": t[:1000]})
    g.agg(au, P, {"status": 200, "body": t[1000:]}, from_id=t[999]["a"] + 1)
    return {"name": "쪽 넘김", "symbol": "ethusdt", "market": "futures", "mono0": M0, "wall0": W0,
            "setup": [["gap", P * 1000, P * 1000 + 60_000, "ws_reconnect"]], "steps": ["cycle"], "responses": g.resp}


def errors(rng) -> dict:
    """429(Retry-After) → 시간초과 → 정상 · 빈 1분봉 → 418(banned until) → 정상 · 1분봉 대조(빠진 분·불일치) · 대조 500."""
    g = Gen(rng, "ethusdt", 2500.0, 0.1, 2)
    ku, au = (urlparse(u).path for u in URLS["futures"])
    K1, K2, K3 = NOW - 420, NOW - 480, NOW - 540
    setup = [["gap", m * 1000, m * 1000 + 60_000, "ws_reconnect"] for m in (K1, K2, K3)]
    t1, t2 = g.trades(K1, 25), g.trades(K2, 25)
    kl = lambda m, t: {"symbol": "ETHUSDT", "interval": "1m", "startTime": m * 1000, "limit": 1}   # noqa: E731
    row = lambda m, t: [m * 1000, "1", "1", "1", "1", Gen.vol(t), m * 1000 + 59_999, "0", 1, "0", "0", "0"]   # noqa: E731
    g.put(ku, kl(K1, t1), {"status": 429, "headers": {"retry-after": "7", "x-mbx-used-weight-1m": "2400"}, "body": {"code": -1003, "msg": "Too many requests"}},
          {"status": 200, "body": [row(K1, t1)]}, {"status": 200, "body": [row(K1, t1)]})
    g.agg(au, K1, {"timeout": True}, {"status": 200, "body": t1})
    banned = int((W0 + 4 * 60 + 30) * 1000)   # 넷째 주기(~W0+243) 뒤 다섯째(~W0+303) 전에 풀린다 -- Rust 는 그동안 안 보낸다
    g.put(ku, kl(K2, t2), {"status": 200, "body": []},
          {"status": 418, "body": {"code": -1003, "msg": f"Way too many requests; IP(1.2.3.4) banned until {banned}. Please use the websocket for live updates to avoid bans."}},
          {"status": 200, "body": [row(K2, t2)]})
    g.agg(au, K2, {"status": 200, "body": t2})
    g.kline(ku, K3, "0")
    g.agg(au, K3, {"status": 200, "body": []})
    # 1분봉 대조 대상: 검사 기록 없는 끝난 분들(V1 은 1분봉에 없음 · V2 불일치 · V3 맞음)
    V1, V2, V3 = NOW - 1200, NOW - 1260, NOW - 1380
    tv = {m: g.trades(m, 8) for m in (V1, V2, V3)}
    setup += [["write", live_rows(tv[m], "ethusdt")] for m in (V1, V2, V3)]
    # 대조 시각 = 다섯 주기 뒤(가짜 시계). 그때의 «끝난 분» 에 교체한 K1·K2·K3 는 verify 기록이 있어 빠진다
    vrows = [[m * 1000, "1", "1", "1", "1", v, m * 1000 + 59_999, "0", 1, "0", "0", "0"]
             for m, v in ((V3, Gen.vol(tv[V3])), (V3 + 60, "5.000"), (V2, Gen.vol(tv[V2], scale=0.9)))]
    g.put(ku, {"symbol": "ETHUSDT", "interval": "1m", "startTime": V3 * 1000, "limit": (V1 - V3) // 60 + 2}, {"status": 200, "body": vrows})
    g.put(ku, {"symbol": "ETHUSDT", "interval": "1m", "startTime": V1 * 1000, "limit": 2}, {"status": 500, "body": {"code": -1000}})
    return {"name": "오류", "symbol": "ethusdt", "market": "futures", "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["cycle"] * 5 + ["verify", "verify"], "responses": g.resp}


def spot(rng) -> dict:
    """현물 SOL: /api/v3 주소 · ETH 아닌 심볼은 한 주기 4분(공백 분 6개 중 최근 4개만)."""
    g = Gen(rng, "solusdt", 140.0, 0.01, 2)
    ku, au = (urlparse(u).path for u in URLS["spot"])
    ms = [NOW - 960 - 60 * k for k in range(6)]
    setup = [["gap", m * 1000 + 5_000, m * 1000 + 25_000, "ws_reconnect"] for m in ms]
    for m in ms[:4]:
        t = g.trades(m, 6, secs=(5, 25))
        g.kline(ku, m, Gen.vol(t))
        g.agg(au, m, {"status": 200, "body": t})
    return {"name": "현물", "symbol": "solusdt", "market": "spot", "mono0": M0, "wall0": W0, "setup": setup, "steps": ["cycle"], "responses": g.resp}


# ── 원본 돌리기 ──────────────────────────────────────────────────────────────
class _Stop(BaseException):
    pass


class _Status(Exception):
    def __init__(self, status: int):
        super().__init__(f"{status}")
        self.status = status


def python_reference(sc: dict, d: Path) -> dict:
    m = _mod()
    clock = {"mono": sc["mono0"], "wall": sc["wall0"]}
    loops = {"n": 0}

    def fake_sleep(s, *a, **k):
        if sys._getframe(1).f_code.co_name == "backfill_loop":
            loops["n"] += 1
            if loops["n"] > 1:
                raise _Stop()          # 한 주기 끝
        clock["mono"] += s
        clock["wall"] += s

        async def _c():
            return None
        return _c()

    async def direct(f, *a, **k):
        return f(*a, **k)

    sent: list = []
    resp = {k: list(v) for k, v in sc["responses"].items()}

    class R:
        def __init__(self, r):
            self.status, self.headers, self._body = r["status"], r.get("headers", {}), r["body"]

        def raise_for_status(self):
            if self.status >= 400:
                raise _Status(self.status)

        async def json(self):
            return self._body

    class Req:
        def __init__(self, url, params):
            self.target = f"{urlparse(url).path}?{urlencode(params)}"

        async def __aenter__(self):
            sent.append([clock["mono"], clock["wall"], self.target])
            q = resp.get(self.target)
            gv._want(bool(q), ("정해 둔 응답이 없다", self.target))
            r = q.pop(0)
            if r.get("timeout"):
                raise asyncio.TimeoutError()
            return R(r)

        async def __aexit__(self, *a):
            return False

    class Sess:
        def get(self, url, params=None, timeout=None):
            return Req(url, params)

    old = m.time, m.asyncio
    m.time = gv._ns(time, time=lambda: clock["wall"], monotonic=lambda: clock["mono"])
    m.asyncio = gv._ns(asyncio, sleep=fake_sleep, to_thread=direct)
    m._BN_PACE["next"] = 0.0
    sym, market = sc["symbol"], sc["market"]
    ku, au = URLS[market]
    db = d / ("binance_spot_tape.sqlite" if market == "spot" else "binance_tape.sqlite")
    try:
        with gv._quiet():
            store = m.TapeStore(db, sym, m.BUCKETS.get(sym, 0.01))
            for op in sc["setup"]:
                if op[0] == "write":
                    store.write([tuple(r) for r in op[1]])
                elif op[0] == "gap":
                    store.record_gap(op[1], op[2], op[3])
                elif op[0] == "verify":
                    with store._connect() as con:
                        con.execute("INSERT INTO verify_1m VALUES (?,?,?,?,?,CURRENT_TIMESTAMP)", [sym, *op[1:]])
                elif op[0] == "backfill":
                    with store._connect() as con:
                        con.execute("INSERT INTO backfill_1m VALUES (?,?,'setup',0,0.0,0.0,0.0,?,?,'setup',CURRENT_TIMESTAMP)", [sym, *op[1:]])
            for step in sc["steps"]:
                if step == "cycle":
                    loops["n"] = 0
                    fetch = m.binance_minute_fetcher(Sess(), sym, m.BUCKETS.get(sym, 0.01), ku, au)
                    try:   # per_cycle·source·note·lookback = 원본 collect() 의 호출 그대로
                        asyncio.run(m.backfill_loop(store, fetch, per_cycle=10 if sym == "ethusdt" else 4, source="binance rest aggTrades",
                                                    note="buy_max/sell_max NULL(REST 에 개별 체결 최대가 없다) · 주문=aggTrade",
                                                    lookback_s=m.BINANCE_BACKFILL_LOOKBACK_S))
                    except _Stop:
                        pass
                else:
                    asyncio.run(m.verify_recent(store, Sess(), ku))
        gv.close_shared()
    finally:
        m.time, m.asyncio = old
        m._BN_PACE["next"] = 0.0
    return {"name": sc["name"], "db": {db.name: gv.dump(db, TABLES)}, "requests": sent,
            "unused": sorted(k for k, v in resp.items() if v)}


def main() -> None:
    rng = random.Random(SEED)
    scs = [mixed(rng), paging(rng), errors(rng), spot(rng)]
    out = []
    for sc in scs:
        with tempfile.TemporaryDirectory() as td:
            out.append(python_reference(sc, Path(td)))
        g = out[-1]
        t = g["db"][next(iter(g["db"]))]["tables"]
        print(f"{sc['name']}: 요청 {len(g['requests'])} · 남은 응답 {g['unused']} · " + " · ".join(f"{k} {len(v)}행" for k, v in t.items() if v))
    (FIX / "rest_input.json").write_text(json.dumps({"seed": SEED, "scenarios": scs}, separators=(",", ":"), ensure_ascii=False))
    (FIX / "rest_golden.json").write_text(json.dumps({"scenarios": out}, separators=(",", ":"), ensure_ascii=False))


if __name__ == "__main__":
    main()
