#!/usr/bin/env python3
"""OKX·Bybit 체결 테이프 REST 일치 시험의 입력·정답지 (2026-11-21). gen_rest_fixture.py(바이낸스)와 같은 방식.

원본을 **그대로** 돌린다: OKX `assert_ct_val` · `verify_recent` · `okx_minute_fetcher`+`_pace` 와 공용 `backfill_loop`·`TapeStore`,
Bybit `verify_recent`. 바뀌는 것은 바깥뿐 -- REST = 가짜 aiohttp 세션(요청 경로?쿼리를 열쇠로 정해 둔 응답을 차례로 · aiohttp json() 의
본문 형식 검사 흉내), 시계 = 모듈의 time·asyncio.sleep 을 가짜로(잠이 두 시계를 민다), `backfill_loop` 한 주기 = 잠 하나 + 몸통 한 번.
나간 요청 (단조, 벽, 경로?쿼리)과 결과 DB 가 정답지. rtcollect/tests/venue_rest_parity.rs 가 127.0.0.1 가짜 HTTP 서버로 Rust 와 비교한다.
합성 자료만. 네트워크 없음(소켓 connect 를 막는다).

    python3 rtcollect/tools/gen_venue_rest_fixture.py        # tests/fixtures/venue_rest_{input,golden}.json
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
import gen_venue_fixture as gv  # noqa: E402

FIX = gv.FIX
SEED = 20261124
W0 = 1_763_683_200.0 + 5 * 3600 + 0.25
M0 = 70_000.0
NOW = int(W0)
INST = "ETH-USDT-SWAP"
CU, TU, IU = "/api/v5/market/history-candles", "/api/v5/market/history-trades", "/api/v5/public/instruments"
BU = "/v5/market/kline"
J = "application/json"


def okx_mod():
    return __import__("scripts.live_okx_trade_tape_collector_20260923", fromlist=["x"])


def bybit_mod():
    return __import__("scripts.live_bybit_trade_tape_collector_20261006", fromlist=["x"])


class Gen:
    def __init__(self, rng):
        self.rng, self.tid, self.resp = rng, 900_000_000, {}

    def put(self, path, params, *rs):
        self.resp.setdefault(f"{path}?{urlencode(params)}", []).extend(rs)

    def trades(self, m, n, secs=(0, 60), dup=0.3):
        """그 분의 OKX 개별 체결(오래된 것부터, tradeId 증가) -- 연속 (ts, side, px) 같은 체결이 섞인다(묶기)."""
        out = []
        ts = sorted(m * 1000 + self.rng.randrange(secs[0] * 1000, secs[1] * 1000) for _ in range(n))
        prev = None
        for t in ts:
            self.tid += 1
            if prev and self.rng.random() < dup:
                d = {**prev, "tradeId": str(self.tid), "sz": f"{self.rng.randint(1, 40)}"}
            else:
                d = {"instId": INST, "side": self.rng.choice(("buy", "sell")), "sz": f"{self.rng.randint(1, 400)}",
                     "px": f"{2500 + self.rng.randrange(0, 30) * 0.01:.2f}", "source": "0", "tradeId": str(self.tid), "ts": str(t)}
            out.append(d)
            prev = d
        return out

    @staticmethod
    def vol_ccy(fills, scale=1.0):
        return repr(round(sum(float(f["sz"]) for f in fills) * 0.1 * scale, 6))

    def candle(self, m, vol, confirm="1", **r):
        row = [str(m * 1000), "2500", "2501", "2499", "2500.5", "0", vol, "0", confirm]
        self.put(CU, {"instId": INST, "bar": "1m", "after": (m + 60) * 1000, "limit": 1},
                 {"status": 200, "ctype": J, "raw": json.dumps({"code": "0", "msg": "", "data": [row]}), **r})

    def pages(self, m, fills, before=3, page=100):
        """history-trades: 경계 1건(type=2, 분 끝 이전 가장 새 체결) → tradeId 로 거꾸로 100건씩(type=1). before = 분 앞 체결 몇 건(멈춤 조건)."""
        old = self.trades(m - 1, before, secs=(59, 60), dup=0)
        for i, f in enumerate(old):                            # 분 앞이라 tradeId 가 작아야 한다
            f["tradeId"] = str(int(fills[0]["tradeId"]) - 10 - i)
            f["ts"] = str(m * 1000 - 1 - i)
        allf = sorted(fills + old, key=lambda f: int(f["tradeId"]), reverse=True)   # 새 것부터
        edge = allf[0]
        self.put(TU, {"instId": INST, "type": 2, "after": (m + 60) * 1000 + 1, "limit": 1}, ok([edge]))
        after = int(edge["tradeId"]) + 1
        i = 0
        while True:
            pg = [f for f in allf if int(f["tradeId"]) < after][:page]
            self.put(TU, {"instId": INST, "type": 1, "after": after, "limit": page}, ok(pg))
            if not pg or int(pg[-1]["ts"]) < m * 1000 or len(pg) < page:
                break
            after = int(pg[-1]["tradeId"])
            i += 1


def ok(data, **r):
    return {"status": 200, "ctype": J, "raw": json.dumps({"code": "0", "msg": "", "data": data}), **r}


def live_rows(fills, skip_secs=()):
    """라이브 행(원본 OkxTapeBuffer 로 만든 것 -- 묶음은 group_fills) · skip_secs 초는 뺀다."""
    m = okx_mod()
    b = m.OkxTapeBuffer(0.1, INST)
    for ts, px, sz, sell, n in m.group_fills(sorted(fills, key=lambda f: int(f["tradeId"]))):
        if (ts // 1000) % 60 not in skip_secs:
            b.add_okx_order(ts, px, sz * 0.1, sell, n)
    return [list(r) for r in b.take_closed(everything=True)]


def okx_backfill(rng) -> dict:
    g = Gen(rng)
    A, B, C, D, E, F = NOW - 600, NOW - 660, NOW - 720, NOW - 780, NOW - 840, NOW - 900
    setup = []
    # A: 공백 분 · 체결 250건 = 3쪽(마지막 쪽에서 분 앞 체결을 만나 멈춤) · 1분봉과 맞음 → 교체
    fa = g.trades(A, 250)
    setup.append(["gap", A * 1000, (A + 60) * 1000, "ws_reconnect"])
    g.candle(A, g.vol_ccy(fa))
    g.pages(A, fa)
    # B: 공백 20~35초 · 라이브는 공백 밖 초 → 라이브+REST 가 맞음
    fb = g.trades(B, 60)
    setup += [["write", live_rows(fb, skip_secs=range(20, 35))], ["gap", B * 1000 + 20_000, B * 1000 + 35_000, "ws_reconnect"]]
    g.candle(B, g.vol_ccy(fb))
    g.pages(B, fb)
    # C: 1분봉 +5% → 안 맞음 -- 주기마다 또(3번이면 포기)
    fc = g.trades(C, 30)
    setup.append(["gap", C * 1000 + 5_000, C * 1000 + 50_000, "ws_reconnect"])
    for _ in range(3):
        g.candle(C, g.vol_ccy(fc, 1.05))
        g.pages(C, fc)
    # D: 확정 전 봉(confirm 0) → 건너뜀 · 다음 주기엔 확정
    fd = g.trades(D, 20)
    setup.append(["gap", D * 1000, D * 1000 + 30_000, "ws_reconnect"])
    g.candle(D, g.vol_ccy(fd), confirm="0")
    g.candle(D, g.vol_ccy(fd))
    g.pages(D, fd)
    # E: 400(영구 거부) → 실패로 기록하고 다음 분 · F: 429 → 이 주기 멈춤(슬롯 +10초) → 다음 주기 정상
    setup += [["gap", E * 1000, E * 1000 + 10_000, "ws_reconnect"], ["gap", F * 1000, F * 1000 + 10_000, "ws_reconnect"]]
    g.put(CU, {"instId": INST, "bar": "1m", "after": (E + 60) * 1000, "limit": 1}, *[{"status": 400, "ctype": J, "raw": '{"code":"51000","msg":"x"}'}] * 3)
    ff = g.trades(F, 12)
    g.put(CU, {"instId": INST, "bar": "1m", "after": (F + 60) * 1000, "limit": 1}, {"status": 429, "ctype": J, "raw": '{"code":"50011","msg":"Too Many Requests"}'})
    g.candle(F, g.vol_ccy(ff))
    g.pages(F, ff)
    return {"name": "OKX 복구", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["cycle", "cycle", "cycle"], "responses": g.resp}


def okx_codes(rng) -> dict:
    """code != "0"(200) → 상태 없는 예외 = 주기 멈춤 · 본문 형식 아님 · 빈 data(분 없음) · 빈 쪽."""
    g = Gen(rng)
    A, B = NOW - 300, NOW - 360
    setup = [["gap", A * 1000, A * 1000 + 9_000, "ws_reconnect"], ["gap", B * 1000, B * 1000 + 9_000, "ws_reconnect"]]
    p = lambda m: {"instId": INST, "bar": "1m", "after": (m + 60) * 1000, "limit": 1}  # noqa: E731
    g.put(CU, p(A), {"status": 200, "ctype": J, "raw": '{"code":"50001","msg":"busy","data":[]}'},
          {"status": 200, "ctype": "text/plain", "raw": '{"code":"0","data":[]}'}, ok([]), ok([]), ok([]))
    g.put(CU, p(B), ok([]), {"timeout": True}, ok([[str(B * 1000), "1", "1", "1", "1", "0", "0.0", "0", "1"]]))
    g.put(TU, {"instId": INST, "type": 2, "after": (B + 60) * 1000 + 1, "limit": 1}, ok([]))      # 체결 없음 → 빈 분(1분봉 0 → 맞음)
    return {"name": "OKX 응답 코드", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["cycle"] * 5, "responses": g.resp}


def okx_verify(rng) -> dict:
    """1분봉 대조: 밀린 분이 100분 넘게 흩어짐 → 구간 둘 · confirm 0 빼기 · 500 이면 그만 · 모양 틀린 행 = 예외(원본은 WS 를 끊는다)."""
    g = Gen(rng)
    near = [NOW - 300 - 60 * k for k in range(4)]
    far = [NOW - 300 - 60 * 150 - 60 * k for k in range(3)]
    setup = []
    rows = {}
    for m in near + far:
        f = g.trades(m, 6)
        rows[m] = f
        setup.append(["write", live_rows(f)])
    def crow(m, scale=1.0, confirm="1"):
        return [str(m * 1000), "1", "1", "1", "1", "0", Gen.vol_ccy(rows[m], scale), "0", confirm]
    q = lambda a: {"instId": INST, "bar": "1m", "limit": "100", "after": str((a + 60) * 1000)}  # noqa: E731
    g.put(CU, q(near[0]), ok([crow(near[0]), crow(near[1], 0.9), crow(near[2], confirm="0"), crow(near[3])]))
    g.put(CU, q(far[0]), ok([crow(far[0]), crow(far[2], 1.2)]))
    # 둘째 대조: 남은 분(near[2] · far[1])
    g.put(CU, q(near[2]), {"status": 500, "ctype": J, "raw": '{"code":"50000"}'})
    # 셋째: 모양 틀린 행(9칸 미만) → 예외
    g.put(CU, q(near[2]), ok([["1", "2"]]))
    return {"name": "OKX 대조", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["verify", "verify", "verify"], "responses": g.resp}


def bybit_verify(rng) -> dict:
    g = Gen(rng)
    ms = [NOW - 300 - 60 * k for k in range(5)]
    setup = []
    vols = {}
    for m in ms:
        q = round(rng.uniform(1, 9), 3)
        vols[m] = q
        setup.append(["write", [[m + 7, 250000, q, 0.0, 1, 0, q, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0, 0, 1, 0]]])
    setup.append(["gap", ms[1] * 1000, ms[1] * 1000 + 5000, "ws_reconnect"])
    p = {"category": "linear", "symbol": "ETHUSDT", "interval": "1", "start": ms[-1] * 1000, "end": ms[0] * 1000, "limit": 1000}
    lst = [[str(m * 1000), "1", "1", "1", "1", repr(vols[m] * (0.8 if m == ms[1] else 1.0)), "0"] for m in ms[:-1]]
    g.resp.setdefault(f"{BU}?{urlencode(p)}", []).append({"status": 200, "ctype": J, "raw": json.dumps({"retCode": 0, "result": {"category": "linear", "list": lst}})})
    p2 = {**p, "start": ms[-1] * 1000, "end": ms[-1] * 1000}
    g.resp.setdefault(f"{BU}?{urlencode(p2)}", []).extend([
        {"status": 403, "ctype": "text/html", "raw": "<html>403</html>"},
        {"status": 200, "ctype": J, "raw": '{"retCode":10001,"retMsg":"x","result":{}}'},
        {"status": 200, "ctype": J, "raw": json.dumps({"retCode": 0, "result": {"list": [[str(ms[-1] * 1000), "1", "1", "1", "1", repr(vols[ms[-1]])]]}})}])
    return {"name": "Bybit 대조", "venue": "bybit", "symbol": "ETHUSDT", "mono0": M0, "wall0": W0, "setup": setup,
            "steps": ["verify", "verify", "verify", "verify"], "responses": g.resp}


def ctval(rng) -> dict:
    p = {"instType": "SWAP", "instId": INST}
    body = lambda ct: {"code": "0", "data": [{"instId": INST, "ctVal": ct, "ctValCcy": "ETH", "ctType": "linear"}]}  # noqa: E731
    rs = [{"status": 200, "ctype": J, "raw": json.dumps(body("0.1"))}, {"status": 200, "ctype": J, "raw": json.dumps(body("0.01"))},
          {"status": 500, "ctype": "text/html", "raw": "x"}, {"status": 200, "ctype": J, "raw": '{"code":"0","data":[]}'}, {"timeout": True},
          {"status": 200, "ctype": J, "raw": json.dumps({"data": [{"ctVal": 0.1}]})}, {"status": 403, "ctype": J, "raw": json.dumps(body("0.1"))}]
    return {"name": "OKX ctVal", "venue": "ctval", "symbol": INST, "mono0": M0, "wall0": W0, "setup": [], "steps": ["ctval"] * len(rs),
            "responses": {f"{IU}?{urlencode(p)}": rs}}


def okx_edges(rng) -> list:
    """양성 대조 보강: 한 주기 15분(공백 분 16개 → 최근 15개만) · 창 7일(3일 전 공백도 후보) · 대조 구간 경계(앵커 - 100분 = 둘째 구간)."""
    out = []
    g = Gen(rng)
    ms = [NOW - 300 - 60 * k for k in range(16)]
    for m in ms[:15]:
        g.put(CU, {"instId": INST, "bar": "1m", "after": (m + 60) * 1000, "limit": 1}, ok([]))
    out.append({"name": "OKX 한 주기 15분", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0,
                "setup": [["gap", m * 1000, m * 1000 + 9_000, "ws_reconnect"] for m in ms], "steps": ["cycle"], "responses": g.resp})
    g = Gen(rng)
    old = NOW - 3 * 86400
    g.put(CU, {"instId": INST, "bar": "1m", "after": (old + 60) * 1000, "limit": 1}, ok([]))
    out.append({"name": "OKX 창 7일", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0,
                "setup": [["gap", old * 1000, old * 1000 + 9_000, "ws_reconnect"]], "steps": ["cycle"], "responses": g.resp})
    g = Gen(rng)
    a = NOW - 600
    setup = [["write", live_rows(g.trades(m, 4))] for m in (a, a - 100 * 60)]
    q = lambda x: {"instId": INST, "bar": "1m", "limit": "100", "after": str((x + 60) * 1000)}  # noqa: E731
    g.put(CU, q(a), ok([]))
    g.put(CU, q(a - 100 * 60), ok([]))
    out.append({"name": "OKX 대조 구간 경계", "venue": "okx", "symbol": INST, "mono0": M0, "wall0": W0, "setup": setup, "steps": ["verify"], "responses": g.resp})
    return out


# ── 원본 돌리기 ──────────────────────────────────────────────────────────────
class _Stop(BaseException):
    pass


class _Status(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status = status


def python_reference(sc, d: Path) -> dict:
    om, bn, bm = okx_mod(), okx_mod()._bn, bybit_mod()
    clock = {"mono": sc["mono0"], "wall": sc["wall0"]}
    loops = {"n": 0}

    def fake_sleep(s, *a, **k):
        if sys._getframe(1).f_code.co_name == "backfill_loop":
            loops["n"] += 1
            if loops["n"] > 1:
                raise _Stop()
        clock["mono"] += s
        clock["wall"] += s

        async def _c():
            return None
        return _c()

    async def direct(f, *a, **k):
        return f(*a, **k)

    sent, outcome, logs = [], [], []
    resp = {k: list(v) for k, v in sc["responses"].items()}

    class R:
        def __init__(self, r):
            self.status, self.r = r["status"], r

        def raise_for_status(self):
            if self.status >= 400:
                raise _Status(self.status)

        async def json(self):
            import re
            if not re.match(r"^application/(?:[\w.+-]+?\+)?json", self.r["ctype"].split(";")[0].strip().lower()):
                raise RuntimeError("ContentTypeError")
            return json.loads(self.r["raw"]) if self.r["raw"].strip() else None

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
        def get(self, url, params=None, headers=None, timeout=None):
            if "okx" in url:
                gv._want(headers == om.HTTP_HEADERS, headers)     # User-Agent(없으면 403)
            return Req(url, params)

    mods = (om, bn, bm)
    old = [(x.time, x.asyncio) for x in mods]
    for x in mods:
        x.time = gv._ns(time, time=lambda: clock["wall"], monotonic=lambda: clock["mono"])
        x.asyncio = gv._ns(asyncio, sleep=fake_sleep, to_thread=direct)
    old_log = om.log
    om.log = logs.append
    om._PACE["next"] = 0.0
    sym = sc["symbol"]
    db = d / ("bybit_tape.sqlite" if sc["venue"] == "bybit" else "okx_tape.sqlite")
    try:
        with gv._quiet():
            store = bn.TapeStore(db, sym, om.BUCKETS.get(sym, 0.01))
            for op in sc["setup"]:
                if op[0] == "write":
                    store.write([tuple(r) for r in op[1]])
                elif op[0] == "gap":
                    store.record_gap(op[1], op[2], op[3])
            for step in sc["steps"]:
                if step == "cycle":
                    loops["n"] = 0
                    fetch = om.okx_minute_fetcher(Sess(), sym, om.BUCKETS[sym], om.CT_VALS[sym])
                    try:   # 원본 collect() 의 호출 그대로(per_cycle 15 · lookback 기본)
                        asyncio.run(bn.backfill_loop(store, fetch, per_cycle=15, source="okx rest history-trades(type=1)",
                                                     note="주문 = 연속 (ts,side,px) 묶음 -- 라이브보다 주문수 ~7% 적음, 총량·체결수 정확"))
                    except _Stop:
                        pass
                elif step == "verify":
                    try:
                        if sc["venue"] == "bybit":
                            asyncio.run(bm.verify_recent(Sess(), {sym: store}))
                        else:
                            asyncio.run(om.verify_recent(store, Sess(), sym))
                        outcome.append("ok")
                    except Exception as e:  # noqa: BLE001 -- 원본에선 WS 루프의 except 가 받는다(재연결)
                        outcome.append(f"raised {type(e).__name__}")
                else:
                    logs.clear()
                    try:
                        asyncio.run(om.assert_ct_val(Sess(), sym, om.CT_VALS[sym]))
                        outcome.append("pass" if any(x.startswith("ctVal 확인") for x in logs) else "fail-open")
                    except SystemExit:
                        outcome.append("exit")
        gv.close_shared()
    finally:
        for x, (t, a) in zip(mods, old):
            x.time, x.asyncio = t, a
        om.log = old_log
        om._PACE["next"] = 0.0
    tables = {"trade_tape_1s": "symbol, ts_sec, price_bin", "gaps": "rowid", "verify_1m": "rowid", "backfill_1m": "rowid"}
    return {"name": sc["name"], "db": {db.name: gv.dump(db, tables)}, "requests": sent, "outcome": outcome,
            "unused": sorted(k for k, v in resp.items() if v)}


def main() -> None:
    rng = random.Random(SEED)
    scs = [okx_backfill(rng), okx_codes(rng), okx_verify(rng), bybit_verify(rng), ctval(rng)]
    scs += okx_edges(rng)
    out = []
    for sc in scs:
        with tempfile.TemporaryDirectory() as td:
            out.append(python_reference(sc, Path(td)))
        g = out[-1]
        t = g["db"][next(iter(g["db"]))]["tables"]
        print(f"{sc['name']}: 요청 {len(g['requests'])} · 결과 {g['outcome']} · 남은 응답 {len(g['unused'])} · "
              + " · ".join(f"{k} {len(v)}행" for k, v in t.items() if v))
    (FIX / "venue_rest_input.json").write_text(json.dumps({"seed": SEED, "scenarios": scs}, separators=(",", ":"), ensure_ascii=False))
    (FIX / "venue_rest_golden.json").write_text(json.dumps({"scenarios": out}, separators=(",", ":"), ensure_ascii=False))
    print("파일", *[f"{(FIX / f'venue_rest_{x}.json').stat().st_size // 1024}KB" for x in ("input", "golden")])


if __name__ == "__main__":
    main()
