#!/usr/bin/env python3
"""**OKX 컨텍스트 수집기** — 미결제약정·마크가격·펀딩·청산. (2026-09-23)

왜 한 파일인가: 넷 다 **같은 WS 엔드포인트**에 얹히고 합쳐서 **5.2 msg/s**(실측)뿐이다.
체결·호가처럼 포맷이 다른 것도 아니라 프로세스를 넷으로 쪼갤 이유가 없다. 반대로 체결·호가와
합치지 않는 이유는 그쪽이 초당 20~51건이라 duckdb 쓰기 지연이 호가 수집을 막을 수 있어서다.

🔴**OI 는 WS `open-interest` 로 받는다 -- REST 폴링은 «잡음»만 더한다.** (2026-09-23 정정)
  한때 REST 0.25초 폴링으로 바꿨었다(「REST 가 값 변화를 2.7배 더 본다」). 그런데 그 «변화»의
  **60%가 A->B->A 되돌림**이었다(재시작 후 418행 중 250) -- 응답 노드마다 값이 달라 왔다 갔다
  한 것이다. 동시 실측 75초: REST 가 본 고유 값 10개가 **WS 10개와 정확히 같았다**(10/10),
  REST 는 되돌림 4번만 더했다. 즉 거래소 OI 는 5~10초마다 갱신되고 WS 가 그걸 전부 준다.
  게다가 대시보드도 같은 IP 에서 0.25초로 폴링해 둘이 한도(20req/2s)의 80% 를 먹었고, 조사
  스크립트 하나가 더 붙자 **429** 가 났다. WS 는 요청 한도가 없고 `ts` 가 진짜 거래소 시각이다.
  ⚠️이 날 05:30~23:59 사이 `okx_oi` 행은 REST 폴링분이라 되돌림 잡음이 섞여 있다(meta 참고).

실측 페이로드(2026-09-23, ETH-USDT-SWAP, 150초):
  open-interest(REST)       {"oi":"5941713.19","oiCcy":"594171.319","oiUsd":"...","ts":...}
  mark-price         746건  {"markPx":"2745.02","ts":...}
  funding-rate         3건  {"fundingRate":"0.0000588","fundingTime":"...","min/maxFundingRate",...}
  liquidation-orders  10건  {"instId":"...","details":[{"bkPx","bkLoss","posSide","side","sz","ts"}]}

🔴**`oi` 와 `sz` 는 계약 수다.** OI 는 거래소가 `oiCcy`(기초자산 단위)를 같이 주므로 그걸 쓰고,
  **둘이 어긋나면 뜬다** -- 실측에서 `oi * 0.1 == oiCcy` 가 소수점까지 맞았다(5941713.19 →
  594171.319). 청산 `sz` 는 우리가 곱한다.
🔴**청산의 `bkPx` 는 파산가격이지 체결가가 아니다.** 바이낸스 `@forceOrder` 의 `p`(주문가)/
  `ap`(평균체결가)와 **같은 것이 아니다**. 두 거래소 청산가를 같은 축에 놓으면 안 된다.
⭐`side` 는 **강제 주문의 방향**이고 `posSide` 는 **청산된 포지션의 방향**이다(short 청산 →
  side=buy). 바이낸스 `@forceOrder` 의 `S` 와 같은 규약이라 `side` 끼리는 비교해도 된다.

⚠️청산은 `instType:SWAP` 으로 **전 종목**이 온다(실측 10건 중 ETH 는 1건 -- 희소하다). 전부
  적는다: 시장 전체 캐스케이드는 그 자체로 값이 있고 0.067 msg/s 라 공짜다. 다만 `ctVal` 을
  아는 종목만 `sz_base` 를 채우고 **나머지는 NULL** 이다 -- 0 이 아니라 NULL 인 것이 중요하다
  (「없었다」와 「환산 못 한다」는 다른 말이다. 체결 테이프의 크기구간 NULL 과 같은 규약).

봇과 완전 분리: 자기 WS · 자기 duckdb · 주문 없음. 바이낸스 REST weight 를 쓰지 않는다.

사용:
  python scripts/live_okx_context_collector_20260923.py                       # ETH-USDT-SWAP
  OKX_CTX_INST=BTC-USDT-SWAP python scripts/live_okx_context_collector_20260923.py
  python scripts/live_okx_context_collector_20260923.py --selftest            # 네트워크·DB 없이
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_tape = importlib.import_module("scripts.live_okx_trade_tape_collector_20260923")
CT_VALS = _tape.CT_VALS
assert_ct_val = _tape.assert_ct_val
log = _tape.log

# 🔴청산은 `instType=SWAP` 로 **전 종목**이 한 구독에 온다. 종목마다 프로세스를 띄우면 같은 청산이
#   프로세스 수만큼 중복 저장된다(2026-09-26). 그래서 ETH 프로세스 **하나만** 받는다 -- 다른 종목의
#   청산도 거기에 이미 있고, ctVal 을 아는 종목은 sz_base 까지 채워진다.
LIQ_INST = "ETH-USDT-SWAP"
# ponytail: 마크가격 5/s 를 **중복 제거 없이** 그대로 적는다(~440k행/일, ~10MB/일). 「값이
# 바뀔 때만」으로 줄이면 45% 쯤 아끼지만 «안 바뀐 것»과 «안 온 것»의 구분이 gaps 에만 남아
# 미묘해진다. 디스크가 급해지면 그때 바꾼다.
WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
FLUSH_SECONDS = 10.0
RECV_TIMEOUT = 40.0      # 마크는 5/s 지만 펀딩은 분 단위라 체결보다 여유를 둔다

SCHEMA = (
    """CREATE TABLE IF NOT EXISTS okx_oi(
         inst VARCHAR, ts_ms BIGINT, oi_contracts DOUBLE, oi_base DOUBLE, oi_usd DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS okx_mark(
         inst VARCHAR, ts_ms BIGINT, mark_px DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS okx_funding(
         inst VARCHAR, ts_ms BIGINT, funding_rate DOUBLE, funding_time BIGINT,
         min_rate DOUBLE, max_rate DOUBLE, method VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS okx_liquidations(
         inst_id VARCHAR, inst_family VARCHAR, ts_ms BIGINT, side VARCHAR, pos_side VARCHAR,
         bk_px DOUBLE, sz_contracts DOUBLE, sz_base DOUBLE, bk_loss DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS gaps(
         channel VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)""",
)
INSERTS = {"okx_oi": 5, "okx_mark": 3, "okx_funding": 7, "okx_liquidations": 9}


def _f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def parse_message(payload: dict, ct_vals: dict) -> dict[str, list[tuple]]:
    """WS 메시지 하나 → {표 이름: 행 목록}. 못 믿을 행은 조용히 버린다.

    🔴OI 는 거래소가 준 `oiCcy` 를 쓰되 `oi * ctVal` 과 대조한다 -- 어긋나면 계약 크기가
      바뀐 것이고, 그건 조용히 지나가면 안 되는 사건이다(호출자가 로그로 떠든다)."""
    out: dict[str, list[tuple]] = {}
    ch = (payload.get("arg") or {}).get("channel")
    rows = payload.get("data") or []
    if ch == "open-interest":
        for r in rows:
            inst, ts = str(r.get("instId", "")), int(_f(r.get("ts"), 0) or 0)
            oi, oi_ccy = _f(r.get("oi")), _f(r.get("oiCcy"))
            if not (inst and ts > 0 and oi is not None and oi_ccy is not None):
                continue
            out.setdefault("okx_oi", []).append((inst, ts, oi, oi_ccy, _f(r.get("oiUsd"))))
    elif ch == "mark-price":
        for r in rows:
            inst, ts, px = str(r.get("instId", "")), int(_f(r.get("ts"), 0) or 0), _f(r.get("markPx"))
            if inst and ts > 0 and px and px > 0:
                out.setdefault("okx_mark", []).append((inst, ts, px))
    elif ch == "funding-rate":
        for r in rows:
            inst, ts = str(r.get("instId", "")), int(_f(r.get("ts"), 0) or 0)
            rate = _f(r.get("fundingRate"))
            if not (inst and rate is not None):
                continue
            out.setdefault("okx_funding", []).append((
                inst, ts or int(time.time() * 1000), rate,
                int(_f(r.get("fundingTime"), 0) or 0), _f(r.get("minFundingRate")),
                _f(r.get("maxFundingRate")), str(r.get("method", ""))))
    elif ch == "liquidation-orders":
        for r in rows:
            inst_id, family = str(r.get("instId", "")), str(r.get("instFamily", ""))
            ct = ct_vals.get(inst_id)
            for d in r.get("details") or []:
                ts, sz = int(_f(d.get("ts"), 0) or 0), _f(d.get("sz"))
                if not (inst_id and ts > 0 and sz and sz > 0):
                    continue
                out.setdefault("okx_liquidations", []).append((
                    inst_id, family, ts, str(d.get("side", "")), str(d.get("posSide", "")),
                    _f(d.get("bkPx")), sz,
                    sz * ct if ct is not None else None,   # 🔴모르는 종목은 NULL
                    _f(d.get("bkLoss"))))
    return out


def oi_mismatch(rows: list[tuple], ct_vals: dict) -> tuple | None:
    """`oi * ctVal != oiCcy` 인 첫 행. 계약 크기가 갈라진 순간을 잡는다."""
    for inst, ts, oi, oi_ccy, _usd in rows:
        ct = ct_vals.get(inst)
        if ct is None or oi_ccy == 0:
            continue
        if abs(oi * ct - oi_ccy) > abs(oi_ccy) * 1e-9:
            return (inst, ts, oi, oi_ccy, oi * ct)
    return None


class ContextStore:
    """체결 테이프와 같은 규약: **쓸 때만 연결하고 바로 닫는다**(붙들면 감시기·연구 쿼리가
    BLOCKED 되어 거짓 CRITICAL 이 난다). 못 쓴 행은 버리지 않고 다음 주기에 재시도한다."""

    PENDING_CAP = 200_000

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.pending: dict[str, list[tuple]] = {}
        with self._connect() as con:
            for ddl in SCHEMA:
                con.execute(ddl)

    def _connect(self):
        return _tape._bn.duckdb_connect_retry(self.db_path)   # 읽는 쪽과 잠금 충돌이면 잠깐 기다린다

    def set_meta(self, pairs) -> None:
        with self._connect() as con:
            for key, value in pairs:
                con.execute("DELETE FROM meta WHERE key = ?", [key])
                con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    def stage(self, batch: dict[str, list[tuple]]) -> None:
        for table, rows in batch.items():
            self.pending.setdefault(table, []).extend(rows)

    async def aflush(self) -> int:
        """쓰기를 스레드로 뺀다 -- close() 의 체크포인트 fsync(서버 ~0.5초)가 WS 수신을 막지 않게.
        pending 은 루프 스레드에서만 만진다: 떼어서 넘기고, 실패하면 앞에 되돌려 붙인다."""
        total = sum(len(v) for v in self.pending.values())
        if not total:
            return 0
        batch, self.pending = self.pending, {}
        exc = await asyncio.to_thread(self._write, batch)
        if exc is None:
            return total
        for table, rows in batch.items():
            self.pending[table] = rows + self.pending.get(table, [])
        if total > self.PENDING_CAP:
            self.pending.clear()
            log(f"⚠️쓰기가 계속 막혀 {total}행 버림 -- 그 구간은 소급 복원 불가다")
        else:
            log(f"쓰기 보류 {total}행, 다음 주기 재시도: {type(exc).__name__}")
        return 0

    def _write(self, batch: dict[str, list[tuple]]) -> Exception | None:
        try:
            with self._connect() as con:
                # 🔴트랜잭션 하나로 -- 자동커밋이면 행마다 fsync 라 10초 주기의 ~50행이 8~10초
                #   걸려 락을 늘 쥐고(읽기 0/60) OI 폴러가 굶었다(2026-09-23 실측).
                con.begin()
                for table, rows in batch.items():
                    if rows:
                        marks = ",".join("?" * INSERTS[table])
                        con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
                con.commit()
            return None
        except Exception as exc:  # noqa: BLE001 -- 대개 읽는 쪽이 잡고 있는 락이다
            return exc

    def record_gap(self, channel: str, from_ms: int, to_ms: int, reason: str) -> None:
        if to_ms - from_ms < 1500:
            return
        try:
            with self._connect() as con:
                con.execute("INSERT INTO gaps VALUES (?,?,?,?)", [channel, from_ms, to_ms, reason])
            log(f"gap {(to_ms - from_ms) / 1000:.1f}s 기록 ({reason})")
        except Exception as exc:  # noqa: BLE001
            log(f"gap 기록 실패(수집은 계속): {type(exc).__name__}")


async def collect(inst: str, db_path: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    ct_val = CT_VALS.get(inst)
    if ct_val is None:
        raise SystemExit(f"🔴{inst} 의 ctVal 을 모른다 -- CT_VALS 에 실측값을 적고 다시 돌릴 것")
    liqs = inst == LIQ_INST
    store = ContextStore(db_path)
    store.set_meta([(f"ct_val:{inst}", repr(ct_val)),
                    ("liquidation_scope", "instType=SWAP (전 종목) · sz_base 는 ctVal 아는 것만" if liqs
                     else f"없음 -- {LIQ_INST} 프로세스(okx_context.duckdb)가 전 종목을 받는다"),
                    ("bk_px_note", "파산가격이다 -- 바이낸스 forceOrder 의 체결가와 다르다"),
                    ("okx_oi_note", "2026-09-23 05:30~23:59 KST 행은 REST 0.25초 폴링분이라 "
                     "A->B->A 되돌림(응답 노드 불일치) 잡음이 ~60% 섞여 있다. 그 뒤는 WS "
                     "open-interest(진짜 거래소 ts, 5~10초 간격)")])
    args = [{"channel": c, "instId": inst}
            for c in ("open-interest", "mark-price", "funding-rate")]
    if liqs:
        args.append({"channel": "liquidation-orders", "instType": "SWAP"})
    log(f"{inst} 컨텍스트 수집 시작 (ctVal {ct_val}, db {db_path})")
    down_from = int(time.time() * 1000)
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        await assert_ct_val(session, inst, ct_val)
        await _ws_loop(session, inst, args, store, down_from)


async def _ws_loop(session, inst: str, args: list, store: "ContextStore",
                   down_from: int) -> None:
    from aiohttp import WSMsgType
    flushed_at = time.monotonic()
    wrote = 0
    while True:
        try:
            async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                await ws.send_json({"op": "subscribe", "args": args})
                store.record_gap("all", down_from, int(time.time() * 1000), "reconnect")
                log("스트림 연결됨")
                while True:
                    msg = await ws.receive(timeout=RECV_TIMEOUT)
                    if msg.type is not WSMsgType.TEXT:
                        break
                    if msg.data == "pong":
                        continue
                    try:
                        payload = json.loads(msg.data)
                    except ValueError:
                        continue
                    if payload.get("event") == "error":
                        log(f"🔴구독 거부: {payload.get('msg')} (code {payload.get('code')})")
                        break
                    if payload.get("event"):
                        continue
                    batch = parse_message(payload, CT_VALS)
                    bad = oi_mismatch(batch.get("okx_oi", []), CT_VALS)
                    if bad:
                        log(f"🔴OI 단위가 갈라졌다 {bad[0]}: oiCcy {bad[3]} vs oi*ctVal "
                            f"{bad[4]} -- CT_VALS 를 확인할 것(계속 쌓기는 한다)")
                    store.stage(batch)
                    now = time.monotonic()
                    if now - flushed_at >= FLUSH_SECONDS:
                        flushed_at = now
                        wrote += await store.aflush()
                        if wrote and wrote % 5000 < 200:
                            log(f"누적 {wrote:,}행")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            log(f"{RECV_TIMEOUT:.0f}초 무음 -- 재연결")
        except Exception as exc:  # noqa: BLE001
            log(f"연결 실패, 3초 뒤 재시도: {type(exc).__name__} {exc}")
        down_from = int(time.time() * 1000)
        await store.aflush()
        await asyncio.sleep(3.0)


def selftest() -> None:
    """네트워크·DB 없이 파싱 규칙만 점검한다. 페이로드는 2026-09-23 실측 원문이다."""
    ct = {"ETH-USDT-SWAP": 0.1}

    oi = parse_message({"arg": {"channel": "open-interest"}, "data": [
        {"instId": "ETH-USDT-SWAP", "instType": "SWAP", "oi": "5941713.19000001727",
         "oiCcy": "594171.319000001727", "oiUsd": "1631065629.5", "ts": "1790092405208"}]}, ct)
    assert list(oi) == ["okx_oi"], oi
    r = oi["okx_oi"][0]
    assert r[0] == "ETH-USDT-SWAP" and r[1] == 1790092405208
    assert abs(r[2] * 0.1 - r[3]) < 1e-6, ("oi * ctVal 이 oiCcy 와 맞아야 한다", r)
    assert oi_mismatch(oi["okx_oi"], ct) is None
    # 계약 크기가 갈라지면 잡는가 (ctVal 을 10배로 속여 본다)
    assert oi_mismatch(oi["okx_oi"], {"ETH-USDT-SWAP": 1.0}) is not None, "단위 갈라짐을 못 잡았다"
    assert oi_mismatch(oi["okx_oi"], {}) is None, "모르는 종목은 판단하지 않는다"

    mk = parse_message({"arg": {"channel": "mark-price"}, "data": [
        {"instId": "ETH-USDT-SWAP", "markPx": "2745.02", "ts": "1790092414237"}]}, ct)
    assert mk["okx_mark"] == [("ETH-USDT-SWAP", 1790092414237, 2745.02)], mk
    assert parse_message({"arg": {"channel": "mark-price"}, "data": [
        {"instId": "ETH-USDT-SWAP", "markPx": "0", "ts": "1"}]}, ct) == {}, "0 가격은 버린다"

    fr = parse_message({"arg": {"channel": "funding-rate"}, "data": [
        {"instId": "ETH-USDT-SWAP", "fundingRate": "0.0000588298968880",
         "fundingTime": "1790092800000", "minFundingRate": "-0.0075",
         "maxFundingRate": "0.0075", "method": "current_period", "ts": "1790092500000"}]}, ct)
    f = fr["okx_funding"][0]
    assert (f[2], f[3], f[5], f[6]) == (0.0000588298968880, 1790092800000, 0.0075,
                                        "current_period"), f

    lq = parse_message({"arg": {"channel": "liquidation-orders"}, "data": [
        {"instId": "ETH-USDT-SWAP", "instFamily": "ETH-USDT", "instType": "SWAP", "details": [
            {"bkLoss": "0", "bkPx": "2735.52", "ccy": "", "posSide": "short",
             "side": "buy", "sz": "7.97", "ts": "1790088612984"}]},
        {"instId": "PENGU-USDT-SWAP", "instFamily": "PENGU-USDT", "instType": "SWAP", "details": [
            {"bkLoss": "0", "bkPx": "0.009151", "posSide": "long", "side": "sell",
             "sz": "388", "ts": "1790092422591"}]}]}, ct)
    rows = lq["okx_liquidations"]
    assert len(rows) == 2, "모르는 종목도 버리지 않는다"
    eth = [x for x in rows if x[0] == "ETH-USDT-SWAP"][0]
    assert (eth[3], eth[4], eth[5], eth[6]) == ("buy", "short", 2735.52, 7.97), eth
    assert abs(eth[7] - 0.797) < 1e-12, ("sz 는 계약 수다", eth)
    pengu = [x for x in rows if x[0] == "PENGU-USDT-SWAP"][0]
    assert pengu[7] is None, ("ctVal 을 모르면 0 이 아니라 NULL 이다", pengu)

    assert parse_message({"arg": {"channel": "tickers"}, "data": [{}]}, ct) == {}
    assert parse_message({"event": "subscribe"}, ct) == {}
    for table, width in INSERTS.items():
        got = {"okx_oi": oi.get("okx_oi"), "okx_mark": mk.get("okx_mark"),
               "okx_funding": fr.get("okx_funding"), "okx_liquidations": rows}[table]
        assert all(len(x) == width for x in got), (table, width, got)
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inst", default=os.getenv("OKX_CTX_INST", "ETH-USDT-SWAP").upper())
    ap.add_argument("--db", type=Path, default=Path(os.environ["OKX_CTX_DB_PATH"])
                    if os.getenv("OKX_CTX_DB_PATH") else None)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    try:
        asyncio.run(collect(a.inst, a.db or _tape.default_db(a.inst, "okx_context")))
    except KeyboardInterrupt:
        log("종료")


if __name__ == "__main__":
    main()
