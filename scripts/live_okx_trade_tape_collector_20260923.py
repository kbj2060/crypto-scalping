#!/usr/bin/env python3
"""**OKX 체결 테이프 수집기** — 바이낸스 수급이 못 보는 나머지 절반. (2026-09-23)

왜 필요한가: 이 저장소의 「수급」은 `dashboard/server.py::supply_1s_cell` 기준으로 **바이낸스
ETHUSDT 무기한의 테이커 델타 하나**다. 그런데 바이낸스는 ETH 테이커 플로우의 **38~43%**뿐이다
(2026-09-17~19 3일 |순델타| 점유: bn_perp 41.6 · **okx 34.8** · bybit 19.1 · bn_spot 4.5%).
사용자 관찰 「수급이 다 플러스인데 가격이 떨어질 때가 있다」를 백필로 판정한 결과:

  동시각 창에서 BN CVD>0 인데 하락 = 창의 10.3%(1분) / 9.7%(5분).
  그중 **OKX 를 더하면 부호가 음수로 뒤집히는 것 30.9%(1분) / 46.4%(5분)**,
  대조군(같은 조건인데 상승)은 7.4% / 5.0% ⇒ **순이득 +23.5 / +41.4pp**, 3일 전부 같은 부호.

🔴**이 수집기가 고치는 것은 «설명»이지 «예측»이 아니다.** 같은 판정에서 다음 창 수익률 부호
  적중은 BN단독 49.74% → 합산4 49.84%(SE ±0.76), IC −0.0137 → −0.0171 로 **둘 다 동전**이었다.
  CVD 는 동행지표지 선행지표가 아니다. 이 데이터를 「예측이 좋아진다」는 근거로 쓰면 안 된다.
🔴그리고 합산해도 «전 거래소 매수 우위인데 하락» 이 61.6%(1분) 남는다 — 그건 **흡수**(매도
  리밋벽이 테이커 매수를 다 받아냄)이고 CVD 로는 못 푼다. 호가(depthDiff) 쪽 일이다.
  전체 기록: 이 파일 최초 커밋 메시지와 memory/cvd_multivenue_explains_not_predicts_20260922.md.

⭐**바이낸스 수집기의 되묶기(TakerOrderAggregator)를 이식하면 안 된다 — 이중 집계가 된다.**
  2026-09-23 실측(90초, ETH-USDT-SWAP, 두 채널 동시):
      `trades-all`(business)  4,810건 · tradeId 증분 **항상 1** = 개별 체결   ← 바이낸스 @trade
      `trades`    (public)    1,780건 · `count` 필드 보유       = 테이커 주문 ← 바이낸스 aggTrade
      sz 합은 **28,855.6 계약으로 정확히 동일**(비율 1.0000x), 건수만 2.70x.
  즉 OKX 는 주문 단위를 **정답으로 직접** 준다. 바이낸스에서 ±1% 오차를 감수하며 재구성했던
  일을 여기서는 할 필요가 없고, 하면 이미 묶인 것을 또 묶는다.
⭐그래서 `trades` **한 채널만** 받는다. `count` 를 더하면 개별 체결 수가 정확히 나오므로
  `trades-all` 을 같이 받을 이유도 없다(2026-09-23 실측: count 합 11,492 vs trades-all 행수
  11,494 — 차 2건은 대조 창의 경계 효과이고, tradeId 구간에 구멍·겹침이 없음을 함께 확인했다).
⭐한 주문의 모든 체결이 **같은 ms** 다(같은 실측에서 첫 체결 ts 일치 5,442/5,442 · 마지막 체결
  ts 일치 5,442/5,442). 바이낸스에서 「주문이 초 경계를 걸쳐 구간 합 > 총량」이 됐던 문제가
  여기서는 원천적으로 없다.

🔴**`sz` 는 ETH 가 아니라 계약 수다**(ETH-USDT-SWAP `ctVal`=0.1 ETH, API 확인). 안 곱하면 이
  거래소가 합산 델타를 10배로 혼자 지배한다. 시작할 때 REST 로 대조하고, **다르면 뜬다**(아래
  `assert_ct_val`). 값이 조용히 틀리느니 안 도는 게 낫다.
🔴**`side` 는 테이커 측면 그 자체다**(buy/sell). 바이낸스 `m`(매수자가 메이커)처럼 뒤집으면
  안 된다 -- 부호가 통째로 반대가 된다.

⚠️**`buy_max`/`sell_max` 는 바이낸스와 뜻이 다르다.** 바이낸스 표에서 이 칸은 「그 초의 가장 큰
  **개별 체결**」인데, 여기서는 「가장 큰 **테이커 주문**」이다(개별 체결이 아예 안 오므로 달리
  할 방법이 없다). `meta` 표의 `max_unit` 에 박아 둔다 -- 두 표를 UNION 하는 쿼리는 이 칸만
  빼거나 단위를 맞춰야 한다. 나머지 칸(총량·건수·크기구간·주문수)은 뜻이 정확히 같다.

크기 경계(리테일 $10k / 고래 $100k)는 **바이낸스 수집기에서 import 한다** -- 복사하면 언젠가
한쪽만 바뀌고, 그날부터 두 화면의 「고래」가 다른 것을 뜻하게 된다.

봇과 완전 분리: 자기 WS · 자기 duckdb · 주문 없음 · 죽어도 봇에 영향 없다. 바이낸스 REST
weight 를 한 톨도 쓰지 않는다(다른 거래소다).

사용:
  python scripts/live_okx_trade_tape_collector_20260923.py                    # ETH-USDT-SWAP
  OKX_TAPE_INST=BTC-USDT-SWAP python scripts/live_okx_trade_tape_collector_20260923.py
  python scripts/live_okx_trade_tape_collector_20260923.py --selftest         # 네트워크·DB 없이
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
if str(ROOT) not in sys.path:                      # supervisor 밖에서 직접 실행할 때
    sys.path.insert(0, str(ROOT))

# 🔴복사가 아니라 import 다. 경계 상수(RETAIL/WHALE)와 초 버퍼의 «닫힌 초는 되살리지 않는다»
#   규칙은 바이낸스 표와 **같은 것이어야** 두 표를 나란히 읽을 수 있다.
_bn = importlib.import_module("scripts.live_trade_tape_collector_20260916")
TapeBuffer = _bn.TapeBuffer
TapeStore = _bn.TapeStore
RETAIL_MAX_USD = _bn.RETAIL_MAX_USD
WHALE_MIN_USD = _bn.WHALE_MIN_USD
log = _bn.log

DEFAULT_DB = ROOT / "data" / "live" / "okx_trade_tape.duckdb"
# ponytail: 바이낸스 테이프와 같은 «단일 duckdb» 다 -- 회수 경로가 없다(DELETE 는 파일을 안
# 줄인다). 하이퍼리퀴드 수집기가 겪고 날짜별 파일로 옮긴 그 문제다. ~330k행/일이라 당장은
# 문제가 아니고, 디스크가 급해지면 날짜별 분할이 업그레이드 경로다.

WS_URL = "wss://ws.okx.com:8443/ws/v5/public"
CANDLES_URL = "https://www.okx.com/api/v5/market/candles"
INSTRUMENTS_URL = "https://www.okx.com/api/v5/public/instruments"
HISTORY_TRADES_URL = "https://www.okx.com/api/v5/market/history-trades"
HISTORY_CANDLES_URL = "https://www.okx.com/api/v5/market/history-candles"
# 🔴User-Agent 가 없으면 OKX REST 가 **403** 을 준다(2026-09-23 실측). WS 는 상관없다.
HTTP_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; crypto-scalping-collector/1.0)"}

# 가격빈은 바이낸스 표와 **같은 뜻**이어야 한다(대략 가격의 0.4bp). ETH 는 양쪽 다 0.1.
BUCKETS = {"ETH-USDT-SWAP": 0.1, "BTC-USDT-SWAP": 1.0, "SOL-USDT-SWAP": 0.01}
# 계약당 기초자산 수량. 하드코딩하는 이유는 바이낸스 수집기가 경계를 하드코딩한 이유와 같다 --
# 쌓인 행의 뜻이 영원히 고정돼야 한다. 대신 시작할 때 REST 와 대조해서 «갈라진 순간» 에 뜬다.
CT_VALS = {"ETH-USDT-SWAP": 0.1, "BTC-USDT-SWAP": 0.01, "SOL-USDT-SWAP": 1.0}

FLUSH_SECONDS = 5.0
VERIFY_SECONDS = 300.0
VERIFY_TOLERANCE = 1e-4
RECV_TIMEOUT = 25.0      # 이 시간 조용하면 "ping" 을 보낸다(OKX 는 30초 무응답이면 끊는다)


class OkxTapeBuffer(TapeBuffer):
    """바이낸스 버퍼와 같은 칸 구조인데, **주문 하나가 한 번에 모든 칸을 채운다**.

    바이낸스는 개별 체결(`add`)과 되묶은 주문(`add_order`)이 따로 들어오지만, OKX `trades` 는
    이미 주문 단위이고 `count` 로 개별 체결 수를 함께 준다. 그래서 호출이 하나다.
    🔴그 결과 `buy_max`(4~5번 칸)의 단위만 달라진다 -- 도크스트링 ⚠️ 항목 참고."""

    add_okx_order = TapeBuffer.add_agg      # 같은 규칙을 공용 버퍼가 갖고 있다(REST 복구도 쓴다)


def parse_trade(t: dict, ct_val: float) -> tuple[int, float, float, bool, int] | None:
    """OKX `trades` 한 행 → (ts_ms, price, qty_eth, sell, fills). 못 믿을 행은 None.

    🔴`sz` 는 계약 수다. `side` 는 **테이커** 측면이라 뒤집지 않는다."""
    try:
        price = float(t["px"])
        contracts = float(t["sz"])
        ts_ms = int(t["ts"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (price > 0 and contracts > 0 and ts_ms > 0):
        return None
    side = str(t.get("side", "")).lower()
    if side not in ("buy", "sell"):
        return None
    try:
        fills = int(t.get("count") or 1)
    except (TypeError, ValueError):
        fills = 1
    return ts_ms, price, contracts * ct_val, side == "sell", max(fills, 1)


async def assert_ct_val(session, inst: str, expected: float) -> None:
    """계약 크기가 하드코딩과 갈라졌으면 **뜬다**.

    🔴여기서 관대하면 안 된다 -- ctVal 이 틀리면 이 거래소의 델타가 통째로 10배가 되고,
      합산 CVD 는 조용히 OKX 하나가 지배하게 된다. 그 상태로 며칠 쌓이면 판정이 전부 거짓이다.
    ⭐단 **REST 가 안 될 때는 통과시킨다** -- 「다르다는 증거」가 있을 때만 멈춘다. 거래소
      API 가 잠깐 죽었다고 수집기가 안 뜨면 그 시간은 영구 손실이다."""
    try:
        async with session.get(INSTRUMENTS_URL, headers=HTTP_HEADERS,
                               params={"instType": "SWAP", "instId": inst}) as r:
            body = await r.json()
        got = float(body["data"][0]["ctVal"])
        ccy = str(body["data"][0].get("ctValCcy", ""))
    except Exception as exc:  # noqa: BLE001
        log(f"⚠️ctVal 대조 실패({type(exc).__name__}) -- 하드코딩 {expected} 로 계속한다")
        return
    if abs(got - expected) > 1e-12:
        raise SystemExit(
            f"🔴{inst} 의 ctVal 이 갈라졌다: 코드 {expected} vs 거래소 {got} ({ccy}). "
            "이대로 쌓으면 물량이 통째로 틀린다 -- CT_VALS 를 고치고 이미 쌓인 행을 재환산할 것.")
    log(f"ctVal 확인 {inst} = {got} {ccy}")


async def verify_recent(store, session, inst: str) -> None:
    """직전 분들을 OKX kline 과 대조해 `verify_1m` 에 남긴다. 실패해도 수집은 계속한다.

    🔴`confirm`=="0" 은 **형성 중인 봉**이라 반드시 뺀다. 넣으면 매 검사마다 「부족하다」가
      뜨고, 진짜 유실과 구별이 안 된다.
    ⭐물량은 `volCcy`(ETH)를 쓴다 -- `vol` 은 계약 수다. 2026-09-23 실측에서 5개 봉 전부
      `vol * 0.1 == volCcy` 로 정확히 일치했다."""
    minutes = store.unverified_minutes()
    if not minutes:
        return
    # 🔴`candles` 는 최근 100분뿐이라 밀린 분을 영영 못 봤다 -> 과거용 history-candles(요청당 100분).
    # 🔴그리고 **구간마다** 받는다: 묶음의 최근 분 하나에만 맞춰 100분을 받으면, 새 분이 늘 최근에
    #   생기므로 100분보다 오래된 밀린 분은 영원히 범위 밖이었다(2026-09-24: 19:48 이전 65분 정체).
    kvol: dict[int, float] = {}
    left = sorted(minutes, reverse=True)
    for _ in range(6):                        # 30분이 흩어져도 보통 2~4번이면 덮는다(한도 20/2초)
        if not left:
            break
        anchor = left[0]
        try:
            async with session.get(HISTORY_CANDLES_URL, headers=HTTP_HEADERS,
                                   params={"instId": inst, "bar": "1m", "limit": "100",
                                           "after": str((anchor + 60) * 1000)}) as r:
                if r.status != 200:
                    break
                body = await r.json()
        except Exception as exc:  # noqa: BLE001
            log(f"kline 조회 실패(수집은 계속): {type(exc).__name__}")
            break
        kvol.update({int(row[0]) // 1000: float(row[6])
                     for row in body.get("data", []) if str(row[8]) == "1"})
        left = [m for m in left if m < anchor - 99 * 60]
        await asyncio.sleep(0.12)
    if not kvol:
        return
    results = store.verify(kvol, minutes)
    ok = [r for r in results if abs(r[1]) <= VERIFY_TOLERANCE]
    if ok:
        log(f"완전성 OK {len(ok)}분 "
            f"(최근 {time.strftime('%H:%M', time.localtime(max(r[0] for r in ok)))})")
    for ts_min, rel, gapped in results:
        if abs(rel) <= VERIFY_TOLERANCE:
            continue
        stamp = time.strftime("%H:%M", time.localtime(ts_min))
        if gapped:
            log(f"완전성 {stamp} rel_err {rel:+.4%} -- 기록된 공백과 겹친다(예상된 부족)")
        else:
            log(f"⚠️완전성 {stamp} rel_err {rel:+.4%} -- 체결 유실. OKX 일별 덤프"
                "(okx.com/cdn/okex/traderecords, 🔴파일 경계 UTC+8)로 덮어써야 한다")


def group_fills(fills: list[dict]) -> list[tuple[int, float, float, bool, int]]:
    """REST 개별 체결(tradeId 순) -> 라이브 `trades` 와 같은 묶음 (ts_ms, px, sz계약, sell, 체결수).

    2026-09-24 같은 분 WS/REST 동시 실측: 연속된 (ts, side, px) 묶음 191개가 WS 메시지와 **수량까지
    191/191 일치**했다. 다만 WS 는 같은 키를 다시 쪼개 메시지가 205개였다 -- 복구한 분의 «주문 수»는
    라이브보다 ~7% 적게 나온다(총량·체결수는 정확). backfill_1m.note 에 적는다."""
    out: list[list] = []
    key = None
    for f in fills:
        k = (f["ts"], f["side"], f["px"])
        if k == key:
            out[-1][2] += float(f["sz"])
            out[-1][4] += 1
        else:
            out.append([int(f["ts"]), float(f["px"]), float(f["sz"]), f["side"] == "sell", 1])
            key = k
    return [tuple(x) for x in out]


def okx_minute_fetcher(session, inst: str, bucket: float, ct_val: float):
    """OKX REST 로 한 분을 다시 만든다. 🔴페이지는 **tradeId(type=1)** 로 넘긴다 -- ts(type=2)로
    넘기면 같은 ms 의 체결이 경계에서 빠져 2026-09-24 실측 −9.4% 였다(tradeId 로는 연속 1,853건,
    1분봉 volCcy 와 오차 0). 요청 한도 20회/2초 -- 사이에 0.12초 쉰다."""
    async def get(url: str, params: dict) -> list:
        async with session.get(url, headers=HTTP_HEADERS, params=params) as r:
            r.raise_for_status()
            body = await r.json()
        if str(body.get("code")) != "0":
            raise RuntimeError(f"okx {body.get('code')} {body.get('msg')}")
        return body.get("data") or []

    async def fetch(m: int):
        start, end = m * 1000, (m + 60) * 1000
        c = await get(HISTORY_CANDLES_URL, {"instId": inst, "bar": "1m", "after": end, "limit": 1})
        if not c or int(c[0][0]) != start or str(c[0][8]) != "1":
            return [], None
        edge = await get(HISTORY_TRADES_URL, {"instId": inst, "type": 2, "after": end + 1, "limit": 1})
        fills: list[dict] = []
        after = int(edge[0]["tradeId"]) + 1 if edge else None
        while after is not None:
            await asyncio.sleep(0.12)
            page = await get(HISTORY_TRADES_URL, {"instId": inst, "type": 1, "after": after,
                                                  "limit": 100})
            if not page:
                break
            fills += [f for f in page if start <= int(f["ts"]) < end]
            after = int(page[-1]["tradeId"])
            if int(page[-1]["ts"]) < start or len(page) < 100:
                break
        fills.sort(key=lambda f: int(f["tradeId"]))
        buf = OkxTapeBuffer(bucket)
        for ts, px, sz, sell, n in group_fills(fills):
            buf.add_okx_order(ts, px, sz * ct_val, sell, n)
        return buf.take_closed(everything=True), float(c[0][6])
    return fetch


async def collect(inst: str, db_path: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    ct_val = CT_VALS.get(inst)
    if ct_val is None:
        raise SystemExit(f"🔴{inst} 의 ctVal 을 모른다 -- CT_VALS 에 실측값을 적고 다시 돌릴 것")
    bucket = BUCKETS.get(inst, 0.01)
    store = TapeStore(db_path, inst, bucket)
    with store._connect() as con:                       # 단위를 데이터와 함께 남긴다
        for key, value in ((f"ct_val:{inst}", repr(ct_val)),
                           (f"max_unit:{inst}", "taker_order"),
                           (f"source:{inst}", "okx ws v5 public trades (already order-aggregated)")):
            con.execute("DELETE FROM meta WHERE key = ?", [key])
            con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])
    buffer = OkxTapeBuffer(bucket)
    last_ms = store.last_ts_ms()
    log(f"{inst} 수집 시작 (빈 {bucket}, ctVal {ct_val}, db {db_path})")
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        await assert_ct_val(session, inst, ct_val)
        backfill = asyncio.create_task(_bn.backfill_loop(  # noqa: F841 -- 수집이 끝날 때까지 돈다
            store, okx_minute_fetcher(session, inst, bucket, ct_val), per_cycle=5,
            source="okx rest history-trades(type=1)",
            note="주문 = 연속 (ts,side,px) 묶음 -- 라이브보다 주문수 ~7% 적음, 총량·체결수 정확"))
        flushed_at = verified_at = time.monotonic()
        while True:
            try:
                async with session.ws_connect(WS_URL, heartbeat=20) as ws:
                    await ws.send_json({"op": "subscribe",
                                        "args": [{"channel": "trades", "instId": inst}]})
                    first = True
                    while True:
                        msg = await ws.receive(timeout=RECV_TIMEOUT)
                        if msg.type is not WSMsgType.TEXT:
                            break                       # CLOSE/ERROR -- 바깥에서 재연결
                        if msg.data == "pong":
                            continue
                        try:
                            payload = json.loads(msg.data)
                        except ValueError:
                            continue
                        if payload.get("event") == "error":
                            log(f"🔴구독 거부: {payload.get('msg')} (code {payload.get('code')})")
                            break
                        if payload.get("event"):         # subscribe ack 등
                            continue
                        if (payload.get("arg") or {}).get("channel") != "trades":
                            continue
                        for row in payload.get("data") or []:
                            parsed = parse_trade(row, ct_val)
                            if parsed is None:
                                continue
                            ts_ms, price, qty, sell, fills = parsed
                            if first:
                                first = False
                                store.record_gap(
                                    last_ms or (ts_ms // 60_000) * 60_000, ts_ms,
                                    "ws_reconnect" if last_ms else "startup")
                                log("스트림 연결됨")
                            buffer.add_okx_order(ts_ms, price, qty, sell, fills)
                            last_ms = ts_ms
                        now = time.monotonic()
                        if now - flushed_at >= FLUSH_SECONDS:
                            flushed_at = now
                            await asyncio.to_thread(store.write, buffer.take_closed())  # 스레드로: close() 체크포인트 fsync(~0.5초)가 WS 수신을 막지 않게
                        if now - verified_at >= VERIFY_SECONDS:
                            verified_at = now
                            await verify_recent(store, session, inst)
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                log("25초 무음 -- 재연결")               # ping 을 보내느니 다시 붙는 게 짧다
            except Exception as exc:  # noqa: BLE001
                log(f"연결 실패, 3초 뒤 재시도: {type(exc).__name__} {exc}")
            await asyncio.to_thread(store.write, buffer.take_closed())
            await asyncio.sleep(3.0)


def selftest() -> None:
    """네트워크·DB 없이 OKX 고유 규칙만 점검한다."""
    # ── 파싱: 계약→ETH · 테이커 측면 · count ───────────────────────────────
    row = {"instId": "ETH-USDT-SWAP", "tradeId": "1", "px": "2500.0", "sz": "40",
           "side": "sell", "ts": "1790000000000", "count": "7"}
    ts, px, qty, sell, fills = parse_trade(row, 0.1)
    assert (ts, px, sell, fills) == (1790000000000, 2500.0, True, 7), (ts, px, sell, fills)
    assert qty == 4.0, ("sz 는 계약 수다 -- ctVal 을 곱해야 ETH 다", qty)
    assert parse_trade({**row, "side": "buy"}, 0.1)[3] is False, "side 는 뒤집지 않는다"
    assert parse_trade({**row, "count": None}, 0.1)[4] == 1, "count 가 없으면 체결 1건으로 본다"
    for bad in ({**row, "px": "0"}, {**row, "sz": "0"}, {**row, "side": "NA"},
                {**row, "ts": "0"}, {k: v for k, v in row.items() if k != "px"}):
        assert parse_trade(bad, 0.1) is None, ("못 믿을 행은 버린다", bad)

    # ── 칸 채우기: 바이낸스 표와 같은 자리에 같은 뜻이 들어가는가 ─────────────
    buf = OkxTapeBuffer(0.1)
    buf.add_okx_order(2_000_000, 2500.0, 3.9, sell=False, fills=2)    # $9,750   리테일
    buf.add_okx_order(2_000_100, 2500.0, 4.0, sell=False, fills=1)    # $10,000  중형(경계 제외)
    buf.add_okx_order(2_000_200, 2500.0, 39.9, sell=False, fills=5)   # $99,750  중형
    buf.add_okx_order(2_000_300, 2500.0, 40.0, sell=True, fills=9)    # $100,000 고래(경계 포함)
    buf.add_okx_order(2_001_000, 2500.0, 1.0, sell=False, fills=1)    # 다음 초 -- 위 초를 닫는다
    r = buf.take_closed()[0]
    assert r[:2] == (2000, 25000), r
    assert (r[2], r[3]) == (47.8, 40.0), ("총량", r)
    assert (r[4], r[5]) == (8, 9), ("건수는 count 합 = 개별 체결 수", r)
    assert (r[6], r[7]) == (39.9, 40.0), ("최대는 **주문** 기준이다", r)
    assert (r[8], r[9]) == (3.9, 0.0), ("리테일 물량", r)
    assert (r[14], r[15]) == (1, 0), ("리테일 건수", r)
    assert (r[10], r[11], r[12], r[13]) == (0.0, 40.0, 0, 1), ("고래", r)
    assert (r[16], r[17]) == (3, 1), ("전체 주문수", r)
    # 중형은 칸이 없다 -- 바이낸스 표와 **같은 뺄셈**이 닫혀야 한다.
    assert round(r[2] - r[8] - r[10], 6) == 43.9, ("중형 매수 물량", r)
    assert round(r[3] - r[9] - r[11], 6) == 0.0, ("중형 매도 물량", r)
    assert r[16] - r[14] - r[12] == 2, ("중형 매수 주문수", r)
    assert r[17] - r[15] - r[13] == 0, ("중형 매도 주문수", r)
    # 건수(개별 체결)와 주문수는 **다른 단위**다. 8 체결 / 3 주문.
    assert r[4] > r[16] and r[5] > r[17], ("체결 수는 주문 수보다 크거나 같다", r)

    # ── 닫힌 초는 되살아나지 않는다(상속받은 규칙이 그대로 사는지) ────────────
    late = OkxTapeBuffer(0.1)
    late.add_okx_order(5_000_000, 2500.0, 1.0, sell=False, fills=1)
    late.add_okx_order(5_002_000, 2500.0, 1.0, sell=False, fills=1)
    assert len(late.take_closed()) == 1, "5000 초가 나갔다"
    late.add_okx_order(5_000_000, 2500.0, 40.0, sell=False, fills=1)
    assert late.take_closed() == [], "닫힌 초가 되살아났다"
    assert late.dropped_late == 1, late.dropped_late

    # ── 경계 상수가 바이낸스 표와 **같은 객체**인가 ───────────────────────────
    assert RETAIL_MAX_USD == _bn.RETAIL_MAX_USD == 10_000.0
    assert WHALE_MIN_USD == _bn.WHALE_MIN_USD == 100_000.0
    assert OkxTapeBuffer.WIDTH == _bn.TapeBuffer.WIDTH == 16

    # ── REST 복구의 묶기: 연속된 (ts, side, px) 가 한 주문 ──────────────────
    f = lambda i, ts, side, px, sz: {"tradeId": str(i), "ts": str(ts), "side": side, "px": px, "sz": sz}
    g = group_fills([f(1, 100, "buy", "2500.0", "3"), f(2, 100, "buy", "2500.0", "2"),
                     f(3, 100, "buy", "2500.1", "1"), f(4, 100, "sell", "2500.1", "1"),
                     f(5, 101, "sell", "2500.1", "1"), f(6, 101, "sell", "2500.1", "4")])
    assert g == [(100, 2500.0, 5.0, False, 2), (100, 2500.1, 1.0, False, 1),
                 (100, 2500.1, 1.0, True, 1), (101, 2500.1, 5.0, True, 2)], g
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--inst", default=os.getenv("OKX_TAPE_INST", "ETH-USDT-SWAP").upper())
    ap.add_argument("--db", type=Path, default=Path(os.getenv("OKX_TAPE_DB_PATH", DEFAULT_DB)))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        asyncio.run(collect(args.inst, args.db))
    except KeyboardInterrupt:
        log("종료")


if __name__ == "__main__":
    main()
