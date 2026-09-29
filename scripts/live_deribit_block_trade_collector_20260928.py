#!/usr/bin/env python3
"""**Deribit 옵션 통합 수집기** — 대시보드 «옵션» 카드의 원천 전부를 **duckdb 한 파일**에. (2026-09-28)

2026-09-28 사용자 «옵션 데이터들 모아서 하나의 duckdb 에»: 상주 프로세스 하나가 `deribit_options.duckdb` 에
  ① ETH 옵션 체결 전체 + 블록 거래(WS 실시간, 아래)            -> option_trades · gaps
  ② ETH·BTC 옵션 체인 스냅샷(10분, collect_deribit_option_gex) -> option_chain_snapshot · gex_summary
  ③ 옵션 요약(10분: DVOL·실현7일·만기별 ATM IV/RR/BF/미결제/P·C/max pain·감마 곡선/플립·보험용 OTM 가격) -> option_summary
를 쓴다. 옛 두 파일(deribit_gex.duckdb = 매시 cron · deribit_option_trades.duckdb)의 이력은 첫 기동 때 한 번 옮긴다.
상태 JSON 둘(deribit_gex_state.json · deribit_block_trades_state.json)은 그대로 -- 대시보드 계약은 안 바뀐다.
🔴매시 GEX cron 은 **끈다**(같은 상태 파일을 두 곳이 쓰면 안 된다).

원천(실측 2026-09-28, 키 없음):
  WS  `wss://www.deribit.com/ws/api/v2` · `public/subscribe` · 채널 `trades.option.ETH.100ms`
      🔴`trades.option.ETH.raw` 는 비인증 구독이 거부된다(code 13778
      `raw_subscriptions_not_available_for_unauthorized`). 100ms 는 묶음 전송일 뿐 체결은 전부 온다.
  REST `public/get_last_trades_by_currency_and_time`(kind=option, sorting=asc) -- 재기동 틈 백필.
체결 필드(실측): timestamp, trade_id('ETH-311974188' 문자열), trade_seq, instrument_name, direction
  (테이커 방향), amount=contracts(ETH), price(ETH 프리미엄), mark_price, index_price, iv, tick_direction,
  그리고 **블록일 때만** block_trade_id('BLOCK-289556'), block_trade_leg_count, block_rfq_id, combo_id,
  combo_trade_id. 실측 REST 1,000건(3.3시간) 중 블록 7건 -- 드물다.
  🔴block_trade_leg_count 는 **선물 헤지 다리까지 센** 다리 수일 수 있다. 이 채널은 옵션만 주므로
  상태 JSON 에 legs_seen(받은 옵션 다리 수)을 따로 낸다.

옵션 체결 전체를 원본 그대로 적는다(~7천 건/일이라 집계할 이유가 없다). 중복은 trade_id PK 로 버린다
-- WS 와 백필이 겹쳐도 안전하다. 봇과 완전 분리 · 주문 없음 · 🔴바이낸스를 한 톨도 안 부른다.

대시보드는 duckdb 를 열지 않는다(단일 writer 락) -- flush 마다 `deribit_block_trades_state.json` 을
tmp→os.replace 로 떨군다(GEX_STATE_PATH 와 같은 관례). 전략 라벨은 넣지 않는다(원자료만).

사용:
  python3 scripts/live_deribit_block_trade_collector_20260928.py
  python3 scripts/live_deribit_block_trade_collector_20260928.py --selftest   # 네트워크·DB 없이
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from datetime import datetime, timezone
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # collect_deribit_option_gex_20260815 (같은 폴더)

ROOT = Path(__file__).resolve().parents[1]
LIVE = Path(os.getenv("DERIBIT_BT_DIR", str(ROOT / "data/live")))
DB = LIVE / "deribit_options.duckdb"
OLD_DBS = (LIVE / "deribit_gex.duckdb", LIVE / "deribit_option_trades.duckdb")   # 첫 기동 때 이력을 옮길 옛 파일
CHAIN_SEC = 600.0                   # 체인·요약 주기(옛 매시 cron 을 10분으로)
STATE_PATH = LIVE / "deribit_block_trades_state.json"
WS = "wss://www.deribit.com/ws/api/v2"
REST = "https://www.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
# 2026-09-28 네 코인(사용자 지시 «sol, eth, btc, xrp»). SOL·XRP 는 Deribit 에서 USDC 결제 선형 옵션이라 `USDC` 채널 하나에
#   HYPE·AVAX·TRX 까지 섞여 온다 -- 접두어로 네 코인만 남긴다. 선형은 가격이 USDC 라 프리미엄에 지수를 곱하지 않는다.
API_CURRENCIES = ("ETH", "BTC", "USDC")
CHANNELS = tuple(f"trades.option.{c}.100ms" for c in API_CURRENCIES)
COIN_PREFIX = (("ETH-", "ETH"), ("BTC-", "BTC"), ("SOL_USDC-", "SOL"), ("XRP_USDC-", "XRP"))
COINS = tuple(c for _, c in COIN_PREFIX)


def coin_of(inst: str) -> str | None:
    return next((c for pre, c in COIN_PREFIX if inst.startswith(pre)), None)
STATE_WINDOW_MS = 24 * 3600 * 1000
BACKFILL_MAX_MS = STATE_WINDOW_MS   # 첫 기동이면 24h 를 채워 상태 JSON 이 바로 찬다
FLUSH_SEC = 20.0
IDLE_SEC = 90.0                     # 하트비트 30초라 90초 무음 = 죽은 연결

COLS = ("ts_ms", "trade_id", "trade_seq", "instrument_name", "direction", "amount", "price",
        "mark_price", "index_price", "iv", "tick_direction", "liquidation", "is_block",
        "block_trade_id", "block_trade_leg_count", "block_rfq_id", "combo_id", "combo_trade_id")


def row(t: dict) -> tuple:
    b = t.get("block_trade_id")
    return (int(t["timestamp"]), str(t["trade_id"]), t.get("trade_seq"), t["instrument_name"],
            t["direction"], float(t["amount"]), float(t["price"]), t.get("mark_price"),
            t.get("index_price"), t.get("iv"), t.get("tick_direction"), t.get("liquidation"),
            b is not None, b, t.get("block_trade_leg_count"),
            None if t.get("block_rfq_id") is None else str(t["block_rfq_id"]),
            t.get("combo_id"), t.get("combo_trade_id"))


def group_blocks(rows: list[dict]) -> list[dict]:
    """블록 다리 행 → 블록 단위 목록(최신 먼저). 합계는 USD: 명목 = Σ amount×index,
    프리미엄 = Σ amount×price×index (**총액** -- 매수·매도 다리 부호를 상쇄하지 않는다; 순액은 legs 의 direction 으로)."""
    blocks: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (r["ts_ms"], r["trade_id"])):
        b = blocks.setdefault(r["block_trade_id"], {
            "block_trade_id": r["block_trade_id"], "ts_ms": r["ts_ms"],
            "leg_count": r["block_trade_leg_count"], "legs_seen": 0,
            "block_rfq_id": r["block_rfq_id"], "combo_id": r["combo_id"],
            "notional_usd": 0.0, "premium_usd": 0.0, "legs": []})
        idx = r["index_price"] or 0.0
        b["legs_seen"] += 1
        b["notional_usd"] += r["amount"] * idx
        b["premium_usd"] += r["amount"] * r["price"] * (1.0 if "_USDC-" in r["instrument_name"] else idx)
        b["legs"].append({k: r[k] for k in ("instrument_name", "direction", "amount", "price",
                                            "iv", "mark_price", "index_price", "trade_id")})
    for b in blocks.values():
        b["notional_usd"] = round(b["notional_usd"], 2)
        b["premium_usd"] = round(b["premium_usd"], 2)
    return sorted(blocks.values(), key=lambda b: b["ts_ms"], reverse=True)


def _connect():
    """🔴다른 프로세스(연구 쿼리·일회성 백필)가 파일을 잡고 있으면 duckdb 는 즉시 IOException 이다(단일 writer).
    2026-09-28 서버 실측: 그 예외가 WS 루프까지 올라가 연결을 끊고 버퍼를 버렸다 -- 여기서 최대 30초 기다린다."""
    import duckdb
    LIVE.mkdir(parents=True, exist_ok=True)
    for i in range(15):
        try:
            con = duckdb.connect(str(DB)); break
        except duckdb.IOException as e:
            if "lock" not in str(e).lower() or i == 14:
                raise
            time.sleep(2)
    con.execute("""CREATE TABLE IF NOT EXISTS option_trades (
        ts_ms BIGINT, trade_id VARCHAR PRIMARY KEY, trade_seq BIGINT, instrument_name VARCHAR,
        direction VARCHAR, amount DOUBLE, price DOUBLE, mark_price DOUBLE, index_price DOUBLE,
        iv DOUBLE, tick_direction INTEGER, liquidation VARCHAR, is_block BOOLEAN,
        block_trade_id VARCHAR, block_trade_leg_count INTEGER, block_rfq_id VARCHAR,
        combo_id VARCHAR, combo_trade_id VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS gaps (
        from_ms BIGINT, to_ms BIGINT, reason VARCHAR, backfilled INTEGER)""")
    return con


def migrate_once() -> None:
    """옛 두 파일의 이력을 새 파일로 한 번 옮긴다(대상 테이블이 비어 있을 때만 -- 다시 켜도 중복 없음).
    🔴옛 파일은 지우지 않는다(되돌리기용). 매시 cron 이 켜져 있으면 옛 gex 파일이 잠겨 있을 수 있다 -- 그때는 건너뛴다."""
    import collect_deribit_option_gex_20260815 as gex
    con = _connect()
    try:
        gex.ensure_tables(con)
        for old in OLD_DBS:
            if not old.exists():
                continue
            try:
                con.execute(f"ATTACH '{old}' AS old (READ_ONLY)")
            except Exception as e:
                print(f"옮기기 건너뜀 {old.name}: {e}", flush=True); continue
            try:
                have = {r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_catalog = 'old'").fetchall()}
                for t in ("option_trades", "gaps", "option_chain_snapshot", "gex_summary", "option_summary"):
                    if t in have and con.execute(f"SELECT count(*) FROM main.{t}").fetchone()[0] == 0:
                        # 🔴한 테이블이 실패(스키마 어긋남·손상)해도 수집기는 떠야 한다 -- 여기서 새면 run() 밖이라 크래시 루프가 된다.
                        try:
                            n = con.execute(f"SELECT count(*) FROM old.{t}").fetchone()[0]
                            con.execute(f"INSERT INTO main.{t} SELECT * FROM old.{t}")
                            print(f"옮김 {old.name}.{t} {n}행", flush=True)
                        except Exception as e:
                            print(f"옮기기 실패 {old.name}.{t} -- 건너뜀: {type(e).__name__}: {str(e)[:160]}", flush=True)
            finally:
                con.execute("DETACH old")
    finally:
        con.close()


def chain_poll() -> None:
    """체인 스냅샷 + GEX + 옵션 요약 + deribit_gex_state.json -- collect_deribit_option_gex 의 poll_once 를 이 파일에."""
    import collect_deribit_option_gex_20260815 as gex
    gex.STATE_PATH = LIVE / "deribit_gex_state.json"     # 상태 파일도 이 수집기의 폴더를 따른다(시험 폴더 포함)
    con = _connect()
    try:
        gex.ensure_tables(con)
        gex.poll_once(con)
    finally:
        con.close()


async def chain_loop() -> None:
    while True:
        t0 = time.time()
        try:
            await asyncio.to_thread(chain_poll)
        except Exception as e:           # 체인 조회가 실패해도 체결 수집은 계속 -- 다음 주기에 다시
            print(f"체인 폴링 실패: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(max(30.0, CHAIN_SEC - (time.time() - t0)))


def hourly_flow(rows) -> dict:
    """2026-09-28 옵션 순매수 흐름(사용자 선택 2-A) -- 코인별 정시(UTC=KST 정시) 버킷 25개(지난 24시간 + 진행 중).
    cb/cs/pb/ps = 콜·풋 테이커 매수·매도 수량(기초자산 단위), dlt = 옵션으로 산 순델타(콜 매수·풋 매도 +).
    델타는 체결 시점의 IV·지수로 블랙-숄즈(r=0, 선도 대신 지수 -- ponytail: 먼 만기는 캐리만큼 어긋난다, 흐름 방향엔 영향 작음).
    🔴방향은 Deribit 공개 체결의 «테이커 방향»이다(문서 확인 09-28). 개시/청산은 모른다."""
    import collect_deribit_option_gex_20260815 as gex
    now_h = int(time.time() // 3600) * 3_600_000
    hours = [now_h - i * 3_600_000 for i in range(24, -1, -1)]
    out = {c: {h: {"h": h, "cb": 0.0, "cs": 0.0, "pb": 0.0, "ps": 0.0, "dlt": 0.0} for h in hours} for c in COINS}
    for ts, inst, direction, amt, iv, ix in rows:
        c = coin_of(inst)
        spec = gex._parse_instrument(inst) if c else None
        b = out[c].get(int(ts // 3_600_000) * 3_600_000) if spec else None
        if b is None:
            continue
        call = spec["option_type"] == "call"
        buy = direction == "buy"
        b[("cb" if buy else "cs") if call else ("pb" if buy else "ps")] += float(amt)
        yrs = (spec["expiration_ts"].timestamp() - ts / 1000) / (365.0 * 86400)
        d = gex._bs_delta(float(ix or 0), spec["strike"], float(iv or 0), yrs, call)
        b["dlt"] += (1 if buy else -1) * float(amt) * d
    return {c: [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in b.items()} for b in hs.values()] for c, hs in out.items()}


def last_ts() -> dict:
    """목록(ETH·BTC·USDC)별 마지막 저장 시각. 🔴전체 max 하나로 잡으면 새로 붙은 목록(2026-09-28 BTC·USDC)이
    ETH 의 최신 시각부터만 채워져 24시간 이력이 통째로 빠진다(실제로 블록 0건이 났다)."""
    con = _connect()
    try:
        return {api: con.execute("SELECT max(ts_ms) FROM option_trades WHERE instrument_name LIKE ?",
                                 ["%_USDC-%" if api == "USDC" else f"{api}-%"]).fetchone()[0]
                for api in API_CURRENCIES}
    finally:
        con.close()


def write(rows: list[tuple], gaps: list[tuple] | None = None) -> int:
    """행을 넣고(중복 무시) 상태 JSON 을 새로 쓴다. 넣은 새 행 수를 돌려준다."""
    con = _connect()
    try:
        before = con.execute("SELECT count(*) FROM option_trades").fetchone()[0]
        con.begin()      # 🔴자동커밋이면 행마다 fsync(HL 수집기 참고)
        if rows:
            con.executemany(f"INSERT OR IGNORE INTO option_trades VALUES ({','.join('?' * len(COLS))})", rows)
        for g in gaps or []:
            con.execute("INSERT INTO gaps VALUES (?,?,?,?)", list(g))
        con.commit()
        added = con.execute("SELECT count(*) FROM option_trades").fetchone()[0] - before
        cur = con.execute(f"SELECT {','.join(COLS)} FROM option_trades WHERE is_block AND ts_ms >= ?",
                          [int(time.time() * 1000) - STATE_WINDOW_MS])
        legs = [dict(zip(COLS, r)) for r in cur.fetchall()]
        by_coin = {c: group_blocks([x for x in legs if coin_of(x["instrument_name"]) == c]) for c in COINS}
        flow = hourly_flow(con.execute("SELECT ts_ms, instrument_name, direction, amount, iv, index_price FROM option_trades "
                                       "WHERE ts_ms >= ?", [(int(time.time() // 3600) - 24) * 3_600_000]).fetchall())
    finally:
        con.close()      # 붙들고 있으면 연구 쿼리가 막힌다
    # blocks/n_blocks = ETH(옛 계약 그대로) · blocks_by_coin = 네 코인
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "coins": list(COINS),
           "source": "deribit " + " · ".join(CHANNELS), "window_hours": STATE_WINDOW_MS // 3_600_000,
           "n_blocks": len(by_coin["ETH"]), "blocks": by_coin["ETH"],
           "blocks_by_coin": by_coin, "n_blocks_by_coin": {c: len(v) for c, v in by_coin.items()},
           "flow_by_coin": flow}
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)
    return added


def backfill(since_by_api: dict, until_ms: int) -> list[tuple]:
    """REST 로 [since, until] 옵션 체결을 오래된 것부터 끝까지 넘긴다. 겹침은 PK 가 버린다."""
    import requests
    out: list[tuple] = []
    for api in API_CURRENCIES:
        start = since_by_api[api]
        while True:
            r = requests.get(REST, timeout=20, params={
                "currency": api, "kind": "option", "start_timestamp": start, "end_timestamp": until_ms,
                "count": 1000, "sorting": "asc", "include_old": "true"})
            r.raise_for_status()
            res = r.json()["result"]
            trades = res.get("trades") or []
            out += [row(t) for t in trades if coin_of(t["instrument_name"])]
            nxt = max((t["timestamp"] for t in trades), default=start)
            # ponytail: 같은 ms 에 1,000건 넘게 몰리면 그 ms 의 나머지를 놓친다 -- 옵션 체결 밀도로는 없음
            if not res.get("has_more") or nxt <= start:
                break
            start = nxt
    return out


async def run() -> None:
    import websockets
    try:
        await asyncio.to_thread(migrate_once)
    except Exception as e:            # 이력 옮기기는 부가 작업 -- 실패해도 수집은 시작한다
        print(f"이력 옮기기 실패 -- 건너뜀: {type(e).__name__}: {e}", flush=True)
    asyncio.get_running_loop().create_task(chain_loop())   # 체인·요약 10분 -- WS 재연결과 무관하게 돈다
    buf: list[tuple] = []
    last = time.time()
    down: tuple[int, str] | None = None
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
                                          "params": {"channels": list(CHANNELS)}}))
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "public/set_heartbeat",
                                          "params": {"interval": 30}}))
                # 구독 **뒤에** 백필한다 -- 그래야 백필 끝과 WS 시작 사이에 틈이 없다(겹침은 PK 가 버림).
                now = int(time.time() * 1000)
                seen = await asyncio.to_thread(last_ts)      # (last 는 아래 flush 타이머 이름이다)
                since = {api: max(seen.get(api) or 0, now - BACKFILL_MAX_MS) for api in API_CURRENCIES}
                bf = await asyncio.to_thread(backfill, since, now)
                gap = (down[0], now, down[1], None) if down else None
                # 2026-09-30 검증: 24h 넘게 꺼져 있었으면 백필(최대 24h)이 못 메운 구간이 남는다 -- «복구 불가»로 적어 두면
                #   체결 기반 딜러 값(_taker_flow)이 그 뒤에 상장된 종목만 «처음부터 본 종목»으로 센다.
                lost = [(seen[api], since[api], f"unrecoverable {api}", 0) for api in API_CURRENCIES
                        if seen.get(api) and seen[api] < since[api]]
                added = await asyncio.to_thread(write, bf, ([gap[:3] + (len(bf),)] if gap else []) + lost)
                print(f"구독 {'·'.join(CHANNELS)} · 백필 {len(bf)}건(새 {added}) since={since} · {DB}", flush=True)
                down = None
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), IDLE_SEC))
                    if "error" in m:
                        raise RuntimeError(f"deribit error {m['error']}")
                    p = m.get("params") or {}
                    if p.get("type") == "test_request":    # 하트비트 응답 안 하면 서버가 끊는다
                        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "public/test", "params": {}}))
                    elif p.get("channel") in CHANNELS:
                        buf += [row(t) for t in p["data"] if coin_of(t["instrument_name"])]
                    if time.time() - last >= FLUSH_SEC:
                        n = len(buf)
                        try:
                            added = await asyncio.to_thread(write, buf)
                        except Exception as e:     # 쓰기 실패는 WS 를 끊을 이유가 아니다 -- 버퍼를 들고 다음 주기에 다시
                            print(f"  기록 보류 {n}행: {type(e).__name__}: {str(e)[:120]}", flush=True)
                            last = time.time()
                            continue
                        print(f"  +{n}행(새 {added}, 블록 {sum(r[12] for r in buf)})", flush=True)
                        buf, last = [], time.time()
        except Exception as e:
            # 틈은 다음 연결의 REST 백필이 메우고, 끊긴 구간은 gaps 에 «메운 건수»와 함께 남긴다.
            print(f"WS 끊김: {type(e).__name__}: {e} — 5초 후 재연결", flush=True)
            down = down or (int(time.time() * 1000), type(e).__name__)
            try:
                await asyncio.to_thread(write, buf)
                buf = []
            except Exception as e2:   # 버리지 않고 다음 연결에서 다시 쓴다(백필도 메우지만 24h 를 넘기면 영구 유실이라)
                print(f"  잔여 {len(buf)}행 기록 보류: {e2}", flush=True)
                buf = buf[-200_000:]
            await asyncio.sleep(5)


def _selftest() -> None:
    base = {"iv": 46.0, "mark_price": 0.02, "tick_direction": 0, "trade_seq": 1}
    t = [dict(base, timestamp=2000, trade_id="ETH-2", instrument_name="ETH-9OCT26-2550-P", direction="sell",
              amount=125.0, price=0.0163, index_price=2000.0, block_trade_id="BLOCK-1",
              block_trade_leg_count=2, block_rfq_id=56452, combo_id="C1"),
         dict(base, timestamp=2000, trade_id="ETH-1", instrument_name="ETH-16OCT26-2550-P", direction="buy",
              amount=125.0, price=0.0245, index_price=2000.0, block_trade_id="BLOCK-1",
              block_trade_leg_count=2, block_rfq_id=56452, combo_id="C1"),
         dict(base, timestamp=3000, trade_id="ETH-3", instrument_name="ETH-30SEP26-2600-C", direction="buy",
              amount=10.0, price=0.01, index_price=2000.0, block_trade_id="BLOCK-2", block_trade_leg_count=2),
         dict(base, timestamp=4000, trade_id="ETH-4", instrument_name="ETH-30SEP26-2600-C", direction="buy",
              amount=1.0, price=0.01, index_price=2000.0)]
    rows = [dict(zip(COLS, row(x))) for x in t]
    assert [r["is_block"] for r in rows] == [True, True, True, False]
    assert rows[0]["block_rfq_id"] == "56452", "rfq id 는 문자열로 정규화"
    g = group_blocks([r for r in rows if r["is_block"]])
    assert [b["block_trade_id"] for b in g] == ["BLOCK-2", "BLOCK-1"], "최신 먼저"
    b1 = g[1]
    assert b1["legs_seen"] == 2 and b1["leg_count"] == 2
    assert [l["trade_id"] for l in b1["legs"]] == ["ETH-1", "ETH-2"], "같은 ms 면 trade_id 순"
    assert b1["notional_usd"] == 500000.0                       # 250 ETH × 2000
    assert b1["premium_usd"] == round(125 * 0.0163 * 2000 + 125 * 0.0245 * 2000, 2)
    assert g[0]["legs_seen"] == 1 and g[0]["leg_count"] == 2, "선물 다리 등 못 받은 다리는 legs_seen 으로 드러난다"
    lin = group_blocks([dict(zip(COLS, row(dict(base, timestamp=5000, trade_id="U-1", instrument_name="XRP_USDC-2OCT26-1d55-C",
                                                  direction="buy", amount=1000.0, price=0.02, index_price=1.5,
                                                  block_trade_id="BLOCK-9", block_trade_leg_count=1))))])
    assert lin[0]["premium_usd"] == 20.0, "선형(USDC) 옵션은 가격이 이미 USDC -- 지수를 곱하지 않는다"
    assert coin_of("SOL_USDC-2OCT26-120-P") == "SOL" and coin_of("HYPE_USDC-2OCT26-40-P") is None
    # 흐름: 콜 매수 1 · 풋 매수 2(델타 음수 → 순델타는 콜 +, 풋 매수 −) · 버킷 밖/모르는 종목은 버린다
    tnow = int(time.time() * 1000)
    fl = hourly_flow([(tnow, "ETH-30DEC26-2000-C", "buy", 1.0, 50.0, 2000.0), (tnow, "ETH-30DEC26-2000-P", "buy", 2.0, 50.0, 2000.0),
                      (tnow - 90 * 3_600_000, "ETH-30DEC26-2000-C", "buy", 5.0, 50.0, 2000.0), (tnow, "HYPE_USDC-30DEC26-40-P", "buy", 9.0, 50.0, 40.0)])
    last = fl["ETH"][-1]
    assert len(fl["ETH"]) == 25 and last["cb"] == 1.0 and last["pb"] == 2.0 and sum(b["cb"] for b in fl["ETH"]) == 1.0
    assert 0.5 < last["dlt"] + 2 * 0.45 < 1.5 and last["dlt"] < 0.5, last   # ATM 콜 +0.5 남짓, 풋 2개 −0.9 남짓
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); raise SystemExit(0)
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        pass
