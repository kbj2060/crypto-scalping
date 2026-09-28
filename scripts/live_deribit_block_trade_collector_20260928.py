#!/usr/bin/env python3
"""**Deribit ETH 옵션 체결·블록 거래 수집기** — 대시보드 «옵션» 카드의 «최근 블록 거래» 원천. (2026-09-28)

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LIVE = Path(os.getenv("DERIBIT_BT_DIR", str(ROOT / "data/live")))
DB = LIVE / "deribit_option_trades.duckdb"
STATE_PATH = LIVE / "deribit_block_trades_state.json"
WS = "wss://www.deribit.com/ws/api/v2"
REST = "https://www.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
CURRENCY = "ETH"
CHANNEL = f"trades.option.{CURRENCY}.100ms"
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
        b["premium_usd"] += r["amount"] * r["price"] * idx
        b["legs"].append({k: r[k] for k in ("instrument_name", "direction", "amount", "price",
                                            "iv", "mark_price", "index_price", "trade_id")})
    for b in blocks.values():
        b["notional_usd"] = round(b["notional_usd"], 2)
        b["premium_usd"] = round(b["premium_usd"], 2)
    return sorted(blocks.values(), key=lambda b: b["ts_ms"], reverse=True)


def _connect():
    import duckdb
    LIVE.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    con.execute("""CREATE TABLE IF NOT EXISTS option_trades (
        ts_ms BIGINT, trade_id VARCHAR PRIMARY KEY, trade_seq BIGINT, instrument_name VARCHAR,
        direction VARCHAR, amount DOUBLE, price DOUBLE, mark_price DOUBLE, index_price DOUBLE,
        iv DOUBLE, tick_direction INTEGER, liquidation VARCHAR, is_block BOOLEAN,
        block_trade_id VARCHAR, block_trade_leg_count INTEGER, block_rfq_id VARCHAR,
        combo_id VARCHAR, combo_trade_id VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS gaps (
        from_ms BIGINT, to_ms BIGINT, reason VARCHAR, backfilled INTEGER)""")
    return con


def last_ts() -> int | None:
    con = _connect()
    try:
        return con.execute("SELECT max(ts_ms) FROM option_trades").fetchone()[0]
    finally:
        con.close()


def write(rows: list[tuple], gap: tuple | None = None) -> int:
    """행을 넣고(중복 무시) 상태 JSON 을 새로 쓴다. 넣은 새 행 수를 돌려준다."""
    con = _connect()
    try:
        before = con.execute("SELECT count(*) FROM option_trades").fetchone()[0]
        con.begin()      # 🔴자동커밋이면 행마다 fsync(HL 수집기 참고)
        if rows:
            con.executemany(f"INSERT OR IGNORE INTO option_trades VALUES ({','.join('?' * len(COLS))})", rows)
        if gap:
            con.execute("INSERT INTO gaps VALUES (?,?,?,?)", list(gap))
        con.commit()
        added = con.execute("SELECT count(*) FROM option_trades").fetchone()[0] - before
        cur = con.execute(f"SELECT {','.join(COLS)} FROM option_trades WHERE is_block AND ts_ms >= ?",
                          [int(time.time() * 1000) - STATE_WINDOW_MS])
        blocks = group_blocks([dict(zip(COLS, r)) for r in cur.fetchall()])
    finally:
        con.close()      # 붙들고 있으면 연구 쿼리가 막힌다
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "currency": CURRENCY,
           "source": f"deribit {CHANNEL}", "window_hours": STATE_WINDOW_MS // 3_600_000,
           "n_blocks": len(blocks), "blocks": blocks}
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    tmp.replace(STATE_PATH)
    return added


def backfill(since_ms: int, until_ms: int) -> list[tuple]:
    """REST 로 [since, until] 옵션 체결을 오래된 것부터 끝까지 넘긴다. 겹침은 PK 가 버린다."""
    import requests
    out: list[tuple] = []
    start = since_ms
    while True:
        r = requests.get(REST, timeout=20, params={
            "currency": CURRENCY, "kind": "option", "start_timestamp": start, "end_timestamp": until_ms,
            "count": 1000, "sorting": "asc", "include_old": "true"})
        r.raise_for_status()
        res = r.json()["result"]
        trades = res.get("trades") or []
        out += [row(t) for t in trades]
        nxt = max((t["timestamp"] for t in trades), default=start)
        # ponytail: 같은 ms 에 1,000건 넘게 몰리면 그 ms 의 나머지를 놓친다 -- 옵션 체결 밀도로는 없음
        if not res.get("has_more") or nxt <= start:
            return out
        start = nxt


async def run() -> None:
    import websockets
    buf: list[tuple] = []
    last = time.time()
    down: tuple[int, str] | None = None
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, ping_timeout=20) as ws:
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "public/subscribe",
                                          "params": {"channels": [CHANNEL]}}))
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": 2, "method": "public/set_heartbeat",
                                          "params": {"interval": 30}}))
                # 구독 **뒤에** 백필한다 -- 그래야 백필 끝과 WS 시작 사이에 틈이 없다(겹침은 PK 가 버림).
                now = int(time.time() * 1000)
                since = max(await asyncio.to_thread(last_ts) or 0, now - BACKFILL_MAX_MS)
                bf = await asyncio.to_thread(backfill, since, now)
                gap = (down[0], now, down[1], None) if down else None
                added = await asyncio.to_thread(write, bf, gap and gap[:3] + (len(bf),))
                print(f"구독 {CHANNEL} · 백필 {len(bf)}건(새 {added}) since={since} · {DB}", flush=True)
                down = None
                while True:
                    m = json.loads(await asyncio.wait_for(ws.recv(), IDLE_SEC))
                    if "error" in m:
                        raise RuntimeError(f"deribit error {m['error']}")
                    p = m.get("params") or {}
                    if p.get("type") == "test_request":    # 하트비트 응답 안 하면 서버가 끊는다
                        await ws.send(json.dumps({"jsonrpc": "2.0", "id": 3, "method": "public/test", "params": {}}))
                    elif p.get("channel") == CHANNEL:
                        buf += [row(t) for t in p["data"]]
                    if time.time() - last >= FLUSH_SEC:
                        n = len(buf)
                        added = await asyncio.to_thread(write, buf)
                        print(f"  +{n}행(새 {added}, 블록 {sum(r[12] for r in buf)})", flush=True)
                        buf, last = [], time.time()
        except Exception as e:
            # 틈은 다음 연결의 REST 백필이 메우고, 끊긴 구간은 gaps 에 «메운 건수»와 함께 남긴다.
            print(f"WS 끊김: {type(e).__name__}: {e} — 5초 후 재연결", flush=True)
            down = down or (int(time.time() * 1000), type(e).__name__)
            try:
                await asyncio.to_thread(write, buf)
            except Exception as e2:
                print(f"  잔여 {len(buf)}행 기록 실패: {e2}", flush=True)
            buf = []
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
