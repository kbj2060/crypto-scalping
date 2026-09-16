#!/usr/bin/env python3
"""**하이퍼리퀴드 주소단위 체결 수집기** — 「공시된 의도」를 읽는 축의 원천. (2026-09-16)

왜 필요한가: 이 저장소의 방향 축은 전부 **가격·호가 이력에서 의도를 추론**하는 일이었고,
§5.36 H 가 그 축의 벽을 쟀다(비용 0 에서도 못 넘음 ⇒ 구속조건은 정보). 하이퍼리퀴드는 다르다 --
완전 온체인 LOB 이라 **체결마다 매수·매도 주소가 공개**되고, 프로토콜 네이티브 TWAP 은 시작
시점에 조건을 공시하고 활성 내내 보인다. 즉 추론이 아니라 **공시된 방향성 주문흐름**이다.
근거 논문: arXiv:2606.15715 (Barone & Lillo, 2026-06) -- hidden metaorder 430만 vs 가시 TWAP
46.5만 비교. 가시 TWAP 은 체결비용·영구충격이 낮고, 활성 동안 호가가 **흡수 측으로 기운다**.

🔴왜 지금인가(P0): 과거분은 살 수 없다. 노드 아카이브 `hl-mainnet-node-data` 가 **Requester-Pays**
라 익명 접근이 거부되고(실측 2026-09-16), 익스플로러 경로는 404, `rpc.hyperliquid.xyz/evm` 은
HyperEVM 이라 주문서(HyperCore)를 안 준다. `twapHistory`/`userTwapSliceFills` 는 **주소별**
엔드포인트라 시장 전체를 못 준다. **지금 안 받으면 영원히 없다** -- 호가 래스터 수집기와 같은 논리.

⭐그런데 공개 WS 체결 피드에 주소가 이미 들어 있다(실측: 61/61건 `users` 보유, 25초에 고유주소 45):
  {"coin":"ETH","side":"B","px":"2402.5","sz":"4.4514","time":..,"tid":..,
   "hash":"0x..","users":["0x매수자","0x매도자"]}
키 없음·무료·실시간. 논문의 핵심 방법(주소별 메타오더 재구성)이 그대로 가능하다.

🔴**집계하지 않는다.** 체결 테이프(`live_trade_tape_collector_20260916.py`)는 11.3M행/일이라
1초×$0.1 빈으로 줄였지만, 여기는 실측 ~2.4건/초 = **~210k행/일**(54배 작다). 그리고 이 축의
전부가 **주소**다 -- 집계하는 순간 메타오더 재구성이 불가능해진다. 원본 그대로 적는다.

봇과 완전 분리: 자기 WS · 자기 duckdb · 주문 없음 · 죽어도 봇에 영향 없다. 바이낸스 REST
weight 를 한 톨도 쓰지 않는다(다른 거래소다).

사용:
  python3 scripts/live_hyperliquid_trade_collector_20260916.py            # ETH
  HL_COINS=ETH,BTC python3 scripts/live_hyperliquid_trade_collector_20260916.py
  python3 scripts/live_hyperliquid_trade_collector_20260916.py --selftest  # 네트워크·DB 없이
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/live/hyperliquid_trades.duckdb"
WS = "wss://api.hyperliquid.xyz/ws"
COINS = [c.strip().upper() for c in os.getenv("HL_COINS", "ETH").split(",") if c.strip()]
FLUSH_N = 500          # 이만큼 모이면 쓴다. duckdb 는 프로세스 하나만 열 수 있어 짧게 잡고 닫는다.
FLUSH_SEC = 20.0


def parse(msg: dict) -> list[tuple]:
    """WS 메시지 → 행 목록. `users` 가 없으면 버린다(이 수집기의 존재 이유가 그 필드다)."""
    if msg.get("channel") != "trades":
        return []
    out = []
    for t in msg.get("data") or []:
        users = t.get("users") or []
        if len(users) != 2:
            continue
        out.append((str(t["coin"]), int(t["time"]), int(t["tid"]), str(t.get("hash", "")),
                    str(t["side"]), float(t["px"]), float(t["sz"]),
                    str(users[0]).lower(), str(users[1]).lower()))
    return out


def write(rows: list[tuple], gap: tuple | None = None) -> None:
    import duckdb
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS hl_trades (
            coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR,
            px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)""")
        con.execute("CREATE TABLE IF NOT EXISTS gaps (coin VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)")
        if rows:
            con.executemany("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
        if gap:
            con.execute("INSERT INTO gaps VALUES (?,?,?,?)", list(gap))
    finally:
        con.close()          # 붙들고 있으면 감시기·연구 쿼리가 BLOCKED 된다.


async def run() -> None:
    import websockets
    buf: list[tuple] = []
    last = time.time()
    while True:
        down_from = int(time.time() * 1000)
        try:
            async with websockets.connect(WS, ping_interval=20, ping_timeout=20) as ws:
                for coin in COINS:
                    await ws.send(json.dumps({"method": "subscribe",
                                              "subscription": {"type": "trades", "coin": coin}}))
                print(f"구독 {COINS} · {DB}", flush=True)
                while True:
                    buf += parse(json.loads(await ws.recv()))
                    if len(buf) >= FLUSH_N or (buf and time.time() - last >= FLUSH_SEC):
                        write(buf); print(f"  +{len(buf)}행", flush=True); buf, last = [], time.time()
        except Exception as e:
            # 구멍은 **메운다고 되는 게 아니라 기록하는 것**이다 -- 체결 테이프 수집기와 같은 규약.
            print(f"WS 끊김: {type(e).__name__}: {e} — 5초 후 재연결", flush=True)
            try:
                write(buf, gap=(",".join(COINS), down_from, int(time.time() * 1000), type(e).__name__))
            except Exception as e2:
                print(f"  gap 기록 실패(보류): {e2}", flush=True)
            buf = []
            await asyncio.sleep(5)


def _selftest() -> None:
    m = {"channel": "trades", "data": [
        {"coin": "ETH", "side": "B", "px": "2402.5", "sz": "4.45", "time": 1789565174423,
         "tid": 137993694461296, "hash": "0xabc", "users": ["0xAA", "0xBB"]},
        {"coin": "ETH", "side": "A", "px": "1", "sz": "1", "time": 1, "tid": 2, "users": ["0xAA"]},
    ]}
    r = parse(m)
    assert len(r) == 1, "users 가 2개가 아닌 체결은 버려야 한다"
    assert r[0][0] == "ETH" and r[0][2] == 137993694461296
    assert r[0][7] == "0xaa" and r[0][8] == "0xbb", "주소는 소문자로 정규화한다"
    assert parse({"channel": "l2Book", "data": {}}) == []
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
