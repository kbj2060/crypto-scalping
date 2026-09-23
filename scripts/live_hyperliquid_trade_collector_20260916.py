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
import gzip
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 🔴UTC 날짜별 파일. 단일 duckdb 로 두면 회수 경로가 없다 -- DELETE 는 파일을 안 줄이고
#   (2026-09-22 실측: 300만 행 중 절반 삭제 + CHECKPOINT 후에도 41.7MB 그대로),
#   수집기 노드에서 «차면 옮기고 지운다»가 성립하지 않는다. bookTicker/depthdiff 가 시각별로
#   쪼개 gzip 하는 것과 같은 규약을, 여기서는 **하루** 단위로 쓴다(~13MB/일이라 시각은 과하다).
#   orderflow/ 아래 두는 이유: 기존 벌크 전송 스크립트가 그 경로의 *.gz 를 그대로 집어간다.
DB_ROOT = Path(os.getenv("HL_ROOT", str(ROOT / "data/live/orderflow/hyperliquid")))


def db_path(ts: float | None = None) -> Path:
    """그 UTC 날짜의 파일. write() 가 flush 마다 새로 연결하므로 자정에 자연히 갈린다."""
    d = datetime.fromtimestamp(ts if ts is not None else time.time(), timezone.utc).strftime("%Y-%m-%d")
    return DB_ROOT / "_".join(COINS) / f"{d}.duckdb"


def _gzip_done(path: Path) -> None:
    """끝난 날짜 파일을 압축한다. 🔴원본은 **압축 성공 뒤에만** 지운다(depthdiff 와 같은 규약)."""
    gz = path.with_suffix(path.suffix + ".gz")
    if gz.exists() or not path.exists():
        return
    try:
        with open(path, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)
        if gz.stat().st_size > 0:
            before, after = path.stat().st_size, gz.stat().st_size
            path.unlink()
            print(f"  압축 {path.name} {before/1e6:.1f}MB -> {after/1e6:.1f}MB", flush=True)
    except Exception as exc:
        print(f"  압축 실패 {path.name}: {exc} -- 원본 유지", flush=True)
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


_last_db: Path | None = None


def write(rows: list[tuple], gap: tuple | None = None) -> None:
    global _last_db
    import duckdb
    DB = db_path()
    if _last_db is not None and _last_db != DB:
        _gzip_done(_last_db)      # 날짜가 넘어갔다 -- 어제 파일은 이제 안 바뀐다
    _last_db = DB
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(DB))
    try:
        con.execute("""CREATE TABLE IF NOT EXISTS hl_trades (
            coin VARCHAR, ts_ms BIGINT, tid BIGINT, hash VARCHAR, side VARCHAR,
            px DOUBLE, sz DOUBLE, buyer VARCHAR, seller VARCHAR)""")
        con.execute("CREATE TABLE IF NOT EXISTS gaps (coin VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)")
        con.begin()      # 🔴자동커밋이면 500행 = fsync 500번이다(체결 테이프 TapeStore.write 참고)
        if rows:
            con.executemany("INSERT INTO hl_trades VALUES (?,?,?,?,?,?,?,?,?)", rows)
        if gap:
            con.execute("INSERT INTO gaps VALUES (?,?,?,?)", list(gap))
        con.commit()
    finally:
        con.close()          # 붙들고 있으면 감시기·연구 쿼리가 BLOCKED 된다.


async def run() -> None:
    import websockets
    buf: list[tuple] = []
    last = time.time()
    down: tuple[int, str] | None = None   # (끊긴 시각, 이유) -- 다시 붙으면 그때 공백으로 적는다
    while True:
        try:
            async with websockets.connect(WS, ping_interval=20, ping_timeout=20) as ws:
                for coin in COINS:
                    await ws.send(json.dumps({"method": "subscribe",
                                              "subscription": {"type": "trades", "coin": coin}}))
                print(f"구독 {COINS} · {db_path()}", flush=True)
                # 🔴공백은 «끊긴 시각 → 다시 붙은 시각»이다. 예전엔 from 을 **연결 시작**에 찍어
                #   연결돼 있던 ~2.8시간 전체가 gaps 에 들어갔다(2026-09-23 실측: 기록 10,150초 vs
                #   실제 무체결 최대 348초).
                if down:
                    try:     # 기록 실패로 막 붙은 연결을 끊지 않는다
                        await asyncio.to_thread(write, [], gap=(",".join(COINS), down[0], int(time.time() * 1000), down[1]))
                    except Exception as e2:
                        print(f"  gap 기록 실패: {e2}", flush=True)
                    down = None
                while True:
                    buf += parse(json.loads(await ws.recv()))
                    if len(buf) >= FLUSH_N or (buf and time.time() - last >= FLUSH_SEC):
                        await asyncio.to_thread(write, buf); print(f"  +{len(buf)}행", flush=True); buf, last = [], time.time()
        except Exception as e:
            # 구멍은 **메운다고 되는 게 아니라 기록하는 것**이다 -- 체결 테이프 수집기와 같은 규약.
            print(f"WS 끊김: {type(e).__name__}: {e} — 5초 후 재연결", flush=True)
            down = down or (int(time.time() * 1000), type(e).__name__)
            try:
                await asyncio.to_thread(write, buf)
            except Exception as e2:
                print(f"  잔여 {len(buf)}행 기록 실패: {e2}", flush=True)
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
