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
import math
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


def _shape(o: list[dict]) -> str:
    """옵션 다리(같은 종목 상계 후, 만기·유형·행사가 순) → 구조 이름. 2026-09-30 연구에서 Deribit 자체 구조 코드와 313/313 일치
    (research_eth_option_trade_identity_20260930.classify 와 같은 규칙)."""
    n, exps, kinds = len(o), {x["exp"] for x in o}, {x["kind"] for x in o}
    s = [1 if x["q"] > 0 else -1 for x in o]
    q = [abs(x["q"]) for x in o]
    eq = lambda a, b: abs(a - b) <= 1e-9 * max(a, b)   # noqa: E731
    if n == 1:
        return "single"
    if n == 2 and len(exps) == 1:
        if kinds == {"C", "P"}:
            c, p = (o[0], o[1]) if o[0]["kind"] == "C" else (o[1], o[0])
            if (c["q"] > 0) == (p["q"] > 0):
                return "straddle" if c["strike"] == p["strike"] else "strangle"
            return "synthetic" if c["strike"] == p["strike"] else "risk_reversal"
        return "other" if s[0] == s[1] else ("vertical" if eq(q[0], q[1]) else "ratio_spread")
    if n == 2:
        if len(kinds) == 1 and s[0] != s[1] and eq(q[0], q[1]):
            return "calendar" if o[0]["strike"] == o[1]["strike"] else "diagonal"
        return "other"
    if len(exps) != 1:
        return "other"
    if n == 3 and len(kinds) == 1:
        if s[0] == s[2] != s[1] and eq(q[0], q[2]) and eq(q[1], q[0] + q[2]):
            return "butterfly"
        if eq(q[0], q[1]) and eq(q[1], q[2]) and (s[0] != s[1] == s[2] or s[0] == s[1] != s[2]):
            return "ladder"
        return "other"
    if n == 4 and all(eq(x, q[0]) for x in q):
        if len(kinds) == 1:
            return "condor" if s[0] == s[3] != s[1] == s[2] else "other"
        puts, calls = [x for x in o if x["kind"] == "P"], [x for x in o if x["kind"] == "C"]
        if len(puts) != 2:
            return "other"
        if len({x["strike"] for x in o}) == 2 and puts[0]["strike"] == calls[0]["strike"]:
            lo_c = calls[0]["q"] > 0
            return "box" if lo_c != (puts[0]["q"] > 0) and (calls[1]["q"] > 0) != lo_c else "other"
        if (puts[0]["q"] > 0) != (puts[1]["q"] > 0) and (calls[0]["q"] > 0) != (calls[1]["q"] > 0) \
                and (puts[1]["q"] > 0) == (calls[0]["q"] > 0):
            if puts[1]["strike"] == calls[0]["strike"]:
                return "iron_butterfly"
            return "iron_condor" if puts[1]["strike"] < calls[0]["strike"] else "other"
    return "other"


def block_shape(legs: list[dict], ts_ms: int, fut: list[dict] | None = None) -> dict:
    """블록 한 건(테이커 관점)의 구조 · 선물 헤지 다리 유무 · 순델타 쪽 · 순베가 쪽(2026-10-01 사용자 «새로 보여 줄 것»).
    그릭스는 체결 IV·지수로 BS(r=0) -- 부호만 쓴다(크기로 «방향 vs 변동성»을 가르는 건 가정 의존이라 안 한다).
    쪽 = |순| < 총의 10% 면 neutral(선물 헤지가 붙은 패키지는 순델타 비율 중앙 0.04)."""
    import collect_deribit_option_gex_20260815 as gex
    net: dict = {}
    for l in legs:
        net[l["instrument_name"]] = net.get(l["instrument_name"], 0.0) + (1 if l["direction"] == "buy" else -1) * float(l["amount"])
    o = []
    for inst, qq in net.items():
        sp = gex._parse_instrument(inst)
        if sp and abs(qq) > 1e-12:
            o.append({"exp": sp["expiration_ts"], "kind": "C" if sp["option_type"] == "call" else "P", "strike": sp["strike"], "q": qq})
    o.sort(key=lambda x: (x["exp"], x["kind"], x["strike"]))
    nd = gd = nv = gv = 0.0
    for l in legs:
        sp = gex._parse_instrument(l["instrument_name"])
        S, iv = float(l.get("index_price") or 0), float(l.get("iv") or 0)
        yrs = (sp["expiration_ts"].timestamp() - ts_ms / 1000) / (365.0 * 86400) if sp else 0
        if not sp or S <= 0 or iv <= 0 or yrs <= 0:
            continue
        sig = iv / 100
        d1 = (math.log(S / sp["strike"]) + 0.5 * sig * sig * yrs) / (sig * math.sqrt(yrs))
        de = gex._bs_delta(S, sp["strike"], iv, yrs, sp["option_type"] == "call")
        ve = S * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(yrs) / 100
        qq = (1 if l["direction"] == "buy" else -1) * float(l["amount"])
        nd, gd, nv, gv = nd + qq * de, gd + abs(qq * de), nv + qq * ve, gv + abs(qq * ve)
    for f in fut or []:     # 선물 amount: 역선물(ETH·BTC) = USD 명목 → 코인 · USDC 선형 = 코인 수량
        qq = (1 if f["direction"] == "buy" else -1) * float(f["amount"])
        d = qq if "_USDC" in f["instrument_name"] else qq / float(f["price"])
        nd, gd = nd + d, gd + abs(d)
    side = lambda v, g, pos, neg: None if g <= 0 else ("neutral" if abs(v) < 0.1 * g else (pos if v > 0 else neg))   # noqa: E731
    return {"structure": _shape(o) if o else "future_only", "hedge": bool(fut),
            "net_delta": round(nd, 4), "net_vega_usd": round(nv, 2),        # 2026-10-01 블록 24시간 합(요청자 쪽) 용
            "delta_side": side(nd, gd, "long", "short"), "vol_side": side(nv, gv, "long_vol", "short_vol")}


def group_blocks(rows: list[dict], fut: dict | None = None) -> list[dict]:
    """블록 다리 행 → 블록 단위 목록(최신 먼저). 합계는 USD: 명목 = Σ amount×index,
    프리미엄 = Σ amount×price×index (**총액** -- 매수·매도 다리 부호를 상쇄하지 않는다; 순액은 legs 의 direction 으로).
    fut = {block_trade_id: [선물 헤지 다리]}(reconcile 이 채운 block_future_legs) → 블록마다 block_shape."""
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
        try:
            b.update(block_shape(b["legs"], b["ts_ms"], (fut or {}).get(b["block_trade_id"])))
        except Exception:    # 이름 규칙 밖 종목 -- 구조만 빠진다
            pass
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
    # 2026-10-01 연구(research_eth_option_trade_identity_20260930) 뒤 추가: 블록 RFQ 패키지(다리 비율·패키지가·헤지)와
    #   블록의 선물 헤지 다리(옵션 채널만 구독해 못 받던 것). 둘 다 reconcile() 이 매시 채운다.
    con.execute("""CREATE TABLE IF NOT EXISTS block_rfq_trades (
        rfq_id VARCHAR PRIMARY KEY, ts_ms BIGINT, api VARCHAR, direction VARCHAR, amount DOUBLE, price DOUBLE,
        mark_price DOUBLE, combo_id VARCHAR, legs VARCHAR, hedge VARCHAR, index_prices VARCHAR)""")
    con.execute("""CREATE TABLE IF NOT EXISTS block_future_legs (
        trade_id VARCHAR PRIMARY KEY, ts_ms BIGINT, instrument_name VARCHAR, direction VARCHAR, amount DOUBLE,
        price DOUBLE, index_price DOUBLE, block_trade_id VARCHAR)""")
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


# 2026-09-30 검증: 체결 버퍼는 20초마다만 DB 로 갔다 -- 10분 체인 폴링이 직전 ≤20초 체결을 못 봐 딜러 값이 한 주기 늦었다
#   (BTC 3건이 감마 18%). 버퍼를 모듈에 두고, 체인 폴링 직전에도 비운다. 두 곳이 동시에 비우지 않게 잠금 하나로 묶는다 --
#   붙이기는 끝에만 하고 지우기는 잠금 안에서 «쓴 만큼 앞에서»만 하므로 그 사이 들어온 체결은 남는다.
_BUF: list[tuple] = []
_BUF_LOCK = asyncio.Lock()


async def flush_buf() -> tuple[int, int, int]:
    """버퍼를 DB 에 쓰고 쓴 만큼 앞에서 지운다. 쓰기 실패는 예외 그대로(버퍼는 남는다). (행 수, 새 행, 블록)"""
    async with _BUF_LOCK:
        rows = _BUF[:]
        added = await asyncio.to_thread(write, rows)
        del _BUF[:len(rows)]
        return len(rows), added, sum(r[12] for r in rows)


async def chain_loop() -> None:
    while True:
        t0 = time.time()
        try:
            if _BUF:
                await flush_buf()
        except Exception as e:           # 못 쓰면 WS 루프의 다음 flush 가 다시 시도한다 -- 체인은 그대로 간다
            print(f"체인 전 flush 보류: {type(e).__name__}: {e}", flush=True)
        try:
            await asyncio.to_thread(chain_poll)
        except Exception as e:           # 체인 조회가 실패해도 체결 수집은 계속 -- 다음 주기에 다시
            print(f"체인 폴링 실패: {type(e).__name__}: {e}", flush=True)
        if time.time() - _RECON["at"] >= RECON_SEC:
            _RECON["at"] = time.time()
            try:
                print(f"대조 {await asyncio.to_thread(reconcile)}", flush=True)
            except Exception as e:       # 대조는 부가 작업 -- 다음 시간에 다시
                print(f"대조 실패: {type(e).__name__}: {e}", flush=True)
        await asyncio.sleep(next_chain_wait(time.time()))


# 2026-10-01 연구: 체인 폴링이 기동 시각 기준(:50:27 등)이라 10분 스냅샷 사이 ΔOI(신규/청산 판별)가 정시 칸과 어긋났다 --
#   정시(:00, :10 …) + 5초에 맞춘다. 남은 시간이 30초보다 짧으면 한 칸 건너뛴다(연속 폴링 방지).
def next_chain_wait(now: float) -> float:
    w = CHAIN_SEC - (now % CHAIN_SEC) + 5.0
    return w + CHAIN_SEC if w < 30.0 else w


RECON_SEC = 3600.0
RECON_LOOKBACK_MS = 3 * 3600 * 1000     # history 반영이 늦어도 겹치게 3시간
_RECON = {"at": 0.0}
HIST = "https://history.deribit.com/api/v2/public/"
API_PUB = "https://www.deribit.com/api/v2/public/"
FUT_API = {"ETH": "ETH", "BTC": "BTC", "SOL": "USDC", "XRP": "USDC"}


def _get(base: str, method: str, **params) -> dict:
    import requests
    r = requests.get(base + method, params=params, timeout=30)
    r.raise_for_status()
    return r.json()["result"]


def reconcile() -> dict:
    """매시 한 번(2026-10-01 연구 뒤 추가). ① history.deribit.com 으로 지난 3시간 옵션 체결을 다시 받아 빠진 행을 넣고
    강제청산 표시(liquidation)를 채운다 -- 서버는 같은 구간 12건 중 1건만 갖고 있었다(WS 가 안 싣는 것으로 추정).
    ② 다리 수(block_trade_leg_count)가 옵션 다리보다 많은 블록 = 선물 헤지 다리가 있는 블록 → 같은 ms 선물 체결에서 찾아 저장.
    ③ 블록 RFQ 패키지(get_block_rfq_trades, 보존 ~6일)를 저장."""
    now = int(time.time() * 1000)
    since = now - RECON_LOOKBACK_MS
    rows: list[tuple] = []
    for api in API_CURRENCIES:
        t = since
        while True:
            res = _get(HIST, "get_last_trades_by_currency_and_time", currency=api, kind="option", start_timestamp=t,
                       end_timestamp=now, count=1000, sorting="asc", include_old="true")
            tr = res.get("trades") or []
            rows += [row(x) for x in tr if coin_of(x["instrument_name"])]
            nxt = max((x["timestamp"] for x in tr), default=t)
            if not res.get("has_more") or nxt <= t:
                break
            t = nxt
    liq = [(r[11], r[1]) for r in rows if r[11]]
    fut: list[tuple] = []
    rfq: list[tuple] = []
    con = _connect()
    try:
        con.begin()
        if rows:
            con.executemany(f"INSERT OR IGNORE INTO option_trades VALUES ({','.join('?' * len(COLS))})", rows)
        if liq:
            con.executemany("UPDATE option_trades SET liquidation = ? WHERE trade_id = ? AND liquidation IS NULL", liq)
        con.commit()
        need = con.execute("""SELECT block_trade_id, min(ts_ms), any_value(instrument_name) FROM option_trades
                              WHERE is_block AND ts_ms >= ? GROUP BY 1
                              HAVING max(block_trade_leg_count) > count(*)
                                 AND block_trade_id NOT IN (SELECT block_trade_id FROM block_future_legs)""",
                           [since]).fetchall()
        last_rfq = dict(con.execute("SELECT api, max(ts_ms) FROM block_rfq_trades GROUP BY 1").fetchall())
    finally:
        con.close()
    for bid, ts, inst in need:
        base = API_PUB if now - ts < 20 * 3600 * 1000 else HIST
        for x in _get(base, "get_last_trades_by_currency_and_time", currency=FUT_API[coin_of(inst)], kind="future",
                      start_timestamp=ts, end_timestamp=ts, count=100).get("trades") or []:
            if x.get("block_trade_id") == bid:
                fut.append((str(x["trade_id"]), int(x["timestamp"]), x["instrument_name"], x["direction"], float(x["amount"]),
                            float(x["price"]), x.get("index_price"), bid))
    for api in API_CURRENCIES:
        cont = None
        try:
            for _ in range(20):
                r = _get(API_PUB, "get_block_rfq_trades", currency=api, count=50, **({"continuation": cont} if cont else {}))
                got = r.get("block_rfqs") or []
                rfq += [(str(b["id"]), int(b["timestamp"]), api, b.get("direction"), b.get("amount"),
                         ((b.get("trades") or [{}])[0]).get("price"), b.get("mark_price"), b.get("combo_id"),
                         json.dumps(b.get("legs") or []), json.dumps(b.get("hedge")) if b.get("hedge") else None,
                         json.dumps(b.get("index_prices") or {})) for b in got]
                cont = r.get("continuation")
                if not cont or not got or min(b["timestamp"] for b in got) <= (last_rfq.get(api) or 0):
                    break
        except Exception as e:           # RFQ 목록이 없는 통화(USDC 등)는 건너뛴다
            print(f"  RFQ {api} 건너뜀: {type(e).__name__}: {str(e)[:80]}", flush=True)
    if fut or rfq:
        con = _connect()
        try:
            con.begin()
            if fut:
                con.executemany("INSERT OR IGNORE INTO block_future_legs VALUES (?,?,?,?,?,?,?,?)", fut)
            if rfq:
                con.executemany("INSERT OR IGNORE INTO block_rfq_trades VALUES (?,?,?,?,?,?,?,?,?,?,?)", rfq)
            con.commit()
        finally:
            con.close()
    return {"체결": len(rows), "강제청산": len(liq), "헤지 다리": len(fut), "RFQ": len(rfq)}


_DOI = {"key": None, "val": {}}


def hourly_doi(con) -> dict:
    """코인·정시별 미결제 변화(ΔOI, 기초자산 수량) = 그 시각 첫 체인 스냅샷 → 다음 시각 첫 스냅샷, **두 스냅샷에 다 있는 종목만**
    (만기로 사라지는 종목의 감소는 청산이 아니다 -- 2026-10-01 연구). 스냅샷이 바뀔 때만 다시 센다(write 는 20초마다)."""
    try:
        key = con.execute("SELECT max(recorded_at_utc) FROM option_chain_snapshot").fetchone()[0]
    except Exception:
        return {}
    if key is None or key == _DOI["key"]:
        return _DOI["val"]
    rows = con.execute("""
        WITH s AS (SELECT recorded_at_utc t, currency, instrument_name, open_interest FROM option_chain_snapshot
                   WHERE recorded_at_utc >= now() - INTERVAL 26 HOUR),
             hs AS (SELECT currency, date_trunc('hour', t) h, min(t) t0 FROM s GROUP BY 1, 2),
             a AS (SELECT s.currency, hs.h, s.instrument_name, s.open_interest oi FROM s JOIN hs ON s.currency = hs.currency AND s.t = hs.t0)
        SELECT a.currency, epoch_ms(a.h), sum(b.oi - a.oi) FROM a JOIN a b
          ON a.currency = b.currency AND a.instrument_name = b.instrument_name AND b.h = a.h + INTERVAL 1 HOUR
        GROUP BY 1, 2""").fetchall()
    _DOI.update(key=key, val={(c, int(h)): float(d) for c, h, d in rows})
    return _DOI["val"]


def hourly_flow(rows, doi: dict | None = None) -> dict:
    """2026-09-28 옵션 순매수 흐름(사용자 선택 2-A) -- 코인별 정시(UTC=KST 정시) 버킷 25개(지난 24시간 + 진행 중).
    cb/cs/pb/ps = 콜·풋 테이커 매수·매도 수량(기초자산 단위), dlt = 옵션으로 산 순델타(콜 매수·풋 매도 +).
    델타는 체결 시점의 IV·지수로 블랙-숄즈(r=0, 선도 대신 지수 -- ponytail: 먼 만기는 캐리만큼 어긋난다, 흐름 방향엔 영향 작음).
    🔴방향은 Deribit 공개 체결의 «테이커 방향»이다(문서 확인 09-28). 개시/청산은 모른다."""
    import collect_deribit_option_gex_20260815 as gex
    now_h = int(time.time() // 3600) * 3_600_000
    hours = [now_h - i * 3_600_000 for i in range(24, -1, -1)]
    # 2026-10-01 liq = 강제청산 체결 수량 · doi = 그 시간 미결제 변화(hourly_doi, 없으면 None) → 화면이 신규 비율 (V+ΔOI)/2V 를 낸다
    # 2026-10-01 vg = 테이커 순베가($/1 vol pt, 매수 +) · vgb = 그중 블록 -- «변동성을 사는 흐름»(Alexander 외 2023, Ni·Pan·Poteshman 2008)
    out = {c: {h: {"h": h, "cb": 0.0, "cs": 0.0, "pb": 0.0, "ps": 0.0, "dlt": 0.0, "vg": 0.0, "vgb": 0.0, "liq": 0.0, "doi": (doi or {}).get((c, h))}
               for h in hours} for c in COINS}
    for ts, inst, direction, amt, iv, ix, *rest in rows:
        c = coin_of(inst)
        spec = gex._parse_instrument(inst) if c else None
        b = out[c].get(int(ts // 3_600_000) * 3_600_000) if spec else None
        if b is None:
            continue
        call = spec["option_type"] == "call"
        buy = direction == "buy"
        b[("cb" if buy else "cs") if call else ("pb" if buy else "ps")] += float(amt)
        if rest and rest[0]:
            b["liq"] += float(amt)
        yrs = (spec["expiration_ts"].timestamp() - ts / 1000) / (365.0 * 86400)
        d = gex._bs_delta(float(ix or 0), spec["strike"], float(iv or 0), yrs, call)
        b["dlt"] += (1 if buy else -1) * float(amt) * d
        S, sig = float(ix or 0), float(iv or 0) / 100
        if S > 0 and sig > 0 and yrs > 0:
            d1 = (math.log(S / spec["strike"]) + 0.5 * sig * sig * yrs) / (sig * math.sqrt(yrs))
            vg = (1 if buy else -1) * float(amt) * S * math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi) * math.sqrt(yrs) / 100
            b["vg"] += vg
            if len(rest) > 1 and rest[1]:
                b["vgb"] += vg
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
        fut: dict = {}
        try:
            for bid, inst, d, a, p in con.execute("SELECT block_trade_id, instrument_name, direction, amount, price FROM block_future_legs "
                                                  "WHERE ts_ms >= ?", [int(time.time() * 1000) - STATE_WINDOW_MS]).fetchall():
                fut.setdefault(bid, []).append({"instrument_name": inst, "direction": d, "amount": a, "price": p})
        except Exception:    # 표가 아직 없으면(옛 DB) 헤지 표시만 빠진다
            fut = {}
        by_coin = {c: group_blocks([x for x in legs if coin_of(x["instrument_name"]) == c], fut) for c in COINS}
        flow = hourly_flow(con.execute("SELECT ts_ms, instrument_name, direction, amount, iv, index_price, liquidation, is_block FROM option_trades "
                                       "WHERE ts_ms >= ?", [(int(time.time() // 3600) - 24) * 3_600_000]).fetchall(),
                           hourly_doi(con))
    finally:
        con.close()      # 붙들고 있으면 연구 쿼리가 막힌다
    # blocks/n_blocks = ETH(옛 계약 그대로) · blocks_by_coin = 네 코인
    # 2026-10-01 블록 요청자 쪽 24시간 순델타(코인)·순베가($/1 vol pt) 합 -- 블록이 방향을 샀나 변동성을 샀나(서술)
    bsum = {c: {"net_delta": round(sum(b.get("net_delta") or 0 for b in v), 3), "net_vega_usd": round(sum(b.get("net_vega_usd") or 0 for b in v), 1),
                "n": len(v)} for c, v in by_coin.items()}
    out = {"generated_at": datetime.now(timezone.utc).isoformat(), "coins": list(COINS), "block_sum_by_coin": bsum,
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
                #   gaps 표에 체결 이력이 빈 구간이 남는다(2026-10-02 딜러 감마 제거 뒤에도 원자료 결손 기록으로 유지).
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
                        _BUF.extend(row(t) for t in p["data"] if coin_of(t["instrument_name"]))
                    if time.time() - last >= FLUSH_SEC:
                        try:
                            n, added, nb = await flush_buf()
                        except Exception as e:     # 쓰기 실패는 WS 를 끊을 이유가 아니다 -- 버퍼를 들고 다음 주기에 다시
                            print(f"  기록 보류 {len(_BUF)}행: {type(e).__name__}: {str(e)[:120]}", flush=True)
                            last = time.time()
                            continue
                        print(f"  +{n}행(새 {added}, 블록 {nb})", flush=True)
                        last = time.time()
        except Exception as e:
            # 틈은 다음 연결의 REST 백필이 메우고, 끊긴 구간은 gaps 에 «메운 건수»와 함께 남긴다.
            print(f"WS 끊김: {type(e).__name__}: {e} — 5초 후 재연결", flush=True)
            down = down or (int(time.time() * 1000), type(e).__name__)
            try:
                await flush_buf()
            except Exception as e2:   # 버리지 않고 다음 연결에서 다시 쓴다(백필도 메우지만 24h 를 넘기면 영구 유실이라)
                print(f"  잔여 {len(_BUF)}행 기록 보류: {e2}", flush=True)
                del _BUF[:-200_000]
            await asyncio.sleep(5)


def _selftest() -> None:
    L = lambda inst, d, a, iv=50.0: {"instrument_name": inst, "direction": d, "amount": a, "iv": iv, "index_price": 2600.0}   # noqa: E731
    ts = 1_790_000_000_000     # 2026-09-21
    st = block_shape([L("ETH-16OCT26-2600-C", "sell", 100), L("ETH-16OCT26-2600-P", "sell", 100)], ts)
    assert st["structure"] == "straddle" and st["vol_side"] == "short_vol" and st["hedge"] is False, st
    vs = block_shape([L("ETH-16OCT26-2600-C", "buy", 50), L("ETH-16OCT26-2800-C", "sell", 50)], ts)
    assert vs["structure"] == "vertical" and vs["delta_side"] == "long", vs
    hg = block_shape([L("ETH-16OCT26-2600-C", "buy", 100)], ts,
                     [{"instrument_name": "ETH-PERPETUAL", "direction": "sell", "amount": 0.5 * 100 * 2600, "price": 2600.0}])
    assert hg["structure"] == "single" and hg["hedge"] and hg["delta_side"] == "neutral", hg      # ATM 콜 ≈ 0.5Δ 를 선물로 상쇄
    fl = hourly_flow([(ts, "ETH-16OCT26-2600-C", "buy", 3.0, 50.0, 2600.0, "T"), (ts, "ETH-16OCT26-2600-P", "sell", 2.0, 50.0, 2600.0, None)],
                     {("ETH", ts // 3_600_000 * 3_600_000): 1.5})
    # hourly_flow 는 «지금부터 24시간 전»만 버킷을 만든다 -- 이 ts 는 과거라 버킷 밖일 수 있으니, 있을 때만 확인
    hb = [b for b in fl["ETH"] if b["h"] == ts // 3_600_000 * 3_600_000]
    assert not hb or (hb[0]["liq"] == 3.0 and hb[0]["doi"] == 1.5 and hb[0]["cb"] == 3.0 and hb[0]["ps"] == 2.0), hb
    now_ts = int(time.time() * 1000)
    fl2 = hourly_flow([(now_ts, "ETH-16OCT27-2600-C", "buy", 3.0, 50.0, 2600.0, "T")], {("ETH", now_ts // 3_600_000 * 3_600_000): 1.5})
    b2 = [b for b in fl2["ETH"] if b["h"] == now_ts // 3_600_000 * 3_600_000][0]
    assert b2["liq"] == 3.0 and b2["doi"] == 1.5 and b2["cb"] == 3.0, b2
    fl3 = hourly_flow([(now_ts, "ETH-16OCT27-2600-C", "buy", 2.0, 50.0, 2600.0, None, True), (now_ts, "ETH-16OCT27-2600-P", "sell", 1.0, 50.0, 2600.0, None, False)])
    b3 = [b for b in fl3["ETH"] if b["h"] == now_ts // 3_600_000 * 3_600_000][0]
    assert b3["vg"] > 0 and abs(b3["vgb"] - 2 * b3["vg"]) < 1e-9, b3       # 같은 행사가 콜 2 매수(블록)·풋 1 매도 → 순베가 = 블록의 절반
    b0 = 600.0 * 1_666_667                                              # 10분 경계
    assert next_chain_wait(b0 + 200) == 600 - 200 + 5                    # :03:20 → 다음 :10:05
    assert next_chain_wait(b0 + 590) == 600 - 590 + 5 + 600              # 15초 남으면 한 칸 건너뜀
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
