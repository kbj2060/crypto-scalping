#!/usr/bin/env python3
"""**하이퍼리퀴드 고래 포지션 수집기** — 실제 진입가·청산가·레버리지. (2026-09-24)

왜: HL 은 `clearinghouseState` 로 **아무 주소의 실제 포지션**(수량·진입가·청산가·레버리지)을 공개한다.
  바이낸스 청산 지도는 추정이지만 이건 실제 값이다(2026-09-24 실측: 24h 상위 테이커 하나가
  ETH +36,686 · 25x · 청산가 2,543.87). 그리고 **스냅샷은 소급이 불가능하다** -- 쌓아야 나중에
  «가격이 큰 청산 뭉치로 끌려가는가 / 뭉치를 지날 때 연쇄가 나는가»를 검정할 수 있다.
  같은 날 검정에서 HL 흐름·«스마트 지갑»(5분)·HL 가격 리드는 전부 기각됐다 -- 이 축이 HL 의
  유일한 고유 자산이다(memory/hyperliquid_data_research_20260924.md).

대상: 최근 48시간 HL 체결(`live_hyperliquid_trade_collector_20260916.py` 의 날짜별 duckdb) 거래액
  상위 UNIVERSE_N 주소(테이커+메이커). 6시간마다 다시 뽑는다. 🔴«거래가 많은 주소» ≠ «포지션이 큰
  주소» -- 오래 들고만 있는 고래는 빠진다. HL 에 전 주소 포지션 목록 API 는 없다(업그레이드 경로:
  청산가가 가까운 주소를 계속 붙들어 두기).
요청 한도: info 가중치 1,200/분 · clearinghouseState 가중치 2 -- 0.4초 간격(150회/분 = 300)이면 25%.
시각: `ts_ms` 는 **서버 시계**다(응답에 거래소 시각 `time` 이 오면 그것을 `ex_ms` 로 같이 적는다).

⭐고래 청산 감지(2026-09-24 추가): 바퀴 사이에 포지션이 **줄었거나 사라진** 주소 중, 그 사이 HL 1분봉
  고가~저가가 **직전 청산가 ±LIQ_TOL** 을 지나간 주소만 `userFillsByTime` 으로 조회해, 본인이 청산당한
  체결(`liquidation.liquidatedUser == 그 주소`)을 `hl_liquidations` 에 적는다. 모든 테이커를 조회하는
  전수 식별은 너무 비싸다(급변 8분·64회 조회에 ETH 청산 1건) -- 관심 대상(큰손)만 정확히 잡는다.
  🔴청산 체결의 `liquidation` 표시는 **상대방 쪽 기록에도** 붙는다 -- 그래서 liquidatedUser 로 거른다.

⭐여러 코인(2026-09-26): `clearinghouseState` 는 한 주소의 **전 코인** 포지션을 한 번에 준다. 그래서
  코인마다 프로세스를 띄우면 같은 주소를 코인 수만큼 다시 조회한다(5개면 1,500/분 > 한도 1,200).
  대신 `HL_POS_COINS=BTC,SOL,XRP,HYPE` 로 **한 프로세스**가 코인별 상위 주소의 합집합을 돌고, 응답
  하나에서 그 코인들을 전부 적는다. ETH 프로세스는 그대로 둔다 -- 2.5분 바퀴가 사전등록 검정
  (research_hl_whale_liq_magnet_20260924.py)의 전제라 합치면 ETH 바퀴가 길어진다.
  두 프로세스 합계 300+300 = 600/분(50%). 다코인 바퀴는 합집합 크기 x 0.4초(최대 1,200주소 = 8분).

사용:
  python scripts/live_hyperliquid_positions_collector_20260924.py                        # ETH
  HL_POS_COINS=BTC,SOL,XRP,HYPE python scripts/live_hyperliquid_positions_collector_20260924.py
  python scripts/live_hyperliquid_positions_collector_20260924.py --selftest
"""
from __future__ import annotations

import argparse
import asyncio
import glob
import importlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_tape = importlib.import_module("scripts.live_trade_tape_collector_20260916")   # duckdb_connect_retry · log
log = _tape.log

INFO_URL = "https://api.hyperliquid.xyz/info"
COINS = [c.strip().upper() for c in os.getenv("HL_POS_COINS", os.getenv("HL_POS_COIN", "ETH")).split(",")
         if c.strip()]
# ETH 단독은 기존 파일 그대로(대시보드·연구가 이 경로를 읽는다). 다른 조합은 자기 파일 -- duckdb 는 writer 가 하나다.
_DEFAULT_DB = ("hyperliquid_positions.duckdb" if COINS == ["ETH"]
               else f"hyperliquid_positions_{'_'.join(c.lower() for c in COINS)}.duckdb")
DB = Path(os.getenv("HL_POS_DB", str(ROOT / "data" / "live" / _DEFAULT_DB)))
TRADES_ROOT = Path(os.getenv("HL_ROOT", str(ROOT / "data" / "live" / "orderflow" / "hyperliquid")))
UNIVERSE_N = 300        # 코인마다. 다코인이면 코인별 상위 N 의 합집합이다
UNIVERSE_HOURS = 48
UNIVERSE_REFRESH_S = 6 * 3600
REQ_GAP_S = 0.4
LIQ_TOL = 0.005        # 청산가가 바퀴 사이 가격 범위의 ±0.5% 안이면 조회
# ponytail: 단일 duckdb -- ETH 포지션 있는 주소만 적어 ~수만 행/일이라 당장은 작다. 커지면 날짜별 분할.

DDL = (
    """CREATE TABLE IF NOT EXISTS hl_positions(
         ts_ms BIGINT, ex_ms BIGINT, user VARCHAR, coin VARCHAR, szi DOUBLE, entry_px DOUBLE,
         liq_px DOUBLE, leverage DOUBLE, lev_type VARCHAR, margin_used DOUBLE, position_value DOUBLE,
         unrealized_pnl DOUBLE, cum_funding DOUBLE, account_value DOUBLE)""",
    """CREATE TABLE IF NOT EXISTS hl_universe(
         ts_ms BIGINT, user VARCHAR, rank INTEGER, notional_48h DOUBLE)""",
    # 2026-09-26 다코인: 순위는 코인별이다. 이전 행(NULL)은 전부 ETH 단독 프로세스가 적은 것이다.
    "ALTER TABLE hl_universe ADD COLUMN IF NOT EXISTS coin VARCHAR",
    """CREATE TABLE IF NOT EXISTS hl_liquidations(
         detected_ms BIGINT, tid BIGINT, user VARCHAR, coin VARCHAR, fill_ms BIGINT, px DOUBLE,
         sz DOUBLE, side VARCHAR, dir VARCHAR, start_position DOUBLE, closed_pnl DOUBLE,
         mark_px DOUBLE, method VARCHAR)""",
    """CREATE TABLE IF NOT EXISTS hl_cycles(
         ts_ms BIGINT, n_users INTEGER, n_ok INTEGER, n_pos INTEGER, seconds DOUBLE)""",
)
WIDTH = 14


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_state(state: dict, user: str, coins, ts_ms: int) -> list[tuple]:
    """`clearinghouseState` 응답 -> `coins` 에 든 코인별 포지션 행(포지션 0 인 코인은 뺀다).
    🔴`liquidationPx` 는 교차 증거금에서 **없을 수 있다**(None -- 0 이 아니다: 담보가 충분해 계산상
      청산가가 없거나 음수)."""
    acct = _f((state.get("marginSummary") or {}).get("accountValue"))
    ex_ms = int(state["time"]) if str(state.get("time", "")).isdigit() else None
    out = []
    for ap in state.get("assetPositions") or []:
        p = ap.get("position") or {}
        coin = p.get("coin")
        if coin not in coins:
            continue
        szi = _f(p.get("szi"))
        if not szi:
            continue
        lev = p.get("leverage") or {}
        out.append((ts_ms, ex_ms, user, coin, szi, _f(p.get("entryPx")), _f(p.get("liquidationPx")),
                    _f(lev.get("value")), str(lev.get("type") or ""), _f(p.get("marginUsed")),
                    _f(p.get("positionValue")), _f(p.get("unrealizedPnl")),
                    _f((p.get("cumFunding") or {}).get("sinceOpen")), acct))
    return out


def liquidation_candidates(prev: dict[str, tuple[float, float | None]],
                           cur: dict[str, float], low: float, high: float,
                           tol: float = LIQ_TOL) -> list[str]:
    """직전 바퀴 {주소: (szi, 청산가)} · 이번 {주소: szi} · 그 사이 가격 저가/고가 -> 조회할 주소.
    포지션이 줄었거나(부호가 바뀐 것 포함) 사라졌고, 직전 청산가가 [저가·(1-tol), 고가·(1+tol)] 안."""
    out = []
    for u, (szi, liq) in prev.items():
        if not liq or not szi:
            continue
        now = cur.get(u, 0.0)
        shrank = now * szi <= 0 or abs(now) < abs(szi) * 0.999
        if shrank and low * (1 - tol) <= liq <= high * (1 + tol):
            out.append(u)
    return out


def parse_liq_fills(fills: list[dict], user: str, coin: str, detected_ms: int) -> list[tuple]:
    """`userFillsByTime` -> 그 주소가 **청산당한** 체결만(상대방 쪽 기록은 뺀다)."""
    rows = []
    for f in fills or []:
        lq = f.get("liquidation") or {}
        if f.get("coin") != coin or str(lq.get("liquidatedUser", "")).lower() != user.lower():
            continue
        rows.append((detected_ms, int(f.get("tid") or 0), user, coin, int(f.get("time") or 0),
                     _f(f.get("px")), _f(f.get("sz")), str(f.get("side") or ""), str(f.get("dir") or ""),
                     _f(f.get("startPosition")), _f(f.get("closedPnl")), _f(lq.get("markPx")),
                     str(lq.get("method") or "")))
    return rows


def universe(now: float, coin: str) -> list[tuple[str, float]]:
    """그 코인의 최근 UNIVERSE_HOURS 체결 거래액 상위 주소(테이커+메이커). 날짜별 duckdb 를 read_only 로."""
    import duckdb
    since = int((now - UNIVERSE_HOURS * 3600) * 1000)
    days = {(datetime.fromtimestamp(now, timezone.utc) - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(3)}
    files = [f for f in glob.glob(str(TRADES_ROOT / coin / "*.duckdb")) if Path(f).stem in days]
    agg: dict[str, float] = {}
    for f in files:
        for _ in range(25):
            try:
                con = duckdb.connect(f, read_only=True)
                break
            except duckdb.IOException:
                time.sleep(0.2)
        else:
            continue
        try:
            rows = con.execute("""
                SELECT u, sum(n) FROM (
                  SELECT buyer AS u, px * sz AS n FROM hl_trades WHERE coin = ? AND ts_ms >= ?
                  UNION ALL SELECT seller, px * sz FROM hl_trades WHERE coin = ? AND ts_ms >= ?)
                GROUP BY 1""", [coin, since, coin, since]).fetchall()
        finally:
            con.close()
        for u, n in rows:
            agg[u] = agg.get(u, 0.0) + float(n)
    return sorted(agg.items(), key=lambda kv: -kv[1])[:UNIVERSE_N]


def merge_universe(ranked: dict[str, list[tuple[str, float]]]) -> list[str]:
    """코인별 순위 -> 조회할 주소(합집합). 순서는 코인들을 번갈아 1위부터 -- 바퀴 도중 끊겨도
    어느 코인도 통째로 빠지지 않는다."""
    users: list[str] = []
    seen: set[str] = set()
    for i in range(max((len(v) for v in ranked.values()), default=0)):
        for coin in ranked:
            if i < len(ranked[coin]) and ranked[coin][i][0] not in seen:
                seen.add(ranked[coin][i][0])
                users.append(ranked[coin][i][0])
    return users


def write(sql_rows: dict[str, list[tuple]]) -> None:
    with _tape.duckdb_connect_retry(DB) as con:
        con.begin()                         # 🔴한 번에 -- 자동커밋이면 행마다 fsync 다
        for table, rows in sql_rows.items():
            if rows:
                marks = ",".join("?" * len(rows[0]))
                con.executemany(f"INSERT INTO {table} VALUES ({marks})", rows)
        con.commit()


async def _info(session, body: dict):
    async with session.post(INFO_URL, json=body) as r:
        r.raise_for_status()
        return await r.json()


async def detect_liquidations(session, prev, rows, prev_t0: float, seen_tids: set[int]) -> list[tuple]:
    """직전 바퀴와 비교해 청산 후보만 조회한다(liquidation_candidates 도크스트링). 코인마다 따로 본다
    -- 가격 범위가 코인마다 다르다. `prev` = {코인: {주소: (szi, 청산가)}}."""
    now_ms = int(time.time() * 1000)
    out: list[tuple] = []
    fills_cache: dict[str, list] = {}      # 한 주소가 두 코인의 후보여도 조회는 한 번(응답이 전 코인이다)
    for coin in COINS:
        if not prev.get(coin):
            continue
        candles = await _info(session, {"type": "candleSnapshot", "req": {
            "coin": coin, "interval": "1m", "startTime": int(prev_t0 * 1000) - 60_000, "endTime": now_ms}})
        if not candles:
            continue
        low = min(float(c["l"]) for c in candles)
        high = max(float(c["h"]) for c in candles)
        cands = liquidation_candidates(prev[coin], {r[2]: r[4] for r in rows if r[3] == coin}, low, high)
        got_coin: list[tuple] = []
        for u in cands:
            if u not in fills_cache:
                fills_cache[u] = await _info(session, {"type": "userFillsByTime", "user": u,
                                                       "startTime": int(prev_t0 * 1000) - 60_000,
                                                       "endTime": now_ms})
                await asyncio.sleep(REQ_GAP_S)
            got = [r for r in parse_liq_fills(fills_cache[u], u, coin, now_ms) if r[1] not in seen_tids]
            seen_tids.update(r[1] for r in got)
            if got:
                sz = sum(r[6] for r in got)
                p0 = prev[coin][u]
                log(f"🔥고래 청산 {u[:10]}… {coin} {sz:,.1f} (직전 {p0[0]:+,.1f} · 청산가 {p0[1]:,.2f} · "
                    f"{len(got)}체결 · {got[0][12]})")
            got_coin += got
        if cands and not got_coin:
            log(f"{coin} 청산 후보 {len(cands)}주소 조회 -- 청산 체결 없음(스스로 줄임)")
        out += got_coin
    return out


async def run() -> None:
    from aiohttp import ClientSession, ClientTimeout
    DB.parent.mkdir(parents=True, exist_ok=True)
    with _tape.duckdb_connect_retry(DB) as con:
        for ddl in DDL:
            con.execute(ddl)
    users: list[str] = []
    picked_at = 0.0
    prev: dict[str, dict[str, tuple[float, float | None]]] = {}   # 직전 바퀴 {코인: {주소: (szi, 청산가)}}
    prev_t0 = 0.0
    seen_tids: set[int] = set()
    coins = set(COINS)
    label = ",".join(COINS)
    log(f"{label} 포지션 수집 시작 (코인별 상위 {UNIVERSE_N}주소 · {REQ_GAP_S}초 간격 · db {DB})")
    async with ClientSession(timeout=ClientTimeout(total=15)) as session:
        while True:
            t0 = time.time()
            if not users or t0 - picked_at >= UNIVERSE_REFRESH_S:
                ranked = {c: await asyncio.to_thread(universe, t0, c) for c in COINS}
                ranked = {c: v for c, v in ranked.items() if v}
                if ranked:
                    users, picked_at = merge_universe(ranked), t0
                    await asyncio.to_thread(write, {"hl_universe": [
                        (int(t0 * 1000), u, i + 1, n, c)
                        for c, v in ranked.items() for i, (u, n) in enumerate(v)]})
                    log(f"대상 {len(users)}주소 갱신 (" + " · ".join(
                        f"{c} {len(v)}·1위 ${v[0][1]/1e6:,.0f}M" for c, v in ranked.items()) + ")")
                elif not users:
                    log("대상 주소를 못 뽑았다(체결 DB 없음?) -- 60초 뒤 다시")
                    await asyncio.sleep(60)
                    continue
            rows, ok = [], 0
            for u in users:
                try:
                    async with session.post(INFO_URL, json={"type": "clearinghouseState", "user": u}) as r:
                        if r.status == 429:
                            log("429 -- 30초 쉰다")
                            await asyncio.sleep(30)
                            continue
                        r.raise_for_status()
                        state = await r.json()
                    ok += 1
                    rows += parse_state(state, u, coins, int(time.time() * 1000))
                except Exception as exc:  # noqa: BLE001 -- 한 주소 실패로 바퀴를 멈추지 않는다
                    if ok == 0 and u == users[0]:
                        log(f"조회 실패: {type(exc).__name__} {exc}")
                await asyncio.sleep(REQ_GAP_S)
            took = time.time() - t0
            liq_rows: list[tuple] = []
            if prev and ok:
                try:
                    liq_rows = await detect_liquidations(session, prev, rows, prev_t0, seen_tids)
                except Exception as exc:  # noqa: BLE001 -- 감지 실패로 수집을 멈추지 않는다
                    log(f"청산 감지 실패: {type(exc).__name__} {exc}")
            if ok:
                prev = {c: {r[2]: (r[4], r[6]) for r in rows if r[3] == c} for c in COINS}
                prev_t0 = t0
            try:
                await asyncio.to_thread(write, {"hl_positions": rows, "hl_liquidations": liq_rows,
                                                "hl_cycles": [(int(t0 * 1000), len(users), ok, len(rows), took)]})
            except Exception as exc:  # noqa: BLE001
                log(f"쓰기 실패(이번 바퀴 유실): {type(exc).__name__} {exc}")
            per = []
            for c in COINS:
                cr = [r for r in rows if r[3] == c]
                long_n = sum(r[4] for r in cr if r[4] > 0)
                short_n = -sum(r[4] for r in cr if r[4] < 0)
                per.append(f"{c} 보유 {len(cr)}주소 · 롱 {long_n:,.0f} 숏 {short_n:,.0f}")
            log(f"바퀴 {took:.0f}s · 응답 {ok}/{len(users)} · " + " | ".join(per))


def selftest() -> None:
    state = {"marginSummary": {"accountValue": "7666554.1"}, "time": 1790258000123,
             "assetPositions": [
                 {"type": "oneWay", "position": {"coin": "BTC", "szi": "1.0"}},
                 {"type": "oneWay", "position": {
                     "coin": "ETH", "szi": "36686.1", "entryPx": "2658.57", "liquidationPx": "2543.8665",
                     "leverage": {"type": "cross", "value": 25}, "marginUsed": "3900000.0",
                     "positionValue": "97800000.0", "unrealizedPnl": "12345.6",
                     "cumFunding": {"allTime": "1.0", "sinceOpen": "-50.5"}}}]}
    rs = parse_state(state, "0xabc", {"ETH"}, 1)
    assert len(rs) == 1 and len(rs[0]) == WIDTH, rs
    r = rs[0]
    assert (r[2], r[4], r[5], r[6], r[7], r[8]) == ("0xabc", 36686.1, 2658.57, 2543.8665, 25.0, "cross"), r
    assert r[1] == 1790258000123 and r[12] == -50.5 and r[13] == 7666554.1, r
    none_liq = json.loads(json.dumps(state)); none_liq["assetPositions"][1]["position"]["liquidationPx"] = None
    assert parse_state(none_liq, "0xabc", {"ETH"}, 1)[0][6] is None, "청산가 없음은 NULL(0 이 아니다)"
    flat = json.loads(json.dumps(state)); flat["assetPositions"][1]["position"]["szi"] = "0.0"
    assert parse_state(flat, "0xabc", {"ETH"}, 1) == [], "포지션 0 은 적지 않는다"
    assert parse_state({"assetPositions": []}, "0xabc", {"ETH"}, 1) == []
    # 다코인: 응답 하나에서 요청한 코인만 전부(여기선 BTC·ETH), 요청 안 한 코인은 뺀다
    both = parse_state(state, "0xabc", {"ETH", "BTC"}, 1)
    assert sorted(x[3] for x in both) == ["BTC", "ETH"], both
    assert [x[3] for x in parse_state(state, "0xabc", {"BTC", "SOL"}, 1)] == ["BTC"]
    # 합집합: 코인을 번갈아 1위부터, 중복 주소는 한 번
    assert merge_universe({"BTC": [("a", 9.0), ("b", 5.0), ("c", 1.0)],
                           "SOL": [("b", 7.0), ("d", 3.0)]}) == ["a", "b", "d", "c"]

    # ── 청산 후보: 줄었거나 사라졌고, 청산가가 그 사이 가격 범위 근처 ─────────────
    prev = {"a": (100.0, 2540.0),     # 사라짐 · 청산가가 범위 안 -> 후보
            "b": (100.0, 2540.0),     # 그대로 -> 아님
            "c": (100.0, 2400.0),     # 사라졌지만 청산가가 멀다 -> 아님
            "d": (-50.0, 2700.0),     # 숏이 줄었고 청산가가 고가 +0.3% -> 후보
            "e": (100.0, None),       # 청산가 없음 -> 아님
            "f": (100.0, 2560.0)}     # 부호가 뒤집힘(롱 -> 숏) · 범위 안 -> 후보
    cur = {"b": 100.0, "d": -20.0, "e": 0.0, "f": -10.0}
    assert sorted(liquidation_candidates(prev, cur, low=2545.0, high=2692.0)) == ["a", "d", "f"]
    # ── 청산 체결: 본인이 청산당한 것만(상대방 쪽 기록은 뺀다) ─────────────────────
    fills = [{"coin": "ETH", "tid": 1, "time": 5, "px": "2541", "sz": "30", "side": "A", "dir": "Close Long",
              "startPosition": "100", "closedPnl": "-900", "liquidation": {"liquidatedUser": "0xAA", "markPx": "2540.5", "method": "market"}},
             {"coin": "ETH", "tid": 2, "time": 6, "px": "2541", "sz": "5", "side": "B", "liquidation": {"liquidatedUser": "0xbb"}},
             {"coin": "BTC", "tid": 3, "liquidation": {"liquidatedUser": "0xaa"}},
             {"coin": "ETH", "tid": 4, "px": "2600", "sz": "1"}]
    lr = parse_liq_fills(fills, "0xaa", "ETH", 9)
    assert len(lr) == 1 and lr[0][1] == 1 and lr[0][6] == 30.0 and lr[0][11] == 2540.5 and lr[0][12] == "market", lr
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--selftest", action="store_true")
    if ap.parse_args().selftest:
        selftest()
        return
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        log("종료")


if __name__ == "__main__":
    main()
