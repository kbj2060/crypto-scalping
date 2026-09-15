#!/usr/bin/env python3
"""**체결 테이프 수집기** — 1초 × 가격빈으로 모은 공격적 매수/매도. (2026-09-16)

왜 필요한가: 이 저장소는 호가는 연속으로 쌓지만(orderflow/ 래스터·bookTicker·depthDiff)
**체결은 어디에도 연속으로 안 쌓인다**. 2026-09-15 실측 — `l2_anomaly_trades` 는 이상치 구간만
이라 최근 1시간 12봉 중 9봉이 **0%**, 나머지도 3~17% 였고, `maker_fill_shadow` 는 시뮬레이션
원장이며 `microstructure.duckdb` 는 1분 집계(가격 없음)다. 방향 예측 재료로 쓰려면 새로 받아야
한다. (풋프린트 대시보드는 자기 WS 로 자기 창만 들고 있다 — duckdb 는 프로세스 하나만 열 수
있어서 공유가 불가능하다. 그래서 둘은 서로를 모른다.)

**왜 원시 틱이 아니라 1초 집계인가 — 실측하고 정했다(2026-09-16, ETHUSDT):**
  원시 틱        11.3M 행/일   (체결 크기 중앙값 **0.010 ETH** — 대부분 먼지 체결이다)
  1초 × $0.01    1.94M 행/일   (초당 22.5빈)
  1초 × $0.1      326k 행/일   (초당 3.8빈)  ← 채택, ~4GB/년
$0.1 은 $2,400 자산에서 0.4bp 다. 이 저장소의 왕복 비용(5.88bp)보다 15배 가늘어서 어떤 결정도
이 격자 때문에 바뀌지 않는다. 잃는 건 «한 초 안의 체결 순서»뿐이고, 크기 분포는 빈마다
`n`(건수)과 `max`(그 초의 최대 체결)로 남긴다 — 고래 프린트(상위1% 13.6 ETH)는 보존된다.
원시 틱이 정말 필요해지면 `data.binance.vision` 일별 zip 으로 소급 재구성한다(호가와 달리
체결은 공개돼 있다 — docs/dashboard_orderflow_footprint_heatmap_design_20260914.md §1).

**자기 검증이 내장돼 있다(선택이 아니다).** 체결이 빠지면 델타가 **조용히** 틀어진다 — 화면도
쿼리도 아무 말을 안 한다. 그래서 5분마다 직전 분들의 `sum(buy_qty+sell_qty)` 를 같은 분의
kline `volume` 과 대조해 `verify_1m` 에 남긴다. 연구 쿼리는 이 표를 조인해 나쁜 분을 빼면 된다.
WS 가 끊긴 구간은 `gaps` 에 기록한다 — 메우지 않고 **기록만** 한다(메우면 그게 진짜인지
아닌지 알 수 없어진다. 메울 때는 벌크 zip 으로 그 날을 통째로 덮어쓴다).

트레이딩 봇과 완전 분리: 자기 WS · 자기 duckdb · 주문 없음 · 죽어도 봇에 영향 없다.
(래스터/bookTicker 수집기 도크스트링의 규약을 그대로 따른다.)
⚠️@aggTrade 는 2026-09-02 부터 바이낸스가 배달을 멈췄다(구독은 조용히 성공하고 메시지가 0건).
  `@trade`(개별 체결)를 쓴다. `m`=매수자가 메이커 → **공격자는 매도자**.

사용:
  python scripts/live_trade_tape_collector_20260916.py                 # ethusdt
  TAPE_SYMBOL=btcusdt python scripts/live_trade_tape_collector_20260916.py
  python scripts/live_trade_tape_collector_20260916.py --selftest      # 네트워크·DB 없이 로직 점검
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data" / "live" / "trade_tape.duckdb"
# 가격빈. 자산마다 틱과 가격대가 달라 한 값을 못 쓴다 -- 대략 «가격의 0.4bp» 로 맞췄다.
BUCKETS = {"ethusdt": 0.1, "btcusdt": 1.0, "solusdt": 0.01, "xrpusdt": 0.0001, "hypeusdt": 0.001}
WS_URL = "wss://fstream.binance.com/ws/{symbol}@trade"
KLINES_URL = "https://fapi.binance.com/fapi/v1/klines"
FLUSH_SECONDS = 5.0      # 완결된 초만 쓴다 -- 한 초는 정확히 한 번 기록된다
VERIFY_SECONDS = 300.0
VERIFY_TOLERANCE = 1e-4  # 이보다 어긋나면 로그로 떠든다(행은 어차피 남긴다)
SCHEMA_VERSION = 1


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] {msg}", flush=True)


class TapeBuffer:
    """(초, 가격빈) -> [매수량, 매도량, 매수건수, 매도건수, 매수최대, 매도최대].

    DB 도 네트워크도 모른다 -- 그래서 --selftest 가 이 클래스만 찔러 볼 수 있다."""

    def __init__(self, bucket: float) -> None:
        self.bucket = bucket
        self.rows: dict[tuple[int, int], list[float]] = {}
        self.max_sec = 0

    def add(self, ts_ms: int, price: float, qty: float, sell: bool) -> None:
        sec = ts_ms // 1000
        self.max_sec = max(self.max_sec, sec)
        cell = self.rows.get((sec, round(price / self.bucket)))
        if cell is None:
            cell = self.rows[(sec, round(price / self.bucket))] = [0.0, 0.0, 0, 0, 0.0, 0.0]
        i = 1 if sell else 0
        cell[i] += qty
        cell[2 + i] += 1
        cell[4 + i] = max(cell[4 + i], qty)

    def take_closed(self) -> list[tuple]:
        """진행 중인 초(max_sec)를 빼고 꺼낸다. 그 초는 아직 체결이 더 올 수 있다."""
        done = [(sec, b, c) for (sec, b), c in self.rows.items() if sec < self.max_sec]
        for sec, b, _ in done:
            del self.rows[(sec, b)]
        return sorted(
            (sec, b, c[0], c[1], int(c[2]), int(c[3]), c[4], c[5]) for sec, b, c in done)


class TapeStore:
    def __init__(self, db_path: Path, symbol: str, bucket: float) -> None:
        import duckdb

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.con = duckdb.connect(str(db_path))
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS trade_tape_1s(
              symbol VARCHAR, ts_sec BIGINT, price_bin INTEGER,
              buy_qty DOUBLE, sell_qty DOUBLE, buy_n INTEGER, sell_n INTEGER,
              buy_max DOUBLE, sell_max DOUBLE)""")
        # 끊긴 구간. 연구 쿼리는 이 표를 봐야 «0» 과 «모름» 을 구분할 수 있다.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS gaps(
              symbol VARCHAR, from_ms BIGINT, to_ms BIGINT, reason VARCHAR)""")
        # 분별 완전성. kline volume 과 맞는지 -- 체결 유실은 이 표에서만 드러난다.
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS verify_1m(
              symbol VARCHAR, ts_min BIGINT, tape_qty DOUBLE, kline_qty DOUBLE,
              rel_err DOUBLE, checked_at TIMESTAMP)""")
        self.con.execute("CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)")
        for key, value in (("schema_version", str(SCHEMA_VERSION)),
                           (f"bucket:{symbol}", repr(bucket))):
            self.con.execute("DELETE FROM meta WHERE key = ?", [key])
            self.con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    def last_ts_ms(self) -> int:
        row = self.con.execute(
            "SELECT max(ts_sec) FROM trade_tape_1s WHERE symbol = ?", [self.symbol]).fetchone()
        return int(row[0] + 1) * 1000 if row and row[0] else 0

    def write(self, rows: list[tuple]) -> None:
        if rows:
            self.con.executemany(
                "INSERT INTO trade_tape_1s VALUES (?,?,?,?,?,?,?,?,?)",
                [(self.symbol, *r) for r in rows])

    def record_gap(self, from_ms: int, to_ms: int, reason: str) -> None:
        if to_ms - from_ms < 1500:   # 재연결 한 번에 1초 미만이면 기록할 값이 없다
            return
        self.con.execute("INSERT INTO gaps VALUES (?,?,?,?)",
                         [self.symbol, from_ms, to_ms, reason])
        log(f"gap {(to_ms - from_ms) / 1000:.1f}s 기록 ({reason})")

    def minute_has_gap(self, ts_min: int) -> bool:
        """그 분이 기록된 공백과 겹치는가. 겹치면 그 분이 부족한 건 «유실»이 아니라 «안 받은 것»
        이다 -- 둘을 같은 말로 경고하면 매번 켤 때마다 울려서 아무도 안 본다."""
        # 밀리초 환산을 SQL 안에서 하면 안 된다 -- duckdb 가 바인드 파라미터를 INT32 로 보고
        # `1789485840 * 1000` 에서 오버플로를 낸다(2026-09-16 시험에서 실제로 터졌다. 검증이
        # 매번 예외로 죽어 «조용히 검증 안 함» 이 될 뻔했고, 로그에만 재연결로 보였다).
        row = self.con.execute(
            "SELECT count(*) FROM gaps WHERE symbol = ? AND from_ms < ? AND to_ms > ?",
            [self.symbol, (ts_min + 60) * 1000, ts_min * 1000]).fetchone()
        return bool(row[0])

    def unverified_minutes(self, limit: int = 5) -> list[int]:
        rows = self.con.execute("""
            SELECT DISTINCT ts_sec // 60 * 60 AS m FROM trade_tape_1s
            WHERE symbol = ? AND ts_sec < ? - 60
              AND NOT EXISTS (SELECT 1 FROM verify_1m v
                              WHERE v.symbol = trade_tape_1s.symbol AND v.ts_min = m)
            ORDER BY m DESC LIMIT ?""", [self.symbol, int(time.time()), limit]).fetchall()
        return [int(r[0]) for r in rows]

    def minute_qty(self, ts_min: int) -> float:
        row = self.con.execute("""
            SELECT coalesce(sum(buy_qty + sell_qty), 0) FROM trade_tape_1s
            WHERE symbol = ? AND ts_sec >= ? AND ts_sec < ? + 60""",
            [self.symbol, ts_min, ts_min]).fetchone()
        return float(row[0])

    def record_verify(self, ts_min: int, tape_qty: float, kline_qty: float) -> float:
        rel = (tape_qty - kline_qty) / kline_qty if kline_qty else 0.0
        self.con.execute("INSERT INTO verify_1m VALUES (?,?,?,?,?,now())",
                         [self.symbol, ts_min, tape_qty, kline_qty, rel])
        return rel


async def verify_recent(store: TapeStore, session) -> None:
    """직전 분들을 kline volume 과 대조해 verify_1m 에 남긴다. 실패해도 수집은 계속한다."""
    minutes = store.unverified_minutes()
    if not minutes:
        return
    params = {"symbol": store.symbol.upper(), "interval": "1m",
              "startTime": min(minutes) * 1000, "limit": len(minutes) + 2}
    async with session.get(KLINES_URL, params=params) as response:
        if response.status != 200:
            return
        klines = await response.json()
    kvol = {int(k[0]) // 1000: float(k[5]) for k in klines}
    for ts_min in minutes:
        if ts_min not in kvol:
            continue
        rel = store.record_verify(ts_min, store.minute_qty(ts_min), kvol[ts_min])
        if abs(rel) <= VERIFY_TOLERANCE:
            continue
        stamp = time.strftime('%H:%M', time.localtime(ts_min))
        if store.minute_has_gap(ts_min):
            log(f"완전성 {stamp} rel_err {rel:+.4%} -- 기록된 공백과 겹친다(예상된 부족)")
        else:
            log(f"⚠️완전성 {stamp} rel_err {rel:+.4%}"
                " -- 체결 유실. 그 구간은 벌크 zip 으로 덮어써야 한다")


async def collect(symbol: str, db_path: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    bucket = BUCKETS.get(symbol, 0.01)
    store = TapeStore(db_path, symbol, bucket)
    buffer = TapeBuffer(bucket)
    last_ms = store.last_ts_ms()      # 지난 판이 남긴 끝 -- 재시작 공백을 gaps 에 적으려고
    log(f"{symbol} 수집 시작 (빈 {bucket}, db {db_path})")
    # total=None 을 **명시**한다: aiohttp 기본 5분이라 그냥 두면 5분마다 끊긴다.
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        flushed_at = verified_at = time.monotonic()
        while True:
            try:
                async with session.ws_connect(WS_URL.format(symbol=symbol), heartbeat=30) as ws:
                    first = True
                    async for msg in ws:
                        if msg.type is not WSMsgType.TEXT:
                            break
                        trade = json.loads(msg.data)
                        if trade.get("e") != "trade":
                            continue
                        price, qty = float(trade["p"]), float(trade["q"])
                        if not (price > 0 and qty > 0):
                            # 바이낸스가 {"p":"0","q":"0","X":"NA","st":1} 를 섞어 보낸다
                            # (2026-09-16 실측 12,312건 중 37건=0.3%). 합계는 안 틀리지만
                            # **가격 0 자리에 빈이 생긴다** -- 받는 쪽에서 막는다.
                            continue
                        ts_ms = int(trade["T"])
                        if first:
                            first = False
                            # 처음 켠 판이면 그 «분의 시작부터 여기까지»가 안 받은 구간이다.
                            # 적어두지 않으면 아래 완전성 검사가 그걸 유실로 오해한다.
                            store.record_gap(last_ms or (ts_ms // 60_000) * 60_000, ts_ms,
                                             "ws_reconnect" if last_ms else "startup")
                            log("스트림 연결됨")
                        buffer.add(ts_ms, price, qty, bool(trade["m"]))
                        last_ms = ts_ms
                        now = time.monotonic()
                        if now - flushed_at >= FLUSH_SECONDS:
                            flushed_at = now
                            store.write(buffer.take_closed())
                        if now - verified_at >= VERIFY_SECONDS:
                            verified_at = now
                            await verify_recent(store, session)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 한 번의 끊김이 수집기를 죽이면 그 뒤가
                # 통째로 빈다. 빈 구간은 소급 불가가 아니지만(zip) 알아채는 게 늦어진다.
                log(f"연결 실패, 3초 뒤 재시도: {type(exc).__name__} {exc}")
            store.write(buffer.take_closed())
            await asyncio.sleep(3.0)


def selftest() -> None:
    """네트워크·DB 없이 집계 규칙만 점검한다."""
    buf = TapeBuffer(0.1)
    buf.add(1_000_000, 2440.04, 1.5, sell=False)   # -> 빈 24400 (반올림)
    buf.add(1_000_400, 2440.02, 0.5, sell=True)    # 같은 초·같은 빈, 반대쪽
    buf.add(1_000_900, 2440.44, 3.0, sell=False)   # 같은 초, 다른 빈(24404)
    buf.add(1_001_000, 2440.04, 9.0, sell=False)   # 다음 초 -> 진행 중이라 안 나온다
    rows = buf.take_closed()
    assert [r[:2] for r in rows] == [(1000, 24400), (1000, 24404)], rows
    sec, _bin, buy, sell, buy_n, sell_n, buy_max, sell_max = rows[0]
    assert (buy, sell, buy_n, sell_n, buy_max, sell_max) == (1.5, 0.5, 1, 1, 1.5, 0.5), rows[0]
    assert buf.rows and buf.max_sec == 1001, "진행 중인 초는 남아 있어야 한다"
    assert buf.take_closed() == [], "같은 초를 두 번 쓰면 안 된다"
    buf.add(1_002_000, 2440.04, 2.0, sell=True)    # 1001 초가 완결됨
    assert [r[:2] for r in buf.take_closed()] == [(1001, 24400)]
    print("selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--symbol", default=os.getenv("TAPE_SYMBOL", "ethusdt").lower())
    parser.add_argument("--db", type=Path, default=Path(os.getenv("TAPE_DB_PATH", DEFAULT_DB)))
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        return
    try:
        asyncio.run(collect(args.symbol, args.db))
    except KeyboardInterrupt:
        log("종료")


if __name__ == "__main__":
    main()
