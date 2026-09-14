#!/usr/bin/env python3
"""peg-maker 집행 섀도우 워커 (2026-08-22).

실주문 없이(공개 스트림만) 가상 peg/static maker 주문을 라이브로 굴려 실효 비용(bp/leg)을
기록한다 — maker 체결 시뮬레이션(v1 raw L2 / v2 aggTrades 단독,
`docs/experiments/eth_maker_fill_simulation_l2_20260822.md`)의 라이브 검증 축.
시뮬 예측치(저변동 3.1~3.3bp, 고변동 역방향 조건부 3.8~4.0bp)와 실측 분포를 대조한다.

- 스트림: fstream.binance.com combined — {symbol}@bookTicker + {symbol}@trade
  (2026-09-02 @aggTrade에서 전환 — 바이낸스가 @aggTrade 전송을 중단했는데 구독은 에러 없이
   수락돼 8일간 trade_msgs=0으로 조용히 실패하고 있었다. 아래 heartbeat 경고 참조.)
  (websockets, tail_risk_interceptor.py와 동일 스택/재연결 패턴).
- 가상 주문 체결 규칙(시뮬 v1과 동일한 보수 규칙): 내 가격을 뚫는 반대측 aggressor 체결 →
  체결 / 내 가격에서의 체결은 진입 시점 L1 표시수량(=앞선 큐, bookTicker B/A) 소진 후만 /
  호가 크로스 → 보장 체결 / 큐 감소는 체결로만(취소 무시) / 배치·리페그 후 200ms 지연.
- 도착 스케줄: MAKER_SHADOW_SPACING_S(기본 300s)마다 buy/sell × policies(기본 peg,static),
  타임아웃 MAKER_SHADOW_TIMEOUT_S(기본 120s) → taker 폴백 가격 기록.
- ⚠️결정시점 동기화 arm은 **2026-09-15에 제거**했다(2026-08-24 도입). 봇의 final_action은
  23일 중 99.5%를 값 2에 머물러 전이가 4건뿐이었고(0.17건/일) leg 16개만 만들었다 —
  통계적으로 아무것도 못 한다. 그 대가로 10초마다 봇 duckdb(292MB)에 락을 걸어 ETH에서만
  5,705회 충돌했다. BTC/SOL 테이블은 애초에 final_action을 담지 않아(record_reason=
  omega4_6_1_shadow_decision, 키가 asset/price뿐) 전이 0건이었고, init이 빈 결과를 받으면
  last_ts=""가 그대로 남아 `recorded_at_kst > ''`가 영구 ConversionException이 됐다
  (BTC 86,417 · SOL 86,418회). 원장의 trigger 컬럼은 과거 행(ETH decision 16건)을 위해 남긴다.
- 기록: 자체 duckdb(단일 writer 원칙, data/live/maker_fill_shadow.duckdb) — leg 원장 +
  하트비트(스트림 카운터; tail_risk의 '핸드셰이크만 확인한 가짜 양성' 전례 재발 방지).
- 스트림 stale(북 나이>30s) 상태의 leg는 비용을 지어내지 않고 aborted_stale로 폐기 기록.

이 파일은 라이브 봇(trading_bot.py 계열)과 완전히 독립이며 실주문 경로가 없다.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SYMBOL = os.environ.get("MAKER_SHADOW_SYMBOL", "ETHUSDT").upper()
DB_PATH = Path(os.environ.get("MAKER_SHADOW_DB_PATH", str(ROOT / "data/live/maker_fill_shadow.duckdb")))
SPACING_S = int(os.environ.get("MAKER_SHADOW_SPACING_S", "300"))
TIMEOUT_S = int(os.environ.get("MAKER_SHADOW_TIMEOUT_S", "120"))
POLICIES = [p.strip() for p in os.environ.get("MAKER_SHADOW_POLICIES", "peg,static").split(",") if p.strip()]
MAKER_FEE_BP = 2.0
TAKER_FEE_BP = 5.0
LATENCY_MS = 200
STALE_S = 30.0
HEARTBEAT_S = 300

# 2026-09-02: @aggTrade -> @trade. Binance USD-M stopped delivering @aggTrade -- the
# subscription is still ACCEPTED (SUBSCRIBE returns result:null, no error) but zero messages
# ever arrive, which is why this failed silently for 8+ days with trade_msgs=0 while
# book_msgs reached 293M. Verified 2026-09-02 by direct probe: ethusdt@aggTrade and
# btcusdt@aggTrade both deliver nothing, while ethusdt@trade delivers ~320 msg/s.
# @trade carries the same four fields this worker uses (p/q/m/T) and is FINER grained than
# aggTrade (individual fills rather than same-price aggregates), which is strictly better
# for the queue-consumption fill rule.
WS_URL = f"wss://fstream.binance.com/stream?streams={SYMBOL.lower()}@bookTicker/{SYMBOL.lower()}@trade"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("maker_fill_shadow")


@dataclass
class Book:
    bid: float = 0.0
    ask: float = 0.0
    bid_qty: float = 0.0
    ask_qty: float = 0.0
    ex_ts: int = 0
    local_ts: float = 0.0

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    def age_s(self) -> float:
        return time.time() - self.local_ts if self.local_ts else 1e9


@dataclass
class Leg:
    policy: str
    side: str            # "buy" / "sell"
    timeout_s: int
    arrival: Book
    my_px: float = 0.0
    queue: float = 0.0
    active_from_ms: int = 0
    repegs: int = 0
    done: bool = False
    result: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        buy = self.side == "buy"
        self.my_px = self.arrival.bid if buy else self.arrival.ask
        self.queue = self.arrival.bid_qty if buy else self.arrival.ask_qty
        self.active_from_ms = self.arrival.ex_ts + LATENCY_MS
        self.deadline_ms = self.arrival.ex_ts + self.timeout_s * 1000

    def _finish(self, filled: bool, mode: str, px: float, t_ms: int) -> None:
        buy = self.side == "buy"
        sgn = 1.0 if buy else -1.0
        price_bp = sgn * (px - self.arrival.mid) / self.arrival.mid * 1e4
        fee = MAKER_FEE_BP if filled else TAKER_FEE_BP
        self.done = True
        self.result = {"filled": filled, "fill_mode": mode, "fill_price": px,
                       "fill_t_ms": t_ms, "cost_bp": price_bp + fee}

    def on_trade(self, px: float, qty: float, is_buyer_maker: bool, ex_ts: int) -> None:
        if self.done or ex_ts < self.active_from_ms:
            return
        if ex_ts > self.deadline_ms:
            return
        buy = self.side == "buy"
        if buy:
            if is_buyer_maker:                      # 매도 aggressor(bid에서 체결)
                if px < self.my_px - 1e-9:
                    self._finish(True, "trade_through", self.my_px, ex_ts - self.arrival.ex_ts)
                elif abs(px - self.my_px) < 1e-9:
                    self.queue -= qty
                    if self.queue < -1e-9:
                        self._finish(True, "queue_exhaust", self.my_px, ex_ts - self.arrival.ex_ts)
            elif px <= self.my_px + 1e-9:           # ask측 체결이 내 bid 이하 = 크로스
                self._finish(True, "quote_cross_trade", self.my_px, ex_ts - self.arrival.ex_ts)
        else:
            if not is_buyer_maker:                  # 매수 aggressor(ask에서 체결)
                if px > self.my_px + 1e-9:
                    self._finish(True, "trade_through", self.my_px, ex_ts - self.arrival.ex_ts)
                elif abs(px - self.my_px) < 1e-9:
                    self.queue -= qty
                    if self.queue < -1e-9:
                        self._finish(True, "queue_exhaust", self.my_px, ex_ts - self.arrival.ex_ts)
            elif px >= self.my_px - 1e-9:
                self._finish(True, "quote_cross_trade", self.my_px, ex_ts - self.arrival.ex_ts)

    def on_book(self, book: Book) -> None:
        if self.done or book.ex_ts < self.active_from_ms:
            return
        buy = self.side == "buy"
        if buy and book.ask <= self.my_px + 1e-9:
            self._finish(True, "quote_cross", self.my_px, book.ex_ts - self.arrival.ex_ts)
            return
        if not buy and book.bid >= self.my_px - 1e-9:
            self._finish(True, "quote_cross", self.my_px, book.ex_ts - self.arrival.ex_ts)
            return
        if self.policy == "peg":
            if buy and book.bid > self.my_px + 1e-9:
                self.my_px = book.bid
                self.queue = book.bid_qty
                self.active_from_ms = book.ex_ts + LATENCY_MS
                self.repegs += 1
            elif not buy and book.ask < self.my_px - 1e-9:
                self.my_px = book.ask
                self.queue = book.ask_qty
                self.active_from_ms = book.ex_ts + LATENCY_MS
                self.repegs += 1

    def check_timeout(self, book: Book) -> None:
        if self.done:
            return
        now_ms = int(time.time() * 1000)
        if book.ex_ts >= self.deadline_ms or now_ms - LATENCY_MS > self.deadline_ms + 2000:
            if book.age_s() > STALE_S:
                self.done = True
                self.result = {"filled": False, "fill_mode": "aborted_stale", "fill_price": None,
                               "fill_t_ms": self.timeout_s * 1000, "cost_bp": None}
                return
            fb = book.ask if self.side == "buy" else book.bid
            self._finish(False, "taker_fallback", fb, self.timeout_s * 1000)


class Worker:
    def __init__(self) -> None:
        self.book = Book()
        self.legs: list[Leg] = []
        self.book_msgs = 0
        self.trade_msgs = 0
        self.legs_done = 0
        self._init_db()

    def _init_db(self) -> None:
        import duckdb
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(DB_PATH))
        self.con.execute("""
            create table if not exists maker_fill_shadow_legs(
              recorded_at_utc timestamp, symbol varchar, policy varchar, timeout_s integer,
              side varchar, arrival_ex_ts bigint, arrival_bid double, arrival_ask double,
              arrival_mid double, arrival_bid_qty double, arrival_ask_qty double,
              spread_bp double, filled boolean, fill_mode varchar, fill_price double,
              fill_t_ms bigint, repegs integer, cost_bp double,
              maker_fee_bp double, taker_fee_bp double)""")
        self.con.execute("""
            create table if not exists maker_fill_shadow_heartbeat(
              recorded_at_utc timestamp, symbol varchar, book_msgs bigint, trade_msgs bigint,
              legs_done bigint, legs_active integer, book_age_s double)""")
        # trigger 컬럼은 결정시점 arm(2026-08-24~09-15) 잔재다. 과거 행에 'decision'이 남아
        # 있으므로 컬럼과 값 기록은 유지하고, 새 행은 항상 'schedule'이다.
        existing = {c[1] for c in self.con.execute("PRAGMA table_info('maker_fill_shadow_legs')").fetchall()}
        if "trigger" not in existing:
            self.con.execute("ALTER TABLE maker_fill_shadow_legs ADD COLUMN trigger VARCHAR")
        self.con.execute("UPDATE maker_fill_shadow_legs SET trigger='schedule' WHERE trigger IS NULL")

    def write_leg(self, leg: Leg) -> None:
        r = leg.result
        a = leg.arrival
        self.con.execute(
            """insert into maker_fill_shadow_legs (
                 recorded_at_utc, symbol, policy, timeout_s, side, arrival_ex_ts,
                 arrival_bid, arrival_ask, arrival_mid, arrival_bid_qty, arrival_ask_qty,
                 spread_bp, filled, fill_mode, fill_price, fill_t_ms, repegs, cost_bp,
                 maker_fee_bp, taker_fee_bp, trigger
               ) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            [datetime.now(timezone.utc), SYMBOL, leg.policy, leg.timeout_s, leg.side,
             a.ex_ts, a.bid, a.ask, a.mid, a.bid_qty, a.ask_qty,
             (a.ask - a.bid) / a.mid * 1e4, r["filled"], r["fill_mode"], r["fill_price"],
             r["fill_t_ms"], leg.repegs, r["cost_bp"], MAKER_FEE_BP, TAKER_FEE_BP, "schedule"])
        self.legs_done += 1
        logger.info("leg done: %s %s %s cost=%s mode=%s t=%sms repegs=%d",
                    leg.policy, leg.side, f"T{leg.timeout_s}",
                    f"{r['cost_bp']:.2f}bp" if r["cost_bp"] is not None else "NA",
                    r["fill_mode"], r["fill_t_ms"], leg.repegs)

    def heartbeat(self) -> None:
        self.con.execute("insert into maker_fill_shadow_heartbeat values (?,?,?,?,?,?,?)",
                         [datetime.now(timezone.utc), SYMBOL, self.book_msgs, self.trade_msgs,
                          self.legs_done, len(self.legs), self.book.age_s()])
        logger.info("heartbeat: book_msgs=%d trade_msgs=%d legs_done=%d active=%d book_age=%.1fs",
                    self.book_msgs, self.trade_msgs, self.legs_done, len(self.legs), self.book.age_s())
        # 2026-09-02: the @aggTrade outage was invisible for 8+ days because a dead trade stream
        # degrades silently -- fills simply fall back to quote_cross only, and every number the
        # worker reports still looks plausible. Make that state LOUD instead.
        if self.book_msgs > 100_000 and self.trade_msgs == 0:
            logger.error("TRADE STREAM DEAD: book_msgs=%d but trade_msgs=0 -- fill rule is running "
                         "on quote_cross only, cost figures are a biased subsample. Check the "
                         "stream name in WS_URL.", self.book_msgs)

    def _sweep(self) -> None:
        for leg in self.legs:
            leg.check_timeout(self.book)
        for leg in [l for l in self.legs if l.done]:
            self.write_leg(leg)
        self.legs = [l for l in self.legs if not l.done]

    async def ws_loop(self) -> None:
        import websockets
        delay = 3.0
        while True:
            try:
                async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("WS connected: %s", WS_URL)
                    delay = 3.0
                    async for raw in ws:
                        msg = json.loads(raw)
                        data = msg.get("data", {})
                        ev = data.get("e", "bookTicker" if "b" in data and "a" in data else "")
                        if ev == "bookTicker" or ("b" in data and "B" in data and "a" in data):
                            self.book = Book(bid=float(data["b"]), ask=float(data["a"]),
                                             bid_qty=float(data["B"]), ask_qty=float(data["A"]),
                                             ex_ts=int(data.get("T") or data.get("E") or time.time() * 1000),
                                             local_ts=time.time())
                            self.book_msgs += 1
                            for leg in self.legs:
                                leg.on_book(self.book)
                        elif ev in ("trade", "aggTrade"):   # aggTrade kept so a restored stream still routes
                            self.trade_msgs += 1
                            px = float(data["p"]); qty = float(data["q"])
                            bm = bool(data["m"]); ex_ts = int(data["T"])
                            for leg in self.legs:
                                leg.on_trade(px, qty, bm, ex_ts)
                        self._sweep()
            except Exception as e:  # noqa: BLE001 — 재연결 루프(기존 수집기 패턴)
                logger.warning("WS error: %r — reconnect in %.0fs", e, delay)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 60.0)

    async def scheduler_loop(self) -> None:
        await asyncio.sleep(5)
        while True:
            if self.book.age_s() < 5.0 and self.book.bid > 0:
                arrival = Book(**vars(self.book))
                for policy in POLICIES:
                    for side in ("buy", "sell"):
                        self.legs.append(Leg(policy=policy, side=side, timeout_s=TIMEOUT_S, arrival=arrival))
                logger.info("spawned %d legs @ bid=%.2f ask=%.2f", 2 * len(POLICIES), arrival.bid, arrival.ask)
            else:
                logger.warning("book stale (age=%.1fs) — skip arrival", self.book.age_s())
            await asyncio.sleep(SPACING_S)

    async def ticker_loop(self) -> None:
        while True:
            self._sweep()
            await asyncio.sleep(1.0)

    async def heartbeat_loop(self) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_S)
            self.heartbeat()

    async def run(self) -> None:
        logger.info("maker_fill_shadow start: symbol=%s spacing=%ds timeout=%ds policies=%s db=%s",
                    SYMBOL, SPACING_S, TIMEOUT_S, POLICIES, DB_PATH)
        await asyncio.gather(self.ws_loop(), self.scheduler_loop(),
                             self.ticker_loop(), self.heartbeat_loop())


if __name__ == "__main__":
    asyncio.run(Worker().run())
