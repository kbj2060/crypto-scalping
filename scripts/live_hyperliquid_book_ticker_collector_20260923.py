#!/usr/bin/env python3
"""**하이퍼리퀴드 최우선 호가 + 자산 컨텍스트 수집기** — DEX 쪽 호가/OI. (2026-09-23)

왜: `live_hyperliquid_trade_collector_20260916.py` 가 **체결만** 받고 있다(주소단위 -- 그게 그
수집기의 존재 이유였다). 그래서 이 저장소는 DEX 의 호가·OI·펀딩을 하나도 안 들고 있다.
하이퍼리퀴드는 ETH 테이커 플로우의 **8.3%**($2.1B/일, 2026-09-22 실측)로 DEX 중 압도적이고
(Lighter 의 5.5배, 코인베이스 현물보다도 크다), 호가는 체결과 달리 **소급 재구성이 불가능**하다.

🔴**여기서 깊은 북은 받을 수 없다 -- 거래소의 한계다.** `l2Book` 을 두 번 독립 측정했고
  (기본 / `nSigFigs=5`) 둘 다 **60초에 13~14건, 중앙 간격 5.4초**였다. 같은 연결에서 `bbo` 는
  8.8/s 로 오므로 북은 그보다 훨씬 자주 변한다 -- 즉 `l2Book` 스트림 자체가 샘플링된 것이다.
  게다가 20레벨 고정(집계)이라 벽을 볼 수 없다. **하이퍼리퀴드로는 흡수/호가벽 축을 못 한다.**
  이 수집기는 «최우선 호가 + 컨텍스트»까지가 천장이고, 그 이상은 이 거래소에 없다.

⭐**`sz` 는 ETH 직접이다**(OKX 처럼 계약 수가 아니다 -- 실측 `{"px":"2747.4","sz":"174.0396","n":13}`,
  $478k 에 해당해 크기가 말이 된다). ctVal 환산이 **없다** -- 넣으면 그게 버그다.
⭐파일 포맷은 바이낸스/OKX 와 완전히 같다(magic `BTKR` · 32B 헤더 · 32B 행). `HourFile` 을
  복사하지 않고 **바이낸스 수집기에서 import** 한다. 거래소 구분은 디렉터리다.

🔴**`bbo` 에는 seqId 가 없다.** 순서는 `time`(ms)뿐이고 같은 ms 에 여러 갱신이 온다. 그래서
  «역행이면 버린다」는 되지만 «같은 ms 면 버린다」는 안 된다 -- 진짜 갱신을 지운다. 완전히
  같은 (time, bid, bidsz, ask, asksz) 일 때만 중복으로 본다.
🔴**`activeAssetCtx` 에는 타임스탬프가 아예 없다.** 수신 시각을 찍되 컬럼 이름을 `recv_ms` 로
  두고 meta 에 명시한다 -- 거래소 시각인 척하면 나중에 지연 분석이 통째로 거짓이 된다.
  (`bbo` 의 `time` 은 거래소 시각이라 `.bt` 행에는 그것을 쓴다.)

용량: bbo 8.8/s = 24MB/일(원본), 시각 회전 때 gzip 하므로 ~7MB/일. ctx 1/s = ~3MB/일.

봇과 완전 분리: 자기 WS · 자기 디렉터리 · 주문 없음. 기존 체결 수집기와 **다른 프로세스**다
(그쪽은 자기 duckdb 를 쓰고, duckdb 는 writer 가 하나다).

사용:
  python scripts/live_hyperliquid_book_ticker_collector_20260923.py            # ETH
  HL_BT_COIN=BTC python scripts/live_hyperliquid_book_ticker_collector_20260923.py
  python scripts/live_hyperliquid_book_ticker_collector_20260923.py --selftest # 네트워크 없이
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

_bn = importlib.import_module("scripts.live_book_ticker_collector_20260914")
HourFile = _bn.HourFile
MAGIC, VER, HDR, ROW = _bn.MAGIC, _bn.VER, _bn.HDR, _bn.ROW

COIN = os.getenv("HL_BT_COIN", "ETH").upper()
BT_ROOT = Path(os.getenv("HL_BT_ROOT",
                         str(ROOT / "data" / "live" / "orderflow" / "hyperliquid_bookticker")))
CTX_DB = Path(os.getenv("HL_CTX_DB_PATH", str(ROOT / "data" / "live" / "hyperliquid_context.duckdb")))
WS_URL = "wss://api.hyperliquid.xyz/ws"
RECV_TIMEOUT = 30.0      # 이만큼 조용하면 ping 을 보낸다(끊지 않는다 -- 재연결이 더 비싸다)
CTX_FLUSH_SECONDS = 30.0

CTX_DDL = """CREATE TABLE IF NOT EXISTS hl_asset_ctx(
    coin VARCHAR, recv_ms BIGINT, funding DOUBLE, open_interest DOUBLE, premium DOUBLE,
    oracle_px DOUBLE, mark_px DOUBLE, mid_px DOUBLE, day_ntl_vlm DOUBLE, day_base_vlm DOUBLE)"""
CTX_WIDTH = 10

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hl_bookticker")


def parse_bbo(data: dict) -> tuple[int, float, float, float, float] | None:
    """`bbo` → (ts_ms, bid_px, bid_sz, ask_px, ask_sz). 🔴sz 는 이미 ETH 다."""
    lv = data.get("bbo")
    if not isinstance(lv, list) or len(lv) != 2 or not (lv[0] and lv[1]):
        return None                      # 한쪽이 비면(null) 버린다 -- HL 은 실제로 null 을 보낸다
    try:
        ts = int(data["time"])
        bid_px, bid_sz = float(lv[0]["px"]), float(lv[0]["sz"])
        ask_px, ask_sz = float(lv[1]["px"]), float(lv[1]["sz"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (ts > 0 and bid_px > 0 and ask_px > bid_px and bid_sz > 0 and ask_sz > 0):
        return None
    return ts, bid_px, bid_sz, ask_px, ask_sz


def parse_ctx(data: dict, recv_ms: int) -> tuple | None:
    """`activeAssetCtx` → 행 하나. 🔴거래소가 시각을 안 주므로 `recv_ms` 는 **우리 시계**다."""
    ctx = data.get("ctx")
    if not isinstance(ctx, dict):
        return None
    def g(key):
        try:
            return float(ctx[key])
        except (KeyError, TypeError, ValueError):
            return None
    oi = g("openInterest")
    if oi is None:
        return None                      # OI 가 이 표의 존재 이유다 -- 없으면 적을 값이 없다
    return (str(data.get("coin", "")), recv_ms, g("funding"), oi, g("premium"),
            g("oraclePx"), g("markPx"), g("midPx"), g("dayNtlVlm"), g("dayBaseVlm"))


class CtxStore:
    """쓸 때만 열고 닫는다(저장소 규약). 못 쓴 행은 버리지 않고 다음 주기에 재시도한다."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.pending: list[tuple] = []
        with self._connect() as con:
            con.execute(CTX_DDL)
            con.execute("CREATE TABLE IF NOT EXISTS meta(key VARCHAR, value VARCHAR)")
            for key, value in (("recv_ms", "🔴우리 시계다 -- activeAssetCtx 에는 거래소 시각이 없다"),
                               ("sz_unit", "ETH (계약 수가 아니다)"),
                               ("l2book_note", "l2Book 은 중앙 5.4초 간격이라 받지 않는다")):
                con.execute("DELETE FROM meta WHERE key = ?", [key])
                con.execute("INSERT INTO meta VALUES (?, ?)", [key, value])

    def _connect(self):
        import duckdb

        return duckdb.connect(str(self.db_path))

    def write(self, rows: list[tuple]) -> None:
        self.pending.extend(rows)
        if not self.pending:
            return
        try:
            marks = ",".join("?" * CTX_WIDTH)
            with self._connect() as con:
                con.begin()      # 🔴자동커밋이면 행마다 fsync 다(체결 테이프 TapeStore.write 참고)
                con.executemany(f"INSERT INTO hl_asset_ctx VALUES ({marks})", self.pending)
                con.commit()
            self.pending.clear()
        except Exception as exc:  # noqa: BLE001 -- 대개 읽는 쪽이 잡고 있는 락이다
            if len(self.pending) > 50_000:
                self.pending = self.pending[len(self.pending) // 2:]
                log.warning("쓰기가 계속 막혀 절반 버림")
            else:
                log.info("ctx 쓰기 보류 %d행: %s", len(self.pending), type(exc).__name__)


async def run(coin: str, root: Path, ctx_db: Path) -> None:
    from aiohttp import ClientSession, ClientTimeout, WSMsgType

    out = HourFile(root, coin)
    store = CtxStore(ctx_db)
    ctx_buf: list[tuple] = []
    last_ts = -1
    last_row: tuple | None = None
    n = dropped = n_ctx = 0
    t0 = time.time()
    log.info("수집 시작 %s → %s · ctx → %s", coin, root, ctx_db)
    async with ClientSession(timeout=ClientTimeout(total=None)) as session:
        flushed_at = time.monotonic()
        while True:
            try:
                async with session.ws_connect(WS_URL, heartbeat=25, max_msg_size=0) as ws:
                    for kind in ("bbo", "activeAssetCtx"):
                        await ws.send_json({"method": "subscribe",
                                            "subscription": {"type": kind, "coin": coin}})
                    log.info("연결 %s (%s)", WS_URL, coin)
                    while True:
                        try:
                            msg = await ws.receive(timeout=RECV_TIMEOUT)
                        except asyncio.TimeoutError:
                            await ws.send_json({"method": "ping"})   # 죽었으면 여기서 터진다
                            continue
                        if msg.type is not WSMsgType.TEXT:
                            break
                        try:
                            payload = json.loads(msg.data)
                        except ValueError:
                            continue
                        channel = payload.get("channel")
                        if channel == "bbo":
                            parsed = parse_bbo(payload.get("data") or {})
                            if parsed is None:
                                dropped += 1
                                continue
                            if parsed[0] < last_ts or parsed == last_row:
                                dropped += 1      # 역행 또는 완전 중복. 같은 ms 자체는 정상이다
                                continue
                            last_ts, last_row = parsed[0], parsed
                            out.write(*parsed, last_ts)
                            n += 1
                            if n % 20000 == 0:
                                el = time.time() - t0
                                log.info("%s 누적 %d행 · %.1f행/초 · 버림 %d · ctx %d",
                                         coin, n, n / max(el, 1e-9), dropped, n_ctx)
                        elif channel == "activeAssetCtx":
                            row = parse_ctx(payload.get("data") or {}, int(time.time() * 1000))
                            if row is not None:
                                ctx_buf.append(row)
                                n_ctx += 1
                        now = time.monotonic()
                        if now - flushed_at >= CTX_FLUSH_SECONDS:
                            flushed_at = now
                            await asyncio.to_thread(store.write, ctx_buf)  # 스레드로: close() 체크포인트 fsync(~0.5초)가 WS 수신을 막지 않게
                            ctx_buf = []
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 -- 빈 구간은 영원히 못 채운다. 다시 붙는다.
                log.warning("WS 끊김 %s: %s — 3초 뒤 재연결", type(exc).__name__, exc)
            out.close()
            await asyncio.to_thread(store.write, ctx_buf)
            ctx_buf = []
            await asyncio.sleep(3)


def selftest() -> None:
    """네트워크·DB 없이 파싱과 바이트 왕복을 점검한다. 페이로드는 2026-09-23 실측 원문이다."""
    import struct
    import tempfile

    data = {"coin": "ETH", "time": 1790093073473,
            "bbo": [{"px": "2747.4", "sz": "174.0396", "n": 13},
                    {"px": "2747.5", "sz": "107.5582", "n": 16}]}
    p = parse_bbo(data)
    assert p == (1790093073473, 2747.4, 174.0396, 2747.5, 107.5582), p
    # 🔴sz 를 ctVal 로 곱하면 안 된다 -- OKX 와 다르다. 값이 그대로여야 한다.
    assert p[2] == float(data["bbo"][0]["sz"]), "HL sz 는 이미 ETH 다"
    assert parse_bbo({**data, "bbo": [None, data["bbo"][1]]}) is None, "한쪽 null 은 버린다"
    assert parse_bbo({**data, "bbo": [data["bbo"][1], data["bbo"][0]]}) is None, "교차는 버린다"
    assert parse_bbo({"time": 1, "bbo": []}) is None
    assert parse_bbo({}) is None

    ctx = {"coin": "ETH", "ctx": {"funding": "0.0000125", "openInterest": "1144067.9077999997",
                                  "prevDayPx": "2758.6", "dayNtlVlm": "1344109438.2353596687",
                                  "premium": "0.0004733986", "oraclePx": "2746.1",
                                  "markPx": "2747.4", "midPx": "2747.45",
                                  "impactPxs": ["2747.4", "2747.5"],
                                  "dayBaseVlm": "487476.9786999995"}}
    r = parse_ctx(ctx, 1790093073999)
    assert len(r) == CTX_WIDTH, (len(r), CTX_WIDTH)
    assert r[:5] == ("ETH", 1790093073999, 0.0000125, 1144067.9077999997, 0.0004733986), r
    assert r[5:8] == (2746.1, 2747.4, 2747.45), r
    assert parse_ctx({"coin": "ETH", "ctx": {"markPx": "1"}}, 1) is None, "OI 가 없으면 안 적는다"
    assert parse_ctx({"coin": "ETH"}, 1) is None
    # 빠진 칸은 0 이 아니라 None 이어야 한다(「0이었다」와 「안 왔다」는 다른 말이다)
    partial = parse_ctx({"coin": "ETH", "ctx": {"openInterest": "5", "markPx": "2"}}, 1)
    assert partial[2] is None and partial[5] is None and partial[3] == 5.0, partial

    # 바이트 왕복 — 기존 리더(read_bt)의 구조체로 되읽는다
    BT_HDR, BT_ROW = struct.Struct("<4sHHqQII"), struct.Struct("<qdfdf")
    assert (BT_HDR.size, BT_ROW.size) == (HDR.size, ROW.size) == (32, 32)
    with tempfile.TemporaryDirectory() as td:
        hf = HourFile(Path(td), "ETH")
        hf.write(*p, p[0])
        hf.close()
        f = sorted(Path(td).glob("ETH/*.bt"))[0]
        raw = f.read_bytes()
        magic, ver, rowsz, hour_ms, first_u, _flags, _ = BT_HDR.unpack_from(raw, 0)
        assert magic == b"BTKR" and rowsz == 32 and ver == VER, (magic, rowsz, ver)
        assert hour_ms % 3_600_000 == 0 and hour_ms <= p[0] < hour_ms + 3_600_000
        assert first_u == p[0], "seqId 가 없으므로 그 자리에 time 을 넣는다"
        r0 = BT_ROW.unpack_from(raw, 32)
        assert r0[0] == p[0] and r0[1] == p[1] and r0[3] == p[3], r0
        assert abs(r0[2] - p[2]) < p[2] * 1e-6 and abs(r0[4] - p[4]) < p[4] * 1e-6, r0
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coin", default=COIN)
    ap.add_argument("--root", default=str(BT_ROOT))
    ap.add_argument("--ctx-db", type=Path, default=CTX_DB)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    try:
        asyncio.run(run(a.coin.upper(), Path(a.root), a.ctx_db))
    except KeyboardInterrupt:
        log.info("종료")


if __name__ == "__main__":
    main()
