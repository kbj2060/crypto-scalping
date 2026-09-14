#!/usr/bin/env python3
"""**최우선 호가(bookTicker) 수집기** — micro-price 와 OFI 의 유일한 정확한 원천. (2026-09-14)

왜 별도로 띄우는가: 같은 날 호가 래스터($0.50 빈)로 micro-price 를 재려다 **인공물**을 잡았다.
ETH 틱은 $0.01 인데 래스터는 빈이 $0.50 이라 50배 거칠고, 빈에서 복원한 최우선 호가의 중점과
진짜 mid 사이 반올림 오차가 「mid 가 빈 안 어디에 있나」를 만들어 1초 IC +0.229 로 나타났다.
진짜 큐 불균형은 −0.029 였다. 항등식 `(MP−mid)/spread = QI/2 + 빈격자잔차` 로 분해해 확인했다.
⇒ 문헌(Cont·Kukanov·Stoikov 2013 · Stoikov 2018)이 요구하는 재료는 **최우선 호가의 가격과 수량**
   그 자체다. `@bookTicker` 는 그것만 보내고, **최우선이 바뀔 때마다** 보낸다(스냅샷 폴링 아님).

트레이딩 봇과 완전 분리: 자기 WS · 자기 디렉터리 · 죽어도 봇에 영향 없다. 주문은 내지 않는다.
(래스터 수집기 도크스트링의 규약을 그대로 따른다.)

저장 포맷 — 래스터와 같은 이유로 고정폭 플랫파일(duckdb 는 DELETE 해도 파일이 안 줄어든다):
  <BT_ROOT>/<SYMBOL>/<YYYY-MM-DDTHH>.bt   (UTC 시각별)
  헤더 32B: magic"BTKR" u16 ver, u16 rowsz, i64 hour_start_ms, u64 first_update_id, u32 flags, u32 rsv
  행 32B: i64 ts_ms(거래소 T), f64 bid_px, f32 bid_qty, f64 ask_px, f32 ask_qty
  ⭐행이 **불규칙 시각**이다(최우선이 바뀔 때만). 초 격자로 맞추는 건 분석 쪽 일이다 --
    수집기가 리샘플하면 그 순간 원본이 사라진다.
  ⭐`u`(update id)로 순서를 검증한다. 역행/중복은 버리고 카운터만 올린다.

🔴용량 — **실측하고 설계를 바꿨다**: 처음엔 「초당 수십 건」으로 잡았는데 2026-09-14 실측이
  **1,129행/초**(22배)였다. 원본 그대로면 **3.1GB/일**이고 서버 여유 547GB 로 176일뿐이다.
  그래서 **시각이 바뀔 때 지난 파일을 gzip 한다**(별도 스레드). 고정폭에 가격이 반복되므로
  3~5배 줄어 ~0.8GB/일이 되고, 현재 시각 파일만 비압축이라 실시간 읽기는 그대로다.
  ⚠️추정은 설계를 정하고 실측은 설계를 바꾼다 — 이 주석이 그 기록이다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import gzip
import shutil
import struct
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("BT_SYMBOL", "ethusdt").lower()
BT_ROOT = Path(os.getenv("BT_ROOT", str(ROOT / "data" / "live" / "orderflow" / "bookticker")))
_WS_URL = "wss://fstream.binance.com/ws/{symbol}@bookTicker"

MAGIC = b"BTKR"
VER = 1
HDR = struct.Struct("<4sHHqQII")          # 32B
ROW = struct.Struct("<qdfdf")             # 32B: ts_ms, bid_px, bid_qty, ask_px, ask_qty
assert HDR.size == 32 and ROW.size == 32

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("bookticker")


def _gzip_done(path: Path) -> None:
    """끝난 시각 파일을 압축한다. 🔴원본은 **압축이 성공한 뒤에만** 지운다 -- 중간에 죽으면
    둘 다 남는 게 낫지, 원본만 사라지는 건 영원히 못 되돌린다."""
    gz = path.with_suffix(path.suffix + ".gz")
    try:
        with open(path, "rb") as fi, gzip.open(gz, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)
        if gz.stat().st_size > 0:
            before, after = path.stat().st_size, gz.stat().st_size
            path.unlink()
            log.info("압축 %s · %.1fMB → %.1fMB (%.1f배)", path.name, before / 1e6, after / 1e6,
                     before / max(after, 1))
    except Exception as exc:
        log.warning("압축 실패 %s: %s — 원본 유지", path.name, exc)


class HourFile:
    """시각별 파일 하나. 열릴 때 헤더를 쓰고, 시각이 바뀌면 닫고 새로 연다."""

    def __init__(self, root: Path, symbol: str):
        self.dir = root / symbol.upper()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hour: str | None = None
        self.fh = None
        self.rows = 0

    def _path(self, ts_ms: int) -> tuple[str, Path]:
        dt = datetime.fromtimestamp(ts_ms / 1000, timezone.utc)
        key = dt.strftime("%Y-%m-%dT%H")
        return key, self.dir / f"{key}.bt"

    def write(self, ts_ms: int, bid: float, bq: float, ask: float, aq: float,
              first_u: int) -> None:
        key, path = self._path(ts_ms)
        if key != self.hour:
            self.close(compress=True)
            new = not path.exists()
            self.fh = open(path, "ab")
            if new:
                hour_ms = int(datetime.strptime(key, "%Y-%m-%dT%H")
                              .replace(tzinfo=timezone.utc).timestamp() * 1000)
                self.fh.write(HDR.pack(MAGIC, VER, ROW.size, hour_ms, first_u, 0, 0))
            self.hour, self.rows = key, 0
            log.info("파일 전환 %s", path.name)
        self.fh.write(ROW.pack(ts_ms, bid, bq, ask, aq))
        self.rows += 1
        if self.rows % 2000 == 0:          # 크래시 때 잃는 양을 2,000행(≈40초)으로 묶는다
            self.fh.flush()

    def close(self, compress: bool = False) -> None:
        path = self.dir / f"{self.hour}.bt" if self.hour else None
        if self.fh:
            try:
                self.fh.flush(); self.fh.close()
            except OSError:
                pass
        self.fh = None
        if compress and path and path.exists():
            threading.Thread(target=_gzip_done, args=(path,), daemon=True).start()


async def run(symbol: str, root: Path) -> None:
    import websockets
    url = _WS_URL.format(symbol=symbol)
    out = HourFile(root, symbol)
    last_u = -1
    n = dropped = 0
    t0 = time.time()
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                log.info("연결 %s", url)
                async for raw in ws:
                    d = json.loads(raw)
                    u = int(d.get("u", 0))
                    if u <= last_u:            # 역행/중복 — 버린다(순서는 u 가 보장)
                        dropped += 1
                        continue
                    last_u = u
                    ts = int(d.get("T") or d.get("E") or time.time() * 1000)
                    out.write(ts, float(d["b"]), float(d["B"]), float(d["a"]), float(d["A"]), u)
                    n += 1
                    if n % 20000 == 0:
                        el = time.time() - t0
                        log.info("%s 누적 %d행 · %.1f행/초 · 버림 %d · 스프레드 %.2f",
                                 symbol.upper(), n, n / max(el, 1e-9), dropped,
                                 float(d["a"]) - float(d["b"]))
        except asyncio.CancelledError:
            raise
        except Exception as exc:               # 끊기면 다시 붙는다 — 빈 구간은 영원히 못 채운다
            log.warning("WS 끊김 %s: %s — 3초 뒤 재연결", type(exc).__name__, exc)
            out.close()
            await asyncio.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Binance 무기한선물 최우선 호가 수집기")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--root", default=str(BT_ROOT))
    a = ap.parse_args()
    log.info("수집 시작 %s → %s", a.symbol.upper(), a.root)
    try:
        asyncio.run(run(a.symbol.lower(), Path(a.root)))
    except KeyboardInterrupt:
        log.info("종료")


if __name__ == "__main__":
    main()
