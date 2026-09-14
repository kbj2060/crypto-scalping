#!/usr/bin/env python3
"""**주문단위 호가 변경(diff depth) 수집기** — 「달력 대기」를 오늘 시작한다. (2026-09-15)

왜: 이 저장소의 방향 기각 **넷**이 스스로 같은 재개 조건을 적어놨다 — *주문단위 L2 · 달력 대기*
(`eth_breakout_reversal_3axis_cvd_oi_book_closed_20260908` · `eth_direction_label_rebuild_closed_20260914`
 · `eth_scale_in_axis_closed_six_variants_20260914` · `eth_entry_layer_reconstruction_gates_built_20260903`).
그런데 **아무도 그 달력을 시작하지 않았다.** 연속 수집기가 없으면 1년을 기다려도 데이터는 없다.

09-08 이 명시한 원리적 한계: **30초 bookDepth 스냅샷은 스푸핑을 못 본다**(제출→취소가 초 단위).
기존 원천 셋 중 「연속 + 깊이 + 서브초」를 동시에 주는 게 하나도 없다:
  bookTicker(연속·서브초·**최우선만**) · 래스터(연속·깊음·**1초**·$0.50빈) · L2이상(깊음·0.1초·**383창뿐**)
`@depth@100ms` 가 그 빈칸이다 — 전체 북의 **모든 변경**을 100ms 마다.

🔴용량 — **두 번 실측했고 두 번째가 첫 번째를 뒤집었다**(추정하면 틀린다, bookTicker 때 22배 빗나갔다):
  스트림 비교(25초): `@depth@100ms` 7.8건/초·레벨 중앙 76/최대 773 vs `depth20@100ms` 7.6건/초·40고정.
  depth20 은 싸지만 ETH 틱이 $0.01 이라 **$0.20 밴드**뿐이라 벽을 못 본다 ⇒ diff 전체북을 쓴다.
  실파일 50초 실측: **9.8건/초 · 2,259MB/일(원본)** 인데 **gzip 압축률이 6.9배가 아니라 4.3배**다
  (JSON 이라 고정폭보다 덜 눌린다) ⇒ **약 524MB/일**. 서버 여유 547GB 기준 약 **2.8년치**.
  ⚠️처음 도크스트링에 152MB/일이라 적었다가 실파일로 재서 고쳤다 — 스트림 프로브의 바이트수와
    실제 저장 바이트수는 다르다. 고정폭으로 바꾸면 ~350MB/일이 되겠지만 674k행/일에 그 복잡도는
    안 산다.

⭐포맷은 **JSONL + 시각별 gzip**이다. bookTicker 는 5,800만행/일이라 고정폭 바이너리가 필요했지만
  여기는 **674k행/일**(87배 적다) — 포맷 복잡도가 정당화되지 않는다. 거래소 페이로드를 **그대로**
  적는다. 수집기가 해석하면 그 순간 원본이 사라진다.

🔴diff 는 증분이라 그것만으론 절대 깊이를 복원할 수 없다. **시각마다 REST 스냅샷 1건**을 파일
  첫 줄에 넣는다(`{"_snapshot":...}`). 이게 없으면 최초 스냅샷부터 전부 체이닝해야 하고 한 번의
  구멍이 **이후 전부를 영구히** 못 쓰게 만든다. 시각당 REST 1회는 무시할 만한 비용이다.
  `pu`(previous update id) 로 연속성을 검증한다 — 끊기면 카운터만 올리고 계속 적는다(구멍을
  **기록**하는 게 조용히 이어붙이는 것보다 낫다).

봇과 완전 분리: 자기 WS · 자기 디렉터리 · 주문 없음. supervisor 는 bookTicker 규약을 그대로 따른다.
"""
from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import os
import shutil
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SYMBOL = os.getenv("DD_SYMBOL", "ethusdt").lower()
DD_ROOT = Path(os.getenv("DD_ROOT", str(ROOT / "data" / "live" / "orderflow" / "depthdiff")))
_WS = "wss://fstream.binance.com/ws/{symbol}@depth@100ms"
_REST = "https://fapi.binance.com/fapi/v1/depth?symbol={SYM}&limit=1000"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("depthdiff")


def _gzip_done(path: Path) -> None:
    """끝난 시각 파일을 압축한다. 🔴원본은 **압축 성공 뒤에만** 지운다."""
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


def rest_snapshot(symbol: str) -> dict | None:
    """절대 깊이의 기준점. 실패해도 수집은 계속한다 — diff 는 diff 대로 값이 있다."""
    try:
        with urllib.request.urlopen(_REST.format(SYM=symbol.upper()), timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:
        log.warning("스냅샷 실패: %s — diff 만 적는다", exc)
        return None


class HourFile:
    def __init__(self, root: Path, symbol: str):
        self.dir = root / symbol.upper()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.symbol = symbol
        self.hour: str | None = None
        self.fh = None
        self.rows = 0

    def write(self, ts_ms: int, line: str) -> None:
        key = datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H")
        if key != self.hour:
            self.close(compress=True)
            path = self.dir / f"{key}.jsonl"
            new = not path.exists()
            self.fh = open(path, "a")
            if new:
                snap = rest_snapshot(self.symbol)
                self.fh.write(json.dumps({"_snapshot": snap, "at_ms": int(time.time() * 1000)},
                                         separators=(",", ":")) + "\n")
            self.hour, self.rows = key, 0
            log.info("파일 전환 %s (스냅샷 %s)", path.name, "있음" if new else "이어쓰기")
        self.fh.write(line + "\n")
        self.rows += 1
        if self.rows % 300 == 0:              # 크래시 때 잃는 양을 300행(≈40초)으로 묶는다
            self.fh.flush()

    def close(self, compress: bool = False) -> None:
        path = self.dir / f"{self.hour}.jsonl" if self.hour else None
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
    out = HourFile(root, symbol)
    prev_u = None
    n = gaps = 0
    t0 = time.time()
    while True:
        try:
            async with websockets.connect(_WS.format(symbol=symbol),
                                          ping_interval=20, ping_timeout=10) as ws:
                log.info("연결 %s@depth@100ms", symbol.upper())
                prev_u = None
                async for raw in ws:
                    d = json.loads(raw)
                    ts = int(d.get("T") or d.get("E") or time.time() * 1000)
                    if prev_u is not None and d.get("pu") != prev_u:
                        gaps += 1                 # 구멍은 숨기지 않고 센다 + 파일에 표시
                        out.write(ts, json.dumps({"_gap": True, "expected_pu": prev_u,
                                                  "got_pu": d.get("pu")}, separators=(",", ":")))
                    prev_u = d.get("u")
                    out.write(ts, raw if isinstance(raw, str) else raw.decode())
                    n += 1
                    if n % 5000 == 0:
                        el = time.time() - t0
                        log.info("%s 누적 %d건 · %.1f건/초 · 구멍 %d · 레벨 %d",
                                 symbol.upper(), n, n / max(el, 1e-9), gaps,
                                 len(d.get("b", [])) + len(d.get("a", [])))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.warning("WS 끊김 %s: %s — 3초 뒤 재연결", type(exc).__name__, exc)
            out.close()
            await asyncio.sleep(3)


def main() -> None:
    ap = argparse.ArgumentParser(description="Binance 무기한선물 주문단위 호가변경 수집기")
    ap.add_argument("--symbol", default=SYMBOL)
    ap.add_argument("--root", default=str(DD_ROOT))
    a = ap.parse_args()
    log.info("수집 시작 %s → %s", a.symbol.upper(), a.root)
    try:
        asyncio.run(run(a.symbol.lower(), Path(a.root)))
    except KeyboardInterrupt:
        log.info("종료")


if __name__ == "__main__":
    main()
