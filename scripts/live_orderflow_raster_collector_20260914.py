#!/usr/bin/env python3
"""오더플로우 호가 래스터 수집기 — 대시보드 히트맵(bookmap 류)의 유일한 원천. (P0, 2026-09-14)

설계 원문: docs/dashboard_orderflow_footprint_heatmap_design_20260914.md

왜 지금 띄우는가: 히트맵의 과거는 **재구성이 불가능**하다. 저장소에 있는 L2 셋 중
depth20@100ms 는 폭이 $0.20, 벌크 bookDepth 는 가격별이 아닌 ±%누적, l2_anomaly 는 트리거
순간 수십 건뿐이다. 체결(풋프린트)은 data.binance.vision 일별 zip 으로 언제든 소급
재구성되지만 호가 잔량은 지금 안 받으면 영원히 없다. 그래서 이 파일이 P0 다.

왜 REST 폴링이 아니라 WS diff 인가: `depth?limit=1000` 을 1Hz 로 돌리면 weight 1,200/분
(상한 2,400/분/IP)을 혼자 먹는데 **실주문 봇과 같은 IP** 다. WS diff 는 실측 6.2 KB/s
(0.55 GB/일), weight 0. 스냅샷은 최초 1회 + 드리프트/시퀀스단절 때만.

트레이딩 봇과 완전 분리: 자기 WS, 자기 디렉터리. 이 저장소의 확립된 수집기 규약
(l2_anomaly_snapshot_collector.py 도크스트링)을 그대로 따른다. 죽어도 봇에 영향 없다.

저장 포맷 — duckdb 가 아니라 고정폭 플랫파일:
  히트맵 1행 = 1초 = 이미지 한 열이다. 질의 대상이 아니라 이미지라서 seek 한 번이면 끝이고,
  보존 만료가 `rm` 한 줄이다(duckdb 는 DELETE 해도 파일이 안 줄어든다 — 이게 진짜 이유).

  <OF_ROOT>/<SYMBOL>/<YYYY-MM-DDTHH>.f32   (UTC 시각별)
  헤더 32B: magic"FLWR" u16 ver, u16 n_bins, f32 bin_size, u32 flags,
            i64 hour_start_ms, u32 dt_ms, u32 reserved
  행 976B(n_bins=240): i64 ts_ms, i32 bin_lo, f32 mid, f32[n_bins] qty
  오프셋 = 32 + (초 인덱스) * 행크기      ← 파싱 0

  ⭐qty 는 **부호가 방향**이다: + 매수호가(bid), − 매도호가(ask). 한 배열로 절반 용량.
  ⭐mid=NaN 인 행 = 그 초의 북이 무효(재동기중/수집중단). 렌더러는 회색 열로 그린다.
    수집이 끊긴 구간을 보간하면 없는 유동성을 그리는 것이므로 **절대 채우지 않는다**.
  ⭐헤더에 ver/n_bins/bin_size 를 박아둔 이유: P1 에서 체결 레이어(trade dots)를 행에
    덧붙일 때 ver=2 로 올리면 되고, 옛 파일은 그대로 읽힌다. 포맷 마이그레이션 비용 0.

먼 구간 워밍업(알려진 성질, 버그 아님): REST 스냅샷의 최대 limit=1000 은 ETH 에서 ±0.43%
밖에 못 덮는다. 그 바깥 잔량은 diff 스트림이 "변한 레벨"을 보내줄 때만 채워지므로, 재시작
직후 ±0.43% 밖은 비어 있다가 수십 분에 걸쳐 찬다. 그래서 재동기는 **덮어쓰기가 아니라
병합**이다(스냅샷 범위 안만 교체, 바깥은 누적분 유지) — 안 그러면 30분마다 애써 모은 먼
구간을 스스로 지운다.

quant_ai conda env (websockets, aiohttp, numpy). 이 저장소의 다른 어떤 파일도 쓰지 않는다.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import gzip
import math
import os
import shutil
import struct
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger("OrderflowRaster")

ROOT = Path(__file__).resolve().parents[1]

SYMBOL = os.getenv("OF_SYMBOL", "ethusdt").lower()
BIN_SIZE = float(os.getenv("OF_BIN_SIZE", "0.5"))
N_BINS = int(os.getenv("OF_N_BINS", "240"))
OF_ROOT = Path(os.getenv("OF_ROOT", str(ROOT / "data" / "live" / "orderflow" / "raster")))
RETENTION_DAYS = int(os.getenv("OF_RETENTION_DAYS", "14"))
# 2026-09-19 만료 파일을 지우기 전에 **떠낸다**(사용자 결정). 화면은 14일이면 되지만
# 연구는 아니다 -- 호가벽 축 판정이 ~2026-11-14 인데 그때 래스터는 직전 14일뿐이었다.
# 실측 3.5MB -> 1.1MB(3.2배, 0.05s/파일) = 26MB/일. 60일 1.6GB.
# 🔴압축본은 seek 이 안 된다 -- 화면 경로(read_window)는 **비압축 14일만** 본다. 연구 전용이다.
OF_ARCHIVE_ROOT = Path(os.getenv("OF_ARCHIVE_ROOT",
                                 str(ROOT / "data" / "live" / "orderflow" / "raster_archive")))
OF_ARCHIVE = os.getenv("OF_ARCHIVE", "1") != "0"

_DEPTH_WS_URL = "wss://fstream.binance.com/ws/{symbol}@depth@500ms"
_SNAPSHOT_URL = "https://fapi.binance.com/fapi/v1/depth?symbol={SYMBOL}&limit=1000"

PRUNE_PCT = 4.0            # 이 밖의 레벨은 버린다(래스터가 ±2.4% 라 여유 있게)
RESYNC_DRIFT_PCT = 1.0     # 마지막 스냅샷 mid 대비 이만큼 밀리면 재동기(가지치기 구멍 방지)
RESYNC_INTERVAL_SEC = 1800 # 주기적 재동기(누적 드리프트/버그 자가치유)
STALE_SEC = 5.0            # 마지막 depth 이벤트가 이보다 오래되면 그 초는 무효

MAGIC = b"FLWR"
VERSION = 1
HEADER = struct.Struct("<4sHHfIqII")   # 32B
ROW_HEAD = struct.Struct("<qif")       # 16B: ts_ms, bin_lo, mid
assert HEADER.size == 32 and ROW_HEAD.size == 16


def row_bytes(n_bins: int = N_BINS) -> int:
    return ROW_HEAD.size + 4 * n_bins


def _hour_path(symbol: str, hour_start_ms: int, root: Path | None = None) -> Path:
    base = (root or OF_ROOT) / symbol.upper()
    stamp = datetime.fromtimestamp(hour_start_ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H")
    return base / f"{stamp}.f32"


# ── 시퀀스 상태기계 (순수함수라 --selftest 가 네트워크 없이 검증한다) ──────────────
def seq_decision(synced: bool, last_u: int, ev: dict) -> str:
    """바이낸스 USD-M diff depth 동기화 규약. 반환: apply | skip | resnapshot.

    규약: 스냅샷 lastUpdateId L 이후, 첫 적용 이벤트는 U <= L <= u 를 만족해야 하고,
    그 다음부터는 pu 가 직전 u 와 같아야 한다(끊기면 스냅샷을 다시 받는다)."""
    u, U, pu = int(ev["u"]), int(ev["U"]), int(ev.get("pu", -1))
    if not synced:
        if u < last_u:
            return "skip"          # 스냅샷보다 낡은 이벤트
        if U > last_u:
            return "resnapshot"    # 스냅샷과 스트림 사이에 구멍 — 다시 받는다
        return "apply"
    return "apply" if pu == last_u else "resnapshot"


def next_tick(target_sec: int, now: float) -> int:
    """다음에 쓸 초. "경계까지 남은 시간을 잔다"가 아니라 **목표 초를 명시적으로 센다**.

    앞의 방식은 본문 시간과 스케줄러 지연이 누적되면 조용히 틀어진다 -- 2026-09-14 실측에서
    서버는 33~34초마다 한 초를 건너뛰었고(무효행 3.14%), 로컬은 반대로 165초에 182회 기록
    (같은 초 중복)했다. 목표 초를 세면 둘 다 구조적으로 불가능하다.

    1초 이내로 늦었으면 **지금 속한 초를 즉시 쓴다**(그 초의 북 상태는 여전히 유효한 표본이다).
    더 밀렸으면 현재 초로 따라잡고, 건너뛴 초는 writer 가 무효행으로 메운다(보간 아님)."""
    nxt = target_sec + 1
    now_sec = int(now)
    return nxt if nxt >= now_sec else now_sec


class _RasterWriter:
    """시각별 고정폭 파일에 초당 한 행. 수집이 멈췄던 초는 무효행으로 메워 오프셋 산식을 지킨다."""

    def __init__(self, symbol: str, n_bins: int = N_BINS, bin_size: float = BIN_SIZE,
                 root: Path | None = None) -> None:
        self.symbol, self.n_bins, self.bin_size = symbol, n_bins, bin_size
        self.root = root or OF_ROOT
        self.row = row_bytes(n_bins)
        self._hour_ms: int | None = None
        self._fh = None
        self._blank = np.zeros(n_bins, dtype="<f4").tobytes()

    def _open(self, hour_ms: int) -> None:
        if self._fh is not None:
            self._fh.close()
        path = _hour_path(self.symbol, hour_ms, self.root)
        path.parent.mkdir(parents=True, exist_ok=True)
        fresh = not path.exists() or path.stat().st_size < HEADER.size
        self._fh = open(path, "r+b" if not fresh else "w+b")
        if fresh:
            self._fh.write(HEADER.pack(MAGIC, VERSION, self.n_bins, self.bin_size, 0,
                                       hour_ms, 1000, 0))
            self._fh.flush()
        self._hour_ms = hour_ms

    def write(self, sec_ms: int, bin_lo: int, mid: float, qty: np.ndarray | None) -> None:
        hour_ms = sec_ms - sec_ms % 3_600_000
        if hour_ms != self._hour_ms:
            self._open(hour_ms)
        idx = (sec_ms - hour_ms) // 1000
        target = HEADER.size + idx * self.row
        self._fh.seek(0, os.SEEK_END)
        size = self._fh.tell()
        if size < target:  # 프로세스가 죽어 있던 초들 — 무효행으로 메운다(보간 아님)
            pad = ROW_HEAD.pack(0, 0, math.nan) + self._blank
            self._fh.write(pad * ((target - size) // self.row))
        self._fh.seek(target)
        body = self._blank if qty is None else np.asarray(qty, dtype="<f4").tobytes()
        self._fh.write(ROW_HEAD.pack(sec_ms, bin_lo, mid) + body)
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def read_window(symbol: str, to_ms: int, cols: int, agg: int = 1,
                root: Path | None = None) -> dict:
    """[to_ms − cols*agg 초, to_ms] 를 **절대 가격축**으로 정렬해 돌려준다.

    행마다 bin_lo 가 다르므로(그 초의 mid 중심) 이미지로 그리려면 공통 격자로 옮겨야 한다.
    그 정렬이 여기서 끝나므로 API/렌더러는 계산할 게 없다. agg>1 은 창을 max 로 접는다
    (평균은 큰 벽 하나를 뭉개서 히트맵의 의미 자체를 지운다)."""
    total = cols * agg
    t_end = (to_ms // 1000) * 1000
    t0 = t_end - (total - 1) * 1000

    ts = np.zeros(total, dtype=np.int64)
    lo = np.zeros(total, dtype=np.int64)
    mid = np.full(total, np.nan, dtype=np.float32)
    n_bins = N_BINS
    bin_size = BIN_SIZE
    raw: np.ndarray | None = None

    hour = t0 - t0 % 3_600_000
    while hour <= t_end:
        path = _hour_path(symbol, hour, root)
        if path.exists():
            with open(path, "rb") as fh:
                head = fh.read(HEADER.size)
                if len(head) == HEADER.size:
                    magic, ver, hb, hbin, _flags, hour_ms, dt_ms, _rsv = HEADER.unpack(head)
                    if magic == MAGIC and ver == VERSION:
                        if raw is None:
                            n_bins, bin_size = hb, hbin
                            raw = np.zeros((total, n_bins), dtype=np.float32)
                        if hb == n_bins:
                            i0 = max(0, (hour - t0) // 1000)
                            j0 = max(0, (t0 - hour) // 1000)
                            j1 = min(3600, (t_end - hour) // 1000 + 1)
                            if j1 > j0:
                                fh.seek(HEADER.size + j0 * row_bytes(n_bins))
                                buf = fh.read((j1 - j0) * row_bytes(n_bins))
                                got = len(buf) // row_bytes(n_bins)
                                if got:
                                    blk = np.frombuffer(buf[:got * row_bytes(n_bins)], dtype=np.uint8)
                                    blk = blk.reshape(got, row_bytes(n_bins))
                                    ts[i0:i0 + got] = blk[:, :8].copy().view(np.int64).ravel()
                                    lo[i0:i0 + got] = blk[:, 8:12].copy().view(np.int32).ravel()
                                    mid[i0:i0 + got] = blk[:, 12:16].copy().view(np.float32).ravel()
                                    raw[i0:i0 + got] = blk[:, 16:].copy().view(np.float32)
        hour += 3_600_000

    if raw is None:
        raw = np.zeros((total, n_bins), dtype=np.float32)

    valid = np.isfinite(mid) & (ts > 0)
    if not valid.any():
        return {"symbol": symbol, "t0_ms": int(t0), "dt_s": agg, "cols": cols,
                "bin_size": float(bin_size), "bin_lo": 0, "n_bins": 0,
                "qty": np.zeros((cols, 0), np.float32), "mid": np.full(cols, np.nan, np.float32),
                "valid_ratio": 0.0}

    g_lo = int(lo[valid].min())
    g_hi = int((lo[valid] + n_bins).max())
    out = np.zeros((total, g_hi - g_lo), dtype=np.float32)
    for i in np.nonzero(valid)[0]:
        s = int(lo[i]) - g_lo
        out[i, s:s + n_bins] = raw[i]

    if agg > 1:  # 축소: 절대격자에서 |값| 최대를 남긴다(부호 유지)
        o = out.reshape(cols, agg, -1)
        pick = np.abs(o).argmax(axis=1)[:, None, :]
        out = np.take_along_axis(o, pick, axis=1)[:, 0, :]
        m = mid.reshape(cols, agg)
        mid = np.where(np.isfinite(m).any(axis=1),
                       np.nanmax(np.where(np.isfinite(m), m, -np.inf), axis=1), np.nan).astype(np.float32)
    return {"symbol": symbol, "t0_ms": int(t0), "dt_s": agg, "cols": cols,
            "bin_size": float(bin_size), "bin_lo": g_lo, "n_bins": out.shape[1],
            "qty": out, "mid": mid, "valid_ratio": float(valid.mean())}


def _archive_then_unlink(p: Path, symbol: str) -> None:
    """만료 래스터를 gzip 으로 떠내고 **압축이 성공한 뒤에만** 원본을 지운다.

    형제 수집기(`live_book_ticker_collector_20260914._gzip_done`)와 같은 규약이다. 임포트하지
    않고 복제한 이유는 이 수집기의 독립 규약 때문이다(도크스트링: 저장소의 다른 어떤 파일도
    쓰지 않는다 -- 죽어도 봇에 영향 없다).

    데이터 유실 경로라 세 가지를 지킨다:
      1. 임시명 -> os.replace. 직접 <dst>.gz 로 쓰다 죽으면 **잘린 gz** 가 남고, 다음 실행이
         그걸 «이미 보관됨»으로 읽어 원본을 지운다(영원히 못 되돌린다).
      2. 이미 온전한 .gz 가 있으면 다시 압축하지 않고 원본만 지운다(멱등 -- 압축 뒤 unlink
         직전에 죽은 경우).
      3. 실패하면 **원본을 유지**하고 경고만 남긴다. 다음 시간에 다시 시도된다.
    """
    day = p.stem.split("T")[0]                 # 2026-09-14T12 -> 2026-09-14 (일별 보관)
    dst_dir = OF_ARCHIVE_ROOT / symbol.upper() / day
    dst = dst_dir / (p.name + ".gz")
    try:
        if dst.exists() and dst.stat().st_size > 0:
            p.unlink(missing_ok=True)          # 2
            return
        dst_dir.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        with open(p, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
            shutil.copyfileobj(fi, fo, length=1 << 20)
        if tmp.stat().st_size <= 0:
            raise OSError("빈 압축본")
        tmp.replace(dst)                       # 1
        before, after = p.stat().st_size, dst.stat().st_size
        p.unlink()
        logger.info("보관 %s → %s · %.1fMB → %.1fMB (%.1f배)", p.name, day,
                    before / 1e6, after / 1e6, before / max(after, 1))
    except Exception as exc:  # noqa: BLE001 -- 보관 실패가 수집을 멈추면 안 된다
        logger.warning("보관 실패 %s: %s — 원본 유지(다음 시간 재시도)", p.name, exc)


class OrderflowRasterCollector:
    def __init__(self, symbol: str = SYMBOL) -> None:
        self.symbol = symbol.lower()
        self._running = False
        self._bids: dict[float, float] = {}
        self._asks: dict[float, float] = {}
        self._synced = False
        self._last_u = 0
        self._last_event_ts = 0.0
        self._snap_mid = 0.0
        self._snap_ts = 0.0
        self._want_resync = False
        self._writer = _RasterWriter(self.symbol)
        self.rows_written = 0
        self.rows_invalid = 0
        self.ticks_late = 0

    # ── 호가북 ────────────────────────────────────────────────────────────────
    async def _fetch_snapshot(self) -> dict:
        import aiohttp
        url = _SNAPSHOT_URL.format(SYMBOL=self.symbol.upper())
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=10)) as r:
                return await r.json(content_type=None)

    def _merge_snapshot(self, snap: dict) -> int:
        """스냅샷 범위 **안**만 교체하고 바깥 누적분은 남긴다(도크스트링의 워밍업 항목)."""
        bids = {float(p): float(q) for p, q in snap.get("bids", []) if float(q) > 0}
        asks = {float(p): float(q) for p, q in snap.get("asks", []) if float(q) > 0}
        if bids and asks:
            lo, hi = min(bids), max(asks)
            self._bids = {p: q for p, q in self._bids.items() if p < lo} | bids
            self._asks = {p: q for p, q in self._asks.items() if p > hi} | asks
            self._snap_mid = (max(bids) + min(asks)) / 2
        self._snap_ts = time.time()
        return int(snap.get("lastUpdateId", 0))

    def _apply(self, ev: dict) -> None:
        for side, book in (("b", self._bids), ("a", self._asks)):
            for p_s, q_s in ev.get(side, ()):
                p, q = float(p_s), float(q_s)
                if q == 0.0:
                    book.pop(p, None)
                else:
                    book[p] = q

    def _prune(self, mid: float) -> None:
        lo, hi = mid * (1 - PRUNE_PCT / 100), mid * (1 + PRUNE_PCT / 100)
        self._bids = {p: q for p, q in self._bids.items() if p >= lo}
        self._asks = {p: q for p, q in self._asks.items() if p <= hi}

    async def _depth_loop(self) -> None:
        import websockets
        url = _DEPTH_WS_URL.format(symbol=self.symbol)
        delay = 3.0
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20, ping_timeout=10) as ws:
                    logger.info("depth WS 연결: %s", url)
                    delay = 3.0
                    self._synced = False
                    need_snapshot = True
                    while self._running:
                        # ping/pong 이 살아 있어도 스트림만 멎는 경우가 실제로 있었다
                        # (l2_anomaly 수집기 주석). recv 를 묶어 두면 멎을 때 재접속된다.
                        raw = await asyncio.wait_for(ws.recv(), timeout=35.0)
                        ev = json.loads(raw)
                        if ev.get("e") != "depthUpdate":
                            continue
                        if need_snapshot or self._want_resync:
                            # 스냅샷을 받는 동안 도착하는 이벤트는 WS 큐에 쌓인다(유실 아님).
                            self._last_u = self._merge_snapshot(await self._fetch_snapshot())
                            self._synced = False
                            need_snapshot = self._want_resync = False
                            continue
                        what = seq_decision(self._synced, self._last_u, ev)
                        if what == "skip":
                            continue
                        if what == "resnapshot":
                            logger.warning("시퀀스 단절(last_u=%s U=%s pu=%s) — 재동기",
                                           self._last_u, ev.get("U"), ev.get("pu"))
                            self._synced = False
                            need_snapshot = True
                            continue
                        self._apply(ev)
                        self._last_u = int(ev["u"])
                        self._synced = True
                        self._last_event_ts = time.time()
            except Exception as e:
                if self._running:
                    logger.warning("depth WS 끊김, %.0fs 후 재접속: %s", delay, e)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 60.0)

    # ── 래스터 ────────────────────────────────────────────────────────────────
    def _row(self) -> tuple[int, float, np.ndarray] | None:
        if not self._synced or not self._bids or not self._asks:
            return None
        if time.time() - self._last_event_ts > STALE_SEC:
            return None
        best_bid, best_ask = max(self._bids), min(self._asks)
        if not (best_ask > best_bid > 0):
            return None
        mid = (best_bid + best_ask) / 2
        bin_lo = int(mid // BIN_SIZE) - N_BINS // 2
        arr = np.zeros(N_BINS, dtype=np.float32)
        for p, q in self._bids.items():
            i = int(p // BIN_SIZE) - bin_lo
            if 0 <= i < N_BINS:
                arr[i] += q
        for p, q in self._asks.items():
            i = int(p // BIN_SIZE) - bin_lo
            if 0 <= i < N_BINS:
                arr[i] -= q          # 부호가 방향. 스프레드가 걸친 한 빈만 상계된다.
        return bin_lo, mid, arr

    async def _raster_loop(self) -> None:
        tick = 0
        target = int(time.time()) + 1
        late_ms: list[float] = []
        body_ms: list[float] = []
        while self._running:
            delay = target - time.time()
            if delay > 0:
                await asyncio.sleep(delay)
            else:
                self.ticks_late += 1
            woke = time.time()
            late_ms.append((woke - target) * 1000)
            sec_ms = target * 1000
            row = self._row()
            if row is None:
                self._writer.write(sec_ms, 0, math.nan, None)
                self.rows_invalid += 1
            else:
                self._writer.write(sec_ms, *row)
                self.rows_written += 1
                mid = row[1]
                if (abs(mid - self._snap_mid) / mid * 100 > RESYNC_DRIFT_PCT
                        or time.time() - self._snap_ts > RESYNC_INTERVAL_SEC):
                    self._want_resync = True
                if tick % 60 == 0:
                    self._prune(mid)
            body_ms.append((time.time() - woke) * 1000)
            nxt = next_tick(target, time.time())
            if nxt > target + 1:
                # 왜 건너뛰었는지를 그 자리에서 남긴다 -- 사후에 파일만 보면 "무효행"일 뿐,
                # 지각인지 본문 지연인지 구분이 안 된다(2026-09-14 에 두 번 헛짚었다).
                logger.warning("틱 건너뜀 %d→%d  지각 %.0fms 본문 %.0fms",
                               target, nxt, late_ms[-1], body_ms[-1])
            target = nxt
            tick += 1
            if tick % 300 == 0:
                lm, bm = sorted(late_ms), sorted(body_ms)
                q = lambda v, f: v[min(len(v) - 1, int(len(v) * f))]
                logger.info("래스터 %d행(무효 %d, 지각 %d) 호가 bid/ask=%d/%d | "
                            "깨어남지각 p50 %.0f p99 %.0f max %.0fms · 본문 p50 %.0f p99 %.0f max %.0fms",
                            self.rows_written, self.rows_invalid, self.ticks_late,
                            len(self._bids), len(self._asks),
                            q(lm, .5), q(lm, .99), lm[-1], q(bm, .5), q(bm, .99), bm[-1])
                late_ms.clear(); body_ms.clear()
            if tick % 3600 == 0:
                self._purge_old()



    def _purge_old(self) -> None:
        cutoff = datetime.now(timezone.utc) - timedelta(days=RETENTION_DAYS)
        base = OF_ROOT / self.symbol.upper()
        for p in base.glob("*.f32"):
            try:
                stamp = datetime.strptime(p.stem, "%Y-%m-%dT%H").replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if stamp < cutoff:
                _archive_then_unlink(p, self.symbol) if OF_ARCHIVE else p.unlink(missing_ok=True)

    async def run(self) -> None:
        self._running = True
        logger.info("오더플로우 래스터 수집 시작 %s bin=%.2f n_bins=%d → %s",
                    self.symbol, BIN_SIZE, N_BINS, OF_ROOT / self.symbol.upper())
        try:
            await asyncio.gather(self._depth_loop(), self._raster_loop())
        finally:
            self._running = False
            self._writer.close()


# ── 자체점검: 네트워크 없이 시퀀스 상태기계 + 파일 오프셋/정렬 왕복 ────────────────
def _selftest() -> None:
    import shutil
    import tempfile

    ev = lambda U, u, pu: {"U": U, "u": u, "pu": pu}
    assert seq_decision(False, 100, ev(10, 50, 9)) == "skip"          # 스냅샷보다 낡음
    assert seq_decision(False, 100, ev(90, 110, 89)) == "apply"       # U<=L<=u 로 걸침
    assert seq_decision(False, 100, ev(101, 130, 100)) == "resnapshot"  # 사이에 구멍
    assert seq_decision(True, 130, ev(131, 140, 130)) == "apply"      # pu 체인 정상
    assert seq_decision(True, 130, ev(150, 160, 149)) == "resnapshot"  # 체인 단절

    assert next_tick(100, 100.5) == 101      # 제때 -- 다음 초를 기다린다
    assert next_tick(100, 100.999) == 101
    assert next_tick(100, 101.5) == 101      # 1초 이내 지각 -- 지금 속한 초를 즉시 쓴다
    assert next_tick(100, 105.2) == 105      # 심한 지각 -- 따라잡고 101~104 는 무효행
    assert next_tick(100, 101.0) == 101

    tmp = Path(tempfile.mkdtemp())
    try:
        w = _RasterWriter("testusdt", root=tmp)
        t0 = 1_787_000_000_000 - 1_787_000_000_000 % 3_600_000  # 시각 경계
        a = np.zeros(N_BINS, np.float32); a[10] = 5.0; a[11] = -7.0
        w.write(t0 + 0, 1000, 2500.0, a)
        w.write(t0 + 3000, 1000, 2500.5, a)   # 1,2초는 수집 중단 → 무효행으로 메워져야 한다
        b = np.zeros(N_BINS, np.float32); b[9] = 3.0
        w.write(t0 + 4000, 1002, 2501.0, b)   # bin_lo 가 다른 행 → 절대격자 정렬 대상
        w.close()

        size = _hour_path("testusdt", t0, tmp).stat().st_size
        assert size == HEADER.size + 5 * row_bytes(), size

        r = read_window("testusdt", t0 + 4000, cols=5, root=tmp)
        assert r["n_bins"] == N_BINS + 2, r["n_bins"]          # bin_lo 1000..1002 로 2칸 확장
        assert r["bin_lo"] == 1000 and abs(r["valid_ratio"] - 0.6) < 1e-6, r["valid_ratio"]
        assert math.isnan(float(r["mid"][1])) and math.isnan(float(r["mid"][2]))  # 구멍은 NaN
        assert r["qty"][0, 10] == 5.0 and r["qty"][0, 11] == -7.0
        assert r["qty"][4, 9 + 2] == 3.0, r["qty"][4, 9 + 2]   # 정렬 후 2칸 밀린 자리
        assert r["qty"][1].sum() == 0.0                        # 무효행은 0

        r2 = read_window("testusdt", t0 + 4000, cols=1, agg=5, root=tmp)
        assert r2["qty"].shape == (1, N_BINS + 2)
        assert r2["qty"][0, 11] == -7.0                        # 접을 때 부호 유지(|값| 최대)

        # 시각 경계 롤오버: 스모크로는 절대 안 지나가는 경로라 여기서 반드시 짚는다.
        # 새 파일이 열리고, 그 안의 오프셋은 **새 시각 기준**으로 다시 0부터 세야 한다.
        w2 = _RasterWriter("testusdt", root=tmp)
        c = np.zeros(N_BINS, np.float32); c[20] = 11.0
        w2.write(t0 + 3_599_000, 1000, 2499.0, c)              # 시각의 마지막 초
        w2.write(t0 + 3_600_000 + 2000, 1001, 2502.0, c)       # 다음 시각의 2초
        w2.close()
        assert _hour_path("testusdt", t0 + 3_600_000, tmp).stat().st_size == \
            HEADER.size + 3 * row_bytes()                      # 0,1초는 무효행 + 2초
        r3 = read_window("testusdt", t0 + 3_600_000 + 2000, cols=4, root=tmp)
        assert abs(float(r3["mid"][0]) - 2499.0) < 1e-3        # 파일 경계를 넘어 이어 읽는다
        assert math.isnan(float(r3["mid"][1])) and math.isnan(float(r3["mid"][2]))
        assert abs(float(r3["mid"][3]) - 2502.0) < 1e-3
        assert r3["qty"][3, 20 + 1] == 11.0                    # bin_lo 1001 → 1칸 밀린 자리
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest OK")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="네트워크 없이 상태기계·파일 왕복 검증")
    ap.add_argument("--seconds", type=float, default=0.0, help=">0 이면 그 시간만 수집하고 종료")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.selftest:
        _selftest()
        return
    c = OrderflowRasterCollector()
    try:
        if args.seconds > 0:
            asyncio.run(asyncio.wait_for(c.run(), timeout=args.seconds))
        else:
            asyncio.run(c.run())
    except (KeyboardInterrupt, asyncio.TimeoutError):
        logger.info("종료: 래스터 %d행(무효 %d)", c.rows_written, c.rows_invalid)


if __name__ == "__main__":
    main()
