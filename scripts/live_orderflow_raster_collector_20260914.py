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
# 2026-09-20 보관 포맷을 gzip -> **parquet** 으로(사용자 지시). gzip 이 더 작지만(1.04 vs
# 1.87MB/시간) **읽을 코드가 없다** -- read_window 는 비압축 .f32 만 seek 한다. 10-13 호가벽
# 재측정 때 «파일은 있는데 못 읽는» 상태가 된다. parquet 은 duckdb 가 바로 질의한다.
# ⭐화면 경로는 그대로 .f32 다(seek 0ms vs parquet+duckdb 329ms 실측). 역할을 나눈 것이지
#   바꾼 게 아니다 -- 체결이 «메모리+스냅샷(화면) / trade_tape.duckdb(연구)» 로 갈린 것과 같다.
OF_ARCHIVE_DB = OF_ARCHIVE_ROOT / "orderbook.duckdb"

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


def row_stats(a: "np.ndarray", dt_s: int = 1) -> dict:
    """행(가격빈)마다 시간축을 접어 «지금·바닥·최대·재깔림·최근변화»를 낸다.

    `a` 는 유효 열만 남긴 **절대값** 잔량 (축: 시간 x 가격빈).

    ⭐min 하나만 쓰면 「한 번도 안 빠진 양」밖에 못 본다. 실측 600초 창(ETH, 247빈):
      sum(min) 103,531 < sum(now) 170,775 < sum(max) 310,024 << **sum(refill) 1,029,281**
      -- 걸린 양의 **6배**가 창 안에서 다시 깔린다(refill/drain = 1.00, 순수 회전).
      화면은 그 넷 중 가장 작은 걸 그리고 있었다.
    ⭐재깔림은 지터가 아니라 **덩어리**다: -2.05% 빈은 증분 p90 이 1,750 ETH(= 블록 크기)
      이고 10분에 82번 올라간다. 같은 창의 +1.94% 빈은 refill 641 로 사실상 정적인데,
      pers 로 줄 세우면 **재깔리는 쪽이 더 짧은 막대**였다.
    🔴`refill` 은 **하한**이다 -- agg>1 로 접힌 열을 받으면 dt_s 보다 짧은 왕복은 사라진다.
    🔴«지지·저항»도 «스푸핑»도 아니다. 「가격이 다가오면 빠지는가」는 10분 창에서 자격
      빈이 9개뿐(near/far 중앙 0.875)이라 **검정력이 없다**. 여기 값은 전부 서술이다.
    """
    if a.ndim != 2 or a.shape[0] == 0:
        raise ValueError("a must be (time, bins) with time > 0")
    d = np.diff(a, axis=0)
    n60 = max(1, int(round(60.0 / max(dt_s, 1))))
    d60 = (a[-n60:].mean(axis=0) - a[-6 * n60:-n60].mean(axis=0)
           if a.shape[0] >= 6 * n60 else np.zeros(a.shape[1], np.float32))
    # ⭐재깔림의 «단위». 배수(refill/peak)만으로는 「1,780 ETH 블록을 18번 다시 깐 것」과
    #   「15 ETH 를 2,000번 깐 것」이 같은 값이 된다 -- 전자는 작업자 한 명이 한 자리를
    #   지키는 것이고 후자는 알고리즘 잔물결이라 전혀 다른 행동이다.
    #   실측(1,946행): rho(배수, 블록크기) 0.642 · rho(배수, 증분 중앙값) **0.077** ·
    #   같은 배수 5분위 안에서 p90/peak 이 0.012~0.947(79배)로 갈린다 = 별개 축이다.
    # 🔴«개수 기준» 분위는 안 된다. 증분은 «큰 재호가 몇 번 + 작은 지터 수백 번»이라
    #   중앙값도 p90 도 지터를 잰다 -- selftest 에서 0.5 짜리 199번 + 500 짜리 1번을 주면
    #   p90 이 0.5 를 답한다. 재깔림은 **큰 덩어리가 물량을 지배**하므로 물량으로 가중한다.
    #   blk = 증분을 오름차순으로 쌓았을 때 **누적 물량이 절반을 넘는 지점의 증분 크기**.
    #   위 예에서 500(전체 물량의 83%)을 답한다.
    up = np.clip(d, 0, None)
    n_up = (up > 0).sum(axis=0).astype(np.float32)
    srt = np.sort(up, axis=0)
    csum = np.cumsum(srt, axis=0)
    idx = (csum >= csum[-1] * 0.5).argmax(axis=0)
    blk = srt[idx, np.arange(a.shape[1])].astype(np.float32)
    return {"inst": a[-1], "pers": a.min(axis=0), "peak": a.max(axis=0),
            "refill": up.sum(axis=0), "d60": d60, "blk": blk, "n_up": n_up}


def approach_ratio(w: dict, *, near_pct: float = 0.35, far_lo: float = 0.35,
                   far_hi: float = 1.2, min_obs_s: float = 60.0) -> dict:
    """가격대마다 «가격이 다가왔을 때 두꺼워졌나 얇아졌나»를 잰다.

    같은 빈이 **가까웠던 때와 멀었던 때를 둘 다 겪어야** 비교가 성립하므로, 창이 짧으면
    아무것도 안 나온다(2026-09-20 실측 자격 빈: 15분 22 · 1시간 54 · **4시간 82**).

    🔴날값(near평균/far평균)은 「호가는 원래 mid 근처가 두껍다」에 업혀 있다. 같은 빈이
      near 일 때는 책의 두꺼운 부분에, far 일 때는 얇은 꼬리에 있을 뿐이다. 그래서 **같은
      시각·같은 거리대의 평균 잔량으로 나눈 뒤** 비교한다. 정규화가 실제로 상당 부분을
      걷어낸다(실측 중앙 1.44 -> 1.22 · grow>1.25 가 65% -> 46%).
    🔴남은 잔차도 **판정이 아니라 관측치**다. 지배적 방향이 «다가오면 커진다»(46%)이고
      «얇아진다»는 16% 꼬리다 -- 스푸핑 통설과 반대다. 창 하나·자산 하나·홀드아웃 없음.
    🔴자격 빈은 전체의 28%뿐이다. 나머지는 NaN 이다 -- 화면은 **없는 값을 0으로 그리면
      안 된다**(«얇다»와 «모른다»가 같은 그림이 된다).
    """
    mid = w["mid"]
    ok = np.isfinite(mid) & (mid > 0)   # mid 로 나눈다 -- 0 이 섞이면 거리가 inf 다
    n_bins = int(w["n_bins"])
    out = np.full(n_bins, np.nan, dtype=np.float32)
    if int(ok.sum()) < 10 or n_bins == 0:
        return {"bin_lo": int(w["bin_lo"]), "n_bins": n_bins,
                "bin_size": float(w["bin_size"]), "ratio": out, "qualifying": 0}
    a = np.abs(w["qty"][ok]).astype(np.float32)
    md = mid[ok].astype(np.float32)
    price = ((w["bin_lo"] + np.arange(n_bins)) * w["bin_size"]).astype(np.float32)
    dm = np.abs(price[None, :] - md[:, None]) / md[:, None] * 100.0

    # 거리대별 평균으로 정규화. bincount 한 번이면 끝난다 -- 밴드마다 마스크를 돌리면
    # (14400 x 293) x 50 이라 폴링 경로에 못 쓴다.
    edges = np.arange(0.0, 2.5001, 0.05, dtype=np.float32)
    k = len(edges) - 1
    bkt = np.clip(np.digitize(dm, edges) - 1, 0, k - 1).astype(np.int64)
    t = a.shape[0]
    idx = (np.arange(t, dtype=np.int64)[:, None] * k + bkt).ravel()
    sums = np.bincount(idx, weights=a.ravel(), minlength=t * k).reshape(t, k)
    cnts = np.bincount(idx, minlength=t * k).reshape(t, k)
    norm = np.take_along_axis(sums / np.maximum(cnts, 1), bkt, axis=1)
    rel = a / np.maximum(norm, 1e-9)

    near = dm < near_pct
    far = (dm >= far_lo) & (dm < far_hi)
    n_near, n_far = near.sum(axis=0), far.sum(axis=0)
    need = max(1, int(round(min_obs_s / max(float(w["dt_s"]), 1.0))))
    num = (rel * near).sum(axis=0) / np.maximum(n_near, 1)
    den = (rel * far).sum(axis=0) / np.maximum(n_far, 1)
    qual = (n_near >= need) & (n_far >= need) & (a.max(axis=0) > 0) & (den > 0)
    out[qual] = (num[qual] / den[qual]).astype(np.float32)
    return {"bin_lo": int(w["bin_lo"]), "n_bins": n_bins, "bin_size": float(w["bin_size"]),
            "ratio": out, "qualifying": int(qual.sum())}


def vol_scores(w: dict, idx=None, *, lookback_s: int = 600, past_s: int = 300,
               band_pct: float = 0.5, min_bins: int = 5) -> tuple:
    """«앞으로 얼마나 흔들릴까»의 두 재료를 낸다 -- (직전 실현변동 bp, 재깔림).

    2026-09-20 에 실제로 재서 고른 조합이다. 후보를 전부 같은 자로 재고 **떨어뜨린** 결과:

      5분위별 실제 |수익률|(300초 뒤) 중앙 · 표본 7,864 · 독립 일수 7
        합성(재깔림+블록−지속)  6.1 7.3 9.8 11.0 **7.7**  비단조 · 1.19배
        재깔림 단독             6.2 7.1 8.3 12.0 **8.4**  비단조 · 1.31배
        직전 실현변동 단독      6.7 7.1 7.4  8.3  11.4    단조   · 1.63배
        **직전변동 + 재깔림**   6.0 7.2 8.2  8.7  11.5    단조   · 1.77배

    🔴호가 축만으로 만든 신호는 전부 **최상위 분위에서 꺾인다**. 상관계수만 보면(재깔림
      +0.166) 쓸 만해 보이는데 분위로 쪼개면 Q5 가 Q4 보다 조용하다 -- 화면에 문턱을
      띄우려면 상관이 아니라 이 표를 봐야 한다.
    🔴그래서 호가는 **주역이 아니라 증분**이다. 그리고 그 증분(1.63 -> 1.77배, +0.140)은
      일블록 부트스트랩 CI [-0.014, +0.301] 로 **0 을 포함한다**. 부분 상관에서는 섰지만
      (+0.139, CI 0 배제) 배수에서는 못 섰다. 독립 일수 7 이 한계다.
    🔴블록·지속은 뺐다. 넣으면 오히려 나빠진다(1.77 -> 1.19배).

    반환: (past_bp, refill) 두 1차원 배열. idx=None 이면 마지막 초 하나.
    자격 미달(mid 결측 · 밴드 안 유효 빈 부족)인 인덱스는 **빠진다** -- 0 으로 채우면
    「조용하다」로 읽힌다.
    """
    mid, dt = w["mid"], max(int(w["dt_s"]), 1)
    lb, pb = lookback_s // dt, past_s // dt
    n_bins = int(w["n_bins"])
    if n_bins == 0 or len(mid) <= lb:
        return np.zeros(0, np.float32), np.zeros(0, np.float32)
    a = np.abs(w["qty"])
    price = (w["bin_lo"] + np.arange(n_bins)) * w["bin_size"]
    ok = np.isfinite(mid) & (mid > 0)
    past, refill = [], []
    for i in (range(len(mid) - 1, len(mid)) if idx is None else idx):
        if i < lb or i >= len(mid) or not ok[i] or not ok[i - pb]:
            continue
        m = float(mid[i])
        win = a[i - lb:i + 1]
        peak = win.max(axis=0)
        sel = (np.abs((price - m) / m * 100.0) <= band_pct) & (peak > 0)
        if int(sel.sum()) < min_bins:
            continue
        up = np.clip(np.diff(win[:, sel], axis=0), 0, None)
        past.append(abs(m - float(mid[i - pb])) / m * 1e4)
        refill.append(float(up.sum() / max(float(peak[sel].sum()), 1e-9)))
    return np.asarray(past, np.float32), np.asarray(refill, np.float32)


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
    # 🔴2026-09-20 구멍(sparse hole)은 **0 으로 읽힌다** -- 수집기가 시각 중간에 시작하면
    #   그 앞 초들은 파일에 안 쓰이고, 그 자리를 읽으면 ts=0·mid=0.0·qty=0 이 나온다.
    #   mid=0.0 은 «유한값»이라 소비자의 isfinite() 를 그냥 통과한다(그래서 여기 ts>0
    #   가드가 원래 있었다). 무효행을 NaN 으로 바꿔 이 포맷의 계약과 맞춘다 --
    #   "mid=NaN 인 행 = 그 초의 북이 무효"(위 도크스트링). 안 고치면 소비자마다
    #   0 을 «유동성 0» 으로 세서 pers 가 0 이 되고 refill 이 부풀고 1/mid 가 터진다.
    mid = np.where(valid, mid, np.nan).astype(np.float32)
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


def _f32_to_parquet(src: Path, dst: Path) -> None:
    """고정폭 .f32 한 시간치를 parquet 으로. 0 인 빈은 버린다(롱 포맷) -- 실측 864k행/1.6MB.

    🔴지연 임포트다. 이 수집기는 도크스트링대로 라이브 루프에서 websockets/aiohttp/numpy 만
      쓴다(RSS 67MB). pyarrow 를 모듈 상단에 두면 매 재시작마다 그 비용을 문다.
    """
    import pyarrow as pa            # noqa: PLC0415
    import pyarrow.parquet as pq    # noqa: PLC0415

    raw = src.read_bytes()
    if len(raw) < HEADER.size:
        raise OSError("헤더보다 짧다")
    magic, ver, nb, binsz, _flags, _hour, _dt, _rsv = HEADER.unpack(raw[:HEADER.size])
    if magic != MAGIC:
        raise OSError("magic 불일치")
    rb = row_bytes(nb)
    n = (len(raw) - HEADER.size) // rb
    if n <= 0:
        raise OSError("행이 없다")
    blk = np.frombuffer(raw[HEADER.size:HEADER.size + n * rb], dtype=np.uint8).reshape(n, rb)
    ts = blk[:, :8].copy().view(np.int64).ravel()
    lo = blk[:, 8:12].copy().view(np.int32).ravel()
    mid = blk[:, 12:16].copy().view(np.float32).ravel()
    qty = blk[:, 16:].copy().view(np.float32)
    # mid=NaN 인 초는 그 초의 북이 무효다 -- 보관에서도 **빼지 않고** 남긴다(결측을 0 으로
    # 바꾸면 «없는 유동성»이 되고, 그건 화면 규약과 같은 이유로 금지다). qty 만 0 이라 행이
    # 안 생기고, mid 테이블로 무효 초를 따로 알 수 있다.
    r, c = np.nonzero(qty)
    tbl = pa.table({
        "ts_ms": pa.array(ts[r], pa.int64()),
        "bin": pa.array((lo[r] + c).astype(np.int32), pa.int32()),
        "qty": pa.array(qty[r, c], pa.float32()),        # 부호가 방향: + 비드 / - 아스크
    })
    meta = pa.table({"ts_ms": pa.array(ts, pa.int64()), "mid": pa.array(mid, pa.float32())})
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    pq.write_table(tbl, tmp, compression="zstd")
    tmp.replace(dst)
    mtmp = dst.with_name(dst.stem + "_mid.parquet.part")
    mdst = dst.with_name(dst.stem + "_mid.parquet")
    pq.write_table(meta, mtmp, compression="zstd")
    mtmp.replace(mdst)


def _ensure_archive_db() -> None:
    """연구용 duckdb. **데이터를 복사하지 않고** parquet 글롭 위에 뷰만 얹는다 --
    `SELECT * FROM book` 이 바로 된다. 복사하면 디스크가 두 배가 되고 둘이 갈릴 수 있다.
    🔴단일 writer 문제도 이 방식이면 없다: 이 파일에는 뷰 정의뿐이고 실제 바이트는
      불변 parquet 이다(체결 쪽 trade_tape.duckdb 가 «쓸 때만 연다»로 푸는 그 문제)."""
    import duckdb  # noqa: PLC0415
    con = duckdb.connect(str(OF_ARCHIVE_DB))
    try:
        g = str(OF_ARCHIVE_ROOT / "*" / "*" / "*T[0-9][0-9].f32.parquet")
        m = str(OF_ARCHIVE_ROOT / "*" / "*" / "*_mid.parquet")
        con.execute(f"CREATE OR REPLACE VIEW book AS SELECT * FROM read_parquet('{g}')")
        con.execute(f"CREATE OR REPLACE VIEW book_mid AS SELECT * FROM read_parquet('{m}')")
    finally:
        con.close()


def _archive_then_unlink(p: Path, symbol: str) -> None:
    """만료 래스터를 parquet 으로 떠내고 **성공한 뒤에만** 원본을 지운다.

    형제 수집기(`live_book_ticker_collector_20260914._gzip_done`)와 같은 규약이다. 임포트하지
    않고 복제한 이유는 이 수집기의 독립 규약 때문이다(도크스트링: 저장소의 다른 어떤 파일도
    쓰지 않는다 -- 죽어도 봇에 영향 없다).

    데이터 유실 경로라 세 가지를 지킨다:
      1. 임시명 -> os.replace. 직접 쓰다 죽으면 **잘린 파일**이 남고, 다음 실행이 그걸
         «이미 보관됨»으로 읽어 원본을 지운다(영원히 못 되돌린다).
      2. 이미 온전한 보관본이 있으면 다시 쓰지 않고 원본만 지운다(멱등).
      3. 실패하면 **원본을 유지**하고 경고만 남긴다. 다음 시간에 다시 시도된다.
    🔴parquet 변환이 실패하면(pyarrow 부재 등) gzip 으로 떨어진다 -- 읽기는 불편해도
      **원본을 잃는 것보다 낫다**. 그 경우 로그에 남는다.
    """
    day = p.stem.split("T")[0]                 # 2026-09-14T12 -> 2026-09-14 (일별 보관)
    dst_dir = OF_ARCHIVE_ROOT / symbol.upper() / day
    dst = dst_dir / (p.name + ".parquet")
    gz = dst_dir / (p.name + ".gz")
    try:
        if (dst.exists() and dst.stat().st_size > 0) or (gz.exists() and gz.stat().st_size > 0):
            p.unlink(missing_ok=True)          # 2
            return
        before = p.stat().st_size
        try:
            _f32_to_parquet(p, dst)            # 1 (내부에서 .part -> replace)
            after, how = dst.stat().st_size, "parquet"
            try:
                _ensure_archive_db()
            except Exception as exc:  # noqa: BLE001 -- 뷰는 언제든 다시 만들 수 있다
                logger.warning("보관 뷰 갱신 실패: %s (parquet 은 정상)", exc)
        except Exception as exc:  # noqa: BLE001 -- 3 의 폴백
            logger.warning("parquet 변환 실패 %s: %s — gzip 으로 보관한다", p.name, exc)
            dst_dir.mkdir(parents=True, exist_ok=True)
            tmp = gz.with_suffix(gz.suffix + ".part")
            with open(p, "rb") as fi, gzip.open(tmp, "wb", compresslevel=6) as fo:
                shutil.copyfileobj(fi, fo, length=1 << 20)
            if tmp.stat().st_size <= 0:
                raise OSError("빈 압축본") from exc
            tmp.replace(gz)
            after, how = gz.stat().st_size, "gzip"
        p.unlink()
        logger.info("보관 %s → %s/%s · %.1fMB → %.1fMB (%.1f배)", p.name, day, how,
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

    # 행 통계: «한 번 깔고 앉은 벽»과 «같은 블록을 다시 까는 벽»이 갈려야 한다 -- 화면이
    # 막대 농도로 말하는 게 정확히 이 차이라, 여기서 안 짚으면 어디서도 안 짚는다.
    t = np.zeros((400, 3), np.float32)
    t[:, 0] = 100.0                        # 정적: 내내 100, 한 번도 안 바뀜
    t[::2, 1] = 80.0                       # 재깔림: 80 블록을 깔았다 뺐다 199번
    t[:340, 2] = 10.0; t[340:, 2] = 50.0   # 최근 60초에 쌓는 중
    st = row_stats(t, dt_s=1)
    assert st["inst"].tolist() == [100.0, 0.0, 50.0]
    assert st["pers"].tolist() == [100.0, 0.0, 10.0]
    assert st["peak"].tolist() == [100.0, 80.0, 50.0]
    assert st["refill"][0] == 0.0 and abs(float(st["refill"][1]) - 80.0 * 199) < 1e-3
    assert st["d60"].tolist() == [0.0, 0.0, 40.0]        # 정적 0 · 회전 0 · 신축 +40
    # 재깔림의 «단위»: 같은 배수라도 한 번에 얼마씩 깔렸는지는 다르다.
    assert st["n_up"].tolist() == [0.0, 199.0, 1.0]       # 정적 0회 · 블록 199회 · 신축 1회
    assert abs(float(st["blk"][1]) - 80.0) < 1e-3, st["blk"][1]   # 블록 크기 = 80
    assert st["blk"][0] == 0.0                            # 안 올라간 행은 0
    # 🔴개수 기준 분위(중앙값·p90)였다면 잔물결에 묻힌다 -- 물량 가중이어야 블록이 잡힌다.
    t2 = np.zeros((400, 1), np.float32)
    t2[:, 0] = 10.0
    t2[::2, 0] += 0.5                                     # 0.5 짜리 지터 199번
    t2[100, 0] += 500.0                                   # 500 짜리 블록 한 번
    b2 = row_stats(t2, dt_s=1)
    assert b2["blk"][0] > 100.0, b2["blk"][0]             # 지터에 안 묻힌다
    assert row_stats(t, dt_s=3)["refill"][1] == st["refill"][1]  # dt_s 는 d60 만 바꾼다

    # 접근행동: 가격이 다가올 때 «얇아지는 빈»과 «두꺼워지는 빈»을 갈라야 한다.
    # mid 를 위아래로 흔들어 같은 빈이 near/far 를 둘 다 겪게 만든다.
    T, NB = 400, 6
    q = np.zeros((T, NB), np.float32)
    # 🔴흔들 폭이 좁으면 far 구간(0.35~1.2%)을 아무 빈도 못 겪어 전부 NaN 이다.
    #   ±25달러(±1%)면 모든 빈이 near 와 far 를 둘 다 지난다.
    mv = 2500.0 + 25.0 * np.sin(np.arange(T) / 40.0)
    px = 2500.0 + np.arange(NB) * 0.5
    for j in range(T):
        d = np.abs(px - mv[j]) / mv[j] * 100.0
        q[j] = np.maximum(100.0 - 60.0 * d, 5.0)            # 거리가 멀수록 얇다(공통 성질)
        q[j, 1] *= 0.2 if d[1] < 0.2 else 1.0               # 1번 빈만 다가오면 빠진다
        q[j, 4] *= 3.0 if d[4] < 0.2 else 1.0               # 4번 빈만 다가오면 쌓는다
    ww = {"qty": q, "mid": mv.astype(np.float32), "n_bins": NB, "bin_lo": 5000,
          "bin_size": 0.5, "dt_s": 1}
    ar = approach_ratio(ww, min_obs_s=20.0)["ratio"]
    fin = np.isfinite(ar)
    assert fin.any(), "자격 빈이 하나도 안 나오면 창/임계가 잘못된 것이다"
    assert ar[1] < 1.0 < ar[4], (ar[1], ar[4])              # 빠지는 빈 < 1 < 쌓는 빈

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
        # 구멍 읽기: 파일 앞부분을 안 쓰고 늘리면 그 자리는 0 으로 읽힌다.
        # mid=0.0 은 isfinite 를 통과하므로 read_window 가 NaN 으로 바꿔줘야 한다.
        hole = _hour_path("holeusdt", t0, tmp)
        hole.parent.mkdir(parents=True, exist_ok=True)
        with open(hole, "wb") as fh:
            fh.write(HEADER.pack(MAGIC, VERSION, N_BINS, BIN_SIZE, 0, t0, 1000, 0))
            fh.truncate(HEADER.size + 3 * row_bytes())      # 0,1,2초가 구멍
            fh.seek(HEADER.size + 3 * row_bytes())
            fh.write(ROW_HEAD.pack(t0 + 3000, 1000, 2500.0))
            fh.write(np.zeros(N_BINS, np.float32).tobytes())
        rh = read_window("holeusdt", t0 + 3000, cols=4, root=tmp)
        assert np.isnan(rh["mid"][:3]).all(), rh["mid"][:3]   # 0.0 이 아니라 NaN 이어야
        assert abs(float(rh["mid"][3]) - 2500.0) < 1e-3

    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # ── vol_scores: 두 재료가 «각각» 반응하는가 ──────────────────────────────
    #   섞어서 하나로 내면 어느 쪽이 움직였는지 못 본다. 한 축씩 흔들어 본다.
    T, NB = 700, 40
    base = np.full((T, NB), 100.0, np.float32)
    flat = {"mid": np.full(T, 2500.0, np.float32), "qty": base.copy(), "dt_s": 1,
            "n_bins": NB, "bin_lo": int(2500.0 / BIN_SIZE) - NB // 2, "bin_size": BIN_SIZE}
    pa, rf = vol_scores(flat)
    assert pa.shape == (1,) and abs(float(pa[0])) < 1e-6, pa      # 안 움직였다
    assert float(rf[0]) < 1e-6, rf                                # 다시 깔린 것도 없다

    # 🔴가격 격자는 20칸($0.5 x 40)뿐이라 mid 를 크게 움직이면 밴드가 격자 밖으로 나가
    #   자격 미달로 «빠진다». 계단 하나로 «300초 전 대비»만 본다.
    step = {**flat, "mid": flat["mid"].copy()}
    step["mid"][400:] = 2502.0
    pa2, _ = vol_scores(step)
    assert 7.5 < float(pa2[0]) < 8.5, pa2         # |2502-2500|/2502 = 7.99bp

    churn = {**flat, "qty": base.copy()}
    churn["qty"][1::2] = 40.0                     # 매초 60 빠졌다 다시 깔린다
    _, rf2 = vol_scores(churn)
    assert float(rf2[0]) > 1.0, rf2               # 재깔림이 peak 합을 넘는다
    assert float(rf2[0]) > 100 * float(rf[0]) + 1.0

    # 자격 미달은 **빠진다** -- 0 으로 채우면 「조용하다」로 읽힌다
    gap = {**flat, "mid": flat["mid"].copy()}
    gap["mid"][-1] = np.nan
    assert vol_scores(gap)[0].size == 0
    assert vol_scores(flat, range(0, T, 100))[0].size == 1        # i<600 은 창이 모자란다

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
