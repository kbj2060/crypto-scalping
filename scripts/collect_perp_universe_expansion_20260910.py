#!/usr/bin/env python3
"""무기한 종목 확대 수집 — klines 5분봉 + metrics (2026-09-10).

사용자: *"우선 상위 10종으로 진행해줘"* — [쏠림 페이드](../docs/xsec_crowding_fade_fill_shortfall_20260910.md)
의 다음 단계. 횡단면 전략에서 종목을 늘리면 **매 시점 선택 정밀도**가 올라간다.

## 채택 기준 (결과 전 고정)
① 패널 60종 밖 ② **2024-01-01 이전 상장**(패널 시작 2023-12-31 이전 이력 필요)
③ 24시간 달러거래량 상위순
⚠️②를 안 걸면 토큰화 주식·원자재(XAU/XAG/SPCX/SKHYNIX/SOXL 등)가 상위를 채운다 —
   거래량은 크지만 이력이 없고, **암호자산 횡단면의 성격 자체를 바꾼다**.

## 산출물은 기존 배치와 **정확히 같은 형식**이라야 한다
klines  `binance_data/klines/{SYM}/{SYM}-5m-api.csv`
        헤더 `timestamp,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote,ignore`
        ⚠️아카이브는 `open_time`(ms)·`count`·`taker_buy_volume`·`taker_buy_quote_volume` 이라 **개명·변환 필수**.
metrics `binance_data/metrics/{SYM}-metrics-YYYY-MM-DD.zip` (원본 zip 그대로)

사용법 `python3 scripts/collect_perp_universe_expansion_20260910.py SYM1 SYM2 ...`
"""
from __future__ import annotations

import io
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path("/home/kbj20/crypto-scalping")
KDIR = ROOT / "binance_data/klines"
MDIR = ROOT / "binance_data/metrics"
BASE = "https://data.binance.vision/data/futures/um"
START, END = pd.Timestamp("2023-12-01"), pd.Timestamp("2026-08-31")
UA = {"User-Agent": "research/1.0"}
WORKERS = 8

# 아카이브 → 저장소 컬럼명
RENAME = {"open_time": "timestamp", "count": "trades",
          "taker_buy_volume": "taker_buy_base", "taker_buy_quote_volume": "taker_buy_quote"}
COLS = ["timestamp", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def log(m):
    print(f"[collect {time.strftime('%H:%M:%S')}] {m}", flush=True)


def get(url: str, tries=3):
    """없으면 None(상장 전 구간은 정상적으로 404 다). 오류만 재시도."""
    for i in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(1.5 * (i + 1))
        except Exception:
            time.sleep(1.5 * (i + 1))
    return None


def klines(sym: str) -> int:
    out = KDIR / sym / f"{sym}-5m-api.csv"
    if out.exists():
        log(f"  {sym} klines 이미 있음 — 건너뜀")
        return 0
    months = pd.date_range(START, END, freq="MS").strftime("%Y-%m").tolist()
    parts = {}
    with ThreadPoolExecutor(WORKERS) as ex:
        fs = {ex.submit(get, f"{BASE}/monthly/klines/{sym}/5m/{sym}-5m-{m}.zip"): m for m in months}
        for f in as_completed(fs):
            b = f.result()
            if not b:
                continue
            try:
                z = zipfile.ZipFile(io.BytesIO(b))
                parts[fs[f]] = pd.read_csv(io.BytesIO(z.read(z.namelist()[0])))
            except Exception:
                pass
    if not parts:
        log(f"  🔴 {sym} klines 0개월 — 수집 실패")
        return 0
    d = pd.concat([parts[m] for m in sorted(parts)], ignore_index=True).rename(columns=RENAME)
    d["timestamp"] = pd.to_datetime(d["timestamp"], unit="ms")
    d = d.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    for c in COLS:
        if c not in d.columns:
            d[c] = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    d[COLS].to_csv(out, index=False)
    log(f"  ✅ {sym} klines {len(parts)}개월 · {len(d):,}봉 · {d.timestamp.min()} → {d.timestamp.max()}")
    return len(d)


def metrics(sym: str) -> int:
    days = pd.date_range(START, END, freq="D").strftime("%Y-%m-%d").tolist()
    todo = [x for x in days if not (MDIR / f"{sym}-metrics-{x}.zip").exists()]
    if not todo:
        log(f"  {sym} metrics 이미 완비")
        return 0
    MDIR.mkdir(parents=True, exist_ok=True)
    n = 0
    with ThreadPoolExecutor(WORKERS) as ex:
        fs = {ex.submit(get, f"{BASE}/daily/metrics/{sym}/{sym}-metrics-{x}.zip"): x for x in todo}
        for f in as_completed(fs):
            b = f.result()
            if not b:
                continue
            (MDIR / f"{sym}-metrics-{fs[f]}.zip").write_bytes(b)
            n += 1
    log(f"  ✅ {sym} metrics {n}/{len(todo)}일 수집")
    return n


def main(argv) -> int:
    syms = [s.upper() for s in argv[1:]]
    if not syms:
        log("종목을 인자로 넘겨라"); return 2
    log(f"대상 {len(syms)}종: {' '.join(syms)}")
    for i, s in enumerate(syms, 1):
        log(f"[{i}/{len(syms)}] {s}")
        klines(s)
        metrics(s)
    log("완료 — 다음: 패널 재빌드")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
