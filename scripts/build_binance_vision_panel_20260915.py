#!/usr/bin/env python3
"""**data.binance.vision 5분 패널 빌더** — 자산·기간을 동시에 늘린다. (2026-09-15)

왜: `direction_event_trigger_expansion_and_oos_audit_20260915` 의 결론이 「E|r| 게이트가 완전
표본외에서 부호를 뒤집지만(−4.32→+1.75bp/일 · 0/9→7/9) **CI 가 0 을 포함한다**」였고, 병목을
**검정력**으로 특정했다. 늘릴 수 있는 것이 둘 실측으로 확인됐다:
  · **기간** — `metrics`(OI/LSR) 가 **2022-01 부터** 있다(2021 은 404). 965일 → **1,695일 = 1.76배**
  · **자산** — LTC·DOT·TRX·ATOM·NEAR·FIL·APT·ARB·OP·INJ·SUI **11종** 전부 klines+metrics 존재
    (APT/ARB/INJ/SUI 는 2023 부터 — 상장일이 달라 **불균형 패널**이고 그대로 둔다)

🔴**생존편향은 더 나빠진다**: 20자산 전부 «지금도 상장돼 있다»
([[xsec_crowding_universe_survivorship_20260910]]). 「자산 무관」이라고 쓰지 않는다 —
쓸 수 있는 진술은 「현재 상장된 주요 20자산에서」까지다.

원본 zip 은 저장하지 않는다. 자산마다 `data/binance_vision/panel/{SYM}.parquet` 하나로 합친다
(일 단위 파편 1.8만개 대신 20개 — 연구 스크립트가 통째로 읽는다).
"""
from __future__ import annotations

import argparse
import io
import sys
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/binance_vision/panel"
BASE = "https://data.binance.vision/data/futures/um/daily"
KCOLS = ["open_time", "high", "low", "close", "volume", "count", "taker_buy_volume"]
MCOLS = ["sum_open_interest", "count_toptrader_long_short_ratio",
         "sum_toptrader_long_short_ratio", "count_long_short_ratio",
         "sum_taker_long_short_vol_ratio"]
NEW11 = ["LTC", "DOT", "TRX", "ATOM", "NEAR", "FIL", "APT", "ARB", "OP", "INJ", "SUI"]
OLD9 = ["ETH", "BTC", "SOL", "BNB", "XRP", "DOGE", "ADA", "AVAX", "LINK"]


def fetch(url: str) -> str | None:
    """없는 날(404)은 조용히 건너뛴다 — 상장 전이거나 거래소 휴지일이다."""
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            z = zipfile.ZipFile(io.BytesIO(r.read()))
            return z.read(z.namelist()[0]).decode()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise
    except Exception:
        return None


def day(sym: str, d: str) -> pd.DataFrame | None:
    k = fetch(f"{BASE}/klines/{sym}/5m/{sym}-5m-{d}.zip")
    if k is None:
        return None
    kd = pd.read_csv(io.StringIO(k))
    if "open_time" not in kd.columns:          # 2023 이전 일부 파일은 헤더가 없다
        kd = pd.read_csv(io.StringIO(k), header=None, names=[
            "open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "count", "taker_buy_volume", "taker_buy_quote_volume", "ignore"])
    kd = kd[KCOLS].copy()
    kd["timestamp"] = pd.to_datetime(kd["open_time"], unit="ms")
    kd = kd.drop(columns=["open_time"]).rename(
        columns={"count": "trades", "taker_buy_volume": "taker_buy_base"})
    m = fetch(f"{BASE}/metrics/{sym}/{sym}-metrics-{d}.zip")
    if m is not None:
        md = pd.read_csv(io.StringIO(m))
        if "create_time" in md.columns:
            md["timestamp"] = pd.to_datetime(md["create_time"])
            kd = kd.merge(md[["timestamp"] + [c for c in MCOLS if c in md.columns]],
                          on="timestamp", how="left")
    for c in MCOLS:                             # 메트릭 없는 날은 NaN 으로 남긴다
        if c not in kd.columns:
            kd[c] = float("nan")
    return kd


def build(sym: str, start: str, end: str, workers: int) -> int:
    s = f"{sym}USDT"
    dates = [d.strftime("%Y-%m-%d") for d in pd.date_range(start, end, freq="D")]
    with ThreadPoolExecutor(workers) as ex:
        parts = [p for p in ex.map(lambda d: day(s, d), dates) if p is not None]
    if not parts:
        print(f"  [{sym}] 파일 0 — 건너뜀")
        return 0
    df = pd.concat(parts).drop_duplicates("timestamp").sort_values("timestamp")
    df = df.reset_index(drop=True)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT / f"{s}.parquet")
    oi = df["sum_open_interest"].notna().mean()
    print(f"  [{sym}] {len(df):,}행 · {df.timestamp.min().date()}~{df.timestamp.max().date()} "
          f"· OI 결측아님 {oi:.1%} · {(OUT/f'{s}.parquet').stat().st_size/1e6:.1f}MB")
    return len(df)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", default="new11", help="new11 | old9 | all20 | 쉼표 목록")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-08-22")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    sel = {"new11": NEW11, "old9": OLD9, "all20": OLD9 + NEW11}.get(
        a.assets, a.assets.upper().split(","))
    print(f"자산 {len(sel)} · {a.start} ~ {a.end} · 스레드 {a.workers}")
    tot = 0
    for s in sel:
        tot += build(s, a.start, a.end, a.workers)
    print(f"\n합계 {tot:,}행 → {OUT}")
    return 0


def _selfcheck() -> None:
    """실제 한 날을 받아 형식·정렬을 확인한다(네트워크 필요)."""
    d = day("LTCUSDT", "2024-01-15")
    assert d is not None and len(d) == 288, f"5분봉 288개가 아니다: {None if d is None else len(d)}"
    assert d["sum_open_interest"].notna().all(), "메트릭 병합 실패"
    assert (d["timestamp"].diff().dropna() == pd.Timedelta("5min")).all(), "격자 불연속"
    assert d["high"].ge(d["close"]).all() and d["low"].le(d["close"]).all(), "OHLC 불일치"
    # 상장 전 날짜는 None 이어야 한다(404 를 예외로 터뜨리지 않는다)
    assert day("SUIUSDT", "2022-06-15") is None, "상장 전인데 데이터가 왔다"
    print("자체점검 통과")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck(); raise SystemExit(0)
    raise SystemExit(main())
