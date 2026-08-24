#!/usr/bin/env python3
"""spot-perp basis 서브프로젝트용 신규 데이터: ETHUSDT SPOT 5분봉 전체 이력 수집(2024-01-01~
현재). 기존 저장소는 fapi.binance.com(선물/perp)만 수집해왔음(scripts/extend_klines_20260713.py
등에서 확인) -- api.binance.com(현물) 컬럼은 이 프로젝트에 아예 없었음, 이번이 최초 수집.
계정 자격증명 미사용, extend_klines_20260713.py와 동일하게 공개 REST만 사용(라이브 트레이딩
계정 상태 건드리지 않는다는 그 스크립트의 설계 원칙 유지)."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
INTERVAL = "5m"
BASE_URL = "https://api.binance.com/api/v3/klines"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]


def fetch_klines(symbol: str, interval: str, start_ms: int, end_ms: int) -> list[list]:
    rows: list[list] = []
    cursor = start_ms
    while cursor < end_ms:
        params = {"symbol": symbol, "interval": interval, "startTime": cursor, "endTime": end_ms, "limit": 1000}
        resp = requests.get(BASE_URL, params=params, timeout=30)
        if resp.status_code != 200:
            raise RuntimeError(f"Binance spot API error {resp.status_code}: {resp.text[:500]}")
        batch = resp.json()
        if not isinstance(batch, list):
            raise RuntimeError(f"Unexpected Binance spot API response: {batch}")
        if not batch:
            break
        rows.extend(batch)
        last_open_time = batch[-1][0]
        cursor = last_open_time + 1
        if len(rows) % 20000 < 1000:
            print(f"  ...{len(rows):,}행 누적, 최신={pd.to_datetime(last_open_time, unit='ms')}", flush=True)
        if len(batch) < 1000:
            break
        time.sleep(0.15)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True, choices=["ETHUSDT", "BTCUSDT"])
    ap.add_argument("--start", default="2024-01-01")
    args = ap.parse_args()
    symbol = args.symbol

    out_dir = ROOT / "binance_data/klines_spot" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{symbol}-5m-spot.csv"

    start_ms = int(pd.Timestamp(args.start, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
    print(f"{symbol} spot 5m klines 수집: {args.start} ~ now", flush=True)

    raw = fetch_klines(symbol, INTERVAL, start_ms, end_ms)
    if not raw:
        raise RuntimeError(f"{symbol}: no klines returned at all -- check symbol/endpoint")

    df = pd.DataFrame(raw, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    for c in ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_base", "taker_buy_quote"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["trades"] = pd.to_numeric(df["trades"], errors="coerce").astype("Int64")
    df = df[["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "trades", "taker_buy_base", "taker_buy_quote"]]
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_bad = int(df[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad}개 행에 결측 OHLC -- 오염된 파일을 쓰지 않음")

    expected_bars = int((df["timestamp"].max() - df["timestamp"].min()).total_seconds() / 300) + 1
    coverage = len(df) / expected_bars
    print(f"{symbol}: {len(df):,}행, {df['timestamp'].min()} ~ {df['timestamp'].max()}, "
          f"5분간격 커버리지={coverage:.1%}(gap 존재시 100% 미만)", flush=True)

    df.to_csv(out_path, index=False)
    print(f"[저장] {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
