#!/usr/bin/env python3
"""ETH 5m 선물 klines 아카이브 백필 2021-12~2023-12 (2026-08-23).

용도: 히스토리 확장 실험(wide24 HMM 학습창을 2022~로 연장) 원료. 캐노니컬 피쳐 프레임은
2024-01 시작이라 그 이전 OHLCV/taker가 리포에 없다 — data.binance.vision 월별 zip에서 수집.

출처: https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/5m/
출력: data/eth_5m_2021_2023_archive.csv (open_time은 bar 시작 ms — 캐노니컬 조인 시
      종료라벨 보정은 피쳐 빌더 쪽 책임, metrics 아카이브 +5분 보정과 동일 주의)
"""
from __future__ import annotations

import io
import sys
import zipfile
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/eth_5m_2021_2023_archive.csv"
URL = "https://data.binance.vision/data/futures/um/monthly/klines/ETHUSDT/5m/ETHUSDT-5m-{ym}.zip"
COLUMNS = ["open_time", "open", "high", "low", "close", "volume", "close_time", "quote_volume",
           "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
MONTHS = [f"2021-{m:02d}" for m in (12,)] + [f"{y}-{m:02d}" for y in (2022, 2023) for m in range(1, 13)]


def fetch_month(ym: str) -> pd.DataFrame | None:
    try:
        raw = urlopen(URL.format(ym=ym), timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                first = f.read(64).decode("utf-8", errors="replace")
            with z.open(z.namelist()[0]) as f:
                # 구형 zip은 헤더 없음, 신형은 헤더 포함 — 첫 필드가 숫자인지로 판별
                has_header = not first.split(",")[0].strip().isdigit()
                df = pd.read_csv(f, header=0 if has_header else None)
                df.columns = COLUMNS
                return df
    except Exception as e:  # noqa: BLE001 — 결측월은 리포트
        print(f"  MISS {ym}: {type(e).__name__}: {e}", flush=True)
        return None


def main() -> int:
    frames, missed = [], []
    for ym in MONTHS:
        df = fetch_month(ym)
        if df is None:
            missed.append(ym)
        else:
            frames.append(df)
            print(f"  {ym}: {len(df)} rows", flush=True)
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    merged.to_csv(OUT, index=False)

    ts = pd.to_datetime(merged["open_time"], unit="ms")
    grid = pd.date_range(ts.min(), ts.max(), freq="5min")
    print(f"\nrows={len(merged)} | {ts.min()} ~ {ts.max()}")
    print(f"5분 그리드 커버리지: {len(merged)/len(grid)*100:.2f}%")
    print(f"결측월: {missed}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
