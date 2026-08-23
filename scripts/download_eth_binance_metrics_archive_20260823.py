#!/usr/bin/env python3
"""ETH Binance metrics 아카이브 백필 (2026-08-23).

배경: 캐노니컬 ETH 피쳐는 아카이브 8컬럼 중 3개(sum_open_interest_value/
sum_toptrader_long_short_ratio/count_long_short_ratio)만 포함하며 값은 아카이브와
완전일치(2025년 겹침구간 99.8~100% 실측). BTC는 metrics4 확장판
(btc_features_2024_2026_metrics4_20260802.csv)으로 빠진 2컬럼까지 이미 확보했으나 ETH는
없다 — 이 비대칭을 해소하기 위해 전체 8컬럼을 2024-01-01~최신까지 아카이브에서 받아
병합 저장한다. 2022~2023은 아카이브에 존재하지만(2021-12-01~) TRAIN 규약이 2024-01
시작이라 이번엔 받지 않는다.

출처: https://data.binance.vision/data/futures/um/daily/metrics/ETHUSDT/
출력: data/TOTAL_ETHUSDT_metrics_2024_2026.csv (기존 수동본 TOTAL_ETHUSDT_metrics.csv는
      무수정 보존 — 2025-01~2026-01 구간 교차검증용으로 그대로 유효)
"""
from __future__ import annotations

import io
import sys
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import os
SYMBOL = os.environ.get("METRICS_SYMBOL", "ETHUSDT").upper()   # 2026-08-23: BTC/SOL 백필용 파라미터화
# 2026-08-23(2차): 히스토리 확장 실험용 기간 파라미터화 — 기본값은 기존 동작과 동일
_OUT_LABEL = os.environ.get("METRICS_OUT_LABEL", "2024_2026")
OUT = ROOT / f"data/TOTAL_{SYMBOL}_metrics_{_OUT_LABEL}.csv"
URL = "https://data.binance.vision/data/futures/um/daily/metrics/" + SYMBOL + "/" + SYMBOL + "-metrics-{d}.zip"
START = date.fromisoformat(os.environ.get("METRICS_START", "2024-01-01"))
_END_ENV = os.environ.get("METRICS_END", "")
END = date.fromisoformat(_END_ENV) if _END_ENV else date.today() - timedelta(days=1)   # 아카이브는 전일까지 게시됨


def fetch_day(d: date) -> pd.DataFrame | None:
    ds = d.isoformat()
    try:
        raw = urlopen(URL.format(d=ds), timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            with z.open(z.namelist()[0]) as f:
                return pd.read_csv(f)
    except Exception as e:  # noqa: BLE001 — 결측일은 수집 후 리포트
        print(f"  MISS {ds}: {type(e).__name__}", flush=True)
        return None


def main() -> int:
    days = []
    d = START
    while d <= END:
        days.append(d)
        d += timedelta(days=1)
    print(f"downloading {len(days)} days ({START} ~ {END}) ...", flush=True)

    frames: dict[str, pd.DataFrame] = {}
    missed: list[str] = []
    with ThreadPoolExecutor(max_workers=12) as pool:
        futs = {pool.submit(fetch_day, d): d for d in days}
        done = 0
        for fut in as_completed(futs):
            d = futs[fut]
            df = fut.result()
            if df is None:
                missed.append(d.isoformat())
            else:
                frames[d.isoformat()] = df
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(days)}", flush=True)

    merged = pd.concat([frames[k] for k in sorted(frames)], ignore_index=True)
    merged["create_time"] = pd.to_datetime(merged["create_time"])
    # ⚠️ 타임스탬프 컨벤션 보정 (+5분): 아카이브 원본 create_time은 버킷 "시작" 시각인데,
    # 이 리포의 기존 수동본(TOTAL_ETHUSDT_metrics.csv)과 캐노니컬 피쳐는 버킷 "종료" 시각
    # 라벨링을 쓴다 — 2026-08-23 실측: 원본 그대로는 겹침구간 완전일치 0%, +5분 shift 시
    # 100.00% 완전일치(105,350행). 보정 없이 캐노니컬에 timestamp 조인하면 1-bar
    # 미래참조(lookahead)가 생기므로 반드시 이 보정을 유지할 것.
    merged["create_time"] = merged["create_time"] + pd.Timedelta(minutes=5)
    merged = merged.drop_duplicates(subset=["create_time"]).sort_values("create_time").reset_index(drop=True)
    merged.to_csv(OUT, index=False)

    # 무결성 리포트
    full_grid = pd.date_range(merged.create_time.min(), merged.create_time.max(), freq="5min")
    coverage = len(merged) / len(full_grid)
    print(f"\nrows={len(merged)} | {merged.create_time.min()} ~ {merged.create_time.max()}")
    print(f"5분 그리드 커버리지: {coverage*100:.2f}% (아카이브 자체가 일부 버킷 미게시 — 원천 특성)")
    print(f"결측일: {len(missed)}개 {missed[:10]}")
    print(f"컬럼: {merged.columns.tolist()}")
    print(f"-> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
