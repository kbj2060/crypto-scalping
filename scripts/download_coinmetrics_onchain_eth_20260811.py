"""Odyssey 신규 데이터소스 후보 6(ETH 온체인 순유입/공급): scripts/download_coinmetrics_onchain_20260804.py(BTC 전용)와
동일한 CoinMetrics Community API를 ETH로 확장. 원본 스크립트는 건드리지 않는다(BTC 쪽 다른 의존자가 있을 수 있음).

라이브 확인(2026-08-11): ETH는 `HashRate`가 전부 null(2022-09 The Merge로 PoS 전환 -- 개념 자체가 없음), 나머지
6개(AdrActCnt/CapMVRVCur/FlowInExNtv/FlowOutExNtv/SplyExNtv/TxCnt)는 정상 반환 확인. 원본 BTC 리스트에서
HashRate만 제외.

캐비어트: 응답에 `<metric>-status: "flash"`와 `-status-time`이 붙는다 -- CoinMetrics가 나중에 값을 리비전할 수
있다는 뜻이고, 이 raw 다운로드는 "지금 기준 최신값"만 받아온다(과거 특정 시점에 실제로 게시돼 있던 스냅샷이
아님). 무료tier에 리비전 히스토리 엔드포인트가 없어 완전한 인과성 보장은 안 됨 -- 진단 스크립트에서 최소
1일 지연을 두는 것과 별개로 이 한계를 감안해야 한다."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data/onchain/coinmetrics"
OUT_PATH = OUT_DIR / "eth_onchain_daily.csv"
URL = "https://community-api.coinmetrics.io/v4/timeseries/asset-metrics"
METRICS = ["AdrActCnt", "CapMVRVCur", "FlowInExNtv", "FlowOutExNtv", "SplyExNtv", "TxCnt"]
START_DEFAULT = pd.Timestamp("2025-01-01", tz="UTC")  # h48qual TRAIN 시작(2025-01)보다 이르게, 여유 포함
PAGE_SIZE = 10000


def fetch(start: pd.Timestamp, end: pd.Timestamp) -> list[dict]:
    resp = requests.get(URL, params={
        "assets": "eth",
        "metrics": ",".join(METRICS),
        "frequency": "1d",
        "start_time": start.strftime("%Y-%m-%d"),
        "end_time": end.strftime("%Y-%m-%d"),
        "page_size": PAGE_SIZE,
    }, timeout=30)
    if resp.status_code != 200:
        raise RuntimeError(f"CoinMetrics API error {resp.status_code}: {resp.text[:300]}")
    payload = resp.json()
    if "next_page_token" in payload:
        raise RuntimeError("unexpected pagination -- range exceeded PAGE_SIZE, widen PAGE_SIZE or add a pagination loop")
    return payload["data"]


def main() -> int:
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH)
        existing["time"] = pd.to_datetime(existing["time"])
        start = existing["time"].max() + pd.Timedelta(days=1)
    else:
        existing = pd.DataFrame()
        start = START_DEFAULT

    end = pd.Timestamp.now(tz="UTC")
    if start >= end:
        print(f"up-to-date: rows={len(existing)}")
        return 0

    raw = fetch(start, end)
    if not raw:
        print(f"no-new-data: rows={len(existing)}")
        return 0

    new_df = pd.DataFrame(raw)
    new_df["time"] = pd.to_datetime(new_df["time"])
    for col in METRICS:
        new_df[col] = pd.to_numeric(new_df[col], errors="coerce")
    new_df = new_df[["time"] + METRICS]

    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)

    n_bad = int(combined[METRICS].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} rows have non-finite on-chain values -- refusing to write")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: rows={len(combined)} (+{len(new_df)} new), "
          f"range={combined['time'].min()}..{combined['time'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
