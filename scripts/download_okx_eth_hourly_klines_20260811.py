"""신규 데이터소스 후보 4(거래소간 포지셔닝 괴리) 절반 -- 가격 basis. 원래 계획은 펀딩 스프레드
+ 가격 basis 둘 다였으나, 라이브 확인(2026-08-11) 결과 OKX `funding-rate-history` 공개
엔드포인트는 실제로 최근 1~2개월치만 주고 `since`를 사실상 무시한다(네이티브 API로 직접
`before` 커서 페이지네이션해도 더 과거로 안 넘어감) -- VAL/OOS 백필이 안 돼 펀딩 스프레드
절반은 보류. 반면 OKX 캔들(가격) 히스토리는 `since`가 정상 동작하고 2025-01-01까지 확인됨 --
가격 basis(Binance-OKX 퍼프 가격 괴리) 절반만 이걸로 진행한다.

해상도: 1시간봉. 온체인처럼 느리게 변하는 지표가 아니라 근실시간 시장 미시구조라 5분봉이 더
정확하겠지만, 300개/페이지 제한으로 5분봉 전체 기간(1.5년치, ~15.7만개 bar)을 받으려면
페이지 수가 지나치게 많아진다(vs 1시간봉 ~48페이지). "가장 싼 검증 스텝" 취지에 맞춰 1시간봉으로
먼저 싸게 검증 -- 신호가 있으면 그때 5분봉으로 정밀화."""
from __future__ import annotations

import time
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "data/derivatives/okx_eth_hourly_klines.csv"
START_DEFAULT = pd.Timestamp("2025-01-01", tz="UTC")
SYMBOL = "ETH/USDT:USDT"
TIMEFRAME = "1h"
PAGE_LIMIT = 300


def main() -> int:
    okx = ccxt.okx()

    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, parse_dates=["timestamp"])
        start = existing["timestamp"].max() + pd.Timedelta(hours=1)
    else:
        existing = pd.DataFrame()
        start = START_DEFAULT

    end = pd.Timestamp.now(tz="UTC")
    if start >= end:
        print(f"up-to-date: rows={len(existing)}")
        return 0

    rows = []
    since_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)
    while since_ms < end_ms:
        batch = okx.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, since=since_ms, limit=PAGE_LIMIT)
        if not batch:
            break
        rows.extend(batch)
        last_ts = batch[-1][0]
        if last_ts <= since_ms:
            break
        since_ms = last_ts + 1
        time.sleep(okx.rateLimit / 1000.0)

    if not rows:
        print(f"no-new-data: rows={len(existing)}")
        return 0

    new_df = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
    new_df["timestamp"] = pd.to_datetime(new_df["ts_ms"], unit="ms", utc=True)
    new_df = new_df[["timestamp", "open", "high", "low", "close", "volume"]]

    combined = pd.concat([existing, new_df], ignore_index=True) if len(existing) else new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    n_bad = int(combined[["open", "high", "low", "close"]].isna().any(axis=1).sum())
    if n_bad:
        raise RuntimeError(f"{n_bad} rows have non-finite OHLC values -- refusing to write")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH}: rows={len(combined)} (+{len(new_df)} new), "
          f"range={combined['timestamp'].min()}..{combined['timestamp'].max()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
