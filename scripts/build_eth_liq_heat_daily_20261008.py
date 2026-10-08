"""ETH 일봉 청산 히트맵 소급 빌더 (2026-10-08, 사용자 «일봉에도 청산 히트맵이 보이게»).

하루 한 열 = 그날 UTC 마감 시점의 «살아 있는 추정 청산 밀도» -- dashboard.footprint_daily.heat_for_day
(= 라이브 청산맵과 같은 compute_spliced_levels, 입력 창 HEAT_LOOKBACK_H=168 · 마감 뒤 봉은 안 봄).
원천: data.binance.vision USDⓈ-M 1시간봉(2019-12~, 월 파일 · 아직 월 파일이 없는 달은 일 파일). 바이낸스 REST 아님.
산출: data/footprint_daily/ETHUSDT_liqheat.parquet  day(date) · price(칸 가운데) · w(0~1, 쪽마다 최대 대비) · s(지도 전체 중 몫, 합 1)
실행: python scripts/build_eth_liq_heat_daily_20261008.py [--selftest]
"""
from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dashboard.footprint_daily import HEAT_LOOKBACK_H, heat_for_day  # noqa: E402

MAIN = Path("/home/kbj20/crypto-scalping")
OUT = MAIN / "data/footprint_daily"
WORK = MAIN / "tmp/fp_daily_build/k1h"
BASE = "https://data.binance.vision/data/futures/um"
SYM = "ETHUSDT"


def _zip_rows(url: str) -> list[list[str]]:
    try:
        raw = urllib.request.urlopen(url, timeout=60).read()
    except urllib.error.HTTPError:
        return []
    z = zipfile.ZipFile(io.BytesIO(raw))
    return [r.split(",") for r in z.read(z.namelist()[0]).decode().splitlines() if r[:1].isdigit()]


def load_1h() -> pd.DataFrame:
    """2019-12 부터 어제까지 1시간봉(지난 달은 로컬 캐시)."""
    WORK.mkdir(parents=True, exist_ok=True)
    parts, m, last = [], pd.Period("2019-12", "M"), pd.Period(date.today(), "M")
    while m <= last:
        f = WORK / f"{m}.parquet"
        if f.exists() and m != last:
            parts.append(pd.read_parquet(f)); m += 1; continue
        rows = _zip_rows(f"{BASE}/monthly/klines/{SYM}/1h/{SYM}-1h-{m}.zip") if m != last else []
        if not rows:                               # 월 파일이 아직 없는 달 → 일 파일
            d = m.start_time.date()
            while d < min(m.end_time.date() + timedelta(days=1), date.today()):
                rows += _zip_rows(f"{BASE}/daily/klines/{SYM}/1h/{SYM}-1h-{d}.zip")
                d += timedelta(days=1)
        df = pd.DataFrame({"t": [int(r[0]) for r in rows], "high": [float(r[2]) for r in rows], "low": [float(r[3]) for r in rows],
                           "close": [float(r[4]) for r in rows], "volume": [float(r[5]) for r in rows]})
        if m != last and len(df):
            df.to_parquet(f)
        parts.append(df); m += 1
    k = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    k["timestamp"] = pd.to_datetime(k.t, unit="ms", utc=True)
    return k


def build(k: pd.DataFrame) -> pd.DataFrame:
    rows, d = [], date(2020, 1, 1)
    end = (k.timestamp.max() + pd.Timedelta(hours=1)).date()      # 마감된 마지막 날까지
    while d < end:
        rows += [(d, p, w, s) for p, w, s in heat_for_day(k, d)]
        d += timedelta(days=1)
    return pd.DataFrame(rows, columns=["day", "price", "w", "s"])


def selftest() -> None:
    import numpy as np
    ts = pd.date_range("2026-01-01", periods=24 * 9, freq="1h", tz="UTC")
    px = 100 + np.sin(np.arange(len(ts)) / 5.0)
    k = pd.DataFrame({"timestamp": ts, "high": px + 0.4, "low": px - 0.4, "close": px, "volume": np.full(len(ts), 10.0)})
    a = heat_for_day(k, date(2026, 1, 8))
    k2 = k.copy()
    k2.loc[k2.timestamp >= pd.Timestamp("2026-01-09", tz="UTC"), ["high", "low", "close"]] *= 1.5   # 마감 뒤 봉을 바꿔도
    assert a and heat_for_day(k2, date(2026, 1, 8)) == a, "마감 뒤 봉을 봤다(미래참조)"
    assert all(0 < w <= 1 and s > 0 for _, w, s in a) and abs(sum(s for *_, s in a) - 1) < 1e-9   # 몫 합 = 1
    from scripts.live_liquidation_map_20260824 import compute_spliced_levels       # w = 라이브 heatmap_bins 그대로
    end = pd.Timestamp("2026-01-09", tz="UTC")
    win = k[(k.timestamp < end) & (k.timestamp >= end - pd.Timedelta(hours=HEAT_LOOKBACK_H))].reset_index(drop=True)
    ref = compute_spliced_levels(win, float(win.close.iloc[-1]))["heatmap_bins"]
    assert [(b["price"], b["weight_pct"]) for b in ref if b["weight_pct"] > 0] == [(p, w) for p, w, _ in a]
    assert heat_for_day(k, date(2025, 12, 31)) == []         # 창에 봉이 없으면 없음
    print(f"selftest OK -- 마감 뒤 봉 무시(인과) · w 0~1 · 창 {HEAT_LOOKBACK_H}h")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        k = load_1h()
        h = build(k)
        OUT.mkdir(parents=True, exist_ok=True)
        h.to_parquet(OUT / f"{SYM}_liqheat.parquet", index=False)
        print("1시간봉", len(k), k.timestamp.min(), "~", k.timestamp.max(), "· 히트맵 일", h.day.nunique(), "칸", len(h))
