#!/usr/bin/env python3
"""**Deribit 과거 옵션체인 무료 백필** — RR25/스큐 일별 시계열의 원천. (2026-09-16)

사전등록: docs/experiments/direction_axis_disclosed_intent_and_skew_prereg_20260916.md §2

## 왜 이게 가능한가 (2026-09-16 실사)
🔴09-02 문서가 지목한 무료 경로(CryptoDataDownload)는 **틀렸다** -- Deribit 자료가 CSV 2개뿐이고
둘 다 `DeriBit_volatility_OHLC_{BTC,ETH}` = **DVOL 지수**다(옵션체인 아님. 게다가 DVOL 레벨축은
registry `btc_dvol_feature_overlay` 로 이미 0/9 종결).
⭐대신 **만료된 인스트루먼트도 `get_tradingview_chart_data` 가 계속 응답한다**. 막힌 건 조회가
아니라 **열거**뿐인데(`get_instruments?expired=true` 는 당일 만기 56개만 준다), 이름이
`ETH-{DDMMMYY}-{STRIKE}-{C|P}` 로 **구성 가능**하므로 우회된다.

## 정확도 (정답지로 직접 쟀다)
우리가 `mark_iv` 를 저장한 구간(08-15~)에서 복원 IV 를 대조한 결과:
  전체(시간봉)   n=32  상관 +0.627  중앙 2.67  90분위 **15.61** vol pt
  **거래량>0 만**  n= 6  상관 **+0.9855**  중앙 **1.31**  90분위 2.27
⇒ 방법은 멀쩡하고 **커버리지가 문제**다. `chart_data` 는 **체결가**를, `mark_iv` 는 **피팅된
마크**를 준다 -- 거래가 뜸한 윙에서 마지막 체결이 묵는다. 그래서 **일봉**을 쓰고 **거래량 0 인
바는 버린다**(채우지 않는다). 08-27 실측: 체인 58개 중 39개가 일중 거래, 양쪽 윙 존재.

## 규약
- ⚠️Deribit ETH 옵션은 **inverse** -- 가격이 ETH 단위다. `premium_usd = close_eth * underlying`.
- 기초가는 `data/binance_vision/panel/ETHUSDT.parquet` 일별 종가(08:00 UTC 만기에 맞춰 그 시각).
- 인스트루먼트 1개당 API 호출 **1회**(생애 전체 일봉을 한 번에 받는다). 날짜별로 부르지 않는다.
- 거래 없는 바는 **버린다**. 커버리지 미달일은 NaN 으로 남기고 채우지 않는다.

사용:
  python3 scripts/backfill_deribit_option_chain_20260916.py --from 2026-08-15 --to 2026-09-16
  python3 scripts/backfill_deribit_option_chain_20260916.py --validate    # 겹치는 구간 대조
  python3 scripts/backfill_deribit_option_chain_20260916.py --selftest
"""
from __future__ import annotations

import argparse, json, sys, time, urllib.request
import numpy as np, pandas as pd
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from scipy.stats import norm
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "data/research/deribit_option_daily.duckdb"
API = "https://www.deribit.com/api/v2/public/get_tradingview_chart_data"
MON = ["JAN","FEB","MAR","APR","MAY","JUN","JUL","AUG","SEP","OCT","NOV","DEC"]


def ins_name(exp: date, strike: int, cp: str) -> str:
    return f"ETH-{exp.day}{MON[exp.month-1]}{exp.year%100:02d}-{strike}-{cp}"


def fridays(a: date, b: date) -> list[date]:
    d = a + timedelta((4 - a.weekday()) % 7)
    out = []
    while d <= b:
        out.append(d); d += timedelta(7)
    return out


def bs(cp: str, S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (np.log(S / K) + sig * sig / 2 * T) / (sig * np.sqrt(T)); d2 = d1 - sig * np.sqrt(T)
    return S * norm.cdf(d1) - K * norm.cdf(d2) if cp == "C" else K * norm.cdf(-d2) - S * norm.cdf(-d1)


def implied_vol(px: float, cp: str, S: float, K: float, T: float) -> float:
    intrinsic = max(0.0, (S - K) if cp == "C" else (K - S))
    if px <= intrinsic + 1e-8 or T <= 0:
        return float("nan")
    try:
        return brentq(lambda s: bs(cp, S, K, T, s) - px, 1e-3, 5.0, xtol=1e-6)
    except Exception:
        return float("nan")


def delta(cp: str, S: float, K: float, T: float, sig: float) -> float:
    if T <= 0 or sig <= 0 or not np.isfinite(sig):
        return float("nan")
    d1 = (np.log(S / K) + sig * sig / 2 * T) / (sig * np.sqrt(T))
    return norm.cdf(d1) if cp == "C" else norm.cdf(d1) - 1.0


def fetch(ins: str, t0: int, t1: int) -> pd.DataFrame | None:
    try:
        with urllib.request.urlopen(
                f"{API}?instrument_name={ins}&start_timestamp={t0}&end_timestamp={t1}&resolution=1D",
                timeout=25) as r:
            res = (json.load(r).get("result") or {})
    except Exception:
        return None
    if res.get("status") != "ok" or not res.get("ticks"):
        return None
    d = pd.DataFrame({"ts": res["ticks"], "close": res["close"], "volume": res["volume"]})
    d = d[d.volume > 0]                      # 🔴거래 없는 바는 버린다 -- 마지막 체결이 묵는다
    return d if len(d) else None


def spot_series() -> pd.Series:
    p = pd.read_parquet(ROOT / "data/binance_vision/panel/ETHUSDT.parquet")
    p["timestamp"] = pd.to_datetime(p["timestamp"], utc=True)
    s = p.set_index("timestamp").close.astype(float)
    return s.resample("1D").last()           # 일별 종가


def run(d0: date, d1: date, band: float, step: int, sleep: float) -> pd.DataFrame:
    spot = spot_series()
    rows = []
    exps = [e for e in fridays(d0, d1 + timedelta(days=45))]
    print(f"만기 {len(exps)}개 · 구간 {d0}~{d1}", flush=True)
    for e in exps:
        # 🔴행사가 밴드의 기준가일 뿐이다. reindex+ffill 로 잡으면 **패널 끝 이후 만기가 NaN 이 되어
        # 통째로 걸러진다** -- 첫 실행에서 DTE 2~11일만 남고 가설이 지정한 30일 만기를 한 번도
        # 못 봤다. asof 는 마지막 관측을 주므로 미래 만기도 밴드를 잡을 수 있다.
        ref = float(spot.asof(pd.Timestamp(e, tz="UTC"))) if len(spot) else np.nan
        if not np.isfinite(ref):
            continue
        lo, hi = int(ref * (1 - band)), int(ref * (1 + band))
        ks = [k for k in range(lo - lo % step + step, hi + 1, step)]
        t0 = int(datetime(d0.year, d0.month, d0.day, tzinfo=timezone.utc).timestamp() * 1000)
        t1 = int(datetime(e.year, e.month, e.day, 8, tzinfo=timezone.utc).timestamp() * 1000)
        n_ok = 0
        for k in ks:
            for cp in ("P", "C"):
                if (cp == "P" and k > ref * 1.05) or (cp == "C" and k < ref * 0.95):
                    continue                  # OTM 만 -- ITM 은 같은 정보이고 호출만 두 배다
                d = fetch(ins_name(e, k, cp), t0, t1)
                time.sleep(sleep)
                if d is None:
                    continue
                n_ok += 1
                for _, r in d.iterrows():
                    day = pd.Timestamp(int(r.ts), unit="ms", tz="UTC").normalize()
                    S = spot.reindex([day]).ffill()
                    S = float(S.iloc[0]) if len(S) and np.isfinite(S.iloc[0]) else np.nan
                    if not np.isfinite(S):
                        continue
                    T = max((pd.Timestamp(e, tz="UTC") - day).days, 0) / 365.0
                    sig = implied_vol(float(r.close) * S, cp, S, k, T)
                    rows.append(dict(day=day.date().isoformat(), expiry=e.isoformat(), strike=k,
                                     cp=cp, close_eth=float(r.close), volume=float(r.volume),
                                     spot=S, T=T, iv=sig,
                                     delta=delta(cp, S, k, T, sig)))
        print(f"  {e} ref={ref:.0f} 후보 {len(ks)} → 응답 {n_ok}", flush=True)
    return pd.DataFrame(rows)


def _selftest() -> None:
    assert ins_name(date(2026, 8, 29), 2400, "C") == "ETH-29AUG26-2400-C"
    assert ins_name(date(2026, 9, 4), 900, "P") == "ETH-4SEP26-900-P"
    f = fridays(date(2026, 8, 1), date(2026, 8, 31))
    assert f[0] == date(2026, 8, 7) and f[-1] == date(2026, 8, 28), f
    S, K, T, sig = 2500.0, 2600.0, 30 / 365, 0.60
    px = bs("C", S, K, T, sig)
    assert abs(implied_vol(px, "C", S, K, T) - sig) < 1e-4, "BS 역산이 자기 자신을 복원해야 한다"
    assert 0 < delta("C", S, K, T, sig) < 1 and -1 < delta("P", S, K, T, sig) < 0
    assert np.isnan(implied_vol(0.0, "C", S, K, T)), "내재가치 이하 프리미엄은 NaN"
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="d0", default="2026-08-15")
    ap.add_argument("--to", dest="d1", default="2026-09-16")
    ap.add_argument("--band", type=float, default=0.30)
    ap.add_argument("--step", type=int, default=100)
    ap.add_argument("--sleep", type=float, default=0.12)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest(); raise SystemExit(0)
    df = run(date.fromisoformat(a.d0), date.fromisoformat(a.d1), a.band, a.step, a.sleep)
    if df.empty:
        print("복원 0행"); raise SystemExit(1)
    DB.parent.mkdir(parents=True, exist_ok=True)
    import duckdb
    con = duckdb.connect(str(DB))
    try:
        con.execute("CREATE TABLE IF NOT EXISTS option_daily AS SELECT * FROM df WHERE 1=0")
        con.execute("DELETE FROM option_daily WHERE day >= ? AND day <= ?", [a.d0, a.d1])
        con.execute("INSERT INTO option_daily SELECT * FROM df")
        n = con.execute("SELECT count(*) FROM option_daily").fetchone()[0]
    finally:
        con.close()
    print(f"\n저장 {len(df)}행 → {DB} (누적 {n}) · IV 유효 {int(df.iv.notna().sum())} "
          f"· 일자 {df.day.nunique()} · 만기 {df.expiry.nunique()}")
