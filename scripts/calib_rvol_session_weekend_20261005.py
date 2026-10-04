"""«오늘 거래량» 배지 주말 보정 상수 재측정 (2026-10-05, 분기마다 1회).

live_eth_breakout_detector_20260911.py 의 RVOL_WEEKEND_MED · RVOL_WEEKEND_LO/HI 를 다시 잰다.
세션 RVOL = 오늘(UTC) 누적 거래대금 / 같은 5분 슬롯 누적의 직전 14일 중앙값(워커 _rvol 과 같은 식).
주말(UTC 토·일) 모든 슬롯 값의 중앙·q25·q75 를 최근 기간에서 낸다. 거래대금 = 1분봉 v × c(아카이브에 quote 없음).
  python scripts/calib_rvol_session_weekend_20261005.py            # 기본: 최근 21개월
  python scripts/calib_rvol_session_weekend_20261005.py 2025-01-01 2026-09-30
  python scripts/calib_rvol_session_weekend_20261005.py --selftest
"""
from __future__ import annotations

import glob
import sys

import numpy as np
import pandas as pd

K1M = "/home/kbj20/crypto-scalping/data/binance_vision/klines1m/ETHUSDT-1m-*.parquet"


def session_rvol(m5: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """5분 거래대금 → (일 × 슬롯 세션 RVOL 표, 일별 주말 여부)."""
    day = m5.index.floor("D")
    P = pd.DataFrame({"cum": m5.groupby(day).cumsum().values, "slot": m5.index.hour * 12 + m5.index.minute // 5,
                      "day": day}).pivot_table(index="day", columns="slot", values="cum")
    R = P / P.rolling(14, min_periods=5).median().shift(1)
    return R, pd.Series(R.index.dayofweek >= 5, index=R.index)


def main(a: str, b: str) -> None:
    fs = [f for f in sorted(glob.glob(K1M)) if f[-15:-8] >= str(pd.Timestamp(a) - pd.Timedelta(days=31))[:7]]
    d = pd.concat(pd.read_parquet(f, columns=["t", "c", "v"]) for f in fs)
    d.index = pd.to_datetime(d.t, unit="ms")
    R, wk = session_rvol((d.v * d.c).sort_index().resample("5min").sum())
    v = R.loc[a:b][wk.loc[a:b]].to_numpy().ravel()
    v = v[np.isfinite(v)]
    med, lo, hi = np.quantile(v, [.5, .25, .75])
    print(f"{a} ~ {b} 주말 슬롯 {v.size:,}개 · RVOL_WEEKEND_MED = {med:.3f} · RVOL_WEEKEND_LO, HI = {lo:.2f}, {hi:.2f}")


def selftest() -> None:
    idx = pd.date_range("2026-01-05", periods=28 * 288, freq="5min")            # 월요일부터 4주
    m5 = pd.Series(np.where(idx.dayofweek >= 5, 50.0, 100.0), index=idx)         # 주말 거래대금 = 평일의 절반
    R, wk = session_rvol(m5)
    late = R.iloc[14:]                                                            # 14일 기준선이 찬 뒤
    assert np.allclose(late[~wk.iloc[14:]].to_numpy(), 1.0)                      # 평일 = 평소(중앙값이 평일)
    assert np.allclose(late[wk.iloc[14:]].to_numpy(), 0.5)                       # 주말 = 0.5배로 눌린다
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        selftest()
    else:
        today = pd.Timestamp.now().normalize()
        args = sys.argv[1:] or [str((today - pd.DateOffset(months=21)).date()), str((today - pd.Timedelta(days=1)).date())]
        main(*args)
