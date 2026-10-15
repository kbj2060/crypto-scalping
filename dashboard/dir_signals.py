"""풋프린트 머리 «지평 사다리»의 방향 신호 둘 (2026-10-15 연구 → 사용자 시안 C).

  cb  : 코인베이스 프리미엄 7일 변화 → 향후 7일 (scripts/research_short_fresh_20261015.py)
        z = (ln(코인베이스 BTC-USD / 바이낸스 BTCUSDT) 7일 평균의 7일 변화) ÷ 직전 365일 표준편차
  z91 : 달러·유동성 점수 → 향후 4~13주 (scripts/research_trend_mid_long_20261015.py)
        Z = z(−Δ91 ln DTWEXBGS) + z(Δ91 ln(WALCL − RRP − TGA)), z = ÷ 직전 730일 표준편차
  |값| ≥ 1 만 방향, 사이는 «의견 없음». 둘 다 전진 판정 전 참고용(docs/prereg/*_20261015.json).

UTC 일봉 완결분만 쓴다. FRED 는 발표 지연만큼 밀어 그날 알 수 있던 값만 쓴다(연구와 같은 지연).

    python dashboard/dir_signals.py --selftest
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request

import numpy as np
import pandas as pd

K = 1.0
FRED_LAG = {"DTWEXBGS": 10, "WALCL": 2, "RRPONTSYD": 1, "WTREGEN": 2}   # 연구 스크립트 FRED_LAG 와 같다


def _get(url: str, timeout: float = 30.0):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def asof(s: pd.Series, idx: pd.DatetimeIndex, lag: int) -> pd.Series:
    """날짜 d 값을 d+lag 부터 쓰도록 밀고 idx 에 앞채움."""
    s = s.dropna().copy()
    s.index = s.index + pd.Timedelta(days=lag)
    return s.reindex(s.index.union(idx)).ffill().reindex(idx)


def liq_score(dxy, walcl, rrp, tga, W: int = 91, sd_win: int = 730) -> pd.Series:
    nl = walcl / 1e3 - rrp.fillna(0) - tga / 1e3
    a = -np.log(dxy / dxy.shift(W))
    b = np.log(nl / nl.shift(W))
    return a / a.rolling(sd_win, min_periods=365).std() + b / b.rolling(sd_win, min_periods=365).std()


def cb_premium_z(cb_close: pd.Series, bn_close: pd.Series) -> pd.Series:
    p = np.log(cb_close / bn_close).dropna()
    p7 = p.rolling(7, min_periods=5).mean()
    d7 = p7 - p7.shift(7)
    return d7 / d7.rolling(365, min_periods=180).std()


def _fred(sid: str, key: str) -> pd.Series:
    q = urllib.parse.urlencode({"series_id": sid, "api_key": key, "file_type": "json", "observation_start": "2015-01-01"})
    obs = _get(f"https://api.stlouisfed.org/fred/series/observations?{q}")["observations"]
    return pd.Series({pd.Timestamp(o["date"]): float(o["value"]) for o in obs if o["value"] not in (".", "")}, dtype=float)


def _coinbase_btc(days: int, now: float) -> pd.Series:
    today = int(now // 86400 * 86400)          # 오늘 UTC 0시 = 형성 중 봉 시작 → 그 전까지만
    rows, end = [], today
    while days > 0:
        n = min(days, 300)
        start = end - n * 86400
        q = (f"granularity=86400&start={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(start))}"
             f"&end={time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(end - 1))}")
        rows += _get(f"https://api.exchange.coinbase.com/products/BTC-USD/candles?{q}")
        end, days = start, days - n
    return pd.Series({pd.Timestamp(r[0], unit="s"): float(r[4]) for r in rows if r[0] < today}).sort_index()


def _binance_btc(now: float) -> pd.Series:
    raw = _get("https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000")
    return pd.Series({pd.Timestamp(int(r[0]), unit="ms"): float(r[4]) for r in raw if int(r[6]) < now * 1000}).sort_index()


def _sig(z: pd.Series) -> dict:
    z = z.dropna()
    if z.empty:
        return {"ok": False, "reason": "no_data"}
    st = lambda v: 1 if v >= K else -1 if v <= -K else 0
    s, run = st(z.iloc[-1]), 0
    for v in reversed(z.values):
        if st(v) != s:
            break
        run += 1
    return {"ok": True, "date": str(z.index[-1].date()), "v": round(float(z.iloc[-1]), 3), "state": s, "run": run,
            "hist": [[str(d.date()), round(float(v), 3)] for d, v in z.tail(21).items()]}


def dir_signals_payload(fred_key: str | None = None, now: float | None = None) -> dict:
    now = time.time() if now is None else now
    out: dict = {"ok": True, "k": K, "generated_at": int(now)}
    try:
        out["cb"] = _sig(cb_premium_z(_coinbase_btc(420, now), _binance_btc(now)))
    except Exception as e:  # 한쪽이 실패해도 다른 쪽은 보인다
        out["cb"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:160]}
    key = fred_key or os.getenv("FRED_API_KEY")
    try:
        if not key:
            raise RuntimeError("FRED_API_KEY 없음")
        idx = pd.date_range("2016-01-01", pd.Timestamp(now, unit="s").normalize() - pd.Timedelta(days=1))
        ins = [asof(_fred(sid, key), idx, FRED_LAG[sid]) for sid in ("DTWEXBGS", "WALCL", "RRPONTSYD", "WTREGEN")]
        out["z91"] = _sig(liq_score(*ins))
    except Exception as e:   # 오류 문구에 URL(키 포함)이 섞일 수 있다
        out["z91"] = {"ok": False, "reason": f"{type(e).__name__}: {e}"[:160].replace(key or "\0", "***")}
    return out


def selftest() -> None:
    idx = pd.date_range("2018-01-01", periods=1500)
    rng = np.random.default_rng(0)
    walk = lambda: pd.Series(np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))) * 100, idx)
    ins = [walk(), walk() * 7e3, walk() * 5, walk() * 9e2]
    z = liq_score(*ins)
    for t in idx[[900, 1200, -1]]:   # 인과성: t 까지만 남겨도 같은 값
        assert abs(liq_score(*[s.loc[:t] for s in ins]).loc[t] - z.loc[t]) < 1e-9
    cb, bn = walk(), walk()
    c = cb_premium_z(cb, bn)
    assert abs(cb_premium_z(cb.loc[:idx[1000]], bn.loc[:idx[1000]]).loc[idx[1000]] - c.loc[idx[1000]]) < 1e-9
    s = _sig(pd.Series([0.2, -1.2, -1.5, -1.1], pd.date_range("2026-10-01", periods=4)))
    assert (s["state"], s["run"], s["v"]) == (-1, 3, -1.1), s
    assert _sig(pd.Series([1.3, 0.4], pd.date_range("2026-10-01", periods=2)))["state"] == 0
    a = asof(pd.Series([1.0, 2.0], pd.to_datetime(["2026-10-01", "2026-10-08"])), pd.date_range("2026-10-01", "2026-10-20"), 10)
    assert np.isnan(a.loc["2026-10-10"]) and a.loc["2026-10-11"] == 1.0 and a.loc["2026-10-18"] == 2.0
    print("dir_signals selftest ok")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(json.dumps(dir_signals_payload(), ensure_ascii=False)[:800])
