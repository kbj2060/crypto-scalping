"""풋프린트 머리 «지평 사다리»의 방향 신호 둘 (2026-10-15 연구 → 사용자 시안 C).

  cb  : 코인베이스 프리미엄 7일 변화 → 향후 7일 (scripts/research_short_fresh_20261015.py)
        z = (ln(코인베이스 BTC-USD / 바이낸스 BTCUSDT) 7일 평균의 7일 변화) ÷ 직전 365일 표준편차
  z91 : 달러·유동성 점수 → 향후 4~13주 (scripts/research_trend_mid_long_20261015.py)
        Z = z(−Δ91 ln DTWEXBGS) + z(Δ91 ln(WALCL − RRP − TGA)), z = ÷ 직전 730일 표준편차
  |값| ≥ 1 만 방향, 사이는 «의견 없음». 둘 다 전진 판정 전 참고용(docs/prereg/*_20261015.json).

  vol4h: 4시간 변동성 예측(ETH, 화면 표시 전용 -- 주문 크기에는 안 쓴다) = scripts/live_eth_sizing_vol_model_20260912
        GBM 22피쳐 8시드. 평소 = 최근 30일 예측 중앙값 · 확대 확률 = sigmoid(−1.4950 + 5.1671·ln(예측/직전 4시간)).

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


VOL_P = (-1.4950, 5.1671)   # 확대 확률 상수 기본값(아티팩트 expand_logit 이 없을 때) -- 10-15 긴 데이터 모델의 2025-09~2026-10 표본외 예측으로 적합(십분위 보정 오차 ≤3.5pp) · 재학습하면 다시 적합


def vol4h_from_klines(rows: list, art: dict, now_ms: float) -> dict:
    """바이낸스 5분 klines(완결봉만 남김, 시간순) → 4시간 변동성 예측. art = load_model() 결과."""
    from scripts import live_eth_sizing_vol_model_20260912 as svm
    rows = [r for r in rows if int(r[6]) < now_ms]
    if len(rows) < svm.WARMUP + 288:
        return {"ok": False, "reason": f"봉 부족({len(rows)})"}
    ts = pd.to_datetime([int(r[0]) for r in rows], unit="ms")
    col = lambda i: np.array([float(r[i]) for r in rows])
    c = col(4)
    X = svm.build_features(pd.Series(ts), c, col(7), col(8), col(2), col(3)).iloc[svm.WARMUP:]
    X = X[np.isfinite(X.to_numpy(float)).all(1)]
    p = svm.predict_vol(art["models"], X.tail(30 * 288))
    a0, a1 = art.get("expand_logit") or VOL_P   # 매달 재학습 때 표본외로 다시 맞춘 상수(아티팩트에 있으면 그것)
    pred, ref, rv48 = float(p[-1]), float(np.median(p)), float(np.exp(X["rv48"].iloc[-1]))
    s4 = pred * np.sqrt(svm.HOLD)
    return {"ok": True, "asof_ms": int(rows[-1][6]) + 1, "px": float(c[-1]), "sig4h_pct": round(100 * s4, 3),
            "sig4h_usd": round(float(c[-1]) * s4, 2 if c[-1] >= 10 else 4),   # XRP $1.3 -- 2자리면 $0.01 "lvl": round(pred / ref, 3), "x_now": round(pred / rv48, 3),
            "p_expand": round(float(1 / (1 + np.exp(-(a0 + a1 * np.log(pred / rv48))))), 3),
            "ref_days": round(len(p) / 288, 1), "model_train_end": art.get("train_end")}


def open_rev_from_klines(daily: list, m5: list, now_ms: float) -> dict:
    """2026-10-16 ⑤ 종합의 «09시 개장 되돌림»(ETH) -- 전진 기록기 decide() 그대로(같은 정의 = 화면 = 기록).
    daily/m5 = 바이낸스 klines 행(형성 봉 포함 가능 -- 닫힌 봉만 쓴다). 오늘 00:30 UTC 전이면 «대기»(open_pos 0, 오늘 진입·청산 시각만)."""
    from scripts.live_eth_direction_forward_recorder_20261010 import decide
    day0 = int(now_ms) // 86_400_000 * 86_400_000
    entry, exit_ = day0 + 1_800_000, day0 + 16_200_000                        # 00:30 · 04:30 UTC
    if now_ms < entry:
        return {"ok": True, "pending": True, "date": time.strftime("%Y-%m-%d", time.gmtime(day0 / 1000)), "open_pos": 0,
                "entry_ms": entry, "exit_due_ms": exit_}
    d_closed = [k for k in daily if int(k[6]) < now_ms]                           # 어제까지(오늘 일봉은 형성 중)
    m_closed = [k for k in m5 if int(k[6]) < now_ms]
    d = decide(d_closed, m_closed, now_ms, "ETHUSDT", 0)
    return {"ok": True, "pending": False, **{k: d[k] for k in ("date", "f30_bp", "sigma30_bp", "open_pos", "entry_px", "exit_due_ms")},
            "entry_ms": entry}


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
    # vol4h: 형성 중 봉은 버리고, 상수 예측이면 평소 대비 1배
    class _M:
        def predict(self, A):
            return np.full(len(A), np.log(0.002))
    t0 = 1_700_000_000_000
    rows = [[t0 + i * 300_000, 0, 101 + (i % 7), 99 - (i % 5), 100 + np.sin(i / 9), 0, t0 + i * 300_000 + 299_999, 1e6 + i, 500 + i % 11]
            for i in range(9000)]
    v = vol4h_from_klines(rows, {"models": [_M()]}, now_ms=rows[-1][6] - 1)   # 마지막 봉은 아직 형성 중
    assert v["ok"] and v["lvl"] == 1.0 and v["asof_ms"] == rows[-2][6] + 1, v
    assert abs(v["sig4h_pct"] - 100 * 0.002 * np.sqrt(48)) < 1e-3, v
    # open_rev: 00:30 전 = 대기 · 뒤 = 기록기 decide 와 같은 값(첫 30분 +40bp ≫ 0.5σ → 숏)
    day0 = 1_792_108_800_000                                                    # 2026-10-16 00:00 UTC
    dly = [[day0 - (92 - i) * 86_400_000, 0, 0, 0, 2000 + i, 0, day0 - (91 - i) * 86_400_000 - 1] for i in range(92)]
    m5 = [[day0 - (300 - i) * 300_000, 0, 0, 0, 2400 * (1 + 1e-4 * np.sin(i)), 0, day0 - (299 - i) * 300_000 - 1] for i in range(300)]
    m5 += [[day0 + j * 300_000, 0, 0, 0, m5[-1][4] * (1.004 if j == 5 else 1.002), 0, day0 + (j + 1) * 300_000 - 1] for j in range(6)]
    assert open_rev_from_klines(dly, m5, day0 + 1_700_000)["pending"]
    o = open_rev_from_klines(dly, m5, day0 + 1_860_000)
    assert o["open_pos"] == -1 and o["entry_ms"] == day0 + 1_800_000 and o["exit_due_ms"] == day0 + 16_200_000 and o["f30_bp"] > 30, o
    print("dir_signals selftest ok")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))   # scripts.* 임포트
    if "--selftest" in sys.argv:
        selftest()
    else:
        print(json.dumps(dir_signals_payload(), ensure_ascii=False)[:800])
