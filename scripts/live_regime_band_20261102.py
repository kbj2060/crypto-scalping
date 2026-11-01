#!/usr/bin/env python3
"""횡보/추세(S/R 띠 4h 이탈) 분별 서빙 (2026-11-02). 학습과 같은 함수(frame_for)로 피쳐를 만든다 -- 학습·서빙 같은 식.
입력: 코인별 5분봉 패널 {코인: DataFrame(index=시가 시각 UTC naive, 열 o h l c qv tbqv n)} -- 마지막 봉이 정시 마감 봉(시가 분 = 55)이고 길이 ≥ 2500.
출력: p_trend(앞으로 4시간 안에 종가가 가장 가까운 지지~저항 띠를 이탈할 확률), regime(p≥.5 추세 / 아니면 횡보), confidence=|p−.5|×2.
정확도 참고(시험 2026-06~10, 확신도별): 0.8↑ 95% · 0.6~0.8 86% · 0.4~0.6 75% · 0.2~0.4 67% · 0~0.2 57%. 연구 문서 docs/experiments/regime_acc_20261102.md
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "tmp" / "regime_acc_20261102" / "regime_band_h48_model.joblib"
LOOKBACK_H = 24                                    # 대시보드 청산맵과 같은 창


def _mod(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


rf = _mod("rf", "research_regime_f_20261102.py")
acc = _mod("acc", "research_regime_acc_20261102.py")
LM = _mod("lm", "live_liquidation_map_20260824.py")


def from_klines(k: pd.DataFrame) -> pd.DataFrame:
    """바이낸스 klines 형식(열 open_time|timestamp, open, high, low, close, quote_volume, taker_buy_quote, trades) → 패널."""
    t = pd.to_datetime(k["timestamp"] if "timestamp" in k else k["open_time"])
    return pd.DataFrame({"o": k.open.to_numpy(float), "h": k.high.to_numpy(float), "l": k.low.to_numpy(float), "c": k.close.to_numpy(float),
                         "qv": k.quote_volume.to_numpy(float), "tbqv": k.taker_buy_quote.to_numpy(float), "n": k.trades.to_numpy(float)},
                        index=pd.DatetimeIndex(t).tz_localize(None))


def level_at(df: pd.DataFrame, t: int):
    """경계 t(초)에 알 수 있던 청산맵 레벨 -- 학습 스냅샷(research_xrp_lowvol_sweep_fade_v2.snapshots)과 같은 계산."""
    d = pd.DataFrame({"timestamp": df.index.tz_localize("UTC"), "open": df.o.values, "high": df.h.values, "low": df.l.values, "close": df.c.values,
                      "volume": (df.qv / df.c).values, "qv": df.qv.values})
    g = df.assign(v=df.qv / df.c).resample("1h")
    h1 = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "n": g.c.count()})
    h1 = h1[h1.n == 12].drop(columns="n").reset_index().rename(columns={"open_time": "timestamp", "index": "timestamp"})
    h1["timestamp"] = pd.to_datetime(h1["timestamp"]).dt.tz_localize("UTC")
    h, m = LM._bar_arrays(h1), LM._bar_arrays(d)
    k = int(np.searchsorted(m[0], t))
    win = LM._window_at(h, m, t, LOOKBACK_H)
    r = LM.compute_spliced_levels(win, float(d.close.iloc[k - 1]), d.iloc[max(0, k - 2200):k])
    return float(d.close.iloc[k - 1]), [x["price"] for x in r["support_levels"]], [x["price"] for x in r["resistance_levels"]]


def predict_latest(panels: dict, coin: str, bundle=None) -> dict:
    df = panels[coin]
    assert df.index[-1].minute == 55 and len(df) >= 2500, "마지막 봉은 정시 마감 봉(시가 55분)이고 길이 ≥ 2500 이어야 한다"
    bundle = bundle or joblib.load(MODEL)
    b = panels["BTC"].reindex(df.index)
    bc = b.c
    btc = pd.DataFrame({"er48": (bc - bc.shift(48)).abs() / bc.diff().abs().rolling(48).sum(),
                        "rv": np.log(np.sqrt((np.log(bc).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(bc).diff() ** 2).rolling(288).sum()))})
    i = np.array([len(df) - 1])
    t = int(df.index[-1].timestamp()) + 300
    X, R1, S1, sig, p = acc.frame_for(df, i, btc, {t: level_at(df, t)})
    X = X.replace([np.inf, -np.inf], np.nan)
    pr = float(bundle["model"].predict_proba(X[bundle["features"]])[:, 1][0])
    return dict(coin=coin, ts=df.index[-1] + pd.Timedelta(minutes=5), p_trend=pr, regime="추세" if pr >= 0.5 else "횡보", confidence=abs(pr - 0.5) * 2,
                resistance=float(R1[0]), support=float(S1[0]), features=X[bundle["features"]].iloc[0].to_dict())
