#!/usr/bin/env python3
"""횡보/추세 분별 모델의 제로샷 일반화 (2026-11-02): 학습에 쓴 적 없는 코인(ADA·AVAX·BNB·DOGE·LINK)에 동결 구성을 적용.
사전등록 docs/prereg/regime_zeroshot_20261102.md.   python scripts/research_regime_zeroshot_20261102.py snap <COIN> | build | eval
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _mod(name, rel):
    sp = importlib.util.spec_from_file_location(name, ROOT / "scripts" / rel)
    m = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(m)
    return m


S = _mod("serve", "live_regime_band_20261102.py")      # rf · acc · LM 포함
V2 = _mod("v2", "research_xrp_lowvol_sweep_fade_v2_20261102.py")
rf, acc, LM = S.rf, S.acc, S.LM
NEW = ("ADA", "AVAX", "BNB", "DOGE", "LINK")
OUT = ROOT / "tmp" / "regime_zeroshot_20261102"
CSV = "/home/kbj20/crypto-scalping/data/{}_5m_2024_2026.csv"
TEST_START = "2026-06-01"


def load_csv(coin):
    c = pd.read_csv(CSV.format(coin.lower()), usecols=["timestamp", "open", "high", "low", "close", "quote_volume", "taker_buy_quote", "trades"], parse_dates=["timestamp"])
    df = pd.DataFrame({"o": c.open.values, "h": c.high.values, "l": c.low.values, "c": c.close.values, "qv": c.quote_volume.values,
                       "tbqv": c.taker_buy_quote.values, "n": c.trades.values}, index=pd.DatetimeIndex(c.timestamp))
    df.index.name = "open_time"
    assert df.index.is_monotonic_increasing and not df.index.duplicated().any()
    assert (df.index.to_series().diff().dropna() == pd.Timedelta("5min")).all(), "5분 격자 아님"
    return df


def dh1(df):
    d = pd.DataFrame({"timestamp": df.index.tz_localize("UTC"), "open": df.o.values, "high": df.h.values, "low": df.l.values, "close": df.c.values,
                      "volume": (df.qv / df.c).values, "qv": df.qv.values})
    g = df.assign(v=df.qv / df.c).resample("1h")
    h1 = pd.DataFrame({"high": g.h.max(), "low": g.l.min(), "close": g.c.last(), "volume": g.v.sum(), "n": g.c.count()})
    h1 = h1[h1.n == 12].drop(columns="n").reset_index().rename(columns={"open_time": "timestamp", "index": "timestamp"})
    h1["timestamp"] = pd.to_datetime(h1["timestamp"]).dt.tz_localize("UTC")
    return d, h1


def snap(coin):
    OUT.mkdir(parents=True, exist_ok=True)
    V2.LEVELS = OUT / f"levels_{coin}.pkl"
    d, h1 = dh1(load_csv(coin))
    print(coin, len(V2.snapshots(d, h1)))


def build():
    btc_full = rf.load("BTC")
    parts = []
    for coin in NEW:
        df = load_csv(coin)
        b = btc_full.reindex(df.index).c
        btc = pd.DataFrame({"er48": (b - b.shift(48)).abs() / b.diff().abs().rolling(48).sum(),
                            "rv": np.log(np.sqrt((np.log(b).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(b).diff() ** 2).rolling(288).sum()))})
        i = rf.decisions(df)
        snaps = pickle.load(open(OUT / f"levels_{coin}.pkl", "rb"))
        X, R1, S1, sig, p = acc.frame_for(df, i, btc, snaps)
        c = df.c.to_numpy()
        from numpy.lib.stride_tricks import sliding_window_view as sw
        ok = i + 48 < len(c)
        W = np.full((len(i), 48), np.nan)
        W[ok] = sw(c, 48)[i[ok] + 1]
        X["yA48"] = np.where(ok, ((W > R1[:, None]) | (W < S1[:, None])).any(1), np.nan)
        X["coin"], X["ts"] = coin, df.index[i + 1]
        parts.append(X)
    d = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    d.to_parquet(OUT / "data.parquet")
    print(d.groupby("coin").size().to_dict())


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(rf.SEED)
    old = pd.read_parquet(acc.OUT / "data.parquet")
    ALL = [c for c in old.columns if c not in acc.META]
    cols = [c for c in ALL if c not in acc.HIST]
    purge = pd.Timedelta(minutes=240)
    tr = old.dropna(subset=["yA48"])
    tr = tr[tr.ts < pd.Timestamp(rf.VAL_END) - purge]                                    # 시험 시점 모델과 같은 학습 구간(4코인, ~2026-05)
    hgb = HGB(max_depth=3, learning_rate=0.03, max_iter=600, l2_regularization=1.0, min_samples_leaf=200, early_stopping=False, random_state=0).fit(tr[cols], tr.yA48.astype(int))
    geo = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000)).fit(tr[acc.GEO], tr.yA48.astype(int))
    d = pd.read_parquet(OUT / "data.parquet").dropna(subset=["yA48"])
    d["day"] = d.ts.dt.date
    out = {}
    for name, x in (("제로샷 코인+시간 (2026-06~08)", d[d.ts >= TEST_START]), ("제로샷 코인 (2024-01~2026-05, 시간은 학습과 겹침)", d[d.ts < TEST_START])):
        x = x.reset_index(drop=True)
        yy = x.yA48.to_numpy().astype(int)
        p, q = hgb.predict_proba(x[cols])[:, 1], geo.predict_proba(x[acc.GEO])[:, 1]
        okm, okg = ((p >= .5) == yy).astype(float), ((q >= .5) == yy).astype(float)
        maj = max(yy.mean(), 1 - yy.mean())
        r = dict(n=len(x), trend_rate=float(yy.mean()), majority=float(maj), acc_model=float(okm.mean()), acc_geo=float(okg.mean()), auc_model=float(roc_auc_score(yy, p)),
                 auc_geo=float(roc_auc_score(yy, q)), gain_vs_geo=float((okm - okg).mean()), gain_vs_geo_ci95=rf.day_ci(x.day.to_numpy(), okm - okg, rng).tolist(),
                 by_coin={c: dict(n=int((x.coin == c).sum()), model=float(okm[(x.coin == c).to_numpy()].mean()), geo=float(okg[(x.coin == c).to_numpy()].mean()),
                                  majority=float(max(yy[(x.coin == c).to_numpy()].mean(), 1 - yy[(x.coin == c).to_numpy()].mean()))) for c in NEW})
        conf = np.abs(p - .5) * 2
        r["by_conf"] = [dict(conf=f"{lo:.1f}-{hi:.1f}", n=int(((conf >= lo) & (conf < hi)).sum()), acc=float(okm[(conf >= lo) & (conf < hi)].mean())) for lo, hi in zip((0, .2, .4, .6, .8), (.2, .4, .6, .8, 1.01))]
        out[name] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(OUT / "eval.json", "w"), indent=1, default=float, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    cmd = sys.argv[1]
    {"snap": lambda: snap(sys.argv[2]), "build": build, "eval": evaluate}[cmd]()
