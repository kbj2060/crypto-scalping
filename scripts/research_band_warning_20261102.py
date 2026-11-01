#!/usr/bin/env python3
"""띠 이탈 경고 모델 (2026-11-02). 15분마다 «앞으로 h 안에 가장 가까운 지지~저항 띠를 종가로 벗어나는가»를 짧은 지평으로 경고한다.
사전등록 docs/prereg/band_warning_20261102.md.   python scripts/research_band_warning_20261102.py build | eval
"""
from __future__ import annotations

import importlib.util
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view as sw

ROOT = Path(__file__).resolve().parents[1]
sp = importlib.util.spec_from_file_location("rf", ROOT / "scripts" / "research_regime_f_20261102.py")
rf = importlib.util.module_from_spec(sp)
sp.loader.exec_module(rf)
OUT = ROOT / "tmp" / "band_warning_20261102"
HS = (6, 12, 24)                 # 30분 · 1시간 · 2시간 (5분봉 수)
STEP, WARM = 3, 2016             # 15분 결정 간격 · 워밍업 7일
TRAIN_END, VAL_END = rf.TRAIN_END, rf.VAL_END
GEO = ["dR", "dS", "dmin", "width", "bpos"]
VOLF = ["rv12_288", "rv48_288", "rv288_2016", "lrv288"]
Q = (0.02, 0.05, 0.10)           # 경고로 올리는 결정 비율(예산)
NBOOT, SEED = 4000, 20261102


def edge_frame(df, i, btc, snaps):
    """15분 결정 인덱스 i 의 피쳐·띠. 레벨 = 그 시간 정시 스냅샷(정시에 알 수 있던 집합)을 현재가 기준으로 위·아래로 다시 나눠 가장 가까운 한 쌍."""
    c, h, l, qv, tb = (df[x].to_numpy() for x in ("c", "h", "l", "qv", "tbqv"))
    ts = df.index.values.astype("datetime64[s]").astype("int64")
    p = c[i]
    R1, S1 = np.full(len(i), np.nan), np.full(len(i), np.nan)
    for k, ix in enumerate(i):
        sn = snaps.get(((ts[ix] + 300) // 3600) * 3600)
        if sn is None:
            continue
        lv = np.asarray(sn[1] + sn[2], float)
        up, dn = lv[lv > p[k]], lv[lv < p[k]]
        if len(up):
            R1[k] = up.min()
        if len(dn):
            S1[k] = dn.max()
    r1 = np.diff(np.log(c), prepend=np.nan)
    sig = np.sqrt(pd.Series(r1 ** 2).rolling(48).sum().to_numpy())[i]
    X = rf.features(df, i, btc)
    X["dR"], X["dS"] = (R1 / p - 1) / sig, (1 - S1 / p) / sig
    X["dmin"], X["width"], X["bpos"] = np.minimum(X.dR, X.dS), (R1 - S1) / p / sig, (p - S1) / (R1 - S1)
    for k in (3, 6, 12):                                                      # 띠 가장자리로 다가오는 속도(σ 단위)·모멘텀
        pk = c[i - k]
        X[f"apR{k}"], X[f"apS{k}"] = (R1 / pk - 1) / sig - X.dR, (1 - S1 / pk) / sig - X.dS
        X[f"ret{k}"] = (p / pk - 1) / sig
    X["vol3"] = pd.Series(qv).rolling(3).sum().to_numpy()[i] / (pd.Series(qv).rolling(2016).mean().to_numpy()[i] * 3)
    X["tk3"] = pd.Series(2 * tb - qv).rolling(3).sum().to_numpy()[i] / pd.Series(qv).rolling(3).sum().to_numpy()[i]
    X["rng3"] = (pd.Series(h).rolling(3).max().to_numpy()[i] - pd.Series(l).rolling(3).min().to_numpy()[i]) / p / sig
    W = sw(c, 12)[i - 11]
    X["nearR12"], X["nearS12"] = (np.abs(W / R1[:, None] - 1) / sig[:, None] < 0.25).mean(1), (np.abs(W / S1[:, None] - 1) / sig[:, None] < 0.25).mean(1)
    X["lvl_age"] = ((ts[i] + 300) % 3600) / 3600                              # 레벨이 만들어진 지 얼마나 됐나(0~0.75)
    for H in HS:                                                              # 라벨: i+1 부터 H 봉 안에 종가가 띠 밖
        ok = i + H < len(c)
        Wf = np.full((len(i), H), np.nan)
        Wf[ok] = sw(c, H)[i[ok] + 1]
        X[f"y{H}"] = np.where(ok, ((Wf > R1[:, None]) | (Wf < S1[:, None])).any(1), np.nan)
    ok = i + 24 < len(c)
    Wf = np.full((len(i), 24), np.nan)
    Wf[ok] = sw(c, 24)[i[ok] + 1]
    br = (Wf > R1[:, None]) | (Wf < S1[:, None])
    X["tb"] = np.where(ok & br.any(1), br.argmax(1) + 1, np.nan)               # 첫 이탈까지 걸린 봉 수(최대 24)
    X["R1"], X["S1"] = R1, S1
    return X


def build():
    b = rf.load("BTC").c
    parts = []
    for coin in rf.COINS:
        df = rf.load(coin)
        btc = pd.DataFrame({"er48": (b - b.shift(48)).abs() / b.diff().abs().rolling(48).sum(),
                            "rv": np.log(np.sqrt((np.log(b).diff() ** 2).rolling(48).sum()) / np.sqrt((np.log(b).diff() ** 2).rolling(288).sum()))})
        i = np.arange(len(df))
        i = i[(i % STEP == STEP - 1) & (i >= WARM)]
        snaps = pickle.load(open(ROOT / "tmp" / "levels_4coin_20261102" / f"levels_{coin}.pkl", "rb"))
        X = edge_frame(df, i, btc, snaps)
        X["coin"], X["ts"] = coin, df.index[i + 1]                              # 결정 시각 T = 봉 i 마감 = 다음 봉 시가
        parts.append(X)
    d = pd.concat(parts, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    OUT.mkdir(parents=True, exist_ok=True)
    d.to_parquet(OUT / "data.parquet")
    for H in HS:
        for w, (a, b2) in rf.SPLITS.items():
            x = d[rf.window(d.ts, a, b2)]
            print(f"h{H * 5}분 {w}: n={len(x)} 이탈율={x[f'y{H}'].mean():.3f} 띠결측={x.dmin.isna().mean():.3f}")


def day_ci(day, x, rng, lo=5, hi=95):
    g = pd.DataFrame({"d": day, "x": x}).groupby("d").x.agg(["sum", "count"]).to_numpy()
    idx = rng.integers(0, len(g), (NBOOT, len(g)))
    return np.percentile(g[idx, 0].sum(1) / g[idx, 1].sum(1), [lo, hi])


def prec_at(score, y, q):
    thr = np.quantile(score, 1 - q)
    sel = score >= thr
    return float(y[sel].mean()), sel


def evaluate():
    from sklearn.ensemble import HistGradientBoostingClassifier as HGB
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(SEED)
    d = pd.read_parquet(OUT / "data.parquet")
    d["day"] = d.ts.dt.date
    META = {"coin", "ts", "day", "R1", "S1", "tb"} | {f"y{h}" for h in HS}
    FE = [c for c in d.columns if c not in META]
    res = {"n_features": len(FE)}
    cfgs = {"d3": dict(max_depth=3, learning_rate=0.03, max_iter=600, min_samples_leaf=200), "d5": dict(max_depth=5, learning_rate=0.05, max_iter=300, min_samples_leaf=200),
            "d4": dict(max_depth=4, learning_rate=0.03, max_iter=400, min_samples_leaf=400)}
    for H in HS:
        y = f"y{H}"
        x = d.dropna(subset=[y, "dmin"])
        pur = pd.Timedelta(minutes=5 * 24)
        tr, va, te = x[x.ts < pd.Timestamp(TRAIN_END) - pur], x[rf.window(x.ts, TRAIN_END, VAL_END)], x[x.ts >= pd.Timestamp(VAL_END)]
        trva = x[x.ts < pd.Timestamp(VAL_END) - pur]
        mk = lambda cfg: HGB(l2_regularization=1.0, early_stopping=False, random_state=0, **cfg)               # noqa: E731
        lg = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))   # noqa: E731
        sel = {}
        for nm, cfg in cfgs.items():                                                                           # 검증 AUC 로 구성 1개 선택
            m = mk(cfg).fit(tr[FE], tr[y].astype(int))
            sel[nm] = roc_auc_score(va[y].astype(int), m.predict_proba(va[FE])[:, 1])
        best = max(sel, key=sel.get)
        r = {"val_auc_by_cfg": sel, "selected": best}
        for w, (fit, ev) in {"검증": (tr, va), "시험": (trva, te)}.items():
            yy = ev[y].astype(int).to_numpy()
            P = {"M": mk(cfgs[best]).fit(fit[FE], fit[y].astype(int)).predict_proba(ev[FE])[:, 1],
                 "B1_거리": lg().fit(fit[GEO], fit[y].astype(int)).predict_proba(ev[GEO])[:, 1],
                 "B3_거리+변동성": lg().fit(fit[GEO + VOLF], fit[y].astype(int)).predict_proba(ev[GEO + VOLF])[:, 1],
                 "B4_최소거리": -ev.dmin.to_numpy()}
            base = yy.mean()
            out = {"n": len(ev), "base": float(base)}
            for nm, s in P.items():
                out[nm] = {"auc": float(roc_auc_score(yy, s))}
                for q in Q:
                    pr, _ = prec_at(s, yy, q)
                    out[nm][f"prec@{int(q * 100)}%"] = pr
            # 모델 − 최선 기준선(같은 비율에서) 정밀도 차의 일 블록 CI (주 판정 h=1h, q=5%)
            if H == 12:
                for q in Q:
                    pm, sm = prec_at(P["M"], yy, q)
                    bl = max(("B1_거리", "B3_거리+변동성", "B4_최소거리"), key=lambda n: prec_at(P[n], yy, q)[0])
                    _, sb = prec_at(P[bl], yy, q)
                    # 같은 크기 비교를 위해 각각 상위 q 의 정밀도를 일 단위로 모은다
                    dd = pd.DataFrame({"day": ev.day.to_numpy(), "m": sm * yy, "ms": sm.astype(float), "b": sb * yy, "bs": sb.astype(float)})
                    g = dd.groupby("day").sum().to_numpy()
                    idx = rng.integers(0, len(g), (NBOOT, len(g)))
                    diff = g[idx, 0].sum(1) / np.maximum(g[idx, 1].sum(1), 1) - g[idx, 2].sum(1) / np.maximum(g[idx, 3].sum(1), 1)
                    out[f"diff_vs_{bl}@{int(q * 100)}%"] = dict(mean=float(pm - prec_at(P[bl], yy, q)[0]), ci90=np.percentile(diff, [5, 95]).tolist())
                # 이탈 사건 포착: 경고 상위 5% 에서 «이탈 15분 전 이상 + 60분 안» 에 울린 비율, 하루 오경보
                s = P["M"]
                thr = np.quantile(s, 0.95)
                alarm = s >= thr
                tb_abs = ev.ts.to_numpy() + (ev.tb.to_numpy() * 5) * np.timedelta64(1, "m")
                evt = pd.DataFrame({"coin": ev.coin.to_numpy(), "tb": tb_abs, "y": yy, "alarm": alarm, "lead": ev.tb.to_numpy() * 5}).query("y==1 and lead<=60")
                ev_ok = evt.groupby(["coin", "tb"]).apply(lambda z: bool(((z.lead >= 15) & z.alarm).any()))
                out["event_recall_lead15_60"] = float(ev_ok.mean())
                out["events"] = int(len(ev_ok))
                out["false_alarms_per_coin_day"] = float(((alarm) & (yy == 0)).sum() / (ev.ts.dt.normalize().nunique() * 4))
                out["alarms_per_coin_day"] = float(alarm.sum() / (ev.ts.dt.normalize().nunique() * 4))
                # 가장자리 근처(최소거리 < 1σ)에서의 정밀도
                near = (ev.dmin.to_numpy() < 1.0)
                out["near_edge_n"] = int(near.sum())
                out["near_edge_base"] = float(yy[near].mean())
                out["near_edge_M_auc"] = float(roc_auc_score(yy[near], s[near]))
                out["near_edge_B4_auc"] = float(roc_auc_score(yy[near], P["B4_최소거리"][near]))
            r[w] = out
        res[f"h{H * 5}분"] = r
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / "eval.json", "w"), indent=1, default=float, ensure_ascii=False)
    for k, v in res.items():
        if isinstance(v, dict):
            print("\n==", k, "선택", v["selected"], {a: round(b, 3) for a, b in v["val_auc_by_cfg"].items()})
            for w in ("검증", "시험"):
                o = v[w]
                print(f"  [{w}] n={o['n']:,} 이탈율 {o['base']:.3f}")
                for nm in ("M", "B1_거리", "B3_거리+변동성", "B4_최소거리"):
                    print("     ", nm.ljust(12), {a: round(b, 3) for a, b in o[nm].items()})
                for kk in o:
                    if kk.startswith(("diff_", "event_", "events", "false_", "alarms_", "near_")):
                        print("      ", kk, o[kk])


if __name__ == "__main__":
    {"build": build, "eval": evaluate}[sys.argv[1]]()
