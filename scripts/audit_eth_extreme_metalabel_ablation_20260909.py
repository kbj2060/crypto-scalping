#!/usr/bin/env python3
"""극점 타깃 메타라벨 **기하 피쳐 절제(ablation)** — 라벨 기하를 외운 것인가 (2026-09-09).

감사 B 에서 모델 없는 규칙 "아래꼬리 깊이 상위10%"만으로 정밀도 44.4%가 나왔다. 라벨
`min(low[i+1..i+12]) >= low[i]` 의 기준선이 low[i] 라서, **꼬리가 깊을수록 라벨이 기하학적으로
쉬워진다**. 모델이 그걸 외운 건지, 그 위에 진짜 정보를 얹은 건지 가른다.

절제 대상(기하군 11개): lower/upper_wick_ratio · pos_in_range{12,48,144} ·
                       dist_lo{12,48,144}_atr · dist_hi{12,48,144}_atr
세 팔: 전체 피쳐 / 기하군 제거 / 기하군만
판정: 기하군 제거에서 정밀도·순bp 가 살아남으면 진짜, 44% 부근으로 주저앉으면 라벨 기하 학습.

⚠️API 재수집 없이 klcache 의 2년치 parquet 을 직접 읽는다(다른 세션과 CPU 를 나눠 쓰는 중).
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path: sys.path.insert(0, str(_p))
import live_evidence_signal_dashboard_20260823 as EV
import build_eth_anchor_label_dataset_20260907 as B

OUT = ROOT / "tmp/eth_signal_map_20260909"; CACHE = OUT / "klcache"
W, H_EVAL, COST = 12, 48, 10.0
SEEDS = [20260909, 771233, 305610, 517758, 961476]
VAL0, OOS0 = pd.Timestamp("2026-04-01"), pd.Timestamp("2026-06-16")
GEO = (["lower_wick_ratio", "upper_wick_ratio"]
       + [f"pos_in_range{w}" for w in (12, 48, 144)]
       + [f"dist_lo{w}_atr" for w in (12, 48, 144)]
       + [f"dist_hi{w}_atr" for w in (12, 48, 144)])
RNG = np.random.default_rng(20260909)


def newest(sym: str) -> pd.DataFrame:
    best, bn = None, 0
    for f in glob.glob(str(CACHE / f"{sym}_5m_*.parquet")):
        d = pd.read_parquet(f)
        if len(d) > bn: best, bn = d, len(d)
    return best.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    kl, btc = newest("ETHUSDT"), newest("BTCUSDT")
    print(f"캐시 사용: ETH {len(kl):,}봉 {kl.timestamp.iloc[0]} ~ {kl.timestamp.iloc[-1]}", flush=True)
    try:
        fund = EV.fetch_funding_history(limit=1000)
        fund["calc_time"] = pd.to_datetime(fund["calc_time"]).dt.tz_localize(None)
    except Exception:
        fund = None
    sig = EV.compute_signals(kl, btc_df=btc, funding_df=fund)
    ts = pd.to_datetime(sig["timestamp"])
    op = sig.open.to_numpy(float); hi_ = sig.high.to_numpy(float); lo_ = sig.low.to_numpy(float)
    cl = sig.close.to_numpy(float); atr = sig.atr_pct.to_numpy(float); n = len(sig)
    lo, hi = 900, n - H_EVAL - W - 3
    btc_cl = btc.set_index("timestamp")["close"].reindex(pd.DatetimeIndex(ts)).ffill().to_numpy(float)

    S = pd.DataFrame(index=range(n))
    for c in ("p_fast", "p_slow", "delta_z", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
              "ret3_z", "atr_pct", "dem", "kalman_dev_z"):
        S[c] = sig[c].to_numpy(float)
    c_s = pd.Series(cl)
    S["atr_pctile"] = pd.Series(atr).rolling(2016, min_periods=500).rank(pct=True).to_numpy()
    for w in (12, 48, 144):
        S[f"ret{w}"] = (c_s / c_s.shift(w) - 1).to_numpy() / np.maximum(atr, 1e-9)
        rmin = pd.Series(lo_).rolling(w).min().to_numpy(); rmax = pd.Series(hi_).rolling(w).max().to_numpy()
        S[f"pos_in_range{w}"] = (cl - rmin) / np.maximum(rmax - rmin, 1e-9)
        S[f"dist_lo{w}_atr"] = (cl - rmin) / np.maximum(cl * atr, 1e-9)
        S[f"dist_hi{w}_atr"] = (rmax - cl) / np.maximum(cl * atr, 1e-9)
    b_s = pd.Series(btc_cl)
    S["btc_ret12"] = (b_s / b_s.shift(12) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["btc_ret48"] = (b_s / b_s.shift(48) - 1).to_numpy() / np.maximum(atr, 1e-9)
    S["eth_btc_div"] = S["ret12"] - S["btc_ret12"]
    S["hour"] = ts.dt.hour.to_numpy(); S["weekday"] = ts.dt.weekday.to_numpy()
    for s in B.SIGNALS: S[f"f_{s}"] = 0.0
    FEATS = list(S.columns) + ["n_signals", "is_bottom"]

    rows = []
    for sd, long in (("bottom", True), ("top", False)):
        fires = {s: sig[f"{sd}_{s}"].fillna(False).to_numpy(bool) for s in B.SIGNALS}
        anyf = np.zeros(n, bool); cnt = np.zeros(n, int)
        for s in B.SIGNALS: anyf |= fires[s]; cnt += fires[s].astype(int)
        idx = np.flatnonzero(anyf); idx = idx[(idx >= lo) & (idx <= hi)]
        fwd = np.array([lo_[i + 1:i + 1 + W].min() if long else hi_[i + 1:i + 1 + W].max() for i in idx])
        X = S.iloc[idx].copy()
        for s in B.SIGNALS: X[f"f_{s}"] = fires[s][idx].astype(float)
        X["n_signals"] = cnt[idx].astype(float); X["is_bottom"] = 1.0 if long else 0.0
        X["_i"] = idx; X["_ts"] = ts.iloc[idx].to_numpy(); X["_long"] = long
        X["_y"] = ((fwd >= lo_[idx]) if long else (fwd <= hi_[idx])).astype(int)
        rows.append(X)
    A = pd.concat(rows, ignore_index=True).sort_values("_ts").reset_index(drop=True)
    A[FEATS] = A[FEATS].replace([np.inf, -np.inf], np.nan)
    A = A.dropna(subset=FEATS).reset_index(drop=True)
    tr = A[A._ts < VAL0]; oo = A[A._ts >= OOS0]
    days = (oo._ts.max() - oo._ts.min()).total_seconds() / 86400
    k10 = max(int(len(oo) * 0.10), 20); h = oo._ts.median()
    print(f"모집단 {len(A):,} · TRAIN {len(tr):,} · OOS {len(oo):,} ({days:.0f}일) "
          f"· 기저 {oo._y.mean()*100:.1f}%\n", flush=True)

    def netbp(d):
        i = d._i.to_numpy(); e = op[i + 1]; x = cl[i + H_EVAL]; r = (x - e) / e * 1e4
        return np.where(d._long.to_numpy(), r, -r) - COST

    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    print("=" * 104)
    print(f"{'팔':<20}{'피쳐수':>7}{'AUC':>9}{'상위10% 정밀도':>15}{'순bp':>9}{'전반':>8}{'후반':>8}{'시드범위':>18}")
    res = []
    for lab, feats in (("전체 피쳐", FEATS),
                       ("기하군 제거", [f for f in FEATS if f not in GEO]),
                       ("기하군만", GEO + ["is_bottom"])):
        P = np.zeros(len(oo)); per = []
        for sd_ in SEEDS:
            m = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_depth=6,
                                               l2_regularization=1.0, random_state=sd_,
                                               early_stopping=True, validation_fraction=0.15)
            m.fit(tr[feats], tr._y)
            p = m.predict_proba(oo[feats])[:, 1]; P += p / len(SEEDS)
            per.append(netbp(oo.assign(p=p).nlargest(k10, "p")).mean())
        q = oo.assign(p=P).nlargest(k10, "p"); g = oo.assign(p=P)
        q1 = g[g._ts < h].nlargest(max(int((g._ts < h).sum() * 0.1), 10), "p")
        q2 = g[g._ts >= h].nlargest(max(int((g._ts >= h).sum() * 0.1), 10), "p")
        per = np.array(per)
        print(f"{lab:<20}{len(feats):>7}{roc_auc_score(oo._y, P):>9.4f}{q._y.mean()*100:>14.1f}%"
              f"{netbp(q).mean():>9.2f}{netbp(q1).mean():>8.2f}{netbp(q2).mean():>8.2f}"
              f"{f'[{per.min():+.2f},{per.max():+.2f}]':>18}")
        res.append(dict(arm=lab, n_feat=len(feats), auc=round(float(roc_auc_score(oo._y, P)), 4),
                        prec=round(float(q._y.mean()), 4), net=round(float(netbp(q).mean()), 2),
                        h1=round(float(netbp(q1).mean()), 2), h2=round(float(netbp(q2).mean()), 2),
                        seed_min=round(float(per.min()), 2), seed_max=round(float(per.max()), 2)))
    wick = np.where(oo._long, oo.lower_wick_ratio, oo.upper_wick_ratio)
    qw = oo.iloc[np.argsort(-wick)[:k10]]
    print(f"{'모델없음(꼬리깊이)':<20}{1:>7}{'-':>9}{qw._y.mean()*100:>14.1f}%{netbp(qw).mean():>9.2f}")
    print(f"{'기저(전건)':<20}{'-':>7}{'-':>9}{oo._y.mean()*100:>14.1f}%{netbp(oo).mean():>9.2f}")
    pd.DataFrame(res).to_csv(OUT / "metalabel_ablation.csv", index=False)
    print(f"\n커버리지 상위10% = {k10}건 = {k10/days:.1f}건/일")
    print(json.dumps({"done": True, **{r["arm"]: r["net"] for r in res}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
