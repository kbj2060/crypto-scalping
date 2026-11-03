#!/usr/bin/env python3
"""대시보드 추가 후보 3종 검정 (사전등록: docs/prereg/dashboard_candidates_jump_xsec_hit_20261103.md).

T1 점프(급락) 위험 증분 AUC · T2 횡단면 상대 강도 IC · T3 방향 예측 적중 초과.
사용: python3 scripts/research_dashboard_candidates_20261103.py [--selftest] [T1 T2 T3]
데이터는 개발 PC data/ (메인 체크아웃). 서버 REST 호출 없음.
"""
import json, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", ROOT.parents[2] / "data"))  # 워크트리 → 메인 체크아웃
OUT = ROOT / "tmp" / "dashboard_candidates_20261103"
TR = ("2024-01-01", "2025-12-31"); VA = ("2026-01-01", "2026-05-31"); TE = ("2026-06-01", "2026-08-22 23:55")
B = 2000
RNG = np.random.default_rng(20261103)
COINS = ["eth", "btc", "sol", "xrp", "ada", "avax", "bnb", "doge", "link"]


def load(coin):
    """5분봉 → 봉 시작시각(UTC) 인덱스. close_time 으로 시각을 만든다(KST 시프트 파일 회피)."""
    fs = [f for f in DATA.glob(f"{coin}_5m_*.csv") if "2022_2023" not in f.name and "2021" not in f.name]
    d = pd.concat([pd.read_csv(f) for f in fs])
    d["t"] = pd.to_datetime(((d.close_time.astype("int64") + 1) // 300000 - 1) * 300000, unit="ms")
    d = d.drop_duplicates("t").set_index("t").sort_index()
    return d[["open", "high", "low", "close", "volume", "taker_buy_base"]].astype(float).reindex(
        pd.date_range(d.index[0], d.index[-1], freq="5min"))


def fwd_min(low, h):
    """min(low[t+1..t+h])"""
    return low.iloc[::-1].rolling(h, min_periods=h).min().iloc[::-1].shift(-1)


def day_boot(day, stat, n=B):
    """UTC 일 블록 부트스트랩. stat(일 인덱스 배열)→스칼라. 일 단위 집계는 stat 안에서."""
    days = np.unique(day); out = np.empty(n)
    for i in range(n):
        out[i] = stat(RNG.choice(days, len(days)))
    return out


def holm(ps, alpha=0.10):
    order = np.argsort(ps); ok = [False] * len(ps)
    for rank, i in enumerate(order):
        if ps[i] <= alpha / (len(ps) - rank): ok[i] = True
        else: break
    return ok


# ---------------------------------------------------------------- T3
def t3():
    seed = pd.read_parquet(DATA / "live" / "eth_dirshort_oos_seed.parquet").set_index("ts")
    k = load("eth"); res = {}
    start_eval = seed.index[0] + pd.Timedelta(days=14)
    for name, col, h in [("P6_15m", "p15m", 3), ("P7_1h", "p1h", 12)]:
        ret = (k.close.shift(-h) / k.close - 1).reindex(seed.index)
        p = seed[col]; day = p.index.normalize()
        thr_hi = pd.Series(np.nan, index=p.index); thr_lo = thr_hi.copy()
        for d in np.unique(day[day >= start_eval.normalize()]):  # 인과: 그 날 0시 이전 예측만
            past = p[p.index < d].values
            m = day == d
            thr_hi[m] = np.quantile(past, .8); thr_lo[m] = np.quantile(past, .2)
        ok = ret.notna() & thr_hi.notna() & (p.index <= pd.Timestamp(TE[1]))
        df = pd.DataFrame({"day": day, "up": (ret > 0).astype(float), "dn": (ret < 0).astype(float),
                           "top": p >= thr_hi, "bot": p <= thr_lo})[ok]
        g = df.groupby("day").apply(lambda x: pd.Series({
            "n": len(x), "up": x.up.sum(), "dn": x.dn.sum(), "nt": x.top.sum(), "ht": x.up[x.top].sum(),
            "nb": x.bot.sum(), "hb": x.dn[x.bot].sum()}))
        a = g.values  # 열: n up dn nt ht nb hb

        def parts(idx):
            s = a[idx].sum(0)
            return s[4] / s[3] - s[1] / s[0], s[6] / s[5] - s[2] / s[0], s[3], s[5]
        def pooled(idx):
            u, d_, _, _ = parts(idx); return (u + d_) / 2
        full = np.arange(len(g))
        bs = np.array([pooled(RNG.choice(full, len(full))) for _ in range(B)])
        u, d_, nt, nb = parts(full)
        s = a.sum(0)
        res[name] = {"days": len(g), "rows": int(s[0]), "top_hit": f"{int(s[4])}/{int(nt)}", "top_base_up": s[1] / s[0],
                     "bot_hit": f"{int(s[6])}/{int(nb)}", "bot_base_dn": s[2] / s[0],
                     "excess_top_vs_long": u, "excess_bot_vs_short": d_, "pooled": pooled(full),
                     "lo90": float(np.quantile(bs, .10)), "ci90": [float(np.quantile(bs, .05)), float(np.quantile(bs, .95))],
                     "pass": bool(np.quantile(bs, .10) > 0)}
        mo = df.assign(m=df.index.strftime("%Y-%m")).groupby("m").apply(lambda x: {
            "top": f"{int(x.up[x.top].sum())}/{int(x.top.sum())}", "bot": f"{int(x.dn[x.bot].sum())}/{int(x.bot.sum())}"})
        res[name]["monthly"] = mo.to_dict()
    return res


# ---------------------------------------------------------------- T2
def t2():
    px = pd.DataFrame({c: load(c).close for c in COINS})
    px = px[px.index >= "2023-12-20"]
    t = px.index[(px.index.hour % 4 == 3) & (px.index.minute == 55)]  # 4h 경계 직전 마감봉
    lab = (px.shift(-48) / px - 1).loc[t]
    lab = lab.sub(lab.mean(1), axis=0)
    res = {}; ics = {}
    for lb in [48, 288, 864, 2016]:
        f = (px / px.shift(lb) - 1).loc[t]; f = f.sub(f.mean(1), axis=0)
        m = f.notna().all(1) & lab.notna().all(1) & (f.index >= "2024-01-01")
        fr, lr = f[m].rank(axis=1), lab[m].rank(axis=1)
        ic = (fr.sub(fr.mean(1), axis=0) * lr.sub(lr.mean(1), axis=0)).sum(1) / np.sqrt(
            (fr.sub(fr.mean(1), axis=0) ** 2).sum(1) * (lr.sub(lr.mean(1), axis=0) ** 2).sum(1))
        ics[lb] = (ic, f[m], lab[m])
    names = {48: "P2_4h", 288: "P3_24h", 864: "P4_72h", 2016: "P5_168h"}
    pvals = []
    for lb, (ic, f, l) in ics.items():
        per = {k: float(ic[(ic.index >= a) & (ic.index <= b)].mean()) for k, (a, b) in
               {"train": TR, "val": VA, "test": TE}.items()}
        te = ic[(ic.index >= TE[0]) & (ic.index <= TE[1])]; day = te.index.normalize().values
        dsum = pd.DataFrame({"d": day, "v": te.values}).groupby("d").v.agg(["sum", "count"]).values
        bs = np.array([ (lambda i: dsum[i, 0].sum() / dsum[i, 1].sum())(RNG.choice(len(dsum), len(dsum))) for _ in range(B)])
        p = float((bs >= 0).mean()); pvals.append(p)
        # 보조: 반전 롱숏(하위 3 롱·상위 3 숏) 비용 전 bp/4h, 회전
        mt = (f.index >= TE[0]) & (f.index <= TE[1])
        fr = f[mt].rank(axis=1); L = (fr <= 3).astype(float); S = (fr >= len(COINS) - 2).astype(float)
        w = (L - S) / 3
        spread = (w * l[mt]).sum(1).mean() * 1e4
        turn = (w.diff().abs().sum(1) / 2).mean()
        res[names[lb]] = {"mean_ic": per, "test_n": int(len(te)), "test_days": int(len(dsum)), "upper90": float(np.quantile(bs, .90)),
                          "p_ic_ge0": p, "reversal_longshort_bp_per_4h": float(spread), "turnover_per_4h": float(turn)}
    holm_ok = holm(pvals)
    for (lb, n), ok in zip(names.items(), holm_ok):
        r = res[n]; r["pass"] = bool(ok and all(v < 0 for v in r["mean_ic"].values()))
    return res


# ---------------------------------------------------------------- T1
def feats():
    k = load("eth")
    m = pd.read_csv(DATA / "TOTAL_ETHUSDT_metrics_2024_2026.csv", parse_dates=["create_time"])
    m["t"] = m.create_time - pd.Timedelta(minutes=5)  # 버킷 종료 라벨 → 봉 시작시각
    m = m.drop_duplicates("t").set_index("t")[["sum_open_interest", "sum_toptrader_long_short_ratio"]]
    k = k.join(m)
    c, h, l, v = k.close, k.high, k.low, k.volume
    z = lambda s: (s - s.rolling(2016).mean()) / s.rolling(2016).std()
    rv = np.log(np.sqrt((np.log(c).diff() ** 2).rolling(288).sum()))
    hr = k.index.hour + k.index.minute / 60
    B_ = pd.DataFrame({
        "rng12": np.log((h.rolling(12).max() - l.rolling(12).min()) / c), "rng48": np.log((h.rolling(48).max() - l.rolling(48).min()) / c),
        "rv288": rv, "hs": np.sin(2 * np.pi * hr / 24), "hc": np.cos(2 * np.pi * hr / 24)})
    tb = k.taker_buy_base.rolling(12).sum() / v.rolling(12).sum()
    X = pd.DataFrame({
        "oi12": np.log(k.sum_open_interest / k.sum_open_interest.shift(12)), "oi48": np.log(k.sum_open_interest / k.sum_open_interest.shift(48)),
        "tls": z(np.log(k.sum_toptrader_long_short_ratio)), "tb12": z(tb), "r288": c / c.shift(288) - 1,
        "vr": v.rolling(12).sum() / (v.rolling(288).sum() / 24)})
    y_dd = fwd_min(l, 12) / c - 1
    return B_, X, y_dd


def t1():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, log_loss
    Bf, Xf, dd = feats()
    ok = np.isfinite(Bf).all(axis=1) & np.isfinite(Xf).all(axis=1) & dd.notna()
    ix = Bf.index
    tr = ok & (ix >= TR[0]) & (ix <= TR[1] + " 23:55"); va = ok & (ix >= VA[0]) & (ix <= VA[1] + " 23:55"); te = ok & (ix >= TE[0]) & (ix <= TE[1])
    thr = float(np.quantile(dd[tr], .02)); y = (dd <= thr).astype(int)
    trs = tr & (np.arange(len(ix)) % 3 == 0)  # 학습은 3봉마다(라벨 12봉 겹침, 속도) — 모든 모델 동일
    out = {"drop_threshold": thr, "base_rate": {k: float(y[m].mean()) for k, m in {"train": tr, "val": va, "test": te}.items()}}

    def fit_eval(cols, Xd, pick=True, mask_tr=trs):
        A = Xd[cols]; mu, sd = A[mask_tr].mean(), A[mask_tr].std()
        Z = (A - mu) / sd
        best = None
        for C in ([0.01, 0.1, 1] if pick else [0.1]):
            mdl = LogisticRegression(C=C, max_iter=500).fit(Z[mask_tr], y[mask_tr])
            ll = log_loss(y[va], mdl.predict_proba(Z[va])[:, 1])
            if best is None or ll < best[0]: best = (ll, mdl, C)
        return best[1], Z, best[2]
    allf = pd.concat([Bf, Xf], axis=1)
    mb, Zb, Cb = fit_eval(list(Bf.columns), allf); mx, Zx, Cx = fit_eval(list(allf.columns), allf)
    pb = mb.predict_proba(Zb[te])[:, 1]; px_ = mx.predict_proba(Zx[te])[:, 1]; yt = y[te].values
    tday = ix[te].normalize().values
    ev_days = int(pd.Series(yt, index=tday).groupby(level=0).max().sum())
    d_auc = roc_auc_score(yt, px_) - roc_auc_score(yt, pb)
    d_ll = log_loss(yt, pb) - log_loss(yt, px_)
    udays = np.unique(tday); idx_by_day = {d: np.where(tday == d)[0] for d in udays}
    bs = []
    for _ in range(B):
        ii = np.concatenate([idx_by_day[d] for d in RNG.choice(udays, len(udays))])
        if yt[ii].min() == yt[ii].max(): continue
        bs.append(roc_auc_score(yt[ii], px_[ii]) - roc_auc_score(yt[ii], pb[ii]))
    lo90 = float(np.quantile(bs, .10))
    # 위약: 학습 X를 일 단위로 섞기 200회 → 같은 시험 ΔAUC
    day_all = ix.normalize()
    dtr = np.unique(day_all[trs]); plc = []
    Xcols = list(Xf.columns)
    base = allf[Xcols].values
    tgt = {d: np.where((day_all == d) & trs.values)[0] for d in dtr}
    src = {d: np.where((day_all == d) & ok.values)[0] for d in dtr}
    for r in range(200):
        perm = dict(zip(dtr, RNG.permutation(dtr)))
        Xv = base.copy()
        for d in dtr:  # 학습 행의 X 를 다른 날의 X 로 교체(날 단위 순열, 유한 행만 원천)
            a, b = tgt[d], src[perm[d]]; n = min(len(a), len(b))
            Xv[a[:n]] = base[b[:n]]
        Xp = allf.copy(); Xp[Xcols] = Xv
        m_, Z_, _ = fit_eval(list(allf.columns), Xp, pick=False)
        plc.append(roc_auc_score(yt, m_.predict_proba(Zx[te])[:, 1]) - roc_auc_score(yt, pb))
    # 지연 위약: X 를 하루(288봉) 뒤로 민 입력
    Xl = allf.copy(); Xl[Xcols] = allf[Xcols].shift(288)
    ok2 = ok & np.isfinite(Xl).all(axis=1)
    tr2 = trs & ok2
    ml, Zl, _ = fit_eval(list(allf.columns), Xl.fillna(0), pick=False, mask_tr=tr2)
    d_lag = roc_auc_score(yt, ml.predict_proba(Zl[te])[:, 1]) - roc_auc_score(yt, pb)
    coef = dict(zip(allf.columns, map(float, mx.coef_[0])))
    out.update({"test_rows": int(te.sum()), "test_events": int(yt.sum()), "test_event_days": ev_days, "test_days": len(udays),
                "auc_base": float(roc_auc_score(yt, pb)), "auc_full": float(roc_auc_score(yt, px_)), "dAUC": float(d_auc),
                "dAUC_lo90": lo90, "dLogLoss": float(d_ll), "C_base": Cb, "C_full": Cx,
                "placebo_dAUC_p95": float(np.quantile(plc, .95)), "placebo_dAUC_mean": float(np.mean(plc)),
                "lag1d_dAUC": float(d_lag), "coef_full": coef})
    out["P1"] = {"status": "보류(검정력)" if ev_days < 15 else ("통과" if (lo90 > 0 and d_ll > 0) else "불통과")}
    return out


# ---------------------------------------------------------------- selftest
def selftest():
    low = pd.Series(np.random.default_rng(1).random(40))
    f = fwd_min(low, 5)
    for t in range(30): assert abs(f.iloc[t] - low.iloc[t + 1:t + 6].min()) < 1e-12
    assert holm([.001, .5, .02, .9]) == [True, False, True, False] or holm([.001, .5, .02, .9])[0]
    print("selftest ok")


if __name__ == "__main__":
    if "--selftest" in sys.argv: selftest(); sys.exit(0)
    sel = [a for a in sys.argv[1:] if a in ("T1", "T2", "T3")] or ["T1", "T2", "T3"]
    OUT.mkdir(parents=True, exist_ok=True)
    for name, fn in {"T1": t1, "T2": t2, "T3": t3}.items():
        if name in sel:
            r = fn(); (OUT / f"{name}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float))
            print(name, json.dumps(r, ensure_ascii=False, indent=1, default=float), flush=True)
