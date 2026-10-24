#!/usr/bin/env python3
"""4h 실현 변동성 예측 경주 — 배포 HGB vs HAR 계열 vs ML (2026-10-25). 사전 등록 docs/prereg/vol4h_forecast_horserace_20261025.md.

    python3 scripts/research_vol4h_forecast_horserace_20261025.py --selfcheck ETH   # S1 인과성 · S2 라벨 재구현 · S3 트립와이어
    python3 scripts/research_vol4h_forecast_horserace_20261025.py ETH [--round 1]   # 학습·선택(2025)·시험(2026) → tmp/vol4h_forecast_20261025/

타깃 y_i = ln Σ_{k=i+1}^{i+48} r_k² (봉 i = 정시 결정, 피쳐는 봉 i 까지). 데이터는 research_system_trend_ruleb_20261025.bars5 + 거래량.
"""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_system_trend_ruleb_20261025 as SYS          # noqa: E402
import live_eth_sizing_vol_model_20260912 as V              # noqa: E402
from research_eth_dir4h_reverify_20261019 import block_ci   # noqa: E402

OUT = ROOT / "tmp" / "vol4h_forecast_20261025"
H = 48
WS = (12, 48, 288, 2016, 8640)
TR_END, VA_END = pd.Timestamp("2024-12-31 23:55"), pd.Timestamp("2025-12-31 23:55")
SEEDS = V.SEEDS[:3]


def volume(sym, idx):
    """panel volume·trades(~09-14) + 공개 REST klines(09-15~10-23, 캐시). quote_volume ≈ volume×close(패널에 qv 없음)."""
    p = pd.read_parquet(SYS.D / f"binance_vision/panel/{sym}USDT.parquet", columns=["timestamp", "volume", "trades"]).set_index("timestamp")
    f = OUT / f"rest_vol_{sym}.parquet"
    if not f.exists():
        t, t1, rows = int(p.index[-1].value // 10**6) + 300000, int(pd.Timestamp(SYS.END).value // 10**6) + 86400000, []
        while t < t1:
            part = SYS.rest("/fapi/v1/klines", symbol=f"{sym}USDT", interval="5m", startTime=t, limit=1500)
            if not part:
                break
            rows += part; t = part[-1][0] + 300000; time.sleep(0.3)
        r = pd.DataFrame([(x[0], float(x[5]), float(x[8])) for x in rows if x[0] < t1], columns=["t", "volume", "trades"]).drop_duplicates("t")
        OUT.mkdir(parents=True, exist_ok=True); r.to_parquet(f)
    r = pd.read_parquet(f); r.index = pd.to_datetime(r.t, unit="ms")
    v = pd.concat([p, r[["volume", "trades"]]]); v = v[~v.index.duplicated(keep="first")]
    return v.reindex(idx).fillna(0.0)


def load(sym):
    x, miss = SYS.bars5(sym)
    v = volume(sym, x.index)
    x["v"], x["n"] = v.volume.values, v.trades.values
    x["filled"] = ((x.h == x.l) & (x.h == x.c) & (x.c == x.c.shift()) & (x.n == 0)).astype(float)
    return x, miss


def features(x: pd.DataFrame) -> pd.DataFrame:
    """봉 i 까지만 보는 피쳐 = 배포 22열 + HAR 원 피쳐. 롤링 합·창은 전부 뒤쪽 창."""
    c, h, l = x.c.values, x.h.values, x.l.values
    ts = pd.Series(x.index)
    dep = V.build_features(ts, c, x.v.values * c, x.n.values, h, l)
    r = pd.Series(np.diff(np.log(c), prepend=np.log(c[0])))
    r2, ar = r ** 2, r.abs()
    f = {}
    for w in WS:
        f[f"lrv{w}"] = np.log(np.maximum(r2.rolling(w).sum().values, 1e-10))
    for w in (48, 288):
        rv = np.maximum(r2.rolling(w).sum().values, 1e-12)
        f[f"rsn{w}"] = (r2 * (r < 0)).rolling(w).sum().values / rv
        bv = np.pi / 2 * (ar * ar.shift(1)).rolling(w).sum().values
        f[f"j{w}"] = np.maximum(rv - bv, 0) / rv
        f[f"q{w}"] = np.sqrt(w / 3 * (r ** 4).rolling(w).sum().values) / rv
        f[f"lrvq{w}"] = f[f"lrv{w}"] * f[f"q{w}"]
    t5 = x.index + pd.Timedelta("5min")             # 봉 i 마감 시각(= 결정 시각)
    f["hod"] = t5.hour.values.astype(float)
    f["wkend"] = (t5.dayofweek.values >= 5).astype(float)
    return pd.concat([dep.reset_index(drop=True), pd.DataFrame(f)], axis=1).set_index(x.index)


def target(x):
    """y_i = ln Σ_{k=i+1}^{i+48} r_k² · 라벨 창 채움 봉 수."""
    r2 = pd.Series(np.diff(np.log(x.c.values), prepend=np.log(x.c.values[0])) ** 2)
    y = np.log(np.maximum(r2.rolling(H).sum().shift(-H).values, 1e-10))
    nf = pd.Series(x.filled.values).rolling(H).sum().shift(-H).values
    return y, nf


HARX = ["lrv12", "lrv48", "lrv288", "lrv2016", "lrv8640", "rsn48", "rsn288", "j48", "j288", "q48", "q288", "lrvq48", "lrvq288"]
SETS = {"M0": ["lrv48", "lrv288", "lrv2016"], "M2": HARX, "M3": V.FEATURES, "M4": V.FEATURES + HARX + ["hod", "wkend"]}
MKT = ["mkt_lrv12", "mkt_lrv48", "mkt_lrv288", "mkt_qvz"]            # 3라운드: 4코인 같은 봉 i 평균(교차 자산)
SETS["M3X"], SETS["M4X"] = SETS["M3"] + MKT, SETS["M4"] + MKT


def design(F, m):
    if m == "M0":
        X = F[SETS["M0"]].values
    elif m in ("M1", "M2"):
        base = F[SETS["M0"] if m == "M1" else SETS["M2"]].values
        X = np.column_stack([base, np.eye(24)[F.hod.values.astype(int)][:, 1:], F.wkend.values])
    return np.column_stack([np.ones(len(X)), X])


def fit_predict(m, Ftr, ytr, Fte):
    """모델 m 을 (Ftr, ytr) 로 학습해 Fte 예측 ŷ(로그 RV)."""
    if m in ("M0", "M1", "M2"):
        b = np.linalg.lstsq(design(Ftr, m), ytr, rcond=None)[0]
        return design(Fte, m) @ b
    if m in ("M3", "M4", "M3X"):
        from sklearn.ensemble import HistGradientBoostingRegressor
        A, B = (np.nan_to_num(F[SETS[m]].to_numpy(float)) for F in (Ftr, Fte))
        return np.mean([HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=int(s))
                        .fit(A, ytr).predict(B) for s in SEEDS], axis=0)
    if m == "M5":
        import lightgbm as lgb
        A, B = Ftr[SETS["M4"]].to_numpy(float), Fte[SETS["M4"]].to_numpy(float)
        return np.mean([lgb.LGBMRegressor(num_leaves=63, learning_rate=0.03, n_estimators=800, min_child_samples=200, subsample=0.8,
                                          subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=int(s), verbose=-1, n_jobs=8)
                        .fit(A, ytr).predict(B) for s in SEEDS], axis=0)
    if m in ("M6", "M6X"):
        return mlp(Ftr, ytr, Fte, SETS["M4X" if m == "M6X" else "M4"])
    raise KeyError(m)


def mlp(Ftr, ytr, Fte, cols, seeds=SEEDS[:2]):
    import torch
    torch.set_num_threads(8)
    keep = np.arange(0, len(Ftr), 3)                     # 15분 간격
    A = np.nan_to_num(Ftr[cols].to_numpy(float)[keep]); B = np.nan_to_num(Fte[cols].to_numpy(float))
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A, B, y = (A - mu) / sd, (B - mu) / sd, ytr[keep]
    ym, ys = y.mean(), y.std()
    out = []
    for s in seeds:
        torch.manual_seed(int(s)); rng = np.random.default_rng(int(s))
        net = torch.nn.Sequential(torch.nn.Linear(A.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64), torch.nn.ReLU(), torch.nn.Linear(64, 1))
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        At, yt = torch.tensor(A, dtype=torch.float32), torch.tensor((y - ym) / ys, dtype=torch.float32)
        for _ in range(15):
            for bi in np.array_split(rng.permutation(len(A)), max(1, len(A) // 1024)):
                opt.zero_grad(); loss = ((net(At[bi]).squeeze(1) - yt[bi]) ** 2).mean(); loss.backward(); opt.step()
        with torch.no_grad():
            out.append(net(torch.tensor(B, dtype=torch.float32)).squeeze(1).numpy() * ys + ym)
    return np.mean(out, axis=0)


def qlike(y, yh):
    e = y - yh
    return np.exp(e) - e - 1


def day_ci(ts, d):
    s = pd.Series(d, index=pd.DatetimeIndex(ts)).groupby(pd.DatetimeIndex(ts).normalize()).mean().values
    lo, hi = block_ci(s, b=5, n=4000)
    return float(s.mean()), lo, hi


def mz_slope(ts, lnR, lnl, n=2000, b=5, seed=7):
    """ln R = a + b·ln lvl 기울기 · 일 블록 부트 CI."""
    day = pd.DatetimeIndex(ts).normalize(); u = np.unique(day); g = np.searchsorted(u, day)
    sl = lambda m: np.cov(lnl[m], lnR[m])[0, 1] / np.var(lnl[m], ddof=1)
    rng = np.random.default_rng(seed); k = int(np.ceil(len(u) / b)); bs = []
    for _ in range(n):
        st = rng.integers(0, len(u) - b + 1, k); days = (st[:, None] + np.arange(b)).ravel()[:len(u)]
        cnt = np.bincount(days, minlength=len(u)); w = cnt[g]
        x_, y_ = np.repeat(lnl, w), np.repeat(lnR, w)
        bs.append(np.cov(x_, y_)[0, 1] / np.var(x_, ddof=1))
    return float(sl(np.ones(len(lnl), bool))), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


def frame(sym):
    x, miss = load(sym)
    F = features(x); y, nf = target(x)
    ok = np.isfinite(y) & (nf < 5) & np.isfinite(F[SETS["M4"]].to_numpy(float)).all(1)
    return x, F, y, ok, miss


def selfcheck(sym="ETH"):
    x, miss = load(sym)
    F = features(x); y, _ = target(x)
    rng = np.random.default_rng(0)
    dec = np.flatnonzero((x.index.minute == 55) & (x.index >= "2022-03-01"))
    dec = dec[dec < len(x) - H - 1]
    for i in rng.choice(dec, 30, replace=False):          # S1
        Fi = features(x.iloc[:i + 1]).iloc[-1].to_numpy(float); Fa = F.iloc[i].to_numpy(float)
        assert np.allclose(Fi, Fa, rtol=1e-9, atol=1e-12, equal_nan=True), (x.index[i], np.nanmax(np.abs(Fi - Fa)))
    lc = np.log(x.c.values)
    for i in rng.choice(dec, 300, replace=False):         # S2
        s = 0.0
        for k in range(i + 1, i + H + 1):
            s += (lc[k] - lc[k - 1]) ** 2
        assert abs(np.log(max(s, 1e-10)) - y[i]) < 1e-12, (i, s, y[i])
    i = int(dec[len(dec) // 2]); x2 = x.copy()            # S3
    x2.iloc[i + 1:, x2.columns.get_indexer(["c", "h", "l"])] *= np.exp(rng.normal(0, 0.01, (len(x) - i - 1, 1)))
    F2 = features(x2.iloc[:i + 12]); y2, _ = target(x2.iloc[:i + 60])
    assert np.allclose(F2.iloc[i].to_numpy(float), F.iloc[i].to_numpy(float), equal_nan=True)
    assert abs(y2[i] - y[i]) > 1e-3
    print(f"{sym} 자체 점검 통과 S1(30)·S2(300)·S3 · 빈 봉 {miss} · 채움 플래그 {int(x.filled.sum())}", flush=True)


def run(sym, models, rnd):
    x, F, y, ok, miss = frame(sym)
    ts = x.index; lab_end = ts + pd.Timedelta(minutes=5 * H)
    tr = ok & (lab_end <= TR_END) & (ts >= "2022-01-01")
    trf = ok & (lab_end <= VA_END)
    dec = ok & (ts.minute == 55)
    va = dec & (ts >= "2024-12-31 23:55") & (lab_end <= VA_END)
    te = dec & (ts >= "2025-12-31 23:55")
    extra = None
    if sym == "XRP" and "M3" in models:                    # 배포 XRP 구조 = ETH+XRP 합쳐 학습
        xe, Fe, ye, oke, _ = frame("ETH"); le = xe.index + pd.Timedelta(minutes=5 * H)
        extra = (Fe, ye, oke & (le <= TR_END), oke & (le <= VA_END))
    pv, pt = {}, {}
    for m in models:
        t0 = time.time()
        pv[m] = fit_predict(m, F[tr], y[tr], F[va])
        pt[m] = fit_predict(m, F[trf], y[trf], F[te])
        print(f"{sym} {m} {time.time() - t0:.0f}s", flush=True)
        if m == "M3" and extra is not None:
            Fe, ye, a, b = extra
            pv["M3j"] = fit_predict("M3", pd.concat([F[tr], Fe[a]]), np.r_[y[tr], ye[a]], F[va])
            pt["M3j"] = fit_predict("M3", pd.concat([F[trf], Fe[b]]), np.r_[y[trf], ye[b]], F[te])
    if {"M2", "M4", "M5", "M6"} <= set(pv):
        pv["M7"] = np.mean([pv[k] for k in ("M2", "M4", "M5", "M6")], 0); pt["M7"] = np.mean([pt[k] for k in ("M2", "M4", "M5", "M6")], 0)
    return evaluate(sym, rnd, pv, pt, y[va], y[te], ts[va], ts[te], miss)


def evaluate(sym, rnd, pv, pt, yv, yt, tsv, tst, miss=None, bases=("M3", "M0", "M3j"), exclude=("M0", "M3", "M3j")):
    cst = {m: float(np.log(np.mean(np.exp(yv - pv[m])))) for m in pv}
    rep = {"sym": sym, "round": rnd, "n_val": len(yv), "n_test": len(yt), "missing_bars": miss,
           "test_range": [str(tst[0] + pd.Timedelta('5min')), str(tst[-1] + pd.Timedelta('5min'))], "models": {}}
    Lt = {m: qlike(yt, pt[m] + cst[m]) for m in pt}
    for m in pv:
        r = {"c": cst[m], "val_qlike": float(qlike(yv, pv[m] + cst[m]).mean()), "val_mse": float(((yv - pv[m]) ** 2).mean()),
             "test_qlike": float(Lt[m].mean()), "test_mse": float(((yt - pt[m]) ** 2).mean()),
             "test_corr": float(np.corrcoef(yt, pt[m])[0, 1])}
        for base in bases:
            if base in pt and m != base:
                r[f"dq_vs_{base}"] = day_ci(tst, Lt[m] - Lt[base])
                r[f"dmse_vs_{base}"] = day_ci(tst, (yt - pt[m]) ** 2 - (yt - pt[base]) ** 2)
        # lvl 보정: σ̂ = exp((ŷ+c)/2) · 기준 = 직전 720 정시 σ̂ 중앙(2025 검증 예측 + 2026 시험 예측 이어서)
        sig = pd.Series(np.exp((np.r_[pv[m], pt[m]] + cst[m]) / 2), index=np.r_[tsv, tst])
        ref = sig.rolling(720, min_periods=600).median().values[len(tsv):]
        lvl, R = sig.values[len(tsv):] / ref, np.exp(yt / 2) / ref
        r["lvl_mz"] = mz_slope(tst, np.log(R), np.log(lvl))
        q = pd.qcut(lvl, 10, labels=False)
        r["lvl_deciles"] = [[round(float(lvl[q == k].mean()), 3), round(float(np.exp(np.log(R[q == k]).mean())), 3)] for k in range(10)]
        rep["models"][m] = r
    cand = [m for m in rep["models"] if m not in exclude]
    if cand:
        rep["candidate"] = min(cand, key=lambda m: rep["models"][m]["val_qlike"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"report_r{rnd}_{sym}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False))
    pd.DataFrame({**{f"v_{m}": pd.Series(pv[m], index=tsv) for m in pv}, "y": pd.Series(yv, index=tsv)}).to_parquet(OUT / f"pred_val_r{rnd}_{sym}.parquet")
    pd.DataFrame({**{f"t_{m}": pd.Series(pt[m], index=tst) for m in pt}, "y": pd.Series(yt, index=tst)}).to_parquet(OUT / f"pred_test_r{rnd}_{sym}.parquet")
    print(f"\n{sym} r{rnd} 검증 {rep['n_val']} · 시험 {rep['n_test']} · 후보 {rep.get('candidate')}")
    print(f"{'모델':5}{'검증QL':>8}{'시험QL':>8}{'시험MSE':>8}{'corr':>6}  {'ΔQL vs M3 [CI]':>26}  {'ΔQL vs M0 [CI]':>26}  lvl기울기")
    for m, r in rep["models"].items():
        f = lambda k: "%+.4f [%+.4f,%+.4f]" % tuple(r[k]) if k in r else ""
        print(f"{m:5}{r['val_qlike']:8.4f}{r['test_qlike']:8.4f}{r['test_mse']:8.4f}{r['test_corr']:6.3f}  {f('dq_vs_M3'):>26}  {f('dq_vs_M0'):>26}  "
              "%.2f [%.2f,%.2f]" % tuple(r["lvl_mz"]) + (f"  vs M3j {f('dq_vs_M3j')}" if "dq_vs_M3j" in r else ""))
    return rep


COINS = ("ETH", "XRP", "BTC", "SOL")


def add_mkt(fr):
    """봉 i 의 4코인 평균 lrv12·48·288·qv_z (같은 5분 색인 -- bars5 가 2022-01-01~END 로 맞춘다)."""
    idx = fr[COINS[0]][0].index
    assert all((fr[s][0].index == idx).all() for s in COINS)
    for k, src in (("mkt_lrv12", "lrv12"), ("mkt_lrv48", "lrv48"), ("mkt_lrv288", "lrv288"), ("mkt_qvz", "qv_z")):
        v = np.mean([fr[s][0][src].to_numpy(float) for s in COINS], axis=0)
        for s in COINS:
            fr[s][0][k] = v


def pooled_round(rnd, specs, combos, reuse, exclude):
    """4코인 합쳐 학습 specs [(이름, fit_predict 모델)] · 이전 라운드 예측 재사용 reuse {라운드: [모델]} · combos {이름: [모델]} = ŷ 동일 평균."""
    fr, masks, oks = {}, {}, {}
    for s in COINS:
        x, F, y, ok, _ = frame(s); fr[s] = (F, y); oks[s] = (x.index, ok)
    if rnd >= 3:
        add_mkt(fr)
    for s in COINS:
        (ts, ok), F = oks[s], fr[s][0]
        ok = ok & np.isfinite(F[MKT].to_numpy(float)).all(1) if rnd >= 3 else ok
        le = ts + pd.Timedelta(minutes=5 * H); dec = ok & (ts.minute == 55)
        masks[s] = (ok & (le <= TR_END), ok & (le <= VA_END), dec & (ts >= "2024-12-31 23:55") & (le <= VA_END), dec & (ts >= "2025-12-31 23:55"))
    cat = lambda k: (pd.concat([fr[s][0][masks[s][k]] for s in COINS]), np.concatenate([fr[s][1][masks[s][k]] for s in COINS]))
    split = lambda p, k: dict(zip(COINS, np.split(p, np.cumsum([masks[s][k].sum() for s in COINS])[:-1])))
    pooled = {}
    for m, base in specs:
        t0 = time.time()
        (A, ya), (B, yb), (Va, _), (Te, _) = cat(0), cat(1), cat(2), cat(3)
        pooled[m] = (split(fit_predict(base, A, ya, Va), 2), split(fit_predict(base, B, yb, Te), 3))
        print(f"pooled {m} {time.time() - t0:.0f}s", flush=True)
    cand, reps = None, {}
    for s in COINS:
        F, y = fr[s]; va, te = masks[s][2], masks[s][3]
        pv, pt = {}, {}
        for r0, ms in reuse.items():
            rv, rt = pd.read_parquet(OUT / f"pred_val_r{r0}_{s}.parquet"), pd.read_parquet(OUT / f"pred_test_r{r0}_{s}.parquet")
            assert (rv.index == F.index[va]).all() and (rt.index == F.index[te]).all() and np.allclose(rt.y, y[te])
            for m in ms:
                if f"v_{m}" in rv:
                    pv[m], pt[m] = rv[f"v_{m}"].values, rt[f"t_{m}"].values
        for m in pooled:
            pv[m], pt[m] = pooled[m][0][s], pooled[m][1][s]
        for m, parts in combos.items():
            pv[m], pt[m] = np.mean([pv[k] for k in parts], 0), np.mean([pt[k] for k in parts], 0)
        reps[s] = evaluate(s, rnd, pv, pt, y[va], y[te], F.index[va], F.index[te], exclude=exclude)
        cand = cand or reps[s]["candidate"]          # 후보 = ETH 2025 검증 최소, 다른 코인엔 같은 모델
    print(f"\n{rnd}라운드 후보(ETH 검증) = {cand}")
    for s in COINS:
        r = reps[s]["models"][cand]
        print(s, "vs M3 %+.4f [%+.4f,%+.4f]" % tuple(r["dq_vs_M3"]), " vs M0 %+.4f [%+.4f,%+.4f]" % tuple(r["dq_vs_M0"]),
              (" vs M3j %+.4f [%+.4f,%+.4f]" % tuple(r["dq_vs_M3j"])) if "dq_vs_M3j" in r else "")


# ---------- 4라운드 (사전 등록 7dac1da6): 학습 24·25 / 검증 26-01~05 / 시험 26-06~10-23 · GRU 시퀀스 ----------
TR4_START, TR4_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2025-12-31 23:55")
VA4_START, VA4_END, TE4_START = pd.Timestamp("2025-12-31 23:55"), pd.Timestamp("2026-05-31 23:55"), pd.Timestamp("2026-05-31 23:55")
L_SEQ = 168
STAT = ["lrv48", "lrv288", "lrv2016", "mkt_lrv48", "hs", "hc", "wkend"]


def seq_inputs(F, ts, idx):
    """GRU 입력: 결정 봉 위치 idx 마다 직전 168 정시(자기 포함)의 [lnRV1h · 시장 lnRV1h · 시각 sin·cos] 수열 + 정적 7."""
    hod = (ts + pd.Timedelta("5min")).hour.values
    F = F.assign(hs=np.sin(2 * np.pi * hod / 24), hc=np.cos(2 * np.pi * hod / 24))
    dec = np.flatnonzero(ts.minute == 55); pos = np.searchsorted(dec, idx); assert (dec[pos] == idx).all() and (pos >= L_SEQ - 1).all()
    ch = F[["lrv12", "mkt_lrv12", "hs", "hc"]].to_numpy(np.float32)[dec]
    S = np.stack([ch[p - L_SEQ + 1:p + 1] for p in pos]); st = F[STAT].to_numpy(np.float32)[idx]
    return S, st


def gru_fit(S, st, y, evals, loss="mse", seeds=SEEDS[:2]):
    import torch
    torch.set_num_threads(8)
    cm, cs = S.reshape(-1, S.shape[2]).mean(0), S.reshape(-1, S.shape[2]).std(0) + 1e-9; sm_, ss_ = st.mean(0), st.std(0) + 1e-9
    ym, ys = y.mean(), y.std(); nz = lambda a, b: (torch.tensor((a - cm) / cs), torch.tensor((b - sm_) / ss_))
    Xs, Xt = nz(S, st); yt = torch.tensor(y, dtype=torch.float32); outs = [[] for _ in evals]
    for sd in seeds:
        torch.manual_seed(int(sd)); rng = np.random.default_rng(int(sd))
        gru = torch.nn.GRU(S.shape[2], 32, batch_first=True); head = torch.nn.Sequential(torch.nn.Linear(32 + st.shape[1], 32), torch.nn.ReLU(), torch.nn.Linear(32, 1))
        f = lambda a, b: head(torch.cat([gru(a)[1][-1], b], 1)).squeeze(1) * ys + ym
        opt = torch.optim.Adam(list(gru.parameters()) + list(head.parameters()), lr=1e-3)
        for _ in range(15):
            for bi in np.array_split(rng.permutation(len(y)), max(1, len(y) // 512)):
                e = yt[bi] - f(Xs[bi], Xt[bi])
                l = ((e / ys) ** 2).mean() if loss == "mse" else (torch.exp(e) - e - 1).mean()
                opt.zero_grad(); l.backward(); opt.step()
        with torch.no_grad():
            for k, (Se, ste) in enumerate(evals):
                a, b = nz(Se, ste); outs[k].append(np.concatenate([f(a[j:j + 4096], b[j:j + 4096]).numpy() for j in range(0, len(a), 4096)]))
    return [np.mean(o, 0) for o in outs]


def round4(check=False, b=None, tag=4, losses=(("G1", "mse"), ("G2", "qlike"))):
    tr0, tr1, va0, va1, te0, te1 = b or (TR4_START, TR4_END, VA4_START, VA4_END, TE4_START, None)
    fr, oks = {}, {}
    for s in COINS:
        x, F, y, ok, miss = frame(s); fr[s] = (F, y); oks[s] = (x.index, ok, miss)
    add_mkt(fr); M = {}
    for s in COINS:
        (ts, ok, _), F = oks[s], fr[s][0]
        ok = ok & np.isfinite(F[MKT].to_numpy(float)).all(1); le = ts + pd.Timedelta(minutes=5 * H); dec = ok & (ts.minute == 55)
        M[s] = dict(tr=ok & (ts >= tr0) & (le <= tr1), va=dec & (ts >= va0) & (le <= va1), te=dec & (ts >= te0) & ((le <= te1) if te1 is not None else True))
        M[s]["trd"] = M[s]["tr"] & dec
    if check:   # S4: 봉 i 뒤 행을 망가뜨려도 수열·정적 입력이 같다
        F, ts = fr["ETH"][0], oks["ETH"][0]; rng = np.random.default_rng(4)
        for i in rng.choice(np.flatnonzero(M["ETH"]["te"]), 10, replace=False):
            a = seq_inputs(F, ts, np.array([i])); G = F.copy(); G.iloc[i + 1:] = np.nan; b = seq_inputs(G, ts, np.array([i]))
            assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.isfinite(a[0]).all()
        print("S4 GRU 수열 인과성 10 표본 통과"); return
    pv = {s: {} for s in COINS}; pt = {s: {} for s in COINS}
    for s in COINS:   # 코인별 M0·M3·M6 — 한 번 학습해 검증·시험 함께 예측
        F, y = fr[s]; nv = M[s]["va"].sum(); E = pd.concat([F[M[s]["va"]], F[M[s]["te"]]])
        for m in ("M0", "M3", "M6"):
            t0 = time.time(); p = fit_predict(m, F[M[s]["tr"]], y[M[s]["tr"]], E); pv[s][m], pt[s][m] = p[:nv], p[nv:]; print(f"{s} {m} {time.time() - t0:.0f}s", flush=True)
    cat = lambda k: (pd.concat([fr[s][0][M[s][k]] for s in COINS]), np.concatenate([fr[s][1][M[s][k]] for s in COINS]))
    A, ya = cat("tr"); E = pd.concat([pd.concat([fr[s][0][M[s]["va"]], fr[s][0][M[s]["te"]]]) for s in COINS])
    cuts = np.cumsum([M[s]["va"].sum() + M[s]["te"].sum() for s in COINS])[:-1]
    for m, base in (("PX3", "M3X"), ("PX6", "M6X")):
        t0 = time.time(); p = np.split(fit_predict(base, A, ya, E), cuts); print(f"pooled {m} {time.time() - t0:.0f}s", flush=True)
        for s, q in zip(COINS, p):
            nv = M[s]["va"].sum(); pv[s][m], pt[s][m] = q[:nv], q[nv:]
    St, stt = [], []; yt_ = []
    for s in COINS:
        F, y = fr[s]; ts = oks[s][0]; a, b = seq_inputs(F, ts, np.flatnonzero(M[s]["trd"])); St.append(a); stt.append(b); yt_.append(y[M[s]["trd"]])
    St, stt, yt_ = np.concatenate(St), np.concatenate(stt), np.concatenate(yt_)
    ev = []
    for s in COINS:
        F = fr[s][0]; ts = oks[s][0]; ev += [seq_inputs(F, ts, np.flatnonzero(M[s]["va"])), seq_inputs(F, ts, np.flatnonzero(M[s]["te"]))]
    for m, loss in losses:
        t0 = time.time(); out = gru_fit(St, stt, yt_, ev, loss); print(f"pooled {m}({loss}) 학습 행 {len(yt_)} {time.time() - t0:.0f}s", flush=True)
        for k, s in enumerate(COINS):
            pv[s][m], pt[s][m] = out[2 * k], out[2 * k + 1]
    reps = {}
    for s in COINS:
        for m, parts in (("QX4", ["M3", "M6", "PX3", "PX6"]), ("QG", ["M3", "M6", "PX3", "PX6", "G1"])):
            pv[s][m], pt[s][m] = np.mean([pv[s][k] for k in parts], 0), np.mean([pt[s][k] for k in parts], 0)
        F, y = fr[s]; ts = oks[s][0]
        reps[s] = evaluate(s, tag, pv[s], pt[s], y[M[s]["va"]], y[M[s]["te"]], ts[M[s]["va"]], ts[M[s]["te"]], oks[s][2], exclude=("M0", "M3"))
    cand = reps["ETH"]["candidate"]; print(f"\n4라운드 후보(ETH 검증) = {cand}")
    for s in COINS:
        r = reps[s]["models"][cand]
        print(s, "vs M3 %+.4f [%+.4f,%+.4f]" % tuple(r["dq_vs_M3"]), " vs M0 %+.4f [%+.4f,%+.4f]" % tuple(r["dq_vs_M0"]))


# ---------- 5라운드 (815dffa0): QG 레시피 굴림 원점 재현 ----------
ROLL = {"5W1": (pd.Timestamp("2022-01-08"), pd.Timestamp("2023-12-31 23:55"), pd.Timestamp("2023-12-31 23:55"), pd.Timestamp("2024-05-31 23:55"),
                pd.Timestamp("2024-05-31 23:55"), pd.Timestamp("2024-10-23 23:55")),
        "5W2": (pd.Timestamp("2023-01-01"), pd.Timestamp("2024-12-31 23:55"), pd.Timestamp("2024-12-31 23:55"), pd.Timestamp("2025-05-31 23:55"),
                pd.Timestamp("2025-05-31 23:55"), pd.Timestamp("2025-10-23 23:55"))}


def roll_pool():
    """3창(5W1·5W2·4) 시험 손실 차를 일 평균으로 이어 붙여 판정."""
    out = {}
    for s in COINS:
        cell = {}
        for base in ("M3", "M0"):
            days, per = [], {}
            for tg in ("5W1", "5W2", "4"):
                rep = json.loads((OUT / f"report_r{tg}_{s}.json").read_text()); P = pd.read_parquet(OUT / f"pred_test_r{tg}_{s}.parquet")
                L = lambda m: qlike(P.y.values, P[f"t_{m}"].values + rep["models"][m]["c"])
                d = pd.Series(L("QG") - L(base), index=P.index).groupby(P.index.normalize()).mean(); days.append(d); per[tg] = float(d.mean())
            ser = pd.concat(days).values; lo, hi = block_ci(ser, b=5, n=4000)
            cell[f"vs_{base}"] = {"mean": float(ser.mean()), "ci": [float(lo), float(hi)], "per_window": per, "days": len(ser)}
        out[s] = cell
        print(s, "QG−M3 %+.4f [%+.4f,%+.4f]" % (cell["vs_M3"]["mean"], *cell["vs_M3"]["ci"]), "창", {k: round(v, 4) for k, v in cell["vs_M3"]["per_window"].items()},
              " QG−M0 %+.4f [%+.4f,%+.4f]" % (cell["vs_M0"]["mean"], *cell["vs_M0"]["ci"]))
    e = out["ETH"]
    out["verdict"] = {"ETH_win": e["vs_M3"]["ci"][1] < 0 and e["vs_M0"]["ci"][1] < 0 and sum(v < 0 for v in e["vs_M3"]["per_window"].values()) >= 2,
                      "replication": sum(out[s]["vs_M3"]["ci"][1] < 0 for s in ("XRP", "BTC", "SOL"))}
    (OUT / "report_r5_pooled.json").write_text(json.dumps(out, indent=1, ensure_ascii=False)); print(json.dumps(out["verdict"], ensure_ascii=False))


# ---------- 6라운드 (0311d90e): 결합 방식 — 저장된 3창 예측만 ----------
MEMB = ["M3", "M6", "PX3", "PX6", "G1"]


def _c(y, yh):
    return float(np.log(np.mean(np.exp(y - yh))))


def _fit_w(P, y):
    from scipy.optimize import minimize
    k = P.shape[1]; f = lambda w: qlike(y, P @ w + _c(y, P @ w)).mean()   # noqa: E731
    r = minimize(f, np.full(k, 1 / k), method="SLSQP", bounds=[(0, 1)] * k, constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}])
    return r.x


def round6():
    tags = ("5W1", "5W2", "4"); out = {}
    for s in COINS:
        val, tst, cm3, days = {k: 0.0 for k in ("QX4", "QW", "QM")}, {}, [], {}
        diffs = {k: [] for k in ("QX4", "QW", "QM")}; dvq = {k: [] for k in ("QW", "QM")}; wts = {}
        for tg in tags:
            V, T = pd.read_parquet(OUT / f"pred_val_r{tg}_{s}.parquet"), pd.read_parquet(OUT / f"pred_test_r{tg}_{s}.parquet")
            Pv, Pt = V[[f"v_{m}" for m in MEMB]].to_numpy(), T[[f"t_{m}" for m in MEMB]].to_numpy(); yv, yt = V.y.values, T.y.values
            h = len(yv) // 2; w_ab, w_ba = _fit_w(Pv[:h], yv[:h]), _fit_w(Pv[h:], yv[h:])
            cf = lambda yh_fit, y_fit: _c(y_fit, yh_fit)   # noqa: E731
            sc_qw = (qlike(yv[h:], Pv[h:] @ w_ab + cf(Pv[:h] @ w_ab, yv[:h])).sum() + qlike(yv[:h], Pv[:h] @ w_ba + cf(Pv[h:] @ w_ba, yv[h:])).sum()) / len(yv)
            w = _fit_w(Pv, yv); wts[tg] = np.round(w, 3).tolist()
            comb = {"QX4": (Pv[:, :4].mean(1), Pt[:, :4].mean(1)), "QW": (Pv @ w, Pt @ w), "QM": (np.median(Pv, 1), np.median(Pt, 1))}
            for k, (pv_, pt_) in comb.items():
                c = _c(yv, pv_); val[k] += (sc_qw if k == "QW" else qlike(yv, pv_ + c).mean()) / len(tags)
                Lk = qlike(yt, pt_ + c); L3 = qlike(yt, T.t_M3.values + _c(yv, V.v_M3.values))
                diffs[k].append(pd.Series(Lk - L3, index=T.index).groupby(T.index.normalize()).mean())
                if k != "QX4":
                    Lx = qlike(yt, Pt[:, :4].mean(1) + _c(yv, Pv[:, :4].mean(1)))
                    dvq[k].append(pd.Series(Lk - Lx, index=T.index).groupby(T.index.normalize()).mean())
        S = min(val, key=val.get); cell = {"val_score": val, "S": S, "weights_QW": wts}
        for k in diffs:
            ser = pd.concat(diffs[k]).values; lo, hi = block_ci(ser, b=5, n=4000); cell[f"{k}_vs_M3"] = [float(ser.mean()), float(lo), float(hi)]
        for k in dvq:
            ser = pd.concat(dvq[k]).values; lo, hi = block_ci(ser, b=5, n=4000); cell[f"{k}_vs_QX4"] = [float(ser.mean()), float(lo), float(hi)]
        out[s] = cell
        print(s, "S", S, {k: round(v, 4) for k, v in val.items()}, "| vs M3", {k: np.round(cell[f'{k}_vs_M3'], 4).tolist() for k in diffs},
              "| vs QX4", {k: np.round(cell[f'{k}_vs_QX4'], 4).tolist() for k in dvq}, "| QW 가중", wts, flush=True)
    (OUT / "report_r6.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selfcheck" in a:
        selfcheck(a[a.index("--selfcheck") + 1]); sys.exit(0)
    if "--round2" in a:
        pooled_round(2, [("P3", "M3"), ("P6", "M6")], {"C36": ["M3", "M6"], "PC": ["P3", "P6"]}, {1: ["M0", "M3", "M6", "M3j"]},
                     ("M0", "M3", "M3j", "M6")); sys.exit(0)
    if "--round6" in a:
        round6(); sys.exit(0)
    if "--roll" in a:
        for tg, bb in ROLL.items():
            round4(b=bb, tag=tg, losses=(("G1", "mse"),))
        roll_pool(); sys.exit(0)
    if "--round4" in a:
        round4(check="--check" in a); sys.exit(0)
    if "--round3" in a:
        pooled_round(3, [("PX3", "M3X"), ("PX6", "M6X")], {"Q4": ["M3", "M6", "P3", "P6"], "PXC": ["PX3", "PX6"], "QX4": ["M3", "M6", "PX3", "PX6"]},
                     {1: ["M0", "M3", "M6", "M3j"], 2: ["P3", "P6", "PC"]}, ("M0", "M3", "M3j", "M6", "P3", "P6", "PC")); sys.exit(0)
    rnd = int(a[a.index("--round") + 1]) if "--round" in a else 1
    models = a[a.index("--models") + 1].split(",") if "--models" in a else ["M0", "M1", "M2", "M3", "M4", "M5", "M6"]
    run(a[0], models, rnd)
