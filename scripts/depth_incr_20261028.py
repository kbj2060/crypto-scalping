#!/usr/bin/env python3
"""호가 깊이 10열 → 대시보드 예측 모델 4종 증분 시험 (2026-10-28). 사전등록 docs/prereg/depth_incr_dashboard_models_20261028.md (29c3eed1).

A = 배포 레시피를 같은 창(학습 2024~25)으로 · B = A + 깊이 10열 · 위약 B′(L) = 깊이를 L 일 과거 값으로.
서버 저장소 루트에서(서버 scripts/ 에 두지 않음 — tmp/depth_incr_20261028/ 사본):
    nice -n 19 env OMP_NUM_THREADS=2 python tmp/depth_incr_20261028/depth_incr_20261028.py selfcheck|dir4h|box|qx4|card30|report
"""
import json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path.cwd()
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT))
OUT = ROOT / "tmp/depth_incr_20261028"
LIQ = ROOT / "data/liqlab"
DCOLS = ["dimb1", "dimb2", "dimb5", "dimb1_d1h", "dimb2_d1h", "dimb5_d1h", "depth1_z", "bid_near", "ask_near", "d_age"]
LAGS = (7, 13, 19, 25, 31, 37, 43, 49, 55)
TR0, VA0, TE0 = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-01-01"), pd.Timestamp("2026-06-01")
BOX_SEEDS = (1716463990, 1783687091, 508351009, 434771877, 1978411480)
CARD_SEEDS = (1805214274, 1798703776, 2066915355, 1773210922, 1874492600)


# ── 깊이 10열 (방향 세션 diag_rl_dir_liqfeat_20261028.feats 의 book 블록과 같은 식) ──────────────
def depth_from(E):
    b = {k: E[f"d-{k}"] for k in (1, 2, 5)}; a = {k: E[f"d{k}"] for k in (1, 2, 5)}
    imb = {k: (b[k] - a[k]) / (b[k] + a[k]) for k in (1, 2, 5)}; ld1 = np.log(b[1] + a[1])
    return pd.DataFrame(dict(**{f"dimb{k}": imb[k] for k in (1, 2, 5)}, **{f"dimb{k}_d1h": imb[k] - imb[k].shift(12) for k in (1, 2, 5)},
                             depth1_z=ld1 - ld1.rolling(288, min_periods=144).mean(), bid_near=b[1] / b[5], ask_near=a[1] / a[5],
                             d_age=E.d_age))[DCOLS]


def depth(sym):
    """sym = 'ETH' 등. 색인 = 5분봉 시작 τ, 값은 τ+5분에 확정."""
    return depth_from(pd.read_parquet(LIQ / f"{sym}USDT_5m.parquet"))


def lagged(D, days):
    """행 τ ← 깊이(τ − days 일). 지연 위약."""
    return D.shift(freq=pd.Timedelta(days=days)).reindex(D.index)


# ── 채점 ──────────────────────────────────────────────────────────────
def boot_means(x, b=5, n=4000, seed=7):
    rng = np.random.default_rng(seed); k = int(np.ceil(len(x) / b))
    ix = (rng.integers(0, len(x) - b + 1, size=(n, k))[:, :, None] + np.arange(b)).reshape(n, -1)[:, :len(x)]
    return x[ix].mean(1)


def lower_better(daily):
    """일 평균 차(B − A, 낮을수록 좋음) → 평균 · 단측 90% 상한 · 단측 p."""
    x = np.asarray(daily, float); bm = boot_means(x)
    return dict(mean=float(x.mean()), up90=float(np.quantile(bm, .9)), p=float((bm >= 0).mean()), days=len(x))


def daily(ts, v):
    return pd.Series(v, index=pd.DatetimeIndex(ts)).groupby(pd.DatetimeIndex(ts).normalize()).mean()


def grade(val_ok, strict_bound, test_ok, wins, extra=True):
    if val_ok and strict_bound and wins == len(LAGS) and extra:
        return "엄격 통과"
    if val_ok and test_ok and wins >= 5 and extra:
        return "후보"
    return "불통과"


def plant_col(y, seed=20261028):
    z = (y - np.nanmean(y)) / np.nanstd(y)
    return z + np.random.default_rng(seed).normal(0, 3, len(y))


def save(name, rep):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False, default=float))
    print(json.dumps(rep.get("verdict", rep), ensure_ascii=False, default=float), flush=True)


def brier_ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return (p - y) ** 2, -(y * np.log(p) + (1 - y) * np.log(1 - p))


# ── 장치 점검 S1 ──────────────────────────────────────────────────────
def selfcheck():
    rng = np.random.default_rng(1)
    for s in ("ETH", "BTC", "SOL", "XRP"):
        E = pd.read_parquet(LIQ / f"{s}USDT_5m.parquet"); full = depth_from(E)
        assert full.index[-1] == pd.Timestamp("2026-10-23 23:55"), full.index[-1]
        for i in rng.integers(2000, len(E) - 10, 5):
            cut = depth_from(E.iloc[:i + 1])
            assert np.allclose(cut.iloc[-300:].to_numpy(float), full.iloc[i - 299:i + 1].to_numpy(float), equal_nan=True), (s, E.index[i])
        L = lagged(full, 7); j = full.index.get_loc(pd.Timestamp("2025-03-10 12:00"))
        assert np.allclose(L.iloc[j].to_numpy(float), full.loc[pd.Timestamp("2025-03-03 12:00")].to_numpy(float), equal_nan=True)
        print(f"S1 {s} 절단 인과 5표본 · 지연 위약 정렬 통과 · 결측 {float(full.isna().any(axis=1).mean()):.4f}", flush=True)


# ── T1 dir4h ─────────────────────────────────────────────────────────
def dir4h():
    import live_eth_dir4h_20261015 as L
    from research_eth_dir4h_reverify_20261019 import ic_stats, pearson, month_fixed_q, phase_daily
    from sklearn.ensemble import HistGradientBoostingRegressor
    frames = {}
    for c in ["ETH"] + L.COINS:
        a = pd.read_parquet(ROOT / f"data/binance_vision/panel/{c}USDT.parquet"); b = pd.read_parquet(ROOT / f"data/live/dir4h_panel/{c}USDT.parquet")
        p = pd.concat([a, b]); p = p[p.timestamp >= "2023-01-01"].drop_duplicates("timestamp", keep="first").sort_values("timestamp")
        frames[c] = p.reset_index(drop=True)
    t0 = time.time(); F = L.build_features(frames); print(f"dir4h 피쳐 {len(F)}행 {time.time() - t0:.0f}s · 끝 {F.ts.iloc[-1]}", flush=True)
    ts, y = F.ts.to_numpy(), F.y_4h.to_numpy()
    D = depth("ETH")
    months = pd.date_range("2025-01-01", "2026-10-01", freq="MS"); PURGE = np.timedelta64(5 * L.H, "m")

    def walk(X):
        pred = np.full(len(ts), np.nan); ok = np.isfinite(y)
        for m in months:
            cut = np.datetime64(m) - PURGE
            tr = (ts >= cut - np.timedelta64(365, "D")) & (ts < cut) & ok
            te = (ts >= np.datetime64(m)) & (ts < np.datetime64(m + pd.offsets.MonthBegin(1)))
            pred[te] = HistGradientBoostingRegressor(**L.HGB).fit(X[tr][::2], y[tr][::2]).predict(X[te])
        return pred
    base = F[L.FEATS].to_numpy(np.float32)
    withd = lambda DD: np.column_stack([base, DD.reindex(pd.DatetimeIndex(ts)).to_numpy(np.float32)])  # noqa: E731
    fin = np.isfinite(y)
    uni = {c: round(float(pd.Series(D[c].reindex(pd.DatetimeIndex(ts)).to_numpy()[fin]).corr(pd.Series(y[fin]), method="spearman")), 4) for c in DCOLS}
    assert max(abs(v) for v in uni.values()) < .1, f"S4 단일 깊이 열 IC 과대 {uni}"
    win = {"val": (VA0, TE0), "test": (TE0, pd.Timestamp(ts[fin][-1]) + pd.Timedelta("5min"))}
    day = pd.DatetimeIndex(ts).normalize()

    def score(pa, pb):
        out = {}
        for k, (a, b) in win.items():
            m = (ts >= np.datetime64(a)) & (ts < np.datetime64(b)) & fin & np.isfinite(pa) & np.isfinite(pb)
            m &= ts + np.timedelta64(5 * L.H, "m") <= np.datetime64(b)          # 라벨 끝이 창 안
            S1, S0 = ic_stats(pb[m], y[m], day[m]), ic_stats(pa[m], y[m], day[m])
            ix = (np.random.default_rng(7).integers(0, len(S1) - 4, size=(4000, int(np.ceil(len(S1) / 5))))[:, :, None] + np.arange(5)).reshape(4000, -1)[:, :len(S1)]
            d = pearson(S1[ix].sum(1)) - pearson(S0[ix].sum(1))
            out[k] = dict(ic_a=float(pearson(S0.sum(0))), ic_b=float(pearson(S1.sum(0))), d=float(pearson(S1.sum(0)) - pearson(S0.sum(0))),
                          lo90=float(np.quantile(d, .1)), p=float((d <= 0).mean()), rows=int(m.sum()))
        return out
    pa = walk(base); pb = walk(withd(D)); r = score(pa, pb)
    rep = {"S4_univariate_spearman": uni, "A_ic_2025on": float(pearson(ic_stats(pa[fin & np.isfinite(pa)], y[fin & np.isfinite(pa)], day[fin & np.isfinite(pa)]).sum(0))),
           "real": r}
    pp = walk(np.column_stack([base, plant_col(y).astype(np.float32)])); rep["S2_plant"] = score(pa, pp)["test"]
    assert rep["S2_plant"]["lo90"] > 0, f"S2 양성 대조 실패 {rep['S2_plant']}"
    pnl = {}
    for nm, p in (("A", pa), ("B", pb)):
        lo, hi = month_fixed_q(p, ts, months)
        pnl[nm] = {k: phase_daily(p, lo, hi, ts, y, str(a.date()), str(b.date())) for k, (a, b) in win.items()}
    rep["pnl_bp_day_4bp"] = {k: dict(A=float(pnl["A"][k].mean()), B=float(pnl["B"][k].mean()), **lower_better(-(pnl["B"][k] - pnl["A"][k]).to_numpy())) for k in win}
    val_ok, test_ok = r["val"]["d"] > 0, r["test"]["d"] > 0
    rep["placebo"] = {}
    if val_ok and test_ok:
        for Lg in LAGS:
            rep["placebo"][Lg] = score(pa, walk(withd(lagged(D, Lg))))["test"]["d"]; print(f"  위약 {Lg}일 ΔIC {rep['placebo'][Lg]:+.4f}", flush=True)
    wins = sum(r["test"]["d"] > v for v in rep["placebo"].values())
    rep["verdict"] = dict(trial="T1 dir4h", val_d=r["val"]["d"], test_d=r["test"]["d"], test_lo90=r["test"]["lo90"], p=r["test"]["p"],
                          placebo_wins=wins, grade=grade(val_ok, r["test"]["lo90"] > 0, test_ok, wins))
    np.savez(OUT / "dir4h_preds.npz", ts=ts, y=y, pa=pa, pb=pb)
    save("dir4h", rep)


# ── T2~T4 박스 ETH ────────────────────────────────────────────────────
def box():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    R = pd.read_parquet(ROOT / "tmp/liqlab_20261028/box_rows.parquet"); B = joblib.load(ROOT / "dashboard/box_hold_hgb_20261016.joblib")
    D = depth("ETH")
    W = {"30m": 30, "1h": 60, "4h": 240}
    for s in B["scales"]:
        X = R[R.scale == s].sort_values("ts").reset_index(drop=True); cols = B["scales"][s]["cols"]
        assert (X.ts.dt.minute % 5 == 4).all(), s
        tau = pd.DatetimeIndex(X.ts) - pd.Timedelta("4min"); le = X.ts + pd.Timedelta(minutes=W[s] + 1)
        tr = ((X.ts >= TR0) & (le < VA0)).to_numpy(); va = ((X.ts >= VA0) & (le < TE0)).to_numpy(); te = (X.ts >= TE0).to_numpy()
        y = X.y.to_numpy(int)

        def fitp(M):
            return np.mean([HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                                           l2_regularization=1.0, random_state=sd).fit(M[tr], y[tr]).predict_proba(M[va | te])[:, 1]
                            for sd in BOX_SEEDS], 0)
        base = X[cols].to_numpy(float); withd = lambda DD: np.column_stack([base, DD.reindex(tau).to_numpy(float)])  # noqa: E731
        ev = np.flatnonzero(va | te); isv = va[ev]

        def score(pa, pb):
            out = {}
            for k, m in (("val", isv), ("test", ~isv)):
                yy, t = y[ev][m], X.ts.to_numpy()[ev][m]
                (ba, la), (bb, lb) = brier_ll(yy, pa[m]), brier_ll(yy, pb[m])
                out[k] = dict(brier_a=float(ba.mean()), brier_b=float(bb.mean()), **lower_better(daily(t, bb - ba).to_numpy()),
                              dll=float((lb - la).mean()), dauc=float(roc_auc_score(yy, pb[m]) - roc_auc_score(yy, pa[m])), rows=int(m.sum()))
            return out
        t0 = time.time(); pa = fitp(base); pb = fitp(withd(D)); r = score(pa, pb); print(f"박스 {s} A·B {time.time() - t0:.0f}s", flush=True)
        dep = B["scales"][s]["model"].predict_proba(X.loc[ev[~isv], cols])[:, 1] if "model" in B["scales"][s] else None
        rep = {"real": r, "deployed_as_is_brier_test": None if dep is None else float(brier_ll(y[ev][~isv], dep)[0].mean())}
        rep["S2_plant"] = score(pa, fitp(np.column_stack([base, plant_col(y.astype(float))])))["test"]
        assert rep["S2_plant"]["up90"] < 0, f"S2 양성 대조 실패 {s} {rep['S2_plant']}"
        val_ok, test_ok = r["val"]["mean"] < 0, r["test"]["mean"] < 0; rep["placebo"] = {}
        if val_ok and test_ok:
            for Lg in LAGS:
                rep["placebo"][Lg] = score(pa, fitp(withd(lagged(D, Lg))))["test"]["mean"]; print(f"  박스 {s} 위약 {Lg}일 {rep['placebo'][Lg]:+.6f}", flush=True)
        wins = sum(r["test"]["mean"] < v for v in rep["placebo"].values())
        rep["verdict"] = dict(trial=f"박스 ETH {s}", val_d=r["val"]["mean"], test_d=r["test"]["mean"], test_up90=r["test"]["up90"], p=r["test"]["p"],
                              placebo_wins=wins, grade=grade(val_ok, r["test"]["up90"] < 0, test_ok, wins))
        save(f"box_{s}", rep)


# ── T5 QX4 ───────────────────────────────────────────────────────────
def qx4():
    import torch
    _snt = torch.set_num_threads; torch.set_num_threads = lambda n: _snt(int(os.environ.get("TORCH_THREADS", 2)))   # 라이브 서버 — 기본 2 · S3 재현은 원 레시피 8
    import research_system_trend_ruleb_20261025 as SYS
    SYS.D = ROOT / "data"                                                          # 원 코드는 로컬 경로 고정
    import research_vol4h_forecast_horserace_20261025 as R
    assert all((SYS.OUT / f"rest_k5_{s}.parquet").exists() and (R.OUT / f"rest_vol_{s}.parquet").exists() for s in R.COINS), "REST 캐시 복사 먼저"
    fr, oks = {}, {}
    for s in R.COINS:
        x, F, y, ok, miss = R.frame(s); fr[s] = (F, y); oks[s] = (x.index, ok)
    R.add_mkt(fr)
    DD = {s: depth(s) for s in R.COINS}
    M = {}
    for s in R.COINS:
        (ts, ok), F = oks[s], fr[s][0]
        ok = ok & np.isfinite(F[R.MKT].to_numpy(float)).all(1); le = ts + pd.Timedelta(minutes=5 * R.H); dec = ok & (ts.minute == 55)
        M[s] = dict(tr=ok & (ts >= R.TR4_START) & (le <= R.TR4_END), va=dec & (ts >= R.VA4_START) & (le <= R.VA4_END), te=dec & (ts >= R.TE4_START))
    A = pd.concat([fr[s][0][M[s]["tr"]] for s in R.COINS]); ya = np.concatenate([fr[s][1][M[s]["tr"]] for s in R.COINS])
    E = pd.concat([pd.concat([fr[s][0][M[s]["va"]], fr[s][0][M[s]["te"]]]) for s in R.COINS])
    cuts = np.cumsum([M[s]["va"].sum() + M[s]["te"].sum() for s in R.COINS])[:-1]
    keys = {s: (fr[s][0].index[M[s]["tr"]], fr[s][0].index[M[s]["va"] | M[s]["te"]]) for s in R.COINS}

    def put(lagd):
        for nm, Z in (("A", A), ("E", E)):
            parts = [(DD[s] if lagd is None else lagged(DD[s], lagd)).reindex(keys[s][0 if nm == "A" else 1]).to_numpy(float) for s in R.COINS]
            Z[DCOLS] = np.concatenate(parts)

    def pooled(extra):
        R.SETS["M3XD"], R.SETS["M4XD"] = R.SETS["M3X"] + extra, R.SETS["M4X"] + extra
        t0 = time.time()
        o = {m: dict(zip(R.COINS, np.split(R.fit_predict(b, A, ya, E), cuts))) for m, b in (("PX3", "M3XD" if extra else "M3X"), ("PX6", "M6XD" if extra else "M6X"))}
        print(f"  합쳐 학습 +{len(extra)}열 {time.time() - t0:.0f}s", flush=True); return o
    frz = {s: (pd.read_parquet(ROOT / f"tmp/liqlab_20261028/qx4/pred_val_r4_{s}.parquet"), pd.read_parquet(ROOT / f"tmp/liqlab_20261028/qx4/pred_test_r4_{s}.parquet")) for s in R.COINS}

    def score(pxa, pxb):
        out = {}
        for s in R.COINS:
            F, y = fr[s]; ts = oks[s][0]; nv = M[s]["va"].sum(); tsv, tst = ts[M[s]["va"]], ts[M[s]["te"]]
            V, T = frz[s][0].reindex(tsv), frz[s][1].reindex(tst); assert V[["v_M3", "v_M6"]].notna().all().all() and T[["t_M3", "t_M6"]].notna().all().all(), s
            q = {}
            for arm, px in (("A", pxa), ("B", pxb)):
                pv = np.mean([V.v_M3, V.v_M6, px["PX3"][s][:nv], px["PX6"][s][:nv]], 0); pt = np.mean([T.t_M3, T.t_M6, px["PX3"][s][nv:], px["PX6"][s][nv:]], 0)
                c = float(np.log(np.mean(np.exp(y[M[s]["va"]] - pv))))
                q[arm] = (R.qlike(y[M[s]["va"]], pv + c), R.qlike(y[M[s]["te"]], pt + c))
            out[s] = {"val": lower_better(daily(tsv, q["B"][0] - q["A"][0]).to_numpy()), "test": lower_better(daily(tst, q["B"][1] - q["A"][1]).to_numpy())}
        return out
    put(None)
    pa = pooled([])
    rep = {"S3_repro_corr": {}}
    for s in R.COINS:
        nv = M[s]["va"].sum(); V, T = frz[s][0].reindex(keys[s][1][:nv]), frz[s][1].reindex(keys[s][1][nv:])
        for m in ("PX3", "PX6"):
            fz = np.r_[V[f"v_{m}"].to_numpy(), T[f"t_{m}"].to_numpy()]; rep["S3_repro_corr"][f"{s}_{m}"] = float(np.corrcoef(fz, pa[m][s])[0, 1])
    print("S3", rep["S3_repro_corr"], flush=True)
    # §8 사후 변경(QX4 결과 전): MLP(PX6)는 기계 간 수치 차로 .996~.997(스레드 2·8 같음) → 결정적 PX3 만 관문, PX6 는 기록
    assert min(v for k, v in rep["S3_repro_corr"].items() if k.endswith("PX3")) >= .999, "S3 재현 실패"
    pb = pooled(DCOLS); r = score(pa, pb); rep["real"] = r
    A["plant"] = plant_col(ya); E["plant"] = plant_col(np.concatenate([np.r_[fr[s][1][M[s]["va"]], fr[s][1][M[s]["te"]]] for s in R.COINS]), seed=7)
    rep["S2_plant"] = score(pa, pooled(["plant"]))["ETH"]["test"]
    assert rep["S2_plant"]["up90"] < 0, f"S2 양성 대조 실패 {rep['S2_plant']}"
    e = r["ETH"]; ncoin = sum(r[s]["test"]["mean"] < 0 for s in R.COINS)
    val_ok, test_ok = e["val"]["mean"] < 0, e["test"]["mean"] < 0; rep["placebo"] = {}
    if val_ok and test_ok and ncoin >= 3:
        for Lg in LAGS:
            put(Lg); rep["placebo"][Lg] = score(pa, pooled(DCOLS))["ETH"]["test"]["mean"]; print(f"  QX4 위약 {Lg}일 {rep['placebo'][Lg]:+.5f}", flush=True)
    wins = sum(e["test"]["mean"] < v for v in rep["placebo"].values())
    rep["verdict"] = dict(trial="QX4 (ETH 주·4코인 복제)", val_d=e["val"]["mean"], test_d=e["test"]["mean"], test_up90=e["test"]["up90"], p=e["test"]["p"],
                          coins_improved=ncoin, placebo_wins=wins, grade=grade(val_ok, e["test"]["up90"] < 0, test_ok, wins, ncoin >= 3))
    save("qx4", rep)


# ── T6·T7 card30 ─────────────────────────────────────────────────────
def card30():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from scipy.optimize import minimize_scalar
    P = ROOT / "tmp/card30_direction_20260926"
    X0 = pd.read_parquet(P / "dataset.parquet"); K = json.loads((P / "feats.json").read_text())["K"]; D = depth("ETH")
    lg = lambda p: np.log(p / (1 - p)); sg = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    for task in ("dir", "reach"):
        X = X0[X0.res.isin((0, 1))] if task == "dir" else X0[X0.res.isin((0, 1, 2, -1))]
        X = X.reset_index(drop=True); y = ((X.res == 1) if task == "dir" else (X.res != 2)).to_numpy(int)
        le = X.ts + pd.Timedelta("35min")
        tr = ((X.ts >= TR0) & (le < VA0)).to_numpy(); va = ((X.ts >= VA0) & (le < TE0)).to_numpy(); te = (X.ts >= TE0).to_numpy()
        base = X[K].to_numpy(float); tau = pd.DatetimeIndex(X.ts); withd = lambda DD: np.column_stack([base, DD.reindex(tau).to_numpy(float)])  # noqa: E731

        def fitp(M):
            ms = [HistGradientBoostingClassifier(max_iter=500, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0,
                                                 early_stopping=True, validation_fraction=0.15, n_iter_no_change=30, random_state=sd).fit(M[tr], y[tr])
                  for sd in CARD_SEEDS]
            pv, pt = (np.clip(np.mean([m.predict_proba(M[w])[:, 1] for m in ms], 0), 1e-6, 1 - 1e-6) for w in (va, te))
            bs = float(y[va].mean())
            k = float(minimize_scalar(lambda k: brier_ll(y[va], sg(k * lg(pv) + (1 - k) * lg(bs)))[1].mean(), bounds=(0, 2), method="bounded").x)
            return sg(k * lg(pv) + (1 - k) * lg(bs)), sg(k * lg(pt) + (1 - k) * lg(bs))

        def score(pa, pb):
            out = {}
            for kk, w, i in (("val", va, 0), ("test", te, 1)):
                yy, t = y[w], X.ts.to_numpy()[w]
                (ba, la), (bb, lb) = brier_ll(yy, pa[i]), brier_ll(yy, pb[i])
                out[kk] = dict(brier_a=float(ba.mean()), brier_b=float(bb.mean()), **lower_better(daily(t, bb - ba).to_numpy()),
                               dll=float((lb - la).mean()), auc_a=float(roc_auc_score(yy, pa[i])), dauc=float(roc_auc_score(yy, pb[i]) - roc_auc_score(yy, pa[i])), rows=int(w.sum()))
            return out
        t0 = time.time(); pa = fitp(base); pb = fitp(withd(D)); r = score(pa, pb); print(f"card30 {task} A·B {time.time() - t0:.0f}s", flush=True)
        rep = {"real": r, "S2_plant": score(pa, fitp(np.column_stack([base, plant_col(y.astype(float))])))["test"]}
        assert rep["S2_plant"]["up90"] < 0, f"S2 양성 대조 실패 {task} {rep['S2_plant']}"
        val_ok, test_ok = r["val"]["mean"] < 0, r["test"]["mean"] < 0; rep["placebo"] = {}
        if val_ok and test_ok:
            for Lg in LAGS:
                rep["placebo"][Lg] = score(pa, fitp(withd(lagged(D, Lg))))["test"]["mean"]; print(f"  card30 {task} 위약 {Lg}일 {rep['placebo'][Lg]:+.6f}", flush=True)
        wins = sum(r["test"]["mean"] < v for v in rep["placebo"].values())
        rep["verdict"] = dict(trial=f"card30 {task}", val_d=r["val"]["mean"], test_d=r["test"]["mean"], test_up90=r["test"]["up90"], p=r["test"]["p"],
                              placebo_wins=wins, grade=grade(val_ok, r["test"]["up90"] < 0, test_ok, wins))
        save(f"card30_{task}", rep)


def report():
    vs = []
    for f in ("dir4h", "box_30m", "box_1h", "box_4h", "qx4", "card30_dir", "card30_reach"):
        p = OUT / f"{f}.json"
        if p.exists():
            vs.append(json.loads(p.read_text())["verdict"])
    ps = sorted((v["p"], i) for i, v in enumerate(vs)); m = len(vs)
    alive = True
    for r, (p, i) in enumerate(ps):          # Holm α .10 (단측 p)
        alive = alive and p <= .10 / (m - r); vs[i]["holm_pass"] = alive
    save("summary", {"trials": m, "verdict": vs})


if __name__ == "__main__":
    {"selfcheck": selfcheck, "dir4h": dir4h, "box": box, "qx4": qx4, "card30": card30, "report": report}[sys.argv[1]]()
