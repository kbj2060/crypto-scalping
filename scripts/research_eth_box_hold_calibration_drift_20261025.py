#!/usr/bin/env python3
"""박스 유지 확률 — 국면 이동에 강한 보정 (2026-10-25). 사전등록 docs/eth_box_hold_calibration_drift_prereg_20261025.md.

  python scripts/research_eth_box_hold_calibration_drift_20261025.py --selftest
  python scripts/research_eth_box_hold_calibration_drift_20261025.py --build     # 1분봉~10-23 · 격자 · 라벨(독립) · 피쳐 · 관문 L1~L4
  python scripts/research_eth_box_hold_calibration_drift_20261025.py --round 1   # 후보 예측 · 2025 선택 · 2026 1회 채점
"""
from __future__ import annotations

import glob
import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression, isotonic_regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
sys.path.insert(0, str(ROOT))
from dashboard.box_hold import B as BH, NEED, features as dash_features  # noqa: E402

BV = MAIN / "data/binance_vision"
DEC = MAIN / ".claude/worktrees/pingpong-long-short-hedge-de2ac6/tmp/range_hold_20261016/decisions.parquet"   # L1 대조용(~10-14)
OUT = ROOT / "tmp/box_calib_20261025"
VIS = OUT / "vision_1m"                 # data.binance.vision 일 파일 10-01~10-23 (10-19~23 CHECKSUM 대조해 받음)
END = pd.Timestamp("2026-10-23 23:59")  # 이 분(봉 시작)까지만 쓴다
WIN = {"30m": 30, "1h": 60, "4h": 240}
F_BASE = ["log_ds", "log_vr", "log_rng", "hs", "hc", "wkd"]
LEAN = BH["scales"]["1h"]["cols"]
MIN = np.timedelta64(1, "m")
log = lambda m: print(m, flush=True)  # noqa: E731


def load_1m() -> pd.DataFrame:
    parts = [pd.read_parquet(f)[["t", "h", "l", "c", "v", "tb"]] for f in sorted(glob.glob(f"{BV}/klines1m/ETHUSDT-1m-*.parquet")) +
             sorted(glob.glob(f"{BV}/klines1m_daily/ETHUSDT-1m-*.parquet"))]
    for z in sorted(glob.glob(str(VIS / "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({"t": x.open_time, "h": x.high, "l": x.low, "c": x.close, "v": x.volume, "tb": x.taker_buy_volume}))
    K = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    K = K[K.t <= END.value // 10 ** 6].reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), "1분봉 빈칸"
    assert K.t.iloc[-1] == END.value // 10 ** 6, "10-23 23:59 까지 없음"
    return K


def labels(h: np.ndarray, l: np.ndarray, c: np.ndarray, W: int) -> np.ndarray:
    """독립 재구현(창 뷰): y[i] = max h[i+1..i+W] < c[i]+½(max h[i−W+1..i] − min l[i−W+1..i]) & min l[i+1..i+W] > c[i]−½(…). 못 정하면 −1."""
    n = len(c)
    mx, mn = sliding_window_view(h, W).max(1), sliding_window_view(l, W).min(1)   # mx[k] = max h[k..k+W−1]
    y = np.full(n, -1, np.int8)
    i = np.arange(W - 1, n - W)                    # 과거 창 완결 · 미래 창 완결
    half = 0.5 * (mx[i - W + 1] - mn[i - W + 1])
    y[i] = ((mx[i + 1] < c[i] + half) & (mn[i + 1] > c[i] - half)).astype(np.int8)
    return y


def base_rate(y: np.ndarray, W: int, days: int = 30) -> np.ndarray:
    """국면 피쳐 B_rate30[i] = 매분 라벨 y[j](j ∈ [i−days·1440, i−W], 즉 i 마감까지 해결된 것)의 평균. 유효 라벨 절반 미만이면 NaN."""
    ok = (y >= 0).astype(float)
    cs, cn = np.r_[0, np.cumsum(np.where(y >= 0, y, 0))], np.r_[0, np.cumsum(ok)]
    i = np.arange(len(y))
    hi = i - W + 1                                  # j ≤ i−W → 누적합 끝(배타) = i−W+1
    lo = np.maximum(i - days * 1440, 0)
    hi = np.maximum(hi, lo)
    s, m = cs[hi] - cs[lo], cn[hi] - cn[lo]
    return np.where(m >= days * 1440 / 2, s / np.maximum(m, 1), np.nan)


def load_imp():
    spec = importlib.util.spec_from_file_location("imp", ROOT / "scripts/research_eth_box_hold_improve_20261016.py")
    imp = importlib.util.module_from_spec(spec); spec.loader.exec_module(imp)
    return imp


def base_feats(K: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    """기하 6칸(dashboard features 와 같은 식) -- 원 decisions.parquet 는 ~10-14 라 직접 만든다(L1 에서 대조)."""
    h, l, c = (K[k].to_numpy(float) for k in ("h", "l", "c"))
    r2 = np.r_[0, np.diff(np.log(c))] ** 2
    s24 = np.sqrt(pd.Series(r2).rolling(1440).mean().to_numpy())
    s60 = np.sqrt(pd.Series(r2).rolling(60).mean().to_numpy())
    i = D.i.to_numpy()
    W = D.scale.map(WIN).to_numpy()
    rng = np.empty(len(D))
    for nm, w in WIN.items():
        m = W == w
        rr = pd.Series(h).rolling(w).max().to_numpy() - pd.Series(l).rolling(w).min().to_numpy()
        rng[m] = rr[i[m]]
    ts = pd.DatetimeIndex(D.ts)
    hr = ts.hour + ts.minute / 60
    return pd.DataFrame({"log_ds": np.log(0.5 * rng / c[i] / (s24[i] * np.sqrt(W))), "log_vr": np.log(np.maximum(s60[i], 1e-6) / s24[i]),
                         "log_rng": np.log(rng / c[i] * 1e4), "hs": np.sin(2 * np.pi * hr / 24), "hc": np.cos(2 * np.pi * hr / 24),
                         "wkd": (ts.dayofweek >= 5).astype(int)}, index=D.index)


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    K = load_1m()
    t = K.t.to_numpy()
    tsK = pd.to_datetime(t, unit="ms")
    mins = t // 60000
    h, l, c = (K[k].to_numpy(float) for k in ("h", "l", "c"))
    start = np.searchsorted(t, pd.Timestamp("2022-01-01 23:59").value // 10 ** 6)
    rows = []
    Y = {}
    for sc, W in WIN.items():
        Y[sc] = labels(h, l, c, W)
        step = 60 if sc == "4h" else 5
        ii = np.arange(start, len(K))
        rngW = pd.Series(h).rolling(W).max().to_numpy() - pd.Series(l).rolling(W).min().to_numpy()
        ii = ii[(mins[ii] % step == step - 1) & (Y[sc][ii] >= 0) & (rngW[ii] > 0)]   # 평평한 창 = 대시보드 ok False
        rows.append(pd.DataFrame({"scale": sc, "ts": tsK[ii], "i": ii, "y": Y[sc][ii], "B_rate30": base_rate(Y[sc], W)[ii]}))
    D = pd.concat(rows, ignore_index=True)
    log(f"격자 {D.groupby('scale').ts.agg(['min', 'max', 'count']).to_dict()}")
    # L1 기존 산출과 같은 행·같은 정답·같은 기하 칸
    O = pd.read_parquet(DEC)
    O = O[O.scale.isin(list(WIN))]
    m = D.merge(O, on=["scale", "ts"], how="outer", indicator=True)
    cut = O.ts.max() - pd.Timedelta("4h")
    mm = m[m.ts <= cut]
    assert (mm._merge == "both").all(), mm._merge.value_counts()
    assert (mm.y == mm.hold).all(), (mm.y != mm.hold).sum()
    G = base_feats(K, D)
    chk = D[["scale", "ts"]].join(G).merge(O, on=["scale", "ts"], suffixes=("", "_o"))
    for col in F_BASE:
        assert np.allclose(chk[col], chk[col + "_o"], rtol=1e-9, atol=1e-9, equal_nan=True), col
    log(f"L1 OK — 겹치는 {len(mm):,}행 라벨 100% 일치 · 기하 6칸 일치")
    # L2 반복문 직접 계산
    rs = np.random.default_rng(7)
    for k in rs.choice(len(D), 3000, replace=False):
        i, W = int(D.i[k]), WIN[D.scale[k]]
        half = 0.5 * (h[i - W + 1:i + 1].max() - l[i - W + 1:i + 1].min())
        want = int(max(h[i + 1:i + W + 1]) < c[i] + half and min(l[i + 1:i + W + 1]) > c[i] - half)
        assert D.y[k] == want, (D.scale[k], D.ts[k])
    log("L2 OK — 무작위 3,000행 반복문 일치")
    # L4 국면 피쳐: i−W 이후 라벨을 망가뜨려도 불변
    for sc, W in WIN.items():
        for i in rs.integers(100000, len(K) - 1000, 20):
            y2 = Y[sc].copy(); y2[i - W + 1:] = 1 - np.clip(y2[i - W + 1:], 0, 1)
            a, b = base_rate(Y[sc], W)[i], base_rate(y2, W)[i]
            assert (np.isnan(a) and np.isnan(b)) or a == b, (sc, i)
    log("L4 OK — 국면 피쳐는 i−W 까지 해결된 라벨만")
    # 피쳐(배포 칸) = 배포 연구 features() 그대로
    imp = load_imp()
    X = imp.features(D.assign(ts=D.ts), K)
    assert np.allclose(imp.PROF, np.asarray(BH["prof"], float)), "요일×시각 표가 배포와 다름"
    A = pd.concat([D, G, X[[x for x in LEAN if x not in F_BASE]]], axis=1)
    # L3 경계: 대시보드 features() 를 i 까지 자른 1분봉으로 = 연구 피쳐 (10-15~ 행 포함)
    evl = json.loads((imp.OUT / "events.json").read_text())
    iso = lambda v: pd.Timestamp(int(v), unit="ms", tz="UTC").isoformat()  # noqa: E731
    evl = [{"time_utc": iso(v), "category": "fomc", "title_ko": "FOMC 금리결정 발표"} for v in evl["major"]] + \
          [{"time_utc": iso(v), "category": "econ", "detail": "release_id=180"} for v in evl["claims"]]
    pick = pd.concat([A[A.ts >= "2026-10-15"].sample(20, random_state=1), A[A.M_major_in == 1].sample(10, random_state=2), A.sample(10, random_state=3)])
    worst = 0.0
    for _, row in pick.iterrows():
        i = int(row.i)
        k = K.iloc[i - NEED + 1:i + 1]
        k1m = pd.DataFrame({"timestamp": pd.to_datetime(k.t, unit="ms", utc=True), "high": k.h, "low": k.l, "close": k.c,
                            "volume": k.v, "taker_buy_base": k.tb}).reset_index(drop=True)
        f = dash_features(k1m, evl)[row.scale]
        cols = BH["scales"][row.scale]["cols"]   # 4h 배포 = 기하 6칸 (4h R_squeeze 는 서빙 1499봉 nanmedian ≠ 연구 1440 중앙값 -- 4h 모델이 안 씀)
        a, b = np.array([f[x] for x in cols], float), row[cols].to_numpy(float)
        ok = np.isfinite(b)
        worst = max(worst, float(np.max(np.abs(a[ok] - b[ok]))))
        assert np.allclose(a[ok], b[ok], rtol=1e-6, atol=1e-8), (row.scale, row.ts, {x: (u, v) for x, u, v in zip(cols, a, b) if not np.isclose(u, v, rtol=1e-6, atol=1e-8)})
    log(f"L3 OK — i 까지 자른 1분봉의 대시보드 피쳐 = 연구 피쳐 {len(pick)}행(10-15~ 20행) 최대 차 {worst:.1e}")
    A = A.replace([np.inf, -np.inf], np.nan)
    for sc in WIN:
        n = len(A[(A.scale == sc) & (A.ts < "2025-01-01")].dropna(subset=LEAN))
        assert n == BH["scales"][sc]["n"], (sc, n, BH["scales"][sc]["n"])
    log("배포 학습 행 수 재현 OK")
    A.drop(columns=["i"]).to_parquet(OUT / "rows.parquet")
    log(f"→ {OUT / 'rows.parquet'} {len(A):,}행")


# ---------- 보정 ----------
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sig(z):
    return 1 / (1 + np.exp(-z))


def fit_platt(p, y):
    m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p)[:, None], y)
    return float(m.coef_[0, 0]), float(m.intercept_[0])


def fit_shift(p, y):
    z, d = logit(p), 0.0
    for _ in range(50):                             # 절편 하나 뉴턴
        q = sig(z + d)
        g, H = np.sum(q - y), np.sum(q * (1 - q))
        d -= g / H
        if abs(g / H) < 1e-10:
            break
    return d


def beta_x(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.c_[np.log(p), -np.log(1 - p)]


def static(kind, p_fit, y_fit, p_app):
    if kind == "platt":
        a, b = fit_platt(p_fit, y_fit); return sig(a * logit(p_app) + b)
    if kind == "beta":   # ponytail: 제약 없는 3모수 로지스틱(Kull 의 a,b≥0 처리 생략) -- 비단조가 나오면 a·b 부호 보고
        m = LogisticRegression(C=1e6, max_iter=1000).fit(beta_x(p_fit), y_fit); return m.predict_proba(beta_x(p_app))[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4).fit(p_fit, y_fit)
    return iso.predict(p_app)


def venn_abers(pf, yf, pa):
    """귀납 벤–에이버스(Vovk·Petej 2014 · IVAP 2015): 보정 집합 ∪ {(z,0)} / ∪ {(z,1)} 등조 회귀의 z 값 → (p0, p1). 같은 점수는 한 칸."""
    u, inv = np.unique(pf, return_inverse=True); s, n = np.bincount(inv, yf), np.bincount(inv).astype(float)
    za, ia = np.unique(pa, return_inverse=True); p0, p1 = np.empty(len(za)), np.empty(len(za))
    for k, z in enumerate(za):
        i = int(np.searchsorted(u, z)); tie = i < len(u) and u[i] == z
        for lab, o in ((0.0, p0), (1.0, p1)):
            s2, n2 = (s.copy(), n.copy()) if tie else (np.insert(s, i, 0.0), np.insert(n, i, 0.0))
            s2[i] += lab; n2[i] += 1
            o[k] = isotonic_regression(s2 / n2, sample_weight=n2)[i]
    return p0[ia], p1[ia]


def rolling(kind, ts, p, y, W, N, t0="2025-01-01"):
    """매월 1일 00:00 다시 맞춤 · 직전 N일 · 해결된 행(ts+W+1분 ≤ 맞춤 시각) · t0 이후만 · 28일 미만이면 항등."""
    out = p.copy()
    t0 = np.datetime64(t0)
    months = pd.date_range(t0, pd.Timestamp(ts.max()) + pd.offsets.MonthBegin(1), freq="MS").to_numpy()
    for T, T2 in zip(months[:-1], months[1:]):
        app = (ts >= T) & (ts < T2)
        if not app.any():
            continue
        w0 = max(T - np.timedelta64(N, "D"), t0)
        fit = (ts >= w0) & (ts + (W + 1) * MIN <= T)
        assert not fit.any() or (ts[fit] + (W + 1) * MIN).max() <= T          # L5
        if T - w0 < np.timedelta64(28, "D") or fit.sum() < 100:
            continue
        if kind == "platt":
            a, b = fit_platt(p[fit], y[fit]); out[app] = sig(a * logit(p[app]) + b)
        elif kind == "va":
            p0, p1 = venn_abers(p[fit], y[fit], p[app]); out[app] = p1 / (1 - p0 + p1)
        else:
            out[app] = sig(logit(p[app]) + fit_shift(p[fit], y[fit]))
    return out


def ops(ts, p, y, W, t0="2025-01-01", gamma=0.1, rho=100.0):
    """Online Platt Scaling (Gupta·Ramdas 2023) ONS -- 지연 라벨: 행 k 는 ts_k + W ≤ ts_i 일 때만 갱신에 들어간다."""
    out = p.copy()
    idx = np.where(ts >= np.datetime64(t0))[0]
    th, A = np.array([1.0, 0.0]), rho * np.eye(2)
    z = logit(p)
    res = ts + W * MIN
    k = 0
    for n, i in enumerate(idx):
        while k < n and res[idx[k]] <= ts[i]:      # L5: 해결된 것만
            j = idx[k]; x = np.array([z[j], 1.0])
            g = (sig(th @ x) - y[j]) * x
            A += np.outer(g, g)
            th = th - np.linalg.solve(A, g) / gamma
            nr = np.linalg.norm(th)
            if nr > 100:                            # ponytail: 유클리드 공 사영(논문은 A-노름) -- 반지름 100 은 실제로 안 닿는다
                th *= 100 / nr
            k += 1
        out[i] = sig(th @ np.array([z[i], 1.0]))
    return out


# ---------- 채점 ----------
def ece(y, p, nb=10):
    q = pd.qcut(p, nb, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
    return float((g.n * (g.p - g.y).abs()).sum() / g.n.sum()), float((g.p - g.y).abs().max()), g


def metrics(y, p, q20, q80):
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    e, mx, _ = ece(y, p)
    lo, hi = p < q20, p > q80
    return {"brier": float(np.mean((p - y) ** 2)), "logloss": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
            "ece_pp": e * 100, "calmax_pp": mx * 100, "auc": float(roc_auc_score(y, p)), "bias_pp": float((p.mean() - y.mean()) * 100),
            "break_share": float(lo.mean()), "break_hold": float(y[lo].mean()) if lo.any() else None,
            "stay_share": float(hi.mean()), "stay_hold": float(y[hi].mean()) if hi.any() else None}


def day_boot(vals, days, B=2000, seed=0, alpha=0.05):
    """일 블록 부트스트랩 (1−alpha) CI of mean(vals) (일 합·건수로 빠르게)."""
    codes, uniq = pd.factorize(days)
    s, n = np.bincount(codes, vals), np.bincount(codes).astype(float)
    rng = np.random.default_rng(seed)
    w = rng.multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=B)
    est = (w @ s) / (w @ n)
    return [float(np.quantile(est, alpha / 2)), float(np.quantile(est, 1 - alpha / 2))]


def auc_boot(y, a, b, days, B=300, seed=1):
    codes, uniq = pd.factorize(days)
    order = np.argsort(codes, kind="stable"); st = np.searchsorted(codes[order], np.arange(len(uniq))); en = np.r_[st[1:], len(codes)]
    bl = [order[x:z] for x, z in zip(st, en)]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        ii = np.concatenate([bl[k] for k in rng.integers(0, len(bl), len(bl))])
        out.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [float(np.quantile(out, .025)), float(np.quantile(out, .975))]


HGB = dict(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500, l2_regularization=1.0, random_state=0)


def candidates_r1(sc, A):
    """배포 행(2025~) 위 후보 예측 dict, 2025 선택 점수용 교차적합 예측 dict."""
    W, cf = WIN[sc], BH["scales"][sc]
    g = A[A.scale == sc]
    tr = g[g.ts < "2025-01-01"].dropna(subset=LEAN)
    te = g[g.ts >= "2025-01-01"].dropna(subset=cf["cols"]).reset_index(drop=True)
    ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
    p0 = cf["model"].predict_proba(te[cf["cols"]])[:, 1]
    P = {"C0 배포": p0}
    P["C1 기하로지스틱"] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[F_BASE], tr.y).predict_proba(te[F_BASE])[:, 1]
    c8 = cf["cols"] + ["B_rate30"]
    P["C8 국면피쳐HGB"] = HistGradientBoostingClassifier(**HGB).fit(tr[c8], tr.y).predict_proba(te[c8])[:, 1]
    y25 = ts < np.datetime64("2026-01-01")
    h1 = ts < np.datetime64("2025-07-01")
    sel = {}                                          # 2025 선택 점수용(정적 = 반기 교차 적합)
    for kind, code in (("platt", "C2 정적Platt"), ("beta", "C3 정적베타"), ("iso", "C4 정적등조")):
        P[code] = np.where(y25, np.nan, static(kind, p0[y25], y[y25], p0))
        cv = np.full(len(p0), np.nan)
        cv[y25 & h1] = static(kind, p0[y25 & ~h1], y[y25 & ~h1], p0[y25 & h1])
        cv[y25 & ~h1] = static(kind, p0[y25 & h1], y[y25 & h1], p0[y25 & ~h1])
        sel[code] = cv
    for N in (60, 90, 180):
        P[f"C5 굴림Platt{N}"] = rolling("platt", ts, p0, y, W, N)
        P[f"C6 굴림절편{N}"] = rolling("shift", ts, p0, y, W, N)
    P["C7 OPS"] = ops(ts, p0, y, W)
    for k, v in P.items():
        sel.setdefault(k, v)
    return te, y, P, sel


def round1():
    A = pd.read_parquet(OUT / "rows.parquet")
    rep = {}
    for sc in WIN:
        cf = BH["scales"][sc]
        te, y, P, sel = candidates_r1(sc, A)
        ts = te.ts.to_numpy()
        m25 = ts < np.datetime64("2026-01-01")
        score25 = {k: float(np.mean((v[m25] - y[m25]) ** 2)) for k, v in sel.items()}
        S = min((k for k in score25 if k != "C0 배포"), key=score25.get)
        log(f"\n[{sc}] 2025 Brier(선택용) " + " · ".join(f"{k} {v:.5f}" for k, v in sorted(score25.items(), key=lambda kv: kv[1])) + f"\n  → 선택 S = {S}")
        m26 = ~m25
        y6, d6 = y[m26], pd.DatetimeIndex(ts[m26]).normalize().to_numpy()
        p06 = P["C0 배포"][m26]
        r = {"n_2026": int(m26.sum()), "days_2026": int(len(np.unique(d6))), "hold_2026": float(y6.mean()), "select_2025_brier": score25, "S": S, "cands": {}}
        base = metrics(y6, p06, cf["q20"], cf["q80"])
        for k, v in P.items():
            p6 = v[m26]
            mt = metrics(y6, p6, cf["q20"], cf["q80"])
            if k != "C0 배포":
                pc, p0c = np.clip(p6, 1e-6, 1 - 1e-6), np.clip(p06, 1e-6, 1 - 1e-6)
                mt["dbrier"] = mt["brier"] - base["brier"]
                mt["dbrier_ci"] = day_boot((p6 - y6) ** 2 - (p06 - y6) ** 2, d6)
                ll = lambda q: -(y6 * np.log(q) + (1 - y6) * np.log(1 - q))  # noqa: E731
                mt["dlogloss_ci"] = day_boot(ll(pc) - ll(p0c), d6)
                mt["dauc"] = mt["auc"] - base["auc"]
                mt["dauc_ci"] = auc_boot(y6, p6, p06, d6)
            mt["bias_ci_pp"] = [x * 100 for x in day_boot(p6 - y6, d6)]
            r["cands"][k] = mt
            log(f"  {k:14s} Brier {mt['brier']:.5f} {('Δ%+.5f %s' % (mt['dbrier'], np.round(mt['dbrier_ci'], 5).tolist())) if 'dbrier' in mt else '':32s} "
                f"LL {mt['logloss']:.4f} ECE {mt['ece_pp']:.2f}pp max {mt['calmax_pp']:.1f} AUC {mt['auc']:.4f} 치우침 {mt['bias_pp']:+.2f}{np.round(mt['bias_ci_pp'], 2).tolist()} "
                f"깨질쪽 {mt['break_share'] * 100:.0f}%→{(mt['break_hold'] or 0) * 100:.1f}% 머물쪽 {mt['stay_share'] * 100:.0f}%→{(mt['stay_hold'] or 0) * 100:.1f}%")
        s = r["cands"][S]
        r["verdict"] = {"W1": s["dbrier_ci"][1] < 0, "W2": s["logloss"] <= base["logloss"], "W3": s["ece_pp"] <= base["ece_pp"], "W4": s["auc"] >= base["auc"] - 0.002}
        r["win"] = all(r["verdict"].values())
        log(f"  판정 S={S}: {r['verdict']} → {'이김' if r['win'] else '못 이김'}")
        r["reliability_2026"] = {k: ece(y6, P[k][m26])[2].round(4).to_dict("list") for k in ("C0 배포", S)}
        if sc == "4h":
            dts = pd.DatetimeIndex(ts[m26])
            last90 = dts >= pd.Timestamp("2026-07-26")
            r["bias_4h"] = {}
            for k in ("C0 배포", S, "C1 기하로지스틱"):
                p6 = P[k][m26]
                r["bias_4h"][k] = {"2026": [float((p6.mean() - y6.mean()) * 100)] + [x * 100 for x in day_boot(p6 - y6, d6)],
                                   "최근90일": [float((p6[last90].mean() - y6[last90].mean()) * 100)] + [x * 100 for x in day_boot((p6 - y6)[last90], d6[last90])],
                                   "pred_real_90": [float(p6[last90].mean()), float(y6[last90].mean())],
                                   "월별": {mo: round(float((p6[dts.month == mo].mean() - y6[dts.month == mo].mean()) * 100), 2) for mo in range(1, 11)}}
                log(f"  4h 치우침 {k}: {r['bias_4h'][k]}")
        rep[sc] = r
    (OUT / "round1.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'round1.json'}")


def round2():
    """2라운드(사전등록 2라운드 표) -- 후보 고정, 본페로니 CI."""
    A = pd.read_parquet(OUT / "rows.parquet")
    rep = {}
    for sc, W in WIN.items():
        cf = BH["scales"][sc]
        g = A[A.scale == sc]
        te = g[g.ts >= "2026-01-01"].dropna(subset=cf["cols"]).reset_index(drop=True)
        ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
        P = {"C0 배포": cf["model"].predict_proba(te[cf["cols"]])[:, 1]}
        tr = g[g.ts + pd.Timedelta(minutes=W + 1) <= "2026-01-01"].dropna(subset=LEAN)          # 라벨 창이 2026 전에 끝남
        assert (tr.ts + pd.Timedelta(minutes=W + 1)).max() <= pd.Timestamp("2026-01-01")
        m2 = HistGradientBoostingClassifier(**HGB).fit(tr[cf["cols"]], tr.y)
        P["R2a 재학습~2025"] = m2.predict_proba(te[cf["cols"]])[:, 1]
        P["R2b 재학습+굴림절편180"] = rolling("shift", ts, P["R2a 재학습~2025"], y, W, 180, t0="2026-01-01")
        info = {}
        if sc == "4h":
            tr24 = g[g.ts < "2025-01-01"].dropna(subset=LEAN)
            m27 = HistGradientBoostingClassifier(**HGB).fit(tr24[LEAN], tr24.y)
            g25 = g[g.ts >= "2025-01-01"].dropna(subset=LEAN).reset_index(drop=True)      # 27칸 유한 행(2025~)
            t25, y25a = g25.ts.to_numpy(), g25.y.to_numpy().astype(float)
            q = m27.predict_proba(g25[LEAN])[:, 1]
            in25 = t25 < np.datetime64("2026-01-01"); h1 = t25 < np.datetime64("2025-07-01")
            cv = np.full(len(q), np.nan)
            cv[in25 & h1] = static("platt", q[in25 & ~h1], y25a[in25 & ~h1], q[in25 & h1])
            cv[in25 & ~h1] = static("platt", q[in25 & h1], y25a[in25 & h1], q[in25 & ~h1])
            roll180 = rolling("platt", t25, q, y25a, W, 180)
            sc25 = {"정적Platt": float(np.mean((cv[in25] - y25a[in25]) ** 2)), "굴림Platt180": float(np.mean((roll180[in25] - y25a[in25]) ** 2)),
                    "27칸 그대로": float(np.mean((q[in25] - y25a[in25]) ** 2)), "C0 배포(같은 행)": float(np.mean((cf["model"].predict_proba(g25[cf["cols"]])[:, 1][in25] - y25a[in25]) ** 2))}
            pick = min(("정적Platt", "굴림Platt180"), key=sc25.get)
            full = static("platt", q[in25], y25a[in25], q) if pick == "정적Platt" else roll180
            info["R2c_2025_brier"], info["R2c_pick"] = sc25, pick
            log(f"[4h] R2c 2025 Brier {sc25} → {pick}")
            s26 = pd.Series(full[~in25], index=t25[~in25])
            ok = np.isin(ts, t25[~in25])
            assert ok.mean() > 0.995, ok.mean()                                             # 27칸 결측 행은 극소수 → 그 행 빼고 모두 같은 행
            te, ts, y = te[ok].reset_index(drop=True), ts[ok], y[ok]
            P = {k: v[ok] for k, v in P.items()}
            P[f"R2c 4h27칸+{pick}"] = s26.reindex(ts).to_numpy()
        k = len(P) - 1
        alpha = 0.05 / k
        d6 = pd.DatetimeIndex(ts).normalize().to_numpy()
        base = metrics(y, P["C0 배포"], cf["q20"], cf["q80"])
        r = {"n": int(len(y)), "days": int(len(np.unique(d6))), "ci_level": 1 - alpha, "info": info, "cands": {"C0 배포": base}}
        # 진단: 2026 사후 상수 절편(상한)
        d_or = fit_shift(P["C0 배포"], y)
        por = sig(logit(P["C0 배포"]) + d_or)
        r["oracle_shift"] = {"delta_logit": d_or, "dbrier": float(np.mean((por - y) ** 2) - base["brier"]),
                             "dbrier_ci95": day_boot((por - y) ** 2 - (P["C0 배포"] - y) ** 2, d6)}
        log(f"\n[{sc}] 2026 n {len(y):,} · 일 {r['days']} · CI {100 * (1 - alpha):.2f}% · 사후 절편 상한 Δ로짓 {d_or:+.3f} ΔBrier {r['oracle_shift']['dbrier']:+.5f}{np.round(r['oracle_shift']['dbrier_ci95'], 5).tolist()}")
        log(f"  C0 배포          Brier {base['brier']:.5f} LL {base['logloss']:.4f} ECE {base['ece_pp']:.2f}pp max {base['calmax_pp']:.1f} AUC {base['auc']:.4f} 치우침 {base['bias_pp']:+.2f}")
        for kk, p6 in P.items():
            if kk == "C0 배포":
                continue
            mt = metrics(y, p6, cf["q20"], cf["q80"])
            mt["dbrier"] = mt["brier"] - base["brier"]
            mt["dbrier_ci"] = day_boot((p6 - y) ** 2 - (P["C0 배포"] - y) ** 2, d6, alpha=alpha)
            mt["dauc"] = mt["auc"] - base["auc"]
            mt["dauc_ci"] = auc_boot(y, p6, P["C0 배포"], d6)
            mt["bias_ci_pp"] = [x * 100 for x in day_boot(p6 - y, d6)]
            mt["verdict"] = {"W1": mt["dbrier_ci"][1] < 0, "W2": mt["logloss"] <= base["logloss"], "W3": mt["ece_pp"] <= base["ece_pp"],
                             "W4": mt["auc"] >= base["auc"] - 0.002}
            mt["win"] = all(mt["verdict"].values())
            if sc == "4h":
                dts = pd.DatetimeIndex(ts); l90 = dts >= pd.Timestamp("2026-07-26")
                mt["bias_90_pp"] = float((p6[l90].mean() - y[l90].mean()) * 100)
                mt["월별"] = {mo: round(float((p6[dts.month == mo].mean() - y[dts.month == mo].mean()) * 100), 2) for mo in range(1, 11)}
            r["cands"][kk] = mt
            log(f"  {kk:22s} Brier {mt['brier']:.5f} Δ{mt['dbrier']:+.5f}{np.round(mt['dbrier_ci'], 5).tolist()} LL {mt['logloss']:.4f} ECE {mt['ece_pp']:.2f}pp "
                f"max {mt['calmax_pp']:.1f} AUC {mt['auc']:.4f} Δ{mt['dauc']:+.4f}{np.round(mt['dauc_ci'], 4).tolist()} 치우침 {mt['bias_pp']:+.2f}{np.round(mt['bias_ci_pp'], 2).tolist()} "
                f"깨질쪽 {mt['break_share'] * 100:.0f}%→{(mt['break_hold'] or 0) * 100:.1f}% 머물쪽 {mt['stay_share'] * 100:.0f}%→{(mt['stay_hold'] or 0) * 100:.1f}% "
                f"{mt['verdict']} → {'이김' if mt['win'] else '못 이김'}" + (f" · 최근90일 {mt['bias_90_pp']:+.2f} 월별 {mt['월별']}" if sc == "4h" else ""))
        rep[sc] = r
    (OUT / "round2.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'round2.json'}")


SEEDS = [248472, 153368, 413812, 863711, 207280]   # 3라운드 사전등록(무작위 추출)


def round3():
    """3라운드: 2라운드 승자(30분·1h R2a · 4h R2c) 시드 5개 · 분기 4조각 · 4h 칸/보정층 진단."""
    A = pd.read_parquet(OUT / "rows.parquet")
    rep = {}
    for sc, W in WIN.items():
        cf = BH["scales"][sc]
        g = A[A.scale == sc]
        if sc == "4h":
            te = g[g.ts >= "2026-01-01"].dropna(subset=LEAN).reset_index(drop=True)
            tr = g[g.ts < "2025-01-01"].dropna(subset=LEAN)
            g25 = g[g.ts >= "2025-01-01"].dropna(subset=LEAN).reset_index(drop=True)
        else:
            te = g[g.ts >= "2026-01-01"].dropna(subset=cf["cols"]).reset_index(drop=True)
            tr = g[g.ts + pd.Timedelta(minutes=W + 1) <= "2026-01-01"].dropna(subset=LEAN)
        ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
        p0 = cf["model"].predict_proba(te[cf["cols"]])[:, 1]
        d6 = pd.DatetimeIndex(ts).normalize().to_numpy()
        qtr = np.select([ts < np.datetime64("2026-04-01"), ts < np.datetime64("2026-07-01"), ts < np.datetime64("2026-10-01")], ["Q1", "Q2", "Q3"], "10월")

        def pred(seed):
            if sc != "4h":
                return HistGradientBoostingClassifier(**{**HGB, "random_state": seed}).fit(tr[cf["cols"]], tr.y).predict_proba(te[cf["cols"]])[:, 1], None
            m = HistGradientBoostingClassifier(**{**HGB, "random_state": seed}).fit(tr[LEAN], tr.y)
            q = m.predict_proba(g25[LEAN])[:, 1]
            t25, y25 = g25.ts.to_numpy(), g25.y.to_numpy().astype(float)
            in25 = t25 < np.datetime64("2026-01-01")
            rp = rolling("platt", t25, q, y25, W, 180)
            diag = {"27칸 그대로": q[~in25], "27칸+정적Platt(2025)": static("platt", q[in25], y25[in25], q[~in25]), "n_iter": int(m.n_iter_)}
            assert (t25[~in25] == ts).all()
            return rp[~in25], diag

        r = {"seeds": {}, "quarters": {}}
        base_b = float(np.mean((p0 - y) ** 2))
        for seed in [0] + SEEDS:
            p, diag = pred(seed)
            d = (p - y) ** 2 - (p0 - y) ** 2
            r["seeds"][seed] = {"dbrier": float(d.mean()), "ci95": day_boot(d, d6), "dauc": float(roc_auc_score(y, p) - roc_auc_score(y, p0)),
                                "bias_pp": float((p.mean() - y.mean()) * 100), "ece_pp": ece(y, p)[0] * 100}
            if seed == 0:
                for qn in ("Q1", "Q2", "Q3", "10월"):
                    mq = qtr == qn
                    r["quarters"][qn] = {"n": int(mq.sum()), "dbrier": float(d[mq].mean()), "ci95": day_boot(d[mq], d6[mq])}
                if diag:
                    r["diag_4h"] = {k: (v if k == "n_iter" else {"dbrier": float(np.mean((v - y) ** 2) - base_b), "ci95": day_boot((v - y) ** 2 - (p0 - y) ** 2, d6),
                                                              "dauc": float(roc_auc_score(y, v) - roc_auc_score(y, p0)), "bias_pp": float((v.mean() - y.mean()) * 100),
                                                              "ece_pp": ece(y, v)[0] * 100}) for k, v in diag.items()}
            log(f"[{sc}] 시드 {seed}: {r['seeds'][seed]}" + (f" n_iter {diag['n_iter']}" if diag else ""))
        r["S1"] = all(v["dbrier"] < 0 for k, v in r["seeds"].items() if k in SEEDS)
        r["S2"] = sum(v["dbrier"] < 0 for v in r["quarters"].values()) >= 3
        log(f"[{sc}] 분기 {r['quarters']}")
        if "diag_4h" in r:
            log(f"[{sc}] 진단 {r['diag_4h']}")
        log(f"[{sc}] S1 {r['S1']} · S2 {r['S2']} → {'배포 권고' if r['S1'] and r['S2'] else '권고 안 함'}")
        rep[sc] = r
    (OUT / "round3.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'round3.json'}")


V0, V1, VM, T0 = "2026-01-01", "2026-06-01", "2026-03-17", "2026-06-01"   # 4라운드 분할(검증 01-01~05-31 · 시험 06-01~)


def layers(ts, p, y, W):
    """보정층 → 2026 예측(시험용) · 검증 선택 점수용 예측(정적 = 검증 반기 교차 적합)."""
    va, h1 = (ts >= np.datetime64(V0)) & (ts < np.datetime64(V1)), ts < np.datetime64(VM)
    out, sel = {"없음": p}, {}
    for kind, nm in (("platt", "정적Platt"), ("beta", "정적베타"), ("iso", "정적등조")):
        out[nm] = np.where(va, np.nan, static(kind, p[va], y[va], p))
        cv = np.full(len(p), np.nan)
        cv[va & h1] = static(kind, p[va & ~h1], y[va & ~h1], p[va & h1])
        cv[va & ~h1] = static(kind, p[va & h1], y[va & h1], p[va & ~h1])
        sel[nm] = cv
    for N in (60, 90, 180):
        out[f"굴림Platt{N}"] = rolling("platt", ts, p, y, W, N, t0=V0)
        out[f"굴림절편{N}"] = rolling("shift", ts, p, y, W, N, t0=V0)
    out["OPS"] = ops(ts, p, y, W, t0=V0)
    for k, v in out.items():
        sel.setdefault(k, v)
    return out, sel


def round4():
    """4라운드(분할 변경): 학습 2024~25 · 검증 2026-01~05 · 시험 06-01~10-23."""
    A = pd.read_parquet(OUT / "rows.parquet")
    rep = {}
    for sc, W in WIN.items():
        cf = BH["scales"][sc]
        g = A[A.scale == sc]
        c27 = LEAN
        need = c27 + ["B_rate30"]
        tr = g[(g.ts >= "2024-01-01") & (g.ts + pd.Timedelta(minutes=W + 1) <= V0)].dropna(subset=need)
        assert (tr.ts + pd.Timedelta(minutes=W + 1)).max() <= pd.Timestamp(V0)
        te = g[g.ts >= V0].dropna(subset=need).reset_index(drop=True)                     # 모든 후보가 같은 행
        ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
        hgb = lambda cols: HistGradientBoostingClassifier(**HGB).fit(tr[cols], tr.y).predict_proba(te[cols])[:, 1]  # noqa: E731
        n1 = cf["cols"]
        M = {"C0": cf["model"].predict_proba(te[cf["cols"]])[:, 1], "N1": hgb(n1)}
        if sc == "4h":
            M["N2"] = hgb(c27)
        M["N3"] = hgb((c27 if sc == "4h" else n1) + ["B_rate30"])
        G = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[F_BASE], tr.y).predict_proba(te[F_BASE])[:, 1]
        P, SEL = {"G 기하로지스틱": G}, {"G 기하로지스틱": G}
        for mk, pm in M.items():
            out, sel = layers(ts, pm, y, W)
            for lk in out:
                P[f"{mk}+{lk}"], SEL[f"{mk}+{lk}"] = out[lk], sel[lk]
        va = (ts >= np.datetime64(V0)) & (ts < np.datetime64(V1))
        tt = ts >= np.datetime64(T0)
        assert (pd.DatetimeIndex(ts[tt]) + pd.Timedelta(minutes=W + 1)).max() <= END + pd.Timedelta(minutes=1)
        vb = {k: float(np.mean((v[va] - y[va]) ** 2)) for k, v in SEL.items()}
        S = min((k for k in vb if k != "C0+없음"), key=vb.get)
        top = sorted(vb.items(), key=lambda kv: kv[1])[:8]
        log(f"\n[{sc}] 학습 {len(tr):,}행 · 검증 {va.sum():,} · 시험 {tt.sum():,} · 검증 Brier C0 {vb['C0+없음']:.5f} · 상위 " + " · ".join(f"{k} {v:.5f}" for k, v in top) + f"\n  → S = {S}")
        y6, d6 = y[tt], pd.DatetimeIndex(ts[tt]).normalize().to_numpy()
        p0 = P["C0+없음"][tt]
        base = metrics(y6, p0, cf["q20"], cf["q80"])
        base["bias_ci_pp"] = [x * 100 for x in day_boot(p0 - y6, d6)]
        r = {"train_rows": int(len(tr)), "val_rows": int(va.sum()), "test_rows": int(tt.sum()), "test_days": int(len(np.unique(d6))),
             "test_hold": float(y6.mean()), "val_brier": vb, "S": S, "cands": {"C0+없음": base}}
        show = [S, "N1+없음", "G 기하로지스틱"] + (["N2+없음", "N2+굴림Platt180", "C0+굴림절편90"] if sc == "4h" else ["N1+굴림절편180"])
        for k in dict.fromkeys(show):
            if k == "C0+없음":
                continue
            p6 = P[k][tt]
            mt = metrics(y6, p6, cf["q20"], cf["q80"])
            mt["dbrier"] = mt["brier"] - base["brier"]
            mt["dbrier_ci"] = day_boot((p6 - y6) ** 2 - (p0 - y6) ** 2, d6)
            mt["dauc"], mt["dauc_ci"] = mt["auc"] - base["auc"], auc_boot(y6, p6, p0, d6)
            mt["bias_ci_pp"] = [x * 100 for x in day_boot(p6 - y6, d6)]
            r["cands"][k] = mt
        for k, mt in r["cands"].items():
            log(f"  {k:22s} Brier {mt['brier']:.5f} " + (f"Δ{mt['dbrier']:+.5f}{np.round(mt['dbrier_ci'], 5).tolist()} " if "dbrier" in mt else " " * 34) +
                f"LL {mt['logloss']:.4f} ECE {mt['ece_pp']:.2f}pp max {mt['calmax_pp']:.1f} AUC {mt['auc']:.4f}" +
                (f" Δ{mt['dauc']:+.4f}{np.round(mt['dauc_ci'], 4).tolist()}" if "dauc" in mt else "") +
                f" 치우침 {mt['bias_pp']:+.2f}{np.round(mt['bias_ci_pp'], 2).tolist()} 깨질쪽 {mt['break_share'] * 100:.0f}%→{(mt['break_hold'] or 0) * 100:.1f}% "
                f"머물쪽 {mt['stay_share'] * 100:.0f}%→{(mt['stay_hold'] or 0) * 100:.1f}%")
        s_ = r["cands"][S]
        r["verdict"] = {"W1": s_["dbrier_ci"][1] < 0, "W2": s_["logloss"] <= base["logloss"], "W3": s_["ece_pp"] <= base["ece_pp"], "W4": s_["auc"] >= base["auc"] - 0.002}
        r["win"] = all(r["verdict"].values())
        dts = pd.DatetimeIndex(ts[tt])
        r["monthly_bias_pp"] = {k: {mo: round(float((P[k][tt][dts.month == mo].mean() - y6[dts.month == mo].mean()) * 100), 2) for mo in range(6, 11)} for k in ("C0+없음", S)}
        r["reliability_test"] = {k: ece(y6, P[k][tt])[2].round(4).to_dict("list") for k in ("C0+없음", S)}
        log(f"  월별 치우침(pp) {r['monthly_bias_pp']}")
        log(f"  판정 S={S}: {r['verdict']} → {'이김' if r['win'] else '못 이김'}")
        rep[sc] = r
    (OUT / "round4.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"→ {OUT / 'round4.json'}")


def two_tier_r4():
    """10-26 사용자 2단 기준(되돌리기 쉬운 예측 개선 = 단측 90%) 사후 적용 — 4라운드 S 의 W1 만 다시 계산(판정 숫자 불변).
    C0(배포 모델)·굴림/정적 층은 학습이 없어 4라운드와 같은 값. HGB(N1·N3) 후보는 random_state=0 으로 재적합."""
    A = pd.read_parquet(OUT / "rows.parquet"); r4 = json.loads((OUT / "round4.json").read_text()); rep = {}
    for sc, W in WIN.items():
        cf, S = BH["scales"][sc], r4[sc]["S"]
        g = A[A.scale == sc]; need = LEAN + ["B_rate30"]
        tr = g[(g.ts >= "2024-01-01") & (g.ts + pd.Timedelta(minutes=W + 1) <= V0)].dropna(subset=need)
        te = g[g.ts >= V0].dropna(subset=need).reset_index(drop=True); ts, y = te.ts.to_numpy(), te.y.to_numpy().astype(float)
        mk, lk = S.split("+")
        cols = {"C0": None, "N1": cf["cols"], "N3": (LEAN if sc == "4h" else cf["cols"]) + ["B_rate30"]}[mk]
        pm = cf["model"].predict_proba(te[cf["cols"]])[:, 1] if mk == "C0" else \
            HistGradientBoostingClassifier(**HGB).fit(tr[cols], tr.y).predict_proba(te[cols])[:, 1]
        p0 = cf["model"].predict_proba(te[cf["cols"]])[:, 1]; pS = layers(ts, pm, y, W)[0][lk]
        tt = ts >= np.datetime64(T0); y6, d6 = y[tt], pd.DatetimeIndex(ts[tt]).normalize().to_numpy()
        d = (pS[tt] - y6) ** 2 - (p0[tt] - y6) ** 2
        ci95, ci80 = day_boot(d, d6), day_boot(d, d6, alpha=0.2)                # 80% 양측 상한 = 단측 90% 상한
        assert abs(float(d.mean()) - r4[sc]["cands"][S]["dbrier"]) < 1e-9, (sc, float(d.mean()), r4[sc]["cands"][S]["dbrier"])
        codes, uniq = pd.factorize(d6); s_, n_ = np.bincount(codes, d), np.bincount(codes).astype(float)
        w = np.random.default_rng(0).multinomial(len(uniq), np.full(len(uniq), 1 / len(uniq)), size=2000)
        v = r4[sc]["verdict"]
        rep[sc] = dict(S=S, dbrier=float(d.mean()), ci95=ci95, upper_one_sided90=ci80[1], p_not_better=float((((w @ s_) / (w @ n_)) >= 0).mean()),
                       W1_two_tier=ci80[1] < 0, W2=v["W2"], W3=v["W3"], W4=v["W4"])
        rep[sc]["conditional_adopt"] = bool(rep[sc]["W1_two_tier"] and v["W2"] and v["W3"] and v["W4"])
        log(f"[{sc}] {rep[sc]}")
    (OUT / "two_tier_r4.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


BOX_ROLL = {"W1": ("2022-01-01", "2024-01-01", "2024-06-01", "2024-10-24"), "W2": ("2023-01-01", "2025-01-01", "2025-06-01", "2025-10-24"),
            "W3": ("2024-01-01", "2026-01-01", "2026-06-01", "2026-10-24")}   # (학습 시작, 검증 시작, 시험 시작, 시험 끝(제외))


def round5():
    """5라운드(b22210bd): 고정 레시피 A(배포 레시피 재학습) vs B(굴림층·4h 27칸) 를 3창에서 같은 학습 조건으로."""
    A0 = pd.read_parquet(OUT / "rows.parquet"); rep = {}
    for sc, W in WIN.items():
        cf = BH["scales"][sc]; g = A0[A0.scale == sc]; ca, cb = cf["cols"], (LEAN if sc == "4h" else cf["cols"]); need = list(dict.fromkeys(ca + cb))
        acc, per = {"A": [], "B": [], "y": [], "d": []}, {}
        for w, (t0, v0, s0, e0) in BOX_ROLL.items():
            tr = g[(g.ts >= t0) & (g.ts + pd.Timedelta(minutes=W + 1) <= v0)].dropna(subset=need)
            ev = g[(g.ts >= v0) & (g.ts + pd.Timedelta(minutes=W + 1) <= e0)].dropna(subset=need).reset_index(drop=True)
            assert (tr.ts + pd.Timedelta(minutes=W + 1)).max() <= pd.Timestamp(v0) and (ev.ts + pd.Timedelta(minutes=W + 1)).max() <= pd.Timestamp(e0)
            ts, y = ev.ts.to_numpy(), ev.y.to_numpy().astype(float)
            fitp = lambda cols: HistGradientBoostingClassifier(**HGB).fit(tr[cols], tr.y).predict_proba(ev[cols])[:, 1]  # noqa: E731
            pA = fitp(ca); pB0 = fitp(cb) if sc == "4h" else pA
            pB = rolling("platt" if sc == "4h" else "shift", ts, pB0, y, W, 180, t0=v0)
            tt = ts >= np.datetime64(s0); d = pd.DatetimeIndex(ts[tt]).normalize().to_numpy()
            mA, mB = metrics(y[tt], pA[tt], cf["q20"], cf["q80"]), metrics(y[tt], pB[tt], cf["q20"], cf["q80"])
            per[w] = {"train_rows": len(tr), "test_rows": int(tt.sum()), "dbrier": mB["brier"] - mA["brier"],
                      "dbrier_ci": day_boot((pB[tt] - y[tt]) ** 2 - (pA[tt] - y[tt]) ** 2, d), "A": mA, "B": mB}
            for k, v in (("A", pA[tt]), ("B", pB[tt]), ("y", y[tt]), ("d", d)):
                acc[k].append(v)
            log(f"[{sc} {w}] 학습 {len(tr):,} 시험 {tt.sum():,} · ΔBrier {per[w]['dbrier']:+.5f}{np.round(per[w]['dbrier_ci'], 5).tolist()} · "
                f"치우침 A {mA['bias_pp']:+.2f} B {mB['bias_pp']:+.2f}pp · ECE A {mA['ece_pp']:.2f} B {mB['ece_pp']:.2f} · AUC A {mA['auc']:.4f} B {mB['auc']:.4f}")
        pa, pb, yy, dd = (np.concatenate(acc[k]) for k in ("A", "B", "y", "d"))
        mA, mB = metrics(yy, pa, cf["q20"], cf["q80"]), metrics(yy, pb, cf["q20"], cf["q80"]); ci = day_boot((pb - yy) ** 2 - (pa - yy) ** 2, dd)
        v = {"W1": ci[1] < 0, "W2": mB["logloss"] <= mA["logloss"], "W3": mB["ece_pp"] <= mA["ece_pp"], "W4": mB["auc"] >= mA["auc"] - 0.002,
             "W5": sum(x["dbrier"] < 0 for x in per.values()) >= 2}
        rep[sc] = {"pooled": {"dbrier": mB["brier"] - mA["brier"], "dbrier_ci": ci, "A": mA, "B": mB}, "windows": per, "verdict": v, "win": all(v.values())}
        log(f"[{sc} 3창] ΔBrier {mB['brier'] - mA['brier']:+.5f}{np.round(ci, 5).tolist()} · LL A {mA['logloss']:.4f} B {mB['logloss']:.4f} · ECE A {mA['ece_pp']:.2f} B {mB['ece_pp']:.2f} · "
            f"AUC A {mA['auc']:.4f} B {mB['auc']:.4f} · 치우침 A {mA['bias_pp']:+.2f} B {mB['bias_pp']:+.2f} → {v} {'이김' if all(v.values()) else '못 이김'}")
    (OUT / "round5.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def round8():
    """8라운드: 1h A(배포 레시피 재학습) vs A+굴림Platt90 — W1·W2 판정(단측 90%), W3 보고."""
    A0 = pd.read_parquet(OUT / "rows.parquet"); sc, W = "1h", WIN["1h"]; cf = BH["scales"][sc]; g = A0[A0.scale == sc]; ca = cf["cols"]
    acc, per = {k: [] for k in ("A", "B", "y", "d")}, {}
    for w, (t0, v0, s0, e0) in BOX_ROLL.items():
        tr = g[(g.ts >= t0) & (g.ts + pd.Timedelta(minutes=W + 1) <= v0)].dropna(subset=ca)
        ev = g[(g.ts >= v0) & (g.ts + pd.Timedelta(minutes=W + 1) <= e0)].dropna(subset=ca).reset_index(drop=True)
        ts, y = ev.ts.to_numpy(), ev.y.to_numpy().astype(float)
        pA = HistGradientBoostingClassifier(**HGB).fit(tr[ca], tr.y).predict_proba(ev[ca])[:, 1]
        pB = rolling("platt", ts, pA, y, W, 90, t0=v0)
        tt = ts >= np.datetime64(s0); d = pd.DatetimeIndex(ts[tt]).normalize().to_numpy()
        mA, mB = metrics(y[tt], pA[tt], cf["q20"], cf["q80"]), metrics(y[tt], pB[tt], cf["q20"], cf["q80"])
        per[w] = {"dbrier": mB["brier"] - mA["brier"], "dbrier_ci95": day_boot((pB[tt] - y[tt]) ** 2 - (pA[tt] - y[tt]) ** 2, d), "A": mA, "B": mB}
        if w != "W3":
            for k, v in (("A", pA[tt]), ("B", pB[tt]), ("y", y[tt]), ("d", d)):
                acc[k].append(v)
        log(f"[1h {w}{' 보고' if w == 'W3' else ''}] ΔBrier {per[w]['dbrier']:+.5f}{np.round(per[w]['dbrier_ci95'], 5).tolist()} · ECE A {mA['ece_pp']:.2f} B {mB['ece_pp']:.2f} · "
            f"LL A {mA['logloss']:.4f} B {mB['logloss']:.4f} · AUC A {mA['auc']:.4f} B {mB['auc']:.4f} · 치우침 A {mA['bias_pp']:+.2f} B {mB['bias_pp']:+.2f}")
    pa, pb, yy, dd = (np.concatenate(acc[k]) for k in ("A", "B", "y", "d"))
    mA, mB = metrics(yy, pa, cf["q20"], cf["q80"]), metrics(yy, pb, cf["q20"], cf["q80"]); dv = (pb - yy) ** 2 - (pa - yy) ** 2
    ci80 = day_boot(dv, dd, alpha=0.2)
    v = {"V1": ci80[1] < 0, "V2": mB["logloss"] <= mA["logloss"], "V3": mB["ece_pp"] <= mA["ece_pp"], "V4": mB["auc"] >= mA["auc"] - 0.002,
         "V5": per["W1"]["dbrier"] < 0 and per["W2"]["dbrier"] < 0}
    rep = {"pooled_W1W2": {"dbrier": mB["brier"] - mA["brier"], "upper_one_sided90": ci80[1], "ci95": day_boot(dv, dd), "A": mA, "B": mB},
           "windows": per, "verdict": v, "win": all(v.values())}
    log(f"[1h W1+W2] ΔBrier {rep['pooled_W1W2']['dbrier']:+.5f} 단측90% 상한 {ci80[1]:+.5f} · ECE A {mA['ece_pp']:.2f} B {mB['ece_pp']:.2f} · → {v} {'유지' if rep['win'] else '철회'}")
    (OUT / "round8.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def round6(layers_=None, name="6"):
    """6라운드: 4h 27칸 HGB + 보정층 B(Platt180)·B2(절편180)·B3(절편90) — 3창 검증 선택 · 5라운드 관문. 7라운드 = 같은 틀에 VA180·VA90."""
    A0 = pd.read_parquet(OUT / "rows.parquet"); sc, W = "4h", WIN["4h"]; cf = BH["scales"][sc]; g = A0[A0.scale == sc]
    ca, cb = cf["cols"], LEAN; need = list(dict.fromkeys(ca + cb)); layers_ = layers_ or {"B": ("platt", 180), "B2": ("shift", 180), "B3": ("shift", 90)}
    acc = {k: [] for k in ("A", "y", "d", *layers_)}; vb = {k: 0.0 for k in layers_}; per = {}; perw = {}
    for w, (t0, v0, s0, e0) in BOX_ROLL.items():
        tr = g[(g.ts >= t0) & (g.ts + pd.Timedelta(minutes=W + 1) <= v0)].dropna(subset=need)
        ev = g[(g.ts >= v0) & (g.ts + pd.Timedelta(minutes=W + 1) <= e0)].dropna(subset=need).reset_index(drop=True)
        ts, y = ev.ts.to_numpy(), ev.y.to_numpy().astype(float)
        fitp = lambda cols: HistGradientBoostingClassifier(**HGB).fit(tr[cols], tr.y).predict_proba(ev[cols])[:, 1]  # noqa: E731
        pA, p27 = fitp(ca), fitp(cb); va, tt = ts < np.datetime64(s0), ts >= np.datetime64(s0); d = pd.DatetimeIndex(ts[tt]).normalize().to_numpy()
        acc["A"].append(pA[tt]); acc["y"].append(y[tt]); acc["d"].append(d); per[w] = {}
        perw[w] = {"A": {"ece_pp": ece(y[tt], pA[tt])[0] * 100, "bias_pp": float((pA[tt].mean() - y[tt].mean()) * 100)}}
        for k, (kind, N) in layers_.items():
            pk = rolling(kind, ts, p27, y, W, N, t0=v0); vb[k] += float(np.mean((pk[va] - y[va]) ** 2)); acc[k].append(pk[tt])
            per[w][k] = float(np.mean((pk[tt] - y[tt]) ** 2) - np.mean((pA[tt] - y[tt]) ** 2))
            perw[w][k] = {"dbrier_ci": day_boot((pk[tt] - y[tt]) ** 2 - (pA[tt] - y[tt]) ** 2, d), "ece_pp": ece(y[tt], pk[tt])[0] * 100,
                          "bias_pp": float((pk[tt].mean() - y[tt].mean()) * 100), "auc": float(roc_auc_score(y[tt], pk[tt]))}
        log(f"  [{w}] " + " · ".join(f"{k} ECE {x['ece_pp']:.2f} 치우침 {x['bias_pp']:+.2f}" for k, x in perw[w].items()) + f" · ΔBrier {per[w]}")
    S = min(vb, key=vb.get); pa, yy, dd, pb = (np.concatenate(acc[k]) for k in ("A", "y", "d", S))
    mA, mB = metrics(yy, pa, cf["q20"], cf["q80"]), metrics(yy, pb, cf["q20"], cf["q80"]); ci = day_boot((pb - yy) ** 2 - (pa - yy) ** 2, dd)
    v = {"W1": ci[1] < 0, "W2": mB["logloss"] <= mA["logloss"], "W3": mB["ece_pp"] <= mA["ece_pp"], "W4": mB["auc"] >= mA["auc"] - 0.002,
         "W5": sum(per[w][S] < 0 for w in per) >= 2}
    rep = {"val_brier_sum": vb, "S": S, "windows_dbrier": per, "windows": perw, "pooled": {"dbrier": mB["brier"] - mA["brier"], "dbrier_ci": ci, "A": mA, "B": mB}, "verdict": v, "win": all(v.values())}
    for k in layers_:   # 서술: 다른 후보의 3창 합친 수치
        pk = np.concatenate(acc[k]); mk = metrics(yy, pk, cf["q20"], cf["q80"]); rep[f"desc_{k}"] = {"ece_pp": mk["ece_pp"], "bias_pp": mk["bias_pp"], "brier": mk["brier"], "auc": mk["auc"]}
    log(f"[4h {name}R] 검증 Brier 합 {vb} → S = {S} · 3창 ΔBrier {mB['brier'] - mA['brier']:+.5f}{np.round(ci, 5).tolist()} · LL A {mA['logloss']:.4f} S {mB['logloss']:.4f} · "
        f"ECE A {mA['ece_pp']:.2f} S {mB['ece_pp']:.2f} · AUC A {mA['auc']:.4f} S {mB['auc']:.4f} · 치우침 A {mA['bias_pp']:+.2f} S {mB['bias_pp']:+.2f} · 창별 {per} → {v} {'이김' if all(v.values()) else '못 이김'}")
    log("  서술 " + " · ".join(f"{k}: ECE {rep[f'desc_{k}']['ece_pp']:.2f} 치우침 {rep[f'desc_{k}']['bias_pp']:+.2f}" for k in layers_))
    (OUT / f"round{name}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))


def selftest():
    rs = np.random.default_rng(0)
    c = 100 + rs.normal(size=500).cumsum(); h = c + rs.random(500); l = c - rs.random(500)
    for W in (3, 7):
        y = labels(h, l, c, W)
        for i in range(W - 1, 500 - W):
            half = 0.5 * (h[i - W + 1:i + 1].max() - l[i - W + 1:i + 1].min())
            assert y[i] == int(h[i + 1:i + W + 1].max() < c[i] + half and l[i + 1:i + W + 1].min() > c[i] - half)
        assert (y[-W:] == -1).all() and (y[:W - 1] == -1).all()
        br = base_rate(y, W, days=1)                # 1일 = 1440 > 500 → 절반 못 채워 전부 NaN
        assert np.isnan(br).all()
    yy = (rs.random(3000) < .3).astype(np.int8)
    br = base_rate(yy, 5, days=1)
    i = 2500
    assert abs(br[i] - yy[i - 1440:i - 4].mean()) < 1e-12        # j ∈ [i−1440, i−5]
    p = rs.random(20000); y2 = (rs.random(20000) < p).astype(float)
    assert abs(fit_shift(np.clip(p - .1, .01, .99), y2)) > .2 and abs(fit_shift(p, y2)) < .05
    a, b = fit_platt(p, y2)
    assert abs(sig(a * logit(.3) + b) - .3) < .03          # 이미 보정된 p → 항등에 가깝다
    ts = np.datetime64("2025-01-01") + np.arange(20000) * 30 * MIN
    q = ops(ts, np.clip(p - .15, .01, .99), y2, 240)
    assert abs(q[-5000:].mean() - y2[-5000:].mean()) < .02, "OPS 가 치우침을 못 따라감"
    r = rolling("shift", ts, np.clip(p - .15, .01, .99), y2, 240, 60)
    assert abs(r[-5000:].mean() - y2[-5000:].mean()) < .02
    pf, yf = np.r_[.1, .2, .3, .4, .5, .6, .7, .8], np.r_[0, 0, 1, 0, 1, 0, 1, 1.]
    p0, p1 = venn_abers(pf, yf, np.r_[.05, .3, .35, .65, .95])
    assert (p0 <= p1).all() and (p0 >= 0).all() and (p1 <= 1).all() and p0[-1] < 1 and p1[0] > 0, (p0, p1)
    tsv, pv, yv = ts[:6000], np.clip(p[:6000] - .15, .01, .99), y2[:6000].copy()    # 인과: 03-01 맞춤 → 04-01 전 출력은 그 뒤 해결 라벨과 무관
    T, T2 = np.datetime64("2025-03-01"), np.datetime64("2025-04-01")
    r1 = rolling("va", tsv, pv, yv, 240, 30); yv[tsv + 241 * MIN > T] = 1 - yv[tsv + 241 * MIN > T]; r2 = rolling("va", tsv, pv, yv, 240, 30)
    m = (tsv >= T) & (tsv < T2); assert m.any() and np.array_equal(r1[m], r2[m]) and not np.array_equal(r1[tsv >= T2], r2[tsv >= T2])
    assert abs(r1[tsv >= T].mean() - y2[:6000][tsv >= T].mean()) < .03, "VA 가 치우침을 못 고침"
    lo, hi = day_boot(y2, np.arange(20000) // 50, B=500)
    assert lo < y2.mean() < hi
    log("selftest OK")


if __name__ == "__main__":
    selftest()
    if "--build" in sys.argv:
        build()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "1":
        round1()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "2":
        round2()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "3":
        round3()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "4":
        round4()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "5":
        round5()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "6":
        round6()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "7":
        round6({"VA180": ("va", 180), "VA90": ("va", 90)}, "7")
    if "--two-tier" in sys.argv:
        two_tier_r4()
    if "--round" in sys.argv and sys.argv[sys.argv.index("--round") + 1] == "8":
        round8()
