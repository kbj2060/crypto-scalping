#!/usr/bin/env python3
"""**박스 유지 확률 정확도 높이기** — 배포 모델(F 로지스틱 6칸)에 변동성 단서 묶음을 붙여 본다(2026-10-16).

왜: 사용자 «내가 중요하게 생각하는 박스 유지 모델을 좀 더 정확도 높일 수 있는지 검사».
기준: dashboard/box_hold.py = research_eth_range_hold_multiscale_20261016.py 의 F(기하 d_σ·σ1h/σ24·창폭·시각·주말), 2022~24 학습 로지스틱,
  2025·2026 AUC .76~.78. 라벨·F 칸은 그 연구 산출 decisions.parquet 그대로(같은 행, 같은 정답).
붙이는 묶음(전부 결정 분 i 의 1분봉 마감까지 = 라이브에서 계산 가능, 정답은 i+1 분부터):
  V 변동성 기간구조  log σ(5·15·60·240분·7일)/σ24 · 파킨슨 σ 1h/24h · 1h 최대 1분 점프/σ24 · log σ24 수준
  R 박스 모양      창 안 위치(가장자리 거리) · |창 이동|/창폭(추세성) · 창폭/직전 창폭(압축) · 창폭/지난 24h 창폭 중앙값
  Q 활동          창 거래량/7일 평균 · 5분 거래량/창 평균 · |테이커 불균형| 창·1h
  S 계절(미래 시각은 이미 앎)  다음 W분이 걸친 «요일×시각» 평균 변동(2022~24 학습 구간으로만 계산)/전체 평균
  M 예정 발표(미리 앎)  다음 W분 안 주요 발표(CPI·고용·PPI·PCE·GDP·소매판매 08:30 ET · FOMC 14:00 ET) · 실업수당(목 08:30) · 직전 60분 발표
  X 교차·파생(5분 패널, 결정 시각까지 닫힌 봉, metrics 한 행 더 늦춤)  BTC σ1h/σ24 · |OI 1h·24h 변화| · |프리미엄|·|Δ프리미엄 1h| · 상위 포지션 비 1h 변화
모델: F 로지스틱(기준) · F+묶음 하나씩(로지스틱) · F+전부(로지스틱) · HGB(F) · HGB(전부). 행은 모든 칸이 있는 공통 행(공정 비교).
채점: 2025·2026 따로 AUC · ΔAUC(일 블록 부트스트랩) · 로그손실 · BSS · 10분위 보정 최대오차 · 학습 20/80분위 문턱 칸의 실제 유지율.
판정(사전에 정함): «개선» = 2025·2026 둘 다 ΔAUC CI 하한 > 0 이고 보정 최대오차 ≤ 기준. 크기는 ΔAUC 로 보고.
  python scripts/research_eth_box_hold_improve_20261016.py            # 특징 만들고 평가
  python scripts/research_eth_box_hold_improve_20261016.py --selftest
"""
from __future__ import annotations

import glob
import json
import os
import sys
import zipfile
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
MAIN = Path("/home/kbj20/crypto-scalping")
BV = MAIN / "data/binance_vision"
# ponytail: 라벨·F 칸은 배포 연구의 산출(다른 워크트리 tmp). 그 스크립트를 다시 돌리면 같은 파일이 나온다.
DEC = MAIN / ".claude/worktrees/pingpong-long-short-hedge-de2ac6/tmp/range_hold_20261016/decisions.parquet"
EXTRA = MAIN / ".claude/worktrees/pingpong-long-short-hedge-de2ac6/tmp/pingpong_hedge_20261016/vision_1m"
OUT = ROOT / "tmp/box_improve_20261016"
SCALES = {"30m": 30, "1h": 60, "4h": 240}
F_BASE = ["log_ds", "log_vr", "log_rng", "hs", "hc", "wkd"]
ET = ZoneInfo("America/New_York")
log = lambda m: print(m, flush=True)  # noqa: E731

# 2022~24 FOMC 결정일(연준 공식 일정) — 2025~26 은 trading_bot_modules/omega6_live.L6_FOMC_DECISION_DATES
FOMC_22_24 = ["2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
              "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
              "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18"]
MAJOR = {"cpi": 10, "nfp": 50, "gdp": 53, "pce": 54, "ppi": 46, "retail_sales": 9}
CLAIMS = 180


def load_1m() -> pd.DataFrame:
    parts = [pd.read_parquet(f) for f in sorted(glob.glob(f"{BV}/klines1m/ETHUSDT-1m-*.parquet")) + sorted(glob.glob(f"{BV}/klines1m_daily/ETHUSDT-1m-*.parquet"))]
    parts = [p[["t", "h", "l", "c", "v", "tb"]] for p in parts]
    for z in sorted(glob.glob(str(EXTRA / "*.zip"))):
        with zipfile.ZipFile(z) as zf:
            x = pd.read_csv(zf.open(zf.namelist()[0]))
        parts.append(pd.DataFrame({"t": x.open_time, "h": x.high, "l": x.low, "c": x.close, "v": x.volume, "tb": x.taker_buy_volume}))
    K = pd.concat(parts).drop_duplicates("t").sort_values("t").reset_index(drop=True)
    assert (np.diff(K.t) == 60000).all(), "1분봉 빈칸"
    return K


def roll(x, w, fn="mean", mp=None):
    return getattr(pd.Series(x).rolling(w, min_periods=mp or w), fn)().to_numpy()


def events() -> dict[str, np.ndarray]:
    """예정 발표 시각(UTC ms). FRED release/dates 는 날짜만 → 관례 시각(08:30 ET, FOMC 14:00 ET). 캐시 OUT/events.json."""
    cache = OUT / "events.json"
    if cache.exists():
        return {k: np.array(v, dtype=np.int64) for k, v in json.loads(cache.read_text()).items()}
    import requests
    key = os.getenv("FRED_API_KEY")
    if not key:
        for envf in (ROOT / ".env", MAIN / ".env"):
            if envf.exists():
                for line in envf.read_text().splitlines():
                    if line.startswith("FRED_API_KEY="):
                        key = line.split("=", 1)[1].strip().strip('"')
    assert key, "FRED_API_KEY 없음"

    def dates(rid):
        r = requests.get("https://api.stlouisfed.org/fred/release/dates", timeout=30, params=dict(
            release_id=rid, api_key=key, file_type="json", realtime_start="2021-12-01", realtime_end="2026-12-31",
            include_release_dates_with_no_data="true", limit=10000, sort_order="asc"))
        r.raise_for_status()
        return [d["date"] for d in r.json()["release_dates"] if "2021-12-01" <= d["date"] <= "2026-12-31"]

    def utc_ms(day, hh, mm):
        return int(pd.Timestamp(day).replace(hour=hh, minute=mm).tz_localize(ET).tz_convert("UTC").value // 10**6)

    sys.path.insert(0, str(MAIN))
    from trading_bot_modules.omega6_live import L6_FOMC_DECISION_DATES
    fomc = FOMC_22_24 + [d for y in (2025, 2026) for d in L6_FOMC_DECISION_DATES.get(y, ())]
    major = sorted({utc_ms(d, 8, 30) for rid in MAJOR.values() for d in dates(rid)} | {utc_ms(d, 14, 0) for d in fomc})
    claims = sorted({utc_ms(d, 8, 30) for d in dates(CLAIMS)})
    out = {"major": major, "claims": claims}
    cache.write_text(json.dumps(out))
    log(f"발표 시각: 주요 {len(major)} · 실업수당 {len(claims)}")
    return {k: np.array(v, dtype=np.int64) for k, v in out.items()}


def next_ev(t_close: np.ndarray, ev: np.ndarray) -> np.ndarray:
    """결정 마감 시각 이후 첫 발표까지 분(없으면 큰 값)."""
    j = np.searchsorted(ev, t_close, side="right")
    nxt = np.where(j < len(ev), ev[np.minimum(j, len(ev) - 1)], np.iinfo(np.int64).max // 2)
    return (nxt - t_close) / 60000.0


def prev_ev(t_close: np.ndarray, ev: np.ndarray) -> np.ndarray:
    j = np.searchsorted(ev, t_close, side="right") - 1
    prv = np.where(j >= 0, ev[np.maximum(j, 0)], -(np.iinfo(np.int64).max // 2))
    return (t_close - prv) / 60000.0


def features(D: pd.DataFrame, K: pd.DataFrame) -> pd.DataFrame:
    t = K.t.to_numpy(); h, l, c, v, tb = (K[k].to_numpy(float) for k in ("h", "l", "c", "v", "tb"))
    r = np.r_[0, np.diff(np.log(c))]
    r2 = r ** 2
    s24 = np.sqrt(roll(r2, 1440))
    pk = np.log(h / l) ** 2 / (4 * np.log(2))
    pos = pd.Series(np.arange(len(t)), index=t)
    i = pos.reindex(D.ts.astype("int64").to_numpy() // 10**6).to_numpy()
    assert np.isfinite(i).all(), "결정 분이 1분봉에 없음"
    i = i.astype(int)
    tclose = t[i] + 60000
    X = pd.DataFrame(index=D.index)
    # V
    for w, nm in ((5, "5m"), (15, "15m"), (60, "1h"), (240, "4h"), (10080, "7d")):
        X[f"V_lsig{nm}"] = np.log(np.sqrt(roll(r2, w, mp=w // 2))[i] + 1e-9) - np.log(s24[i])
    X["V_lsig24"] = np.log(s24[i])
    X["V_park1h"] = np.log(np.sqrt(roll(pk, 60))[i] + 1e-9) - np.log(s24[i])
    X["V_park24"] = np.log(np.sqrt(roll(pk, 1440))[i] + 1e-9) - np.log(s24[i])
    X["V_jump1h"] = roll(np.abs(r), 60, "max")[i] / s24[i]
    # R / Q (창 W 는 행마다 다르다)
    for nm, W in SCALES.items():
        m = (D.scale == nm).to_numpy()
        ii = i[m]
        whi, wlo = roll(h, W, "max"), roll(l, W, "min")
        rng = whi - wlo
        prng = np.r_[np.full(W, np.nan), rng[:-W]]
        p = (c - wlo) / rng
        X.loc[m, "R_edge"] = np.minimum(p, 1 - p)[ii]
        X.loc[m, "R_trend"] = (np.abs(c - np.r_[np.full(W, np.nan), c[:-W]]) / rng)[ii]
        X.loc[m, "R_compress"] = np.log(rng / prng)[ii]
        X.loc[m, "R_squeeze"] = np.log(rng / roll(rng, 1440, "median", mp=720))[ii]
        vW = roll(v, W)
        X.loc[m, "Q_volz"] = np.log(vW / (roll(v, 10080, mp=5040) * 1.0))[ii]
        X.loc[m, "Q_vol5_vs_W"] = np.log(roll(v, 5) / vW)[ii]
        X.loc[m, "Q_imbW"] = np.abs(roll(2 * tb - v, W, "sum") / roll(v, W, "sum"))[ii]
    X["Q_imb1h"] = np.abs(roll(2 * tb - v, 60, "sum") / roll(v, 60, "sum"))[i]
    # S: 다음 W분 «요일×시각» 평균 변동 / 전체 — 학습 구간(2022~24)만으로 계산
    ts1 = pd.to_datetime(t, unit="ms")
    how = (ts1.dayofweek * 24 + ts1.hour).to_numpy()
    tr = t < pd.Timestamp("2025-01-01").value // 10**6
    prof = pd.Series(r2[tr]).groupby(how[tr]).mean().reindex(range(168)).to_numpy()
    prof = prof / np.nanmean(r2[tr])
    pm = prof[how]
    for nm, W in SCALES.items():
        m = (D.scale == nm).to_numpy()
        fut = np.r_[pd.Series(pm[::-1]).rolling(W).mean().to_numpy()[::-1][1:], np.nan]   # i+1..i+W 평균
        X.loc[m, "S_season_fut"] = np.log(fut[i[m]])
    X["S_season_now"] = np.log(pd.Series(pm).rolling(60).mean().to_numpy()[i])
    # M
    ev = events()
    W_row = D.scale.map(SCALES).to_numpy()
    nm_ = next_ev(tclose, ev["major"])
    X["M_major_in"] = (nm_ <= W_row).astype(float)
    X["M_major_pos"] = np.where(nm_ <= W_row, nm_ / W_row, 1.0)
    X["M_major_recent"] = (prev_ev(tclose, ev["major"]) <= 60).astype(float)
    X["M_claims_in"] = (next_ev(tclose, ev["claims"]) <= W_row).astype(float)
    # X: 5분 패널 — 결정 마감까지 닫힌 5분봉(시작 ≤ 마감 − 5분), metrics 는 한 행 더 늦춤
    def at5(ts5_ms, arr, extra_lag=0):
        k = np.searchsorted(ts5_ms, tclose - 300000, side="right") - 1 - extra_lag
        ok = (k >= 0) & (tclose - 300000 - ts5_ms[np.maximum(k, 0)] <= 15 * 60000)
        return np.where(ok, arr[np.maximum(k, 0)], np.nan)
    P = pd.read_parquet(BV / "panel/ETHUSDT.parquet").sort_values("timestamp")
    B = pd.read_parquet(BV / "panel/BTCUSDT.parquet").sort_values("timestamp")
    pt = P.timestamp.astype("int64").to_numpy() // 10**6
    bt = B.timestamp.astype("int64").to_numpy() // 10**6
    br = np.r_[0, np.diff(np.log(B.close.ffill().to_numpy()))]
    X["X_btc_vr"] = at5(bt, np.log(np.sqrt(roll(br ** 2, 12)) / np.sqrt(roll(br ** 2, 288))))
    oi = np.log(P.sum_open_interest.ffill().to_numpy())
    X["X_abs_doi1h"] = at5(pt, np.abs(oi - np.r_[np.full(12, np.nan), oi[:-12]]) * 1e4, 1)
    X["X_abs_doi24h"] = at5(pt, np.abs(oi - np.r_[np.full(288, np.nan), oi[:-288]]) * 1e4, 1)
    tp = np.log(P.sum_toptrader_long_short_ratio.ffill().to_numpy())
    X["X_abs_dtoppos1h"] = at5(pt, np.abs(tp - np.r_[np.full(12, np.nan), tp[:-12]]), 1)
    pr = pd.read_parquet(BV / "eth_premium_5m.parquet").sort_values("open_time")
    prt = pr.open_time.astype("int64").to_numpy() // 10**6 if np.issubdtype(pr.open_time.dtype, np.datetime64) else pr.open_time.to_numpy(np.int64)
    pc = pr.close.to_numpy(float) * 1e4
    X["X_abs_prem"] = at5(prt, np.abs(pc))
    X["X_abs_dprem1h"] = at5(prt, np.abs(pc - np.r_[np.full(12, np.nan), pc[:-12]]))
    return X


GROUPS = ("V", "R", "Q", "S", "M", "X")


def block_ci(y, a, b, day, B=200, seed=0):
    """ΔAUC(a − b) 일 블록 부트스트랩 95% CI."""
    codes, uniq = pd.factorize(day)
    order = np.argsort(codes, kind="stable"); st = np.searchsorted(codes[order], np.arange(len(uniq))); en = np.r_[st[1:], len(codes)]
    rng = np.random.default_rng(seed); out = []
    for _ in range(B):
        ii = np.concatenate([order[st[k]:en[k]] for k in rng.integers(0, len(uniq), len(uniq))])
        out.append(roc_auc_score(y[ii], a[ii]) - roc_auc_score(y[ii], b[ii]))
    return [round(float(np.quantile(out, .025)), 4), round(float(np.quantile(out, .975)), 4)]


def calib(y, p):
    q = pd.qcut(p, 10, labels=False, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": y, "q": q}).groupby("q").mean()
    return float((g.p - g.y).abs().max())


def evaluate(D: pd.DataFrame, X: pd.DataFrame) -> dict:
    cols_all = F_BASE + [c for c in X.columns]
    A = pd.concat([D[["scale", "ts", "hold"] + F_BASE], X], axis=1).replace([np.inf, -np.inf], np.nan)   # 거래량 0·직전 창폭 0 분의 log(0)
    rep = {}
    for nm in SCALES:
        g = A[A.scale == nm].dropna()
        tr, te = g[g.ts < "2025-01-01"], g[g.ts >= "2025-01-01"]
        log(f"\n[{nm}] 공통 행 학습 {len(tr):,} · 채점 {len(te):,}({te.ts.min():%Y-%m-%d}~{te.ts.max():%Y-%m-%d}) · 학습 유지율 {tr.hold.mean():.3f}")
        specs = {"F(기준)": ("lr", F_BASE)}
        for G in GROUPS:
            specs[f"F+{G}"] = ("lr", F_BASE + [c for c in X.columns if c.startswith(G + "_")])
        specs["F+전부"] = ("lr", cols_all)
        specs["HGB F"] = ("hgb", F_BASE)
        specs["HGB 전부"] = ("hgb", cols_all)
        P, thr = {}, {}
        for k, (kind, cols) in specs.items():
            if kind == "lr":
                mdl = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
            else:
                mdl = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,
                                                     l2_regularization=1.0, random_state=0)
            mdl.fit(tr[cols], tr.hold)
            ptr = mdl.predict_proba(tr[cols])[:, 1]
            thr[k] = (np.quantile(ptr, .2), np.quantile(ptr, .8))
            P[k] = mdl.predict_proba(te[cols])[:, 1]
        rep[nm] = {}
        for yr in (2025, 2026):
            mk = (te.ts.dt.year == yr).to_numpy()
            y = te.hold.to_numpy()[mk]; day = te.ts.dt.floor("D").to_numpy()[mk]
            base = np.full(mk.sum(), tr.hold.mean())
            log(f"  {yr} n {mk.sum():,} 유지율 {y.mean():.3f}")
            log(f"    {'모델':10s} AUC     ΔAUC vs F [95% CI]        로그손실  BSS     보정최대  깨질쪽(<q20)·머물쪽(>q80) 실제 유지율")
            rep[nm][yr] = {}
            for k in specs:
                p = P[k][mk]
                a = roc_auc_score(y, p)
                ci = [0, 0] if k == "F(기준)" else block_ci(y, p, P["F(기준)"][mk], day)
                ll = log_loss(y, p); bss = 1 - np.mean((p - y) ** 2) / np.mean((base - y) ** 2)
                ce = calib(y, p); lo, hi = thr[k]
                rlo, rhi = y[p < lo].mean(), y[p > hi].mean()
                log(f"    {k:10s} {a:.4f}  {a - roc_auc_score(y, P['F(기준)'][mk]):+.4f} {ci}  {ll:.4f}  {bss:+.3f}  {ce * 100:4.1f}pp   "
                    f"{rlo * 100:4.1f}% ({np.mean(p < lo) * 100:.0f}% 시간) · {rhi * 100:4.1f}% ({np.mean(p > hi) * 100:.0f}% 시간)")
                rep[nm][yr][k] = dict(auc=a, dauc_ci=ci, logloss=ll, bss=bss, cal=ce, lo_hold=rlo, hi_hold=rhi)
    return rep


def evaluate_lean(D: pd.DataFrame, X: pd.DataFrame) -> dict:
    """배포 후보: X(효과 ≈0) 빼고 HGB · 7일 σ 빼기(서빙 1분봉 1500개로 계산 가능) · 보정(2022~23 학습 → 2024 isotonic → 2025·26 채점)."""
    from sklearn.isotonic import IsotonicRegression
    A = pd.concat([D[["scale", "ts", "hold"] + F_BASE], X], axis=1).replace([np.inf, -np.inf], np.nan)
    noX = F_BASE + [c for c in X.columns if not c.startswith("X_")]
    no7d = [c for c in noX if c not in ("V_lsig7d", "Q_volz")]
    hgb = lambda: HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=500,  # noqa: E731
                                                 l2_regularization=1.0, random_state=0)
    rep = {}
    for nm in SCALES:
        g = A[A.scale == nm].dropna()
        tr, te = g[g.ts < "2025-01-01"], g[g.ts >= "2025-01-01"]
        P = {"F(기준)": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(tr[F_BASE], tr.hold).predict_proba(te[F_BASE])[:, 1]}
        for k, cols in (("HGB 전부", F_BASE + list(X.columns)), ("HGB −X", noX), ("HGB −X−7일", no7d)):
            P[k] = hgb().fit(tr[cols], tr.hold).predict_proba(te[cols])[:, 1]
        a, b = g[g.ts < "2024-01-01"], g[(g.ts >= "2024-01-01") & (g.ts < "2025-01-01")]
        m = hgb().fit(a[no7d], a.hold)
        iso = IsotonicRegression(out_of_bounds="clip").fit(m.predict_proba(b[no7d])[:, 1], b.hold)
        P["HGB −X−7일 +보정"] = iso.predict(m.predict_proba(te[no7d])[:, 1])
        rep[nm] = {}
        log(f"\n[{nm}] 배포 후보")
        for yr in (2025, 2026):
            mk = (te.ts.dt.year == yr).to_numpy(); y = te.hold.to_numpy()[mk]; day = te.ts.dt.floor("D").to_numpy()[mk]
            for k, pp in P.items():
                p = pp[mk]; auc = roc_auc_score(y, p)
                ci = [0, 0] if k == "F(기준)" else block_ci(y, p, P["F(기준)"][mk], day)
                log(f"  {yr} {k:16s} AUC {auc:.4f} Δ {auc - roc_auc_score(y, P['F(기준)'][mk]):+.4f} {ci} 보정최대 {calib(y, p) * 100:4.1f}pp "
                    f"로그손실 {log_loss(y, np.clip(p, 1e-4, 1 - 1e-4)):.4f}")
                rep[nm].setdefault(str(yr), {})[k] = dict(auc=auc, dauc_ci=ci, cal=calib(y, p))
    return rep


def selftest():
    t0 = pd.Timestamp("2026-03-11").value // 10**6
    ev = np.array([t0 + 13 * 3600_000 + 30 * 60_000], dtype=np.int64)            # 2026-03-11 13:30 UTC (= 08:30 EDT? 3월 8일 이후 EDT → 12:30 UTC)
    tc = np.array([ev[0] - 10 * 60000, ev[0], ev[0] + 5 * 60000])
    assert next_ev(tc, ev)[0] == 10 and next_ev(tc, ev)[1] > 1e6, next_ev(tc, ev)   # 발표 바로 그 분 = «다음 발표»가 아니라 «직전»
    assert prev_ev(tc, ev)[1] == 0 and prev_ev(tc, ev)[2] == 5
    u = int(pd.Timestamp("2026-03-11").replace(hour=8, minute=30).tz_localize(ET).tz_convert("UTC").value // 10**6)
    assert pd.Timestamp(u, unit="ms").hour == 12                                     # EDT(3월 둘째 일요일 이후) 08:30 = 12:30 UTC
    u2 = int(pd.Timestamp("2026-01-14").replace(hour=8, minute=30).tz_localize(ET).tz_convert("UTC").value // 10**6)
    assert pd.Timestamp(u2, unit="ms").hour == 13                                    # EST = 13:30 UTC
    x = np.arange(10.0)
    assert roll(x, 3)[2] == 1.0 and np.isnan(roll(x, 3)[1])
    log("selftest OK")


def leak_check(D, X):
    """특징이 결정 분 이후 1분봉을 쓰지 않는지: 결정 분 다음 봉들을 망가뜨려도 그 행 특징(S·M 제외 = 미리 아는 일정)이 같아야 한다."""
    K = load_1m()
    sub = D[D.scale == "1h"].iloc[200000:200003]
    ii = (pd.Series(np.arange(len(K)), index=K.t).reindex(sub.ts.astype("int64") // 10**6).to_numpy()).astype(int)
    K2 = K.copy()
    j = ii[0] + 1
    K2.loc[j:, ["h", "l", "c"]] *= 1.5; K2.loc[j:, ["v", "tb"]] *= 7
    a, b = features(sub.iloc[:1], K), features(sub.iloc[:1], K2)
    cols = [c for c in a.columns if not c.startswith(("S_", "M_", "X_"))]
    assert np.allclose(a[cols].to_numpy(float), b[cols].to_numpy(float), equal_nan=True), (a[cols] - b[cols]).T
    log("누수 점검 OK — 결정 분 뒤 1분봉을 바꿔도 V·R·Q 특징 불변")


if __name__ == "__main__":
    selftest()
    if "--selftest" in sys.argv:
        sys.exit()
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DEC)
    D = D[D.scale.isin(list(SCALES))].reset_index(drop=True)
    if "--lean" in sys.argv:
        rep = evaluate_lean(D, pd.read_parquet(OUT / "features.parquet"))
        (OUT / "report_lean.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        sys.exit()
    if "--eval-only" in sys.argv:
        rep = evaluate(D, pd.read_parquet(OUT / "features.parquet"))
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
        sys.exit()
    K = load_1m()
    leak_check(D, None)
    X = features(D, K)
    log("특징 결측 비율: " + json.dumps(X.isna().mean().round(3).to_dict()))
    X.to_parquet(OUT / "features.parquet")
    rep = evaluate(D, X)
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
