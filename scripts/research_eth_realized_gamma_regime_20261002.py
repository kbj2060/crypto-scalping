#!/usr/bin/env python3
"""실현 감마 국면 검증 (2026-10-02). 딜러가 누구인지 가정하지 않고 «움직임이 눌리는 장 / 커지는 장»을 가격에서 직접 잰다.

지표(시각 t = 정시, t 까지 닫힌 5분봉만):
  RG1_W = 직전 W 의 5분 로그수익률 1차 자기상관(표준 ACF 추정치)   W = 4h(주) · 1h · 12h(보조)
  RG2_W = VR = Var(겹치지 않는 15분 수익률) / (3·Var(5분 수익률)), 직전 W
  가격점 P(τ) = τ 에 끝나는 5분봉 종가. 지표는 가격점 [t−W, t], 결과는 [t+g, t+g+H] (g = 5분 = 한 봉 공백, 가격점도 안 겹침).
가설·판정 기준은 아래 CRITERIA(결과 보기 전 고정). 판정은 보류 구간(2025-01 ~ 2026-09). 바이낸스 REST 호출 없음(로컬 CSV·parquet).

  python scripts/research_eth_realized_gamma_regime_20261002.py
  python scripts/research_eth_realized_gamma_regime_20261002.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_dealer_assumption_compare_20261002 as A    # noqa: E402  judge · END

OUT = ROOT / "tmp/realized_gamma_regime_20261002"
KL = next(p for p in (ROOT / "binance_data/klines/ETHUSDT", Path.home() / "crypto-scalping/binance_data/klines/ETHUSDT") if p.exists())
M5, H_MS, D_MS = 300_000, 3_600_000, 86_400_000
ms = lambda s: pd.Timestamp(s, tz="UTC").value // 10**6
D0, SPLIT, D1 = ms("2021-01-01"), ms("2025-01-01"), ms("2026-10-01")
GAP = 1                      # 결과 창 시작 = t + 1봉
NB, SEED = 2000, 20261002
WINS = {"1h": 12, "4h": 48, "12h": 144}

# 결과 보기 전 고정(2026-10-02).
CRITERIA = {
    "split": {"discovery": "2021-01-01 ~ 2024-12-31", "holdout": "2025-01-01 ~ 2026-09-30", "judged_on": "holdout"},
    "rule": A.CRITERIA["rule"] + " · 부트스트랩 = UTC 일 블록 2000회",
    "primary_indicator": "RG1_4h(주) · RG2_4h(병기 판정) · 1h/12h 는 표만",
    "pct": "RG 분위 = 발견 구간 분포 기준 백분위(0~1) — 보류 구간에도 같은 문턱",
    "H1": {"def": "4h 격자(겹치지 않는 결과 창). Spearman(RG(t), 다음 4h 같은 측정치) · 보조: 분위 Q5−Q1 평균 차", "theory_sign": +1, "E": 0.05},
    "H2": {"def": "1h 격자. z = 수익률 / σ_t(직전 24h 5분 RV 로 1h 표준편차). r_next_z ~ 1 + r_past_z + pct + r_past_z×pct, "
                  "상호작용 계수(= RG 최하→최상 β 차) · 보조: 분위별 β·Q5−Q1 · |r_past_z|>1 부분표본 · 시 고정효과 · RV 삼분위 층화 · "
                  "vol 상호작용 통제 · 대조군(셔플 RG, 1일 지연 RG)", "theory_sign": +1, "E": 0.05},
    "H3": {"def": "1h 격자. log RV(다음 4h) ~ log RV 1h·24h·7d + pct (+시 고정효과 변형), pct 계수", "theory_sign": +1, "E": 0.05},
    "H4": {"def": "2026 만, 보조. Spearman(GEX_week(t), 다음 4h RG1·RG2). 이론: 롱감마(GEX+) → RG 음 → 부호 −. "
                  "T·T_block·T_rfq = 2026 전체(~A.END) · 다섯 가정 전부 = 47일(2026-08-15~A.END). 보조: 같은 시각 직전 4h RG",
           "theory_sign": -1, "E": 0.05},
    "boundary": "지표 가격점 ≤ t, 결과 가격점 ≥ t+5분(g=1봉). 1분봉 RG 는 보조(호가 튕김이 음의 자기상관을 만듦)",
}


# ───────────────────────── 데이터 ─────────────────────────
def load_5m() -> pd.Series:
    """5분봉 종가, index = 봉 끝 시각(ms). 2021-01~11 vision 월 · 2021-12~2025 api csv · 2026 = 1분 parquet 을 5분으로."""
    parts = []
    for f in sorted((KL / "vision5m").glob("ETHUSDT-5m-2021-*.csv")) + [KL / "ETHUSDT-5m-2023-12.csv"]:   # api csv 에 2023-12 가 빠짐
        d = pd.read_csv(f, header=None, usecols=[0, 4], names=["t", "c"])
        parts.append(d.apply(pd.to_numeric, errors="coerce").dropna())
    a = pd.read_csv(KL / "ETHUSDT-5m-api.csv", usecols=["timestamp", "close"])
    parts.append(pd.DataFrame({"t": pd.to_datetime(a["timestamp"], utc=True).astype("int64") // 10**6, "c": a["close"]}))
    old = pd.concat(parts).drop_duplicates("t", keep="last").set_index("t")["c"]
    old = old[(old.index >= D0) & (old.index < ms("2026-01-01"))]
    m1 = pd.read_parquet(ROOT / "tmp/opt_validate_20261002/ethusdt_1m.parquet", columns=["open_time", "close"])
    g = m1.groupby(m1["open_time"] // M5 * M5)["close"]
    new = g.last()[g.size() == 5]
    s = pd.concat([old, new]); s.index = s.index.astype("int64") + M5          # open → 끝 시각
    return s.reindex(np.arange(D0 + M5, D1 + M5, M5, dtype="int64"))


def load_1m() -> pd.Series:
    """1분 종가, index = 봉 끝 시각. 2023-12-31~ api csv + 2026 parquet(보조 지표용)."""
    a = pd.read_csv(KL / "ETHUSDT-1m-api.csv", usecols=["timestamp", "close"])
    a = pd.Series(a["close"].to_numpy(), index=pd.to_datetime(a["timestamp"], utc=True).astype("int64") // 10**6)
    a = a[a.index < ms("2026-01-01")]
    m1 = pd.read_parquet(ROOT / "tmp/opt_validate_20261002/ethusdt_1m.parquet", columns=["open_time", "close"])
    s = pd.concat([a, m1.set_index("open_time")["close"]]); s = s[~s.index.duplicated(keep="last")]
    s.index = s.index + 60_000
    return s.reindex(np.arange(int(s.index.min()), D1 + 60_000, 60_000, dtype="int64"))


# ───────────────────────── 지표 ─────────────────────────
def acf1(w: np.ndarray) -> np.ndarray:
    x = w - w.mean(1, keepdims=True)
    return (x[:, 1:] * x[:, :-1]).sum(1) / (x * x).sum(1)


def vr(w: np.ndarray) -> np.ndarray:
    r15 = w.reshape(len(w), -1, 3).sum(2)
    return r15.var(1, ddof=1) / (3 * w.var(1, ddof=1))


def trail(r: np.ndarray, i: np.ndarray, n: int) -> np.ndarray:
    """가격점 인덱스 i 에서 끝나는 직전 n 개 수익률(r[k] = logp[k] − logp[k−1]). 범위 밖 = NaN 행."""
    W = sliding_window_view(np.concatenate([np.full(n, np.nan), r]), n)      # W[k] = r[k−n+1 .. k]
    j = np.clip(i, 0, len(r) - 1)
    out = W[j + 1].copy(); out[(i < 0) | (i >= len(r))] = np.nan
    return out


def panel(logp: pd.Series, every: int) -> pd.DataFrame:
    """logp: 5분 가격점(끝 시각 격자). every = 표본 간격(5분 봉 수). 행 = 시각 t."""
    lp = logp.to_numpy(); r = np.diff(lp, prepend=np.nan)
    t = logp.index.to_numpy(); i = np.flatnonzero(t % (every * M5) == 0)
    d = pd.DataFrame({"t": t[i]})
    for nm, n in WINS.items():
        w = trail(r, i, n)
        d[f"rg1_{nm}"], d[f"rg2_{nm}"] = acf1(w), vr(w)
    fw = trail(r, i + GAP + 48, 48)                                 # 가격점 [t+g, t+g+4h]
    d["f_rg1_4h"], d["f_rg2_4h"] = acf1(fw), vr(fw)
    d["rv_next4h"] = (fw ** 2).sum(1)
    for nm, n in (("rv1h", 12), ("rv24h", 288), ("rv7d", 2016)):
        d[nm] = (trail(r, i, n) ** 2).sum(1)
    at = lambda k: lp[np.clip(k, 0, len(lp) - 1)]                   # noqa: E731
    ok = lambda k: (k >= 0) & (k < len(lp))                          # noqa: E731
    d["r_past"] = np.where(ok(i - 12), at(i) - at(i - 12), np.nan)
    d["r_next"] = np.where(ok(i + GAP + 12), at(i + GAP + 12) - at(i + GAP), np.nan)
    d["r_next_g0"] = np.where(ok(i + 12), at(i + 12) - at(i), np.nan)    # 보조: 공백 없음(가격점 t 공유)
    sig = np.sqrt(d["rv24h"] / 24)
    d["rp_z"], d["rn_z"], d["rn0_z"] = d["r_past"] / sig, d["r_next"] / sig, d["r_next_g0"] / sig
    d["day"] = d["t"] // D_MS; d["hour"] = (d["t"] // H_MS) % 24
    return d


# ───────────────────────── 통계 ─────────────────────────
def boot_ols(X: np.ndarray, y: np.ndarray, day: np.ndarray, nb=NB, seed=SEED) -> tuple[np.ndarray, np.ndarray, int]:
    """OLS 계수(전체)와 일 블록 부트스트랩 계수(nb×k). 일별 충분통계(X'X, X'y)를 일 표본 횟수로 가중합."""
    m = np.isfinite(X).all(1) & np.isfinite(y); X, y, day = X[m], y[m], day[m]
    o = np.argsort(day, kind="stable"); X, y, day = X[o], y[o], day[o]
    st = np.flatnonzero(np.r_[True, day[1:] != day[:-1]]); k = X.shape[1]
    XX = np.add.reduceat(np.einsum("ni,nj->nij", X, X).reshape(len(X), -1), st).reshape(len(st), k, k)
    Xy = np.add.reduceat(X * y[:, None], st)
    full = np.linalg.lstsq(XX.sum(0), Xy.sum(0), rcond=None)[0]
    rng = np.random.default_rng(seed); D = len(st)
    C = np.stack([np.bincount(rng.integers(0, D, D), minlength=D) for _ in range(nb)]).astype(float)
    A_ = np.einsum("bd,dij->bij", C, XX) + 1e-12 * np.eye(k)
    return full, np.linalg.solve(A_, (C @ Xy)[..., None])[..., 0], D


def contrast(fit, c, sign, E) -> dict:
    full, bs, D = fit; c = np.asarray(c, float)
    lo, hi = np.percentile(bs @ c, [2.5, 97.5])
    return A.judge(float(full @ c), float(lo), float(hi), D, sign, E)


def spearman(x, y, day, sign, E) -> dict:
    m = np.isfinite(x) & np.isfinite(y)
    rx = pd.Series(x[m]).rank().to_numpy(); ry = pd.Series(y[m]).rank().to_numpy()
    rx, ry = (rx - rx.mean()) / rx.std(), (ry - ry.mean()) / ry.std()
    j = contrast(boot_ols(np.c_[np.ones(len(rx)), rx], ry, day[m]), [0, 1], sign, E)   # 표준화 순위의 기울기 = Spearman
    return {**j, "n": int(m.sum())}


def pct_of(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    ref = np.sort(ref[np.isfinite(ref)])
    return np.where(np.isfinite(x), np.searchsorted(ref, x, "right") / len(ref), np.nan)


def dummies(v: np.ndarray, n: int) -> np.ndarray:
    """NaN 은 NaN 행(boot_ols 가 버림)."""
    return np.where(np.isfinite(v)[:, None], (v[:, None] == np.arange(n)[None, :]).astype(float), np.nan)


def qbin(p: np.ndarray, n: int) -> np.ndarray:
    return np.minimum(np.floor(p * n), n - 1)


# ───────────────────────── 가설 ─────────────────────────
def h1(d4: pd.DataFrame, ind: str, fut: str, ref: np.ndarray) -> dict:
    out = {}
    for per, m in periods(d4).items():
        s = d4[m]; day = s["day"].to_numpy()
        q = qbin(pct_of(s[ind].to_numpy(), ref), 5)
        fit = boot_ols(dummies(q, 5), s[fut].to_numpy(), day)
        out[per] = {"spearman": spearman(s[ind].to_numpy(), s[fut].to_numpy(), day, +1, 0.05),
                    "q5_minus_q1": contrast(fit, [-1, 0, 0, 0, 1], +1, 0.05),
                    "mean_by_quintile": [round(float(v), 4) for v in fit[0]]}
    return out


def h2_X(s, p, rpcol="rp_z", hour_fe=False, extra=None):
    rp = s[rpcol].to_numpy()
    base = dummies(s["hour"].to_numpy(), 24) if hour_fe else np.ones((len(s), 1))
    X = np.c_[base, rp, p, rp * p] if extra is None else np.c_[base, rp, p, rp * p, extra]
    c = np.zeros(X.shape[1]); c[base.shape[1] + 2] = 1
    return X, c


def h2(d: pd.DataFrame, ind: str, ref: np.ndarray, full=True) -> dict:
    out = {}
    for per, m in periods(d).items():
        s = d[m].reset_index(drop=True); day = s["day"].to_numpy(); y = s["rn_z"].to_numpy()
        p = pct_of(s[ind].to_numpy(), ref)
        X, c = h2_X(s, p); r = {"interaction": contrast(boot_ols(X, y, day), c, +1, 0.05), "n_hours": int(np.isfinite(X).all(1).sum())}
        if full:
            q = qbin(p, 5); Dq = dummies(q, 5); rp = s["rp_z"].to_numpy()
            fq = boot_ols(np.c_[Dq, Dq * rp[:, None]], y, day)
            r["beta_by_quintile"] = [contrast(fq, np.eye(10)[5 + k], -1, 0.05) for k in range(5)]
            r["beta_q5_minus_q1"] = contrast(fq, np.eye(10)[9] - np.eye(10)[5], +1, 0.05)
            big = np.abs(rp) > 1
            Xb, cb = h2_X(s[big], p[big]); r["big_move_interaction"] = {**contrast(boot_ols(Xb, y[big], day[big]), cb, +1, 0.05),
                                                                        "n": int(big.sum())}
            fb = boot_ols(np.c_[Dq, Dq * rp[:, None]][big], y[big], day[big])
            r["big_move_beta_by_quintile"] = [round(float(fb[0][5 + k]), 4) for k in range(5)]
            Xh, ch = h2_X(s, p, hour_fe=True); r["hour_fe"] = contrast(boot_ols(Xh, y, day), ch, +1, 0.05)
            Xg, cg = h2_X(s, p); r["gap0_label"] = contrast(boot_ols(Xg, s["rn0_z"].to_numpy(), day), cg, +1, 0.05)
            lv = np.log(s["rv24h"].to_numpy()); lv = (lv - np.nanmean(lv)) / np.nanstd(lv)
            Xv, cv = h2_X(s, p, extra=np.c_[lv, rp * lv]); r["vol_interaction_ctrl"] = contrast(boot_ols(Xv, y, day), cv, +1, 0.05)
            vt = qbin(pct_of(s["rv24h"].to_numpy(), REF_RV), 3)
            r["by_rv_tercile"] = {}
            for k in range(3):
                mk = vt == k; Xk, ck = h2_X(s[mk], p[mk])
                r["by_rv_tercile"][f"T{k + 1}"] = contrast(boot_ols(Xk, y[mk], day[mk]), ck, +1, 0.05)
            ps = np.random.default_rng(SEED).permutation(p)
            Xs, cs = h2_X(s, ps); r["ctrl_shuffled_rg"] = contrast(boot_ols(Xs, y, day), cs, +1, 0.05)
            pl = pct_of(s[ind].shift(24).to_numpy(), ref)
            Xl, cl = h2_X(s, pl); r["ctrl_rg_lag1d"] = contrast(boot_ols(Xl, y, day), cl, +1, 0.05)
            # 화면 문구용: 큰 움직임(|r_past_z|>1)의 RG 삼분위별 «되돌림 비율»(다음 1h 부호 반대)·평균 지속 bp
            t3 = qbin(p, 3)
            rev = np.where(s["r_next"].notna(), (np.sign(s["r_next"]) == -np.sign(s["r_past"])).astype(float), np.nan)
            cont_bp = (np.sign(s["r_past"]) * s["r_next"] * 1e4).to_numpy()
            frev = boot_ols(dummies(t3, 3)[big], rev[big], day[big]); fbp = boot_ols(dummies(t3, 3)[big], cont_bp[big], day[big])
            r["big_move_by_tercile"] = {f"T{k + 1}": {"reversal_rate": ci_of(frev, k), "continuation_bp": ci_of(fbp, k),
                                                      "n": int((big & (t3 == k)).sum())} for k in range(3)}
            r["big_move_tercile_diff_T1_minus_T3"] = {"reversal_rate": ci_diff(frev), "continuation_bp": ci_diff(fbp)}
            r["big_move_all"] = {"reversal_rate": round(float(np.nanmean(rev[big])), 4), "continuation_bp": round(float(np.nanmean(cont_bp[big])), 2)}
        out[per] = r
    return out


def ci_of(fit, k):
    lo, hi = np.percentile(fit[1][:, k], [2.5, 97.5]); return [round(float(fit[0][k]), 4), round(float(lo), 4), round(float(hi), 4)]


def ci_diff(fit):
    v = fit[1][:, 0] - fit[1][:, 2]; lo, hi = np.percentile(v, [2.5, 97.5])
    return [round(float(fit[0][0] - fit[0][2]), 4), round(float(lo), 4), round(float(hi), 4)]


def h3(d: pd.DataFrame, ind: str, ref: np.ndarray) -> dict:
    out = {}
    for per, m in periods(d).items():
        s = d[m]; day = s["day"].to_numpy(); y = np.log(s["rv_next4h"].to_numpy())
        p = pct_of(s[ind].to_numpy(), ref)
        L = np.log(s[["rv1h", "rv24h", "rv7d"]].to_numpy())
        X = np.c_[np.ones(len(s)), L, p]
        Xh = np.c_[dummies(s["hour"].to_numpy(), 24), L, p]
        out[per] = {"pct_coef": contrast(boot_ols(X, y, day), np.eye(5)[4], +1, 0.05),
                    "pct_coef_hour_fe": contrast(boot_ols(Xh, y, day), np.eye(28)[27], +1, 0.05),
                    "pct_coef_no_har": contrast(boot_ols(np.c_[np.ones(len(s)), p], y, day), [0, 1], +1, 0.05)}
    return out


def periods(d):
    return {"discovery": (d["t"] >= D0) & (d["t"] < SPLIT), "holdout": (d["t"] >= SPLIT) & (d["t"] < D1)}


def h4(logp: pd.Series) -> dict:
    gx = {"T": pd.read_parquet(ROOT / "tmp/dealer_gex_reconstruct_2026_20261001/hourly_dealer_gex.parquet", columns=["ts", "gex_week"]),
          "T_block": pd.read_parquet(ROOT / "tmp/dealer_assumption_matrix_20261002/tblock_gex_hourly.parquet", columns=["ts", "gex_week"]),
          "T_rfq": pd.read_parquet(ROOT / "tmp/block_rfq_dealer_20261002/pass_rfq/tblock_gex_hourly.parquet", columns=["ts", "gex_week"]),
          "C": pd.read_parquet(ROOT / "tmp/gamma_rehedge_footprint_2026_20261001/conv_gex_hourly.parquet", columns=["ts", "gex_week"])}
    gx["R"] = gx["C"].assign(gex_week=-gx["C"]["gex_week"])
    lp = logp.to_numpy(); r = np.diff(lp, prepend=np.nan); t5 = logp.index.to_numpy()
    out = {}
    for a, g in gx.items():
        g = g[g["ts"] < A.END]
        te = -(-g["ts"].to_numpy() // M5) * M5                       # 스냅샷 시각을 다음 5분 경계로(그 시각에 이미 알려짐)
        i = np.searchsorted(t5, te)
        fw, tw = trail(r, i + GAP + 48, 48), trail(r, i, 48)
        d = pd.DataFrame({"t": te, "gex": g["gex_week"].to_numpy(), "f1": acf1(fw), "f2": vr(fw), "c1": acf1(tw), "c2": vr(tw),
                          "day": te // D_MS})
        for wn, lo in (("2026", ms("2026-01-01")), ("47d", ms("2026-08-15"))):
            if a in ("C", "R") and wn == "2026":
                continue
            s = d[d["t"] >= lo]; day = s["day"].to_numpy(); x = s["gex"].to_numpy()
            out[f"{a}|{wn}"] = {k: spearman(x, s[c].to_numpy(), day, -1, 0.05)
                                for k, c in (("fwd_rg1_4h", "f1"), ("fwd_rg2_4h", "f2"), ("cur_rg1_4h", "c1"), ("cur_rg2_4h", "c2"))}
            print("H4", a, wn, {k: (v["est"], v["ci"], v["verdict"]) for k, v in out[f"{a}|{wn}"].items()}, flush=True)
    return out


# ───────────────────────── 실행 ─────────────────────────
REF_RV = None


def main():
    global REF_RV
    OUT.mkdir(parents=True, exist_ok=True)
    c5 = load_5m(); logp = np.log(c5)
    res = {"criteria": CRITERIA, "data": {"n_5m": int(len(c5)), "missing_5m": int(c5.isna().sum()),
                                          "first": str(pd.to_datetime(c5.first_valid_index(), unit="ms")),
                                          "last": str(pd.to_datetime(c5.last_valid_index(), unit="ms"))}}
    # 2026 1분→5분 vs api 5분 겹침 대조(양성 대조)
    a = pd.read_csv(KL / "ETHUSDT-5m-api.csv", usecols=["timestamp", "close"])
    a = pd.Series(a["close"].to_numpy(), index=pd.to_datetime(a["timestamp"], utc=True).astype("int64") // 10**6 + M5)
    ov = a[a.index > ms("2026-01-01")].reindex(c5.index).dropna(); dd = (ov - c5.reindex(ov.index)).abs()
    res["data"]["overlap_2026_api_vs_1m"] = {"n": int(dd.notna().sum()), "max_abs_diff": float(dd.max())}
    print(res["data"], flush=True)

    d = panel(logp, 12)                                  # 1h 격자
    d4 = d[d["hour"] % 4 == 0].reset_index(drop=True)    # 4h 격자(H1)
    disc = periods(d)["discovery"]
    REF_RV = d.loc[disc, "rv24h"].to_numpy()
    res["describe"] = {per: {c: round(float(d.loc[m, c].mean()), 4) for c in ("rg1_1h", "rg1_4h", "rg1_12h", "rg2_1h", "rg2_4h", "rg2_12h")}
                       for per, m in periods(d).items()}
    res["corr_rg1_rg2_4h"] = round(float(d["rg1_4h"].corr(d["rg2_4h"], method="spearman")), 3)
    for ind in ("rg1_4h", "rg2_4h", "rg1_1h", "rg2_1h", "rg1_12h", "rg2_12h"):
        ref = d.loc[disc, ind].to_numpy(); main_ind = ind.endswith("4h")
        fut = "f_" + ind[:3] + "_4h"
        res[ind] = {"H1": h1(d4, ind, fut, ref), "H2": h2(d, ind, ref, full=main_ind), "H3": h3(d, ind, ref)}
        print(ind, "H1", {p: v["spearman"]["est"] for p, v in res[ind]["H1"].items()},
              "H2", {p: (v["interaction"]["est"], v["interaction"]["ci"]) for p, v in res[ind]["H2"].items()},
              "H3", {p: v["pct_coef"]["est"] for p, v in res[ind]["H3"].items()}, flush=True)

    # 1분 보조: 직전 4h 의 1분 수익률 자기상관(호가 튕김 포함)
    l1 = np.log(load_1m()); r1 = np.diff(l1.to_numpy(), prepend=np.nan); t1 = l1.index.to_numpy()
    i1 = np.searchsorted(t1, d["t"].to_numpy()); ok = (i1 < len(t1)) & (t1[np.minimum(i1, len(t1) - 1)] == d["t"].to_numpy())
    d["rg1m_4h"] = np.where(ok, acf1(trail(r1, np.where(ok, i1, -1), 240)), np.nan)
    m24 = d["t"] < SPLIT
    ref1 = d.loc[m24, "rg1m_4h"].to_numpy()
    res["rg1m_4h_aux"] = {"note": "1분 RG 는 호가 튕김(bid-ask bounce)이 음의 자기상관을 섞음 — 발견 구간은 2024 만(1분 자료 2023-12-31~)",
                          "mean": {p: round(float(d.loc[m, "rg1m_4h"].mean()), 4) for p, m in periods(d).items()},
                          "spearman_with_rg1_4h": round(float(d["rg1m_4h"].corr(d["rg1_4h"], method="spearman")), 3),
                          "H2": h2(d, "rg1m_4h", ref1, full=False)}
    res["H4"] = h4(logp)
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    write_summary(res)


def write_summary(res):
    f = lambda j: f"**{j['verdict']}** {j['est']:+.4f} [{j['ci'][0]:+.4f}, {j['ci'][1]:+.4f}] {j['n_days']}일"   # noqa: E731
    L = ["# 실현 감마 국면 검증 (2026-10-02)", "", f"자료: {res['data']}", "",
         "## H1~H3 (판정 = 보류 2025-01~2026-09)", "",
         "| 지표 | 가설 | 발견 2021~2024 | 보류 2025~2026-09 |", "|---|---|---|---|"]
    for ind in ("rg1_4h", "rg2_4h", "rg1_1h", "rg2_1h", "rg1_12h", "rg2_12h"):
        R_ = res[ind]
        rows = [("H1 Spearman", lambda p: R_["H1"][p]["spearman"]), ("H1 Q5−Q1", lambda p: R_["H1"][p]["q5_minus_q1"]),
                ("H2 상호작용", lambda p: R_["H2"][p]["interaction"]), ("H3 pct 계수", lambda p: R_["H3"][p]["pct_coef"])]
        for nm, g in rows:
            L.append(f"| {ind} | {nm} | {f(g('discovery'))} | {f(g('holdout'))} |")
    for ind in ("rg1_4h", "rg2_4h"):
        L += ["", f"## H2 세부 — {ind}", "", "| 항목 | 발견 | 보류 |", "|---|---|---|"]
        H = res[ind]["H2"]
        for k in ("beta_q5_minus_q1", "big_move_interaction", "hour_fe", "vol_interaction_ctrl", "gap0_label", "ctrl_shuffled_rg", "ctrl_rg_lag1d"):
            L.append(f"| {k} | {f(H['discovery'][k])} | {f(H['holdout'][k])} |")
        for k in ("T1", "T2", "T3"):
            L.append(f"| RV 삼분위 {k} | {f(H['discovery']['by_rv_tercile'][k])} | {f(H['holdout']['by_rv_tercile'][k])} |")
        L.append("| 분위별 β (Q1..Q5) | " + " · ".join(f"{b['est']:+.3f}" for b in H["discovery"]["beta_by_quintile"]) + " | "
                 + " · ".join(f"{b['est']:+.3f}" for b in H["holdout"]["beta_by_quintile"]) + " |")
        for k in ("T1", "T2", "T3"):
            a, b = H["discovery"]["big_move_by_tercile"][k], H["holdout"]["big_move_by_tercile"][k]
            L.append(f"| 큰 움직임 RG {k} 되돌림률 · 지속 bp | {a['reversal_rate']} · {a['continuation_bp']} (n={a['n']}) | "
                     f"{b['reversal_rate']} · {b['continuation_bp']} (n={b['n']}) |")
        L.append(f"| 큰 움직임 T1−T3 | {H['discovery']['big_move_tercile_diff_T1_minus_T3']} | {H['holdout']['big_move_tercile_diff_T1_minus_T3']} |")
    L += ["", "## 1분 RG 보조", "", f"```\n{json.dumps({k: v for k, v in res['rg1m_4h_aux'].items() if k != 'H2'}, ensure_ascii=False)}\n```",
          f"H2 상호작용 발견(2024) {f(res['rg1m_4h_aux']['H2']['discovery']['interaction'])} · 보류 {f(res['rg1m_4h_aux']['H2']['holdout']['interaction'])}",
          "", "## H4 GEX_week × 실현 국면 (Spearman, 이론 부호 −)", "", "| 가정 | 다음 4h RG1 | 다음 4h RG2 | 직전 4h RG1 | 직전 4h RG2 |", "|---|---|---|---|---|"]
    for k, v in res["H4"].items():
        L.append(f"| {k} | " + " | ".join(f(v[c]) for c in ("fwd_rg1_4h", "fwd_rg2_4h", "cur_rg1_4h", "cur_rg2_4h")) + " |")
    L += ["", "## 주의", "",
          "- 48개 수익률의 ACF 표준오차 ≈ 1/√48 ≈ 0.14, 소표본 편향 ≈ −1/48. 4h RG 한 값은 잡음이 대부분이다(H1 순위상관이 작은 이유).",
          "- H2 단위: z = 수익률/σ_t. 상호작용 = RG 백분위 0→1 사이 β 차. 큰 움직임 = |r_past_z|>1, 되돌림률 = 다음 1h 부호 반대 비율.",
          "- 2026 은 1분 parquet 을 5분으로 묶음. api 5분과 겹침 74,179봉 중 10봉만 다름(api 쪽 실시간 수집 미완결 봉).",
          "- H4: GEX 는 느리게 움직여 실효 표본이 시간 수보다 훨씬 작다. 일 블록 CI 도 낙관적일 수 있다."]
    (OUT / "summary.md").write_text("\n".join(L) + "\n")


def selftest():
    # VR: 교대(+1,−1,…) 48개 → 15분 합 ±1 16개 → (16/15)/(3·48/47)
    w = np.tile([1.0, -1.0], 24)[None, :]
    assert abs(vr(w)[0] - (16 / 15) / (3 * 48 / 47)) < 1e-12
    assert vr(np.repeat(np.tile([1.0, -1.0], 8), 3)[None, :])[0] > 2.5          # 3봉씩 같은 방향 = 추세 → VR > 1
    assert acf1(w)[0] < -0.9
    # 시점 경계: t 이후 가격을 바꿔도 지표 불변 · t 이하 가격을 바꿔도 결과(r_next·다음 4h) 불변
    rng = np.random.default_rng(0); idx = np.arange(D0 + M5, D0 + M5 * 4000, M5, dtype="int64")
    lp = pd.Series(np.cumsum(rng.normal(0, 1e-3, len(idx))), index=idx)
    base = panel(lp, 12); k = 200; t = base["t"].iat[k]
    fut = lp.copy(); fut[fut.index > t] += rng.normal(0, 1e-2, int((fut.index > t).sum()))
    past = lp.copy(); past[past.index <= t] += rng.normal(0, 1e-2, int((past.index <= t).sum()))
    a, b = panel(fut, 12).iloc[k], panel(past, 12).iloc[k]
    for c in ("rg1_4h", "rg2_4h", "rg1_12h", "r_past", "rv24h", "rv7d"):
        assert base[c].iat[k] == a[c] or (np.isnan(base[c].iat[k]) and np.isnan(a[c])), c
    for c in ("r_next", "f_rg1_4h", "f_rg2_4h", "rv_next4h"):
        assert abs(base[c].iat[k] - b[c]) < 1e-12, c
    assert abs(base["r_next_g0"].iat[k] - b["r_next_g0"]) > 0                    # 대조: 공백 없는 라벨은 가격점 t 를 공유한다
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
