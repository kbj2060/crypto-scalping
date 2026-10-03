#!/usr/bin/env python3
"""미결제 감마 지도 × 실현 감마 국면(RG) · 미결제 감마 경계(플립) 검정 (2026-10-02).

근사 1 = 행사가별 크기(미결제 × BS 감마 × S² × 1%, 부호 없음) + 전체 부호는 RG 국면(rg1_12h).
근사 2 = 관행(콜 +OI · 풋 −OI)과 뒤집은 관행이 같은 자리에 내는 플립 = 부호 없는 «경계선».
감마·플립은 옛 수집기(4a55ed69 collect_deribit_option_gex_20260815.py 의 _gamma/flip_of) 그대로:
  지수 ±15% 25점 · 만기별 선도가를 같은 비율로 옮김 · 지금가에 가장 가까운 부호 전환 선형 보간.
시점: 스냅샷 시각을 다음 5분 경계 te 로 올린다(그 시각에 알려짐). 지표 = te 까지 닫힌 5분봉, 결과 = te+5분 가격점부터.
자료: Deribit ETH 체인(2026-08-15~10-01) · 바이낸스 ETHUSDT 1분→5분(RG 연구 load_5m). 바이낸스 REST 호출 없음.

  python scripts/research_eth_oi_gamma_map_flip_rg_20261002.py
  python scripts/research_eth_oi_gamma_map_flip_rg_20261002.py --selftest
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_eth_realized_gamma_regime_20261002 as R   # noqa: E402  load_5m · panel · acf1 · trail · boot_ols
from research_eth_dealer_assumption_compare_20261002 import judge   # noqa: E402

SRC = ROOT / "tmp/opt_validate_20261002"
OUT = ROOT / "tmp/oi_gamma_map_flip_rg_20261002"
M5, H_MS, D_MS = R.M5, R.H_MS, R.D_MS
ms = R.ms
T0 = ms("2026-08-15")
RG_LO, RG_HI = -0.0769, 0.0124          # rg1_12h 발견 구간(2021~24) 삼분위 — 아래에서 재현 확인
KGRID = 0.85 + 0.0125 * np.arange(25)
SEED = 20261002

# 결과 보기 전 고정(2026-10-02). 결과를 보고 바꾸지 않는다.
CRITERIA = {
    "window": "2026-08-15 ~ 결과 창이 닫히는 마지막 시각(1분봉 2026-09-30 23:59 까지) · 정시에 가장 가까운 스냅샷(±30분) 하나",
    "rule": "일 블록 부트스트랩 2000회 95% CI. 단측(이론 부호): CI 0 배제 & 이론 부호 = 지지 · 반대 부호 = 반대 · "
            "CI ⊂ (−E,+E) = 근거 없음 · 그 외 = 검정력 부족(n_req = n·(1.96·SE/E)²). 양측(H2): CI 0 배제 = 차이 있음(부호 표시)",
    "regime": f"RG = rg1_12h(te 까지 144개 5분 수익률 ACF1). 되돌림(< {RG_LO}) = −1 · 보통 = 0 · 추세(> {RG_HI}) = +1",
    "Y1": "ln RV(te+5분부터 4h, 5분 수익률 제곱합) − HAR 예측(ln RV 1h·24h·7d + UTC 시 고정효과, 2025-01-01~2026-08-14 에 적합 = 검정 창 밖)",
    "H1a": "부호 = RG 그 자체 → 동어반복, 검정 대상 아님",
    "H1b": {"main": "Y1 ~ 1 + Mp + reg + reg×Mp, reg×Mp 계수. Mp = M 의 표본 내 백분위(0~1). "
                    "M = Σ(7일 안 만기, |K/지금가−1| ≤ 2%) OI·gamma_bs·선도²·1% (부호 없음)",
            "theory_sign": +1, "E": 0.10,
            "why_sign": "되돌림(reg −1)에서 M↑ → 더 눌림(기울기 b−d<0), 추세(+1)에서 M↑ → 더 커짐(b+d>0) ⇒ d>0",
            "M_alone": "Y1 ~ 1 + Mp, Mp 계수(앞선 연구: GEX 크기 = 활동=변동성 결과 지표로 + 방향 → 이 방향을 이론 부호 +로 둠), E 0.10",
            "aux": ["국면별 Mp 기울기", "4h 격자(겹치지 않는 결과 창)", "ln DVOL 통제", "M_curve(7일 안 전 행사가 감마$ at 지금가, 부호 없음)",
                    "HAR 표본 내(회귀에 HAR 항 직접)", "대조: 셔플 Mp · 1일 늦춘 Mp"]},
    "H1c": {"main": "쌍(te, K): |ln(P_te/K)| ≤ 0.3%, K ∈ 지금가 ±8% 7일 안 만기 행사가. top = 행사가별 감마$(부호 없음) 상위 3 · "
                    "small = 그 띠 중앙값 미만(중간은 제외). 결과 = 다음 1h(가격점 te+5…te+60분 12개) 평균 |ln(P/K)| − |ln(P_te/K)| (bp, 음 = 다가옴/머묾). "
                    "Δ ~ 1 + top + reg + top×reg + d0, top×reg 계수",
            "theory_sign": +1, "E": 5.0,
            "why_sign": "되돌림(reg −1)에서 top 이 small 보다 더 끌어당김(top 효과 더 음) ⇒ top×reg > 0",
            "aux": ["보조 결과: 12개 중 ±0.3% 안 비율(stay, 이론 부호 −)", "국면 조건 없는 top 계수(핀닝 재확인)", "대조: 셔플 top"]},
    "H2a": {"main": "s = 1[P_te > 플립_week]. 결과 f_rg1_4h(E 0.05) · Y1(E 0.10) ~ 1 + s, s 계수(위 − 아래). 양측. 플립 없는 시각 제외(비율 보고)",
            "aux": ["|거리| 오분위 더미 통제", "|거리| > 1% 만", "현재 rg1_12h 통제", "4h 격자", "플립_all", "대조: 셔플 s · 1일 늦춘 플립"]},
    "H2b": {"main": "교차 = sign(P_{te−1h} − 플립_{직전 스냅샷}) ≠ sign(P_te − 플립_{직전 스냅샷}). "
                    "ΔRG = f_rg1_4h − rg1_4h(직전 4h) ~ |r_1h| 오분위 더미 + cross, cross 계수. 양측 E 0.05 · 보조 Y1",
            "note": "보조 가설"},
    "post_hoc": "결과를 본 뒤 추가(판정에 쓰지 않음): H1b 시 고정효과·0DTE 제외 M · H2a 관행 GEX 부호와 s 의 일치율·관행 GEX 통제",
    "boundary": "지표: 체인 스냅샷 ≤ te, 지금가 = 스냅샷 시각에 닫힌 마지막 1분봉 종가(플립·M 기준) · P_te = te 에 끝나는 5분봉 종가. 결과: 가격점 ≥ te+5분",
}


# ───────────────────────── 감마·플립 (옛 수집기 그대로) ─────────────────────────
def profile(fw, k, iv, yrs, w) -> np.ndarray:
    """25 가격점(KGRID × 지수)의 Σ γ·S²·w·1%. S = 선도가 × 같은 비율."""
    ok = (iv > 0) & (yrs > 0)
    fw, k, iv, yrs, w = fw[ok], k[ok], iv[ok], yrs[ok], w[ok]
    S = fw[:, None] * KGRID[None, :]
    sq = (iv * np.sqrt(yrs))[:, None]
    d1 = (np.log(S / k[:, None]) + 0.5 * (iv ** 2 * yrs)[:, None]) / sq
    gam = np.exp(-0.5 * d1 * d1) / np.sqrt(2 * np.pi) / (S * sq)
    return (gam * S * S * w[:, None]).sum(0) * 0.01


def flip_of(prof: np.ndarray, grid=KGRID):
    """지금(비율 1)에 가장 가까운 부호 전환(선형 보간) — 비율로 반환, 없으면 NaN."""
    best = np.nan
    for a, b, ga, gb in zip(grid, grid[1:], prof, prof[1:]):
        if (ga < 0) != (gb < 0):
            c = a + (b - a) * (-ga) / (gb - ga)
            if np.isnan(best) or abs(c - 1) < abs(best - 1):
                best = c
    return best


def known_at(snap_ms):
    return -(-np.asarray(snap_ms) // M5) * M5        # 다음 5분 경계(이미 경계면 그대로)


def last_closed(end_ms: np.ndarray, vals: np.ndarray, at_ms):
    """at_ms 이하에 끝난 마지막 봉 값."""
    j = np.searchsorted(end_ms, at_ms, "right") - 1
    return np.where(j >= 0, vals[np.maximum(j, 0)], np.nan)


# ───────────────────────── 스냅샷 지표 ─────────────────────────
def snapshot_table() -> tuple[pd.DataFrame, list]:
    ch = pd.read_parquet(SRC / "eth_chain.parquet", columns=["recorded_at_utc", "option_type", "strike", "expiration_ts",
                                                              "days_to_expiry", "open_interest", "mark_iv", "underlying_price", "gamma_bs"])
    snaps = ch["recorded_at_utc"].drop_duplicates().sort_values()
    sm = snaps.astype("int64").to_numpy() // 1000                      # us → ms
    hr = np.round(sm / H_MS).astype("int64") * H_MS
    pick = pd.DataFrame({"snap": snaps.to_numpy(), "ms": sm, "hour": hr, "off": np.abs(sm - hr)})
    pick = pick[pick["off"] <= 30 * 60_000].sort_values("off").drop_duplicates("hour").sort_values("hour")
    m1 = pd.read_parquet(SRC / "ethusdt_1m.parquet", columns=["open_time", "close"])
    idx_all = last_closed(m1["open_time"].to_numpy() + 60_000, m1["close"].to_numpy(), pick["ms"].to_numpy())
    rows, ladders = [], []
    g = ch[ch["recorded_at_utc"].isin(set(pick["snap"]))].groupby("recorded_at_utc")
    for (snap, msn, hour), idx in zip(pick[["snap", "ms", "hour"]].itertuples(index=False), idx_all):
        c = g.get_group(snap)
        c = c[c["expiration_ts"] > snap]
        fw = np.where(c["underlying_price"] > 0, c["underlying_price"], idx).astype(float)
        sg = np.where(c["option_type"] == "call", 1.0, -1.0)
        oi, k = c["open_interest"].to_numpy(float), c["strike"].to_numpy(float)
        iv, yrs = c["mark_iv"].to_numpy(float) / 100, c["days_to_expiry"].to_numpy(float) / 365
        wk = c["days_to_expiry"].to_numpy() <= 7
        r = {"hour": hour, "snap_ms": msn, "te": int(known_at(msn)), "idx": idx}
        for nm, sel in (("week", wk), ("all", np.ones(len(c), bool))):
            p = profile(fw[sel], k[sel], iv[sel], yrs[sel], (sg * oi)[sel])
            r[f"flip_{nm}"] = flip_of(p)
            r[f"flip_{nm}_rev"] = flip_of(-p)                           # 뒤집은 관행 — 같은 자리여야 한다
            r[f"gex_conv_{nm}"] = float(p[12])                          # 관행 GEX at 지금가(사후 대조용)
        gd = (oi * c["gamma_bs"].to_numpy(float) * fw ** 2 * 0.01)       # 행사가별 감마$(부호 없음, 사다리와 같은 식)
        near = wk & (np.abs(k / idx - 1) <= 0.02)
        r["M"] = float(gd[near].sum())
        r["M_ex1d"] = float(gd[near & (c["days_to_expiry"].to_numpy() > 1)].sum())   # 사후: 1일 안 만기(0DTE) 제외
        r["M_curve"] = float(profile(fw[wk], k[wk], iv[wk], yrs[wk], oi[wk])[12])   # 지금가(비율 1) 곡선 크기
        band = wk & (np.abs(k / idx - 1) <= 0.08)
        lad = pd.Series(gd[band]).groupby(k[band]).sum()
        rows.append(r); ladders.append((hour, lad))
    return pd.DataFrame(rows), ladders


# ───────────────────────── 통계 도우미 ─────────────────────────
def fit_c(X, y, day, c, sign, E, two_sided=False) -> dict:
    full, bs, D = R.boot_ols(X, y, day)
    c = np.asarray(c, float); lo, hi = np.percentile(bs @ c, [2.5, 97.5])
    j = judge(float(full @ c), float(lo), float(hi), D, sign, E)
    if two_sided and j["verdict"] in ("지지", "반대"):
        j["verdict"] = "차이 있음(" + ("+" if j["est"] > 0 else "−") + ")"
    j["n"] = int((np.isfinite(X).all(1) & np.isfinite(y)).sum())
    return j


def coef(X, y, day, k, sign, E, two_sided=False):
    c = np.zeros(X.shape[1]); c[k] = 1
    return fit_c(X, y, day, c, sign, E, two_sided)


def pct_rank(x):
    return pd.Series(x).rank(pct=True).to_numpy()


def qdum(x, n=5):
    q = pd.qcut(pd.Series(x).rank(method="first"), n, labels=False).to_numpy()
    return R.dummies(q.astype(float), n)


# ───────────────────────── 실행 ─────────────────────────
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c5 = R.load_5m(); logp = np.log(c5)
    res = {"criteria": CRITERIA}

    # HAR(검정 창 밖) + 국면 경계 재현
    hp = R.panel(logp[logp.index >= ms("2020-12-20")], 12)
    disc = (hp["t"] >= R.D0) & (hp["t"] < R.SPLIT)
    q = np.nanquantile(hp.loc[disc, "rg1_12h"], [1 / 3, 2 / 3])
    res["rg_cut_check"] = {"discovery_terciles": [round(float(v), 4) for v in q], "used": [RG_LO, RG_HI]}
    assert abs(q[0] - RG_LO) < 2e-3 and abs(q[1] - RG_HI) < 2e-3, q
    fitm = (hp["t"] >= ms("2025-01-01")) & (hp["t"] < T0)
    def har_X(d):
        return np.c_[R.dummies(d["hour"].to_numpy(), 24), np.log(d[["rv1h", "rv24h", "rv7d"]].to_numpy())]
    hf = hp[fitm]; Xh = har_X(hf); yh = np.log(hf["rv_next4h"].to_numpy()); ok = np.isfinite(Xh).all(1) & np.isfinite(yh)
    beta = np.linalg.lstsq(Xh[ok], yh[ok], rcond=None)[0]

    # 5분 패널(te 조회용)
    P = R.panel(logp[logp.index >= ms("2026-07-25")], 1).set_index("t")
    P["y1"] = np.log(P["rv_next4h"]) - har_X(P.reset_index()) @ beta
    lp = logp.to_numpy(); tl = logp.index.to_numpy()

    S, ladders = snapshot_table()
    S = S.join(P[["rg1_12h", "rg1_4h", "f_rg1_4h", "y1", "rv_next4h", "rv1h", "rv24h", "rv7d"]], on="te")
    S = S[np.isfinite(S["y1"]) & np.isfinite(S["f_rg1_4h"]) & np.isfinite(S["rg1_12h"])].reset_index(drop=True)
    i_te = np.searchsorted(tl, S["te"].to_numpy())
    S["p_te"] = np.exp(lp[i_te])
    S["p_m1h"] = np.exp(lp[i_te - 12])
    S["reg"] = np.where(S["rg1_12h"] < RG_LO, -1, np.where(S["rg1_12h"] > RG_HI, 1, 0)).astype(float)
    S["Mp"] = pct_rank(S["M"].to_numpy()); S["Mcp"] = pct_rank(S["M_curve"].to_numpy())
    dv = pd.read_parquet(SRC / "eth_dvol_1h.parquet")
    S["ldvol"] = np.log(last_closed(dv["ts"].to_numpy() + H_MS, dv["c"].to_numpy(), S["snap_ms"].to_numpy()))
    lag = S.set_index("hour")
    for c in ("Mp", "flip_week", "idx"):
        S[c + "_lag1d"] = lag[c].reindex(S["hour"] - D_MS).to_numpy()
    S["day"] = S["te"] // D_MS
    res["data"] = {"n_hours": len(S), "n_days": int(S["day"].nunique()),
                   "first_te": str(pd.to_datetime(S["te"].min(), unit="ms")), "last_te": str(pd.to_datetime(S["te"].max(), unit="ms")),
                   "regime_share": {str(int(k)): round(float(v), 3) for k, v in S["reg"].value_counts(normalize=True).sort_index().items()},
                   "flip_rev_identical": bool(np.allclose(S["flip_week"], S["flip_week_rev"], equal_nan=True)
                                              and np.allclose(S["flip_all"], S["flip_all_rev"], equal_nan=True))}
    print(res["rg_cut_check"], res["data"], flush=True)
    day = S["day"].to_numpy(); one = np.ones(len(S)); y1 = S["y1"].to_numpy(); reg = S["reg"].to_numpy()

    # ── H1b ──
    def h1b(d, mcol="Mp"):
        m = d[mcol].to_numpy(); rg = d["reg"].to_numpy(); X = np.c_[np.ones(len(d)), m, rg, rg * m]
        return coef(X, d["y1"].to_numpy(), d["day"].to_numpy(), 3, +1, 0.10)
    H = {"main_interaction": h1b(S),
         "M_alone": coef(np.c_[one, S["Mp"]], y1, day, 1, +1, 0.10),
         "reg_alone(복제 확인, 이론 +)": coef(np.c_[one, reg], y1, day, 1, +1, 0.10)}
    Dr = R.dummies(reg + 1, 3); Xs = np.c_[Dr, Dr * S["Mp"].to_numpy()[:, None]]
    H["slope_by_regime"] = {nm: coef(Xs, y1, day, 3 + k, s, 0.10) for k, nm, s in ((0, "되돌림(이론 −)", -1), (1, "보통", -1), (2, "추세(이론 +)", +1))}
    H["grid4h"] = h1b(S[S["hour"] % (4 * H_MS) == 0])
    m = S["Mp"].to_numpy()
    H["ctrl_ldvol"] = coef(np.c_[one, m, reg, reg * m, S["ldvol"]], y1, day, 3, +1, 0.10)
    H["M_curve"] = h1b(S, "Mcp")
    Xin = np.c_[R.dummies(((S["hour"] // H_MS) % 24).to_numpy(), 24), np.log(S[["rv1h", "rv24h", "rv7d"]].to_numpy()), m, reg, reg * m]
    H["har_in_sample"] = coef(Xin, np.log(S["rv_next4h"].to_numpy()), day, Xin.shape[1] - 1, +1, 0.10)
    H["ctrl_shuffled_M"] = h1b(S.assign(Mp=np.random.default_rng(SEED).permutation(m)))
    H["ctrl_M_lag1d"] = h1b(S.assign(Mp=S["Mp_lag1d"]))
    H["M_by_regime_mean"] = S.groupby("reg")["M"].median().round(0).to_dict()
    H["spearman_M_ldvol"] = round(float(S["M"].corr(S["ldvol"], method="spearman")), 3)
    res["H1b"] = H
    print("H1b", {k: (v["est"], v["ci"], v["verdict"]) for k, v in H.items() if isinstance(v, dict) and "est" in v}, flush=True)

    # ── H1c ──
    lad = dict(ladders); pairs = []
    for j, r in S.iterrows():
        L = lad[r["hour"]]
        if len(L) < 4:
            continue
        top = set(L.nlargest(3).index); med = L.median()
        path = lp[i_te[j] + 1: i_te[j] + 13]
        for K, gv in L.items():
            d0 = abs(np.log(r["p_te"] / K))
            if d0 > 0.003 or (K not in top and gv >= med):
                continue
            dist = np.abs(path - np.log(K))
            pairs.append({"day": r["day"], "reg": r["reg"], "top": float(K in top), "d0": d0 * 1e4,
                          "dd": (dist.mean() - d0) * 1e4, "stay": float((dist <= 0.003).mean()), "hour": r["hour"]})
    Q = pd.DataFrame(pairs); qd = Q["day"].to_numpy(); tp = Q["top"].to_numpy(); qr = Q["reg"].to_numpy()
    Xc = np.c_[np.ones(len(Q)), tp, qr, tp * qr, Q["d0"]]
    C = {"main_dd": coef(Xc, Q["dd"].to_numpy(), qd, 3, +1, 5.0),
         "stay(이론 −)": coef(Xc, Q["stay"].to_numpy(), qd, 3, -1, 0.05),
         "top_unconditional(핀닝, 이론 −)": coef(np.c_[np.ones(len(Q)), tp, Q["d0"]], Q["dd"].to_numpy(), qd, 1, -1, 5.0),
         "ctrl_shuffled_top": coef(np.c_[Xc[:, :1], (ts := np.random.default_rng(SEED).permutation(tp)), qr, ts * qr, Q["d0"]],
                                   Q["dd"].to_numpy(), qd, 3, +1, 5.0),
         "n_pairs": {"top": int(tp.sum()), "small": int((1 - tp).sum()),
                     "by_reg_top": Q[Q.top == 1].groupby("reg").size().to_dict(), "by_reg_small": Q[Q.top == 0].groupby("reg").size().to_dict()},
         "cell_mean_dd_bp": {f"reg{a:+.0f}|top{b:.0f}": round(float(v), 2) for (a, b), v in Q.groupby(["reg", "top"])["dd"].mean().items()}}
    res["H1c"] = C
    print("H1c", {k: (v["est"], v["ci"], v["verdict"]) for k, v in C.items() if "est" in v}, C["n_pairs"], flush=True)

    # ── H2a ──
    def flipabs(col="flip_week", idxc="idx"):
        return S[idxc].to_numpy() * S[col].to_numpy()
    fa = flipabs(); dist = S["p_te"].to_numpy() / fa - 1; has = np.isfinite(fa)
    s = (dist > 0).astype(float)
    res["flip_distance"] = {
        "excluded_no_flip_week": round(float(1 - has.mean()), 3), "excluded_no_flip_all": round(float(1 - np.isfinite(S["flip_all"]).mean()), 3),
        "dist_pct_quantiles(지금가/플립−1, %)": {str(p): round(float(np.nanquantile(dist * 100, p)), 2) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "abs_dist_quantiles(%)": {str(p): round(float(np.nanquantile(np.abs(dist) * 100, p)), 2) for p in (0.1, 0.25, 0.5, 0.75, 0.9)},
        "share_above": round(float(s[has].mean()), 3),
        "flip_hourly_abs_change_median(%)": round(float(np.nanmedian(np.abs(np.diff(S["flip_week"].to_numpy())) * 100)), 3)}
    print("flip dist", res["flip_distance"], flush=True)

    def h2(sv, mask, extra=None, ys=(("f_rg1_4h", 0.05), ("y1", 0.10))):
        out = {}
        for yc, E in ys:
            X = np.c_[np.ones(mask.sum()), sv[mask]] if extra is None else np.c_[np.ones(mask.sum()), sv[mask], extra[mask]]
            out[yc] = coef(X, S[yc].to_numpy()[mask], day[mask], 1, +1, E, two_sided=True)
        return out
    G = {"main": h2(s, has)}
    G["ctrl_absdist_quintile"] = h2(s, has, extra=np.nan_to_num(qdum(np.abs(np.where(has, dist, np.nan)))[:, 1:], nan=np.nan))
    G["absdist_gt_1pct"] = h2(s, has & (np.abs(dist) > 0.01))
    G["ctrl_rg1_12h"] = h2(s, has, extra=S["rg1_12h"].to_numpy()[:, None])
    G["grid4h"] = h2(s, has & (S["hour"].to_numpy() % (4 * H_MS) == 0))
    fall = flipabs("flip_all"); hall = np.isfinite(fall); G["flip_all"] = h2((S["p_te"].to_numpy() > fall).astype(float), hall)
    G["ctrl_shuffled_s"] = h2(np.where(has, np.random.default_rng(SEED).permutation(s), np.nan), has)
    fl = flipabs("flip_week_lag1d", "idx_lag1d"); hl = np.isfinite(fl)
    G["ctrl_flip_lag1d"] = h2((S["p_te"].to_numpy() > fl).astype(float), hl)
    G["cell_means"] = {f"{'위' if a else '아래'}": {"n": int((has & (s == a)).sum()),
                                                   "f_rg1_4h": round(float(S["f_rg1_4h"][has & (s == a)].mean()), 4),
                                                   "y1": round(float(S["y1"][has & (s == a)].mean()), 4)} for a in (0, 1)}
    res["H2a"] = G
    print("H2a", {k: {y: (v[y]["est"], v[y]["ci"], v[y]["verdict"]) for y in v} for k, v in G.items() if k != "cell_means"}, flush=True)

    # ── H2b ──
    prev = S.set_index("hour")[["flip_week", "idx"]].reindex(S["hour"] - H_MS).to_numpy()
    fp = prev[:, 0] * prev[:, 1]; okb = np.isfinite(fp)
    cross = (np.sign(S["p_m1h"].to_numpy() - fp) != np.sign(S["p_te"].to_numpy() - fp)).astype(float)
    r1 = np.abs(np.log(S["p_te"] / S["p_m1h"]).to_numpy())
    Dq = qdum(np.where(okb, r1, np.nan)); dRG = (S["f_rg1_4h"] - S["rg1_4h"]).to_numpy()
    Xb = np.c_[Dq, cross][okb]
    res["H2b"] = {"dRG": coef(Xb, dRG[okb], day[okb], 5, +1, 0.05, two_sided=True),
                  "y1": coef(Xb, y1[okb], day[okb], 5, +1, 0.10, two_sided=True),
                  "n_cross": int(cross[okb].sum()), "n_eligible": int(okb.sum()),
                  "cross_abs_r1h_median_bp": round(float(np.median(r1[okb & (cross == 1)]) * 1e4), 1),
                  "noncross_abs_r1h_median_bp": round(float(np.median(r1[okb & (cross == 0)]) * 1e4), 1)}
    print("H2b", res["H2b"], flush=True)

    # ── 사후(판정 제외) ──
    hod = R.dummies(((S["hour"] // H_MS) % 24).to_numpy(), 24)
    gc = S["gex_conv_week"].to_numpy(); gcp = pct_rank(gc)
    res["post_hoc"] = {
        "H1b_hour_fe": coef(np.c_[hod, m, reg, reg * m], y1, day, 26, +1, 0.10),
        "H1b_M_alone_hour_fe": coef(np.c_[hod, m], y1, day, 24, +1, 0.10),
        "H1b_M_ex0dte": h1b(S.assign(Mp=pct_rank(S["M_ex1d"].to_numpy()))),
        "H1b_M_ex0dte_alone": coef(np.c_[one, pct_rank(S["M_ex1d"].to_numpy())], y1, day, 1, +1, 0.10),
        "H2a_s_vs_sign_conv_gex_agree": round(float(((gc > 0) == (s > 0))[has].mean()), 3),
        "H2a_y1_ctrl_conv_gex_pct": coef(np.c_[np.ones(has.sum()), s[has], gcp[has]], y1[has], day[has], 1, +1, 0.10, two_sided=True),
        "conv_gex_pct_alone_y1": coef(np.c_[one, gcp], y1, day, 1, +1, 0.10, two_sided=True),
    }
    print("post_hoc", {k: (v["est"], v["ci"], v["verdict"]) if isinstance(v, dict) else v for k, v in res["post_hoc"].items()}, flush=True)
    print("slope_by_regime", {k: (v["est"], v["ci"]) for k, v in H["slope_by_regime"].items()})

    S.to_parquet(OUT / "hourly.parquet"); Q.to_parquet(OUT / "h1c_pairs.parquet")
    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float))
    print("wrote", OUT / "results.json")


def selftest():
    # 플립 보간: [−1,−1,3] → 1.0 과 1.1 사이 1/4 지점 · 두 전환 중 지금(1)에 가까운 쪽 · 뒤집은 관행(−g)도 같은 자리
    g3 = np.array([0.9, 1.0, 1.1])
    assert abs(flip_of(np.array([-1.0, -1.0, 3.0]), g3) - 1.025) < 1e-12
    pr2 = np.array([1.0, -1.0, 3.0])                      # 전환 0.95 · 1.025 → 지금에 가까운 1.025
    assert abs(flip_of(pr2, g3) - 1.025) < 1e-12 and flip_of(pr2, g3) == flip_of(-pr2, g3)
    assert np.isnan(flip_of(np.array([1.0, 2.0, 3.0]), g3))
    # 시점 경계: te = 다음 5분 경계 · 지금가 = 스냅샷 이하에 닫힌 1분봉
    h = ms("2026-09-01 10:00")
    assert known_at(h) == h and known_at(h + 5_000) == h + M5
    ends = np.array([h - 60_000, h, h + 60_000]); v = np.array([1.0, 2.0, 3.0])
    assert last_closed(ends, v, h + 5_000) == 2.0 and last_closed(ends, v, h - 1) == 1.0
    # 5분 패널(every=1): te 이후 가격을 바꿔도 지표 불변 · te 이하를 바꿔도 결과 불변
    rng = np.random.default_rng(0); idx = np.arange(R.D0 + M5, R.D0 + M5 * 3000, M5, dtype="int64")
    lp = pd.Series(np.cumsum(rng.normal(0, 1e-3, len(idx))), index=idx); k = 2300; t = lp.index[k]
    b0 = R.panel(lp, 1).iloc[k]
    fut = lp.copy(); fut[fut.index > t] += rng.normal(0, 1e-2, int((fut.index > t).sum()))
    past = lp.copy(); past[past.index <= t] += rng.normal(0, 1e-2, int((past.index <= t).sum()))
    a, b = R.panel(fut, 1).iloc[k], R.panel(past, 1).iloc[k]
    assert all(a[c] == b0[c] for c in ("rg1_12h", "rg1_4h", "rv1h", "rv24h", "rv7d"))
    assert all(abs(b[c] - b0[c]) < 1e-12 for c in ("f_rg1_4h", "rv_next4h"))
    print("selftest OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true")
    selftest() if ap.parse_args().selftest else main()
