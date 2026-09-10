#!/usr/bin/env python3
"""횡단면 **특이변동성 · 유동성** 후보 — 겹침 트랜치 사전등록 검정 (2026-09-10). 모델 없음.

[문헌조사](docs/eth_midterm_signal_literature_survey_20260910.md) 의 C·B 등급 후보. A등급(실현분산 분해)은
[주간 비겹침 검정에서 기각](docs/eth_xsec_variance_decomposition_weekly_20260910.md) — **독립 주 48개**가 벽이었다.
이번엔 그 벽을 **설계로 우회할 수 있는지**까지 같이 본다.

## 방향은 문헌이 정한다 (초록 확인 후 고정, 사후 반전 금지)
`ivol`   RIBAF 2020 `10.1016/j.ribaf.2020.101252` — *"idiosyncratic volatility is **positively** related to
         the expected returns"* ⇒ **높은 IVOL 롱 / 낮은 IVOL 숏**. ⚠️주식시장 IVOL 퍼즐과 **반대**이고,
         A등급 JFQA(높은 분산 → 낮은 수익)와도 **반대** — 둘은 다른 변수(총분산 vs 잔차분산)다.
`liqvol` FRL 2021 `10.1016/J.FRL.2021.102031` — *"positive relation between the volatility of liquidity and
         expected returns"* ⇒ **높은 유동성변동성 롱**. ⚠️논문은 **시총 상위 5종**만 썼다(여기는 60종).
`liqlvl` 같은 논문 — *"when liquidity is low, expected returns are high"* ⇒ **낮은 유동성 롱**.
`rv`,`bv` A등급 변수를 **하네스 대조군**으로 같이 돌린다(같은 기계에서 A 결과가 재현되는지 확인).

## 겹침 트랜치 — 검정력 우회 시도
매일 형성 · **7일 보유** · 7개 트랜치 평균 ⇒ 일별 수익 계열(n≈900일). 회전율은 하루 1/7 이라
**비용은 주간 비겹침과 동일**(주당 12bp). 단 ⚠️**독립 정보는 늘지 않는다** — CI 는 **블록 14일** 부트로 잡는다.
정렬 창은 문헌 관례대로 **직전 30일 일간 수익**(ivol 은 등가중 시장 대비 잔차).

## 판정 기준(결과 전 고정)
1차 **주간환산 순수익(비용 12bp 차감) 의 블록부트 95% CI 하한 > 0**
보조 무작위 배정 귀무 · IS/OOS 부호 일관 · 윈저10 · 상위 5% 일 제거
출력 tmp/xsec_ivol_liq_20260910/report.json
"""
from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = Path("/home/kbj20/crypto-scalping/tmp/xsec_perp_screen_20260908/panel.npz")
OUT = ROOT / "tmp/xsec_ivol_liq_20260910"
# 정렬변수: (이름, 방향) 방향 +1 = 높을수록 롱, -1 = 낮을수록 롱
SORTS = {"ivol": +1, "liqvol": +1, "liqlvl": -1, "rv": -1, "bv": -1}
FORM_D, HOLD_D = 30, 7
NUS = [20, 40]
ABS_THR = [0.0, 50e6]
KS = [3, 5, 10]
COST_WK_BP = 12.0
SPLIT = pd.Timestamp("2025-09-01")
BLOCK_D = 14


def log(m):
    print(f"[il {time.strftime('%H:%M:%S')}] {m}", flush=True)


def daily_from_panel():
    d = np.load(PANEL, allow_pickle=True)
    ts, C, Q = pd.DatetimeIndex(d["ts"]), d["C"], d["Q"]
    day = ts.floor("D")
    codes, uniq = pd.factorize(day, sort=True)
    n_d, n_s = len(uniq), C.shape[1]
    close = np.full((n_d, n_s), np.nan)
    dv = np.full((n_d, n_s), np.nan)
    for i in range(n_d):
        rows = np.flatnonzero(codes == i)
        seg = C[rows]
        for j in range(n_s):
            v = seg[:, j][np.isfinite(seg[:, j])]
            if len(v):
                close[i, j] = v[-1]
        q = Q[rows]
        dv[i] = np.nansum(np.where(np.isfinite(q), q, 0.0), axis=0)
    return pd.DatetimeIndex(uniq), close, dv


def build_sorts(days, close, dv):
    n_d, n_s = close.shape
    r = np.full((n_d, n_s), np.nan)
    r[1:] = close[1:] / close[:-1] - 1.0
    ldv = np.log(np.where(dv > 0, dv, np.nan))
    S = {k: np.full((n_d, n_s), np.nan) for k in SORTS}
    for t in range(FORM_D, n_d):
        w = slice(t - FORM_D + 1, t + 1)
        R = r[w]
        ok = np.isfinite(R).sum(axis=0) >= FORM_D - 3
        mkt = np.nanmean(np.where(np.isfinite(R), R, np.nan), axis=1)     # 등가중 시장
        good = np.isfinite(mkt)
        for j in np.flatnonzero(ok):
            y = R[:, j]; m = good & np.isfinite(y)
            if m.sum() < FORM_D - 3:
                continue
            x = mkt[m]; yy = y[m]
            b = np.cov(x, yy, ddof=1)[0, 1] / max(np.var(x, ddof=1), 1e-18)
            resid = yy - b * x
            S["ivol"][t, j] = resid.std(ddof=1)
            S["rv"][t, j] = np.nansum(yy ** 2)
            a = np.abs(yy)
            S["bv"][t, j] = (np.pi / 2) * np.nansum(a[1:] * a[:-1])
        L = ldv[w]
        okl = np.isfinite(L).sum(axis=0) >= FORM_D - 3
        S["liqvol"][t] = np.where(okl, np.nanstd(L, axis=0), np.nan)
        S["liqlvl"][t] = np.where(okl, np.nanmean(L, axis=0), np.nan)
    liq = np.full((n_d, n_s), np.nan)
    for t in range(FORM_D, n_d):
        liq[t] = np.nanmedian(dv[t - FORM_D + 1:t + 1], axis=0)
    fwd = np.full((n_d, n_s), np.nan)
    fwd[:-HOLD_D] = close[HOLD_D:] / close[:-HOLD_D] - 1.0                # 7일 보유
    return S, liq, fwd


def tranche_series(S, liq, fwd, sort, sgn, nu, thr, k, t_ok, rng=None):
    """매일 형성·7일 보유 → 일별 트랜치 수익(bp). 하루 1/7 회전이라 주간 비용과 동일."""
    out = []
    s_all = S[sort]
    for t in np.flatnonzero(t_ok):
        s = s_all[t]; lq = liq[t]; f = fwd[t]
        ok = np.isfinite(s) & np.isfinite(lq) & np.isfinite(f) & (lq >= thr)
        idx = np.flatnonzero(ok)
        if len(idx) < 2 * k + 2:
            continue
        idx = idx[np.argsort(-lq[idx])][:nu]
        if len(idx) < 2 * k:
            continue
        order = rng.permutation(idx) if rng is not None else idx[np.argsort(sgn * s[idx])]
        short_leg, long_leg = order[:k], order[-k:]      # sgn 반영 후 큰 쪽이 롱
        out.append(((f[long_leg].mean() - f[short_leg].mean()) / HOLD_D) * 1e4)
    return np.array(out)


def block_ci(x, block=BLOCK_D, b=2000, seed=0):
    if len(x) < 30:
        return [float("nan")] * 2
    rng = np.random.default_rng(seed); n = len(x); o = []
    for _ in range(b):
        st = rng.integers(0, n, int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) % n for s in st])[:n]
        o.append(x[idx].mean())
    return [float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    days, close, dv = daily_from_panel()
    log(f"일봉 {close.shape} · {days[0].date()} → {days[-1].date()}")
    S, liq, fwd = build_sorts(days, close, dv)
    base_ok = np.isfinite(fwd).sum(axis=1) >= 10
    is_d = base_ok & np.asarray(days < SPLIT)
    oos_d = base_ok & np.asarray(days >= SPLIT)
    n_ind_wk = int(oos_d.sum() / 7)
    log(f"유효일 IS {is_d.sum()} · OOS {oos_d.sum()} (독립 주 환산 ≈ {n_ind_wk})")
    rep = {"form_days": FORM_D, "hold_days": HOLD_D, "cost_week_bp": COST_WK_BP, "block_days": BLOCK_D,
           "directions": {k: ("높을수록 롱" if v > 0 else "낮을수록 롱") for k, v in SORTS.items()},
           "n_is_days": int(is_d.sum()), "n_oos_days": int(oos_d.sum()),
           "independent_weeks_oos": n_ind_wk, "cells": {}}
    rng_null = np.random.default_rng(20260910)
    for sort, nu, thr, k in product(SORTS, NUS, ABS_THR, KS):
        sgn = SORTS[sort]
        gi = tranche_series(S, liq, fwd, sort, sgn, nu, thr, k, is_d)
        go = tranche_series(S, liq, fwd, sort, sgn, nu, thr, k, oos_d)
        if len(gi) < 60 or len(go) < 40:
            continue
        wk_o = go * 7                                   # 주간 환산
        net = wk_o - COST_WK_BP
        nulls = np.array([tranche_series(S, liq, fwd, sort, sgn, nu, thr, k, oos_d, rng_null).mean() * 7
                          for _ in range(60)])
        cell = {"n_is_d": len(gi), "n_oos_d": len(go),
                "gross_is_wk_bp": float(gi.mean() * 7), "gross_oos_wk_bp": float(wk_o.mean()),
                "net_oos_wk_bp": float(net.mean()), "net_ci95": block_ci(net),
                "gross_ci95": block_ci(wk_o),
                "null_p97.5": float(np.percentile(nulls, 97.5)),
                "beats_null": bool(wk_o.mean() > np.percentile(nulls, 97.5)),
                "wins10": float(np.mean(np.clip(wk_o, *np.percentile(wk_o, [10, 90])))),
                "both_windows_pos": bool(gi.mean() > 0 and wk_o.mean() > 0)}
        rep["cells"][f"{sort}|NU{nu}|thr{int(thr/1e6)}M|k{k}"] = cell
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    c = rep["cells"]
    log(f"셀 {len(c)} · **순 CI 하한>0 {sum(v['net_ci95'][0] > 0 for v in c.values())}** "
        f"· 총 CI 하한>0 {sum(v['gross_ci95'][0] > 0 for v in c.values())} "
        f"· 귀무통과 {sum(v['beats_null'] for v in c.values())}(우연 {0.025*len(c):.1f}) "
        f"· 두 창 양수 {sum(v['both_windows_pos'] for v in c.values())}")
    for sort in SORTS:
        sub = {k: v for k, v in c.items() if k.startswith(sort + "|")}
        if not sub:
            continue
        b = max(sub.items(), key=lambda kv: kv[1]["net_oos_wk_bp"])
        log(f"  {sort:7s} 최고 {b[0]:24s} IS {b[1]['gross_is_wk_bp']:+7.1f} OOS총 {b[1]['gross_oos_wk_bp']:+7.1f} "
            f"순 {b[1]['net_oos_wk_bp']:+7.1f} CI[{b[1]['net_ci95'][0]:+.1f},{b[1]['net_ci95'][1]:+.1f}] "
            f"윈저10 {b[1]['wins10']:+7.1f}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
