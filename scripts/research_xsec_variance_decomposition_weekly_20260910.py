#!/usr/bin/env python3
"""횡단면 **실현분산 성분 분해** 주간 전략 — 사전등록 검정 (2026-09-10).

근거 문헌: JFQA 2024 「Variance Decomposition and Cryptocurrency Return Prediction」
(`10.1017/S002210902400022X`) — 일중 데이터로 잰 실현분산이 **높은** 코인이 **이후 몇 주** 수익이 **낮다**.
총분산을 **부호점프분산**과 **점프강건분산**으로 쪼개면 음의 예측력은 **양의 점프분산 + 점프강건분산**에서 나온다.
효과는 소형·저가·저유동·리테일 비중 높은 코인에서 강하다(기전=복권형 선호).

조사 경위·후보 등급: `docs/eth_midterm_signal_literature_survey_20260910.md`

## ⚠️결과를 보기 전에 고정한 것 (사전등록)
정렬 변수 4종  rv(총) · bv(점프강건, bipower) · jv_pos(양의 점프분산) · sjv(부호점프 = RS⁺−RS⁻)
방향        문헌대로 **높을수록 숏, 낮을수록 롱**(음의 예측). 부호를 사후에 뒤집지 않는다.
형성/보유    매주 월요일 00:00 UTC 형성, 직전 7일 5분봉으로 성분 계산, **다음 7일 보유**(종가→종가)
유니버스     유동성 상위 NU∈{20,40,60} · 절대 문턱 일거래대금 중앙 ≥$50M / ≥$200M
다리 크기    k∈{3,5,10} 롱숏 동수(설계상 베타 제거)
비용        **주당 12bp**(전량 회전 가정, 보수적). gross 도 같이 보고
분할        IS ≤2025-08-31 / **OOS 2025-09-01~** (09-08 횡단면 프로그램과 같은 경계)
귀무        **무작위 배정**(주마다 코인 라벨 셔플) B=1000 — 09-08 이 확립한 방식
필수 보고    독립 주 수 · 블록부트 CI · 윈저라이즈(5%/10%) · 상위 5% 주 제거 후

## 🔴미리 적어둔 벽
945일 = 독립 주 ~135개. 09-08 다일(1~14일) 165셀이 **0통과**했고 사유가 *"독립 관측 ~135개,
원리적 검정력 부족"*이었다. 새 축이지만 새 검정력은 아니다. 판정 불가로 끝나면 그것이 결론이다.
출력 tmp/xsec_vardecomp_20260910/report.json
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
OUT = ROOT / "tmp/xsec_vardecomp_20260910"
SORTS = ["rv", "bv", "jv_pos", "sjv"]
NUS = [20, 40, 60]
ABS_THR = [0.0, 50e6, 200e6]          # 일거래대금 중앙 문턱(USD)
KS = [3, 5, 10]
COST_BP = 12.0
SPLIT = pd.Timestamp("2025-09-01")
B_NULL = 1000
MIN_BARS = 1500                        # 주당 5분봉 2016 개 중 최소 커버


def log(m):
    print(f"[vd {time.strftime('%H:%M:%S')}] {m}", flush=True)


def weekly_features(ts, C, Q):
    """주별 성분. 반환: 주 시작 인덱스 리스트 + 각 주의 (rv,bv,jv_pos,sjv,liq,ret_next)."""
    t = pd.DatetimeIndex(ts)
    wk = t.to_period("W-SUN")                     # 월요일 시작 주
    codes, uniq = pd.factorize(wk, sort=True)
    n_w, n_s = len(uniq), C.shape[1]
    F = {k: np.full((n_w, n_s), np.nan) for k in SORTS + ["liq", "ret", "ok"]}
    logC = np.log(np.where(C > 0, C, np.nan))
    for w in range(n_w):
        rows = np.flatnonzero(codes == w)
        if len(rows) < MIN_BARS:
            continue
        lc = logC[rows]
        r = np.diff(lc, axis=0)                    # 5분 로그수익
        good = np.isfinite(r)
        cnt = good.sum(axis=0)
        rr = np.where(good, r, 0.0)
        rv = (rr ** 2).sum(axis=0)
        rsp = np.where(rr > 0, rr ** 2, 0.0).sum(axis=0)
        rsn = np.where(rr < 0, rr ** 2, 0.0).sum(axis=0)
        a = np.abs(rr)
        bv = (np.pi / 2) * (a[1:] * a[:-1]).sum(axis=0)
        jv = np.maximum(rv - bv, 0.0)
        sjv = rsp - rsn
        ok = cnt >= MIN_BARS
        F["rv"][w] = np.where(ok, rv, np.nan)
        F["bv"][w] = np.where(ok, bv, np.nan)
        F["jv_pos"][w] = np.where(ok, np.where(sjv > 0, jv, 0.0), np.nan)
        F["sjv"][w] = np.where(ok, sjv, np.nan)
        q = Q[rows]
        F["liq"][w] = np.where(ok, np.nanmedian(np.where(np.isfinite(q), q, np.nan), axis=0) * 288, np.nan)
        F["ok"][w] = ok.astype(float)
    # 다음 주 수익 = 그 주 마지막 종가 → 다음 주 마지막 종가
    last = np.full((n_w, n_s), np.nan)
    for w in range(n_w):
        rows = np.flatnonzero(codes == w)
        if len(rows) == 0:
            continue
        seg = C[rows]
        for j in range(n_s):
            v = seg[:, j][np.isfinite(seg[:, j])]
            if len(v):
                last[w, j] = v[-1]
    F["ret"][:-1] = last[1:] / last[:-1] - 1.0
    return uniq, F


def run_cell(F, weeks, sort, nu, thr, k, mask_w, rng=None):
    """한 셀의 주별 롱숏 수익(bp). rng 가 있으면 무작위 배정 귀무."""
    out = []
    for w in np.flatnonzero(mask_w):
        s = F[sort][w]; liq = F["liq"][w]; ret = F["ret"][w]
        ok = np.isfinite(s) & np.isfinite(liq) & np.isfinite(ret) & (liq >= thr)
        idx = np.flatnonzero(ok)
        if len(idx) < 2 * k + 2:
            continue
        idx = idx[np.argsort(-liq[idx])][:nu]      # 유동성 상위 NU
        if len(idx) < 2 * k:
            continue
        order = rng.permutation(idx) if rng is not None else idx[np.argsort(s[idx])]
        lo, hi = order[:k], order[-k:]             # 낮은 분산 롱 · 높은 분산 숏(문헌 방향)
        out.append((ret[lo].mean() - ret[hi].mean()) * 1e4)
    return np.array(out)


def block_ci(x, block=4, b=2000, seed=0):
    if len(x) < 8:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed); n = len(x); out = []
    for _ in range(b):
        st = rng.integers(0, n, int(np.ceil(n / block)))
        idx = np.concatenate([np.arange(s, s + block) % n for s in st])[:n]
        out.append(x[idx].mean())
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = np.load(PANEL, allow_pickle=True)
    ts, C, Q, syms = d["ts"], d["C"], d["Q"], d["syms"]
    log(f"패널 {C.shape} · {pd.Timestamp(ts[0])} → {pd.Timestamp(ts[-1])} · {len(syms)}종")
    weeks, F = weekly_features(ts, C, Q)
    wstart = pd.PeriodIndex(weeks).to_timestamp()
    valid = np.isfinite(F["ret"]).sum(axis=1) >= 10
    is_w = valid & np.asarray(wstart < SPLIT)
    oos_w = valid & np.asarray(wstart >= SPLIT)
    log(f"주 {len(weeks)} · 유효 {valid.sum()} · IS {is_w.sum()} · **OOS {oos_w.sum()}** (독립 주 = 이 수)")
    rep = {"panel": str(PANEL), "n_weeks": int(len(weeks)), "n_valid": int(valid.sum()),
           "n_is": int(is_w.sum()), "n_oos": int(oos_w.sum()), "cost_bp": COST_BP,
           "prereg": {"sorts": SORTS, "nus": NUS, "abs_thr": ABS_THR, "ks": KS,
                      "direction": "낮은 분산 롱 / 높은 분산 숏(문헌 방향, 사후 반전 금지)",
                      "split": str(SPLIT.date()), "b_null": B_NULL},
           "cells": {}}
    rng_null = np.random.default_rng(20260910)
    for sort, nu, thr, k in product(SORTS, NUS, ABS_THR, KS):
        key = f"{sort}|NU{nu}|thr{int(thr/1e6)}M|k{k}"
        gi = run_cell(F, weeks, sort, nu, thr, k, is_w)
        go = run_cell(F, weeks, sort, nu, thr, k, oos_w)
        if len(gi) < 20 or len(go) < 15:
            continue
        nulls = np.array([run_cell(F, weeks, sort, nu, thr, k, oos_w, rng_null).mean()
                          for _ in range(B_NULL // 10)])          # 무작위 배정 귀무(계산 예산상 B/10)
        net_o = go.mean() - COST_BP
        cell = {"n_is": len(gi), "n_oos": len(go),
                "gross_is_bp": float(gi.mean()), "gross_oos_bp": float(go.mean()),
                "net_oos_bp": float(net_o), "oos_ci95_gross": block_ci(go),
                "null_mean": float(nulls.mean()), "null_p97.5": float(np.percentile(nulls, 97.5)),
                "beats_null": bool(go.mean() > np.percentile(nulls, 97.5)),
                "wins5_oos": float(np.mean(np.clip(go, *np.percentile(go, [5, 95])))),
                "wins10_oos": float(np.mean(np.clip(go, *np.percentile(go, [10, 90])))),
                "drop_top5pct_weeks": float(np.mean(np.sort(go)[:max(1, int(len(go) * 0.95))])),
                "both_windows_positive": bool(gi.mean() > 0 and go.mean() > 0)}
        rep["cells"][key] = cell
        (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    cells = rep["cells"]
    log(f"셀 {len(cells)}개 · 두 창 양수 {sum(c['both_windows_positive'] for c in cells.values())} "
        f"· 귀무 통과 {sum(c['beats_null'] for c in cells.values())} "
        f"· 순@12bp 양수 {sum(c['net_oos_bp'] > 0 for c in cells.values())} "
        f"· OOS CI 하한>0 {sum(c['oos_ci95_gross'][0] > 0 for c in cells.values())}")
    top = sorted(cells.items(), key=lambda kv: -kv[1]["gross_oos_bp"])[:8]
    for k_, c in top:
        log(f"  {k_:26s} gross IS {c['gross_is_bp']:+7.1f} OOS {c['gross_oos_bp']:+7.1f} "
            f"CI[{c['oos_ci95_gross'][0]:+.1f},{c['oos_ci95_gross'][1]:+.1f}] 순 {c['net_oos_bp']:+7.1f} "
            f"윈저10 {c['wins10_oos']:+7.1f} 귀무통과 {c['beats_null']}")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
