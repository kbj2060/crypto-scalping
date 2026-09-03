#!/usr/bin/env python3
"""v2 상위후보 **견고성 감사** -- 독립일수 · 일단위 군집 CI · 학습셋 부트스트랩.

## 왜

`screen_eth_evidence8_econ_label_v2_20260904.py`가 상위10% 선별 평균 bp 양수 조합 25/64,
무작위귀무 p<0.05 6개를 냈다. 그런데 그 뒤 돌린 5시드 검증이 **비트 단위로 동일한 결과**를
냈다 -- `HistGradientBoosting`은 n<200,000에서 `random_state`가 비닝 서브샘플에만 쓰여
**완전 결정론적**이기 때문이다(조용한 no-op). 즉 모델 분산은 측정된 적이 없다.

이 설정에서 분산의 진짜 원천은 **데이터**다. 세 가지를 잰다:

  ①독립일수  상위10%(k=22~90건)가 며칠에 몰려 있나. 5일이면 평균 bp는 무의미하다.
  ②일단위 군집 부트스트랩 CI  행 단위가 아니라 **날짜 블록**을 재추출한다. 오늘
    진입모델을 죽인 검정이고(행단위 p=0.003이 일단위로는 하한 음수), 이 저장소 표준이다.
  ③학습셋 부트스트랩  TRAIN을 복원추출해 재적합 -> 상위10% 재선별. 모델 분산의 대체 측정.

⚠️TRAIN/VAL만. OOS·HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def _load(n, r):
    s = importlib.util.spec_from_file_location(n, ROOT / r)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


_pf = _load("pf_rob", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

GAP, COST_BP, TOPQ = 12, 10.0, 0.10
SEED, B_DAY, B_TRAIN = 20260904, 2000, 30
# v2 상위 후보만 -- (신호, horizon, 셀, 라벨)
# ⭐64조합 전수. 앞서 상위 6개만 검정한 건 사후선택이었다 -- 진짜 생존율을 알려면 전부 돌려야 한다.
SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELLS = [(3.0, 1.5, 0.1), (4.0, 1.0, 0.1)]
LABS = ["L1_sign", "L2_reg", "L3_tail", "L4_exclmid"]
CAND = [(s_, h_, c_, l_) for s_, h_ in SIGNALS.items() for c_ in CELLS for l_ in LABS]
OUT = ROOT / "data/research/eth_evidence8_econ_topdecile_robustness_20260904/report_full64.json"


def log(m): print(f"[rob] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def build(SIGNAL, HORIZON, sig, eth, long):
    bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    S = sig[["timestamp", bcol, tcol]].copy()
    if S["timestamp"].dt.tz is not None:
        S["timestamp"] = S["timestamp"].dt.tz_localize(None)
    S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
    S = S.loc[S["pos"] >= 0]
    fb = np.zeros(n, bool); ft = np.zeros(n, bool)
    fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
    ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
    kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)
    lts = long["timestamp"].to_numpy()
    if getattr(long["timestamp"].dt, "tz", None) is not None:
        lts = long["timestamp"].dt.tz_localize(None).to_numpy()
    lpos = np.array([pos_of.get(np.datetime64(t), -1) for t in lts])
    is_down = long["is_downside"].to_numpy().astype(bool)
    keep = (lpos >= 0) & (lpos + 1 + HORIZON < n)
    keep &= np.where(is_down, kb[np.clip(lpos, 0, n - 1)], kt[np.clip(lpos, 0, n - 1)])
    D = long.loc[keep].reset_index(drop=True)
    ii = lpos[keep]
    sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
    entry = o[ii + 1]
    H = np.stack([h[i + 1:i + 1 + HORIZON] for i in ii])
    L = np.stack([l[i + 1:i + 1 + HORIZON] for i in ii])
    C = np.stack([c[i + 1:i + 1 + HORIZON] for i in ii])
    return D, ii, sg, entry, H, L, C, kl


def fit_predict(lab, X, net, fit, va, rs):
    from sklearn.ensemble import (HistGradientBoostingClassifier,
                                  HistGradientBoostingRegressor)
    if lab == "L2_reg":
        m = HistGradientBoostingRegressor(random_state=rs, max_iter=300, learning_rate=0.05)
        m.fit(X[fit], net[fit]); return m.predict(X[va])
    if lab == "L1_sign":
        y = (net > 0).astype(int); f2 = fit
    elif lab == "L3_tail":
        y = (net < np.quantile(net[fit], 0.25)).astype(int); f2 = fit
    else:
        lo, hi = np.quantile(net[fit], [0.25, 0.75])
        y = np.where(net >= hi, 1, np.where(net <= lo, 0, -1)); f2 = fit & (y >= 0)
    if len(np.unique(y[f2])) < 2:
        return None
    m = HistGradientBoostingClassifier(random_state=rs, max_iter=300, learning_rate=0.05)
    m.fit(X[f2], y[f2])
    p = m.predict_proba(X[va])[:, 1]
    return -p if lab == "L3_tail" else p


def main() -> int:
    t0 = time.time()
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)
    rng = np.random.default_rng(SEED)
    rep = {}

    print(f"\n{'신호':>24s}{'라벨':>11s}{'k':>5s}{'독립일':>7s}{'최대일비중':>10s}"
          f"{'상위bp':>9s}{'일CI하한':>10s}{'일CI상한':>10s}{'학습부트중앙':>12s}{'양수비':>7s}")
    print("-" * 106)

    cache = {}
    for SIGNAL, HZ, cell, lab in CAND:
        if SIGNAL not in cache:
            cache[SIGNAL] = build(SIGNAL, HZ, sig, eth, long)
        D, ii, sg, entry, H, L, C, kl = cache[SIGNAL]
        X = D[[c_ for c_ in TIER0 if c_ in D.columns]].to_numpy(float)
        a_ = D["atr"].to_numpy(float)
        pn, _ = sim_exit(entry, a_, sg, H, L, C, *cell)
        net = pn * 1e4 - COST_BP
        split = D["split"].to_numpy()
        tr, va = split == "TRAIN", split == "VAL"
        k = max(10, int(round(va.sum() * TOPQ)))
        days = pd.to_datetime(D["timestamp"]).dt.floor("D").to_numpy()

        pred = fit_predict(lab, X, net, tr, va, SEED)
        if pred is None or va.sum() < 100 or tr.sum() < 200:
            continue
        nv, dv = net[va], days[va]
        top = np.argsort(-pred)[:k]
        top_bp, tdays = float(nv[top].mean()), dv[top]
        uniq = np.unique(tdays)
        maxshare = float(pd.Series(tdays).value_counts().iloc[0] / k)

        # ②일 단위 군집 부트스트랩 -- 날짜 블록을 복원추출
        by_day = {d: nv[top][tdays == d] for d in uniq}
        boot = np.empty(B_DAY)
        for b in range(B_DAY):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            boot[b] = np.concatenate([by_day[d] for d in pick]).mean()
        lo_, hi_ = np.percentile(boot, [2.5, 97.5])

        # ③학습셋 부트스트랩 -- **일CI 통과분에만**(비용 통제)
        tri = np.flatnonzero(tr)
        bt = []
        for b in range(B_TRAIN if lo_ > 0 else 0):
            samp = rng.choice(tri, size=len(tri), replace=True)
            m = np.zeros(len(D), bool); m[np.unique(samp)] = True
            p2 = fit_predict(lab, X, net, m, va, SEED + b)
            if p2 is not None:
                bt.append(float(nv[np.argsort(-p2)[:k]].mean()))
        if bt:
            bt = np.array(bt)
            med, pos_frac = float(np.median(bt)), float((bt > 0).mean())
        else:
            med, pos_frac = float("nan"), float("nan")

        print(f"{SIGNAL[:23]:>24s}{lab:>11s}{k:5d}{len(uniq):7d}{maxshare:10.2f}"
              f"{top_bp:9.2f}{lo_:10.2f}{hi_:10.2f}{med:12.2f}{pos_frac:7.2f}"
              f"{'  ⭐' if lo_ > 0 else ''}")
        rep[f"{SIGNAL}|{cell[0]}/{cell[1]}|{lab}"] = {
            "k": k, "independent_days": int(len(uniq)), "max_day_share": maxshare,
            "top_mean_bp": top_bp, "day_ci_lo": float(lo_), "day_ci_hi": float(hi_),
            "train_boot_median": med, "train_boot_positive_frac": pos_frac,
            "n_val": int(va.sum())}

    surv = [k for k, v in rep.items() if v["day_ci_lo"] > 0]
    print()
    log(f"⭐일단위 군집 CI 하한이 0을 넘는 후보: **{len(surv)}/{len(rep)}**")
    for s_ in surv:
        log(f"    {s_}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"candidates": rep, "n_survived_day_ci": len(surv),
                               "B_day": B_DAY, "B_train": B_TRAIN, "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
