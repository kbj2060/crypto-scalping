#!/usr/bin/env python3
"""증거신호 8종 경제성 회귀 -- **통합 풀 학습** (목표: 8종 전부 관문 통과).

## 왜 통합인가

8종을 각각 따로 학습시킨 세 번의 실험에서 통과는 1~2종뿐이었다(fib, taker). 원인은 표본이다:
신호별 VAL이 221~895건뿐이라 상위10%가 **11~90건**이고, 지정가 진입은 체결률 0.37~0.74로
그마저 더 줄여 독립일이 9~25일까지 떨어졌다. 오늘 진입모델을 죽인 게 정확히 이 문제였다
(독립 일수 42~45일).

⇒ **8종을 하나의 풀로 합치고 신호 정체성을 원핫 피쳐로 준다.**
  · 표본이 8배 -> 통계적 파워 확보
  · 모델이 "어떤 신호의 어떤 상황이 수익성 있나"를 신호 간 정보 공유로 학습
  · 라벨은 각 신호가 **자기 HORIZON**으로 계산하므로 bp 단위로 비교 가능

## 설계

    모집단  8종 인과 첫 발동(cluster_dedup 금지) 합집합
    라벨    o[i+1] 진입 -> 자기 HORIZON까지 트레일링 -> 비용 차감 후 net_bp (회귀 타깃)
    피쳐    Tier0 23 + 신호 원핫 8 (+ 선택적으로 horizon)
    모델    TabPFN 회귀 (기준선: HGB 회귀)
    평가    **신호별로** 상위10%를 뽑아 일단위 클러스터 t -- "8종 모두 통과"를 직접 잰다

⚠️TRAIN/VAL만. OOS·HOLDOUT 미터치.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
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


_pf = _load("pf_pool", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
OUT = ROOT / "data/research/eth_evidence8_econ_pooled_reg_20260904/report.json"


def log(m): print(f"[pool] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    n = len(vals)
    if n < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / n
    return vals.mean() / se if se > 0 else np.nan


def main() -> int:
    t0 = time.time()
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}
    ltsr = long["timestamp"]
    lts = (ltsr.dt.tz_localize(None) if ltsr.dt.tz is not None else ltsr).to_numpy()
    LONG = long.copy(); LONG["_ts"] = lts
    lpos_all = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
    isd_all = LONG["is_downside"].to_numpy().astype(bool)
    T0 = [x for x in TIER0 if x in LONG.columns]

    parts = []
    for SIGNAL, HZ in SIGNALS.items():
        bcol, tcol = f"bottom_{SIGNAL}", f"top_{SIGNAL}"
        S = sig[["timestamp", bcol, tcol]].copy()
        if S["timestamp"].dt.tz is not None:
            S["timestamp"] = S["timestamp"].dt.tz_localize(None)
        S["pos"] = [pos_of.get(np.datetime64(t), -1) for t in S["timestamp"].to_numpy()]
        S = S.loc[S["pos"] >= 0]
        fb = np.zeros(n, bool); ft = np.zeros(n, bool)
        fb[S["pos"].to_numpy()] = S[bcol].fillna(False).to_numpy(bool)
        ft[S["pos"].to_numpy()] = S[tcol].fillna(False).to_numpy(bool)
        kb, kt = causal_first_fire(fb, GAP), causal_first_fire(ft, GAP)
        keep = (lpos_all >= 0) & (lpos_all + 1 + HZ < n)
        keep &= np.where(isd_all, kb[np.clip(lpos_all, 0, n - 1)], kt[np.clip(lpos_all, 0, n - 1)])
        if keep.sum() < 300:
            continue
        cols = list(dict.fromkeys(["_ts", "split", "is_downside", "atr"] + T0))  # ⚠️atr가
        # TIER0에도 있어 중복되면 D["atr"]이 2차원이 된다
        D = LONG.loc[keep, cols].reset_index(drop=True)
        ii = lpos_all[keep]
        sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
        entry = o[ii + 1]
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, _ = sim_exit(entry, D["atr"].to_numpy(float), sg, H, L, C, *CELL)
        D["net_bp"] = pn * 1e4 - COST_BP
        D["signal"] = SIGNAL
        D["horizon"] = HZ
        parts.append(D)
        log(f"  {SIGNAL:26s} {len(D):6,}건  평균 {D['net_bp'].mean():+7.2f}bp")

    P = pd.concat(parts, ignore_index=True)
    for s_ in SIGNALS:
        P[f"is_{s_}"] = (P["signal"] == s_).astype(np.int8)
    FEATS = T0 + [f"is_{s_}" for s_ in SIGNALS] + ["horizon"]
    X = P[FEATS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    net = P["net_bp"].to_numpy(float)
    split = P["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    log(f"\n⭐통합 풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,}) · 피쳐 {len(FEATS)}개")
    log(f"  풀 전체 평균 {net.mean():+.2f}bp · VAL 평균 {net[va].mean():+.2f}bp\n")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabpfn import TabPFNRegressor

    res = {}
    for mname in ("HGB_reg", "TabPFN_reg"):
        log(f"=== {mname} ===")
        if mname == "HGB_reg":
            m = HistGradientBoostingRegressor(random_state=SEED, max_iter=400,
                                              learning_rate=0.05)
            m.fit(X[tr], net[tr]); pred = m.predict(X[va])
        else:
            # TabPFN 컨텍스트 상한 -- TRAIN이 크면 무작위 서브샘플(라벨 분포 보존은 회귀라 불필요)
            rng = np.random.default_rng(SEED)
            tri = np.flatnonzero(tr)
            ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
            m = TabPFNRegressor(device="cuda", random_state=SEED,
                                ignore_pretraining_limits=True)
            m.fit(X[ctx], net[ctx]); pred = m.predict(X[va])
        Pva = P.loc[va].reset_index(drop=True)
        nv, dv = net[va], days[va]
        print(f"{'신호':>24s}{'VAL n':>8s}{'k':>5s}{'독립일':>7s}{'전체bp':>9s}{'상위bp':>9s}{'일t':>7s}")
        print("-" * 70)
        per = {}
        for s_ in SIGNALS:
            mk = (Pva["signal"] == s_).to_numpy()
            if mk.sum() < 80:
                continue
            k = max(10, int(round(mk.sum() * TOPQ)))
            sub_pred, sub_net, sub_day = pred[mk], nv[mk], dv[mk]
            top = np.argsort(-sub_pred)[:k]
            tb, tt = float(sub_net[top].mean()), float(cluster_t(sub_net[top], sub_day[top]))
            print(f"{s_[:23]:>24s}{int(mk.sum()):8d}{k:5d}{len(np.unique(sub_day[top])):7d}"
                  f"{float(sub_net.mean()):9.2f}{tb:9.2f}{tt:7.2f}{'  ⭐' if tt > 1.96 else ''}")
            per[s_] = {"n_val": int(mk.sum()), "k": k, "all_mean_bp": float(sub_net.mean()),
                       "top_mean_bp": tb, "cluster_t": tt,
                       "independent_days": int(len(np.unique(sub_day[top])))}
        # 풀 전체 상위10%
        kk = max(10, int(round(va.sum() * TOPQ)))
        tp = np.argsort(-pred)[:kk]
        ptb, ptt = float(nv[tp].mean()), float(cluster_t(nv[tp], dv[tp]))
        npass = sum(1 for v in per.values() if v["cluster_t"] > 1.96)
        log(f"  ⭐풀 전체 상위10%: {ptb:+.2f}bp t={ptt:.2f} (n={kk}, 일{len(np.unique(dv[tp]))})")
        log(f"  ⭐신호별 통과: **{npass}/{len(per)}**\n")
        res[mname] = {"per_signal": per, "pool_top_bp": ptb, "pool_top_t": ptt,
                      "n_passed": npass, "n_signals": len(per)}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"models": res, "cell": list(CELL), "cost_bp": COST_BP,
                               "n_pool": int(len(P)), "features": FEATS,
                               "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
