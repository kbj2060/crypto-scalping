#!/usr/bin/env python3
"""증거신호 8종 경제성 회귀 -- **피쳐 제거(pruning)** (목표가 명시한 마지막 미탐색 축).

## 왜 제거인가

피쳐를 **넣을 때마다 매번 나빠졌다**:
    F1 재료텐서(+51열)      1/16   희석
    F2 klines 여분(+29열)   0/16
    신호 원핫 포함 32피쳐   0/8   (표본 8배인데도)
    FX 비-klines(+145열)    0/8   풀 상위10% -4.85bp (F0 +3.05보다 나쁨, 동일 행 비교)

정보가 부족한 게 아니라 **노이즈가 과한** 쪽이라는 뜻이다. 목표 지시대로 반대 방향을 민다.

## 설계

    랭킹   TRAIN에서만 |Spearman IC(피쳐, net_bp)| 로 정렬 -- ⚠️VAL을 보면 선택편향
    부분집합  상위 3/5/8/12/16/23 (+ 신호 원핫은 항상 포함)
    모델   TabPFN 회귀
    평가   신호별 상위10% 일단위 클러스터 t + 풀 전체

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


_pf = _load("pf_pr", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
# ⭐IC 랭킹이 "상위=전부 변동성, 방향성은 IC~0"으로 나왔다. net_bp는 방향으로 부호가 붙으므로
# 크기만 예측하면 고변동성 거래를 상위로 뽑고 절반은 반대로 크게 잃는다 -- 정확히 역효과다.
# 그래서 IC 상위 절단(변동성을 남김)이 아니라 **변동성 블록 제거**를 시험한다.
VOL_BLOCK = ["atr", "range_width_pct", "atr_percentile_864", "bb_width_pctile", "vol_z",
             "atr_pct", "spread"]
TIME_BLOCK = ["hour_utc", "weekday"]
OUT = ROOT / "data/research/eth_evidence8_econ_pruning_v2_20260904/report.json"


def log(m): print(f"[prune] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    if len(vals) < 5:
        return np.nan
    dev = vals - vals.mean()
    s = sum(dev[days == d].sum() ** 2 for d in np.unique(days))
    se = np.sqrt(s) / len(vals)
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
        cols = list(dict.fromkeys(["_ts", "split", "is_downside", "atr"] + T0))
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
        parts.append(D)
    P = pd.concat(parts, ignore_index=True)
    for s_ in SIGNALS:
        P[f"is_{s_}"] = (P["signal"] == s_).astype(np.int8)
    ONE = [f"is_{s_}" for s_ in SIGNALS]
    net = P["net_bp"].to_numpy(float)
    split = P["split"].to_numpy()
    tr, va = split == "TRAIN", split == "VAL"
    days = pd.to_datetime(P["_ts"]).dt.floor("D").to_numpy()
    log(f"⭐풀 {len(P):,}건 (TRAIN {tr.sum():,} / VAL {va.sum():,})")

    # ⚠️랭킹은 TRAIN에서만 -- VAL을 보면 선택편향
    from scipy.stats import spearmanr
    ic = {}
    for cname in T0:
        v = pd.to_numeric(P[cname], errors="coerce").to_numpy(float)
        m_ = tr & np.isfinite(v)
        if m_.sum() < 500 or np.nanstd(v[m_]) == 0:
            ic[cname] = 0.0
            continue
        r = spearmanr(v[m_], net[m_]).correlation
        ic[cname] = 0.0 if not np.isfinite(r) else abs(float(r))
    rank = sorted(T0, key=lambda x: -ic[x])
    log("  TRAIN |IC| 상위 8: " + " · ".join(f"{c_}={ic[c_]:.4f}" for c_ in rank[:8]))
    log("  하위 5: " + " · ".join(f"{c_}={ic[c_]:.4f}" for c_ in rank[-5:]))

    from tabpfn import TabPFNRegressor
    SUBSETS = {
        "A_all23": T0,
        "B_no_vol": [x for x in T0 if x not in VOL_BLOCK],
        "C_no_vol_no_time": [x for x in T0 if x not in VOL_BLOCK + TIME_BLOCK],
        "D_vol_only": [x for x in T0 if x in VOL_BLOCK],
        "E_ic_bottom12": rank[-12:],
    }
    print(f"\n{'부분집합':>18s}{'피쳐수':>7s}{'풀상위bp':>10s}{'풀일t':>8s}{'통과/8':>8s}   신호별 통과")
    print("-" * 92)
    res = {}
    for k_, base_cols in SUBSETS.items():
        use = list(base_cols) + ONE
        X = np.nan_to_num(P[use].apply(pd.to_numeric, errors="coerce").to_numpy(float),
                          nan=0.0, posinf=0.0, neginf=0.0)
        rng = np.random.default_rng(SEED)
        tri = np.flatnonzero(tr)
        ctx = rng.choice(tri, size=min(18000, len(tri)), replace=False)
        m = TabPFNRegressor(device="cuda", random_state=SEED, ignore_pretraining_limits=True)
        m.fit(X[ctx], net[ctx]); pred = m.predict(X[va])
        Pva = P.loc[va].reset_index(drop=True)
        nv, dv = net[va], days[va]
        per, passed = {}, []
        for s_ in SIGNALS:
            mk = (Pva["signal"] == s_).to_numpy()
            if mk.sum() < 80:
                continue
            kk = max(10, int(round(mk.sum() * TOPQ)))
            sp, sn, sd = pred[mk], nv[mk], dv[mk]
            top = np.argsort(-sp)[:kk]
            tb, tt = float(sn[top].mean()), float(cluster_t(sn[top], sd[top]))
            per[s_] = {"top_mean_bp": tb, "cluster_t": tt,
                       "independent_days": int(len(np.unique(sd[top])))}
            if tt > 1.96:
                passed.append(s_)
        pk = max(10, int(round(va.sum() * TOPQ)))
        tp = np.argsort(-pred)[:pk]
        ptb, ptt = float(nv[tp].mean()), float(cluster_t(nv[tp], dv[tp]))
        print(f"{k_:>18s}{len(use):7d}{ptb:10.2f}{ptt:8.2f}{len(passed):8d}   "
              f"{', '.join(x[:14] for x in passed) if passed else '-'}")
        res[str(k_)] = {"features": list(base_cols), "pool_top_bp": ptb, "pool_top_t": ptt,
                        "n_passed": len(passed), "passed": passed, "per_signal": per}

    best = max(res.items(), key=lambda x: x[1]["n_passed"])
    log(f"\n⭐최다 통과: {best[0]} -> **{best[1]['n_passed']}/8** "
        f"(풀 {best[1]['pool_top_bp']:+.2f}bp t={best[1]['pool_top_t']:.2f})")
    log(f"  IC 랭킹 전체: {rank}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"subsets": res, "train_only_ic": ic, "rank": rank,
                               "n_pool": int(len(P)), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
