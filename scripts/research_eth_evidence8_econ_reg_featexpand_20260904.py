#!/usr/bin/env python3
"""증거신호 8종 **경제성 회귀** -- 피쳐셋 확장 x 모델 비교 (목표: 경제성 관문 통과).

## 배경

`audit_eth_evidence8_econ_topdecile_robustness_20260904.py`가 64조합 중 **2개만** 일단위
군집 CI를 통과시켰다(taker/fib, 둘 다 회귀). 노이즈 기대치가 1.6개라 통계적으로 구분되지
않는다. 그런데 **생존 2개가 전부 회귀(L2_reg)** 였다 -- 부호 분류는 트레일링의 "적게 자주
이기고 크게 가끔 잃는" 구조에서 손익을 결정하는 크기 정보를 버린다.

⇒ 두 축을 동시에 민다:
  **모델**   HGB 회귀 vs **TabPFN 회귀**(이 저장소가 이 표본 크기에서 반복 확인한 기본기)
  **피쳐**   Tier0 23은 전부 klines 파생이다. BTC는 "Tier0에 방향력 없음"으로 종결됐고
             (`btc_v_rebound_econ_label_closed_no_direction_skill_20260902`), 오늘 진입모델은
             **비-klines 축(펀딩/OI/BTC교차자산)** 에서만 정보가 더 있다는 증거를 봤다.

    F0  Tier0 23                      (현행 기준선)
    F1  Tier0 + **재료텐서**(51열)     8신호 OOF proba/pct/age/fold + 레짐 2 -- "지금 다른
                                       신호들이 뭘 보고 있나". ⭐OOF본이라 C버그(스태킹 누수) 없음
    F2  Tier0 + build_all_bar_frame 여분 klines 파생 29개

## 평가

상위10% 선별 평균 bp + **일 단위 군집 CI**(오늘 진입모델을 죽인 검정). 속도를 위해 CI는
클러스터-로버스트 SE의 해석적 근사로 먼저 훑고, 통과분만 부트스트랩으로 재확인한다.

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


_pf = _load("pf_fx", "scripts/research_eth_v_rebound_ensemble_portfolio_sim_20260902.py")
_s1 = _pf._s1
TIER0, sim_exit = _pf.TIER0, _pf.sim_exit

SIGNALS = {"liquidity_sweep": 30, "taker_delta_z_climax": 24, "short_term_return_z": 12,
           "orthogonal_combo": 24, "smt_divergence": 72, "fib_extension_exhaustion": 20,
           "demarker_extreme": 8, "kalman_deviation_meanrev": 12}
CELL = (4.0, 1.0, 0.1)
GAP, COST_BP, TOPQ, SEED = 12, 10.0, 0.10, 20260904
MAT = ROOT / "data/materials/eth_evidence_signal_tensor_oof_20260903/eth_evidence_material_5m.parquet"
OUT = ROOT / "data/research/eth_evidence8_econ_reg_featexpand_20260904/report.json"


def log(m): print(f"[fx] {m}", flush=True)


def causal_first_fire(fire, gap):
    keep = np.zeros(len(fire), bool); last = -10**9
    for i in np.flatnonzero(fire):
        if i - last > gap:
            keep[i] = True
        last = i
    return keep


def cluster_t(vals, days):
    """일 단위 클러스터-로버스트 t. 행 단위 t는 하루 안 상관을 무시해 과대평가된다."""
    n = len(vals)
    if n < 5:
        return np.nan
    mu = vals.mean()
    dev = vals - mu
    s = 0.0
    for d in np.unique(days):
        s += dev[days == d].sum() ** 2
    se = np.sqrt(s) / n
    return mu / se if se > 0 else np.nan


def main() -> int:
    t0 = time.time()
    log("프레임 빌드...")
    sig, feat, eth = _s1.build_sig()
    dummy = np.full(len(sig), "none", dtype=object)
    long = _s1.long_frame_for(sig, feat, dummy, dummy)

    mat = pd.read_parquet(MAT)
    if mat["timestamp"].dt.tz is not None:
        mat["timestamp"] = mat["timestamp"].dt.tz_localize(None)
    MATC = [c for c in mat.columns if c != "timestamp"]
    F2C = [c for c in feat.columns
           if c not in TIER0 and c != "timestamp" and pd.api.types.is_numeric_dtype(feat[c])]
    log(f"  재료텐서 {len(MATC)}열 · F2 여분 {len(F2C)}열")

    kl = eth[["timestamp", "open", "high", "low", "close"]].copy()
    if kl["timestamp"].dt.tz is not None:
        kl["timestamp"] = kl["timestamp"].dt.tz_localize(None)
    o, h, l, c = (kl[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(kl)
    pos_of = {t: i for i, t in enumerate(kl["timestamp"].to_numpy())}

    ltsr = long["timestamp"]
    lts = (ltsr.dt.tz_localize(None) if ltsr.dt.tz is not None else ltsr).to_numpy()
    LONG = long.copy(); LONG["_ts"] = lts
    LONG = LONG.merge(mat.rename(columns={"timestamp": "_ts"}), on="_ts", how="left")
    fe = feat.copy()
    if fe["timestamp"].dt.tz is not None:
        fe["timestamp"] = fe["timestamp"].dt.tz_localize(None)
    LONG = LONG.merge(fe[["timestamp"] + F2C].rename(columns={"timestamp": "_ts"}),
                      on="_ts", how="left", suffixes=("", "_f2"))
    F2C = [c if c in LONG.columns else f"{c}_f2" for c in F2C]
    F2C = [c for c in F2C if c in LONG.columns]

    from sklearn.ensemble import HistGradientBoostingRegressor
    from tabpfn import TabPFNRegressor

    FSETS = {"F0_tier0": list(TIER0),
             "F1_tier0+material": list(TIER0) + MATC,
             "F2_tier0+klines_extra": list(TIER0) + F2C}
    log(f"  피쳐셋 크기: " + " · ".join(f"{k}={len(v)}" for k, v in FSETS.items()))
    print(f"\n{'신호':>22s}{'피쳐셋':>22s}{'모델':>9s}{'k':>5s}{'독립일':>7s}"
          f"{'전체bp':>9s}{'상위bp':>9s}{'일t':>7s}")
    print("-" * 92)

    rep = {}
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

        lpos = np.array([pos_of.get(np.datetime64(t), -1) for t in LONG["_ts"].to_numpy()])
        isd = LONG["is_downside"].to_numpy().astype(bool)
        keep = (lpos >= 0) & (lpos + 1 + HZ < n)
        keep &= np.where(isd, kb[np.clip(lpos, 0, n - 1)], kt[np.clip(lpos, 0, n - 1)])
        D = LONG.loc[keep].reset_index(drop=True)
        ii = lpos[keep]
        if len(D) < 400:
            continue
        sg = np.where(D["is_downside"].to_numpy() == 1, 1.0, -1.0)
        entry = o[ii + 1]
        H = np.stack([h[i + 1:i + 1 + HZ] for i in ii])
        L = np.stack([l[i + 1:i + 1 + HZ] for i in ii])
        C = np.stack([c[i + 1:i + 1 + HZ] for i in ii])
        pn, _ = sim_exit(entry, D["atr"].to_numpy(float), sg, H, L, C, *CELL)
        net = pn * 1e4 - COST_BP
        split = D["split"].to_numpy()
        tr, va = split == "TRAIN", split == "VAL"
        if tr.sum() < 300 or va.sum() < 100:
            continue
        days = pd.to_datetime(D["_ts"]).dt.floor("D").to_numpy()
        k = max(10, int(round(va.sum() * TOPQ)))
        nv, dv = net[va], days[va]
        base = float(nv.mean())

        for fname, cols in FSETS.items():
            use = [x for x in cols if x in D.columns]
            X = D[use].apply(pd.to_numeric, errors="coerce").to_numpy(float)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            for mname in ("HGB_reg", "TabPFN_reg"):
                try:
                    if mname == "HGB_reg":
                        m = HistGradientBoostingRegressor(random_state=SEED, max_iter=300,
                                                          learning_rate=0.05)
                    else:
                        m = TabPFNRegressor(device="cuda", random_state=SEED,
                                            ignore_pretraining_limits=True)
                    m.fit(X[tr], net[tr])
                    pred = m.predict(X[va])
                    top = np.argsort(-pred)[:k]
                    tb, tt = float(nv[top].mean()), float(cluster_t(nv[top], dv[top]))
                    star = "  ⭐" if tt > 1.96 else ""
                    print(f"{SIGNAL[:21]:>22s}{fname:>22s}{mname:>9s}{k:5d}"
                          f"{len(np.unique(dv[top])):7d}{base:9.2f}{tb:9.2f}{tt:7.2f}{star}")
                    rep[f"{SIGNAL}|{fname}|{mname}"] = {
                        "k": k, "n_feat": len(use), "all_mean_bp": base, "top_mean_bp": tb,
                        "cluster_t": tt, "independent_days": int(len(np.unique(dv[top]))),
                        "n_val": int(va.sum())}
                except Exception as e:                            # noqa: BLE001
                    log(f"  ⚠️{SIGNAL}|{fname}|{mname}: {type(e).__name__}: {str(e)[:60]}")
        print("-" * 92)

    surv = {k_: v for k_, v in rep.items() if v["cluster_t"] > 1.96}
    print()
    log(f"⭐일단위 클러스터 t>1.96 : **{len(surv)}/{len(rep)}** (노이즈 기대 {len(rep)*0.025:.1f}개)")
    for k_, v in sorted(surv.items(), key=lambda x: -x[1]["cluster_t"])[:12]:
        log(f"    {k_:<58s} 상위{v['top_mean_bp']:+7.2f}bp t={v['cluster_t']:.2f} "
            f"(전체{v['all_mean_bp']:+.2f}, 일{v['independent_days']})")
    by_fs, by_md = {}, {}
    for k_, v in rep.items():
        _, fs, md = k_.split("|")
        by_fs.setdefault(fs, []).append(v["cluster_t"] > 1.96)
        by_md.setdefault(md, []).append(v["cluster_t"] > 1.96)
    log("  피쳐셋별 통과: " + " · ".join(f"{k_}={sum(v)}/{len(v)}" for k_, v in by_fs.items()))
    log("  모델별 통과  : " + " · ".join(f"{k_}={sum(v)}/{len(v)}" for k_, v in by_md.items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"combos": rep, "cell": list(CELL), "cost_bp": COST_BP,
                               "n_survived": len(surv), "oos_touched": False,
                               "runtime_sec": round(time.time() - t0, 1)},
                              ensure_ascii=False, indent=2))
    log(f"산출: {OUT} ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
