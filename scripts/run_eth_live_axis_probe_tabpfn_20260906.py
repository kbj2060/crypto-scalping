#!/usr/bin/env python3
"""실시간 축 탐침 — TabPFN/HGB (2026-09-06). **서버 실행**.

⚠️ 단일 TEST 창이다(확인 창 없음). 다중검정 위험이 커서 **사전에 주 비교를 하나로 고정**한다:
    주 비교 = `y_dec` 라벨에서 **all(실시간 포함) − base(실시간 제외)** 의 TEST AUC 차.
나머지 조합은 보조 진단으로만 보고한다. 추가로 **라벨 셔플 귀무**(TRAIN 라벨을 일 단위로 섞어
같은 파이프라인을 20회 재적합)를 돌려 이 파이프라인이 잡음에서 만들어내는 AUC 분포를 같이 낸다.

n(TEST)=1,314 -> AUC 표준오차 ~0.0155. 0.53 미만은 0.5와 구분 불가로 읽는다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "tmp/eth_live_axis_probe_20260906/dataset.parquet"
OUT = ROOT / "tmp/eth_live_axis_probe_tabpfn_20260906"
SEEDS = [11, 23, 47, 71, 97]
N_NULL = 20
DEVICE = "cuda"
RNG = np.random.default_rng(20260906)
NON_FEAT = {"y_order", "y_dec", "margin", "pnl_fade", "pnl_cont", "cls3", "split", "timestamp"}
CORE = ["votes", "is_downside", "atr_pct", "signal_id"]


def log(m): print(f"[probe-ml] {m}", flush=True)


def day_ci(v, d, B=2000):
    ud = np.unique(d); idx = {x: np.flatnonzero(d == x) for x in ud}
    o = np.empty(B)
    for b in range(B):
        p = RNG.choice(ud, len(ud), replace=True)
        o[b] = np.concatenate([v[idx[x]] for x in p]).mean()
    return float(np.percentile(o, 2.5)), float(np.percentile(o, 97.5))


def pred(m, X, proba, bs=8000):
    return np.concatenate([(m.predict_proba(X.iloc[k:k + bs])[:, 1] if proba else m.predict(X.iloc[k:k + bs]))
                           for k in range(0, len(X), bs)])


def fit_predict(engine, task, Xtr, ytr, Xte, cat_idx, seeds=SEEDS):
    P = np.zeros(len(Xte))
    for sd in seeds:
        if engine == "tabpfn":
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            m = (TabPFNClassifier if task == "clf" else TabPFNRegressor)(
                device=DEVICE, random_state=sd, ignore_pretraining_limits=True,
                categorical_features_indices=cat_idx or None)
        else:
            m = (HistGradientBoostingClassifier if task == "clf" else HistGradientBoostingRegressor)(
                max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
                l2_regularization=1.0, random_state=sd)
        m.fit(Xtr, ytr)
        P += pred(m, Xte, task == "clf") / len(seeds)
    return P


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(DS)
    tr, te = (D["split"] == "TRAIN").to_numpy(), (D["split"] == "TEST").to_numpy()
    allf = [c for c in D.columns if c not in NON_FEAT]
    lvf = [c for c in allf if c.startswith("lv_")]
    base = [c for c in allf if not c.startswith("lv_")]
    live_only = CORE + lvf
    log(f"{len(D):,}행  TRAIN {tr.sum():,} / TEST {te.sum():,}  |  base {len(base)} · live {len(lvf)} · all {len(allf)}")
    days_te = pd.Series(D.loc[te, "timestamp"]).dt.floor("D").to_numpy()
    pf, pc = D["pnl_fade"].to_numpy(), D["pnl_cont"].to_numpy()
    res = {}

    for fs, cols in (("base(실시간 제외)", base), ("live_only", live_only), ("all(실시간 포함)", allf)):
        cat = [cols.index("signal_id")] if "signal_id" in cols else []
        Xtr, Xte = D.loc[tr, cols], D.loc[te, cols]
        for task, label in (("clf", "y_order"), ("clf", "y_dec"), ("reg", "margin")):
            ytr, yte = D.loc[tr, label].to_numpy(), D.loc[te, label].to_numpy()
            for eng in ("tabpfn", "hgb"):
                t0 = time.time()
                try:
                    P = fit_predict(eng, task, Xtr, ytr, Xte, cat)
                except Exception as ex:                                        # noqa: BLE001
                    log(f"{fs}|{label}|{eng} 실패: {ex}"); continue
                r = {"featureset": fs, "label": label, "engine": eng, "n_feat": len(cols),
                     "sec": round(time.time() - t0, 1)}
                r["score"] = (float(roc_auc_score(yte, P)) if task == "clf"
                              else float(spearmanr(yte, P).statistic))
                arms = {}
                for q in (0.10, 0.20, 0.30):
                    use = P >= np.quantile(P, 1 - q)
                    d = np.where(use, pf[te], pc[te]) - pc[te]
                    lo, hi = day_ci(d, days_te)
                    arms[f"q{q}"] = {"n": int(use.sum()), "delta": float(d.mean()), "ci": [lo, hi]}
                if task == "reg":
                    use = P > 0
                    d = np.where(use, pf[te], pc[te]) - pc[te]
                    arms["margin>0"] = {"n": int(use.sum()), "delta": float(d.mean()),
                                        "ci": list(day_ci(d, days_te)) if use.any() else [0.0, 0.0]}
                r["arms"] = arms
                res[f"{fs}|{label}|{eng}"] = r
                log(f"{fs:<16}|{label:<8}|{eng:<7} TEST {r['score']:+.4f}  ({r['sec']}s)")

    # 라벨 셔플 귀무 — 주 비교 구성(all|y_dec|tabpfn)에서 파이프라인이 잡음에 주는 AUC 분포
    log("라벨 셔플 귀무 20회 ...")
    cols = allf; cat = [cols.index("signal_id")]
    Xtr, Xte = D.loc[tr, cols], D.loc[te, cols]
    ytr = D.loc[tr, "y_dec"].to_numpy(); yte = D.loc[te, "y_dec"].to_numpy()
    dtr = pd.Series(D.loc[tr, "timestamp"]).dt.floor("D").to_numpy()
    nulls = []
    for i in range(N_NULL):
        ysh = ytr.copy()
        for dday in np.unique(dtr):                       # 일 단위 블록 셔플(자기상관 보존)
            m = dtr == dday
            ysh[m] = RNG.permutation(ysh[m])
        ysh = RNG.permutation(ysh)
        try:
            P = fit_predict("tabpfn", "clf", Xtr, ysh, Xte, cat, seeds=[11])
            nulls.append(float(roc_auc_score(yte, P)))
        except Exception as ex:                                                # noqa: BLE001
            log(f"null {i} 실패: {ex}"); break
    obs = res.get("all(실시간 포함)|y_dec|tabpfn", {}).get("score")
    null_summary = {"n": len(nulls), "mean": float(np.mean(nulls)) if nulls else None,
                    "p95": float(np.percentile(nulls, 95)) if nulls else None,
                    "max": float(np.max(nulls)) if nulls else None,
                    "obs_pctile": float((np.array(nulls) < obs).mean() * 100) if nulls and obs else None}
    res["_null_shuffle"] = null_summary

    (OUT / "results.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float))
    print("\n" + "=" * 92)
    print(f"{'피쳐셋':<18}{'라벨':<9}{'엔진':<8}{'TEST 지표':>11}   {'라우팅 q0.2 Δ [일군집 CI]':>30}")
    for k, r in res.items():
        if k.startswith("_"):
            continue
        a = r["arms"]["q0.2"]
        print(f"{r['featureset']:<18}{r['label']:<9}{r['engine']:<8}{r['score']:>11.4f}   "
              f"{a['delta']:>+8.2f} [{a['ci'][0]:+7.2f},{a['ci'][1]:+7.2f}]")
    print("\n=== 주 비교 (사전 고정): y_dec 에서 all − base ===")
    for eng in ("tabpfn", "hgb"):
        b = res.get(f"base(실시간 제외)|y_dec|{eng}", {}).get("score")
        a = res.get(f"all(실시간 포함)|y_dec|{eng}", {}).get("score")
        lo = res.get(f"live_only|y_dec|{eng}", {}).get("score")
        if b is not None and a is not None:
            print(f"  {eng:<8} base {b:.4f} -> all {a:.4f}  (Δ {a-b:+.4f})   live_only {lo:.4f}")
    ns = res["_null_shuffle"]
    print(f"\n=== 라벨 셔플 귀무 (all|y_dec|tabpfn, {ns['n']}회) ===")
    print(f"  귀무 평균 {ns['mean']:.4f} · p95 {ns['p95']:.4f} · 최대 {ns['max']:.4f}"
          f"  -> 관측 {obs:.4f} 은 귀무의 {ns['obs_pctile']:.0f} 백분위" if ns["n"] else "  실패")
    print(f"\n산출물: {OUT/'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
