#!/usr/bin/env python3
"""앵커 방향 예측 — **TabPFN** + 같은 대조군 (2026-09-07). 서버(GPU) 실행 전용.

로컬 기준선(로짓)은 VAL 0.5308 / OOS 0.5889 를 냈지만 **라벨 셔플 귀무**(일 안 셔플)가
VAL 0.5212(p95 0.5487) · OOS 0.5415(p95 0.5942) 라 백분위 70/92 로 **미통과**였다.
TabPFN 은 데이터셋별 학습이 없는 in-context 추론이라 HGB/로짓의 과적합 실패모드에 다른
귀납 편향을 준다 -- 모델만 바꿔 직접 비교한다(피쳐·라벨·분할 동일).

⭐**셔플 귀무를 같은 모델로 병기한다.** 순진한 0.5 기준은 이 문제에서 틀렸다는 것이
로컬 대조군에서 이미 확인됐다(일군집 구조 때문에 귀무가 0.52~0.54).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "tmp/eth_anchor_features_20260907/features.parquet"
OUT = ROOT / "tmp/eth_anchor_tabpfn_20260907"
SEEDS = [11, 23, 47, 71, 97]
B_NULL = 20
DEVICE = "cuda"


def fit(Xtr, ytr, Xte, seed, device=DEVICE):
    from tabpfn import TabPFNClassifier
    clf = TabPFNClassifier(device=device, random_state=seed)
    clf.fit(Xtr, ytr.astype(int))
    return clf.predict_proba(Xte)[:, 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(SRC)
    meta = json.loads((ROOT / "tmp/eth_anchor_features_20260907/meta.json").read_text())
    F = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[F].to_numpy(float)
    X = np.where(np.isfinite(X), X, np.nan)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    rng = np.random.default_rng(20260907)
    res = {}

    for lab, yv in (("y2", D["y2"].to_numpy(float)),
                    ("y3_clean", np.where(D["y3"].to_numpy() != 1,
                                          (D["y3"].to_numpy() == 2).astype(float), np.nan))):
        ok = np.isfinite(yv)
        tr, va, oo = ok & (sp == "TRAIN"), ok & (sp == "VAL"), ok & (sp == "OOS")
        te = va | oo
        print(f"\n=== {lab}: TRAIN {tr.sum()} / VAL {va.sum()} / OOS {oo.sum()} · "
              f"양성률 {yv[tr].mean():.3f}/{yv[va].mean():.3f}/{yv[oo].mean():.3f}", flush=True)
        # 관측
        aucs = {"VAL": [], "OOS": []}
        for s in SEEDS:
            p = fit(X[tr], yv[tr], X[te], s)
            pa = np.full(len(yv), np.nan); pa[te] = p
            for k, m in (("VAL", va), ("OOS", oo)):
                aucs[k].append(roc_auc_score(yv[m], pa[m]))
            print(f"   seed {s}: VAL {aucs['VAL'][-1]:.4f} · OOS {aucs['OOS'][-1]:.4f}", flush=True)
        obs = {k: float(np.mean(v)) for k, v in aucs.items()}
        sd = {k: float(np.std(v)) for k, v in aucs.items()}
        # 셔플 귀무 (같은 모델, 일 안 셔플)
        print(f"   셔플 귀무 B={B_NULL} ...", flush=True)
        nulls = {"VAL": [], "OOS": []}
        dtr = day[tr]; y0 = yv[tr].copy()
        for b in range(B_NULL):
            ys = y0.copy()
            for d in np.unique(dtr):
                m = dtr == d
                ys[m] = rng.permutation(ys[m])
            yy = yv.copy(); yy[tr] = ys
            p = fit(X[tr], yy[tr], X[te], SEEDS[0])
            pa = np.full(len(yv), np.nan); pa[te] = p
            for k, m in (("VAL", va), ("OOS", oo)):
                nulls[k].append(roc_auc_score(yv[m], pa[m]))
        res[lab] = {"obs": obs, "seed_sd": sd,
                    "null": {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                                 "obs_pctile": float((np.array(v) < obs[k]).mean() * 100)}
                             for k, v in nulls.items()},
                    "seed_aucs": {k: [float(x) for x in v] for k, v in aucs.items()}}
        for k in ("VAL", "OOS"):
            r = res[lab]["null"][k]
            print(f"   {k}: 관측 {obs[k]:.4f}±{sd[k]:.4f} · 귀무 {r['mean']:.4f} (p95 {r['p95']:.4f})"
                  f" · 백분위 {r['obs_pctile']:.0f}", flush=True)
    (OUT / "tabpfn.json").write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
