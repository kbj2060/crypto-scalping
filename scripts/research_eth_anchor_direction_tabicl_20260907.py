#!/usr/bin/env python3
"""앵커 되돌림/지속 — **TabICLv2** 3팔 x 피쳐셋 9종 (2026-09-07).

사용자: *"tabiclv2 먼저 진행해줘"* → 설계 승인 후 실행.

## 왜 TabICLv2
표본 300~48K · 컬럼 2~100 구간에 사전학습된 tabular foundation model(arXiv 2602.11139).
우리 표본(TRAIN 3,237)이 정확히 그 구간이다. **하이퍼파라미터 튜닝이 불필요**하다고 주장하므로
HP 그리드가 사라져 다중검정 부담이 줄고, 부록 M 의 이월성 문제(rho -0.937)를 우회한다.
체크포인트 `tabicl-classifier-v2-20260212.ckpt` 는 토큰 없이 HF 에서 자동 다운로드된다.

## Step 0/1 실측 (2026-09-07)
CPU fit+predict n_estimators=2 약 19초 · 8 약 53초.
⭐**양성대조(크기 라벨, dirtop40) VAL AUC 0.7581(n=2) / 0.7600(n=8)**
   -- HGB 0.650 · raw atr_pct 0.708 을 **처음으로 넘었다**. 파이프라인 정상 + 모델이 실제로 강하다.

## ⚠️`fit(X, y)` 에 sample_weight 가 없다
=> wbin 팔의 `w_final`(고유도 x 품질) 가중을 **못 쓴다**. 가중 없이 돌리고 그 사실을 명시한다.
   hard/three 는 원래 w_uniq 가중이었고 고유도가 0.9 수준이라 손실이 작다.

## 피쳐셋 9종 (전부 TRAIN 내부에서만 선택 -- 컬럼 상한 100 주의)
  dirtop20/40/80  TRAIN 전용 방향 순위(TRAIN 을 시간순 2등분해 두 반쪽 부호일치 요구)
  perm20/40/80    TRAIN CV 순열중요도
  volrange(16)    카테고리 스윕 최다 상위축
  nocorr(~90)     |rho|>0.9 제거(라벨 미사용)
  all150          **상한 초과 -- 대조로만**

## 판정 (실행 전 고정)
통과 = VAL·OOS 둘 다 일군집 CI 하한 > 0.5 ∧ 날블록 셔플 귀무 p95 초과 ∧ HGB 기준선 초과.
검정력: wbin OOS 340 -> AUC SE 0.031(0.56 미만 무의미) · hard OOS 107 -> 0.055(0.61 미만).
27셀이므로 **통과 개수를 귀무 분포와 비교**한다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1]
SRCD = ROOT / "tmp/eth_anchor_features154_20260907"
RANK = ROOT / "tmp/eth_anchor_direction_feature_ranking_20260907"
OUT = ROOT / "tmp/eth_anchor_tabicl_20260907"
N_EST = 4
BOOT = 1500
SEED = 20260907


def day_auc_ci(y, p, days, rng, B=BOOT):
    uniq = np.unique(days)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], p[ii]))
    if len(out) < B // 3:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def arm_spec(D, arm):
    y3 = D["y3"].to_numpy()
    if arm == "hard":
        return D["is_clean"].to_numpy(bool), (y3 == 2).astype(int), False
    if arm == "wbin":
        y = D["y_bin"].to_numpy()
        return np.isfinite(y), np.nan_to_num(y).astype(int), False
    return np.isfinite(y3), y3.astype(int), True


def score(y, p, multi):
    """이진: AUC. 3클래스: 깨끗한 두 클래스에서 P(2)/(P(0)+P(2)) -- 방향 식별력."""
    if not multi:
        return roc_auc_score(y, p[:, 1]), None
    cl = y != 1
    den = p[cl][:, 0] + p[cl][:, 2]
    dir_p = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
    d = roc_auc_score((y[cl] == 2).astype(int), dir_p) if len(np.unique(y[cl])) > 1 else np.nan
    m = roc_auc_score((y == 1).astype(int), p[:, 1])
    return d, m


def run_cell(D, X, arm, fcols, rng, shuffle=False, n_est=N_EST):
    m, y, multi = arm_spec(D, arm)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    tr = m & (sp == "TRAIN")
    yt = y.copy()
    if shuffle:                                   # 날 블록 셔플
        days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            if len(src[d]):
                yt[src[d]] = np.resize(y[src[d2]], len(src[d]))
    Xf = np.nan_to_num(X[:, fcols].astype(np.float32))
    clf = TabICLClassifier(device="cpu", n_estimators=n_est, random_state=SEED, verbose=False)
    clf.fit(Xf[tr], yt[tr])
    out = {}
    for w in ("VAL", "OOS"):
        te = m & (sp == w)
        p = clf.predict_proba(Xf[te])
        d, mx = score(y[te], p, multi)
        out[f"{w}_n"] = int(te.sum()); out[f"{w}_auc"] = d
        if multi:
            out[f"{w}_mixed"] = mx
            cl = y[te] != 1
            den = p[cl][:, 0] + p[cl][:, 2]
            dp = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
            lo, hi = day_auc_ci((y[te][cl] == 2).astype(int), dp, day[te][cl], rng)
        else:
            lo, hi = day_auc_ci(y[te], p[:, 1], day[te], rng)
        out[f"{w}_lo"], out[f"{w}_hi"] = lo, hi
    return out


def build_feature_sets(D, cols):
    sets = {}
    for k in (20, 40, 80):
        f = RANK / f"dirtop{k}_train.json"
        if f.exists():
            sets[f"dirtop{k}"] = [c for c in json.loads(f.read_text()) if c in cols]
    cv = ROOT / "tmp/eth_anchor_f154_eval_20260907/cv_grid.csv"          # perm 순위 재사용 불가 -> 재계산
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.inspection import permutation_importance
    m, y, multi = arm_spec(D, "wbin")
    tr = m & (D["split"].to_numpy() == "TRAIN")
    X = np.nan_to_num(D[cols].to_numpy(np.float64))
    idx = D["bar_idx"].to_numpy()[tr]; o = np.argsort(idx); cut = int(len(o) * 0.75)
    a, b = o[:cut], o[cut + 48:]
    mdl = HistGradientBoostingClassifier(random_state=0, max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=40, l2_regularization=1.0)
    mdl.fit(X[tr][a], y[tr][a])
    pi = permutation_importance(mdl, X[tr][b], y[tr][b], n_repeats=3, random_state=0, scoring="roc_auc")
    rank = np.argsort(-pi.importances_mean)
    for k in (20, 40, 80):
        sets[f"perm{k}"] = [cols[i] for i in rank[:k]]
    import re
    sets["volrange"] = [c for c in cols if re.search(r"vol|atr|parkinson|garman|range|semivar|kurt", c, re.I)]
    C = np.corrcoef(X[tr], rowvar=False)
    keep = []
    for j in range(len(cols)):
        if all(abs(C[j, k2]) <= 0.9 for k2 in keep):
            keep.append(j)
    sets["nocorr"] = [cols[i] for i in keep]
    sets["all150"] = list(cols)
    return sets


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    D = pd.read_parquet(SRCD / "features154.parquet")
    meta = json.loads((SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64)
    ci = {c: i for i, c in enumerate(cols)}
    sets = build_feature_sets(D, cols)
    print(f"[입력] {D.shape} · 피쳐풀 {len(cols)} · split {D.split.value_counts().to_dict()}")
    for k, v in sets.items():
        print(f"   {k:<10}{len(v):>4}개{'  ⚠️100 초과' if len(v) > 100 else ''}")
    HGB = {"hard": (0.4457, 0.4749), "three": (0.5261, 0.5104), "wbin": (0.5247, 0.5254)}
    rows = []
    for arm in ("hard", "three", "wbin"):
        print(f"\n[{arm}] (HGB 기준선 VAL {HGB[arm][0]:.4f} / OOS {HGB[arm][1]:.4f})", flush=True)
        for sname, feats in sets.items():
            fc = [ci[c] for c in feats]
            t0 = time.time()
            r = run_cell(D, X, arm, fc, rng)
            r.update({"arm": arm, "featset": sname, "n_feat": len(fc), "sec": round(time.time() - t0, 1)})
            r["beats_hgb"] = bool(r["VAL_auc"] > HGB[arm][0] and r["OOS_auc"] > HGB[arm][1])
            r["ci_ok"] = bool(r.get("VAL_lo", 0) > 0.5 and r.get("OOS_lo", 0) > 0.5)
            rows.append(r)
            print(f"   {sname:<10}{len(fc):>4}개 VAL {r['VAL_auc']:.4f}[{r.get('VAL_lo',np.nan):.3f},{r.get('VAL_hi',np.nan):.3f}]"
                  f" · OOS {r['OOS_auc']:.4f}[{r.get('OOS_lo',np.nan):.3f},{r.get('OOS_hi',np.nan):.3f}]"
                  f" · {r['sec']}s{'  ✅CI' if r['ci_ok'] else ''}{'  ⬆HGB' if r['beats_hgb'] else ''}", flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / "cells.csv", index=False)
    cand = R[R.ci_ok | R.beats_hgb]
    print(f"\n=== CI 통과 {int(R.ci_ok.sum())} · HGB 초과 {int(R.beats_hgb.sum())} / {len(R)}셀 ===")
    print("\n[귀무] 날 블록 셔플 -- 후보 셀에만 B=15", flush=True)
    nulls = []
    for r in cand.itertuples():
        fc = [ci[c] for c in sets[r.featset]]
        nv, no = [], []
        for _ in range(15):
            rr = run_cell(D, X, r.arm, fc, rng, shuffle=True, n_est=2)
            nv.append(rr["VAL_auc"]); no.append(rr["OOS_auc"])
        nulls.append({"arm": r.arm, "featset": r.featset,
                      "VAL_p95": float(np.nanpercentile(nv, 95)), "OOS_p95": float(np.nanpercentile(no, 95)),
                      "VAL_auc": r.VAL_auc, "OOS_auc": r.OOS_auc})
        n = nulls[-1]
        n["PASS"] = bool(r.ci_ok and r.beats_hgb and r.VAL_auc > n["VAL_p95"] and r.OOS_auc > n["OOS_p95"])
        print(f"   {r.arm:<7}{r.featset:<10} 귀무 p95 V {n['VAL_p95']:.4f}/O {n['OOS_p95']:.4f}"
              f" → {'✅통과' if n['PASS'] else '미달'}", flush=True)
    if nulls:
        pd.DataFrame(nulls).to_csv(OUT / "nulls.csv", index=False)
        print(f"\n최종 통과 {sum(n['PASS'] for n in nulls)}셀")
    (OUT / "summary.json").write_text(json.dumps({"cells": rows, "nulls": nulls}, indent=2,
                                                 ensure_ascii=False, default=float))
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
