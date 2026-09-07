#!/usr/bin/env python3
"""MASHT (MultiRocket+Hydra → TabPFN) 를 방향 축에 적용 (2026-09-07, 서버 GPU).

사용자: *"MASHT를 적용해야하는데 너의 설계를 보여줘. TabPFN3도 이미 모델은 갖고 있으니
       TabPFN로 진행해도 좋아"* → *"서버로 gpu 로 돌려줘"*

논문: MASHT (arXiv 2607.19234) -- MultiRocket + Hydra 랜덤 합성곱 피쳐를 연접해
고정차원 표로 만들고 사전학습 TabPFN 에 in-context 로 먹인다. UTF-112 평균정확도
0.892 로 HIVE-COTE 2.0(0.891) 과 동률, MR-Hydra 대비 65/112 승.

## 입력
`build_eth_anchor_window_tensor_20260907.py` 가 만든 `(4755, 8채널, 48봉)` 창.
창은 `[t-47, t]` 로 **앵커 봉을 포함해서 끝난다**(미래 없음, path_atr[t]==0 트립와이어 PASS).
변환 결과 `masht.npy` = **(4743, 2784)** -- MultiRocket 2016(n_kernels=252) + Hydra 768(n_groups=16).
⚠️논문 예산은 우리 규모에서 2000(각 1000)인데 aeon 입도상 정확히 1000씩은 불가능하다
(n_kernels<252 는 내부 `n_kernels//84` 구조 때문에 ZeroDivisionError). 2784 가 가장 가까운
실현 가능점이고, 라벨을 안 보는 무작위 부분표집으로 억지로 맞추지 않았다
(논문의 "축소·선택 없음" 이 핵심 설계라서).

## 🔴TabPFN 8.5.0 의 자동 피쳐 부분표집
`FEATURE_SUBSAMPLING_IMPORTANCE_TOP_K_COUNT="auto"` 라 피쳐가 200 을 넘으면 추정기마다
**상위 150 중요피쳐 + 나머지 무작위 채움** 앙상블을 쓴다. 논문의 "축소 없음"과 다르므로
`n_estimators` 를 리포트에 기록한다(추정기 수가 곧 피쳐 커버리지다).

## 격자 (사전지정 3팔 × 3셋 = 9셀)
  `masht2784`    Rocket 피쳐만 -- 경로 모양 자체에 방향이 있나
  `dirtop20`     대조. **같은 TabPFN 으로** 돌려 "모델 교체 효과"를 분리한다
                 (TabICLv2 값은 부록 R: hard 0.5479/0.5538/0.5337)
  `masht+top20`  증분이 있나

## 필수 대조군
  P1 양성대조   라벨만 크기(range_pct > TRAIN 중앙값)로 교체. 경로 창에서 크기조차
                못 뽑으면 구현이 깨진 것이다.
  C1 날블록귀무 학습 라벨만 날 단위로 섞고 재적합. Rocket 은 랜덤 사영이라
                잡음을 맞출 수 있어 이 축에서는 특히 필수다.

## 판정
세 창(VAL/OOS/HOLDOUT_SPENT) 모두 일군집 CI 하한 > 0.5 ∧ 귀무 p95 초과.
증분은 `dirtop20` 대비 **같은 날 표집 짝비교 CI** 로만 본다.

## 사전 확률 (정직하게)
- MASHT 가 이기는 건 **단변량**(0.892 vs HC2 0.891). **다변량은 SOTA 가 아니다**
  (0.795 vs HC2 0.805). 우리는 8채널 다변량 -- 논문이 못 이긴 쪽이다.
- 같은 가설의 약한 버전인 지연 블록이 **경로 증분 0/3 전 팔**로 죽었다
  (`research_eth_anchor_lag_eval_20260907.py`, 세 창 CI 0/12).
- 반대로 될 수도 있는 근거: TabPFN 은 파라미터를 적합하지 않는 in-context 라
  3,237표본 × 2,784피쳐가 TabM 을 죽인 그 과적합과는 **다른 종류**다.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
WT = ROOT / "tmp/eth_anchor_window_tensor_20260907"
F154 = ROOT / "tmp/eth_anchor_features154_20260907"
RANK = ROOT / "tmp/eth_anchor_direction_feature_ranking_20260907/dirtop20_train.json"
OUT = ROOT / "tmp/eth_anchor_masht_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
SEED = 20260907
N_EST = 4
BOOT = 1500


def day_auc_ci(y, p, days, rng, B=BOOT):
    uniq = np.unique(days)
    if len(uniq) < 5:
        return (np.nan, np.nan)
    idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) > 1:
            out.append(roc_auc_score(y[ii], p[ii]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > B // 3 \
        else (np.nan, np.nan)


def diff_ci(y, p1, p2, days, rng, B=BOOT):
    uniq = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) > 1:
            out.append(roc_auc_score(y[ii], p1[ii]) - roc_auc_score(y[ii], p2[ii]))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))) if len(out) > B // 3 \
        else (np.nan, np.nan)


def arm_spec(I):
    y3 = I["y3"].to_numpy()
    return {
        "hard": (I["is_clean"].to_numpy(bool), (y3 == 2).astype(int), False),
        "three": (np.isfinite(y3), np.nan_to_num(y3).astype(int), True),
        "wbin": (np.isfinite(I["y_bin"].to_numpy()), np.nan_to_num(I["y_bin"].to_numpy()).astype(int), False),
    }


def fit_eval(X, I, mask, y, multi, rng, n_est=N_EST, shuffle=False, device="cuda"):
    from tabpfn import TabPFNClassifier
    sp = I["split"].to_numpy(); day = I["timestamp"].dt.floor("D").to_numpy()
    tr = mask & (sp == "TRAIN")
    yt = y.copy()
    if shuffle:
        uq = np.unique(day[tr]); perm = rng.permutation(uq)
        src = {d: np.flatnonzero(tr & (day == d)) for d in uq}
        for d, d2 in zip(uq, perm):
            if len(src[d]):
                yt[src[d]] = np.resize(y[src[d2]], len(src[d]))
    Xf = np.nan_to_num(X.astype(np.float32))
    clf = TabPFNClassifier(device=device, n_estimators=n_est, random_state=SEED,
                           ignore_pretraining_limits=True, memory_saving_mode=True)
    clf.fit(Xf[tr], yt[tr])
    out = {}
    for w in WINS:
        te = mask & (sp == w)
        if te.sum() < 30:
            continue
        p = clf.predict_proba(Xf[te])
        if multi:
            cl = y[te] != 1
            den = p[cl][:, 0] + p[cl][:, 2]
            pr = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
            yy = (y[te][cl] == 2).astype(int); dd = day[te][cl]
        else:
            pr = p[:, 1]; yy = y[te]; dd = day[te]
        if len(np.unique(yy)) < 2:
            continue
        out[f"{w}_n"] = int(len(yy)); out[f"{w}_auc"] = float(roc_auc_score(yy, pr))
        lo, hi = day_auc_ci(yy, pr, dd, rng)
        out[f"{w}_lo"], out[f"{w}_hi"] = lo, hi
        out[f"{w}_pred"], out[f"{w}_y"], out[f"{w}_day"] = pr, yy, dd
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cuda":
        free, tot = torch.cuda.mem_get_info()
        print(f"[GPU] {torch.cuda.get_device_name(0)} · 여유 {free/1e9:.1f}/{tot/1e9:.1f}GB", flush=True)

    M = np.load(WT / "masht.npy")
    v = np.load(WT / "valid.npy")
    I = pd.read_parquet(WT / "index.parquet")
    assert len(I) == len(v) and int(v.sum()) == len(M), "행 정렬 불일치"
    I = I[v].reset_index(drop=True)                      # masht.npy 는 유효행만
    D = pd.read_parquet(F154 / "features154.parquet")[v].reset_index(drop=True)
    fm = json.loads((F154 / "meta.json").read_text())
    top20 = [c for c in json.loads(RANK.read_text()) if c in D.columns][:20]
    T = D[top20].to_numpy(np.float64)
    print(f"[입력] MASHT {M.shape} · dirtop20 {T.shape} · split {I.split.value_counts().to_dict()}", flush=True)

    SETS = {"masht2784": M, "dirtop20": T, "masht+top20": np.concatenate([M, T], axis=1)}
    arms = arm_spec(I)
    rows, keep = [], {}
    print(f"\n=== 3팔 × 3셋 × 세 창 (TabPFN n_estimators={N_EST}, {dev}) ===", flush=True)
    for arm, (mask, y, multi) in arms.items():
        for sname, Xs in SETS.items():
            t0 = time.time()
            try:
                r = fit_eval(Xs, I, mask, y, multi, rng, device=dev)
            except Exception as e:
                print(f"   {arm:<7}{sname:<13} 🔴실패 {type(e).__name__}: {str(e)[:90]}", flush=True)
                continue
            rec = {"arm": arm, "featset": sname, "n_feat": Xs.shape[1],
                   "sec": round(time.time() - t0, 1)}
            for w in WINS:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
                rec[f"{w}_n"] = r.get(f"{w}_n", 0)
            rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
            rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
            rows.append(rec); keep[(arm, sname)] = r
            print(f"   {arm:<7}{sname:<13} " + " · ".join(
                f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS)
                + f" · min3 {rec['min3']:.4f} ({rec['sec']}s)", flush=True)

    print("\n=== 증분: MASHT - dirtop20 (같은 날 표집 짝비교 CI) ===", flush=True)
    inc = []
    for arm in arms:
        if (arm, "dirtop20") not in keep:
            continue
        b = keep[(arm, "dirtop20")]
        for sname in ("masht2784", "masht+top20"):
            if (arm, sname) not in keep:
                continue
            c = keep[(arm, sname)]; rec = {"arm": arm, "featset": sname}; n_gt0 = 0
            for w in WINS:
                if f"{w}_pred" not in c or f"{w}_pred" not in b:
                    continue
                assert (c[f"{w}_y"] == b[f"{w}_y"]).all(), "평가 라벨 불일치"
                lo, hi = diff_ci(c[f"{w}_y"], c[f"{w}_pred"], b[f"{w}_pred"], c[f"{w}_day"], rng)
                rec[f"{w}_d"] = c[f"{w}_auc"] - b[f"{w}_auc"]
                rec[f"{w}_lo"], rec[f"{w}_hi"] = lo, hi
                n_gt0 += int(np.isfinite(lo) and lo > 0)
            rec["n_win_gt0"] = n_gt0; inc.append(rec)
            print(f"   {arm:<7}{sname:<13} " + " · ".join(
                f"{w[:3]} Δ{rec.get(f'{w}_d', np.nan):+.4f}"
                f"[{rec.get(f'{w}_lo', np.nan):+.3f},{rec.get(f'{w}_hi', np.nan):+.3f}]"
                for w in WINS) + f"  창>0: {n_gt0}/3", flush=True)

    print("\n=== P1 양성대조: 라벨만 크기로 교체 (구현 정상성) ===", flush=True)
    rp = I["range_pct"].to_numpy()
    thr = float(np.nanmedian(rp[I["split"].to_numpy() == "TRAIN"]))
    y_sz = (rp > thr).astype(int); m_sz = np.isfinite(rp)
    p1 = fit_eval(M, I, m_sz, y_sz, False, rng, device=dev)
    atr = I["atr_pct"].to_numpy()
    for w in WINS:
        if f"{w}_auc" in p1:
            s = m_sz & (I["split"].to_numpy() == w)
            a = roc_auc_score(y_sz[s], atr[s])
            print(f"   {w:<14} MASHT {p1[f'{w}_auc']:.4f}[{p1[f'{w}_lo']:.3f}] · atr단독 {a:.4f}", flush=True)
    p1_ok = all(p1.get(f"{w}_lo", 0) > 0.5 for w in WINS if f"{w}_lo" in p1)
    print(f"   → 양성대조 {'✅통과 (구현 정상)' if p1_ok else '🔴실패 -- 아래 방향 결과는 무효'}", flush=True)

    A = pd.DataFrame(rows)
    print("\n=== 날 블록 셔플 귀무 B=8 (min3 최고 셀) ===", flush=True)
    nulls = []
    if len(A):
        r = A.loc[A.min3.idxmax()]
        mask, y, multi = arms[r.arm]
        a = {w: [] for w in WINS}
        for _ in range(8):
            rr = fit_eval(SETS[r.featset], I, mask, y, multi, rng, n_est=2, shuffle=True, device=dev)
            for w in WINS:
                if f"{w}_auc" in rr:
                    a[w].append(rr[f"{w}_auc"])
        rec = {"arm": r.arm, "featset": r.featset}
        for w in WINS:
            rec[f"{w}_p95"] = float(np.percentile(a[w], 95)) if a[w] else np.nan
            rec[f"{w}_p"] = float(np.mean(np.array(a[w]) >= r[f"{w}_auc"])) if a[w] else np.nan
        rec["PASS"] = bool(r.ci3 and all(r[f"{w}_auc"] > rec[f"{w}_p95"] for w in WINS
                                         if np.isfinite(rec[f"{w}_p95"])))
        nulls.append(rec)
        print(f"   {r.arm}/{r.featset} " + " · ".join(
            f"{w[:3]} p95 {rec[f'{w}_p95']:.4f}(p={rec[f'{w}_p']:.3f})" for w in WINS)
            + f" → {'✅PASS' if rec['PASS'] else '❌'}", flush=True)

    A.to_csv(OUT / "masht_eval.csv", index=False)
    pd.DataFrame(inc).to_csv(OUT / "increments.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "nulls.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(
        {"n_feat_masht": int(M.shape[1]), "multirocket": 2016, "hydra": 768,
         "n_estimators": N_EST, "device": dev, "positive_control_pass": bool(p1_ok),
         "ci3_pass": int(A.ci3.sum()) if len(A) else 0, "cells": len(A),
         "best_min3": float(A.min3.max()) if len(A) else None,
         "budget_note": "논문 예산 2000(각1000) vs 실제 2784 -- aeon 입도 한계, 무작위 부분표집 안 함",
         "tabpfn_auto_feature_subsampling": "200열 초과 시 추정기마다 상위150+무작위 채움"},
        indent=1, ensure_ascii=False))
    print("\n" + "=" * 92, flush=True)
    print(f"세 창 CI 통과 {int(A.ci3.sum()) if len(A) else 0}/{len(A)} · "
          f"min3 최고 {A.min3.max():.4f}" if len(A) else "결과 없음", flush=True)
    print(f"증분 두 창 이상 CI>0: {sum(1 for x in inc if x['n_win_gt0'] >= 2)}개", flush=True)
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
