#!/usr/bin/env python3
"""앵커 방향 예측 — **피쳐 분석 + 기준선 모델 + 대조군 전체** (2026-09-07).

TabPFN 은 로컬에 torch 가 없어 서버에서 돌린다. 여기서는 **먼저 정직한 기준선**을 세운다 --
이 저장소 규율상 대조군 없는 헤드라인 숫자는 무효이므로, 대조군을 먼저 다 깔고 그 위에 모델을 얹는다.

## 라벨 (v2, 둘 다)
  y2  이진(먼저 닿기) -- 표본 유지, `w_train`(고유도 x 품질) 가중
  y3  3클래스(지속승 2 / 혼재 1 / 되돌림승 0) -- clean 만 뽑으면 이진, 혼재 예측도 별도 과제

## 대조군 (사전 등록)
  C1 크기 단독      atr_pct 만으로 학습 -- 이 저장소는 "크기는 배워지고 방향은 안 배워진다"
  C2 음성 대조      라벨을 `f1`(결단력, 앵커가 못 움직이는 축)로 바꿔 같은 파이프라인
  C3 방향 뒤집기    y -> 1-y. 대칭적으로 나빠져야 한다
  C4 라벨 셔플      TRAIN 라벨을 일 내 셔플 B=30회 -> VAL/OOS AUC 귀무 분포
  C5 시드 안정성    8시드
  C6 수치 취약성    float32/64 x 컬럼 순서 3변형 (게이트 T2)

## 판정 (사전 고정)
검정력 상한: OOS n=340(이진) -> AUC SE ~= 0.031. **AUC 0.56 미만은 못 가른다.**
통과 = VAL·OOS 둘 다 AUC 하한(일군집 부트) > 0.5 ∧ 셔플 귀무 95백분위 초과 ∧ C1 대비 우위.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = ROOT / "tmp/eth_anchor_features_20260907/features.parquet"
OUT = ROOT / "tmp/eth_anchor_baseline_20260907"
SEEDS = [11, 23, 47, 71, 97, 131, 173, 211]
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
          l2_regularization=1.0, early_stopping=False)
BOOT = 2000


def day_auc_ci(y, p, days, rng, B=BOOT):
    """일군집 부트스트랩 AUC CI."""
    u, inv = np.unique(days, return_inverse=True)
    g = len(u)
    if g < 3:
        return (np.nan, np.nan)
    idx_by = [np.flatnonzero(inv == k) for k in range(g)]
    out = np.empty(B)
    for b in range(B):
        pick = rng.integers(0, g, g)
        ii = np.concatenate([idx_by[k] for k in pick])
        yy = y[ii]
        out[b] = roc_auc_score(yy, p[ii]) if len(np.unique(yy)) > 1 else np.nan
    out = out[np.isfinite(out)]
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def fit_predict(Xtr, ytr, wtr, Xte, seed, model="hgb", cols=None):
    if cols is not None:
        Xtr, Xte = Xtr[:, cols], Xte[:, cols]
    if model == "hgb":
        m = HistGradientBoostingClassifier(random_state=seed, **HP)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict_proba(Xte)[:, 1]
    med = np.nanmedian(Xtr, axis=0)                    # TRAIN 중앙값으로만 대체(누수 방지)
    med = np.where(np.isfinite(med), med, 0.0)
    A_ = np.where(np.isfinite(Xtr), Xtr, med)
    B0 = np.where(np.isfinite(Xte), Xte, med)
    sc = StandardScaler()
    A = sc.fit_transform(A_)
    B_ = sc.transform(B0)
    m = LogisticRegression(max_iter=2000, C=0.5, random_state=seed)
    m.fit(A, ytr, sample_weight=wtr)
    return m.predict_proba(B_)[:, 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(SRC)
    meta = json.loads((ROOT / "tmp/eth_anchor_features_20260907/meta.json").read_text())
    FCOLS = [c for c in meta["feature_cols"] if c in D.columns]
    print(f"[1/5] 데이터 {len(D):,}행 · 피쳐 {len(FCOLS)}개", flush=True)

    D["day"] = D["timestamp"].dt.floor("D")
    sp = D["split"].to_numpy()
    ytab = {}
    ytab["y2"] = D["y2"].to_numpy(float)
    clean = D["y3"].to_numpy() != 1
    y3b = np.where(clean, (D["y3"].to_numpy() == 2).astype(float), np.nan)
    ytab["y3_clean"] = y3b
    X = D[FCOLS].to_numpy(float)
    W = D["w_train"].to_numpy(float)
    Wu = D["w_uniq"].to_numpy(float)

    print("\n[2/5] 피쳐 분석 — 단일피쳐 AUC(가중, y2) 상위/하위", flush=True)
    m_all = np.isfinite(ytab["y2"])
    rows = []
    for j, c in enumerate(FCOLS):
        v = X[:, j]; mm = m_all & np.isfinite(v)
        for w_ in ("VAL", "OOS"):
            k = mm & (sp == w_)
            if k.sum() < 100 or len(np.unique(ytab["y2"][k])) < 2:
                continue
            rows.append({"feat": c, "win": w_, "auc": roc_auc_score(ytab["y2"][k], v[k])})
    FA = pd.DataFrame(rows).pivot(index="feat", columns="win", values="auc")
    FA["mean_dev"] = ((FA - 0.5).abs()).mean(axis=1)
    FA = FA.sort_values("mean_dev", ascending=False)
    print(FA.head(8).to_string(float_format=lambda x: f"{x:.3f}"))
    print("  ...")
    print(FA.tail(3).to_string(float_format=lambda x: f"{x:.3f}"))
    FA.to_csv(OUT / "single_feature_auc.csv")

    res = {}
    rng = np.random.default_rng(20260907)
    for lab in ("y2", "y3_clean"):
        y = ytab[lab]
        ok = np.isfinite(y)
        tr = ok & (sp == "TRAIN"); va = ok & (sp == "VAL"); oo = ok & (sp == "OOS")
        print(f"\n[3/5] 모델 · 라벨 {lab}  TRAIN {tr.sum()} / VAL {va.sum()} / OOS {oo.sum()}"
              f" · 양성률 {y[tr].mean():.3f}/{y[va].mean():.3f}/{y[oo].mean():.3f}", flush=True)
        arms = {}
        for nm, cols, model in (("전체피쳐 HGB", None, "hgb"), ("전체피쳐 로짓", None, "logit"),
                                ("C1 크기단독(atr_pct)", [FCOLS.index("atr_pct")], "hgb"),
                                ("C1b 크기+votes", [FCOLS.index("atr_pct"), FCOLS.index("votes")], "hgb")):
            aucs = {"VAL": [], "OOS": []}
            for s in SEEDS:
                p = fit_predict(X[tr], y[tr], W[tr], X[va | oo], s, model, cols)
                pall = np.full(len(y), np.nan); pall[va | oo] = p
                for w_, msk in (("VAL", va), ("OOS", oo)):
                    aucs[w_].append(roc_auc_score(y[msk], pall[msk]))
            a = {w_: (float(np.mean(v)), float(np.std(v))) for w_, v in aucs.items()}
            # 대표 시드로 CI
            p = fit_predict(X[tr], y[tr], W[tr], X[va | oo], SEEDS[0], model, cols)
            pall = np.full(len(y), np.nan); pall[va | oo] = p
            ci = {w_: day_auc_ci(y[msk], pall[msk], D["day"].to_numpy()[msk], rng)
                  for w_, msk in (("VAL", va), ("OOS", oo))}
            arms[nm] = {"auc": a, "ci": ci}
            print(f"   {nm:<22} VAL {a['VAL'][0]:.4f}±{a['VAL'][1]:.4f} [{ci['VAL'][0]:.3f},{ci['VAL'][1]:.3f}]"
                  f"   OOS {a['OOS'][0]:.4f}±{a['OOS'][1]:.4f} [{ci['OOS'][0]:.3f},{ci['OOS'][1]:.3f}]", flush=True)
        res[lab] = arms
    (OUT / "arms.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    np.save(OUT / "_cache.npy", np.array([0]))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
