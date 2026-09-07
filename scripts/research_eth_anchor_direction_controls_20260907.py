#!/usr/bin/env python3
"""앵커 방향 예측 — **대조군 전체** (2026-09-07).

기준선(`research_eth_anchor_direction_baseline_20260907.py`)이 로짓 OOS AUC 0.589
[0.526, 0.654] 를 냈다. VAL 은 0.531 [0.481, 0.581] 로 0.5 를 포함하므로 **단일 창**이고,
이 저장소 규율상 그대로는 무효다. 여기서 대조군을 전부 깔아 그 숫자의 정체를 확정한다.

  C2 음성 대조   라벨을 `f1`(결단력 = |up-dn|/(up+dn) 상위 절반)으로 바꿔 같은 파이프라인.
                 부록 E 에서 앵커가 **못 움직이는** 축이므로 여기서 비슷한 AUC 가 나오면
                 모델이 배우는 건 방향이 아니라 파이프라인 아티팩트다.
  C3 방향 뒤집기 y -> 1-y. AUC 가 1-AUC 로 대칭 반전해야 한다(안 하면 채점 결함).
  C4 라벨 셔플   TRAIN 라벨을 **일(day) 안에서** 셔플 B=40회 -> VAL/OOS AUC 귀무 분포.
                 관측이 95백분위를 넘어야 한다.
  C5 표본 재표집 ⚠️HGB/로짓은 결정적이라 random_state 만 바꾸면 분산이 0 이다(기준선 실측).
                 대신 **TRAIN 을 일 단위 부트스트랩**해 40회 재적합 -> 추정치 안정성.
  C6 수치 취약성 float32/64 x 컬럼순서 3변형 (게이트 T2). 부호 반전 = FAIL,
                 변형 간 폭이 |추정치-0.5| 의 50% 초과 = FLAG.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

SRC = ROOT / "tmp/eth_anchor_features_20260907/features.parquet"
OUT = ROOT / "tmp/eth_anchor_controls_20260907"
B_NULL, B_BOOT = 40, 40
HP = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40,
          l2_regularization=1.0, early_stopping=False)


def logit_fit(Xtr, ytr, wtr, Xte, dtype=np.float64, order=None, seed=0):
    if order is not None:
        Xtr, Xte = Xtr[:, order], Xte[:, order]
    Xtr, Xte = Xtr.astype(dtype), Xte.astype(dtype)
    med = np.nanmedian(Xtr, axis=0); med = np.where(np.isfinite(med), med, 0.0)
    A = np.where(np.isfinite(Xtr), Xtr, med); B_ = np.where(np.isfinite(Xte), Xte, med)
    sc = StandardScaler(); A = sc.fit_transform(A); B_ = sc.transform(B_)
    m = LogisticRegression(max_iter=2000, C=0.5, random_state=seed)
    m.fit(A, ytr, sample_weight=wtr)
    return m.predict_proba(B_)[:, 1]


def hgb_fit(Xtr, ytr, wtr, Xte, seed=0):
    m = HistGradientBoostingClassifier(random_state=seed, **HP)
    m.fit(Xtr, ytr, sample_weight=wtr)
    return m.predict_proba(Xte)[:, 1]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    D = pd.read_parquet(SRC)
    meta = json.loads((ROOT / "tmp/eth_anchor_features_20260907/meta.json").read_text())
    F = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[F].to_numpy(float); W = D["w_train"].to_numpy(float)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    y = D["y2"].to_numpy(float)
    ok = np.isfinite(y)
    tr, va, oo = ok & (sp == "TRAIN"), ok & (sp == "VAL"), ok & (sp == "OOS")
    te = va | oo
    rng = np.random.default_rng(20260907)

    def run(yv, wv=None, dtype=np.float64, order=None, trmask=None):
        t = tr if trmask is None else trmask
        w = W if wv is None else wv
        p = logit_fit(X[t], yv[t], w[t], X[te], dtype, order)
        pa = np.full(len(yv), np.nan); pa[te] = p
        return {k: roc_auc_score(y[m], pa[m]) for k, m in (("VAL", va), ("OOS", oo))}

    res = {}
    base = run(y)
    print(f"기준선(로짓) VAL {base['VAL']:.4f} · OOS {base['OOS']:.4f}\n", flush=True)
    res["base"] = base

    # ---- C2 음성 대조: f1(결단력) 라벨
    print("[C2] 음성 대조 — 결단력 f1 라벨", flush=True)
    import build_eth_anchor_label_dataset_20260907 as B
    eth = B._load_kl(B.ETH_KL)
    kl = eth[eth["timestamp"] <= pd.Timestamp("2026-03-31 23:55")].reset_index(drop=True)
    Hh, Lo, C = (kl[c].to_numpy(float) for c in ("high", "low", "close"))
    kp = D["kp"].to_numpy()
    HB = 48
    w48 = np.arange(HB)[None, :] + (kp + 1)[:, None]
    e = kl["open"].to_numpy(float)[kp + 1]
    up = (Hh[w48].max(axis=1) - e) / e; dn = (e - Lo[w48].min(axis=1)) / e
    f1 = np.abs(up - dn) / np.maximum(up + dn, 1e-12)
    thr = np.nanmedian(f1[tr])
    yf = (f1 > thr).astype(float)
    c2 = run(yf)
    c2 = {k: roc_auc_score(yf[m], np.full(m.sum(), np.nan) if False else v) for k, (m, v) in
          zip(("VAL", "OOS"), ((va, None), (oo, None)))} if False else None
    p = logit_fit(X[tr], yf[tr], W[tr], X[te])
    pa = np.full(len(y), np.nan); pa[te] = p
    c2 = {k: roc_auc_score(yf[m], pa[m]) for k, m in (("VAL", va), ("OOS", oo))}
    res["C2_f1"] = c2
    print(f"     f1 라벨 AUC  VAL {c2['VAL']:.4f} · OOS {c2['OOS']:.4f}   "
          f"(방향 라벨 {base['VAL']:.4f}/{base['OOS']:.4f})", flush=True)

    # ---- C3 방향 뒤집기
    c3 = run(1.0 - y)
    c3 = {k: 1.0 - v for k, v in c3.items()}  # 뒤집힌 라벨 AUC 를 원래 축으로 환산
    p = logit_fit(X[tr], 1.0 - y[tr], W[tr], X[te])
    pa = np.full(len(y), np.nan); pa[te] = p
    c3 = {k: roc_auc_score(y[m], pa[m]) for k, m in (("VAL", va), ("OOS", oo))}
    res["C3_flip"] = c3
    print(f"[C3] 방향 뒤집기 (원 라벨 기준 AUC, 1-base 여야 정상)  VAL {c3['VAL']:.4f} (기대 {1-base['VAL']:.4f})"
          f" · OOS {c3['OOS']:.4f} (기대 {1-base['OOS']:.4f})", flush=True)

    # ---- C4 라벨 셔플 귀무 (일 안에서)
    print(f"[C4] 라벨 셔플 귀무 B={B_NULL} ...", flush=True)
    nulls = {"VAL": [], "OOS": []}
    dtr = day[tr]; ytr0 = y[tr].copy()
    for b in range(B_NULL):
        ys = ytr0.copy()
        for d in np.unique(dtr):
            m = dtr == d
            ys[m] = rng.permutation(ys[m])
        yy = y.copy(); yy[tr] = ys
        p = logit_fit(X[tr], yy[tr], W[tr], X[te])
        pa = np.full(len(y), np.nan); pa[te] = p
        for k, m in (("VAL", va), ("OOS", oo)):
            nulls[k].append(roc_auc_score(y[m], pa[m]))
    res["C4_null"] = {k: {"mean": float(np.mean(v)), "p95": float(np.percentile(v, 95)),
                          "obs_pctile": float((np.array(v) < base[k]).mean() * 100)} for k, v in nulls.items()}
    for k in ("VAL", "OOS"):
        r = res["C4_null"][k]
        print(f"     {k}: 귀무 평균 {r['mean']:.4f} · p95 {r['p95']:.4f} · 관측 백분위 {r['obs_pctile']:.0f}", flush=True)

    # ---- C5 TRAIN 일 단위 부트스트랩
    print(f"[C5] TRAIN 일 부트스트랩 B={B_BOOT} ...", flush=True)
    tri = np.flatnonzero(tr); ud = np.unique(day[tri])
    by = {d: tri[day[tri] == d] for d in ud}
    boots = {"VAL": [], "OOS": []}
    for b in range(B_BOOT):
        pick = rng.choice(len(ud), len(ud))
        idx = np.concatenate([by[ud[k]] for k in pick])
        p = logit_fit(X[idx], y[idx], W[idx], X[te])
        pa = np.full(len(y), np.nan); pa[te] = p
        for k, m in (("VAL", va), ("OOS", oo)):
            boots[k].append(roc_auc_score(y[m], pa[m]))
    res["C5_boot"] = {k: {"mean": float(np.mean(v)), "sd": float(np.std(v)),
                          "p2.5": float(np.percentile(v, 2.5)), "p97.5": float(np.percentile(v, 97.5)),
                          "frac_gt_0.5": float(np.mean(np.array(v) > 0.5))} for k, v in boots.items()}
    for k in ("VAL", "OOS"):
        r = res["C5_boot"][k]
        print(f"     {k}: {r['mean']:.4f}±{r['sd']:.4f} [{r['p2.5']:.4f},{r['p97.5']:.4f}] · >0.5 비율 {r['frac_gt_0.5']:.2f}", flush=True)

    # ---- C6 수치 취약성
    print("[C6] 수치 취약성 (float32/64 x 컬럼순서 3) ...", flush=True)
    vals = {"VAL": [], "OOS": []}
    for dt in (np.float64, np.float32):
        for oi in range(3):
            order = None if oi == 0 else rng.permutation(len(F))
            r = run(y, dtype=dt, order=order)
            for k in ("VAL", "OOS"):
                vals[k].append(r[k])
    res["C6"] = {k: {"min": float(min(v)), "max": float(max(v)),
                     "spread": float(max(v) - min(v)),
                     "sign_flip": bool(min(v) < 0.5 < max(v))} for k, v in vals.items()}
    for k in ("VAL", "OOS"):
        r = res["C6"][k]
        rel = r["spread"] / max(abs(base[k] - 0.5), 1e-9)
        print(f"     {k}: [{r['min']:.4f}, {r['max']:.4f}] 폭 {r['spread']:.4f} "
              f"(|추정치-0.5| 의 {rel*100:.0f}%) · 부호반전 {r['sign_flip']}", flush=True)

    (OUT / "controls.json").write_text(json.dumps(res, indent=2, ensure_ascii=False, default=float))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
