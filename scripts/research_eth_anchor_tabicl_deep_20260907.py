#!/usr/bin/env python3
"""TabICLv2 심화 — **세 창 전수 재평가 + 크기 축 정식 검정** (2026-09-07).

사용자: *"TabICLv2는 이 정도면 써도 되지 않아? 좀 더 TabICLv2를 파보자"*

## 두 갈래로 판다
**A. 방향 — 27셀 전부를 세 창에서** (부록 O 는 VAL/OOS 로 고른 **1셀만** 세 번째 창에서 봤다).
   27셀 각각을 HOLDOUT_SPENT(2026-04-01~07-31, n=462)에서도 평가해 **세 창 일관성 지도**를 만든다.
   VAL·OOS 로 고르는 절차 자체가 우연을 만들 수 있으니, 세 창 전부를 놓고 본다.

**B. 크기 — TabICLv2 가 실제로 강한 축을 정식 검정** (지금까지 **양성대조로만** 썼다).
   부록 O §O2: 크기 라벨 VAL AUC **0.758** vs HGB 0.650 · raw `atr_pct` 0.708.
   **모델이 처음으로 atr_pct 를 넘었다.** 그렇다면 그 자체가 산출물이 될 수 있다:
     B1 세 창 전부에서 되는가
     B2 `atr_pct` **단독 대비 증분**이 있는가 (배포 사이징이 이미 atr_pct 를 쓴다 -- 그걸 넘어야 의미)
     B3 분류 대신 **회귀**(TabICLRegressor)로 `range_pct` 를 직접 맞히면
     B4 날 블록 셔플 귀무

## 판정
A: 세 창 모두 CI 하한 > 0.5 인 셀이 있는가 (부록 O 의 "두 창만" 기준을 강화)
B: 세 창 모두 `atr_pct` 단독을 유의하게 초과하는가 (일군집 짝비교 CI)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from tabicl import TabICLClassifier

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tabicl_20260907 as T  # noqa: E402

OUT = ROOT / "tmp/eth_anchor_tabicl_deep_20260907"
WINS = ("VAL", "OOS", "HOLDOUT_SPENT")
N_EST = 4


def run3(D, X, arm, fcols, rng, n_est=N_EST, y_override=None, multi_override=None):
    """세 창 전부에서 평가."""
    m, y, multi = T.arm_spec(D, arm)
    if y_override is not None:
        y = y_override; m = np.isfinite(y); multi = bool(multi_override)
        y = np.nan_to_num(y).astype(int)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    tr = m & (sp == "TRAIN")
    Xf = np.nan_to_num(X[:, fcols].astype(np.float32))
    clf = TabICLClassifier(device="cpu", n_estimators=n_est, random_state=T.SEED, verbose=False)
    clf.fit(Xf[tr], y[tr])
    out = {}
    for w in WINS:
        te = m & (sp == w)
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
        out[f"{w}_n"] = int(len(yy))
        out[f"{w}_auc"] = float(roc_auc_score(yy, pr)) if len(np.unique(yy)) > 1 else np.nan
        lo, hi = T.day_auc_ci(yy, pr, dd, rng)
        out[f"{w}_lo"], out[f"{w}_hi"] = lo, hi
        out[f"{w}_pred"] = pr; out[f"{w}_y"] = yy; out[f"{w}_day"] = dd
    return out


def diff_ci(y, p1, p2, days, rng, B=1500):
    """같은 날 표집을 공유한 AUC 차이 CI."""
    uniq = np.unique(days); idx = {d: np.flatnonzero(days == d) for d in uniq}
    out = []
    for _ in range(B):
        ii = np.concatenate([idx[d] for d in rng.choice(uniq, len(uniq), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        out.append(roc_auc_score(y[ii], p1[ii]) - roc_auc_score(y[ii], p2[ii]))
    if len(out) < B // 3:
        return (np.nan, np.nan)
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(T.SRCD / "features154.parquet")
    meta = json.loads((T.SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    X = D[cols].to_numpy(np.float64); ci = {c: i for i, c in enumerate(cols)}
    sets = T.build_feature_sets(D, cols)
    sp = D["split"].to_numpy()
    print(f"[입력] {D.shape} · split {D.split.value_counts().to_dict()}", flush=True)

    # ---------------- A. 방향: 27셀 전부를 세 창에서
    print("\n=== A. 방향 27셀 × 세 창 ===", flush=True)
    rows = []
    for arm in ("hard", "three", "wbin"):
        for sname, feats in sets.items():
            fc = [ci[c] for c in feats]
            t0 = time.time()
            r = run3(D, X, arm, fc, rng)
            rec = {"arm": arm, "featset": sname, "n_feat": len(fc), "sec": round(time.time() - t0, 1)}
            for w in WINS:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
            rec["min3"] = np.nanmin([rec[f"{w}_auc"] for w in WINS])
            rec["ci3"] = bool(all(rec[f"{w}_lo"] > 0.5 for w in WINS if np.isfinite(rec[f"{w}_lo"])))
            rows.append(rec)
            print(f"   {arm:<7}{sname:<10} " + " · ".join(
                f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS)
                + f" · min3 {rec['min3']:.4f}{'  ✅세창CI' if rec['ci3'] else ''}", flush=True)
    A = pd.DataFrame(rows); A.to_csv(OUT / "direction_3windows.csv", index=False)
    print(f"\n   세 창 CI 통과: {int(A.ci3.sum())}/{len(A)} · min3 최고 {A.min3.max():.4f}"
          f" ({A.loc[A.min3.idxmax(),'arm']}/{A.loc[A.min3.idxmax(),'featset']})")
    for w1, w2 in (("VAL_auc", "OOS_auc"), ("OOS_auc", "HOLDOUT_SPENT_auc"), ("VAL_auc", "HOLDOUT_SPENT_auc")):
        s = A[[w1, w2]].dropna()
        print(f"   창 간 순위 상관 rho({w1[:3]}, {w2[:3]}) = {spearmanr(s[w1], s[w2])[0]:+.3f}")

    # ---------------- B. 크기 축 정식 검정
    print("\n=== B. 크기 축 (TabICLv2 가 실제로 강한 곳) ===", flush=True)
    med = float(np.nanmedian(D.loc[sp == "TRAIN", "range_pct"]))
    ysz = (D["range_pct"].to_numpy() > med).astype(int)
    ai = [ci["atr_pct"]] if "atr_pct" in ci else [0]
    brows = []
    for sname in ("dirtop40", "perm40", "volrange", "all150"):
        fc = [ci[c] for c in sets[sname]]
        r = run3(D, X, "wbin", fc, rng, y_override=ysz.astype(float), multi_override=False)
        rec = {"featset": sname, "n_feat": len(fc)}
        for w in WINS:
            rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan); rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
        brows.append((rec, r))
        print(f"   {sname:<10} " + " · ".join(
            f"{w[:3]} {rec[f'{w}_auc']:.4f}[{rec[f'{w}_lo']:.3f}]" for w in WINS), flush=True)
    # atr 단독 대비 증분
    ra = run3(D, X, "wbin", ai, rng, y_override=ysz.astype(float), multi_override=False)
    print(f"   {'atr 단독':<10} " + " · ".join(
        f"{w[:3]} {ra.get(f'{w}_auc',np.nan):.4f}[{ra.get(f'{w}_lo',np.nan):.3f}]" for w in WINS))
    best_rec, best_r = max(brows, key=lambda x: np.nanmin([x[0][f"{w}_auc"] for w in WINS]))
    print(f"\n   ⭐{best_rec['featset']} 의 atr 단독 대비 증분 (같은 날 표집 짝비교 CI):")
    inc = {}
    for w in WINS:
        if f"{w}_pred" not in best_r or f"{w}_pred" not in ra:
            continue
        lo, hi = diff_ci(best_r[f"{w}_y"], best_r[f"{w}_pred"], ra[f"{w}_pred"], best_r[f"{w}_day"], rng)
        d = best_r[f"{w}_auc"] - ra[f"{w}_auc"]
        inc[w] = {"diff": d, "lo": lo, "hi": hi}
        print(f"      {w:<14} Δ {d:+.4f} [{lo:+.4f}, {hi:+.4f}] → {'초과' if lo > 0 else '미달'}")
    # 귀무
    print(f"\n   크기 축 날 블록 귀무 B=15 ({best_rec['featset']})", flush=True)
    fc = [ci[c] for c in sets[best_rec["featset"]]]
    day = D["timestamp"].dt.floor("D").to_numpy(); tr = sp == "TRAIN"
    nl = {w: [] for w in WINS}
    for _ in range(15):
        ys = ysz.copy(); days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            if len(src[d]):
                ys[src[d]] = np.resize(ysz[src[d2]], len(src[d]))
        rr = run3(D, X, "wbin", fc, rng, n_est=2, y_override=ys.astype(float), multi_override=False)
        for w in WINS:
            nl[w].append(rr.get(f"{w}_auc", np.nan))
    verdict = {}
    for w in WINS:
        a = np.array([x for x in nl[w] if np.isfinite(x)])
        obs = best_rec[f"{w}_auc"]
        verdict[w] = {"obs": obs, "lo": best_rec[f"{w}_lo"], "null_p95": float(np.percentile(a, 95)),
                      "p": float(np.mean(a >= obs)), "inc_lo": inc.get(w, {}).get("lo", np.nan)}
        verdict[w]["pass"] = bool(best_rec[f"{w}_lo"] > 0.5 and obs > np.percentile(a, 95)
                                  and inc.get(w, {}).get("lo", -9) > 0)
        print(f"      {w:<14} 관측 {obs:.4f} vs 귀무 p95 {np.percentile(a,95):.4f} (p={verdict[w]['p']:.3f})"
              f" · atr증분 하한 {verdict[w]['inc_lo']:+.4f} → {'통과' if verdict[w]['pass'] else '미달'}")
    print(f"\n=== 크기 축 세 창 모두 통과: {'✅ 예' if all(v['pass'] for v in verdict.values()) else '❌ 아니오'} ===")
    (OUT / "summary.json").write_text(json.dumps(
        {"direction_ci3": int(A.ci3.sum()), "size_best": best_rec["featset"],
         "size_verdict": verdict, "size_increment": inc}, indent=2, ensure_ascii=False, default=float))
    print(f"저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
