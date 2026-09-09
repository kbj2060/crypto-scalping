#!/usr/bin/env python3
"""앵커 방향 예측 — **3팔 x 하이퍼파라미터 튜닝 x 피쳐선택** 병렬 (2026-09-07).

사용자: *"위 3가지를 모두 병렬적으로 진행해보자. 이제 하이퍼파라미터 튜닝하고 피쳐선택까지."*

## ⭐제1원칙 — 튜닝·선택은 **TRAIN 내부에서만**
VAL 로 튜닝하면 VAL 이 오염돼 "두 창 통과"라는 판정 자체가 무의미해진다.
여기서는 TRAIN 을 **시간순 purged K-fold**(fold 경계에 라벨 창 H=48봉 엠바고)로 잘라
그 안에서만 HP·피쳐를 고르고, **VAL/OOS 는 최종 1회 평가에만** 쓴다.
피쳐선택도 fold 안에서만 라벨을 본다 (블랙리스트 누수 사고: `move_atr_mult` 가 새어 AUC 1.0000).

## 팔 3종 (사용자가 고른 세 옵션 전부)
  ① hard   y3 in {0,2} 이진 · w_uniq 가중         TRAIN  663 / VAL 153 / OOS 107
  ② three  y3 3클래스 · w_uniq 가중               TRAIN 2655 / VAL 612 / OOS 444
           평가 2축 분리: (a) 혼재 식별력 P(혼재) · (b) 방향 식별력 깨끗한 건의 P(2)/(P(0)+P(2))
  ③ wbin   y_bin 이진 · w_final(고유도 x 품질) 가중 TRAIN 1982 / VAL 450 / OOS 340

## 피쳐셋 5종 (팔별로 TRAIN CV 가 고른다)
  all41 · nocorr(|rho|>0.9 제거) · perm10 · perm20(TRAIN CV 순열중요도 상위) · atr(대조)

## HP 그리드 12셀
  learning_rate {0.03,0.05,0.10} x max_leaf_nodes {7,15,31} x min_samples_leaf {20,60} 중 12조합

## 판정 (사전 고정)
검정력: ③ OOS n=340 -> AUC SE ~0.031(0.56 미만 무의미) · ① n=107 -> ~0.055(0.61 미만).
통과 = VAL·OOS 둘 다 일군집 CI 하한 > 0.5 ∧ 일 내 셔플 귀무 p95 초과.
⭐추가 진단 **이월성**: TRAIN CV 점수와 VAL/OOS 점수의 Spearman rho -- 튜닝이 밖으로 이어지는가.
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_baseline_eval_20260907 as BE  # noqa: E402

SRC = BE.SRC
OUT = ROOT / "tmp/eth_anchor_direction_tuned_20260907"
SEEDS = [11, 23, 47]
FINAL_SEEDS = BE.SEEDS
N_FOLDS, EMBARGO = 4, 48
NULL_B = 20
HP_GRID = [dict(learning_rate=lr, max_leaf_nodes=ln, min_samples_leaf=ms, max_iter=300, l2_regularization=1.0)
           for lr, ln, ms in itertools.product((0.03, 0.05, 0.10), (7, 15, 31), (20, 60))][:12]


def arm_spec(D, arm):
    y3 = D["y3"].to_numpy()
    if arm == "hard":
        m = D["is_clean"].to_numpy(); y = (y3 == 2).astype(float); w = D["w_uniq"].to_numpy(); multi = False
    elif arm == "wbin":
        y = D["y_bin"].to_numpy(); m = np.isfinite(y); w = D["w_final"].to_numpy(); multi = False
    else:
        m = np.isfinite(y3); y = y3.copy(); w = D["w_uniq"].to_numpy(); multi = True
    w = np.where(np.isfinite(w) & (w > 0), w, 1e-6)
    return m, y, w, multi


def score(y, p, multi, y3=None):
    """이진: AUC. 3클래스: (혼재 AUC, 방향 AUC) 중 **방향**을 선택 점수로 쓴다."""
    if not multi:
        return roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan
    cl = y != 1
    if cl.sum() < 20 or len(np.unique(y[cl] == 2)) < 2:
        return np.nan
    den = p[cl][:, 0] + p[cl][:, 2]
    return roc_auc_score((y[cl] == 2).astype(int), np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5))


def fit_pred(Xtr, ytr, wtr, Xte, hp, seed, multi):
    m = HistGradientBoostingClassifier(random_state=seed, **hp)
    m.fit(Xtr, ytr, sample_weight=wtr)
    p = m.predict_proba(Xte)
    return p if multi else p[:, 1]


def purged_folds(idx, day, k=N_FOLDS, embargo_bars=EMBARGO):
    """시간순 K-fold + fold 경계 엠바고(라벨 창 길이만큼 앞뒤 제거)."""
    order = np.argsort(idx)
    parts = np.array_split(order, k)
    out = []
    for j in range(k):
        te = parts[j]
        lo, hi = idx[te].min() - embargo_bars, idx[te].max() + embargo_bars
        tr = np.array([o for o in order if idx[o] < lo or idx[o] > hi])
        if len(tr) > 100 and len(te) > 30:
            out.append((tr, te))
    return out


def feature_sets(D, cols, F, arm, rng):
    """TRAIN 안에서만 만든 피쳐셋 5종. 순열중요도도 TRAIN CV fold 안에서만 라벨을 본다."""
    m, y, w, multi = arm_spec(D, arm)
    tr = m & (D["split"].to_numpy() == "TRAIN")
    Xt = F[tr]
    sets = {"all41": list(range(len(cols))), "atr": [cols.index("atr_pct")]}
    # nocorr: |rho|>0.9 쌍에서 뒤 열 제거 (라벨 미사용)
    C = np.corrcoef(np.nan_to_num(Xt), rowvar=False)
    keep = []
    for j in range(len(cols)):
        if all(abs(C[j, k]) <= 0.9 for k in keep):
            keep.append(j)
    sets["nocorr"] = keep
    # perm: TRAIN 내부 단일 분할로 순열중요도
    idx = D["bar_idx"].to_numpy()[tr]
    o = np.argsort(idx); cut = int(len(o) * 0.75)
    a, b = o[:cut], o[cut + EMBARGO:]
    if len(b) > 50:
        mdl = HistGradientBoostingClassifier(random_state=0, **HP_GRID[0])
        mdl.fit(Xt[a], y[tr][a], sample_weight=w[tr][a])
        sc = "roc_auc_ovr" if multi else "roc_auc"
        pi = permutation_importance(mdl, Xt[b], y[tr][b], n_repeats=5, random_state=0, scoring=sc)
        rank = np.argsort(-pi.importances_mean)
        sets["perm10"] = sorted(rank[:10].tolist()); sets["perm20"] = sorted(rank[:20].tolist())
    return sets


def cv_search(D, F, cols, arm, rng):
    m, y, w, multi = arm_spec(D, arm)
    tr = m & (D["split"].to_numpy() == "TRAIN")
    idx = D["bar_idx"].to_numpy()[tr]
    day = D["timestamp"].dt.floor("D").to_numpy()[tr]
    Xt, yt, wt = F[tr], y[tr], w[tr]
    folds = purged_folds(idx, day)
    fs = feature_sets(D, cols, F, arm, rng)
    rows = []
    for fname, fcols in fs.items():
        for hi, hp in enumerate(HP_GRID):
            s = []
            for a, b in folds:
                try:
                    p = np.mean([fit_pred(Xt[a][:, fcols], yt[a], wt[a], Xt[b][:, fcols], hp, sd, multi)
                                 for sd in SEEDS], axis=0)
                    s.append(score(yt[b], p, multi))
                except Exception:                                   # noqa: BLE001
                    s.append(np.nan)
            s = np.array(s, float)
            rows.append({"arm": arm, "feat": fname, "hp": hi, "n_feat": len(fcols),
                         "cv_mean": float(np.nanmean(s)), "cv_sd": float(np.nanstd(s)),
                         "cv_folds": int(np.isfinite(s).sum())})
    return pd.DataFrame(rows), fs


def final_eval(D, F, arm, fcols, hp, rng, shuffle=False, seeds=None):
    m, y, w, multi = arm_spec(D, arm)
    sp = D["split"].to_numpy(); day = D["timestamp"].dt.floor("D").to_numpy()
    tr = m & (sp == "TRAIN")
    yt = y.copy()
    if shuffle:
        # 🔴2026-09-07: **날 블록 셔플**을 쓴다. 일 내 셔플은 라벨이 날 단위로 뭉친 축(특히 혼재)에서
        # 라벨을 거의 안 바꿔 귀무가 0.74 로 부풀었다(날 블록은 0.49). 귀무는 라벨의 군집 구조를 깨야 한다.
        days = np.unique(day[tr]); perm = rng.permutation(days)
        src = {d: np.flatnonzero(tr & (day == d)) for d in days}
        for d, d2 in zip(days, perm):
            dst = src[d]
            if len(dst):
                yt[dst] = np.resize(y[src[d2]], len(dst))
    out = {"arm": arm, "n_feat": len(fcols)}
    for nm in ("VAL", "OOS"):
        te = m & (sp == nm)
        p = np.mean([fit_pred(F[tr][:, fcols], yt[tr], w[tr], F[te][:, fcols], hp, sd, multi)
                     for sd in (seeds or FINAL_SEEDS)], axis=0)
        out[f"{nm}_n"] = int(te.sum())
        out[f"{nm}_auc"] = score(y[te], p, multi)
        if multi:
            a_y = (y[te] == 1).astype(int)
            out[f"{nm}_mixed_auc"] = float(roc_auc_score(a_y, p[:, 1]))
            lo, hi = BE.day_auc_ci(a_y, p[:, 1], day[te], rng)
            out[f"{nm}_mixed_lo"] = lo
            cl = y[te] != 1
            den = p[cl][:, 0] + p[cl][:, 2]
            bp = np.where(den > 0, p[cl][:, 2] / np.maximum(den, 1e-12), 0.5)
            lo, hi = BE.day_auc_ci((y[te][cl] == 2).astype(int), bp, day[te][cl], rng)
            out[f"{nm}_lo"], out[f"{nm}_hi"] = lo, hi
        else:
            lo, hi = BE.day_auc_ci(y[te], p, day[te], rng)
            out[f"{nm}_lo"], out[f"{nm}_hi"] = lo, hi
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(SRC)
    cols = BE.feature_cols(D)
    F = D[cols].to_numpy(np.float64)
    print(f"[입력] {D.shape} · 피쳐 {len(cols)} · HP {len(HP_GRID)}셀 · fold {N_FOLDS}(엠바고 {EMBARGO}봉)", flush=True)
    allcv, finals, nulls = [], [], []
    for arm in ("hard", "three", "wbin"):
        print(f"\n[{arm}] TRAIN 내부 CV 탐색 ...", flush=True)
        cv, fs = cv_search(D, F, cols, arm, rng)
        cv["arm"] = arm; allcv.append(cv)
        best = cv.loc[cv.cv_mean.idxmax()]
        print(f"   최적: feat={best.feat}({int(best.n_feat)}개) hp={int(best.hp)} · CV {best.cv_mean:.4f}±{best.cv_sd:.4f}", flush=True)
        fcols = fs[best.feat]; hp = HP_GRID[int(best.hp)]
        r = final_eval(D, F, arm, fcols, hp, rng); r.update({"feat": best.feat, "hp": int(best.hp),
                                                            "cv_mean": float(best.cv_mean)})
        finals.append(r); print(f"   최종: {r}", flush=True)
        print(f"   셔플 귀무 B={NULL_B} ...", flush=True)
        nv, no = [], []
        for _ in range(NULL_B):
            rr = final_eval(D, F, arm, fcols, hp, rng, shuffle=True, seeds=SEEDS[:1])
            nv.append(rr.get("VAL_auc", np.nan)); no.append(rr.get("OOS_auc", np.nan))
        nulls.append({"arm": arm,
                      "VAL_p95": float(np.nanpercentile(nv, 95)), "VAL_mean": float(np.nanmean(nv)),
                      "OOS_p95": float(np.nanpercentile(no, 95)), "OOS_mean": float(np.nanmean(no))})
        print(f"   귀무: {nulls[-1]}", flush=True)
    CV = pd.concat(allcv, ignore_index=True); CV.to_csv(OUT / "cv_grid.csv", index=False)
    FN = pd.DataFrame(finals); FN.to_csv(OUT / "final.csv", index=False)
    NL = pd.DataFrame(nulls); NL.to_csv(OUT / "nulls.csv", index=False)

    print("\n" + "=" * 108)
    print(f"{'팔':<7}{'피쳐셋':<9}{'n':>4}{'HP':>3}{'CV':>8}{'VAL AUC':>9}{'[CI]':>18}{'OOS AUC':>9}{'[CI]':>18}{'귀무p95(V/O)':>16}")
    for r, n in zip(finals, nulls):
        vci = "[{:.3f}, {:.3f}]".format(r.get("VAL_lo", np.nan), r.get("VAL_hi", np.nan))
        oci = "[{:.3f}, {:.3f}]".format(r.get("OOS_lo", np.nan), r.get("OOS_hi", np.nan))
        nul = "{:.3f}/{:.3f}".format(n["VAL_p95"], n["OOS_p95"])
        print(f"{r['arm']:<7}{r['feat']:<9}{r['n_feat']:>4}{r['hp']:>3}{r['cv_mean']:>8.4f}"
              f"{r['VAL_auc']:>9.4f}{vci:>18}{r['OOS_auc']:>9.4f}{oci:>18}{nul:>16}")
    print("\n3클래스 혼재 식별력:", {k: round(v, 4) for k, v in finals[1].items() if "mixed" in k})
    print("\n=== ⭐이월성: TRAIN CV 점수가 VAL/OOS 로 이어지는가 (팔별 그리드 전체) ===")
    for arm in ("hard", "three", "wbin"):
        sub = CV[CV.arm == arm].nlargest(8, "cv_mean")
        ev = [final_eval(D, F, arm, feature_sets(D, cols, F, arm, rng)[r.feat], HP_GRID[int(r.hp)], rng,
                         seeds=SEEDS[:1]) for r in sub.itertuples()]
        vo = np.array([e["OOS_auc"] for e in ev]); cvv = sub.cv_mean.to_numpy()
        ok = np.isfinite(vo) & np.isfinite(cvv)
        rho = spearmanr(cvv[ok], vo[ok])[0] if ok.sum() > 3 else np.nan
        print(f"   {arm:<7} 상위8셀 rho(CV, OOS) = {rho:+.3f} · OOS 범위 [{np.nanmin(vo):.4f}, {np.nanmax(vo):.4f}]")
    (OUT / "summary.json").write_text(json.dumps({"finals": finals, "nulls": nulls}, indent=2, ensure_ascii=False, default=float))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
