#!/usr/bin/env python3
"""**DC 엔지니어링 150피쳐**(감사 통과분)로 3팔 x 튜닝 x 피쳐선택 (2026-09-07).

사용자: *"150여개 피쳐가 있자나"*

부록 M/N 과 **완전히 같은 절차** -- 바뀌는 것은 피쳐 출처뿐이다.
  입력 `tmp/eth_anchor_features154_20260907/features154.parquet` (4,755 x 238)
       = 앵커·라벨 + 감사 통과 150피쳐(leak_likely 0 · pass 150 · 제외 4)
  TRAIN 3,237 / VAL 612 / OOS 444 -- 199 자체제작본(TRAIN 2,655)보다 **22% 많다**
       (154세트는 2024-01-01 부터, 199본은 bookdepth 때문에 2024-04-20 부터)
  튜닝·선택은 TRAIN 내부 purged K-fold 에서만 · VAL/OOS 최종 1회 · 날 블록 귀무
피쳐셋: all150 / perm20 / perm40 / perm80 / atr(대조, `atr_pct`)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import research_eth_anchor_direction_tuned_20260907 as TU  # noqa: E402

SRCD = ROOT / "tmp/eth_anchor_features154_20260907"
OUT = ROOT / "tmp/eth_anchor_f154_eval_20260907"
TU.HP_GRID = TU.HP_GRID[:6]
TU.SEEDS = [11, 23]
TU.NULL_B = 15
ATR_COL = "atr_pct"


def feature_sets(D, cols, F, arm, rng):
    m, y, w, multi = TU.arm_spec(D, arm)
    tr = m & (D["split"].to_numpy() == "TRAIN")
    Xt = F[tr]
    sets = {"all150": list(range(len(cols)))}
    if ATR_COL in cols:
        sets["atr"] = [cols.index(ATR_COL)]
    idx = D["bar_idx"].to_numpy()[tr]
    o = np.argsort(idx); cut = int(len(o) * 0.75)
    a, b = o[:cut], o[cut + TU.EMBARGO:]
    if len(b) > 50:
        mdl = HistGradientBoostingClassifier(random_state=0, **TU.HP_GRID[0])
        mdl.fit(Xt[a], y[tr][a], sample_weight=w[tr][a])
        pi = permutation_importance(mdl, Xt[b], y[tr][b], n_repeats=3, random_state=0,
                                    scoring=("roc_auc_ovr" if multi else "roc_auc"))
        rank = np.argsort(-pi.importances_mean)
        for k in (20, 40, 80):
            sets[f"perm{k}"] = sorted(rank[:k].tolist())
    return sets


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260907)
    D = pd.read_parquet(SRCD / "features_154.parquet") if (SRCD / "features_154.parquet").exists() \
        else pd.read_parquet(SRCD / "features154.parquet")
    meta = json.loads((SRCD / "meta.json").read_text())
    cols = [c for c in meta["feature_cols"] if c in D.columns]
    if ATR_COL in D.columns and ATR_COL not in cols:
        cols = cols + [ATR_COL]
    F = D[cols].to_numpy(np.float64)
    print(f"[입력] {D.shape} · 피쳐 {len(cols)} · split {D.split.value_counts().to_dict()}", flush=True)
    TU.feature_sets = feature_sets
    finals, nulls, allcv = [], [], []
    for arm in ("hard", "three", "wbin"):
        print(f"\n[{arm}] TRAIN 내부 CV ...", flush=True)
        cv, fs = TU.cv_search(D, F, cols, arm, rng)
        cv["arm"] = arm; allcv.append(cv)
        best = cv.loc[cv.cv_mean.idxmax()]
        print(f"   최적: {best.feat}({int(best.n_feat)}개) hp={int(best.hp)} · CV {best.cv_mean:.4f}", flush=True)
        r = TU.final_eval(D, F, arm, fs[best.feat], TU.HP_GRID[int(best.hp)], rng)
        r.update({"feat": best.feat, "hp": int(best.hp), "cv_mean": float(best.cv_mean)})
        finals.append(r)
        print(f"   최종: VAL {r['VAL_auc']:.4f} [{r.get('VAL_lo',np.nan):.3f},{r.get('VAL_hi',np.nan):.3f}] "
              f"· OOS {r['OOS_auc']:.4f} [{r.get('OOS_lo',np.nan):.3f},{r.get('OOS_hi',np.nan):.3f}]", flush=True)
        nv, no = [], []
        for _ in range(TU.NULL_B):
            rr = TU.final_eval(D, F, arm, fs[best.feat], TU.HP_GRID[int(best.hp)], rng, shuffle=True, seeds=TU.SEEDS[:1])
            nv.append(rr["VAL_auc"]); no.append(rr["OOS_auc"])
        nulls.append({"arm": arm, "VAL_p95": float(np.nanpercentile(nv, 95)), "OOS_p95": float(np.nanpercentile(no, 95))})
        print(f"   귀무 p95: V {nulls[-1]['VAL_p95']:.4f} / O {nulls[-1]['OOS_p95']:.4f}", flush=True)
    CV = pd.concat(allcv, ignore_index=True); CV.to_csv(OUT / "cv_grid.csv", index=False)
    pd.DataFrame(finals).to_csv(OUT / "final.csv", index=False)
    pd.DataFrame(nulls).to_csv(OUT / "nulls.csv", index=False)
    print("\n" + "=" * 104)
    print(f"{'팔':<7}{'피쳐셋':<9}{'n':>5}{'CV':>8}{'VAL AUC':>9}{'[CI]':>18}{'OOS AUC':>9}{'[CI]':>18}{'귀무p95(V/O)':>16}")
    for r, n in zip(finals, nulls):
        vci = "[{:.3f}, {:.3f}]".format(r.get("VAL_lo", np.nan), r.get("VAL_hi", np.nan))
        oci = "[{:.3f}, {:.3f}]".format(r.get("OOS_lo", np.nan), r.get("OOS_hi", np.nan))
        nul = "{:.3f}/{:.3f}".format(n["VAL_p95"], n["OOS_p95"])
        print(f"{r['arm']:<7}{r['feat']:<9}{r['n_feat']:>5}{r['cv_mean']:>8.4f}{r['VAL_auc']:>9.4f}{vci:>18}"
              f"{r['OOS_auc']:>9.4f}{oci:>18}{nul:>16}")
    print("\n3클래스 혼재:", {k: round(v, 4) for k, v in finals[1].items() if "mixed" in k})
    print("\n=== 피쳐셋별 CV 최고 ===")
    for arm in ("hard", "three", "wbin"):
        sub = CV[CV.arm == arm].groupby("feat").cv_mean.max().sort_values(ascending=False)
        print(f"   {arm:<7} " + " · ".join(f"{k} {v:.4f}" for k, v in sub.items()))
    (OUT / "summary.json").write_text(json.dumps({"finals": finals, "nulls": nulls}, indent=2, ensure_ascii=False, default=float))
    print(f"\n저장: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
