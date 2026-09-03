#!/usr/bin/env python3
"""L0 기반 **모델 수준** 가정 재시험 -- 정직한 L3 라벨에서 (2026-09-03).

구조 수준(깊이·대기·슬롯·팔구성·청산)은 `audit_eth_entry_l3_assumption_retest_20260903.py`
에서 끝냈다. 여기서는 모델 쪽 결정을 다시 훑는다. 전부 L0 위에서 골라진 것들이다:

  ⓐ 아키텍처   -- B9~B16이 "HGB squared_error"를 남겼고 이후 사용자가 TabPFN을 택했다
  ⓑ 피쳐 선별  -- B2/B8/B15/B16이 "161 전부"를 남겼다(선별·PCA·임베딩 전부 기각)
  ⓒ 유지율     -- τ=40bp에서 유도된 0.2037

⭐그리고 구조 재시험에서 **신호방향만(arm1)이 양팔보다 낫다**는 반전이 나왔으므로,
   모든 평가를 **양팔 / arm1-only 두 모집단에서** 낸다.

⚠️여기서 나온 최선을 채택 근거로 쓰지 않는다. 창은 소진됐다. 묻는 것은 하나다 --
   **"L0에서 고른 모델 결정이 L3에서도 성립하는가."**
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
L3D = ROOT / "tmp/eth_entry_1m_resolved_20260903/labels_1m_all.csv"
OUT = ROOT / "tmp/eth_entry_l3_model_assumptions_20260903"
DEPTH, WAIT, NSLOT, KEEP0, SUB = 3.0, 6, 4, 0.2037, 18000
W3 = ("VAL", "OOS", "HOLDOUT")


def log(m): print(f"[mdl] {m}", flush=True)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    LAB = pd.read_csv(L3D, parse_dates=["timestamp"])
    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    A = D.merge(LAB[["timestamp", "signal", "arm", "depth", "btf", "y_L3"]],
                on=["timestamp", "signal", "arm", "depth", "btf"], how="left")
    A = A[np.isfinite(A.y_L3)].reset_index(drop=True)
    dsel = ((A.depth == DEPTH) & (A.btf <= WAIT)).to_numpy()
    tr = (A.split == "TRAIN").to_numpy()
    y = A["y_L3"].to_numpy(float)
    lab = (y > 0.0040).astype(int)

    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "y_L3", "split", "timestamp",
                       "i", "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in A.columns if c.endswith("_r136")] + \
        [c for c in A.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if A[c].dtype.kind in "fiub"]))
    FE = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    Xdf = A[FE].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    Xdf = Xdf.fillna(Xdf[tr].median())
    X = Xdf.to_numpy(np.float32)
    itr = np.flatnonzero(tr); prow = np.flatnonzero(dsel)
    POP = {"양팔": dsel, "arm1만": dsel & (A.arm == 1).to_numpy()}
    log(f"모집단 {len(A):,} · 후보 {len(prow):,} · TRAIN {len(itr):,} · 피쳐 {len(FE)}")

    def evalp(score, pop_mask, keep_frac=KEEP0):
        """TRAIN에서만 임계값을 유도하고 창별 성과를 낸다."""
        thr = float(np.quantile(score[tr & pop_mask], 1 - keep_frac))
        out = {}
        for w in W3:
            m = pop_mask & (A.split == w).to_numpy() & (score > thr)
            d = A[m]
            t = slotN(d.assign(y=y[m]), NSLOT)
            out[w] = (float(np.mean(t) * 1e4) if len(t) else 0.0, int(len(t)))
        return out

    def nofilter(pop_mask):
        out = {}
        for w in W3:
            m = pop_mask & (A.split == w).to_numpy()
            d = A[m]
            t = slotN(d.assign(y=y[m]), NSLOT)
            out[w] = (float(np.mean(t) * 1e4) if len(t) else 0.0, int(len(t)))
        return out

    def show(tag, res, pop):
        print(f"{tag:26s}{pop:8s}" + "".join(f"{res[w][0]:+9.2f}(n{res[w][1]:4d})" for w in W3))

    print(f"\n{'':26s}{'모집단':8s}" + "".join(f"{w:>15s}" for w in W3))
    for pn, pm in POP.items():
        show("무필터", nofilter(pm), pn)

    # ---- ⓐ 아키텍처 ----
    log("\nⓐ 아키텍처...")
    scores = {}
    scores["HGB squared_error"] = np.mean(
        [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
         .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    scores["HGB absolute_error"] = np.mean(
        [HistGradientBoostingRegressor(random_state=s, loss="absolute_error", **HP)
         .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    scores["HGB quantile0.6"] = np.mean(
        [HistGradientBoostingRegressor(random_state=s, loss="quantile", quantile=0.6, **HP)
         .fit(X[tr], y[tr]).predict(X) for s in SEEDS], axis=0)
    scores["HGB 분류(y>40bp)"] = np.mean(
        [HistGradientBoostingClassifier(random_state=s, **HP)
         .fit(X[tr], lab[tr]).predict_proba(X)[:, 1] for s in SEEDS], axis=0)
    log("  HGB 4종 완료")

    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier
    ps = []
    for k, sd in enumerate(SEEDS):
        rs = np.random.default_rng(sd).choice(itr, size=min(SUB, len(itr)), replace=False)
        m = TabPFNClassifier(device="cuda", random_state=sd)
        m.fit(X[rs], lab[rs])
        f = np.full(len(A), -np.inf); f[prow] = m.predict_proba(X[prow])[:, 1]
        ps.append(f)
        log(f"  TabPFN 멤버{k}")
    scores["TabPFN 분류"] = np.mean(ps, axis=0)

    print(f"\n=== ⓐ 아키텍처 (유지율 {KEEP0}, 161피쳐) ===")
    print(f"{'':26s}{'모집단':8s}" + "".join(f"{w:>15s}" for w in W3))
    for nm, sc in scores.items():
        for pn, pm in POP.items():
            show(nm, evalp(sc, pm), pn)

    # ---- ⓑ 피쳐 선별 (L3 기준 중요도) ----
    log("\nⓑ 피쳐 선별...")
    m0 = HistGradientBoostingRegressor(random_state=SEEDS[0], loss="squared_error", **HP).fit(X[tr], y[tr])
    sub = np.random.default_rng(0).choice(np.flatnonzero(tr), size=min(8000, int(tr.sum())), replace=False)
    imp = permutation_importance(m0, X[sub], y[sub], n_repeats=3, random_state=0, n_jobs=4)
    order = np.argsort(-imp.importances_mean)
    print(f"\n  L3 기준 상위 10 피쳐: {[FE[i] for i in order[:10]]}")
    print(f"\n=== ⓑ 피쳐 선별 (HGB squared_error, 유지율 {KEEP0}) ===")
    print(f"{'k':>6s}{'모집단':8s}" + "".join(f"{w:>15s}" for w in W3))
    for k in (15, 30, 50, 100, len(FE)):
        cols = order[:k]
        sc = np.mean([HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                      .fit(X[np.ix_(itr, cols)], y[tr]).predict(X[:, cols]) for s in SEEDS], axis=0)
        for pn, pm in POP.items():
            r = evalp(sc, pm)
            star = " ←배포" if k == len(FE) else ""
            print(f"{k:6d}{pn:8s}" + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:4d})" for w in W3) + star)

    # ---- ⓒ 유지율 ----
    print(f"\n=== ⓒ 유지율 (HGB squared_error · TabPFN, 161피쳐) ===")
    print(f"{'모델':18s}{'유지율':>7s}{'모집단':8s}" + "".join(f"{w:>15s}" for w in W3))
    for nm in ("HGB squared_error", "TabPFN 분류"):
        for kf in (0.05, 0.10, KEEP0, 0.30, 0.50):
            for pn, pm in POP.items():
                r = evalp(scores[nm], pm, kf)
                star = " ←배포" if abs(kf - KEEP0) < 1e-9 else ""
                print(f"{nm:18s}{kf:7.3f}{pn:8s}"
                      + "".join(f"{r[w][0]:+9.2f}(n{r[w][1]:4d})" for w in W3) + star)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
