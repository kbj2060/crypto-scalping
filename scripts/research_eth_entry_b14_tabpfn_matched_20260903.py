#!/usr/bin/env python3
"""B14: TabPFN 분류 정밀 재검정 -- 동수 시드 + 데이터량 분리 + 대조군 (2026-09-03, 서버).

B12에서 TabPFN 분류가 1차 관문(양 창 승리)을 통과했다(VAL +5.29 / OOS +0.42 / HOLDOUT +8.70).
그런데 2차 기준을 판정할 수 없었다 -- 멤버 4개 vs 시드 5개로 **개수도 성격도 달라** 짝지을 수 없고,
OOS 우위 +0.42bp가 양쪽 산포(7.0 / 11.6)에 완전히 묻힌다.

재설계
------
  ① **동수(5)** -- TabPFN 멤버 5개, HGB 시드 5개
  ② ⭐**데이터량 분리** -- HGB를 **TabPFN과 똑같은 18k 서브샘플**로도 학습한다.
       HGB-full vs HGB-sub  = 데이터량의 값어치
       HGB-sub  vs TabPFN   = **동일 데이터에서의 순수 학습기 차이**
     이게 없으면 "학습기가 낫다"와 "데이터가 많다"를 구분할 수 없다.
  ③ **비짝(unpaired) 비교** -- 멤버와 시드는 자연스러운 짝이 없으므로 두 표본으로 보고
     순열검정으로 분포 이동을 잰다.
  ④ **대조군 3종 재통과** -- 무작위 필터 · 시간블록 부트스트랩 (TabPFN 앙상블로)

⭐사전등록: TabPFN이 채택되려면 ①앙상블이 VAL·OOS 양 창 승리 ②순열검정 p<0.05로 분포 이동
③대조군 3종 통과. 하나라도 미달이면 **HGB 회귀 동결 유지**(10번째 후보라 근소한 승리는 근거 부족).
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
from sklearn.ensemble import (HistGradientBoostingClassifier,  # noqa: E402
                              HistGradientBoostingRegressor)

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
OUT = ROOT / "tmp/eth_entry_b14_matched_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
LABEL_THR, SUB, NMEM = 0.0040, 18000, 5
B_RND, B_PERM = 150, 20000
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b14] {m}", flush=True)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    X = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    X = X.fillna(X[tr].median())
    y = D["y"].to_numpy(); lab = (y > LABEL_THR).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    itr = np.flatnonzero(tr)
    pred_rows = np.flatnonzero(dsel); Xp = X.iloc[pred_rows].to_numpy()
    log(f"TRAIN {len(itr):,} · 피쳐 {len(FEATS)} · 예측대상 {len(pred_rows):,} · 멤버 {NMEM}")

    # 동일 서브샘플 5개 (HGB-sub와 TabPFN이 공유)
    subs = [np.random.default_rng(s).choice(itr, size=min(SUB, len(itr)), replace=False) for s in SEEDS]

    def pol(pfull, wn, frac=None, tau=None):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = pfull[m]
        thr = tau if tau is not None else np.quantile(pv, 1 - frac)
        return stat(slotN(w[pv > thr], NSLOT))[1]

    def expand(p_sub):
        f = np.full(len(D), -np.inf); f[pred_rows] = p_sub; return f

    # A. HGB-full
    hgb_full = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
                .fit(X[tr], y[tr]).predict(X) for s in SEEDS]
    pA = np.mean(hgb_full, axis=0)
    fracs = {wn: float((pA[dsel & (D.split == wn).to_numpy()] > TAU0).mean())
             for wn in ("VAL", "OOS", "HOLDOUT")}
    log("A HGB-full 학습 완료 · 유지비율 " + " ".join(f"{k} {v:.1%}" for k, v in fracs.items()))

    # B. HGB-sub (같은 18k 서브샘플)
    hgb_sub = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
               .fit(X.iloc[rs], y[rs]).predict(X) for s, rs in zip(SEEDS, subs)]
    pB = np.mean(hgb_sub, axis=0)
    log("B HGB-sub(18k) 학습 완료")

    # C. TabPFN-sub (동일 서브샘플)
    tp = []
    for k, (s, rs) in enumerate(zip(SEEDS, subs)):
        clf = TabPFNClassifier(device="cuda", random_state=s)
        clf.fit(X.iloc[rs].to_numpy(), lab[rs])
        tp.append(expand(clf.predict_proba(Xp)[:, 1]))
        log(f"  C TabPFN 멤버 {k+1}/{NMEM}")
    pC = np.mean(tp, axis=0)

    log("\n=== 앙상블 비교 (동일 유지비율) ===")
    print(f"{'arm':22s} " + " ".join(f"{w:>11s}" for w in ("VAL", "OOS", "HOLDOUT")))
    ens = {}
    for nm, p, use_tau in (("A HGB-full (100%)", pA, True),
                            ("B HGB-sub (22%)", pB, False),
                            ("C TabPFN-sub (22%)", pC, False)):
        ens[nm] = {wn: pol(p, wn, tau=TAU0) if use_tau else pol(p, wn, frac=fracs[wn])
                   for wn in ("VAL", "OOS", "HOLDOUT")}
        print(f"{nm:22s} " + " ".join(f"{ens[nm][w]:+10.2f}" for w in ("VAL", "OOS", "HOLDOUT")))
    log(f"\n데이터량의 값어치 (A−B): " + " ".join(
        f"{w} {ens['A HGB-full (100%)'][w]-ens['B HGB-sub (22%)'][w]:+.2f}" for w in ("VAL", "OOS", "HOLDOUT")))
    log(f"동일 데이터 학습기 차이 (C−B): " + " ".join(
        f"{w} {ens['C TabPFN-sub (22%)'][w]-ens['B HGB-sub (22%)'][w]:+.2f}" for w in ("VAL", "OOS", "HOLDOUT")))

    # 개별 멤버 분포 + 순열검정 (비짝)
    log("\n=== 개별 멤버 분포 (동수 5) + 순열검정 ===")
    perm = {}
    for wn in ("VAL", "OOS"):
        a = np.array([pol(p, wn, tau=TAU0) for p in hgb_full])
        c = np.array([pol(p, wn, frac=fracs[wn]) for p in tp])
        obs = c.mean() - a.mean()
        pooled = np.concatenate([a, c])
        cnt = 0
        for _ in range(B_PERM):
            pm = RNG.permutation(pooled)
            if (pm[len(a):].mean() - pm[:len(a)].mean()) >= obs: cnt += 1
        pv = cnt / B_PERM
        perm[wn] = {"hgb": a.tolist(), "tabpfn": c.tolist(), "diff": float(obs), "p": float(pv)}
        log(f"  {wn:5s} HGB {np.round(a,1).tolist()} | TabPFN {np.round(c,1).tolist()}")
        log(f"        평균차 {obs:+.2f}bp · 순열검정 p={pv:.4f} {'✅' if pv < 0.05 else '❌ (분포 이동 없음)'}")

    # 대조군 (TabPFN 앙상블)
    log("\n=== 대조군 (TabPFN 앙상블) ===")
    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy()
        w = D[m]; pv = pC[m]; thr = np.quantile(pv, 1 - fracs[wn]); keep = pv > thr
        real = stat(slotN(w[keep], NSLOT))[1]
        allm = stat(slotN(w, NSLOT))[1]
        rr = np.array([stat(slotN(w[RNG.random(len(w)) < fracs[wn]], NSLOT))[1] for _ in range(B_RND)])
        sub2 = w[keep].sort_values("fi"); v = slotN(w[keep], NSLOT)
        s2 = sub2.iloc[:len(v)].copy(); s2["y2"] = v; s2["day"] = (s2.fi // 288).astype(int)
        days = s2.day.unique()
        bs = np.array([np.concatenate([s2.loc[s2.day == dd, "y2"].to_numpy()
                       for dd in RNG.choice(days, len(days), replace=True)]).mean() * 1e4
                       for _ in range(2000)])
        ctrl[wn] = {"real": real, "keep_all": allm, "rnd": float(rr.mean()),
                    "p_rnd": float((rr >= real).mean()),
                    "ci": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))],
                    "blocks": int(len(days))}
        log(f"  {wn:8s} 실제 {real:+6.2f} | 무필터 {allm:+6.2f} | 무작위 {rr.mean():+6.2f} "
            f"p={float((rr>=real).mean()):.3f} | CI [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}] 블록 {len(days)}일")

    A, C = ens["A HGB-full (100%)"], ens["C TabPFN-sub (22%)"]
    g1 = (C["VAL"] > A["VAL"]) and (C["OOS"] > A["OOS"])
    g2 = all(perm[w]["p"] < 0.05 for w in ("VAL", "OOS"))
    g3 = all(ctrl[w]["p_rnd"] < 0.05 for w in ("VAL", "OOS", "HOLDOUT"))
    log(f"\n⭐사전등록 판정")
    log(f"  ① 앙상블 양 창 승리   {'✅' if g1 else '❌'}")
    log(f"  ② 순열검정 p<0.05 양창 {'✅' if g2 else '❌'}")
    log(f"  ③ 대조군 3창 통과      {'✅' if g3 else '❌'}")
    log(f"  → {'**TabPFN 채택 (v2 동결 필요)**' if (g1 and g2 and g3) else '**HGB 회귀 동결 유지**'}")

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump({"ensembles": ens, "perm": perm, "controls": ctrl, "keep_fracs": fracs,
               "gates": {"ensemble_wins": bool(g1), "perm_sig": bool(g2), "controls_pass": bool(g3)},
               "adopt_tabpfn": bool(g1 and g2 and g3)}, open(OUT / "b14_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
