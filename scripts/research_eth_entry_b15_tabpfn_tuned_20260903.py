#!/usr/bin/env python3
"""B15: TabPFN 공식 권고 적용 후 재검정 -- 회귀·분류 (2026-09-03, 서버).

docs.priorlabs.ai/improving-performance 와 /capabilities/embeddings 를 읽고 적용한다.
지금까지의 TabPFN 비교가 **불공정했을 가능성**이 있다:

  ⚠️문서: "데이터를 최대한 raw로 넣어라. 결측 대체·스케일링은 성능을 해친다"
     → 우리는 X.fillna(median)을 했다. 명시적으로 하지 말라는 것.
  ⚠️문서: "피쳐가 100개 이상이면 선별해서 attention 효율을 높여라"
     → 우리는 161개를 넣었다. HGB는 피쳐 수에 둔감했지만(B8) TabPFN은 문서상 불리하다.
  · n_estimators(기본 auto) / ignore_pretraining_limits(기본 False) 도 있다.
  · TabPFNEmbedding은 tabpfn_extensions 미설치라 이번엔 제외(quant_ai는 라이브 공유 env라
    함부로 설치하지 않는다 -- TabFM 때 별도 env를 쓴 것과 같은 이유).

1단계(스크린, 멤버 2개): 기법 조합을 훑어 최선을 찾는다.
2단계(확정, 멤버 5개): 최선 조합으로 HGB-full과 동수 비교 + 순열검정 + 대조군.

⭐사전등록(B14와 동일): ①앙상블 양 창 승리 ②순열검정 p<0.05 양 창 ③대조군 3창 통과.
   하나라도 미달이면 HGB 회귀 동결 유지.
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
from sklearn.ensemble import HistGradientBoostingRegressor  # noqa: E402

from research_eth_entry_b1_cumulative_arms_20260903 import SEEDS, HP, stat  # noqa: E402
from research_eth_entry_b6_expand_20260903 import slotN  # noqa: E402

B6 = ROOT / "tmp/eth_entry_b6_expand_20260903"
RANK = ROOT / "tmp/eth_entry_b8_featsel_20260903/ranked_features.csv"
OUT = ROOT / "tmp/eth_entry_b15_tabpfn_tuned_20260903"
DEPTH, WAIT, TAU0, NSLOT = 3.0, 6, 0.0040, 4
LABEL_THR, SUB = 0.0040, 18000
B_RND, B_PERM = 150, 20000
RNG = np.random.default_rng(20260903)


def log(m): print(f"[b15] {m}", flush=True)


def main() -> int:
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("TABPFN_TOKEN="):
                os.environ["TABPFN_TOKEN"] = line.split("=", 1)[1].strip().strip('"')
    from tabpfn import TabPFNClassifier, TabPFNRegressor

    D = pd.read_csv(B6 / "fills.csv", parse_dates=["timestamp"], low_memory=False)
    cfg = json.loads((ROOT / "tmp/eth_causal_population_metalabel_20260902/config.json").read_text())
    base = [c for c in cfg["features"] if c != "is_bottom"]
    excl = set(base + ["arm", "sig_id", "atr_pct", "depth", "y", "split", "timestamp", "i",
                       "side", "signal", "fi", "ei", "btf", "lim", "sd", "pred"])
    R = [c for c in D.columns if c.endswith("_r136")] + \
        [c for c in D.columns if c not in excl and not c.endswith("_r136")]
    R = list(dict.fromkeys([c for c in R if D[c].dtype.kind in "fiub"]))
    FEATS = list(dict.fromkeys(base + ["arm", "sig_id", "atr_pct", "depth"] + R))
    Xraw = D[FEATS].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    tr = (D.split == "TRAIN").to_numpy()
    Ximp = Xraw.fillna(Xraw[tr].median())
    y = D["y"].to_numpy(); lab = (y > LABEL_THR).astype(int)
    dsel = ((D.depth == DEPTH) & (D.btf <= WAIT)).to_numpy()
    itr = np.flatnonzero(tr); pred_rows = np.flatnonzero(dsel)
    log(f"TRAIN {len(itr):,} · 피쳐 {len(FEATS)} · 예측 {len(pred_rows):,} · 결측률 {float(Xraw.isna().mean().mean()):.4%}")

    ranked = pd.read_csv(RANK)["feature"].tolist() if RANK.exists() else FEATS
    TOP30 = [c for c in ranked if c in FEATS][:30]
    TOP50 = [c for c in ranked if c in FEATS][:50]
    log(f"선별 후보: top30({len(TOP30)}) top50({len(TOP50)}) · 상위8 {TOP30[:8]}")

    # HGB 기준선
    hgb = [HistGradientBoostingRegressor(random_state=s, loss="squared_error", **HP)
           .fit(Ximp[tr], y[tr]).predict(Ximp) for s in SEEDS]
    pA = np.mean(hgb, axis=0)
    fracs = {wn: float((pA[dsel & (D.split == wn).to_numpy()] > TAU0).mean())
             for wn in ("VAL", "OOS", "HOLDOUT")}
    def pol(pf, wn, frac=None, tau=None):
        m = dsel & (D.split == wn).to_numpy(); w = D[m]; pv = pf[m]
        thr = tau if tau is not None else np.quantile(pv, 1 - frac)
        return stat(slotN(w[pv > thr], NSLOT))[1]
    hgbA = {wn: pol(pA, wn, tau=TAU0) for wn in ("VAL", "OOS", "HOLDOUT")}
    log("HGB-full 기준선 " + " ".join(f"{k} {v:+.2f}" for k, v in hgbA.items()))

    def expand(ps): 
        f = np.full(len(D), -np.inf); f[pred_rows] = ps; return f

    def build(task, feats, impute, n_est, nmem, ipl=False):
        outs = []
        Xu = (Ximp if impute else Xraw)[feats]
        Xp = Xu.iloc[pred_rows].to_numpy()
        for k in range(nmem):
            rs = np.random.default_rng(SEEDS[k]).choice(itr, size=min(SUB, len(itr)), replace=False)
            kw = dict(device="cuda", random_state=SEEDS[k], ignore_pretraining_limits=ipl)
            if n_est is not None: kw["n_estimators"] = n_est
            if task == "cls":
                m = TabPFNClassifier(**kw); m.fit(Xu.iloc[rs].to_numpy(), lab[rs])
                outs.append(expand(m.predict_proba(Xp)[:, 1]))
            else:
                m = TabPFNRegressor(**kw); m.fit(Xu.iloc[rs].to_numpy(), y[rs])
                outs.append(expand(m.predict(Xp)))
        return outs

    # ---- 1단계 스크린 (멤버 2) ----
    CFGS = [
        ("T0 기준(161,대체)",      FEATS, True,  None, False),
        ("T1 raw결측(161)",        FEATS, False, None, False),
        ("T2 선별30(대체)",        TOP30, True,  None, False),
        ("T3 raw+선별30",          TOP30, False, None, False),
        ("T4 raw+선별50",          TOP50, False, None, False),
        ("T5 raw+선별30+est8",     TOP30, False, 8,    False),
        ("T6 raw+선별30+ipl",      TOP30, False, None, True),
    ]
    log("\n=== 1단계 스크린 (멤버 2, VAL/OOS) ===")
    print(f"{'config':24s} {'task':5s} " + " ".join(f"{w:>9s}" for w in ("VAL", "OOS")))
    screen = []
    for task in ("cls", "reg"):
        for nm, fs, imp, ne, ipl in CFGS:
            try:
                ms = build(task, fs, imp, ne, 2, ipl)
                p = np.mean(ms, axis=0)
                v, o = pol(p, "VAL", frac=fracs["VAL"]), pol(p, "OOS", frac=fracs["OOS"])
                screen.append({"task": task, "cfg": nm, "VAL": round(v, 2), "OOS": round(o, 2)})
                print(f"{nm:24s} {task:5s} {v:+9.2f} {o:+9.2f}")
            except Exception as e:
                print(f"{nm:24s} {task:5s}   실패 {type(e).__name__}: {str(e)[:60]}")
    S = pd.DataFrame(screen)
    S["score"] = S[["VAL", "OOS"]].min(axis=1)      # 양 창 최소값으로 고름(한쪽만 좋은 걸 배제)
    best = S.loc[S.score.idxmax()]
    log(f"\n⭐1단계 최선: {best.task} / {best.cfg} (VAL {best.VAL:+.2f} OOS {best.OOS:+.2f})")

    # ---- 2단계 확정 (멤버 5 + 순열검정 + 대조군) ----
    nm, fs, imp, ne, ipl = [c for c in CFGS if c[0] == best.cfg][0]
    ms = build(best.task, fs, imp, ne, 5, ipl)
    pC = np.mean(ms, axis=0)
    ensC = {wn: pol(pC, wn, frac=fracs[wn]) for wn in ("VAL", "OOS", "HOLDOUT")}
    log(f"\n=== 2단계 확정 ({best.task} / {nm}, 멤버 5) ===")
    print(f"{'':22s} " + " ".join(f"{w:>11s}" for w in ("VAL", "OOS", "HOLDOUT")))
    print(f"{'A HGB-full':22s} " + " ".join(f"{hgbA[w]:+10.2f}" for w in ("VAL", "OOS", "HOLDOUT")))
    print(f"{'C TabPFN-tuned':22s} " + " ".join(f"{ensC[w]:+10.2f}" for w in ("VAL", "OOS", "HOLDOUT")))

    perm = {}
    for wn in ("VAL", "OOS"):
        a = np.array([pol(p, wn, tau=TAU0) for p in hgb])
        c = np.array([pol(p, wn, frac=fracs[wn]) for p in ms])
        obs = c.mean() - a.mean(); pooled = np.concatenate([a, c])
        cnt = sum(1 for _ in range(B_PERM)
                  if (lambda pm: pm[len(a):].mean() - pm[:len(a)].mean())(RNG.permutation(pooled)) >= obs)
        perm[wn] = {"hgb": a.tolist(), "tabpfn": c.tolist(), "diff": float(obs), "p": cnt / B_PERM}
        log(f"  {wn:5s} HGB {np.round(a,1).tolist()} | TabPFN {np.round(c,1).tolist()} "
            f"평균차 {obs:+.2f} p={cnt/B_PERM:.4f}")

    ctrl = {}
    for wn in ("VAL", "OOS", "HOLDOUT"):
        m = dsel & (D.split == wn).to_numpy(); w = D[m]; pv = pC[m]
        thr = np.quantile(pv, 1 - fracs[wn]); keep = pv > thr
        real = stat(slotN(w[keep], NSLOT))[1]
        rr = np.array([stat(slotN(w[RNG.random(len(w)) < fracs[wn]], NSLOT))[1] for _ in range(B_RND)])
        ctrl[wn] = {"real": real, "rnd": float(rr.mean()), "p": float((rr >= real).mean())}
        log(f"  대조군 {wn:8s} 실제 {real:+6.2f} vs 무작위 {rr.mean():+6.2f} p={float((rr>=real).mean()):.3f}")

    g1 = ensC["VAL"] > hgbA["VAL"] and ensC["OOS"] > hgbA["OOS"]
    g2 = all(perm[w]["p"] < 0.05 for w in ("VAL", "OOS"))
    g3 = all(ctrl[w]["p"] < 0.05 for w in ("VAL", "OOS", "HOLDOUT"))
    log(f"\n⭐사전등록 판정  ①양창승리 {'✅' if g1 else '❌'}  ②순열 p<0.05 {'✅' if g2 else '❌'}  "
        f"③대조군 {'✅' if g3 else '❌'}")
    log(f"  → {'**TabPFN 채택 (v2 동결 필요)**' if (g1 and g2 and g3) else '**HGB 회귀 동결 유지**'}")

    OUT.mkdir(parents=True, exist_ok=True)
    S.to_csv(OUT / "screen.csv", index=False)
    json.dump({"screen": screen, "best": {"task": best.task, "cfg": best.cfg},
               "hgb": hgbA, "tabpfn": ensC, "perm": perm, "controls": ctrl,
               "adopt": bool(g1 and g2 and g3),
               "embeddings": "skipped: tabpfn_extensions not installed in quant_ai (shared live env)"},
              open(OUT / "b15_report.json", "w"), indent=2)
    log(f"\n산출: {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
