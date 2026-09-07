#!/usr/bin/env python3
"""방향 축 전 결과 취합 → 섀도우 Top-1 선정 (2026-09-07).

사용자: *"지금까지 tabICL과 MASHT과 tabm과 hgb 등 이때까지 테스트한 결과들 다 취합해서
       각 피쳐별과 각 arm 별 최고 조합을 가지고 최고 Top1 을 새도우를 돌려보자"*

## ⚠️승자의 저주를 먼저 계산한다
N 셀의 잡음 추정치에서 최대값을 고르면 그 점추정은 위로 편향된다.
셀당 표준편차 sd 일 때 편향 ≈ sd × E[max of N standard normals].
이 표를 함께 내서 **섀도우의 기대치를 얼마나 할인해야 하는지** 명시한다.

섀도우는 **전방** 테스트이므로 선택 편향이 검정의 타당성을 해치지는 않는다 --
기대치만 낮춘다. 그래서 사전등록 임계는 할인 후 값으로 잡아야 한다.

## 취합 대상
TabICLv2(4실험) · TabPFN/MASHT · TabM(2) · HGB(4) — 각 CSV/JSON 의 창별 AUC.
공통 지표: VAL/OOS/HOLDOUT_SPENT AUC, min3(세 창 최악), mean3.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/eth_anchor_allresults_20260907"
W = ("VAL", "OOS", "HOLDOUT_SPENT")

CSV_SRC = [
    ("TabICLv2", "tmp/eth_anchor_tabicl_deep_20260907/direction_3windows.csv"),
    ("TabICLv2", "tmp/eth_anchor_trend_eval_20260907/trend_eval.csv"),
    ("TabICLv2", "tmp/eth_anchor_regime_ablation_20260907/ablation.csv"),
    ("TabICLv2", "tmp/eth_anchor_lag_eval_20260907/lag_eval.csv"),
    ("TabPFN/MASHT", "tmp/eth_anchor_masht_20260907/masht_eval.csv"),
    ("TabICLv2/WF", "tmp/eth_anchor_walkforward_20260907/walkforward.csv"),
]


def load_json_cfgs(tag, path, arm_key="arm", fs_key="featset"):
    p = ROOT / path
    if not p.exists():
        return []
    d = json.loads(p.read_text())
    cfgs = d.get("configs", d if isinstance(d, list) else [])
    out = []
    for c in cfgs if isinstance(cfgs, list) else []:
        r = {"model": tag, "arm": c.get(arm_key, c.get("arm", "?")),
             "featset": str(c.get(fs_key, c.get("feat", c.get("featset", "?"))))}
        for w in W:
            r[f"{w}_auc"] = c.get(f"{w}_auc", np.nan)
            r[f"{w}_lo"] = c.get(f"{w}_lo", np.nan)
        out.append(r)
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for model, path in CSV_SRC:
        p = ROOT / path
        if not p.exists():
            continue
        A = pd.read_csv(p)
        if "arm" not in A.columns:
            continue
        A = A[A.get("axis", "direction") == "direction"] if "axis" in A.columns else A
        for _, r in A.iterrows():
            rec = {"model": model, "arm": str(r["arm"]),
                   "featset": str(r.get("featset", r.get("feat", "?"))), "src": Path(path).parent.name}
            for w in W:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
            rows.append(rec)
    for tag, path in [("TabM-3head", "tmp/eth_anchor_tabm3head_20260907/summary.json"),
                      ("TabM-perhead", "tmp/eth_anchor_tabm_perhead_20260907/summary.json")]:
        rows += [dict(r, src=Path(path).parent.name) for r in load_json_cfgs(tag, path)]
    for tag, path in [("HGB", "tmp/eth_anchor_direction_tuned_20260907/final.csv"),
                      ("HGB", "tmp/eth_anchor_expanded_eval_20260907/final.csv"),
                      ("HGB", "tmp/eth_anchor_f154_eval_20260907/final.csv"),
                      ("HGB", "tmp/eth_anchor_baseline_eval_20260907/arms.csv")]:
        p = ROOT / path
        if not p.exists():
            continue
        A = pd.read_csv(p)
        for _, r in A.iterrows():
            rec = {"model": tag, "arm": str(r.get("arm", "?")),
                   "featset": str(r.get("feat", r.get("tag", r.get("featset", "?")))),
                   "src": Path(path).parent.name}
            for w in W:
                rec[f"{w}_auc"] = r.get(f"{w}_auc", np.nan)
                rec[f"{w}_lo"] = r.get(f"{w}_lo", np.nan)
            rows.append(rec)

    A = pd.DataFrame(rows)
    A = A[A.arm.isin(["hard", "three", "wbin", "A1", "A2"])].reset_index(drop=True)
    # 🔴대조군 제거 -- 방향 라벨이 아니거나(P1 크기) 방향을 뒤집은(C3) 행이 섞여 있었다
    CTRL = "크기|^P1|^C1|^C3|^C6|뒤집|shuffle|귀무"
    n0 = len(A)
    A = A[~A.featset.str.contains(CTRL, na=False, regex=True)].reset_index(drop=True)
    print(f"[대조군 제외] {n0} → {len(A)}셀 ({n0-len(A)}행: P1 크기라벨·C3 뒤집기·C1 atr단독·C6 수치변형)",
          flush=True)
    aucs = A[[f"{w}_auc" for w in W]]
    A["n_win"] = aucs.notna().sum(axis=1)
    A["min3"] = aucs.min(axis=1)
    A["mean3"] = aucs.mean(axis=1)
    A["ci3"] = (A[[f"{w}_lo" for w in W]] > 0.5).all(axis=1) & A[[f"{w}_lo" for w in W]].notna().all(axis=1)
    A = A.sort_values("mean3", ascending=False).reset_index(drop=True)
    A.to_csv(OUT / "all_direction_results.csv", index=False)

    print("=" * 112, flush=True)
    print(f"취합 {len(A)}셀 · 모델 {A.model.nunique()}종 · 세 창 모두 있는 셀 {int((A.n_win==3).sum())}", flush=True)
    print(A.groupby("model").agg(셀=("arm", "size"), mean3최고=("mean3", "max"),
                                 min3최고=("min3", "max")).round(4).to_string(), flush=True)

    print("\n" + "=" * 112, flush=True)
    print("팔별 최고 (세 창 전부 있는 셀만, mean3 기준)", flush=True)
    print("=" * 112, flush=True)
    F = A[A.n_win == 3]
    for arm in ("hard", "three", "wbin"):
        s = F[F.arm == arm]
        if not len(s):
            continue
        print(f"\n  [{arm}]  n={len(s)}셀")
        for _, r in s.nlargest(3, "mean3").iterrows():
            print(f"    {r.model:<14}{r.featset:<14} "
                  + " · ".join(f"{w[:3]} {r[f'{w}_auc']:.4f}" for w in W)
                  + f"  mean3 {r.mean3:.4f} min3 {r.min3:.4f}")

    print("\n" + "=" * 112, flush=True)
    print("피쳐셋별 최고 (mean3)", flush=True)
    print("=" * 112, flush=True)
    g = F.loc[F.groupby("featset").mean3.idxmax()].nlargest(12, "mean3")
    for _, r in g.iterrows():
        print(f"  {r.featset:<15}{r.model:<14}{r.arm:<7} "
              + " · ".join(f"{w[:3]} {r[f'{w}_auc']:.4f}" for w in W) + f"  mean3 {r.mean3:.4f}")

    print("\n" + "=" * 112, flush=True)
    print("⭐전체 Top-5 (mean3)", flush=True)
    print("=" * 112, flush=True)
    top = F.nlargest(5, "mean3")
    for i, (_, r) in enumerate(top.iterrows(), 1):
        print(f"  {i}. {r.model:<14}{r.arm:<7}{r.featset:<14} "
              + " · ".join(f"{w[:3]} {r[f'{w}_auc']:.4f}" for w in W)
              + f"  mean3 {r.mean3:.4f} min3 {r.min3:.4f}  CI3 {'✅' if r.ci3 else '❌'}")

    # ---- 승자의 저주 할인
    N = len(F)
    # 표집 sd 는 **일군집 CI 폭**에서 뽑는다 (창간 산포가 아니라).
    hw = []
    for w in W:
        lo, hi = F.get(f"{w}_lo"), F.get(f"{w}_hi")
        if lo is not None and hi is not None:
            d = (pd.to_numeric(hi, errors="coerce") - pd.to_numeric(lo, errors="coerce")) / 2
            hw += [x for x in d.dropna()]
    hw_med = float(np.median(hw)) if hw else 0.065
    sd_win = hw_med / 1.96                      # 창별 AUC 의 표집 sd
    sd_mean3 = sd_win / np.sqrt(3)              # 세 창 평균의 sd
    # 셀들은 서로 강하게 상관(같은 데이터·중첩 피쳐셋) -- 유효 독립 후보수로 보수화
    n_eff = F.groupby(["arm", "model"]).ngroups * 3
    rng = np.random.default_rng(0)
    def emax(n):
        return float(np.mean(np.max(rng.standard_normal((4000, max(int(n), 2))), axis=1)))
    best = float(top.iloc[0].mean3)
    print("\n" + "=" * 112, flush=True)
    print("⚠️승자의 저주 할인", flush=True)
    print("=" * 112, flush=True)
    print(f"  후보 셀 N={N} · 일군집 CI 반폭(중앙) {hw_med:.4f} → 창별 sd {sd_win:.4f} "
          f"→ mean3 sd {sd_mean3:.4f}", flush=True)
    for n, tag in [(N, f"독립 가정 N={N}"), (n_eff, f"상관 보정 N_eff≈{n_eff}")]:
        b = sd_mean3 * emax(n)
        print(f"    {tag:<24} E[max] {emax(n):.3f} · 편향 {b:.4f} → "
              f"Top-1 {best:.4f} 할인 후 {best - b:.4f}", flush=True)
    bias = sd_mean3 * emax(n_eff)
    print(f"\n  ⇒ 섀도우 기대치는 **할인 후**({best - bias:.4f})로 잡는다. "
          f"원값 {best:.4f} 를 그대로 기대하면 안 된다.", flush=True)
    (OUT / "top.json").write_text(json.dumps(
        {"top5": top[["model", "arm", "featset", "mean3", "min3"]].to_dict("records"),
         "n_candidates": N, "winners_curse_bias": bias,
         "top1_raw": best, "top1_discounted": best - bias}, indent=1, ensure_ascii=False))
    print(f"\n저장: {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
