"""피쳐 감축 사전 게이트 — 102+13열 중 몇 열이 실제로 일하는가 (dev CPU).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909` · 사용자 결정 2026-09-10.

왜 이 실험이 생겼나
------------------
증거신호 6열 사전 게이트에서 `p_slow` 가 **121열 중 2위**, `p_fast` 가 7위로 올라왔는데
정확도는 오히려 −0.41pp 였다. 새 열이 기존 열을 **대체**할 뿐 더하지 않았다는 뜻이고,
그건 뒤집으면 **기존 102열 중 상당수가 잉여**일 수 있다는 신호다.

⚠️ 기대 효과를 정확히 적는다(2026-09-10 정정):
   열을 줄여도 **행은 안 늘어난다** — 균형 컨텍스트 천장은 메모리가 아니라 소수 클래스
   크기(9,243 x 3 ≈ 28k)가 정한다. 실제 기대는 (a) 잡음 희석 감소, (b) TabPFN 사전학습
   분포와의 정합, (c) **속도**(결정층 리플레이가 봉당 호출이라 이게 작지 않다).

공짜 감축 하나
-------------
direction/quality 에서 `POS_COLS` 13열은 `_base_input` 이 **전부 0 으로 채운다**. 상수 열이다.
현직 TabM 은 n_features=115 에 묶여 있어 못 버리지만, TabPFN 은 새로 적합하므로 버릴 수 있다.
이 13열 제거는 정보 손실이 **정확히 0** 이다 — 모델 무관하게 옳다.

선택 편향 방지
-------------
순위는 **TRAIN 에서만** 매긴다(TRAIN 부분표본 순열 중요도). VAL 에서 순위를 매기고 같은
VAL 로 점수를 내면 부풀려진다. VAL 은 K 를 고르는 데만 쓴다(validation_only 규칙).

대조군
-----
같은 K 의 **무작위 부분집합**을 함께 돌린다. 상위-K 가 무작위-K 를 못 이기면 순위 자체가
정보가 없다는 뜻이고, 그러면 감축은 "우연히 잘 맞는 부분집합 찾기"가 된다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import train_eval_omega4_3head_parent72_regimespine_balnobb_20260909 as spine  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as p72  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
from research_omega461_tabicl_direction_head_20260909 import (  # noqa: E402
    DIR_LBL, QUAL_LBL, QUERY_SEED, QUERY_CAP, _score,
)

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/feature_reduction"
K_GRID = [10, 20, 30, 50, 75, 102]
RANK_ROWS = 20000        # TRAIN 부분표본(순열 중요도 비용)
N_RANDOM = 3             # K 마다 무작위 대조군 개수


def _hgb(seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    return HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=31, learning_rate=0.06,
                                          l2_regularization=1.0, random_state=int(seed),
                                          early_stopping=False)


def _fit_score(xtr, ytr, xq, yq, seed=0):
    from sklearn.utils.class_weight import compute_sample_weight
    m = _hgb(seed)
    m.fit(xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return m, _score(yq, m.predict(xq).astype(np.int64))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="zig075", choices=["zig075", "h48qual"])
    ap.add_argument("--target", default="direction", choices=["direction", "quality"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(QUERY_SEED)

    base_cols = spine._install(args.component)
    q_mode, q_dir = (("quality_label_action", QUAL_LBL) if args.target == "quality"
                     else ("same_as_direction", None))
    fr = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                             quality_mode=q_mode, quality_label_dir=q_dir,
                             quality_min_edge=0.0010, quality_max_mae=0.0100,
                             quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
    ycol = "omega4_quality_action" if args.target == "quality" else "zigzag_action"
    tr, va = fr["train_raw"], fr["val_raw"]
    x_tr = parent._base_input(tr, base_cols)
    x_va = parent._base_input(va, base_cols)
    ytr = tr[ycol].to_numpy(np.int64)
    yva = va[ycol].to_numpy(np.int64)

    # ── 상수 열 확인 (공짜 감축) ──
    nun = x_tr.nunique()
    const_cols = [c for c in x_tr.columns if int(nun[c]) <= 1]
    live_cols = [c for c in x_tr.columns if c not in const_cols]
    print(f"[열] 전체 {x_tr.shape[1]} · 상수 {len(const_cols)} · 살아있는 열 {len(live_cols)}", flush=True)
    print(f"  상수 열: {const_cols}", flush=True)

    qidx = np.sort(rng.choice(len(yva), size=min(QUERY_CAP, len(yva)), replace=False))
    yq = yva[qidx]
    rep: dict[str, Any] = {"component": args.component, "target": args.target,
                           "n_cols_total": int(x_tr.shape[1]), "const_cols": const_cols,
                           "n_live": len(live_cols), "k_grid": K_GRID,
                           "rank_rows": RANK_ROWS, "n_random_controls": N_RANDOM, "cells": {}}

    # ── 기준선 두 개 ──
    _m, s_all = _fit_score(x_tr.to_numpy(np.float32), ytr,
                           x_va.iloc[qidx].to_numpy(np.float32), yq)
    _m2, s_live = _fit_score(x_tr[live_cols].to_numpy(np.float32), ytr,
                             x_va[live_cols].iloc[qidx].to_numpy(np.float32), yq)
    rep["baseline_all"] = s_all
    rep["baseline_live"] = s_live
    print(f"\n[기준선] 전체 {x_tr.shape[1]}열 bal_acc {s_all['bal_acc']:.4f}  ·  "
          f"상수 제거 {len(live_cols)}열 {s_live['bal_acc']:.4f}  "
          f"(Δ {s_live['bal_acc']-s_all['bal_acc']:+.4f} — 0 이어야 정상)", flush=True)

    # ── 순위: TRAIN 부분표본 순열 중요도 (VAL 미사용 — 선택 편향 방지) ──
    from sklearn.inspection import permutation_importance
    sub = np.sort(rng.choice(len(ytr), size=min(RANK_ROWS, len(ytr)), replace=False))
    mrank, _ = _fit_score(x_tr[live_cols].to_numpy(np.float32), ytr,
                          x_tr[live_cols].iloc[sub].to_numpy(np.float32), ytr[sub])
    print(f"\n[순위] TRAIN {len(sub):,}행 순열 중요도 계산 중…", flush=True)
    pi = permutation_importance(mrank, x_tr[live_cols].iloc[sub].to_numpy(np.float32), ytr[sub],
                                n_repeats=3, random_state=QUERY_SEED, n_jobs=-1)
    order = np.argsort(-pi.importances_mean)
    ranked = [live_cols[i] for i in order]
    rep["ranking"] = ranked
    print("  상위 15열: " + ", ".join(ranked[:15]), flush=True)

    # ── K 스윕 + 무작위 대조군 ──
    print(f"\n{'='*96}\n[K 스윕]  상위-K vs 무작위-K", flush=True)
    for K in K_GRID:
        if K > len(live_cols):
            continue
        top = ranked[:K]
        _m3, s_top = _fit_score(x_tr[top].to_numpy(np.float32), ytr,
                                x_va[top].iloc[qidx].to_numpy(np.float32), yq)
        rnd = []
        for r in range(N_RANDOM):
            pick = list(rng.choice(live_cols, size=K, replace=False))
            _m4, s_r = _fit_score(x_tr[pick].to_numpy(np.float32), ytr,
                                  x_va[pick].iloc[qidx].to_numpy(np.float32), yq)
            rnd.append(s_r["bal_acc"])
        rep["cells"][str(K)] = {"top": s_top, "random_bal_acc": rnd,
                                "random_median": statistics.median(rnd),
                                "delta_vs_live": s_top["bal_acc"] - s_live["bal_acc"],
                                "delta_vs_random": s_top["bal_acc"] - statistics.median(rnd)}
        c = rep["cells"][str(K)]
        print(f"  K={K:4d}  상위 {s_top['bal_acc']:.4f}  무작위중앙 {c['random_median']:.4f}  "
              f"(Δ순위 {c['delta_vs_random']:+.4f})   전체대비 {c['delta_vs_live']:+.4f}", flush=True)

    best_k, best = max(rep["cells"].items(), key=lambda kv: kv[1]["top"]["bal_acc"])
    gain = best["delta_vs_live"]
    rank_works = all(v["delta_vs_random"] > 0 for v in rep["cells"].values())
    print(f"\n{'='*96}\n[판정]", flush=True)
    print(f"  최선 K={best_k}  bal_acc {best['top']['bal_acc']:.4f}  "
          f"(전체 {len(live_cols)}열 대비 {gain:+.4f})", flush=True)
    print(f"  순위가 무작위를 전 K 에서 이기는가: {'✅' if rank_works else '❌'} "
          f"— 아니면 순위에 정보가 없다는 뜻", flush=True)
    print(f"  상수 {len(const_cols)}열 제거는 정보 손실 0 — 모델 무관하게 적용 가능", flush=True)
    verdict = ("감축이 정확도를 올린다 — TabPFN 게이트로" if gain > 0 and rank_works else
               "감축해도 정확도는 안 오른다 — 다만 상수열 제거와 속도 이득은 그대로 유효")
    print(f"\n  → {verdict}", flush=True)
    rep["verdict"] = {"best_k": int(best_k), "gain_vs_live": gain,
                      "ranking_beats_random_all_k": bool(rank_works), "text": verdict}
    p = OUT / f"reduction_{args.target}_{args.component}.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
