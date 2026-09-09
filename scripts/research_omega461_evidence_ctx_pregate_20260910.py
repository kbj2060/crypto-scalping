"""증거신호 6열 사전 게이트 — 한계 정보량 + 누수 트립와이어 (dev CPU, GPU 불필요).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
사용자 제안 2026-09-10: direction/quality 입력에 대시보드 증거신호·이벤트 트리거를 넣어보자.

왜 사전 게이트를 두는가
----------------------
**이 주입은 2026-08-14 에 이미 한 번 실패했다.** `build_eth_evidence_signal_context_features_
20260814.py` 문서 첫머리에 경위가 있다 — 증거신호를 Omega 헤드 입력으로 넣는 사전 게이트가
"weak / VAL-contradicted" 로 돌아왔고 그래서 사이드카 주입으로 방향을 틀었다. exit_head 실험
문서는 더 구체적이다: 증거신호와 최종 수익률의 상관이 **가장 중요한 선별창 VAL 에서 유의미하게
반대 방향**이었다(rho +0.056, p=0.0006).
그때와 다른 점은 (1) 헤드가 exit 이 아니라 direction, (2) 모델이 TabM 이 아니라 in-context,
(3) "설계가 다른 두 모델이 같은 자리에 멈췄다 = 데이터 한계" 라는 진단이 선다는 것.
그래도 이력이 부정적이므로 45분짜리 GPU 게이트 전에 **몇 분짜리 CPU 확인**을 먼저 둔다.

무엇을 재는가 — Δ 하나가 정보량이자 누수 신호다
----------------------------------------------
같은 모델(HGB)·같은 행·같은 시드에서 **102열만** vs **102열 + 증거 6열**의 VAL balanced accuracy.
질의 행은 TabPFN 게이트와 **같은 고정 6,000행**이라 숫자가 직접 비교된다.

사전 등록 판정대
  P1 정보량   : Δ 중앙 > 0 이고 3/5 시드 이상.  Δ<=0 이면 여기서 닫는다 —
                기존 102열이 이미 담고 있는 정보라는 뜻이다(주문흐름·모멘텀 계열이 이미 있다).
  P2 트립와이어: Δ 중앙 > **+0.02** 면 통과가 아니라 **누수 의심**으로 정지한다.
                CLAUDE.md 사건-라벨 경계 계약: 경계 피쳐군 제거시 정확도 2pp 이상 변하면
                누수 강력 의심. 여기선 부호가 반대(추가)일 뿐 같은 문턱이다.
  → 통과 구간은 **0 < Δ <= 0.02**. 너무 작으면 무용, 너무 크면 수상하다.

보조 진단 (CLAUDE.md 탐지 신호 그대로)
  · 6열의 순열 중요도 순위
  · 6열 각각의 **단변량 분위 효과** (십분위별 라벨 분포)
  순열 중요도는 높은데 단변량 분위 효과가 평평하면 상호작용이 아니라 누수를 먼저 의심한다.
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
    DIR_LBL, QUAL_LBL, SEEDS, QUERY_SEED, QUERY_CAP, _score, _binom_ge,
)

CTX_DIR = ROOT / "tmp/causal_regen_20260516/eth_zig075_evidence_signal_context_20260814"
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/evidence_ctx"
NEW_COLS = ["trend_ctx_taker_delta_z", "trend_ctx_p_fast", "trend_ctx_p_slow",
            "trend_ctx_ret3_z", "trend_ctx_liquidity_sweep_low", "trend_ctx_liquidity_sweep_high"]
P2_TRIPWIRE = 0.02


def _attach(x: pd.DataFrame, frame: pd.DataFrame, split: str) -> pd.DataFrame:
    ctx = pd.read_csv(CTX_DIR / f"{split}_context_features.csv", parse_dates=["timestamp"])
    if len(ctx) != len(frame):
        raise RuntimeError(f"{split}: 컨텍스트 {len(ctx)}행 != 프레임 {len(frame)}행")
    if not ctx["timestamp"].reset_index(drop=True).equals(
            pd.to_datetime(frame["timestamp"]).reset_index(drop=True)):
        raise RuntimeError(f"{split}: timestamp 정렬 불일치 — 조용한 어긋남 방지를 위해 중단")
    out = x.copy()
    for c in NEW_COLS:
        out[c] = pd.to_numeric(ctx[c], errors="raise").to_numpy(np.float32)
    return out


def _fit_score(xtr, ytr, xq, yq, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.utils.class_weight import compute_sample_weight
    m = HistGradientBoostingClassifier(max_iter=220, max_leaf_nodes=31, learning_rate=0.06,
                                       l2_regularization=1.0, random_state=int(seed),
                                       early_stopping=False)
    m.fit(xtr, ytr, sample_weight=compute_sample_weight("balanced", ytr))
    return m, _score(yq, m.predict(xq).astype(np.int64))


def _univariate(x: pd.DataFrame, y: np.ndarray, col: str) -> dict:
    """십분위별 라벨 분포 — 단변량 효과가 평평한지 본다(누수 탐지 보조)."""
    v = x[col].to_numpy(np.float64)
    if len(np.unique(v)) <= 2:                       # 사건형 0/1
        return {"kind": "binary",
                "rate_by_value": {str(int(u)): np.bincount(y[v == u], minlength=3).tolist()
                                  for u in np.unique(v)}}
    q = pd.qcut(pd.Series(v).rank(method="first"), 10, labels=False)
    tab = [np.bincount(y[q == d], minlength=3) for d in range(10)]
    share = [float(t[1] / max(t.sum(), 1)) for t in tab]     # 롱 라벨 비율
    return {"kind": "decile", "long_share_by_decile": [round(s, 4) for s in share],
            "spread": round(max(share) - min(share), 4)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="zig075", choices=["zig075", "h48qual"])
    ap.add_argument("--target", default="direction", choices=["direction", "quality"])
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    base_cols = spine._install(args.component)
    if args.target == "quality":
        q_mode, q_dir = "quality_label_action", QUAL_LBL
    else:
        q_mode, q_dir = "same_as_direction", None
    fr = p72._prepare_frames(disable_tp_sl=False, direction_label_dir=DIR_LBL,
                             quality_mode=q_mode, quality_label_dir=q_dir,
                             quality_min_edge=0.0010, quality_max_mae=0.0100,
                             quality_min_mfe_mae=1.20, quality_max_hold_bars=288)
    ycol = "omega4_quality_action" if args.target == "quality" else "zigzag_action"
    tr, va = fr["train_raw"], fr["val_raw"]
    xtr_b = parent._base_input(tr, base_cols)
    xva_b = parent._base_input(va, base_cols)
    ytr = tr[ycol].to_numpy(np.int64)
    yva = va[ycol].to_numpy(np.int64)
    xtr_a = _attach(xtr_b, tr, "train")
    xva_a = _attach(xva_b, va, "validation")
    print(f"[데이터] TRAIN {len(ytr):,} · VAL {len(yva):,} · 기존 {xtr_b.shape[1]}열 "
          f"→ 증강 {xtr_a.shape[1]}열 (+{len(NEW_COLS)})", flush=True)

    qidx = np.sort(np.random.default_rng(QUERY_SEED).choice(len(yva),
                                                            size=min(QUERY_CAP, len(yva)), replace=False))
    yq = yva[qidx]
    print(f"  질의 {len(yq):,}행 (TabPFN 게이트와 동일한 고정 표본)", flush=True)

    rep: dict[str, Any] = {"component": args.component, "target": args.target, "seeds": seeds,
                           "new_cols": NEW_COLS, "p2_tripwire": P2_TRIPWIRE,
                           "n_train": int(len(ytr)), "n_query": int(len(yq)), "per_seed": {}}

    print("\n[단변량 분위 효과] 평평하면 그 자체로 정보가 없다는 뜻", flush=True)
    rep["univariate"] = {c: _univariate(xtr_a, ytr, c) for c in NEW_COLS}
    for c, u in rep["univariate"].items():
        if u["kind"] == "decile":
            print(f"  {c:34s} 롱비율 십분위 폭 {u['spread']:.4f}  {u['long_share_by_decile']}", flush=True)
        else:
            print(f"  {c:34s} (사건형) {u['rate_by_value']}", flush=True)

    deltas = []
    print(f"\n{'='*92}\n[시드별] 102열 vs 102열+증거6열", flush=True)
    for seed in seeds:
        _mb, sb = _fit_score(xtr_b.to_numpy(np.float32), ytr,
                             xva_b.iloc[qidx].to_numpy(np.float32), yq, seed)
        ma, sa = _fit_score(xtr_a.to_numpy(np.float32), ytr,
                            xva_a.iloc[qidx].to_numpy(np.float32), yq, seed)
        d = sa["bal_acc"] - sb["bal_acc"]
        deltas.append(d)
        rep["per_seed"][str(seed)] = {"base": sb, "augmented": sa, "delta_bal_acc": d}
        print(f"  s{seed}  기존 {sb['bal_acc']:.4f} → 증강 {sa['bal_acc']:.4f}  Δ {d:+.4f}", flush=True)

    # 순열 중요도는 마지막 시드 모델에서 한 번만(비용)
    try:
        from sklearn.inspection import permutation_importance
        r = permutation_importance(ma, xva_a.iloc[qidx].to_numpy(np.float32), yq,
                                   n_repeats=3, random_state=QUERY_SEED, n_jobs=-1)
        order = np.argsort(-r.importances_mean)
        cols = list(xva_a.columns)
        rank = {c: int(np.where(order == cols.index(c))[0][0]) + 1 for c in NEW_COLS}
        rep["perm_importance_rank"] = rank
        print(f"\n[순열 중요도 순위] (전체 {len(cols)}열 중)", flush=True)
        for c, k in sorted(rank.items(), key=lambda kv: kv[1]):
            print(f"  {k:4d}위  {c}", flush=True)
    except Exception as exc:
        rep["perm_importance_rank"] = f"{type(exc).__name__}: {exc}"
        print(f"\n[순열 중요도] 건너뜀 — {exc}", flush=True)

    med = statistics.median(deltas)
    n_pos = sum(1 for d in deltas if d > 0)
    p1 = med > 0 and n_pos * 2 > len(deltas)
    p2 = med > P2_TRIPWIRE
    print(f"\n{'='*92}\n[사전 게이트]", flush=True)
    print(f"  Δ bal_acc 중앙 {med:+.4f}  [{min(deltas):+.4f}, {max(deltas):+.4f}]  "
          f"양수 {n_pos}/{len(deltas)}  귀무 p={_binom_ge(n_pos, len(deltas)):.3f}", flush=True)
    print(f"  P1 정보량 (Δ>0 & 과반)            : {'✅' if p1 else '❌'}", flush=True)
    print(f"  P2 누수 트립와이어 (Δ>{P2_TRIPWIRE:.2f} 이면 정지) : "
          f"{'🔴 발동 — 누수 의심' if p2 else '✅ 미발동'}", flush=True)
    verdict = ("누수 의심 — GPU 게이트 진행 금지, 경계 먼저 조사" if p2 else
               "통과 — 3단계(TabPFN 5시드) 진행 가능" if p1 else
               "정보량 없음 — 여기서 닫는다(기존 102열이 이미 담고 있다)")
    print(f"\n  → {verdict}", flush=True)
    rep["verdict"] = {"median_delta": med, "n_positive": n_pos, "P1": bool(p1),
                      "P2_leak_tripwire": bool(p2), "text": verdict}
    p = OUT / f"pregate_{args.target}_{args.component}.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
