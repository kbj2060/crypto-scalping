"""exit head 모델 A/B — 현직 TabM vs TabPFN v3 (짝지은 5시드 게이트).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계획 3번(사용자 승인 2026-09-09): direction → **exit** → quality → 결합 → 사이드카.
direction 결과: TabICL +0.0075 / TabPFN v3 +0.0055, 사실상 동률. TabPFN 이 GPU 2.7배 적고
빨라서 다음 페이즈는 TabPFN 으로 간다(사용자 결정).

direction head 게이트와 무엇이 다른가
------------------------------------
exit head 는 **봉 행이 아니라 라이프사이클(포지션) 행**을 쓴다:
  · 입력 115열 = base_cols 102(`cur_` 접두사로 저장) + POS_COLS 13(**실제 값**, direction 은 0)
  · 라벨 2클래스, 진입 사건에서 파생 — 봉마다 있는 게 아니다
  · 추론 라우팅은 `hard._route_id(frame)` 로 그 봉의 레짐 argmax → 전문가 선택
    (`_metrics_with_shared_exit` 의 실제 경로. 여기서도 그대로 재현한다)

학습/평가 분리 — 왜 VAL 에서 데이터셋을 새로 짓는가
--------------------------------------------------
p72 는 exit 데이터셋을 **TRAIN 프레임에서만** 짓고(`frames["train_df"]`, < 2025-10-01),
전문가 내부에서 85/15 로 쪼개 조기종료에 쓴다. 그래서 그 15% 도 현직 TabM 이 **본 행**이다.
그 위에서 비교하면 현직에 유리하게 기운다.

대신 **같은 빌더를 VAL 프레임에 그대로 돌려** 질의셋을 만든다. VAL 구간(2025-10-01~12-31)의
라이프사이클 행은 현직 exit head 가 학습에서 본 적이 없고, TabPFN 도 컨텍스트에 넣지 않는다 —
양쪽 모두에게 진짜 홀드아웃이다. 라벨 정의·피쳐 구성은 빌더를 재구현하지 않고 그대로 호출해
동일성을 보장한다(`entry_label_terminal_giveback`, Phase 2 스윕이 실제로 쓴 모드).

⚠️ `risk_margin=None/risk_leverage=None` 은 p72 부트스트랩과 동일하다 — 이 번들에 대응하는
사이드카가 아직 없으므로 pos_notional/pos_leverage/pos_exposure 가 BASE_TEMPLATE 상수로
채워진다. 리포트에 `risk_sizing_source` 로 기록한다(Position-Feature Parity 계약).
두 모델이 **같은 피쳐**를 보므로 비교 자체는 편향되지 않는다.

킬 게이트는 direction 과 동일(K1 대조군 / K2 현직 / K2b 시드 과반 / K3 컨텍스트 단조).
시드 과반은 사용자 지시에 따른 **페이즈 진행** 기준이며 승격 근거가 아니다.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
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
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
from research_omega461_tabicl_direction_head_20260909 import (  # noqa: E402
    DIR_LBL, ROUTE_COLS_SPINE, SEEDS, QUERY_SEED, _binom_ge, _controls, _score,
    _select_ctx, _fit_predict,
)

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/exit_head"
# exit 라벨은 9.2% 만 양성(TRAIN 27,228/2,772)이라 **균형 천장이 2x2,772 = 5,544행**이다.
# 그 위 칸은 균형이 깨진 상태를 재는 것이고, 그러면 다수 클래스 쪽으로 쏠려 acc 는 기저율
# (90.15%)로 오르고 bal_acc 는 떨어진다 — 실측 확인(8k 균형 0.7004 → 전량 0.6500, acc 는
# 0.7792 → 0.9043 으로 반대 방향). 사다리를 천장 안으로 자른다. direction 에서 한 것과 같다.
LADDER = [1000, 2000, 3500, 5500]
QUERY_CAP = 6000
EXIT_KW = dict(max_samples=30000, terminal_window=3, adverse_unreal=-0.010,
               min_mfe_for_giveback=0.006, giveback_min=0.65)   # Phase 2 스윕 인자와 동일
COST_MULT = 3.0


def _build_exit(frame: pd.DataFrame, base_cols: list[str], what: str) -> tuple:
    """p72 의 빌더를 그대로 호출한다(재구현 금지). TRAIN/VAL 에 같은 함수·같은 인자."""
    fee, slip = omega._load_fee_slip()
    tabm = omega._read(omega.TABM_2025)
    df, _src = omega._align(frame, tabm, what)
    state = parent._base_input(df, base_cols)
    x_raw, y, frame_exit, diag = p72._build_exit_dataset_entry_label_terminal_giveback(
        df, state, risk_margin=None, risk_leverage=None,
        fee=fee, slip=slip, cost_mult=COST_MULT, **EXIT_KW)
    x = parent._exit_input_from_position_rows(x_raw, base_cols)
    return x, np.asarray(y, dtype=np.int64), frame_exit, diag


def _incumbent_exit(x_q: pd.DataFrame, route_q: np.ndarray, base_cols: list[str],
                    comp: str, seed: int) -> dict | None:
    """현직 TabM 의 exit head — `_metrics_with_shared_exit` 와 같은 경로(레짐 argmax 라우팅)."""
    import torch
    pdir = (ROOT / "tmp/causal_regen_20260516" /
            f"omega4_3head_parent72_loose_entry_quality_20260620_regimespine_balnobb_{comp}_s{seed}_20260909")
    bp = pdir / "true_3head_tabm_bundle.pt"
    if not bp.exists():
        return None
    bundle = torch.load(bp, map_location="cpu", weights_only=False)
    if list(bundle["base_cols"]) != list(base_cols):
        raise RuntimeError("현직 번들 base_cols 불일치 — 짝 비교 불가")
    dev = parent._device("cpu")
    preds = {e: parent._predict_payload(bundle["models"][e], x_q, device=dev)
             for e in hard.EXPERT_NAMES}
    exit_p = parent._routed(preds, route_q, "exit", 2)
    return {"bundle": str(bp), "pred": np.argmax(exit_p, axis=1).astype(np.int64)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="zig075", choices=["zig075", "h48qual"])
    ap.add_argument("--model", default="tabpfn", choices=["tabpfn", "tabicl"])
    ap.add_argument("--seeds", default=",".join(str(x) for x in SEEDS))
    ap.add_argument("--n-estimators", type=int, default=16)
    ap.add_argument("--balance-context", default="balanced", choices=["balanced", "raw"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="global,moe")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    balanced = str(args.balance_context) == "balanced"

    base_cols = spine._install(args.component)
    frames = p72._prepare_frames(
        disable_tp_sl=False, direction_label_dir=DIR_LBL,
        quality_mode="same_as_direction", quality_label_dir=None,
        quality_min_edge=0.0010, quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20, quality_max_hold_bars=288)

    x_tr, y_tr, f_tr, diag_tr = _build_exit(frames["train_raw"], base_cols, "exit train")
    x_va, y_va, f_va, diag_va = _build_exit(frames["val_raw"], base_cols, "exit val")
    print(f"\n[exit 데이터셋] TRAIN {len(y_tr):,}행 · VAL {len(y_va):,}행 · 피쳐 {x_tr.shape[1]}열",
          flush=True)
    print(f"  TRAIN 클래스 {np.bincount(y_tr, minlength=2).tolist()}  "
          f"VAL {np.bincount(y_va, minlength=2).tolist()}", flush=True)
    _rsrc = next((f"{k}={diag_tr[k]}" for k in diag_tr if "risk" in k.lower()), "미기록")
    print(f"  risk_sizing_source = {_rsrc} "
          f"(사이드카 부재 → BASE_TEMPLATE 상수, 양 모델 동일)", flush=True)

    qrng = np.random.default_rng(QUERY_SEED)
    qidx = (np.arange(len(y_va)) if len(y_va) <= QUERY_CAP
            else np.sort(qrng.choice(len(y_va), size=QUERY_CAP, replace=False)))
    xq = x_va.iloc[qidx].reset_index(drop=True)
    yq = y_va[qidx]
    route_tr = np.argmax(f_tr[ROUTE_COLS_SPINE].to_numpy(np.float64), axis=1)
    route_q = np.argmax(f_va[ROUTE_COLS_SPINE].to_numpy(np.float64), axis=1)[qidx]
    print(f"  질의 {len(yq):,}행 (VAL 라이프사이클 행 — 현직 exit head 도 학습에서 본 적 없음)",
          flush=True)

    rep: dict[str, Any] = {
        "head": "exit", "component": args.component, "model": str(args.model), "seeds": seeds,
        "n_estimators": int(args.n_estimators), "balance_context": str(args.balance_context),
        "exit_label_mode": "entry_label_terminal_giveback", "exit_kwargs": EXIT_KW,
        "cost_mult": COST_MULT, "ladder": LADDER,
        "n_train": int(len(y_tr)), "n_val": int(len(y_va)), "n_query": int(len(yq)),
        "risk_sizing_source": {k: diag_tr[k] for k in diag_tr if "risk" in k.lower()} or None,
        "exit_diag": {"train": {k: v for k, v in diag_tr.items() if not isinstance(v, (list, dict))},
                      "val": {k: v for k, v in diag_va.items() if not isinstance(v, (list, dict))}},
        "seed_rule": "majority (user 2026-09-09) — phase progression only, NOT a promotion basis"}

    rep["controls"] = _controls(y_tr, yq, np.random.default_rng(QUERY_SEED))
    print(f"\n[대조군]", flush=True)
    for k, v in rep["controls"].items():
        print(f"  {k:18s} bal_acc {v['bal_acc']:.4f}  acc {v['acc']:.4f}", flush=True)
    ctrl_best = max(v["bal_acc"] for k, v in rep["controls"].items() if k != "majority")

    rep["per_seed"] = {}
    for seed in seeds:
        print(f"\n{'='*96}\n[시드 {seed}]", flush=True)
        one: dict[str, Any] = {}
        inc = _incumbent_exit(xq, route_q, base_cols, args.component, seed)
        one["incumbent"] = None if inc is None else {"bundle": inc["bundle"], **_score(yq, inc["pred"])}
        if one["incumbent"]:
            i = one["incumbent"]
            print(f"  현직 TabM exit  bal_acc {i['bal_acc']:.4f}  acc {i['acc']:.4f}  "
                  f"macroF1 {i['macro_f1']:.4f}", flush=True)
        else:
            print("  현직 번들 없음 — K2 제외", flush=True)
        one["arms"] = {}
        for arm in arms:
            print(f"  [팔: {arm}]{'' if arm == 'global' else '  라우팅=balnobb'}", flush=True)
            out = {}
            for rung in LADDER:
                n_ctx = len(y_tr) if rung == 0 else min(rung, len(y_tr))
                tag = "전량" if rung == 0 else f"{rung//1000}k"
                pool = np.arange(len(y_tr))
                if arm == "global":
                    sel, cdiag = _select_ctx(y_tr, pool, n_ctx, balanced, seed)
                else:
                    sel, cdiag = pool[-n_ctx:], {"per_expert": True}
                try:
                    if arm == "global":
                        pred, meta = _fit_predict(x_tr.iloc[sel].reset_index(drop=True), y_tr[sel],
                                                  xq, n_estimators=args.n_estimators, seed=seed,
                                                  device=args.device, kind=str(args.model))
                    else:
                        pred = np.full(len(yq), -1, dtype=np.int64)
                        meta = {"fit_s": 0.0, "predict_s": 0.0, "peak_gpu_mib": 0.0,
                                "context_rows": 0, "expert_rows": {}}
                        for e in range(3):
                            cm, _d = _select_ctx(y_tr, sel[route_tr[sel] == e], n_ctx // 3,
                                                 balanced, seed + e)
                            qm = np.where(route_q == e)[0]
                            meta["expert_rows"][hard.EXPERT_NAMES[e]] = [int(len(cm)), int(len(qm))]
                            if len(qm) == 0:
                                continue
                            if len(cm) < 50 or len(np.unique(y_tr[cm])) < 2:
                                pred[qm] = int(np.bincount(y_tr[sel], minlength=2).argmax())
                                continue
                            pp, m = _fit_predict(x_tr.iloc[cm].reset_index(drop=True), y_tr[cm],
                                                 xq.iloc[qm].reset_index(drop=True),
                                                 n_estimators=args.n_estimators, seed=seed,
                                                 device=args.device, kind=str(args.model))
                            pred[qm] = pp
                            meta["fit_s"] += m["fit_s"]; meta["predict_s"] += m["predict_s"]
                            meta["context_rows"] += m["context_rows"]
                            meta["peak_gpu_mib"] = max(meta["peak_gpu_mib"], m["peak_gpu_mib"])
                        if (pred < 0).any():
                            raise RuntimeError("moe: 채워지지 않은 질의 행")
                    sc = _score(yq, pred)
                    out[tag] = {**sc, **meta, "ctx_diag": cdiag}
                    print(f"    ctx {tag:5s} ({meta['context_rows']:>6,}행)  bal_acc {sc['bal_acc']:.4f}"
                          f"  acc {sc['acc']:.4f}  F1 {sc['macro_f1']:.4f}  "
                          f"pred {meta['predict_s']:.0f}s  GPU {meta['peak_gpu_mib']:.0f}MiB", flush=True)
                except Exception as exc:
                    out[tag] = {"error": f"{type(exc).__name__}: {exc}"}
                    print(f"    ctx {tag:5s}  ❌ {type(exc).__name__}: {str(exc)[:140]}", flush=True)
            one["arms"][arm] = out
        rep["per_seed"][str(seed)] = one
        (OUT / f"exit_{args.model}_{args.component}_5seed.json").write_text(
            json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")

    # ── 집계 · 게이트 ──
    tags = [("전량" if r == 0 else f"{r//1000}k") for r in LADDER]
    cells = {}
    for arm in arms:
        for tag in tags:
            accs, deltas = [], []
            for s in seeds:
                one = rep["per_seed"][str(s)]
                d = one["arms"].get(arm, {}).get(tag)
                if not d or "bal_acc" not in d:
                    continue
                accs.append(d["bal_acc"])
                if one["incumbent"]:
                    deltas.append(d["bal_acc"] - one["incumbent"]["bal_acc"])
            if accs:
                cells[f"{arm}|{tag}"] = {
                    "n_seeds": len(accs), "median_bal_acc": statistics.median(accs),
                    "min": min(accs), "max": max(accs), "n_delta": len(deltas),
                    "median_delta": statistics.median(deltas) if deltas else None,
                    "n_better": sum(1 for x in deltas if x > 0)}
    inc_accs = [rep["per_seed"][str(s)]["incumbent"]["bal_acc"] for s in seeds
                if rep["per_seed"][str(s)]["incumbent"]]
    print(f"\n{'='*96}\n[시드 집계] N={len(seeds)} (VAL 전용)", flush=True)
    if inc_accs:
        print(f"  현직 TabM exit 중앙 {statistics.median(inc_accs):.4f} "
              f"[{min(inc_accs):.4f}, {max(inc_accs):.4f}]", flush=True)
    print(f"  대조군 최고 {ctrl_best:.4f}\n", flush=True)
    print(f"  {'셀':14s} {'중앙bal':>8s} {'[최소,최대]':>17s} {'Δ중앙':>9s} {'이김':>6s}", flush=True)
    for k, v in cells.items():
        dv = "n/a" if v["median_delta"] is None else f"{v['median_delta']:+.4f}"
        print(f"  {k:14s} {v['median_bal_acc']:8.4f} [{v['min']:.4f},{v['max']:.4f}] {dv:>9s} "
              f"{v['n_better']:>3d}/{v['n_delta']:<2d}", flush=True)
    rep["cells"] = cells

    best_k, best = max(cells.items(), key=lambda kv: kv[1]["median_bal_acc"])
    n_d = best["n_delta"]
    k1 = best["median_bal_acc"] > ctrl_best
    k2 = best["median_delta"] is not None and best["median_delta"] > 0
    k2b = n_d > 0 and best["n_better"] * 2 > n_d
    k2b_strict = best["n_better"] == n_d == len(seeds)
    seed_p = _binom_ge(best["n_better"], n_d) if n_d else None
    mono = {}
    for arm in arms:
        xs = [cells[f"{arm}|{t}"]["median_bal_acc"] for t in tags if f"{arm}|{t}" in cells]
        mono[arm] = round(xs[-1] - xs[0], 4) if len(xs) >= 2 else None
    k3 = any(v is not None and v > 0 for v in mono.values())
    allp = k1 and k2 and k2b and k3
    print(f"\n{'='*96}\n[킬 게이트]  최선 셀 {best_k}  중앙 {best['median_bal_acc']:.4f}", flush=True)
    print(f"  K1  대조군 우위 (>{ctrl_best:.4f})       : {'✅' if k1 else '❌'}", flush=True)
    print(f"  K2  현직 우위 (Δ중앙 "
          f"{'n/a' if best['median_delta'] is None else format(best['median_delta'], '+.4f')})   "
          f": {'✅' if k2 else '❌'}", flush=True)
    print(f"  K2b 시드 과반 ({best['n_better']}/{n_d})            : {'✅' if k2b else '❌'}"
          f"   [귀무 p={seed_p:.3f} · 엄격 {'✅' if k2b_strict else '❌'}]", flush=True)
    print(f"  K3  컨텍스트 단조 {mono}   : {'✅' if k3 else '❌'}", flush=True)
    print(f"\n  → 다음 페이즈 진행 {'허가' if allp else '불가'}"
          f" — 정확도 통과는 채택 근거가 아니다(판정은 결합 후 결정층 PnL).", flush=True)
    rep["verdict"] = {"best_cell": best_k, "median_bal_acc": best["median_bal_acc"],
                      "control_best_bal_acc": ctrl_best, "K1": bool(k1), "K2": bool(k2),
                      "K2b_seed_majority": bool(k2b), "K2b_seed_all_agree_strict": bool(k2b_strict),
                      "seed_rule_null_p": seed_p, "K3": bool(k3), "context_monotonicity": mono,
                      "next_phase_allowed": bool(allp)}
    p = OUT / f"exit_{args.model}_{args.component}_5seed.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
