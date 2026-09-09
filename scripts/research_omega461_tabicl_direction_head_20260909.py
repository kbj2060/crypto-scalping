"""TabICL 을 Omega 4.6.1 **direction head** 에 설계 특성대로 적용한다 (Stage 0 게이트).

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계약: docs/model_contracts/omega461_regimegbm_rebuild_contract.md
결정 근거: 2026-09-09 사용자 "tabicl과 tabpfn 의 설계 특징에 맞게 잘 적용해줘" / "병행해줘"

왜 direction head 인가 — TabICL 의 설계 사거리
---------------------------------------------
TabPFN 은 소표본 in-context 학습기다(이 저장소 관례상 컨텍스트 상한 18,000행). 반면 TabICL 은
**대형 테이블(~100k행)** 을 겨냥해 만들어졌다 — column-wise 임베딩으로 각 피쳐의 분포를 먼저
요약하고, 그 위에서 row-wise 어텐션을 돌린 뒤 데이터셋 전체를 컨텍스트로 삼는 2단 구조다.

Omega 4.6.1 안에서 이 사거리에 실제로 들어오는 헤드는 direction head 뿐이다:
  · direction head : 2025 학습 프레임 전체(수만~10만행) × 102 base col × 3클래스  ← TabICL 영역
  · risk sidecar   : 학습 표본 75건(롱31/숏44)                                    ← TabPFN 영역(트랙 ①)
사이드카에 TabICL 을 쓰는 건 설계 오용이고, direction head 에 TabPFN 을 쓰면 컨텍스트 상한 때문에
학습행의 80% 이상을 버려야 한다. 그래서 두 트랙을 나눈다.

설계에 맞춘 3가지 결정
---------------------
1) **표준화하지 않는다.** TabICL 의 column-wise 임베딩은 분포 인식형이고 내부 `norm_methods`
   (none/power 등)를 앙상블 축으로 쓴다. 밖에서 z-score 를 먹여 넣으면 그 축을 죽인다.
   그래서 `parent._base_input` 의 **원시 102열**을 그대로 넣는다(`_standardize_apply` 미사용).
2) **MoE 라우팅을 sample_weight 로 못 넘긴다.** 현행 TabM 은 레짐 확률을 행별 가중치로 써서
   전문가 3개를 소프트 분할한다. TabICL 은 sample_weight 를 받지 않으므로 두 팔로 나눈다:
     · `global` — 전문가 분할 없음. 레짐 6열이 이미 피쳐 안에 있고, in-context 학습기는
                  컨텍스트 전체를 조건으로 삼으므로 레짐별 거동을 문맥에서 학습한다.
                  **이쪽이 TabICL 설계에 맞는 형태다**(MoE 는 작은 MLP 의 표현력 한계를
                  우회하려던 장치인데, TabICL 엔 그 제약이 없다).
     · `moe`    — argmax(레짐)으로 컨텍스트를 하드 분할한 현행 구조의 충실한 이식.
3) **컨텍스트 사다리.** TabICL 의 주장 자체가 "컨텍스트가 클수록 좋다"이므로 그 주장을
   이 데이터에서 직접 잰다(4k→전량). 동시에 이 8GB 카드(라이브 스택과 공유, 여유 ~2.8GB)에서
   어느 사다리 칸까지 실제로 올라갈 수 있는지가 배포 가능성의 상한이다. OOM 은 실패로 기록만
   하고 다음 칸으로 넘어간다 — 조용히 축소하지 않는다.

사전 등록 킬 게이트 (VAL 에서만 판정, `validation_only`)
-------------------------------------------------------
K1  대조군 우위 : TabICL 최선 팔의 VAL balanced accuracy 가 **자명 대조군 4종 전부**
                  (always_0/1/2 + stratified prior) 보다 높아야 한다.
                  ※ 이 저장소에서 라벨×모델 조합 7건을 닫은 게 정확히 이 대조군이다.
K2  현직 우위   : 같은 VAL 행에서 현직 TabM(Phase 2 balnobb 번들, 라우팅 적용) 의
                  balanced accuracy 를 넘어야 한다.
K3  컨텍스트 단조: 컨텍스트를 4k→전량으로 늘릴 때 bal_acc 가 **증가**해야 한다
                  (증가하지 않으면 TabICL 을 쓸 이유 자체가 없다 — TabPFN 으로 충분).
K1~K3 을 모두 통과해야 Stage 1(경제성: `_prediction_output` → 결정 → PnL)로 간다.
정확도만 오르고 경제성이 0인 사례가 이 저장소에 반복해 있었으므로, K1~K3 통과는
**진행 허가일 뿐 채택 근거가 아니다.**
"""
from __future__ import annotations

import argparse
import json
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
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/tabicl_direction"
ROUTE_COLS_NEW = [f"{spine.NEW_PREFIX}{c}_prob" for c in ("bull", "bear", "chop")]
LADDER = [4000, 8000, 16000, 32000, 64000, 0]      # 0 = 전량
QUERY_CAP = 6000                                    # 사다리 단계의 VAL 질의 표본(고정 시드)
SEED = 615372041


def _bal_acc(y: np.ndarray, p: np.ndarray) -> float:
    accs = []
    for c in np.unique(y):
        m = y == c
        if m.sum():
            accs.append(float((p[m] == c).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def _macro_f1(y: np.ndarray, p: np.ndarray) -> float:
    f1s = []
    for c in np.unique(y):
        tp = float(((p == c) & (y == c)).sum())
        fp = float(((p == c) & (y != c)).sum())
        fn = float(((p != c) & (y == c)).sum())
        f1s.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1s)) if f1s else float("nan")


def _score(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {"bal_acc": _bal_acc(y, p), "acc": float((p == y).mean()), "macro_f1": _macro_f1(y, p),
            "n": int(len(y))}


def _controls(y_ctx: np.ndarray, y_q: np.ndarray, rng: np.random.Generator) -> dict[str, dict]:
    """자명 대조군 — 이 저장소에서 조합 7건을 닫은 바로 그 대조군."""
    out = {}
    for c in (0, 1, 2):
        out[f"always_{c}"] = _score(y_q, np.full(len(y_q), c, dtype=np.int64))
    prior = np.bincount(y_ctx, minlength=3).astype(np.float64)
    prior /= prior.sum()
    out["stratified_prior"] = _score(y_q, rng.choice(3, size=len(y_q), p=prior).astype(np.int64))
    out["majority"] = out[f"always_{int(np.argmax(prior))}"]
    return out


def _tabicl(n_estimators: int, seed: int, device: str):
    from tabicl import TabICLClassifier
    return TabICLClassifier(n_estimators=int(n_estimators), random_state=int(seed),
                            device=device, offload_mode="auto", batch_size=2, verbose=False)


def _gpu_mib() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 2 ** 20)
    except Exception:
        pass
    return float("nan")


def _fit_predict(x_ctx: pd.DataFrame, y_ctx: np.ndarray, x_q: pd.DataFrame, *,
                 n_estimators: int, seed: int, device: str) -> tuple[np.ndarray, dict]:
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    clf = _tabicl(n_estimators, seed, device)
    t0 = time.time()
    clf.fit(x_ctx.to_numpy(np.float32), y_ctx)
    t_fit = time.time() - t0
    t0 = time.time()
    pred = clf.predict(x_q.to_numpy(np.float32)).astype(np.int64)
    t_pred = time.time() - t0
    return pred, {"fit_s": round(t_fit, 1), "predict_s": round(t_pred, 1),
                  "peak_gpu_mib": round(_gpu_mib(), 1), "context_rows": int(len(y_ctx))}


def _incumbent(val_raw: pd.DataFrame, x_val: pd.DataFrame, comp: str, seed: int,
               qidx: np.ndarray) -> dict | None:
    """Phase 2 balnobb 부모 번들의 direction head — 같은 VAL 행, 같은 라우팅."""
    import torch
    pdir = (ROOT / "tmp/causal_regen_20260516" /
            f"omega4_3head_parent72_loose_entry_quality_20260620_regimespine_balnobb_{comp}_s{seed}_20260909")
    bp = pdir / "true_3head_tabm_bundle.pt"
    if not bp.exists():
        return None
    bundle = torch.load(bp, map_location="cpu", weights_only=False)
    if list(bundle["base_cols"]) != list(x_val.columns):
        raise RuntimeError("현직 번들 base_cols 가 이 프레임과 다르다 — 짝 비교 불가")
    dev = parent._device("cpu")
    preds = {e: parent._predict_payload(bundle["models"][e], x_val, device=dev) for e in hard.EXPERT_NAMES}
    route = np.argmax(val_raw[ROUTE_COLS_NEW].to_numpy(np.float64), axis=1).astype(np.int64)
    direction = parent._routed(preds, route, "direction", 3)
    return {"bundle": str(bp), "pred": np.argmax(direction, axis=1).astype(np.int64)[qidx]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", default="zig075", choices=["zig075", "h48qual"])
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-estimators", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--arms", default="global,moe")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.seed))

    base_cols = spine._install(args.component)
    frames = parent._prepare_frames(disable_tp_sl=False)
    train_raw, val_raw = frames["train_raw"], frames["val_raw"]
    x_tr = parent._base_input(train_raw, base_cols)
    y_tr = train_raw["zigzag_action"].to_numpy(np.int64)
    x_va = parent._base_input(val_raw, base_cols)
    y_va = val_raw["zigzag_action"].to_numpy(np.int64)
    print(f"\n[데이터] TRAIN {len(y_tr):,}행 · VAL {len(y_va):,}행 · 피쳐 {x_tr.shape[1]}열", flush=True)
    print(f"  TRAIN 클래스 분포 {np.bincount(y_tr, minlength=3).tolist()}  "
          f"VAL {np.bincount(y_va, minlength=3).tolist()}", flush=True)

    qidx = np.sort(rng.choice(len(y_va), size=min(QUERY_CAP, len(y_va)), replace=False))
    xq, yq = x_va.iloc[qidx].reset_index(drop=True), y_va[qidx]
    print(f"  질의 표본 {len(yq):,}행 (고정 시드 {args.seed})", flush=True)

    rep: dict[str, Any] = {"component": args.component, "seed": int(args.seed),
                           "n_estimators": int(args.n_estimators),
                           "n_train": int(len(y_tr)), "n_val": int(len(y_va)),
                           "n_query": int(len(yq)), "ladder": LADDER}

    rep["controls"] = _controls(y_tr, yq, rng)
    print(f"\n[대조군] (VAL 질의 {len(yq):,}행)", flush=True)
    for k, v in rep["controls"].items():
        print(f"  {k:18s} bal_acc {v['bal_acc']:.4f}  acc {v['acc']:.4f}  macroF1 {v['macro_f1']:.4f}", flush=True)
    ctrl_best = max(v["bal_acc"] for k, v in rep["controls"].items() if k != "majority")

    inc = _incumbent(val_raw, x_va, args.component, int(args.seed), qidx)
    if inc is None:
        print("\n[현직] Phase 2 번들 없음 — K2 판정 불가", flush=True)
        rep["incumbent"] = None
    else:
        rep["incumbent"] = {"bundle": inc["bundle"], **_score(yq, inc["pred"])}
        s = rep["incumbent"]
        print(f"\n[현직 TabM] bal_acc {s['bal_acc']:.4f}  acc {s['acc']:.4f}  macroF1 {s['macro_f1']:.4f}",
              flush=True)

    route_tr = np.argmax(train_raw[ROUTE_COLS_NEW].to_numpy(np.float64), axis=1)
    route_q = np.argmax(val_raw[ROUTE_COLS_NEW].to_numpy(np.float64), axis=1)[qidx]

    rep["arms"] = {}
    for arm in [a.strip() for a in args.arms.split(",") if a.strip()]:
        rep["arms"][arm] = {}
        print(f"\n{'='*92}\n[팔: {arm}] 컨텍스트 사다리", flush=True)
        for rung in LADDER:
            n_ctx = len(y_tr) if rung == 0 else min(rung, len(y_tr))
            tag = "전량" if rung == 0 else f"{rung//1000}k"
            # 컨텍스트는 **가장 최근** n_ctx 행 — 인과적으로 VAL 직전 구간이고, 무작위 부분표집이
            # 아니라 실제 배포에서 쓸 수 있는 형태다.
            sel = np.arange(len(y_tr) - n_ctx, len(y_tr))
            try:
                if arm == "global":
                    pred, meta = _fit_predict(x_tr.iloc[sel].reset_index(drop=True), y_tr[sel], xq,
                                              n_estimators=args.n_estimators, seed=args.seed,
                                              device=args.device)
                else:
                    pred = np.full(len(yq), -1, dtype=np.int64)
                    meta = {"fit_s": 0.0, "predict_s": 0.0, "peak_gpu_mib": 0.0, "context_rows": 0,
                            "expert_rows": {}}
                    for e in range(3):
                        cm = sel[route_tr[sel] == e]
                        qm = np.where(route_q == e)[0]
                        meta["expert_rows"][hard.EXPERT_NAMES[e]] = [int(len(cm)), int(len(qm))]
                        if len(qm) == 0:
                            continue
                        if len(cm) < 50 or len(np.unique(y_tr[cm])) < 2:
                            pred[qm] = int(np.bincount(y_tr[sel], minlength=3).argmax())
                            continue
                        p, m = _fit_predict(x_tr.iloc[cm].reset_index(drop=True), y_tr[cm],
                                            xq.iloc[qm].reset_index(drop=True),
                                            n_estimators=args.n_estimators, seed=args.seed,
                                            device=args.device)
                        pred[qm] = p
                        meta["fit_s"] += m["fit_s"]; meta["predict_s"] += m["predict_s"]
                        meta["context_rows"] += m["context_rows"]
                        meta["peak_gpu_mib"] = max(meta["peak_gpu_mib"], m["peak_gpu_mib"])
                    if (pred < 0).any():
                        raise RuntimeError("moe: 예측이 채워지지 않은 질의 행이 있다")
                sc = _score(yq, pred)
                rep["arms"][arm][tag] = {**sc, **meta}
                print(f"  ctx {tag:5s} ({meta['context_rows']:>7,}행)  bal_acc {sc['bal_acc']:.4f}  "
                      f"acc {sc['acc']:.4f}  macroF1 {sc['macro_f1']:.4f}  "
                      f"fit {meta['fit_s']:.0f}s pred {meta['predict_s']:.0f}s "
                      f"peakGPU {meta['peak_gpu_mib']:.0f}MiB", flush=True)
            except Exception as exc:  # OOM 포함 — 기록하고 계속(조용한 축소 금지)
                rep["arms"][arm][tag] = {"error": f"{type(exc).__name__}: {exc}"}
                print(f"  ctx {tag:5s}  ❌ {type(exc).__name__}: {str(exc)[:160]}", flush=True)
                try:
                    import torch
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    # ── 킬 게이트 판정 ──
    ok = {k: v for arm in rep["arms"] for k, v in
          [(f"{arm}|{t}", d) for t, d in rep["arms"][arm].items() if "bal_acc" in d]}
    print(f"\n{'='*92}\n[킬 게이트]", flush=True)
    if not ok:
        print("  실행 가능한 팔이 하나도 없다 — 판정 불가", flush=True)
        rep["verdict"] = {"K1": None, "K2": None, "K3": None}
    else:
        best_k, best = max(ok.items(), key=lambda kv: kv[1]["bal_acc"])
        k1 = best["bal_acc"] > ctrl_best
        k2 = None if rep["incumbent"] is None else best["bal_acc"] > rep["incumbent"]["bal_acc"]
        mono = {}
        for arm in rep["arms"]:
            xs = [(LADDER[i] or len(y_tr), rep["arms"][arm][t]["bal_acc"])
                  for i, t in enumerate(["4k", "8k", "16k", "32k", "64k", "전량"])
                  if t in rep["arms"][arm] and "bal_acc" in rep["arms"][arm][t]]
            mono[arm] = (round(xs[-1][1] - xs[0][1], 4), len(xs)) if len(xs) >= 2 else None
        k3 = any(v is not None and v[0] > 0 for v in mono.values())
        print(f"  최선 팔  : {best_k}  bal_acc {best['bal_acc']:.4f}", flush=True)
        print(f"  K1 대조군 우위 (>{ctrl_best:.4f})         : {'✅' if k1 else '❌'}", flush=True)
        print(f"  K2 현직 우위  "
              f"({'n/a' if rep['incumbent'] is None else format(rep['incumbent']['bal_acc'], '.4f')})"
              f"            : {'n/a' if k2 is None else ('✅' if k2 else '❌')}", flush=True)
        print(f"  K3 컨텍스트 단조 증가 {mono}  : {'✅' if k3 else '❌'}", flush=True)
        rep["verdict"] = {"best_arm": best_k, "best_bal_acc": best["bal_acc"],
                          "control_best_bal_acc": ctrl_best, "K1": bool(k1),
                          "K2": None if k2 is None else bool(k2), "K3": bool(k3),
                          "context_monotonicity": mono}
        allp = bool(k1) and (k2 is not False) and bool(k3)
        print(f"\n  → Stage 1(경제성) 진행 {'허가' if allp else '불가'}"
              f" — 통과해도 채택 근거는 아니다(경제성 별도)", flush=True)
        rep["verdict"]["stage1_allowed"] = allp

    p = OUT / f"tabicl_direction_{args.component}_s{args.seed}.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    print(f"\n산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
