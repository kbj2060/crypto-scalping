"""TabICL 라이브 서빙 가능성 검증 — kv_cache 모드별 상주 메모리·봉당 지연.

하위 프로젝트: `omega461_regimegbm_rebuild_20260909`
계획 2번(사용자 승인 2026-09-09): "kv_cache 서빙 검증 — 블로커. GPU 가 비는 대로."

왜 이게 블로커인가
------------------
부모를 TabICL 3모델로 대체하면 in-context 모델이 **동시에 여러 개** 상주해야 한다:
  zig075  : direction (+ quality 는 same_as_direction 이라 축퇴 → 불필요) + exit
  h48qual : direction + quality + exit
= 최대 5개. 그런데 카드는 8GB 한 장이고 **라이브 스택(trading_bot + 대시보드)과 공유**한다.
연구 실행에서 잰 단일 모델 피크가 3.6~6.7 GiB 였으므로 캐시 없이는 산술적으로 안 올라간다.

in-context 학습기는 `fit()` 이 컨텍스트를 저장할 뿐이고 `predict()` 가 매번 컨텍스트 전체를
다시 통과한다. 라이브는 5분봉마다 **1행**을 물어보는데, 그 1행 때문에 32k 컨텍스트를 매번
인코딩하면 지연·메모리 둘 다 감당이 안 된다. `kv_cache` 가 그 재계산을 없애는 기제다:
  False  : 캐시 없음
  "kv"   : 컬럼임베딩 + ICL 트랜스포머 KV 캐시 (빠르지만 큰 컨텍스트에서 메모리 과다)
  "repr" : 컬럼임베딩 KV + row interaction 출력 캐시 (ICL 부분 ~24배 절약, 예측 때 ICL 재실행)

무엇을 재는가
------------
  1) `torch.cuda.mem_get_info()` 로 **실제 여유 메모리**(라이브 스택 점유 후 남는 양)
  2) 모드별 fit 직후 **상주** 메모리(피크가 아니라 계속 물고 있는 양) — 동시 상주 가능성의 근거
  3) 모델을 1개씩 늘려가며 N개까지 올라가는지 (OOM 은 기록하고 중단)
  4) **봉당 지연** = 1행 predict 반복 측정. 라이브의 실제 질의 형태다.

데이터는 합성이다 — 이 검증의 답(메모리·지연)은 **형상**이 결정하고 분포는 무관하다.
합성으로 여유가 없으면 실데이터로도 없다. 여유가 있으면 실제 빌드에서 실데이터로 재확인한다.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tmp/omega461_regimegbm_rebuild_20260909/tabicl_serving"
MIB = 1024 ** 2


def _free_total() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return float("nan"), float("nan")
    free, total = torch.cuda.mem_get_info()
    return free / MIB, total / MIB


def _resident() -> float:
    return torch.cuda.memory_allocated() / MIB if torch.cuda.is_available() else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-ctx", type=int, default=32000, help="direction head 최선 사다리 칸")
    ap.add_argument("--n-feat", type=int, default=115, help="base_cols 102 + POS_COLS 13")
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--n-models", type=int, default=5, help="동시 상주 목표(zig075 2 + h48qual 3)")
    ap.add_argument("--n-estimators", type=int, default=16)
    ap.add_argument("--modes", default="repr,kv,False")
    ap.add_argument("--n-latency", type=int, default=20, help="봉당 지연 측정 반복")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from tabicl import TabICLClassifier
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(615372041)
    X = rng.standard_normal((args.n_ctx, args.n_feat)).astype(np.float32)
    y = rng.integers(0, args.n_classes, args.n_ctx)
    x1 = rng.standard_normal((1, args.n_feat)).astype(np.float32)   # 라이브 = 봉당 1행

    f0, tot = _free_total()
    rep = {"n_ctx": args.n_ctx, "n_feat": args.n_feat, "n_classes": args.n_classes,
           "n_estimators": args.n_estimators, "target_models": args.n_models,
           "gpu_total_mib": round(tot, 1), "gpu_free_at_start_mib": round(f0, 1),
           "note": "합성 데이터. 메모리·지연은 형상이 결정하므로 분포는 무관하다.",
           "modes": {}}
    print(f"[GPU] 전체 {tot:,.0f} MiB · 시작 시 여유 {f0:,.0f} MiB "
          f"(= 라이브 스택 점유 후 남는 실제 예산)", flush=True)
    print(f"[형상] 컨텍스트 {args.n_ctx:,}행 × {args.n_feat}열 × {args.n_classes}클래스 · "
          f"n_estimators={args.n_estimators}\n", flush=True)

    for raw in [m.strip() for m in args.modes.split(",") if m.strip()]:
        mode = False if raw in ("False", "false", "0") else raw
        print(f"{'='*88}\n[kv_cache = {raw}]", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        held, res = [], {"models": [], "error": None}
        try:
            for i in range(args.n_models):
                clf = TabICLClassifier(n_estimators=args.n_estimators, random_state=615372041,
                                       device=args.device, offload_mode="auto", batch_size=2,
                                       kv_cache=mode, verbose=False)
                t0 = time.time(); clf.fit(X, y); t_fit = time.time() - t0
                held.append(clf)
                lat = []
                for _ in range(args.n_latency):
                    t1 = time.time(); clf.predict(x1); lat.append(time.time() - t1)
                free_now, _ = _free_total()
                row = {"model_index": i + 1, "fit_s": round(t_fit, 1),
                       "resident_mib": round(_resident(), 1),
                       "gpu_free_mib": round(free_now, 1),
                       "bar_latency_median_s": round(float(np.median(lat)), 3),
                       "bar_latency_max_s": round(float(np.max(lat)), 3)}
                res["models"].append(row)
                print(f"  모델 {i+1}/{args.n_models}  fit {row['fit_s']:6.1f}s  "
                      f"상주 {row['resident_mib']:8.1f} MiB  여유 {row['gpu_free_mib']:8.1f} MiB  "
                      f"봉당 {row['bar_latency_median_s']:6.3f}s (최대 {row['bar_latency_max_s']:.3f}s)",
                      flush=True)
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
            print(f"  ❌ 모델 {len(held)+1} 에서 실패 — {type(exc).__name__}: {str(exc)[:170]}", flush=True)
        res["models_loaded"] = len(held)
        res["fits_target"] = len(held) >= args.n_models
        res["peak_mib"] = round(torch.cuda.max_memory_allocated() / MIB, 1) if torch.cuda.is_available() else None
        print(f"  → {len(held)}/{args.n_models} 상주  피크 {res['peak_mib']:,.1f} MiB  "
              f"{'✅ 목표 충족' if res['fits_target'] else '❌ 목표 미달'}\n", flush=True)
        rep["modes"][raw] = res
        del held
        torch.cuda.empty_cache()

    ok = [m for m, r in rep["modes"].items() if r["fits_target"]]
    rep["verdict"] = {
        "serving_feasible_modes": ok,
        "blocker_cleared": bool(ok),
        "reason": ("동시 상주 목표를 채운 모드 없음 — 부모 3모델 대체는 이 카드에서 서빙 불가. "
                   "컨텍스트 축소 / CPU offload / 컴포넌트 순차 평가 중 택일 필요."
                   if not ok else f"서빙 가능 모드: {', '.join(ok)}")}
    print(f"{'='*88}\n[판정] {rep['verdict']['reason']}", flush=True)
    p = OUT / f"kv_cache_serving_ctx{args.n_ctx}.json"
    p.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"산출물: {p}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
