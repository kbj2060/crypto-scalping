#!/usr/bin/env python3
"""사이징: **TabPFN vs GBM**, 표본 수를 통제해서 (2026-09-12).

사용자: *"tabpfn 모델을 써보는건 어때?"*

## 왜 표본 통제가 핵심인가
TabPFN 은 학습 행 수에 상한이 있다(이 저장소 기록: ~18,000 넘으면 개선폭 소멸,
[[feedback_gbm_proxy_fails_when_sample_size_is_the_driver_20260902]]). 우리 학습셋은 63,893 행이다.
그냥 붙이면 TabPFN 이 져도 **«모델이 나빠서»인지 «표본을 못 써서»인지 갈리지 않는다**.
⇒ 세 팔을 같이 둔다:
    GBM(전체 63,893) · GBM(같은 부분표본 N) · TabPFN(같은 부분표본 N)
  뒤 둘의 차이가 **모델 차이**, 앞 둘의 차이가 **표본 효과**다.

## 🔴GPU 를 안 쓴다
서버 GPU 여유가 1.25 GiB / 8 GiB 뿐이고(대시보드 극점·V자 모델이 점유), 경합이 대시보드
타임아웃을 낸 전례가 있다([[feedback_shared_gpu_contention_causes_dashboard_timeouts_20260903],
V자반등 3초→43초). CPU 로 돌린다 — 느리지만 라이브 위험이 0 이다.

입력은 `research_sizing_feature_expansion_20260912.py` 가 저장한 행렬 그대로다.
**같은 표본·같은 피쳐**여야 비교가 성립한다.

실행: python scripts/research_sizing_tabpfn_vs_gbm_20260912.py [N_TRAIN]
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "tmp" / "sizing_feature_expansion_20260912" / "matrix.npz"
OUT = ROOT / "tmp" / "sizing_tabpfn_20260912"
LEV = 50.0
SEED = 20260912
N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
BATCH = 2000                      # 예측 배치 — 한 번에 다 넣으면 CPU 메모리가 튄다


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def arm_stats(w, ret, mae, abucket, hbucket) -> dict:
    w = w / w.mean()
    pnl = w * ret
    sa = [float(pnl[abucket == b].std(ddof=1)) for b in range(5)]
    sh = [float(pnl[hbucket == b].std(ddof=1)) for b in range(6)]
    return {"flat_atr": max(sa) / min(sa), "flat_hour": max(sh) / min(sh),
            "sd": float(pnl.std(ddof=1)), "p01": float(np.percentile(pnl, 1)),
            "mae_p99": float(np.percentile(w * mae, 99)),
            "stopout_50x": float(np.mean(w * mae > 1e4 / LEV))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")      # ⭐GPU 를 아예 안 보이게 한다
    z = np.load(IN, allow_pickle=True)
    X, y, ret, mae, ap = z["X"], z["y"], z["ret"], z["mae"], z["ap"]
    is_tr, abucket, hbucket = z["is_tr"], z["abucket_ev"], z["hbucket_ev"]
    names = [str(s) for s in z["names"]]
    ev = ~is_tr
    log(f"학습 {is_tr.sum():,} · 평가 {ev.sum():,} · 피쳐 {len(names)} · 부분표본 N={N_TRAIN:,}")

    rng = np.random.default_rng(SEED)
    tr_idx = np.flatnonzero(is_tr)
    sub = rng.choice(tr_idx, size=min(N_TRAIN, len(tr_idx)), replace=False)
    Xs, ys = np.nan_to_num(X[sub], nan=0.0, posinf=0.0, neginf=0.0), np.log(y[sub])
    Xe = np.nan_to_num(X[ev], nan=0.0, posinf=0.0, neginf=0.0)

    preds: dict[str, np.ndarray] = {}

    t0 = time.time()
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=0)
    m.fit(np.nan_to_num(X[is_tr], nan=0.0, posinf=0.0, neginf=0.0), np.log(y[is_tr]))
    preds["GBM 전체표본"] = np.exp(m.predict(Xe))
    log(f"GBM 전체표본 {time.time() - t0:.1f}s")

    t0 = time.time()
    m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6, random_state=0)
    m.fit(Xs, ys)
    preds[f"GBM N={N_TRAIN//1000}k"] = np.exp(m.predict(Xe))
    log(f"GBM 부분표본 {time.time() - t0:.1f}s")

    try:
        from tabpfn import TabPFNRegressor
        t0 = time.time()
        tp = TabPFNRegressor(device="cpu")
        tp.fit(Xs, ys)
        log(f"TabPFN 적합 {time.time() - t0:.1f}s · 예측 시작({len(Xe):,}행, 배치 {BATCH})")
        out = []
        for i in range(0, len(Xe), BATCH):
            out.append(tp.predict(Xe[i:i + BATCH]))
            log(f"  {min(i + BATCH, len(Xe)):,}/{len(Xe):,} · {time.time() - t0:.0f}s 경과")
        preds[f"TabPFN N={N_TRAIN//1000}k"] = np.exp(np.concatenate(out))
        log(f"TabPFN 총 {time.time() - t0:.1f}s")
    except Exception as exc:  # noqa: BLE001 -- 못 돌면 GBM 결과만 남기고 정직하게 기록
        log(f"🔴 TabPFN 실패: {type(exc).__name__}: {exc}")

    rep = {"n_train_full": int(is_tr.sum()), "n_train_sub": int(len(sub)),
           "n_oos": int(ev.sum()), "features": names, "arms": {}}
    log("=" * 100)
    log("예측 상관(OOS, log-log): " + " · ".join(
        f"{k} {np.corrcoef(np.log(np.maximum(v, 1e-12)), np.log(y[ev]))[0, 1]:.3f}"
        for k, v in preds.items()) +
        f" · 원시ATR {np.corrcoef(np.log(ap[ev]), np.log(y[ev]))[0, 1]:.3f}")
    log(f"{'팔':>18} {'ATR평탄도':>10} {'시간평탄도':>11} {'전체SD':>9} {'하위1%':>10} "
        f"{'MAE99':>9} {'50배청산율':>10}")
    arms = {"1/ATR (현행 배포)": 1.0 / ap[ev]}
    for k, v in preds.items():
        arms[f"1/{k}"] = 1.0 / np.maximum(v, 1e-12)
    for name, w in arms.items():
        cell = arm_stats(w, ret[ev], mae[ev], abucket, hbucket)
        rep["arms"][name] = cell
        log(f"{name:>18} {cell['flat_atr']:>10.2f} {cell['flat_hour']:>11.2f} {cell['sd']:>9.1f} "
            f"{cell['p01']:>+10.1f} {cell['mae_p99']:>9.1f} {cell['stopout_50x']:>9.2%}")
    log("=" * 100)
    log("⭐같은 N 의 GBM 과 TabPFN 차이 = **모델 차이**.")
    log("   GBM 전체표본과 GBM 부분표본 차이 = **표본 효과**. 둘을 섞어 읽으면 안 된다.")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log(f"저장 {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
