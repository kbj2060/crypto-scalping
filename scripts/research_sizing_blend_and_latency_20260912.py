#!/usr/bin/env python3
"""TabPFN 실배포 가능성 — **1행 지연**과 **GBM 혼합** (2026-09-12).

사용자: *"tabpfn 결과 나오면 같이 반영해줘"*

## 왜 다시 재나
직전에 «TabPFN 예측 1,095초» 를 근거로 배포를 반대했는데, 그건 **18,024행 배치**다.
라이브 워커는 5분에 **1행**을 부른다. 배치 비용으로 단건 비용을 단정한 건 잘못된 추론이다.
TabPFN 은 매 호출에서 학습 문맥(여기선 5,000행)을 다시 처리하므로 단건에도 고정비가 있다 --
그 고정비가 실제로 얼마인지는 **재봐야** 안다.

같이 보는 것: **GBM + TabPFN 로그 평균 혼합**. 둘의 오차가 다르면 혼합이 각각보다 낫다.

⚠️GPU 를 안 쓴다(여유 1.25/8 GiB, 대시보드 경합 전례). CPU 단건 지연이 워커 주기(300초)에
비해 충분히 작으면 **GPU 없이 배포 가능**이다.

실행: python scripts/research_sizing_blend_and_latency_20260912.py
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
IN = ROOT / "tmp" / "sizing_feature_expansion_20260912" / "matrix.npz"
OUT = ROOT / "tmp" / "sizing_blend_20260912"
LEV = 50.0
N_SUB = 5000
SEEDS = (990143, 220759, 380136, 923411, 331386, 602160, 700199, 982044)
SMALL = ["atr288", "rv12", "rv48", "rv288", "hour_sin", "hour_cos", "dow", "qv_z", "nt_z", "volexp"]
LAT_TRIALS = 12


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def metrics(w, ret, mae) -> dict:
    w = w / w.mean()
    pnl = w * ret
    return {"sd": float(pnl.std(ddof=1)), "p01": float(np.percentile(pnl, 1)),
            "mae_p99": float(np.percentile(w * mae, 99)),
            "stopout_50x": float(np.mean(w * mae > 1e4 / LEV))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    z = np.load(IN, allow_pickle=True)
    X, y, ret, mae, ap, is_tr = z["X"], z["y"], z["ret"], z["mae"], z["ap"], z["is_tr"]
    names = [str(s) for s in z["names"]]
    ev = ~is_tr
    cols = [names.index(k) for k in SMALL if k in names]
    Xf = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    log(f"학습 {is_tr.sum():,} · 평가 {ev.sum():,} · 피쳐 {len(cols)}(소형 집합)")

    # ── GBM 시드 앙상블 (배포 후보) ───────────────────────────────────────────────
    t0 = time.time()
    gbms = []
    for s in SEEDS:
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06, max_depth=6,
                                          random_state=int(s))
        m.fit(Xf[is_tr][:, cols], np.log(y[is_tr]))
        gbms.append(m)
    p_gbm = np.exp(np.mean([m.predict(Xf[ev][:, cols]) for m in gbms], axis=0))
    log(f"GBM 시드 {len(SEEDS)}개 적합+예측 {time.time() - t0:.1f}s")

    rng = np.random.default_rng(20260912)
    sub = rng.choice(np.flatnonzero(is_tr), size=N_SUB, replace=False)

    preds = {"GBM(시드8)": p_gbm}
    lat = {}
    try:
        from tabpfn import TabPFNRegressor
        tp = TabPFNRegressor(device="cpu")
        t0 = time.time()
        tp.fit(Xf[sub][:, cols], np.log(y[sub]))
        log(f"TabPFN 적합(N={N_SUB:,}, 피쳐 {len(cols)}) {time.time() - t0:.1f}s")

        # ⭐라이브 단건 지연 -- 워커는 5분에 1행을 부른다
        one = Xf[ev][:1, cols]
        tp.predict(one)                                   # 워밍업(첫 호출은 초기화가 섞인다)
        ts = []
        for i in range(LAT_TRIALS):
            t0 = time.time(); tp.predict(Xf[ev][i:i + 1, cols]); ts.append(time.time() - t0)
        lat = {"n": LAT_TRIALS, "median_s": float(np.median(ts)),
               "p90_s": float(np.percentile(ts, 90)), "max_s": float(max(ts))}
        log(f"⭐TabPFN **1행** 지연 — 중앙 {lat['median_s']:.2f}s · p90 {lat['p90_s']:.2f}s · "
            f"최대 {lat['max_s']:.2f}s  (워커 주기 300s)")

        t0 = time.time()
        out = [tp.predict(Xf[ev][i:i + 2000, cols]) for i in range(0, ev.sum(), 2000)]
        preds["TabPFN(5k)"] = np.exp(np.concatenate(out))
        log(f"TabPFN 전체 평가 {time.time() - t0:.0f}s")
    except Exception as exc:  # noqa: BLE001
        log(f"🔴 TabPFN 실패: {type(exc).__name__}: {exc}")

    if "TabPFN(5k)" in preds:
        # 로그 공간 혼합 -- 예측 대상이 양수·우편향이라 기하평균이 자연스럽다
        for wgt in (0.25, 0.5, 0.75):
            preds[f"혼합 G{1 - wgt:.2f}/T{wgt:.2f}"] = np.exp(
                (1 - wgt) * np.log(preds["GBM(시드8)"]) + wgt * np.log(preds["TabPFN(5k)"]))

    base = metrics(1.0 / ap[ev], ret[ev], mae[ev])
    log("=" * 96)
    log("예측 상관(OOS, log-log): " + " · ".join(
        f"{k} {np.corrcoef(np.log(np.maximum(v, 1e-12)), np.log(y[ev]))[0, 1]:.3f}"
        for k, v in preds.items()))
    log(f"{'팔':>20} {'SD':>9} {'SD 변화':>10} {'하위1%':>10} {'MAE99':>9} "
        f"{'청산율':>9} {'청산율 변화':>11}")
    log(f"{'1/ATR (현행)':>20} {base['sd']:>9.1f} {'—':>10} {base['p01']:>+10.1f} "
        f"{base['mae_p99']:>9.1f} {base['stopout_50x']:>9.2%} {'—':>11}")
    rep = {"base": base, "latency_1row": lat, "arms": {}}
    for k, v in preds.items():
        cell = metrics(1.0 / np.maximum(v, 1e-12), ret[ev], mae[ev])
        rep["arms"][k] = cell
        log(f"{'1/' + k:>20} {cell['sd']:>9.1f} {cell['sd'] / base['sd'] - 1:>+10.1%} "
            f"{cell['p01']:>+10.1f} {cell['mae_p99']:>9.1f} {cell['stopout_50x']:>9.2%} "
            f"{cell['stopout_50x'] - base['stopout_50x']:>+11.2%}p")
    (OUT / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float))
    log("=" * 96)
    log(f"저장 {OUT / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
