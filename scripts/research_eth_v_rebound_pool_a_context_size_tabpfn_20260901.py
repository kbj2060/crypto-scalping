#!/usr/bin/env python3
"""풀 A(현행 배포 구성) 재학습 -- TabPFN 컨텍스트 크기 vs 성능/레이턴시 실측.

## 왜

2026-09-01 후보풀 3종 비교에서 **풀 A(기존 9트리거)를 유지하는 게 답**으로 결론났다. 그런데
풀 A는 현재 배포 구성 그 자체라 그대로 재학습하면 지금 모델이 재현될 뿐이다 -- A를 유지하면서
실제로 개선 여지가 있는 지점을 찾다가, **배포판이 레이턴시 때문에 TRAIN 17,961건 중 6,000건만
frozen context로 쓰고 있다**는 것이 유력한 후보로 드러났다.

GBM 프록시 실측(로컬): VAL AUC가 6,000건 0.7987 -> 전체 17,969건 0.8093으로 **+0.0106**
(시드 std 0.0039의 2.7배 = 노이즈 아님). 9,000건에서 이미 0.8058로 대부분 회수된다.

**그러나 배포판은 TabPFN이고 GBM과 스케일 특성이 다르다**(in-context learning, 권장 1만행 상한,
현재 파이프라인은 `ignore_pretraining_limits=True`로 우회 중). 그래서 서버 GPU에서 직접 잰다.
동시에 **레이턴시**도 재야 한다 -- 6,000건을 고른 이유가 바로 그것이었고(서버 실측 fit+predict
2.88초), 성능 이득이 레이턴시 비용을 정당화하는지가 실제 판단 기준이다.

## 측정

배포된 학습 데이터를 그대로 쓴다(재빌드 없음):
`data/labels/eth_5m_v_rebound_multitrigger_20260831/eth_5m_v_rebound_multitrigger_features_tier0.csv`
-- 여기서 TRAIN(< 2025-09-01) / VAL(2025-09-01 ~ 2025-12-31)만. ⚠️OOS/HOLDOUT 미터치.

컨텍스트 크기 {6000(현행), 9000, 12000, 전체} × 4시드(이 프로젝트 공용):
  - VAL AUC (분류 성능)
  - **라이브 레이턴시**: fit + predict(1행) -- 라이브가 매 사이클 실제로 하는 일과 동일
  - 컨텍스트의 자연 라벨비율 보존 여부(배포판은 재균형 안 함 -- 검증 수치가 그 비율 기준이므로)

Run on the server (GPU) via handoff:
  handoff.sh launch server <job> -- python scripts/research_eth_v_rebound_pool_a_context_size_tabpfn_20260901.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

DATA_DIR = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_20260831"
FEATURES_CSV = DATA_DIR / "eth_5m_v_rebound_multitrigger_features_tier0.csv"
OUT_JSON = ROOT / "data/research/eth_v_rebound_pool_a_context_size_20260901/report.json"

TRAIN_END = pd.Timestamp("2025-09-01", tz="UTC")
VAL_END = pd.Timestamp("2026-01-01", tz="UTC")
SEEDS = [20260829, 141592, 271828, 577215]
CONTEXT_SIZES = [6000, 9000, 12000, None]  # None = 전체 TRAIN

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864", "range_width_pct",
    "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z", "p_fast", "p_slow", "ret3_z",
    "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio", "upper_wick_ratio",
    "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def log(msg: str) -> None:
    print(f"[ctx_size] {msg}", flush=True)


def main() -> int:
    from sklearn.metrics import roc_auc_score
    from tabpfn import TabPFNClassifier
    import torch

    log(f"cuda available: {torch.cuda.is_available()}")
    df = pd.read_csv(FEATURES_CSV, parse_dates=["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df = df.loc[df["outcome"].isin(["V자반등", "지지/횡보"])].copy()
    df["label"] = (df["outcome"] == "V자반등").astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    tr = df.loc[df["timestamp"] < TRAIN_END]
    va = df.loc[(df["timestamp"] >= TRAIN_END) & (df["timestamp"] < VAL_END)]
    log(f"TRAIN n={len(tr)} (base={tr['label'].mean():.4f})  VAL n={len(va)} (base={va['label'].mean():.4f})")

    Xva, yva = va[FEATURE_COLUMNS].to_numpy(dtype=float), va["label"].to_numpy()
    one_row = Xva[:1]

    results = {}
    for size in CONTEXT_SIZES:
        key = "full" if size is None else str(size)
        n_use = len(tr) if size is None else min(size, len(tr))
        aucs, fit_s, pred_s, ratios = [], [], [], []
        for sd in SEEDS:
            rng = np.random.default_rng(sd)
            idx = np.sort(rng.choice(len(tr), size=n_use, replace=False)) if n_use < len(tr) else np.arange(len(tr))
            sub = tr.iloc[idx]
            Xtr, ytr = sub[FEATURE_COLUMNS].to_numpy(dtype=float), sub["label"].to_numpy()
            ratios.append(float(ytr.mean()))

            clf = TabPFNClassifier(device="cuda", random_state=sd, ignore_pretraining_limits=True)
            t0 = time.time(); clf.fit(Xtr, ytr); t_fit = time.time() - t0
            # 라이브가 매 사이클 하는 일: fit된 상태에서 1행 예측
            t0 = time.time(); clf.predict_proba(one_row); t_pred = time.time() - t0
            proba = clf.predict_proba(Xva)[:, 1]
            aucs.append(roc_auc_score(yva, proba))
            fit_s.append(t_fit); pred_s.append(t_pred)

        a = np.array(aucs)
        results[key] = {
            "n_context": int(n_use),
            "auc_mean": round(float(a.mean()), 4), "auc_std": round(float(a.std()), 4),
            "label_ratio_mean": round(float(np.mean(ratios)), 4),
            "fit_sec_mean": round(float(np.mean(fit_s)), 2),
            "predict1_sec_mean": round(float(np.mean(pred_s)), 3),
            "live_cycle_sec_mean": round(float(np.mean(fit_s) + np.mean(pred_s)), 2),
        }
        r = results[key]
        log(f"  ctx={key:>6s} (n={n_use:5d})  VAL AUC {r['auc_mean']:.4f}±{r['auc_std']:.4f}  "
            f"| fit {r['fit_sec_mean']:.2f}s + predict1 {r['predict1_sec_mean']:.3f}s "
            f"= 라이브 사이클 {r['live_cycle_sec_mean']:.2f}s  | 라벨비율 {r['label_ratio_mean']:.4f}")

    base = results["6000"]["auc_mean"]
    log("\n=== 현행(6000) 대비 ===")
    for k, r in results.items():
        r["delta_vs_6000"] = round(r["auc_mean"] - base, 4)
        r["latency_x_vs_6000"] = round(r["live_cycle_sec_mean"] / results["6000"]["live_cycle_sec_mean"], 2)
        log(f"  ctx={k:>6s}: AUC {r['delta_vs_6000']:+.4f}  레이턴시 {r['latency_x_vs_6000']}x "
            f"({r['live_cycle_sec_mean']:.2f}s)")

    report = {
        "signal": "v_rebound_pool_a_context_size", "asset": "ETHUSDT",
        "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "scope": {"model": "TabPFN (배포판과 동일)", "pool": "A = 현행 배포 9트리거 구성",
                  "holdout_touched": False, "oos_touched": False, "live_code_changed": False,
                  "splits": {"TRAIN": f"< {TRAIN_END}", "VAL": f"{TRAIN_END} .. {VAL_END}"},
                  "purpose": ("풀 A 유지 결정 후, 배포판이 레이턴시 때문에 쓰는 6,000행 컨텍스트가 "
                              "성능을 깎는지 + 늘리면 레이턴시를 얼마나 지불하는지 실측")},
        "deployed_context_size": 6000, "seeds": SEEDS, "results": results,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"report saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
