#!/usr/bin/env python3
"""TabPFN cheap-gate: ZDC(wick-앵커) 라벨의 9트리거 통합 candidate pool.

research_eth_v_rebound_multitrigger_tabpfn_cheap_gate_20260831.py(기존 giveback 라벨용,
단일시드)를 템플릿으로 재사용하되 계획서(swift-doodling-grove.md Step D) 지시대로 이 하위계열
(V자반등/ZDC) 고정 4시드 앙상블(SEEDS=[20260829,141592,271828,577215], multitrigger HOLDOUT
스크립트와 동일 관례)로 강화. 차이점:
  - 라벨: giveback outcome 문자열매칭 대신 ZDC의 hit(True/False) 컬럼 직접 사용.
  - LABEL_WINDOW=24h(=MAX_LOOKFORWARD_BARS 288봉) -- ZDC는 giveback의 고정 60분과 달리 해상
    시점이 이벤트마다 가변(수분~24시간)이라, 임베고는 안전하게 최대치로 고정(보수적, 경계행
    일부 손실 감수). 고정 60분을 그대로 썼다간 Fresh-Forward 오염 위험(계획서 Step D 지시사항).
  - 4시드 앙상블(확률 평균) + VAL 순열중요도(필수검증 항목).
  - v7b/giveback판과의 AUC 직접비교는 참고치로만 표시(문제정의 자체가 다름 -- 계획서 Step D
    중단기준 명시사항, "AUC가 낮다"만으로 실패판정 안 함).

Must run in the quant_ai conda env on the SERVER (GPU required, TABPFN_TOKEN already
authenticated there).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from tabpfn import TabPFNClassifier

ROOT = Path(__file__).resolve().parents[1]
FEATURES_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_zigzag_direction_20260901/eth_5m_v_rebound_multitrigger_zigzag_direction_features_tier0.csv"
OUT_DIR = ROOT / "tmp/eth_v_rebound_multitrigger_zigzag_direction_tabpfn_cheap_gate_20260901"

VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # untouched by this script -- cheap_gate only
LABEL_WINDOW = pd.Timedelta(hours=24)  # MAX_LOOKFORWARD_BARS=288 conservative embargo
SEEDS = [20260829, 141592, 271828, 577215]  # V자반등/ZDC 하위계열 고정 4시드 관례

FEATURE_COLUMNS = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday",
    "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]


def embargoed_split(df: pd.DataFrame) -> dict:
    ts = df["timestamp"]
    window_end = ts + LABEL_WINDOW
    return {
        "train": df.loc[(ts < VAL_START) & (window_end < VAL_START)],
        "val": df.loc[(ts >= VAL_START) & (ts <= VAL_END) & (window_end < OOS_START)],
        "oos": df.loc[(ts >= OOS_START) & (ts <= OOS_END) & (ts < HOLDOUT_START)],
    }


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_acc = float(max(y.mean(), 1.0 - y.mean()))
    accuracy = float((pred == y).mean())
    return {
        "n": int(len(y)), "accuracy": round(accuracy, 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "naive_majority_class_accuracy": round(naive_acc, 4),
        "beats_naive_accuracy": bool(accuracy > naive_acc),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(FEATURES_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["hit_bool"] = df["hit"].astype(str).map({"True": True, "False": False})
    df = df[df["hit_bool"].isin([True, False])].copy()
    df["label"] = df["hit_bool"].astype(int)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).reset_index(drop=True)

    parts = embargoed_split(df)
    for name, part in parts.items():
        print(f"{name}: n={len(part)} label_rate={part['label'].mean():.4f}")

    over_limit = len(parts["train"]) > 10000
    print(f"\ntrain n={len(parts['train'])} {'EXCEEDS' if over_limit else 'within'} TabPFN's "
          f"designed <=10000-row range -- ignore_pretraining_limits={over_limit}")

    seed_results = {"train": [], "val": [], "oos": []}
    seed_probas = {"train": [], "val": [], "oos": []}
    for seed in SEEDS:
        print(f"\n=== seed={seed}: fitting TabPFNClassifier (device=cuda) ===", flush=True)
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        for name in ("train", "val", "oos"):
            proba = clf.predict_proba(parts[name][FEATURE_COLUMNS])[:, 1]
            seed_probas[name].append(proba)
            r = evaluate(proba, parts[name]["label"].to_numpy())
            seed_results[name].append(r)
            print(f"  {name:5s} n={r['n']:5d} acc={r['accuracy']:.4f} bal_acc={r['balanced_accuracy']:.4f} "
                  f"auc={r['auc']:.4f} beats_naive={r['beats_naive_accuracy']}", flush=True)

    ensemble_results = {}
    ensemble_probas = {}
    for name in ("train", "val", "oos"):
        avg_proba = np.mean(seed_probas[name], axis=0)
        ensemble_probas[name] = avg_proba
        ensemble_results[name] = evaluate(avg_proba, parts[name]["label"].to_numpy())

    auc_by_seed = {name: [r["auc"] for r in seed_results[name]] for name in ("train", "val", "oos")}
    seed_stability = {
        name: {"mean": float(np.mean(auc_by_seed[name])), "std": float(np.std(auc_by_seed[name])),
               "min": float(np.min(auc_by_seed[name])), "max": float(np.max(auc_by_seed[name]))}
        for name in ("train", "val", "oos")
    }

    print("\n=== 4-seed ensemble (평균확률) ===")
    for name in ("train", "val", "oos"):
        r = ensemble_results[name]
        s = seed_stability[name]
        print(f"  {name:5s} n={r['n']:5d} auc={r['auc']:.4f} (per-seed mean={s['mean']:.4f} std={s['std']:.4f} "
              f"range=[{s['min']:.4f},{s['max']:.4f}]) bal_acc={r['balanced_accuracy']:.4f} beats_naive={r['beats_naive_accuracy']}")

    print("\n=== VAL 순열중요도 (앙상블 평균확률 기준, 1-pass) ===", flush=True)
    val_df = parts["val"]
    base_auc = ensemble_results["val"]["auc"]
    rng = np.random.default_rng(20260901)
    perm_importance = {}
    clfs = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed, ignore_pretraining_limits=over_limit)
        clf.fit(parts["train"][FEATURE_COLUMNS], parts["train"]["label"].to_numpy())
        clfs.append(clf)
    for col in FEATURE_COLUMNS:
        shuffled = val_df.copy()
        shuffled[col] = rng.permutation(shuffled[col].to_numpy())
        probas = [clf.predict_proba(shuffled[FEATURE_COLUMNS])[:, 1] for clf in clfs]
        avg_proba = np.mean(probas, axis=0)
        shuffled_auc = roc_auc_score(val_df["label"].to_numpy(), avg_proba)
        perm_importance[col] = round(base_auc - shuffled_auc, 5)
        print(f"  {col:22s} delta_auc={perm_importance[col]:+.5f}", flush=True)
    top_features = sorted(perm_importance.items(), key=lambda kv: -kv[1])[:10]

    comparison_note = ("giveback(30/60min V자반등) 라벨과는 문제정의 자체가 다름(ZDC=지그재그류 "
                        "단일임계치 반전확인, 해상시점 가변) -- AUC 직접비교는 참고치일 뿐 pass/fail "
                        "기준 아님(계획서 Step D 명시).")

    report = {
        "seeds": SEEDS, "device": "cuda", "ignore_pretraining_limits": over_limit,
        "feature_columns": FEATURE_COLUMNS, "label_window_hours": 24,
        "per_seed_results": seed_results, "ensemble_results": ensemble_results,
        "seed_stability": seed_stability, "permutation_importance_val": perm_importance,
        "top_10_features_by_perm_importance": top_features,
        "comparison_note": comparison_note,
        # 4-seed ensemble, from data/research/eth_v_rebound_multitrigger_holdout_20260831/holdout_report.json
        # (이전 버전은 이 값 대신 그 리포트의 comparison.v7b_sweep_only를 잘못 넣었던 라벨링 버그 -- 20260901 수정)
        "giveback_reference": {"val_auc": 0.8292, "oos_auc": 0.8127, "holdout_auc": 0.8465},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
