#!/usr/bin/env python3
"""변동성클러스터링 신호(체크A/B 둘 다 통과)의 마지막 관문 -- 이 세션 전체가 최종 판정에
써온 기준(실제 비용 반영 벤치마크 백테스트)을 그대로 적용. TRAIN에서만 방향규칙(부호+임계값)을
정하고 VAL/OOS엔 그대로 고정 적용(look-ahead 없음), max(always_long,always_short) 대비 증분."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/eth_distributional_regression_return_labels_20260819"
ROUNDTRIP_COST_BP = 10.0
VOL_FEATURES = ["volatility_z", "atr_pct_rank_288", "realized_vol_ratio", "garch_vol_z", "compression_score"]
HORIZONS_BAR = {"h48": 48, "h96": 96}
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}

import sys
sys.path.insert(0, str(ROOT / "scripts"))
import eth_directional_change_tabm_training_canonicaldata_20260819 as canon  # noqa: E402
omega = canon.omega


def _load() -> pd.DataFrame:
    train, eval_df = omega._load_omega_frames()[:2]
    feat_frames = []
    for feat in (train, eval_df):
        f = feat[["timestamp", *VOL_FEATURES]].copy()
        f["timestamp"] = pd.to_datetime(f["timestamp"])
        feat_frames.append(f)
    f2024 = pd.read_csv(ROOT / "data/splits/year_oos/training_features_2024.csv", usecols=["timestamp", *VOL_FEATURES])
    f2024["timestamp"] = pd.to_datetime(f2024["timestamp"])
    feat_frames.insert(0, f2024)
    feat_all = pd.concat(feat_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")

    perp_frames = []
    for f in ["data/splits/year_oos/training_features_2024.csv",
              "data/splits/year_oos/training_features_2025.csv",
              "data/splits/year_oos/training_features_2026_rebuilt.csv"]:
        p = pd.read_csv(ROOT / f, usecols=["timestamp", "close"], parse_dates=["timestamp"])
        perp_frames.append(p)
    perp = pd.concat(perp_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    perp["log_return"] = np.log(perp["close"] / perp["close"].shift(1))

    df = feat_all.merge(perp[["timestamp", "log_return"]], on="timestamp", how="inner")
    return df.sort_values("timestamp").set_index("timestamp")


def _periodic_return_bp(sub: pd.DataFrame, vf: str, h: int, threshold: float, sign_rule: int) -> tuple[float, int]:
    n = len(sub)
    entry_idx = np.arange(0, n - h, h)
    sig = pd.to_numeric(sub[vf], errors="coerce").to_numpy()
    logret = sub["log_return"].to_numpy()
    rets_bp = []
    for i in entry_idx:
        s = sig[i]
        if np.isnan(s):
            continue
        direction = sign_rule * np.sign(s - threshold)
        if direction == 0:
            continue
        fwd = np.nansum(logret[i + 1:i + 1 + h])
        rets_bp.append(direction * fwd * 10000.0 - ROUNDTRIP_COST_BP)
    return (float(np.mean(rets_bp)), len(rets_bp)) if rets_bp else (float("nan"), 0)


def _benchmark_bp(sub: pd.DataFrame, h: int, always_long: bool) -> float:
    n = len(sub)
    entry_idx = np.arange(0, n - h, h)
    logret = sub["log_return"].to_numpy()
    d = 1.0 if always_long else -1.0
    rets_bp = [d * np.nansum(logret[i + 1:i + 1 + h]) * 10000.0 - ROUNDTRIP_COST_BP for i in entry_idx]
    return float(np.mean(rets_bp)) if rets_bp else float("nan")


def main() -> None:
    df = _load()
    train_sub = df.loc[SPLITS["TRAIN"][0]:SPLITS["TRAIN"][1]]

    results = {}
    print("방향규칙: TRAIN median을 임계값+TRAIN IC부호로 고정, VAL/OOS 불변 적용", flush=True)
    print(f"{'signal':20s} {'h':>4s} | {'TRAIN incr':>11s} {'VAL incr':>10s} {'OOS incr':>10s} | 3split전부양수?", flush=True)
    n_all_positive = 0
    for vf in VOL_FEATURES:
        train_vals = pd.to_numeric(train_sub[vf], errors="coerce")
        threshold = float(train_vals.median())
        for h_name, h in HORIZONS_BAR.items():
            # TRAIN IC 부호로 방향 결정: (signal-threshold)와 fwd_return의 TRAIN 내 상관 부호
            lbl = pd.read_csv(LABEL_DIR / "fwd_return_labels_2024.csv", usecols=["timestamp", f"fwd_logret_{h_name}"], parse_dates=["timestamp"])
            lbl2 = pd.read_csv(LABEL_DIR / "fwd_return_labels_2025.csv", usecols=["timestamp", f"fwd_logret_{h_name}"], parse_dates=["timestamp"])
            lbl_all = pd.concat([lbl, lbl2], ignore_index=True).set_index("timestamp")
            joined = train_sub[[vf]].join(lbl_all, how="inner")
            from scipy.stats import spearmanr
            train_ic = spearmanr(joined[vf], joined[f"fwd_logret_{h_name}"], nan_policy="omit").statistic
            sign_rule = 1 if train_ic > 0 else -1

            cells = {}
            for split_name, (start, end) in SPLITS.items():
                sub = df.loc[start:end]
                bp, n_trades = _periodic_return_bp(sub, vf, h, threshold, sign_rule)
                bench = max(_benchmark_bp(sub, h, True), _benchmark_bp(sub, h, False))
                cells[split_name] = {"strategy_bp": bp, "n_trades": n_trades, "benchmark_bp": bench, "increment_bp": bp - bench}
            results[(vf, h_name)] = {"train_ic": float(train_ic), "sign_rule": sign_rule, "threshold": threshold, "cells": cells}
            incs = {s: cells[s]["increment_bp"] for s in SPLITS}
            all_pos = all(v > 0 for v in incs.values())
            n_all_positive += int(all_pos)
            print(f"{vf:20s} {h_name:>4s} | {incs['TRAIN']:>+11.1f} {incs['VAL']:>+10.1f} {incs['OOS']:>+10.1f} | "
                  f"{'YES' if all_pos else 'no'}", flush=True)

    print(f"\n3-split 전부 양수(벤치마크 초과) 조합: {n_all_positive}/{len(results)}", flush=True)

    out = {f"{k[0]}_{k[1]}": v for k, v in results.items()}
    out_path = ROOT / "tmp/eth_distributional_regression_volatility_backtest_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
