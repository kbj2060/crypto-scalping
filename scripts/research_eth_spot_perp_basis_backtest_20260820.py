#!/usr/bin/env python3
"""spot-perp basis IC스캔 후속 벤치마크 백테스트. 방향규칙은 TRAIN IC 부호로만 결정(VAL/OOS는
방향선택에 전혀 안 씀 -- 이 로드맵의 ETF플로우/스테이블코인 cheap gate와 동일 원칙, look-ahead
없는 사전등록). h-bar 겹침 없는 주기적 재진입(매 h bar마다 새 포지션, 이전 포지션 종료 후
진입 -- 5분봉에서 48bar짜리 포지션을 매bar마다 새로 열면 겹침회계가 복잡해지므로 단순화).
비용은 이 저장소가 반복 인용하는 왕복 10bp 고정 가정."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ROUNDTRIP_COST_BP = 10.0

Z_WINDOW = 48
ROC_WINDOW = 12
HORIZONS = [1, 3, 12, 48]
SPLITS = {
    "TRAIN": ("2024-01-01", "2025-08-31"),
    "VAL": ("2025-09-01", "2025-12-31"),
    "OOS": ("2026-01-01", "2026-03-31"),
}


def _load_basis_frame() -> pd.DataFrame:
    spot = pd.read_csv(ROOT / "binance_data/klines_spot/ETHUSDT/ETHUSDT-5m-spot.csv",
                        usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "spot_close"})
    perp_frames = []
    for f in ["data/splits/year_oos/training_features_2024.csv",
              "data/splits/year_oos/training_features_2025.csv",
              "data/splits/year_oos/training_features_2026_rebuilt.csv"]:
        p = pd.read_csv(ROOT / f, usecols=["timestamp", "close"], parse_dates=["timestamp"]).rename(columns={"close": "perp_close"})
        perp_frames.append(p)
    perp = pd.concat(perp_frames, ignore_index=True).drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    df = perp.merge(spot, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    df["basis_raw"] = (df["perp_close"] - df["spot_close"]) / df["spot_close"]
    roll = df["basis_raw"].rolling(Z_WINDOW)
    df["basis_z48"] = (df["basis_raw"] - roll.mean()) / roll.std()
    df["basis_roc12"] = df["basis_raw"] - df["basis_raw"].shift(ROC_WINDOW)
    df["log_return"] = np.log(df["perp_close"] / df["perp_close"].shift(1))
    return df.set_index("timestamp")


def _periodic_strategy_return_bp(sub: pd.DataFrame, sig_col: str, h: int, sign_rule: int) -> tuple[float, int]:
    """sign_rule: TRAIN에서 정한 고정 방향(+1=신호와 같은방향 베팅, -1=역방향/contrarian).
    매 h bar마다 겹침없이 재진입, 비용은 매 트레이드 왕복 10bp 고정 차감. 반환: 평균bp/트레이드, 트레이드수."""
    n = len(sub)
    entry_idx = np.arange(0, n - h, h)
    if len(entry_idx) == 0:
        return float("nan"), 0
    sig = sub[sig_col].to_numpy()
    logret = sub["log_return"].to_numpy()
    rets_bp = []
    for i in entry_idx:
        s = sig[i]
        if np.isnan(s):
            continue
        direction = sign_rule * np.sign(s)
        if direction == 0:
            continue
        fwd = np.nansum(logret[i + 1:i + 1 + h])
        rets_bp.append(direction * fwd * 10000.0 - ROUNDTRIP_COST_BP)
    if not rets_bp:
        return float("nan"), 0
    return float(np.mean(rets_bp)), len(rets_bp)


def _benchmark_return_bp(sub: pd.DataFrame, h: int, always_long: bool) -> float:
    n = len(sub)
    entry_idx = np.arange(0, n - h, h)
    logret = sub["log_return"].to_numpy()
    rets_bp = []
    for i in entry_idx:
        fwd = np.nansum(logret[i + 1:i + 1 + h])
        d = 1.0 if always_long else -1.0
        rets_bp.append(d * fwd * 10000.0 - ROUNDTRIP_COST_BP)
    return float(np.mean(rets_bp)) if rets_bp else float("nan")


def main() -> None:
    df = _load_basis_frame()
    ic_scan = json.loads((ROOT / "tmp/eth_spot_perp_basis_ic_scan_20260820.json").read_text())["ic_scan"]

    results = {}
    for sig_col in ["basis_raw", "basis_z48", "basis_roc12"]:
        for h in HORIZONS:
            train_ic = ic_scan[f"TRAIN|{sig_col}|h{h}"]["ic"]
            sign_rule = -1 if train_ic < 0 else 1  # TRAIN IC 부호로만 결정(contrarian if TRAIN IC<0)
            cell = {}
            for split_name, (start, end) in SPLITS.items():
                sub = df.loc[start:end]
                bp, n_trades = _periodic_strategy_return_bp(sub, sig_col, h, sign_rule)
                bench_long = _benchmark_return_bp(sub, h, always_long=True)
                bench_short = _benchmark_return_bp(sub, h, always_long=False)
                bench = max(bench_long, bench_short)
                cell[split_name] = {"strategy_bp": bp, "n_trades": n_trades,
                                     "benchmark_bp": bench, "increment_bp": bp - bench}
            results[(sig_col, h)] = {"train_ic_sign": "neg(contrarian)" if sign_rule == -1 else "pos(momentum)", "cells": cell}

    print(f"벤치마크: max(always_long,always_short) 대비 증분(bp/트레이드), 방향규칙은 TRAIN IC부호로만 고정", flush=True)
    print(f"{'signal':14s} {'h':>4s} {'rule':16s} | {'TRAIN incr':>11s} {'VAL incr':>10s} {'OOS incr':>10s} | 3split전부양수?", flush=True)
    n_all_positive = 0
    for (sig_col, h), r in results.items():
        incs = {s: r["cells"][s]["increment_bp"] for s in SPLITS}
        all_pos = all(v > 0 for v in incs.values())
        n_all_positive += int(all_pos)
        print(f"{sig_col:14s} {h:>4d} {r['train_ic_sign']:16s} | "
              f"{incs['TRAIN']:>+11.1f} {incs['VAL']:>+10.1f} {incs['OOS']:>+10.1f} | {'YES' if all_pos else 'no'}", flush=True)

    print(f"\n3-split 전부 양수(벤치마크 초과)인 (신호,호라이즌) 조합: {n_all_positive}/{len(results)}", flush=True)

    out = {f"{k[0]}_h{k[1]}": v for k, v in results.items()}
    out_path = ROOT / "tmp/eth_spot_perp_basis_backtest_20260820.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"\n[report] {out_path}")


if __name__ == "__main__":
    main()
