#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH conformal veto step 3: fix the severe adjacent-episode label correlation found in
docs/experiments/eth_candidate_conformal_veto_episode_labels_20260816.md (lag-1 autocorrelation
0.55-0.85, effective N 6-8x smaller than raw episode count) with concurrency-based uniqueness
weighting (Lopez de Prado style), applied to the already-saved episode-label parquets. Does NOT
re-simulate anything -- reads scripts/research_eth_candidate_conformal_veto_episode_labels_20260816.py's
output (entry_signal_i, hold_bars already saved per episode) and only adds weight columns.

Two diagnostics, both required before any HGB training is trusted:
1. Full lag-N (not just lag-1) autocorrelation function of the `full` label, N up to 300 bars --
   informative for documentation of how far the correlation reaches, though the uniqueness weights
   below don't depend on picking a single embargo width (concurrency weighting handles arbitrary
   overlap directly, which is why Lopez de Prado's approach is preferred over a fixed embargo cutoff
   here).
2. Per-episode uniqueness weight: for episode i spanning bars [entry_i, exit_i], average uniqueness
   u_i = mean over t in [entry_i, exit_i] of 1/c_t, where c_t = number of episodes (same window,
   same component) whose span covers bar t. Weight sums are reported as a weighted-effective-N
   cross-check against the earlier AR(1) approximation.

Output: same parquet files, rewritten in place with an added `uniqueness_weight` column, plus a
report of the lag-N ACF and weighted effective N per window/component.

fresh_forward_bar_by_bar=true (weights are computed from already-causal spans, no new data touched).
No GPU, no model training in this script.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_episode_labels_20260816"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_uniqueness_weights_20260816"

TRAIN_WINDOWS = ("2025q1", "2025q2", "2025q3")
CALIBRATION_WINDOW = "val"
MAX_LAG = 300


def log(msg: str) -> None:
    print(f"[candidate_conformal_veto_uniqueness] {msg}", flush=True)


def _lag_acf(full: np.ndarray, max_lag: int) -> list[float]:
    x = full - full.mean()
    denom = float(np.sum(x * x))
    if denom <= 0.0 or len(x) < 3:
        return []
    out = []
    for lag in range(1, min(max_lag, len(x) - 2) + 1):
        num = float(np.sum(x[:-lag] * x[lag:]))
        out.append(num / denom)
    return out


def _first_lag_below(acf: list[float], threshold: float) -> int | None:
    for lag, rho in enumerate(acf, start=1):
        if abs(rho) < threshold:
            return lag
    return None


def _uniqueness_weights(entry: np.ndarray, exit_: np.ndarray, n_bars: int) -> np.ndarray:
    """entry/exit are inclusive bar indices per episode, same window+component. n_bars = length of
    that window's frame (used to size the concurrency array; spans are clipped to it)."""
    concurrency = np.zeros(int(n_bars), dtype=np.float64)
    e0 = np.clip(entry, 0, n_bars - 1).astype(np.int64)
    e1 = np.clip(exit_, 0, n_bars - 1).astype(np.int64)
    for a, b in zip(e0, e1):
        concurrency[a:b + 1] += 1.0
    inv_c = 1.0 / np.maximum(concurrency, 1.0)
    weights = np.empty(len(e0), dtype=np.float64)
    for k, (a, b) in enumerate(zip(e0, e1)):
        weights[k] = float(np.mean(inv_c[a:b + 1]))
    return weights


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"design": "concurrency-based uniqueness weighting + lag-N ACF diagnostic on already-simulated episode labels", "windows": {}}

    for wname in (*TRAIN_WINDOWS, CALIBRATION_WINDOW):
        for name in ("h48qual", "zig075"):
            path = LABEL_DIR / f"episode_labels_{wname}_{name}.parquet"
            df = pd.read_parquet(path).sort_values("entry_signal_i").reset_index(drop=True)
            entry = df["entry_signal_i"].to_numpy(dtype=np.int64) + 1
            exit_ = entry + df["hold_bars"].to_numpy(dtype=np.int64)
            n_bars = int(exit_.max()) + 1  # exactly tight: largest bar index any span touches
            weights = _uniqueness_weights(entry, exit_, n_bars)
            df["uniqueness_weight"] = weights
            df.to_parquet(path, index=False)

            full = df["full"].to_numpy(dtype=np.float64)
            acf = _lag_acf(full, MAX_LAG)
            lag_below_02 = _first_lag_below(acf, 0.2)
            lag_below_01 = _first_lag_below(acf, 0.1)
            raw_n = int(len(df))
            weighted_n = float(weights.sum())
            key = f"{wname}_{name}"
            report["windows"][key] = {
                "raw_n": raw_n,
                "weighted_effective_n": weighted_n,
                "weight_ratio": weighted_n / raw_n if raw_n else None,
                "acf_lag1": acf[0] if acf else None,
                "acf_lag5": acf[4] if len(acf) > 4 else None,
                "acf_lag20": acf[19] if len(acf) > 19 else None,
                "acf_lag100": acf[99] if len(acf) > 99 else None,
                "first_lag_below_0.2": lag_below_02,
                "first_lag_below_0.1": lag_below_01,
                "median_hold_bars": float(df["hold_bars"].median()),
            }
            log(f"{wname} {name}: raw_n={raw_n} weighted_n={weighted_n:.1f} ratio={weighted_n/raw_n:.3f} "
                f"acf1={acf[0]:.3f} acf5={acf[4]:.3f} acf20={(acf[19] if len(acf)>19 else float('nan')):.3f} "
                f"first<0.2@lag={lag_below_02} first<0.1@lag={lag_below_01} median_hold={df['hold_bars'].median():.0f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
