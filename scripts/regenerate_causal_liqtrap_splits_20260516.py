#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, minimum_filter1d


ROOT = Path(__file__).resolve().parents[1]


def _causal_liqtrap(df: pd.DataFrame, *, window: int = 48, eq_tol: float = 0.001, confirm: int = 3) -> np.ndarray:
    high = pd.to_numeric(df["high"], errors="coerce").ffill().fillna(0.0).to_numpy(np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").ffill().fillna(0.0).to_numpy(np.float64)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().fillna(0.0).to_numpy(np.float64)
    n = len(df)
    signal = np.zeros(n, dtype=np.float64)
    if n <= window:
        return signal

    kernel = 2 * int(confirm) + 1
    is_swing_high = high == maximum_filter1d(high, size=kernel, mode="nearest")
    is_swing_low = low == minimum_filter1d(low, size=kernel, mode="nearest")

    for i in range(int(window), n):
        start = max(0, i - int(window))
        end = i - int(confirm) + 1
        if end <= start:
            continue

        eq_high = 0.0
        swing_h_idx = np.where(is_swing_high[start:end])[0]
        if len(swing_h_idx) >= 2:
            vals = high[start:end][swing_h_idx]
            rel = np.abs(vals[:, None] - vals[None, :]) / (vals[:, None] + 1e-8)
            np.fill_diagonal(rel, 1.0)
            has_match = (rel < float(eq_tol)).any(axis=1)
            if has_match.any():
                eq_high = float(vals[np.where(has_match)[0][0]])

        eq_low = 0.0
        swing_l_idx = np.where(is_swing_low[start:end])[0]
        if len(swing_l_idx) >= 2:
            vals = low[start:end][swing_l_idx]
            rel = np.abs(vals[:, None] - vals[None, :]) / (vals[:, None] + 1e-8)
            np.fill_diagonal(rel, 1.0)
            has_match = (rel < float(eq_tol)).any(axis=1)
            if has_match.any():
                eq_low = float(vals[np.where(has_match)[0][0]])

        if eq_high > 0.0 and high[i] > eq_high and close[i] < eq_high:
            signal[i] = -np.tanh((high[i] - eq_high) / (eq_high * 0.001 + 1e-8))
        elif eq_low > 0.0 and low[i] < eq_low and close[i] > eq_low:
            signal[i] = np.tanh((eq_low - low[i]) / (eq_low * 0.001 + 1e-8))

    return signal


def _rewrite(src: Path, dst: Path) -> dict[str, object]:
    df = pd.read_csv(src)
    if "sig_liquidity_trap" in df.columns:
        old = pd.to_numeric(df["sig_liquidity_trap"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    else:
        old = np.zeros(len(df), dtype=np.float64)
    new = _causal_liqtrap(df)
    df["sig_liquidity_trap"] = new.astype(np.float32)
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    diff = np.abs(old - new)
    ts = pd.to_datetime(df["timestamp"], errors="coerce") if "timestamp" in df.columns else pd.Series(dtype="datetime64[ns]")
    return {
        "src": str(src),
        "dst": str(dst),
        "rows": int(len(df)),
        "timestamp_start": str(ts.min()) if len(ts) else None,
        "timestamp_end": str(ts.max()) if len(ts) else None,
        "changed_gt_1e-6": int((diff > 1e-6).sum()),
        "changed_rate_gt_1e-6": float((diff > 1e-6).mean()) if len(diff) else 0.0,
        "old_nonzero_rate": float((np.abs(old) > 1e-8).mean()) if len(old) else 0.0,
        "new_nonzero_rate": float((np.abs(new) > 1e-8).mean()) if len(new) else 0.0,
        "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Regenerate yearly feature splits with causal sig_liquidity_trap only.")
    p.add_argument("--out-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/splits")
    p.add_argument("--report-out", type=Path, default=ROOT / "tmp/causal_regen_20260516/causal_liqtrap_split_report.json")
    args = p.parse_args()

    sources = {
        "training_features_2024.csv": ROOT / "data/splits/year_oos/training_features_2024.csv",
        "training_features_2025.csv": ROOT / "data/splits/year_oos/training_features_2025.csv",
        "training_features_2026_rebuilt.csv": ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
    }
    report = {"outputs": []}
    for name, src in sources.items():
        report["outputs"].append(_rewrite(src, args.out_dir / name))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
