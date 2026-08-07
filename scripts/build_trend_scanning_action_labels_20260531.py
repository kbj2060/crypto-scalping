#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_20260531"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_frame(path: Path, *, expected_year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(expected_year)]:
        raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame


@njit(parallel=True)
def _trend_scan_fast(values: np.ndarray, windows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CAUSALITY FIX 2026-08-04: the window for output index t must end AT t (use only
    values[t-L+1 .. t]), not start at t (the original used values[t .. t+L-1], which reads up to
    L-1 bars INTO THE FUTURE relative to t -- confirmed via direct empirical recomputation against
    stored ts_t_value on both BTC and ETH saved outputs, see project memory /
    scripts/build_btc_1h_trendscan_causal_fix_20260804.py for the verification methodology)."""
    n = len(values)
    out_t = np.zeros(n, dtype=np.float64)
    out_l = np.full(n, -1, dtype=np.int32)
    out_beta = np.zeros(n, dtype=np.float64)
    for t in prange(n):
        best_t = 0.0
        best_l = -1
        best_beta = 0.0
        for wi in range(len(windows)):
            L = int(windows[wi])
            if L <= 2 or t - L + 1 < 0:
                continue
            start = t - L + 1
            mean_x = (L - 1) / 2.0
            var_x_sum = L * (L * L - 1.0) / 12.0
            mean_y = 0.0
            ok = True
            for k in range(L):
                v = values[start + k]
                if not np.isfinite(v):
                    ok = False
                    break
                mean_y += v
            if not ok:
                continue
            mean_y /= L
            cov_xy = 0.0
            for k in range(L):
                cov_xy += (k - mean_x) * (values[start + k] - mean_y)
            beta = cov_xy / var_x_sum
            alpha = mean_y - beta * mean_x
            rss = 0.0
            for k in range(L):
                residual = values[start + k] - (alpha + beta * k)
                rss += residual * residual
            if rss <= 1e-12:
                t_val = 0.0
            else:
                se_beta = np.sqrt(rss / (L - 2.0)) / np.sqrt(var_x_sum)
                if se_beta <= 1e-12:
                    t_val = 0.0
                else:
                    t_val = beta / se_beta
            if abs(t_val) > abs(best_t):
                best_t = t_val
                best_l = L
                best_beta = beta
        out_t[t] = best_t
        out_l[t] = best_l
        out_beta[t] = best_beta
    return out_t, out_l, out_beta


def build_trend_scanning_labels(
    frame: pd.DataFrame,
    *,
    windows: list[int],
    threshold: float,
    price_col: str,
    use_log_price: bool,
) -> pd.DataFrame:
    prices = pd.to_numeric(frame[price_col], errors="coerce").to_numpy(dtype=np.float64)
    values = np.log(np.maximum(prices, 1e-12)) if use_log_price else prices
    win = np.array(sorted(set(int(w) for w in windows if int(w) > 2)), dtype=np.int32)
    if len(win) == 0:
        raise ValueError("at least one window > 2 is required")
    t_values, opt_l, betas = _trend_scan_fast(values, win)
    labels = np.zeros(len(frame), dtype=np.int8)
    labels[(np.abs(t_values) >= float(threshold)) & (betas > 0.0)] = 1
    labels[(np.abs(t_values) >= float(threshold)) & (betas < 0.0)] = 2
    forward_ret = np.zeros(len(frame), dtype=np.float32)
    valid = opt_l > 0
    idx = np.flatnonzero(valid)
    end = idx + opt_l[idx] - 1
    ok = end < len(prices)
    idx = idx[ok]
    end = end[ok]
    forward_ret[idx] = ((prices[end] / np.maximum(prices[idx], 1e-12)) - 1.0).astype(np.float32)

    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    out["wave3_action"] = labels
    out["wave3_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["ts_t_value"] = t_values.astype(np.float32)
    out["ts_opt_L"] = opt_l.astype(np.int16)
    out["ts_beta"] = betas.astype(np.float32)
    out["ts_forward_return"] = forward_ret
    return out


def _summary(labels: pd.DataFrame) -> dict[str, Any]:
    counts = labels["wave3_action"].value_counts().sort_index().to_dict()
    total = max(len(labels), 1)
    active = labels[labels["wave3_action"] != 0]
    return {
        "rows": int(len(labels)),
        "counts": {str(k): int(v) for k, v in counts.items()},
        "ratios": {str(k): float(v) / total for k, v in counts.items()},
        "active_rows": int(len(active)),
        "abs_t_value_quantiles": {
            str(q): float(np.nanquantile(np.abs(labels["ts_t_value"].to_numpy(dtype=np.float64)), q))
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "opt_L_counts": {str(int(k)): int(v) for k, v in labels["ts_opt_L"].value_counts().sort_index().to_dict().items()},
    }


def _parse_windows(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--input-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--input-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--input-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--windows", default="6,12,24,36,48,72")
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--price-col", default="close")
    p.add_argument("--raw-price", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    windows = _parse_windows(args.windows)
    audit: dict[str, Any] = {
        "type": "trend_scanning_3class_action_labels",
        "params": {
            "windows": windows,
            "threshold": float(args.threshold),
            "price_col": str(args.price_col),
            "use_log_price": not bool(args.raw_price),
        },
        "artifacts": {},
        "summaries": {},
        "contract": {
            "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "linear_regression_t_value_scan": True,
            "target_like_columns": ["ts_t_value", "ts_opt_L", "ts_beta", "ts_forward_return"],
            "uses_future_only_for_offline_labeling": True,
            "not_active_until_promoted": True,
        },
    }
    for year, path in [(2024, args.input_2024), (2025, args.input_2025), (2026, args.input_2026)]:
        frame = _read_frame(path, expected_year=year)
        labels = build_trend_scanning_labels(
            frame,
            windows=windows,
            threshold=float(args.threshold),
            price_col=str(args.price_col),
            use_log_price=not bool(args.raw_price),
        )
        out = args.out_dir / f"wave3_action_labels_{year}.csv"
        labels.to_csv(out, index=False)
        audit["artifacts"][str(year)] = str(out)
        audit["summaries"][str(year)] = _summary(labels)
    audit_path = args.out_dir / "wave3_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
