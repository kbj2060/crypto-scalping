#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "tmp/causal_regen_20260516/omega4_research_labels_20260624"
PRICE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
SPLIT_TS = pd.Timestamp("2025-10-01")
ACTION_NAME = {0: "CASH", 1: "LONG", 2: "SHORT"}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    raise TypeError(f"not json serializable: {type(obj)!r}")


def _read_price_frame(path: Path) -> pd.DataFrame:
    required = ["timestamp", "open", "high", "low", "close"]
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=required, parse_dates=["timestamp"], low_memory=False)
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} missing columns: {missing}")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    for col in ("open", "high", "low", "close"):
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    return frame


def _rolling_slope_t_value(log_close: np.ndarray, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(len(log_close))
    t_stat = np.full(n, np.nan, dtype=np.float64)
    fwd_ret = np.full(n, np.nan, dtype=np.float64)
    h = int(horizon)
    valid = n - h
    if h < 3 or valid <= 0:
        return t_stat, fwd_ret

    future = log_close[1:]
    windows = np.lib.stride_tricks.sliding_window_view(future, h)
    windows = windows[:valid]
    x = np.arange(h, dtype=np.float64)
    x_centered = x - float(x.mean())
    sxx = float(np.dot(x_centered, x_centered))

    y_mean = windows.mean(axis=1)
    beta = ((windows - y_mean[:, None]) * x_centered[None, :]).sum(axis=1) / sxx
    alpha = y_mean - beta * float(x.mean())
    fitted = alpha[:, None] + beta[:, None] * x[None, :]
    resid = windows - fitted
    sse = (resid * resid).sum(axis=1)
    se_beta = np.sqrt(np.maximum(sse / float(h - 2), 0.0) / sxx)
    with np.errstate(divide="ignore", invalid="ignore"):
        vals = beta / se_beta
    vals[~np.isfinite(vals)] = np.nan
    t_stat[:valid] = vals
    fwd_ret[:valid] = log_close[h:] - log_close[:valid]
    return t_stat, fwd_ret


def _trend_scanning_labels(
    price: pd.DataFrame,
    *,
    horizons: list[int],
    min_abs_t: float,
    min_edge: float,
    confirm_bars: int,
    min_hold_bars: int,
    cash_confirm_bars: int,
) -> pd.DataFrame:
    log_close = np.log(pd.to_numeric(price["close"], errors="raise").to_numpy(dtype=np.float64))
    best_abs_t = np.full(len(price), -np.inf, dtype=np.float64)
    best_t = np.full(len(price), np.nan, dtype=np.float64)
    best_h = np.zeros(len(price), dtype=np.int64)
    best_ret = np.full(len(price), np.nan, dtype=np.float64)

    for horizon in horizons:
        t_stat, fwd_ret = _rolling_slope_t_value(log_close, int(horizon))
        ok = np.isfinite(t_stat) & (np.abs(t_stat) > best_abs_t)
        best_abs_t[ok] = np.abs(t_stat[ok])
        best_t[ok] = t_stat[ok]
        best_h[ok] = int(horizon)
        best_ret[ok] = fwd_ret[ok]

    raw_action = np.zeros(len(price), dtype=np.int64)
    long = (best_t >= float(min_abs_t)) & (best_ret >= float(min_edge))
    short = (best_t <= -float(min_abs_t)) & (best_ret <= -float(min_edge))
    raw_action[long] = 1
    raw_action[short] = 2
    action = _smooth_actions(
        raw_action,
        confirm_bars=int(confirm_bars),
        min_hold_bars=int(min_hold_bars),
        cash_confirm_bars=int(cash_confirm_bars),
    )

    out = price[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = action
    out["zigzag_action_name"] = [ACTION_NAME[int(x)] for x in action]
    out["trend_scan_raw_action"] = raw_action
    out["research_label_family"] = "trend_scanning"
    out["trend_scan_t_value"] = np.nan_to_num(best_t, nan=0.0, posinf=0.0, neginf=0.0)
    out["trend_scan_horizon"] = best_h
    out["trend_scan_forward_log_return"] = np.nan_to_num(best_ret, nan=0.0, posinf=0.0, neginf=0.0)
    out["trend_scan_abs_t_threshold"] = float(min_abs_t)
    out["trend_scan_min_edge"] = float(min_edge)
    out["trend_scan_confirm_bars"] = int(confirm_bars)
    out["trend_scan_min_hold_bars"] = int(min_hold_bars)
    out["trend_scan_cash_confirm_bars"] = int(cash_confirm_bars)
    return out


def _smooth_actions(
    raw_action: np.ndarray,
    *,
    confirm_bars: int,
    min_hold_bars: int,
    cash_confirm_bars: int,
) -> np.ndarray:
    confirm = max(int(confirm_bars), 0)
    min_hold = max(int(min_hold_bars), 0)
    cash_confirm = max(int(cash_confirm_bars), 1)
    raw = np.asarray(raw_action, dtype=np.int64)
    if confirm <= 1 and min_hold <= 0 and cash_confirm <= 1:
        return raw.copy()

    out = np.zeros(len(raw), dtype=np.int64)
    state = 0
    hold = 10**9
    pending = -1
    pending_count = 0
    for i, action in enumerate(raw):
        action = int(action)
        if action == state:
            pending = -1
            pending_count = 0
        else:
            if action == pending:
                pending_count += 1
            else:
                pending = action
                pending_count = 1
            required = cash_confirm if action == 0 else max(confirm, 1)
            can_switch = action == 0 or state == 0 or hold >= min_hold
            if can_switch and pending_count >= required:
                state = action
                hold = 0
                pending = -1
                pending_count = 0
        out[i] = state
        hold += 1
    return out


def _vol_scaled_forward_return_labels(
    price: pd.DataFrame,
    *,
    horizon: int,
    vol_window: int,
    threshold: float,
    min_periods: int,
) -> pd.DataFrame:
    close = pd.to_numeric(price["close"], errors="raise").to_numpy(dtype=np.float64)
    log_close = np.log(close)
    h = int(horizon)
    fwd = np.full(len(price), np.nan, dtype=np.float64)
    if len(price) > h:
        fwd[:-h] = log_close[h:] - log_close[:-h]
    log_ret = pd.Series(log_close).diff()
    realized_vol = log_ret.rolling(int(vol_window), min_periods=int(min_periods)).std().to_numpy(dtype=np.float64)
    scaled_vol = realized_vol * math.sqrt(float(h))
    with np.errstate(divide="ignore", invalid="ignore"):
        score = fwd / scaled_vol
    score[~np.isfinite(score)] = np.nan

    action = np.zeros(len(price), dtype=np.int64)
    action[score >= float(threshold)] = 1
    action[score <= -float(threshold)] = 2

    out = price[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = action
    out["zigzag_action_name"] = [ACTION_NAME[int(x)] for x in action]
    out["research_label_family"] = "vol_scaled_forward_return"
    out["vol_scaled_horizon"] = int(h)
    out["vol_scaled_vol_window"] = int(vol_window)
    out["vol_scaled_forward_log_return"] = np.nan_to_num(fwd, nan=0.0, posinf=0.0, neginf=0.0)
    out["vol_scaled_realized_vol"] = np.nan_to_num(scaled_vol, nan=0.0, posinf=0.0, neginf=0.0)
    out["vol_scaled_score"] = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    out["vol_scaled_threshold"] = float(threshold)
    return out


def _summary(label_frame: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(label_frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    counts = pd.Series(y).value_counts().sort_index()
    out = {
        "rows": int(len(label_frame)),
        "counts": {str(int(k)): int(v) for k, v in counts.items()},
        "active_ratio": float((y != 0).mean()) if len(y) else 0.0,
        "long_ratio": float((y == 1).mean()) if len(y) else 0.0,
        "short_ratio": float((y == 2).mean()) if len(y) else 0.0,
    }
    numeric_cols = [
        c
        for c in label_frame.columns
        if c.endswith("_score") or c.endswith("_t_value") or c.endswith("_forward_log_return")
    ]
    for col in numeric_cols:
        vals = pd.to_numeric(label_frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(vals):
            out[col] = {
                "mean": float(vals.mean()),
                "p10": float(vals.quantile(0.10)),
                "p50": float(vals.quantile(0.50)),
                "p90": float(vals.quantile(0.90)),
            }
    return out


def _write_label_set(
    *,
    name: str,
    labels_by_year: dict[int, pd.DataFrame],
    config: dict[str, Any],
) -> Path:
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {"name": name, "config": config, "years": {}}
    for year, labels in labels_by_year.items():
        labels.to_csv(out_dir / f"zigzag_action_labels_{int(year)}.csv", index=False)
        audit["years"][str(year)] = _summary(labels)
        if int(year) == 2025:
            audit["validation_split"] = {
                "train_before_2025_10_01": _summary(labels[labels["timestamp"] < SPLIT_TS].reset_index(drop=True)),
                "validation_from_2025_10_01": _summary(labels[labels["timestamp"] >= SPLIT_TS].reset_index(drop=True)),
            }
    with (out_dir / "label_audit.json").open("w", encoding="utf-8") as fh:
        json.dump(audit, fh, indent=2, default=_json_default)
    return out_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trend-horizons", default="12,24,48,96,192")
    ap.add_argument("--trend-min-abs-t", type=float, default=1.50)
    ap.add_argument("--trend-min-edge", type=float, default=0.0005)
    ap.add_argument("--trend-confirm-bars", type=int, default=0)
    ap.add_argument("--trend-min-hold-bars", type=int, default=0)
    ap.add_argument("--trend-cash-confirm-bars", type=int, default=1)
    ap.add_argument("--vol-horizon", type=int, default=48)
    ap.add_argument("--vol-window", type=int, default=192)
    ap.add_argument("--vol-threshold", type=float, default=0.50)
    ap.add_argument("--vol-min-periods", type=int, default=48)
    args = ap.parse_args()

    trend_horizons = [int(x.strip()) for x in str(args.trend_horizons).split(",") if x.strip()]
    if not trend_horizons:
        raise RuntimeError("at least one trend horizon is required")

    prices = {year: _read_price_frame(path) for year, path in PRICE_FILES.items()}

    trend_labels = {
        year: _trend_scanning_labels(
            frame,
            horizons=trend_horizons,
            min_abs_t=float(args.trend_min_abs_t),
            min_edge=float(args.trend_min_edge),
            confirm_bars=int(args.trend_confirm_bars),
            min_hold_bars=int(args.trend_min_hold_bars),
            cash_confirm_bars=int(args.trend_cash_confirm_bars),
        )
        for year, frame in prices.items()
    }
    vol_labels = {
        year: _vol_scaled_forward_return_labels(
            frame,
            horizon=int(args.vol_horizon),
            vol_window=int(args.vol_window),
            threshold=float(args.vol_threshold),
            min_periods=int(args.vol_min_periods),
        )
        for year, frame in prices.items()
    }

    trend_name = (
        "trend_scanning_t"
        + "_".join(str(x) for x in trend_horizons)
        + f"_abs{str(args.trend_min_abs_t).replace('.', 'p')}"
        + f"_edge{str(args.trend_min_edge).replace('.', 'p')}"
        + f"_confirm{int(args.trend_confirm_bars)}"
        + f"_hold{int(args.trend_min_hold_bars)}"
    )
    vol_name = (
        f"vol_scaled_fwd{int(args.vol_horizon)}"
        + f"_thr{str(args.vol_threshold).replace('.', 'p')}"
        + f"_vol{int(args.vol_window)}"
    )
    trend_dir = _write_label_set(
        name=trend_name,
        labels_by_year=trend_labels,
        config={
            "label_family": "trend_scanning",
            "horizons": trend_horizons,
            "min_abs_t": float(args.trend_min_abs_t),
            "min_edge": float(args.trend_min_edge),
            "confirm_bars": int(args.trend_confirm_bars),
            "min_hold_bars": int(args.trend_min_hold_bars),
            "cash_confirm_bars": int(args.trend_cash_confirm_bars),
            "window_definition": "future close bars t+1 through t+h, selected by max abs t-value",
        },
    )
    vol_dir = _write_label_set(
        name=vol_name,
        labels_by_year=vol_labels,
        config={
            "label_family": "vol_scaled_forward_return",
            "horizon": int(args.vol_horizon),
            "vol_window": int(args.vol_window),
            "threshold": float(args.vol_threshold),
            "min_periods": int(args.vol_min_periods),
            "score": "forward_log_return / (rolling_log_return_std * sqrt(horizon))",
        },
    )
    print(json.dumps({"trend_dir": str(trend_dir), "vol_dir": str(vol_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
