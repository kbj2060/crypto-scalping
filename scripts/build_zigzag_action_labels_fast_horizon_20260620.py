#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_fast_horizon_20260620"
PRICE_FILES = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}


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
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    years = sorted(frame["timestamp"].dt.year.dropna().astype(int).unique().tolist())
    if years != [int(expected_year)]:
        raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame


def _atr_pct(frame: pd.DataFrame, window: int) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    prev = np.roll(close, 1)
    prev[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev), np.abs(low - prev)))
    atr = pd.Series(tr).ewm(span=int(window), adjust=False, min_periods=1).mean().to_numpy(dtype=np.float64)
    return atr / np.maximum(close, 1.0e-12)


def _summ(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    s = sorted(vals)

    def q(p: float) -> float:
        return float(s[min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))])

    return {"count": len(vals), "mean": float(mean(vals)), "median": float(median(vals)), "p75": q(0.75), "p90": q(0.90), "p95": q(0.95), "p99": q(0.99), "max": float(max(vals))}


def _segments(labels: np.ndarray) -> list[dict[str, int]]:
    if len(labels) == 0:
        return []
    out: list[dict[str, int]] = []
    cur = int(labels[0])
    start = 0
    for i in range(1, len(labels)):
        val = int(labels[i])
        if val != cur:
            out.append({"action": cur, "start": start, "end": i - 1, "length": i - start})
            cur = val
            start = i
    out.append({"action": cur, "start": start, "end": len(labels) - 1, "length": len(labels) - start})
    return out


def build_fast_horizon_labels(
    frame: pd.DataFrame,
    *,
    max_horizon: int,
    atr_window: int,
    tp_atr_mult: float,
    sl_atr_mult: float,
    tp_min: float,
    tp_max: float,
    sl_min: float,
    sl_max: float,
    min_utility: float,
    time_penalty: float,
    adverse_penalty: float,
    transition_buffer: int,
) -> pd.DataFrame:
    n = len(frame)
    open_ = pd.to_numeric(frame["open"], errors="coerce").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").to_numpy(dtype=np.float64)
    atr = _atr_pct(frame, int(atr_window))
    labels = np.zeros(n, dtype=np.int8)
    horizon = np.zeros(n, dtype=np.int16)
    reason = np.full(n, "cash", dtype=object)
    utility = np.zeros(n, dtype=np.float32)
    tp_arr = np.zeros(n, dtype=np.float32)
    sl_arr = np.zeros(n, dtype=np.float32)
    hold_arr = np.zeros(n, dtype=np.int16)
    mfe_arr = np.zeros(n, dtype=np.float32)
    mae_arr = np.zeros(n, dtype=np.float32)
    soft = np.zeros((n, 3), dtype=np.float32)
    soft[:, 0] = 1.0

    for i in range(0, n - 2):
        entry_i = i + 1
        entry = float(open_[entry_i])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        tp = float(np.clip(float(atr[i]) * float(tp_atr_mult), float(tp_min), float(tp_max)))
        sl = float(np.clip(float(atr[i]) * float(sl_atr_mult), float(sl_min), float(sl_max)))
        if sl >= tp:
            continue
        best = {"side": 0, "utility": 0.0, "hold": 0, "mfe": 0.0, "mae": 0.0, "reason": "cash"}
        end_i = min(n - 1, entry_i + int(max_horizon))
        for side in (1, -1):
            mfe = 0.0
            mae = 0.0
            exit_h = int(max_horizon)
            exit_ret = 0.0
            exit_reason = "max_horizon"
            for j in range(entry_i, end_i + 1):
                if side > 0:
                    hi_ret = (float(high[j]) - entry) / entry
                    lo_ret = (float(low[j]) - entry) / entry
                    close_ret = (float(close[j]) - entry) / entry
                else:
                    hi_ret = (entry - float(low[j])) / entry
                    lo_ret = (entry - float(high[j])) / entry
                    close_ret = (entry - float(close[j])) / entry
                h = j - entry_i
                mfe = max(mfe, hi_ret)
                mae = min(mae, lo_ret)
                hit_tp = hi_ret >= tp
                hit_sl = lo_ret <= -abs(sl)
                if hit_tp or hit_sl:
                    exit_h = h
                    if hit_sl and hit_tp:
                        exit_ret = -abs(sl)
                        exit_reason = "stop_loss_both_touch"
                    elif hit_tp:
                        exit_ret = tp
                        exit_reason = "take_profit"
                    else:
                        exit_ret = -abs(sl)
                        exit_reason = "stop_loss"
                    break
                exit_ret = close_ret
            util = float(exit_ret) - float(time_penalty) * float(exit_h) - float(adverse_penalty) * max(0.0, abs(float(mae)) - abs(sl))
            if util > float(best["utility"]):
                best = {"side": side, "utility": util, "hold": exit_h, "mfe": mfe, "mae": mae, "reason": exit_reason}
        if float(best["utility"]) >= float(min_utility):
            action = 1 if int(best["side"]) > 0 else 2
            labels[i] = action
            horizon[i] = int(best["hold"])
            reason[i] = str(best["reason"])
            utility[i] = np.float32(best["utility"])
            tp_arr[i] = np.float32(tp)
            sl_arr[i] = np.float32(sl)
            hold_arr[i] = np.int16(max(0, min(int(best["hold"]), np.iinfo(np.int16).max)))
            mfe_arr[i] = np.float32(best["mfe"])
            mae_arr[i] = np.float32(best["mae"])
            logits = np.zeros(3, dtype=np.float64)
            logits[action] = float(best["utility"]) / max(float(atr[i]), 1.0e-6)
            logits[0] = max(0.0, -float(best["utility"])) / max(float(atr[i]), 1.0e-6)
            logits -= float(np.max(logits))
            probs = np.exp(logits)
            probs /= max(float(probs.sum()), 1.0e-12)
            soft[i] = probs.astype(np.float32)

    buf = int(max(0, transition_buffer))
    if buf > 0:
        change = np.flatnonzero(labels != np.roll(labels, 1))
        change = change[change > 0]
        for idx in change:
            lo_i = max(0, int(idx) - buf)
            hi_i = min(n, int(idx) + buf + 1)
            labels[lo_i:hi_i] = 0
            horizon[lo_i:hi_i] = 0
            reason[lo_i:hi_i] = "transition_buffer"
            utility[lo_i:hi_i] = 0.0
            tp_arr[lo_i:hi_i] = 0.0
            sl_arr[lo_i:hi_i] = 0.0
            hold_arr[lo_i:hi_i] = 0
            mfe_arr[lo_i:hi_i] = 0.0
            mae_arr[lo_i:hi_i] = 0.0
            soft[lo_i:hi_i] = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    out = frame[["timestamp", "open", "high", "low", "close"]].copy()
    out["zigzag_action"] = labels
    out["zigzag_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["zigzag_fast_horizon_bars"] = horizon
    out["zigzag_fast_exit_reason"] = reason
    out["zigzag_fast_utility"] = utility
    out["zigzag_fast_tp_price_move"] = tp_arr
    out["zigzag_fast_sl_price_move"] = sl_arr
    out["zigzag_fast_hold_bars"] = hold_arr
    out["zigzag_fast_mfe"] = mfe_arr
    out["zigzag_fast_mae"] = mae_arr
    out["zigzag_soft_cash"] = soft[:, 0]
    out["zigzag_soft_long"] = soft[:, 1]
    out["zigzag_soft_short"] = soft[:, 2]
    return out


def _summary(labels: pd.DataFrame) -> dict[str, Any]:
    y = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int8)
    segs = _segments(y)
    active = [s for s in segs if int(s["action"]) != 0]
    by_action = {}
    for action, name in [(0, "cash"), (1, "long"), (2, "short")]:
        by_action[name] = _summ([float(s["length"]) for s in segs if int(s["action"]) == action])
    counts = Counter(int(v) for v in y)
    active_rows = labels[labels["zigzag_action"] != 0]
    return {
        "rows": int(len(labels)),
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "ratios": {str(k): float(v) / max(len(labels), 1) for k, v in sorted(counts.items())},
        "segments": int(len(segs)),
        "active_segments": int(len(active)),
        "segment_length_bars": {
            "all": _summ([float(s["length"]) for s in segs]),
            "active": _summ([float(s["length"]) for s in active]),
            **by_action,
        },
        "active_hold_bars": _summ(pd.to_numeric(active_rows["zigzag_fast_hold_bars"], errors="coerce").dropna().astype(float).tolist()),
        "exit_reasons": active_rows["zigzag_fast_exit_reason"].astype(str).value_counts().to_dict(),
        "active_utility": _summ(pd.to_numeric(active_rows["zigzag_fast_utility"], errors="coerce").dropna().astype(float).tolist()),
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--max-horizon", type=int, default=48)
    p.add_argument("--atr-window", type=int, default=48)
    p.add_argument("--tp-atr-mult", type=float, default=1.15)
    p.add_argument("--sl-atr-mult", type=float, default=0.85)
    p.add_argument("--tp-min", type=float, default=0.005)
    p.add_argument("--tp-max", type=float, default=0.014)
    p.add_argument("--sl-min", type=float, default=0.004)
    p.add_argument("--sl-max", type=float, default=0.010)
    p.add_argument("--min-utility", type=float, default=0.0010)
    p.add_argument("--time-penalty", type=float, default=0.000015)
    p.add_argument("--adverse-penalty", type=float, default=0.25)
    p.add_argument("--transition-buffer", type=int, default=2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "zigzag_3class_fast_horizon_action_labels",
        "params": {
            "max_horizon": int(args.max_horizon),
            "atr_window": int(args.atr_window),
            "tp_atr_mult": float(args.tp_atr_mult),
            "sl_atr_mult": float(args.sl_atr_mult),
            "tp_bounds": [float(args.tp_min), float(args.tp_max)],
            "sl_bounds": [float(args.sl_min), float(args.sl_max)],
            "min_utility": float(args.min_utility),
            "time_penalty": float(args.time_penalty),
            "adverse_penalty": float(args.adverse_penalty),
            "transition_buffer": int(args.transition_buffer),
        },
        "contract": {
            "label_mapping": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "label_column": "zigzag_action",
            "soft_label_columns": ["zigzag_soft_cash", "zigzag_soft_long", "zigzag_soft_short"],
            "diagnostic_columns": [
                "zigzag_fast_horizon_bars",
                "zigzag_fast_exit_reason",
                "zigzag_fast_utility",
                "zigzag_fast_tp_price_move",
                "zigzag_fast_sl_price_move",
                "zigzag_fast_hold_bars",
                "zigzag_fast_mfe",
                "zigzag_fast_mae",
            ],
            "uses_future_only_for_offline_labeling": True,
            "max_horizon_bars": int(args.max_horizon),
            "time_penalized": True,
            "legacy_zigzag_segment_fill": False,
        },
        "artifacts": {},
        "summaries": {},
    }
    for year, path in PRICE_FILES.items():
        frame = _read_frame(path, expected_year=year)
        labels = build_fast_horizon_labels(
            frame,
            max_horizon=int(args.max_horizon),
            atr_window=int(args.atr_window),
            tp_atr_mult=float(args.tp_atr_mult),
            sl_atr_mult=float(args.sl_atr_mult),
            tp_min=float(args.tp_min),
            tp_max=float(args.tp_max),
            sl_min=float(args.sl_min),
            sl_max=float(args.sl_max),
            min_utility=float(args.min_utility),
            time_penalty=float(args.time_penalty),
            adverse_penalty=float(args.adverse_penalty),
            transition_buffer=int(args.transition_buffer),
        )
        out = args.out_dir / f"zigzag_action_labels_{year}.csv"
        labels.to_csv(out, index=False)
        audit["artifacts"][str(year)] = str(out)
        audit["summaries"][str(year)] = _summary(labels)
    audit_path = args.out_dir / "zigzag_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
