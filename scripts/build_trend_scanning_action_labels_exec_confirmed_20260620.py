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
SMOOTHED_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_smoothed_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_exec_confirmed_20260620"


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


def _q(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    return float(s[min(len(s) - 1, max(0, int(round((len(s) - 1) * p))))])


def _summ(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "median": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0}
    return {
        "count": len(vals),
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "p75": _q(vals, 0.75),
        "p90": _q(vals, 0.90),
        "p95": _q(vals, 0.95),
        "p99": _q(vals, 0.99),
        "max": float(max(vals)),
    }


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


def _summary(labels: np.ndarray) -> dict[str, Any]:
    segs = _segments(labels)
    active = [s for s in segs if int(s["action"]) != 0]
    counts = Counter(int(v) for v in labels)
    return {
        "rows": int(len(labels)),
        "counts": {str(k): int(v) for k, v in sorted(counts.items())},
        "ratios": {str(k): float(v) / max(len(labels), 1) for k, v in sorted(counts.items())},
        "segments": int(len(segs)),
        "active_segments": int(len(active)),
        "active_segment_length": _summ([float(s["length"]) for s in active]),
    }


def _confirm_execution(frame: pd.DataFrame, *, max_horizon: int, tp_price_move: float, sl_price_move: float, min_mfe_sl_ratio: float) -> pd.DataFrame:
    labels = pd.to_numeric(frame["wave3_action"], errors="raise").to_numpy(dtype=np.int8)
    open_ = pd.to_numeric(frame["open"], errors="raise").to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    out = np.zeros(len(frame), dtype=np.int8)
    reason = np.full(len(frame), "cash", dtype=object)
    hold = np.zeros(len(frame), dtype=np.int16)
    mfe_arr = np.zeros(len(frame), dtype=np.float32)
    mae_arr = np.zeros(len(frame), dtype=np.float32)
    util = np.zeros(len(frame), dtype=np.float32)
    for i in range(len(frame) - 2):
        action = int(labels[i])
        if action == 0:
            continue
        side = 1 if action == 1 else -1
        entry_i = i + 1
        entry = float(open_[entry_i])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        end_i = min(len(frame) - 1, entry_i + int(max_horizon))
        mfe = 0.0
        mae = 0.0
        exit_ret = 0.0
        exit_reason = "max_horizon"
        exit_hold = int(max_horizon)
        for j in range(entry_i, end_i + 1):
            if side > 0:
                hi_ret = (float(high[j]) - entry) / entry
                lo_ret = (float(low[j]) - entry) / entry
                close_ret = (float(close[j]) - entry) / entry
            else:
                hi_ret = (entry - float(low[j])) / entry
                lo_ret = (entry - float(high[j])) / entry
                close_ret = (entry - float(close[j])) / entry
            mfe = max(mfe, hi_ret)
            mae = min(mae, lo_ret)
            hit_tp = hi_ret >= float(tp_price_move)
            hit_sl = lo_ret <= -abs(float(sl_price_move))
            exit_ret = close_ret
            exit_hold = j - entry_i
            if hit_tp or hit_sl:
                if hit_sl and hit_tp:
                    exit_reason = "stop_loss_both_touch"
                    exit_ret = -abs(float(sl_price_move))
                elif hit_tp:
                    exit_reason = "take_profit"
                    exit_ret = float(tp_price_move)
                else:
                    exit_reason = "stop_loss"
                    exit_ret = -abs(float(sl_price_move))
                break
        confirmed = exit_reason == "take_profit" or (mfe >= float(min_mfe_sl_ratio) * abs(float(sl_price_move)) and exit_ret > 0.0)
        if confirmed:
            out[i] = action
            reason[i] = exit_reason
            hold[i] = np.int16(max(0, min(exit_hold, np.iinfo(np.int16).max)))
            mfe_arr[i] = np.float32(mfe)
            mae_arr[i] = np.float32(mae)
            util[i] = np.float32(exit_ret)
    res = frame.copy()
    res["wave3_action_smoothed"] = labels
    res["wave3_action"] = out
    res["zigzag_action"] = out
    res["wave3_action_name"] = pd.Series(out).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    res["exec_confirm_reason"] = reason
    res["exec_confirm_hold_bars"] = hold
    res["exec_confirm_mfe"] = mfe_arr
    res["exec_confirm_mae"] = mae_arr
    res["exec_confirm_utility_price_move"] = util
    return res


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--smoothed-dir", type=Path, default=SMOOTHED_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--max-horizon", type=int, default=384)
    p.add_argument("--tp-price-move", type=float, default=0.0578)
    p.add_argument("--sl-price-move", type=float, default=0.0311)
    p.add_argument("--min-mfe-sl-ratio", type=float, default=1.20)
    args = p.parse_args()
    src_dir = args.smoothed_dir if args.smoothed_dir.is_absolute() else ROOT / args.smoothed_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "trend_scanning_smoothed_execution_confirmed",
        "source_smoothed_dir": str(src_dir),
        "params": {
            "max_horizon": int(args.max_horizon),
            "tp_price_move": float(args.tp_price_move),
            "sl_price_move": float(args.sl_price_move),
            "min_mfe_sl_ratio": float(args.min_mfe_sl_ratio),
        },
        "artifacts": {},
        "summaries": {},
    }
    for year in (2024, 2025, 2026):
        path = src_dir / f"wave3_action_labels_{year}.csv"
        frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
        labels = _confirm_execution(
            frame,
            max_horizon=int(args.max_horizon),
            tp_price_move=float(args.tp_price_move),
            sl_price_move=float(args.sl_price_move),
            min_mfe_sl_ratio=float(args.min_mfe_sl_ratio),
        )
        out_path = out_dir / f"wave3_action_labels_{year}.csv"
        labels.to_csv(out_path, index=False)
        labels.to_csv(out_dir / f"zigzag_action_labels_{year}.csv", index=False)
        audit["artifacts"][str(year)] = str(out_path)
        audit["summaries"][str(year)] = {
            "smoothed": _summary(pd.to_numeric(labels["wave3_action_smoothed"], errors="raise").to_numpy(dtype=np.int8)),
            "exec_confirmed": _summary(pd.to_numeric(labels["wave3_action"], errors="raise").to_numpy(dtype=np.int8)),
            "confirm_reasons": labels.loc[labels["wave3_action"] != 0, "exec_confirm_reason"].astype(str).value_counts().to_dict(),
        }
    audit_path = out_dir / "wave3_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
