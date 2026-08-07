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
RAW_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_20260531"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/trend_scanning_action_labels_smoothed_20260620"


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


def _state_machine(
    t_values: np.ndarray,
    betas: np.ndarray,
    *,
    enter_t: float,
    exit_t: float,
    flip_t: float,
) -> np.ndarray:
    labels = np.zeros(len(t_values), dtype=np.int8)
    state = 0
    for i, (t_val, beta) in enumerate(zip(t_values, betas)):
        strength = abs(float(t_val))
        side = 1 if float(beta) > 0.0 else 2 if float(beta) < 0.0 else 0
        if state == 0:
            if side != 0 and strength >= float(enter_t):
                state = side
        elif side == state and strength >= float(exit_t):
            pass
        elif side != 0 and side != state and strength >= float(flip_t):
            state = side
        elif strength < float(exit_t):
            state = 0
        labels[i] = state
    return labels


def _remove_short_active(labels: np.ndarray, min_active_len: int) -> np.ndarray:
    out = labels.copy()
    for seg in _segments(out):
        if int(seg["action"]) != 0 and int(seg["length"]) < int(min_active_len):
            out[int(seg["start"]) : int(seg["end"]) + 1] = 0
    return out


def _fill_same_side_cash_gaps(labels: np.ndarray, max_gap: int) -> np.ndarray:
    out = labels.copy()
    segs = _segments(out)
    for idx in range(1, len(segs) - 1):
        prev_seg = segs[idx - 1]
        seg = segs[idx]
        next_seg = segs[idx + 1]
        if int(seg["action"]) == 0 and int(seg["length"]) <= int(max_gap) and int(prev_seg["action"]) == int(next_seg["action"]) and int(prev_seg["action"]) != 0:
            out[int(seg["start"]) : int(seg["end"]) + 1] = int(prev_seg["action"])
    return out


def _apply_transition_buffer(labels: np.ndarray, bars: int) -> np.ndarray:
    out = labels.copy()
    if int(bars) <= 0:
        return out
    changes = np.flatnonzero(out != np.roll(out, 1))
    changes = changes[changes > 0]
    for idx in changes:
        lo = max(0, int(idx) - int(bars))
        hi = min(len(out), int(idx) + int(bars) + 1)
        out[lo:hi] = 0
    return out


def _summary(labels: np.ndarray) -> dict[str, Any]:
    segs = _segments(labels)
    active = [s for s in segs if int(s["action"]) != 0]
    by_action = {}
    for action, name in [(0, "cash"), (1, "long"), (2, "short")]:
        by_action[name] = _summ([float(s["length"]) for s in segs if int(s["action"]) == action])
    counts = Counter(int(v) for v in labels)
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
    }


def _build_one(raw_path: Path, *, args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(raw_path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close", "ts_t_value", "ts_beta"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"{raw_path} missing required columns: {missing}")
    t_values = pd.to_numeric(raw["ts_t_value"], errors="raise").to_numpy(dtype=np.float64)
    betas = pd.to_numeric(raw["ts_beta"], errors="raise").to_numpy(dtype=np.float64)
    labels = _state_machine(t_values, betas, enter_t=args.enter_t, exit_t=args.exit_t, flip_t=args.flip_t)
    labels = _fill_same_side_cash_gaps(labels, int(args.max_same_side_cash_gap))
    labels = _remove_short_active(labels, int(args.min_active_len))
    labels = _apply_transition_buffer(labels, int(args.transition_buffer))
    labels = _remove_short_active(labels, int(args.min_active_len))
    out = raw.copy()
    out["wave3_action_raw"] = pd.to_numeric(raw.get("wave3_action", 0), errors="coerce").fillna(0).astype(np.int8)
    out["wave3_action"] = labels
    out["wave3_action_name"] = pd.Series(labels).map({0: "CASH", 1: "LONG", 2: "SHORT"}).to_numpy()
    out["zigzag_action"] = labels
    return out, {"raw": _summary(out["wave3_action_raw"].to_numpy(dtype=np.int8)), "smoothed": _summary(labels)}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--enter-t", type=float, default=9.0)
    p.add_argument("--exit-t", type=float, default=5.0)
    p.add_argument("--flip-t", type=float, default=10.0)
    p.add_argument("--min-active-len", type=int, default=12)
    p.add_argument("--max-same-side-cash-gap", type=int, default=6)
    p.add_argument("--transition-buffer", type=int, default=1)
    args = p.parse_args()

    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else ROOT / args.raw_dir
    out_dir = args.out_dir if args.out_dir.is_absolute() else ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "smoothed_trend_scanning_3class_action_labels",
        "source_raw_dir": str(raw_dir),
        "params": {
            "enter_t": float(args.enter_t),
            "exit_t": float(args.exit_t),
            "flip_t": float(args.flip_t),
            "min_active_len": int(args.min_active_len),
            "max_same_side_cash_gap": int(args.max_same_side_cash_gap),
            "transition_buffer": int(args.transition_buffer),
        },
        "artifacts": {},
        "summaries": {},
    }
    for year in (2024, 2025, 2026):
        raw_path = raw_dir / f"wave3_action_labels_{year}.csv"
        labels, summary = _build_one(raw_path, args=args)
        out_path = out_dir / f"wave3_action_labels_{year}.csv"
        labels.to_csv(out_path, index=False)
        zig_path = out_dir / f"zigzag_action_labels_{year}.csv"
        labels.to_csv(zig_path, index=False)
        audit["artifacts"][str(year)] = str(out_path)
        audit["summaries"][str(year)] = summary
    audit_path = out_dir / "wave3_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
