#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_zigzag_action_labels_fast_horizon_20260620 as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_no_max_horizon_conservative_20260620"


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def build_no_max_horizon_labels(
    frame: pd.DataFrame,
    *,
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
    atr = base._atr_pct(frame, int(atr_window))
    labels = np.zeros(n, dtype=np.int8)
    horizon = np.zeros(n, dtype=np.int32)
    reason = np.full(n, "cash", dtype=object)
    utility = np.zeros(n, dtype=np.float32)
    tp_arr = np.zeros(n, dtype=np.float32)
    sl_arr = np.zeros(n, dtype=np.float32)
    hold_arr = np.zeros(n, dtype=np.int32)
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
        for side in (1, -1):
            mfe = 0.0
            mae = 0.0
            exit_h = 0
            exit_ret: float | None = None
            exit_reason = "no_touch_to_year_end"
            for j in range(entry_i, n):
                if side > 0:
                    hi_ret = (float(high[j]) - entry) / entry
                    lo_ret = (float(low[j]) - entry) / entry
                else:
                    hi_ret = (entry - float(low[j])) / entry
                    lo_ret = (entry - float(high[j])) / entry
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
            if exit_ret is None:
                continue
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
            hold_arr[i] = int(best["hold"])
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


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--atr-window", type=int, default=48)
    p.add_argument("--tp-atr-mult", type=float, default=1.05)
    p.add_argument("--sl-atr-mult", type=float, default=0.80)
    p.add_argument("--tp-min", type=float, default=0.0045)
    p.add_argument("--tp-max", type=float, default=0.012)
    p.add_argument("--sl-min", type=float, default=0.0038)
    p.add_argument("--sl-max", type=float, default=0.009)
    p.add_argument("--min-utility", type=float, default=0.0009)
    p.add_argument("--time-penalty", type=float, default=0.000015)
    p.add_argument("--adverse-penalty", type=float, default=0.25)
    p.add_argument("--transition-buffer", type=int, default=2)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, Any] = {
        "type": "zigzag_3class_no_max_horizon_conservative_action_labels",
        "params": {
            "max_horizon": None,
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
            "max_horizon_bars": None,
            "time_penalized": True,
            "legacy_zigzag_segment_fill": False,
            "no_touch_policy": "cash",
        },
        "artifacts": {},
        "summaries": {},
    }
    for year, path in base.PRICE_FILES.items():
        frame = base._read_frame(path, expected_year=year)
        labels = build_no_max_horizon_labels(
            frame,
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
        audit["summaries"][str(year)] = base._summary(labels)
    audit_path = args.out_dir / "zigzag_action_label_audit.json"
    audit["artifacts"]["audit"] = str(audit_path)
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default) + "\n")
    print(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
