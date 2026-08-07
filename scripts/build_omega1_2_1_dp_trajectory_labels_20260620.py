#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_dp_trajectory_daytrade_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"

MAX_AGE = 96
LEVERAGE = 2.0
MARGIN_FRACTION_FOR_LABEL = 0.025
NOTIONAL = LEVERAGE * MARGIN_FRACTION_FOR_LABEL
FEE_PER_SIDE = 0.0001 * 3.0
HOLD_PENALTY = 0.000002
MIN_ENTRY_EDGE = 0.00008
TP_BOUNDS = (0.006, 0.050)
SL_BOUNDS = (0.004, 0.035)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _read_frame(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _clip(x: float, lo: float, hi: float) -> float:
    return min(max(float(x), float(lo)), float(hi))


def _atr_pct(df: pd.DataFrame, lookback: int = 48) -> np.ndarray:
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return (tr.rolling(lookback, min_periods=1).mean() / close).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _simulate_entry_path(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    policy_long: np.ndarray,
    policy_short: np.ndarray,
    *,
    start_i: int,
    side: int,
) -> dict[str, Any]:
    entry = float(close[start_i])
    if entry <= 0.0:
        return {"hold": 0, "raw": 0.0, "mfe": 0.0, "mae": 0.0, "exit_i": start_i, "reason": "bad_entry"}
    i = start_i
    age = 1
    mfe = 0.0
    mae = 0.0
    exit_i = min(start_i + 1, len(close) - 1)
    reason = "forced_end"
    while i < len(close) - 1 and age <= MAX_AGE:
        j = i + 1
        if side > 0:
            hi_raw = (float(high[j]) - entry) / entry
            lo_raw = (float(low[j]) - entry) / entry
        else:
            hi_raw = (entry - float(low[j])) / entry
            lo_raw = (entry - float(high[j])) / entry
        mfe = max(mfe, hi_raw)
        mae = min(mae, lo_raw)
        action = int(policy_long[i, age] if side > 0 else policy_short[i, age])
        exit_i = j
        if action == 1:
            reason = "dp_exit"
            break
        i = j
        age += 1
    else:
        reason = "max_age"
    exit_px = float(close[exit_i])
    raw = (exit_px - entry) / entry if side > 0 else (entry - exit_px) / entry
    return {"hold": int(exit_i - start_i), "raw": raw, "mfe": mfe, "mae": mae, "exit_i": int(exit_i), "reason": reason}


def _build_dp_labels(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    n = len(df)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    atr = _atr_pct(df)
    next_ret = np.zeros(n, dtype=np.float64)
    next_ret[:-1] = close[1:] / np.maximum(close[:-1], 1e-12) - 1.0

    v_flat = np.zeros(n + 1, dtype=np.float64)
    v_long = np.zeros((n + 1, MAX_AGE + 2), dtype=np.float64)
    v_short = np.zeros((n + 1, MAX_AGE + 2), dtype=np.float64)
    p_flat = np.zeros(n, dtype=np.int8)  # 0 CASH, 1 ENTER_LONG, 2 ENTER_SHORT
    p_long = np.zeros((n, MAX_AGE + 1), dtype=np.int8)  # 0 HOLD, 1 EXIT
    p_short = np.zeros((n, MAX_AGE + 1), dtype=np.int8)

    entry_cost = FEE_PER_SIDE * NOTIONAL
    exit_cost = FEE_PER_SIDE * NOTIONAL
    for i in range(n - 2, -1, -1):
        ret = float(next_ret[i]) * NOTIONAL
        cash_v = v_flat[i + 1]
        enter_long = -entry_cost + ret - HOLD_PENALTY + v_long[i + 1, 1]
        enter_short = -entry_cost - ret - HOLD_PENALTY + v_short[i + 1, 1]
        vals = (cash_v, enter_long, enter_short)
        best = int(np.argmax(vals))
        if best != 0 and vals[best] - cash_v < MIN_ENTRY_EDGE:
            best = 0
        p_flat[i] = best
        v_flat[i] = vals[best]
        for age in range(MAX_AGE, 0, -1):
            exit_v = -exit_cost + v_flat[i + 1]
            if age >= MAX_AGE:
                v_long[i, age] = exit_v
                v_short[i, age] = exit_v
                p_long[i, age] = 1
                p_short[i, age] = 1
                continue
            hold_long = ret - HOLD_PENALTY + v_long[i + 1, age + 1]
            hold_short = -ret - HOLD_PENALTY + v_short[i + 1, age + 1]
            if exit_v >= hold_long:
                v_long[i, age] = exit_v
                p_long[i, age] = 1
            else:
                v_long[i, age] = hold_long
            if exit_v >= hold_short:
                v_short[i, age] = exit_v
                p_short[i, age] = 1
            else:
                v_short[i, age] = hold_short

    labels: list[dict[str, Any]] = []
    action_rows: list[dict[str, Any]] = []
    side_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    horizon_counts: Counter[str] = Counter()
    for i in range(n - MAX_AGE - 2):
        action = int(p_flat[i])
        side = 1 if action == 1 else -1 if action == 2 else 0
        if side == 0:
            sim = {"hold": 0, "raw": 0.0, "mfe": 0.0, "mae": 0.0, "reason": "cash"}
            edge = 0.0
        else:
            sim = _simulate_entry_path(close, high, low, p_long, p_short, start_i=i, side=side)
            edge = max(0.0, (v_long[i + 1, 1] if side > 0 else v_short[i + 1, 1]) - v_flat[i + 1])
        mfe = float(sim["mfe"])
        mae = float(sim["mae"])
        atr_i = _clip(float(atr[i]), 0.002, 0.040)
        if side == 0 or edge <= 0.0:
            tp = 0.0
            sl = 0.0
            utility = 0.0
        else:
            tp = _clip(max(0.70 * max(mfe, 0.0), 1.15 * atr_i), *TP_BOUNDS)
            sl = _clip(max(1.10 * abs(min(mae, 0.0)), 0.85 * atr_i), *SL_BOUNDS)
            if sl >= tp:
                tp = _clip(sl * 1.35, *TP_BOUNDS)
            utility = float(sim["raw"]) * NOTIONAL - 2.0 * FEE_PER_SIDE * NOTIONAL - HOLD_PENALTY * int(sim["hold"])
        side_name = "LONG" if side > 0 else "SHORT" if side < 0 else "CASH"
        reason = str(sim["reason"])
        side_counts[side_name] += 1
        reason_counts[reason] += 1
        horizon_counts[str(int(sim["hold"]))] += 1
        labels.append(
            {
                "timestamp": df.iloc[i]["timestamp"],
                "split": split,
                "close": float(close[i]),
                "label_side": side_name,
                "label_side_id": side,
                "label_horizon": int(sim["hold"]),
                "label_tp_price_move": float(tp),
                "label_sl_price_move": float(sl),
                "label_utility": float(utility),
                "label_net_return": float(sim["raw"]) * NOTIONAL if side else 0.0,
                "label_hold_bars": int(sim["hold"]),
                "label_reason": reason,
                "label_mfe": mfe,
                "label_mae": mae,
                "dp_flat_value": float(v_flat[i]),
                "dp_entry_edge": float(edge),
                "dp_action": "ENTER_LONG" if action == 1 else "ENTER_SHORT" if action == 2 else "CASH",
                "dp_state": "FLAT",
            }
        )
        action_rows.append(
            {
                "timestamp": df.iloc[i]["timestamp"],
                "state": "FLAT",
                "optimal_action": "ENTER_LONG" if action == 1 else "ENTER_SHORT" if action == 2 else "CASH",
                "optimal_side": side_name if side else "NONE",
                "age": 0,
                "oracle_value": float(v_flat[i]),
            }
        )
        if side:
            age = 1
            j = i
            while j < min(n - 1, i + int(sim["hold"])):
                act = int(p_long[j, age] if side > 0 else p_short[j, age])
                action_rows.append(
                    {
                        "timestamp": df.iloc[j]["timestamp"],
                        "state": side_name,
                        "optimal_action": "EXIT" if act == 1 else "HOLD",
                        "optimal_side": side_name,
                        "age": int(age),
                        "oracle_value": float(v_long[j, age] if side > 0 else v_short[j, age]),
                    }
                )
                if act == 1:
                    break
                j += 1
                age += 1

    lab = pd.DataFrame(labels)
    path = pd.DataFrame(action_rows)
    active = lab[lab["label_side_id"] != 0]
    diag = {
        "split": split,
        "rows": int(len(lab)),
        "side_counts": dict(side_counts),
        "reason_counts": dict(reason_counts),
        "hold_quantiles_active": {
            str(q): float(active["label_hold_bars"].quantile(q)) if len(active) else 0.0
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "horizon_top_counts": dict(Counter(horizon_counts).most_common(12)),
        "utility_quantiles_active": {
            str(q): float(active["label_utility"].quantile(q)) if len(active) else 0.0
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
    }
    return lab, path, diag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = _read_frame(TRAIN_CSV)
    oos = _read_frame(EVAL_CSV)
    train_labels, train_path, train_diag = _build_dp_labels(train, "train_2025")
    oos_labels, oos_path, oos_diag = _build_dp_labels(oos, "oos_2026")
    train_labels.to_csv(OUT_DIR / "train_2025_multihorizon_tb_labels.csv", index=False)
    oos_labels.to_csv(OUT_DIR / "oos_2026_multihorizon_tb_labels.csv", index=False)
    train_path.to_csv(OUT_DIR / "train_2025_dp_action_path_labels.csv", index=False)
    oos_path.to_csv(OUT_DIR / "oos_2026_dp_action_path_labels.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "status": "labels_built",
        "label_mode": "finite_state_dp_optimal_trajectory",
        "dp_actions": ["ENTER_LONG", "ENTER_SHORT", "HOLD", "EXIT", "CASH"],
        "params": {
            "max_age": MAX_AGE,
            "leverage": LEVERAGE,
            "margin_fraction_for_label": MARGIN_FRACTION_FOR_LABEL,
            "notional_for_label": NOTIONAL,
            "fee_per_side": FEE_PER_SIDE,
            "hold_penalty": HOLD_PENALTY,
            "min_entry_edge": MIN_ENTRY_EDGE,
            "tp_bounds": TP_BOUNDS,
            "sl_bounds": SL_BOUNDS,
        },
        "risk_contract": {
            "notional": "margin_fraction * leverage",
            "pnl": "price_move * notional - fees",
            "tp_sl_targets": "price_move targets, not leverage-multiplied account thresholds",
        },
        "train_2025": train_diag,
        "oos_2026": oos_diag,
        "artifacts": {
            "train_entry_labels": str((OUT_DIR / "train_2025_multihorizon_tb_labels.csv").relative_to(ROOT)),
            "oos_entry_labels": str((OUT_DIR / "oos_2026_multihorizon_tb_labels.csv").relative_to(ROOT)),
            "train_path_labels": str((OUT_DIR / "train_2025_dp_action_path_labels.csv").relative_to(ROOT)),
            "oos_path_labels": str((OUT_DIR / "oos_2026_dp_action_path_labels.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "label_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
