#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_omega1_2_1_dp_trajectory_labels_20260620 as dp


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_full_label_pack_daytrade_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = dp.TRAIN_CSV
EVAL_CSV = dp.EVAL_CSV

TREND_WINDOWS = [6, 12, 24, 48, 96]
TREND_T_THRESHOLD = 1.8
QUANTILE_HORIZON = 96
QUANTILES = [0.50, 0.70, 0.85, 0.95]
SURVIVAL_BUCKETS = [3, 6, 12, 24, 48, 96]
BANDIT_MARGIN_GRID = [0.015, 0.025, 0.04]
BANDIT_LEVERAGE = 2.0
FEE_PER_SIDE = dp.FEE_PER_SIDE


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _trend_scan(close: np.ndarray) -> pd.DataFrame:
    logp = np.log(np.maximum(close, 1e-12))
    n = len(close)
    side = np.zeros(n, dtype=np.int8)
    best_t = np.zeros(n, dtype=np.float64)
    best_beta = np.zeros(n, dtype=np.float64)
    best_window = np.zeros(n, dtype=np.int16)
    for i in range(n):
        for window in TREND_WINDOWS:
            if i + window >= n or window <= 2:
                continue
            y = logp[i : i + window]
            x = np.arange(window, dtype=np.float64)
            x_mean = float(x.mean())
            y_mean = float(y.mean())
            denom = float(((x - x_mean) ** 2).sum())
            if denom <= 0.0:
                continue
            beta = float(((x - x_mean) * (y - y_mean)).sum() / denom)
            alpha = y_mean - beta * x_mean
            resid = y - (alpha + beta * x)
            rss = float((resid**2).sum())
            if rss <= 1e-12:
                t_value = 0.0
            else:
                se = (rss / (window - 2.0)) ** 0.5 / (denom**0.5)
                t_value = beta / se if se > 1e-12 else 0.0
            if abs(t_value) > abs(best_t[i]):
                best_t[i] = t_value
                best_beta[i] = beta
                best_window[i] = window
        if abs(best_t[i]) >= TREND_T_THRESHOLD:
            side[i] = 1 if best_beta[i] > 0.0 else -1 if best_beta[i] < 0.0 else 0
    return pd.DataFrame(
        {
            "trend_side_id": side,
            "trend_tvalue": best_t,
            "trend_beta": best_beta,
            "trend_window": best_window,
            "trend_strength": np.abs(best_t),
        }
    )


def _path_quantiles(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    n = len(close)
    for i in range(n):
        entry = float(close[i])
        end = min(n, i + QUANTILE_HORIZON + 1)
        if entry <= 0.0 or i + 2 >= n:
            rows.append({})
            continue
        hs = high[i + 1 : end]
        ls = low[i + 1 : end]
        long_mfe = (hs - entry) / entry
        long_mae = (ls - entry) / entry
        short_mfe = (entry - ls) / entry
        short_mae = (entry - hs) / entry
        row: dict[str, float] = {}
        for q in QUANTILES:
            key = int(q * 100)
            row[f"mfe_long_q{key}"] = float(np.quantile(long_mfe, q)) if len(long_mfe) else 0.0
            row[f"mae_long_q{key}"] = float(abs(np.quantile(long_mae, 1.0 - q))) if len(long_mae) else 0.0
            row[f"mfe_short_q{key}"] = float(np.quantile(short_mfe, q)) if len(short_mfe) else 0.0
            row[f"mae_short_q{key}"] = float(abs(np.quantile(short_mae, 1.0 - q))) if len(short_mae) else 0.0
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def _simulate_reward(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    i: int,
    *,
    side: int,
    margin_fraction: float,
    tp_price_move: float,
    sl_price_move: float,
) -> tuple[float, int, str]:
    entry = float(close[i])
    if entry <= 0.0 or side == 0:
        return 0.0, 0, "cash"
    notional = margin_fraction * BANDIT_LEVERAGE
    end = min(len(close) - 1, i + QUANTILE_HORIZON)
    exit_i = end
    raw = (float(close[end]) - entry) / entry if side > 0 else (entry - float(close[end])) / entry
    reason = "vertical"
    for j in range(i + 1, end + 1):
        if side > 0:
            hi_raw = (float(high[j]) - entry) / entry
            lo_raw = (float(low[j]) - entry) / entry
        else:
            hi_raw = (entry - float(low[j])) / entry
            lo_raw = (entry - float(high[j])) / entry
        if lo_raw <= -abs(sl_price_move):
            raw = -abs(sl_price_move)
            exit_i = j
            reason = "stop_loss"
            break
        if hi_raw >= tp_price_move:
            raw = tp_price_move
            exit_i = j
            reason = "take_profit"
            break
    reward = raw * notional - 2.0 * FEE_PER_SIDE * notional
    reward -= 0.0000015 * max(0, exit_i - i)
    return float(reward), int(exit_i - i), reason


def _build_bandit(df: pd.DataFrame, labels: pd.DataFrame, quant: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []
    for i, lab in labels.iterrows():
        side = int(lab["label_side_id"])
        if side == 0:
            rows.append({"bandit_best_margin_fraction": 0.0, "bandit_best_reward": 0.0, "bandit_best_hold": 0, "bandit_best_reason": "cash"})
            continue
        tp = float(quant.loc[i, f"mfe_{'long' if side > 0 else 'short'}_q70"])
        sl = float(quant.loc[i, f"mae_{'long' if side > 0 else 'short'}_q70"])
        tp = min(max(tp, dp.TP_BOUNDS[0]), dp.TP_BOUNDS[1])
        sl = min(max(sl, dp.SL_BOUNDS[0]), dp.SL_BOUNDS[1])
        best: dict[str, Any] | None = None
        for margin in BANDIT_MARGIN_GRID:
            reward, hold, reason = _simulate_reward(high, low, close, int(i), side=side, margin_fraction=margin, tp_price_move=tp, sl_price_move=sl)
            row = {
                "timestamp": lab["timestamp"],
                "split": split,
                "side": "LONG" if side > 0 else "SHORT",
                "margin_fraction": margin,
                "leverage": BANDIT_LEVERAGE,
                "notional": margin * BANDIT_LEVERAGE,
                "tp_price_move": tp,
                "sl_price_move": sl,
                "counterfactual_reward": reward,
                "hold_bars": hold,
                "reason": reason,
            }
            table_rows.append(row)
            if best is None or reward > float(best["counterfactual_reward"]):
                best = row
        if best is None:
            rows.append({"bandit_best_margin_fraction": 0.0, "bandit_best_reward": 0.0, "bandit_best_hold": 0, "bandit_best_reason": "cash"})
        else:
            rows.append(
                {
                    "bandit_best_margin_fraction": float(best["margin_fraction"]),
                    "bandit_best_reward": float(best["counterfactual_reward"]),
                    "bandit_best_hold": int(best["hold_bars"]),
                    "bandit_best_reason": str(best["reason"]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(table_rows)


def _survival_from_labels(labels: pd.DataFrame) -> pd.DataFrame:
    hold = pd.to_numeric(labels["label_hold_bars"], errors="coerce").fillna(0).astype(int)
    event = (labels["label_side_id"].astype(int) != 0).astype(int)
    bucket_ids = []
    for h in hold:
        bid = 0
        for j, b in enumerate(SURVIVAL_BUCKETS, start=1):
            if h <= b:
                bid = j
                break
        bucket_ids.append(bid if bid else len(SURVIVAL_BUCKETS))
    return pd.DataFrame(
        {
            "survival_exit_event": event,
            "survival_time_to_exit": hold,
            "survival_duration_bucket": bucket_ids,
            "survival_censored": 0,
        }
    )


def _compose_labels(df: pd.DataFrame, split: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    dp_labels, dp_path, dp_diag = dp._build_dp_labels(df, split)
    n = len(dp_labels)
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)[:n]
    high = pd.to_numeric(df["high"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)[:n]
    low = pd.to_numeric(df["low"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)[:n]
    trend = _trend_scan(close)
    quant = _path_quantiles(close, high, low)
    survival = _survival_from_labels(dp_labels)
    bandit, bandit_table = _build_bandit(df.iloc[:n].reset_index(drop=True), dp_labels, quant, split)

    out = pd.concat([dp_labels.reset_index(drop=True), trend, quant, survival, bandit], axis=1)
    side = out["label_side_id"].astype(int)
    trend_side = out["trend_side_id"].astype(int)
    agree = (side != 0) & ((trend_side == 0) | (trend_side == side))
    reward_ok = pd.to_numeric(out["bandit_best_reward"], errors="coerce").fillna(0.0) > 0.0
    out.loc[~(agree & reward_ok), ["label_side", "label_side_id", "label_horizon", "label_tp_price_move", "label_sl_price_move", "label_utility"]] = [
        "CASH",
        0,
        0,
        0.0,
        0.0,
        0.0,
    ]
    side = out["label_side_id"].astype(int)
    long_mask = side > 0
    short_mask = side < 0
    out.loc[long_mask, "label_tp_price_move"] = out.loc[long_mask, "mfe_long_q70"].clip(dp.TP_BOUNDS[0], dp.TP_BOUNDS[1])
    out.loc[long_mask, "label_sl_price_move"] = out.loc[long_mask, "mae_long_q70"].clip(dp.SL_BOUNDS[0], dp.SL_BOUNDS[1])
    out.loc[short_mask, "label_tp_price_move"] = out.loc[short_mask, "mfe_short_q70"].clip(dp.TP_BOUNDS[0], dp.TP_BOUNDS[1])
    out.loc[short_mask, "label_sl_price_move"] = out.loc[short_mask, "mae_short_q70"].clip(dp.SL_BOUNDS[0], dp.SL_BOUNDS[1])
    out.loc[side != 0, "label_utility"] = out.loc[side != 0, "bandit_best_reward"].astype(float)
    out.loc[side != 0, "label_horizon"] = out.loc[side != 0, "survival_time_to_exit"].astype(int).clip(1, QUANTILE_HORIZON)
    counts = Counter(out["label_side"].astype(str))
    active = out[out["label_side_id"].astype(int) != 0]
    diag = {
        "split": split,
        "rows": int(len(out)),
        "dp_diag": dp_diag,
        "final_side_counts": dict(counts),
        "active_hold_quantiles": {
            str(q): float(active["label_horizon"].quantile(q)) if len(active) else 0.0
            for q in (0.5, 0.75, 0.9, 0.95, 0.99)
        },
        "trend_side_counts": {str(k): int(v) for k, v in Counter(trend["trend_side_id"].astype(int)).items()},
        "bandit_reason_counts": dict(Counter(bandit["bandit_best_reason"].astype(str))),
    }
    return out, dp_path, bandit_table, diag


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = dp._read_frame(TRAIN_CSV)
    oos = dp._read_frame(EVAL_CSV)
    train_labels, train_path, train_bandit, train_diag = _compose_labels(train, "train_2025")
    oos_labels, oos_path, oos_bandit, oos_diag = _compose_labels(oos, "oos_2026")
    train_labels.to_csv(OUT_DIR / "train_2025_multihorizon_tb_labels.csv", index=False)
    oos_labels.to_csv(OUT_DIR / "oos_2026_multihorizon_tb_labels.csv", index=False)
    train_path.to_csv(OUT_DIR / "train_2025_dp_action_path_labels.csv", index=False)
    oos_path.to_csv(OUT_DIR / "oos_2026_dp_action_path_labels.csv", index=False)
    train_bandit.to_csv(OUT_DIR / "train_2025_bandit_action_reward_table.csv", index=False)
    oos_bandit.to_csv(OUT_DIR / "oos_2026_bandit_action_reward_table.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "status": "labels_built",
        "label_mode": "full_label_pack_trend_dp_quantile_survival_bandit",
        "components": {
            "trend_scanning": {"windows": TREND_WINDOWS, "t_threshold": TREND_T_THRESHOLD},
            "dp_optimal_trajectory": {"actions": ["ENTER_LONG", "ENTER_SHORT", "HOLD", "EXIT", "CASH"], "max_age": dp.MAX_AGE},
            "mfe_mae_quantiles": {"horizon": QUANTILE_HORIZON, "quantiles": QUANTILES},
            "survival": {"buckets": SURVIVAL_BUCKETS},
            "contextual_bandit": {"margin_grid": BANDIT_MARGIN_GRID, "leverage": BANDIT_LEVERAGE},
        },
        "risk_contract": {
            "notional": "margin_fraction * leverage",
            "pnl": "price_move * notional - fees",
            "tp_sl_targets": "price_move targets; account thresholds are price_move * notional",
        },
        "train_2025": train_diag,
        "oos_2026": oos_diag,
        "artifacts": {
            "train_labels": str((OUT_DIR / "train_2025_multihorizon_tb_labels.csv").relative_to(ROOT)),
            "oos_labels": str((OUT_DIR / "oos_2026_multihorizon_tb_labels.csv").relative_to(ROOT)),
            "train_path_labels": str((OUT_DIR / "train_2025_dp_action_path_labels.csv").relative_to(ROOT)),
            "oos_path_labels": str((OUT_DIR / "oos_2026_dp_action_path_labels.csv").relative_to(ROOT)),
            "train_bandit_table": str((OUT_DIR / "train_2025_bandit_action_reward_table.csv").relative_to(ROOT)),
            "oos_bandit_table": str((OUT_DIR / "oos_2026_bandit_action_reward_table.csv").relative_to(ROOT)),
        },
    }
    (OUT_DIR / "label_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), flush=True)


if __name__ == "__main__":
    main()
