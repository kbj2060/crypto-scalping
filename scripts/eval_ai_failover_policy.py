#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, prepare_features, predict_policy_frame  # noqa: E402
from scripts.eval_lifecycle_ai_stress import AI_GROUPS, _stress_frame  # noqa: E402


DEFAULT_BASE_POLICY = ROOT / "data/ensemble/supervised/fully_learned_ai_combo_grid/patchtst__tide__dlinear.pkl"
DEFAULT_DROPOUT_POLICY = ROOT / "data/ensemble/supervised/fully_learned_ai_dropout/patchtst_tide_dlinear_dropout.pkl"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/ai_failover_policy_patchtst_tide_dlinear_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _fill_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    col = "open" if "open" in df.columns else "close"
    price = float(pd.to_numeric(df[col], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return price * (1.0 + slip if entry else 1.0 - slip)
    return price * (1.0 - slip if entry else 1.0 + slip)


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return max(len(df) / 288.0, 1e-8)
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _group_zero_mask(df: pd.DataFrame, group: str, *, eps: float) -> np.ndarray:
    cols = [c for c in AI_GROUPS.get(group, []) if c in df.columns]
    if not cols:
        return np.ones(len(df), dtype=bool)
    mat = df.loc[:, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).abs().to_numpy(dtype=np.float64)
    return np.sum(mat, axis=1) <= float(eps)


def _health_masks(df: pd.DataFrame, *, eps: float) -> dict[str, np.ndarray]:
    masks = {group: _group_zero_mask(df, group, eps=eps) for group in AI_GROUPS}
    stacked = np.vstack([m.astype(bool) for m in masks.values()])
    masks["any_zero_group"] = np.any(stacked, axis=0)
    masks["two_plus_zero_groups"] = np.sum(stacked, axis=0) >= 2
    masks["all_zero_groups"] = np.all(stacked, axis=0)
    return masks


def _decisions(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    close = _close(df)
    feat = prepare_features(df, side_hint=0, close=close)
    return predict_policy_frame(bundle, feat)


def _combine_decisions(base: pd.DataFrame, dropout: pd.DataFrame, mask: np.ndarray, *, mode: str) -> pd.DataFrame:
    out = base.copy()
    m = np.asarray(mask, dtype=bool)
    if mode == "dropout_on_unhealthy":
        out.loc[m, :] = dropout.loc[m, :].to_numpy()
    elif mode == "cash_on_unhealthy":
        out.loc[m, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
        out.loc[m, "leverage"] = 1.0
    elif mode == "half_notional_on_unhealthy":
        cols = ["notional_exposure", "position_fraction", "take_profit", "stop_loss"]
        out.loc[m, cols] = out.loc[m, cols] * 0.5
    elif mode == "base":
        pass
    elif mode == "dropout":
        out = dropout.copy()
    else:
        raise ValueError(f"unknown combine mode: {mode}")
    return out


def backtest_decisions(df: pd.DataFrame, decisions: pd.DataFrame, *, fee: float, slip: float) -> dict[str, Any]:
    close = _close(df)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    next_cooldown = 0
    cooldown_left = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    action_counts = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}

    def mark_equity(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            age = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and age >= max_hold:
                reason = "max_hold"
            if reason:
                exit_price = _fill_price(df, min(i + 1, len(df) - 1), pos, slip, entry=False)
                if pos > 0:
                    raw = (exit_price - entry_price) / max(entry_price, 1e-12)
                else:
                    raw = (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                pos = 0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos == 0:
            if cooldown_left > 0:
                cooldown_left -= 1
                action_counts["cash"] += 1
                continue
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                action_counts["cash"] += 1
                continue
            pos = int(dec.side)
            action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
            entry_price = _fill_price(df, min(i + 1, len(df) - 1), pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(dec.notional_exposure)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
    if pos != 0:
        eq, _ = mark_equity(len(df) - 1)
        cash = eq
        exits["open_at_end"] = exits.get("open_at_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "action_counts": action_counts,
        "exits": exits,
    }


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {k: bt.get(k) for k in ("pnl", "mdd", "trades", "wr", "trades_per_day", "long_entries", "short_entries", "avg_notional", "avg_leverage")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate AI health failover policies for the fully learned governor.")
    p.add_argument("--base-policy", type=Path, default=DEFAULT_BASE_POLICY)
    p.add_argument("--dropout-policy", type=Path, default=DEFAULT_DROPOUT_POLICY)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--zero-eps", type=float, default=1e-12)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    base_bundle = joblib.load(args.base_policy)
    dropout_bundle = joblib.load(args.dropout_policy)
    raw_eval = _read(args.eval_csv)
    modes = [
        "normal",
        "patchtst_zero",
        "tide_zero",
        "dlinear_zero",
        "patchtst__tide_zero",
        "patchtst__dlinear_zero",
        "tide__dlinear_zero",
        "all_ai_zero",
        "patchtst_stale_1d",
        "tide_stale_1d",
        "dlinear_stale_1d",
    ]
    results: dict[str, Any] = {}
    for stress_mode in modes:
        df, stress_meta = _stress_frame(raw_eval, stress_mode)
        masks = _health_masks(df, eps=float(args.zero_eps))
        base_dec = _decisions(df, base_bundle)
        dropout_dec = _decisions(df, dropout_bundle)
        strategies = {
            "base": _combine_decisions(base_dec, dropout_dec, masks["any_zero_group"], mode="base"),
            "dropout": _combine_decisions(base_dec, dropout_dec, masks["any_zero_group"], mode="dropout"),
            "dropout_on_any_zero": _combine_decisions(base_dec, dropout_dec, masks["any_zero_group"], mode="dropout_on_unhealthy"),
            "cash_on_two_plus_zero": _combine_decisions(base_dec, dropout_dec, masks["two_plus_zero_groups"], mode="cash_on_unhealthy"),
            "half_notional_on_any_zero": _combine_decisions(base_dec, dropout_dec, masks["any_zero_group"], mode="half_notional_on_unhealthy"),
        }
        results[stress_mode] = {
            "stress": stress_meta,
            "health": {
                key: float(mask.mean())
                for key, mask in masks.items()
            },
            "strategies": {
                name: _compact(backtest_decisions(df, dec, fee=float(args.fee), slip=float(args.slip)))
                for name, dec in strategies.items()
            },
        }
    report = {
        "type": "ai_failover_policy_patchtst_tide_dlinear_2026",
        "base_policy": str(args.base_policy),
        "dropout_policy": str(args.dropout_policy),
        "eval_csv": str(args.eval_csv),
        "results": results,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "normal": results["normal"], "all_ai_zero": results["all_ai_zero"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
