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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_POLICY,
    DEFAULT_TRAIN_CSV,
    _audit,
    _close,
    _decisions,
    _days,
    _quality_scaled_decisions,
)
from scripts.eval_lifecycle_ai_stress import _stress_frame  # noqa: E402


DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_risk_overlay_grid_hf_v4_2026.json"


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in (
            "pnl",
            "mdd",
            "trades",
            "trades_per_day",
            "wr",
            "avg_notional",
            "avg_leverage",
            "long_entries",
            "short_entries",
            "entry_blocks",
            "exits",
        )
    }


def backtest_hf_risk_overlay(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    max_daily_trades: int,
    daily_loss_limit: float,
    daily_dd_limit: float,
    global_dd_cut: float,
    global_dd_mult: float,
    loss_cooldown_bars: int,
    loss_streak_soft: int,
    loss_streak_mult: float,
    max_notional: float,
    daily_profit_boost_start: float = 999.0,
    daily_profit_boost_mult: float = 1.0,
    equity_high_boost_dd: float = -1.0,
    equity_high_boost_mult: float = 1.0,
    trailing_trigger: float = 999.0,
    trailing_gap: float = 999.0,
    min_hold_before_risk_exit: int = 2,
) -> dict[str, Any]:
    close = _close(df)
    fill_col = "open" if "open" in df.columns else "close"
    fill_px = (
        pd.to_numeric(df[fill_col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )
    if "timestamp" in df.columns:
        day_codes = pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("D").astype("int64").to_numpy()
    else:
        day_codes = (np.arange(len(df), dtype=np.int64) // 288).astype(np.int64)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    take_profits = pd.to_numeric(decisions["take_profit"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    stop_losses = pd.to_numeric(decisions["stop_loss"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    max_holds = pd.to_numeric(decisions["max_hold_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    cooldowns = pd.to_numeric(decisions["cooldown_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
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
    cooldown_left = 0
    model_cooldown = 0
    loss_cooldown_left = 0
    loss_streak = 0
    peak_unrealized = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    entry_blocks: dict[str, int] = {}

    day_key: Any = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def fill_price(idx: int, side: int, *, entry: bool) -> float:
        price = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
        if side > 0:
            return price * (1.0 + slip if entry else 1.0 - slip)
        return price * (1.0 - slip if entry else 1.0 + slip)

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        if pos > 0:
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def block(reason: str) -> None:
        entry_blocks[reason] = entry_blocks.get(reason, 0) + 1

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, cooldown_left, model_cooldown
        nonlocal trades, wins, loss_streak, loss_cooldown_left, daily_trades
        nonlocal peak_unrealized
        exit_price = fill_price(min(i + 1, len(df) - 1), pos, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(loss_cooldown_bars))
        exits[reason] = exits.get(reason, 0) + 1
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        cooldown_left = int(model_cooldown)
        model_cooldown = 0
        peak_unrealized = 0.0

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            age = i - entry_idx
            reason = ""
            if take_profit > 0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif peak_unrealized >= float(trailing_trigger) and unreal <= peak_unrealized - float(trailing_gap):
                reason = "trailing_lock"
            elif max_hold > 0 and age >= max_hold:
                reason = "max_hold"
            elif age >= int(min_hold_before_risk_exit) and unreal < 0.0 and daily_dd >= float(daily_dd_limit):
                reason = "daily_dd_risk_exit"
            elif age >= int(min_hold_before_risk_exit) and unreal < 0.0 and account_dd >= float(global_dd_cut) * 1.35:
                reason = "account_dd_risk_exit"
            if reason:
                close_position(i, reason)
                continue

        if pos != 0:
            continue
        if cooldown_left > 0:
            cooldown_left -= 1
            block("model_cooldown")
            continue
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(max_daily_trades):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(daily_loss_limit)):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(daily_dd_limit)):
            block("daily_dd_lock")
            continue

        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            block("cash_signal")
            continue

        n = float(notionals[i])
        if account_dd >= float(global_dd_cut):
            n *= float(global_dd_mult)
        if loss_streak >= int(loss_streak_soft):
            steps = loss_streak - int(loss_streak_soft) + 1
            n *= float(loss_streak_mult) ** float(max(0, steps))
        if daily_realized >= float(daily_profit_boost_start):
            n *= float(daily_profit_boost_mult)
        if float(equity_high_boost_dd) >= 0.0 and account_dd <= float(equity_high_boost_dd):
            n *= float(equity_high_boost_mult)
        n = float(np.clip(n, 0.0, float(max_notional)))
        if n <= 1e-8:
            block("zero_notional")
            continue

        pos = int(sides[i])
        entry_price = fill_price(min(i + 1, len(df) - 1), pos, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = float(leverages[i])
        take_profit = float(take_profits[i])
        stop_loss = float(stop_losses[i])
        max_hold = int(max_holds[i])
        model_cooldown = int(cooldowns[i])
        peak_unrealized = 0.0
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage

    if pos != 0:
        close_position(len(df) - 2, "forced_end")

    entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "entry_blocks": entry_blocks,
        "exits": exits,
    }


def _entry_configs() -> list[dict[str, float]]:
    return [
        {"notional_mult": 1.5, "quality_floor": 0.00, "confidence_floor": 0.0, "max_notional": 3.6},
        {"notional_mult": 2.0, "quality_floor": 0.01, "confidence_floor": 0.0, "max_notional": 3.6},
        {"notional_mult": 2.75, "quality_floor": 0.01, "confidence_floor": 0.0, "max_notional": 3.6},
        {"notional_mult": 3.5, "quality_floor": 0.02, "confidence_floor": 0.0, "max_notional": 3.6},
    ]


def _risk_configs() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for max_daily in (12, 16, 20):
        for daily_loss in (0.025, 0.04, 0.06):
            for daily_dd in (0.035, 0.05, 0.07):
                for loss_cd in (6, 12, 24):
                    for dd_cut, dd_mult in ((0.12, 0.45), (0.18, 0.60), (0.25, 0.75)):
                        out.append(
                            {
                                "max_daily_trades": max_daily,
                                "daily_loss_limit": daily_loss,
                                "daily_dd_limit": daily_dd,
                                "global_dd_cut": dd_cut,
                                "global_dd_mult": dd_mult,
                                "loss_cooldown_bars": loss_cd,
                                "loss_streak_soft": 2,
                                "loss_streak_mult": 0.65,
                                "max_notional": 3.6,
                            }
                        )
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Search causal account-state risk overlays for HF entry policy.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    eval_df = _read(args.eval_csv)
    base_dec = _decisions(eval_df, policy)
    rows: list[dict[str, Any]] = []
    for entry_cfg in _entry_configs():
        dec = _quality_scaled_decisions(base_dec, **entry_cfg)
        for risk_cfg in _risk_configs():
            bt = backtest_hf_risk_overlay(eval_df, dec, fee=float(args.fee), slip=float(args.slip), **risk_cfg)
            name = (
                f"m{entry_cfg['notional_mult']}_q{entry_cfg['quality_floor']}"
                f"_d{risk_cfg['daily_loss_limit']}_dd{risk_cfg['daily_dd_limit']}"
                f"_max{risk_cfg['max_daily_trades']}_cd{risk_cfg['loss_cooldown_bars']}"
                f"_g{risk_cfg['global_dd_cut']}"
            )
            rows.append({"name": name, "entry_config": entry_cfg, "risk_config": risk_cfg, "eval": _compact(bt)})

    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    goal = [r for r in ranked if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0]
    mdd30 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -30.0]
    mdd25 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -25.0]

    top_for_stress = (mdd30 or goal)[:5]
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        key = f"cost_{mult:g}x"
        cost_stress[key] = []
        for row in top_for_stress:
            dec = _quality_scaled_decisions(base_dec, **row["entry_config"])
            bt = backtest_hf_risk_overlay(
                eval_df,
                dec,
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
                **row["risk_config"],
            )
            cost_stress[key].append({"name": row["name"], "eval": _compact(bt)})

    ai_stress: dict[str, Any] = {}
    for mode in ("normal", "all_ai_zero", "patchtst_zero", "tide_zero", "dlinear_zero"):
        df, meta = _stress_frame(eval_df, mode)
        dec0 = _decisions(df, policy)
        ai_stress[mode] = {"stress": meta, "results": []}
        for row in top_for_stress[:3]:
            dec = _quality_scaled_decisions(dec0, **row["entry_config"])
            bt = backtest_hf_risk_overlay(
                df,
                dec,
                fee=float(args.fee),
                slip=float(args.slip),
                **row["risk_config"],
            )
            ai_stress[mode]["results"].append({"name": row["name"], "eval": _compact(bt)})

    report = {
        "type": "hf_risk_overlay_grid_hf_v4_2026",
        "policy": str(args.policy),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "grid_size": len(rows),
        "grid": rows,
        "ranked_by_pnl": [{"name": r["name"], **r["eval"]} for r in ranked[:30]],
        "ranked_goal_5_to_20_trades_per_day": [{"name": r["name"], **r["eval"]} for r in goal[:30]],
        "ranked_goal_mdd_lte_30": [{"name": r["name"], **r["eval"]} for r in mdd30[:30]],
        "ranked_goal_mdd_lte_25": [{"name": r["name"], **r["eval"]} for r in mdd25[:30]],
        "cost_stress": cost_stress,
        "ai_stress": ai_stress,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({
        "report": str(args.report_out),
        "top_goal": report["ranked_goal_5_to_20_trades_per_day"][:8],
        "top_mdd30": report["ranked_goal_mdd_lte_30"][:8],
        "top_mdd25": report["ranked_goal_mdd_lte_25"][:8],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
