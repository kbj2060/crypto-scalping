#!/usr/bin/env python3
"""Materialize the frozen split-Oracle OOS walk-forward ledger and trade chart."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_eth_split_oracle_3head_20260724 as trained  # noqa: E402


omega = trained.omega
parent = trained.parent
OUT_DIR = trained.OUT_DIR


def replay(frame: pd.DataFrame, decisions: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    arrays = {column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close")}
    active = omega._active(decisions)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = notional = leverage = take_profit = stop_loss = 0.0
    entry_idx = entry_signal_idx = 0
    entry_fee = 0.0
    max_hold = cooldown = next_cooldown = 0
    trades: list[dict[str, Any]] = []
    curve: list[dict[str, Any]] = []

    def close_trade(signal_i: int, reason: str) -> None:
        nonlocal cash, pos, cooldown
        filled, exit_price, exit_fee, route = omega._try_execution(
            arrays, signal_i, pos, entry=False, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            return
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * exit_fee * notional
        trades.append(
            {
                "entry_signal_index": entry_signal_idx,
                "entry_index": entry_idx,
                "exit_signal_index": int(signal_i),
                "entry_timestamp": frame["timestamp"].iloc[entry_idx],
                "exit_timestamp": frame["timestamp"].iloc[min(signal_i + 1, len(frame) - 1)],
                "side": pos,
                "reason": reason,
                "win": int(cash > entry_equity),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": notional,
                "margin_fraction": notional / max(leverage, 1.0e-12),
                "leverage": leverage,
                "trade_return": cash / max(entry_equity, 1.0e-12) - 1.0,
                "equity_after": cash,
                "exit_route": route,
            }
        )
        pos = 0
        cooldown = int(next_cooldown)

    for i in range(0, len(frame) - 2):
        if pos:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            equity = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            equity = cash
        peak = max(peak, equity)
        mdd = min(mdd, equity / max(peak, 1.0e-12) - 1.0)
        curve.append({"timestamp": frame["timestamp"].iloc[i], "equity": equity, "drawdown": equity / max(peak, 1.0e-12) - 1.0})
        if pos:
            hold = i - entry_signal_idx
            reason = ""
            if take_profit > 0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                close_trade(i, reason)
                continue
        if pos:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = decisions.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, price, fee_paid, _route = omega._try_execution(
            arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            continue
        pos = side
        entry_price = price
        entry_equity = cash
        entry_signal_idx = i
        entry_idx = min(i + 1, len(frame) - 1)
        entry_fee = fee_paid
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        cash -= cash * entry_fee * notional

    if pos:
        fill_i = len(frame) - 1
        exit_price = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades.append(
            {
                "entry_signal_index": entry_signal_idx,
                "entry_index": entry_idx,
                "exit_signal_index": fill_i,
                "entry_timestamp": frame["timestamp"].iloc[entry_idx],
                "exit_timestamp": frame["timestamp"].iloc[fill_i],
                "side": pos,
                "reason": "forced_end",
                "win": int(cash > entry_equity),
                "entry_price": entry_price,
                "exit_price": exit_price,
                "notional": notional,
                "margin_fraction": notional / max(leverage, 1.0e-12),
                "leverage": leverage,
                "trade_return": cash / max(entry_equity, 1.0e-12) - 1.0,
                "equity_after": cash,
                "exit_route": "forced_end",
            }
        )
    ledger = pd.DataFrame(trades)
    equity = pd.DataFrame(curve)
    metrics = {
        "pnl": (cash - 1.0) * 100.0,
        "mdd": mdd * 100.0,
        "trades": int(len(ledger)),
        "wr": float(ledger["win"].mean()) if len(ledger) else 0.0,
    }
    return metrics, ledger, equity


def plot(frame: pd.DataFrame, ledger: pd.DataFrame, equity: pd.DataFrame, path: Path) -> dict[str, str]:
    end = pd.to_datetime(frame["timestamp"].iloc[-1])
    start = end - pd.Timedelta(days=7)
    view = frame.loc[frame["timestamp"].between(start, end)]
    shown = ledger.loc[pd.to_datetime(ledger["entry_timestamp"]).between(start, end)]
    fig, (price_axis, equity_axis) = plt.subplots(2, 1, figsize=(17, 9), sharex=False, gridspec_kw={"height_ratios": [3, 1]})
    price_axis.plot(view["timestamp"], view["close"], color="#334155", linewidth=0.9)
    for row in shown.itertuples(index=False):
        color = "#16a34a" if row.side > 0 else "#dc2626"
        price_axis.scatter(row.entry_timestamp, row.entry_price, marker="^" if row.side > 0 else "v", s=38, color=color, zorder=4)
        price_axis.scatter(row.exit_timestamp, row.exit_price, marker="x", s=30, color=color, zorder=4)
        price_axis.plot([row.entry_timestamp, row.exit_timestamp], [row.entry_price, row.exit_price], color=color, alpha=0.45, linewidth=0.7)
    price_axis.set_title("ETH split-Oracle student — frozen OOS trades (last 7 days)")
    price_axis.set_ylabel("ETHUSDT")
    price_axis.grid(alpha=0.15)
    equity_axis.plot(equity["timestamp"], equity["equity"], color="#0f766e", linewidth=1.1)
    equity_axis.set_title("Full OOS fresh-forward equity")
    equity_axis.set_ylabel("Equity")
    equity_axis.set_xlabel("UTC")
    equity_axis.grid(alpha=0.15)
    equity_axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return {"trade_window_start": str(start), "trade_window_end": str(end)}


def main() -> int:
    report = json.loads((OUT_DIR / "report.json").read_text(encoding="utf-8"))
    q = float(report["selection"]["quality_threshold"])
    scale = float(report["selection"]["notional_scale"])
    q_tag = f"q{int(round(q * 100)):03d}"
    frame = trained.prepare_frames()["oos"]
    prediction = pd.read_csv(OUT_DIR / f"oos_predictions_{q_tag}.csv", parse_dates=["timestamp"], low_memory=False)
    frame, prediction = omega._align(frame, prediction, "frozen_oos_chart")
    decisions = trained.apply_notional_scale(parent._to_decisions(prediction, oof=False), scale=scale)
    fee, slip = omega._load_fee_slip()
    metrics, ledger, equity = replay(frame, decisions, fee=fee, slip=slip, cost_mult=1.0)
    expected = report["oos"]
    for key in ("pnl", "mdd", "trades", "wr"):
        if not np.isclose(float(metrics[key]), float(expected[key]), atol=1.0e-10):
            raise RuntimeError(f"chart replay mismatch for {key}: {metrics[key]} != {expected[key]}")
    ledger_path = OUT_DIR / "oos_fresh_forward_trade_ledger.csv"
    equity_path = OUT_DIR / "oos_fresh_forward_equity.csv"
    chart_path = OUT_DIR / "oos_fresh_forward_trade_chart.png"
    ledger.to_csv(ledger_path, index=False)
    equity.to_csv(equity_path, index=False)
    chart_meta = plot(frame, ledger, equity, chart_path)
    chart_report = {
        "metrics": metrics,
        "frozen_quality_threshold": q,
        "frozen_notional_scale": scale,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "materialization_only_no_reselection": True,
        "chart": {"path": str(chart_path), **chart_meta},
        "ledger": str(ledger_path),
        "equity": str(equity_path),
    }
    (OUT_DIR / "oos_chart_report.json").write_text(json.dumps(chart_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(chart_report, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
