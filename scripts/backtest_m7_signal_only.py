#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class BacktestResult:
    pnl_pct: float
    mdd_pct: float
    sharpe: float
    trades: int
    win_rate_pct: float
    avg_hold_bars: float
    avg_pnl_per_trade_pct: float
    profit_factor: float
    long_trades: int
    short_trades: int


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "timestamp" not in df.columns:
        raise KeyError("timestamp column missing")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    required = ["open", "high", "low", "close"]
    for c in required:
        if c not in df.columns:
            raise KeyError(f"{c} column missing")
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=required).reset_index(drop=True)

    defaults = {
        "m7_action": 0.0,
        "m7_confidence": 0.0,
        "m7_size": 0.0,
        "m7_gate_block": 0.0,
        "m7_hold_pred": 6.0,
        "m7_target_hold": 6.0,
        "m7_tp_offset": 0.004,
        "m7_sl_offset": 0.003,
    }
    for c, v in defaults.items():
        if c not in df.columns:
            df[c] = v
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(v)
    return df


def _mdd(eq: np.ndarray) -> float:
    peak = np.maximum.accumulate(eq)
    dd = eq / np.maximum(peak, 1e-12) - 1.0
    return float(dd.min()) * 100.0


def _sharpe(eq: np.ndarray, bars_per_year: int = 365 * 24 * 12) -> float:
    if len(eq) < 10:
        return 0.0
    r = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
    s = float(np.std(r))
    if s < 1e-12:
        return 0.0
    return float(np.mean(r) / s * math.sqrt(bars_per_year))


def run_backtest(df: pd.DataFrame, fee_bps: float, slip_bps: float) -> BacktestResult:
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0
    eq = [1.0]
    trades: list[dict] = []

    pos = 0
    entry_px = 0.0
    tp_px = 0.0
    sl_px = 0.0
    hold_bars = 0
    hold_limit = 0
    entry_i = -1

    open_px = df["open"].to_numpy(np.float64)
    high_px = df["high"].to_numpy(np.float64)
    low_px = df["low"].to_numpy(np.float64)
    close_px = df["close"].to_numpy(np.float64)
    action = df["m7_action"].to_numpy(np.float64)
    conf = np.clip(df["m7_confidence"].to_numpy(np.float64), 0.0, 1.0)
    size = np.clip(df["m7_size"].to_numpy(np.float64), 0.0, 1.0)
    gate = df["m7_gate_block"].to_numpy(np.float64)
    hold_pred = np.nan_to_num(df["m7_target_hold"].to_numpy(np.float64), nan=6.0)
    hold_fallback = np.nan_to_num(df["m7_hold_pred"].to_numpy(np.float64), nan=6.0)
    tp_off = np.clip(np.nan_to_num(df["m7_tp_offset"].to_numpy(np.float64), nan=0.004), 0.001, 0.04)
    sl_off = np.clip(np.nan_to_num(df["m7_sl_offset"].to_numpy(np.float64), nan=0.003), 0.001, 0.04)

    for i in range(len(df) - 1):
        cur_sig = int(np.sign(action[i])) if gate[i] < 0.5 else 0
        next_open = float(open_px[i + 1])
        next_high = float(high_px[i + 1])
        next_low = float(low_px[i + 1])
        next_close = float(close_px[i + 1])

        if pos == 0:
            if cur_sig == 0:
                eq.append(eq[-1])
                continue
            strength = max(float(conf[i]), float(size[i]), 0.2)
            _ = strength  # reserved for future sizing; keep trade notional fixed for fair compare
            pos = cur_sig
            entry_px = next_open * (1.0 + slip if pos > 0 else 1.0 - slip)
            tp = float(tp_off[i])
            sl = float(sl_off[i])
            tp_px = entry_px * (1.0 + tp if pos > 0 else 1.0 - tp)
            sl_px = entry_px * (1.0 - sl if pos > 0 else 1.0 + sl)
            target = float(hold_pred[i]) if np.isfinite(hold_pred[i]) and hold_pred[i] > 0 else float(hold_fallback[i])
            hold_limit = int(np.clip(round(target), 1, 24))
            hold_bars = 0
            entry_i = i + 1
            eq.append(eq[-1] * (1.0 - fee))
            continue

        hold_bars += 1
        exit_px = None
        reason = None

        if pos > 0:
            if next_low <= sl_px:
                exit_px = sl_px * (1.0 - slip)
                reason = "sl"
            elif next_high >= tp_px:
                exit_px = tp_px * (1.0 - slip)
                reason = "tp"
        else:
            if next_high >= sl_px:
                exit_px = sl_px * (1.0 + slip)
                reason = "sl"
            elif next_low <= tp_px:
                exit_px = tp_px * (1.0 + slip)
                reason = "tp"

        exit_signal = int(np.sign(action[i])) if gate[i] < 0.5 else 0
        if exit_px is None and hold_bars >= hold_limit:
            exit_px = next_close * (1.0 - slip if pos > 0 else 1.0 + slip)
            reason = "time"
        elif exit_px is None and exit_signal == 0:
            exit_px = next_close * (1.0 - slip if pos > 0 else 1.0 + slip)
            reason = "flat"
        elif exit_px is None and exit_signal == -pos:
            exit_px = next_close * (1.0 - slip if pos > 0 else 1.0 + slip)
            reason = "flip"

        if exit_px is None:
            mtm = (next_close / max(entry_px, 1e-12) - 1.0) if pos > 0 else (entry_px / max(next_close, 1e-12) - 1.0)
            eq.append(eq[-1] * (1.0 + mtm))
            continue

        pnl = (exit_px / max(entry_px, 1e-12) - 1.0) if pos > 0 else (entry_px / max(exit_px, 1e-12) - 1.0)
        pnl -= fee
        eq.append(eq[-1] * (1.0 + pnl))
        trades.append(
            {
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_i": entry_i,
                "exit_i": i + 1,
                "hold_bars": hold_bars,
                "pnl_frac": pnl,
                "reason": reason,
            }
        )
        pos = 0
        entry_px = tp_px = sl_px = 0.0
        hold_bars = hold_limit = 0
        entry_i = -1

    eq_arr = np.asarray(eq, dtype=np.float64)
    pnl_pct = float((eq_arr[-1] - 1.0) * 100.0)
    wins = [t for t in trades if float(t["pnl_frac"]) > 0.0]
    gp = sum(max(float(t["pnl_frac"]), 0.0) for t in trades)
    gl = -sum(min(float(t["pnl_frac"]), 0.0) for t in trades)
    pf = float(gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else 0.0)

    return BacktestResult(
        pnl_pct=pnl_pct,
        mdd_pct=_mdd(eq_arr),
        sharpe=_sharpe(eq_arr),
        trades=len(trades),
        win_rate_pct=float(100.0 * len(wins) / len(trades)) if trades else 0.0,
        avg_hold_bars=float(np.mean([t["hold_bars"] for t in trades])) if trades else 0.0,
        avg_pnl_per_trade_pct=float(100.0 * np.mean([t["pnl_frac"] for t in trades])) if trades else 0.0,
        profit_factor=pf,
        long_trades=sum(1 for t in trades if t["side"] == "LONG"),
        short_trades=sum(1 for t in trades if t["side"] == "SHORT"),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Simple M7-only backtest from precomputed m7 csv")
    ap.add_argument("--csv", required=True)
    ap.add_argument("--fee-bps", type=float, default=2.0)
    ap.add_argument("--slip-bps", type=float, default=1.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    df = _load(args.csv)
    result = {
        "csv": args.csv,
        "rows": int(len(df)),
        "start": str(df["timestamp"].iloc[0]) if len(df) else None,
        "end": str(df["timestamp"].iloc[-1]) if len(df) else None,
        "fee_bps": float(args.fee_bps),
        "slip_bps": float(args.slip_bps),
        "result": asdict(run_backtest(df, args.fee_bps, args.slip_bps)),
    }

    out = args.out.strip()
    if out:
        out_path = Path(out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
