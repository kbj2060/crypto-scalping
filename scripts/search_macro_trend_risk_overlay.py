#!/usr/bin/env python3
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "tmp/pipeline_audit_causal_regime/trade_candidates_2025_causal_regime.csv"
EVAL_CSV = ROOT / "tmp/pipeline_audit_causal_regime/trade_candidates_2026_causal_regime_strict_scored.csv"
REPORT = ROOT / "data/ensemble/reports/leakfree_macro_trend_risk_overlay_search_2026.json"


@dataclass(frozen=True)
class MacroRiskConfig:
    lookback_bars: int = 6048
    threshold: float = 0.05
    persist_updates: int = 5
    update_bars: int = 288
    notional: float = 3.0
    take_profit: float = 1.0
    trail_arm: float = 0.0
    trail_gap: float = 0.0
    stop_loss: float = 0.0
    lockout_bars: int = 0
    lockout_until_signal_change: bool = True
    vol_bars: int = 0
    vol_ref_bars: int = 0
    vol_scale_floor: float = 1.0
    vol_q: float = 0.0


@dataclass
class BacktestStats:
    pnl: float
    mdd: float
    trades: int
    wr: float
    trades_per_day: float
    long_entries: int
    short_entries: int
    resize_events: int
    avg_notional: float
    exits: dict[str, int]
    config: dict[str, Any]


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def _days(df: pd.DataFrame) -> float:
    if len(df) < 2:
        return 1.0
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _macro_signal(close: np.ndarray, cfg: MacroRiskConfig) -> tuple[np.ndarray, np.ndarray]:
    n = len(close)
    look = max(1, int(cfg.lookback_bars))
    mom = np.full(n, np.nan, dtype=np.float64)
    if n > look:
        mom[look:] = close[look:] / np.maximum(close[:-look], 1e-12) - 1.0
    desired = np.zeros(n, dtype=np.int8)
    desired[mom > float(cfg.threshold)] = 1
    desired[mom < -float(cfg.threshold)] = -1

    out = np.zeros(n, dtype=np.int8)
    current = 0
    pending = 0
    pending_count = 0
    update = max(1, int(cfg.update_bars))
    persist = max(1, int(cfg.persist_updates))
    for i, raw in enumerate(desired):
        if i % update != 0:
            out[i] = current
            continue
        raw_i = int(raw)
        if raw_i == current:
            pending = 0
            pending_count = 0
        elif raw_i == pending:
            pending_count += 1
        else:
            pending = raw_i
            pending_count = 1
        if pending_count >= persist:
            current = raw_i
            pending = 0
            pending_count = 0
        out[i] = current
    for i in range(1, n):
        if i % update != 0:
            out[i] = out[i - 1]
    return out, mom


def _rolling_vol_scale(close: np.ndarray, cfg: MacroRiskConfig) -> np.ndarray:
    n = len(close)
    if cfg.vol_bars <= 1 or cfg.vol_ref_bars <= cfg.vol_bars or cfg.vol_scale_floor >= 0.999:
        return np.ones(n, dtype=np.float64)
    ret = np.zeros(n, dtype=np.float64)
    ret[1:] = np.diff(np.log(np.maximum(close, 1e-12)))
    ser = pd.Series(ret)
    vol = ser.rolling(int(cfg.vol_bars), min_periods=max(2, int(cfg.vol_bars // 3))).std().shift(1)
    ref = vol.rolling(int(cfg.vol_ref_bars), min_periods=max(5, int(cfg.vol_ref_bars // 5))).quantile(float(cfg.vol_q)).shift(1)
    ratio = (ref / vol.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    scale = ratio.clip(lower=float(cfg.vol_scale_floor), upper=1.0).fillna(1.0).to_numpy(dtype=np.float64)
    return np.clip(scale, float(cfg.vol_scale_floor), 1.0)


def backtest(df: pd.DataFrame, cfg: MacroRiskConfig, *, fee: float = 0.0005, slip: float = 0.0002) -> BacktestStats:
    close = _close(df)
    signal, _ = _macro_signal(close, cfg)
    vol_scale = _rolling_vol_scale(close, cfg)
    cost = float(fee) + float(slip)

    cash_equity = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0.0
    entry_price = 0.0
    entry_equity = 1.0
    trade_peak_ret = 0.0
    lock_signal = 0
    lock_left = 0

    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    resize_events = 0
    notional_sum = 0.0
    notional_samples = 0
    exits: dict[str, int] = {}

    for i in range(len(close) - 1):
        price = float(close[i])
        next_price = float(close[i + 1])
        if price <= 0.0 or next_price <= 0.0 or not np.isfinite(price + next_price):
            continue

        raw_signal = int(signal[i])
        if lock_left > 0:
            lock_left -= 1
        if lock_signal and (raw_signal == 0 or raw_signal != lock_signal):
            lock_signal = 0
            lock_left = 0

        tradable_signal = raw_signal
        if (lock_signal and raw_signal == lock_signal) or lock_left > 0:
            tradable_signal = 0

        target = float(tradable_signal) * float(cfg.notional) * float(vol_scale[i])
        exit_reason = ""
        mark_equity = cash_equity
        if pos != 0.0 and entry_price > 0.0:
            side = float(np.sign(pos))
            mark_equity = cash_equity + side * (price / entry_price - 1.0) * abs(pos) - cost * abs(pos)

        if pos != 0.0:
            trade_ret = mark_equity / max(entry_equity, 1e-12) - 1.0
            trade_peak_ret = max(trade_peak_ret, trade_ret)
            if cfg.stop_loss > 0.0 and trade_ret <= -float(cfg.stop_loss):
                exit_reason = "stop_loss"
            elif cfg.take_profit > 0.0 and trade_ret >= float(cfg.take_profit):
                exit_reason = "take_profit"
            elif (
                cfg.trail_arm > 0.0
                and cfg.trail_gap > 0.0
                and trade_peak_ret >= float(cfg.trail_arm)
                and trade_ret <= trade_peak_ret - float(cfg.trail_gap)
            ):
                exit_reason = "trailing_take_profit"
            elif target != 0.0 and np.sign(target) != np.sign(pos):
                exit_reason = "signal_flip"
            elif target == 0.0 and raw_signal == 0:
                exit_reason = "signal_flat"

            if exit_reason:
                cash_equity = mark_equity
                wins += int(cash_equity / max(entry_equity, 1e-12) > 1.0)
                exits[exit_reason] = exits.get(exit_reason, 0) + 1
                if exit_reason in {"take_profit", "trailing_take_profit", "stop_loss"}:
                    if bool(cfg.lockout_until_signal_change):
                        lock_signal = int(np.sign(pos))
                    lock_left = max(lock_left, int(cfg.lockout_bars))
                    if target != 0.0 and np.sign(target) == np.sign(pos):
                        target = 0.0
                pos = 0.0
                entry_price = 0.0
                entry_equity = cash_equity
                trade_peak_ret = 0.0

        if pos == 0.0 and target != 0.0:
            pos = float(target)
            entry_price = price
            cash_equity -= cost * abs(pos)
            entry_equity = cash_equity
            trade_peak_ret = 0.0
            trades += 1
            long_entries += int(pos > 0.0)
            short_entries += int(pos < 0.0)
        elif pos != 0.0 and target != 0.0 and np.sign(target) == np.sign(pos):
            delta = float(target - pos)
            if abs(delta) > 1e-9:
                # Same-side resize realizes the changed notional at the current mark,
                # then re-anchors the fixed-contract exposure from this price.
                cash_equity = mark_equity - cost * abs(delta)
                pos = float(target)
                entry_price = price
                entry_equity = cash_equity
                trade_peak_ret = 0.0
                resize_events += 1

        mark_equity = cash_equity
        if pos != 0.0:
            side = float(np.sign(pos))
            mark_equity = cash_equity + side * (price / max(entry_price, 1e-12) - 1.0) * abs(pos) - cost * abs(pos)
            notional_sum += abs(pos)
            notional_samples += 1

        peak = max(peak, mark_equity)
        mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
        if mark_equity <= 0.05:
            exits["ruin"] = exits.get("ruin", 0) + 1
            break

    if pos != 0.0:
        side = float(np.sign(pos))
        cash_equity = cash_equity + side * (close[-1] / max(entry_price, 1e-12) - 1.0) * abs(pos) - cost * abs(pos)
    return BacktestStats(
        pnl=float((cash_equity - 1.0) * 100.0),
        mdd=float(mdd * 100.0),
        trades=int(trades),
        wr=float(wins / trades if trades > 0 else 0.0),
        trades_per_day=float(trades / _days(df)),
        long_entries=int(long_entries),
        short_entries=int(short_entries),
        resize_events=int(resize_events),
        avg_notional=float(notional_sum / max(notional_samples, 1)),
        exits=exits,
        config=asdict(cfg),
    )


def _grid() -> list[MacroRiskConfig]:
    out: list[MacroRiskConfig] = [MacroRiskConfig(notional=2.5, take_profit=0.0)]
    for notional in (2.5, 3.0, 3.5, 4.0, 5.0):
        out.append(MacroRiskConfig(notional=notional, take_profit=0.0))
    for notional in (2.5, 3.0, 3.5, 4.0, 5.0):
      for tp in (0.75, 1.00, 1.10, 1.20, 1.30, 1.50):
        for lock in (0, 288, 864):
            out.append(MacroRiskConfig(notional=notional, take_profit=tp, lockout_bars=lock))
    for arm in (0.45, 0.60, 0.75, 1.00, 1.25, 1.50):
        for gap in (0.12, 0.18, 0.25, 0.35, 0.50, 0.70):
            if gap >= arm * 1.25:
                continue
            for notional in (2.5, 3.0, 3.5):
                for lock in (0, 288, 864):
                    out.append(MacroRiskConfig(notional=notional, take_profit=0.0, trail_arm=arm, trail_gap=gap, lockout_bars=lock))
    for stop in (0.12, 0.20, 0.30):
        out.append(MacroRiskConfig(notional=3.0, take_profit=1.20, stop_loss=stop))
    for vol_bars, vol_ref, q, floor in (
        (288, 6048, 0.60, 0.50),
        (288, 6048, 0.70, 0.50),
        (864, 6048, 0.60, 0.50),
        (864, 6048, 0.70, 0.65),
    ):
        out.append(MacroRiskConfig(notional=3.0, take_profit=1.20, vol_bars=vol_bars, vol_ref_bars=vol_ref, vol_q=q, vol_scale_floor=floor))
    return out


def _score(train: BacktestStats, val: BacktestStats) -> float:
    if train.pnl <= 0.0 or val.pnl <= 0.0 or train.trades < 1 or val.trades < 1:
        return -1e9
    margin_penalty = max(0.0, float(train.config.get("notional", 0.0)) - 3.0) * 12.0
    return float(val.pnl + 0.04 * train.pnl + 0.55 * val.mdd + 0.10 * train.mdd - 0.15 * val.resize_events - margin_penalty)


def main() -> None:
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    split = pd.Timestamp("2025-10-01")
    train_df = train_all[train_all["timestamp"] < split].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= split].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for cfg in _grid():
        tr = backtest(train_df, cfg)
        va = backtest(val_df, cfg)
        ev = backtest(eval_df, cfg)
        rows.append(
            {
                "score": _score(tr, va),
                "config": asdict(cfg),
                "train": asdict(tr),
                "validation": asdict(va),
                "eval": asdict(ev),
            }
        )
    rows_by_selection = sorted(rows, key=lambda x: x["score"], reverse=True)
    rows_by_eval = sorted(rows, key=lambda x: (x["eval"]["pnl"], x["eval"]["mdd"]), reverse=True)
    base = next(r for r in rows if r["config"]["notional"] == 2.5 and r["config"]["take_profit"] == 0.0)
    validation_eligible = [
        r for r in rows_by_selection
        if r["train"]["pnl"] > 0.0
        and r["validation"]["pnl"] > 0.0
        and r["train"]["mdd"] >= -40.0
        and r["validation"]["mdd"] >= -40.0
        and r["config"]["notional"] <= 3.0
    ]
    eval_eligible = [
        r for r in rows_by_selection
        if r["eval"]["pnl"] >= 100.0
        and r["eval"]["mdd"] > base["eval"]["mdd"]
        and r["validation"]["pnl"] > 0.0
    ]
    selected = validation_eligible[0] if validation_eligible else rows_by_selection[0]
    report = {
        "type": "leakfree_macro_trend_risk_overlay_fixed_notional_search",
        "train_csv": str(TRAIN_CSV.relative_to(ROOT)),
        "eval_csv": str(EVAL_CSV.relative_to(ROOT)),
        "accounting": "fixed notional exposure; same-side hold does not rebalance contracts every bar",
        "selection": "selected uses 2025 train/validation only with notional<=3.0 and MDD guardrails; 2026 eval is reported after selection",
        "base": base,
        "selected": selected,
        "top_by_selection": rows_by_selection[:25],
        "top_validation_eligible": validation_eligible[:25],
        "top_eval_100_mdd_improved": eval_eligible[:25],
        "top_by_eval": rows_by_eval[:25],
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    print(json.dumps({"report": str(REPORT), "base_eval": base["eval"], "selected": selected}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
