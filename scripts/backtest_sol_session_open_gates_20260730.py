#!/usr/bin/env python3
"""Research-only causal test of a TradFi-session-open entry and exit gate on SOL.

The entry signal is decided only after the first 30 minutes of an actual LSE,
NYSE, or JPX cash-session open have closed.  It requires a break of the prior
30-minute range, unusual volume, and taker-flow confirmation.  The trade is
filled at the next bar open.  Exits are TP/SL barriers or a fixed one-hour
time gate.  Parameters are selected on the validation window only and frozen
before the fixed OOS evaluation.

This is intentionally not connected to the live bot or any model artifact.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/splits/year_oos/sol_raw_frame_2024_2026.csv"
OUT_DIR = ROOT / "data/research/session_open_gates_20260730"
VAL_START, VAL_END = pd.Timestamp("2025-09-01", tz="UTC"), pd.Timestamp("2025-12-31 23:59:59", tz="UTC")
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
ONE_WAY_COST_BPS = 5.0


@dataclass(frozen=True)
class GateConfig:
    volume_multiple: float
    take_profit_atr: float
    stop_loss_atr: float
    max_hold_bars: int = 12


def _load() -> pd.DataFrame:
    cols = ["timestamp", "open", "high", "low", "close", "volume", "quote_volume", "taker_buy_quote"]
    df = pd.read_csv(RAW, usecols=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if not df["timestamp"].is_monotonic_increasing:
        raise RuntimeError("timestamp ordering contract failed")
    for col in cols[1:]:
        df[col] = pd.to_numeric(df[col], errors="raise")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise RuntimeError("price contract failed")
    if (df[["volume", "quote_volume"]] < 0).any().any():
        raise RuntimeError("volume contract failed")
    return df


def _open_confirmation_times(df: pd.DataFrame) -> dict[pd.Timestamp, str]:
    """Return each actual cash-market open + 30 minute confirmation timestamp."""
    first = df["timestamp"].iloc[0].date()
    last = df["timestamp"].iloc[-1].date()
    out: dict[pd.Timestamp, str] = {}
    for name, calendar_name in (("europe", "LSE"), ("us", "NYSE"), ("japan", "JPX")):
        schedule = mcal.get_calendar(calendar_name).schedule(start_date=first, end_date=last)
        for opened in schedule["market_open"]:
            confirm_at = pd.Timestamp(opened).tz_convert("UTC") + pd.Timedelta(minutes=30)
            if confirm_at in out:
                raise RuntimeError(f"duplicate session confirmation time: {confirm_at}")
            out[confirm_at] = name
    return out


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    prev_close = out["close"].shift(1)
    true_range = pd.concat(
        [(out["high"] - out["low"]), (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    # All rolling quantities are shifted: an entry gate uses only bars closed before the signal bar.
    out["atr_12"] = true_range.rolling(12, min_periods=12).mean().shift(1)
    out["volume_mean_288"] = out["volume"].rolling(288, min_periods=72).mean().shift(1)
    out["pre_high"] = out["high"].rolling(6, min_periods=6).max().shift(1)
    out["pre_low"] = out["low"].rolling(6, min_periods=6).min().shift(1)
    out["taker_buy_fraction"] = (out["taker_buy_quote"] / out["quote_volume"].replace(0.0, np.nan)).clip(0.0, 1.0)
    out["session"] = out["timestamp"].map(_open_confirmation_times(out))
    return out


def _metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"trades": 0, "pnl_pct": 0.0, "mdd_pct": 0.0, "win_rate": 0.0, "profit_factor": 0.0}
    equity = (1.0 + ledger["net_return"]).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    gains = ledger.loc[ledger["net_return"] > 0.0, "net_return"].sum()
    losses = -ledger.loc[ledger["net_return"] < 0.0, "net_return"].sum()
    return {
        "trades": int(len(ledger)),
        "pnl_pct": float((equity.iloc[-1] - 1.0) * 100.0),
        "mdd_pct": float(drawdown.min() * 100.0),
        "win_rate": float((ledger["net_return"] > 0.0).mean()),
        "profit_factor": float(gains / losses) if losses > 0.0 else None,
        "median_hold_bars": float(ledger["hold_bars"].median()),
        "exit_reasons": {str(k): int(v) for k, v in ledger["exit_reason"].value_counts().items()},
    }


def _replay(frame: pd.DataFrame, cfg: GateConfig) -> pd.DataFrame:
    """Single causal walk. Signal at close[i], enter at open[i+1], no stored trade ledger."""
    rows: list[dict[str, Any]] = []
    pos = 0
    entry_i = -1
    entry_px = tp_px = sl_px = 0.0
    entry_session = ""
    cost = ONE_WAY_COST_BPS / 10_000.0

    for i in range(len(frame) - 1):
        bar = frame.iloc[i]
        if pos:
            # Conservative ordering removes an ambiguous intra-bar TP/SL benefit.
            hit_sl = (pos > 0 and bar["low"] <= sl_px) or (pos < 0 and bar["high"] >= sl_px)
            hit_tp = (pos > 0 and bar["high"] >= tp_px) or (pos < 0 and bar["low"] <= tp_px)
            exit_reason = ""
            if hit_sl:
                exit_px, exit_reason = sl_px, "stop_loss"
            elif hit_tp:
                exit_px, exit_reason = tp_px, "take_profit"
            elif i - entry_i >= cfg.max_hold_bars:
                exit_px, exit_reason = float(frame["open"].iloc[i + 1]), "time_gate"
            else:
                continue

            gross = pos * (exit_px / entry_px - 1.0)
            rows.append({
                "entry_timestamp": frame["timestamp"].iloc[entry_i].isoformat(),
                "exit_timestamp": frame["timestamp"].iloc[i].isoformat(),
                "session": entry_session,
                "side": "long" if pos > 0 else "short",
                "entry_price": entry_px,
                "exit_price": exit_px,
                "hold_bars": int(i - entry_i),
                "exit_reason": exit_reason,
                "gross_return": gross,
                "net_return": gross - 2.0 * cost,
            })
            pos = 0
            continue

        if pd.isna(bar["session"]) or any(pd.isna(bar[x]) for x in ("atr_12", "volume_mean_288", "pre_high", "pre_low")):
            continue
        volume_ok = float(bar["volume"]) >= cfg.volume_multiple * float(bar["volume_mean_288"])
        long_ok = float(bar["close"]) > float(bar["pre_high"]) and float(bar["taker_buy_fraction"]) >= 0.52
        short_ok = float(bar["close"]) < float(bar["pre_low"]) and float(bar["taker_buy_fraction"]) <= 0.48
        if not volume_ok or not (long_ok or short_ok):
            continue

        pos = 1 if long_ok else -1
        entry_i = i + 1
        entry_px = float(frame["open"].iloc[entry_i])
        atr_move = float(bar["atr_12"]) / entry_px
        tp_move = cfg.take_profit_atr * atr_move
        sl_move = cfg.stop_loss_atr * atr_move
        tp_px = entry_px * (1.0 + pos * tp_move)
        sl_px = entry_px * (1.0 - pos * sl_move)
        entry_session = str(bar["session"])
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = _prepare(_load())
    val = frame[(frame["timestamp"] >= VAL_START) & (frame["timestamp"] <= VAL_END)].reset_index(drop=True)
    oos = frame[(frame["timestamp"] >= OOS_START) & (frame["timestamp"] <= OOS_END)].reset_index(drop=True)
    if val.empty or oos.empty:
        raise RuntimeError("validation or OOS frame is empty")

    configs = [GateConfig(*values) for values in itertools.product((1.0, 1.25), (1.5, 2.0), (1.0, 1.25))]
    validation = []
    for cfg in configs:
        ledger = _replay(val, cfg)
        validation.append({"config": asdict(cfg), "metrics": _metrics(ledger)})
    eligible = [x for x in validation if x["metrics"]["trades"] >= 12]
    if not eligible:
        raise RuntimeError("no validation config reached 12 trades; do not select a sparse gate")
    selected = max(eligible, key=lambda x: (x["metrics"]["pnl_pct"], -abs(x["metrics"]["mdd_pct"])))
    selected_cfg = GateConfig(**selected["config"])
    val_ledger = _replay(val, selected_cfg)
    oos_ledger = _replay(oos, selected_cfg)
    val_ledger.to_csv(OUT_DIR / "validation_ledger.csv", index=False)
    oos_ledger.to_csv(OUT_DIR / "oos_ledger.csv", index=False)
    report = {
        "strategy": "session_open_compression_confirmed_expansion",
        "asset": "SOLUSDT",
        "data": str(RAW),
        "entry_gate": "actual LSE/NYSE/JPX open + 30m; prior-30m range break; volume >= rolling 24h mean multiple; taker-buy fraction confirms side",
        "exit_gate": "ATR-derived TP/SL, conservative SL-first same-bar resolution, or 12-bar time gate",
        "one_way_cost_bps": ONE_WAY_COST_BPS,
        "selection_window": [str(VAL_START), str(VAL_END)],
        "oos_window": [str(OOS_START), str(OOS_END)],
        "validation_grid": validation,
        "selected_config": asdict(selected_cfg),
        "selected_validation_metrics": _metrics(val_ledger),
        "frozen_oos_metrics": _metrics(oos_ledger),
        "audit": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "entry_timing": "signal on confirmation-bar close, fill next bar open",
            "calendar": "pandas_market_calendars LSE/NYSE/JPX; actual holidays and DST schedule",
            "promotion_eligible": False,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
