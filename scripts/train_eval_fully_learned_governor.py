#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    FullyLearnedGovernorConfig,
    build_training_set,
    prepare_features,
    predict_policy_frame,
    train_policy,
)


DEFAULT_TRAIN_CSV = ROOT / "tmp/pipeline_audit_causal_regime/trade_candidates_2025_causal_regime.csv"
DEFAULT_EVAL_CSV = ROOT / "data/ensemble/event_driven/trade_candidates_2026_causal_regime_predicted_45m_hgb_telemetry.csv"
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/fully_learned_governor_policy_v1.pkl"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/fully_learned_governor_policy_v1_2026.json"


def _float_tuple(value: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    if value is None or str(value).strip() == "":
        return default
    return tuple(float(x.strip()) for x in str(value).split(",") if x.strip())


def _int_tuple(value: str | None, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None or str(value).strip() == "":
        return default
    return tuple(int(x.strip()) for x in str(value).split(",") if x.strip())


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


def backtest_policy(df: pd.DataFrame, bundle: dict[str, Any], *, fee: float, slip: float, record_trades: bool = False) -> dict[str, Any]:
    close = _close(df)
    feat = prepare_features(df, side_hint=0, close=close)
    decisions = predict_policy_frame(bundle, feat)
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
    peak_unrealized = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    notional_sum = 0.0
    leverage_sum = 0.0
    tp_sum = 0.0
    sl_sum = 0.0
    hold_sum = 0.0
    records: list[dict[str, Any]] = []

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

    open_record: dict[str, Any] | None = None
    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                if pos > 0:
                    raw = (exit_price - entry_price) / max(entry_price, 1e-12)
                else:
                    raw = (entry_price - exit_price) / max(entry_price, 1e-12)
                realized = raw * notional
                before = cash
                cash = cash * (1.0 + realized)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if open_record is not None and record_trades:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_idx": int(i),
                            "exit_timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "peak_unrealized_pct": float(peak_unrealized * 100.0),
                        }
                    )
                    records.append(out)
                pos = 0
                entry_price = 0.0
                notional = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                peak_unrealized = 0.0
                open_record = None
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
            action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
            fill_idx = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
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
            tp_sum += take_profit
            sl_sum += stop_loss
            hold_sum += max_hold
            if record_trades:
                open_record = {
                    "entry_idx": int(i),
                    "entry_timestamp": str(df["timestamp"].iloc[i]) if "timestamp" in df.columns else "",
                    "side": "LONG" if pos > 0 else "SHORT",
                    "notional_exposure": float(notional),
                    "leverage": float(leverage),
                    "position_fraction": float(dec.position_fraction),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                    "cooldown_bars": int(next_cooldown),
                    "quality_score": float(dec.quality_score),
                    "confidence": float(dec.confidence),
                }
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        if pos > 0:
            raw = (exit_price - entry_price) / max(entry_price, 1e-12)
        else:
            raw = (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        if open_record is not None and record_trades:
            out_record = dict(open_record)
            out_record.update(
                {
                    "exit_idx": int(fill_idx),
                    "exit_timestamp": str(df["timestamp"].iloc[fill_idx]) if "timestamp" in df.columns else "",
                    "exit_reason": "forced_end",
                    "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                    "peak_unrealized_pct": float(peak_unrealized * 100.0),
                }
            )
            records.append(out_record)
    pnl = (cash - 1.0) * 100.0
    n_entries = max(long_entries + short_entries, 1)
    out = {
        "pnl": float(pnl),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "avg_take_profit": float(tp_sum / n_entries),
        "avg_stop_loss": float(sl_sum / n_entries),
        "avg_max_hold_bars": float(hold_sum / n_entries),
        "action_counts": action_counts,
        "exits": exits,
    }
    if record_trades:
        out["trade_records"] = records
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate fully learned governor policy.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--stride-bars", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=384)
    p.add_argument("--max-train-horizon-bars", type=int, default=864)
    p.add_argument("--adverse-penalty", type=float, default=0.85)
    p.add_argument("--size-penalty", type=float, default=0.018)
    p.add_argument("--hold-penalty", type=float, default=0.004)
    p.add_argument("--turnover-bonus", type=float, default=0.0015)
    p.add_argument("--cash-score", type=float, default=0.0)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--notional-buckets", type=str, default=None)
    p.add_argument("--leverage-buckets", type=str, default=None)
    p.add_argument("--take-profit-buckets", type=str, default=None)
    p.add_argument("--stop-loss-buckets", type=str, default=None)
    p.add_argument("--max-hold-buckets", type=str, default=None)
    p.add_argument("--cooldown-buckets", type=str, default=None)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--record-trades", action="store_true", default=False)
    p.add_argument("--trade-records-out", type=Path, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp("2025-10-01")
    if "timestamp" in train_all.columns:
        train_df = train_all[train_all["timestamp"] < split_ts].reset_index(drop=True)
        val_df = train_all[train_all["timestamp"] >= split_ts].reset_index(drop=True)
    else:
        cut = int(len(train_all) * 0.75)
        train_df, val_df = train_all.iloc[:cut].reset_index(drop=True), train_all.iloc[cut:].reset_index(drop=True)
    cfg = FullyLearnedGovernorConfig(
        notional_buckets=_float_tuple(args.notional_buckets, FullyLearnedGovernorConfig.notional_buckets),
        leverage_buckets=_float_tuple(args.leverage_buckets, FullyLearnedGovernorConfig.leverage_buckets),
        take_profit_buckets=_float_tuple(args.take_profit_buckets, FullyLearnedGovernorConfig.take_profit_buckets),
        stop_loss_buckets=_float_tuple(args.stop_loss_buckets, FullyLearnedGovernorConfig.stop_loss_buckets),
        max_hold_buckets=_int_tuple(args.max_hold_buckets, FullyLearnedGovernorConfig.max_hold_buckets),
        cooldown_buckets=_int_tuple(args.cooldown_buckets, FullyLearnedGovernorConfig.cooldown_buckets),
        max_train_horizon_bars=int(args.max_train_horizon_bars),
        adverse_penalty=float(args.adverse_penalty),
        size_penalty=float(args.size_penalty),
        hold_penalty=float(args.hold_penalty),
        turnover_bonus=float(args.turnover_bonus),
        cash_score=float(args.cash_score),
        fee=float(args.fee),
        slip=float(args.slip),
    )
    x, y, meta = build_training_set(
        train_df,
        cfg=cfg,
        stride_bars=int(args.stride_bars),
        batch_size=int(args.batch_size),
    )
    bundle = train_policy(x, y, cfg=cfg, random_state=int(args.random_state))
    bundle["train_csv"] = str(args.train_csv)
    bundle["training_meta"] = meta
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)
    train_bt = backtest_policy(train_df, bundle, fee=float(args.fee), slip=float(args.slip))
    val_bt = backtest_policy(val_df, bundle, fee=float(args.fee), slip=float(args.slip))
    eval_bt = backtest_policy(eval_df, bundle, fee=float(args.fee), slip=float(args.slip), record_trades=bool(args.record_trades))
    if bool(args.record_trades) and args.trade_records_out is not None:
        args.trade_records_out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(eval_bt.get("trade_records", [])).to_csv(args.trade_records_out, index=False)
    report = {
        "type": "fully_learned_governor_policy_v1",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_out": str(args.model_out),
        "config": asdict(cfg),
        "training_meta": meta,
        "label_distribution": bundle.get("label_distribution", {}),
        "train": train_bt,
        "validation": val_bt,
        "eval": eval_bt,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": str(args.model_out), "report": str(args.report_out), "train": train_bt, "validation": val_bt, "eval": {k: v for k, v in eval_bt.items() if k != "trade_records"}}, ensure_ascii=False))


if __name__ == "__main__":
    main()
