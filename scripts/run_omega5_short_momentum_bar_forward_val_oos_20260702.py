#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_omega5_live_only_shadow_loop_20260702 import (  # noqa: E402
    Candidate,
    finite_float,
    side_for_candidate,
)


CANDIDATE = Candidate(
    "omega5_live_short_momentum_v2",
    max_hold_minutes=25,
    take_profit=0.0045,
    stop_loss=0.0030,
    notional=1.0,
)
ROUNDTRIP_COST_PER_NOTIONAL = 0.0006

DEFAULT_VAL_FEATURES = ROOT / "data/splits/year_oos/training_features_2025.csv"
DEFAULT_OOS_FEATURES = (
    ROOT / "tmp/causal_regen_20260516/extended_oos_20260702/"
    "training_features_2026_0101_0702_m7_ai_for_omega5_parity.csv"
)
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/omega5_short_momentum_v2_bar_forward_val_oos_20260702"

REQUIRED_COLUMNS = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "rsi",
    "log_return",
    "btc_ret_1",
    "net_taker_ratio",
    "taker_acceleration",
    "oi_change_rate",
    "compression_release_down",
    "upper_wick_z",
)
OPTIONAL_COLUMNS = (
    "m7_prob_up",
    "m7_prob_dn",
    "m7_confidence",
    "jump_z",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "hma_slope",
    "bb_width",
    "smart_money_flow",
    "cvd_slope_12",
    "compression_release_up",
    "lower_wick_z",
)


def json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        return obj
    raise TypeError(type(obj).__name__)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def available_usecols(path: Path) -> list[str]:
    cols = list(pd.read_csv(path, nrows=0).columns)
    missing = [col for col in REQUIRED_COLUMNS if col not in cols]
    if missing:
        raise RuntimeError(f"{path} missing required columns: {missing}")
    return [col for col in (*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS) if col in cols]


def load_split(path: Path, start: str, end: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    usecols = available_usecols(path)
    df = pd.read_csv(path, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    sliced = df[(df["timestamp"] >= start_ts) & (df["timestamp"] < end_ts)].copy().reset_index(drop=True)
    if sliced.empty:
        raise RuntimeError(f"{path} produced empty split for {start}..{end}")
    null_counts = sliced[list(c for c in REQUIRED_COLUMNS if c != "timestamp")].isna().sum()
    null_counts = null_counts[null_counts > 0].sort_values(ascending=False)
    return sliced, {
        "path": str(path),
        "source_rows": int(len(df)),
        "split_rows": int(len(sliced)),
        "source_start": str(df["timestamp"].min()),
        "source_end": str(df["timestamp"].max()),
        "split_start": str(sliced["timestamp"].min()),
        "split_end": str(sliced["timestamp"].max()),
        "columns_used": usecols,
        "required_null_counts": {str(k): int(v) for k, v in null_counts.items()},
    }


def bar_minutes(prev_ts: pd.Timestamp | None, ts: pd.Timestamp) -> float:
    if prev_ts is None:
        return 5.0
    minutes = (ts - prev_ts).total_seconds() / 60.0
    if not math.isfinite(minutes) or minutes <= 0.0:
        return 5.0
    return minutes


def close_position(position: dict[str, Any], row: dict[str, Any], exit_ts: pd.Timestamp, reason: str, exit_price: float) -> dict[str, Any]:
    entry_price = float(position["entry_price"])
    side = int(position["side"])
    raw_move = side * (float(exit_price) / entry_price - 1.0)
    net_account_pnl = raw_move * float(position["notional"]) - ROUNDTRIP_COST_PER_NOTIONAL * float(position["notional"])
    return {
        "candidate": CANDIDATE.name,
        "entry_timestamp": position["entry_timestamp"].isoformat(),
        "exit_timestamp": exit_ts.isoformat(),
        "side": side,
        "entry_price": entry_price,
        "exit_price": float(exit_price),
        "reason": reason,
        "raw_price_move": raw_move,
        "net_account_pnl": net_account_pnl,
        "notional": float(position["notional"]),
        "hold_minutes": (exit_ts - position["entry_timestamp"]).total_seconds() / 60.0,
        "entry_features": dict(position.get("entry_features") or {}),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }


def entry_features(row: dict[str, Any]) -> dict[str, float]:
    keys = (
        "rsi",
        "log_return",
        "btc_ret_1",
        "net_taker_ratio",
        "taker_acceleration",
        "oi_change_rate",
        "compression_release_down",
        "upper_wick_z",
        "jump_z",
        "close",
    )
    return {key: finite_float(row, key) for key in keys}


def run_walk_forward(frame: pd.DataFrame, split_name: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    ledger: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    position: dict[str, Any] | None = None
    prev_ts: pd.Timestamp | None = None

    for idx, row_obj in enumerate(frame.to_dict("records")):
        ts = pd.Timestamp(row_obj["timestamp"])
        minutes = bar_minutes(prev_ts, ts)
        prev_ts = ts

        if position is not None:
            high = finite_float(row_obj, "high")
            low = finite_float(row_obj, "low")
            close = finite_float(row_obj, "close")
            entry = float(position["entry_price"])
            side = int(position["side"])
            favorable = (high / entry - 1.0) if side > 0 else (entry / max(low, 1.0e-12) - 1.0)
            adverse = (low / entry - 1.0) if side > 0 else (entry / max(high, 1.0e-12) - 1.0)
            hold_minutes = (ts - position["entry_timestamp"]).total_seconds() / 60.0
            resolved = None
            if favorable >= CANDIDATE.take_profit:
                exit_price = entry * (1.0 + side * CANDIDATE.take_profit)
                resolved = close_position(position, row_obj, ts, "take_profit", exit_price)
            elif adverse <= -CANDIDATE.stop_loss:
                exit_price = entry * (1.0 - side * CANDIDATE.stop_loss)
                resolved = close_position(position, row_obj, ts, "stop_loss", exit_price)
            elif hold_minutes >= CANDIDATE.max_hold_minutes:
                resolved = close_position(position, row_obj, ts, "time_exit", close)
            if resolved is not None:
                ledger.append(resolved)
                position = None

        side = 0 if position is not None else side_for_candidate(CANDIDATE, row_obj)
        would_enter = bool(side != 0 and position is None)
        decisions.append(
            {
                "split": split_name,
                "row_index": int(idx),
                "timestamp": ts.isoformat(),
                "side": int(side),
                "would_enter": would_enter,
                "in_position_after_decision": bool(would_enter or position is not None),
                "close": finite_float(row_obj, "close"),
                "fresh_forward_bar_by_bar": True,
                "future_rows_used_for_entry": False,
            }
        )
        if would_enter:
            position = {
                "entry_timestamp": ts,
                "entry_price": finite_float(row_obj, "close"),
                "side": int(side),
                "notional": CANDIDATE.notional,
                "entry_features": entry_features(row_obj),
            }

    open_position = {}
    if position is not None:
        open_position = {
            "entry_timestamp": position["entry_timestamp"].isoformat(),
            "entry_price": float(position["entry_price"]),
            "side": int(position["side"]),
            "notional": float(position["notional"]),
            "note": "not force-closed at split end",
        }

    return pd.DataFrame(ledger), pd.DataFrame(decisions), {
        "open_position": open_position,
        "bar_count": int(len(frame)),
        "approx_bar_minutes": float(minutes if len(frame) else 0.0),
    }


def summarize_ledger(ledger: pd.DataFrame, decisions: pd.DataFrame, split_extra: dict[str, Any]) -> dict[str, Any]:
    if ledger.empty:
        returns = []
    else:
        returns = [float(x) for x in ledger["net_account_pnl"].tolist()]
    curve = []
    running = 0.0
    for value in returns:
        running += value
        curve.append(running)
    peak = 0.0
    mdd = 0.0
    for value in curve:
        peak = max(peak, value)
        mdd = min(mdd, value - peak)
    compound_curve = []
    equity = 1.0
    for value in returns:
        equity *= 1.0 + value
        compound_curve.append(equity)
    compound_peak = 1.0
    compound_mdd = 0.0
    for value in compound_curve:
        compound_peak = max(compound_peak, value)
        compound_mdd = min(compound_mdd, value / max(compound_peak, 1.0e-12) - 1.0)
    wins = sum(1 for value in returns if value > 0.0)
    return {
        "pnl": float(sum(returns)),
        "pnl_pct": float(sum(returns) * 100.0),
        "mdd": float(mdd),
        "mdd_pct": float(mdd * 100.0),
        "compound_pnl": float(equity - 1.0),
        "compound_pnl_pct": float((equity - 1.0) * 100.0),
        "compound_mdd": float(compound_mdd),
        "compound_mdd_pct": float(compound_mdd * 100.0),
        "trades": int(len(returns)),
        "wins": int(wins),
        "wr": float(wins / len(returns)) if returns else None,
        "avg_hold_minutes": float(ledger["hold_minutes"].mean()) if not ledger.empty else 0.0,
        "max_hold_minutes": float(ledger["hold_minutes"].max()) if not ledger.empty else 0.0,
        "reason_counts": dict(Counter(ledger["reason"].astype(str))) if not ledger.empty else {},
        "decision_rows": int(len(decisions)),
        "entry_rows": int(decisions["would_enter"].sum()) if not decisions.empty else 0,
        **split_extra,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    val_df, val_source = load_split(Path(args.validation_features), args.validation_start, args.validation_end)
    oos_df, oos_source = load_split(Path(args.oos_features), args.oos_start, args.oos_end)

    val_ledger, val_decisions, val_extra = run_walk_forward(val_df, "validation")
    oos_ledger, oos_decisions, oos_extra = run_walk_forward(oos_df, "oos")

    val_ledger.to_csv(out_dir / "validation_bar_forward_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "oos_bar_forward_ledger.csv", index=False)
    val_decisions.to_csv(out_dir / "validation_bar_forward_decisions.csv", index=False)
    oos_decisions.to_csv(out_dir / "oos_bar_forward_decisions.csv", index=False)

    report = {
        "schema_version": "omega5.short_momentum_v2.bar_forward_val_oos_report.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "candidate": CANDIDATE.name,
        "fresh_forward_definition": "fixed historical validation/OOS split, causal 5m bar-by-bar walk-forward",
        "fresh_forward_bar_by_bar": True,
        "feature_frame_replay_only": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "candidate_config": {
            "side": "short_only",
            "take_profit_price_move": CANDIDATE.take_profit,
            "stop_loss_price_move": CANDIDATE.stop_loss,
            "max_hold_minutes": CANDIDATE.max_hold_minutes,
            "notional": CANDIDATE.notional,
            "roundtrip_cost_per_notional": ROUNDTRIP_COST_PER_NOTIONAL,
            "execution_mode": "single_position_no_pyramiding",
            "same_bar_bracket_ambiguity": "take_profit_first_matches_existing_shadow_runner",
        },
        "splits": {
            "validation": {
                "start": args.validation_start,
                "end_exclusive": args.validation_end,
                "source": val_source,
            },
            "oos": {
                "start": args.oos_start,
                "end_exclusive": args.oos_end,
                "source": oos_source,
            },
        },
        "metrics": {
            "validation": summarize_ledger(val_ledger, val_decisions, val_extra),
            "oos": summarize_ledger(oos_ledger, oos_decisions, oos_extra),
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "validation_ledger": str(out_dir / "validation_bar_forward_ledger.csv"),
            "oos_ledger": str(out_dir / "oos_bar_forward_ledger.csv"),
            "validation_decisions": str(out_dir / "validation_bar_forward_decisions.csv"),
            "oos_decisions": str(out_dir / "oos_bar_forward_decisions.csv"),
        },
    }
    write_json(out_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-features", default=str(DEFAULT_VAL_FEATURES))
    parser.add_argument("--oos-features", default=str(DEFAULT_OOS_FEATURES))
    parser.add_argument("--validation-start", default="2025-09-01 00:00:00")
    parser.add_argument("--validation-end", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-start", default="2026-01-01 00:00:00")
    parser.add_argument("--oos-end", default="2026-04-01 00:00:00")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    args = parser.parse_args()
    report = run(args)
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
