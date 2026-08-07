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


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_v4_balanced_h144.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_entry_overlay_grid_hf_v4_2026.json"


def _read(path: Path, only_ts: bool = False) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp"] if only_ts else None)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _close(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)


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


def _audit(train_csv: Path, eval_csv: Path, policy: dict[str, Any]) -> dict[str, Any]:
    train_raw = pd.read_csv(train_csv, usecols=["timestamp"])
    eval_raw = pd.read_csv(eval_csv, usecols=["timestamp"])
    t1_raw = pd.to_datetime(train_raw["timestamp"], errors="coerce")
    t2_raw = pd.to_datetime(eval_raw["timestamp"], errors="coerce")
    t1 = t1_raw.dropna()
    t2 = t2_raw.dropna()
    overlap = set(t1.astype("int64").tolist()) & set(t2.astype("int64").tolist())
    return {
        "train_rows": int(len(train_raw)),
        "eval_rows": int(len(eval_raw)),
        "train_valid_timestamp_rows": int(len(t1)),
        "eval_valid_timestamp_rows": int(len(t2)),
        "train_range": [str(t1.min()), str(t1.max())],
        "eval_range": [str(t2.min()), str(t2.max())],
        "timestamp_overlap_rows": int(len(overlap)),
        "train_duplicate_timestamps": int(t1_raw.duplicated().sum()),
        "eval_duplicate_timestamps": int(t2_raw.duplicated().sum()),
        "policy_feature_count": int(len(policy.get("feature_cols", []))),
        "label_distribution": policy.get("label_distribution", {}),
    }


def _decisions(df: pd.DataFrame, policy: dict[str, Any]) -> pd.DataFrame:
    close = _close(df)
    feat = prepare_features(df, side_hint=0, close=close)
    return predict_policy_frame(policy, feat)


def _quality_scaled_decisions(dec: pd.DataFrame, *, notional_mult: float, max_notional: float, quality_floor: float, confidence_floor: float) -> pd.DataFrame:
    out = dec.copy()
    q = pd.to_numeric(out["quality_score"], errors="coerce").fillna(0.0)
    c = pd.to_numeric(out["confidence"], errors="coerce").fillna(0.0)
    active = (out["action"].astype(int) != ACTION_CASH) & (q >= float(quality_floor)) & (c >= float(confidence_floor))
    blocked = (out["action"].astype(int) != ACTION_CASH) & ~active
    out.loc[blocked, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[blocked, "leverage"] = 1.0
    out.loc[active, "notional_exposure"] = np.minimum(pd.to_numeric(out.loc[active, "notional_exposure"], errors="coerce") * float(notional_mult), float(max_notional))
    out.loc[active, "position_fraction"] = out.loc[active, "notional_exposure"] / np.maximum(pd.to_numeric(out.loc[active, "leverage"], errors="coerce"), 1e-12)
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
    cooldown_left = 0
    next_cooldown = 0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    action_counts = {"cash": 0, "long": 0, "short": 0}

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

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            age = i - entry_idx
            reason = ""
            if take_profit > 0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and age >= max_hold:
                reason = "max_hold"
            if reason:
                exit_price = _fill_price(df, min(i + 1, len(df) - 1), pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
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
            d = decisions.iloc[i]
            if int(d.action) == ACTION_CASH or int(d.side) == 0:
                action_counts["cash"] += 1
                continue
            pos = int(d.side)
            action_counts["long" if int(d.action) == ACTION_LONG else "short"] += 1
            entry_price = _fill_price(df, min(i + 1, len(df) - 1), pos, slip, entry=True)
            entry_equity = cash
            entry_idx = i
            notional = float(d.notional_exposure)
            leverage = float(d.leverage)
            take_profit = float(d.take_profit)
            stop_loss = float(d.stop_loss)
            max_hold = int(d.max_hold_bars)
            next_cooldown = int(d.cooldown_bars)
            cash -= cash * fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
    if pos != 0:
        exit_price = _fill_price(df, len(df) - 1, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
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
        "action_counts": action_counts,
        "exits": exits,
    }


def _configs() -> list[dict[str, float]]:
    out = []
    for mult in (1.0, 1.5, 2.0, 2.75, 3.5, 4.5):
        for q in (-1.0, 0.0, 0.01, 0.02, 0.035):
            for conf in (0.0, 0.38, 0.45):
                out.append({"notional_mult": mult, "quality_floor": q, "confidence_floor": conf, "max_notional": 3.60})
    return out


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {k: bt.get(k) for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate HF entry notional/quality overlays with audit and stress.")
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
    rows = []
    for cfg in _configs():
        dec = _quality_scaled_decisions(base_dec, **cfg)
        bt = backtest_decisions(eval_df, dec, fee=float(args.fee), slip=float(args.slip))
        rows.append({"name": f"m{cfg['notional_mult']}_q{cfg['quality_floor']}_c{cfg['confidence_floor']}", "config": cfg, "eval": _compact(bt)})
    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    ranked_goal = [
        r for r in ranked
        if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 25.0
    ]
    stress_modes = ["normal", "all_ai_zero", "patchtst_zero", "tide_zero", "dlinear_zero"]
    stress = {}
    top_cfgs = [r["config"] for r in ranked_goal[:3]]
    for mode in stress_modes:
        df, meta = _stress_frame(eval_df, mode)
        dec0 = _decisions(df, policy)
        stress[mode] = {"stress": meta, "results": []}
        for cfg in top_cfgs:
            dec = _quality_scaled_decisions(dec0, **cfg)
            stress[mode]["results"].append({"config": cfg, "eval": _compact(backtest_decisions(df, dec, fee=float(args.fee), slip=float(args.slip)))})
    report = {
        "type": "hf_entry_overlay_grid_hf_v4_2026",
        "policy": str(args.policy),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "grid": rows,
        "ranked_by_pnl": [{"name": r["name"], **r["eval"]} for r in ranked[:20]],
        "ranked_goal_5_to_25_trades_per_day": [{"name": r["name"], **r["eval"]} for r in ranked_goal[:20]],
        "stress": stress,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "top": report["ranked_goal_5_to_25_trades_per_day"][:8], "stress": stress}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
