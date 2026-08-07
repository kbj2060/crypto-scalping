#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402


MODEL_ID = "hf_v13_clean_regime_trend_profit_lock_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_validation_selected_exposure_20260511/v13_clean_regime_validation_selected_exposure.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_trend_profit_lock_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_trend_profit_lock_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_trend_profit_lock_20260511_audit.json"


@dataclass(frozen=True)
class LockConfig:
    long_trend_bias_floor: float
    max_notional: float
    profit_lock_equity: float


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _grid() -> list[LockConfig]:
    return [
        LockConfig(float(thr), float(cap), float(target))
        for thr in (0.00, 0.05, 0.10, 0.15, 0.20)
        for cap in (1.85, 1.90, 1.95, 2.00, 2.10, 2.20)
        for target in (1.95, 2.00, 2.02, 2.05, 2.10)
    ]


def backtest(df: pd.DataFrame, bundle: dict[str, Any], cfg: LockConfig, *, fee: float, slip: float, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
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
    cooldown = 0
    next_cooldown = 0
    locked = False
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    blocks: dict[str, int] = {}
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if eq >= cfg.profit_lock_equity and pos == 0:
            locked = True
        if pos != 0:
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            elif eq >= cfg.profit_lock_equity:
                reason = "profit_lock_exit"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]), "exit_reason": reason, "trade_pnl_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "cash_after": float(cash)})
                    ledger.append(out)
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if locked:
            blocks["profit_locked"] = blocks.get("profit_locked", 0) + 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        row = df.iloc[i]
        if int(dec.side) > 0 and float(row.get("clean_regime_2024_unsup_v4_trend_bias", 0.0) or 0.0) < cfg.long_trend_bias_floor:
            blocks["long_trend_bias_floor"] = blocks.get("long_trend_bias_floor", 0) + 1
            continue
        fill_idx = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = min(float(dec.notional_exposure), float(cfg.max_notional))
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
        if record:
            open_record = {
                "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                "side": "LONG" if pos > 0 else "SHORT",
                "entry_price": float(entry_price),
                "notional_exposure": float(notional),
                "leverage": float(leverage),
                "position_fraction": float(notional / max(leverage, 1e-12)),
                "quality_score": float(dec.quality_score),
                "confidence": float(dec.confidence),
                "long_trend_bias_floor": float(cfg.long_trend_bias_floor),
                "profit_lock_equity": float(cfg.profit_lock_equity),
            }
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "block_reason_counts": blocks,
        "exits": exits,
    }
    if record:
        out["ledger"] = ledger
    return out


def _score(r: dict[str, Any]) -> float:
    pnl = float(r["pnl"])
    mdd = abs(float(r["mdd"]))
    if int(r["trades"]) < 20:
        return -1e9 + pnl
    return float(pnl - 2.0 * mdd - max(0.0, 100.0 - pnl) * 5.0 - max(0.0, mdd - 15.0) * 15.0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select trend-gate/profit-lock on validation, then fixed OOS.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    cfg = dict(bundle["config"])
    train = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train[train["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train, eval_df, list(bundle.get("feature_cols") or []))
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for lock_cfg in _grid():
        r = backtest(val, bundle, lock_cfg, fee=float(cfg["fee"]), slip=float(cfg["slip"]), decisions=val_dec)
        row = {"config": asdict(lock_cfg), "validation_cost1": r, "selection_score": _score(r)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no selected config")
    selected = LockConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest(eval_df, bundle, selected, fee=float(cfg["fee"]) * mult, slip=float(cfg["slip"]) * mult, decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("ledger", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v13_clean_regime_trend_profit_lock_manifest.pkl"
    joblib.dump({"model_id": MODEL_ID, "base_model": str(args.model), "config": asdict(selected), "selection_policy": "Selected on 2025 Oct-Dec validation only; 2026 fixed OOS not used for selection."}, model_path)
    grid_path = args.report_out.with_name(args.report_out.stem + "_validation_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] < 100.0:
        warnings.append("oos_pnl_below_100pct")
    if metrics["cost1"]["mdd"] < -15.0:
        warnings.append("oos_mdd_target_not_met")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote_candidate" if not blocking and metrics["cost1"]["pnl"] >= 100.0 and metrics["cost1"]["mdd"] >= -15.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed, used only after config selection",
        "selected_config": asdict(selected),
        "feature_audit": feature_audit,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Validation-selected LONG clean-trend gate plus max-notional cap plus profit-lock governor.",
        "base_model": str(args.model),
        "model": str(model_path),
        "split_policy": "Config selected on 2025 Oct-Dec validation only; 2026 fixed OOS not used for selection.",
        "selected_config": asdict(selected),
        "selection_score": best["selection_score"],
        "selection_result": {k: v for k, v in best.items() if k != "selection_score"},
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "validation_grid": str(grid_path), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected_config": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
