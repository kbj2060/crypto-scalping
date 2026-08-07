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

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, FullyLearnedGovernorConfig, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _read  # noqa: E402


MODEL_ID = "hf_v13_clean_regime_mdd_governor_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_validation_selected_exposure_20260511/v13_clean_regime_validation_selected_exposure.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_mdd_governor_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_mdd_governor_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_clean_regime_mdd_governor_20260511_audit.json"


@dataclass(frozen=True)
class GovernorConfig:
    dd_soft: float
    dd_hard: float
    dd_scale: float
    loss_streak_limit: int
    loss_cooldown_bars: int
    min_quality_after_loss: float
    risk_off_cap_after_loss: float
    stop_loss_scale: float
    max_notional: float


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


def _close(df: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(df["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=np.float64)


def _fill_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["open"], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _days(df: pd.DataFrame) -> float:
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _grid() -> list[GovernorConfig]:
    rows: list[GovernorConfig] = []
    for dd_soft, dd_hard, dd_scale in ((0.08, 0.15, 0.45), (0.10, 0.18, 0.55), (0.12, 0.22, 0.65)):
        for loss_limit, cooldown in ((1, 24), (2, 36), (2, 72)):
            for stop_scale in (0.55, 0.70, 0.85):
                rows.append(
                    GovernorConfig(
                        dd_soft=dd_soft,
                        dd_hard=dd_hard,
                        dd_scale=dd_scale,
                        loss_streak_limit=loss_limit,
                        loss_cooldown_bars=cooldown,
                        min_quality_after_loss=0.070,
                        risk_off_cap_after_loss=0.78,
                        stop_loss_scale=stop_scale,
                        max_notional=2.20,
                    )
                )
    return rows


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    gov: GovernorConfig,
    *,
    fee: float,
    slip: float,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
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
    next_cooldown = 0
    cooldown_left = 0
    gov_cooldown = 0
    loss_streak = 0
    peak_unrealized = 0.0
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
        cur_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            hold_bars = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "governed_stop_loss"
            elif max_hold > 0 and hold_bars >= max_hold:
                reason = "learned_max_hold"
            elif cur_dd >= gov.dd_hard:
                reason = "dd_hard_exit"
            if reason:
                fill_idx = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_idx, pos, slip, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee * notional
                trades += 1
                won = cash > entry_equity
                wins += int(won)
                loss_streak = 0 if won else loss_streak + 1
                if loss_streak >= gov.loss_streak_limit:
                    gov_cooldown = max(gov_cooldown, int(gov.loss_cooldown_bars))
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                            "exit_reason": reason,
                            "trade_pnl_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "cash_after": float(cash),
                            "drawdown_at_exit": float(cur_dd),
                            "peak_unrealized_pct": float(peak_unrealized * 100.0),
                        }
                    )
                    ledger.append(out)
                pos = 0
                notional = 0.0
                leverage = 1.0
                cooldown_left = int(next_cooldown)
                next_cooldown = 0
                peak_unrealized = 0.0
                open_record = None
                continue

        if pos != 0:
            continue
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
        if gov_cooldown > 0:
            gov_cooldown -= 1
            blocks["loss_cooldown"] = blocks.get("loss_cooldown", 0) + 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        if cur_dd >= gov.dd_hard:
            blocks["dd_hard_block"] = blocks.get("dd_hard_block", 0) + 1
            continue
        row = df.iloc[i]
        risk_off = float(row.get("clean_regime_2024_unsup_v4_risk_off_prob", 0.0) or 0.0)
        if loss_streak > 0 and (float(dec.quality_score) < gov.min_quality_after_loss or risk_off > gov.risk_off_cap_after_loss):
            blocks["post_loss_quality_risk_gate"] = blocks.get("post_loss_quality_risk_gate", 0) + 1
            continue
        scale = 1.0
        if cur_dd >= gov.dd_soft:
            scale = gov.dd_scale
        fill_idx = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_idx, pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = min(float(dec.notional_exposure) * scale, float(gov.max_notional))
        leverage = float(dec.leverage)
        take_profit = float(dec.take_profit) * scale
        stop_loss = max(0.0015, float(dec.stop_loss) * float(gov.stop_loss_scale) * max(scale, 0.65))
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        cash -= cash * fee * notional
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
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
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
                "max_hold_bars": int(max_hold),
                "quality_score": float(dec.quality_score),
                "confidence": float(dec.confidence),
                "drawdown_at_entry": float(cur_dd),
                "loss_streak_at_entry": int(loss_streak),
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


def _score(r1: dict[str, Any], r2: dict[str, Any], r3: dict[str, Any]) -> float:
    mdd = abs(float(r1["mdd"]))
    pnl = float(r1["pnl"])
    if int(r1["trades"]) < 20:
        return -1e9 + pnl
    target_penalty = max(0.0, mdd - 15.0) * 10.0
    pnl_floor_penalty = max(0.0, 100.0 - pnl) * 4.0
    stress_penalty = max(0.0, -float(r2["pnl"])) * 2.0 + max(0.0, -float(r3["pnl"])) * 3.0
    return float(pnl + 0.25 * float(r2["pnl"]) + 0.10 * float(r3["pnl"]) - 2.0 * mdd - target_penalty - pnl_floor_penalty - stress_penalty)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select MDD governor on 2025 validation and evaluate fixed 2026 OOS.")
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
    cfg = FullyLearnedGovernorConfig(**dict(bundle.get("config", {})))
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    val_decisions = predict_policy_frame(bundle, val, close=_close(val))
    eval_decisions = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for gov in _grid():
        r1 = backtest(val, bundle, gov, fee=cfg.fee, slip=cfg.slip, decisions=val_decisions)
        r2 = backtest(val, bundle, gov, fee=cfg.fee * 2.0, slip=cfg.slip * 2.0, decisions=val_decisions)
        r3 = backtest(val, bundle, gov, fee=cfg.fee * 3.0, slip=cfg.slip * 3.0, decisions=val_decisions)
        row = {"governor": asdict(gov), "validation_cost1": r1, "validation_cost2": r2, "validation_cost3": r3, "selection_score": _score(r1, r2, r3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("no governor selected")
    selected = GovernorConfig(**best["governor"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest(eval_df, bundle, selected, fee=cfg.fee * mult, slip=cfg.slip * mult, decisions=eval_decisions, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("ledger", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v13_clean_regime_mdd_governor_manifest.pkl"
    payload = {
        "model_id": MODEL_ID,
        "base_model": str(args.model),
        "governor": asdict(selected),
        "selection_policy": "MDD governor selected on 2025-10-01..2025-12-31 validation only; 2026 fixed OOS not used for selection.",
    }
    joblib.dump(payload, model_path)
    grid_path = args.report_out.with_name(args.report_out.stem + "_validation_grid.json")
    grid_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["mdd"] < -15.0:
        warnings.append("oos_mdd_target_not_met")
    if metrics["cost1"]["pnl"] < 0.0:
        warnings.append("oos_pnl_negative")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote_candidate" if not blocking and metrics["cost1"]["mdd"] >= -15.0 and metrics["cost1"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed, used only after governor selection",
        "selected_governor": asdict(selected),
        "feature_audit": feature_audit,
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Validation-selected runtime drawdown/loss-streak governor over clean v13 exposure-selected HF policy.",
        "base_model": str(args.model),
        "model": str(model_path),
        "split_policy": "Governor selected on 2025 Oct-Dec validation only; 2026 fixed OOS not used for selection.",
        "selected_governor": asdict(selected),
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
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected_governor": asdict(selected), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
