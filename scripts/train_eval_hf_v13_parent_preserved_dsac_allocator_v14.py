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
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame  # noqa: E402
from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402


MODEL_ID = "hf_v13_parent_preserved_dsac_allocator_v14_20260511"
DEFAULT_MODEL = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_validation_selected_exposure_20260511/v13_clean_regime_validation_selected_exposure.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_DSAC = ROOT / "data/ensemble/ckpt/best_dsac_agents.pth"
DEFAULT_DSAC_CONFIG = ROOT / "data/ensemble/ckpt/dsac_train_config_latest.json"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_parent_preserved_dsac_allocator_v14_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_parent_preserved_dsac_allocator_v14_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_parent_preserved_dsac_allocator_v14_20260511_grid.csv"


@dataclass(frozen=True)
class AllocatorConfig:
    name: str
    same_boost_th: float
    same_boost: float
    opposite_th: float
    opposite_scale: float
    weak_score_th: float
    weak_scale: float
    dd_start: float
    dd_scale: float
    min_mult: float
    max_mult: float


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


def _safe_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    if not np.isfinite(x):
        return float(default)
    return x


def _load_actor(path: Path, device: str) -> tuple[GaussianActor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt


def _flat_dsac_signals(df: pd.DataFrame, ckpt_path: Path, device: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    actor, ckpt = _load_actor(ckpt_path, device)
    router = DSACRouter(actor, device=device)
    cols = [c for c in df.columns if c != "timestamp"]
    vals = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    arr = vals.to_numpy(dtype=np.float64, copy=False)
    sides: list[int] = []
    scores: list[float] = []
    raws: list[float] = []
    for row in arr:
        features = {k: float(v) for k, v in zip(cols, row)}
        action, _kelly, info = router.decide(features, {"type": None, "entry_price": 0.0, "unrealized": 0.0, "mdd": 0.0, "hold_count": 0.0})
        raw = _safe_float((info or {}).get("raw_action", 0.0), 0.0)
        sides.append(1 if int(action) == 1 else (-1 if int(action) == 2 else 0))
        scores.append(abs(raw))
        raws.append(raw)
    meta = {
        "path": str(ckpt_path),
        "device": device,
        "ckpt_epoch": int(ckpt.get("epoch", -1) or -1),
        "ckpt_best_score": _safe_float(ckpt.get("best_score", np.nan), np.nan),
        "ckpt_state_dim": int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM),
    }
    return pd.DataFrame({"dsac_side": sides, "dsac_score": scores, "dsac_raw_action": raws}), meta


def _grid() -> list[AllocatorConfig]:
    rows: list[AllocatorConfig] = []
    idx = 0
    for opposite_scale in (0.35, 0.55, 0.75):
        for dd_start, dd_scale in ((0.12, 0.55), (0.16, 0.65)):
            rows.append(
                AllocatorConfig(
                    name=f"dsac_soft_alloc_{idx}",
                    same_boost_th=0.22,
                    same_boost=1.08,
                    opposite_th=0.18,
                    opposite_scale=opposite_scale,
                    weak_score_th=0.10,
                    weak_scale=0.85,
                    dd_start=dd_start,
                    dd_scale=dd_scale,
                    min_mult=0.35,
                    max_mult=1.15,
                )
            )
            idx += 1
    rows.append(
        AllocatorConfig(
            name="dsac_noop_parent",
            same_boost_th=999.0,
            same_boost=1.0,
            opposite_th=999.0,
            opposite_scale=1.0,
            weak_score_th=-1.0,
            weak_scale=1.0,
            dd_start=999.0,
            dd_scale=1.0,
            min_mult=1.0,
            max_mult=1.0,
        )
    )
    return rows


def _entry_multiplier(dec: pd.Series, dsac: pd.Series, cfg: AllocatorConfig, current_dd_abs: float) -> tuple[float, str]:
    parent_side = int(dec.side)
    dsac_side = int(dsac.dsac_side)
    score = _safe_float(dsac.dsac_score, 0.0)
    mult = 1.0
    reasons: list[str] = []
    if dsac_side == parent_side and score >= cfg.same_boost_th:
        mult *= cfg.same_boost
        reasons.append("dsac_same_boost")
    elif dsac_side == -parent_side and score >= cfg.opposite_th:
        mult *= cfg.opposite_scale
        reasons.append("dsac_opposite_scale")
    elif score < cfg.weak_score_th or dsac_side == 0:
        mult *= cfg.weak_scale
        reasons.append("dsac_weak_scale")
    if current_dd_abs >= cfg.dd_start:
        mult *= cfg.dd_scale
        reasons.append("dd_scale")
    mult = float(np.clip(mult, cfg.min_mult, cfg.max_mult))
    return mult, "+".join(reasons) if reasons else "parent_keep"


def backtest_with_allocator(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    dsac: pd.DataFrame,
    cfg: AllocatorConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    record_trades: bool = False,
) -> dict[str, Any]:
    close = _close(df)
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
    peak_unrealized = 0.0
    trades = wins = long_entries = short_entries = 0
    action_counts: dict[str, int] = {"cash": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}
    reasons: dict[str, int] = {}
    notional_sum = leverage_sum = mult_sum = 0.0
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    fee_eff = fee * float(cost_mult)
    slip_eff = slip * float(cost_mult)

    def mark_equity(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark_equity(i)
        peak = max(peak, eq)
        current_dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
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
                exit_price = _fill_price(df, fill_idx, pos, slip_eff, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record_trades and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "peak_unrealized_pct": float(peak_unrealized * 100.0),
                            "fee_exit_pct": float(fee_eff * notional * 100.0),
                        }
                    )
                    records.append(out)
                pos = 0
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
            mult, mult_reason = _entry_multiplier(dec, dsac.iloc[i], cfg, current_dd_abs)
            action_counts["long" if int(dec.action) == ACTION_LONG else "short"] += 1
            fill_idx = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(df, fill_idx, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            base_notional = float(dec.notional_exposure)
            base_fraction = float(dec.position_fraction)
            notional = max(0.0, base_notional * mult)
            leverage = float(dec.leverage)
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += leverage
            mult_sum += mult
            reasons[mult_reason] = reasons.get(mult_reason, 0) + 1
            if record_trades:
                open_record = {
                    "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                    "entry_fill_timestamp": str(df["timestamp"].iloc[fill_idx]),
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "base_notional_exposure": float(base_notional),
                    "notional_exposure": float(notional),
                    "base_position_fraction": float(base_fraction),
                    "position_fraction": float(base_fraction * mult),
                    "leverage": float(leverage),
                    "allocator_multiplier": float(mult),
                    "allocator_reason": mult_reason,
                    "dsac_side": int(dsac["dsac_side"].iloc[i]),
                    "dsac_score": float(dsac["dsac_score"].iloc[i]),
                    "dsac_raw_action": float(dsac["dsac_raw_action"].iloc[i]),
                    "drawdown_abs_at_entry": float(current_dd_abs),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                    "cooldown_bars": int(next_cooldown),
                    "quality_score": float(dec.quality_score),
                    "confidence": float(dec.confidence),
                    "fee_entry_pct": float(fee_eff * notional * 100.0),
                }
    if pos != 0:
        fill_idx = len(df) - 1
        exit_price = _fill_price(df, fill_idx, pos, slip_eff, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "avg_allocator_multiplier": float(mult_sum / n_entries),
        "action_counts": action_counts,
        "exits": exits,
        "allocator_reasons": reasons,
    }
    if record_trades:
        out["trade_records"] = records
    return out


def _score(v1: dict[str, Any], v2: dict[str, Any], v3: dict[str, Any]) -> float:
    pnl = float(v1["pnl"])
    mdd_abs = abs(float(v1["mdd"]))
    trades = int(v1["trades"])
    if trades < 20:
        return -1e9 + pnl
    penalty = 0.0
    if pnl < 0.0:
        penalty += abs(pnl) * 3.0
    if float(v2["pnl"]) < 0.0:
        penalty += abs(float(v2["pnl"])) * 1.5
    if float(v3["pnl"]) < 0.0:
        penalty += abs(float(v3["pnl"])) * 2.0
    # Validation PnL is noisy and negative for the current parent, so selection prefers
    # survival, MDD compression, cost resilience, and trade preservation.
    return float(pnl + 0.20 * float(v2["pnl"]) + 0.10 * float(v3["pnl"]) - 2.0 * mdd_abs + 1.0 * min(float(v1["trades_per_day"]), 2.0) - penalty)


def _audit_dsac_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"status": "warn", "warnings": ["dsac_train_config_missing"], "path": str(path)}
    obj = json.loads(path.read_text(encoding="utf-8"))
    csv_path = str(obj.get("csv_path", ""))
    blocking = []
    warnings = []
    if "2026" in csv_path:
        blocking.append("dsac_config_csv_mentions_2026")
    if "2025" not in csv_path:
        warnings.append("dsac_config_csv_not_explicitly_2025")
    return {"status": "pass" if not blocking else "fail", "blocking": blocking, "warnings": warnings, "path": str(path), "csv_path": csv_path, "saved_at": obj.get("saved_at")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parent-preserved v13 DSAC allocator: side/entry/exit unchanged, notional multiplier only.")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--dsac-ckpt", type=Path, default=DEFAULT_DSAC)
    p.add_argument("--dsac-config", type=Path, default=DEFAULT_DSAC_CONFIG)
    p.add_argument("--device", default="cuda")
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.model)
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    validation = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = list(bundle.get("feature_cols") or [])
    feature_audit = _audit_contract(train_all, eval_df, feature_cols)
    dsac_cfg_audit = _audit_dsac_config(args.dsac_config)
    val_dsac, val_dsac_meta = _flat_dsac_signals(validation, args.dsac_ckpt, args.device)
    oos_dsac, oos_dsac_meta = _flat_dsac_signals(eval_df, args.dsac_ckpt, val_dsac_meta["device"])
    cfg_base = dict(bundle["config"])
    fee = float(cfg_base["fee"])
    slip = float(cfg_base["slip"])

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest_with_allocator(validation, bundle, val_dsac, cfg, fee=fee, slip=slip, cost_mult=1.0)
        v2 = backtest_with_allocator(validation, bundle, val_dsac, cfg, fee=fee, slip=slip, cost_mult=2.0)
        v3 = backtest_with_allocator(validation, bundle, val_dsac, cfg, fee=fee, slip=slip, cost_mult=3.0)
        row = {
            "config": asdict(cfg),
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _score(v1, v2, v3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    if best is None:
        raise RuntimeError("empty allocator grid")
    selected_cfg = AllocatorConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest_with_allocator(eval_df, bundle, oos_dsac, selected_cfg, fee=fee, slip=slip, cost_mult=float(mult), record_trades=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger_path.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result

    grid = pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "selection_score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_cost2_pnl": r["validation_cost2"]["pnl"],
                "val_cost3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    )
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    if dsac_cfg_audit["status"] == "fail":
        blocking.extend(dsac_cfg_audit.get("blocking", []))
    warnings.extend(dsac_cfg_audit.get("warnings", []))
    if int(feature_audit.get("train_eval_timestamp_overlap", 0)) != 0:
        blocking.append("train_eval_timestamp_overlap")
    if selected_cfg.min_mult < 0.0:
        blocking.append("allocator_can_flip_or_short_circuit_parent")
    if metrics["cost1"]["pnl"] < 100.0:
        warnings.append("oos_cost1_below_100pct_target")
    if abs(float(metrics["cost1"]["mdd"])) > 15.0:
        warnings.append("oos_mdd_above_15pct_target")

    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and metrics["cost1"]["pnl"] >= 100.0 and abs(float(metrics["cost1"]["mdd"])) <= 15.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS, evaluated after selection only",
        "parent_preservation": {
            "side_flip_allowed": False,
            "entry_retime_allowed": False,
            "exit_contract_changed": False,
            "action_space": "notional_multiplier_only",
            "multiplier_bounds": [selected_cfg.min_mult, selected_cfg.max_mult],
        },
        "feature_audit": feature_audit,
        "dsac_config_audit": dsac_cfg_audit,
        "selected_config": asdict(selected_cfg),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V13 clean-regime parent policy preserved; DSAC actor is used only as a causal notional multiplier allocator with validation-only config selection.",
        "model": str(args.model),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "dsac": {"validation_meta": val_dsac_meta, "oos_meta": oos_dsac_meta, "config_audit": dsac_cfg_audit},
        "split_policy": "Grid selected on 2025 Oct-Dec validation only; 2026 is fixed OOS and never used for selection.",
        "selected_config": asdict(selected_cfg),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.parent.mkdir(parents=True, exist_ok=True)
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "selected": asdict(selected_cfg), "metrics": metrics, "verdict": audit["verdict"]}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
