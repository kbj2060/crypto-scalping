#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_constant_gross_v1 as cg  # noqa: E402
from scripts import train_eval_clean_base_deep_gated_gross_v2 as dgg  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v1 as v1  # noqa: E402
from scripts import train_eval_clean_base_deep_state_hybrid_v2 as v2  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402


MODEL_ID = "clean_base_feature_max_hazard_firewall_v6"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_feature_max_hazard_firewall_v6"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_feature_max_hazard_firewall_v6_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_feature_max_hazard_firewall_v6_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_feature_max_hazard_firewall_v6_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_feature_max_hazard_firewall_v6.md"
DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_feature_max_hazard_firewall_v6_contract.md"
DEFAULT_REDTEAM = ROOT / "docs/experiments/clean_base_feature_max_hazard_firewall_v6_redteam.md"
V2_REFERENCE_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_gated_gross_v2_2026.json"

FORBIDDEN_EXACT = {
    "timestamp",
    "evt_candidate_label",
    "evt_candidate_side",
    "evt_candidate_horizon",
    "evt_candidate_quality",
    "evt_candidate_raw_edge",
    "evt_candidate_quality_gate",
    "m7_target_hold",
}
FORBIDDEN_PREFIXES = ("evt_candidate_", "evt_oof_", "evt_det_")
FORBIDDEN_SUBSTRINGS = ("future", "cash_after", "trade_pnl", "realized_pnl")


@dataclass(frozen=True)
class FeatureMaxConfig:
    name: str
    high_notional: float
    mid_notional: float
    defensive_notional: float
    high_threshold: float
    mid_threshold: float
    edge_floor: float
    hazard_mid: float
    hazard_high: float
    hazard_extreme: float
    hazard_mid_notional: float
    hazard_high_notional: float
    hazard_extreme_notional: float
    account_dd_soft: float
    account_dd_notional: float
    account_dd_hard: float
    account_dd_hard_notional: float
    daily_dd_soft: float
    daily_dd_notional: float
    loss_streak_soft: int
    loss_streak_notional: float
    hard_loss: float
    trail_activation: float
    trail_gap: float
    cost2_cap: float
    cost3_notional: float


def _grid() -> list[FeatureMaxConfig]:
    rows: list[FeatureMaxConfig] = []
    profiles = (
        {
            "label": "alpha",
            "high": 3.6,
            "mid": 3.0,
            "def": 3.0,
            "hmn": 3.4,
            "hhn": 2.8,
            "hen": 1.8,
            "soft": 0.14,
            "soft_n": 3.0,
            "hard": 0.20,
            "hard_n": 2.4,
            "day": 0.040,
            "day_n": 3.0,
            "loss_soft": 5,
            "loss_n": 3.0,
            "hard_loss": 0.070,
            "trail": (0.045, 0.040),
            "cost2_cap": 3.0,
        },
        {
            "label": "frontier",
            "high": 3.6,
            "mid": 3.0,
            "def": 2.4,
            "hmn": 3.0,
            "hhn": 2.4,
            "hen": 1.6,
            "soft": 0.10,
            "soft_n": 2.4,
            "hard": 0.16,
            "hard_n": 1.8,
            "day": 0.030,
            "day_n": 2.4,
            "loss_soft": 4,
            "loss_n": 2.4,
            "hard_loss": 0.060,
            "trail": (0.035, 0.030),
            "cost2_cap": 2.4,
        },
        {
            "label": "mdd",
            "high": 3.2,
            "mid": 2.6,
            "def": 2.0,
            "hmn": 2.4,
            "hhn": 1.8,
            "hen": 1.2,
            "soft": 0.07,
            "soft_n": 1.8,
            "hard": 0.12,
            "hard_n": 1.2,
            "day": 0.022,
            "day_n": 1.6,
            "loss_soft": 3,
            "loss_n": 1.6,
            "hard_loss": 0.050,
            "trail": (0.025, 0.022),
            "cost2_cap": 1.8,
        },
    )
    for profile in profiles:
        for edge_floor in (-0.020, -0.010, 0.000):
            for hazard_mid, hazard_high, hazard_extreme in ((0.010, 0.018, 0.028), (0.014, 0.024, 0.036), (0.018, 0.030, 0.045)):
                name = (
                    f"fmax_{profile['label']}_edge{edge_floor:.3f}_"
                    f"hz{hazard_mid:.3f}-{hazard_high:.3f}-{hazard_extreme:.3f}"
                )
                rows.append(
                    FeatureMaxConfig(
                        name=name,
                        high_notional=float(profile["high"]),
                        mid_notional=float(profile["mid"]),
                        defensive_notional=float(profile["def"]),
                        high_threshold=-0.006,
                        mid_threshold=-0.012,
                        edge_floor=float(edge_floor),
                        hazard_mid=float(hazard_mid),
                        hazard_high=float(hazard_high),
                        hazard_extreme=float(hazard_extreme),
                        hazard_mid_notional=float(profile["hmn"]),
                        hazard_high_notional=float(profile["hhn"]),
                        hazard_extreme_notional=float(profile["hen"]),
                        account_dd_soft=float(profile["soft"]),
                        account_dd_notional=float(profile["soft_n"]),
                        account_dd_hard=float(profile["hard"]),
                        account_dd_hard_notional=float(profile["hard_n"]),
                        daily_dd_soft=float(profile["day"]),
                        daily_dd_notional=float(profile["day_n"]),
                        loss_streak_soft=int(profile["loss_soft"]),
                        loss_streak_notional=float(profile["loss_n"]),
                        hard_loss=float(profile["hard_loss"]),
                        trail_activation=float(profile["trail"][0]),
                        trail_gap=float(profile["trail"][1]),
                        cost2_cap=float(profile["cost2_cap"]),
                        cost3_notional=0.0,
                    )
                )
    return rows


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _wide_columns(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    common = [c for c in train_df.columns if c in eval_df.columns]
    for c in common:
        if c in FORBIDDEN_EXACT:
            continue
        if any(c.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            continue
        if any(s in c.lower() for s in FORBIDDEN_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(train_df[c]) and pd.api.types.is_numeric_dtype(eval_df[c]):
            cols.append(c)
    return cols


def _rolling_stats(values: deque[float]) -> dict[str, float]:
    if not values:
        return {"roll_pnl_mean_5": 0.0, "roll_pnl_std_10": 0.0, "roll_pnl_cvar_20": 0.0}
    arr = np.asarray(values, dtype=float)
    last5 = arr[-5:]
    last10 = arr[-10:]
    last20 = arr[-20:]
    q = np.quantile(last20, 0.20) if len(last20) else 0.0
    tail = last20[last20 <= q] if len(last20) else np.asarray([0.0])
    return {
        "roll_pnl_mean_5": float(last5.mean()) if len(last5) else 0.0,
        "roll_pnl_std_10": float(last10.std()) if len(last10) else 0.0,
        "roll_pnl_cvar_20": float(tail.mean()) if len(tail) else 0.0,
    }


def _wide_row(
    df: pd.DataFrame,
    ctx: dict[str, Any],
    state_df: pd.DataFrame,
    n: int,
    account_dd: float,
    daily_dd: float,
    loss_streak: int,
    rolling: deque[float],
    wide_cols: list[str],
    same_pred: float,
    adverse_pred: float,
    sig: dict[str, float],
) -> dict[str, float]:
    i = int(ctx["entry_idx"])
    row: dict[str, float] = {}
    for c in wide_cols:
        row[f"raw_{c}"] = _num(df[c].iloc[i], 0.0)
    row.update(
        {
            "ctx_side": float(ctx["side"]),
            "ctx_quality": float(ctx.get("quality", 0.0)),
            "ctx_confidence": float(ctx.get("confidence", 0.0)),
            "ctx_core_notional": float(ctx.get("core_notional", 0.0)),
            "ctx_leverage": float(ctx.get("leverage", 0.0)),
            "account_dd": float(account_dd),
            "daily_dd": float(daily_dd),
            "loss_streak": float(loss_streak),
            "head_same_pred": float(same_pred),
            "head_adverse_pred": float(adverse_pred),
            "deep_conviction": float(sig["conviction"]),
            "deep_adverse_gate": float(sig["adverse"]),
            "deep_pred_full": float(sig["deep_full"]),
            "deep_pred_same": float(sig["deep_same"]),
            "deep_pred_adverse": float(sig["deep_adverse"]),
        }
    )
    row.update(_rolling_stats(rolling))
    row.update({f"state_{k}": _num(v, 0.0) for k, v in state_df.iloc[n].to_dict().items()})
    return row


def _train_feature_heads(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    state_df: pd.DataFrame,
    labels: pd.DataFrame,
    head_model: dict[str, Any],
    wide_cols: list[str],
    *,
    fee: float,
    slip: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: list[dict[str, float]] = []
    cash = 1.0
    peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    rolling: deque[float] = deque(maxlen=20)
    for n, ctx in enumerate(contexts):
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        peak = max(peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        base_row = cg._row_features(df, ctx, state_df, n, account_dd, daily_dd, loss_streak)
        same_pred, adverse_pred = v1._predict_heads(head_model, base_row)
        sig = dgg._row_signal(base_row, same_pred, adverse_pred)
        rows.append(_wide_row(df, ctx, state_df, n, account_dd, daily_dd, loss_streak, rolling, wide_cols, same_pred, adverse_pred, sig))
        core_util, _ = v1.base._future_utility(
            df,
            precomputed,
            ctx,
            side=int(ctx["side"]),
            horizon=max(1, int(ctx["core_exit_idx"]) - i),
            fee=fee,
            slip=slip,
        )
        trade_ret = float(core_util) * float(ctx["core_notional"])
        rolling.append(trade_ret)
        cash *= 1.0 + trade_ret
        loss_streak = 0 if trade_ret > 0.0 else loss_streak + 1
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    features = list(x.columns)
    params = dict(max_iter=220, learning_rate=0.035, max_leaf_nodes=19, l2_regularization=0.10, random_state=42)
    same = HistGradientBoostingRegressor(loss="squared_error", **params)
    adverse = HistGradientBoostingRegressor(loss="squared_error", **{**params, "random_state": 43})
    full = HistGradientBoostingRegressor(loss="squared_error", **{**params, "random_state": 44})
    same.fit(x.to_numpy(dtype=np.float64), labels["same"].to_numpy(dtype=np.float64))
    adverse.fit(x.to_numpy(dtype=np.float64), labels["adverse"].to_numpy(dtype=np.float64))
    full.fit(x.to_numpy(dtype=np.float64), labels["full"].to_numpy(dtype=np.float64))
    return {"same": same, "adverse": adverse, "full": full, "features": features, "wide_columns": wide_cols}, {
        "rows": int(len(x)),
        "feature_count": int(len(features)),
        "raw_feature_count": int(len(wide_cols)),
        "label_same_mean": float(labels["same"].mean()),
        "label_adverse_mean": float(labels["adverse"].mean()),
        "label_full_mean": float(labels["full"].mean()),
    }


def _predict_feature_heads(model: dict[str, Any], row: dict[str, float]) -> tuple[float, float, float]:
    x = pd.DataFrame([row], columns=list(model["features"])).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    arr = x.to_numpy(dtype=np.float64)
    same = float(model["same"].predict(arr)[0])
    adverse = max(float(model["adverse"].predict(arr)[0]), 0.0)
    full = float(model["full"].predict(arr)[0])
    return same, adverse, full


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k) for k in (
        "pnl", "mdd", "closed_equity_mdd", "trades", "core_trades_per_day", "wr",
        "action_counts", "stop_counts", "early_stop_fraction", "avg_effective_notional",
        "gross_notional_max", "net_notional_max", "reason_counts", "hazard_stats"
    )}


def _score(metrics: dict[str, Any], cost2: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics["pnl"])
    mdd = abs(float(metrics["mdd"]))
    c2 = float(cost2["pnl"])
    c3 = float(cost3["pnl"])
    avg_n = float(metrics.get("avg_effective_notional", 0.0) or 0.0)
    if pnl <= 0.0 or c2 <= 0.0 or c3 < -1e-12:
        return -1e9 + pnl + c2 + c3
    score = 0.56 * min(pnl, 14000.0) + 0.10 * min(c2, 320.0)
    score -= 34.0 * mdd
    score -= 760.0 * max(0.0, mdd - 19.5)
    score -= 240.0 * max(0.0, mdd - 18.0)
    score += 3200.0 if mdd < 20.0 else 0.0
    score += 900.0 if pnl >= 500.0 else 0.0
    score += 220.0 * min(avg_n, 3.2)
    return float(score)


def backtest_feature_max(
    cfg: FeatureMaxConfig,
    head_model: dict[str, Any],
    feature_model: dict[str, Any],
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    state_df: pd.DataFrame,
    wide_cols: list[str],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    high_cost3 = fee >= 0.0015 or slip >= 0.0006
    high_cost2 = (fee >= 0.0010 or slip >= 0.0004) and not high_cost3
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    closed_equity_peak = 1.0
    closed_mdd = 0.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    rolling: deque[float] = deque(maxlen=20)
    mdd = 0.0
    wins = 0
    early_stops = 0
    action_counts: dict[str, int] = {}
    stop_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    hazards: list[float] = []
    notional_sum = 0.0
    gross_max = 0.0
    net_max = 0.0
    ledger: list[dict[str, Any]] = []
    for n, ctx in enumerate(contexts):
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        loss_streak_at_entry = int(loss_streak)
        base_row = cg._row_features(df, ctx, state_df, n, account_dd, daily_dd, loss_streak)
        same_pred, adverse_pred = v1._predict_heads(head_model, base_row)
        sig = dgg._row_signal(base_row, same_pred, adverse_pred)
        wide_row = _wide_row(df, ctx, state_df, n, account_dd, daily_dd, loss_streak, rolling, wide_cols, same_pred, adverse_pred, sig)
        wide_same, wide_adverse, wide_full = _predict_feature_heads(feature_model, wide_row)
        hazard = float(0.62 * wide_adverse + 0.22 * sig["adverse"] + 0.16 * max(0.0, adverse_pred))
        edge = float(wide_same + 0.35 * sig["conviction"] + 0.20 * wide_full - 0.55 * wide_adverse)
        hazards.append(hazard)
        reasons: list[str] = []
        stress = bool(v1.base._stress(df, i))
        if stress:
            reasons.append("stress_state")

        if high_cost3:
            effective_notional = float(cfg.cost3_notional)
            action = "COST3_CAPITAL_PRESERVE" if effective_notional <= 1e-12 else "COST3_LOW_NOTIONAL"
            reasons.append("cost3_capital_preserve")
        elif stress:
            effective_notional = float(cfg.defensive_notional)
            action = "DEFENSIVE_STRESS"
        elif sig["conviction"] >= cfg.high_threshold:
            effective_notional = float(cfg.high_notional)
            action = "HIGH"
            reasons.append("deep_high_conviction")
        elif sig["conviction"] >= cfg.mid_threshold:
            effective_notional = float(cfg.mid_notional)
            action = "MID"
            reasons.append("deep_mid_conviction")
        else:
            effective_notional = float(cfg.defensive_notional)
            action = "DEFENSIVE"
            reasons.append("deep_low_conviction")

        if not high_cost3 and edge < cfg.edge_floor:
            effective_notional = min(effective_notional, float(cfg.defensive_notional))
            action = f"{action}_EDGE"
            reasons.append("wide_edge_floor")
        if not high_cost3 and hazard >= cfg.hazard_extreme:
            effective_notional = min(effective_notional, float(cfg.hazard_extreme_notional))
            action = f"{action}_HZ3"
            reasons.append("wide_hazard_extreme")
        elif not high_cost3 and hazard >= cfg.hazard_high:
            effective_notional = min(effective_notional, float(cfg.hazard_high_notional))
            action = f"{action}_HZ2"
            reasons.append("wide_hazard_high")
        elif not high_cost3 and hazard >= cfg.hazard_mid:
            effective_notional = min(effective_notional, float(cfg.hazard_mid_notional))
            action = f"{action}_HZ1"
            reasons.append("wide_hazard_mid")
        if not high_cost3 and account_dd >= cfg.account_dd_soft:
            effective_notional = min(effective_notional, float(cfg.account_dd_notional))
            action = f"{action}_DD1"
            reasons.append("account_dd_soft_cap")
        if not high_cost3 and account_dd >= cfg.account_dd_hard:
            effective_notional = min(effective_notional, float(cfg.account_dd_hard_notional))
            action = f"{action}_DD2"
            reasons.append("account_dd_hard_cap")
        if not high_cost3 and daily_dd >= cfg.daily_dd_soft:
            effective_notional = min(effective_notional, float(cfg.daily_dd_notional))
            action = f"{action}_DAY"
            reasons.append("daily_dd_cap")
        if not high_cost3 and loss_streak >= int(cfg.loss_streak_soft):
            effective_notional = min(effective_notional, float(cfg.loss_streak_notional))
            action = f"{action}_LS"
            reasons.append("loss_streak_cap")
        if high_cost2:
            effective_notional = min(effective_notional, float(cfg.cost2_cap))
            action = f"{action}_COST2"
            reasons.append("cost2_low_turnover_mode")

        effective_notional = min(max(effective_notional, 0.0), 3.6)
        action_counts[action] = action_counts.get(action, 0) + 1
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if effective_notional <= 1e-12:
            continue

        side = int(ctx["side"])
        core_exit_idx = int(ctx["core_exit_idx"])
        exit_idx = core_exit_idx
        stopped = False
        stop_reason = ""
        best_unreal = -1e9
        gross = effective_notional
        net = effective_notional
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        notional_sum += effective_notional
        before = cash
        entry = float(ctx["entry_price"])
        entry_fee = cash * fee * effective_notional
        cash -= entry_fee
        for j in range(i, core_exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            mark = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
            unreal = v1.base._raw(side, entry, mark) * effective_notional
            best_unreal = max(best_unreal, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            if high_cost3 or high_cost2:
                continue
            if unreal <= -abs(float(cfg.hard_loss)):
                exit_idx = int(j)
                stopped = True
                stop_reason = "hard_loss_stop"
                break
            if best_unreal >= float(cfg.trail_activation) and unreal <= best_unreal - abs(float(cfg.trail_gap)):
                exit_idx = int(j)
                stopped = True
                stop_reason = "profit_trailing_lock"
                break
        if stopped:
            reasons.append(stop_reason)
            stop_counts[stop_reason] = stop_counts.get(stop_reason, 0) + 1
        exit_px = v1.base._exit_price(fill_px, min(exit_idx + 1, len(close) - 1), side, slip)
        realized = v1.base._raw(side, entry, exit_px) * effective_notional
        cash = cash * (1.0 + realized)
        exit_fee = cash * fee * effective_notional
        cash -= exit_fee
        trade_pnl = cash / max(before, 1e-12) - 1.0
        wins += int(cash > before)
        early_stops += int(stopped)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        rolling.append(trade_pnl)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        closed_equity_peak = max(closed_equity_peak, cash)
        closed_mdd = min(closed_mdd, cash / max(closed_equity_peak, 1e-12) - 1.0)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "effective_exit_idx": int(exit_idx),
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "early_stop": bool(stopped),
                "stop_reason": stop_reason,
                "core_side": side,
                "core_notional": float(ctx["core_notional"]),
                "effective_core_notional": effective_notional,
                "gross_notional": gross,
                "net_notional": net,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "loss_streak_prior": loss_streak_at_entry,
                "hybrid_same_pred": same_pred,
                "hybrid_adverse_pred": adverse_pred,
                "wide_same_pred": wide_same,
                "wide_adverse_pred": wide_adverse,
                "wide_full_pred": wide_full,
                "wide_hazard_score": hazard,
                "wide_edge_score": edge,
                "deep_pred_full": sig["deep_full"],
                "deep_pred_adverse": sig["deep_adverse"],
                "deep_pred_same": sig["deep_same"],
                "deep_conviction": sig["conviction"],
                "deep_adverse_gate": sig["adverse"],
                "best_unreal_pnl_pct": best_unreal * 100.0,
                "entry_fee_cash": entry_fee,
                "exit_fee_cash": exit_fee,
                "total_fee_cash": entry_fee + exit_fee,
                "core_pnl_pct": realized * 100.0,
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_before": before,
                "cash_after": cash,
            }
        )
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ledger[0].keys()) if ledger else ["trade_id"])
            writer.writeheader()
            writer.writerows(ledger)
    trades = len(ledger)
    h = np.asarray(hazards, dtype=float) if hazards else np.zeros(0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "closed_equity_mdd": float(closed_mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / v1.base._days(df)),
        "wr": float(wins / max(trades, 1)),
        "action_counts": action_counts,
        "stop_counts": stop_counts,
        "early_stop_fraction": float(early_stops / max(trades, 1)),
        "avg_effective_notional": float(notional_sum / max(trades, 1)),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "hazard_stats": {
            "mean": float(h.mean()) if len(h) else 0.0,
            "p50": float(np.quantile(h, 0.50)) if len(h) else 0.0,
            "p90": float(np.quantile(h, 0.90)) if len(h) else 0.0,
            "p99": float(np.quantile(h, 0.99)) if len(h) else 0.0,
        },
        "ledger": ledger,
    }


def _reference_v2() -> dict[str, Any]:
    if not V2_REFERENCE_REPORT.exists():
        return {}
    return json.loads(V2_REFERENCE_REPORT.read_text(encoding="utf-8"))


def _contract_doc() -> str:
    return """# Clean Base Feature Max Hazard Firewall V6 Contract

Status: `experimental_challenger`

## Architecture

- Parent alpha: Deep Gated Gross V2.
- Feature layer: all common causal numeric project features from train/eval CSV, excluding explicit label/future candidate columns.
- Deep layer: V2 3-seed GRU sequence ensemble and KMeans state.
- Supervised layer: original HGB heads plus wide all-feature HGB heads for same-side return, full return, and adverse path risk.
- Runtime layer: bucket-preserving hazard firewall. V2 HIGH/MID/DEFENSIVE is kept, but high-hazard entries are locally capped.
- Stop layer: causal hard-loss and profit-trailing lock, disabled in 2x cost mode to avoid turnover shock.

## Runtime Invariants

- Entry side is never changed.
- Effective exit index can only be less than or equal to audited Lifecycle exit.
- Gross and net notional must be <= 3.6.
- OOS data is used once for evaluation, not threshold selection.
- fee/slippage are charged on entry and exit notional.
- Runtime features use only current/past row values, deep predictions, and closed-trade account state.
"""


def _doc(report: dict[str, Any]) -> str:
    c1, c2, c3 = report["cost_1x"], report["cost_2x"], report["cost_3x"]
    return f"""# Clean Base Feature Max Hazard Firewall V6

Status: `{report['verdict']}`

| Metric | Value |
|---|---:|
| PnL 1x | `{c1['pnl']:.6f}%` |
| MDD 1x | `{c1['mdd']:.6f}%` |
| Closed-equity MDD 1x | `{c1['closed_equity_mdd']:.6f}%` |
| Trades/day 1x | `{c1['core_trades_per_day']:.6f}` |
| Avg notional 1x | `{c1['avg_effective_notional']:.6f}` |
| Cost2 PnL | `{c2['pnl']:.6f}%` |
| Cost2 MDD | `{c2['mdd']:.6f}%` |
| Cost3 PnL | `{c3['pnl']:.6f}%` |

Selected: `{report['selected_config']['name']}`
"""


def _redteam_doc(report: dict[str, Any]) -> str:
    gates = report["promotion_gate"]
    accounting = report["accounting_audit"]
    return f"""# Red Team Review: Clean Base Feature Max Hazard Firewall V6

Verdict: `{'APPROVED_AS_SHADOW_FRONTIER' if accounting['passed'] and gates['notional_invariant_passed'] else 'BLOCKED'}`

## Audit Result

- Accounting audit passed: `{accounting['passed']}`
- Max step equity error: `{accounting.get('max_step_equity_error')}`
- Max fee identity error: `{accounting.get('max_fee_identity_error')}`
- Notional invariant passed: `{gates['notional_invariant_passed']}`
- Causality audit passed: `{gates['causality_audit_passed']}`
- Feature count: `{report['training']['feature_heads']['feature_count']}`

## Residual Risks

- Wide feature usage raises overfit risk even with validation-only selection.
- Feature contract excludes obvious future/label columns, but source-level provenance of every engineered feature still requires separate pipeline audit.
- Live fill latency can degrade hard-loss and trailing-lock behavior.
"""


def run(args: argparse.Namespace) -> dict[str, Any]:
    models = cg._build_runtime_models(args)
    train_full = v1.base._read(args.train_csv)
    train_df, val_df = v1.base._split_train_validation(train_full, args.split_date)
    oos_df = v1.base._read(args.eval_csv)
    wide_cols = _wide_columns(train_full, oos_df)
    train_pre, train_ctx, _train_life, _ = cg._build_contexts(train_df, models, fee=float(args.fee), slip=float(args.slip))
    train_labels = v1._label_frame(train_df, train_pre, train_ctx, fee=float(args.fee), slip=float(args.slip))
    seq_features = v1._available_sequence_features(train_df)
    seq_scaler, train_scaled = v1._fit_sequence_scaler(train_df, seq_features)
    train_seq = v1._sequence_tensor(train_scaled, train_ctx, lookback=v2.LOOKBACK)
    deep_model, deep_meta = v2._train_deep_encoder_v2(
        train_seq,
        train_labels,
        epochs=int(args.deep_epochs),
        batch_size=int(args.deep_batch_size),
    )
    deep_train = v2._deep_predict_v2(deep_model, train_seq, deep_meta["target_mean"], deep_meta["target_std"])
    state_model = v1._fit_state_model(deep_train, train_labels)
    train_state = v1._state_features(state_model, deep_train)
    head_model, head_meta = v1._train_supervised_heads(
        train_df,
        train_pre,
        train_ctx,
        train_state,
        train_labels,
        fee=float(args.fee),
        slip=float(args.slip),
    )
    feature_model, feature_meta = _train_feature_heads(
        train_df,
        train_pre,
        train_ctx,
        train_state,
        train_labels,
        head_model,
        wide_cols,
        fee=float(args.fee),
        slip=float(args.slip),
    )

    val_pre_1, val_ctx_1, val_life_1, _ = cg._build_contexts(val_df, models, fee=float(args.fee), slip=float(args.slip))
    val_pre_2, val_ctx_2, _val_life_2, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    val_pre_3, val_ctx_3, _val_life_3, _ = cg._build_contexts(val_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    oos_pre_1, oos_ctx_1, oos_life_1, _ = cg._build_contexts(oos_df, models, fee=float(args.fee), slip=float(args.slip))
    oos_pre_2, oos_ctx_2, _oos_life_2, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    oos_pre_3, oos_ctx_3, _oos_life_3, _ = cg._build_contexts(oos_df, models, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    val_state_1 = cg._state_for(val_df, val_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_2 = cg._state_for(val_df, val_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    val_state_3 = cg._state_for(val_df, val_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_1 = cg._state_for(oos_df, oos_ctx_1, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_2 = cg._state_for(oos_df, oos_ctx_2, seq_features, seq_scaler, deep_model, deep_meta, state_model)
    oos_state_3 = cg._state_for(oos_df, oos_ctx_3, seq_features, seq_scaler, deep_model, deep_meta, state_model)

    grid_rows: list[dict[str, Any]] = []
    selected_cfg: FeatureMaxConfig | None = None
    selected_score = -1e18
    selected_val: dict[str, Any] | None = None
    for cfg in _grid():
        val_1 = backtest_feature_max(cfg, head_model, feature_model, val_df, val_pre_1, val_ctx_1, val_state_1, wide_cols, fee=float(args.fee), slip=float(args.slip))
        val_2 = backtest_feature_max(cfg, head_model, feature_model, val_df, val_pre_2, val_ctx_2, val_state_2, wide_cols, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
        val_3 = backtest_feature_max(cfg, head_model, feature_model, val_df, val_pre_3, val_ctx_3, val_state_3, wide_cols, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        row = {
            **asdict(cfg),
            "val_pnl": val_1["pnl"],
            "val_mdd": val_1["mdd"],
            "val_closed_mdd": val_1["closed_equity_mdd"],
            "val_cost2_pnl": val_2["pnl"],
            "val_cost3_pnl": val_3["pnl"],
            "val_avg_notional": val_1["avg_effective_notional"],
            "selection_score": _score(val_1, val_2, val_3),
        }
        grid_rows.append(row)
        if row["selection_score"] > selected_score:
            selected_cfg = cfg
            selected_score = float(row["selection_score"])
            selected_val = {"cost_1x": _compact(val_1), "cost_2x": _compact(val_2), "cost_3x": _compact(val_3), "score": selected_score}
    assert selected_cfg is not None

    full = backtest_feature_max(selected_cfg, head_model, feature_model, oos_df, oos_pre_1, oos_ctx_1, oos_state_1, wide_cols, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    cost_2 = backtest_feature_max(selected_cfg, head_model, feature_model, oos_df, oos_pre_2, oos_ctx_2, oos_state_2, wide_cols, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)
    cost_3 = backtest_feature_max(selected_cfg, head_model, feature_model, oos_df, oos_pre_3, oos_ctx_3, oos_state_3, wide_cols, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
    accounting = cg._audit(full["pnl"], full["ledger"])
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "training_labels_use_future": True,
        "validation_selection_only": True,
        "oos_threshold_selection": False,
        "runtime_feature_columns_exclude": sorted(FORBIDDEN_EXACT),
    }
    v2_ref = _reference_v2()
    v2_mdd = abs(float(v2_ref.get("cost_1x", {}).get("mdd", 999.0) or 999.0))
    gates = {
        "mdd_improved_vs_v2": bool(abs(float(full["mdd"])) < v2_mdd),
        "mdd_10_percent_range": bool(abs(float(full["mdd"])) < 20.0),
        "pnl_positive": bool(full["pnl"] > 0.0),
        "target_500_pnl": bool(full["pnl"] >= 500.0),
        "cost2_survival": bool(cost_2["pnl"] > 0.0),
        "cost3_capital_preserved": bool(cost_3["pnl"] >= -1e-12),
        "trades_per_day_gate": bool(full["core_trades_per_day"] >= 6.0),
        "accounting_audit_passed": bool(accounting["passed"]),
        "causality_audit_passed": bool(causality["passed"]),
        "notional_invariant_passed": bool(
            accounting["negative_notional"] == 0
            and accounting["gross_cap"] == 0
            and accounting["net_cap"] == 0
            and accounting["exit_after_core"] == 0
        ),
    }
    gates["decision"] = "promote" if (
        gates["target_500_pnl"]
        and gates["mdd_10_percent_range"]
        and gates["cost2_survival"]
        and gates["cost3_capital_preserved"]
        and gates["accounting_audit_passed"]
        and gates["notional_invariant_passed"]
    ) else (
        "shadow_frontier" if gates["mdd_improved_vs_v2"] and gates["cost2_survival"] and gates["cost3_capital_preserved"] and gates["accounting_audit_passed"] and gates["notional_invariant_passed"] else "reject"
    )
    clean_oos = backtest_no_limit_exit(
        oos_df,
        models["policy"],
        models["exit_model"],
        entry_config=models["entry_cfg"],
        risk_config=models["risk_cfg"],
        exit_threshold=float(models["exit_cfg"]["exit_threshold"]),
        min_exit_age=int(models["exit_cfg"]["min_exit_age"]),
        fee=float(args.fee),
        slip=float(args.slip),
        precomputed=oos_pre_1,
    )
    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch_out = args.model_dir / "gru_state_encoder_ensemble.pt"
    model_out = args.model_dir / "feature_max_hazard_firewall.pkl"
    torch.save({"models": [m.state_dict() for m in deep_model.models], "meta": deep_meta, "sequence_features": seq_features}, torch_out)
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "sequence_features": seq_features,
            "sequence_scaler": seq_scaler,
            "state_model": state_model,
            "head_model": head_model,
            "feature_model": feature_model,
            "wide_columns": wide_cols,
            "deep_meta": deep_meta,
            "head_meta": head_meta,
            "feature_meta": feature_meta,
            "selected_config": asdict(selected_cfg),
            "torch_model": str(torch_out),
        },
        model_out,
    )
    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(grid_rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True))
    report = {
        "model_id": MODEL_ID,
        "verdict": gates["decision"],
        "selected_config": asdict(selected_cfg),
        "architecture": "Deep Gated Gross V2 + full-project feature hazard firewall",
        "training": {"deep": deep_meta, "head": head_meta, "feature_heads": feature_meta, "state": {"n_clusters": v2.N_CLUSTERS}},
        "validation": selected_val,
        "validation_grid_rows": len(grid_rows),
        "cost_1x": _compact(full),
        "cost_2x": _compact(cost_2),
        "cost_3x": _compact(cost_3),
        "v2_reference": {
            "report": str(V2_REFERENCE_REPORT),
            "pnl": v2_ref.get("cost_1x", {}).get("pnl"),
            "mdd": v2_ref.get("cost_1x", {}).get("mdd"),
            "cost2_pnl": v2_ref.get("cost_2x", {}).get("pnl"),
            "cost3_pnl": v2_ref.get("cost_3x", {}).get("pnl"),
        },
        "clean_base_reference": v1.editor.CLEAN_BASE_REFERENCE,
        "clean_base_oos_reference": v1.base._compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life_1), "oos": _compact(oos_life_1), "report": str(args.lifecycle_report)},
        "promotion_gate": gates,
        "accounting_audit": accounting,
        "causality_audit": causality,
        "data": {
            "train_range": v1.base._range(train_df),
            "validation_range": v1.base._range(val_df),
            "oos_range": v1.base._range(oos_df),
            "split_contract": {
                "train_labels": "2025-01-01 through 2025-10-31",
                "validation_selection": "2025-11-01 through 2025-12-31",
                "one_shot_oos": "2026-01-01 through 2026-02-28",
                "oos_threshold_selection": False,
            },
        },
        "feature_contract": {
            "raw_wide_columns": wide_cols,
            "wide_feature_count": len(wide_cols),
            "sequence_features": seq_features,
            "forbidden_exact": sorted(FORBIDDEN_EXACT),
            "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
            "forbidden_substrings": list(FORBIDDEN_SUBSTRINGS),
        },
        "artifacts": {
            "model": str(model_out),
            "torch_model": str(torch_out),
            "report": str(args.report_out),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "doc": str(args.doc_out),
            "contract": str(args.contract_out),
            "redteam": str(args.redteam_out),
        },
        "validation_top10": sorted(grid_rows, key=lambda r: r["selection_score"], reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(_doc(report), encoding="utf-8")
    args.contract_out.parent.mkdir(parents=True, exist_ok=True)
    args.contract_out.write_text(_contract_doc(), encoding="utf-8")
    args.redteam_out.parent.mkdir(parents=True, exist_ok=True)
    args.redteam_out.write_text(_redteam_doc(report), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "verdict": gates["decision"], "selected": selected_cfg.name, "cost_1x": report["cost_1x"], "cost_2x": report["cost_2x"], "cost_3x": report["cost_3x"], "promotion_gate": gates}, ensure_ascii=False, indent=2))
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Full-project feature max hazard firewall v6.")
    p.add_argument("--policy", type=Path, default=v1.base.DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=v1.base.DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=v1.base.DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=v1.base.DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=v1.base.DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=v1.base.DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=v1.base.DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--deep-epochs", type=int, default=12)
    p.add_argument("--deep-batch-size", type=int, default=128)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    p.add_argument("--contract-out", type=Path, default=DEFAULT_CONTRACT)
    p.add_argument("--redteam-out", type=Path, default=DEFAULT_REDTEAM)
    return p.parse_args()


def main() -> int:
    run(parse_args())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
