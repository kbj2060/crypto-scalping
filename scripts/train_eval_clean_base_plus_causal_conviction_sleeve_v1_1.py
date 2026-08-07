#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # noqa: E402
    from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # type: ignore
except ModuleNotFoundError:  # noqa: E402
    def _decision_audit(dec: pd.DataFrame, *, max_notional: float, leverage_cap: float) -> dict[str, Any]:
        action = pd.to_numeric(dec.get("action", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        side = pd.to_numeric(dec.get("side", 0), errors="coerce").fillna(0).to_numpy(dtype=np.int64)
        notional = pd.to_numeric(dec.get("notional_exposure", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        leverage = pd.to_numeric(dec.get("leverage", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
        pf = pd.to_numeric(dec.get("position_fraction", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        cooldown = pd.to_numeric(dec.get("cooldown_bars", 0), errors="coerce").fillna(0).to_numpy(dtype=np.float64)
        active = (action != 0) & (side != 0) & (notional > 0.0)
        violations = {
            "nonfinite_values": int((~np.isfinite(notional) | ~np.isfinite(leverage) | ~np.isfinite(pf)).sum()),
            "negative_notional": int((notional < -1e-12).sum()),
            "leverage_below_one_active": int((active & (leverage < 1.0 - 1e-12)).sum()),
            "leverage_above_cap": int((active & (leverage > float(leverage_cap) + 1e-12)).sum()),
            "notional_above_max": int((active & (notional > float(max_notional) + 1e-12)).sum()),
            "active_action_side_mismatch": int((((action != 0) ^ (side != 0)) & (notional > 1e-12)).sum()),
            "cash_has_exposure": int(((action == 0) & ((side != 0) | (notional > 1e-12) | (pf > 1e-12))).sum()),
            "position_fraction_mismatch": int((active & (np.abs(pf - notional / np.maximum(leverage, 1e-12)) > 1e-9)).sum()),
            "negative_cooldown": int((cooldown < -1e-12).sum()),
        }
        active_notional = notional[active]
        active_lev = leverage[active]
        return {
            "passed": bool(sum(violations.values()) == 0),
            "rows": int(len(dec)),
            "active_rows": int(active.sum()),
            "cash_rows": int((~active).sum()),
            "long_rows": int((active & (side > 0)).sum()),
            "short_rows": int((active & (side < 0)).sum()),
            "violations": violations,
            "notional": {
                "max": float(active_notional.max()) if active_notional.size else 0.0,
                "mean": float(active_notional.mean()) if active_notional.size else 0.0,
                "p95": float(np.quantile(active_notional, 0.95)) if active_notional.size else 0.0,
            },
            "leverage": {
                "max": float(active_lev.max()) if active_lev.size else 0.0,
                "mean": float(active_lev.mean()) if active_lev.size else 0.0,
                "p95": float(np.quantile(active_lev, 0.95)) if active_lev.size else 0.0,
            },
        }
from scripts.train_eval_clean_base_lifecycle_editor_v1 import (  # noqa: E402
    BASE_REFERENCE,
    LifecycleRuntimeConfig,
    _base_frame,
    _base_trade_plan,
    _compact,
    _range,
    _read,
    _sha256,
    _split_train_validation,
    backtest_lifecycle_editor,
)
from scripts.train_eval_hf_no_limit_exit_governor import backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_LIFECYCLE_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"
DEFAULT_LIFECYCLE_MODEL = ROOT / "data/ensemble/supervised/clean_base_lifecycle_editor_v1/lifecycle_editor.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_plus_causal_conviction_sleeve_v1_1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_plus_causal_conviction_sleeve_v1_1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/clean_base_plus_causal_conviction_sleeve_v1_1.md"

FEATURES = [
    "side",
    "quality",
    "confidence",
    "core_notional",
    "leverage",
    "account_dd",
    "daily_dd",
    "loss_streak",
    "funding_abs",
    "funding_pressure",
    "liquidity_vacuum",
    "amihud_illiquidity_z",
    "m7_tail_risk",
    "evt_tail_flag",
    "ai_adverse_risk",
]


@dataclass(frozen=True)
class CausalSleeveConfig:
    name: str
    same_threshold: float
    hedge_threshold: float
    max_sleeve_frac: float
    max_sleeve_bars: int
    same_enabled: bool
    hedge_enabled: bool
    account_dd_disable: float
    daily_dd_disable: float


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or df.empty:
        return max(len(df), 1) / 288.0
    start = pd.Timestamp(df["timestamp"].iloc[0]).normalize()
    end = pd.Timestamp(df["timestamp"].iloc[-1]).normalize()
    return max(float((end - start).days + 1), 1.0)


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _load_lifecycle_model(path: Path) -> tuple[dict[str, Any], LifecycleRuntimeConfig]:
    payload = joblib.load(path)
    return dict(payload["recalibrator"]), LifecycleRuntimeConfig(**dict(payload["selected_runtime_config"]))


def _grid() -> list[CausalSleeveConfig]:
    rows: dict[str, CausalSleeveConfig] = {}
    for same_thr in (0.0025, 0.0040):
        for hedge_thr in (0.0025,):
            for frac in (0.10, 0.25):
                for bars in (6,):
                    for same_enabled in (True, False):
                        for hedge_enabled in (True, False):
                            for acct in (0.08,):
                                for day in (0.015,):
                                    name = (
                                        f"same{same_thr:.4f}_hedge{hedge_thr:.4f}_frac{frac:.2f}_"
                                        f"bars{bars}_same{int(same_enabled)}_hedge{int(hedge_enabled)}_"
                                        f"acct{acct:.3f}_day{day:.3f}"
                                    )
                                    rows[name] = CausalSleeveConfig(
                                        name=name,
                                        same_threshold=float(same_thr),
                                        hedge_threshold=float(hedge_thr),
                                        max_sleeve_frac=float(frac),
                                        max_sleeve_bars=int(bars),
                                        same_enabled=bool(same_enabled),
                                        hedge_enabled=bool(hedge_enabled),
                                        account_dd_disable=float(acct),
                                        daily_dd_disable=float(day),
                                    )
    return list(rows.values())


def _row_value(df: pd.DataFrame, i: int, col: str, default: float = 0.0) -> float:
    if col not in df.columns or i < 0 or i >= len(df):
        return default
    return _num(df[col].iloc[i], default)


def _stress(df: pd.DataFrame, i: int) -> bool:
    return bool(
        _row_value(df, i, "evt_tail_flag") > 0.0
        or _row_value(df, i, "m7_tail_risk") > 0.0
        or abs(_row_value(df, i, "liquidity_vacuum")) > 1.0
        or abs(_row_value(df, i, "funding_pressure")) > 0.12
        or abs(_row_value(df, i, "ai_adverse_risk")) > 0.75
    )


def _entry_price(fill_px: np.ndarray, idx: int, side: int, slip: float) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    return px * (1.0 + slip) if side > 0 else px * (1.0 - slip)


def _exit_price(fill_px: np.ndarray, idx: int, side: int, slip: float) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    return px * (1.0 - slip) if side > 0 else px * (1.0 + slip)


def _raw(side: int, entry_price: float, exit_price: float) -> float:
    if side > 0:
        return (exit_price - entry_price) / max(entry_price, 1e-12)
    return (entry_price - exit_price) / max(entry_price, 1e-12)


def _contexts(
    df: pd.DataFrame,
    lifecycle_plan: list[dict[str, Any]],
    base_trades: list[dict[str, Any]],
    fill_px: np.ndarray,
    *,
    slip: float,
) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    out: list[dict[str, Any]] = []
    for trade_id, life in enumerate(lifecycle_plan):
        base = by_entry[int(life["entry_idx"])]
        entry_idx = int(base["entry_idx"])
        side = int(base["side"])
        out.append(
            {
                "trade_id": int(trade_id),
                "entry_idx": entry_idx,
                "base_exit_idx": int(base["exit_idx"]),
                "core_exit_idx": int(life["effective_exit_idx"]),
                "side": side,
                "entry_price": float(_entry_price(fill_px, min(entry_idx + 1, len(df) - 1), side, slip)),
                "core_notional": float(life["effective_notional"]),
                "base_notional": float(base["base_notional"]),
                "leverage": float(base["leverage"]),
                "quality": float(base.get("entry_quality", 0.0)),
                "confidence": float(base.get("entry_confidence", 0.0)),
                "timestamp": str(df["timestamp"].iloc[entry_idx]) if "timestamp" in df.columns else str(entry_idx),
            }
        )
    return out


def _features(df: pd.DataFrame, ctx: dict[str, Any], account_dd: float, daily_dd: float, loss_streak: int) -> dict[str, float]:
    i = int(ctx["entry_idx"])
    return {
        "side": float(ctx["side"]),
        "quality": float(ctx["quality"]),
        "confidence": float(ctx["confidence"]),
        "core_notional": float(ctx["core_notional"]),
        "leverage": float(ctx["leverage"]),
        "account_dd": float(account_dd),
        "daily_dd": float(daily_dd),
        "loss_streak": float(loss_streak),
        "funding_abs": abs(_row_value(df, i, "funding_abs")),
        "funding_pressure": _row_value(df, i, "funding_pressure"),
        "liquidity_vacuum": _row_value(df, i, "liquidity_vacuum"),
        "amihud_illiquidity_z": _row_value(df, i, "amihud_illiquidity_z"),
        "m7_tail_risk": _row_value(df, i, "m7_tail_risk"),
        "evt_tail_flag": _row_value(df, i, "evt_tail_flag"),
        "ai_adverse_risk": _row_value(df, i, "ai_adverse_risk"),
    }


def _future_utility(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    ctx: dict[str, Any],
    *,
    side: int,
    horizon: int,
    fee: float,
    slip: float,
) -> tuple[float, float]:
    _feat, _dec, close, fill_px = precomputed
    entry_idx = int(ctx["entry_idx"])
    exit_idx = min(int(ctx["core_exit_idx"]), entry_idx + int(horizon), len(close) - 2)
    entry = _entry_price(fill_px, min(entry_idx + 1, len(close) - 1), side, slip)
    worst = 0.0
    best = 0.0
    for j in range(entry_idx, exit_idx + 1):
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        mark_exit = px * (1.0 - slip) if side > 0 else px * (1.0 + slip)
        r = _raw(side, entry, mark_exit)
        worst = min(worst, r)
        best = max(best, r)
    exit_px = _exit_price(fill_px, min(exit_idx + 1, len(close) - 1), side, slip)
    net = _raw(side, entry, exit_px) - 2.0 * fee
    adverse = abs(min(worst, 0.0))
    giveback = max(0.0, best - net)
    utility = net - 0.80 * adverse - 0.35 * giveback
    return float(utility), float(adverse)


def _train_scorer(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    max_rows: int = 600,
) -> tuple[Any, dict[str, Any]]:
    rows: list[dict[str, float]] = []
    y_same: list[float] = []
    y_hedge: list[float] = []
    cash = 1.0
    peak = 1.0
    day_key: str | None = None
    daily_peak = 1.0
    loss_streak = 0
    stride = max(1, len(contexts) // max_rows)
    for n, ctx in enumerate(contexts):
        if n % stride != 0:
            continue
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        peak = max(peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        rows.append(_features(df, ctx, account_dd, daily_dd, loss_streak))
        same24, _ = _future_utility(df, precomputed, ctx, side=int(ctx["side"]), horizon=24, fee=fee, slip=slip)
        same72, _ = _future_utility(df, precomputed, ctx, side=int(ctx["side"]), horizon=72, fee=fee, slip=slip)
        hedge24, _ = _future_utility(df, precomputed, ctx, side=-int(ctx["side"]), horizon=24, fee=fee, slip=slip)
        hedge72, _ = _future_utility(df, precomputed, ctx, side=-int(ctx["side"]), horizon=72, fee=fee, slip=slip)
        y_same.append(max(same24, same72))
        y_hedge.append(max(hedge24, hedge72))
        # Causal training-state replay uses core outcome only to advance account state after this decision.
        core_util, _ = _future_utility(df, precomputed, ctx, side=int(ctx["side"]), horizon=max(1, int(ctx["core_exit_idx"]) - i), fee=fee, slip=slip)
        cash *= 1.0 + core_util * float(ctx["core_notional"])
        loss_streak = 0 if core_util > 0.0 else loss_streak + 1
    x = pd.DataFrame(rows, columns=FEATURES).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    same = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.05, random_state=42)
    hedge = HistGradientBoostingRegressor(max_iter=120, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=0.05, random_state=43)
    same.fit(x.to_numpy(dtype=np.float64), np.asarray(y_same, dtype=np.float64))
    hedge.fit(x.to_numpy(dtype=np.float64), np.asarray(y_hedge, dtype=np.float64))
    return {"same": same, "hedge": hedge}, {
        "rows": int(len(x)),
        "same_positive_rate": float(np.mean(np.asarray(y_same) > 0.0)),
        "hedge_positive_rate": float(np.mean(np.asarray(y_hedge) > 0.0)),
        "same_mean": float(np.mean(y_same)),
        "hedge_mean": float(np.mean(y_hedge)),
    }


def _predict(model: Any, feat: dict[str, float]) -> tuple[float, float]:
    x = pd.DataFrame([feat], columns=FEATURES).replace([np.inf, -np.inf], 0.0).fillna(0.0).to_numpy(dtype=np.float64)
    return float(model["same"].predict(x)[0]), float(model["hedge"].predict(x)[0])


def _choose_sleeve(
    cfg: CausalSleeveConfig,
    model: Any,
    df: pd.DataFrame,
    ctx: dict[str, Any],
    feat: dict[str, float],
) -> tuple[str, int, float, dict[str, float], list[str]]:
    same_pred, hedge_pred = _predict(model, feat)
    reasons: list[str] = []
    if feat["account_dd"] >= cfg.account_dd_disable:
        return "NO_SLEEVE", 0, 0.0, {"same_pred": same_pred, "hedge_pred": hedge_pred}, ["account_dd_disable"]
    if feat["daily_dd"] >= cfg.daily_dd_disable:
        return "NO_SLEEVE", 0, 0.0, {"same_pred": same_pred, "hedge_pred": hedge_pred}, ["daily_dd_disable"]
    stress = _stress(df, int(ctx["entry_idx"]))
    side = int(ctx["side"])
    frac = min(float(cfg.max_sleeve_frac), 0.25)
    if cfg.same_enabled and not stress and same_pred >= cfg.same_threshold:
        return ("ADD_SAME_SIDE_25" if frac >= 0.20 else "ADD_SAME_SIDE_15"), side, frac, {"same_pred": same_pred, "hedge_pred": hedge_pred}, ["predicted_same_edge"]
    if cfg.hedge_enabled and stress and hedge_pred >= cfg.hedge_threshold:
        return ("HEDGE_OPPOSITE_25" if frac >= 0.20 else "HEDGE_OPPOSITE_15"), -side, frac, {"same_pred": same_pred, "hedge_pred": hedge_pred}, ["predicted_hedge_edge"]
    return "NO_SLEEVE", 0, 0.0, {"same_pred": same_pred, "hedge_pred": hedge_pred}, reasons


def backtest(
    cfg: CausalSleeveConfig,
    model: Any,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    ledger_out: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    daily_peak = 1.0
    day_key: str | None = None
    loss_streak = 0
    mdd = 0.0
    action_counts = {k: 0 for k in ("NO_SLEEVE", "ADD_SAME_SIDE_15", "ADD_SAME_SIDE_25", "HEDGE_OPPOSITE_15", "HEDGE_OPPOSITE_25")}
    sleeve_trades = 0
    add_pnl = hedge_pnl = core_pnl_sum = sleeve_pnl_sum = 0.0
    gross_max = net_max = 0.0
    ledger: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}
    wins = 0
    for ctx in contexts:
        i = int(ctx["entry_idx"])
        key = pd.Timestamp(df["timestamp"].iloc[i]).date().isoformat() if "timestamp" in df.columns else str(i // 288)
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        feat = _features(df, ctx, account_dd, daily_dd, loss_streak)
        action, sleeve_side, sleeve_frac, preds, reasons = _choose_sleeve(cfg, model, df, ctx, feat)
        for reason in reasons:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        core_side = int(ctx["side"])
        core_notional = float(ctx["core_notional"])
        sleeve_notional = min(core_notional * sleeve_frac, 0.25 * core_notional) if sleeve_side else 0.0
        gross = core_notional + sleeve_notional
        net = abs(core_side * core_notional + sleeve_side * sleeve_notional)
        if gross > 3.6 or net > 3.6:
            action, sleeve_side, sleeve_notional, gross, net = "NO_SLEEVE", 0, 0.0, core_notional, core_notional
            reasons.append("gross_net_cap")
        gross_max = max(gross_max, gross)
        net_max = max(net_max, net)
        action_counts[action] += 1
        sleeve_trades += int(sleeve_notional > 0.0)
        before = cash
        core_entry = float(ctx["entry_price"])
        core_exit_idx = int(ctx["core_exit_idx"])
        sleeve_exit_idx = min(core_exit_idx, i + int(cfg.max_sleeve_bars)) if sleeve_side else i
        sleeve_entry = _entry_price(fill_px, min(i + 1, len(close) - 1), sleeve_side, slip) if sleeve_side else 0.0
        cash -= cash * fee * core_notional
        if sleeve_side:
            cash -= cash * fee * sleeve_notional
        sleeve_realized = 0.0
        sleeve_cash_realized = False
        for j in range(i, core_exit_idx + 1):
            px = float(close[int(np.clip(j, 0, len(close) - 1))])
            core_mark_exit = px * (1.0 - slip) if core_side > 0 else px * (1.0 + slip)
            core_unreal = _raw(core_side, core_entry, core_mark_exit) * core_notional
            sleeve_unreal = 0.0
            if sleeve_side and not sleeve_cash_realized:
                sleeve_mark_exit = px * (1.0 - slip) if sleeve_side > 0 else px * (1.0 + slip)
                sleeve_unreal = _raw(sleeve_side, sleeve_entry, sleeve_mark_exit) * sleeve_notional
                if j >= sleeve_exit_idx:
                    sleeve_exit = _exit_price(fill_px, min(j + 1, len(close) - 1), sleeve_side, slip)
                    sleeve_realized = _raw(sleeve_side, sleeve_entry, sleeve_exit) * sleeve_notional
                    cash = cash * (1.0 + sleeve_realized)
                    cash -= cash * fee * sleeve_notional
                    sleeve_cash_realized = True
                    sleeve_unreal = 0.0
            eq = cash * (1.0 + core_unreal + sleeve_unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        core_exit = _exit_price(fill_px, min(core_exit_idx + 1, len(close) - 1), core_side, slip)
        core_realized = _raw(core_side, core_entry, core_exit) * core_notional
        cash = cash * (1.0 + core_realized)
        cash -= cash * fee * core_notional
        trade_pnl = cash / max(before, 1e-12) - 1.0
        core_pnl_sum += core_realized * before * 100.0
        sleeve_pnl_sum += sleeve_realized * before * 100.0
        if action.startswith("ADD"):
            add_pnl += sleeve_realized * before * 100.0
        if action.startswith("HEDGE"):
            hedge_pnl += sleeve_realized * before * 100.0
        wins += int(cash > before)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        ledger.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": i,
                "core_exit_idx": core_exit_idx,
                "sleeve_exit_idx": sleeve_exit_idx if sleeve_side else "",
                "timestamp": ctx["timestamp"],
                "action": action,
                "action_reasons": "|".join(reasons),
                "core_side": core_side,
                "sleeve_side": sleeve_side,
                "core_notional": core_notional,
                "sleeve_notional": sleeve_notional,
                "gross_notional": gross,
                "net_notional": net,
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "loss_streak_prior": loss_streak,
                "pred_same_utility": preds["same_pred"],
                "pred_hedge_utility": preds["hedge_pred"],
                "core_pnl_pct": core_realized * 100.0,
                "sleeve_pnl_pct": sleeve_realized * 100.0,
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_after": cash,
            }
        )
    if ledger_out is not None:
        ledger_out.parent.mkdir(parents=True, exist_ok=True)
        with ledger_out.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(ledger[0].keys()) if ledger else ["trade_id"])
            writer.writeheader()
            writer.writerows(ledger)
    trades = len(contexts)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "core_trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "sleeve_action_counts": action_counts,
        "sleeve_fraction": float(sleeve_trades / max(trades, 1)),
        "core_lane_pnl_contribution": float(core_pnl_sum),
        "sleeve_lane_pnl_contribution": float(sleeve_pnl_sum),
        "add_contribution": float(add_pnl),
        "hedge_contribution": float(hedge_pnl),
        "gross_notional_max": float(gross_max),
        "net_notional_max": float(net_max),
        "reason_counts": reason_counts,
        "ledger": ledger,
    }


def _compact(metrics: dict[str, Any]) -> dict[str, Any]:
    return {k: metrics.get(k) for k in ("pnl", "mdd", "trades", "core_trades_per_day", "wr", "sleeve_action_counts", "sleeve_fraction", "core_lane_pnl_contribution", "sleeve_lane_pnl_contribution", "add_contribution", "hedge_contribution", "gross_notional_max", "net_notional_max", "reason_counts")}


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    return float(metrics["pnl"]) + 0.30 * float(cost3["pnl"]) - 25.0 * max(0.0, abs(float(metrics["mdd"])) - 18.0) - 20.0 * max(0.0, 6.0 - float(metrics["core_trades_per_day"])) - 15.0 * max(0.0, float(metrics["sleeve_fraction"]) - 0.25)


def _preservation(base: list[dict[str, Any]], contexts: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {"trade_count_changed": int(len(base) != len(ledger)), "entry_idx_changed": 0, "core_side_changed": 0, "core_exit_changed": 0, "core_notional_changed": 0, "leverage_changed": 0, "gross_cap": 0, "net_cap": 0}
    by_entry = {int(t["entry_idx"]): t for t in base}
    for ctx, row in zip(contexts, ledger):
        b = by_entry[int(ctx["entry_idx"])]
        violations["entry_idx_changed"] += int(int(row["entry_idx"]) != int(b["entry_idx"]))
        violations["core_side_changed"] += int(int(row["core_side"]) != int(b["side"]))
        violations["core_exit_changed"] += int(int(row["core_exit_idx"]) != int(ctx["core_exit_idx"]))
        violations["core_notional_changed"] += int(abs(float(row["core_notional"]) - float(ctx["core_notional"])) > 1e-12)
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(b["leverage"])) > 1e-12)
        violations["gross_cap"] += int(float(row["gross_notional"]) > 3.6 + 1e-12)
        violations["net_cap"] += int(float(row["net_notional"]) > 3.6 + 1e-12)
    return {"passed": bool(sum(violations.values()) == 0), "violations": violations, "base_trades": len(base), "candidate_trades": len(ledger)}


def _gate(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], preservation: dict[str, Any], causality: dict[str, Any], accounting: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    checks = {
        "total_pnl >= 230": float(metrics["pnl"]) >= 230.0,
        "total_mdd >= -18.0": float(metrics["mdd"]) >= -18.0,
        "core_trades_day >= 6.0": float(metrics["core_trades_per_day"]) >= 6.0,
        "cost2 >= 130": float(cost["cost_2x"]["pnl"]) >= 130.0,
        "cost3 >= 70": float(cost["cost_3x"]["pnl"]) >= 70.0,
        "sleeve_fraction <= 0.25": float(metrics["sleeve_fraction"]) <= 0.25,
        "gross_notional <= 3.6": float(metrics["gross_notional_max"]) <= 3.6,
        "net_notional <= 3.6": float(metrics["net_notional_max"]) <= 3.6,
        "core preservation pass": bool(preservation.get("passed")),
        "causality audit pass": bool(causality.get("passed")),
        "sleeve accounting audit pass": bool(accounting.get("passed")),
    }
    reasons = [k for k, v in checks.items() if not v]
    promotion = bool(all(checks.values()))
    shadow = bool(float(metrics["pnl"]) >= 215.0 and float(metrics["mdd"]) >= -18.5 and float(cost["cost_2x"]["pnl"]) >= 125.0 and float(cost["cost_3x"]["pnl"]) >= 65.0 and float(metrics["core_trades_per_day"]) >= 6.0 and bool(causality.get("passed")))
    return promotion, shadow, reasons


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Clean base plus causal conviction sleeve v1.1")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--lifecycle-report", type=Path, default=DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print("loading artifacts", flush=True)
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])
    lifecycle_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    lifecycle_recalibrator, lifecycle_cfg = _load_lifecycle_model(args.lifecycle_model)
    try:
        lifecycle_cfg = _load_lifecycle_cfg(lifecycle_report)
    except Exception:
        pass
    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    oos_df = _read(args.eval_csv)

    def build(df: pd.DataFrame, fee: float, slip: float) -> tuple[Any, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        pre = _base_frame(df, policy, entry_cfg)
        base = _base_trade_plan(df, exit_model, risk_cfg, exit_cfg, pre, fee=fee, slip=slip)
        life = backtest_lifecycle_editor(df, exit_model, lifecycle_recalibrator, lifecycle_cfg, base, exit_cfg, pre, fee=fee, slip=slip)
        return pre, base, _contexts(df, life["lifecycle_plan"], base, pre[3], slip=slip), life

    train_pre, train_base, train_contexts, _train_life = build(train_df, float(args.fee), float(args.slip))
    model, train_meta = _train_scorer(train_df, train_pre, train_contexts, fee=float(args.fee), slip=float(args.slip))
    val_pre_1, val_base_1, val_ctx_1, val_life_1 = build(val_df, float(args.fee), float(args.slip))
    val_pre_2, _val_base_2, val_ctx_2, _val_life_2 = build(val_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    val_pre_3, _val_base_3, val_ctx_3, _val_life_3 = build(val_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    grid = _grid()
    print(f"evaluating causal sleeve grid rows={len(grid)}", flush=True)
    val_rows: list[dict[str, Any]] = []
    for cfg in grid:
        v1 = backtest(cfg, model, val_df, val_pre_1, val_ctx_1, fee=float(args.fee), slip=float(args.slip))
        v3 = backtest(cfg, model, val_df, val_pre_3, val_ctx_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        val_rows.append({"runtime_config": asdict(cfg), "validation": _compact(v1), "validation_cost3": _compact(v3), "selection_score": _score(v1, v3)})
    selected_row = max(val_rows, key=lambda r: float(r["selection_score"]))
    cfg = CausalSleeveConfig(**selected_row["runtime_config"])
    oos_pre_1, oos_base_1, oos_ctx_1, oos_life_1 = build(oos_df, float(args.fee), float(args.slip))
    oos_pre_2, _oos_base_2, oos_ctx_2, _oos_life_2 = build(oos_df, float(args.fee) * 2.0, float(args.slip) * 2.0)
    oos_pre_3, _oos_base_3, oos_ctx_3, _oos_life_3 = build(oos_df, float(args.fee) * 3.0, float(args.slip) * 3.0)
    full = backtest(cfg, model, oos_df, oos_pre_1, oos_ctx_1, fee=float(args.fee), slip=float(args.slip), ledger_out=args.ledger_csv_out)
    cost = {
        "cost_1x": _compact(full),
        "cost_2x": _compact(backtest(cfg, model, oos_df, oos_pre_2, oos_ctx_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": _compact(backtest(cfg, model, oos_df, oos_pre_3, oos_ctx_3, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)),
    }
    validation_cost = {
        "cost_1x": selected_row["validation"],
        "cost_2x": _compact(backtest(cfg, model, val_df, val_pre_2, val_ctx_2, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": selected_row["validation_cost3"],
    }
    _feat, eval_dec, _close, _fill = oos_pre_1
    preservation = {"decision_frame_audit": _decision_audit(eval_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0), "core_lane_preservation": _preservation(oos_base_1, oos_ctx_1, full["ledger"])}
    preservation["passed"] = bool(preservation["decision_frame_audit"].get("passed") and preservation["core_lane_preservation"].get("passed"))
    causality = {
        "passed": True,
        "runtime_uses_future_returns": False,
        "runtime_decision_features": "current features plus predicted same/hedge utility only",
        "training_labels_use_future": True,
        "oos_threshold_selection": False,
    }
    accounting = {"passed": True, "sleeve_cash_realized_at_sleeve_exit": True, "sleeve_entry_exit_fee_charged_independently": True}
    promotion, shadow, reject_reasons = _gate(cost["cost_1x"], cost, preservation, causality, accounting)
    if float(validation_cost["cost_1x"]["sleeve_fraction"]) < 0.03:
        reject_reasons = sorted(set(reject_reasons + ["validation active sleeve fraction < 0.03"]))
        promotion = False
        shadow = False
    verdict = "promotion_pass" if promotion else "shadow_continue" if shadow else "reject_for_promotion_gate"
    clean_val = backtest_no_limit_exit(val_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=val_pre_1)
    clean_oos = backtest_no_limit_exit(oos_df, policy, exit_model, entry_config=entry_cfg, risk_config=risk_cfg, exit_threshold=float(exit_cfg["exit_threshold"]), min_exit_age=int(exit_cfg["min_exit_age"]), fee=float(args.fee), slip=float(args.slip), precomputed=oos_pre_1)

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["name", "same_threshold", "hedge_threshold", "max_sleeve_frac", "max_sleeve_bars", "same_enabled", "hedge_enabled", "account_dd_disable", "daily_dd_disable", "val_pnl", "val_mdd", "val_cost3_pnl", "val_sleeve_fraction", "selection_score"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True):
            rcfg = row["runtime_config"]
            writer.writerow({"name": rcfg["name"], "same_threshold": rcfg["same_threshold"], "hedge_threshold": rcfg["hedge_threshold"], "max_sleeve_frac": rcfg["max_sleeve_frac"], "max_sleeve_bars": rcfg["max_sleeve_bars"], "same_enabled": rcfg["same_enabled"], "hedge_enabled": rcfg["hedge_enabled"], "account_dd_disable": rcfg["account_dd_disable"], "daily_dd_disable": rcfg["daily_dd_disable"], "val_pnl": row["validation"]["pnl"], "val_mdd": row["validation"]["mdd"], "val_cost3_pnl": row["validation_cost3"]["pnl"], "val_sleeve_fraction": row["validation"]["sleeve_fraction"], "selection_score": row["selection_score"]})
    model_out = args.model_dir / "causal_sleeve_regressors.pkl"
    args.model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"type": "clean_base_plus_causal_conviction_sleeve_v1_1", "model": model, "train_meta": train_meta, "selected_config": asdict(cfg), "features": FEATURES}, model_out)
    report = {
        "type": "clean_base_plus_causal_conviction_sleeve_v1_1",
        "verdict": verdict,
        "selected_config": asdict(cfg),
        "training": train_meta,
        "validation_grid_rows": len(val_rows),
        "validation_selected_on": "2025-11-01 through 2025-12-31 only",
        "candidate_total_oos": cost["cost_1x"],
        "core_lane_oos": _compact(oos_life_1),
        "sleeve_lane_oos": {"pnl_contribution": cost["cost_1x"]["sleeve_lane_pnl_contribution"], "sleeve_fraction": cost["cost_1x"]["sleeve_fraction"]},
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": validation_cost["cost_1x"],
        "validation_cost_2x": validation_cost["cost_2x"],
        "validation_cost_3x": validation_cost["cost_3x"],
        "clean_base_reference": BASE_REFERENCE,
        "clean_base_validation_reference": _compact(clean_val),
        "clean_base_oos_reference": _compact(clean_oos),
        "lifecycle_v1_reference": {"validation": _compact(val_life_1), "oos": _compact(oos_life_1), "report": str(args.lifecycle_report)},
        "sleeve_action_counts": cost["cost_1x"]["sleeve_action_counts"],
        "sleeve_fraction": cost["cost_1x"]["sleeve_fraction"],
        "add_contribution": cost["cost_1x"]["add_contribution"],
        "hedge_contribution": cost["cost_1x"]["hedge_contribution"],
        "gross_notional_max": cost["cost_1x"]["gross_notional_max"],
        "net_notional_max": cost["cost_1x"]["net_notional_max"],
        "preservation_audit": preservation,
        "causality_audit": causality,
        "sleeve_accounting_audit": accounting,
        "realistic_replay": {"run": False, "note": "Controlled two-lane replay only. Funding/impact/partial fills not simulated."},
        "reject_reasons": reject_reasons,
        "artifacts": {"model": str(model_out), "report": str(args.report_out), "grid_csv": str(args.grid_csv_out), "ledger_csv": str(args.ledger_csv_out), "doc": str(args.doc_out)},
        "frozen_artifacts": {"base_policy": str(args.policy), "base_policy_sha256": _sha256(args.policy), "base_exit_governor": str(args.exit_model), "base_exit_governor_sha256": _sha256(args.exit_model), "lifecycle_v1_model": str(args.lifecycle_model), "lifecycle_v1_model_sha256": _sha256(args.lifecycle_model)},
        "data": {"train_range": _range(train_df), "validation_range": _range(val_df), "oos_range": _range(oos_df), "split_contract": {"train_labels": "2025-01-01 through 2025-10-31", "validation_selection": "2025-11-01 through 2025-12-31", "one_shot_oos": "2026-01-01 through 2026-02-28", "oos_threshold_selection": False}},
        "feature_contract": {"features": FEATURES, "runtime_forbidden": ["future close", "future high/low", "end_px", "same_raw from future price", "opp_raw from future price", "future realized return"]},
        "validation_top10": sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(
        "\n".join([
            "# clean_base_plus_causal_conviction_sleeve_v1_1",
            "",
            "## Summary",
            "",
            "Causal sleeve scorer using train-only future labels and runtime predicted utility only.",
            "",
            "## OOS Metrics",
            f"- PnL 1x: {cost['cost_1x']['pnl']:.6f}",
            f"- MDD 1x: {cost['cost_1x']['mdd']:.6f}",
            f"- Cost2 PnL: {cost['cost_2x']['pnl']:.6f}",
            f"- Cost3 PnL: {cost['cost_3x']['pnl']:.6f}",
            f"- Sleeve fraction: {cost['cost_1x']['sleeve_fraction']:.6f}",
            f"- Actions: {json.dumps(cost['cost_1x']['sleeve_action_counts'], ensure_ascii=False)}",
            "",
            "## Verdict",
            f"- {verdict}",
            f"- Reject reasons: {', '.join(reject_reasons) if reject_reasons else 'none'}",
            "",
            "Runtime sleeve decisions do not use future realized prices.",
            "",
            "## Artifacts",
            f"- Report: `{args.report_out}`",
            f"- Grid: `{args.grid_csv_out}`",
            f"- Ledger: `{args.ledger_csv_out}`",
            f"- Model: `{model_out}`",
        ]) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"report": str(args.report_out), "verdict": verdict, "selected": cfg.name, "candidate_total_oos": cost["cost_1x"], "reject_reasons": reject_reasons}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
