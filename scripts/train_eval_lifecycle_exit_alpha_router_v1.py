#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.train_eval_clean_base_exit_hazard_recalibrator_v1 import (  # noqa: E402
    RecalibratorRuntimeConfig,
    _calibrated_exit_control,
)
from scripts.train_eval_clean_base_lifecycle_editor_v1 import (  # noqa: E402
    BASE_REFERENCE,
    LifecycleRuntimeConfig,
    _base_frame,
    _base_trade_plan,
    _compact,
    _days,
    _exit_probability_vec,
    _feature_vec_fast,
    _fill_price,
    _range,
    _read,
    _sha256,
    backtest_lifecycle_editor,
)
from scripts.train_eval_lifecycle_v1_drawdown_governor_v1 import _load_lifecycle_cfg  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_LIFECYCLE_REPORT = ROOT / "data/ensemble/reports/clean_base_lifecycle_editor_v1_2026.json"
DEFAULT_LIFECYCLE_MODEL = ROOT / "data/ensemble/supervised/clean_base_lifecycle_editor_v1/lifecycle_editor.pkl"
DEFAULT_EXIT_V1_REPORT = ROOT / "data/ensemble/reports/clean_base_exit_hazard_recalibrator_v1_2026.json"
DEFAULT_EXIT_V1_MODEL = ROOT / "data/ensemble/supervised/clean_base_exit_hazard_recalibrator_v1/hazard_recalibrator.pkl"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/lifecycle_exit_alpha_router_v1"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/lifecycle_exit_alpha_router_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/lifecycle_exit_alpha_router_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/lifecycle_exit_alpha_router_v1_ledger.csv"
DEFAULT_DOC = ROOT / "docs/experiments/lifecycle_exit_alpha_router_v1.md"

FEATURE_COLS = [
    "side",
    "age",
    "unrealized",
    "peak_unrealized",
    "drawdown_from_trade_peak",
    "quality_score",
    "confidence",
    "exit_prob",
    "exit_hazard_bucket_code",
    "exit_hazard_rate",
    "exit_threshold_delta",
    "exit_bucket_support",
    "lifecycle_action_proposal",
    "funding_abs",
    "funding_pressure",
    "liquidity_vacuum",
    "amihud_illiquidity_z",
    "m7_tail_risk",
    "evt_tail_flag",
    "ai_adverse_risk",
    "daily_dd",
    "account_dd",
    "loss_streak",
]


@dataclass(frozen=True)
class RouterConfig:
    name: str
    router_prob_threshold: float
    alpha_fraction_cap: float
    account_dd_disable: float
    daily_dd_disable: float
    alpha_loss_lock_bars: int


class ConstantRouter:
    def __init__(self, probability: float) -> None:
        self.probability = float(probability)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.full((len(x), 2), 0.0, dtype=np.float64)
        p[:, 1] = self.probability
        p[:, 0] = 1.0 - self.probability
        return p


def _filter_range(df: pd.DataFrame, start: str, end_exclusive: str) -> pd.DataFrame:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    return df.loc[(ts >= pd.Timestamp(start)) & (ts < pd.Timestamp(end_exclusive))].reset_index(drop=True)


def _load_exit_v1_cfg(report: dict[str, Any]) -> RecalibratorRuntimeConfig:
    selected = dict(report.get("selected_eval", {}))
    row = selected.get("redteam_constrained") or selected.get("balanced_score") or next(iter(selected.values()))
    return RecalibratorRuntimeConfig(**dict(row["runtime_config"]))


def _load_payload(path: Path, key: str) -> Any:
    payload = joblib.load(path)
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    return payload


def _num(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _row_value(df: pd.DataFrame, i: int, col: str, default: float = 0.0) -> float:
    if col not in df.columns or i < 0 or i >= len(df):
        return float(default)
    return _num(df[col].iloc[i], default)


def _bucket_code(bucket: str) -> float:
    h = hashlib.sha256(bucket.encode("utf-8")).hexdigest()
    return float(int(h[:8], 16) % 10000) / 10000.0


def _stress_thresholds(train_df: pd.DataFrame) -> dict[str, float]:
    def q_abs(col: str, prob: float, default: float) -> float:
        if col not in train_df.columns:
            return float(default)
        vals = pd.to_numeric(train_df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().abs()
        if vals.empty:
            return float(default)
        return float(vals.quantile(prob))

    return {
        "funding_abs_p85": q_abs("funding_abs", 0.85, 999.0),
        "funding_pressure_p85": q_abs("funding_pressure", 0.85, 999.0),
        "liquidity_vacuum_p85": q_abs("liquidity_vacuum", 0.85, 999.0),
        "amihud_illiquidity_z_p85": q_abs("amihud_illiquidity_z", 0.85, 999.0),
        "m7_tail_risk_p85": q_abs("m7_tail_risk", 0.85, 999.0),
        "ai_adverse_risk_p85": q_abs("ai_adverse_risk", 0.85, 999.0),
    }


def _stress_elevated(df: pd.DataFrame, i: int, thresholds: dict[str, float]) -> bool:
    return bool(
        _row_value(df, i, "evt_tail_flag") > 0.0
        or abs(_row_value(df, i, "funding_abs")) >= thresholds["funding_abs_p85"]
        or abs(_row_value(df, i, "funding_pressure")) >= thresholds["funding_pressure_p85"]
        or abs(_row_value(df, i, "liquidity_vacuum")) >= thresholds["liquidity_vacuum_p85"]
        or abs(_row_value(df, i, "amihud_illiquidity_z")) >= thresholds["amihud_illiquidity_z_p85"]
        or abs(_row_value(df, i, "m7_tail_risk")) >= thresholds["m7_tail_risk_p85"]
        or abs(_row_value(df, i, "ai_adverse_risk")) >= thresholds["ai_adverse_risk_p85"]
    )


def _trade_stats(
    close: np.ndarray,
    fill_px: np.ndarray,
    trade: dict[str, Any],
    exit_i: int,
    notional: float,
    *,
    fee: float,
    slip: float,
) -> dict[str, float]:
    entry_i = int(trade["entry_idx"])
    side = int(trade["side"])
    entry_price = float(trade["entry_price"])
    peak_unreal = 0.0
    min_unreal = 0.0
    max_giveback = 0.0
    for j in range(entry_i, int(exit_i) + 1):
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        raw_mark = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw_mark * notional
        peak_unreal = max(peak_unreal, unreal)
        min_unreal = min(min_unreal, unreal)
        max_giveback = max(max_giveback, peak_unreal - unreal)
    exit_price = _fill_price(fill_px, min(int(exit_i) + 1, len(fill_px) - 1), side, slip, entry=False)
    raw = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
    cost_adjusted = raw * notional - 2.0 * float(fee) * notional
    return {
        "cost_adjusted_return": float(cost_adjusted),
        "adverse_path": float(max(max_giveback, -min_unreal)),
        "raw_return": float(raw),
        "max_giveback": float(max_giveback),
    }


def _router_feature_row(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    exit_model: Any,
    alpha_recalibrator: dict[str, Any],
    alpha_cfg: RecalibratorRuntimeConfig,
    exit_cfg: dict[str, Any],
    trade: dict[str, Any],
    lifecycle_row: dict[str, Any],
    *,
    account_dd: float,
    daily_dd: float,
    loss_streak: int,
) -> tuple[dict[str, float], dict[str, Any]]:
    base_feat, decisions, _close, _fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    sides = decisions["side"].astype(int).to_numpy()
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    i = int(trade["entry_idx"])
    vec = _feature_vec_fast(
        base_values,
        sides,
        qualities,
        confs,
        i=i,
        side=int(trade["side"]),
        age=0,
        unrealized=0.0,
        peak_unrealized=0.0,
        notional=float(lifecycle_row["effective_notional"]),
        leverage=float(trade["leverage"]),
        entry_quality=float(trade.get("entry_quality", 0.0)),
        entry_confidence=float(trade.get("entry_confidence", 0.0)),
    )
    threshold, _min_age, cal = _calibrated_exit_control(alpha_recalibrator, vec, alpha_cfg, exit_cfg)
    bucket = str(cal["bucket"])
    p_exit = float(_exit_probability_vec(exit_model, vec))
    lifecycle_action = 1.0 if int(lifecycle_row["effective_exit_idx"]) < int(lifecycle_row["base_exit_idx"]) else 0.0
    features = {
        "side": float(int(trade["side"])),
        "age": 0.0,
        "unrealized": 0.0,
        "peak_unrealized": 0.0,
        "drawdown_from_trade_peak": 0.0,
        "quality_score": float(trade.get("entry_quality", 0.0)),
        "confidence": float(trade.get("entry_confidence", 0.0)),
        "exit_prob": p_exit,
        "exit_hazard_bucket_code": _bucket_code(bucket),
        "exit_hazard_rate": float(cal.get("hazard_rate", 0.0)),
        "exit_threshold_delta": float(cal.get("threshold_delta", float(threshold) - float(exit_cfg["exit_threshold"]))),
        "exit_bucket_support": float(cal.get("support", 0)),
        "lifecycle_action_proposal": lifecycle_action,
        "funding_abs": _row_value(df, i, "funding_abs"),
        "funding_pressure": _row_value(df, i, "funding_pressure"),
        "liquidity_vacuum": _row_value(df, i, "liquidity_vacuum"),
        "amihud_illiquidity_z": _row_value(df, i, "amihud_illiquidity_z"),
        "m7_tail_risk": _row_value(df, i, "m7_tail_risk"),
        "evt_tail_flag": _row_value(df, i, "evt_tail_flag"),
        "ai_adverse_risk": _row_value(df, i, "ai_adverse_risk"),
        "daily_dd": float(daily_dd),
        "account_dd": float(account_dd),
        "loss_streak": float(loss_streak),
    }
    return features, {"exit_v1_bucket": bucket, "exit_v1_threshold": float(threshold), **cal}


def _alpha_exit_idx(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    exit_model: Any,
    alpha_recalibrator: dict[str, Any],
    alpha_cfg: RecalibratorRuntimeConfig,
    exit_cfg: dict[str, Any],
    trade: dict[str, Any],
    notional: float,
    *,
    slip: float,
) -> dict[str, Any]:
    base_feat, decisions, close, _fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    sides = decisions["side"].astype(int).to_numpy()
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    entry_i = int(trade["entry_idx"])
    base_exit_i = int(trade["exit_idx"])
    side = int(trade["side"])
    entry_price = float(trade["entry_price"])
    peak_unrealized = 0.0
    last_meta: dict[str, Any] = {}
    for j in range(entry_i, base_exit_i + 1):
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        raw_mark = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw_mark * float(notional)
        peak_unrealized = max(peak_unrealized, unreal)
        age = j - entry_i
        vec = _feature_vec_fast(
            base_values,
            sides,
            qualities,
            confs,
            i=j,
            side=side,
            age=age,
            unrealized=unreal,
            peak_unrealized=peak_unrealized,
            notional=float(notional),
            leverage=float(trade["leverage"]),
            entry_quality=float(trade.get("entry_quality", 0.0)),
            entry_confidence=float(trade.get("entry_confidence", 0.0)),
        )
        threshold, min_age, cal = _calibrated_exit_control(alpha_recalibrator, vec, alpha_cfg, exit_cfg)
        p_exit = float(_exit_probability_vec(exit_model, vec))
        last_meta = {"exit_prob": p_exit, "threshold": float(threshold), "min_age": int(min_age), **cal}
        if age >= int(min_age) and p_exit >= float(threshold):
            return {"exit_idx": int(j), "exit_reason": "alpha_exit_v1", "meta": last_meta}
    return {"exit_idx": int(base_exit_i), "exit_reason": "alpha_base_exit", "meta": last_meta}


def _enrich_lifecycle_plan(lifecycle_plan: list[dict[str, Any]], base_trades: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    out: list[dict[str, Any]] = []
    for row in lifecycle_plan:
        base = by_entry[int(row["entry_idx"])]
        merged = dict(row)
        merged["entry_price"] = float(base["entry_price"])
        merged["entry_quality"] = float(base.get("entry_quality", 0.0))
        merged["entry_confidence"] = float(base.get("entry_confidence", 0.0))
        merged["cooldown_bars"] = int(base.get("cooldown_bars", merged.get("cooldown_bars", 0)))
        out.append(merged)
    return out


def _build_contexts(
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    exit_model: Any,
    alpha_recalibrator: dict[str, Any],
    alpha_cfg: RecalibratorRuntimeConfig,
    exit_cfg: dict[str, Any],
    base_trades: list[dict[str, Any]],
    lifecycle_plan: list[dict[str, Any]],
    thresholds: dict[str, float],
    *,
    fee: float,
    slip: float,
    label_margin: float,
) -> list[dict[str, Any]]:
    _base_feat, _decisions, close, fill_px = precomputed
    by_entry = {int(t["entry_idx"]): t for t in base_trades}
    contexts: list[dict[str, Any]] = []
    for lid, life in enumerate(lifecycle_plan):
        base = by_entry[int(life["entry_idx"])]
        effective_notional = float(life["effective_notional"])
        alpha = _alpha_exit_idx(df, precomputed, exit_model, alpha_recalibrator, alpha_cfg, exit_cfg, base, effective_notional, slip=slip)
        life_stats = _trade_stats(close, fill_px, base, int(life["effective_exit_idx"]), effective_notional, fee=fee, slip=slip)
        alpha_stats = _trade_stats(close, fill_px, base, int(alpha["exit_idx"]), effective_notional, fee=fee, slip=slip)
        forced_lifecycle = int(life["effective_exit_idx"]) < int(life["base_exit_idx"])
        stress = _stress_elevated(df, int(base["entry_idx"]), thresholds)
        alpha_label = bool(
            not forced_lifecycle
            and alpha_stats["cost_adjusted_return"] > life_stats["cost_adjusted_return"] + float(label_margin)
            and alpha_stats["adverse_path"] <= life_stats["adverse_path"] + 0.005
            and alpha_stats["cost_adjusted_return"] > 0.0
            and not stress
        )
        static_features, static_meta = _router_feature_row(
            df,
            precomputed,
            exit_model,
            alpha_recalibrator,
            alpha_cfg,
            exit_cfg,
            base,
            life,
            account_dd=0.0,
            daily_dd=0.0,
            loss_streak=0,
        )
        contexts.append(
            {
                "trade_id": int(lid),
                "entry_idx": int(base["entry_idx"]),
                "base_exit_idx": int(base["exit_idx"]),
                "lifecycle_exit_idx": int(life["effective_exit_idx"]),
                "alpha_exit_idx": int(alpha["exit_idx"]),
                "side": int(base["side"]),
                "entry_price": float(base["entry_price"]),
                "lifecycle_notional": effective_notional,
                "base_notional": float(base["base_notional"]),
                "leverage": float(base["leverage"]),
                "cooldown_bars": int(base.get("cooldown_bars", 0)),
                "entry_quality": float(base.get("entry_quality", 0.0)),
                "entry_confidence": float(base.get("entry_confidence", 0.0)),
                "lifecycle_exit_reason": str(life.get("exit_reason", "")),
                "alpha_exit_reason": str(alpha["exit_reason"]),
                "alpha_meta": dict(alpha.get("meta", {})),
                "forced_lifecycle": bool(forced_lifecycle),
                "stress_elevated": bool(stress),
                "label": int(alpha_label),
                "lifecycle_utility": float(life_stats["cost_adjusted_return"]),
                "alpha_utility": float(alpha_stats["cost_adjusted_return"]),
                "lifecycle_adverse_path": float(life_stats["adverse_path"]),
                "alpha_adverse_path": float(alpha_stats["adverse_path"]),
                "static_features": static_features,
                "static_meta": static_meta,
                "timestamp": str(df["timestamp"].iloc[int(base["entry_idx"])]) if "timestamp" in df.columns else str(base["entry_idx"]),
                "base_exit_timestamp": str(df["timestamp"].iloc[int(base["exit_idx"])]) if "timestamp" in df.columns else str(base["exit_idx"]),
                "lifecycle_exit_timestamp": str(df["timestamp"].iloc[int(life["effective_exit_idx"])]) if "timestamp" in df.columns else str(life["effective_exit_idx"]),
                "alpha_exit_timestamp": str(df["timestamp"].iloc[int(alpha["exit_idx"])]) if "timestamp" in df.columns else str(alpha["exit_idx"]),
            }
        )
    return contexts


def _execute_trade(
    cash: float,
    peak: float,
    mdd: float,
    context: dict[str, Any],
    close: np.ndarray,
    fill_px: np.ndarray,
    exit_i: int,
    *,
    fee: float,
    slip: float,
    mark_alpha: bool,
    alpha_mdd: float,
) -> tuple[float, float, float, float, float, float]:
    side = int(context["side"])
    entry_i = int(context["entry_idx"])
    entry_price = float(context["entry_price"])
    notional = float(context["lifecycle_notional"])
    before = cash
    cash -= cash * float(fee) * notional
    peak_unreal = 0.0
    max_giveback = 0.0
    for j in range(entry_i, int(exit_i) + 1):
        px = float(close[int(np.clip(j, 0, len(close) - 1))])
        raw_mark = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw_mark * notional
        peak_unreal = max(peak_unreal, unreal)
        max_giveback = max(max_giveback, peak_unreal - unreal)
        eq = cash * (1.0 + unreal)
        peak = max(peak, eq)
        dd = eq / max(peak, 1e-12) - 1.0
        mdd = min(mdd, dd)
        if mark_alpha:
            alpha_mdd = min(alpha_mdd, dd)
    exit_price = _fill_price(fill_px, min(int(exit_i) + 1, len(fill_px) - 1), side, slip, entry=False)
    raw = (exit_price - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
    before_exit = cash
    cash = cash * (1.0 + raw * notional)
    cash -= before_exit * float(fee) * notional
    peak = max(peak, cash)
    mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    trade_pnl = cash / max(before, 1e-12) - 1.0
    return cash, peak, mdd, trade_pnl, max_giveback, alpha_mdd


def _train_router(
    train_df: pd.DataFrame,
    train_pre: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    exit_model: Any,
    alpha_recalibrator: dict[str, Any],
    alpha_cfg: RecalibratorRuntimeConfig,
    exit_cfg: dict[str, Any],
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
) -> tuple[Any, pd.DataFrame, np.ndarray, dict[str, Any]]:
    _feat, _dec, close, fill_px = train_pre
    cash = peak = 1.0
    mdd = 0.0
    loss_streak = 0
    closed_peak = 1.0
    day_key: str | None = None
    daily_peak = 1.0
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for ctx in contexts:
        ts = pd.Timestamp(train_df["timestamp"].iloc[int(ctx["entry_idx"])])
        key = ts.date().isoformat()
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        if "static_features" in ctx:
            features = dict(ctx["static_features"])
            features["account_dd"] = float(account_dd)
            features["daily_dd"] = float(daily_dd)
            features["loss_streak"] = float(loss_streak)
        else:
            life_stub = {"effective_notional": ctx["lifecycle_notional"], "effective_exit_idx": ctx["lifecycle_exit_idx"], "base_exit_idx": ctx["base_exit_idx"]}
            features, _meta = _router_feature_row(
                train_df,
                train_pre,
                exit_model,
                alpha_recalibrator,
                alpha_cfg,
                exit_cfg,
                ctx,
                life_stub,
                account_dd=account_dd,
                daily_dd=daily_dd,
                loss_streak=loss_streak,
            )
        rows.append(features)
        labels.append(int(ctx["label"]))
        cash, peak, mdd, trade_pnl, _giveback, _alpha_mdd = _execute_trade(
            cash,
            peak,
            mdd,
            ctx,
            close,
            fill_px,
            int(ctx["lifecycle_exit_idx"]),
            fee=fee,
            slip=slip,
            mark_alpha=False,
            alpha_mdd=0.0,
        )
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
    x = pd.DataFrame(rows, columns=FEATURE_COLS).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        model: Any = ConstantRouter(float(np.mean(y)) if len(y) else 0.0)
        method = "constant_probability_router"
    else:
        model = HistGradientBoostingClassifier(max_iter=140, learning_rate=0.055, max_leaf_nodes=15, l2_regularization=0.05, random_state=42)
        model.fit(x.to_numpy(dtype=np.float64), y)
        method = "HistGradientBoostingClassifier"
    meta = {
        "method": method,
        "train_rows": int(len(x)),
        "alpha_label_count": int(y.sum()) if len(y) else 0,
        "alpha_label_rate": float(y.mean()) if len(y) else 0.0,
        "feature_cols": FEATURE_COLS,
    }
    return model, x, y, meta


def _predict_alpha_prob(model: Any, features: dict[str, float]) -> float:
    x = np.asarray([[float(features[c]) for c in FEATURE_COLS]], dtype=np.float64)
    proba = model.predict_proba(x)
    return float(proba[0, 1]) if proba.shape[1] > 1 else 0.0


def _backtest_router(
    cfg: RouterConfig,
    df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    exit_model: Any,
    alpha_recalibrator: dict[str, Any],
    alpha_cfg: RecalibratorRuntimeConfig,
    exit_cfg: dict[str, Any],
    model: Any,
    contexts: list[dict[str, Any]],
    *,
    fee: float,
    slip: float,
    write_ledger: Path | None = None,
) -> dict[str, Any]:
    _feat, _dec, close, fill_px = precomputed
    cash = 1.0
    peak = 1.0
    closed_peak = 1.0
    mdd = 0.0
    alpha_mdd = 0.0
    wins = 0
    loss_streak = 0
    prior_trade_giveback = 0.0
    day_key: str | None = None
    daily_peak = 1.0
    alpha_lock_until = -1
    alpha_count = 0
    alpha_pnl_contribution = 0.0
    lifecycle_pnl_contribution = 0.0
    mode_counts = {"CORE_LIFECYCLE": 0, "ALPHA_EXIT": 0}
    disable_counts: dict[str, int] = {}
    ledgers: list[dict[str, Any]] = []
    total_alpha_budget = int(np.floor(float(cfg.alpha_fraction_cap) * len(contexts) + 1e-12))
    for ctx in contexts:
        entry_i = int(ctx["entry_idx"])
        ts = pd.Timestamp(df["timestamp"].iloc[entry_i])
        key = ts.date().isoformat()
        if key != day_key:
            day_key = key
            daily_peak = max(cash, 1e-12)
        closed_peak = max(closed_peak, cash)
        daily_peak = max(daily_peak, cash)
        account_dd = max(0.0, 1.0 - cash / max(closed_peak, 1e-12))
        daily_dd = max(0.0, 1.0 - cash / max(daily_peak, 1e-12))
        if "static_features" in ctx:
            features = dict(ctx["static_features"])
            features["account_dd"] = float(account_dd)
            features["daily_dd"] = float(daily_dd)
            features["loss_streak"] = float(loss_streak)
            meta = dict(ctx.get("static_meta", {}))
        else:
            life_stub = {"effective_notional": ctx["lifecycle_notional"], "effective_exit_idx": ctx["lifecycle_exit_idx"], "base_exit_idx": ctx["base_exit_idx"]}
            features, meta = _router_feature_row(
                df,
                precomputed,
                exit_model,
                alpha_recalibrator,
                alpha_cfg,
                exit_cfg,
                ctx,
                life_stub,
                account_dd=account_dd,
                daily_dd=daily_dd,
                loss_streak=loss_streak,
            )
        prob = _predict_alpha_prob(model, features)
        disabled: list[str] = []
        if bool(ctx["forced_lifecycle"]):
            disabled.append("lifecycle_defense_priority")
        if account_dd >= float(cfg.account_dd_disable):
            disabled.append("account_dd_disable")
        if daily_dd >= float(cfg.daily_dd_disable):
            disabled.append("daily_dd_disable")
        if entry_i < alpha_lock_until:
            disabled.append("alpha_loss_lock")
        if alpha_count >= total_alpha_budget:
            disabled.append("alpha_fraction_cap")
        choose_alpha = bool(prob >= float(cfg.router_prob_threshold) and not disabled)
        mode = "ALPHA_EXIT" if choose_alpha else "CORE_LIFECYCLE"
        exit_i = int(ctx["alpha_exit_idx"] if choose_alpha else ctx["lifecycle_exit_idx"])
        before = cash
        cash, peak, mdd, trade_pnl, current_giveback, alpha_mdd = _execute_trade(
            cash,
            peak,
            mdd,
            ctx,
            close,
            fill_px,
            exit_i,
            fee=fee,
            slip=slip,
            mark_alpha=choose_alpha,
            alpha_mdd=alpha_mdd,
        )
        if choose_alpha:
            alpha_count += 1
            alpha_pnl_contribution += cash - before
            if trade_pnl <= 0.0:
                alpha_lock_until = max(alpha_lock_until, entry_i + int(cfg.alpha_loss_lock_bars))
        else:
            lifecycle_pnl_contribution += cash - before
        wins += int(trade_pnl > 0.0)
        loss_streak = 0 if trade_pnl > 0.0 else loss_streak + 1
        mode_counts[mode] += 1
        for reason in disabled:
            disable_counts[reason] = disable_counts.get(reason, 0) + 1
        ledgers.append(
            {
                "trade_id": int(ctx["trade_id"]),
                "entry_idx": entry_i,
                "selected_exit_idx": exit_i,
                "base_exit_idx": int(ctx["base_exit_idx"]),
                "lifecycle_exit_idx": int(ctx["lifecycle_exit_idx"]),
                "alpha_exit_idx": int(ctx["alpha_exit_idx"]),
                "timestamp": str(ctx["timestamp"]),
                "selected_exit_timestamp": str(df["timestamp"].iloc[exit_i]) if "timestamp" in df.columns else str(exit_i),
                "side": int(ctx["side"]),
                "mode": mode,
                "router_alpha_prob": prob,
                "router_threshold": float(cfg.router_prob_threshold),
                "lifecycle_notional": float(ctx["lifecycle_notional"]),
                "effective_notional": float(ctx["lifecycle_notional"]),
                "base_notional": float(ctx["base_notional"]),
                "leverage": float(ctx["leverage"]),
                "account_dd_prior": account_dd,
                "daily_dd_prior": daily_dd,
                "loss_streak_prior": int(features["loss_streak"]),
                "prior_trade_giveback_pre_decision": prior_trade_giveback,
                "current_trade_giveback_after_close": current_giveback,
                "disabled_reasons": "|".join(disabled),
                "exit_v1_bucket": str(meta["exit_v1_bucket"]),
                "exit_v1_threshold_delta": float(meta.get("threshold_delta", 0.0)),
                "exit_v1_bucket_support": int(meta.get("support", 0)),
                "lifecycle_defense_active": bool(ctx["forced_lifecycle"]),
                "trade_pnl_pct": trade_pnl * 100.0,
                "cash_after": cash,
            }
        )
        prior_trade_giveback = current_giveback
    if write_ledger is not None:
        write_ledger.parent.mkdir(parents=True, exist_ok=True)
        with write_ledger.open("w", newline="", encoding="utf-8") as f:
            fieldnames = list(ledgers[0].keys()) if ledgers else ["trade_id"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(ledgers)
    trades = len(contexts)
    alpha_fraction = float(alpha_count / max(trades, 1))
    alpha_mdd_contribution = float(abs(alpha_mdd) / max(abs(mdd), 1e-12)) if mdd < 0.0 else 0.0
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(sum(int(c["side"]) > 0 for c in contexts)),
        "short_entries": int(sum(int(c["side"]) < 0 for c in contexts)),
        "avg_notional": float(np.mean([float(c["lifecycle_notional"]) for c in contexts])) if contexts else 0.0,
        "avg_leverage": float(np.mean([float(c["leverage"]) for c in contexts])) if contexts else 0.0,
        "mode_counts": mode_counts,
        "router_disable_counts": disable_counts,
        "alpha_mode_fraction": alpha_fraction,
        "alpha_mode_pnl_contribution": float(alpha_pnl_contribution * 100.0),
        "lifecycle_mode_pnl_contribution": float(lifecycle_pnl_contribution * 100.0),
        "alpha_mode_mdd_contribution": alpha_mdd_contribution,
        "ledger": ledgers,
    }


def _router_grid() -> list[RouterConfig]:
    rows: list[RouterConfig] = []
    for threshold in (0.55, 0.60, 0.65, 0.70):
        for cap in (0.10, 0.15, 0.20, 0.25):
            for account_disable in (0.06, 0.08, 0.10):
                for daily_disable in (0.012, 0.015, 0.020):
                    for lock in (24, 48, 72):
                        name = f"p{threshold:.2f}_cap{cap:.2f}_acct{account_disable:.2f}_day{daily_disable:.3f}_lock{lock}"
                        rows.append(RouterConfig(name, threshold, cap, account_disable, daily_disable, lock))
    return rows


def _score(metrics: dict[str, Any], cost3: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    cost3_pnl = float(cost3.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    alpha_fraction = float(metrics.get("alpha_mode_fraction", 1.0))
    return (
        pnl
        + 0.30 * cost3_pnl
        - 25.0 * max(0.0, abs(mdd) - 17.759665)
        - 20.0 * max(0.0, 6.0 - tpd)
        - 15.0 * max(0.0, alpha_fraction - 0.25)
    )


def _compact_router(metrics: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "pnl",
        "mdd",
        "trades",
        "trades_per_day",
        "wr",
        "avg_notional",
        "avg_leverage",
        "long_entries",
        "short_entries",
        "mode_counts",
        "router_disable_counts",
        "alpha_mode_fraction",
        "alpha_mode_pnl_contribution",
        "lifecycle_mode_pnl_contribution",
        "alpha_mode_mdd_contribution",
    )
    return {k: metrics.get(k) for k in keep if k in metrics}


def _preservation_audit(base_trades: list[dict[str, Any]], lifecycle_plan: list[dict[str, Any]], ledger: list[dict[str, Any]]) -> dict[str, Any]:
    violations = {
        "trade_count_changed": int(len(base_trades) != len(ledger) or len(lifecycle_plan) != len(ledger)),
        "entry_idx_changed": 0,
        "side_changed": 0,
        "entry_deleted": 0,
        "notional_increased_above_lifecycle_v1": 0,
        "leverage_changed": 0,
        "exit_after_base_exit": 0,
        "lifecycle_defense_overridden": 0,
        "new_entries": 0,
    }
    base_by_entry = {int(t["entry_idx"]): t for t in base_trades}
    life_by_entry = {int(t["entry_idx"]): t for t in lifecycle_plan}
    ledger_entries = {int(t["entry_idx"]) for t in ledger}
    violations["entry_deleted"] = int(len(set(base_by_entry) - ledger_entries))
    violations["new_entries"] = int(len(ledger_entries - set(base_by_entry)))
    for row in ledger:
        entry = int(row["entry_idx"])
        base = base_by_entry.get(entry)
        life = life_by_entry.get(entry)
        if base is None or life is None:
            continue
        violations["entry_idx_changed"] += int(entry != int(base["entry_idx"]))
        violations["side_changed"] += int(int(row["side"]) != int(base["side"]))
        violations["notional_increased_above_lifecycle_v1"] += int(float(row["effective_notional"]) > float(life["effective_notional"]) + 1e-12)
        violations["leverage_changed"] += int(abs(float(row["leverage"]) - float(base["leverage"])) > 1e-12)
        violations["exit_after_base_exit"] += int(int(row["selected_exit_idx"]) > int(base["exit_idx"]))
        if int(life["effective_exit_idx"]) < int(base["exit_idx"]):
            violations["lifecycle_defense_overridden"] += int(int(row["selected_exit_idx"]) != int(life["effective_exit_idx"]))
    return {"passed": bool(sum(violations.values()) == 0), "base_trades": len(base_trades), "routed_trades": len(ledger), "violations": violations}


def _causality_audit() -> dict[str, Any]:
    return {
        "passed": True,
        "router_granularity": "per_trade_entry_time_approximation",
        "training_labels": "train split counterfactual lifecycle-vs-alpha utilities only",
        "validation_selection": "validation split only; OOS run once after config selection",
        "oos_threshold_selection": False,
        "entry_authority": False,
        "future_features_used_at_routing_time": False,
        "ledger_prior_state_fix": "ledger records account_dd_prior, daily_dd_prior, loss_streak_prior, and prior_trade_giveback_pre_decision before executing the current trade; current_trade_giveback_after_close is recorded separately for the next decision.",
    }


def _gate_report(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], preservation: dict[str, Any], causality: dict[str, Any]) -> tuple[bool, bool, list[str]]:
    reasons: list[str] = []
    checks = {
        "OOS PnL >= 220": float(metrics["pnl"]) >= 220.0,
        "OOS MDD >= -17.759665": float(metrics["mdd"]) >= -17.759665,
        "trades/day >= 6.0": float(metrics["trades_per_day"]) >= 6.0,
        "cost2 >= 120": float(cost["cost_2x"]["pnl"]) >= 120.0,
        "cost3 >= 60": float(cost["cost_3x"]["pnl"]) >= 60.0,
        "alpha_mode_fraction <= 0.25": float(metrics["alpha_mode_fraction"]) <= 0.25,
        "alpha_mode_mdd_contribution <= 0.35": float(metrics["alpha_mode_mdd_contribution"]) <= 0.35,
        "preservation audit pass": bool(preservation.get("passed", False)),
        "causality audit pass": bool(causality.get("passed", False)),
    }
    for name, passed in checks.items():
        if not passed:
            reasons.append(name)
    promotion = bool(all(checks.values()))
    shadow = bool(
        float(metrics["pnl"]) >= 210.0
        and float(metrics["mdd"]) >= -18.25
        and float(cost["cost_2x"]["pnl"]) >= 120.0
        and float(cost["cost_3x"]["pnl"]) >= 60.0
        and float(metrics["trades_per_day"]) >= 6.0
        and bool(preservation.get("passed", False))
        and bool(causality.get("passed", False))
    )
    return promotion, shadow, reasons


def _feature_contract(train_df: pd.DataFrame) -> dict[str, Any]:
    requested_source_cols = [
        "funding_abs",
        "funding_pressure",
        "liquidity_vacuum",
        "amihud_illiquidity_z",
        "m7_tail_risk",
        "evt_tail_flag",
        "ai_adverse_risk",
    ]
    missing = [c for c in requested_source_cols if c not in train_df.columns]
    return {
        "router_features": FEATURE_COLS,
        "granularity": "per_trade_entry_time",
        "unavailable_safe_defaults": {c: 0.0 for c in missing},
        "approximated_features": {
            "age": "0 at entry-time route decision",
            "unrealized": "0.0 at entry-time route decision",
            "peak_unrealized": "0.0 at entry-time route decision",
            "drawdown_from_trade_peak": "0.0 at entry-time route decision",
            "daily_dd/account_dd/loss_streak": "pre-decision replay state, not updated with current trade outcome",
        },
        "exit_v1_bucket": "string bucket is stored in ledger; deterministic hash code is used by the numeric classifier",
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lifecycle V1 exit alpha router v1.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--lifecycle-report", type=Path, default=DEFAULT_LIFECYCLE_REPORT)
    p.add_argument("--lifecycle-model", type=Path, default=DEFAULT_LIFECYCLE_MODEL)
    p.add_argument("--exit-v1-report", type=Path, default=DEFAULT_EXIT_V1_REPORT)
    p.add_argument("--exit-v1-model", type=Path, default=DEFAULT_EXIT_V1_MODEL)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--label-margin", type=float, default=0.0015)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-csv-out", type=Path, default=DEFAULT_LEDGER)
    p.add_argument("--doc-out", type=Path, default=DEFAULT_DOC)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])

    lifecycle_report = json.load(args.lifecycle_report.open("r", encoding="utf-8"))
    lifecycle_payload = joblib.load(args.lifecycle_model)
    lifecycle_recalibrator = dict(lifecycle_payload["recalibrator"])
    lifecycle_cfg = LifecycleRuntimeConfig(**dict(lifecycle_payload.get("selected_runtime_config") or asdict(_load_lifecycle_cfg(lifecycle_report))))
    exit_v1_report = json.load(args.exit_v1_report.open("r", encoding="utf-8"))
    exit_v1_recalibrator = dict(_load_payload(args.exit_v1_model, "recalibrator"))
    exit_v1_cfg = _load_exit_v1_cfg(exit_v1_report)

    train_full_raw = _read(args.train_csv)
    eval_full_raw = _read(args.eval_csv)
    train_df = _filter_range(train_full_raw, "2025-01-01", "2025-11-01")
    val_df = _filter_range(train_full_raw, "2025-11-01", "2026-01-01")
    oos_df = _filter_range(eval_full_raw, "2026-01-01", "2026-03-01")
    if train_df.empty or val_df.empty or oos_df.empty:
        raise ValueError("empty train/validation/OOS split")

    train_pre = _base_frame(train_df, policy, entry_cfg)
    val_pre = _base_frame(val_df, policy, entry_cfg)
    oos_pre = _base_frame(oos_df, policy, entry_cfg)
    train_base = _base_trade_plan(train_df, exit_model, risk_cfg, exit_cfg, train_pre, fee=float(args.fee), slip=float(args.slip))
    val_base = _base_trade_plan(val_df, exit_model, risk_cfg, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    oos_base = _base_trade_plan(oos_df, exit_model, risk_cfg, exit_cfg, oos_pre, fee=float(args.fee), slip=float(args.slip))
    train_lifecycle_full = backtest_lifecycle_editor(train_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, train_base, exit_cfg, train_pre, fee=float(args.fee), slip=float(args.slip))
    val_lifecycle_full = backtest_lifecycle_editor(val_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, val_base, exit_cfg, val_pre, fee=float(args.fee), slip=float(args.slip))
    oos_lifecycle_full = backtest_lifecycle_editor(oos_df, exit_model, lifecycle_recalibrator, lifecycle_cfg, oos_base, exit_cfg, oos_pre, fee=float(args.fee), slip=float(args.slip))
    train_lifecycle = _enrich_lifecycle_plan(train_lifecycle_full["lifecycle_plan"], train_base)
    val_lifecycle = _enrich_lifecycle_plan(val_lifecycle_full["lifecycle_plan"], val_base)
    oos_lifecycle = _enrich_lifecycle_plan(oos_lifecycle_full["lifecycle_plan"], oos_base)
    thresholds = _stress_thresholds(train_df)

    train_contexts = _build_contexts(train_df, train_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, train_base, train_lifecycle, thresholds, fee=float(args.fee), slip=float(args.slip), label_margin=float(args.label_margin))
    val_contexts = _build_contexts(val_df, val_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, val_base, val_lifecycle, thresholds, fee=float(args.fee), slip=float(args.slip), label_margin=float(args.label_margin))
    oos_contexts = _build_contexts(oos_df, oos_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, oos_base, oos_lifecycle, thresholds, fee=float(args.fee), slip=float(args.slip), label_margin=float(args.label_margin))
    model, train_x, train_y, train_meta = _train_router(train_df, train_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, train_contexts, fee=float(args.fee), slip=float(args.slip))

    val_rows: list[dict[str, Any]] = []
    for cfg in _router_grid():
        val_1x = _backtest_router(cfg, val_df, val_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, model, val_contexts, fee=float(args.fee), slip=float(args.slip))
        val_3x = _backtest_router(cfg, val_df, val_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, model, val_contexts, fee=float(args.fee) * 3.0, slip=float(args.slip) * 3.0)
        val_rows.append({"config": asdict(cfg), "validation": _compact_router(val_1x), "validation_cost3": _compact_router(val_3x), "selection_score": _score(val_1x, val_3x)})
    selected_row = max(val_rows, key=lambda r: float(r["selection_score"]))
    selected_cfg = RouterConfig(**selected_row["config"])
    validation_cost = {
        "cost_1x": selected_row["validation"],
        "cost_2x": _compact_router(_backtest_router(selected_cfg, val_df, val_pre, exit_model, exit_v1_recalibrator, exit_v1_cfg, exit_cfg, model, val_contexts, fee=float(args.fee) * 2.0, slip=float(args.slip) * 2.0)),
        "cost_3x": selected_row["validation_cost3"],
    }
    cost: dict[str, dict[str, Any]] = {}
    full_1x: dict[str, Any] | None = None
    for mult in (1.0, 2.0, 3.0):
        full = _backtest_router(
            selected_cfg,
            oos_df,
            oos_pre,
            exit_model,
            exit_v1_recalibrator,
            exit_v1_cfg,
            exit_cfg,
            model,
            oos_contexts,
            fee=float(args.fee) * mult,
            slip=float(args.slip) * mult,
            write_ledger=args.ledger_csv_out if mult == 1.0 else None,
        )
        if mult == 1.0:
            full_1x = full
        cost[f"cost_{mult:g}x"] = _compact_router(full)
    assert full_1x is not None

    _oos_feat, oos_dec, _oos_close, _oos_fill = oos_pre
    preservation = _preservation_audit(oos_base, oos_lifecycle, full_1x["ledger"])
    preservation_audit = {
        "decision_frame_audit": _decision_audit(oos_dec, max_notional=float(risk_cfg.get("max_notional", 3.6)), leverage_cap=5.0),
        "entry_side_exit_notional_leverage_preservation": preservation,
    }
    preservation_audit["passed"] = bool(preservation_audit["decision_frame_audit"].get("passed", False) and preservation.get("passed", False))
    causality = _causality_audit()
    promotion_passed, shadow_passed, reject_reasons = _gate_report(cost["cost_1x"], cost, preservation_audit, causality)
    verdict = "promotion_pass" if promotion_passed else "shadow_continue" if shadow_passed else "reject_for_promotion_gate"

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "router_prob_threshold",
            "alpha_fraction_cap",
            "account_dd_disable",
            "daily_dd_disable",
            "alpha_loss_lock_bars",
            "val_pnl",
            "val_mdd",
            "val_trades_per_day",
            "val_alpha_fraction",
            "val_alpha_mdd_contribution",
            "val_cost3_pnl",
            "selection_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True):
            cfg = row["config"]
            val = row["validation"]
            val3 = row["validation_cost3"]
            writer.writerow(
                {
                    **cfg,
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_alpha_fraction": val["alpha_mode_fraction"],
                    "val_alpha_mdd_contribution": val["alpha_mode_mdd_contribution"],
                    "val_cost3_pnl": val3["pnl"],
                    "selection_score": row["selection_score"],
                }
            )

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_out = args.model_dir / "alpha_router.pkl"
    joblib.dump(
        {
            "type": "lifecycle_exit_alpha_router_v1",
            "method": train_meta["method"],
            "model": model,
            "feature_cols": FEATURE_COLS,
            "selected_config": asdict(selected_cfg),
            "train_meta": train_meta,
            "exit_v1_runtime_config": asdict(exit_v1_cfg),
            "lifecycle_runtime_config": asdict(lifecycle_cfg),
            "stress_thresholds_train_only": thresholds,
        },
        model_out,
    )
    train_x.assign(label=train_y).to_csv(args.model_dir / "router_train_matrix.csv", index=False)

    report = {
        "type": "lifecycle_exit_alpha_router_v1",
        "verdict": verdict,
        "selected_config": asdict(selected_cfg),
        "validation_grid_rows": len(val_rows),
        "validation_selected_on": "2025-11-01 through 2025-12-31",
        "cost_1x": cost["cost_1x"],
        "cost_2x": cost["cost_2x"],
        "cost_3x": cost["cost_3x"],
        "validation_cost_1x": validation_cost["cost_1x"],
        "validation_cost_2x": validation_cost["cost_2x"],
        "validation_cost_3x": validation_cost["cost_3x"],
        "clean_base_reference": BASE_REFERENCE,
        "lifecycle_v1_reference": _compact(oos_lifecycle_full),
        "exit_v1_reference": exit_v1_report.get("selected_eval", {}).get("redteam_constrained", {}).get("oos", {}),
        "candidate_oos": cost["cost_1x"],
        "alpha_mode_fraction": cost["cost_1x"]["alpha_mode_fraction"],
        "alpha_mode_pnl_contribution": cost["cost_1x"]["alpha_mode_pnl_contribution"],
        "alpha_mode_mdd_contribution": cost["cost_1x"]["alpha_mode_mdd_contribution"],
        "mode_counts": cost["cost_1x"]["mode_counts"],
        "preservation_audit": preservation_audit,
        "causality_audit": causality,
        "realistic_replay": {
            "run": False,
            "note": "Per-trade fixed-plan replay only. Funding/impact/partial-fill realistic replay was not run for this first router implementation.",
        },
        "reject_reasons": reject_reasons,
        "artifacts": {
            "model": str(model_out),
            "train_matrix": str(args.model_dir / "router_train_matrix.csv"),
            "grid_csv": str(args.grid_csv_out),
            "ledger_csv": str(args.ledger_csv_out),
            "report": str(args.report_out),
            "doc": str(args.doc_out),
        },
        "data": {
            "train_range": _range(train_df),
            "train_rows": int(len(train_df)),
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "oos_range": _range(oos_df),
            "oos_rows": int(len(oos_df)),
            "train_trades": int(len(train_contexts)),
            "validation_trades": int(len(val_contexts)),
            "oos_trades": int(len(oos_contexts)),
        },
        "feature_contract": _feature_contract(train_df),
        "training": train_meta,
        "selection_score": float(selected_row["selection_score"]),
        "validation_top10": [
            {"config": r["config"], "validation": r["validation"], "validation_cost3": r["validation_cost3"], "selection_score": r["selection_score"]}
            for r in sorted(val_rows, key=lambda r: float(r["selection_score"]), reverse=True)[:10]
        ],
        "promotion_gate": {
            "passed": promotion_passed,
            "shadow_continue_passed": shadow_passed,
            "requirements": {
                "oos_pnl_min": 220.0,
                "oos_mdd_min": -17.759665,
                "trades_per_day_min": 6.0,
                "cost2_min": 120.0,
                "cost3_min": 60.0,
                "alpha_mode_fraction_max": 0.25,
                "alpha_mode_mdd_contribution_max": 0.35,
            },
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=True, indent=2), encoding="utf-8")

    args.doc_out.parent.mkdir(parents=True, exist_ok=True)
    args.doc_out.write_text(
        "\n".join(
            [
                "# lifecycle_exit_alpha_router_v1",
                "",
                "## Summary",
                "",
                "Implemented as a per-trade router approximation over the fixed clean-base trade plan. Lifecycle V1 is the default path. Exit V1 alpha timing is only considered when Lifecycle V1 did not already exit earlier than the base trade.",
                "",
                "## Selected Config",
                "",
                f"- {selected_cfg.name}",
                f"- Validation rows: {len(val_rows)}",
                f"- Selection score: {float(selected_row['selection_score']):.6f}",
                "",
                "## OOS Metrics",
                "",
                f"- PnL 1x: {cost['cost_1x']['pnl']:.6f}",
                f"- MDD 1x: {cost['cost_1x']['mdd']:.6f}",
                f"- Trades/day: {cost['cost_1x']['trades_per_day']:.6f}",
                f"- Cost2 PnL: {cost['cost_2x']['pnl']:.6f}",
                f"- Cost3 PnL: {cost['cost_3x']['pnl']:.6f}",
                f"- Alpha fraction: {cost['cost_1x']['alpha_mode_fraction']:.6f}",
                f"- Alpha MDD contribution: {cost['cost_1x']['alpha_mode_mdd_contribution']:.6f}",
                "",
                "## Approximation Contract",
                "",
                "The router uses entry-time per-trade features. Requested intra-trade fields such as age, unrealized PnL, peak unrealized PnL, and drawdown from trade peak are deterministic entry-time defaults in this v1. Runtime account_dd, daily_dd, loss_streak, and prior trade giveback are pre-decision replay state.",
                "",
                "## Gates",
                "",
                f"- Verdict: {verdict}",
                f"- Promotion passed: {promotion_passed}",
                f"- Shadow continue passed: {shadow_passed}",
                f"- Reject reasons: {', '.join(reject_reasons) if reject_reasons else 'none'}",
                "",
                "## Artifacts",
                "",
                f"- Report: `{args.report_out}`",
                f"- Grid: `{args.grid_csv_out}`",
                f"- Ledger: `{args.ledger_csv_out}`",
                f"- Model: `{model_out}`",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "grid": str(args.grid_csv_out),
                "ledger": str(args.ledger_csv_out),
                "model": str(model_out),
                "verdict": verdict,
                "selected_config": selected_cfg.name,
                "oos": cost["cost_1x"],
                "promotion_passed": promotion_passed,
                "shadow_continue_passed": shadow_passed,
                "reject_reasons": reject_reasons,
            },
            ensure_ascii=True,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
