#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, FEATURE_COLS, prepare_features, predict_policy_frame  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit, _quality_scaled_decisions  # noqa: E402
from scripts.eval_hf_risk_overlay_grid import _read  # noqa: E402
from scripts.eval_lifecycle_ai_stress import _stress_frame  # noqa: E402


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_v4_balanced_h144.pkl"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/hf_entry_grid/hf_no_limit_exit_governor.pkl"
DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/hf_no_limit_exit_governor_2026.json"


CONTEXT_COLS = [
    "gx_side",
    "gx_age_bars",
    "gx_age_log",
    "gx_unrealized",
    "gx_peak_unrealized",
    "gx_drawdown_from_peak",
    "gx_notional",
    "gx_leverage",
    "gx_entry_quality",
    "gx_entry_confidence",
    "gx_current_same_side",
    "gx_current_opposite_side",
    "gx_current_quality",
    "gx_current_confidence",
]
MODEL_COLS = list(FEATURE_COLS) + CONTEXT_COLS


MONTHLY_BALANCED_ENTRY = {"notional_mult": 1.5, "quality_floor": 0.0, "confidence_floor": 0.0, "max_notional": 3.6}
MONTHLY_BALANCED_RISK = {
    "max_daily_trades": 16,
    "daily_loss_limit": 0.04,
    "daily_dd_limit": 0.035,
    "global_dd_cut": 0.12,
    "global_dd_mult": 0.45,
    "loss_cooldown_bars": 12,
    "loss_streak_soft": 2,
    "loss_streak_mult": 0.65,
    "max_notional": 3.6,
    "daily_profit_boost_start": 0.015,
    "daily_profit_boost_mult": 1.10,
}


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _fill_array(df: pd.DataFrame) -> np.ndarray:
    col = "open" if "open" in df.columns else "close"
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _days(df: pd.DataFrame) -> float:
    if "timestamp" not in df.columns or len(df) < 2:
        return max(len(df) / 288.0, 1e-8)
    return max((df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / 86400.0, 1e-8)


def _fill_price(fill_px: np.ndarray, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _raw_from_entry(fill_px: np.ndarray, idx: int, side: int, entry_price: float, slip: float) -> float:
    px0 = float(fill_px[int(np.clip(idx, 0, len(fill_px) - 1))])
    px = px0 * (1.0 - slip if side > 0 else 1.0 + slip)
    if side > 0:
        return float((px - entry_price) / max(entry_price, 1e-12))
    return float((entry_price - px) / max(entry_price, 1e-12))


def _future_raw_from_entry(fill_px: np.ndarray, start_idx: int, end_idx: int, side: int, entry_price: float, slip: float) -> np.ndarray:
    start = int(np.clip(start_idx, 0, len(fill_px) - 1))
    end = int(np.clip(end_idx, 0, len(fill_px) - 1))
    if end < start:
        return np.zeros(0, dtype=np.float64)
    px = fill_px[start : end + 1].astype(np.float64, copy=False)
    if side > 0:
        exit_px = px * (1.0 - slip)
        return (exit_px - entry_price) / max(entry_price, 1e-12)
    exit_px = px * (1.0 + slip)
    return (entry_price - exit_px) / max(entry_price, 1e-12)


def _base_frame(df: pd.DataFrame, policy: dict[str, Any], entry_config: dict[str, float]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    close = _close(df)
    fill_px = _fill_array(df)
    feat = prepare_features(df, side_hint=0, close=close)
    dec = predict_policy_frame(policy, feat)
    dec = _quality_scaled_decisions(dec, **entry_config)
    return feat, dec, close, fill_px


def _slice_precomputed(
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    mask: pd.Series | np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    base_feat, decisions, close, fill_px = precomputed
    return (
        base_feat.iloc[idx].reset_index(drop=True),
        decisions.iloc[idx].reset_index(drop=True),
        close[idx],
        fill_px[idx],
    )


def _feature_row(
    base_feat: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    i: int,
    side: int,
    age: int,
    unrealized: float,
    peak_unrealized: float,
    notional: float,
    leverage: float,
    entry_quality: float,
    entry_confidence: float,
) -> dict[str, float]:
    dec = decisions.iloc[int(i)]
    current_side = int(getattr(dec, "side", 0))
    row = base_feat.iloc[int(i)].to_dict()
    row["side_hint"] = float(side)
    row.update(
        {
            "gx_side": float(side),
            "gx_age_bars": float(age),
            "gx_age_log": float(np.log1p(max(age, 0))),
            "gx_unrealized": float(unrealized),
            "gx_peak_unrealized": float(peak_unrealized),
            "gx_drawdown_from_peak": float(peak_unrealized - unrealized),
            "gx_notional": float(notional),
            "gx_leverage": float(leverage),
            "gx_entry_quality": float(entry_quality),
            "gx_entry_confidence": float(entry_confidence),
            "gx_current_same_side": float(current_side == side),
            "gx_current_opposite_side": float(current_side == -side),
            "gx_current_quality": float(getattr(dec, "quality_score", 0.0)),
            "gx_current_confidence": float(getattr(dec, "confidence", 0.0)),
        }
    )
    return {c: float(row.get(c, 0.0) or 0.0) for c in MODEL_COLS}


def collect_exit_samples(
    df: pd.DataFrame,
    policy: dict[str, Any],
    *,
    entry_config: dict[str, float],
    fee: float,
    slip: float,
    entry_stride: int,
    min_age: int,
    max_age: int,
    age_stride: int,
    future_horizon: int,
    exit_edge: float,
    adverse_gap: float,
    max_samples: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    base_feat, decisions, _, fill_px = _base_frame(df, policy, entry_config)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    entry_count = 0
    active = np.flatnonzero((actions != ACTION_CASH) & (sides != 0))
    active = active[active < len(df) - max(int(max_age), int(future_horizon)) - 3]
    active = active[:: max(1, int(entry_stride))]
    for entry_i in active:
        side = int(sides[int(entry_i)])
        notional = float(notionals[int(entry_i)])
        if notional <= 1e-8:
            continue
        entry_count += 1
        entry_price = _fill_price(fill_px, int(entry_i) + 1, side, slip, entry=True)
        peak_unrealized = 0.0
        for age in range(int(min_age), int(max_age) + 1, max(1, int(age_stride))):
            i = int(entry_i) + int(age)
            if i >= len(df) - 3:
                break
            raw_now = _raw_from_entry(fill_px, i + 1, side, entry_price, slip)
            immediate = raw_now * notional - float(fee) * notional
            unreal = raw_now * notional
            peak_unrealized = max(peak_unrealized, unreal)
            end = min(len(df) - 2, i + int(future_horizon))
            if end <= i + 1:
                continue
            future_raw = _future_raw_from_entry(fill_px, i + 2, end + 1, side, entry_price, slip)
            if future_raw.size == 0:
                continue
            future_net = future_raw * notional - float(fee) * notional
            future_best = float(np.max(future_net))
            future_worst = float(np.min(future_net))
            continuation_edge = future_best - immediate
            label = int(
                continuation_edge <= float(exit_edge)
                or (future_worst <= immediate - abs(float(adverse_gap)) and continuation_edge <= float(exit_edge) * 3.0)
            )
            rows.append(
                _feature_row(
                    base_feat,
                    decisions,
                    i=i,
                    side=side,
                    age=int(age),
                    unrealized=unreal,
                    peak_unrealized=peak_unrealized,
                    notional=notional,
                    leverage=float(leverages[int(entry_i)]),
                    entry_quality=float(qualities[int(entry_i)]),
                    entry_confidence=float(confs[int(entry_i)]),
                )
            )
            labels.append(label)
    x = pd.DataFrame(rows, columns=MODEL_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    if len(x) > int(max_samples):
        rng = np.random.default_rng(int(seed))
        take = np.sort(rng.choice(len(x), size=int(max_samples), replace=False))
        x = x.iloc[take].reset_index(drop=True)
        y = y[take]
    meta = {
        "entries_sampled": int(entry_count),
        "samples": int(len(x)),
        "exit_labels": int(y.sum()) if len(y) else 0,
        "exit_label_rate": float(y.mean()) if len(y) else 0.0,
        "entry_stride": int(entry_stride),
        "min_age": int(min_age),
        "max_age": int(max_age),
        "age_stride": int(age_stride),
        "future_horizon": int(future_horizon),
        "exit_edge": float(exit_edge),
        "adverse_gap": float(adverse_gap),
        "max_samples": int(max_samples),
    }
    return x, y, meta


def train_exit_model(x: pd.DataFrame, y: np.ndarray, *, seed: int) -> Any:
    if len(np.unique(y)) < 2:
        raise ValueError("exit governor needs both exit and hold labels")
    exit_rate = float(np.mean(y))
    pos_weight = min(4.0, max(0.75, (1.0 - exit_rate) / max(exit_rate, 1e-6)))
    sample_weight = np.where(y == 1, pos_weight, 1.0).astype(np.float64)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(
            C=0.35,
            max_iter=1000,
            solver="lbfgs",
            random_state=int(seed),
        ),
    )
    model.fit(x.to_numpy(dtype=np.float32, copy=False), y, logisticregression__sample_weight=sample_weight)
    return model


def _exit_probability(model: Any, row: dict[str, float]) -> float:
    arr = np.asarray([[float(row.get(c, 0.0) or 0.0) for c in MODEL_COLS]], dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        proba = model.predict_proba(arr)
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return 0.0
    return float(proba[0, int(np.flatnonzero(classes == 1)[0])])


def _exit_probability_vec(model: Any, arr: np.ndarray) -> float:
    if hasattr(model, "named_steps") and "logisticregression" in model.named_steps:
        x = arr.astype(np.float64, copy=True)
        imp = model.named_steps.get("simpleimputer")
        if imp is not None:
            stats = np.asarray(imp.statistics_, dtype=np.float64)
            bad = ~np.isfinite(x)
            if bad.any():
                x[bad] = stats[bad]
        scaler = model.named_steps.get("standardscaler")
        if scaler is not None:
            x = (x - np.asarray(scaler.mean_, dtype=np.float64)) / np.maximum(np.asarray(scaler.scale_, dtype=np.float64), 1e-12)
        lr = model.named_steps["logisticregression"]
        z = float(np.dot(x, np.asarray(lr.coef_[0], dtype=np.float64)) + float(lr.intercept_[0]))
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0))))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        proba = model.predict_proba(arr.reshape(1, -1).astype(np.float32, copy=False))
    classes = np.asarray(model.classes_, dtype=int)
    if 1 not in classes:
        return 0.0
    return float(proba[0, int(np.flatnonzero(classes == 1)[0])])


def _feature_vec_fast(
    base_values: np.ndarray,
    side_values: np.ndarray,
    quality_values: np.ndarray,
    confidence_values: np.ndarray,
    *,
    i: int,
    side: int,
    age: int,
    unrealized: float,
    peak_unrealized: float,
    notional: float,
    leverage: float,
    entry_quality: float,
    entry_confidence: float,
) -> np.ndarray:
    out = np.zeros(len(MODEL_COLS), dtype=np.float32)
    out[: len(FEATURE_COLS)] = base_values[int(i)]
    out[0] = float(side)
    j = len(FEATURE_COLS)
    current_side = int(side_values[int(i)])
    ctx = (
        float(side),
        float(age),
        float(np.log1p(max(age, 0))),
        float(unrealized),
        float(peak_unrealized),
        float(peak_unrealized - unrealized),
        float(notional),
        float(leverage),
        float(entry_quality),
        float(entry_confidence),
        float(current_side == side),
        float(current_side == -side),
        float(quality_values[int(i)]),
        float(confidence_values[int(i)]),
    )
    out[j : j + len(CONTEXT_COLS)] = np.asarray(ctx, dtype=np.float32)
    return out


def _date_codes(df: pd.DataFrame) -> np.ndarray:
    if "timestamp" in df.columns:
        return pd.to_datetime(df["timestamp"], errors="coerce").dt.floor("D").astype("int64").to_numpy()
    return (np.arange(len(df), dtype=np.int64) // 288).astype(np.int64)


def backtest_no_limit_exit(
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    *,
    entry_config: dict[str, float],
    risk_config: dict[str, Any],
    exit_threshold: float,
    min_exit_age: int,
    fee: float,
    slip: float,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray] | None = None,
) -> dict[str, Any]:
    base_feat, decisions, close, fill_px = precomputed if precomputed is not None else _base_frame(df, policy, entry_config)
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    day_codes = _date_codes(df)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    cooldowns = pd.to_numeric(decisions["cooldown_bars"], errors="coerce").fillna(0).to_numpy(dtype=np.int64)
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    model_cooldown = 0
    cooldown_left = 0
    loss_cooldown_left = 0
    loss_streak = 0
    peak_unrealized = 0.0
    entry_quality = 0.0
    entry_confidence = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    entry_blocks: dict[str, int] = {}
    exit_probs: list[float] = []
    day_key: int | None = None
    daily_start_cash = 1.0
    daily_peak_eq = 1.0
    daily_trades = 0

    def block(reason: str) -> None:
        entry_blocks[reason] = entry_blocks.get(reason, 0) + 1

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

    def close_position(i: int, reason: str) -> None:
        nonlocal cash, pos, entry_price, notional, leverage, cooldown_left, model_cooldown
        nonlocal trades, wins, loss_streak, loss_cooldown_left, daily_trades, peak_unrealized
        exit_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee * notional
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_config.get("loss_cooldown_bars", 0)))
        exits[reason] = exits.get(reason, 0) + 1
        pos = 0
        entry_price = 0.0
        notional = 0.0
        leverage = 1.0
        cooldown_left = int(model_cooldown)
        model_cooldown = 0
        peak_unrealized = 0.0

    for i in range(0, len(df) - 2):
        key = int(day_codes[i])
        eq, unreal = mark(i)
        if key != day_key:
            day_key = key
            daily_start_cash = max(eq, 1e-12)
            daily_peak_eq = max(eq, 1e-12)
            daily_trades = 0
        peak = max(peak, eq)
        daily_peak_eq = max(daily_peak_eq, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        account_dd = max(0.0, 1.0 - eq / max(peak, 1e-12))
        daily_dd = max(0.0, 1.0 - eq / max(daily_peak_eq, 1e-12))
        daily_realized = cash / max(daily_start_cash, 1e-12) - 1.0

        if pos != 0:
            peak_unrealized = max(peak_unrealized, unreal)
            age = i - entry_idx
            if age >= int(min_exit_age):
                row_vec = _feature_vec_fast(
                    base_values,
                    sides,
                    qualities,
                    confs,
                    i=i,
                    side=pos,
                    age=age,
                    unrealized=unreal,
                    peak_unrealized=peak_unrealized,
                    notional=notional,
                    leverage=leverage,
                    entry_quality=entry_quality,
                    entry_confidence=entry_confidence,
                )
                p_exit = _exit_probability_vec(exit_model, row_vec)
                exit_probs.append(p_exit)
                if p_exit >= float(exit_threshold):
                    close_position(i, "exit_governor")
                    continue
            continue

        if cooldown_left > 0:
            cooldown_left -= 1
            block("model_cooldown")
            continue
        if loss_cooldown_left > 0:
            loss_cooldown_left -= 1
            block("loss_cooldown")
            continue
        if daily_trades >= int(risk_config.get("max_daily_trades", 999999)):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(risk_config.get("daily_loss_limit", 0.0))):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(risk_config.get("daily_dd_limit", 0.0))):
            block("daily_dd_lock")
            continue
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            block("cash_signal")
            continue

        n = float(notionals[i])
        if account_dd >= float(risk_config.get("global_dd_cut", 999.0)):
            n *= float(risk_config.get("global_dd_mult", 1.0))
        if loss_streak >= int(risk_config.get("loss_streak_soft", 999999)):
            steps = loss_streak - int(risk_config.get("loss_streak_soft", 999999)) + 1
            n *= float(risk_config.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(risk_config.get("daily_profit_boost_start", 999.0)):
            n *= float(risk_config.get("daily_profit_boost_mult", 1.0))
        if float(risk_config.get("equity_high_boost_dd", -1.0)) >= 0.0 and account_dd <= float(risk_config.get("equity_high_boost_dd", -1.0)):
            n *= float(risk_config.get("equity_high_boost_mult", 1.0))
        n = float(np.clip(n, 0.0, float(risk_config.get("max_notional", 3.6))))
        if n <= 1e-8:
            block("zero_notional")
            continue

        pos = int(sides[i])
        entry_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = float(leverages[i])
        model_cooldown = int(cooldowns[i])
        cash -= cash * fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        peak_unrealized = 0.0
        entry_quality = float(qualities[i])
        entry_confidence = float(confs[i])

    if pos != 0:
        close_position(len(df) - 2, "forced_end")
    entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "trades_per_day": float(trades / _days(df)),
        "wr": float(wins / max(trades, 1)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / entries),
        "avg_leverage": float(leverage_sum / entries),
        "entry_blocks": entry_blocks,
        "exits": exits,
        "exit_prob_mean": float(np.mean(exit_probs)) if exit_probs else 0.0,
        "exit_prob_p95": float(np.quantile(exit_probs, 0.95)) if exit_probs else 0.0,
    }


def _compact(bt: dict[str, Any]) -> dict[str, Any]:
    return {
        k: bt.get(k)
        for k in ("pnl", "mdd", "trades", "trades_per_day", "wr", "avg_notional", "avg_leverage", "long_entries", "short_entries", "entry_blocks", "exits", "exit_prob_mean", "exit_prob_p95")
    }


def _threshold_grid() -> list[dict[str, Any]]:
    rows = []
    for th in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70):
        for age in (3, 6, 12):
            rows.append({"exit_threshold": th, "min_exit_age": age})
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate HF exit governor with no TP/SL/max-hold execution triggers.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--entry-stride", type=int, default=12)
    p.add_argument("--min-age", type=int, default=3)
    p.add_argument("--max-age", type=int, default=288)
    p.add_argument("--age-stride", type=int, default=6)
    p.add_argument("--future-horizon", type=int, default=144)
    p.add_argument("--exit-edge", type=float, default=0.0015)
    p.add_argument("--adverse-gap", type=float, default=0.012)
    p.add_argument("--max-samples", type=int, default=300000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    x, y, sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=MONTHLY_BALANCED_ENTRY,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=int(args.entry_stride),
        min_age=int(args.min_age),
        max_age=int(args.max_age),
        age_stride=int(args.age_stride),
        future_horizon=int(args.future_horizon),
        exit_edge=float(args.exit_edge),
        adverse_gap=float(args.adverse_gap),
        max_samples=int(args.max_samples),
        seed=int(args.seed),
    )
    model = train_exit_model(x, y, seed=int(args.seed))
    bundle = {
        "type": "hf_no_limit_exit_governor",
        "model": model,
        "feature_cols": MODEL_COLS,
        "entry_config": MONTHLY_BALANCED_ENTRY,
        "risk_config": MONTHLY_BALANCED_RISK,
        "sample_meta": sample_meta,
        "removed_execution_triggers": ["take_profit", "stop_loss", "max_hold_bars"],
    }
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_out)

    eval_precomputed = _base_frame(eval_df, policy, MONTHLY_BALANCED_ENTRY)
    rows = []
    for cfg in _threshold_grid():
        bt = backtest_no_limit_exit(
            eval_df,
            policy,
            model,
            entry_config=MONTHLY_BALANCED_ENTRY,
            risk_config=MONTHLY_BALANCED_RISK,
            fee=float(args.fee),
            slip=float(args.slip),
            precomputed=eval_precomputed,
            **cfg,
        )
        rows.append({"name": f"exit{cfg['exit_threshold']}_age{cfg['min_exit_age']}", "config": cfg, "eval": _compact(bt)})
    ranked = sorted(rows, key=lambda r: float(r["eval"].get("pnl") or -1e18), reverse=True)
    goal = [r for r in ranked if 5.0 <= float(r["eval"].get("trades_per_day") or 0.0) <= 20.0]
    mdd15 = [r for r in goal if float(r["eval"].get("mdd") or -1e18) >= -15.0]
    top = (mdd15 or goal or ranked)[:6]

    monthly = []
    jan = eval_df["timestamp"] < pd.Timestamp("2026-02-01")
    feb = eval_df["timestamp"] >= pd.Timestamp("2026-02-01")
    jan_df = eval_df.loc[jan].reset_index(drop=True)
    feb_df = eval_df.loc[feb].reset_index(drop=True)
    jan_precomputed = _slice_precomputed(eval_precomputed, jan)
    feb_precomputed = _slice_precomputed(eval_precomputed, feb)
    for row in top:
        cfg = row["config"]
        jan_eval = _compact(backtest_no_limit_exit(jan_df, policy, model, entry_config=MONTHLY_BALANCED_ENTRY, risk_config=MONTHLY_BALANCED_RISK, fee=float(args.fee), slip=float(args.slip), precomputed=jan_precomputed, **cfg))
        feb_eval = _compact(backtest_no_limit_exit(feb_df, policy, model, entry_config=MONTHLY_BALANCED_ENTRY, risk_config=MONTHLY_BALANCED_RISK, fee=float(args.fee), slip=float(args.slip), precomputed=feb_precomputed, **cfg))
        monthly.append({"name": row["name"], "full": row["eval"], "jan": jan_eval, "feb": feb_eval, "min_month_pnl": float(min(jan_eval["pnl"], feb_eval["pnl"]))})
    monthly_balanced = sorted(monthly, key=lambda r: (float(r["min_month_pnl"]), float(r["full"]["pnl"])), reverse=True)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in top[:4]:
            bt = backtest_no_limit_exit(
                eval_df,
                policy,
                model,
                entry_config=MONTHLY_BALANCED_ENTRY,
                risk_config=MONTHLY_BALANCED_RISK,
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
                precomputed=eval_precomputed,
                **row["config"],
            )
            cost_stress[f"cost_{mult:g}x"].append({"name": row["name"], "eval": _compact(bt)})

    ai_stress: dict[str, Any] = {}
    selected_cfgs = monthly_balanced[:3] or monthly[:3]
    cfg_by_name = {r["name"]: r["config"] for r in rows}
    for mode in ("normal", "all_ai_zero", "patchtst_zero", "tide_zero", "dlinear_zero"):
        df, meta = _stress_frame(eval_df, mode)
        stress_precomputed = _base_frame(df, policy, MONTHLY_BALANCED_ENTRY)
        ai_stress[mode] = {"stress": meta, "results": []}
        for row in selected_cfgs:
            bt = backtest_no_limit_exit(
                df,
                policy,
                model,
                entry_config=MONTHLY_BALANCED_ENTRY,
                risk_config=MONTHLY_BALANCED_RISK,
                fee=float(args.fee),
                slip=float(args.slip),
                precomputed=stress_precomputed,
                **cfg_by_name[row["name"]],
            )
            ai_stress[mode]["results"].append({"name": row["name"], "eval": _compact(bt)})

    report = {
        "type": "hf_no_limit_exit_governor_2026",
        "policy": str(args.policy),
        "model_out": str(args.model_out),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "accounting": "entry fee, exit fee, entry/exit slippage, forced-end close fee; TP/SL/max-hold execution triggers are disabled",
        "sample_meta": sample_meta,
        "removed_execution_triggers": ["take_profit", "stop_loss", "max_hold_bars"],
        "entry_config": MONTHLY_BALANCED_ENTRY,
        "risk_config": MONTHLY_BALANCED_RISK,
        "grid": rows,
        "ranked_by_pnl": [{"name": r["name"], **r["eval"]} for r in ranked[:20]],
        "ranked_goal_5_to_20": [{"name": r["name"], **r["eval"]} for r in goal[:20]],
        "ranked_goal_mdd_lte_15": [{"name": r["name"], **r["eval"]} for r in mdd15[:20]],
        "monthly_balanced": monthly_balanced,
        "cost_stress": cost_stress,
        "ai_stress": ai_stress,
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"model": str(args.model_out), "report": str(args.report_out), "sample_meta": sample_meta, "top": report["ranked_goal_mdd_lte_15"][:8], "monthly_balanced": monthly_balanced[:5]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
