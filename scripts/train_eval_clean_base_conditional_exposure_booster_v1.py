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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, FEATURE_COLS  # noqa: E402
from scripts.audit_current_rank1_baseline_safety_2026 import _decision_audit  # noqa: E402
from scripts.run_clean_scope_muzero_az_reaudit_2026 import realistic_ledger_replay  # noqa: E402
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    CONTEXT_COLS,
    MODEL_COLS,
    _base_frame,
    _date_codes,
    _days,
    _exit_probability_vec,
    _feature_vec_fast,
    _fill_price,
    _future_raw_from_entry,
)


DEFAULT_POLICY = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_v4_clean_train_to_2025_10.pkl"
DEFAULT_EXIT = ROOT / "data/ensemble/supervised/clean_scope_muzero_az_2026/hf_no_limit_exit_clean_train_to_2025_10.pkl"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/clean_scope_muzero_az_reaudit_2026.json"
DEFAULT_TRAIN_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL_CSV = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/clean_base_conditional_exposure_booster_v1/booster.pkl"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_conditional_exposure_booster_v1_2026.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_conditional_exposure_booster_v1_grid.csv"
DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_conditional_exposure_booster_v1_realistic_ledger.csv"

BOOST_VALUES = np.asarray([1.0, 1.15, 1.30, 1.45, 1.60], dtype=np.float64)
BASE_REFERENCE = {
    "pnl": 177.3298088749005,
    "mdd": -17.75966486035323,
    "trades": 363,
    "trades_per_day": 6.1875,
    "cost_2x_pnl": 92.25487780535948,
    "cost_3x_pnl": -7.969394502459748,
}

BOOSTER_CONTEXT_COLS = [
    "cb_side",
    "cb_notional",
    "cb_leverage",
    "cb_position_fraction",
    "cb_quality",
    "cb_confidence",
    "cb_cooldown",
    "cb_funding_abs",
    "cb_funding_pressure",
    "cb_liquidity_vacuum",
    "cb_amihud_illiquidity_z",
    "cb_m7_tail_risk",
    "cb_evt_tail_flag",
    "cb_daily_realized",
    "cb_daily_dd",
    "cb_account_dd",
    "cb_loss_streak",
    "cb_loss_cooldown_left",
]
BOOSTER_COLS = list(FEATURE_COLS) + BOOSTER_CONTEXT_COLS


@dataclass(frozen=True)
class BoosterRuntimeConfig:
    name: str
    prob_floor: float
    max_boost: float
    max_notional: float
    boost_account_dd_cap: float
    boost_daily_dd_cap: float
    allow_after_loss_streak: bool


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _split_train_validation(df: pd.DataFrame, split_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    split = pd.Timestamp(split_date)
    return df.loc[ts < split].reset_index(drop=True), df.loc[ts >= split].reset_index(drop=True)


def _num_col(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return (
        pd.to_numeric(df[col], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .fillna(float(default))
        .to_numpy(dtype=np.float64)
    )


def _range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    return [str(df["timestamp"].iloc[0]), str(df["timestamp"].iloc[-1])]


def _predict_label_proba(model: Any, row: np.ndarray) -> dict[int, float]:
    with np.errstate(all="ignore"):
        proba = model.predict_proba(row.reshape(1, -1).astype(np.float32, copy=False))[0]
    classes = np.asarray(model.classes_, dtype=int)
    return {int(c): float(proba[i]) for i, c in enumerate(classes)}


def _class_to_boost(label: int) -> float:
    idx = int(np.clip(label, 0, len(BOOST_VALUES) - 1))
    return float(BOOST_VALUES[idx])


def _booster_row(
    base_values: np.ndarray,
    decisions: pd.DataFrame,
    df: pd.DataFrame,
    *,
    i: int,
    side: int,
    notional: float,
    leverage: float,
    daily_realized: float,
    daily_dd: float,
    account_dd: float,
    loss_streak: int,
    loss_cooldown_left: int,
) -> dict[str, float]:
    dec = decisions.iloc[int(i)]
    row = {col: float(base_values[int(i), j]) for j, col in enumerate(FEATURE_COLS)}
    row.update(
        {
            "cb_side": float(side),
            "cb_notional": float(notional),
            "cb_leverage": float(leverage),
            "cb_position_fraction": float(notional / max(leverage, 1e-12)),
            "cb_quality": float(getattr(dec, "quality_score", 0.0)),
            "cb_confidence": float(getattr(dec, "confidence", 0.0)),
            "cb_cooldown": float(getattr(dec, "cooldown_bars", 0.0)),
            "cb_funding_abs": float(abs(row.get("funding_abs", 0.0))),
            "cb_funding_pressure": float(abs(row.get("funding_pressure", 0.0))),
            "cb_liquidity_vacuum": float(row.get("liquidity_vacuum", 0.0)),
            "cb_amihud_illiquidity_z": float(row.get("amihud_illiquidity_z", 0.0)),
            "cb_m7_tail_risk": float(row.get("m7_tail_risk", 0.0)),
            "cb_evt_tail_flag": float(row.get("evt_tail_flag", 0.0)),
            "cb_daily_realized": float(daily_realized),
            "cb_daily_dd": float(daily_dd),
            "cb_account_dd": float(account_dd),
            "cb_loss_streak": float(loss_streak),
            "cb_loss_cooldown_left": float(loss_cooldown_left),
        }
    )
    return {c: float(row.get(c, 0.0) or 0.0) for c in BOOSTER_COLS}


def _state_thresholds(df: pd.DataFrame, base_feat: pd.DataFrame, decisions: pd.DataFrame) -> dict[str, float]:
    active = (
        (decisions["action"].astype(int).to_numpy() != ACTION_CASH)
        & (decisions["side"].astype(int).to_numpy() != 0)
        & (pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64) > 0.0)
    )
    feat = base_feat.loc[active] if active.any() else base_feat
    def q(col: str, quantile: float, fallback: float) -> float:
        if col not in feat.columns or feat.empty:
            return float(fallback)
        values = pd.to_numeric(feat[col], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if values.empty:
            return float(fallback)
        return float(np.quantile(np.abs(values.to_numpy(dtype=np.float64)), quantile))

    return {
        "funding_abs_p85": q("funding_abs", 0.85, 0.001),
        "funding_pressure_p85": q("funding_pressure", 0.85, 1.0),
        "liquidity_vacuum_p85": q("liquidity_vacuum", 0.85, 1.0),
        "amihud_illiquidity_z_p85": q("amihud_illiquidity_z", 0.85, 2.0),
        "m7_tail_risk_p85": q("m7_tail_risk", 0.85, 1.0),
    }


def _bad_boost_state(row: dict[str, float], thresholds: dict[str, float], cfg: BoosterRuntimeConfig) -> bool:
    if float(row.get("cb_account_dd", 0.0)) > float(cfg.boost_account_dd_cap):
        return True
    if float(row.get("cb_daily_dd", 0.0)) > float(cfg.boost_daily_dd_cap):
        return True
    if not bool(cfg.allow_after_loss_streak) and float(row.get("cb_loss_streak", 0.0)) > 0.0:
        return True
    if float(row.get("cb_funding_abs", 0.0)) > float(thresholds["funding_abs_p85"]):
        return True
    if float(row.get("cb_funding_pressure", 0.0)) > float(thresholds["funding_pressure_p85"]):
        return True
    if float(row.get("cb_liquidity_vacuum", 0.0)) > float(thresholds["liquidity_vacuum_p85"]):
        return True
    if float(row.get("cb_amihud_illiquidity_z", 0.0)) > float(thresholds["amihud_illiquidity_z_p85"]):
        return True
    if float(row.get("cb_m7_tail_risk", 0.0)) > float(thresholds["m7_tail_risk_p85"]):
        return True
    if float(row.get("cb_evt_tail_flag", 0.0)) > 0.5:
        return True
    return False


def _predict_boost(
    model: Any,
    row: dict[str, float],
    thresholds: dict[str, float],
    cfg: BoosterRuntimeConfig,
) -> tuple[float, float, int]:
    if _bad_boost_state(row, thresholds, cfg):
        return 1.0, 1.0, 0
    arr = np.asarray([float(row.get(c, 0.0) or 0.0) for c in BOOSTER_COLS], dtype=np.float32)
    probs = _predict_label_proba(model, arr)
    allowed = [idx for idx, val in enumerate(BOOST_VALUES) if float(val) <= float(cfg.max_boost) + 1e-12]
    if not allowed:
        return 1.0, 1.0, 0
    best = max(allowed, key=lambda idx: probs.get(int(idx), 0.0))
    prob = float(probs.get(int(best), 0.0))
    if best <= 0 or prob < float(cfg.prob_floor):
        return 1.0, prob, int(best)
    return _class_to_boost(best), prob, int(best)


def _make_training_set(
    train_df: pd.DataFrame,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    horizon: int,
    sample_stride: int,
    max_notional: float,
    min_label_edge: float,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any], dict[str, float]]:
    base_feat, decisions, _close, fill_px = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    thresholds = _state_thresholds(train_df, base_feat, decisions)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    active = np.flatnonzero((actions != ACTION_CASH) & (sides != 0) & (notionals > 1e-8))
    active = active[active < len(train_df) - int(horizon) - 3]
    if int(sample_stride) > 1:
        active = active[:: int(sample_stride)]
    rng = np.random.default_rng(int(seed))
    if len(active) > 80000:
        active = np.sort(rng.choice(active, size=80000, replace=False))

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    weights: list[float] = []
    score_margin: list[float] = []
    bad_state_count = 0

    neutral_cfg = BoosterRuntimeConfig(
        name="label_neutral",
        prob_floor=0.0,
        max_boost=float(BOOST_VALUES[-1]),
        max_notional=float(max_notional),
        boost_account_dd_cap=1.0,
        boost_daily_dd_cap=1.0,
        allow_after_loss_streak=True,
    )

    for i in active:
        i = int(i)
        side = int(sides[i])
        base_n = float(notionals[i])
        lev = float(leverages[i])
        row = _booster_row(
            base_values,
            decisions,
            train_df,
            i=i,
            side=side,
            notional=base_n,
            leverage=lev,
            daily_realized=0.0,
            daily_dd=0.0,
            account_dd=0.0,
            loss_streak=0,
            loss_cooldown_left=0,
        )
        entry_price = _fill_price(fill_px, min(i + 1, len(train_df) - 1), side, float(slip), entry=True)
        raw = _future_raw_from_entry(fill_px, i + 2, min(i + int(horizon) + 1, len(train_df) - 1), side, entry_price, float(slip))
        if raw.size < 3:
            continue
        scores = []
        for boost in BOOST_VALUES:
            n = min(base_n * float(boost), float(max_notional))
            path = raw * n - 2.0 * float(fee) * n
            best = float(np.max(path))
            final = float(path[-1])
            mean_front = float(np.mean(path[: min(len(path), 24)]))
            adverse = max(0.0, -float(np.min(path)))
            peak_to_trough = float(np.max(np.maximum.accumulate(path) - path))
            size_penalty = 0.006 * (n / max(float(max_notional), 1e-12)) ** 2
            scores.append(0.50 * best + 0.25 * final + 0.25 * mean_front - 2.20 * adverse - 0.75 * peak_to_trough - size_penalty)
        scores_arr = np.asarray(scores, dtype=np.float64)
        best_idx = int(np.argmax(scores_arr))
        margin = float(scores_arr[best_idx] - scores_arr[0])
        if _bad_boost_state(row, thresholds, neutral_cfg):
            best_idx = 0
            bad_state_count += 1
        elif best_idx > 0 and margin < float(min_label_edge):
            best_idx = 0
        rows.append(row)
        labels.append(best_idx)
        weights.append(float(1.0 + min(8.0, max(0.0, margin) * 300.0)))
        score_margin.append(margin)

    x = pd.DataFrame(rows, columns=BOOSTER_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    unique, counts = np.unique(y, return_counts=True) if y.size else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    meta = {
        "samples": int(len(x)),
        "active_candidates": int(len(active)),
        "horizon": int(horizon),
        "sample_stride": int(sample_stride),
        "boost_values": [float(v) for v in BOOST_VALUES],
        "label_distribution": {str(int(k)): int(v) for k, v in zip(unique, counts)},
        "bad_state_forced_base_labels": int(bad_state_count),
        "score_margin_mean": float(np.mean(score_margin)) if score_margin else 0.0,
        "score_margin_p95": float(np.quantile(score_margin, 0.95)) if score_margin else 0.0,
    }
    return x, y, w, meta, thresholds


def _train_booster(x: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, seed: int) -> Any:
    if len(np.unique(y)) <= 1:
        class ConstantBooster:
            classes_ = np.asarray([0], dtype=np.int64)

            def predict_proba(self, arr: np.ndarray) -> np.ndarray:
                return np.ones((len(arr), 1), dtype=np.float64)

        return ConstantBooster()
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.035,
            max_leaf_nodes=15,
            min_samples_leaf=90,
            l2_regularization=0.12,
            random_state=int(seed),
        ),
    )
    model.fit(x.to_numpy(dtype=np.float32, copy=False), y, histgradientboostingclassifier__sample_weight=w)
    return model


def backtest_booster(
    df: pd.DataFrame,
    exit_model: Any,
    booster_model: Any,
    thresholds: dict[str, float],
    runtime_cfg: BoosterRuntimeConfig,
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    base_feat, decisions, close, fill_px = precomputed
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
    boost_counts: dict[str, int] = {}
    boost_sum = 0.0
    boosted_entries = 0
    max_entry_notional = 0.0
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
        cash -= before * float(fee) * notional
        trades += 1
        daily_trades += 1
        is_win = cash > entry_equity
        wins += int(is_win)
        loss_streak = 0 if is_win else loss_streak + 1
        if not is_win:
            loss_cooldown_left = max(loss_cooldown_left, int(risk_cfg.get("loss_cooldown_bars", 0)))
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
            if age >= int(exit_cfg["min_exit_age"]):
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
                if p_exit >= float(exit_cfg["exit_threshold"]):
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
        if daily_trades >= int(risk_cfg.get("max_daily_trades", 999999)):
            block("daily_trade_budget")
            continue
        if daily_realized <= -abs(float(risk_cfg.get("daily_loss_limit", 0.0))):
            block("daily_loss_lock")
            continue
        if daily_dd >= abs(float(risk_cfg.get("daily_dd_limit", 0.0))):
            block("daily_dd_lock")
            continue
        if int(actions[i]) == ACTION_CASH or int(sides[i]) == 0:
            block("cash_signal")
            continue

        n = float(notionals[i])
        if account_dd >= float(risk_cfg.get("global_dd_cut", 999.0)):
            n *= float(risk_cfg.get("global_dd_mult", 1.0))
        if loss_streak >= int(risk_cfg.get("loss_streak_soft", 999999)):
            steps = loss_streak - int(risk_cfg.get("loss_streak_soft", 999999)) + 1
            n *= float(risk_cfg.get("loss_streak_mult", 1.0)) ** float(max(0, steps))
        if daily_realized >= float(risk_cfg.get("daily_profit_boost_start", 999.0)):
            n *= float(risk_cfg.get("daily_profit_boost_mult", 1.0))
        if n <= 1e-8:
            block("zero_notional")
            continue

        lev = float(leverages[i])
        booster_features = _booster_row(
            base_values,
            decisions,
            df,
            i=i,
            side=int(sides[i]),
            notional=n,
            leverage=lev,
            daily_realized=daily_realized,
            daily_dd=daily_dd,
            account_dd=account_dd,
            loss_streak=loss_streak,
            loss_cooldown_left=loss_cooldown_left,
        )
        boost, _prob, _label = _predict_boost(booster_model, booster_features, thresholds, runtime_cfg)
        n = float(np.clip(n * float(boost), 0.0, float(runtime_cfg.max_notional)))
        if n <= 1e-8:
            block("zero_notional")
            continue
        boost_key = f"{boost:.2f}"
        boost_counts[boost_key] = boost_counts.get(boost_key, 0) + 1
        boost_sum += float(boost)
        boosted_entries += int(boost > 1.000001)
        max_entry_notional = max(max_entry_notional, n)

        pos = int(sides[i])
        entry_price = _fill_price(fill_px, min(i + 1, len(df) - 1), pos, slip, entry=True)
        entry_equity = cash
        entry_idx = i
        notional = n
        leverage = lev
        model_cooldown = int(cooldowns[i])
        cash -= cash * float(fee) * notional
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
        "boost_counts": boost_counts,
        "boosted_entries": int(boosted_entries),
        "avg_boost": float(boost_sum / entries),
        "max_entry_notional": float(max_entry_notional),
    }


def _static_boost_precomputed(
    df: pd.DataFrame,
    booster_model: Any,
    thresholds: dict[str, float],
    runtime_cfg: BoosterRuntimeConfig,
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    base_feat, dec, close, fill = precomputed
    base_values = base_feat.to_numpy(dtype=np.float32, copy=False)
    out = dec.copy()
    sides = out["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (sides != 0) & (notionals > 1e-8)
    for i in np.flatnonzero(active):
        row = _booster_row(
            base_values,
            out,
            df,
            i=int(i),
            side=int(sides[int(i)]),
            notional=float(notionals[int(i)]),
            leverage=float(leverages[int(i)]),
            daily_realized=0.0,
            daily_dd=0.0,
            account_dd=0.0,
            loss_streak=0,
            loss_cooldown_left=0,
        )
        boost, _prob, _label = _predict_boost(booster_model, row, thresholds, runtime_cfg)
        new_n = float(np.clip(float(notionals[int(i)]) * boost, 0.0, float(runtime_cfg.max_notional)))
        out.loc[int(i), "notional_exposure"] = new_n
        out.loc[int(i), "position_fraction"] = new_n / max(float(leverages[int(i)]), 1e-12)
    return base_feat, out, close, fill


def _score(metrics: dict[str, Any]) -> float:
    pnl = float(metrics.get("pnl", -1e9))
    mdd = float(metrics.get("mdd", -1e9))
    tpd = float(metrics.get("trades_per_day", 0.0))
    coverage_penalty = max(0.0, 5.5 - tpd) * 65.0
    mdd_penalty = max(0.0, -17.759665 - mdd) * 8.0
    return pnl + 5.0 * mdd - coverage_penalty - mdd_penalty


def _runtime_grid(max_notional_values: list[float]) -> list[BoosterRuntimeConfig]:
    rows: list[BoosterRuntimeConfig] = []
    for max_notional in max_notional_values:
        for max_boost in (1.15, 1.30, 1.45, 1.60):
            for prob_floor in (0.30, 0.40, 0.50, 0.60):
                for account_dd_cap, daily_dd_cap in ((0.015, 0.008), (0.030, 0.015), (0.050, 0.020)):
                    name = f"boost{max_boost:.2f}_p{prob_floor:.2f}_maxn{max_notional:.1f}_add{account_dd_cap:.3f}_ddd{daily_dd_cap:.3f}"
                    rows.append(
                        BoosterRuntimeConfig(
                            name=name,
                            prob_floor=float(prob_floor),
                            max_boost=float(max_boost),
                            max_notional=float(max_notional),
                            boost_account_dd_cap=float(account_dd_cap),
                            boost_daily_dd_cap=float(daily_dd_cap),
                            allow_after_loss_streak=False,
                        )
                    )
    return rows


def _promotable(metrics: dict[str, Any], cost: dict[str, dict[str, Any]], invariant: dict[str, Any]) -> bool:
    return bool(
        float(metrics.get("pnl", -1e9)) >= BASE_REFERENCE["pnl"]
        and float(metrics.get("mdd", -1e9)) >= BASE_REFERENCE["mdd"]
        and float(metrics.get("trades_per_day", 0.0)) >= 5.5
        and float(cost["cost_1x"].get("pnl", -1e9)) > 0.0
        and float(cost["cost_2x"].get("pnl", -1e9)) > 0.0
        and bool(invariant.get("passed", False))
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate clean-base conditional exposure booster v1.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-model", type=Path, default=DEFAULT_EXIT)
    p.add_argument("--audit-report", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--split-date", default="2025-11-01")
    p.add_argument("--horizon", type=int, default=96)
    p.add_argument("--sample-stride", type=int, default=2)
    p.add_argument("--min-label-edge", type=float, default=0.0012)
    p.add_argument("--max-notionals", default="3.6,4.2,4.8,5.4")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--grid-csv-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--ledger-out", type=Path, default=DEFAULT_LEDGER)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    policy = joblib.load(args.policy)
    exit_payload = joblib.load(args.exit_model)
    exit_model = exit_payload["model"] if isinstance(exit_payload, dict) and "model" in exit_payload else exit_payload
    audit = json.load(args.audit_report.open("r", encoding="utf-8"))
    selected = audit["control_selection"]["selected"]
    entry_cfg = dict(selected["entry_config"])
    risk_cfg_base = dict(selected["risk_config"])
    exit_cfg = dict(selected["exit_config"])

    train_full = _read(args.train_csv)
    train_df, val_df = _split_train_validation(train_full, args.split_date)
    eval_df = _read(args.eval_csv)

    train_pre = _base_frame(train_df, policy, entry_cfg)
    val_pre = _base_frame(val_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)

    max_notional_values = [float(x.strip()) for x in str(args.max_notionals).split(",") if x.strip()]
    label_max_notional = max(max_notional_values)
    x, y, w, label_meta, thresholds = _make_training_set(
        train_df,
        train_pre,
        fee=float(args.fee),
        slip=float(args.slip),
        horizon=int(args.horizon),
        sample_stride=int(args.sample_stride),
        max_notional=float(label_max_notional),
        min_label_edge=float(args.min_label_edge),
        seed=int(args.seed),
    )
    booster_model = _train_booster(x, y, w, seed=int(args.seed))
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": booster_model,
            "feature_cols": list(BOOSTER_COLS),
            "boost_values": BOOST_VALUES.astype(float).tolist(),
            "thresholds": thresholds,
            "label_meta": label_meta,
            "base_policy": str(args.policy),
            "base_exit_model": str(args.exit_model),
            "entry_config": entry_cfg,
            "risk_config": risk_cfg_base,
            "exit_config": exit_cfg,
        },
        args.model_out,
    )

    val_rows: list[dict[str, Any]] = []
    for runtime_cfg in _runtime_grid(max_notional_values):
        risk_cfg = dict(risk_cfg_base)
        risk_cfg["max_notional"] = float(runtime_cfg.max_notional)
        val_metrics = backtest_booster(
            val_df,
            exit_model,
            booster_model,
            thresholds,
            runtime_cfg,
            risk_cfg,
            exit_cfg,
            val_pre,
            fee=float(args.fee),
            slip=float(args.slip),
        )
        val_rows.append({"runtime_config": asdict(runtime_cfg), "validation": val_metrics, "validation_score": _score(val_metrics)})

    selected_balanced = max(val_rows, key=lambda r: float(r["validation_score"]))
    constrained = [
        r
        for r in val_rows
        if float(r["validation"]["pnl"]) >= float(selected["eval"]["pnl"])
        and float(r["validation"]["mdd"]) >= float(selected["eval"]["mdd"]) - 2.0
        and float(r["validation"]["trades_per_day"]) >= 10.5
    ]
    selected_constrained = max(constrained, key=lambda r: float(r["validation"]["pnl"])) if constrained else selected_balanced
    selected_max_pnl = max(val_rows, key=lambda r: float(r["validation"]["pnl"]))
    selection = {
        "balanced_score": selected_balanced,
        "redteam_constrained": selected_constrained,
        "max_validation_pnl": selected_max_pnl,
    }

    eval_results: dict[str, Any] = {}
    for label, row in selection.items():
        runtime_cfg = BoosterRuntimeConfig(**row["runtime_config"])
        risk_cfg = dict(risk_cfg_base)
        risk_cfg["max_notional"] = float(runtime_cfg.max_notional)
        cost = {
            f"cost_{mult:g}x": backtest_booster(
                eval_df,
                exit_model,
                booster_model,
                thresholds,
                runtime_cfg,
                risk_cfg,
                exit_cfg,
                eval_pre,
                fee=float(args.fee) * mult,
                slip=float(args.slip) * mult,
            )
            for mult in (1.0, 2.0, 3.0)
        }
        static_pre = _static_boost_precomputed(eval_df, booster_model, thresholds, runtime_cfg, eval_pre)
        _feat, dec, _close, _fill = static_pre
        invariant = _decision_audit(dec, max_notional=float(runtime_cfg.max_notional), leverage_cap=5.0)
        realistic = realistic_ledger_replay(
            eval_df,
            exit_model,
            risk_cfg,
            exit_cfg,
            static_pre,
            fee=float(args.fee),
            slip=float(args.slip),
            funding_mult=1.0,
            impact_per_notional=0.00008,
            partial_fill_ratio=0.96,
            maintenance_margin=0.006,
            liquidation_fee=0.002,
        )
        eval_results[label] = {
            "runtime_config": asdict(runtime_cfg),
            "validation": row["validation"],
            "validation_score": row["validation_score"],
            "oos": cost["cost_1x"],
            "cost_stress": cost,
            "decision_invariant_audit": invariant,
            "realistic_replay_static_diagnostic": realistic["eval"],
            "promotable_by_contract": _promotable(cost["cost_1x"], cost, invariant),
        }
        if label == "redteam_constrained":
            args.ledger_out.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(realistic["ledger"]).to_csv(args.ledger_out, index=False)
            eval_results[label]["realistic_ledger"] = str(args.ledger_out)

    args.grid_csv_out.parent.mkdir(parents=True, exist_ok=True)
    with args.grid_csv_out.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "name",
            "prob_floor",
            "max_boost",
            "max_notional",
            "boost_account_dd_cap",
            "boost_daily_dd_cap",
            "val_pnl",
            "val_mdd",
            "val_trades",
            "val_trades_per_day",
            "val_avg_notional",
            "val_avg_boost",
            "val_boosted_entries",
            "validation_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True):
            cfg = row["runtime_config"]
            val = row["validation"]
            writer.writerow(
                {
                    "name": cfg["name"],
                    "prob_floor": cfg["prob_floor"],
                    "max_boost": cfg["max_boost"],
                    "max_notional": cfg["max_notional"],
                    "boost_account_dd_cap": cfg["boost_account_dd_cap"],
                    "boost_daily_dd_cap": cfg["boost_daily_dd_cap"],
                    "val_pnl": val["pnl"],
                    "val_mdd": val["mdd"],
                    "val_trades": val["trades"],
                    "val_trades_per_day": val["trades_per_day"],
                    "val_avg_notional": val["avg_notional"],
                    "val_avg_boost": val["avg_boost"],
                    "val_boosted_entries": val["boosted_entries"],
                    "validation_score": row["validation_score"],
                }
            )

    report = {
        "type": "clean_base_conditional_exposure_booster_v1",
        "note": "Frozen clean base entry/side/timing. Booster can only multiply notional on admitted base entries; it cannot block entries or flip side.",
        "data": {
            "train_range": _range(train_df),
            "train_rows": int(len(train_df)),
            "validation_range": _range(val_df),
            "validation_rows": int(len(val_df)),
            "eval_range": _range(eval_df),
            "eval_rows": int(len(eval_df)),
        },
        "base_reference": BASE_REFERENCE,
        "artifacts": {
            "booster_model": str(args.model_out),
            "grid_csv": str(args.grid_csv_out),
            "redteam_constrained_realistic_ledger": str(args.ledger_out),
        },
        "training": {
            "label_meta": label_meta,
            "thresholds": thresholds,
            "feature_cols": BOOSTER_COLS,
        },
        "validation_top10": [
            {"runtime_config": r["runtime_config"], "validation": r["validation"], "validation_score": r["validation_score"]}
            for r in sorted(val_rows, key=lambda r: float(r["validation_score"]), reverse=True)[:10]
        ],
        "selected": {label: row["runtime_config"]["name"] for label, row in selection.items()},
        "selected_eval": eval_results,
        "promotion_gate": {
            "oos_pnl_min": BASE_REFERENCE["pnl"],
            "oos_mdd_min": BASE_REFERENCE["mdd"],
            "trades_per_day_min": 5.5,
            "cost_1x_2x_positive": True,
            "cost_3x_reported": True,
            "decision_invariant_required": True,
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "model": str(args.model_out),
                "selected": report["selected"],
                "redteam_constrained_oos": eval_results["redteam_constrained"]["oos"],
                "redteam_constrained_promotable": eval_results["redteam_constrained"]["promotable_by_contract"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

