#!/usr/bin/env python3
"""Train and gate the BTC v2 Direction + purged-OOF Meta candidate.

Historical validation is a development gate only.  Q1 is reported after the
candidate is frozen and never participates in selection.  Promotion remains
blocked until the preregistered post-2026-07-17 future window has enough data.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import btc_v2_research_core_20260716 as core  # noqa: E402


HOURLY_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706"
FIVE_MINUTE_DIR = ROOT / "data/splits/year_oos"
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708"
TRENDSCAN_DIR = ROOT / "tmp/causal_regen_20260516/btc_best_mean_pnl_trendscan_labels_20260715"
DEFAULT_ROOT = ROOT / "tmp/causal_regen_20260516/btc_v2_direction_meta_20260716"

TRAIN_END_EXCLUSIVE = pd.Timestamp("2025-08-29 00:00:00")
VALIDATION_START = pd.Timestamp("2025-09-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-12-31 23:55:00")
Q1_START = pd.Timestamp("2026-01-01 00:00:00")
Q1_END = pd.Timestamp("2026-03-31 23:55:00")
FUTURE_START = pd.Timestamp("2026-07-17 00:00:00")
FUTURE_MIN_DAYS = 90
FUTURE_MIN_TRADES = 50

NON_FEATURE = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
EXPECTED_FEATURE_COUNT = 28
MICROSTRUCTURE_COLUMNS = (
    "oi_change_rate",
    "net_taker_ratio",
    "taker_acceleration",
    "cvp_volume_imbalance",
    "funding_roc_12",
    "funding_roc_48",
    "funding_z_score",
    "funding_abs",
    "funding_pressure",
    "cvd_slope_12",
    "cvd_slope_48",
    "price_cvd_divergence",
    "cvd_breakout_z",
    "funding_oi_divergence",
    "oi_up_price_down",
    "oi_up_price_up",
)
META_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80)
DIRECTION_CONFIDENCE = (0.0, 0.45, 0.50, 0.55)
DIRECTION_MARGIN = (0.0, 0.05, 0.10, 0.15)
CONFIRMATION_HOURS = (1, 2, 3)
BALANCE_MODES = (False, True)
STRESS_COST = 0.0042


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_hourly() -> tuple[pd.DataFrame, list[str], list[Path]]:
    frames = []
    sources = []
    expected_columns: list[str] | None = None
    for year in (2024, 2025, 2026):
        path = HOURLY_DIR / f"sigma9_btc_1h_{year}.parquet"
        frame = pd.read_parquet(path)
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
        if expected_columns is None:
            expected_columns = list(frame.columns)
        elif list(frame.columns) != expected_columns:
            raise RuntimeError(f"hourly feature contract mismatch: {path}")
        frames.append(frame)
        sources.append(path)
    hourly = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    if not hourly["timestamp"].diff().dropna().eq(pd.Timedelta(hours=1)).all():
        raise RuntimeError("hourly BTC frame is not continuous")
    features = [column for column in hourly.columns if column not in NON_FEATURE]
    if len(features) != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(f"expected {EXPECTED_FEATURE_COUNT} stationary BTC features, got {len(features)}")
    forbidden = [
        column
        for column in features
        if column in {"open", "high", "low", "close"}
        or column.startswith(("btc_", "eth_", "cross_"))
        or any(token in column.lower() for token in ("target", "future", "label", "pnl"))
    ]
    if forbidden:
        raise RuntimeError(f"forbidden BTC v2 feature columns: {forbidden}")
    finite = np.isfinite(hourly[features].to_numpy(dtype=np.float64)).all(axis=1)
    first_complete = int(np.flatnonzero(finite)[0])
    if not finite[first_complete:].all():
        raise RuntimeError("non-finite hourly features after warm-up")
    hourly = hourly.iloc[first_complete:].reset_index(drop=True)
    return hourly, features, sources


def _read_labels(hourly: pd.DataFrame, direction_label: str) -> tuple[pd.DataFrame, list[Path]]:
    labels = []
    paths = []
    for year in (2024, 2025, 2026):
        if direction_label == "zigzag":
            path = ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv"
            frame = pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"])
            frame = frame.rename(columns={"zigzag_action": "direction_action"})
            frame["direction_weight"] = 1.0
        elif direction_label == "trendscan":
            path = TRENDSCAN_DIR / f"btc_1h_trendscan_t2_labels_{year}.parquet"
            frame = pd.read_parquet(path, columns=["timestamp", "action_id", "trend_t_value"])
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
            frame = frame.rename(columns={"action_id": "direction_action"})
            frame["direction_weight"] = np.clip(frame["trend_t_value"].abs(), 0.5, 12.0)
            frame = frame.drop(columns="trend_t_value")
        else:
            raise ValueError(f"unsupported direction label: {direction_label}")
        labels.append(frame)
        paths.append(path)
    label_frame = pd.concat(labels, ignore_index=True).drop_duplicates("timestamp")
    merged = hourly.merge(label_frame, on="timestamp", how="left", validate="one_to_one")
    if merged.loc[merged["timestamp"] <= Q1_END, "direction_action"].isna().any():
        raise RuntimeError(f"missing hourly {direction_label} label")
    merged["direction_action"] = merged["direction_action"].fillna(0).astype(np.int8)
    return merged, paths


def _read_tape(feature_set: str) -> tuple[pd.DataFrame, pd.DataFrame | None, list[Path]]:
    frames = []
    paths = []
    base_columns = ["timestamp", "open", "high", "low", "close"]
    usecols = base_columns + (list(MICROSTRUCTURE_COLUMNS) if feature_set == "f1" else [])
    for year in (2024, 2025, 2026):
        path = FIVE_MINUTE_DIR / f"btc_features_{year}.csv"
        frames.append(pd.read_csv(path, usecols=usecols, parse_dates=["timestamp"]))
        paths.append(path)
    combined = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    microstructure = None
    if feature_set == "f1":
        combined["hour"] = combined["timestamp"].dt.floor("h")
        microstructure = combined.groupby("hour", sort=True)[list(MICROSTRUCTURE_COLUMNS)].last().reset_index()
        microstructure = microstructure.rename(
            columns={"hour": "timestamp", **{column: f"btc_micro_{column}" for column in MICROSTRUCTURE_COLUMNS}}
        )
        micro_values = microstructure.drop(columns="timestamp").to_numpy(dtype=np.float64)
        if not np.isfinite(micro_values).all():
            raise RuntimeError("non-finite BTC microstructure feature")
    tape = combined[base_columns].copy()
    if not tape["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("BTC 5-minute execution tape is not continuous")
    previous_close = tape["close"].shift(1)
    true_range = pd.concat(
        [
            tape["high"] - tape["low"],
            (tape["high"] - previous_close).abs(),
            (tape["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tape["atr_pct"] = true_range.rolling(192, min_periods=192).mean() / tape["close"]
    if not np.isfinite(tape.loc[tape["timestamp"] >= pd.Timestamp("2024-01-02"), "atr_pct"]).all():
        raise RuntimeError("ATR is not finite after warm-up")
    return tape, microstructure, paths


def _fit_meta(
    x: np.ndarray,
    hourly: pd.DataFrame,
    tape: pd.DataFrame,
    oof_probability: np.ndarray,
    probability: np.ndarray,
    train_mask: np.ndarray,
    confirmation_hours: int,
    label_cost: float,
    reentry_hours: int,
    meta_model: str,
    horizon_bars: int,
    momentum_return: np.ndarray | None,
    meta_target: str,
    execution_contract: core.ExecutionContract,
    monthly_refit: bool,
) -> tuple[list[Any], np.ndarray, dict[str, Any]]:
    if monthly_refit and meta_model != "hgb":
        raise RuntimeError("monthly Meta refit currently requires meta_model=hgb")
    oof_rows = train_mask & np.isfinite(oof_probability).all(axis=1)
    oof_action = np.zeros(len(hourly), dtype=np.int8)
    oof_action[oof_rows] = oof_probability[oof_rows].argmax(axis=1).astype(np.int8)
    candidate = oof_rows & core.cadenced_events(
        oof_action, confirmation_hours=confirmation_hours, reentry_hours=reentry_hours
    )
    if momentum_return is not None:
        candidate &= ((oof_action == 1) & (momentum_return > 0.0)) | (
            (oof_action == 2) & (momentum_return < 0.0)
        )
    if meta_target == "terminal":
        target, net_return, eligible = core.terminal_meta_targets(
            hourly, oof_probability, candidate, tape, horizon_bars=horizon_bars, cost=label_cost
        )
    else:
        target, net_return, eligible = core.execution_meta_targets(
            hourly, oof_probability, candidate, tape, contract=execution_contract, cost=label_cost
        )
    meta_train = candidate & eligible
    if meta_train.sum() < 100 or np.unique(target[meta_train]).size != 2:
        raise RuntimeError(f"insufficient OOF meta rows: {int(meta_train.sum())}")
    oof_x = core.meta_matrix(x, np.nan_to_num(oof_probability, nan=0.0))
    inference_x = core.meta_matrix(x, probability)
    if meta_model == "none":
        models = []
        score = np.ones(len(x), dtype=np.float64)
        return models, score, {
            "rows": int(meta_train.sum()),
            "positive_rate": float(target[meta_train].mean()),
            "mean_net_return": float(np.mean(net_return[meta_train])),
            "label_cost": float(label_cost),
            "model": meta_model,
            "target": meta_target,
        }
    if meta_model == "hgb_reg":
        models = []
        clipped_target = np.clip(net_return[meta_train], -0.03, 0.03)
        raw_score = np.zeros(len(x), dtype=np.float64)
        train_raw = np.zeros(int(meta_train.sum()), dtype=np.float64)
        for seed in core.SEEDS:
            model = HistGradientBoostingRegressor(
                loss="absolute_error",
                learning_rate=0.035,
                max_iter=260,
                max_depth=4,
                max_leaf_nodes=31,
                min_samples_leaf=35,
                l2_regularization=1.0,
                early_stopping=False,
                random_state=int(seed),
            )
            model.fit(oof_x[meta_train], clipped_target)
            raw_score += model.predict(inference_x)
            train_raw += model.predict(oof_x[meta_train])
            models.append(model)
        raw_score /= len(models)
        train_raw /= len(models)
        reference = np.sort(train_raw)
        score = np.searchsorted(reference, raw_score, side="right") / max(len(reference), 1)
        return {"models": models, "score_reference": reference}, score, {
            "rows": int(meta_train.sum()),
            "positive_rate": float(target[meta_train].mean()),
            "mean_net_return": float(np.mean(net_return[meta_train])),
            "label_cost": float(label_cost),
            "model": meta_model,
            "target": meta_target,
            "loss": "absolute_error",
            "target_clip": [-0.03, 0.03],
        }
    if meta_model == "hgb_regime":
        regime, regime_detail = core.causal_regime_ids(hourly, train_mask)
        regime_models: dict[int, list[Any]] = {}
        score = np.zeros(len(x), dtype=np.float64)
        regime_rows: dict[str, int] = {}
        for state in range(4):
            state_train = meta_train & (regime == state)
            regime_rows[str(state)] = int(state_train.sum())
            if state_train.sum() < 100 or np.unique(target[state_train]).size != 2:
                raise RuntimeError(
                    f"insufficient regime Meta rows for state {state}: {int(state_train.sum())}"
                )
            regime_models[state] = core.fit_classifiers(oof_x[state_train], target[state_train])
            use = regime == state
            score[use] = core.predict_binary(regime_models[state], inference_x[use])
        return regime_models, score, {
            "rows": int(meta_train.sum()),
            "regime_rows": regime_rows,
            "regime_contract": regime_detail,
            "positive_rate": float(target[meta_train].mean()),
            "mean_net_return": float(np.mean(net_return[meta_train])),
            "label_cost": float(label_cost),
            "model": meta_model,
            "target": meta_target,
        }
    if meta_model == "hgb_side":
        oof_side = oof_probability.argmax(axis=1)
        inference_side = probability.argmax(axis=1)
        side_models: dict[int, list[Any]] = {}
        score = np.zeros(len(x), dtype=np.float64)
        for side in (1, 2):
            side_train = meta_train & (oof_side == side)
            if side_train.sum() < 100 or np.unique(target[side_train]).size != 2:
                raise RuntimeError(f"insufficient side-specific meta rows for side {side}: {int(side_train.sum())}")
            side_models[side] = core.fit_classifiers(oof_x[side_train], target[side_train])
            use = inference_side == side
            score[use] = core.predict_binary(side_models[side], inference_x[use])
        return side_models, score, {
            "rows": int(meta_train.sum()),
            "long_rows": int((meta_train & (oof_side == 1)).sum()),
            "short_rows": int((meta_train & (oof_side == 2)).sum()),
            "positive_rate": float(target[meta_train].mean()),
            "mean_net_return": float(np.mean(net_return[meta_train])),
            "label_cost": float(label_cost),
            "model": meta_model,
            "target": meta_target,
        }
    if meta_model == "hgb":
        models = core.fit_classifiers(oof_x[meta_train], target[meta_train])
    elif meta_model == "catboost":
        models = []
        for seed in core.SEEDS:
            model = CatBoostClassifier(
                loss_function="Logloss",
                iterations=320,
                depth=5,
                learning_rate=0.035,
                l2_leaf_reg=3.0,
                auto_class_weights="Balanced",
                random_seed=int(seed),
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            model.fit(oof_x[meta_train], target[meta_train])
            models.append(model)
    else:
        raise ValueError(f"unsupported meta model: {meta_model}")
    score = core.predict_binary(models, inference_x)
    refit_rows: list[dict[str, Any]] = []
    serialized_models: Any = models
    if monthly_refit:
        live_action = probability.argmax(axis=1).astype(np.int8)
        live_candidate = core.cadenced_events(
            live_action, confirmation_hours=confirmation_hours, reentry_hours=reentry_hours
        )
        if momentum_return is not None:
            live_candidate &= ((live_action == 1) & (momentum_return > 0.0)) | (
                (live_action == 2) & (momentum_return < 0.0)
            )
        if meta_target == "terminal":
            live_target, _, live_eligible = core.terminal_meta_targets(
                hourly, probability, live_candidate, tape, horizon_bars=horizon_bars, cost=label_cost
            )
        else:
            live_target, _, live_eligible = core.execution_meta_targets(
                hourly, probability, live_candidate, tape, contract=execution_contract, cost=label_cost
            )
        timestamp = pd.to_datetime(hourly["timestamp"])
        month_models: dict[str, list[Any]] = {}
        refit_x = inference_x.copy()
        refit_x[meta_train] = oof_x[meta_train]
        full_outcome_delay = pd.Timedelta(hours=1, minutes=5 * execution_contract.max_hold_bars)
        for month_start in pd.date_range(VALIDATION_START.normalize(), timestamp.max(), freq="MS"):
            next_month = month_start + pd.offsets.MonthBegin(1)
            supplement = (
                live_candidate
                & live_eligible
                & timestamp.ge(TRAIN_END_EXCLUSIVE).to_numpy()
                & timestamp.lt(month_start - full_outcome_delay).to_numpy()
            )
            refit_mask = meta_train | supplement
            refit_target = target.copy()
            refit_target[supplement] = live_target[supplement]
            if np.unique(refit_target[refit_mask]).size != 2:
                raise RuntimeError(f"monthly Meta refit lost a class at {month_start}")
            fitted = core.fit_classifiers(refit_x[refit_mask], refit_target[refit_mask])
            predict_month = timestamp.ge(month_start).to_numpy() & timestamp.lt(next_month).to_numpy()
            score[predict_month] = core.predict_binary(fitted, inference_x[predict_month])
            month_models[str(month_start.date())] = fitted
            refit_rows.append(
                {
                    "month": str(month_start.date()),
                    "fit_rows": int(refit_mask.sum()),
                    "supplement_rows": int(supplement.sum()),
                    "outcome_cutoff_exclusive": month_start - full_outcome_delay,
                }
            )
        serialized_models = {"base": models, "monthly": month_models}
    return serialized_models, score, {
        "rows": int(meta_train.sum()),
        "positive_rate": float(target[meta_train].mean()),
        "mean_net_return": float(np.mean(net_return[meta_train])),
        "label_cost": float(label_cost),
        "model": meta_model,
        "target": meta_target,
        "monthly_refit": monthly_refit,
        "monthly_refit_rows": refit_rows,
    }


def _fit_catboost_direction_oof(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
    *,
    balance_waves: bool,
    sample_weight: np.ndarray | None,
) -> tuple[list[Any], np.ndarray, np.ndarray, list[dict[str, Any]]]:
    train_indices = np.flatnonzero(train_mask)
    weights = (
        np.asarray(sample_weight, dtype=np.float64)[train_indices].copy()
        if sample_weight is not None
        else np.ones(len(train_indices), dtype=np.float64)
    )
    if balance_waves:
        weights *= core.wave_balanced_weights(y[train_indices])
    weights /= weights.mean()
    oof = np.full((len(x), 3), np.nan, dtype=np.float64)
    folds = []
    splitter = TimeSeriesSplit(n_splits=5, gap=core.PURGE_HOURS)
    for fold, (fit_local, test_local) in enumerate(splitter.split(train_indices), start=1):
        fit_idx, test_idx = train_indices[fit_local], train_indices[test_local]
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=420,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced",
            random_seed=int(core.SEEDS[(fold - 1) % len(core.SEEDS)]),
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x[fit_idx], y[fit_idx], sample_weight=weights[fit_local])
        oof[test_idx] = core.predict_probability([model], x[test_idx], 3)
        folds.append(
            {
                "fold": fold,
                "fit_start": int(fit_idx[0]),
                "fit_end": int(fit_idx[-1]),
                "oof_start": int(test_idx[0]),
                "oof_end": int(test_idx[-1]),
                "gap_rows": int(test_idx[0] - fit_idx[-1] - 1),
            }
        )
    models = []
    for seed in core.SEEDS:
        model = CatBoostClassifier(
            loss_function="MultiClass",
            iterations=420,
            depth=6,
            learning_rate=0.03,
            l2_leaf_reg=4.0,
            auto_class_weights="Balanced",
            random_seed=int(seed),
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        model.fit(x[train_indices], y[train_indices], sample_weight=weights)
        models.append(model)
    return models, core.predict_probability(models, x, 3), oof, folds


def _score_calibration(
    hourly: pd.DataFrame,
    tape: pd.DataFrame,
    direction_probability: np.ndarray,
    direction_event: np.ndarray,
    meta_score: np.ndarray,
    use_mask: np.ndarray,
    horizon_bars: int,
    meta_target: str,
    execution_contract: core.ExecutionContract,
) -> dict[str, Any]:
    if meta_target == "terminal":
        _, net_return, eligible = core.terminal_meta_targets(
            hourly, direction_probability, direction_event, tape, horizon_bars=horizon_bars
        )
    else:
        _, net_return, eligible = core.execution_meta_targets(
            hourly, direction_probability, direction_event, tape, contract=execution_contract
        )
    use = direction_event & eligible & use_mask & np.isfinite(meta_score)
    if use.sum() < 20:
        return {"rows": int(use.sum()), "spearman": None, "top_quintile_lift": None, "quintiles": []}
    frame = pd.DataFrame({"score": meta_score[use], "net_return": net_return[use]})
    if frame["score"].nunique() < 5:
        return {"rows": int(len(frame)), "spearman": None, "top_quintile_lift": None, "quintiles": []}
    spearman = float(frame["score"].corr(frame["net_return"], method="spearman"))
    frame["quintile"] = pd.qcut(frame["score"], 5, labels=False, duplicates="drop")
    quintiles = frame.groupby("quintile", observed=True)["net_return"].agg(["count", "mean"]).reset_index()
    top_mean = float(quintiles.iloc[-1]["mean"])
    lift = top_mean - float(frame["net_return"].mean())
    return {
        "rows": int(len(frame)),
        "spearman": spearman,
        "top_quintile_lift": lift,
        "quintiles": quintiles.to_dict(orient="records"),
    }


def _candidate_row(
    *,
    name: str,
    hourly: pd.DataFrame,
    tape: pd.DataFrame,
    direction_probability: np.ndarray,
    meta_score: np.ndarray,
    confirmation_hours: int,
    long_threshold: float,
    short_threshold: float,
    direction_confidence: float,
    direction_margin: float,
    calibrations: dict[tuple[float, float], dict[str, Any]],
    reentry_hours: int,
    execution_contract: core.ExecutionContract,
    momentum_return: np.ndarray | None,
) -> tuple[dict[str, Any], np.ndarray, pd.DataFrame, pd.DataFrame]:
    direction_action = direction_probability.argmax(axis=1).astype(np.int8)
    event = core.cadenced_events(
        direction_action, confirmation_hours=confirmation_hours, reentry_hours=reentry_hours
    )
    if momentum_return is not None:
        event &= ((direction_action == 1) & (momentum_return > 0.0)) | (
            (direction_action == 2) & (momentum_return < 0.0)
        )
    confidence = direction_probability.max(axis=1)
    margin = np.abs(direction_probability[:, 1] - direction_probability[:, 2])
    calibration = calibrations[(direction_confidence, direction_margin)]
    selected = (
        event
        & (
            ((direction_action == 1) & (meta_score >= long_threshold))
            | ((direction_action == 2) & (meta_score >= short_threshold))
        )
        & (confidence >= direction_confidence)
        & (margin >= direction_margin)
    )
    action = np.where(selected, direction_action, 0).astype(np.int8)
    signal = core.hourly_to_five_signal(hourly, action, tape)
    metrics, ledger, curve = core.replay(
        tape, signal, VALIDATION_START, VALIDATION_END, contract=execution_contract
    )
    stress, _, _ = core.replay(
        tape,
        signal,
        VALIDATION_START,
        VALIDATION_END,
        contract=execution_contract,
        round_trip_cost=STRESS_COST,
    )
    months = core.monthly_compound(ledger)
    positive_months = sum(value > 0.0 for value in months.values())
    concentration = core.top_trade_concentration(ledger)
    gates = {
        "positive_pnl": metrics["pnl_pct"] > 0.0,
        "mdd_within_8pct": metrics["mdd_pct"] >= -8.0,
        "at_least_40_trades": metrics["trades"] >= 40,
        "three_of_four_positive_months": len(months) == 4 and positive_months >= 3,
        "cost3_nonnegative": stress["pnl_pct"] >= 0.0,
        "score_spearman": calibration["spearman"] is not None and calibration["spearman"] > 0.10,
        "top_quintile_lift": calibration["top_quintile_lift"] is not None
        and calibration["top_quintile_lift"] >= 0.0015,
        "top3_concentration": concentration is not None and concentration <= 0.50,
    }
    row = {
        "name": name,
        "confirmation_hours": confirmation_hours,
        "reentry_hours": reentry_hours,
        "long_meta_threshold": long_threshold,
        "short_meta_threshold": short_threshold,
        "meta_threshold": long_threshold if long_threshold == short_threshold else None,
        "direction_confidence": direction_confidence,
        "direction_margin": direction_margin,
        **metrics,
        "cost3_pnl_pct": stress["pnl_pct"],
        "monthly_pnl_pct": months,
        "positive_months": positive_months,
        "score_calibration": calibration,
        "top3_profit_concentration": concentration,
        "gates": gates,
        "gates_passed": int(sum(gates.values())),
        "all_historical_gates_passed": bool(all(gates.values())),
    }
    return row, signal, ledger, curve


def _candidate_rank(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["gates_passed"],
        row["positive_months"],
        row["cost3_pnl_pct"],
        row["pnl_pct"],
        row["calmar"],
        -0.5 * (row["long_meta_threshold"] + row["short_meta_threshold"]),
    )


def _plot_equity(curve: pd.DataFrame, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.step(curve["timestamp"], 100.0 * (curve["equity"] - 1.0), where="post", color="#1565c0")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative PnL (%)")
    ax.grid(alpha=0.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--direction-label", choices=("zigzag", "trendscan"), default="zigzag")
    parser.add_argument("--meta-label-cost", type=float, default=core.ROUND_TRIP_COST)
    parser.add_argument("--feature-set", choices=("f0", "f1"), default="f0")
    parser.add_argument("--direction-feature-set", choices=("same", "f0"), default="same")
    parser.add_argument("--reentry-hours", type=int, default=0)
    parser.add_argument(
        "--meta-model",
        choices=("none", "hgb", "hgb_side", "hgb_regime", "hgb_reg", "catboost"),
        default="hgb",
    )
    parser.add_argument("--balance-mode", choices=("all", "unweighted", "weighted"), default="all")
    parser.add_argument("--confirmation-hours", type=int, choices=(1, 2, 3))
    parser.add_argument("--direction-model", choices=("hgb", "catboost"), default="hgb")
    parser.add_argument("--horizon-bars", type=int, choices=(72, 144, 288, 576), default=72)
    parser.add_argument("--momentum-confirm", type=int, choices=(0, 3, 6, 12), default=0)
    parser.add_argument("--meta-target", choices=("terminal", "execution"), default="terminal")
    parser.add_argument("--side-threshold-grid", action="store_true")
    parser.add_argument("--direction-label-weight", action="store_true")
    parser.add_argument("--monthly-meta-refit", action="store_true")
    parser.add_argument("--tp-atr", type=float, default=8.0)
    parser.add_argument("--sl-atr", type=float, default=4.0)
    parser.add_argument("--min-tp", type=float, default=0.008)
    parser.add_argument("--max-tp", type=float, default=0.030)
    parser.add_argument("--min-sl", type=float, default=0.005)
    parser.add_argument("--max-sl", type=float, default=0.015)
    args = parser.parse_args()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out_dir or DEFAULT_ROOT / run_id
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"immutable artifact directory already exists: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    hourly, features, hourly_sources = _read_hourly()
    hourly, label_sources = _read_labels(hourly, args.direction_label)
    tape, microstructure, tape_sources = _read_tape(args.feature_set)
    if microstructure is not None:
        common_end = pd.Timestamp(microstructure["timestamp"].max())
        hourly = hourly.loc[hourly["timestamp"] <= common_end].reset_index(drop=True)
        hourly = hourly.merge(microstructure, on="timestamp", how="left", validate="one_to_one")
        micro_features = [f"btc_micro_{column}" for column in MICROSTRUCTURE_COLUMNS]
        if hourly[micro_features].isna().any().any():
            raise RuntimeError("missing BTC microstructure feature after hourly merge")
        features = features + micro_features
    x = hourly[features].to_numpy(dtype=np.float64)
    direction_features = features if args.direction_feature_set == "same" else features[:EXPECTED_FEATURE_COUNT]
    direction_x = hourly[direction_features].to_numpy(dtype=np.float64)
    y = hourly["direction_action"].to_numpy(dtype=np.int8)
    direction_sample_weight = (
        hourly["direction_weight"].to_numpy(dtype=np.float64) if args.direction_label_weight else None
    )
    train_mask = hourly["timestamp"].lt(TRAIN_END_EXCLUSIVE).to_numpy()
    if not (0.0 < args.min_tp <= args.max_tp and 0.0 < args.min_sl <= args.max_sl):
        raise RuntimeError("invalid TP/SL clamp contract")
    execution_contract = core.ExecutionContract(
        tp_atr_multiple=args.tp_atr,
        sl_atr_multiple=args.sl_atr,
        min_tp=args.min_tp,
        max_tp=args.max_tp,
        min_sl=args.min_sl,
        max_sl=args.max_sl,
        max_hold_bars=args.horizon_bars,
    )
    momentum_return = (
        None
        if args.momentum_confirm == 0
        else hourly[f"logret_{args.momentum_confirm}"].to_numpy(dtype=np.float64)
    )

    search_rows: list[dict[str, Any]] = []
    bundles: dict[str, Any] = {}
    best_payload: tuple[dict[str, Any], np.ndarray, pd.DataFrame, pd.DataFrame] | None = None
    balance_modes = {
        "all": BALANCE_MODES,
        "unweighted": (False,),
        "weighted": (True,),
    }[args.balance_mode]
    confirmation_modes = (args.confirmation_hours,) if args.confirmation_hours else CONFIRMATION_HOURS
    for balance_waves in balance_modes:
        if args.direction_model == "hgb":
            direction_models, direction_probability, oof_probability, folds = core.fit_direction_oof(
                direction_x,
                y,
                train_mask,
                balance_waves=balance_waves,
                sample_weight=direction_sample_weight,
            )
        else:
            direction_models, direction_probability, oof_probability, folds = _fit_catboost_direction_oof(
                direction_x,
                y,
                train_mask,
                balance_waves=balance_waves,
                sample_weight=direction_sample_weight,
            )
        if not all(row["gap_rows"] == core.PURGE_HOURS for row in folds):
            raise RuntimeError("OOF embargo contract failed")
        for confirmation_hours in confirmation_modes:
            meta_models, meta_score, meta_detail = _fit_meta(
                x,
                hourly,
                tape,
                oof_probability,
                direction_probability,
                train_mask,
                confirmation_hours,
                args.meta_label_cost,
                args.reentry_hours,
                args.meta_model,
                args.horizon_bars,
                momentum_return,
                args.meta_target,
                execution_contract,
                args.monthly_meta_refit,
            )
            model_name = f"wave{int(balance_waves)}_confirm{confirmation_hours}_re{args.reentry_hours}"
            direction_action = direction_probability.argmax(axis=1).astype(np.int8)
            direction_event = core.cadenced_events(
                direction_action,
                confirmation_hours=confirmation_hours,
                reentry_hours=args.reentry_hours,
            )
            if momentum_return is not None:
                direction_event &= ((direction_action == 1) & (momentum_return > 0.0)) | (
                    (direction_action == 2) & (momentum_return < 0.0)
                )
            validation_mask = hourly["timestamp"].between(VALIDATION_START, VALIDATION_END).to_numpy()
            confidence = direction_probability.max(axis=1)
            margin = np.abs(direction_probability[:, 1] - direction_probability[:, 2])
            calibrations = {}
            for direction_confidence in DIRECTION_CONFIDENCE:
                for direction_margin in DIRECTION_MARGIN:
                    pre_meta_event = (
                        direction_event
                        & (confidence >= direction_confidence)
                        & (margin >= direction_margin)
                    )
                    calibrations[(direction_confidence, direction_margin)] = _score_calibration(
                        hourly,
                        tape,
                        direction_probability,
                        pre_meta_event,
                        meta_score,
                        validation_mask,
                        args.horizon_bars,
                        args.meta_target,
                        execution_contract,
                    )
            bundles[model_name] = {
                "direction_models": direction_models,
                "meta_models": meta_models,
                "folds": folds,
                "meta_detail": meta_detail,
            }
            threshold_pairs = (
                [(long_value, short_value) for long_value in META_THRESHOLDS for short_value in META_THRESHOLDS]
                if args.side_threshold_grid
                else [(value, value) for value in META_THRESHOLDS]
            )
            for long_threshold, short_threshold in threshold_pairs:
                for direction_confidence in DIRECTION_CONFIDENCE:
                    for direction_margin in DIRECTION_MARGIN:
                        payload = _candidate_row(
                            name=model_name,
                            hourly=hourly,
                            tape=tape,
                            direction_probability=direction_probability,
                            meta_score=meta_score,
                            confirmation_hours=confirmation_hours,
                            long_threshold=long_threshold,
                            short_threshold=short_threshold,
                            direction_confidence=direction_confidence,
                            direction_margin=direction_margin,
                            calibrations=calibrations,
                            reentry_hours=args.reentry_hours,
                            execution_contract=execution_contract,
                            momentum_return=momentum_return,
                        )
                        row = payload[0]
                        row["balance_waves"] = balance_waves
                        search_rows.append(row)
                        if best_payload is None or _candidate_rank(row) > _candidate_rank(best_payload[0]):
                            best_payload = payload

    if best_payload is None:
        raise RuntimeError("candidate search produced no result")
    best, best_signal, validation_ledger, validation_curve = best_payload
    best_bundle = bundles[best["name"]]

    q1_metrics, q1_ledger, q1_curve = core.replay(
        tape, best_signal, Q1_START, Q1_END, contract=execution_contract
    )
    q1_stress, _, _ = core.replay(
        tape, best_signal, Q1_START, Q1_END, contract=execution_contract, round_trip_cost=STRESS_COST
    )
    q1_metrics["cost3_pnl_pct"] = q1_stress["pnl_pct"]

    data_end = pd.Timestamp(tape["timestamp"].max())
    future_days = max(0, int((data_end - FUTURE_START).total_seconds() // 86400))
    future_available = data_end >= FUTURE_START
    future_metrics: dict[str, Any] | None = None
    future_ledger = pd.DataFrame()
    if future_available:
        future_metrics, future_ledger, _ = core.replay(
            tape, best_signal, FUTURE_START, data_end, contract=execution_contract
        )
    future_gate = {
        "start": FUTURE_START,
        "data_end": data_end,
        "available": future_available,
        "days": future_days,
        "minimum_days": FUTURE_MIN_DAYS,
        "trades": int(len(future_ledger)),
        "minimum_trades": FUTURE_MIN_TRADES,
        "passed": bool(
            future_metrics is not None
            and future_days >= FUTURE_MIN_DAYS
            and len(future_ledger) >= FUTURE_MIN_TRADES
            and future_metrics["pnl_pct"] > 0.0
        ),
        "metrics": future_metrics,
    }

    validation_ledger.to_csv(out_dir / "validation_ledger.csv", index=False)
    q1_ledger.to_csv(out_dir / "q1_diagnostic_ledger.csv", index=False)
    pd.DataFrame(
        [
            {
                key: value
                for key, value in row.items()
                if key not in {"monthly_pnl_pct", "score_calibration", "gates"}
            }
            for row in search_rows
        ]
    ).to_csv(out_dir / "candidate_search.csv", index=False)
    _plot_equity(validation_curve, out_dir / "validation_equity.png", "BTC v2 validation equity")
    _plot_equity(q1_curve, out_dir / "q1_diagnostic_equity.png", "BTC v2 Q1 diagnostic equity")

    bundle_path = out_dir / "btc_v2_research_bundle.joblib"
    joblib.dump(
        {
            "feature_columns": features,
            "direction_feature_columns": direction_features,
            "selected": best,
            "direction_models": best_bundle["direction_models"],
            "meta_models": best_bundle["meta_models"],
            "execution_contract": execution_contract,
            "train_end_exclusive": TRAIN_END_EXCLUSIVE,
        },
        bundle_path,
    )
    sources = hourly_sources + label_sources + tape_sources
    report = {
        "model_id": f"btc_v2_direction_meta_{run_id}",
        "status": "research_only_not_promoted",
        "selection_contract": {
            "training_end_exclusive": TRAIN_END_EXCLUSIVE,
            "validation": [VALIDATION_START, VALIDATION_END],
            "q1_diagnostic_only": [Q1_START, Q1_END],
            "q1_used_for_selection": False,
            "future_window": [FUTURE_START, None],
        },
        "feature_contract": {
            "count": len(features),
            "columns": features,
            "btc_native_stationary_only": True,
            "cross_asset_features": False,
            "legacy_aliases": False,
            "feature_set": args.feature_set,
            "direction_feature_set": args.direction_feature_set,
        },
        "direction_label": args.direction_label,
        "direction_model": args.direction_model,
        "direction_label_weight": args.direction_label_weight,
        "meta_label_cost": args.meta_label_cost,
        "meta_model": args.meta_model,
        "meta_target": args.meta_target,
        "monthly_meta_refit": args.monthly_meta_refit,
        "momentum_confirm_hours": args.momentum_confirm,
        "oof_contract": {
            "purge_hours": core.PURGE_HOURS,
            "true_direction_oof": True,
            "meta_uses_oof_direction_events_only": True,
            "folds": best_bundle["folds"],
            "meta_detail": best_bundle["meta_detail"],
        },
        "execution_contract": {
            **execution_contract.__dict__,
            "round_trip_cost": core.ROUND_TRIP_COST,
            "stress_cost": STRESS_COST,
            "same_bar_policy": "stop_first_conservative",
            "next_bar_entry": True,
        },
        "selected_candidate": best,
        "historical_gate_passed": best["all_historical_gates_passed"],
        "q1_diagnostic": q1_metrics,
        "future_gate": future_gate,
        "promotion_eligible": bool(best["all_historical_gates_passed"] and future_gate["passed"]),
        "fresh_forward_contract": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
        },
        "source_hashes": {str(path.relative_to(ROOT)): _sha256(path) for path in sources},
        "artifacts": {
            "bundle": bundle_path.name,
            "validation_ledger": "validation_ledger.csv",
            "q1_diagnostic_ledger": "q1_diagnostic_ledger.csv",
            "candidate_search": "candidate_search.csv",
        },
    }
    report_path = out_dir / "report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=_json_default)
    manifest = {
        path.name: _sha256(path)
        for path in sorted(out_dir.iterdir())
        if path.is_file() and path.name != "manifest.sha256.json"
    }
    with (out_dir / "manifest.sha256.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    print(
        json.dumps(
            {
                "out_dir": out_dir,
                "selected": best,
                "q1_diagnostic": q1_metrics,
                "future_gate": future_gate,
                "promotion_eligible": report["promotion_eligible"],
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0 if best["all_historical_gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
