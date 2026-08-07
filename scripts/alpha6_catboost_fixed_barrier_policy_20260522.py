#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "alpha6_catboost_fixed_barrier_policy_20260522"
DEFAULT_FEATURE_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"
DEFAULT_SPEC_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_fixed_barrier_policy_20260522"


@dataclass(frozen=True)
class PolicyConfig:
    notional_buckets: tuple[float, ...] = (0.10, 0.15, 0.25, 0.40, 0.65, 1.00)
    tp_atr_buckets: tuple[float, ...] = (0.8, 1.2, 1.6, 2.2, 3.0)
    sl_atr_buckets: tuple[float, ...] = (0.5, 0.8, 1.1, 1.5, 2.0)
    max_hold_buckets: tuple[int, ...] = (3, 6, 12, 24, 48, 96)
    cooldown_buckets: tuple[int, ...] = (0, 1, 3, 6, 12)
    max_train_horizon_bars: int = 96
    fixed_notional: float = 0.25
    fixed_tp_atr: float = 2.2
    fixed_sl_atr: float = 1.1
    fixed_max_hold_bars: int = 24
    fixed_cooldown_bars: int = 3
    efficient_tp_bars: int = 0
    fee: float = 0.0004
    slip: float = 0.00015
    cash_score: float = 0.0
    min_net_edge: float = 0.00045
    dynamic_min_edge_atr_frac: float = 0.15
    direction_margin: float = 0.00025
    mae_penalty_lambda: float = 0.35
    tp_min: float = 0.0015
    tp_max: float = 0.0120
    sl_min: float = 0.0008
    sl_max: float = 0.0100


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_spec(spec_dir: Path, variant: str) -> dict[str, Any]:
    path = spec_dir / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(path)
    spec = json.loads(path.read_text())
    spec["features"] = list(spec.get("features") or spec.get("feature_cols") or [])
    if not spec["features"]:
        raise ValueError(f"empty feature spec: {path}")
    return spec


def _label_frame(label_dir: Path) -> pd.DataFrame:
    frames = []
    wanted = ["timestamp", "label_action", "dataset_split"]
    for name in ("alpha5_13_hgb_atr_barrier_labels_train.parquet", "alpha5_13_hgb_atr_barrier_labels_val.parquet"):
        path = label_dir / name
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[c for c in wanted if c in available])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).dropna(subset=["timestamp"])
    return out.drop_duplicates("timestamp", keep="last")


def _read_feature_frame(feature_csv: Path, features: list[str], extra_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    columns = pd.read_csv(feature_csv, nrows=0).columns.tolist()
    available = set(columns)
    present = [c for c in features if c in available]
    missing = [c for c in features if c not in available]
    keep = []
    for col in ["timestamp", "open", "high", "low", "close", "atr14_pct", *extra_cols, *present]:
        if col in available and col not in keep:
            keep.append(col)
    frame = pd.read_csv(feature_csv, usecols=keep, parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame, present, missing


def _numeric_matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = frame[cols].copy()
    for col in cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x.replace([np.inf, -np.inf], np.nan)


def _feature_matrix(
    train: pd.DataFrame,
    val: pd.DataFrame,
    cols: list[str],
    *,
    use_pca: bool,
    pca_components: int,
) -> tuple[np.ndarray, np.ndarray, list[str], Pipeline]:
    x_train = _numeric_matrix(train, cols)
    x_val = _numeric_matrix(val, cols)
    if use_pca:
        n_comp = int(max(1, min(pca_components, len(cols), len(train) - 1)))
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=n_comp, random_state=42)),
            ]
        )
        return pipe.fit_transform(x_train), pipe.transform(x_val), [f"pca_{i:02d}" for i in range(n_comp)], pipe
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    return pipe.fit_transform(x_train), pipe.transform(x_val), cols, pipe


def _atr_tp_sl_from_idx(atr: np.ndarray, cfg: PolicyConfig, tp_idx: np.ndarray, sl_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tp_mult = np.asarray(cfg.tp_atr_buckets, dtype=np.float64)[np.clip(tp_idx.astype(int), 0, len(cfg.tp_atr_buckets) - 1)]
    sl_mult = np.asarray(cfg.sl_atr_buckets, dtype=np.float64)[np.clip(sl_idx.astype(int), 0, len(cfg.sl_atr_buckets) - 1)]
    tp = np.clip(tp_mult * atr, cfg.tp_min, cfg.tp_max)
    sl = np.clip(sl_mult * atr, cfg.sl_min, cfg.sl_max)
    return tp, sl


def _nearest_index(values: tuple[float, ...] | tuple[int, ...], target: float) -> int:
    arr = np.asarray(values, dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(target))))


def _build_lifecycle_labels(frame: pd.DataFrame, cfg: PolicyConfig, *, stride_bars: int, batch_size: int) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    h = int(min(cfg.max_train_horizon_bars, max(1, cfg.fixed_max_hold_bars)))
    valid = np.arange(0, max(0, len(frame) - h - 1), max(1, int(stride_bars)), dtype=np.int64)
    if valid.size == 0:
        raise ValueError("no train candidates for lifecycle labels")
    notional_idx = _nearest_index(cfg.notional_buckets, cfg.fixed_notional)
    tp_idx = _nearest_index(cfg.tp_atr_buckets, cfg.fixed_tp_atr)
    sl_idx = _nearest_index(cfg.sl_atr_buckets, cfg.fixed_sl_atr)
    hold_idx = _nearest_index(cfg.max_hold_buckets, cfg.fixed_max_hold_bars)
    cooldown_idx = _nearest_index(cfg.cooldown_buckets, cfg.fixed_cooldown_bars)
    y = {
        "action": np.zeros(valid.size, dtype=np.int64),
        "notional": np.full(valid.size, notional_idx, dtype=np.int64),
        "take_profit": np.full(valid.size, tp_idx, dtype=np.int64),
        "stop_loss": np.full(valid.size, sl_idx, dtype=np.int64),
        "max_hold": np.full(valid.size, hold_idx, dtype=np.int64),
        "cooldown": np.full(valid.size, cooldown_idx, dtype=np.int64),
        "quality": np.zeros(valid.size, dtype=np.int64),
        "edge": np.zeros(valid.size, dtype=np.float64),
        "survival_lower": np.full(valid.size, float(h), dtype=np.float64),
        "survival_upper": np.full(valid.size, -1.0, dtype=np.float64),
    }
    horizons = np.arange(1, h + 1, dtype=np.int64)
    step_index = np.arange(h, dtype=np.int64)[None, :]
    row_idx_cache: dict[int, np.ndarray] = {}
    cost = 2.0 * float(cfg.fee + cfg.slip) * float(cfg.fixed_notional)
    for start in range(0, valid.size, int(batch_size)):
        end = min(start + int(batch_size), valid.size)
        idx = valid[start:end]
        n = len(idx)
        row_idx = row_idx_cache.setdefault(n, np.arange(n))
        entry = np.maximum(close[idx], 1e-12)
        fut_close = close[idx[:, None] + horizons[None, :]]
        fut_high = high[idx[:, None] + horizons[None, :]]
        fut_low = low[idx[:, None] + horizons[None, :]]
        close_ret = fut_close / entry[:, None] - 1.0
        atr_now = atr[idx]
        tp = np.clip(float(cfg.fixed_tp_atr) * atr_now, cfg.tp_min, cfg.tp_max)
        sl = np.clip(float(cfg.fixed_sl_atr) * atr_now, cfg.sl_min, cfg.sl_max)
        min_edge_now = np.maximum(
            float(cfg.min_net_edge),
            atr_now * float(cfg.dynamic_min_edge_atr_frac) * float(cfg.fixed_notional),
        )

        long_tp = fut_high >= entry[:, None] * (1.0 + tp[:, None])
        long_sl = fut_low <= entry[:, None] * (1.0 - sl[:, None])
        short_tp = fut_low <= entry[:, None] * (1.0 - tp[:, None])
        short_sl = fut_high >= entry[:, None] * (1.0 + sl[:, None])

        side_scores: list[np.ndarray] = []
        side_times: list[np.ndarray] = []
        side_observed: list[np.ndarray] = []
        max_eff_i = int(cfg.efficient_tp_bars) - 1
        for side, tp_hit, sl_hit in ((1.0, long_tp, long_sl), (-1.0, short_tp, short_sl)):
            any_hit = (tp_hit | sl_hit).any(axis=1)
            first_hit = np.where(any_hit, (tp_hit | sl_hit).argmax(axis=1), h - 1).astype(np.int64)
            path_mask = step_index[:, :h] <= first_hit[:, None]
            ambiguous = tp_hit[row_idx, first_hit] & sl_hit[row_idx, first_hit]
            raw_win = tp_hit[row_idx, first_hit] & ~ambiguous
            efficient = True if int(cfg.efficient_tp_bars) <= 0 else first_hit <= max_eff_i
            late_win = raw_win & ~efficient
            win = raw_win & efficient
            loss = sl_hit[row_idx, first_hit] | ambiguous | late_win
            if side > 0:
                adverse_path = np.maximum(0.0, 1.0 - fut_low / entry[:, None])
            else:
                adverse_path = np.maximum(0.0, fut_high / entry[:, None] - 1.0)
            mae = np.max(np.where(path_mask, adverse_path, 0.0), axis=1)
            time_decay = 1.0 / np.sqrt(first_hit.astype(np.float64) + 2.0)
            tp_edge = tp * time_decay - float(cfg.mae_penalty_lambda) * mae
            side_ret = np.where(win, tp_edge, np.where(loss, -sl, 0.0))
            side_scores.append(side_ret * float(cfg.fixed_notional) - cost)
            side_times.append(first_hit.astype(np.float64) + 1.0)
            side_observed.append(any_hit.astype(bool))

        long_score, short_score = side_scores
        best_long = ((long_score - short_score) > float(cfg.direction_margin)) & (long_score > min_edge_now)
        best_short = ((short_score - long_score) > float(cfg.direction_margin)) & (short_score > min_edge_now)
        y["action"][start:end] = np.where(best_long, 1, np.where(best_short, 2, 0)).astype(np.int64)
        y["quality"][start:end] = (best_long | best_short).astype(np.int64)
        y["edge"][start:end] = np.maximum.reduce([long_score, short_score, np.zeros(n, dtype=np.float64)])
        chosen_time = np.where(best_long, side_times[0], np.where(best_short, side_times[1], float(h)))
        chosen_observed = np.where(best_long, side_observed[0], np.where(best_short, side_observed[1], False))
        y["survival_lower"][start:end] = chosen_time
        y["survival_upper"][start:end] = np.where(chosen_observed, chosen_time, -1.0)
    meta = {
        "candidates": int(valid.size),
        "stride_bars": int(stride_bars),
        "max_train_horizon_bars": int(h),
        "labeling": "fixed_atr_triple_barrier",
        "fixed_notional": float(cfg.fixed_notional),
        "fixed_tp_atr": float(cfg.fixed_tp_atr),
        "fixed_sl_atr": float(cfg.fixed_sl_atr),
        "fixed_max_hold_bars": int(cfg.fixed_max_hold_bars),
        "fixed_cooldown_bars": int(cfg.fixed_cooldown_bars),
        "efficient_tp_bars": int(cfg.efficient_tp_bars),
        "min_net_edge": float(cfg.min_net_edge),
        "dynamic_min_edge_atr_frac": float(cfg.dynamic_min_edge_atr_frac),
        "direction_margin": float(cfg.direction_margin),
        "mae_penalty_lambda": float(cfg.mae_penalty_lambda),
    }
    return valid, y, meta


def _classifier_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "MultiClass",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _binary_classifier_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "Logloss",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _ranker_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "YetiRank",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _survival_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "SurvivalAft",
        "iterations": int(args.iterations),
        "learning_rate": float(args.learning_rate),
        "depth": int(args.depth),
        "l2_leaf_reg": float(args.l2_leaf_reg),
        "random_seed": int(seed),
        "allow_writing_files": False,
        "verbose": int(args.verbose),
        "thread_count": -1,
    }
    if args.task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    return params


def _fit_classifier(x: np.ndarray, y: np.ndarray, weight: np.ndarray, args: argparse.Namespace, seed: int) -> CatBoostClassifier | None:
    if np.unique(y).size < 2:
        return None
    model = CatBoostClassifier(**_classifier_params(args, seed))
    model.fit(Pool(x, y, weight=weight))
    return model


def _group_id(frame: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    return pd.factorize(ts.dt.strftime("%Y-%m-%d"), sort=False)[0].astype(np.int64)


def _fit_models(x: np.ndarray, y: dict[str, np.ndarray], args: argparse.Namespace, train_rows: pd.DataFrame) -> dict[str, Any]:
    trade = y["action"] != 0
    action_w = np.where(trade, 1.0, float(args.cash_action_weight))
    edge_w = np.clip(np.abs(y.get("edge", np.zeros_like(action_w, dtype=np.float64))) / 0.003, 0.05, 1.0)
    weight = np.maximum(action_w, edge_w)
    models: dict[str, Any] = {
        "action_model": _fit_classifier(x, y["action"], weight, args, args.seed),
        "quality_head": str(args.quality_head),
        "default_bucket_indexes": {},
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict()
            for key, vals in y.items()
            if key not in {"edge", "survival_lower", "survival_upper"}
        },
    }
    if str(args.quality_head) == "ranker":
        quality_model = CatBoostRanker(**_ranker_params(args, args.seed + 99))
        rank_target = np.asarray(y["edge"], dtype=np.float64)
        quality_model.fit(Pool(x, rank_target, group_id=_group_id(train_rows)))
    else:
        quality_model = CatBoostClassifier(**_binary_classifier_params(args, args.seed + 99))
        quality_model.fit(Pool(x, y["quality"], weight=weight))
    models["quality_model"] = quality_model
    for key in ("notional", "take_profit", "stop_loss", "max_hold", "cooldown"):
        vals = y[key][trade]
        models["default_bucket_indexes"][key] = int(pd.Series(vals).mode().iloc[0]) if len(vals) else 0
    if str(args.exit_head) == "survival" and np.sum(trade) >= 32:
        target = np.column_stack([y["survival_lower"][trade], y["survival_upper"][trade]])
        survival_model = CatBoostRegressor(**_survival_params(args, args.seed + 199))
        survival_model.fit(Pool(x[trade], target))
        models["survival_model"] = survival_model
        models["exit_head"] = "survival"
    else:
        models["exit_head"] = "fixed"
    models["label_distribution"]["edge_mean"] = float(np.mean(y["edge"]))
    models["label_distribution"]["edge_p95"] = float(np.quantile(y["edge"], 0.95))
    models["label_distribution"]["survival_observed_rate"] = float(np.mean(y["survival_upper"][trade] > 0.0)) if np.any(trade) else 0.0
    return models


def _predict_bucket(models: dict[str, Any], key: str, x: np.ndarray, bucket_count: int) -> tuple[np.ndarray, np.ndarray]:
    model = models.get(f"{key}_model")
    if model is None:
        default = int(models["default_bucket_indexes"].get(key, 0))
        return np.full(len(x), default, dtype=np.int64), np.ones(len(x), dtype=np.float64)
    proba = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    idx = classes[np.argmax(proba, axis=1)]
    return np.clip(idx.astype(np.int64), 0, bucket_count - 1), np.max(proba, axis=1)


def _predict_policy(models: dict[str, Any], x: np.ndarray, frame: pd.DataFrame, cfg: PolicyConfig) -> pd.DataFrame:
    action_proba = models["action_model"].predict_proba(x)
    action_classes = np.asarray(models["action_model"].classes_, dtype=int)
    action = action_classes[np.argmax(action_proba, axis=1)].astype(np.int64)
    action_conf = np.max(action_proba, axis=1)
    if str(models.get("quality_head", "classifier")) == "ranker":
        quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    else:
        quality_proba = models["quality_model"].predict_proba(x)
        quality_classes = np.asarray(models["quality_model"].classes_, dtype=int)
        if 1 in set(quality_classes.tolist()):
            quality = quality_proba[:, int(np.flatnonzero(quality_classes == 1)[0])]
        else:
            quality = np.zeros(len(x), dtype=np.float64)
    notional_i, c1 = _predict_bucket(models, "notional", x, len(cfg.notional_buckets))
    tp_i, c2 = _predict_bucket(models, "take_profit", x, len(cfg.tp_atr_buckets))
    sl_i, c3 = _predict_bucket(models, "stop_loss", x, len(cfg.sl_atr_buckets))
    hold_i, c4 = _predict_bucket(models, "max_hold", x, len(cfg.max_hold_buckets))
    cool_i, c5 = _predict_bucket(models, "cooldown", x, len(cfg.cooldown_buckets))
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    tp, sl = _atr_tp_sl_from_idx(atr, cfg, tp_i, sl_i)
    notional = np.asarray(cfg.notional_buckets, dtype=np.float64)[notional_i]
    max_hold = np.asarray(cfg.max_hold_buckets, dtype=np.int64)[hold_i]
    if str(models.get("exit_head", "fixed")) == "survival" and models.get("survival_model") is not None:
        raw_survival = np.asarray(models["survival_model"].predict(x), dtype=np.float64)
        survival_bars = np.rint(np.exp(np.clip(raw_survival, np.log(1.0), np.log(float(cfg.fixed_max_hold_bars))))).astype(np.int64)
        max_hold = np.clip(survival_bars, 1, int(cfg.fixed_max_hold_bars))
    cooldown = np.asarray(cfg.cooldown_buckets, dtype=np.int64)[cool_i]
    conf = np.mean(np.vstack([action_conf, quality, c1, c2, c3, c4, c5]), axis=0)
    return pd.DataFrame(
        {
            "action": action,
            "quality_score": quality,
            "confidence": conf,
            "notional": notional,
            "take_profit": tp,
            "stop_loss": sl,
            "max_hold_bars": max_hold,
            "cooldown_bars": cooldown,
        },
        index=frame.index,
    )


def _fill_price(frame: pd.DataFrame, i: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(frame["open"], errors="coerce").ffill().iloc[int(np.clip(i, 0, len(frame) - 1))])
    if entry:
        return px * (1.0 + slip if side > 0 else 1.0 - slip)
    return px * (1.0 - slip if side > 0 else 1.0 + slip)


def _days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    span = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0 if len(ts) > 1 else 1.0
    return float(max(span, 1.0))


def _backtest(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    threshold: float,
    fee: float,
    slip: float,
    entry_pullback_atr: float,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    exposure = 0.0
    hold = 0
    max_hold = 0
    cooldown = 0
    tp = sl = 0.0
    trades = wins = long_entries = short_entries = missed_entries = 0
    exits: dict[str, int] = {}
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exposure_sum = 0.0

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, row: pd.Series) -> None:
        nonlocal side, entry, entry_equity, exposure, hold, max_hold, cooldown, tp, sl, cash, exposure_sum, long_entries, short_entries, missed_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        exposure = float(np.clip(row.notional, 0.01, 2.0))
        if float(entry_pullback_atr) > 0.0:
            pullback = float(entry_pullback_atr) * max(float(atr[fill_i]), 0.0)
            if side > 0:
                limit_px = float(open_px[fill_i]) * (1.0 - pullback)
                if float(low[fill_i]) > limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 + slip)
            else:
                limit_px = float(open_px[fill_i]) * (1.0 + pullback)
                if float(high[fill_i]) < limit_px:
                    side = 0
                    exposure = 0.0
                    missed_entries += 1
                    return
                entry = limit_px * (1.0 - slip)
        else:
            entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        max_hold = int(max(1, row.max_hold_bars))
        cooldown = 0
        tp = float(max(row.take_profit, 1e-4))
        sl = float(max(row.stop_loss, 1e-4))
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None, next_cooldown: int = 0) -> None:
        nonlocal side, entry, cash, hold, tp, sl, exposure, trades, wins, cooldown
        if fill_px is None:
            fill_px = _fill_price(frame, min(i + 1, len(frame) - 1), side, slip, entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * fee * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        tp = sl = exposure = 0.0
        cooldown = int(max(0, next_cooldown))

    for i in range(len(frame) - 2):
        row = dec.iloc[i]
        desired = int(row.action) if float(row.quality_score) >= float(threshold) else 0
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold += 1
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + tp)
                sl_hit = low[i] <= entry * (1.0 - sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - sl) * (1.0 - slip), int(row.cooldown_bars))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + tp) * (1.0 - slip), int(row.cooldown_bars))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - sl) * (1.0 - slip), int(row.cooldown_bars))
            else:
                tp_hit = low[i] <= entry * (1.0 - tp)
                sl_hit = high[i] >= entry * (1.0 + sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + sl) * (1.0 + slip), int(row.cooldown_bars))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - tp) * (1.0 + slip), int(row.cooldown_bars))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + sl) * (1.0 + slip), int(row.cooldown_bars))
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side != 0 and hold >= max_hold:
            exit_pos(i, "max_hold", next_cooldown=int(row.cooldown_bars))
        elif side == 0:
            cooldown = max(0, cooldown - 1)
            if cooldown == 0 and desired != 0:
                enter(i, 1 if desired == 1 else -1, row)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(exposure_sum / max(trades, 1)),
        "missed_entries": int(missed_entries),
        "action_counts": action_counts,
        "exits": exits,
    }


def _threshold_grid(dec: pd.DataFrame, n: int) -> np.ndarray:
    active = dec.loc[dec["action"] != 0, "quality_score"].to_numpy(dtype=np.float64)
    active = active[np.isfinite(active)]
    if active.size == 0:
        return np.array([np.inf])
    return np.unique(np.quantile(active, np.linspace(0.10, 0.995, int(n))))


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 15:
        return -1e6 + float(c1["pnl"])
    tpd = float(c1["trades_per_day"])
    density_pen = max(0.0, 5.0 - tpd) * 4.0 + max(0.0, tpd - 10.0) * 3.0
    return (
        float(c1["pnl"])
        + 0.35 * float(c2["pnl"])
        + 0.10 * float(c3["pnl"])
        - 0.25 * abs(float(c1["mdd"]))
        - density_pen
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 fixed-barrier CatBoost action/meta policy.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="stable48_global_pca32")
    ap.add_argument("--iterations", type=int, default=700)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--l2-leaf-reg", type=float, default=6.0)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--thresholds", type=int, default=70)
    ap.add_argument("--fixed-notional", type=float, default=PolicyConfig.fixed_notional)
    ap.add_argument("--fixed-tp-atr", type=float, default=PolicyConfig.fixed_tp_atr)
    ap.add_argument("--fixed-sl-atr", type=float, default=PolicyConfig.fixed_sl_atr)
    ap.add_argument("--fixed-max-hold-bars", type=int, default=PolicyConfig.fixed_max_hold_bars)
    ap.add_argument("--fixed-cooldown-bars", type=int, default=PolicyConfig.fixed_cooldown_bars)
    ap.add_argument("--efficient-tp-bars", type=int, default=PolicyConfig.efficient_tp_bars)
    ap.add_argument("--min-net-edge", type=float, default=PolicyConfig.min_net_edge)
    ap.add_argument("--dynamic-min-edge-atr-frac", type=float, default=PolicyConfig.dynamic_min_edge_atr_frac)
    ap.add_argument("--direction-margin", type=float, default=PolicyConfig.direction_margin)
    ap.add_argument("--mae-penalty-lambda", type=float, default=PolicyConfig.mae_penalty_lambda)
    ap.add_argument("--entry-pullback-atr", type=float, default=0.0)
    ap.add_argument("--quality-head", choices=["classifier", "ranker"], default="classifier")
    ap.add_argument("--exit-head", choices=["fixed", "survival"], default="fixed")
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(
        PolicyConfig(),
        fixed_notional=float(args.fixed_notional),
        fixed_tp_atr=float(args.fixed_tp_atr),
        fixed_sl_atr=float(args.fixed_sl_atr),
        fixed_max_hold_bars=int(args.fixed_max_hold_bars),
        fixed_cooldown_bars=int(args.fixed_cooldown_bars),
        efficient_tp_bars=int(args.efficient_tp_bars),
        min_net_edge=float(args.min_net_edge),
        dynamic_min_edge_atr_frac=float(args.dynamic_min_edge_atr_frac),
        direction_margin=float(args.direction_margin),
        mae_penalty_lambda=float(args.mae_penalty_lambda),
    )
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0
    extra_cols = [
        "market_state_2024_unsup_v5_risk_off_prob",
        "market_state_2024_unsup_v5_trend_prob",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "regime4_pred_instability_prob",
        "regime4_pred_whipsaw_prob",
    ]
    feat, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), extra_cols)
    frame = feat.merge(_label_frame(args.label_dir), on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame[frame["dataset_split"].astype(str).str.lower().ne("train")].copy()
    if args.smoke:
        train = train.iloc[: min(len(train), 5000)].copy()
        val = val.iloc[: min(len(val), 3000)].copy()
        args.iterations = min(args.iterations, 20)
        args.thresholds = min(args.thresholds, 8)
        args.stride_bars = max(args.stride_bars, 6)
    x_train_all, x_val, model_features, pipe = _feature_matrix(
        train,
        val,
        present,
        use_pca=use_pca,
        pca_components=int(spec.get("extra_pca_components") or 0),
    )
    valid, y, label_meta = _build_lifecycle_labels(train, cfg, stride_bars=args.stride_bars, batch_size=args.batch_size)
    x_train = x_train_all[valid]
    print(
        f"[alpha6-v2] variant={args.variant} train_rows={len(train)} val_rows={len(val)} label_candidates={len(valid)} raw_features={len(present)} model_features={len(model_features)} use_pca={use_pca}",
        flush=True,
    )
    models = _fit_models(x_train, y, args, train.iloc[valid].reset_index(drop=True))
    dec = _predict_policy(models, x_val, val, cfg)
    rows = []
    best: dict[str, Any] | None = None
    for th in _threshold_grid(dec, args.thresholds):
        bt = {
            f"cost{m}": _backtest(
                val,
                dec,
                threshold=float(th),
                fee=cfg.fee * m,
                slip=cfg.slip * m,
                entry_pullback_atr=float(args.entry_pullback_atr),
            )
            for m in (1, 2, 3)
        }
        score = _score(bt["cost1"], bt["cost2"], bt["cost3"])
        row = {
            "threshold": float(th),
            "score": float(score),
            "pnl": float(bt["cost1"]["pnl"]),
            "mdd": float(bt["cost1"]["mdd"]),
            "trades": int(bt["cost1"]["trades"]),
            "trades_per_day": float(bt["cost1"]["trades_per_day"]),
            "wr": float(bt["cost1"]["wr"]),
            "long_entries": int(bt["cost1"]["long_entries"]),
            "short_entries": int(bt["cost1"]["short_entries"]),
            "avg_notional": float(bt["cost1"]["avg_notional"]),
            "missed_entries": int(bt["cost1"]["missed_entries"]),
            "exits": json.dumps(bt["cost1"]["exits"], sort_keys=True),
        }
        rows.append(row)
        if best is None or row["score"] > best["summary"]["score"]:
            best = {"summary": row, "backtest": bt}
    assert best is not None

    prefix = args.out_dir / args.variant
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(f"{prefix}_threshold_grid.csv", index=False)
    pred = val[["timestamp", "open", "high", "low", "close", "label_action"]].copy()
    for col in dec.columns:
        pred[col] = dec[col].to_numpy()
    pred.to_csv(f"{prefix}_val_predictions.csv", index=False)
    artifact = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "config": asdict(cfg),
        "feature_cols": present,
        "model_features": model_features,
        "missing_features": missing,
        "use_pca": use_pca,
        "pipeline": pipe,
        "models": models,
    }
    joblib.dump(artifact, f"{prefix}_bundle.joblib")
    summary = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "label_meta": label_meta,
        "label_distribution": models["label_distribution"],
        "raw_feature_count": int(len(present)),
        "missing_features": missing,
        "model_feature_count": int(len(model_features)),
        "use_pca": bool(use_pca),
        "best": best["summary"],
        "best_backtest": best["backtest"],
        "params": vars(args),
    }
    Path(f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
