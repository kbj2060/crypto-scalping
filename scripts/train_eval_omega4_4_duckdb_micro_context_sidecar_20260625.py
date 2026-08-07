#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import duckdb
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_ID = "omega4_4_duckdb_micro_context_sidecar_20260625"
BASE_MODEL_ID = "omega4_4_topdown_reproducible_architecture_baseline_20260623"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_SUPERVISED_DIR = ROOT / "data/ensemble/supervised" / MODEL_ID
KST = "Asia/Seoul"

MICRO_BASE_COLS = [
    "obi",
    "taker_buy_ratio",
    "spoofing_score",
    "nif_whale",
    "nif_retail",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "kelly_mult",
    "signal_bias",
    "shadow_toxicity_score",
    "shadow_queue_collapse",
    "shadow_absorption_score",
    "shadow_queue_bias",
    "shadow_regime_conf",
    "recent_trade_count_5m",
    "recent_trade_notional_5m",
    "recent_whale_count_5m",
]
TAIL_BASE_COLS = {
    "long_usd_1m",
    "short_usd_1m",
    "long_mu_1m",
    "short_mu_1m",
    "long_sigma_1m",
    "short_sigma_1m",
    "shadow_aftershock_prob",
    "liq_event_count_1m",
    "ws_connected",
    "ws_stale",
    "ws_age_sec",
    "valid_liq_stream",
    "schema_version_tail",
}


@dataclass(frozen=True)
class ThresholdConfig:
    horizon_bars: int
    proba_threshold: float
    proba_gap: float
    min_trade_count_5m: float
    toxicity_cap: float


def _mdd_pct(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    eq = np.cumprod(1.0 + returns.astype(np.float64))
    running = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(running, 1e-12)) - 1.0
    return float(np.min(dd) * 100.0)


def _load_table(path: Path, table: str) -> pd.DataFrame:
    con = duckdb.connect(str(path), read_only=True)
    try:
        return con.execute(f"select * from {table} order by ts").fetchdf()
    finally:
        con.close()


def _load_polymarket_price(path: Path) -> tuple[pd.DataFrame, float]:
    con = duckdb.connect(str(path), read_only=True)
    try:
        empty_ratio = con.execute(
            """
            select avg(case when coalesce(markets_json, '') = '[]' then 1.0 else 0.0 end)
            from polymarket_markets_10s_json
            """
        ).fetchone()[0]
        price = con.execute(
            """
            select ts, current_price
            from polymarket_markets_10s_json
            where current_price is not null
            order by ts
            """
        ).fetchdf()
    finally:
        con.close()
    return price, float(empty_ratio or 0.0)


def _load_duckdb_frame(live_dir: Path, feature_mode: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    micro = _load_table(live_dir / "microstructure.duckdb", "microstructure_1m")
    poly, empty_poly_ratio = _load_polymarket_price(live_dir / "polymarket.duckdb")
    tail: pd.DataFrame | None = None
    if feature_mode != "microstructure_only":
        tail = _load_table(live_dir / "tail_risk.duckdb", "tail_risk_1m")

    frames = [micro, poly] if tail is None else [micro, tail, poly]
    for frame in frames:
        frame["ts"] = pd.to_datetime(frame["ts"], errors="coerce")
        frame.dropna(subset=["ts"], inplace=True)

    micro = micro.sort_values("ts").groupby("ts").last().reset_index()
    if tail is not None:
        tail = tail.sort_values("ts").groupby("ts").last().reset_index()
    poly["ts"] = poly["ts"].dt.floor("min")
    poly_1m = (
        poly.sort_values("ts")
        .groupby("ts")
        .agg(
            current_price=("current_price", "last"),
        )
        .reset_index()
    )

    frame = poly_1m.merge(micro, on="ts", how="inner")
    if tail is not None:
        frame = frame.merge(tail, on="ts", how="left", suffixes=("", "_tail"))
    frame = frame.sort_values("ts").reset_index(drop=True)
    frame["price"] = pd.to_numeric(frame["current_price"], errors="coerce")
    frame = frame.dropna(subset=["price"]).reset_index(drop=True)

    meta = {
        "micro_rows_raw": int(len(micro)),
        "tail_rows_raw": None if tail is None else int(len(tail)),
        "polymarket_rows_1m": int(len(poly_1m)),
        "joined_rows": int(len(frame)),
        "start_ts": str(frame["ts"].min()) if len(frame) else "",
        "end_ts": str(frame["ts"].max()) if len(frame) else "",
        "polymarket_empty_markets_ratio": empty_poly_ratio,
        "feature_mode": feature_mode,
    }
    return frame, meta


def _add_numeric_features(frame: pd.DataFrame, feature_mode: str, feature_windows: list[int]) -> pd.DataFrame:
    out = frame.copy()
    windows = sorted(set(int(w) for w in feature_windows))
    if not windows:
        raise ValueError("feature_windows must not be empty")

    bool_cols = [c for c in out.columns if out[c].dtype == bool]
    for col in bool_cols:
        out[col] = out[col].astype(float)

    numeric_cols = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c])]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    additions: dict[str, Any] = {}
    if feature_mode != "microstructure_only":
        additions["ret_1m"] = out["price"].pct_change(1)
        for window in windows:
            additions[f"ret_{window}m"] = out["price"].pct_change(window)
        for window in [w for w in windows if w >= 5]:
            additions[f"vol_{window}m"] = additions["ret_1m"].rolling(window, min_periods=max(2, window // 3)).std()

    def s(col: str, default: float = 0.0) -> pd.Series:
        if col not in out.columns:
            return pd.Series(default, index=out.index, dtype=float)
        return pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)

    taker_imbalance = (s("taker_buy_ratio", 0.5) - 0.5) * 2.0
    trade_count = s("recent_trade_count_5m")
    trade_notional = s("recent_trade_notional_5m")
    whale_count = s("recent_whale_count_5m")
    flow_pressure = s("obi") * taker_imbalance
    queue_edge = s("shadow_absorption_score") - s("shadow_queue_collapse")
    micro_cross_features = {
        "taker_buy_imbalance": taker_imbalance,
        "nif_whale_retail_spread": s("nif_whale") - s("nif_retail"),
        "flow_pressure": flow_pressure,
        "whale_flow_pressure": s("nif_whale") * taker_imbalance,
        "retail_flow_pressure": s("nif_retail") * taker_imbalance,
        "queue_edge": queue_edge,
        "queue_bias_edge": s("shadow_queue_bias") * queue_edge,
        "toxicity_abs_obi": s("shadow_toxicity_score") * s("obi").abs(),
        "toxicity_flow_pressure": s("shadow_toxicity_score") * flow_pressure,
        "trade_intensity_log": np.log1p(trade_count.clip(lower=0.0)),
        "notional_per_trade": trade_notional / trade_count.replace(0.0, np.nan),
        "whale_trade_share": whale_count / trade_count.replace(0.0, np.nan),
    }
    additions.update(micro_cross_features)

    base_cols = list(MICRO_BASE_COLS)
    if feature_mode != "microstructure_only":
        base_cols.extend(
            [
                "long_usd_1m",
                "short_usd_1m",
                "shadow_aftershock_prob",
                "liq_event_count_1m",
            ]
        )
    for col in [c for c in base_cols if c in out.columns]:
        s = pd.to_numeric(out[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        additions[f"{col}_d1"] = s.diff()
        for window in windows:
            mean = s.rolling(window, min_periods=1).mean()
            additions[f"{col}_r{window}"] = mean
            if window >= 10:
                std = s.rolling(window, min_periods=max(2, window // 3)).std()
                additions[f"{col}_std{window}"] = std
                additions[f"{col}_z{window}"] = (s - mean) / std.replace(0.0, np.nan)

    for col, series in micro_cross_features.items():
        series = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
        additions[f"{col}_d1"] = series.diff()
        for window in [w for w in windows if w >= 3]:
            additions[f"{col}_r{window}"] = series.rolling(window, min_periods=1).mean()

    out = pd.concat([out, pd.DataFrame(additions, index=out.index)], axis=1)
    return out


def _tail_root(col: str) -> str:
    suffixes = ["_d1"]
    for window in (3, 5, 10, 15, 30, 45, 60, 90, 120):
        suffixes.extend([f"_r{window}", f"_std{window}", f"_z{window}"])
    for suffix in suffixes:
        if col.endswith(suffix):
            return col[: -len(suffix)]
    return col


def _feature_columns(frame: pd.DataFrame, feature_mode: str) -> list[str]:
    excluded = {
        "ts",
        "markets_json",
        "snapshot_json",
        "current_price",
        "price",
        "fwd_ret",
        "long_gross_ret",
        "short_gross_ret",
        "barrier_label_margin",
        "barrier_exit_bar",
        "target",
        "valid_horizon",
    }
    columns = [
        c
        for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]
    if feature_mode != "microstructure_only":
        return columns

    micro_columns: list[str] = []
    for col in columns:
        root = _tail_root(col)
        if col.startswith(("ret_", "vol_")):
            continue
        if root in TAIL_BASE_COLS or root.endswith("_tail"):
            continue
        micro_columns.append(col)
    return micro_columns


def _first_index(mask: np.ndarray) -> int | None:
    hits = np.flatnonzero(mask)
    if len(hits) == 0:
        return None
    return int(hits[0])


def _barrier_return(
    tp_idx: int | None,
    sl_idx: int | None,
    *,
    tp_return: float,
    sl_return: float,
    final_return: float,
) -> tuple[float, int]:
    if tp_idx is None and sl_idx is None:
        return float(final_return), -1
    if sl_idx is None or (tp_idx is not None and tp_idx < sl_idx):
        return float(tp_return), int(tp_idx or 0) + 1
    return -float(sl_return), int(sl_idx) + 1


def _label_frame(
    frame: pd.DataFrame,
    horizon: int,
    label_threshold: float,
    *,
    label_mode: str,
    barrier_tp: float,
    barrier_sl: float,
    cost_per_notional: float,
) -> pd.DataFrame:
    out = frame.copy()
    future_ts = out["ts"].shift(-horizon)
    future_price = out["price"].shift(-horizon)
    out["fwd_ret"] = (future_price / out["price"]) - 1.0
    out["valid_horizon"] = (
        out["fwd_ret"].notna()
        & ((future_ts - out["ts"]) <= pd.Timedelta(minutes=int(horizon) + 2))
    )
    out["target"] = np.where(
        out["fwd_ret"] > label_threshold,
        1,
        np.where(out["fwd_ret"] < -label_threshold, -1, 0),
    )
    if label_mode == "forward_return":
        return out.loc[out["valid_horizon"]].reset_index(drop=True)
    if label_mode != "barrier":
        raise ValueError(f"unsupported label_mode: {label_mode}")

    prices = out["price"].to_numpy(np.float64)
    timestamps = pd.to_datetime(out["ts"])
    long_gross = np.full(len(out), np.nan, dtype=np.float64)
    short_gross = np.full(len(out), np.nan, dtype=np.float64)
    exit_bar = np.full(len(out), -1, dtype=np.int32)
    target = np.zeros(len(out), dtype=np.int32)
    valid = np.zeros(len(out), dtype=bool)

    for i in range(0, max(0, len(out) - int(horizon))):
        start_price = prices[i]
        if not np.isfinite(start_price) or start_price <= 0.0:
            continue
        final_idx = i + int(horizon)
        if timestamps.iloc[final_idx] - timestamps.iloc[i] > pd.Timedelta(minutes=int(horizon) + 2):
            continue
        path = (prices[i + 1 : final_idx + 1] / start_price) - 1.0
        if len(path) != int(horizon) or not np.isfinite(path).all():
            continue

        long_tp_idx = _first_index(path >= float(barrier_tp))
        long_sl_idx = _first_index(path <= -float(barrier_sl))
        short_tp_idx = _first_index(path <= -float(barrier_tp))
        short_sl_idx = _first_index(path >= float(barrier_sl))

        long_ret, long_exit = _barrier_return(
            long_tp_idx,
            long_sl_idx,
            tp_return=float(barrier_tp),
            sl_return=float(barrier_sl),
            final_return=float(path[-1]),
        )
        short_ret, short_exit = _barrier_return(
            short_tp_idx,
            short_sl_idx,
            tp_return=float(barrier_tp),
            sl_return=float(barrier_sl),
            final_return=float(-path[-1]),
        )
        long_gross[i] = long_ret
        short_gross[i] = short_ret
        exit_bar[i] = max(long_exit, short_exit)
        long_net = long_ret - float(cost_per_notional)
        short_net = short_ret - float(cost_per_notional)
        if long_net > 0.0 or short_net > 0.0:
            target[i] = 1 if long_net >= short_net else -1
        valid[i] = True

    out["long_gross_ret"] = long_gross
    out["short_gross_ret"] = short_gross
    out["barrier_exit_bar"] = exit_bar
    out["barrier_label_margin"] = np.maximum(long_gross, short_gross) - float(cost_per_notional)
    out["target"] = target
    out["valid_horizon"] = valid
    return out.loc[out["valid_horizon"]].reset_index(drop=True)


def _split_indices(n: int) -> tuple[slice, slice, slice]:
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n)


def _trade_eval(
    probs: np.ndarray,
    classes: np.ndarray,
    fwd_ret: np.ndarray,
    frame: pd.DataFrame,
    cfg: ThresholdConfig,
    *,
    cost_per_notional: float,
) -> dict[str, Any]:
    class_map = {int(cls): i for i, cls in enumerate(classes)}
    p_long = probs[:, class_map[1]] if 1 in class_map else np.zeros(len(probs), dtype=float)
    p_short = probs[:, class_map[-1]] if -1 in class_map else np.zeros(len(probs), dtype=float)
    p_flat = probs[:, class_map[0]] if 0 in class_map else np.zeros(len(probs), dtype=float)

    top = np.maximum(p_long, p_short)
    gap = np.abs(p_long - p_short)
    side = np.where(p_long >= p_short, 1.0, -1.0)

    trade_count = pd.to_numeric(frame.get("recent_trade_count_5m", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    toxicity = pd.to_numeric(frame.get("shadow_toxicity_score", 0.0), errors="coerce").fillna(0.0).to_numpy(float)
    warmup = frame.get("warmup_30m_ready", True)
    if isinstance(warmup, pd.Series):
        warmup_arr = warmup.fillna(False).astype(bool).to_numpy()
    else:
        warmup_arr = np.ones(len(frame), dtype=bool)

    active = (
        warmup_arr
        & (top >= float(cfg.proba_threshold))
        & (gap >= float(cfg.proba_gap))
        & (top > p_flat)
        & (trade_count >= float(cfg.min_trade_count_5m))
        & (toxicity <= float(cfg.toxicity_cap))
    )
    if "long_gross_ret" in frame.columns and "short_gross_ret" in frame.columns:
        long_gross = pd.to_numeric(frame["long_gross_ret"], errors="coerce").fillna(0.0).to_numpy(float)
        short_gross = pd.to_numeric(frame["short_gross_ret"], errors="coerce").fillna(0.0).to_numpy(float)
        gross_returns = np.where(side > 0.0, long_gross, short_gross)
        returns = (gross_returns - float(cost_per_notional))[active]
    else:
        returns = ((side * fwd_ret) - float(cost_per_notional))[active]
    if len(returns) == 0:
        return {
            "pnl_pct": 0.0,
            "mdd_pct": 0.0,
            "wr": 0.0,
            "trades": 0,
            "avg_trade_pct": 0.0,
            "long_trades": 0,
            "short_trades": 0,
        }
    return {
        "pnl_pct": float(returns.sum() * 100.0),
        "mdd_pct": _mdd_pct(returns),
        "wr": float((returns > 0.0).mean()),
        "trades": int(len(returns)),
        "avg_trade_pct": float(returns.mean() * 100.0),
        "long_trades": int(((side > 0) & active).sum()),
        "short_trades": int(((side < 0) & active).sum()),
    }


def _best_threshold(
    probs: np.ndarray,
    classes: np.ndarray,
    fwd_ret: np.ndarray,
    frame: pd.DataFrame,
    horizon: int,
    *,
    cost_per_notional: float,
    min_validation_trades: int,
    max_validation_mdd_pct: float,
    selection_objective: str,
    proba_thresholds: list[float],
    proba_gaps: list[float],
    min_trade_counts_5m: list[float],
    toxicity_caps: list[float],
) -> tuple[ThresholdConfig, dict[str, Any]]:
    best_cfg: ThresholdConfig | None = None
    best_metrics: dict[str, Any] | None = None
    best_key: tuple[float, ...] | None = None
    for proba_threshold in proba_thresholds:
        for proba_gap in proba_gaps:
            for min_trade_count_5m in min_trade_counts_5m:
                for toxicity_cap in toxicity_caps:
                    cfg = ThresholdConfig(
                        horizon_bars=int(horizon),
                        proba_threshold=float(proba_threshold),
                        proba_gap=float(proba_gap),
                        min_trade_count_5m=float(min_trade_count_5m),
                        toxicity_cap=float(toxicity_cap),
                    )
                    metrics = _trade_eval(
                        probs,
                        classes,
                        fwd_ret,
                        frame,
                        cfg,
                        cost_per_notional=cost_per_notional,
                    )
                    if metrics["trades"] < int(min_validation_trades):
                        continue
                    mdd_abs = abs(float(metrics["mdd_pct"]))
                    if mdd_abs > float(max_validation_mdd_pct):
                        continue
                    if selection_objective == "risk_adjusted":
                        key = (
                            float(metrics["pnl_pct"]) / max(1.0, mdd_abs),
                            float(metrics["avg_trade_pct"]),
                            float(metrics["wr"]),
                            float(metrics["pnl_pct"]),
                            -float(metrics["trades"]),
                        )
                    else:
                        key = (metrics["pnl_pct"], metrics["wr"], -mdd_abs, metrics["trades"])
                    if best_key is None or key > best_key:
                        best_cfg = cfg
                        best_metrics = metrics
                        best_key = key
    if best_cfg is None or best_metrics is None:
        cfg = ThresholdConfig(
            horizon_bars=int(horizon),
            proba_threshold=0.72,
            proba_gap=0.16,
            min_trade_count_5m=2500.0,
            toxicity_cap=0.75,
        )
        return cfg, _trade_eval(
            probs,
            classes,
            fwd_ret,
            frame,
            cfg,
            cost_per_notional=cost_per_notional,
        )
    return best_cfg, best_metrics


def _live_trade_count(live_dir: Path, start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> int:
    try:
        from scripts.backtest_polymarket_news_overlay import _load_trades

        return int(len(_load_trades(str(live_dir / "dashboard_events.jsonl"), start_utc=start_utc, end_utc=end_utc)))
    except Exception:
        return 0


def _float_grid(raw: str) -> list[float]:
    values = [float(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"empty float grid: {raw!r}")
    return values


def _int_grid(raw: str) -> list[int]:
    values = [int(x) for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError(f"empty int grid: {raw!r}")
    return values


def _candidate_key(candidate: dict[str, Any], objective: str) -> tuple[float, ...]:
    val = candidate["metrics"]["validation"]
    mdd_abs = abs(float(val["mdd_pct"]))
    if objective == "risk_adjusted":
        return (
            float(val["pnl_pct"]) / max(1.0, mdd_abs),
            float(val["avg_trade_pct"]),
            float(val["wr"]),
            float(val["pnl_pct"]),
            -float(val["trades"]),
        )
    return (float(val["pnl_pct"]), float(val["wr"]), -mdd_abs, float(val["trades"]))


def _balanced_sample_weight(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    weights = {int(cls): float(len(y)) / (float(len(classes)) * float(count)) for cls, count in zip(classes, counts)}
    return np.asarray([weights[int(value)] for value in y], dtype=np.float64)


def _run(args: argparse.Namespace) -> dict[str, Any]:
    live_dir = Path(args.live_dir)
    model_id = str(args.model_id)
    raw, data_meta = _load_duckdb_frame(live_dir, str(args.feature_mode))
    feature_windows = _int_grid(str(args.feature_windows))
    feat_frame = _add_numeric_features(raw, str(args.feature_mode), feature_windows)

    candidates: list[dict[str, Any]] = []
    best_bundle: dict[str, Any] | None = None
    for horizon in [int(x) for x in args.horizons.split(",") if str(x).strip()]:
        label_threshold = float(args.label_threshold_5m if horizon <= 5 else args.label_threshold)
        labeled = _label_frame(
            feat_frame,
            horizon,
            label_threshold,
            label_mode=str(args.label_mode),
            barrier_tp=float(args.barrier_tp),
            barrier_sl=float(args.barrier_sl),
            cost_per_notional=float(args.cost_per_notional),
        )
        if len(labeled) < int(args.min_rows):
            candidates.append(
                {
                    "horizon_bars": horizon,
                    "skipped": True,
                    "reason": "not_enough_rows",
                    "rows": int(len(labeled)),
                }
            )
            continue

        train_slice, val_slice, oos_slice = _split_indices(len(labeled))
        feature_cols = _feature_columns(labeled, str(args.feature_mode))
        x = labeled[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(np.float32)
        y = labeled["target"].astype(int).to_numpy()

        model = HistGradientBoostingClassifier(
            max_iter=int(args.max_iter),
            learning_rate=float(args.learning_rate),
            max_leaf_nodes=int(args.max_leaf_nodes),
            l2_regularization=float(args.l2_regularization),
            random_state=int(args.seed),
        )
        train_weights = _balanced_sample_weight(y[train_slice]) if bool(args.balance_classes) else None
        model.fit(x[train_slice], y[train_slice], sample_weight=train_weights)
        classes = np.asarray(model.classes_, dtype=int)

        split_rows: dict[str, dict[str, Any]] = {}
        for name, split in (("train", train_slice), ("validation", val_slice), ("oos", oos_slice)):
            pred = model.predict(x[split])
            split_rows[name] = {
                "rows": int(len(labeled.iloc[split])),
                "start_ts": str(labeled["ts"].iloc[split].min()),
                "end_ts": str(labeled["ts"].iloc[split].max()),
                "balanced_accuracy": float(balanced_accuracy_score(y[split], pred)),
                "target_counts": {
                    str(int(k)): int(v) for k, v in zip(*np.unique(y[split], return_counts=True))
                },
            }

        val_probs = model.predict_proba(x[val_slice])
        val_frame = labeled.iloc[val_slice].reset_index(drop=True)
        val_cfg, val_metrics = _best_threshold(
            val_probs,
            classes,
            val_frame["fwd_ret"].to_numpy(float),
            val_frame,
            horizon,
            cost_per_notional=float(args.cost_per_notional),
            min_validation_trades=int(args.min_validation_trades),
            max_validation_mdd_pct=float(args.max_validation_mdd_pct),
            selection_objective=str(args.selection_objective),
            proba_thresholds=_float_grid(str(args.proba_thresholds)),
            proba_gaps=_float_grid(str(args.proba_gaps)),
            min_trade_counts_5m=_float_grid(str(args.min_trade_counts_5m)),
            toxicity_caps=_float_grid(str(args.toxicity_caps)),
        )

        train_frame = labeled.iloc[train_slice].reset_index(drop=True)
        oos_frame = labeled.iloc[oos_slice].reset_index(drop=True)
        train_metrics = _trade_eval(
            model.predict_proba(x[train_slice]),
            classes,
            train_frame["fwd_ret"].to_numpy(float),
            train_frame,
            val_cfg,
            cost_per_notional=float(args.cost_per_notional),
        )
        oos_metrics = _trade_eval(
            model.predict_proba(x[oos_slice]),
            classes,
            oos_frame["fwd_ret"].to_numpy(float),
            oos_frame,
            val_cfg,
            cost_per_notional=float(args.cost_per_notional),
        )

        candidate = {
            "horizon_bars": int(horizon),
            "label_mode": str(args.label_mode),
            "label_threshold": float(label_threshold),
            "barrier_tp": float(args.barrier_tp),
            "barrier_sl": float(args.barrier_sl),
            "feature_count": int(len(feature_cols)),
            "selected_on": "validation_only",
            "selected_config": asdict(val_cfg),
            "splits": split_rows,
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "oos": oos_metrics,
            },
        }
        candidates.append(candidate)
        if best_bundle is None or _candidate_key(candidate, str(args.selection_objective)) > _candidate_key(
            best_bundle["candidate"],
            str(args.selection_objective),
        ):
            best_bundle = {
                "candidate": candidate,
                "model": model,
                "classes": classes.tolist(),
                "feature_cols": feature_cols,
            }

    if best_bundle is None:
        raise RuntimeError("no candidate could be trained")

    best = best_bundle["candidate"]
    start_utc = pd.Timestamp(data_meta["start_ts"]).tz_convert("UTC")
    end_utc = pd.Timestamp(data_meta["end_ts"]).tz_convert("UTC")
    live_trades = _live_trade_count(live_dir, start_utc, end_utc)

    val = best["metrics"]["validation"]
    oos = best["metrics"]["oos"]
    validation_pass = bool(val["pnl_pct"] > 0.0 and val["trades"] >= int(args.min_validation_trades))
    oos_pass = bool(oos["pnl_pct"] > 0.0 and oos["trades"] >= int(args.min_oos_trades))
    baseline_overlap_pass = bool(live_trades > 0)
    verdict = (
        "RESEARCH_PASS_NOT_BASELINE_UPGRADE"
        if validation_pass and oos_pass and baseline_overlap_pass
        else "BLOCKED_NO_CLEAN_BASELINE_UPGRADE"
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "duckdb_micro_context_sidecar.pkl"
    bundle = {
        "model_id": model_id,
        "base_model_id": BASE_MODEL_ID,
        "model": best_bundle["model"],
        "classes": best_bundle["classes"],
        "feature_cols": best_bundle["feature_cols"],
        "feature_mode": str(args.feature_mode),
        "feature_windows": feature_windows,
        "label_mode": str(args.label_mode),
        "barrier_tp": float(args.barrier_tp),
        "barrier_sl": float(args.barrier_sl),
        "balance_classes": bool(args.balance_classes),
        "selected_config": best["selected_config"],
        "enabled_for_baseline": False,
        "verdict": verdict,
    }
    joblib.dump(bundle, model_path)

    report = {
        "model_id": model_id,
        "base_model_id": BASE_MODEL_ID,
        "design": (
            "Causal DuckDB context sidecar for Omega4.4. "
            "The parent, exit head, and risk sidecar are not retrained. This candidate can only "
            "gate or scale future entries if it passes validation/OOS and baseline-overlap checks."
        ),
        "data_contract": {
            "source": str(live_dir),
            "feature_mode": str(args.feature_mode),
            "feature_contract": (
                "microstructure DuckDB numeric features only; Polymarket current_price is used only "
                "as a label/PnL anchor"
                if str(args.feature_mode) == "microstructure_only"
                else "microstructure, tail-risk, and price-derived context features"
            ),
            "uses_polymarket_event_markets": False,
            "polymarket_note": "Recent markets_json is empty; only current_price timestamp anchor is used.",
            "label": "future price return class generated after current timestamp; gap-checked horizon prevents downtime leakage",
            "label_mode": str(args.label_mode),
            "barrier_tp": float(args.barrier_tp),
            "barrier_sl": float(args.barrier_sl),
            "feature_windows": feature_windows,
            "balance_classes": bool(args.balance_classes),
            "cost_per_notional": float(args.cost_per_notional),
            "split_policy": "chronological 60/20/20; threshold selected on validation only; OOS fixed after selection",
            "selection_objective": str(args.selection_objective),
            "max_validation_mdd_pct": float(args.max_validation_mdd_pct),
            "proba_thresholds": _float_grid(str(args.proba_thresholds)),
            "proba_gaps": _float_grid(str(args.proba_gaps)),
            "min_trade_counts_5m": _float_grid(str(args.min_trade_counts_5m)),
            "toxicity_caps": _float_grid(str(args.toxicity_caps)),
        },
        "data_meta": data_meta,
        "live_closed_trade_overlap_count": live_trades,
        "selected_candidate": best,
        "all_candidates": candidates,
        "checks": {
            "validation_positive": validation_pass,
            "oos_positive": oos_pass,
            "baseline_trade_overlap_available": baseline_overlap_pass,
            "no_live_wiring_changed": True,
            "enabled_for_baseline": False,
        },
        "verdict": verdict,
        "artifacts": {
            "model": str(model_path),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    supervised_dir = Path(args.supervised_dir)
    supervised_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "model_id": model_id,
        "base_model_id": BASE_MODEL_ID,
        "status": "blocked_not_promoted" if verdict.startswith("BLOCKED") else "research_pass_not_live_wired",
        "verdict": verdict,
        "feature_mode": str(args.feature_mode),
        "feature_windows": feature_windows,
        "label_mode": str(args.label_mode),
        "barrier_tp": float(args.barrier_tp),
        "barrier_sl": float(args.barrier_sl),
        "balance_classes": bool(args.balance_classes),
        "report": str(out_dir / "report.json"),
        "model": str(model_path),
        "enabled_for_baseline": False,
        "reason": (
            "No clean baseline upgrade unless validation/OOS are both positive and baseline live-trade overlap exists."
        ),
    }
    (supervised_dir / "candidate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Omega4.4 DuckDB micro context sidecar.")
    p.add_argument("--model-id", default=MODEL_ID)
    p.add_argument("--feature-mode", choices=["full_context", "microstructure_only"], default="full_context")
    p.add_argument("--live-dir", type=Path, default=ROOT / "data/live")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--supervised-dir", type=Path, default=DEFAULT_SUPERVISED_DIR)
    p.add_argument("--horizons", default="5,15,30,60")
    p.add_argument("--feature-windows", default="3,5,15,30,60")
    p.add_argument("--label-mode", choices=["forward_return", "barrier"], default="forward_return")
    p.add_argument("--label-threshold", type=float, default=0.0015)
    p.add_argument("--label-threshold-5m", type=float, default=0.0010)
    p.add_argument("--barrier-tp", type=float, default=0.0030)
    p.add_argument("--barrier-sl", type=float, default=0.0015)
    p.add_argument("--cost-per-notional", type=float, default=0.0014)
    p.add_argument("--min-rows", type=int, default=5000)
    p.add_argument("--min-validation-trades", type=int, default=10)
    p.add_argument("--min-oos-trades", type=int, default=5)
    p.add_argument("--max-validation-mdd-pct", type=float, default=100.0)
    p.add_argument("--selection-objective", choices=["pnl", "risk_adjusted"], default="pnl")
    p.add_argument("--proba-thresholds", default="0.40,0.48,0.56,0.64,0.72")
    p.add_argument("--proba-gaps", default="0.02,0.06,0.10,0.16")
    p.add_argument("--min-trade-counts-5m", default="0.0,1000.0,2500.0")
    p.add_argument("--toxicity-caps", default="0.75,1.00,1.50,2.10")
    p.add_argument("--max-iter", type=int, default=180)
    p.add_argument("--learning-rate", type=float, default=0.045)
    p.add_argument("--max-leaf-nodes", type=int, default=15)
    p.add_argument("--l2-regularization", type=float, default=0.05)
    p.add_argument("--balance-classes", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = _run(args)
    selected = report["selected_candidate"]
    print(
        json.dumps(
            {
                "model_id": report["model_id"],
                "verdict": report["verdict"],
                "selected_horizon": selected["horizon_bars"],
                "validation": selected["metrics"]["validation"],
                "oos": selected["metrics"]["oos"],
                "live_closed_trade_overlap_count": report["live_closed_trade_overlap_count"],
                "report": report["artifacts"]["report"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
