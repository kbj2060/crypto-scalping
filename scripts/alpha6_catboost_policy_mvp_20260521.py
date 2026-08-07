#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from catboost import CatBoostRanker, Pool
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "alpha6_catboost_policy_mvp_20260521"
DEFAULT_FEATURE_CSV = ROOT / "tmp/causal_regen_20260516/dsac_feature_inventory_regime_fixed_20260521/rl_training_2025_direction_router_feature_inventory_base_with_family_pca.csv"
DEFAULT_SPEC_DIR = ROOT / "tmp/causal_regen_20260516/dsac_feature_variant_specs_regime_fixed_20260521"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_13_hgb_high_quality_training_data_20260518"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_policy_mvp_20260521"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _read_spec(spec_dir: Path, variant: str) -> dict[str, Any]:
    path = spec_dir / f"{variant}.json"
    if not path.exists():
        raise FileNotFoundError(f"missing feature spec: {path}")
    spec = json.loads(path.read_text())
    features = list(spec.get("features") or spec.get("feature_cols") or [])
    if not features:
        raise ValueError(f"empty feature list in {path}")
    spec["features"] = features
    return spec


def _label_frame(label_dir: Path) -> pd.DataFrame:
    parts = []
    for name in ("alpha5_13_hgb_atr_barrier_labels_train.parquet", "alpha5_13_hgb_atr_barrier_labels_val.parquet"):
        path = label_dir / name
        if not path.exists():
            raise FileNotFoundError(f"missing alpha5 label file: {path}")
        cols = [
            "timestamp",
            "label_action",
            "label_tp_pct",
            "label_sl_pct",
            "label_confidence",
            "label_sample_weight",
            "meta_long_score",
            "meta_short_score",
            "dataset_split",
        ]
        available = set(pq.ParquetFile(path).schema.names)
        frame = pd.read_parquet(path, columns=[c for c in cols if c in available])
        parts.append(frame)
    out = pd.concat(parts, ignore_index=True)
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce")
    out = out.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")
    return out


def _read_feature_frame(feature_csv: Path, features: list[str], extra_cols: list[str]) -> tuple[pd.DataFrame, list[str], list[str]]:
    all_cols = pd.read_csv(feature_csv, nrows=0).columns.tolist()
    available = set(all_cols)
    missing = [c for c in features if c not in available]
    present = [c for c in features if c in available]
    keep = []
    for c in ["timestamp", "open", "high", "low", "close", *extra_cols, *present]:
        if c in available and c not in keep:
            keep.append(c)
    frame = pd.read_csv(feature_csv, usecols=keep, parse_dates=["timestamp"], low_memory=False)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return frame, present, missing


def _clean_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
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
) -> tuple[np.ndarray, np.ndarray, list[str], Pipeline | None]:
    x_train = _clean_numeric(train, cols)
    x_val = _clean_numeric(val, cols)
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


def _sigmoid(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -8.0, 8.0)
    return 1.0 / (1.0 + np.exp(-z))


def _rank_targets(frame: pd.DataFrame, side: int, mode: str) -> np.ndarray:
    label = pd.to_numeric(frame["label_action"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)
    conf = pd.to_numeric(frame.get("label_confidence", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    score_col = "meta_long_score" if side == 1 else "meta_short_score"
    meta = pd.to_numeric(frame.get(score_col, 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    if mode == "meta_sigmoid":
        # Side-specific alpha5 meta score is the closest available continuous rank label:
        # high score means this timestamp is better for that side, regardless of the final discrete action label.
        return _sigmoid(meta).astype(np.float32)
    meta_bonus = np.clip(meta, 0.0, 5.0) * 0.05
    return np.where(label == side, 1.0 + 0.25 * conf + meta_bonus, 0.0).astype(np.float32)


def _group_id(frame: pd.DataFrame) -> np.ndarray:
    ts = pd.to_datetime(frame["timestamp"])
    return ts.dt.strftime("%Y-%m-%d").to_numpy()


def _fit_ranker(
    x_train: np.ndarray,
    y_train: np.ndarray,
    group_train: np.ndarray,
    *,
    iterations: int,
    learning_rate: float,
    depth: int,
    task_type: str,
    seed: int,
    verbose: int,
) -> CatBoostRanker:
    params: dict[str, Any] = {
        "loss_function": "YetiRank",
        "iterations": int(iterations),
        "learning_rate": float(learning_rate),
        "depth": int(depth),
        "random_seed": int(seed),
        "verbose": int(verbose),
        "allow_writing_files": False,
        "thread_count": -1,
    }
    if task_type.upper() == "GPU":
        params.update({"task_type": "GPU", "devices": "0"})
    model = CatBoostRanker(**params)
    model.fit(Pool(x_train, y_train, group_id=group_train))
    return model


def _fill_price(frame: pd.DataFrame, i: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(frame["open"], errors="coerce").ffill().iloc[int(np.clip(i, 0, len(frame) - 1))])
    if entry:
        return px * (1.0 + slip if side > 0 else 1.0 - slip)
    return px * (1.0 - slip if side > 0 else 1.0 + slip)


def _days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    if len(ts) < 2:
        return 1.0
    span = (ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0
    return float(max(span, 1.0))


def _direction_metrics(actions: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    trade = actions != 0
    n_trade = int(np.sum(trade))
    out: dict[str, Any] = {"coverage": float(np.mean(trade)), "trades_pred": n_trade}
    if n_trade == 0:
        out.update({"trade_precision": 0.0, "balanced_trade_precision": 0.0, "long_precision": 0.0, "short_precision": 0.0, "long_pred": 0, "short_pred": 0})
        return out
    out["trade_precision"] = float(np.mean(actions[trade] == labels[trade]))
    parts = []
    for cls, name in ((1, "long"), (2, "short")):
        mask = trade & (actions == cls)
        if np.any(mask):
            p = float(np.mean(labels[mask] == cls))
            out[f"{name}_precision"] = p
            out[f"{name}_pred"] = int(np.sum(mask))
            parts.append(p)
        else:
            out[f"{name}_precision"] = 0.0
            out[f"{name}_pred"] = 0
    out["balanced_trade_precision"] = float(np.mean(parts)) if parts else 0.0
    return out


def _backtest_barrier(
    frame: pd.DataFrame,
    actions: np.ndarray,
    tp_pct: np.ndarray,
    sl_pct: np.ndarray,
    notional: np.ndarray,
    *,
    fee: float,
    slip: float,
    max_hold_bars: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    hold = 0
    tp = 0.0
    sl = 0.0
    exposure = 0.0
    trades = wins = long_entries = short_entries = 0
    exits: dict[str, int] = {}
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exposure_sum = 0.0

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, cash, hold, tp, sl, exposure, long_entries, short_entries, exposure_sum
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        exposure = float(np.clip(notional[i], 0.01, 1.0))
        entry = _fill_price(frame, fill_i, side, float(slip), entry=True)
        entry_equity = cash
        cash -= cash * float(fee) * exposure
        hold = 0
        tp = float(max(tp_pct[i], 1e-4))
        sl = float(max(sl_pct[i], 1e-4))
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, tp, sl, exposure, trades, wins
        if fill_px is None:
            fill_i = min(i + 1, len(frame) - 1)
            fill_px = _fill_price(frame, fill_i, side, float(slip), entry=False)
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * exposure)
        cash -= before * float(fee) * exposure
        trades += 1
        wins += int(cash > entry_equity)
        exits[reason] = exits.get(reason, 0) + 1
        side = 0
        entry = 0.0
        hold = 0
        tp = 0.0
        sl = 0.0
        exposure = 0.0

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold += 1
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + tp)
                sl_hit = low[i] <= entry * (1.0 - sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - sl) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + tp) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - sl) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - tp)
                sl_hit = high[i] >= entry * (1.0 + sl)
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + sl) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - tp) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + sl) * (1.0 + float(slip)))
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side != 0 and int(max_hold_bars) > 0 and hold >= int(max_hold_bars):
            exit_pos(i, "max_hold")
        elif side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1)
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
        "action_counts": action_counts,
        "exits": exits,
    }


def _guardrail_notional(frame: pd.DataFrame) -> np.ndarray:
    base = np.full(len(frame), 0.25, dtype=np.float64)
    risk_cols = [
        "market_state_2024_unsup_v5_risk_off_prob",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "regime4_pred_instability_prob",
        "regime4_pred_whipsaw_prob",
    ]
    trend_cols = [
        "market_state_2024_unsup_v5_trend_prob",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "regime4_pred_trend_prob",
    ]
    conf_cols = ["clean_regime4_state24_sticky090_v2_confidence", "regime4_pred_confidence", "market_state_2024_unsup_v5_confidence"]
    risk = np.zeros(len(frame), dtype=np.float64)
    trend = np.zeros(len(frame), dtype=np.float64)
    conf = np.zeros(len(frame), dtype=np.float64)
    for col in risk_cols:
        if col in frame:
            risk = np.maximum(risk, pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    for col in trend_cols:
        if col in frame:
            trend = np.maximum(trend, pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    for col in conf_cols:
        if col in frame:
            conf = np.maximum(conf, pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64))
    base[(risk > 0.60)] = 0.15
    base[(risk <= 0.45) & (trend > 0.55) & (conf > 0.45)] = 0.40
    return base


def _guardrail_score_adjustment(frame: pd.DataFrame) -> np.ndarray:
    adj = np.zeros(len(frame), dtype=np.float64)
    for col, weight in (
        ("market_state_2024_unsup_v5_risk_off_prob", -0.15),
        ("clean_regime4_state24_sticky090_v2_instability_prob", -0.10),
        ("clean_regime4_state24_sticky090_v2_whipsaw_prob", -0.08),
        ("clean_regime4_state24_sticky090_v2_confidence", 0.03),
        ("clean_regime4_state24_sticky090_v2_trend_prob", 0.03),
    ):
        if col in frame:
            adj += weight * pd.to_numeric(frame[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return adj


def _atr_tp_sl(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    tp = np.clip(1.5 * atr, 0.0015, 0.0120)
    sl = np.clip(1.0 * atr, 0.0008, 0.0100)
    return tp, sl


def _eval_policy(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    max_hold: int,
    fee: float,
    slip: float,
    labels: np.ndarray,
) -> dict[str, Any]:
    tp, sl = _atr_tp_sl(frame)
    notional = _guardrail_notional(frame)
    bt = {
        f"cost{m}": _backtest_barrier(
            frame,
            actions,
            tp,
            sl,
            notional,
            fee=float(fee) * float(m),
            slip=float(slip) * float(m),
            max_hold_bars=int(max_hold),
        )
        for m in (1, 2, 3)
    }
    dm = _direction_metrics(actions, labels)
    c1, c2, c3 = bt["cost1"], bt["cost2"], bt["cost3"]
    if int(c1["trades"]) < 15:
        alpha5_score = -1e6 + float(c1["pnl"])
        alpha6_score = alpha5_score
    else:
        alpha5_score = (
            18.0 * float(dm["balanced_trade_precision"])
            + 10.0 * float(dm["trade_precision"])
            + float(c1["pnl"])
            + 0.35 * float(c2["pnl"])
            + 0.10 * float(c3["pnl"])
            - 0.22 * abs(float(c1["mdd"]))
            - max(0.0, 0.10 - float(dm["coverage"])) * 12.0
            - max(0.0, float(c1["trades_per_day"]) - 2.5) * 2.5
        )
        tpd = float(c1["trades_per_day"])
        density_pen = max(0.0, 5.0 - tpd) * 4.0 + max(0.0, tpd - 10.0) * 3.0
        alpha6_score = (
            14.0 * float(dm["balanced_trade_precision"])
            + 8.0 * float(dm["trade_precision"])
            + float(c1["pnl"])
            + 0.35 * float(c2["pnl"])
            + 0.10 * float(c3["pnl"])
            - 0.25 * abs(float(c1["mdd"]))
            - density_pen
        )
    return {"backtest": bt, "direction": dm, "alpha5_score": float(alpha5_score), "alpha6_score": float(alpha6_score)}


def _threshold_grid(score: np.ndarray, n: int = 80) -> np.ndarray:
    finite = score[np.isfinite(score)]
    if len(finite) == 0:
        return np.array([0.0], dtype=np.float64)
    qs = np.linspace(0.50, 0.995, n)
    return np.unique(np.quantile(finite, qs))


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 CatBoost policy MVP: entry ranker + ATR TP/SL + regime guardrail.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="stable48_global_pca32")
    ap.add_argument("--iterations", type=int, default=800)
    ap.add_argument("--learning-rate", type=float, default=0.045)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-hold", type=int, default=96)
    ap.add_argument("--fee", type=float, default=0.0004)
    ap.add_argument("--slip", type=float, default=0.00015)
    ap.add_argument("--thresholds", type=int, default=80)
    ap.add_argument("--target-mode", choices=["meta_sigmoid", "binary"], default="meta_sigmoid")
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    spec = _read_spec(args.spec_dir, args.variant)
    features = list(spec["features"])
    pca_components = int(spec.get("extra_pca_components") or 0)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and pca_components > 0
    extra_cols = [
        "atr14_pct",
        "market_state_2024_unsup_v5_risk_off_prob",
        "market_state_2024_unsup_v5_trend_prob",
        "market_state_2024_unsup_v5_confidence",
        "clean_regime4_state24_sticky090_v2_instability_prob",
        "clean_regime4_state24_sticky090_v2_whipsaw_prob",
        "clean_regime4_state24_sticky090_v2_confidence",
        "clean_regime4_state24_sticky090_v2_trend_prob",
        "regime4_pred_instability_prob",
        "regime4_pred_whipsaw_prob",
        "regime4_pred_trend_prob",
        "regime4_pred_confidence",
    ]
    feat, present, missing = _read_feature_frame(args.feature_csv, features, extra_cols)
    labels = _label_frame(args.label_dir)
    frame = feat.merge(labels, on="timestamp", how="inner").sort_values("timestamp").reset_index(drop=True)
    train = frame[frame["dataset_split"].astype(str).str.lower().eq("train")].copy()
    val = frame[frame["dataset_split"].astype(str).str.lower().ne("train")].copy()
    if args.smoke:
        train = train.iloc[: min(len(train), 4000)].copy()
        val = val.iloc[: min(len(val), 2500)].copy()
        args.iterations = min(args.iterations, 20)
        args.thresholds = min(args.thresholds, 8)
    if len(train) < 100 or len(val) < 100:
        raise RuntimeError(f"insufficient train/val rows after merge: train={len(train)} val={len(val)}")
    x_train, x_val, model_features, pipe = _feature_matrix(train, val, present, use_pca=use_pca, pca_components=pca_components)
    group_train = _group_id(train)
    y_long = _rank_targets(train, 1, args.target_mode)
    y_short = _rank_targets(train, 2, args.target_mode)

    print(f"[alpha6] variant={args.variant} rows train={len(train)} val={len(val)} raw_features={len(present)} missing={len(missing)} model_features={len(model_features)} use_pca={use_pca}", flush=True)
    long_model = _fit_ranker(x_train, y_long, group_train, iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, task_type=args.task_type, seed=args.seed, verbose=100)
    short_model = _fit_ranker(x_train, y_short, group_train, iterations=args.iterations, learning_rate=args.learning_rate, depth=args.depth, task_type=args.task_type, seed=args.seed + 17, verbose=100)

    long_score = np.asarray(long_model.predict(x_val), dtype=np.float64)
    short_score = np.asarray(short_model.predict(x_val), dtype=np.float64)
    score_adj = _guardrail_score_adjustment(val)
    best_long = long_score >= short_score
    best_score = np.maximum(long_score, short_score) + score_adj
    side = np.where(best_long, 1, 2).astype(np.int32)
    true_labels = pd.to_numeric(val["label_action"], errors="coerce").fillna(0).to_numpy(dtype=np.int32)

    rows = []
    best: dict[str, Any] | None = None
    for threshold in _threshold_grid(best_score, args.thresholds):
        actions = np.where(best_score >= threshold, side, 0).astype(np.int32)
        result = _eval_policy(val, actions, max_hold=args.max_hold, fee=args.fee, slip=args.slip, labels=true_labels)
        c1 = result["backtest"]["cost1"]
        dm = result["direction"]
        row = {
            "threshold": float(threshold),
            "alpha6_score": float(result["alpha6_score"]),
            "alpha5_score": float(result["alpha5_score"]),
            "pnl": float(c1["pnl"]),
            "mdd": float(c1["mdd"]),
            "trades": int(c1["trades"]),
            "trades_per_day": float(c1["trades_per_day"]),
            "wr": float(c1["wr"]),
            "long_entries": int(c1["long_entries"]),
            "short_entries": int(c1["short_entries"]),
            "avg_notional": float(c1["avg_notional"]),
            "trade_precision": float(dm["trade_precision"]),
            "balanced_trade_precision": float(dm["balanced_trade_precision"]),
            "coverage": float(dm["coverage"]),
            "exits": json.dumps(c1["exits"], sort_keys=True),
        }
        rows.append(row)
        if best is None or row["alpha6_score"] > float(best["summary"]["alpha6_score"]):
            best = {"summary": row, "result": result, "actions": actions}
    if best is None:
        raise RuntimeError("no threshold candidates evaluated")

    grid = pd.DataFrame(rows).sort_values("alpha6_score", ascending=False)
    grid.to_csv(args.out_dir / f"{args.variant}_threshold_grid.csv", index=False)
    pred = val[["timestamp", "open", "high", "low", "close", "label_action", "label_tp_pct", "label_sl_pct"]].copy()
    pred["long_score"] = long_score
    pred["short_score"] = short_score
    pred["score_adjustment"] = score_adj
    pred["best_score"] = best_score
    pred["action"] = best["actions"]
    pred.to_csv(args.out_dir / f"{args.variant}_val_predictions.csv", index=False)

    long_model.save_model(str(args.out_dir / f"{args.variant}_long_ranker.cbm"))
    short_model.save_model(str(args.out_dir / f"{args.variant}_short_ranker.cbm"))
    if pipe is not None:
        import joblib

        joblib.dump(pipe, args.out_dir / f"{args.variant}_feature_pipeline.joblib")
    try:
        fi_long = long_model.get_feature_importance(Pool(x_train, y_long, group_id=group_train))
        fi_short = short_model.get_feature_importance(Pool(x_train, y_short, group_id=group_train))
        pd.DataFrame({"feature": model_features, "long_importance": fi_long, "short_importance": fi_short}).sort_values(
            ["long_importance", "short_importance"], ascending=False
        ).to_csv(args.out_dir / f"{args.variant}_feature_importance.csv", index=False)
    except Exception as exc:
        print(f"[alpha6] feature importance skipped: {exc}", flush=True)

    summary = {
        "model_id": MODEL_ID,
        "variant": args.variant,
        "feature_csv": args.feature_csv,
        "spec": args.spec_dir / f"{args.variant}.json",
        "out_dir": args.out_dir,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "raw_feature_count": int(len(present)),
        "missing_features": missing,
        "use_pca": bool(use_pca),
        "model_feature_count": int(len(model_features)),
        "best": best["summary"],
        "best_detail": best["result"],
        "params": vars(args),
    }
    (args.out_dir / f"{args.variant}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(summary["best"], ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
