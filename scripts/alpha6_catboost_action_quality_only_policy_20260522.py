#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor, Pool

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha6_catboost_5head_policy_20260522 import (  # noqa: E402
    DEFAULT_FEATURE_CSV,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPEC_DIR,
    _days,
    _feature_matrix,
    _fill_price,
    _json_default,
    _label_frame,
    _read_feature_frame,
    _read_spec,
    _score,
)


MODEL_ID = "alpha6_catboost_action_quality_only_policy_20260522"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha6_catboost_action_quality_only_current_tail111_20260522"


@dataclass(frozen=True)
class ActionQualityConfig:
    fixed_notional: float = 1.0
    max_train_horizon_bars: int = 96
    fee: float = 0.0004
    slip: float = 0.00015
    cash_score: float = 0.0008
    min_net_edge: float = 0.00045
    dynamic_min_edge_atr_frac: float = 0.15
    direction_margin: float = 0.00025
    mae_penalty_lambda: float = 0.55
    path_vol_penalty_lambda: float = 0.08
    terminal_weight: float = 0.35
    mfe_weight: float = 0.65


class _ConstantClassifier:
    def __init__(self, cls: int) -> None:
        self.cls = int(cls)
        self.classes_ = np.asarray([self.cls], dtype=int)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return np.ones((len(x), 1), dtype=np.float64)


class _ConstantRegressor:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.full(len(x), self.value, dtype=np.float64)


def _build_action_quality_labels(
    frame: pd.DataFrame,
    cfg: ActionQualityConfig,
    *,
    stride_bars: int,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    atr = pd.to_numeric(frame.get("atr14_pct", 0.003), errors="coerce").fillna(0.003).to_numpy(dtype=np.float64)
    h = int(cfg.max_train_horizon_bars)
    valid = np.arange(0, max(0, len(frame) - h - 1), max(1, int(stride_bars)), dtype=np.int64)
    if valid.size == 0:
        raise ValueError("no train candidates for action-quality labels")
    y = {
        "action": np.zeros(valid.size, dtype=np.int64),
        "quality": np.full(valid.size, float(cfg.cash_score), dtype=np.float64),
    }
    horizons = np.arange(1, h + 1, dtype=np.int64)
    cost = 2.0 * float(cfg.fee + cfg.slip) * float(cfg.fixed_notional)
    for start in range(0, valid.size, int(batch_size)):
        end = min(start + int(batch_size), valid.size)
        idx = valid[start:end]
        entry = np.maximum(close[idx], 1e-12)
        fut = close[idx[:, None] + horizons[None, :]]
        raw_ret = fut / entry[:, None] - 1.0
        atr_now = atr[idx]
        min_edge = np.maximum(float(cfg.min_net_edge), atr_now * float(cfg.dynamic_min_edge_atr_frac) * float(cfg.fixed_notional))
        scores: dict[int, np.ndarray] = {}
        for action, side in ((1, 1.0), (2, -1.0)):
            path = raw_ret * side
            terminal = path[:, -1]
            mfe = np.max(path, axis=1)
            mae = np.maximum(0.0, -np.min(path, axis=1))
            path_vol = np.nanstd(path, axis=1)
            time_to_best = np.argmax(path, axis=1).astype(np.float64) + 1.0
            fast_mfe = mfe / np.sqrt(time_to_best)
            score = (
                (float(cfg.terminal_weight) * terminal + float(cfg.mfe_weight) * fast_mfe) * float(cfg.fixed_notional)
                - float(cfg.mae_penalty_lambda) * mae * float(cfg.fixed_notional)
                - float(cfg.path_vol_penalty_lambda) * path_vol * float(cfg.fixed_notional)
                - cost
            )
            scores[action] = score
        long_score = scores[1]
        short_score = scores[2]
        choose_long = ((long_score - short_score) > float(cfg.direction_margin)) & (long_score > min_edge)
        choose_short = ((short_score - long_score) > float(cfg.direction_margin)) & (short_score > min_edge)
        y["action"][start:end] = np.where(choose_long, 1, np.where(choose_short, 2, 0)).astype(np.int64)
        y["quality"][start:end] = np.where(choose_long, long_score, np.where(choose_short, short_score, float(cfg.cash_score)))
    meta = {
        "candidates": int(valid.size),
        "stride_bars": int(stride_bars),
        "max_train_horizon_bars": int(h),
        "trained_heads": ["action", "quality"],
        "removed_heads": ["notional", "take_profit", "stop_loss", "max_hold", "cooldown"],
        "labeling_basis": "risk_adjusted_directional_path_no_tp_sl",
        **asdict(cfg),
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


def _regressor_params(args: argparse.Namespace, seed: int) -> dict[str, Any]:
    params: dict[str, Any] = {
        "loss_function": "RMSE",
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


def _fit_models(x: np.ndarray, y: dict[str, np.ndarray], args: argparse.Namespace) -> dict[str, Any]:
    trade = y["action"] != 0
    q_w = np.clip(np.abs(y["quality"]), 0.03, 1.0)
    weight = np.maximum(np.where(trade, 1.0, float(args.cash_action_weight)), q_w)
    if np.unique(y["action"]).size < 2:
        action_model: Any = _ConstantClassifier(int(y["action"][0]) if len(y["action"]) else 0)
    else:
        action_model = CatBoostClassifier(**_classifier_params(args, args.seed))
        action_model.fit(Pool(x, y["action"], weight=weight))
    if np.unique(y["quality"]).size < 2:
        quality_model: Any = _ConstantRegressor(float(y["quality"][0]) if len(y["quality"]) else 0.0)
    else:
        quality_model = CatBoostRegressor(**_regressor_params(args, args.seed + 99))
        quality_model.fit(Pool(x, y["quality"], weight=weight))
    return {
        "action_model": action_model,
        "quality_model": quality_model,
        "label_distribution": {
            "action": pd.Series(y["action"]).value_counts().sort_index().to_dict(),
            "quality_mean": float(np.mean(y["quality"])),
            "quality_p95": float(np.quantile(y["quality"], 0.95)),
        },
    }


def _predict_policy(models: dict[str, Any], x: np.ndarray, cfg: ActionQualityConfig) -> pd.DataFrame:
    action_proba = models["action_model"].predict_proba(x)
    action_classes = np.asarray(models["action_model"].classes_, dtype=int)
    action = action_classes[np.argmax(action_proba, axis=1)].astype(np.int64)
    quality = np.asarray(models["quality_model"].predict(x), dtype=np.float64)
    return pd.DataFrame(
        {
            "action": action,
            "quality_score": quality,
            "confidence": np.max(action_proba, axis=1),
            "notional": np.full(len(x), float(cfg.fixed_notional), dtype=np.float64),
        }
    )


def _backtest_action_only(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    threshold: float,
    fee: float,
    slip: float,
    min_hold_bars: int,
) -> dict[str, Any]:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    exposure = 0.0
    hold = 0
    trades = wins = long_entries = short_entries = 0
    exposure_sum = 0.0
    action_counts = {"flat": 0, "long": 0, "short": 0}
    exits: dict[str, int] = {}

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * exposure)

    def enter(i: int, new_side: int, notional: float) -> None:
        nonlocal side, entry, entry_equity, exposure, hold, cash, exposure_sum, long_entries, short_entries
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        exposure = float(np.clip(notional, 0.01, 2.0))
        entry = _fill_price(frame, fill_i, side, slip, entry=True)
        entry_equity = cash
        cash -= cash * fee * exposure
        hold = 0
        exposure_sum += exposure
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str) -> None:
        nonlocal side, entry, cash, hold, exposure, trades, wins
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
        exposure = 0.0

    for i in range(len(frame) - 2):
        row = dec.iloc[i]
        desired = int(row.action) if float(row.quality_score) >= float(threshold) else 0
        action_counts["flat" if desired == 0 else "long" if desired == 1 else "short"] += 1
        if side != 0:
            hold += 1
            if hold >= int(min_hold_bars):
                if desired == 0:
                    exit_pos(i, "model_cash")
                elif (desired == 1 and side < 0) or (desired == 2 and side > 0):
                    exit_pos(i, "model_flip")
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1, float(row.notional))
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


def _threshold_grid(dec: pd.DataFrame, n: int) -> np.ndarray:
    active = dec.loc[dec["action"] != 0, "quality_score"].to_numpy(dtype=np.float64)
    active = active[np.isfinite(active)]
    if active.size == 0:
        return np.array([np.inf])
    return np.unique(np.quantile(active, np.linspace(0.10, 0.995, int(n))))


def main() -> None:
    ap = argparse.ArgumentParser(description="Alpha6 action/quality-only CatBoost ablation with fixed notional and no TP/SL.")
    ap.add_argument("--feature-csv", type=Path, default=DEFAULT_FEATURE_CSV)
    ap.add_argument("--spec-dir", type=Path, default=DEFAULT_SPEC_DIR)
    ap.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--iterations", type=int, default=650)
    ap.add_argument("--learning-rate", type=float, default=0.055)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--l2-leaf-reg", type=float, default=5.0)
    ap.add_argument("--task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stride-bars", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--thresholds", type=int, default=70)
    ap.add_argument("--fixed-notional", type=float, default=1.0)
    ap.add_argument("--min-hold-bars", type=int, default=1)
    ap.add_argument("--cash-action-weight", type=float, default=0.35)
    ap.add_argument("--verbose", type=int, default=100)
    ap.add_argument("--no-pca", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    cfg = replace(ActionQualityConfig(), fixed_notional=float(args.fixed_notional))
    spec = _read_spec(args.spec_dir, args.variant)
    use_pca = bool(spec.get("extra_pca_enable")) and not args.no_pca and int(spec.get("extra_pca_components") or 0) > 0
    feat, present, missing = _read_feature_frame(args.feature_csv, list(spec["features"]), [])
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
    valid, y, label_meta = _build_action_quality_labels(train, cfg, stride_bars=args.stride_bars, batch_size=args.batch_size)
    x_train = x_train_all[valid]
    print(
        f"[alpha6-action-quality] variant={args.variant} train_rows={len(train)} val_rows={len(val)} labels={len(valid)} features={len(model_features)} use_pca={use_pca}",
        flush=True,
    )
    models = _fit_models(x_train, y, args)
    dec = _predict_policy(models, x_val, cfg)
    rows = []
    best: dict[str, Any] | None = None
    for th in _threshold_grid(dec, args.thresholds):
        bt = {
            f"cost{m}": _backtest_action_only(
                val,
                dec,
                threshold=float(th),
                fee=cfg.fee * m,
                slip=cfg.slip * m,
                min_hold_bars=int(args.min_hold_bars),
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
