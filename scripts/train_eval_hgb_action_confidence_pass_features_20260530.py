#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "hgb_action_confidence_pass_features_20260530"
DEFAULT_CANDIDATE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529"
DEFAULT_CHRONOS_DIR = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"
DEFAULT_REGIME3_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/hgb_action_confidence_pass_features_20260530"

BASE_REQUIRED_FEATURES = [
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_vol_regime_pct",
    "tide_vol_raw",
    "tide_vol_zscore",
    "m7_hold_pred",
    "m7_q10",
    "m7_q90",
    "m7_qwidth",
    "m7_quality_pred",
    "cvp_regime",
    "regime_trending",
]

CHRONOS_REQUIRED_FEATURES = [
    "chronos_atr14_upside_band_ewm3",
    "chronos_atr14_width_ewm6",
    "chronos_atr14_width",
    "chronos_atr14_large_move_score",
    "chronos_realized_vol24_width",
    "chronos_realized_vol24_large_move_score",
]

REGIME3_REQUIRED_FEATURES = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]

OPTIONAL_FEATURES = [
    "regime_persistence",
    "m7_long_adverse_prob",
    "m7_long_mae_q90",
    "m7_short_adverse_prob",
    "m7_short_mae_q90",
    "m7_tradeability_score",
    "m7_entry_long_offset",
    "m7_entry_short_offset",
    "m7_tp_offset",
    "m7_sl_offset",
    "ai_anchor_revert_prob",
    "ai_anchor_overheat",
    "ai_anchor_trend_escape_prob",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
]

FEATURE_EXCLUDE = {
    "tp_sl_action_score",
}


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


def _read_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "open", "high", "low", "close", "tp_sl_action_score"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _chronos_features(path: Path, prefix: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"])
    required = {"timestamp", "q10", "q50", "q90", "width"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required Chronos columns: {missing}")
    width = pd.to_numeric(frame["width"], errors="coerce").clip(lower=0.0)
    q50 = pd.to_numeric(frame["q50"], errors="coerce")
    q90 = pd.to_numeric(frame["q90"], errors="coerce").clip(lower=0.0)
    out = pd.DataFrame({"timestamp": frame["timestamp"]})
    out[f"chronos_{prefix}_width"] = width
    out[f"chronos_{prefix}_large_move_score"] = width * (1.0 + q50.abs())
    out[f"chronos_{prefix}_upside_band_ewm3"] = q90.ewm(span=3, adjust=False, min_periods=1).mean()
    out[f"chronos_{prefix}_width_ewm6"] = width.ewm(span=6, adjust=False, min_periods=1).mean()
    return out.dropna(subset=["timestamp"]).drop_duplicates("timestamp", keep="last")


def _exact_join(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], source: str, *, allow_tail_drop: bool = False) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    missing = {col: int(merged[col].isna().sum()) for col in cols if int(merged[col].isna().sum()) > 0}
    if missing:
        miss_any = merged[cols].isna().any(axis=1).to_numpy()
        miss_idx = np.flatnonzero(miss_any)
        tail_only = miss_idx.size > 0 and np.array_equal(miss_idx, np.arange(len(merged) - miss_idx.size, len(merged)))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - miss_idx.size].reset_index(drop=True)
        raise RuntimeError(f"{source} exact timestamp join has missing values: {missing}")
    return merged


def _add_sidecars(frame: pd.DataFrame, *, year: int, chronos_dir: Path, regime3_dir: Path) -> pd.DataFrame:
    tag = "val2025" if int(year) == 2025 else "oos2026"
    atr = _chronos_features(chronos_dir / f"atr14_pct_{tag}_chronos.csv", "atr14")
    atr = atr.rename(
        columns={
            "chronos_atr14_upside_band_ewm3": "chronos_atr14_upside_band_ewm3",
            "chronos_atr14_width_ewm6": "chronos_atr14_width_ewm6",
            "chronos_atr14_width": "chronos_atr14_width",
            "chronos_atr14_large_move_score": "chronos_atr14_large_move_score",
        }
    )
    rv = _chronos_features(chronos_dir / f"realized_vol_24_{tag}_chronos.csv", "realized_vol24")
    frame = _exact_join(
        frame,
        atr,
        [
            "chronos_atr14_upside_band_ewm3",
            "chronos_atr14_width_ewm6",
            "chronos_atr14_width",
            "chronos_atr14_large_move_score",
        ],
        f"Chronos atr14 {tag}",
    )
    frame = _exact_join(
        frame,
        rv,
        ["chronos_realized_vol24_width", "chronos_realized_vol24_large_move_score"],
        f"Chronos realized_vol_24 {tag}",
    )
    regime_name = "training_features_2025_regime3_stability_risk_h6.csv" if int(year) == 2025 else "training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
    regime = pd.read_csv(regime3_dir / regime_name, parse_dates=["timestamp"])
    frame = _exact_join(frame, regime, REGIME3_REQUIRED_FEATURES, f"Regime3 stability h6 {year}", allow_tail_drop=True)
    return frame


def _days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="coerce")
    span = (ts.max() - ts.min()).total_seconds() / 86400.0
    return float(max(span, 1.0))


def _build_labels(score: pd.Series, threshold: float) -> np.ndarray:
    values = pd.to_numeric(score, errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return np.where(values > threshold, 1, np.where(values < -threshold, 2, 0)).astype(np.int64)


def _all_numeric_feature_cols(train: pd.DataFrame, oos: pd.DataFrame) -> list[str]:
    common = [col for col in train.columns if col in oos.columns]
    out: list[str] = []
    for col in common:
        if col == "timestamp" or col in FEATURE_EXCLUDE:
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(oos[col]):
            out.append(col)
    if not out:
        raise ValueError("all_numeric feature mode produced an empty feature list")
    return out


def _read_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        obj = json.loads(path.read_text())
        if isinstance(obj, dict):
            obj = obj.get("feature_cols") or obj.get("features")
        if not isinstance(obj, list):
            raise ValueError(f"{path} does not contain a feature list")
        return [str(x) for x in obj]
    frame = pd.read_csv(path)
    for col in ("feature", "feature_col", "feature_cols"):
        if col in frame.columns:
            return [str(x) for x in frame[col].dropna().tolist()]
    raise ValueError(f"{path} must contain one of feature/feature_col/feature_cols")


def _predict_actions(model: Pipeline, frame: pd.DataFrame, feature_cols: list[str], confidence_threshold: float) -> pd.DataFrame:
    proba = model.predict_proba(frame[feature_cols])
    classes = list(model.named_steps["model"].classes_)
    full = np.zeros((len(frame), 3), dtype=np.float64)
    for j, cls in enumerate(classes):
        full[:, int(cls)] = proba[:, j]
    raw_action = np.argmax(full, axis=1).astype(np.int64)
    confidence = np.max(full, axis=1)
    action = np.where(confidence >= float(confidence_threshold), raw_action, 0).astype(np.int64)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            "action": action,
            "confidence": confidence,
            "raw_action": raw_action,
            "p_cash": full[:, 0],
            "p_long": full[:, 1],
            "p_short": full[:, 2],
        }
    )


def _backtest_barrier(
    frame: pd.DataFrame,
    actions: np.ndarray,
    *,
    fee: float,
    slip: float,
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
    exposure: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    side = 0
    entry = 0.0
    entry_equity = 1.0
    entry_ts: Any = None
    hold = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    exits: dict[str, int] = {}
    ledger: list[dict[str, Any]] = []

    def equity(i: int) -> float:
        if side == 0:
            return cash
        px = close[int(np.clip(i, 0, len(close) - 1))]
        raw = (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)
        return cash * (1.0 + raw * float(exposure))

    def enter(i: int, new_side: int) -> None:
        nonlocal side, entry, entry_equity, cash, hold, long_entries, short_entries, entry_ts
        fill_i = min(i + 1, len(frame) - 1)
        side = int(new_side)
        entry = open_px[fill_i] * (1.0 + float(slip) if side > 0 else 1.0 - float(slip))
        entry_equity = cash
        cash -= cash * float(fee) * float(exposure)
        hold = 0
        entry_ts = frame["timestamp"].iloc[fill_i]
        long_entries += int(side > 0)
        short_entries += int(side < 0)

    def exit_pos(i: int, reason: str, fill_px: float | None = None) -> None:
        nonlocal side, entry, cash, hold, trades, wins, entry_ts
        if fill_px is None:
            fill_i = min(i + 1, len(frame) - 1)
            fill_px = open_px[fill_i] * (1.0 - float(slip) if side > 0 else 1.0 + float(slip))
        before_fee_cash = cash
        raw = (fill_px - entry) / max(entry, 1e-12) if side > 0 else (entry - fill_px) / max(entry, 1e-12)
        cash = cash * (1.0 + raw * float(exposure))
        cash -= before_fee_cash * float(fee) * float(exposure)
        pnl = cash / max(entry_equity, 1e-12) - 1.0
        trades += 1
        wins += int(pnl > 0.0)
        exits[reason] = exits.get(reason, 0) + 1
        ledger.append(
            {
                "entry_timestamp": entry_ts,
                "exit_timestamp": frame["timestamp"].iloc[min(i + 1, len(frame) - 1)],
                "side": "LONG" if side > 0 else "SHORT",
                "reason": reason,
                "entry": float(entry),
                "exit": float(fill_px),
                "hold_bars": int(hold),
                "pnl_pct": float(pnl * 100.0),
                "equity": float(cash),
            }
        )
        side = 0
        entry = 0.0
        hold = 0
        entry_ts = None

    for i in range(len(frame) - 2):
        desired = int(actions[i])
        if side != 0:
            hold += 1
            if side > 0:
                tp_hit = high[i] >= entry * (1.0 + float(tp_pct))
                sl_hit = low[i] <= entry * (1.0 - float(sl_pct))
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 - float(sl_pct)) * (1.0 - float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 + float(tp_pct)) * (1.0 - float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 - float(sl_pct)) * (1.0 - float(slip)))
            else:
                tp_hit = low[i] <= entry * (1.0 - float(tp_pct))
                sl_hit = high[i] >= entry * (1.0 + float(sl_pct))
                if tp_hit and sl_hit:
                    exit_pos(i, "ambiguous_bar_sl_first", entry * (1.0 + float(sl_pct)) * (1.0 + float(slip)))
                elif tp_hit:
                    exit_pos(i, "tp", entry * (1.0 - float(tp_pct)) * (1.0 + float(slip)))
                elif sl_hit:
                    exit_pos(i, "sl", entry * (1.0 + float(sl_pct)) * (1.0 + float(slip)))
        eq = equity(i)
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if side != 0 and hold >= int(max_hold_bars):
            exit_pos(i, "max_hold")
        elif side == 0 and desired != 0:
            enter(i, 1 if desired == 1 else -1)
    if side != 0:
        exit_pos(len(frame) - 2, "end")
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(frame)),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exits": exits,
    }
    return metrics, pd.DataFrame(ledger)


def _evaluate(
    frame: pd.DataFrame,
    decisions: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    tp_pct: float,
    sl_pct: float,
    max_hold_bars: int,
    exposure: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    out: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for mult in (1, 2, 3):
        metrics, ledger = _backtest_barrier(
            frame,
            decisions["action"].to_numpy(dtype=np.int64),
            fee=float(fee) * float(mult),
            slip=float(slip) * float(mult),
            tp_pct=float(tp_pct),
            sl_pct=float(sl_pct),
            max_hold_bars=int(max_hold_bars),
            exposure=float(exposure),
        )
        out[f"cost{mult}"] = metrics
        ledgers[f"cost{mult}"] = ledger
    return out, ledgers


def _score(metrics: dict[str, Any]) -> float:
    c1 = metrics["cost1"]
    c3 = metrics["cost3"]
    trades = int(c3["trades"])
    if trades < 20:
        return -1e6 + float(c3["pnl"])
    calmar = float(c3["pnl"]) / max(abs(float(c3["mdd"])), 1.0)
    density_pen = max(0.0, 1.0 - float(c3["trades_per_day"])) * 5.0 + max(0.0, float(c3["trades_per_day"]) - 8.0) * 1.5
    return float(2.0 * calmar + 0.15 * float(c3["pnl"]) + 0.05 * float(c1["pnl"]) - density_pen)


def _select_threshold(
    model: Pipeline,
    frame: pd.DataFrame,
    feature_cols: list[str],
    grid: list[float],
    args: argparse.Namespace,
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    for threshold in grid:
        dec = _predict_actions(model, frame, feature_cols, threshold)
        metrics, _ = _evaluate(
            frame,
            dec,
            fee=args.fee,
            slip=args.slip,
            tp_pct=args.tp_pct,
            sl_pct=args.sl_pct,
            max_hold_bars=args.max_hold_bars,
            exposure=args.exposure,
        )
        rows.append({"confidence_threshold": float(threshold), "score": _score(metrics), "metrics": metrics})
    best = max(rows, key=lambda row: float(row["score"]))
    return float(best["confidence_threshold"]), rows


def _write_permutation_importance(
    model: Pipeline,
    frame: pd.DataFrame,
    labels: np.ndarray,
    feature_cols: list[str],
    out_path: Path,
    *,
    sample_rows: int,
    seed: int,
) -> list[str]:
    if len(frame) == 0:
        raise ValueError("empty validation frame for permutation importance")
    rng = np.random.default_rng(int(seed))
    n = int(min(max(sample_rows, 1), len(frame)))
    idx = np.sort(rng.choice(np.arange(len(frame)), size=n, replace=False))
    x = frame.iloc[idx][feature_cols]
    y = labels[idx]
    result = permutation_importance(
        model,
        x,
        y,
        scoring="balanced_accuracy",
        n_repeats=3,
        random_state=int(seed),
        n_jobs=1,
    )
    base_pred = model.predict(x)
    rows = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    rows["baseline_balanced_accuracy"] = float(balanced_accuracy_score(y, base_pred))
    rows.to_csv(out_path, index=False)
    return rows["feature"].tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train HGB action+confidence head on active/pass AI-M7-Regime features.")
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--regime3-dir", type=Path, default=DEFAULT_REGIME3_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--feature-mode", choices=("pass", "all_numeric"), default="pass")
    parser.add_argument("--feature-list-file", type=Path, default=None)
    parser.add_argument("--permutation-sample-rows", type=int, default=20000)
    parser.add_argument("--label-score-threshold", type=float, default=0.08)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--confidence-grid", default="0.34,0.38,0.42,0.46,0.50,0.55,0.60,0.65,0.70,0.75")
    parser.add_argument("--max-iter", type=int, default=220)
    parser.add_argument("--learning-rate", type=float, default=0.045)
    parser.add_argument("--max-leaf-nodes", type=int, default=31)
    parser.add_argument("--l2-regularization", type=float, default=0.08)
    parser.add_argument("--min-samples-leaf", type=int, default=40)
    parser.add_argument("--tp-pct", type=float, default=0.018)
    parser.add_argument("--sl-pct", type=float, default=0.010)
    parser.add_argument("--max-hold-bars", type=int, default=48)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slip", type=float, default=0.00015)
    parser.add_argument("--exposure", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=53001)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_2025 = _read_candidates(args.candidate_dir / "trade_candidates_2025_alpha6_current_tail111_exact.csv")
    oos_2026 = _read_candidates(args.candidate_dir / "trade_candidates_2026_alpha6_current_tail111_exact.csv")
    train_2025 = _add_sidecars(train_2025, year=2025, chronos_dir=args.chronos_dir, regime3_dir=args.regime3_dir)
    oos_2026 = _add_sidecars(oos_2026, year=2026, chronos_dir=args.chronos_dir, regime3_dir=args.regime3_dir)

    required = BASE_REQUIRED_FEATURES + CHRONOS_REQUIRED_FEATURES + REGIME3_REQUIRED_FEATURES
    if args.feature_list_file is not None:
        feature_cols = _read_feature_list(args.feature_list_file)
        missing = sorted([col for col in feature_cols if col not in train_2025.columns or col not in oos_2026.columns])
        if missing:
            raise ValueError(f"feature-list contract failed: {missing}")
        optional = []
    elif args.feature_mode == "all_numeric":
        feature_cols = _all_numeric_feature_cols(train_2025, oos_2026)
        optional = []
    else:
        missing = sorted([col for col in required if col not in train_2025.columns or col not in oos_2026.columns])
        if missing:
            raise ValueError(f"required feature contract failed: {missing}")
        optional = [col for col in OPTIONAL_FEATURES if col in train_2025.columns and col in oos_2026.columns]
        feature_cols = required + optional

    split = int(len(train_2025) * (1.0 - float(args.val_fraction)))
    if split <= 1000 or split >= len(train_2025) - 1000:
        raise ValueError(f"invalid validation split: {split} of {len(train_2025)}")
    fit_frame = train_2025.iloc[:split].reset_index(drop=True)
    val_frame = train_2025.iloc[split:].reset_index(drop=True)
    y_fit = _build_labels(fit_frame["tp_sl_action_score"], float(args.label_score_threshold))
    y_full = _build_labels(train_2025["tp_sl_action_score"], float(args.label_score_threshold))

    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=float(args.learning_rate),
                    max_iter=int(args.max_iter),
                    max_leaf_nodes=int(args.max_leaf_nodes),
                    l2_regularization=float(args.l2_regularization),
                    min_samples_leaf=int(args.min_samples_leaf),
                    random_state=int(args.seed),
                ),
            ),
        ]
    )
    fit_weights = compute_sample_weight(class_weight="balanced", y=y_fit)
    model.fit(fit_frame[feature_cols], y_fit, model__sample_weight=fit_weights)
    val_labels = _build_labels(val_frame["tp_sl_action_score"], float(args.label_score_threshold))
    permutation_rank: list[str] = []
    if int(args.permutation_sample_rows) > 0:
        permutation_rank = _write_permutation_importance(
            model,
            val_frame,
            val_labels,
            feature_cols,
            args.out_dir / "permutation_importance_val.csv",
            sample_rows=int(args.permutation_sample_rows),
            seed=int(args.seed),
        )
    grid = [float(x.strip()) for x in str(args.confidence_grid).split(",") if x.strip()]
    best_threshold, val_grid = _select_threshold(model, val_frame, feature_cols, grid, args)

    final_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    loss="log_loss",
                    learning_rate=float(args.learning_rate),
                    max_iter=int(args.max_iter),
                    max_leaf_nodes=int(args.max_leaf_nodes),
                    l2_regularization=float(args.l2_regularization),
                    min_samples_leaf=int(args.min_samples_leaf),
                    random_state=int(args.seed),
                ),
            ),
        ]
    )
    full_weights = compute_sample_weight(class_weight="balanced", y=y_full)
    final_model.fit(train_2025[feature_cols], y_full, model__sample_weight=full_weights)

    val_decisions = _predict_actions(model, val_frame, feature_cols, best_threshold)
    oos_decisions = _predict_actions(final_model, oos_2026, feature_cols, best_threshold)
    val_metrics, val_ledgers = _evaluate(
        val_frame,
        val_decisions,
        fee=args.fee,
        slip=args.slip,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        max_hold_bars=args.max_hold_bars,
        exposure=args.exposure,
    )
    oos_metrics, oos_ledgers = _evaluate(
        oos_2026,
        oos_decisions,
        fee=args.fee,
        slip=args.slip,
        tp_pct=args.tp_pct,
        sl_pct=args.sl_pct,
        max_hold_bars=args.max_hold_bars,
        exposure=args.exposure,
    )

    val_decisions.to_csv(args.out_dir / "val_decisions.csv", index=False)
    oos_decisions.to_csv(args.out_dir / "oos_decisions.csv", index=False)
    for name, ledger in val_ledgers.items():
        ledger.to_csv(args.out_dir / f"val_{name}_ledger.csv", index=False)
    for name, ledger in oos_ledgers.items():
        ledger.to_csv(args.out_dir / f"oos_{name}_ledger.csv", index=False)
    joblib.dump({"model": final_model, "feature_cols": feature_cols, "confidence_threshold": best_threshold}, args.out_dir / "model.joblib")

    summary = {
        "model_id": MODEL_ID,
        "feature_cols": feature_cols,
        "feature_mode": str(args.feature_mode),
        "feature_count": int(len(feature_cols)),
        "required_features": required,
        "optional_features_used": optional,
        "permutation_top20": permutation_rank[:20],
        "label_score_threshold": float(args.label_score_threshold),
        "label_counts_fit": {str(k): int(v) for k, v in zip(*np.unique(y_fit, return_counts=True))},
        "label_counts_full_2025": {str(k): int(v) for k, v in zip(*np.unique(y_full, return_counts=True))},
        "confidence_threshold": float(best_threshold),
        "val_grid": val_grid,
        "validation": val_metrics,
        "oos_2026": oos_metrics,
        "artifacts": {
            "out_dir": str(args.out_dir),
            "model": str(args.out_dir / "model.joblib"),
            "val_decisions": str(args.out_dir / "val_decisions.csv"),
            "oos_decisions": str(args.out_dir / "oos_decisions.csv"),
        },
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_default))
    print(json.dumps(summary, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
