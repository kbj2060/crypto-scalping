#!/usr/bin/env python3
"""Train and fresh-forward test the independent BTC v2 regime trend-scan model.

The parent is trained only on BTC OHLCV-derived 1-hour features. A parent row
is available one hour after its left-labelled bucket starts. Entry is allowed
only on that new-signal event and only when the 2024-fit BTC HMM agrees with
the side. Positions are evaluated on every causal 5-minute bar.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "btc_v2_regime_trendscan_hgb_20260714"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
HOURLY_DIR = ROOT / "tmp/causal_regen_20260516/sigma9_1h_btc_20260706"
FIVE_MINUTE_FILES = tuple(ROOT / f"data/splits/year_oos/btc_features_{year}.csv" for year in (2025, 2026))
REGIME_DIR = ROOT / "data/ensemble/supervised/btc_regime3_current_hmm_sensitive_wide24_20260708"
REGIME_PREFIX = "regime3_current_sensitive_wide24_"

TRAIN_END = pd.Timestamp("2025-06-30 23:59:59")
VALIDATION_START = pd.Timestamp("2025-07-01")
VALIDATION_MID = pd.Timestamp("2025-10-01")
OOS_START = pd.Timestamp("2026-01-01")
SEEDS = (270705, 270710, 270715, 270720, 270725)

NON_FEATURE_COLUMNS = {"timestamp", "open", "high", "low", "close", "ts_action", "ts_t_value", "ts_opt_L"}
QUALITY_GRID = (0.50, 0.55, 0.60, 0.65, 0.70)
REGIME_GRID = (0.25, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50)

MARGIN_FRACTION = 0.30
LEVERAGE = 2.0
NOTIONAL = MARGIN_FRACTION * LEVERAGE
FEE_RATE = 0.0005
SLIP_RATE = 0.0002
MAKER_FEE_MULT = 0.20
LEGACY_STOP_ATR_ACCOUNT = 1.5
LEGACY_TRAIL_ATR_ACCOUNT = 5.0
LEGACY_ARM_ATR_ACCOUNT = 2.0
STOP_ATR_PRICE = LEGACY_STOP_ATR_ACCOUNT / NOTIONAL
TRAIL_ATR_PRICE = LEGACY_TRAIL_ATR_ACCOUNT / NOTIONAL
ARM_ATR_PRICE = LEGACY_ARM_ATR_ACCOUNT / NOTIONAL
MAX_HOLD_BARS = 144 * 12
COOLDOWN_BARS = 3 * 12


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _read_hourly() -> tuple[pd.DataFrame, list[str]]:
    frames: list[pd.DataFrame] = []
    expected_columns: list[str] | None = None
    for year in (2024, 2025, 2026):
        path = HOURLY_DIR / f"sigma9_btc_1h_{year}.parquet"
        if not path.exists():
            raise FileNotFoundError(path)
        current = pd.read_parquet(path)
        current["timestamp"] = pd.to_datetime(current["timestamp"], errors="raise")
        if expected_columns is None:
            expected_columns = list(current.columns)
        elif list(current.columns) != expected_columns:
            raise RuntimeError(f"hourly BTC feature contract differs: {path}")
        frames.append(current)
    hourly = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    if hourly["timestamp"].duplicated().any():
        raise RuntimeError("duplicate hourly BTC timestamps")
    feature_columns = [column for column in hourly.columns if column not in NON_FEATURE_COLUMNS]
    if len(feature_columns) != 28:
        raise RuntimeError(f"expected 28 independent BTC features, got {len(feature_columns)}")
    forbidden = [column for column in feature_columns if any(token in column.lower() for token in ("target", "future", "label", "pnl"))]
    if forbidden:
        raise RuntimeError(f"forbidden hourly BTC features: {forbidden}")
    return hourly, feature_columns


def _fit_parent(
    hourly: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[HistGradientBoostingClassifier], pd.DataFrame, dict[str, Any]]:
    train_mask = hourly["timestamp"].le(TRAIN_END).to_numpy()
    x_train = hourly.loc[train_mask, feature_columns].to_numpy(dtype=np.float64)
    y_train = hourly.loc[train_mask, "ts_action"].to_numpy(dtype=np.int64)
    sample_weight = np.clip(np.abs(hourly.loc[train_mask, "ts_t_value"].to_numpy(dtype=np.float64)), 0.5, 12.0)
    x_all = hourly[feature_columns].to_numpy(dtype=np.float64)
    probability_sum = np.zeros((len(hourly), 3), dtype=np.float64)
    models: list[HistGradientBoostingClassifier] = []

    for seed in SEEDS:
        model = HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.03,
            max_iter=250,
            max_depth=4,
            l2_regularization=1.0,
            max_leaf_nodes=31,
            min_samples_leaf=80,
            early_stopping=False,
            random_state=int(seed),
            class_weight="balanced",
        )
        model.fit(x_train, y_train, sample_weight=sample_weight)
        raw = model.predict_proba(x_all)
        class_index = {int(value): index for index, value in enumerate(model.classes_)}
        probability = np.zeros((len(hourly), 3), dtype=np.float64)
        for action in (0, 1, 2):
            if action in class_index:
                probability[:, action] = raw[:, class_index[action]]
        probability_sum += probability
        models.append(model)

    probability = probability_sum / len(models)
    action = probability.argmax(axis=1).astype(np.int8)
    quality = probability[np.arange(len(hourly)), action]
    signal = pd.DataFrame(
        {
            "source_timestamp": hourly["timestamp"],
            "available_timestamp": hourly["timestamp"] + pd.Timedelta(hours=1),
            "parent_p_cash": probability[:, 0],
            "parent_p_long": probability[:, 1],
            "parent_p_short": probability[:, 2],
            "parent_action": action,
            "parent_quality": quality,
            "parent_atr_pct": pd.to_numeric(hourly["atr_pct"], errors="raise").to_numpy(dtype=np.float64),
        }
    )
    counts = pd.Series(y_train).value_counts().sort_index()
    report = {
        "train_start": hourly.loc[train_mask, "timestamp"].iloc[0],
        "train_end": hourly.loc[train_mask, "timestamp"].iloc[-1],
        "train_rows": int(train_mask.sum()),
        "label_counts": {str(int(key)): int(value) for key, value in counts.items()},
        "feature_count": len(feature_columns),
        "seeds": list(SEEDS),
    }
    return models, signal, report


def _read_five_minute() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in FIVE_MINUTE_FILES:
        if not path.exists():
            raise FileNotFoundError(path)
        current = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
        frames.append(current)
    frame = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    delta = frame["timestamp"].diff().dropna()
    if not bool(delta.eq(pd.Timedelta(minutes=5)).all()):
        raise RuntimeError("BTC execution tape is not continuous at five-minute frequency")

    regime_frames: list[pd.DataFrame] = []
    regime_columns = [
        f"{REGIME_PREFIX}bull_prob",
        f"{REGIME_PREFIX}bear_prob",
        f"{REGIME_PREFIX}chop_prob",
        f"{REGIME_PREFIX}confidence",
        f"{REGIME_PREFIX}margin",
    ]
    for year in (2025, 2026):
        path = REGIME_DIR / f"btc_features_{year}_regime3_current_sensitive_hmm_wide24.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        regime_frames.append(pd.read_csv(path, usecols=["timestamp", *regime_columns], parse_dates=["timestamp"]))
    regime = pd.concat(regime_frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    frame = frame.merge(regime, on="timestamp", how="left", validate="one_to_one")
    if frame[regime_columns].isna().any().any():
        raise RuntimeError("BTC execution tape has missing causal HMM probabilities")
    return frame


def _merge_signal(frame: pd.DataFrame, signal: pd.DataFrame) -> pd.DataFrame:
    if signal["available_timestamp"].iloc[-1] < frame["timestamp"].iloc[-1].floor("h"):
        raise RuntimeError(
            "hourly BTC parent is stale; rerun build_1h_trendscan_dataset_btc_20260706.py before this evaluation"
        )
    merged = pd.merge_asof(
        frame.sort_values("timestamp"),
        signal.sort_values("available_timestamp"),
        left_on="timestamp",
        right_on="available_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    required = ["available_timestamp", "parent_action", "parent_quality", "parent_atr_pct"]
    if merged[required].isna().any().any():
        bad = merged.loc[merged[required].isna().any(axis=1), "timestamp"]
        raise RuntimeError(f"missing available BTC parent signal: {bad.head(10).tolist()}")
    merged["is_new_parent_signal"] = merged["available_timestamp"].ne(merged["available_timestamp"].shift(1))
    return merged.reset_index(drop=True)


def _candidate_side(frame: pd.DataFrame, *, quality_threshold: float, regime_threshold: float | None) -> np.ndarray:
    action = pd.to_numeric(frame["parent_action"], errors="raise").to_numpy(dtype=np.int8)
    quality = pd.to_numeric(frame["parent_quality"], errors="raise").to_numpy(dtype=np.float64)
    bull = pd.to_numeric(frame[f"{REGIME_PREFIX}bull_prob"], errors="raise").to_numpy(dtype=np.float64)
    bear = pd.to_numeric(frame[f"{REGIME_PREFIX}bear_prob"], errors="raise").to_numpy(dtype=np.float64)
    event = frame["is_new_parent_signal"].to_numpy(dtype=bool)
    side = np.zeros(len(frame), dtype=np.int8)
    long_ok = event & (action == 1) & (quality >= quality_threshold)
    short_ok = event & (action == 2) & (quality >= quality_threshold)
    if regime_threshold is not None:
        long_ok &= bull >= regime_threshold
        short_ok &= bear >= regime_threshold
    side[long_ok] = 1
    side[short_ok] = -1
    return side


def _exit_fill(arrays: dict[str, np.ndarray], signal_i: int, side: int) -> tuple[int, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(arrays["open"]) - 1)
    limit_price = float(arrays["open"][fill_i])
    touched = bool(arrays["high"][fill_i] >= limit_price) if side > 0 else bool(arrays["low"][fill_i] <= limit_price)
    if touched:
        return fill_i, limit_price, FEE_RATE * MAKER_FEE_MULT, "maker_limit"
    close = float(arrays["close"][fill_i])
    price = close * (1.0 - SLIP_RATE) if side > 0 else close * (1.0 + SLIP_RATE)
    return fill_i, price, FEE_RATE, "market_fallback"


def _fresh_forward_replay(frame: pd.DataFrame, candidate_side: np.ndarray) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray]:
    if len(frame) != len(candidate_side):
        raise RuntimeError("candidate side and execution tape length mismatch")
    arrays = {column: pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=np.float64) for column in ("open", "high", "low", "close")}
    atr = pd.to_numeric(frame["parent_atr_pct"], errors="raise").to_numpy(dtype=np.float64)
    cash = 1.0
    peak_equity = 1.0
    mdd = 0.0
    position = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_signal_i = -1
    entry_fill_i = -1
    entry_atr = 0.0
    peak_move = 0.0
    cooldown_until = -1
    rows: list[dict[str, Any]] = []
    equity_curve = np.ones(len(frame), dtype=np.float64)

    for row_i in range(len(frame) - 1):
        if position != 0:
            close = float(arrays["close"][row_i])
            move = (close * (1.0 - SLIP_RATE) - entry_price) / entry_price if position > 0 else (entry_price - close * (1.0 + SLIP_RATE)) / entry_price
            equity = cash * (1.0 + move * NOTIONAL)
            peak_move = max(peak_move, move)
        else:
            move = 0.0
            equity = cash
        equity_curve[row_i] = equity
        peak_equity = max(peak_equity, equity)
        mdd = min(mdd, equity / max(peak_equity, 1.0e-12) - 1.0)

        if position != 0:
            hold_bars = row_i - entry_fill_i
            reason = ""
            if move <= -STOP_ATR_PRICE * entry_atr:
                reason = "stop_loss"
            elif peak_move >= ARM_ATR_PRICE * entry_atr and peak_move - move >= TRAIL_ATR_PRICE * entry_atr:
                reason = "trailing_exit"
            elif hold_bars >= MAX_HOLD_BARS:
                reason = "time_exit"
            if reason:
                fill_i, exit_price, exit_fee, route = _exit_fill(arrays, row_i, position)
                raw_return = (exit_price - entry_price) / entry_price if position > 0 else (entry_price - exit_price) / entry_price
                before = cash
                cash = cash * (1.0 + raw_return * NOTIONAL)
                cash -= before * exit_fee * NOTIONAL
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
                rows.append(
                    {
                        "entry_signal_i": entry_signal_i,
                        "entry_fill_i": entry_fill_i,
                        "exit_signal_i": row_i,
                        "exit_fill_i": fill_i,
                        "entry_timestamp": frame["timestamp"].iloc[entry_signal_i],
                        "entry_fill_timestamp": frame["timestamp"].iloc[entry_fill_i],
                        "exit_timestamp": frame["timestamp"].iloc[row_i],
                        "exit_fill_timestamp": frame["timestamp"].iloc[fill_i],
                        "parent_source_timestamp": frame["source_timestamp"].iloc[entry_signal_i],
                        "side": position,
                        "reason": reason,
                        "route": route,
                        "hold_bars": hold_bars,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "entry_atr_price_move": entry_atr,
                        "peak_price_move": peak_move,
                        "raw_return": raw_return,
                        "trade_return": trade_return,
                        "win": int(trade_return > 0.0),
                        "margin_fraction": MARGIN_FRACTION,
                        "leverage": LEVERAGE,
                        "notional": NOTIONAL,
                    }
                )
                position = 0
                cooldown_until = row_i + COOLDOWN_BARS
                equity_curve[fill_i] = cash
                continue

        if position != 0 or row_i < cooldown_until or int(candidate_side[row_i]) == 0:
            continue
        side = int(candidate_side[row_i])
        fill_i = row_i + 1
        entry_price_candidate = float(arrays["open"][fill_i])
        touched = bool(arrays["low"][fill_i] <= entry_price_candidate) if side > 0 else bool(arrays["high"][fill_i] >= entry_price_candidate)
        if not touched:
            continue
        position = side
        entry_price = entry_price_candidate
        entry_equity = cash
        entry_signal_i = row_i
        entry_fill_i = fill_i
        entry_atr = max(float(atr[row_i]), 1.0e-6)
        peak_move = 0.0
        cash -= cash * FEE_RATE * MAKER_FEE_MULT * NOTIONAL

    if position != 0:
        fill_i = len(frame) - 1
        close = float(arrays["close"][fill_i])
        exit_price = close * (1.0 - SLIP_RATE) if position > 0 else close * (1.0 + SLIP_RATE)
        raw_return = (exit_price - entry_price) / entry_price if position > 0 else (entry_price - exit_price) / entry_price
        before = cash
        cash = cash * (1.0 + raw_return * NOTIONAL)
        cash -= before * FEE_RATE * NOTIONAL
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        rows.append(
            {
                "entry_signal_i": entry_signal_i,
                "entry_fill_i": entry_fill_i,
                "exit_signal_i": fill_i,
                "exit_fill_i": fill_i,
                "entry_timestamp": frame["timestamp"].iloc[entry_signal_i],
                "entry_fill_timestamp": frame["timestamp"].iloc[entry_fill_i],
                "exit_timestamp": frame["timestamp"].iloc[fill_i],
                "exit_fill_timestamp": frame["timestamp"].iloc[fill_i],
                "parent_source_timestamp": frame["source_timestamp"].iloc[entry_signal_i],
                "side": position,
                "reason": "forced_end",
                "route": "market_end",
                "hold_bars": fill_i - entry_fill_i,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "entry_atr_price_move": entry_atr,
                "peak_price_move": peak_move,
                "raw_return": raw_return,
                "trade_return": trade_return,
                "win": int(trade_return > 0.0),
                "margin_fraction": MARGIN_FRACTION,
                "leverage": LEVERAGE,
                "notional": NOTIONAL,
            }
        )
    equity_curve[-1] = cash
    peak_curve = np.maximum.accumulate(equity_curve)
    mdd = min(mdd, float(np.min(equity_curve / np.maximum(peak_curve, 1.0e-12) - 1.0)))
    ledger = pd.DataFrame(rows)
    reasons = ledger["reason"].value_counts().to_dict() if len(ledger) else {}
    wins = int(ledger["win"].sum()) if len(ledger) else 0
    duration_days = max((frame["timestamp"].iloc[-1] - frame["timestamp"].iloc[0]).total_seconds() / 86400.0, 1.0e-9)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)) if len(ledger) else 0.0,
        "trades_per_day": float(len(ledger) / duration_days),
        "long_entries": int((ledger["side"] > 0).sum()) if len(ledger) else 0,
        "short_entries": int((ledger["side"] < 0).sum()) if len(ledger) else 0,
        "avg_hold_bars": float(ledger["hold_bars"].mean()) if len(ledger) else 0.0,
        "median_hold_bars": float(ledger["hold_bars"].median()) if len(ledger) else 0.0,
        "exit_reasons": {str(key): int(value) for key, value in reasons.items()},
        "margin_fraction": MARGIN_FRACTION,
        "leverage": LEVERAGE,
        "notional": NOTIONAL,
    }
    return metrics, ledger, equity_curve


def _period(frame: pd.DataFrame, side: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, pd.DataFrame]:
    mask = frame["timestamp"].ge(start).to_numpy() & frame["timestamp"].le(end).to_numpy()
    current = frame.loc[mask].reset_index(drop=True)
    metrics, ledger, equity = _fresh_forward_replay(current, side[mask])
    return metrics, ledger, equity, current


def _select_policy(frame: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    validation_end = OOS_START - pd.Timedelta(minutes=5)
    rows: list[dict[str, Any]] = []
    for quality_threshold in QUALITY_GRID:
        for regime_threshold in (*REGIME_GRID, None):
            side = _candidate_side(frame, quality_threshold=quality_threshold, regime_threshold=regime_threshold)
            first, _, _, _ = _period(frame, side, VALIDATION_START, VALIDATION_MID - pd.Timedelta(minutes=5))
            second, _, _, _ = _period(frame, side, VALIDATION_MID, validation_end)
            full, _, _, _ = _period(frame, side, VALIDATION_START, validation_end)
            eligible = (
                first["pnl"] > 0.0
                and second["pnl"] > 0.0
                and full["mdd"] >= -15.0
                and full["trades"] >= 30
            )
            rows.append(
                {
                    "quality_threshold": quality_threshold,
                    "regime_mode": "trend_agree" if regime_threshold is not None else "off",
                    "regime_threshold": regime_threshold,
                    "first_pnl": first["pnl"],
                    "first_mdd": first["mdd"],
                    "first_trades": first["trades"],
                    "second_pnl": second["pnl"],
                    "second_mdd": second["mdd"],
                    "second_trades": second["trades"],
                    "full_pnl": full["pnl"],
                    "full_mdd": full["mdd"],
                    "full_trades": full["trades"],
                    "full_wr": full["wr"],
                    "eligible": bool(eligible),
                    "selection_score": float(min(first["pnl"] + 0.5 * first["mdd"], second["pnl"] + 0.5 * second["mdd"])),
                }
            )
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["eligible"]]
    if len(eligible) == 0:
        raise RuntimeError("no BTC v2 policy passed both validation halves")
    selected = eligible.sort_values(["selection_score", "full_pnl", "full_mdd"], ascending=False).iloc[0].to_dict()
    selected["selection_rule"] = "maximize worst-half (pnl + 0.5*mdd), with both halves pnl>0, full mdd>=-15, trades>=30"
    return selected, grid


def _write_chart(frame: pd.DataFrame, ledger: pd.DataFrame, equity: np.ndarray, output: Path) -> None:
    timestamp = pd.to_datetime(frame["timestamp"])
    price = pd.to_numeric(frame["close"], errors="raise")
    figure, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True, gridspec_kw={"height_ratios": [2.0, 1.0]})
    axes[0].plot(timestamp, price, color="#20262e", linewidth=0.8, label="BTC close")
    if len(ledger):
        for side_value, marker, color, label in ((1, "^", "#168f5b", "Long"), (-1, "v", "#d04a3a", "Short")):
            current = ledger.loc[ledger["side"] == side_value]
            index = current["entry_fill_i"].to_numpy(dtype=np.int64)
            axes[0].scatter(timestamp.iloc[index], price.iloc[index], marker=marker, color=color, s=32, label=label, zorder=3)
    axes[0].set_ylabel("BTC price (USDT)")
    axes[0].legend(loc="upper left", ncol=3)
    axes[0].grid(alpha=0.18)
    axes[1].plot(timestamp, (equity - 1.0) * 100.0, color="#2369a2", linewidth=1.1)
    axes[1].axhline(0.0, color="#777777", linewidth=0.7)
    axes[1].set_ylabel("Compound PnL (%)")
    axes[1].set_xlabel("Timestamp (UTC)")
    axes[1].grid(alpha=0.18)
    figure.suptitle("BTC v2 regime trend-scan | 5-minute fresh-forward OOS")
    figure.tight_layout()
    figure.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(figure)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print("stage=load_hourly_btc_features", flush=True)
    hourly, feature_columns = _read_hourly()
    print("stage=fit_independent_btc_parent", flush=True)
    models, signal, parent_report = _fit_parent(hourly, feature_columns)
    print("stage=merge_causal_5m_regime_and_parent", flush=True)
    execution = _merge_signal(_read_five_minute(), signal)
    print("stage=validation_policy_selection", flush=True)
    selected, grid = _select_policy(execution)
    quality_threshold = float(selected["quality_threshold"])
    regime_threshold = float(selected["regime_threshold"]) if pd.notna(selected["regime_threshold"]) else None
    side = _candidate_side(execution, quality_threshold=quality_threshold, regime_threshold=regime_threshold)

    validation_end = OOS_START - pd.Timedelta(minutes=5)
    validation_metrics, validation_ledger, validation_equity, validation_frame = _period(execution, side, VALIDATION_START, validation_end)
    comparison_metrics, _, _, _ = _period(execution, side, VALIDATION_MID, validation_end)
    print("stage=fresh_forward_oos_once", flush=True)
    oos_end = execution["timestamp"].iloc[-1]
    oos_metrics, oos_ledger, oos_equity, oos_frame = _period(execution, side, OOS_START, oos_end)
    q1_metrics, _, _, _ = _period(execution, side, OOS_START, pd.Timestamp("2026-03-31 23:55"))

    bundle_path = args.out_dir / "btc_v2_regime_trendscan_bundle.joblib"
    grid_path = args.out_dir / "validation_policy_grid.csv"
    prediction_path = args.out_dir / "validation_oos_predictions.csv"
    validation_ledger_path = args.out_dir / "validation_ledger.csv"
    oos_ledger_path = args.out_dir / "oos_ledger.csv"
    chart_path = args.out_dir / "oos_equity_chart.png"
    report_path = args.out_dir / "report.json"

    joblib.dump(
        {
            "model_id": MODEL_ID,
            "models": models,
            "feature_columns": feature_columns,
            "seeds": list(SEEDS),
            "policy": {
                "quality_threshold": quality_threshold,
                "regime_mode": str(selected["regime_mode"]),
                "regime_threshold": regime_threshold,
            },
            "execution": {
                "decision_interval_minutes": 5,
                "parent_interval_minutes": 60,
                "margin_fraction": MARGIN_FRACTION,
                "leverage": LEVERAGE,
                "notional": NOTIONAL,
                "stop_atr_price": STOP_ATR_PRICE,
                "trail_atr_price": TRAIL_ATR_PRICE,
                "arm_atr_price": ARM_ATR_PRICE,
                "max_hold_bars": MAX_HOLD_BARS,
                "cooldown_bars": COOLDOWN_BARS,
            },
        },
        bundle_path,
    )
    grid.to_csv(grid_path, index=False)
    validation_ledger.to_csv(validation_ledger_path, index=False)
    oos_ledger.to_csv(oos_ledger_path, index=False)
    execution.loc[execution["timestamp"].ge(VALIDATION_START), [
        "timestamp",
        "source_timestamp",
        "available_timestamp",
        "parent_p_cash",
        "parent_p_long",
        "parent_p_short",
        "parent_action",
        "parent_quality",
        "parent_atr_pct",
        f"{REGIME_PREFIX}bull_prob",
        f"{REGIME_PREFIX}bear_prob",
        f"{REGIME_PREFIX}chop_prob",
        "is_new_parent_signal",
    ]].assign(candidate_side=side[execution["timestamp"].ge(VALIDATION_START).to_numpy()]).to_csv(prediction_path, index=False)
    _write_chart(oos_frame, oos_ledger, oos_equity, chart_path)

    validation_beats_v1 = comparison_metrics["pnl"] > 6.69 and comparison_metrics["mdd"] >= -12.11
    oos_beats_v1 = oos_metrics["pnl"] > 10.52 and oos_metrics["mdd"] >= -16.46
    report = {
        "model_id": MODEL_ID,
        "status": "research_negative_result_not_adopted",
        "live_changed": False,
        "design": {
            "parent": "five-seed HistGradientBoosting trend-scan classifier on 28 BTC-only 1h OHLCV features",
            "direction_gate": "2024-fit BTC HMM must agree with parent side at the new parent-signal event",
            "event_sampling": "one entry decision per newly available 1h parent signal; open positions evaluated every 5 minutes",
            "paper_inspirations": {
                "TabM_ensemble_principle": "https://arxiv.org/abs/2410.24210",
                "PatchTST_multiscale_patch_principle": "https://arxiv.org/abs/2211.14730",
                "FOIL_temporal_environment_invariance": "https://arxiv.org/abs/2406.09130",
            },
        },
        "parent_training": parent_report,
        "feature_contract": {
            "feature_count": len(feature_columns),
            "feature_columns": feature_columns,
            "btc_only": True,
            "funding_oi_or_eth_reference_used": False,
        },
        "causal_contract": {
            "hourly_bucket": "left-labelled [t,t+1h)",
            "parent_available_timestamp": "source_timestamp + 1h",
            "entry_fill": "next 5-minute bar maker limit at open",
            "exit_check_interval_minutes": 5,
            "entry_only_on_new_parent_signal": True,
            "hmm_fit_period": "2024",
            "warmup_performed": True,
        },
        "risk_contract": {
            "margin_fraction": MARGIN_FRACTION,
            "leverage": LEVERAGE,
            "notional": NOTIONAL,
            "notional_formula": "margin_fraction * leverage",
            "stop_atr_price_move": STOP_ATR_PRICE,
            "trail_atr_price_move": TRAIL_ATR_PRICE,
            "arm_atr_price_move": ARM_ATR_PRICE,
            "price_lines_not_multiplied_by_leverage": True,
            "max_hold_bars_5m": MAX_HOLD_BARS,
            "cooldown_bars_5m": COOLDOWN_BARS,
        },
        "splits": {
            "train": {"start": parent_report["train_start"], "end": parent_report["train_end"]},
            "validation_selection": {"start": VALIDATION_START, "end": validation_end, "halves": ["2025-07-01..2025-09-30", "2025-10-01..2025-12-31"]},
            "oos": {"start": OOS_START, "end": oos_end},
            "default_project_boundary_deviation": "validation starts 2025-07-01 to obtain two pre-OOS stability halves; v1 comparison remains 2025-10-01..2025-12-31",
        },
        "policy_selection": {
            "oos_used_for_selection": False,
            "selected": selected,
            "grid_rows": int(len(grid)),
        },
        "metrics": {
            "validation_full_2025_h2": validation_metrics,
            "validation_v1_comparison_2025_q4": comparison_metrics,
            "oos_extended_2026_to_2026_07_12": oos_metrics,
            "oos_frozen_q1_2026": q1_metrics,
        },
        "v1_reference": {
            "validation_q4": {"pnl": 6.69, "mdd": -12.11},
            "oos_extended": {"pnl": 10.52, "mdd": -16.46, "trades": 31, "wr": 0.355},
            "validation_beats_v1_pnl_and_mdd": bool(validation_beats_v1),
            "oos_beats_v1_pnl_and_mdd": bool(oos_beats_v1),
            "comparison_caveat": "v1 uses dynamic sidecar sizing and L0.5/S2.5 scale; v2 uses fixed margin 0.30, leverage 2, notional 0.60",
        },
        "promotion": {
            "promotion_ready": False,
            "reason": "failed OOS: negative PnL and worse MDD than BTC v1; live adapter and promotion audit must not proceed",
            "live_model_remains": "BTC v1",
        },
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "artifacts": {
            "bundle": bundle_path,
            "validation_policy_grid": grid_path,
            "predictions": prediction_path,
            "validation_ledger": validation_ledger_path,
            "oos_ledger": oos_ledger_path,
            "oos_chart": chart_path,
        },
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": report_path, "selected": selected, "metrics": report["metrics"], "promotion": report["promotion"]}, default=_json_default, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
