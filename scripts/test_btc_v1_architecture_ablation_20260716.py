#!/usr/bin/env python3
"""Causal BTC v1 architecture ablation on the stabilized parent signal.

Research-only sequence:
0. saved learned-exit replay (reference only);
1. remove learned exit: ATR(192) TP/SL plus a 72-bar maximum hold;
2. replace exposure with fixed conservative margin/leverage;
3. admit entries only when the parent action changes into LONG/SHORT;
4. replace the parent with a separate zigzag Direction model;
5. add a separately trained purged-OOF Meta take/skip model.

Stages 1-3 hold the stabilized parent signal fixed.  Stages 4-5 are a clean
Direction/Meta research architecture and therefore intentionally use a new
signal source.  All entries occur on the next 5-minute bar.  The replay reads
no saved trade ledger or saved parent exit timestamp.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import eval_btc_v1_label_family_suite_20260715 as label_suite  # noqa: E402


PARENT_DIR = ROOT / "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48stable_c6d12_fullstack_fulltrain_fullexit_20260716"
REFERENCE_REPORT = ROOT / "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48stable_c6d12_fullstack_fulltrain_fullexit_q055_q1fresh_20260716/report.json"
ZIGZAG_DIR = ROOT / "tmp/causal_regen_20260516/btc_zigzag_action_labels_20260708"
FIVE_MINUTE_DIR = ROOT / "data/splits/year_oos"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v1_architecture_ablation_20260716"

TRAIN_END = pd.Timestamp("2025-09-30 23:55:00")
VALIDATION_START = pd.Timestamp("2025-10-01 00:00:00")
VALIDATION_END = pd.Timestamp("2025-12-31 23:55:00")
OOS_START = pd.Timestamp("2026-01-01 00:00:00")
OOS_END = pd.Timestamp("2026-03-31 23:55:00")
DIRECTION_META_EMBARGO_HOURS = 72
DIRECTION_META_TRAIN_END = VALIDATION_START - pd.Timedelta(hours=DIRECTION_META_EMBARGO_HOURS)

ATR_WINDOW = 192
TP_ATR_MULTIPLE = 12.0
SL_ATR_MULTIPLE = 6.0
MIN_TP = 0.075
MIN_SL = 0.040
MAX_TP = 0.220
MAX_SL = 0.120
MAX_HOLD_BARS = 72
ROUND_TRIP_COST = 0.0014
RECOMMENDED_TP_ATR_MULTIPLE = 8.0
RECOMMENDED_SL_ATR_MULTIPLE = 4.0
RECOMMENDED_MIN_TP = 0.008
RECOMMENDED_MIN_SL = 0.005
RECOMMENDED_MAX_TP = 0.030
RECOMMENDED_MAX_SL = 0.015
CURRENT_FIXED_NOTIONAL = 0.40
CONSERVATIVE_MARGIN_FRACTION = 0.15
FIXED_LEVERAGE = 2.0
CONSERVATIVE_NOTIONAL = CONSERVATIVE_MARGIN_FRACTION * FIXED_LEVERAGE
META_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
REGRESSION_RANK_QUANTILES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90)
SEEDS = (310713, 310719, 310727)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _read_five_minute() -> pd.DataFrame:
    frames = []
    for year in (2025, 2026):
        path = FIVE_MINUTE_DIR / f"btc_features_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=["timestamp", "open", "high", "low", "close"], parse_dates=["timestamp"])
        frames.append(frame)
    out = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    out = out.loc[out["timestamp"] <= OOS_END + pd.Timedelta(hours=12)].reset_index(drop=True)
    delta = out["timestamp"].diff().dropna()
    if not delta.eq(pd.Timedelta(minutes=5)).all():
        raise RuntimeError("BTC 5-minute execution tape is not continuous")
    previous_close = out["close"].shift(1)
    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - previous_close).abs(),
            (out["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    out["atr_pct"] = tr.rolling(ATR_WINDOW, min_periods=ATR_WINDOW).mean() / out["close"]
    if not np.isfinite(out.loc[out["timestamp"] >= VALIDATION_START, "atr_pct"]).all():
        raise RuntimeError("ATR is not finite in evaluation range")
    return out


def _action_column(frame: pd.DataFrame) -> str:
    matches = [column for column in frame.columns if column.endswith("_final_action")]
    if len(matches) != 1:
        raise RuntimeError(f"expected one final action column, got {matches}")
    return matches[0]


def _read_parent_signal(tape: pd.DataFrame) -> np.ndarray:
    parts = []
    for filename in ("validation_predictions_q055.csv", "oos_predictions_q055.csv"):
        path = PARENT_DIR / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, parse_dates=["timestamp"])
        action_col = _action_column(frame)
        parts.append(frame[["timestamp", action_col]].rename(columns={action_col: "action"}))
    prediction = pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp")
    joined = tape[["timestamp"]].merge(prediction, on="timestamp", how="left", validate="one_to_one")
    return joined["action"].fillna(0).to_numpy(dtype=np.int8)


def _event_only(action: np.ndarray) -> np.ndarray:
    previous = np.r_[0, action[:-1]]
    return np.where((action != 0) & (action != previous), action, 0).astype(np.int8)


def _replay(
    tape: pd.DataFrame,
    signal: np.ndarray,
    start: pd.Timestamp,
    end: pd.Timestamp,
    notional: float,
    tp_atr_multiple: float = TP_ATR_MULTIPLE,
    sl_atr_multiple: float = SL_ATR_MULTIPLE,
    min_tp: float = MIN_TP,
    min_sl: float = MIN_SL,
    max_tp: float = MAX_TP,
    max_sl: float = MAX_SL,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    timestamps = tape["timestamp"].to_numpy()
    mask = tape["timestamp"].between(start, end).to_numpy()
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise RuntimeError("empty replay interval")
    split_start, split_end = int(indices[0]), int(indices[-1])
    open_price = tape["open"].to_numpy(dtype=np.float64)
    high_price = tape["high"].to_numpy(dtype=np.float64)
    low_price = tape["low"].to_numpy(dtype=np.float64)
    close_price = tape["close"].to_numpy(dtype=np.float64)
    atr_pct = tape["atr_pct"].to_numpy(dtype=np.float64)
    equity = 1.0
    peak = 1.0
    mdd = 0.0
    busy_until = split_start - 1
    rows: list[dict[str, Any]] = []
    curve_rows = [{"timestamp": tape["timestamp"].iloc[split_start], "equity": equity}]
    for signal_i in range(split_start, split_end + 1):
        if signal_i <= busy_until or int(signal[signal_i]) == 0 or signal_i + 1 > split_end:
            continue
        entry_i = signal_i + 1
        maximum_exit = min(entry_i + MAX_HOLD_BARS, split_end)
        if maximum_exit <= entry_i:
            continue
        side = 1 if int(signal[signal_i]) == 1 else -1
        entry = float(open_price[entry_i])
        tp_move = float(np.clip(tp_atr_multiple * atr_pct[signal_i], min_tp, max_tp))
        sl_move = float(np.clip(sl_atr_multiple * atr_pct[signal_i], min_sl, max_sl))
        if side > 0:
            tp_price, sl_price = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        else:
            tp_price, sl_price = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
        exit_i = maximum_exit
        exit_price = float(close_price[exit_i])
        exit_reason = "max_hold_72"
        for bar_i in range(entry_i, maximum_exit + 1):
            if side > 0:
                stop_hit = low_price[bar_i] <= sl_price
                target_hit = high_price[bar_i] >= tp_price
            else:
                stop_hit = high_price[bar_i] >= sl_price
                target_hit = low_price[bar_i] <= tp_price
            if stop_hit:
                exit_i, exit_price, exit_reason = bar_i, float(sl_price), "stop_loss"
                break
            if target_hit:
                exit_i, exit_price, exit_reason = bar_i, float(tp_price), "take_profit"
                break
        raw_return = side * (exit_price / entry - 1.0)
        account_return = float(notional * (raw_return - ROUND_TRIP_COST))
        equity *= max(1.0 + account_return, 1e-9)
        peak = max(peak, equity)
        mdd = min(mdd, equity / peak - 1.0)
        busy_until = exit_i
        rows.append(
            {
                "signal_timestamp": tape["timestamp"].iloc[signal_i],
                "entry_timestamp": tape["timestamp"].iloc[entry_i],
                "exit_timestamp": tape["timestamp"].iloc[exit_i],
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "tp_price_move": tp_move,
                "sl_price_move": sl_move,
                "raw_return": raw_return,
                "account_return": account_return,
                "equity": equity,
                "exit_reason": exit_reason,
            }
        )
        curve_rows.append({"timestamp": tape["timestamp"].iloc[exit_i], "equity": equity})
    ledger = pd.DataFrame(rows)
    curve = pd.DataFrame(curve_rows)
    pnl = equity - 1.0
    metrics = {
        "pnl_pct": 100.0 * pnl,
        "mdd_pct": 100.0 * mdd,
        "calmar": float(pnl / abs(mdd)) if mdd < 0 else 0.0,
        "trades": int(len(ledger)),
        "win_rate": float((ledger["account_return"] > 0).mean()) if len(ledger) else 0.0,
        "long_trades": int((ledger["side"] > 0).sum()) if len(ledger) else 0,
        "short_trades": int((ledger["side"] < 0).sum()) if len(ledger) else 0,
        "exit_reasons": ledger["exit_reason"].value_counts().to_dict() if len(ledger) else {},
        "notional": float(notional),
    }
    return metrics, ledger, curve


def _read_hourly_with_zigzag() -> tuple[pd.DataFrame, list[str]]:
    frame, features = label_suite._read_hourly()
    labels = []
    for year in (2024, 2025, 2026):
        path = ZIGZAG_DIR / f"zigzag_action_labels_{year}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        labels.append(pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"]))
    label_frame = pd.concat(labels, ignore_index=True).drop_duplicates("timestamp")
    frame = frame.merge(label_frame, on="timestamp", how="left", validate="one_to_one")
    if frame.loc[frame["timestamp"] <= OOS_END, "zigzag_action"].isna().any():
        raise RuntimeError("missing zigzag labels at hourly timestamps")
    frame["zigzag_action"] = frame["zigzag_action"].astype(np.int8)
    return frame, features


def _fit_direction_oof(
    x: np.ndarray,
    y: np.ndarray,
    train_mask: np.ndarray,
) -> tuple[list[HistGradientBoostingClassifier], np.ndarray, np.ndarray, list[dict[str, int]]]:
    train_indices = np.flatnonzero(train_mask)
    oof = np.full((len(x), 3), np.nan, dtype=np.float64)
    fold_rows = []
    splitter = TimeSeriesSplit(n_splits=5, gap=DIRECTION_META_EMBARGO_HOURS)
    for fold, (fit_local, test_local) in enumerate(splitter.split(train_indices), start=1):
        fit_idx, test_idx = train_indices[fit_local], train_indices[test_local]
        models = label_suite._fit_classifiers(x[fit_idx], y[fit_idx], seeds=(SEEDS[(fold - 1) % len(SEEDS)],))
        oof[test_idx] = label_suite._predict_action_probability(models, x[test_idx])
        fold_rows.append(
            {
                "fold": fold,
                "fit_rows": len(fit_idx),
                "purge_hours": DIRECTION_META_EMBARGO_HOURS,
                "oof_rows": len(test_idx),
            }
        )
    final_models = label_suite._fit_classifiers(x[train_indices], y[train_indices], seeds=SEEDS)
    probability = label_suite._predict_action_probability(final_models, x)
    return final_models, probability, oof, fold_rows


def _meta_matrix(x: np.ndarray, probability: np.ndarray) -> np.ndarray:
    side = probability.argmax(axis=1)
    confidence = probability.max(axis=1)
    margin = np.abs(probability[:, 1] - probability[:, 2])
    signed_side = np.where(side == 1, 1.0, np.where(side == 2, -1.0, 0.0))
    return np.column_stack([x, probability, confidence, margin, signed_side])


def _hourly_event(action: np.ndarray) -> np.ndarray:
    previous = np.r_[0, action[:-1]]
    return (action != 0) & (action != previous)


def _meta_targets(
    hourly: pd.DataFrame,
    probability: np.ndarray,
    candidate: np.ndarray,
    tape: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    tape_index = pd.Series(np.arange(len(tape)), index=tape["timestamp"])
    target = np.zeros(len(hourly), dtype=np.int8)
    eligible = np.zeros(len(hourly), dtype=bool)
    action = probability.argmax(axis=1)
    for i in np.flatnonzero(candidate):
        available = hourly["timestamp"].iloc[i] + pd.Timedelta(hours=1)
        entry_timestamp = available + pd.Timedelta(minutes=5)
        if entry_timestamp not in tape_index.index:
            continue
        entry_i = int(tape_index.loc[entry_timestamp])
        exit_i = entry_i + MAX_HOLD_BARS
        if exit_i >= len(tape):
            continue
        side = 1 if int(action[i]) == 1 else -1
        raw = side * (float(tape["close"].iloc[exit_i]) / float(tape["open"].iloc[entry_i]) - 1.0)
        target[i] = int(raw > ROUND_TRIP_COST)
        eligible[i] = True
    return target, eligible


def _triple_barrier_meta_targets(
    hourly: pd.DataFrame,
    probability: np.ndarray,
    candidate: np.ndarray,
    tape: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Label OOF side decisions with the exact recommended execution barriers."""
    tape_index = pd.Series(np.arange(len(tape)), index=tape["timestamp"])
    target = np.zeros(len(hourly), dtype=np.int8)
    net_return = np.full(len(hourly), np.nan, dtype=np.float64)
    eligible = np.zeros(len(hourly), dtype=bool)
    reasons: list[str] = []
    action = probability.argmax(axis=1)
    for i in np.flatnonzero(candidate):
        available = hourly["timestamp"].iloc[i] + pd.Timedelta(hours=1)
        if available not in tape_index.index:
            continue
        signal_i = int(tape_index.loc[available])
        entry_i = signal_i + 1
        maximum_exit = entry_i + MAX_HOLD_BARS
        if maximum_exit >= len(tape):
            continue
        side = 1 if int(action[i]) == 1 else -1
        entry = float(tape["open"].iloc[entry_i])
        atr_pct = float(tape["atr_pct"].iloc[signal_i])
        tp_move = float(
            np.clip(
                RECOMMENDED_TP_ATR_MULTIPLE * atr_pct,
                RECOMMENDED_MIN_TP,
                RECOMMENDED_MAX_TP,
            )
        )
        sl_move = float(
            np.clip(
                RECOMMENDED_SL_ATR_MULTIPLE * atr_pct,
                RECOMMENDED_MIN_SL,
                RECOMMENDED_MAX_SL,
            )
        )
        if side > 0:
            tp_price, sl_price = entry * (1.0 + tp_move), entry * (1.0 - sl_move)
        else:
            tp_price, sl_price = entry * (1.0 - tp_move), entry * (1.0 + sl_move)
        exit_price = float(tape["close"].iloc[maximum_exit])
        reason = "time_barrier"
        for bar_i in range(entry_i, maximum_exit + 1):
            high = float(tape["high"].iloc[bar_i])
            low = float(tape["low"].iloc[bar_i])
            if side > 0:
                stop_hit, target_hit = low <= sl_price, high >= tp_price
            else:
                stop_hit, target_hit = high >= sl_price, low <= tp_price
            if stop_hit:
                exit_price, reason = float(sl_price), "stop_loss"
                break
            if target_hit:
                exit_price, reason = float(tp_price), "take_profit"
                break
        raw_return = side * (exit_price / entry - 1.0)
        net_return[i] = raw_return - ROUND_TRIP_COST
        target[i] = int(net_return[i] > 0.0)
        eligible[i] = True
        reasons.append(reason)
    return target, net_return, eligible, pd.Series(reasons).value_counts().to_dict()


def _fit_net_return_regressors(x: np.ndarray, y: np.ndarray) -> list[HistGradientBoostingRegressor]:
    models = []
    for seed in SEEDS:
        model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.035,
            max_iter=220,
            max_depth=4,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=int(seed),
        )
        model.fit(x, y)
        models.append(model)
    return models


def _predict_net_return(models: list[HistGradientBoostingRegressor], x: np.ndarray) -> np.ndarray:
    prediction = np.zeros(len(x), dtype=np.float64)
    for model in models:
        prediction += model.predict(x)
    return prediction / len(models)


def _hourly_to_five_signal(hourly: pd.DataFrame, action: np.ndarray, tape: pd.DataFrame) -> np.ndarray:
    event_frame = pd.DataFrame(
        {
            "timestamp": hourly["timestamp"] + pd.Timedelta(hours=1),
            "action": np.asarray(action, dtype=np.int8),
        }
    )
    event_frame = event_frame.loc[event_frame["action"] != 0]
    mapped = tape[["timestamp"]].merge(event_frame, on="timestamp", how="left", validate="one_to_one")
    return mapped["action"].fillna(0).to_numpy(dtype=np.int8)


def _plot_equity(curves: dict[str, pd.DataFrame], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(14, 6))
    for name, curve in curves.items():
        ax.step(curve["timestamp"], 100.0 * (curve["equity"] - 1.0), where="post", label=name, linewidth=1.8)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.set_ylabel("Cumulative PnL (%)")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_trades(tape: pd.DataFrame, ledger: pd.DataFrame, path: Path, stage: str) -> None:
    start, end = OOS_START, OOS_START + pd.Timedelta(days=21)
    price = tape.loc[tape["timestamp"].between(start, end)]
    trades = ledger.loc[pd.to_datetime(ledger["entry_timestamp"]).between(start, end)] if len(ledger) else ledger
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(price["timestamp"], price["close"], color="#263238", linewidth=1.0, label="BTC close")
    if len(trades):
        long = trades.loc[trades["side"] > 0]
        short = trades.loc[trades["side"] < 0]
        ax.scatter(long["entry_timestamp"], long["entry_price"], marker="^", s=36, color="#00897b", label="LONG")
        ax.scatter(short["entry_timestamp"], short["entry_price"], marker="v", s=36, color="#e53935", label="SHORT")
        ax.scatter(trades["exit_timestamp"], trades["exit_price"], marker="x", s=25, color="#3949ab", label="EXIT")
    ax.set_title(f"{stage} — Q1 first 21 days")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tape = _read_five_minute()
    parent_action = _read_parent_signal(tape)
    parent_event_action = _event_only(parent_action)
    stage_signals: dict[str, np.ndarray] = {
        "01_no_learned_exit": parent_action,
        "02_fixed_sizing": parent_action,
        "03_parent_event_only": parent_event_action,
    }
    stage_notionals = {
        "01_no_learned_exit": CURRENT_FIXED_NOTIONAL,
        "02_fixed_sizing": CONSERVATIVE_NOTIONAL,
        "03_parent_event_only": CONSERVATIVE_NOTIONAL,
    }

    hourly, feature_columns = _read_hourly_with_zigzag()
    x = hourly[feature_columns].to_numpy(dtype=np.float64)
    y = hourly["zigzag_action"].to_numpy(dtype=np.int8)
    train_mask = hourly["timestamp"].lt(DIRECTION_META_TRAIN_END).to_numpy()
    direction_models, direction_probability, oof_probability, fold_rows = _fit_direction_oof(x, y, train_mask)

    direction_action = direction_probability.argmax(axis=1).astype(np.int8)
    direction_event = _hourly_event(direction_action)
    direction_event_action = np.where(direction_event, direction_action, 0).astype(np.int8)
    stage_signals["04_split_direction"] = _hourly_to_five_signal(hourly, direction_event_action, tape)
    stage_notionals["04_split_direction"] = CONSERVATIVE_NOTIONAL

    oof_rows = train_mask & np.isfinite(oof_probability).all(axis=1)
    oof_action = np.zeros(len(hourly), dtype=np.int8)
    oof_action[oof_rows] = oof_probability[oof_rows].argmax(axis=1).astype(np.int8)
    oof_candidate = oof_rows & _hourly_event(oof_action)
    meta_target, meta_eligible = _meta_targets(hourly, oof_probability, oof_candidate, tape)
    meta_train = oof_candidate & meta_eligible
    if meta_train.sum() < 100 or np.unique(meta_target[meta_train]).size != 2:
        raise RuntimeError(f"insufficient purged OOF meta rows: {int(meta_train.sum())}")
    oof_meta_x = _meta_matrix(x, np.nan_to_num(oof_probability, nan=0.0))
    meta_models = label_suite._fit_binary_models(oof_meta_x[meta_train], meta_target[meta_train])
    meta_x = _meta_matrix(x, direction_probability)
    meta_probability = label_suite._predict_binary(meta_models, meta_x)
    (
        triple_meta_target,
        triple_meta_net_return,
        triple_meta_eligible,
        triple_meta_reason_counts,
    ) = _triple_barrier_meta_targets(hourly, oof_probability, oof_candidate, tape)
    triple_meta_train = oof_candidate & triple_meta_eligible
    if triple_meta_train.sum() < 100 or np.unique(triple_meta_target[triple_meta_train]).size != 2:
        raise RuntimeError(f"insufficient triple-barrier OOF meta rows: {int(triple_meta_train.sum())}")
    triple_meta_models = label_suite._fit_binary_models(
        oof_meta_x[triple_meta_train], triple_meta_target[triple_meta_train]
    )
    triple_meta_probability = label_suite._predict_binary(triple_meta_models, meta_x)
    net_return_models = _fit_net_return_regressors(
        oof_meta_x[triple_meta_train], triple_meta_net_return[triple_meta_train]
    )
    net_return_score = _predict_net_return(net_return_models, meta_x)
    train_net_return_score = net_return_score[triple_meta_train]
    rank_target = pd.Series(triple_meta_net_return[triple_meta_train]).rank(method="average", pct=True).to_numpy()
    net_return_rank_models = _fit_net_return_regressors(oof_meta_x[triple_meta_train], rank_target)
    net_return_rank_score = _predict_net_return(net_return_rank_models, meta_x)
    train_net_return_rank_score = net_return_rank_score[triple_meta_train]
    recommended_replay = {
        "tp_atr_multiple": RECOMMENDED_TP_ATR_MULTIPLE,
        "sl_atr_multiple": RECOMMENDED_SL_ATR_MULTIPLE,
        "min_tp": RECOMMENDED_MIN_TP,
        "min_sl": RECOMMENDED_MIN_SL,
        "max_tp": RECOMMENDED_MAX_TP,
        "max_sl": RECOMMENDED_MAX_SL,
    }

    validation_meta_rows = []
    validation_curves: dict[str, pd.DataFrame] = {}
    oos_curves: dict[str, pd.DataFrame] = {}
    results: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for stage in ("01_no_learned_exit", "02_fixed_sizing", "03_parent_event_only", "04_split_direction"):
        val_metrics, val_ledger, val_curve = _replay(
            tape, stage_signals[stage], VALIDATION_START, VALIDATION_END, stage_notionals[stage]
        )
        oos_metrics, oos_ledger, oos_curve = _replay(tape, stage_signals[stage], OOS_START, OOS_END, stage_notionals[stage])
        results[stage] = {"validation": val_metrics, "oos_q1": oos_metrics}
        validation_curves[stage] = val_curve
        oos_curves[stage] = oos_curve
        ledgers[stage] = oos_ledger
        val_ledger.to_csv(out_dir / f"{stage}_validation_ledger.csv", index=False)
        oos_ledger.to_csv(out_dir / f"{stage}_oos_q1_ledger.csv", index=False)

    for threshold in META_THRESHOLDS:
        selected = direction_event & (meta_probability >= threshold)
        action = np.where(selected, direction_action, 0).astype(np.int8)
        signal = _hourly_to_five_signal(hourly, action, tape)
        metrics, _, _ = _replay(tape, signal, VALIDATION_START, VALIDATION_END, CONSERVATIVE_NOTIONAL)
        validation_meta_rows.append({"threshold": threshold, **metrics})
    eligible = [row for row in validation_meta_rows if row["trades"] >= 10]
    selected_meta = max(eligible or validation_meta_rows, key=lambda row: (row["calmar"], row["pnl_pct"], -row["threshold"]))
    meta_threshold = float(selected_meta["threshold"])
    meta_action = np.where(direction_event & (meta_probability >= meta_threshold), direction_action, 0).astype(np.int8)
    meta_signal = _hourly_to_five_signal(hourly, meta_action, tape)
    val_metrics, val_ledger, val_curve = _replay(
        tape, meta_signal, VALIDATION_START, VALIDATION_END, CONSERVATIVE_NOTIONAL
    )
    oos_metrics, oos_ledger, oos_curve = _replay(tape, meta_signal, OOS_START, OOS_END, CONSERVATIVE_NOTIONAL)
    results["05_split_direction_meta"] = {"validation": val_metrics, "oos_q1": oos_metrics}
    validation_curves["05_split_direction_meta"] = val_curve
    oos_curves["05_split_direction_meta"] = oos_curve
    ledgers["05_split_direction_meta"] = oos_ledger
    val_ledger.to_csv(out_dir / "05_split_direction_meta_validation_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "05_split_direction_meta_oos_q1_ledger.csv", index=False)
    pd.DataFrame(validation_meta_rows).to_csv(out_dir / "meta_threshold_validation.csv", index=False)

    val_metrics, val_ledger, val_curve = _replay(
        tape,
        meta_signal,
        VALIDATION_START,
        VALIDATION_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    oos_metrics, oos_ledger, oos_curve = _replay(
        tape,
        meta_signal,
        OOS_START,
        OOS_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    results["06_recommended_sltp_terminal_meta"] = {"validation": val_metrics, "oos_q1": oos_metrics}
    validation_curves["06_recommended_sltp_terminal_meta"] = val_curve
    oos_curves["06_recommended_sltp_terminal_meta"] = oos_curve
    ledgers["06_recommended_sltp_terminal_meta"] = oos_ledger
    val_ledger.to_csv(out_dir / "06_recommended_sltp_terminal_meta_validation_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "06_recommended_sltp_terminal_meta_oos_q1_ledger.csv", index=False)

    triple_validation_rows = []
    for threshold in META_THRESHOLDS:
        selected = direction_event & (triple_meta_probability >= threshold)
        action = np.where(selected, direction_action, 0).astype(np.int8)
        signal = _hourly_to_five_signal(hourly, action, tape)
        metrics, _, _ = _replay(
            tape,
            signal,
            VALIDATION_START,
            VALIDATION_END,
            CONSERVATIVE_NOTIONAL,
            **recommended_replay,
        )
        triple_validation_rows.append({"threshold": threshold, **metrics})
    triple_eligible = [row for row in triple_validation_rows if row["trades"] >= 10]
    selected_triple_meta = max(
        triple_eligible or triple_validation_rows,
        key=lambda row: (row["calmar"], row["pnl_pct"], -row["threshold"]),
    )
    triple_meta_threshold = float(selected_triple_meta["threshold"])
    triple_action = np.where(
        direction_event & (triple_meta_probability >= triple_meta_threshold), direction_action, 0
    ).astype(np.int8)
    triple_signal = _hourly_to_five_signal(hourly, triple_action, tape)
    val_metrics, val_ledger, val_curve = _replay(
        tape,
        triple_signal,
        VALIDATION_START,
        VALIDATION_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    oos_metrics, oos_ledger, oos_curve = _replay(
        tape,
        triple_signal,
        OOS_START,
        OOS_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    results["07_recommended_sltp_triple_meta"] = {"validation": val_metrics, "oos_q1": oos_metrics}
    validation_curves["07_recommended_sltp_triple_meta"] = val_curve
    oos_curves["07_recommended_sltp_triple_meta"] = oos_curve
    ledgers["07_recommended_sltp_triple_meta"] = oos_ledger
    val_ledger.to_csv(out_dir / "07_recommended_sltp_triple_meta_validation_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "07_recommended_sltp_triple_meta_oos_q1_ledger.csv", index=False)
    pd.DataFrame(triple_validation_rows).to_csv(out_dir / "triple_meta_threshold_validation.csv", index=False)

    regression_validation_rows = []
    for quantile in REGRESSION_RANK_QUANTILES:
        score_threshold = float(np.quantile(train_net_return_score, quantile))
        selected = direction_event & (net_return_score >= score_threshold)
        action = np.where(selected, direction_action, 0).astype(np.int8)
        signal = _hourly_to_five_signal(hourly, action, tape)
        metrics, _, _ = _replay(
            tape,
            signal,
            VALIDATION_START,
            VALIDATION_END,
            CONSERVATIVE_NOTIONAL,
            **recommended_replay,
        )
        regression_validation_rows.append(
            {"training_score_quantile": quantile, "score_threshold": score_threshold, **metrics}
        )
    regression_eligible = [row for row in regression_validation_rows if row["trades"] >= 10]
    selected_regression = max(
        regression_eligible or regression_validation_rows,
        key=lambda row: (row["calmar"], row["pnl_pct"], -row["training_score_quantile"]),
    )
    regression_quantile = float(selected_regression["training_score_quantile"])
    regression_threshold = float(selected_regression["score_threshold"])
    regression_action = np.where(
        direction_event & (net_return_score >= regression_threshold), direction_action, 0
    ).astype(np.int8)
    regression_signal = _hourly_to_five_signal(hourly, regression_action, tape)
    val_metrics, val_ledger, val_curve = _replay(
        tape,
        regression_signal,
        VALIDATION_START,
        VALIDATION_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    oos_metrics, oos_ledger, oos_curve = _replay(
        tape,
        regression_signal,
        OOS_START,
        OOS_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    results["08_recommended_sltp_net_return_regression"] = {"validation": val_metrics, "oos_q1": oos_metrics}
    validation_curves["08_recommended_sltp_net_return_regression"] = val_curve
    oos_curves["08_recommended_sltp_net_return_regression"] = oos_curve
    ledgers["08_recommended_sltp_net_return_regression"] = oos_ledger
    val_ledger.to_csv(out_dir / "08_recommended_sltp_net_return_regression_validation_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "08_recommended_sltp_net_return_regression_oos_q1_ledger.csv", index=False)
    pd.DataFrame(regression_validation_rows).to_csv(out_dir / "net_return_regression_rank_validation.csv", index=False)

    rank_validation_rows = []
    for quantile in REGRESSION_RANK_QUANTILES:
        score_threshold = float(np.quantile(train_net_return_rank_score, quantile))
        selected = direction_event & (net_return_rank_score >= score_threshold)
        action = np.where(selected, direction_action, 0).astype(np.int8)
        signal = _hourly_to_five_signal(hourly, action, tape)
        metrics, _, _ = _replay(
            tape,
            signal,
            VALIDATION_START,
            VALIDATION_END,
            CONSERVATIVE_NOTIONAL,
            **recommended_replay,
        )
        rank_validation_rows.append(
            {"training_score_quantile": quantile, "score_threshold": score_threshold, **metrics}
        )
    rank_eligible = [row for row in rank_validation_rows if row["trades"] >= 10]
    selected_rank = max(
        rank_eligible or rank_validation_rows,
        key=lambda row: (row["calmar"], row["pnl_pct"], -row["training_score_quantile"]),
    )
    rank_quantile = float(selected_rank["training_score_quantile"])
    rank_threshold = float(selected_rank["score_threshold"])
    rank_action = np.where(
        direction_event & (net_return_rank_score >= rank_threshold), direction_action, 0
    ).astype(np.int8)
    rank_signal = _hourly_to_five_signal(hourly, rank_action, tape)
    val_metrics, val_ledger, val_curve = _replay(
        tape,
        rank_signal,
        VALIDATION_START,
        VALIDATION_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    oos_metrics, oos_ledger, oos_curve = _replay(
        tape,
        rank_signal,
        OOS_START,
        OOS_END,
        CONSERVATIVE_NOTIONAL,
        **recommended_replay,
    )
    results["09_recommended_sltp_net_return_rank"] = {"validation": val_metrics, "oos_q1": oos_metrics}
    validation_curves["09_recommended_sltp_net_return_rank"] = val_curve
    oos_curves["09_recommended_sltp_net_return_rank"] = oos_curve
    ledgers["09_recommended_sltp_net_return_rank"] = oos_ledger
    val_ledger.to_csv(out_dir / "09_recommended_sltp_net_return_rank_validation_ledger.csv", index=False)
    oos_ledger.to_csv(out_dir / "09_recommended_sltp_net_return_rank_oos_q1_ledger.csv", index=False)
    pd.DataFrame(rank_validation_rows).to_csv(out_dir / "net_return_rank_validation.csv", index=False)

    with REFERENCE_REPORT.open("r", encoding="utf-8") as handle:
        reference = json.load(handle)["omega4_2_replayed_baseline"]
    results["00_learned_exit_reference"] = {
        "validation": {
            "pnl_pct": reference["validation"]["pnl"],
            "mdd_pct": reference["validation"]["mdd"],
            "trades": reference["validation"]["trades"],
            "win_rate": reference["validation"]["wr"],
            "exit_reasons": reference["validation"]["exit_reasons"],
        },
        "oos_q1": {
            "pnl_pct": reference["oos"]["pnl"],
            "mdd_pct": reference["oos"]["mdd"],
            "trades": reference["oos"]["trades"],
            "win_rate": reference["oos"]["wr"],
            "exit_reasons": reference["oos"]["exit_reasons"],
        },
        "note": "Saved metrics only; its ledger was not used as replay input.",
    }

    summary_rows = []
    order = [
        "00_learned_exit_reference",
        "01_no_learned_exit",
        "02_fixed_sizing",
        "03_parent_event_only",
        "04_split_direction",
        "05_split_direction_meta",
        "06_recommended_sltp_terminal_meta",
        "07_recommended_sltp_triple_meta",
        "08_recommended_sltp_net_return_regression",
        "09_recommended_sltp_net_return_rank",
    ]
    for stage in order:
        summary_rows.append(
            {
                "stage": stage,
                **{f"validation_{key}": value for key, value in results[stage]["validation"].items() if key != "exit_reasons"},
                **{f"oos_q1_{key}": value for key, value in results[stage]["oos_q1"].items() if key != "exit_reasons"},
            }
        )
    pd.DataFrame(summary_rows).to_csv(out_dir / "architecture_ablation_summary.csv", index=False)
    _plot_equity(validation_curves, out_dir / "equity_ablation_validation.png", "BTC architecture ablation — validation")
    _plot_equity(
        oos_curves,
        out_dir / "equity_ablation_oos_q1.png",
        "BTC architecture ablation — Q1 diagnostic (not promotion evidence)",
    )
    for stage, ledger in ledgers.items():
        _plot_trades(tape, ledger, out_dir / f"{stage}_oos_q1_trade_chart.png", stage)

    joblib.dump(
        {
            "feature_columns": feature_columns,
            "direction_models": direction_models,
            "meta_models": meta_models,
            "meta_threshold": meta_threshold,
            "triple_barrier_meta_models": triple_meta_models,
            "triple_barrier_meta_threshold": triple_meta_threshold,
            "net_return_regression_models": net_return_models,
            "net_return_regression_training_quantile": regression_quantile,
            "net_return_regression_score_threshold": regression_threshold,
            "net_return_rank_models": net_return_rank_models,
            "net_return_rank_training_quantile": rank_quantile,
            "net_return_rank_score_threshold": rank_threshold,
            "recommended_replay": recommended_replay,
            "training_end_exclusive": DIRECTION_META_TRAIN_END,
        },
        out_dir / "split_direction_meta_research_bundle.joblib",
    )
    report = {
        "model_id": "btc_v1_architecture_ablation_20260716",
        "status": "research_only_not_promoted",
        "promotion_eligible": False,
        "promotion_blocker": "Q1 was replayed during implementation and after the mandatory 72-hour embargo correction.",
        "objective": "Test architecture changes one at a time after BTC H48 stabilization.",
        "split": {
            "train_end": TRAIN_END,
            "direction_meta_train_end_exclusive": DIRECTION_META_TRAIN_END,
            "direction_meta_validation_embargo_hours": DIRECTION_META_EMBARGO_HOURS,
            "validation": [VALIDATION_START, VALIDATION_END],
            "oos_q1": [OOS_START, OOS_END],
            "boundary_deviation": "Validation begins 2025-10-01 to match the stabilized parent artifact; default policy begins 2025-09-01.",
        },
        "execution_contract": {
            "bar_interval": "5m",
            "entry": "next bar open",
            "atr_window": ATR_WINDOW,
            "tp_atr_multiple": TP_ATR_MULTIPLE,
            "sl_atr_multiple": SL_ATR_MULTIPLE,
            "min_tp_price_move": MIN_TP,
            "min_sl_price_move": MIN_SL,
            "max_tp_price_move": MAX_TP,
            "max_sl_price_move": MAX_SL,
            "max_hold_bars": MAX_HOLD_BARS,
            "round_trip_cost": ROUND_TRIP_COST,
            "same_bar_tp_sl_policy": "stop_first_conservative",
            "recommended_sltp": {
                "tp_atr_multiple": RECOMMENDED_TP_ATR_MULTIPLE,
                "sl_atr_multiple": RECOMMENDED_SL_ATR_MULTIPLE,
                "min_tp_price_move": RECOMMENDED_MIN_TP,
                "min_sl_price_move": RECOMMENDED_MIN_SL,
                "max_tp_price_move": RECOMMENDED_MAX_TP,
                "max_sl_price_move": RECOMMENDED_MAX_SL,
            },
        },
        "direction_meta": {
            "direction_target": "BTC zigzag_action",
            "direction_features": feature_columns,
            "direction_model": "3-seed HistGradientBoostingClassifier ensemble",
            "meta_model": "separate HistGradientBoostingClassifier ensemble",
            "meta_target": "OOF direction-side net 72-bar return positive",
            "primary_oof": True,
            "purge_hours": DIRECTION_META_EMBARGO_HOURS,
            "oof_meta_rows": int(meta_train.sum()),
            "oof_meta_positive_rate": float(meta_target[meta_train].mean()),
            "folds": fold_rows,
            "selected_meta_threshold_validation_only": meta_threshold,
            "meta_threshold_search": validation_meta_rows,
            "triple_barrier_meta": {
                "target": "OOF direction-side first-hit TP/SL/time-barrier net return positive",
                "execution_aligned": True,
                "oof_rows": int(triple_meta_train.sum()),
                "positive_rate": float(triple_meta_target[triple_meta_train].mean()),
                "label_exit_reasons": triple_meta_reason_counts,
                "selected_threshold_validation_only": triple_meta_threshold,
                "threshold_search": triple_validation_rows,
            },
            "net_return_regression_rank": {
                "target": "OOF direction-side first-hit TP/SL/time-barrier raw return minus round-trip cost",
                "continuous_target": True,
                "model": "3-seed HistGradientBoostingRegressor ensemble",
                "ranking_reference": "training score distribution only",
                "target_rows": int(triple_meta_train.sum()),
                "target_mean": float(np.mean(triple_meta_net_return[triple_meta_train])),
                "target_median": float(np.median(triple_meta_net_return[triple_meta_train])),
                "selected_training_score_quantile_validation_only": regression_quantile,
                "selected_score_threshold": regression_threshold,
                "rank_validation_search": regression_validation_rows,
            },
            "net_return_percentile_rank": {
                "target": "percentile rank of OOF execution-aligned triple-barrier net return",
                "continuous_ordinal_target": True,
                "model": "3-seed HistGradientBoostingRegressor ensemble",
                "ranking_reference": "training target and score distributions only",
                "selected_training_score_quantile_validation_only": rank_quantile,
                "selected_score_threshold": rank_threshold,
                "rank_validation_search": rank_validation_rows,
            },
        },
        "results": results,
        "fresh_forward_contract": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "oos_used_for_selection": False,
            "oos_replay_count_during_development": 6,
        },
        "artifacts": {
            "summary": out_dir / "architecture_ablation_summary.csv",
            "validation_equity_chart": out_dir / "equity_ablation_validation.png",
            "oos_equity_chart": out_dir / "equity_ablation_oos_q1.png",
            "bundle": out_dir / "split_direction_meta_research_bundle.joblib",
        },
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False, default=_json_default)
    print(
        json.dumps(
            {
                "out_dir": str(out_dir),
                "meta_threshold": meta_threshold,
                "triple_barrier_meta_threshold": triple_meta_threshold,
                "net_return_regression_quantile": regression_quantile,
                "net_return_regression_threshold": regression_threshold,
                "net_return_rank_quantile": rank_quantile,
                "net_return_rank_threshold": rank_threshold,
                "results": results,
            },
            indent=2,
            default=_json_default,
        )
    )


if __name__ == "__main__":
    main()
