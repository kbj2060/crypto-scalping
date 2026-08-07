#!/usr/bin/env python3
"""Research-only ETH Omega4.6.1 censored stopping-value successor.

This stage trains only on entries emitted by the frozen live router.  It combines:

* four competing-risk landmark heads (TP first / SL first / neither by 12, 48, 96, 384 bars),
* a bootstrap estimate of the value advantage of EXIT over HOLD, and
* a one-sided temporal conformal penalty that makes uncertain decisions abstain to frozen SLTP.

The EXIT label is re-entry aware.  For every historical position state, the counterfactual exits
at the next executable open and then runs the frozen entry router while the original position
would still have occupied the slot.  The HOLD label follows the original SLTP lifecycle.  The two
paths reconverge at the original exit, so labels do not include unrelated far-future returns.

No live module, environment setting, or production artifact is changed.  All available evaluation
windows were already consumed by earlier research and are diagnostic-only here.
"""

from __future__ import annotations

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import brier_score_loss, mean_squared_error


ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import audit_eth_omega461_live_chop_hazard_composition_20260724 as composition  # noqa: E402
import research_eth_omega461_competing_risk_rescue_20260724 as hazard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402


MODEL_ID = "eth_omega461_censored_stopping_value_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_START, TRAIN_END = hazard.TRAIN_START, hazard.TRAIN_END
EXTENSION_START, EXTENSION_END = "2026-04-01", "2026-07-12 09:00:00"
HORIZONS = (12, 48, 96, 384)
N_BOOTSTRAPS = 3
SEED = 260726
SOURCE_IDS = {name: idx for idx, name in enumerate(hazard.greedy.PRIORITY)}
NON_FEATURE_COLUMNS = {
    "episode_id", "entry_timestamp", "state_timestamp", "source_component", "baseline_cause",
    "bars_to_baseline_exit", "advantage", "sample_weight", "q_exit", "q_hold",
    *(f"risk_label_h{h}" for h in HORIZONS),
}


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _raw_move(price: float, *, side: int, entry_price: float) -> float:
    return float(side * (price - entry_price) / max(entry_price, 1.0e-12))


def _bar_moves(arrays: dict[str, np.ndarray], i: int, *, side: int, entry_price: float) -> tuple[float, float, float]:
    close_move = _raw_move(float(arrays["close"][i]), side=side, entry_price=entry_price)
    if side > 0:
        best = _raw_move(float(arrays["high"][i]), side=side, entry_price=entry_price)
        worst = _raw_move(float(arrays["low"][i]), side=side, entry_price=entry_price)
    else:
        best = _raw_move(float(arrays["low"][i]), side=side, entry_price=entry_price)
        worst = _raw_move(float(arrays["high"][i]), side=side, entry_price=entry_price)
    return close_move, best, worst


def _entry_option(
    components: dict[str, dict[str, Any]],
    active_masks: dict[str, np.ndarray],
    arrays: dict[str, np.ndarray],
    i: int,
    *,
    fee_eff: float,
    slip_eff: float,
) -> dict[str, Any] | None:
    for name in hazard.greedy.PRIORITY:
        comp = components[name]
        if not bool(active_masks[name][i]):
            continue
        side = int(comp["dec"]["side"].iloc[i])
        if side == 0:
            continue
        margin = float(comp["margin"][i])
        leverage = float(comp["leverage"][i])
        if margin <= 0.0 or leverage <= 0.0:
            continue
        scale = hazard.greedy.SCALE_MAP.get(f"{name}_{'L' if side > 0 else 'S'}", 1.0)
        leverage = min(leverage * scale, hazard.greedy.LEVERAGE_CAP)
        base_notional = min(margin * leverage, hazard.greedy.NOTIONAL_CAP)
        notional = min(base_notional * composition.ENTRY_NOTIONAL_MULTIPLIER, composition.MAX_ENTRY_NOTIONAL)
        chop_probability = float(arrays["chop"][i])
        threshold = composition.CHOP_SOFT_SIZE_THRESHOLD
        chop_multiplier = 1.0 if chop_probability < threshold else max(
            0.0, 1.0 - (chop_probability - threshold) / (1.0 - threshold)
        )
        notional *= chop_multiplier
        if notional <= 0.0:
            continue
        leverage = notional / max(margin, 1.0e-12)
        filled, entry_price, entry_fee, route = hazard.omega._try_execution(
            arrays, i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            continue
        return {
            "signal_i": int(i), "entry_i": int(i + 1), "source_component": name,
            "side": side, "entry_price": float(entry_price), "entry_fee": float(entry_fee),
            "entry_route": route, "margin_fraction": margin, "leverage": leverage,
            "notional": notional, "take_profit": float(comp["dec"]["take_profit"].iloc[i]),
            "stop_loss": float(comp["dec"]["stop_loss"].iloc[i]),
            "chop_probability": chop_probability,
            "sizing_multiplier": notional / max(base_notional, 1.0e-12),
        }
    return None


def _position_path(
    arrays: dict[str, np.ndarray], option: dict[str, Any], *, fee_eff: float, slip_eff: float,
    end_i: int,
) -> dict[str, Any]:
    side = int(option["side"])
    entry_price = float(option["entry_price"])
    cause = "end_censored"
    exit_signal_i = int(end_i)
    states: list[tuple[int, float, float, float]] = []
    mfe = 0.0
    mae = 0.0
    for i in range(int(option["entry_i"]), int(end_i) + 1):
        move, best, worst = _bar_moves(arrays, i, side=side, entry_price=entry_price)
        mfe = max(mfe, best)
        mae = min(mae, worst)
        states.append((i, move, mfe, mae))
        if option["stop_loss"] > 0.0 and worst <= -abs(float(option["stop_loss"])):
            cause, exit_signal_i = "stop_loss", i
            break
        if option["take_profit"] > 0.0 and best >= float(option["take_profit"]):
            cause, exit_signal_i = "take_profit", i
            break
    filled, exit_price, exit_fee, route = hazard.omega._try_execution(
        arrays, exit_signal_i, side, entry=False, fee_base=fee_eff, slip_base=slip_eff
    )
    if not filled:
        raise RuntimeError("frozen exit execution unexpectedly missed")
    raw_exit = _raw_move(float(exit_price), side=side, entry_price=entry_price)
    notional = float(option["notional"])
    multiplier = (1.0 - float(option["entry_fee"]) * notional) * (
        1.0 + raw_exit * notional - float(exit_fee) * notional
    )
    return {
        "cause": cause, "exit_signal_i": int(exit_signal_i), "exit_price": float(exit_price),
        "exit_fee": float(exit_fee), "exit_route": route, "raw_exit": raw_exit,
        "multiplier": float(multiplier), "states": states,
    }


def _early_exit_multiplier(
    arrays: dict[str, np.ndarray], option: dict[str, Any], state_i: int, *, fee_eff: float, slip_eff: float,
) -> float:
    filled, exit_price, exit_fee, _ = hazard.omega._try_execution(
        arrays, state_i, int(option["side"]), entry=False, fee_base=fee_eff, slip_base=slip_eff
    )
    if not filled:
        raise RuntimeError("counterfactual exit execution unexpectedly missed")
    raw_exit = _raw_move(float(exit_price), side=int(option["side"]), entry_price=float(option["entry_price"]))
    n = float(option["notional"])
    return float((1.0 - float(option["entry_fee"]) * n) * (1.0 + raw_exit * n - float(exit_fee) * n))


def _flat_values_until(
    arrays: dict[str, np.ndarray], entry_options: list[dict[str, Any] | None], start_i: int, end_i: int,
    *, fee_eff: float, slip_eff: float,
) -> np.ndarray:
    values = np.ones(end_i + 2, dtype=np.float64)
    for i in range(end_i, start_i - 1, -1):
        option = entry_options[i]
        if option is None:
            values[i] = values[i + 1]
            continue
        outcome = _position_path(arrays, option, fee_eff=fee_eff, slip_eff=slip_eff, end_i=end_i)
        exit_i = int(outcome["exit_signal_i"])
        values[i] = float(outcome["multiplier"]) * values[min(exit_i + 1, end_i + 1)]
    return values


def _risk_label(cause: str, bars_to_exit: int, horizon: int) -> int:
    if bars_to_exit > horizon or cause not in {"take_profit", "stop_loss"}:
        return 0
    return 1 if cause == "take_profit" else 2


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    out = {
        col: pd.to_numeric(frame[col], errors="raise").to_numpy(dtype=np.float64)
        for col in ("open", "high", "low", "close")
    }
    out["chop"] = pd.to_numeric(
        frame["regime3_current_sensitive_wide24_chop_prob"], errors="raise"
    ).to_numpy(dtype=np.float64)
    return out


def build_router_dataset(frame: pd.DataFrame, components: dict[str, dict[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    arrays = _arrays(frame)
    fee, slip = hazard.omega._load_fee_slip()
    fee_eff, slip_eff = float(fee), float(slip)
    active_masks = {name: hazard.omega._active(comp["dec"]) for name, comp in components.items()}
    entry_options = [
        _entry_option(components, active_masks, arrays, i, fee_eff=fee_eff, slip_eff=slip_eff)
        for i in range(len(frame) - 1)
    ] + [None]
    rows: list[dict[str, Any]] = []
    episodes: list[dict[str, Any]] = []
    i = 0
    episode_id = 0
    last_i = len(frame) - 2
    while i <= last_i:
        option = entry_options[i]
        if option is None:
            i += 1
            continue
        outcome = _position_path(arrays, option, fee_eff=fee_eff, slip_eff=slip_eff, end_i=last_i)
        baseline_exit_i = int(outcome["exit_signal_i"])
        if outcome["cause"] == "end_censored" or baseline_exit_i - int(option["entry_i"]) < 2:
            i = baseline_exit_i + 1
            continue
        flat_values = _flat_values_until(
            arrays, entry_options, int(option["entry_i"]) + 1, baseline_exit_i,
            fee_eff=fee_eff, slip_eff=slip_eff,
        )
        usable_states = [state for state in outcome["states"] if state[0] < baseline_exit_i]
        episode_len = max(len(usable_states), 1)
        for state_i, move, mfe, mae in usable_states:
            comp = components[str(option["source_component"])]
            feature = hazard._feature_row(
                comp, row_i=state_i, side=int(option["side"]),
                hold=max(state_i - int(option["entry_i"]), 0), move=move, mfe=mfe, mae=mae,
                notional=float(option["notional"]), leverage=float(option["leverage"]),
                take_profit=float(option["take_profit"]), stop_loss=float(option["stop_loss"]),
            )
            for name, source_id in SOURCE_IDS.items():
                feature[f"component_{name}"] = float(source_id == SOURCE_IDS[str(option["source_component"])])
            bars_to_exit = baseline_exit_i - state_i
            q_hold = float(outcome["multiplier"])
            q_exit = _early_exit_multiplier(
                arrays, option, state_i, fee_eff=fee_eff, slip_eff=slip_eff
            ) * float(flat_values[state_i + 1])
            feature.update({
                "episode_id": episode_id, "entry_timestamp": frame["timestamp"].iloc[int(option["signal_i"])],
                "state_timestamp": frame["timestamp"].iloc[state_i],
                "source_component": str(option["source_component"]),
                "baseline_cause": str(outcome["cause"]), "bars_to_baseline_exit": bars_to_exit,
                "q_exit": q_exit, "q_hold": q_hold,
                "advantage": float(np.log(max(q_exit, 1.0e-8)) - np.log(max(q_hold, 1.0e-8))),
                "sample_weight": 1.0 / episode_len,
            })
            for horizon in HORIZONS:
                feature[f"risk_label_h{horizon}"] = _risk_label(str(outcome["cause"]), bars_to_exit, horizon)
            rows.append(feature)
        episodes.append({
            "episode_id": episode_id, "source_component": option["source_component"],
            "cause": outcome["cause"], "signal_i": option["signal_i"],
            "entry_i": option["entry_i"], "exit_signal_i": baseline_exit_i,
        })
        episode_id += 1
        i = baseline_exit_i + 1
    data = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if data.empty:
        raise RuntimeError("empty live-router stopping dataset")
    diagnostics = {
        "rows": int(len(data)), "episodes": int(data["episode_id"].nunique()),
        "cause_counts": dict(Counter(str(row["cause"]) for row in episodes)),
        "component_counts": dict(Counter(str(row["source_component"]) for row in episodes)),
        "positive_advantage_rows": int((data["advantage"] > 0.0).sum()),
        "positive_advantage_episodes": int(data.groupby("episode_id")["advantage"].max().gt(0.0).sum()),
        "advantage_quantiles": {str(q): float(data["advantage"].quantile(q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
        "label_semantics": "log(Q_exit_reentry_aware / Q_hold_frozen_sltp), reconverged at baseline exit",
    }
    return data, diagnostics


def _feature_columns(data: pd.DataFrame) -> list[str]:
    return [col for col in data.columns if col not in NON_FEATURE_COLUMNS]


def _aligned_proba(model: HistGradientBoostingClassifier, x: np.ndarray) -> np.ndarray:
    raw = model.predict_proba(x)
    out = np.zeros((len(x), 3), dtype=np.float64)
    for idx, label in enumerate(model.classes_):
        out[:, int(label)] = raw[:, idx]
    return out


def fit_bundle(data: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    feature_cols = _feature_columns(data)
    x = data[feature_cols].to_numpy(dtype=np.float64)
    y_adv = data["advantage"].to_numpy(dtype=np.float64)
    weights = data["sample_weight"].to_numpy(dtype=np.float64)
    episodes = data["episode_id"].to_numpy(dtype=np.int64)
    episode_time = data.groupby("episode_id", sort=False)["entry_timestamp"].first()
    split_time = pd.to_datetime(episode_time).quantile(0.80)
    diag_episode_ids = set(episode_time[pd.to_datetime(episode_time) > split_time].index)
    diag_mask = np.asarray([episode in diag_episode_ids for episode in episodes], dtype=bool)
    train_mask = ~diag_mask
    if int(train_mask.sum()) == 0 or int(diag_mask.sum()) == 0:
        raise RuntimeError("empty temporal calibration split")

    diagnostic_reg = HistGradientBoostingRegressor(
        learning_rate=0.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=40,
        l2_regularization=2.0, early_stopping=False, random_state=SEED,
    ).fit(x[train_mask], y_adv[train_mask], sample_weight=weights[train_mask])
    diag_pred = diagnostic_reg.predict(x[diag_mask])
    overprediction = diag_pred - y_adv[diag_mask]
    diagnostics: dict[str, Any] = {
        "feature_count": len(feature_cols), "calibration_split_timestamp": str(split_time),
        "train_rows": int(train_mask.sum()), "calibration_rows": int(diag_mask.sum()),
        "advantage_rmse": float(mean_squared_error(y_adv[diag_mask], diag_pred) ** 0.5),
        "overprediction_quantiles": {str(q): float(np.quantile(overprediction, q)) for q in (0.5, 0.8, 0.9, 0.95)},
    }

    groups = np.unique(episodes)
    rng = np.random.default_rng(SEED)
    advantage_models: list[HistGradientBoostingRegressor] = []
    risk_models: dict[int, list[HistGradientBoostingClassifier]] = {h: [] for h in HORIZONS}
    for boot in range(N_BOOTSTRAPS):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        idx = np.concatenate([np.flatnonzero(episodes == group) for group in sampled])
        reg = HistGradientBoostingRegressor(
            learning_rate=0.05, max_iter=160, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=False, random_state=SEED + 10 + boot,
        ).fit(x[idx], y_adv[idx], sample_weight=weights[idx])
        advantage_models.append(reg)
        for horizon in HORIZONS:
            y = data[f"risk_label_h{horizon}"].to_numpy(dtype=np.int64)
            counts = np.bincount(y, minlength=3)
            class_weight = np.asarray([len(y) / max(3 * count, 1) for count in counts], dtype=np.float64)
            clf = HistGradientBoostingClassifier(
                learning_rate=0.05, max_iter=140, max_leaf_nodes=15, min_samples_leaf=40,
                l2_regularization=2.0, early_stopping=False, random_state=SEED + horizon + boot,
            ).fit(x[idx], y[idx], sample_weight=weights[idx] * class_weight[y[idx]])
            risk_models[horizon].append(clf)

    for horizon in HORIZONS:
        y_diag = data.loc[diag_mask, f"risk_label_h{horizon}"].to_numpy(dtype=np.int64)
        diagnostic_clf = HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=140, max_leaf_nodes=15, min_samples_leaf=40,
            l2_regularization=2.0, early_stopping=False, random_state=SEED + 1000 + horizon,
        ).fit(x[train_mask], data.loc[train_mask, f"risk_label_h{horizon}"].to_numpy(dtype=np.int64), sample_weight=weights[train_mask])
        p = _aligned_proba(diagnostic_clf, x[diag_mask])
        diagnostics[f"risk_h{horizon}"] = {
            "class_counts": np.bincount(data[f"risk_label_h{horizon}"].to_numpy(dtype=np.int64), minlength=3).tolist(),
            "sl_brier": float(brier_score_loss((y_diag == 2).astype(np.int64), p[:, 2])),
            "tp_brier": float(brier_score_loss((y_diag == 1).astype(np.int64), p[:, 1])),
        }
    return {
        "kind": "censored_stopping_value", "feature_columns": feature_cols,
        "advantage_models": advantage_models, "risk_models": risk_models,
        "overprediction_residuals": np.asarray(overprediction, dtype=np.float64),
        "horizons": HORIZONS, "risk_classes": ("neither", "take_profit", "stop_loss"),
    }, diagnostics


def _predict_path(bundle: dict[str, Any], feature_rows: list[dict[str, float]], confidence: float) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[col]) for col in bundle["feature_columns"]] for row in feature_rows], dtype=np.float64)
    advantage = np.min(np.vstack([model.predict(x) for model in bundle["advantage_models"]]), axis=0)
    conformal_penalty = max(0.0, float(np.quantile(bundle["overprediction_residuals"], confidence)))
    advantage_lcb = advantage - conformal_penalty
    sl_probability_lcb = np.min(
        np.vstack([_aligned_proba(model, x)[:, 2] for model in bundle["risk_models"][96]]), axis=0
    )
    return advantage_lcb, sl_probability_lcb


def _position_feature_rows(
    frame: pd.DataFrame, arrays: dict[str, np.ndarray], components: dict[str, dict[str, Any]],
    option: dict[str, Any], *, end_i: int,
) -> tuple[list[int], list[dict[str, float]]]:
    indices: list[int] = []
    rows: list[dict[str, float]] = []
    mfe = 0.0
    mae = 0.0
    comp = components[str(option["source_component"])]
    for i in range(int(option["entry_i"]), end_i + 1):
        move, best, worst = _bar_moves(arrays, i, side=int(option["side"]), entry_price=float(option["entry_price"]))
        mfe = max(mfe, best)
        mae = min(mae, worst)
        row = hazard._feature_row(
            comp, row_i=i, side=int(option["side"]), hold=i - int(option["entry_i"]),
            move=move, mfe=mfe, mae=mae, notional=float(option["notional"]),
            leverage=float(option["leverage"]), take_profit=float(option["take_profit"]),
            stop_loss=float(option["stop_loss"]),
        )
        for name, source_id in SOURCE_IDS.items():
            row[f"component_{name}"] = float(source_id == SOURCE_IDS[str(option["source_component"])])
        indices.append(i)
        rows.append(row)
        if option["stop_loss"] > 0.0 and worst <= -abs(float(option["stop_loss"])):
            break
        if option["take_profit"] > 0.0 and best >= float(option["take_profit"]):
            break
    return indices, rows


def replay(
    frame: pd.DataFrame, components: dict[str, dict[str, Any]], bundle: dict[str, Any] | None,
    *, confidence: float, advantage_buffer: float, sl_probability_min: float, persistence: int,
    cost_mult: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = _arrays(frame)
    fee, slip = hazard.omega._load_fee_slip()
    fee_eff, slip_eff = float(fee) * cost_mult, float(slip) * cost_mult
    active_masks = {name: hazard.omega._active(comp["dec"]) for name, comp in components.items()}
    cash = 1.0
    peak = 1.0
    close_mdd = 0.0
    rows: list[dict[str, Any]] = []
    reasons: Counter[str] = Counter()
    i = 0
    last_i = len(frame) - 2
    while i <= last_i:
        option = _entry_option(components, active_masks, arrays, i, fee_eff=fee_eff, slip_eff=slip_eff)
        if option is None:
            i += 1
            continue
        entry_equity = cash
        cash *= 1.0 - float(option["entry_fee"]) * float(option["notional"])
        indices, feature_rows = _position_feature_rows(frame, arrays, components, option, end_i=last_i)
        if bundle is None:
            advantage_lcb = np.full(len(indices), -np.inf)
            sl_lcb = np.zeros(len(indices))
        else:
            advantage_lcb, sl_lcb = _predict_path(bundle, feature_rows, confidence)
        streak = 0
        mfe = 0.0
        mae = 0.0
        reason = "end_censored"
        exit_signal_i = indices[-1]
        decision_advantage = float("nan")
        decision_sl_probability = float("nan")
        for path_idx, state_i in enumerate(indices):
            move, best, worst = _bar_moves(arrays, state_i, side=int(option["side"]), entry_price=float(option["entry_price"]))
            mfe = max(mfe, best)
            mae = min(mae, worst)
            equity = cash * (1.0 + move * float(option["notional"]))
            peak = max(peak, equity)
            close_mdd = min(close_mdd, equity / max(peak, 1.0e-12) - 1.0)
            if option["stop_loss"] > 0.0 and worst <= -abs(float(option["stop_loss"])):
                reason, exit_signal_i = "stop_loss", state_i
                break
            if option["take_profit"] > 0.0 and best >= float(option["take_profit"]):
                reason, exit_signal_i = "take_profit", state_i
                break
            hit = bool(advantage_lcb[path_idx] > advantage_buffer and sl_lcb[path_idx] >= sl_probability_min)
            streak = streak + 1 if hit else 0
            if bundle is not None and streak >= persistence:
                reason, exit_signal_i = "stopping_value", state_i
                decision_advantage = float(advantage_lcb[path_idx])
                decision_sl_probability = float(sl_lcb[path_idx])
                break
        filled, exit_price, exit_fee, exit_route = hazard.omega._try_execution(
            arrays, exit_signal_i, int(option["side"]), entry=False, fee_base=fee_eff, slip_base=slip_eff
        )
        if not filled:
            raise RuntimeError("exit execution unexpectedly missed")
        raw_exit = _raw_move(float(exit_price), side=int(option["side"]), entry_price=float(option["entry_price"]))
        before_exit = cash
        cash *= 1.0 + raw_exit * float(option["notional"])
        cash -= before_exit * float(exit_fee) * float(option["notional"])
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
        reasons[reason] += 1
        rows.append({
            "entry_signal_i": option["signal_i"], "entry_i": option["entry_i"], "exit_i": exit_signal_i,
            "entry_timestamp": str(frame["timestamp"].iloc[int(option["signal_i"])]),
            "exit_timestamp": str(frame["timestamp"].iloc[exit_signal_i]),
            "side": option["side"], "source_component": option["source_component"], "reason": reason,
            "trade_return": trade_return, "net_per_notional": trade_return / max(float(option["notional"]), 1.0e-12),
            "mae_price_move": mae, "mfe_price_move": mfe, "notional": option["notional"],
            "margin_fraction": option["margin_fraction"], "leverage": option["leverage"],
            "entry_chop_probability": option["chop_probability"], "entry_sizing_multiplier": option["sizing_multiplier"],
            "advantage_lcb": decision_advantage, "sl_probability_lcb_h96": decision_sl_probability,
            "entry_route": option["entry_route"], "exit_route": exit_route,
        })
        i = exit_signal_i + 1
    ledger = pd.DataFrame(rows)
    if ledger.empty:
        raise RuntimeError("empty replay ledger")
    returns = ledger["trade_return"].to_numpy(dtype=np.float64)
    realized_mdd = composition._realized_mdd(returns)
    monthly = []
    month_frame = ledger.assign(month=pd.to_datetime(ledger["exit_timestamp"]).dt.to_period("M").astype(str))
    for month, group in month_frame.groupby("month", sort=True):
        month_returns = group["trade_return"].to_numpy(dtype=np.float64)
        monthly.append({
            "month": month, "pnl": float((np.prod(1.0 + month_returns) - 1.0) * 100.0),
            "realized_mdd": composition._realized_mdd(month_returns), "trades": int(len(group)),
        })
    metrics = {
        "pnl": float((cash - 1.0) * 100.0), "close_mark_to_market_mdd": float(close_mdd * 100.0),
        "realized_mdd": realized_mdd, "trades": int(len(ledger)), "wr": float(np.mean(returns > 0.0)),
        "exit_reasons": dict(reasons), "avg_notional": float(ledger["notional"].mean()), "monthly": monthly,
    }
    return metrics, ledger


def _prepare_split(start: str, end: str, *, base_csv: Path, wide24_csv: Path, prediction_split: str, oof: bool):
    frame = sweep.load_frame(start, end, base_csv=base_csv, wide24_csv=wide24_csv)
    common_timestamps = set(frame["timestamp"])
    for name, cfg in sweep.COMPONENTS.items():
        prediction_path = hazard._prediction_path(prediction_split, name, cfg)
        prediction_timestamps = set(
            pd.to_datetime(pd.read_csv(prediction_path, usecols=["timestamp"])["timestamp"], errors="raise")
        )
        common_timestamps.intersection_update(prediction_timestamps)
    frame = frame[frame["timestamp"].isin(common_timestamps)].reset_index(drop=True)
    if frame.empty:
        raise RuntimeError(f"{prediction_split}: no common frame/prediction timestamps")
    components = {
        name: hazard.prepare_split(
            name, cfg, frame, hazard._prediction_path(prediction_split, name, cfg), oof=oof, pre_quality=False
        )
        for name, cfg in sweep.COMPONENTS.items()
    }
    return frame, components


def _passes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    pnl_floor = 0.90 * baseline["pnl"] if baseline["pnl"] >= 0.0 else 1.10 * baseline["pnl"]
    return bool(
        candidate["pnl"] >= pnl_floor
        and candidate["close_mark_to_market_mdd"] >= baseline["close_mark_to_market_mdd"]
        and candidate["realized_mdd"] >= baseline["realized_mdd"]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale_name in (
        "validation_selected_ledger.csv",
        "oos_live_baseline_ledger.csv",
        "oos_candidate_ledger.csv",
        "extension_live_baseline_ledger.csv",
        "extension_candidate_ledger.csv",
    ):
        (OUT_DIR / stale_name).unlink(missing_ok=True)
    print("stage=prepare_train", flush=True)
    train_frame, train_components = _prepare_split(
        TRAIN_START, TRAIN_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025,
        prediction_split="train", oof=True,
    )
    print("stage=build_reentry_aware_dataset", flush=True)
    dataset, dataset_diagnostics = build_router_dataset(train_frame, train_components)
    dataset.to_csv(OUT_DIR / "train_live_router_stopping_dataset.csv.gz", index=False, compression="gzip")
    print(f"stage=fit rows={len(dataset)} episodes={dataset['episode_id'].nunique()}", flush=True)
    bundle, fit_diagnostics = fit_bundle(dataset)
    with (OUT_DIR / "model.pkl").open("wb") as handle:
        pickle.dump(bundle, handle)

    print("stage=prepare_validation", flush=True)
    val_frame, val_components = _prepare_split(
        sweep.VAL_START, sweep.VAL_END, base_csv=sweep.BASE_2025, wide24_csv=sweep.WIDE24_2025,
        prediction_split="validation", oof=True,
    )
    baseline, baseline_ledger = replay(
        val_frame, val_components, None, confidence=0.9, advantage_buffer=1.0,
        sl_probability_min=1.0, persistence=1, cost_mult=1.0,
    )
    baseline_ledger.to_csv(OUT_DIR / "validation_live_baseline_ledger.csv", index=False)
    ranking_rows: list[dict[str, Any]] = []
    ledgers: dict[tuple[float, float, float, int], pd.DataFrame] = {}
    for confidence in (0.80, 0.90):
        for advantage_buffer in (0.0, 0.001):
            for sl_probability_min in (0.50, 0.60):
                for persistence in (1, 3):
                    print(
                        f"stage=validation confidence={confidence} buffer={advantage_buffer} "
                        f"psl={sl_probability_min} persistence={persistence}", flush=True,
                    )
                    metrics, ledger = replay(
                        val_frame, val_components, bundle, confidence=confidence,
                        advantage_buffer=advantage_buffer, sl_probability_min=sl_probability_min,
                        persistence=persistence, cost_mult=1.0,
                    )
                    key = (confidence, advantage_buffer, sl_probability_min, persistence)
                    ledgers[key] = ledger
                    ranking_rows.append({
                        "confidence": confidence, "advantage_buffer": advantage_buffer,
                        "sl_probability_min": sl_probability_min, "persistence": persistence,
                        **metrics,
                        "model_exit_count": int(metrics["exit_reasons"].get("stopping_value", 0)),
                        "passes_dev_gate": bool(
                            metrics["exit_reasons"].get("stopping_value", 0) > 0
                            and _passes(metrics, baseline)
                        ),
                    })
    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["passes_dev_gate", "close_mark_to_market_mdd", "realized_mdd", "pnl"],
        ascending=[False, False, False, False],
    )
    ranking.to_csv(OUT_DIR / "validation_development_ranking.csv", index=False)
    passing = [row for row in ranking_rows if row["passes_dev_gate"]]
    selected = max(
        passing, key=lambda row: (row["close_mark_to_market_mdd"], row["realized_mdd"], row["pnl"])
    ) if passing else None

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "development_rejected" if selected is None else "development_selected_diagnostic_only",
        "deployment_verdict": "do_not_apply_to_live",
        "selected": selected,
        "selection_diagnostics": {
            "max_model_exit_count_across_safe_grid": int(max(row["model_exit_count"] for row in ranking_rows)),
            "all_safe_candidates_abstained": bool(
                max(row["model_exit_count"] for row in ranking_rows) == 0
            ),
        },
        "dataset_diagnostics": dataset_diagnostics, "fit_diagnostics": fit_diagnostics,
        "validation": {"live_baseline": baseline, "selected": selected},
        "protocol": {
            "train": [TRAIN_START, TRAIN_END], "validation": [sweep.VAL_START, sweep.VAL_END],
            "fresh_forward_bar_by_bar": True, "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
            "future_rows_used_for_training_labels_only": True,
            "barrier_order": "intrabar_stop_loss_first_then_take_profit_then_model",
            "execution": "signal_on_completed_bar_then_next_open_limit_or_market_fallback",
            "live_contract": {
                "duration_gate_off": True, "entry_notional_multiplier": composition.ENTRY_NOTIONAL_MULTIPLIER,
                "max_entry_notional": composition.MAX_ENTRY_NOTIONAL,
                "chop_soft_size_threshold": composition.CHOP_SOFT_SIZE_THRESHOLD,
            },
            "evaluation_limitation": "All evaluation intervals were previously consumed; development diagnostic only, never promotion evidence.",
        },
    }
    if selected is not None:
        key = (
            float(selected["confidence"]), float(selected["advantage_buffer"]),
            float(selected["sl_probability_min"]), int(selected["persistence"]),
        )
        ledgers[key].to_csv(OUT_DIR / "validation_selected_ledger.csv", index=False)
        evaluations = (
            ("oos", sweep.OOS_START, sweep.OOS_END, sweep.BASE_2026, sweep.WIDE24_2026, "oos", False),
            ("extension", EXTENSION_START, EXTENSION_END, sweep.BASE_2026, sweep.WIDE24_2026, "oos", False),
        )
        for name, start, end, base_csv, wide24_csv, prediction_split, oof in evaluations:
            print(f"stage=diagnostic split={name}", flush=True)
            frame, components = _prepare_split(
                start, end, base_csv=base_csv, wide24_csv=wide24_csv,
                prediction_split=prediction_split, oof=oof,
            )
            split_result: dict[str, Any] = {}
            for cost_mult in (1.0, 2.0, 3.0):
                base_metrics, base_ledger = replay(
                    frame, components, None, confidence=0.9, advantage_buffer=1.0,
                    sl_probability_min=1.0, persistence=1, cost_mult=cost_mult,
                )
                candidate_metrics, candidate_ledger = replay(
                    frame, components, bundle, confidence=key[0], advantage_buffer=key[1],
                    sl_probability_min=key[2], persistence=key[3], cost_mult=cost_mult,
                )
                tag = f"cost{int(cost_mult)}"
                split_result[tag] = {
                    "live_baseline": base_metrics, "candidate": candidate_metrics,
                    "passes_same_cost_baseline": _passes(candidate_metrics, base_metrics),
                }
                if cost_mult == 1.0:
                    base_ledger.to_csv(OUT_DIR / f"{name}_live_baseline_ledger.csv", index=False)
                    candidate_ledger.to_csv(OUT_DIR / f"{name}_candidate_ledger.csv", index=False)
            report[f"{name}_diagnostic"] = split_result
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps({
        "report": str(OUT_DIR / "report.json"), "status": report["status"],
        "selected": selected,
    }, indent=2, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
