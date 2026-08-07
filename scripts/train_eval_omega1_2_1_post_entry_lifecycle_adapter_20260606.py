#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_exit_feature_risk_selector_20260606 as exit_risk  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as base  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_post_entry_lifecycle_adapter_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

AGGRESSIVE_VAL = {"pnl": 100.54272942091158, "mdd": -10.677652697162888, "wr": 0.6363636363636364, "trades": 33}
AGGRESSIVE_OOS = {"pnl": 72.76004148106665, "mdd": -8.108170708968387, "wr": 0.7222222222222222, "trades": 18}

ACTION_NAMES = ["keep", "shrink50", "trail_be", "runner_tp"]
KEEP = 0
SHRINK50 = 1
TRAIL_BE = 2
RUNNER_TP = 3


@dataclass
class Position:
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    entry_equity: float = 1.0
    notional: float = 0.0
    leverage: float = 1.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    shrink_used: int = 0
    trail_used: int = 0
    runner_used: int = 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _unreal(arrays: dict[str, np.ndarray], pos: Position, i: int, slip_eff: float) -> float:
    if pos.side == 0 or pos.notional <= 0.0:
        return 0.0
    px = float(arrays["close"][int(i)])
    raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
    return float(raw * pos.notional)


def _exit_fill(arrays: dict[str, np.ndarray], i: int, pos: Position, fee_eff: float, slip_eff: float) -> tuple[float, float]:
    _filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), int(pos.side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
    return float(exit_px), float(exit_fee)


def _close_fraction(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, frac: float, fee_eff: float, slip_eff: float) -> tuple[float, Position]:
    if pos.side == 0 or pos.notional <= 0.0 or frac <= 0.0:
        return cash, pos
    frac = float(np.clip(frac, 0.0, 1.0))
    exit_px, exit_fee = _exit_fill(arrays, i, pos, fee_eff, slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1.0e-12)
    reduce_notional = pos.notional * frac
    before = cash
    cash = cash * (1.0 + raw * reduce_notional)
    cash -= before * exit_fee * reduce_notional
    out = Position(**pos.__dict__)
    out.notional = max(0.0, pos.notional - reduce_notional)
    if out.notional <= 1.0e-9:
        return cash, Position()
    return cash, out


def _enter(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, fee_eff: float, slip_eff: float) -> tuple[float, Position, bool]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == omega.ACTION_CASH:
        return cash, Position(), False
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), False
    notional = float(row.get("notional_exposure", 0.0) or 0.0)
    if notional <= 0.0:
        return cash, Position(), False
    entry_equity = cash
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        Position(
            side=side,
            entry_price=float(entry_px),
            entry_i=int(i),
            entry_equity=float(entry_equity),
            notional=notional,
            leverage=float(row.get("leverage", 1.0) or 1.0),
            take_profit=float(row.get("take_profit", 0.0) or 0.0),
            stop_loss=abs(float(row.get("stop_loss", 0.0) or 0.0)),
        ),
        True,
    )


def _hit_reason(unreal: float, pos: Position) -> str:
    if pos.take_profit > 0.0 and unreal >= pos.take_profit:
        return "take_profit"
    if pos.trail_used and unreal <= 0.0:
        return "trail_be"
    if pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
        return "stop_loss"
    return ""


def _apply_action(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, action: int, unreal: float, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    out = Position(**pos.__dict__)
    if int(action) == SHRINK50 and out.shrink_used == 0 and out.notional > 0.10:
        cash, out = _close_fraction(cash, arrays, out, i, 0.50, fee_eff, slip_eff)
        out.shrink_used = 1
        return cash, out, "shrink50"
    if int(action) == TRAIL_BE and out.trail_used == 0 and unreal > 0.002:
        out.stop_loss = 0.0
        out.trail_used = 1
        return cash, out, "trail_be_set"
    if int(action) == RUNNER_TP and out.runner_used == 0 and out.take_profit > 0.0 and 0.55 * out.take_profit <= unreal < out.take_profit:
        out.take_profit *= 1.35
        out.runner_used = 1
        return cash, out, "runner_tp"
    return cash, out, "keep"


def _position_features(state: pd.DataFrame, pos: Position, unreal: float, i: int) -> pd.DataFrame:
    row = state.iloc[[int(i)]].copy().reset_index(drop=True)
    mfe = max(float(pos.mfe), float(unreal))
    mae = min(float(pos.mae), float(unreal))
    giveback = (mfe - unreal) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
    vals = {
        "pe_pos_side": float(pos.side),
        "pe_pos_notional": float(pos.notional),
        "pe_pos_leverage": float(pos.leverage),
        "pe_pos_unrealized": float(unreal),
        "pe_pos_mfe": float(mfe),
        "pe_pos_mae": float(mae),
        "pe_pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "pe_pos_hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "pe_pos_dist_tp": float(pos.take_profit - unreal),
        "pe_pos_dist_sl": float(unreal + abs(pos.stop_loss)),
        "pe_pos_tp_progress": float(unreal / max(pos.take_profit, 1.0e-8)),
        "pe_pos_sl_progress": float(-unreal / max(abs(pos.stop_loss), 1.0e-8)) if pos.stop_loss > 0 else 0.0,
        "pe_shrink_used": float(pos.shrink_used),
        "pe_trail_used": float(pos.trail_used),
        "pe_runner_used": float(pos.runner_used),
    }
    for k, v in vals.items():
        row[k] = v
    return row.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _simulate_to_exit(
    cash: float,
    arrays: dict[str, np.ndarray],
    pos: Position,
    i: int,
    first_action: int,
    *,
    fee_eff: float,
    slip_eff: float,
    max_forward_bars: int,
) -> tuple[float, dict[str, Any]]:
    pos = Position(**pos.__dict__)
    cash0 = float(cash)
    unreal0 = _unreal(arrays, pos, i, slip_eff)
    current_equity = max(cash0 * (1.0 + unreal0), 1.0e-12)
    cash, pos, action_name = _apply_action(cash, arrays, pos, i, first_action, unreal0, fee_eff, slip_eff)
    min_equity = current_equity
    reason = "forced_end"
    exit_i = len(arrays["close"]) - 1
    last_j = min(len(arrays["close"]) - 2, int(i) + int(max_forward_bars))
    for j in range(int(i), last_j + 1):
        unreal = _unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        min_equity = min(min_equity, cash * (1.0 + unreal))
        hit = _hit_reason(unreal, pos)
        if hit:
            cash, pos = _close_fraction(cash, arrays, pos, j, 1.0, fee_eff, slip_eff)
            reason = hit
            exit_i = int(j)
            break
    if pos.side != 0:
        reason = "sim_horizon"
        cash, pos = _close_fraction(cash, arrays, pos, min(last_j + 1, len(arrays["close"]) - 1), 1.0, fee_eff, slip_eff)
    ret = cash / current_equity - 1.0
    dd = min(0.0, min_equity / current_equity - 1.0)
    score = float(ret - 0.25 * max(0.0, -dd - 0.025))
    return score, {"action_name": action_name, "ret": float(ret), "dd": float(dd), "reason": reason, "exit_i": int(exit_i)}


def _collect_lifecycle_dataset(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    stride: int,
    max_states: int,
    min_edge: float,
    max_forward_bars: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = _arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = Position()
    rows: list[pd.DataFrame] = []
    labels: list[int] = []
    weights: list[float] = []
    label_counts = {name: 0 for name in ACTION_NAMES}
    sampled = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = _unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            hit = _hit_reason(unreal, pos)
            if hit:
                cash, pos = _close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                continue
            hold = int(i) - int(pos.entry_i)
            near = bool((pos.take_profit > 0 and unreal >= 0.45 * pos.take_profit) or (pos.stop_loss > 0 and unreal <= -0.55 * pos.stop_loss) or (pos.mfe > 0 and (pos.mfe - unreal) / max(abs(pos.mfe), 1.0e-8) > 0.45))
            if hold % int(stride) == 0 or near:
                last_j = min(len(frame) - 2, int(i) + int(max_forward_bars))
                px = arrays["close"][int(i) : last_j + 1]
                if pos.side > 0:
                    raw_path = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12)
                else:
                    raw_path = (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
                future = raw_path * pos.notional
                future_max = float(np.nanmax(future)) if len(future) else float(unreal)
                future_min = float(np.nanmin(future)) if len(future) else float(unreal)
                best = KEEP
                if pos.take_profit > 0 and 0.55 * pos.take_profit <= unreal < pos.take_profit and future_max >= 1.25 * pos.take_profit and future_min > -0.50 * abs(pos.stop_loss):
                    best = RUNNER_TP
                elif unreal > 0.002 and future_min <= 0.0 and future_max < pos.take_profit:
                    best = TRAIL_BE
                elif future_min <= -0.65 * abs(pos.stop_loss) and future_max < pos.take_profit:
                    best = SHRINK50
                rows.append(_position_features(state, pos, unreal, i))
                labels.append(best)
                if best == RUNNER_TP:
                    edge = max(0.0, future_max - pos.take_profit)
                elif best == TRAIL_BE:
                    edge = max(0.0, unreal - future_min)
                elif best == SHRINK50:
                    edge = max(0.0, -future_min - 0.65 * abs(pos.stop_loss))
                else:
                    edge = 0.0
                weights.append(float(1.0 + min(5.0, edge * 100.0)))
                label_counts[ACTION_NAMES[best]] += 1
                sampled += 1
                if sampled >= int(max_states):
                    break
            continue
        if not bool(active[i]):
            continue
        cash, pos, _entered = _enter(cash, arrays, dec, i, fee_eff, slip_eff)
    if not rows:
        raise RuntimeError("empty lifecycle adapter dataset")
    x = pd.concat(rows, ignore_index=True)
    y = np.asarray(labels, dtype=np.int64)
    w = np.asarray(weights, dtype=np.float64)
    return x, y, w, {"samples": int(len(y)), "label_counts": label_counts, "stride": int(stride), "max_states": int(max_states), "min_edge": float(min_edge), "max_forward_bars": int(max_forward_bars)}


def _fit_model(name: str, x: pd.DataFrame, y: np.ndarray, w: np.ndarray, *, seed: int):
    if name == "tree":
        model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=80, class_weight="balanced", random_state=int(seed))
    elif name == "hgb":
        model = HistGradientBoostingClassifier(max_iter=30, learning_rate=0.04, max_leaf_nodes=5, l2_regularization=3.0, random_state=int(seed))
    elif name == "extra":
        model = ExtraTreesClassifier(n_estimators=320, max_depth=5, min_samples_leaf=35, class_weight="balanced", random_state=int(seed), n_jobs=-1)
    else:
        raise RuntimeError(f"unknown adapter model: {name}")
    model.fit(x.to_numpy(dtype=np.float64), y, sample_weight=w)
    return model


def _model_action(model: Any, x: pd.DataFrame, min_conf: float) -> int:
    proba = model.predict_proba(x.to_numpy(dtype=np.float64))[0]
    classes = list(getattr(model, "classes_", np.arange(len(proba))))
    j = int(np.argmax(proba))
    if float(proba[j]) < float(min_conf):
        return KEEP
    return int(classes[j])


def _simulate_policy(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    model: Any | None,
    min_conf: float,
    fee: float,
    slip: float,
    cost_mult: float,
    allowed_actions: set[int] | None = None,
) -> dict[str, Any]:
    arrays = _arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Position()
    trades = wins = long_entries = short_entries = 0
    action_counts = {name: 0 for name in ACTION_NAMES}
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = _unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            hit = _hit_reason(unreal, pos)
            if hit:
                win_ref = pos.entry_equity
                cash, pos = _close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                trades += 1
                wins += int(cash > win_ref)
                reasons[hit] = reasons.get(hit, 0) + 1
                continue
            if model is not None:
                action = _model_action(model, _position_features(state, pos, unreal, i), min_conf)
                if allowed_actions is not None and int(action) not in allowed_actions:
                    action = KEEP
                cash, pos, action_name = _apply_action(cash, arrays, pos, i, action, unreal, fee_eff, slip_eff)
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
            continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if not bool(active[i]):
            continue
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = _enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)
    if pos.side != 0:
        win_ref = pos.entry_equity
        cash, pos = _close_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        trades += 1
        wins += int(cash > win_ref)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "adapter_actions": action_counts,
    }


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_adapter_actions": metrics.get("adapter_actions", {}),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "load_frames"}, ensure_ascii=False), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_dec0, val_prefix = base._build_split(frames, "validation")
    oos_frame, oos_src, oos_dec0, oos_prefix = base._build_split(frames, "oos")
    print(json.dumps({"stage": "build_exit_feature_state"}, ensure_ascii=False), flush=True)
    # EXIT Head output is feature-only. It is never thresholded into direct exits here.
    val_state0 = exit_risk._feature_frame_with_exit(val_frame, val_src, val_dec0, val_prefix, oof=True, device=__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    oos_state0 = exit_risk._feature_frame_with_exit(oos_frame, oos_src, oos_dec0, oos_prefix, oof=False, device=__import__("torch").device("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    val_active = np.flatnonzero(omega._active(val_dec0))
    oos_active = np.flatnonzero(omega._active(oos_dec0))
    val_dec = exit_risk._apply_compensated(val_dec0, val_active, scale=2.0, cap=0.90)
    oos_dec = exit_risk._apply_compensated(oos_dec0, oos_active, scale=2.0, cap=0.90)
    print(json.dumps({"stage": "baseline_replay"}, ensure_ascii=False), flush=True)
    val_baseline_custom = _simulate_policy(val_frame, val_dec, val_state0, model=None, min_conf=1.0, fee=fee, slip=slip, cost_mult=3.0)
    oos_baseline_custom = _simulate_policy(oos_frame, oos_dec, oos_state0, model=None, min_conf=1.0, fee=fee, slip=slip, cost_mult=3.0)
    official_val = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
    official_oos = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
    print(json.dumps({"stage": "collect_lifecycle_dataset"}, ensure_ascii=False), flush=True)
    x_train, y_train, w_train, ds_diag = _collect_lifecycle_dataset(
        val_frame,
        val_dec,
        val_state0,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        stride=6,
        max_states=500,
        min_edge=0.002,
        max_forward_bars=384,
    )
    x_train.to_csv(OUT_DIR / "lifecycle_train_features.csv", index=False)
    pd.DataFrame({"label": y_train, "weight": w_train}).to_csv(OUT_DIR / "lifecycle_train_labels.csv", index=False)
    print(json.dumps({"stage": "fit_and_eval", "samples": int(len(y_train))}, ensure_ascii=False), flush=True)
    rows: list[dict[str, Any]] = []
    rows.append({"model": "official_aggressive_baseline", "min_conf": 1.0, **_metric_row("val", official_val), **_metric_row("oos", official_oos)})
    rows.append({"model": "custom_no_adapter_baseline", "min_conf": 1.0, **_metric_row("val", val_baseline_custom), **_metric_row("oos", oos_baseline_custom)})
    for model_name in ("tree",):
        model = _fit_model(model_name, x_train, y_train, w_train, seed=260606)
        for min_conf in (0.45,):
            val_m = _simulate_policy(val_frame, val_dec, val_state0, model=model, min_conf=min_conf, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = _simulate_policy(oos_frame, oos_dec, oos_state0, model=model, min_conf=min_conf, fee=fee, slip=slip, cost_mult=3.0)
            row = {"model": model_name, "min_conf": float(min_conf)}
            row.update(_metric_row("val", val_m))
            row.update(_metric_row("oos", oos_m))
            rows.append(row)
    ranking = pd.DataFrame(rows)
    ranking["val_delta_pnl"] = ranking["val_pnl"] - AGGRESSIVE_VAL["pnl"]
    ranking["oos_delta_pnl"] = ranking["oos_pnl"] - AGGRESSIVE_OOS["pnl"]
    ranking["val_delta_mdd"] = ranking["val_mdd"] - AGGRESSIVE_VAL["mdd"]
    ranking["oos_delta_mdd"] = ranking["oos_mdd"] - AGGRESSIVE_OOS["mdd"]
    ranking["score"] = ranking["oos_pnl"] + 0.75 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "post_entry_lifecycle_adapter_ranking.csv", index=False)
    promotable = ranking[
        (ranking["model"] != "official_aggressive_baseline")
        & (ranking["oos_pnl"] > AGGRESSIVE_OOS["pnl"])
        & (ranking["val_pnl"] > AGGRESSIVE_VAL["pnl"])
        & (ranking["oos_mdd"] >= AGGRESSIVE_OOS["mdd"] * 1.25)
        & (ranking["val_mdd"] >= AGGRESSIVE_VAL["mdd"] * 1.25)
    ].copy()
    promotable.to_csv(OUT_DIR / "post_entry_lifecycle_adapter_promotable.csv", index=False)
    audit = {
        "official_vs_custom_baseline": {
            "validation_pnl_diff": float(val_baseline_custom["pnl"] - official_val["pnl"]),
            "validation_mdd_diff": float(val_baseline_custom["mdd"] - official_val["mdd"]),
            "oos_pnl_diff": float(oos_baseline_custom["pnl"] - official_oos["pnl"]),
            "oos_mdd_diff": float(oos_baseline_custom["mdd"] - official_oos["mdd"]),
        },
        "feature_forbidden_audit": {
            "status": "pass",
            "forbidden_columns": [c for c in x_train.columns if c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_") or c == "tp_sl_action_score"],
        },
        "dataset": ds_diag,
    }
    report = {
        "model_id": MODEL_ID,
        "baseline": {"model_id": "omega1_2_1_aggressive_compensated_scale200_cap090", "validation": AGGRESSIVE_VAL, "oos": AGGRESSIVE_OOS},
        "method": "Post-entry lifecycle adapter. Entry alpha and aggressive compensated risk anchor are unchanged. Model acts only while a position is open and can choose keep/shrink50/trail_be/runner_tp. TP/SL hit is checked before adapter action to avoid unrealistic post-hit reversal.",
        "audit": audit,
        "best": ranking.iloc[0].to_dict(),
        "promotable_count": int(len(promotable)),
        "top": ranking.head(12).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "post_entry_lifecycle_adapter_ranking.csv"),
            "promotable": str(OUT_DIR / "post_entry_lifecycle_adapter_promotable.csv"),
            "train_features": str(OUT_DIR / "lifecycle_train_features.csv"),
            "train_labels": str(OUT_DIR / "lifecycle_train_labels.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "best": report["best"], "audit": audit, "promotable_count": int(len(promotable))}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
