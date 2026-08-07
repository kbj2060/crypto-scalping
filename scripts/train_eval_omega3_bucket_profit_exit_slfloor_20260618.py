#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega3_margin_cap1_bucket_20260618 as exp  # noqa: E402


MODEL_ID = "omega3_bucket_profit_exit_slfloor_fixedlev2_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ENTRY_CANDIDATE = {
    "cal_q": 0.80,
    "ev_min": 0.004,
    "utility_cfg_id": 0,
    "utility_min": -0.001,
    "margin_min": 0.0,
}
EXIT_THRESHOLDS = (0.45, 0.50, 0.55, 0.60, 0.65, 0.70)
SL_PRICE_MOVE_FLOOR = 0.025
PROFIT_EXIT_MIN_PROGRESS = 0.60


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


def _risk_predictions(x_val: pd.DataFrame, x_oos: pd.DataFrame, risk_labels: pd.DataFrame) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    val_tp_l, val_tp_s, oos_tp_l, oos_tp_s, tp_diag = exp._fit_predict_risk_head(
        x_val,
        x_oos,
        risk_labels,
        "long_tp_price_move",
        "short_tp_price_move",
        seed=282101,
        clip_min=float(exp.PARENT_RISK_TP_MIN),
        clip_max=float(exp.PARENT_RISK_TP_MAX),
    )
    val_sl_l, val_sl_s, oos_sl_l, oos_sl_s, sl_diag = exp._fit_predict_risk_head(
        x_val,
        x_oos,
        risk_labels,
        "long_sl_price_move",
        "short_sl_price_move",
        seed=282201,
        clip_min=float(exp.PARENT_RISK_SL_MIN),
        clip_max=float(exp.PARENT_RISK_SL_MAX),
    )
    val_margin_l, val_margin_s, oos_margin_l, oos_margin_s, margin_diag = exp._fit_predict_margin_bucket_head(
        x_val,
        x_oos,
        risk_labels,
        "long_margin_fraction",
        "short_margin_fraction",
        seed=282401,
    )
    return (
        {
            "long_tp_price_move": val_tp_l,
            "short_tp_price_move": val_tp_s,
            "long_sl_price_move": val_sl_l,
            "short_sl_price_move": val_sl_s,
            "long_margin_fraction": val_margin_l,
            "short_margin_fraction": val_margin_s,
        },
        {
            "long_tp_price_move": oos_tp_l,
            "short_tp_price_move": oos_tp_s,
            "long_sl_price_move": oos_sl_l,
            "short_sl_price_move": oos_sl_s,
            "long_margin_fraction": oos_margin_l,
            "short_margin_fraction": oos_margin_s,
        },
        {"tp_price_move": tp_diag, "sl_price_move": sl_diag, "margin_bucket": margin_diag},
    )


def _apply_sl_floor(risk_pred: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {str(k): np.asarray(v, dtype=np.float64).copy() for k, v in risk_pred.items()}
    out["long_sl_price_move"] = np.maximum(out["long_sl_price_move"], float(SL_PRICE_MOVE_FLOOR))
    out["short_sl_price_move"] = np.maximum(out["short_sl_price_move"], float(SL_PRICE_MOVE_FLOOR))
    return out


def _entry_actions(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    path_labels: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    ev_labels, ev_diag = exp._utility_from_path_labels(
        path_labels,
        exp.RISK,
        {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0},
    )
    ev_vl, ev_vs, ev_ol, ev_os, ev_fit = exp._fit_predict_lower_bound(
        x_val,
        x_oos,
        ev_labels,
        "long_net",
        "short_net",
        seed=280000,
        cal_q=float(ENTRY_CANDIDATE["cal_q"]),
    )
    labels, utility_diag = exp._utility_from_path_labels(path_labels, exp.RISK, exp.UTILITY_CFGS[int(ENTRY_CANDIDATE["utility_cfg_id"])])
    uvl, uvs, uol, uos, utility_fit = exp._fit_predict_lower_bound(
        x_val,
        x_oos,
        labels,
        "long_utility",
        "short_utility",
        seed=281000 + int(ENTRY_CANDIDATE["utility_cfg_id"]) * 100,
        cal_q=0.50,
    )
    val_ev_a, val_ev_c = exp._actions_from_scores(ev_vl, ev_vs, float(ENTRY_CANDIDATE["ev_min"]))
    oos_ev_a, oos_ev_c = exp._actions_from_scores(ev_ol, ev_os, float(ENTRY_CANDIDATE["ev_min"]))
    val_a, val_c, val_filter = exp._apply_agreement(
        val_ev_a,
        val_ev_c,
        uvl,
        uvs,
        utility_min=float(ENTRY_CANDIDATE["utility_min"]),
        margin_min=float(ENTRY_CANDIDATE["margin_min"]),
    )
    oos_a, oos_c, oos_filter = exp._apply_agreement(
        oos_ev_a,
        oos_ev_c,
        uol,
        uos,
        utility_min=float(ENTRY_CANDIDATE["utility_min"]),
        margin_min=float(ENTRY_CANDIDATE["margin_min"]),
    )
    return (
        val_a,
        val_c,
        oos_a,
        oos_c,
        {
            "ev_labels": ev_diag,
            "ev_fit": ev_fit,
            "utility_labels": utility_diag,
            "utility_fit": utility_fit,
            "filter": {"validation": val_filter, "oos": oos_filter},
        },
    )


def _position_features(
    arrays: dict[str, np.ndarray],
    pos: Any,
    i: int,
    active: np.ndarray,
    risk: Any,
    fee_eff: float,
    slip_eff: float,
) -> dict[str, float]:
    px = float(arrays["close"][i])
    raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
    unreal = float(raw * pos.notional)
    start = max(0, int(i) - 24)
    closes = np.asarray(arrays["close"][start : int(i) + 1], dtype=np.float64)
    ret1 = np.diff(closes) / np.maximum(closes[:-1], 1.0e-12) if len(closes) > 1 else np.asarray([0.0], dtype=np.float64)
    hold_bars = max(0, int(i) - int(pos.entry_i))
    tp_price_move = float(pos.take_profit) / max(float(pos.notional), 1.0e-12)
    sl_price_move = float(pos.stop_loss) / max(float(pos.notional), 1.0e-12)
    return {
        "side": float(pos.side),
        "raw_return": float(raw),
        "unreal": unreal,
        "abs_unreal": float(abs(unreal)),
        "tp_price_move": tp_price_move,
        "sl_price_move": sl_price_move,
        "distance_to_tp": float(tp_price_move - raw),
        "distance_to_sl": float(raw + sl_price_move),
        "progress_to_tp": float(raw / max(tp_price_move, 1.0e-12)),
        "progress_to_sl": float(max(0.0, -raw) / max(sl_price_move, 1.0e-12)),
        "notional": float(pos.notional),
        "leverage": float(pos.leverage),
        "margin_fraction": float(pos.notional) / max(float(pos.leverage), 1.0e-12),
        "hold_frac": float(hold_bars / max(int(pos.max_hold_bars), 1)),
        "hold_bars_tanh": float(np.tanh(hold_bars / 96.0)),
        "ret_sum_6": float(np.sum(ret1[-6:])) if len(ret1) else 0.0,
        "ret_sum_24": float(np.sum(ret1[-24:])) if len(ret1) else 0.0,
        "ret_vol_12": float(np.std(ret1[-12:])) if len(ret1) >= 2 else 0.0,
        "ret_vol_24": float(np.std(ret1[-24:])) if len(ret1) >= 2 else 0.0,
        "parent_active_now": 1.0 if bool(active[i]) else 0.0,
    }


def _label_exit_now(arrays: dict[str, np.ndarray], pos: Any, i: int, active: np.ndarray, fee_eff: float, slip_eff: float) -> int:
    if bool(active[i]):
        return 1
    px = float(arrays["close"][i])
    current_raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
    current_unreal = current_raw * float(pos.notional)
    if current_unreal <= 0.0:
        return 0
    end_i = min(len(arrays["close"]) - 2, int(pos.entry_i) + int(pos.max_hold_bars), int(i) + 48)
    future: list[float] = []
    for j in range(int(i) + 1, end_i + 1):
        px_j = float(arrays["close"][j])
        raw_j = (px_j * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px_j * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
        future.append(float(raw_j * pos.notional))
        if bool(active[j]):
            break
    if not future:
        return 1 if current_unreal > 0.0 else 0
    future_arr = np.asarray(future, dtype=np.float64)
    future_best = float(np.max(future_arr))
    future_worst = float(np.min(future_arr))
    giveback_risk = future_worst < current_unreal - 0.006
    no_upside = future_best < current_unreal + 0.002
    return int(giveback_risk and no_upside)


def _collect_exit_training(
    payload: dict[str, Any],
    risk_pred: dict[str, np.ndarray],
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = exp.sleeve._arrays(frame)
    active = exp.omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = exp.sleeve.Position()
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            risk = exp.sleeve.FallbackRisk("open", pos.take_profit, pos.stop_loss, pos.notional, pos.leverage, pos.max_hold_bars)
            feats = _position_features(arrays, pos, i, active, risk, fee_eff, slip_eff)
            if pos.sleeve == "fallback" and float(feats["unreal"]) > 0.0 and float(feats["progress_to_tp"]) >= float(PROFIT_EXIT_MIN_PROGRESS):
                rows.append(feats)
                labels.append(_label_exit_now(arrays, pos, i, active, fee_eff, slip_eff))
            if bool(active[i]) or int(i) - int(pos.entry_i) >= int(pos.max_hold_bars):
                cash, _win = exp.sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                pos = exp.sleeve.Position()
                continue
            if pos.sleeve == "fallback" and float(feats["unreal"]) <= -abs(float(pos.stop_loss)):
                cash, _win = exp.sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                pos = exp.sleeve.Position()
                continue
        if pos.side == 0 and not bool(active[i]):
            side_action = int(fallback_action[i]) if i < len(fallback_action) else exp.sleeve.ACTION_CASH
            conf = float(fallback_conf[i]) if i < len(fallback_conf) else 0.0
            if side_action in (exp.sleeve.ACTION_LONG, exp.sleeve.ACTION_SHORT) and conf >= 0.0:
                side = 1 if side_action == exp.sleeve.ACTION_LONG else -1
                risk = exp._risk_from_predictions(int(i), side, risk_pred, exp.RISK)
                cash, pos, _entered = exp.sleeve._open_position(cash, arrays, i, side, "fallback", risk, None, fee_eff, slip_eff)
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    diag = {
        "rows": int(len(x)),
        "positive": int(y.sum()) if len(y) else 0,
        "positive_rate": float(y.mean()) if len(y) else 0.0,
        "features": list(x.columns),
    }
    return x, y, diag


def _metrics_with_learned_exit(
    payload: dict[str, Any],
    risk_pred: dict[str, np.ndarray],
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    exit_model: HistGradientBoostingClassifier,
    exit_features: list[str],
    exit_threshold: float,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = exp.sleeve._arrays(frame)
    active = exp.omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = exp.sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = learned_exit_count = 0
    reasons: dict[str, int] = {}
    fallback_tp: list[float] = []
    fallback_sl: list[float] = []
    fallback_notional: list[float] = []
    fallback_leverage: list[float] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
            unreal = raw * pos.notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            if pos.sleeve == "primary":
                if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                    reason = "take_profit"
                elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                    reason = "stop_loss"
                elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                    reason = "max_hold"
            elif bool(active[i]):
                reason = "primary_takeover"
            elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                reason = "max_hold"
            elif unreal <= -abs(float(pos.stop_loss)):
                reason = "stop_loss"
            else:
                risk = exp.sleeve.FallbackRisk("open", pos.take_profit, pos.stop_loss, pos.notional, pos.leverage, pos.max_hold_bars)
                feats = _position_features(arrays, pos, i, active, risk, fee_eff, slip_eff)
                if float(feats["unreal"]) > 0.0 and float(feats["progress_to_tp"]) >= float(PROFIT_EXIT_MIN_PROGRESS):
                    x = pd.DataFrame([{c: float(feats.get(c, 0.0)) for c in exit_features}], columns=exit_features)
                    proba = exit_model.predict_proba(x.to_numpy(dtype=np.float64))[0]
                    classes = list(exit_model.classes_)
                    exit_prob = float(proba[classes.index(1)]) if 1 in classes else 0.0
                    if exit_prob >= float(exit_threshold):
                        reason = "profit_exit"
                        learned_exit_count += 1
            if reason:
                cash, win = exp.sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = exp.sleeve.Position()
            else:
                continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entered = exp.sleeve._open_position(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
        else:
            side_action = int(fallback_action[i]) if i < len(fallback_action) else exp.sleeve.ACTION_CASH
            conf = float(fallback_conf[i]) if i < len(fallback_conf) else 0.0
            if side_action in (exp.sleeve.ACTION_LONG, exp.sleeve.ACTION_SHORT) and conf >= 0.0:
                side = 1 if side_action == exp.sleeve.ACTION_LONG else -1
                row_risk = exp._risk_from_predictions(int(i), side, risk_pred, exp.RISK)
                cash, pos, entered = exp.sleeve._open_position(cash, arrays, i, side, "fallback", row_risk, None, fee_eff, slip_eff)
                if entered:
                    fallback_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
                    fallback_tp.append(float(row_risk.take_profit))
                    fallback_sl.append(float(row_risk.stop_loss))
                    fallback_notional.append(float(row_risk.notional))
                    fallback_leverage.append(float(row_risk.leverage))
    if pos.side != 0:
        cash, win = exp.sleeve._close_position(cash, arrays, pos, len(frame) - 2, fee_eff, slip_eff)
        trades += 1
        wins += int(win)
        reasons[f"{pos.sleeve}_forced_end"] = reasons.get(f"{pos.sleeve}_forced_end", 0) + 1
    days = max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1.0)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / days),
        "avg_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "avg_leverage": float(np.mean(fallback_leverage)) if fallback_leverage else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(reasons.get("fallback_primary_takeover", 0)),
        "learned_exit_count": int(learned_exit_count),
        "fallback_avg_tp": float(np.mean(fallback_tp)) if fallback_tp else 0.0,
        "fallback_avg_sl": float(np.mean(fallback_sl)) if fallback_sl else 0.0,
        "fallback_avg_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "fallback_avg_leverage": float(np.mean(fallback_leverage)) if fallback_leverage else 0.0,
        "fallback_avg_tp_price_move": float(np.mean(np.asarray(fallback_tp) / np.maximum(np.asarray(fallback_notional), 1.0e-12))) if fallback_tp else 0.0,
        "fallback_avg_sl_price_move": float(np.mean(np.asarray(fallback_sl) / np.maximum(np.asarray(fallback_notional), 1.0e-12))) if fallback_sl else 0.0,
    }


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_fallback_entries": int(metrics["fallback_entries"]),
        f"{prefix}_learned_exit_count": int(metrics.get("learned_exit_count", 0)),
        f"{prefix}_reasons": dict(metrics["exit_reasons"]),
        f"{prefix}_fallback_avg_tp": float(metrics.get("fallback_avg_tp", 0.0) or 0.0),
        f"{prefix}_fallback_avg_sl": float(metrics.get("fallback_avg_sl", 0.0) or 0.0),
        f"{prefix}_fallback_avg_notional": float(metrics.get("fallback_avg_notional", 0.0) or 0.0),
        f"{prefix}_fallback_avg_leverage": float(metrics.get("fallback_avg_leverage", 0.0) or 0.0),
        f"{prefix}_fallback_avg_tp_price_move": float(metrics.get("fallback_avg_tp_price_move", 0.0) or 0.0),
        f"{prefix}_fallback_avg_sl_price_move": float(metrics.get("fallback_avg_sl_price_move", 0.0) or 0.0),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_payload, oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    path_labels, path_diag = exp._path_label_table(val_payload, exp.RISK)
    risk_labels, risk_diag = exp._risk_label_table(path_labels, exp.RISK)
    val_risk_pred, oos_risk_pred, risk_fit = _risk_predictions(x_val, x_oos, risk_labels)
    val_risk_pred = _apply_sl_floor(val_risk_pred)
    oos_risk_pred = _apply_sl_floor(oos_risk_pred)
    val_a, val_c, oos_a, oos_c, entry_diag = _entry_actions(x_val, x_oos, path_labels)
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    x_exit, y_exit, exit_diag = _collect_exit_training(
        val_payload,
        val_risk_pred,
        val_a,
        val_c,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
    )
    if len(np.unique(y_exit)) < 2:
        raise RuntimeError(f"exit training labels require two classes: {exit_diag}")
    exit_model = HistGradientBoostingClassifier(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=290001)
    exit_model.fit(x_exit.to_numpy(dtype=np.float64), y_exit)
    rows: list[dict[str, Any]] = []
    base_val = exp._metrics_with_parent_risk(val_payload["frame"], val_payload["dec"], val_risk_pred, val_a, val_c, exp.RISK, 0.0, fee=fee, slip=slip, cost_mult=3.0)
    base_oos = exp._metrics_with_parent_risk(oos_payload["frame"], oos_payload["dec"], oos_risk_pred, oos_a, oos_c, exp.RISK, 0.0, fee=fee, slip=slip, cost_mult=3.0)
    rows.append({"candidate": "bucket_hard_sltp_reference", **_metric_row("val", base_val), **_metric_row("oos", base_oos)})
    for threshold in EXIT_THRESHOLDS:
        val_m = _metrics_with_learned_exit(
            val_payload,
            val_risk_pred,
            val_a,
            val_c,
            exit_model,
            list(x_exit.columns),
            threshold,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
        )
        oos_m = _metrics_with_learned_exit(
            oos_payload,
            oos_risk_pred,
            oos_a,
            oos_c,
            exit_model,
            list(x_exit.columns),
            threshold,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
        )
        rows.append({"candidate": f"learned_exit_thr{threshold:.2f}", "exit_threshold": float(threshold), **_metric_row("val", val_m), **_metric_row("oos", oos_m)})
    ranking = pd.DataFrame(rows)
    ranking["val_score"] = ranking["val_pnl"] + 5.0 * ranking["val_wr"] + 0.20 * ranking["val_mdd"] - 0.15 * ranking["val_trades"]
    ranking = ranking.sort_values(["val_score", "val_pnl"], ascending=False).reset_index(drop=True)
    selected = ranking.iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "status": "diagnostic_learned_exit_eval",
        "method": "Bucket fallback entry/risk model is kept fixed. Fallback hard SL is retained with a widened SL price-move floor, while hard TP is replaced by a conservative profit-exit classifier that can only fire on profitable fallback positions near TP. The classifier uses SL/TP distance/progress, unrealized PnL, hold age, and market features. Parent_takeover and max_hold remain hard safety exits.",
        "entry_candidate": dict(ENTRY_CANDIDATE),
        "exit_training": exit_diag,
        "diagnostics": {
            "path_labels": path_diag,
            "risk_labels": risk_diag,
            "risk_fit": risk_fit,
            "entry": entry_diag,
            "exit_features": list(x_exit.columns),
            "exit_thresholds": [float(x) for x in EXIT_THRESHOLDS],
            "sl_price_move_floor": float(SL_PRICE_MOVE_FLOOR),
            "profit_exit_min_progress": float(PROFIT_EXIT_MIN_PROGRESS),
        },
        "selected_by_validation": selected,
        "rows": ranking.to_dict(orient="records"),
    }
    ranking.to_csv(OUT_DIR / "learned_exit_ranking.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
