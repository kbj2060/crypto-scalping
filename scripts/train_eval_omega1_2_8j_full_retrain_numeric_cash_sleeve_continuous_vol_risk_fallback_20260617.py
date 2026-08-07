#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full_parent  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8j_full_retrain_numeric_cash_sleeve_continuous_vol_risk_fallback_20260617"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
RISK = sleeve.FallbackRisk("continuous_vol_risk_label_base_n081_l2_h192", 0.052, 0.028, 0.81, 2.0, 192)
BASELINE_ID = "omega1_2_true_3head_tabm_20260603_full_retrain_cash_alpha43_20260608"
BASE_POSITION_FRACTION = 0.405
BASE_LEVERAGE = 2.0
BASE_NOTIONAL = BASE_POSITION_FRACTION * BASE_LEVERAGE
BASE_TP = 0.052
BASE_SL = 0.028
BASE_TP_PRICE_MOVE = BASE_TP / BASE_NOTIONAL
BASE_SL_PRICE_MOVE = BASE_SL / BASE_NOTIONAL
PRICE_SCALE_MIN = 0.75
PRICE_SCALE_MAX = 1.65
LEVERAGE_MIN = 1.20
LEVERAGE_MAX = 2.80
UTILITY_CFGS = (
    {"stop_penalty": 0.003, "mae_penalty": 0.0, "time_penalty": 0.0},
    {"stop_penalty": 0.003, "mae_penalty": 0.20, "time_penalty": 0.0},
    {"stop_penalty": 0.003, "mae_penalty": 0.20, "time_penalty": 0.001},
)


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


def _build_payloads() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    if not full_parent.PARENT_DIR.exists():
        raise RuntimeError(f"missing full-retrain parent artifact: {full_parent.PARENT_DIR}")
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full_parent._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full_parent._build_split(frames, "oos")
    meta = {"fee": float(fee), "slip": float(slip), "parent_dir": str(full_parent.PARENT_DIR)}
    return (
        {"frame": val_frame, "dec": val_dec, "features": val_features, **meta},
        {"frame": oos_frame, "dec": oos_dec, "features": oos_features, **meta},
        meta,
    )


def _simulate_side_detail(payload: dict[str, Any], i: int, side: int, risk: sleeve.FallbackRisk) -> dict[str, Any]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0

    cash = 1.0
    cash, pos, entered = sleeve._open_position(cash, arrays, int(i), int(side), "fallback", risk, None, fee_eff, slip_eff)
    if not entered:
        return {"net": -1.0e-6, "score": -1.0e-6, "mae": 0.0, "mfe": 0.0, "stop": 0, "bars": 0, "reason": "entry_miss"}

    max_i = min(len(frame) - 2, int(pos.entry_i) + int(risk.max_hold_bars))
    mfe = 0.0
    mae = 0.0
    exit_i = max_i
    reason = "max_hold"
    for j in range(int(pos.entry_i), max_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
        unreal = raw * float(pos.notional)
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        exit_i = int(j)
        if unreal >= float(risk.take_profit):
            reason = "take_profit"
            break
        if unreal <= -abs(float(risk.stop_loss)):
            reason = "stop_loss"
            break
        if bool(active[j]):
            reason = "primary_takeover"
            break

    before = cash
    cash, _win = sleeve._close_position(cash, arrays, pos, exit_i, fee_eff, slip_eff)
    net = float(cash - 1.0)
    if not np.isfinite(before):
        raise RuntimeError("non-finite cash during label simulation")
    adverse = abs(min(float(mae), 0.0))
    bars = max(0, int(exit_i) - int(pos.entry_i))
    return {
        "net": net,
        "score": net,
        "mae": float(mae),
        "mfe": float(mfe),
        "stop": int(reason == "stop_loss"),
        "bars": int(bars),
        "reason": reason,
    }


def _numeric_label_table(payload: dict[str, Any], risk: sleeve.FallbackRisk, utility_cfg: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    active = omega._active(payload["dec"].reset_index(drop=True))
    max_hold = max(int(risk.max_hold_bars), 1)
    rows: list[dict[str, Any]] = []
    for i in np.flatnonzero(~active):
        if i >= len(frame) - int(risk.max_hold_bars) - 3:
            continue
        long_d = _simulate_side_detail(payload, int(i), 1, risk)
        short_d = _simulate_side_detail(payload, int(i), -1, risk)

        def util(d: dict[str, Any]) -> float:
            adverse = abs(min(float(d["mae"]), 0.0))
            time_frac = min(float(d["bars"]) / float(max_hold), 1.0)
            return float(d["net"]) - float(utility_cfg["stop_penalty"]) * int(d["stop"]) - float(utility_cfg["mae_penalty"]) * adverse - float(utility_cfg["time_penalty"]) * time_frac

        rows.append(
            {
                "i": int(i),
                "long_net": float(long_d["net"]),
                "short_net": float(short_d["net"]),
                "long_utility": util(long_d),
                "short_utility": util(short_d),
                "long_reason": str(long_d["reason"]),
                "short_reason": str(short_d["reason"]),
                "long_stop": int(long_d["stop"]),
                "short_stop": int(short_d["stop"]),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "valid_cash_rows": int(len(labels)),
        "utility_cfg": dict(utility_cfg),
        "long_positive": int((labels["long_utility"] > 0.0).sum()) if len(labels) else 0,
        "short_positive": int((labels["short_utility"] > 0.0).sum()) if len(labels) else 0,
        "long_reason_counts": labels["long_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
        "short_reason_counts": labels["short_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
    }
    return labels, diag


def _path_label_table(payload: dict[str, Any], risk: sleeve.FallbackRisk) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    active = omega._active(payload["dec"].reset_index(drop=True))
    rows: list[dict[str, Any]] = []
    cash_idx = np.flatnonzero(~active)
    for row_id, i in enumerate(cash_idx):
        if row_id % 1000 == 0:
            print(json.dumps({"stage": "path_labels", "row": int(row_id), "total_cash": int(len(cash_idx))}, ensure_ascii=True), flush=True)
        if i >= len(frame) - int(risk.max_hold_bars) - 3:
            continue
        long_d = _simulate_side_detail(payload, int(i), 1, risk)
        short_d = _simulate_side_detail(payload, int(i), -1, risk)
        rows.append(
            {
                "i": int(i),
                "long_net": float(long_d["net"]),
                "short_net": float(short_d["net"]),
                "long_mae": float(long_d["mae"]),
                "short_mae": float(short_d["mae"]),
                "long_bars": int(long_d["bars"]),
                "short_bars": int(short_d["bars"]),
                "long_stop": int(long_d["stop"]),
                "short_stop": int(short_d["stop"]),
                "long_reason": str(long_d["reason"]),
                "short_reason": str(short_d["reason"]),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "valid_cash_rows": int(len(labels)),
        "long_net_positive": int((labels["long_net"] > 0.0).sum()) if len(labels) else 0,
        "short_net_positive": int((labels["short_net"] > 0.0).sum()) if len(labels) else 0,
        "long_reason_counts": labels["long_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
        "short_reason_counts": labels["short_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
    }
    return labels, diag


def _utility_from_path_labels(path_labels: pd.DataFrame, risk: sleeve.FallbackRisk, utility_cfg: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = path_labels.copy()
    max_hold = max(int(risk.max_hold_bars), 1)
    labels["long_utility"] = (
        labels["long_net"].astype(float)
        - float(utility_cfg["stop_penalty"]) * labels["long_stop"].astype(float)
        - float(utility_cfg["mae_penalty"]) * np.maximum(-labels["long_mae"].astype(float), 0.0)
        - float(utility_cfg["time_penalty"]) * np.minimum(labels["long_bars"].astype(float) / float(max_hold), 1.0)
    )
    labels["short_utility"] = (
        labels["short_net"].astype(float)
        - float(utility_cfg["stop_penalty"]) * labels["short_stop"].astype(float)
        - float(utility_cfg["mae_penalty"]) * np.maximum(-labels["short_mae"].astype(float), 0.0)
        - float(utility_cfg["time_penalty"]) * np.minimum(labels["short_bars"].astype(float) / float(max_hold), 1.0)
    )
    diag = {
        "valid_cash_rows": int(len(labels)),
        "utility_cfg": dict(utility_cfg),
        "long_positive": int((labels["long_utility"] > 0.0).sum()) if len(labels) else 0,
        "short_positive": int((labels["short_utility"] > 0.0).sum()) if len(labels) else 0,
    }
    return labels, diag


def _vol_score(frame: pd.DataFrame) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    ret = close.pct_change().fillna(0.0)
    return ret.rolling(96, min_periods=24).std().bfill().fillna(0.0).to_numpy(dtype=np.float64)


def _vol_thresholds(frame: pd.DataFrame) -> tuple[float, float]:
    vol = _vol_score(frame)
    return float(np.quantile(vol, 0.60)), float(np.quantile(vol, 0.85))


def _vol_risk_for_row(i: int, vol: np.ndarray, q60: float, q85: float) -> sleeve.FallbackRisk:
    del q85
    ref = max(float(q60), 1.0e-12)
    v = max(float(vol[int(i)]), ref * 0.25)
    price_scale = float(np.clip(v / ref, PRICE_SCALE_MIN, PRICE_SCALE_MAX))
    leverage = float(np.clip(BASE_LEVERAGE / price_scale, LEVERAGE_MIN, LEVERAGE_MAX))
    notional = float(BASE_POSITION_FRACTION * leverage)
    take_profit = float(BASE_TP_PRICE_MOVE * price_scale * notional)
    stop_loss = float(BASE_SL_PRICE_MOVE * price_scale * notional)
    return sleeve.FallbackRisk("continuous_vol_risk", take_profit, stop_loss, notional, leverage, 192)


def _metrics_with_vol_sltp(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    threshold: float,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    vol_quantiles: tuple[float, float],
) -> dict[str, Any]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    vol = _vol_score(frame)
    q60, q85 = vol_quantiles
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    primary_takeovers = 0
    bucket_counts: dict[str, int] = {}
    fallback_tp: list[float] = []
    fallback_sl: list[float] = []
    fallback_leverage: list[float] = []
    fallback_notional: list[float] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1.0e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1.0e-12)
            unreal = raw * pos.notional
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)
            reason = ""
            if pos.take_profit > 0.0 and unreal >= pos.take_profit:
                reason = "take_profit"
            elif pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
                reason = "stop_loss"
            elif pos.max_hold_bars > 0 and int(i) - int(pos.entry_i) >= pos.max_hold_bars:
                reason = "max_hold"
            elif pos.sleeve == "fallback" and bool(active[i]):
                reason = "primary_takeover"
                primary_takeovers += 1
            if reason:
                cash, win = sleeve._close_position(cash, arrays, pos, i, fee_eff, slip_eff)
                trades += 1
                wins += int(win)
                reasons[f"{pos.sleeve}_{reason}"] = reasons.get(f"{pos.sleeve}_{reason}", 0) + 1
                pos = sleeve.Position()
            else:
                continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
        if bool(active[i]):
            row = dec.iloc[int(i)]
            side = int(row.get("side", 0) or 0)
            if side != 0:
                cash, pos, entered = sleeve._open_position(cash, arrays, i, side, "primary", None, row, fee_eff, slip_eff)
                if entered:
                    primary_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
        else:
            side_action = int(fallback_action[i]) if i < len(fallback_action) else sleeve.ACTION_CASH
            conf = float(fallback_conf[i]) if i < len(fallback_conf) else 0.0
            if side_action in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) and conf >= float(threshold):
                side = 1 if side_action == sleeve.ACTION_LONG else -1
                row_risk = _vol_risk_for_row(i, vol, q60, q85)
                cash, pos, entered = sleeve._open_position(cash, arrays, i, side, "fallback", row_risk, None, fee_eff, slip_eff)
                if entered:
                    fallback_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
                    bucket_counts[row_risk.name] = bucket_counts.get(row_risk.name, 0) + 1
                    fallback_tp.append(float(row_risk.take_profit))
                    fallback_sl.append(float(row_risk.stop_loss))
                    fallback_leverage.append(float(row_risk.leverage))
                    fallback_notional.append(float(row_risk.notional))
    if pos.side != 0:
        cash, win = sleeve._close_position(cash, arrays, pos, len(frame) - 2, fee_eff, slip_eff)
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
        "avg_notional": float(RISK.notional),
        "avg_leverage": float(RISK.leverage),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(primary_takeovers),
        "fallback_avg_tp": float(np.mean(fallback_tp)) if fallback_tp else 0.0,
        "fallback_avg_sl": float(np.mean(fallback_sl)) if fallback_sl else 0.0,
        "fallback_avg_leverage": float(np.mean(fallback_leverage)) if fallback_leverage else 0.0,
        "fallback_avg_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "fallback_avg_tp_price_move": float(np.mean(np.asarray(fallback_tp) / np.maximum(np.asarray(fallback_notional), 1.0e-12))) if fallback_tp else 0.0,
        "fallback_avg_sl_price_move": float(np.mean(np.asarray(fallback_sl) / np.maximum(np.asarray(fallback_notional), 1.0e-12))) if fallback_sl else 0.0,
        "fallback_bucket_counts": bucket_counts,
    }


def _chron_folds(idx: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    n = len(idx)
    for train_frac, end_frac in ((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00)):
        train_end = int(n * train_frac)
        val_end = int(n * end_frac)
        if train_end >= 100 and val_end > train_end:
            folds.append((idx[:train_end], idx[train_end:val_end]))
    return folds


def _fit_predict_lower_bound(
    x_val: pd.DataFrame,
    x_oos: pd.DataFrame,
    labels: pd.DataFrame,
    long_col: str,
    short_col: str,
    *,
    seed: int,
    cal_q: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = labels["i"].to_numpy(dtype=np.int64)
    y_long = np.zeros(len(x_val), dtype=np.float64)
    y_short = np.zeros(len(x_val), dtype=np.float64)
    y_long[idx] = labels[long_col].to_numpy(dtype=np.float64)
    y_short[idx] = labels[short_col].to_numpy(dtype=np.float64)
    val_long = np.zeros(len(x_val), dtype=np.float64)
    val_short = np.zeros(len(x_val), dtype=np.float64)
    folds_meta: list[dict[str, Any]] = []
    for fold_id, (tr, va) in enumerate(_chron_folds(idx)):
        ml = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + fold_id * 10 + 1))
        ms = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + fold_id * 10 + 2))
        ml.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), y_long[tr])
        ms.fit(x_val.iloc[tr].to_numpy(dtype=np.float64), y_short[tr])
        pred_l = ml.predict(x_val.iloc[va].to_numpy(dtype=np.float64)).astype(np.float64)
        pred_s = ms.predict(x_val.iloc[va].to_numpy(dtype=np.float64)).astype(np.float64)
        ql = float(np.quantile(np.abs(y_long[tr] - ml.predict(x_val.iloc[tr].to_numpy(dtype=np.float64))), cal_q))
        qs = float(np.quantile(np.abs(y_short[tr] - ms.predict(x_val.iloc[tr].to_numpy(dtype=np.float64))), cal_q))
        val_long[va] = pred_l - ql
        val_short[va] = pred_s - qs
        folds_meta.append({"fold": int(fold_id), "train_rows": int(len(tr)), "val_rows": int(len(va)), "long_abs_resid_q": ql, "short_abs_resid_q": qs})

    ml = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + 101))
    ms = HistGradientBoostingRegressor(max_iter=160, learning_rate=0.035, max_leaf_nodes=9, l2_regularization=2.0, random_state=int(seed + 102))
    x_train = x_val.iloc[idx].to_numpy(dtype=np.float64)
    ml.fit(x_train, y_long[idx])
    ms.fit(x_train, y_short[idx])
    ql = float(np.quantile(np.abs(y_long[idx] - ml.predict(x_train)), cal_q))
    qs = float(np.quantile(np.abs(y_short[idx] - ms.predict(x_train)), cal_q))
    oos_long = ml.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64) - ql
    oos_short = ms.predict(x_oos.to_numpy(dtype=np.float64)).astype(np.float64) - qs
    diag = {"target_cols": [long_col, short_col], "cal_q": float(cal_q), "folds": folds_meta, "final_long_abs_resid_q": ql, "final_short_abs_resid_q": qs}
    return val_long, val_short, oos_long, oos_short, diag


def _actions_from_scores(long_s: np.ndarray, short_s: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    best_long = long_s >= short_s
    best = np.where(best_long, long_s, short_s)
    action = np.where(best > float(threshold), np.where(best_long, sleeve.ACTION_LONG, sleeve.ACTION_SHORT), sleeve.ACTION_CASH).astype(np.int64)
    conf = np.clip((best - float(threshold)) / 0.02, 0.0, 1.0).astype(np.float64)
    return action, conf


def _apply_agreement(
    ev_action: np.ndarray,
    ev_conf: np.ndarray,
    util_long: np.ndarray,
    util_short: np.ndarray,
    *,
    utility_min: float,
    margin_min: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    long_ok = (ev_action == sleeve.ACTION_LONG) & (util_long > float(utility_min)) & ((util_long - util_short) >= float(margin_min))
    short_ok = (ev_action == sleeve.ACTION_SHORT) & (util_short > float(utility_min)) & ((util_short - util_long) >= float(margin_min))
    keep = long_ok | short_ok
    support = np.where(ev_action == sleeve.ACTION_LONG, util_long, np.where(ev_action == sleeve.ACTION_SHORT, util_short, 0.0))
    action = np.where(keep, ev_action, sleeve.ACTION_CASH).astype(np.int64)
    conf = np.where(keep, np.minimum(ev_conf, np.clip((support - float(utility_min)) / 0.02, 0.0, 1.0)), 0.0).astype(np.float64)
    active = np.isin(ev_action, [sleeve.ACTION_LONG, sleeve.ACTION_SHORT])
    return action, conf, {"ev_active_rows": int(active.sum()), "kept_rows": int(keep.sum()), "veto_rows": int((active & ~keep).sum()), "keep_rate_on_ev_active": float(keep.sum() / max(active.sum(), 1))}


def _metric_row(candidate: str, family: str, cfg_id: int | None, cal_q: float, ev_min: float, utility_min: float | None, margin_min: float | None, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {
        "candidate": candidate,
        "family": family,
        "utility_cfg_id": cfg_id,
        "cal_q": float(cal_q),
        "ev_min": float(ev_min),
        "utility_min": None if utility_min is None else float(utility_min),
        "margin_min": None if margin_min is None else float(margin_min),
    }
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_fallback_avg_tp"] = float(val_m.get("fallback_avg_tp", 0.0) or 0.0)
    row["val_fallback_avg_sl"] = float(val_m.get("fallback_avg_sl", 0.0) or 0.0)
    row["oos_fallback_avg_tp"] = float(oos_m.get("fallback_avg_tp", 0.0) or 0.0)
    row["oos_fallback_avg_sl"] = float(oos_m.get("fallback_avg_sl", 0.0) or 0.0)
    row["val_fallback_avg_leverage"] = float(val_m.get("fallback_avg_leverage", 0.0) or 0.0)
    row["oos_fallback_avg_leverage"] = float(oos_m.get("fallback_avg_leverage", 0.0) or 0.0)
    row["val_fallback_avg_notional"] = float(val_m.get("fallback_avg_notional", 0.0) or 0.0)
    row["oos_fallback_avg_notional"] = float(oos_m.get("fallback_avg_notional", 0.0) or 0.0)
    row["val_fallback_avg_tp_price_move"] = float(val_m.get("fallback_avg_tp_price_move", 0.0) or 0.0)
    row["oos_fallback_avg_tp_price_move"] = float(oos_m.get("fallback_avg_tp_price_move", 0.0) or 0.0)
    row["val_fallback_avg_sl_price_move"] = float(val_m.get("fallback_avg_sl_price_move", 0.0) or 0.0)
    row["oos_fallback_avg_sl_price_move"] = float(oos_m.get("fallback_avg_sl_price_move", 0.0) or 0.0)
    row["val_fallback_bucket_counts"] = val_m.get("fallback_bucket_counts", {})
    row["oos_fallback_bucket_counts"] = oos_m.get("fallback_bucket_counts", {})
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = _build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("validation/oos feature columns mismatch")
    fee = float(meta["fee"])
    slip = float(meta["slip"])
    vol_quantiles = _vol_thresholds(val_payload["frame"])
    base_val = omega._metrics(val_payload["frame"], val_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_oos = omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=fee, slip=slip, cost_mult=3.0)
    base_val_sleeve = {**base_val, "primary_entries": base_val["long_entries"] + base_val["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}
    base_oos_sleeve = {**base_oos, "primary_entries": base_oos["long_entries"] + base_oos["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}

    path_labels, path_diag = _path_label_table(val_payload, RISK)
    ev_labels, ev_diag = _utility_from_path_labels(path_labels, RISK, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
    utility_preds: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    diagnostics: dict[str, Any] = {
        "mode": "full_retrain_parent_numeric_cash_sleeve_with_continuous_vol_risk_fallback_only",
        "baseline_model_id": BASELINE_ID,
        "parent_artifact": meta["parent_dir"],
        "risk": asdict(RISK),
        "risk_semantics": "take_profit/stop_loss are account-PnL thresholds; price move is threshold / notional_exposure",
        "continuous_vol_risk": {
            "vol_source": "past close pct_change rolling std, window=96, min_periods=24",
            "application": "fallback positions only; primary parent rows keep their original parent risk contract",
            "validation_quantiles": {"q60": vol_quantiles[0], "q85": vol_quantiles[1]},
            "formula": {
                "price_scale": "clip(row_vol / validation_q60, 0.75, 1.65)",
                "leverage": "clip(2.0 / price_scale, 1.20, 2.80)",
                "notional": "0.405 * leverage",
                "take_profit": "(0.052 / 0.81) * price_scale * notional",
                "stop_loss": "(0.028 / 0.81) * price_scale * notional",
            },
        },
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "baseline": {"validation": base_val_sleeve, "oos": base_oos_sleeve},
        "path_labels": path_diag,
        "ev_labels": ev_diag,
    }
    for cfg_id, cfg in enumerate(UTILITY_CFGS):
        print(json.dumps({"stage": "fit_utility", "cfg_id": int(cfg_id), "config": cfg}, ensure_ascii=True), flush=True)
        labels, diag = _utility_from_path_labels(path_labels, RISK, cfg)
        vl, vs, ol, os, fit_diag = _fit_predict_lower_bound(x_val, x_oos, labels, "long_utility", "short_utility", seed=281000 + cfg_id * 100, cal_q=0.50)
        utility_preds[cfg_id] = (vl, vs, ol, os)
        diagnostics[f"utility_cfg_{cfg_id}"] = {"config": cfg, "labels": diag, "fit": fit_diag}

    rows: list[dict[str, Any]] = [
        {
            "candidate": "full_retrain_primary_only",
            "family": "baseline",
            "utility_cfg_id": None,
            "cal_q": None,
            "ev_min": None,
            "utility_min": None,
            "margin_min": None,
            **sleeve._metric_row("val", base_val_sleeve),
            **sleeve._metric_row("oos", base_oos_sleeve),
            "val_delta_pnl": 0.0,
            "oos_delta_pnl": 0.0,
        }
    ]

    for cal_q in (0.50, 0.65, 0.80):
        print(json.dumps({"stage": "fit_ev", "cal_q": float(cal_q)}, ensure_ascii=True), flush=True)
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit_diag = _fit_predict_lower_bound(x_val, x_oos, ev_labels, "long_net", "short_net", seed=280000, cal_q=cal_q)
        diagnostics[f"ev_lower_bound_cal_q{cal_q:.2f}"] = ev_fit_diag
        for ev_min in (0.001, 0.002, 0.003, 0.004):
            val_ev_a, val_ev_c = _actions_from_scores(ev_vl, ev_vs, ev_min)
            oos_ev_a, oos_ev_c = _actions_from_scores(ev_ol, ev_os, ev_min)
            val_m = _metrics_with_vol_sltp(val_payload["frame"], val_payload["dec"], val_ev_a, val_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0, vol_quantiles=vol_quantiles)
            oos_m = _metrics_with_vol_sltp(oos_payload["frame"], oos_payload["dec"], oos_ev_a, oos_ev_c, 0.0, fee=fee, slip=slip, cost_mult=3.0, vol_quantiles=vol_quantiles)
            ev_name = f"full_retrain_ev_cal{cal_q:.2f}_ev{ev_min:.3f}"
            rows.append(_metric_row(ev_name, "ev_lower_bound_only", None, cal_q, ev_min, None, None, val_m, oos_m, base_val_sleeve, base_oos_sleeve))
            for cfg_id, (uvl, uvs, uol, uos) in utility_preds.items():
                for utility_min in (-0.001, 0.0, 0.001, 0.002):
                    for margin_min in (0.0, 0.001, 0.002):
                        val_a, val_c, val_filter = _apply_agreement(val_ev_a, val_ev_c, uvl, uvs, utility_min=utility_min, margin_min=margin_min)
                        oos_a, oos_c, oos_filter = _apply_agreement(oos_ev_a, oos_ev_c, uol, uos, utility_min=utility_min, margin_min=margin_min)
                        cand = f"full_retrain_ev_cal{cal_q:.2f}_ev{ev_min:.3f}_numcfg{cfg_id}_u{utility_min:.3f}_m{margin_min:.3f}"
                        diagnostics[f"{cand}_filter"] = {"validation": val_filter, "oos": oos_filter}
                        val_m = _metrics_with_vol_sltp(val_payload["frame"], val_payload["dec"], val_a, val_c, 0.0, fee=fee, slip=slip, cost_mult=3.0, vol_quantiles=vol_quantiles)
                        oos_m = _metrics_with_vol_sltp(oos_payload["frame"], oos_payload["dec"], oos_a, oos_c, 0.0, fee=fee, slip=slip, cost_mult=3.0, vol_quantiles=vol_quantiles)
                        rows.append(_metric_row(cand, "ev_lower_bound_numeric_agreement_veto", cfg_id, cal_q, ev_min, utility_min, margin_min, val_m, oos_m, base_val_sleeve, base_oos_sleeve))

    ranking = pd.DataFrame(rows)
    ranking["selection_score_val_only"] = (
        ranking["val_delta_pnl"].fillna(0.0)
        + 0.12 * ranking["val_fallback_entries"].fillna(0.0)
        + 8.0 * ranking["val_wr"].fillna(0.0)
        + 0.20 * ranking["val_mdd"].fillna(0.0)
    )
    ranking = ranking.sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "full_retrain_numeric_cash_sleeve_ranking.csv", index=False)
    hybrid = ranking[ranking["family"].eq("ev_lower_bound_numeric_agreement_veto")].copy()
    selected = hybrid.iloc[0].to_dict() if len(hybrid) else ranking.iloc[0].to_dict()
    best_oos = (hybrid.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict() if len(hybrid) else ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict())
    best_controls = ranking[~ranking["family"].eq("ev_lower_bound_numeric_agreement_veto")].head(5).to_dict(orient="records")

    blockers: list[str] = []
    bad = [c for c in x_val.columns if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")]
    if bad:
        blockers.append(f"forbidden feature columns: {bad[:20]}")
    if len(hybrid) == 0:
        blockers.append("no numeric hybrid candidates produced")
    if list(x_val.columns) != list(x_oos.columns):
        blockers.append("validation/oos feature column mismatch")

    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_full_retrain_numeric_eval" if not blockers else "redteam_fail",
        "method": "Full-retrained 3-head parent artifact is used as parent. Cash sleeve EV lower-bound and numeric utility agreement/veto regressors are newly fit on validation full-retrain features and evaluated on fixed 2026 OOS. Fallback positions use continuous volatility-normalized account-PnL TP/SL and leverage instead of discrete risk templates.",
        "selection_policy": "hybrid_validation_only_no_oos_selection; EV-only rows are controls, OOS is diagnostic",
        "diagnostics": diagnostics,
        "baseline": {"validation": base_val_sleeve, "oos": base_oos_sleeve},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "best_ev_only_controls": best_controls,
        "top20_hybrid": hybrid.head(20).to_dict(orient="records"),
        "top20_all_including_controls": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "full_retrain_numeric_cash_sleeve_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos_diagnostic": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
