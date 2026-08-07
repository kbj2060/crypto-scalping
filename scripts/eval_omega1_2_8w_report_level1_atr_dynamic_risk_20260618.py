#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_8b_full_retrain_numeric_cash_sleeve_leverage_only_20260616 as exp  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_8w_report_level1_atr_dynamic_risk_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
MAX_HOLD_BARS = 192


@dataclass(frozen=True)
class Level1Cfg:
    name: str
    sl_atr_mult: float
    tp_atr_mult: float
    target_stop_risk: float
    min_notional: float
    max_notional: float
    base_leverage: float
    max_leverage: float


@dataclass(frozen=True)
class RowRisk:
    name: str
    take_profit: float
    stop_loss: float
    notional: float
    leverage: float
    max_hold_bars: int


CFGS = (
    Level1Cfg("atr_sl10_tp15_risk028_n030_100_lev2", 1.0, 1.5, 0.028, 0.30, 1.00, 2.0, 2.0),
    Level1Cfg("atr_sl15_tp225_risk028_n030_100_lev2", 1.5, 2.25, 0.028, 0.30, 1.00, 2.0, 2.0),
    Level1Cfg("atr_sl15_tp300_risk028_n030_100_lev2", 1.5, 3.0, 0.028, 0.30, 1.00, 2.0, 2.0),
    Level1Cfg("atr_sl20_tp300_risk028_n030_100_lev2", 2.0, 3.0, 0.028, 0.30, 1.00, 2.0, 2.0),
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


def _reason_count(reasons: Any, key: str) -> int:
    if not isinstance(reasons, dict):
        return 0
    return int(reasons.get(key, 0) or 0)


def _base_sleeve_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        **metrics,
        "primary_entries": int(metrics["long_entries"] + metrics["short_entries"]),
        "fallback_entries": 0,
        "primary_takeovers": 0,
        "exit_reasons": dict(metrics.get("exit_reasons") or {}),
    }


def _atr_pct(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise").astype(float)
    low = pd.to_numeric(frame["low"], errors="raise").astype(float)
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(period, min_periods=2).mean().ffill().fillna(tr).fillna(0.0)
    return (atr / close.clip(lower=1.0e-12)).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)


def _realized_vol(frame: pd.DataFrame, window: int = 24) -> np.ndarray:
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    ret = close.pct_change().fillna(0.0)
    vol = ret.rolling(window, min_periods=4).std().ffill().fillna(0.0)
    return vol.to_numpy(dtype=np.float64)


def _risk_arrays(frame: pd.DataFrame, cfg: Level1Cfg) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    atr = np.maximum(_atr_pct(frame), 1.0e-6)
    vol = np.maximum(_realized_vol(frame), 1.0e-8)
    vol_ref = (
        pd.Series(vol)
        .rolling(96, min_periods=24)
        .median()
        .ffill()
        .fillna(float(np.nanmedian(vol)) if np.isfinite(np.nanmedian(vol)) else 1.0e-4)
        .to_numpy(dtype=np.float64)
    )
    sl_price = np.maximum(atr * float(cfg.sl_atr_mult), 1.0e-6)
    tp_price = np.maximum(atr * float(cfg.tp_atr_mult), 1.0e-6)
    notional = np.clip(float(cfg.target_stop_risk) / sl_price, float(cfg.min_notional), float(cfg.max_notional))
    leverage = np.clip(float(cfg.base_leverage) * np.maximum(vol_ref, 1.0e-8) / vol, 1.0, float(cfg.max_leverage))
    stop_loss = notional * sl_price
    take_profit = notional * tp_price
    arrays = {
        "atr_pct": atr,
        "realized_vol": vol,
        "vol_ref": vol_ref,
        "sl_price_pct": sl_price,
        "tp_price_pct": tp_price,
        "notional": notional,
        "leverage": leverage,
        "stop_loss": stop_loss,
        "take_profit": take_profit,
    }
    diag = {
        "cfg": cfg.__dict__,
        "atr_pct_mean": float(np.mean(atr)),
        "atr_pct_p10": float(np.quantile(atr, 0.10)),
        "atr_pct_p50": float(np.quantile(atr, 0.50)),
        "atr_pct_p90": float(np.quantile(atr, 0.90)),
        "notional_mean": float(np.mean(notional)),
        "notional_p10": float(np.quantile(notional, 0.10)),
        "notional_p50": float(np.quantile(notional, 0.50)),
        "notional_p90": float(np.quantile(notional, 0.90)),
        "leverage_mean": float(np.mean(leverage)),
        "leverage_p10": float(np.quantile(leverage, 0.10)),
        "leverage_p50": float(np.quantile(leverage, 0.50)),
        "leverage_p90": float(np.quantile(leverage, 0.90)),
        "take_profit_mean": float(np.mean(take_profit)),
        "stop_loss_mean": float(np.mean(stop_loss)),
        "tp_price_pct_mean": float(np.mean(tp_price)),
        "sl_price_pct_mean": float(np.mean(sl_price)),
    }
    return arrays, diag


def _row_risk(cfg: Level1Cfg, risk_arrays: dict[str, np.ndarray], i: int) -> RowRisk:
    j = int(i)
    return RowRisk(
        cfg.name,
        float(risk_arrays["take_profit"][j]),
        float(risk_arrays["stop_loss"][j]),
        float(risk_arrays["notional"][j]),
        float(risk_arrays["leverage"][j]),
        MAX_HOLD_BARS,
    )


def _open_dynamic_position(
    cash: float,
    arrays: dict[str, np.ndarray],
    i: int,
    side: int,
    source: str,
    risk: RowRisk,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, sleeve.Position, bool]:
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled or float(risk.notional) <= 0.0:
        return cash, sleeve.Position(), False
    entry_equity = cash
    cash -= cash * float(entry_fee) * float(risk.notional)
    return (
        cash,
        sleeve.Position(
            sleeve=source,
            side=int(side),
            entry_price=float(entry_px),
            entry_i=int(i),
            entry_equity=float(entry_equity),
            notional=float(risk.notional),
            leverage=float(risk.leverage),
            take_profit=float(risk.take_profit),
            stop_loss=abs(float(risk.stop_loss)),
            max_hold_bars=int(risk.max_hold_bars),
        ),
        True,
    )


def _simulate_side_detail(payload: dict[str, Any], i: int, side: int, cfg: Level1Cfg, risk_arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    risk = _row_risk(cfg, risk_arrays, int(i))
    cash = 1.0
    cash, pos, entered = _open_dynamic_position(cash, arrays, int(i), int(side), "fallback", risk, fee_eff, slip_eff)
    if not entered:
        return {"net": -1.0e-6, "mae": 0.0, "mfe": 0.0, "stop": 0, "bars": 0, "reason": "entry_miss", "risk": risk}
    max_i = min(len(frame) - 2, int(pos.entry_i) + int(pos.max_hold_bars))
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
        if unreal >= float(pos.take_profit):
            reason = "take_profit"
            break
        if unreal <= -abs(float(pos.stop_loss)):
            reason = "stop_loss"
            break
        if bool(active[j]):
            reason = "primary_takeover"
            break
    cash, _win = sleeve._close_position(cash, arrays, pos, exit_i, fee_eff, slip_eff)
    return {
        "net": float(cash - 1.0),
        "mae": float(mae),
        "mfe": float(mfe),
        "stop": int(reason == "stop_loss"),
        "bars": int(max(0, exit_i - int(pos.entry_i))),
        "reason": reason,
        "risk": risk,
    }


def _path_label_table(payload: dict[str, Any], cfg: Level1Cfg, risk_arrays: dict[str, np.ndarray]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = payload["frame"].reset_index(drop=True)
    active = omega._active(payload["dec"].reset_index(drop=True))
    rows: list[dict[str, Any]] = []
    cash_idx = np.flatnonzero(~active)
    for row_id, i in enumerate(cash_idx):
        if row_id % 1000 == 0:
            print(json.dumps({"stage": "path_labels", "cfg": cfg.name, "row": int(row_id), "total_cash": int(len(cash_idx))}, ensure_ascii=True), flush=True)
        if i >= len(frame) - MAX_HOLD_BARS - 3:
            continue
        long_d = _simulate_side_detail(payload, int(i), 1, cfg, risk_arrays)
        short_d = _simulate_side_detail(payload, int(i), -1, cfg, risk_arrays)
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
                "long_tp": float(long_d["risk"].take_profit),
                "short_tp": float(short_d["risk"].take_profit),
                "long_sl": float(long_d["risk"].stop_loss),
                "short_sl": float(short_d["risk"].stop_loss),
                "long_notional": float(long_d["risk"].notional),
                "short_notional": float(short_d["risk"].notional),
                "long_leverage": float(long_d["risk"].leverage),
                "short_leverage": float(short_d["risk"].leverage),
            }
        )
    labels = pd.DataFrame(rows)
    diag = {
        "valid_cash_rows": int(len(labels)),
        "long_net_positive": int((labels["long_net"] > 0.0).sum()) if len(labels) else 0,
        "short_net_positive": int((labels["short_net"] > 0.0).sum()) if len(labels) else 0,
        "long_reason_counts": labels["long_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
        "short_reason_counts": labels["short_reason"].value_counts().sort_index().to_dict() if len(labels) else {},
        "avg_take_profit": float(np.mean(pd.concat([labels["long_tp"], labels["short_tp"]]))) if len(labels) else 0.0,
        "avg_stop_loss": float(np.mean(pd.concat([labels["long_sl"], labels["short_sl"]]))) if len(labels) else 0.0,
        "avg_notional": float(np.mean(pd.concat([labels["long_notional"], labels["short_notional"]]))) if len(labels) else 0.0,
        "avg_leverage": float(np.mean(pd.concat([labels["long_leverage"], labels["short_leverage"]]))) if len(labels) else 0.0,
    }
    return labels, diag


def _utility_from_path_labels(path_labels: pd.DataFrame, utility_cfg: dict[str, float]) -> tuple[pd.DataFrame, dict[str, Any]]:
    labels = path_labels.copy()
    max_hold = max(int(MAX_HOLD_BARS), 1)
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
    return labels, {
        "valid_cash_rows": int(len(labels)),
        "utility_cfg": dict(utility_cfg),
        "long_positive": int((labels["long_utility"] > 0.0).sum()) if len(labels) else 0,
        "short_positive": int((labels["short_utility"] > 0.0).sum()) if len(labels) else 0,
    }


def _metrics_with_dynamic_risk(
    payload: dict[str, Any],
    cfg: Level1Cfg,
    risk_arrays: dict[str, np.ndarray],
    fallback_action: np.ndarray,
    fallback_conf: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    frame = payload["frame"].reset_index(drop=True)
    dec = payload["dec"].reset_index(drop=True)
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    fee_eff = float(payload["fee"]) * 3.0
    slip_eff = float(payload["slip"]) * 3.0
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = sleeve.Position()
    trades = wins = 0
    primary_entries = fallback_entries = long_entries = short_entries = 0
    primary_takeovers = 0
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
            action = int(fallback_action[i]) if i < len(fallback_action) else sleeve.ACTION_CASH
            conf = float(fallback_conf[i]) if i < len(fallback_conf) else 0.0
            if action in (sleeve.ACTION_LONG, sleeve.ACTION_SHORT) and conf >= float(threshold):
                side = 1 if action == sleeve.ACTION_LONG else -1
                risk = _row_risk(cfg, risk_arrays, int(i))
                cash, pos, entered = _open_dynamic_position(cash, arrays, i, side, "fallback", risk, fee_eff, slip_eff)
                if entered:
                    fallback_entries += 1
                    long_entries += int(side > 0)
                    short_entries += int(side < 0)
                    fallback_tp.append(float(risk.take_profit))
                    fallback_sl.append(float(risk.stop_loss))
                    fallback_notional.append(float(risk.notional))
                    fallback_leverage.append(float(risk.leverage))
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
        "avg_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "avg_leverage": float(np.mean(fallback_leverage)) if fallback_leverage else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "primary_entries": int(primary_entries),
        "fallback_entries": int(fallback_entries),
        "primary_takeovers": int(primary_takeovers),
        "fallback_avg_tp": float(np.mean(fallback_tp)) if fallback_tp else 0.0,
        "fallback_avg_sl": float(np.mean(fallback_sl)) if fallback_sl else 0.0,
        "fallback_avg_notional": float(np.mean(fallback_notional)) if fallback_notional else 0.0,
        "fallback_avg_leverage": float(np.mean(fallback_leverage)) if fallback_leverage else 0.0,
    }


def _metric_row(candidate: str, cfg: Level1Cfg, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "risk_cfg": cfg.name}
    row.update(sleeve._metric_row("val", val_m))
    row.update(sleeve._metric_row("oos", oos_m))
    row["val_delta_pnl"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_pnl"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    for split in ("val", "oos"):
        reasons = row[f"{split}_reasons"]
        row[f"{split}_fallback_stop_loss"] = _reason_count(reasons, "fallback_stop_loss")
        row[f"{split}_fallback_take_profit"] = _reason_count(reasons, "fallback_take_profit")
        row[f"{split}_fallback_max_hold"] = _reason_count(reasons, "fallback_max_hold")
        row[f"{split}_fallback_primary_takeover"] = _reason_count(reasons, "fallback_primary_takeover")
    row["val_fallback_stop_rate"] = float(row["val_fallback_stop_loss"] / max(int(row["val_fallback_entries"]), 1))
    row["selection_score_val_only"] = (
        row["val_delta_pnl"]
        + 0.04 * row["val_fallback_entries"]
        + 8.0 * row["val_wr"]
        + 0.20 * row["val_mdd"]
        - 1.50 * row["val_fallback_stop_loss"]
        - 0.35 * row["val_fallback_max_hold"]
        - 0.50 * row["val_fallback_primary_takeover"]
        - 6.0 * row["val_fallback_stop_rate"]
    )
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "build_payloads", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    val_payload, oos_payload, meta = exp._build_payloads()
    x_val = val_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_oos = oos_payload["features"].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if list(x_val.columns) != list(x_oos.columns):
        raise RuntimeError("validation/oos feature columns mismatch")
    parent_val = _base_sleeve_metrics(omega._metrics(val_payload["frame"], val_payload["dec"], fee=float(meta["fee"]), slip=float(meta["slip"]), cost_mult=3.0))
    parent_oos = _base_sleeve_metrics(omega._metrics(oos_payload["frame"], oos_payload["dec"], fee=float(meta["fee"]), slip=float(meta["slip"]), cost_mult=3.0))

    diagnostics: dict[str, Any] = {
        "method": "Report Level 1: ATR SL/TP + target-risk notional + inverse realized-vol leverage. Dynamic risk is computed from current/past bars only.",
        "parent_artifact": meta["parent_dir"],
        "feature_count": int(x_val.shape[1]),
        "features": list(x_val.columns),
        "parent_baseline": {"validation": parent_val, "oos": parent_oos},
    }
    rows: list[dict[str, Any]] = []
    for cfg_id, cfg in enumerate(CFGS):
        print(json.dumps({"stage": "cfg_start", "cfg": cfg.__dict__}, ensure_ascii=True), flush=True)
        val_risk_arrays, val_risk_diag = _risk_arrays(val_payload["frame"], cfg)
        oos_risk_arrays, oos_risk_diag = _risk_arrays(oos_payload["frame"], cfg)
        path_labels, path_diag = _path_label_table(val_payload, cfg, val_risk_arrays)
        ev_labels, ev_diag = _utility_from_path_labels(path_labels, {"stop_penalty": 0.0, "mae_penalty": 0.0, "time_penalty": 0.0})
        utility_labels, utility_diag = _utility_from_path_labels(path_labels, exp.UTILITY_CFGS[1])
        ev_vl, ev_vs, ev_ol, ev_os, ev_fit = exp._fit_predict_lower_bound(x_val, x_oos, ev_labels, "long_net", "short_net", seed=286000 + cfg_id * 1000, cal_q=0.80)
        u_vl, u_vs, u_ol, u_os, utility_fit = exp._fit_predict_lower_bound(x_val, x_oos, utility_labels, "long_utility", "short_utility", seed=286500 + cfg_id * 1000, cal_q=0.50)
        diagnostics[cfg.name] = {
            "config": cfg.__dict__,
            "validation_risk": val_risk_diag,
            "oos_risk": oos_risk_diag,
            "path_labels": path_diag,
            "ev_labels": ev_diag,
            "utility_labels": utility_diag,
            "ev_fit": ev_fit,
            "utility_fit": utility_fit,
        }
        for ev_min in (0.001, 0.002, 0.003):
            val_ev_a, val_ev_c = exp._actions_from_scores(ev_vl, ev_vs, ev_min)
            oos_ev_a, oos_ev_c = exp._actions_from_scores(ev_ol, ev_os, ev_min)
            for utility_min in (-0.001, 0.0, 0.001):
                val_a, val_c, val_filter = exp._apply_agreement(val_ev_a, val_ev_c, u_vl, u_vs, utility_min=utility_min, margin_min=0.0)
                oos_a, oos_c, oos_filter = exp._apply_agreement(oos_ev_a, oos_ev_c, u_ol, u_os, utility_min=utility_min, margin_min=0.0)
                cand = f"{cfg.name}_ev{ev_min:.3f}_u{utility_min:.3f}"
                val_m = _metrics_with_dynamic_risk(val_payload, cfg, val_risk_arrays, val_a, val_c, 0.0)
                oos_m = _metrics_with_dynamic_risk(oos_payload, cfg, oos_risk_arrays, oos_a, oos_c, 0.0)
                row = _metric_row(cand, cfg, val_m, oos_m, parent_val, parent_oos)
                row["ev_min"] = float(ev_min)
                row["utility_min"] = float(utility_min)
                row["val_filter"] = val_filter
                row["oos_filter"] = oos_filter
                rows.append(row)
    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_pnl", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "level1_atr_dynamic_risk_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl"], ascending=False).iloc[0].to_dict()
    blockers: list[str] = []
    bad_features = [c for c in x_val.columns if c == "tp_sl_action_score" or c.startswith("clean_regime4_") or c.startswith("regime4_pred_") or c.startswith("teacher_")]
    if bad_features:
        blockers.append(f"forbidden feature columns: {bad_features[:20]}")
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_level1_atr_dynamic_risk_oof_eval" if not blockers else "redteam_fail",
        "selection_policy": "validation_oof_only; OOS diagnostic only; no live export",
        "baseline": {"parent_only_validation": parent_val, "parent_only_oos": parent_oos},
        "diagnostics": diagnostics,
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "top20": ranking.head(20).to_dict(orient="records"),
        "redteam_pass": not blockers,
        "redteam_blockers": blockers,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "level1_atr_dynamic_risk_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "status": report["status"], "selected": selected, "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
