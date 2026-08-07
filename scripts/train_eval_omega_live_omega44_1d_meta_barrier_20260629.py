#!/usr/bin/env python3
from __future__ import annotations

import json
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import full_replay_omega4_4_v18_short_aged_profit_overlays_20260625 as v18  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


MODEL_ID = "omega_live_omega44_1d_meta_barrier_20260629"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"
MODEL_DIR = OUT_DIR / "models"

LIVE_PRED_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
LIVE_TRAIN_PRED = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_parent_train_inference_20260620/train_predictions_2025_jan_sep_true3head_in_sample.csv"

MAX_HOLD_BARS = 288
LEVERAGE_CAP = 5.0
TARGET_PNL = 100.0
TARGET_MDD = -20.0
TP_SL_GRID = ((0.026, 0.014), (0.035, 0.018), (0.052, 0.028))


@dataclass(frozen=True)
class MetaSpec:
    variant: str
    tp: float
    sl: float
    top_frac: float
    min_edge: float
    side_margin: float
    notional: float
    side_filter: int
    dd_governor: bool


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


def _tag(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".").replace(".", "p").replace("-", "m")


def _duration_days(frame: pd.DataFrame) -> float:
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    return max((ts.iloc[-1] - ts.iloc[0]).total_seconds() / 86400.0, 1.0e-9)


def _load_live_predictions(split: str) -> tuple[pd.DataFrame, str]:
    if split == "train":
        return pd.read_csv(LIVE_TRAIN_PRED, parse_dates=["timestamp"], low_memory=False), "omega1_regime3_expertdq_train_"
    if split == "validation":
        return pd.read_csv(LIVE_PRED_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"], low_memory=False), "omega1_regime3_expertdq_oof_"
    if split == "oos":
        return pd.read_csv(LIVE_PRED_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"], low_memory=False), "omega1_regime3_expertdq_"
    raise RuntimeError(f"unknown split: {split}")


def _prepare_omega44_outputs(device: torch.device) -> tuple[dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], dict[str, Any]]:
    report = json.loads(v18.REPORT_PATH.read_text(encoding="utf-8"))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    omega.TRAIN_CSV = Path(report["risk_model"]["train_csv"])
    omega.EVAL_CSV = Path(report["risk_model"]["eval_csv"])
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(report["risk_model"]["direction_label_dir"]),
        quality_mode=str(report["risk_model"]["quality_mode"]),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    bundle = torch.load(Path(report["baseline_bundle"]), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    out: dict[str, tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = {}
    for split, frame_key, oof in (("train", "train_raw", True), ("validation", "val_raw", True), ("oos", "oos_raw", False)):
        frame = frames[frame_key].reset_index(drop=True)
        _x, src, dec = risk._predict_decisions(
            frame,
            oof=oof,
            models=models,
            base_cols=base_cols,
            quality_threshold=float(report["contract"]["quality_threshold"]),
            device=device,
        )
        out[split] = (frame, src.reset_index(drop=True), dec.reset_index(drop=True))
    return out, {"report": report, "base_cols": base_cols}


def _align_split(
    frame: pd.DataFrame,
    omega44_src: pd.DataFrame,
    omega44_dec: pd.DataFrame,
    live_pred: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    work = frame.reset_index(names="frame_i")
    live = live_pred.copy()
    live["timestamp"] = pd.to_datetime(live["timestamp"], errors="raise")
    merged = work[["frame_i", "timestamp"]].merge(live, on="timestamp", how="inner", validate="one_to_one")
    idx = pd.to_numeric(merged["frame_i"], errors="raise").to_numpy(dtype=np.int64)
    aligned_frame = frame.iloc[idx].reset_index(drop=True)
    aligned_o44_src = omega44_src.iloc[idx].reset_index(drop=True)
    aligned_o44_dec = omega44_dec.iloc[idx].reset_index(drop=True)
    aligned_live = merged.drop(columns=["frame_i"]).reset_index(drop=True)
    diag = {
        "input_rows": int(len(frame)),
        "aligned_rows": int(len(aligned_frame)),
        "dropped_rows": int(len(frame) - len(aligned_frame)),
        "start": str(aligned_frame["timestamp"].iloc[0]) if len(aligned_frame) else None,
        "end": str(aligned_frame["timestamp"].iloc[-1]) if len(aligned_frame) else None,
    }
    return aligned_frame, aligned_o44_src, aligned_o44_dec, aligned_live, diag


def _rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    open_ = pd.to_numeric(frame["open"], errors="raise")
    volume = pd.to_numeric(frame.get("volume", pd.Series(np.zeros(len(frame)))), errors="coerce").fillna(0.0)
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    out = pd.DataFrame(index=frame.index)
    out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    out["body_pct"] = ((close - open_) / close).replace([np.inf, -np.inf], np.nan)
    out["atr14_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)
    out["volume_z24"] = (volume - volume.rolling(24, min_periods=6).mean()) / volume.rolling(24, min_periods=6).std().replace(0.0, np.nan)
    for lag in (1, 3, 6, 12, 24, 48, 96):
        out[f"ret_{lag}"] = close.pct_change(lag).replace([np.inf, -np.inf], np.nan)
    for win in (12, 24, 48, 96, 192):
        out[f"ret_vol_{win}"] = ret.rolling(win, min_periods=max(4, win // 4)).std()
        out[f"range_mean_{win}"] = out["bar_range_pct"].rolling(win, min_periods=max(4, win // 4)).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema55 = close.ewm(span=55, adjust=False).mean()
    out["ema9_21_gap"] = ((ema9 - ema21) / close).replace([np.inf, -np.inf], np.nan)
    out["ema21_55_gap"] = ((ema21 - ema55) / close).replace([np.inf, -np.inf], np.nan)
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    minute = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _numeric_model_features(df: pd.DataFrame, prefix_in: str, prefix_out: str) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        if col == "timestamp" or not str(col).startswith(prefix_in):
            continue
        name = prefix_out + str(col)[len(prefix_in) :]
        if col.endswith("router_expert"):
            dummies = pd.get_dummies(df[col].astype(str).replace({"chop_expert": "chop"}), prefix=name, dtype=np.float32)
            out = pd.concat([out, dummies], axis=1)
        else:
            out[name] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    return out


def _feature_frame(
    frame: pd.DataFrame,
    live_pred: pd.DataFrame,
    live_prefix: str,
    omega44_src: pd.DataFrame,
    omega44_dec: pd.DataFrame,
    split: str,
) -> pd.DataFrame:
    if split in ("train", "validation"):
        o44_prefix = "omega1_regime3_expertdq_oof_"
    else:
        o44_prefix = "omega1_regime3_expertdq_"
    base = _rolling_features(frame)
    live = _numeric_model_features(live_pred, live_prefix, "live_")
    o44 = _numeric_model_features(omega44_src, o44_prefix, "omega44_")
    dec = pd.DataFrame(index=frame.index)
    dec["omega44_action"] = pd.to_numeric(omega44_dec["action"], errors="raise").to_numpy(dtype=np.float32)
    dec["omega44_side"] = pd.to_numeric(omega44_dec["side"], errors="raise").to_numpy(dtype=np.float32)
    dec["omega44_quality_score"] = pd.to_numeric(omega44_dec["quality_score"], errors="raise").to_numpy(dtype=np.float32)
    dec["omega44_confidence"] = pd.to_numeric(omega44_dec["confidence"], errors="raise").to_numpy(dtype=np.float32)
    live_action = pd.to_numeric(live_pred[f"{live_prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    live_final = pd.to_numeric(live_pred[f"{live_prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    live_side = np.where(live_action == 1, 1, np.where(live_action == 2, -1, 0))
    live_final_side = np.where(live_final == 1, 1, np.where(live_final == 2, -1, 0))
    o44_side = pd.to_numeric(omega44_dec["side"], errors="raise").to_numpy(dtype=np.int64)
    dec["live_side"] = live_side.astype(np.float32)
    dec["live_final_side"] = live_final_side.astype(np.float32)
    dec["model_side_agree"] = (live_side == o44_side).astype(np.float32)
    dec["model_side_opposed"] = ((live_side * o44_side) < 0).astype(np.float32)
    out = pd.concat([base.reset_index(drop=True), live.reset_index(drop=True), o44.reset_index(drop=True), dec.reset_index(drop=True)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate feature columns: {dup[:20]}")
    return out.astype(np.float32)


def _simulate_label(
    arrays: dict[str, np.ndarray],
    signal_i: int,
    side: int,
    *,
    tp: float,
    sl: float,
    fee_eff: float,
    slip_eff: float,
) -> float:
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(signal_i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return 0.0
    entry_i = min(int(signal_i) + 1, len(arrays["open"]) - 1)
    cash = 1.0 - float(entry_fee)
    exit_i = min(entry_i + MAX_HOLD_BARS, len(arrays["close"]) - 2)
    for i in range(entry_i, exit_i + 1):
        move = _price_move_close(arrays, int(i), side=int(side), entry_price=float(entry_px), slip_eff=slip_eff)
        if move >= float(tp) or move <= -abs(float(sl)) or int(i) - int(entry_i) >= MAX_HOLD_BARS:
            filled_exit, exit_px, exit_fee, _exit_route = omega._try_execution(arrays, int(i), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
            if not filled_exit:
                continue
            raw = (exit_px - entry_px) / max(entry_px, 1.0e-12) if int(side) > 0 else (entry_px - exit_px) / max(entry_px, 1.0e-12)
            before = cash
            cash = cash * (1.0 + raw)
            cash -= before * float(exit_fee)
            return float(cash - 1.0)
    exit_px = omega._fill_price(arrays, exit_i, int(side), slip_eff, entry=False)
    raw = (exit_px - entry_px) / max(entry_px, 1.0e-12) if int(side) > 0 else (entry_px - exit_px) / max(entry_px, 1.0e-12)
    before = cash
    cash = cash * (1.0 + raw)
    cash -= before * fee_eff
    return float(cash - 1.0)


def _price_move_close(arrays: dict[str, np.ndarray], row_i: int, *, side: int, entry_price: float, slip_eff: float) -> float:
    px = float(arrays["close"][int(row_i)])
    if int(side) > 0:
        return float((px * (1.0 - slip_eff) - float(entry_price)) / max(float(entry_price), 1.0e-12))
    return float((float(entry_price) - px * (1.0 + slip_eff)) / max(float(entry_price), 1.0e-12))


def _label_arrays(frame: pd.DataFrame, *, tp: float, sl: float, fee: float, slip: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    n = max(0, len(frame) - MAX_HOLD_BARS - 2)
    y_long = np.zeros(len(frame), dtype=np.float32)
    y_short = np.zeros(len(frame), dtype=np.float32)
    for i in range(n):
        y_long[i] = _simulate_label(arrays, i, 1, tp=tp, sl=sl, fee_eff=fee_eff, slip_eff=slip_eff)
        y_short[i] = _simulate_label(arrays, i, -1, tp=tp, sl=sl, fee_eff=fee_eff, slip_eff=slip_eff)
    mask = np.arange(len(frame)) < n
    diag = {
        "rows": int(len(frame)),
        "usable_rows": int(n),
        "long_positive_rate": float((y_long[mask] > 0.0).mean()) if n else 0.0,
        "short_positive_rate": float((y_short[mask] > 0.0).mean()) if n else 0.0,
        "long_p50": float(np.quantile(y_long[mask], 0.50)) if n else 0.0,
        "short_p50": float(np.quantile(y_short[mask], 0.50)) if n else 0.0,
        "long_p90": float(np.quantile(y_long[mask], 0.90)) if n else 0.0,
        "short_p90": float(np.quantile(y_short[mask], 0.90)) if n else 0.0,
    }
    return y_long, y_short, diag


def _fit_models(x_train: pd.DataFrame, y_long: np.ndarray, y_short: np.ndarray, mask: np.ndarray, seed: int) -> tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor]:
    cols = list(x_train.columns)
    x = x_train.loc[mask, cols].to_numpy(dtype=np.float32)
    common = dict(max_iter=120, learning_rate=0.045, max_leaf_nodes=15, l2_regularization=1.0, min_samples_leaf=45, random_state=int(seed))
    long_model = HistGradientBoostingRegressor(**common)
    short_model = HistGradientBoostingRegressor(**{**common, "random_state": int(seed) + 17})
    long_model.fit(x, np.asarray(y_long, dtype=np.float32)[mask])
    short_model.fit(x, np.asarray(y_short, dtype=np.float32)[mask])
    return long_model, short_model


def _predict_pair(models: tuple[HistGradientBoostingRegressor, HistGradientBoostingRegressor], x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32)
    return models[0].predict(arr).astype(np.float64), models[1].predict(arr).astype(np.float64)


def _build_signal(pred_long: np.ndarray, pred_short: np.ndarray, spec: MetaSpec) -> np.ndarray:
    score = np.maximum(pred_long, pred_short)
    side = np.where(pred_long >= pred_short, 1, -1).astype(np.int64)
    edge_gap = np.abs(pred_long - pred_short)
    active = (score >= float(spec.min_edge)) & (edge_gap >= float(spec.side_margin))
    if int(spec.side_filter) != 0:
        active &= side == int(spec.side_filter)
    return np.where(active, side, 0).astype(np.int64)


def _notional(spec: MetaSpec, dd: float) -> float:
    n = float(spec.notional)
    if spec.dd_governor:
        if dd <= -0.16:
            n *= 0.35
        elif dd <= -0.12:
            n *= 0.55
        elif dd <= -0.08:
            n *= 0.75
    return float(np.clip(n, 0.0, LEVERAGE_CAP))


def _replay(frame: pd.DataFrame, signal: np.ndarray, spec: MetaSpec, *, fee: float, slip: float) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    notional = 0.0
    leverage = LEVERAGE_CAP
    margin_fraction = 0.0
    mfe = 0.0
    mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = margin_sum = leverage_sum = 0.0
    max_hold_seen = 0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = _price_move_close(arrays, int(i), side=pos, entry_price=entry_price, slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        dd = eq / max(peak, 1.0e-12) - 1.0
        mdd = min(mdd, dd)
        if pos != 0:
            hold = max(int(i) - int(entry_i), 0)
            max_hold_seen = max(max_hold_seen, hold)
            reason = ""
            if move >= float(spec.tp):
                reason = "take_profit"
            elif move <= -abs(float(spec.sl)):
                reason = "stop_loss"
            elif hold >= MAX_HOLD_BARS:
                reason = "max_hold_1d"
            if reason:
                filled, exit_px, exit_fee, route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * float(exit_fee) * notional
                ret = cash / max(entry_equity, 1.0e-12) - 1.0
                trades += 1
                win = int(cash > entry_equity)
                wins += win
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "entry_signal_i": int(entry_signal_i),
                        "entry_i": int(entry_i),
                        "exit_i": int(i),
                        "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                        "exit_timestamp": str(frame["timestamp"].iloc[int(i)]),
                        "side": int(pos),
                        "reason": reason,
                        "win": int(win),
                        "raw_exit_price_move": float(raw_exit),
                        "mfe_price_move": float(mfe),
                        "mae_price_move": float(mae),
                        "trade_return": float(ret),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "take_profit": float(spec.tp),
                        "stop_loss": float(spec.sl),
                        "hold_bars": int(hold),
                        "cash_after": float(cash),
                        "exit_route": route,
                    }
                )
                pos = 0
                continue
        if pos != 0:
            continue
        side = int(signal[int(i)]) if int(i) < len(signal) else 0
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        entry_dd = cash / max(peak, 1.0e-12) - 1.0
        row_notional = _notional(spec, entry_dd)
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        notional = row_notional
        leverage = LEVERAGE_CAP
        margin_fraction = notional / max(leverage, 1.0e-12)
        cash -= cash * float(entry_fee) * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        margin_sum += margin_fraction
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        hold = max(len(frame) - 1 - int(entry_i), 0)
        max_hold_seen = max(max_hold_seen, hold)
        ret = cash / max(entry_equity, 1.0e-12) - 1.0
        trades += 1
        win = int(cash > entry_equity)
        wins += win
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(
            {
                "entry_signal_i": int(entry_signal_i),
                "entry_i": int(entry_i),
                "exit_i": int(len(frame) - 1),
                "entry_timestamp": str(frame["timestamp"].iloc[int(entry_signal_i)]),
                "exit_timestamp": str(frame["timestamp"].iloc[-1]),
                "side": int(pos),
                "reason": "forced_end",
                "win": int(win),
                "raw_exit_price_move": float(raw_exit),
                "mfe_price_move": float(mfe),
                "mae_price_move": float(mae),
                "trade_return": float(ret),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "take_profit": float(spec.tp),
                "stop_loss": float(spec.sl),
                "hold_bars": int(hold),
                "cash_after": float(cash),
                "exit_route": "forced_end",
            }
        )
    ledger = pd.DataFrame(rows)
    if not ledger.empty:
        max_hold_seen = int(max(max_hold_seen, int((pd.to_numeric(ledger["exit_i"]) - pd.to_numeric(ledger["entry_i"])).max())))
    n_entries = max(long_entries + short_entries, 1)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(trades),
            "wr": float(wins / trades) if trades else 0.0,
            "trades_per_day": float(trades / _duration_days(frame)),
            "avg_notional": float(notional_sum / n_entries),
            "avg_margin_fraction": float(margin_sum / n_entries),
            "avg_leverage": float(leverage_sum / n_entries),
            "max_leverage": LEVERAGE_CAP if trades else 0.0,
            "max_hold_bars": int(max_hold_seen),
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_reasons": reasons,
        },
        ledger,
    )


def _candidate_specs(tp: float, sl: float, pred_long: np.ndarray, pred_short: np.ndarray) -> list[MetaSpec]:
    max_score = np.maximum(pred_long, pred_short)
    positive = max_score[np.isfinite(max_score)]
    specs: list[MetaSpec] = []
    for top_frac in (0.005, 0.01, 0.02, 0.04, 0.08, 0.12, 0.18, 0.25, 0.35):
        q = 1.0 - float(top_frac)
        edge_thr = float(np.quantile(positive, q)) if len(positive) else 0.0
        for extra_edge in (0.0, 0.0015, 0.003, 0.006):
            min_edge = edge_thr + float(extra_edge)
            for side_margin in (0.0, 0.001, 0.0025, 0.005):
                for notional in (0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5):
                    for side_filter in (0, -1):
                        for dd_governor in (False, True):
                            specs.append(
                                MetaSpec(
                                    variant=(
                                        f"tp{_tag(tp)}_sl{_tag(sl)}_top{_tag(top_frac)}"
                                        f"_edge{_tag(min_edge)}_gap{_tag(side_margin)}_n{_tag(notional)}"
                                        f"{'_shortonly' if side_filter < 0 else ''}"
                                        f"{'_ddgov' if dd_governor else ''}"
                                    ),
                                    tp=float(tp),
                                    sl=float(sl),
                                    top_frac=float(top_frac),
                                    min_edge=float(min_edge),
                                    side_margin=float(side_margin),
                                    notional=float(notional),
                                    side_filter=int(side_filter),
                                    dd_governor=bool(dd_governor),
                                )
                            )
    return specs


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    device = parent._device("cuda")
    omega44, meta = _prepare_omega44_outputs(device)
    fee, slip = omega._load_fee_slip()

    split_data: dict[str, dict[str, Any]] = {}
    align_diag: dict[str, Any] = {}
    for split, (frame, o44_src, o44_dec) in omega44.items():
        live_pred, live_prefix = _load_live_predictions(split)
        aligned_frame, aligned_o44_src, aligned_o44_dec, aligned_live, diag = _align_split(frame, o44_src, o44_dec, live_pred)
        align_diag[split] = diag
        features = _feature_frame(aligned_frame, aligned_live, live_prefix, aligned_o44_src, aligned_o44_dec, split)
        split_data[split] = {
            "frame": aligned_frame,
            "features": features,
            "live_prefix": live_prefix,
        }

    feature_cols = list(split_data["train"]["features"].columns)
    for split in ("validation", "oos"):
        missing = sorted(set(feature_cols) - set(split_data[split]["features"].columns))
        extra = sorted(set(split_data[split]["features"].columns) - set(feature_cols))
        if missing or extra:
            raise RuntimeError(f"{split} feature contract mismatch missing={missing[:10]} extra={extra[:10]}")
        split_data[split]["features"] = split_data[split]["features"].reindex(columns=feature_cols).astype(np.float32)

    rows: list[dict[str, Any]] = []
    label_diag: dict[str, Any] = {}
    model_paths: dict[str, dict[str, str]] = {}
    best_ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for pair_idx, (tp, sl) in enumerate(TP_SL_GRID):
        print(json.dumps({"stage": "label_start", "tp": tp, "sl": sl}, ensure_ascii=False), flush=True)
        y_long, y_short, diag = _label_arrays(split_data["train"]["frame"], tp=tp, sl=sl, fee=fee, slip=slip)
        mask = np.arange(len(y_long)) < int(diag["usable_rows"])
        label_diag[f"tp{tp}_sl{sl}"] = diag
        models = _fit_models(split_data["train"]["features"], y_long, y_short, mask, seed=260629 + pair_idx * 100)
        pair_tag = f"tp{_tag(tp)}_sl{_tag(sl)}"
        long_path = MODEL_DIR / f"{pair_tag}_long_hgb.pkl"
        short_path = MODEL_DIR / f"{pair_tag}_short_hgb.pkl"
        with long_path.open("wb") as f:
            pickle.dump(models[0], f)
        with short_path.open("wb") as f:
            pickle.dump(models[1], f)
        model_paths[pair_tag] = {"long": str(long_path), "short": str(short_path)}
        val_long, val_short = _predict_pair(models, split_data["validation"]["features"])
        oos_long, oos_short = _predict_pair(models, split_data["oos"]["features"])
        specs = _candidate_specs(tp, sl, val_long, val_short)
        for idx, spec in enumerate(specs, start=1):
            rec: dict[str, Any] = {
                "variant": spec.variant,
                "tp": spec.tp,
                "sl": spec.sl,
                "top_frac": spec.top_frac,
                "min_edge": spec.min_edge,
                "side_margin": spec.side_margin,
                "notional": spec.notional,
                "side_filter": spec.side_filter,
                "dd_governor": spec.dd_governor,
                "max_hold_contract_bars": MAX_HOLD_BARS,
                "leverage_cap": LEVERAGE_CAP,
            }
            for split, pred_pair in (("validation", (val_long, val_short)), ("oos", (oos_long, oos_short))):
                signal = _build_signal(pred_pair[0], pred_pair[1], spec)
                metrics, ledger = _replay(split_data[split]["frame"], signal, spec, fee=fee, slip=slip)
                for key, value in metrics.items():
                    rec[f"{split}_{key}"] = json.dumps(value, ensure_ascii=False, sort_keys=True) if key == "exit_reasons" else value
                if len(rows) < 5:
                    best_ledgers[(split, spec.variant)] = ledger
            rec["pass_target"] = (
                rec["validation_pnl"] >= TARGET_PNL
                and rec["oos_pnl"] >= TARGET_PNL
                and rec["validation_mdd"] >= TARGET_MDD
                and rec["oos_mdd"] >= TARGET_MDD
                and rec["validation_max_hold_bars"] <= MAX_HOLD_BARS
                and rec["oos_max_hold_bars"] <= MAX_HOLD_BARS
                and rec["validation_max_leverage"] <= LEVERAGE_CAP
                and rec["oos_max_leverage"] <= LEVERAGE_CAP
            )
            rec["target_score"] = min(float(rec["validation_pnl"]), float(rec["oos_pnl"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["validation_mdd"])) - 4.0 * max(0.0, TARGET_MDD - float(rec["oos_mdd"]))
            rows.append(rec)
            if idx % 500 == 0:
                print(json.dumps({"stage": "grid_progress", "pair": pair_tag, "idx": idx, "total": len(specs), "last_val": rec["validation_pnl"], "last_oos": rec["oos_pnl"]}, ensure_ascii=False), flush=True)

    grid = pd.DataFrame(rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "meta_barrier_grid.csv", index=False)
    passed = grid[grid["pass_target"]].copy()
    passed.to_csv(OUT_DIR / "target_pass.csv", index=False)

    # Save ledgers for top candidates after final ranking.
    saved: list[str] = []
    for _, row in grid.head(12).iterrows():
        pair_tag = f"tp{_tag(float(row.tp))}_sl{_tag(float(row.sl))}"
        with open(model_paths[pair_tag]["long"], "rb") as f:
            long_model = pickle.load(f)
        with open(model_paths[pair_tag]["short"], "rb") as f:
            short_model = pickle.load(f)
        spec = MetaSpec(
            variant=str(row.variant),
            tp=float(row.tp),
            sl=float(row.sl),
            top_frac=float(row.top_frac),
            min_edge=float(row.min_edge),
            side_margin=float(row.side_margin),
            notional=float(row.notional),
            side_filter=int(row.side_filter),
            dd_governor=bool(row.dd_governor),
        )
        for split in ("validation", "oos"):
            pred_long, pred_short = _predict_pair((long_model, short_model), split_data[split]["features"])
            signal = _build_signal(pred_long, pred_short, spec)
            _metrics, ledger = _replay(split_data[split]["frame"], signal, spec, fee=fee, slip=slip)
            out_path = LEDGER_DIR / f"{split}_{spec.variant}_ledger.csv"
            ledger.to_csv(out_path, index=False)
            saved.append(str(out_path))

    report = {
        "model_id": MODEL_ID,
        "source_models": {
            "live_model": "omega3_aggressive_compensated_scale200_cap090_20260618 parent outputs",
            "omega44_model": "omega4_4_v18_baseline_20260624 parent outputs",
        },
        "method": "Train long/short HGB regressors on train-only one-day triple-barrier net return labels; select thresholds/notional on validation; OOS readout after selection.",
        "contract": {
            "validation_pnl_min": TARGET_PNL,
            "oos_pnl_min": TARGET_PNL,
            "mdd_floor_pct": TARGET_MDD,
            "max_hold_bars": MAX_HOLD_BARS,
            "leverage_cap": LEVERAGE_CAP,
            "tp_sl_contract": "direct price-move barriers",
        },
        "alignment": align_diag,
        "label_diag": label_diag,
        "feature_count": int(len(feature_cols)),
        "feature_columns": feature_cols,
        "model_paths": model_paths,
        "pass_count": int(len(passed)),
        "top20": grid.head(20).to_dict(orient="records"),
        "passed": passed.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "grid": str(OUT_DIR / "meta_barrier_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
        },
        "notes": [
            "OOS was not used to fit labels, models, thresholds, or notional selection.",
            "Train live-parent predictions are in-sample historical inference; validation and OOS remain out-of-sample readouts for this meta layer.",
        ],
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "pass_count": int(len(passed)), "top": grid.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
