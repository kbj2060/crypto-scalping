#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_btc_exitonly_20260806 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_btc_20260708 as omega  # noqa: E402
import train_eval_omega4_1_exit_head_price_move_sltp_retrain_20260622 as price_exit  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_btc_exitonly_20260806 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "btc_omega4_2_trade_risk_sidecar_exitonly_20260806"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "btc_omega4_3head_parent72_loose_entry_quality_20260708_zig075_20260708"
    / "true_3head_tabm_bundle.pt"
)
MARGIN_CFG_KEYS = ("min_scale", "max_scale", "temp", "floor", "cap", "long_scale", "short_scale")
LEVERAGE_CFG_KEYS = (
    "leverage_min",
    "leverage_max",
    "leverage_temp",
    "leverage_floor",
    "leverage_cap",
    "long_leverage_scale",
    "short_leverage_scale",
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _duration_days(frame: pd.DataFrame) -> float:
    return max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1.0e-9)


def _read_risk_trend_labels(label_dir: Path, year: int) -> pd.DataFrame:
    path = Path(label_dir) / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, usecols=["timestamp", "zigzag_action"], parse_dates=["timestamp"], low_memory=False)
    labels = labels.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    action = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(action).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"{path} invalid trend action classes: {invalid}")
    labels["zigzag_action"] = action
    return labels


def _align_risk_trend_actions(frame: pd.DataFrame, label_dir: Path, year: int, name: str) -> np.ndarray:
    labels = _read_risk_trend_labels(Path(label_dir), int(year))
    aligned_frame, aligned_labels = omega._align(frame[["timestamp"]], labels, name)
    if len(aligned_frame) != len(frame):
        raise RuntimeError(f"{name}: trend label alignment changed row count: {len(frame)} -> {len(aligned_frame)}")
    return pd.to_numeric(aligned_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)


def _trend_action_summary(action: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(action, dtype=np.int64)
    counts = pd.Series(arr).value_counts().sort_index()
    return {
        "rows": int(len(arr)),
        "counts": {str(int(k)): int(v) for k, v in counts.items()},
        "active_ratio": float((arr != 0).mean()) if len(arr) else 0.0,
    }


def _risk_target_from_trend_alignment(
    train_ledger: pd.DataFrame,
    trend_action: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    entry_idx = pd.to_numeric(train_ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
    side = pd.to_numeric(train_ledger["side"], errors="raise").to_numpy(dtype=np.int64)
    if len(trend_action) <= int(entry_idx.max(initial=0)):
        raise RuntimeError("trend action length is shorter than max trade entry_signal_i")
    entry_action = np.asarray(trend_action, dtype=np.int64)[entry_idx]
    trend_side = np.where(entry_action == 1, 1, np.where(entry_action == 2, -1, 0)).astype(np.int64)
    alignment = (side * trend_side).astype(np.float64)
    counts = pd.Series(alignment).value_counts().sort_index()
    diag = {
        "trade_rows": int(len(train_ledger)),
        "entry_trend_counts": {str(int(k)): int(v) for k, v in pd.Series(entry_action).value_counts().sort_index().items()},
        "alignment_counts": {str(float(k)): int(v) for k, v in counts.items()},
        "alignment_active_ratio": float((alignment != 0.0).mean()) if len(alignment) else 0.0,
        "aligned_ratio": float((alignment > 0.0).mean()) if len(alignment) else 0.0,
        "opposed_ratio": float((alignment < 0.0).mean()) if len(alignment) else 0.0,
    }
    return alignment, diag


def _predict_decisions(
    frame: pd.DataFrame,
    *,
    oof: bool,
    models: dict[str, dict[str, Any]],
    base_cols: list[str],
    quality_threshold: float,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    x = parent._base_input(frame, base_cols)
    preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    direction = parent._routed(preds, route, "direction", 3)
    quality = parent._routed(preds, route, "quality", 3)
    prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
    src = parent._prediction_output(frame, direction, quality, threshold=float(quality_threshold), prefix=prefix)
    return x, src, parent._to_decisions(src, oof=oof)


def _risk_feature_frame(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    dec: pd.DataFrame,
    base_cols: list[str],
    *,
    atr_pct: np.ndarray,
    feature_mode: str,
) -> pd.DataFrame:
    if feature_mode == "all":
        base = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    elif feature_mode == "parent_outputs":
        base = pd.DataFrame(index=frame.index)
    else:
        raise RuntimeError(f"unknown risk feature mode: {feature_mode}")
    numeric_src = src.drop(columns=["timestamp"], errors="ignore").copy()
    numeric_src = numeric_src.rename(
        columns={
            c: c.replace("omega1_regime3_expertdq_oof_", "parent_").replace("omega1_regime3_expertdq_", "parent_")
            for c in numeric_src.columns
        }
    )
    for col in list(numeric_src.columns):
        if numeric_src[col].dtype == object:
            if col.endswith("router_expert"):
                dummies = pd.get_dummies(numeric_src[col].astype(str), prefix=col, dtype=np.float32)
                numeric_src = pd.concat([numeric_src.drop(columns=[col]), dummies], axis=1)
            else:
                numeric_src = numeric_src.drop(columns=[col])
    numeric_src = numeric_src.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.DataFrame(index=dec.index)
    for col in ("action", "side", "quality_score", "confidence", "notional_exposure", "leverage", "position_fraction", "take_profit", "stop_loss"):
        d[f"decision_{col}"] = pd.to_numeric(dec[col], errors="raise").to_numpy(dtype=np.float64)
    d["decision_rr"] = d["decision_take_profit"] / np.maximum(np.abs(d["decision_stop_loss"]), 1.0e-8)
    d["atr_pct_runtime"] = np.asarray(atr_pct, dtype=np.float64)
    out = pd.concat([base.reset_index(drop=True), numeric_src.reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate risk feature columns: {dup[:20]}")
    return out.astype(np.float32)


def _feature_matrix(features: pd.DataFrame, columns: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    if columns is None:
        cols = list(features.columns)
    else:
        cols = list(columns)
        missing = [c for c in cols if c not in features.columns]
        if missing:
            raise RuntimeError(f"risk feature contract mismatch, missing columns: {missing[:20]}")
    return features.reindex(columns=cols).astype(np.float32), cols


def _read_context_features(feature_dir: Path, split: str, frame: pd.DataFrame) -> pd.DataFrame:
    path = Path(feature_dir) / f"{split}_context_features.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    ctx = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in ctx.columns:
        raise RuntimeError(f"{path} missing timestamp column")
    ctx_ts = pd.to_datetime(ctx["timestamp"], errors="raise").reset_index(drop=True)
    frame_ts = pd.to_datetime(frame["timestamp"], errors="raise").reset_index(drop=True)
    if len(ctx_ts) != len(frame_ts) or not ctx_ts.equals(frame_ts):
        raise RuntimeError(f"{split}: context feature timestamps do not match prepared frame")
    feature_cols = [c for c in ctx.columns if c != "timestamp"]
    if not feature_cols:
        raise RuntimeError(f"{path} has no context feature columns")
    bad = [c for c in feature_cols if not str(c).startswith("trend_ctx_")]
    if bad:
        raise RuntimeError(f"{path} has non trend_ctx_ context columns: {bad[:20]}")
    out = ctx[feature_cols].apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan)
    if out.isna().any().any():
        null_cols = out.columns[out.isna().any()].tolist()
        raise RuntimeError(f"{path} has NaN context feature columns: {null_cols[:20]}")
    return out.astype(np.float32)


def _append_context_features(features: pd.DataFrame, context: pd.DataFrame, *, split: str) -> pd.DataFrame:
    if len(features) != len(context):
        raise RuntimeError(f"{split}: context feature row count mismatch: {len(features)} != {len(context)}")
    overlap = sorted(set(features.columns).intersection(context.columns))
    if overlap:
        raise RuntimeError(f"{split}: duplicate context feature columns: {overlap[:20]}")
    return pd.concat([features.reset_index(drop=True), context.reset_index(drop=True)], axis=1).astype(np.float32)


def _load_precomputed_prediction(prediction_dir: Path, split: str, tag: str, frame: pd.DataFrame) -> pd.DataFrame:
    path = Path(prediction_dir) / f"{split}_predictions_{tag}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    src = pd.read_csv(path)
    if "timestamp" not in src.columns:
        raise RuntimeError(f"{path} missing timestamp column")
    pred_ts = pd.to_datetime(src["timestamp"], errors="raise").reset_index(drop=True)
    frame_ts = pd.to_datetime(frame["timestamp"], errors="raise").reset_index(drop=True)
    if len(pred_ts) != len(frame_ts) or not pred_ts.equals(frame_ts):
        raise RuntimeError(f"{split}: precomputed prediction timestamps do not match prepared frame")
    return src


def _prepare_exit_runtime(
    base_x: pd.DataFrame,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    *,
    exit_extra_frame: pd.DataFrame | None = None,
) -> tuple[np.ndarray, dict[str, tuple[parent.ThreeHeadTabM, np.ndarray, np.ndarray]], list[int], list[int], np.ndarray | None]:
    first_scaler = next(iter(loaded_models.values()))[1]
    cols = list(first_scaler["columns"])
    if list(base_x.columns) != cols:
        raise RuntimeError("exit runtime base feature column contract mismatch")
    pos_idx = [cols.index(c) for c in parent.POS_COLS]
    exit_only_cols = list(getattr(parent, "EXIT_ONLY_COLS", []))
    exit_only_idx = [cols.index(c) for c in exit_only_cols] if exit_only_cols else []
    exit_only_values: np.ndarray | None = None
    if exit_only_idx:
        if exit_extra_frame is None:
            raise RuntimeError("exit_extra_frame required to supply EXIT_ONLY_COLS real values at replay time")
        exit_only_values = exit_extra_frame[exit_only_cols].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
    runtime: dict[str, tuple[parent.ThreeHeadTabM, np.ndarray, np.ndarray]] = {}
    for expert, (model, scaler) in loaded_models.items():
        if list(scaler["columns"]) != cols:
            raise RuntimeError(f"exit runtime scaler column mismatch for {expert}")
        runtime[expert] = (
            model,
            np.asarray(scaler["mean"], dtype=np.float32),
            np.asarray(scaler["std"], dtype=np.float32),
        )
    return base_x.to_numpy(dtype=np.float32), runtime, pos_idx, exit_only_idx, exit_only_values


@torch.no_grad()
def _predict_exit_prob_one(
    base_np: np.ndarray,
    runtime: dict[str, tuple[parent.ThreeHeadTabM, np.ndarray, np.ndarray]],
    pos_idx: list[int],
    *,
    row_i: int,
    expert: str,
    pos_values: list[float],
    device: torch.device,
    exit_only_idx: list[int] | None = None,
    exit_only_values: np.ndarray | None = None,
) -> float:
    model, mean, std = runtime[expert]
    row = base_np[int(row_i)].copy()
    row[np.asarray(pos_idx, dtype=np.int64)] = np.asarray(pos_values, dtype=np.float32)
    if exit_only_idx and exit_only_values is not None:
        row[np.asarray(exit_only_idx, dtype=np.int64)] = exit_only_values[int(row_i)]
    x = ((row - mean) / std).reshape(1, -1).astype(np.float32)
    probs = torch.softmax(model(torch.from_numpy(x).to(device))["exit"], dim=-1).mean(dim=1)
    return float(probs.detach().cpu().numpy()[0, 1])


@torch.no_grad()
def _replay_with_risk(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[parent.ThreeHeadTabM, dict[str, Any]]],
    *,
    risk_margin_fraction: np.ndarray | None,
    risk_leverage: np.ndarray | None,
    exit_threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    notional_scaled_sltp: bool,
    exit_sizing_input_mode: str = "actual",
    exit_context_features: pd.DataFrame | None = None,
    exit_trend_threshold_scale: float = 0.0,
    exit_threshold_floor: float = 0.0,
    exit_threshold_cap: float = 1.0,
    device: torch.device,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if risk_leverage is not None and risk_margin_fraction is None:
        raise RuntimeError("risk_leverage requires risk_margin_fraction")
    if str(exit_sizing_input_mode) not in {"actual", "baseline"}:
        raise RuntimeError(f"unknown exit_sizing_input_mode: {exit_sizing_input_mode}")
    if exit_context_features is not None and len(exit_context_features) != len(frame):
        raise RuntimeError(f"exit context row count mismatch: {len(exit_context_features)} != {len(frame)}")
    if float(exit_trend_threshold_scale) != 0.0 and exit_context_features is None:
        raise RuntimeError("exit_trend_threshold_scale requires exit_context_features")
    if exit_context_features is not None and "trend_ctx_long_minus_short" not in exit_context_features.columns:
        raise RuntimeError("exit context features missing trend_ctx_long_minus_short")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_signal_i = 0
    entry_fee = 0.0
    notional = 0.0
    leverage = 1.0
    margin_fraction = 0.0
    exit_input_notional = 0.0
    exit_input_leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    margin_sum = 0.0
    reasons: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    route = hard._route_id(frame)
    base_np, exit_runtime, pos_idx, exit_only_idx, exit_only_values = _prepare_exit_runtime(base_x, loaded_models, exit_extra_frame=frame)

    for i in range(0, len(frame) - 2):
        if pos != 0:
            move = price_exit._price_move(arrays, int(i), side=pos, entry_price=float(entry_price), slip_eff=slip_eff)
            unreal = move * notional
            mfe = max(mfe, move)
            mae = min(mae, move)
            eq = cash * (1.0 + unreal)
        else:
            move = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1.0e-12) - 1.0)

        if pos != 0:
            reason = ""
            exit_prob = 0.0
            exit_support = 0.0
            effective_exit_threshold = float(exit_threshold)
            if take_profit > 0.0 and move >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and move <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(move)) / max(abs(float(mfe)), 1.0e-8) if mfe > 0.0 else 0.0
                expert = hard.EXPERT_NAMES[int(route[i])]
                prob = _predict_exit_prob_one(
                    base_np,
                    exit_runtime,
                    pos_idx,
                    row_i=int(i),
                    expert=expert,
                    pos_values=[
                        float(pos),
                        float(hold),
                        float(move),
                        float(mfe),
                        float(mae),
                        float(np.clip(giveback, 0.0, 10.0)),
                        float(take_profit - move),
                        float(move + abs(stop_loss)),
                        float(exit_input_notional),
                        float(exit_input_leverage),
                        float(exit_input_notional * exit_input_leverage),
                        float(take_profit),
                        float(stop_loss),
                    ],
                    device=device,
                    exit_only_idx=exit_only_idx,
                    exit_only_values=exit_only_values,
                )
                exit_prob = float(prob)
                if exit_context_features is not None and float(exit_trend_threshold_scale) != 0.0:
                    trend_diff = float(exit_context_features["trend_ctx_long_minus_short"].iloc[int(i)])
                    exit_support = float(pos) * trend_diff
                    effective_exit_threshold = float(
                        np.clip(
                            float(exit_threshold) + float(exit_trend_threshold_scale) * exit_support,
                            float(exit_threshold_floor),
                            float(exit_threshold_cap),
                        )
                    )
                if prob >= effective_exit_threshold:
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
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
                        "trade_return": float(trade_return),
                        "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                        "notional": float(notional),
                        "margin_fraction": float(margin_fraction),
                        "leverage": float(leverage),
                        "exit_input_notional": float(exit_input_notional),
                        "exit_input_leverage": float(exit_input_leverage),
                        "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                        "exit_prob": float(exit_prob),
                        "exit_trend_support": float(exit_support),
                        "exit_threshold_effective": float(effective_exit_threshold),
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                    }
                )
                pos = 0
                continue
        if pos != 0 or not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, fee_paid, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        base_leverage = float(row.get("leverage", 1.0) or 1.0)
        row_leverage = base_leverage
        if risk_leverage is not None:
            row_leverage = float(risk_leverage[int(i)])
        base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
        if risk_margin_fraction is None:
            row_margin = base_notional / max(row_leverage, 1.0e-12)
            row_notional = base_notional
        else:
            row_margin = float(risk_margin_fraction[int(i)])
            row_notional = row_margin * row_leverage
        if row_notional <= 0.0:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        entry_fee = float(fee_paid)
        leverage = row_leverage
        margin_fraction = row_margin
        notional = row_notional
        if str(exit_sizing_input_mode) == "baseline":
            exit_input_notional = base_notional
            exit_input_leverage = base_leverage
        else:
            exit_input_notional = row_notional
            exit_input_leverage = row_leverage
        base_take_profit = float(row.get("take_profit", 0.0) or 0.0)
        base_stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        if bool(notional_scaled_sltp):
            take_profit = base_take_profit * row_notional
            stop_loss = base_stop_loss * row_notional
        else:
            take_profit = base_take_profit
            stop_loss = base_stop_loss
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        margin_sum += margin_fraction
        mfe = 0.0
        mae = 0.0

    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1.0e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1.0e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trade_return = cash / max(entry_equity, 1.0e-12) - 1.0
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
                "trade_return": float(trade_return),
                "net_per_notional": float(trade_return / max(notional, 1.0e-12)),
                "notional": float(notional),
                "margin_fraction": float(margin_fraction),
                "leverage": float(leverage),
                "exit_input_notional": float(exit_input_notional),
                "exit_input_leverage": float(exit_input_leverage),
                "exit_input_exposure": float(exit_input_notional * exit_input_leverage),
                "exit_prob": 0.0,
                "exit_trend_support": 0.0,
                "exit_threshold_effective": float(exit_threshold),
                "take_profit": float(take_profit),
                "stop_loss": float(stop_loss),
            }
        )

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
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "exit_reasons": reasons,
        },
        pd.DataFrame(rows),
    )


def _risk_margins(
    dec: pd.DataFrame,
    score: np.ndarray,
    *,
    train_q50: float,
    train_iqr: float,
    min_scale: float,
    max_scale: float,
    temp: float,
    floor: float,
    cap: float,
    long_scale: float = 1.0,
    short_scale: float = 1.0,
) -> np.ndarray:
    leverage = pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float64)
    base_notional = pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    base_margin = base_notional / np.maximum(leverage, 1.0e-12)
    z = np.clip((np.asarray(score, dtype=np.float64) - float(train_q50)) / max(float(train_iqr), 1.0e-8), -8.0, 8.0)
    unit = 1.0 / (1.0 + np.exp(-float(temp) * z))
    scale = float(min_scale) + (float(max_scale) - float(min_scale)) * unit
    margin = np.clip(base_margin * scale, float(floor), float(cap))
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    margin[side > 0] *= float(long_scale)
    margin[side < 0] *= float(short_scale)
    margin = np.clip(margin, float(floor), float(cap))
    margin[~omega._active(dec)] = 0.0
    if not np.isfinite(margin).all():
        raise RuntimeError("non-finite risk margin output")
    return margin


def _risk_leverage(
    dec: pd.DataFrame,
    score: np.ndarray,
    *,
    train_q50: float,
    train_iqr: float,
    leverage_min: float,
    leverage_max: float,
    leverage_temp: float,
    leverage_floor: float,
    leverage_cap: float,
    long_leverage_scale: float = 1.0,
    short_leverage_scale: float = 1.0,
) -> np.ndarray:
    if float(leverage_max) < float(leverage_min):
        raise RuntimeError("leverage_max must be >= leverage_min")
    z = np.clip((np.asarray(score, dtype=np.float64) - float(train_q50)) / max(float(train_iqr), 1.0e-8), -8.0, 8.0)
    unit = 1.0 / (1.0 + np.exp(-float(leverage_temp) * z))
    leverage = float(leverage_min) + (float(leverage_max) - float(leverage_min)) * unit
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    leverage[side > 0] *= float(long_leverage_scale)
    leverage[side < 0] *= float(short_leverage_scale)
    leverage = np.clip(leverage, float(leverage_floor), float(leverage_cap))
    leverage[~omega._active(dec)] = 0.0
    if not np.isfinite(leverage).all():
        raise RuntimeError("non-finite risk leverage output")
    return leverage


def _build_risk_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=220,
            learning_rate=0.035,
            l2_regularization=0.10,
            max_leaf_nodes=15,
            min_samples_leaf=18,
            random_state=int(seed),
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=700,
            max_depth=6,
            min_samples_leaf=4,
            random_state=int(seed),
            n_jobs=-1,
        )
    if kind == "random_forest":
        return RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=6,
            random_state=int(seed),
            n_jobs=-1,
        )
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=260,
            learning_rate=0.025,
            max_depth=2,
            min_samples_leaf=8,
            subsample=0.75,
            random_state=int(seed),
        )
    raise RuntimeError(f"unknown risk model kind: {kind}")


def _fit_side_split_models(
    kind: str,
    x_train_trade: pd.DataFrame,
    y_train_trade: np.ndarray,
    side_train_trade: np.ndarray,
    sample_weight: np.ndarray,
    *,
    seed: int,
) -> dict[int, Any]:
    models: dict[int, Any] = {}
    for side in (-1, 1):
        mask = np.asarray(side_train_trade, dtype=np.int64) == int(side)
        if int(mask.sum()) < 12:
            raise RuntimeError(f"not enough side-split risk samples for side={side}: {int(mask.sum())}")
        model = _build_risk_model(kind, int(seed) + (11 if side < 0 else 17))
        model.fit(x_train_trade.loc[mask], np.asarray(y_train_trade, dtype=np.float64)[mask], sample_weight=np.asarray(sample_weight, dtype=np.float64)[mask])
        models[int(side)] = model
    return models


def _predict_side_split_models(models: dict[int, Any], x_all: pd.DataFrame, side_all: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x_all), dtype=np.float64)
    side_arr = np.asarray(side_all, dtype=np.int64)
    for side, model in models.items():
        mask = side_arr == int(side)
        if bool(mask.any()):
            out[mask] = np.asarray(model.predict(x_all.loc[mask]), dtype=np.float64)
    out[side_arr == 0] = 0.0
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite side-split risk predictions")
    return out


def _quality_values(features: pd.DataFrame) -> np.ndarray:
    if "parent_quality_for_action" in features.columns:
        return pd.to_numeric(features["parent_quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    if "decision_quality_score" in features.columns:
        return pd.to_numeric(features["decision_quality_score"], errors="raise").to_numpy(dtype=np.float64)
    raise RuntimeError("quality score blend requested but quality feature is missing")


def _blend_scores(
    model_scores: np.ndarray,
    quality_scores: np.ndarray,
    *,
    train_model_q50: float,
    train_model_iqr: float,
    train_quality_q50: float,
    train_quality_iqr: float,
    blend: float,
) -> np.ndarray:
    b = float(blend)
    if b <= 0.0:
        return np.asarray(model_scores, dtype=np.float64)
    model_z = (np.asarray(model_scores, dtype=np.float64) - float(train_model_q50)) / max(float(train_model_iqr), 1.0e-8)
    quality_z = (np.asarray(quality_scores, dtype=np.float64) - float(train_quality_q50)) / max(float(train_quality_iqr), 1.0e-8)
    return (1.0 - b) * model_z + b * quality_z


def _ledger_metrics_with_margins(
    frame: pd.DataFrame,
    ledger: pd.DataFrame,
    margins: np.ndarray | None,
    leverage_override: np.ndarray | None = None,
    *,
    tail_budget: float = 0.02,
    tail_penalty: float = 1.0,
    liquidation_buffer: float = 0.12,
    liquidation_penalty: float = 0.25,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if leverage_override is not None and margins is None:
        raise RuntimeError("leverage_override requires margins")
    if ledger.empty:
        raise RuntimeError("empty ledger for margin replay")
    out = ledger.copy().reset_index(drop=True)
    if margins is None:
        new_margin = pd.to_numeric(out["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
        new_notional = pd.to_numeric(out["notional"], errors="raise").to_numpy(dtype=np.float64)
        new_leverage = pd.to_numeric(out["leverage"], errors="raise").to_numpy(dtype=np.float64)
    else:
        entry_idx = pd.to_numeric(out["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
        if leverage_override is None:
            new_leverage = pd.to_numeric(out["leverage"], errors="raise").to_numpy(dtype=np.float64)
        else:
            new_leverage = np.asarray(leverage_override, dtype=np.float64)[entry_idx]
        new_margin = np.asarray(margins, dtype=np.float64)[entry_idx]
        new_notional = new_margin * new_leverage
    net_per_notional = pd.to_numeric(out["net_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    mae = pd.to_numeric(out["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    log_growth_sum = 0.0
    tail_excess_sum = 0.0
    liquidation_excess_sum = 0.0
    log_risk_utility_sum = 0.0
    for j in range(len(out)):
        adverse_eq = cash * (1.0 + min(float(mae[j]), 0.0) * float(new_notional[j]))
        peak = max(peak, cash)
        mdd = min(mdd, adverse_eq / max(peak, 1.0e-12) - 1.0)
        before = cash
        account_return = float(net_per_notional[j]) * float(new_notional[j])
        log_growth = float(np.log1p(max(account_return, -0.999999)))
        tail_excess = max(-float(mae[j]) * float(new_notional[j]) - float(tail_budget), 0.0)
        liquidation_excess = max(-float(mae[j]) * float(new_leverage[j]) - float(liquidation_buffer), 0.0)
        log_risk_utility = log_growth - float(tail_penalty) * tail_excess - float(liquidation_penalty) * liquidation_excess
        log_growth_sum += log_growth
        tail_excess_sum += tail_excess
        liquidation_excess_sum += liquidation_excess
        log_risk_utility_sum += log_risk_utility
        cash = cash * (1.0 + account_return)
        wins += int(cash > before)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1.0e-12) - 1.0)
    out["risk_margin_fraction"] = new_margin
    out["risk_leverage"] = new_leverage
    out["risk_notional"] = new_notional
    out["risk_trade_return"] = net_per_notional * new_notional
    n = max(len(out), 1)
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(len(out)),
            "wr": float(wins / n),
            "trades_per_day": float(len(out) / _duration_days(frame)),
            "avg_notional": float(np.mean(new_notional)) if len(out) else 0.0,
            "avg_margin_fraction": float(np.mean(new_margin)) if len(out) else 0.0,
            "avg_leverage": float(np.mean(new_leverage)) if len(out) else 0.0,
            "log_growth_sum": float(log_growth_sum),
            "tail_excess_sum": float(tail_excess_sum),
            "liquidation_excess_sum": float(liquidation_excess_sum),
            "log_risk_utility": float(log_risk_utility_sum),
            "long_entries": int(np.sum(side > 0)),
            "short_entries": int(np.sum(side < 0)),
            "exit_reasons": {str(k): int(v) for k, v in out["reason"].value_counts().to_dict().items()},
        },
        out,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--precomputed-prediction-dir", type=Path, default=None)
    ap.add_argument("--precomputed-prediction-tag", default="q070")
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--regime3-current-2025", type=Path, default=omega.REGIME3_CURRENT_2025)
    ap.add_argument("--regime3-current-2026", type=Path, default=omega.REGIME3_CURRENT_2026)
    ap.add_argument("--quality-mode", choices=["same_as_direction"], default="same_as_direction")
    ap.add_argument("--train-csv", type=Path, default=None)
    ap.add_argument("--eval-csv", type=Path, default=None)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--out-suffix", default="v1")
    ap.add_argument("--reuse-ledgers", action="store_true")
    ap.add_argument("--ledger-source-dir", type=Path, default=None)
    ap.add_argument("--model-kind", choices=["hgb", "extra_trees", "random_forest", "gradient_boosting"], default="hgb")
    ap.add_argument("--risk-feature-mode", choices=["all", "parent_outputs"], default="all")
    ap.add_argument("--risk-extra-feature-cols", default="", help="comma-separated raw frame columns appended to the risk feature matrix regardless of --risk-feature-mode")
    ap.add_argument("--side-split-model", action="store_true")
    ap.add_argument("--score-quality-blend", type=float, default=0.0)
    ap.add_argument("--side-aware-grid", action="store_true")
    ap.add_argument("--dynamic-leverage", action="store_true")
    ap.add_argument("--require-dynamic-leverage-mapping", action="store_true")
    ap.add_argument("--compact-grid", action="store_true")
    ap.add_argument("--live-exposure-grid", action="store_true")
    ap.add_argument("--min-validation-avg-notional", type=float, default=0.0)
    ap.add_argument("--max-validation-avg-notional", type=float, default=0.0)
    ap.add_argument("--full-replay-top-k", type=int, default=1)
    ap.add_argument("--selection-objective", choices=["pnl", "log_risk"], default="pnl")
    ap.add_argument("--selection-scope", choices=["validation_oos_guard", "validation_only"], default="validation_oos_guard")
    ap.add_argument("--log-tail-budget", type=float, default=0.02)
    ap.add_argument("--log-tail-penalty", type=float, default=1.0)
    ap.add_argument("--log-liquidation-buffer", type=float, default=0.12)
    ap.add_argument("--log-liquidation-penalty", type=float, default=0.25)
    ap.add_argument("--max-validation-mdd-abs", type=float, default=8.0)
    ap.add_argument("--max-oos-mdd-abs", type=float, default=6.0)
    ap.add_argument("--target-mae-penalty", type=float, default=0.0)
    ap.add_argument("--risk-target-mode", choices=["net", "trend_alignment", "net_plus_trend_alignment"], default="net")
    ap.add_argument("--risk-trend-label-dir", type=Path, default=None)
    ap.add_argument("--risk-trend-alpha", type=float, default=0.02)
    ap.add_argument("--risk-trend-active-weight", type=float, default=3.0)
    ap.add_argument("--risk-context-feature-dir", type=Path, default=None)
    ap.add_argument("--notional-scaled-sltp", action="store_true")
    ap.add_argument("--exit-sizing-input-mode", choices=["actual", "baseline"], default="actual")
    ap.add_argument("--exit-context-feature-dir", type=Path, default=None)
    ap.add_argument("--exit-trend-threshold-scale", type=float, default=0.0)
    ap.add_argument("--exit-threshold-floor", type=float, default=0.55)
    ap.add_argument("--exit-threshold-cap", type=float, default=0.95)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--disable-exit-only-feature", action="store_true", help="ablation: match a parent bundle trained with EXIT_ONLY_COLS=[] (no swing_transition_prob at all)")
    args = ap.parse_args()

    if bool(args.disable_exit_only_feature):
        parent.EXIT_ONLY_COLS = []

    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    ledger_dir = Path(args.ledger_source_dir) if args.ledger_source_dir is not None else out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("stage=load_bundle", flush=True)
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)

    print("stage=prepare_frames", flush=True)
    if args.train_csv is not None:
        omega.TRAIN_CSV = Path(args.train_csv)
    if args.eval_csv is not None:
        omega.EVAL_CSV = Path(args.eval_csv)
    omega.REGIME3_CURRENT_2025 = Path(args.regime3_current_2025)
    omega.REGIME3_CURRENT_2026 = Path(args.regime3_current_2026)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()

    if args.precomputed_prediction_dir is None:
        print("stage=predict_parent", flush=True)
        x_train, train_src, train_dec_base = _predict_decisions(
            frames["train_raw"],
            oof=True,
            models=models,
            base_cols=base_cols,
            quality_threshold=float(args.quality_threshold),
            device=device,
        )
        x_val, val_src, val_dec_base = _predict_decisions(
            frames["val_raw"],
            oof=True,
            models=models,
            base_cols=base_cols,
            quality_threshold=float(args.quality_threshold),
            device=device,
        )
        x_oos, oos_src, oos_dec_base = _predict_decisions(
            frames["oos_raw"],
            oof=False,
            models=models,
            base_cols=base_cols,
            quality_threshold=float(args.quality_threshold),
            device=device,
        )
    else:
        print("stage=load_precomputed_parent_predictions", flush=True)
        pred_dir = Path(args.precomputed_prediction_dir)
        tag = str(args.precomputed_prediction_tag)
        train_src = _load_precomputed_prediction(pred_dir, "train", tag, frames["train_raw"])
        val_src = _load_precomputed_prediction(pred_dir, "validation", tag, frames["val_raw"])
        oos_src = _load_precomputed_prediction(pred_dir, "oos", tag, frames["oos_raw"])
        x_train = parent._base_input(frames["train_raw"], base_cols)
        x_val = parent._base_input(frames["val_raw"], base_cols)
        x_oos = parent._base_input(frames["oos_raw"], base_cols)
        train_dec_base = parent._to_decisions(train_src, oof=True)
        val_dec_base = parent._to_decisions(val_src, oof=True)
        oos_dec_base = parent._to_decisions(oos_src, oof=False)

    print("stage=apply_omega4_2_atr_contract", flush=True)
    train_dec, train_atr_diag = atr_eval._apply_atr_safety_sltp(
        train_dec_base,
        frames["train_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )
    val_dec, val_atr_diag = atr_eval._apply_atr_safety_sltp(
        val_dec_base,
        frames["val_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )
    oos_dec, oos_atr_diag = atr_eval._apply_atr_safety_sltp(
        oos_dec_base,
        frames["oos_raw"],
        atr_window=int(args.atr_window),
        tp_mult=float(args.tp_mult),
        sl_mult=float(args.sl_mult),
        min_tp=float(args.min_tp),
        min_sl=float(args.min_sl),
        max_tp=float(args.max_tp),
        max_sl=float(args.max_sl),
    )
    train_atr = atr_eval._atr_pct(frames["train_raw"], int(args.atr_window))
    val_atr = atr_eval._atr_pct(frames["val_raw"], int(args.atr_window))
    oos_atr = atr_eval._atr_pct(frames["oos_raw"], int(args.atr_window))

    print("stage=replay_baseline_ledgers", flush=True)
    train_ledger_path = ledger_dir / "train_baseline_trade_ledger.csv"
    val_ledger_path = ledger_dir / "validation_baseline_trade_ledger.csv"
    oos_ledger_path = ledger_dir / "oos_baseline_trade_ledger.csv"
    if bool(args.reuse_ledgers) and train_ledger_path.exists() and val_ledger_path.exists() and oos_ledger_path.exists():
        print("stage=reuse_baseline_ledgers", flush=True)
        train_ledger = pd.read_csv(train_ledger_path)
        val_base_ledger = pd.read_csv(val_ledger_path)
        oos_base_ledger = pd.read_csv(oos_ledger_path)
        train_base_m, _ = _ledger_metrics_with_margins(frames["train_raw"], train_ledger, None)
        val_base_m, _ = _ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, None)
        oos_base_m, _ = _ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, None)
    else:
        train_base_m, train_ledger = _replay_with_risk(
            frames["train_raw"], x_train, train_dec, loaded, risk_margin_fraction=None, risk_leverage=None, exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), notional_scaled_sltp=bool(args.notional_scaled_sltp), exit_sizing_input_mode=str(args.exit_sizing_input_mode), device=device
        )
        val_base_m, val_base_ledger = _replay_with_risk(
            frames["val_raw"], x_val, val_dec, loaded, risk_margin_fraction=None, risk_leverage=None, exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), notional_scaled_sltp=bool(args.notional_scaled_sltp), exit_sizing_input_mode=str(args.exit_sizing_input_mode), device=device
        )
        oos_base_m, oos_base_ledger = _replay_with_risk(
            frames["oos_raw"], x_oos, oos_dec, loaded, risk_margin_fraction=None, risk_leverage=None, exit_threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), notional_scaled_sltp=bool(args.notional_scaled_sltp), exit_sizing_input_mode=str(args.exit_sizing_input_mode), device=device
        )
    train_base_ledger_m, train_base_ledger_sized = _ledger_metrics_with_margins(frames["train_raw"], train_ledger, None)
    val_base_ledger_m, val_base_ledger_sized = _ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, None)
    oos_base_ledger_m, oos_base_ledger_sized = _ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, None)
    if train_ledger.empty:
        raise RuntimeError("empty train ledger for risk sidecar")

    print("stage=build_risk_features", flush=True)
    train_features = _risk_feature_frame(frames["train_raw"], train_src, train_dec, base_cols, atr_pct=train_atr, feature_mode=str(args.risk_feature_mode))
    val_features = _risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode=str(args.risk_feature_mode))
    oos_features = _risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode=str(args.risk_feature_mode))
    extra_feature_cols = [c.strip() for c in str(args.risk_extra_feature_cols).split(",") if c.strip()]
    if extra_feature_cols:
        for name, features, raw in (
            ("train", train_features, frames["train_raw"]),
            ("val", val_features, frames["val_raw"]),
            ("oos", oos_features, frames["oos_raw"]),
        ):
            missing = [c for c in extra_feature_cols if c not in raw.columns]
            if missing:
                raise RuntimeError(f"{name}: risk extra feature cols missing from raw frame: {missing}")
        for split_name, features, raw in (
            ("train", train_features, frames["train_raw"]),
            ("val", val_features, frames["val_raw"]),
            ("oos", oos_features, frames["oos_raw"]),
        ):
            extra = raw[extra_feature_cols].reset_index(drop=True).apply(pd.to_numeric, errors="raise").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
            if split_name == "train":
                train_features = pd.concat([train_features.reset_index(drop=True), extra], axis=1)
            elif split_name == "val":
                val_features = pd.concat([val_features.reset_index(drop=True), extra], axis=1)
            else:
                oos_features = pd.concat([oos_features.reset_index(drop=True), extra], axis=1)
    risk_context_feature_cols: list[str] = []
    if args.risk_context_feature_dir is not None:
        print("stage=append_risk_context_features", flush=True)
        train_context = _read_context_features(Path(args.risk_context_feature_dir), "train", frames["train_raw"])
        val_context = _read_context_features(Path(args.risk_context_feature_dir), "validation", frames["val_raw"])
        oos_context = _read_context_features(Path(args.risk_context_feature_dir), "oos", frames["oos_raw"])
        risk_context_feature_cols = list(train_context.columns)
        if list(val_context.columns) != risk_context_feature_cols or list(oos_context.columns) != risk_context_feature_cols:
            raise RuntimeError("risk context feature column contract mismatch across splits")
        train_features = _append_context_features(train_features, train_context, split="train")
        val_features = _append_context_features(val_features, val_context, split="validation")
        oos_features = _append_context_features(oos_features, oos_context, split="oos")
    exit_val_context = None
    exit_oos_context = None
    if args.exit_context_feature_dir is not None:
        print("stage=load_exit_context_features", flush=True)
        exit_val_context = _read_context_features(Path(args.exit_context_feature_dir), "validation", frames["val_raw"])
        exit_oos_context = _read_context_features(Path(args.exit_context_feature_dir), "oos", frames["oos_raw"])
    x_train_trade, risk_cols = _feature_matrix(train_features.iloc[train_ledger["entry_signal_i"].to_numpy(dtype=np.int64)].reset_index(drop=True))
    net_target = pd.to_numeric(train_ledger["net_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    mae_target = pd.to_numeric(train_ledger["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)
    sample_weight = 1.0 + np.clip(-mae_target * 25.0, 0.0, 3.0)
    side_train_trade = pd.to_numeric(train_ledger["side"], errors="raise").to_numpy(dtype=np.int64)
    risk_trend_diag: dict[str, Any] = {"mode": "none"}
    if str(args.risk_target_mode) != "net":
        if args.risk_trend_label_dir is None:
            raise RuntimeError("--risk-trend-label-dir is required when --risk-target-mode is not net")
        train_trend_action = _align_risk_trend_actions(frames["train_raw"], Path(args.risk_trend_label_dir), 2025, "risk train trend labels")
        val_trend_action = _align_risk_trend_actions(frames["val_raw"], Path(args.risk_trend_label_dir), 2025, "risk validation trend labels")
        oos_trend_action = _align_risk_trend_actions(frames["oos_raw"], Path(args.risk_trend_label_dir), 2026, "risk oos trend labels")
        trend_alignment, trend_trade_diag = _risk_target_from_trend_alignment(train_ledger, train_trend_action)
        trend_active = trend_alignment != 0.0
        if str(args.risk_target_mode) == "trend_alignment":
            y_train_trade = trend_alignment
        elif str(args.risk_target_mode) == "net_plus_trend_alignment":
            y_train_trade = net_target + float(args.target_mae_penalty) * mae_target + float(args.risk_trend_alpha) * trend_alignment
        else:
            raise RuntimeError(f"unknown risk target mode: {args.risk_target_mode}")
        sample_weight = sample_weight * (1.0 + float(args.risk_trend_active_weight) * trend_active.astype(np.float64))
        risk_trend_diag = {
            "mode": str(args.risk_target_mode),
            "label_dir": str(args.risk_trend_label_dir),
            "alpha": float(args.risk_trend_alpha),
            "active_weight": float(args.risk_trend_active_weight),
            "train_label_summary": _trend_action_summary(train_trend_action),
            "validation_label_summary": _trend_action_summary(val_trend_action),
            "oos_label_summary": _trend_action_summary(oos_trend_action),
            "train_trade_alignment": trend_trade_diag,
        }
    else:
        y_train_trade = net_target + float(args.target_mae_penalty) * mae_target

    print("stage=train_risk_sidecar", flush=True)
    x_train_all, _ = _feature_matrix(train_features, risk_cols)
    x_val_all, _ = _feature_matrix(val_features, risk_cols)
    x_oos_all, _ = _feature_matrix(oos_features, risk_cols)
    if bool(args.side_split_model):
        risk_model = _fit_side_split_models(
            str(args.model_kind),
            x_train_trade,
            y_train_trade,
            side_train_trade,
            sample_weight,
            seed=int(args.seed),
        )
        train_side_all = pd.to_numeric(train_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        val_side_all = pd.to_numeric(val_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        oos_side_all = pd.to_numeric(oos_dec["side"], errors="raise").to_numpy(dtype=np.int64)
        train_model_score = _predict_side_split_models(risk_model, x_train_all, train_side_all)
        val_model_score = _predict_side_split_models(risk_model, x_val_all, val_side_all)
        oos_model_score = _predict_side_split_models(risk_model, x_oos_all, oos_side_all)
        train_trade_model_score = _predict_side_split_models(risk_model, x_train_trade, side_train_trade)
    else:
        risk_model = _build_risk_model(str(args.model_kind), int(args.seed))
        risk_model.fit(x_train_trade, y_train_trade, sample_weight=sample_weight)
        train_model_score = np.asarray(risk_model.predict(x_train_all), dtype=np.float64)
        val_model_score = np.asarray(risk_model.predict(x_val_all), dtype=np.float64)
        oos_model_score = np.asarray(risk_model.predict(x_oos_all), dtype=np.float64)
        train_trade_model_score = np.asarray(risk_model.predict(x_train_trade), dtype=np.float64)
    if float(args.score_quality_blend) > 0.0:
        train_quality = _quality_values(train_features)
        val_quality = _quality_values(val_features)
        oos_quality = _quality_values(oos_features)
        train_trade_quality = train_quality[train_ledger["entry_signal_i"].to_numpy(dtype=np.int64)]
        train_model_q50 = float(np.quantile(train_trade_model_score, 0.50))
        train_model_iqr = float(np.quantile(train_trade_model_score, 0.75) - np.quantile(train_trade_model_score, 0.25))
        train_quality_q50 = float(np.quantile(train_trade_quality, 0.50))
        train_quality_iqr = float(np.quantile(train_trade_quality, 0.75) - np.quantile(train_trade_quality, 0.25))
        train_score = _blend_scores(
            train_model_score,
            train_quality,
            train_model_q50=train_model_q50,
            train_model_iqr=train_model_iqr,
            train_quality_q50=train_quality_q50,
            train_quality_iqr=train_quality_iqr,
            blend=float(args.score_quality_blend),
        )
        val_score = _blend_scores(
            val_model_score,
            val_quality,
            train_model_q50=train_model_q50,
            train_model_iqr=train_model_iqr,
            train_quality_q50=train_quality_q50,
            train_quality_iqr=train_quality_iqr,
            blend=float(args.score_quality_blend),
        )
        oos_score = _blend_scores(
            oos_model_score,
            oos_quality,
            train_model_q50=train_model_q50,
            train_model_iqr=train_model_iqr,
            train_quality_q50=train_quality_q50,
            train_quality_iqr=train_quality_iqr,
            blend=float(args.score_quality_blend),
        )
        train_trade_score = train_score[train_ledger["entry_signal_i"].to_numpy(dtype=np.int64)]
    else:
        train_score = train_model_score
        val_score = val_model_score
        oos_score = oos_model_score
        train_trade_score = train_trade_model_score
    train_q50 = float(np.quantile(train_trade_score, 0.50))
    train_iqr = float(np.quantile(train_trade_score, 0.75) - np.quantile(train_trade_score, 0.25))

    print("stage=grid_risk_mapping", flush=True)
    candidates: list[dict[str, float]] = []
    leverage_specs: list[dict[str, float]] = [{}]
    if bool(args.dynamic_leverage):
        leverage_specs = [
            {
                "leverage_min": 1.0,
                "leverage_max": 2.0,
                "leverage_temp": 1.0,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 1.0,
                "short_leverage_scale": 1.0,
            },
            {
                "leverage_min": 1.0,
                "leverage_max": 2.5,
                "leverage_temp": 1.35,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 0.90,
                "short_leverage_scale": 1.10,
            },
            {
                "leverage_min": 1.0,
                "leverage_max": 3.0,
                "leverage_temp": 1.35,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 0.85,
                "short_leverage_scale": 1.15,
            },
            {
                "leverage_min": 1.25,
                "leverage_max": 3.0,
                "leverage_temp": 1.70,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 0.75,
                "short_leverage_scale": 1.25,
            },
            {
                "leverage_min": 1.50,
                "leverage_max": 3.0,
                "leverage_temp": 1.70,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 0.75,
                "short_leverage_scale": 1.25,
            },
        ]
    if bool(args.live_exposure_grid):
        leverage_specs = [
            {
                "leverage_min": 2.0,
                "leverage_max": 2.0,
                "leverage_temp": 1.0,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 1.0,
                "short_leverage_scale": 1.0,
            },
            {
                "leverage_min": 1.75,
                "leverage_max": 2.25,
                "leverage_temp": 1.0,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 1.0,
                "short_leverage_scale": 1.0,
            },
            {
                "leverage_min": 1.75,
                "leverage_max": 2.50,
                "leverage_temp": 1.35,
                "leverage_floor": 1.0,
                "leverage_cap": 3.0,
                "long_leverage_scale": 0.95,
                "short_leverage_scale": 1.05,
            },
        ]
        min_scale_values = (1.0, 1.25, 1.50, 1.75)
        max_scale_values = (1.75, 2.00, 2.25, 2.50)
        temp_values = (0.70, 1.00, 1.35, 1.70)
        floor_values = (0.18, 0.22, 0.26, 0.30)
        cap_values = (0.36, 0.40, 0.45)
    elif bool(args.compact_grid):
        min_scale_values = (0.75, 0.85, 0.95)
        max_scale_values = (1.45, 1.55, 1.65)
        temp_values = (1.35, 1.70, 2.10)
        floor_values = (0.06, 0.08)
        cap_values = (0.28, 0.32)
    else:
        min_scale_values = (0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95)
        max_scale_values = (1.25, 1.35, 1.45, 1.55, 1.65, 1.75)
        temp_values = (0.7, 1.0, 1.35, 1.7, 2.1)
        floor_values = (0.06, 0.08, 0.10, 0.12)
        cap_values = (0.22, 0.24, 0.26, 0.28, 0.32)
    for min_scale in min_scale_values:
        for max_scale in max_scale_values:
            if max_scale <= min_scale:
                continue
            for temp in temp_values:
                for floor in floor_values:
                    for cap in cap_values:
                        if cap <= floor:
                            continue
                        side_pairs = [(1.0, 1.0)]
                        if bool(args.side_aware_grid):
                            if bool(args.live_exposure_grid):
                                side_pairs = [
                                    (0.75, 1.0),
                                    (0.75, 1.15),
                                    (0.75, 1.25),
                                    (0.85, 1.25),
                                    (1.0, 1.0),
                                    (1.0, 1.25),
                                ]
                            else:
                                side_pairs = [
                                    (0.55, 1.0),
                                    (0.65, 1.0),
                                    (0.75, 1.0),
                                    (0.85, 1.0),
                                    (0.75, 1.10),
                                    (0.65, 1.15),
                                    (0.55, 1.25),
                                    (0.65, 1.25),
                                    (0.75, 1.25),
                                    (0.55, 1.35),
                                ]
                        for long_scale, short_scale in side_pairs:
                            margin_cfg = {
                                    "min_scale": float(min_scale),
                                    "max_scale": float(max_scale),
                                    "temp": float(temp),
                                    "floor": float(floor),
                                    "cap": float(cap),
                                    "long_scale": float(long_scale),
                                    "short_scale": float(short_scale),
                            }
                            for leverage_cfg in leverage_specs:
                                candidates.append({**margin_cfg, **leverage_cfg})

    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    log_risk_kwargs = {
        "tail_budget": float(args.log_tail_budget),
        "tail_penalty": float(args.log_tail_penalty),
        "liquidation_buffer": float(args.log_liquidation_buffer),
        "liquidation_penalty": float(args.log_liquidation_penalty),
    }
    for idx, cfg in enumerate(candidates):
        margin_cfg = {k: float(cfg[k]) for k in MARGIN_CFG_KEYS}
        leverage_cfg = {k: float(cfg[k]) for k in LEVERAGE_CFG_KEYS if k in cfg}
        val_margin = _risk_margins(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **margin_cfg)
        oos_margin = _risk_margins(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **margin_cfg)
        val_leverage = _risk_leverage(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **leverage_cfg) if leverage_cfg else None
        oos_leverage = _risk_leverage(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **leverage_cfg) if leverage_cfg else None
        val_m, _ = _ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, val_margin, val_leverage, **log_risk_kwargs)
        oos_m, _ = _ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, oos_margin, oos_leverage, **log_risk_kwargs)
        name = f"risk_{idx:03d}"
        row = {
            "variant": name,
            **cfg,
            "validation_pnl": float(val_m["pnl"]),
            "validation_mdd": float(val_m["mdd"]),
            "validation_trades": int(val_m["trades"]),
            "validation_wr": float(val_m["wr"]),
            "validation_avg_notional": float(val_m["avg_notional"]),
            "validation_avg_margin": float(val_m["avg_margin_fraction"]),
            "validation_avg_leverage": float(val_m["avg_leverage"]),
            "validation_log_growth_sum": float(val_m["log_growth_sum"]),
            "validation_tail_excess_sum": float(val_m["tail_excess_sum"]),
            "validation_liquidation_excess_sum": float(val_m["liquidation_excess_sum"]),
            "validation_log_risk_utility": float(val_m["log_risk_utility"]),
            "oos_pnl": float(oos_m["pnl"]),
            "oos_mdd": float(oos_m["mdd"]),
            "oos_trades": int(oos_m["trades"]),
            "oos_wr": float(oos_m["wr"]),
            "oos_avg_notional": float(oos_m["avg_notional"]),
            "oos_avg_margin": float(oos_m["avg_margin_fraction"]),
            "oos_avg_leverage": float(oos_m["avg_leverage"]),
            "oos_log_growth_sum": float(oos_m["log_growth_sum"]),
            "oos_tail_excess_sum": float(oos_m["tail_excess_sum"]),
            "oos_liquidation_excess_sum": float(oos_m["liquidation_excess_sum"]),
            "oos_log_risk_utility": float(oos_m["log_risk_utility"]),
        }
        rows.append(row)
        results[name] = {"config": cfg, "validation": val_m, "oos": oos_m}

    min_trade_ratio = 0.95
    val_trade_floor = int(np.floor(int(val_base_ledger_m["trades"]) * min_trade_ratio))
    validation_mdd_floor = -abs(float(args.max_validation_mdd_abs))
    oos_mdd_floor = -abs(float(args.max_oos_mdd_abs))
    def exposure_ok(row: dict[str, Any]) -> bool:
        avg_notional = float(row["validation_avg_notional"])
        if float(args.min_validation_avg_notional) > 0.0 and avg_notional < float(args.min_validation_avg_notional):
            return False
        if float(args.max_validation_avg_notional) > 0.0 and avg_notional > float(args.max_validation_avg_notional):
            return False
        if bool(args.require_dynamic_leverage_mapping):
            leverage_span = float(row.get("leverage_max", 0.0)) - float(row.get("leverage_min", 0.0))
            long_scale = float(row.get("long_leverage_scale", 1.0))
            short_scale = float(row.get("short_leverage_scale", 1.0))
            if leverage_span <= 0.0 and long_scale == 1.0 and short_scale == 1.0:
                return False
        return True

    if str(args.selection_scope) == "validation_only":
        eligible = [r for r in rows if int(r["validation_trades"]) >= val_trade_floor and float(r["validation_mdd"]) >= validation_mdd_floor]
    else:
        eligible = [
            r
            for r in rows
            if int(r["validation_trades"]) >= val_trade_floor
            and float(r["validation_mdd"]) >= validation_mdd_floor
            and float(r["oos_mdd"]) >= oos_mdd_floor
        ]
        if not eligible:
            eligible = [r for r in rows if int(r["validation_trades"]) >= val_trade_floor and float(r["validation_mdd"]) >= validation_mdd_floor]
    eligible = [r for r in eligible if exposure_ok(r)]
    if not eligible and (float(args.min_validation_avg_notional) > 0.0 or float(args.max_validation_avg_notional) > 0.0):
        raise RuntimeError(
            "no eligible risk mapping after validation average notional constraint: "
            f"min={float(args.min_validation_avg_notional):.4f}, max={float(args.max_validation_avg_notional):.4f}"
        )
    if not eligible:
        eligible = [r for r in rows if int(r["validation_trades"]) >= val_trade_floor]
    if str(args.selection_objective) == "log_risk" and str(args.selection_scope) == "validation_only":
        selected_key = lambda r: (float(r["validation_log_risk_utility"]), float(r["validation_mdd"]), float(r["validation_pnl"]))
    elif str(args.selection_objective) == "log_risk":
        selected_key = lambda r: (float(r["validation_log_risk_utility"]), float(r["validation_mdd"]), float(r["oos_log_risk_utility"]))
    elif str(args.selection_scope) == "validation_only":
        selected_key = lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"]), float(r["validation_log_risk_utility"]))
    else:
        selected_key = lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"]), float(r["oos_pnl"]))
    selected = max(eligible, key=selected_key)
    full_replay_selection: list[dict[str, Any]] = []
    replay_pool = sorted(eligible, key=selected_key, reverse=True)[: max(1, int(args.full_replay_top_k))]
    for cand in replay_pool:
        cand_margin_cfg = {k: float(cand[k]) for k in MARGIN_CFG_KEYS}
        cand_leverage_cfg = {k: float(cand[k]) for k in LEVERAGE_CFG_KEYS if k in cand}
        cand_val_margin = _risk_margins(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **cand_margin_cfg)
        cand_oos_margin = _risk_margins(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **cand_margin_cfg)
        cand_val_leverage = (
            _risk_leverage(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **cand_leverage_cfg) if cand_leverage_cfg else None
        )
        cand_oos_leverage = (
            _risk_leverage(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **cand_leverage_cfg) if cand_leverage_cfg else None
        )
        cand_val_replay_m, cand_val_replay_ledger = _replay_with_risk(
            frames["val_raw"],
            x_val,
            val_dec,
            loaded,
            risk_margin_fraction=cand_val_margin,
            risk_leverage=cand_val_leverage,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            notional_scaled_sltp=bool(args.notional_scaled_sltp),
            exit_sizing_input_mode=str(args.exit_sizing_input_mode),
            exit_context_features=exit_val_context,
            exit_trend_threshold_scale=float(args.exit_trend_threshold_scale),
            exit_threshold_floor=float(args.exit_threshold_floor),
            exit_threshold_cap=float(args.exit_threshold_cap),
            device=device,
        )
        cand_oos_replay_m, cand_oos_replay_ledger = _replay_with_risk(
            frames["oos_raw"],
            x_oos,
            oos_dec,
            loaded,
            risk_margin_fraction=cand_oos_margin,
            risk_leverage=cand_oos_leverage,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            notional_scaled_sltp=bool(args.notional_scaled_sltp),
            exit_sizing_input_mode=str(args.exit_sizing_input_mode),
            exit_context_features=exit_oos_context,
            exit_trend_threshold_scale=float(args.exit_trend_threshold_scale),
            exit_threshold_floor=float(args.exit_threshold_floor),
            exit_threshold_cap=float(args.exit_threshold_cap),
            device=device,
        )
        cand_val_log_m, _ = _ledger_metrics_with_margins(frames["val_raw"], cand_val_replay_ledger, None, **log_risk_kwargs)
        cand_oos_log_m, _ = _ledger_metrics_with_margins(frames["oos_raw"], cand_oos_replay_ledger, None, **log_risk_kwargs)
        for key in ("log_growth_sum", "tail_excess_sum", "liquidation_excess_sum", "log_risk_utility"):
            cand_val_replay_m[key] = cand_val_log_m[key]
            cand_oos_replay_m[key] = cand_oos_log_m[key]
        full_replay_selection.append(
            {
                "variant": str(cand["variant"]),
                "mapping": {**cand_margin_cfg, **cand_leverage_cfg},
                "ledger_validation": {k: cand[k] for k in cand if str(k).startswith("validation_")},
                "ledger_oos": {k: cand[k] for k in cand if str(k).startswith("oos_")},
                "validation": cand_val_replay_m,
                "oos": cand_oos_replay_m,
            }
        )
    if str(args.selection_scope) == "validation_only":
        full_eligible = [
            r
            for r in full_replay_selection
            if int(r["validation"]["trades"]) >= val_trade_floor and float(r["validation"]["mdd"]) >= validation_mdd_floor
        ]
    else:
        full_eligible = [
            r
            for r in full_replay_selection
            if int(r["validation"]["trades"]) >= val_trade_floor
            and float(r["validation"]["mdd"]) >= validation_mdd_floor
            and float(r["oos"]["mdd"]) >= oos_mdd_floor
        ]
    if full_eligible:
        if str(args.selection_objective) == "log_risk" and str(args.selection_scope) == "validation_only":
            selected_full = max(full_eligible, key=lambda r: (float(r["validation"]["log_risk_utility"]), float(r["validation"]["mdd"]), float(r["validation"]["pnl"])))
        elif str(args.selection_objective) == "log_risk":
            selected_full = max(full_eligible, key=lambda r: (float(r["validation"]["log_risk_utility"]), float(r["validation"]["mdd"]), float(r["oos"]["log_risk_utility"])))
        elif str(args.selection_scope) == "validation_only":
            selected_full = max(full_eligible, key=lambda r: (float(r["validation"]["pnl"]), float(r["validation"]["mdd"]), float(r["validation"]["log_risk_utility"])))
        else:
            selected_full = max(full_eligible, key=lambda r: (float(r["validation"]["pnl"]), float(r["validation"]["mdd"]), float(r["oos"]["pnl"])))
        selected = next(r for r in rows if str(r["variant"]) == str(selected_full["variant"]))
    selected_margin_cfg = {k: float(selected[k]) for k in MARGIN_CFG_KEYS}
    selected_leverage_cfg = {k: float(selected[k]) for k in LEVERAGE_CFG_KEYS if k in selected}
    selected_cfg = {**selected_margin_cfg, **selected_leverage_cfg}
    selected_val_margin = _risk_margins(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **selected_margin_cfg)
    selected_oos_margin = _risk_margins(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **selected_margin_cfg)
    selected_val_leverage = (
        _risk_leverage(val_dec, val_score, train_q50=train_q50, train_iqr=train_iqr, **selected_leverage_cfg) if selected_leverage_cfg else None
    )
    selected_oos_leverage = (
        _risk_leverage(oos_dec, oos_score, train_q50=train_q50, train_iqr=train_iqr, **selected_leverage_cfg) if selected_leverage_cfg else None
    )
    selected_val_m, selected_val_ledger = _ledger_metrics_with_margins(frames["val_raw"], val_base_ledger, selected_val_margin, selected_val_leverage, **log_risk_kwargs)
    selected_oos_m, selected_oos_ledger = _ledger_metrics_with_margins(frames["oos_raw"], oos_base_ledger, selected_oos_margin, selected_oos_leverage, **log_risk_kwargs)
    selected_val_replay_m, selected_val_replay_ledger = _replay_with_risk(
        frames["val_raw"],
        x_val,
        val_dec,
        loaded,
        risk_margin_fraction=selected_val_margin,
        risk_leverage=selected_val_leverage,
        exit_threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        notional_scaled_sltp=bool(args.notional_scaled_sltp),
        exit_sizing_input_mode=str(args.exit_sizing_input_mode),
        exit_context_features=exit_val_context,
        exit_trend_threshold_scale=float(args.exit_trend_threshold_scale),
        exit_threshold_floor=float(args.exit_threshold_floor),
        exit_threshold_cap=float(args.exit_threshold_cap),
        device=device,
    )
    selected_oos_replay_m, selected_oos_replay_ledger = _replay_with_risk(
        frames["oos_raw"],
        x_oos,
        oos_dec,
        loaded,
        risk_margin_fraction=selected_oos_margin,
        risk_leverage=selected_oos_leverage,
        exit_threshold=float(args.exit_threshold),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        notional_scaled_sltp=bool(args.notional_scaled_sltp),
        exit_sizing_input_mode=str(args.exit_sizing_input_mode),
        exit_context_features=exit_oos_context,
        exit_trend_threshold_scale=float(args.exit_trend_threshold_scale),
        exit_threshold_floor=float(args.exit_threshold_floor),
        exit_threshold_cap=float(args.exit_threshold_cap),
        device=device,
    )
    selected_val_replay_log_m, _ = _ledger_metrics_with_margins(frames["val_raw"], selected_val_replay_ledger, None, **log_risk_kwargs)
    selected_oos_replay_log_m, _ = _ledger_metrics_with_margins(frames["oos_raw"], selected_oos_replay_ledger, None, **log_risk_kwargs)
    for key in ("log_growth_sum", "tail_excess_sum", "liquidation_excess_sum", "log_risk_utility"):
        selected_val_replay_m[key] = selected_val_replay_log_m[key]
        selected_oos_replay_m[key] = selected_oos_replay_log_m[key]
    if (
        int(selected_val_replay_m["trades"]) < val_trade_floor
        or float(selected_val_replay_m["mdd"]) < validation_mdd_floor
    ):
        raise RuntimeError(
            "selected risk mapping failed final validation replay constraint: "
            f"trades={int(selected_val_replay_m['trades'])} floor={val_trade_floor}, "
            f"mdd={float(selected_val_replay_m['mdd']):.4f} floor={validation_mdd_floor:.4f}"
        )

    if str(args.selection_objective) == "log_risk" and str(args.selection_scope) == "validation_only":
        ranking = pd.DataFrame(rows).sort_values(["validation_log_risk_utility", "validation_mdd", "validation_pnl"], ascending=[False, False, False])
    elif str(args.selection_objective) == "log_risk":
        ranking = pd.DataFrame(rows).sort_values(["validation_log_risk_utility", "validation_mdd", "oos_log_risk_utility"], ascending=[False, False, False])
    elif str(args.selection_scope) == "validation_only":
        ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "validation_mdd", "validation_log_risk_utility"], ascending=[False, False, False])
    else:
        ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "validation_mdd", "oos_pnl"], ascending=[False, False, False])
    ranking.to_csv(out_dir / "risk_mapping_ranking.csv", index=False)
    train_base_ledger_sized.to_csv(out_dir / "train_baseline_trade_ledger.csv", index=False)
    val_base_ledger_sized.to_csv(out_dir / "validation_baseline_trade_ledger.csv", index=False)
    oos_base_ledger_sized.to_csv(out_dir / "oos_baseline_trade_ledger.csv", index=False)
    selected_val_ledger.to_csv(out_dir / "validation_selected_risk_trade_ledger.csv", index=False)
    selected_oos_ledger.to_csv(out_dir / "oos_selected_risk_trade_ledger.csv", index=False)
    selected_val_replay_ledger.to_csv(out_dir / "validation_selected_risk_replayed_trade_ledger.csv", index=False)
    selected_oos_replay_ledger.to_csv(out_dir / "oos_selected_risk_replayed_trade_ledger.csv", index=False)
    with (out_dir / "risk_sidecar.pkl").open("wb") as f:
        pickle.dump(
            {
                "model": risk_model,
                "feature_columns": risk_cols,
                "train_score_q50": train_q50,
                "train_score_iqr": train_iqr,
                "selected_mapping": selected_cfg,
                "model_kind": str(args.model_kind),
                "risk_feature_mode": str(args.risk_feature_mode),
                "side_split_model": bool(args.side_split_model),
                "score_quality_blend": float(args.score_quality_blend),
                "dynamic_leverage": bool(args.dynamic_leverage),
                "selection_objective": str(args.selection_objective),
                "selection_scope": str(args.selection_scope),
                "log_risk_params": log_risk_kwargs,
                "target_mae_penalty": float(args.target_mae_penalty),
                "risk_target_mode": str(args.risk_target_mode),
                "risk_trend_label_dir": str(args.risk_trend_label_dir) if args.risk_trend_label_dir is not None else None,
                "risk_trend_diag": risk_trend_diag,
                "risk_context_feature_dir": str(args.risk_context_feature_dir) if args.risk_context_feature_dir is not None else None,
                "risk_context_feature_cols": risk_context_feature_cols,
                "precomputed_prediction_dir": str(args.precomputed_prediction_dir) if args.precomputed_prediction_dir is not None else None,
                "precomputed_prediction_tag": str(args.precomputed_prediction_tag) if args.precomputed_prediction_dir is not None else None,
                "notional_scaled_sltp": bool(args.notional_scaled_sltp),
                "exit_sizing_input_mode": str(args.exit_sizing_input_mode),
                "exit_context_feature_dir": str(args.exit_context_feature_dir) if args.exit_context_feature_dir is not None else None,
                "exit_trend_threshold_scale": float(args.exit_trend_threshold_scale),
                "exit_threshold_floor": float(args.exit_threshold_floor),
                "exit_threshold_cap": float(args.exit_threshold_cap),
                "contract": (
                    "Experimental notional-scaled SLTP: entry price-move barriers are multiplied by entry notional; sidecar scales margin_fraction and optionally leverage; notional=margin_fraction*leverage."
                    if bool(args.notional_scaled_sltp)
                    else "Omega 4.2 parent direction/quality/exit and ATR SLTP are unchanged; sidecar scales margin_fraction and optionally leverage; notional=margin_fraction*leverage; SLTP remains raw price-move barriers."
                ),
            },
            f,
        )

    report = {
        "model_id": MODEL_ID,
        "base_model": "omega4_2_atr192_tp12_sl6_floor_tp075_sl040_exit070_20260622",
        "baseline_bundle": str(args.baseline_bundle),
        "design": "Separate trade-level risk sidecar. Omega 4.2 direction, quality, exit head, and ATR price-move SLTP are unchanged. The sidecar predicts trade net_per_notional from entry-time features and maps the score to margin_fraction and, when enabled, leverage.",
        "risk_model": {
            "model_kind": str(args.model_kind),
            "risk_feature_mode": str(args.risk_feature_mode),
            "risk_extra_feature_cols": extra_feature_cols,
            "side_split_model": bool(args.side_split_model),
            "score_quality_blend": float(args.score_quality_blend),
            "dynamic_leverage": bool(args.dynamic_leverage),
            "require_dynamic_leverage_mapping": bool(args.require_dynamic_leverage_mapping),
            "selection_objective": str(args.selection_objective),
            "selection_scope": str(args.selection_scope),
            "log_risk_params": log_risk_kwargs,
            "notional_scaled_sltp": bool(args.notional_scaled_sltp),
            "live_exposure_grid": bool(args.live_exposure_grid),
            "min_validation_avg_notional": float(args.min_validation_avg_notional),
            "max_validation_avg_notional": float(args.max_validation_avg_notional),
            "train_csv": str(omega.TRAIN_CSV),
            "eval_csv": str(omega.EVAL_CSV),
            "direction_label_dir": str(args.direction_label_dir),
            "regime3_current_2025": str(args.regime3_current_2025),
            "regime3_current_2026": str(args.regime3_current_2026),
            "quality_mode": str(args.quality_mode),
            "risk_target_mode": str(args.risk_target_mode),
            "risk_trend_label_dir": str(args.risk_trend_label_dir) if args.risk_trend_label_dir is not None else None,
            "risk_context_feature_dir": str(args.risk_context_feature_dir) if args.risk_context_feature_dir is not None else None,
            "risk_context_feature_cols": risk_context_feature_cols,
            "precomputed_prediction_dir": str(args.precomputed_prediction_dir) if args.precomputed_prediction_dir is not None else None,
            "precomputed_prediction_tag": str(args.precomputed_prediction_tag) if args.precomputed_prediction_dir is not None else None,
            "exit_sizing_input_mode": str(args.exit_sizing_input_mode),
            "exit_context_feature_dir": str(args.exit_context_feature_dir) if args.exit_context_feature_dir is not None else None,
            "exit_trend_threshold_scale": float(args.exit_trend_threshold_scale),
            "exit_threshold_floor": float(args.exit_threshold_floor),
            "exit_threshold_cap": float(args.exit_threshold_cap),
        },
        "contract": {
            "quality_threshold": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
            "atr_window": int(args.atr_window),
            "take_profit_atr_multiple": float(args.tp_mult),
            "stop_loss_atr_multiple": float(args.sl_mult),
            "floor_take_profit_price_move": float(args.min_tp),
            "floor_stop_loss_price_move": float(args.min_sl),
            "cap_take_profit_price_move": float(args.max_tp),
            "cap_stop_loss_price_move": float(args.max_sl),
            "risk_sizing": "notional = margin_fraction * leverage",
            "sltp": (
                "experimental: raw directional price_move compared to entry price-move barriers multiplied by entry notional"
                if bool(args.notional_scaled_sltp)
                else "raw directional price_move compared to TP/SL price-move barriers; margin/notional do not change barrier location"
            ),
            "notional_scaled_sltp": bool(args.notional_scaled_sltp),
            "exit_sizing_input_mode": str(args.exit_sizing_input_mode),
            "exit_sizing_input": (
                "exit head receives baseline parent pos_notional/pos_leverage/pos_exposure; PnL still uses sidecar notional=margin_fraction*leverage"
                if str(args.exit_sizing_input_mode) == "baseline"
                else "exit head receives actual replay pos_notional/pos_leverage/pos_exposure from sidecar sizing"
            ),
            "exit_trend_overlay": (
                "exit threshold is adjusted by position_side * trend_ctx_long_minus_short from exit_context_feature_dir"
                if args.exit_context_feature_dir is not None and float(args.exit_trend_threshold_scale) != 0.0
                else "disabled"
            ),
            "parent_prediction_source": (
                "precomputed_prediction_artifacts"
                if args.precomputed_prediction_dir is not None
                else "current_runtime_forward_pass"
            ),
            "parent_prediction_promotion_requirement": (
                "train/validation/oos prediction CSVs must exist in precomputed_prediction_dir with the exact precomputed_prediction_tag"
                if args.precomputed_prediction_dir is not None
                else "not promotable without a separate artifact-integrity audit proving current forward pass matches saved prediction artifacts"
            ),
        },
        "train_baseline_replay": train_base_m,
        "omega4_2_replayed_baseline": {"validation": val_base_m, "oos": oos_base_m},
        "ledger_sizing_baseline": {"train": train_base_ledger_m, "validation": val_base_ledger_m, "oos": oos_base_ledger_m},
        "atr_diag": {"train": train_atr_diag, "validation": val_atr_diag, "oos": oos_atr_diag},
        "risk_label": {
            "rows": int(len(train_ledger)),
            "target": (
                "trade trend_alignment from risk_trend_label_dir at entry_signal_i"
                if str(args.risk_target_mode) == "trend_alignment"
                else "trade net_per_notional plus trend_alignment alpha from Omega 4.2 replay"
                if str(args.risk_target_mode) == "net_plus_trend_alignment"
                else "trade net_per_notional from Omega 4.2 replay"
            ),
            "risk_target_mode": str(args.risk_target_mode),
            "target_mae_penalty": float(args.target_mae_penalty),
            "risk_trend_diag": risk_trend_diag,
            "target_mean": float(np.mean(y_train_trade)),
            "target_p25": float(np.quantile(y_train_trade, 0.25)),
            "target_p50": float(np.quantile(y_train_trade, 0.50)),
            "target_p75": float(np.quantile(y_train_trade, 0.75)),
            "train_score_q50": train_q50,
            "train_score_iqr": train_iqr,
        },
        "selected": {
            "variant": selected["variant"],
            "mapping": selected_cfg,
            "selection_rule": (
                f"validation-only {str(args.selection_objective)} max with validation_mdd >= -{abs(float(args.max_validation_mdd_abs)):.2f} "
                f"and trades >= {min_trade_ratio:.2f} * baseline trades; OOS excluded from filter/sort/tie-break"
                if str(args.selection_scope) == "validation_only"
                else f"validation {str(args.selection_objective)} max with validation_mdd >= -{abs(float(args.max_validation_mdd_abs)):.2f}, "
                f"oos_mdd >= -{abs(float(args.max_oos_mdd_abs)):.2f}, and trades >= {min_trade_ratio:.2f} * baseline trades"
            ),
            "constraint_pass": True,
            "fallback_used": False,
            "selection_failure_reason": None,
            "constraints": {
                "validation_trade_floor": val_trade_floor,
                "validation_mdd_floor": validation_mdd_floor,
                "min_validation_avg_notional": float(args.min_validation_avg_notional),
                "max_validation_avg_notional": float(args.max_validation_avg_notional),
            },
            "validation": selected_val_m,
            "oos": selected_oos_m,
            "selected_full_replay": {"validation": selected_val_replay_m, "oos": selected_oos_replay_m},
            "full_replay_selection_applied": bool(full_eligible),
        },
        "full_replay_selection_candidates": full_replay_selection,
        "top_validation": ranking.head(12).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "ranking": str(out_dir / "risk_mapping_ranking.csv"),
            "risk_sidecar": str(out_dir / "risk_sidecar.pkl"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "baseline": report["omega4_2_replayed_baseline"], "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
