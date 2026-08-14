#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_3head_parent72_loose_entry_quality_20260620"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
CMAMBA_FEATURE_PREFIX = "regime3_cmamba_h6_sidecar_"


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _focal_modulate(ce: torch.Tensor, gamma: float) -> torch.Tensor:
    """Apply Lin et al. 2017 focal modulation (1-p_t)^gamma to per-sample CE.
    gamma<=0 is a no-op (returns ce unchanged) so existing bundles reproduce exactly.
    Class-balance (alpha_t) is left to the caller's existing sample weights."""
    if float(gamma) <= 0.0:
        return ce
    p_t = torch.exp(-ce)
    return ((1.0 - p_t).clamp(min=0.0) ** float(gamma)) * ce


def _read_labels(label_dir: Path, year: int, *, require_diagnostics: bool) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    required = {"timestamp", "zigzag_action"}
    if require_diagnostics:
        required |= {
            "zigzag_path_edge",
            "zigzag_path_mae",
            "zigzag_path_mfe",
            "zigzag_soft_long",
            "zigzag_soft_short",
        }
    missing = sorted(required - set(labels.columns))
    if missing:
        raise RuntimeError(f"{path} missing Omega4 label columns: {missing}")
    out = labels[list(required)].dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    y = pd.to_numeric(out["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    invalid = sorted(set(np.unique(y).tolist()) - {0, 1, 2})
    if invalid:
        raise RuntimeError(f"{path} invalid zigzag_action classes: {invalid}")
    return out


def _quality_target_hard_rule(labels: pd.DataFrame) -> np.ndarray:
    action = pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    soft_long = pd.to_numeric(labels["zigzag_soft_long"], errors="raise").to_numpy(dtype=np.float64)
    soft_short = pd.to_numeric(labels["zigzag_soft_short"], errors="raise").to_numpy(dtype=np.float64)
    edge = pd.to_numeric(labels["zigzag_path_edge"], errors="raise").to_numpy(dtype=np.float64)
    mae = pd.to_numeric(labels["zigzag_path_mae"], errors="raise").to_numpy(dtype=np.float64)
    mfe = pd.to_numeric(labels["zigzag_path_mfe"], errors="raise").to_numpy(dtype=np.float64)
    side_soft = np.where(action == 1, soft_long, np.where(action == 2, soft_short, 0.0))
    mfe_mae = mfe / np.maximum(mae, 0.001)
    good = (action != 0) & (side_soft >= 0.70) & (edge > 0.0) & (mae <= 0.010) & (mfe_mae >= 1.50)
    out = np.zeros(len(labels), dtype=np.int64)
    out[good] = action[good]
    return out


def _quality_target(labels: pd.DataFrame, *, mode: str, quality_labels: pd.DataFrame | None) -> np.ndarray:
    if mode == "same_as_direction":
        return pd.to_numeric(labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    if mode == "hard_rule":
        return _quality_target_hard_rule(labels)
    if quality_labels is None:
        raise RuntimeError(f"quality_labels required for quality mode: {mode}")
    if mode == "quality_label_action":
        return pd.to_numeric(quality_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    if mode == "quality_label_hard_rule":
        return _quality_target_hard_rule(quality_labels)
    raise RuntimeError(f"unknown quality mode: {mode}")


def _quality_target_barrier_meta_action(frame: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[np.ndarray, dict[str, Any]]:
    required = {"zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"barrier meta quality target missing columns: {missing}")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    out = np.zeros(len(frame), dtype=np.int64)
    active = 0
    filled = 0
    positive = 0
    net_values: list[float] = []
    for i in range(0, len(frame) - 2):
        a = int(action[i])
        if a not in (1, 2):
            continue
        active += 1
        side = 1 if a == 1 else -1
        ok, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not ok:
            net_values.append(-1.0)
            continue
        filled += 1
        entry_i = min(int(i) + 1, len(frame) - 1)
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        net, _final_i, _reason = exit_head._continue_to_barrier_net(
            arrays,
            start_i=entry_i,
            side=side,
            entry_price=float(entry_price),
            cash_after_entry_fee=cash_after_entry_fee,
            notional=notional,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
        )
        net_values.append(float(net))
        if float(net) > 0.0:
            out[i] = a
            positive += 1
    arr = np.asarray(net_values, dtype=np.float64) if net_values else np.asarray([0.0], dtype=np.float64)
    return out, {
        "mode": "barrier_meta_action",
        "active_rows": int(active),
        "filled_entries": int(filled),
        "positive_rows": int(positive),
        "positive_rate_active": float(positive / max(active, 1)),
        "net_mean": float(arr.mean()),
        "net_p10": float(np.quantile(arr, 0.10)),
        "net_p50": float(np.quantile(arr, 0.50)),
        "net_p90": float(np.quantile(arr, 0.90)),
    }


def _quality_target_risk_adjusted_barrier_meta_action(
    frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    min_edge: float,
    max_mae: float,
    min_mfe_mae: float,
    max_hold_bars: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    required = {"zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"risk-adjusted barrier meta quality target missing columns: {missing}")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    max_hold = int(max_hold_bars)
    out = np.zeros(len(frame), dtype=np.int64)
    active = 0
    filled = 0
    positive = 0
    reason_counts: dict[str, int] = {}
    net_values: list[float] = []
    mfe_values: list[float] = []
    mae_values: list[float] = []
    ratio_values: list[float] = []
    hold_values: list[int] = []
    for i in range(0, len(frame) - 2):
        a = int(action[i])
        if a not in (1, 2):
            continue
        active += 1
        side = 1 if a == 1 else -1
        ok, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not ok:
            reason_counts["entry_not_filled"] = reason_counts.get("entry_not_filled", 0) + 1
            net_values.append(-1.0)
            continue
        filled += 1
        entry_i = min(int(i) + 1, len(frame) - 1)
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        net, final_i, _reason = exit_head._continue_to_barrier_net(
            arrays,
            start_i=entry_i,
            side=side,
            entry_price=float(entry_price),
            cash_after_entry_fee=cash_after_entry_fee,
            notional=notional,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
        )
        end_i = max(min(int(final_i), len(frame) - 1), int(entry_i))
        mfe = 0.0
        mae = 0.0
        for row_i in range(int(entry_i), end_i + 1):
            px = float(arrays["close"][int(row_i)])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, float(unreal))
            mae = min(mae, float(unreal))
        mae_abs = abs(float(mae))
        mfe_mae = float(mfe) / max(mae_abs, 1.0e-8)
        hold_bars = max(int(end_i) - int(entry_i), 0)
        net_values.append(float(net))
        mfe_values.append(float(mfe))
        mae_values.append(float(mae_abs))
        ratio_values.append(float(mfe_mae))
        hold_values.append(int(hold_bars))
        if float(net) <= float(min_edge):
            reason = "net_edge_fail"
        elif mae_abs > float(max_mae):
            reason = "mae_fail"
        elif mfe_mae < float(min_mfe_mae):
            reason = "mfe_mae_fail"
        elif max_hold > 0 and hold_bars > max_hold:
            reason = "hold_fail"
        else:
            reason = "pass"
            out[i] = a
            positive += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    net_arr = np.asarray(net_values, dtype=np.float64) if net_values else np.asarray([0.0], dtype=np.float64)
    mfe_arr = np.asarray(mfe_values, dtype=np.float64) if mfe_values else np.asarray([0.0], dtype=np.float64)
    mae_arr = np.asarray(mae_values, dtype=np.float64) if mae_values else np.asarray([0.0], dtype=np.float64)
    ratio_arr = np.asarray(ratio_values, dtype=np.float64) if ratio_values else np.asarray([0.0], dtype=np.float64)
    hold_arr = np.asarray(hold_values, dtype=np.float64) if hold_values else np.asarray([0.0], dtype=np.float64)
    return out, {
        "mode": "risk_adjusted_barrier_meta_action",
        "active_rows": int(active),
        "filled_entries": int(filled),
        "positive_rows": int(positive),
        "positive_rate_active": float(positive / max(active, 1)),
        "reason_counts": reason_counts,
        "thresholds": {
            "min_edge": float(min_edge),
            "max_mae": float(max_mae),
            "min_mfe_mae": float(min_mfe_mae),
            "max_hold_bars": int(max_hold),
        },
        "net_mean": float(net_arr.mean()),
        "net_p10": float(np.quantile(net_arr, 0.10)),
        "net_p50": float(np.quantile(net_arr, 0.50)),
        "net_p90": float(np.quantile(net_arr, 0.90)),
        "mfe_p50": float(np.quantile(mfe_arr, 0.50)),
        "mae_p50": float(np.quantile(mae_arr, 0.50)),
        "mfe_mae_p50": float(np.quantile(ratio_arr, 0.50)),
        "hold_bars_p50": float(np.quantile(hold_arr, 0.50)),
        "hold_bars_p90": float(np.quantile(hold_arr, 0.90)),
    }


def _prepare_frames(
    *,
    disable_tp_sl: bool,
    direction_label_dir: Path,
    quality_mode: str,
    quality_label_dir: Path | None,
    quality_min_edge: float,
    quality_max_mae: float,
    quality_min_mfe_mae: float,
    quality_max_hold_bars: int,
) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    need_direction_diag = str(quality_mode) == "hard_rule"
    need_quality_diag = str(quality_mode) == "quality_label_hard_rule"
    label_2025 = _read_labels(direction_label_dir, 2025, require_diagnostics=need_direction_diag)
    label_2026 = _read_labels(direction_label_dir, 2026, require_diagnostics=need_direction_diag)
    q_label_2025 = None
    q_label_2026 = None
    if str(quality_mode).startswith("quality_label_"):
        if quality_label_dir is None:
            raise RuntimeError("quality_label_dir is required when quality_mode starts with quality_label_")
        q_label_2025 = _read_labels(quality_label_dir, 2025, require_diagnostics=need_quality_diag)
        q_label_2026 = _read_labels(quality_label_dir, 2026, require_diagnostics=need_quality_diag)
    train_all, train_labels = omega._align(train_all, label_2025, "omega4 train labels")
    eval_df, eval_labels = omega._align(eval_df, label_2026, "omega4 oos labels")
    if q_label_2025 is not None:
        train_all_for_q, train_quality_labels = omega._align(train_all[["timestamp"]], q_label_2025, "omega4 train quality labels")
        if len(train_all_for_q) != len(train_all):
            raise RuntimeError("omega4 train quality alignment changed row count")
    else:
        train_quality_labels = train_labels
    if q_label_2026 is not None:
        eval_for_q, eval_quality_labels = omega._align(eval_df[["timestamp"]], q_label_2026, "omega4 oos quality labels")
        if len(eval_for_q) != len(eval_df):
            raise RuntimeError("omega4 oos quality alignment changed row count")
    else:
        eval_quality_labels = eval_labels
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    if str(quality_mode) == "barrier_meta_action":
        fee, slip = omega._load_fee_slip()
        train_all["omega4_quality_action"], train_quality_diag = _quality_target_barrier_meta_action(train_all, fee=fee, slip=slip, cost_mult=3.0)
        eval_df["omega4_quality_action"], eval_quality_diag = _quality_target_barrier_meta_action(eval_df, fee=fee, slip=slip, cost_mult=3.0)
    elif str(quality_mode) == "risk_adjusted_barrier_meta_action":
        fee, slip = omega._load_fee_slip()
        train_all["omega4_quality_action"], train_quality_diag = _quality_target_risk_adjusted_barrier_meta_action(
            train_all,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            min_edge=float(quality_min_edge),
            max_mae=float(quality_max_mae),
            min_mfe_mae=float(quality_min_mfe_mae),
            max_hold_bars=int(quality_max_hold_bars),
        )
        eval_df["omega4_quality_action"], eval_quality_diag = _quality_target_risk_adjusted_barrier_meta_action(
            eval_df,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            min_edge=float(quality_min_edge),
            max_mae=float(quality_max_mae),
            min_mfe_mae=float(quality_min_mfe_mae),
            max_hold_bars=int(quality_max_hold_bars),
        )
    else:
        train_all["omega4_quality_action"] = _quality_target(train_labels, mode=str(quality_mode), quality_labels=train_quality_labels)
        eval_df["omega4_quality_action"] = _quality_target(eval_labels, mode=str(quality_mode), quality_labels=eval_quality_labels)
        train_quality_diag = {"mode": str(quality_mode)}
        eval_quality_diag = {"mode": str(quality_mode)}
    train_raw = train_all[train_all["timestamp"] < parent.SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= parent.SPLIT_TS].reset_index(drop=True)

    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, train_src = omega._align(train_raw, tabm_2025, "train")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    if disable_tp_sl:
        train_fixed = exit_head._disable_tp_sl(train_fixed)
    s_train_label = parent._base_input(train_df, feature_cols)
    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "train_df": train_df,
        "train_fixed": train_fixed,
        "s_train_label": s_train_label,
        "feature_cols": feature_cols,
        "overlay_report": overlay_report,
        "label_quality_summary": {
            "train": _label_summary(train_raw),
            "validation": _label_summary(val_raw),
            "oos": _label_summary(eval_df),
        },
        "quality_target_diag": {
            "train_all": train_quality_diag,
            "oos": eval_quality_diag,
        },
        "label_contract": {
            "direction_label_dir": str(direction_label_dir),
            "quality_mode": str(quality_mode),
            "quality_label_dir": str(quality_label_dir) if quality_label_dir is not None else None,
        },
    }


def _label_summary(frame: pd.DataFrame) -> dict[str, Any]:
    y_dir = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    y_qual = pd.to_numeric(frame["omega4_quality_action"], errors="raise").to_numpy(dtype=np.int64)
    return {
        "rows": int(len(frame)),
        "direction_counts": {str(k): int(v) for k, v in pd.Series(y_dir).value_counts().sort_index().items()},
        "quality_counts": {str(k): int(v) for k, v in pd.Series(y_qual).value_counts().sort_index().items()},
        "quality_active_ratio": float((y_qual != 0).mean()) if len(y_qual) else 0.0,
    }


def _fit_expert_omega4(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
    direction_class_weights: dict[int, float],
    quality_class_weights: dict[int, float],
    direction_focal_gamma: float = 0.0,
    hard_regime_filter: bool = False,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_probs = parent._route_probs(route_frame)
    exit_route_probs = parent._route_probs(exit_route_frame)
    if hard_regime_filter:
        route_w = (route_probs.argmax(axis=1) == int(expert_idx)).astype(np.float32)
        exit_w = (exit_route_probs.argmax(axis=1) == int(expert_idx)).astype(np.float32)
    else:
        route_w = route_probs[:, int(expert_idx)].astype(np.float32)
        exit_w = exit_route_probs[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    dir_w *= np.asarray([float(direction_class_weights.get(int(y), 1.0)) for y in y_dir_np], dtype=np.float32)
    qual_w *= np.asarray([float(quality_class_weights.get(int(y), 1.0)) for y in y_qual_np], dtype=np.float32)
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid Omega4 sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = parent.ThreeHeadTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yqb, wb, qwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yqb = yqb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            qwb = qwb.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True)
            ye = ye.to(device, non_blocking=True)
            we = we.to(device, non_blocking=True)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3),
                yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir_k = _focal_modulate(loss_dir_k, direction_focal_gamma)
            loss_qual_k = torch.nn.functional.cross_entropy(
                out_dir["quality"].reshape(-1, 3),
                yqb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(parent.CFG.quality_loss_weight) * loss_qual + float(parent.CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vdir = _focal_modulate(vdir, direction_focal_gamma)
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(parent.CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                    + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
                )
                .detach()
                .cpu()
            )
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": parent.CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_dir.columns),
        "quality_target": "omega4_quality_action",
        "direction_class_weights": {str(k): float(v) for k, v in direction_class_weights.items()},
        "quality_class_weights": {str(k): float(v) for k, v in quality_class_weights.items()},
        "direction_focal_gamma": float(direction_focal_gamma),
        "hard_regime_filter": bool(hard_regime_filter),
    }
    torch.save(payload, model_path)
    return payload


def _parse_class_weights(raw: str) -> dict[int, float]:
    out = {0: 1.0, 1: 1.0, 2: 1.0}
    text = str(raw).strip()
    if not text:
        return out
    for part in text.split(","):
        if not part.strip():
            continue
        if ":" not in part:
            raise RuntimeError(f"invalid class weight entry: {part!r}")
        key, value = part.split(":", 1)
        cls = int(key.strip())
        if cls not in out:
            raise RuntimeError(f"invalid action class weight key: {cls}")
        weight = float(value.strip())
        if not np.isfinite(weight) or weight <= 0.0:
            raise RuntimeError(f"invalid action class weight value for {cls}: {weight}")
        out[cls] = weight
    return out


def _metric_row(name: str, metrics: dict[str, Any], quality_threshold: float) -> dict[str, Any]:
    return {
        "variant": f"q{quality_threshold:.2f}".replace(".", "p"),
        "quality_threshold": float(quality_threshold),
        f"{name}_pnl": float(metrics["pnl"]),
        f"{name}_mdd": float(metrics["mdd"]),
        f"{name}_wr": float(metrics["wr"]),
        f"{name}_trades": int(metrics["trades"]),
    }


def _apply_compensated(dec: pd.DataFrame, *, scale: float, cap: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active_idx = np.flatnonzero(omega._active(out))
    if len(active_idx) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    new_notional = np.minimum(base_notional * float(scale), float(cap))
    ratio = new_notional / np.maximum(base_notional, 1.0e-12)
    out.loc[active_idx, "notional_exposure"] = new_notional
    out.loc[active_idx, "position_fraction"] = new_notional
    out.loc[active_idx, "take_profit"] = pd.to_numeric(out.loc[active_idx, "take_profit"], errors="raise").to_numpy(dtype=np.float64) * ratio
    out.loc[active_idx, "stop_loss"] = pd.to_numeric(out.loc[active_idx, "stop_loss"], errors="raise").to_numpy(dtype=np.float64) * ratio
    return out


def _build_exit_dataset_entry_label_path_optimal(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    exit_edge_min: float,
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"entry-label path-optimal exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("entry-label path-optimal exit frame/state length mismatch")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    exit_edges: list[float] = []
    reason_counts: dict[str, int] = {}
    segment_count = 0
    skipped_segments = 0
    positive_count = 0
    segment_id = -1
    i = 0
    last_i = len(frame) - 2
    while i < last_i:
        side_action = int(action[i])
        if side_action not in (1, 2):
            i += 1
            continue
        start_i = i
        while i < last_i and int(action[i]) == side_action:
            i += 1
        end_i = min(i - 1, last_i)
        side = 1 if side_action == 1 else -1
        segment_id += 1
        filled, entry_price, entry_fee, _route = omega._try_execution(
            arrays,
            int(start_i),
            side,
            entry=True,
            fee_base=fee_eff,
            slip_base=slip_eff,
        )
        entry_i = min(int(start_i) + 1, len(frame) - 1)
        if not filled or end_i < entry_i:
            skipped_segments += 1
            continue
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        path_idx = np.arange(entry_i, end_i + 1, dtype=np.int64)
        exit_net = np.zeros(len(path_idx), dtype=np.float64)
        exit_fill_i = np.zeros(len(path_idx), dtype=np.int64)
        for k, row_i in enumerate(path_idx):
            net, fill_i, _reason = exit_head._exit_fill_net(
                arrays,
                signal_i=int(row_i),
                side=side,
                entry_price=float(entry_price),
                cash_after_entry_fee=cash_after_entry_fee,
                notional=notional,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
            exit_net[k] = float(net)
            exit_fill_i[k] = int(fill_i)
        suffix_best_value = np.maximum.accumulate(exit_net[::-1])[::-1]
        suffix_best_pos_from_here = np.zeros(len(exit_net), dtype=np.int64)
        best_pos = len(exit_net) - 1
        best_value = exit_net[best_pos]
        for k in range(len(exit_net) - 1, -1, -1):
            if exit_net[k] >= best_value:
                best_value = exit_net[k]
                best_pos = k
            suffix_best_pos_from_here[k] = best_pos

        entry_state = state.iloc[int(start_i)]
        mfe = 0.0
        mae = 0.0
        for k, row_i in enumerate(path_idx):
            px = float(arrays["close"][int(row_i)])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if k == len(path_idx) - 1:
                future_value = float("-inf")
                edge = 0.0
                label = 1
                reason = "segment_end_forced_exit"
                best_future_i = int(row_i)
                best_future_fill_i = int(exit_fill_i[k])
                best_future_net = float(exit_net[k])
            else:
                future_pos = int(suffix_best_pos_from_here[k + 1])
                best_future_net = float(suffix_best_value[k + 1])
                future_value = best_future_net
                edge = float(exit_net[k] - future_value)
                label = int(edge >= float(exit_edge_min))
                reason = "oracle_dp_exit_now" if label else "oracle_dp_hold"
                best_future_i = int(path_idx[future_pos])
                best_future_fill_i = int(exit_fill_i[future_pos])
            row = exit_head._position_feature_row(
                state,
                entry_state,
                row_i=int(row_i),
                side=side,
                entry_price=float(entry_price),
                entry_i=int(entry_i),
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mfe=mfe,
                mae=mae,
                unreal=unreal,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            exit_edges.append(float(edge))
            frow = frame.iloc[int(row_i)].copy()
            frow["exit_path_segment_id"] = int(segment_id)
            frow["exit_path_entry_signal_i"] = int(start_i)
            frow["exit_path_entry_i"] = int(entry_i)
            frow["exit_path_end_i"] = int(end_i)
            frow["exit_path_side"] = int(side)
            frow["exit_path_hold_bars"] = int(max(int(row_i) - int(entry_i), 0))
            frow["exit_path_entry_price"] = float(entry_price)
            frow["exit_path_now_net"] = float(exit_net[k])
            frow["exit_path_best_future_net"] = float(best_future_net)
            frow["exit_path_future_value"] = float(future_value) if np.isfinite(future_value) else float("nan")
            frow["exit_path_edge"] = float(edge)
            frow["exit_path_label"] = int(label)
            frow["exit_path_reason"] = reason
            frow["exit_path_best_future_i"] = int(best_future_i)
            frow["exit_path_best_future_fill_i"] = int(best_future_fill_i)
            frow["exit_path_mfe"] = float(mfe)
            frow["exit_path_mae"] = float(mae)
            frow["exit_path_unrealized"] = float(unreal)
            frame_rows.append(frow)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
        segment_count += 1
        if max_samples > 0 and len(rows) >= int(max_samples):
            break
    if not rows:
        raise RuntimeError("empty entry-label path-optimal Exit Head dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "exit_edge_mean": float(np.mean(exit_edges)) if exit_edges else 0.0,
        "exit_edge_p50": float(np.quantile(exit_edges, 0.50)) if exit_edges else 0.0,
        "exit_edge_p90": float(np.quantile(exit_edges, 0.90)) if exit_edges else 0.0,
        "exit_edge_p99": float(np.quantile(exit_edges, 0.99)) if exit_edges else 0.0,
        "continued_exit_reasons": reason_counts,
        "used_segments": int(segment_count),
        "skipped_segments": int(skipped_segments),
        "risk_template": {
            "notional": notional,
            "leverage": leverage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        },
        "label_mode": "entry_label_path_optimal_stopping_every_in_position_bar",
    }


def _build_exit_dataset_entry_label_terminal_giveback(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_samples: int,
    terminal_window: int = 3,
    adverse_unreal: float = -0.010,
    min_mfe_for_giveback: float = 0.006,
    giveback_min: float = 0.65,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    required = {"timestamp", "zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"entry-label terminal-giveback exit dataset missing columns: {missing}")
    if len(frame) != len(state):
        raise RuntimeError("entry-label terminal-giveback exit frame/state length mismatch")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    leverage = float(omega.BASE_TEMPLATE["leverage"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    tw = max(int(terminal_window), 1)

    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    reason_counts: dict[str, int] = {}
    used_segments = 0
    skipped_segments = 0
    positive_count = 0
    segment_id = -1
    i = 0
    last_i = len(frame) - 2
    while i < last_i:
        side_action = int(action[i])
        if side_action not in (1, 2):
            i += 1
            continue
        start_i = i
        while i < last_i and int(action[i]) == side_action:
            i += 1
        end_i = min(i - 1, last_i)
        side = 1 if side_action == 1 else -1
        segment_id += 1
        filled, entry_price, entry_fee, _route = omega._try_execution(
            arrays,
            int(start_i),
            side,
            entry=True,
            fee_base=fee_eff,
            slip_base=slip_eff,
        )
        del entry_fee
        entry_i = min(int(start_i) + 1, len(frame) - 1)
        if not filled or end_i < entry_i:
            skipped_segments += 1
            continue
        entry_state = state.iloc[int(start_i)]
        mfe = 0.0
        mae = 0.0
        for row_i in range(entry_i, end_i + 1):
            px = float(arrays["close"][int(row_i)])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            giveback = (mfe - unreal) / max(abs(mfe), 1.0e-8) if mfe > 0.0 else 0.0
            bars_to_segment_end = int(end_i) - int(row_i)
            terminal = bars_to_segment_end < tw
            adverse = unreal <= float(adverse_unreal)
            gave_back = mfe >= float(min_mfe_for_giveback) and giveback >= float(giveback_min) and unreal > 0.0
            if terminal:
                label = 1
                reason = "terminal_window_exit"
            elif adverse:
                label = 1
                reason = "adverse_unreal_exit"
            elif gave_back:
                label = 1
                reason = "mfe_giveback_exit"
            else:
                label = 0
                reason = "hold"
            row = exit_head._position_feature_row(
                state,
                entry_state,
                row_i=int(row_i),
                side=side,
                entry_price=float(entry_price),
                entry_i=int(entry_i),
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mfe=mfe,
                mae=mae,
                unreal=unreal,
            )
            rows.append(row)
            labels.append(label)
            positive_count += int(label)
            frow = frame.iloc[int(row_i)].copy()
            frow["exit_path_segment_id"] = int(segment_id)
            frow["exit_path_entry_signal_i"] = int(start_i)
            frow["exit_path_entry_i"] = int(entry_i)
            frow["exit_path_end_i"] = int(end_i)
            frow["exit_path_side"] = int(side)
            frow["exit_path_hold_bars"] = int(max(int(row_i) - int(entry_i), 0))
            frow["exit_terminal_giveback_label"] = int(label)
            frow["exit_terminal_giveback_reason"] = reason
            frow["exit_path_mfe"] = float(mfe)
            frow["exit_path_mae"] = float(mae)
            frow["exit_path_unrealized"] = float(unreal)
            frow["exit_path_giveback"] = float(giveback)
            frow["exit_path_bars_to_segment_end"] = int(bars_to_segment_end)
            frame_rows.append(frow)
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
        used_segments += 1
        if max_samples > 0 and len(rows) >= int(max_samples):
            break
    if not rows:
        raise RuntimeError("empty entry-label terminal-giveback Exit Head dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(positive_count),
        "negative_count": int(len(y) - positive_count),
        "continued_exit_reasons": reason_counts,
        "used_segments": int(used_segments),
        "skipped_segments": int(skipped_segments),
        "risk_template": {
            "notional": notional,
            "leverage": leverage,
            "take_profit": take_profit,
            "stop_loss": stop_loss,
        },
        "label_mode": "entry_label_terminal_giveback_every_in_position_bar",
        "terminal_window": int(tw),
        "adverse_unreal": float(adverse_unreal),
        "min_mfe_for_giveback": float(min_mfe_for_giveback),
        "giveback_min": float(giveback_min),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--exit-label-mode", choices=["independent_entry_hold_offsets", "entry_label_path_optimal", "entry_label_terminal_giveback"], default="independent_entry_hold_offsets")
    ap.add_argument("--exit-terminal-window", type=int, default=3)
    ap.add_argument("--exit-adverse-unreal", type=float, default=-0.010)
    ap.add_argument("--exit-min-mfe-for-giveback", type=float, default=0.006)
    ap.add_argument("--exit-giveback-min", type=float, default=0.65)
    ap.add_argument("--quality-thresholds", default="0.40,0.45,0.50,0.55,0.60")
    ap.add_argument("--direction-label-dir", type=Path, default=LABEL_DIR)
    ap.add_argument("--quality-mode", choices=["same_as_direction", "hard_rule", "quality_label_action", "quality_label_hard_rule", "barrier_meta_action", "risk_adjusted_barrier_meta_action"], default="hard_rule")
    ap.add_argument("--quality-label-dir", type=Path, default=None)
    ap.add_argument("--quality-min-edge", type=float, default=0.0010)
    ap.add_argument("--quality-max-mae", type=float, default=0.0100)
    ap.add_argument("--quality-min-mfe-mae", type=float, default=1.20)
    ap.add_argument("--quality-max-hold-bars", type=int, default=288)
    ap.add_argument("--max-exit-samples", type=int, default=12000)
    ap.add_argument("--max-train-rows", type=int, default=30000)
    ap.add_argument("--disable-tp-sl", action="store_true")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260620)
    ap.add_argument("--out-suffix", default="smoke_e4_train30k_exit12k")
    ap.add_argument("--drop-cmamba-features", action="store_true")
    ap.add_argument("--direction-class-weights", default="")
    ap.add_argument("--quality-class-weights", default="")
    ap.add_argument("--direction-focal-gamma", type=float, default=0.0)
    ap.add_argument("--hard-regime-filter", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    direction_class_weights = _parse_class_weights(str(args.direction_class_weights))
    quality_class_weights = _parse_class_weights(str(args.quality_class_weights))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames(
        disable_tp_sl=bool(args.disable_tp_sl),
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=Path(args.quality_label_dir) if args.quality_label_dir is not None else None,
        quality_min_edge=float(args.quality_min_edge),
        quality_max_mae=float(args.quality_max_mae),
        quality_min_mfe_mae=float(args.quality_min_mfe_mae),
        quality_max_hold_bars=int(args.quality_max_hold_bars),
    )
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    dropped_feature_cols: list[str] = []
    if bool(args.drop_cmamba_features):
        dropped_feature_cols = [c for c in base_cols if str(c).startswith(CMAMBA_FEATURE_PREFIX)]
        if not dropped_feature_cols:
            raise RuntimeError("drop-cmamba-features requested but no cmamba sidecar feature columns were present")
        base_cols = [c for c in base_cols if not str(c).startswith(CMAMBA_FEATURE_PREFIX)]
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_train = parent._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_raw["omega4_quality_action"].to_numpy(dtype=np.int64)
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_quality = y_quality[:limit]
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
    else:
        train_fit_frame = train_raw

    hold_offsets = [int(x.strip()) for x in str(args.exit_hold_offsets).split(",") if x.strip()]
    if str(args.exit_label_mode) == "entry_label_path_optimal":
        x_exit_raw, y_exit, frame_exit, exit_diag = _build_exit_dataset_entry_label_path_optimal(
            frames["train_df"],
            frames["s_train_label"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            exit_edge_min=float(args.exit_edge_min),
            max_samples=int(args.max_exit_samples),
        )
    elif str(args.exit_label_mode) == "entry_label_terminal_giveback":
        x_exit_raw, y_exit, frame_exit, exit_diag = _build_exit_dataset_entry_label_terminal_giveback(
            frames["train_df"],
            frames["s_train_label"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            max_samples=int(args.max_exit_samples),
            terminal_window=int(args.exit_terminal_window),
            adverse_unreal=float(args.exit_adverse_unreal),
            min_mfe_for_giveback=float(args.exit_min_mfe_for_giveback),
            giveback_min=float(args.exit_giveback_min),
        )
    else:
        x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
            frames["train_df"],
            frames["s_train_label"],
            frames["train_fixed"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            exit_edge_min=float(args.exit_edge_min),
            hold_offsets=hold_offsets,
            max_samples=int(args.max_exit_samples),
        )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_omega4(
            x_train,
            y_train,
            y_quality,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_3head_tabm.pt",
            direction_class_weights=direction_class_weights,
            quality_class_weights=quality_class_weights,
            direction_focal_gamma=float(args.direction_focal_gamma),
            hard_regime_filter=bool(args.hard_regime_filter),
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_3head_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_raw(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, dict[str, np.ndarray]], np.ndarray]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: parent._predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        return x, preds, route

    _x_train_pred, train_preds, train_route = predict_raw(train_raw)
    _x_val, val_preds, val_route = predict_raw(val_raw)
    _x_oos, oos_preds, oos_route = predict_raw(oos_raw)
    train_direction = parent._routed(train_preds, train_route, "direction", 3)
    train_quality = parent._routed(train_preds, train_route, "quality", 3)
    val_direction = parent._routed(val_preds, val_route, "direction", 3)
    val_quality = parent._routed(val_preds, val_route, "quality", 3)
    oos_direction = parent._routed(oos_preds, oos_route, "direction", 3)
    oos_quality = parent._routed(oos_preds, oos_route, "quality", 3)

    q_values = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    saved_predictions: dict[str, str] = {}
    prediction_artifacts: dict[str, dict[str, str]] = {}
    for q in q_values:
        q_tag = f"q{int(round(float(q) * 100.0)):03d}"
        train_src = parent._prediction_output(train_raw, train_direction, train_quality, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
        val_src = parent._prediction_output(val_raw, val_direction, val_quality, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
        oos_src_oof = parent._prediction_output(oos_raw, oos_direction, oos_quality, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
        oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
        train_path = out_dir / f"train_predictions_{q_tag}.csv"
        val_path = out_dir / f"validation_predictions_{q_tag}.csv"
        oos_path = out_dir / f"oos_predictions_{q_tag}.csv"
        train_src.to_csv(train_path, index=False)
        val_src.to_csv(val_path, index=False)
        oos_src.to_csv(oos_path, index=False)
        prediction_artifacts[q_tag] = {
            "train": str(train_path),
            "validation": str(val_path),
            "oos": str(oos_path),
        }
        val_dec = parent._to_decisions(val_src, oof=True)
        oos_dec = parent._to_decisions(oos_src, oof=False)
        if bool(args.disable_tp_sl):
            val_dec = exit_head._disable_tp_sl(val_dec)
            oos_dec = exit_head._disable_tp_sl(oos_dec)
        val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        val_aggr = omega._metrics(val_raw, _apply_compensated(val_dec, scale=2.0, cap=0.90), fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_aggr = omega._metrics(oos_raw, _apply_compensated(oos_dec, scale=2.0, cap=0.90), fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        key = f"q{q:.2f}".replace(".", "p")
        reports[key] = {"validation": val_m, "oos": oos_m, "validation_aggressive_scale200_cap090": val_aggr, "oos_aggressive_scale200_cap090": oos_aggr}
        row = _metric_row("validation", val_m, q)
        row.update(_metric_row("oos", oos_m, q))
        row.update(
            {
                "validation_aggressive_pnl": float(val_aggr["pnl"]),
                "validation_aggressive_mdd": float(val_aggr["mdd"]),
                "validation_aggressive_wr": float(val_aggr["wr"]),
                "validation_aggressive_trades": int(val_aggr["trades"]),
                "oos_aggressive_pnl": float(oos_aggr["pnl"]),
                "oos_aggressive_mdd": float(oos_aggr["mdd"]),
                "oos_aggressive_wr": float(oos_aggr["wr"]),
                "oos_aggressive_trades": int(oos_aggr["trades"]),
            }
        )
        rows.append(row)
        if abs(float(q) - 0.45) < 1.0e-12:
            legacy_val_path = out_dir / "validation_predictions_2025_true3head_q045.csv"
            legacy_oos_path = out_dir / "oos_predictions_2026_true3head_q045.csv"
            val_src.to_csv(legacy_val_path, index=False)
            oos_src.to_csv(legacy_oos_path, index=False)
            saved_predictions = {"validation_q045": str(legacy_val_path), "oos_q045": str(legacy_oos_path)}
    # VAL-primary ranking is the selection view (OOS must never be the primary sort key for a
    # threshold that gets deployed) — see docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md
    # principle 1. `ranking_by_oos_pnl` is kept for information only, not for selection.
    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["oos_pnl"])), reverse=True)
    pd.DataFrame(rows).to_csv(out_dir / "quality_threshold_ranking.csv", index=False)
    ranking_by_oos_pnl = sorted(rows, key=lambda r: (float(r["oos_pnl"]), float(r["validation_pnl"])), reverse=True)
    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080",
        "label_dir": str(args.direction_label_dir),
        "label_contract": frames["label_contract"],
        "design": "Omega4 3-head TabM variant. Direction head and Quality head can use independent offline label targets. No label diagnostic columns are model inputs.",
        "quality_target_rule": {
            "mode": str(args.quality_mode),
            "active_action_required": True,
            "net_return_after_cost_min": float(args.quality_min_edge),
            "mae_max": float(args.quality_max_mae),
            "mfe_mae_min": float(args.quality_min_mfe_mae),
            "max_hold_bars": int(args.quality_max_hold_bars),
            "otherwise": "CASH",
        },
        "class_weight_overrides": {
            "direction": {str(k): float(v) for k, v in direction_class_weights.items()},
            "quality": {str(k): float(v) for k, v in quality_class_weights.items()},
        },
        "direction_focal_gamma": float(args.direction_focal_gamma),
        "hard_regime_filter": bool(args.hard_regime_filter),
        "input_contract": {
            "base_feature_count": len(base_cols),
            "position_feature_count": len(parent.POS_COLS),
            "total_features": len(base_cols) + len(parent.POS_COLS),
            "position_cols": parent.POS_COLS,
            "dropped_feature_cols": dropped_feature_cols,
        },
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
        "risk_template": {"max_hold_bars": omega.BASE_TEMPLATE["max_hold"], "cooldown_bars": omega.BASE_TEMPLATE["cooldown"], "tp_sl_disabled": bool(args.disable_tp_sl)},
        "canonical_prediction_contract": {
            "required_for_promotion": True,
            "tag_format": "qXXX where XXX is round(quality_threshold * 100), e.g. q055",
            "splits": ["train", "validation", "oos"],
            "train_and_validation_prefix": "omega1_regime3_expertdq_oof",
            "oos_prefix": "omega1_regime3_expertdq",
            "risk_sidecar_precomputed_prediction_dir": str(out_dir),
            "risk_sidecar_precomputed_prediction_tag_values": sorted(prediction_artifacts.keys()),
        },
        "label_quality_summary": frames["label_quality_summary"],
        "quality_target_diag": frames["quality_target_diag"],
        "exit_label": {"mode": str(args.exit_label_mode), "exit_edge_min": float(args.exit_edge_min), "hold_offsets": hold_offsets, "diag": exit_diag},
        "summaries": summaries,
        "results": reports,
        "selection_scope": "validation_only",
        "ranking_by_validation_pnl": rows,
        "ranking_by_oos_pnl": ranking_by_oos_pnl,
        "prediction_artifacts": prediction_artifacts,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "quality_threshold_ranking.csv"), "report": str(out_dir / "report.json"), **saved_predictions},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__}, out_dir / "true_3head_tabm_bundle.pt")
    print(json.dumps({"report": str(out_dir / "report.json"), "top": rows[:5], "q045": reports.get("q0p45")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
