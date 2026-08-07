#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_softfloor00_tabm_diffusion_risk_20260603"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"
EVAL_CSV = ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"
TABM_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_tabm_20260602/soft_floor_0p00"
TABM_2025 = TABM_DIR / "training_features_2025_soft_floor_0p00_omega1_regime3_expertdq_oof_20260602.csv"
TABM_2026 = TABM_DIR / "training_features_2026_rebuilt_soft_floor_0p00_omega1_regime3_expertdq_20260602.csv"
REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv"
REGIME3_CMAMBA_2025 = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2025_regime3_cryptomamba_h6_sidecar_20260601.csv"
REGIME3_CMAMBA_2026 = ROOT / "data/ensemble/supervised/regime3_cryptomamba_h6_sidecar_20260601/training_features_2026_rebuilt_regime3_cryptomamba_h6_sidecar_20260601.csv"
REGIME3_RISK_2025 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2025_regime3_stability_risk_h6.csv"
REGIME3_RISK_2026 = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530/training_features_2026_rebuilt_regime3_stability_risk_h6.csv"
SPLIT_TS = pd.Timestamp("2025-10-01")
ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2

BASE_TEMPLATE = {
    "notional": 0.45,
    "leverage": 2.0,
    "take_profit": 0.026,
    "stop_loss": 0.014,
    "max_hold": 72,
    "cooldown": 6,
}
FEE_RATE = 0.0005
SLIP_RATE = 0.0002
MAKER_FEE_MULT = 0.20
EXPERT_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90}
RISK_BOUNDS = {
    "take_profit": (0.008, 0.050),
    "stop_loss": (0.006, 0.035),
    "leverage": (1.0, 5.0),
    "notional": (0.10, 0.90),
}
RISK_COLS = ["take_profit", "stop_loss", "leverage", "notional"]

RISK_BOUND_PRESETS = {
    "absolute": {
        "take_profit": (0.008, 0.050),
        "stop_loss": (0.006, 0.035),
        "leverage": (1.0, 5.0),
        "notional": (0.10, 0.90),
    },
    "anchor_delta20": {
        "take_profit": (BASE_TEMPLATE["take_profit"] * 0.80, BASE_TEMPLATE["take_profit"] * 1.20),
        "stop_loss": (BASE_TEMPLATE["stop_loss"] * 0.80, BASE_TEMPLATE["stop_loss"] * 1.20),
        "leverage": (BASE_TEMPLATE["leverage"] * 0.80, BASE_TEMPLATE["leverage"] * 1.20),
        "notional": (BASE_TEMPLATE["notional"] * 0.80, BASE_TEMPLATE["notional"] * 1.20),
    },
    "anchor_delta35": {
        "take_profit": (BASE_TEMPLATE["take_profit"] * 0.65, BASE_TEMPLATE["take_profit"] * 1.35),
        "stop_loss": (BASE_TEMPLATE["stop_loss"] * 0.65, BASE_TEMPLATE["stop_loss"] * 1.35),
        "leverage": (BASE_TEMPLATE["leverage"] * 0.65, BASE_TEMPLATE["leverage"] * 1.35),
        "notional": (BASE_TEMPLATE["notional"] * 0.65, BASE_TEMPLATE["notional"] * 1.35),
    },
    "anchor_safe_size": {
        "take_profit": (BASE_TEMPLATE["take_profit"] * 0.80, BASE_TEMPLATE["take_profit"] * 1.25),
        "stop_loss": (BASE_TEMPLATE["stop_loss"] * 0.85, BASE_TEMPLATE["stop_loss"] * 1.35),
        "leverage": (1.0, BASE_TEMPLATE["leverage"] * 1.15),
        "notional": (BASE_TEMPLATE["notional"] * 0.45, BASE_TEMPLATE["notional"] * 1.05),
    },
    "anchor_exit35_size_neutral": {
        "take_profit": (BASE_TEMPLATE["take_profit"] * 0.65, BASE_TEMPLATE["take_profit"] * 1.35),
        "stop_loss": (BASE_TEMPLATE["stop_loss"] * 0.65, BASE_TEMPLATE["stop_loss"] * 1.35),
        "leverage": (BASE_TEMPLATE["leverage"] * 0.70, BASE_TEMPLATE["leverage"] * 1.20),
        "notional": (BASE_TEMPLATE["notional"] * 0.75, BASE_TEMPLATE["notional"] * 1.00),
    },
}

DENY_PREFIXES = ("clean_regime4_", "regime4_pred_", "regime3_pred_", "teacher_", "teacher_oof_", "a5dir_")
DENY_TOKENS = ("target", "future", "label", "pnl", "zigzag", "wave3", "tp_sl_action_score")
NON_FEATURE_COLS = {"timestamp"}
REGIME3_CURRENT_COLS = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]
REGIME3_CMAMBA_COLS = [
    "regime3_cmamba_h6_sidecar_bull_prob",
    "regime3_cmamba_h6_sidecar_bear_prob",
    "regime3_cmamba_h6_sidecar_chop_prob",
    "regime3_cmamba_h6_sidecar_class_id",
    "regime3_cmamba_h6_sidecar_confidence",
    "regime3_cmamba_h6_sidecar_transition_prob",
    "regime3_cmamba_h6_sidecar_stability_score",
]
REGIME3_RISK_COLS = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{path} missing timestamp")
    return df.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _edge_name(mask: pd.Series) -> str | None:
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return None
    if np.array_equal(idx, np.arange(len(idx))):
        return "head"
    if np.array_equal(idx, np.arange(len(mask) - len(idx), len(mask))):
        return "tail"
    return None


def _overlay_required(base: pd.DataFrame, source: Path, cols: list[str], *, tag: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    src = _read(source)
    missing = [c for c in cols if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing required columns: {missing}")
    out = base.copy()
    src_ts = set(pd.to_datetime(src["timestamp"], errors="raise"))
    missing_ts = out.loc[~pd.to_datetime(out["timestamp"], errors="raise").isin(src_ts), "timestamp"]
    dropped: list[dict[str, Any]] = []
    if len(missing_ts) > 0:
        miss = missing_ts.reset_index(drop=True)
        head = out["timestamp"].head(len(miss)).reset_index(drop=True)
        tail = out["timestamp"].tail(len(miss)).reset_index(drop=True)
        if miss.equals(head):
            edge = "head"
        elif miss.equals(tail):
            edge = "tail"
        else:
            raise RuntimeError(f"{tag}: non-edge missing timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"edge": edge, "rows": int(len(miss)), "first": str(miss.iloc[0]), "last": str(miss.iloc[-1]), "path": str(source)})
        out = out.loc[pd.to_datetime(out["timestamp"], errors="raise").isin(src_ts)].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after overlay")
    nan_mask = out[cols].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        raise RuntimeError(f"{tag}: non-edge NaN timestamps: {out.loc[nan_mask, 'timestamp'].head(20).tolist()}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "path": str(source), "reason": "edge_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    return out, {"path": str(source), "cols": list(cols), "dropped_edge_rows": dropped}


def _load_omega_frames() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    train, train_current = _overlay_required(train, REGIME3_CURRENT_2025, REGIME3_CURRENT_COLS, tag="train_regime3_current")
    eval_df, eval_current = _overlay_required(eval_df, REGIME3_CURRENT_2026, REGIME3_CURRENT_COLS, tag="eval_regime3_current")
    train, train_cmamba = _overlay_required(train, REGIME3_CMAMBA_2025, REGIME3_CMAMBA_COLS, tag="train_regime3_cmamba")
    eval_df, eval_cmamba = _overlay_required(eval_df, REGIME3_CMAMBA_2026, REGIME3_CMAMBA_COLS, tag="eval_regime3_cmamba")
    train, train_risk = _overlay_required(train, REGIME3_RISK_2025, REGIME3_RISK_COLS, tag="train_regime3_risk")
    eval_df, eval_risk = _overlay_required(eval_df, REGIME3_RISK_2026, REGIME3_RISK_COLS, tag="eval_regime3_risk")
    return train, eval_df, {
        "train_current": train_current,
        "eval_current": eval_current,
        "train_cmamba": train_cmamba,
        "eval_cmamba": eval_cmamba,
        "train_risk": train_risk,
        "eval_risk": eval_risk,
    }


def _require_unique_timestamps(df: pd.DataFrame, name: str) -> None:
    if "timestamp" not in df.columns:
        raise RuntimeError(f"{name}: missing timestamp")
    ts = pd.to_datetime(df["timestamp"], errors="raise")
    dup = ts.duplicated()
    if bool(dup.any()):
        raise RuntimeError(f"{name}: duplicate timestamps: {df.loc[dup, 'timestamp'].head(10).tolist()}")


def _align(frame: pd.DataFrame, src: pd.DataFrame, name: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_unique_timestamps(frame, f"{name}_frame")
    _require_unique_timestamps(src, f"{name}_source")
    frame_ts = pd.to_datetime(frame["timestamp"], errors="raise")
    src_ts = pd.to_datetime(src["timestamp"], errors="raise")
    lookup = pd.Series(np.arange(len(src), dtype=np.int64), index=src_ts)
    mask = frame_ts.isin(set(src_ts))
    out_frame = frame.loc[mask].reset_index(drop=True)
    if len(out_frame) == 0:
        raise RuntimeError(f"{name}: empty timestamp intersection")
    idx = lookup.loc[pd.to_datetime(out_frame["timestamp"], errors="raise")].to_numpy(dtype=np.int64)
    out_src = src.iloc[idx].reset_index(drop=True)
    if not out_frame["timestamp"].astype(str).reset_index(drop=True).equals(out_src["timestamp"].astype(str).reset_index(drop=True)):
        raise RuntimeError(f"{name}: timestamp order mismatch")
    return out_frame, out_src


def _forbidden_feature(name: str) -> bool:
    low = name.lower()
    return name.startswith(DENY_PREFIXES) or any(tok in low for tok in DENY_TOKENS)


def _numeric_feature_cols(train: pd.DataFrame, eval_df: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in train.columns:
        if col in NON_FEATURE_COLS or col not in eval_df.columns:
            continue
        if _forbidden_feature(str(col)):
            continue
        if pd.api.types.is_numeric_dtype(train[col]) and pd.api.types.is_numeric_dtype(eval_df[col]):
            cols.append(str(col))
    bad = [c for c in cols if _forbidden_feature(c)]
    if bad:
        raise RuntimeError(f"forbidden feature columns passed audit: {bad[:40]}")
    if len(cols) < 80:
        raise RuntimeError(f"unexpectedly small feature set: {len(cols)}")
    return cols


def _tabm_prefix(oof: bool) -> str:
    return "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"


def _source_state(src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = _tabm_prefix(oof)
    required = [
        "router_expert",
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "dir_action",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
        "quality_threshold",
        "final_action",
    ]
    missing = [f"{prefix}{c}" for c in required if f"{prefix}{c}" not in src.columns]
    if missing:
        raise RuntimeError(f"missing Omega1.2 TabM source columns: {missing}")
    out = pd.DataFrame(index=src.index)
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for name in ("bull", "bear", "chop_expert"):
        out[f"tabm_router_{name}"] = (expert == name).astype(float).to_numpy()
    for suffix in required[1:]:
        out[f"tabm_{suffix}"] = pd.to_numeric(src[f"{prefix}{suffix}"], errors="raise").to_numpy(dtype=np.float64)
    out["tabm_long_quality_edge"] = out["tabm_quality_p_long"] - out["tabm_quality_p_cash"]
    out["tabm_short_quality_edge"] = out["tabm_quality_p_short"] - out["tabm_quality_p_cash"]
    out["tabm_abs_side_edge"] = np.abs(out["tabm_dir_side_edge"].to_numpy(dtype=np.float64))
    return out


def _to_fixed_decisions(src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = _tabm_prefix(oof)
    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({ACTION_CASH, ACTION_LONG, ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    active = action != ACTION_CASH
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    router = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    dec = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, float(BASE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(BASE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(BASE_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(BASE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(BASE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(BASE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(BASE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(src[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "router_expert": router.to_numpy(),
        }
    )
    return _apply_expert_scale(dec)


def _active(dec: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != ACTION_CASH
    ) & (
        pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64) != 0
    ) & (
        pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64) > 0
    )


def _apply_expert_scale(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    for expert, scale in EXPERT_SCALES.items():
        mask = active & out["router_expert"].astype(str).eq(expert)
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * float(scale)
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * float(scale)
    out["expert_scale_bull"] = float(EXPERT_SCALES["bull"])
    out["expert_scale_bear"] = float(EXPERT_SCALES["bear"])
    out["expert_scale_chop"] = float(EXPERT_SCALES["chop_expert"])
    return out


def _build_state_frame(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, *, oof: bool, feature_cols: list[str]) -> pd.DataFrame:
    base = frame.reindex(columns=feature_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    d = pd.DataFrame(index=dec.index)
    for col in ("action", "side", "quality_score", "confidence", "notional_exposure", "leverage", "take_profit", "stop_loss"):
        d[f"fixed_{col}"] = pd.to_numeric(dec[col], errors="raise").to_numpy(dtype=np.float64)
    d["fixed_rr"] = d["fixed_take_profit"] / np.maximum(np.abs(d["fixed_stop_loss"]), 1e-8)
    out = pd.concat([base.reset_index(drop=True), _source_state(src, oof=oof).reset_index(drop=True), d.reset_index(drop=True)], axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate state columns: {dup[:20]}")
    return out


def _fit_norm(df: pd.DataFrame) -> dict[str, Any]:
    x = df.to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    return {"columns": list(df.columns), "mean": mean, "std": std}


def _apply_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"state frame missing normalized columns: {missing[:20]}")
    x = df.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num((x - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _risk_to_unit(risk: np.ndarray) -> np.ndarray:
    out = np.empty_like(risk, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = RISK_BOUNDS[col]
        out[:, j] = (np.clip(risk[:, j], lo, hi) - lo) / (hi - lo)
    return out


def _unit_to_risk(unit: np.ndarray) -> np.ndarray:
    u = np.clip(unit, 0.0, 1.0)
    out = np.empty_like(u, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = RISK_BOUNDS[col]
        out[:, j] = lo + u[:, j] * (hi - lo)
    return out


def _risk_to_model(risk: np.ndarray) -> np.ndarray:
    return (_risk_to_unit(risk) * 2.0 - 1.0).astype(np.float32)


def _model_to_risk(x: np.ndarray) -> np.ndarray:
    return _unit_to_risk((np.clip(x, -1.0, 1.0) + 1.0) * 0.5)


def _sample_risks(rng: np.random.Generator, count: int) -> np.ndarray:
    u = rng.random((count, 4), dtype=np.float32)
    # Bias part of the sample cloud around the current Omega1.2 fixed template.
    n_anchor = min(count, max(4, count // 4))
    anchor = np.asarray([[BASE_TEMPLATE["take_profit"], BASE_TEMPLATE["stop_loss"], BASE_TEMPLATE["leverage"], BASE_TEMPLATE["notional"]]], dtype=np.float32)
    noise = rng.normal(0.0, [0.006, 0.005, 0.55, 0.12], size=(n_anchor, 4)).astype(np.float32)
    anchored = anchor + noise
    u[:n_anchor] = _risk_to_unit(anchored)
    return _unit_to_risk(u)


def _set_risk_bounds(preset: str) -> None:
    if preset not in RISK_BOUND_PRESETS:
        raise RuntimeError(f"unknown risk bounds preset: {preset}")
    RISK_BOUNDS.clear()
    RISK_BOUNDS.update({k: (float(v[0]), float(v[1])) for k, v in RISK_BOUND_PRESETS[preset].items()})


def _fill_price(arrays: dict[str, np.ndarray], idx: int, side: int, slip_eff: float, *, entry: bool) -> float:
    px = float(arrays["open"][int(np.clip(idx, 0, len(arrays["open"]) - 1))])
    if side > 0:
        return px * (1.0 + slip_eff if entry else 1.0 - slip_eff)
    return px * (1.0 - slip_eff if entry else 1.0 + slip_eff)


def _close_fallback_price(arrays: dict[str, np.ndarray], idx: int, side: int, slip_eff: float, *, entry: bool) -> float:
    px = float(arrays["close"][int(np.clip(idx, 0, len(arrays["close"]) - 1))])
    if side > 0:
        return px * (1.0 + slip_eff if entry else 1.0 - slip_eff)
    return px * (1.0 - slip_eff if entry else 1.0 + slip_eff)


def _limit_price(arrays: dict[str, np.ndarray], signal_i: int, side: int, *, entry: bool) -> float:
    anchor_i = int(np.clip(int(signal_i) + 1, 0, len(arrays["open"]) - 1))
    px = float(arrays["open"][anchor_i])
    if not np.isfinite(px) or px <= 0.0:
        return 0.0
    return px


def _limit_touched(arrays: dict[str, np.ndarray], fill_i: int, price: float, side: int, *, entry: bool) -> bool:
    fill_i = int(np.clip(fill_i, 0, len(arrays["open"]) - 1))
    high = float(arrays["high"][fill_i])
    low = float(arrays["low"][fill_i])
    is_buy = (side > 0 and entry) or (side < 0 and not entry)
    if is_buy:
        return bool(low <= price)
    return bool(high >= price)


def _try_execution(
    arrays: dict[str, np.ndarray],
    signal_i: int,
    side: int,
    *,
    entry: bool,
    fee_base: float,
    slip_base: float,
) -> tuple[bool, float, float, str]:
    fill_i = min(int(signal_i) + 1, len(arrays["open"]) - 1)
    limit_px = _limit_price(arrays, signal_i, side, entry=entry)
    if limit_px > 0.0 and _limit_touched(arrays, fill_i, limit_px, side, entry=entry):
        return True, float(limit_px), float(fee_base * MAKER_FEE_MULT), "signal_immediate_maker_limit"
    if entry:
        return False, 0.0, 0.0, "signal_immediate_limit_miss"
    return True, float(_close_fallback_price(arrays, fill_i, side, slip_base, entry=False)), float(fee_base), "exit_market_fallback_after_limit_miss_close"


def _simulate_trade(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    i: int,
    dec_row: pd.Series,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, dict[str, Any]]:
    action = int(dec_row.get("action", 0) or 0)
    side = int(dec_row.get("side", 0) or 0)
    notional = float(dec_row.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "exit_i": int(i), "net": 0.0}
    signal_i = int(i)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    filled, entry, entry_fee, entry_route = _try_execution(arrays, signal_i, side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return 0.0, {"active": 0, "exit_i": int(i), "net": 0.0}
    entry_i = min(signal_i + 1, len(frame) - 1)
    tp = max(float(dec_row.get("take_profit", 0.0) or 0.0), 1e-8)
    sl = max(abs(float(dec_row.get("stop_loss", 0.0) or 0.0)), 1e-8)
    hold = int(dec_row.get("max_hold_bars", 0) or 0)
    end_i = min(signal_i + hold, len(frame) - 2) if hold > 0 else len(frame) - 2
    entry_equity = 1.0
    cash = entry_equity - entry_equity * entry_fee * notional
    exit_fill: float | None = None
    exit_fee = fee_eff
    exit_reason = "max_hold" if hold > 0 else "forced_end"
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, end_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip_eff)) / max(entry, 1e-12)
        unreal = raw * notional
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        bar_hold = int(j) - signal_i
        if unreal <= -abs(sl):
            _, exit_fill, exit_fee, _ = _try_execution(arrays, int(j), side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "stop_loss"
            end_i = j
            break
        if unreal >= tp:
            _, exit_fill, exit_fee, _ = _try_execution(arrays, int(j), side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "take_profit"
            end_i = j
            break
        if hold > 0 and bar_hold >= hold:
            _, exit_fill, exit_fee, _ = _try_execution(arrays, int(j), side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "max_hold"
            end_i = j
            break
    if exit_fill is None:
        exit_fill = _fill_price(arrays, min(end_i + 1, len(frame) - 1), side, slip_eff, entry=False)
    raw_exit = (exit_fill - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_fill) / max(entry, 1e-12)
    before_exit_fee = cash
    cash = cash * (1.0 + raw_exit * notional)
    cash -= before_exit_fee * exit_fee * notional
    net = float(cash - 1.0)
    exposure = float(notional) * float(dec_row.get("leverage", 1.0) or 1.0)
    score = net - 0.20 * max(0.0, -mae - 0.018) * exposure - 0.015 * max(0.0, exposure - 2.0)
    return score, {"active": 1, "exit_i": int(end_i), "net": net, "win": int(cash > entry_equity), "exit_reason": exit_reason, "entry_route": entry_route, "mfe": float(mfe), "mae": float(mae)}


def _expert_scale_for_row(row: pd.Series) -> float:
    return float(EXPERT_SCALES.get(str(row.get("router_expert", "")), 1.0))


def _with_risk(row: pd.Series, risk: np.ndarray, *, apply_expert_scale: bool = False) -> pd.Series:
    out = row.copy()
    notional = float(risk[3])
    if apply_expert_scale:
        notional *= _expert_scale_for_row(row)
    out.loc["take_profit"] = float(risk[0])
    out.loc["stop_loss"] = float(risk[1])
    out.loc["leverage"] = float(risk[2])
    out.loc["notional_exposure"] = float(notional)
    out.loc["position_fraction"] = float(notional)
    out.loc["max_hold_bars"] = int(BASE_TEMPLATE["max_hold"])
    out.loc["cooldown_bars"] = int(BASE_TEMPLATE["cooldown"])
    return out


def _apply_generated_risk(base_dec: pd.DataFrame, risk: np.ndarray) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    active = _active(out)
    for idx in np.flatnonzero(active):
        out.iloc[int(idx)] = _with_risk(out.iloc[int(idx)], risk[int(idx)])
    # Generated notional is base notional; expert scale remains a separate unchanged layer.
    return _apply_expert_scale(out)


def _metrics(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = _active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_idx = 0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    max_hold = 0
    cooldown = 0
    next_cooldown = 0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold = int(i) - int(entry_idx)
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "max_hold"
            if reason:
                filled, exit_px, exit_fee, _route = _try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = _try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = px
        entry_equity = cash
        entry_idx = int(i)
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        max_hold = int(row.get("max_hold_bars", 0) or 0)
        next_cooldown = int(row.get("cooldown_bars", 0) or 0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = _fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1e-9)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


@dataclass
class DiffusionDataset:
    states: np.ndarray
    risks: np.ndarray
    weights: np.ndarray
    scorer_states: np.ndarray
    scorer_risks: np.ndarray
    scorer_rewards: np.ndarray


def _build_diffusion_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    dec: pd.DataFrame,
    *,
    seed: int,
    samples_per_row: int,
    keep_top_k: int,
    max_active_rows: int,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[DiffusionDataset, dict[str, Any]]:
    active_idx = np.flatnonzero(_active(dec) & (np.arange(len(dec)) < len(dec) - 3))
    rng = np.random.default_rng(int(seed))
    total_active = int(len(active_idx))
    if max_active_rows > 0 and len(active_idx) > max_active_rows:
        active_idx = np.sort(rng.choice(active_idx, size=int(max_active_rows), replace=False))
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    x_good: list[np.ndarray] = []
    y_good: list[np.ndarray] = []
    w_good: list[float] = []
    x_all: list[np.ndarray] = []
    r_all: list[np.ndarray] = []
    reward_all: list[float] = []
    reason_counts: dict[str, int] = {}
    best_rewards: list[float] = []
    for i in active_idx:
        risks = _sample_risks(rng, int(samples_per_row))
        rewards = np.empty(len(risks), dtype=np.float32)
        for j, rv in enumerate(risks):
            dec_row = _with_risk(dec.iloc[int(i)], rv, apply_expert_scale=True)
            score, meta = _simulate_trade(frame, arrays, int(i), dec_row, fee=fee, slip=slip, cost_mult=cost_mult)
            rewards[j] = float(score)
            reason = str(meta.get("exit_reason", "inactive"))
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        order = np.argsort(rewards)[::-1]
        top = order[: max(1, int(keep_top_k))]
        best_rewards.append(float(rewards[order[0]]))
        scale = float(np.std(rewards))
        if not np.isfinite(scale) or scale < 1e-6:
            scale = 1.0
        local_w = np.exp(np.clip((rewards[top] - float(np.median(rewards))) / scale, -4.0, 4.0))
        for k, j in enumerate(top):
            x_good.append(states[int(i)])
            y_good.append(_risk_to_model(risks[j : j + 1])[0])
            w_good.append(float(local_w[k]))
        # Keep all sampled points for the rerank scorer.
        x_all.extend([states[int(i)]] * len(risks))
        r_all.extend(_risk_to_model(risks))
        reward_all.extend(rewards.tolist())
    if not x_good:
        raise RuntimeError("no active training samples for diffusion")
    rewards_np = np.asarray(reward_all, dtype=np.float32)
    r_mean = float(np.nanmean(rewards_np))
    r_std = float(np.nanstd(rewards_np))
    if not np.isfinite(r_std) or r_std < 1e-6:
        r_std = 1.0
    diagnostics = {
        "total_active_rows": total_active,
        "used_active_rows": int(len(active_idx)),
        "samples_per_row": int(samples_per_row),
        "keep_top_k": int(keep_top_k),
        "diffusion_samples": int(len(x_good)),
        "scorer_samples": int(len(x_all)),
        "best_reward_mean": float(np.mean(best_rewards)) if best_rewards else 0.0,
        "reward_mean": r_mean,
        "reward_std": r_std,
        "counterfactual_exit_reasons": reason_counts,
    }
    return (
        DiffusionDataset(
            states=np.asarray(x_good, dtype=np.float32),
            risks=np.asarray(y_good, dtype=np.float32),
            weights=np.asarray(w_good, dtype=np.float32),
            scorer_states=np.asarray(x_all, dtype=np.float32),
            scorer_risks=np.asarray(r_all, dtype=np.float32),
            scorer_rewards=np.clip((rewards_np - r_mean) / r_std, -8.0, 8.0).astype(np.float32),
        ),
        diagnostics,
    )


class SinusoidalTime(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freq = torch.exp(torch.arange(half, device=t.device, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        x = t.float().unsqueeze(1) * freq.unsqueeze(0)
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=1)
        if emb.shape[1] < self.dim:
            emb = torch.cat([emb, torch.zeros(len(t), 1, device=t.device)], dim=1)
        return emb


class DiffusionRiskPolicy(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, t_dim: int = 64) -> None:
        super().__init__()
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.time = SinusoidalTime(t_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden + t_dim + 4, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, state: torch.Tensor, noisy_risk: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([self.state(state), self.time(t), noisy_risk], dim=1))


class RiskScorer(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 192) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 4, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, risk_model: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, risk_model], dim=1)).squeeze(1)


def _cosine_beta_schedule(steps: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(0, steps, steps + 1, dtype=np.float64)
    ac = np.cos(((x / steps) + 0.008) / 1.008 * math.pi * 0.5) ** 2
    ac = ac / ac[0]
    betas = 1.0 - (ac[1:] / ac[:-1])
    betas = np.clip(betas, 1e-5, 0.05)
    alpha = 1.0 - betas
    alpha_bar = np.cumprod(alpha)
    return betas.astype(np.float32), alpha_bar.astype(np.float32)


def _train_diffusion(data: DiffusionDataset, *, state_dim: int, device: torch.device, steps: int, batch_size: int, lr: float, diffusion_steps: int) -> tuple[DiffusionRiskPolicy, dict[str, Any], np.ndarray]:
    model = DiffusionRiskPolicy(state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    _, alpha_bar = _cosine_beta_schedule(diffusion_steps)
    ab = torch.from_numpy(alpha_bar).to(device)
    ds = TensorDataset(torch.from_numpy(data.states), torch.from_numpy(data.risks), torch.from_numpy(data.weights))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            s, y, w = next(it)
        except StopIteration:
            it = iter(dl)
            s, y, w = next(it)
        s = s.to(device)
        y = y.to(device)
        w = w.to(device)
        t = torch.randint(0, int(diffusion_steps), (len(s),), device=device)
        noise = torch.randn_like(y)
        a = ab[t].view(-1, 1)
        noisy = torch.sqrt(a) * y + torch.sqrt(1.0 - a) * noise
        pred = model(s, noisy, t)
        loss = (((pred - noise) ** 2).mean(dim=1) * w).sum() / torch.clamp(w.sum(), min=1.0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            last = {"step": int(step), "diffusion_loss": float(loss.detach().cpu())}
        if step % 1000 == 0:
            print(json.dumps({"stage": "diffusion_progress", **last}, ensure_ascii=False), flush=True)
    return model.cpu(), last, alpha_bar


def _train_scorer(data: DiffusionDataset, *, state_dim: int, device: torch.device, steps: int, batch_size: int, lr: float) -> tuple[RiskScorer, dict[str, Any]]:
    model = RiskScorer(state_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.scorer_states), torch.from_numpy(data.scorer_risks), torch.from_numpy(data.scorer_rewards))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, max(1, int(steps)) + 1):
        try:
            s, r, y = next(it)
        except StopIteration:
            it = iter(dl)
            s, r, y = next(it)
        s = s.to(device)
        r = r.to(device)
        y = y.to(device)
        pred = model(s, r)
        loss = torch.nn.functional.smooth_l1_loss(pred, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            last = {"step": int(step), "scorer_loss": float(loss.detach().cpu())}
    return model.cpu(), last


@torch.no_grad()
def _sample_policy(model: DiffusionRiskPolicy, states: np.ndarray, *, device: torch.device, samples: int, diffusion_steps: int, alpha_bar: np.ndarray) -> np.ndarray:
    model = model.to(device)
    model.eval()
    ab = torch.from_numpy(alpha_bar).to(device)
    out: list[np.ndarray] = []
    for start in range(0, len(states), 2048):
        s0 = torch.from_numpy(states[start : start + 2048]).to(device)
        s = s0.repeat_interleave(int(samples), dim=0)
        x = torch.randn((len(s), 4), device=device)
        for ti in range(int(diffusion_steps) - 1, -1, -1):
            t = torch.full((len(s),), ti, device=device, dtype=torch.long)
            eps = model(s, x, t)
            a = ab[ti]
            pred_x0 = (x - torch.sqrt(1.0 - a) * eps) / torch.sqrt(a)
            if ti > 0:
                a_prev = ab[ti - 1]
                x = torch.sqrt(a_prev) * pred_x0 + torch.sqrt(1.0 - a_prev) * eps
            else:
                x = pred_x0
        out.append(x.clamp(-1.0, 1.0).cpu().numpy().reshape(len(s0), int(samples), 4))
    return np.concatenate(out, axis=0) if out else np.zeros((0, int(samples), 4), dtype=np.float32)


@torch.no_grad()
def _rerank(scorer: RiskScorer, states: np.ndarray, candidates: np.ndarray, *, device: torch.device, exposure_penalty: float) -> np.ndarray:
    scorer = scorer.to(device)
    scorer.eval()
    chosen: list[np.ndarray] = []
    for start in range(0, len(states), 1024):
        s0 = torch.from_numpy(states[start : start + 1024]).to(device)
        c0 = torch.from_numpy(candidates[start : start + 1024]).to(device)
        b, k, _ = c0.shape
        scores = scorer(s0.repeat_interleave(k, dim=0), c0.reshape(b * k, 4)).reshape(b, k)
        if exposure_penalty > 0.0:
            risks = torch.from_numpy(_model_to_risk(c0.cpu().numpy().reshape(b * k, 4))).to(device).reshape(b, k, 4)
            exposure = risks[:, :, 2] * risks[:, :, 3]
            scores = scores - float(exposure_penalty) * exposure
        idx = torch.argmax(scores, dim=1)
        chosen.append(c0[torch.arange(b, device=device), idx].cpu().numpy())
    return np.concatenate(chosen, axis=0) if chosen else np.zeros((0, 4), dtype=np.float32)


def _risk_distribution(dec: pd.DataFrame) -> dict[str, Any]:
    active = _active(dec)
    out: dict[str, Any] = {"active_rows": int(active.sum())}
    if not bool(active.any()):
        return out
    for col in ("take_profit", "stop_loss", "leverage", "notional_exposure"):
        arr = pd.to_numeric(dec.loc[active, col], errors="raise").to_numpy(dtype=np.float64)
        out[col] = {
            "mean": float(np.mean(arr)),
            "p10": float(np.quantile(arr, 0.10)),
            "p50": float(np.quantile(arr, 0.50)),
            "p90": float(np.quantile(arr, 0.90)),
            "max": float(np.max(arr)),
        }
    return out


def _load_fee_slip() -> tuple[float, float]:
    return float(FEE_RATE), float(SLIP_RATE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3500)
    ap.add_argument("--scorer-steps", type=int, default=1600)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--diffusion-steps", type=int, default=24)
    ap.add_argument("--samples-per-row", type=int, default=64)
    ap.add_argument("--keep-top-k", type=int, default=8)
    ap.add_argument("--rerank-samples", type=int, default=32)
    ap.add_argument("--max-active-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--risk-bounds-preset", choices=sorted(RISK_BOUND_PRESETS), default="absolute")
    ap.add_argument("--rerank-exposure-penalty", type=float, default=0.0)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--seed", type=int, default=260603)
    ap.add_argument("--disable-hold-cooldown", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    _set_risk_bounds(str(args.risk_bounds_preset))
    if bool(args.disable_hold_cooldown):
        BASE_TEMPLATE["max_hold"] = 0
        BASE_TEMPLATE["cooldown"] = 0
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    train_all, eval_df, overlay_report = _load_omega_frames()
    tabm_2025 = _read(TABM_2025)
    tabm_2026 = _read(TABM_2026)
    feature_cols = _numeric_feature_cols(train_all, eval_df)

    train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    train_df, train_src = _align(train_raw, tabm_2025, "train")
    val_df, val_src = _align(val_raw, tabm_2025, "validation")
    oos_df, oos_src = _align(eval_df, tabm_2026, "oos")

    train_fixed = _to_fixed_decisions(train_src, oof=True)
    val_fixed = _to_fixed_decisions(val_src, oof=True)
    oos_fixed = _to_fixed_decisions(oos_src, oof=False)

    s_train = _build_state_frame(train_df, train_src, train_fixed, oof=True, feature_cols=feature_cols)
    s_val = _build_state_frame(val_df, val_src, val_fixed, oof=True, feature_cols=feature_cols)
    s_oos = _build_state_frame(oos_df, oos_src, oos_fixed, oof=False, feature_cols=feature_cols)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_oos = _apply_norm(s_oos, norm)

    fee, slip = _load_fee_slip()
    data, data_diag = _build_diffusion_dataset(
        train_df,
        x_train,
        train_fixed,
        seed=int(args.seed),
        samples_per_row=int(args.samples_per_row),
        keep_top_k=int(args.keep_top_k),
        max_active_rows=int(args.max_active_rows),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "feature_count": int(len(feature_cols)),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(oos_df)),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    policy, policy_diag, alpha_bar = _train_diffusion(
        data,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        diffusion_steps=int(args.diffusion_steps),
    )
    scorer, scorer_diag = _train_scorer(
        data,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.scorer_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )

    val_candidates = _sample_policy(policy, x_val, device=device, samples=max(1, int(args.rerank_samples)), diffusion_steps=int(args.diffusion_steps), alpha_bar=alpha_bar)
    oos_candidates = _sample_policy(policy, x_oos, device=device, samples=max(1, int(args.rerank_samples)), diffusion_steps=int(args.diffusion_steps), alpha_bar=alpha_bar)
    val_direct_risk = _model_to_risk(val_candidates[:, 0, :])
    oos_direct_risk = _model_to_risk(oos_candidates[:, 0, :])
    val_rerank_risk = _model_to_risk(_rerank(scorer, x_val, val_candidates, device=device, exposure_penalty=float(args.rerank_exposure_penalty)))
    oos_rerank_risk = _model_to_risk(_rerank(scorer, x_oos, oos_candidates, device=device, exposure_penalty=float(args.rerank_exposure_penalty)))

    decisions = {
        "fixed_template": (val_fixed, oos_fixed),
        "diffusion_direct": (_apply_generated_risk(val_fixed, val_direct_risk), _apply_generated_risk(oos_fixed, oos_direct_risk)),
        "diffusion_sample_rerank": (_apply_generated_risk(val_fixed, val_rerank_risk), _apply_generated_risk(oos_fixed, oos_rerank_risk)),
    }
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for name, (vdec, odec) in decisions.items():
        val_cost = _metrics(val_df, vdec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_cost = _metrics(oos_df, odec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        reports[name] = {
            "validation": val_cost,
            "oos": oos_cost,
            "validation_risk_distribution": _risk_distribution(vdec),
            "oos_risk_distribution": _risk_distribution(odec),
        }
        rows.append({"variant": name, "split": "validation", **{k: v for k, v in val_cost.items() if k != "exit_reasons"}})
        rows.append({"variant": name, "split": "oos", **{k: v for k, v in oos_cost.items() if k != "exit_reasons"}})
    ranking = pd.DataFrame(rows)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    decisions["diffusion_direct"][0].to_csv(out_dir / "validation_decisions_diffusion_direct.csv", index=False)
    decisions["diffusion_direct"][1].to_csv(out_dir / "oos_2026_decisions_diffusion_direct.csv", index=False)
    decisions["diffusion_sample_rerank"][0].to_csv(out_dir / "validation_decisions_diffusion_sample_rerank.csv", index=False)
    decisions["diffusion_sample_rerank"][1].to_csv(out_dir / "oos_2026_decisions_diffusion_sample_rerank.csv", index=False)

    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dim": int(x_train.shape[1]),
            "state_columns": list(norm["columns"]),
            "normalizer": norm,
            "risk_cols": RISK_COLS,
            "risk_bounds": RISK_BOUNDS,
            "diffusion_steps": int(args.diffusion_steps),
            "alpha_bar": alpha_bar,
            "policy_state_dict": policy.state_dict(),
            "scorer_state_dict": scorer.state_dict(),
        },
        out_dir / "diffusion_risk_policy.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Omega1.2 soft_floor_0p00 TabM ExpertDQ is frozen. Only fixed TP/SL/leverage/notional template is replaced by reward-weighted conditional diffusion. Max-hold/cooldown are disabled only when --disable-hold-cooldown is set; expert scale layer remains unchanged.",
        "tabm_source": {
            "train_oof": str(TABM_2025),
            "oos": str(TABM_2026),
            "variant": "soft_floor_0p00",
            "tabm_frozen": True,
        },
        "frame_overlay": overlay_report,
        "forbidden_feature_policy": {"deny_prefixes": DENY_PREFIXES, "deny_tokens": DENY_TOKENS, "feature_count": len(feature_cols)},
        "risk_replacement": {
            "replaced": RISK_COLS,
            "unchanged": ["expert_scale_layer"],
            "max_hold_bars": int(BASE_TEMPLATE["max_hold"]),
            "cooldown_bars": int(BASE_TEMPLATE["cooldown"]),
            "hold_cooldown_disabled": bool(args.disable_hold_cooldown),
            "bounds": RISK_BOUNDS,
        },
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult), "entry_exit_notional_fee": True},
        "training": {
            "device": str(device),
            "steps": int(args.steps),
            "scorer_steps": int(args.scorer_steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "diffusion_steps": int(args.diffusion_steps),
            "rerank_samples": int(args.rerank_samples),
            "risk_bounds_preset": str(args.risk_bounds_preset),
            "rerank_exposure_penalty": float(args.rerank_exposure_penalty),
            "disable_hold_cooldown": bool(args.disable_hold_cooldown),
            "data_diag": data_diag,
            "policy_diag": policy_diag,
            "scorer_diag": scorer_diag,
        },
        "results": reports,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "model": str(out_dir / "diffusion_risk_policy.pt"),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": reports}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
