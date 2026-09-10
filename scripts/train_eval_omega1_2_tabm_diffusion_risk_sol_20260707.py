#!/usr/bin/env python3
"""SOL copy of the shared `omega` utility module
(train_eval_omega1_2_tabm_diffusion_risk_20260603.py), imported by the SOL
parent trainer as `omega`.

Changes vs the ETH original:
- TRAIN_CSV/EVAL_CSV point at SOL's own FeatureEngineer output
  (data/splits/year_oos/sol_features_{2025,2026}.csv), not the ETH
  alpha6/alpha7 accreted trade-candidate lineage.
- REGIME3_CURRENT_2025/2026 point at SOL's own wide24 HMM overlay.
- The TABM_2025/2026 (frozen Omega1.2 regime3-routed expert direction/quality
  TabM predictions) and the cmamba/stability-risk regime3 overlays are
  DROPPED from _load_omega_frames(). Verified empirically against ETH's own
  promoted h48qual/zig075 report.json: both use
  exit_label.mode == "entry_label_terminal_giveback", which builds exit-head
  training examples directly from zigzag_action segments and never touches
  the TABM-derived `train_fixed`/`train_df` alignment. The cmamba/risk
  overlays are also confirmed not to survive into ETH's own trained
  `base_cols` (102 = 96 FeatureEngineer + 6 regime3-current-wide24 only), so
  building SOL equivalents of those two extra regime3 sidecars would be
  wasted work. This keeps the SOL build faithful to the actual production
  recipe rather than the diffusion-risk-policy module's own unrelated
  standalone experiment (this file's own `main()`/diffusion-training code is
  unused by the parent trainer and is kept only for import compatibility).
"""
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
MODEL_ID = "sol_omega1_2_softfloor00_tabm_diffusion_risk_20260707"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TRAIN_CSV = ROOT / "data/splits/year_oos/sol_features_2025.csv"
EVAL_CSV = ROOT / "data/splits/year_oos/sol_features_2026.csv"
REGIME3_CURRENT_2025 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_2025_regime3_current_sensitive_hmm_wide24.csv"
REGIME3_CURRENT_2026 = ROOT / "data/ensemble/supervised/sol_regime3_current_hmm_sensitive_wide24_20260707/sol_features_2026_regime3_current_sensitive_hmm_wide24.csv"
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
    return train, eval_df, {
        "train_current": train_current,
        "eval_current": eval_current,
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


def _active(dec: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != ACTION_CASH
    ) & (
        pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64) != 0
    ) & (
        pd.to_numeric(dec["notional_exposure"], errors="raise").to_numpy(dtype=np.float64) > 0
    )


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


def _load_fee_slip() -> tuple[float, float]:
    return float(FEE_RATE), float(SLIP_RATE)


def main() -> int:  # pragma: no cover - unused by the SOL parent trainer; kept for import parity only
    raise SystemExit("this SOL omega module is a library only; the diffusion-policy standalone experiment was not ported")


if __name__ == "__main__":
    raise SystemExit(main())
