#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _json_default  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import (  # noqa: E402
    ACTIVE_SCALES,
    ACTIVE_TEMPLATE,
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    EXPERTDQ_DIR,
    _align_source_to_frame,
    _expertdq_paths,
    _load_csv,
    _require_unique_timestamps,
    _to_decisions,
)  # noqa: E402
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import _load_frames_max  # noqa: E402


MODEL_ID = "omega1_expertdq_dsac_risk_allocator_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

NOTIONAL_BUCKETS = (0.45, 0.80)
LEVERAGE_BUCKETS = (2.0, 3.0)
TP_BUCKETS = (0.026, 0.050)
SL_BUCKETS = (0.014, 0.035)
HOLD_BUCKETS = (72, 96)
COOLDOWN_BUCKETS = (6,)

EXCLUDE_COL_SUBSTRINGS = (
    "future",
    "target",
    "label",
    "zigzag",
    "tp_sl_action_score",
    "wave3_action",
)
EXCLUDE_PREFIXES = (
    "teacher_",
    "teacher_oof_",
)


@dataclass(frozen=True)
class RiskSpec:
    flat_id: int
    veto: int
    notional_id: int
    leverage_id: int
    tp_id: int
    sl_id: int
    hold_id: int
    cooldown_id: int


def _build_action_space() -> tuple[list[RiskSpec], dict[tuple[int, int, int, int, int, int, int], int]]:
    specs = [RiskSpec(0, 1, 0, 0, 0, 0, 0, 0)]
    lookup = {(1, 0, 0, 0, 0, 0, 0): 0}
    flat = 1
    for n_i in range(len(NOTIONAL_BUCKETS)):
        for l_i in range(len(LEVERAGE_BUCKETS)):
            for tp_i in range(len(TP_BUCKETS)):
                for sl_i in range(len(SL_BUCKETS)):
                    for hold_i in range(len(HOLD_BUCKETS)):
                        for cd_i in range(len(COOLDOWN_BUCKETS)):
                            key = (0, n_i, l_i, tp_i, sl_i, hold_i, cd_i)
                            specs.append(RiskSpec(flat, *key))
                            lookup[key] = flat
                            flat += 1
    return specs, lookup


ACTION_SPECS, ACTION_LOOKUP = _build_action_space()
ACTION_DIM = len(ACTION_SPECS)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = _num(dec, "action").astype(np.int64)
    side = _num(dec, "side").astype(np.int64)
    notional = _num(dec, "notional_exposure")
    return (action != ACTION_CASH) & (side != 0) & (notional > 0.0)


def _numeric_feature_cols(frame: pd.DataFrame) -> list[str]:
    cols: list[str] = []
    for col in frame.columns:
        name = str(col)
        lname = name.lower()
        if name == "timestamp":
            continue
        if any(lname.startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        if any(s in lname for s in EXCLUDE_COL_SUBSTRINGS):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(name)
    return cols


def _source_state(src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    out = pd.DataFrame(index=src.index)
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for name in ("bull", "bear", "chop_expert"):
        out[f"expertdq_router_{name}"] = (expert == name).astype(float).to_numpy()
    for suffix in (
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
    ):
        col = f"{prefix}{suffix}"
        if col not in src.columns:
            raise RuntimeError(f"missing expert-DQ state column: {col}")
        out[f"expertdq_{suffix}"] = pd.to_numeric(src[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["expertdq_long_quality_edge"] = out["expertdq_quality_p_long"] - out["expertdq_quality_p_cash"]
    out["expertdq_short_quality_edge"] = out["expertdq_quality_p_short"] - out["expertdq_quality_p_cash"]
    out["expertdq_abs_side_edge"] = np.abs(out["expertdq_dir_side_edge"].to_numpy(dtype=np.float64))
    return out


def _decision_state(dec: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=dec.index)
    for col in (
        "action",
        "side",
        "notional_exposure",
        "leverage",
        "position_fraction",
        "take_profit",
        "stop_loss",
        "max_hold_bars",
        "cooldown_bars",
        "quality_score",
        "confidence",
    ):
        if col not in dec.columns:
            raise RuntimeError(f"decision state missing: {col}")
        out[f"decision_{col}"] = pd.to_numeric(dec[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["decision_side_x_quality"] = out["decision_side"] * out["decision_quality_score"]
    out["decision_side_x_confidence"] = out["decision_side"] * out["decision_confidence"]
    out["decision_rr"] = out["decision_take_profit"] / np.maximum(np.abs(out["decision_stop_loss"]), 1e-8)
    return out


def _build_state_frame(frame: pd.DataFrame, dec: pd.DataFrame, src: pd.DataFrame, *, oof: bool, feature_cols: list[str] | None = None) -> pd.DataFrame:
    cols = feature_cols or _numeric_feature_cols(frame)
    x_base = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    parts = [
        x_base.reset_index(drop=True),
        _source_state(src.reset_index(drop=True), oof=oof).reset_index(drop=True),
        _decision_state(dec.reset_index(drop=True)).reset_index(drop=True),
    ]
    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate Omega1 DSAC state columns: {dup[:20]}")
    return out


def _fit_norm(df: pd.DataFrame) -> dict[str, Any]:
    x = df.to_numpy(dtype=np.float32)
    mean = np.nanmean(x, axis=0)
    std = np.nanstd(x, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    return {"columns": list(df.columns), "mean": mean.astype(np.float32), "std": std.astype(np.float32)}


def _apply_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"state frame missing normalized columns: {missing[:20]}")
    x = df.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num((x - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _zero_row(row: pd.Series) -> pd.Series:
    out = row.copy()
    for col, value in (
        ("action", 0),
        ("side", 0),
        ("notional_exposure", 0.0),
        ("position_fraction", 0.0),
        ("take_profit", 0.0),
        ("stop_loss", 0.0),
        ("max_hold_bars", 0),
        ("cooldown_bars", 0),
    ):
        out.loc[col] = value
    out.loc["leverage"] = 1.0
    return out


def _apply_action(row: pd.Series, flat_id: int) -> pd.Series:
    spec = ACTION_SPECS[int(flat_id)]
    if spec.veto:
        return _zero_row(row)
    out = row.copy()
    notional = float(NOTIONAL_BUCKETS[spec.notional_id])
    leverage = float(LEVERAGE_BUCKETS[spec.leverage_id])
    out.loc["notional_exposure"] = notional
    out.loc["leverage"] = leverage
    out.loc["position_fraction"] = notional / max(leverage, 1e-8)
    out.loc["take_profit"] = float(TP_BUCKETS[spec.tp_id])
    out.loc["stop_loss"] = float(SL_BUCKETS[spec.sl_id])
    out.loc["max_hold_bars"] = int(HOLD_BUCKETS[spec.hold_id])
    out.loc["cooldown_bars"] = int(COOLDOWN_BUCKETS[spec.cooldown_id])
    return out


def _simulate_action(frame: pd.DataFrame, arrays: dict[str, np.ndarray], i: int, dec_row: pd.Series, flat_id: int, *, fee: float, slip: float, cost_mult: float) -> tuple[float, dict[str, Any]]:
    dec = _apply_action(dec_row, flat_id)
    action = int(dec.get("action", 0) or 0)
    side = int(dec.get("side", 0) or 0)
    notional = float(dec.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "exit_i": int(i)}
    entry_i = min(int(i) + 1, len(frame) - 1)
    entry_px = float(arrays["open"][entry_i])
    if entry_px <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "exit_i": int(i)}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    entry = entry_px * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    tp = float(dec.get("take_profit", 0.0) or 0.0)
    sl = abs(float(dec.get("stop_loss", 0.0) or 0.0))
    hold = max(int(dec.get("max_hold_bars", 0) or 0), 1)
    end_i = min(entry_i + hold, len(frame) - 1)
    exit_fill: float | None = None
    exit_reason = "hold"
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = float(arrays["high"][j]) / max(entry, 1e-12) - 1.0
            adverse = float(arrays["low"][j]) / max(entry, 1e-12) - 1.0
        else:
            favorable = entry / max(float(arrays["low"][j]), 1e-12) - 1.0
            adverse = entry / max(float(arrays["high"][j]), 1e-12) - 1.0
        if adverse <= -sl:
            if side > 0:
                trigger_px = entry * max(1.0 - sl, 1e-8)
                exit_fill = trigger_px * (1.0 - slip_eff)
            else:
                trigger_px = entry / max(1.0 - sl, 1e-8)
                exit_fill = trigger_px * (1.0 + slip_eff)
            exit_reason = "stop_loss"
            end_i = j
            break
        if favorable >= tp:
            if side > 0:
                trigger_px = entry * (1.0 + tp)
                exit_fill = trigger_px * (1.0 - slip_eff)
            else:
                trigger_px = entry / max(1.0 + tp, 1e-8)
                exit_fill = trigger_px * (1.0 + slip_eff)
            exit_reason = "take_profit"
            end_i = j
            break
    if exit_fill is None:
        exit_px = float(arrays["close"][end_i])
        exit_fill = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    entry_notional = float(notional)
    qty = entry_notional / max(entry, 1e-12)
    exit_notional = qty * max(float(exit_fill), 0.0)
    gross = exit_notional - entry_notional if side > 0 else entry_notional - exit_notional
    net = float(gross - fee_eff * entry_notional - fee_eff * exit_notional)
    return net, {"active": 1, "net": net, "win": int(net > 0.0), "exit_reason": exit_reason, "exit_i": int(end_i)}


def _fast_replay_metrics(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float = 3.0) -> dict[str, Any]:
    arrays = {k: _num(frame, k) for k in ("open", "high", "low", "close")}
    active = _active(dec)
    next_allowed = 0
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    wins = 0
    trades = 0
    for i in range(len(frame) - 3):
        if i < next_allowed or not bool(active[i]):
            continue
        reward, meta = _simulate_action(frame, arrays, i, dec.iloc[i], _row_to_action_id(dec.iloc[i]), fee=fee, slip=slip, cost_mult=cost_mult)
        if int(meta.get("active", 0)) != 1:
            continue
        trades += 1
        wins += int(reward > 0.0)
        equity += float(reward) * 100.0
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
        exit_i = int(meta.get("exit_i", i))
        cooldown = max(int(dec.iloc[i].get("cooldown_bars", 0) or 0), 0)
        next_allowed = max(i + 1, exit_i + cooldown)
    return {
        "pnl": float(equity),
        "mdd": float(mdd),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
    }


def _nearest_idx(value: float, buckets: tuple[float | int, ...]) -> int:
    arr = np.asarray(buckets, dtype=np.float64)
    return int(np.argmin(np.abs(arr - float(value))))


def _row_to_action_id(row: pd.Series) -> int:
    action = int(row.get("action", 0) or 0)
    side = int(row.get("side", 0) or 0)
    notional = float(row.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0
    key = (
        0,
        _nearest_idx(notional, NOTIONAL_BUCKETS),
        _nearest_idx(float(row.get("leverage", 1.0) or 1.0), LEVERAGE_BUCKETS),
        _nearest_idx(float(row.get("take_profit", 0.0) or 0.0), TP_BUCKETS),
        _nearest_idx(abs(float(row.get("stop_loss", 0.0) or 0.0)), SL_BUCKETS),
        _nearest_idx(int(row.get("max_hold_bars", 1) or 1), HOLD_BUCKETS),
        _nearest_idx(int(row.get("cooldown_bars", 0) or 0), COOLDOWN_BUCKETS),
    )
    return ACTION_LOOKUP[key]


def _flat_id_tensor(veto: torch.Tensor, n: torch.Tensor, lev: torch.Tensor, tp: torch.Tensor, sl: torch.Tensor, hold: torch.Tensor, cd: torch.Tensor) -> torch.Tensor:
    keep_id = 1 + (((((n * len(LEVERAGE_BUCKETS) + lev) * len(TP_BUCKETS) + tp) * len(SL_BUCKETS) + sl) * len(HOLD_BUCKETS) + hold) * len(COOLDOWN_BUCKETS) + cd)
    return torch.where(veto == 1, torch.zeros_like(keep_id), keep_id.long())


def _components_from_flat(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = flat.long()
    veto = torch.where(flat == 0, torch.ones_like(flat), torch.zeros_like(flat))
    idx = torch.clamp(flat - 1, min=0)
    cd = idx % len(COOLDOWN_BUCKETS)
    idx = idx // len(COOLDOWN_BUCKETS)
    hold = idx % len(HOLD_BUCKETS)
    idx = idx // len(HOLD_BUCKETS)
    sl = idx % len(SL_BUCKETS)
    idx = idx // len(SL_BUCKETS)
    tp = idx % len(TP_BUCKETS)
    idx = idx // len(TP_BUCKETS)
    lev = idx % len(LEVERAGE_BUCKETS)
    n = idx // len(LEVERAGE_BUCKETS)
    return veto, n, lev, tp, sl, hold, cd


class AutoRegRiskActor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.veto_head = nn.Linear(hidden, 2)
        self.notional_head = nn.Linear(hidden + 2, len(NOTIONAL_BUCKETS))
        self.leverage_head = nn.Linear(hidden + 2 + len(NOTIONAL_BUCKETS), len(LEVERAGE_BUCKETS))
        self.tp_head = nn.Linear(hidden + 2 + len(NOTIONAL_BUCKETS) + len(LEVERAGE_BUCKETS), len(TP_BUCKETS))
        self.sl_head = nn.Linear(hidden + 2 + len(NOTIONAL_BUCKETS) + len(LEVERAGE_BUCKETS) + len(TP_BUCKETS), len(SL_BUCKETS))
        self.hold_head = nn.Linear(hidden + 2 + len(NOTIONAL_BUCKETS) + len(LEVERAGE_BUCKETS) + len(TP_BUCKETS) + len(SL_BUCKETS), len(HOLD_BUCKETS))
        self.cooldown_head = nn.Linear(hidden + 2 + len(NOTIONAL_BUCKETS) + len(LEVERAGE_BUCKETS) + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS), len(COOLDOWN_BUCKETS))

    @staticmethod
    def _one_hot(idx: torch.Tensor, n: int) -> torch.Tensor:
        return F.one_hot(idx.clamp(0, n - 1), num_classes=n).float()

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        logp = torch.zeros(len(x), device=x.device)
        veto_dist = torch.distributions.Categorical(logits=self.veto_head(h))
        veto = veto_dist.sample()
        logp = logp + veto_dist.log_prob(veto)
        keep = veto == 0
        v = self._one_hot(veto, 2)
        n_dist = torch.distributions.Categorical(logits=self.notional_head(torch.cat([h, v], dim=-1)))
        n = n_dist.sample()
        logp = logp + torch.where(keep, n_dist.log_prob(n), torch.zeros_like(logp))
        n_oh = self._one_hot(n, len(NOTIONAL_BUCKETS))
        l_dist = torch.distributions.Categorical(logits=self.leverage_head(torch.cat([h, v, n_oh], dim=-1)))
        lev = l_dist.sample()
        logp = logp + torch.where(keep, l_dist.log_prob(lev), torch.zeros_like(logp))
        l_oh = self._one_hot(lev, len(LEVERAGE_BUCKETS))
        tp_dist = torch.distributions.Categorical(logits=self.tp_head(torch.cat([h, v, n_oh, l_oh], dim=-1)))
        tp = tp_dist.sample()
        logp = logp + torch.where(keep, tp_dist.log_prob(tp), torch.zeros_like(logp))
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))
        sl_dist = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, v, n_oh, l_oh, tp_oh], dim=-1)))
        sl = sl_dist.sample()
        logp = logp + torch.where(keep, sl_dist.log_prob(sl), torch.zeros_like(logp))
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))
        h_dist = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh], dim=-1)))
        hold = h_dist.sample()
        logp = logp + torch.where(keep, h_dist.log_prob(hold), torch.zeros_like(logp))
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))
        cd_dist = torch.distributions.Categorical(logits=self.cooldown_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh, hold_oh], dim=-1)))
        cd = cd_dist.sample()
        logp = logp + torch.where(keep, cd_dist.log_prob(cd), torch.zeros_like(logp))
        return _flat_id_tensor(veto, n, lev, tp, sl, hold, cd), logp

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        veto = torch.argmax(self.veto_head(h), dim=-1)
        v = self._one_hot(veto, 2)
        n = torch.argmax(self.notional_head(torch.cat([h, v], dim=-1)), dim=-1)
        n_oh = self._one_hot(n, len(NOTIONAL_BUCKETS))
        lev = torch.argmax(self.leverage_head(torch.cat([h, v, n_oh], dim=-1)), dim=-1)
        l_oh = self._one_hot(lev, len(LEVERAGE_BUCKETS))
        tp = torch.argmax(self.tp_head(torch.cat([h, v, n_oh, l_oh], dim=-1)), dim=-1)
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))
        sl = torch.argmax(self.sl_head(torch.cat([h, v, n_oh, l_oh, tp_oh], dim=-1)), dim=-1)
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))
        hold = torch.argmax(self.hold_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh], dim=-1)), dim=-1)
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))
        cd = torch.argmax(self.cooldown_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh, hold_oh], dim=-1)), dim=-1)
        return _flat_id_tensor(veto, n, lev, tp, sl, hold, cd)

    def log_prob_for_flat(self, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        veto, n, lev, tp, sl, hold, cd = _components_from_flat(flat)
        logp = torch.distributions.Categorical(logits=self.veto_head(h)).log_prob(veto)
        keep = veto == 0
        v = self._one_hot(veto, 2)
        n_dist = torch.distributions.Categorical(logits=self.notional_head(torch.cat([h, v], dim=-1)))
        logp = logp + torch.where(keep, n_dist.log_prob(n), torch.zeros_like(logp))
        n_oh = self._one_hot(n, len(NOTIONAL_BUCKETS))
        l_dist = torch.distributions.Categorical(logits=self.leverage_head(torch.cat([h, v, n_oh], dim=-1)))
        logp = logp + torch.where(keep, l_dist.log_prob(lev), torch.zeros_like(logp))
        l_oh = self._one_hot(lev, len(LEVERAGE_BUCKETS))
        tp_dist = torch.distributions.Categorical(logits=self.tp_head(torch.cat([h, v, n_oh, l_oh], dim=-1)))
        logp = logp + torch.where(keep, tp_dist.log_prob(tp), torch.zeros_like(logp))
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))
        sl_dist = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, v, n_oh, l_oh, tp_oh], dim=-1)))
        logp = logp + torch.where(keep, sl_dist.log_prob(sl), torch.zeros_like(logp))
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))
        h_dist = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh], dim=-1)))
        logp = logp + torch.where(keep, h_dist.log_prob(hold), torch.zeros_like(logp))
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))
        cd_dist = torch.distributions.Categorical(logits=self.cooldown_head(torch.cat([h, v, n_oh, l_oh, tp_oh, sl_oh, hold_oh], dim=-1)))
        logp = logp + torch.where(keep, cd_dist.log_prob(cd), torch.zeros_like(logp))
        return logp


class Critic(nn.Module):
    def __init__(self, state_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(256, 192),
            nn.SiLU(),
            nn.Linear(192, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class OfflineDataset:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    best_actions: np.ndarray


def _canonical_action_ids() -> list[int]:
    ids = [0]
    targets = [
        (0, 0, 0, 0, 0, 0, 0),  # current Omega1 template: 0.45/2.0/tp.026/sl.014/72/cd6
        (0, 1, 0, 0, 0, 0, 0),
        (0, 1, 1, 1, 1, 1, 0),
    ]
    ids.extend(ACTION_LOOKUP[t] for t in targets)
    return ids


def _sample_action_ids(rng: np.random.Generator, count: int) -> np.ndarray:
    ids = set(_canonical_action_ids())
    while len(ids) < int(count):
        ids.add(int(rng.integers(1, ACTION_DIM)))
    return np.asarray(sorted(ids), dtype=np.int64)


def _action_name(flat_id: int) -> str:
    spec = ACTION_SPECS[int(flat_id)]
    if spec.veto:
        return "veto"
    return (
        f"n{NOTIONAL_BUCKETS[spec.notional_id]:.2f}_l{LEVERAGE_BUCKETS[spec.leverage_id]:.1f}_"
        f"tp{TP_BUCKETS[spec.tp_id]:.3f}_sl{SL_BUCKETS[spec.sl_id]:.3f}_"
        f"h{HOLD_BUCKETS[spec.hold_id]}_cd{COOLDOWN_BUCKETS[spec.cooldown_id]}"
    )


def _action_count_names(counts: dict[int, int], *, limit: int = 20) -> dict[str, int]:
    return {_action_name(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]}


def _build_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    samples_per_row: int,
    max_active_rows: int,
) -> tuple[OfflineDataset, dict[str, Any]]:
    active = _active(dec)
    idxs = np.flatnonzero(active & (np.arange(len(frame)) < len(frame) - 3))
    arrays = {k: _num(frame, k) for k in ("open", "high", "low", "close")}
    rng = np.random.default_rng(260602)
    total_active_rows = int(len(idxs))
    if int(max_active_rows) > 0 and len(idxs) > int(max_active_rows):
        idxs = np.sort(rng.choice(idxs, size=int(max_active_rows), replace=False))
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    best_list: list[int] = []
    oracle_counts: dict[int, int] = {}
    net_sum: dict[int, list[float]] = {}
    for i in idxs:
        action_ids = _sample_action_ids(rng, samples_per_row)
        row_rewards: list[tuple[int, float, dict[str, Any]]] = []
        best_a = 0
        best_r = -1e18
        for flat_id in action_ids:
            reward, meta = _simulate_action(frame, arrays, int(i), dec.iloc[int(i)], int(flat_id), fee=fee, slip=slip, cost_mult=cost_mult)
            row_rewards.append((int(flat_id), float(reward), meta))
            if reward > best_r:
                best_r = float(reward)
                best_a = int(flat_id)
        oracle_counts[best_a] = oracle_counts.get(best_a, 0) + 1
        for flat_id, reward, meta in row_rewards:
            s_list.append(states[int(i)])
            sp_list.append(states[min(int(i) + 1, len(states) - 1)])
            a_list.append(flat_id)
            r_list.append(reward)
            d_list.append(1.0)
            best_list.append(best_a)
            if int(meta["active"]) == 1:
                net_sum.setdefault(flat_id, []).append(float(meta["net"]))
    rewards = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards = np.clip(rewards / scale, -8.0, 8.0).astype(np.float32)
    means = {k: float(np.mean(v)) for k, v in net_sum.items() if v}
    diagnostics = {
        "active_rows": int(len(idxs)),
        "total_active_rows": int(total_active_rows),
        "samples_per_row": int(samples_per_row),
        "sample_count": int(len(rewards)),
        "reward_scale": float(scale),
        "oracle_top_actions": _action_count_names(oracle_counts, limit=15),
        "mean_net_top_actions": {_action_name(k): v for k, v in sorted(means.items(), key=lambda kv: kv[1], reverse=True)[:15]},
    }
    return (
        OfflineDataset(
            states=np.asarray(s_list, dtype=np.float32),
            next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64),
            rewards=rewards,
            dones=np.asarray(d_list, dtype=np.float32),
            best_actions=np.asarray(best_list, dtype=np.int64),
        ),
        diagnostics,
    )


def _train_dsac(data: OfflineDataset, *, state_dim: int, device: torch.device, steps: int, batch_size: int, lr: float, bc_coef: float) -> tuple[AutoRegRiskActor, dict[str, Any]]:
    actor = AutoRegRiskActor(state_dim).to(device)
    q1 = Critic(state_dim).to(device)
    q2 = Critic(state_dim).to(device)
    tq1 = Critic(state_dim).to(device)
    tq2 = Critic(state_dim).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.12), device=device, requires_grad=True)
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=lr, weight_decay=1e-5)
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=lr, weight_decay=1e-5)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    ds = TensorDataset(
        torch.from_numpy(data.states),
        torch.from_numpy(data.next_states),
        torch.from_numpy(data.actions),
        torch.from_numpy(data.rewards),
        torch.from_numpy(data.dones),
        torch.from_numpy(data.best_actions),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            s, sp, a, r, d, best_a = next(it)
        except StopIteration:
            it = iter(dl)
            s, sp, a, r, d, best_a = next(it)
        s = s.to(device)
        sp = sp.to(device)
        a = a.to(device)
        r = r.to(device)
        d = d.to(device)
        best_a = best_a.to(device)
        with torch.no_grad():
            na, nlogp = actor.sample(sp)
            next_q = torch.min(tq1(sp), tq2(sp)).gather(1, na.view(-1, 1)).squeeze(1)
            y = r + (1.0 - d) * 0.995 * (next_q - log_alpha.exp() * nlogp)
        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.smooth_l1_loss(qa1, y) + F.smooth_l1_loss(qa2, y)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()

        pa, plogp = actor.sample(s)
        pq = torch.min(q1(s), q2(s)).gather(1, pa.view(-1, 1)).squeeze(1)
        bc_loss = -actor.log_prob_for_flat(s, best_a).mean()
        actor_loss = (log_alpha.exp() * plogp - pq).mean() + float(bc_coef) * bc_loss
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = (-plogp).mean().detach()
        alpha_loss = -(log_alpha * (entropy - 2.0)).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(3.0))
        with torch.no_grad():
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(0.99).add_(0.01 * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(0.99).add_(0.01 * p.data)
        if step % 250 == 0:
            last = {
                "step": int(step),
                "q_loss": float(q_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "bc_loss": float(bc_loss.detach().cpu()),
                "alpha": float(log_alpha.exp().detach().cpu()),
                "entropy": float(entropy.cpu()),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), last


def _policy_actions(actor: AutoRegRiskActor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            out.append(actor.greedy(x).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _compose_decisions(base: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = base.copy().reset_index(drop=True)
    active = _active(out)
    for i in np.flatnonzero(active):
        out.iloc[int(i)] = _apply_action(out.iloc[int(i)], int(actions[int(i)]))
    out.loc[~active] = out.loc[~active].apply(_zero_row, axis=1)
    return out


def _fixed_decisions(base: pd.DataFrame, flat_id: int) -> pd.DataFrame:
    return _compose_decisions(base, np.full(len(base), int(flat_id), dtype=np.int64))


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    counts: dict[int, int] = {}
    for a in actions[np.asarray(active, dtype=bool)]:
        counts[int(a)] = counts.get(int(a), 0) + 1
    return _action_count_names(counts, limit=20)


def _load_variant_frames(variant: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_all, eval_df, overlay = _load_frames_max()
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_src_path, oos_src_path = _expertdq_paths(variant)
    train_src_all = _load_csv(val_src_path)
    oos_src = _load_csv(oos_src_path)
    _require_unique_timestamps(train_src_all, f"{variant} train/oof")
    _require_unique_timestamps(oos_src, f"{variant} oos")
    train_frame, _, train_src = _align_source_to_frame(train_df, pd.DataFrame(index=train_df.index), train_src_all)
    val_frame, _, val_src = _align_source_to_frame(val_df, pd.DataFrame(index=val_df.index), train_src_all)
    oos_frame, _, oos_src2 = _align_source_to_frame(eval_df, pd.DataFrame(index=eval_df.index), oos_src)
    return train_frame, val_frame, oos_frame, train_src, val_src, oos_src2, overlay


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p10")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.08)
    ap.add_argument("--samples-per-row", type=int, default=96)
    ap.add_argument("--max-active-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(260602)
    out_dir = OUT_DIR / str(args.variant)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")

    train_df, val_df, oos_df, train_src, val_src, oos_src, overlay = _load_variant_frames(str(args.variant))
    train_dec = _apply_scale(_to_decisions(train_src, oof=True), **ACTIVE_SCALES)
    val_dec = _apply_scale(_to_decisions(val_src, oof=True), **ACTIVE_SCALES)
    oos_dec = _apply_scale(_to_decisions(oos_src, oof=False), **ACTIVE_SCALES)

    feature_cols = _numeric_feature_cols(train_df)
    s_train = _build_state_frame(train_df, train_dec, train_src, oof=True, feature_cols=feature_cols)
    s_val = _build_state_frame(val_df, val_dec, val_src, oof=True, feature_cols=feature_cols)
    s_oos = _build_state_frame(oos_df, oos_dec, oos_src, oof=False, feature_cols=feature_cols)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_oos = _apply_norm(s_oos, norm)

    parent_cfg = joblib.load(v31.DEFAULT_PARENT)["config"]
    fee = float(parent_cfg["fee"])
    slip = float(parent_cfg["slip"])
    dataset, data_diag = _build_dataset(
        train_df,
        x_train,
        train_dec,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        samples_per_row=int(args.samples_per_row),
        max_active_rows=int(args.max_active_rows),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "variant": args.variant,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "action_dim": int(ACTION_DIM),
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(oos_df)),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    actor, train_diag = _train_dsac(
        dataset,
        state_dim=int(x_train.shape[1]),
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        bc_coef=float(args.bc_coef),
    )
    a_train = _policy_actions(actor, x_train, device=device)
    a_val = _policy_actions(actor, x_val, device=device)
    a_oos = _policy_actions(actor, x_oos, device=device)

    dsac_train = _compose_decisions(train_dec, a_train)
    dsac_val = _compose_decisions(val_dec, a_val)
    dsac_oos = _compose_decisions(oos_dec, a_oos)
    template_id = ACTION_LOOKUP[(0, 0, 0, 0, 0, 0, 0)]
    conservative_id = ACTION_LOOKUP[(0, 0, 0, 0, 0, 0, 0)]
    aggressive_id = ACTION_LOOKUP[(0, 1, 1, 1, 1, 1, 0)]
    variants = {
        "fixed_omega1_template": (train_dec, val_dec, oos_dec),
        "dsac_risk_allocator": (dsac_train, dsac_val, dsac_oos),
    }
    rows: list[dict[str, Any]] = []
    for split, frame, idx in (("val", val_df, 1), ("oos", oos_df, 2)):
        for name, decs in variants.items():
            metrics = _fast_replay_metrics(frame, decs[idx], fee=fee, slip=slip, cost_mult=float(args.cost_mult))
            row = {"split": split, "variant": name, "cost": 3, **metrics}
            row["selection_score"] = _score(pd.Series(row))
            rows.append(row)
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)
    val_rank = grid[(grid["split"] == "val") & (grid["cost"] == 3)].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"])
    selected_oos = grid[(grid["split"] == "oos") & (grid["cost"] == 3) & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    dsac_oos_row = grid[(grid["split"] == "oos") & (grid["cost"] == 3) & (grid["variant"] == "dsac_risk_allocator")].iloc[0].to_dict()
    fixed_oos_row = grid[(grid["split"] == "oos") & (grid["cost"] == 3) & (grid["variant"] == "fixed_omega1_template")].iloc[0].to_dict()

    model_path = out_dir / "omega1_expertdq_dsac_risk_allocator.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": str(args.variant),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "buckets": {
                "notional": NOTIONAL_BUCKETS,
                "leverage": LEVERAGE_BUCKETS,
                "tp": TP_BUCKETS,
                "sl": SL_BUCKETS,
                "hold": HOLD_BUCKETS,
                "cooldown": COOLDOWN_BUCKETS,
            },
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Omega1 supervised expert-local decision/quality is frozen. DSAC owns only risk/execution heads: notional, leverage, TP, SL, max-hold, cooldown, plus veto.",
        "selection_basis": "2025Q4 validation Cost3 only; 2026 OOS is report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "base_risk_template": ACTIVE_TEMPLATE,
        "base_expert_scales": ACTIVE_SCALES,
        "feature_cols": feature_cols,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "samples_per_row": int(args.samples_per_row),
            "cost_mult": float(args.cost_mult),
            "reward_label": "complete_trade_net_pnl_after_entry_exit_fee_slippage",
            "data_diag": data_diag,
            "train_diag": train_diag,
            "action_usage": {
                "train": _usage(a_train, _active(train_dec)),
                "val": _usage(a_val, _active(val_dec)),
                "oos": _usage(a_oos, _active(oos_dec)),
            },
        },
        "selected_by_val_cost3": {
            "variant": selected_variant,
            "oos_cost3": selected_oos,
        },
        "dsac_oos_cost3": dsac_oos_row,
        "fixed_template_oos_cost3": fixed_oos_row,
        "delta_dsac_vs_fixed_template_oos_cost3_pnl": float(dsac_oos_row["pnl"]) - float(fixed_oos_row["pnl"]),
        "overlay": overlay,
        "artifacts": {
            "summary": str(out_dir / "summary.json"),
            "grid": str(grid_path),
            "model": str(model_path),
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "selected": summary["selected_by_val_cost3"], "dsac_oos_cost3": dsac_oos_row, "fixed_template_oos_cost3": fixed_oos_row}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
