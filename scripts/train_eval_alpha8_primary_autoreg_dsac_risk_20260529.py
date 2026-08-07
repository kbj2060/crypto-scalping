#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
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

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    DECISION_COLS,
    EVAL_CSV,
    FORBIDDEN_PREFIXES,
    SOURCE_COLS,
    _apply_norm,
    _audit_frame_contract,
    _directional_features,
    _fit_norm,
    _safe_num,
    TRAIN_CSV,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha8_primary_autoreg_dsac_tradepnl_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TP_BUCKETS = (0.110, 0.120, 0.175, 0.200, 0.225)
SL_BUCKETS = (4.0, 5.0, 6.0)
HOLD_BUCKETS = (0.50, 0.75, 1.00)
MULT_BUCKETS = (1.00, 1.10, 1.20, 1.35, 1.50, 1.75)
CAP_BUCKETS = (5.0, 7.5, 10.0)

TEACHER_COLS = [
    "teacher_long_edge",
    "teacher_short_edge",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_quantile_skew",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


@dataclass(frozen=True)
class ActionSpec:
    flat_id: int
    veto: int
    tp_id: int
    sl_id: int
    hold_id: int
    mult_id: int
    cap_id: int


def _build_action_space() -> tuple[list[ActionSpec], dict[tuple[int, int, int, int, int, int], int]]:
    specs = [ActionSpec(0, 1, 0, 0, 0, 0, 0)]
    lookup = {(1, 0, 0, 0, 0, 0): 0}
    flat = 1
    for tp_i in range(len(TP_BUCKETS)):
        for sl_i in range(len(SL_BUCKETS)):
            for hold_i in range(len(HOLD_BUCKETS)):
                for mult_i in range(len(MULT_BUCKETS)):
                    for cap_i in range(len(CAP_BUCKETS)):
                        key = (0, tp_i, sl_i, hold_i, mult_i, cap_i)
                        specs.append(ActionSpec(flat, *key))
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


def _assert_clean(df: pd.DataFrame, *, name: str) -> None:
    bad = [c for c in df.columns if str(c).startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _num(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.full(len(df), float(default), dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).to_numpy(dtype=np.float64)


def _active(dec: pd.DataFrame) -> np.ndarray:
    action = _num(dec, "action").astype(np.int64)
    side = _num(dec, "side").astype(np.int64)
    notional = _num(dec, "notional_exposure")
    return (action != ACTION_CASH) & (side != 0) & (notional > 0.0)


def _state_frame(frame: pd.DataFrame, primary: pd.DataFrame) -> pd.DataFrame:
    _audit_frame_contract(frame, name="alpha8_primary_autoreg_state")
    _assert_clean(frame, name="alpha8_primary_autoreg_state")
    parts: list[pd.DataFrame] = [_directional_features(frame).reset_index(drop=True)]
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in SOURCE_COLS}, index=frame.index).reset_index(drop=True))
    for col in TEACHER_COLS:
        if col not in frame.columns:
            raise RuntimeError(f"certified teacher feature missing: {col}")
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in TEACHER_COLS}, index=frame.index).reset_index(drop=True))
    d = pd.DataFrame(index=frame.index)
    for col in DECISION_COLS:
        if col not in primary.columns:
            raise RuntimeError(f"primary decision missing column: {col}")
        d[f"primary_{col}"] = _num(primary, col)
    d["primary_rr"] = d["primary_take_profit"] / np.maximum(np.abs(d["primary_stop_loss"]), 1e-8)
    d["primary_margin_fraction"] = d["primary_notional_exposure"] / np.maximum(d["primary_leverage"], 1e-8)
    d["primary_side_x_confidence"] = d["primary_side"] * d["primary_confidence"]
    d["primary_side_x_quality"] = d["primary_side"] * d["primary_quality_score"]
    parts.append(d.reset_index(drop=True))
    out = pd.concat(parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate alpha8 primary autoreg state columns: {dup[:20]}")
    return out


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
    base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
    base_leverage = float(row.get("leverage", 1.0) or 1.0)
    base_tp = float(row.get("take_profit", 0.0) or 0.0)
    base_sl = abs(float(row.get("stop_loss", 0.0) or 0.0))
    base_hold = max(int(row.get("max_hold_bars", 0) or 0), 1)
    mult = MULT_BUCKETS[spec.mult_id]
    cap = CAP_BUCKETS[spec.cap_id]
    notional = float(min(max(base_notional * mult, 0.0), cap))
    leverage = max(base_leverage, 1e-8)
    out.loc["notional_exposure"] = notional
    out.loc["position_fraction"] = notional / leverage
    out.loc["take_profit"] = max(base_tp, 1e-8) * TP_BUCKETS[spec.tp_id]
    out.loc["stop_loss"] = max(base_sl, 1e-8) * SL_BUCKETS[spec.sl_id]
    out.loc["max_hold_bars"] = int(max(1, round(base_hold * HOLD_BUCKETS[spec.hold_id])))
    return out


def _simulate_action(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    i: int,
    dec_row: pd.Series,
    flat_id: int,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, dict[str, Any]]:
    dec = _apply_action(dec_row, flat_id)
    action = int(dec.get("action", 0) or 0)
    side = int(dec.get("side", 0) or 0)
    notional = float(dec.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "mae": 0.0}
    entry_i = min(int(i) + 1, len(frame) - 1)
    entry_px = float(arrays["open"][entry_i])
    if entry_px <= 0.0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0, "mae": 0.0}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    entry = entry_px * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    tp = float(dec.get("take_profit", 0.0) or 0.0)
    sl = abs(float(dec.get("stop_loss", 0.0) or 0.0))
    hold = max(int(dec.get("max_hold_bars", 0) or 0), 1)
    end_i = min(entry_i + hold, len(frame) - 1)
    exit_fill: float | None = None
    mae = 0.0
    mfe = 0.0
    exit_i = end_i
    exit_reason = "hold"
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = (float(arrays["high"][j]) / max(entry, 1e-12) - 1.0) * notional
            adverse = (float(arrays["low"][j]) / max(entry, 1e-12) - 1.0) * notional
        else:
            favorable = (entry / max(float(arrays["low"][j]), 1e-12) - 1.0) * notional
            adverse = (entry / max(float(arrays["high"][j]), 1e-12) - 1.0) * notional
        mfe = max(mfe, favorable)
        mae = min(mae, adverse)
        if adverse <= -sl:
            raw_stop = max(sl / max(notional, 1e-12), 0.0)
            if side > 0:
                trigger_px = entry * max(1.0 - raw_stop, 1e-8)
                exit_fill = trigger_px * (1.0 - slip_eff)
            else:
                trigger_px = entry / max(1.0 - raw_stop, 1e-8)
                exit_fill = trigger_px * (1.0 + slip_eff)
            exit_i = j
            exit_reason = "stop_loss"
            break
        if favorable >= tp:
            raw_tp = max(tp / max(notional, 1e-12), 0.0)
            if side > 0:
                trigger_px = entry * (1.0 + raw_tp)
                exit_fill = trigger_px * (1.0 - slip_eff)
            else:
                trigger_px = entry / max(1.0 + raw_tp, 1e-8)
                exit_fill = trigger_px * (1.0 + slip_eff)
            exit_i = j
            exit_reason = "take_profit"
            break
    if exit_fill is None:
        exit_px = float(arrays["close"][end_i])
        exit_fill = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    entry_notional = float(notional)
    qty = entry_notional / max(entry, 1e-12)
    exit_notional = qty * max(float(exit_fill), 0.0)
    gross = exit_notional - entry_notional if side > 0 else entry_notional - exit_notional
    entry_fee = fee_eff * entry_notional
    exit_fee = fee_eff * exit_notional
    net = float(gross - entry_fee - exit_fee)
    win = int(net > 0.0)
    # Trade-PnL mode: the label/reward is the complete realized trade outcome
    # after entry/exit cost. No win-rate, MAE, or holding-time shaping here.
    return float(net), {"active": 1, "net": net, "win": win, "mae": mae, "mfe": mfe, "exit_reason": exit_reason, "exit_notional": exit_notional}


def _canonical_action_ids() -> list[int]:
    targets = [
        (0, 2, 1, 1, 1, 1),  # tp=.175 sl=5 hold=.75 mult=1.1 cap=7.5
        (0, 3, 1, 1, 1, 1),  # tp=.20 sl=5 hold=.75 mult=1.1 cap=7.5
        (0, 2, 1, 1, 2, 1),  # tp=.175 sl=5 hold=.75 mult=1.2 cap=7.5
        (0, 3, 1, 1, 2, 1),  # alpha8 54/55 cap=7.5
        (0, 3, 1, 1, 2, 2),  # alpha8 55 cap=10
        (0, 3, 0, 1, 3, 1),  # tp=.20 sl=4 hold=.75 mult=1.35
        (0, 2, 0, 1, 3, 1),  # tp=.175 sl=4 hold=.75 mult=1.35
        (0, 3, 1, 1, 5, 0),  # aggressive cap=5
        (0, 1, 0, 0, 0, 0),  # micro
    ]
    return [0, *[ACTION_LOOKUP[t] for t in targets]]


def _sample_action_ids(rng: np.random.Generator, count: int) -> np.ndarray:
    ids = set(_canonical_action_ids())
    max_id = ACTION_DIM - 1
    while len(ids) < count:
        ids.add(int(rng.integers(1, max_id + 1)))
    return np.asarray(sorted(ids), dtype=np.int64)


@dataclass
class OfflineDataset:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    best_actions: np.ndarray


def _build_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    primary: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    samples_per_row: int,
) -> tuple[OfflineDataset, dict[str, Any]]:
    active = _active(primary)
    idxs = np.flatnonzero(active & (np.arange(len(frame)) < len(frame) - 3))
    arrays = {
        "open": _num(frame, "open"),
        "high": _num(frame, "high"),
        "low": _num(frame, "low"),
        "close": _num(frame, "close"),
    }
    rng = np.random.default_rng(80529)
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    best_list: list[int] = []
    win_by_action: dict[int, list[int]] = {}
    net_by_action: dict[int, list[float]] = {}
    oracle_counts: dict[int, int] = {}
    for i in idxs:
        action_ids = _sample_action_ids(rng, int(samples_per_row))
        best_a = 0
        best_r = -1e18
        row_items: list[tuple[np.ndarray, np.ndarray, int, float, float, dict[str, Any]]] = []
        for flat_id in action_ids:
            reward, meta = _simulate_action(frame, arrays, int(i), primary.iloc[int(i)], int(flat_id), fee=fee, slip=slip, cost_mult=cost_mult)
            if reward > best_r:
                best_r = reward
                best_a = int(flat_id)
            row_items.append((states[int(i)], states[min(int(i) + 1, len(states) - 1)], int(flat_id), float(reward), 1.0, meta))
        for s, sp, flat_id, reward, done, meta in row_items:
            s_list.append(s)
            sp_list.append(sp)
            a_list.append(int(flat_id))
            r_list.append(float(reward))
            d_list.append(float(done))
            best_list.append(int(best_a))
            if int(meta["active"]) == 1:
                win_by_action.setdefault(int(flat_id), []).append(int(meta["win"]))
                net_by_action.setdefault(int(flat_id), []).append(float(meta["net"]))
        oracle_counts[best_a] = oracle_counts.get(best_a, 0) + 1
    rewards = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards = np.clip(rewards / scale, -8.0, 8.0).astype(np.float32)
    diagnostics = {
        "active_rows": int(len(idxs)),
        "samples_per_row": int(samples_per_row),
        "sample_count": int(len(rewards)),
        "reward_scale": float(scale),
        "oracle_top_actions": _action_count_names(oracle_counts, limit=12),
        "canonical_action_ids": _action_count_names({i: 1 for i in _canonical_action_ids()}, limit=20),
        "action_net_mean_top": _action_metric_names(net_by_action, limit=12),
        "action_win_rate_top": _action_metric_names(win_by_action, limit=12),
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


def _action_name(flat_id: int) -> str:
    spec = ACTION_SPECS[int(flat_id)]
    if spec.veto:
        return "veto"
    return (
        f"tp{TP_BUCKETS[spec.tp_id]:.3f}_sl{SL_BUCKETS[spec.sl_id]:.1f}_"
        f"h{HOLD_BUCKETS[spec.hold_id]:.2f}_m{MULT_BUCKETS[spec.mult_id]:.2f}_c{CAP_BUCKETS[spec.cap_id]:.1f}"
    )


def _action_count_names(counts: dict[int, int], *, limit: int) -> dict[str, int]:
    items = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[: int(limit)]
    return {_action_name(k): int(v) for k, v in items}


def _action_metric_names(values: dict[int, list[float | int]], *, limit: int) -> dict[str, float]:
    means = {k: float(np.mean(v)) for k, v in values.items() if v}
    items = sorted(means.items(), key=lambda kv: kv[1], reverse=True)[: int(limit)]
    return {_action_name(k): float(v) for k, v in items}


class AutoRegActor(nn.Module):
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
        self.tp_head = nn.Linear(hidden + 2, len(TP_BUCKETS))
        self.sl_head = nn.Linear(hidden + 2 + len(TP_BUCKETS), len(SL_BUCKETS))
        self.hold_head = nn.Linear(hidden + 2 + len(TP_BUCKETS) + len(SL_BUCKETS), len(HOLD_BUCKETS))
        self.mult_head = nn.Linear(hidden + 2 + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS), len(MULT_BUCKETS))
        self.cap_head = nn.Linear(
            hidden + 2 + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS) + len(MULT_BUCKETS),
            len(CAP_BUCKETS),
        )

    def _one_hot(self, idx: torch.Tensor, n: int) -> torch.Tensor:
        return F.one_hot(idx.clamp(0, n - 1), num_classes=n).float()

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        logp_total = torch.zeros(len(x), device=x.device)
        veto_logits = self.veto_head(h)
        veto_dist = torch.distributions.Categorical(logits=veto_logits)
        veto = veto_dist.sample()
        logp_total = logp_total + veto_dist.log_prob(veto)
        v_oh = self._one_hot(veto, 2)

        tp_dist = torch.distributions.Categorical(logits=self.tp_head(torch.cat([h, v_oh], dim=-1)))
        tp = tp_dist.sample()
        logp_total = logp_total + torch.where(veto == 0, tp_dist.log_prob(tp), torch.zeros_like(logp_total))
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))

        sl_dist = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, v_oh, tp_oh], dim=-1)))
        sl = sl_dist.sample()
        logp_total = logp_total + torch.where(veto == 0, sl_dist.log_prob(sl), torch.zeros_like(logp_total))
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))

        hold_dist = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, v_oh, tp_oh, sl_oh], dim=-1)))
        hold = hold_dist.sample()
        logp_total = logp_total + torch.where(veto == 0, hold_dist.log_prob(hold), torch.zeros_like(logp_total))
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))

        mult_dist = torch.distributions.Categorical(logits=self.mult_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh], dim=-1)))
        mult = mult_dist.sample()
        logp_total = logp_total + torch.where(veto == 0, mult_dist.log_prob(mult), torch.zeros_like(logp_total))
        mult_oh = self._one_hot(mult, len(MULT_BUCKETS))

        cap_dist = torch.distributions.Categorical(logits=self.cap_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)))
        cap = cap_dist.sample()
        logp_total = logp_total + torch.where(veto == 0, cap_dist.log_prob(cap), torch.zeros_like(logp_total))
        flat = _flat_id_tensor(veto, tp, sl, hold, mult, cap)
        return flat, logp_total

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        veto = torch.argmax(self.veto_head(h), dim=-1)
        v_oh = self._one_hot(veto, 2)
        tp = torch.argmax(self.tp_head(torch.cat([h, v_oh], dim=-1)), dim=-1)
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))
        sl = torch.argmax(self.sl_head(torch.cat([h, v_oh, tp_oh], dim=-1)), dim=-1)
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))
        hold = torch.argmax(self.hold_head(torch.cat([h, v_oh, tp_oh, sl_oh], dim=-1)), dim=-1)
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))
        mult = torch.argmax(self.mult_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh], dim=-1)), dim=-1)
        mult_oh = self._one_hot(mult, len(MULT_BUCKETS))
        cap = torch.argmax(self.cap_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)), dim=-1)
        return _flat_id_tensor(veto, tp, sl, hold, mult, cap)

    def log_prob_for_flat(self, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        veto, tp, sl, hold, mult, cap = _components_from_flat(flat)
        logp = torch.distributions.Categorical(logits=self.veto_head(h)).log_prob(veto)
        keep = veto == 0
        v_oh = self._one_hot(veto, 2)

        tp_dist = torch.distributions.Categorical(logits=self.tp_head(torch.cat([h, v_oh], dim=-1)))
        logp = logp + torch.where(keep, tp_dist.log_prob(tp), torch.zeros_like(logp))
        tp_oh = self._one_hot(tp, len(TP_BUCKETS))

        sl_dist = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, v_oh, tp_oh], dim=-1)))
        logp = logp + torch.where(keep, sl_dist.log_prob(sl), torch.zeros_like(logp))
        sl_oh = self._one_hot(sl, len(SL_BUCKETS))

        hold_dist = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, v_oh, tp_oh, sl_oh], dim=-1)))
        logp = logp + torch.where(keep, hold_dist.log_prob(hold), torch.zeros_like(logp))
        hold_oh = self._one_hot(hold, len(HOLD_BUCKETS))

        mult_dist = torch.distributions.Categorical(logits=self.mult_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh], dim=-1)))
        logp = logp + torch.where(keep, mult_dist.log_prob(mult), torch.zeros_like(logp))
        mult_oh = self._one_hot(mult, len(MULT_BUCKETS))

        cap_dist = torch.distributions.Categorical(logits=self.cap_head(torch.cat([h, v_oh, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)))
        logp = logp + torch.where(keep, cap_dist.log_prob(cap), torch.zeros_like(logp))
        return logp


def _flat_id_tensor(veto: torch.Tensor, tp: torch.Tensor, sl: torch.Tensor, hold: torch.Tensor, mult: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    keep_id = 1 + (((tp * len(SL_BUCKETS) + sl) * len(HOLD_BUCKETS) + hold) * len(MULT_BUCKETS) + mult) * len(CAP_BUCKETS) + cap
    return torch.where(veto == 1, torch.zeros_like(keep_id), keep_id.long())


def _components_from_flat(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = flat.long()
    veto = torch.where(flat == 0, torch.ones_like(flat), torch.zeros_like(flat))
    idx = torch.clamp(flat - 1, min=0)
    cap = idx % len(CAP_BUCKETS)
    idx = idx // len(CAP_BUCKETS)
    mult = idx % len(MULT_BUCKETS)
    idx = idx // len(MULT_BUCKETS)
    hold = idx % len(HOLD_BUCKETS)
    idx = idx // len(HOLD_BUCKETS)
    sl = idx % len(SL_BUCKETS)
    tp = idx // len(SL_BUCKETS)
    return veto, tp, sl, hold, mult, cap


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(256, 192),
            nn.SiLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_autoreg_dsac(
    data: OfflineDataset,
    *,
    state_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    bc_coef: float,
) -> tuple[AutoRegActor, dict[str, Any]]:
    actor = AutoRegActor(state_dim).to(device)
    q1 = Critic(state_dim, ACTION_DIM).to(device)
    q2 = Critic(state_dim, ACTION_DIM).to(device)
    tq1 = Critic(state_dim, ACTION_DIM).to(device)
    tq2 = Critic(state_dim, ACTION_DIM).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.12), device=device, requires_grad=True)
    target_entropy = 2.0
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
    last = {}
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
        label_logp = actor.log_prob_for_flat(s, best_a)
        bc_loss = -label_logp.mean()
        actor_loss = (log_alpha.exp() * plogp - pq).mean() + float(bc_coef) * bc_loss
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = (-plogp).mean().detach()
        alpha_loss = -(log_alpha * (entropy - target_entropy)).mean()
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
            print(json.dumps({"stage": "autoreg_dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), last


def _policy_actions(actor: AutoRegActor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            out.append(actor.greedy(x).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _compose_decisions(primary: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    active = _active(out)
    for i in np.flatnonzero(active):
        out.iloc[int(i)] = _apply_action(out.iloc[int(i)], int(actions[int(i)]))
    out.loc[~active] = out.loc[~active].apply(_zero_row, axis=1)
    return out


def _fixed_action_id(tp: float, sl: float, hold: float, mult: float, cap: float) -> int:
    return ACTION_LOOKUP[
        (
            0,
            TP_BUCKETS.index(float(tp)),
            SL_BUCKETS.index(float(sl)),
            HOLD_BUCKETS.index(float(hold)),
            MULT_BUCKETS.index(float(mult)),
            CAP_BUCKETS.index(float(cap)),
        )
    ]


def _fixed_decisions(primary: pd.DataFrame, flat_id: int) -> pd.DataFrame:
    actions = np.full(len(primary), int(flat_id), dtype=np.int64)
    return _compose_decisions(primary, actions)


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    counts: dict[int, int] = {}
    for a in actions[np.asarray(active, dtype=bool)]:
        counts[int(a)] = counts.get(int(a), 0) + 1
    return _action_count_names(counts, limit=20)


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) or 0.0) + 130.0 * float(row.get("wr", 0.0) or 0.0) - 0.45 * abs(float(row.get("mdd", 0.0) or 0.0)) + 0.015 * trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.08)
    ap.add_argument("--samples-per-row", type=int, default=96)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    _seed_everything(290529)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    primary = joblib.load(baseline.primary_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    p_train = _predict_scaled(primary, train_df, primary_rt).reset_index(drop=True)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)

    s_train = _state_frame(train_df, p_train)
    s_val = _state_frame(val_df, p_val)
    s_eval = _state_frame(eval_df, p_eval)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_eval = _apply_norm(s_eval, norm)

    evaluator = OfficialCost3()
    dataset, data_diag = _build_dataset(
        train_df,
        x_train,
        p_train,
        fee=float(evaluator.fee),
        slip=float(evaluator.slip),
        cost_mult=float(args.cost_mult),
        samples_per_row=int(args.samples_per_row),
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "action_dim": ACTION_DIM,
                "train_rows": len(train_df),
                "val_rows": len(val_df),
                "oos_rows": len(eval_df),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    actor, train_diag = _train_autoreg_dsac(
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
    a_eval = _policy_actions(actor, x_eval, device=device)

    dsac_train = _compose_decisions(p_train, a_train)
    dsac_val = _compose_decisions(p_val, a_val)
    dsac_eval = _compose_decisions(p_eval, a_eval)
    base54_id = _fixed_action_id(0.200, 5.0, 0.75, 1.20, 7.5)
    highwr_id = _fixed_action_id(0.200, 5.0, 0.75, 1.10, 7.5)
    aggressive_id = _fixed_action_id(0.200, 5.0, 0.75, 1.75, 5.0)

    variants = {
        "primary_parent": (p_train, p_val, p_eval),
        "fixed_alpha8_54": (_fixed_decisions(p_train, base54_id), _fixed_decisions(p_val, base54_id), _fixed_decisions(p_eval, base54_id)),
        "fixed_highwr_110": (_fixed_decisions(p_train, highwr_id), _fixed_decisions(p_val, highwr_id), _fixed_decisions(p_eval, highwr_id)),
        "fixed_aggressive_175": (_fixed_decisions(p_train, aggressive_id), _fixed_decisions(p_val, aggressive_id), _fixed_decisions(p_eval, aggressive_id)),
        "autoreg_dsac": (dsac_train, dsac_val, dsac_eval),
    }
    rows: list[dict[str, Any]] = []
    for split, frame, idx in (("train", train_df, 0), ("val", val_df, 1), ("oos", eval_df, 2)):
        for name, decs in variants.items():
            m = evaluator(frame, decs[idx])
            rows.append({"split": split, "variant": name, **m})
    grid = pd.DataFrame(rows)
    grid["selection_score"] = grid.apply(_score, axis=1)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)
    val_rank = grid[(grid["split"] == "val") & (grid["variant"].isin(["autoreg_dsac", "fixed_alpha8_54", "fixed_highwr_110", "fixed_aggressive_175"]))].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"])
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    baseline_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_alpha8_54")].iloc[0].to_dict()
    model_path = OUT_DIR / "alpha8_primary_autoreg_dsac_tradepnl.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "selected_variant": selected_variant,
            "state_dim": int(x_train.shape[1]),
            "action_dim": ACTION_DIM,
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "buckets": {
                "tp": TP_BUCKETS,
                "sl": SL_BUCKETS,
                "hold": HOLD_BUCKETS,
                "mult": MULT_BUCKETS,
                "cap": CAP_BUCKETS,
            },
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )
    summary = {
        "model_id": MODEL_ID,
        "design": "Primary-only Alpha8 autoregressive DSAC. Alpha7 Primary owns direction; DSAC sequentially selects veto, TP, SL, hold, notional multiplier, and cap buckets. Reward/label is full realized trade net PnL after cost, with exact entry/exit notional fee accounting and best-bucket behavioral cloning auxiliary loss.",
        "live_wired": False,
        "selection_basis": "2025Q4 validation official Cost3 score; 2026 OOS is reported only.",
        "baseline_alpha8_54_bucket": {
            "tp": 0.200,
            "sl": 5.0,
            "hold": 0.75,
            "mult": 1.20,
            "cap": 7.5,
            "flat_id": int(base54_id),
        },
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_prefix_count": 0,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": ACTION_DIM,
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "samples_per_row": int(args.samples_per_row),
            "reward_label": "full_trade_net_pnl_after_cost",
            "reward_accounting": "qty=entry_notional/entry_fill; exit_notional=qty*exit_fill; net=gross_pnl-entry_fee-exit_fee",
            "dataset_diagnostics": data_diag,
            "train_diag": train_diag,
            "action_usage": {
                "train": _usage(a_train, _active(p_train)),
                "val": _usage(a_val, _active(p_val)),
                "oos": _usage(a_eval, _active(p_eval)),
            },
        },
        "selected": {
            "variant": selected_variant,
            "val": grid[(grid["split"] == "val") & (grid["variant"] == selected_variant)].iloc[0].to_dict(),
            "oos": selected_oos,
            "delta_vs_fixed_alpha8_54_oos_pnl": float(selected_oos["pnl"]) - float(baseline_oos["pnl"]),
        },
        "fixed_alpha8_54_oos": baseline_oos,
        "artifacts": {
            "summary": str(OUT_DIR / "summary.json"),
            "grid": str(grid_path),
            "model": str(model_path),
        },
        "audit": {
            "feature_contract_fail_fast": True,
            "legacy_compat_alias": False,
            "selection_uses_2026": False,
            "official_accounting": "OfficialCost3",
            "reward_official_parity_note": "OfficialCost3 evaluation remains the source of rank truth; reward accounting is stricter on exit_notional fees to avoid bucket-label fee approximation.",
        },
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps({"summary": str(OUT_DIR / "summary.json"), "selected": summary["selected"], "fixed_alpha8_54_oos": baseline_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
