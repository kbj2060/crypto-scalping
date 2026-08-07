#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
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
from mamba_ssm import Mamba

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.sweep_alpha8_origin_scaled_combo_20260529 import OfficialCost3  # noqa: E402
from scripts.train_eval_alpha7_directional_dsac_router_20260529 import (  # noqa: E402
    EVAL_CSV as DEFAULT_EVAL_CSV,
    FORBIDDEN_PREFIXES,
    TRAIN_CSV as DEFAULT_TRAIN_CSV,
    _apply_norm,
    _fit_norm,
)
from scripts.train_eval_alpha8_dsac_iqn_risk_selector_20260529 import _state_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = os.environ.get("ALPHA8_RISK_MODEL_ID", "alpha8_highcap_mamba_seq_dsac_risk_20260529")
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_CSV = Path(os.environ.get("ALPHA8_TRAIN_CSV", str(DEFAULT_TRAIN_CSV)))
EVAL_CSV = Path(os.environ.get("ALPHA8_EVAL_CSV", str(DEFAULT_EVAL_CSV)))

TP_BUCKETS = (0.110, 0.120, 0.175, 0.200, 0.225)
SL_BUCKETS = (4.0, 5.0, 6.0)
HOLD_BUCKETS = (0.50, 0.75, 1.00)
MULT_BUCKETS = (1.00, 1.10, 1.20, 1.35, 1.50, 1.75)
CAP_BUCKETS = (5.0, 7.5, 10.0)
KEEP_ACTION_DIM = len(TP_BUCKETS) * len(SL_BUCKETS) * len(HOLD_BUCKETS) * len(MULT_BUCKETS) * len(CAP_BUCKETS)
VETO_ACTION_ID = KEEP_ACTION_DIM
ACTION_DIM = KEEP_ACTION_DIM + 1


@dataclass(frozen=True)
class RewardMatrix:
    active_idx: np.ndarray
    states: np.ndarray
    rewards_raw: np.ndarray
    rewards_scaled: np.ndarray
    reward_scale: float


def _sequence_states(states: np.ndarray, seq_len: int) -> np.ndarray:
    seq_len = int(max(1, seq_len))
    x = np.asarray(states, dtype=np.float32)
    if seq_len == 1:
        return x[:, None, :]
    pad = np.repeat(x[:1], seq_len - 1, axis=0)
    padded = np.concatenate([pad, x], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_shape=seq_len, axis=0)
    return np.ascontiguousarray(np.transpose(windows, (0, 2, 1))).astype(np.float32)


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


def _action_components_np(action_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a = np.asarray(action_ids, dtype=np.int64)
    veto = (a == VETO_ACTION_ID).astype(np.int64)
    idx = np.clip(a, 0, KEEP_ACTION_DIM - 1)
    cap = idx % len(CAP_BUCKETS)
    idx = idx // len(CAP_BUCKETS)
    mult = idx % len(MULT_BUCKETS)
    idx = idx // len(MULT_BUCKETS)
    hold = idx % len(HOLD_BUCKETS)
    idx = idx // len(HOLD_BUCKETS)
    sl = idx % len(SL_BUCKETS)
    tp = idx // len(SL_BUCKETS)
    return veto, tp, sl, hold, mult, cap


def _action_components_t(action_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    a = action_ids.long()
    veto = torch.where(a == VETO_ACTION_ID, torch.ones_like(a), torch.zeros_like(a))
    idx = torch.clamp(a, min=0, max=KEEP_ACTION_DIM - 1)
    cap = idx % len(CAP_BUCKETS)
    idx = idx // len(CAP_BUCKETS)
    mult = idx % len(MULT_BUCKETS)
    idx = idx // len(MULT_BUCKETS)
    hold = idx % len(HOLD_BUCKETS)
    idx = idx // len(HOLD_BUCKETS)
    sl = idx % len(SL_BUCKETS)
    tp = idx // len(SL_BUCKETS)
    return veto, tp, sl, hold, mult, cap


def _flat_id_tensor(veto: torch.Tensor, tp: torch.Tensor, sl: torch.Tensor, hold: torch.Tensor, mult: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
    keep = 1 + (((tp * len(SL_BUCKETS) + sl) * len(HOLD_BUCKETS) + hold) * len(MULT_BUCKETS) + mult) * len(CAP_BUCKETS) + cap
    keep = keep - 1
    return torch.where(veto == 1, torch.full_like(keep, VETO_ACTION_ID), keep.long())


def _fixed_action_id(tp: float, sl: float, hold: float, mult: float, cap: float) -> int:
    return int(
        (((TP_BUCKETS.index(float(tp)) * len(SL_BUCKETS) + SL_BUCKETS.index(float(sl))) * len(HOLD_BUCKETS) + HOLD_BUCKETS.index(float(hold))) * len(MULT_BUCKETS) + MULT_BUCKETS.index(float(mult))) * len(CAP_BUCKETS)
        + CAP_BUCKETS.index(float(cap))
    )


def _action_name(flat_id: int) -> str:
    if int(flat_id) == VETO_ACTION_ID:
        return "veto"
    _, tp, sl, hold, mult, cap = _action_components_np(np.asarray([flat_id], dtype=np.int64))
    return f"tp{TP_BUCKETS[int(tp[0])]:.3f}_sl{SL_BUCKETS[int(sl[0])]:.1f}_h{HOLD_BUCKETS[int(hold[0])]:.2f}_m{MULT_BUCKETS[int(mult[0])]:.2f}_c{CAP_BUCKETS[int(cap[0])]:.1f}"


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
    if int(flat_id) == VETO_ACTION_ID:
        return _zero_row(row)
    _, tp_i, sl_i, hold_i, mult_i, cap_i = _action_components_np(np.asarray([flat_id], dtype=np.int64))
    out = row.copy()
    base_notional = float(row.get("notional_exposure", 0.0) or 0.0)
    base_leverage = max(float(row.get("leverage", 1.0) or 1.0), 1e-8)
    base_tp = max(float(row.get("take_profit", 0.0) or 0.0), 1e-8)
    base_sl = max(abs(float(row.get("stop_loss", 0.0) or 0.0)), 1e-8)
    base_hold = max(int(row.get("max_hold_bars", 0) or 0), 1)
    notional = float(min(max(base_notional * MULT_BUCKETS[int(mult_i[0])], 0.0), CAP_BUCKETS[int(cap_i[0])]))
    out.loc["notional_exposure"] = notional
    out.loc["position_fraction"] = notional / base_leverage
    out.loc["take_profit"] = base_tp * TP_BUCKETS[int(tp_i[0])]
    out.loc["stop_loss"] = base_sl * SL_BUCKETS[int(sl_i[0])]
    out.loc["max_hold_bars"] = int(max(1, round(base_hold * HOLD_BUCKETS[int(hold_i[0])])))
    return out


def _compose_decisions(dec: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    for i in np.flatnonzero(active):
        out.iloc[int(i)] = _apply_action(out.iloc[int(i)], int(actions[int(i)]))
    out.loc[~active] = out.loc[~active].apply(_zero_row, axis=1)
    return out


def _fixed_decisions(dec: pd.DataFrame, flat_id: int) -> pd.DataFrame:
    return _compose_decisions(dec, np.full(len(dec), int(flat_id), dtype=np.int64))


def _simulate_one(
    *,
    i: int,
    side: int,
    base_notional: float,
    base_tp: float,
    base_sl: float,
    base_hold: int,
    flat_id: int,
    open_px: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    fee_eff: float,
    slip_eff: float,
) -> float:
    if flat_id == VETO_ACTION_ID or side == 0 or base_notional <= 0.0:
        return 0.0
    _, tp_i, sl_i, hold_i, mult_i, cap_i = _action_components_np(np.asarray([flat_id], dtype=np.int64))
    notional = float(min(max(base_notional * MULT_BUCKETS[int(mult_i[0])], 0.0), CAP_BUCKETS[int(cap_i[0])]))
    if notional <= 0.0:
        return 0.0
    entry_i = min(i + 1, len(open_px) - 1)
    entry0 = float(open_px[entry_i])
    if entry0 <= 0.0:
        return 0.0
    entry = entry0 * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
    tp = max(base_tp, 1e-8) * TP_BUCKETS[int(tp_i[0])]
    sl = max(base_sl, 1e-8) * SL_BUCKETS[int(sl_i[0])]
    hold = int(max(1, round(max(base_hold, 1) * HOLD_BUCKETS[int(hold_i[0])])))
    end_i = min(entry_i + hold, len(open_px) - 1)
    exit_fill: float | None = None
    for j in range(entry_i + 1, end_i + 1):
        if side > 0:
            favorable = (float(high[j]) / max(entry, 1e-12) - 1.0) * notional
            adverse = (float(low[j]) / max(entry, 1e-12) - 1.0) * notional
        else:
            favorable = (entry / max(float(low[j]), 1e-12) - 1.0) * notional
            adverse = (entry / max(float(high[j]), 1e-12) - 1.0) * notional
        if adverse <= -sl:
            raw_stop = max(sl / max(notional, 1e-12), 0.0)
            if side > 0:
                exit_fill = entry * max(1.0 - raw_stop, 1e-8) * (1.0 - slip_eff)
            else:
                exit_fill = entry / max(1.0 - raw_stop, 1e-8) * (1.0 + slip_eff)
            break
        if favorable >= tp:
            raw_tp = max(tp / max(notional, 1e-12), 0.0)
            if side > 0:
                exit_fill = entry * (1.0 + raw_tp) * (1.0 - slip_eff)
            else:
                exit_fill = entry / max(1.0 + raw_tp, 1e-8) * (1.0 + slip_eff)
            break
    if exit_fill is None:
        exit_fill = float(close[end_i]) * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    entry_notional = float(notional)
    qty = entry_notional / max(entry, 1e-12)
    exit_notional = qty * max(float(exit_fill), 0.0)
    gross = exit_notional - entry_notional if side > 0 else entry_notional - exit_notional
    return float(gross - fee_eff * entry_notional - fee_eff * exit_notional)


def _build_full_reward_matrix(
    frame: pd.DataFrame,
    states: np.ndarray,
    combo: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_active_rows: int | None,
) -> tuple[RewardMatrix, dict[str, Any]]:
    active_idx = np.flatnonzero(_active(combo) & (np.arange(len(frame)) < len(frame) - 3)).astype(np.int64)
    if max_active_rows is not None and int(max_active_rows) > 0:
        active_idx = active_idx[: int(max_active_rows)]
    open_px = _num(frame, "open")
    high = _num(frame, "high")
    low = _num(frame, "low")
    close = _num(frame, "close")
    side = _num(combo, "side").astype(np.int64)
    base_notional = _num(combo, "notional_exposure")
    base_tp = _num(combo, "take_profit")
    base_sl = np.abs(_num(combo, "stop_loss"))
    base_hold = np.maximum(_num(combo, "max_hold_bars").astype(np.int64), 1)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    rewards = np.zeros((len(active_idx), ACTION_DIM), dtype=np.float32)
    actions = np.arange(ACTION_DIM, dtype=np.int64)
    for row_n, i in enumerate(active_idx):
        for a in actions:
            rewards[row_n, int(a)] = _simulate_one(
                i=int(i),
                side=int(side[int(i)]),
                base_notional=float(base_notional[int(i)]),
                base_tp=float(base_tp[int(i)]),
                base_sl=float(base_sl[int(i)]),
                base_hold=int(base_hold[int(i)]),
                flat_id=int(a),
                open_px=open_px,
                high=high,
                low=low,
                close=close,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
        if (row_n + 1) % 1000 == 0:
            print(json.dumps({"stage": "reward_matrix", "rows_done": int(row_n + 1), "rows_total": int(len(active_idx))}), flush=True)
    scale = float(np.nanstd(rewards))
    if not np.isfinite(scale) or scale < 1e-8:
        scale = 1.0
    scaled = np.clip(rewards / scale, -8.0, 8.0).astype(np.float32)
    best = np.argmax(rewards, axis=1)
    counts = np.bincount(best, minlength=ACTION_DIM)
    top = sorted([(i, int(v)) for i, v in enumerate(counts) if v > 0], key=lambda kv: kv[1], reverse=True)[:12]
    diag = {
        "active_rows": int(len(active_idx)),
        "action_dim": int(ACTION_DIM),
        "counterfactual_count": int(len(active_idx) * ACTION_DIM),
        "reward_scale": float(scale),
        "oracle_top_actions": {_action_name(i): int(v) for i, v in top},
        "reward_raw_mean": float(np.mean(rewards)),
        "reward_raw_std": float(np.std(rewards)),
    }
    return RewardMatrix(active_idx=active_idx, states=states[active_idx].astype(np.float32), rewards_raw=rewards, rewards_scaled=scaled, reward_scale=scale), diag


def _group_slices(columns: list[str]) -> list[tuple[list[int], str]]:
    groups = {
        "primary": [],
        "fallback": [],
        "combo": [],
        "teacher": [],
        "meta": [],
        "market": [],
    }
    teacher = {
        "teacher_long_edge",
        "teacher_short_edge",
        "teacher_side_margin",
        "teacher_side_disagreement",
        "teacher_quantile_skew",
        "teacher_uncertainty",
        "teacher_tail_warning",
    }
    for i, col in enumerate(columns):
        c = str(col)
        if c.startswith("primary_"):
            groups["primary"].append(i)
        elif c.startswith("fallback_"):
            groups["fallback"].append(i)
        elif c.startswith("combo_"):
            groups["combo"].append(i)
        elif c in teacher:
            groups["teacher"].append(i)
        elif any(x in c for x in ("origin_", "agree", "disagree", "minus_")):
            groups["meta"].append(i)
        else:
            groups["market"].append(i)
    out: list[tuple[list[int], str]] = []
    for name in ("market", "teacher", "primary", "fallback", "combo", "meta"):
        idx = groups[name]
        if not idx:
            continue
        out.append((idx, name))
    return out


class GroupTransformerEncoder(nn.Module):
    def __init__(self, state_dim: int, groups: list[tuple[list[int], str]], hidden: int = 256, d_model: int = 32) -> None:
        super().__init__()
        self.group_names = [str(n) for _, n in groups]
        self.group_indices: list[str] = []
        for i, (idx, _) in enumerate(groups):
            name = f"group_idx_{i}"
            self.register_buffer(name, torch.tensor([int(x) for x in idx], dtype=torch.long), persistent=False)
            self.group_indices.append(name)
        self.proj = nn.ModuleList([nn.Sequential(nn.Linear(len(idx), d_model), nn.LayerNorm(d_model), nn.SiLU()) for idx, _ in groups])
        self.raw = nn.Sequential(nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.out = nn.Sequential(
            nn.Linear(hidden + len(self.group_indices) * d_model, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = []
        for name, proj in zip(self.group_indices, self.proj):
            tokens.append(proj(x.index_select(1, getattr(self, name))))
        grouped = torch.cat(tokens, dim=1)
        return self.out(torch.cat([self.raw(x), grouped], dim=1))


class MambaSequenceEncoder(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 96, d_model: int = 96, d_state: int = 16, d_conv: int = 4, expand: int = 2) -> None:
        super().__init__()
        self.input = nn.Sequential(nn.Linear(state_dim, d_model), nn.LayerNorm(d_model), nn.SiLU())
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"MambaSequenceEncoder expects [B,T,D] or [B,D], got shape={tuple(x.shape)}")
        y = self.mamba(self.input(x))
        return self.out(y[:, -1, :])


class LastVetoActor(nn.Module):
    def __init__(self, state_dim: int, groups: list[tuple[int, int, str]], hidden: int = 256) -> None:
        super().__init__()
        self.feat = MambaSequenceEncoder(state_dim, hidden=hidden, d_model=hidden)
        self.tp_head = nn.Linear(hidden, len(TP_BUCKETS))
        self.sl_head = nn.Linear(hidden + len(TP_BUCKETS), len(SL_BUCKETS))
        self.hold_head = nn.Linear(hidden + len(TP_BUCKETS) + len(SL_BUCKETS), len(HOLD_BUCKETS))
        self.mult_head = nn.Linear(hidden + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS), len(MULT_BUCKETS))
        self.cap_head = nn.Linear(hidden + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS) + len(MULT_BUCKETS), len(CAP_BUCKETS))
        self.veto_head = nn.Linear(hidden + len(TP_BUCKETS) + len(SL_BUCKETS) + len(HOLD_BUCKETS) + len(MULT_BUCKETS) + len(CAP_BUCKETS), 2)

    @staticmethod
    def _oh(idx: torch.Tensor, n: int) -> torch.Tensor:
        return F.one_hot(idx.clamp(0, n - 1), num_classes=n).float()

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.feat(x)
        logp = torch.zeros(len(x), device=x.device)
        tp_d = torch.distributions.Categorical(logits=self.tp_head(h))
        tp = tp_d.sample()
        logp = logp + tp_d.log_prob(tp)
        tp_oh = self._oh(tp, len(TP_BUCKETS))
        sl_d = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, tp_oh], dim=-1)))
        sl = sl_d.sample()
        logp = logp + sl_d.log_prob(sl)
        sl_oh = self._oh(sl, len(SL_BUCKETS))
        hold_d = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, tp_oh, sl_oh], dim=-1)))
        hold = hold_d.sample()
        logp = logp + hold_d.log_prob(hold)
        hold_oh = self._oh(hold, len(HOLD_BUCKETS))
        mult_d = torch.distributions.Categorical(logits=self.mult_head(torch.cat([h, tp_oh, sl_oh, hold_oh], dim=-1)))
        mult = mult_d.sample()
        logp = logp + mult_d.log_prob(mult)
        mult_oh = self._oh(mult, len(MULT_BUCKETS))
        cap_d = torch.distributions.Categorical(logits=self.cap_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)))
        cap = cap_d.sample()
        logp = logp + cap_d.log_prob(cap)
        cap_oh = self._oh(cap, len(CAP_BUCKETS))
        veto_d = torch.distributions.Categorical(logits=self.veto_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh, cap_oh], dim=-1)))
        veto = veto_d.sample()
        logp = logp + veto_d.log_prob(veto)
        return _flat_id_tensor(veto, tp, sl, hold, mult, cap), logp

    def log_prob_for_flat(self, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        h = self.feat(x)
        veto, tp, sl, hold, mult, cap = _action_components_t(flat)
        logp = torch.distributions.Categorical(logits=self.tp_head(h)).log_prob(tp)
        tp_oh = self._oh(tp, len(TP_BUCKETS))
        sl_d = torch.distributions.Categorical(logits=self.sl_head(torch.cat([h, tp_oh], dim=-1)))
        logp = logp + sl_d.log_prob(sl)
        sl_oh = self._oh(sl, len(SL_BUCKETS))
        hold_d = torch.distributions.Categorical(logits=self.hold_head(torch.cat([h, tp_oh, sl_oh], dim=-1)))
        logp = logp + hold_d.log_prob(hold)
        hold_oh = self._oh(hold, len(HOLD_BUCKETS))
        mult_d = torch.distributions.Categorical(logits=self.mult_head(torch.cat([h, tp_oh, sl_oh, hold_oh], dim=-1)))
        logp = logp + mult_d.log_prob(mult)
        mult_oh = self._oh(mult, len(MULT_BUCKETS))
        cap_d = torch.distributions.Categorical(logits=self.cap_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)))
        logp = logp + cap_d.log_prob(cap)
        cap_oh = self._oh(cap, len(CAP_BUCKETS))
        veto_d = torch.distributions.Categorical(logits=self.veto_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh, cap_oh], dim=-1)))
        return logp + veto_d.log_prob(veto)

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feat(x)
        tp = torch.argmax(self.tp_head(h), dim=-1)
        tp_oh = self._oh(tp, len(TP_BUCKETS))
        sl = torch.argmax(self.sl_head(torch.cat([h, tp_oh], dim=-1)), dim=-1)
        sl_oh = self._oh(sl, len(SL_BUCKETS))
        hold = torch.argmax(self.hold_head(torch.cat([h, tp_oh, sl_oh], dim=-1)), dim=-1)
        hold_oh = self._oh(hold, len(HOLD_BUCKETS))
        mult = torch.argmax(self.mult_head(torch.cat([h, tp_oh, sl_oh, hold_oh], dim=-1)), dim=-1)
        mult_oh = self._oh(mult, len(MULT_BUCKETS))
        cap = torch.argmax(self.cap_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh], dim=-1)), dim=-1)
        cap_oh = self._oh(cap, len(CAP_BUCKETS))
        veto = torch.argmax(self.veto_head(torch.cat([h, tp_oh, sl_oh, hold_oh, mult_oh, cap_oh], dim=-1)), dim=-1)
        return _flat_id_tensor(veto, tp, sl, hold, mult, cap)


class QuantileActionCritic(nn.Module):
    def __init__(self, state_dim: int, groups: list[tuple[int, int, str]], hidden: int = 256, emb: int = 12, n_quantiles: int = 32) -> None:
        super().__init__()
        self.state = MambaSequenceEncoder(state_dim, hidden=hidden, d_model=hidden)
        self.tp_emb = nn.Embedding(len(TP_BUCKETS), emb)
        self.sl_emb = nn.Embedding(len(SL_BUCKETS), emb)
        self.hold_emb = nn.Embedding(len(HOLD_BUCKETS), emb)
        self.mult_emb = nn.Embedding(len(MULT_BUCKETS), emb)
        self.cap_emb = nn.Embedding(len(CAP_BUCKETS), emb)
        self.veto_emb = nn.Embedding(2, emb)
        self.action = nn.Sequential(nn.Linear(emb * 6, hidden), nn.LayerNorm(hidden), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.SiLU(), nn.Linear(hidden, n_quantiles))

    def _action_emb(self, flat: torch.Tensor) -> torch.Tensor:
        veto, tp, sl, hold, mult, cap = _action_components_t(flat)
        return self.action(torch.cat([self.tp_emb(tp), self.sl_emb(sl), self.hold_emb(hold), self.mult_emb(mult), self.cap_emb(cap), self.veto_emb(veto)], dim=-1))

    def forward(self, state: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat([self.state(state), self._action_emb(flat)], dim=-1))


def _quantile_huber(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    td = target.unsqueeze(1) - pred
    abs_td = td.abs()
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    weight = (taus.view(1, -1) - (td.detach() < 0.0).float()).abs()
    return (weight * huber).mean()


def _topk_weighted_bc_loss(actor: LastVetoActor, states: torch.Tensor, rewards: torch.Tensor, *, top_k: int, temp: float) -> torch.Tensor:
    vals, idx = torch.topk(rewards, k=int(top_k), dim=1)
    weights = torch.softmax(vals / max(float(temp), 1e-6), dim=1)
    flat_states = states.repeat_interleave(int(top_k), dim=0)
    flat_actions = idx.reshape(-1)
    logp = actor.log_prob_for_flat(flat_states, flat_actions).view(len(states), int(top_k))
    return -(weights.detach() * logp).sum(dim=1).mean()


def _train_distributional(
    matrix: RewardMatrix,
    *,
    state_dim: int,
    groups: list[tuple[int, int, str]],
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    n_quantiles: int,
    awac_coef: float,
    cvar_coef: float,
    entropy_coef: float,
    top_k: int,
    awac_temp: float,
    hidden: int,
) -> tuple[LastVetoActor, QuantileActionCritic, QuantileActionCritic, dict[str, Any]]:
    actor = LastVetoActor(state_dim, groups, hidden=int(hidden)).to(device)
    q1 = QuantileActionCritic(state_dim, groups, hidden=int(hidden), n_quantiles=n_quantiles).to(device)
    q2 = QuantileActionCritic(state_dim, groups, hidden=int(hidden), n_quantiles=n_quantiles).to(device)
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=lr, weight_decay=1e-5)
    states = torch.from_numpy(matrix.states).to(device)
    rewards = torch.from_numpy(matrix.rewards_scaled).to(device)
    taus = torch.linspace(0.5 / n_quantiles, 1.0 - 0.5 / n_quantiles, n_quantiles, device=device)
    n_rows = int(states.shape[0])
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        row = torch.randint(0, n_rows, (int(batch_size),), device=device)
        act = torch.randint(0, ACTION_DIM, (int(batch_size),), device=device)
        s = states[row]
        target = rewards[row, act]
        q1_pred = q1(s, act)
        q2_pred = q2(s, act)
        critic_loss = _quantile_huber(q1_pred, target, taus) + _quantile_huber(q2_pred, target, taus)
        opt_q.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 1.0)
        opt_q.step()

        a_samp, logp = actor.sample(s)
        qa1 = q1(s, a_samp)
        qa2 = q2(s, a_samp)
        qmin = torch.minimum(qa1, qa2)
        k = max(1, int(n_quantiles * 0.25))
        q_cvar = torch.sort(qmin, dim=1).values[:, :k].mean(dim=1)
        awac = _topk_weighted_bc_loss(actor, s, rewards[row], top_k=int(top_k), temp=float(awac_temp))
        actor_loss = float(awac_coef) * awac - float(cvar_coef) * q_cvar.mean() + float(entropy_coef) * logp.mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
        opt_actor.step()
        if step % 250 == 0:
            last = {
                "step": int(step),
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "awac_loss": float(awac.detach().cpu()),
                "q_cvar": float(q_cvar.mean().detach().cpu()),
                "entropy": float((-logp).mean().detach().cpu()),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "dist_dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), q1.cpu(), q2.cpu(), last


def _actor_actions(actor: LastVetoActor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 4096):
            x = torch.from_numpy(states[start : start + 4096]).to(device)
            out.append(actor.greedy(x).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _critic_actions(q1: QuantileActionCritic, q2: QuantileActionCritic, states: np.ndarray, *, device: torch.device, cvar_frac: float, action_chunk: int = 128) -> np.ndarray:
    q1 = q1.to(device).eval()
    q2 = q2.to(device).eval()
    out: list[np.ndarray] = []
    all_actions = torch.arange(ACTION_DIM, device=device, dtype=torch.long)
    with torch.no_grad():
        for start in range(0, len(states), 512):
            x = torch.from_numpy(states[start : start + 512]).to(device)
            best_score = torch.full((len(x),), -1e9, device=device)
            best_action = torch.zeros((len(x),), dtype=torch.long, device=device)
            for a0 in range(0, ACTION_DIM, int(action_chunk)):
                acts = all_actions[a0 : a0 + int(action_chunk)]
                xx = x.repeat_interleave(len(acts), dim=0)
                aa = acts.repeat(len(x))
                q = torch.minimum(q1(xx, aa), q2(xx, aa))
                q_sorted = torch.sort(q, dim=1).values
                k = max(1, int(q.shape[1] * float(cvar_frac)))
                score = q_sorted[:, :k].mean(dim=1).view(len(x), len(acts))
                vals, idx = score.max(dim=1)
                upd = vals > best_score
                best_score = torch.where(upd, vals, best_score)
                best_action = torch.where(upd, acts[idx], best_action)
            out.append(best_action.cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    counts: dict[int, int] = {}
    for a in actions[np.asarray(active, dtype=bool)]:
        counts[int(a)] = counts.get(int(a), 0) + 1
    return {_action_name(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:20]}


def _metrics_rows(evaluator: OfficialCost3, splits: list[tuple[str, pd.DataFrame, dict[str, pd.DataFrame]]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, frame, variants in splits:
        for variant, dec in variants.items():
            rows.append({"split": split, "variant": variant, **evaluator(frame, dec)})
    return pd.DataFrame(rows)


def _limit_eval(frame: pd.DataFrame, variants: dict[str, pd.DataFrame], max_rows: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    if int(max_rows) <= 0:
        return frame, variants
    n = min(int(max_rows), len(frame))
    return frame.iloc[:n].reset_index(drop=True), {k: v.iloc[:n].reset_index(drop=True) for k, v in variants.items()}


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) or 0.0) + 130.0 * float(row.get("wr", 0.0) or 0.0) - 0.45 * abs(float(row.get("mdd", 0.0) or 0.0)) + 0.015 * trades


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--seq-len", type=int, default=24)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--n-quantiles", type=int, default=32)
    ap.add_argument("--awac-coef", type=float, default=0.35)
    ap.add_argument("--cvar-coef", type=float, default=1.0)
    ap.add_argument("--entropy-coef", type=float, default=0.015)
    ap.add_argument("--top-k", type=int, default=8)
    ap.add_argument("--awac-temp", type=float, default=0.18)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--max-active-rows", type=int, default=0)
    ap.add_argument("--eval-max-rows", type=int, default=0)
    ap.add_argument("--include-critic-eval", action="store_true")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    _seed_everything(290531)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    baseline = get_live_baseline()
    train_all = _read(TRAIN_CSV)
    eval_df = _read(EVAL_CSV)
    _assert_clean(train_all, name="train_all")
    _assert_clean(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary = joblib.load(baseline.primary_parent)
    fallback = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    p_train = _predict_scaled(primary, train_df, primary_rt).reset_index(drop=True)
    p_val = _predict_scaled(primary, val_df, primary_rt).reset_index(drop=True)
    p_eval = _predict_scaled(primary, eval_df, primary_rt).reset_index(drop=True)
    f_train = _predict_scaled(fallback, train_df, fallback_rt).reset_index(drop=True)
    f_val = _predict_scaled(fallback, val_df, fallback_rt).reset_index(drop=True)
    f_eval = _predict_scaled(fallback, eval_df, fallback_rt).reset_index(drop=True)
    combo_train = _combine_primary_fallback(p_train, f_train).reset_index(drop=True)
    combo_val = _combine_primary_fallback(p_val, f_val).reset_index(drop=True)
    combo_eval = _combine_primary_fallback(p_eval, f_eval).reset_index(drop=True)

    s_train = _state_frame(train_df, p_train, f_train, combo_train)
    s_val = _state_frame(val_df, p_val, f_val, combo_val)
    s_eval = _state_frame(eval_df, p_eval, f_eval, combo_eval)
    norm = _fit_norm(s_train)
    x_train = _apply_norm(s_train, norm)
    x_val = _apply_norm(s_val, norm)
    x_eval = _apply_norm(s_eval, norm)
    x_train_seq = _sequence_states(x_train, int(args.seq_len))
    x_val_seq = _sequence_states(x_val, int(args.seq_len))
    x_eval_seq = _sequence_states(x_eval, int(args.seq_len))
    groups = _group_slices(list(norm["columns"]))

    evaluator = OfficialCost3()
    matrix, data_diag = _build_full_reward_matrix(
        train_df,
        x_train_seq,
        combo_train,
        fee=float(evaluator.fee),
        slip=float(evaluator.slip),
        cost_mult=float(args.cost_mult),
        max_active_rows=int(args.max_active_rows) if int(args.max_active_rows) > 0 else None,
    )
    print(
        json.dumps(
            {
                "stage": "train_start",
                "model_id": MODEL_ID,
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "seq_len": int(args.seq_len),
                "action_dim": int(ACTION_DIM),
                "groups": groups,
                "train_rows": int(len(train_df)),
                "val_rows": int(len(val_df)),
                "oos_rows": int(len(eval_df)),
                "data_diag": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    actor, q1, q2, train_diag = _train_distributional(
        matrix,
        state_dim=int(x_train.shape[1]),
        groups=groups,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        n_quantiles=int(args.n_quantiles),
        awac_coef=float(args.awac_coef),
        cvar_coef=float(args.cvar_coef),
        entropy_coef=float(args.entropy_coef),
        top_k=int(args.top_k),
        awac_temp=float(args.awac_temp),
        hidden=int(args.hidden),
    )

    eval_n_train = len(train_df) if int(args.eval_max_rows) <= 0 else min(int(args.eval_max_rows), len(train_df))
    eval_n_val = len(val_df) if int(args.eval_max_rows) <= 0 else min(int(args.eval_max_rows), len(val_df))
    eval_n_oos = len(eval_df) if int(args.eval_max_rows) <= 0 else min(int(args.eval_max_rows), len(eval_df))
    train_df_eval = train_df.iloc[:eval_n_train].reset_index(drop=True)
    val_df_eval = val_df.iloc[:eval_n_val].reset_index(drop=True)
    eval_df_eval = eval_df.iloc[:eval_n_oos].reset_index(drop=True)
    combo_train_eval = combo_train.iloc[:eval_n_train].reset_index(drop=True)
    combo_val_eval = combo_val.iloc[:eval_n_val].reset_index(drop=True)
    combo_oos_eval = combo_eval.iloc[:eval_n_oos].reset_index(drop=True)

    actor_train = _actor_actions(actor, x_train_seq[:eval_n_train], device=device)
    actor_val = _actor_actions(actor, x_val_seq[:eval_n_val], device=device)
    actor_eval = _actor_actions(actor, x_eval_seq[:eval_n_oos], device=device)

    fixed52_id = _fixed_action_id(0.200, 5.0, 0.75, 1.10, 7.5)
    fixed54_id = _fixed_action_id(0.200, 5.0, 0.75, 1.20, 7.5)
    fixed55_id = _fixed_action_id(0.200, 5.0, 0.75, 1.20, 10.0)
    fixed60_id = _fixed_action_id(0.200, 5.0, 0.75, 1.75, 7.5)
    variants = {
        "baseline_combo": (combo_train_eval, combo_val_eval, combo_oos_eval),
        "fixed_52_highwr": (_fixed_decisions(combo_train_eval, fixed52_id), _fixed_decisions(combo_val_eval, fixed52_id), _fixed_decisions(combo_oos_eval, fixed52_id)),
        "fixed_54_highcap": (_fixed_decisions(combo_train_eval, fixed54_id), _fixed_decisions(combo_val_eval, fixed54_id), _fixed_decisions(combo_oos_eval, fixed54_id)),
        "fixed_55_highcap": (_fixed_decisions(combo_train_eval, fixed55_id), _fixed_decisions(combo_val_eval, fixed55_id), _fixed_decisions(combo_oos_eval, fixed55_id)),
        "fixed_60_aggressive": (_fixed_decisions(combo_train_eval, fixed60_id), _fixed_decisions(combo_val_eval, fixed60_id), _fixed_decisions(combo_oos_eval, fixed60_id)),
        "dist_actor_greedy": (_compose_decisions(combo_train_eval, actor_train), _compose_decisions(combo_val_eval, actor_val), _compose_decisions(combo_oos_eval, actor_eval)),
    }
    action_usage = {
        "actor_greedy": {"train": _usage(actor_train, _active(combo_train_eval)), "val": _usage(actor_val, _active(combo_val_eval)), "oos": _usage(actor_eval, _active(combo_oos_eval))}
    }
    if bool(args.include_critic_eval):
        cvar25_train = _critic_actions(q1, q2, x_train_seq[:eval_n_train], device=device, cvar_frac=0.25)
        cvar25_val = _critic_actions(q1, q2, x_val_seq[:eval_n_val], device=device, cvar_frac=0.25)
        cvar25_eval = _critic_actions(q1, q2, x_eval_seq[:eval_n_oos], device=device, cvar_frac=0.25)
        cvar35_train = _critic_actions(q1, q2, x_train_seq[:eval_n_train], device=device, cvar_frac=0.35)
        cvar35_val = _critic_actions(q1, q2, x_val_seq[:eval_n_val], device=device, cvar_frac=0.35)
        cvar35_eval = _critic_actions(q1, q2, x_eval_seq[:eval_n_oos], device=device, cvar_frac=0.35)
        mean_train = _critic_actions(q1, q2, x_train_seq[:eval_n_train], device=device, cvar_frac=1.0)
        mean_val = _critic_actions(q1, q2, x_val_seq[:eval_n_val], device=device, cvar_frac=1.0)
        mean_eval = _critic_actions(q1, q2, x_eval_seq[:eval_n_oos], device=device, cvar_frac=1.0)
        variants.update(
            {
                "dist_critic_cvar25": (_compose_decisions(combo_train_eval, cvar25_train), _compose_decisions(combo_val_eval, cvar25_val), _compose_decisions(combo_oos_eval, cvar25_eval)),
                "dist_critic_cvar35": (_compose_decisions(combo_train_eval, cvar35_train), _compose_decisions(combo_val_eval, cvar35_val), _compose_decisions(combo_oos_eval, cvar35_eval)),
                "dist_critic_mean": (_compose_decisions(combo_train_eval, mean_train), _compose_decisions(combo_val_eval, mean_val), _compose_decisions(combo_oos_eval, mean_eval)),
            }
        )
        action_usage.update(
            {
                "critic_cvar25": {"train": _usage(cvar25_train, _active(combo_train_eval)), "val": _usage(cvar25_val, _active(combo_val_eval)), "oos": _usage(cvar25_eval, _active(combo_oos_eval))},
                "critic_cvar35": {"train": _usage(cvar35_train, _active(combo_train_eval)), "val": _usage(cvar35_val, _active(combo_val_eval)), "oos": _usage(cvar35_eval, _active(combo_oos_eval))},
                "critic_mean": {"train": _usage(mean_train, _active(combo_train_eval)), "val": _usage(mean_val, _active(combo_val_eval)), "oos": _usage(mean_eval, _active(combo_oos_eval))},
            }
        )
    grid = _metrics_rows(
        evaluator,
        [
            ("train", train_df_eval, {k: v[0] for k, v in variants.items()}),
            ("val", val_df_eval, {k: v[1] for k, v in variants.items()}),
            ("oos", eval_df_eval, {k: v[2] for k, v in variants.items()}),
        ],
    )
    grid["selection_score"] = grid.apply(_score, axis=1)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)
    val_rank = grid[(grid["split"] == "val") & (grid["variant"] != "baseline_combo")].sort_values("selection_score", ascending=False)
    selected_variant = str(val_rank.iloc[0]["variant"])
    selected_oos = grid[(grid["split"] == "oos") & (grid["variant"] == selected_variant)].iloc[0].to_dict()
    fixed54_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_54_highcap")].iloc[0].to_dict()
    model_path = OUT_DIR / f"{MODEL_ID}.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "selected_variant": selected_variant,
            "state_dim": int(x_train.shape[1]),
            "seq_len": int(args.seq_len),
            "action_dim": int(ACTION_DIM),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "groups": groups,
            "buckets": {"tp": TP_BUCKETS, "sl": SL_BUCKETS, "hold": HOLD_BUCKETS, "mult": MULT_BUCKETS, "cap": CAP_BUCKETS},
            "actor_state_dict": actor.state_dict(),
            "q1_state_dict": q1.state_dict(),
            "q2_state_dict": q2.state_dict(),
        },
        model_path,
    )
    summary = {
        "model_id": MODEL_ID,
        "design": "Alpha8 high-cap combo direction unchanged. Mamba sequence encoder plus distributional action-embedding twin critic and last-veto autoregressive actor select only TP/SL/hold/mult/cap/veto risk buckets.",
        "live_wired": False,
        "selection_basis": "2025Q4 validation official Cost3 score; 2026 OOS is reported only.",
        "baseline_model_id": baseline.model_id,
        "allowed_regime_surfaces": ["clean_regime4_state24_sticky090_v2_*", "regime4_pred_*"],
        "forbidden_prefixes": list(FORBIDDEN_PREFIXES),
        "forbidden_prefix_count": 0,
        "fixed_54_highcap_bucket": {"tp": 0.2, "sl": 5.0, "hold": 0.75, "mult": 1.2, "cap": 7.5},
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "hidden": int(args.hidden),
            "seq_len": int(args.seq_len),
            "encoder": "mamba_ssm.Mamba",
            "n_quantiles": int(args.n_quantiles),
            "include_critic_eval": bool(args.include_critic_eval),
            "eval_max_rows": int(args.eval_max_rows),
            "dataset_diagnostics": data_diag,
            "train_diag": train_diag,
            "action_usage": action_usage,
            "reward_label": "full_counterfactual_trade_net_pnl_after_exact_entry_exit_notional_cost",
        },
        "selected": {
            "variant": selected_variant,
            "val": grid[(grid["split"] == "val") & (grid["variant"] == selected_variant)].iloc[0].to_dict(),
            "oos": selected_oos,
            "delta_vs_fixed_54_highcap_oos_pnl": float(selected_oos["pnl"]) - float(fixed54_oos["pnl"]),
        },
        "fixed_54_highcap_oos": fixed54_oos,
        "artifacts": {"summary": str(OUT_DIR / "summary.json"), "grid": str(grid_path), "model": str(model_path)},
        "audit": {"feature_contract_fail_fast": True, "legacy_contract_layer": False, "selection_uses_2026": False, "official_accounting": "OfficialCost3"},
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n")
    print(json.dumps({"summary": str(summary_path), "selected": summary["selected"], "fixed_54_highcap_oos": fixed54_oos}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
