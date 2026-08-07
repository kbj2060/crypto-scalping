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
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import _json_default  # noqa: E402
from scripts.eval_omega1_regime3_expertdq_risk_replay_20260602 import ACTIVE_SCALES, ACTIVE_TEMPLATE  # noqa: E402
from scripts.train_eval_omega1_expertdq_dsac_risk_allocator_20260602 import (  # noqa: E402
    ACTION_CASH,
    OUT_DIR as RISK_OUT_DIR,
    _active,
    _apply_norm,
    _build_state_frame,
    _fit_norm,
    _load_variant_frames,
    _num,
    _numeric_feature_cols,
    _to_decisions,
    _zero_row,
)


MODEL_ID = "omega1_expertdq_dsac_risk4_factored_notime_constrained_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

NOTIONAL_BUCKETS = (0.35, 0.45, 0.65, 0.80)
LEVERAGE_BUCKETS = (1.5, 2.0, 3.0)
TP_BUCKETS = (0.018, 0.026, 0.040, 0.055)
SL_BUCKETS = (0.010, 0.014, 0.022, 0.035)


@dataclass(frozen=True)
class Risk4Spec:
    flat_id: int
    veto: int
    notional_id: int
    leverage_id: int
    tp_id: int
    sl_id: int


def _build_action_space() -> tuple[list[Risk4Spec], dict[tuple[int, int, int, int, int], int]]:
    specs = [Risk4Spec(0, 1, 0, 0, 0, 0)]
    lookup = {(1, 0, 0, 0, 0): 0}
    flat = 1
    for n_i in range(len(NOTIONAL_BUCKETS)):
        for l_i in range(len(LEVERAGE_BUCKETS)):
            for tp_i in range(len(TP_BUCKETS)):
                for sl_i in range(len(SL_BUCKETS)):
                    key = (0, n_i, l_i, tp_i, sl_i)
                    specs.append(Risk4Spec(flat, *key))
                    lookup[key] = flat
                    flat += 1
    return specs, lookup


ACTION_SPECS, ACTION_LOOKUP = _build_action_space()
ACTION_DIM = len(ACTION_SPECS)


def _is_valid_action_id(flat_id: int) -> bool:
    spec = ACTION_SPECS[int(flat_id)]
    if spec.veto:
        return True
    # Avoid liquidation-prone combinations under high leverage.
    if spec.leverage_id == 2 and spec.sl_id >= 2:
        return False
    if float(TP_BUCKETS[spec.tp_id]) / max(float(SL_BUCKETS[spec.sl_id]), 1e-12) > 3.5:
        return False
    if float(NOTIONAL_BUCKETS[spec.notional_id]) * float(LEVERAGE_BUCKETS[spec.leverage_id]) > 2.0:
        return False
    return True


VALID_ACTION_IDS = np.asarray([i for i in range(ACTION_DIM) if _is_valid_action_id(i)], dtype=np.int64)
VALID_ACTION_MASK = torch.tensor([_is_valid_action_id(i) for i in range(ACTION_DIM)], dtype=torch.bool)


@dataclass
class OfflineDataset:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    best_actions: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _quantile_huber_loss(
    pred_q: torch.Tensor,
    target_q: torch.Tensor,
    taus: torch.Tensor,
    kappa: float = 1.0,
) -> torch.Tensor:
    td = target_q.unsqueeze(1) - pred_q.unsqueeze(2)
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    tau = taus.view(1, -1, 1)
    weight = (tau - (td.detach() < 0).float()).abs()
    return (weight * huber / kappa).mean()


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
    out.loc["max_hold_bars"] = 0
    out.loc["cooldown_bars"] = 0
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
    end_i = len(frame) - 1
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


def _nearest_idx(value: float, buckets: tuple[float, ...]) -> int:
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
    )
    return ACTION_LOOKUP[key]


def _fast_replay_metrics(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
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
        next_allowed = max(i + 1, int(meta.get("exit_i", i)))
    return {"pnl": float(equity), "mdd": float(mdd), "trades": int(trades), "wr": float(wins / trades) if trades else 0.0}


def _flat_id_tensor(veto: torch.Tensor, n: torch.Tensor, lev: torch.Tensor, tp: torch.Tensor, sl: torch.Tensor) -> torch.Tensor:
    keep_id = 1 + (((n * len(LEVERAGE_BUCKETS) + lev) * len(TP_BUCKETS) + tp) * len(SL_BUCKETS) + sl)
    return torch.where(veto == 1, torch.zeros_like(keep_id), keep_id.long())


def _components_from_flat(flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flat = flat.long()
    veto = torch.where(flat == 0, torch.ones_like(flat), torch.zeros_like(flat))
    idx = torch.clamp(flat - 1, min=0)
    sl = idx % len(SL_BUCKETS)
    idx = idx // len(SL_BUCKETS)
    tp = idx % len(TP_BUCKETS)
    idx = idx // len(TP_BUCKETS)
    lev = idx % len(LEVERAGE_BUCKETS)
    n = idx // len(LEVERAGE_BUCKETS)
    return veto, n, lev, tp, sl


class Risk4Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, cond_dim: int = 16) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.veto_head = nn.Linear(hidden, 2)
        self.veto_emb = nn.Embedding(2, cond_dim)
        self.notional_head = nn.Linear(hidden + cond_dim, len(NOTIONAL_BUCKETS))
        self.notional_emb = nn.Embedding(len(NOTIONAL_BUCKETS), cond_dim)
        self.leverage_head = nn.Linear(hidden + cond_dim, len(LEVERAGE_BUCKETS))
        self.leverage_emb = nn.Embedding(len(LEVERAGE_BUCKETS), cond_dim)
        self.tp_head = nn.Linear(hidden + cond_dim, len(TP_BUCKETS))
        self.tp_emb = nn.Embedding(len(TP_BUCKETS), cond_dim)
        self.sl_head = nn.Linear(hidden + cond_dim, len(SL_BUCKETS))
        lev_sl = torch.ones(len(LEVERAGE_BUCKETS), len(SL_BUCKETS), dtype=torch.bool)
        lev_sl[2, 2:] = False
        exposure = torch.ones(len(NOTIONAL_BUCKETS), len(LEVERAGE_BUCKETS), dtype=torch.bool)
        for n_i, n_value in enumerate(NOTIONAL_BUCKETS):
            for lev_i, lev_value in enumerate(LEVERAGE_BUCKETS):
                if float(n_value) * float(lev_value) > 2.0:
                    exposure[n_i, lev_i] = False
        tp_sl = torch.ones(len(TP_BUCKETS), len(SL_BUCKETS), dtype=torch.bool)
        for tp_i, tp_value in enumerate(TP_BUCKETS):
            for sl_i, sl_value in enumerate(SL_BUCKETS):
                if float(tp_value) / max(float(sl_value), 1e-12) > 3.5:
                    tp_sl[tp_i, sl_i] = False
        self.register_buffer("lev_sl_mask", lev_sl)
        self.register_buffer("exposure_mask", exposure)
        self.register_buffer("tp_sl_mask", tp_sl)
        self.register_buffer("valid_flat_mask", VALID_ACTION_MASK.clone())

    def _decode(
        self,
        x: torch.Tensor,
        *,
        greedy: bool,
        teacher: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        h = self.backbone(x)
        logp = torch.zeros(len(x), device=x.device)
        out: dict[str, torch.Tensor] = {}

        t_comp = _components_from_flat(teacher) if teacher is not None else None
        veto_logits = self.veto_head(h)
        veto_dist = torch.distributions.Categorical(logits=veto_logits)
        veto = t_comp[0] if t_comp is not None else (torch.argmax(veto_logits, dim=-1) if greedy else veto_dist.sample())
        logp = logp + veto_dist.log_prob(veto)
        out["veto_logits"] = veto_logits
        out["veto"] = veto
        keep = veto == 0

        n_logits = self.notional_head(torch.cat([h, self.veto_emb(veto)], dim=-1))
        n_dist = torch.distributions.Categorical(logits=n_logits)
        n = t_comp[1] if t_comp is not None else (torch.argmax(n_logits, dim=-1) if greedy else n_dist.sample())
        logp = logp + torch.where(keep, n_dist.log_prob(n), torch.zeros_like(logp))
        out["notional_logits"] = n_logits
        out["notional"] = n

        lev_logits = self.leverage_head(torch.cat([h, self.notional_emb(n)], dim=-1))
        lev_logits = lev_logits.masked_fill(~self.exposure_mask[n], -1e9)
        lev_dist = torch.distributions.Categorical(logits=lev_logits)
        lev = t_comp[2] if t_comp is not None else (torch.argmax(lev_logits, dim=-1) if greedy else lev_dist.sample())
        logp = logp + torch.where(keep, lev_dist.log_prob(lev), torch.zeros_like(logp))
        out["leverage_logits"] = lev_logits
        out["leverage"] = lev

        tp_logits = self.tp_head(torch.cat([h, self.leverage_emb(lev)], dim=-1))
        tp_dist = torch.distributions.Categorical(logits=tp_logits)
        tp = t_comp[3] if t_comp is not None else (torch.argmax(tp_logits, dim=-1) if greedy else tp_dist.sample())
        logp = logp + torch.where(keep, tp_dist.log_prob(tp), torch.zeros_like(logp))
        out["tp_logits"] = tp_logits
        out["tp"] = tp

        sl_logits = self.sl_head(torch.cat([h, self.tp_emb(tp)], dim=-1))
        sl_valid = self.lev_sl_mask[lev] & self.tp_sl_mask[tp]
        sl_logits = sl_logits.masked_fill(~sl_valid, -1e9)
        sl_dist = torch.distributions.Categorical(logits=sl_logits)
        sl = t_comp[4] if t_comp is not None else (torch.argmax(sl_logits, dim=-1) if greedy else sl_dist.sample())
        logp = logp + torch.where(keep, sl_dist.log_prob(sl), torch.zeros_like(logp))
        out["sl_logits"] = sl_logits
        out["sl"] = sl

        flat = _flat_id_tensor(veto, n, lev, tp, sl)
        return flat, logp, out

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # Compatibility view for diagnostics/BC: assemble independent logits into flat action scores.
        h = self.backbone(x)
        veto_logits = self.veto_head(h)
        scores = torch.full((len(x), ACTION_DIM), -1e9, device=x.device)
        scores[:, 0] = veto_logits[:, 1]
        no_veto = torch.zeros(len(x), dtype=torch.long, device=x.device)
        n_logits = self.notional_head(torch.cat([h, self.veto_emb(no_veto)], dim=-1))
        for n_i in range(len(NOTIONAL_BUCKETS)):
            n = torch.full((len(x),), n_i, dtype=torch.long, device=x.device)
            lev_logits = self.leverage_head(torch.cat([h, self.notional_emb(n)], dim=-1))
            lev_logits = lev_logits.masked_fill(~self.exposure_mask[n], -1e9)
            for lev_i in range(len(LEVERAGE_BUCKETS)):
                lev = torch.full((len(x),), lev_i, dtype=torch.long, device=x.device)
                tp_logits = self.tp_head(torch.cat([h, self.leverage_emb(lev)], dim=-1))
                for tp_i in range(len(TP_BUCKETS)):
                    tp = torch.full((len(x),), tp_i, dtype=torch.long, device=x.device)
                    sl_logits = self.sl_head(torch.cat([h, self.tp_emb(tp)], dim=-1))
                    sl_logits = sl_logits.masked_fill(~(self.lev_sl_mask[lev] & self.tp_sl_mask[tp]), -1e9)
                    for sl_i in range(len(SL_BUCKETS)):
                        flat = ACTION_LOOKUP[(0, n_i, lev_i, tp_i, sl_i)]
                        scores[:, flat] = veto_logits[:, 0] + n_logits[:, n_i] + lev_logits[:, lev_i] + tp_logits[:, tp_i] + sl_logits[:, sl_i]
        return scores.masked_fill(~self.valid_flat_mask.to(x.device).view(1, -1), -1e9)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat, logp, _ = self._decode(x, greedy=False)
        return flat, logp

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        flat, _, _ = self._decode(x, greedy=True)
        return flat

    def log_prob_for_flat(self, x: torch.Tensor, flat: torch.Tensor) -> torch.Tensor:
        _, logp, _ = self._decode(x, greedy=False, teacher=flat)
        return logp


def _initialize_actor_conservative(actor: Risk4Actor) -> None:
    with torch.no_grad():
        actor.notional_head.bias.copy_(torch.tensor([0.8, 0.4, -0.4, -0.8], dtype=actor.notional_head.bias.dtype))
        actor.leverage_head.bias.copy_(torch.tensor([0.5, 0.2, -0.7], dtype=actor.leverage_head.bias.dtype))
        actor.tp_head.bias.copy_(torch.tensor([0.8, 0.4, -0.4, -0.8], dtype=actor.tp_head.bias.dtype))
        actor.sl_head.bias.copy_(torch.tensor([-0.2, 0.4, 0.4, -0.2], dtype=actor.sl_head.bias.dtype))


class Risk4DistributionalCritic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, n_quantiles: int = 32) -> None:
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.value = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, self.n_quantiles))
        self.advantage = nn.Sequential(nn.Linear(hidden, 128), nn.SiLU(), nn.Linear(128, ACTION_DIM * self.n_quantiles))
        self.register_buffer("valid_flat_mask", VALID_ACTION_MASK.clone())

    def forward_dist(self, x: torch.Tensor) -> torch.Tensor:
        h = self.feature(x)
        value = self.value(h).unsqueeze(1)
        adv = self.advantage(h).view(len(x), ACTION_DIM, self.n_quantiles)
        valid = self.valid_flat_mask.to(x.device).view(1, ACTION_DIM, 1)
        adv = adv.masked_fill(~valid, -1e6)
        adv_mean = adv.masked_fill(~valid, 0.0).sum(dim=1, keepdim=True) / valid.sum().clamp_min(1)
        return value + adv - adv_mean

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_dist(x).mean(dim=-1)


def _canonical_action_ids() -> list[int]:
    ids = [0]
    for key in (
        (0, 1, 1, 1, 1),  # current-ish 0.45/2.0/0.026/0.014
        (0, 3, 1, 3, 3),
        (0, 0, 0, 0, 0),
    ):
        ids.append(ACTION_LOOKUP[key])
    return ids


def _sample_action_ids(rng: np.random.Generator, count: int) -> np.ndarray:
    if int(count) >= len(VALID_ACTION_IDS):
        return VALID_ACTION_IDS.copy()
    ids = set(_canonical_action_ids())
    while len(ids) < int(count):
        ids.add(int(rng.choice(VALID_ACTION_IDS)))
    return np.asarray(sorted(ids), dtype=np.int64)


def _action_name(flat_id: int) -> str:
    spec = ACTION_SPECS[int(flat_id)]
    if spec.veto:
        return "veto"
    return (
        f"n{NOTIONAL_BUCKETS[spec.notional_id]:.2f}_l{LEVERAGE_BUCKETS[spec.leverage_id]:.1f}_"
        f"tp{TP_BUCKETS[spec.tp_id]:.3f}_sl{SL_BUCKETS[spec.sl_id]:.3f}"
    )


def _action_count_names(counts: dict[int, int], *, limit: int = 20) -> dict[str, int]:
    return {_action_name(k): int(v) for k, v in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:limit]}


def _best_action_component_distribution(best_actions: list[int]) -> dict[str, Any]:
    active_specs = [ACTION_SPECS[int(a)] for a in best_actions if not ACTION_SPECS[int(a)].veto]
    veto_count = sum(1 for a in best_actions if ACTION_SPECS[int(a)].veto)
    out: dict[str, Any] = {"veto_count": int(veto_count), "active_count": int(len(active_specs))}
    dims = {
        "notional": (NOTIONAL_BUCKETS, [s.notional_id for s in active_specs]),
        "leverage": (LEVERAGE_BUCKETS, [s.leverage_id for s in active_specs]),
        "tp": (TP_BUCKETS, [s.tp_id for s in active_specs]),
        "sl": (SL_BUCKETS, [s.sl_id for s in active_specs]),
    }
    for name, (buckets, idxs) in dims.items():
        counts = {f"{float(v):.3f}": 0 for v in buckets}
        for idx in idxs:
            counts[f"{float(buckets[int(idx)]):.3f}"] += 1
        total = max(len(idxs), 1)
        out[name] = {k: {"count": int(v), "freq": float(v / total)} for k, v in counts.items()}
    wide_tp = sum(1 for s in active_specs if s.tp_id >= 2)
    out["wide_tp_freq"] = float(wide_tp / max(len(active_specs), 1))
    return out


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
    oracle_risk_penalty: float,
) -> tuple[OfflineDataset, dict[str, Any]]:
    active_idxs = np.flatnonzero(_active(dec) & (np.arange(len(frame)) < len(frame) - 3))
    rng = np.random.default_rng(260602)
    total_active_rows = int(len(active_idxs))
    if int(max_active_rows) > 0 and len(active_idxs) > int(max_active_rows):
        active_idxs = np.sort(rng.choice(active_idxs, size=int(max_active_rows), replace=False))
    arrays = {k: _num(frame, k) for k in ("open", "high", "low", "close")}
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    best_list: list[int] = []
    best_per_row: list[int] = []
    oracle_counts: dict[int, int] = {}
    net_sum: dict[int, list[float]] = {}
    for i in active_idxs:
        action_ids = _sample_action_ids(rng, samples_per_row)
        best_a = 0
        best_r = -1e18
        row_rewards: list[tuple[int, float, dict[str, Any]]] = []
        for flat_id in action_ids:
            reward, meta = _simulate_action(frame, arrays, int(i), dec.iloc[int(i)], int(flat_id), fee=fee, slip=slip, cost_mult=cost_mult)
            row_rewards.append((int(flat_id), float(reward), meta))
            spec = ACTION_SPECS[int(flat_id)]
            risk_penalty = 0.0
            if not spec.veto and float(oracle_risk_penalty) > 0.0:
                risk_penalty = float(oracle_risk_penalty) * (
                    float(NOTIONAL_BUCKETS[spec.notional_id]) * float(LEVERAGE_BUCKETS[spec.leverage_id])
                    + float(TP_BUCKETS[spec.tp_id])
                )
            oracle_score = float(reward) - risk_penalty
            if oracle_score > best_r:
                best_r = oracle_score
                best_a = int(flat_id)
        oracle_counts[best_a] = oracle_counts.get(best_a, 0) + 1
        best_per_row.append(best_a)
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
    diag = {
        "active_rows": int(len(active_idxs)),
        "total_active_rows": total_active_rows,
        "samples_per_row": int(samples_per_row),
        "sample_count": int(len(rewards)),
        "reward_scale": float(scale),
        "oracle_risk_penalty": float(oracle_risk_penalty),
        "oracle_top_actions": _action_count_names(oracle_counts, limit=15),
        "best_action_component_distribution": _best_action_component_distribution(best_per_row),
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
        diag,
    )


def _train_dsac(
    data: OfflineDataset,
    *,
    state_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    bc_coef: float,
    n_quantiles: int,
    cvar_frac: float,
    pessimism_weight: float,
    target_entropy: float,
) -> tuple[Risk4Actor, dict[str, Any]]:
    actor = Risk4Actor(state_dim).to(device)
    _initialize_actor_conservative(actor)
    q1 = Risk4DistributionalCritic(state_dim, n_quantiles=n_quantiles).to(device)
    q2 = Risk4DistributionalCritic(state_dim, n_quantiles=n_quantiles).to(device)
    tq1 = Risk4DistributionalCritic(state_dim, n_quantiles=n_quantiles).to(device)
    tq2 = Risk4DistributionalCritic(state_dim, n_quantiles=n_quantiles).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.12), device=device, requires_grad=True)
    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=lr, weight_decay=1e-5)
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=lr, weight_decay=1e-5)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    taus = torch.linspace(
        0.5 / int(n_quantiles),
        1.0 - 0.5 / int(n_quantiles),
        int(n_quantiles),
        device=device,
        dtype=torch.float32,
    )
    cvar_k = max(1, int(int(n_quantiles) * float(cvar_frac)))
    pessimism_weight = float(np.clip(pessimism_weight, 0.5, 1.0))
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
            tq1_all = tq1.forward_dist(sp)
            tq2_all = tq2.forward_dist(sp)
            idx = na.view(-1, 1, 1).expand(-1, 1, int(n_quantiles))
            tq1_next = tq1_all.gather(1, idx).squeeze(1)
            tq2_next = tq2_all.gather(1, idx).squeeze(1)
            tq_min = torch.minimum(tq1_next, tq2_next)
            tq_max = torch.maximum(tq1_next, tq2_next)
            chosen_tq = pessimism_weight * tq_min + (1.0 - pessimism_weight) * tq_max
            tq_mean = chosen_tq.mean(dim=1, keepdim=True)
            tq_centered = chosen_tq - tq_mean
            entropy_term = torch.clamp(log_alpha.exp().detach(), min=1e-4) * nlogp.view(-1, 1)
            target_mean = r.view(-1, 1) + (1.0 - d.view(-1, 1)) * 0.995 * (tq_mean - entropy_term)
            target_q = target_mean + (1.0 - d.view(-1, 1)) * 0.995 * tq_centered
        qa1 = q1.forward_dist(s).gather(1, a.view(-1, 1, 1).expand(-1, 1, int(n_quantiles))).squeeze(1)
        qa2 = q2.forward_dist(s).gather(1, a.view(-1, 1, 1).expand(-1, 1, int(n_quantiles))).squeeze(1)
        q_loss = _quantile_huber_loss(qa1, target_q, taus) + _quantile_huber_loss(qa2, target_q, taus)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()

        pa, plogp = actor.sample(s)
        q1_new = q1.forward_dist(s).gather(1, pa.view(-1, 1, 1).expand(-1, 1, int(n_quantiles))).squeeze(1)
        q2_new = q2.forward_dist(s).gather(1, pa.view(-1, 1, 1).expand(-1, 1, int(n_quantiles))).squeeze(1)
        q1_s, _ = torch.sort(q1_new, dim=1)
        q2_s, _ = torch.sort(q2_new, dim=1)
        c1 = q1_s[:, :cvar_k].mean(dim=1)
        c2 = q2_s[:, :cvar_k].mean(dim=1)
        c_min = torch.minimum(c1, c2)
        c_max = torch.maximum(c1, c2)
        pq = pessimism_weight * c_min + (1.0 - pessimism_weight) * c_max
        bc_loss = -actor.log_prob_for_flat(s, best_a).mean()
        actor_loss = (log_alpha.exp() * plogp - pq).mean() + float(bc_coef) * bc_loss
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = (-plogp).mean().detach()
        alpha_loss = (log_alpha * (entropy - float(target_entropy))).mean()
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
                "target_entropy": float(target_entropy),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), last


def _policy_actions(actor: Risk4Actor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
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


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def _metrics_row(split: str, variant: str, frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    metrics = _fast_replay_metrics(frame, dec, fee=fee, slip=slip, cost_mult=cost_mult)
    row = {"split": split, "variant": variant, "cost": 3, **metrics}
    row["selection_score"] = _score(pd.Series(row))
    return row


def _usage(actions: np.ndarray, active: np.ndarray) -> dict[str, int]:
    counts: dict[int, int] = {}
    for a in actions[np.asarray(active, dtype=bool)]:
        counts[int(a)] = counts.get(int(a), 0) + 1
    return _action_count_names(counts, limit=20)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p00")
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--bc-coef", type=float, default=0.0)
    ap.add_argument("--n-quantiles", type=int, default=32)
    ap.add_argument("--cvar-frac", type=float, default=0.25)
    ap.add_argument("--pessimism-weight", type=float, default=0.80)
    ap.add_argument("--target-entropy", type=float, default=2.0)
    ap.add_argument("--oracle-risk-penalty", type=float, default=0.0)
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
    train_dec = _to_decisions(train_src, oof=True)
    val_dec = _to_decisions(val_src, oof=True)
    oos_dec = _to_decisions(oos_src, oof=False)

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
        oracle_risk_penalty=float(args.oracle_risk_penalty),
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
                "n_quantiles": int(args.n_quantiles),
                "cvar_frac": float(args.cvar_frac),
                "pessimism_weight": float(args.pessimism_weight),
                "target_entropy": float(args.target_entropy),
                "oracle_risk_penalty": float(args.oracle_risk_penalty),
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
        n_quantiles=int(args.n_quantiles),
        cvar_frac=float(args.cvar_frac),
        pessimism_weight=float(args.pessimism_weight),
        target_entropy=float(args.target_entropy),
    )
    a_train = _policy_actions(actor, x_train, device=device)
    a_val = _policy_actions(actor, x_val, device=device)
    a_oos = _policy_actions(actor, x_oos, device=device)

    dsac_train = _compose_decisions(train_dec, a_train)
    dsac_val = _compose_decisions(val_dec, a_val)
    dsac_oos = _compose_decisions(oos_dec, a_oos)
    rows = [
        _metrics_row("val", "fixed_omega1_template", val_df, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "fixed_omega1_template", oos_df, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("val", "dsac_risk4_allocator", val_df, dsac_val, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "dsac_risk4_allocator", oos_df, dsac_oos, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    ]
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    model_path = out_dir / "omega1_expertdq_dsac_risk4_allocator.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": str(args.variant),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "valid_action_dim": int(len(VALID_ACTION_IDS)),
            "valid_action_ids": VALID_ACTION_IDS.tolist(),
            "n_quantiles": int(args.n_quantiles),
            "cvar_frac": float(args.cvar_frac),
            "pessimism_weight": float(args.pessimism_weight),
            "target_entropy": float(args.target_entropy),
            "oracle_risk_penalty": float(args.oracle_risk_penalty),
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "buckets": {
                "notional": NOTIONAL_BUCKETS,
                "leverage": LEVERAGE_BUCKETS,
                "tp": TP_BUCKETS,
                "sl": SL_BUCKETS,
            },
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )
    fast_dsac = grid[(grid["split"] == "oos") & (grid["variant"] == "dsac_risk4_allocator")].iloc[0].to_dict()
    fast_fixed = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_omega1_template")].iloc[0].to_dict()
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Omega1 supervised expert-local decision/quality is frozen. DSAC owns only notional/leverage/TP/SL plus veto. Max-hold and cooldown are removed from this allocator and replay: exits are TP/SL or dataset end, with no cooldown. Actor is factored autoregressive; critic is twin dueling quantile-distributional MLP with CVaR actor objective and controlled pessimism; high-leverage/wide-SL actions are masked.",
        "selection_basis": "2025Q4 validation no-time-exit replay only; 2026 OOS is report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "risk_template": ACTIVE_TEMPLATE,
        "expert_scales": ACTIVE_SCALES,
        "feature_cols": feature_cols,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "valid_action_dim": int(len(VALID_ACTION_IDS)),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "bc_coef": float(args.bc_coef),
            "n_quantiles": int(args.n_quantiles),
            "cvar_frac": float(args.cvar_frac),
            "pessimism_weight": float(args.pessimism_weight),
            "target_entropy": float(args.target_entropy),
            "oracle_risk_penalty": float(args.oracle_risk_penalty),
            "samples_per_row": int(args.samples_per_row),
            "cost_mult": float(args.cost_mult),
            "reward_label": "complete_trade_net_pnl_after_entry_exit_fee_slippage; no max-hold, no cooldown",
            "implemented_report_items": [
                "factored_autoregressive_action_space",
                "leverage_sl_action_mask",
                "tp_sl_ratio_action_mask",
                "notional_leverage_exposure_mask",
                "conservative_actor_initialization",
                "best_action_distribution_audit",
                "reduced_bc_coef",
                "twin_dueling_quantile_distributional_critic",
                "quantile_huber_loss",
                "cvar_actor_loss",
                "controlled_pessimism_twin_critic_blend",
                "dsac_t_style_mean_centered_target",
                "bc_disabled_by_default",
                "configurable_entropy_target",
                "optional_risk_adjusted_oracle_for_bc_experiments",
                "factored_best_action_imitation_loss",
            ],
            "deferred_report_items": [
                "max_bars_duration_penalty_excluded_by_user_request",
                "mamba_temporal_state_encoder",
                "full_gradient_surgery",
                "quality_adaptive_notional",
                "per_dimension_entropy_alpha",
            ],
            "data_diag": data_diag,
            "train_diag": train_diag,
            "action_usage": {"train": _usage(a_train, _active(train_dec)), "val": _usage(a_val, _active(val_dec)), "oos": _usage(a_oos, _active(oos_dec))},
        },
        "fast_replay": {"fixed_oos_cost3": fast_fixed, "dsac_oos_cost3": fast_dsac, "delta_pnl": float(fast_dsac["pnl"]) - float(fast_fixed["pnl"])},
        "official_recheck": None,
        "official_delta_oos_cost3_pnl": None,
        "overlay": overlay,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path), "model": str(model_path), "prior_risk_allocator_dir": str(RISK_OUT_DIR)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(out_dir / "summary.json"),
                "fast_oos_fixed_cost3": fast_fixed,
                "fast_oos_dsac_cost3": fast_dsac,
                "fast_delta_oos_cost3_pnl": summary["fast_replay"]["delta_pnl"],
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
