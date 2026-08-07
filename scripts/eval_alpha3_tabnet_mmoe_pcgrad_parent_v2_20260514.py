#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from sklearn.preprocessing import QuantileTransformer
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3_exec  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_tabnet_mmoe_pcgrad_parent_v2_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_tabnet_mmoe_pcgrad_parent_v2_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_tabnet_mmoe_pcgrad_parent_v2_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_tabnet_mmoe_pcgrad_parent_v2_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_tabnet_mmoe_pcgrad_parent_v2_20260514_grid.csv"


try:
    from entmax import entmax15 as _entmax15
except Exception:  # pragma: no cover - optional dependency
    _entmax15 = None


@dataclass(frozen=True)
class ParentBuckets:
    notional: tuple[float, ...]
    leverage: tuple[float, ...]
    take_profit: tuple[float, ...]
    stop_loss: tuple[float, ...]
    max_hold: tuple[int, ...]
    cooldown: tuple[int, ...]


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    confidence_floor: float
    quality_floor: float
    uncertainty_max: float
    notional_scale: float
    max_notional: float
    use_expected_buckets: bool


class GradientScaler(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = float(alpha)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output * float(ctx.alpha), None


def sparse_attention(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    if _entmax15 is not None:
        return _entmax15(logits, dim=dim)
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted = torch.sort(z, descending=True, dim=dim).values
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.ndim
    view[dim] = -1
    k = k.view(view)
    z_cumsum = z_sorted.cumsum(dim)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(dim, k_z.long() - 1) - 1) / k_z
    return torch.clamp(z - tau, min=0.0)


class GLUBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim * 2)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v, g = self.fc(x).chunk(2, dim=-1)
        return self.dropout(self.norm(v * torch.sigmoid(g)))


class TabNetDecisionStep(nn.Module):
    def __init__(self, n_features: int, hidden: int, dropout: float = 0.10) -> None:
        super().__init__()
        self.masker = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_features),
        )
        self.transform = nn.Sequential(
            GLUBlock(n_features, hidden, dropout),
            GLUBlock(hidden, hidden, dropout),
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor, prior: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mask = sparse_attention(self.masker(state) + torch.log(prior.clamp_min(1e-6)), dim=-1)
        step_state = self.transform(x * mask)
        new_prior = prior * (1.0 - mask).clamp(0.05, 1.0)
        entropy = -(mask.clamp_min(1e-8) * mask.clamp_min(1e-8).log()).sum(dim=-1).mean()
        return step_state, new_prior, entropy


class MultiTaskTabNetParent(nn.Module):
    def __init__(
        self,
        n_features: int,
        buckets: ParentBuckets,
        *,
        hidden: int = 128,
        steps: int = 4,
        experts: int = 4,
        virtual_nodes: int = 7,
        grad_alpha: float = 0.10,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.n_raw_features = int(n_features)
        self.virtual_nodes = int(virtual_nodes)
        self.n_features = int(n_features + virtual_nodes)
        self.steps = int(steps)
        self.n_experts = int(experts)
        self.grad_alpha = float(grad_alpha)
        self.buckets = buckets
        self.input_norm = nn.LayerNorm(self.n_features)
        self.initial_state = nn.Sequential(nn.Linear(self.n_features, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.steps_net = nn.ModuleList([TabNetDecisionStep(self.n_features, hidden, dropout) for _ in range(self.steps)])
        self.shared_norm = nn.LayerNorm(hidden)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    GLUBlock(hidden, hidden, dropout),
                    GLUBlock(hidden, hidden, dropout),
                )
                for _ in range(self.n_experts)
            ]
        )
        self.task_gates = nn.ModuleDict(
            {
                task: nn.Linear(hidden, self.n_experts)
                for task in ("action", "quality", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
            }
        )
        self.action_trunk = nn.Sequential(GLUBlock(hidden, hidden, dropout), GLUBlock(hidden, hidden, dropout))
        self.action_head = nn.Linear(hidden, 3)
        self.action_log_temp = nn.Parameter(torch.log(torch.tensor(1.5)))
        self.param_trunk = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            GLUBlock(hidden, hidden, dropout),
        )
        self.quality_head = nn.Linear(hidden, 1)
        self.notional_head = nn.Linear(hidden, len(buckets.notional))
        self.leverage_head = nn.Linear(hidden, len(buckets.leverage))
        self.take_profit_head = nn.Linear(hidden, len(buckets.take_profit))
        self.stop_loss_head = nn.Linear(hidden, len(buckets.stop_loss))
        self.max_hold_head = nn.Linear(hidden, len(buckets.max_hold))
        self.cooldown_head = nn.Linear(hidden, len(buckets.cooldown))
        self.log_vars = nn.Parameter(torch.zeros(8))

    def _append_virtual_nodes(self, x: torch.Tensor) -> torch.Tensor:
        if self.virtual_nodes <= 0:
            return x
        sink = torch.zeros(x.shape[0], self.virtual_nodes, device=x.device, dtype=x.dtype)
        return torch.cat([x, sink], dim=-1)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.input_norm(self._append_virtual_nodes(x))
        state = self.initial_state(x)
        prior = torch.ones(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        agg = torch.zeros_like(state)
        entropy_terms: list[torch.Tensor] = []
        for step in self.steps_net:
            step_state, prior, entropy = step(x, state, prior)
            agg = agg + F.relu(step_state)
            state = step_state
            entropy_terms.append(entropy)
        shared = self.shared_norm(agg / float(self.steps))
        expert_stack = torch.stack([expert(shared) for expert in self.experts], dim=1)
        def route(task: str) -> torch.Tensor:
            gate = torch.softmax(self.task_gates[task](shared), dim=-1).unsqueeze(-1)
            return torch.sum(expert_stack * gate, dim=1)

        action_hidden = self.action_trunk(route("action"))
        temp = torch.exp(self.action_log_temp).clamp(0.50, 5.00)
        action_logits = self.action_head(action_hidden) / temp
        scaled_action_hidden = GradientScaler.apply(action_hidden, self.grad_alpha)
        quality_hidden = self.param_trunk(torch.cat([route("quality"), scaled_action_hidden], dim=-1))
        notional_hidden = self.param_trunk(torch.cat([route("notional"), scaled_action_hidden], dim=-1))
        leverage_hidden = self.param_trunk(torch.cat([route("leverage"), scaled_action_hidden], dim=-1))
        tp_hidden = self.param_trunk(torch.cat([route("take_profit"), scaled_action_hidden], dim=-1))
        sl_hidden = self.param_trunk(torch.cat([route("stop_loss"), scaled_action_hidden], dim=-1))
        hold_hidden = self.param_trunk(torch.cat([route("max_hold"), scaled_action_hidden], dim=-1))
        cooldown_hidden = self.param_trunk(torch.cat([route("cooldown"), scaled_action_hidden], dim=-1))
        return {
            "action": action_logits,
            "quality": self.quality_head(quality_hidden).squeeze(-1),
            "notional": self.notional_head(notional_hidden),
            "leverage": self.leverage_head(leverage_hidden),
            "take_profit": self.take_profit_head(tp_hidden),
            "stop_loss": self.stop_loss_head(sl_hidden),
            "max_hold": self.max_hold_head(hold_hidden),
            "cooldown": self.cooldown_head(cooldown_hidden),
            "mask_entropy": torch.stack(entropy_terms).mean(),
            "temperature": temp,
        }


class ParentDataset(Dataset):
    def __init__(self, x: np.ndarray, y: dict[str, np.ndarray], soft: dict[str, np.ndarray]) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
        self.y = {
            "action": torch.as_tensor(y["action"], dtype=torch.long),
            "quality": torch.as_tensor(y["quality"], dtype=torch.float32),
            "notional": torch.as_tensor(y["notional"], dtype=torch.long),
            "leverage": torch.as_tensor(y["leverage"], dtype=torch.long),
            "take_profit": torch.as_tensor(y["take_profit"], dtype=torch.long),
            "stop_loss": torch.as_tensor(y["stop_loss"], dtype=torch.long),
            "max_hold": torch.as_tensor(y["max_hold"], dtype=torch.long),
            "cooldown": torch.as_tensor(y["cooldown"], dtype=torch.long),
        }
        self.soft = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in soft.items()}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.x[idx], {k: v[idx] for k, v in self.y.items()}, {k: v[idx] for k, v in self.soft.items()}


class ExpandingQuantileNormalizer:
    def __init__(self, *, min_rows: int = 2000, n_quantiles: int = 1024) -> None:
        self.min_rows = int(min_rows)
        self.n_quantiles = int(n_quantiles)
        self.snapshots: list[tuple[pd.Timestamp, QuantileTransformer, np.ndarray, list[str]]] = []

    def fit_snapshots(self, x: pd.DataFrame, timestamps: pd.Series) -> None:
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        cols = list(x.columns)
        arr = x.reset_index(drop=True).replace([np.inf, -np.inf], np.nan).astype(float)
        months = sorted(ts.dt.to_period("M").unique())
        snapshots: list[tuple[pd.Timestamp, QuantileTransformer, np.ndarray, list[str]]] = []
        for period in months:
            cutoff = period.to_timestamp()
            fit_mask = ts < cutoff
            if int(fit_mask.sum()) < self.min_rows:
                continue
            fit_x = arr.loc[fit_mask, cols]
            med = fit_x.median(axis=0).to_numpy(dtype=np.float32)
            filled = fit_x.fillna(pd.Series(med, index=cols))
            qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, max(16, len(filled))),
                output_distribution="normal",
                subsample=None,
                random_state=20260514,
            )
            qt.fit(filled.to_numpy(dtype=np.float32))
            snapshots.append((cutoff, qt, med, cols))
        if not snapshots:
            fit_x = arr.loc[:, cols]
            med = fit_x.median(axis=0).to_numpy(dtype=np.float32)
            filled = fit_x.fillna(pd.Series(med, index=cols))
            qt = QuantileTransformer(
                n_quantiles=min(self.n_quantiles, max(16, len(filled))),
                output_distribution="normal",
                subsample=None,
                random_state=20260514,
            )
            qt.fit(filled.to_numpy(dtype=np.float32))
            snapshots.append((ts.min(), qt, med, cols))
        self.snapshots = snapshots

    def transform(self, x: pd.DataFrame, timestamps: pd.Series) -> np.ndarray:
        if not self.snapshots:
            raise ValueError("normalizer_not_fitted")
        ts = pd.to_datetime(timestamps).reset_index(drop=True)
        arr = x.reset_index(drop=True).replace([np.inf, -np.inf], np.nan).astype(float)
        out = np.zeros((len(arr), len(self.snapshots[-1][3])), dtype=np.float32)
        for si, snap in enumerate(self.snapshots):
            cutoff, qt, med, cols = snap
            nxt = self.snapshots[si + 1][0] if si + 1 < len(self.snapshots) else pd.Timestamp.max
            mask = (ts >= cutoff) & (ts < nxt)
            if si == 0:
                mask = ts < nxt
            loc = np.flatnonzero(mask.to_numpy())
            if not len(loc):
                continue
            chunk = arr.iloc[loc].reindex(columns=cols)
            chunk = chunk.fillna(pd.Series(med, index=cols))
            z = qt.transform(chunk.to_numpy(dtype=np.float32)).astype(np.float32)
            out[loc] = np.nan_to_num(z, nan=0.0, posinf=5.0, neginf=-5.0).clip(-5.0, 5.0)
        return out


def _parent_cfg() -> FullyLearnedGovernorConfig:
    return FullyLearnedGovernorConfig(
        notional_buckets=(0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14),
        leverage_buckets=(1.5, 2.0, 3.0, 4.0, 5.0),
        take_profit_buckets=(0.007, 0.011, 0.018, 0.030, 0.050, 0.090, 0.180, 0.450, 0.900),
        stop_loss_buckets=(0.004, 0.006, 0.009, 0.014, 0.022, 0.035, 0.055),
        max_hold_buckets=(6, 12, 24, 48, 96, 192, 288),
        cooldown_buckets=(0, 1, 3, 6, 12, 24, 48),
        max_train_horizon_bars=288,
        cash_score=0.020,
        adverse_penalty=2.45,
        size_penalty=0.180,
        hold_penalty=0.042,
        turnover_bonus=0.0012,
        max_margin_fraction=1.10,
    )


def _buckets(cfg: FullyLearnedGovernorConfig) -> ParentBuckets:
    return ParentBuckets(
        notional=tuple(float(x) for x in cfg.notional_buckets),
        leverage=tuple(float(x) for x in cfg.leverage_buckets),
        take_profit=tuple(float(x) for x in cfg.take_profit_buckets),
        stop_loss=tuple(float(x) for x in cfg.stop_loss_buckets),
        max_hold=tuple(int(x) for x in cfg.max_hold_buckets),
        cooldown=tuple(int(x) for x in cfg.cooldown_buckets),
    )


def _candidate_indices(n: int, cfg: FullyLearnedGovernorConfig, stride: int) -> np.ndarray:
    return np.arange(0, max(0, n - int(cfg.max_train_horizon_bars) - 1), max(1, int(stride)), dtype=np.int64)


def _onehot(y: np.ndarray, n_classes: int) -> np.ndarray:
    out = np.zeros((len(y), int(n_classes)), dtype=np.float32)
    out[np.arange(len(y)), np.clip(y.astype(int), 0, n_classes - 1)] = 1.0
    return out


def _proba_full(model: Any, x: pd.DataFrame, n_classes: int, fallback: np.ndarray | None = None) -> np.ndarray:
    if model is None:
        assert fallback is not None
        return fallback.astype(np.float32)
    p = model.predict_proba(x)
    classes = np.asarray(model.classes_, dtype=int)
    out = np.zeros((len(x), int(n_classes)), dtype=np.float32)
    out[:, np.clip(classes, 0, n_classes - 1)] = p.astype(np.float32)
    s = out.sum(axis=1, keepdims=True)
    return out / np.maximum(s, 1e-12)


def _hgb_soft_targets(parent: dict[str, Any], x: pd.DataFrame, y_hard: dict[str, np.ndarray], cfg: FullyLearnedGovernorConfig) -> dict[str, np.ndarray]:
    side_hint = np.where(y_hard["action"] == ACTION_LONG, 1.0, np.where(y_hard["action"] == ACTION_SHORT, -1.0, 0.0))
    x_side = x.copy()
    if "side_hint" in x_side.columns:
        x_side["side_hint"] = side_hint
    out = {
        "action": _proba_full(parent.get("action_model"), x, 3, _onehot(y_hard["action"], 3)),
        "quality": np.asarray(parent["quality_model"].predict(x), dtype=np.float32) if "quality_model" in parent else np.asarray(y_hard["quality"], dtype=np.float32),
        "notional": _proba_full(parent.get("notional_model"), x_side, len(cfg.notional_buckets), _onehot(y_hard["notional"], len(cfg.notional_buckets))),
        "leverage": _proba_full(parent.get("leverage_model"), x_side, len(cfg.leverage_buckets), _onehot(y_hard["leverage"], len(cfg.leverage_buckets))),
        "take_profit": _proba_full(parent.get("take_profit_model"), x_side, len(cfg.take_profit_buckets), _onehot(y_hard["take_profit"], len(cfg.take_profit_buckets))),
        "stop_loss": _proba_full(parent.get("stop_loss_model"), x_side, len(cfg.stop_loss_buckets), _onehot(y_hard["stop_loss"], len(cfg.stop_loss_buckets))),
        "max_hold": _proba_full(parent.get("max_hold_model"), x_side, len(cfg.max_hold_buckets), _onehot(y_hard["max_hold"], len(cfg.max_hold_buckets))),
        "cooldown": _proba_full(parent.get("cooldown_model"), x_side, len(cfg.cooldown_buckets), _onehot(y_hard["cooldown"], len(cfg.cooldown_buckets))),
    }
    return out


def _kl_soft(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.kl_div(F.log_softmax(logits, dim=-1), target, reduction="batchmean")


def _homo_loss(losses: list[torch.Tensor], log_vars: torch.Tensor) -> torch.Tensor:
    total = losses[0].new_tensor(0.0)
    for i, loss in enumerate(losses):
        s = log_vars[i].clamp(-4.0, 4.0)
        total = total + torch.exp(-s) * loss + s
    return total


def _contradiction_penalty(out: dict[str, torch.Tensor], buckets: ParentBuckets) -> torch.Tensor:
    action_prob = torch.softmax(out["action"], dim=-1)
    trade_prob = action_prob[:, ACTION_LONG] + action_prob[:, ACTION_SHORT]
    lev_vals = torch.tensor(buckets.leverage, device=out["leverage"].device, dtype=out["leverage"].dtype)
    sl_vals = torch.tensor(buckets.stop_loss, device=out["stop_loss"].device, dtype=out["stop_loss"].dtype)
    lev = torch.softmax(out["leverage"], dim=-1) @ lev_vals
    sl = torch.softmax(out["stop_loss"], dim=-1) @ sl_vals
    return torch.mean(trade_prob * F.relu(lev * sl - 0.11) ** 2)


def _task_losses(
    model: MultiTaskTabNetParent,
    out: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    soft: dict[str, torch.Tensor],
    *,
    action_weight: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
) -> tuple[list[torch.Tensor], torch.Tensor, dict[str, float]]:
    trade = y["action"] != ACTION_CASH
    hard_action = F.cross_entropy(out["action"], y["action"], weight=action_weight)
    kd_action = _kl_soft(out["action"], soft["action"])
    if epoch <= warmup_epochs:
        mix = 0.0
    else:
        mix = min(1.0, float(epoch - warmup_epochs) / 20.0)
    action_loss = (1.0 - mix) * kd_action + mix * hard_action
    quality_target = soft["quality"] if epoch <= warmup_epochs else ((1.0 - mix) * soft["quality"] + mix * y["quality"])
    quality_loss = F.smooth_l1_loss(out["quality"], quality_target)

    bucket_losses: list[torch.Tensor] = []
    for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
        if epoch <= warmup_epochs:
            bucket_losses.append(_kl_soft(out[key][trade], soft[key][trade]) if bool(trade.any()) else out["action"].new_tensor(0.0))
        elif bool(trade.any()):
            hard = F.cross_entropy(out[key][trade], y[key][trade])
            kd = _kl_soft(out[key][trade], soft[key][trade])
            bucket_losses.append((1.0 - mix) * kd + mix * hard)
        else:
            bucket_losses.append(out["action"].new_tensor(0.0))

    losses = [action_loss, quality_loss, *bucket_losses]
    regularizer = 1e-3 * out["mask_entropy"] + 0.05 * _contradiction_penalty(out, model.buckets)
    bucket_mean = torch.stack(bucket_losses).mean() if bucket_losses else out["action"].new_tensor(0.0)
    return losses, regularizer, {
        "action": float(action_loss.detach().cpu()),
        "quality": float(quality_loss.detach().cpu()),
        "bucket": float(bucket_mean.detach().cpu()),
        "mix": float(mix),
        "temp": float(out["temperature"].detach().cpu()),
    }


def _loss(
    model: MultiTaskTabNetParent,
    out: dict[str, torch.Tensor],
    y: dict[str, torch.Tensor],
    soft: dict[str, torch.Tensor],
    *,
    action_weight: torch.Tensor,
    epoch: int,
    warmup_epochs: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    losses, regularizer, parts = _task_losses(
        model,
        out,
        y,
        soft,
        action_weight=action_weight,
        epoch=epoch,
        warmup_epochs=warmup_epochs,
    )
    task = _homo_loss(losses, model.log_vars)
    loss = task + regularizer
    return loss, {
        "action": parts["action"],
        "quality": parts["quality"],
        "bucket": parts["bucket"],
        "mix": parts["mix"],
        "temp": parts["temp"],
    }


def _weighted_task_losses(model: MultiTaskTabNetParent, task_losses: list[torch.Tensor]) -> list[torch.Tensor]:
    weighted: list[torch.Tensor] = []
    for i, loss in enumerate(task_losses):
        s = model.log_vars[i].clamp(-4.0, 4.0)
        weighted.append(torch.exp(-s) * loss + s)
    return weighted


def _flatten_grads(grads: tuple[torch.Tensor | None, ...], params: list[torch.nn.Parameter]) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    for grad, param in zip(grads, params):
        if grad is None:
            chunks.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(grad.reshape(-1))
    return torch.cat(chunks)


def _assign_flat_grad(params: list[torch.nn.Parameter], flat: torch.Tensor) -> None:
    offset = 0
    for param in params:
        n = param.numel()
        grad = flat[offset : offset + n].view_as(param).detach()
        if param.grad is None:
            param.grad = grad.clone()
        else:
            param.grad.copy_(grad)
        offset += n


def _pcgrad_backward(
    losses: list[torch.Tensor],
    regularizer: torch.Tensor,
    params: list[torch.nn.Parameter],
) -> float:
    if not losses:
        regularizer.backward()
        return float(regularizer.detach().cpu())
    flat_grads: list[torch.Tensor] = []
    for loss in losses:
        grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        flat_grads.append(_flatten_grads(grads, params))
    projected: list[torch.Tensor] = []
    order_base = np.arange(len(flat_grads))
    for i, grad in enumerate(flat_grads):
        gi = grad.clone()
        order = np.random.permutation(order_base)
        for j in order:
            if i == int(j):
                continue
            gj = flat_grads[int(j)]
            denom = torch.dot(gj, gj).clamp_min(1e-12)
            dot = torch.dot(gi, gj)
            if bool(dot < 0):
                gi = gi - dot / denom * gj
        projected.append(gi)
    merged = torch.stack(projected, dim=0).mean(dim=0)
    _assign_flat_grad(params, merged)
    regularizer.backward()
    return float((torch.stack([x.detach() for x in losses]).sum() + regularizer.detach()).cpu())


def _train_model(
    model: MultiTaskTabNetParent,
    train_ds: ParentDataset,
    val_ds: ParentDataset,
    *,
    epochs: int,
    warmup_epochs: int,
    batch_size: int,
    device: torch.device,
    patience: int,
) -> dict[str, Any]:
    model.to(device)
    action_counts = torch.bincount(train_ds.y["action"], minlength=3).float()
    action_weight = action_counts.sum() / action_counts.clamp_min(1.0)
    action_weight[ACTION_CASH] *= 0.28
    action_weight[ACTION_LONG] *= 1.20
    action_weight[ACTION_SHORT] *= 1.20
    action_weight = (action_weight / action_weight.mean().clamp_min(1e-6)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=4, min_lr=8e-6)
    swa_model = AveragedModel(model)
    swa_start = max(int(epochs * 0.65), warmup_epochs + 5)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        parts_acc = {"action": 0.0, "quality": 0.0, "bucket": 0.0}
        params = [p for p in model.parameters() if p.requires_grad]
        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            sb = {k: v.to(device) for k, v in sb.items()}
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            task_losses, regularizer, parts = _task_losses(
                model,
                out,
                yb,
                sb,
                action_weight=action_weight,
                epoch=epoch,
                warmup_epochs=warmup_epochs,
            )
            weighted = _weighted_task_losses(model, task_losses)
            loss_value = _pcgrad_backward(weighted, regularizer, params)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss_value) * len(xb)
            count += len(xb)
            for k in parts_acc:
                parts_acc[k] += float(parts[k]) * len(xb)
        if epoch >= swa_start:
            swa_model.update_parameters(model)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb = xb.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                sb = {k: v.to(device) for k, v in sb.items()}
                vl, _ = _loss(model, model(xb), yb, sb, action_weight=action_weight, epoch=epoch, warmup_epochs=warmup_epochs)
                vtotal += float(vl.item()) * len(xb)
                vcount += len(xb)
        train_loss = total / max(count, 1)
        val_loss = vtotal / max(vcount, 1)
        scheduler.step(val_loss)
        lr = float(opt.param_groups[0]["lr"])
        entry = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "temp": float(torch.exp(model.action_log_temp.detach()).clamp(0.5, 5.0).cpu()),
            "log_vars": [float(x) for x in model.log_vars.detach().cpu().tolist()],
            "parts": {k: parts_acc[k] / max(count, 1) for k in parts_acc},
        }
        history.append(entry)
        print(
            f"[{MODEL_ID}] epoch={epoch:03d} train={train_loss:.5f} val={val_loss:.5f} "
            f"lr={lr:.2e} temp={entry['temp']:.3f} action={entry['parts']['action']:.4f} bucket={entry['parts']['bucket']:.4f}",
            flush=True,
        )
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience and epoch > warmup_epochs + 5:
            print(f"[{MODEL_ID}] early_stop epoch={epoch} best_val={best_val:.5f}", flush=True)
            break
    if len(train_loader) > 0:
        try:
            update_bn(train_loader, swa_model, device=device)
            model.load_state_dict(swa_model.module.state_dict())
            print(f"[{MODEL_ID}] applied_swa start={swa_start}", flush=True)
        except Exception as exc:
            print(f"[{MODEL_ID}] swa_skip err={exc}", flush=True)
            if best_state is not None:
                model.load_state_dict(best_state)
    elif best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu").eval()
    return {"best_val_loss": float(best_val), "history": history, "epochs_ran": len(history), "swa_start": int(swa_start), "used_entmax": bool(_entmax15 is not None)}


def _predict_outputs(model: MultiTaskTabNetParent, x: np.ndarray, *, device: torch.device, batch_size: int, mc_passes: int = 5) -> dict[str, np.ndarray]:
    model.to(device)
    keys = ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
    outs: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    outs["quality"] = []
    outs["action_uncertainty"] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=device)
            quality_passes: list[torch.Tensor] = []
            probs: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
            for _ in range(max(1, int(mc_passes))):
                model.train(mc_passes > 1)
                pred = model(xb)
                quality_passes.append(pred["quality"])
                for key in keys:
                    probs[key].append(torch.softmax(pred[key].clamp(-8.0, 8.0), dim=-1))
            action_stack = torch.stack(probs["action"], dim=0)
            outs["quality"].append(torch.stack(quality_passes, dim=0).mean(dim=0).detach().cpu().numpy())
            outs["action_uncertainty"].append(action_stack.std(dim=0).mean(dim=1).detach().cpu().numpy())
            for key in keys:
                outs[key].append(torch.stack(probs[key], dim=0).mean(dim=0).detach().cpu().numpy())
    model.to("cpu").eval()
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _bucket_value(proba: np.ndarray, values: tuple[float, ...], expected: bool) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(values, dtype=np.float64)
    p = proba[:, : len(vals)]
    conf = np.max(p, axis=1)
    if expected:
        return p @ vals, conf
    return vals[np.argmax(p, axis=1)], conf


def _decisions_from_outputs(outputs: dict[str, np.ndarray], cfg: FullyLearnedGovernorConfig, rt: RuntimeConfig, index: pd.Index) -> pd.DataFrame:
    action_p = outputs["action"]
    pred_action = np.argmax(action_p, axis=1).astype(np.int64)
    pred_conf = np.max(action_p, axis=1)
    uncertainty = np.asarray(outputs["action_uncertainty"], dtype=np.float64)
    side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0)).astype(np.int64)
    notional, c1 = _bucket_value(outputs["notional"], tuple(float(x) for x in cfg.notional_buckets), rt.use_expected_buckets)
    leverage, c2 = _bucket_value(outputs["leverage"], tuple(float(x) for x in cfg.leverage_buckets), rt.use_expected_buckets)
    tp, c3 = _bucket_value(outputs["take_profit"], tuple(float(x) for x in cfg.take_profit_buckets), rt.use_expected_buckets)
    sl, c4 = _bucket_value(outputs["stop_loss"], tuple(float(x) for x in cfg.stop_loss_buckets), rt.use_expected_buckets)
    mh, c5 = _bucket_value(outputs["max_hold"], tuple(float(x) for x in cfg.max_hold_buckets), rt.use_expected_buckets)
    cd, c6 = _bucket_value(outputs["cooldown"], tuple(float(x) for x in cfg.cooldown_buckets), rt.use_expected_buckets)
    quality = np.asarray(outputs["quality"], dtype=np.float64)
    active = (
        (pred_action != ACTION_CASH)
        & (side != 0)
        & (pred_conf >= float(rt.confidence_floor))
        & (quality >= float(rt.quality_floor))
        & (uncertainty <= float(rt.uncertainty_max))
    )
    notional = np.clip(notional * float(rt.notional_scale), 0.0, float(rt.max_notional))
    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    action = np.where(active, pred_action, ACTION_CASH).astype(np.int64)
    side = np.where(active, side, 0).astype(np.int64)
    confidence = np.mean(np.vstack([pred_conf, c1, c2, c3, c4, c5, c6]), axis=0)
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": tp.astype(np.float64),
            "stop_loss": sl.astype(np.float64),
            "max_hold_bars": np.rint(mh).astype(np.int64),
            "cooldown_bars": np.rint(cd).astype(np.int64),
            "quality_score": quality.astype(np.float64),
            "confidence": confidence.astype(np.float64),
            "tabnet_action_confidence": pred_conf.astype(np.float64),
            "tabnet_action_uncertainty": uncertainty.astype(np.float64),
        },
        index=index,
    )
    cash = out["action"].astype(int).to_numpy() == ACTION_CASH
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _runtime_grid() -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for conf in (0.22, 0.30, 0.38, 0.46, 0.54):
        for q in (-0.080, -0.040, -0.015, 0.0):
            for unc in (0.090, 0.130, 0.180):
                for scale, cap in ((1.00, 2.75), (1.15, 3.10), (1.30, 3.60), (1.45, 4.14)):
                    for expected in (False, True):
                        rows.append(RuntimeConfig(f"mmoe_c{conf:.2f}_q{q:.3f}_u{unc:.3f}_s{scale:.2f}_cap{cap:.2f}_{'exp' if expected else 'arg'}", conf, q, unc, scale, cap, expected))
    return rows


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1.get("trades", 0)) < 60:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(1.30 * c1["pnl"] + 0.30 * c2["pnl"] + 0.18 * c3["pnl"] - 0.20 * abs(c1["mdd"]) + 0.05 * float(c1.get("trades", 0)))


def _metrics_alpha3(df, parent, jackpot_model, add_cfg, q, decisions, overlay, limit_cfg, *, fee, slip) -> dict[str, Any]:
    return {
        f"cost{mult}": alpha3_exec.backtest_signal_limit(
            df,
            parent,
            jackpot_model,
            add_cfg,
            q,
            decisions,
            overlay,
            limit_cfg,
            fee=fee,
            slip=slip,
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _selected_alpha3_limit_cfg() -> alpha3_exec.ImmediateLimitConfig:
    audit = json.loads((ROOT / "data/ensemble/reports/alpha2_1_signal_immediate_limit_20260514_audit.json").read_text(encoding="utf-8"))
    cfg = dict(audit.get("selected_config", {}) or {})
    return alpha3_exec.ImmediateLimitConfig(
        name=str(cfg.get("name", "next_open_limit_offset2_entry_fallback_fee20")),
        anchor=str(cfg.get("anchor", "next_open")),
        entry_offset_bps=float(cfg.get("entry_offset_bps", 2.0)),
        exit_offset_bps=float(cfg.get("exit_offset_bps", 2.0)),
        penetration_bps=float(cfg.get("penetration_bps", 0.5)),
        maker_fee_mult=float(cfg.get("maker_fee_mult", 0.20)),
        entry_miss=str(cfg.get("entry_miss", "market_fallback")),
        exit_miss=str(cfg.get("exit_miss", "market_fallback")),
    )


def main() -> int:
    p = argparse.ArgumentParser(description="Alpha3 parent replacement with TabNet MMoE + PCGrad.")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--warmup-epochs", type=int, default=8)
    p.add_argument("--patience", type=int, default=14)
    p.add_argument("--stride", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(20260514)
    np.random.seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    fee = float(dict(parent["config"])["fee"])
    slip = float(dict(parent["config"])["slip"])
    cfg = _parent_cfg()
    buckets = _buckets(cfg)
    feature_cols = list(parent.get("feature_cols") or [])
    overlay = next(v for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20").overlay
    limit_cfg = _selected_alpha3_limit_cfg()

    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    print(f"[{MODEL_ID}] rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)} features={len(feature_cols)}", flush=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)

    train_x_raw, y_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    idx_train = _candidate_indices(len(train_df), cfg, int(args.stride))
    if len(idx_train) != len(train_x_raw):
        raise ValueError(f"candidate_index_mismatch idx={len(idx_train)} x={len(train_x_raw)}")
    train_ts = train_df.iloc[idx_train]["timestamp"].reset_index(drop=True)
    soft_train = _hgb_soft_targets(parent, train_x_raw, y_train, cfg)

    val_teacher = predict_policy_frame(parent, val_df, close=_close(val_df))
    idx_val = _candidate_indices(len(val_df), cfg, max(2, int(args.stride)))
    val_features_full = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    val_x_ds = val_features_full.iloc[idx_val].reset_index(drop=True)
    val_ts_ds = val_df.iloc[idx_val]["timestamp"].reset_index(drop=True)
    y_val = {
        "action": val_teacher.iloc[idx_val]["action"].astype(int).to_numpy(dtype=np.int64),
        "quality": pd.to_numeric(val_teacher.iloc[idx_val]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "notional": np.zeros(len(idx_val), dtype=np.int64),
        "leverage": np.zeros(len(idx_val), dtype=np.int64),
        "take_profit": np.zeros(len(idx_val), dtype=np.int64),
        "stop_loss": np.zeros(len(idx_val), dtype=np.int64),
        "max_hold": np.zeros(len(idx_val), dtype=np.int64),
        "cooldown": np.zeros(len(idx_val), dtype=np.int64),
    }
    for key, col, vals in (
        ("notional", "notional_exposure", cfg.notional_buckets),
        ("leverage", "leverage", cfg.leverage_buckets),
        ("take_profit", "take_profit", cfg.take_profit_buckets),
        ("stop_loss", "stop_loss", cfg.stop_loss_buckets),
        ("max_hold", "max_hold_bars", cfg.max_hold_buckets),
        ("cooldown", "cooldown_bars", cfg.cooldown_buckets),
    ):
        arr = pd.to_numeric(val_teacher.iloc[idx_val][col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        b = np.asarray(vals, dtype=np.float64)
        y_val[key] = np.argmin(np.abs(arr[:, None] - b[None, :]), axis=1).astype(np.int64)
    soft_val = _hgb_soft_targets(parent, val_x_ds, y_val, cfg)

    normalizer = ExpandingQuantileNormalizer(min_rows=2000, n_quantiles=1024)
    normalizer.fit_snapshots(train_x_raw, train_ts)
    x_train = normalizer.transform(train_x_raw, train_ts)
    x_val_ds = normalizer.transform(val_x_ds, val_ts_ds)
    val_x_full = normalizer.transform(val_features_full, val_df["timestamp"])
    eval_features_full = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    eval_x_full = normalizer.transform(eval_features_full, eval_df["timestamp"])

    train_ds = ParentDataset(x_train, y_train, soft_train)
    val_ds = ParentDataset(x_val_ds, y_val, soft_val)
    model = MultiTaskTabNetParent(len(feature_cols), buckets, experts=4)
    training = _train_model(
        model,
        train_ds,
        val_ds,
        epochs=int(18 if args.quick else args.epochs),
        warmup_epochs=int(min(args.warmup_epochs, 5 if args.quick else args.warmup_epochs)),
        batch_size=int(args.batch_size),
        device=device,
        patience=int(6 if args.quick else args.patience),
    )

    print(f"[{MODEL_ID}] predicting validation/oos", flush=True)
    val_outputs = _predict_outputs(model, val_x_full, device=device, batch_size=int(args.batch_size), mc_passes=3 if args.quick else 5)
    eval_outputs = _predict_outputs(model, eval_x_full, device=device, batch_size=int(args.batch_size), mc_passes=3 if args.quick else 5)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_hgb_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    rows: list[dict[str, Any]] = []
    best_rt: RuntimeConfig | None = None
    best_score = -1e18
    grid = _runtime_grid()
    if args.quick:
        grid = [
            r
            for r in grid
            if r.confidence_floor in (0.30, 0.38, 0.46)
            and r.quality_floor in (-0.040, -0.015)
            and r.uncertainty_max in (0.130, 0.180)
            and r.notional_scale in (1.15, 1.30)
            and not r.use_expected_buckets
        ]
    print(f"[{MODEL_ID}] selecting runtime configs={len(grid)} on 2025Q4", flush=True)
    for rt in grid:
        dec = _decisions_from_outputs(val_outputs, cfg, rt, val_df.index)
        metrics = _metrics_alpha3(val_df, parent, jackpot_model, add_cfg, val_q, dec, overlay, limit_cfg, fee=fee, slip=slip)
        score = _score(metrics)
        row = {
            **asdict(rt),
            "score": score,
            "val_cost1_pnl": metrics["cost1"]["pnl"],
            "val_cost1_mdd": metrics["cost1"]["mdd"],
            "val_cost1_trades": metrics["cost1"]["trades"],
            "val_cost2_pnl": metrics["cost2"]["pnl"],
            "val_cost3_pnl": metrics["cost3"]["pnl"],
        }
        rows.append(row)
        if score > best_score:
            best_score = score
            best_rt = rt
            print(f"[{MODEL_ID}] new best {rt.name} score={score:.2f} val_c1={row['val_cost1_pnl']:.2f} mdd={row['val_cost1_mdd']:.2f}", flush=True)
    assert best_rt is not None

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    tabnet_dec = _decisions_from_outputs(eval_outputs, cfg, best_rt, eval_df.index)
    baseline_metrics = _metrics_alpha3(eval_df, parent, jackpot_model, add_cfg, eval_q, eval_hgb_dec, overlay, limit_cfg, fee=fee, slip=slip)
    tabnet_metrics = _metrics_alpha3(eval_df, parent, jackpot_model, add_cfg, eval_q, tabnet_dec, overlay, limit_cfg, fee=fee, slip=slip)
    print(f"[{MODEL_ID}] baseline c1={baseline_metrics['cost1']['pnl']:.2f} mdd={baseline_metrics['cost1']['mdd']:.2f} c2={baseline_metrics['cost2']['pnl']:.2f} c3={baseline_metrics['cost3']['pnl']:.2f}", flush=True)
    print(f"[{MODEL_ID}] tabnet   c1={tabnet_metrics['cost1']['pnl']:.2f} mdd={tabnet_metrics['cost1']['mdd']:.2f} c2={tabnet_metrics['cost2']['pnl']:.2f} c3={tabnet_metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUT_DIR / "alpha3_tabnet_mmoe_pcgrad_parent_v2.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "normalizer": normalizer,
            "config": asdict(cfg),
            "selected_runtime": asdict(best_rt),
            "training": training,
            "buckets": asdict(buckets),
        },
        model_path,
    )
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if tabnet_metrics["cost1"]["pnl"] <= baseline_metrics["cost1"]["pnl"]:
        warnings.append("tabnet_parent_did_not_improve_alpha3_cost1")
    if tabnet_metrics["cost1"]["mdd"] < baseline_metrics["cost1"]["mdd"]:
        warnings.append("tabnet_parent_worsened_alpha3_mdd")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and tabnet_metrics["cost1"]["pnl"] > baseline_metrics["cost1"]["pnl"] and tabnet_metrics["cost1"]["mdd"] >= baseline_metrics["cost1"]["mdd"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "normalizer": "expanding_monthly_quantile_fit_prior_months",
        "training_techniques": [
            "TabNet shared sparse feature selector",
            "MMoE 4-expert routed task heads",
            "PCGrad projected multitask gradients",
            "HGB soft-label distillation warmup",
            "oracle-label fine-tuning",
            "gradient scaler alpha=0.10",
            "homoscedastic multitask loss",
            "ReduceLROnPlateau",
            "early stopping",
            "gradient clipping",
            "SWA best-effort",
            "MC dropout inference",
            "virtual sink nodes",
            "entmax15 if installed else sparsemax fallback",
        ],
        "selected_runtime": asdict(best_rt),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 parent replacement using TabNet sparse feature selection, 4-expert MMoE task routing, and PCGrad multitask training. Frozen Alpha3 downstream layers are preserved; only parent decision frame is replaced.",
        "baseline": {"name": "alpha3_hgb_parent", "metrics": baseline_metrics, "score": _score(baseline_metrics)},
        "candidate": {"name": "alpha3_tabnet_mmoe_pcgrad_parent_v2", "selected_runtime": asdict(best_rt), "metrics": tabnet_metrics, "score": _score(tabnet_metrics)},
        "training": training,
        "audit": audit,
        "artifacts": {"model": str(model_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
