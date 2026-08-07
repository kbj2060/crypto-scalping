#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as hgb_sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402


MODEL_ID = "omega4_4_rl_risk_sidecar_20260623"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
OMEGA44_PARENT_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623"
    / "true_3head_tabm_bundle.pt"
)
OMEGA44_LEDGER_SOURCE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_2_trade_risk_sidecar_20260622_v14_topdown_best_parent_e2_train15k_exit15k_exit075_valonly_logrisk_tail050_20260623"
)
MARGIN_BUCKETS = np.asarray([0.06, 0.12, 0.20, 0.28], dtype=np.float32)
LEVERAGE_BUCKETS = np.asarray([1.0, 1.5, 2.0, 2.5], dtype=np.float32)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _action_grid() -> pd.DataFrame:
    rows = []
    idx = 0
    for margin in MARGIN_BUCKETS:
        for leverage in LEVERAGE_BUCKETS:
            rows.append(
                {
                    "action_id": int(idx),
                    "margin_fraction": float(margin),
                    "leverage": float(leverage),
                    "notional": float(margin * leverage),
                }
            )
            idx += 1
    return pd.DataFrame(rows)


def _counterfactual_reward_matrix(
    ledger: pd.DataFrame,
    *,
    tail_budget: float,
    tail_penalty: float,
    liquidation_buffer: float,
    liquidation_penalty: float,
) -> tuple[np.ndarray, pd.DataFrame]:
    grid = _action_grid()
    net = pd.to_numeric(ledger["net_per_notional"], errors="raise").to_numpy(dtype=np.float64)
    mae = pd.to_numeric(ledger["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)
    rewards = np.zeros((len(ledger), len(grid)), dtype=np.float32)
    for _, row in grid.iterrows():
        action_id = int(row["action_id"])
        notional = float(row["notional"])
        leverage = float(row["leverage"])
        account_return = net * notional
        log_growth = np.log1p(np.maximum(account_return, -0.999999))
        tail_excess = np.maximum(-mae * notional - float(tail_budget), 0.0)
        liquidation_excess = np.maximum(-mae * leverage - float(liquidation_buffer), 0.0)
        rewards[:, action_id] = (
            log_growth
            - float(tail_penalty) * tail_excess
            - float(liquidation_penalty) * liquidation_excess
        ).astype(np.float32)
    if not np.isfinite(rewards).all():
        raise RuntimeError("non-finite counterfactual rewards")
    return rewards, grid


def _counterfactual_tail_matrix(
    ledger: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    tail_budget: float,
    liquidation_buffer: float,
    liquidation_penalty: float,
    stop_loss_penalty: float,
) -> np.ndarray:
    mae = pd.to_numeric(ledger["mae_price_move"], errors="raise").to_numpy(dtype=np.float64)
    reason = ledger["reason"].astype(str).to_numpy()
    is_stop = reason == "stop_loss"
    losses = np.zeros((len(ledger), len(grid)), dtype=np.float32)
    for _, row in grid.iterrows():
        action_id = int(row["action_id"])
        notional = float(row["notional"])
        leverage = float(row["leverage"])
        tail_excess = np.maximum(-mae * notional - float(tail_budget), 0.0)
        liquidation_excess = np.maximum(-mae * leverage - float(liquidation_buffer), 0.0)
        stop_loss_cost = np.where(is_stop, float(stop_loss_penalty) * notional, 0.0)
        losses[:, action_id] = (
            tail_excess
            + float(liquidation_penalty) * liquidation_excess
            + stop_loss_cost
        ).astype(np.float32)
    if not np.isfinite(losses).all():
        raise RuntimeError("non-finite counterfactual tail losses")
    return losses


def _nearest_behavior_action(ledger: pd.DataFrame, grid: pd.DataFrame) -> np.ndarray:
    margin = pd.to_numeric(ledger["margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(ledger["leverage"], errors="raise").to_numpy(dtype=np.float64)
    gm = grid["margin_fraction"].to_numpy(dtype=np.float64)
    gl = grid["leverage"].to_numpy(dtype=np.float64)
    dist = ((margin[:, None] - gm[None, :]) / 0.28) ** 2 + ((leverage[:, None] - gl[None, :]) / 2.5) ** 2
    return np.argmin(dist, axis=1).astype(np.int64)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    z = (arr - mean) / std
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32), {
        "columns": list(x.columns),
        "mean": mean,
        "std": std,
    }


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"feature contract mismatch, missing columns: {missing[:20]}")
    arr = x.reindex(columns=cols).to_numpy(dtype=np.float32)
    z = (arr - np.asarray(scaler["mean"], dtype=np.float32)) / np.asarray(scaler["std"], dtype=np.float32)
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class QNet(nn.Module):
    def __init__(self, in_dim: int, actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, in_dim: int, actions: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(128, 96),
            nn.SiLU(),
            nn.Linear(96, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ContinuousCritic(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim + 2, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.Linear(96, 1),
        )

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, a], dim=1)).squeeze(1)


class ContinuousActor(nn.Module):
    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x))


class QuantileQNet(nn.Module):
    def __init__(self, in_dim: int, actions: int, quantiles: int) -> None:
        super().__init__()
        self.actions = int(actions)
        self.quantiles = int(quantiles)
        self.net = nn.Sequential(
            nn.Linear(in_dim, 160),
            nn.LayerNorm(160),
            nn.SiLU(),
            nn.Dropout(0.06),
            nn.Linear(160, 96),
            nn.SiLU(),
            nn.Linear(96, int(actions) * int(quantiles)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(-1, self.actions, self.quantiles)


def _fit_bandit_qnet(x: np.ndarray, rewards: np.ndarray, *, epochs: int, seed: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    model = QNet(x.shape[1], rewards.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=2.0e-4)
    xt = torch.from_numpy(x).to(device)
    rt = torch.from_numpy(rewards).to(device)
    losses: list[float] = []
    for _ in range(int(epochs)):
        pred = model(xt)
        loss = F.smooth_l1_loss(pred, rt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {"kind": "bandit_qnet", "model": model, "loss_last": losses[-1], "loss_min": min(losses)}


def _fit_tail_qnet(x: np.ndarray, tail_losses: np.ndarray, *, epochs: int, seed: int, device: torch.device) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    model = QNet(x.shape[1], tail_losses.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1.2e-3, weight_decay=2.0e-4)
    xt = torch.from_numpy(x).to(device)
    yt = torch.from_numpy(tail_losses).to(device)
    losses: list[float] = []
    for _ in range(int(epochs)):
        pred = model(xt)
        loss = F.smooth_l1_loss(pred, yt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        losses.append(float(loss.detach().cpu()))
    return {"kind": "tail_qnet", "model": model, "loss_last": losses[-1], "loss_min": min(losses)}


def _expectile(values: np.ndarray, tau: float) -> np.ndarray:
    out = np.median(values, axis=1).astype(np.float64)
    for _ in range(80):
        diff = values - out[:, None]
        w = np.where(diff > 0.0, float(tau), 1.0 - float(tau))
        new = (w * values).sum(axis=1) / np.maximum(w.sum(axis=1), 1.0e-12)
        if float(np.max(np.abs(new - out))) < 1.0e-9:
            out = new
            break
        out = new
    return out.astype(np.float32)


def _fit_iql_awac(
    x: np.ndarray,
    rewards: np.ndarray,
    behavior_action: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    q_model = QNet(x.shape[1], rewards.shape[1]).to(device)
    policy = PolicyNet(x.shape[1], rewards.shape[1]).to(device)
    q_opt = torch.optim.AdamW(q_model.parameters(), lr=1.2e-3, weight_decay=2.0e-4)
    p_opt = torch.optim.AdamW(policy.parameters(), lr=1.0e-3, weight_decay=2.0e-4)
    xt = torch.from_numpy(x).to(device)
    rt = torch.from_numpy(rewards).to(device)
    bt = torch.from_numpy(behavior_action.astype(np.int64)).to(device)
    value = _expectile(rewards, tau=0.72)
    adv = rewards - value[:, None]
    soft_target = np.exp(np.clip(adv / 0.020, -8.0, 8.0))
    soft_target = soft_target / np.maximum(soft_target.sum(axis=1, keepdims=True), 1.0e-12)
    st = torch.from_numpy(soft_target.astype(np.float32)).to(device)
    q_losses: list[float] = []
    p_losses: list[float] = []
    for _ in range(int(epochs)):
        pred = q_model(xt)
        q_loss = F.smooth_l1_loss(pred, rt)
        q_opt.zero_grad(set_to_none=True)
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(q_model.parameters(), 2.0)
        q_opt.step()

        logits = policy(xt)
        logp = F.log_softmax(logits, dim=1)
        awr_loss = -(st * logp).sum(dim=1).mean()
        bc_loss = F.cross_entropy(logits, bt)
        p_loss = awr_loss + 0.10 * bc_loss
        p_opt.zero_grad(set_to_none=True)
        p_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 2.0)
        p_opt.step()
        q_losses.append(float(q_loss.detach().cpu()))
        p_losses.append(float(p_loss.detach().cpu()))
    return {
        "kind": "iql_awac",
        "q_model": q_model,
        "policy": policy,
        "q_loss_last": q_losses[-1],
        "policy_loss_last": p_losses[-1],
    }


def _normalize_actions(grid: pd.DataFrame) -> np.ndarray:
    margin = grid["margin_fraction"].to_numpy(dtype=np.float32)
    leverage = grid["leverage"].to_numpy(dtype=np.float32)
    return np.column_stack(
        [
            (margin - float(MARGIN_BUCKETS.min())) / max(float(MARGIN_BUCKETS.max() - MARGIN_BUCKETS.min()), 1.0e-8),
            (leverage - float(LEVERAGE_BUCKETS.min())) / max(float(LEVERAGE_BUCKETS.max() - LEVERAGE_BUCKETS.min()), 1.0e-8),
        ]
    ).astype(np.float32)


def _fit_td3_bc(
    x: np.ndarray,
    rewards: np.ndarray,
    behavior_action: np.ndarray,
    grid: pd.DataFrame,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    critic = ContinuousCritic(x.shape[1]).to(device)
    actor = ContinuousActor(x.shape[1]).to(device)
    c_opt = torch.optim.AdamW(critic.parameters(), lr=1.1e-3, weight_decay=2.0e-4)
    a_opt = torch.optim.AdamW(actor.parameters(), lr=7.5e-4, weight_decay=1.0e-4)
    xt = torch.from_numpy(x).to(device)
    action_grid = _normalize_actions(grid)
    ag = torch.from_numpy(action_grid).to(device)
    behavior = torch.from_numpy(action_grid[behavior_action]).to(device)
    xx = np.repeat(x, len(grid), axis=0)
    aa = np.tile(action_grid, (len(x), 1))
    yy = rewards.reshape(-1)
    xt_all = torch.from_numpy(xx.astype(np.float32)).to(device)
    at_all = torch.from_numpy(aa.astype(np.float32)).to(device)
    yt_all = torch.from_numpy(yy.astype(np.float32)).to(device)
    batch = min(2048, len(yt_all))
    c_losses: list[float] = []
    a_losses: list[float] = []
    for epoch in range(int(epochs)):
        idx = torch.randint(0, len(yt_all), (batch,), device=device)
        pred = critic(xt_all[idx], at_all[idx])
        c_loss = F.smooth_l1_loss(pred, yt_all[idx])
        c_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
        c_opt.step()
        if epoch < max(100, int(epochs * 0.20)):
            act = actor(xt)
            a_loss = F.mse_loss(act, behavior)
        else:
            act = actor(xt)
            q = critic(xt, act)
            bc = F.mse_loss(act, behavior)
            a_loss = -q.mean() + 0.18 * bc
        a_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        a_opt.step()
        c_losses.append(float(c_loss.detach().cpu()))
        a_losses.append(float(a_loss.detach().cpu()))
    return {
        "kind": "td3_bc_continuous",
        "critic": critic,
        "actor": actor,
        "critic_loss_last": c_losses[-1],
        "actor_loss_last": a_losses[-1],
    }


def _fit_dsac_contextual(
    x: np.ndarray,
    rewards: np.ndarray,
    behavior_action: np.ndarray,
    *,
    epochs: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    quantiles = 16
    critic = QuantileQNet(x.shape[1], rewards.shape[1], quantiles).to(device)
    actor = PolicyNet(x.shape[1], rewards.shape[1]).to(device)
    c_opt = torch.optim.AdamW(critic.parameters(), lr=1.0e-3, weight_decay=2.0e-4)
    a_opt = torch.optim.AdamW(actor.parameters(), lr=8.0e-4, weight_decay=1.0e-4)
    xt = torch.from_numpy(x).to(device)
    rt = torch.from_numpy(rewards).to(device)
    bt = torch.from_numpy(behavior_action.astype(np.int64)).to(device)
    c_losses: list[float] = []
    a_losses: list[float] = []
    for _ in range(int(epochs)):
        pred = critic(xt)
        target = rt[:, :, None].expand_as(pred)
        c_loss = F.smooth_l1_loss(pred, target)
        c_opt.zero_grad(set_to_none=True)
        c_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 2.0)
        c_opt.step()

        with torch.no_grad():
            q_mean = critic(xt).mean(dim=2)
        logits = actor(xt)
        prob = F.softmax(logits, dim=1)
        logp = F.log_softmax(logits, dim=1)
        entropy_objective = (prob * (0.006 * logp - q_mean)).sum(dim=1).mean()
        bc_loss = F.cross_entropy(logits, bt)
        a_loss = entropy_objective + 0.08 * bc_loss
        a_opt.zero_grad(set_to_none=True)
        a_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), 2.0)
        a_opt.step()
        c_losses.append(float(c_loss.detach().cpu()))
        a_losses.append(float(a_loss.detach().cpu()))
    return {
        "kind": "dsac_contextual",
        "critic": critic,
        "actor": actor,
        "critic_loss_last": c_losses[-1],
        "actor_loss_last": a_losses[-1],
        "quantiles": quantiles,
    }


@torch.no_grad()
def _discrete_actions(policy: dict[str, Any], x: np.ndarray, *, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x).to(device)
    kind = str(policy["kind"])
    if kind == "bandit_qnet":
        logits = policy["model"](xt)
        return torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    if kind == "iql_awac":
        logits = policy["policy"](xt)
        return torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    if kind == "dsac_contextual":
        logits = policy["actor"](xt)
        return torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
    raise RuntimeError(f"not a discrete policy: {kind}")


@torch.no_grad()
def _guarded_bandit_actions(
    policy: dict[str, Any],
    x: np.ndarray,
    dec: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    device: torch.device,
) -> np.ndarray:
    xt = torch.from_numpy(x).to(device)
    return_score = policy["return_model"](xt)
    tail_score = policy["tail_model"](xt)
    score = return_score - float(policy["tail_lambda"]) * tail_score
    score_np = score.detach().cpu().numpy().astype(np.float64)
    tail_np = tail_score.detach().cpu().numpy().astype(np.float64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    notionals = grid["notional"].to_numpy(dtype=np.float64)
    long_cap = float(policy["long_notional_cap"])
    short_cap = float(policy["short_notional_cap"])
    short_boost_cap = float(policy["short_boost_notional_cap"])
    tail_safe = float(policy["tail_safe_threshold"])
    allowed = np.ones_like(score_np, dtype=bool)
    long_mask = side > 0
    short_mask = side < 0
    long_allowed = np.broadcast_to(notionals[None, :] <= long_cap + 1.0e-12, score_np.shape)
    short_allowed = np.broadcast_to(notionals[None, :] <= short_cap + 1.0e-12, score_np.shape)
    short_boost_allowed = (notionals[None, :] <= short_boost_cap + 1.0e-12) & (tail_np <= tail_safe)
    allowed[long_mask] = long_allowed[long_mask]
    allowed[short_mask] = short_allowed[short_mask] | short_boost_allowed[short_mask]
    score_np[~allowed] = -1.0e12
    inactive = ~omega._active(dec)
    if bool(inactive.any()):
        score_np[inactive, :] = -1.0e12
        score_np[inactive, 0] = 0.0
    return np.argmax(score_np, axis=1).astype(np.int64)


@torch.no_grad()
def _tail_for_actions(policy: dict[str, Any], x: np.ndarray, action_ids: np.ndarray, *, device: torch.device) -> np.ndarray:
    xt = torch.from_numpy(x).to(device)
    tail = policy["tail_model"](xt).detach().cpu().numpy().astype(np.float64)
    return tail[np.arange(len(action_ids)), np.asarray(action_ids, dtype=np.int64)]


@torch.no_grad()
def _continuous_action_values(policy: dict[str, Any], x: np.ndarray, *, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    xt = torch.from_numpy(x).to(device)
    raw = policy["actor"](xt).detach().cpu().numpy().astype(np.float64)
    margin = float(MARGIN_BUCKETS.min()) + raw[:, 0] * float(MARGIN_BUCKETS.max() - MARGIN_BUCKETS.min())
    leverage = float(LEVERAGE_BUCKETS.min()) + raw[:, 1] * float(LEVERAGE_BUCKETS.max() - LEVERAGE_BUCKETS.min())
    return np.clip(margin, float(MARGIN_BUCKETS.min()), float(MARGIN_BUCKETS.max())), np.clip(
        leverage, float(LEVERAGE_BUCKETS.min()), float(LEVERAGE_BUCKETS.max())
    )


def _arrays_from_discrete(dec: pd.DataFrame, action_ids: np.ndarray, grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    margin = grid["margin_fraction"].to_numpy(dtype=np.float64)[np.asarray(action_ids, dtype=np.int64)]
    leverage = grid["leverage"].to_numpy(dtype=np.float64)[np.asarray(action_ids, dtype=np.int64)]
    active = omega._active(dec)
    margin[~active] = 0.0
    leverage[~active] = 0.0
    return margin, leverage


def _arrays_from_continuous(dec: pd.DataFrame, margin: np.ndarray, leverage: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    margin = np.asarray(margin, dtype=np.float64).copy()
    leverage = np.asarray(leverage, dtype=np.float64).copy()
    active = omega._active(dec)
    margin[~active] = 0.0
    leverage[~active] = 0.0
    return margin, leverage


def _hgb_arrays_from_sidecar(
    sidecar: dict[str, Any],
    features: pd.DataFrame,
    dec: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if float(sidecar.get("score_quality_blend", 0.0)) != 0.0:
        raise RuntimeError("HGB overlay does not support score_quality_blend artifacts")
    x_all, _ = hgb_sidecar._feature_matrix(features, list(sidecar["feature_columns"]))
    risk_model = sidecar["model"]
    if bool(sidecar.get("side_split_model", False)):
        side_all = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
        score = hgb_sidecar._predict_side_split_models(risk_model, x_all, side_all)
    else:
        score = np.asarray(risk_model.predict(x_all), dtype=np.float64)
    mapping = dict(sidecar["selected_mapping"])
    margin_cfg = {k: float(mapping[k]) for k in hgb_sidecar.MARGIN_CFG_KEYS}
    leverage_cfg = {k: float(mapping[k]) for k in hgb_sidecar.LEVERAGE_CFG_KEYS if k in mapping}
    margin = hgb_sidecar._risk_margins(
        dec,
        score,
        train_q50=float(sidecar["train_score_q50"]),
        train_iqr=float(sidecar["train_score_iqr"]),
        **margin_cfg,
    )
    if leverage_cfg:
        leverage = hgb_sidecar._risk_leverage(
            dec,
            score,
            train_q50=float(sidecar["train_score_q50"]),
            train_iqr=float(sidecar["train_score_iqr"]),
            **leverage_cfg,
        )
    else:
        leverage = pd.to_numeric(dec["leverage"], errors="raise").to_numpy(dtype=np.float64)
        leverage[~omega._active(dec)] = 0.0
    return margin, leverage


def _hgb_arrays_by_split(data: dict[str, Any], sidecar_path: Path) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    if not sidecar_path.exists():
        raise RuntimeError(f"missing HGB sidecar artifact: {sidecar_path}")
    with sidecar_path.open("rb") as f:
        sidecar = pickle.load(f)
    arrays = {
        split: _hgb_arrays_from_sidecar(sidecar, data["features"][split], data["decisions"][split])
        for split in ("train", "validation", "oos")
    }
    return arrays, sidecar


def _overlay_arrays(
    policy: dict[str, Any],
    split: str,
    x: np.ndarray,
    dec: pd.DataFrame,
    grid: pd.DataFrame,
    *,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    hgb_margin, hgb_leverage = policy["hgb_arrays"][split]
    action_ids = _guarded_bandit_actions(policy["bandit_policy"], x, dec, grid, device=device)
    bandit_margin, bandit_leverage = _arrays_from_discrete(dec, action_ids, grid)
    tail = _tail_for_actions(policy["bandit_policy"], x, action_ids, device=device)
    hgb_notional = np.asarray(hgb_margin, dtype=np.float64) * np.asarray(hgb_leverage, dtype=np.float64)
    bandit_notional = np.asarray(bandit_margin, dtype=np.float64) * np.asarray(bandit_leverage, dtype=np.float64)
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy(dtype=np.int64)
    active = omega._active(dec)
    target = hgb_notional.copy()
    down_mask = bandit_notional < hgb_notional
    downsize_mode = str(policy.get("downsize_mode", "all"))
    if downsize_mode == "long_only":
        down_mask &= side > 0
    elif downsize_mode == "none":
        down_mask &= False
    elif downsize_mode != "all":
        raise RuntimeError(f"unknown overlay downsize_mode: {downsize_mode}")
    target[down_mask] = np.maximum(bandit_notional[down_mask], hgb_notional[down_mask] * float(policy["down_floor"]))
    safe_up = (bandit_notional > hgb_notional) & (tail <= float(policy["tail_safe_threshold"]))
    cap = np.where(side > 0, float(policy["long_notional_cap"]), float(policy["short_notional_cap"]))
    cap[(side < 0) & safe_up] = float(policy["short_boost_notional_cap"])
    target[safe_up] = np.minimum(
        np.minimum(bandit_notional[safe_up], hgb_notional[safe_up] * float(policy["up_cap"])),
        cap[safe_up],
    )
    target[~active] = 0.0
    leverage = np.asarray(hgb_leverage, dtype=np.float64).copy()
    leverage[~active] = 0.0
    margin = np.divide(target, np.maximum(leverage, 1.0e-12), out=np.zeros_like(target), where=leverage > 0.0)
    margin = np.clip(margin, 0.0, float(policy.get("margin_cap", 1.0)))
    margin[~active] = 0.0
    return margin, leverage


def _prepare_omega44(args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    models: dict[str, dict[str, Any]] = bundle["models"]
    base_cols = list(bundle["base_cols"])
    loaded = parent._load_payloads(models, device=device)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    x_train, train_src, train_dec_base = hgb_sidecar._predict_decisions(
        frames["train_raw"],
        oof=True,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )
    x_val, val_src, val_dec_base = hgb_sidecar._predict_decisions(
        frames["val_raw"],
        oof=True,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )
    x_oos, oos_src, oos_dec_base = hgb_sidecar._predict_decisions(
        frames["oos_raw"],
        oof=False,
        models=models,
        base_cols=base_cols,
        quality_threshold=float(args.quality_threshold),
        device=device,
    )
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
    train_features = hgb_sidecar._risk_feature_frame(frames["train_raw"], train_src, train_dec, base_cols, atr_pct=train_atr, feature_mode="parent_outputs")
    val_features = hgb_sidecar._risk_feature_frame(frames["val_raw"], val_src, val_dec, base_cols, atr_pct=val_atr, feature_mode="parent_outputs")
    oos_features = hgb_sidecar._risk_feature_frame(frames["oos_raw"], oos_src, oos_dec, base_cols, atr_pct=oos_atr, feature_mode="parent_outputs")

    ledger_dir = Path(args.ledger_source_dir)
    required = [
        ledger_dir / "train_baseline_trade_ledger.csv",
        ledger_dir / "validation_baseline_trade_ledger.csv",
        ledger_dir / "oos_baseline_trade_ledger.csv",
    ]
    if not bool(args.reuse_ledgers) or not all(p.exists() for p in required):
        train_base_m, train_ledger = hgb_sidecar._replay_with_risk(
            frames["train_raw"],
            x_train,
            train_dec,
            loaded,
            risk_margin_fraction=None,
            risk_leverage=None,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        val_base_m, val_ledger = hgb_sidecar._replay_with_risk(
            frames["val_raw"],
            x_val,
            val_dec,
            loaded,
            risk_margin_fraction=None,
            risk_leverage=None,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        oos_base_m, oos_ledger = hgb_sidecar._replay_with_risk(
            frames["oos_raw"],
            x_oos,
            oos_dec,
            loaded,
            risk_margin_fraction=None,
            risk_leverage=None,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
    else:
        train_ledger = pd.read_csv(required[0])
        val_ledger = pd.read_csv(required[1])
        oos_ledger = pd.read_csv(required[2])
        train_base_m, _ = hgb_sidecar._ledger_metrics_with_margins(frames["train_raw"], train_ledger, None)
        val_base_m, _ = hgb_sidecar._ledger_metrics_with_margins(frames["val_raw"], val_ledger, None)
        oos_base_m, _ = hgb_sidecar._ledger_metrics_with_margins(frames["oos_raw"], oos_ledger, None)

    return {
        "frames": frames,
        "loaded": loaded,
        "base_cols": base_cols,
        "base_x": {"train": x_train, "validation": x_val, "oos": x_oos},
        "decisions": {"train": train_dec, "validation": val_dec, "oos": oos_dec},
        "features": {"train": train_features, "validation": val_features, "oos": oos_features},
        "ledgers": {"train": train_ledger, "validation": val_ledger, "oos": oos_ledger},
        "baseline_metrics": {"train": train_base_m, "validation": val_base_m, "oos": oos_base_m},
        "atr_diag": {"train": train_atr_diag, "validation": val_atr_diag, "oos": oos_atr_diag},
        "fee": fee,
        "slip": slip,
    }


def _evaluate_policy(
    name: str,
    policy: dict[str, Any],
    data: dict[str, Any],
    scaler: dict[str, Any],
    grid: pd.DataFrame,
    out_dir: Path,
    args: argparse.Namespace,
    *,
    device: torch.device,
    log_risk_kwargs: dict[str, float],
) -> dict[str, Any]:
    features = data["features"]
    decisions = data["decisions"]
    ledgers = data["ledgers"]
    frames = data["frames"]
    loaded = data["loaded"]
    base_x = data["base_x"]
    out: dict[str, Any] = {"policy": name, "kind": policy["kind"]}
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    frame_key = {"train": "train_raw", "validation": "val_raw", "oos": "oos_raw"}
    for split in ("train", "validation", "oos"):
        x_all = _standardize_apply(features[split], scaler)
        if "fixed_arrays" in policy:
            arrays[split] = policy["fixed_arrays"][split]
        elif policy["kind"] == "td3_bc_continuous":
            margin, leverage = _continuous_action_values(policy, x_all, device=device)
            arrays[split] = _arrays_from_continuous(decisions[split], margin, leverage)
        elif policy["kind"] == "bandit_guarded_tail":
            action_ids = _guarded_bandit_actions(policy, x_all, decisions[split], grid, device=device)
            arrays[split] = _arrays_from_discrete(decisions[split], action_ids, grid)
        elif policy["kind"] == "hgb_bandit_overlay":
            arrays[split] = _overlay_arrays(policy, split, x_all, decisions[split], grid, device=device)
        else:
            action_ids = _discrete_actions(policy, x_all, device=device)
            arrays[split] = _arrays_from_discrete(decisions[split], action_ids, grid)

    for split in ("validation", "oos"):
        margin, leverage = arrays[split]
        sizing_m, sizing_ledger = hgb_sidecar._ledger_metrics_with_margins(frames[frame_key[split]], ledgers[split], margin, leverage, **log_risk_kwargs)
        replay_m, replay_ledger = hgb_sidecar._replay_with_risk(
            frames[frame_key[split]],
            base_x[split],
            decisions[split],
            loaded,
            risk_margin_fraction=margin,
            risk_leverage=leverage,
            exit_threshold=float(args.exit_threshold),
            fee=float(data["fee"]),
            slip=float(data["slip"]),
            cost_mult=float(args.cost_mult),
            device=device,
        )
        replay_log_m, _ = hgb_sidecar._ledger_metrics_with_margins(frames[frame_key[split]], replay_ledger, None, **log_risk_kwargs)
        for key in ("log_growth_sum", "tail_excess_sum", "liquidation_excess_sum", "log_risk_utility"):
            replay_m[key] = replay_log_m[key]
        sizing_ledger.to_csv(out_dir / f"{split}_{name}_risk_trade_ledger.csv", index=False)
        replay_ledger.to_csv(out_dir / f"{split}_{name}_risk_replayed_trade_ledger.csv", index=False)
        out[split] = sizing_m
        out[f"{split}_full_replay"] = replay_m
    out["train_predicted_action_summary"] = _action_summary(arrays["train"][0], arrays["train"][1], ledgers["train"])
    return out


def _action_summary(margin_array: np.ndarray, leverage_array: np.ndarray, ledger: pd.DataFrame) -> dict[str, Any]:
    idx = pd.to_numeric(ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
    margins = np.asarray(margin_array, dtype=np.float64)[idx]
    leverages = np.asarray(leverage_array, dtype=np.float64)[idx]
    notionals = margins * leverages
    return {
        "rows": int(len(idx)),
        "avg_margin_fraction": float(np.mean(margins)) if len(idx) else 0.0,
        "avg_leverage": float(np.mean(leverages)) if len(idx) else 0.0,
        "avg_notional": float(np.mean(notionals)) if len(idx) else 0.0,
        "margin_counts": {f"{k:.4f}": int(v) for k, v in pd.Series(np.round(margins, 4)).value_counts().sort_index().items()},
        "leverage_counts": {f"{k:.4f}": int(v) for k, v in pd.Series(np.round(leverages, 4)).value_counts().sort_index().items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=OMEGA44_PARENT_BUNDLE)
    ap.add_argument("--ledger-source-dir", type=Path, default=OMEGA44_LEDGER_SOURCE)
    ap.add_argument("--reuse-ledgers", action="store_true")
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.75)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=2400)
    ap.add_argument("--seed", type=int, default=260623)
    ap.add_argument("--out-suffix", default="v1_full")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--log-tail-budget", type=float, default=0.02)
    ap.add_argument("--log-tail-penalty", type=float, default=0.5)
    ap.add_argument("--log-liquidation-buffer", type=float, default=0.12)
    ap.add_argument("--log-liquidation-penalty", type=float, default=0.25)
    ap.add_argument("--max-validation-mdd-abs", type=float, default=7.0)
    ap.add_argument("--hgb-sidecar-path", type=Path, default=None)
    ap.add_argument("--guard-tail-lambda", type=float, default=1.0)
    ap.add_argument("--guard-stop-loss-penalty", type=float, default=0.035)
    ap.add_argument("--guard-long-notional-cap", type=float, default=0.40)
    ap.add_argument("--guard-short-notional-cap", type=float, default=0.56)
    ap.add_argument("--guard-short-boost-notional-cap", type=float, default=0.70)
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    print("stage=prepare_omega44_data", flush=True)
    data = _prepare_omega44(args, device)
    log_risk_kwargs = {
        "tail_budget": float(args.log_tail_budget),
        "tail_penalty": float(args.log_tail_penalty),
        "liquidation_buffer": float(args.log_liquidation_buffer),
        "liquidation_penalty": float(args.log_liquidation_penalty),
    }

    train_ledger = data["ledgers"]["train"]
    train_features = data["features"]["train"]
    train_idx = pd.to_numeric(train_ledger["entry_signal_i"], errors="raise").to_numpy(dtype=np.int64)
    x_train_trade_df = train_features.iloc[train_idx].reset_index(drop=True)
    x_train, scaler = _standardize_fit(x_train_trade_df)
    rewards, grid = _counterfactual_reward_matrix(train_ledger, **log_risk_kwargs)
    tail_losses = _counterfactual_tail_matrix(
        train_ledger,
        grid,
        tail_budget=float(args.log_tail_budget),
        liquidation_buffer=float(args.log_liquidation_buffer),
        liquidation_penalty=float(args.log_liquidation_penalty),
        stop_loss_penalty=float(args.guard_stop_loss_penalty),
    )
    behavior_action = _nearest_behavior_action(train_ledger, grid)

    print("stage=train_bandit_qnet", flush=True)
    policies: list[dict[str, Any]] = []
    bandit_policy = _fit_bandit_qnet(x_train, rewards, epochs=int(args.epochs), seed=int(args.seed) + 1, device=device)
    policies.append(bandit_policy)

    print("stage=train_tail_guard_qnet", flush=True)
    tail_policy = _fit_tail_qnet(x_train, tail_losses, epochs=int(args.epochs), seed=int(args.seed) + 5, device=device)
    guarded_bandit_policy = {
        "name": "bandit_guarded_tail",
        "kind": "bandit_guarded_tail",
        "return_model": bandit_policy["model"],
        "tail_model": tail_policy["model"],
        "return_loss_last": bandit_policy["loss_last"],
        "tail_loss_last": tail_policy["loss_last"],
        "tail_lambda": float(args.guard_tail_lambda),
        "tail_safe_threshold": float(np.quantile(tail_losses, 0.25)),
        "long_notional_cap": float(args.guard_long_notional_cap),
        "short_notional_cap": float(args.guard_short_notional_cap),
        "short_boost_notional_cap": float(args.guard_short_boost_notional_cap),
        "stop_loss_penalty": float(args.guard_stop_loss_penalty),
    }
    policies.append(guarded_bandit_policy)

    print("stage=train_iql_awac", flush=True)
    policies.append(_fit_iql_awac(x_train, rewards, behavior_action, epochs=int(args.epochs), seed=int(args.seed) + 2, device=device))

    print("stage=train_td3_bc_continuous", flush=True)
    policies.append(_fit_td3_bc(x_train, rewards, behavior_action, grid, epochs=int(args.epochs), seed=int(args.seed) + 3, device=device))

    print("stage=train_dsac_contextual", flush=True)
    policies.append(_fit_dsac_contextual(x_train, rewards, behavior_action, epochs=int(args.epochs), seed=int(args.seed) + 4, device=device))

    hgb_sidecar_path = Path(args.hgb_sidecar_path) if args.hgb_sidecar_path is not None else Path(args.ledger_source_dir) / "risk_sidecar.pkl"
    print("stage=load_hgb_overlay_base", flush=True)
    hgb_arrays, hgb_artifact = _hgb_arrays_by_split(data, hgb_sidecar_path)
    overlay_specs = [
        ("all", 0.75, 1.10),
        ("all", 0.85, 1.15),
        ("none", 1.00, 1.15),
        ("long_only", 0.75, 1.10),
        ("long_only", 0.85, 1.15),
    ]
    for downsize_mode, down_floor, up_cap in overlay_specs:
        policies.append(
            {
                "name": (
                    f"hgb_bandit_overlay_{downsize_mode}_"
                    f"down{int(round(down_floor * 100)):03d}_up{int(round(up_cap * 100)):03d}"
                ),
                "kind": "hgb_bandit_overlay",
                "hgb_arrays": hgb_arrays,
                "hgb_sidecar_path": str(hgb_sidecar_path),
                "hgb_selected_mapping": dict(hgb_artifact["selected_mapping"]),
                "bandit_policy": guarded_bandit_policy,
                "tail_safe_threshold": float(guarded_bandit_policy["tail_safe_threshold"]),
                "downsize_mode": str(downsize_mode),
                "down_floor": float(down_floor),
                "up_cap": float(up_cap),
                "margin_cap": 1.0,
                "long_notional_cap": float(args.guard_long_notional_cap),
                "short_notional_cap": float(args.guard_short_notional_cap),
                "short_boost_notional_cap": float(args.guard_short_boost_notional_cap),
            }
        )

    print("stage=evaluate_policies", flush=True)
    results = []
    for policy in policies:
        results.append(
            _evaluate_policy(
                str(policy.get("name", policy["kind"])),
                policy,
                data,
                scaler,
                grid,
                out_dir,
                args,
                device=device,
                log_risk_kwargs=log_risk_kwargs,
            )
        )

    validation_mdd_floor = -abs(float(args.max_validation_mdd_abs))
    eligible = [r for r in results if float(r["validation"]["mdd"]) >= validation_mdd_floor]
    if not eligible:
        eligible = list(results)
    selected = max(
        eligible,
        key=lambda r: (
            float(r["validation"]["log_risk_utility"]),
            float(r["validation"]["mdd"]),
            float(r["validation"]["pnl"]),
        ),
    )

    artifact = {
        "model_id": MODEL_ID,
        "selected_policy": selected["policy"],
        "scaler": scaler,
        "action_grid": grid.to_dict(orient="records"),
        "policies": {
            str(p.get("name", p["kind"])): {
                k: v
                for k, v in p.items()
                if k
                not in {
                    "model",
                    "q_model",
                    "policy",
                    "critic",
                    "actor",
                    "return_model",
                    "tail_model",
                    "bandit_policy",
                    "hgb_arrays",
                }
            }
            for p in policies
        },
        "state_dicts": {},
        "contract": "Omega4.4 parent/exit/ATR SLTP unchanged; RL sidecar only chooses entry-time margin_fraction/leverage; notional=margin_fraction*leverage; OOS readout only.",
    }
    for p in policies:
        state: dict[str, Any] = {}
        for key in ("model", "q_model", "policy", "critic", "actor", "return_model", "tail_model"):
            if key in p:
                state[key] = p[key].state_dict()
        artifact["state_dicts"][str(p.get("name", p["kind"]))] = state
    torch.save(artifact, out_dir / "rl_risk_sidecar.pt")

    for split in ("train", "validation", "oos"):
        data["ledgers"][split].to_csv(out_dir / f"{split}_baseline_trade_ledger.csv", index=False)

    ranking = pd.DataFrame(
        [
            {
                "policy": r["policy"],
                "validation_pnl": r["validation"]["pnl"],
                "validation_mdd": r["validation"]["mdd"],
                "validation_wr": r["validation"]["wr"],
                "validation_trades": r["validation"]["trades"],
                "validation_avg_notional": r["validation"]["avg_notional"],
                "validation_avg_margin": r["validation"]["avg_margin_fraction"],
                "validation_avg_leverage": r["validation"]["avg_leverage"],
                "validation_log_risk_utility": r["validation"]["log_risk_utility"],
                "oos_pnl": r["oos"]["pnl"],
                "oos_mdd": r["oos"]["mdd"],
                "oos_wr": r["oos"]["wr"],
                "oos_trades": r["oos"]["trades"],
                "oos_avg_notional": r["oos"]["avg_notional"],
                "oos_avg_margin": r["oos"]["avg_margin_fraction"],
                "oos_avg_leverage": r["oos"]["avg_leverage"],
                "oos_log_risk_utility": r["oos"]["log_risk_utility"],
                "validation_full_replay_pnl": r["validation_full_replay"]["pnl"],
                "validation_full_replay_mdd": r["validation_full_replay"]["mdd"],
                "validation_full_replay_log_risk_utility": r["validation_full_replay"]["log_risk_utility"],
                "oos_full_replay_pnl": r["oos_full_replay"]["pnl"],
                "oos_full_replay_mdd": r["oos_full_replay"]["mdd"],
                "oos_full_replay_log_risk_utility": r["oos_full_replay"]["log_risk_utility"],
            }
            for r in results
        ]
    ).sort_values(["validation_log_risk_utility", "validation_mdd", "validation_pnl"], ascending=[False, False, False])
    ranking.to_csv(out_dir / "rl_policy_ranking.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "display_version": "Omega4.4 RL risk sidecar full test",
        "baseline_model": "omega4_4_topdown_reproducible_architecture_baseline_20260623",
        "baseline_bundle": str(args.baseline_bundle),
        "ledger_source_dir": str(args.ledger_source_dir),
        "design": "Omega4.4 parent, quality gate, exit head, and ATR price-move SLTP are frozen. Risk sidecar is replaced by offline RL-style policies that choose entry-time margin_fraction and leverage.",
        "algorithms": [
            "bandit_qnet_full_counterfactual",
            "bandit_guarded_tail_qnet",
            "hgb_bandit_guarded_overlay",
            "iql_awac_discrete_behavior_regularized",
            "td3_bc_continuous_behavior_regularized",
            "dsac_contextual_distributional_discrete",
        ],
        "contract": {
            "quality_threshold": float(args.quality_threshold),
            "exit_threshold": float(args.exit_threshold),
            "selection_scope": "validation_only",
            "oos_usage_policy": "OOS excluded from policy selection; OOS is selected-row readout only.",
            "risk_action_space": {
                "margin_fraction_buckets": [float(x) for x in MARGIN_BUCKETS],
                "leverage_buckets": [float(x) for x in LEVERAGE_BUCKETS],
                "discrete_actions": int(len(grid)),
                "continuous_bounds": {
                    "margin_fraction_min": float(MARGIN_BUCKETS.min()),
                    "margin_fraction_max": float(MARGIN_BUCKETS.max()),
                    "leverage_min": float(LEVERAGE_BUCKETS.min()),
                    "leverage_max": float(LEVERAGE_BUCKETS.max()),
                },
            },
            "reward": {
                "formula": "log(1 + net_per_notional * margin_fraction * leverage) - tail_penalty * tail_excess - liquidation_penalty * liquidation_excess",
                **log_risk_kwargs,
            },
            "notional_contract": "notional = margin_fraction * leverage",
            "sltp_contract": "raw directional price_move barriers; margin/notional do not move TP/SL lines",
            "full_replay_dynamic_exit": "diagnostic only",
            "guard": {
                "tail_lambda": float(args.guard_tail_lambda),
                "tail_safe_threshold": float(guarded_bandit_policy["tail_safe_threshold"]),
                "stop_loss_penalty": float(args.guard_stop_loss_penalty),
                "long_notional_cap": float(args.guard_long_notional_cap),
                "short_notional_cap": float(args.guard_short_notional_cap),
                "short_boost_notional_cap": float(args.guard_short_boost_notional_cap),
                "hgb_overlay_base": str(hgb_sidecar_path),
            },
        },
        "data": {
            "train_trades": int(len(data["ledgers"]["train"])),
            "validation_trades": int(len(data["ledgers"]["validation"])),
            "oos_trades": int(len(data["ledgers"]["oos"])),
            "risk_feature_columns": list(scaler["columns"]),
        },
        "baseline_metrics": data["baseline_metrics"],
        "selected": selected,
        "results": results,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "ranking": str(out_dir / "rl_policy_ranking.csv"),
            "rl_risk_sidecar": str(out_dir / "rl_risk_sidecar.pt"),
        },
        "dsac_scope_note": "DSAC is tested here as a contextual distributional risk sidecar. It is not a full bar-level SAC/DSAC trading environment because the promoted Omega4.4 contract freezes parent entry and exit timing.",
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    with (out_dir / "rl_risk_sidecar.pkl").open("wb") as f:
        pickle.dump({"selected_policy": selected["policy"], "ranking": ranking.to_dict(orient="records")}, f)
    print(json.dumps({"report": str(out_dir / "report.json"), "selected": selected, "ranking": ranking.head(8).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
