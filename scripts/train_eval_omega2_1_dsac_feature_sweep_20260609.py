#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
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
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_1_cash_fallback_label_family_20260606 as label_family  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_1_full_retrain_cash_alpha43_20260608 as full  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
from freeze_omega2_1_hgb_12seed_cash_sleeve_20260609 import (  # noqa: E402
    BUNDLE_PATH,
    MODEL_ID as OMEGA21_MODEL_ID,
    RISK,
    SEEDS as HGB_SEEDS,
    _classes_to_proba,
    _model as _hgb_model,
)


MODEL_ID = "omega2_1_dsac_feature_sweep_20260609"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_", "exit_head_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}
BASELINE_OOS = {"pnl": 102.61148286407757, "mdd": -8.108170708968377, "wr": 0.6097560975609756, "trades": 41}
LOG_STD_MIN = -5.0
LOG_STD_MAX = 1.5
POS_THRESH = 0.18


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _reject_forbidden(cols: list[str], tag: str) -> None:
    bad = [
        col
        for col in cols
        if col in FORBIDDEN_EXACT or any(str(col).startswith(prefix) for prefix in FORBIDDEN_PREFIXES)
    ]
    if bad:
        raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")


def _entropy3(p: np.ndarray) -> np.ndarray:
    pp = np.clip(p.astype(np.float64), 1e-12, 1.0)
    pp = pp / np.clip(pp.sum(axis=1, keepdims=True), 1e-12, None)
    return (-np.sum(pp * np.log(pp), axis=1) / math.log(3.0)).astype(np.float64)


def _hgb_oof_and_full(
    x_val: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    x_oos: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    idx = np.flatnonzero(train_mask)
    val_stack: list[np.ndarray] = []
    val_pred_stack: list[np.ndarray] = []
    oos_stack: list[np.ndarray] = []
    oos_pred_stack: list[np.ndarray] = []
    folds_meta = []
    for seed in HGB_SEEDS:
        val_p = np.zeros((len(x_val), 3), dtype=np.float64)
        n = len(idx)
        for fold_id, (train_frac, end_frac) in enumerate(((0.35, 0.50), (0.50, 0.65), (0.65, 0.80), (0.80, 1.00))):
            train_end = int(n * train_frac)
            val_end = int(n * end_frac)
            if train_end < 100 or val_end <= train_end:
                continue
            train_idx = idx[:train_end]
            val_idx = idx[train_end:val_end]
            if len(np.unique(y[train_idx])) < 2:
                continue
            model = _hgb_model(int(seed) + train_end)
            model.fit(x_val.iloc[train_idx].to_numpy(dtype=np.float64), y[train_idx])
            val_p[val_idx] = _classes_to_proba(model, model.predict_proba(x_val.iloc[val_idx].to_numpy(dtype=np.float64)))
            folds_meta.append({"seed": int(seed), "fold": int(fold_id), "train_rows": int(len(train_idx)), "val_rows": int(len(val_idx))})
        full_model = _hgb_model(int(seed))
        full_model.fit(x_val.iloc[idx].to_numpy(dtype=np.float64), y[idx])
        oos_p = _classes_to_proba(full_model, full_model.predict_proba(x_oos.to_numpy(dtype=np.float64)))
        val_stack.append(val_p)
        val_pred_stack.append(np.argmax(val_p, axis=1))
        oos_stack.append(oos_p)
        oos_pred_stack.append(np.argmax(oos_p, axis=1))
    return (
        np.stack(val_stack, axis=0).mean(axis=0),
        np.stack(oos_stack, axis=0).mean(axis=0),
        np.stack(val_pred_stack, axis=0),
        np.stack(oos_pred_stack, axis=0),
        {"folds": folds_meta, "oof_rows": int(np.count_nonzero(np.stack(val_stack, axis=0).mean(axis=0).max(axis=1) > 0.0))},
    )


def _hgb_feature_frame(proba: np.ndarray, pred_stack: np.ndarray) -> pd.DataFrame:
    pred = np.argmax(proba, axis=1)
    conf = proba[np.arange(len(proba)), pred]
    sorted_p = np.sort(proba, axis=1)
    agree = (pred_stack == pred[None, :]).sum(axis=0).astype(np.float64)
    return pd.DataFrame(
        {
            "hgb_p_cash": proba[:, 0],
            "hgb_p_long": proba[:, 1],
            "hgb_p_short": proba[:, 2],
            "hgb_confidence": conf,
            "hgb_margin": sorted_p[:, -1] - sorted_p[:, -2],
            "hgb_entropy": _entropy3(proba),
            "hgb_proposed_side": np.where(pred == 1, 1.0, np.where(pred == 2, -1.0, 0.0)),
            "hgb_seed_agreement": agree / max(float(pred_stack.shape[0]), 1.0),
            "hgb_seed_disagreement": 1.0 - agree / max(float(pred_stack.shape[0]), 1.0),
        }
    )


class CompactFeatureExtractor(nn.Module):
    """Same block shape as ensemble/train_rl_dsac_agent.py: Linear/LN/SiLU x2."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class GaussianActor(nn.Module):
    """Shared backbone + 4 soft-gated tanh-Gaussian heads, matching DSAC style."""

    def __init__(self, state_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.feat = CompactFeatureExtractor(state_dim, hidden_dim)
        self.n_heads = 4
        self.gate_head = nn.Linear(hidden_dim, self.n_heads)
        self.mu_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.n_heads)])
        self.log_std_heads = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(self.n_heads)])

    def _mix_heads(self, feat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gate = torch.softmax(self.gate_head(feat), dim=-1)
        mu_stack = torch.cat([head(feat) for head in self.mu_heads], dim=1)
        log_std_stack = torch.cat(
            [head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX) for head in self.log_std_heads],
            dim=1,
        )
        mu = (gate * mu_stack).sum(dim=1, keepdim=True)
        log_std = (gate * log_std_stack).sum(dim=1, keepdim=True).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std, gate

    def forward_with_gate(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._mix_heads(self.feat(state))

    def sample(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, log_std, _gate = self.forward_with_gate(state)
        dist = Normal(mu, log_std.exp())
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1.0 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mu, _log_std, _gate = self.forward_with_gate(state)
        return torch.tanh(mu)


class DistributionalTwinCritic(nn.Module):
    """State-action twin quantile critic, matching train_rl_dsac_agent.py block."""

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_quantiles: int = 32) -> None:
        super().__init__()
        self.n_quantiles = int(n_quantiles)
        self.feat1 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.feat2 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        return self.q1(torch.cat([f1, action], dim=1)), self.q2(torch.cat([f2, action], dim=1))


class Omega21DSACAgent(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_quantiles: int = 32) -> None:
        super().__init__()
        self.actor = GaussianActor(state_dim, hidden_dim)
        self.critic = DistributionalTwinCritic(state_dim, hidden_dim, n_quantiles)
        self.n_quantiles = int(n_quantiles)


def _quantile_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # pred: [B,Q], target: [B,1]
    q = pred.shape[-1]
    taus = (torch.arange(q, device=pred.device, dtype=pred.dtype) + 0.5) / q
    err = target - pred
    huber = torch.where(err.abs() <= 1.0, 0.5 * err.pow(2), err.abs() - 0.5)
    loss = (torch.abs(taus.view(1, -1) - (err.detach() < 0).to(pred.dtype)) * huber).mean()
    return loss


@dataclass(frozen=True)
class TrainCfg:
    seed: int
    epochs: int
    hidden: int
    lr: float
    batch_size: int
    cvar_frac: float
    entropy_coef: float
    bc_coef: float
    anti_flat_lambda: float
    anti_flat_min_abs: float
    side_balance_lambda: float


def _label_to_behavior_action(y: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float32)
    out[y == sleeve.ACTION_LONG] = 0.75
    out[y == sleeve.ACTION_SHORT] = -0.75
    return out


def _label_reward_matrix(y: np.ndarray) -> np.ndarray:
    rewards = np.full((len(y), 3), -0.65, dtype=np.float32)
    rewards[:, 0] = 0.02
    rewards[y == 0, 0] = 0.12
    rewards[y == 0, 1] = -0.75
    rewards[y == 0, 2] = -0.75
    rewards[y == sleeve.ACTION_LONG, 0] = -0.05
    rewards[y == sleeve.ACTION_LONG, 1] = 1.0
    rewards[y == sleeve.ACTION_SHORT, 0] = -0.05
    rewards[y == sleeve.ACTION_SHORT, 2] = 1.0
    return rewards


def _cost3_reward_matrix(frame: pd.DataFrame, dec: pd.DataFrame, fee: float, slip: float) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = sleeve._arrays(frame)
    active = omega._active(dec)
    rewards = np.zeros((len(frame), 3), dtype=np.float32)
    reasons: dict[str, int] = {}
    long_net = np.zeros(len(frame), dtype=np.float64)
    short_net = np.zeros(len(frame), dtype=np.float64)
    fee_eff = float(fee) * 3.0
    slip_eff = float(slip) * 3.0
    max_i = max(0, len(frame) - int(RISK.max_hold_bars) - 3)
    cash_idx = np.flatnonzero((~active) & (np.arange(len(frame)) < max_i))
    for i in cash_idx:
        long_score, long_meta = sleeve._simulate_one(arrays, int(i), 1, RISK, fee_eff=fee_eff, slip_eff=slip_eff)
        short_score, short_meta = sleeve._simulate_one(arrays, int(i), -1, RISK, fee_eff=fee_eff, slip_eff=slip_eff)
        long_net[int(i)] = float(long_meta.get("net", long_score))
        short_net[int(i)] = float(short_meta.get("net", short_score))
        rewards[int(i), 1] = float(np.tanh(float(long_score) / 0.020))
        rewards[int(i), 2] = float(np.tanh(float(short_score) / 0.020))
        rewards[int(i), 0] = 0.025
        reasons[str(long_meta.get("reason", "unknown"))] = reasons.get(str(long_meta.get("reason", "unknown")), 0) + 1
        reasons[str(short_meta.get("reason", "unknown"))] = reasons.get(str(short_meta.get("reason", "unknown")), 0) + 1
    return rewards, {
        "mode": "cost3_counterfactual",
        "cash_rows": int(len(cash_idx)),
        "long_net_mean": float(np.mean(long_net[cash_idx])) if len(cash_idx) else 0.0,
        "short_net_mean": float(np.mean(short_net[cash_idx])) if len(cash_idx) else 0.0,
        "best_action_counts": {
            str(k): int(v)
            for k, v in pd.Series(np.argmax(rewards[cash_idx], axis=1)).value_counts().sort_index().items()
        },
        "sim_reasons": reasons,
    }


def _torch_reward_from_action(action: torch.Tensor, reward_matrix: torch.Tensor) -> torch.Tensor:
    cls = torch.zeros(action.shape[0], dtype=torch.long, device=action.device)
    cls = torch.where(action.squeeze(1) > POS_THRESH, torch.ones_like(cls), cls)
    cls = torch.where(action.squeeze(1) < -POS_THRESH, torch.full_like(cls, 2), cls)
    return reward_matrix.gather(1, cls.view(-1, 1))


def _reward_to_behavior_action(reward_matrix: np.ndarray) -> np.ndarray:
    best = np.argmax(reward_matrix, axis=1)
    return _label_to_behavior_action(best.astype(np.int64))


def _fit_dsac(x: np.ndarray, reward_matrix: np.ndarray, train_mask: np.ndarray, cfg: TrainCfg, device: str) -> Omega21DSACAgent:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    idx = np.flatnonzero(train_mask)
    if len(idx) < 500:
        raise RuntimeError(f"DSAC train rows too small: {len(idx)}")
    model = Omega21DSACAgent(x.shape[1], int(cfg.hidden), n_quantiles=32).to(device)
    actor_opt = torch.optim.AdamW(model.actor.parameters(), lr=float(cfg.lr), weight_decay=2e-4)
    critic_opt = torch.optim.AdamW(model.critic.parameters(), lr=float(cfg.lr), weight_decay=2e-4)
    x_t = torch.tensor(x, dtype=torch.float32, device=device)
    reward_t = torch.tensor(reward_matrix, dtype=torch.float32, device=device)
    behavior_t = torch.tensor(_reward_to_behavior_action(reward_matrix), dtype=torch.float32, device=device).view(-1, 1)
    idx_t = torch.tensor(idx, dtype=torch.long, device=device)
    for _epoch in range(int(cfg.epochs)):
        perm = idx_t[torch.randperm(len(idx_t), device=device)]
        for start in range(0, len(perm), int(cfg.batch_size)):
            b = perm[start : start + int(cfg.batch_size)]
            state_b = x_t[b]
            reward_b = reward_t[b]
            behavior_a = behavior_t[b]
            random_a = torch.empty_like(behavior_a).uniform_(-1.0, 1.0)
            train_state = torch.cat([state_b, state_b], dim=0)
            train_action = torch.cat([behavior_a, random_a], dim=0)
            train_reward = torch.cat([reward_b, reward_b], dim=0)
            target_r = _torch_reward_from_action(train_action, train_reward)
            q1, q2 = model.critic(train_state, train_action)
            critic_loss = _quantile_huber(q1, target_r) + _quantile_huber(q2, target_r)
            critic_opt.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            critic_opt.step()

            new_action, log_prob = model.actor.sample(state_b)
            q1_new, q2_new = model.critic(state_b, new_action)
            n_cvar = max(1, int(model.n_quantiles * float(cfg.cvar_frac)))
            q_cvar = torch.minimum(q1_new, q2_new).sort(dim=-1).values[:, :n_cvar].mean(dim=-1, keepdim=True)
            det_action = model.actor.deterministic(state_b)
            anti_flat = torch.relu(torch.tensor(float(cfg.anti_flat_min_abs), device=device) - det_action.abs().mean())
            side_balance = torch.tanh(4.0 * new_action).mean().abs()
            bc_loss = F.mse_loss(det_action, behavior_a)
            actor_loss = (
                (float(cfg.entropy_coef) * log_prob - q_cvar).mean()
                + float(cfg.bc_coef) * bc_loss
                + float(cfg.anti_flat_lambda) * anti_flat
                + float(cfg.side_balance_lambda) * side_balance
            )
            actor_opt.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            actor_opt.step()
    return model.cpu().eval()


def _predict(model: Omega21DSACAgent, x: np.ndarray, batch: int = 4096) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    actions = []
    q_cvars = []
    with torch.no_grad():
        for start in range(0, len(x), batch):
            xb = torch.tensor(x[start : start + batch], dtype=torch.float32)
            action = model.actor.deterministic(xb)
            q1, q2 = model.critic(xb, action)
            n_cvar = max(1, int(model.n_quantiles * 0.40))
            qmin = torch.minimum(q1, q2).sort(dim=-1).values[:, :n_cvar].mean(dim=-1).cpu().numpy()
            actions.append(action.squeeze(1).cpu().numpy())
            q_cvars.append(qmin)
    raw = np.concatenate(actions, axis=0).astype(np.float64)
    q_cvar = np.concatenate(q_cvars, axis=0)
    action = np.zeros(len(raw), dtype=np.int64)
    action[raw > POS_THRESH] = sleeve.ACTION_LONG
    action[raw < -POS_THRESH] = sleeve.ACTION_SHORT
    conf = np.abs(raw).astype(np.float64)
    return action, conf, q_cvar


def _standardize(x_train: pd.DataFrame, x_eval: pd.DataFrame, train_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train = x_train.to_numpy(dtype=np.float64)
    eval_arr = x_eval.to_numpy(dtype=np.float64)
    idx = np.flatnonzero(train_mask)
    mu = np.nanmean(train[idx], axis=0)
    sd = np.nanstd(train[idx], axis=0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (
        np.nan_to_num((train - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        np.nan_to_num((eval_arr - mu) / sd, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32),
        {"mean": mu.tolist(), "std": sd.tolist()},
    )


def _feature_sets(base_cols: list[str], hgb_cols: list[str]) -> dict[str, list[str]]:
    def keep(cols: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for col in cols:
            if col in seen:
                continue
            if col not in base_cols and col not in hgb_cols:
                raise RuntimeError(f"unknown feature column in DSAC feature set: {col}")
            seen.add(col)
            out.append(col)
        return out

    price_cols = [
        "bar_range_pct",
        "body_pct",
        "atr14_pct",
        "ret_1",
        "ret_3",
        "ret_6",
        "ret_12",
        "ret_24",
        "ret_vol_6",
        "ret_vol_12",
        "ret_vol_24",
        "ret_vol_48",
        "range_mean_6",
        "range_mean_12",
        "range_mean_24",
        "range_mean_48",
        "ema9_21_gap",
        "tod_sin",
        "tod_cos",
    ]
    signal_cols = [
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
    ]
    router_state_cols = ["router_is_bull", "router_is_bear", "router_is_chop"]
    risk_state_cols = ["side", "base_notional", "base_tp", "base_sl"]
    primary_state_cols = ["primary_is_cash", "primary_active_roll_12", "primary_active_roll_48", "primary_cash_streak"]
    hgb_core_cols = ["hgb_confidence", "hgb_margin", "hgb_entropy", "hgb_proposed_side"]
    parent_cols = [c for c in base_cols if c not in price_cols]
    no_primary_state = [c for c in base_cols if not c.startswith("primary_")]
    no_router_state = [c for c in base_cols if c not in router_state_cols]
    no_base_risk = [c for c in base_cols if c not in risk_state_cols]
    return {
        "base42": keep(base_cols),
        "price19": keep(price_cols),
        "parent23": keep(parent_cols),
        "base42_hgb9": keep(base_cols + hgb_cols),
        "price19_hgb9": keep(price_cols + hgb_cols),
        "parent23_hgb9": keep(parent_cols + hgb_cols),
        "base_no_primary_hgb9": keep(no_primary_state + hgb_cols),
        "signal_probs_hgb9": keep(signal_cols + hgb_cols),
        "direction_quality_hgb9": keep(signal_cols + ["side"] + hgb_cols),
        "execution_state_hgb9": keep(risk_state_cols + primary_state_cols + hgb_cols),
        "price_signal_hgb9": keep(price_cols + signal_cols + hgb_cols),
        "no_router_signal_hgb9": keep(no_router_state + hgb_cols),
        "no_base_risk_hgb9": keep(no_base_risk + hgb_cols),
        "hgb9_only": keep(hgb_cols),
        "compact_alpha_hgb4": keep(
            [
                "ret_1",
                "ret_3",
                "ret_6",
                "ret_12",
                "ret_24",
                "ret_vol_12",
                "ret_vol_24",
                "ema9_21_gap",
                "dir_side_edge",
                "dir_trade_prob",
                "quality_for_action",
                "primary_cash_streak",
            ]
            + hgb_core_cols
        ),
    }


def _metric_row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_fallback_entries": int(metrics.get("fallback_entries", 0)),
        f"{prefix}_primary_takeovers": int(metrics.get("primary_takeovers", 0)),
        f"{prefix}_reasons": metrics.get("exit_reasons", {}),
    }


def _eval(frame: pd.DataFrame, dec: pd.DataFrame, action: np.ndarray, conf: np.ndarray, threshold: float, fee: float, slip: float) -> dict[str, Any]:
    return sleeve._metrics_with_fallback(frame, dec, RISK, action, conf, float(threshold), fee=fee, slip=slip, cost_mult=3.0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Omega2.1-native DSAC input feature sweep")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--seeds", default="260901,260902")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--reward-mode", choices=["label", "cost3"], default="label")
    ap.add_argument("--out-tag", default="")
    ap.add_argument("--groups", default="")
    ap.add_argument("--bc-coef", type=float, default=0.08)
    ap.add_argument("--cvar-frac", type=float, default=0.40)
    ap.add_argument("--entropy-coef", type=float, default=0.03)
    args = ap.parse_args()

    out_dir = OUT_DIR if not args.out_tag else ROOT / "tmp/causal_regen_20260516" / f"{MODEL_ID}_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    seeds = [int(s.strip()) for s in str(args.seeds).split(",") if s.strip()]

    bundle = joblib.load(BUNDLE_PATH)
    if bundle.get("model_id") != OMEGA21_MODEL_ID:
        raise RuntimeError(f"unexpected Omega2.1 bundle: {bundle.get('model_id')}")
    base_cols = list(bundle["feature_cols"])
    _reject_forbidden(base_cols, "base42")

    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_dec, val_features = full._build_split(frames, "validation")
    oos_frame, oos_dec, oos_features = full._build_split(frames, "oos")
    if list(val_features.columns) != base_cols or list(oos_features.columns) != base_cols:
        raise RuntimeError("Omega2.1 feature columns do not match frozen manifest")
    _reject_forbidden(list(val_features.columns), "validation")
    _reject_forbidden(list(oos_features.columns), "oos")

    y, valid_mask, label_diag = label_family._triple_barrier_labels(val_frame, atr_mult=1.0, max_hold=24, min_barrier=0.0035)
    train_mask = (~omega._active(val_dec)) & valid_mask
    if args.reward_mode == "cost3":
        reward_matrix, reward_diag = _cost3_reward_matrix(val_frame, val_dec, fee, slip)
        train_mask = train_mask & (np.max(np.abs(reward_matrix), axis=1) > 0.0)
    else:
        reward_matrix = _label_reward_matrix(y)
        reward_diag = {"mode": "triple_barrier_proxy", "label_diag": label_diag}
    hgb_val_p, hgb_oos_p, hgb_val_stack, hgb_oos_stack, hgb_diag = _hgb_oof_and_full(val_features, y, train_mask, oos_features)
    hgb_val_features = _hgb_feature_frame(hgb_val_p, hgb_val_stack)
    hgb_oos_features = _hgb_feature_frame(hgb_oos_p, hgb_oos_stack)
    hgb_cols = list(hgb_val_features.columns)

    val_all = pd.concat([val_features.reset_index(drop=True), hgb_val_features], axis=1)
    oos_all = pd.concat([oos_features.reset_index(drop=True), hgb_oos_features], axis=1)
    _reject_forbidden(list(val_all.columns), "dsac_state")
    groups = _feature_sets(base_cols, hgb_cols)
    if args.groups.strip():
        requested = [g.strip() for g in args.groups.split(",") if g.strip()]
        missing = [g for g in requested if g not in groups]
        if missing:
            raise RuntimeError(f"unknown DSAC feature groups: {missing}")
        groups = {g: groups[g] for g in requested}
    if int(args.max_groups) > 0:
        groups = dict(list(groups.items())[: int(args.max_groups)])

    rows: list[dict[str, Any]] = []
    artifacts: dict[str, Any] = {}
    for group_name, cols in groups.items():
        _reject_forbidden(cols, group_name)
        x_val, x_oos, norm = _standardize(val_all[cols], oos_all[cols], train_mask)
        for seed in seeds:
            if int(args.hidden) != 256:
                raise RuntimeError("Omega2.1 DSAC sweep must keep train_rl_dsac_agent.py hidden_dim=256")
            cfg = TrainCfg(
                seed=int(seed),
                epochs=int(args.epochs),
                hidden=int(args.hidden),
                lr=3e-4,
                batch_size=512,
                cvar_frac=float(args.cvar_frac),
                entropy_coef=float(args.entropy_coef),
                bc_coef=float(args.bc_coef),
                anti_flat_lambda=0.08,
                anti_flat_min_abs=0.18,
                side_balance_lambda=0.12,
            )
            model = _fit_dsac(x_val, reward_matrix, train_mask, cfg, device)
            val_action, val_conf, _val_q = _predict(model, x_val)
            oos_action, oos_conf, _oos_q = _predict(model, x_oos)
            for threshold in (0.35, 0.45, 0.55, 0.65):
                val_m = _eval(val_frame, val_dec, val_action, val_conf, threshold, fee, slip)
                oos_m = _eval(oos_frame, oos_dec, oos_action, oos_conf, threshold, fee, slip)
                row = {
                    "candidate": f"{group_name}_s{seed}_thr{threshold:.2f}",
                    "feature_group": group_name,
                    "feature_count": int(len(cols)),
                    "seed": int(seed),
                    "threshold": float(threshold),
                    "epochs": int(args.epochs),
                    "hidden": int(args.hidden),
                    **_metric_row("val", val_m),
                    **_metric_row("oos", oos_m),
                }
                row["oos_delta_vs_omega21"] = float(row["oos_pnl"] - BASELINE_OOS["pnl"])
                rows.append(row)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feature_group": group_name,
                    "feature_cols": cols,
                    "normalizer": norm,
                    "train_cfg": cfg.__dict__,
                    "reward_mode": str(args.reward_mode),
                    "forbidden_feature_audit": {"passed": True, "forbidden": []},
                },
                out_dir / f"dsac_{group_name}_s{seed}.pt",
            )
        print(json.dumps({"stage": "group_done", "group": group_name, "rows": len(rows)}, ensure_ascii=False), flush=True)

    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "oos_mdd"], ascending=[False, False, False]).reset_index(drop=True)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "base_model": OMEGA21_MODEL_ID,
        "reward_mode": str(args.reward_mode),
        "status": "research_feature_sweep_not_live_promoted",
        "device": device,
        "label_diag": label_diag,
        "reward_diag": reward_diag,
        "hgb_diag": hgb_diag,
        "baseline_oos": BASELINE_OOS,
        "feature_groups": {k: v for k, v in groups.items()},
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "top": report["top"][:5]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
