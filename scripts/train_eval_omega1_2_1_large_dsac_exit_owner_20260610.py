#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_1_large_dsac_exit_owner_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
POS_FEATURE_COLUMNS = [
    "pos_side",
    "pos_notional",
    "pos_margin_notional",
    "pos_leverage",
    "pos_unrealized",
    "pos_mfe",
    "pos_mae",
    "pos_giveback",
    "pos_hold_bars",
    "pos_dist_tp",
    "pos_dist_sl",
    "pos_tp_progress",
    "pos_sl_progress",
    "pos_floor_unreal",
    "pos_reduced",
    "pos_tightened",
]


class CompactFeatureExtractor(nn.Module):
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


class DiscreteExitActor(nn.Module):
    """Large DSAC-style categorical actor for exit-only actions.

    The original DSAC actor is tanh-Gaussian for continuous position actions.
    Exit editing is naturally discrete, so this keeps the same large state
    encoder and entropy/CVaR policy objective but exposes action logits.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 256, n_actions: int = 4) -> None:
        super().__init__()
        self.feat = CompactFeatureExtractor(state_dim, hidden_dim)
        self.gate_head = nn.Linear(hidden_dim, 4)
        self.action_heads = nn.ModuleList([nn.Linear(hidden_dim, n_actions) for _ in range(4)])

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.feat(state)
        gate = torch.softmax(self.gate_head(feat), dim=-1)
        logits_stack = torch.stack([h(feat) for h in self.action_heads], dim=1)
        logits = (gate.unsqueeze(-1) * logits_stack).sum(dim=1)
        return logits, gate


class DiscreteDistributionalTwinCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_actions: int = 4, n_quantiles: int = 32) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.feat1 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_actions * self.n_quantiles),
        )
        self.feat2 = CompactFeatureExtractor(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_actions * self.n_quantiles),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.q1(self.feat1(state)).view(-1, self.n_actions, self.n_quantiles)
        q2 = self.q2(self.feat2(state)).view(-1, self.n_actions, self.n_quantiles)
        return q1, q2


class LargeDSACExitOwner(nn.Module):
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 256,
        n_actions: int = 4,
        n_quantiles: int = 32,
        cvar_frac: float = 0.40,
    ) -> None:
        super().__init__()
        self.actor = DiscreteExitActor(state_dim, hidden_dim, n_actions)
        self.critic = DiscreteDistributionalTwinCritic(state_dim, hidden_dim, n_actions, n_quantiles)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)
        self.cvar_frac = float(cvar_frac)

    def critic_cvar(self, state: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.critic(state)
        k = max(1, int(self.n_quantiles * self.cvar_frac))
        q1s, _ = torch.sort(q1, dim=-1)
        q2s, _ = torch.sort(q2, dim=-1)
        c1 = q1s[:, :, :k].mean(dim=-1)
        c2 = q2s[:, :, :k].mean(dim=-1)
        return torch.minimum(c1, c2)

    def q_mean(self, state: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.critic(state)
        return 0.5 * (q1.mean(dim=-1) + q2.mean(dim=-1))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _fit_norm(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return {"columns": list(x.columns), "mean": mean, "std": std}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"missing normalized columns: {missing[:20]}")
    arr = x.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num((arr - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _apply_norm_fast_one(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    # _pos_features already preserves the training column order. Keep a strict
    # equality check so feature-contract drift fails immediately instead of
    # silently reordering or filling columns.
    cols = list(norm["columns"])
    if list(x.columns) != cols:
        missing = [c for c in cols if c not in x.columns]
        extra = [c for c in x.columns if c not in cols]
        raise RuntimeError(f"exit DSAC state contract mismatch; missing={missing[:10]} extra={extra[:10]}")
    arr = x.to_numpy(dtype=np.float32, copy=False)
    return np.nan_to_num((arr - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _quantile_regression_loss(pred: torch.Tensor, target: torch.Tensor, taus: torch.Tensor) -> torch.Tensor:
    # pred: [B,A,N], target: [B,A]
    td = target.unsqueeze(-1) - pred
    abs_td = td.abs()
    huber = torch.where(abs_td <= 1.0, 0.5 * td.pow(2), abs_td - 0.5)
    tau = taus.view(1, 1, -1)
    weight = (tau - (td.detach() < 0).float()).abs()
    return (weight * huber).mean()


def _allowed_mask_from_states(x: pd.DataFrame) -> np.ndarray:
    hold = pd.to_numeric(x["pos_hold_bars"], errors="raise").to_numpy(dtype=np.float64)
    unreal = pd.to_numeric(x["pos_unrealized"], errors="raise").to_numpy(dtype=np.float64)
    mfe = pd.to_numeric(x["pos_mfe"], errors="raise").to_numpy(dtype=np.float64)
    giveback = pd.to_numeric(x["pos_giveback"], errors="raise").to_numpy(dtype=np.float64)
    reduced = pd.to_numeric(x["pos_reduced"], errors="raise").to_numpy(dtype=np.float64)
    tightened = pd.to_numeric(x["pos_tightened"], errors="raise").to_numpy(dtype=np.float64)
    mask = np.zeros((len(x), len(base.ACTION_NAMES)), dtype=bool)
    mask[:, base.HOLD] = True
    mask[:, base.TIGHTEN_SL] = (hold >= 2) & (mfe >= 0.025) & (tightened < 0.5)
    mask[:, base.REDUCE50] = (hold >= 2) & (unreal >= 0.035) & (reduced < 0.5)
    mask[:, base.FULL_EXIT] = (hold >= 2) & (((mfe >= 0.04) & (giveback >= 0.65)) | (unreal <= -0.045))
    return mask


def _pos_feature_values(pos: base.Position, unreal: float, i: int) -> np.ndarray:
    mfe = max(pos.mfe, unreal)
    mae = min(pos.mae, unreal)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    vals = [
        float(pos.side),
        float(pos.notional),
        float(pos.margin_notional),
        float(pos.leverage),
        float(unreal),
        float(mfe),
        float(mae),
        float(np.clip(giveback, 0.0, 10.0)),
        float(max(int(i) - int(pos.entry_i), 0)),
        float(pos.take_profit - unreal),
        float(unreal + abs(pos.stop_loss)),
        float(unreal / max(pos.take_profit, 1e-8)),
        float(-unreal / max(abs(pos.stop_loss), 1e-8)) if pos.stop_loss > 0 else 0.0,
        float(pos.floor_unreal),
        float(pos.reduced),
        float(pos.tightened),
    ]
    return np.asarray(vals, dtype=np.float32)


def _allowed_mask_from_pos(pos: base.Position, unreal: float, i: int) -> np.ndarray:
    vals = _pos_feature_values(pos, unreal, i)
    hold = float(vals[8])
    mfe = float(vals[5])
    giveback = float(vals[7])
    reduced = float(vals[14])
    tightened = float(vals[15])
    mask = np.zeros((len(base.ACTION_NAMES),), dtype=bool)
    mask[base.HOLD] = True
    mask[base.TIGHTEN_SL] = (hold >= 2) and (mfe >= 0.025) and (tightened < 0.5)
    mask[base.REDUCE50] = (hold >= 2) and (unreal >= 0.035) and (reduced < 0.5)
    mask[base.FULL_EXIT] = (hold >= 2) and (((mfe >= 0.04) and (giveback >= 0.65)) or (unreal <= -0.045))
    return mask


def _train_large_dsac(
    x: pd.DataFrame,
    rewards: np.ndarray,
    *,
    epochs: int,
    seed: int,
    cvar_frac: float,
    entropy_coef: float,
    cql_coef: float,
    actor_coef: float,
    bc_coef: float,
) -> tuple[LargeDSACExitOwner, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = _fit_norm(x)
    xt = torch.from_numpy(_apply_norm(x, norm)).to(device)
    rt = torch.from_numpy(rewards.astype(np.float32)).to(device)
    allowed = torch.from_numpy(_allowed_mask_from_states(x)).to(device)
    model = LargeDSACExitOwner(
        state_dim=xt.shape[1],
        hidden_dim=256,
        n_actions=rt.shape[1],
        n_quantiles=32,
        cvar_frac=float(cvar_frac),
    ).to(device)
    opt_actor = torch.optim.AdamW(model.actor.parameters(), lr=3e-4, weight_decay=2e-4)
    opt_critic = torch.optim.AdamW(model.critic.parameters(), lr=3e-4, weight_decay=2e-4)
    taus = torch.linspace(0.5 / 32, 1.0 - 0.5 / 32, 32, device=device, dtype=torch.float32)
    batch = min(512, len(xt))
    losses: list[dict[str, float]] = []
    for epoch in range(int(epochs)):
        perm = torch.randperm(len(xt), device=device)
        c_total = 0.0
        a_total = 0.0
        seen = 0
        for start in range(0, len(xt), batch):
            idx = perm[start : start + batch]
            xb = xt[idx]
            rb = rt[idx]
            mb = allowed[idx]
            q1, q2 = model.critic(xb)
            critic_loss = _quantile_regression_loss(q1, rb, taus) + _quantile_regression_loss(q2, rb, taus)
            if float(cql_coef) > 0.0:
                q_mean = 0.5 * (q1.mean(dim=-1) + q2.mean(dim=-1))
                best = torch.argmax(rb.masked_fill(~mb, -1e9), dim=1)
                cql = (torch.logsumexp(q_mean.masked_fill(~mb, -1e9), dim=1) - q_mean.gather(1, best[:, None]).squeeze(1)).mean()
                critic_loss = critic_loss + float(cql_coef) * cql
            opt_critic.zero_grad(set_to_none=True)
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.critic.parameters(), 1.0)
            opt_critic.step()

            logits, _gate = model.actor(xb)
            logits = logits.masked_fill(~mb, -1e9)
            probs = torch.softmax(logits, dim=1)
            log_probs = torch.log_softmax(logits, dim=1)
            q_cvar = model.critic_cvar(xb).detach().masked_fill(~mb, -1e9)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            actor_loss = -(probs * q_cvar).sum(dim=1).mean() - float(entropy_coef) * entropy
            if float(bc_coef) > 0.0:
                best = torch.argmax(rb.masked_fill(~mb, -1e9), dim=1)
                actor_loss = actor_loss + float(bc_coef) * F.cross_entropy(logits, best)
            actor_loss = float(actor_coef) * actor_loss
            opt_actor.zero_grad(set_to_none=True)
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 1.0)
            opt_actor.step()
            c_total += float(critic_loss.detach().cpu()) * len(idx)
            a_total += float(actor_loss.detach().cpu()) * len(idx)
            seen += len(idx)
        if epoch < 3 or epoch >= int(epochs) - 3 or (epoch + 1) % 100 == 0:
            losses.append({"epoch": int(epoch + 1), "critic_loss": c_total / max(seen, 1), "actor_loss": a_total / max(seen, 1)})
    model.eval()
    model.norm = norm  # type: ignore[attr-defined]
    diag = {
        "device": str(device),
        "epochs": int(epochs),
        "hidden_dim": 256,
        "n_quantiles": 32,
        "cvar_frac": float(cvar_frac),
        "entropy_coef": float(entropy_coef),
        "cql_coef": float(cql_coef),
        "actor_coef": float(actor_coef),
        "bc_coef": float(bc_coef),
        "loss_trace": losses,
    }
    return model, diag


@torch.no_grad()
def _dsac_action(model: LargeDSACExitOwner, x: pd.DataFrame, *, min_adv: float, allowed_full_exit: bool) -> int:
    device = next(model.parameters()).device
    arr = torch.from_numpy(_apply_norm_fast_one(x, model.norm)).to(device)  # type: ignore[attr-defined]
    mask = torch.from_numpy(_allowed_mask_from_states(x)).to(device)
    if not bool(allowed_full_exit):
        mask[:, base.FULL_EXIT] = False
    logits, _gate = model.actor(arr)
    q_cvar = model.critic_cvar(arr)
    # Value-guided policy score keeps the actor from blindly imitating noisy top actions.
    score = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1) + 0.75 * q_cvar.masked_fill(~mask, -1e9)
    q = score[0].detach().cpu().numpy().astype(np.float64)
    best = int(np.argmax(q))
    if best == base.HOLD:
        return base.HOLD
    q_hold = float(q[base.HOLD])
    if float(q[best] - q_hold) < float(min_adv):
        return base.HOLD
    return best


@torch.no_grad()
def _dsac_action_fast(
    model: LargeDSACExitOwner,
    *,
    state_values: np.ndarray,
    pos: base.Position,
    unreal: float,
    i: int,
    min_adv: float,
    allowed_full_exit: bool,
) -> int:
    norm = model.norm  # type: ignore[attr-defined]
    row = np.concatenate([state_values.astype(np.float32, copy=False), _pos_feature_values(pos, unreal, i)])
    if row.shape[0] != len(norm["columns"]):
        raise RuntimeError(f"exit DSAC state length mismatch: got={row.shape[0]} expected={len(norm['columns'])}")
    arr = np.nan_to_num((row[None, :] - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    device = next(model.parameters()).device
    tensor = torch.from_numpy(arr).to(device)
    mask_np = _allowed_mask_from_pos(pos, unreal, i)[None, :]
    if not bool(allowed_full_exit):
        mask_np[:, base.FULL_EXIT] = False
    mask = torch.from_numpy(mask_np).to(device)
    logits, _gate = model.actor(tensor)
    q_cvar = model.critic_cvar(tensor)
    score = torch.log_softmax(logits.masked_fill(~mask, -1e9), dim=1) + 0.75 * q_cvar.masked_fill(~mask, -1e9)
    q = score[0].detach().cpu().numpy().astype(np.float64)
    cvar = q_cvar[0].detach().cpu().numpy().astype(np.float64)
    best = int(np.argmax(q))
    if best == base.HOLD:
        return base.HOLD
    # Promotion gate must be based on critic value, not actor confidence.
    # Otherwise a confident actor can trigger exits even when the risk critic
    # does not estimate a real advantage over holding.
    if float(cvar[best] - cvar[base.HOLD]) < float(min_adv):
        return base.HOLD
    return best


def _simulate_policy_large(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    model: LargeDSACExitOwner | None,
    min_adv: float,
    fee: float,
    slip: float,
    cost_mult: float,
    allowed_full_exit: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = base._arrays(frame)
    state_values = state.to_numpy(dtype=np.float32, copy=False)
    expected_cols = list(state.columns) + POS_FEATURE_COLUMNS
    if model is not None and list(model.norm["columns"]) != expected_cols:  # type: ignore[attr-defined]
        missing = [c for c in model.norm["columns"] if c not in expected_cols]  # type: ignore[attr-defined]
        extra = [c for c in expected_cols if c not in model.norm["columns"]]  # type: ignore[attr-defined]
        raise RuntimeError(f"exit DSAC feature contract mismatch; missing={missing[:10]} extra={extra[:10]}")
    active = np.asarray(base.omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = base.Position()
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    action_counts = {name: 0 for name in base.ACTION_NAMES}
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = base._unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = base._hit_reason(unreal, pos)
            if not reason and model is not None:
                action = _dsac_action_fast(
                    model,
                    state_values=state_values[int(i)],
                    pos=pos,
                    unreal=unreal,
                    i=int(i),
                    min_adv=min_adv,
                    allowed_full_exit=allowed_full_exit,
                )
                before_pos = base.Position(**pos.__dict__)
                cash, pos, action_name = base._apply_action(cash, arrays, pos, i, action, unreal, fee_eff, slip_eff)
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
                if before_pos.side != 0 and pos.side == 0:
                    reason = "large_dsac_full_exit"
                    net_pct = float((cash / max(before_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                    trades.append(net_pct)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    rows.append(_ledger_row(frame, arrays, before_pos, i, cash, net_pct, reason))
                    continue
            if reason:
                close_pos = base.Position(**pos.__dict__)
                cash, pos, _ = base._close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(_ledger_row(frame, arrays, close_pos, i, cash, net_pct, reason))
            continue

        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if not bool(active[i]):
            continue
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = base._enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)

    if pos.side != 0:
        close_pos = base.Position(**pos.__dict__)
        cash, pos, _ = base._close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append(_ledger_row(frame, arrays, close_pos, len(frame) - 1, cash, net_pct, "forced_end"))
    arr = np.asarray(trades, dtype=np.float64)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "adapter_actions": action_counts,
    }
    return metrics, pd.DataFrame(rows)


def _ledger_row(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    pos: base.Position,
    exit_i: int,
    cash: float,
    net_pct: float,
    reason: str,
) -> dict[str, Any]:
    return {
        "trade_id": -1,
        "side": "LONG" if pos.side > 0 else "SHORT",
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(exit_i),
        "entry_time": str(frame["timestamp"].iloc[int(pos.entry_signal_i)]),
        "exit_time": str(frame["timestamp"].iloc[int(exit_i)]),
        "entry_price": float(pos.entry_price),
        "exit_price": float(arrays["close"][int(exit_i)]),
        "effective_exposure": float(pos.notional),
        "margin_notional": float(pos.margin_notional),
        "leverage": float(pos.leverage),
        "tp_equity_ret": float(pos.take_profit),
        "sl_equity_ret": float(pos.stop_loss),
        "net_trade_return_pct": float(net_pct),
        "mfe_pct": float(pos.mfe * 100.0),
        "mae_pct": float(pos.mae * 100.0),
        "exit_reason": str(reason),
        "cash_after": float(cash),
    }


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_adapter_actions": metrics.get("adapter_actions", {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1200)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-states", type=int, default=3600)
    ap.add_argument("--max-forward-bars", type=int, default=288)
    ap.add_argument("--seed", type=int, default=260610)
    ap.add_argument("--generators", default="high,low")
    ap.add_argument("--min-advs", default="0,0.001,0.0025,0.005,0.01,0.02")
    ap.add_argument("--full-exit-modes", default="0,1")
    ap.add_argument("--cvar-frac", type=float, default=0.40)
    ap.add_argument("--entropy-coef", type=float, default=0.02)
    ap.add_argument("--cql-coef", type=float, default=0.04)
    ap.add_argument("--actor-coef", type=float, default=1.0)
    ap.add_argument("--bc-coef", type=float, default=0.0)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = base.omega._load_fee_slip()
    splits = base._build_splits()
    built: dict[str, dict[str, Any]] = {}
    for threshold_name, thresholds in (("high", base.HIGH_THRESHOLDS), ("low", base.LOW_THRESHOLDS)):
        built[threshold_name] = {}
        for split, payload in splits.items():
            dec = base._to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=thresholds)
            state = base._state_base(payload["frame"], payload["src"], dec, payload["prefix"])
            built[threshold_name][split] = {"frame": payload["frame"], "dec": dec, "state": state}

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    requested = tuple(x.strip() for x in str(args.generators).split(",") if x.strip())
    unknown = [x for x in requested if x not in {"high", "low"}]
    if unknown:
        raise RuntimeError(f"unknown generators: {unknown}")
    for threshold_name in requested:
        val = built[threshold_name]["validation"]
        oos = built[threshold_name]["oos"]
        x_train, rewards, data_diag = base._collect_dataset(
            val["frame"],
            val["dec"],
            val["state"],
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            stride=int(args.stride),
            max_states=int(args.max_states),
            max_forward_bars=int(args.max_forward_bars),
        )
        model, train_diag = _train_large_dsac(
            x_train,
            rewards,
            epochs=int(args.epochs),
            seed=int(args.seed),
            cvar_frac=float(args.cvar_frac),
            entropy_coef=float(args.entropy_coef),
            cql_coef=float(args.cql_coef),
            actor_coef=float(args.actor_coef),
            bc_coef=float(args.bc_coef),
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "norm": model.norm,  # type: ignore[attr-defined]
                "actions": base.ACTION_NAMES,
                "train_diag": train_diag,
                "model_id": MODEL_ID,
            },
            OUT_DIR / f"{threshold_name}_large_dsac_exit_owner.pt",
        )
        # Backtest calls the exit policy one bar at a time. CPU inference is
        # materially faster than thousands of tiny CUDA launches here.
        model.to(torch.device("cpu"))
        x_train.to_csv(OUT_DIR / f"{threshold_name}_train_states.csv", index=False)
        pd.DataFrame(rewards, columns=[f"reward_{a}" for a in base.ACTION_NAMES]).to_csv(OUT_DIR / f"{threshold_name}_train_rewards.csv", index=False)

        val_base, val_base_ledger = _simulate_policy_large(val["frame"], val["dec"], val["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        oos_base, oos_base_ledger = _simulate_policy_large(oos["frame"], oos["dec"], oos["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        val_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_validation_baseline_ledger.csv", index=False)
        oos_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_oos_baseline_ledger.csv", index=False)
        rows.append({"candidate_generator": threshold_name, "policy": "baseline_no_dsac_exit", "min_adv": None, **_row("val", val_base), **_row("oos", oos_base)})

        for allowed_full_exit in tuple(bool(int(x.strip())) for x in str(args.full_exit_modes).split(",") if x.strip()):
            for min_adv in tuple(float(x.strip()) for x in str(args.min_advs).split(",") if x.strip()):
                val_m, val_ledger = _simulate_policy_large(
                    val["frame"],
                    val["dec"],
                    val["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                oos_m, oos_ledger = _simulate_policy_large(
                    oos["frame"],
                    oos["dec"],
                    oos["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                row = {
                    "candidate_generator": threshold_name,
                    "policy": "large_dsac_exit_full_exit" if allowed_full_exit else "large_dsac_exit_defensive",
                    "min_adv": float(min_adv),
                    **_row("val", val_m),
                    **_row("oos", oos_m),
                }
                rows.append(row)
                tag = f"{threshold_name}_{'full' if allowed_full_exit else 'def'}_adv{str(min_adv).replace('.', 'p')}"
                val_ledger.to_csv(OUT_DIR / f"validation_{tag}_ledger.csv", index=False)
                oos_ledger.to_csv(OUT_DIR / f"oos_{tag}_ledger.csv", index=False)
        reports[threshold_name] = {
            "dataset": data_diag,
            "training": train_diag,
            "thresholds": base.HIGH_THRESHOLDS if threshold_name == "high" else base.LOW_THRESHOLDS,
        }

    ranking = pd.DataFrame(rows)
    high_base = ranking[(ranking["candidate_generator"] == "high") & (ranking["policy"] == "baseline_no_dsac_exit")].iloc[0]
    ranking["delta_vs_high_base_oos_pnl"] = ranking["oos_pnl"] - float(high_base["oos_pnl"])
    ranking["delta_vs_high_base_val_pnl"] = ranking["val_pnl"] - float(high_base["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "large_dsac_exit_owner_ranking.csv", index=False)
    promotable = ranking[
        (ranking["policy"] != "baseline_no_dsac_exit")
        & (ranking["oos_pnl"] > float(high_base["oos_pnl"]))
        & (ranking["val_pnl"] > float(high_base["val_pnl"]) * 0.85)
        & (ranking["oos_mdd"] >= float(high_base["oos_mdd"]) * 1.25)
    ].copy()
    promotable.to_csv(OUT_DIR / "large_dsac_exit_owner_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_high_threshold": high_base.to_dict(),
        "threshold_reports": reports,
        "architecture": {
            "state_encoder": "CompactFeatureExtractor Linear/LayerNorm/SiLU x2, hidden_dim=256",
            "actor": "4-head gated categorical exit actor, hidden_dim=256",
            "critic": "twin distributional critic, n_quantiles=32",
            "objective": "critic quantile regression on counterfactual exit rewards + CVaR actor objective",
            "actions": base.ACTION_NAMES,
            "entry_owner": "frozen omega1_2_1_true_leverage_price_barrier_scale200_cap090",
        },
        "promotable_count": int(len(promotable)),
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "large_dsac_exit_owner_ranking.csv"),
            "promotable": str(OUT_DIR / "large_dsac_exit_owner_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top5": ranking.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
