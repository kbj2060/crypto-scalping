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


MODEL_ID = "omega1_expertdq_dsac_proposal_overlay_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

OVERLAY_ACTIONS = {
    0: "veto",
    1: "keep",
    2: "reduce_notional_0p75",
    3: "reduce_notional_0p50",
    4: "tight_tp_sl_0p75",
    5: "conservative_0p50_cap2_tight",
}
ACTION_DIM = len(OVERLAY_ACTIONS)


@dataclass
class OverlayDataset:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _apply_overlay(row: pd.Series, action_id: int) -> pd.Series:
    action_id = int(action_id)
    if action_id == 0:
        return _zero_row(row)
    out = row.copy()
    if action_id == 2:
        out.loc["notional_exposure"] = float(out.get("notional_exposure", 0.0) or 0.0) * 0.75
    elif action_id == 3:
        out.loc["notional_exposure"] = float(out.get("notional_exposure", 0.0) or 0.0) * 0.50
    elif action_id == 4:
        out.loc["take_profit"] = float(out.get("take_profit", 0.0) or 0.0) * 0.75
        out.loc["stop_loss"] = abs(float(out.get("stop_loss", 0.0) or 0.0)) * 0.75
    elif action_id == 5:
        out.loc["notional_exposure"] = float(out.get("notional_exposure", 0.0) or 0.0) * 0.50
        out.loc["leverage"] = min(float(out.get("leverage", 1.0) or 1.0), 2.0)
        out.loc["take_profit"] = float(out.get("take_profit", 0.0) or 0.0) * 0.75
        out.loc["stop_loss"] = abs(float(out.get("stop_loss", 0.0) or 0.0)) * 0.75
    elif action_id != 1:
        raise ValueError(f"unknown overlay action: {action_id}")
    leverage = max(float(out.get("leverage", 1.0) or 1.0), 1e-8)
    out.loc["position_fraction"] = float(out.get("notional_exposure", 0.0) or 0.0) / leverage
    return out


def _simulate_row(
    frame: pd.DataFrame,
    arrays: dict[str, np.ndarray],
    i: int,
    row: pd.Series,
    action_id: int,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[float, dict[str, Any]]:
    dec = _apply_overlay(row, action_id)
    action = int(dec.get("action", 0) or 0)
    side = int(dec.get("side", 0) or 0)
    notional = float(dec.get("notional_exposure", 0.0) or 0.0)
    if action == ACTION_CASH or side == 0 or notional <= 0.0:
        return 0.0, {"active": 0, "exit_i": int(i), "win": 0}
    entry_i = min(int(i) + 1, len(frame) - 1)
    entry_px = float(arrays["open"][entry_i])
    if entry_px <= 0.0:
        return 0.0, {"active": 0, "exit_i": int(i), "win": 0}
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
            trigger_px = entry * max(1.0 - sl, 1e-8) if side > 0 else entry / max(1.0 - sl, 1e-8)
            exit_fill = trigger_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            exit_reason = "stop_loss"
            end_i = j
            break
        if favorable >= tp:
            trigger_px = entry * (1.0 + tp) if side > 0 else entry / max(1.0 + tp, 1e-8)
            exit_fill = trigger_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
            exit_reason = "take_profit"
            end_i = j
            break
    if exit_fill is None:
        exit_px = float(arrays["close"][end_i])
        exit_fill = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
    qty = notional / max(entry, 1e-12)
    exit_notional = qty * max(float(exit_fill), 0.0)
    gross = exit_notional - notional if side > 0 else notional - exit_notional
    net = float(gross - fee_eff * notional - fee_eff * exit_notional)
    return net, {"active": 1, "exit_i": int(end_i), "win": int(net > 0.0), "exit_reason": exit_reason}


def _fast_replay_metrics(frame: pd.DataFrame, dec: pd.DataFrame, overlays: np.ndarray, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    arrays = {k: _num(frame, k) for k in ("open", "high", "low", "close")}
    active = _active(dec)
    next_allowed = 0
    equity = 0.0
    peak = 0.0
    mdd = 0.0
    wins = 0
    trades = 0
    usage: dict[int, int] = {}
    for i in range(len(frame) - 3):
        if i < next_allowed or not bool(active[i]):
            continue
        action_id = int(overlays[i])
        usage[action_id] = usage.get(action_id, 0) + 1
        reward, meta = _simulate_row(frame, arrays, i, dec.iloc[i], action_id, fee=fee, slip=slip, cost_mult=cost_mult)
        if int(meta.get("active", 0)) != 1:
            next_allowed = i + 1
            continue
        trades += 1
        wins += int(reward > 0.0)
        equity += float(reward) * 100.0
        peak = max(peak, equity)
        mdd = min(mdd, equity - peak)
        cooldown = max(int(dec.iloc[i].get("cooldown_bars", 0) or 0), 0)
        next_allowed = max(i + 1, int(meta.get("exit_i", i)) + cooldown)
    return {
        "pnl": float(equity),
        "mdd": float(mdd),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "usage": {OVERLAY_ACTIONS[k]: int(v) for k, v in sorted(usage.items())},
    }


def _build_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_active_rows: int,
) -> tuple[OverlayDataset, dict[str, Any]]:
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
    best_counts: dict[int, int] = {}
    net_sum: dict[int, list[float]] = {}
    for i in active_idxs:
        best_a = 0
        best_r = -1e18
        for action_id in range(ACTION_DIM):
            reward, meta = _simulate_row(frame, arrays, int(i), dec.iloc[int(i)], action_id, fee=fee, slip=slip, cost_mult=cost_mult)
            s_list.append(states[int(i)])
            sp_list.append(states[min(int(i) + 1, len(states) - 1)])
            a_list.append(action_id)
            r_list.append(float(reward))
            d_list.append(1.0)
            if int(meta.get("active", 0)) == 1:
                net_sum.setdefault(action_id, []).append(float(reward))
            if float(reward) > best_r:
                best_r = float(reward)
                best_a = int(action_id)
        best_counts[best_a] = best_counts.get(best_a, 0) + 1
    rewards = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards = np.clip(rewards / scale, -8.0, 8.0).astype(np.float32)
    return (
        OverlayDataset(
            states=np.asarray(s_list, dtype=np.float32),
            next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64),
            rewards=rewards,
            dones=np.asarray(d_list, dtype=np.float32),
        ),
        {
            "active_rows": int(len(active_idxs)),
            "total_active_rows": total_active_rows,
            "sample_count": int(len(rewards)),
            "reward_scale": float(scale),
            "oracle_best_counts": {OVERLAY_ACTIONS[k]: int(v) for k, v in sorted(best_counts.items())},
            "mean_net_by_action": {OVERLAY_ACTIONS[k]: float(np.mean(v)) for k, v in sorted(net_sum.items()) if v},
        },
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, ACTION_DIM),
        )

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def sample(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dist = torch.distributions.Categorical(logits=self.logits(x))
        action = dist.sample()
        return action, dist.log_prob(action)

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.logits(x), dim=-1)


class Critic(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(0.04),
            nn.Linear(hidden, 192),
            nn.SiLU(),
            nn.Linear(192, ACTION_DIM),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dsac(
    data: OverlayDataset,
    *,
    state_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    target_entropy: float,
) -> tuple[Actor, dict[str, Any]]:
    actor = Actor(state_dim).to(device)
    q1 = Critic(state_dim).to(device)
    q2 = Critic(state_dim).to(device)
    log_alpha = torch.tensor(math.log(0.10), device=device, requires_grad=True)
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
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            s, _sp, a, r, _d = next(it)
        except StopIteration:
            it = iter(dl)
            s, _sp, a, r, _d = next(it)
        s = s.to(device)
        a = a.to(device)
        r = r.to(device)
        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.smooth_l1_loss(qa1, r) + F.smooth_l1_loss(qa2, r)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()

        pa, plogp = actor.sample(s)
        pq = torch.min(q1(s), q2(s)).gather(1, pa.view(-1, 1)).squeeze(1)
        actor_loss = (log_alpha.exp() * plogp - pq).mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = (-plogp).mean().detach()
        alpha_loss = (log_alpha * (entropy - float(target_entropy))).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(2.0))
        if step % 250 == 0:
            last = {
                "step": int(step),
                "q_loss": float(q_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "alpha": float(log_alpha.exp().detach().cpu()),
                "entropy": float(entropy.cpu()),
                "target_entropy": float(target_entropy),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "dsac_progress", **last}, ensure_ascii=False), flush=True)
    return actor.cpu(), last


def _policy_actions(actor: Actor, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            out.append(actor.greedy(x).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _score(row: pd.Series) -> float:
    trades = int(row.get("trades", 0) or 0)
    if trades < 30:
        return -1e9 + float(row.get("pnl", 0.0) or 0.0)
    return float(row.get("pnl", 0.0) + 130.0 * row.get("wr", 0.0) - 0.45 * abs(row.get("mdd", 0.0)) + 0.015 * trades)


def _metrics_row(split: str, variant: str, frame: pd.DataFrame, dec: pd.DataFrame, overlays: np.ndarray, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    metrics = _fast_replay_metrics(frame, dec, overlays, fee=fee, slip=slip, cost_mult=cost_mult)
    usage = metrics.pop("usage")
    row = {"split": split, "variant": variant, "cost": 3, **metrics, "usage_json": json.dumps(usage, ensure_ascii=False)}
    row["selection_score"] = _score(pd.Series(row))
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="soft_floor_0p00")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2.0e-4)
    ap.add_argument("--target-entropy", type=float, default=1.2)
    ap.add_argument("--max-active-rows", type=int, default=12000)
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
        target_entropy=float(args.target_entropy),
    )
    a_train = _policy_actions(actor, x_train, device=device)
    a_val = _policy_actions(actor, x_val, device=device)
    a_oos = _policy_actions(actor, x_oos, device=device)
    keep_val = np.ones(len(val_df), dtype=np.int64)
    keep_oos = np.ones(len(oos_df), dtype=np.int64)
    rows = [
        _metrics_row("val", "fixed_keep_proposal", val_df, val_dec, keep_val, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "fixed_keep_proposal", oos_df, oos_dec, keep_oos, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("val", "dsac_proposal_overlay", val_df, val_dec, a_val, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        _metrics_row("oos", "dsac_proposal_overlay", oos_df, oos_dec, a_oos, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    ]
    grid = pd.DataFrame(rows)
    grid_path = out_dir / "grid.csv"
    grid.to_csv(grid_path, index=False)

    model_path = out_dir / "omega1_expertdq_dsac_proposal_overlay.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "variant": str(args.variant),
            "state_dim": int(x_train.shape[1]),
            "action_dim": int(ACTION_DIM),
            "overlay_actions": OVERLAY_ACTIONS,
            "state_columns": list(norm["columns"]),
            "state_normalizer": norm,
            "actor_state_dict": actor.state_dict(),
        },
        model_path,
    )
    fixed_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "fixed_keep_proposal")].iloc[0].to_dict()
    dsac_oos = grid[(grid["split"] == "oos") & (grid["variant"] == "dsac_proposal_overlay")].iloc[0].to_dict()
    summary = {
        "model_id": MODEL_ID,
        "variant": str(args.variant),
        "design": "Omega1 supervised expert-local decision/quality/risk proposal is frozen. DSAC can only veto, keep, or conservatively reduce/tighten the proposal.",
        "selection_basis": "2025Q4 validation fast replay diagnostic; 2026 OOS report-only.",
        "selection_uses_2026": False,
        "legacy_compat_alias": False,
        "risk_template": ACTIVE_TEMPLATE,
        "expert_scales": ACTIVE_SCALES,
        "overlay_actions": OVERLAY_ACTIONS,
        "training": {
            "device": str(device),
            "state_dim": int(x_train.shape[1]),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "target_entropy": float(args.target_entropy),
            "cost_mult": float(args.cost_mult),
            "reward_label": "complete_trade_net_pnl_after_entry_exit_fee_slippage for each conservative proposal overlay",
            "data_diag": data_diag,
            "train_diag": train_diag,
            "action_usage": {
                "train": {OVERLAY_ACTIONS[k]: int(v) for k, v in sorted(zip(*np.unique(a_train, return_counts=True)))},
                "val": {OVERLAY_ACTIONS[k]: int(v) for k, v in sorted(zip(*np.unique(a_val, return_counts=True)))},
                "oos": {OVERLAY_ACTIONS[k]: int(v) for k, v in sorted(zip(*np.unique(a_oos, return_counts=True)))},
            },
        },
        "fast_replay": {
            "fixed_oos_cost3": fixed_oos,
            "dsac_oos_cost3": dsac_oos,
            "delta_pnl": float(dsac_oos["pnl"]) - float(fixed_oos["pnl"]),
        },
        "overlay": overlay,
        "artifacts": {"summary": str(out_dir / "summary.json"), "grid": str(grid_path), "model": str(model_path)},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(out_dir / "summary.json"), "fixed_oos_cost3": fixed_oos, "dsac_oos_cost3": dsac_oos, "delta_pnl": summary["fast_replay"]["delta_pnl"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
