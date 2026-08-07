#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_mamba_sac_lifecycle_controller_20260604"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

ACTION_NAMES = ["hold_or_skip", "enter_base", "enter_aggressive", "reduce50", "full_exit", "resize_up", "reverse"]
HOLD_OR_SKIP = 0
ENTER_BASE = 1
ENTER_AGGRESSIVE = 2
REDUCE50 = 3
FULL_EXIT = 4
RESIZE_UP = 5
REVERSE = 6

POS_COLS = [
    "lc_pos_side",
    "lc_pos_notional",
    "lc_pos_entry_raw",
    "lc_pos_unrealized",
    "lc_pos_mfe",
    "lc_pos_mae",
    "lc_pos_giveback",
    "lc_pos_hold_bars",
    "lc_pos_dist_tp",
    "lc_pos_dist_sl",
]


@dataclass
class OfflineData:
    seq: np.ndarray
    q_targets: np.ndarray
    best_actions: np.ndarray
    weights: np.ndarray


@dataclass
class Position:
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    notional: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _fit_norm(df: pd.DataFrame) -> dict[str, Any]:
    arr = df.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": list(df.columns), "median": med.astype(np.float32), "scale": scale.astype(np.float32)}


def _apply_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"lifecycle state missing columns: {missing[:20]}")
    arr = df[cols].to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    return np.tanh(np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _rolling_sequences(arr: np.ndarray, seq_len: int) -> np.ndarray:
    pad = np.repeat(arr[:1], max(int(seq_len) - 1, 0), axis=0)
    padded = np.concatenate([pad, arr], axis=0)
    view = np.lib.stride_tricks.sliding_window_view(padded, int(seq_len), axis=0)
    return np.swapaxes(view, 1, 2).copy().astype(np.float32)


def _base_state(state: pd.DataFrame) -> pd.DataFrame:
    out = state.copy().reset_index(drop=True)
    for col in POS_COLS:
        out[col] = 0.0
    return out


def _position_values(arrays: dict[str, np.ndarray], pos: Position, i: int, *, slip_eff: float) -> dict[str, float]:
    if pos.side == 0 or pos.notional <= 0.0:
        return {c: 0.0 for c in POS_COLS}
    px = float(arrays["close"][int(i)])
    raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
    unreal = raw * pos.notional
    mfe = max(float(pos.mfe), unreal)
    mae = min(float(pos.mae), unreal)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0 else 0.0
    return {
        "lc_pos_side": float(pos.side),
        "lc_pos_notional": float(pos.notional),
        "lc_pos_entry_raw": float((px - pos.entry_price) / max(pos.entry_price, 1e-12)),
        "lc_pos_unrealized": float(unreal),
        "lc_pos_mfe": float(mfe),
        "lc_pos_mae": float(mae),
        "lc_pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "lc_pos_hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "lc_pos_dist_tp": float(pos.take_profit - unreal),
        "lc_pos_dist_sl": float(unreal + abs(pos.stop_loss)),
    }


def _state_row(base_state: pd.DataFrame, arrays: dict[str, np.ndarray], pos: Position, i: int, *, slip_eff: float) -> pd.DataFrame:
    row = base_state.iloc[[int(i)]].copy().reset_index(drop=True)
    vals = _position_values(arrays, pos, int(i), slip_eff=slip_eff)
    for k, v in vals.items():
        row[k] = v
    return row


def _seq_for_state(base_seq_raw: np.ndarray, row_norm: np.ndarray, i: int) -> np.ndarray:
    seq = base_seq_raw[int(i)].copy()
    seq[-1, :] = row_norm[0]
    return seq


def _fill_exit(arrays: dict[str, np.ndarray], i: int, side: int, slip_eff: float) -> float:
    return omega._fill_price(arrays, int(i), int(side), float(slip_eff), entry=False)


def _realize_fraction(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, frac: float, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, float]:
    if pos.side == 0 or pos.notional <= 0.0 or frac <= 0.0:
        return cash, pos, 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    exit_px = _fill_exit(arrays, int(i), pos.side, slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1e-12)
    reduce_notional = pos.notional * frac
    before = cash
    cash = cash * (1.0 + raw * reduce_notional)
    cash -= before * float(fee_eff) * reduce_notional
    new_pos = Position(**pos.__dict__)
    new_pos.notional = max(0.0, pos.notional - reduce_notional)
    if new_pos.notional <= 1e-9:
        new_pos = Position()
    return cash, new_pos, raw * reduce_notional


def _enter_position(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, mult: float, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == omega.ACTION_CASH:
        return cash, Position(), "no_signal"
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), "entry_miss"
    notional = float(np.clip(float(row.get("notional_exposure", 0.0) or 0.0) * float(mult), 0.0, 1.20))
    if notional <= 0.0:
        return cash, Position(), "zero_exposure"
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        Position(
            side=side,
            entry_price=float(entry_px),
            entry_i=min(int(i) + 1, len(arrays["close"]) - 1),
            notional=notional,
            take_profit=float(row.get("take_profit", 0.0) or 0.0),
            stop_loss=abs(float(row.get("stop_loss", 0.0) or 0.0)),
        ),
        "entry",
    )


def _apply_action(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, pos: Position, i: int, action: int, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    if pos.side == 0:
        if action == ENTER_BASE:
            return _enter_position(cash, arrays, dec, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        if action == ENTER_AGGRESSIVE:
            return _enter_position(cash, arrays, dec, i, 1.25, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "skip"
    if action == FULL_EXIT:
        cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "full_exit"
    if action == REDUCE50:
        cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "reduce50"
    if action == RESIZE_UP:
        add_notional = min(0.25, max(0.0, 1.20 - pos.notional))
        if add_notional <= 0.0:
            return cash, pos, "resize_up_blocked"
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), pos.side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            return cash, pos, "resize_up_miss"
        before_notional = pos.notional
        pos.entry_price = (pos.entry_price * before_notional + float(px) * add_notional) / max(before_notional + add_notional, 1e-12)
        pos.notional = before_notional + add_notional
        cash -= cash * float(entry_fee) * add_notional
        return cash, pos, "resize_up"
    if action == REVERSE:
        old_side = pos.side
        cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        row = dec.iloc[int(i)].copy()
        dec_tmp = dec.copy()
        dec_tmp.loc[dec_tmp.index[int(i)], "side"] = -old_side
        dec_tmp.loc[dec_tmp.index[int(i)], "action"] = omega.ACTION_SHORT if old_side > 0 else omega.ACTION_LONG
        cash, pos, reason = _enter_position(cash, arrays, dec_tmp, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "reverse" if reason == "entry" else f"reverse_{reason}"
    return cash, pos, "hold"


def _continue_to_end(cash: float, arrays: dict[str, np.ndarray], pos: Position, start_i: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, Position, str, int]:
    if pos.side == 0:
        return cash, pos, "flat", int(start_i)
    last_i = len(arrays["close"]) - 2 if int(max_bars) <= 0 else min(len(arrays["close"]) - 2, int(start_i) + int(max_bars))
    reason = "sim_horizon" if int(max_bars) > 0 else "forced_end"
    exit_i = last_i
    for j in range(int(start_i), last_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
        unreal = raw * pos.notional
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        if pos.stop_loss > 0.0 and unreal <= -pos.stop_loss:
            reason = "stop_loss"
            exit_i = int(j)
            break
        if pos.take_profit > 0.0 and unreal >= pos.take_profit:
            reason = "take_profit"
            exit_i = int(j)
            break
    cash, pos, _ = _realize_fraction(cash, arrays, pos, exit_i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
    return cash, pos, reason, int(exit_i)


def _position_open_at(arrays: dict[str, np.ndarray], pos: Position, target_i: int, *, slip_eff: float) -> Position | None:
    if pos.side == 0:
        return None
    out = Position(**pos.__dict__)
    for j in range(int(pos.entry_i), int(target_i) + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - out.entry_price) / max(out.entry_price, 1e-12) if out.side > 0 else (out.entry_price - px * (1.0 + slip_eff)) / max(out.entry_price, 1e-12)
        unreal = raw * out.notional
        out.mfe = max(out.mfe, unreal)
        out.mae = min(out.mae, unreal)
        if out.stop_loss > 0.0 and unreal <= -out.stop_loss:
            return None
        if out.take_profit > 0.0 and unreal >= out.take_profit:
            return None
    return out


def _counterfactual_reward(arrays: dict[str, np.ndarray], dec: pd.DataFrame, pos: Position, i: int, action: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, str]:
    cash = 1.0
    pos_copy = Position(**pos.__dict__)
    cash, pos_copy, reason = _apply_action(cash, arrays, dec, pos_copy, i, action, fee_eff=fee_eff, slip_eff=slip_eff)
    if pos_copy.side != 0:
        cash, _pos, cont_reason, _exit_i = _continue_to_end(cash, arrays, pos_copy, max(int(i) + 1, pos_copy.entry_i), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
        reason = cont_reason if reason in {"hold", "entry"} else reason
    reward = float(cash - 1.0)
    if action == RESIZE_UP:
        reward -= 0.0015
    elif action == REVERSE:
        reward -= 0.0025
    elif action == REDUCE50:
        reward -= 0.0005
    return reward, reason


def _allowed_actions(
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    pos: Position,
    i: int,
    *,
    slip_eff: float,
    disable_resize: bool,
    disable_reverse: bool,
) -> list[int]:
    if pos.side == 0:
        if bool(omega._active(dec)[int(i)]):
            return [HOLD_OR_SKIP, ENTER_BASE, ENTER_AGGRESSIVE]
        return [HOLD_OR_SKIP]
    vals = _position_values(arrays, pos, int(i), slip_eff=slip_eff)
    unreal = float(vals["lc_pos_unrealized"])
    hold_bars = float(vals["lc_pos_hold_bars"])
    giveback = float(vals["lc_pos_giveback"])
    allowed = [HOLD_OR_SKIP]
    if hold_bars >= 2 and unreal > 0.003:
        allowed.append(REDUCE50)
    if hold_bars >= 2 and (unreal > 0.004 or unreal < -0.003 or giveback > 0.50):
        allowed.append(FULL_EXIT)
    if not bool(disable_resize) and hold_bars >= 2 and unreal > 0.004 and pos.notional < 0.90:
        allowed.append(RESIZE_UP)
    row_side = int(dec.iloc[int(i)].get("side", 0) or 0) if bool(omega._active(dec)[int(i)]) else 0
    if not bool(disable_reverse) and hold_bars >= 2 and row_side == -pos.side and unreal < -0.006:
        allowed.append(REVERSE)
    return allowed


class MambaDiscreteActorCritic(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.enc = feat_coord.old_coord.MambaEncoder(input_dim, d_model, emb_dim)
        self.actor = nn.Sequential(nn.Linear(emb_dim, 96), nn.SiLU(), nn.Linear(96, n_actions))
        self.critic = nn.Sequential(nn.Linear(emb_dim, 96), nn.SiLU(), nn.Linear(96, n_actions))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(seq)
        return self.actor(h), self.critic(h)


def _build_dataset(
    frames: dict[str, Any],
    *,
    seq_len: int,
    max_entries: int,
    samples_per_entry: int,
    seed: int,
    fee: float,
    slip: float,
    cost_mult: float,
    max_sim_bars: int,
    min_action_edge: float,
    disable_resize: bool,
    disable_reverse: bool,
    position_only_training: bool,
    norm: dict[str, Any],
) -> tuple[OfflineData, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    frame = frames["train_df"]
    state = _base_state(frames["s_train"])
    dec = frames["train_dec"]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = _apply_norm(state, norm)
    base_seq = _rolling_sequences(base_norm, seq_len)
    active_idx = np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - max(4, int(max_sim_bars) + 2 if max_sim_bars > 0 else 4)))
    if max_entries > 0 and len(active_idx) > max_entries:
        active_idx = np.sort(rng.choice(active_idx, size=int(max_entries), replace=False))
    seqs: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    best_actions: list[int] = []
    weights: list[float] = []
    reason_counts: dict[str, int] = {}
    for entry_i in active_idx:
        candidate_states: list[tuple[int, Position]] = [] if bool(position_only_training) else [(int(entry_i), Position())]
        cash, pos, reason = _enter_position(1.0, arrays, dec, int(entry_i), 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        if pos.side != 0:
            offsets = sorted(set([1, 2, 3, 6, 12, 24, 48, 96]))
            if samples_per_entry > 0:
                offsets = offsets[: int(samples_per_entry)]
            for off in offsets:
                j = min(pos.entry_i + int(off), len(frame) - 3)
                open_pos = _position_open_at(arrays, pos, int(j), slip_eff=slip_eff)
                if open_pos is None:
                    break
                candidate_states.append((int(j), open_pos))
        for i, pos_state in candidate_states:
            row = _state_row(state, arrays, pos_state, i, slip_eff=slip_eff)
            row_norm = _apply_norm(row, norm)
            rewards = np.full(len(ACTION_NAMES), -0.02, dtype=np.float32)
            allowed = set(_allowed_actions(arrays, dec, pos_state, i, slip_eff=slip_eff, disable_resize=disable_resize, disable_reverse=disable_reverse))
            for action in range(len(ACTION_NAMES)):
                if action not in allowed:
                    continue
                if pos_state.side == 0 and action not in (HOLD_OR_SKIP, ENTER_BASE, ENTER_AGGRESSIVE):
                    continue
                if pos_state.side != 0 and action in (ENTER_BASE, ENTER_AGGRESSIVE):
                    continue
                reward, action_reason = _counterfactual_reward(arrays, dec, pos_state, i, action, fee_eff=fee_eff, slip_eff=slip_eff, max_bars=int(max_sim_bars))
                rewards[action] = float(reward)
                reason_counts[action_reason] = reason_counts.get(action_reason, 0) + 1
            best = int(np.argmax(rewards))
            if float(rewards[best]) <= float(min_action_edge):
                best = HOLD_OR_SKIP
                rewards[HOLD_OR_SKIP] = max(float(rewards[HOLD_OR_SKIP]), 0.0)
            scale = max(float(np.std(rewards)), 1e-4)
            seqs.append(_seq_for_state(base_seq, row_norm, i))
            qs.append(rewards)
            best_actions.append(best)
            weights.append(float(np.exp(np.clip((float(rewards[best]) - float(np.median(rewards))) / scale, -4.0, 4.0))))
    if not seqs:
        raise RuntimeError("empty lifecycle controller dataset")
    q_arr = np.asarray(qs, dtype=np.float32)
    return (
        OfflineData(np.asarray(seqs, dtype=np.float32), q_arr, np.asarray(best_actions, dtype=np.int64), np.asarray(weights, dtype=np.float32)),
        {
            "entries": int(len(active_idx)),
            "samples": int(len(seqs)),
            "best_action_counts": {ACTION_NAMES[i]: int(v) for i, v in enumerate(np.bincount(np.asarray(best_actions), minlength=len(ACTION_NAMES)))},
            "q_mean": float(np.mean(q_arr)),
            "q_best_mean": float(np.mean(q_arr.max(axis=1))),
            "counterfactual_reasons": reason_counts,
        },
    )


def _train(
    data: OfflineData,
    *,
    device: torch.device,
    steps: int,
    batch_size: int,
    lr: float,
    class_balance_actor: bool,
) -> tuple[MambaDiscreteActorCritic, dict[str, Any]]:
    model = MambaDiscreteActorCritic(data.seq.shape[-1], len(ACTION_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.seq), torch.from_numpy(data.q_targets), torch.from_numpy(data.best_actions), torch.from_numpy(data.weights))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
    class_w = torch.ones(len(ACTION_NAMES), dtype=torch.float32, device=device)
    if bool(class_balance_actor):
        counts = np.bincount(data.best_actions, minlength=len(ACTION_NAMES)).astype(np.float32)
        counts[counts < 1.0] = 1.0
        inv = counts.sum() / (len(ACTION_NAMES) * counts)
        class_w = torch.from_numpy(np.clip(inv, 0.25, 8.0).astype(np.float32)).to(device)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        try:
            seq_b, q_b, a_b, w_b = next(it)
        except StopIteration:
            it = iter(dl)
            seq_b, q_b, a_b, w_b = next(it)
        seq_b = seq_b.to(device)
        q_b = q_b.to(device)
        a_b = a_b.to(device)
        w_b = w_b.to(device)
        logits, q_pred = model(seq_b)
        critic_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_b)
        ce = torch.nn.functional.cross_entropy(logits, a_b, reduction="none")
        actor_w = w_b * class_w[a_b]
        actor_loss = (ce * actor_w).sum() / torch.clamp(actor_w.sum(), min=1.0)
        probs = torch.softmax(logits, dim=1)
        policy_q = (probs * q_pred.detach()).sum(dim=1).mean()
        loss = critic_loss + actor_loss - 0.25 * policy_q
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            last = {"step": int(step), "critic_loss": float(critic_loss.detach().cpu()), "actor_loss": float(actor_loss.detach().cpu()), "policy_q": float(policy_q.detach().cpu())}
    return model.cpu(), last


@torch.no_grad()
def _select_action(model: MambaDiscreteActorCritic, seq: np.ndarray, allowed: list[int], *, device: torch.device, select_mode: str) -> int:
    model = model.to(device)
    model.eval()
    logits, q = model(torch.from_numpy(seq[None, :, :]).to(device))
    if str(select_mode) == "q_only":
        score = q
    else:
        # Conservative: use critic-filtered actor score instead of actor logits alone.
        score = torch.softmax(logits, dim=1) * torch.clamp(q, min=-0.05, max=0.05).add(0.05)
    mask = torch.full_like(score, -1.0)
    mask[:, [int(a) for a in allowed]] = 1.0
    score = torch.where(mask > 0, score, torch.full_like(score, -1e9))
    return int(torch.argmax(score, dim=1).detach().cpu().item())


def _replay(
    frames: dict[str, Any],
    split: str,
    model: MambaDiscreteActorCritic,
    norm: dict[str, Any],
    *,
    seq_len: int,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    disable_resize: bool,
    disable_reverse: bool,
    select_mode: str,
    force_parent_entry: bool,
    force_entry_mult: float,
) -> dict[str, Any]:
    frame = frames[f"{split}_df"]
    state = _base_state(frames[f"s_{split}"])
    dec = frames[f"{split}_dec"]
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = _apply_norm(state, norm)
    base_seq = _rolling_sequences(base_norm, seq_len)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Position()
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    partials = 0
    resizes = 0
    reverses = 0
    reasons: dict[str, int] = {}
    active = omega._active(dec)
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            vals = _position_values(arrays, pos, i, slip_eff=slip_eff)
            pos.mfe = max(pos.mfe, vals["lc_pos_unrealized"])
            pos.mae = min(pos.mae, vals["lc_pos_unrealized"])
            eq = cash * (1.0 + vals["lc_pos_unrealized"])
            if pos.stop_loss > 0.0 and vals["lc_pos_unrealized"] <= -pos.stop_loss:
                before = cash
                cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                trades += 1
                wins += int(cash > before)
                reasons["stop_loss"] = reasons.get("stop_loss", 0) + 1
                continue
            if pos.take_profit > 0.0 and vals["lc_pos_unrealized"] >= pos.take_profit:
                before = cash
                cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                trades += 1
                wins += int(cash > before)
                reasons["take_profit"] = reasons.get("take_profit", 0) + 1
                continue
        else:
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos.side == 0 and not bool(active[i]):
            continue
        if pos.side == 0 and bool(force_parent_entry):
            cash, pos, reason = _enter_position(cash, arrays, dec, i, float(force_entry_mult), fee_eff=fee_eff, slip_eff=slip_eff)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "entry":
                long_entries += int(pos.side > 0)
                short_entries += int(pos.side < 0)
            continue
        row = _state_row(state, arrays, pos, i, slip_eff=slip_eff)
        seq = _seq_for_state(base_seq, _apply_norm(row, norm), i)
        allowed = _allowed_actions(arrays, dec, pos, i, slip_eff=slip_eff, disable_resize=disable_resize, disable_reverse=disable_reverse)
        action = _select_action(model, seq, allowed, device=device, select_mode=select_mode)
        if pos.side == 0 and action not in (ENTER_BASE, ENTER_AGGRESSIVE):
            reasons["skip"] = reasons.get("skip", 0) + 1
            continue
        if pos.side != 0 and action in (ENTER_BASE, ENTER_AGGRESSIVE):
            action = HOLD_OR_SKIP
        before = cash
        old_side = pos.side
        old_notional = pos.notional
        cash, pos, reason = _apply_action(cash, arrays, dec, pos, i, action, fee_eff=fee_eff, slip_eff=slip_eff)
        reasons[reason] = reasons.get(reason, 0) + 1
        if old_side == 0 and pos.side != 0:
            long_entries += int(pos.side > 0)
            short_entries += int(pos.side < 0)
        if reason in {"full_exit", "reverse"} or (old_side != 0 and pos.side == 0):
            trades += 1
            wins += int(cash > before)
        if reason == "reduce50":
            partials += 1
        if reason == "resize_up":
            resizes += 1
        if reason.startswith("reverse"):
            reverses += 1
    if pos.side != 0:
        before = cash
        cash, pos, _ = _realize_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        trades += 1
        wins += int(cash > before)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "partials": int(partials),
        "resizes": int(resizes),
        "reverses": int(reverses),
        "reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-threshold", type=float, default=0.65)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=500)
    ap.add_argument("--samples-per-entry", type=int, default=5)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--min-action-edge", type=float, default=0.002)
    ap.add_argument("--disable-resize", action="store_true")
    ap.add_argument("--disable-reverse", action="store_true")
    ap.add_argument("--class-balance-actor", action="store_true")
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--position-only-training", action="store_true")
    ap.add_argument("--force-parent-entry", action="store_true")
    ap.add_argument("--force-entry-mult", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260604)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = feat_coord._prepare_frames(feat_coord.DEFAULT_3HEAD_DIR, quality_threshold=float(args.quality_threshold), device=device)
    fee, slip = omega._load_fee_slip()
    state_cols = [c for c in _base_state(frames["s_train"]).columns if c != "timestamp"]
    bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    norm = _fit_norm(_base_state(frames["s_train"])[state_cols])
    data, data_diag = _build_dataset(
        frames,
        seq_len=int(args.seq_len),
        max_entries=int(args.max_train_entries),
        samples_per_entry=int(args.samples_per_entry),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_sim_bars=int(args.train_max_sim_bars),
        min_action_edge=float(args.min_action_edge),
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        position_only_training=bool(args.position_only_training),
        norm=norm,
    )
    print(json.dumps({"stage": "lifecycle_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    model, train_diag = _train(
        data,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        class_balance_actor=bool(args.class_balance_actor),
    )
    val = _replay(
        frames,
        "val",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    oos = _replay(
        frames,
        "oos",
        model,
        norm,
        seq_len=int(args.seq_len),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        disable_resize=bool(args.disable_resize),
        disable_reverse=bool(args.disable_reverse),
        select_mode=str(args.select_mode),
        force_parent_entry=bool(args.force_parent_entry),
        force_entry_mult=float(args.force_entry_mult),
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Discrete Mamba offline SAC-style lifecycle controller. Exit Head is feature-only. Actions include hold/skip, enter_base, enter_aggressive, reduce50, full_exit, resize_up, and reverse. Position resize/reduce fees are charged on delta notional.",
        "action_names": ACTION_NAMES,
        "quality_threshold": float(args.quality_threshold),
        "state_columns": state_cols,
        "training": {
            "seq_len": int(args.seq_len),
            "max_train_entries": int(args.max_train_entries),
            "samples_per_entry": int(args.samples_per_entry),
            "train_max_sim_bars": int(args.train_max_sim_bars),
            "min_action_edge": float(args.min_action_edge),
            "disable_resize": bool(args.disable_resize),
            "disable_reverse": bool(args.disable_reverse),
            "class_balance_actor": bool(args.class_balance_actor),
            "select_mode": str(args.select_mode),
            "position_only_training": bool(args.position_only_training),
            "force_parent_entry": bool(args.force_parent_entry),
            "force_entry_mult": float(args.force_entry_mult),
            "steps": int(args.steps),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "data_diag": data_diag,
            "train_diag": train_diag,
        },
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult), "delta_notional_resize_fee": True, "partial_exit_fee": True},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "model": str(out_dir / "lifecycle_controller.pt")},
    }
    torch.save({"model_state_dict": model.state_dict(), "normalizer": norm, "seq_len": int(args.seq_len), "state_columns": state_cols, "action_names": ACTION_NAMES}, out_dir / "lifecycle_controller.pt")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
