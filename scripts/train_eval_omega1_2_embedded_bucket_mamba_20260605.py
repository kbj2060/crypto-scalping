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
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_embedded_bucket_mamba_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

TP_BUCKETS = np.asarray([0.018, 0.022, 0.026, 0.030, 0.034], dtype=np.float32)
SL_BUCKETS = np.asarray([0.008, 0.010, 0.012, 0.014, 0.018], dtype=np.float32)
NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
ENTRY_BUCKETS = np.asarray(
    [(t, s, n, l) for t in range(len(TP_BUCKETS)) for s in range(len(SL_BUCKETS)) for n in range(len(NOTIONAL_BUCKETS)) for l in range(len(LEVERAGE_BUCKETS))],
    dtype=np.int64,
)

HOLD_OR_SKIP = 0
FIRST_ENTRY = 1
REDUCE50 = FIRST_ENTRY + len(ENTRY_BUCKETS)
FULL_EXIT = REDUCE50 + 1
ACTION_NAMES = ["hold_or_skip", *[f"enter_tp{t}_sl{s}_n{n}_lev{l}" for t, s, n, l in ENTRY_BUCKETS], "reduce50", "full_exit"]


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


class EmbeddedBucketMamba(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.enc = feat_coord.old_coord.MambaEncoder(input_dim, d_model, emb_dim)
        self.actor = nn.Sequential(nn.Linear(emb_dim, 128), nn.SiLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(emb_dim, 128), nn.SiLU(), nn.Linear(128, n_actions))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(seq)
        return self.actor(h), self.critic(h)


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
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _risk_for_entry_action(action: int) -> dict[str, float]:
    idx = int(action) - FIRST_ENTRY
    if idx < 0 or idx >= len(ENTRY_BUCKETS):
        raise RuntimeError(f"entry action has no bucket risk: {action}")
    t, s, n, l = [int(x) for x in ENTRY_BUCKETS[idx]]
    return {"tp": float(TP_BUCKETS[t]), "sl": float(SL_BUCKETS[s]), "notional": float(NOTIONAL_BUCKETS[n]), "leverage": float(LEVERAGE_BUCKETS[l])}


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


def _enter_bucket(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, action: int, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == omega.ACTION_CASH:
        return cash, Position(), "no_signal"
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), "entry_miss"
    risk = _risk_for_entry_action(int(action))
    notional = float(np.clip(risk["notional"], 0.0, 1.20))
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        Position(side=side, entry_price=float(entry_px), entry_i=min(int(i) + 1, len(arrays["close"]) - 1), notional=notional, take_profit=float(risk["tp"]), stop_loss=abs(float(risk["sl"]))),
        "entry",
    )


def _position_values(arrays: dict[str, np.ndarray], pos: Position, i: int, *, slip_eff: float) -> dict[str, float]:
    if pos.side == 0 or pos.notional <= 0.0:
        return {c: 0.0 for c in lifecycle.POS_COLS}
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
    for k, v in _position_values(arrays, pos, i, slip_eff=slip_eff).items():
        row[k] = v
    return row


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


def _allowed_actions(dec: pd.DataFrame, pos: Position, i: int) -> list[int]:
    if pos.side == 0:
        if bool(omega._active(dec)[int(i)]):
            return [HOLD_OR_SKIP, *range(FIRST_ENTRY, FIRST_ENTRY + len(ENTRY_BUCKETS))]
        return [HOLD_OR_SKIP]
    hold_bars = int(i) - int(pos.entry_i)
    allowed = [HOLD_OR_SKIP]
    if hold_bars >= 2:
        allowed.extend([REDUCE50, FULL_EXIT])
    return allowed


def _apply_action(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, pos: Position, i: int, action: int, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    if pos.side == 0:
        if FIRST_ENTRY <= int(action) < FIRST_ENTRY + len(ENTRY_BUCKETS):
            return _enter_bucket(cash, arrays, dec, i, int(action), fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "skip"
    if int(action) == FULL_EXIT:
        cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "full_exit"
    if int(action) == REDUCE50:
        cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
        return cash, pos, "reduce50"
    return cash, pos, "hold"


def _counterfactual_reward(arrays: dict[str, np.ndarray], dec: pd.DataFrame, pos: Position, i: int, action: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, str]:
    cash = 1.0
    pos_copy = Position(**pos.__dict__)
    cash, pos_copy, reason = _apply_action(cash, arrays, dec, pos_copy, i, int(action), fee_eff=fee_eff, slip_eff=slip_eff)
    if pos_copy.side != 0:
        cash, _pos, cont_reason, _exit_i = _continue_to_end(cash, arrays, pos_copy, max(int(i) + 1, pos_copy.entry_i), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
        reason = cont_reason if reason in {"hold", "entry"} else reason
    reward = float(cash - 1.0)
    if int(action) == REDUCE50:
        reward -= 0.0005
    return reward, reason


def _base_decision_frames(threehead_dir: Path, quality_threshold: float, device: torch.device) -> dict[str, Any]:
    return feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)


def _build_state_cols(frames: dict[str, Any]) -> list[str]:
    return [c for c in lifecycle._base_state(frames["s_train"]).columns if c != "timestamp"]


def _sample_entry_actions(rng: np.random.Generator, n: int) -> np.ndarray:
    base = []
    for ids in [(2, 3, 3, 1), (1, 2, 2, 1), (3, 3, 3, 1), (4, 4, 2, 1), (0, 1, 1, 0), (2, 2, 4, 2)]:
        loc = np.flatnonzero(np.all(ENTRY_BUCKETS == np.asarray(ids), axis=1))
        if len(loc):
            base.append(FIRST_ENTRY + int(loc[0]))
    total = np.arange(FIRST_ENTRY, FIRST_ENTRY + len(ENTRY_BUCKETS), dtype=np.int64)
    if int(n) <= 0 or int(n) >= len(total):
        return total
    rest = np.setdiff1d(total, np.asarray(base, dtype=np.int64), assume_unique=False)
    take = rng.choice(rest, size=max(int(n) - len(base), 0), replace=False)
    return np.sort(np.unique(np.concatenate([np.asarray(base, dtype=np.int64), take])))


def _build_dataset(
    frames: dict[str, Any],
    *,
    seq_len: int,
    max_entries: int,
    entry_actions_per_state: int,
    samples_per_entry: int,
    seed: int,
    fee: float,
    slip: float,
    cost_mult: float,
    max_sim_bars: int,
    min_action_edge: float,
    norm: dict[str, Any],
) -> tuple[OfflineData, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    frame = frames["train_df"]
    state = lifecycle._base_state(frames["s_train"])
    dec = frames["train_dec"]
    arrays = _arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = lifecycle._apply_norm(state, norm)
    base_seq = lifecycle._rolling_sequences(base_norm, seq_len)
    active_idx = np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - max(4, int(max_sim_bars) + 2 if max_sim_bars > 0 else 4)))
    if int(max_entries) > 0 and len(active_idx) > int(max_entries):
        active_idx = np.sort(rng.choice(active_idx, size=int(max_entries), replace=False))
    entry_actions = _sample_entry_actions(rng, int(entry_actions_per_state))
    seqs: list[np.ndarray] = []
    qs: list[np.ndarray] = []
    best_actions: list[int] = []
    weights: list[float] = []
    reason_counts: dict[str, int] = {}
    default_base_action = int(_sample_entry_actions(rng, 1)[0])
    for entry_i in active_idx:
        candidate_states: list[tuple[int, Position]] = [(int(entry_i), Position())]
        cash, pos, _reason = _enter_bucket(1.0, arrays, dec, int(entry_i), default_base_action, fee_eff=fee_eff, slip_eff=slip_eff)
        if pos.side != 0:
            offsets = sorted(set([1, 2, 3, 6, 12, 24, 48, 96]))[: int(samples_per_entry)]
            for off in offsets:
                j = min(pos.entry_i + int(off), len(frame) - 3)
                open_pos = _position_open_at(arrays, pos, int(j), slip_eff=slip_eff)
                if open_pos is None:
                    break
                candidate_states.append((int(j), open_pos))
        for i, pos_state in candidate_states:
            row = _state_row(state, arrays, pos_state, i, slip_eff=slip_eff)
            row_norm = lifecycle._apply_norm(row, norm)
            rewards = np.full(len(ACTION_NAMES), -0.02, dtype=np.float32)
            allowed = _allowed_actions(dec, pos_state, i)
            if pos_state.side == 0:
                allowed_eval = [HOLD_OR_SKIP, *entry_actions.tolist()]
            else:
                allowed_eval = allowed
            for action in allowed_eval:
                reward, action_reason = _counterfactual_reward(arrays, dec, pos_state, i, int(action), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=int(max_sim_bars))
                rewards[int(action)] = float(reward)
                reason_counts[action_reason] = reason_counts.get(action_reason, 0) + 1
            best = int(np.argmax(rewards))
            if float(rewards[best]) <= float(min_action_edge):
                best = HOLD_OR_SKIP
                rewards[HOLD_OR_SKIP] = max(float(rewards[HOLD_OR_SKIP]), 0.0)
            scale = max(float(np.std(rewards)), 1e-4)
            seqs.append(lifecycle._seq_for_state(base_seq, row_norm, i))
            qs.append(rewards)
            best_actions.append(best)
            weights.append(float(np.exp(np.clip((float(rewards[best]) - float(np.median(rewards))) / scale, -4.0, 4.0))))
    if not seqs:
        raise RuntimeError("empty embedded bucket dataset")
    q_arr = np.asarray(qs, dtype=np.float32)
    return (
        OfflineData(np.asarray(seqs, dtype=np.float32), q_arr, np.asarray(best_actions, dtype=np.int64), np.asarray(weights, dtype=np.float32)),
        {
            "entries": int(len(active_idx)),
            "samples": int(len(seqs)),
            "entry_actions_evaluated": int(len(entry_actions)),
            "best_action_counts_top": {ACTION_NAMES[i]: int(v) for i, v in sorted(enumerate(np.bincount(np.asarray(best_actions), minlength=len(ACTION_NAMES))), key=lambda x: x[1], reverse=True)[:20] if v},
            "q_mean": float(np.mean(q_arr)),
            "q_best_mean": float(np.mean(q_arr.max(axis=1))),
            "counterfactual_reasons": reason_counts,
        },
    )


def _train(data: OfflineData, *, device: torch.device, steps: int, batch_size: int, lr: float) -> tuple[EmbeddedBucketMamba, dict[str, Any]]:
    model = EmbeddedBucketMamba(data.seq.shape[-1], len(ACTION_NAMES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(data.seq), torch.from_numpy(data.q_targets), torch.from_numpy(data.best_actions), torch.from_numpy(data.weights))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=True)
    it = iter(dl)
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
        actor_loss = (ce * w_b).sum() / torch.clamp(w_b.sum(), min=1.0)
        probs = torch.softmax(logits, dim=1)
        policy_q = (probs * q_pred.detach()).sum(dim=1).mean()
        entropy = -(probs * torch.clamp(probs, min=1e-8).log()).sum(dim=1).mean()
        loss = critic_loss + actor_loss - 0.20 * policy_q - 0.002 * entropy
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        if step % 250 == 0:
            last = {"step": int(step), "critic_loss": float(critic_loss.detach().cpu()), "actor_loss": float(actor_loss.detach().cpu()), "policy_q": float(policy_q.detach().cpu()), "entropy": float(entropy.detach().cpu())}
    return model.cpu(), last


@torch.no_grad()
def _select_action(model: EmbeddedBucketMamba, seq: np.ndarray, allowed: list[int], *, device: torch.device) -> int:
    model = model.to(device)
    model.eval()
    logits, q = model(torch.from_numpy(seq[None, :, :]).to(device))
    score = torch.softmax(logits, dim=1) * torch.clamp(q, min=-0.05, max=0.05).add(0.05)
    mask = torch.full_like(score, -1e9)
    mask[:, [int(a) for a in allowed]] = 0.0
    return int(torch.argmax(score + mask, dim=1).detach().cpu().item())


def _replay(frames: dict[str, Any], split: str, model: EmbeddedBucketMamba, norm: dict[str, Any], *, seq_len: int, fee: float, slip: float, cost_mult: float, device: torch.device) -> dict[str, Any]:
    frame = frames[f"{split}_df"]
    state = lifecycle._base_state(frames[f"s_{split}"])
    dec = frames[f"{split}_dec"]
    arrays = _arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = lifecycle._apply_norm(state, norm)
    base_seq = lifecycle._rolling_sequences(base_norm, seq_len)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Position()
    trades = wins = long_entries = short_entries = partials = 0
    reasons: dict[str, int] = {}
    active = omega._active(dec)
    entry_bucket_counts: dict[str, int] = {}
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
        row = _state_row(state, arrays, pos, i, slip_eff=slip_eff)
        seq = lifecycle._seq_for_state(base_seq, lifecycle._apply_norm(row, norm), i)
        action = _select_action(model, seq, _allowed_actions(dec, pos, i), device=device)
        if pos.side == 0 and action == HOLD_OR_SKIP:
            reasons["skip"] = reasons.get("skip", 0) + 1
            continue
        before = cash
        old_side = pos.side
        cash, pos, reason = _apply_action(cash, arrays, dec, pos, i, action, fee_eff=fee_eff, slip_eff=slip_eff)
        reasons[reason] = reasons.get(reason, 0) + 1
        if old_side == 0 and pos.side != 0:
            long_entries += int(pos.side > 0)
            short_entries += int(pos.side < 0)
            entry_bucket_counts[ACTION_NAMES[int(action)]] = entry_bucket_counts.get(ACTION_NAMES[int(action)], 0) + 1
        if reason == "reduce50":
            partials += 1
        if reason == "full_exit" or (old_side != 0 and pos.side == 0):
            trades += 1
            wins += int(cash > before)
    if pos.side != 0:
        before = cash
        cash, pos, _ = _realize_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        trades += 1
        wins += int(cash > before)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    top_buckets = dict(sorted(entry_bucket_counts.items(), key=lambda x: x[1], reverse=True)[:10])
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / trades) if trades else 0.0, "long_entries": int(long_entries), "short_entries": int(short_entries), "partials": int(partials), "resizes": 0, "reverses": 0, "reasons": reasons, "top_entry_buckets": top_buckets}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--max-train-entries", type=int, default=300)
    ap.add_argument("--entry-actions-per-state", type=int, default=96)
    ap.add_argument("--samples-per-entry", type=int, default=6)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--min-action-edge", type=float, default=0.002)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260660)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _base_decision_frames(Path(args.threehead_dir), float(args.quality_threshold), device)
    fee, slip = omega._load_fee_slip()
    state_cols = _build_state_cols(frames)
    bad = [c for c in state_cols if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden embedded bucket state columns passed audit: {bad[:20]}")
    norm = lifecycle._fit_norm(lifecycle._base_state(frames["s_train"])[state_cols])
    data, data_diag = _build_dataset(
        frames,
        seq_len=int(args.seq_len),
        max_entries=int(args.max_train_entries),
        entry_actions_per_state=int(args.entry_actions_per_state),
        samples_per_entry=int(args.samples_per_entry),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_sim_bars=int(args.train_max_sim_bars),
        min_action_edge=float(args.min_action_edge),
        norm=norm,
    )
    print(json.dumps({"stage": "embedded_bucket_mamba_train_start", "device": str(device), "seq_shape": list(data.seq.shape), "n_actions": len(ACTION_NAMES), "data_diag": data_diag}, ensure_ascii=False), flush=True)
    model, train_diag = _train(data, device=device, steps=int(args.steps), batch_size=int(args.batch_size), lr=float(args.lr))
    common = dict(seq_len=int(args.seq_len), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val = _replay(frames, "val", model, norm, **common)
    oos = _replay(frames, "oos", model, norm, **common)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "normalizer": norm,
            "seq_len": int(args.seq_len),
            "state_columns": state_cols,
            "action_names": ACTION_NAMES,
            "tp_buckets": TP_BUCKETS,
            "sl_buckets": SL_BUCKETS,
            "notional_buckets": NOTIONAL_BUCKETS,
            "leverage_buckets": LEVERAGE_BUCKETS,
            "entry_buckets": ENTRY_BUCKETS,
        },
        out_dir / "embedded_bucket_lifecycle_controller.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Embedded Bucket Mamba. Mamba lifecycle action space directly selects entry TP/SL/notional/leverage bucket combinations plus hold/reduce/full_exit. No named risk templates.",
        "accounting_note": "Lifecycle replay uses notional_exposure as effective account exposure; leverage is retained as selected bucket metadata but not multiplied again in replay PnL.",
        "quality_threshold": float(args.quality_threshold),
        "bucket_space": {"tp": TP_BUCKETS.tolist(), "sl": SL_BUCKETS.tolist(), "notional": NOTIONAL_BUCKETS.tolist(), "leverage": LEVERAGE_BUCKETS.tolist(), "n_entry_combos": int(len(ENTRY_BUCKETS)), "n_actions": int(len(ACTION_NAMES))},
        "state_columns": state_cols,
        "training": {"data_diag": data_diag, "train_diag": train_diag, "min_action_edge": float(args.min_action_edge), "steps": int(args.steps)},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "model": str(out_dir / "embedded_bucket_lifecycle_controller.pt")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
