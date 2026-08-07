#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
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

import train_eval_omega1_2_post_lifecycle_bucket_adapter_20260605 as base  # noqa: E402


MODEL_ID = "omega1_2_autoreg_softbucket_risk_20260606"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_STABLE_ADAPTER = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega1_2_post_lifecycle_bucket_adapter_20260605_hgb_base_nogate_traink3_replayk2_s260693"
    / "post_bucket_adapter.pkl"
)

TP_FINE = np.linspace(0.018, 0.080, 32, dtype=np.float32)
SL_FINE = np.linspace(0.010, 0.060, 32, dtype=np.float32)
MARGIN_FINE = np.asarray([0.20, 0.25, 0.30, 0.3375, 0.375, 0.405, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70], dtype=np.float32)
LEV_FINE = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
TAUS = torch.tensor([0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95], dtype=torch.float32)


@dataclass
class RiskSpec:
    tp_id: int
    sl_id: int
    margin_id: int
    lev_id: int


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _json_default(obj: Any) -> Any:
    return base._json_default(obj)


def _risk_values(ids: np.ndarray | tuple[int, int, int, int], *, notional_cap: float, compensate_ref: float, compensate_sltp: bool) -> dict[str, float]:
    t, s, m, l = [int(x) for x in ids]
    margin = float(MARGIN_FINE[m])
    lev = float(LEV_FINE[l])
    effective = float(np.clip(margin * lev, 0.0, float(notional_cap)))
    tp = float(TP_FINE[t])
    sl = float(SL_FINE[s])
    if bool(compensate_sltp):
        ref = max(float(compensate_ref), 1e-8)
        tp = float(tp / ref * effective)
        sl = float(sl / ref * effective)
    return {"tp": tp, "sl": sl, "margin": margin, "leverage": lev, "notional": effective}


def _enter_with_values(
    cash: float,
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    risk: dict[str, float],
    *,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, base.Position, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == base.omega.ACTION_CASH:
        return cash, base.Position(), "no_signal"
    filled, entry_px, entry_fee, _route = base.omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, base.Position(), "entry_miss"
    notional = float(risk["notional"])
    cash -= cash * float(entry_fee) * notional
    return cash, base.Position(side=side, entry_price=float(entry_px), entry_i=min(int(i) + 1, len(arrays["close"]) - 1), notional=notional, take_profit=float(risk["tp"]), stop_loss=abs(float(risk["sl"]))), "entry"


def _simulate_values(
    arrays: dict[str, np.ndarray],
    dec: pd.DataFrame,
    i: int,
    ids: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_bars: int,
    notional_cap: float,
    compensate_ref: float,
    compensate_sltp: bool,
) -> tuple[float, str]:
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    risk = _risk_values(ids, notional_cap=notional_cap, compensate_ref=compensate_ref, compensate_sltp=compensate_sltp)
    cash, pos, reason = _enter_with_values(1.0, arrays, dec, int(i), risk, fee_eff=fee_eff, slip_eff=slip_eff)
    if pos.side != 0:
        cash, reason = base._continue_to_end(cash, arrays, pos, max(int(i) + 1, pos.entry_i), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
    return float(cash - 1.0), reason


def _sample_risk_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    anchors = [
        (12, 10, 6, 1),
        (12, 10, 8, 2),
        (16, 12, 8, 2),
        (20, 16, 8, 2),
        (24, 18, 8, 2),
        (12, 10, 6, 4),
        (20, 16, 6, 4),
    ]
    out: list[tuple[int, int, int, int]] = list(anchors)
    while len(out) < int(n):
        out.append(
            (
                int(rng.integers(0, len(TP_FINE))),
                int(rng.integers(0, len(SL_FINE))),
                int(rng.integers(0, len(MARGIN_FINE))),
                int(rng.integers(0, len(LEV_FINE))),
            )
        )
    return np.asarray(out[: int(n)], dtype=np.int64)


def _build_counterfactual_dataset(
    x_train: pd.DataFrame,
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    entry_idx: np.ndarray,
    *,
    seed: int,
    candidates_per_row: int,
    fee: float,
    slip: float,
    cost_mult: float,
    max_bars: int,
    notional_cap: float,
    compensate_ref: float,
    compensate_sltp: bool,
    top_weight_temp: float,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    arrays = base._arrays(frame)
    state_rows: list[int] = []
    action_ids: list[np.ndarray] = []
    rewards: list[float] = []
    weights: list[float] = []
    best_ids: list[np.ndarray] = []
    reasons: dict[str, int] = {}
    for row_i, i in enumerate(entry_idx):
        candidates = _sample_risk_ids(rng, int(candidates_per_row))
        row_rewards: list[float] = []
        row_reasons: list[str] = []
        for ids in candidates:
            reward, reason = _simulate_values(
                arrays,
                dec,
                int(i),
                ids,
                fee=fee,
                slip=slip,
                cost_mult=cost_mult,
                max_bars=max_bars,
                notional_cap=notional_cap,
                compensate_ref=compensate_ref,
                compensate_sltp=compensate_sltp,
            )
            row_rewards.append(float(reward))
            row_reasons.append(str(reason))
            reasons[reason] = reasons.get(reason, 0) + 1
        rr = np.asarray(row_rewards, dtype=np.float64)
        centered = rr - float(np.max(rr))
        ww = np.exp(np.clip(centered / max(float(top_weight_temp), 1e-6), -12.0, 0.0))
        ww = ww / max(float(np.sum(ww)), 1e-12)
        best = candidates[int(np.argmax(rr))]
        for ids, reward, weight in zip(candidates, rr, ww):
            state_rows.append(int(row_i))
            action_ids.append(np.asarray(ids, dtype=np.int64))
            rewards.append(float(reward))
            weights.append(float(weight) * float(len(candidates)))
        best_ids.append(np.asarray(best, dtype=np.int64))
    return (
        x_train,
        np.asarray(state_rows, dtype=np.int64),
        np.asarray(action_ids, dtype=np.int64),
        np.asarray(rewards, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
        {
            "entry_rows": int(len(entry_idx)),
            "samples": int(len(rewards)),
            "best_tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(best_ids)[:, 0], minlength=len(TP_FINE)))},
            "best_sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(best_ids)[:, 1], minlength=len(SL_FINE)))},
            "best_margin_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(best_ids)[:, 2], minlength=len(MARGIN_FINE)))},
            "best_leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(best_ids)[:, 3], minlength=len(LEV_FINE)))},
            "exit_reasons": reasons,
            "reward_mean": float(np.mean(rewards)) if rewards else 0.0,
            "reward_max": float(np.max(rewards)) if rewards else 0.0,
        },
    )


class AutoregSoftBucketRisk(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 192, quantiles: int = len(TAUS)) -> None:
        super().__init__()
        self.state = nn.Sequential(nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.tp_head = nn.Linear(hidden, len(TP_FINE))
        self.tp_emb = nn.Embedding(len(TP_FINE), 16)
        self.sl_head = nn.Sequential(nn.Linear(hidden + 16, hidden), nn.SiLU(), nn.Linear(hidden, len(SL_FINE)))
        self.sl_emb = nn.Embedding(len(SL_FINE), 16)
        self.margin_head = nn.Sequential(nn.Linear(hidden + 32, hidden), nn.SiLU(), nn.Linear(hidden, len(MARGIN_FINE)))
        self.margin_emb = nn.Embedding(len(MARGIN_FINE), 16)
        self.lev_head = nn.Sequential(nn.Linear(hidden + 48, hidden), nn.SiLU(), nn.Linear(hidden, len(LEV_FINE)))
        self.lev_emb = nn.Embedding(len(LEV_FINE), 8)
        self.critic = nn.Sequential(nn.Linear(hidden + 56, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, quantiles))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.state(x)

    def forward_teacher(self, x: torch.Tensor, ids: torch.Tensor) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]:
        h = self.encode(x)
        tp_logits = self.tp_head(h)
        tp_e = self.tp_emb(ids[:, 0])
        sl_logits = self.sl_head(torch.cat([h, tp_e], dim=1))
        sl_e = self.sl_emb(ids[:, 1])
        margin_logits = self.margin_head(torch.cat([h, tp_e, sl_e], dim=1))
        margin_e = self.margin_emb(ids[:, 2])
        lev_logits = self.lev_head(torch.cat([h, tp_e, sl_e, margin_e], dim=1))
        lev_e = self.lev_emb(ids[:, 3])
        quant = self.critic(torch.cat([h, tp_e, sl_e, margin_e, lev_e], dim=1))
        return (tp_logits, sl_logits, margin_logits, lev_logits), quant

    def action_logits(self, x: torch.Tensor, ids: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        tp_logits = self.tp_head(h)
        tp_id = torch.argmax(tp_logits, dim=1) if ids is None else ids[:, 0]
        tp_e = self.tp_emb(tp_id)
        sl_logits = self.sl_head(torch.cat([h, tp_e], dim=1))
        sl_id = torch.argmax(sl_logits, dim=1) if ids is None else ids[:, 1]
        sl_e = self.sl_emb(sl_id)
        margin_logits = self.margin_head(torch.cat([h, tp_e, sl_e], dim=1))
        margin_id = torch.argmax(margin_logits, dim=1) if ids is None else ids[:, 2]
        margin_e = self.margin_emb(margin_id)
        lev_logits = self.lev_head(torch.cat([h, tp_e, sl_e, margin_e], dim=1))
        return tp_logits, sl_logits, margin_logits, lev_logits

    def quantiles_for_ids(self, x: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        tp_e = self.tp_emb(ids[:, 0])
        sl_e = self.sl_emb(ids[:, 1])
        margin_e = self.margin_emb(ids[:, 2])
        lev_e = self.lev_emb(ids[:, 3])
        return self.critic(torch.cat([h, tp_e, sl_e, margin_e, lev_e], dim=1))


def _quantile_huber(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    tau = TAUS.to(pred.device).view(1, -1)
    err = target.view(-1, 1) - pred
    huber = torch.where(err.abs() <= 1.0, 0.5 * err.pow(2), err.abs() - 0.5)
    return (torch.abs(tau - (err.detach() < 0).float()) * huber).mean(dim=1)


def _train_policy(
    x_norm: np.ndarray,
    state_rows: np.ndarray,
    action_ids: np.ndarray,
    rewards: np.ndarray,
    weights: np.ndarray,
    *,
    steps: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> tuple[AutoregSoftBucketRisk, dict[str, Any]]:
    model = AutoregSoftBucketRisk(x_norm.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(
        torch.from_numpy(x_norm[state_rows].astype(np.float32)),
        torch.from_numpy(action_ids.astype(np.int64)),
        torch.from_numpy(rewards.astype(np.float32)),
        torch.from_numpy(weights.astype(np.float32)),
    )
    dl = DataLoader(ds, batch_size=min(int(batch_size), len(ds)), shuffle=True, drop_last=False)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        for xb, ab, rb, wb in dl:
            xb, ab, rb, wb = xb.to(device), ab.to(device), rb.to(device), wb.to(device)
            logits, quant = model.forward_teacher(xb, ab)
            ce = (
                nn.functional.cross_entropy(logits[0], ab[:, 0], reduction="none")
                + nn.functional.cross_entropy(logits[1], ab[:, 1], reduction="none")
                + nn.functional.cross_entropy(logits[2], ab[:, 2], reduction="none")
                + nn.functional.cross_entropy(logits[3], ab[:, 3], reduction="none")
            )
            actor_loss = (ce * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            critic_loss = (_quantile_huber(quant, rb) * torch.clamp(wb, min=0.05)).mean()
            entropy = sum(torch.distributions.Categorical(logits=l).entropy().mean() for l in logits)
            loss = actor_loss + 2.0 * critic_loss - 0.005 * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        if step % 100 == 0 or step == int(steps):
            last = {
                "step": int(step),
                "actor_loss": float(actor_loss.detach().cpu()),
                "critic_loss": float(critic_loss.detach().cpu()),
                "entropy": float(entropy.detach().cpu()),
            }
    return model.cpu(), last


@torch.no_grad()
def _select_ids(
    model: AutoregSoftBucketRisk,
    x: np.ndarray,
    *,
    device: torch.device,
    mode: str,
    cvar_frac: float,
    rescale_threshold: float,
    rescale_floor: float,
    topk: int,
) -> tuple[np.ndarray, float, float]:
    model = model.to(device)
    model.eval()
    xb = torch.from_numpy(x[None, :].astype(np.float32)).to(device)
    logits = model.action_logits(xb)
    if str(mode) == "policy_only":
        ids = np.asarray([int(torch.argmax(l, dim=1).item()) for l in logits], dtype=np.int64)
        q = model.quantiles_for_ids(xb, torch.from_numpy(ids[None, :]).to(device)).detach().cpu().numpy().reshape(-1)
        cvar = float(np.mean(np.sort(q)[: max(1, int(len(q) * float(cvar_frac)))]))
        return ids, cvar, 1.0
    choices = []
    for l in logits:
        p = torch.softmax(l, dim=1).detach().cpu().numpy().reshape(-1)
        top = np.argsort(p)[::-1][: max(1, int(topk))]
        choices.append([(int(i), float(p[i])) for i in top])
    candidates: list[tuple[np.ndarray, float]] = []
    for t, pt in choices[0]:
        for s, ps in choices[1]:
            for m, pm in choices[2]:
                for le, pl in choices[3]:
                    candidates.append((np.asarray([t, s, m, le], dtype=np.int64), float(pt * ps * pm * pl)))
    ids_np = np.asarray([c[0] for c in candidates], dtype=np.int64)
    q = model.quantiles_for_ids(xb.repeat(len(ids_np), 1), torch.from_numpy(ids_np).to(device)).detach().cpu().numpy()
    k = max(1, int(q.shape[1] * float(cvar_frac)))
    cvar = np.mean(np.sort(q, axis=1)[:, :k], axis=1)
    prior = np.asarray([c[1] for c in candidates], dtype=np.float64)
    score = cvar + 0.0025 * np.log(np.clip(prior, 1e-12, None))
    best = int(np.argmax(score))
    scale = 1.0
    if str(mode) == "cvar_rescale":
        scale = float(np.clip(float(cvar[best]) / max(float(rescale_threshold), 1e-8), float(rescale_floor), 1.0)) if float(cvar[best]) < float(rescale_threshold) else 1.0
    return ids_np[best], float(cvar[best]), scale


def _risk_values_with_scale(ids: np.ndarray, scale: float, *, notional_cap: float, compensate_ref: float, compensate_sltp: bool) -> dict[str, float]:
    risk = _risk_values(ids, notional_cap=notional_cap, compensate_ref=compensate_ref, compensate_sltp=compensate_sltp)
    if float(scale) < 0.999:
        raw_margin = risk["margin"] * float(scale)
        risk["margin"] = float(max(raw_margin, 0.05))
        risk["notional"] = float(np.clip(risk["margin"] * risk["leverage"], 0.0, float(notional_cap)))
        if bool(compensate_sltp):
            risk["tp"] = float(TP_FINE[int(ids[0])] / max(float(compensate_ref), 1e-8) * risk["notional"])
            risk["sl"] = float(SL_FINE[int(ids[1])] / max(float(compensate_ref), 1e-8) * risk["notional"])
    return risk


def _nearest_id(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(value))))


def _fine_ids_from_base_ids(ids: np.ndarray) -> np.ndarray:
    risk = base._risk_from_ids(ids)
    return np.asarray(
        [
            _nearest_id(TP_FINE, float(risk["tp"])),
            _nearest_id(SL_FINE, abs(float(risk["sl"]))),
            _nearest_id(MARGIN_FINE, float(risk["notional"])),
            _nearest_id(LEV_FINE, float(risk["leverage"])),
        ],
        dtype=np.int64,
    )


def _load_baseline_adapter(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not Path(path).exists():
        raise RuntimeError(f"conservative baseline adapter missing: {path}")
    with Path(path).open("rb") as f:
        artifact = pickle.load(f)
    if "models" not in artifact or "normalizer" not in artifact:
        raise RuntimeError("conservative baseline adapter artifact contract mismatch")
    return artifact


@torch.no_grad()
def _cvar_for_ids(model: AutoregSoftBucketRisk, x: np.ndarray, ids: np.ndarray, *, device: torch.device, cvar_frac: float) -> float:
    model = model.to(device)
    model.eval()
    xb = torch.from_numpy(x[None, :].astype(np.float32)).to(device)
    idb = torch.from_numpy(ids[None, :].astype(np.int64)).to(device)
    q = model.quantiles_for_ids(xb, idb).detach().cpu().numpy().reshape(-1)
    k = max(1, int(len(q) * float(cvar_frac)))
    return float(np.mean(np.sort(q)[:k]))


def _replay_neural(
    frames: dict[str, Any],
    split: str,
    lifecycle_model: base.lifecycle.MambaDiscreteActorCritic,
    lifecycle_ckpt: dict[str, Any],
    model: AutoregSoftBucketRisk,
    norm: dict[str, Any],
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    select_mode: str,
    replay_enter_topk: int,
    mode: str,
    cvar_frac: float,
    rescale_threshold: float,
    rescale_floor: float,
    candidate_topk: int,
    notional_cap: float,
    compensate_ref: float,
    compensate_sltp: bool,
    conservative_baseline: dict[str, Any] | None,
    conservative_margin: float,
) -> dict[str, Any]:
    frame = frames[f"{split}_df"]
    state = base.lifecycle._base_state(frames[f"s_{split}"])
    dec = frames[f"{split}_dec"]
    arrays = base._arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = base.lifecycle._apply_norm(state, lifecycle_ckpt["normalizer"])
    base_seq = base.lifecycle._rolling_sequences(base_norm, int(lifecycle_ckpt["seq_len"]))
    active = base.omega._active(dec)
    cash = peak = 1.0
    mdd = 0.0
    pos = base.Position()
    lifecycle_pos = base.lifecycle.Position()
    trades = wins = long_entries = short_entries = 0
    reasons: dict[str, int] = {}
    risk_counts: dict[str, int] = {}
    cvars: list[float] = []
    scales: list[float] = []
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            vals = base._position_values(arrays, pos, i, slip_eff=slip_eff)
            pos.mfe = max(pos.mfe, vals["lc_pos_unrealized"])
            pos.mae = min(pos.mae, vals["lc_pos_unrealized"])
            eq = cash * (1.0 + vals["lc_pos_unrealized"])
            if pos.stop_loss > 0.0 and vals["lc_pos_unrealized"] <= -pos.stop_loss:
                before = cash
                cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                lifecycle_pos = base.lifecycle.Position()
                trades += 1
                wins += int(cash > before)
                reasons["stop_loss"] = reasons.get("stop_loss", 0) + 1
                continue
            if pos.take_profit > 0.0 and vals["lc_pos_unrealized"] >= pos.take_profit:
                before = cash
                cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                lifecycle_pos = base.lifecycle.Position()
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
        lc_row = base.lifecycle._state_row(state, arrays, lifecycle_pos, i, slip_eff=slip_eff)
        allowed = base.lifecycle._allowed_actions(arrays, dec, lifecycle_pos, i, slip_eff=slip_eff, disable_resize=True, disable_reverse=True)
        scores = base._lifecycle_scores(lifecycle_model, lifecycle_ckpt, base_seq, lc_row, allowed, i, device=device, select_mode=select_mode)
        lc_action = int(np.argmax(scores))
        if lifecycle_pos.side == 0 and lc_action not in (base.lifecycle.ENTER_BASE, base.lifecycle.ENTER_AGGRESSIVE):
            top_actions = np.argsort(scores)[::-1][: max(int(replay_enter_topk), 1)]
            enter_scores = [(base.lifecycle.ENTER_BASE, float(scores[base.lifecycle.ENTER_BASE])), (base.lifecycle.ENTER_AGGRESSIVE, float(scores[base.lifecycle.ENTER_AGGRESSIVE]))]
            enter_scores = [(a, s) for a, s in enter_scores if np.isfinite(s) and s > -1e8 and int(a) in set(int(x) for x in top_actions)]
            if not enter_scores:
                reasons["skip"] = reasons.get("skip", 0) + 1
                continue
            lc_action = int(max(enter_scores, key=lambda x: x[1])[0])
            reasons["topk_enter_candidate"] = reasons.get("topk_enter_candidate", 0) + 1
        if lifecycle_pos.side == 0:
            feat = base._adapter_feature_row(lc_row, lc_action)
            x = base._apply_norm(feat, norm)[0]
            ids, cvar, scale = _select_ids(model, x, device=device, mode=mode, cvar_frac=cvar_frac, rescale_threshold=rescale_threshold, rescale_floor=rescale_floor, topk=candidate_topk)
            risk = _risk_values_with_scale(ids, scale, notional_cap=notional_cap, compensate_ref=compensate_ref, compensate_sltp=compensate_sltp)
            if conservative_baseline is not None:
                base_ids, _base_meta = base._predict_hgb_ids(conservative_baseline["models"], feat, conservative_baseline["normalizer"])
                base_ids = base_ids[0]
                base_fine_ids = _fine_ids_from_base_ids(base_ids)
                base_cvar = _cvar_for_ids(model, x, base_fine_ids, device=device, cvar_frac=cvar_frac)
                if float(cvar) < float(base_cvar) + float(conservative_margin):
                    base_risk = base._risk_from_ids(base_ids)
                    risk = {
                        "tp": float(base_risk["tp"]),
                        "sl": abs(float(base_risk["sl"])),
                        "margin": float(base_risk["margin_notional"]),
                        "leverage": float(base_risk["leverage"]),
                        "notional": float(base_risk["notional"]),
                    }
                    ids = base_fine_ids
                    cvar = float(base_cvar)
                    scale = 1.0
                    reasons["conservative_baseline_fallback"] = reasons.get("conservative_baseline_fallback", 0) + 1
            before = cash
            cash, pos, reason = _enter_with_values(cash, arrays, dec, i, risk, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base._to_lifecycle_position(pos)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "entry":
                long_entries += int(pos.side > 0)
                short_entries += int(pos.side < 0)
                key = str(tuple(int(x) for x in ids))
                risk_counts[key] = risk_counts.get(key, 0) + 1
                cvars.append(float(cvar))
                scales.append(float(scale))
            continue
        if lc_action == base.lifecycle.FULL_EXIT:
            before = cash
            cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base.lifecycle.Position()
            trades += 1
            wins += int(cash > before)
            reasons["full_exit"] = reasons.get("full_exit", 0) + 1
        elif lc_action == base.lifecycle.REDUCE50:
            cash, pos, _ = base._realize_fraction(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = base._to_lifecycle_position(pos)
            reasons["reduce50"] = reasons.get("reduce50", 0) + 1
        else:
            reasons["hold"] = reasons.get("hold", 0) + 1
    if pos.side != 0:
        before = cash
        cash, pos, _ = base._realize_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
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
        "reasons": reasons,
        "top_risks": dict(sorted(risk_counts.items(), key=lambda x: x[1], reverse=True)[:10]),
        "cvar_mean": float(np.mean(cvars)) if cvars else 0.0,
        "scale_mean": float(np.mean(scales)) if scales else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=base.feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--baseline-lifecycle-dir", type=Path, default=base.BASELINE_LIFECYCLE_DIR)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--max-label-rows", type=int, default=0)
    ap.add_argument("--candidates-per-row", type=int, default=128)
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--enter-topk", type=int, default=3)
    ap.add_argument("--replay-enter-topk", type=int, default=2)
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--notional-cap", type=float, default=1.2)
    ap.add_argument("--compensate-sltp-by-notional", action="store_true")
    ap.add_argument("--compensate-ref-notional", type=float, default=0.45)
    ap.add_argument("--top-weight-temp", type=float, default=0.006)
    ap.add_argument("--steps", type=int, default=700)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--mode", choices=["policy_only", "cvar_select", "cvar_rescale"], default="cvar_rescale")
    ap.add_argument("--cvar-frac", type=float, default=0.35)
    ap.add_argument("--rescale-threshold", type=float, default=0.006)
    ap.add_argument("--rescale-floor", type=float, default=0.35)
    ap.add_argument("--candidate-topk", type=int, default=3)
    ap.add_argument("--conservative-baseline-path", type=Path, default=None)
    ap.add_argument("--conservative-margin", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=260800)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed(int(args.seed))
    device = base._device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = base._base_frames(Path(args.threehead_dir), float(args.quality_threshold), device)
    lifecycle_model, lifecycle_ckpt = base._load_baseline_lifecycle(Path(args.baseline_lifecycle_dir))
    bad = [c for c in lifecycle_ckpt["state_columns"] if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    fee, slip = base.omega._load_fee_slip()
    conservative_baseline = _load_baseline_adapter(args.conservative_baseline_path)
    x_train, entry_idx, _lc_actions, collect_diag = base._collect_train_entries(
        frames,
        lifecycle_model,
        lifecycle_ckpt,
        device=device,
        select_mode=str(args.select_mode),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.max_label_rows),
        enter_topk=int(args.enter_topk),
    )
    x_train, state_rows, action_ids, rewards, weights, cf_diag = _build_counterfactual_dataset(
        x_train,
        frames["train_df"],
        frames["train_dec"],
        entry_idx,
        seed=int(args.seed),
        candidates_per_row=int(args.candidates_per_row),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_bars=int(args.train_max_sim_bars),
        notional_cap=float(args.notional_cap),
        compensate_ref=float(args.compensate_ref_notional),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        top_weight_temp=float(args.top_weight_temp),
    )
    x_norm, norm = base._fit_norm(x_train)
    model, train_diag = _train_policy(
        x_norm,
        state_rows,
        action_ids,
        rewards,
        weights,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
    )
    ckpt = {
        "model_state_dict": model.state_dict(),
        "normalizer": norm,
        "state_columns": norm["columns"],
        "tp_fine": TP_FINE,
        "sl_fine": SL_FINE,
        "margin_fine": MARGIN_FINE,
        "lev_fine": LEV_FINE,
        "args": vars(args),
    }
    torch.save(ckpt, out_dir / "autoreg_softbucket_risk.pt")
    val = _replay_neural(
        frames,
        "val",
        lifecycle_model,
        lifecycle_ckpt,
        model,
        norm,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        select_mode=str(args.select_mode),
        replay_enter_topk=int(args.replay_enter_topk),
        mode=str(args.mode),
        cvar_frac=float(args.cvar_frac),
        rescale_threshold=float(args.rescale_threshold),
        rescale_floor=float(args.rescale_floor),
        candidate_topk=int(args.candidate_topk),
        notional_cap=float(args.notional_cap),
        compensate_ref=float(args.compensate_ref_notional),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        conservative_baseline=conservative_baseline,
        conservative_margin=float(args.conservative_margin),
    )
    oos = _replay_neural(
        frames,
        "oos",
        lifecycle_model,
        lifecycle_ckpt,
        model,
        norm,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        device=device,
        select_mode=str(args.select_mode),
        replay_enter_topk=int(args.replay_enter_topk),
        mode=str(args.mode),
        cvar_frac=float(args.cvar_frac),
        rescale_threshold=float(args.rescale_threshold),
        rescale_floor=float(args.rescale_floor),
        candidate_topk=int(args.candidate_topk),
        notional_cap=float(args.notional_cap),
        compensate_ref=float(args.compensate_ref_notional),
        compensate_sltp=bool(args.compensate_sltp_by_notional),
        conservative_baseline=conservative_baseline,
        conservative_margin=float(args.conservative_margin),
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Frozen Omega1.2 3-head TabM + frozen Mamba lifecycle; post-lifecycle HGB bucket adapter is replaced by autoregressive fine soft-bucket risk policy and distributional trajectory critic.",
        "architecture": {
            "policy": "autoregressive heads TP -> SL -> margin notional -> leverage",
            "critic": "state + action embeddings -> trade-level quantile outputs",
            "selection_mode": str(args.mode),
            "use_leverage_exposure": True,
            "notional_cap": float(args.notional_cap),
            "compensate_sltp_by_notional": bool(args.compensate_sltp_by_notional),
            "conservative_baseline_path": str(args.conservative_baseline_path) if args.conservative_baseline_path else None,
            "conservative_margin": float(args.conservative_margin),
        },
        "feature_audit": {
            "adapter_columns": int(len(norm["columns"])),
            "forbidden_count": int(len([c for c in norm["columns"] if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")])),
        },
        "risk_space": {
            "tp": TP_FINE.tolist(),
            "sl": SL_FINE.tolist(),
            "margin_notional": MARGIN_FINE.tolist(),
            "leverage": LEV_FINE.tolist(),
        },
        "training": {
            "collect_diag": collect_diag,
            "counterfactual_diag": cf_diag,
            "train_diag": train_diag,
            "steps": int(args.steps),
            "candidates_per_row": int(args.candidates_per_row),
        },
        "results": {"validation": val, "oos": oos},
        "artifacts": {
            "out_dir": str(out_dir),
            "model": str(out_dir / "autoreg_softbucket_risk.pt"),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
