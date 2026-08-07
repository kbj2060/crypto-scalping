#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, FEATURE_COLS  # noqa: E402
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import (  # noqa: E402
    DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL,
    PolicyValueNet,
    PVBundle,
    _predict_pv,
    _train_pv,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import _base_frame, _compact, backtest_no_limit_exit  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import (  # noqa: E402
    DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL,
    ENTRY_ACTIONS,
    MZBundle,
    MuZeroNet,
    _load_az_exit,
    _monthly,
    _planned_decisions,
    _plan_scores,
)


RISK_SCALES = np.asarray([0.0, 0.50, 0.75, 1.0, 1.25, 1.50], dtype=np.float64)
RISK_ACTIONS = tuple(f"scale_{s:.2f}" for s in RISK_SCALES)
DEFAULT_AZ_RISK_OUT = ROOT / "data/ensemble/supervised/zero_style/az_risk_overlay.pt"
DEFAULT_MZ_RISK_OUT = ROOT / "data/ensemble/supervised/zero_style/mz_risk_overlay.pt"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/zero_style_risk_overlay_2026.json"


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _standardize_pair(x: np.ndarray, x_next: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    x_next = np.asarray(x_next, dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    zn = (np.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return z.astype(np.float32), zn.astype(np.float32), mean, std


class RiskMuZeroNet(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden_dim: int = 192, latent_dim: int = 128):
        super().__init__()
        self.n_actions = int(n_actions)
        self.representation = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.SiLU(),
        )
        self.prediction = nn.Sequential(
            nn.Linear(int(latent_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.policy = nn.Linear(int(hidden_dim), int(n_actions))
        self.value = nn.Linear(int(hidden_dim), 1)
        self.dynamics = nn.Sequential(
            nn.Linear(int(latent_dim) + int(n_actions), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(latent_dim)),
            nn.LayerNorm(int(latent_dim)),
            nn.SiLU(),
        )
        self.reward = nn.Linear(int(latent_dim), 1)

    def initial(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.representation(x)
        logits, value = self.predict_from_latent(h)
        return h, logits, value

    def predict_from_latent(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.prediction(h)
        return self.policy(z), torch.tanh(self.value(z)).squeeze(-1)

    def recurrent(self, h: torch.Tensor, action_onehot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        h_next = self.dynamics(torch.cat([h, action_onehot], dim=-1))
        reward = torch.tanh(self.reward(h_next)).squeeze(-1)
        logits, value = self.predict_from_latent(h_next)
        return h_next, reward, logits, value


@dataclass
class MZRiskBundle:
    net: RiskMuZeroNet
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]
    actions: tuple[str, ...]


def _load_mz_entry(path: Path, device: str) -> MZBundle:
    payload = torch.load(path, map_location=device, weights_only=False)
    hidden = int(payload["state_dict"]["representation.0.weight"].shape[0])
    latent = int(payload["state_dict"]["representation.3.weight"].shape[0])
    net = MuZeroNet(len(payload["feature_cols"]), len(ENTRY_ACTIONS), hidden_dim=hidden, latent_dim=latent).to(device)
    net.load_state_dict(payload["state_dict"])
    return MZBundle(net, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), list(payload["feature_cols"]), ENTRY_ACTIONS)


def _mz_entry_decisions(
    df: pd.DataFrame,
    policy: dict[str, Any],
    entry_cfg: dict[str, Any],
    mz_entry: MZBundle,
    *,
    device: str,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feat, dec, close, fill = _base_frame(df, policy, entry_cfg)
    x = feat.reindex(columns=mz_entry.feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scores, probs, vals = _plan_scores(mz_entry, x, device=device, gamma=0.70, prior_weight=0.16, depth=1)
    mz_dec = _planned_decisions(dec, scores, probs, vals, score_floor=0.0, confidence_floor=0.0, value_floor=-0.05)
    return feat, mz_dec, close, fill, scores, probs, vals


def _state_frame(feat: pd.DataFrame, dec: pd.DataFrame, scores: np.ndarray, probs: np.ndarray, vals: np.ndarray) -> pd.DataFrame:
    out = feat.reindex(columns=FEATURE_COLS).copy()
    for i in range(scores.shape[1]):
        out[f"mz_entry_score_{i}"] = scores[:, i]
        out[f"mz_entry_prob_{i}"] = probs[:, i]
    out["mz_entry_value"] = vals
    out["base_side"] = pd.to_numeric(dec["side"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["base_notional"] = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["base_leverage"] = pd.to_numeric(dec["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    out["base_quality"] = pd.to_numeric(dec["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    out["base_confidence"] = pd.to_numeric(dec["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _risk_targets(
    df: pd.DataFrame,
    state: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    horizon: int,
    dynamics_step: int,
    fee: float,
    slip: float,
    temperature: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    actions = dec["action"].astype(int).to_numpy()
    sides = dec["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    idx = np.flatnonzero((actions != ACTION_CASH) & (sides != 0) & (notionals > 0.0))
    idx = idx[idx < len(df) - max(int(horizon), int(dynamics_step)) - 3]
    if len(idx) > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_samples), replace=False))
    x = state.iloc[idx].to_numpy(dtype=np.float32)
    x_next = state.iloc[idx + int(dynamics_step)].to_numpy(dtype=np.float32)
    pi = np.zeros((len(idx), len(RISK_SCALES)), dtype=np.float32)
    value = np.zeros(len(idx), dtype=np.float32)
    reward = np.zeros((len(idx), len(RISK_SCALES)), dtype=np.float32)
    full_cost = 2.0 * float(fee + slip)
    step_cost = float(fee + slip)
    for j, i in enumerate(idx):
        side = int(sides[i])
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(horizon)]
        side_ret = fut / base - 1.0 if side > 0 else base / np.maximum(fut, 1e-12) - 1.0
        step_px = float(close[int(i + int(dynamics_step))])
        step_ret = step_px / base - 1.0 if side > 0 else base / max(step_px, 1e-12) - 1.0
        vals: list[float] = []
        for a, scale in enumerate(RISK_SCALES):
            n = float(notionals[i]) * float(scale)
            if n <= 0.0:
                vals.append(0.0)
                reward[j, a] = 0.0
                continue
            path = side_ret * n
            run_min = np.minimum.accumulate(path)
            run_max = np.maximum.accumulate(path)
            adverse = np.maximum(0.0, -run_min)
            giveback = np.maximum(0.0, run_max - path)
            # Risk overlay target: maximize best achievable edge while punishing adverse path, giveback, and turnover.
            score_path = path - 0.70 * adverse - 0.18 * giveback - full_cost * n
            vals.append(float(np.max(score_path)))
            reward[j, a] = float(np.tanh((step_ret * n - step_cost * n) / 0.035))
        vals_np = np.asarray(vals, dtype=np.float64)
        z = vals_np / max(float(temperature), 1e-6)
        z = z - np.max(z)
        p = np.exp(z)
        p = p / max(float(p.sum()), 1e-12)
        pi[j] = p.astype(np.float32)
        value[j] = float(np.tanh(np.max(vals_np) / 0.08))
    meta = {
        "samples": int(len(idx)),
        "horizon": int(horizon),
        "dynamics_step": int(dynamics_step),
        "temperature": float(temperature),
        "target_argmax": {RISK_ACTIONS[i]: int((np.argmax(pi, axis=1) == i).sum()) for i in range(len(RISK_ACTIONS))} if len(pi) else {},
        "reward_mean": float(np.mean(reward)) if len(reward) else 0.0,
        "reward_std": float(np.std(reward)) if len(reward) else 0.0,
    }
    return x, x_next, pi, value, reward, meta


def _train_mz_risk(
    x: np.ndarray,
    x_next: np.ndarray,
    pi: np.ndarray,
    value: np.ndarray,
    reward: np.ndarray,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> tuple[RiskMuZeroNet, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(int(seed))
    xz, xnz, mean, std = _standardize_pair(x, x_next)
    ds = TensorDataset(
        torch.from_numpy(xz),
        torch.from_numpy(xnz),
        torch.from_numpy(np.asarray(pi, dtype=np.float32)),
        torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(-1)),
        torch.from_numpy(np.asarray(reward, dtype=np.float32)),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    net = RiskMuZeroNet(xz.shape[1], len(RISK_SCALES), hidden_dim=int(hidden_dim), latent_dim=int(latent_dim)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
    eye = torch.eye(len(RISK_SCALES), dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(int(epochs)):
        total = 0.0
        n = 0
        net.train()
        for xb, xnb, pib, vb, rb in loader:
            xb = xb.to(device)
            xnb = xnb.to(device)
            pib = pib.to(device)
            vb = vb.to(device)
            rb = rb.to(device)
            h, logits, pred_v = net.initial(xb)
            with torch.no_grad():
                h_target = F.normalize(net.representation(xnb), dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pib * logp).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(pred_v, vb)
            h_rep = h[:, None, :].expand(-1, len(RISK_SCALES), -1).reshape(-1, h.shape[-1])
            a_rep = eye[None, :, :].expand(len(xb), -1, -1).reshape(-1, len(RISK_SCALES))
            h_child, pred_r, _, child_v = net.recurrent(h_rep, a_rep)
            h_child = h_child.reshape(len(xb), len(RISK_SCALES), -1)
            pred_r = pred_r.reshape(len(xb), len(RISK_SCALES))
            child_v = child_v.reshape(len(xb), len(RISK_SCALES))
            dyn_loss = F.smooth_l1_loss(F.normalize(h_child, dim=-1), h_target[:, None, :].expand_as(h_child))
            reward_loss = F.smooth_l1_loss(pred_r, rb)
            child_value_loss = F.smooth_l1_loss(child_v, vb[:, None].expand(-1, len(RISK_SCALES)))
            entropy = -(torch.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
            loss = policy_loss + 0.65 * value_loss + 0.45 * reward_loss + 0.20 * dyn_loss + 0.10 * child_value_loss - 0.01 * entropy
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
    meta = {
        "samples": int(len(xz)),
        "state_dim": int(xz.shape[1]),
        "epochs": int(epochs),
        "hidden_dim": int(hidden_dim),
        "latent_dim": int(latent_dim),
        "final_loss": float(losses[-1]) if losses else None,
        "value_mean": float(np.mean(value)) if len(value) else 0.0,
        "value_std": float(np.std(value)) if len(value) else 0.0,
        "policy_entropy_mean": float(-(pi * np.log(np.maximum(pi, 1e-9))).sum(axis=1).mean()) if len(pi) else 0.0,
    }
    return net, mean, std, meta


def _predict_mz_risk(
    bundle: MZRiskBundle,
    x: np.ndarray,
    *,
    device: str,
    gamma: float,
    prior_weight: float,
    depth: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = (np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) - bundle.mean) / bundle.std
    scores_all: list[np.ndarray] = []
    probs_all: list[np.ndarray] = []
    values_all: list[np.ndarray] = []
    eye = torch.eye(len(RISK_SCALES), dtype=torch.float32, device=device)
    bundle.net.eval()
    with torch.no_grad():
        for s in range(0, len(arr), 8192):
            xb = torch.from_numpy(arr[s : s + 8192]).to(device)
            h, logits, value = bundle.net.initial(xb)
            probs = torch.softmax(logits, dim=-1)
            h_rep = h[:, None, :].expand(-1, len(RISK_SCALES), -1).reshape(-1, h.shape[-1])
            a_rep = eye[None, :, :].expand(len(xb), -1, -1).reshape(-1, len(RISK_SCALES))
            h1, r1, logits1, v1 = bundle.net.recurrent(h_rep, a_rep)
            h1 = h1.reshape(len(xb), len(RISK_SCALES), -1)
            r1 = r1.reshape(len(xb), len(RISK_SCALES))
            v1 = v1.reshape(len(xb), len(RISK_SCALES))
            score = r1 + float(gamma) * v1 + float(prior_weight) * torch.log(torch.clamp(probs, min=1e-8))
            if int(depth) >= 2:
                child_probs = torch.softmax(logits1.reshape(len(xb), len(RISK_SCALES), len(RISK_SCALES)), dim=-1)
                h1_rep = h1[:, :, None, :].expand(-1, -1, len(RISK_SCALES), -1).reshape(-1, h.shape[-1])
                a2_rep = eye[None, None, :, :].expand(len(xb), len(RISK_SCALES), -1, -1).reshape(-1, len(RISK_SCALES))
                _, r2, _, v2 = bundle.net.recurrent(h1_rep, a2_rep)
                r2 = r2.reshape(len(xb), len(RISK_SCALES), len(RISK_SCALES))
                v2 = v2.reshape(len(xb), len(RISK_SCALES), len(RISK_SCALES))
                child_score = r2 + float(gamma) * v2 + float(prior_weight) * torch.log(torch.clamp(child_probs, min=1e-8))
                score = r1 + float(gamma) * torch.max(child_score, dim=-1).values + float(prior_weight) * torch.log(torch.clamp(probs, min=1e-8))
            scores_all.append(score.detach().cpu().numpy().astype(np.float32))
            probs_all.append(probs.detach().cpu().numpy().astype(np.float32))
            values_all.append(value.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(scores_all), np.concatenate(probs_all), np.concatenate(values_all)


def _apply_scale(dec: pd.DataFrame, scale_idx: np.ndarray) -> pd.DataFrame:
    out = dec.copy()
    scale = RISK_SCALES[np.asarray(scale_idx, dtype=np.int64)]
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(active, notional * scale, 0.0)
    # Preserve existing leverage buckets; clamp notional so margin does not exceed 1.0.
    notional = np.minimum(notional, np.maximum(lev, 1e-12))
    flat = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "position_fraction"] = notional / np.maximum(lev, 1e-12)
    out.loc[flat, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[flat, "leverage"] = 1.0
    return out


def _run_bt(
    name: str,
    eval_df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    monthly: bool = False,
) -> dict[str, Any]:
    bt = backtest_no_limit_exit(
        eval_df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(fee),
        slip=float(slip),
        precomputed=precomputed,
    )
    row = {"name": name, "eval": _compact(bt)}
    if monthly:
        row["monthly"] = _monthly(eval_df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, precomputed, fee, slip)
    row["score"] = float(row["eval"]["pnl"] + 3.0 * row["eval"]["mdd"])
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AlphaZero/MuZero risk overlay on fixed MuZero-entry + AlphaZero-exit governor.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--az-risk-out", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--mz-risk-out", type=Path, default=DEFAULT_MZ_RISK_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--samples", type=int, default=90000)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.010)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    az_exit_model = _load_az_exit(args.az_model, device)
    mz_entry = _load_mz_entry(args.mz_entry_model, device)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)

    train_feat, train_dec, _, _, train_scores, train_probs, train_vals = _mz_entry_decisions(train_df, policy, entry_cfg, mz_entry, device=device)
    train_state = _state_frame(train_feat, train_dec, train_scores, train_probs, train_vals)
    x, x_next, pi, value, reward, label_meta = _risk_targets(
        train_df,
        train_state,
        train_dec,
        horizon=int(args.horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed),
    )

    az_net, az_mean, az_std, az_meta = _train_pv(
        x,
        pi,
        value,
        n_actions=len(RISK_SCALES),
        hidden_dim=int(args.hidden_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
    )
    mz_net, mz_mean, mz_std, mz_meta = _train_mz_risk(
        x,
        x_next,
        pi,
        value,
        reward,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed) + 11,
    )

    eval_feat, eval_dec, eval_close, eval_fill, eval_scores, eval_probs, eval_vals = _mz_entry_decisions(eval_df, policy, entry_cfg, mz_entry, device=device)
    eval_state = _state_frame(eval_feat, eval_dec, eval_scores, eval_probs, eval_vals)
    eval_x = eval_state.to_numpy(dtype=np.float32)
    base_pre = (eval_feat, eval_dec, eval_close, eval_fill)
    rows: list[dict[str, Any]] = [
        _run_bt("fixed_mz_entry_az_exit0.45", eval_df, policy, az_exit_model, entry_cfg, risk_cfg, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, base_pre, fee=args.fee, slip=args.slip),
    ]

    az_bundle = PVBundle(az_net, az_mean, az_std, list(eval_state.columns), RISK_ACTIONS)
    az_probs, az_values = _predict_pv(az_bundle, eval_x, device)
    for conf_floor in (0.0, 0.35, 0.50, 0.65):
        for value_floor in (-0.15, -0.02, 0.05, 0.15):
            idx = np.argmax(az_probs, axis=1)
            conf = az_probs.max(axis=1)
            idx = np.where((conf < conf_floor) | (az_values < value_floor), 3, idx)  # scale 1.0 fallback.
            dec = _apply_scale(eval_dec, idx)
            rows.append(
                _run_bt(
                    f"az_risk_cf{conf_floor:.2f}_vf{value_floor:.2f}",
                    eval_df,
                    policy,
                    az_exit_model,
                    entry_cfg,
                    risk_cfg,
                    {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
                    (eval_feat, dec, eval_close, eval_fill),
                    fee=args.fee,
                    slip=args.slip,
                )
            )

    mz_bundle = MZRiskBundle(mz_net, mz_mean, mz_std, list(eval_state.columns), RISK_ACTIONS)
    mz_cache: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for gamma in (0.55, 0.70, 0.85):
        for prior_w in (0.0, 0.08, 0.16):
            for depth in (1, 2):
                key = (gamma, prior_w, depth)
                mz_cache[key] = _predict_mz_risk(mz_bundle, eval_x, device=device, gamma=gamma, prior_weight=prior_w, depth=depth)
                scores, probs, vals = mz_cache[key]
                for score_floor in (-0.20, 0.0, 0.12):
                    for conf_floor in (0.0, 0.35):
                        idx = np.argmax(scores, axis=1)
                        conf = probs.max(axis=1)
                        idx = np.where((scores.max(axis=1) < score_floor) | (conf < conf_floor), 3, idx)
                        dec = _apply_scale(eval_dec, idx)
                        rows.append(
                            _run_bt(
                                f"mz_risk_g{gamma:.2f}_p{prior_w:.2f}_d{depth}_sf{score_floor:.2f}_cf{conf_floor:.2f}",
                                eval_df,
                                policy,
                                az_exit_model,
                                entry_cfg,
                                risk_cfg,
                                {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
                                (eval_feat, dec, eval_close, eval_fill),
                                fee=args.fee,
                                slip=args.slip,
                            )
                        )

    ranked_pnl = sorted(rows, key=lambda r: float(r["eval"]["pnl"]), reverse=True)
    ranked_score = sorted(rows, key=lambda r: float(r["score"]), reverse=True)
    chosen: list[dict[str, Any]] = []
    for row in ranked_pnl[:4] + ranked_score[:4] + [rows[0]]:
        if row["name"] not in {r["name"] for r in chosen}:
            chosen.append(row)

    def reconstruct(name: str) -> pd.DataFrame:
        if name == "fixed_mz_entry_az_exit0.45":
            return eval_dec
        if name.startswith("az_risk"):
            parts = name.split("_")
            conf_floor = float(parts[2].replace("cf", ""))
            value_floor = float(parts[3].replace("vf", ""))
            idx = np.argmax(az_probs, axis=1)
            conf = az_probs.max(axis=1)
            idx = np.where((conf < conf_floor) | (az_values < value_floor), 3, idx)
            return _apply_scale(eval_dec, idx)
        parts = name.split("_")
        gamma = float(parts[2].replace("g", ""))
        prior_w = float(parts[3].replace("p", ""))
        depth = int(parts[4].replace("d", ""))
        score_floor = float(parts[5].replace("sf", ""))
        conf_floor = float(parts[6].replace("cf", ""))
        scores, probs, _ = mz_cache[(gamma, prior_w, depth)]
        idx = np.argmax(scores, axis=1)
        conf = probs.max(axis=1)
        idx = np.where((scores.max(axis=1) < score_floor) | (conf < conf_floor), 3, idx)
        return _apply_scale(eval_dec, idx)

    selected_detail = []
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for row in chosen:
        dec = reconstruct(row["name"])
        selected_detail.append(
            _run_bt(row["name"], eval_df, policy, az_exit_model, entry_cfg, risk_cfg, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, (eval_feat, dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True)
        )
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in chosen:
            dec = reconstruct(row["name"])
            cost_stress[f"cost_{mult:g}x"].append(
                _run_bt(row["name"], eval_df, policy, az_exit_model, entry_cfg, risk_cfg, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, (eval_feat, dec, eval_close, eval_fill), fee=args.fee * mult, slip=args.slip * mult)
            )

    args.az_risk_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "alphazero_style_risk_overlay",
            "state_dict": az_net.state_dict(),
            "mean": az_mean.astype(np.float32),
            "std": az_std.astype(np.float32),
            "feature_cols": list(eval_state.columns),
            "actions": list(RISK_ACTIONS),
            "scales": RISK_SCALES.astype(np.float32),
            "train_meta": az_meta,
            "label_meta": label_meta,
        },
        args.az_risk_out,
    )
    args.mz_risk_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "muzero_style_risk_overlay",
            "state_dict": mz_net.state_dict(),
            "mean": mz_mean.astype(np.float32),
            "std": mz_std.astype(np.float32),
            "feature_cols": list(eval_state.columns),
            "actions": list(RISK_ACTIONS),
            "scales": RISK_SCALES.astype(np.float32),
            "train_meta": mz_meta,
            "label_meta": label_meta,
        },
        args.mz_risk_out,
    )
    report = {
        "type": "zero_style_risk_overlay_2026",
        "note": "Fixed best MuZero entry + AlphaZero exit0.45 is preserved. AlphaZero/MuZero overlays only modify risk notional scale.",
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "mz_entry_model": str(args.mz_entry_model),
        "az_exit_model": str(args.az_model),
        "az_risk_out": str(args.az_risk_out),
        "mz_risk_out": str(args.mz_risk_out),
        "label_meta": label_meta,
        "train_meta": {"alphazero_risk": az_meta, "muzero_risk": mz_meta},
        "az_policy_distribution": {RISK_ACTIONS[i]: int((np.argmax(az_probs, axis=1) == i).sum()) for i in range(len(RISK_ACTIONS))},
        "ranked_by_pnl": ranked_pnl[:40],
        "ranked_by_score": ranked_score[:40],
        "selected_detail": selected_detail,
        "cost_stress": cost_stress,
        "decision": {
            "best_pnl_name": ranked_pnl[0]["name"],
            "best_pnl": ranked_pnl[0]["eval"]["pnl"],
            "best_score_name": ranked_score[0]["name"],
            "best_score": ranked_score[0]["score"],
            "fixed_combo_pnl": rows[0]["eval"]["pnl"],
            "fixed_combo_mdd": rows[0]["eval"]["mdd"],
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "az_model": str(args.az_risk_out), "mz_model": str(args.mz_risk_out), "decision": report["decision"], "top": ranked_pnl[:8]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
