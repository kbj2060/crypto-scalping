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
    AZExitModel,
    DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL,
    EXIT_ACTIONS,
    PolicyValueNet,
    PVBundle,
    _monthly,
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
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    MODEL_COLS,
    _base_frame,
    _compact,
    backtest_no_limit_exit,
)


ENTRY_ACTIONS = ("block", "half", "keep", "boost")
ACTION_SCALES = np.asarray([0.0, 0.5, 1.0, 1.25], dtype=np.float64)
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/muzero_style/mz_latent_governor.pt"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/muzero_style_governor_2026.json"


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


class MuZeroNet(nn.Module):
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
class MZBundle:
    net: MuZeroNet
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]
    actions: tuple[str, ...]


def _make_targets(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    feat: pd.DataFrame,
    *,
    search_horizon: int,
    dynamics_step: int,
    fee: float,
    slip: float,
    temperature: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    idx = np.flatnonzero((actions != ACTION_CASH) & (sides != 0) & (notionals > 0.0))
    idx = idx[idx < len(df) - max(int(search_horizon), int(dynamics_step)) - 3]
    if len(idx) > int(max_samples):
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(idx, size=int(max_samples), replace=False))

    x = feat.iloc[idx].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    x_next = feat.iloc[idx + int(dynamics_step)].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pi = np.zeros((len(idx), len(ENTRY_ACTIONS)), dtype=np.float32)
    value = np.zeros(len(idx), dtype=np.float32)
    step_reward = np.zeros((len(idx), len(ENTRY_ACTIONS)), dtype=np.float32)
    full_cost = 2.0 * float(fee + slip)
    step_cost = float(fee + slip)
    for j, i in enumerate(idx):
        side = int(sides[i])
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(search_horizon)]
        side_ret = fut / base - 1.0 if side > 0 else base / np.maximum(fut, 1e-12) - 1.0
        step_px = float(close[int(i + int(dynamics_step))])
        one_step_ret = step_px / base - 1.0 if side > 0 else base / max(step_px, 1e-12) - 1.0
        vals: list[float] = []
        for a, scale in enumerate(ACTION_SCALES):
            n = float(notionals[i]) * float(scale)
            if n <= 0.0:
                vals.append(0.0)
                step_reward[j, a] = 0.0
                continue
            path = side_ret * n
            run_min = np.minimum.accumulate(path)
            score_path = path - 0.55 * np.maximum(0.0, -run_min) - full_cost * n
            vals.append(float(np.max(score_path)))
            step_reward[j, a] = float(np.tanh((one_step_ret * n - step_cost * n) / 0.035))
        vals_np = np.asarray(vals, dtype=np.float64)
        z = vals_np / max(float(temperature), 1e-6)
        z = z - np.max(z)
        p = np.exp(z)
        p = p / max(float(p.sum()), 1e-12)
        pi[j] = p.astype(np.float32)
        value[j] = float(np.tanh(np.max(vals_np) / 0.08))

    meta = {
        "samples": int(len(idx)),
        "search_horizon": int(search_horizon),
        "dynamics_step": int(dynamics_step),
        "temperature": float(temperature),
        "target_policy_argmax": {ENTRY_ACTIONS[i]: int((np.argmax(pi, axis=1) == i).sum()) for i in range(len(ENTRY_ACTIONS))} if len(pi) else {},
        "reward_mean": float(np.mean(step_reward)) if len(step_reward) else 0.0,
        "reward_std": float(np.std(step_reward)) if len(step_reward) else 0.0,
    }
    return x, x_next, pi, value, step_reward, meta


def _train_muzero(
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
) -> tuple[MuZeroNet, np.ndarray, np.ndarray, dict[str, Any]]:
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
    net = MuZeroNet(xz.shape[1], len(ENTRY_ACTIONS), hidden_dim=int(hidden_dim), latent_dim=int(latent_dim)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
    eye = torch.eye(len(ENTRY_ACTIONS), dtype=torch.float32, device=device)
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
                h_target = net.representation(xnb)
                h_target = F.normalize(h_target, dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pib * logp).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(pred_v, vb)

            h_rep = h[:, None, :].expand(-1, len(ENTRY_ACTIONS), -1).reshape(-1, h.shape[-1])
            a_rep = eye[None, :, :].expand(len(xb), -1, -1).reshape(-1, len(ENTRY_ACTIONS))
            h_pred, pred_r, child_logits, child_v = net.recurrent(h_rep, a_rep)
            h_pred_norm = F.normalize(h_pred, dim=-1).reshape(len(xb), len(ENTRY_ACTIONS), -1)
            dyn_loss = F.smooth_l1_loss(h_pred_norm, h_target[:, None, :].expand_as(h_pred_norm))
            reward_loss = F.smooth_l1_loss(pred_r.reshape(len(xb), len(ENTRY_ACTIONS)), rb)
            child_v_loss = F.smooth_l1_loss(child_v.reshape(len(xb), len(ENTRY_ACTIONS)), vb[:, None].expand(-1, len(ENTRY_ACTIONS)))
            entropy = -(torch.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
            loss = policy_loss + 0.65 * value_loss + 0.45 * reward_loss + 0.20 * dyn_loss + 0.10 * child_v_loss - 0.01 * entropy
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


def _plan_scores(
    bundle: MZBundle,
    x: np.ndarray,
    *,
    device: str,
    gamma: float,
    prior_weight: float,
    depth: int,
    batch: int = 8192,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = (np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0) - bundle.mean) / bundle.std
    all_scores: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []
    all_values: list[np.ndarray] = []
    eye = torch.eye(len(bundle.actions), dtype=torch.float32, device=device)
    bundle.net.eval()
    with torch.no_grad():
        for s in range(0, len(arr), int(batch)):
            xb = torch.from_numpy(arr[s : s + int(batch)]).to(device)
            h, logits, value = bundle.net.initial(xb)
            probs = torch.softmax(logits, dim=-1)
            h_rep = h[:, None, :].expand(-1, len(bundle.actions), -1).reshape(-1, h.shape[-1])
            a_rep = eye[None, :, :].expand(len(xb), -1, -1).reshape(-1, len(bundle.actions))
            h1, r1, logits1, v1 = bundle.net.recurrent(h_rep, a_rep)
            h1 = h1.reshape(len(xb), len(bundle.actions), -1)
            r1 = r1.reshape(len(xb), len(bundle.actions))
            v1 = v1.reshape(len(xb), len(bundle.actions))
            score = r1 + float(gamma) * v1 + float(prior_weight) * torch.log(torch.clamp(probs, min=1e-8))
            if int(depth) >= 2:
                child_probs = torch.softmax(logits1.reshape(len(xb), len(bundle.actions), len(bundle.actions)), dim=-1)
                h1_rep = h1[:, :, None, :].expand(-1, -1, len(bundle.actions), -1).reshape(-1, h.shape[-1])
                a2_rep = eye[None, None, :, :].expand(len(xb), len(bundle.actions), -1, -1).reshape(-1, len(bundle.actions))
                _, r2, _, v2 = bundle.net.recurrent(h1_rep, a2_rep)
                r2 = r2.reshape(len(xb), len(bundle.actions), len(bundle.actions))
                v2 = v2.reshape(len(xb), len(bundle.actions), len(bundle.actions))
                child_score = r2 + float(gamma) * v2 + float(prior_weight) * torch.log(torch.clamp(child_probs, min=1e-8))
                score = r1 + float(gamma) * torch.max(child_score, dim=-1).values + float(prior_weight) * torch.log(torch.clamp(probs, min=1e-8))
            all_scores.append(score.detach().cpu().numpy().astype(np.float32))
            all_probs.append(probs.detach().cpu().numpy().astype(np.float32))
            all_values.append(value.detach().cpu().numpy().astype(np.float32))
    return (
        np.concatenate(all_scores) if all_scores else np.zeros((0, len(bundle.actions)), dtype=np.float32),
        np.concatenate(all_probs) if all_probs else np.zeros((0, len(bundle.actions)), dtype=np.float32),
        np.concatenate(all_values) if all_values else np.zeros(0, dtype=np.float32),
    )


def _planned_decisions(
    base_dec: pd.DataFrame,
    scores: np.ndarray,
    root_probs: np.ndarray,
    values: np.ndarray,
    *,
    score_floor: float,
    confidence_floor: float,
    value_floor: float,
) -> pd.DataFrame:
    out = base_dec.copy()
    best = np.argmax(scores, axis=1)
    confidence = root_probs.max(axis=1)
    max_score = scores.max(axis=1)
    block = (max_score < float(score_floor)) | (confidence < float(confidence_floor)) | (np.asarray(values) < float(value_floor))
    best = np.where(block, 0, best)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    lev = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    notional = np.where(active, notional * ACTION_SCALES[best], 0.0)
    flat = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "position_fraction"] = notional / np.maximum(lev, 1e-12)
    out.loc[flat, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[flat, "leverage"] = 1.0
    return out


def _load_az_exit(path: Path, device: str) -> Any | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    sec = payload["exit"]
    hidden = int(sec["state_dict"]["trunk.0.weight"].shape[0])
    net = PolicyValueNet(len(sec["feature_cols"]), len(EXIT_ACTIONS), hidden_dim=hidden).to(device)
    net.load_state_dict(sec["state_dict"])
    bundle = PVBundle(net, np.asarray(sec["mean"], dtype=np.float32), np.asarray(sec["std"], dtype=np.float32), list(MODEL_COLS), EXIT_ACTIONS)
    return AZExitModel(bundle, device)


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
    p = argparse.ArgumentParser(description="MuZero-style latent dynamics planner for current HF no-limit governor.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1.3e-3)
    p.add_argument("--search-horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--samples", type=int, default=90000)
    p.add_argument("--temperature", type=float, default=0.012)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    policy = joblib.load(args.policy)
    exit_bundle = joblib.load(args.exit_bundle)
    base_exit_model = exit_bundle["model"] if isinstance(exit_bundle, dict) and "model" in exit_bundle else exit_bundle
    az_exit_model = _load_az_exit(args.az_model, device)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    train_df = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_pre = _base_frame(train_df, policy, entry_cfg)
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    train_feat, train_dec, _, _ = train_pre
    eval_feat, eval_dec, eval_close, eval_fill = eval_pre

    x, x_next, pi, value, reward, label_meta = _make_targets(
        train_df,
        train_dec,
        train_feat.reindex(columns=FEATURE_COLS),
        search_horizon=int(args.search_horizon),
        dynamics_step=int(args.dynamics_step),
        fee=float(args.fee),
        slip=float(args.slip),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed),
    )
    net, mean, std, train_meta = _train_muzero(
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
        seed=int(args.seed),
    )
    bundle = MZBundle(net, mean, std, list(FEATURE_COLS), ENTRY_ACTIONS)
    eval_x = eval_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)

    rows: list[dict[str, Any]] = []
    rows.append(_run_bt("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, eval_pre, fee=args.fee, slip=args.slip))
    plan_cache: dict[tuple[float, float, int], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for gamma in (0.55, 0.70, 0.85):
        for prior_w in (0.0, 0.08, 0.16):
            for depth in (1, 2):
                plan_cache[(gamma, prior_w, depth)] = _plan_scores(bundle, eval_x, device=device, gamma=gamma, prior_weight=prior_w, depth=depth)
                scores, probs, vals = plan_cache[(gamma, prior_w, depth)]
                for score_floor in (-0.35, -0.15, 0.00, 0.12, 0.24):
                    for conf_floor in (0.0, 0.35, 0.50):
                        for value_floor in (-0.20, -0.05, 0.05):
                            dec = _planned_decisions(
                                eval_dec,
                                scores,
                                probs,
                                vals,
                                score_floor=score_floor,
                                confidence_floor=conf_floor,
                                value_floor=value_floor,
                            )
                            tag = f"mz_g{gamma:.2f}_p{prior_w:.2f}_d{depth}_sf{score_floor:.2f}_cf{conf_floor:.2f}_vf{value_floor:.2f}"
                            rows.append(_run_bt(tag, eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, (eval_feat, dec, eval_close, eval_fill), fee=args.fee, slip=args.slip))
                            if az_exit_model is not None and score_floor in (-0.15, 0.00, 0.12) and conf_floor in (0.0, 0.35) and value_floor in (-0.05, 0.05):
                                rows.append(
                                    _run_bt(
                                        f"{tag}_azexit0.45",
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
    for row in ranked_pnl[:3] + ranked_score[:3] + [next(r for r in rows if r["name"] == "baseline_hf_no_limit")]:
        if row["name"] not in {r["name"] for r in chosen}:
            chosen.append(row)

    def reconstruct(name: str) -> tuple[Any, dict[str, Any], pd.DataFrame]:
        if name == "baseline_hf_no_limit":
            return base_exit_model, exit_cfg, eval_dec
        use_az = name.endswith("_azexit0.45")
        base_name = name.removesuffix("_azexit0.45")
        parts = base_name.split("_")
        gamma = float(parts[1].replace("g", ""))
        prior_w = float(parts[2].replace("p", ""))
        depth = int(parts[3].replace("d", ""))
        score_floor = float(parts[4].replace("sf", ""))
        conf_floor = float(parts[5].replace("cf", ""))
        value_floor = float(parts[6].replace("vf", ""))
        scores, probs, vals = plan_cache[(gamma, prior_w, depth)]
        dec = _planned_decisions(eval_dec, scores, probs, vals, score_floor=score_floor, confidence_floor=conf_floor, value_floor=value_floor)
        if use_az and az_exit_model is not None:
            return az_exit_model, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, dec
        return base_exit_model, exit_cfg, dec

    selected_detail = []
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for row in chosen:
        ex_model, cfg, dec = reconstruct(row["name"])
        selected_detail.append(_run_bt(row["name"], eval_df, policy, ex_model, entry_cfg, risk_cfg, cfg, (eval_feat, dec, eval_close, eval_fill), fee=args.fee, slip=args.slip, monthly=True))
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in chosen:
            ex_model, cfg, dec = reconstruct(row["name"])
            cost_stress[f"cost_{mult:g}x"].append(_run_bt(row["name"], eval_df, policy, ex_model, entry_cfg, risk_cfg, cfg, (eval_feat, dec, eval_close, eval_fill), fee=args.fee * mult, slip=args.slip * mult))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "muzero_style_latent_governor",
            "state_dict": net.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "feature_cols": list(FEATURE_COLS),
            "actions": list(ENTRY_ACTIONS),
            "train_meta": train_meta,
            "label_meta": label_meta,
        },
        args.model_out,
    )
    first_scores, first_probs, first_values = next(iter(plan_cache.values()))
    report = {
        "type": "muzero_style_governor_2026",
        "note": "MuZero-inspired adaptation: learned representation/dynamics/reward/policy/value on 2025 market replay, then causal depth-1/2 latent planning on 2026 current features. There is no true self-play because market path is exogenous.",
        "model_out": str(args.model_out),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "label_meta": label_meta,
        "train_meta": train_meta,
        "plan_score_quantiles_first_cache": np.quantile(first_scores.max(axis=1), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(6).tolist(),
        "root_conf_quantiles_first_cache": np.quantile(first_probs.max(axis=1), [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(6).tolist(),
        "root_value_quantiles_first_cache": np.quantile(first_values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0]).round(6).tolist(),
        "ranked_by_pnl": ranked_pnl[:40],
        "ranked_by_score": ranked_score[:40],
        "selected_detail": selected_detail,
        "cost_stress": cost_stress,
        "decision": {
            "best_pnl_name": ranked_pnl[0]["name"],
            "best_pnl": ranked_pnl[0]["eval"]["pnl"],
            "best_score_name": ranked_score[0]["name"],
            "best_score": ranked_score[0]["score"],
            "baseline_pnl": next(r for r in rows if r["name"] == "baseline_hf_no_limit")["eval"]["pnl"],
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "model": str(args.model_out), "decision": report["decision"], "top_pnl": ranked_pnl[:6], "top_score": ranked_score[:6]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
