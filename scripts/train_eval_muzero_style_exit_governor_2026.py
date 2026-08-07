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
    _feature_row,
    _fill_price,
    _future_raw_from_entry,
    _raw_from_entry,
    backtest_no_limit_exit,
)
from scripts.train_eval_muzero_style_governor_2026 import (  # noqa: E402
    DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL,
    ENTRY_ACTIONS,
    MZBundle,
    MuZeroNet,
    _planned_decisions,
    _plan_scores,
)


EXIT_PLANNER_ACTIONS = ("hold", "exit")
DEFAULT_MODEL_OUT = ROOT / "data/ensemble/supervised/muzero_style/mz_exit_governor.pt"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/muzero_style_exit_governor_2026.json"


def _standardize_pair(x: np.ndarray, x_next: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float32)
    x_next = np.asarray(x_next, dtype=np.float32)
    mean = np.nanmean(x, axis=0).astype(np.float32)
    std = np.nanstd(x, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    zn = (np.nan_to_num(x_next, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return z.astype(np.float32), zn.astype(np.float32), mean, std


def _layer_norm_np(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=1, keepdims=True)
    return (x - mean) / np.sqrt(var + 1e-5) * weight + bias


def _silu_np(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


class MuZeroExitNet(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 192, latent_dim: int = 128):
        super().__init__()
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
        self.policy = nn.Linear(int(hidden_dim), 2)
        self.value = nn.Linear(int(hidden_dim), 1)
        self.dynamics = nn.Sequential(
            nn.Linear(int(latent_dim) + 2, int(hidden_dim)),
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
class MZExitBundle:
    net: MuZeroExitNet
    mean: np.ndarray
    std: np.ndarray
    feature_cols: list[str]


class MZExitModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, bundle: MZExitBundle, *, gamma: float = 0.70, prior_weight: float = 0.08, temperature: float = 0.40):
        self.bundle = bundle
        self.gamma = float(gamma)
        self.prior_weight = float(prior_weight)
        self.temperature = float(temperature)
        sd = {k: v.detach().cpu().numpy().astype(np.float32) for k, v in bundle.net.state_dict().items()}
        self._sd = sd

    def _rep(self, arr: np.ndarray) -> np.ndarray:
        sd = self._sd
        h = arr @ sd["representation.0.weight"].T + sd["representation.0.bias"]
        h = _silu_np(_layer_norm_np(h, sd["representation.1.weight"], sd["representation.1.bias"]))
        h = h @ sd["representation.3.weight"].T + sd["representation.3.bias"]
        return _silu_np(_layer_norm_np(h, sd["representation.4.weight"], sd["representation.4.bias"]))

    def _pred(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        sd = self._sd
        z = h @ sd["prediction.0.weight"].T + sd["prediction.0.bias"]
        z = _silu_np(_layer_norm_np(z, sd["prediction.1.weight"], sd["prediction.1.bias"]))
        logits = z @ sd["policy.weight"].T + sd["policy.bias"]
        value = np.tanh((z @ sd["value.weight"].T + sd["value.bias"]).reshape(-1))
        return logits, value

    def _dyn(self, h: np.ndarray, action_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sd = self._sd
        a = np.zeros((len(h), 2), dtype=np.float32)
        a[:, int(action_idx)] = 1.0
        x = np.concatenate([h, a], axis=1)
        z = x @ sd["dynamics.0.weight"].T + sd["dynamics.0.bias"]
        z = _silu_np(_layer_norm_np(z, sd["dynamics.1.weight"], sd["dynamics.1.bias"]))
        z = z @ sd["dynamics.3.weight"].T + sd["dynamics.3.bias"]
        h_next = _silu_np(_layer_norm_np(z, sd["dynamics.4.weight"], sd["dynamics.4.bias"]))
        reward = np.tanh((h_next @ sd["reward.weight"].T + sd["reward.bias"]).reshape(-1))
        _, value = self._pred(h_next)
        return h_next, reward, value

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        arr = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) - self.bundle.mean) / self.bundle.std
        h = self._rep(arr)
        logits, _ = self._pred(h)
        logits = logits - logits.max(axis=1, keepdims=True)
        root_probs = np.exp(logits)
        root_probs = root_probs / np.maximum(root_probs.sum(axis=1, keepdims=True), 1e-12)
        _, r_hold, v_hold = self._dyn(h, 0)
        _, r_exit, v_exit = self._dyn(h, 1)
        scores = np.stack(
            [
                r_hold + self.gamma * v_hold + self.prior_weight * np.log(np.maximum(root_probs[:, 0], 1e-8)),
                r_exit + self.gamma * v_exit + self.prior_weight * np.log(np.maximum(root_probs[:, 1], 1e-8)),
            ],
            axis=1,
        )
        scores = scores / max(self.temperature, 1e-6)
        scores = scores - scores.max(axis=1, keepdims=True)
        p = np.exp(scores)
        return (p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)).astype(np.float64)


def _collect_exit_muzero_targets(
    df: pd.DataFrame,
    policy: dict[str, Any],
    entry_cfg: dict[str, Any],
    *,
    fee: float,
    slip: float,
    entry_stride: int,
    min_age: int,
    max_age: int,
    age_stride: int,
    future_horizon: int,
    dynamics_step: int,
    adverse_penalty: float,
    temperature: float,
    max_samples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    base_feat, decisions, _, fill_px = _base_frame(df, policy, entry_cfg)
    actions = decisions["action"].astype(int).to_numpy()
    sides = decisions["side"].astype(int).to_numpy()
    notionals = pd.to_numeric(decisions["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverages = pd.to_numeric(decisions["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    qualities = pd.to_numeric(decisions["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    confs = pd.to_numeric(decisions["confidence"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)

    rows: list[dict[str, float]] = []
    next_rows: list[dict[str, float]] = []
    pi_rows: list[np.ndarray] = []
    value_rows: list[float] = []
    reward_rows: list[np.ndarray] = []
    child_value_rows: list[np.ndarray] = []
    active = np.flatnonzero((actions != 0) & (sides != 0) & (notionals > 0.0))
    active = active[active < len(df) - max(int(max_age), int(future_horizon), int(dynamics_step)) - 4]
    active = active[:: max(1, int(entry_stride))]
    for entry_i in active:
        side = int(sides[int(entry_i)])
        notional = float(notionals[int(entry_i)])
        if notional <= 1e-8:
            continue
        entry_price = _fill_price(fill_px, int(entry_i) + 1, side, slip, entry=True)
        peak_unrealized = 0.0
        for age in range(int(min_age), int(max_age) + 1, max(1, int(age_stride))):
            i = int(entry_i) + int(age)
            if i >= len(df) - int(dynamics_step) - 4:
                break
            raw_now = _raw_from_entry(fill_px, i + 1, side, entry_price, slip)
            immediate = raw_now * notional - float(fee) * notional
            unreal = raw_now * notional
            peak_unrealized = max(peak_unrealized, unreal)
            end = min(len(df) - 2, i + int(future_horizon))
            if end <= i + 1:
                continue
            future_raw = _future_raw_from_entry(fill_px, i + 2, end + 1, side, entry_price, slip)
            if future_raw.size == 0:
                continue
            future_net = future_raw * notional - float(fee) * notional
            future_best = float(np.max(future_net))
            future_worst = float(np.min(future_net))
            hold_score = future_best - float(adverse_penalty) * max(0.0, immediate - future_worst)
            exit_score = float(immediate)
            scores = np.asarray([hold_score, exit_score], dtype=np.float64)
            z = scores / max(float(temperature), 1e-6)
            z = z - np.max(z)
            p = np.exp(z)
            p = p / max(float(p.sum()), 1e-12)

            next_i = min(len(df) - 3, i + int(dynamics_step))
            raw_next = _raw_from_entry(fill_px, next_i + 1, side, entry_price, slip)
            next_unreal = raw_next * notional
            between_raw = _future_raw_from_entry(fill_px, i + 2, next_i + 1, side, entry_price, slip)
            if between_raw.size:
                peak_next = max(peak_unrealized, float(np.max(between_raw * notional)))
            else:
                peak_next = max(peak_unrealized, next_unreal)

            rows.append(
                _feature_row(
                    base_feat,
                    decisions,
                    i=i,
                    side=side,
                    age=int(age),
                    unrealized=unreal,
                    peak_unrealized=peak_unrealized,
                    notional=notional,
                    leverage=float(leverages[int(entry_i)]),
                    entry_quality=float(qualities[int(entry_i)]),
                    entry_confidence=float(confs[int(entry_i)]),
                )
            )
            next_rows.append(
                _feature_row(
                    base_feat,
                    decisions,
                    i=next_i,
                    side=side,
                    age=int(age) + int(dynamics_step),
                    unrealized=next_unreal,
                    peak_unrealized=peak_next,
                    notional=notional,
                    leverage=float(leverages[int(entry_i)]),
                    entry_quality=float(qualities[int(entry_i)]),
                    entry_confidence=float(confs[int(entry_i)]),
                )
            )
            hold_step = (next_unreal - unreal) - float(slip) * notional
            pi_rows.append(p.astype(np.float32))
            value_rows.append(float(np.tanh(np.max(scores) / 0.08)))
            reward_rows.append(np.asarray([np.tanh(hold_step / 0.035), np.tanh(exit_score / 0.035)], dtype=np.float32))
            child_value_rows.append(np.asarray([np.tanh(hold_score / 0.08), 0.0], dtype=np.float32))

    if not rows:
        raise RuntimeError("no MuZero exit samples collected")
    rng = np.random.default_rng(int(seed))
    take = np.arange(len(rows))
    if len(take) > int(max_samples):
        take = np.sort(rng.choice(take, size=int(max_samples), replace=False))
    x = pd.DataFrame(rows).reindex(columns=MODEL_COLS).iloc[take].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    x_next = pd.DataFrame(next_rows).reindex(columns=MODEL_COLS).iloc[take].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    pi = np.asarray(pi_rows, dtype=np.float32)[take]
    value = np.asarray(value_rows, dtype=np.float32)[take]
    reward = np.asarray(reward_rows, dtype=np.float32)[take]
    child_value = np.asarray(child_value_rows, dtype=np.float32)[take]
    meta = {
        "samples": int(len(x)),
        "raw_samples": int(len(rows)),
        "future_horizon": int(future_horizon),
        "dynamics_step": int(dynamics_step),
        "temperature": float(temperature),
        "adverse_penalty": float(adverse_penalty),
        "target_argmax": {EXIT_PLANNER_ACTIONS[i]: int((np.argmax(pi, axis=1) == i).sum()) for i in range(2)},
        "reward_mean": float(np.mean(reward)),
        "reward_std": float(np.std(reward)),
    }
    return x, x_next, pi, value, reward, child_value, meta


def _train_exit_muzero(
    x: np.ndarray,
    x_next: np.ndarray,
    pi: np.ndarray,
    value: np.ndarray,
    reward: np.ndarray,
    child_value: np.ndarray,
    *,
    hidden_dim: int,
    latent_dim: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
    seed: int,
) -> tuple[MuZeroExitNet, np.ndarray, np.ndarray, dict[str, Any]]:
    torch.manual_seed(int(seed))
    xz, xnz, mean, std = _standardize_pair(x, x_next)
    ds = TensorDataset(
        torch.from_numpy(xz),
        torch.from_numpy(xnz),
        torch.from_numpy(np.asarray(pi, dtype=np.float32)),
        torch.from_numpy(np.asarray(value, dtype=np.float32).reshape(-1)),
        torch.from_numpy(np.asarray(reward, dtype=np.float32)),
        torch.from_numpy(np.asarray(child_value, dtype=np.float32)),
    )
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    net = MuZeroExitNet(xz.shape[1], hidden_dim=int(hidden_dim), latent_dim=int(latent_dim)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(lr), weight_decay=1e-4)
    eye = torch.eye(2, dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(int(epochs)):
        total = 0.0
        n = 0
        net.train()
        for xb, xnb, pib, vb, rb, cvb in loader:
            xb = xb.to(device)
            xnb = xnb.to(device)
            pib = pib.to(device)
            vb = vb.to(device)
            rb = rb.to(device)
            cvb = cvb.to(device)
            h, logits, pred_v = net.initial(xb)
            with torch.no_grad():
                h_next_target = F.normalize(net.representation(xnb), dim=-1)
            logp = F.log_softmax(logits, dim=-1)
            policy_loss = -(pib * logp).sum(dim=-1).mean()
            value_loss = F.smooth_l1_loss(pred_v, vb)
            h_rep = h[:, None, :].expand(-1, 2, -1).reshape(-1, h.shape[-1])
            a_rep = eye[None, :, :].expand(len(xb), -1, -1).reshape(-1, 2)
            h_child, pred_r, _, pred_cv = net.recurrent(h_rep, a_rep)
            h_child = h_child.reshape(len(xb), 2, -1)
            pred_r = pred_r.reshape(len(xb), 2)
            pred_cv = pred_cv.reshape(len(xb), 2)
            hold_dyn_loss = F.smooth_l1_loss(F.normalize(h_child[:, 0, :], dim=-1), h_next_target)
            exit_terminal_loss = torch.mean(h_child[:, 1, :] ** 2) * 0.02
            reward_loss = F.smooth_l1_loss(pred_r, rb)
            child_value_loss = F.smooth_l1_loss(pred_cv, cvb)
            entropy = -(torch.softmax(logits, dim=-1) * logp).sum(dim=-1).mean()
            loss = policy_loss + 0.65 * value_loss + 0.55 * reward_loss + 0.20 * hold_dyn_loss + 0.15 * child_value_loss + exit_terminal_loss - 0.01 * entropy
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
        "value_mean": float(np.mean(value)),
        "value_std": float(np.std(value)),
        "policy_entropy_mean": float(-(pi * np.log(np.maximum(pi, 1e-9))).sum(axis=1).mean()),
    }
    return net, mean, std, meta


def _load_mz_entry(path: Path, device: str) -> MZBundle:
    payload = torch.load(path, map_location=device, weights_only=False)
    hidden = int(payload["state_dict"]["representation.0.weight"].shape[0])
    latent = int(payload["state_dict"]["representation.3.weight"].shape[0])
    net = MuZeroNet(len(payload["feature_cols"]), len(ENTRY_ACTIONS), hidden_dim=hidden, latent_dim=latent).to(device)
    net.load_state_dict(payload["state_dict"])
    return MZBundle(net, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), list(payload["feature_cols"]), ENTRY_ACTIONS)


def _load_az_exit(path: Path, device: str) -> Any | None:
    if not path.exists():
        return None
    payload = torch.load(path, map_location=device, weights_only=False)
    sec = payload["exit"]
    hidden = int(sec["state_dict"]["trunk.0.weight"].shape[0])
    net = PolicyValueNet(len(sec["feature_cols"]), len(EXIT_ACTIONS), hidden_dim=hidden).to(device)
    net.load_state_dict(sec["state_dict"])
    return AZExitModel(PVBundle(net, np.asarray(sec["mean"], dtype=np.float32), np.asarray(sec["std"], dtype=np.float32), list(MODEL_COLS), EXIT_ACTIONS), device)


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
    p = argparse.ArgumentParser(description="MuZero-style exit planner for current best MuZero entry governor.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_OUT)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=16)
    p.add_argument("--hidden-dim", type=int, default=192)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1.2e-3)
    p.add_argument("--samples", type=int, default=90000)
    p.add_argument("--future-horizon", type=int, default=144)
    p.add_argument("--dynamics-step", type=int, default=12)
    p.add_argument("--temperature", type=float, default=0.010)
    p.add_argument("--adverse-penalty", type=float, default=0.65)
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
    eval_pre = _base_frame(eval_df, policy, entry_cfg)
    eval_feat, eval_dec, eval_close, eval_fill = eval_pre

    x, x_next, pi, value, reward, child_value, label_meta = _collect_exit_muzero_targets(
        train_df,
        policy,
        entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=24,
        min_age=3,
        max_age=288,
        age_stride=12,
        future_horizon=int(args.future_horizon),
        dynamics_step=int(args.dynamics_step),
        adverse_penalty=float(args.adverse_penalty),
        temperature=float(args.temperature),
        max_samples=int(args.samples),
        seed=int(args.seed),
    )
    net, mean, std, train_meta = _train_exit_muzero(
        x,
        x_next,
        pi,
        value,
        reward,
        child_value,
        hidden_dim=int(args.hidden_dim),
        latent_dim=int(args.latent_dim),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        device=device,
        seed=int(args.seed),
    )
    bundle = MZExitBundle(net, mean, std, list(MODEL_COLS))

    mz_entry = _load_mz_entry(args.mz_entry_model, device)
    entry_x = eval_feat.reindex(columns=mz_entry.feature_cols).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    scores, root_probs, root_values = _plan_scores(mz_entry, entry_x, device=device, gamma=0.70, prior_weight=0.16, depth=1)
    best_entry_dec = _planned_decisions(
        eval_dec,
        scores,
        root_probs,
        root_values,
        score_floor=0.00,
        confidence_floor=0.00,
        value_floor=-0.05,
    )
    best_entry_pre = (eval_feat, best_entry_dec, eval_close, eval_fill)

    rows: list[dict[str, Any]] = []
    rows.append(_run_bt("baseline_hf_no_limit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, eval_pre, fee=args.fee, slip=args.slip))
    rows.append(_run_bt("mz_entry_base_exit", eval_df, policy, base_exit_model, entry_cfg, risk_cfg, exit_cfg, best_entry_pre, fee=args.fee, slip=args.slip))
    if az_exit_model is not None:
        rows.append(
            _run_bt(
                "mz_entry_az_exit0.45",
                eval_df,
                policy,
                az_exit_model,
                entry_cfg,
                risk_cfg,
                {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]},
                best_entry_pre,
                fee=args.fee,
                slip=args.slip,
            )
        )

    for gamma in (0.60, 0.75):
        for prior_w in (0.0, 0.12):
            for temp in (0.35, 0.55):
                model = MZExitModel(bundle, gamma=gamma, prior_weight=prior_w, temperature=temp)
                for th in (0.45, 0.54, 0.63):
                    rows.append(
                        _run_bt(
                            f"mz_entry_mz_exit_g{gamma:.2f}_p{prior_w:.2f}_t{temp:.2f}_th{th:.2f}",
                            eval_df,
                            policy,
                            model,
                            entry_cfg,
                            risk_cfg,
                            {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]},
                            best_entry_pre,
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

    def reconstruct(name: str) -> tuple[Any, dict[str, Any], tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]]:
        if name == "baseline_hf_no_limit":
            return base_exit_model, exit_cfg, eval_pre
        if name == "mz_entry_base_exit":
            return base_exit_model, exit_cfg, best_entry_pre
        if name == "mz_entry_az_exit0.45" and az_exit_model is not None:
            return az_exit_model, {"exit_threshold": 0.45, "min_exit_age": exit_cfg["min_exit_age"]}, best_entry_pre
        parts = name.split("_")
        gamma = float(parts[4].replace("g", ""))
        prior_w = float(parts[5].replace("p", ""))
        temp = float(parts[6].replace("t", ""))
        th = float(parts[7].replace("th", ""))
        return MZExitModel(bundle, gamma=gamma, prior_weight=prior_w, temperature=temp), {"exit_threshold": th, "min_exit_age": exit_cfg["min_exit_age"]}, best_entry_pre

    selected_detail = []
    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for row in chosen:
        model, cfg, pre = reconstruct(row["name"])
        selected_detail.append(_run_bt(row["name"], eval_df, policy, model, entry_cfg, risk_cfg, cfg, pre, fee=args.fee, slip=args.slip, monthly=True))
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = []
        for row in chosen:
            model, cfg, pre = reconstruct(row["name"])
            cost_stress[f"cost_{mult:g}x"].append(_run_bt(row["name"], eval_df, policy, model, entry_cfg, risk_cfg, cfg, pre, fee=args.fee * mult, slip=args.slip * mult))

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "muzero_style_exit_governor",
            "state_dict": net.state_dict(),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "feature_cols": list(MODEL_COLS),
            "actions": list(EXIT_PLANNER_ACTIONS),
            "train_meta": train_meta,
            "label_meta": label_meta,
        },
        args.model_out,
    )
    report = {
        "type": "muzero_style_exit_governor_2026",
        "note": "MuZero-inspired exit planner trained only on 2025 lifecycle states. 2026 inference uses current lifecycle features plus learned dynamics/reward/value, not future prices.",
        "model_out": str(args.model_out),
        "entry_model": str(args.mz_entry_model),
        "audit": _audit(args.train_csv, args.eval_csv, policy),
        "label_meta": label_meta,
        "train_meta": train_meta,
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
            "mz_entry_az_exit_pnl": next((r["eval"]["pnl"] for r in rows if r["name"] == "mz_entry_az_exit0.45"), None),
        },
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "model": str(args.model_out), "decision": report["decision"], "top_pnl": ranked_pnl[:8], "top_score": ranked_score[:8]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
