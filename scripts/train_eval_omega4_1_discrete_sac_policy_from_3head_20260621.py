#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_1_discrete_sac_policy_from_3head_20260621"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_OMEGA4_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_smoke_loose_entry_loose_quality_terminal_giveback_exit_e2_train15k_exit15k_q070"
)
BASELINE_OOS = {"pnl": 7.5133, "mdd": -5.6140, "trades": 100, "wr": 0.63}


@dataclass(frozen=True)
class PolicyAction:
    name: str
    side: int
    margin_fraction: float

    @property
    def notional(self) -> float:
        return float(self.margin_fraction) * 3.0


ACTION_GRID = [
    PolicyAction("flat", 0, 0.0),
    PolicyAction("long_m005", 1, 0.05),
    PolicyAction("long_m010", 1, 0.10),
    PolicyAction("long_m020", 1, 0.20),
    PolicyAction("long_m030", 1, 0.30),
    PolicyAction("short_m005", -1, 0.05),
    PolicyAction("short_m010", -1, 0.10),
    PolicyAction("short_m020", -1, 0.20),
    PolicyAction("short_m030", -1, 0.30),
]


class Replay:
    def __init__(self, capacity: int = 200_000) -> None:
        self.capacity = int(capacity)
        self.buf: list[tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.pos = 0

    def add(self, s: np.ndarray, a: int, r: float, ns: np.ndarray, done: bool) -> None:
        item = (s.copy(), int(a), float(r), ns.copy(), float(done))
        if len(self.buf) < self.capacity:
            self.buf.append(item)
        else:
            self.buf[self.pos] = item
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = random.sample(self.buf, int(n))
        s, a, r, ns, d = zip(*batch)
        rewards = np.asarray(r, dtype=np.float32)
        scale = float(max(rewards.std(), 0.005))
        return (
            torch.tensor(np.asarray(s), dtype=torch.float32),
            torch.tensor(a, dtype=torch.long),
            torch.tensor((rewards - float(rewards.mean())) / scale, dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.asarray(ns), dtype=torch.float32),
            torch.tensor(d, dtype=torch.float32).unsqueeze(1),
        )

    def __len__(self) -> int:
        return len(self.buf)


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.q1 = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))
        self.q2 = nn.Sequential(nn.Linear(state_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, action_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q1(x), self.q2(x)


class DiscreteSAC:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        device: str,
        *,
        lr: float = 3.0e-4,
        gamma: float = 0.97,
        tau: float = 0.02,
        alpha_init: float = 0.08,
        alpha_min: float = 0.005,
        alpha_max: float = 0.25,
    ) -> None:
        self.device = torch.device(device)
        self.actor = Actor(state_dim, action_dim).to(self.device)
        self.critic = Critic(state_dim, action_dim).to(self.device)
        self.target = copy.deepcopy(self.critic).to(self.device)
        self.actor_opt = torch.optim.AdamW(self.actor.parameters(), lr=lr, weight_decay=1.0e-4)
        self.critic_opt = torch.optim.AdamW(self.critic.parameters(), lr=lr, weight_decay=1.0e-4)
        self.log_alpha = torch.tensor([math.log(alpha_init)], dtype=torch.float32, device=self.device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)
        self.alpha_min = float(alpha_min)
        self.alpha_max = float(alpha_max)
        self.target_entropy = 0.50 * math.log(action_dim)
        self.gamma = float(gamma)
        self.tau = float(tau)

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        x = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        with torch.no_grad():
            probs = torch.softmax(self.actor(x), dim=-1)[0]
        if deterministic:
            return int(torch.argmax(probs).item())
        return int(torch.distributions.Categorical(probs=probs).sample().item())

    def update(self, replay: Replay, batch_size: int) -> dict[str, float]:
        if len(replay) < int(batch_size):
            return {}
        s, a, r, ns, d = replay.sample(batch_size)
        s, a, r, ns, d = s.to(self.device), a.to(self.device), r.to(self.device), ns.to(self.device), d.to(self.device)
        alpha = self.log_alpha.exp().clamp(self.alpha_min, self.alpha_max)
        with torch.no_grad():
            npb = torch.softmax(self.actor(ns), dim=-1)
            nlp = torch.log(npb + 1.0e-8)
            tq1, tq2 = self.target(ns)
            nv = (npb * (torch.minimum(tq1, tq2) - alpha * nlp)).sum(dim=1, keepdim=True)
            target = r + self.gamma * (1.0 - d) * nv
        q1, q2 = self.critic(s)
        q1a = q1.gather(1, a.unsqueeze(1))
        q2a = q2.gather(1, a.unsqueeze(1))
        critic_loss = F.mse_loss(q1a, target) + F.mse_loss(q2a, target)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        probs = torch.softmax(self.actor(s), dim=-1)
        logp = torch.log(probs + 1.0e-8)
        pq1, pq2 = self.critic(s)
        actor_loss = (probs * (alpha.detach() * logp - torch.minimum(pq1, pq2))).sum(dim=1).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        entropy = -(probs * logp).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        with torch.no_grad():
            self.log_alpha.clamp_(math.log(self.alpha_min), math.log(self.alpha_max))
            for tp, p in zip(self.target.parameters(), self.critic.parameters()):
                tp.data.mul_(1.0 - self.tau).add_(self.tau * p.data)
        return {"critic_loss": float(critic_loss.item()), "actor_loss": float(actor_loss.item()), "alpha": float(alpha.item()), "entropy": float(entropy.mean().item())}


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


def _load_bundle(model_dir: Path) -> dict[str, Any]:
    path = Path(model_dir) / "true_3head_tabm_bundle.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    bundle = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("models", "base_cols", "pos_cols"):
        if key not in bundle:
            raise RuntimeError(f"{path} missing {key}")
    return bundle


def _predict_heads(frame: pd.DataFrame, bundle: dict[str, Any], *, device: torch.device) -> dict[str, np.ndarray]:
    x = parent._base_input(frame, list(bundle["base_cols"]))
    preds = {expert: parent._predict_payload(bundle["models"][expert], x, device=device) for expert in hard.EXPERT_NAMES}
    route = hard._route_id(frame)
    return {
        "direction": parent._routed(preds, route, "direction", 3),
        "quality": parent._routed(preds, route, "quality", 3),
        "exit": parent._routed(preds, route, "exit", 2),
    }


def _prepare_frames(model_dir: Path, *, device: torch.device) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
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
    bundle = _load_bundle(model_dir)
    return {
        **frames,
        "train_heads": _predict_heads(frames["train_raw"], bundle, device=device),
        "val_heads": _predict_heads(frames["val_raw"], bundle, device=device),
        "oos_heads": _predict_heads(frames["oos_raw"], bundle, device=device),
        "model_dir": str(model_dir),
    }


def _feature_frame(frame: pd.DataFrame, heads: dict[str, np.ndarray]) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    out["ret1"] = pd.Series(close).pct_change().fillna(0.0).to_numpy(dtype=np.float64)
    out["ret6"] = pd.Series(close).pct_change(6).fillna(0.0).to_numpy(dtype=np.float64)
    out["ret24"] = pd.Series(close).pct_change(24).fillna(0.0).to_numpy(dtype=np.float64)
    out["hl_range"] = (pd.to_numeric(frame["high"], errors="raise") - pd.to_numeric(frame["low"], errors="raise")) / pd.to_numeric(frame["close"], errors="raise").clip(lower=1.0e-12)
    out["dir_cash"], out["dir_long"], out["dir_short"] = heads["direction"][:, 0], heads["direction"][:, 1], heads["direction"][:, 2]
    out["qual_cash"], out["qual_long"], out["qual_short"] = heads["quality"][:, 0], heads["quality"][:, 1], heads["quality"][:, 2]
    out["exit_hold"], out["exit_exit"] = heads["exit"][:, 0], heads["exit"][:, 1]
    out["dir_edge"] = np.maximum(out["dir_long"], out["dir_short"]) - out["dir_cash"]
    out["qual_edge"] = np.maximum(out["qual_long"], out["qual_short"]) - out["qual_cash"]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _prior_from_heads(heads: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    long_score = heads["direction"][:, 1] * heads["quality"][:, 1]
    short_score = heads["direction"][:, 2] * heads["quality"][:, 2]
    prior_side = np.where(long_score >= short_score, 1, -1).astype(np.int64)
    strength = np.abs(long_score - short_score).astype(np.float64)
    prior_side[strength < 0.02] = 0
    return prior_side, strength


def _fit_norm(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1.0e-8)] = 1.0
    return {"columns": list(x.columns), "median": med.astype(np.float32), "scale": scale.astype(np.float32)}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = sorted(set(cols) - set(x.columns))
    if missing:
        raise RuntimeError(f"SAC feature frame missing columns: {missing}")
    arr = x[cols].to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    return np.tanh(np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


class Omega4SacEnv:
    def __init__(
        self,
        frame: pd.DataFrame,
        state_x: np.ndarray,
        *,
        fee: float,
        slip: float,
        cost_mult: float,
        start: int = 0,
        end: int | None = None,
        dd_penalty: float = 0.05,
        turnover_penalty: float = 0.0010,
        min_hold_bars: int = 3,
        target_hold_bars: int = 96,
        flip_penalty: float = 0.0020,
        early_flip_penalty: float = 0.0060,
        prior_side: np.ndarray | None = None,
        prior_strength: np.ndarray | None = None,
        prior_penalty: float = 0.0020,
    ) -> None:
        self.frame = frame.reset_index(drop=True)
        self.state_x = np.asarray(state_x, dtype=np.float32)
        self.fee_eff = float(fee) * float(cost_mult)
        self.slip_eff = float(slip) * float(cost_mult)
        self.start = max(int(start), 0)
        self.end = min(int(end) if end is not None else len(self.frame) - 2, len(self.frame) - 2)
        self.dd_penalty = float(dd_penalty)
        self.turnover_penalty = float(turnover_penalty)
        self.min_hold_bars = max(int(min_hold_bars), 0)
        self.target_hold_bars = max(int(target_hold_bars), 1)
        self.flip_penalty = float(flip_penalty)
        self.early_flip_penalty = float(early_flip_penalty)
        self.prior_side = np.asarray(prior_side, dtype=np.int64) if prior_side is not None else np.zeros(len(self.frame), dtype=np.int64)
        self.prior_strength = np.asarray(prior_strength, dtype=np.float64) if prior_strength is not None else np.zeros(len(self.frame), dtype=np.float64)
        self.prior_penalty = float(prior_penalty)
        self.arrays = {c: pd.to_numeric(self.frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
        self.reset()

    @property
    def state_dim(self) -> int:
        return int(self.state_x.shape[1] + 6)

    def _obs(self) -> np.ndarray:
        px = float(self.arrays["close"][self.i])
        if self.side == 0:
            pos = np.zeros(6, dtype=np.float32)
        else:
            raw = (px * (1.0 - self.slip_eff) - self.entry_price) / max(self.entry_price, 1e-12) if self.side > 0 else (self.entry_price - px * (1.0 + self.slip_eff)) / max(self.entry_price, 1e-12)
            unreal = raw * self.notional
            self.mfe = max(self.mfe, float(unreal))
            self.mae = min(self.mae, float(unreal))
            pos = np.asarray(
                [
                    float(self.side),
                    float(self.margin_fraction),
                    float(self.notional),
                    float(unreal),
                    float(self.mfe),
                    float(max(self.i - self.entry_i, 0)) / 1440.0,
                ],
                dtype=np.float32,
            )
        return np.concatenate([self.state_x[self.i], pos], axis=0).astype(np.float32)

    def reset(self) -> np.ndarray:
        self.i = self.start
        self.cash = 1.0
        self.peak = 1.0
        self.mdd = 0.0
        self.side = 0
        self.entry_price = 0.0
        self.entry_cash = 1.0
        self.entry_i = 0
        self.margin_fraction = 0.0
        self.notional = 0.0
        self.mfe = 0.0
        self.mae = 0.0
        self.trades = 0
        self.wins = 0
        self.long_entries = 0
        self.short_entries = 0
        self.margin_sum = 0.0
        self.exit_reasons: dict[str, int] = {}
        return self._obs()

    def _mark_equity(self) -> float:
        if self.side == 0:
            return float(self.cash)
        px = float(self.arrays["close"][self.i])
        raw = (px * (1.0 - self.slip_eff) - self.entry_price) / max(self.entry_price, 1e-12) if self.side > 0 else (self.entry_price - px * (1.0 + self.slip_eff)) / max(self.entry_price, 1e-12)
        return float(self.cash * (1.0 + raw * self.notional))

    def _exit(self, reason: str) -> None:
        if self.side == 0:
            return
        filled, px, exit_fee, _route = omega._try_execution(self.arrays, int(self.i), self.side, entry=False, fee_base=self.fee_eff, slip_base=self.slip_eff)
        if not filled:
            return
        raw = (px - self.entry_price) / max(self.entry_price, 1e-12) if self.side > 0 else (self.entry_price - px) / max(self.entry_price, 1e-12)
        before = self.cash
        self.cash = self.cash * (1.0 + raw * self.notional)
        self.cash -= before * exit_fee * self.notional
        self.trades += 1
        self.wins += int(self.cash > self.entry_cash)
        self.exit_reasons[reason] = self.exit_reasons.get(reason, 0) + 1
        self.side = 0
        self.entry_price = 0.0
        self.margin_fraction = 0.0
        self.notional = 0.0
        self.mfe = 0.0
        self.mae = 0.0

    def _enter(self, action: PolicyAction) -> None:
        if action.side == 0:
            return
        filled, px, entry_fee, _route = omega._try_execution(self.arrays, int(self.i), action.side, entry=True, fee_base=self.fee_eff, slip_base=self.slip_eff)
        if not filled:
            return
        self.side = int(action.side)
        self.entry_price = float(px)
        self.entry_cash = float(self.cash)
        self.entry_i = int(self.i)
        self.margin_fraction = float(action.margin_fraction)
        self.notional = float(action.notional)
        self.cash -= self.cash * float(entry_fee) * self.notional
        self.long_entries += int(self.side > 0)
        self.short_entries += int(self.side < 0)
        self.margin_sum += float(action.margin_fraction)

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        signal_i = int(self.i)
        before = self._mark_equity()
        action = ACTION_GRID[int(action_id)]
        changed = False
        flipped_or_closed = False
        hold_bars = max(int(self.i) - int(self.entry_i), 0)
        can_change = self.side == 0 or hold_bars >= self.min_hold_bars
        if self.side != 0 and action.side != self.side and can_change:
            self._exit("policy_flat_or_flip")
            changed = True
            flipped_or_closed = True
        if self.side == 0 and action.side != 0:
            self._enter(action)
            changed = True
        self.i += 1
        eq = self._mark_equity()
        self.peak = max(self.peak, eq)
        dd = min(0.0, eq / max(self.peak, 1e-12) - 1.0)
        self.mdd = min(self.mdd, dd)
        contradicts_prior = action.side != 0 and int(self.prior_side[signal_i]) != 0 and int(action.side) != int(self.prior_side[signal_i])
        reward = float(eq - before) - self.dd_penalty * max(0.0, -dd) - (self.turnover_penalty if changed else 0.0)
        if flipped_or_closed:
            early_ratio = max(float(self.target_hold_bars - hold_bars), 0.0) / float(self.target_hold_bars)
            reward -= self.flip_penalty + self.early_flip_penalty * early_ratio
        if contradicts_prior:
            reward -= self.prior_penalty * float(np.clip(self.prior_strength[signal_i] / 0.25, 0.0, 1.0))
        done = self.i >= self.end
        if done and self.side != 0:
            self._exit("forced_end")
            eq = self.cash
        return (self._obs() if not done else np.zeros(self.state_dim, dtype=np.float32)), float(reward), bool(done), {"equity": float(eq)}

    def metrics(self) -> dict[str, Any]:
        duration = max((pd.to_datetime(self.frame["timestamp"].iloc[self.end]) - pd.to_datetime(self.frame["timestamp"].iloc[self.start])).total_seconds() / 86400.0, 1e-9)
        entries = max(self.long_entries + self.short_entries, 1)
        return {
            "pnl": float((self.cash - 1.0) * 100.0),
            "mdd": float(self.mdd * 100.0),
            "trades": int(self.trades),
            "wr": float(self.wins / self.trades) if self.trades else 0.0,
            "trades_per_day": float(self.trades / duration),
            "avg_margin_fraction": float(self.margin_sum / entries),
            "fixed_leverage": 3.0,
            "long_entries": int(self.long_entries),
            "short_entries": int(self.short_entries),
            "exit_reasons": self.exit_reasons,
        }


def _train_agent(env: Omega4SacEnv, *, steps: int, warmup: int, batch_size: int, seed: int, device: str) -> tuple[DiscreteSAC, dict[str, Any]]:
    random.seed(int(seed))
    agent = DiscreteSAC(env.state_dim, len(ACTION_GRID), device, gamma=0.97, alpha_init=0.08, alpha_min=0.005, alpha_max=0.25)
    replay = Replay(capacity=max(int(steps) + 100, 10_000))
    state = env.reset()
    last: dict[str, Any] = {}
    for step in range(int(steps)):
        if step < int(warmup):
            action_id = int(np.random.randint(len(ACTION_GRID)))
        else:
            action_id = int(agent.act(state, deterministic=False))
        next_state, reward, done, _info = env.step(action_id)
        replay.add(state, action_id, reward, next_state, done)
        state = next_state
        if len(replay) >= int(batch_size):
            last = agent.update(replay, int(batch_size)) or last
        if done:
            state = env.reset()
    return agent, {"steps": int(steps), "replay_size": int(len(replay)), "last_update": last}


def _eval_agent(
    agent: DiscreteSAC,
    frame: pd.DataFrame,
    x: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    turnover_penalty: float,
    min_hold_bars: int,
    target_hold_bars: int,
    flip_penalty: float,
    early_flip_penalty: float,
    prior_side: np.ndarray,
    prior_strength: np.ndarray,
    prior_penalty: float,
) -> dict[str, Any]:
    env = Omega4SacEnv(
        frame,
        x,
        fee=fee,
        slip=slip,
        cost_mult=cost_mult,
        start=0,
        end=len(frame) - 2,
        turnover_penalty=float(turnover_penalty),
        min_hold_bars=int(min_hold_bars),
        target_hold_bars=int(target_hold_bars),
        flip_penalty=float(flip_penalty),
        early_flip_penalty=float(early_flip_penalty),
        prior_side=prior_side,
        prior_strength=prior_strength,
        prior_penalty=float(prior_penalty),
    )
    state = env.reset()
    while True:
        action_id = int(agent.act(state, deterministic=True))
        state, _reward, done, _info = env.step(action_id)
        if done:
            break
    return env.metrics()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--omega4-model-dir", type=Path, default=DEFAULT_OMEGA4_DIR)
    ap.add_argument("--train-rows", type=int, default=5000)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--turnover-penalty", type=float, default=0.0010)
    ap.add_argument("--min-hold-bars", type=int, default=3)
    ap.add_argument("--target-hold-bars", type=int, default=96)
    ap.add_argument("--flip-penalty", type=float, default=0.0020)
    ap.add_argument("--early-flip-penalty", type=float, default=0.0060)
    ap.add_argument("--prior-penalty", type=float, default=0.0020)
    ap.add_argument("--seed", type=int, default=260621)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    ap.add_argument("--out-suffix", default="smoke_train5k_steps1500")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    frames = _prepare_frames(Path(args.omega4_model_dir), device=device)
    train_frame = frames["train_raw"].iloc[: int(args.train_rows)].reset_index(drop=True)
    train_heads = {k: v[: len(train_frame)] for k, v in frames["train_heads"].items()}
    x_train_df = _feature_frame(train_frame, train_heads)
    x_val_df = _feature_frame(frames["val_raw"], frames["val_heads"])
    x_oos_df = _feature_frame(frames["oos_raw"], frames["oos_heads"])
    train_prior_side, train_prior_strength = _prior_from_heads(train_heads)
    val_prior_side, val_prior_strength = _prior_from_heads(frames["val_heads"])
    oos_prior_side, oos_prior_strength = _prior_from_heads(frames["oos_heads"])
    norm = _fit_norm(x_train_df)
    x_train = _apply_norm(x_train_df, norm)
    x_val = _apply_norm(x_val_df, norm)
    x_oos = _apply_norm(x_oos_df, norm)
    train_env = Omega4SacEnv(
        train_frame,
        x_train,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        start=0,
        end=len(train_frame) - 2,
        turnover_penalty=float(args.turnover_penalty),
        min_hold_bars=int(args.min_hold_bars),
        target_hold_bars=int(args.target_hold_bars),
        flip_penalty=float(args.flip_penalty),
        early_flip_penalty=float(args.early_flip_penalty),
        prior_side=train_prior_side,
        prior_strength=train_prior_strength,
        prior_penalty=float(args.prior_penalty),
    )
    agent, train_summary = _train_agent(
        train_env,
        steps=int(args.steps),
        warmup=int(args.warmup),
        batch_size=int(args.batch_size),
        seed=int(args.seed),
        device=str(device),
    )
    val_metrics = _eval_agent(
        agent,
        frames["val_raw"],
        x_val,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        turnover_penalty=float(args.turnover_penalty),
        min_hold_bars=int(args.min_hold_bars),
        target_hold_bars=int(args.target_hold_bars),
        flip_penalty=float(args.flip_penalty),
        early_flip_penalty=float(args.early_flip_penalty),
        prior_side=val_prior_side,
        prior_strength=val_prior_strength,
        prior_penalty=float(args.prior_penalty),
    )
    oos_metrics = _eval_agent(
        agent,
        frames["oos_raw"],
        x_oos,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        turnover_penalty=float(args.turnover_penalty),
        min_hold_bars=int(args.min_hold_bars),
        target_hold_bars=int(args.target_hold_bars),
        flip_penalty=float(args.flip_penalty),
        early_flip_penalty=float(args.early_flip_penalty),
        prior_side=oos_prior_side,
        prior_strength=oos_prior_strength,
        prior_penalty=float(args.prior_penalty),
    )
    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega4_1_exit_thr_0p70",
        "baseline_oos": BASELINE_OOS,
        "design": "Discrete SAC policy over Omega 4.1 three-head probabilities. TP/SL/max-hold risk template is removed; action owns target side and margin fraction. Fixed leverage=3, notional=margin_fraction*3.",
        "action_contract": [a.__dict__ | {"notional": a.notional} for a in ACTION_GRID],
        "hard_bounds": {"max_margin_fraction": 0.30, "fixed_leverage": 3.0, "max_notional": 0.90},
        "execution_safety": {
            "min_hold_bars": int(args.min_hold_bars),
            "target_hold_bars": int(args.target_hold_bars),
            "turnover_penalty": float(args.turnover_penalty),
            "flip_penalty": float(args.flip_penalty),
            "early_flip_penalty": float(args.early_flip_penalty),
            "prior_penalty": float(args.prior_penalty),
        },
        "training": train_summary,
        "results": {"validation": val_metrics, "oos": oos_metrics},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "policy": str(out_dir / "discrete_sac_policy.pt")},
    }
    torch.save(
        {
            "actor_state_dict": agent.actor.state_dict(),
            "critic_state_dict": agent.critic.state_dict(),
            "norm": norm,
            "action_grid": [a.__dict__ for a in ACTION_GRID],
            "state_dim": train_env.state_dim,
        },
        out_dir / "discrete_sac_policy.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
