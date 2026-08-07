from __future__ import annotations

import argparse
import copy
import logging
import os
import random
import sys
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Categorical

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_THIS_DIR)
for _p in (_ROOT_DIR, _THIS_DIR, os.path.join(_ROOT_DIR, "ensemble")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CSV_DEFAULT = "data/rl_training_2025_unified_supdir_cat_gate.csv"
EDITOR_STATE_SCHEMA = "unified_trade_editor_supdir_v3"
EDITOR_STATE_DIM = 14
EDITOR_ACTIONS = ["HOLD", "OPEN_LONG", "OPEN_SHORT", "CLOSE"]
ACT_HOLD = 0
ACT_OPEN_LONG = 1
ACT_OPEN_SHORT = 2
ACT_CLOSE = 3
POSITION_IDX = 10
GATE_PROB_IDX = 5
MIN_HOLD_BARS = 8
REENTRY_COOLDOWN_BARS = 4
MAX_TRADES_PER_EPISODE = 180
OPEN_CHURN_PENALTY = 0.004
CLOSE_CHURN_PENALTY = 0.002
STRONG_EDGE_ENTRY_BONUS = 0.0015
MIN_VAL_TRADES_FOR_BEST = 30
EARLY_STOP_PATIENCE = 2

REQ_COLS = [
    "close",
    "ud_sup_short_prob",
    "ud_sup_flat_prob",
    "ud_sup_long_prob",
    "ud_sup_edge",
    "ud_gate_take_prob",
    "smart_money_flow",
    "taker_acceleration",
    "trade_intensity",
    "garch_vol_z",
    "regime_bull",
    "regime_bear",
    "regime_chop",
    "regime_whipsaw",
    "regime_normal",
]


def _safe(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _resolve_runtime_device(requested: str) -> str:
    req = (requested or "auto").strip().lower()
    if req == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if req == "cpu":
        return "cpu"
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return "cuda"
    raise ValueError(f"invalid device: {requested}")


def _regime_name_from_row(row: dict[str, Any]) -> str:
    vals = {
        "bull": _safe(row.get("regime_bull", 0.0)),
        "bear": _safe(row.get("regime_bear", 0.0)),
        "chop": _safe(row.get("regime_chop", 0.0)),
        "whipsaw": _safe(row.get("regime_whipsaw", 0.0)),
        "normal": _safe(row.get("regime_normal", 0.0)),
    }
    return max(vals.items(), key=lambda kv: kv[1])[0]


def build_editor_state(row: dict[str, Any], pos: dict[str, Any] | None = None) -> np.ndarray:
    pos = pos or {}
    pos_type = pos.get("type")
    pos_sign = 1.0 if pos_type == "LONG" else (-1.0 if pos_type == "SHORT" else 0.0)
    unreal = _safe(pos.get("unrealized", 0.0), 0.0)
    hold = float(np.clip(_safe(pos.get("hold_count", 0.0), 0.0) / 96.0, 0.0, 1.0))
    long_p = _safe(row.get("ud_sup_long_prob", 0.0))
    flat_p = _safe(row.get("ud_sup_flat_prob", 0.0))
    short_p = _safe(row.get("ud_sup_short_prob", 0.0))
    state = np.array(
        [
            long_p,
            flat_p,
            short_p,
            float(np.tanh(_safe(row.get("ud_sup_edge", 0.0)) / 0.25)),
            float(max(long_p, flat_p, short_p)),
            _safe(row.get("ud_gate_take_prob", 0.0)),
            float(np.tanh(_safe(row.get("smart_money_flow", 0.0)) / 0.05)),
            float(np.tanh(_safe(row.get("taker_acceleration", 0.0)) / 0.05)),
            float(np.tanh(_safe(row.get("trade_intensity", 0.0)) / 5.0)),
            float(np.tanh(_safe(row.get("garch_vol_z", 0.0)) / 3.0)),
            pos_sign,
            float(np.tanh(unreal / 0.02)),
            hold,
            float(
                np.argmax(
                    [
                        _safe(row.get("regime_bull", 0.0)),
                        _safe(row.get("regime_bear", 0.0)),
                        _safe(row.get("regime_chop", 0.0)),
                        _safe(row.get("regime_whipsaw", 0.0)),
                        _safe(row.get("regime_normal", 0.0)),
                    ]
                )
                / 4.0
            ),
        ],
        dtype=np.float32,
    )
    return np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=-1.0)


def action_mask_from_state_tensor(state: torch.Tensor) -> torch.Tensor:
    pos = state[..., POSITION_IDX]
    gate_prob = state[..., GATE_PROB_IDX]
    flat = pos.abs() < 0.5
    in_pos = ~flat
    mask = torch.zeros((*state.shape[:-1], len(EDITOR_ACTIONS)), dtype=torch.bool, device=state.device)
    mask[..., ACT_HOLD] |= True
    open_ok = flat & (gate_prob >= 0.55)
    mask[..., ACT_OPEN_LONG] |= open_ok
    mask[..., ACT_OPEN_SHORT] |= open_ok
    mask[..., ACT_CLOSE] |= in_pos
    return mask


def _masked_logits(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~mask, -1e9)


class ReplayBuffer:
    def __init__(self, capacity: int = 200000):
        self.capacity = max(1000, int(capacity))
        self.data: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=self.capacity)

    def push(self, state, action, reward, next_state, done):
        self.data.append((state.copy(), int(action), float(reward), next_state.copy(), float(done)))

    def __len__(self):
        return len(self.data)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.asarray(s, dtype=np.float32),
            np.asarray(a, dtype=np.int64),
            np.asarray(r, dtype=np.float32),
            np.asarray(ns, dtype=np.float32),
            np.asarray(d, dtype=np.float32),
        )


class RegimeBalancedReplay:
    def __init__(self, capacity: int = 200000, recent_mix_ratio: float = 0.30, recent_window: int = 50000):
        self._global = ReplayBuffer(capacity)
        self._recent = ReplayBuffer(recent_window)
        per_cap = max(500, capacity // 5)
        self._by_regime = {k: ReplayBuffer(per_cap) for k in ("bull", "bear", "chop", "whipsaw", "normal")}
        self.recent_mix_ratio = float(np.clip(recent_mix_ratio, 0.0, 0.9))

    def push(self, state, action, reward, next_state, done, regime: str = "normal"):
        key = regime if regime in self._by_regime else "normal"
        self._global.push(state, action, reward, next_state, done)
        self._recent.push(state, action, reward, next_state, done)
        self._by_regime[key].push(state, action, reward, next_state, done)

    def __len__(self):
        return len(self._global)

    def sample(self, batch_size: int):
        n_recent = int(batch_size * self.recent_mix_ratio)
        n_bal = batch_size - n_recent
        chunks = []
        if n_bal > 0:
            weights = {"normal": 0.30, "chop": 0.15, "whipsaw": 0.20, "bull": 0.15, "bear": 0.20}
            taken = 0
            for regime, w in weights.items():
                q = int(n_bal * w)
                rb = self._by_regime[regime]
                if q > 0 and len(rb) >= q:
                    chunks.append(rb.sample(q))
                    taken += q
            if taken < n_bal:
                chunks.append(self._global.sample(n_bal - taken))
        if n_recent > 0:
            if len(self._recent) >= n_recent:
                chunks.append(self._recent.sample(n_recent))
            else:
                chunks.append(self._global.sample(n_recent))
        s = np.concatenate([c[0] for c in chunks], axis=0)
        a = np.concatenate([c[1] for c in chunks], axis=0)
        r = np.concatenate([c[2] for c in chunks], axis=0)
        ns = np.concatenate([c[3] for c in chunks], axis=0)
        d = np.concatenate([c[4] for c in chunks], axis=0)
        idx = np.random.permutation(len(s))
        return s[idx], a[idx], r[idx], ns[idx], d[idx]


class FeatureExtractor(nn.Module):
    def __init__(self, state_dim: int = EDITOR_STATE_DIM, hidden_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DiscreteActor(nn.Module):
    def __init__(self, state_dim: int = EDITOR_STATE_DIM, hidden_dim: int = 192, action_dim: int = len(EDITOR_ACTIONS)):
        super().__init__()
        self.feat = FeatureExtractor(state_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.logits(self.feat(state))

    def sample(self, state: torch.Tensor, mask: torch.Tensor | None = None):
        logits = self.forward(state)
        if mask is not None:
            logits = _masked_logits(logits, mask)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action).unsqueeze(-1)
        probs = torch.softmax(logits, dim=-1)
        return action, log_prob, probs

    def deterministic(self, state: torch.Tensor, mask: torch.Tensor | None = None):
        logits = self.forward(state)
        if mask is not None:
            logits = _masked_logits(logits, mask)
        return torch.argmax(logits, dim=-1)


class TwinQCritic(nn.Module):
    def __init__(self, state_dim: int = EDITOR_STATE_DIM, hidden_dim: int = 192, action_dim: int = len(EDITOR_ACTIONS)):
        super().__init__()
        self.feat1 = FeatureExtractor(state_dim, hidden_dim)
        self.feat2 = FeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim))
        self.q2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, action_dim))

    def forward(self, state: torch.Tensor):
        return self.q1(self.feat1(state)), self.q2(self.feat2(state))


class DiscreteSACAgent:
    def __init__(
        self,
        state_dim: int = EDITOR_STATE_DIM,
        hidden_dim: int = 192,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha_init: float = 0.03,
        alpha_min: float = 5e-3,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.alpha_min = float(alpha_min)
        self.actor = DiscreteActor(state_dim, hidden_dim).to(device)
        self.critic = TwinQCritic(state_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        self.log_alpha = torch.tensor([np.log(max(alpha_init, alpha_min))], dtype=torch.float32, device=device, requires_grad=True)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        self.target_entropy = 0.95 * np.log(3.0)
        self.memory = RegimeBalancedReplay()
        self._updates = 0

    @property
    def alpha(self) -> float:
        return float(torch.clamp(self.log_alpha.exp(), min=self.alpha_min).item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> int:
        st = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask = action_mask_from_state_tensor(st)
        with torch.no_grad():
            if deterministic:
                act = self.actor.deterministic(st, mask=mask)
            else:
                act, _, _ = self.actor.sample(st, mask=mask)
        return int(act.item())

    def update(self, batch_size: int = 256) -> dict[str, float]:
        if len(self.memory) < batch_size:
            return {}
        s, a, r, ns, d = self.memory.sample(batch_size)
        s = torch.as_tensor(s, dtype=torch.float32, device=self.device)
        a = torch.as_tensor(a, dtype=torch.long, device=self.device)
        r = torch.as_tensor(r, dtype=torch.float32, device=self.device).unsqueeze(1)
        ns = torch.as_tensor(ns, dtype=torch.float32, device=self.device)
        d = torch.as_tensor(d, dtype=torch.float32, device=self.device).unsqueeze(1)
        s_mask = action_mask_from_state_tensor(s)
        ns_mask = action_mask_from_state_tensor(ns)

        with torch.no_grad():
            next_logits = _masked_logits(self.actor(ns), ns_mask)
            next_probs = torch.softmax(next_logits, dim=-1)
            next_log_probs = torch.log(next_probs + 1e-8)
            tq1, tq2 = self.critic_target(ns)
            tq = torch.minimum(tq1, tq2)
            alpha = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
            next_v = (next_probs * (tq - alpha * next_log_probs)).sum(dim=1, keepdim=True)
            target_q = r + self.gamma * (1.0 - d) * next_v

        q1_all, q2_all = self.critic(s)
        q1 = q1_all.gather(1, a.unsqueeze(1))
        q2 = q2_all.gather(1, a.unsqueeze(1))
        critic_loss = torch.nn.functional.mse_loss(q1, target_q) + torch.nn.functional.mse_loss(q2, target_q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        logits = _masked_logits(self.actor(s), s_mask)
        probs = torch.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-8)
        q1_pi, q2_pi = self.critic(s)
        q_pi = torch.minimum(q1_pi, q2_pi)
        alpha = torch.clamp(self.log_alpha.exp().detach(), min=self.alpha_min)
        flat_pen = probs[:, ACT_HOLD].mean()
        actor_loss = (probs * (alpha * log_probs - q_pi)).sum(dim=1).mean() + 0.01 * flat_pen
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        entropy = -(probs * log_probs).sum(dim=1, keepdim=True)
        alpha_loss = -(self.log_alpha * (entropy.detach() - self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        with torch.no_grad():
            self.log_alpha.data.clamp_(min=float(np.log(self.alpha_min)))

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)
        self._updates += 1
        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha": self.alpha,
            "entropy": float(entropy.mean().item()),
        }


class EditorTradingEnv:
    def __init__(
        self,
        df: pd.DataFrame,
        fee: float = 0.0005,
        slip: float = 0.0002,
        phase: str = "train",
        window_min: int = 1500,
        terminal_force_close_penalty: float = 0.02,
    ):
        self.df = df.reset_index(drop=True)
        self._close = pd.to_numeric(self.df["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
        self.fee = float(fee)
        self.slip = float(slip)
        self.phase = phase
        self.window_min = int(window_min)
        self.terminal_force_close_penalty = float(terminal_force_close_penalty)
        self.n = len(self.df)
        self.reset()

    def reset(self, start_idx: int | None = None):
        if self.phase == "train":
            max_start = max(0, self.n - self.window_min - 2)
            self.start_idx = random.randint(0, max_start) if max_start > 0 else 0
            self.end_idx = min(self.n - 1, self.start_idx + self.window_min)
        else:
            self.start_idx = 0 if start_idx is None else int(start_idx)
            self.end_idx = self.n - 1
        self.i = self.start_idx
        self.balance = 1.0
        self.pos: str | None = None
        self.entry_price = 0.0
        self.hold_count = 0
        self.unrealized = 0.0
        self.trades = 0
        self.wins = 0
        self.long_count = 0
        self.short_count = 0
        self.force_close_long = 0
        self.force_close_short = 0
        self.peak_equity = 1.0
        self.max_drawdown = 0.0
        self.last_trade_bar = -999999
        self.bars_since_close = 999999
        self.realized_pnls: list[float] = []
        return self._get_state()

    def _pos_dict(self) -> dict[str, Any]:
        return {
            "type": self.pos,
            "unrealized": self.unrealized,
            "hold_count": self.hold_count,
        }

    def _get_state(self) -> np.ndarray:
        return build_editor_state(self.df.iloc[self.i].to_dict(), self._pos_dict())

    def _pnl(self, entry: float, exit_price: float, side: str) -> float:
        if side == "LONG":
            return (exit_price - entry) / max(entry, 1e-8)
        return (entry - exit_price) / max(entry, 1e-8)

    def _equity(self, mark: float) -> float:
        if self.pos is None:
            return self.balance
        pnl = self._pnl(self.entry_price, mark, self.pos)
        return self.balance * (1.0 + pnl)

    def _open(self, side: str, price: float):
        if self.pos is not None:
            return
        fill = price * (1.0 + self.slip) if side == "LONG" else price * (1.0 - self.slip)
        self.balance *= (1.0 - self.fee)
        self.pos = side
        self.entry_price = float(fill)
        self.hold_count = 0
        self.unrealized = 0.0
        self.last_trade_bar = self.i
        self.bars_since_close = 0
        if side == "LONG":
            self.long_count += 1
        else:
            self.short_count += 1

    def _close_position(self, price: float, forced: bool = False):
        if self.pos is None:
            return 0.0
        fill = price * (1.0 - self.slip) if self.pos == "LONG" else price * (1.0 + self.slip)
        pnl = self._pnl(self.entry_price, fill, self.pos)
        self.balance *= max(1e-8, (1.0 + pnl) * (1.0 - self.fee))
        self.trades += 1
        if pnl > 0:
            self.wins += 1
        self.realized_pnls.append(float(pnl))
        if forced:
            if self.pos == "LONG":
                self.force_close_long += 1
            else:
                self.force_close_short += 1
        self.pos = None
        self.entry_price = 0.0
        self.hold_count = 0
        self.unrealized = 0.0
        self.last_trade_bar = self.i
        self.bars_since_close = 0
        return float(pnl)

    def step(self, action: int):
        action = int(np.clip(action, 0, len(EDITOR_ACTIONS) - 1))
        row = self.df.iloc[self.i].to_dict()
        price = float(self._close[self.i])
        next_price = float(self._close[min(self.i + 1, self.end_idx)])
        reward = 0.0
        strong_edge = abs(_safe(row.get("ud_sup_edge", 0.0))) >= 0.10
        long_prob = _safe(row.get("ud_sup_long_prob", 0.0))
        short_prob = _safe(row.get("ud_sup_short_prob", 0.0))
        regime_bull = _safe(row.get("regime_bull", 0.0))
        regime_normal = _safe(row.get("regime_normal", 0.0))

        if self.pos is None:
            if self.trades >= MAX_TRADES_PER_EPISODE:
                action = ACT_HOLD
            if self.bars_since_close < REENTRY_COOLDOWN_BARS:
                action = ACT_HOLD if action in (ACT_OPEN_LONG, ACT_OPEN_SHORT) else action
            if action == ACT_OPEN_LONG and long_prob >= max(short_prob + 0.06, 0.48) and _safe(row.get("ud_sup_edge", 0.0)) >= 0.03:
                self._open("LONG", price)
                reward -= OPEN_CHURN_PENALTY
                if strong_edge:
                    reward += STRONG_EDGE_ENTRY_BONUS
            elif (
                action == ACT_OPEN_SHORT
                and short_prob >= max(long_prob + (0.10 if (regime_bull >= 0.5 or regime_normal >= 0.5) else 0.06), 0.50 if (regime_bull >= 0.5 or regime_normal >= 0.5) else 0.48)
                and _safe(row.get("ud_sup_edge", 0.0)) <= (-0.08 if (regime_bull >= 0.5 or regime_normal >= 0.5) else -0.03)
            ):
                self._open("SHORT", price)
                reward -= OPEN_CHURN_PENALTY
                if strong_edge:
                    reward += STRONG_EDGE_ENTRY_BONUS
            else:
                reward -= 0.0005 if strong_edge else 0.0
        else:
            if action == ACT_CLOSE and self.hold_count >= MIN_HOLD_BARS:
                pnl = self._close_position(price)
                reward += pnl - CLOSE_CHURN_PENALTY
            else:
                self.hold_count += 1

        if self.pos is not None:
            self.unrealized = self._pnl(self.entry_price, next_price, self.pos)
            reward += 0.35 * self.unrealized
            if self.unrealized <= -0.025:
                reward += self._close_position(next_price, forced=True) - self.terminal_force_close_penalty
            elif self.hold_count >= 96:
                reward += self._close_position(next_price, forced=True) - 0.01

        equity = self._equity(next_price)
        self.peak_equity = max(self.peak_equity, equity)
        dd = (equity / max(self.peak_equity, 1e-8)) - 1.0
        self.max_drawdown = min(self.max_drawdown, dd)

        self.i += 1
        self.bars_since_close += 1
        done = self.i >= self.end_idx
        if done and self.pos is not None:
            reward += self._close_position(next_price, forced=True) - 0.005
        next_state = build_editor_state(self.df.iloc[min(self.i, self.end_idx)].to_dict(), self._pos_dict())
        info = {"regime": _regime_name_from_row(row)}
        return next_state, float(reward), bool(done), info


def evaluate(agent: DiscreteSACAgent, env: EditorTradingEnv) -> dict[str, float]:
    state = env.reset(start_idx=0)
    done = False
    while not done:
        action = agent.act(state, deterministic=True)
        state, _, done, _ = env.step(action)
    pnl = (env.balance - 1.0) * 100.0
    wr = (env.wins / max(env.trades, 1)) * 100.0
    return {
        "pnl": float(pnl),
        "trades": int(env.trades),
        "wr": float(wr),
        "mdd": float(env.max_drawdown * 100.0),
        "longs": int(env.long_count),
        "shorts": int(env.short_count),
        "fcl": int(env.force_close_long),
        "fcs": int(env.force_close_short),
    }


def startup_check(csv_path: str) -> None:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)
    df = pd.read_csv(csv_path, nrows=2)
    missing = [c for c in REQ_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    ex = build_editor_state(df.iloc[0].to_dict(), pos={})
    if ex.shape[0] != EDITOR_STATE_DIM:
        raise RuntimeError(f"state_dim mismatch: {ex.shape[0]} != {EDITOR_STATE_DIM}")
    logger.info("startup check ok: unified_trade_editor | schema=%s | state_dim=%d | actions=%s", EDITOR_STATE_SCHEMA, EDITOR_STATE_DIM, ",".join(EDITOR_ACTIONS))


def train(
    csv_path: str,
    episodes: int = 30,
    train_ratio: float = 0.8,
    batch_size: int = 256,
    warmup_steps: int = 2000,
    updates_per_step: int = 1,
    device: str = "auto",
    best_path: str = "data/ensemble/ckpt/best_unified_trade_editor.pth",
):
    df = pd.read_csv(csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val = df.iloc[split_idx:].reset_index(drop=True)
    device = _resolve_runtime_device(device)
    env = EditorTradingEnv(df_train, phase="train")
    val_env = EditorTradingEnv(df_val, phase="val")
    agent = DiscreteSACAgent(device=device)
    best = -1e18
    bad_val = 0
    os.makedirs(os.path.dirname(best_path), exist_ok=True)
    total_steps = 0
    for ep in range(1, episodes + 1):
        state = env.reset()
        done = False
        ep_reward = 0.0
        last_stats = {}
        while not done:
            if total_steps < warmup_steps:
                st = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                mask = action_mask_from_state_tensor(st).squeeze(0).detach().cpu().numpy()
                valid = np.flatnonzero(mask)
                action = int(np.random.choice(valid))
            else:
                action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)
            progress = env.i / max(env.end_idx, 1)
            agent.memory.push(state, action, reward, next_state, done, regime=info.get("regime", "normal"))
            state = next_state
            ep_reward += reward
            total_steps += 1
            if len(agent.memory) >= batch_size:
                for _ in range(updates_per_step):
                    stats = agent.update(batch_size=batch_size)
                    if stats:
                        last_stats = stats
        logger.info(
            "Ep %04d | PnL:%6.1f%% Tr:%4d WR:%4.0f%% AvgRew:%+.4f | buf:%d | alpha:%.4f | Ent:%.3f",
            ep,
            (env.balance - 1.0) * 100.0,
            env.trades,
            (env.wins / max(env.trades, 1)) * 100.0,
            ep_reward / max(env.i - env.start_idx, 1),
            len(agent.memory),
            float(last_stats.get("alpha", agent.alpha)),
            float(last_stats.get("entropy", 0.0)),
        )
        if ep % 10 == 0:
            res = evaluate(agent, val_env)
            trade_pen = max(0.0, res["trades"] - 400) * 0.03
            score = res["pnl"] - 0.5 * abs(min(res["mdd"], 0.0)) - trade_pen
            logger.info(
                "    [VAL] PnL:%.2f%% | Tr:%4d | WR:%2.0f%% | MDD:%.2f%% | L:%4d S:%4d | FCL:%3d FCS:%3d | Score:%.2f",
                res["pnl"], res["trades"], res["wr"], res["mdd"], res["longs"], res["shorts"], res["fcl"], res["fcs"], score,
            )
            if res["trades"] >= MIN_VAL_TRADES_FOR_BEST and res["pnl"] > 0.0 and score > best:
                best = score
                bad_val = 0
                torch.save(
                    {
                        "state_dim": EDITOR_STATE_DIM,
                        "state_schema": EDITOR_STATE_SCHEMA,
                        "actor": agent.actor.state_dict(),
                        "critic": agent.critic.state_dict(),
                    },
                    best_path,
                )
                logger.info("    [NEW BEST] saved | score=%.2f | pnl=%.2f%%", score, res["pnl"])
            else:
                bad_val += 1
            logger.info("    [VAL CTRL] bad_val=%d best=%.2f", bad_val, best if best > -1e17 else float("nan"))
            if bad_val >= EARLY_STOP_PATIENCE:
                logger.info("    [EARLY STOP] no valid positive val result for %d evals", bad_val)
                break


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train RL unified trade editor")
    p.add_argument("--csv-path", default=CSV_DEFAULT)
    p.add_argument("--startup-check-only", action="store_true")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--device", default="auto")
    p.add_argument("--fresh-start", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    startup_check(args.csv_path)
    if not args.startup_check_only:
        train(csv_path=args.csv_path, episodes=args.episodes, device=args.device)
