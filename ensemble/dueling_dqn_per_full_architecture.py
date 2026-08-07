from __future__ import annotations

from collections import namedtuple
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


Experience = namedtuple(
    "Experience",
    [
        "state",
        "action",
        "reward",
        "next_state",
        "done",
        "position_side",
        "next_position_side",
        "bc_action",
    ],
)


class StateBuilder:
    """Build state vectors from market features and live position context."""

    def __init__(self, feature_cols: Sequence[str], max_hold_bars: int = 48) -> None:
        self.feature_cols = list(feature_cols)
        self.max_hold_bars = int(max_hold_bars)
        self.state_dim = int(len(self.feature_cols) + 7)

    def build(self, row: pd.Series, position: Mapping[str, float]) -> np.ndarray:
        market = row.reindex(self.feature_cols).to_numpy(dtype=np.float32, copy=True)
        market = np.where(np.isfinite(market), market, 0.0).astype(np.float32)
        context = np.asarray(
            [
                float(position.get("side", 0.0)),
                float(position.get("unrealized_pnl", 0.0)),
                float(position.get("hold_bars", 0.0)) / max(float(self.max_hold_bars), 1.0),
                float(position.get("entry_price_dist", 0.0)),
                float(position.get("drawdown_from_peak", 0.0)),
                float(position.get("bars_since_last_trade", 0.0)) / 50.0,
                float(position.get("daily_trade_count", 0.0)) / 20.0,
            ],
            dtype=np.float32,
        )
        return np.concatenate([market, context]).astype(np.float32, copy=False)


class ConditionedStateBuilder:
    """Build separated market/regime/context tensors for a single conditioned agent."""

    def __init__(
        self,
        market_feature_cols: Sequence[str],
        regime_feature_cols: Sequence[str],
        max_hold_bars: int = 48,
    ) -> None:
        self.market_feature_cols = list(market_feature_cols)
        self.regime_feature_cols = list(regime_feature_cols)
        self.max_hold_bars = int(max_hold_bars)
        self.market_dim = int(len(self.market_feature_cols))
        self.regime_dim = int(len(self.regime_feature_cols))
        self.context_dim = 7
        self.state_dim = int(self.market_dim + self.regime_dim + self.context_dim)

    def build_parts(
        self,
        row: pd.Series,
        position: Mapping[str, float],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        market = row.reindex(self.market_feature_cols).to_numpy(dtype=np.float32, copy=True)
        market = np.where(np.isfinite(market), market, 0.0).astype(np.float32)
        regime = row.reindex(self.regime_feature_cols).to_numpy(dtype=np.float32, copy=True)
        regime = np.where(np.isfinite(regime), regime, 0.0).astype(np.float32)
        regime_sum = float(np.sum(np.clip(regime, 0.0, None)))
        if regime_sum > 0.0:
            regime = regime / regime_sum
        context = np.asarray(
            [
                float(position.get("side", 0.0)),
                float(position.get("unrealized_pnl", 0.0)),
                float(position.get("hold_bars", 0.0)) / max(float(self.max_hold_bars), 1.0),
                float(position.get("entry_price_dist", 0.0)),
                float(position.get("drawdown_from_peak", 0.0)),
                float(position.get("bars_since_last_trade", 0.0)) / 50.0,
                float(position.get("daily_trade_count", 0.0)) / 20.0,
            ],
            dtype=np.float32,
        )
        return market, regime.astype(np.float32, copy=False), context

    def build(self, row: pd.Series, position: Mapping[str, float]) -> np.ndarray:
        market, regime, context = self.build_parts(row, position)
        return np.concatenate([market, regime, context]).astype(np.float32, copy=False)


class InputNormalizer:
    """Fit on train data only, then transform market features without lookahead."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.median: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None

    def fit(self, x_train: np.ndarray) -> "InputNormalizer":
        x = np.asarray(x_train, dtype=np.float32)
        median = np.nanmedian(np.where(np.isfinite(x), x, np.nan), axis=0).astype(np.float32)
        median = np.where(np.isfinite(median), median, 0.0).astype(np.float32)
        clean = np.where(np.isfinite(x), x, median)
        mean = clean.mean(axis=0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = clean.std(axis=0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > self.eps), std, 1.0).astype(np.float32)
        self.median = median
        self.mean = mean
        self.std = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.median is None or self.mean is None or self.std is None:
            raise RuntimeError("InputNormalizer must be fit before transform")
        arr = np.asarray(x, dtype=np.float32)
        arr = np.where(np.isfinite(arr), arr, self.median)
        return ((arr - self.mean) / self.std).astype(np.float32)


class PositionTracker:
    """Track live position state for state construction and environment updates."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.side = 0
        self.entry_price = 0.0
        self.peak_price = 0.0
        self.hold_bars = 0
        self.bars_since_last = 0
        self.daily_trade_count = 0
        self._last_day: Any = None

    def _roll_day(self, timestamp: Any) -> None:
        if timestamp is None:
            return
        day = pd.Timestamp(timestamp).date()
        if self._last_day is None:
            self._last_day = day
            return
        if day != self._last_day:
            self._last_day = day
            self.daily_trade_count = 0

    def step_market(self, current_price: float, timestamp: Any = None) -> None:
        self._roll_day(timestamp)
        if self.side != 0:
            self.hold_bars += 1
            if self.side > 0:
                self.peak_price = max(float(self.peak_price), float(current_price))
            else:
                if self.peak_price == 0.0:
                    self.peak_price = float(current_price)
                self.peak_price = min(float(self.peak_price), float(current_price))
        self.bars_since_last += 1

    def apply_action(self, action: int, current_price: float) -> None:
        if action == ActionSpace.OPEN_LONG:
            self.side = 1
            self.entry_price = float(current_price)
            self.peak_price = float(current_price)
            self.hold_bars = 0
            self.bars_since_last = 0
            self.daily_trade_count += 1
            return
        if action == ActionSpace.OPEN_SHORT:
            self.side = -1
            self.entry_price = float(current_price)
            self.peak_price = float(current_price)
            self.hold_bars = 0
            self.bars_since_last = 0
            self.daily_trade_count += 1
            return
        if action == ActionSpace.CLOSE:
            self.side = 0
            self.entry_price = 0.0
            self.peak_price = 0.0
            self.hold_bars = 0
            self.bars_since_last = 0
            self.daily_trade_count += 1

    def to_dict(self, current_price: float) -> dict[str, float]:
        if self.side == 0 or self.entry_price <= 0.0:
            return {
                "side": 0.0,
                "unrealized_pnl": 0.0,
                "hold_bars": 0.0,
                "entry_price_dist": 0.0,
                "drawdown_from_peak": 0.0,
                "bars_since_last_trade": float(self.bars_since_last),
                "daily_trade_count": float(self.daily_trade_count),
            }
        ret = (float(current_price) / max(float(self.entry_price), 1e-12) - 1.0) * float(self.side)
        peak_ret = (float(self.peak_price) / max(float(self.entry_price), 1e-12) - 1.0) * float(self.side)
        drawdown = ret - peak_ret
        return {
            "side": float(self.side),
            "unrealized_pnl": float(np.clip(ret, -0.1, 0.1)),
            "hold_bars": float(self.hold_bars),
            "entry_price_dist": float(np.clip(float(current_price) / max(float(self.entry_price), 1e-12) - 1.0, -0.05, 0.05)),
            "drawdown_from_peak": float(np.clip(drawdown, -0.05, 0.0)),
            "bars_since_last_trade": float(self.bars_since_last),
            "daily_trade_count": float(self.daily_trade_count),
        }


class ActionSpace:
    FLAT = 0
    OPEN_LONG = 1
    OPEN_SHORT = 2
    CLOSE = 3
    HOLD_LONG = 4
    HOLD_SHORT = 5
    N_ACTIONS = 6

    VALID_ACTIONS = {
        "flat": [FLAT, OPEN_LONG, OPEN_SHORT],
        "long": [CLOSE, HOLD_LONG],
        "short": [CLOSE, HOLD_SHORT],
    }
    EXPLORATION_WEIGHTS = {
        "flat": {FLAT: 0.85, OPEN_LONG: 0.075, OPEN_SHORT: 0.075},
        "long": {CLOSE: 0.10, HOLD_LONG: 0.90},
        "short": {CLOSE: 0.10, HOLD_SHORT: 0.90},
    }

    @staticmethod
    def side_key(position_side: int) -> str:
        if int(position_side) > 0:
            return "long"
        if int(position_side) < 0:
            return "short"
        return "flat"

    @staticmethod
    def get_mask(position_side: int) -> torch.Tensor:
        mask = torch.full((ActionSpace.N_ACTIONS,), float("-inf"))
        for action in ActionSpace.VALID_ACTIONS[ActionSpace.side_key(position_side)]:
            mask[action] = 0.0
        return mask

    @staticmethod
    def batch_mask(position_sides: Sequence[int]) -> torch.Tensor:
        return torch.stack([ActionSpace.get_mask(int(side)) for side in position_sides], dim=0)

    @staticmethod
    def configure_exploration(
        *,
        flat_weight: float,
        open_long_weight: float,
        open_short_weight: float,
        close_long_weight: float,
        close_short_weight: float,
    ) -> None:
        flat = max(float(flat_weight), 0.0)
        open_long = max(float(open_long_weight), 0.0)
        open_short = max(float(open_short_weight), 0.0)
        close_long = max(float(close_long_weight), 0.0)
        close_short = max(float(close_short_weight), 0.0)
        ActionSpace.EXPLORATION_WEIGHTS = {
            "flat": {ActionSpace.FLAT: flat, ActionSpace.OPEN_LONG: open_long, ActionSpace.OPEN_SHORT: open_short},
            "long": {ActionSpace.CLOSE: close_long, ActionSpace.HOLD_LONG: max(1.0 - close_long, 0.0)},
            "short": {ActionSpace.CLOSE: close_short, ActionSpace.HOLD_SHORT: max(1.0 - close_short, 0.0)},
        }

    @staticmethod
    def sample_exploration_action(position_side: int) -> int:
        valid = ActionSpace.VALID_ACTIONS[ActionSpace.side_key(position_side)]
        weights = ActionSpace.EXPLORATION_WEIGHTS[ActionSpace.side_key(position_side)]
        probs = np.asarray([weights[a] for a in valid], dtype=np.float64)
        if float(probs.sum()) <= 1e-12:
            probs = np.ones_like(probs)
        probs = probs / max(float(probs.sum()), 1e-12)
        return int(np.random.choice(valid, p=probs))

    @staticmethod
    def apply(action: int, position_side: int, *, min_hold_bars: int, hold_bars: int) -> int:
        if int(position_side) != 0 and int(hold_bars) < int(min_hold_bars) and int(action) == ActionSpace.CLOSE:
            return ActionSpace.HOLD_LONG if int(position_side) > 0 else ActionSpace.HOLD_SHORT
        return int(action)

    @staticmethod
    def next_side(position_side: int, action: int) -> int:
        if int(action) == ActionSpace.OPEN_LONG:
            return 1
        if int(action) == ActionSpace.OPEN_SHORT:
            return -1
        if int(action) == ActionSpace.CLOSE:
            return 0
        return int(position_side)


class DuelingDQN(nn.Module):
    def __init__(
        self,
        state_dim: int = 94,
        action_dim: int = 6,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.backbone = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden1)),
            nn.LayerNorm(int(hidden1)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.LayerNorm(int(hidden2)),
            nn.SiLU(),
        )
        value_mid = max(int(hidden2) // 2, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(int(hidden2), value_mid),
            nn.SiLU(),
            nn.Linear(value_mid, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(int(hidden2), value_mid),
            nn.SiLU(),
            nn.Linear(value_mid, int(action_dim)),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)

    def forward(self, state: torch.Tensor, action_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.backbone(state)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        if action_mask is not None:
            q = q + action_mask
        return q

    @torch.no_grad()
    def select_action(
        self,
        state: np.ndarray,
        *,
        position_side: int,
        epsilon: float = 0.05,
        temperature: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> int:
        return self.select_train_action(
            state,
            position_side=position_side,
            epsilon=epsilon,
            device=device,
        )

    @torch.no_grad()
    def select_train_action(
        self,
        state: np.ndarray,
        *,
        position_side: int,
        epsilon: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> int:
        if float(np.random.random()) < float(epsilon):
            return ActionSpace.sample_exploration_action(position_side)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask = ActionSpace.get_mask(position_side).to(device=device).unsqueeze(0)
        q = self.forward(state_t, action_mask=mask)
        return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def select_inference_action(
        self,
        state: np.ndarray,
        *,
        position_side: int,
        temperature: float = 0.5,
        stochastic: bool = False,
        device: torch.device | str = "cpu",
    ) -> int:
        state_t = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        mask = ActionSpace.get_mask(position_side).to(device=device).unsqueeze(0)
        q = self.forward(state_t, action_mask=mask)
        probs = F.softmax(q / max(float(temperature), 1e-4), dim=1)
        if stochastic:
            return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())
        return int(torch.argmax(probs, dim=1).item())


class ConditionedDuelingDQN(nn.Module):
    """Single Dueling DQN conditioned on explicit regime probabilities."""

    def __init__(
        self,
        market_dim: int = 87,
        regime_dim: int = 4,
        context_dim: int = 7,
        action_dim: int = 6,
        regime_hidden: int = 16,
        hidden1: int = 256,
        hidden2: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.market_dim = int(market_dim)
        self.regime_dim = int(regime_dim)
        self.context_dim = int(context_dim)
        self.action_dim = int(action_dim)
        self.regime_encoder = nn.Sequential(
            nn.Linear(self.regime_dim, int(regime_hidden)),
            nn.SiLU(),
        )
        total_dim = self.market_dim + int(regime_hidden) + self.context_dim
        self.backbone = nn.Sequential(
            nn.Linear(total_dim, int(hidden1)),
            nn.LayerNorm(int(hidden1)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden1), int(hidden2)),
            nn.LayerNorm(int(hidden2)),
            nn.SiLU(),
        )
        value_mid = max(int(hidden2) // 2, 1)
        self.value_stream = nn.Sequential(
            nn.Linear(int(hidden2), value_mid),
            nn.SiLU(),
            nn.Linear(value_mid, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(int(hidden2), value_mid),
            nn.SiLU(),
            nn.Linear(value_mid, self.action_dim),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2.0))
                nn.init.zeros_(module.bias)

    def forward(
        self,
        market: torch.Tensor,
        regime_probs: torch.Tensor,
        context: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        regime_emb = self.regime_encoder(regime_probs)
        x = torch.cat([market, regime_emb, context], dim=1)
        x = self.backbone(x)
        value = self.value_stream(x)
        advantage = self.advantage_stream(x)
        q = value + (advantage - advantage.mean(dim=1, keepdim=True))
        if action_mask is not None:
            q = q + action_mask
        return q

    @torch.no_grad()
    def select_action(
        self,
        market: np.ndarray,
        regime_probs: np.ndarray,
        context: np.ndarray,
        *,
        position_side: int,
        epsilon: float = 0.05,
        temperature: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> int:
        return self.select_train_action(
            market,
            regime_probs,
            context,
            position_side=position_side,
            epsilon=epsilon,
            device=device,
        )

    @torch.no_grad()
    def select_train_action(
        self,
        market: np.ndarray,
        regime_probs: np.ndarray,
        context: np.ndarray,
        *,
        position_side: int,
        epsilon: float = 0.05,
        device: torch.device | str = "cpu",
    ) -> int:
        if float(np.random.random()) < float(epsilon):
            return ActionSpace.sample_exploration_action(position_side)
        market_t = torch.as_tensor(market, dtype=torch.float32, device=device).unsqueeze(0)
        regime_t = torch.as_tensor(regime_probs, dtype=torch.float32, device=device).unsqueeze(0)
        context_t = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        mask = ActionSpace.get_mask(position_side).to(device=device).unsqueeze(0)
        q = self.forward(market_t, regime_t, context_t, action_mask=mask)
        return int(torch.argmax(q, dim=1).item())

    @torch.no_grad()
    def select_inference_action(
        self,
        market: np.ndarray,
        regime_probs: np.ndarray,
        context: np.ndarray,
        *,
        position_side: int,
        temperature: float = 0.5,
        stochastic: bool = False,
        device: torch.device | str = "cpu",
    ) -> int:
        market_t = torch.as_tensor(market, dtype=torch.float32, device=device).unsqueeze(0)
        regime_t = torch.as_tensor(regime_probs, dtype=torch.float32, device=device).unsqueeze(0)
        context_t = torch.as_tensor(context, dtype=torch.float32, device=device).unsqueeze(0)
        mask = ActionSpace.get_mask(position_side).to(device=device).unsqueeze(0)
        q = self.forward(market_t, regime_t, context_t, action_mask=mask)
        probs = F.softmax(q / max(float(temperature), 1e-4), dim=1)
        if stochastic:
            return int(torch.multinomial(probs.squeeze(0), num_samples=1).item())
        return int(torch.argmax(probs, dim=1).item())


class SumTree:
    def __init__(self, capacity: int) -> None:
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * int(capacity) - 1, dtype=np.float64)
        self.data = np.empty(int(capacity), dtype=object)
        self.n_entries = 0
        self.write = 0

    def _propagate(self, idx: int, change: float) -> None:
        parent = (int(idx) - 1) // 2
        self.tree[parent] += float(change)
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        left = 2 * int(idx) + 1
        right = left + 1
        if left >= len(self.tree):
            return int(idx)
        if float(s) <= float(self.tree[left]):
            return self._retrieve(left, s)
        return self._retrieve(right, float(s) - float(self.tree[left]))

    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data: Experience) -> None:
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float) -> None:
        change = float(priority) - float(self.tree[int(idx)])
        self.tree[int(idx)] = float(priority)
        self._propagate(int(idx), change)

    def get(self, s: float) -> tuple[int, float, Experience]:
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return int(idx), float(self.tree[idx]), self.data[data_idx]


class PrioritizedReplayBuffer:
    def __init__(
        self,
        capacity: int = 200_000,
        alpha: float = 0.65,
        beta: float = 0.40,
        beta_max: float = 1.00,
        eps: float = 1e-6,
    ) -> None:
        self.tree = SumTree(int(capacity))
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta_start = float(beta)
        self.beta = float(beta)
        self.beta_max = float(beta_max)
        self.eps = float(eps)
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        position_side: int,
        next_position_side: int,
        bc_action: int = -1,
    ) -> None:
        clean_reward = float(np.clip(float(reward) if np.isfinite(float(reward)) else 0.0, -5.0, 5.0))
        exp = Experience(state, int(action), clean_reward, next_state, bool(done), int(position_side), int(next_position_side), int(bc_action))
        priority = float(self.max_priority) ** self.alpha
        if not np.isfinite(priority) or priority <= 0.0:
            priority = 1.0
        self.tree.add(priority, exp)

    def sample(self, batch_size: int) -> tuple[list[Experience], list[int], torch.Tensor]:
        batch: list[Experience] = []
        indices: list[int] = []
        weights: list[float] = []
        total_raw = self.tree.total()
        total = max(float(total_raw) if np.isfinite(total_raw) else 0.0, self.eps)
        segment = total / max(int(batch_size), 1)
        n = max(self.tree.n_entries, 1)
        for i in range(int(batch_size)):
            a = segment * float(i)
            b = segment * float(i + 1)
            s = float(np.random.uniform(a, b))
            idx, priority, exp = self.tree.get(s)
            prob = max(float(priority) / total, self.eps)
            weight = (n * prob) ** (-self.beta)
            batch.append(exp)
            indices.append(idx)
            weights.append(weight)
        w = np.asarray(weights, dtype=np.float32)
        w = np.where(np.isfinite(w), w, 1.0).astype(np.float32)
        w = w / max(float(np.max(w)), 1e-12)
        return batch, indices, torch.as_tensor(w, dtype=torch.float32)

    def update_priorities(self, indices: Sequence[int], td_errors: Sequence[float]) -> None:
        for idx, td_err in zip(indices, td_errors):
            err = abs(float(td_err)) if np.isfinite(float(td_err)) else 0.0
            priority = (err + self.eps) ** self.alpha
            priority = float(np.clip(priority, self.eps, 1e6))
            self.max_priority = max(float(self.max_priority), float(priority))
            self.tree.update(int(idx), float(priority))

    def anneal_beta(self, current_step: int, total_steps: int) -> None:
        frac = float(np.clip(float(current_step) / max(float(total_steps), 1.0), 0.0, 1.0))
        self.beta = min(self.beta_max, self.beta_start + (self.beta_max - self.beta_start) * frac)


class RewardFunction:
    def __init__(
        self,
        fee_rate: float = 0.0005,
        min_hold_bars: int = 6,
        max_hold_bars: int = 48,
        hold_penalty: float = 0.001,
        trade_penalty: float = 0.010,
        regime_bonus: float = 0.003,
        reward_scale: float = 10.0,
        edge_bonus: float = 0.20,
        flat_opportunity_penalty: float = 0.12,
        opportunity_threshold: float = 0.08,
    ) -> None:
        self.fee_rate = float(fee_rate)
        self.min_hold_bars = int(min_hold_bars)
        self.max_hold_bars = int(max_hold_bars)
        self.hold_penalty = float(hold_penalty)
        self.trade_penalty = float(trade_penalty)
        self.regime_bonus = float(regime_bonus)
        self.reward_scale = float(reward_scale)
        self.edge_bonus = float(edge_bonus)
        self.flat_opportunity_penalty = float(flat_opportunity_penalty)
        self.opportunity_threshold = float(opportunity_threshold)

    def compute(
        self,
        *,
        action: int,
        position: Mapping[str, float],
        current_price: float,
        next_price: float,
        regime: str,
        daily_trades: int,
        ev_long: float,
        ev_short: float,
    ) -> float:
        reward = 0.0
        side = float(position.get("side", 0.0))
        long_edge = max(float(ev_long), 0.0)
        short_edge = max(float(ev_short), 0.0)
        best_edge = max(long_edge, short_edge)
        if side != 0.0:
            step_ret = (float(next_price) / max(float(current_price), 1e-12) - 1.0) * side
            reward += step_ret * self.reward_scale
            aligned_edge = long_edge if side > 0.0 else short_edge
            adverse_edge = short_edge if side > 0.0 else long_edge
            reward += self.edge_bonus * float(np.clip(aligned_edge - adverse_edge, -1.0, 1.0))
        elif int(action) == ActionSpace.FLAT and best_edge > self.opportunity_threshold:
            reward -= self.flat_opportunity_penalty * float(np.clip(best_edge - self.opportunity_threshold, 0.0, 1.0))
        if int(action) == ActionSpace.OPEN_LONG:
            reward -= self.fee_rate * self.reward_scale
            reward += self.edge_bonus * float(np.clip(long_edge - short_edge, -1.0, 1.0))
            if float(ev_long) > 0.005:
                reward += self.regime_bonus * self.reward_scale
            elif float(ev_long) < -0.005:
                reward -= self.regime_bonus * self.reward_scale
        elif int(action) == ActionSpace.OPEN_SHORT:
            reward -= self.fee_rate * self.reward_scale
            reward += self.edge_bonus * float(np.clip(short_edge - long_edge, -1.0, 1.0))
            if float(ev_short) > 0.005:
                reward += self.regime_bonus * self.reward_scale
            elif float(ev_short) < -0.005:
                reward -= self.regime_bonus * self.reward_scale
        elif int(action) == ActionSpace.CLOSE:
            reward -= self.fee_rate * self.reward_scale
            if float(position.get("hold_bars", 0.0)) < float(self.min_hold_bars):
                remaining = (float(self.min_hold_bars) - float(position.get("hold_bars", 0.0))) / max(float(self.min_hold_bars), 1.0)
                reward -= self.hold_penalty * self.reward_scale * float(np.clip(remaining, 0.0, 1.0))
        if int(daily_trades) > 8:
            reward -= self.trade_penalty * float(int(daily_trades) - 8) * 10.0
        if str(regime) == "bear" and side > 0.0:
            reward -= 1.0
        if str(regime) == "bull" and side < 0.0:
            reward -= 1.0
        return float(np.clip(reward, -5.0, 5.0))


def normalize_batch_rewards(rewards: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize sampled rewards inside a batch after storage-time clipping."""

    rewards = torch.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)
    centered = rewards - rewards.median()
    std = torch.std(centered, unbiased=False)
    if not torch.isfinite(std) or float(std.detach().cpu()) < float(eps):
        return centered
    return centered / (std + float(eps))


@dataclass
class DQNTrainerConfig:
    state_dim: int = 94
    action_dim: int = 6
    lr: float = 5e-4
    gamma: float = 0.92
    batch_size: int = 256
    target_update: int = 200
    grad_clip: float = 3.0
    buffer_size: int = 200_000
    per_alpha: float = 0.65
    per_beta: float = 0.40
    per_beta_max: float = 1.0
    normalize_rewards: bool = True
    min_buffer_size: int = 10_000
    bc_weight: float = 0.0


@dataclass
class ConditionedDQNTrainerConfig(DQNTrainerConfig):
    market_dim: int = 87
    regime_dim: int = 4
    context_dim: int = 7
    regime_hidden: int = 16


class DQNTrainer:
    def __init__(self, cfg: DQNTrainerConfig | None = None, *, device: torch.device | str | None = None) -> None:
        self.cfg = cfg or DQNTrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = DuelingDQN(self.cfg.state_dim, self.cfg.action_dim).to(self.device)
        self.target = DuelingDQN(self.cfg.state_dim, self.cfg.action_dim).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(self.cfg.lr))
        self.buffer = PrioritizedReplayBuffer(
            capacity=int(self.cfg.buffer_size),
            alpha=float(self.cfg.per_alpha),
            beta=float(self.cfg.per_beta),
            beta_max=float(self.cfg.per_beta_max),
        )
        self.step_count = 0

    def push_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        position_side: int,
        next_position_side: int,
        bc_action: int = -1,
    ) -> None:
        self.buffer.add(state, action, reward, next_state, done, position_side, next_position_side, bc_action)

    def train_step(self, total_steps: int) -> float | None:
        if self.buffer.tree.n_entries < max(int(self.cfg.batch_size) * 4, int(self.cfg.min_buffer_size)):
            return None
        self.buffer.anneal_beta(self.step_count, total_steps)
        batch, indices, is_weights = self.buffer.sample(int(self.cfg.batch_size))
        states = torch.as_tensor(np.asarray([e.state for e in batch], dtype=np.float32), device=self.device)
        actions = torch.as_tensor([e.action for e in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.asarray([e.next_state for e in batch], dtype=np.float32), device=self.device)
        dones = torch.as_tensor([float(e.done) for e in batch], dtype=torch.float32, device=self.device)
        curr_sides = [int(e.position_side) for e in batch]
        next_sides = [int(e.next_position_side) for e in batch]
        is_weights = is_weights.to(self.device)

        curr_masks = ActionSpace.batch_mask(curr_sides).to(self.device)
        next_masks = ActionSpace.batch_mask(next_sides).to(self.device)
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(self.cfg.normalize_rewards):
            rewards = normalize_batch_rewards(rewards)
        else:
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        with torch.no_grad():
            next_q_online = self.online(next_states, next_masks)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target(next_states, next_masks)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + float(self.cfg.gamma) * next_q * (1.0 - dones)

        q_values = self.online(states, curr_masks)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = torch.nan_to_num((td_target - q_pred).detach().abs(), nan=0.0, posinf=1e6, neginf=0.0).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        td_loss = (is_weights * F.smooth_l1_loss(q_pred, td_target.detach(), reduction="none")).mean()
        bc_actions = torch.as_tensor([int(getattr(e, "bc_action", -1)) for e in batch], dtype=torch.int64, device=self.device)
        bc_mask = bc_actions >= 0
        if bool(torch.any(bc_mask)) and float(self.cfg.bc_weight) > 0.0:
            td_loss = td_loss + float(self.cfg.bc_weight) * F.cross_entropy(q_values[bc_mask], bc_actions[bc_mask])
        if not torch.isfinite(td_loss):
            return None
        self.optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), float(self.cfg.grad_clip))
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % int(self.cfg.target_update) == 0:
            self._soft_update(tau=0.005)
        return float(td_loss.detach().cpu())

    def _soft_update(self, tau: float = 0.005) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(float(tau) * online_param.data + (1.0 - float(tau)) * target_param.data)


class ConditionedDQNTrainer:
    """Single-agent DQN trainer with explicit regime conditioning."""

    def __init__(
        self,
        cfg: ConditionedDQNTrainerConfig | None = None,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        self.cfg = cfg or ConditionedDQNTrainerConfig()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.online = ConditionedDuelingDQN(
            market_dim=self.cfg.market_dim,
            regime_dim=self.cfg.regime_dim,
            context_dim=self.cfg.context_dim,
            action_dim=self.cfg.action_dim,
            regime_hidden=self.cfg.regime_hidden,
        ).to(self.device)
        self.target = ConditionedDuelingDQN(
            market_dim=self.cfg.market_dim,
            regime_dim=self.cfg.regime_dim,
            context_dim=self.cfg.context_dim,
            action_dim=self.cfg.action_dim,
            regime_hidden=self.cfg.regime_hidden,
        ).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(self.cfg.lr))
        self.buffer = PrioritizedReplayBuffer(
            capacity=int(self.cfg.buffer_size),
            alpha=float(self.cfg.per_alpha),
            beta=float(self.cfg.per_beta),
            beta_max=float(self.cfg.per_beta_max),
        )
        self.step_count = 0

    def _split_state(self, states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = int(self.cfg.market_dim)
        r = int(self.cfg.regime_dim)
        c = int(self.cfg.context_dim)
        market = states[:, :m]
        regime = states[:, m : m + r]
        context = states[:, m + r : m + r + c]
        return market, regime, context

    def push_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        position_side: int,
        next_position_side: int,
        bc_action: int = -1,
    ) -> None:
        self.buffer.add(state, action, reward, next_state, done, position_side, next_position_side, bc_action)

    def train_step(self, total_steps: int) -> float | None:
        if self.buffer.tree.n_entries < max(int(self.cfg.batch_size) * 4, int(self.cfg.min_buffer_size)):
            return None
        self.buffer.anneal_beta(self.step_count, total_steps)
        batch, indices, is_weights = self.buffer.sample(int(self.cfg.batch_size))
        states = torch.as_tensor(np.asarray([e.state for e in batch], dtype=np.float32), device=self.device)
        actions = torch.as_tensor([e.action for e in batch], dtype=torch.int64, device=self.device)
        rewards = torch.as_tensor([e.reward for e in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.asarray([e.next_state for e in batch], dtype=np.float32), device=self.device)
        dones = torch.as_tensor([float(e.done) for e in batch], dtype=torch.float32, device=self.device)
        curr_sides = [int(e.position_side) for e in batch]
        next_sides = [int(e.next_position_side) for e in batch]
        curr_masks = ActionSpace.batch_mask(curr_sides).to(self.device)
        next_masks = ActionSpace.batch_mask(next_sides).to(self.device)
        is_weights = is_weights.to(self.device)
        states = torch.nan_to_num(states, nan=0.0, posinf=0.0, neginf=0.0)
        next_states = torch.nan_to_num(next_states, nan=0.0, posinf=0.0, neginf=0.0)
        if bool(self.cfg.normalize_rewards):
            rewards = normalize_batch_rewards(rewards)
        else:
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=5.0, neginf=-5.0).clamp(-5.0, 5.0)

        market, regime, context = self._split_state(states)
        next_market, next_regime, next_context = self._split_state(next_states)

        with torch.no_grad():
            next_q_online = self.online(next_market, next_regime, next_context, action_mask=next_masks)
            next_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.target(next_market, next_regime, next_context, action_mask=next_masks)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            td_target = rewards + float(self.cfg.gamma) * next_q * (1.0 - dones)

        q_values = self.online(market, regime, context, action_mask=curr_masks)
        q_pred = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        td_errors = torch.nan_to_num((td_target - q_pred).detach().abs(), nan=0.0, posinf=1e6, neginf=0.0).cpu().numpy()
        self.buffer.update_priorities(indices, td_errors)

        td_loss = (is_weights * F.smooth_l1_loss(q_pred, td_target.detach(), reduction="none")).mean()
        bc_actions = torch.as_tensor([int(getattr(e, "bc_action", -1)) for e in batch], dtype=torch.int64, device=self.device)
        bc_mask = bc_actions >= 0
        if bool(torch.any(bc_mask)) and float(self.cfg.bc_weight) > 0.0:
            td_loss = td_loss + float(self.cfg.bc_weight) * F.cross_entropy(q_values[bc_mask], bc_actions[bc_mask])
        if not torch.isfinite(td_loss):
            return None
        self.optimizer.zero_grad(set_to_none=True)
        td_loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), float(self.cfg.grad_clip))
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % int(self.cfg.target_update) == 0:
            self._soft_update(tau=0.005)
        return float(td_loss.detach().cpu())

    def _soft_update(self, tau: float = 0.005) -> None:
        for target_param, online_param in zip(self.target.parameters(), self.online.parameters()):
            target_param.data.copy_(float(tau) * online_param.data + (1.0 - float(tau)) * target_param.data)


class TradingDQNPipeline:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: Sequence[str],
        regime_col: str,
        *,
        total_episodes: int = 300,
        trainer_cfg: DQNTrainerConfig | None = None,
        reward_fn: RewardFunction | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.regime_col = str(regime_col)
        self.state_builder = StateBuilder(feature_cols)
        self.position = PositionTracker()
        self.reward_fn = reward_fn or RewardFunction()
        cfg = trainer_cfg or DQNTrainerConfig(state_dim=self.state_builder.state_dim)
        cfg.state_dim = self.state_builder.state_dim
        self.trainer = DQNTrainer(cfg)
        self.total_episodes = int(total_episodes)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995

    def run(self) -> tuple[list[float], list[float]]:
        epsilon = float(self.epsilon_start)
        total_steps = max(len(self.df) - 1, 1) * max(self.total_episodes, 1)
        losses: list[float] = []
        episode_rewards: list[float] = []
        for episode in range(self.total_episodes):
            self.position.reset()
            episode_reward = 0.0
            daily_trades = 0
            for t in range(len(self.df) - 1):
                row = self.df.iloc[t]
                next_row = self.df.iloc[t + 1]
                self.position.step_market(float(row["close"]), row.get("timestamp"))
                pos_dict = self.position.to_dict(float(row["close"]))
                state = self.state_builder.build(row, pos_dict)
                raw_action = self.trainer.online.select_action(
                    state,
                    position_side=int(pos_dict["side"]),
                    epsilon=float(epsilon),
                    temperature=0.50,
                    device=self.trainer.device,
                )
                action = ActionSpace.apply(
                    raw_action,
                    int(pos_dict["side"]),
                    min_hold_bars=int(self.reward_fn.min_hold_bars),
                    hold_bars=int(pos_dict["hold_bars"]),
                )
                reward = self.reward_fn.compute(
                    action=action,
                    position=pos_dict,
                    current_price=float(row["close"]),
                    next_price=float(next_row["close"]),
                    regime=str(row.get(self.regime_col, "")),
                    daily_trades=int(daily_trades),
                    ev_long=float(row.get("ev_long", 0.0)),
                    ev_short=float(row.get("ev_short", 0.0)),
                )
                self.position.apply_action(int(action), float(next_row["close"]))
                next_pos = self.position.to_dict(float(next_row["close"]))
                next_state = self.state_builder.build(next_row, next_pos)
                done = bool(t == len(self.df) - 2)
                next_side = ActionSpace.next_side(int(pos_dict["side"]), int(action))
                self.trainer.push_experience(state, int(action), float(reward), next_state, done, int(pos_dict["side"]), int(next_side))
                loss = self.trainer.train_step(total_steps)
                if loss is not None:
                    losses.append(float(loss))
                episode_reward += float(reward)
                if int(action) in (ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT, ActionSpace.CLOSE):
                    daily_trades += 1
            epsilon = max(float(self.epsilon_end), float(epsilon) * float(self.epsilon_decay))
            episode_rewards.append(float(episode_reward))
            if episode % 10 == 0:
                tail = float(np.mean(losses[-100:])) if losses else 0.0
                print(
                    f"EP {episode:4d} | Reward: {episode_reward:8.2f} | Loss: {tail:.4f} | "
                    f"eps: {epsilon:.3f} | beta: {self.trainer.buffer.beta:.3f}"
                )
        return losses, episode_rewards


class SingleAgentTradingDQNPipeline:
    """Train one conditioned DQN agent instead of regime specialists."""

    def __init__(
        self,
        df: pd.DataFrame,
        market_feature_cols: Sequence[str],
        regime_feature_cols: Sequence[str],
        regime_col: str,
        *,
        total_episodes: int = 300,
        trainer_cfg: ConditionedDQNTrainerConfig | None = None,
        reward_fn: RewardFunction | None = None,
    ) -> None:
        self.df = df.reset_index(drop=True).copy()
        self.regime_col = str(regime_col)
        self.state_builder = ConditionedStateBuilder(
            market_feature_cols=market_feature_cols,
            regime_feature_cols=regime_feature_cols,
        )
        self.position = PositionTracker()
        self.reward_fn = reward_fn or RewardFunction()
        cfg = trainer_cfg or ConditionedDQNTrainerConfig(
            state_dim=self.state_builder.state_dim,
            market_dim=self.state_builder.market_dim,
            regime_dim=self.state_builder.regime_dim,
            context_dim=self.state_builder.context_dim,
        )
        cfg.state_dim = self.state_builder.state_dim
        cfg.market_dim = self.state_builder.market_dim
        cfg.regime_dim = self.state_builder.regime_dim
        cfg.context_dim = self.state_builder.context_dim
        self.trainer = ConditionedDQNTrainer(cfg)
        self.total_episodes = int(total_episodes)
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.epsilon_decay = 0.995

    def run(self) -> tuple[list[float], list[float]]:
        epsilon = float(self.epsilon_start)
        total_steps = max(len(self.df) - 1, 1) * max(self.total_episodes, 1)
        losses: list[float] = []
        episode_rewards: list[float] = []
        for episode in range(self.total_episodes):
            self.position.reset()
            episode_reward = 0.0
            daily_trades = 0
            for t in range(len(self.df) - 1):
                row = self.df.iloc[t]
                next_row = self.df.iloc[t + 1]
                self.position.step_market(float(row["close"]), row.get("timestamp"))
                pos_dict = self.position.to_dict(float(row["close"]))
                market, regime_probs, context = self.state_builder.build_parts(row, pos_dict)
                state = np.concatenate([market, regime_probs, context]).astype(np.float32, copy=False)
                raw_action = self.trainer.online.select_action(
                    market,
                    regime_probs,
                    context,
                    position_side=int(pos_dict["side"]),
                    epsilon=float(epsilon),
                    temperature=0.50,
                    device=self.trainer.device,
                )
                action = ActionSpace.apply(
                    raw_action,
                    int(pos_dict["side"]),
                    min_hold_bars=int(self.reward_fn.min_hold_bars),
                    hold_bars=int(pos_dict["hold_bars"]),
                )
                reward = self.reward_fn.compute(
                    action=action,
                    position=pos_dict,
                    current_price=float(row["close"]),
                    next_price=float(next_row["close"]),
                    regime=str(row.get(self.regime_col, "")),
                    daily_trades=int(daily_trades),
                    ev_long=float(row.get("ev_long", 0.0)),
                    ev_short=float(row.get("ev_short", 0.0)),
                )
                self.position.apply_action(int(action), float(next_row["close"]))
                next_pos = self.position.to_dict(float(next_row["close"]))
                next_market, next_regime_probs, next_context = self.state_builder.build_parts(next_row, next_pos)
                next_state = np.concatenate([next_market, next_regime_probs, next_context]).astype(np.float32, copy=False)
                done = bool(t == len(self.df) - 2)
                next_side = ActionSpace.next_side(int(pos_dict["side"]), int(action))
                self.trainer.push_experience(state, int(action), float(reward), next_state, done, int(pos_dict["side"]), int(next_side))
                loss = self.trainer.train_step(total_steps)
                if loss is not None:
                    losses.append(float(loss))
                episode_reward += float(reward)
                if int(action) in (ActionSpace.OPEN_LONG, ActionSpace.OPEN_SHORT, ActionSpace.CLOSE):
                    daily_trades += 1
            epsilon = max(float(self.epsilon_end), float(epsilon) * float(self.epsilon_decay))
            episode_rewards.append(float(episode_reward))
            if episode % 10 == 0:
                tail = float(np.mean(losses[-100:])) if losses else 0.0
                print(
                    f"EP {episode:4d} | Reward: {episode_reward:8.2f} | Loss: {tail:.4f} | "
                    f"eps: {epsilon:.3f} | beta: {self.trainer.buffer.beta:.3f}"
                )
        return losses, episode_rewards


class HMMSpecialistEnsemble:
    def __init__(self, specialists: Mapping[str, DQNTrainer], hmm_threshold: float = 0.65) -> None:
        self.specialists = dict(specialists)
        self.hmm_threshold = float(hmm_threshold)

    @torch.no_grad()
    def predict(self, state: np.ndarray, hmm_probs: Mapping[str, float], position_side: int) -> torch.Tensor:
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        mask = ActionSpace.get_mask(position_side).unsqueeze(0)
        max_regime = max(hmm_probs, key=hmm_probs.get)
        max_prob = float(hmm_probs[max_regime])
        if max_prob >= float(self.hmm_threshold):
            return self.specialists[max_regime].online(state_t, mask).squeeze(0)
        q = None
        for regime, prob in hmm_probs.items():
            q_reg = self.specialists[regime].online(state_t, mask)
            q = q_reg * float(prob) if q is None else q + q_reg * float(prob)
        assert q is not None
        return q.squeeze(0)


__all__ = [
    "ActionSpace",
    "ConditionedDQNTrainer",
    "ConditionedDQNTrainerConfig",
    "ConditionedDuelingDQN",
    "ConditionedStateBuilder",
    "DQNTrainer",
    "DQNTrainerConfig",
    "DuelingDQN",
    "Experience",
    "HMMSpecialistEnsemble",
    "InputNormalizer",
    "PositionTracker",
    "PrioritizedReplayBuffer",
    "RewardFunction",
    "SingleAgentTradingDQNPipeline",
    "StateBuilder",
    "SumTree",
    "TradingDQNPipeline",
    "normalize_batch_rewards",
]
