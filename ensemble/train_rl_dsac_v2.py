#!/usr/bin/env python3
"""
DSAC V2 코인 트레이딩 에이전트 (개선된 Distributional Soft Actor-Critic)
=========================================================================
train_rl_dsac_agent.py + train_rl_sac_agent.py 기반 완전 독립 파일.

개선 사항:
  1. 보상 함수 개선
       r6 진입 비용 패널티:  -0.01  → -0.002  (진입 과도 억제 해소)
       r4 시간 감쇠 임계값:  12봉   → 24봉    (정상 보유 패널티 제거)
       r3 품질 보상:          sqrt 스케일 + force_close -0.30→-0.20 + 손실 -0.05→-0.08

  2. RunningRewardNorm (Welford 온라인 알고리즘 보상 정규화)

  3. Prioritized Experience Replay (PER)
       SumTree + PrioritizedReplayBuffer (alpha=0.6, beta 0.4 → 1.0)

  4. N-Step Returns (n=3) — 진입→수익 크레딧 할당 개선

  5. CompactFeatureExtractorV2 — 잔차 연결(Residual Connection)

  6. n_quantiles: 32 → 64  (분포 추정 해상도 2배)

  7. 하이퍼파라미터: episodes=1500, early_stop_patience=20, min_buffer=8192
"""

import copy
import gc
import logging
import os
import random
import argparse
import sys
import warnings
from collections import deque
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import Normal

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, "ensemble"), os.path.join(_ROOT_DIR, "strategies")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ─── 유틸리티 / 환경 임포트 (기존 파일에서 재활용) ──────────────────────────
from ensemble.train_rl_agent import (  # noqa: E402
    MultiTimeframeFeatures,
    OnlineHMMDetector,
    REGIME_COLS,
    STATE_CONF,
    STATE_PRED,
)
from ensemble.train_rl_dsac_agent import (  # noqa: E402
    DSAC_STATE_DIM,
    DSACCompactTradingEnv as _BaseDSACEnv,
    _safe_float,
    _sigmoid,
    _norm_tanh,
    _pick_first,
    _normalize_prob3,
    _prob_entropy_norm,
    _quantile_huber_loss,
    _POS_THRESH,
    _CLOSE_THRESH,
    LOG_STD_MIN,
    LOG_STD_MAX,
)

# State dim (V2도 동일한 26D compact state 사용)
DSAC_V2_STATE_DIM = DSAC_STATE_DIM  # 26


# ═══════════════════════════════════════════════════════════════════════════
# 1. DSACCompactTradingEnvV2 — 보상 함수(step) 개선
# ═══════════════════════════════════════════════════════════════════════════
class DSACCompactTradingEnvV2(_BaseDSACEnv):
    """개선된 보상 함수를 가진 DSAC compact 환경 (V2).

    _BaseDSACEnv의 __init__ / _build_state / _get_stacked_state 를 그대로 쓰고,
    step() 내 r3 / r4 / r6만 수정한다.
    """

    def step(self, action: float):  # noqa: C901
        """SACTradingEnv.step() 와 동일하나 r3/r4/r6 개선."""
        action = float(np.clip(action, -1.0, 1.0))
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step

        prev_portfolio_value = (
            self.balance * (1.0 + self.unrealized_pnl)
            if self.pos is not None
            else self.balance
        )

        abs_action = abs(action)
        leverage_rate = abs_action

        # ── 강제 청산 조건 ──
        force_close = (self.pos is not None and self.unrealized_pnl <= -0.025)

        is_entering_long = is_entering_short = is_closing = is_adjusting = False

        if force_close:
            is_closing = True
        elif self.pos is None:
            if action > _POS_THRESH:
                is_entering_long = True
            elif action < -_POS_THRESH:
                is_entering_short = True
        else:
            if abs_action < _CLOSE_THRESH:
                is_closing = True
            elif self.pos == "LONG" and action < -_POS_THRESH:
                is_closing = True
            elif self.pos == "SHORT" and action > _POS_THRESH:
                is_closing = True
            else:
                is_adjusting = True

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close

        # ── 거래 실행 ──
        if is_entering_long:
            self.pos = "LONG"
            self.entry_price = fill_price * (1 + self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_entering_short:
            self.pos = "SHORT"
            self.entry_price = fill_price * (1 - self.slip)
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_adjusting and self.pos is not None:
            lev_delta = abs(leverage_rate - self.current_leverage)
            if lev_delta > 0.05:
                self.balance -= self.balance * self.fee * lev_delta
                self.current_leverage = leverage_rate
        elif is_closing and self.pos is not None:
            base_balance = self.balance
            if self.pos == "LONG":
                realized_pnl = (fill_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - fill_price * (1 + self.slip)) / self.entry_price
            realized_pnl *= self.current_leverage
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self.pos = None
            self.current_leverage = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        # ── 스텝 전진 ──
        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == "LONG":
                raw_pnl = (next_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                raw_pnl = (self.entry_price - next_price * (1 + self.slip)) / self.entry_price
            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        # ── 보상 계산 ──
        cur_port = (
            self.balance * (1.0 + self.unrealized_pnl)
            if self.pos is not None
            else self.balance
        )
        step_delta = (cur_port - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -0.01:
            dd_ratio = abs(self.unrealized_pnl) / 0.025
            r2_drawdown = -0.1 * (dd_ratio ** 2)

        # ★ r3 개선: sqrt 스케일, force_close -0.30→-0.20, 손실 -0.05→-0.08
        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.20
            elif self._last_realized_pnl > 0:
                r3_quality = 0.20 * min(float(np.sqrt(self._last_realized_pnl / 0.005)), 1.0)
            else:
                r3_quality = -0.08

        # ★ r4 개선: 임계값 12봉 → 24봉
        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > 24:
            r4_time_decay = -0.003 * (self.hold_count - 24) / 72.0

        r5_idle = 0.0
        if self.pos is None:
            regime_step = min(max(decision_step, 0), len(self._feat_np) - 1)
            regime_raw = self._feat_np[regime_step]
            o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
            regime_vec = regime_raw[o: o + self._n_regime]
            regime_idx = int(np.argmax(regime_vec))
            if regime_idx in (2, 3):
                r5_idle = -0.003
            elif regime_idx in (0, 1):
                r5_idle = -0.0003
            else:
                r5_idle = -0.001

        # ★ r6 개선: -0.01 → -0.005 (과도한 거래 억제)
        r6_trade_cost = 0.0
        if is_entering_long or is_entering_short:
            r6_trade_cost = -0.005 * leverage_rate

        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost
        reward = float(np.tanh(raw_reward))

        # ── 에피소드 종료 시 강제 청산 ──
        if done and self.pos is not None:
            base_balance = self.balance
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            if self.pos == "LONG":
                ep_realized = (ep_end_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            terminal_r = float(np.tanh(ep_realized * 50.0))
            if ep_realized > 0:
                terminal_r += 0.20 * min(float(np.sqrt(ep_realized / 0.005)), 1.0)
            else:
                terminal_r -= 0.08
            reward = float(np.tanh(raw_reward + terminal_r))
            self.pos = None

        info = {
            "pnl_pct": (self.balance / self.initial_balance - 1) * 100,
            "wr": self.win_trades / max(1, self.total_trades),
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info


# ═══════════════════════════════════════════════════════════════════════════
# 2. RunningRewardNorm — Welford 온라인 보상 정규화
# ═══════════════════════════════════════════════════════════════════════════
class RunningRewardNorm:
    """Welford 온라인 알고리즘 기반 보상 정규화.

    처음 100 샘플은 정규화 없이 원본 보상 그대로 사용.
    이후 (x - mean) / std 로 정규화 후 [-clip, clip] 클립.
    """

    def __init__(self, clip: float = 5.0):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.clip = float(clip)

    def update(self, x: float) -> float:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)
        if not self.ready:
            return float(x)
        std = float(np.sqrt(self.M2 / max(self.n - 1, 1)))
        normed = (x - self.mean) / max(std, 1e-8)
        return float(np.clip(normed, -self.clip, self.clip))

    @property
    def ready(self) -> bool:
        return self.n >= 100


# ═══════════════════════════════════════════════════════════════════════════
# 3. Prioritized Experience Replay — SumTree + PrioritizedReplayBuffer
# ═══════════════════════════════════════════════════════════════════════════
class SumTree:
    """O(log n) 우선순위 샘플링을 위한 Sum-Segment Tree."""

    def __init__(self, capacity: int):
        self.capacity = int(capacity)
        self._tree = np.zeros(2 * self.capacity, dtype=np.float64)  # 세그먼트 트리
        self._data_ptr = 0  # 다음 쓰기 위치 (data/leaf 인덱스)

    # ── 내부 헬퍼 ──────────────────────────────────────────────────────────
    def _propagate(self, leaf_idx: int, delta: float):
        """리프에서 루트까지 변경량 전파."""
        idx = leaf_idx  # tree 인덱스
        while idx > 1:
            idx //= 2
            self._tree[idx] += delta

    def _retrieve(self, idx: int, value: float) -> int:
        """value에 해당하는 리프 tree 인덱스 반환."""
        while True:
            left = 2 * idx
            if left >= 2 * self.capacity:
                return idx
            if value <= self._tree[left]:
                idx = left
            else:
                value -= self._tree[left]
                idx = left + 1

    # ── 공개 메서드 ────────────────────────────────────────────────────────
    @property
    def total(self) -> float:
        return float(self._tree[1])

    def add(self, priority: float, data_idx: int):
        """data_idx 위치에 priority 저장."""
        leaf_pos = self._data_ptr + self.capacity  # tree 인덱스
        delta = float(priority) - self._tree[leaf_pos]
        self._tree[leaf_pos] = float(priority)
        self._propagate(leaf_pos, delta)
        self._data_ptr = (self._data_ptr + 1) % self.capacity

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """n개 샘플: (data_indices, tree_indices, priorities)."""
        segment = self.total / n
        data_idxs = np.empty(n, dtype=np.int64)
        tree_idxs = np.empty(n, dtype=np.int64)
        priorities = np.empty(n, dtype=np.float64)
        for i in range(n):
            lo = segment * i
            hi = segment * (i + 1)
            v = np.random.uniform(lo, hi)
            tree_idx = self._retrieve(1, v)
            data_idx = tree_idx - self.capacity
            data_idxs[i] = int(data_idx % self.capacity)
            tree_idxs[i] = tree_idx
            priorities[i] = max(self._tree[tree_idx], 1e-12)
        return data_idxs, tree_idxs, priorities

    def update(self, tree_indices: np.ndarray, priorities: np.ndarray):
        """배치 우선순위 갱신."""
        for ti, p in zip(tree_indices, priorities):
            ti = int(ti)
            delta = float(p) - self._tree[ti]
            self._tree[ti] = float(p)
            self._propagate(ti, delta)


class PrioritizedReplayBuffer:
    """TD-error 기반 Prioritized Experience Replay 버퍼.

    Args:
        capacity:   최대 저장 경험 수
        alpha:      우선순위 지수 (0=균일, 1=완전 우선순위)
        beta:       중요도 샘플링 초기값
        beta_end:   beta 최종값 (어닐링)
        beta_steps: beta 어닐링 스텝 수
        eps:        최소 우선순위 (zero-priority 방지)
    """

    def __init__(
        self,
        capacity: int = 500_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_end: float = 1.0,
        beta_steps: int = 200_000,
        eps: float = 1e-5,
    ):
        self._cap = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self._beta_end = float(beta_end)
        self._beta_steps = int(beta_steps)
        self._beta_inc = (beta_end - beta) / max(beta_steps, 1)
        self.eps = float(eps)

        self._tree = SumTree(self._cap)
        self._ptr = 0
        self._size = 0

        self._s: np.ndarray | None = None
        self._a = np.empty(self._cap, np.float32)
        self._r = np.empty(self._cap, np.float32)
        self._ns: np.ndarray | None = None
        self._d = np.empty(self._cap, np.bool_)
        self._max_priority: float = 1.0

    def push(self, state, action, reward, next_state, done, td_error: float = 1.0):
        if self._s is None:
            sdim = len(state)
            self._s = np.empty((self._cap, sdim), np.float32)
            self._ns = np.empty((self._cap, sdim), np.float32)
        p = self._ptr
        self._s[p] = state
        self._a[p] = float(action)
        self._r[p] = float(reward)
        self._ns[p] = next_state
        self._d[p] = bool(done)

        priority = (abs(float(td_error)) + self.eps) ** self.alpha
        priority = max(priority, self._max_priority)
        self._tree.add(priority, p)
        self._max_priority = max(self._max_priority, priority)

        self._ptr = (p + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size: int):
        """(s, a, r, ns, d, weights, tree_indices) 반환."""
        data_idxs, tree_idxs, priorities = self._tree.sample(batch_size)

        # 중요도 샘플링 가중치
        probs = priorities / max(self._tree.total, 1e-12)
        weights = (self._size * probs) ** (-self.beta)
        weights = (weights / max(weights.max(), 1e-12)).astype(np.float32)

        s = self._s[data_idxs]
        a = self._a[data_idxs]
        r = self._r[data_idxs]
        ns = self._ns[data_idxs]
        d = self._d[data_idxs].astype(np.float32)
        return s, a, r, ns, d, weights, tree_idxs

    def update_priorities(self, tree_indices: np.ndarray, td_errors: np.ndarray):
        priorities = (np.abs(td_errors) + self.eps) ** self.alpha
        self._tree.update(tree_indices, priorities)
        self._max_priority = max(self._max_priority, float(priorities.max()))

    def anneal_beta(self, step: int):
        """학습 스텝마다 호출 — beta를 beta_end까지 선형 증가."""
        self.beta = min(self._beta_end, self.beta + self._beta_inc)

    def __len__(self) -> int:
        return self._size


# ═══════════════════════════════════════════════════════════════════════════
# 4. NStepBuffer — n-step 리턴 계산
# ═══════════════════════════════════════════════════════════════════════════
class NStepBuffer:
    """n-step 리턴을 계산해 PrioritizedReplayBuffer 에 저장.

    에피소드 루프에서 env.step() 직후 push() 를 호출하면,
    n 스텝 후에 자동으로 replay_buffer.push() 가 실행된다.
    """

    def __init__(self, n: int = 3, gamma: float = 0.99):
        self.n = int(n)
        self.gamma = float(gamma)
        self._buf: deque[tuple] = deque(maxlen=n)  # (s, a, r, done) 임시 저장

    def push(
        self,
        s: np.ndarray,
        a: float,
        r: float,
        ns: np.ndarray,
        done: bool,
        replay_buffer: PrioritizedReplayBuffer,
    ) -> None:
        self._buf.append((s, a, r, done))

        if len(self._buf) == self.n:
            s0, a0, _, _ = self._buf[0]
            r_nstep = sum(self.gamma ** i * self._buf[i][2] for i in range(self.n))
            done_n = any(self._buf[i][3] for i in range(self.n))
            replay_buffer.push(s0, a0, r_nstep, ns, done_n)

        if done and len(self._buf) > 0:
            # 에피소드 종료 시 남은 trailing transitions 처리
            remaining = list(self._buf)
            n_remaining = len(remaining)
            for start in range(n_remaining):
                if start == 0 and n_remaining == self.n:
                    continue  # 이미 위에서 처리함
                s_i, a_i, _, _ = remaining[start]
                r_nstep = sum(
                    self.gamma ** j * remaining[start + j][2]
                    for j in range(n_remaining - start)
                )
                replay_buffer.push(s_i, a_i, r_nstep, ns, True)
            self._buf.clear()

    def reset(self):
        self._buf.clear()


# ═══════════════════════════════════════════════════════════════════════════
# 5. 신경망 — CompactFeatureExtractorV2 (잔차 연결)
# ═══════════════════════════════════════════════════════════════════════════
class CompactFeatureExtractorV2(nn.Module):
    """DSAC V2 전용 MLP 인코더 — 잔차 연결(Residual Connection) 추가.

    기존 직렬 LayerNorm+SiLU MLP 대비 gradient vanishing 개선,
    수렴 속도 향상.
    """

    def __init__(self, state_dim: int = DSAC_V2_STATE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.input_proj = nn.Linear(state_dim, hidden_dim)
        self.block1 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.block2 = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(state)       # [B, hidden_dim]
        x = x + self.block1(x)          # 잔차 연결 1
        x = x + self.block2(x)          # 잔차 연결 2
        return x


class GaussianActorV2(nn.Module):
    """V2 compact state → action ∈ [-1, +1] (tanh squashed Gaussian)."""

    def __init__(self, state_dim: int = DSAC_V2_STATE_DIM, hidden_dim: int = 256):
        super().__init__()
        self.feat = CompactFeatureExtractorV2(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor):
        feat = self.feat(state)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state: torch.Tensor):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state: torch.Tensor) -> torch.Tensor:
        mu, _ = self.forward(state)
        return torch.tanh(mu)


class DistributionalTwinCriticV2(nn.Module):
    """V2 Twin Critic — CompactFeatureExtractorV2 + n_quantiles=64."""

    def __init__(
        self,
        state_dim: int = DSAC_V2_STATE_DIM,
        hidden_dim: int = 256,
        n_quantiles: int = 64,
    ):
        super().__init__()
        self.n_quantiles = int(n_quantiles)

        self.feat1 = CompactFeatureExtractorV2(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

        self.feat2 = CompactFeatureExtractorV2(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.n_quantiles),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        x1 = torch.cat([f1, action], dim=1)
        x2 = torch.cat([f2, action], dim=1)
        return self.q1(x1), self.q2(x2)  # [B, N], [B, N]


# ═══════════════════════════════════════════════════════════════════════════
# 6. DSACAgentV2 — PER + 중요도 샘플링
# ═══════════════════════════════════════════════════════════════════════════
class DSACAgentV2:
    """Distributional Soft Actor-Critic V2 (PER + CVaR + Residual Network)."""

    def __init__(
        self,
        state_dim: int = DSAC_V2_STATE_DIM,
        hidden_dim: int = 256,
        lr_actor: float = 3e-4,
        lr_critic: float = 3e-4,
        lr_alpha: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        n_quantiles: int = 64,
        cvar_frac: float = 0.25,
        device: str = "cuda",
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        per_beta_steps: int = 200_000,
    ):
        self.device = device
        self.gamma = float(gamma)
        self.tau = float(tau)
        self.n_quantiles = int(n_quantiles)
        self.cvar_frac = float(cvar_frac)
        self.update_count = 0

        self.actor = GaussianActorV2(state_dim, hidden_dim).to(device)
        self.critic = DistributionalTwinCriticV2(state_dim, hidden_dim, self.n_quantiles).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.taus = torch.linspace(
            0.5 / self.n_quantiles,
            1.0 - 0.5 / self.n_quantiles,
            self.n_quantiles,
            device=device,
            dtype=torch.float32,
        )

        self.memory = PrioritizedReplayBuffer(
            capacity=500_000,
            alpha=per_alpha,
            beta=per_beta,
            beta_steps=per_beta_steps,
        )

    @property
    def alpha(self) -> float:
        return float(self.log_alpha.exp().item())

    def act(self, state: np.ndarray, deterministic: bool = False) -> float:
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_ts)
            else:
                action, _ = self.actor.sample(state_ts)
        return float(action.cpu().item())

    def _target_quantiles(
        self, ns: torch.Tensor, r: torch.Tensor, d: torch.Tensor
    ) -> torch.Tensor:
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns)
            tq1, tq2 = self.critic_target(ns, next_action)

            tq1_m = tq1.mean(dim=1, keepdim=True)
            tq2_m = tq2.mean(dim=1, keepdim=True)
            chosen_tq = torch.where(tq1_m <= tq2_m, tq1, tq2)

            entropy_term = self.log_alpha.exp().detach() * next_log_prob
            target_q = r + self.gamma * (1.0 - d) * (chosen_tq - entropy_term)
            return target_q

    def _cvar_min(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        k = max(1, int(self.n_quantiles * self.cvar_frac))
        q1_s, _ = torch.sort(q1, dim=1)
        q2_s, _ = torch.sort(q2, dim=1)
        c1 = q1_s[:, :k].mean(dim=1, keepdim=True)
        c2 = q2_s[:, :k].mean(dim=1, keepdim=True)
        return torch.min(c1, c2)

    def update(self, batch_size: int = 256) -> dict:
        if len(self.memory) < batch_size:
            return {}

        s, a, r, ns, d, weights, tree_idx = self.memory.sample(batch_size)

        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.FloatTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        w  = torch.FloatTensor(weights).unsqueeze(1).to(self.device)  # [B,1]

        target_q = self._target_quantiles(ns, r, d)  # [B, N]

        q1, q2 = self.critic(s, a)
        # ★ 중요도 샘플링 가중치를 critic loss에 적용
        loss_q1 = _quantile_huber_loss(q1, target_q, self.taus)
        loss_q2 = _quantile_huber_loss(q2, target_q, self.taus)
        critic_loss = (w * (loss_q1 + loss_q2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ★ TD error 계산 후 우선순위 업데이트
        with torch.no_grad():
            td_errors = (
                target_q.mean(dim=1) - q1.mean(dim=1)
            ).abs().cpu().numpy()
        self.memory.update_priorities(tree_idx, td_errors)
        self.memory.anneal_beta(self.update_count)
        self.update_count += 1

        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_cvar = self._cvar_min(q1_new, q2_new)
        alpha = self.log_alpha.exp().detach()
        actor_loss = (alpha * log_prob - q_cvar).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1.0 - self.tau) * tp.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss":  float(actor_loss.item()),
            "alpha":       float(self.log_alpha.exp().item()),
            "mean_q":      float(torch.min(q1_new.mean(dim=1), q2_new.mean(dim=1)).mean().item()),
            "cvar_q":      float(q_cvar.mean().item()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 7. DSACRouterV2 — 라이브 추론 (GaussianActorV2 사용)
# ═══════════════════════════════════════════════════════════════════════════
class DSACRouterV2:
    """V2 라이브 추론 라우터.

    기존 DSACRouter 와 동일한 인터페이스(decide), GaussianActorV2 사용.
    """

    def __init__(self, actor: GaussianActorV2, device: str = "cuda"):
        self.actor = actor
        self.device = device
        self._prev_close: float | None = None
        self._ret_hist: deque[float] = deque(maxlen=64)
        self._spread_hist: deque[float] = deque(maxlen=64)

    # _build_compact_state는 DSACRouter 와 완전히 동일하므로 원본에서 위임
    def _build_compact_state(self, features: dict[str, Any], pos: dict[str, Any]) -> np.ndarray:
        from ensemble.train_rl_dsac_agent import DSACRouter as _RefRouter
        _ref = _RefRouter.__new__(_RefRouter)
        _ref._prev_close = self._prev_close
        _ref._ret_hist = self._ret_hist
        _ref._spread_hist = self._spread_hist
        state = _ref._build_compact_state(features, pos)
        self._prev_close = _ref._prev_close
        self._ret_hist = _ref._ret_hist
        self._spread_hist = _ref._spread_hist
        return state

    def _state_tensor(self, features: dict, pos: dict) -> torch.Tensor:
        vec = self._build_compact_state(features or {}, pos or {})
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, features: dict, pos: dict) -> tuple[int, float, dict]:
        state = self._state_tensor(features, pos)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor.deterministic(state)
        action_val = float(action.cpu().item())
        abs_action = abs(action_val)

        cur_pos = pos.get("type") if isinstance(pos, dict) else None
        if cur_pos is not None:
            if abs_action < _CLOSE_THRESH:
                action_int, leverage = 0, 0.0
            elif cur_pos == "LONG" and action_val < -_POS_THRESH:
                action_int, leverage = 0, 0.0
            elif cur_pos == "SHORT" and action_val > _POS_THRESH:
                action_int, leverage = 0, 0.0
            else:
                action_int = 1 if cur_pos == "LONG" else 2
                leverage = abs_action
        else:
            if action_val > _POS_THRESH:
                action_int, leverage = 1, abs_action
            elif action_val < -_POS_THRESH:
                action_int, leverage = 2, abs_action
            else:
                action_int, leverage = 0, 0.0

        info = {
            "agent": "DSAC_V2",
            "raw_action": round(action_val, 4),
            "kelly": float(leverage),
            "long_edge": max(action_val, 0.0),
            "short_edge": max(-action_val, 0.0),
            "score": float(abs_action),
            "state_dim": DSAC_V2_STATE_DIM,
        }
        return action_int, leverage, info


# ═══════════════════════════════════════════════════════════════════════════
# 8. train_v2 — 학습 루프
# ═══════════════════════════════════════════════════════════════════════════
def train_v2(
    csv_path: str = "data/rl_training_data_full.csv",
    train_ratio: float = 0.8,
    episodes: int = 1500,
    fresh_start: bool = False,
    use_lr_scheduler: bool = True,
    lr_factor: float = 0.5,
    lr_patience: int = 3,
    lr_min: float = 1e-5,
    early_stop_patience: int = 20,
    val_interval: int = 10,
    n_step: int = 3,
):
    if not os.path.exists(csv_path):
        logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")
        return

    df = pd.read_csv(csv_path)
    logger.info("[DATA] csv_path=%s | rows=%d", csv_path, len(df))
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        if ts.notna().any():
            years = sorted(pd.Series(ts.dt.year.dropna().unique()).astype(int).tolist())
            logger.info("[DATA] ts_range=%s -> %s | years=%s", ts.min(), ts.max(), years)

    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val   = df.iloc[split_idx:].reset_index(drop=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)
    logger.info("DSAC V2 state dim: %d  |  n_quantiles: 64  |  n_step: %d", DSAC_V2_STATE_DIM, n_step)

    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    logger.info("[HMM] 초기 학습 완료.")

    logger.info("[MTF] 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train["close"].values.astype(np.float32))
    mtf_val   = MultiTimeframeFeatures(df_val["close"].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    train_hmm = copy.deepcopy(hmm_detector)
    env = DSACCompactTradingEnvV2(
        df_train, phase="train",
        hmm_detector=train_hmm, mtf_features=mtf_train,
    )
    agent = DSACAgentV2(
        DSAC_V2_STATE_DIM,
        hidden_dim=256,
        n_quantiles=64,
        cvar_frac=0.25,
        device=device,
        per_alpha=0.6,
        per_beta=0.4,
        per_beta_steps=200_000,
    )

    nep            = int(episodes)
    batch          = 256
    update_freq    = 4
    min_buffer     = 8192   # 4096 → 8192
    warmup_steps   = 10000
    global_step    = 0

    best_val_score = -float("inf")
    best_val_pnl   = -float("inf")
    bad_val_count  = 0

    nstep_buf   = NStepBuffer(n=n_step, gamma=agent.gamma)

    actor_scheduler = critic_scheduler = None
    if use_lr_scheduler:
        actor_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.actor_optimizer, mode="max", factor=float(lr_factor),
            patience=max(1, int(lr_patience)), min_lr=float(lr_min),
            threshold=1e-3, threshold_mode="rel",
        )
        critic_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            agent.critic_optimizer, mode="max", factor=float(lr_factor),
            patience=max(1, int(lr_patience)), min_lr=float(lr_min),
            threshold=1e-3, threshold_mode="rel",
        )

    logger.info(
        "[TRAIN CFG] val_interval=%d | lr_sched=%s (factor=%.3f patience=%d) "
        "| early_stop_patience=%d | n_step=%d | min_buffer=%d",
        val_interval, "ON" if use_lr_scheduler else "OFF",
        lr_factor, lr_patience, early_stop_patience, n_step, min_buffer,
    )

    os.makedirs("data/ensemble/ckpt", exist_ok=True)
    ckpt_path = "data/ensemble/ckpt/dsac_v2_checkpoint.pth"
    best_path = "data/ensemble/ckpt/best_dsac_v2_agents.pth"

    start_ep = 1
    if (not fresh_start) and os.path.exists(ckpt_path):
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            agent.actor.load_state_dict(ckpt["actor"])
            agent.critic.load_state_dict(ckpt["critic"])
            agent.critic_target.load_state_dict(ckpt["critic_target"])
            agent.log_alpha.data.copy_(ckpt["log_alpha"])
            agent.actor_optimizer.load_state_dict(ckpt["actor_opt"])
            agent.critic_optimizer.load_state_dict(ckpt["critic_opt"])
            agent.alpha_optimizer.load_state_dict(ckpt["alpha_opt"])
            global_step    = int(ckpt.get("global_step", 0))
            best_val_pnl   = float(ckpt.get("best_val_pnl", -float("inf")))
            best_val_score = float(ckpt.get("best_val_score", -float("inf")))
            bad_val_count  = int(ckpt.get("bad_val_count", 0))
            start_ep       = int(ckpt.get("epoch", 0)) + 1
            agent.update_count = int(ckpt.get("update_count", 0))
            logger.info(
                "♻️ [복원] ep=%d | global_step=%d | best_pnl=%.2f%%",
                start_ep - 1, global_step, best_val_pnl,
            )
            if use_lr_scheduler and actor_scheduler is not None:
                try:
                    if "actor_sched" in ckpt:
                        actor_scheduler.load_state_dict(ckpt["actor_sched"])
                    if "critic_sched" in ckpt:
                        critic_scheduler.load_state_dict(ckpt["critic_sched"])
                except Exception as e:
                    logger.warning("⚠️ LR scheduler 상태 복원 실패: %s", e)
        except Exception as e:
            logger.warning("⚠️ 체크포인트 복원 실패 (아키텍처 변경 가능, fresh start 권장): %s", e)

    # ── 버퍼 워밍업 ──────────────────────────────────────────────────────────
    if len(agent.memory) < min_buffer:
        refill_steps = max(warmup_steps, min_buffer)
        logger.info("[WARMUP] 버퍼 비어있음 → %d 스텝 랜덤 탐험으로 리필", refill_steps)
        warmup_env = DSACCompactTradingEnvV2(
            df_train, phase="train",
            hmm_detector=copy.deepcopy(hmm_detector), mtf_features=mtf_train,
        )
        ws = warmup_env.reset()
        w_nstep = NStepBuffer(n=n_step, gamma=agent.gamma)
        for _ in range(refill_steps):
            wa = np.random.uniform(-1.0, 1.0)
            wns, wr, wd, _ = warmup_env.step(wa)
            w_nstep.push(ws, wa, wr, wns, wd, agent.memory)
            ws = wns
            if wd:
                ws = warmup_env.reset()
                w_nstep.reset()
        logger.info("[WARMUP 완료] 버퍼: %d", len(agent.memory))

    def _save_checkpoint(ep: int):
        actor_sched_state  = actor_scheduler.state_dict()  if actor_scheduler  is not None else None
        critic_sched_state = critic_scheduler.state_dict() if critic_scheduler is not None else None
        torch.save(
            {
                "actor":          agent.actor.state_dict(),
                "critic":         agent.critic.state_dict(),
                "critic_target":  agent.critic_target.state_dict(),
                "log_alpha":      agent.log_alpha.data,
                "actor_opt":      agent.actor_optimizer.state_dict(),
                "critic_opt":     agent.critic_optimizer.state_dict(),
                "alpha_opt":      agent.alpha_optimizer.state_dict(),
                "global_step":    global_step,
                "best_val_pnl":   best_val_pnl,
                "best_val_score": best_val_score,
                "bad_val_count":  bad_val_count,
                "epoch":          ep,
                "update_count":   agent.update_count,
                "state_dim":      DSAC_V2_STATE_DIM,
                "actor_sched":    actor_sched_state,
                "critic_sched":   critic_sched_state,
            },
            ckpt_path,
        )

    ep = start_ep
    try:
        for ep in range(start_ep, nep + 1):
            state    = env.reset()
            ep_reward = 0.0
            done     = False
            last_stats: dict = {}
            nstep_buf.reset()

            while not done:
                global_step += 1

                if global_step < warmup_steps:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    action = agent.act(state, deterministic=False)

                next_state, reward, done, _ = env.step(action)
                ep_reward += reward

                # ★ N-Step Buffer 를 통해 메모리에 저장
                nstep_buf.push(state, action, reward, next_state, done, agent.memory)
                state = next_state

                if global_step % update_freq == 0 and len(agent.memory) >= min_buffer:
                    last_stats = agent.update(batch)

            pnl  = (env.balance / env.initial_balance - 1.0) * 100.0
            _cvar = float(last_stats.get("cvar_q", 0.0))
            logger.info(
                "Ep %04d | PnL:%6.1f%% Tr:%4d WR:%4.0f%% Rew:%7.3f | buf:%6d | α:%.4f | CVaR_Q:%+.4f",
                ep, pnl, env.total_trades, env.win_rate * 100,
                ep_reward, len(agent.memory), agent.alpha, _cvar,
            )

            if ep % max(1, int(val_interval)) == 0:
                val_hmm = copy.deepcopy(hmm_detector)
                # val 에피소드 길이를 train과 맞춤 (8192봉 이하)
                df_val_capped = df_val.iloc[:8192].reset_index(drop=True)
                val_env = DSACCompactTradingEnvV2(
                    df_val_capped, phase="val",
                    hmm_detector=val_hmm, mtf_features=mtf_val,
                )

                val_state = val_env.reset()
                val_done  = False
                val_peak_eq  = float(val_env.initial_balance)
                val_mdd_pct  = 0.0
                agent.actor.eval()
                while not val_done:
                    with torch.no_grad():
                        val_action = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, _ = val_env.step(val_action)
                    cur_eq = val_env.balance * (
                        1.0 + val_env.unrealized_pnl if val_env.pos is not None else 1.0
                    )
                    val_peak_eq = max(val_peak_eq, cur_eq)
                    val_mdd_pct = min(val_mdd_pct, (cur_eq / max(val_peak_eq, 1e-8) - 1.0) * 100.0)
                agent.actor.train()

                val_pnl = (val_env.balance / val_env.initial_balance - 1.0) * 100.0
                val_wr  = val_env.win_rate
                if val_env.total_trades == 0:
                    val_trade_score = -5.0
                elif val_pnl > 0:
                    val_trade_score = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    val_trade_score = -min(val_env.total_trades / 30.0, 1.0) * 10.0
                val_score = val_pnl * 3.0 + val_wr * 60.0 + val_trade_score + val_mdd_pct * 2.0

                logger.info(
                    "    [VAL] PnL:%6.2f%% | Tr:%4d | WR:%.0f%% | MDD:%.2f%% | Score:%.2f",
                    val_pnl, val_env.total_trades, val_wr * 100, val_mdd_pct, val_score,
                )

                improved = val_score > best_val_score
                if improved:
                    best_val_score, best_val_pnl = val_score, val_pnl
                    bad_val_count = 0
                    torch.save(
                        {
                            "actor":      agent.actor.state_dict(),
                            "critic":     agent.critic.state_dict(),
                            "best_pnl":   best_val_pnl,
                            "best_score": best_val_score,
                            "epoch":      ep,
                            "state_dim":  DSAC_V2_STATE_DIM,
                            "meta": {
                                "algo": "DSAC_V2",
                                "n_quantiles": agent.n_quantiles,
                                "cvar_frac": agent.cvar_frac,
                                "residual": True,
                                "per": True,
                                "n_step": n_step,
                            },
                        },
                        best_path,
                    )
                    logger.info("    🎉 [NEW BEST] 저장 완료 (PnL:%.2f%%)", best_val_pnl)
                else:
                    bad_val_count += 1

                if use_lr_scheduler and actor_scheduler is not None and critic_scheduler is not None:
                    prev_lr_a = float(agent.actor_optimizer.param_groups[0]["lr"])
                    prev_lr_c = float(agent.critic_optimizer.param_groups[0]["lr"])
                    actor_scheduler.step(val_score)
                    critic_scheduler.step(val_score)
                    new_lr_a = float(agent.actor_optimizer.param_groups[0]["lr"])
                    new_lr_c = float(agent.critic_optimizer.param_groups[0]["lr"])
                    if new_lr_a < prev_lr_a or new_lr_c < prev_lr_c:
                        logger.info(
                            "    📉 [LR DROP] actor %.3e→%.3e | critic %.3e→%.3e",
                            prev_lr_a, new_lr_a, prev_lr_c, new_lr_c,
                        )
                    else:
                        logger.info(
                            "    [LR] actor %.3e | critic %.3e | bad_val=%d",
                            new_lr_a, new_lr_c, bad_val_count,
                        )

                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    train_hmm.A     = hmm_detector.A.copy()
                    train_hmm.mu    = hmm_detector.mu.copy()
                    train_hmm.sigma = hmm_detector.sigma.copy()
                    train_hmm.pi    = hmm_detector.pi.copy()
                    train_hmm._obs_mean = hmm_detector._obs_mean.copy()
                    train_hmm._obs_std  = hmm_detector._obs_std.copy()
                    logger.info("    [HMM] 온라인 업데이트 완료")

                _save_checkpoint(ep)
                if int(early_stop_patience) > 0 and bad_val_count >= int(early_stop_patience):
                    logger.info(
                        "⏹️ [EARLY STOP] bad_val_count=%d >= patience=%d | "
                        "best_score=%.2f | best_pnl=%.2f%%",
                        bad_val_count, early_stop_patience, best_val_score, best_val_pnl,
                    )
                    break

            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단.")
        _save_checkpoint(ep)


# ═══════════════════════════════════════════════════════════════════════════
# 9. CLI
# ═══════════════════════════════════════════════════════════════════════════
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DSAC V2 agent (improved reward + PER + N-step)")
    p.add_argument("--csv-path",              default="data/rl_training_data_full.csv")
    p.add_argument("--train-ratio",           type=float, default=0.8)
    p.add_argument("--episodes",              type=int,   default=1500)
    p.add_argument("--fresh-start",           action="store_true")
    p.add_argument("--val-interval",          type=int,   default=10)
    p.add_argument("--no-lr-scheduler",       action="store_true")
    p.add_argument("--lr-factor",             type=float, default=0.5)
    p.add_argument("--lr-patience",           type=int,   default=3)
    p.add_argument("--lr-min",                type=float, default=1e-5)
    p.add_argument("--early-stop-patience",   type=int,   default=20)
    p.add_argument("--n-step",                type=int,   default=3,  help="N-step return 길이")
    p.add_argument(
        "--startup-check-only",
        action="store_true",
        help="임포트/인수 검증 후 즉시 종료",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_rl_dsac_v2")
        raise SystemExit(0)
    train_v2(
        csv_path=args.csv_path,
        train_ratio=args.train_ratio,
        episodes=args.episodes,
        fresh_start=args.fresh_start,
        use_lr_scheduler=not args.no_lr_scheduler,
        lr_factor=args.lr_factor,
        lr_patience=args.lr_patience,
        lr_min=args.lr_min,
        early_stop_patience=args.early_stop_patience,
        val_interval=args.val_interval,
        n_step=args.n_step,
    )
