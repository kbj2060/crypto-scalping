"""
SAC 코인 트레이딩 에이전트 (Soft Actor-Critic, Continuous Action)
================================================================
기존 IQN 6-Agent MoE 시스템의 피쳐/환경/융합모듈을 그대로 재활용하되,
이산 2-Action → 연속 1-Action (-1 ~ +1)으로 전환.

action 의미:
  -1.0 = 풀 숏 (100% 사이즈)
  -0.3 = 약한 숏 (30% 사이즈)
   0.0 = 관망 (무포지션)
  +0.3 = 약한 롱 (30% 사이즈)
  +1.0 = 풀 롱 (100% 사이즈)

장점:
  - "진입/유지/청산"이 action 크기의 연속적 변화로 자연스럽게 결정
  - 포지션 사이징이 action에 내장 → 별도 Kelly sizer 불필요
  - entropy regularization으로 레짐 변화에 로버스트
  - 2-Action의 Q(hold)≈Q(exit) 진동 문제가 원천적으로 없음

재활용: OnlineHMMDetector, MultiTimeframeFeatures, MarketAttentionEncoder
        TradingEnv._build_state, 5-Component 보상 함수, 피쳐/상수 정의
교체:   RobustIQN → SAC Actor + Twin Critic
        IQNAgent → SACAgent
        6-Agent MoE + GatingNet → 단일 SAC
        GatingRouter7 → SACRouter (라이브 추론용)
"""

import os, sys, logging, random, copy, gc, argparse
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, 'ensemble'), os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path: sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 기존 융합 모듈 import (train_rl_agent.py에서 재활용)
# ═══════════════════════════════════════════════════════════════════════════
from ensemble.train_rl_agent import (
    OnlineHMMDetector,
    MultiTimeframeFeatures,
    MarketAttentionEncoder,
    STATE_PRED, STATE_CONF, STATE_ELITE, STATE_ALPHA, STATE_SYNTH,
    REGIME_COLS,
    HMM_N_STATES, HMM_DIM, MTF_DIM,
    FEATURE_DIM, STATE_DIM, STACK_N, STACKED_STATE_DIM,
)

# ═══════════════════════════════════════════════════════════════════════════
# 1. SAC용 TradingEnv — 연속 action (-1 ~ +1)
# ═══════════════════════════════════════════════════════════════════════════
# 기존 TradingEnv의 _build_state, 보상 함수를 그대로 유지하되,
# step()의 action 해석만 연속값으로 변경

# 포지션 전환 임계값
_POS_THRESH = 0.15    # |action| > 이 값이면 포지션 진입/유지
_CLOSE_THRESH = 0.05  # |action| < 이 값이면 청산 (포지션 보유 중)


def _signed_log_return(entry_price: float, mark_price: float, side: str) -> float:
    entry = max(float(entry_price), 1e-8)
    mark = max(float(mark_price), 1e-8)
    log_change = float(np.log(mark / entry))
    return log_change if side == 'LONG' else -log_change

class SACTradingEnv:
    """SAC용 연속 action 트레이딩 환경.
    
    action ∈ [-1, +1]:
      action > +POS_THRESH  → LONG 진입/유지 (크기 = |action|)
      action < -POS_THRESH  → SHORT 진입/유지 (크기 = |action|)
      |action| ≤ CLOSE_THRESH → 청산 또는 관망
      그 사이 → 데드존 (현재 상태 유지)
    """

    def __init__(
        self,
        df,
        initial_balance=10000.0,
        fee=0.0005,
        slip=0.0002,
        phase='train',
        hmm_detector=None,
        mtf_features=None,
        side_mode='both',
        reward_beta=None,
        specialist_pos_thresh=None,
        specialist_close_thresh=None,
        specialist_min_opportunity_move=None,
        specialist_min_breakout=None,
        specialist_idle_penalty=None,
    ):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.hmm_detector = hmm_detector
        self.side_mode = "both"

        if mtf_features is not None:
            self.mtf = mtf_features
        else:
            self.mtf = MultiTimeframeFeatures(self.df['close'].values.astype(np.float32))

        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1

        feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
        feat_df = (
            self.df.reindex(columns=feat_cols, fill_value=0.0)
            .apply(pd.to_numeric, errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        self._feat_np  = feat_df.to_numpy(dtype=np.float32)
        self._close_np = self.df['close'].values.astype(np.float32)
        self._high_np = (
            self.df['high'].values.astype(np.float32)
            if 'high' in self.df.columns else self._close_np.copy()
        )
        self._low_np = (
            self.df['low'].values.astype(np.float32)
            if 'low' in self.df.columns else self._close_np.copy()
        )
        self._open_np  = (
            self.df['open'].values.astype(np.float32)
            if 'open' in self.df.columns else self._close_np.copy()
        )
        self._m7_tp_price_np = (
            pd.to_numeric(self.df.get('m7_tp_price', 0.0), errors='coerce')
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._m7_sl_price_np = (
            pd.to_numeric(self.df.get('m7_sl_price', 0.0), errors='coerce')
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._m7_target_hold_np = (
            pd.to_numeric(self.df.get('m7_target_hold', 0.0), errors='coerce')
            .replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
        )
        self._n_pred   = len(STATE_PRED)
        self._n_conf   = len(STATE_CONF)
        self._n_elite  = len(STATE_ELITE)
        self._n_alpha  = len(STATE_ALPHA)
        self._n_regime = len(REGIME_COLS)
        self._n_synth  = len(STATE_SYNTH)
        self._frame_stack = deque(maxlen=STACK_N)

        _hmm_cols = ['log_return', 'garch_vol_z', 'oi_change_rate']
        self._hmm_obs_np = {
            col: self.df[col].fillna(0).values.astype(np.float32)
            if col in self.df.columns else np.zeros(len(self.df), dtype=np.float32)
            for col in _hmm_cols
        }
        self._train_start_by_regime = self._build_train_start_buckets()
        self.reset()

    def _build_train_start_buckets(self):
        buckets = {k: [] for k in ['bull', 'bear', 'chop', 'whipsaw', 'normal']}
        if self.phase != 'train':
            return buckets
        max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        if max_start <= 0 or not all(c in self.df.columns for c in REGIME_COLS):
            return buckets
        regime_mat = self.df.loc[:max_start, REGIME_COLS].to_numpy(dtype=np.float32)
        for idx, row in enumerate(regime_mat):
            reg_i = int(np.argmax(row))
            reg_name = REGIME_COLS[reg_i].replace('regime_', '')
            if reg_name in buckets:
                buckets[reg_name].append(idx)
        return buckets

    def _sample_train_start(self, max_start: int) -> int:
        return random.randint(0, max_start)

    def reset(self, start_idx=None):
        if self.phase == 'train':
            max_start = max(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
            self.start_step = start_idx if start_idx is not None else random.randint(0, max_start)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = min(self.start_step + self.MAX_EPISODE_STEPS, len(self.df) - 1)

        self.balance = self.initial_balance
        self.pos = None          # None / 'LONG' / 'SHORT'
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.hold_count = 0

        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = False
        self._last_closed_side = ""
        self._last_closed_hold_count = 0

        if self.hmm_detector is not None:
            self.hmm_detector.reset_episode()

        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step))

    def step(self, action: float):
        """action ∈ [-1, +1] 연속값으로 포지션 결정.

        Returns: (next_state, reward, done, info)
        """
        action = float(np.clip(action, -1.0, 1.0))
        current_price = self._close_np[self.current_step]
        # [Bugfix] 현재 봉 close를 본 뒤, 체결은 다음 봉 open 기준으로 수행.
        fill_step = min(self.current_step + 1, len(self._open_np) - 1)
        fill_price = float(self._open_np[fill_step])
        decision_step = self.current_step

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        # ── action 해석 ──
        abs_action = abs(action)
        desired_dir = 'LONG' if action > 0 else 'SHORT' if action < 0 else None
        leverage_rate = abs_action   # action 크기 = 포지션 사이즈

        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.025:
            force_close = True

        is_entering_long = False
        is_entering_short = False
        is_closing = False
        is_adjusting = False

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
            elif (self.pos == 'LONG' and action < -_POS_THRESH):
                is_closing = True
            elif (self.pos == 'SHORT' and action > _POS_THRESH):
                is_closing = True
            else:
                is_adjusting = True

        # ── 거래 실행 ──
        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close
        self._last_closed_side = ""
        self._last_closed_hold_count = 0

        if is_entering_long:
            self.pos = 'LONG'
            self.entry_price = fill_price
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_entering_short:
            self.pos = 'SHORT'
            self.entry_price = fill_price
            self.entry_idx = fill_step
            self.current_leverage = leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
        elif is_adjusting and self.pos is not None:
            # 동일 방향 보유 중: 레버리지 동적 조정
            # 변경분(|신규 - 기존|)에 대해서만 수수료 부과
            old_lev = self.current_leverage
            new_lev = leverage_rate
            lev_delta = abs(new_lev - old_lev)
            if lev_delta > 0.05:  # 5% 이상 변동 시에만 조정 (노이즈 방지)
                self.balance -= self.balance * self.fee * lev_delta
                self.current_leverage = new_lev
        elif is_closing and self.pos is not None:
            closed_side = str(self.pos)
            closed_hold_count = int(self.hold_count)
            base_balance = self.balance
            realized_pnl = (
                _signed_log_return(self.entry_price, fill_price, self.pos)
                - 2.0 * self.slip
            ) * self.current_leverage
            self.balance = base_balance * (1.0 + realized_pnl)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if realized_pnl > 0:
                self.win_trades += 1
            self._just_closed = True
            self._last_realized_pnl = realized_pnl
            self._last_closed_side = closed_side
            self._last_closed_hold_count = closed_hold_count
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
            raw_pnl = (
                _signed_log_return(self.entry_price, next_price, self.pos)
                - 2.0 * self.slip
            )
            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)

        # ── 보상 (기존 5-Component 그대로) ──
        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -0.01:
            dd_ratio = abs(self.unrealized_pnl) / 0.025
            r2_drawdown = -0.1 * (dd_ratio ** 2)

        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.30
            elif self._last_realized_pnl > 0:
                r3_quality = 0.10 * min(self._last_realized_pnl / 0.01, 1.0)
            else:
                r3_quality = -0.10

        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > 12:
            r4_time_decay = -0.003 * (self.hold_count - 12) / 72.0

        r5_idle = 0.0

        # R6 (SAC 신규): 명시적 거래 비용 패널티 — 진입 시 수수료를 직접 페널티로 부과
        r6_trade_cost = 0.0
        if is_entering_long or is_entering_short:
            r6_trade_cost = -0.01 * leverage_rate   # 레버리지에 비례한 진입 비용

        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle + r6_trade_cost
        reward = float(np.tanh(raw_reward))

        # 에피소드 종료 시 강제 청산 — reward에 청산 PnL 반영
        if done and self.pos is not None:
            base_balance = self.balance
            # 일반 청산과 동일하게 다음봉 open 기준 체결 (close 사용 시 look-ahead 불일치)
            ep_fill_step = min(self.current_step, len(self._open_np) - 1)
            ep_end_price = float(self._open_np[ep_fill_step])
            ep_realized = (
                _signed_log_return(self.entry_price, ep_end_price, self.pos)
                - 2.0 * self.slip
            ) * self.current_leverage
            self.balance = base_balance * (1.0 + ep_realized)
            self.balance -= base_balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0:
                self.win_trades += 1
            # [Bug 4 Fix] 종료 청산 PnL을 terminal reward에 추가
            terminal_r = float(np.tanh(ep_realized * 50.0))
            if ep_realized > 0:
                terminal_r += 0.10 * min(ep_realized / 0.01, 1.0)
            else:
                terminal_r -= 0.10
            reward = float(np.tanh(raw_reward + terminal_r))
            self.pos = None

        info = {
            'pnl_pct': (self.balance / self.initial_balance - 1) * 100,
            'wr': self.win_trades / max(1, self.total_trades),
            'force_closed': bool(self._just_closed and self._was_force_closed),
            'closed_side': self._last_closed_side,
            'closed_hold_count': int(self._last_closed_hold_count),
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    @property
    def win_rate(self):
        return self.win_trades / max(1, self.total_trades)

    def _get_stacked_state(self, raw_state):
        self._frame_stack.append(raw_state)
        pad = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        return np.concatenate(frames)

    def _build_state(self, idx):
        """기존 TradingEnv._build_state와 동일한 피쳐 구성."""
        if idx < 0 or idx >= len(self._feat_np):
            return np.zeros(STATE_DIM, dtype=np.float32)
        row = self._feat_np[idx]
        o = 0
        preds  = row[o:o + self._n_pred];  o += self._n_pred
        confs  = row[o:o + self._n_conf];  o += self._n_conf
        signal = preds * confs
        elite  = row[o:o + self._n_elite]; o += self._n_elite
        alpha6 = row[o:o + self._n_alpha]; o += self._n_alpha
        regime_raw = row[o:o + self._n_regime]; o += self._n_regime
        regime_idx = np.array([float(np.argmax(regime_raw))], dtype=np.float32)
        synth2 = row[o:o + self._n_synth]

        close = self._close_np[idx]
        pos_features = np.array([
            1.0 if self.pos == 'LONG' else (-1.0 if self.pos == 'SHORT' else 0.0),
            self.entry_price / close - 1 if self.pos is not None else 0.0,
            np.tanh(self.unrealized_pnl / 0.02),
            np.clip(self.max_drawdown / 0.05, -1.0, 1.0),
            self.hold_count / 144
        ], dtype=np.float32)

        if self.hmm_detector is not None:
            row_dict = {col: float(self._hmm_obs_np[col][idx]) for col in self._hmm_obs_np}
            hmm_feat = self.hmm_detector.get_features(row_dict)
        else:
            hmm_feat = np.zeros(HMM_DIM, dtype=np.float32)

        mtf_feat = self.mtf.get(idx)

        return np.nan_to_num(
            np.concatenate([signal, elite, alpha6, regime_idx, hmm_feat, synth2, pos_features, mtf_feat]),
            0.0
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. 리플레이 버퍼 (연속 action용)
# ═══════════════════════════════════════════════════════════════════════════
class ReplayBuffer:
    """SAC용 간단한 리플레이 버퍼 — 연속 action (float) 저장."""

    def __init__(self, capacity=500000):
        self._cap = capacity
        self._ptr = 0
        self._size = 0
        self._s = None
        self._a = np.empty(capacity, np.float32)   # 연속 action
        self._r = np.empty(capacity, np.float32)
        self._ns = None
        self._d = np.empty(capacity, np.bool_)

    def push(self, state, action, reward, next_state, done):
        if self._s is None:
            sdim = len(state)
            self._s  = np.empty((self._cap, sdim), np.float32)
            self._ns = np.empty((self._cap, sdim), np.float32)
        p = self._ptr
        self._s[p]  = state
        self._a[p]  = action
        self._r[p]  = reward
        self._ns[p] = next_state
        self._d[p]  = done
        self._ptr  = (p + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self._size, size=batch_size)
        return (self._s[idx], self._a[idx], self._r[idx],
                self._ns[idx], self._d[idx].astype(np.float32))

    def __len__(self):
        return self._size


# ═══════════════════════════════════════════════════════════════════════════
# 3. SAC 모델 아키텍처
# ═══════════════════════════════════════════════════════════════════════════

# ── 공유 피쳐 추출기 ──
class FeatureExtractor(nn.Module):
    """MarketAttentionEncoder + MLP로 state를 압축."""

    def __init__(self, state_dim=STACKED_STATE_DIM, hidden_dim=256,
                 raw_state_dim=STATE_DIM):
        super().__init__()
        self._raw_state_dim = raw_state_dim
        self.attn_encoder = MarketAttentionEncoder(out_dim=hidden_dim, raw_state_dim=raw_state_dim)
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.SiLU(),
        )

    def forward(self, state):
        last_frame_start = state.shape[1] - self._raw_state_dim
        market_feat = state[:, last_frame_start: last_frame_start + FEATURE_DIM]
        attn_out = self.attn_encoder(market_feat)
        return self.mlp(torch.cat([state, attn_out], dim=1))


# ── Actor (Gaussian Policy) ──
LOG_STD_MIN = -20
LOG_STD_MAX = 2

class GaussianActor(nn.Module):
    """state → action ∈ [-1, +1] (tanh squashed Gaussian)."""

    def __init__(self, state_dim=STACKED_STATE_DIM, hidden_dim=256):
        super().__init__()
        self.feat = FeatureExtractor(state_dim, hidden_dim)
        self.mu_head = nn.Linear(hidden_dim, 1)
        self.log_std_head = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        feat = self.feat(state)
        mu = self.mu_head(feat)
        log_std = self.log_std_head(feat).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, state):
        """reparameterized sample + log_prob (entropy 계산용)."""
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mu, std)
        x_t = dist.rsample()                       # reparameterization trick
        action = torch.tanh(x_t)                    # squash to [-1, +1]
        # log_prob with tanh correction
        log_prob = dist.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def deterministic(self, state):
        """추론 시 결정론적 action."""
        mu, _ = self.forward(state)
        return torch.tanh(mu)


# ── Twin Critic ──
class TwinCritic(nn.Module):
    """Q1, Q2 두 개의 Critic — SAC twin clipping."""

    def __init__(self, state_dim=STACKED_STATE_DIM, hidden_dim=256):
        super().__init__()
        self.feat1 = FeatureExtractor(state_dim, hidden_dim)
        self.q1 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.feat2 = FeatureExtractor(state_dim, hidden_dim)
        self.q2 = nn.Sequential(
            nn.Linear(hidden_dim + 1, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        """action: (batch, 1) 연속값."""
        f1 = self.feat1(state)
        f2 = self.feat2(state)
        q1 = self.q1(torch.cat([f1, action], dim=1))
        q2 = self.q2(torch.cat([f2, action], dim=1))
        return q1, q2


# ═══════════════════════════════════════════════════════════════════════════
# 4. SAC Agent
# ═══════════════════════════════════════════════════════════════════════════
class SACAgent:
    """Soft Actor-Critic with automatic entropy tuning."""

    def __init__(self, state_dim=STACKED_STATE_DIM, hidden_dim=256,
                 lr_actor=3e-4, lr_critic=3e-4, lr_alpha=3e-4,
                 gamma=0.99, tau=0.005, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = GaussianActor(state_dim, hidden_dim).to(device)
        self.critic = TwinCritic(state_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)
        self.critic_target.eval()

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Automatic entropy tuning
        # target_entropy = -dim(action) = -1 for 1D action
        self.target_entropy = -1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr_alpha)

        self.memory = ReplayBuffer(capacity=500000)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def act(self, state, deterministic=False):
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                action = self.actor.deterministic(state_ts)
            else:
                action, _ = self.actor.sample(state_ts)
        return float(action.cpu().item())

    def update(self, batch_size=256):
        if len(self.memory) < batch_size:
            return {}

        s, a, r, ns, d = self.memory.sample(batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.FloatTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        alpha = self.log_alpha.exp().detach()

        # ── Critic 업데이트 ──
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(ns)
            q1_target, q2_target = self.critic_target(ns, next_action)
            q_target = torch.min(q1_target, q2_target) - alpha * next_log_prob
            target_q = r + self.gamma * (1 - d) * q_target

        q1, q2 = self.critic(s, a)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # ── Actor 업데이트 ──
        new_action, log_prob = self.actor.sample(s)
        q1_new, q2_new = self.critic(s, new_action)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # ── Alpha (entropy temperature) 업데이트 ──
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ── Target network soft update ──
        for tp, p in zip(self.critic_target.parameters(), self.critic.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        return {
            'critic_loss': float(critic_loss.item()),
            'actor_loss': float(actor_loss.item()),
            'alpha': float(self.log_alpha.exp().item()),
            'mean_q': float(q_new.mean().item()),
        }


# ═══════════════════════════════════════════════════════════════════════════
# 5. SACRouter — 라이브 추론용 (trading_bot.py에서 import)
# ═══════════════════════════════════════════════════════════════════════════
class SACRouter:
    """학습된 SAC Actor로 라이브 트레이딩 신호 생성.

    GatingRouter7 대체: 단일 Actor가 연속 action을 출력.
    action → 방향 + 포지션 사이즈 동시 결정.
    """

    def __init__(self, actor, device='cuda', hmm_detector=None, mtf_features=None):
        self.actor = actor
        self.device = device
        self.hmm = hmm_detector
        self.mtf = mtf_features
        self._frame_stack = deque(maxlen=STACK_N)

    def _state_tensor(self, features, pos):
        """기존 GatingRouter7._state_tensor과 동일한 state 구성."""
        preds  = np.array([features.get(c, 0.) for c in STATE_PRED], dtype=np.float32)
        confs  = np.array([features.get(c, 0.) for c in STATE_CONF], dtype=np.float32)
        signal = preds * confs
        elite  = np.array([features.get(c, 0.) for c in STATE_ELITE], dtype=np.float32)
        alpha6 = np.array([features.get(c, 0.) for c in STATE_ALPHA], dtype=np.float32)
        regime_raw = np.array([features.get(c, 0.) for c in REGIME_COLS], dtype=np.float32)
        regime_idx = np.array([float(np.argmax(regime_raw))], dtype=np.float32)
        synth2 = np.array([features.get(c, 0.) for c in STATE_SYNTH], dtype=np.float32)

        cur_p = features.get('close', 1.0)
        pt = pos.get('type')
        pos_arr = np.array([
            1.0 if pt == 'LONG' else (-1.0 if pt == 'SHORT' else 0.0),
            pos.get('entry_price', cur_p) / cur_p - 1 if pt else 0.0,
            pos.get('unrealized', 0.), pos.get('mdd', 0.), pos.get('hold_norm', 0.)
        ], dtype=np.float32)

        if self.hmm is not None:
            hmm_feat = self.hmm.get_features(features)
        else:
            hmm_feat = np.zeros(HMM_DIM, dtype=np.float32)

        if self.mtf is not None:
            mtf_feat = self.mtf.get(int(features.get('_step_idx', -1)))
        else:
            mtf_feat = np.zeros(MTF_DIM, dtype=np.float32)

        raw = np.concatenate([signal, elite, alpha6, regime_idx, hmm_feat, synth2, pos_arr, mtf_feat])
        self._frame_stack.append(raw)
        pad = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        vec = np.concatenate(frames)
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def decide(self, features, pos):
        """라이브 추론: features + pos → (action_int, leverage, info).

        Returns:
            action_int: 0=관망, 1=LONG, 2=SHORT (기존 인터페이스 호환)
            leverage: 0.0~1.0 (|action| = 포지션 사이즈)
            info: dict
        """
        state = self._state_tensor(features, pos)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor.deterministic(state)
        action_val = float(action.cpu().item())

        # HMM 정보
        if self.hmm is not None:
            hmm_probs = self.hmm._alpha.copy()
            hmm_state = int(np.argmax(hmm_probs))
            hmm_names = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
            hmm_info = {'hmm_state': hmm_names[hmm_state],
                        'hmm_probs': hmm_probs.round(3).tolist()}
        else:
            hmm_info = {}

        abs_action = abs(action_val)

        # 기존 인터페이스 호환: action_int + leverage
        cur_pos = pos.get('type')

        if cur_pos is not None:
            # 포지션 보유 중
            if abs_action < _CLOSE_THRESH:
                action_int = 0   # 청산
                leverage = 0.0
            elif (cur_pos == 'LONG' and action_val < -_POS_THRESH):
                action_int = 0   # 반전 청산
                leverage = 0.0
            elif (cur_pos == 'SHORT' and action_val > _POS_THRESH):
                action_int = 0   # 반전 청산
                leverage = 0.0
            else:
                # 유지
                action_int = 1 if cur_pos == 'LONG' else 2
                leverage = abs_action
        else:
            # 무포지션
            if action_val > _POS_THRESH:
                action_int = 1   # LONG
                leverage = abs_action
            elif action_val < -_POS_THRESH:
                action_int = 2   # SHORT
                leverage = abs_action
            else:
                action_int = 0   # 관망
                leverage = 0.0

        info = {
            'agent': 'SAC',
            'raw_action': round(action_val, 4),
            'kelly': leverage,     # 기존 호환: kelly = leverage
            'long_edge': max(action_val, 0.0),
            'short_edge': max(-action_val, 0.0),
            'score': abs_action,
            **hmm_info,
        }

        return action_int, leverage, info


# ═══════════════════════════════════════════════════════════════════════════
# 6. 학습 루프
# ═══════════════════════════════════════════════════════════════════════════
def train(csv_path: str = 'data/rl_training_data_full.csv', train_ratio: float = 0.8, episodes: int = 1000):
    if not os.path.exists(csv_path):
        return logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")

    df = pd.read_csv(csv_path)
    split_idx = int(len(df) * float(train_ratio))
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val   = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")

    # ── 융합 모듈 초기화 (기존과 동일) ──
    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    logger.info("[HMM] 초기 학습 완료.")

    logger.info("[MTF] 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train['close'].values.astype(np.float32))
    mtf_val   = MultiTimeframeFeatures(df_val['close'].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    # ── 환경 & 에이전트 ──
    train_hmm = copy.deepcopy(hmm_detector)
    env = SACTradingEnv(df_train, phase='train', hmm_detector=train_hmm, mtf_features=mtf_train)
    agent = SACAgent(STACKED_STATE_DIM, hidden_dim=256, device=device)

    # ── 학습 파라미터 ──
    NEP = int(episodes)
    BATCH = 256
    UPDATE_FREQ = 4        # 4스텝마다 업데이트
    MIN_BUFFER = 4096      # 최소 버퍼 사이즈
    WARMUP_STEPS = 10000   # 초기 랜덤 탐험 스텝
    global_step = 0

    # ── Best 선정 안정화 파라미터 ──
    BEST_WR_MIN = 0.55
    BEST_PNL_MIN = 0.0
    BEST_TR_MIN = 200
    BEST_WINDOW_K = 5

    best_val_score = -float('inf')
    best_val_pnl = -float('inf')
    best_rolling_score = -float('inf')
    val_score_history = []
    os.makedirs('data/ensemble/ckpt', exist_ok=True)
    CKPT_PATH = 'data/ensemble/ckpt/sac_checkpoint.pth'
    BEST_PATH = 'data/ensemble/ckpt/best_sac_agents.pth'

    def _save_best_snapshot(epoch: int, recovered_from_checkpoint: bool = False):
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'best_pnl': best_val_pnl,
            'best_score': best_val_score,
            'best_rolling_score': best_rolling_score,
            'epoch': epoch,
            'meta': {
                'algo': 'SAC',
                'best_window_k': BEST_WINDOW_K,
                'wr_min': BEST_WR_MIN,
                'pnl_min': BEST_PNL_MIN,
                'tr_min': BEST_TR_MIN,
                'recovered_from_checkpoint': bool(recovered_from_checkpoint),
            },
        }, BEST_PATH)

    # ── 체크포인트 복원 ──
    start_ep = 1
    if os.path.exists(CKPT_PATH):
        try:
            ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
            agent.actor.load_state_dict(ckpt['actor'])
            agent.critic.load_state_dict(ckpt['critic'])
            agent.critic_target.load_state_dict(ckpt['critic_target'])
            agent.log_alpha.data.copy_(ckpt['log_alpha'])
            agent.actor_optimizer.load_state_dict(ckpt['actor_opt'])
            agent.critic_optimizer.load_state_dict(ckpt['critic_opt'])
            agent.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
            global_step = ckpt.get('global_step', 0)
            best_val_pnl = ckpt.get('best_val_pnl', -float('inf'))
            best_val_score = ckpt.get('best_val_score', -float('inf'))
            best_rolling_score = ckpt.get('best_rolling_score', -float('inf'))
            val_score_history = ckpt.get('val_score_history', [])
            start_ep = ckpt.get('epoch', 0) + 1
            logger.info(f"♻️ [복원] ep={start_ep-1} | global_step={global_step} | best_pnl={best_val_pnl:.2f}%")
            # [Bug 3 Fix] 리플레이 버퍼는 복원되지 않으므로,
            # 버퍼가 MIN_BUFFER 미만이면 warmup을 강제 재실행
            if len(agent.memory) < MIN_BUFFER:
                _refill_steps = max(WARMUP_STEPS, MIN_BUFFER)
                logger.info(f"    [WARMUP 재실행] 버퍼 비어있음 → {_refill_steps} 스텝 랜덤 탐험으로 리필")
                _warmup_env = SACTradingEnv(df_train, phase='train',
                                            hmm_detector=copy.deepcopy(hmm_detector),
                                            mtf_features=mtf_train)
                _ws = _warmup_env.reset()
                for _ in range(_refill_steps):
                    _wa = np.random.uniform(-1.0, 1.0)
                    _wns, _wr, _wd, _ = _warmup_env.step(_wa)
                    agent.memory.push(_ws, _wa, _wr, _wns, _wd)
                    _ws = _wns
                    if _wd:
                        _ws = _warmup_env.reset()
                logger.info(f"    [WARMUP 완료] 버퍼: {len(agent.memory)}")

            if not os.path.exists(BEST_PATH):
                _save_best_snapshot(epoch=start_ep - 1, recovered_from_checkpoint=True)
                logger.warning("⚠️ best_sac_agents.pth 없음 → 복원 가중치로 best 파일을 즉시 재생성했습니다.")
        except Exception as e:
            logger.warning(f"⚠️ 체크포인트 복원 실패: {e}")

    def _save_checkpoint(ep):
        torch.save({
            'actor': agent.actor.state_dict(),
            'critic': agent.critic.state_dict(),
            'critic_target': agent.critic_target.state_dict(),
            'log_alpha': agent.log_alpha.data,
            'actor_opt': agent.actor_optimizer.state_dict(),
            'critic_opt': agent.critic_optimizer.state_dict(),
            'alpha_opt': agent.alpha_optimizer.state_dict(),
            'global_step': global_step,
            'best_val_pnl': best_val_pnl,
            'best_val_score': best_val_score,
            'best_rolling_score': best_rolling_score,
            'val_score_history': val_score_history,
            'epoch': ep,
        }, CKPT_PATH)

    # ── 학습 루프 ──
    ep = start_ep  # [Bug 5 Fix] 루프 진입 전 초기화
    try:
        for ep in range(start_ep, NEP + 1):
            state = env.reset()
            ep_reward = 0.0
            done = False

            while not done:
                global_step += 1

                # Warmup: 초기에는 랜덤 action으로 버퍼 채우기
                if global_step < WARMUP_STEPS:
                    action = np.random.uniform(-1.0, 1.0)
                else:
                    action = agent.act(state, deterministic=False)

                next_state, reward, done, info = env.step(action)
                agent.memory.push(state, action, reward, next_state, done)
                ep_reward += reward
                state = next_state

                # 업데이트
                if global_step % UPDATE_FREQ == 0 and len(agent.memory) >= MIN_BUFFER:
                    agent.update(BATCH)

            # ── 에피소드 로깅 ──
            pnl = (env.balance / env.initial_balance - 1) * 100
            logger.info(
                f"Ep {ep:04d} | PnL:{pnl:6.1f}% Tr:{env.total_trades:4d} "
                f"WR:{env.win_rate * 100:4.0f}% Rew:{ep_reward:7.3f} | "
                f"buf:{len(agent.memory):6d} | α:{agent.alpha:.4f}"
            )

            # ── Validation (매 10 에피소드) ──
            if ep % 10 == 0:
                val_hmm = copy.deepcopy(hmm_detector)
                val_env = SACTradingEnv(df_val, phase='val', hmm_detector=val_hmm, mtf_features=mtf_val)

                val_state = val_env.reset()
                val_done = False
                agent.actor.eval()
                while not val_done:
                    with torch.no_grad():
                        val_action = agent.act(val_state, deterministic=True)
                    val_state, _, val_done, _ = val_env.step(val_action)
                agent.actor.train()

                val_pnl = (val_env.balance / val_env.initial_balance - 1) * 100
                val_wr  = val_env.win_rate   # 0~1 비율 (퍼센트 아님)
                # val_score: PnL(%) × 5 + WR(0~1) × 20 + 거래활성도
                val_trade_score = 0.0
                if val_env.total_trades == 0:
                    val_trade_score = -5.0
                elif val_pnl > 0:
                    val_trade_score = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    val_trade_score = -min(val_env.total_trades / 30.0, 1.0) * 10.0
                val_score = val_pnl * 5.0 + val_wr * 20.0 + val_trade_score
                val_score_history.append(float(val_score))
                if len(val_score_history) > 200:
                    val_score_history = val_score_history[-200:]
                rolling_ready = len(val_score_history) >= BEST_WINDOW_K
                rolling_score = float(np.median(val_score_history[-BEST_WINDOW_K:])) if rolling_ready else -float('inf')

                quality_reasons = []
                if val_wr < BEST_WR_MIN:
                    quality_reasons.append(f"wr<{BEST_WR_MIN:.2f}")
                if val_pnl < BEST_PNL_MIN:
                    quality_reasons.append(f"pnl<{BEST_PNL_MIN:.1f}")
                if val_env.total_trades < BEST_TR_MIN:
                    quality_reasons.append(f"tr<{BEST_TR_MIN}")
                quality_pass = len(quality_reasons) == 0

                logger.info(
                    f"    [VAL] PnL:{val_pnl:6.2f}% | Tr:{val_env.total_trades:4d} | "
                    f"WR:{val_wr*100:.0f}% | Score:{val_score:.2f}"
                )
                logger.info(
                    f"    [VAL-BEST] quality:{'PASS' if quality_pass else 'BLOCK'} "
                    f"({','.join(quality_reasons) if quality_reasons else 'ok'}) | "
                    f"roll({BEST_WINDOW_K})={rolling_score:.2f} | best_roll={best_rolling_score:.2f}"
                )

                if quality_pass and rolling_ready and rolling_score > best_rolling_score:
                    best_rolling_score = rolling_score
                    best_val_score, best_val_pnl = val_score, val_pnl
                    _save_best_snapshot(epoch=ep, recovered_from_checkpoint=False)
                    logger.info(
                        f"    🎉 [NEW BEST] 저장 완료 (PnL:{best_val_pnl:.2f}% | "
                        f"score:{best_val_score:.2f} | roll:{best_rolling_score:.2f})"
                    )

                # HMM 온라인 업데이트
                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    train_hmm.A = hmm_detector.A.copy()
                    train_hmm.mu = hmm_detector.mu.copy()
                    train_hmm.sigma = hmm_detector.sigma.copy()
                    train_hmm.pi = hmm_detector.pi.copy()
                    train_hmm._obs_mean = hmm_detector._obs_mean.copy()
                    train_hmm._obs_std = hmm_detector._obs_std.copy()
                    logger.info("    [HMM] 온라인 업데이트 완료")

                _save_checkpoint(ep)

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단.")
        _save_checkpoint(ep)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Train SAC agent')
    p.add_argument('--csv-path', default='data/rl_training_data_full.csv')
    p.add_argument('--train-ratio', type=float, default=0.8)
    p.add_argument('--episodes', type=int, default=1000)
    p.add_argument(
        '--startup-check-only',
        action='store_true',
        help='Validate imports/arguments and exit without training',
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.startup_check_only:
        logger.info('startup check ok: train_rl_sac_agent')
        raise SystemExit(0)
    train(csv_path=args.csv_path, train_ratio=args.train_ratio, episodes=args.episodes)
