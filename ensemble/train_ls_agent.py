"""
Dual-Brain Single-Agent IQN Pipeline  (v4)
========================================================================================
핵심 아키텍처 변경:
  기존: RobustIQN(Long) + RobustIQN(Short) + HCRouter  →  2개 뇌 + 중재자
  신규: DualBrainIQN  →  1개 뇌 (롱반구 + 숏반구) + SimpleRouter

왜 더 나은가:
  - 공유 백본(TemporalEncoder + MultiHeadContextGate)으로 시장 맥락을 한 번만 계산
  - 롱헤드/숏헤드가 동일한 shared_feat에서 분기 → Q-value 스케일 통일 → 직접 비교 가능
  - HCRouter의 삼중 AND 조건(Q-adv + ML + 방향) 제거 → SimpleRouter argmax로 단순화
  - 에이전트 1개 / 환경 1개 / 버퍼 1개 → 구조 단순화 + 학습 효율 향상

액션 공간 통일 (train/val 동일):
  0 = flat/청산  1 = long 진입  2 = short 진입
"""
import os, sys, logging, random, argparse, copy
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

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
# [상수 및 차원 정의]
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PRED = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst', 'pred_tide', 'pred_mdjd', 'pred_ridge']
MODEL_CONF = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst', 'conf_tide', 'conf_mdjd', 'conf_ridge']
ELITE_COLS = ['sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze']
ALPHA_7_COLS = ['session_us', 'hour_cos', 'cvp_poc_dist', 'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate']
REGIME_COLS = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']

STATE_PRED  = ['pred_tide', 'pred_ridge', 'pred_patchtst', 'pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_mdjd']
STATE_CONF  = ['conf_tide', 'conf_ridge', 'conf_ttm', 'conf_chronos', 'conf_timesfm', 'conf_mdjd', 'conf_patchtst']
STATE_ELITE = ['evt_excess_z', 'sig_orderblock', 'sig_ai_squeeze', 'sig_oi_divergence', 'sig_whale', 'sig_garch_regime', 'jump_z', 'jump_flag', 'evt_tail_flag']
STATE_ALPHA = ['hour_cos', 'garch_vol', 'garch_vol_z', 'breakout_strength', 'fvg_dist', 'cvp_poc_dist', 'session_us', 'oi_change_rate', 'cvp_volume_imbalance']
STATE_SYNTH = ['ou_funding_z', 'fcsz', 'vebr', 'ofti', 'cada', 'tlad', 'svps', 'mshd', 'fdlv', 'wpad', 'fvci', 'kel', 'mtmb', 'ou_halflife']
# [D-01 수정] FEATURE_DIM: 피처 목록 변경(elite 9, alpha 9, synth 14)에 맞게 자동 계산
# pred(7)+conf(7)+stats(3)+elite(9)+alpha(9)+regime(5)+synth(14) = 54
FEATURE_DIM = len(STATE_PRED) + len(STATE_CONF) + 3 + len(STATE_ELITE) + len(STATE_ALPHA) + len(REGIME_COLS) + len(STATE_SYNTH)
STATE_DIM = FEATURE_DIM + 4          # 54 + 4 = 58
STACK_N = 4
STACKED_STATE_DIM = STATE_DIM * STACK_N   # 232
CVAR_THRESHOLD = 0.25

# ── [D-01/D-02/D-03 수정] 피처 그룹 슬라이스 오프셋 — 하드코딩 제거, 변수 기반 자동 계산 ──
# _build_state() 순서: pred conf stats elite alpha regime synth pos(4)
_O_PRED   = 0
_O_CONF   = _O_PRED  + len(STATE_PRED)
_O_STATS  = _O_CONF  + len(STATE_CONF)
_O_ELITE  = _O_STATS + 3
_O_ALPHA  = _O_ELITE + len(STATE_ELITE)
_O_REGIME = _O_ALPHA + len(STATE_ALPHA)
_O_SYNTH  = _O_REGIME + len(REGIME_COLS)

# [D-02 수정] _DIM_OTHER를 실제 피처 개수로 자동 계산 (구 버전 고정값 20 제거)
_DIM_PRED_CONF = len(STATE_PRED) + len(STATE_CONF) + 3           # pred+conf+stats
_DIM_REGIME    = len(REGIME_COLS)                                  # regime
_DIM_OTHER     = len(STATE_ELITE) + len(STATE_ALPHA) + len(STATE_SYNTH)  # elite+alpha+synth

# ═══════════════════════════════════════════════════════════════════════════
# 거래 환경 (TradingEnv)
# ═══════════════════════════════════════════════════════════════════════════
class TradingEnv:
    def __init__(self, df, initial_balance=10000.0, fee=0.0005, slip=0.0002, phase='train', agent_role='long'):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.agent_role = agent_role
        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1
        self.MAX_LEVERAGE = 1.0

        feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
        self._feat_np  = self.df[feat_cols].values.astype(np.float32)
        self._close_np = self.df['close'].values.astype(np.float32)
        self._n_pred, self._n_conf = len(STATE_PRED), len(STATE_CONF)
        self._n_elite, self._n_alpha = len(STATE_ELITE), len(STATE_ALPHA)
        self._n_regime, self._n_synth = len(REGIME_COLS), len(STATE_SYNTH)
        self._frame_stack = deque(maxlen=STACK_N)
        self.reset()

    def reset(self, start_idx=None):
        if self.phase == 'train':
            self.start_step = start_idx if start_idx is not None else random.randint(0, len(self.df) - self.MAX_EPISODE_STEPS - 1)
        else:
            self.start_step = 0

        self.current_step = self.start_step
        self.end_step = self.start_step + self.MAX_EPISODE_STEPS
        self.balance = self.initial_balance
        self.pos = None
        self.entry_price = 0.0
        self.entry_idx = 0
        self.current_leverage = 0.0
        self.total_trades = 0
        self.win_trades = 0
        self.active_steps = 0
        self.unrealized_pnl = 0.0
        self.prev_unrealized_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_pnl = 0.0
        self.hold_count = 0
        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step), is_reset=True)

    def step(self, action, leverage_rate=1.0):
        current_price = self._close_np[self.current_step]

        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.05:
            force_close = True

        is_entering_long, is_entering_short, is_closing = False, False, False

        if force_close:
            is_closing = True
        else:
            if action == 0 and self.pos is not None: is_closing = True
            elif action == 1 and self.pos is None:   is_entering_long  = True
            elif action == 2 and self.pos is None:   is_entering_short = True
            elif action == 0 and self.pos is None:   pass

        realized_pnl = 0.0

        if is_entering_long:
            self.pos, self.entry_price, self.entry_idx = 'LONG', current_price * (1 + self.slip), self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
            self.active_steps += 1

        elif is_entering_short:
            self.pos, self.entry_price, self.entry_idx = 'SHORT', current_price * (1 - self.slip), self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
            self.active_steps += 1

        elif is_closing:
            if self.pos == 'LONG':
                realized_pnl = (current_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                realized_pnl = (self.entry_price - current_price * (1 + self.slip)) / self.entry_price
            realized_pnl *= self.current_leverage
            self.balance += self.balance * realized_pnl
            self.balance -= self.balance * self.fee * self.current_leverage
            self.total_trades += 1
            if realized_pnl > 0: self.win_trades += 1
            self.pos, self.current_leverage, self.hold_count, self.unrealized_pnl = None, 0.0, 0, 0.0
            self.peak_pnl, self.max_drawdown = 0.0, 0.0

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[self.current_step] if not done else current_price

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx
            if self.pos == 'LONG':
                self.unrealized_pnl = (next_price - self.entry_price) / self.entry_price * self.current_leverage
            else:
                self.unrealized_pnl = (self.entry_price - next_price) / self.entry_price * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
            self.active_steps += 1

        # ─── 보상 계산 ───
        if is_closing:
            # [수정 ②] 이중 수수료 제거: 진입/청산 시 balance에서 이미 수수료가 차감되었음
            net_pnl = realized_pnl * 100.0
            # [수정 ①] 청산 보상 스케일링: log1p 대신 tanh 적용하여 -1~1 범위로 정규화
            reward = float(np.tanh(net_pnl / 5.0))
        else:
            # 보유/대기 스텝 보상
            prev_portfolio_value = self.balance + (self.balance * self.prev_unrealized_pnl if True else 0.0)
            current_portfolio_value = self.balance + (self.balance * self.unrealized_pnl if self.pos is not None else 0.0)
            reward_diff = (current_portfolio_value - prev_portfolio_value) / self.initial_balance * 100.0
            # [수정 ①] 보유 보상 스케일링: 청산 보상과 동일하게 tanh 적용하여 스케일 대통합
            reward = float(np.tanh(reward_diff / 5.0))

        # [수정 ③] 이전 미실현 손익 갱신 오류 수정 (pos=None일 때 0으로 리셋하여 오염 방지)
        self.prev_unrealized_pnl = self.unrealized_pnl if self.pos is not None else 0.0

        info = {
            'pnl_pct': (self.balance / self.initial_balance - 1) * 100,
            'wr': self.win_trades / max(1, self.total_trades),
            'timeout': done and not force_close
        }
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    @property
    def win_rate(self): return self.win_trades / max(1, self.total_trades)

    def _get_stacked_state(self, raw_state, is_reset=False):
        if is_reset or len(self._frame_stack) == 0:
            for _ in range(STACK_N):
                self._frame_stack.append(raw_state)
        else:
            self._frame_stack.append(raw_state)
        return np.concatenate(self._frame_stack)

    def _build_state(self, idx):
        if idx < 0 or idx >= len(self._feat_np): return np.zeros(STATE_DIM, dtype=np.float32)
        row = self._feat_np[idx]
        o = 0
        preds  = row[o:o+self._n_pred];   o += self._n_pred
        confs  = row[o:o+self._n_conf];   o += self._n_conf
        stats  = np.array([preds.mean(), preds.std(), confs.mean()], dtype=np.float32)
        elite  = row[o:o+self._n_elite];  o += self._n_elite
        alpha7 = row[o:o+self._n_alpha];  o += self._n_alpha
        regimes= row[o:o+self._n_regime]; o += self._n_regime
        synth  = row[o:]

        pos_features = np.array([
            1.0 if self.pos == 'LONG' else (-1.0 if self.pos == 'SHORT' else 0.0),
            np.tanh(self.unrealized_pnl / 0.02),
            np.clip(self.max_drawdown / 0.05, -1.0, 1.0),
            self.hold_count / 144
        ], dtype=np.float32)

        return np.nan_to_num(np.concatenate([preds, confs, stats, elite, alpha7, regimes, synth, pos_features]), 0.0)

# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════
class PrioritizedReplayBuffer:
    def __init__(self, capacity=200000, alpha=0.6, beta=0.4, beta_anneal_steps=600_000):
        self._cap, self._ptr, self._size, self._push_count = capacity, 0, 0, 0
        self._buf_s, self._buf_ns = None, None
        self._buf_a = np.empty(capacity, np.int32)
        self._buf_r = np.empty(capacity, np.float32)
        self._buf_d = np.empty(capacity, np.float32)
        self._priorities = np.zeros(capacity, np.float32)
        self.alpha, self.beta, self._beta_start, self._beta_anneal_steps = alpha, beta, beta, beta_anneal_steps
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        self._push_count += 1
        if self._buf_s is None:
            sdim = len(state)
            self._buf_s, self._buf_ns  = np.empty((self._cap, sdim), np.float32), np.empty((self._cap, sdim), np.float32)
        p = self._ptr
        self._buf_s[p], self._buf_a[p], self._buf_r[p] = state, action, reward
        self._buf_ns[p], self._buf_d[p], self._priorities[p] = next_state, float(done), self.max_priority
        self._ptr = (p + 1) % self._cap
        self._size = min(self._size + 1, self._cap)

    def sample(self, batch_size):
        # [B-03] size < batch_size*2 구간에서 비복원 추출 시 ValueError 방지
        if self._size < batch_size * 2:
            return None
        self.beta = min(1.0, self._beta_start + (1.0 - self._beta_start) * (self._push_count / self._beta_anneal_steps))
        pri = self._priorities[:self._size] ** self.alpha
        probs = pri / (pri.sum() + 1e-8)
        indices = np.random.choice(self._size, batch_size, p=probs, replace=False)
        weights = (1.0 / (self._size * probs[indices] + 1e-8)) ** self.beta
        weights = (weights / weights.max()).astype(np.float32)
        return (self._buf_s[indices], self._buf_a[indices], self._buf_r[indices], self._buf_ns[indices], self._buf_d[indices], indices, weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            # [수정 ④] alpha 이중 적용 제거: 여기서는 원시값만 저장하고 sample()에서 alpha 적용
            p = float(abs(err) + 1e-6)
            self._priorities[idx] = p
            if p > self.max_priority: self.max_priority = p

    def __len__(self): return self._size

# ═══════════════════════════════════════════════════════════════════════════
# 모델 및 에이전트 (RobustIQN + LayerNorm 적용)
# ═══════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.05):
        super().__init__()
        self.in_features, self.out_features = in_features, out_features
        self.weight_mu, self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features)), nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu, self.bias_sigma = nn.Parameter(torch.empty(out_features)), nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        self._sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range); self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5); self.bias_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5)

    def _f(self, x): return x.sign() * x.abs().sqrt()
    def sample_noise(self):
        eps_i, eps_j = self._f(torch.randn(self.in_features, device=self.weight_mu.device)), self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_j.ger(eps_i)); self.bias_epsilon.copy_(eps_j)
    def zero_noise(self): self.weight_epsilon.zero_(); self.bias_epsilon.zero_()
    def forward(self, x):
        w = self.weight_mu + self.weight_sigma * self.weight_epsilon if self.training else self.weight_mu
        b = self.bias_mu + self.bias_sigma * self.bias_epsilon if self.training else self.bias_mu
        return F.linear(x, w, b)


# ═══════════════════════════════════════════════════════════════════════════
# [A-01] TemporalEncoder
# feat_extractor(MLP) 대체 — 프레임 스택을 시계열로 처리해 순서 정보 보존
# ═══════════════════════════════════════════════════════════════════════════
class TemporalEncoder(nn.Module):
    """
    (B, STACK_N * STATE_DIM) → (B, hidden_dim)

    단순 concat MLP는 프레임 순서를 무시한다.
    프레임을 (B, STACK_N, STATE_DIM)으로 reshape 후 GRU에 통과시켜
    시장 모멘텀·추세를 담은 hidden state를 추출한다.
    """
    def __init__(self, frame_dim: int, hidden_dim: int = 64, stack_n: int = STACK_N):
        super().__init__()
        self.stack_n   = stack_n
        self.frame_dim = frame_dim
        self.proj      = nn.Linear(frame_dim, hidden_dim)
        # [M-03 수정] norm_proj 제거 — RobustIQN.input_norm이 이미 전체 입력을 정규화함
        # proj → norm_proj → GRU 의 이중 정규화는 그래디언트 흐름을 약화시킴
        self.gru       = nn.GRU(hidden_dim, hidden_dim, num_layers=1, batch_first=True)
        self.norm_out  = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        frames = x.view(B, self.stack_n, self.frame_dim)    # (B, 4, STATE_DIM)
        frames = F.silu(self.proj(frames))                  # (B, 4, hidden_dim) — norm_proj 제거
        out, _ = self.gru(frames)                           # (B, 4, hidden_dim)
        return self.norm_out(out[:, -1])                    # (B, hidden_dim)


# ═══════════════════════════════════════════════════════════════════════════
# [A-02] MultiHeadContextGate
# 단일 context_gate 대체 — 피처 그룹별 독립 게이팅
# ═══════════════════════════════════════════════════════════════════════════
class MultiHeadContextGate(nn.Module):
    """
    마지막 프레임의 피처를 3그룹으로 분리해 독립 게이트를 계산한다.

    그룹 구성 (STATE_DIM=46 기준, pos 4개 제외 42차원):
      - pred_conf : pred(7) + conf(7) + stats(3) = 17차원
      - regime    : regime(5)                   =  5차원
      - other     : elite(4) + alpha(7) + synth(9) = 20차원

    각 게이트가 독립적으로 feat에 주의를 기울이므로
    레짐 전환 구간에서 regime_gate가 즉각 반응 가능하다.
    """
    def __init__(self, dim_pred_conf: int = _DIM_PRED_CONF,
                 dim_regime: int = _DIM_REGIME,
                 dim_other: int = _DIM_OTHER,
                 out_dim: int = 64):
        super().__init__()
        self.dim_pred_conf = dim_pred_conf
        self.dim_regime    = dim_regime
        self.dim_other     = dim_other
        # 그룹별 게이트 레이어
        self.gate_pred_conf = nn.Linear(dim_pred_conf, out_dim)
        self.gate_regime    = nn.Linear(dim_regime,    out_dim)
        self.gate_other     = nn.Linear(dim_other,     out_dim)
        # 3 게이트 합산 → 최종 게이트 벡터
        self.mix = nn.Linear(out_dim * 3, out_dim)

    def forward(self, all_frames: torch.Tensor) -> torch.Tensor:
        B, T, D = all_frames.shape
        flat  = all_frames.reshape(B * T, D)
        market = flat[:, :FEATURE_DIM]

        pred_conf = market[:, :self.dim_pred_conf]
        regime    = market[:, _O_REGIME:_O_REGIME + self.dim_regime]
        other = torch.cat([
            market[:, _O_ELITE:_O_REGIME],
            market[:, _O_SYNTH:FEATURE_DIM],
        ], dim=-1)

        # [수정 ⑦] 내부 sigmoid 제거. 선형 결합 후 mix의 결과에만 최종적으로 sigmoid 적용
        g = torch.cat([
            self.gate_pred_conf(pred_conf),
            self.gate_regime(regime),
            self.gate_other(other),
        ], dim=-1)                                              
        gate_per_frame = torch.sigmoid(self.mix(g))            
        gate_per_frame = gate_per_frame.view(B, T, -1)         
        return gate_per_frame.mean(dim=1)


# ═══════════════════════════════════════════════════════════════════════════
# [A-03] EpistemicUncertaintyHead
# 고정 CVaR → 불확실성 기반 동적 CVaR 조정
# ═══════════════════════════════════════════════════════════════════════════
class EpistemicUncertaintyHead(nn.Module):
    """
    공유 feature에서 불확실성 스코어 [0, 1]을 출력한다.

    act()에서 dynamic_cvar = CVAR_THRESHOLD * (1 - 0.5 * unc) 로 사용:
      unc=1.0 → CVaR 0.125 (매우 보수적, 학습 초기)
      unc=0.0 → CVaR 0.250 (기존 고정값, 학습 안정 후)

    update()에서는 분위수 평균 shared를 입력으로 받아 배치 단위로 계산하며
    보조 엔트로피 정규화 손실 unc_loss를 반환한다 (과도한 불확실성 억제).
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        """feat: (B, hidden_dim) → unc: (B, 1)"""
        return self.head(feat)


# ═══════════════════════════════════════════════════════════════════════════
# DualBrainIQN — 공유 백본 + 롱반구/숏반구 이중 헤드
# ═══════════════════════════════════════════════════════════════════════════
class DualBrainIQN(nn.Module):
    """
    뇌 1개, 반구 2개.

    공유 백본(TemporalEncoder + MultiHeadContextGate)이 시장 맥락을 파악하고,
    롱 어댑터/숏 어댑터가 각자의 관점에서 Q-value를 계산한다.
    두 헤드가 동일한 shared_feat에서 분기하므로 Q-value 스케일이 통일되어
    SimpleRouter가 adv_long vs adv_short를 직접 비교할 수 있다.

    출력:
      q_long  : (B, NQ, 2)  — [flat_long, enter_long]
      q_short : (B, NQ, 2)  — [flat_short, enter_short]
      tau     : (B, NQ, 1)
      unc     : (B, 1)
    """
    def __init__(self, state_dim: int, hidden_dim: int = 64,
                 raw_state_dim: int = None):
        super().__init__()
        self.hidden_dim = hidden_dim
        _raw = raw_state_dim if raw_state_dim is not None else state_dim

        # 입력 정규화
        self.input_norm = nn.LayerNorm(state_dim)

        # ── 공유 백본 ──
        self.temporal_enc = TemporalEncoder(
            frame_dim=_raw, hidden_dim=hidden_dim, stack_n=STACK_N)
        self.context_gate = MultiHeadContextGate(
            dim_pred_conf=_DIM_PRED_CONF, dim_regime=_DIM_REGIME,
            dim_other=_DIM_OTHER, out_dim=hidden_dim)

        # ── 롱반구 (Long-specialist) ──
        self.long_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.long_phi     = nn.Linear(hidden_dim, hidden_dim)
        self.long_v       = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.long_a       = nn.Sequential(nn.SiLU(), NoisyLinear(hidden_dim, 2, sigma_init=0.5))

        # ── 숏반구 (Short-specialist) ──
        self.short_adapter = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
        self.short_phi     = nn.Linear(hidden_dim, hidden_dim)
        self.short_v       = nn.Sequential(nn.SiLU(), nn.Linear(hidden_dim, 1))
        self.short_a       = nn.Sequential(nn.SiLU(), NoisyLinear(hidden_dim, 2, sigma_init=0.5))

        # ── 불확실성 헤드 (공유) ──
        self.unc_head = EpistemicUncertaintyHead(hidden_dim)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.sample_noise()

    def _iqn_head(self, feat, phi, v_head, a_head, tau):
        """IQN Dueling 계산 — 롱/숏 헤드 공통 로직"""
        NQ = tau.size(1)
        cos_tau = torch.cos(
            tau * torch.arange(1, self.hidden_dim + 1, device=feat.device).float() * torch.pi)
        phi_x  = phi(cos_tau)                                        # (B, NQ, H)
        shared = feat.unsqueeze(1).expand(-1, NQ, -1) * phi_x       # (B, NQ, H)
        v = v_head(shared)                                           # (B, NQ, 1)
        a = a_head(shared)                                           # (B, NQ, 2)
        return v + a - a.mean(dim=-1, keepdim=True)                  # (B, NQ, 2)

    def forward(self, state: torch.Tensor, num_quantiles: int = 8):
        B = state.size(0)
        state_norm = self.input_norm(state)

        frames_raw = state_norm.view(B, STACK_N, STATE_DIM)
        feat = self.temporal_enc(state_norm)                         
        gate = self.context_gate(frames_raw)                         
        shared_feat = feat * gate                                    

        unc = self.unc_head(shared_feat)                             

        # [수정 ⑥] 분위수(Tau) 롱/숏 헤드 분리: 각 반구가 독립적인 분위수를 탐색
        tau_long = torch.rand(B, num_quantiles, 1, device=state.device)
        tau_short = torch.rand(B, num_quantiles, 1, device=state.device)

        long_feat = self.long_adapter(shared_feat)
        q_long  = self._iqn_head(long_feat,  self.long_phi,  self.long_v,  self.long_a,  tau_long)

        short_feat = self.short_adapter(shared_feat)
        q_short = self._iqn_head(short_feat, self.short_phi, self.short_v, self.short_a, tau_short)

        # 5개의 값을 반환하도록 변경됨
        return q_long, q_short, tau_long, tau_short, unc


# ═══════════════════════════════════════════════════════════════════════════
# DualBrainAgent — 에이전트 1개로 롱/숏 동시 학습
# ═══════════════════════════════════════════════════════════════════════════
class DualBrainAgent:
    NUM_QUANTILES = 32
    CVAR_UNC_SCALE = 0.5

    def __init__(self, model: DualBrainIQN, lr=5e-5, gamma=0.99,
                 tau=0.005, device='cuda', cvar_threshold=CVAR_THRESHOLD):
        self.model        = model
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        self.optimizer    = torch.optim.AdamW(model.parameters(), lr=lr)
        self.memory       = None
        self.gamma, self.tau, self.device = gamma, tau, device
        self.cvar_threshold = cvar_threshold

    def _cvar_q(self, q_long, q_short, tau_long, tau_short, unc_score):
        # [수정 ⑥ 연계] 분리된 tau_long, tau_short 각각에 대해 정렬 및 CVaR 추출
        NQ = self.NUM_QUANTILES
        dynamic_cvar = self.cvar_threshold * (1.0 - self.CVAR_UNC_SCALE * unc_score)
        k = max(1, int(NQ * dynamic_cvar))
        sort_idx_l = tau_long[0, :, 0].argsort()
        sort_idx_s = tau_short[0, :, 0].argsort()
        q_l = q_long[0][sort_idx_l][:k].mean(dim=0)
        q_s = q_short[0][sort_idx_s][:k].mean(dim=0)
        return q_l, q_s

    def act(self, state, eps=0.0):
        if self.model.training: self.model.reset_noise()
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_long, q_short, tau_long, tau_short, unc = self.model(state_ts, num_quantiles=self.NUM_QUANTILES)
            unc_score = unc[0, 0].item()
            q_l, q_s  = self._cvar_q(q_long, q_short, tau_long, tau_short, unc_score)

        adv_long  = q_l[1].item() - q_l[0].item()
        adv_short = q_s[1].item() - q_s[0].item()

        if eps > 0.0 and random.random() < eps:
            return random.randint(0, 2)

        # [수정 ⑧] 동적 임계값 적용 (스케일 조정) - 훈련 시 과매매 방지
        dynamic_threshold = 0.005 * (1.0 + unc_score)

        if adv_long > adv_short and adv_long > dynamic_threshold: return 1
        elif adv_short > adv_long and adv_short > dynamic_threshold: return 2
        return 0

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        result = self.memory.sample(batch_size)
        if result is None: return
        s, a, r, ns, d, per_indices, per_weights = result

        per_w = torch.FloatTensor(per_weights).to(self.device)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).to(self.device)          
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        NQ = self.NUM_QUANTILES

        self.model.reset_noise()
        B = s.size(0)
        # [수정 ⑥ 연계] 반환값 언패킹 맞춤
        q_long, q_short, tau_long, tau_short, unc_online = self.model(s, num_quantiles=NQ)

        a_long  = torch.where(a == 1, torch.ones_like(a), torch.zeros_like(a))
        a_short = torch.where(a == 2, torch.ones_like(a), torch.zeros_like(a))

        idx_long  = a_long.unsqueeze(1).unsqueeze(2).expand(-1, NQ, 1)   
        idx_short = a_short.unsqueeze(1).unsqueeze(2).expand(-1, NQ, 1)
        q_a_long  = q_long.gather(2,  idx_long).squeeze(2)               
        q_a_short = q_short.gather(2, idx_short).squeeze(2)              

        with torch.no_grad():
            q_long_t, q_short_t, _, _, _ = self.target_model(ns, num_quantiles=NQ)
            best_long_a  = q_long_t.mean(dim=1).argmax(dim=1)    
            best_short_a = q_short_t.mean(dim=1).argmax(dim=1)   
            qt_long  = q_long_t.gather(2, best_long_a.unsqueeze(1).unsqueeze(2).expand(-1,NQ,1)).squeeze(2)
            qt_short = q_short_t.gather(2, best_short_a.unsqueeze(1).unsqueeze(2).expand(-1,NQ,1)).squeeze(2)

            # [수정 ③] 두 헤드의 평균 타깃 배제 -> 롱/숏 각각 독립적인 타깃 부여
            target_long  = r + self.gamma * (1 - d) * qt_long
            target_short = r + self.gamma * (1 - d) * qt_short

        def qr_loss_fn(q_a, tgt, tau):
            td  = tgt.unsqueeze(1) - q_a.unsqueeze(2)           
            hub = F.huber_loss(td, torch.zeros_like(td), reduction='none', delta=1.0)
            return (torch.abs(tau - (td.detach() < 0).float()) * hub).mean(dim=1).mean(dim=1)

        loss_long  = qr_loss_fn(q_a_long,  target_long, tau_long)
        loss_short = qr_loss_fn(q_a_short, target_short, tau_short)
        
        is_long  = (a == 1).float()
        is_short = (a == 2).float()
        is_flat  = (a == 0).float()
        
        # [수정 ③] FLAT(a=0)일 때 두 헤드가 병합되지 않고 각각 자기 헤드의 Loss만 계산하도록 분리
        qr_loss = ((loss_long * (1 - is_short) + 
                    loss_short * (1 - is_long)) * per_w).mean()

        # 불확실성 보조 손실 및 PER 업데이트 (양 헤드 오차 평균 반영)
        td_abs_long = (target_long.unsqueeze(1) - q_a_long.unsqueeze(2)).detach().abs().mean(dim=(1,2))
        td_abs_short = (target_short.unsqueeze(1) - q_a_short.unsqueeze(2)).detach().abs().mean(dim=(1,2))
        td_abs = (td_abs_long + td_abs_short) / 2.0

        if not hasattr(self, '_unc_ema'): self._unc_ema = td_abs.mean().item()
        self._unc_ema = 0.99 * self._unc_ema + 0.01 * td_abs.mean().item()
        unc_target = torch.clamp(td_abs / (self._unc_ema * 3.0 + 1e-8), 0.0, 1.0).unsqueeze(1)
        unc_loss = F.mse_loss(unc_online, unc_target) * 0.05

        loss = qr_loss + unc_loss
        self.memory.update_priorities(per_indices, td_abs.cpu().numpy())

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ═══════════════════════════════════════════════════════════════════════════
# SimpleRouter — DualBrainIQN 전용, 단순 argmax 기반 의사결정
# ═══════════════════════════════════════════════════════════════════════════
class SimpleRouter:
    def __init__(self, model: DualBrainIQN, device='cuda',
                 cvar_threshold=CVAR_THRESHOLD):
        self.model  = model
        self.device = device
        self.cvar_threshold = cvar_threshold

    def decide(self, state_array, pos):
        state = torch.tensor(state_array, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # [수정 ⑥ 연계] 반환값 언패킹
            q_long, q_short, tau_long, tau_short, unc = self.model(state, num_quantiles=32)
            if torch.isnan(q_long).any(): print("WARNING: q_long contains NaN")
            unc_score = unc[0, 0].item()
            dynamic_cvar = self.cvar_threshold * (1.0 - 0.5 * unc_score)
            k = max(4, int(32 * dynamic_cvar))
            
            sort_idx_l = tau_long[0, :, 0].argsort()
            sort_idx_s = tau_short[0, :, 0].argsort()
            q_l = q_long[0][sort_idx_l][:k].mean(dim=0).cpu()
            q_s = q_short[0][sort_idx_s][:k].mean(dim=0).cpu()

        adv_long  = q_l[1].item() - q_l[0].item()
        adv_short = q_s[1].item() - q_s[0].item()
        cur_pos   = pos.get('type')

        if cur_pos == 'LONG':
            if adv_long < 0: return 0, 0.0, {'agent': 'LONG_EXIT'}
            return 1, 1.0, {'agent': 'HOLD_LONG'}
        elif cur_pos == 'SHORT':
            if adv_short < 0: return 0, 0.0, {'agent': 'SHORT_EXIT'}
            return 2, 1.0, {'agent': 'HOLD_SHORT'}

        # [수정 ⑧] Val과 Train의 정책 완벽 동기화 (Agent.act와 동일)
        dynamic_threshold = 0.005 * (1.0 + unc_score)
        
        if adv_long > adv_short and adv_long > dynamic_threshold:
            return 1, 1.0, {'agent': 'LONG_ENTRY', 'adv': adv_long, 'thresh': dynamic_threshold}
        elif adv_short > adv_long and adv_short > dynamic_threshold:
            return 2, 1.0, {'agent': 'SHORT_ENTRY', 'adv': adv_short, 'thresh': dynamic_threshold}
            
        return 0, 0.0, {'agent': 'FLAT', 'adv_L': adv_long, 'adv_S': adv_short, 'thresh': dynamic_threshold}


# ═══════════════════════════════════════════════════════════════════════════
# 메인 훈련 루프
# ═══════════════════════════════════════════════════════════════════════════
def train_ls():
    CSV_PATH = 'data/ensemble/rl_training_data_full.csv'
    if not os.path.exists(CSV_PATH): return logger.error("데이터가 없습니다.")

    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_val   = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_train_reg = df_train[REGIME_COLS].values.astype(np.float32)
    MAX_EP = 4096; _safe_end = len(df_train) - MAX_EP - 1

    ri = {r: REGIME_COLS.index(f'regime_{r}') for r in ['bull', 'bear', 'chop', 'whipsaw', 'normal']}
    good_starts = [i for i in range(_safe_end)
                   if df_train_reg[i, ri['bull']] == 1.0 or df_train_reg[i, ri['bear']] == 1.0]
    if len(good_starts) < 100: good_starts = list(range(_safe_end))

    # ── 모델/에이전트 초기화 ──
    model = DualBrainIQN(STACKED_STATE_DIM, hidden_dim=64, raw_state_dim=STATE_DIM).to(device)
    agent = DualBrainAgent(model, device=device, cvar_threshold=CVAR_THRESHOLD)
    agent.memory = PrioritizedReplayBuffer(200000)

    # ── 커리큘럼 러닝 (Curriculum Learning) 목표 설정 ──
    TARGET_FEE = 0.0005   # 최종 도달할 실전 수수료 (0.05%)
    TARGET_SLIP = 0.0002  # 최종 도달할 실전 슬리피지 (0.02%)
    CURRICULUM_EPS = 300  # 300 에피소드에 걸쳐 모래주머니를 서서히 채웁니다

    # 단일 환경 초기화 (초기 수수료는 루프 안에서 0으로 덮어씌워집니다)
    env = TradingEnv(df_train, phase='train', agent_role='dual', fee=TARGET_FEE, slip=TARGET_SLIP)

    NEP, BATCH, UPDATE_FREQ, MIN_BUFFER = 1000, 512, 64, 2048
    global_step = 0
    EPS_START, EPS_END, EPS_DECAY_STEPS = 1.0, 0.01, 200_000 # EPS_END를 0.05에서 0.01로 더 낮춰 완전히 자율에 맡깁니다.

    os.makedirs('data/ensemble/ckpt', exist_ok=True)
    best_val_score = -float('inf')
    val_pnl_history = []
    start_ep = 1
    CHECKPOINT_PATH = 'data/ensemble/ckpt/dualbrain_checkpoint.pth'

    def _save_checkpoint(epoch):
        torch.save({
            'model': model.state_dict(),
            'optimizer': agent.optimizer.state_dict(),
            'global_step': global_step,
            'best_val_score': best_val_score,
            'val_pnl_history': val_pnl_history,
            'epoch': epoch,
        }, CHECKPOINT_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt['model'], strict=False)
            agent.optimizer.load_state_dict(ckpt['optimizer'])
            start_ep   = ckpt['epoch'] + 1
            global_step = ckpt['global_step']
            best_val_score = ckpt['best_val_score']
            val_pnl_history = ckpt.get('val_pnl_history', [])
            logger.info(f"♻️ 재시작 ep={start_ep}")
        except Exception as e:
            logger.warning(f"체크포인트 로드 실패({e}), 처음부터 시작")

    try:
        ep = start_ep
        for ep in range(start_ep, NEP + 1):
            
            # 수수료 커리큘럼 없이 100% 적용
            env.fee = TARGET_FEE
            env.slip = TARGET_SLIP
            curriculum_ratio = 1.0 

            s_idx = random.choice(good_starts) if random.random() < 0.7 else random.randint(0, _safe_end)
            state = env.reset(s_idx)
            ep_reward, done = 0.0, False

            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

            while not done:
                global_step += 1
                eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

                if env.current_step >= env.end_step or env.current_step >= len(df_train) - 1:
                    break

                a = agent.act(state, eps)
                ns, r, done, info = env.step(a)

                real_done = 0.0 if info.get('timeout', False) else float(done)
                was_idle  = (a == 0 and env.pos is None)

                # [수정 ⑤] FLAT 상태일 때 보상을 0.0으로 강제하는 것을 삭제하여 관망의 진짜 가치를 학습
                agent.memory.push(state, a, r, ns, real_done)
                
                if not was_idle: ep_reward += r
                state = ns

                if global_step % UPDATE_FREQ == 0:
                    if len(agent.memory) >= MIN_BUFFER:
                        agent.update(BATCH)

            _SIGMA_FLOOR = 0.3
            sigma_vals = []
            for m in model.modules():
                if isinstance(m, NoisyLinear):
                    m.weight_sigma.data.clamp_(min=_SIGMA_FLOOR)
                    m.bias_sigma.data.clamp_(min=_SIGMA_FLOOR)
                    sigma_vals.append(m.weight_sigma.data.mean().item())
            avg_sigma = np.mean(sigma_vals) if sigma_vals else 0.0

            _s_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                # [수정 ⑥ 연계] 반환값 개수 수정
                _, _, _, _, _unc = model(_s_t, num_quantiles=8)
            avg_unc = _unc[0, 0].item()

            pnl     = (env.balance / env.initial_balance - 1) * 100
            rpt_pct = (env.fee + env.slip) * 2 * 100
            fee_drag = env.total_trades * rpt_pct
            rpt = ep_reward / max(1, env.total_trades)
            
            logger.info(
                f"Ep {ep:04d} PnL:{pnl:6.1f}%  Tr:{env.total_trades:4d}  WR:{env.win_rate*100:4.0f}%  "
                f"Rew/tr:{rpt:+.3f} | fee≈{fee_drag:.1f}% (C-Ratio:{curriculum_ratio*100:3.0f}%) | buf:{len(agent.memory):6d} | eps:{eps:.3f} | σ:{avg_sigma:.4f} | unc:{avg_unc:.3f}"
            )

            if ep % 10 == 0:
                router  = SimpleRouter(model, device=device)
                
                # 1. 검증 환경 초기화 시 phase='train'과 동일한 로직을 타도록 강제할 수 없으나,
                # df_val의 컬럼 순서가 df_train과 100% 동일한지 확인. (이미 위에서 분할했으므로 동일함)
                val_env = TradingEnv(df_val, phase='val', agent_role='dual', fee=TARGET_FEE, slip=TARGET_SLIP)
                obs = val_env.reset() 
                d  = False

                model.eval()
                try:
                    flat_count = 0
                    q_stats = {'long_max': [], 'short_max': []} # 디버깅용 Q값 추적
                    
                    while not d:
                        # 2. pos_info 딕셔너리 구성 (그대로 유지)
                        pos_info = {'type': val_env.pos, 'entry_price': val_env.entry_price,
                                    'unrealized': val_env.unrealized_pnl, 'mdd': val_env.max_drawdown,
                                    'hold_norm': val_env.hold_count / 144}
                        
                        # 3. 라우터 결정 (router 내부의 임계값은 없는 상태여야 함)
                        action, _, info_r = router.decide(obs, pos_info)
                        
                        # 디버깅: 에이전트가 뱉어내는 실제 Q-Advantage 값 수집
                        if 'adv_L' in info_r: q_stats['long_max'].append(info_r['adv_L'])
                        if 'adv_S' in info_r: q_stats['short_max'].append(info_r['adv_S'])
                        
                        if info_r.get('agent') == 'FLAT': flat_count += 1
                        
                        obs, _, d, _ = val_env.step(action, leverage_rate=1.0)
                finally:
                    model.train()

                val_pnl_pct = (val_env.balance / val_env.initial_balance - 1) * 100
                val_pnl_history.append(val_pnl_pct)

                # 4. [디버그 로그 출력] 왜 거래를 안 하는지 실체를 확인합니다.
                mean_adv_l = np.mean(q_stats['long_max']) if q_stats['long_max'] else 0.0
                mean_adv_s = np.mean(q_stats['short_max']) if q_stats['short_max'] else 0.0
                
                if val_env.total_trades == 0:
                    logger.warning(f"    [VAL DIAG] Tr=0 — FLAT {flat_count}회 | Avg Adv(L): {mean_adv_l:.5f}, Avg Adv(S): {mean_adv_s:.5f}")

                if len(val_pnl_history) >= 3:
                    _h = val_pnl_history[-10:]
                    sharpe_est = float(np.clip(np.mean(_h) / max(float(np.std(_h)), 0.1), -10.0, 10.0))
                else:
                    sharpe_est = 0.0

                trade_activity = min(val_env.total_trades / 20.0, 1.0) * 5.0
                if val_pnl_pct <= 0:
                    val_score = val_pnl_pct - (trade_activity * 0.5)
                else:
                    val_score = val_pnl_pct * 1.5 + val_env.win_rate * 100 * 0.05 + sharpe_est * 3 + (trade_activity * 0.5)

                logger.info(f"    [VAL] PnL:{val_pnl_pct:.2f}% | Tr:{val_env.total_trades} | WR:{val_env.win_rate*100:.0f}% | Score:{val_score:.2f}")

                if val_score > best_val_score:
                    best_val_score = val_score
                    torch.save({'model': model.state_dict()}, 'data/ensemble/ckpt/best_dualbrain.pth')
                _save_checkpoint(ep)

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단. 체크포인트 저장.")
        _save_checkpoint(ep)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['generate_csv', 'train'])
    args = parser.parse_args()
    if args.mode == 'train': train_ls()