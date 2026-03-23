"""
Trading Router — 6-Agent Single-Directional MoE (v4 — Restructured + Phase 1/2/3 Fix)
================================================================================
[v4 변경 이력]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[Phase 1] 즉시 수정 — 학습 안정성 복구
  A. 에이전트별 독립 start_idx: 6개 에이전트가 각자 레짐 풀에서 시작점 샘플링
  B. HMM alpha 에피소드 리셋: env.reset() 시 alpha=pi, obs_buffer 클리어
  C. Kelly 워밍업: ep<100 동안 고정 레버리지 0.5, 이후 점진 활성화

[Phase 2] 보상 함수 재설계 — 학습 방향 교정
  D. 5-Component 보상:
     R1. PnL delta (×100 + tanh)  — 핵심 수익 신호
     R2. 드로다운 감시자 (²)       — 빠질수록 기하급수적 패널티
     R3. 트레이드 품질 (청산 시)    — 수익/손절/강제청산 차등
     R4. 시간 감쇠 (12봉 이후)     — 장기 홀딩 기회비용
     R5. 레짐 적응형 관망 패널티    — 추세장: 진입 압력 / 횡보: 관망 허용
  E. Replay 계층화 샘플링: 관망 priority 0.3배 감쇄
  F. pred_consensus 보너스 제거: 피쳐 해킹 방지

[Phase 3] 아키텍처 수정 — Phase 1-2 효과 확인 후 적용
  G. GatingNet Supervised Learning 전환:
     - REINFORCE 제거 → 레짐별 에이전트 밸리데이션 PnL 기반 소프트 라벨
     - cross-entropy 학습, 에이전트와 동기화된 업데이트 주기
  H. 피쳐 축소 반영 (사용자 수정본 유지):
     - STATE_DIM: signal(7)+elite(3)+alpha(6)+regime(1)+hmm(5)+synth(2)+pos(5)+mtf(3) = 32
     - STACK_N=2 → STACKED=64
     - Attention 4그룹: signal / elite+alpha / regime+hmm / synth

[기존 유지]
1. 6개 에이전트: bull/bear/chop_long/chop_short/normal_long/normal_short (2-Action)
2. 7-Way GatingNet: [flat, bull, bear, chop_L, chop_S, norm_L, norm_S]
3. OnlineHMMDetector: 4-state HMM (Bull/Bear/HVChop/LVRange)
4. KellyCriterionSizer: CVaR IQN 분위 기반 동적 포지션 사이징 (워밍업 추가)
5. MultiTimeframeFeatures: MTF_DIM=3 (4h_ret, 4h_trend, align)
6. MarketAttentionEncoder: 4그룹 Self-Attention (경량 ~3K params)
"""
import os, sys, logging, random, argparse, gc, copy
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import pytorch_lightning as pl
import warnings

logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pytorch_lightning")
warnings.filterwarnings("ignore", category=DeprecationWarning)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_SCRIPT_DIR)
for _p in [_ROOT_DIR, _SCRIPT_DIR, os.path.join(_ROOT_DIR, 'ensemble'), os.path.join(_ROOT_DIR, 'strategies')]:
    if _p not in sys.path: sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SuppressOutput:
    def __enter__(self):
        self._original_stdout = sys.stdout
        self._original_stderr = sys.stderr
        self._devnull_out = open(os.devnull, 'w')
        self._devnull_err = open(os.devnull, 'w')
        sys.stdout = self._devnull_out
        sys.stderr = self._devnull_err
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout = self._original_stdout
            sys.stderr = self._original_stderr
        finally:
            self._devnull_out.close()
            self._devnull_err.close()

# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ①] OnlineHMMDetector — 온라인 EM 기반 4-state HMM 레짐 감지기
# ═══════════════════════════════════════════════════════════════════════════
class OnlineHMMDetector:
    """온라인 EM(Baum-Welch)으로 학습하는 4-state HMM 레짐 감지기.

    은닉 상태 4개:
        0: Bull-Trend     (상승 추세, 낮은 변동성)
        1: Bear-Trend     (하락 추세, 낮은 변동성)
        2: High-Vol-Chop  (급등락 혼조, 높은 변동성)
        3: Low-Vol-Range  (저변동 횡보, 좁은 레인지)

    출력 피처 (HMM_DIM = 5차원):
        hmm_probs[0..3]  : 현재 스텝의 각 은닉 상태 사후확률 (합=1)
        hmm_entropy      : 상태 분포 엔트로피 (불확실성, 0~log4)

    [Phase 1-B] reset_episode(): 에피소드 시작 시 alpha/obs_buffer 리셋
    """

    N_STATES  = 4
    OBS_DIM   = 3
    MIN_STD   = 1e-3
    WINDOW    = 512

    def __init__(self):
        self.A = np.full((self.N_STATES, self.N_STATES), 0.05 / (self.N_STATES - 1))
        np.fill_diagonal(self.A, 0.85)
        self.A /= self.A.sum(axis=1, keepdims=True)

        self.pi = np.ones(self.N_STATES) / self.N_STATES

        self.mu = np.array([
            [ 0.8, -0.5, 0.3],
            [-0.8, -0.5, -0.3],
            [ 0.0,  1.5,  0.0],
            [ 0.0, -1.0,  0.0],
        ], dtype=np.float64)
        self.sigma = np.array([
            [0.5, 0.4, 0.5],
            [0.5, 0.4, 0.5],
            [1.0, 0.6, 0.8],
            [0.3, 0.3, 0.3],
        ], dtype=np.float64)

        self._obs_buffer: deque = deque(maxlen=self.WINDOW)
        self._alpha: np.ndarray = self.pi.copy()

        self._obs_mean = np.zeros(self.OBS_DIM)
        self._obs_std  = np.ones(self.OBS_DIM)
        self._fitted = False

    # ── [Phase 1-B] 에피소드 리셋 ──────────────────────────────────────────
    def reset_episode(self):
        """에피소드 시작 시 alpha를 초기 분포로 리셋하고 obs_buffer 클리어.
        이전 에피소드의 레짐 상태가 새 에피소드로 오염되는 것을 방지."""
        self._alpha = self.pi.copy()
        self._obs_buffer.clear()

    def _extract_obs(self, row: dict) -> np.ndarray:
        raw = np.array([
            float(row.get('log_return',      0.0)),
            float(row.get('garch_vol_z',     0.0)),
            float(row.get('oi_change_rate',  0.0)),
        ], dtype=np.float64)
        return (raw - self._obs_mean) / (self._obs_std + 1e-8)

    def _emission_log_prob(self, obs: np.ndarray) -> np.ndarray:
        diff  = obs[None, :] - self.mu
        var   = np.maximum(self.sigma ** 2, self.MIN_STD ** 2)
        lp    = -0.5 * np.sum((diff ** 2) / var + np.log(2 * np.pi * var), axis=1)
        return lp

    def _forward_step(self, obs: np.ndarray) -> np.ndarray:
        log_emit  = self._emission_log_prob(obs)
        predicted = self._alpha @ self.A
        log_joint = np.log(predicted + 1e-300) + log_emit
        log_joint -= log_joint.max()
        alpha_new  = np.exp(log_joint)
        alpha_new /= alpha_new.sum() + 1e-300
        self._alpha = alpha_new
        return alpha_new

    def fit(self, df: pd.DataFrame, n_iter: int = 30) -> None:
        needed = ['log_return', 'garch_vol_z', 'oi_change_rate']
        raw_mat = np.zeros((len(df), 3), dtype=np.float64)
        for i, col in enumerate(needed):
            if col in df.columns:
                raw_mat[:, i] = df[col].fillna(0).values
        self._obs_mean = raw_mat.mean(axis=0)
        self._obs_std  = raw_mat.std(axis=0).clip(min=1e-6)

        obs_seq = (raw_mat - self._obs_mean) / (self._obs_std + 1e-8)
        T = len(obs_seq)

        for _ in range(n_iter):
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(T)])

            log_alpha = np.zeros((T, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(
                        log_alpha[t-1] + np.log(self.A[:, j] + 1e-300)
                    ) + log_emit[t, j]

            log_beta = np.zeros((T, self.N_STATES))
            for t in range(T - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(
                        np.log(self.A[i, :] + 1e-300) + log_emit[t+1] + log_beta[t+1]
                    )

            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((T - 1, self.N_STATES, self.N_STATES))
            for t in range(T - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (log_alpha[t, i]
                                           + np.log(self.A[i, j] + 1e-300)
                                           + log_emit[t+1, j]
                                           + log_beta[t+1, j])
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
            xi = np.exp(log_xi)

            self.pi = gamma[0] / (gamma[0].sum() + 1e-300)
            self.A  = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            self.A /= self.A.sum(axis=1, keepdims=True) + 1e-300

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                self.mu[s]    = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff          = obs_seq - self.mu[s]
                self.sigma[s] = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w.sum()).clip(self.MIN_STD)

        self._alpha = gamma[-1]
        self._obs_buffer.extend(obs_seq[-self.WINDOW:].tolist())
        self._fitted = True
        logger.info(f"[HMM] fit 완료 | mu=\n{self.mu.round(3)}")

    def get_features(self, row: dict) -> np.ndarray:
        obs   = self._extract_obs(row)
        probs = self._forward_step(obs)
        ent   = float(-np.sum(probs * np.log(probs + 1e-300)))
        ent_n = ent / np.log(self.N_STATES + 1e-8)
        self._obs_buffer.append(obs.tolist())
        return np.concatenate([probs, [ent_n]]).astype(np.float32)

    def update_online(self, n_iter: int = 5) -> None:
        if len(self._obs_buffer) < 64:
            return
        obs_seq = np.array(self._obs_buffer, dtype=np.float64)
        T = len(obs_seq)

        A_old = self.A.copy()
        for _ in range(n_iter):
            log_emit = np.stack([self._emission_log_prob(obs_seq[t]) for t in range(T)])
            log_alpha = np.zeros((T, self.N_STATES))
            log_alpha[0] = np.log(self.pi + 1e-300) + log_emit[0]
            for t in range(1, T):
                for j in range(self.N_STATES):
                    log_alpha[t, j] = np.logaddexp.reduce(
                        log_alpha[t-1] + np.log(self.A[:, j] + 1e-300)
                    ) + log_emit[t, j]

            log_beta = np.zeros((T, self.N_STATES))
            for t in range(T - 2, -1, -1):
                for i in range(self.N_STATES):
                    log_beta[t, i] = np.logaddexp.reduce(
                        np.log(self.A[i, :] + 1e-300) + log_emit[t+1] + log_beta[t+1]
                    )

            log_gamma = log_alpha + log_beta
            log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            log_xi = np.zeros((T - 1, self.N_STATES, self.N_STATES))
            for t in range(T - 1):
                for i in range(self.N_STATES):
                    for j in range(self.N_STATES):
                        log_xi[t, i, j] = (log_alpha[t, i]
                                           + np.log(self.A[i, j] + 1e-300)
                                           + log_emit[t+1, j]
                                           + log_beta[t+1, j])
                log_xi[t] -= np.logaddexp.reduce(log_xi[t].reshape(-1))
            xi = np.exp(log_xi)

            A_new = xi.sum(axis=0) / (gamma[:-1].sum(axis=0, keepdims=True).T + 1e-300)
            A_new /= A_new.sum(axis=1, keepdims=True) + 1e-300
            self.A = 0.8 * A_old + 0.2 * A_new

            for s in range(self.N_STATES):
                w = gamma[:, s] + 1e-300
                new_mu    = (w[:, None] * obs_seq).sum(axis=0) / w.sum()
                diff      = obs_seq - new_mu
                new_sigma = np.sqrt((w[:, None] * diff ** 2).sum(axis=0) / w.sum()).clip(self.MIN_STD)
                self.mu[s]    = 0.85 * self.mu[s]    + 0.15 * new_mu
                self.sigma[s] = 0.85 * self.sigma[s] + 0.15 * new_sigma

        self._alpha = gamma[-1]


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ②] KellyCriterionSizer — CVaR IQN 분위 기반 동적 포지션 사이징
# [Phase 1-C] warmup_until_ep: 지정 에폭까지 고정 레버리지 사용
# ═══════════════════════════════════════════════════════════════════════════
class KellyCriterionSizer:
    def __init__(self, half_kelly: float = 0.5,
                 min_lev: float = 0.1, max_lev: float = 1.0,
                 uncertainty_cap: float = 0.5,
                 warmup_lev: float = 0.5, warmup_until_ep: int = 100):
        self.half_kelly      = half_kelly
        self.min_lev         = min_lev
        self.max_lev         = max_lev
        self.uncertainty_cap = uncertainty_cap
        self.warmup_lev      = warmup_lev
        self.warmup_until_ep = warmup_until_ep
        self._current_ep     = 0

    def set_epoch(self, ep: int):
        self._current_ep = ep

    def compute(self, q_quantiles: torch.Tensor, gating_confidence: float = 1.0) -> float:
        # [Phase 1-C] 워밍업 기간에는 고정 레버리지
        if self._current_ep < self.warmup_until_ep:
            return self.warmup_lev

        with torch.no_grad():
            q = q_quantiles.float().cpu()
            if q.shape[1] < 2:
                return self.min_lev

            q0 = q[:, 0]
            q1 = q[:, 1]

            win_rate = float((q1 > q0).float().mean())
            adv = q1 - q0
            pos_mask = adv > 0
            neg_mask = adv < 0

            pos_mean = float(adv[pos_mask].mean()) if pos_mask.any() else 0.0
            neg_mean = float(adv[neg_mask].abs().mean()) if neg_mask.any() else 1e-6
            payoff   = pos_mean / (neg_mean + 1e-8)

            p, q_val = win_rate, 1.0 - win_rate
            b        = max(payoff, 0.1)
            f_star   = (p * b - q_val) / b
            f_half   = max(f_star * self.half_kelly, 0.0)

            total_var = float(q1.var() + 1e-8)
            adv_var   = float(adv.var() + 1e-8)
            uncertainty = min(adv_var / total_var, self.uncertainty_cap)
            f_penalized = f_half * (1.0 - uncertainty)

            f_final = f_penalized * float(np.clip(gating_confidence, 0.0, 1.0))

            # [Phase 1-C] 워밍업→실전 전환 시 점진적 블렌딩
            if self._current_ep < self.warmup_until_ep + 50:
                blend = (self._current_ep - self.warmup_until_ep) / 50.0
                kelly_lev = float(np.clip(f_final, self.min_lev, self.max_lev))
                return self.warmup_lev * (1 - blend) + kelly_lev * blend

            return float(np.clip(f_final, self.min_lev, self.max_lev))

    def log_stats(self, q_quantiles: torch.Tensor) -> dict:
        with torch.no_grad():
            q = q_quantiles.float().cpu()
            q0, q1 = q[:, 0], q[:, 1]
            adv      = q1 - q0
            win_rate = float((adv > 0).float().mean())
            adv_pos  = float(adv[adv > 0].mean()) if (adv > 0).any() else 0.0
            adv_neg  = float(adv[adv < 0].abs().mean()) if (adv < 0).any() else 0.0
            payoff   = adv_pos / (adv_neg + 1e-8)
            f_star   = (win_rate * payoff - (1 - win_rate)) / (payoff + 1e-8)
        return {'win_rate': round(win_rate, 3), 'payoff': round(payoff, 3),
                'f_star': round(f_star, 3), 'f_half': round(max(f_star * self.half_kelly, 0), 3)}


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ③] MultiTimeframeFeatures — MTF_DIM=3 (4h_ret, 4h_trend, align)
# ═══════════════════════════════════════════════════════════════════════════
class MultiTimeframeFeatures:
    """MTF_DIM=3: [4h_ret, 4h_trend, htf_alignment]"""

    _RET_SCALE     = 50.0
    _VOL_SCALE     = 10.0
    _VOL_1H_WINDOW = 4

    def __init__(self, close_arr: np.ndarray, w1h: int = 1, w4h: int = 4):
        self.w1h  = w1h
        self.w4h  = w4h
        self._cache = self._precompute(close_arr.astype(np.float64))

    @staticmethod
    def _linreg_slope(y: np.ndarray) -> float:
        n = len(y)
        if n < 3: return 0.0
        x  = np.arange(n, dtype=np.float64)
        xm, ym = x.mean(), y.mean()
        denom = ((x - xm) ** 2).sum()
        if denom < 1e-12: return 0.0
        slope = ((x - xm) * (y - ym)).sum() / denom
        price_range = max(y.max() - y.min(), abs(ym) * 0.001, 1e-8)
        return float(np.clip(slope * n / price_range, -1.0, 1.0))

    @staticmethod
    def _logret_slope(logret: np.ndarray) -> float:
        if len(logret) < 2: return 0.0
        mean_ret = logret.mean()
        if abs(mean_ret) < 1e-3: return 0.0
        return float(np.clip(mean_ret * 100.0, -1.0, 1.0))

    def _precompute(self, close: np.ndarray) -> np.ndarray:
        T   = len(close)
        out = np.zeros((T, MTF_DIM), dtype=np.float32)
        logret = np.zeros(T, dtype=np.float64)
        logret[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-8))

        for i in range(T):
            sv     = max(0, i - self._VOL_1H_WINDOW + 1)
            lr1w   = logret[sv:i+1]
            trend1 = self._logret_slope(lr1w) if len(lr1w) >= 2 else 0.0

            s4     = max(0, i - self.w4h + 1)
            c4     = close[s4:i+1]
            ret4   = float(np.tanh((c4[-1] / c4[0] - 1) * self._RET_SCALE)) if len(c4) > 1 else 0.0
            trend4 = self._linreg_slope(c4) if len(c4) >= 3 else 0.0
            align  = float(np.sign(trend1) * np.sign(trend4)) if (trend1 != 0 and trend4 != 0) else 0.0

            out[i] = [ret4, trend4, align]

        logger.info(
            f"[MTF] 선계산 완료 | shape={out.shape} | "
            f"4h_ret μ={out[:,0].mean():.3f} σ={out[:,0].std():.3f} | "
            f"align: +{(out[:,2]>0).mean()*100:.1f}% / 0:{(out[:,2]==0).mean()*100:.1f}% / -{(out[:,2]<0).mean()*100:.1f}%"
        )
        return out

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= len(self._cache):
            return np.zeros(MTF_DIM, dtype=np.float32)
        return self._cache[idx]


# ═══════════════════════════════════════════════════════════════════════════
# [융합 모듈 ④] MarketAttentionEncoder — 4그룹 Self-Attention
# ═══════════════════════════════════════════════════════════════════════════
class MarketAttentionEncoder(nn.Module):
    """4그룹 토큰: signal(7) / elite_alpha(12) / regime_hmm(6) / synth(2)
    FEATURE_DIM 내 순서: signal(7) elite(6) alpha(6) regime(1) hmm(5) synth(2) = 27
    """

    _GROUPS = [
        ('signal',       0,  7),   # signal(7) = pred × conf
        ('elite_alpha',  7, 12),   # elite(6) + alpha(6) = 12
        ('regime_hmm',  19,  6),   # regime(1) + hmm(5) = 6
        ('synth',       25,  2),   # synth(2): ofti + kel
    ]
    D_MODEL  = 16
    N_HEADS  = 2
    N_LAYERS = 2

    def __init__(self, out_dim: int, raw_state_dim: int = None):
        super().__init__()
        self.out_dim = out_dim
        self.proj = nn.ModuleList([
            nn.Sequential(nn.Linear(g_dim, self.D_MODEL), nn.LayerNorm(self.D_MODEL))
            for _, _, g_dim in self._GROUPS
        ])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.D_MODEL, nhead=self.N_HEADS,
            dim_feedforward=self.D_MODEL * 2,
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.attn = nn.TransformerEncoder(encoder_layer, num_layers=self.N_LAYERS)
        n_groups = len(self._GROUPS)
        self.out_proj = nn.Sequential(
            nn.Linear(n_groups * self.D_MODEL, out_dim),
            nn.LayerNorm(out_dim),
            nn.SiLU()
        )

    def forward(self, raw_feat: torch.Tensor) -> torch.Tensor:
        tokens = []
        for (_, start, length), proj_layer in zip(self._GROUPS, self.proj):
            g = raw_feat[:, start:start + length]
            tokens.append(proj_layer(g))
        tokens = torch.stack(tokens, dim=1)
        attended = self.attn(tokens)
        flat = attended.flatten(1)
        return self.out_proj(flat)


# ═══════════════════════════════════════════════════════════════════════════
# [상수 및 차원 정의] — 사용자 피쳐 축소 반영
# ═══════════════════════════════════════════════════════════════════════════
MODEL_PRED = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst', 'pred_tide', 'pred_mdjd', 'pred_ridge']
MODEL_CONF = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst', 'conf_tide', 'conf_mdjd', 'conf_ridge']

ELITE_COLS = ['sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze']
ALPHA_7_COLS = ['session_us', 'hour_cos', 'cvp_poc_dist', 'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate']
REGIME_COLS = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']

TARGET_COL = 'log_return'
SYNTHETIC_ALPHA_COLS = ['ofti', 'kel', 'mta_funding', 'svps', 'cada', 'mshd', 'fvci',
                        'wpad', 'fdlv', 'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz']

STATE_PRED  = ['pred_tide', 'pred_ridge', 'pred_patchtst', 'pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_mdjd']
STATE_CONF  = ['conf_tide', 'conf_ridge', 'conf_patchtst', 'conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_mdjd']
STATE_ELITE = ['sig_ai_squeeze', 'sig_whale', 'sig_oi_divergence', 'sig_volume_confirm', 'sig_liquidity_trap', 'sig_trend_health']
STATE_ALPHA = ['hour_cos', 'garch_vol_z', 'breakout_strength', 'fvg_dist', 'oi_change_rate', 'cvp_volume_imbalance']
STATE_SYNTH = ['ofti', 'kel']

HMM_N_STATES = 4
HMM_DIM      = HMM_N_STATES + 1   # 5
MTF_DIM      = 3
FEATURE_DIM  = len(STATE_PRED) + len(STATE_ELITE) + len(STATE_ALPHA) + 1 + HMM_DIM + len(STATE_SYNTH)
# = 7(signal) + 6(elite) + 6(alpha) + 1(regime_idx) + 5(hmm) + 2(synth) = 27
STATE_DIM    = FEATURE_DIM + 5 + MTF_DIM   # 35
STACK_N      = 2
STACKED_STATE_DIM = STATE_DIM * STACK_N  # 70


# ═══════════════════════════════════════════════════════════════════════════
# 2. 거래 환경 (TradingEnv) — [Phase 2] 5-Component 보상 함수
# ═══════════════════════════════════════════════════════════════════════════
class TradingEnv:
    STATE_DIM = STATE_DIM

    def __init__(self, df, initial_balance=10000.0, fee=0.0005, slip=0.0002, phase='train', agent_role='neutral',
                 hmm_detector=None, mtf_features=None):
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.fee = fee
        self.slip = slip
        self.phase = phase
        self.agent_role = agent_role
        self.hmm_detector = hmm_detector
        if mtf_features is not None:
            self.mtf = mtf_features
        else:
            close_arr = self.df['close'].values.astype(np.float32)
            self.mtf = MultiTimeframeFeatures(close_arr)

        self.MAX_EPISODE_STEPS = 4096 if phase == 'train' else len(self.df) - 1
        self.MAX_LEVERAGE = 1.0

        feat_cols = STATE_PRED + STATE_CONF + STATE_ELITE + STATE_ALPHA + REGIME_COLS + STATE_SYNTH
        self._feat_np  = self.df[feat_cols].values.astype(np.float32)
        self._close_np = self.df['close'].values.astype(np.float32)
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

        # [Phase 2-F] 청산 추적 변수 (트레이드 품질 보상용)
        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = False

        # [Phase 1-B] HMM alpha 리셋
        if self.hmm_detector is not None:
            self.hmm_detector.reset_episode()

        self._frame_stack.clear()
        return self._get_stacked_state(self._build_state(self.current_step))

    def step(self, action, leverage_rate=1.0):
        current_price = self._close_np[self.current_step]

        if self.pos is not None:
            prev_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            prev_portfolio_value = self.balance

        global_action = action
        if self.agent_role in ['bear', 'chop_short', 'normal_short'] and action == 1:
            global_action = 2

        force_close = False
        if self.pos is not None and self.unrealized_pnl <= -0.025:
            force_close = True

        is_entering_long = False
        is_entering_short = False
        is_closing = False

        if force_close:
            is_closing = True
        else:
            if global_action == 0:
                if self.pos is not None: is_closing = True
            elif global_action == 1:
                if self.pos is None: is_entering_long = True
                elif self.pos == 'SHORT': is_closing = True
            elif global_action == 2:
                if self.pos is None: is_entering_short = True
                elif self.pos == 'LONG': is_closing = True

        # [Phase 2] 청산 추적 초기화
        self._just_closed = False
        self._last_realized_pnl = 0.0
        self._was_force_closed = force_close

        if is_entering_long:
            self.pos = 'LONG'
            self.entry_price = current_price * (1 + self.slip)
            self.entry_idx = self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
            self.active_steps += 1

        elif is_entering_short:
            self.pos = 'SHORT'
            self.entry_price = current_price * (1 - self.slip)
            self.entry_idx = self.current_step
            self.current_leverage = self.MAX_LEVERAGE * leverage_rate
            self.balance -= self.balance * self.fee * self.current_leverage
            self.active_steps += 1

        elif is_closing:
            if self.pos == 'LONG': realized_pnl = (current_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else: realized_pnl = (self.entry_price - current_price * (1 + self.slip)) / self.entry_price

            realized_pnl *= self.current_leverage
            self.balance += self.balance * realized_pnl
            self.balance -= self.balance * self.fee * self.current_leverage

            self.total_trades += 1
            if realized_pnl > 0: self.win_trades += 1

            # [Phase 2] 청산 기록
            self._just_closed = True
            self._last_realized_pnl = realized_pnl

            self.pos = None
            self.current_leverage = 0.0
            self.hold_count = 0
            self.unrealized_pnl = 0.0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        self.current_step += 1
        done = self.current_step >= self.end_step
        next_price = self._close_np[self.current_step] if not done else current_price

        if self.pos is not None:
            self.hold_count = self.current_step - self.entry_idx

            if self.pos == 'LONG':
                est_exit_price = next_price * (1 - self.slip)
                raw_pnl = (est_exit_price - self.entry_price) / self.entry_price
            else:
                est_exit_price = next_price * (1 + self.slip)
                raw_pnl = (self.entry_price - est_exit_price) / self.entry_price

            self.unrealized_pnl = raw_pnl * self.current_leverage
            self.max_drawdown = min(self.max_drawdown, self.unrealized_pnl)
            self.peak_pnl = max(self.peak_pnl, self.unrealized_pnl)
            self.active_steps += 1

        # ══════════════════════════════════════════════════════════════
        # [Phase 2-D] 5-Component 보상 함수
        # ══════════════════════════════════════════════════════════════

        # R1. PnL delta (tanh로 자연 바운딩)
        # ×50 스케일: 0.1% 변동 → 0.05 → tanh=0.05 (선형 구간)
        #             수수료(-0.05%) → -0.025 → tanh=-0.025 (진입 비용이 과도하게 크지 않음)
        if self.pos is not None:
            current_portfolio_value = self.balance * (1.0 + self.unrealized_pnl)
        else:
            current_portfolio_value = self.balance

        step_delta = (current_portfolio_value - prev_portfolio_value) / (prev_portfolio_value + 1e-8) * 50.0
        r1_pnl = float(np.tanh(step_delta))

        # R2. 드로다운 감시자 (제곱 스케일 — 빠질수록 기하급수적)
        # -1% 이상 빠져야 패널티 시작 (수수료 + 슬리피지 자연 손실 구간은 허용)
        r2_drawdown = 0.0
        if self.pos is not None and self.unrealized_pnl < -0.01:
            dd_ratio = abs(self.unrealized_pnl) / 0.025
            r2_drawdown = -0.1 * (dd_ratio ** 2)

        # R3. 트레이드 품질 (청산 시점에만 발생)
        r3_quality = 0.0
        if self._just_closed:
            if self._was_force_closed:
                r3_quality = -0.30       # 강제청산: 큰 패널티
            elif self._last_realized_pnl > 0:
                r3_quality = 0.15 * min(self._last_realized_pnl / 0.01, 1.0)  # 수익 청산
            else:
                r3_quality = -0.05       # 손절: 작은 패널티 (빠른 손절 유도)

        # R4. 시간 감쇠 (12봉 이후 점진적 비용)
        r4_time_decay = 0.0
        if self.pos is not None and self.hold_count > 12:
            r4_time_decay = -0.003 * (self.hold_count - 12) / 72.0

        # R5. 레짐 적응형 관망 패널티
        # [FIX] 수수료 비용(진입 시 tanh ≈ -0.025)보다 충분히 작아야 함
        # 기존 -0.015는 10스텝만 관망해도 -0.15 → 진입보다 비싸서 과잉 거래 유발
        # 수정: 추세장에서도 100스텝 관망 ≈ -0.3 → 한 번의 수익 트레이드로 상쇄 가능
        r5_idle = 0.0
        if self.pos is None:
            regime_raw = self._feat_np[min(self.current_step, len(self._feat_np)-1)]
            o = self._n_pred + self._n_conf + self._n_elite + self._n_alpha
            regime_vec = regime_raw[o:o+self._n_regime]
            regime_idx = int(np.argmax(regime_vec))
            # 0=chop, 1=whipsaw, 2=bull, 3=bear, 4=normal
            if regime_idx in (2, 3):    # 추세장: 약한 진입 압력
                r5_idle = -0.003
            elif regime_idx in (0, 1):  # 횡보/위프소: 관망 거의 무료
                r5_idle = -0.0003
            else:                       # normal
                r5_idle = -0.001
                r5_idle = -0.005

        # 종합 보상 (tanh로 자연 바운딩)
        raw_reward = r1_pnl + r2_drawdown + r3_quality + r4_time_decay + r5_idle
        reward = float(np.tanh(raw_reward))

        # 에피소드 종료 시 강제 청산
        if done and self.pos is not None:
            ep_end_price = self._close_np[min(self.current_step, len(self._close_np) - 1)]
            if self.pos == 'LONG':
                ep_realized = (ep_end_price * (1 - self.slip) - self.entry_price) / self.entry_price
            else:
                ep_realized = (self.entry_price - ep_end_price * (1 + self.slip)) / self.entry_price
            ep_realized *= self.current_leverage
            self.balance += self.balance * ep_realized
            self.balance -= self.balance * self.fee * self.current_leverage
            self.total_trades += 1
            if ep_realized > 0: self.win_trades += 1
            self.pos = None
            self.current_leverage = 0.0
            self.unrealized_pnl = 0.0
            self.hold_count = 0
            self.peak_pnl = 0.0
            self.max_drawdown = 0.0

        info = {'pnl_pct': (self.balance / self.initial_balance - 1) * 100, 'wr': self.win_trades / max(1, self.total_trades)}
        return self._get_stacked_state(self._build_state(self.current_step)), reward, done, info

    @property
    def win_rate(self): return self.win_trades / max(1, self.total_trades)

    def _get_stacked_state(self, raw_state):
        self._frame_stack.append(raw_state)
        pad = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        return np.concatenate(frames)

    def _build_state(self, idx):
        if idx < 0 or idx >= len(self._feat_np):
            return np.zeros(STATE_DIM, dtype=np.float32)

        row = self._feat_np[idx]
        o = 0
        preds      = row[o:o+self._n_pred];   o += self._n_pred
        confs      = row[o:o+self._n_conf];   o += self._n_conf
        signal     = preds * confs
        elite      = row[o:o+self._n_elite];  o += self._n_elite
        alpha6     = row[o:o+self._n_alpha];  o += self._n_alpha
        regime_raw = row[o:o+self._n_regime]; o += self._n_regime
        regime_idx = np.array([float(np.argmax(regime_raw))], dtype=np.float32)
        synth2     = row[o:o+self._n_synth]

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
# 2-2. 리플레이 버퍼 — [Phase 2-E] 관망 priority 감쇄
# ═══════════════════════════════════════════════════════════════════════════
class PrioritizedRegimeReplayBuffer:
    """[Phase 2-E] 관망 경험의 priority에 IDLE_DECAY=0.3 적용하여 자연 희석"""
    IDLE_DECAY = 0.3

    def __init__(self, capacity=200000, target_regimes=None, warmup_steps=14000,
                 alpha=0.8, beta=0.2, beta_anneal_steps=600_000):
        self.target_regimes     = target_regimes or []
        self.warmup_steps       = warmup_steps
        self._push_count        = 0
        self._total_stored      = 0
        self._cap               = capacity
        self._ptr               = 0
        self._size              = 0
        self._buf_s             = None
        self._buf_a             = np.empty(capacity, np.int32)
        self._buf_r             = np.empty(capacity, np.float32)
        self._buf_ns            = None
        self._buf_d             = np.empty(capacity, np.bool_)
        self._priorities        = np.zeros(capacity, np.float32)
        self.alpha              = alpha
        self.beta               = beta
        self._beta_start        = beta
        self._beta_anneal_steps = beta_anneal_steps
        self.max_priority       = 1.0

    def push(self, state, action, reward, next_state, done, current_regimes_dict, in_pos=False):
        self._push_count += 1
        should_add = False
        if self._push_count < self.warmup_steps or in_pos:
            should_add = True
        else:
            is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
            if is_target or random.random() < 0.1:
                should_add = True
        if not should_add:
            return
        if self._buf_s is None:
            sdim = len(state)
            self._buf_s  = np.empty((self._cap, sdim), np.float32)
            self._buf_ns = np.empty((self._cap, sdim), np.float32)
        is_target = any(current_regimes_dict.get(r, 0.0) == 1.0 for r in self.target_regimes)
        init_priority = self.max_priority * (1.5 if (self._push_count < self.warmup_steps and is_target) else 1.0)

        # [Phase 2-E] 관망 경험 priority 감쇄
        if not in_pos:
            init_priority *= self.IDLE_DECAY

        p = self._ptr
        self._buf_s[p]      = state
        self._buf_a[p]      = action
        self._buf_r[p]      = reward
        self._buf_ns[p]     = next_state
        self._buf_d[p]      = done
        self._priorities[p] = init_priority
        self._ptr          = (p + 1) % self._cap
        self._size         = min(self._size + 1, self._cap)
        self._total_stored += 1

    def sample(self, batch_size):
        self.beta = min(1.0, self._beta_start + (1.0 - self._beta_start) * (self._total_stored / self._beta_anneal_steps))
        pri   = self._priorities[:self._size] ** self.alpha
        probs = pri / (pri.sum() + 1e-8)
        indices = np.random.choice(self._size, batch_size, p=probs, replace=True)
        weights = (1.0 / (self._size * probs[indices] + 1e-8)) ** self.beta
        weights = (weights / weights.max()).astype(np.float32)
        return (self._buf_s[indices], self._buf_a[indices],
                self._buf_r[indices], self._buf_ns[indices],
                self._buf_d[indices].astype(np.float32), indices, weights)

    def update_priorities(self, indices, td_errors):
        for idx, err in zip(indices, td_errors):
            p = float(abs(err) + 1e-6) ** self.alpha
            self._priorities[idx] = p
            if p > self.max_priority:
                self.max_priority = p

    def __len__(self): return self._size


# ═══════════════════════════════════════════════════════════════════════════
# 3. 모델 아키텍처 (NoisyLinear, RobustIQN, IQNAgent)
# ═══════════════════════════════════════════════════════════════════════════
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon',   torch.empty(out_features))
        self._sigma_init = sigma_init
        self.reset_parameters()
        self.sample_noise()

    def reset_parameters(self):
        mu_range = 1.0 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self._sigma_init / self.in_features ** 0.5)

    def _f(self, x): return x.sign() * x.abs().sqrt()

    def sample_noise(self):
        eps_i = self._f(torch.randn(self.in_features,  device=self.weight_mu.device))
        eps_j = self._f(torch.randn(self.out_features, device=self.weight_mu.device))
        self.weight_epsilon.copy_(eps_j.ger(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def zero_noise(self):
        self.weight_epsilon.zero_()
        self.bias_epsilon.zero_()

    def forward(self, x):
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_epsilon
            b = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            w, b = self.weight_mu, self.bias_mu
        return F.linear(x, w, b)

class RobustIQN(nn.Module):
    def __init__(self, state_dim, action_dim=2, hidden_dim=128, raw_state_dim=None):
        super().__init__()
        self.action_dim = action_dim
        _raw = raw_state_dim if raw_state_dim is not None else state_dim

        self.attn_encoder = MarketAttentionEncoder(out_dim=hidden_dim, raw_state_dim=_raw)
        self.feat_extractor = nn.Sequential(
            nn.Linear(state_dim + hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 64),                     nn.LayerNorm(64),         nn.SiLU()
        )
        # [BUGFIX] context_gate: pos(5)를 제외한 마켓 피쳐 = FEATURE_DIM(27) + MTF_DIM(3) = 30
        # 기존 _raw - 5 = 27이지만 연속 슬라이싱 시 pos 앞 3차원 포함 + mtf 누락 버그
        # → FEATURE_DIM과 MTF_DIM을 명시적으로 사용하여 정확한 추출 보장
        self._market_dim    = FEATURE_DIM + MTF_DIM  # 27 + 3 = 30
        self._raw_state_dim = _raw
        self.context_gate   = nn.Linear(self._market_dim, 64)
        self._cos_dim       = 64  # IQN cos basis 차원 (phi와 동기화)
        self.phi            = nn.Linear(self._cos_dim, 64)

        self.v_head = nn.Sequential(nn.SiLU(), nn.Linear(64, 1))
        self.a_head = nn.Sequential(nn.SiLU(), NoisyLinear(64, action_dim, sigma_init=0.05))

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear): m.sample_noise()

    def forward(self, state, num_quantiles=8):
        batch_size = state.size(0)

        last_frame_start = state.shape[1] - self._raw_state_dim
        market_feat      = state[:, last_frame_start : last_frame_start + FEATURE_DIM]
        attn_out         = self.attn_encoder(market_feat)

        feat_input = torch.cat([state, attn_out], dim=1)
        feat       = self.feat_extractor(feat_input)

        # [BUGFIX] pos(5)를 건너뛰고 FEATURE_DIM(24) + MTF(3)를 정확히 추출
        # STATE 레이아웃: [signal(7) elite(3) alpha(6) regime(1) hmm(5) synth(2)] [pos(5)] [mtf(3)]
        #                 |<-------------- FEATURE_DIM=24 ------------->|  skip    |<-MTF->|
        pos_start = last_frame_start + FEATURE_DIM
        mtf_start = pos_start + 5
        market_no_pos = torch.cat([
            state[:, last_frame_start : pos_start],       # FEATURE_DIM [0:24]
            state[:, mtf_start : last_frame_start + STATE_DIM]  # MTF [29:32]
        ], dim=1)
        gate = torch.sigmoid(self.context_gate(market_no_pos))
        feat = feat * gate

        tau     = torch.rand(batch_size, num_quantiles, 1, device=state.device)
        cos_tau = torch.cos(tau * torch.arange(1, self._cos_dim + 1, device=state.device).float() * torch.pi)
        phi_x   = self.phi(cos_tau)
        shared  = feat.unsqueeze(1).expand(-1, num_quantiles, -1) * phi_x

        v = self.v_head(shared)
        a = self.a_head(shared)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q, tau


class IQNAgent:
    NUM_QUANTILES = 32

    def __init__(self, model, lr=5e-5, gamma=0.99, tau=0.005, device='cuda', cvar_threshold=0.25):
        self.model = model
        self.state_dim = model.feat_extractor[0].in_features
        self.target_model = copy.deepcopy(model).to(device)
        self.target_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        self.memory = None
        self.gamma = gamma
        self.tau = tau
        self.device = device
        self.cvar_threshold = cvar_threshold

    def act(self, state, eps=0.0):
        if eps > 0.0 and random.random() < eps:
            return random.randrange(self.model.action_dim)
        if self.model.training and hasattr(self.model, 'reset_noise'):
            self.model.reset_noise()
        state_ts = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_quantiles, tau = self.model(state_ts, num_quantiles=self.NUM_QUANTILES)
            sort_idx = tau[0, :, 0].argsort()
            q_sorted = q_quantiles[0][sort_idx]
            cvar_k   = max(1, int(self.NUM_QUANTILES * self.cvar_threshold))
            q = q_sorted[:cvar_k].mean(dim=0)
        return torch.argmax(q).item()

    def update(self, batch_size):
        if len(self.memory) < batch_size: return
        is_per = isinstance(self.memory, PrioritizedRegimeReplayBuffer)
        if is_per:
            s, a, r, ns, d, per_indices, per_weights = self.memory.sample(batch_size)
            per_w = torch.FloatTensor(per_weights).to(self.device)
        else:
            s, a, r, ns, d = self.memory.sample(batch_size)
        s  = torch.FloatTensor(s).to(self.device)
        a  = torch.LongTensor(a).unsqueeze(1).to(self.device)
        r  = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d  = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        NQ = self.NUM_QUANTILES
        if hasattr(self.model, 'reset_noise'):
            self.model.reset_noise()
        q, tau_online = self.model(s, num_quantiles=NQ)
        q_a = q.gather(2, a.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)

        with torch.no_grad():
            self.model.eval()
            next_actions = self.model(ns, num_quantiles=NQ)[0].mean(dim=1).argmax(dim=1, keepdim=True)
            self.model.train()
            q_target, _  = self.target_model(ns, num_quantiles=NQ)
            q_target_a   = q_target.gather(2, next_actions.unsqueeze(1).expand(-1, NQ, -1)).squeeze(2)
            target = r + self.gamma * (1 - d) * q_target_a

        td_error  = target.unsqueeze(1) - q_a.unsqueeze(2)
        huber     = F.huber_loss(td_error, torch.zeros_like(td_error), reduction='none', delta=1.0)
        tau_exp   = tau_online
        indicator = (td_error.detach() < 0).float()
        loss_per_sample = (torch.abs(tau_exp - indicator) * huber).mean(dim=1).mean(dim=1)

        if is_per:
            loss = (loss_per_sample * per_w).mean()
            td_err_np = td_error.detach().abs().mean(dim=(1, 2)).cpu().numpy()
            self.memory.update_priorities(per_indices, td_err_np)
        else:
            loss = loss_per_sample.mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        for tp, p in zip(self.target_model.parameters(), self.model.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


# ═══════════════════════════════════════════════════════════════════════════
# 4. [Phase 3-G] GatingNet7 — Supervised Learning 전환
# ═══════════════════════════════════════════════════════════════════════════
class GatingNet7(nn.Module):
    """시장 상태 → [flat, bull, bear, chop_L, chop_S, norm_L, norm_S] (7-way)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 32), nn.SiLU(),
            nn.Linear(32, 7)
        )

    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits / 0.5, dim=-1)


class GatingPerformanceTracker:
    """[Phase 3-G] 에이전트별 레짐별 밸리데이션 PnL을 추적하여 소프트 라벨 생성.

    Supervised GatingNet 학습의 핵심:
      - 각 에피소드 종료 시 에이전트별 PnL을 레짐별로 기록
      - 최근 N 에피소드의 평균 PnL로 소프트 라벨 생성
      - GatingNet을 cross-entropy로 학습 (REINFORCE 대체)

    장점:
      - REINFORCE의 고분산 제거
      - 에이전트 학습 속도와 GatingNet의 비동기 문제 해결
      - 명시적인 "어떤 레짐에서 어떤 에이전트가 잘했는가" 학습
    """
    _REGIME_MAP = {
        'bull':         'regime_bull',
        'bear':         'regime_bear',
        'chop_long':    'regime_chop',
        'chop_short':   'regime_chop',
        'normal_long':  'regime_normal',
        'normal_short': 'regime_normal',
    }
    _AGENT_TO_GATE_IDX = {
        'flat': 0, 'bull': 1, 'bear': 2,
        'chop_long': 3, 'chop_short': 4,
        'normal_long': 5, 'normal_short': 6,
    }
    _REGIME_NAMES = ['chop', 'whipsaw', 'bull', 'bear', 'normal']

    def __init__(self, window: int = 20):
        self.window = window
        # {regime_name: {agent_name: deque([pnl1, pnl2, ...])}}
        self._records: dict = {
            r: {a: deque(maxlen=window)
                for a in ['flat', 'bull', 'bear', 'chop_long', 'chop_short', 'normal_long', 'normal_short']}
            for r in self._REGIME_NAMES
        }

    def record(self, agent_name: str, dominant_regime: str, pnl_pct: float):
        """에피소드 종료 시 호출: 해당 에이전트의 레짐별 PnL 기록"""
        if dominant_regime in self._records:
            self._records[dominant_regime][agent_name].append(pnl_pct)

    def get_soft_labels(self, regime_name: str, temperature: float = 2.0) -> np.ndarray:
        """특정 레짐에서의 에이전트별 평균 PnL → softmax 소프트 라벨 (7,)

        flat 에이전트의 PnL은 항상 0으로 가정 (기준선).
        temperature로 label smoothness 조절 (높을수록 uniform에 가까움).
        """
        scores = np.zeros(7, dtype=np.float32)
        for agent_name, gate_idx in self._AGENT_TO_GATE_IDX.items():
            records = self._records.get(regime_name, {}).get(agent_name, deque())
            if len(records) >= 3:
                scores[gate_idx] = float(np.mean(records))
            else:
                scores[gate_idx] = 0.0  # 데이터 부족 시 flat과 동일

        # softmax with temperature
        scores = scores / (temperature + 1e-8)
        exp_s = np.exp(scores - scores.max())
        return exp_s / (exp_s.sum() + 1e-8)

    def has_enough_data(self, min_records: int = 5) -> bool:
        """모든 레짐에서 최소 min_records개 이상의 기록이 있는지"""
        for r_name in self._REGIME_NAMES:
            total = sum(len(d) for d in self._records[r_name].values())
            if total < min_records:
                return False
        return True


def train_gating_supervised(gating_net, optimizer, tracker: GatingPerformanceTracker,
                            df_train, device, hmm_detector=None, mtf_features=None,
                            n_samples: int = 2048, temperature: float = 2.0):
    """[Phase 3-G] GatingNet Supervised Learning 학습 스텝.

    1. df_train에서 랜덤 state를 샘플링
    2. 해당 state의 레짐을 확인
    3. tracker에서 해당 레짐의 소프트 라벨 가져옴
    4. GatingNet 출력과 cross-entropy 계산
    """
    if not tracker.has_enough_data(min_records=3):
        return 0.0

    gating_net.train()

    env = TradingEnv(df_train, phase='val', agent_role='neutral', fee=0.0005,
                     hmm_detector=hmm_detector, mtf_features=mtf_features)

    states = []
    labels = []
    max_idx = len(df_train) - 2

    for _ in range(n_samples):
        idx = random.randint(0, max_idx)
        state = env._build_state(idx)
        stacked = np.concatenate([np.zeros(STATE_DIM, np.float32)] * (STACK_N - 1) + [state])
        states.append(stacked)

        # 해당 봉의 레짐 확인
        row = env._feat_np[idx]
        o = env._n_pred + env._n_conf + env._n_elite + env._n_alpha
        regime_vec = row[o:o+env._n_regime]
        regime_idx = int(np.argmax(regime_vec))
        regime_name = GatingPerformanceTracker._REGIME_NAMES[regime_idx]

        soft_label = tracker.get_soft_labels(regime_name, temperature=temperature)
        labels.append(soft_label)

    states_t = torch.FloatTensor(np.array(states)).to(device)
    labels_t = torch.FloatTensor(np.array(labels)).to(device)

    pred = gating_net(states_t)
    loss = -(labels_t * torch.log(pred + 1e-8)).sum(dim=-1).mean()

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(gating_net.parameters(), 1.0)
    optimizer.step()

    return loss.item()


# ═══════════════════════════════════════════════════════════════════════════
# 5. GatingRouter7 (추론용 — 변경 없음)
# ═══════════════════════════════════════════════════════════════════════════
class GatingRouter7:
    _W_IDX = {'flat': 0, 'bull': 1, 'bear': 2, 'chop_long': 3, 'chop_short': 4, 'normal_long': 5, 'normal_short': 6}
    _CVAR_THRESH = {'bull': 0.60, 'bear': 0.40, 'chop_long': 0.50, 'chop_short': 0.50, 'normal_long': 0.50, 'normal_short': 0.50}

    def __init__(self, models_dict, gating_net, device='cuda',
                 hmm_detector=None, kelly_sizer=None, mtf_features=None,
                 current_ep: int = 0):
        self.models     = {k: v.eval() for k, v in models_dict.items()}
        self.gating_net = gating_net.eval()
        self.device     = device
        self._active_agent = None
        self._frame_stack  = deque(maxlen=STACK_N)
        self.hmm   = hmm_detector
        self.kelly = kelly_sizer or KellyCriterionSizer()
        self.mtf   = mtf_features
        # [FIX] 학습 초기에는 q_z 임계값을 낮춰서 진입 허용, 점진 상승
        # ep=0→0.1, ep=200→0.3, ep=500+→0.5 (최종)
        self._q_z_threshold = float(np.clip(0.1 + current_ep * 0.002, 0.1, 0.5))

    def _state_tensor(self, features, pos):
        preds      = np.array([features.get(c, 0.) for c in STATE_PRED],   dtype=np.float32)
        confs      = np.array([features.get(c, 0.) for c in STATE_CONF],   dtype=np.float32)
        signal     = preds * confs
        elite      = np.array([features.get(c, 0.) for c in STATE_ELITE],  dtype=np.float32)
        alpha6     = np.array([features.get(c, 0.) for c in STATE_ALPHA],  dtype=np.float32)
        regime_raw = np.array([features.get(c, 0.) for c in REGIME_COLS],  dtype=np.float32)
        regime_idx = np.array([float(np.argmax(regime_raw))],               dtype=np.float32)
        synth2     = np.array([features.get(c, 0.) for c in STATE_SYNTH],  dtype=np.float32)
        cur_p = features.get('close', 1.0)
        pt    = pos.get('type')
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
        pad    = STACK_N - len(self._frame_stack)
        frames = [np.zeros(STATE_DIM, np.float32)] * pad + list(self._frame_stack)
        vec    = np.concatenate(frames)
        return torch.tensor(vec, dtype=torch.float32).unsqueeze(0).to(self.device)

    def _cvar_q(self, model, state, agent_name='normal_long'):
        threshold = self._CVAR_THRESH.get(agent_name, 0.50)
        nq = 32
        with torch.no_grad():
            q_quants, tau = model(state, num_quantiles=nq)
            sort_idx = tau[0, :, 0].argsort()
            k = max(4, int(nq * threshold))
            return q_quants[0][sort_idx][:k].mean(dim=0).cpu()

    def _full_quantiles(self, model, state, nq=32):
        with torch.no_grad():
            q_quants, tau = model(state, num_quantiles=nq)
            sort_idx = tau[0, :, 0].argsort()
            return q_quants[0][sort_idx].cpu()

    def decide(self, features, pos):
        cur_pos = pos.get('type')
        state   = self._state_tensor(features, pos)

        if self.hmm is not None:
            hmm_probs = self.hmm._alpha.copy()
            hmm_state = int(np.argmax(hmm_probs))
            hmm_names = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
            _hmm_info = {'hmm_state': hmm_names[hmm_state], 'hmm_probs': hmm_probs.round(3).tolist()}
        else:
            _hmm_info = {}

        if cur_pos is not None:
            agent_name = self._active_agent or ('normal_long' if cur_pos == 'LONG' else 'normal_short')
            q_cvar     = self._cvar_q(self.models[agent_name], state, agent_name=agent_name)
            best       = int(q_cvar.argmax().item())
            eval_best = best
            if agent_name in ['bear', 'chop_short', 'normal_short'] and best == 1:
                eval_best = 2
            wants_reverse = (cur_pos == 'LONG' and eval_best == 2) or (cur_pos == 'SHORT' and eval_best == 1)
            if best == 0 or wants_reverse:
                self._active_agent = None
                return 0, 0.0, {'agent': f'{agent_name}_exit', **_hmm_info}
            else:
                hold_action = 1 if cur_pos == 'LONG' else 2
                return hold_action, 0.0, {'agent': 'HOLD', **_hmm_info}

        with torch.no_grad():
            w = self.gating_net(state)[0].cpu()

        q_map = {}
        for name in ['bull', 'bear', 'chop_long', 'chop_short', 'normal_long', 'normal_short']:
            q_map[name] = self._cvar_q(self.models[name], state, agent_name=name)

        def _adv_n(q):
            best = q.argmax().item()
            if best == 0 or float(q[best]) <= 0: return 0.0
            return float(min(max(0., q[best] - q[0]), 0.1) * 10.0)

        scores = {
            'flat':         w[0].item(),
            'bull':         w[1].item() * (1 + _adv_n(q_map['bull'])),
            'bear':         w[2].item() * (1 + _adv_n(q_map['bear'])),
            'chop_long':    w[3].item() * (1 + _adv_n(q_map['chop_long'])),
            'chop_short':   w[4].item() * (1 + _adv_n(q_map['chop_short'])),
            'normal_long':  w[5].item() * (1 + _adv_n(q_map['normal_long'])),
            'normal_short': w[6].item() * (1 + _adv_n(q_map['normal_short'])),
        }

        long_edge  = scores['bull'] + scores.get('normal_long', 0.) * 0.5 - scores['flat']
        short_edge = scores['bear'] + scores.get('normal_short', 0.) * 0.5 - scores['flat']
        _edge_info = {'long_edge': long_edge, 'short_edge': short_edge}

        best_name = max(scores, key=scores.get)
        _w_arr = w.numpy()
        if best_name == 'flat':
            return 0, 0.0, {'agent': 'FLAT', **_edge_info, 'kelly': 0.0, 'gating_w': _w_arr, **_hmm_info}

        q_sel        = q_map[best_name]
        agent_action = int(q_sel.argmax().item())
        if agent_action == 0:
            return 0, 0.0, {'agent': f'{best_name}_declined', **_edge_info, 'kelly': 0.0, 'gating_w': _w_arr, **_hmm_info}

        q_vals = q_sel.float()
        q_std  = q_vals.std().item()
        q_adv  = float(q_vals[agent_action] - q_vals[0])
        q_z    = q_adv / (q_std + 1e-6)

        if q_z < self._q_z_threshold:
            return 0, 0.0, {'agent': f'{best_name}_low_conviction(z={q_z:.2f},thr={self._q_z_threshold:.2f})', **_edge_info, 'kelly': 0.0, 'gating_w': _w_arr, **_hmm_info}

        gating_conf = float(w[self._W_IDX[best_name]].item())
        q_full      = self._full_quantiles(self.models[best_name], state)
        lev         = self.kelly.compute(q_full, gating_confidence=gating_conf)
        kelly_stats = self.kelly.log_stats(q_full)

        val_action = agent_action
        if best_name in ['bear', 'chop_short', 'normal_short'] and agent_action == 1:
            val_action = 2

        self._active_agent = best_name
        return val_action, lev, {
            'agent': best_name, 'score': scores[best_name], 'kelly': lev,
            **kelly_stats, **_edge_info, 'gating_w': _w_arr, **_hmm_info,
        }


# ═══════════════════════════════════════════════════════════════════════════
# 6. 학습 루프 — Phase 1/2/3 전체 통합
# ═══════════════════════════════════════════════════════════════════════════
def train():
    CSV_PATH = 'data/rl_training_data_full.csv'
    if not os.path.exists(CSV_PATH):
        return logger.error("데이터가 없습니다. --mode generate_csv 실행 요망")

    df = pd.read_csv(CSV_PATH)
    split_idx = int(len(df) * 0.8)
    df_train  = df.iloc[:split_idx].reset_index(drop=True)
    df_val    = df.iloc[split_idx:].reset_index(drop=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    df_train_reg = df_train[REGIME_COLS].values.astype(np.float32)

    MAX_EP    = 4096
    _safe_end = len(df_train) - MAX_EP - 1

    agent_configs = {
        'bull':         {'action_dim': 2, 'target_regimes': ['regime_bull'],                   'cvar_threshold': 0.60},
        'bear':         {'action_dim': 2, 'target_regimes': ['regime_bear'],                   'cvar_threshold': 0.40},
        'chop_long':    {'action_dim': 2, 'target_regimes': ['regime_chop', 'regime_whipsaw'], 'cvar_threshold': 0.50},
        'chop_short':   {'action_dim': 2, 'target_regimes': ['regime_chop', 'regime_whipsaw'], 'cvar_threshold': 0.50},
        'normal_long':  {'action_dim': 2, 'target_regimes': ['regime_normal'],                 'cvar_threshold': 0.50},
        'normal_short': {'action_dim': 2, 'target_regimes': ['regime_normal'],                 'cvar_threshold': 0.50},
    }
    agent_names = list(agent_configs.keys())

    ri = {r: REGIME_COLS.index(f'regime_{r}') for r in ['bull', 'bear', 'chop', 'whipsaw', 'normal']}

    regime_starts = {
        'bull':         [i for i in range(_safe_end) if df_train_reg[i, ri['bull']] == 1.0],
        'bear':         [i for i in range(_safe_end) if df_train_reg[i, ri['bear']] == 1.0],
        'chop_long':    [i for i in range(_safe_end) if df_train_reg[i, ri['chop']] == 1.0 or df_train_reg[i, ri['whipsaw']] == 1.0],
        'chop_short':   [i for i in range(_safe_end) if df_train_reg[i, ri['chop']] == 1.0 or df_train_reg[i, ri['whipsaw']] == 1.0],
        'normal_long':  [i for i in range(_safe_end) if df_train_reg[i, ri['normal']] == 1.0],
        'normal_short': [i for i in range(_safe_end) if df_train_reg[i, ri['normal']] == 1.0],
    }
    for name in agent_names:
        if len(regime_starts[name]) < 100:
            regime_starts[name] = list(range(_safe_end))

    envs, models, agents = {}, {}, {}

    logger.info("[HMM] OnlineHMMDetector 초기 학습 시작...")
    hmm_detector = OnlineHMMDetector()
    hmm_detector.fit(df_train, n_iter=30)
    kelly_sizer  = KellyCriterionSizer(half_kelly=0.5, min_lev=0.1, max_lev=1.0,
                                        warmup_lev=0.5, warmup_until_ep=100)
    logger.info("[HMM] 초기 학습 완료. Kelly Sizer 초기화 완료 (워밍업: ep<100 → lev=0.5).")

    logger.info("[MTF] 훈련 데이터 멀티타임프레임 피처 선계산 중...")
    mtf_train = MultiTimeframeFeatures(df_train['close'].values.astype(np.float32))
    logger.info("[MTF] 검증 데이터 멀티타임프레임 피처 선계산 중...")
    mtf_val   = MultiTimeframeFeatures(df_val['close'].values.astype(np.float32))
    logger.info("[MTF] 선계산 완료.")

    for name, cfg in agent_configs.items():
        envs[name] = TradingEnv(df_train, phase='train', agent_role=name, fee=0.0005,
                                hmm_detector=hmm_detector, mtf_features=mtf_train)
        models[name] = RobustIQN(STACKED_STATE_DIM, cfg['action_dim'], raw_state_dim=STATE_DIM).to(device)
        agents[name] = IQNAgent(models[name], device=device, cvar_threshold=cfg['cvar_threshold'])
        agents[name].memory = PrioritizedRegimeReplayBuffer(
            200000, target_regimes=cfg['target_regimes'])

    NEP             = 1000
    BATCH           = 512
    UPDATE_FREQ     = 64
    MIN_BUFFER      = 2048
    global_step     = 0
    EPS_START       = 1.0
    EPS_END         = 0.01
    EPS_DECAY_STEPS = 400000

    os.makedirs('data/ensemble', exist_ok=True)
    best_val_pnl   = -float('inf')
    best_val_score = -float('inf')
    val_pnl_history: list = []
    start_ep       = 1
    os.makedirs('data/ensemble/ckpt', exist_ok=True)
    CHECKPOINT_PATH = 'data/ensemble/ckpt/rl_checkpoint.pth'

    gating_net       = GatingNet7(STACKED_STATE_DIM).to(device)
    gating_optimizer = torch.optim.Adam(gating_net.parameters(), lr=1e-3)

    # [Phase 3-G] Supervised GatingNet 성능 추적기
    gating_tracker = GatingPerformanceTracker(window=20)

    def _save_checkpoint(epoch):
        save_dict = {
            'global_step': global_step, 'best_val_pnl': best_val_pnl,
            'best_val_score': best_val_score, 'val_pnl_history': val_pnl_history, 'epoch': epoch,
            'gating_net': gating_net.state_dict(),
            'gating_opt': gating_optimizer.state_dict(),
        }
        for name in agent_names:
            save_dict[f'model_{name}'] = models[name].state_dict()
            save_dict[f'opt_{name}'] = agents[name].optimizer.state_dict()
        torch.save(save_dict, CHECKPOINT_PATH)

    if os.path.exists(CHECKPOINT_PATH):
        ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        arch_ok = True
        for name in agent_names:
            try:
                models[name].load_state_dict(ckpt[f'model_{name}'], strict=False)
                agents[name].target_model.load_state_dict(models[name].state_dict(), strict=False)
                agents[name].optimizer.load_state_dict(ckpt[f'opt_{name}'])
            except RuntimeError as e:
                logger.warning(f"⚠️ [{name}] 아키텍처 불일치로 가중치 스킵: {e}")
                arch_ok = False
        global_step, best_val_pnl = ckpt['global_step'], ckpt['best_val_pnl']
        best_val_score, val_pnl_history = ckpt['best_val_score'], ckpt.get('val_pnl_history', [])
        start_ep = ckpt['epoch'] + 1 if arch_ok else 1

        if 'gating_net' in ckpt:
            try:
                gating_net.load_state_dict(ckpt['gating_net'], strict=False)
                gating_optimizer.load_state_dict(ckpt['gating_opt'])
                logger.info("✅ GatingNet7 복원 완료")
            except RuntimeError:
                logger.warning("⚠️ GatingNet 아키텍처 불일치. 새로 초기화합니다.")
        else:
            logger.info("⚠️ 체크포인트에 gating_net 없음 → 새로 초기화")

        if arch_ok:
            logger.info(f"♻️ [복원] ep={ckpt['epoch']} → {start_ep} | best_pnl={best_val_pnl:.2f}%")
        else:
            logger.info(f"🆕 [아키텍처 변경] 가중치 초기화 후 ep=1 부터 재학습")

    try:
        for ep in range(start_ep, NEP + 1):
            # [Phase 1-C] Kelly 에폭 업데이트
            kelly_sizer.set_epoch(ep)

            # [Phase 1-A] 에이전트별 독립 start_idx
            def _sample_start(agent_name):
                pool = regime_starts[agent_name]
                if pool and random.random() < 0.7:
                    return random.choice(pool)
                return random.randint(0, _safe_end)

            env_states = {}
            ep_rewards = {name: 0.0 for name in agent_names}

            # [Phase 1-A] 각 에이전트가 독립적으로 시작점 샘플링
            for name in agent_names:
                independent_start = _sample_start(name)
                env_states[name] = envs[name].reset(independent_start)

            eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))
            done_flags = {name: False for name in agent_names}

            while not all(done_flags.values()):
                global_step += 1
                eps = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (global_step / EPS_DECAY_STEPS))

                for name in agent_names:
                    if done_flags[name]:
                        continue

                    env, agent, s = envs[name], agents[name], env_states[name]

                    if env.current_step >= env.end_step or env.current_step >= len(df_train) - 1:
                        done_flags[name] = True
                        continue

                    current_regimes = {r: float(df_train_reg[env.current_step, col_i]) for col_i, r in enumerate(REGIME_COLS)}

                    was_in_pos = env.pos is not None
                    a = agent.act(s, eps)
                    ns, r, d, _ = env.step(a)

                    in_pos = was_in_pos or (env.pos is not None)
                    agent.memory.push(s, a, r, ns, d, current_regimes, in_pos=in_pos)
                    ep_rewards[name] += r

                    env_states[name] = ns
                    if d:
                        done_flags[name] = True

                if global_step % UPDATE_FREQ == 0:
                    for name in agent_names:
                        if len(agents[name].memory) >= MIN_BUFFER:
                            agents[name].update(BATCH)

            # 에피소드 로깅 + [Phase 3-G] 에이전트 성능 기록
            _SIGMA_FLOOR = 0.05 / (64 ** 0.5)
            _REGIME_NAMES_LIST = ['chop', 'whipsaw', 'bull', 'bear', 'normal']
            for name in agent_names:
                env = envs[name]
                pnl = (env.balance / 10000 - 1) * 100
                noisy_layers = [m for m in models[name].modules() if isinstance(m, NoisyLinear)]
                current_floor = _SIGMA_FLOOR if pnl > -3.0 else _SIGMA_FLOOR * 5
                for nl in noisy_layers:
                    nl.weight_sigma.data.clamp_(min=current_floor)
                    nl.bias_sigma.data.clamp_(min=current_floor)
                avg_sigma = sum(m.weight_sigma.abs().mean().item() for m in noisy_layers) / max(1, len(noisy_layers))
                logger.info(
                    f"Ep {ep:04d} [{name:12s}] "
                    f"PnL:{pnl:6.1f}% Tr:{env.total_trades:4d} WR:{env.win_rate*100:4.0f}% "
                    f"Rew:{ep_rewards[name]:7.3f} | "
                    f"buf:{len(agents[name].memory):6d} | eps:{eps:.3f} | σ:{avg_sigma:.4f}"
                )

                # [Phase 3-G] 에이전트 레짐별 PnL 기록
                # 시작점의 dominant regime 기준
                start_step = env.start_step
                if start_step < len(df_train_reg):
                    dom_idx = int(np.argmax(df_train_reg[start_step]))
                    dom_regime = _REGIME_NAMES_LIST[dom_idx]
                    gating_tracker.record(name, dom_regime, pnl)

            # [Phase 3-G] GatingNet Supervised 학습 (ep%5, ep>=30)
            if ep % 5 == 0 and ep >= 30:
                g_loss = train_gating_supervised(
                    gating_net, gating_optimizer, gating_tracker,
                    df_train, device, hmm_detector=hmm_detector,
                    mtf_features=mtf_train, n_samples=2048, temperature=2.0)
                logger.info(f"    [GATING-SL] ep={ep} loss={g_loss:.4f}")

            # 밸리데이션 (ep%10)
            if ep % 10 == 0:
                router  = GatingRouter7(models, gating_net, device,
                                        hmm_detector=hmm_detector,
                                        kelly_sizer=kelly_sizer,
                                        mtf_features=mtf_val,
                                        current_ep=ep)
                val_env = TradingEnv(df_val, phase='val', agent_role='neutral', fee=0.0005,
                                     hmm_detector=hmm_detector, mtf_features=mtf_val)
                obs, d = val_env.reset(), False
                kelly_log = []

                while not d:
                    feat = df_val.iloc[val_env.current_step].to_dict()
                    feat['_step_idx'] = val_env.current_step
                    pos_info = {
                        'type': val_env.pos, 'entry_price': val_env.entry_price,
                        'unrealized': val_env.unrealized_pnl, 'mdd': val_env.max_drawdown,
                        'hold_norm': val_env.hold_count / 144
                    }
                    action, leverage_rate, info = router.decide(feat, pos_info)
                    if action in (1, 2) and 'win_rate' in info:
                        kelly_log.append({'lev': leverage_rate, 'wr': info.get('win_rate', 0),
                                          'payoff': info.get('payoff', 0), 'f_star': info.get('f_star', 0)})
                    obs, _, d, _ = val_env.step(action, leverage_rate=leverage_rate)

                val_pnl_pct = (val_env.balance / 10000 - 1) * 100
                val_pnl_history.append(val_pnl_pct)
                if len(val_pnl_history) >= 3:
                    _ph = val_pnl_history[-10:]
                    _std = max(float(np.std(_ph)), 0.1)
                    sharpe_est = float(np.clip(np.mean(_ph) / _std, -10.0, 10.0))
                else:
                    sharpe_est = 0.0

                if val_pnl_pct > 0:
                    trade_activity = min(val_env.total_trades / 30.0, 1.0) * 5.0
                else:
                    trade_activity = -min(val_env.total_trades / 30.0, 1.0) * 10.0

                val_score = (val_pnl_pct * 5.0) + (val_env.win_rate * 20.0) + (sharpe_est * 5.0) + trade_activity

                logger.info(f"    [VAL] PnL:{val_pnl_pct:.2f}% | Tr:{val_env.total_trades} | WR:{val_env.win_rate*100:.0f}% | Score:{val_score:.2f}")

                hmm_s = int(np.argmax(hmm_detector._alpha))
                hmm_names = ['bull-trend', 'bear-trend', 'hv-chop', 'lv-range']
                logger.info(f"    [HMM]  state={hmm_names[hmm_s]} | probs={hmm_detector._alpha.round(3).tolist()}")

                _last_mtf = mtf_val.get(len(df_val) - 1)
                logger.info(f"    [MTF]  4h_ret:{_last_mtf[0]:.3f} 4h_trend:{_last_mtf[1]:.3f} align:{_last_mtf[2]:.0f}")

                if kelly_log:
                    avg_lev = np.mean([k['lev'] for k in kelly_log])
                    avg_wr  = np.mean([k['wr']  for k in kelly_log])
                    logger.info(f"    [KELLY] n={len(kelly_log)} | avg_lev:{avg_lev:.3f} | wr:{avg_wr:.3f}")

                if val_score > best_val_score:
                    best_val_score, best_val_pnl = val_score, val_pnl_pct
                    save_dict = {'best_pnl': best_val_pnl, 'epoch': ep,
                                 'gating_net': gating_net.state_dict()}
                    for name in agent_names: save_dict[f'model_{name}'] = models[name].state_dict()
                    torch.save(save_dict, 'data/ensemble/ckpt/best_rl_agents.pth')
                    logger.info(f"    🎉 [NEW BEST] 저장 완료 (PnL:{best_val_pnl:.2f}%)")

                if ep % 50 == 0:
                    hmm_detector.update_online(n_iter=5)
                    logger.info("    [HMM]  온라인 업데이트 완료")

                _save_checkpoint(ep)

    except KeyboardInterrupt:
        logger.info("⚠️ 학습 중단. 체크포인트 저장 완료.")
        _save_checkpoint(ep)

if __name__ == "__main__":
    train()