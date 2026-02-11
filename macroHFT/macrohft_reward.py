"""MacroHFT Reward v8.2 - Pure PnL Driven, Minimal Heuristics"""

import numpy as np
import torch
import torch.nn as nn
from common import config

# ----------------------------------------------------------------------
# LambdaMetaLearner (동일)
# ----------------------------------------------------------------------
class LambdaMetaLearner(nn.Module):
    def __init__(self, num_experts=3, init_lambda=2.25):
        super().__init__()
        self.log_lambdas = nn.Parameter(torch.full((num_experts,), np.log(init_lambda)))
    def forward(self, expert_idx):
        return torch.exp(self.log_lambdas[expert_idx])
    def get_all_lambdas(self):
        return torch.exp(self.log_lambdas).detach().cpu().numpy()

# ----------------------------------------------------------------------
# AdaptiveTracker (단순화)
# ----------------------------------------------------------------------
class AdaptiveTracker:
    def __init__(self):
        self.reset()
    def reset(self):
        self.episode_pnl = 0.0
        self.peak_pnl = 0.0
        self.drawdown = 0.0
        self.returns = []
    def update(self, step_pnl):
        self.episode_pnl += step_pnl
        self.returns.append(step_pnl)
        if self.episode_pnl > self.peak_pnl:
            self.peak_pnl = self.episode_pnl
        self.drawdown = max(0.0, self.peak_pnl - self.episode_pnl)
    def get_sharpe_ratio(self):
        if len(self.returns) < 5: return 0.0
        std = np.std(self.returns)
        if std < 1e-9: return 0.0
        return np.mean(self.returns) / std

_tracker = AdaptiveTracker()
def reset_reward_tracker(): _tracker.reset()

# ----------------------------------------------------------------------
# Identity Bonus (PnL 비례, 고정값 제거)
# ----------------------------------------------------------------------
def compute_identity_bonus(expert_type, step_pnl, chop_index, volatility_z):
    """고정 보너스 제거, PnL에만 비례"""
    if step_pnl <= 0:
        return 0.0
    if expert_type == 'trend' and chop_index < 40.0:
        return step_pnl * 30.0  # 1% 수익 → +0.3
    elif expert_type == 'volatility' and volatility_z > 1.5:
        return step_pnl * 30.0
    elif expert_type == 'sideways' and chop_index > 55.0 and volatility_z < 0.5:
        return step_pnl * 30.0
    return 0.0

# ----------------------------------------------------------------------
# 메인 보상 함수 (극단적 단순화)
# ----------------------------------------------------------------------
def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, expert_idx=2,
                         chop_index=50.0, volatility_z=0.0, lambda_meta=None):
    
    global _tracker
    _tracker.update(step_pnl)

    expert_map = {0: 'trend', 1: 'volatility', 2: 'sideways'}
    expert_type = expert_map.get(expert_idx, 'sideways')

    # ------------------------------------------------------------------
    # [핵심] PnL 기반 보상 (스케일 통일, 휴리스틱 제거)
    # ------------------------------------------------------------------
    LOSS_AVERSION = lambda_meta if lambda_meta is not None else 2.25
    
    # Step PnL (BASE_SCALE = 100.0, 이전보다 높임)
    if step_pnl >= 0:
        reward = step_pnl * 100.0
    else:
        reward = step_pnl * 100.0 * LOSS_AVERSION
    
    # Realized PnL (step PnL의 2배 가중치)
    if trade_done:
        if realized_pnl >= 0:
            reward += realized_pnl * 200.0
        else:
            reward += realized_pnl * 200.0 * LOSS_AVERSION

    # ------------------------------------------------------------------
    # [최소한의 전문가 정체성] PnL 비례 보너스만 (고정값 0)
    # ------------------------------------------------------------------
    reward += compute_identity_bonus(expert_type, step_pnl, chop_index, volatility_z)

    # ------------------------------------------------------------------
    # [Volatility] 진입세 완전 제거, 손실 패널티만 유지 (절반으로 축소)
    # ------------------------------------------------------------------
    if expert_type == 'volatility':
        if step_pnl < 0:
            reward += step_pnl * 30.0  # 50→30, 20→30 (PnL 비례로 통일)
        # ※ 진입세(action penalty) 제거
        # ※ 푼돈 익절 감점 제거

    # ------------------------------------------------------------------
    # [Sideways] 시간 감점 유지 (미미)
    # ------------------------------------------------------------------
    if expert_type == 'sideways' and holding_time > 0.15:
        reward -= 0.1  # 그대로

    # ------------------------------------------------------------------
    # [Trend] 수익 가속도 (이미 step_pnl에 포함됨, 중복 제거)
    # ------------------------------------------------------------------
    # 삭제: 별도 보너스 제거 (identity bonus로 대체)

    # ------------------------------------------------------------------
    # [Adaptive Risk] 수익권에서만 작동, 임계값 상향
    # ------------------------------------------------------------------
    if _tracker.episode_pnl > 0.02:  # 2% 이상 수익 시
        if _tracker.drawdown > 0.03:
            reward -= (_tracker.drawdown - 0.03) * 30.0
        sharpe = _tracker.get_sharpe_ratio()
        if sharpe > 0.3:
            reward += sharpe * 2.0

    # ------------------------------------------------------------------
    # Soft Clip (강력하게)
    # ------------------------------------------------------------------
    return float(np.tanh(reward / 5.0) * 5.0)  # -5 ~ +5 범위