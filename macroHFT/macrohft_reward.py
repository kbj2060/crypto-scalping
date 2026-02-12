"""MacroHFT Reward v8.3 - PnL Absolute, Minimal Heuristics"""

import numpy as np
import torch
import torch.nn as nn
from common import config

class LambdaMetaLearner(nn.Module):
    def __init__(self, num_experts=3, init_lambda=2.25):
        super().__init__()
        self.log_lambdas = nn.Parameter(torch.full((num_experts,), np.log(init_lambda)))
    def forward(self, expert_idx):
        return torch.exp(self.log_lambdas[expert_idx])

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

def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, expert_idx=2,
                         chop_index=50.0, volatility_z=0.0, lambda_meta=None):
    
    global _tracker
    _tracker.update(step_pnl)

    expert_map = {0: 'trend', 1: 'volatility', 2: 'sideways'}
    expert_type = expert_map.get(expert_idx, 'sideways')
    LOSS_AVERSION = lambda_meta if lambda_meta is not None else 2.25

    reward = 0.0

    # --------------------------------------------------------------
    # [1] STEP PnL - 미실현 손익 변화 (10x)
    # --------------------------------------------------------------
    if step_pnl >= 0:
        reward += step_pnl * 10.0
    else:
        reward += step_pnl * 10.0 * LOSS_AVERSION

    # --------------------------------------------------------------
    # [2] REALIZED PnL - 청산 손익 (200x, 고정 보너스 90% ↓)
    # --------------------------------------------------------------
    if trade_done:
        if realized_pnl >= 0:
            reward += realized_pnl * 200.0          # 300 → 200
            reward += 0.1                          # 1.0 → 0.1
        else:
            reward += realized_pnl * 200.0 * LOSS_AVERSION
            reward -= 0.05                        # 0.5 → 0.05

    # --------------------------------------------------------------
    # [3] SIDEWAYS TIME DECAY - 유지 (미미)
    # --------------------------------------------------------------
    if expert_type == 'sideways' and holding_time > 0.15:
        reward -= 0.1

    # --------------------------------------------------------------
    # [4] ADAPTIVE RISK - 수익권에서만
    # --------------------------------------------------------------
    if _tracker.episode_pnl > 0.01:
        if _tracker.drawdown > 0.02:
            reward -= (_tracker.drawdown - 0.02) * 20.0
        sharpe = _tracker.get_sharpe_ratio()
        if sharpe > 0.2:
            reward += sharpe * 2.0

    # --------------------------------------------------------------
    # [5] SOFT CLIP - 그대로
    # --------------------------------------------------------------
    return reward 