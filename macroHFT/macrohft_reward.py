"""
MacroHFT Reward v4 - 2026 SOTA Research-Aligned
===============================================
Core Philosophy:
1. Prospect Theory (Kahneman & Tversky): Loss Aversion (λ=2.25)
2. Differential Objectives: Trend(LogRet), Vol(Sharpe), Side(MDD)
3. Risk-Aware Layer: Downside Deviation + MDD Penalty
4. Soft Clipping: Tanh based gradient preservation
"""
import numpy as np
import math
from common import config

# ==============================================================================
# Reward Tracker (Global State for Session)
# ==============================================================================
# 주의: 멀티 프로세싱 환경에서는 별도 관리가 필요하나, 현재 단일 프로세스 학습 가정
class RewardTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.returns_history = []
        self.peak_pnl = 0.0
        self.current_drawdown = 0.0
        self.running_mean = 0.0
        self.running_m2 = 0.0 # For variance
        self.count = 0

    def update(self, step_pnl, unrealized_pnl):
        self.returns_history.append(step_pnl)
        
        # Welford's Online Algorithm for Variance
        self.count += 1
        delta = step_pnl - self.running_mean
        self.running_mean += delta / self.count
        delta2 = step_pnl - self.running_mean
        self.running_m2 += delta * delta2
        
        # MDD Tracking
        if unrealized_pnl > self.peak_pnl:
            self.peak_pnl = unrealized_pnl
        drawdown = self.peak_pnl - unrealized_pnl
        self.current_drawdown = max(0.0, drawdown)

    def get_volatility(self):
        if self.count < 2: return 0.0
        return math.sqrt(self.running_m2 / (self.count - 1))

    def get_downside_deviation(self):
        if self.count < 2: return 0.0
        neg_returns = [r for r in self.returns_history if r < 0]
        if not neg_returns: return 0.0
        return np.std(neg_returns)

# 전역 트래커 인스턴스
_tracker = RewardTracker()

def reset_reward_tracker():
    """에피소드 시작 시 호출"""
    global _tracker
    _tracker.reset()

# ==============================================================================
# Main Entry Point
# ==============================================================================

def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, expert_idx=2):
    """
    v4 Entry Point: Updates tracker and delegates to specialized logic.
    """
    global _tracker
    
    # Update Tracker Stats
    current_pnl = realized_pnl if trade_done else step_pnl
    # (Unrealized PnL 추정: step_pnl 누적 혹은 외부에서 주입받아야 하나, 여기선 약식으로 처리)
    # 정확한 MDD 계산을 위해선 누적 PnL이 필요하지만, 여기선 step_pnl 흐름으로 근사
    _tracker.update(step_pnl, current_pnl) # unrealized is approximated by current flow for local mdd

    # Expert Mapping
    expert_map = {0: 'trend', 1: 'volatility', 2: 'sideways'}
    expert_type = expert_map.get(expert_idx, 'sideways')

    # Calculate Raw Reward
    raw_reward = calculate_v4_reward(
        tracker=_tracker,
        expert_type=expert_type,
        step_pnl=step_pnl,
        realized_pnl=realized_pnl,
        trade_done=trade_done,
        holding_time=holding_time,
        action=action,
        current_position=current_position
    )
    
    # [Layer 4] Soft Clipping (Tanh)
    # Gradient를 죽이지 않으면서 극단값 제어
    clip_scale = getattr(config, 'REWARD_CLIP_SCALE', 10.0)
    final_reward = float(np.tanh(raw_reward / clip_scale) * clip_scale)
    
    return final_reward


# ==============================================================================
# v4 Core Logic (4-Layer Architecture)
# ==============================================================================

def calculate_v4_reward(tracker, expert_type, step_pnl, realized_pnl, trade_done, 
                        holding_time, action, current_position):
    
    reward = 0.0
    
    # Config Load
    LOSS_AVERSION = getattr(config, 'REWARD_LOSS_AVERSION', 2.25)
    BASE_MULT = getattr(config, 'REWARD_BASE_MULT', 50.0)
    
    # -------------------------------------------------------------------------
    # [Layer 1] Kahneman-Tversky Asymmetry (Prospect Theory)
    # 수익은 기쁘지만, 손실은 2.25배 더 아프다.
    # -------------------------------------------------------------------------
    pnl_to_eval = realized_pnl if trade_done else step_pnl
    
    if pnl_to_eval >= 0:
        base_component = pnl_to_eval * BASE_MULT
    else:
        base_component = pnl_to_eval * BASE_MULT * LOSS_AVERSION
        
    reward += base_component

    # -------------------------------------------------------------------------
    # [Layer 2] Expert-Specific Objectives
    # -------------------------------------------------------------------------
    
    if expert_type == 'trend':
        # Objective: Log Returns (Compounding) & Consistency
        # 추세는 '복리'로 불어나는 것이 목표 -> 로그 수익률 근사 보상
        if current_position is not None and step_pnl > 0:
            # ln(1+r) ≈ r - r^2/2 (Taylor Series 2nd order)
            log_ret = step_pnl - (step_pnl**2)/2
            reward += log_ret * getattr(config, 'REWARD_TREND_LOG_RETURN_SCALE', 100.0)
            
            # Holding Bonus (Time-Weighted)
            reward += 0.05 * (1 + holding_time)

    elif expert_type == 'volatility':
        # [Fix] "묻지마 변동성 보너스" 완전 삭제
        # 이제는 오직 '실현 손익(Realized PnL)'으로만 평가합니다.
        
        # 1. Trading Tax (매매 횟수 억제)
        # 매수(1)나 매도(2) 행동을 할 때마다 점수를 깎습니다.
        # "확실한 거 아니면 들어가지 마라"는 신호입니다.
        if action in [1, 2]:
            reward -= 2.0 

        # 2. Realized Profit Bonus (익절 시에만 보상)
        if trade_done and realized_pnl > 0:
            vol = tracker.get_volatility()
            if vol > 1e-6:
                sharpe = realized_pnl / vol
                reward += min(sharpe, 3.0) * 2.0 # Sharpe Bonus
            reward += realized_pnl * 50.0 # Profit Bonus

        # 3. Strict Loss Penalty
        # 평가 손실이 나면 가차없이 때립니다. (Momentum Bonus 삭제됨)
        if step_pnl < 0:
            reward += step_pnl * 100.0
            
    elif expert_type == 'sideways':
        # Objective: Max Drawdown Minimization & Mean Reversion
        # MDD가 커지면 페널티를 강력하게 부여
        mdd_threshold = getattr(config, 'REWARD_SIDEWAYS_MDD_THRESHOLD', 0.02)
        if tracker.current_drawdown > mdd_threshold:
            penalty = (tracker.current_drawdown - mdd_threshold) * getattr(config, 'REWARD_MDD_PENALTY_COEF', 20.0)
            reward -= penalty
            
        # Time Decay (오래 물려있으면 감점)
        decay_start = getattr(config, 'REWARD_SIDEWAYS_DECAY_START', 30)
        # holding_time is normalized (0~1), need steps approximation
        # Assuming max_steps=480, holding_time*480 = steps
        steps_held = holding_time * 480
        if steps_held > decay_start:
            reward -= 0.01 * ((steps_held - decay_start) / 10.0)

    # -------------------------------------------------------------------------
    # [Layer 3] Risk Penalty (Common)
    # -------------------------------------------------------------------------
    # Downside Deviation Penalty (하방 리스크 제어)
    downside_dev = tracker.get_downside_deviation()
    if downside_dev > 0.005: # 0.5% 이상의 하방 변동성
        reward -= downside_dev * getattr(config, 'REWARD_DOWNSIDE_PENALTY', 0.5) * 100.0

    return reward