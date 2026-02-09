"""
TD3 전략가(Strategic) 전용 리워드 함수 - Simplified for Phase 1
목표: "Oracle을 따라 방향을 맞추고, PnL을 극대화하라."
"""
import numpy as np

def calculate_td3_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
    """
    [TD3 Reward - Simplified Fix]
    - 이중 레버리지 제거: step_pnl은 이미 (direction * return)
    - 복잡한 페널티 제거하고 순수 PnL 비례 보상
    
    Args:
        step_pnl: 이미 계산된 PnL (소수점 단위, 예: 0.0005 = 0.05%)
        
    Returns:
        reward: -10.0 ~ 10.0 범위의 보상
    """
    # 1. 이중 레버리지 버그 수정
    # 기존: step_pnl * effective_leverage (400배 증폭됨)
    # 수정: step_pnl 자체를 점수로 환산 (0.01% -> 1점)
    
    # step_pnl은 소수점 단위 (예: 0.0005)이므로 100~1000을 곱해 가시적인 점수로 만듦
    reward = step_pnl * 100.0
    
    # 2. 클리핑: 학습 안정성을 위해 (-10, 10) 범위로 제한
    return float(np.clip(reward, -10.0, 10.0))
