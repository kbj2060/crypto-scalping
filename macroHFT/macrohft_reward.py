"""
MacroHFT 전술가(Tactical) 전용 리워드 함수
목표: "타이밍을 뺏어라. 정확한 진입과 청산(Timing)으로 수익을 쌓아라."
"""
import numpy as np

def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
    """
    [MacroHFT Tactical Reward - Soft Penalty Ver.]
    - 진입 페널티를 대폭 완화하여 '시도'를 장려함
    - 대신, 손실을 보았을 때의 타격을 유지하여 '신중함'을 가르침
    - 실현 손익(Realized PnL)에 대한 보상을 더 신뢰
    """
    reward = 0.0
    
    # 1. 평가 손익 (Unrealized PnL)
    # 전술가는 당장의 평가 이익이 중요함. 정직하게 반영.
    reward += step_pnl * 50.0 

    # 2. 실현 손익 (Realized PnL) - 핵심
    if trade_done:
        # 이익은 확실하게 보상, 손실은 확실하게 처벌
        reward += realized_pnl * 100.0 
        
        # [수정] 수수료 페널티 대폭 완화 (0.5 -> 0.05)
        # 이제 쫄지 말고 진입해라. 단, 뇌동매매는 여전히 손해다.
        reward -= 0.05 

    # 3. 시간 비용 (Time Decay)
    # 오래 들고 있으면 기회비용 발생 (약한 압박)
    if current_position is not None and current_position != 'HOLD':
        if step_pnl <= 0:  # 수익이 안 나는데 버티면
            reward -= 0.01 
    
    # 4. 빠른 익절 보너스는 제거 (자연스러운 학습 유도)
    # 기존: 빠른 익절 시 +0.3 보너스 -> 제거
    # 이유: 보상 설계를 간소화하고 realized_pnl에만 집중

    return float(np.clip(reward, -10.0, 10.0))


# ==============================================================================
# 전문가별 특화 리워드 함수 (Expert Specialization)
# ==============================================================================

def calculate_specialized_reward(expert_type, step_pnl, realized_pnl, trade_done, 
                                 holding_time=0, effective_leverage=1.0, current_position=None):
    """
    전문가별 특화된 리워드 계산
    
    Args:
        expert_type: 'trend', 'volatility', 'sideways' 중 하나
        step_pnl: 스텝별 미실현 손익
        realized_pnl: 실현 손익
        trade_done: 거래 완료 여부
        holding_time: 보유 시간 (스텝 수)
        effective_leverage: 유효 레버리지
        current_position: 현재 포지션 ('LONG', 'SHORT', 'HOLD' 또는 None)
    """
    reward = 0.0

    if expert_type == 'trend':
        # [Trend Expert] 추세 지속성에 보상 (Long-run Profit)
        reward += step_pnl * 100.0
        
        # 수익 중일 때 보유 시간에 비례한 보너스 (추세 유지 장려)
        if step_pnl > 0:
            reward += (holding_time * 0.01)
        
        # 실현 손익 강조 (큰 수익 선호)
        if trade_done:
            reward += realized_pnl * 150.0

    elif expert_type == 'volatility':
        # [Volatility Expert] 빠른 익절과 강한 모멘텀 포착 (Hit & Run)
        reward += step_pnl * 50.0
        
        # 실현 손익 극대화 (빠른 수익 실현 장려)
        if trade_done:
            reward += realized_pnl * 200.0
            # Taker 수수료 페널티 반영
            reward -= 0.05

    elif expert_type == 'sideways':
        # [Sideways Expert] 박스권 스캘핑 특화
        
        if not trade_done:
            # 관망 보상 강화 (뇌동매매 억제)
            reward += 0.01
        else:
            # [수정 후] 이익이 났을 때만 리베이트 보너스 지급!
            reward += realized_pnl * 500.0
            
            if realized_pnl > 0:
                reward += 0.05  # 수익 시 강력한 보너스 (리베이트+알파)
            else:
                reward -= 0.02  # 손실 시에는 수수료 페널티 추가 (확인사살)

            # 미세 수익 장려 (박스권 상하단 타겟팅)
            if 0 < realized_pnl < 0.001:
                reward += 0.1

    return float(np.clip(reward, -10.0, 10.0))
