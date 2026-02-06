"""
MacroHFT 전술가(Tactical) 전용 리워드 함수
목표: "타이밍을 뺏어라. 정확한 진입과 청산(Timing)으로 수익을 쌓아라."
"""
import numpy as np

def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
    """
    [MacroHFT Tactical Reward]
    - '금융 치료' 로직 적용 (수익만이 살길)
    - 잦은 매매(Churning) 강력 억제
    - 짧은 보유 시간 선호 (스캘핑/데이 트레이딩 성향)
    """
    reward = 0.0
    
    # 1. 평가 손익 (Unrealized PnL)
    # 전술가는 당장의 평가 이익이 중요함. 정직하게 반영.
    reward += step_pnl * 50.0 

    # 2. 실현 손익 (Realized PnL) - 핵심
    if trade_done:
        # 이익은 확실하게 보상, 손실은 확실하게 처벌
        reward += realized_pnl * 100.0 
        
        # [MacroHFT 특화] 수수료 페널티 (Churning 방지)
        # 매매 횟수를 줄이고 승률을 높여야 함
        reward -= 0.5 

    # 3. 시간 감가 (Time Decay)
    # 전술가는 포지션을 오래 끌면 불리함 (자금 회전율 저하)
    if current_position is not None and current_position != 'HOLD':
        if step_pnl > 0:
            reward += 0.01  # 수익 중이면 버텨도 됨 (소폭 보너스)
        else:
            reward -= 0.05  # 손실 중인데 버티면 페널티 (빠른 손절 유도)
    
    # 4. 빠른 익절 보너스 (MacroHFT 특화)
    # 짧은 시간에 수익을 내면 추가 보상
    if trade_done and realized_pnl > 0 and holding_time < 20:  # 1시간(20틱) 이내 익절
        reward += 0.3  # "빠르고 정확하게 먹었다"
    
    # 5. 방향성 일관성 보너스
    # 같은 방향으로 계속 수익을 내면 트렌드를 잘 타는 것
    # (구현 생략, 필요 시 self.recent_trades 등으로 추적 가능)

    return float(np.clip(reward, -10.0, 10.0))
