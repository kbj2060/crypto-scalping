"""
TD3 전략가(Strategic) 전용 리워드 함수
목표: "변동성을 견디고, 레버리지를 적절히 사용하여 거대한 추세(Big Trend)를 먹는 것."
"""
import numpy as np

def calculate_td3_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
    """
    [TD3 Strategic Reward]
    - 레버리지(action 크기)에 비례한 보상/벌칙
    - 잔파도(Noise)는 무시하고 추세 수익(Trend)에 집중
    - Sortino Ratio 개념 도입 (하방 변동성만 처벌)
    """
    reward = 0.0
    
    # action은 TD3의 출력 (-1 ~ 1), 즉 레버리지 강도
    leverage_intensity = abs(action) if action is not None else 1.0
    
    # 1. 평가 손익 (Unrealized PnL)
    # 전략가는 평가 이익이 커지는 과정을 견뎌야 함.
    # 이익일 때는 레버리지 가중치를 둬서 "잘 질렀다"고 칭찬.
    if step_pnl > 0:
        reward += step_pnl * 100.0 * (1.0 + leverage_intensity)
    else:
        # 손실일 때는 "왜 크게 질렀냐"고 더 크게 혼냄 (위험 회피)
        reward += step_pnl * 100.0 * (1.0 + leverage_intensity * 1.5)

    # 2. 실현 손익 (Realized PnL)
    if trade_done and realized_pnl != 0.0:
        if realized_pnl > 0:
            reward += realized_pnl * 200.0  # 익절은 언제나 옳다
        else:
            reward += realized_pnl * 200.0  # 손절
            
        # [TD3 특화] 짧게 치고 빠지면(Scalping) 페널티
        # 전략가는 진득하게 추세를 먹어야 함 (최소 12스텝=36분)
        if holding_time < 12: 
            reward -= 1.0 

    # 3. 드로우다운 방어
    # 큰 손실 상태로 오래 버티면 가중 처벌 (손절 유도)
    if current_position is not None and current_position != 0 and step_pnl < -0.01:  # -1% 이상 손실 시
        reward -= 0.1 * (holding_time / 10.0)
    
    # 4. 레버리지 효율성 보너스
    # 높은 레버리지로 큰 수익을 냈다면 추가 보상
    if step_pnl > 0.02 and effective_leverage > 10:  # ROE +2% 이상, 10배+ 레버리지
        reward += 0.5  # "고레버리지를 잘 썼다"

    # Reward Clipping (안정성)
    return float(np.clip(reward, -20.0, 20.0))
