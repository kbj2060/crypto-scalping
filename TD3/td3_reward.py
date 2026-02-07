"""
TD3 전략가(Strategic) 전용 리워드 함수
목표: "Big Trend를 먹어라. 변동성을 견디고 거대한 추세를 포착하라."
"""
import numpy as np

def calculate_td3_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, effective_leverage=1.0):
    """
    [TD3 Strategic Reward - Stabilized Ver.]
    - 스케일을 1/10로 축소하여 학습 안정성 확보
    - 변동성 페널티를 강화하여 '안정적 우상향' 유도
    - 손실 회피 성향 주입 (Sortino Ratio 개념)
    """
    reward = 0.0
    
    # action은 레버리지 강도 (연속 값)
    # action이 텐서나 배열일 경우 스칼라로 변환
    if hasattr(action, 'item'):
        action = action.item()
    leverage_intensity = abs(float(action)) if action is not None else 1.0
    
    # 1. 평가 손익 (Unrealized PnL) - 스케일 대폭 축소 (100 → 10)
    # ROE 관점: step_pnl * 레버리지
    roe = step_pnl * effective_leverage
    
    if roe > 0:
        # 수익일 때: 적절한 보상
        reward += roe * 10.0 
    else:
        # 손실일 때: 손실 회피 성향 주입 (2배 페널티)
        # Sortino Ratio처럼 하방 변동성을 더 민감하게 처벌
        reward += roe * 20.0 

    # 2. 실현 손익 (Realized PnL) - 스케일 축소 (200 → 20)
    if trade_done:
        if realized_pnl > 0:
            reward += realized_pnl * 20.0
        else:
            # 손절매는 더 아프게 (30배 페널티)
            reward += realized_pnl * 30.0

        # 잦은 매매 방지 (수수료/슬리피지 비용 현실화)
        reward -= 0.1 

    # 3. 레버리지 과용 페널티 (Risk Control)
    # 손실 구간에서 고레버리지를 쓰고 있다면 추가 감점
    if step_pnl < 0 and leverage_intensity > 0.8:
        reward -= 0.05 * leverage_intensity

    # 4. 청산(Liquidation) 방어 보너스 (생존 보상)
    # 포지션을 잡고 있는데 청산당하지 않고 살아남았다면 미세한 보상
    if current_position is not None and current_position != 'HOLD':
        reward += 0.001

    # 5. 클리핑 범위 확장 (-10 → -20)
    # 스케일을 줄였으므로, 클리핑 범위를 조금 넓혀서 극단적 상황은 정보로 받아들임
    return float(np.clip(reward, -20.0, 20.0))
