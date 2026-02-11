"""
MacroHFT Reward v6.5 - Adaptive Profit First
============================================
목표: 데이터 커리큘럼 없이 보상 함수 자체적으로 난이도를 조절.
1. [생존 모드]: 누적 수익 < 0 이면 오직 PnL 회복에 집중 (Simple v5 스타일)
2. [프로 모드]: 누적 수익 > 0 이면 MDD/Sharpe 관리 (SOTA v4 스타일)
3. [Volatility Fix]: 뇌동매매 원천 봉쇄 (진입 비용 부과, 손실 시 150배 페널티)
"""
import numpy as np
import math
from common import config

# ==============================================================================
# Lightweight Tracker (에피소드 내부 상태 추적용)
# ==============================================================================
class AdaptiveTracker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_pnl = 0.0      # 누적 수익률 (가장 중요)
        self.peak_pnl = 0.0         # MDD 계산용
        self.drawdown = 0.0
        self.returns = []           # Sharpe 계산용

    def update(self, step_pnl):
        self.episode_pnl += step_pnl
        self.returns.append(step_pnl)
        
        # MDD 갱신
        if self.episode_pnl > self.peak_pnl:
            self.peak_pnl = self.episode_pnl
        self.drawdown = max(0.0, self.peak_pnl - self.episode_pnl)

    def get_sharpe_ratio(self):
        if len(self.returns) < 5: return 0.0
        std = np.std(self.returns)
        if std < 1e-9: return 0.0
        return np.mean(self.returns) / std

# 전역 트래커
_tracker = AdaptiveTracker()

def reset_reward_tracker():
    global _tracker
    _tracker.reset()

# ==============================================================================
# Main Reward Logic
# ==============================================================================
def calculate_ppo_reward(self, step_pnl, realized_pnl, trade_done,
                         holding_time=0, action=0, prev_position=None,
                         current_position=None, expert_idx=2):
    global _tracker
    
    # 1. 트래커 업데이트
    _tracker.update(step_pnl)

    # Expert 매핑
    expert_map = {0: 'trend', 1: 'volatility', 2: 'sideways'}
    expert_type = expert_map.get(expert_idx, 'sideways')

    # -------------------------------------------------------------------------
    # [Layer 1] Base PnL Reward (Fundamental)
    # -------------------------------------------------------------------------
    # 기본급: 수익나면 +100배, 손실나면 -100배 (단순명료)
    reward = step_pnl * 100.0
    
    if trade_done:
        # 익절/손절 확정 시 보너스/페널티 2배
        reward += realized_pnl * 200.0

    # -------------------------------------------------------------------------
    # [Layer 2] Adaptive Risk Layer (Internal Curriculum)
    # -------------------------------------------------------------------------
    # "곳간에서 인심 난다" -> 수익이 나야 리스크 관리도 한다.
    is_profitable = _tracker.episode_pnl > 0.005 # 0.5% 이상 수익권 진입 시
    
    if is_profitable:
        # [Pro Mode] 수익 지키기 & 퀄리티 관리
        
        # MDD Penalty: 수익 까먹으면 크게 혼냄
        if _tracker.drawdown > 0.02: # 고점 대비 2% 하락 시
            reward -= (_tracker.drawdown - 0.02) * 50.0 
            
        # Sharpe Bonus: 깔끔한 우상향 차트 선호
        sharpe = _tracker.get_sharpe_ratio()
        if sharpe > 0.15:
            reward += sharpe * 5.0
    else:
        # [Survival Mode] 손실 중일 땐 리스크 지표 무시하고 복구에 올인
        # MDD 등으로 추가 감점하면 복구 의지를 꺾음
        pass

    # -------------------------------------------------------------------------
    # [Layer 3] Expert Specific Fixes (Action Correction)
    # -------------------------------------------------------------------------
    
    if expert_type == 'volatility':
        # [핵심 수정] 뇌동매매 박멸 로직
        
        # 1. 진입세(Trading Tax): 매매 버튼 누를 때마다 즉시 감점
        # "확실한 자리 아니면 수수료 아까우니 들어가지 마라"
        if action in [1, 2]: 
            reward -= 1.0  # 꽤 큰 점수 (허들 높임)

        # 2. 짤짤이 방지: 수수료도 안 나오는 익절은 감점
        if trade_done and realized_pnl > 0 and realized_pnl < 0.001:
            reward -= 5.0 

        # 3. 손실 페널티 강화: 방향 틀리면 1.5배 더 아픔
        if step_pnl < 0:
            reward += step_pnl * 50.0 # Base(100) + Extra(50) = 150배

        # 4. (중요) 기존의 '변동성 보너스(abs)'는 절대 넣지 않음. 오직 PnL로만 평가.

    elif expert_type == 'trend':
        # 추세는 수익 날 때 길게 가져가는 게 미덕
        if step_pnl > 0:
            reward += step_pnl * 30.0 # 수익 가속도 보너스

    elif expert_type == 'sideways':
        # 횡보는 시간 끌면 손해 (Time Decay)
        if holding_time > 0.15: # 에피소드 시간 15% 경과 시
            reward -= 0.1

    # Soft Clip (지나친 보상 폭발 방지)
    return float(np.tanh(reward / 20.0) * 20.0)