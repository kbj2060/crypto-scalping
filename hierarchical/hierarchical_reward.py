"""
계층적 리워드 시스템
- MetaController Reward: K스텝 누적 PnL + 레짐 판별 정확도
- TacticalAgent Reward: 즉시 PnL + Goal 정렬 Intrinsic Reward
- Credit Assignment를 분리하여 각 레벨이 자기 역할에 집중
"""
import numpy as np
from collections import deque
import logging
from common import config

logger = logging.getLogger(__name__)


class HierarchicalRewardCalculator:
    """
    두 레벨의 보상을 독립적으로 계산
    
    핵심 원칙:
    1. MetaController: "좋은 방향을 골랐는가?" → K스텝 누적 결과로 평가
    2. TacticalAgent: "주어진 방향에서 잘 실행했는가?" → 즉시 + 정렬 보상
    """
    
    def __init__(self, decision_interval=5):
        self.decision_interval = decision_interval
        
        # MetaController용 누적 변수
        self.meta_cumulative_pnl = 0.0
        self.meta_step_count = 0
        self.meta_direction_at_decision = 0
        
        # TacticalAgent용 추적 변수
        self.tactical_step_pnls = deque(maxlen=100)
        
        # 전체 통계 (로깅용)
        self.episode_total_pnl = 0.0
        self.episode_meta_rewards = []
        self.episode_tactical_rewards = []
    
    def reset(self):
        """에피소드 시작 시 리셋"""
        self.meta_cumulative_pnl = 0.0
        self.meta_step_count = 0
        self.meta_direction_at_decision = 0
        self.episode_total_pnl = 0.0
        self.episode_meta_rewards = []
        self.episode_tactical_rewards = []
    
    # ==================================================================
    # Level 2: MetaController Reward
    # ==================================================================
    
    def on_meta_decision(self, direction: int):
        """MetaController가 새 결정을 내렸을 때 호출"""
        self.meta_cumulative_pnl = 0.0
        self.meta_step_count = 0
        self.meta_direction_at_decision = direction
    
    def accumulate_for_meta(self, step_pnl: float):
        """매 스텝의 PnL을 MetaController 보상 계산용으로 누적"""
        self.meta_cumulative_pnl += step_pnl
        self.meta_step_count += 1
    
    def calculate_meta_reward(self) -> float:
        """
        MetaController 보상 계산 (K스텝마다 호출)
        
        보상 = 방향 일치 PnL + 리스크 조절 보너스
        
        핵심: "올바른 방향을 선택했는가?"에만 집중
        """
        reward = 0.0
        
        # 1. 누적 PnL 기반 보상 (방향 판단 평가)
        # MetaController의 방향과 실제 시장 움직임의 일치도
        if self.meta_direction_at_decision == 0:
            # Flat 선택: 횡보장에서 Flat이면 보상 (수수료 절약)
            # 실제 움직임이 작으면 좋은 판단
            if abs(self.meta_cumulative_pnl) < 0.005:  # 0.5% 미만 움직임
                reward += 0.5  # "맞다, 횡보였다"
            else:
                reward -= 0.2  # "추세가 있었는데 못 잡았다"
        else:
            # Long 또는 Short 선택
            direction_sign = 1.0 if self.meta_direction_at_decision == 1 else -1.0
            aligned_pnl = self.meta_cumulative_pnl * direction_sign
            
            if aligned_pnl > 0:
                # 방향이 맞았다 → 강한 보상 (크기에 비례)
                reward += aligned_pnl * 50.0
            else:
                # 방향이 틀렸다 → 패널티 (더 강하게, 손실 회피)
                reward += aligned_pnl * 80.0
        
        # 2. 방향 전환 비용 (잦은 방향 변경 억제)
        # (이전 방향과 다르면 약간의 비용)
        # → train_hierarchical에서 처리
        
        # 3. 클리핑
        reward = float(np.clip(reward, -15.0, 15.0))
        
        self.episode_meta_rewards.append(reward)
        return reward
    
    # ==================================================================
    # Level 1: TacticalAgent Reward
    # ==================================================================
    
    def calculate_tactical_reward(self, step_pnl: float, trade_done: bool,
                                   realized_pnl: float, action: float,
                                   meta_goal: dict,
                                   effective_leverage: float = 1.0) -> float:
        """
        TacticalAgent 보상 계산 (매 스텝)
        
        = Extrinsic Reward (시장 PnL) + Intrinsic Reward (Goal 정렬)
        
        핵심: "MetaController의 지시를 잘 따랐는가?" + "실제 수익을 냈는가?"
        """
        reward = 0.0
        direction = meta_goal.get('direction', 0)
        risk_budget = meta_goal.get('risk_budget', 0.3)
        
        # ============================================================
        # Part A: Extrinsic Reward (실제 시장 PnL)
        # ============================================================
        
        # 1. Step PnL 반영 (Kelly로 스케일된 레버리지 적용)
        if step_pnl > 0:
            reward += step_pnl * 30.0   # 수익
        else:
            reward += step_pnl * 50.0   # 손실 회피 (Sortino)
        
        # 2. 실현 PnL
        if trade_done:
            if realized_pnl > 0:
                reward += realized_pnl * 80.0
            else:
                reward += realized_pnl * 120.0  # 손실에 더 민감
            reward -= 0.05  # 수수료
        
        # ============================================================
        # Part B: Intrinsic Reward (Goal 정렬)
        # ============================================================
        
        # MetaController의 방향과 TacticalAgent의 행동 일치도
        if direction != 0:
            # Meta가 방향을 지시한 경우
            expected_sign = 1.0 if direction == 1 else -1.0
            action_sign = np.sign(action) if abs(action) > 0.1 else 0.0
            
            if action_sign == expected_sign:
                # 방향 일치 + 수익이면 강한 보너스
                if step_pnl > 0:
                    reward += 0.3  # "지시대로 했고 돈도 벌었다"
                else:
                    reward += 0.05  # "지시대로 했지만 아직 손실 중 (인내 보상)"
            elif action_sign == -expected_sign:
                # 반대 방향 → 페널티 (단, 수익이면 감면)
                if step_pnl > 0:
                    reward -= 0.05  # "반항했지만 돈 벌었으니 봐줌"
                else:
                    reward -= 0.3   # "반항하고 돈도 잃었다"
            # action_sign == 0 (관망): 중립, 별도 보상 없음
        
        else:
            # Meta가 Flat 지시 → 포지션을 잡으면 패널티
            if abs(action) > 0.3:
                reward -= 0.2  # "쉬라고 했는데 왜 들어가?"
            else:
                reward += 0.05  # "잘 쉬고 있다"
        
        # ============================================================
        # Part C: 레버리지 과용 페널티
        # ============================================================
        
        # risk_budget 초과 시 패널티
        max_allowed = risk_budget * getattr(config, 'LEVERAGE', 20)
        if effective_leverage > max_allowed * 1.2:  # 20% 초과
            reward -= 0.3 * (effective_leverage / max_allowed - 1.0)
        
        # 4. 클리핑
        reward = float(np.clip(reward, -15.0, 15.0))
        
        self.episode_tactical_rewards.append(reward)
        self.episode_total_pnl += step_pnl
        
        return reward
    
    def get_episode_stats(self) -> dict:
        """에피소드 통계"""
        return {
            'total_pnl': self.episode_total_pnl,
            'meta_avg_reward': np.mean(self.episode_meta_rewards) if self.episode_meta_rewards else 0.0,
            'tactical_avg_reward': np.mean(self.episode_tactical_rewards) if self.episode_tactical_rewards else 0.0,
            'meta_decisions': len(self.episode_meta_rewards),
            'tactical_steps': len(self.episode_tactical_rewards),
        }
