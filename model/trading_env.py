"""
강화학습 트레이딩 환경 (Refactored for SAC)
- 전략 신호 Pre-calculation 지원
- Action Dead-zone (Exit) 로직 지원
- 보상 함수: 실현 손익(Realized PnL) 중심
"""
import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)

class TradingEnvironment:
    def __init__(self, data_collector, strategies, lookback=None):
        """
        Args:
            data_collector: DataCollector 인스턴스
            strategies: 전략 리스트
            lookback: 충분한 샘플 수 (None이면 config.LOOKBACK 사용)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False

    def get_observation(self, position_info=None, current_index=None):
        """
        현재 상태 관측
        - 전략 신호는 이미 df에 계산되어 있다고 가정 (strategy_0, strategy_1...)
        - next_state 계산 시 인덱스만 +1 해서 호출하면 됨 (Data Leakage 없음)
        
        Args:
            position_info: 포지션 정보 리스트 [pos_val, unrealized_pnl, holding_time]
            current_index: (선택) 특정 시점의 인덱스. 없으면 collector의 현재 인덱스 사용
        Returns:
            (obs_seq, obs_info): 튜플
        """
        try:
            # 1. 인덱스 설정
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            
            # 데이터 유효성 검사
            if curr_idx is None or curr_idx < self.lookback:
                return None
            
            df = self.collector.eth_data
            if curr_idx >= len(df):  # 끝 도달
                return None

            # 2. 시계열 데이터 추출 (이미 피처 엔지니어링 완료됨)
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            # 스케일러가 학습되지 않았으면 경고 (초기화 이슈 방지)
            if not self.scaler_fitted:
                # logger.warning("⚠️ 스케일러가 학습되지 않은 상태에서 관측 시도")
                pass

            # 데이터 슬라이싱
            recent_df = df[target_cols].iloc[curr_idx - self.lookback : curr_idx]
            seq_features = recent_df.values.astype(np.float32)
            
            # 정규화 적용
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)

            # 3. 전략 점수 (Pre-calculated Columns 사용)
            # train_sac.py에서 미리 'strategy_0', 'strategy_1'... 컬럼을 만들어둠
            strategy_scores = []
            for i in range(self.num_strategies):
                col_name = f'strategy_{i}'
                if col_name in df.columns:
                    # 현재 시점(curr_idx-1)의 점수 가져오기
                    score = df[col_name].iloc[curr_idx - 1] 
                else:
                    score = 0.0
                strategy_scores.append(score)
            
            if position_info is None: 
                position_info = [0.0, 0.0, 0.0]
            
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)

            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 오류: {e}")
            return None

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0):
        """
        [SAC 최적화 보상 함수]
        - Reward Scale 대폭 축소 (Q-value 폭발 방지)
        - Step PnL 반영 비율 축소
        - Exit 보상/페널티 스케일 조정
        """
        reward = 0.0
        
        # 1. 포지션 유지 중 PnL 변화 (변동성 반영)
        # PPO(30~100) -> SAC(5)로 대폭 축소하여 변동성 매매 방지
        reward += step_pnl * 5.0
        
        # 2. 결과 보상 (청산 시)
        if trade_done:
            fee = config.TRANSACTION_COST
            net_pnl = realized_pnl - fee
            
            if net_pnl > 0:
                # 익절: PPO(200~250) -> SAC(40)
                reward += net_pnl * 40.0
            else:
                # 손절: PPO(250~300) -> SAC(60)
                # 손실은 수익보다 조금 더 크게 페널티 부여
                reward += net_pnl * 60.0
                # 손절/청산 고정 페널티 (Switching/Exit penalty)
                reward -= 0.05
        else:
            # 3. HOLD 보상 (아주 작게)
            # SAC는 0.00005 수준으로 설정 (Holding Cost 대신 생존/관망 보너스)
            # 불필요한 진입을 막고 좋은 자리를 기다리게 유도
            reward += 0.00005

        # 4. Reward Clipping
        # SAC Critic 안정성을 위해 좁은 범위 [-2, 2] 적용
        return np.clip(reward, -2.0, 2.0)

    def get_state_dim(self):
        return 29
