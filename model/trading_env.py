"""
PPO 전용 트레이딩 환경 (Independent Version)
- Discrete Action Space (0:HOLD, 1:LONG, 2:SHORT)
- xLSTM 입력 최적화 (Sequence + Info Tuple 반환)
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
        self.collector = data_collector
        self.strategies = strategies
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False

    def get_observation(self, position_info=None, current_index=None):
        """
        PPO 상태 관측
        Returns: (obs_seq, obs_info) 튜플 또는 None
        """
        try:
            # 1. 인덱스 확인
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            
            # 데이터 유효성 검사
            if curr_idx is None or curr_idx < self.lookback:
                return None
            
            df = self.collector.eth_data
            if df is None or curr_idx >= len(df):
                return None

            # 2. 시계열 피처 추출
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            # 스케일러 체크
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다.")
                return None

            # 존재하지 않는 컬럼은 0으로 채움
            for col in target_cols:
                if col not in df.columns:
                    df[col] = 0.0

            recent_df = df[target_cols].iloc[curr_idx - self.lookback : curr_idx]
            
            if len(recent_df) < self.lookback:
                return None
                
            seq_features = recent_df.values.astype(np.float32)
            
            # 정규화
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)  # (1, seq, dim)

            # 3. 전략 신호 (Pre-calculated Columns 사용)
            strategy_scores = []
            for i in range(len(self.strategies)):
                col_name = f'strategy_{i}'
                # 미리 계산된 컬럼이 있으면 사용, 없으면 0
                if col_name in df.columns:
                    score = float(df[col_name].iloc[curr_idx])
                else:
                    score = 0.0
                strategy_scores.append(score)
            
            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            
            # 정보 벡터 결합
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)  # (1, info_dim)

            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.debug(f"관측 오류: {e}")
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0):
        """
        PPO 전용 보상 함수
        - Realized PnL(실현 손익)에 집중
        - 스텝 보상(Unrealized)은 노이즈를 줄이기 위해 제거하거나 최소화
        """
        reward = 0.0
        
        if trade_done:
            # 거래 종료 시 (익절/손절)
            transaction_cost = config.TRANSACTION_COST
            net_pnl = pnl - transaction_cost
            
            if net_pnl > 0:
                reward = net_pnl * config.REWARD_MULTIPLIER  # 수익 보상
                reward += np.sqrt(net_pnl * 100) * 0.5  # 제곱근 보너스
                reward += np.tanh(net_pnl * 100) * 0.5  # 승리 보너스
            else:
                reward = net_pnl * config.LOSS_PENALTY_MULTIPLIER  # 손실 페널티 (더 아프게)
        else:
            # 포지션 유지 중
            reward = -config.TIME_COST * holding_time  # 시간 비용 (빨리 승부)
            
        return np.clip(reward, -100, 100)

    def get_state_dim(self):
        return 29
