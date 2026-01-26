"""
PPO 전용 트레이딩 환경 (3-Action Version - Aggressive Trading)
- Action 0: Neutral (HOLD/청산)
- Action 1: Long (진입/유지)
- Action 2: Short (진입/유지)
- 1-Step Switching 지원
- 진입 비용 완전 삭제 (재진입 유도)
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
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            if curr_idx is None or curr_idx < self.lookback: return None
            df = self.collector.eth_data
            if df is None or curr_idx >= len(df): return None

            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            if not self.scaler_fitted: return None
            for col in target_cols:
                if col not in df.columns: df[col] = 0.0

            recent_df = df[target_cols].iloc[curr_idx - self.lookback : curr_idx]
            if len(recent_df) < self.lookback: return None
                
            seq_features = recent_df.values.astype(np.float32)
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)

            strategy_scores = []
            for i in range(len(self.strategies)):
                col_name = f'strategy_{i}'
                if col_name in df.columns: score = float(df[col_name].iloc[curr_idx])
                else: score = 0.0
                strategy_scores.append(score)
            
            if position_info is None: position_info = [0.0, 0.0, 0.0]
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)

            return (obs_seq, obs_info_tensor)
        except Exception: return None

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0, action=0, prev_position=None, current_position=None):
        """
        [PPO Aggressive Trading Reward]
        - 진입 비용 완전 삭제 (재진입 유도)
        - 홀딩/청산 보상 강화
        - 3-Action 구조: 0=Neutral, 1=Long, 2=Short
        """
        reward = 0.0
        
        # 1. HOLD 보상 (수익 추세 강화)
        if action == 0:
            if current_position is not None:
                # 방향 맞으면 200배 보상 (확실한 유인책)
                reward += step_pnl * 200.0 
                # 버티기 수고비
                reward += 0.005 
            else:
                # 관망 보너스
                reward += 0.001

        # 2. [수정] 진입 비용 완전 삭제
        # if prev_position is None and current_position is not None:
        #      reward -= 0.5  <-- 삭제됨 (이제 공짜로 진입 가능)

        # 3. 청산 보상 (승리 쾌감 주입)
        # 3-Action 구조에서는 action=0 (Neutral)이 청산을 의미
        if trade_done:
            fee = getattr(config, 'TRANSACTION_COST', 0.001)
            reward -= fee * 1.0 
            
            net_pnl = realized_pnl - fee
            
            if net_pnl > 0:
                # 익절: x200
                reward += net_pnl * 200.0
                # [추가] 이기면 무조건 +1.0점 보너스 (작은 수익도 칭찬)
                reward += 1.0
                if net_pnl > 0.01: 
                    reward += 5.0  # 대박 보너스
            else:
                # 손절: x200
                reward += net_pnl * 200.0 
                # 손절 페널티 없음 (손실 자체가 고통임)

        return np.clip(reward, -20, 20)

    def get_state_dim(self):
        return 29
