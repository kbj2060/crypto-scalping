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
        [PPO 순수 PnL 기반 Reward]
        - 보너스 제거: 순수 실력으로만 승부
        - 작은 스케일: PPO 학습 안정성 확보
        - 3-Action 구조: 0=Neutral, 1=Long, 2=Short
        """
        reward = 0.0
        
        # 1. 스텝 보상: 평가금액 변동폭 * 10 (변동성을 버티는 힘)
        # (예: 1% 오르면 0.01 * 10 = 0.1점)
        if current_position is not None:
            step_reward = step_pnl * 10.0
            reward += step_reward
        # 포지션 없으면 0 (관망은 보상 없음)
        
        # 2. 종료 보상: 최종 확정 수익 * 50 (결과에 대한 책임)
        # (예: 2% 익절하면 0.02 * 50 = 1.0점)
        if trade_done:
            fee = getattr(config, 'TRANSACTION_COST', 0.001)
            net_pnl = realized_pnl - fee
            terminal_reward = net_pnl * 50.0
            reward += terminal_reward
            # 보너스(+1, +5) 모두 삭제 -> 순수 실력으로만 승부
        
        # 클리핑은 유지하되 범위를 좁힘 (-10 ~ 10 정도가 PPO에 가장 좋음)
        return np.clip(reward, -10, 10)

    def get_state_dim(self):
        return 29
