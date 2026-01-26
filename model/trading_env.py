"""
PPO 전용 트레이딩 환경 (3-Action Version)
- Action 0: Neutral (HOLD/청산)
- Action 1: Long (진입/유지)
- Action 2: Short (진입/유지)
- 1-Step Switching 지원
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
        [3-Action Target Position Reward]
        - 0:Neutral, 1:Long, 2:Short
        - 1-Step Switching 지원에 따른 보상 체계
        Args:
            step_pnl: 이번 스텝의 평가금액 변동 (Unrealized Delta)
            realized_pnl: 확정 손익 (Exit/스위칭 시에만 존재)
            trade_done: 거래 종료 여부
            holding_time: 보유 시간 (사용하지 않음, 호환성 유지)
            action: 현재 취한 행동 (0:Neutral, 1:Long, 2:Short)
            prev_position: 이전 스텝의 포지션 (None, 'LONG', 'SHORT')
            current_position: 현재 스텝의 포지션 (None, 'LONG', 'SHORT')
        """
        reward = 0.0
        
        # 1. HOLD/Maintenance 보너스
        # (이전과 현재가 같음 == 유지 중)
        if prev_position == current_position:
            if current_position is None:
                reward += 0.001 # 관망
            else:
                # 추세 추종 보상 (200배)
                reward += step_pnl * 200.0
                reward += 0.005 # 수고비
        
        # 2. 거래 발생 (진입 or 스위칭 or 청산)
        else:
            # 진입/스위칭 비용 없음 (빠른 태세전환 유도)
            pass

        # 3. 청산/스위칭 결과 보상 (Realized PnL)
        if trade_done:
            fee = getattr(config, 'TRANSACTION_COST', 0.001)
            # 스위칭의 경우 수수료가 두 번 발생할 수도 있지만(청산+진입), 
            # 여기서는 1회분만 반영하거나, 엄격하게 2배 할 수 있음. 
            # 일단 1.5배 정도로 평균 내서 적용
            actual_fee = fee * 1.5 if (prev_position is not None and current_position is not None) else fee
            
            reward -= actual_fee * 1.0
            
            net_pnl = realized_pnl - actual_fee
            
            if net_pnl > 0:
                reward += net_pnl * 200.0
                if net_pnl > 0.01: reward += 5.0
            else:
                reward += net_pnl * 200.0
                
        return np.clip(reward, -20, 20)

    def get_state_dim(self):
        return 29
