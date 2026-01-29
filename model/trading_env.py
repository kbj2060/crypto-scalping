"""
PPO 전용 트레이딩 환경 (Dynamic Neutral Penalty Applied)
- 점진적 Neutral 페널티: 에피소드가 진행될수록 관망에 대한 감점 강화
- 거래 횟수 감소 방지 및 적극적 매매 유도
"""
import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

from . import config
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)

class TradingEnvironment:
    def __init__(self, data_collector, strategies, lookback=None):
        self.collector = data_collector
        self.strategies = strategies
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False
        
        self.trade_count = 0
        self.step_pnl_ema = 0.0
        self.current_episode = 1  # [New] 현재 에피소드 추적

    def reset_reward_states(self):
        self.trade_count = 0
        self.step_pnl_ema = 0.0

    def get_observation(self, position_info=None, current_index=None):
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
            
            seq = self.preprocessor.transform(recent_df.values.astype(np.float32))
            obs_seq = torch.FloatTensor(seq).unsqueeze(0)

            # Info Dim 15
            scores = []
            for i in range(len(self.strategies)):
                col = f'strategy_{i}'
                scores.append(float(df[col].iloc[curr_idx]) if col in df.columns else 0.0)

            if position_info is None: position_info = [0.0, 0.0, 0.0]
            
            obs_info_np = np.concatenate([scores, position_info], dtype=np.float32)
            obs_info = torch.FloatTensor(obs_info_np).unsqueeze(0)
            
            return (obs_seq, obs_info)
        except: return None

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0, action=0, prev_position=None, current_position=None):
        reward = 0.0
        
        # [솔루션 2] Dynamic Neutral Penalty (500→200 가속, 기본 0.002)
        # 200 Ep 기준으로 페널티가 더 빨리 세짐, 기본 숨 쉬는 비용 상향
        if current_position is None and not trade_done:
            dynamic_penalty = min(0.001 * self.current_episode / 200, 0.005)
            base_penalty = 0.002
            total_penalty = max(base_penalty, dynamic_penalty)
            reward -= total_penalty 

        # 2. Step Reward (EMA)
        alpha = 0.33
        if current_position is not None:
            self.step_pnl_ema = alpha * step_pnl + (1 - alpha) * self.step_pnl_ema
        else:
            self.step_pnl_ema = (1 - alpha) * self.step_pnl_ema
            
        reward += self.step_pnl_ema * 50.0

        # 3. Directional Bonus
        if current_position is not None and step_pnl > 0:
            reward += 0.02

        # 4. Terminal Reward
        if trade_done:
            fee = 0.0005 
            net_pnl = realized_pnl - fee
            
            # [수정] 비대칭 보상 (Asymmetric Reward, Prospect Theory 역이용)
            if net_pnl > 0:
                reward += net_pnl * 200.0   # 수익은 x200 (강력한 칭찬)
            else:
                reward += net_pnl * 80.0    # 손실은 x80 (덜 아픈 회초리)
            reward -= 0.001 
            
            if holding_time < 0.02: reward -= 0.03
            if net_pnl < -0.02: reward -= 1.0
            
            self.trade_count += 1
            self.step_pnl_ema = 0.0

        return reward

    def get_state_dim(self):
        return 29