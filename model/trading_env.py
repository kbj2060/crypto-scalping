"""
PPO 전용 트레이딩 환경 (4-Action Version)
- Action 0: HOLD (유지)
- Action 1: ENTER_LONG (진입)
- Action 2: ENTER_SHORT (진입)
- Action 3: EXIT (청산)
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

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0, action=0, prev_position=None, current_position=None):
        """
        [4-Action Optimized Reward - 학습 개선 버전]
        - 0:HOLD, 1:LONG, 2:SHORT, 3:EXIT
        - Valid Entry Cost 제거, 보상 스케일 재조정, 클리핑 범위 확대
        Args:
            step_pnl: 이번 스텝의 평가금액 변동 (Unrealized Delta)
            realized_pnl: 확정 손익 (Exit 시에만 존재)
            trade_done: 거래 종료 여부
            holding_time: 보유 시간 (사용하지 않음, 호환성 유지)
            action: 현재 취한 행동 (0:HOLD, 1:LONG, 2:SHORT, 3:EXIT)
            prev_position: 이전 스텝의 포지션 (None, 'LONG', 'SHORT')
            current_position: 현재 스텝의 포지션 (None, 'LONG', 'SHORT')
        """
        reward = 0.0
        
        # 1. HOLD 보너스 (Trend Riding)
        if action == 0:
            if current_position is not None:
                # 포지션 보유 중 HOLD: 추세가 맞으면(step_pnl > 0) 보상
                reward += step_pnl * 15.0  # x20.0 → x15.0 (약간 완화)
                # 미세 보너스 (조급함 방지)
                reward += 0.001 
            else:
                # 무포지션 HOLD: 관망 보너스
                reward += 0.001

        # 2. Invalid Action 페널티 제거 (마스킹이 이미 차단하므로 불필요)

        # 3. Valid Entry Cost 제거 (진입 장벽 완화)

        # 4. EXIT Rewards (Realized PnL)
        if trade_done and action == 3:
            fee = getattr(config, 'TRANSACTION_COST', 0.001)
            reward -= fee * 10.0 # 수수료
            
            net_pnl = realized_pnl - fee
            if net_pnl > 0:
                reward += net_pnl * 250.0 # 익절 (x200 → x250, 증가)
                if net_pnl > 0.01: reward += 2.0 # 1% 이상 대박 보너스
            else:
                reward += net_pnl * 250.0 # 손절 (x300 → x250, 대칭화)
                reward -= 0.2 # 패배 고정 비용 (-0.5 → -0.2, 완화)

        return np.clip(reward, -20, 20)  # 클리핑 범위 확대 (-10, 10 → -20, 20)

    def get_state_dim(self):
        return 29
