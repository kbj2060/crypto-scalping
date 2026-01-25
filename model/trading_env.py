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

    def calculate_reward(self, step_pnl, realized_pnl, trade_done, holding_time=0):
        """
        [수정된 보상 함수] 
        - 고정 승리 보너스 제거 (단타 어뷰징 방지)
        - 손익 비례 보상 강화
        - 거래 비용(Fee)에 대한 민감도 증가
        
        Args:
            step_pnl: 이번 스텝에서의 평가금액 변화량 (Unrealized Delta)
            realized_pnl: 확정 손익 (청산 시에만 값 있음, 아니면 0)
            trade_done: 거래 종료 여부
            holding_time: 보유 시간
        """
        reward = 0.0
        
        # 1. 과정 보상 (평가금액 변동 반영)
        # 변동성이 작을 때도 학습되도록 스케일링 (x100)
        reward += step_pnl * 100.0 
        
        # 2. 결과 보상 (청산 시)
        if trade_done:
            fee = config.TRANSACTION_COST  # 예: 0.0015 (0.15%)
            net_pnl = realized_pnl - fee
            
            # [핵심 수정] 고정 보너스(+1.0) 제거 -> 순수 수익률 비례 보상으로 변경
            if net_pnl > 0:
                # 수익 날 때는 팍팍 밀어줌 (수익률 1% = +2.0점)
                reward += net_pnl * 200.0  
            else:
                # 손실 날 때는 수익보다 더 아프게 (손실회피 성향)
                reward += net_pnl * 250.0  
            
            # [추가] 잦은 매매 방지용 고정 페널티 (수수료 외에 심리적 비용)
            reward -= 0.2
            
        else:
            # 홀딩 비용 (너무 오래 들고 있지 않도록 미세 조정)
            reward -= 0.001 * holding_time

        # 리워드 클리핑 (학습 안정화)
        return np.clip(reward, -10, 10)

    def get_state_dim(self):
        return 29
