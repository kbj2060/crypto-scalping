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
        [SAC 기반 수정된 보상 함수 for PPO]
        - Unrealized PnL(평가손익) 노이즈 최소화
        - Realized PnL(확정손익) 중심의 강력한 피드백
        
        Args:
            step_pnl: 이번 스텝에서의 평가금액 변화량 (Unrealized Delta)
            realized_pnl: 확정 손익 (청산 시에만 값 있음, 아니면 0)
            trade_done: 거래 종료 여부
            holding_time: 보유 시간
        """
        reward = 0.0
        
        # 1. 거래 확정 시 (Realized PnL) - 여기가 핵심!
        if trade_done:
            # 수수료 반영 (진입/청산 합쳐서 약 0.1% 가정)
            # config.TRANSACTION_COST가 있다면 그것을 사용
            transaction_cost = getattr(config, 'TRANSACTION_COST', 0.001)
            net_pnl = realized_pnl - transaction_cost
            
            if net_pnl > 0:
                # 익절: 수익률 비례 보상 + 성공 보너스
                reward += net_pnl * 500.0
                reward += 1.0  # 성공 샷 보너스
            else:
                # 손절: 손실은 더 아프게 (x600)
                reward += net_pnl * 600.0
        
        else:
            # 2. 포지션 유지 중 (Hold)
            # 시간 경과 페널티 (너무 오래 끄는 것 방지)
            reward -= 0.001 * holding_time

            # [옵션] PPO를 위한 최소한의 가이드 (Step PnL)
            # SAC와 달리 PPO는 과정 점수가 없으면 학습이 너무 느릴 수 있음.
            # 하지만 이전처럼 과도하게 주지 않고, 확정 손익의 1/100 수준으로 미세하게 부여
            # (이 부분이 싫다면 주석 처리해도 됩니다)
            reward += step_pnl * 5.0 

            # [추가] 극단적인 평가 손실 상태라면 추가 페널티 (손절 유도)
            # step_pnl은 '변화량'이므로, 여기서는 누적 수익률을 알 수 없어서
            # 환경에서 현재 unrealized_pnl을 받아와야 정확하지만, 
            # 일단 로직 유지를 위해 생략하거나 필요 시 인자 추가 필요.
            pass

        # [핵심] 보상 스케일링 (신경망 안정성 확보)
        # 기존 로직: reward * 0.1 -> 클리핑
        reward = reward * 0.1

        return np.clip(reward, -5, 5)

    def get_state_dim(self):
        return 29
