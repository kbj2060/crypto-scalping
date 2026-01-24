"""
강화학습 트레이딩 환경 (Clean Ver - Fixed)
- Action Masking 제거 (SAC 최적화)
- 보상 함수 강화 (Net PnL 기반)
- get_observation 인자 오류 수정
"""
import numpy as np
import torch
import logging
import pandas as pd
import sys
import os

# 상위 폴더를 경로에 추가 (config 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from .preprocess import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .mtf_processor import MTFProcessor

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
        Args:
            position_info: 포지션 정보 리스트
            current_index: (선택) 특정 시점의 인덱스. 없으면 collector의 현재 인덱스 사용
        Returns:
            (obs_seq, obs_info): 튜플 (마스크 없음)
        """
        try:
            # 1. 데이터 가져오기 (내부 데이터 직접 접근 최적화)
            candles = None
            
            # 인자로 받은 current_index가 있으면 우선 사용, 없으면 collector에서 조회
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)

            if hasattr(self.collector, 'eth_data') and isinstance(self.collector.eth_data, pd.DataFrame):
                if curr_idx is not None and curr_idx >= self.lookback:
                    candles = self.collector.eth_data.iloc[curr_idx - self.lookback : curr_idx].copy()
            
            if candles is None or len(candles) < self.lookback:
                return None
            
            # 2. 피처 엔지니어링 (이미 되어있다고 가정)
            required_feature = 'rsi_1h'
            if required_feature in candles.columns:
                df = candles
            else:
                return None
            
            # 3. 데이터 추출
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci', 
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'
            ]
            
            recent_df = df[target_cols].iloc[-self.lookback:]
            seq_features = recent_df.values.astype(np.float32)
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)

            # 4. 전략 점수
            strategy_scores = []
            for strategy in self.strategies:
                try:
                    # 전략 분석 시에도 curr_idx를 고려해야 할 수 있으나, 
                    # 현재 구조상 strategy.analyze는 collector 내부 상태를 봄.
                    # 학습 루프에서 collector.current_index를 맞춰주고 있으므로 문제 없음.
                    result = strategy.analyze(self.collector)
                    score = float(result['confidence']) if result and 'confidence' in result else 0.0
                    if result and result.get('signal') == 'SHORT': score = -score
                    strategy_scores.append(score)
                except:
                    strategy_scores.append(0.0)
            
            if position_info is None: position_info = [0.0, 0.0, 0.0]
            
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)

            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 오류: {e}")
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """보상 함수: 수수료를 반영한 순수익(Net PnL) 중심 (config 기반)"""
        reward = 0.0
        
        # 1. 평가 수익 변화량 (즉각 보상)
        reward = pnl_change * config.REWARD_MULTIPLIER
        
        if trade_done:
            # 실질 거래 비용 (config에서 가져옴)
            net_pnl = pnl - config.TRANSACTION_COST
            
            if net_pnl > 0:
                # 순수익 발생 시 큰 보상
                reward += net_pnl * config.REWARD_MULTIPLIER
                reward += np.sqrt(net_pnl * 100) * 0.5
            else:
                # 손실 시 (수수료 포함) 더 큰 페널티
                reward += net_pnl * config.LOSS_PENALTY_MULTIPLIER
        
        # 시간 비용 (기회비용, config에서 가져옴)
        reward -= config.TIME_COST
        
        return np.clip(reward, -100, 100)

    def get_state_dim(self):
        return 29
