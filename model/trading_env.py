"""
강화학습 트레이딩 환경
기존 전략 Score와 시장 데이터를 결합하는 환경 인터페이스
원시 데이터 보존 + Z-Score 정규화
"""
import numpy as np
import torch
import logging
import pandas as pd
from .preprocess import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .mtf_processor import MTFProcessor

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """트레이딩 환경: 상태 관측 및 보상 계산"""
    def __init__(self, data_collector, strategies, lookback=40):
        """
        Args:
            data_collector: DataCollector 인스턴스
            strategies: 전략 리스트
            lookback: 충분한 샘플 수 (기본 40)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback
        
        # 전처리 파이프라인 (Z-Score 정규화)
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False  # 스케일러 학습 여부
        
        # 피처 엔지니어링 모듈 (인스턴스는 매번 새로 생성)
        # FeatureEngineer와 MTFProcessor는 데이터를 받아서 생성하므로 여기서는 초기화하지 않음

    def get_observation(self, position_info=None):
        """
        현재 상태 관측 (29개 고급 시계열 피처 + Z-Score 정규화 + 포지션 정보)
        
        Args:
            position_info: [포지션(1/0/-1), 미실현PnL, 보유시간(정규화)] 리스트
                          - 보유시간: (current_index - entry_index) / max_steps (0~1 사이)
                          None이면 [0.0, 0.0, 0.0]으로 처리
        
        Returns:
            (obs_seq, obs_info): 튜플
                - obs_seq: (1, 40, 29) 텐서 - 29개 시계열 피처
                - obs_info: (1, 13) 텐서 - 전략 점수(10) + 포지션 정보(3)
        """
        try:
            # 1. 원본 데이터 수집 (lookback보다 넉넉하게 가져옴)
            # MTF 계산을 위해 최소 200봉 이상 필요할 수 있음
            candles = self.collector.get_candles('ETH', count=self.lookback + 60)
            if candles is None or len(candles) < self.lookback:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: {self.lookback}개)")
                return None
            
            # 2. [핵심 변경] DQN과 동일한 고급 피처 생성 Pipeline 적용
            # 인덱스가 DatetimeIndex인지 확인 및 변환
            if not isinstance(candles.index, pd.DatetimeIndex):
                if 'timestamp' in candles.columns:
                    candles.index = pd.to_datetime(candles['timestamp'])
                else:
                    # 인덱스가 없으면 현재 시간 기준으로 생성
                    candles.index = pd.date_range(end=pd.Timestamp.now(), periods=len(candles), freq='3min')
            
            # BTC 데이터도 가져오기 (상관관계 피처용)
            btc_candles = self.collector.get_candles('BTC', count=len(candles))
            if btc_candles is not None and len(btc_candles) >= len(candles):
                # 인덱스 정렬
                if not isinstance(btc_candles.index, pd.DatetimeIndex):
                    if 'timestamp' in btc_candles.columns:
                        btc_candles.index = pd.to_datetime(btc_candles['timestamp'])
                    else:
                        btc_candles.index = pd.date_range(end=pd.Timestamp.now(), periods=len(btc_candles), freq='3min')
                # 공통 인덱스로 정렬
                common_index = candles.index.intersection(btc_candles.index)
                if len(common_index) > 0:
                    candles = candles.loc[common_index]
                    btc_candles = btc_candles.loc[common_index]
            else:
                btc_candles = None
            
            # (1) 기본 기술적 지표 (25개)
            feature_engineer = FeatureEngineer(candles, btc_candles)
            df = feature_engineer.generate_features()
            
            if df is None:
                logger.warning("피처 생성 실패")
                return None
            
            # (2) 멀티 타임프레임 지표 (4개)
            mtf_processor = MTFProcessor(df)
            df = mtf_processor.add_mtf_features()
            
            # (3) 학습에 사용할 컬럼만 선택 (DQN에서 검증된 피처들)
            target_cols = [
                'log_return', 'roll_return_6', 'atr_ratio', 'bb_width', 'bb_pos', 
                'rsi', 'macd_hist', 'hma_ratio', 'cci',  # 가격/변동성
                'rvol', 'taker_ratio', 'cvd_change', 'mfi', 'cmf', 'vwap_dist',  # 거래량
                'wick_upper', 'wick_lower', 'range_pos', 'swing_break', 'chop',  # 패턴
                'btc_return', 'btc_rsi', 'btc_corr', 'btc_vol', 'eth_btc_ratio',  # 상관관계
                'rsi_15m', 'trend_15m', 'rsi_1h', 'trend_1h'  # MTF
            ]
            
            # 존재하지 않는 컬럼은 0으로 채움
            for col in target_cols:
                if col not in df.columns:
                    logger.warning(f"피처 {col}이 없습니다. 0으로 채웁니다.")
                    df[col] = 0.0
            
            # 마지막 lookback 개수만큼 자르기
            recent_df = df[target_cols].iloc[-self.lookback:]
            
            # 4. 전처리 (Z-Score)
            seq_features = recent_df.values.astype(np.float32)
            
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
            
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)  # (1, 40, 29)
            
            # 5. 기술적 전략 Score 수집 (LONG/SHORT 부호 인코딩 반영)
            strategy_scores = []
            for strategy in self.strategies:
                try:
                    result = strategy.analyze(self.collector)
                    if result and 'confidence' in result:
                        score = float(result['confidence'])
                        # SHORT 신호는 음수로 인코딩
                        if result.get('signal') == 'SHORT':
                            score = -score
                        strategy_scores.append(score)
                    else:
                        strategy_scores.append(0.0)
                except Exception as e:
                    logger.debug(f"전략 {strategy.name} 분석 실패: {e}")
                    strategy_scores.append(0.0)
            
            # 6. [해결] 10개 전략 점수 + 3개 포지션 정보 = 13차원
            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)  # (1, 13)
            
            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """
        개선된 보상 함수 (공격성 균형)
        - 큰 수익에 대한 인센티브 유지
        - 하지만 제곱이 아닌 sqrt로 완화
        """
        reward = 0.0
        
        reward = pnl_change * 300
        
        if trade_done:
            if pnl > 0:
                # A. 선형 주보상
                reward += pnl * 100
                
                # B. [수정] 제곱 → sqrt (완화된 비선형)
                # 큰 수익에 추가 보너스 주되, 폭주는 방지
                # 1% → sqrt(1) = 1.0 (10% 보너스)
                # 2% → sqrt(2) = 1.41 (14% 보너스)
                # 5% → sqrt(5) = 2.24 (22% 보너스)
                reward += np.sqrt(pnl * 100) * 0.1
                
                # C. 연속 승리 보너스
                reward += np.tanh(pnl * 100) * 0.5
                
            else:
                # 손실 페널티 (약간 더 아프게)
                reward += pnl * 120  # 수익보다 20% 더 페널티
            
            reward -= 0.2
        
        reward -= 0.0005
        
        return np.clip(reward, -100, 100)

    def get_state_dim(self):
        """상태 차원 반환 (29개 시계열 피처)"""
        return 29  # 29개 고급 시계열 피처
