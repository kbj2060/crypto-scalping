"""
강화학습 트레이딩 환경
기존 전략 Score와 시장 데이터를 결합하는 환경 인터페이스
웨이블릿 변환을 이용한 노이즈 제거 포함
"""
import numpy as np
import torch
import logging
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """트레이딩 환경: 상태 관측 및 보상 계산"""
    def __init__(self, data_collector, strategies, lookback=40):
        """
        Args:
            data_collector: DataCollector 인스턴스
            strategies: 전략 리스트
            lookback: 웨이블릿 변환을 위한 충분한 샘플 수 (기본 40)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback  # 웨이블릿 변환을 위해 충분한 샘플 확보
        
        # 전처리 파이프라인 (xLSTM은 -1~1 사이 입력 선호)
        self.preprocessor = DataPreprocessor(feature_range=(-1, 1))

    def get_observation(self):
        """
        현재 상태 관측 (웨이블릿 노이즈 제거 + 정규화)
        
        Returns:
            observation: (1, seq_len, state_dim) 텐서
                - denoised_prices: 웨이블릿으로 노이즈 제거된 가격 데이터
                - volumes: 거래량 데이터
                - strategy_scores: 각 전략의 신뢰도 점수
        """
        try:
            # 1. 원본 데이터 수집 (충분한 길이 확보 - 웨이블릿 변환용)
            candles = self.collector.get_candles('ETH', count=self.lookback)
            if candles is None or len(candles) < self.lookback:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: {self.lookback}개)")
                return None
            
            # 가격 및 거래량 추출
            close_prices = candles['close'].values.astype(np.float32)
            volumes = candles['volume'].values.astype(np.float32)
            
            # 2. Wavelet Denoising 적용 (가격 데이터의 노이즈 제거)
            # 웨이블릿 변환은 시계열 데이터에서 불필요한 노이즈를 제거하고 추세를 선명하게 만듦
            denoised_prices = self.preprocessor.wavelet_denoising(close_prices)
            
            # 3. 기술적 전략 Score 수집
            strategy_scores = []
            for strategy in self.strategies:
                try:
                    result = strategy.analyze(self.collector)
                    if result and 'confidence' in result:
                        score = float(result['confidence'])
                        strategy_scores.append(score)
                    else:
                        strategy_scores.append(0.0)
                except Exception as e:
                    logger.debug(f"전략 {strategy.name} 분석 실패: {e}")
                    strategy_scores.append(0.0)
            
            # 4. 피처 결합 (노이즈 제거된 가격, 거래량, 전략 점수)
            # 마지막 20봉의 윈도우 데이터 생성 (xLSTM 입력용)
            window_size = 20
            denoised_window = denoised_prices[-window_size:]
            volumes_window = volumes[-window_size:]
            
            # 전략 점수를 20봉 시퀀스 전체에 복제
            # 팁: 점수 데이터를 20봉 시퀀스 전체에 붙여주어 xLSTM이 문맥을 이해하게 함
            scores_array = np.array(strategy_scores, dtype=np.float32)
            scores_tiled = np.tile(scores_array, (window_size, 1))  # (20, num_strategies)
            
            # 원시 피처 결합: (20, 2 + num_strategies)
            raw_features = np.column_stack([
                denoised_window,
                volumes_window,
                scores_tiled
            ])
            
            # 5. Min-Max Scaling 적용 (-1~1 범위)
            # xLSTM의 지수 게이팅 폭발을 막기 위한 핵심 단계
            normalized_obs = self.preprocessor.fit_transform(raw_features)
            
            # 배치 차원 추가: (1, 20, 2 + num_strategies)
            observation = torch.FloatTensor(normalized_obs).unsqueeze(0)
            
            return observation
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0):
        """
        보상 계산
        
        Args:
            pnl: 손익 (수익률)
            trade_done: 거래 완료 여부
            holding_time: 보유 시간 (분)
        Returns:
            reward: 보상값
        """
        # 수익률 보상 (100배 스케일링)
        reward = pnl * 100
        
        # 거래 횟수 조절 페널티 (수수료)
        if trade_done:
            reward -= 0.01
        
        # 보유 시간 페널티 (과도한 보유 방지)
        if holding_time > 0:
            reward -= holding_time * 0.001
        
        return reward

    def get_state_dim(self):
        """상태 차원 반환"""
        return 2 + self.num_strategies  # market_data(2) + strategy_scores(num_strategies)
