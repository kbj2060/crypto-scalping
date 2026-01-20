"""
강화학습 트레이딩 환경
기존 전략 Score와 시장 데이터를 결합하는 환경 인터페이스
원시 데이터 보존 + Z-Score 정규화
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
            lookback: 충분한 샘플 수 (기본 40)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback
        
        # 전처리 파이프라인 (Z-Score 정규화)
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False  # 스케일러 학습 여부

    def get_observation(self):
        """
        현재 상태 관측 (원시 데이터 + Z-Score 정규화)
        
        Returns:
            observation: (1, seq_len, state_dim) 텐서
                - close_prices: 원시 가격 데이터 (웨이블릿 제거)
                - volumes: 거래량 데이터
                - strategy_scores: 각 전략의 신뢰도 점수
        """
        try:
            # 1. 원본 데이터 수집
            candles = self.collector.get_candles('ETH', count=self.lookback)
            if candles is None or len(candles) < self.lookback:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: {self.lookback}개)")
                return None
            
            # 가격 및 거래량 추출 (원시 데이터 사용, 웨이블릿 제거)
            close_prices = candles['close'].values.astype(np.float32)
            volumes = candles['volume'].values.astype(np.float32)
            
            # 2. 기술적 전략 Score 수집
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
            
            # 3. 피처 결합 (원시 가격, 거래량, 전략 점수)
            # 마지막 20봉의 윈도우 데이터 생성 (xLSTM 입력용)
            window_size = 20
            prices_window = close_prices[-window_size:]
            volumes_window = volumes[-window_size:]
            
            # 전략 점수를 20봉 시퀀스 전체에 복제
            scores_array = np.array(strategy_scores, dtype=np.float32)
            scores_tiled = np.tile(scores_array, (window_size, 1))  # (20, num_strategies)
            
            # 원시 피처 결합: (20, 2 + num_strategies)
            raw_features = np.column_stack([
                prices_window,
                volumes_window,
                scores_tiled
            ])
            
            # 4. Z-Score 정규화 적용 (fit은 학습 시작 전에 이미 완료됨)
            # transform만 사용하여 가격의 상대적 높낮이 맥락 유지
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
            
            normalized_obs = self.preprocessor.transform(raw_features)
            
            # 배치 차원 추가: (1, 20, 2 + num_strategies)
            observation = torch.FloatTensor(normalized_obs).unsqueeze(0)
            
            return observation
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """
        보상 계산 (현실화된 보상 체계 + 비선형 보상)
        
        Args:
            pnl: 손익 (수익률)
            trade_done: 거래 완료 여부
            holding_time: 보유 시간 (분)
            pnl_change: 이전 스텝 대비 수익률의 변화 (새로 추가)
        Returns:
            reward: 보상값
        """
        reward = 0.0
        
        # 1. 미실현 손익의 '변화량'만 보상 (계속 들고 있다고 보상을 퍼주지 않음)
        # pnl_change가 0이면 보상도 0 (변화가 없으면 보상 없음)
        reward = pnl_change * 100
        
        # 2. 거래가 완료되었을 때만 '실현 수익'에 비선형 보상 부여
        if trade_done:
            if pnl > 0:
                # 수익이 클수록 보상을 제곱으로 부여하여 큰 수익을 유도
                # 예: 0.5% 수익 → (0.005 * 100)^2 / 10 = 0.25
                #     2% 수익 → (0.02 * 100)^2 / 10 = 4.0 (16배 차이!)
                reward += (pnl * 100) ** 2 / 10
            else:
                # 손실은 그대로 페널티 (비선형 적용 안 함)
                reward += pnl * 20
            
            reward -= 0.01  # 수수료 페널티
        
        # 3. 시간 페널티 완화 (기존 -0.0005 -> -0.0001)
        # 큰 추세를 끝까지 타도록 유도
        reward -= 0.0001
        
        return reward

    def get_state_dim(self):
        """상태 차원 반환"""
        return 2 + self.num_strategies  # market_data(2) + strategy_scores(num_strategies)
