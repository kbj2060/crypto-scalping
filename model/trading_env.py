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

    def get_observation(self, position_info=None):
        """
        현재 상태 관측 (7개 핵심 시계열 피처 + Z-Score 정규화 + 포지션 정보)
        
        Args:
            position_info: [포지션(1/0/-1), 미실현PnL, 보유시간(정규화)] 리스트
                          - 보유시간: (current_index - entry_index) / max_steps (0~1 사이)
                          None이면 [0.0, 0.0, 0.0]으로 처리
        
        Returns:
            (obs_seq, obs_info): 튜플
                - obs_seq: (1, 20, 7) 텐서 - 7개 시계열 피처
                - obs_info: (1, 13) 텐서 - 전략 점수(10) + 포지션 정보(3)
        """
        try:
            # 1. 원본 데이터 수집 (마지막 20봉)
            candles = self.collector.get_candles('ETH', count=20)
            if candles is None or len(candles) < 20:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: 20개)")
                return None
            
            close = candles['close'].values.astype(np.float32)
            
            # 2. [해결] 7개 시계열 피처 생성 (차원: 20x7)
            # [최적화] Volume과 Trades에 로그 변환 적용 (거래량 폭발 구간의 극단적 차이 완화)
            volume_log = np.log1p(candles['volume'].values.astype(np.float32))  # log1p = log(1+x)
            trades_log = np.log1p(candles['trades'].values.astype(np.float32))
            
            seq_features = np.column_stack([
                (candles['open'].values - close) / (close + 1e-8),  # f1: Open (close 대비)
                (candles['high'].values - close) / (close + 1e-8),  # f2: High (close 대비)
                (candles['low'].values - close) / (close + 1e-8),   # f3: Low (close 대비)
                np.diff(np.log(close + 1e-8), prepend=np.log(close[0] + 1e-8)),  # f4: Log_Return
                volume_log,  # f5: Volume (로그 변환 후 Z-Score)
                trades_log,   # f6: Trades (로그 변환 후 Z-Score)
                candles['taker_buy_base'].values / (candles['volume'].values + 1e-8)  # f7: Taker_Ratio
            ])
            
            # 3. 전처리 (7개 차원과 정확히 일치해야 함)
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
            
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)  # (1, 20, 7)
            
            # 4. 기술적 전략 Score 수집 (LONG/SHORT 부호 인코딩 반영)
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
            
            # 5. [해결] 10개 전략 점수 + 3개 포지션 정보 = 13차원
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
        """상태 차원 반환 (7개 시계열 피처)"""
        return 7  # 7개 핵심 시계열 피처
