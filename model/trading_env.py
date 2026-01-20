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
        보상 계산 (현실화된 보상 체계 + 비선형 보상 + 강화된 손실 페널티)
        
        Args:
            pnl: 손익 (수익률)
            trade_done: 거래 완료 여부
            holding_time: 보유 시간 (분)
            pnl_change: 이전 스텝 대비 수익률의 변화 (새로 추가)
        Returns:
            reward: 보상값
        """
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
                # [수정] 손실 페널티 강화: 손실도 제곱으로 처리하여 큰 하락을 극도로 방어
                # 수익보다 2배 더 아프게 설계하여 '안전'을 우선시하게 유도
                # 예: -0.5% 손실 → -(0.005 * 100)^2 / 5 = -0.5
                #     -2% 손실 → -(0.02 * 100)^2 / 5 = -8.0
                reward -= (abs(pnl) * 100) ** 2 / 5
            
            # [수정] 수수료 페널티 현실화 (잦은 매매 방지)
            reward -= 0.05
        
        # 3. 시간 페널티 완화 (기존 -0.0005 -> -0.0001)
        # 큰 추세를 끝까지 타도록 유도
        reward -= 0.0001
        
        return reward

    def get_state_dim(self):
        """상태 차원 반환 (7개 시계열 피처)"""
        return 7  # 7개 핵심 시계열 피처
