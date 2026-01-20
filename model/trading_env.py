"""
강화학습 트레이딩 환경
기존 전략 Score와 시장 데이터를 결합하는 환경 인터페이스
원시 데이터 보존 + Z-Score 정규화
변동성 기반 보상 시스템 (Risk-Adjusted Reward)
"""
import numpy as np
import torch
import logging
from collections import deque
from .preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """트레이딩 환경: 상태 관측 및 변동성 기반 보상 계산"""
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
        
        # [추가] 최근 pnl_change 내역을 저장하여 변동성 계산 (최근 100스텝)
        self.pnl_change_history = deque(maxlen=100)

    def get_observation(self, position_info=None):
        """
        현재 상태 관측 (8개 핵심 시계열 피처 + Z-Score 정규화 + 포지션 정보)
        
        Args:
            position_info: [포지션(1/0/-1), 미실현PnL, 보유시간(정규화)] 리스트
                          - 보유시간: (current_index - entry_index) / max_steps (0~1 사이)
                          None이면 [0.0, 0.0, 0.0]으로 처리
        
        Returns:
            (obs_seq, obs_info): 튜플
                - obs_seq: (1, 20, 8) 텐서 - 8개 시계열 피처 (VWAP 이격도 포함)
                - obs_info: (1, 13) 텐서 - 전략 점수(10) + 포지션 정보(3)
        """
        try:
            # 1. 원본 데이터 수집 (마지막 20봉)
            candles = self.collector.get_candles('ETH', count=20)
            if candles is None or len(candles) < 20:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: 20개)")
                return None
            
            close = candles['close'].values.astype(np.float32)
            high = candles['high'].values.astype(np.float32)
            low = candles['low'].values.astype(np.float32)
            volume = candles['volume'].values.astype(np.float32)
            
            # [추가] VWAP 계산 (현재 윈도우 20개 기준 Rolling VWAP)
            # 공식: Sum(Price * Volume) / Sum(Volume)
            tp = (high + low + close) / 3  # Typical Price
            vp = tp * volume
            # np.cumsum을 사용하여 윈도우 내에서의 누적 VWAP 흐름을 생성
            cumulative_vp = np.cumsum(vp)
            cumulative_vol = np.cumsum(volume)
            # 거래량이 0인 구간이 길어지면 문제가 될 수 있으므로 안전 장치 추가
            vwap = cumulative_vp / (cumulative_vol + 1e-8)
            
            # VWAP NaN 체크 (거래량이 모두 0인 경우 대비)
            if np.isnan(vwap).any() or np.isinf(vwap).any():
                logger.warning("VWAP 계산 중 NaN/Inf 발생, close 값으로 대체")
                vwap = np.where(np.isnan(vwap) | np.isinf(vwap), close, vwap)
            
            # 2. 8개 시계열 피처 생성 (차원: 20x8)
            # [최적화] Volume과 Trades에 로그 변환 적용 (거래량 폭발 구간의 극단적 차이 완화)
            # log1p는 음수 입력을 방지하기 위해 사용 (log(1+x))
            volume_log = np.log1p(np.maximum(volume, 0))  # 음수 방지
            trades_raw = candles['trades'].values.astype(np.float32) if 'trades' in candles.columns else np.zeros(20, dtype=np.float32)
            trades_log = np.log1p(np.maximum(trades_raw, 0))  # 음수 방지
            
            seq_features = np.column_stack([
                (candles['open'].values - close) / (close + 1e-8),  # f1: Open (close 대비)
                (high - close) / (close + 1e-8),                    # f2: High (close 대비)
                (low - close) / (close + 1e-8),                     # f3: Low (close 대비)
                np.diff(np.log(close + 1e-8), prepend=np.log(close[0] + 1e-8)),  # f4: Log_Return
                volume_log,                                         # f5: Volume (로그 변환 후 Z-Score)
                trades_log,                                         # f6: Trades (로그 변환 후 Z-Score)
                candles['taker_buy_base'].values / (volume + 1e-8), # f7: Taker_Ratio
                (close - vwap) / (vwap + 1e-8)                      # [NEW] f8: VWAP Deviation (이격도)
            ])
            
            # 3. 전처리 (8개 차원과 정확히 일치해야 함)
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
            
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)  # (1, 20, 8)
            
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
            
            # [긴급 점검] NaN/Inf 체크
            if torch.isnan(obs_seq).any() or torch.isinf(obs_seq).any():
                nan_count = torch.isnan(obs_seq).sum().item()
                inf_count = torch.isinf(obs_seq).sum().item()
                logger.error("🚨 시계열 데이터에 NaN 또는 Inf 발생!")
                logger.error(f"   NaN 개수: {nan_count}, Inf 개수: {inf_count}")
                logger.error(f"   시계열 데이터 샘플: {obs_seq[0, :5, :] if obs_seq.shape[0] > 0 else 'N/A'}")
                return None  # 학습 방지
            
            if torch.isnan(obs_info_tensor).any() or torch.isinf(obs_info_tensor).any():
                nan_count = torch.isnan(obs_info_tensor).sum().item()
                inf_count = torch.isinf(obs_info_tensor).sum().item()
                logger.error("🚨 정보 데이터에 NaN 또는 Inf 발생!")
                logger.error(f"   NaN 개수: {nan_count}, Inf 개수: {inf_count}")
                logger.error(f"   정보 데이터: {obs_info_tensor}")
                return None  # 학습 방지
            
            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """
        보상 함수 개선: 초기 학습 유도를 위한 선형적이고 대칭적인 보상 구조
        
        Args:
            pnl: 손익 (수익률)
            trade_done: 거래 완료 여부
            holding_time: 보유 시간 (분)
            pnl_change: 이전 스텝 대비 수익률의 변화
        Returns:
            reward: 보상값 (클리핑: -10 ~ +10)
        """
        # 보상 스케일링 팩터 (100~300 추천)
        # 코인 시장의 높은 변동성을 고려하여 적절한 수준으로 조정
        scaling_factor = 300  # 기존 1000에서 하향 조정
        
        # 1. 기본 보상 (수익률 변화량)
        # 배율을 낮춰서 신경망이 '미세한 차이'를 학습하게 유도
        reward = pnl_change * scaling_factor
        
        # 2. 거래 완료 시 보상 (Trade Done)
        if trade_done:
            # [핵심 변경] 비선형(제곱) 제거 -> 선형(Linear) 보상으로 변경
            # 손실 페널티를 수익 보상과 1:1 대칭으로 맞춤 (Risk:Reward = 1:1)
            step_reward = pnl * scaling_factor
            
            # 수수료 페널티 완화 (-0.05 -> -0.01)
            # 너무 높으면 진입 자체를 꺼리게 됨
            reward += step_reward - 0.01
            
            # [추가] 큰 수익에 대한 추가 인센티브 (잭팟 보상)
            if pnl > 0.01:  # 1% 이상 수익 시
                reward += 1.0
        
        # 3. 시간 페널티 (유지)
        # 너무 오래 들고 있는 것을 방지하기 위해 아주 작게 유지
        reward -= 0.001
        
        # 4. [중요] 무포지션 기회비용 페널티 제거
        # 초기에는 "관망"도 훌륭한 전략임을 인정해야 함. 억지로 진입시키면 손실만 커짐.
        # if not trade_done and holding_time == 0:
        #    reward -= 0.005 

        # 5. 보상 클리핑 (Reward Clipping)
        # [수정] 보상 클리핑 범위 확장 (-10, 10 -> -100, 100)
        # 이제 대박 수익을 내면 확실하게 큰 보상을 줍니다.
        reward = np.clip(reward, -100, 100)

        return reward

    def get_state_dim(self):
        """상태 차원 반환 (8개 시계열 피처)"""
        return 8  # 8개 핵심 시계열 피처 (VWAP 이격도 포함)
