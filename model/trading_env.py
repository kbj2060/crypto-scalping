"""
강화학습 트레이딩 환경
기존 전략 Score와 시장 데이터를 결합하는 환경 인터페이스
원시 데이터 보존 + Z-Score 정규화
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
    """트레이딩 환경: 상태 관측 및 보상 계산"""
    def __init__(self, data_collector, strategies, lookback=None, min_holding_time=None):
        """
        Args:
            data_collector: DataCollector 인스턴스
            strategies: 전략 리스트
            lookback: 충분한 샘플 수 (None이면 config.LOOKBACK 사용)
            min_holding_time: 최소 보유 캔들 수 (None이면 config.MIN_HOLDING_TIME 사용)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback if lookback is not None else config.LOOKBACK
        self.min_holding_time = min_holding_time if min_holding_time is not None else config.MIN_HOLDING_TIME
        
        # 전처리 파이프라인 (Z-Score 정규화)
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False  # 스케일러 학습 여부
        
        # 피처 엔지니어링 모듈 (인스턴스는 매번 새로 생성)
        # FeatureEngineer와 MTFProcessor는 데이터를 받아서 생성하므로 여기서는 초기화하지 않음

    def get_action_mask(self, current_index, entry_index, current_position):
        """
        [추가] 유효한 행동 마스크 생성
        최소 보유 시간을 채우지 못했으면 포지션 청산/스위칭 금지
        
        Args:
            current_index: 현재 인덱스
            entry_index: 진입 인덱스 (None이면 포지션 없음)
            current_position: 현재 포지션 ('LONG', 'SHORT', None)
        
        Returns:
            torch.FloatTensor: [HOLD, LONG, SHORT] (1=가능, 0=불가능)
        """
        mask = [1.0, 1.0, 1.0]  # 기본적으로 모두 허용
        
        # 포지션이 있고 진입 시점이 기록되어 있다면
        if current_position is not None and entry_index is not None and current_index is not None:
            holding_time = current_index - entry_index
            
            # 최소 보유 시간 미만이면 강제 HOLD (청산/스위칭 불가)
            if holding_time < self.min_holding_time:
                # 포지션 변경 금지: LONG(1) 불가, SHORT(2) 불가 -> HOLD(0)만 가능
                mask = [1.0, 0.0, 0.0]

        return torch.FloatTensor(mask)

    def get_observation(self, position_info=None, current_index=None, entry_index=None, current_position=None):
        """
        현재 상태 관측 (29개 고급 시계열 피처 + Z-Score 정규화 + 포지션 정보)
        
        Args:
            position_info: [포지션(1/0/-1), 미실현PnL, 보유시간(정규화)] 리스트
                          - 보유시간: (current_index - entry_index) / max_steps (0~1 사이)
                          None이면 [0.0, 0.0, 0.0]으로 처리
        
        Returns:
            (obs_seq, obs_info): 튜플
                - obs_seq: (1, 40, 29) 텐서 - 29개 시계열 피처
                - obs_info: (1, num_strategies + 3) 텐서 - 전략 점수 + 포지션 정보(3)
        """
        try:
            # [수정 1] DataCollector의 get_candles 대신 내부 데이터 직접 접근 시도
            # 이유: get_candles가 OHLCV만 남기고 나머지 컬럼을 버리는 경우 방지
            candles = None
            
            # collector가 eth_data 속성을 가지고 있고, DataFrame 형태라면 직접 슬라이싱
            if hasattr(self.collector, 'eth_data') and isinstance(self.collector.eth_data, pd.DataFrame):
                # 인덱스가 주어지지 않으면 collector 내부 인덱스 사용
                curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
                
                if curr_idx is not None and curr_idx >= self.lookback:
                    # 미리 계산된 피처가 포함된 원본 데이터에서 직접 슬라이싱
                    candles = self.collector.eth_data.iloc[curr_idx - self.lookback : curr_idx].copy()
            
            # 직접 접근 실패 시 기존 방식 사용 (Fallback)
            if candles is None or len(candles) < self.lookback:
                candles = self.collector.get_candles('ETH', count=self.lookback + 60)
            
            if candles is None or len(candles) < self.lookback:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: {self.lookback}개)")
                return None
            
            # [수정 2] 피처 존재 여부 확인 로직 (이미 있으면 계산 생략)
            # train_ppo.py에서 _precalculate_features를 돌렸다면 여기에 컬럼이 있어야 함
            required_feature = 'rsi_1h'
            
            if required_feature in candles.columns:
                # ✅ 이미 피처가 있으므로 그대로 사용 (로그 안 뜸, 속도 빠름)
                df = candles
            else:
                # ❌ 피처가 없어서 다시 계산 (여기가 실행되면 로그가 뜸)
                # 라이브 트레이딩이나 데이터가 없을 때만 실행되어야 함
                
                # 인덱스가 DatetimeIndex인지 확인 및 변환
                if not isinstance(candles.index, pd.DatetimeIndex):
                    if 'timestamp' in candles.columns:
                        candles.index = pd.to_datetime(candles['timestamp'])
                    else:
                        # 인덱스가 없으면 현재 시간 기준으로 생성
                        candles.index = pd.date_range(end=pd.Timestamp.now(), periods=len(candles), freq='3min')
                
                # BTC 데이터도 가져오기 (상관관계 피처용)
                btc_candles = None
                if hasattr(self.collector, 'btc_data') and isinstance(self.collector.btc_data, pd.DataFrame):
                    current_idx = getattr(self.collector, 'current_index', None)
                    if current_idx is not None and current_idx >= len(candles):
                        btc_candles = self.collector.btc_data.iloc[current_idx - len(candles) : current_idx].copy()
                
                if btc_candles is None:
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
                
                # 로그 레벨을 잠시 낮춰서 스팸 방지 (선택 사항)
                logging.getLogger('model.feature_engineering').setLevel(logging.WARNING)
                logging.getLogger('model.mtf_processor').setLevel(logging.WARNING)
                
                # (1) 기본 기술적 지표 (25개)
                feature_engineer = FeatureEngineer(candles, btc_candles)
                df = feature_engineer.generate_features()
                
                if df is None:
                    logger.warning("피처 생성 실패")
                    # 로그 레벨 복구
                    logging.getLogger('model.feature_engineering').setLevel(logging.INFO)
                    logging.getLogger('model.mtf_processor').setLevel(logging.INFO)
                    return None
                
                # (2) 멀티 타임프레임 지표 (4개)
                mtf_processor = MTFProcessor(df)
                df = mtf_processor.add_mtf_features()
                
                # 로그 레벨 복구
                logging.getLogger('model.feature_engineering').setLevel(logging.INFO)
                logging.getLogger('model.mtf_processor').setLevel(logging.INFO)
            
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
            
            # 6. 전략 점수 + 포지션 정보 결합 (동적 차원)
            # 전략 점수 개수는 self.strategies의 길이에 따라 결정됨
            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            
            obs_info = np.concatenate([strategy_scores, position_info], dtype=np.float32)
            obs_info_tensor = torch.FloatTensor(obs_info).unsqueeze(0)  # (1, num_strategies + 3)
            
            # [추가] Action Mask 생성
            curr_idx = current_index if current_index is not None else getattr(self.collector, 'current_index', None)
            action_mask = self.get_action_mask(curr_idx, entry_index, current_position)
            
            return (obs_seq, obs_info_tensor, action_mask)
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """
        개선된 보상 함수 (Net PnL 기반 정교화)
        - 큰 수익에 대한 인센티브 유지
        - 하지만 제곱이 아닌 sqrt로 완화
        - 보상 배율: config에서 설정 가능
        - Net PnL 도입: 거래 비용을 반영한 순수익 기반 보상
        """
        reward = 0.0
        
        # 1. 평가 수익 변화량 (config에서 설정)
        reward = pnl_change * config.REWARD_MULTIPLIER
        
        if trade_done:
            # 실질 거래 비용 반영 (config에서 설정)
            net_pnl = pnl - config.TRANSACTION_COST
            
            if net_pnl > 0:
                # 순수익이 났을 때만 칭찬
                reward += net_pnl * config.REWARD_MULTIPLIER
                reward += np.sqrt(net_pnl * 100) * 0.5
                reward += np.tanh(net_pnl * 100) * 0.5
            else:
                # 손실 페널티 (config에서 설정)
                reward += net_pnl * config.LOSS_PENALTY_MULTIPLIER
            
            # 별도 수수료 차감은 제거 (위에서 net_pnl로 반영했으므로)
        
        # 시간 비용 (config에서 설정)
        reward -= config.TIME_COST
        
        return np.clip(reward, -100, 100)

    def get_state_dim(self):
        """상태 차원 반환 (29개 시계열 피처)"""
        return 29  # 29개 고급 시계열 피처
