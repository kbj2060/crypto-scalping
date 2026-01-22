"""
트레이딩 환경: 상태 관측 및 변동성 기반 보상 계산
"""
import numpy as np
import torch
import logging
from collections import deque
from model.preprocess import DataPreprocessor

logger = logging.getLogger(__name__)


class TradingEnvironment:
    """트레이딩 환경: 상태 관측 및 변동성 기반 보상 계산"""
    def __init__(self, data_collector, strategies, lookback=40, selected_features=None):
        """
        Args:
            data_collector: DataCollector 인스턴스
            strategies: 전략 리스트
            lookback: 충분한 샘플 수 (기본 40)
            selected_features: XGBoost로 선택된 피처 리스트 (None이면 기존 8개 사용)
        """
        self.collector = data_collector
        self.strategies = strategies
        self.num_strategies = len(strategies)
        self.lookback = lookback
        self.selected_features = selected_features  # XGBoost 선택 피처 저장
        
        # 전처리 파이프라인 (Z-Score 정규화)
        self.preprocessor = DataPreprocessor()
        self.scaler_fitted = False  # 스케일러 학습 여부
        
        # [추가] 최근 pnl_change 내역을 저장하여 변동성 계산 (최근 100스텝)
        self.pnl_change_history = deque(maxlen=100)
        
        # [추가] 포지션 진입 인덱스 추적 (보상 계산용)
        self.entry_index = None

    def get_observation(self, position_info=None):
        """
        현재 상태 관측 (XGBoost 선택 피처 또는 기존 8개 피처 + Z-Score 정규화 + 포지션 정보)
        
        Args:
            position_info: [포지션(1/0/-1), 미실현PnL, 보유시간(정규화)] 리스트
                          None이면 [0.0, 0.0, 0.0]으로 처리
        
        Returns:
            (obs_seq, obs_info): 튜플
                - obs_seq: (1, 20, num_features) 텐서 - 선택된 피처 또는 8개 시계열 피처
                - obs_info: (1, 3) 텐서 - 포지션 정보만 (DDQN에서는 전략 점수 제외)
        """
        try:
            current_idx = self.collector.current_index
            
            # 선택된 피처가 있으면 XGBoost 선택 피처 사용
            if self.selected_features and len(self.selected_features) > 0:
                # [핵심] 선택된 피처만 슬라이싱
                # [수정] 하드코딩된 20을 self.lookback으로 변경
                seq_len = self.lookback
                start_idx = current_idx - seq_len
                
                if start_idx < 0 or current_idx > len(self.collector.eth_data):
                    logger.warning(f"인덱스 범위 초과: start={start_idx}, current={current_idx}, total={len(self.collector.eth_data)}")
                    return None
                
                # 데이터프레임에서 선택된 컬럼만 추출
                # collector.eth_data에는 이미 모든 피처가 계산되어 있다고 가정
                feature_data = self.collector.eth_data.iloc[start_idx:current_idx]
                
                # [핵심 수정] 피처를 성격에 따라 분리
                # strat_로 시작하는 컬럼(전략)과 그 외(기술지표)로 구분
                strat_cols = [f for f in self.selected_features if f.startswith('strat_') and f in feature_data.columns]
                tech_cols = [f for f in self.selected_features if not f.startswith('strat_') and f in feature_data.columns]
                
                if len(tech_cols) == 0 and len(strat_cols) == 0:
                    logger.warning("선택된 피처가 데이터에 없습니다. 기존 방식으로 전환합니다.")
                    return self._get_observation_fallback(position_info)
                
                # 1. 기술적 지표 처리 (정규화 O)
                if tech_cols:
                    tech_data = feature_data[tech_cols].values.astype(np.float32)
                    # NaN 체크 및 처리
                    if np.isnan(tech_data).any() or np.isinf(tech_data).any():
                        tech_data = np.nan_to_num(tech_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if not self.scaler_fitted:
                        logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
                    
                    # 기술 지표만 정규화
                    tech_data = self.preprocessor.transform(tech_data)
                else:
                    tech_data = np.empty((seq_len, 0), dtype=np.float32)
                
                # 2. 전략 점수 처리 (정규화 X - 원본 유지)
                if strat_cols:
                    strat_data = feature_data[strat_cols].values.astype(np.float32)
                    # 전략 점수는 NaN을 0으로만 채움 (정규화 안 함)
                    strat_data = np.nan_to_num(strat_data, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    strat_data = np.empty((seq_len, 0), dtype=np.float32)
                
                # 3. 다시 결합 (순서 중요: selected_features 순서대로 재배열)
                # selected_features 순서대로 하나씩 조립
                final_seq = []
                for f in self.selected_features:
                    if f in tech_cols:
                        col_idx = tech_cols.index(f)
                        final_seq.append(tech_data[:, col_idx:col_idx+1])
                    elif f in strat_cols:
                        col_idx = strat_cols.index(f)
                        final_seq.append(strat_data[:, col_idx:col_idx+1])
                
                if len(final_seq) == 0:
                    logger.warning("최종 피처가 없습니다.")
                    return None
                
                obs_data = np.hstack(final_seq)
                
                # 4. 텐서 변환
                obs_seq = torch.FloatTensor(obs_data).unsqueeze(0)  # (1, lookback, num_features)
                
            else:
                # 기존 8개 피처 사용 (호환성)
                return self._get_observation_fallback(position_info)
            
            # 5. Info 데이터 (포지션 정보만 사용, 전략 점수는 DDQN에서 제외)
            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            obs_info_tensor = torch.FloatTensor(position_info).unsqueeze(0)  # (1, 3)
            
            # [긴급 점검] NaN/Inf 체크
            if torch.isnan(obs_seq).any() or torch.isinf(obs_seq).any():
                nan_count = torch.isnan(obs_seq).sum().item()
                inf_count = torch.isinf(obs_seq).sum().item()
                logger.error("🚨 시계열 데이터에 NaN 또는 Inf 발생!")
                logger.error(f"   NaN 개수: {nan_count}, Inf 개수: {inf_count}")
                return None
            
            if torch.isnan(obs_info_tensor).any() or torch.isinf(obs_info_tensor).any():
                logger.error("🚨 정보 데이터에 NaN 또는 Inf 발생!")
                return None
            
            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 생성 실패: {e}", exc_info=True)
            return None
    
    def _get_observation_fallback(self, position_info=None):
        """기존 8개 피처 방식 (호환성 유지)"""
        try:
            # 1. 원본 데이터 수집 (마지막 lookback봉)
            candles = self.collector.get_candles('ETH', count=self.lookback)
            if candles is None or len(candles) < self.lookback:
                logger.warning(f"데이터 부족: {len(candles) if candles is not None else 0}개 (필요: {self.lookback}개)")
                return None
            
            close = candles['close'].values.astype(np.float32)
            high = candles['high'].values.astype(np.float32)
            low = candles['low'].values.astype(np.float32)
            volume = candles['volume'].values.astype(np.float32)
            
            # [추가] VWAP 계산 (현재 윈도우 20개 기준 Rolling VWAP)
            tp = (high + low + close) / 3  # Typical Price
            vp = tp * volume
            cumulative_vp = np.cumsum(vp)
            cumulative_vol = np.cumsum(volume)
            vwap = cumulative_vp / (cumulative_vol + 1e-8)
            
            # VWAP NaN 체크
            if np.isnan(vwap).any() or np.isinf(vwap).any():
                logger.warning("VWAP 계산 중 NaN/Inf 발생, close 값으로 대체")
                vwap = np.where(np.isnan(vwap) | np.isinf(vwap), close, vwap)
            
            # 2. 8개 시계열 피처 생성
            volume_log = np.log1p(np.maximum(volume, 0))
            trades_raw = candles['trades'].values.astype(np.float32) if 'trades' in candles.columns else np.zeros(self.lookback, dtype=np.float32)
            trades_log = np.log1p(np.maximum(trades_raw, 0))
            
            seq_features = np.column_stack([
                (candles['open'].values - close) / (close + 1e-8),
                (high - close) / (close + 1e-8),
                (low - close) / (close + 1e-8),
                np.diff(np.log(close + 1e-8), prepend=np.log(close[0] + 1e-8)),
                volume_log,
                trades_log,
                candles['taker_buy_base'].values / (volume + 1e-8) if 'taker_buy_base' in candles.columns else np.zeros(self.lookback, dtype=np.float32),
                (close - vwap) / (vwap + 1e-8)
            ])
            
            # 3. 전처리
            if not self.scaler_fitted:
                logger.warning("스케일러가 fit되지 않았습니다. transform만 수행합니다.")
            
            normalized_seq = self.preprocessor.transform(seq_features)
            obs_seq = torch.FloatTensor(normalized_seq).unsqueeze(0)  # (1, lookback, 8)
            
            # 4. Info 데이터
            if position_info is None:
                position_info = [0.0, 0.0, 0.0]
            obs_info_tensor = torch.FloatTensor(position_info).unsqueeze(0)  # (1, 3)
            
            return (obs_seq, obs_info_tensor)
            
        except Exception as e:
            logger.error(f"관측 생성 실패 (폴백): {e}", exc_info=True)
            return None

    def calculate_reward(self, pnl, trade_done, holding_time=0, pnl_change=0):
        """
        [최종 수정] Reality-Based Reward Function
        - 평가손익(Change) 보상 제거: 희망고문 방지
        - 결과 중심 보상: 오직 청산 시점의 PnL로만 평가
        - 손실에 더 큰 고통 부여 (손실 회피 성향 주입)
        """
        reward = 0.0
        
        # 1. 평가손익 변화 보상 삭제 (또는 극소화)
        # 이 부분이 '착시'의 주범이므로 과감히 0으로 만듭니다.
        # reward += pnl_change * 0.0 
        
        # 대신, 포지션을 잡고 있는 동안의 "불안감(리스크)"을 시간 페널티로 부여
        # entry_index가 있으면 사용하고, 없으면 holding_time 사용
        if self.entry_index is not None and self.collector.current_index > self.entry_index:
            # 포지션 보유 중
            reward -= 0.0005  # 버티는 것 자체가 비용임
        elif holding_time > 0:
            # holding_time 파라미터로 대체 (호환성)
            reward -= 0.0005

        # 2. 거래 종료(청산) 시 보상 - 여기가 핵심
        if trade_done:
            # 수수료 차감 (실전 반영)
            realized_pnl = pnl - 0.0005 
            
            if realized_pnl > 0:
                # [익절] 선형 보상 (제곱 X)
                # 수익률 1% = +1.0점
                reward += realized_pnl * 100.0
                
                # 보너스: 확실한 익절에 대한 인센티브
                if realized_pnl > 0.005:  # 0.5% 이상 수익 시
                    reward += 1.0
            else:
                # [손실] 페널티 강화 (익절보다 1.5배 더 아프게)
                # 손실 1% = -1.5점
                reward += realized_pnl * 150.0 
                
                # 추가 벌점: 뇌동매매 방지
                reward -= 0.5

        # 3. 클리핑 (학습 안정성)
        return np.clip(reward, -10, 10)

    def get_state_dim(self):
        """상태 차원 반환"""
        if self.selected_features and len(self.selected_features) > 0:
            return len(self.selected_features), 3  # (seq_dim, info_dim)
        else:
            return 8, 3  # 기본 8개 피처
