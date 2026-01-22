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
        self.scaler_feature_order = None  # 스케일러 학습 시 사용된 피처 순서 (차원 불일치 방지)
        
        # [추가] 최근 pnl_change 내역을 저장하여 변동성 계산 (최근 100스텝)
        self.pnl_change_history = deque(maxlen=100)
        
        # [추가] 포지션 진입 인덱스 추적 (보상 계산용)
        self.entry_index = None

    def get_observation(self, position_info=None):
        try:
            current_idx = self.collector.current_index
            
            if self.selected_features and len(self.selected_features) > 0:
                seq_len = self.lookback
                start_idx = current_idx - seq_len
                
                if start_idx < 0 or current_idx > len(self.collector.eth_data):
                    return None
                
                feature_data = self.collector.eth_data.iloc[start_idx:current_idx]
                
                strat_cols = [f for f in self.selected_features if f.startswith('strat_') and f in feature_data.columns]
                tech_cols_in_data = [f for f in self.selected_features if not f.startswith('strat_') and f in feature_data.columns]
                
                # 1. 기술적 지표 처리 (차원 불일치 수정)
                if self.scaler_fitted and self.scaler_feature_order is not None:
                    # [수정] 전체 크기의 0 배열 생성
                    full_tech_data = np.zeros((len(feature_data), len(self.scaler_feature_order)), dtype=np.float32)
                    
                    # 존재하는 피처만 해당 인덱스에 매핑
                    for i, feat in enumerate(self.scaler_feature_order):
                        if feat in feature_data.columns:
                            full_tech_data[:, i] = feature_data[feat].values
                    
                    tech_data = full_tech_data
                    
                    # 스케일러 순서대로 tech_cols 업데이트 (후속 결합을 위해)
                    tech_cols = self.scaler_feature_order
                else:
                    tech_cols = tech_cols_in_data
                    tech_data = feature_data[tech_cols].values.astype(np.float32) if tech_cols else np.empty((seq_len, 0), dtype=np.float32)

                # 정규화
                if np.isnan(tech_data).any() or np.isinf(tech_data).any():
                    tech_data = np.nan_to_num(tech_data, nan=0.0, posinf=0.0, neginf=0.0)
                
                tech_data = self.preprocessor.transform(tech_data)
                
                # 2. 전략 점수 처리
                if strat_cols:
                    strat_data = feature_data[strat_cols].values.astype(np.float32)
                    strat_data = np.nan_to_num(strat_data, nan=0.0)
                else:
                    strat_data = np.empty((seq_len, 0), dtype=np.float32)
                
                # 3. 결합 (순서 중요)
                final_seq = []
                for f in self.selected_features:
                    if f in tech_cols:  # tech_cols는 이제 scaler_feature_order와 같음 (fitted일 때)
                        col_idx = tech_cols.index(f)
                        final_seq.append(tech_data[:, col_idx:col_idx+1])
                    elif f in strat_cols:
                        col_idx = strat_cols.index(f)
                        final_seq.append(strat_data[:, col_idx:col_idx+1])
                
                if not final_seq: return None
                
                obs_data = np.hstack(final_seq)
                obs_seq = torch.FloatTensor(obs_data).unsqueeze(0)
                
            else:
                return self._get_observation_fallback(position_info)
            
            if position_info is None: position_info = [0.0, 0.0, 0.0]
            obs_info_tensor = torch.FloatTensor(position_info).unsqueeze(0)
            
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
        [수정] 비선형 시간 페널티 적용
        """
        reward = 0.0

        # 1. 포지션 유지 비용 (비선형 적용)
        # 오래 들고 있을수록 페널티가 기하급수적으로 증가 (존버 방지)
        # 초기에는 부담 없게, 나중엔 아프게
        if self.entry_index is not None and self.collector.current_index > self.entry_index:
            steps_held = self.collector.current_index - self.entry_index
            # 100스텝(300분)까지는 완만, 그 이후 급증
            time_penalty = 0.0005 * (1 + (steps_held / 200.0) ** 2)
            reward -= time_penalty
        elif holding_time > 0:
            reward -= 0.0025  # fallback

        # 2. 청산 시점 보상
        if trade_done:
            realized_pnl = pnl - 0.0005 

            if realized_pnl > 0:
                reward += realized_pnl * 100.0 
                if realized_pnl > 0.005:
                    reward += 1.0
            else:
                reward += realized_pnl * 120.0
                reward -= 0.3

        reward = np.clip(reward, -10, 10)
        return reward

    def get_state_dim(self):
        """상태 차원 반환"""
        if self.selected_features and len(self.selected_features) > 0:
            return len(self.selected_features), 3  # (seq_dim, info_dim)
        else:
            return 8, 3  # 기본 8개 피처
