"""
데이터 전처리 모듈
원시 신호 보존 + 전역 Z-Score 정규화
"""
import numpy as np
import logging
import pickle
import os

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """개선된 데이터 전처리: 원시 신호 보존 + 전역 Z-Score 정규화"""
    def __init__(self):
        self.mean = None
        self.std = None
        # xLSTM은 정규분포 형태의 입력을 받을 때 기울기 소실/폭발이 가장 적음
        logger.info("Z-Score 정규화 모드 활성화 (Wavelet 제거됨)")

    def fit(self, data):
        """전체 학습 데이터셋의 통계량 계산 (한 번만 실행)
        
        Args:
            data: (seq_len, feature_dim) 형태의 배열
        """
        data = np.array(data, dtype=np.float32)
        if data.size == 0:
            logger.warning("빈 데이터로 fit 시도")
            return
        
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        # 0으로 나누기 방지
        self.std[self.std == 0] = 1.0
        logger.info(f"스케일러 학습 완료: Mean shape {self.mean.shape}, Std shape {self.std.shape}")

    def transform(self, data):
        """학습된 통계량으로 변환 (맥락 보존)
        
        Args:
            data: (seq_len, feature_dim) 형태의 배열
        Returns:
            정규화된 데이터 (Z-Score)
        """
        data = np.array(data, dtype=np.float32)
        if data.size == 0:
            return data
        
        if self.mean is None or self.std is None:
            # fit이 안 되었다면 현재 데이터로 임시 변환
            logger.warning("스케일러가 fit되지 않았습니다. 현재 데이터로 임시 변환합니다.")
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            std[std == 0] = 1.0
            return (data - mean) / std
        
        return (data - self.mean) / self.std

    def log_return(self, data):
        """가격 데이터를 로그 수익률로 변환 (정상성 확보를 위해 추천)
        
        Args:
            data: 1D 배열 (가격 시계열)
        Returns:
            로그 수익률 배열
        """
        data = np.array(data, dtype=np.float32)
        if len(data) < 2:
            return np.zeros_like(data)
        
        # 로그 수익률: log(price_t / price_{t-1}) = log(price_t) - log(price_{t-1})
        log_prices = np.log(data + 1e-8)  # 0 방지
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        return log_returns
    
    def save_scaler(self, path='model/scaler.pkl'):
        """스케일러 저장
        
        Args:
            path: 저장 경로 (기본값: model/scaler.pkl)
        """
        if self.mean is None or self.std is None:
            logger.warning("스케일러가 학습되지 않았습니다. 저장할 수 없습니다.")
            return
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
        logger.info(f"💾 스케일러 저장 완료: {path}")
    
    def load_scaler(self, path='model/scaler.pkl'):
        """스케일러 로드
        
        Args:
            path: 로드 경로 (기본값: model/scaler.pkl)
        Returns:
            bool: 로드 성공 여부
        """
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.mean = data['mean']
                    self.std = data['std']
                logger.info(f"✅ 스케일러 로드 완료: {path}")
                return True
            except Exception as e:
                logger.error(f"스케일러 로드 실패: {e}")
                return False
        return False
