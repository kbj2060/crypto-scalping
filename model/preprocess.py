"""
데이터 전처리 모듈 (수정됨)
Global Z-Score -> Rolling (Instance) Normalization 변경
"""
import numpy as np
import logging
import pickle

logger = logging.getLogger(__name__)

class DataPreprocessor:
    """
    Rolling Normalization (Instance Normalization)
    입력된 윈도우(Lookback) 내에서 즉석으로 정규화를 수행하여 
    시계열의 비정상성(Non-stationarity)을 극복하고 로컬 패턴에 집중함.
    """
    def __init__(self):
        self.epsilon = 1e-8
        logger.info("Rolling (Instance) Normalization 모드 활성화")

    def fit(self, data):
        """Rolling 방식에서는 전역 학습이 불필요하지만 호환성을 위해 남겨둠"""
        pass

    def transform(self, data):
        """
        입력 데이터(윈도우) 자체의 통계량을 사용하여 정규화
        Args:
            data: (seq_len, feature_dim) 형태의 배열
        Returns:
            자체 정규화된 데이터
        """
        data = np.array(data, dtype=np.float32)
        if data.size == 0:
            return data
        
        # [핵심] 현재 윈도우의 평균과 표준편차 계산
        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        
        # 표준편차가 0인 경우 1로 대체하여 나눗셈 오류 방지
        std[std < self.epsilon] = 1.0
        
        # 정규화 (Z-Score)
        normalized_data = (data - mean) / std
        
        return normalized_data

    def log_return(self, data):
        data = np.array(data, dtype=np.float32)
        if len(data) < 2:
            return np.zeros_like(data)
        log_prices = np.log(data + 1e-8)
        log_returns = np.diff(log_prices, prepend=log_prices[0])
        return log_returns

    def save(self, filepath):
        """Rolling 방식은 저장할 상태가 없으나 호환성 유지"""
        pass

    def load(self, filepath):
        """Rolling 방식은 로드할 상태가 없으나 호환성 유지"""
        return True
    
    def save_scaler(self, filepath, feature_names=None):
        pass