"""
데이터 전처리 모듈
"""
import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def add_volatility_feature(df):
    """
    보조 변동성 지표 (TD3 Info용)
    """
    if 'close' not in df.columns:
        return df
    df['volatility_20tick'] = df['close'].pct_change().rolling(window=20).std().fillna(0.0)
    return df


class DataPreprocessor:
    """
    Rolling Normalization Helper
    """
    def __init__(self):
        self.epsilon = 1e-8

    def transform(self, data):
        data = np.array(data, dtype=np.float32)
        if data.size == 0:
            return data

        mean = np.mean(data, axis=0)
        std = np.std(data, axis=0)
        std[std < self.epsilon] = 1.0

        return (data - mean) / std
