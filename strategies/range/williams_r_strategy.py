"""
Williams %R 전략
스토캐스틱보다 반응이 빠른 초단기 모멘텀 지표
"""
import logging
import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class WilliamsRStrategy:
    def __init__(self):
        self.name = "Williams %R"
        self.period = 14
        self.upper_threshold = -20  # -20 하향 돌파 시 숏 (과매수 해소)
        self.lower_threshold = -80  # -80 상향 돌파 시 롱 (과매도 해소)
    
    def calculate_williams_r(self, data, period=14):
        """Williams %R 자체 계산"""
        try:
            # 최근 N기간 최고가/최저가
            highest_high = data['high'].rolling(window=period).max()
            lowest_low = data['low'].rolling(window=period).min()
            
            # %R 계산: (Highest High - Close) / (Highest High - Lowest Low) * -100
            # 분모가 0일 경우(고가=저가) 처리
            denom = highest_high - lowest_low
            wr = ((highest_high - data['close']) / (denom + 1e-8)) * -100
            
            # Williams %R은 보통 -100 ~ 0 사이 값을 가짐 (inverted scale)
            # 여기서는 일반적인 (High-Close) 공식을 썼는데, 
            # 트레이딩뷰 표준: (Close - Highest High) / (Highest High - Lowest Low) * 100 과 유사하게 맞춤
            # 위 공식은 -0 (최고점) ~ -100 (최저점) 범위
            
            wr = -100 * ((highest_high - data['close']) / (denom + 1e-8))
            
            return wr
        except Exception as e:
            logger.error(f"Williams %R 계산 오류: {e}")
            return None

    def analyze(self, data_collector):
        try:
            df = data_collector.get_candles('ETH', count=50)
            if df is None or len(df) < self.period + 2:
                return None
            
            wr_series = self.calculate_williams_r(df, self.period)
            if wr_series is None:
                return None
            
            curr_wr = float(wr_series.iloc[-1])
            prev_wr = float(wr_series.iloc[-2])
            
            latest = df.iloc[-1]
            entry_price = float(latest['close'])
            signal = None
            
            # LONG 조건: -80 아래에서 위로 탈출
            if prev_wr < self.lower_threshold and curr_wr >= self.lower_threshold:
                signal = 'LONG'
                logger.debug(f"🔍 [Williams %R] 롱 신호 발생 - %R: {curr_wr:.2f} (이전: {prev_wr:.2f})")
                
            # SHORT 조건: -20 위에서 아래로 이탈
            elif prev_wr > self.upper_threshold and curr_wr <= self.upper_threshold:
                signal = 'SHORT'
                logger.debug(f"🔍 [Williams %R] 숏 신호 발생 - %R: {curr_wr:.2f} (이전: {prev_wr:.2f})")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.60,
                    'strategy': self.name,
                    'williams_r': curr_wr
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Williams %R 전략 분석 실패: {e}")
            return None
