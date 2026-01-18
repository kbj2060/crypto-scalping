"""
1. CVD / 델타 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
import pandas as pd
import numpy as np
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class CVDDeltaStrategy:
    def __init__(self):
        self.name = "CVD Delta"
        self.cvd_ema_period = 21
        self.price_ema_period = 21
        self.delta_smoothing = 5
        self.delta_spike_multiplier = 2.2  # 평균 델타의 2.2배
        self.divergence_lookback = 60  # 30 -> 60 (더 긴 흐름 파악)
        self.divergence_strength = 2  # 5 -> 2 (미세한 변곡점도 포착)
    
    def find_divergence(self, price_data, cvd_data, lookback=60):
        """CVD 다이버전스 탐지 (Lookback 60, Strength 2)"""
        try:
            if len(price_data) < lookback or len(cvd_data) < lookback:
                return None
            
            recent_price = price_data.tail(lookback)
            recent_cvd = cvd_data.tail(lookback)
            
            # 가격 저점/고점 찾기 (Strength 5 = 양쪽 5개씩 확인)
            price_lows = []
            price_highs = []
            cvd_lows = []
            cvd_highs = []
            
            for i in range(self.divergence_strength, len(recent_price) - self.divergence_strength):
                # 가격 저점
                is_low = True
                for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                    if j != i and recent_price.iloc[j]['low'] <= recent_price.iloc[i]['low']:
                        is_low = False
                        break
                if is_low:
                    price_lows.append({'index': i, 'value': recent_price.iloc[i]['low']})
                
                # 가격 고점
                is_high = True
                for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                    if j != i and recent_price.iloc[j]['high'] >= recent_price.iloc[i]['high']:
                        is_high = False
                        break
                if is_high:
                    price_highs.append({'index': i, 'value': recent_price.iloc[i]['high']})
                
                # CVD 저점
                is_cvd_low = True
                cvd_value_i = float(recent_cvd.iloc[i])
                for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                    if j != i:
                        cvd_value_j = float(recent_cvd.iloc[j])
                        if cvd_value_j <= cvd_value_i:
                            is_cvd_low = False
                            break
                if is_cvd_low:
                    cvd_lows.append({'index': i, 'value': cvd_value_i})
                
                # CVD 고점
                is_cvd_high = True
                cvd_value_i = float(recent_cvd.iloc[i])
                for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                    if j != i:
                        cvd_value_j = float(recent_cvd.iloc[j])
                        if cvd_value_j >= cvd_value_i:
                            is_cvd_high = False
                            break
                if is_cvd_high:
                    cvd_highs.append({'index': i, 'value': cvd_value_i})
            
            # Bullish Divergence: 가격 lower-low, CVD higher-low
            if len(price_lows) >= 2 and len(cvd_lows) >= 2:
                price_low1 = price_lows[-2]
                price_low2 = price_lows[-1]
                cvd_low1 = cvd_lows[-2]
                cvd_low2 = cvd_lows[-1]
                
                if (price_low2['value'] < price_low1['value'] and  # 가격 lower-low
                    cvd_low2['value'] > cvd_low1['value']):  # CVD higher-low
                    return 'bullish_divergence'
            
            # Bearish Divergence: 가격 higher-high, CVD lower-high
            if len(price_highs) >= 2 and len(cvd_highs) >= 2:
                price_high1 = price_highs[-2]
                price_high2 = price_highs[-1]
                cvd_high1 = cvd_highs[-2]
                cvd_high2 = cvd_highs[-1]
                
                if (price_high2['value'] > price_high1['value'] and  # 가격 higher-high
                    cvd_high2['value'] < cvd_high1['value']):  # CVD lower-high
                    return 'bearish_divergence'
            
            return None
        except Exception as e:
            logger.error(f"다이버전스 탐지 실패: {e}")
            return None
    
    def analyze(self, data_collector):
        """CVD / 델타 전략 분석 (최적 세팅)"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                return None
            
            # CVD 계산 (EMA 21, 델타 스무딩 5)
            cvd_data = data_collector.calculate_cvd('ETH', lookback=100, ema_period=self.cvd_ema_period, delta_smoothing=self.delta_smoothing)
            if cvd_data is None:
                return None
            
            # Price EMA 21
            price_ema = Indicators.calculate_ema(eth_data, period=self.price_ema_period)
            if price_ema is None:
                return None
            
            # CVD EMA 방향성 필터
            cvd_ema_current = float(cvd_data['cvd_ema'].iloc[-1])
            cvd_ema_prev = float(cvd_data['cvd_ema'].iloc[-2]) if len(cvd_data) >= 2 else None
            price_ema_current = float(price_ema.iloc[-1])
            price_current = float(eth_data.iloc[-1]['close'])
            
            # 최신 델타 (스무딩된 값)
            latest_delta = float(cvd_data['delta_smooth'].iloc[-1])
            delta_mean = abs(float(cvd_data['delta_smooth'].tail(20).mean()))
            delta_spike_threshold = delta_mean * self.delta_spike_multiplier
            
            # CVD 다이버전스 탐지
            price_data = eth_data.tail(self.divergence_lookback)
            cvd_series = cvd_data['cvd_ema']
            divergence = self.find_divergence(price_data, cvd_series, lookback=self.divergence_lookback)
            
            latest = eth_data.iloc[-1]
            prev_candle = eth_data.iloc[-2] if len(eth_data) >= 2 else None
            signal = None
            entry_price = latest['close']
            
            # 롱 조건: CVD-EMA ↑ + 가격 EMA 위 + (다이버전스 또는 델타 스파이크) + 캔들 반전
            if cvd_ema_prev is not None and cvd_ema_current > cvd_ema_prev:  # CVD-EMA 상승
                if price_current > price_ema_current:  # 가격 EMA 위
                    # 다이버전스 또는 델타 스파이크
                    has_divergence = (divergence == 'bullish_divergence')
                    has_delta_spike = latest_delta >= delta_spike_threshold
                    
                    if has_divergence or has_delta_spike:
                        # 다이버전스 확인 시 즉시 진입 (시간적 불일치 해결)
                        # 다이버전스는 이미 반등 시작을 의미하므로, 추가 캔들 반전 조건 제거
                        if has_divergence:
                            # 다이버전스 확인 시 즉시 진입
                            signal = 'LONG'
                            logger.info(f"CVD 롱: CVD-EMA↑, 가격 EMA 위, 다이버전스 확인 즉시 진입")
                        elif has_delta_spike:
                            # 델타 스파이크는 캔들 반전 확인
                            if prev_candle:
                                is_bullish = latest['close'] > latest['open']
                                is_reversal = (latest['close'] > prev_candle['close'] and 
                                             prev_candle['close'] < prev_candle['open'])
                                if is_bullish or is_reversal:
                                    signal = 'LONG'
                                    logger.info(f"CVD 롱: CVD-EMA↑, 가격 EMA 위, 델타 스파이크")
            
            # 숏 조건: CVD-EMA ↓ + 가격 EMA 아래 + (다이버전스 또는 델타 스파이크) + 캔들 반전
            elif cvd_ema_prev is not None and cvd_ema_current < cvd_ema_prev:  # CVD-EMA 하락
                if price_current < price_ema_current:  # 가격 EMA 아래
                    # 다이버전스 또는 델타 스파이크
                    has_divergence = (divergence == 'bearish_divergence')
                    has_delta_spike = latest_delta <= -delta_spike_threshold
                    
                    if has_divergence or has_delta_spike:
                        # 다이버전스 확인 시 즉시 진입 (시간적 불일치 해결)
                        # 다이버전스는 이미 반등 시작을 의미하므로, 추가 캔들 반전 조건 제거
                        if has_divergence:
                            # 다이버전스 확인 시 즉시 진입
                            signal = 'SHORT'
                            logger.info(f"CVD 숏: CVD-EMA↓, 가격 EMA 아래, 다이버전스 확인 즉시 진입")
                        elif has_delta_spike:
                            # 델타 스파이크는 캔들 반전 확인
                            if prev_candle:
                                is_bearish = latest['close'] < latest['open']
                                is_reversal = (latest['close'] < prev_candle['close'] and 
                                              prev_candle['close'] > prev_candle['open'])
                                if is_bearish or is_reversal:
                                    signal = 'SHORT'
                                    logger.info(f"CVD 숏: CVD-EMA↓, 가격 EMA 아래, 델타 스파이크")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.80,  # 최적 세팅으로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"CVD Delta 전략 분석 실패: {e}")
            return None
