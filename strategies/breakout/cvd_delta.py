"""
1. CVD / 델타 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
import pandas as pd
import numpy as np
import traceback
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
        self.delta_spike_multiplier = 1.1  # 평균 델타의 1.1배 (공격적: 1.5 -> 1.1, 거의 없앰)
        self.divergence_lookback = 60  # 30 -> 60 (더 긴 흐름 파악)
        self.divergence_strength = 1  # 2 -> 1 (공격적: 아주 미세한 꺾임도 포착)
    
    def find_divergence(self, price_data, cvd_data, lookback=60):
        """CVD 다이버전스 탐지 (Lookback 60, Strength 2)"""
        try:
            logger.debug(f"[find_divergence] 시작 - price_data 타입: {type(price_data)}, cvd_data 타입: {type(cvd_data)}")
            
            if len(price_data) < lookback or len(cvd_data) < lookback:
                logger.debug(f"[find_divergence] 데이터 부족 - price_data: {len(price_data)}, cvd_data: {len(cvd_data)}, lookback: {lookback}")
                return None
            
            recent_price = price_data.tail(lookback)
            recent_cvd = cvd_data.tail(lookback)
            logger.debug(f"[find_divergence] recent_price 타입: {type(recent_price)}, recent_cvd 타입: {type(recent_cvd)}")
            
            # 가격 저점/고점 찾기 (Strength 5 = 양쪽 5개씩 확인)
            price_lows = []
            price_highs = []
            cvd_lows = []
            cvd_highs = []
            
            for i in range(self.divergence_strength, len(recent_price) - self.divergence_strength):
                try:
                    # 가격 저점
                    is_low = True
                    try:
                        price_low_i_val = recent_price.iloc[i]['low']
                        logger.debug(f"[find_divergence] i={i}, price_low_i 타입: {type(price_low_i_val)}, 값: {price_low_i_val}")
                        price_low_i = float(price_low_i_val)
                    except Exception as e:
                        logger.error(f"[find_divergence] 가격 저점 i={i} 변환 실패: {e}, 타입: {type(price_low_i_val)}")
                        raise
                    
                    for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                        if j != i:
                            try:
                                price_low_j_val = recent_price.iloc[j]['low']
                                price_low_j = float(price_low_j_val)
                                if price_low_j <= price_low_i:
                                    is_low = False
                                    break
                            except Exception as e:
                                logger.error(f"[find_divergence] 가격 저점 j={j} 비교 실패: {e}")
                                raise
                    if is_low:
                        price_lows.append({'index': i, 'value': price_low_i})
                    
                    # 가격 고점
                    is_high = True
                    try:
                        price_high_i_val = recent_price.iloc[i]['high']
                        logger.debug(f"[find_divergence] i={i}, price_high_i 타입: {type(price_high_i_val)}, 값: {price_high_i_val}")
                        price_high_i = float(price_high_i_val)
                    except Exception as e:
                        logger.error(f"[find_divergence] 가격 고점 i={i} 변환 실패: {e}, 타입: {type(price_high_i_val)}")
                        raise
                    
                    for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                        if j != i:
                            try:
                                price_high_j_val = recent_price.iloc[j]['high']
                                price_high_j = float(price_high_j_val)
                                if price_high_j >= price_high_i:
                                    is_high = False
                                    break
                            except Exception as e:
                                logger.error(f"[find_divergence] 가격 고점 j={j} 비교 실패: {e}")
                                raise
                    if is_high:
                        price_highs.append({'index': i, 'value': price_high_i})
                    
                    # CVD 저점
                    is_cvd_low = True
                    try:
                        cvd_value_i_val = recent_cvd.iloc[i]
                        logger.debug(f"[find_divergence] i={i}, cvd_value_i 타입: {type(cvd_value_i_val)}, 값: {cvd_value_i_val}")
                        cvd_value_i = float(cvd_value_i_val)
                    except Exception as e:
                        logger.error(f"[find_divergence] CVD 저점 i={i} 변환 실패: {e}, 타입: {type(cvd_value_i_val)}")
                        raise
                    
                    for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                        if j != i:
                            try:
                                cvd_value_j_val = recent_cvd.iloc[j]
                                cvd_value_j = float(cvd_value_j_val)
                                if cvd_value_j <= cvd_value_i:
                                    is_cvd_low = False
                                    break
                            except Exception as e:
                                logger.error(f"[find_divergence] CVD 저점 j={j} 비교 실패: {e}")
                                raise
                    if is_cvd_low:
                        cvd_lows.append({'index': i, 'value': cvd_value_i})
                    
                    # CVD 고점
                    is_cvd_high = True
                    try:
                        cvd_value_i_val = recent_cvd.iloc[i]
                        logger.debug(f"[find_divergence] i={i}, cvd_value_i(고점) 타입: {type(cvd_value_i_val)}, 값: {cvd_value_i_val}")
                        cvd_value_i = float(cvd_value_i_val)
                    except Exception as e:
                        logger.error(f"[find_divergence] CVD 고점 i={i} 변환 실패: {e}, 타입: {type(cvd_value_i_val)}")
                        raise
                    
                    for j in range(i - self.divergence_strength, i + self.divergence_strength + 1):
                        if j != i:
                            try:
                                cvd_value_j_val = recent_cvd.iloc[j]
                                cvd_value_j = float(cvd_value_j_val)
                                if cvd_value_j >= cvd_value_i:
                                    is_cvd_high = False
                                    break
                            except Exception as e:
                                logger.error(f"[find_divergence] CVD 고점 j={j} 비교 실패: {e}")
                                raise
                    if is_cvd_high:
                        cvd_highs.append({'index': i, 'value': cvd_value_i})
                        
                except Exception as e:
                    logger.error(f"[find_divergence] 루프 i={i} 처리 실패: {e}")
                    logger.error(traceback.format_exc())
                    raise
            
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
            logger.error(f"에러 상세 정보:\n{traceback.format_exc()}")
            return None
    
    def analyze(self, data_collector):
        """CVD / 델타 전략 분석 (최적 세팅)"""
        try:
            logger.debug(f"🔍 [CVD Delta] 전략 분석 시작")
            
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                logger.debug(f"⚠️ [CVD Delta] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}")
                return None
            logger.debug(f"🔍 [CVD Delta] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
            logger.debug(f"[analyze] ETH 데이터 수집 완료: {len(eth_data)}개")
            
            # CVD 계산 (EMA 21, 델타 스무딩 5)
            try:
                cvd_data = data_collector.calculate_cvd('ETH', lookback=100, ema_period=self.cvd_ema_period, delta_smoothing=self.delta_smoothing)
                if cvd_data is None:
                    logger.debug("[analyze] CVD 데이터 계산 실패")
                    return None
                logger.debug(f"[analyze] CVD 데이터 계산 완료, 타입: {type(cvd_data)}, 컬럼: {cvd_data.columns.tolist() if hasattr(cvd_data, 'columns') else 'N/A'}")
            except Exception as e:
                logger.error(f"[analyze] CVD 계산 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # Price EMA 21
            try:
                price_ema = Indicators.calculate_ema(eth_data, period=self.price_ema_period)
                if price_ema is None:
                    logger.debug("[analyze] Price EMA 계산 실패")
                    return None
                logger.debug(f"[analyze] Price EMA 계산 완료, 타입: {type(price_ema)}")
            except Exception as e:
                logger.error(f"[analyze] Price EMA 계산 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # CVD EMA 방향성 필터
            try:
                cvd_ema_current_val = cvd_data['cvd_ema'].iloc[-1]
                logger.debug(f"[analyze] cvd_ema_current 타입: {type(cvd_ema_current_val)}, 값: {cvd_ema_current_val}")
                cvd_ema_current = float(cvd_ema_current_val)
                
                cvd_ema_prev = float(cvd_data['cvd_ema'].iloc[-2]) if len(cvd_data) >= 2 else None
                
                price_ema_current_val = price_ema.iloc[-1]
                logger.debug(f"[analyze] price_ema_current 타입: {type(price_ema_current_val)}, 값: {price_ema_current_val}")
                price_ema_current = float(price_ema_current_val)
                
                price_current_val = eth_data.iloc[-1]['close']
                logger.debug(f"[analyze] price_current 타입: {type(price_current_val)}, 값: {price_current_val}")
                price_current = float(price_current_val)
            except Exception as e:
                logger.error(f"[analyze] 값 변환 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # 최신 델타 (스무딩된 값)
            try:
                latest_delta_val = cvd_data['delta_smooth'].iloc[-1]
                logger.debug(f"[analyze] latest_delta 타입: {type(latest_delta_val)}, 값: {latest_delta_val}")
                latest_delta = float(latest_delta_val)
                delta_mean = abs(float(cvd_data['delta_smooth'].tail(20).mean()))
                delta_spike_threshold = delta_mean * self.delta_spike_multiplier
                logger.debug(f"🔍 [CVD Delta] 델타 분석 - 현재: {latest_delta:.0f}, 평균: {delta_mean:.0f}, 스파이크 임계값: {delta_spike_threshold:.0f} (배수: {self.delta_spike_multiplier})")
                cvd_ema_prev_str = f"{cvd_ema_prev:.0f}" if cvd_ema_prev is not None else 'N/A'
                direction = '↑' if cvd_ema_prev is not None and cvd_ema_current > cvd_ema_prev else '↓' if cvd_ema_prev is not None and cvd_ema_current < cvd_ema_prev else '='
                logger.debug(f"🔍 [CVD Delta] CVD EMA - 현재: {cvd_ema_current:.0f}, 이전: {cvd_ema_prev_str}, 방향: {direction}")
                logger.debug(f"🔍 [CVD Delta] 가격 EMA - 현재: {price_ema_current:.2f}, 가격: {price_current:.2f}, 위치: {'EMA 위' if price_current > price_ema_current else 'EMA 아래'}")
            except Exception as e:
                logger.error(f"[analyze] 델타 계산 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # CVD 다이버전스 탐지
            try:
                price_data = eth_data.tail(self.divergence_lookback)
                cvd_series = cvd_data['cvd_ema']
                logger.debug(f"[analyze] 다이버전스 탐지 시작 - price_data 타입: {type(price_data)}, cvd_series 타입: {type(cvd_series)}")
                divergence = self.find_divergence(price_data, cvd_series, lookback=self.divergence_lookback)
                logger.debug(f"[analyze] 다이버전스 탐지 완료: {divergence}")
                logger.debug(f"🔍 [CVD Delta] 다이버전스 탐지 결과: {divergence if divergence else '없음'}")
            except Exception as e:
                logger.error(f"[analyze] 다이버전스 탐지 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            latest = eth_data.iloc[-1]
            prev_candle = eth_data.iloc[-2] if len(eth_data) >= 2 else None
            signal = None
            
            try:
                entry_price_val = latest['close']
                logger.debug(f"[analyze] entry_price 타입: {type(entry_price_val)}, 값: {entry_price_val}")
                entry_price = float(entry_price_val)
                
                # 최신 캔들 값들을 float로 변환
                latest_close_val = latest['close']
                latest_open_val = latest['open']
                logger.debug(f"[analyze] latest_close 타입: {type(latest_close_val)}, latest_open 타입: {type(latest_open_val)}")
                latest_close = float(latest_close_val)
                latest_open = float(latest_open_val)
            except Exception as e:
                logger.error(f"[analyze] 캔들 값 변환 중 에러: {e}")
                logger.error(traceback.format_exc())
                raise
            
            # 롱 조건: CVD-EMA ↑ + 가격 EMA 위 + (다이버전스 또는 델타 스파이크) + 캔들 반전
            logger.debug(f"🔍 [CVD Delta] 롱 조건 체크 - CVD 상승: {cvd_ema_prev is not None and cvd_ema_current > cvd_ema_prev}, 가격 EMA 위: {price_current > price_ema_current}")
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
                            if prev_candle is not None:
                                prev_close = float(prev_candle['close'])
                                prev_open = float(prev_candle['open'])
                                is_bullish = latest_close > latest_open
                                is_reversal = (latest_close > prev_close and 
                                             prev_close < prev_open)
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
                            if prev_candle is not None:
                                prev_close = float(prev_candle['close'])
                                prev_open = float(prev_candle['open'])
                                is_bearish = latest_close < latest_open
                                is_reversal = (latest_close < prev_close and 
                                              prev_close > prev_open)
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
            logger.error(f"에러 상세 정보:\n{traceback.format_exc()}")
            return None
