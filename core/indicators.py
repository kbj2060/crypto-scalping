"""
기술적 지표 계산 모듈
"""
import pandas as pd
import numpy as np
import talib
import logging

logger = logging.getLogger(__name__)


class Indicators:
    @staticmethod
    def calculate_rsi(data, period=14):
        """RSI 계산"""
        try:
            if len(data) < period + 1:
                return None
            close = data['close'].values
            rsi = talib.RSI(close, timeperiod=period)
            return pd.Series(rsi, index=data.index)
        except Exception as e:
            logger.error(f"RSI 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_ema(data, period):
        """EMA 계산"""
        try:
            if len(data) < period:
                return None
            close = data['close'].values
            ema = talib.EMA(close, timeperiod=period)
            return pd.Series(ema, index=data.index)
        except Exception as e:
            logger.error(f"EMA 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_bollinger_bands(data, period=20, std_dev=2):
        """볼린저 밴드 계산"""
        try:
            if len(data) < period:
                return None
            close = data['close'].values
            upper, middle, lower = talib.BBANDS(
                close, 
                timeperiod=period, 
                nbdevup=std_dev, 
                nbdevdn=std_dev, 
                matype=0
            )
            return {
                'upper': pd.Series(upper, index=data.index),
                'middle': pd.Series(middle, index=data.index),
                'lower': pd.Series(lower, index=data.index)
            }
        except Exception as e:
            logger.error(f"볼린저 밴드 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_bbw(bollinger_bands):
        """볼린저 밴드 폭 (BBW) 계산"""
        try:
            if bollinger_bands is None:
                return None
            bbw = (bollinger_bands['upper'] - bollinger_bands['lower']) / bollinger_bands['middle']
            return bbw
        except Exception as e:
            logger.error(f"BBW 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_keltner_channels(data, period=20, multiplier=1.5):
        """Keltner Channel 계산"""
        try:
            if len(data) < period:
                return None
            
            # EMA (중간선)
            ema = Indicators.calculate_ema(data, period=period)
            if ema is None:
                return None
            
            # ATR 계산
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            atr = talib.ATR(high, low, close, timeperiod=period)
            
            # Keltner Channel
            upper = ema + (pd.Series(atr, index=data.index) * multiplier)
            lower = ema - (pd.Series(atr, index=data.index) * multiplier)
            
            return {
                'upper': upper,
                'middle': ema,
                'lower': lower
            }
        except Exception as e:
            logger.error(f"Keltner Channel 계산 실패: {e}")
            return None
    
    @staticmethod
    def find_swing_points(data, lookback=5):
        """스윙 하이/로우 탐지"""
        try:
            if len(data) < lookback * 2 + 1:
                return None
            
            highs = data['high'].values
            lows = data['low'].values
            
            swing_highs = []
            swing_lows = []
            
            for i in range(lookback, len(data) - lookback):
                # Swing High
                is_swing_high = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and highs[j] >= highs[i]:
                        is_swing_high = False
                        break
                
                if is_swing_high:
                    swing_highs.append({
                        'index': i,
                        'value': highs[i],
                        'timestamp': data.index[i]
                    })
                
                # Swing Low
                is_swing_low = True
                for j in range(i - lookback, i + lookback + 1):
                    if j != i and lows[j] <= lows[i]:
                        is_swing_low = False
                        break
                
                if is_swing_low:
                    swing_lows.append({
                        'index': i,
                        'value': lows[i],
                        'timestamp': data.index[i]
                    })
            
            return {
                'swing_highs': swing_highs,
                'swing_lows': swing_lows
            }
        except Exception as e:
            logger.error(f"스윙 포인트 탐지 실패: {e}")
            return None
    
    @staticmethod
    def calculate_momentum(data, period=3):
        """모멘텀 계산"""
        try:
            if len(data) < period + 1:
                return None
            close = data['close'].values
            momentum = close - np.roll(close, period)
            momentum[:period] = np.nan
            return pd.Series(momentum, index=data.index)
        except Exception as e:
            logger.error(f"모멘텀 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_sma(data, period):
        """SMA 계산"""
        try:
            if len(data) < period:
                return None
            return data.rolling(window=period).mean()
        except Exception as e:
            logger.error(f"SMA 계산 실패: {e}")
            return None
    
    @staticmethod
    def find_fvg(data, lookback=3):
        """Fair Value Gap (FVG) 탐지"""
        try:
            if len(data) < lookback + 2:
                return None
            
            fvgs = []
            highs = data['high'].values
            lows = data['low'].values
            
            for i in range(lookback, len(data) - 1):
                # Bullish FVG: Low[n] > High[n-2]
                if i >= 2 and lows[i] > highs[i-2]:
                    fvgs.append({
                        'type': 'bullish',
                        'index': i,
                        'top': lows[i],
                        'bottom': highs[i-2],
                        'timestamp': data.index[i]
                    })
                
                # Bearish FVG: High[n] < Low[n-2]
                if i >= 2 and highs[i] < lows[i-2]:
                    fvgs.append({
                        'type': 'bearish',
                        'index': i,
                        'top': lows[i-2],
                        'bottom': highs[i],
                        'timestamp': data.index[i]
                    })
            
            return fvgs
        except Exception as e:
            logger.error(f"FVG 탐지 실패: {e}")
            return None
    
    @staticmethod
    def detect_divergence(price_data, indicator_data, lookback=20):
        """다이버전스 탐지 (가격 vs 지표)"""
        try:
            if len(price_data) < lookback or len(indicator_data) < lookback:
                return None
            
            recent_price = price_data.tail(lookback)
            recent_indicator = indicator_data.tail(lookback)
            
            # 최근 고점/저점 찾기
            price_high_idx = recent_price['high'].idxmax()
            price_low_idx = recent_price['low'].idxmin()
            
            indicator_high_idx = recent_indicator.idxmax()
            indicator_low_idx = recent_indicator.idxmin()
            
            # Bearish Divergence: 가격은 고점 상승, 지표는 고점 하락
            if price_high_idx > price_low_idx:  # 최근 고점이 더 최근
                price_prev_high = recent_price.loc[:price_high_idx]['high'].max()
                if price_prev_high < recent_price.loc[price_high_idx]['high']:
                    # 가격 고점 상승
                    indicator_prev_high = recent_indicator.loc[:indicator_high_idx].max()
                    if indicator_prev_high > recent_indicator.loc[indicator_high_idx]:
                        # 지표 고점 하락
                        return 'bearish_divergence'
            
            # Bullish Divergence: 가격은 저점 하락, 지표는 저점 상승
            if price_low_idx > price_high_idx:  # 최근 저점이 더 최근
                price_prev_low = recent_price.loc[:price_low_idx]['low'].min()
                if price_prev_low > recent_price.loc[price_low_idx]['low']:
                    # 가격 저점 하락
                    indicator_prev_low = recent_indicator.loc[:indicator_low_idx].min()
                    if indicator_prev_low < recent_indicator.loc[indicator_low_idx]:
                        # 지표 저점 상승
                        return 'bullish_divergence'
            
            return None
        except Exception as e:
            logger.error(f"다이버전스 탐지 실패: {e}")
            return None
    
    @staticmethod
    def calculate_macd(data, fastperiod=12, slowperiod=26, signalperiod=9):
        """MACD 계산"""
        try:
            if len(data) < slowperiod + signalperiod:
                return None
            close = data['close'].values
            macd, signal, hist = talib.MACD(close, fastperiod=fastperiod, slowperiod=slowperiod, signalperiod=signalperiod)
            return {
                'macd': pd.Series(macd, index=data.index),
                'signal': pd.Series(signal, index=data.index),
                'histogram': pd.Series(hist, index=data.index)
            }
        except Exception as e:
            logger.error(f"MACD 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_atr(data, period=14):
        """ATR (Average True Range) 계산"""
        try:
            if len(data) < period + 1:
                return None
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            atr = talib.ATR(high, low, close, timeperiod=period)
            return pd.Series(atr, index=data.index)
        except Exception as e:
            logger.error(f"ATR 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_stoch_rsi(data, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
        """Stochastic RSI 계산"""
        try:
            if len(data) < rsi_period + stoch_period + k_period:
                return None
            # RSI 계산
            rsi = Indicators.calculate_rsi(data, period=rsi_period)
            if rsi is None:
                return None
            # Stochastic RSI 계산
            stoch_k, stoch_d = talib.STOCH(
                rsi.values, rsi.values, rsi.values,
                fastk_period=stoch_period,
                slowk_period=k_period,
                slowd_period=d_period
            )
            return {
                'k': pd.Series(stoch_k, index=data.index),
                'd': pd.Series(stoch_d, index=data.index)
            }
        except Exception as e:
            logger.error(f"Stochastic RSI 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_vwap(data, session_start='00:00'):
        """VWAP 계산 (세션 기준)"""
        try:
            if len(data) < 1:
                return None
            # 세션별 VWAP 계산 (일일 기준)
            data_copy = data.copy()
            data_copy['date'] = data_copy.index.date
            data_copy['typical_price'] = (data_copy['high'] + data_copy['low'] + data_copy['close']) / 3
            data_copy['pv'] = data_copy['typical_price'] * data_copy['volume']
            
            # 날짜별 그룹화하여 VWAP 계산
            vwap_list = []
            for date, group in data_copy.groupby('date'):
                # 그룹화 컬럼('date')을 제외하고 계산
                cumsum_pv = group['pv'].cumsum()
                cumsum_volume = group['volume'].cumsum()
                vwap_group = cumsum_pv / cumsum_volume
                vwap_list.append(vwap_group)
            
            # 모든 그룹의 VWAP을 하나의 Series로 결합
            if vwap_list:
                vwap = pd.concat(vwap_list)
                # 원본 데이터의 인덱스 순서에 맞게 정렬
                vwap = vwap.reindex(data.index)
                return vwap
            else:
                return None
        except Exception as e:
            logger.error(f"VWAP 계산 실패: {e}")
            return None
    
    @staticmethod
    def calculate_adx(data, period=14):
        """ADX (Average Directional Index) 계산 - 추세 강도 지표"""
        try:
            if len(data) < period + 1:
                return None
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            adx = talib.ADX(high, low, close, timeperiod=period)
            return pd.Series(adx, index=data.index)
        except Exception as e:
            logger.error(f"ADX 계산 실패: {e}")
            return None