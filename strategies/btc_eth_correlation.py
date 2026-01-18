"""
6. BTC 연동 모멘텀 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import Indicators

logger = logging.getLogger(__name__)


class BTCEthCorrelationStrategy:
    def __init__(self):
        self.name = "BTC/ETH Correlation"
        self.rsi_long_threshold = 40  # BTC RSI < 40 → ETH 숏 bias
        self.rsi_short_threshold = 60  # BTC RSI > 60 → ETH 롱 bias
        self.ma_period = 20
        self.ma_consecutive = 3  # MA20 위/아래 3봉 연속
    
    def analyze(self, data_collector):
        """BTC 연동 모멘텀 전략 분석 (최적 세팅)"""
        try:
            btc_data = data_collector.get_candles('BTC', count=50)
            eth_data = data_collector.get_candles('ETH', count=50)
            
            if btc_data is None or eth_data is None:
                return None
            
            if len(btc_data) < 25 or len(eth_data) < 25:
                return None
            
            # BTC 지표 계산
            btc_rsi = Indicators.calculate_rsi(btc_data, period=14)
            btc_ma = Indicators.calculate_sma(btc_data['close'], period=self.ma_period)
            
            if btc_rsi is None or btc_ma is None:
                return None
            
            # BTC RSI 및 MA20 확인
            btc_rsi_latest = float(btc_rsi.iloc[-1])
            btc_current = btc_data.iloc[-1]
            btc_ma_current = float(btc_ma.iloc[-1])
            
            # BTC 임펄스 필터: MA20 위/아래 3봉 연속
            btc_above_ma = True
            btc_below_ma = True
            for i in range(1, self.ma_consecutive + 1):
                if len(btc_data) >= i + 1 and len(btc_ma) >= i + 1:
                    btc_price = float(btc_data.iloc[-i]['close'])
                    btc_ma_val = float(btc_ma.iloc[-i])
                    if btc_price <= btc_ma_val:
                        btc_above_ma = False
                    if btc_price >= btc_ma_val:
                        btc_below_ma = False
            
            eth_current = eth_data.iloc[-1]
            entry_price = float(eth_current['close'])
            signal = None
            
            # 롱 bias: BTC RSI > 60 AND BTC MA20 위 3봉 연속
            if btc_rsi_latest > self.rsi_short_threshold and btc_above_ma:
                # ETH가 상승 추세인지 확인
                eth_prev = eth_data.iloc[-2] if len(eth_data) >= 2 else None
                if eth_prev is not None:
                    eth_current_close = float(eth_current['close'])
                    eth_prev_close = float(eth_prev['close'])
                    if eth_current_close > eth_prev_close:
                        signal = 'LONG'
                        logger.info(f"BTC 연동 롱: BTC RSI={btc_rsi_latest:.2f}, MA20 위 3봉 연속")
            
            # 숏 bias: BTC RSI < 40 AND BTC MA20 아래 3봉 연속
            elif btc_rsi_latest < self.rsi_long_threshold and btc_below_ma:
                # ETH가 하락 추세인지 확인
                eth_prev = eth_data.iloc[-2] if len(eth_data) >= 2 else None
                if eth_prev is not None:
                    eth_current_close = float(eth_current['close'])
                    eth_prev_close = float(eth_prev['close'])
                    if eth_current_close < eth_prev_close:
                        signal = 'SHORT'
                        logger.info(f"BTC 연동 숏: BTC RSI={btc_rsi_latest:.2f}, MA20 아래 3봉 연속")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.78,  # 최적 세팅으로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"BTC/ETH 상관 전략 분석 실패: {e}")
            return None
