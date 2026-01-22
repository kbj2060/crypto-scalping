"""
3. 박스권 Range Top/Bottom 반전 전략
최근 60봉의 고점/저점을 이용한 반전 매매
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class RangeTopBottomStrategy:
    def __init__(self):
        self.name = "Range Top/Bottom"
        self.range_lookback = 60  # 최근 60봉
        self.range_max_width = 0.10  # RangeHigh - RangeLow < 10% (완화: 0.05 -> 0.10, 큰 박스권도 매매)
        self.volume_sma_period = 20
        self.wick_threshold = 0.0  # 꼬리 없어도 됨 (완화: 0.0003 -> 0.0, 몸통으로 닿아도 역매매)
        self.stop_loss_pct = 0.003  # Range 바깥으로 0.3% 이탈
    
    def analyze(self, data_collector):
        """박스권 Range Top/Bottom 반전 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < self.range_lookback + 5:
                return None
            
            # 최근 60봉 데이터
            recent_data = eth_data.tail(self.range_lookback)
            
            # Range 계산
            range_high = float(recent_data['high'].max())
            range_low = float(recent_data['low'].min())
            range_mid = (range_high + range_low) / 2
            range_width = (range_high - range_low) / range_low
            
            # 필터: RangeHigh - RangeLow < 1.5%
            if range_width >= self.range_max_width:
                return None
            
            # 거래량 필터 제거 (반전 시점에는 거래량이 늘어날 수 있으므로)
            
            latest = eth_data.iloc[-1]
            latest_close = float(latest['close'])
            latest_high = float(latest['high'])
            latest_low = float(latest['low'])
            latest_open = float(latest['open'])
            
            signal = None
            entry_price = latest_close
            stop_loss = None
            take_profit = range_mid  # 목표: Mid 또는 진입가 대비 0.4%
            
            # LONG: Price <= RangeLow * 1.005 (꼬리 조건 제거)
            range_low_threshold = range_low * 1.005
            if latest_close <= range_low_threshold:
                signal = 'LONG'
                stop_loss = range_low * (1 - self.stop_loss_pct)  # Range 바깥으로 0.3% 이탈
                take_profit = max(range_mid, entry_price * (1 + 0.004))  # Mid 또는 +0.4%
            
            # SHORT: Price >= RangeHigh * 0.995 (꼬리 조건 제거)
            range_high_threshold = range_high * 0.995
            if latest_close >= range_high_threshold:
                signal = 'SHORT'
                stop_loss = range_high * (1 + self.stop_loss_pct)  # Range 바깥으로 0.3% 이탈
                take_profit = min(range_mid, entry_price * (1 - 0.004))  # Mid 또는 -0.4%
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': 0.73,
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Range Top/Bottom 전략 분석 실패: {e}")
            return None
