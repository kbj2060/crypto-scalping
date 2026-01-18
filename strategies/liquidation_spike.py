"""
7. 청산 스파이크 전략
"""
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import Indicators

logger = logging.getLogger(__name__)


class LiquidationSpikeStrategy:
    def __init__(self):
        self.name = "Liquidation Spike"
        self.time_window_minutes = 3  # 3분 내 청산 스파이크 탐지
        self.min_volume_threshold = 10  # 최소 청산 볼륨 (ETH 수량)
    
    def analyze(self, data_collector):
        """청산 스파이크 전략 분석"""
        try:
            # 청산 스파이크 탐지
            spike_data = data_collector.detect_liquidation_spike(
                symbol='ETH',
                time_window_minutes=self.time_window_minutes,
                min_volume_threshold=self.min_volume_threshold
            )
            
            if spike_data is None or not spike_data['spike_detected']:
                return None
            
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < 5:
                return None
            
            latest = eth_data.iloc[-1]
            entry_price = latest['close']
            signal = None
            
            # 롱 청산 스파이크 (숏 포지션 대량 청산) → 가격 상승 압력 → LONG
            if spike_data['spike_type'] == 'long_liquidation':
                signal = 'LONG'
                logger.info(f"롱 청산 스파이크 탐지: {spike_data['total_volume']:.2f} ETH, {spike_data['count']}건")
            
            # 숏 청산 스파이크 (롱 포지션 대량 청산) → 가격 하락 압력 → SHORT
            elif spike_data['spike_type'] == 'short_liquidation':
                signal = 'SHORT'
                logger.info(f"숏 청산 스파이크 탐지: {spike_data['total_volume']:.2f} ETH, {spike_data['count']}건")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.70,  # 청산 스파이크 신뢰도
                    'strategy': self.name,
                    'spike_data': spike_data  # 추가 정보
                }
            
            return None
            
        except Exception as e:
            logger.error(f"청산 스파이크 전략 분석 실패: {e}")
            return None
