"""
7. 청산 스파이크 전략
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class LiquidationSpikeStrategy:
    def __init__(self):
        self.name = "Liquidation Spike"
        self.time_window_minutes = 3  # 3분 내 청산 스파이크 탐지
        self.min_volume_threshold = 1  # 최소 청산 볼륨 (ETH 수량, 공격적: 5 -> 1, 아주 적은 청산도 신호)
    
    def analyze(self, data_collector):
        """청산 스파이크 전략 분석"""
        try:
            logger.debug(f"🔍 [Liquidation Spike] 전략 분석 시작")
            # 청산 스파이크 탐지
            spike_data = data_collector.detect_liquidation_spike(
                symbol='ETH',
                time_window_minutes=self.time_window_minutes,
                min_volume_threshold=self.min_volume_threshold
            )
            
            logger.debug(f"🔍 [Liquidation Spike] 탐지 결과 - 스파이크: {spike_data['spike_detected'] if spike_data else False}, 타입: {spike_data.get('spike_type', 'N/A') if spike_data else 'N/A'}")
            if spike_data:
                logger.debug(f"🔍 [Liquidation Spike] 상세 - 총 볼륨: {spike_data.get('total_volume', 0):.2f} ETH, 건수: {spike_data.get('count', 0)}, 임계값: {self.min_volume_threshold} ETH")
            
            if spike_data is None or not spike_data['spike_detected']:
                logger.debug(f"⚠️ [Liquidation Spike] 스파이크 없음: spike_data={spike_data is not None}, detected={spike_data.get('spike_detected', False) if spike_data else False}")
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
