"""
4. 오더블록(OB) + FVG 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import Indicators

logger = logging.getLogger(__name__)


class OrderblockFVGStrategy:
    def __init__(self):
        self.name = "Orderblock FVG"
        self.ob_volume_multiplier = 1.7  # 직전 10봉 평균의 1.7배 이상
        self.ob_touch_min = 0.4  # OB의 40% 구간
        self.ob_touch_max = 0.6  # OB의 60% 구간
        self.fvg_min_gap = 0.0005  # 최소 갭 크기: 0.05% (0.08%에서 완화)
        self.fvg_retest_min = 0.3  # 갭의 30% 레벨 리테스트 (하한)
        self.fvg_retest_max = 0.7  # 갭의 70% 레벨 리테스트 (상한)
        self.min_signal_distance = 5  # 최소 신호 거리: 5봉
    
    def find_order_block(self, data, lookback=10):
        """오더블록 탐지 (직전 하락/상승 마지막 강한 캔들)"""
        try:
            if len(data) < lookback + 1:
                return None
            
            recent_data = data.tail(lookback + 1)
            volume_mean = recent_data['volume'].head(lookback).mean()
            
            # 직전 10봉에서 강한 캔들 찾기
            for i in range(len(recent_data) - 2, -1, -1):
                candle = recent_data.iloc[i]
                body = abs(candle['close'] - candle['open'])
                body_pct = body / candle['open']
                
                # 강한 캔들: 몸통 0.3% 이상 + 볼륨 1.7배 이상
                if body_pct >= 0.003 and candle['volume'] >= volume_mean * self.ob_volume_multiplier:
                    if candle['close'] > candle['open']:
                        # 강한 상승 캔들 → Bearish OB
                        return {
                            'type': 'bearish',
                            'high': candle['high'],
                            'low': candle['low'],
                            'body_top': max(candle['open'], candle['close']),
                            'body_bottom': min(candle['open'], candle['close']),
                            'index': i
                        }
                    else:
                        # 강한 하락 캔들 → Bullish OB
                        return {
                            'type': 'bullish',
                            'high': candle['high'],
                            'low': candle['low'],
                            'body_top': max(candle['open'], candle['close']),
                            'body_bottom': min(candle['open'], candle['close']),
                            'index': i
                        }
            
            return None
        except Exception as e:
            logger.error(f"오더블록 탐지 실패: {e}")
            return None
    
    def analyze(self, data_collector):
        """FVG + OB 재진입 전략 분석 (최적 세팅)"""
        try:
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < 20:
                return None
            
            # FVG 탐지 (3-bar FVG)
            fvgs = Indicators.find_fvg(eth_data, lookback=3)
            if fvgs is None or len(fvgs) == 0:
                return None
            
            # 오더블록 탐지
            order_block = self.find_order_block(eth_data, lookback=10)
            
            latest = eth_data.iloc[-1]
            current_price = latest['close']
            
            signal = None
            entry_price = current_price
            stop_loss = None
            
            # 최근 FVG 확인
            for fvg in reversed(fvgs[-5:]):  # 최근 5개만 확인
                fvg_top = fvg['top']
                fvg_bottom = fvg['bottom']
                fvg_range = fvg_top - fvg_bottom
                fvg_range_pct = fvg_range / fvg_bottom
                
                # FVG 최소 갭 크기 필터: 0.05% 이상
                if fvg_range_pct < self.fvg_min_gap:
                    continue
                
                # FVG 30~70% 레벨 리테스트 (범위 확대)
                fvg_retest_min_level = fvg_bottom + (fvg_range * self.fvg_retest_min)
                fvg_retest_max_level = fvg_bottom + (fvg_range * self.fvg_retest_max)
                
                # 가격이 FVG 30~70% 범위에 있는지 확인
                if fvg_retest_min_level <= current_price <= fvg_retest_max_level:
                    # OB 터치 시 동일 방향 확인
                    ob_match = False
                    if order_block:
                        ob_range = order_block['body_top'] - order_block['body_bottom']
                        ob_touch_zone_top = order_block['body_bottom'] + (ob_range * self.ob_touch_max)
                        ob_touch_zone_bottom = order_block['body_bottom'] + (ob_range * self.ob_touch_min)
                        
                        # OB 40~60% 구간 터치 확인
                        if ob_touch_zone_bottom <= current_price <= ob_touch_zone_top:
                            if fvg['type'] == 'bullish' and order_block['type'] == 'bullish':
                                ob_match = True
                            elif fvg['type'] == 'bearish' and order_block['type'] == 'bearish':
                                ob_match = True
                    
                    # OB 매칭 또는 FVG만으로도 신호 발생
                    if ob_match or order_block is None:
                        if fvg['type'] == 'bullish':
                            signal = 'LONG'
                            stop_loss = fvg_bottom * 0.999
                            logger.info(f"FVG + OB Long: FVG 50% 리테스트, OB {'매칭' if ob_match else '없음'}")
                            break
                        elif fvg['type'] == 'bearish':
                            signal = 'SHORT'
                            stop_loss = fvg_top * 1.001
                            logger.info(f"FVG + OB Short: FVG 50% 리테스트, OB {'매칭' if ob_match else '없음'}")
                            break
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'confidence': 0.65,  # 최적 세팅으로 신뢰도 조정
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"오더블록 FVG 전략 분석 실패: {e}")
            return None
