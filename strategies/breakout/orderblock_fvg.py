"""
4. 오더블록(OB) + FVG 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class OrderblockFVGStrategy:
    def __init__(self):
        self.name = "Orderblock FVG"
        self.ob_volume_multiplier = 0.0  # 거래량 조건 사실상 제거
        self.ob_touch_min = 0.4  # OB의 40% 구간
        self.ob_touch_max = 0.6  # OB의 60% 구간
        self.fvg_min_gap = 0.0  # 모든 갭 포착 (0.0001 -> 0.0)
        self.fvg_retest_level = 0.1  # 갭의 10% 레벨 리테스트
        self.fvg_tolerance_pct = 0.9  # 90% 허용 오차
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
                
                # 강한 캔들: 몸통 0.15% 이상 (거래량 필터 제거)
                if body_pct >= 0.0015:
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
            logger.debug(f"🔍 [Orderblock FVG] 전략 분석 시작")
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < 20:
                logger.debug(f"⚠️ [Orderblock FVG] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}")
                return None
            logger.debug(f"🔍 [Orderblock FVG] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
            # FVG 탐지 (3-bar FVG)
            fvgs = Indicators.find_fvg(eth_data, lookback=3)
            if fvgs is None or len(fvgs) == 0:
                logger.debug(f"⚠️ [Orderblock FVG] FVG 탐지 실패: fvgs={fvgs is not None}, len={len(fvgs) if fvgs is not None else 0}")
                return None
            
            # 오더블록 탐지
            order_block = self.find_order_block(eth_data, lookback=10)
            
            latest = eth_data.iloc[-1]
            current_price = latest['close']
            
            signal = None
            entry_price = current_price
            stop_loss = None
            
            logger.debug(f"🔍 [Orderblock FVG] FVG 탐지: {len(fvgs)}개 발견")
            logger.debug(f"🔍 [Orderblock FVG] 오더블록: {order_block['type'] if order_block else '없음'}")
            
            # 최근 FVG 확인
            for idx, fvg in enumerate(reversed(fvgs[-5:])):  # 최근 5개만 확인
                fvg_top = fvg['top']
                fvg_bottom = fvg['bottom']
                fvg_range = fvg_top - fvg_bottom
                fvg_range_pct = fvg_range / fvg_bottom
                
                logger.debug(f"🔍 [Orderblock FVG] FVG #{idx+1} - 타입: {fvg['type']}, 상단: {fvg_top:.2f}, 하단: {fvg_bottom:.2f}, 범위: {fvg_range_pct:.4%}")
                
                # FVG 최소 갭 크기 필터 제거 (모든 갭 허용)
                # if fvg_range_pct < self.fvg_min_gap: continue
                
                # FVG 10% 레벨 리테스트 또는 즉시 진입 (공격적 모드)
                # 옵션 1: FVG 10% 레벨 리테스트 (30% 허용 오차)
                fvg_retest_level = fvg_bottom + (fvg_range * self.fvg_retest_level)
                fvg_tolerance = fvg_range * self.fvg_tolerance_pct  # 30% 허용 오차
                
                # 옵션 2: FVG 생성 즉시 진입 (가격이 갭 범위 내에 있으면)
                in_fvg_range = fvg_bottom <= current_price <= fvg_top
                retest_distance = abs(current_price - fvg_retest_level)
                
                logger.debug(f"🔍 [Orderblock FVG] FVG #{idx+1} 리테스트 분석 - 현재가: {current_price:.2f}, 리테스트 레벨: {fvg_retest_level:.2f}, 거리: {retest_distance:.2f}, 허용: {fvg_tolerance:.2f}")
                logger.debug(f"🔍 [Orderblock FVG] FVG #{idx+1} 범위 체크 - FVG 내부: {in_fvg_range}, 리테스트 범위: {retest_distance <= fvg_tolerance}")
                
                # 가격이 FVG 10% 레벨 ±30% 범위에 있거나, FVG 범위 내에 있으면 진입
                if abs(current_price - fvg_retest_level) <= fvg_tolerance or in_fvg_range:
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
                    
                    # 단독 신호 허용: OB 없이 FVG만 있어도, FVG 없이 OB 리테스트만 있어도 진입
                    if ob_match or order_block is None or in_fvg_range:
                        if fvg['type'] == 'bullish':
                            signal = 'LONG'
                            stop_loss = fvg_bottom * 0.999
                            break
                        elif fvg['type'] == 'bearish':
                            signal = 'SHORT'
                            stop_loss = fvg_top * 1.001
                            break
                    
                    # OB만 있어도 신호 발생 (FVG 없이)
                    if order_block and not in_fvg_range:
                        if order_block['type'] == 'bullish' and ob_match:
                            signal = 'LONG'
                            stop_loss = order_block['body_bottom'] * 0.999
                            break
                        elif order_block['type'] == 'bearish' and ob_match:
                            signal = 'SHORT'
                            stop_loss = order_block['body_top'] * 1.001
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
