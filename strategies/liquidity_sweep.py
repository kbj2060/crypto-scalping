"""
2. 유동성 스윕(Liquidity Sweep) 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from indicators import Indicators

logger = logging.getLogger(__name__)


class LiquiditySweepStrategy:
    def __init__(self):
        self.name = "Liquidity Sweep"
        self.liquidity_lookback = 20  # 고점/저점 비교 봉수: 20봉
        self.sweep_body_max_pct = 0.4  # 스윕 바 길이 < 전체의 40%
        self.stop_loss_percent = 0.0015  # 스윕 extremum ±0.15%
    
    def analyze(self, data_collector):
        """유동성 스윕 전략 분석 (최적 세팅)"""
        try:
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < self.liquidity_lookback + 5:
                return None
            
            # 이전 20봉 안의 주요 유동성 찾기 (현재 캔들 제외)
            recent_data = eth_data.iloc[:-1].tail(self.liquidity_lookback)
            major_high = recent_data['high'].max()
            major_low = recent_data['low'].min()
            
            # 최신 캔들들
            latest = eth_data.iloc[-1]
            prev_candle = eth_data.iloc[-2] if len(eth_data) >= 2 else None
            prev_prev_candle = eth_data.iloc[-3] if len(eth_data) >= 3 else None
            
            signal = None
            stop_loss = None
            entry_price = latest['close']
            
            # 고점 스윕: 이전 20봉 고점 돌파 → 되돌림 마감 → 진입
            if latest['high'] > major_high:
                # 스윕 바 길이(body) < 전체의 40%
                sweep_body = abs(latest['close'] - latest['open'])
                sweep_range = latest['high'] - latest['low']
                body_ratio = sweep_body / sweep_range if sweep_range > 0 else 0
                
                if body_ratio < self.sweep_body_max_pct:
                    # 스윕 후 반전 캔들 1개 확인 (되돌림 마감)
                    if prev_candle and latest['close'] < major_high:
                        signal = 'SHORT'
                        stop_loss = major_high * (1 + self.stop_loss_percent)
                        logger.info(f"Bearish Sweep: 고점 {major_high:.2f} 돌파 후 되돌림")
            
            # 저점 스윕: 이전 20봉 저점 이탈 → 되돌림 마감 → 진입
            if latest['low'] < major_low:
                # 스윕 바 길이(body) < 전체의 40%
                sweep_body = abs(latest['close'] - latest['open'])
                sweep_range = latest['high'] - latest['low']
                body_ratio = sweep_body / sweep_range if sweep_range > 0 else 0
                
                if body_ratio < self.sweep_body_max_pct:
                    # 스윕 후 반전 캔들 1개 확인 (되돌림 마감)
                    if prev_candle and latest['close'] > major_low:
                        signal = 'LONG'
                        stop_loss = major_low * (1 - self.stop_loss_percent)
                        logger.info(f"Bullish Sweep: 저점 {major_low:.2f} 이탈 후 되돌림")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'confidence': 0.75,  # 최적 세팅으로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"유동성 스윕 전략 분석 실패: {e}")
            return None
