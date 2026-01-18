"""
3. 변동성 스퀴즈(Bollinger + Keltner) 전략 최적화 (1500봉 기준)
"""
import logging
import pandas as pd
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class VolatilitySqueezeStrategy:
    def __init__(self):
        self.name = "Volatility Squeeze"
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.keltner_period = 20
        self.keltner_multiplier = 1.5
        self.bbw_squeeze = 0.06  # BBW < 0.06 → 스퀴즈
        self.bbw_explosion = 0.09  # BBW > 0.09 → 폭발
        self.volume_explosion = 1.3  # 거래량 1.3배 이상
    
    def analyze(self, data_collector):
        """볼륨 스퀴즈 전략 분석 (최적 세팅)"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                return None
            
            # 볼린저 밴드 계산 (period 20, std_dev 2.0)
            bb = Indicators.calculate_bollinger_bands(eth_data, period=self.bb_period, std_dev=self.bb_std_dev)
            if bb is None:
                return None
            
            # Keltner Channel 계산 (period 20, multiplier 1.5)
            kc = Indicators.calculate_keltner_channels(eth_data, period=self.keltner_period, multiplier=self.keltner_multiplier)
            if kc is None:
                return None
            
            # BBW 계산
            bbw = Indicators.calculate_bbw(bb)
            if bbw is None:
                return None
            
            # 거래량 SMA
            volume_sma = Indicators.calculate_sma(eth_data['volume'], period=20)
            if volume_sma is None:
                return None
            
            # 최신 값
            latest = eth_data.iloc[-1]
            latest_bbw = bbw.iloc[-1]
            latest_volume = latest['volume']
            latest_volume_sma = volume_sma.iloc[-1]
            
            signal = None
            entry_price = latest['close']
            
            # 최근 10봉 이내에 스퀴즈(0.06 미만)가 있었는지 확인
            was_squeezed = (bbw.tail(10) < self.bbw_squeeze).any()
            # 현재 BBW가 상승세 전환했는지 확인 (이전 봉 대비 증가)
            if len(bbw) >= 2:
                prev_bbw = bbw.iloc[-2]
                is_exploding = latest_bbw > prev_bbw and latest_bbw > self.bbw_squeeze  # 상승 전환 + 스퀴즈 구간 벗어남
            else:
                is_exploding = False
            
            if was_squeezed and is_exploding:
                upper_band = bb['upper'].iloc[-1]
                lower_band = bb['lower'].iloc[-1]
                
                # 폭발 양봉: 상단 돌파 + 거래량
                if (latest['close'] > upper_band and 
                    latest_volume >= latest_volume_sma * self.volume_explosion):
                    signal = 'LONG'
                    logger.info(f"스퀴즈 폭발 Long: 이전 스퀴즈 후 BBW={latest_bbw:.4f}, 상단 돌파, 거래량 {latest_volume/latest_volume_sma:.2f}배")
                
                # 폭발 음봉: 하단 돌파 + 거래량
                elif (latest['close'] < lower_band and 
                      latest_volume >= latest_volume_sma * self.volume_explosion):
                    signal = 'SHORT'
                    logger.info(f"스퀴즈 폭발 Short: 이전 스퀴즈 후 BBW={latest_bbw:.4f}, 하단 돌파, 거래량 {latest_volume/latest_volume_sma:.2f}배")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.70,  # 최적 세팅으로 신뢰도 조정
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"볼륨 스퀴즈 전략 분석 실패: {e}")
            return None
