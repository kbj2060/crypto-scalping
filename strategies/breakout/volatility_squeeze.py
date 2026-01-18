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
        self.bbw_squeeze = 0.08  # BBW < 0.08 → 스퀴즈 (완화: 기존 0.06 → 수축 인정 범위 확대)
        self.bbw_explosion = 0.007  # BBW > 0.007 (0.7%) → 폭발 (0.5%~1.0% 범위에서 중간값)
        self.volume_explosion = 1.1  # 거래량 1.1배 이상 (완화: 기존 1.3 → 신호 빈도 증가)
    
    def analyze(self, data_collector):
        """볼륨 스퀴즈 전략 분석 (최적 세팅)"""
        try:
            logger.debug(f"🔍 [Volatility Squeeze] 전략 분석 시작")
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                logger.debug(f"⚠️ [Volatility Squeeze] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}")
                return None
            logger.debug(f"🔍 [Volatility Squeeze] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
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
            
            # 최근 10봉 이내에 스퀴즈(0.08 미만)가 있었는지 확인 (완화: 기존 0.06)
            was_squeezed = (bbw.tail(10) < self.bbw_squeeze).any()
            # 현재 BBW가 폭발 조건을 만족하는지 확인 (이전 봉 대비 증가 + explosion 임계값 초과)
            if len(bbw) >= 2:
                prev_bbw = float(bbw.iloc[-2])
                latest_bbw_val = float(latest_bbw)
                is_exploding = latest_bbw_val > prev_bbw and latest_bbw_val > self.bbw_explosion  # 상승 전환 + 폭발 임계값 초과
            else:
                is_exploding = False
            
            prev_bbw_str = f"{prev_bbw:.4f}" if len(bbw) >= 2 else 'N/A'
            logger.debug(f"🔍 [Volatility Squeeze] BBW 분석 - 현재: {latest_bbw_val:.4f}, 이전: {prev_bbw_str}, 스퀴즈 임계값: {self.bbw_squeeze:.4f}, 폭발 임계값: {self.bbw_explosion:.4f}")
            logger.debug(f"🔍 [Volatility Squeeze] 조건 체크 - 과거 스퀴즈: {was_squeezed}, 현재 폭발: {is_exploding}")
            logger.debug(f"🔍 [Volatility Squeeze] 거래량 - 현재: {latest_volume:.0f}, 평균: {latest_volume_sma:.0f}, 배수: {latest_volume/latest_volume_sma:.2f}, 필요: {self.volume_explosion:.1f}")
            
            if was_squeezed and is_exploding:
                upper_band = bb['upper'].iloc[-1]
                lower_band = bb['lower'].iloc[-1]
                latest_close = float(latest['close'])
                
                logger.debug(f"🔍 [Volatility Squeeze] 밴드 위치 - 상단: {upper_band:.2f}, 하단: {lower_band:.2f}, 현재가: {latest_close:.2f}")
                logger.debug(f"🔍 [Volatility Squeeze] 돌파 체크 - 상단 돌파: {latest_close > upper_band}, 하단 돌파: {latest_close < lower_band}")
                
                # 폭발 양봉: 상단 돌파 + 거래량
                if (latest_close > upper_band and 
                    latest_volume >= latest_volume_sma * self.volume_explosion):
                    signal = 'LONG'
                    logger.info(f"스퀴즈 폭발 Long: 이전 스퀴즈 후 BBW={latest_bbw_val:.4f}, 상단 돌파, 거래량 {latest_volume/latest_volume_sma:.2f}배")
                elif latest_close > upper_band:
                    logger.debug(f"⚠️ [Volatility Squeeze] 상단 돌파했으나 거래량 부족: {latest_volume/latest_volume_sma:.2f}배 < {self.volume_explosion:.1f}배 필요")
                
                # 폭발 음봉: 하단 돌파 + 거래량
                if (latest_close < lower_band and 
                      latest_volume >= latest_volume_sma * self.volume_explosion):
                    signal = 'SHORT'
                    logger.info(f"스퀴즈 폭발 Short: 이전 스퀴즈 후 BBW={latest_bbw_val:.4f}, 하단 돌파, 거래량 {latest_volume/latest_volume_sma:.2f}배")
                elif latest_close < lower_band:
                    logger.debug(f"⚠️ [Volatility Squeeze] 하단 돌파했으나 거래량 부족: {latest_volume/latest_volume_sma:.2f}배 < {self.volume_explosion:.1f}배 필요")
            else:
                if not was_squeezed:
                    logger.debug(f"⚠️ [Volatility Squeeze] 과거 10봉 내 스퀴즈 없음 (최소 BBW: {float(bbw.tail(10).min()):.4f})")
                if not is_exploding:
                    logger.debug(f"⚠️ [Volatility Squeeze] 폭발 조건 미충족 (BBW 증가: {latest_bbw_val > prev_bbw if len(bbw) >= 2 else False}, 폭발 임계값: {latest_bbw_val > self.bbw_explosion})")
            
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
