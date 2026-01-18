"""
1. Bollinger Band Mean Reversion 전략
횡보장에서 가격이 중심선으로 회귀하려는 성질을 이용
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class BollingerMeanReversionStrategy:
    def __init__(self):
        self.name = "Bollinger Mean Reversion"
        self.bb_period = 20
        self.bb_std_dev = 2.0
        self.bbw_ma_period = 50  # BandWidth MA 기간
        self.rsi_period = 14
        self.atr_period = 14
        self.atr_threshold = 1.2  # ATR < 1.2
        self.macd_hist_threshold = 0.2  # MACD Histogram 절대값 < 0.2
    
    def analyze(self, data_collector):
        """볼린저 밴드 평균 회귀 전략 분석"""
        try:
            logger.debug(f"🔍 [Bollinger Mean Reversion] 전략 분석 시작")
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < max(self.bb_period, self.bbw_ma_period) + 5:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}, 필요: {max(self.bb_period, self.bbw_ma_period) + 5}")
                return None
            logger.debug(f"🔍 [Bollinger Mean Reversion] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
            # 볼린저 밴드 계산
            bb_bands = Indicators.calculate_bollinger_bands(eth_data, period=self.bb_period, std_dev=self.bb_std_dev)
            if bb_bands is None:
                return None
            
            # BBW 계산
            bbw = Indicators.calculate_bbw(bb_bands)
            if bbw is None:
                return None
            
            # BBW MA 계산
            bbw_ma = Indicators.calculate_sma(bbw, period=self.bbw_ma_period)
            if bbw_ma is None or len(bbw_ma) < 1:
                return None
            
            latest_bbw = float(bbw.iloc[-1])
            latest_bbw_ma = float(bbw_ma.iloc[-1])
            
            logger.debug(f"🔍 [Bollinger Mean Reversion] BBW 분석 - 현재: {latest_bbw:.4f}, MA: {latest_bbw_ma:.4f}, MA*1.2: {latest_bbw_ma * 1.2:.4f}")
            
            # 필터: 횡보장 판별
            # BandWidth < BandWidth_MA * 1.2 (완화: 약간의 대역폭 확장 허용)
            if latest_bbw >= latest_bbw_ma * 1.2:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] BBW 필터 미충족: {latest_bbw:.4f} >= {latest_bbw_ma * 1.2:.4f}")
                return None
            
            # ATR 필터 제거 (3분봉 기준 ATR이 1.2는 거의 불가능, 평상시 3~7 범위)
            # atr = Indicators.calculate_atr(eth_data, period=self.atr_period)
            # if atr is None:
            #     return None
            # latest_atr = float(atr.iloc[-1])
            # if latest_atr >= self.atr_threshold:
            #     return None
            
            # MACD Histogram 필터 제거 (진입 문턱 낮춤)
            
            # RSI 계산
            rsi = Indicators.calculate_rsi(eth_data, period=self.rsi_period)
            if rsi is None:
                return None
            latest_rsi = float(rsi.iloc[-1])
            
            latest = eth_data.iloc[-1]
            latest_low = float(latest['low'])
            latest_high = float(latest['high'])
            latest_close = float(latest['close'])
            
            # 볼린저 밴드 값
            bb_upper = float(bb_bands['upper'].iloc[-1])
            bb_lower = float(bb_bands['lower'].iloc[-1])
            bb_middle = float(bb_bands['middle'].iloc[-1])  # Basis = MA(20)
            
            logger.debug(f"🔍 [Bollinger Mean Reversion] 밴드 위치 - 상단: {bb_upper:.2f}, 하단: {bb_lower:.2f}, 중간: {bb_middle:.2f}, 현재가: {latest_close:.2f}")
            logger.debug(f"🔍 [Bollinger Mean Reversion] RSI: {latest_rsi:.2f} (롱 조건: <38, 숏 조건: >62)")
            
            signal = None
            entry_price = latest_close
            stop_loss = None
            take_profit_50 = None  # 1차 청산: Basis 도달 시 50%
            take_profit_100 = None  # 전체 청산: ±0.4%
            
            # LONG: Price <= LowerBB AND RSI < 38
            logger.debug(f"🔍 [Bollinger Mean Reversion] 롱 조건 체크 - 하단 터치: {latest_close <= bb_lower}, RSI < 38: {latest_rsi < 38}")
            if latest_close <= bb_lower and latest_rsi < 38:
                signal = 'LONG'
                stop_loss = entry_price * (1 - 0.0025)  # ±0.25% 손절
                take_profit_50 = bb_middle  # 1차 청산: Basis 도달
                take_profit_100 = entry_price * (1 + 0.004)  # 전체 청산: +0.4%
                logger.info(f"볼린저 평균 회귀 롱: 하단 밴드 터치, RSI={latest_rsi:.2f}, BBW={latest_bbw:.4f}")
            elif latest_close <= bb_lower:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] 하단 터치했으나 RSI 조건 미충족: {latest_rsi:.2f} >= 38")
            elif latest_rsi < 38:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] RSI 조건 충족했으나 하단 미터치: {latest_close:.2f} > {bb_lower:.2f}")
            
            # SHORT: Price >= UpperBB AND RSI > 62
            logger.debug(f"🔍 [Bollinger Mean Reversion] 숏 조건 체크 - 상단 터치: {latest_close >= bb_upper}, RSI > 62: {latest_rsi > 62}")
            if latest_close >= bb_upper and latest_rsi > 62:
                signal = 'SHORT'
                stop_loss = entry_price * (1 + 0.0025)  # ±0.25% 손절
                take_profit_50 = bb_middle  # 1차 청산: Basis 도달
                take_profit_100 = entry_price * (1 - 0.004)  # 전체 청산: -0.4%
                logger.info(f"볼린저 평균 회귀 숏: 상단 밴드 터치, RSI={latest_rsi:.2f}, BBW={latest_bbw:.4f}")
            elif latest_close >= bb_upper:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] 상단 터치했으나 RSI 조건 미충족: {latest_rsi:.2f} <= 62")
            elif latest_rsi > 62:
                logger.debug(f"⚠️ [Bollinger Mean Reversion] RSI 조건 충족했으나 상단 미터치: {latest_close:.2f} < {bb_upper:.2f}")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit_50': take_profit_50,
                    'take_profit_100': take_profit_100,
                    'confidence': 0.75,  # 필터 추가로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"볼린저 평균 회귀 전략 분석 실패: {e}")
            return None
