"""
4. Stoch RSI Mean-Reversion 전략
Stochastic RSI를 이용한 과매수/과매도 반전 매매
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class StochRSIMeanReversionStrategy:
    def __init__(self):
        self.name = "Stoch RSI Mean Reversion"
        self.rsi_period = 14
        self.stoch_period = 14
        self.k_period = 3
        self.d_period = 3
        self.stoch_oversold = 40  # StochRSI K < 40 (완화: 35 -> 40)
        self.stoch_overbought = 60  # StochRSI K > 60 (완화: 65 -> 60)
        self.rsi_neutral_min = 35  # RSI between 35 and 65
        self.rsi_neutral_max = 65  # RSI between 35 and 65
        self.rsi_long_max = 55  # RSI < 55
        self.rsi_short_min = 45  # RSI > 45
        self.macd_hist_threshold = 9999  # MACD 필터 완전 무력화 (완화: 1.0 -> 9999)
        self.take_profit_min = 0.002  # 0.2%
        self.take_profit_max = 0.0035  # 0.35%
        self.stop_loss = 0.002  # 0.2%
    
    def analyze(self, data_collector):
        """Stoch RSI Mean-Reversion 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                return None
            
            # RSI 계산
            rsi = Indicators.calculate_rsi(eth_data, period=self.rsi_period)
            if rsi is None:
                return None
            latest_rsi = float(rsi.iloc[-1])
            
            # 필터: RSI between 35 and 65 (중립 구간)
            if latest_rsi < self.rsi_neutral_min or latest_rsi > self.rsi_neutral_max:
                return None
            
            # MACD Histogram 필터
            macd = Indicators.calculate_macd(eth_data)
            if macd is None:
                return None
            latest_macd_hist = abs(float(macd['histogram'].iloc[-1]))
            if latest_macd_hist >= self.macd_hist_threshold:
                return None
            
            # Stochastic RSI 계산
            stoch_rsi = Indicators.calculate_stoch_rsi(
                eth_data, 
                rsi_period=self.rsi_period,
                stoch_period=self.stoch_period,
                k_period=self.k_period,
                d_period=self.d_period
            )
            if stoch_rsi is None:
                return None
            
            # stoch_rsi는 dict 형태 {'k': Series, 'd': Series}
            if not isinstance(stoch_rsi, dict) or 'k' not in stoch_rsi:
                logger.error(f"Stoch RSI 계산 결과 형식 오류: {type(stoch_rsi)}")
                return None
            
            latest_stoch_k = float(stoch_rsi['k'].iloc[-1])
            
            latest = eth_data.iloc[-1]
            latest_close = float(latest['close'])
            
            signal = None
            entry_price = latest_close
            stop_loss_price = None
            take_profit_price = None
            
            # LONG: StochRSI K < 35 AND RSI < 55 (공격적: 기존 K<20, RSI<48 → K<35, RSI<55)
            if latest_stoch_k < self.stoch_oversold and latest_rsi < self.rsi_long_max:
                signal = 'LONG'
                stop_loss_price = entry_price * (1 - self.stop_loss)
                take_profit_price = entry_price * (1 + self.take_profit_max)  # 0.35%
            
            # SHORT: StochRSI K > 65 AND RSI > 45 (공격적: 기존 K>80, RSI>52 → K>65, RSI>45)
            elif latest_stoch_k > self.stoch_overbought and latest_rsi > self.rsi_short_min:
                signal = 'SHORT'
                stop_loss_price = entry_price * (1 + self.stop_loss)
                take_profit_price = entry_price * (1 - self.take_profit_max)  # 0.35%
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'confidence': 0.74,
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            import traceback
            logger.error(f"Stoch RSI Mean-Reversion 전략 분석 실패: {e}")
            logger.error(f"에러 상세: {traceback.format_exc()}")
            return None
