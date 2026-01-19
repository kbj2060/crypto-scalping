"""
2. VWAP Deviation Mean Reversion 전략
VWAP로부터의 편차를 이용한 평균 회귀 매매
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class VWAPDeviationStrategy:
    def __init__(self):
        self.name = "VWAP Deviation"
        self.deviation_long_threshold = -0.10  # -0.10% 이하 (공격적: 기존 -0.15% → -0.10%)
        self.deviation_short_threshold = 0.10  # +0.10% 이상 (공격적: 기존 +0.15% → +0.10%)
        self.atr_period = 14
        self.atr_threshold = 5.0  # ATR < 5.0 (공격적: 시장 변동성이 있어도 평균 회귀 분석)
        self.cvd_change_period = 5  # CVD 변화량 5봉
        self.cvd_change_threshold = 200000  # 200K 미만
        self.volume_sma_period = 20
    
    def analyze(self, data_collector):
        """VWAP Deviation Mean Reversion 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < 50:
                return None
            
            # VWAP 계산
            vwap = Indicators.calculate_vwap(eth_data)
            if vwap is None or len(vwap) < 1:
                return None
            
            latest = eth_data.iloc[-1]
            latest_close = float(latest['close'])
            latest_vwap = float(vwap.iloc[-1])
            
            # Deviation 계산 (백분율)
            deviation = ((latest_close - latest_vwap) / latest_vwap) * 100
            
            # 필터: ATR < 5.0
            atr = Indicators.calculate_atr(eth_data, period=self.atr_period)
            if atr is None:
                return None
            latest_atr = float(atr.iloc[-1])
            if latest_atr >= self.atr_threshold:
                return None
            
            # 필터: CVD 변화량(5봉) < 200K
            cvd_data = data_collector.calculate_cvd('ETH', lookback=20, ema_period=21, delta_smoothing=5)
            if cvd_data is not None and len(cvd_data) >= self.cvd_change_period:
                cvd_change = abs(float(cvd_data['cvd'].iloc[-1] - cvd_data['cvd'].iloc[-self.cvd_change_period]))
                if cvd_change >= self.cvd_change_threshold:
                    return None
            
            # 거래량 필터 제거 (반전 시점에는 거래량이 늘어날 수 있으므로)
            
            signal = None
            entry_price = latest_close
            stop_loss = None
            take_profit = latest_vwap  # 목표: VWAP 터치
            
            # LONG: Devi <= -0.3%
            if deviation <= self.deviation_long_threshold:
                signal = 'LONG'
                stop_loss = entry_price * (1 - 0.0025)  # ±0.25% 손절
            
            # SHORT: Devi >= +0.3%
            elif deviation >= self.deviation_short_threshold:
                signal = 'SHORT'
                stop_loss = entry_price * (1 + 0.0025)  # ±0.25% 손절
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'confidence': 0.72,
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"VWAP 편차 전략 분석 실패: {e}")
            return None
