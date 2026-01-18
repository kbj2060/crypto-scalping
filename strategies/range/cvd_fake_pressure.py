"""
5. CVD / Delta 역반전 전략 (Fake Pressure)
횡보장에서 가격은 고정, CVD만 쌓일 때 반대 매매
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class CVDFakePressureStrategy:
    def __init__(self):
        self.name = "CVD Fake Pressure"
        self.price_change_lookback = 20  # 가격 변화량 20봉
        self.price_change_threshold = 0.006  # Price 변화량 < 0.6% (0.5~0.8% 범위에서 중간값)
        self.cvd_change_lookback = 20  # CVD 변화량 20봉 (5~20봉)
        self.cvd_change_threshold = 250000  # 250K (200K~300K 범위에서 중간값)
        self.take_profit_min = 0.002  # 0.2%
        self.take_profit_max = 0.004  # 0.4%
        self.stop_loss = 0.0025  # 0.25%
    
    def analyze(self, data_collector):
        """CVD Fake Pressure 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=100)
            if eth_data is None or len(eth_data) < self.price_change_lookback + 5:
                return None
            
            # 가격 변화량 필터: Price 변화량(20봉) < 0.6%
            recent_data = eth_data.tail(self.price_change_lookback)
            price_max = float(recent_data['high'].max())
            price_min = float(recent_data['low'].min())
            price_change_pct = (price_max - price_min) / price_min
            
            if price_change_pct >= self.price_change_threshold:
                return None  # 횡보장이 아님
            
            # CVD 계산
            cvd_data = data_collector.calculate_cvd('ETH', lookback=50, ema_period=21, delta_smoothing=5)
            if cvd_data is None or len(cvd_data) < self.cvd_change_lookback:
                return None
            
            # CVD 변화량 계산 (20봉)
            cvd_current = float(cvd_data['cvd'].iloc[-1])
            cvd_prev = float(cvd_data['cvd'].iloc[-self.cvd_change_lookback])
            cvd_change = cvd_current - cvd_prev
            
            latest = eth_data.iloc[-1]
            latest_close = float(latest['close'])
            
            signal = None
            entry_price = latest_close
            stop_loss_price = None
            take_profit_price = None
            
            # SHORT: CVD가 5~20봉 동안 과도하게 상승 → 그런데 가격은 안 올라가면 → 숏
            if cvd_change > self.cvd_change_threshold:
                signal = 'SHORT'
                stop_loss_price = entry_price * (1 + self.stop_loss)
                take_profit_price = entry_price * (1 - self.take_profit_max)  # 0.4%
                logger.info(f"CVD Fake Pressure 숏: CVD 변화={cvd_change/1000:.0f}K, 가격 변화={price_change_pct:.3%}")
            
            # LONG: CVD가 계속 하락 → 그런데 가격은 안 떨어지면 → 롱
            elif cvd_change < -self.cvd_change_threshold:
                signal = 'LONG'
                stop_loss_price = entry_price * (1 - self.stop_loss)
                take_profit_price = entry_price * (1 + self.take_profit_max)  # 0.4%
                logger.info(f"CVD Fake Pressure 롱: CVD 변화={cvd_change/1000:.0f}K, 가격 변화={price_change_pct:.3%}")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'confidence': 0.76,
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"CVD Fake Pressure 전략 분석 실패: {e}")
            return None
