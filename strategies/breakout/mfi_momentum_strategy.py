"""
MFI (Money Flow Index) 모멘텀 전략
공격적인 자금 유입 탐지 - 가격 상승 시 거래량이 동반되는 진짜 돌파인지 구분
"""
import logging
import sys
import os
import pandas as pd
import numpy as np
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class MFIMomentumStrategy:
    def __init__(self):
        self.name = "MFI Momentum"
        self.period = 14
        self.upper_threshold = 65  # 과매수 진입 (완화: 80 -> 65)
        self.lower_threshold = 35  # 과매도 진입 (완화: 20 -> 35)
    
    def calculate_mfi(self, data, period=14):
        """MFI (Money Flow Index) 계산"""
        try:
            if len(data) < period + 1:
                return None
            
            # Typical Price 계산
            tp = (data['high'] + data['low'] + data['close']) / 3
            
            # Raw Money Flow 계산
            rmf = tp * data['volume']
            
            # Positive/Negative Money Flow 계산
            positive_flow = pd.Series(0.0, index=data.index)
            negative_flow = pd.Series(0.0, index=data.index)
            
            # Typical Price 변화량
            tp_diff = tp.diff()
            
            # Positive flow: 가격 상승 시
            positive_flow = rmf.where(tp_diff > 0, 0)
            
            # Negative flow: 가격 하락 시
            negative_flow = rmf.where(tp_diff < 0, 0)
            
            # Rolling sum 계산
            positive_flow_sum = positive_flow.rolling(window=period).sum()
            negative_flow_sum = negative_flow.rolling(window=period).sum()
            
            # Money Flow Ratio 계산 (0으로 나누기 방지)
            negative_flow_sum_safe = negative_flow_sum.replace(0, np.nan)
            mfr = positive_flow_sum / negative_flow_sum_safe
            
            # MFI 계산 (0~100 범위)
            # mfr가 NaN이면 MFI도 NaN으로 유지
            mfi = 100 - (100 / (1 + mfr))
            
            # 첫 period개는 NaN이므로 그대로 유지
            
            return mfi
        except Exception as e:
            logger.error(f"MFI 계산 실패: {e}")
            return None
    
    def analyze(self, data_collector):
        """MFI Momentum 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < self.period + 1:
                return None
            
            # MFI 계산
            mfi = self.calculate_mfi(eth_data, period=self.period)
            if mfi is None or len(mfi) < 1:
                return None
            
            current_mfi = float(mfi.iloc[-1])
            prev_mfi = float(mfi.iloc[-2]) if len(mfi) >= 2 else 50.0
            
            latest = eth_data.iloc[-1]
            entry_price = float(latest['close'])
            
            signal = None
            
            # 과매수 구간: 강한 돌파 (LONG)
            if current_mfi > self.upper_threshold:
                signal = 'LONG'
                logger.debug(f"🔍 [MFI Momentum] 롱 신호 발생 - MFI: {current_mfi:.2f}")
            
            # 과매도 구간: 강한 이탈 (SHORT)
            elif current_mfi < self.lower_threshold:
                signal = 'SHORT'
                logger.debug(f"🔍 [MFI Momentum] 숏 신호 발생 - MFI: {current_mfi:.2f}")
            
            # 중심선(50) 돌파 전략 추가
            elif current_mfi > 50 and prev_mfi <= 50:
                signal = 'LONG'
                logger.debug(f"🔍 [MFI Momentum] 롱 신호 발생 (중심선 돌파) - MFI: {current_mfi:.2f}")
            elif current_mfi < 50 and prev_mfi >= 50:
                signal = 'SHORT'
                logger.debug(f"🔍 [MFI Momentum] 숏 신호 발생 (중심선 이탈) - MFI: {current_mfi:.2f}")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.82,
                    'strategy': self.name,
                    'mfi': current_mfi  # AI 학습용 추가 정보
                }
            
            return None
            
        except Exception as e:
            logger.error(f"MFI Momentum 전략 분석 실패: {e}")
            return None
