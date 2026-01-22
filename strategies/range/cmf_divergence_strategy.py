"""
CMF (Chaikin Money Flow) 다이버전스 전략
가격과 자금 흐름의 괴리(Absorption) 탐지 - 횡보권 하단/상단에서의 반전 포착
"""
import logging
import sys
import os
import pandas as pd
import numpy as np
# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class CMFDivergenceStrategy:
    def __init__(self):
        self.name = "CMF Divergence"
        self.period = 20
        self.range_threshold = 0.15  # 0.15 이상이면 매집 완료로 판단
    
    def calculate_cmf(self, data, period=20):
        """CMF (Chaikin Money Flow) 계산"""
        try:
            if len(data) < period:
                return None
            
            # Money Flow Multiplier 계산
            # (Close - Low) - (High - Close) / (High - Low)
            high_low_diff = data['high'] - data['low']
            high_low_diff = high_low_diff.replace(0, np.nan)  # 0으로 나누기 방지
            
            mf_multiplier = ((data['close'] - data['low']) - (data['high'] - data['close'])) / high_low_diff
            
            # Money Flow Volume 계산
            mf_volume = mf_multiplier * data['volume']
            
            # CMF 계산: MF Volume의 합 / Volume의 합
            mf_volume_sum = mf_volume.rolling(window=period).sum()
            volume_sum = data['volume'].rolling(window=period).sum()
            
            volume_sum = volume_sum.replace(0, np.nan)  # 0으로 나누기 방지
            cmf = mf_volume_sum / volume_sum
            
            return cmf
        except Exception as e:
            logger.error(f"CMF 계산 실패: {e}")
            return None
    
    def analyze(self, data_collector):
        """CMF Divergence 전략 분석"""
        try:
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < self.period + 1:
                return None
            
            # CMF 계산
            cmf = self.calculate_cmf(eth_data, period=self.period)
            if cmf is None or len(cmf) < 2:
                return None
            
            curr_cmf = float(cmf.iloc[-1])
            prev_cmf = float(cmf.iloc[-2])
            
            latest = eth_data.iloc[-1]
            entry_price = float(latest['close'])
            
            signal = None
            stop_loss_price = None
            take_profit_price = None
            
            # 다이버전스 대신 Zero-Cross 전략으로 변경
            # CMF가 음수에서 양수로 가면 롱 (0선 돌파)
            if prev_cmf < 0 and curr_cmf > 0:
                signal = 'LONG'
                stop_loss_price = entry_price * (1 - 0.0025)  # 0.25% 손절
                take_profit_price = entry_price * (1 + 0.004)  # 0.4% 익절
                logger.debug(f"🔍 [CMF Zero-Cross] 롱 신호 발생 - CMF: {curr_cmf:.4f} (이전: {prev_cmf:.4f}, 0선 돌파)")
            
            # CMF가 양수에서 음수로 가면 숏 (0선 이탈)
            elif prev_cmf > 0 and curr_cmf < 0:
                signal = 'SHORT'
                stop_loss_price = entry_price * (1 + 0.0025)  # 0.25% 손절
                take_profit_price = entry_price * (1 - 0.004)  # 0.4% 익절
                logger.debug(f"🔍 [CMF Zero-Cross] 숏 신호 발생 - CMF: {curr_cmf:.4f} (이전: {prev_cmf:.4f}, 0선 이탈)")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss_price,
                    'take_profit': take_profit_price,
                    'confidence': 0.78,
                    'strategy': self.name,
                    'cmf': curr_cmf  # AI 학습용 추가 정보
                }
            
            return None
            
        except Exception as e:
            logger.error(f"CMF Divergence 전략 분석 실패: {e}")
            return None
