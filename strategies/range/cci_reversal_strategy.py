"""
CCI (Commodity Channel Index) 반전 전략
고빈도 매매용: 과매수/과매도 구간 탈출 시 즉각 신호 발생
"""
import logging
import sys
import os
import pandas as pd
import numpy as np

# 프로젝트 루트 경로 추가 (range 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

logger = logging.getLogger(__name__)


class CCIReversalStrategy:
    def __init__(self):
        self.name = "CCI Reversal"
        self.period = 14
        self.overbought = 100  # +100 하향 돌파 시 숏
        self.oversold = -100   # -100 상향 돌파 시 롱
    
    def calculate_cci(self, data, period=14):
        """CCI 지표 자체 계산 (Indicators 의존성 제거)"""
        try:
            tp = (data['high'] + data['low'] + data['close']) / 3
            sma = tp.rolling(window=period).mean()
            mad = (tp - sma).abs().rolling(window=period).mean()
            
            # 0으로 나누기 방지
            cci = (tp - sma) / (0.015 * mad + 1e-8)
            return cci
        except Exception as e:
            logger.error(f"CCI 계산 오류: {e}")
            return None

    def analyze(self, data_collector):
        try:
            # CCI 계산을 위해 충분한 데이터 가져오기
            df = data_collector.get_candles('ETH', count=50)
            if df is None or len(df) < self.period + 2:
                return None
            
            # CCI 계산
            cci_series = self.calculate_cci(df, self.period)
            if cci_series is None:
                return None
            
            curr_cci = float(cci_series.iloc[-1])
            prev_cci = float(cci_series.iloc[-2])
            
            latest = df.iloc[-1]
            entry_price = float(latest['close'])
            signal = None
            
            # LONG 조건: -100 아래에 있다가 위로 뚫고 올라옴 (Turnaround)
            if prev_cci < self.oversold and curr_cci >= self.oversold:
                signal = 'LONG'
                logger.debug(f"🔍 [CCI Reversal] 롱 신호 발생 - CCI: {curr_cci:.2f} (이전: {prev_cci:.2f})")
            
            # SHORT 조건: +100 위에 있다가 아래로 뚫고 내려옴 (Turnaround)
            elif prev_cci > self.overbought and curr_cci <= self.overbought:
                signal = 'SHORT'
                logger.debug(f"🔍 [CCI Reversal] 숏 신호 발생 - CCI: {curr_cci:.2f} (이전: {prev_cci:.2f})")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,  # 스캘핑이라 손절은 AI나 리스크 매니저에 위임
                    'confidence': 0.60,  # 빈도가 높으므로 신뢰도는 보통으로 설정
                    'strategy': self.name,
                    'cci': curr_cci
                }
            
            return None

        except Exception as e:
            logger.error(f"CCI 전략 분석 실패: {e}")
            return None
