"""
Hull Moving Average (HMA) Momentum 전략
HMA는 지연 시간을 최소화하면서도 가격 곡선을 매끄럽게 만드는 지표입니다.
기울기와 돌파를 동시에 확인하여 진입 타점을 잡습니다.
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


class HMAMomentumStrategy:
    def __init__(self):
        self.name = "HMA Momentum"
        self.period = 14  # HMA 기간
    
    def analyze(self, data_collector):
        """HMA Momentum 전략 분석"""
        try:
            logger.debug(f"🔍 [HMA Momentum] 전략 분석 시작")
            
            # 충분한 데이터 확보 (HMA 계산을 위해 period + sqrt(period) 정도 필요)
            df = data_collector.get_candles('ETH', count=self.period + 10)
            if df is None or len(df) < self.period + 2:
                logger.debug(f"[HMA Momentum] 데이터 부족: {len(df) if df is not None else 0}개 캔들")
                return None
            
            # HMA 계산
            hma = Indicators.calculate_hma(df, period=self.period)
            if hma is None or len(hma) < 2:
                logger.debug("[HMA Momentum] HMA 계산 실패")
                return None
            
            # 현재 및 이전 HMA 값
            current_hma = float(hma.iloc[-1])
            prev_hma = float(hma.iloc[-2])
            
            # 현재 종가
            current_close = float(df['close'].iloc[-1])
            
            # 기울기 계산
            hma_slope = current_hma - prev_hma
            
            signal = None
            
            # LONG 조건: Price > HMA이면 무조건 롱 스코어 부여 (기울기 조건 제거)
            if current_close > current_hma:
                logger.debug(f"[HMA Momentum] 롱 신호 발생 - HMA: {current_hma:.2f}, 기울기: {hma_slope:.4f}, 종가: {current_close:.2f}")
                signal = 'LONG'
            
            # SHORT 조건: Price < HMA이면 숏 (기울기 조건 제거)
            elif current_close < current_hma:
                logger.debug(f"[HMA Momentum] 숏 신호 발생 - HMA: {current_hma:.2f}, 기울기: {hma_slope:.4f}, 종가: {current_close:.2f}")
                signal = 'SHORT'
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': current_close,
                    'stop_loss': None,
                    'confidence': 0.75,  # HMA Momentum 신뢰도
                    'strategy': self.name,
                    'hma': current_hma,  # 추가 정보
                    'hma_slope': hma_slope  # 추가 정보
                }
            
            return None
            
        except Exception as e:
            logger.error(f"HMA Momentum 전략 분석 실패: {e}")
            import traceback
            logger.error(f"에러 상세 정보:\n{traceback.format_exc()}")
            return None
