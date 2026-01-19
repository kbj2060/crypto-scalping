"""
6. BTC 연동 모멘텀 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class BTCEthCorrelationStrategy:
    def __init__(self):
        self.name = "BTC/ETH Correlation"
        self.rsi_long_threshold = 49  # BTC RSI < 49 → ETH 숏 bias (공격적: 45 -> 49)
        self.rsi_short_threshold = 51  # BTC RSI > 51 → ETH 롱 bias (공격적: 55 -> 51)
        self.ma_period = 20
        self.ma_consecutive = 1  # MA20 위/아래 1봉 연속 (공격적: 기존 2 → BTC 돌파 즉시)
    
    def analyze(self, data_collector):
        """BTC 연동 모멘텀 전략 분석 (최적 세팅)"""
        try:
            logger.debug(f"🔍 [BTC/ETH Correlation] 전략 분석 시작")
            btc_data = data_collector.get_candles('BTC', count=50)
            eth_data = data_collector.get_candles('ETH', count=50)
            
            if btc_data is None or eth_data is None:
                logger.debug(f"⚠️ [BTC/ETH Correlation] 데이터 없음: btc={btc_data is not None}, eth={eth_data is not None}")
                return None
            
            if len(btc_data) < 25 or len(eth_data) < 25:
                logger.debug(f"⚠️ [BTC/ETH Correlation] 데이터 부족: btc={len(btc_data)}, eth={len(eth_data)}")
                return None
            logger.debug(f"🔍 [BTC/ETH Correlation] 데이터 확인 완료: BTC {len(btc_data)}개, ETH {len(eth_data)}개 캔들")
            
            # BTC 지표 계산
            btc_rsi = Indicators.calculate_rsi(btc_data, period=14)
            btc_ma = Indicators.calculate_sma(btc_data['close'], period=self.ma_period)
            
            if btc_rsi is None or btc_ma is None:
                return None
            
            # BTC RSI 및 MA20 확인
            btc_rsi_latest = float(btc_rsi.iloc[-1])
            btc_current = btc_data.iloc[-1]
            btc_ma_current = float(btc_ma.iloc[-1])
            btc_price_current = float(btc_current['close'])
            
            logger.debug(f"🔍 [BTC/ETH Correlation] BTC 분석 - 가격: {btc_price_current:.2f}, MA20: {btc_ma_current:.2f}, RSI: {btc_rsi_latest:.2f}")
            
            # BTC 임펄스 필터: MA20 위/아래 1봉 연속
            btc_above_ma = True
            btc_below_ma = True
            for i in range(1, self.ma_consecutive + 1):
                if len(btc_data) >= i + 1 and len(btc_ma) >= i + 1:
                    btc_price = float(btc_data.iloc[-i]['close'])
                    btc_ma_val = float(btc_ma.iloc[-i])
                    if btc_price <= btc_ma_val:
                        btc_above_ma = False
                    if btc_price >= btc_ma_val:
                        btc_below_ma = False
                    logger.debug(f"🔍 [BTC/ETH Correlation] {i}봉 전 - 가격: {btc_price:.2f}, MA20: {btc_ma_val:.2f}, 위: {btc_price > btc_ma_val}, 아래: {btc_price < btc_ma_val}")
            
            logger.debug(f"🔍 [BTC/ETH Correlation] MA20 필터 - 위 {self.ma_consecutive}봉 연속: {btc_above_ma}, 아래 {self.ma_consecutive}봉 연속: {btc_below_ma}")
            
            eth_current = eth_data.iloc[-1]
            entry_price = float(eth_current['close'])
            signal = None
            
            # 롱 bias: BTC RSI > 55 AND BTC MA20 위 2봉 연속
            if btc_rsi_latest > self.rsi_short_threshold and btc_above_ma:
                # ETH가 상승 추세인지 확인
                eth_prev = eth_data.iloc[-2] if len(eth_data) >= 2 else None
                if eth_prev is not None:
                    eth_current_close = float(eth_current['close'])
                    eth_prev_close = float(eth_prev['close'])
                    if eth_current_close > eth_prev_close:
                        signal = 'LONG'
            
            # 숏 bias: BTC RSI < 45 AND BTC MA20 아래 2봉 연속
            elif btc_rsi_latest < self.rsi_long_threshold and btc_below_ma:
                # ETH가 하락 추세인지 확인
                eth_prev = eth_data.iloc[-2] if len(eth_data) >= 2 else None
                if eth_prev is not None:
                    eth_current_close = float(eth_current['close'])
                    eth_prev_close = float(eth_prev['close'])
                    if eth_current_close < eth_prev_close:
                        signal = 'SHORT'
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': 0.78,  # 최적 세팅으로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"BTC/ETH 상관 전략 분석 실패: {e}")
            return None
