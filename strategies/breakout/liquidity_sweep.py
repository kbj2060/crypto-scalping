"""
2. 유동성 스윕(Liquidity Sweep) 전략 최적화 (1500봉 기준)
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class LiquiditySweepStrategy:
    def __init__(self):
        self.name = "Liquidity Sweep"
        self.liquidity_lookback = 20  # 고점/저점 비교 봉수: 20봉
        self.sweep_body_max_pct = 0.4  # 스윕 바 길이 < 전체의 40%
        self.stop_loss_percent = 0.0015  # 스윕 extremum ±0.15%
    
    def analyze(self, data_collector):
        """유동성 스윕 전략 분석 (최적 세팅)"""
        try:
            logger.debug(f"🔍 [Liquidity Sweep] 전략 분석 시작")
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < self.liquidity_lookback + 5:
                logger.debug(f"⚠️ [Liquidity Sweep] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}, 필요: {self.liquidity_lookback + 5}")
                return None
            logger.debug(f"🔍 [Liquidity Sweep] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
            # 이전 20봉 안의 주요 유동성 찾기 (현재 캔들 제외)
            recent_data = eth_data.iloc[:-1].tail(self.liquidity_lookback)
            major_high = float(recent_data['high'].max())
            major_low = float(recent_data['low'].min())
            
            # 최신 캔들들
            latest = eth_data.iloc[-1]
            prev_candle = eth_data.iloc[-2] if len(eth_data) >= 2 else None
            prev_prev_candle = eth_data.iloc[-3] if len(eth_data) >= 3 else None
            
            signal = None
            stop_loss = None
            entry_price = float(latest['close'])
            
            # 최신 캔들 값들을 float로 변환
            latest_high = float(latest['high'])
            latest_low = float(latest['low'])
            latest_close = float(latest['close'])
            latest_open = float(latest['open'])
            
            logger.debug(f"🔍 [Liquidity Sweep] 주요 고점/저점 - 고점: {major_high:.2f}, 저점: {major_low:.2f}")
            logger.debug(f"🔍 [Liquidity Sweep] 현재 캔들 - 고가: {latest_high:.2f}, 저가: {latest_low:.2f}, 종가: {latest_close:.2f}")
            
            # 고점 스윕: 이전 20봉 고점 돌파 → 되돌림 마감 → 진입
            if latest_high > major_high:
                # 스윕 바 길이(body) < 전체의 40%
                sweep_body = abs(latest_close - latest_open)
                sweep_range = latest_high - latest_low
                body_ratio = sweep_body / sweep_range if sweep_range > 0 else 0
                
                logger.debug(f"🔍 [Liquidity Sweep] 고점 스윕 체크 - 고점 돌파: {latest_high > major_high}, 바디 비율: {body_ratio:.2%}, 필요: <{self.sweep_body_max_pct:.2%}")
                
                if body_ratio < self.sweep_body_max_pct:
                    # 스윕 후 반전 캔들 1개 확인 (되돌림 마감)
                    if prev_candle is not None and latest_close < major_high:
                        signal = 'SHORT'
                        stop_loss = major_high * (1 + self.stop_loss_percent)
                        logger.info(f"Bearish Sweep: 고점 {major_high:.2f} 돌파 후 되돌림")
                    else:
                        logger.debug(f"⚠️ [Liquidity Sweep] 고점 돌파했으나 되돌림 미확인 (종가: {latest_close:.2f}, 고점: {major_high:.2f})")
                else:
                    logger.debug(f"⚠️ [Liquidity Sweep] 고점 돌파했으나 바디 비율 과다: {body_ratio:.2%} >= {self.sweep_body_max_pct:.2%}")
            
            # 저점 스윕: 이전 20봉 저점 이탈 → 되돌림 마감 → 진입
            if latest_low < major_low:
                # 스윕 바 길이(body) < 전체의 40%
                sweep_body = abs(latest_close - latest_open)
                sweep_range = latest_high - latest_low
                body_ratio = sweep_body / sweep_range if sweep_range > 0 else 0
                
                logger.debug(f"🔍 [Liquidity Sweep] 저점 스윕 체크 - 저점 이탈: {latest_low < major_low}, 바디 비율: {body_ratio:.2%}, 필요: <{self.sweep_body_max_pct:.2%}")
                
                if body_ratio < self.sweep_body_max_pct:
                    # 스윕 후 반전 캔들 1개 확인 (되돌림 마감)
                    if prev_candle is not None and latest_close > major_low:
                        signal = 'LONG'
                        stop_loss = major_low * (1 - self.stop_loss_percent)
                        logger.info(f"Bullish Sweep: 저점 {major_low:.2f} 이탈 후 되돌림")
                    else:
                        logger.debug(f"⚠️ [Liquidity Sweep] 저점 이탈했으나 되돌림 미확인 (종가: {latest_close:.2f}, 저점: {major_low:.2f})")
                else:
                    logger.debug(f"⚠️ [Liquidity Sweep] 저점 이탈했으나 바디 비율 과다: {body_ratio:.2%} >= {self.sweep_body_max_pct:.2%}")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': stop_loss,
                    'confidence': 0.75,  # 최적 세팅으로 신뢰도 향상
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"유동성 스윕 전략 분석 실패: {e}")
            return None
