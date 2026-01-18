"""
5. 펀딩비 극단 전략
"""
import logging
import sys
import os
# 프로젝트 루트 경로 추가 (breakout 디렉토리에서 2단계 위로)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.indicators import Indicators

logger = logging.getLogger(__name__)


class FundingRateStrategy:
    def __init__(self):
        self.name = "Funding Rate"
        self.funding_threshold = 0.00009  # 0.009% (0.01% 경계값 문제 해결)
    
    def analyze(self, data_collector):
        """펀딩비 전략 분석"""
        try:
            logger.debug(f"🔍 [Funding Rate] 전략 분석 시작")
            eth_funding = data_collector.eth_funding_rate
            if eth_funding is None:
                logger.debug(f"⚠️ [Funding Rate] 펀딩비 데이터 없음")
                return None
            
            eth_data = data_collector.get_candles('ETH', count=50)
            if eth_data is None or len(eth_data) < 30:
                logger.debug(f"⚠️ [Funding Rate] 데이터 부족: eth_data={eth_data is not None}, len={len(eth_data) if eth_data is not None else 0}")
                return None
            logger.debug(f"🔍 [Funding Rate] 데이터 확인 완료: {len(eth_data)}개 캔들")
            
            logger.debug(f"🔍 [Funding Rate] 펀딩비: {eth_funding:.6f} ({eth_funding*100:.4f}%), 임계값: ±{self.funding_threshold:.6f} (±{self.funding_threshold*100:.4f}%)")
            
            # CVD 및 델타 확인 (최적 세팅)
            cvd_data = data_collector.calculate_cvd('ETH', lookback=50, ema_period=21, delta_smoothing=5)
            
            latest = eth_data.iloc[-1]
            entry_price = latest['close']
            signal = None
            
            # Long 유리: Funding <= -0.009% AND 델타·CVD가 반대 방향 (상승)
            if eth_funding <= -self.funding_threshold:
                logger.debug(f"🔍 [Funding Rate] 롱 조건 체크 - 펀딩비: {eth_funding:.6f} <= -{self.funding_threshold:.6f}")
                if cvd_data is not None:
                    # 스무딩된 델타 사용
                    latest_delta = cvd_data.get('delta_smooth', cvd_data['delta']).iloc[-1]
                    cvd_current = cvd_data['cvd_ema'].iloc[-1]
                    cvd_prev = cvd_data['cvd_ema'].iloc[-2] if len(cvd_data) >= 2 else None
                    
                    # 델타·CVD가 반대 방향 (상승) - 펀딩비는 음수(롱 유리)이므로 CVD/델타는 양수여야 함
                    delta_opposite = latest_delta > 0  # 델타가 양수 (반대 방향)
                    cvd_opposite = (cvd_prev is not None and cvd_current > cvd_prev)  # CVD 상승 (반대 방향)
                    
                    cvd_prev_str = f"{cvd_prev:.0f}" if cvd_prev is not None else 'N/A'
                    logger.debug(f"🔍 [Funding Rate] CVD/델타 분석 - 델타: {latest_delta:.0f}, CVD 현재: {cvd_current:.0f}, CVD 이전: {cvd_prev_str}")
                    logger.debug(f"🔍 [Funding Rate] 반대 방향 체크 - 델타 양수: {delta_opposite}, CVD 상승: {cvd_opposite}")
                    
                    if delta_opposite or cvd_opposite:
                        signal = 'LONG'
                        confidence = 0.75
                        if delta_opposite and cvd_opposite:
                            confidence = 0.85
                        logger.info(f"펀딩비 극단 Long: Funding={eth_funding:.4f}%, 델타/CVD 반대 방향 상승")
                    else:
                        logger.debug(f"⚠️ [Funding Rate] 펀딩비 롱 조건 충족했으나 CVD/델타 반대 방향 미충족")
                else:
                    logger.debug(f"⚠️ [Funding Rate] CVD 데이터 없음")
            
            # Short 유리: Funding >= 0.009% AND 델타·CVD가 반대 방향 (하락)
            elif eth_funding >= self.funding_threshold:
                logger.debug(f"🔍 [Funding Rate] 숏 조건 체크 - 펀딩비: {eth_funding:.6f} >= {self.funding_threshold:.6f}")
                if cvd_data is not None:
                    # 스무딩된 델타 사용
                    latest_delta = cvd_data.get('delta_smooth', cvd_data['delta']).iloc[-1]
                    cvd_current = cvd_data['cvd_ema'].iloc[-1]
                    cvd_prev = cvd_data['cvd_ema'].iloc[-2] if len(cvd_data) >= 2 else None
                    
                    # 델타·CVD가 반대 방향 (하락) - 펀딩비는 양수(숏 유리)이므로 CVD/델타는 음수여야 함
                    delta_opposite = latest_delta < 0  # 델타가 음수 (반대 방향)
                    cvd_opposite = (cvd_prev is not None and cvd_current < cvd_prev)  # CVD 하락 (반대 방향)
                    
                    cvd_prev_str = f"{cvd_prev:.0f}" if cvd_prev is not None else 'N/A'
                    logger.debug(f"🔍 [Funding Rate] CVD/델타 분석 - 델타: {latest_delta:.0f}, CVD 현재: {cvd_current:.0f}, CVD 이전: {cvd_prev_str}")
                    logger.debug(f"🔍 [Funding Rate] 반대 방향 체크 - 델타 음수: {delta_opposite}, CVD 하락: {cvd_opposite}")
                    
                    if delta_opposite or cvd_opposite:
                        signal = 'SHORT'
                        confidence = 0.75
                        if delta_opposite and cvd_opposite:
                            confidence = 0.85
                        logger.info(f"펀딩비 극단 Short: Funding={eth_funding:.4f}%, 델타/CVD 반대 방향 하락")
                    else:
                        logger.debug(f"⚠️ [Funding Rate] 펀딩비 숏 조건 충족했으나 CVD/델타 반대 방향 미충족")
                else:
                    logger.debug(f"⚠️ [Funding Rate] CVD 데이터 없음")
            else:
                logger.debug(f"⚠️ [Funding Rate] 펀딩비 임계값 미충족: {eth_funding:.6f} (범위: -{self.funding_threshold:.6f} ~ {self.funding_threshold:.6f})")
            
            if signal:
                return {
                    'signal': signal,
                    'entry_price': entry_price,
                    'stop_loss': None,
                    'confidence': confidence if 'confidence' in locals() else 0.7,
                    'strategy': self.name
                }
            
            return None
            
        except Exception as e:
            logger.error(f"펀딩비 전략 분석 실패: {e}")
            return None
