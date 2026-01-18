"""
리스크 관리 모듈
"""
import logging
import config
from .binance_client import BinanceClient

logger = logging.getLogger(__name__)


class RiskManager:
    def __init__(self):
        self.client = BinanceClient()
        self.max_position_size = config.MAX_POSITION_SIZE
        self.stop_loss_percent = config.STOP_LOSS_PERCENT / 100
    
    def calculate_position_size(self, entry_price, stop_loss_price, risk_amount=None, use_spot=False):
        """포지션 크기 계산 (리스크 기반)
        
        Args:
            entry_price: 진입 가격
            stop_loss_price: 손절 가격
            risk_amount: 리스크 금액 (USDT)
            use_spot: 스팟 거래 여부 (True면 USDT 금액 반환, False면 코인 수량 반환)
        
        Returns:
            스팟 거래: USDT 금액 (매수 시)
            선물 거래: 코인 수량
        """
        try:
            if risk_amount is None:
                risk_amount = self.max_position_size * 0.01  # 최대 포지션의 1% 리스크
            
            # 손절 거리 계산
            if stop_loss_price is None:
                stop_loss_price = entry_price * (1 - self.stop_loss_percent)
            
            price_diff = abs(entry_price - stop_loss_price)
            if price_diff == 0:
                return None
            
            if use_spot:
                # 스팟 거래: USDT 금액 반환 (매수 시 사용)
                position_size = min(risk_amount / (price_diff / entry_price), self.max_position_size)
                return round(position_size, 2)  # USDT는 소수점 2자리
            else:
                # 선물 거래: 코인 수량 반환
                position_size = risk_amount / price_diff
                
                # 최대 포지션 크기 제한
                max_size = self.max_position_size / entry_price
                position_size = min(position_size, max_size)
                
                return round(position_size, 3)
            
        except Exception as e:
            logger.error(f"포지션 크기 계산 실패: {e}")
            return None
    
    def check_position_limit(self):
        """현재 포지션 확인 및 제한 체크"""
        try:
            position = self.client.get_position(config.ETH_SYMBOL)
            if position is None:
                return True  # 포지션 없음, 진입 가능
            
            current_size = abs(position['size'])
            max_size = self.max_position_size / position['entry_price']
            
            if current_size >= max_size * 0.95:  # 95% 이상이면 제한
                logger.warning("최대 포지션 크기에 근접")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"포지션 제한 체크 실패: {e}")
            return False
    
    def should_take_profit(self, entry_price, current_price, side):
        """익절 조건 확인"""
        try:
            if side == 'LONG':
                profit_pct = ((current_price - entry_price) / entry_price) * 100
                # 2% 이상 수익이면 익절 고려
                if profit_pct >= 2.0:
                    return True
            elif side == 'SHORT':
                profit_pct = ((entry_price - current_price) / entry_price) * 100
                if profit_pct >= 2.0:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"익절 조건 확인 실패: {e}")
            return False
    
    def should_stop_loss(self, entry_price, current_price, stop_loss_price, side):
        """손절 조건 확인"""
        try:
            if stop_loss_price is None:
                return False
            
            if side == 'LONG':
                if current_price <= stop_loss_price:
                    return True
            elif side == 'SHORT':
                if current_price >= stop_loss_price:
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"손절 조건 확인 실패: {e}")
            return False
    
    def validate_signal(self, signal_data):
        """신호 유효성 검증"""
        try:
            if signal_data is None:
                return False
            
            required_fields = ['signal', 'entry_price', 'confidence']
            for field in required_fields:
                if field not in signal_data:
                    return False
            
            # 신뢰도 체크
            if signal_data['confidence'] < 0.5:
                logger.warning(f"신뢰도 낮음: {signal_data['confidence']}")
                return False
            
            # 포지션 제한 체크
            if not self.check_position_limit():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"신호 검증 실패: {e}")
            return False
