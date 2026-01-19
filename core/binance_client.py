"""
바이낸스 API 클라이언트
"""
from binance.client import Client
import config
import logging

logger = logging.getLogger(__name__)


class BinanceClient:
    def __init__(self):
        """바이낸스 클라이언트 초기화"""
        # API 키 검증
        if not config.BINANCE_API_KEY or not config.BINANCE_SECRET_KEY:
            raise ValueError(
                "API 키가 설정되지 않았습니다. .env 파일에 BINANCE_API_KEY와 BINANCE_SECRET_KEY를 입력하세요."
            )
        
        if config.BINANCE_API_KEY == 'your_api_key_here' or config.BINANCE_SECRET_KEY == 'your_secret_key_here':
            raise ValueError(
                "API 키를 실제 값으로 변경해주세요. .env 파일을 확인하세요."
            )
        
        try:
            if config.BINANCE_TESTNET:
                self.client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_SECRET_KEY,
                    testnet=True
                )
                # 테스트넷에서 선물 API 접근 가능 여부 확인
                try:
                    # 선물 계정 정보 조회로 권한 확인
                    test_futures = self.client.futures_account()
                    self.use_futures = True
                    logger.info("바이낸스 테스트넷 클라이언트 초기화 (선물 거래 모드)")
                except Exception as e:
                    # 선물 권한이 없으면 스팟 모드로 전환
                    self.use_futures = False
                    logger.debug("테스트넷 선물 API 접근 실패 → 스팟 모드로 전환")
                    logger.debug(f"  이유: {e}")
                    logger.debug("  펀딩비 및 청산 스파이크 전략은 비활성화됩니다.")
                    logger.info("바이낸스 테스트넷 클라이언트 초기화 (스팟 거래 모드)")
            else:
                self.client = Client(
                    config.BINANCE_API_KEY,
                    config.BINANCE_SECRET_KEY
                )
                self.use_futures = True  # 실제 거래소는 선물 거래 사용
                logger.info("바이낸스 실제 거래소 클라이언트 초기화 (선물 거래 모드)")
            
            # 선물 거래 설정 (레버리지 변경)
            if self.use_futures:
                try:
                    self.client.futures_change_leverage(
                        symbol=config.ETH_SYMBOL,
                        leverage=config.LEVERAGE
                    )
                    logger.info(f"레버리지 설정 완료: {config.LEVERAGE}x")
                except Exception as e:
                    logger.debug(f"레버리지 설정 실패 (계속 진행): {e}")
                    logger.debug("API 키에 선물 거래 권한이 없거나 IP 제한이 있을 수 있습니다. (분석 모드에서는 정상)")
        
        except Exception as e:
            error_msg = str(e)
            if "API-key" in error_msg or "-2015" in error_msg:
                logger.error("=" * 60)
                logger.error("API 키 오류 발생!")
                logger.error("가능한 원인:")
                logger.error("1. API 키가 잘못되었습니다")
                if self.use_futures:
                    logger.error("2. API 키에 선물 거래 권한이 없습니다")
                else:
                    logger.error("2. API 키에 스팟 거래 권한이 없습니다")
                logger.error("3. IP 화이트리스트가 설정되어 있습니다")
                logger.error("4. 테스트넷/실거래소 API 키가 일치하지 않습니다")
                logger.error("=" * 60)
                logger.error(f"현재 설정: BINANCE_TESTNET={config.BINANCE_TESTNET}")
                logger.error("=" * 60)
            raise
    
    def get_klines(self, symbol, interval, limit=200):
        """캔들 데이터 조회"""
        try:
            if self.use_futures:
                klines = self.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            else:
                klines = self.client.get_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=limit
                )
            
            if klines and len(klines) > 0:
                logger.debug(f"캔들 데이터 조회 성공: {symbol} - {len(klines)}개 캔들")
            else:
                logger.warning(f"캔들 데이터가 비어있음: {symbol}")
            
            return klines
        except Exception as e:
            logger.error(f"캔들 데이터 조회 실패 ({symbol}): {e}")
            return None
    
    def get_funding_rate(self, symbol):
        """펀딩비 조회 (선물 거래에서만 사용 가능)"""
        try:
            if not self.use_futures:
                # 스팟 거래에서는 펀딩비가 없음
                return None
            funding_info = self.client.futures_funding_rate(symbol=symbol, limit=1)
            if funding_info:
                return float(funding_info[0]['fundingRate'])
            return None
        except Exception as e:
            logger.error(f"펀딩비 조회 실패: {e}")
            return None
    
    def get_ticker(self, symbol):
        """현재 가격 조회"""
        try:
            if self.use_futures:
                ticker = self.client.futures_symbol_ticker(symbol=symbol)
            else:
                ticker = self.client.get_symbol_ticker(symbol=symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"가격 조회 실패: {e}")
            return None
    
    def get_orderbook(self, symbol, limit=20):
        """오더북 조회"""
        try:
            if self.use_futures:
                orderbook = self.client.futures_order_book(symbol=symbol, limit=limit)
            else:
                orderbook = self.client.get_order_book(symbol=symbol, limit=limit)
            return orderbook
        except Exception as e:
            logger.error(f"오더북 조회 실패: {e}")
            return None
    
    def get_aggressive_trades(self, symbol, limit=100):
        """공격적 거래량 조회 (CVD 계산용)"""
        try:
            if self.use_futures:
                trades = self.client.futures_recent_trades(symbol=symbol, limit=limit)
            else:
                trades = self.client.get_recent_trades(symbol=symbol, limit=limit)
            return trades
        except Exception as e:
            logger.error(f"거래 내역 조회 실패: {e}")
            return None
    
    def get_liquidation_orders(self, symbol, limit=100):
        """청산 주문 조회 (선물 거래에서만 사용 가능)"""
        try:
            if not self.use_futures:
                # 스팟 거래에서는 청산이 없음
                return None
            
            liquidation_orders = self.client.futures_liquidation_orders(
                symbol=symbol,
                limit=limit
            )
            return liquidation_orders
        except Exception as e:
            error_msg = str(e)
            # API 권한 오류는 분석 모드에서는 정상 (DEBUG 레벨로 처리)
            if "-2015" in error_msg or "permissions" in error_msg.lower():
                logger.debug(f"청산 주문 조회 실패 (권한 없음, 계속 진행): {e}")
            else:
                logger.warning(f"청산 주문 조회 실패: {e}")
            return None
    
    def place_order(self, symbol, side, quantity, order_type='MARKET', 
                   stop_price=None, price=None, quote_quantity=None):
        """주문 실행
        
        Args:
            symbol: 거래 심볼
            side: BUY 또는 SELL
            quantity: 코인 수량 (매도 시) 또는 USDT 금액 (매수 시 스팟 거래)
            order_type: 주문 타입
            quote_quantity: USDT 기준 금액 (스팟 매수 시 사용)
        """
        try:
            if self.use_futures:
                # 선물 거래
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type,
                    'quantity': quantity
                }
                
                if order_type == 'STOP_MARKET':
                    params['stopPrice'] = stop_price
                elif order_type == 'LIMIT':
                    params['price'] = price
                    params['timeInForce'] = 'GTC'
                
                order = self.client.futures_create_order(**params)
            else:
                # 스팟 거래
                params = {
                    'symbol': symbol,
                    'side': side,
                    'type': order_type
                }
                
                if order_type == 'MARKET':
                    if side == 'BUY':
                        # 매수: USDT 기준 금액 사용
                        if quote_quantity:
                            params['quoteOrderQty'] = quote_quantity
                        else:
                            params['quoteOrderQty'] = quantity  # quantity가 USDT 금액
                    else:
                        # 매도: 코인 수량 사용
                        params['quantity'] = quantity
                elif order_type == 'LIMIT':
                    params['quantity'] = quantity
                    params['price'] = price
                    params['timeInForce'] = 'GTC'
                
                order = self.client.create_order(**params)
            
            logger.info(f"주문 실행: {order}")
            return order
        except Exception as e:
            logger.error(f"주문 실행 실패: {e}")
            return None
    
    def close_position(self, symbol):
        """포지션 청산 (선물) 또는 전체 매도 (스팟)"""
        try:
            if self.use_futures:
                # 선물 거래: 포지션 청산
                position = self.client.futures_position_information(symbol=symbol)[0]
                position_amt = float(position['positionAmt'])
                
                if position_amt == 0:
                    return None
                
                side = 'SELL' if position_amt > 0 else 'BUY'
                quantity = abs(position_amt)
                
                order = self.place_order(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    order_type='MARKET'
                )
                return order
            else:
                # 스팟 거래: 보유 자산 전체 매도
                balance = self.get_balance(symbol.replace('USDT', ''))
                if balance and balance > 0:
                    order = self.place_order(
                        symbol=symbol,
                        side='SELL',
                        quantity=balance,
                        order_type='MARKET'
                    )
                    return order
                return None
        except Exception as e:
            logger.error(f"포지션 청산 실패: {e}")
            return None
    
    def get_position(self, symbol):
        """현재 포지션 정보 조회 (선물) 또는 보유 자산 조회 (스팟)"""
        try:
            if self.use_futures:
                # 선물 거래: 포지션 정보
                positions = self.client.futures_position_information(symbol=symbol)
                for pos in positions:
                    if float(pos['positionAmt']) != 0:
                        return {
                            'size': float(pos['positionAmt']),
                            'entry_price': float(pos['entryPrice']),
                            'unrealized_pnl': float(pos['unRealizedProfit']),
                            'leverage': int(pos['leverage'])
                        }
                return None
            else:
                # 스팟 거래: 보유 자산 정보
                # 주의: 계정 조회 권한이 없으면 None 반환 (정상 동작)
                base_asset = symbol.replace('USDT', '')
                balance = self.get_balance(base_asset)
                if balance is not None and balance > 0:
                    current_price = self.get_ticker(symbol)
                    if current_price:
                        return {
                            'size': balance,
                            'entry_price': current_price,  # 스팟은 평균 매수가 없으므로 현재가 사용
                            'unrealized_pnl': 0,  # 스팟은 실현 손익만 있음
                            'leverage': 1  # 스팟은 레버리지 없음
                        }
                # 자산 조회 실패 또는 보유 자산 없음 (권한 없으면 정상적으로 None 반환)
                return None
        except Exception as e:
            error_msg = str(e)
            # API 키 권한 오류는 조용히 처리 (스팟 모드에서는 정상)
            if "-2015" in error_msg or "Invalid API-key" in error_msg:
                if not self.use_futures:
                    # 스팟 모드에서 권한 없으면 정상 (None 반환)
                    return None
                else:
                    # 선물 모드에서 권한 오류는 경고만
                    logger.debug(f"포지션 조회 실패 (권한 없음, 계속 진행): {e}")
                    return None
            else:
                logger.error(f"포지션 조회 실패: {e}")
                return None
    
    def get_balance(self, asset):
        """보유 자산 조회 (스팟 거래용)"""
        try:
            if self.use_futures:
                # 선물 거래에서는 사용하지 않음
                return None
            
            account = self.client.get_account()
            for balance in account['balances']:
                if balance['asset'] == asset:
                    free = float(balance['free'])
                    locked = float(balance['locked'])
                    total = free + locked
                    if total > 0:
                        return total
            return 0
        except Exception as e:
            # 스팟 거래에서 계정 조회 권한이 없을 수 있음 (조용히 처리)
            # 디버그 레벨로만 로깅하여 경고 메시지 감소
            logger.debug(f"자산 조회 실패 (권한 없음): {e}")
            return None
