"""
실시간 데이터 수집 모듈
"""
import pandas as pd
import numpy as np
from binance_client import BinanceClient
import config
import logging
from datetime import datetime
import time

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self):
        self.client = BinanceClient()
        self.eth_data = None
        self.btc_data = None
        self.eth_funding_rate = None
        self.btc_funding_rate = None
        self.eth_liquidation_data = []  # 청산 데이터 저장
        self.btc_liquidation_data = []  # BTC 청산 데이터 저장
        
    def fetch_historical_data(self, symbol, interval=config.TIMEFRAME, limit=config.LOOKBACK_PERIOD):
        """과거 캔들 데이터 조회 및 DataFrame 변환"""
        try:
            klines = self.client.get_klines(symbol, interval, limit)
            if not klines:
                return None
            
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # 데이터 타입 변환
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                            'quote_volume', 'taker_buy_base', 'taker_buy_quote']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.set_index('timestamp', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"과거 데이터 조회 실패 ({symbol}): {e}")
            return None
    
    def update_data(self):
        """ETH와 BTC 데이터 업데이트"""
        try:
            # ETH 데이터
            self.eth_data = self.fetch_historical_data(config.ETH_SYMBOL)
            
            # BTC 데이터
            self.btc_data = self.fetch_historical_data(config.BTC_SYMBOL)
            
            # 펀딩비 업데이트 (선물 거래에서만)
            if self.client.use_futures:
                self.eth_funding_rate = self.client.get_funding_rate(config.ETH_SYMBOL)
                self.btc_funding_rate = self.client.get_funding_rate(config.BTC_SYMBOL)
                
                # 청산 데이터 업데이트 (선물 거래에서만)
                self.update_liquidation_data('ETH')
                self.update_liquidation_data('BTC')
            
            if self.eth_data is not None and self.btc_data is not None:
                eth_latest = self.eth_data.iloc[-1] if len(self.eth_data) > 0 else None
                btc_latest = self.btc_data.iloc[-1] if len(self.btc_data) > 0 else None
                
                logger.info(f"데이터 업데이트 완료 - ETH: {len(self.eth_data)}개, BTC: {len(self.btc_data)}개")
                if eth_latest is not None:
                    logger.debug(f"ETH 최신 캔들: {eth_latest.name} | 종가: ${eth_latest['close']:.2f}")
                if btc_latest is not None:
                    logger.debug(f"BTC 최신 캔들: {btc_latest.name} | 종가: ${btc_latest['close']:.2f}")
                return True
            else:
                logger.warning("데이터 업데이트 실패")
                if self.eth_data is None:
                    logger.warning("ETH 데이터가 None입니다")
                if self.btc_data is None:
                    logger.warning("BTC 데이터가 None입니다")
                return False
                
        except Exception as e:
            logger.error(f"데이터 업데이트 중 오류: {e}")
            return False
    
    def get_latest_candle(self, symbol='ETH'):
        """최신 캔들 데이터 반환"""
        data = self.eth_data if symbol == 'ETH' else self.btc_data
        if data is not None and len(data) > 0:
            return data.iloc[-1]
        return None
    
    def get_candles(self, symbol='ETH', count=100):
        """최근 N개 캔들 반환"""
        data = self.eth_data if symbol == 'ETH' else self.btc_data
        if data is not None and len(data) > 0:
            return data.tail(count)
        return None
    
    def calculate_cvd(self, symbol='ETH', lookback=100, ema_period=21, delta_smoothing=5):
        """CVD (Cumulative Volume Delta) 계산 (최적 세팅)
        
        Args:
            symbol: 'ETH' or 'BTC'
            lookback: 조회할 캔들 수
            ema_period: CVD EMA 스무딩 기간 (21)
            delta_smoothing: 델타 스무딩 기간 (5)
        """
        try:
            data = self.eth_data if symbol == 'ETH' else self.btc_data
            if data is None or len(data) < lookback:
                return None
            
            recent_data = data.tail(lookback).copy()
            
            # Aggressive Buy/Sell Volume 계산
            recent_data['buy_volume'] = recent_data['taker_buy_quote']
            recent_data['sell_volume'] = recent_data['quote_volume'] - recent_data['taker_buy_quote']
            recent_data['delta'] = recent_data['buy_volume'] - recent_data['sell_volume']
            
            # 델타 스무딩 (5)
            if len(recent_data) >= delta_smoothing:
                recent_data['delta_smooth'] = recent_data['delta'].rolling(window=delta_smoothing).mean()
            else:
                recent_data['delta_smooth'] = recent_data['delta']
            
            recent_data['cvd'] = recent_data['delta'].cumsum()
            
            # CVD EMA 스무딩 (21)
            if len(recent_data) >= ema_period:
                recent_data['cvd_ema'] = recent_data['cvd'].ewm(span=ema_period, adjust=False).mean()
            else:
                recent_data['cvd_ema'] = recent_data['cvd']
            
            return recent_data[['close', 'cvd', 'cvd_ema', 'delta', 'delta_smooth', 'buy_volume', 'sell_volume']]
            
        except Exception as e:
            logger.error(f"CVD 계산 실패: {e}")
            return None
    
    def update_liquidation_data(self, symbol='ETH'):
        """청산 데이터 업데이트 및 스파이크 탐지"""
        try:
            liquidation_orders = self.client.get_liquidation_orders(
                symbol=config.ETH_SYMBOL if symbol == 'ETH' else config.BTC_SYMBOL,
                limit=100
            )
            
            if liquidation_orders is None or len(liquidation_orders) == 0:
                return None
            
            # 최근 청산 데이터 저장
            liquidation_list = self.eth_liquidation_data if symbol == 'ETH' else self.btc_liquidation_data
            
            # 새로운 청산 주문 추가
            for order in liquidation_orders:
                liquidation_list.append({
                    'time': int(order['time']),
                    'price': float(order['price']),
                    'qty': float(order['qty']),
                    'side': order['side']  # 'BUY' (롱 청산) or 'SELL' (숏 청산)
                })
            
            # 최근 100개만 유지
            if len(liquidation_list) > 100:
                liquidation_list = liquidation_list[-100:]
            
            if symbol == 'ETH':
                self.eth_liquidation_data = liquidation_list
            else:
                self.btc_liquidation_data = liquidation_list
            
            return liquidation_list
            
        except Exception as e:
            logger.error(f"청산 데이터 업데이트 실패 ({symbol}): {e}")
            return None
    
    def detect_liquidation_spike(self, symbol='ETH', time_window_minutes=3, min_volume_threshold=10):
        """청산 스파이크 탐지
        
        Args:
            symbol: 'ETH' or 'BTC'
            time_window_minutes: 스파이크 탐지 시간 윈도우 (분)
            min_volume_threshold: 최소 청산 볼륨 (ETH/BTC 수량)
        
        Returns:
            dict: {
                'spike_detected': bool,
                'spike_type': 'long_liquidation' or 'short_liquidation',
                'total_volume': float,
                'count': int
            }
        """
        try:
            liquidation_list = self.eth_liquidation_data if symbol == 'ETH' else self.btc_liquidation_data
            
            if len(liquidation_list) < 5:
                return None
            
            # 최근 N분 내 청산 데이터 필터링
            current_time = int(datetime.now().timestamp() * 1000)
            time_window_ms = time_window_minutes * 60 * 1000
            
            recent_liquidations = [
                liq for liq in liquidation_list
                if (current_time - liq['time']) <= time_window_ms
            ]
            
            if len(recent_liquidations) < 3:
                return None
            
            # 롱 청산 (숏 포지션 청산) vs 숏 청산 (롱 포지션 청산)
            long_liquidation_volume = sum(
                liq['qty'] for liq in recent_liquidations if liq['side'] == 'BUY'
            )
            short_liquidation_volume = sum(
                liq['qty'] for liq in recent_liquidations if liq['side'] == 'SELL'
            )
            
            # 스파이크 탐지
            spike_detected = False
            spike_type = None
            total_volume = 0
            count = 0
            
            if long_liquidation_volume >= min_volume_threshold:
                # 롱 청산 스파이크 (숏 포지션 대량 청산) → 가격 상승 압력
                spike_detected = True
                spike_type = 'long_liquidation'
                total_volume = long_liquidation_volume
                count = len([liq for liq in recent_liquidations if liq['side'] == 'BUY'])
            
            elif short_liquidation_volume >= min_volume_threshold:
                # 숏 청산 스파이크 (롱 포지션 대량 청산) → 가격 하락 압력
                spike_detected = True
                spike_type = 'short_liquidation'
                total_volume = short_liquidation_volume
                count = len([liq for liq in recent_liquidations if liq['side'] == 'SELL'])
            
            if spike_detected:
                return {
                    'spike_detected': True,
                    'spike_type': spike_type,
                    'total_volume': total_volume,
                    'count': count,
                    'time_window_minutes': time_window_minutes
                }
            
            return None
            
        except Exception as e:
            logger.error(f"청산 스파이크 탐지 실패 ({symbol}): {e}")
            return None
