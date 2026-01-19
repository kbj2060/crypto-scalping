"""
실시간 데이터 수집 모듈
"""
import pandas as pd
import numpy as np
from .binance_client import BinanceClient
import config
import logging
from datetime import datetime, timedelta
import time
import os

logger = logging.getLogger(__name__)


class DataCollector:
    def __init__(self, use_saved_data=False):
        """
        Args:
            use_saved_data: True면 저장된 데이터를 로드, False면 실시간 데이터 사용
        """
        self.client = BinanceClient()
        self.eth_data = None
        self.btc_data = None
        self.eth_funding_rate = None
        self.btc_funding_rate = None
        self.eth_liquidation_data = []  # 청산 데이터 저장
        self.btc_liquidation_data = []  # BTC 청산 데이터 저장
        self.use_saved_data = use_saved_data
        self.current_index = 0  # 저장된 데이터 사용 시 현재 인덱스
        
        if use_saved_data:
            self.load_saved_data()
        
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
                try:
                    self.eth_funding_rate = self.client.get_funding_rate(config.ETH_SYMBOL)
                    self.btc_funding_rate = self.client.get_funding_rate(config.BTC_SYMBOL)
                except Exception as e:
                    logger.debug(f"펀딩비 조회 실패 (계속 진행): {e}")
                    self.eth_funding_rate = None
                    self.btc_funding_rate = None
                
                # 청산 데이터 업데이트 (선물 거래에서만)
                try:
                    self.update_liquidation_data('ETH')
                    self.update_liquidation_data('BTC')
                except Exception as e:
                    logger.debug(f"청산 데이터 업데이트 실패 (계속 진행): {e}")
            
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
        if self.use_saved_data:
            # 저장된 데이터 사용 시: 현재 인덱스의 이전 캔들 반환
            data = self.eth_data if symbol == 'ETH' else self.btc_data
            if data is not None and len(data) > 0 and self.current_index > 0:
                return data.iloc[self.current_index - 1]
            return None
        else:
            # 실시간 데이터 사용 시
            data = self.eth_data if symbol == 'ETH' else self.btc_data
            if data is not None and len(data) > 0:
                return data.iloc[-1]
            return None
    
    def get_candles(self, symbol='ETH', count=100):
        """최근 N개 캔들 반환"""
        if self.use_saved_data:
            # 저장된 데이터 사용 시: 현재 인덱스 기준으로 이전 count개 반환
            data = self.eth_data if symbol == 'ETH' else self.btc_data
            if data is not None and len(data) > 0:
                # 현재 인덱스가 충분히 커야 함 (최소 count개 필요)
                if self.current_index >= count:
                    start_idx = self.current_index - count
                    return data.iloc[start_idx:self.current_index]
                else:
                    # 부족한 경우 가능한 만큼만 반환
                    return data.iloc[:self.current_index] if self.current_index > 0 else None
            return None
        else:
            # 실시간 데이터 사용 시
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
            # 저장된 데이터 사용 시: CSV의 청산 데이터 사용
            if self.use_saved_data:
                data = self.eth_data if symbol == 'ETH' else self.btc_data
                if data is None or len(data) < time_window_minutes:
                    return None
                
                # 현재 인덱스 기준으로 최근 N개 캔들의 청산 데이터 집계
                start_idx = max(0, self.current_index - time_window_minutes)
                end_idx = self.current_index
                
                if end_idx <= start_idx:
                    return None
                
                recent_data = data.iloc[start_idx:end_idx]
                
                # 롱 청산 (숏 포지션 청산) vs 숏 청산 (롱 포지션 청산)
                long_liquidation_volume = recent_data['liquidation_long'].sum() if 'liquidation_long' in recent_data.columns else 0.0
                short_liquidation_volume = recent_data['liquidation_short'].sum() if 'liquidation_short' in recent_data.columns else 0.0
                
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
                    count = len(recent_data[recent_data['liquidation_long'] > 0])
                
                elif short_liquidation_volume >= min_volume_threshold:
                    # 숏 청산 스파이크 (롱 포지션 대량 청산) → 가격 하락 압력
                    spike_detected = True
                    spike_type = 'short_liquidation'
                    total_volume = short_liquidation_volume
                    count = len(recent_data[recent_data['liquidation_short'] > 0])
                
                if spike_detected:
                    return {
                        'spike_detected': True,
                        'spike_type': spike_type,
                        'total_volume': total_volume,
                        'count': count,
                        'time_window_minutes': time_window_minutes
                    }
                
                return None
            
            # 실시간 데이터 사용 시: 기존 로직 사용
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
    
    def load_saved_data(self):
        """저장된 데이터 로드 (학습용) - eth_3m_1year.csv와 btc_3m_1year.csv만 사용"""
        try:
            import os
            # 고정된 파일명 사용 (중복 파일 제거)
            eth_file = 'data/eth_3m_1year.csv'
            btc_file = 'data/btc_3m_1year.csv'
            
            if not os.path.exists(eth_file) or not os.path.exists(btc_file):
                logger.warning(f"저장된 데이터 파일을 찾을 수 없습니다: {eth_file}, {btc_file}")
                logger.warning("collect_training_data.py를 먼저 실행하세요.")
                return False
            
            # ETH 데이터 로드
            self.eth_data = pd.read_csv(eth_file, index_col='timestamp', parse_dates=True)
            
            # 청산 데이터 컬럼이 없으면 기본값 0으로 추가
            if 'liquidation_long' not in self.eth_data.columns:
                self.eth_data['liquidation_long'] = 0.0
            if 'liquidation_short' not in self.eth_data.columns:
                self.eth_data['liquidation_short'] = 0.0
            
            logger.info(f"✅ ETH 데이터 로드: {len(self.eth_data)}개 캔들")
            
            # BTC 데이터 로드
            self.btc_data = pd.read_csv(btc_file, index_col='timestamp', parse_dates=True)
            
            # 청산 데이터 컬럼이 없으면 기본값 0으로 추가
            if 'liquidation_long' not in self.btc_data.columns:
                self.btc_data['liquidation_long'] = 0.0
            if 'liquidation_short' not in self.btc_data.columns:
                self.btc_data['liquidation_short'] = 0.0
            
            logger.info(f"✅ BTC 데이터 로드: {len(self.btc_data)}개 캔들")
            
            # 인덱스 초기화
            self.current_index = 0
            
            return True
            
        except Exception as e:
            logger.error(f"저장된 데이터 로드 실패: {e}")
            return False
    
    def get_next_candles(self, count=1):
        """저장된 데이터에서 다음 N개 캔들 반환 (학습용)"""
        if not self.use_saved_data or self.eth_data is None:
            return None
        
        if self.current_index + count > len(self.eth_data):
            return None  # 데이터 끝
        
        # 현재 인덱스부터 count개 반환
        eth_slice = self.eth_data.iloc[self.current_index:self.current_index + count]
        btc_slice = self.btc_data.iloc[self.current_index:self.current_index + count] if self.btc_data is not None else None
        
        self.current_index += count
        
        return {
            'ETH': eth_slice,
            'BTC': btc_slice
        }
    
    def reset_index(self):
        """인덱스를 처음으로 리셋 (새 에피소드 시작 시)"""
        # lookback(40)개가 필요하므로 최소 40부터 시작
        lookback = 40  # TradingEnvironment의 기본 lookback
        self.current_index = lookback
    
    def fetch_historical_klines_batch(self, symbol, interval, start_time, end_time):
        """특정 기간의 캔들 데이터를 배치로 조회 (바이낸스 API 제한 고려)
        
        Args:
            symbol: 거래 심볼 (예: 'ETHUSDT')
            interval: 타임프레임 (예: '3m')
            start_time: 시작 시간 (datetime)
            end_time: 종료 시간 (datetime)
        
        Returns:
            list: 캔들 데이터 리스트
        """
        all_klines = []
        
        # 밀리초 타임스탬프로 변환
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        # 역순으로 가져오기 (최신부터 과거로)
        current_end = end_timestamp
        batch_count = 0
        max_batches = 200  # 안전장치
        
        logger.info(f"데이터 수집 시작: {start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}")
        
        while current_end > start_timestamp and batch_count < max_batches:
            try:
                batch_count += 1
                
                # 한 번에 최대 1000봉 조회
                if self.client.use_futures:
                    # 선물 거래: endTime을 사용하여 역순으로 가져오기
                    klines = self.client.client.futures_klines(
                        symbol=symbol,
                        interval=interval,
                        endTime=current_end,
                        limit=1000
                    )
                else:
                    # 스팟 거래: get_historical_klines 사용
                    current_end_dt = datetime.fromtimestamp(current_end / 1000)
                    start_dt = datetime.fromtimestamp(start_timestamp / 1000)
                    klines = self.client.client.get_historical_klines(
                        symbol=symbol,
                        interval=interval,
                        start_str=start_dt.strftime('%d %b %Y %H:%M:%S'),
                        end_str=current_end_dt.strftime('%d %b %Y %H:%M:%S'),
                        limit=1000
                    )
                
                if not klines or len(klines) == 0:
                    logger.warning("더 이상 데이터가 없습니다.")
                    break
                
                # 타임스탬프 필터링 (필요한 기간만)
                filtered_klines = []
                for k in klines:
                    k_time = int(k[0])  # open time
                    if start_timestamp <= k_time <= end_timestamp:
                        filtered_klines.append(k)
                
                if not filtered_klines:
                    logger.warning("필터링 후 데이터가 없습니다.")
                    break
                
                all_klines.extend(filtered_klines)
                
                # 가장 오래된 캔들의 시간을 다음 endTime으로 설정 (역순)
                oldest_time = min(int(k[0]) for k in filtered_klines)
                current_end = oldest_time - 1
                
                oldest_dt = datetime.fromtimestamp(oldest_time / 1000)
                logger.info(f"  배치 {batch_count}: {len(filtered_klines)}개 수집, 총 {len(all_klines)}개 (가장 오래된: {oldest_dt.strftime('%Y-%m-%d %H:%M:%S')})")
                
                # API 제한 방지
                time.sleep(0.2)
                    
            except Exception as e:
                logger.error(f"데이터 조회 중 오류: {e}")
                time.sleep(1)
                continue
        
        # 시간순으로 정렬 (오래된 것부터)
        all_klines.sort(key=lambda x: int(x[0]))
        
        logger.info(f"총 {len(all_klines)}개 캔들 수집 완료")
        return all_klines
    
    def collect_and_save_historical_data(self, days=365, timeframe=None):
        """과거 데이터 수집 및 CSV 파일로 저장
        
        Args:
            days: 수집할 일수 (기본값: 365일)
            timeframe: 타임프레임 (기본값: config.TIMEFRAME)
        
        Returns:
            bool: 성공 여부
        """
        if timeframe is None:
            timeframe = config.TIMEFRAME
        
        logger.info("=" * 60)
        logger.info(f"📥 {days}일치 학습 데이터 수집 시작")
        logger.info("=" * 60)
        
        # data 폴더 생성
        os.makedirs('data', exist_ok=True)
        
        # 수집 기간 설정
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)
        
        logger.info(f"수집 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"타임프레임: {timeframe}")
        
        # ETH 데이터 수집
        logger.info("")
        logger.info("ETH 데이터 수집 중...")
        eth_klines = self.fetch_historical_klines_batch(
            config.ETH_SYMBOL,
            timeframe,
            start_time,
            end_time
        )
        
        if not eth_klines:
            logger.error("❌ ETH 데이터 수집 실패")
            return False
        
        # ETH DataFrame 생성 및 저장
        eth_df = pd.DataFrame(eth_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 데이터 타입 변환
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_columns:
            eth_df[col] = pd.to_numeric(eth_df[col], errors='coerce')
        
        eth_df.set_index('timestamp', inplace=True)
        eth_df.sort_index(inplace=True)
        
        # 중복 제거
        eth_df = eth_df[~eth_df.index.duplicated(keep='last')]
        
        # 청산 데이터 컬럼 추가 (기본값 0)
        eth_df['liquidation_long'] = 0.0
        eth_df['liquidation_short'] = 0.0
        
        # CSV 저장
        eth_file = f'data/eth_{timeframe}_1year.csv'
        eth_df.to_csv(eth_file)
        logger.info(f"✅ ETH 데이터 저장 완료: {eth_file} ({len(eth_df)}개 캔들)")
        
        # BTC 데이터 수집
        logger.info("")
        logger.info("BTC 데이터 수집 중...")
        btc_klines = self.fetch_historical_klines_batch(
            config.BTC_SYMBOL,
            timeframe,
            start_time,
            end_time
        )
        
        if not btc_klines:
            logger.error("❌ BTC 데이터 수집 실패")
            return False
        
        # BTC DataFrame 생성 및 저장
        btc_df = pd.DataFrame(btc_klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # 데이터 타입 변환
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 
                          'quote_volume', 'taker_buy_base', 'taker_buy_quote']
        for col in numeric_columns:
            btc_df[col] = pd.to_numeric(btc_df[col], errors='coerce')
        
        btc_df.set_index('timestamp', inplace=True)
        btc_df.sort_index(inplace=True)
        
        # 중복 제거
        btc_df = btc_df[~btc_df.index.duplicated(keep='last')]
        
        # 청산 데이터 컬럼 추가 (기본값 0)
        btc_df['liquidation_long'] = 0.0
        btc_df['liquidation_short'] = 0.0
        
        # CSV 저장
        btc_file = f'data/btc_{timeframe}_1year.csv'
        btc_df.to_csv(btc_file)
        logger.info(f"✅ BTC 데이터 저장 완료: {btc_file} ({len(btc_df)}개 캔들)")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info("✅ 데이터 수집 완료!")
        logger.info(f"   ETH: {len(eth_df)}개 캔들")
        logger.info(f"   BTC: {len(btc_df)}개 캔들")
        logger.info(f"   저장 위치: data/ 폴더")
        logger.info("=" * 60)
        
        return True