"""
과거 청산 데이터 수집 및 CSV 파일에 삽입 스크립트
각 캔들 시간대별로 청산 데이터를 집계하여 eth_3m_1year.csv와 btc_3m_1year.csv에 추가
"""
import os
import sys
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.binance_client import BinanceClient

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_liquidation.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def fetch_liquidation_orders_by_time(client, symbol, start_time, end_time):
    """특정 기간의 청산 주문 조회 (바이낸스 API는 최근 데이터만 제공 가능)"""
    try:
        if not client.use_futures:
            logger.warning("선물 거래 모드가 아니므로 청산 데이터를 가져올 수 없습니다.")
            return []
        
        # 바이낸스 선물 API를 사용하여 청산 주문 조회
        # startTime과 endTime을 밀리초 타임스탬프로 변환
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        liquidation_orders = []
        
        # 바이낸스 API는 최근 청산 데이터만 제공하므로, 최대한 많이 가져와서 필터링
        try:
            # 최대 1000개 청산 주문 조회
            orders = client.client.futures_liquidation_orders(
                symbol=symbol,
                limit=1000
            )
            
            if orders:
                # 시간 필터링: 요청한 기간에 해당하는 청산 주문만 선택
                filtered_orders = []
                for order in orders:
                    order_time = int(order.get('time', 0))
                    if start_timestamp <= order_time <= end_timestamp:
                        filtered_orders.append(order)
                
                liquidation_orders.extend(filtered_orders)
                logger.debug(f"  {symbol} 최근 청산 주문 중 {len(filtered_orders)}개가 해당 기간에 해당 (기간: {start_time.strftime('%Y-%m-%d %H:%M')} ~ {end_time.strftime('%Y-%m-%d %H:%M')})")
        
        except Exception as e:
            logger.warning(f"청산 주문 조회 실패 ({symbol}): {e}")
            # API 권한 오류는 무시하고 계속 진행
            if "-2015" not in str(e) and "permissions" not in str(e).lower():
                logger.debug(f"청산 데이터 조회 실패 상세: {e}")
        
        return liquidation_orders
    
    except Exception as e:
        logger.error(f"청산 주문 조회 중 오류 ({symbol}): {e}")
        return []


def aggregate_liquidation_by_candle(liquidation_orders, candle_times):
    """청산 주문을 캔들 시간대별로 집계"""
    # 캔들 시간을 인덱스로 하는 딕셔너리
    liquidation_by_candle = {}
    
    for candle_time in candle_times:
        liquidation_by_candle[candle_time] = {
            'liquidation_long': 0.0,
            'liquidation_short': 0.0
        }
    
    # 각 청산 주문을 해당하는 캔들 시간에 할당
    for order in liquidation_orders:
        try:
            order_time = int(order.get('time', 0))
            order_time_dt = datetime.fromtimestamp(order_time / 1000)
            
            # 해당하는 캔들 시간 찾기 (3분봉 기준)
            # 캔들 시작 시간 = 분을 3의 배수로 내림
            minute = order_time_dt.minute
            candle_minute = (minute // 3) * 3
            candle_time = order_time_dt.replace(minute=candle_minute, second=0, microsecond=0)
            
            if candle_time in liquidation_by_candle:
                qty = float(order.get('qty', 0))
                side = order.get('side', '')
                
                # BUY = 롱 청산 (숏 포지션 청산)
                # SELL = 숏 청산 (롱 포지션 청산)
                if side == 'BUY':
                    liquidation_by_candle[candle_time]['liquidation_long'] += qty
                elif side == 'SELL':
                    liquidation_by_candle[candle_time]['liquidation_short'] += qty
        
        except Exception as e:
            logger.debug(f"청산 주문 처리 중 오류: {e}")
            continue
    
    return liquidation_by_candle


def collect_and_insert_liquidation_data():
    """과거 청산 데이터 수집 및 CSV 파일에 삽입"""
    logger.info("=" * 60)
    logger.info("📥 과거 청산 데이터 수집 및 삽입 시작")
    logger.info("=" * 60)
    
    eth_file = 'data/eth_3m_1year.csv'
    btc_file = 'data/btc_3m_1year.csv'
    
    # 파일 존재 확인
    if not os.path.exists(eth_file) or not os.path.exists(btc_file):
        logger.error(f"데이터 파일을 찾을 수 없습니다: {eth_file}, {btc_file}")
        return False
    
    # CSV 파일 로드
    logger.info("CSV 파일 로드 중...")
    eth_df = pd.read_csv(eth_file, index_col='timestamp', parse_dates=True)
    btc_df = pd.read_csv(btc_file, index_col='timestamp', parse_dates=True)
    
    logger.info(f"ETH 데이터: {len(eth_df)}개 캔들")
    logger.info(f"BTC 데이터: {len(btc_df)}개 캔들")
    
    # 청산 데이터 컬럼 초기화 (없으면 추가)
    if 'liquidation_long' not in eth_df.columns:
        eth_df['liquidation_long'] = 0.0
    if 'liquidation_short' not in eth_df.columns:
        eth_df['liquidation_short'] = 0.0
    if 'liquidation_long' not in btc_df.columns:
        btc_df['liquidation_long'] = 0.0
    if 'liquidation_short' not in btc_df.columns:
        btc_df['liquidation_short'] = 0.0
    
    # 클라이언트 초기화
    try:
        client = BinanceClient()
    except Exception as e:
        logger.error(f"바이낸스 클라이언트 초기화 실패: {e}")
        logger.error("API 키를 확인하거나 분석 모드로 실행하세요.")
        return False
    
    if not client.use_futures:
        logger.warning("선물 거래 모드가 아니므로 청산 데이터를 수집할 수 없습니다.")
        return False
    
    # 데이터 수집 기간 설정
    start_time = eth_df.index[0]
    end_time = eth_df.index[-1]
    
    logger.info(f"수집 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("")
    
    # ETH 청산 데이터 수집
    # 바이낸스 API는 최근 청산 데이터만 제공하므로, 전체 기간에 대해 한 번만 조회
    logger.info("ETH 청산 데이터 수집 중...")
    logger.info("  참고: 바이낸스 API는 최근 청산 데이터만 제공합니다.")
    logger.info("  전체 기간의 데이터를 수집하려면 외부 API 서비스(CoinGlass, CoinAct 등)를 사용해야 할 수 있습니다.")
    
    eth_liquidation_orders = fetch_liquidation_orders_by_time(
        client,
        config.ETH_SYMBOL,
        start_time,
        end_time
    )
    
    logger.info(f"✅ ETH 청산 데이터 수집 완료: 총 {len(eth_liquidation_orders)}개 주문")
    
    # ETH 청산 데이터 집계 및 삽입
    if eth_liquidation_orders:
        logger.info("ETH 청산 데이터 집계 중...")
        eth_liquidation_by_candle = aggregate_liquidation_by_candle(eth_liquidation_orders, eth_df.index)
        
        # DataFrame에 삽입
        for candle_time, liq_data in eth_liquidation_by_candle.items():
            if candle_time in eth_df.index:
                eth_df.loc[candle_time, 'liquidation_long'] = liq_data['liquidation_long']
                eth_df.loc[candle_time, 'liquidation_short'] = liq_data['liquidation_short']
        
        logger.info(f"✅ ETH 청산 데이터 삽입 완료")
    
    # BTC 청산 데이터 수집
    logger.info("")
    logger.info("BTC 청산 데이터 수집 중...")
    
    btc_liquidation_orders = fetch_liquidation_orders_by_time(
        client,
        config.BTC_SYMBOL,
        start_time,
        end_time
    )
    
    logger.info(f"✅ BTC 청산 데이터 수집 완료: 총 {len(btc_liquidation_orders)}개 주문")
    
    # BTC 청산 데이터 집계 및 삽입
    if btc_liquidation_orders:
        logger.info("BTC 청산 데이터 집계 중...")
        btc_liquidation_by_candle = aggregate_liquidation_by_candle(btc_liquidation_orders, btc_df.index)
        
        # DataFrame에 삽입
        for candle_time, liq_data in btc_liquidation_by_candle.items():
            if candle_time in btc_df.index:
                btc_df.loc[candle_time, 'liquidation_long'] = liq_data['liquidation_long']
                btc_df.loc[candle_time, 'liquidation_short'] = liq_data['liquidation_short']
        
        logger.info(f"✅ BTC 청산 데이터 삽입 완료")
    
    # CSV 파일 저장
    logger.info("")
    logger.info("CSV 파일 저장 중...")
    eth_df.to_csv(eth_file)
    btc_df.to_csv(btc_file)
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 청산 데이터 수집 및 삽입 완료!")
    logger.info(f"   ETH: {len(eth_liquidation_orders)}개 청산 주문")
    logger.info(f"   BTC: {len(btc_liquidation_orders)}개 청산 주문")
    logger.info(f"   저장 위치: data/ 폴더")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = collect_and_insert_liquidation_data()
        if success:
            logger.info("청산 데이터가 CSV 파일에 추가되었습니다.")
        else:
            logger.error("청산 데이터 수집 실패")
    except KeyboardInterrupt:
        logger.info("청산 데이터 수집 중단")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
