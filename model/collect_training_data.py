"""
1년치 학습 데이터 수집 스크립트
바이낸스에서 1년치 과거 데이터를 가져와서 data 폴더에 저장합니다.
"""
import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
import time

# 상위 폴더를 경로에 추가 (config, core 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.binance_client import BinanceClient

# 로깅 설정
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/collect_data.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def fetch_historical_klines(client, symbol, interval, start_time, end_time):
    """특정 기간의 캔들 데이터 조회 (바이낸스 API 제한 고려)"""
    all_klines = []
    
    # 밀리초 타임스탬프로 변환
    start_timestamp = int(start_time.timestamp() * 1000)
    end_timestamp = int(end_time.timestamp() * 1000)
    
    # 1년치 데이터는 약 175,200봉 (3분봉 기준)
    # 한 번에 1000봉씩 가져오므로 약 176번 호출 필요
    # 역순으로 가져오기 (최신부터 과거로)
    current_end = end_timestamp
    batch_count = 0
    max_batches = 200  # 안전장치
    
    logger.info(f"데이터 수집 시작: {start_time.strftime('%Y-%m-%d')} ~ {end_time.strftime('%Y-%m-%d')}")
    
    while current_end > start_timestamp and batch_count < max_batches:
        try:
            batch_count += 1
            
            # 한 번에 최대 1000봉 조회
            if client.use_futures:
                # 선물 거래: endTime을 사용하여 역순으로 가져오기
                klines = client.client.futures_klines(
                    symbol=symbol,
                    interval=interval,
                    endTime=current_end,
                    limit=1000
                )
            else:
                # 스팟 거래: get_historical_klines 사용
                current_end_dt = datetime.fromtimestamp(current_end / 1000)
                start_dt = datetime.fromtimestamp(start_timestamp / 1000)
                klines = client.client.get_historical_klines(
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


def collect_one_year_data():
    """1년치 데이터 수집 및 저장"""
    logger.info("=" * 60)
    logger.info("📥 1년치 학습 데이터 수집 시작")
    logger.info("=" * 60)
    
    # data 폴더 생성
    os.makedirs('data', exist_ok=True)
    
    # 클라이언트 초기화
    client = BinanceClient()
    
    # 1년 전부터 현재까지
    end_time = datetime.now()
    start_time = end_time - timedelta(days=365)
    
    logger.info(f"수집 기간: {start_time.strftime('%Y-%m-%d %H:%M:%S')} ~ {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"타임프레임: {config.TIMEFRAME}")
    
    # ETH 데이터 수집
    logger.info("")
    logger.info("ETH 데이터 수집 중...")
    eth_klines = fetch_historical_klines(
        client,
        config.ETH_SYMBOL,
        config.TIMEFRAME,
        start_time,
        end_time
    )
    
    if eth_klines:
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
        
        # CSV 저장
        eth_file = f'data/eth_{config.TIMEFRAME}_1year.csv'
        eth_df.to_csv(eth_file)
        logger.info(f"✅ ETH 데이터 저장 완료: {eth_file} ({len(eth_df)}개 캔들)")
    else:
        logger.error("❌ ETH 데이터 수집 실패")
        return False
    
    # BTC 데이터 수집
    logger.info("")
    logger.info("BTC 데이터 수집 중...")
    btc_klines = fetch_historical_klines(
        client,
        config.BTC_SYMBOL,
        config.TIMEFRAME,
        start_time,
        end_time
    )
    
    if btc_klines:
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
        
        # CSV 저장
        btc_file = f'data/btc_{config.TIMEFRAME}_1year.csv'
        btc_df.to_csv(btc_file)
        logger.info(f"✅ BTC 데이터 저장 완료: {btc_file} ({len(btc_df)}개 캔들)")
    else:
        logger.error("❌ BTC 데이터 수집 실패")
        return False
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 데이터 수집 완료!")
    logger.info(f"   ETH: {len(eth_df)}개 캔들")
    logger.info(f"   BTC: {len(btc_df)}개 캔들")
    logger.info(f"   저장 위치: data/ 폴더")
    logger.info("=" * 60)
    
    return True


if __name__ == '__main__':
    try:
        success = collect_one_year_data()
        if success:
            logger.info("이제 model/train_ppo.py를 실행하여 학습할 수 있습니다.")
        else:
            logger.error("데이터 수집 실패")
    except KeyboardInterrupt:
        logger.info("데이터 수집 중단")
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
