"""
1년치 학습 데이터 수집 스크립트 (래퍼)
DataCollector의 collect_and_save_historical_data 메서드를 호출합니다.
"""
import os
import sys
import logging
import pandas as pd
from datetime import datetime

# 상위 폴더를 경로에 추가 (config, core 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.data_collector import DataCollector

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


def check_data_files_exist(timeframe=None):
    """필수 데이터 파일들이 모두 존재하는지 확인
    
    Args:
        timeframe: 타임프레임 (기본값: config.TIMEFRAME)
    
    Returns:
        bool: 필수 파일(ETH, BTC 캔들)이 모두 존재하면 True
    """
    if timeframe is None:
        timeframe = config.TIMEFRAME
    
    # 필수 파일 목록 (캔들 데이터만 - 펀딩비는 선택사항)
    required_files = [
        f'data/eth_{timeframe}_1year.csv',
        f'data/btc_{timeframe}_1year.csv'
    ]
    
    # 모든 필수 파일 존재 여부 확인
    all_exist = all(os.path.exists(f) for f in required_files)
    
    if all_exist:
        logger.info("=" * 60)
        logger.info("✅ 필수 데이터 파일이 이미 존재합니다.")
        for f in required_files:
            if os.path.exists(f):
                file_size = os.path.getsize(f) / (1024 * 1024)  # MB 단위
                logger.info(f"   ✓ {f} ({file_size:.2f} MB)")
        
        # 펀딩비 파일도 확인 (선택사항)
        funding_file = 'data/eth_funding_rate_1year.csv'
        if os.path.exists(funding_file):
            file_size = os.path.getsize(funding_file) / (1024 * 1024)
            logger.info(f"   ✓ {funding_file} ({file_size:.2f} MB) [선택사항]")
        else:
            logger.info(f"   ⚠ {funding_file} 없음 (선택사항, 필요시 수집됨)")
        
        logger.info("=" * 60)
    
    return all_exist


def get_data_time_range(timeframe=None):
    """기존 캔들 데이터 파일에서 시작/종료 시간 추출
    
    Args:
        timeframe: 타임프레임 (기본값: config.TIMEFRAME)
    
    Returns:
        tuple: (start_time, end_time) datetime 객체 또는 (None, None)
    """
    if timeframe is None:
        timeframe = config.TIMEFRAME
    
    eth_file = f'data/eth_{timeframe}_1year.csv'
    
    try:
        if not os.path.exists(eth_file):
            return None, None
        
        # ETH 파일에서 타임스탬프 읽기
        eth_df = pd.read_csv(eth_file, index_col='timestamp', parse_dates=True)
        if len(eth_df) == 0:
            return None, None
        
        start_time = eth_df.index[0].to_pydatetime()
        end_time = eth_df.index[-1].to_pydatetime()
        
        return start_time, end_time
    except Exception as e:
        logger.error(f"데이터 시간 범위 추출 실패: {e}")
        return None, None


def collect_one_year_data():
    """1년치 데이터 수집 및 저장 (DataCollector 사용)"""
    try:
        timeframe = config.TIMEFRAME
        
        # 필수 파일 존재 여부 확인
        required_files = [
            f'data/eth_{timeframe}_1year.csv',
            f'data/btc_{timeframe}_1year.csv'
        ]
        required_exist = all(os.path.exists(f) for f in required_files)
        
        if required_exist:
            # 필수 파일이 모두 존재하는 경우
            logger.info("=" * 60)
            logger.info("✅ 필수 데이터 파일이 이미 존재합니다.")
            for f in required_files:
                if os.path.exists(f):
                    file_size = os.path.getsize(f) / (1024 * 1024)  # MB 단위
                    logger.info(f"   ✓ {f} ({file_size:.2f} MB)")
            logger.info("=" * 60)
            logger.info("모든 데이터 파일이 존재합니다. 수집을 건너뜁니다.")
            return True
        
        # 필수 파일이 없는 경우 전체 수집
        logger.info("필수 데이터 파일이 없어 전체 데이터를 수집합니다.")
        data_collector = DataCollector(use_saved_data=False)
        success = data_collector.collect_and_save_historical_data(days=365)
        
        return success
    except Exception as e:
        logger.error(f"데이터 수집 중 오류 발생: {e}", exc_info=True)
        return False


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
