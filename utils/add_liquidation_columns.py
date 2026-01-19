"""
기존 CSV 파일에 청산 데이터 컬럼 추가 스크립트
eth_3m_1year.csv와 btc_3m_1year.csv에 liquidation_long, liquidation_short 컬럼 추가
"""
import os
import sys
import pandas as pd
import logging

# 상위 폴더를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def add_liquidation_columns():
    """기존 CSV 파일에 청산 데이터 컬럼 추가"""
    logger.info("=" * 60)
    logger.info("📝 청산 데이터 컬럼 추가 시작")
    logger.info("=" * 60)
    
    eth_file = 'data/eth_3m_1year.csv'
    btc_file = 'data/btc_3m_1year.csv'
    
    # ETH 파일 처리
    if os.path.exists(eth_file):
        logger.info(f"ETH 파일 처리 중: {eth_file}")
        eth_df = pd.read_csv(eth_file, index_col='timestamp', parse_dates=True)
        
        # 청산 데이터 컬럼 추가 (없는 경우만)
        if 'liquidation_long' not in eth_df.columns:
            eth_df['liquidation_long'] = 0.0
            logger.info("  - liquidation_long 컬럼 추가")
        else:
            logger.info("  - liquidation_long 컬럼 이미 존재")
        
        if 'liquidation_short' not in eth_df.columns:
            eth_df['liquidation_short'] = 0.0
            logger.info("  - liquidation_short 컬럼 추가")
        else:
            logger.info("  - liquidation_short 컬럼 이미 존재")
        
        # 백업 생성
        backup_file = eth_file.replace('.csv', '_backup.csv')
        if not os.path.exists(backup_file):
            pd.read_csv(eth_file, index_col='timestamp', parse_dates=True).to_csv(backup_file)
            logger.info(f"  - 백업 파일 생성: {backup_file}")
        
        # 저장
        eth_df.to_csv(eth_file)
        logger.info(f"✅ ETH 파일 업데이트 완료: {len(eth_df)}개 캔들")
    else:
        logger.warning(f"ETH 파일을 찾을 수 없습니다: {eth_file}")
    
    # BTC 파일 처리
    if os.path.exists(btc_file):
        logger.info(f"BTC 파일 처리 중: {btc_file}")
        btc_df = pd.read_csv(btc_file, index_col='timestamp', parse_dates=True)
        
        # 청산 데이터 컬럼 추가 (없는 경우만)
        if 'liquidation_long' not in btc_df.columns:
            btc_df['liquidation_long'] = 0.0
            logger.info("  - liquidation_long 컬럼 추가")
        else:
            logger.info("  - liquidation_long 컬럼 이미 존재")
        
        if 'liquidation_short' not in btc_df.columns:
            btc_df['liquidation_short'] = 0.0
            logger.info("  - liquidation_short 컬럼 추가")
        else:
            logger.info("  - liquidation_short 컬럼 이미 존재")
        
        # 백업 생성
        backup_file = btc_file.replace('.csv', '_backup.csv')
        if not os.path.exists(backup_file):
            pd.read_csv(btc_file, index_col='timestamp', parse_dates=True).to_csv(backup_file)
            logger.info(f"  - 백업 파일 생성: {backup_file}")
        
        # 저장
        btc_df.to_csv(btc_file)
        logger.info(f"✅ BTC 파일 업데이트 완료: {len(btc_df)}개 캔들")
    else:
        logger.warning(f"BTC 파일을 찾을 수 없습니다: {btc_file}")
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("✅ 청산 데이터 컬럼 추가 완료!")
    logger.info("=" * 60)
    logger.info("")
    logger.info("참고: 현재 청산 데이터는 0으로 초기화되어 있습니다.")
    logger.info("실제 청산 데이터를 채우려면 바이낸스 API를 통해 과거 청산 데이터를 수집해야 합니다.")


if __name__ == '__main__':
    try:
        add_liquidation_columns()
    except Exception as e:
        logger.error(f"오류 발생: {e}", exc_info=True)
