"""
1년치 학습 데이터 수집 스크립트 (래퍼)
DataCollector의 collect_and_save_historical_data 메서드를 호출합니다.
"""
import os
import sys
import logging

# 상위 폴더를 경로에 추가 (config, core 모듈 접근용)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


def collect_one_year_data():
    """1년치 데이터 수집 및 저장 (DataCollector 사용)"""
    try:
        # DataCollector 초기화 (use_saved_data=False: 데이터 수집 모드)
        data_collector = DataCollector(use_saved_data=False)
        
        # 1년치 데이터 수집 및 저장
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
