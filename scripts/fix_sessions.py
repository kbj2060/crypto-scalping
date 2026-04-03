import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def update_global_sessions(csv_path: str):
    logger.info(f"🚀 {csv_path} 데이터 로드 시작...")
    df = pd.read_csv(csv_path)
    
    if 'timestamp' not in df.columns:
        logger.error("❌ 'timestamp' 컬럼이 없습니다.")
        return

    # timestamp를 datetime 객체로 변환 (UTC)
    logger.info("⏱️ timestamp UTC 변환 중...")
    ts = pd.to_datetime(df['timestamp'])
    if ts.dt.tz is None:
        ts_utc = ts.dt.tz_localize('UTC')
    else:
        ts_utc = ts.dt.tz_convert('UTC')
        
    start_date = ts_utc.min().date()
    end_date = ts_utc.max().date()
    
    logger.info(f"📅 데이터 기간: {start_date} ~ {end_date}")

    # ===== 1. 아시아 세션 (JPX: 도쿄) =====
    logger.info("🌏 아시아 세션(JPX) 캘린더 스케줄 추출 중...")
    tse = mcal.get_calendar('JPX')
    tse_schedule = tse.schedule(start_date=start_date, end_date=end_date)
    tse_times = mcal.date_range(tse_schedule, frequency='1min')
    df['session_asia'] = ts_utc.isin(tse_times).astype(np.float32)

    # ===== 2. 유럽 세션 (LSE: 런던) =====
    logger.info("💶 유럽 세션(LSE) 캘린더 스케줄 추출 중...")
    lse = mcal.get_calendar('LSE')
    lse_schedule = lse.schedule(start_date=start_date, end_date=end_date)
    lse_times = mcal.date_range(lse_schedule, frequency='1min')
    df['session_europe'] = ts_utc.isin(lse_times).astype(np.float32)

    # ===== 3. 미국 세션 (NYSE: 뉴욕) =====
    logger.info("🦅 미국 세션(NYSE) 캘린더 스케줄 추출 중...")
    nyse = mcal.get_calendar('NYSE')
    nyse_schedule = nyse.schedule(start_date=start_date, end_date=end_date)
    nyse_times = mcal.date_range(nyse_schedule, frequency='1min')
    df['session_us'] = ts_utc.isin(nyse_times).astype(np.float32)

    # 확인 출력
    logger.info("✅ 세션 데이터 업데이트 완료. 요약 정보:")
    logger.info(f"  - 아시아장(session_asia) 활성 시간(개수): {df['session_asia'].sum()}")
    logger.info(f"  - 유럽장(session_europe) 활성 시간(개수): {df['session_europe'].sum()}")
    logger.info(f"  - 미국장(session_us) 활성 시간(개수): {df['session_us'].sum()}")

    logger.info(f"💾 {csv_path} 파일 덮어쓰기 저장 중...")
    df.to_csv(csv_path, index=False)
    logger.info("🎉 모든 과정이 성공적으로 완료되었습니다!")

if __name__ == "__main__":
    CSV_FILE = "/home/llewyn/crypto-scalping/data/training_features_5m.csv"
    update_global_sessions(CSV_FILE)
