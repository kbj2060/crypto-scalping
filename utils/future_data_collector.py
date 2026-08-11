import os
import requests
import pandas as pd
from datetime import datetime, timedelta

# 1. 설정
SYMBOL = "ETHUSDT"
START_DATE = "2025-01-19"
END_DATE = "2026-01-19"
SAVE_FOLDER = "./binance_raw_data"

# 폴더 생성
if not os.path.exists(SAVE_FOLDER):
    os.makedirs(SAVE_FOLDER)

def download_binance_data(data_type, symbol, start_date, end_date):
    """
    data_type: 'metrics' 또는 'fundingRate'
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    
    print(f"--- {data_type} 다운로드 시작 ---")
    
    while current <= end:
        date_str = current.strftime("%Y-%m-%d")
        file_name = f"{symbol}-{data_type}-{date_str}.zip"
        
        # 바이낸스 데이터 서버 경로 규칙
        url = f"https://data.binance.vision/data/futures/um/daily/{data_type}/{symbol}/{file_name}"
        
        save_path = os.path.join(SAVE_FOLDER, file_name)
        
        # 이미 파일이 있으면 건너뜀 (다시 실행할 때 효율적)
        if os.path.exists(save_path):
            current += timedelta(days=1)
            continue
            
        try:
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                with open(save_path, "wb") as f:
                    f.write(response.content)
                print(f"성공: {file_name}")
            elif response.status_code == 404:
                print(f"없음: {file_name} (상장 전이거나 데이터 미생성)")
            else:
                print(f"실패 ({response.status_code}): {file_name}")
        except Exception as e:
            print(f"에러: {file_name} -> {e}")
            
        current += timedelta(days=1)
        # 서버 부하 방지를 위해 살짝 대기
        # time.sleep(0.1)

# 2. 실행
# Metrics(OI 포함)와 FundingRate 각각 실행
download_binance_data("metrics", SYMBOL, START_DATE, END_DATE)
download_binance_data("fundingRate", SYMBOL, START_DATE, END_DATE)

print("\n✨ 모든 다운로드 작업이 완료되었습니다!")