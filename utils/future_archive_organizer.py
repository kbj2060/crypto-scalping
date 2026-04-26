import pandas as pd
import zipfile
import os
import glob

# 데이터가 들어있는 폴더 경로 (사용자 환경에 맞춰 수정하세요)
data_folder = "./binance_raw_data"

def merge_binance_data(folder_path, pattern, output_name):
    # 지정된 폴더 내에서 패턴이 포함된 모든 .zip 파일 검색
    zip_files = glob.glob(os.path.join(folder_path, f"*{pattern}*.zip"))
    zip_files.sort()  # 날짜순 정렬
    
    if not zip_files:
        print(f"❌ {pattern} 파일을 찾을 수 없습니다.")
        return
    
    all_dfs = []
    print(f"📂 {pattern} 통합 시작 (총 {len(zip_files)}개 파일)...")
    
    for zip_file in zip_files:
        try:
            with zipfile.ZipFile(zip_file, 'r') as z:
                # zip 파일 내의 csv 파일명 확인
                for file_info in z.infolist():
                    if file_info.filename.endswith('.csv'):
                        with z.open(file_info.filename) as f:
                            # 데이터 로드
                            df = pd.read_csv(f)
                            all_dfs.append(df)
        except Exception as e:
            print(f"⚠️ {os.path.basename(zip_file)} 처리 중 오류: {e}")

    if all_dfs:
        # 모든 데이터프레임 하나로 합치기
        final_df = pd.concat(all_dfs, ignore_index=True)
        
        # 시간 관련 컬럼 자동 변환 (create_time 또는 calc_time 등)
        time_cols = ['create_time', 'calc_time', 'timestamp']
        for col in time_cols:
            if col in final_df.columns:
                final_df[col] = pd.to_datetime(final_df[col]) if final_df[col].dtype == 'object' else pd.to_datetime(final_df[col], unit='ms')
                final_df = final_df.sort_values(col)
                break
        
        # 중복 제거 및 최종 저장
        final_df = final_df.drop_duplicates()
        final_df.to_csv(output_name, index=False)
        print(f"✅ 통합 완료: {output_name} (총 {len(final_df)} 행)")
    else:
        print(f"❌ {pattern} 데이터를 합치는 데 실패했습니다.")

# --- 실행 ---
# 1. Metrics (일 단위 파일들) 통합
merge_binance_data(data_folder, "metrics", "TOTAL_ETHUSDT_metrics.csv")

# 2. Funding Rate (월 단위 파일들) 통합
merge_binance_data(data_folder, "fundingRate", "TOTAL_ETHFIUSDT_fundingRate.csv")