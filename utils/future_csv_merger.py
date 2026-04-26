import pandas as pd
import numpy as np

def integrate_binance_data():
    print("데이터 로드 중...")
    # 1. 파일 로드
    # index_col을 사용하여 타임스탬프를 바로 인덱스로 설정합니다.
    price_df = pd.read_csv('data/eth_3m_1year.csv')
    metrics_df = pd.read_csv('data/TOTAL_ETHUSDT_metrics.csv')
    funding_df = pd.read_csv('data/TOTAL_ETHFIUSDT_fundingRate.csv')

    # 2. 시간 컬럼 변환 및 정렬
    price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
    metrics_df['create_time'] = pd.to_datetime(metrics_df['create_time'])
    funding_df['calc_time'] = pd.to_datetime(funding_df['calc_time'])

    # 중복 제거 (병합 에러 방지)
    price_df = price_df.drop_duplicates('timestamp').set_index('timestamp').sort_index()
    metrics_df = metrics_df.drop_duplicates('create_time').set_index('create_time').sort_index()
    funding_df = funding_df.drop_duplicates('calc_time').set_index('calc_time').sort_index()

    print("데이터 병합 및 보간 시작...")

    # 3. 미체결 약정(Metrics) 병합: 5분 -> 3분 선형 보간
    # 가격 데이터의 3분 축과 Metrics의 5분 축을 합친 후 보간하고 다시 3분 축만 추출합니다.
    metrics_combined = pd.concat([pd.DataFrame(index=price_df.index), metrics_df], axis=1)
    # symbol 컬럼 등 문자열은 보간이 안되므로 수치형만 선택하거나 제거
    if 'symbol' in metrics_combined.columns:
        metrics_combined = metrics_combined.drop(columns=['symbol'])
    
    metrics_resampled = metrics_combined.interpolate(method='time').reindex(price_df.index)

    # 4. 펀딩비(Funding Rate) 병합: 8시간 -> 3분 직전 값 채우기(Forward Fill)
    # 펀딩비는 다음 갱신 전까지 동일한 값이 유지되므로 ffill이 가장 정확합니다.
    funding_combined = pd.concat([pd.DataFrame(index=price_df.index), funding_df], axis=1)
    funding_resampled = funding_combined.ffill().reindex(price_df.index)

    # 5. 최종 결합
    # 모든 컬럼을 하나로 합칩니다.
    final_df = pd.concat([price_df, metrics_resampled, funding_resampled], axis=1)

    # 시작 시점이 달라 발생하는 앞부분의 결측치는 뒷 데이터로 채움 (bfill)
    final_df = final_df.bfill()

    # 6. 결과 저장
    output_file = 'data/integrated_eth_3m_data.csv'
    final_df.to_csv(output_file)
    
    print("-" * 30)
    print(f"✅ 통합 완료: {output_file}")
    print(f"📊 최종 데이터 크기: {final_df.shape}")
    print(f"📅 기간: {final_df.index.min()} ~ {final_df.index.max()}")
    print("-" * 30)
    print(final_df.head())

if __name__ == "__main__":
    integrate_binance_data()