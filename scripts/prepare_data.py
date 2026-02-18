"""
데이터 수집 및 피처 엔지니어링 통합 스크립트

이 스크립트는 다음 작업을 순차적으로 수행합니다:
1. 바이낸스에서 ETH/BTC 5분봉 데이터 다운로드
2. 4개의 CSV 파일을 공통 시간 범위로 병합
3. 피처 엔지니어링 적용
4. 최종 학습 데이터 저장

사용법:
    python scripts/prepare_data.py
    
출력 파일:
    - data/eth_5m_1year.csv: ETH 5분봉 원본 데이터
    - data/btc_5m_1year.csv: BTC 5분봉 원본 데이터
    - data/training_features_5m.csv: 피처가 추가된 최종 학습 데이터
"""

import sys
import os
from datetime import datetime, timedelta
import logging

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data_collector import DataCollector
from core.feature_engineering import FeatureEngineer
import pandas as pd

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def step1_download_data():
    """
    Step 1: 바이낸스에서 ETH/BTC 5분봉 데이터 다운로드
    
    Returns:
        bool: 성공 여부
    """
    print("\n" + "=" * 80)
    print("STEP 1: 바이낸스 데이터 다운로드")
    print("=" * 80)
    
    # 파일 존재 확인
    eth_path = 'data/eth_5m_1year.csv'
    btc_path = 'data/btc_5m_1year.csv'
    
    if os.path.exists(eth_path) and os.path.exists(btc_path):
        logger.info("⏭️  파일이 이미 존재합니다. Step 1을 건너뜁니다.")
        logger.info(f"  - {eth_path}")
        logger.info(f"  - {btc_path}")
        return True
    
    try:
        collector = DataCollector()
        
        # 2025년 1월 1일부터 현재까지 5분봉 데이터 수집
        success = collector.collect_and_save_historical_data(
            days=400,  # 1년치 데이터
            timeframe='5m'
        )
        
        if success:
            logger.info("✅ 데이터 다운로드 완료")
            logger.info("  - data/eth_5m_1year.csv")
            logger.info("  - data/btc_5m_1year.csv")
            return True
        else:
            logger.error("❌ 데이터 다운로드 실패")
            return False
            
    except Exception as e:
        logger.error(f"❌ 데이터 다운로드 중 오류 발생: {e}")
        return False


def step2_merge_and_engineer_features():
    """
    Step 2: 4개 CSV 파일 병합 및 피처 엔지니어링
    
    입력 파일:
        - data/eth_5m_1year.csv
        - data/btc_5m_1year.csv
        - data/TOTAL_ETHUSDT_metrics.csv
        - data/TOTAL_ETHFIUSDT_fundingRate.csv
    
    출력 파일:
        - data/training_features_5m.csv
    
    Returns:
        bool: 성공 여부
    """
    print("\n" + "=" * 80)
    print("STEP 2: 데이터 병합 및 피처 생성")
    print("=" * 80)
    
    try:
        # 1. 데이터 로드
        print("\n[1/6] CSV 파일 로드 중...")
        
        eth_path = 'data/eth_5m_1year.csv'
        btc_path = 'data/btc_5m_1year.csv'
        metrics_path = 'data/TOTAL_ETHUSDT_metrics.csv'
        funding_path = 'data/TOTAL_ETHFIUSDT_fundingRate.csv'
        
        # 파일 존재 확인
        for path in [eth_path, btc_path, metrics_path, funding_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
        
        # ── ETH 5분봉 데이터 ──
        eth_df = pd.read_csv(eth_path, parse_dates=['timestamp'])
        print(f"  ✓ ETH 5분봉: {len(eth_df):,}행")
        
        # ── BTC 5분봉 데이터 (rename 하지 않음 — _merge_data()가 처리) ──
        btc_df = pd.read_csv(btc_path, parse_dates=['timestamp'])
        btc_df = btc_df[['timestamp', 'close', 'volume', 'quote_volume']]
        print(f"  ✓ BTC 5분봉: {len(btc_df):,}행")
        
        # ── Metrics 데이터 (5분 간격) ──
        metrics_df = pd.read_csv(metrics_path)
        metrics_df['timestamp'] = pd.to_datetime(metrics_df['create_time'])
        metrics_df = metrics_df.drop(columns=['create_time', 'symbol'], errors='ignore')
        print(f"  ✓ Metrics: {len(metrics_df):,}행")
        
        # Metrics 필수 컬럼 검증
        required_metrics = [
            'sum_toptrader_long_short_ratio', 'count_long_short_ratio',
            'sum_open_interest_value', 'sum_open_interest',
        ]
        missing = [c for c in required_metrics if c not in metrics_df.columns]
        if missing:
            raise ValueError(f"Metrics CSV에 필수 컬럼 누락: {missing}\n"
                           f"  실제 컬럼: {list(metrics_df.columns)}")
        
        # ── Funding Rate 데이터 (4시간 간격) ──
        funding_df = pd.read_csv(funding_path)
        funding_df['timestamp'] = pd.to_datetime(funding_df['calc_time'])
        funding_df = funding_df[['timestamp', 'last_funding_rate']]
        print(f"  ✓ Funding Rate: {len(funding_df):,}행")
        
        # 2. 공통 시간 범위 계산
        print("\n[2/6] 공통 시간 범위 계산 중...")
        
        start_time = max(
            eth_df['timestamp'].min(),
            btc_df['timestamp'].min(),
            metrics_df['timestamp'].min(),
            funding_df['timestamp'].min()
        )
        end_time = min(
            eth_df['timestamp'].max(),
            btc_df['timestamp'].max(),
            metrics_df['timestamp'].max(),
            funding_df['timestamp'].max()
        )
        
        print(f"  ✓ 공통 시작 시간: {start_time}")
        print(f"  ✓ 공통 종료 시간: {end_time}")
        
        # 각 데이터프레임을 공통 시간 범위로 필터링
        eth_df = eth_df[(eth_df['timestamp'] >= start_time) & (eth_df['timestamp'] <= end_time)].copy()
        btc_df = btc_df[(btc_df['timestamp'] >= start_time) & (btc_df['timestamp'] <= end_time)].copy()
        metrics_df = metrics_df[(metrics_df['timestamp'] >= start_time) & (metrics_df['timestamp'] <= end_time)].copy()
        funding_df = funding_df[(funding_df['timestamp'] >= start_time) & (funding_df['timestamp'] <= end_time)].copy()
        
        print(f"  ✓ 필터링 후 ETH: {len(eth_df):,}행")
        print(f"  ✓ 필터링 후 BTC: {len(btc_df):,}행")
        print(f"  ✓ 필터링 후 Metrics: {len(metrics_df):,}행")
        print(f"  ✓ 필터링 후 Funding: {len(funding_df):,}행")
        
        # 3. ETH에 Metrics와 Funding Rate 병합
        print("\n[3/6] 데이터 병합 중 (timestamp 기준)...")
        
        # 병합 전 ETH 쪽 중복 컬럼 제거 (metrics와 겹칠 수 있는 컬럼)
        overlap_cols = [c for c in metrics_df.columns 
                       if c in eth_df.columns and c != 'timestamp']
        if overlap_cols:
            logger.warning(f"ETH-Metrics 중복 컬럼 제거: {overlap_cols}")
            eth_df = eth_df.drop(columns=overlap_cols)
        
        # ETH + Metrics
        eth_merged = pd.merge_asof(
            eth_df.sort_values('timestamp'),
            metrics_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('10min')  # 5min → 10min (API 지연 여유)
        )
        print(f"  ✓ ETH + Metrics: {len(eth_merged):,}행")
        
        # Metrics 병합 후 NaN 비율 체크
        metrics_null_pct = eth_merged[required_metrics].isnull().mean()
        for col, pct in metrics_null_pct.items():
            if pct > 0.1:
                logger.warning(f"  ⚠️ {col} NaN 비율: {pct:.1%} (tolerance 범위 밖 데이터 많음)")
        
        # + Funding Rate (4시간 간격이므로 backward)
        eth_merged = pd.merge_asof(
            eth_merged.sort_values('timestamp'),
            funding_df.sort_values('timestamp'),
            on='timestamp',
            direction='backward',
            tolerance=pd.Timedelta('8h')  # 4h → 8h (결측 방지)
        )
        print(f"  ✓ + Funding Rate: {len(eth_merged):,}행")
        
        # 4. 피처 생성
        # process() 내부의 _merge_data()가 ETH + BTC 병합을 수행
        # btc_df는 원본 컬럼명(close, volume, quote_volume) 그대로 전달
        print("\n[4/6] 피처 생성 중...")
        engineer = FeatureEngineer(candle_minutes=5)
        result = engineer.process(eth_merged, btc_df)
        print(f"  ✓ 피처 생성 완료: {len(result):,}행, {len(result.columns)}개 컬럼")
        
        # 5. 최종 확인
        print("\n[5/6] 최종 데이터 확인...")
        print(f"  ✓ 최종 데이터: {len(result):,}행")
        
        # 6. 저장
        print("\n[6/6] 결과 저장 중...")
        output_path = 'data/training_features_5m.csv'
        result.to_csv(output_path, index=False)
        print(f"  ✓ 저장 완료: {output_path}")
        
        # 요약
        print("\n" + "=" * 80)
        print("✅ 피처 엔지니어링 완료!")
        print("=" * 80)
        print(f"📊 최종 데이터셋:")
        print(f"   - 행 수: {len(result):,}")
        print(f"   - 컬럼 수: {len(result.columns)}")
        print(f"   - 기간: {result['timestamp'].min()} ~ {result['timestamp'].max()}")
        print(f"   - 파일: {output_path}")
        print(f"   - 결측치: {result.isnull().sum().sum()}")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ 피처 엔지니어링 중 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """
    메인 실행 함수
    """
    print("\n" + "=" * 80)
    print("데이터 준비 파이프라인 시작")
    print("=" * 80)
    print(f"시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_time = datetime.now()
    
    # Step 1: 데이터 다운로드
    if not step1_download_data():
        logger.error("❌ Step 1 실패: 데이터 다운로드")
        return False
    
    # Step 2: 병합 및 피처 생성
    if not step2_merge_and_engineer_features():
        logger.error("❌ Step 2 실패: 피처 엔지니어링")
        return False
    
    # 완료
    end_time = datetime.now()
    elapsed = end_time - start_time
    
    print("\n" + "=" * 80)
    print("🎉 모든 작업 완료!")
    print("=" * 80)
    print(f"종료 시간: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"소요 시간: {elapsed}")
    print("\n생성된 파일:")
    print("  1. data/eth_5m_1year.csv - ETH 5분봉 원본 데이터")
    print("  2. data/btc_5m_1year.csv - BTC 5분봉 원본 데이터")
    print("  3. data/training_features_5m.csv - 피처가 추가된 최종 학습 데이터")
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
