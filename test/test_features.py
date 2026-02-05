import pandas as pd
import numpy as np
import sys
import os

# 경로 설정
sys.path.append(os.getcwd())

try:
    from common.feature_engineering import FeatureEngineer
except ImportError:
    print("❌ common/feature_engineering.py를 찾을 수 없습니다.")
    sys.exit(1)

def inspect_features():
    print("🔬 Feature Deep Inspection Started...\n")

    # 1. 데이터 로드
    try:
        print("[1/3] Loading Data...")
        btc = pd.read_csv('data/btc_3m_1year.csv')
        eth = pd.read_csv('data/integrated_eth_3m_data.csv')
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return

    # 2. 피처 생성
    print("[2/3] Generating Features...")
    fe = FeatureEngineer()
    df = fe.process(eth, btc)
    
    # 3. 정밀 진단 (Deep Inspection)
    print("\n[3/3] Analyzing Feature Statistics (Last 10,000 candles)...")
    sample = df.iloc[-10000:]
    
    # 검사할 주요 피처 목록 및 정상 범위 예상
    check_list = {
        # Feature Name : (Min Expected, Max Expected, Description)
        'bb_width': (0.002, 1.0, "Absolute Width"),
        'bb_width_z': (-3.0, 3.0, "Relative Squeeze"),  # Z-Score: -2~+2 대부분, 스퀴즈 전략용
        'rsi': (0, 100, "RSI (0~100)"),
        'btc_corr_60': (-1, 1, "Correlation (-1~1)"),
        'whale_retail_ratio': (0, 100, "Whale/Retail Ratio"),
        'net_taker_ratio': (-1, 1, "Net Taker Ratio"),
        'funding_pressure': (-0.1, 0.1, "Funding Pressure"),
        'volatility_z': (-5, 5, "Volatility Z-Score")
    }
    
    print(f"{'Feature':<20} | {'Mean':<10} | {'Min':<10} | {'Max':<10} | {'NaNs':<5} | {'Status'}")
    print("-" * 85)
    
    for col, (min_exp, max_exp, desc) in check_list.items():
        if col not in df.columns:
            print(f"{col:<20} | {'MISSING':<40} | ❌ Not Created")
            continue
            
        series = sample[col]
        mean_val = series.mean()
        min_val = series.min()
        max_val = series.max()
        nan_cnt = series.isna().sum()
        
        # 상태 진단
        status = "✅ OK"
        if nan_cnt > 0:
            status = "⚠️ Has NaNs"
        elif min_val < min_exp and col not in ('volatility_z', 'bb_width_z'):  # Z-Score 계열 예외
            status = f"⚠️ Low Value (Check {min_exp})"
        elif max_val > max_exp and col == 'rsi':
            status = "⚠️ RSI > 100"
        elif mean_val == 0:
            status = "⚠️ All Zeros"
            
        # BB Width 특별 점검 (너무 작으면 문제)
        if col == 'bb_width' and mean_val < 0.002:
            status = "❌ Too Small (Scale Issue?)"
        # bb_width_z: 대부분 -2~+2 구간이면 정상
        if col == 'bb_width_z' and (min_val < -5 or max_val > 5):
            status = "⚠️ Z-Score Extreme (Check Rolling Window)"

        print(f"{col:<20} | {mean_val:10.4f} | {min_val:10.4f} | {max_val:10.4f} | {nan_cnt:<5} | {status}")

    print("-" * 85)
    
    # 4. 전략별 핵심 피처 샘플 출력
    print("\n🧐 Sample Data (Last 5 rows):")
    cols = ['timestamp', 'close', 'bb_width', 'bb_width_z', 'whale_retail_ratio', 'net_taker_ratio']
    print(sample[[c for c in cols if c in sample.columns]].tail())

if __name__ == "__main__":
    inspect_features()