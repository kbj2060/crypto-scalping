"""
strategies 폴더에서 실행하는 테스트 스크립트.
실행: cd strategies && python strategies_test.py
"""
import pandas as pd
import numpy as np
import os
import sys

# 이 파일 기준으로 프로젝트 루트 = strategies 의 상위 폴더
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_THIS_DIR)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# 모듈 Import (프로젝트 루트 기준 common, strategies)
try:
    from common.feature_engineering import FeatureEngineer
    from strategies.elite_alpha import WhaleSentimentDivergence, LiquidationSqueezeHunter
    from strategies.elite_structure_flow import OrderblockFVGStrategy, NetTakerFlowStrategy
    from strategies.elite_standard import BTCEthCorrelation, VolatilitySqueeze, VWAPDeviation, HMAMomentum
except ImportError as e:
    print(f"⚠️ 모듈 Import 실패: {e}")
    print("프로젝트 루트에서 실행하거나, strategies 폴더에서 python strategies_test.py 로 실행해주세요.")
    sys.exit(1)

def run_test():
    print("🚀 System Verification Started...\n")

    # 1. Load Data (프로젝트 루트 또는 data/ 폴더 기준)
    print("[1/4] Loading Raw Data...")
    def _data_path(name):
        for base in (_ROOT, os.path.join(_ROOT, "data")):
            p = os.path.join(base, name)
            if os.path.isfile(p):
                return p
        return name
    try:
        btc_df = pd.read_csv(_data_path("btc_3m_1year.csv"))
        eth_df = pd.read_csv(_data_path("integrated_eth_3m_data.csv"))
        print(f"   BTC Data: {btc_df.shape}")
        print(f"   ETH Data: {eth_df.shape}")
    except FileNotFoundError:
        print("❌ CSV 파일을 찾을 수 없습니다. 루트 또는 data/ 폴더에 btc_3m_1year.csv, integrated_eth_3m_data.csv 를 두세요.")
        return

    # 2. Feature Engineering
    print("\n[2/4] Processing Features (Ultimate Feature Set)...")
    fe = FeatureEngineer()
    try:
        processed_df = fe.process(eth_df, btc_df)
        print(f"   ✅ Processed Shape: {processed_df.shape}")
        
        # 필수 컬럼 존재 확인
        required_features = [
            'whale_retail_ratio', 'whale_conviction', 'smart_money_flow', # Alpha
            'net_taker_ratio', 'taker_acceleration', # Order Flow
            'volatility_z', 'rsi', 'bb_width', # Technical
            'btc_corr_60', 'eth_btc_ratio_change' # Market
        ]
        missing = [f for f in required_features if f not in processed_df.columns]
        if missing:
            print(f"   ❌ Missing Columns: {missing}")
        else:
            print("   ✅ All Core Features Created Successfully.")
            
    except Exception as e:
        print(f"   ❌ Feature Engineering Error: {e}")
        return

    # 3. Initialize Strategies (The Elite 8)
    print("\n[3/4] Initializing Elite 8 Strategies...")
    strategies = [
        WhaleSentimentDivergence(),
        LiquidationSqueezeHunter(),
        OrderblockFVGStrategy(),
        NetTakerFlowStrategy(),
        BTCEthCorrelation(),
        VolatilitySqueeze(),
        VWAPDeviation(),
        HMAMomentum()
    ]
    
    # 4. Run Strategy Backtest (Signal Check)
    print("\n[4/4] Generating Signals (Sampling last 10,000 candles)...")
    
    # 속도를 위해 최근 10,000개 데이터만 샘플링하여 테스트
    sample_df = processed_df.iloc[-10000:].reset_index(drop=True)
    
    results = {}
    
    for strategy in strategies:
        signals = []
        print(f"   👉 Testing: {strategy.name}...", end='\r')
        
        for i, row in sample_df.iterrows():
            # 실제 환경과 유사하게 현재 시점까지의 데이터를 전달 (Lookback 필요 시)
            # 여기서는 속도를 위해 row만 넘기거나 전체 df를 넘깁니다.
            # (Strategy 구현 방식에 따라 df 전달 방식이 다를 수 있음)
            sig = strategy.generate_signal(row, sample_df)
            signals.append(sig)
            
        signal_counts = pd.Series(signals).value_counts().to_dict()
        results[strategy.name] = signal_counts
        print(f"   ✅ {strategy.name:<25}: {signal_counts}")

    print("\n🎉 Test Complete! Check the signal distribution above.")

if __name__ == "__main__":
    run_test()