
import os
import sys
import pandas as pd
import numpy as np

def prepare_timesnet_vwap_data():
    input_path = "/home/llewyn/crypto-scalping/data/splits/year_oos/training_features_2025.csv"
    output_path = "/home/llewyn/crypto-scalping/data/timesnet_vwap_train.csv"
    window = 60 # 5시간 롤링 윈도우
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found.")
        return

    print("Loading data...")
    df = pd.read_csv(input_path)
    
    required = ['timestamp', 'high', 'low', 'close', 'volume']
    for col in required:
        if col not in df.columns:
            print(f"Error: Missing column {col}")
            return

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp')
    
    print(f"Calculating {window}-period Rolling VWAP deviation...")
    tp = (df['high'] + df['low'] + df['close']) / 3
    v = df['volume']
    
    # Rolling VWAP: sum(P*V) / sum(V) over window
    pv_sum = (tp * v).rolling(window=window).sum()
    v_sum = v.rolling(window=window).sum().replace(0, 1)
    df['vwap'] = pv_sum / v_sum
    
    # 이격도 (%)
    df['y'] = (df['close'] / df['vwap'] - 1) * 100
    df['unique_id'] = 'ETH'
    df['ds'] = df['timestamp']
    
    exog_cols = [
        "session_us", "hour_cos", "cvp_poc_dist", "cvp_volume_imbalance",
        "fvg_dist", "breakout_strength", "oi_change_rate", "ofti", "kel",
        "mta_funding", "svps"
    ]
    
    final_cols = ['unique_id', 'ds', 'y'] + [c for c in exog_cols if c in df.columns]
    train_df = df[final_cols].dropna()
    
    train_df.to_csv(output_path, index=False)
    print(f"Success! VWAP Deviation training data saved to {output_path}")
    print(train_df.head())

if __name__ == "__main__":
    prepare_timesnet_vwap_data()
