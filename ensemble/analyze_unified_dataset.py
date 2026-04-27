
import pandas as pd
import numpy as np
import os

path = '/home/llewyn/crypto-scalping/data/rl_training_2025_unified.csv'

if not os.path.exists(path):
    print(f"Error: File not found at {path}")
    exit(1)

df = pd.read_csv(path)

ai_cols = [
    'patchtst_median', 'patchtst_regime_sim',
    'tide_vol_raw', 'tide_vol_zscore',
    'timesnet_cycle_sin', 'timesnet_cycle_cos', 'timesnet_cycle_delta',
    'dlinear_smf_ema', 'dlinear_smf_slope'
]

# Ensure cols exist
ai_cols = [c for c in ai_cols if c in df.columns]

print(f'### [1] Dataset Overview ###')
print(f'Total Rows: {len(df)}')
print(f'Columns: {len(df.columns)}')
print(f'AI Features Found: {len(ai_cols)}')

print('\n### [2] Missing Values (AI Columns) ###')
nan_counts = df[ai_cols].isna().sum()
print(nan_counts)

print('\n### [3] Descriptive Statistics ###')
stats = df[ai_cols].describe().T[['mean', 'std', 'min', 'max']]
print(stats)

print('\n### [4] Correlation Matrix (AI Features) ###')
corr = df[ai_cols].corr()
print(corr.round(3))

# Check for constant values
print('\n### [5] Variance Check (Unique Values) ###')
for col in ai_cols:
    print(f'{col}: {df[col].nunique()} unique values')

# Look-back window check (the first 256 rows should be NaN or filled)
print('\n### [6] Warm-up Check (First 5 rows) ###')
print(df[ai_cols].head(5))
