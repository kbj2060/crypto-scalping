
import os
import sys
import pandas as pd
import numpy as np
import logging

ROOT = '/home/llewyn/crypto-scalping'
sys.path.insert(0, ROOT)

from ensemble.ensemble_router import EnsembleRouter

logging.basicConfig(level=logging.INFO)

n_rows = 1100
df = pd.DataFrame({
    'timestamp': pd.date_range(start='2025-01-01', periods=n_rows, freq='5min'),
    'close': np.random.uniform(2000, 3000, n_rows),
    'session_us': 0, 'hour_cos': 0, 'cvp_poc_dist': 0, 'cvp_volume_imbalance': 0,
    'fvg_dist': 0, 'breakout_strength': 0, 'oi_change_rate': 0, 'ofti': 0, 'kel': 0,
    'mta_funding': 0, 'svps': 0, 'smart_money_flow': 0, 'whale_conviction': 0, 'amihud_illiquidity_z': 0
})

router = EnsembleRouter()
for name, m in router.models.items():
    print(f"Model {name} available: {m.available}")

print('Testing refined features generation...')
# Small data uses live inference path in router (len < 1000)
res = router.get_refined_features(df)
print('\n### Result Sample (last 2 rows) ###')
print(res.tail(2))
print('\nNaN check (AI features should have 1 valid row at the end):')
print(res.notna().sum())
