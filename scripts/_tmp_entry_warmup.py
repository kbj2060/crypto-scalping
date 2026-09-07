import sys, warnings; warnings.filterwarnings("ignore")
from pathlib import Path
ROOT = Path("/home/llewyn/crypto-scalping")
for p in (ROOT, ROOT/"scripts"):
    if str(p) not in sys.path: sys.path.insert(0, str(p))
import numpy as np, pandas as pd, joblib
import live_eth_entry_limit_fade_signal_20260903 as M
from features.engineering import FeatureEngineer
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12

A, CARD = M._art(); FE = A["feature_cols"]
kl, btc_kl, eth_df, btc_df = M._assemble()
live = _with_raw_state12(FeatureEngineer().process(eth_df, btc_df))
print(f"kl {len(kl):,}봉 · FeatureEngineer 출력 {len(live):,}행  (drop {len(kl)-len(live):,})")

F = M._feature_frame(kl, eth_df, btc_df, FE)
cols = [c for c in FE if c in F.columns]
V = F[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf,-np.inf], np.nan)
ok = V.notna().all(axis=1).to_numpy()
print(f"피쳐 {len(cols)}개 전부 유효한 봉: {ok.sum():,}/{len(ok):,}")
if ok.any():
    first = int(np.argmax(ok))
    print(f"  첫 유효 인덱스 {first:,} (= 앞 {first:,}봉 {first/288:.1f}일이 워밍업)")
    print(f"  연속 유효 꼬리 {int(ok[first:].sum()):,}봉 = {ok[first:].sum()/288:.1f}일")
# 어느 컬럼이 워밍업을 지배하나
nanc = V.isna().sum().sort_values(ascending=False)
print("\n결측 상위 12개 컬럼 (봉 수):")
for c, v in nanc.head(12).items():
    if v: print(f"  {c:34s} {int(v):5,}")
print(f"\n결측이 있는 컬럼 {int((nanc>0).sum())}/{len(cols)}")
