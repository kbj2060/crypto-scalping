# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
"""quality_head을 tb_quality 회귀로 바꾸는 안에 맞춰, h48qual(145) + zig075(138, 겹치는 82 제외 56
추가) 합집합 풀(~201) 전체를 다시 스크리닝. 기존 mRMR+knockoff은 zigzag_action/h384_action 같은
이산 클래스 대상 MI라 회귀엔 안 맞았던 도구 -- 이번엔 Spearman 상관 + mutual_info_regression으로
tb_long_quality/tb_short_quality(h384_conservative, 연속값) relevance를 재계산한다.
공통윈도우: zig075 패널(2024-06~2025-06) x h384 라벨 TRAIN(2025-01~2025-09) 교집합 = 2025-01~06."""
import numpy as np, pandas as pd, warnings
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

R = '/home/kbj20/crypto-scalping/'
SC = '/tmp/claude-1000/-home-kbj20-crypto-scalping/f6f0940b-7d19-44da-92ed-ad8db41aed03/scratchpad/'
TR_START, TR_END = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')

DENY_PREFIXES = ('clean_regime4_', 'regime4_pred_', 'regime3_pred_', 'teacher_', 'teacher_oof_', 'a5dir_')
DENY_TOKENS = ('target', 'future', 'label', 'pnl', 'zigzag', 'wave3', 'tp_sl_action_score')
NON_FEATURE = {'timestamp', 'open', 'high', 'low', 'close', 'open_btc', 'high_btc', 'low_btc', 'close_btc'}
# knockoff_quality_h48conservative_384.py와 동일 -- 가격/거의-상수 컬럼, 첫 시도에서 누락해서
# m7_entry_*_price/sum_open_interest_value가 최상위로 새는 걸 확인하고 추가함.
PRICE_LIKE = ['sum_open_interest_value', 'm7_entry_long_price', 'm7_entry_short_price', 'm7_tp_price', 'm7_sl_price']
CONST = ['m7_entry_long_offset', 'm7_entry_short_offset', 'm7_sl_offset', 'm7_gmm_vol_rank',
         'm7_iso_pred', 'm7_iso_anom', 'm7_vae_anom', 'm7_hdb_prob']
# 이 세션에서 이미 가격추세 오염이 확인돼 detrend/diff1으로 치환하기로 한 9개 -- raw는 후보에서
# 빼고 파생판으로 교체 (zig075_knockoff_mrmr.py의 REPLACE와 동일)
REPLACE = {'funding_pressure': ('funding_pressure_diff1', 'diff1'), 'm7_vae_error': ('m7_vae_error_dt288', 'dt288'),
           'last_funding_rate': ('last_funding_rate_dt288', 'dt288'), 'squeeze_power': ('squeeze_power_dt288', 'dt288'),
           'long_squeeze_risk': ('long_squeeze_risk_dt288', 'dt288'), 'funding_abs': ('funding_abs_dt288', 'dt288'),
           'whale_retail_ratio': ('whale_retail_ratio_dt288', 'dt288'),
           'count_long_short_ratio': ('count_long_short_ratio_dt288', 'dt288'),
           'sum_toptrader_long_short_ratio': ('sum_toptrader_long_short_ratio_dt288', 'dt288')}


def _is_candidate(col):
    if col in NON_FEATURE or col in PRICE_LIKE or col in CONST or col in REPLACE:
        return False
    if any(col.startswith(p) for p in DENY_PREFIXES):
        return False
    if any(t in col for t in DENY_TOKENS):
        return False
    return True


print('h48qual 패널 로드 중...', flush=True)
h48 = pd.read_parquet(SC + 'fa_features.parquet')
h48_cols = [c for c in h48.columns if _is_candidate(c) and pd.api.types.is_numeric_dtype(h48[c])]
print(f'h48qual 후보 {len(h48_cols)}개', flush=True)

print('zig075 소스 패널 로드 중...', flush=True)
zig = pd.read_csv(R + 'data/splits/year_oos/eth_features_2024_2026_analysis.csv', low_memory=False)
zig['timestamp'] = pd.to_datetime(zig['timestamp'])
zig_only_cols = [c for c in zig.columns if c not in h48.columns and _is_candidate(c) and pd.api.types.is_numeric_dtype(zig[c])]
print(f'zig075-only 후보 {len(zig_only_cols)}개', flush=True)

# REPLACE 소스 컬럼은 raw 자체는 후보에서 뺐지만 파생 계산엔 필요하니 병합 시점엔 로드
replace_raw_needed = [r for r in REPLACE if r in h48.columns and r not in h48_cols]
h48_load_cols = h48_cols + replace_raw_needed
POOL = sorted(set(h48_cols) | set(zig_only_cols) | {d for d, _ in REPLACE.values()})
print(f'통합 풀 {len(POOL)}개 (raw 후보 {len(h48_cols)+len(zig_only_cols)}개 + REPLACE 파생 {len(REPLACE)}개)', flush=True)

tb = pd.read_csv(R + 'tmp/eth_h384_conservative_triple_barrier_labels_20260811/train_triple_barrier_labels.csv',
                  parse_dates=['timestamp'],
                  usecols=['timestamp', 'tb_long_quality_h384_conservative', 'tb_short_quality_h384_conservative'])

df = h48[['timestamp'] + h48_load_cols].merge(zig[['timestamp'] + zig_only_cols], on='timestamp', how='inner') \
        .merge(tb, on='timestamp', how='inner').sort_values('timestamp').reset_index(drop=True)
for raw, (derived, kind) in REPLACE.items():
    if raw not in df.columns:
        continue
    src = pd.to_numeric(df[raw], errors='coerce').astype(np.float64)
    if kind == 'diff1':
        df[derived] = src.diff(1).fillna(0.0)
    else:
        df[derived] = (src - src.rolling(288, min_periods=96).mean()).fillna(0.0)

tr = (df.timestamp >= TR_START) & (df.timestamp <= TR_END)
X = df.loc[tr, POOL].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)
y_long = df.loc[tr, 'tb_long_quality_h384_conservative'].to_numpy()
y_short = df.loc[tr, 'tb_short_quality_h384_conservative'].to_numpy()
print(f'공통 TRAIN(2025-01~06) n={len(X)}', flush=True)

print('Spearman + MI(regression) 계산 중...', flush=True)
rows = []
for c in POOL:
    x = X[c].to_numpy()
    if np.std(x) < 1e-12:
        continue
    sr_l, _ = spearmanr(x, y_long)
    sr_s, _ = spearmanr(x, y_short)
    rows.append({'feature': c, 'spearman_long': sr_l, 'spearman_short': sr_s,
                 'spearman_max_abs': max(abs(sr_l), abs(sr_s))})
rel = pd.DataFrame(rows).sort_values('spearman_max_abs', ascending=False)

mi_long = mutual_info_regression(X[rel.feature].to_numpy(), y_long, random_state=260620)
mi_short = mutual_info_regression(X[rel.feature].to_numpy(), y_short, random_state=260620)
rel['mi_long'] = mi_long
rel['mi_short'] = mi_short
rel['mi_max'] = rel[['mi_long', 'mi_short']].max(axis=1)

rel.to_csv(SC + 'rescreen_quality_regression_relevance_full.csv', index=False)
print('\n=== Spearman |상관| 상위 30 ===', flush=True)
print(rel.sort_values('spearman_max_abs', ascending=False).head(30)[['feature', 'spearman_long', 'spearman_short', 'mi_long', 'mi_short']].to_string(index=False))
print('\n=== MI(regression) 상위 30 ===', flush=True)
print(rel.sort_values('mi_max', ascending=False).head(30)[['feature', 'mi_long', 'mi_short', 'spearman_long', 'spearman_short']].to_string(index=False))
print(f'\n저장: rescreen_quality_regression_relevance_full.csv  (전체 {len(rel)}개)')
