# 재현성 참고: 이 스크립트는 h48qual 자체 리서치 패널(fa_features.parquet, 145컬럼)에 의존하는데,
# 이 파일은 세션 scratchpad에만 있고 레포에 커밋되지 않았다 -- 기존 FINAL12 dedup 작업과 동일한
# 재현성 갭. docs/experiments/eth_h48qual_final12_feature_selection_20260811.md 참고.
"""rescreen_quality_regression_201pool.py 결과(relevance)에 mRMR식 중복제거 적용.
relevance=mi_max(long/short 중 큰 쪽), redundancy=같은 공통윈도우 상관행렬, |r|>0.5면 낮은 쪽 탈락."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')

R = '/home/kbj20/crypto-scalping/'
SC = '/tmp/claude-1000/-home-kbj20-crypto-scalping/f6f0940b-7d19-44da-92ed-ad8db41aed03/scratchpad/'  # fa_features.parquet만 여기서 로드 (재현성 참고 노트 참고)
OUT_DIR = R + 'tmp/eth_h48qual_odyssey_regression_analysis_20260811/'
TR_START, TR_END = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')

rel = pd.read_csv(OUT_DIR + 'rescreen_quality_regression_relevance_full.csv').sort_values('mi_max', ascending=False)
TOP_N = 30
top = rel.head(TOP_N)['feature'].tolist()
print(f'상위 {TOP_N}개(mi_max 기준)로 중복제거 시작:', flush=True)
print(top, flush=True)

exec(open(R + 'scripts/rescreen_eth_h48qual_quality_regression_pool201_20260811.py').read().split("print('Spearman")[0])

corr = X[top].corr()
remaining = list(top)
dropped = {}
for i, a in enumerate(top):
    if a not in remaining:
        continue
    for b in top[i + 1:]:
        if b not in remaining:
            continue
        r = corr.loc[a, b]
        if abs(r) > 0.5:
            ra = rel.set_index('feature').loc[a, 'mi_max']
            rb = rel.set_index('feature').loc[b, 'mi_max']
            loser = b if ra >= rb else a
            if loser in remaining:
                remaining.remove(loser)
                dropped[loser] = (a if loser == b else b, float(r))

print(f'\n=== 중복제거 후 {len(remaining)}개 생존 ===', flush=True)
rel_idx = rel.set_index('feature')
for f in remaining:
    print(f'  {f:<40s} mi_max={rel_idx.loc[f, "mi_max"]:.4f}', flush=True)
print(f'\n탈락 {len(dropped)}개:', flush=True)
for f, (winner, r) in dropped.items():
    print(f'  {f:<40s} <- 충돌: {winner} (r={r:+.3f})', flush=True)

import json
json.dump({'final': remaining, 'dropped': {k: {'winner': v[0], 'r': v[1]} for k, v in dropped.items()}},
          open(OUT_DIR + 'rescreen_quality_dedup_result.json', 'w'), indent=1, ensure_ascii=False)
print('\n저장: rescreen_quality_dedup_result.json')

print('\n=== 생존 11개 corr(close) 오염 체크 ===', flush=True)
ts_tr = df.loc[tr, 'timestamp'].reset_index(drop=True)
close_tr = ts_tr.to_frame().merge(h48[['timestamp', 'close']], on='timestamp', how='left')['close']
assert len(close_tr) == len(X)
for f in remaining:
    cc = X[f].corr(close_tr)
    flag = ' <-- 오염 의심' if abs(cc) > 0.3 else ''
    print(f'  {f:<40s} corr(close)={cc:+.4f}{flag}', flush=True)
