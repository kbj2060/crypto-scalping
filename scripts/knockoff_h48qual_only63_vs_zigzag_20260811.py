"""direction_head(zigzag_action) 대상, h48qual 후보풀(145)에는 있지만 zig075 후보풀(138)에는
없는 63개 후보(raw 거래량/거래횟수 7개 + ai_*/m7_*/patchtst_* 메타피쳐 약 56개)를 knockoff으로
검증. zig075 소스(eth_features_2024_2026_analysis.csv)엔 ai_*/m7_* 계열이 아예 없어서,
h48qual 소스(trade_candidates_2025_alpha6_current_tail111_exact.csv, 2025년만 존재)에서 가져옴.
TRAIN 윈도우는 zig075의 2024-06~2025-06과 h48qual 패널 존재구간(2025-01~)의 겹치는 부분인
2025-01-01~2025-06-30으로 제한 (전체 매칭 아님, 명시)."""
import time, json, numpy as np, pandas as pd, warnings
from pathlib import Path
from knockpy.knockoff_filter import KnockoffFilter
warnings.filterwarnings('ignore')

R = str(Path(__file__).resolve().parents[1]) + '/'
OUT_DIR = R + 'tmp/eth_h48qual_oracle_label_check_20260811/'
TR_START, TR_END = pd.Timestamp('2025-01-01'), pd.Timestamp('2025-06-30')

ONLY_H48 = ['ai_adverse_risk', 'ai_dir_edge', 'ai_dir_entropy', 'ai_dir_p_down', 'ai_dir_p_flat',
    'ai_dir_p_up', 'ai_flow_exhaustion', 'ai_flow_flip_prob', 'ai_flow_pressure', 'ai_flow_slope',
    'ai_reward_risk', 'ai_vol_regime_pct', 'conf_mdjd', 'conf_patchtst', 'dlinear_smf_ema',
    'dlinear_smf_slope', 'm7_action', 'm7_composite_score', 'm7_confidence', 'm7_expected_ret',
    'm7_gate_block', 'm7_gmm_cluster', 'm7_gmm_conf', 'm7_hold_pred', 'm7_iso_score', 'm7_mtl_dn',
    'm7_mtl_fl', 'm7_mtl_up', 'm7_prob_dn', 'm7_prob_fl', 'm7_prob_up', 'm7_q10', 'm7_q50', 'm7_q90',
    'm7_quality_pred', 'm7_quant_dn', 'm7_quant_fl', 'm7_quant_up', 'm7_qwidth', 'm7_size',
    'm7_tail_risk', 'm7_tp_offset', 'm7_trend_xgb_dn', 'm7_trend_xgb_fl', 'm7_trend_xgb_up',
    'm7_vae_error', 'm7_vae_threshold', 'patchtst_median', 'patchtst_regime_sim', 'pred_mdjd',
    'pred_patchtst', 'quote_volume', 'quote_volume_btc', 'sig_ai_squeeze', 'sig_oi_divergence',
    'sig_whale', 'taker_buy_base', 'taker_buy_quote', 'tide_vol_raw', 'tide_vol_zscore', 'trades',
    'volume', 'volume_btc']

VOLUME_FAMILY = {'quote_volume', 'quote_volume_btc', 'taker_buy_base', 'taker_buy_quote', 'trades', 'volume', 'volume_btc'}


def run_knockoff(X_df, y, tag, seed=260620):
    Xz = ((X_df - X_df.mean()) / X_df.std().replace(0, 1)).to_numpy()
    results = {}
    for fdr in (0.10, 0.20):
        np.random.seed(seed)
        t0 = time.time()
        kf = KnockoffFilter(fstat='randomforest', ksampler='gaussian')
        rej = kf.forward(X=Xz, y=y.astype(float), fdr=fdr)
        sel = [X_df.columns[i] for i in range(len(rej)) if rej[i] == 1]
        print(f'  [{tag}] fdr={fdr:.2f}  n={len(y)}  선택={len(sel)}/{X_df.shape[1]}  ({time.time()-t0:.0f}s)', flush=True)
        print(f'    {sel}', flush=True)
        results[fdr] = sel
        json.dump({str(k): v for k, v in results.items()}, open(OUT_DIR + f'knockoff_only63_{tag}_partial.json', 'w'), indent=1)
    return results


print('h48qual 소스에서 63개 후보 로드 중 (2025-01~06)...', flush=True)
h48_src = pd.read_csv(R + 'tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv',
                       usecols=['timestamp'] + [c for c in ONLY_H48 if c not in VOLUME_FAMILY])
h48_src['timestamp'] = pd.to_datetime(h48_src['timestamp'])

print('zig075 소스에서 volume 계열 + zigzag_action 로드 중...', flush=True)
tech = pd.read_csv(R + 'data/splits/year_oos/eth_features_2024_2026_analysis.csv', low_memory=False,
                    usecols=['timestamp'] + sorted(VOLUME_FAMILY))
tech['timestamp'] = pd.to_datetime(tech['timestamp'])
labels = pd.read_csv(R + 'tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_2025.csv',
                      usecols=['timestamp', 'zigzag_action'])
labels['timestamp'] = pd.to_datetime(labels['timestamp'])

df = labels.merge(tech, on='timestamp', how='inner').merge(h48_src, on='timestamp', how='inner').sort_values('timestamp').reset_index(drop=True)
print(f'병합 후 n={len(df)}  (h48qual 패널 존재구간과 zigzag 라벨 교집합)', flush=True)

tr = (df.timestamp >= TR_START) & (df.timestamp <= TR_END)
X = df[ONLY_H48].apply(pd.to_numeric, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(0.0)[tr].reset_index(drop=True)
act = df['zigzag_action'].to_numpy()[tr.to_numpy()]
print(f'TRAIN(2025-01~06) n={len(X)}', flush=True)

res = {}
res['tradeability'] = run_knockoff(X, (act != 0).astype(int), 'only63-tradeability')
nz = (act != 0)
res['direction'] = run_knockoff(X.loc[nz].reset_index(drop=True), (act[nz] == 1).astype(int), 'only63-direction')

json.dump({k: {str(f): v for f, v in vv.items()} for k, vv in res.items()},
          open(OUT_DIR + 'knockoff_only63_result.json', 'w'), indent=1)
print('\n저장: knockoff_only63_result.json', flush=True)
