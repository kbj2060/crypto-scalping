"""
🚀 [Ultimate RL Feature Evaluator] Quant + ML 통폐합 마스터 스크립트
================================================================
1. Triple Barrier Method (TBM) 적용 (익절/손절/시간 청산)
2. 피처 특성별 자동 평가 (Signal 임계값 vs Continuous 분위수 vs Binary)
3. 퀀트 지표: Rank IC (Information Coefficient), T-Test P-value
4. 머신러닝 지표: Random Forest Feature Importance, Mutual Information (MI)
5. 최종적으로 'RL 에이전트가 쓰기 좋은 A급 피처' 자동 선별
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

# 💡 Sklearn 머신러닝 모듈 추가
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

def apply_triple_barrier(df, tp=0.015, sl=-0.010, max_hold=36):
    """트리플 배리어 라벨링 (기존 로직 유지, 속도 최적화)"""
    closes = df['close'].values
    n = len(closes)
    
    long_returns, short_returns = np.zeros(n), np.zeros(n)
    long_hits, short_hits = np.zeros(n), np.zeros(n)
    
    print(f"📊 캔들 경로 추적 중... (TP: +{tp*100:.1f}%, SL: {sl*100:.1f}%, Time: {max_hold}봉)")
    for i in tqdm(range(n), desc="[Triple Barrier Labeling]", ncols=100):
        entry_price = closes[i]
        if entry_price == 0 or np.isnan(entry_price): continue
            
        l_ret, s_ret = 0.0, 0.0
        hit_l, hit_s = False, False
        hit_l_type, hit_s_type = 0, 0
        
        for j in range(i + 1, min(i + max_hold + 1, n)):
            ret = (closes[j] - entry_price) / entry_price
            
            if not hit_l:
                if ret >= tp: l_ret = tp; hit_l = True; hit_l_type = 1
                elif ret <= sl: l_ret = sl; hit_l = True; hit_l_type = -1
                    
            if not hit_s:
                if ret <= -tp: s_ret = tp; hit_s = True; hit_s_type = 1
                elif ret >= -sl: s_ret = sl; hit_s = True; hit_s_type = -1
                    
            if hit_l and hit_s: break
                
        if not hit_l: l_ret = (closes[min(i + max_hold, n - 1)] - entry_price) / entry_price
        if not hit_s: s_ret = -(closes[min(i + max_hold, n - 1)] - entry_price) / entry_price
            
        long_returns[i], short_returns[i] = l_ret, s_ret
        long_hits[i], short_hits[i] = hit_l_type, hit_s_type
        
    df = df.copy()
    df['tb_long_ret'] = long_returns
    df['tb_short_ret'] = short_returns
    df['tb_long_hit'] = long_hits
    df['tb_short_hit'] = short_hits
    
    return df.iloc[:-max_hold].copy()

def evaluate_master(csv_path, tp=0.015, sl=-0.010, max_hold=36, top_pct=0.2):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna(subset=['close'])
    df = apply_triple_barrier(df, tp, sl, max_hold)

    exclude_cols = ['timestamp', 'close', 'log_return', 'tb_long_ret', 'tb_short_ret', 'tb_long_hit', 'tb_short_hit']
    features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c]) and 'regime_' not in c]

    # 독립 표본 추출 (P-value 계산용)
    df_indep = df.iloc[np.arange(0, len(df), max_hold)]
    
    # =========================================================================
    # 🤖 [Sklearn ML 엔진 가동] 비선형 피처 중요도 분석
    # =========================================================================
    df_clean = df.dropna(subset=features + ['tb_long_ret']).copy()
    X = df_clean[features]
    y = df_clean['tb_long_ret'] # 롱 진입 시의 트리플 배리어 수익률 타겟
    
    print("\n🤖 [Sklearn ML 엔진 가동] 상호정보량(MI) 및 랜덤 포레스트 분석 중...")
    
    # 1. 상호정보량 (Mutual Information) 계산
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=features)

    # 2. Random Forest 피처 중요도 계산
    rf = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_importances = pd.Series(rf.feature_importances_, index=features)

    results = []
    print("\n" + "="*145)
    print(" 🎯 RL 피처 생존 필터링 마스터 리포트 (Quant + ML Cross Validation)")
    print("="*145)

    for feat in tqdm(features, desc="Evaluating Features", ncols=100):
        df_c = df[df[feat].notna()]
        if len(df_c) < 100: continue

        # 3. 랭크 IC (Information Coefficient) 계산 (선형 지표)
        ic_val, _ = stats.spearmanr(df_c[feat], df_c['tb_long_ret'])
        if np.isnan(ic_val): ic_val = 0.0

        # Sklearn 지표 가져오기 (비선형 지표)
        mi_val = mi_series.get(feat, 0.0)
        rf_imp = rf_importances.get(feat, 0.0)

        # 4. 피처 특성별 타점 마스킹 (Signal vs Quantile vs Binary)
        is_signal_feat = feat.startswith('sig_') or feat.startswith('pred_')
        unique_vals = df_c[feat].nunique()
        
        if is_signal_feat:
            threshold = 0.3 if feat.startswith('sig_') else 0.0
            top_mask = df_c[feat] > threshold
            bottom_mask = df_c[feat] < -threshold
            eval_type = f"Signal (>{threshold})"
        elif unique_vals <= 2:
            # 💡 이진(Binary) 피처 구출 로직 (예: session_us)
            top_mask = df_c[feat] == df_c[feat].max()
            bottom_mask = df_c[feat] == df_c[feat].min()
            eval_type = "Binary (1 vs 0)"
        else:
            # 연속형 데이터 상/하위 분위수
            q_high = df_c[feat].quantile(1 - top_pct)
            q_low = df_c[feat].quantile(top_pct)
            top_mask = df_c[feat] > q_high
            bottom_mask = df_c[feat] < q_low
            eval_type = f"Quantile (Top/Bot {top_pct*100:.0f}%)"

        long_cnt = top_mask.sum()
        short_cnt = bottom_mask.sum()
        total_cnt = long_cnt + short_cnt

        if total_cnt == 0: continue

        long_win = (df_c.loc[top_mask, 'tb_long_hit'] == 1).sum()
        short_win = (df_c.loc[bottom_mask, 'tb_short_hit'] == 1).sum()
        long_rets = df_c.loc[top_mask, 'tb_long_ret'].sum()
        short_rets = df_c.loc[bottom_mask, 'tb_short_ret'].sum()

        win_rate = (long_win + short_win) / total_cnt * 100
        expected_edge = (long_rets + short_rets) / total_cnt * 100

        # 독립 표본 T-Test (P-value 계산)
        indep_long = df_indep.loc[df_indep.index.isin(df_c[top_mask].index), 'tb_long_ret']
        indep_short = df_indep.loc[df_indep.index.isin(df_c[bottom_mask].index), 'tb_short_ret']
        combined = np.concatenate([indep_long.values, indep_short.values])

        p_val = 1.0
        if len(combined) > 10:
            t_stat, p_val = stats.ttest_1samp(combined, 0.0)
            if t_stat > 0: p_val = p_val / 2.0 # 단측 검정
            else: p_val = 1.0

        # 💡 [궁극의 RL 추천 필터] 기대수익 양수 & (P-value 0.10 미만 OR IC 0.01 이상 OR RF중요도 0.01 이상 OR MI점수 0.005 이상)
        is_rl_ready = expected_edge > 0 and (p_val < 0.05 or abs(ic_val) >= 0.015 or rf_imp >= 0.015 or mi_val >= 0.010)        

        results.append({
            'feature': feat,
            'type': eval_type,
            'rf_importance': rf_imp,
            'mi_score': mi_val,
            'ic': ic_val,
            'entries': total_cnt,
            'win_rate': win_rate,
            'expected_edge': expected_edge,
            'p_value': p_val,
            'is_rl_ready': is_rl_ready
        })

    # DataFrame 변환 및 정렬 (기대수익률 기준 내림차순 정렬)
    df_res = pd.DataFrame(results).sort_values('expected_edge', ascending=False)
    
    # 출력 포맷팅
    print("\n" + "="*145)
    print(f"{'RL 적합':<5} | {'피처명':<25} | {'RF 중요도':>9} | {'MI 점수':>9} | {'IC 지수':>7} | {'타점':>6} | {'승률':>6} | {'평균수익':>9} | {'P-val':>6} | {'평가방식'}")
    print("-" * 145)
    for _, r in df_res.iterrows():
        mark = "✅" if r['is_rl_ready'] else "  "
        print(f"  {mark:<5} | {r['feature']:<25} | {r['rf_importance']:>9.4f} | {r['mi_score']:>9.4f} | {r['ic']:>7.3f} | {r['entries']:>6} | {r['win_rate']:5.1f}% | {r['expected_edge']:>+8.3f}% | {r['p_value']:>6.4f} | {r['type']}")
    print("="*145)

    # RL 에이전트를 위한 A급 피처 리스트 추출
    rl_features = df_res[df_res['is_rl_ready']]['feature'].tolist()
    print(f"\n🚀 [최종 결론] Quant & ML 교차 검증을 통과한 RL A급 피처 ({len(rl_features)}개):")
    print(rl_features)
    print("\n💡 Tip: train_rl_agent.py의 STATE_DIM을 구성할 때, 위 리스트에 없는 피처는 과감히 버리십시오.")

    return df_res

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data/ensemble/rl_training_data_full.csv')
    args = parser.parse_args()
    
    evaluate_master(args.input)