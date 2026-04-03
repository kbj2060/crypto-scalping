"""
🚀 [Ultimate RL Feature Evaluator] Quant + ML 통폐합 마스터 스크립트
================================================================
1. Triple Barrier Method (TBM) 적용 (익절/손절/시간 청산)
2. 피처 특성별 자동 평가 (Signal 임계값 vs Continuous 분위수 vs Binary)
3. 퀀트 지표: Rank IC (Information Coefficient), T-Test P-value
4. 머신러닝 지표: Random Forest Feature Importance, Mutual Information (MI)
5. pred+conf 복합 신호 평가 (고신뢰 예측이 실제로 더 정확한가)
6. Regime별 조건부 IC 분석
7. 최종적으로 'RL 에이전트가 쓰기 좋은 A급 피처' 자동 선별
"""

import os
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor

# ─── 피처 그룹 정의 (train_rl_agent.py와 동기화) ────────────────────────────
MODEL_PRED          = ['pred_timesfm', 'pred_chronos', 'pred_ttm', 'pred_patchtst', 'pred_tide', 'pred_mdjd']
MODEL_CONF          = ['conf_timesfm', 'conf_chronos', 'conf_ttm', 'conf_patchtst', 'conf_tide', 'conf_mdjd']
ELITE_COLS          = ['sig_whale', 'sig_orderblock', 'sig_oi_divergence', 'sig_ai_squeeze']
ALPHA_7_COLS        = ['session_us', 'hour_cos', 'cvp_poc_dist', 'cvp_volume_imbalance', 'fvg_dist', 'breakout_strength', 'oi_change_rate']
REGIME_COLS         = ['regime_chop', 'regime_whipsaw', 'regime_bull', 'regime_bear', 'regime_normal']
SYNTHETIC_ALPHA_COLS = ['ofti', 'kel', 'mta_funding', 'svps', 'mshd', 'fvci',
                        'wpad', 'fdlv', 'vsdi', 'vebr', 'tlad', 'mtmb', 'fcsz']

# pred 칼럼과 대응하는 conf 칼럼 매핑
PRED_CONF_PAIRS = {p: p.replace('pred_', 'conf_') for p in MODEL_PRED if p.replace('pred_', 'conf_') in MODEL_CONF}

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

def evaluate_master(csv_path, tp_mult=2.0, sl_mult=-1.0, max_hold=36, top_pct=0.2):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna(subset=['close'])
    
    # 1. 동적 배리어 적용
    df = apply_dynamic_triple_barrier(df, tp_mult, sl_mult, max_hold)

    tb_cols = ['tb_long_ret', 'tb_short_ret', 'tb_long_hit', 'tb_short_hit']
    fwd_cols = [f'fwd_ret_{h}' for h in [6, 12, 24, 36]]
    exclude_cols = {'timestamp', 'close', 'log_return'} | set(tb_cols) | set(fwd_cols)

    features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    df_clean = df.dropna(subset=features + tb_cols).copy()
    X = df_clean[features]
    
    # [수정 완료] 명확한 분류 타겟: 롱 익절(+1), 숏 익절(-1), 그 외(0)
    y_dir = np.zeros(len(df_clean))
    y_dir[(df_clean['tb_long_hit'] == 1)] = 1
    y_dir[(df_clean['tb_short_hit'] == 1)] = -1

    print("\n🤖 [Sklearn ML 엔진 가동] 상호정보량(MI) 및 랜덤 포레스트(RF) 분석 중...")
    mi_scores = mutual_info_classif(X, y_dir, random_state=42)
    mi_series = pd.Series(mi_scores, index=features)

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y_dir)
    rf_importances = pd.Series(rf.feature_importances_, index=features)

    results = []
    for feat in tqdm(features, desc="Evaluating Features", ncols=100):
        df_c = df[df[feat].notna()].copy()
        if len(df_c) < 100: continue

        ic_val, _ = stats.spearmanr(df_c[feat], df_c['tb_long_ret'])
        if np.isnan(ic_val): ic_val = 0.0
        
        # ── (신규 로직) IC Decay 분석 ──
        ic_decays = {}
        for h in [6, 12, 24, 36]:
            ic, _ = stats.spearmanr(df_c[feat], df_c[f'fwd_ret_{h}'])
            ic_decays[h] = 0.0 if np.isnan(ic) else ic

        # ── (신규 로직) 10분위 스프레드 (Decile Spread) 분석 ──
        decile_spread = 0.0
        if _feature_type(feat, df_c) == 'continuous':
            try:
                # 값을 기준으로 10등분, 중복값은 버림
                df_c['decile'] = pd.qcut(df_c[feat], 10, labels=False, duplicates='drop')
                q_max, q_min = df_c['decile'].max(), df_c['decile'].min()
                if q_max > q_min:
                    mean_q_max = df_c[df_c['decile'] == q_max]['tb_long_ret'].mean()
                    mean_q_min = df_c[df_c['decile'] == q_min]['tb_long_ret'].mean()
                    decile_spread = (mean_q_max - mean_q_min) * 100 # 퍼센트로 변환
            except Exception:
                pass

        top_mask, bottom_mask, eval_type = _masks(feat, df_c, top_pct)
        stats_res = _edge_stats(df_c, top_mask, bottom_mask, max_hold)
        if stats_res is None: continue

        # RL Ready 조건에 10분위 스프레드와 IC 평균치 추가
        avg_decay_ic = sum(abs(v) for v in ic_decays.values()) / 4.0
        is_rl_ready = (
            stats_res['expected_edge'] > 0 and
            (stats_res['p_value'] < 0.05 or abs(ic_val) >= 0.015
             or rf_importances.get(feat, 0) >= 0.015 or abs(decile_spread) > 0.1)
        )
        
        results.append({
            'feature': feat, 'type': eval_type,
            'rf_imp': rf_importances.get(feat, 0), 'mi_score': mi_series.get(feat, 0), 
            'ic': ic_val, 'ic_decay_avg': avg_decay_ic, 'decile_spread': decile_spread,
            **stats_res, 'is_rl_ready': is_rl_ready
        })

    df_res = pd.DataFrame(results).sort_values('expected_edge', ascending=False)

    W = 160
    print("\n" + "=" * W)
    print(" 🎯 [표 1] RL 피처 마스터 리포트 (Dynamic ATR TBM + Decile Spread)")
    print("=" * W)
    print(f"{'RL':<3} | {'피처명':<27} | {'RF':>7} | {'MI':>7} | {'IC':>6} | {'Spread':>8} | {'IC_D(avg)':>9} | {'승률':>6} | {'기대수익':>9} | {'P-val':>6}")
    print("-" * W)

    for _, r in df_res.iterrows():
        mark = "✅" if r['is_rl_ready'] else "  "
        print(f" {mark:<3} | {r['feature']:<27} | {r['rf_imp']:>7.3f} | {r['mi_score']:>7.3f} | {r['ic']:>6.3f} | {r['decile_spread']:>7.3f}% | {r['ic_decay_avg']:>9.3f} | {r['win_rate']:5.1f}% | {r['expected_edge']:>+8.3f}% | {r['p_value']:>6.4f}")
    print("=" * W)
    
    return df_res
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
                if ret >= tp:  l_ret = tp;  hit_l = True; hit_l_type =  1
                elif ret <= sl: l_ret = sl;  hit_l = True; hit_l_type = -1

            if not hit_s:
                if ret <= -tp: s_ret = tp;  hit_s = True; hit_s_type =  1
                elif ret >= -sl: s_ret = sl; hit_s = True; hit_s_type = -1

            if hit_l and hit_s: break

        if not hit_l: l_ret = (closes[min(i + max_hold, n - 1)] - entry_price) / entry_price
        if not hit_s: s_ret = -(closes[min(i + max_hold, n - 1)] - entry_price) / entry_price

        long_returns[i], short_returns[i] = l_ret, s_ret
        long_hits[i], short_hits[i] = hit_l_type, hit_s_type

    df = df.copy()
    df['tb_long_ret']  = long_returns
    df['tb_short_ret'] = short_returns
    df['tb_long_hit']  = long_hits
    df['tb_short_hit'] = short_hits

    return df.iloc[:-max_hold].copy()


def _feature_type(feat, df_c):
    """피처 특성 자동 판별"""
    if feat.startswith('sig_') or feat.startswith('pred_'):
        return 'signal'
    if feat in REGIME_COLS or df_c[feat].nunique() <= 2:
        return 'binary'
    return 'continuous'


def _masks(feat, df_c, top_pct=0.2):
    """피처 타입별 롱/숏 타점 마스크 반환"""
    ftype = _feature_type(feat, df_c)
    if ftype == 'signal':
        threshold = 0.3 if feat.startswith('sig_') else 0.0
        top_mask    = df_c[feat] > threshold
        bottom_mask = df_c[feat] < -threshold
        eval_type   = f"Signal (>{threshold})"
    elif ftype == 'binary':
        top_mask    = df_c[feat] == df_c[feat].max()
        bottom_mask = df_c[feat] == df_c[feat].min()
        eval_type   = "Binary (1 vs 0)"
    else:
        q_high = df_c[feat].quantile(1 - top_pct)
        q_low  = df_c[feat].quantile(top_pct)
        top_mask    = df_c[feat] > q_high
        bottom_mask = df_c[feat] < q_low
        eval_type   = f"Quantile (Top/Bot {top_pct*100:.0f}%)"
    return top_mask, bottom_mask, eval_type

def _edge_stats(df_c, top_mask, bottom_mask, max_hold=36):
    """승률/기대수익/p-value 독립 표본 계산"""
    long_cnt  = top_mask.sum()
    short_cnt = bottom_mask.sum()
    total_cnt = long_cnt + short_cnt
    if total_cnt == 0:
        return None

    long_win   = (df_c.loc[top_mask,    'tb_long_hit']  == 1).sum()
    short_win  = (df_c.loc[bottom_mask, 'tb_short_hit'] == 1).sum()
    long_rets  = df_c.loc[top_mask,    'tb_long_ret'].sum()
    short_rets = df_c.loc[bottom_mask, 'tb_short_ret'].sum()

    win_rate      = (long_win + short_win) / total_cnt * 100
    expected_edge = (long_rets + short_rets) / total_cnt * 100

    # [수정 완료] 겹침 방지를 위해 마스킹된 타점 중 max_hold 이격된 샘플만 필터링
    long_idx = df_c.index[top_mask]
    short_idx = df_c.index[bottom_mask]
    
    indep_long_rets, last_idx = [], -999
    for idx in long_idx:
        if idx - last_idx >= max_hold:
            indep_long_rets.append(df_c.loc[idx, 'tb_long_ret'])
            last_idx = idx
            
    indep_short_rets, last_idx = [], -999
    for idx in short_idx:
        if idx - last_idx >= max_hold:
            indep_short_rets.append(df_c.loc[idx, 'tb_short_ret'])
            last_idx = idx

    combined = np.concatenate([indep_long_rets, indep_short_rets])

    p_val = 1.0
    if len(combined) > 10:
        t_stat, p_val = stats.ttest_1samp(combined, 0.0)
        p_val = p_val / 2.0 if t_stat > 0 else 1.0

    return {'entries': total_cnt, 'win_rate': win_rate,
            'expected_edge': expected_edge, 'p_value': p_val}

def calculate_atr_pct(df, window=14):
    """(1) ATR 퍼센티지 계산 (고가/저가가 없으면 종가 변동성으로 근사)"""
    if 'high' in df.columns and 'low' in df.columns:
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift(1)).abs()
        tr3 = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window, min_periods=1).mean()
    else:
        # 고가/저가가 없을 경우 종가의 변동성(Rolling Std)을 활용한 근사 ATR
        returns = df['close'].pct_change().fillna(0)
        atr = returns.rolling(window=window, min_periods=1).std() * df['close'] * np.sqrt(window)
    
    return atr / df['close']

def apply_dynamic_triple_barrier(df, tp_mult=2.0, sl_mult=-1.0, max_hold=36):
    """(2) 변동성 조정 트리플 배리어 (Dynamic ATR-based TBM)"""
    atr_pct = calculate_atr_pct(df, window=14)
    
    # 횡보장 노이즈 청산을 막기 위한 최소 임계값 설정 (최소 TP 0.5%, SL -0.3%)
    tp_pcts = np.maximum(atr_pct * tp_mult, 0.005).fillna(0.015).values
    sl_pcts = np.minimum(atr_pct * sl_mult, -0.003).fillna(-0.010).values
    
    closes = df['close'].values
    n = len(closes)

    long_returns, short_returns = np.zeros(n), np.zeros(n)
    long_hits, short_hits = np.zeros(n), np.zeros(n)
    
    # (3) IC Decay용 미래 수익률 사전 계산 (6, 12, 24, 36스텝)
    df = df.copy()
    for h in [6, 12, 24, 36]:
        df[f'fwd_ret_{h}'] = df['close'].shift(-h) / df['close'] - 1.0

    print(f"📊 캔들 경로 추적 중... (동적 ATR 배리어 적용, Time: {max_hold}봉)")
    for i in tqdm(range(n), desc="[Dynamic TBM Labeling]", ncols=100):
        entry_price = closes[i]
        if entry_price == 0 or np.isnan(entry_price): continue

        tp, sl = tp_pcts[i], sl_pcts[i]
        l_ret, s_ret = 0.0, 0.0
        hit_l, hit_s = False, False
        hit_l_type, hit_s_type = 0, 0

        for j in range(i + 1, min(i + max_hold + 1, n)):
            ret = (closes[j] - entry_price) / entry_price

            if not hit_l:
                if ret >= tp:  l_ret = tp;  hit_l = True; hit_l_type =  1
                elif ret <= sl: l_ret = sl;  hit_l = True; hit_l_type = -1

            if not hit_s:
                # [수정 완료] 숏 부호 반전 버그 해결 (수익일 때 양수, 손실일 때 음수)
                if ret <= -tp: s_ret = tp;  hit_s = True; hit_s_type =  1
                elif ret >= -sl: s_ret = sl; hit_s = True; hit_s_type = -1

            if hit_l and hit_s: break

        if not hit_l: l_ret = (closes[min(i + max_hold, n - 1)] - entry_price) / entry_price
        if not hit_s: s_ret = -(closes[min(i + max_hold, n - 1)] - entry_price) / entry_price

        long_returns[i], short_returns[i] = l_ret, s_ret
        long_hits[i], short_hits[i] = hit_l_type, hit_s_type

    df['tb_long_ret']  = long_returns
    df['tb_short_ret'] = short_returns
    df['tb_long_hit']  = long_hits
    df['tb_short_hit'] = short_hits

    return df.iloc[:-max_hold].copy()

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

def evaluate_master(csv_path, tp_mult=2.0, sl_mult=-1.0, max_hold=36, top_pct=0.2):
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path).replace([np.inf, -np.inf], np.nan).dropna(subset=['close'])
    
    # 1. 동적 배리어 적용
    df = apply_dynamic_triple_barrier(df, tp_mult, sl_mult, max_hold)

    tb_cols = ['tb_long_ret', 'tb_short_ret', 'tb_long_hit', 'tb_short_hit']
    fwd_cols = [f'fwd_ret_{h}' for h in [6, 12, 24, 36]]
    exclude_cols = {'timestamp', 'close', 'log_return'} | set(tb_cols) | set(fwd_cols)

    features = [c for c in df.columns if c not in exclude_cols and pd.api.types.is_numeric_dtype(df[c])]

    df_clean = df.dropna(subset=features + tb_cols).copy()
    X = df_clean[features]
    
    # [수정 완료] 명확한 분류 타겟: 롱 익절(+1), 숏 익절(-1), 그 외(0)
    y_dir = np.zeros(len(df_clean))
    y_dir[(df_clean['tb_long_hit'] == 1)] = 1
    y_dir[(df_clean['tb_short_hit'] == 1)] = -1

    print("\n🤖 [Sklearn ML 엔진 가동] 상호정보량(MI) 및 랜덤 포레스트(RF) 분석 중...")
    mi_scores = mutual_info_classif(X, y_dir, random_state=42)
    mi_series = pd.Series(mi_scores, index=features)

    rf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1, class_weight='balanced')
    rf.fit(X, y_dir)
    rf_importances = pd.Series(rf.feature_importances_, index=features)

    results = []
    for feat in tqdm(features, desc="Evaluating Features", ncols=100):
        df_c = df[df[feat].notna()].copy()
        if len(df_c) < 100: continue

        ic_val, _ = stats.spearmanr(df_c[feat], df_c['tb_long_ret'])
        if np.isnan(ic_val): ic_val = 0.0
        
        # ── (신규 로직) IC Decay 분석 ──
        ic_decays = {}
        for h in [6, 12, 24, 36]:
            ic, _ = stats.spearmanr(df_c[feat], df_c[f'fwd_ret_{h}'])
            ic_decays[h] = 0.0 if np.isnan(ic) else ic

        # ── (신규 로직) 10분위 스프레드 (Decile Spread) 분석 ──
        decile_spread = 0.0
        if _feature_type(feat, df_c) == 'continuous':
            try:
                # 값을 기준으로 10등분, 중복값은 버림
                df_c['decile'] = pd.qcut(df_c[feat], 10, labels=False, duplicates='drop')
                q_max, q_min = df_c['decile'].max(), df_c['decile'].min()
                if q_max > q_min:
                    mean_q_max = df_c[df_c['decile'] == q_max]['tb_long_ret'].mean()
                    mean_q_min = df_c[df_c['decile'] == q_min]['tb_long_ret'].mean()
                    decile_spread = (mean_q_max - mean_q_min) * 100 # 퍼센트로 변환
            except Exception:
                pass

        top_mask, bottom_mask, eval_type = _masks(feat, df_c, top_pct)
        stats_res = _edge_stats(df_c, top_mask, bottom_mask, max_hold)
        if stats_res is None: continue

        # RL Ready 조건에 10분위 스프레드와 IC 평균치 추가
        avg_decay_ic = sum(abs(v) for v in ic_decays.values()) / 4.0
        is_rl_ready = (
            stats_res['expected_edge'] > 0 and
            (stats_res['p_value'] < 0.05 or abs(ic_val) >= 0.015
             or rf_importances.get(feat, 0) >= 0.015 or abs(decile_spread) > 0.1)
        )
        
        results.append({
            'feature': feat, 'type': eval_type,
            'rf_imp': rf_importances.get(feat, 0), 'mi_score': mi_series.get(feat, 0), 
            'ic': ic_val, 'ic_decay_avg': avg_decay_ic, 'decile_spread': decile_spread,
            **stats_res, 'is_rl_ready': is_rl_ready
        })

    df_res = pd.DataFrame(results).sort_values('expected_edge', ascending=False)

    W = 160
    print("\n" + "=" * W)
    print(" 🎯 [표 1] RL 피처 마스터 리포트 (Dynamic ATR TBM + Decile Spread)")
    print("=" * W)
    print(f"{'RL':<3} | {'피처명':<27} | {'RF':>7} | {'MI':>7} | {'IC':>6} | {'Spread':>8} | {'IC_D(avg)':>9} | {'승률':>6} | {'기대수익':>9} | {'P-val':>6}")
    print("-" * W)

    for _, r in df_res.iterrows():
        mark = "✅" if r['is_rl_ready'] else "  "
        print(f" {mark:<3} | {r['feature']:<27} | {r['rf_imp']:>7.3f} | {r['mi_score']:>7.3f} | {r['ic']:>6.3f} | {r['decile_spread']:>7.3f}% | {r['ic_decay_avg']:>9.3f} | {r['win_rate']:5.1f}% | {r['expected_edge']:>+8.3f}% | {r['p_value']:>6.4f}")
    print("=" * W)
    
    return df_res



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input',    type=str,   default='data/rl_training_data_full.csv')
    
    # [수정] 고정 퍼센트(tp, sl) 대신 동적 ATR 배수(tp_mult, sl_mult)로 파라미터 변경
    parser.add_argument('--tp_mult',  type=float, default=2.0)  
    parser.add_argument('--sl_mult',  type=float, default=-1.0) 
    
    parser.add_argument('--max_hold', type=int,   default=36)
    parser.add_argument('--top_pct',  type=float, default=0.2)
    args = parser.parse_args()

    # 함수 호출부 인자명 매칭
    evaluate_master(
        args.input, 
        tp_mult=args.tp_mult, 
        sl_mult=args.sl_mult,
        max_hold=args.max_hold, 
        top_pct=args.top_pct
    )
