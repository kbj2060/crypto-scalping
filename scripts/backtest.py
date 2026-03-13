import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report

warnings.filterwarnings('ignore')

# ==========================================
# 1. 스윙 퀀트 알파 산출 엔진
# ==========================================
def calculate_swing_alphas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    eps = 1e-9 
    
    div_positive = np.clip(df['funding_price_divergence'], a_min=0, a_max=None)
    df['alpha_WDLS'] = (df['whale_conviction'] * div_positive) / (df['garman_klass_vol'] + eps) * np.exp(-df['chop_index'])
    
    hurst_inv = np.clip(0.5 - df['hurst_48'], a_min=0, a_max=None)
    df['alpha_VPGO'] = -1.0 * (df['cvp_poc_dist'] / (df['cvp_vah_val_width'] + eps)) * hurst_inv
    
    macd_sign = np.sign(df['macd_hist'])
    df['alpha_SMSB'] = np.log1p(np.abs(df['smart_money_flow'])) * (df['volatility_z'] / (df['bb_width_z'] + eps)) * macd_sign * df['regime_trending']
    
    delta_taker = df['sum_taker_long_short_vol_ratio'].diff()
    illiquidity_positive = np.clip(df['amihud_illiquidity_z'], a_min=0, a_max=None)
    df['alpha_CRD'] = (delta_taker / (df['funding_z_score'] - eps)) * illiquidity_positive
    
    hurst_pos = np.clip(df['hurst_change'], a_min=0, a_max=None)
    df['alpha_DRMS'] = hurst_pos * (df['dual_momentum'] / (df['parkinson_vol'] + eps)) * (1.0 - df['chop_index'])
    
    alpha_cols = ['alpha_WDLS', 'alpha_VPGO', 'alpha_SMSB', 'alpha_CRD', 'alpha_DRMS']
    df[alpha_cols] = df[alpha_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    return df

# ==========================================
# 2. 동적 변동성 청산 백테스트 엔진
# ==========================================
def run_dynamic_volatility_backtest(df: pd.DataFrame, alpha_col: str, 
                                    long_threshold: float, short_threshold: float,
                                    tp_multi: float = 2.0, sl_multi: float = 1.0, 
                                    fee_rate: float = 0.0004) -> dict:
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    vols = df['garman_klass_vol'].values 
    alphas = df[alpha_col].values
    
    n = len(closes)
    positions = np.zeros(n)
    simple_returns = np.zeros(n) 
    trade_triggered = np.zeros(n, dtype=bool)
    
    current_pos = 0  
    entry_price = 0.0
    tp_price = 0.0
    sl_price = 0.0
    
    for i in range(1, n):
        prev_alpha = alphas[i-1]
        prev_vol = max(vols[i-1], 0.005) 
        
        # Exit
        if current_pos == 1:
            if lows[i] <= sl_price:
                simple_returns[i] = (sl_price / closes[i-1]) - 1.0 - fee_rate
                current_pos = 0
                trade_triggered[i] = True
            elif highs[i] >= tp_price:
                simple_returns[i] = (tp_price / closes[i-1]) - 1.0 - fee_rate
                current_pos = 0
                trade_triggered[i] = True
            else:
                simple_returns[i] = (closes[i] / closes[i-1]) - 1.0
                positions[i] = 1
                
        elif current_pos == -1:
            if highs[i] >= sl_price:
                simple_returns[i] = 1.0 - (sl_price / closes[i-1]) - fee_rate
                current_pos = 0
                trade_triggered[i] = True
            elif lows[i] <= tp_price:
                simple_returns[i] = 1.0 - (tp_price / closes[i-1]) - fee_rate
                current_pos = 0
                trade_triggered[i] = True
            else:
                simple_returns[i] = 1.0 - (closes[i] / closes[i-1])
                positions[i] = -1

        # Entry
        if current_pos == 0 and not trade_triggered[i]: 
            if prev_alpha > long_threshold:
                current_pos = 1
                entry_price = closes[i] 
                simple_returns[i] = -fee_rate
                tp_price = entry_price * (1 + (prev_vol * tp_multi))
                sl_price = entry_price * (1 - (prev_vol * sl_multi))
                positions[i] = 1
                trade_triggered[i] = True
                
            elif prev_alpha < short_threshold:
                current_pos = -1
                entry_price = closes[i]
                simple_returns[i] = -fee_rate
                tp_price = entry_price * (1 - (prev_vol * tp_multi))
                sl_price = entry_price * (1 + (prev_vol * sl_multi))
                positions[i] = -1
                trade_triggered[i] = True
                
    cumulative_return_series = np.cumprod(1 + simple_returns)
    cumulative_max = np.maximum.accumulate(cumulative_return_series)
    drawdowns = cumulative_return_series / cumulative_max - 1.0
    
    total_return = cumulative_return_series[-1] - 1.0
    mdd = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    annualization_factor = np.sqrt(8760) 
    mean_ret = np.mean(simple_returns)
    std_ret = np.std(simple_returns) + 1e-9
    sharpe_ratio = (mean_ret / std_ret) * annualization_factor
    
    realized_trades = simple_returns[trade_triggered & (positions == 0)]
    win_rate = len(realized_trades[realized_trades > 0]) / (len(realized_trades) + 1e-9) if len(realized_trades) > 0 else 0
    total_trades = len(realized_trades)
    
    return {
        "Strategy": alpha_col,
        "Total Return (%)": round(total_return * 100, 2),
        "MDD (%)": round(mdd * 100, 2),
        "Sharpe Ratio": round(sharpe_ratio, 2),
        "Win Rate (%)": round(win_rate * 100, 2),
        "Total Trades": total_trades
    }

# ==========================================
# 3. 머신러닝 메타 레이블링용 정답지 추출기
# ==========================================
def create_meta_labeling_dataset(df: pd.DataFrame, alpha_col: str, 
                                 long_threshold: float, short_threshold: float,
                                 tp_multi: float = 3.0, sl_multi: float = 1.0) -> pd.DataFrame:
    df_meta = df.copy()
    closes = df_meta['close'].values
    highs = df_meta['high'].values
    lows = df_meta['low'].values
    vols = df_meta['garman_klass_vol'].values 
    alphas = df_meta[alpha_col].values
    
    n = len(closes)
    df_meta['meta_target'] = np.nan 
    df_meta['signal_side'] = 0 
    
    for i in range(1, n - 120): 
        prev_alpha = alphas[i-1]
        prev_vol = max(vols[i-1], 0.005) 
        
        if prev_alpha > long_threshold:
            entry_price = closes[i]
            tp_price = entry_price * (1 + (prev_vol * tp_multi))
            sl_price = entry_price * (1 - (prev_vol * sl_multi))
            
            for j in range(i+1, min(i+120, n)): 
                if lows[j] <= sl_price:
                    df_meta.at[i, 'meta_target'] = 0 
                    df_meta.at[i, 'signal_side'] = 1
                    break
                elif highs[j] >= tp_price:
                    df_meta.at[i, 'meta_target'] = 1 
                    df_meta.at[i, 'signal_side'] = 1
                    break
                    
        elif prev_alpha < short_threshold:
            entry_price = closes[i]
            tp_price = entry_price * (1 - (prev_vol * tp_multi))
            sl_price = entry_price * (1 + (prev_vol * sl_multi))
            
            for j in range(i+1, min(i+120, n)): 
                if highs[j] >= sl_price:
                    df_meta.at[i, 'meta_target'] = 0 
                    df_meta.at[i, 'signal_side'] = -1
                    break
                elif lows[j] <= tp_price:
                    df_meta.at[i, 'meta_target'] = 1 
                    df_meta.at[i, 'signal_side'] = -1
                    break

    ml_dataset = df_meta.dropna(subset=['meta_target']).copy()
    cols_to_drop = ['alpha_WDLS', 'alpha_VPGO', 'alpha_SMSB', 'alpha_CRD', 'alpha_DRMS']
    ml_dataset = ml_dataset.drop(columns=[c for c in cols_to_drop if c != alpha_col], errors='ignore')
    
    return ml_dataset



# ==========================================
# 4. 역발상 워크포워드 (Broad Entry + Sniper AI) 파이프라인
# ==========================================
if __name__ == "__main__":
    print("🚀 [Step 1] 패러다임 전환: Broad Entry + Sniper AI OOS 파이프라인 가동 중...")
    
    # 1. 데이터 로드
    df = pd.read_csv('data/training_features_5m.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    df_with_alphas = calculate_swing_alphas(df)
    target_alpha = 'alpha_VPGO'
    
    # [핵심 변경] 임계값을 1% 극단값에서 10% (상위 90%, 하위 10%)로 대폭 넓혀 데이터 확보
    active_signals = df_with_alphas[df_with_alphas[target_alpha] != 0][target_alpha]
    auto_long_broad = active_signals.quantile(0.90) 
    auto_short_broad = active_signals.quantile(0.10)
    
    print(f"\n🚀 [Step 2] 1차 Broad 백테스트 (Strategy: {target_alpha}, 10% 그물망)")
    base_result = run_dynamic_volatility_backtest(
        df=df_with_alphas, alpha_col=target_alpha,
        long_threshold=auto_long_broad, short_threshold=auto_short_broad,
        tp_multi=3.0, sl_multi=1.0, fee_rate=0.0004
    )
    print(pd.DataFrame([base_result]).to_string(index=False))
    
    print(f"\n🚀 [Step 3] 풍부해진 학습 데이터 생성 및 Walk-Forward 교차 검증 중...")
    ml_data = create_meta_labeling_dataset(
        df=df_with_alphas, alpha_col=target_alpha,
        long_threshold=auto_long_broad, short_threshold=auto_short_broad,
        tp_multi=3.0, sl_multi=1.0
    )
    
    print(f"📊 그물망으로 확보된 ML 학습용 궤적 데이터: {len(ml_data)}건")
    
    features = [
        'volatility_z', 'macd_hist', 'bb_width_z', 'garman_klass_vol',
        'funding_z_score', 'squeeze_power', 'smart_money_flow', 'signal_side'
    ]
    available_features = [f for f in features if f in ml_data.columns]
    
    X = ml_data[available_features].reset_index(drop=True)
    y = ml_data['meta_target'].reset_index(drop=True)
    
    oos_pred_proba = np.zeros(len(X))
    tscv = TimeSeriesSplit(n_splits=5)
    
    # [핵심 변경] 데이터가 많아졌으므로 모델 규제를 강화하여 스나이퍼 모드로 변경 (max_depth 제한, min_samples_leaf 증가)
    model = RandomForestClassifier(n_estimators=300, max_depth=5, min_samples_leaf=15, 
                                   random_state=42, class_weight='balanced_subsample')
    
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train = y.iloc[train_index]
        
        model.fit(X_train, y_train)
        oos_pred_proba[test_index] = model.predict_proba(X_test)[:, 1]
    
    # 첫 번째 학습 구간 처리
    first_test_start = tscv.split(X).__next__()[1][0]
    oos_pred_proba[:first_test_start] = 0.5 
    
    # [핵심 변경] AI 필터 기준을 매우 엄격하게 상향 (성공 확률 55% 이상일 때만 저격)
    custom_threshold = 0.55 
    ml_data['ai_approved'] = (oos_pred_proba >= custom_threshold).astype(int)
    
    print("\n🚀 [Step 4] 최종 OOS 앙상블: 미래 데이터만을 필터링한 '진짜' 스나이퍼 백테스트")
    df_ensemble = df_with_alphas.copy()
    
    rejected_indices = ml_data[ml_data['ai_approved'] == 0].index
    df_ensemble.loc[rejected_indices, target_alpha] = 0 
    
    ensemble_result = run_dynamic_volatility_backtest(
        df=df_ensemble, alpha_col=target_alpha,
        long_threshold=auto_long_broad, short_threshold=auto_short_broad,
        tp_multi=3.0, sl_multi=1.0, fee_rate=0.0004
    )
    
    ensemble_result['Strategy'] = target_alpha + " + OOS Sniper AI"
    
    final_comparison = pd.DataFrame([base_result, ensemble_result])
    print(final_comparison[['Strategy', 'Total Return (%)', 'MDD (%)', 'Sharpe Ratio', 'Win Rate (%)', 'Total Trades']].to_string(index=False))
    
    # 피처 중요도 출력 (마지막 폴드 모델 기준)
    model.fit(X, y) # 전체 데이터로 마지막 학습하여 중요도 추출
    feature_importances = pd.Series(model.feature_importances_, index=available_features).sort_values(ascending=False)
    print("\n🧠 [AI 피처 중요도] AI는 진입 전 무엇을 가장 중요하게 보았는가?")
    for feat, imp in feature_importances.items():
        print(f" - {feat}: {imp*100:.2f}%")