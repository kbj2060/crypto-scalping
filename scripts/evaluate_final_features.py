import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_advanced_features(csv_path="data/rl_training_2025_unified.csv", horizon=6):
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found: {csv_path}")
        return

    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df[f"forward_return_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1.0
    
    # 35D Features (TimesNet Removed)
    features_a = [
        "m7_q50", "m7_qwidth", "m7_gmm_cluster", "m7_gmm_conf", "m7_gmm_vol_rank",
        "m7_iso_score", "m7_vae_error", "m7_tp_offset", "m7_sl_offset",
        "m7_entry_long_offset", "m7_entry_short_offset", "mtf_trend_1h", "mtf_trend_4h"
    ]
    features_b = [
        "rogers_satchell_vol", "amihud_illiquidity_z",
        "smart_money_flow", "taker_acceleration"
        # Removed 'spread' as it was missing last time
    ]
    features_d_ai = [
        "patchtst_median", "patchtst_regime_sim",
        "tide_vol_raw", "tide_vol_zscore",
        "dlinear_smf_ema", "dlinear_smf_slope"
    ]
    
    all_features = features_a + features_b + features_d_ai
    available_features = [f for f in all_features if f in df.columns]
    
    eval_df = df.dropna(subset=[f"forward_return_{horizon}"] + available_features).copy()
    if len(eval_df) > 50000:
        logger.info("Sampling 50,000 rows for fast VIF & Correlation Matrix computation...")
        eval_df = eval_df.sample(n=50000, random_state=42)
        
    X = eval_df[available_features].astype(np.float32)
    y = eval_df[f"forward_return_{horizon}"].astype(np.float32)
    
    logger.info("\n" + "="*60)
    logger.info("🔍 1. Feature Importance (RandomForest)")
    logger.info("="*60)
    
    model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": available_features, "Importance": importances * 100})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    for i, row in feat_imp.iterrows():
        marker = "🔥" if row["Importance"] >= 5.0 else ("⚠️" if row["Importance"] < 1.0 else "  ")
        logger.info(f"{i+1:02d}. {row['Feature']:<25} : {row['Importance']:5.2f}% {marker}")

    logger.info("\n" + "="*60)
    logger.info("🔗 2. High Correlation Pairs (Spearman > 0.85)")
    logger.info("="*60)
    
    corr_matrix, _ = spearmanr(X)
    high_corr_found = False
    for i in range(len(available_features)):
        for j in range(i+1, len(available_features)):
            if abs(corr_matrix[i, j]) > 0.85:
                logger.info(f"🚨 {available_features[i]} ↔ {available_features[j]}: {corr_matrix[i, j]:.3f}")
                high_corr_found = True
    if not high_corr_found:
        logger.info("✅ No highly correlated pairs found (All < 0.85).")

    logger.info("\n" + "="*60)
    logger.info("🌪️ 3. Multicollinearity Analysis (VIF > 10)")
    logger.info("="*60)
    
    # Calculate VIF manually using inverse of correlation matrix
    corr_mat = np.corrcoef(X.values.T)
    try:
        inv_corr = np.linalg.inv(corr_mat)
        vif_values = np.diag(inv_corr)
        
        vif_data = pd.DataFrame({"Feature": available_features, "VIF": vif_values})
        vif_data = vif_data.sort_values(by="VIF", ascending=False)
        
        high_vif_found = False
        for _, row in vif_data.iterrows():
            if row["VIF"] > 10.0:
                logger.info(f"💥 {row['Feature']:<25} : {row['VIF']:8.2f} (Danger!)")
                high_vif_found = True
        if not high_vif_found:
            logger.info("✅ No severe multicollinearity found (All VIF < 10).")
    except np.linalg.LinAlgError:
        logger.warning("Could not compute VIF due to singular matrix (Perfect collinearity exists!).")
        
    logger.info("="*60)

if __name__ == "__main__":
    evaluate_advanced_features()
