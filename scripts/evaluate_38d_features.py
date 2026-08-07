import os
import sys
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def evaluate_features(csv_path="data/rl_training_2025_unified.csv", horizon=6):
    if not os.path.exists(csv_path):
        logger.error(f"Dataset not found: {csv_path}")
        return

    logger.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Calculate Forward Return (Target)
    df["close"] = pd.to_numeric(df["close"], errors="coerce").fillna(0.0)
    df[f"forward_return_{horizon}"] = df["close"].shift(-horizon) / df["close"] - 1.0
    
    # Feature Lists based on 38D Unified State
    features_a = [
        "m7_q50", "m7_qwidth", "m7_gmm_cluster", "m7_gmm_conf", "m7_gmm_vol_rank",
        "m7_iso_score", "m7_vae_error", "m7_tp_offset", "m7_sl_offset",
        "m7_entry_long_offset", "m7_entry_short_offset", "mtf_trend_1h", "mtf_trend_4h"
    ]
    features_b = [
        "spread", "rogers_satchell_vol", "amihud_illiquidity_z",
        "smart_money_flow", "taker_acceleration"
    ]
    features_d_ai = [
        "patchtst_median", "patchtst_regime_sim",
        "tide_vol_raw", "tide_vol_zscore",
        "timesnet_cycle_sin", "timesnet_cycle_cos", "timesnet_cycle_delta",
        "dlinear_smf_ema", "dlinear_smf_slope"
    ]
    
    all_features = features_a + features_b + features_d_ai
    
    # Keep only available columns
    available_features = [f for f in all_features if f in df.columns]
    missing = set(all_features) - set(available_features)
    if missing:
        logger.warning(f"Missing features in dataset: {missing}")

    # Prepare Data
    eval_df = df.dropna(subset=[f"forward_return_{horizon}"] + available_features).copy()
    
    # For speed, sample if too large
    if len(eval_df) > 100000:
        logger.info("Sampling 100,000 rows for faster evaluation...")
        eval_df = eval_df.sample(n=100000, random_state=42)
        
    X = eval_df[available_features].astype(np.float32)
    y = eval_df[f"forward_return_{horizon}"].astype(np.float32)
    
    logger.info(f"Training RandomForest Regressor on {len(X)} rows with {len(available_features)} features...")
    
    model = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X, y)
    
    # Extract Feature Importances
    importances = model.feature_importances_
    feat_imp = pd.DataFrame({"Feature": available_features, "Importance": importances})
    feat_imp = feat_imp.sort_values(by="Importance", ascending=False).reset_index(drop=True)
    
    # Group Evaluation
    group_imp = {"M7 & Meta (Block A)": 0.0, "Micro & Liquidity (Block B)": 0.0, "AI Models (Block D)": 0.0}
    
    logger.info("\n" + "="*50)
    logger.info(f"🏆 Feature Importance Ranking (Target: {horizon}-step Forward Return)")
    logger.info("="*50)
    
    for i, row in feat_imp.iterrows():
        feat = row["Feature"]
        imp = row["Importance"] * 100
        
        # Attribute to group
        if feat in features_a:
            group_imp["M7 & Meta (Block A)"] += imp
        elif feat in features_b:
            group_imp["Micro & Liquidity (Block B)"] += imp
        elif feat in features_d_ai:
            group_imp["AI Models (Block D)"] += imp
            
        marker = "🔥" if imp >= 5.0 else ("⚠️" if imp < 1.0 and feat in features_d_ai else "  ")
        logger.info(f"{i+1:02d}. {feat:<25} : {imp:5.2f}% {marker}")

    logger.info("\n" + "="*50)
    logger.info("📊 Group Contribution Summary")
    logger.info("="*50)
    for k, v in group_imp.items():
        logger.info(f"{k:<30} : {v:5.2f}%")
        
    logger.info("="*50)
    logger.info("💡 Tip: AI features with < 1.0% importance are likely acting as NOISE in RL state.")

if __name__ == "__main__":
    evaluate_features()
