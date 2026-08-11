#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging

# Setup paths
_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from ensemble.ensemble_router import (
    PatchTSTForecaster,
    TiDEVolatilityForecaster,
    TimesNetCycleForecaster,
    DLinearOFIForecaster
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def main():
    input_path = "data/training_features_5m.csv"
    output_path = "data/training_features_refined_5m.csv"
    
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df)} rows from {input_path}")
    
    # Initialize forecasters
    forecasters = {
        "patchtst": PatchTSTForecaster(),
        "tide": TiDEVolatilityForecaster(),
        "timesnet": TimesNetCycleForecaster(),
        "dlinear": DLinearOFIForecaster()
    }
    
    # Check availability
    for name, f in forecasters.items():
        if not f.available:
            logger.warning(f"Forecaster {name} is NOT available. Check model directories in data/")
    
    # Storage for new features
    new_features = []
    
    # Window size for inference (PatchTST needs 256)
    window_size = 256
    
    # Process rows
    # Note: To avoid look-ahead bias, we only predict using data up to index i
    for i in tqdm(range(window_size, len(df)), desc="Augmenting features"):
        chunk = df.iloc[i-window_size+1 : i+1]
        
        row_features = {}
        for name, f in forecasters.items():
            if f.available:
                try:
                    refined = f.get_refined_features(chunk)
                    row_features.update(refined)
                except Exception as e:
                    # Fill with zeros on failure
                    pass
        
        row_features["timestamp"] = df.iloc[i]["timestamp"]
        new_features.append(row_features)
        
    # Create features dataframe
    feat_df = pd.DataFrame(new_features)
    
    # Merge back to original dataframe
    final_df = df.merge(feat_df, on="timestamp", how="left").fillna(0)
    
    # Save
    final_df.to_csv(output_path, index=False)
    logger.info(f"Saved refined dataset to {output_path}")

if __name__ == "__main__":
    main()
