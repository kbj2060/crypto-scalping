#!/usr/bin/env python3
"""Train TiDE, FITS, DLinear, and Koopa models on specialized targets."""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

# Suppress noisy logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/training_features_5m.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    return p.parse_args()


def compute_dominant_cycle(closes: np.ndarray, window: int = 60) -> np.ndarray:
    """Compute rolling dominant period using FFT."""
    n = len(closes)
    out = np.full(n, 15.0)  # Default 15m cycle
    for i in range(window, n, 5):
        chunk = closes[i - window : i]
        chunk = chunk - np.mean(chunk)
        if np.all(chunk == 0):
            continue
        fft_vals = np.abs(np.fft.rfft(chunk))
        fft_vals[0] = 0  # Remove DC component
        max_idx = np.argmax(fft_vals)
        if max_idx > 0:
            out[i : i + 5] = window / max_idx
    return out


def build_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Build specific targets for alternative models."""
    logger.info("Building alternative targets...")
    work = df.copy()

    # 1. TiDE: Realized Volatility (10 candles ahead)
    if "log_return" in work.columns:
        ret = work["log_return"]
    else:
        ret = work["close"].pct_change().fillna(0)
    work["y_vol"] = ret.rolling(10).std().shift(-10).fillna(0)

    # 2. FITS: Dominant Cycle (10 candles ahead)
    closes = work["close"].values
    cycle = compute_dominant_cycle(closes, window=60)
    work["y_cycle"] = pd.Series(cycle).shift(-10).fillna(15.0)

    # 3. DLinear: OFI Decay
    ofi = work["smart_money_flow"] if "smart_money_flow" in work.columns else work["volume"]
    work["y_ofi_decay"] = ((ofi.shift(-3) - ofi) / 3.0).fillna(0)

    # 4. Koopa: Regime Transition Acceleration
    trend = work["mtf_trend_1h"] if "mtf_trend_1h" in work.columns else work["close"].pct_change(12).fillna(0)
    work["y_koopa_trans"] = (trend.shift(-5) - trend).fillna(0)

    return work


def train_nf_model(df: pd.DataFrame, target_col: str, model_class, model_name: str, out_path: str, max_steps: int):
    """Train and save a NeuralForecast model."""
    if os.path.exists(os.path.join(out_path, "configuration.pkl")):
        logger.info(f"{model_name} already trained at {out_path}. Skipping.")
        return

    from neuralforecast import NeuralForecast
    
    logger.info(f"Training {model_name} on target {target_col}...")
    
    # Exogenous columns (same as PatchTST for compatibility)
    exog_cols = [
        "session_us", "cvp_poc_dist", "cvp_volume_imbalance", 
        "fvg_dist", "oi_change_rate"
    ]
    # Ensure exog cols exist
    for c in exog_cols:
        if c not in df.columns:
            df[c] = 0.0

    # Ensure unique columns (especially when target_col is 'close')
    cols_to_use = list(dict.fromkeys(["close", target_col] + exog_cols))
    df_nf = df[cols_to_use].copy()
    df_nf.ffill(inplace=True)
    df_nf.fillna(0, inplace=True)
    
    df_nf["ds"] = pd.date_range(end=pd.Timestamp.now(), periods=len(df_nf), freq="5min")
    df_nf["unique_id"] = "ETH"
    
    # Only rename if it's not already named 'y' or if we are targeting a different col
    if target_col != "y":
        if "y" in df_nf.columns and target_col != "y":
             df_nf.drop(columns=["y"], inplace=True)
        df_nf.rename(columns={target_col: "y"}, inplace=True)
    
    kwargs = {
        "h": 10,
        "input_size": 30,
        "max_steps": max_steps,
        "scaler_type": "standard",
    }
    if model_name == "TiDE":
        kwargs["hist_exog_list"] = exog_cols
        
    model = model_class(**kwargs)
    
    nf = NeuralForecast(models=[model], freq="5min")
    nf.fit(df=df_nf)
    
    os.makedirs(out_path, exist_ok=True)
    nf.save(path=out_path, overwrite=True)
    logger.info(f"Saved {model_name} to {out_path}")


def main():
    args = parse_args()
    if not os.path.exists(args.data_path):
        logger.error(f"Data not found: {args.data_path}")
        return 1

    df = pd.read_csv(args.data_path)
    if args.limit > 0:
        df = df.tail(args.limit).reset_index(drop=True)
        
    df = build_targets(df)
    
    try:
        from neuralforecast.models import TiDE, TimesNet, DLinear, PatchTST
    except ImportError as e:
        logger.error(f"Failed to import NeuralForecast models. Please update nixtla/neuralforecast: {e}")
        return 1

    # 1. TiDE (Volatility)
    train_nf_model(
        df, "y_vol", 
        lambda **kw: TiDE(hidden_size=256, **kw), 
        "TiDE", os.path.join(args.out_dir, "nf_tide"), args.max_steps
    )
    
    # 2. PatchTST (Price Trend - Master Model)
    train_nf_model(
        df, "close", 
        lambda **kw: PatchTST(revin=False, **kw), 
        "PatchTST", os.path.join(args.out_dir, "nf_patchtst"), args.max_steps
    )
    
    # 2. TimesNet (replaces FITS for Cycle)
    train_nf_model(
        df, "y_cycle", 
        lambda **kw: TimesNet(hidden_size=64, **kw), 
        "TimesNet", os.path.join(args.out_dir, "nf_timesnet"), args.max_steps
    )
    
    # 3. DLinear (OFI Decay)
    train_nf_model(
        df, "y_ofi_decay", 
        lambda **kw: DLinear(moving_avg_window=7, **kw), 
        "DLinear", os.path.join(args.out_dir, "nf_dlinear"), args.max_steps
    )

    logger.info("All alternative models trained successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
