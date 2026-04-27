#!/usr/bin/env python3
"""Train TiDE, FITS, DLinear, and Koopa models on specialized targets."""

import os
import sys
import logging
import argparse
import shutil
import numpy as np
import pandas as pd

# Suppress noisy logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--expected-year", type=int, default=2024)
    p.add_argument("--allow-multi-year", action="store_true")
    p.add_argument("--timesnet-target", choices=["vwap", "cycle"], default="vwap")
    p.add_argument("--force-retrain", action="store_true")
    p.add_argument("--startup-check-only", action="store_true")
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

    # 2b. TimesNet VWAP deviation (%), no future shift
    if all(c in work.columns for c in ["high", "low", "close", "volume"]):
        tp = (work["high"] + work["low"] + work["close"]) / 3.0
        v = pd.to_numeric(work["volume"], errors="coerce").fillna(0.0)
        pv_sum = (tp * v).rolling(window=60, min_periods=60).sum()
        v_sum = v.rolling(window=60, min_periods=60).sum().replace(0, np.nan)
        vwap = pv_sum / (v_sum + 1e-12)
        work["y_vwap_dev"] = ((work["close"] / (vwap + 1e-12)) - 1.0) * 100.0
        work["y_vwap_dev"] = pd.to_numeric(work["y_vwap_dev"], errors="coerce").fillna(0.0)
    else:
        work["y_vwap_dev"] = 0.0

    # 3. DLinear: OFI Decay
    ofi = work["smart_money_flow"] if "smart_money_flow" in work.columns else work["volume"]
    work["y_ofi_decay"] = ((ofi.shift(-3) - ofi) / 3.0).fillna(0)

    # 4. Koopa: Regime Transition Acceleration
    trend = work["mtf_trend_1h"] if "mtf_trend_1h" in work.columns else work["close"].pct_change(12).fillna(0)
    work["y_koopa_trans"] = (trend.shift(-5) - trend).fillna(0)

    # 5. PatchTST: short-horizon return target (not absolute price)
    # Keep scale around a few bps to match downstream tanh normalization.
    work["y_patchtst_ret"] = (
        pd.to_numeric(work["close"], errors="coerce")
        .pct_change()
        .shift(-1)
        .clip(-0.05, 0.05)
        .fillna(0.0)
    )

    return work


def train_nf_model(
    df: pd.DataFrame,
    target_col: str,
    model_class,
    model_name: str,
    out_path: str,
    max_steps: int,
    force_retrain: bool = False,
):
    """Train and save a NeuralForecast model."""
    conf_path = os.path.join(out_path, "configuration.pkl")
    if os.path.exists(conf_path):
        if force_retrain:
            logger.info("force_retrain=True -> removing existing %s model dir: %s", model_name, out_path)
            shutil.rmtree(out_path, ignore_errors=True)
        else:
            logger.info(f"{model_name} already trained at {out_path}. Skipping.")
            return

    from neuralforecast import NeuralForecast
    
    logger.info(f"Training {model_name} on target {target_col}...")
    
    # Exogenous columns (router-compatible)
    exog_cols = [
        "session_us", "hour_cos", "cvp_poc_dist", "cvp_volume_imbalance",
        "fvg_dist", "breakout_strength", "oi_change_rate", "ofti", "kel",
        "mta_funding", "svps"
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

    if "timestamp" in df.columns:
        ds = pd.to_datetime(df["timestamp"], errors="coerce")
    else:
        ds = pd.date_range(end=pd.Timestamp.now(), periods=len(df_nf), freq="5min")
    df_nf["ds"] = ds
    df_nf["unique_id"] = "ETH"
    df_nf = df_nf.dropna(subset=["ds"]).sort_values("ds").reset_index(drop=True)
    
    # Only rename if it's not already named 'y' or if we are targeting a different col
    if target_col != "y":
        if "y" in df_nf.columns and target_col != "y":
             df_nf.drop(columns=["y"], inplace=True)
        df_nf.rename(columns={target_col: "y"}, inplace=True)
    
    kwargs = {"max_steps": max_steps, "scaler_type": "standard"}
    if model_name == "TimesNet":
        kwargs.update({"h": 12, "input_size": 256})
    else:
        kwargs.update({"h": 10, "input_size": 30, "hist_exog_list": exog_cols})

    try:
        model = model_class(**kwargs)
    except Exception as e:
        msg = str(e).lower()
        if "does not support historical exogenous variables" not in msg:
            raise
        logger.warning(
            "%s does not support hist exog in this neuralforecast version. "
            "Retrying without historical exogenous vars.",
            model_name,
        )
        kwargs_no_exog = dict(kwargs)
        kwargs_no_exog.pop("hist_exog_list", None)
        kwargs_no_exog.pop("futr_exog_list", None)
        kwargs_no_exog.pop("stat_exog_list", None)
        model = model_class(**kwargs_no_exog)
    
    nf = NeuralForecast(models=[model], freq="5min")
    nf.fit(df=df_nf)
    
    os.makedirs(out_path, exist_ok=True)
    nf.save(path=out_path, overwrite=True)
    logger.info(f"Saved {model_name} to {out_path}")


def main():
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_alternative_models")
        return 0

    if not os.path.exists(args.data_path):
        logger.error(f"Data not found: {args.data_path}")
        return 1

    df = pd.read_csv(args.data_path)
    if args.limit > 0:
        df = df.tail(args.limit).reset_index(drop=True)
        
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce")
        years = sorted(ts.dropna().dt.year.unique().tolist())
        if (not args.allow_multi_year) and years != [int(args.expected_year)]:
            logger.error(
                "Year guard failed. expected=[%d] actual=%s (use --allow-multi-year to bypass)",
                int(args.expected_year),
                years,
            )
            return 1
        logger.info("Training data years=%s", years)

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
        "TiDE", os.path.join(args.out_dir, "nf_tide"), args.max_steps, force_retrain=args.force_retrain
    )
    
    # 2. PatchTST (Price Trend - Master Model)
    train_nf_model(
        df, "y_patchtst_ret",
        lambda **kw: PatchTST(revin=False, **kw), 
        "PatchTST", os.path.join(args.out_dir, "nf_patchtst"), args.max_steps, force_retrain=args.force_retrain
    )
    
    # 3. TimesNet
    timesnet_target = "y_vwap_dev" if args.timesnet_target == "vwap" else "y_cycle"
    train_nf_model(
        df, timesnet_target,
        lambda **kw: TimesNet(hidden_size=64, **kw), 
        "TimesNet", os.path.join(args.out_dir, "nf_timesnet"), args.max_steps, force_retrain=args.force_retrain
    )

    # 4. DLinear (OFI Decay)
    train_nf_model(
        df, "y_ofi_decay", 
        lambda **kw: DLinear(moving_avg_window=7, **kw), 
        "DLinear", os.path.join(args.out_dir, "nf_dlinear"), args.max_steps, force_retrain=args.force_retrain
    )

    logger.info("All alternative models trained successfully. TimesNet target=%s", timesnet_target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
