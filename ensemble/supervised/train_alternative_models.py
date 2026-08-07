#!/usr/bin/env python3
"""Train TiDE, FITS, DLinear, and Koopa models on specialized targets."""

import os
import sys
import logging
import argparse
import shutil
import json
import glob
from pathlib import Path
import numpy as np
import pandas as pd

# Suppress noisy logs
logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("neuralforecast").setLevel(logging.ERROR)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DLINEAR_FLOW_COLS = [
    "smart_money_flow", "ofi_acceleration", "cvp_volume_imbalance",
    "whale_retail_ratio", "net_taker_ratio", "taker_acceleration",
    "volume", "quote_volume", "taker_buy_base", "taker_buy_quote",
]


def _safe_num(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _rolling_vwap_dev(df: pd.DataFrame, window: int = 60) -> pd.Series:
    if not all(c in df.columns for c in ["high", "low", "close", "volume"]):
        return pd.Series(0.0, index=df.index, dtype="float64")
    high, low, close = _safe_num(df, "high"), _safe_num(df, "low"), _safe_num(df, "close")
    volume = _safe_num(df, "volume").clip(lower=0.0)
    tp = (high + low + close) / 3.0
    pv_sum = (tp * volume).rolling(window=window, min_periods=window).sum()
    v_sum = volume.rolling(window=window, min_periods=window).sum().replace(0, np.nan)
    vwap = pv_sum / (v_sum + 1e-12)
    return ((close / (vwap + 1e-12)) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _future_extremes(df: pd.DataFrame, horizon: int) -> tuple[pd.Series, pd.Series]:
    close = _safe_num(df, "close").clip(lower=1e-12)
    high = _safe_num(df, "high", close) if "high" in df.columns else close
    low = _safe_num(df, "low", close) if "low" in df.columns else close
    fut_high = high.shift(-1).rolling(horizon, min_periods=1).max().shift(-(horizon - 1))
    fut_low = low.shift(-1).rolling(horizon, min_periods=1).min().shift(-(horizon - 1))
    mfe = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mae = (fut_low / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return mfe, mae


def _triple_barrier_edge(df: pd.DataFrame, horizon: int = 6) -> pd.Series:
    close = _safe_num(df, "close").clip(lower=1e-12)
    ret = close.pct_change().fillna(0.0)
    rv = ret.rolling(36, min_periods=6).std().fillna(ret.abs().rolling(36, min_periods=1).mean()).fillna(0.001)
    tp = np.maximum(0.0015, 1.20 * rv)
    sl = np.maximum(0.0010, 0.90 * rv)
    mfe, mae = _future_extremes(df, horizon)
    up = mfe >= tp
    down = mae <= -sl
    edge = np.where(up & ~down, 1.0, np.where(down & ~up, -1.0, 0.0))
    both = up & down
    edge = np.where(both, np.where(mfe.abs() >= mae.abs(), 0.5, -0.5), edge)
    return pd.Series(edge, index=df.index, dtype="float64").fillna(0.0)


def _flow_pressure(df: pd.DataFrame) -> pd.Series:
    smf = _safe_num(df, "smart_money_flow")
    ofi = _safe_num(df, "ofi_acceleration")
    ntr = _safe_num(df, "net_taker_ratio")
    taker = _safe_num(df, "taker_acceleration")
    cvp = _safe_num(df, "cvp_volume_imbalance")
    return np.tanh(1.2 * smf + 0.8 * ofi + 0.6 * ntr + 0.6 * taker + 0.4 * cvp)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-path", default="data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--out-dir", default="data")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--patchtst-steps", type=int, default=0, help="PatchTST max steps. 0 uses --max-steps.")
    p.add_argument("--tide-steps", type=int, default=0, help="TiDE max steps. 0 uses --max-steps.")
    p.add_argument("--timesnet-steps", type=int, default=0, help="TimesNet max steps. 0 uses --max-steps.")
    p.add_argument("--dlinear-steps", type=int, default=0, help="DLinear max steps. 0 uses --max-steps.")
    p.add_argument(
        "--timesnet-accelerator",
        choices=["auto", "cpu", "gpu"],
        default="auto",
        help="Lightning accelerator for TimesNet training. Use cpu if CUDA/NVRTC issues appear.",
    )
    p.add_argument("--timesnet-devices", type=int, default=1)
    p.add_argument(
        "--no-nvrtc-path-patch",
        action="store_true",
        help="Disable automatic LD_LIBRARY_PATH patching for CUDA/NVRTC libraries.",
    )
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

    # 1. TiDE: adverse excursion risk and reward/risk opportunity.
    if "log_return" in work.columns:
        ret = pd.to_numeric(work["log_return"], errors="coerce").fillna(0)
    else:
        ret = _safe_num(work, "close").pct_change().fillna(0)
    work["y_vol"] = ret.rolling(10).std().shift(-10).fillna(0)
    mfe6, mae6 = _future_extremes(work, horizon=6)
    adverse = mae6.abs().clip(0.0, 0.05)
    reward = mfe6.clip(0.0, 0.05)
    work["y_adverse_risk"] = adverse.fillna(0.0)
    work["y_reward_risk"] = (reward / (adverse + 5e-4)).clip(0.0, 8.0).fillna(0.0)

    # 2. FITS: Dominant Cycle (10 candles ahead)
    closes = work["close"].values
    cycle = compute_dominant_cycle(closes, window=60)
    work["y_cycle"] = pd.Series(cycle).shift(-10).fillna(15.0)

    # 2b. TimesNet: forecast the VWAP anchor gap; router derives reversion/escape.
    work["y_vwap_dev"] = (_rolling_vwap_dev(work, window=60) * 100.0).fillna(0.0)
    future_vwap = work["y_vwap_dev"].shift(-12).fillna(0.0)
    cur_vwap = work["y_vwap_dev"].fillna(0.0)
    work["y_anchor_revert"] = (cur_vwap.abs() - future_vwap.abs()).clip(-5.0, 5.0).fillna(0.0)

    # 3. DLinear: forecast future flow pressure; router derives exhaustion/flip.
    ofi = work["smart_money_flow"] if "smart_money_flow" in work.columns else work.get("volume", 0.0)
    work["y_ofi_decay"] = ((ofi.shift(-3) - ofi) / 3.0).fillna(0)
    flow = pd.Series(_flow_pressure(work), index=work.index, dtype="float64")
    work["y_flow_pressure"] = flow.fillna(0.0)
    work["y_flow_exhaustion"] = (flow.abs() - flow.shift(-3).rolling(3, min_periods=1).mean().abs()).fillna(0.0)

    # 4. Koopa: Regime Transition Acceleration
    trend = work["mtf_trend_1h"] if "mtf_trend_1h" in work.columns else work["close"].pct_change(12).fillna(0)
    work["y_koopa_trans"] = (trend.shift(-5) - trend).fillna(0)

    # 5. PatchTST: triple-barrier edge, with legacy short-horizon return retained.
    work["y_patchtst_ret"] = (
        pd.to_numeric(work["close"], errors="coerce")
        .pct_change()
        .shift(-1)
        .clip(-0.05, 0.05)
        .fillna(0.0)
    )
    work["y_dir_edge"] = _triple_barrier_edge(work, horizon=6)

    return work


def train_nf_model(
    df: pd.DataFrame,
    target_col: str,
    model_class,
    model_name: str,
    out_path: str,
    max_steps: int,
    force_retrain: bool = False,
    timesnet_accelerator: str = "auto",
    timesnet_devices: int = 1,
    provenance: dict | None = None,
):
    """Train and save a NeuralForecast model."""
    conf_path = os.path.join(out_path, "configuration.pkl")
    contract_path = os.path.join(out_path, "specialist_contract.json")
    if os.path.exists(conf_path):
        if force_retrain:
            logger.info("force_retrain=True -> removing existing %s model dir: %s", model_name, out_path)
            shutil.rmtree(out_path, ignore_errors=True)
        else:
            if model_name == "DLinear":
                skipped_inputs = [c for c in DLINEAR_FLOW_COLS if c in df.columns]
            else:
                skipped_inputs = [
                    "session_us", "hour_cos", "cvp_poc_dist", "cvp_volume_imbalance",
                    "fvg_dist", "breakout_strength", "oi_change_rate", "ofti", "kel",
                    "mta_funding", "svps",
                ]
            _write_specialist_contract(
                contract_path,
                _specialist_contract(
                    model_name,
                    target_col,
                    skipped_inputs,
                    skipped=True,
                    provenance=provenance,
                    provenance_certified=False,
                ),
            )
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

    if model_name == "DLinear":
        model_exog_cols = [c for c in DLINEAR_FLOW_COLS if c in df.columns]
        cols_to_use = list(dict.fromkeys([target_col] + model_exog_cols))
        forbidden = {"open", "high", "low", "close"}.intersection(cols_to_use)
        if forbidden:
            raise ValueError(f"DLinear specialist input contains forbidden price columns: {sorted(forbidden)}")
    else:
        model_exog_cols = exog_cols
        # Ensure unique columns (especially when target_col is 'close')
        cols_to_use = list(dict.fromkeys(["close", target_col] + model_exog_cols))
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
        kwargs.update({
            "h": 12,
            "input_size": 256,
            "accelerator": timesnet_accelerator,
            "devices": max(1, int(timesnet_devices)),
        })
    elif model_name == "DLinear":
        kwargs.update({"h": 3, "input_size": 30})
    else:
        kwargs.update({"h": 10, "input_size": 30, "hist_exog_list": model_exog_cols})

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
    _write_specialist_contract(
        contract_path,
        _specialist_contract(
            model_name,
            target_col,
            model_exog_cols,
            h=int(kwargs.get("h", 10)),
            input_size=int(kwargs.get("input_size", 30)),
            max_steps=int(max_steps),
            trainer_kwargs={
                k: kwargs[k]
                for k in ["accelerator", "devices"]
                if k in kwargs
            },
            skipped=False,
            provenance=provenance,
            provenance_certified=True,
        ),
    )
    logger.info(f"Saved {model_name} to {out_path}")


def _specialist_contract(
    model_name: str,
    target_col: str,
    input_cols: list[str],
    h: int | None = None,
    input_size: int | None = None,
    max_steps: int | None = None,
    trainer_kwargs: dict | None = None,
    skipped: bool = False,
    provenance: dict | None = None,
    provenance_certified: bool = False,
) -> dict:
    contracts = {
        "PatchTST": {
            "role": "direction",
            "target_description": "6-bar dynamic triple-barrier direction edge (-1, 0, +1)",
            "runtime_y": "past 6-bar barrier-edge proxy derived from price extremes",
            "no_price_input_required": False,
        },
        "TiDE": {
            "role": "adverse_excursion_risk",
            "target_description": "future 6-bar adverse excursion magnitude; reward/risk is derived downstream",
            "runtime_y": "past 6-bar downside excursion risk proxy",
            "no_price_input_required": False,
        },
        "DLinear": {
            "role": "flow_persistence",
            "target_description": "future 3-bar smart-money/OFI pressure forecast",
            "runtime_y": "past smart-money/OFI pressure",
            "no_price_input_required": True,
        },
        "TimesNet": {
            "role": "vwap_anchor_reversion",
            "target_description": "60-bar VWAP deviation forecast; reversion/escape is derived downstream",
            "runtime_y": "past 60-bar rolling VWAP deviation",
            "no_price_input_required": False,
        },
    }
    contract = contracts.get(model_name, {}).copy()
    contract.update(
        {
            "model_name": model_name,
            "target_col": target_col,
            "input_cols": input_cols,
            "h": h,
            "input_size": input_size,
            "max_steps": max_steps,
            "trainer_kwargs": trainer_kwargs or {},
            "skipped_existing_model": skipped,
            "artifact_training_provenance_certified": bool(provenance_certified),
            "provenance": provenance or {},
        }
    )
    if model_name == "DLinear":
        contract["forbidden_input_cols"] = ["open", "high", "low", "close"]
    return contract


def _write_specialist_contract(path: str, contract: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(contract, f, ensure_ascii=False, indent=2, sort_keys=True)


def _steps(model_steps: int, fallback_steps: int) -> int:
    return int(model_steps) if int(model_steps) > 0 else int(fallback_steps)


def _patch_nvrtc_library_path() -> None:
    if os.environ.get("QUANT_AI_NVRTC_PATH_PATCHED") == "1":
        return

    candidates: list[str] = []
    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if conda_prefix:
        patterns = [
            os.path.join(conda_prefix, "lib", "python*", "site-packages", "nvidia", "cu13", "lib"),
            os.path.join(conda_prefix, "lib", "python*", "site-packages", "nvidia", "cuda_nvrtc", "lib"),
        ]
        for pattern in patterns:
            candidates.extend(glob.glob(pattern))

    ollama_cuda_path = "/usr/local/lib/ollama/mlx_cuda_v13"
    if os.path.exists(ollama_cuda_path):
        candidates.append(ollama_cuda_path)

    usable = []
    for path in candidates:
        if os.path.exists(os.path.join(path, "libnvrtc-builtins.so.13.0")) or glob.glob(os.path.join(path, "libnvrtc-builtins.so*")):
            usable.append(path)

    if not usable:
        logger.warning("NVRTC path patch skipped: libnvrtc-builtins.so not found in expected locations.")
        return

    current = [p for p in os.environ.get("LD_LIBRARY_PATH", "").split(":") if p]
    patched = []
    for path in usable + current:
        if path not in patched:
            patched.append(path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(patched)
    os.environ["QUANT_AI_NVRTC_PATH_PATCHED"] = "1"
    logger.info("NVRTC LD_LIBRARY_PATH patched with: %s", usable)

    # glibc/NVRTC may snapshot library paths at process start, so restart once
    # before torch/neuralforecast imports happen.
    os.execvpe(sys.executable, [sys.executable] + sys.argv, os.environ)


def main():
    args = parse_args()
    if args.startup_check_only:
        logger.info("startup check ok: train_alternative_models")
        return 0

    if not bool(args.no_nvrtc_path_patch):
        _patch_nvrtc_library_path()

    if not os.path.exists(args.data_path):
        logger.error(f"Data not found: {args.data_path}")
        return 1

    df = pd.read_csv(args.data_path)
    if args.limit > 0:
        df = df.tail(args.limit).reset_index(drop=True)

    data_provenance = {
        "data_path": str(Path(args.data_path).expanduser().resolve()),
        "expected_year": int(args.expected_year),
        "allow_multi_year": bool(args.allow_multi_year),
        "limit": int(args.limit),
        "rows_after_limit": int(len(df)),
    }
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
        data_provenance.update(
            {
                "actual_years": [int(y) for y in years],
                "timestamp_start": str(ts.min()) if ts.notna().any() else None,
                "timestamp_end": str(ts.max()) if ts.notna().any() else None,
                "timestamp_nulls": int(ts.isna().sum()),
            }
        )
    else:
        data_provenance.update(
            {
                "actual_years": [],
                "timestamp_start": None,
                "timestamp_end": None,
                "timestamp_nulls": None,
            }
        )

    df = build_targets(df)
    steps = {
        "TiDE": _steps(args.tide_steps, args.max_steps),
        "PatchTST": _steps(args.patchtst_steps, args.max_steps),
        "TimesNet": _steps(args.timesnet_steps, args.max_steps),
        "DLinear": _steps(args.dlinear_steps, args.max_steps),
    }
    logger.info(
        "Model max_steps=%s | TimesNet accelerator=%s devices=%d",
        steps,
        args.timesnet_accelerator,
        int(args.timesnet_devices),
    )

    try:
        from neuralforecast.models import TiDE, TimesNet, DLinear, PatchTST
    except ImportError as e:
        logger.error(f"Failed to import NeuralForecast models. Please update nixtla/neuralforecast: {e}")
        return 1

    # 1. TimesNet first: longest training path and most likely to expose CUDA/NVRTC issues.
    timesnet_target = "y_vwap_dev" if args.timesnet_target == "vwap" else "y_anchor_revert"
    train_nf_model(
        df, timesnet_target,
        lambda **kw: TimesNet(hidden_size=64, **kw),
        "TimesNet",
        os.path.join(args.out_dir, "nf_timesnet"),
        steps["TimesNet"],
        force_retrain=args.force_retrain,
        timesnet_accelerator=args.timesnet_accelerator,
        timesnet_devices=args.timesnet_devices,
        provenance=data_provenance,
    )

    # 2. PatchTST (Triple-barrier direction edge)
    train_nf_model(
        df, "y_dir_edge",
        lambda **kw: PatchTST(revin=False, **kw),
        "PatchTST", os.path.join(args.out_dir, "nf_patchtst"), steps["PatchTST"], force_retrain=args.force_retrain, provenance=data_provenance
    )

    # 3. TiDE (Adverse excursion risk)
    train_nf_model(
        df, "y_adverse_risk",
        lambda **kw: TiDE(hidden_size=256, **kw),
        "TiDE", os.path.join(args.out_dir, "nf_tide"), steps["TiDE"], force_retrain=args.force_retrain, provenance=data_provenance
    )

    # 4. DLinear (Flow pressure)
    train_nf_model(
        df, "y_flow_pressure",
        lambda **kw: DLinear(moving_avg_window=7, **kw),
        "DLinear", os.path.join(args.out_dir, "nf_dlinear"), steps["DLinear"], force_retrain=args.force_retrain, provenance=data_provenance
    )

    logger.info("All alternative models trained successfully. TimesNet target=%s", timesnet_target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
