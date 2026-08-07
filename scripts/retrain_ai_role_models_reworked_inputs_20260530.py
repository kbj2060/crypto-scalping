#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_router import (  # noqa: E402
    DLinearOFIForecaster,
    PatchTSTForecaster,
    TiDEVolatilityForecaster,
    TimesNetCycleForecaster,
    _flow_pressure,
    _past_adverse_risk,
    _past_barrier_proxy,
    _rolling_vwap_dev,
)
from ensemble.supervised.train_alternative_models import build_targets  # noqa: E402
from scripts.build_ai_patchmix_direction_core_20260530 import _core_features  # noqa: E402
from scripts.run_ai_role_specific_experiments_20260530 import (  # noqa: E402
    _direction_metrics,
    _risk_metrics,
    _timesnet_metrics,
    _trend_flow_metrics,
)


MODEL_ID = "ai_role_models_reworked_inputs_20260530"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

OLD_PATCH_EXOG = [
    "session_us",
    "hour_cos",
    "cvp_poc_dist",
    "cvp_volume_imbalance",
    "fvg_dist",
    "breakout_strength",
    "oi_change_rate",
    "ofti",
    "kel",
    "mta_funding",
    "svps",
]

AUDITED_DIRECTION = [
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "atr14_pct",
    "realized_vol_24",
    "compression_ratio",
    "funding_roc_288",
    "long_squeeze_risk",
    "crowding_pressure",
    "crowded_short_squeeze_risk",
    "crowded_long_unwind_risk",
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "vwap_dist_96",
    "cvd_288",
    "eth_btc_ret_spread_12",
    "btc_lead_eth_follow_gap_3",
    "btc_volume_impulse_z",
    "range_contraction_breakout_dir",
    "price_cvd_divergence",
    "hour_sin",
    "hour_cos",
    "session_us",
    "cvp_regime",
    "regime_trending",
    "regime_persistence",
]

RISK_EXOG = [
    "ret_1",
    "ret_3",
    "atr14_pct",
    "realized_vol_24",
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "last_funding_rate",
    "funding_pressure",
    "funding_abs",
    "funding_roc_288",
    "funding_oi_divergence",
    "oi_change_rate",
    "long_squeeze_risk",
    "crowding_pressure",
    "crowded_short_squeeze_risk",
    "crowded_long_unwind_risk",
    "cvp_volume_imbalance",
    "cvp_regime",
    "regime_trending",
    "regime_persistence",
]

FLOW_EXOG = [
    "smart_money_flow",
    "ofi_acceleration",
    "cvp_volume_imbalance",
    "whale_retail_ratio",
    "net_taker_ratio",
    "taker_acceleration",
    "taker_buy_base",
    "taker_buy_quote",
    "cvd_288",
    "price_cvd_divergence",
    "funding_oi_divergence",
    "btc_lead_eth_follow_gap_3",
    "btc_volume_impulse_z",
]

CYCLE_EXOG = [
    "hour_sin",
    "hour_cos",
    "session_us",
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "smart_money_flow",
    "ofi_acceleration",
    "net_taker_ratio",
    "taker_acceleration",
    "cvp_volume_imbalance",
    "funding_pressure",
    "oi_change_rate",
    "long_squeeze_risk",
    "crowding_pressure",
    "eth_btc_ret_spread_12",
    "btc_lead_eth_follow_gap_3",
    "price_cvd_divergence",
    "regime_trending",
    "vwap_dist_24",
    "vwap_dist_96",
    "anchored_vwap_session_dist",
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "range_contraction_breakout_dir",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "cvp_regime",
    "regime_persistence",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrain role AI models with audited role-specific inputs, then evaluate 2026.")
    p.add_argument("--train-2024", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-2025", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-2026", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--patchtst-steps", type=int, default=260)
    p.add_argument("--tide-steps", type=int, default=220)
    p.add_argument("--dlinear-steps", type=int, default=220)
    p.add_argument("--timesnet-steps", type=int, default=120)
    p.add_argument("--chunk-size", type=int, default=30000)
    p.add_argument("--timesnet-accelerator", choices=("cpu", "gpu", "auto"), default="cpu")
    p.add_argument("--models", default="patchtst,tide,dlinear,timesnet")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--startup-check-only", action="store_true")
    return p.parse_args()


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def _read_frame(path: Path, *, expected_year: int | None, limit: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if limit > 0:
        frame = frame.tail(int(limit)).reset_index(drop=True)
    if "timestamp" not in frame.columns:
        raise KeyError(f"{path} missing timestamp")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if expected_year is not None:
        years = sorted(frame["timestamp"].dt.year.unique().tolist())
        if years != [int(expected_year)]:
            raise RuntimeError(f"{path} year guard failed: expected={[int(expected_year)]} actual={years}")
    return frame.replace([np.inf, -np.inf], np.nan)


def _required_cols(frame: pd.DataFrame, cols: list[str], *, name: str) -> list[str]:
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{name} missing required input columns: {missing}")
    return list(dict.fromkeys(cols))


def _add_patchmix_core(frame: pd.DataFrame) -> pd.DataFrame:
    core = _core_features(frame)
    out = frame.copy()
    for col in core.columns:
        out[col] = core[col].to_numpy()
    return out


def _nf_frame(frame: pd.DataFrame, target_col: str, exog_cols: list[str], *, include_close: bool) -> pd.DataFrame:
    cols = [target_col, *exog_cols]
    if include_close:
        cols = ["close", *cols]
    cols = list(dict.fromkeys(cols))
    _required_cols(frame, cols, name="nf_frame")
    out = frame[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # Training/inference warmup missing is local-only. Feature contract still requires exact timestamps.
    out = out.ffill().fillna(0.0)
    out["ds"] = pd.to_datetime(frame["timestamp"], errors="raise")
    out["unique_id"] = "ETH"
    out = out.rename(columns={target_col: "y"})
    return out.sort_values("ds").reset_index(drop=True)


def _train_nf_model(
    frame: pd.DataFrame,
    *,
    model_name: str,
    target_col: str,
    exog_cols: list[str],
    out_path: Path,
    max_steps: int,
    include_close: bool,
    timesnet_accelerator: str,
) -> dict[str, Any]:
    from neuralforecast import NeuralForecast
    from neuralforecast.models import DLinear, PatchTST, TiDE, TimesNet

    if out_path.exists():
        shutil.rmtree(out_path)
    df_nf = _nf_frame(frame, target_col, exog_cols, include_close=include_close)
    common = {"max_steps": int(max_steps), "scaler_type": "standard"}
    if model_name == "PatchTST":
        model = PatchTST(h=10, input_size=60, hist_exog_list=exog_cols, revin=False, **common)
    elif model_name == "TiDE":
        model = TiDE(h=10, input_size=60, hist_exog_list=exog_cols, hidden_size=256, **common)
    elif model_name == "DLinear":
        model = DLinear(h=3, input_size=60, moving_avg_window=7, **common)
    elif model_name == "TimesNet":
        model = TimesNet(h=12, input_size=256, hidden_size=64, accelerator=timesnet_accelerator, devices=1, **common)
    else:
        raise ValueError(model_name)
    nf = NeuralForecast(models=[model], freq="5min")
    nf.fit(df=df_nf)
    nf.save(path=str(out_path), overwrite=True)
    contract = {
        "model_id": MODEL_ID,
        "model_name": model_name,
        "target_col": target_col,
        "input_cols": exog_cols,
        "include_close": bool(include_close),
        "train_year": 2024,
        "max_steps": int(max_steps),
        "path": str(out_path),
    }
    (out_path / "reworked_input_contract.json").write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    return contract


def _load_temp_model(model_name: str, model_path: Path):
    from neuralforecast import NeuralForecast

    if model_name == "PatchTST":
        model = PatchTSTForecaster()
    elif model_name == "TiDE":
        model = TiDEVolatilityForecaster()
    elif model_name == "DLinear":
        model = DLinearOFIForecaster()
    elif model_name == "TimesNet":
        model = TimesNetCycleForecaster()
    else:
        raise ValueError(model_name)
    model.nf = NeuralForecast.load(path=str(model_path))
    model.available = True
    return model


def _runtime_nf_frame(frame: pd.DataFrame, contract: dict[str, Any]) -> pd.DataFrame:
    model_name = str(contract["model_name"])
    if model_name == "TiDE":
        y = _past_adverse_risk(frame, horizon=6)
    elif model_name == "DLinear":
        y = _flow_pressure(frame)
    elif model_name == "TimesNet":
        y = _rolling_vwap_dev(frame, window=60) * 100.0
    elif model_name == "PatchTST":
        y = _past_barrier_proxy(frame, horizon=6)
    else:
        raise ValueError(model_name)
    work = frame.copy()
    work["y"] = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    cols = ["y", *list(contract.get("input_cols", []))]
    if bool(contract.get("include_close", False)):
        cols = ["close", *cols]
    cols = list(dict.fromkeys(cols))
    _required_cols(work, cols, name=f"{model_name} runtime")
    return work[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)


def _write_features(frame: pd.DataFrame, out_csv: Path, contracts: dict[str, dict[str, Any]], *, chunk_size: int) -> pd.DataFrame:
    if out_csv.exists():
        got = pd.read_csv(out_csv)
        got["timestamp"] = pd.to_datetime(got["timestamp"], errors="raise")
        return got
    parts = [frame[["timestamp"]].reset_index(drop=True)]
    sanitized: dict[str, list[str]] = {}
    for key, contract in contracts.items():
        model = _load_temp_model(contract["model_name"], Path(contract["path"]))
        model._prepare_data = lambda df, c=contract: _runtime_nf_frame(df, c)
        feat = model.predict_batch(frame, chunk_size=int(chunk_size)).reset_index(drop=True)
        bad = feat.replace([np.inf, -np.inf], np.nan).isna().any()
        if bool(bad.any()):
            cols = bad[bad].index.tolist()
            sanitized[key] = cols
            feat[cols] = feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        parts.append(feat.astype("float32"))
    out = pd.concat(parts, axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    (out_csv.with_suffix(".manifest.json")).write_text(
        json.dumps({"rows": len(out), "columns": list(out.columns), "sanitized_nonfinite_columns": sanitized}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def _merge_exact(base: pd.DataFrame, feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    got = base[["timestamp"]].merge(feat[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    bad = [c for c in cols if got[c].replace([np.inf, -np.inf], np.nan).isna().any()]
    if bad:
        raise RuntimeError(f"exact timestamp join produced missing values: {bad}")
    return got


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: retrain_ai_role_models_reworked_inputs_20260530")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = build_targets(_add_patchmix_core(_read_frame(args.train_2024, expected_year=2024, limit=int(args.limit))))
    score25 = _add_patchmix_core(_read_frame(args.score_2025, expected_year=2025, limit=int(args.limit)))
    score26 = _add_patchmix_core(_read_frame(args.score_2026, expected_year=2026, limit=int(args.limit)))
    requested = {m.strip().lower() for m in str(args.models).split(",") if m.strip()}
    contracts: dict[str, dict[str, Any]] = {}
    if "patchtst" in requested:
        exog = _required_cols(train, list(dict.fromkeys([*OLD_PATCH_EXOG, *AUDITED_DIRECTION])), name="PatchTST")
        contracts["patchtst"] = _train_nf_model(
            train,
            model_name="PatchTST",
            target_col="y_dir_edge",
            exog_cols=exog,
            out_path=args.out_dir / "nf_patchtst",
            max_steps=int(args.patchtst_steps),
            include_close=True,
            timesnet_accelerator=str(args.timesnet_accelerator),
        )
    if "tide" in requested:
        exog = _required_cols(train, RISK_EXOG, name="TiDE")
        contracts["tide"] = _train_nf_model(
            train,
            model_name="TiDE",
            target_col="y_adverse_risk",
            exog_cols=exog,
            out_path=args.out_dir / "nf_tide",
            max_steps=int(args.tide_steps),
            include_close=True,
            timesnet_accelerator=str(args.timesnet_accelerator),
        )
    if "dlinear" in requested:
        exog = _required_cols(train, FLOW_EXOG, name="DLinear")
        contracts["dlinear"] = _train_nf_model(
            train,
            model_name="DLinear",
            target_col="y_flow_pressure",
            exog_cols=exog,
            out_path=args.out_dir / "nf_dlinear",
            max_steps=int(args.dlinear_steps),
            include_close=False,
            timesnet_accelerator=str(args.timesnet_accelerator),
        )
    if "timesnet" in requested:
        exog = _required_cols(train, CYCLE_EXOG, name="TimesNet")
        contracts["timesnet"] = _train_nf_model(
            train,
            model_name="TimesNet",
            target_col="y_vwap_dev",
            exog_cols=exog,
            out_path=args.out_dir / "nf_timesnet",
            max_steps=int(args.timesnet_steps),
            include_close=True,
            timesnet_accelerator=str(args.timesnet_accelerator),
        )

    feat25 = _write_features(score25, args.out_dir / "role_features_2025_reworked.csv", contracts, chunk_size=int(args.chunk_size))
    feat26 = _write_features(score26, args.out_dir / "role_features_2026_reworked.csv", contracts, chunk_size=int(args.chunk_size))

    patch_cols = ["ai_dir_edge", "ai_dir_p_up", "ai_dir_p_down", "ai_dir_p_flat", "ai_dir_entropy"]
    tide_cols = ["ai_adverse_risk", "ai_reward_risk", "ai_vol_regime_pct", "tide_vol_raw", "tide_vol_zscore"]
    dlinear_cols = ["ai_flow_pressure", "ai_flow_exhaustion", "ai_flow_flip_prob", "ai_flow_slope", "dlinear_smf_ema", "dlinear_smf_slope"]
    times_cols = ["ai_anchor_revert_prob", "ai_anchor_overheat", "ai_anchor_trend_escape_prob", "timesnet_cycle_sin", "timesnet_cycle_cos", "timesnet_cycle_delta"]

    summary: dict[str, Any] = {
        "type": MODEL_ID,
        "train_contract": "2024-only NF training; 2025 and 2026 exact timestamp scoring",
        "contracts": contracts,
        "artifacts": {
            "features_2025": str(args.out_dir / "role_features_2025_reworked.csv"),
            "features_2026": str(args.out_dir / "role_features_2026_reworked.csv"),
        },
    }
    if "patchtst" in contracts:
        summary["patchtst_direction_2026"] = _direction_metrics(score26, _merge_exact(score26, feat26, patch_cols), horizons=[6, 12])
    if "tide" in contracts:
        summary["tide_risk_2026"] = _risk_metrics(score26, _merge_exact(score26, feat26, tide_cols))
    if "dlinear" in contracts:
        summary["dlinear_trend_flow_2026"] = _trend_flow_metrics(score26, _merge_exact(score26, feat26, dlinear_cols))
    if "timesnet" in contracts:
        summary["timesnet_cycle_entry_quality_2026"] = _timesnet_metrics(score26, _merge_exact(score26, feat26, times_cols))
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
