#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_router import (  # noqa: E402
    DLinearOFIForecaster,
    PatchTSTForecaster,
    TiDEVolatilityForecaster,
    TimesNetCycleForecaster,
)
from scripts.build_ai_patchmix_direction_core_20260530 import _read_frame  # noqa: E402
from scripts.sweep_ai_patchmix_h6_label_params_20260530 import _labels  # noqa: E402


DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/ai_role_specific_eval_20260530"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run role-specific AI model evaluations without AI-output ensembling.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--chunk-size", type=int, default=6000)
    p.add_argument("--skip-generate", action="store_true")
    p.add_argument("--limit", type=int, default=0)
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


def _write_tsfm_features(frame: pd.DataFrame, out_csv: Path, *, chunk_size: int) -> pd.DataFrame:
    if out_csv.exists():
        got = pd.read_csv(out_csv)
        got["timestamp"] = pd.to_datetime(got["timestamp"], errors="raise")
        return got
    models = [
        ("patchtst", PatchTSTForecaster()),
        ("tide", TiDEVolatilityForecaster()),
        ("dlinear", DLinearOFIForecaster()),
        ("timesnet", TimesNetCycleForecaster()),
    ]
    parts = [frame[["timestamp"]].reset_index(drop=True)]
    availability: dict[str, bool] = {}
    sanitized: dict[str, list[str]] = {}
    for name, model in models:
        availability[name] = bool(model.available)
        if not model.available:
            raise RuntimeError(f"{name} forecaster is unavailable")
        feat = model.predict_batch(frame, chunk_size=int(chunk_size)).reset_index(drop=True)
        bad = feat.replace([np.inf, -np.inf], np.nan).isna().any()
        if bool(bad.any()):
            bad_cols = bad[bad].index.tolist()
            sanitized[name] = bad_cols
            feat[bad_cols] = feat[bad_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            if feat[bad_cols].replace([np.inf, -np.inf], np.nan).isna().any().any():
                raise RuntimeError(f"{name} generated non-finite columns after explicit zero warmup normalization: {bad_cols}")
        parts.append(feat.astype("float32"))
    out = pd.concat(parts, axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    (out_csv.with_suffix(".manifest.json")).write_text(
        json.dumps(
            {
                "rows": len(out),
                "columns": list(out.columns),
                "availability": availability,
                "sanitized_nonfinite_columns": sanitized,
                "sanitization_rule": "TSFM rolling/warmup non-finite values are set to 0.0 and recorded here; no timestamp gaps are allowed.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return out


def _merge_exact(base: pd.DataFrame, feat: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    got = base[["timestamp"]].merge(feat[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    bad = [c for c in cols if got[c].replace([np.inf, -np.inf], np.nan).isna().any()]
    if bad:
        raise RuntimeError(f"exact timestamp join produced missing values: {bad}")
    return got


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _future_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(mode)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def _corr(a: np.ndarray, b: np.ndarray) -> float | None:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if int(ok.sum()) < 10:
        return None
    if float(np.std(a[ok])) <= 1e-12 or float(np.std(b[ok])) <= 1e-12:
        return None
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def _direction_metrics(frame: pd.DataFrame, feat: pd.DataFrame, *, horizons: list[int]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    pred = np.argmax(feat[["ai_dir_p_flat", "ai_dir_p_down", "ai_dir_p_up"]].to_numpy(dtype=float), axis=1)
    edge = feat["ai_dir_edge"].to_numpy(dtype=float)
    for h in horizons:
        lab = _labels(
            frame,
            horizon=h,
            min_edge=0.0012,
            atr_mult=0.18 if h == 6 else 0.22,
            mae_penalty=0.40 if h == 6 else 0.55,
            cost=0.00055,
            margin=0.00025 if h == 6 else 0.00035,
        )
        valid = lab["valid"].to_numpy(dtype=bool)
        y = lab["label"].to_numpy(dtype=int)
        out[f"h{h}"] = {
            "bacc": float(balanced_accuracy_score(y[valid], pred[valid])),
            "up_auc": _safe_auc((y[valid] == 2).astype(int), edge[valid]),
            "down_auc": _safe_auc((y[valid] == 1).astype(int), -edge[valid]),
            "pred_counts": np.bincount(pred[valid], minlength=3).astype(int).tolist(),
        }
    return out


def _chronos_metrics(frame: pd.DataFrame, chronos: pd.DataFrame) -> dict[str, Any]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    fut = close.shift(-6)
    actual = np.log((fut / close).replace([np.inf, -np.inf], np.nan)).to_numpy(dtype=float)
    valid = np.isfinite(actual)
    q10 = chronos["chronos_h6_q10"].to_numpy(dtype=float)
    q50 = chronos["chronos_h6_q50"].to_numpy(dtype=float)
    q90 = chronos["chronos_h6_q90"].to_numpy(dtype=float)
    width = chronos["chronos_h6_width"].to_numpy(dtype=float)
    pred_dir = np.where(q50 > 0.0, 2, np.where(q50 < 0.0, 1, 0))
    lab = _labels(frame, horizon=6, min_edge=0.0012, atr_mult=0.18, mae_penalty=0.40, cost=0.00055, margin=0.00025)
    lv = lab["valid"].to_numpy(dtype=bool) & valid
    y = lab["label"].to_numpy(dtype=int)
    return {
        "h6_bacc_q50_sign": float(balanced_accuracy_score(y[lv], pred_dir[lv])),
        "median_ret_corr": _corr(q50[valid], actual[valid]),
        "downside_auc": _safe_auc((actual[valid] < -0.0012).astype(int), -q10[valid]),
        "large_move_auc": _safe_auc((np.abs(actual[valid]) > np.nanquantile(np.abs(actual[valid]), 0.70)).astype(int), width[valid]),
        "q10_coverage": float(np.mean(actual[valid] >= q10[valid])),
        "q90_coverage": float(np.mean(actual[valid] <= q90[valid])),
    }


def _risk_metrics(frame: pd.DataFrame, feat: pd.DataFrame) -> dict[str, Any]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    out: dict[str, Any] = {}
    for h in (6, 12):
        fut_high = _future_extreme(high, h, "max")
        fut_low = _future_extreme(low, h, "min")
        adverse_range = np.maximum((fut_high / close - 1.0).abs(), (1.0 - fut_low / close).abs()).to_numpy(dtype=float)
        valid = np.isfinite(adverse_range)
        top_risk = adverse_range[valid] > np.nanquantile(adverse_range[valid], 0.70)
        risk_score = feat["tide_vol_zscore"].to_numpy(dtype=float)[valid]
        raw_score = feat["tide_vol_raw"].to_numpy(dtype=float)[valid]
        out[f"h{h}"] = {
            "top30_adverse_auc_z": _safe_auc(top_risk.astype(int), risk_score),
            "top30_adverse_auc_raw": _safe_auc(top_risk.astype(int), raw_score),
            "adverse_corr_z": _corr(risk_score, adverse_range[valid]),
            "adverse_corr_raw": _corr(raw_score, adverse_range[valid]),
        }
    return out


def _trend_flow_metrics(frame: pd.DataFrame, feat: pd.DataFrame) -> dict[str, Any]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    out: dict[str, Any] = {}
    for h in (12, 24):
        ret = (close.shift(-h) / close - 1.0).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
        valid = np.isfinite(ret)
        flow = feat["dlinear_smf_ema"].to_numpy(dtype=float)
        slope = feat["dlinear_smf_slope"].to_numpy(dtype=float)
        out[f"h{h}"] = {
            "trend_auc_flow": _safe_auc((ret[valid] > 0).astype(int), flow[valid]),
            "trend_auc_slope": _safe_auc((ret[valid] > 0).astype(int), slope[valid]),
            "ret_corr_flow": _corr(flow[valid], ret[valid]),
            "ret_corr_slope": _corr(slope[valid], ret[valid]),
        }
    return out


def _timesnet_metrics(frame: pd.DataFrame, feat: pd.DataFrame) -> dict[str, Any]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    fut = close.shift(-6)
    ret = (fut / close - 1.0).replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    valid = np.isfinite(ret)
    quality = (np.abs(ret[valid]) > np.nanquantile(np.abs(ret[valid]), 0.70)).astype(int)
    return {
        "entry_quality_auc_anchor_revert": _safe_auc(quality, feat["ai_anchor_revert_prob"].to_numpy(dtype=float)[valid]),
        "entry_quality_auc_trend_escape": _safe_auc(quality, feat["ai_anchor_trend_escape_prob"].to_numpy(dtype=float)[valid]),
        "cycle_delta_ret_corr": _corr(feat["timesnet_cycle_delta"].to_numpy(dtype=float)[valid], ret[valid]),
    }


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train = _read_frame(args.train_csv, int(args.limit))
    score = _read_frame(args.score_csv, int(args.limit))
    train_feat = _write_tsfm_features(train, args.out_dir / "tsfm_role_features_2025_exact.csv", chunk_size=int(args.chunk_size))
    score_feat = _write_tsfm_features(score, args.out_dir / "tsfm_role_features_2026_exact.csv", chunk_size=int(args.chunk_size))

    patch_cols = ["ai_dir_edge", "ai_dir_p_up", "ai_dir_p_down", "ai_dir_p_flat", "ai_dir_entropy"]
    tide_cols = ["ai_adverse_risk", "ai_reward_risk", "ai_vol_regime_pct", "tide_vol_raw", "tide_vol_zscore"]
    dlinear_cols = ["ai_flow_pressure", "ai_flow_exhaustion", "ai_flow_flip_prob", "ai_flow_slope", "dlinear_smf_ema", "dlinear_smf_slope"]
    times_cols = ["ai_anchor_revert_prob", "ai_anchor_overheat", "ai_anchor_trend_escape_prob", "timesnet_cycle_sin", "timesnet_cycle_cos", "timesnet_cycle_delta"]
    train_patch = _merge_exact(train, train_feat, patch_cols)
    score_patch = _merge_exact(score, score_feat, patch_cols)
    score_tide = _merge_exact(score, score_feat, tide_cols)
    score_dlinear = _merge_exact(score, score_feat, dlinear_cols)
    score_times = _merge_exact(score, score_feat, times_cols)

    chronos_path = ROOT / "tmp/causal_regen_20260516/ai_chronos_h6_direction_20260530/chronos_h6_2026.csv"
    chronos = pd.read_csv(chronos_path)
    chronos["timestamp"] = pd.to_datetime(chronos["timestamp"], errors="raise")
    chronos_cols = [c for c in chronos.columns if c.startswith("chronos_h6_")]
    score_chronos = _merge_exact(score, chronos, chronos_cols)

    summary = {
        "type": "ai_role_specific_eval_20260530",
        "contract": "exact timestamp TSFM regeneration; no cross-model ensembling for role metrics",
        "artifacts": {
            "train_features": str(args.out_dir / "tsfm_role_features_2025_exact.csv"),
            "score_features": str(args.out_dir / "tsfm_role_features_2026_exact.csv"),
        },
        "patchtst_direction": _direction_metrics(score, score_patch, horizons=[6, 12]),
        "chronos_distribution": _chronos_metrics(score, score_chronos),
        "tide_risk": _risk_metrics(score, score_tide),
        "dlinear_trend_flow": _trend_flow_metrics(score, score_dlinear),
        "timesnet_cycle_entry_quality": _timesnet_metrics(score, score_times),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
