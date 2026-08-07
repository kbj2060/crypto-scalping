#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_router import _flow_pressure  # noqa: E402
from scripts.build_ai_patchmix_direction_core_20260530 import _read_frame  # noqa: E402
from scripts.sweep_ai_patchmix_h6_label_params_20260530 import _labels  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/chronos_multiseries_standalone_20260530"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate standalone Chronos multi-series / derived-series direction signals.")
    p.add_argument("--val-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--oos-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-id", default="amazon/chronos-t5-tiny")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--prediction-length", type=int, default=6)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--series", default="log_close,ret6_z,flow_pressure,funding_pressure,cvd_288,price_cvd_divergence,vwap_dist_96,range_breakout")
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


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        raise KeyError(f"missing required series column: {col}")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _zscore(s: pd.Series, window: int = 288) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = x.rolling(window, min_periods=max(12, window // 12)).mean()
    std = x.rolling(window, min_periods=max(12, window // 12)).std()
    return ((x - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _build_series(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, str]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    if name == "log_close":
        return np.log(close.to_numpy(dtype=np.float32)), "delta"
    if name == "ret6_z":
        ret6 = np.log(close / close.shift(6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _zscore(ret6, 288).to_numpy(dtype=np.float32), "level"
    if name == "flow_pressure":
        return pd.Series(_flow_pressure(frame), index=frame.index).fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=np.float32), "level"
    if name == "funding_pressure":
        return _zscore(_num(frame, "funding_pressure"), 288).to_numpy(dtype=np.float32), "level"
    if name == "cvd_288":
        return _zscore(_num(frame, "cvd_288"), 288).to_numpy(dtype=np.float32), "level"
    if name == "price_cvd_divergence":
        return _zscore(_num(frame, "price_cvd_divergence"), 288).to_numpy(dtype=np.float32), "level"
    if name == "vwap_dist_96":
        return _zscore(_num(frame, "vwap_dist_96"), 288).to_numpy(dtype=np.float32), "level"
    if name == "range_breakout":
        return _zscore(_num(frame, "range_contraction_breakout_dir"), 288).to_numpy(dtype=np.float32), "level"
    raise ValueError(f"unknown series: {name}")


def _chronos_quantiles(
    values: np.ndarray,
    *,
    mode: str,
    out_csv: Path,
    timestamps: pd.Series,
    model_id: str,
    context_length: int,
    prediction_length: int,
    stride: int,
    batch_size: int,
) -> pd.DataFrame:
    if out_csv.exists():
        got = pd.read_csv(out_csv)
        got["timestamp"] = pd.to_datetime(got["timestamp"], errors="raise")
        return got
    from chronos import ChronosPipeline

    pipe = ChronosPipeline.from_pretrained(
        model_id,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
    )
    n = len(values)
    idx = np.arange(int(context_length), n, max(1, int(stride)), dtype=np.int64)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    cols = ["q10", "q50", "q90", "width", "mean"]
    out = pd.DataFrame(np.nan, index=np.arange(n), columns=cols, dtype="float32")
    qlevels = [0.1, 0.5, 0.9]
    with torch.no_grad():
        for start in range(0, len(idx), max(1, int(batch_size))):
            batch_idx = idx[start : start + int(batch_size)]
            windows = [torch.as_tensor(values[i - context_length : i], dtype=torch.float32) for i in batch_idx]
            quantiles, mean = pipe.predict_quantiles(windows, prediction_length=int(prediction_length), quantile_levels=qlevels)
            q = quantiles[:, -1, :].detach().float().cpu().numpy()
            m = mean[:, -1].detach().float().cpu().numpy()
            if mode == "delta":
                cur = values[batch_idx]
                vals = np.column_stack([q[:, 0] - cur, q[:, 1] - cur, q[:, 2] - cur, q[:, 2] - q[:, 0], m - cur])
            else:
                vals = np.column_stack([q[:, 0], q[:, 1], q[:, 2], q[:, 2] - q[:, 0], m])
            out.loc[batch_idx, cols] = vals.astype("float32")
    out[cols] = out[cols].ffill().fillna(0.0)
    result = pd.concat([pd.DataFrame({"timestamp": pd.to_datetime(timestamps).reset_index(drop=True)}), out.reset_index(drop=True)], axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    if len(np.unique(y)) < 2:
        return None
    try:
        return float(roc_auc_score(y, score))
    except Exception:
        return None


def _metrics(frame: pd.DataFrame, q: pd.DataFrame, *, threshold: float = 0.0, invert: bool = False) -> dict[str, Any]:
    lab = _labels(frame, horizon=6, min_edge=0.0012, atr_mult=0.18, mae_penalty=0.40, cost=0.00055, margin=0.00025)
    valid = lab["valid"].to_numpy(dtype=bool)
    y = lab["label"].to_numpy(dtype=int)
    sig = q["q50"].to_numpy(dtype=float)
    if invert:
        sig = -sig
    pred = np.where(np.abs(sig) <= float(threshold), 0, np.where(sig > 0.0, 2, 1))
    width = q["width"].to_numpy(dtype=float)
    return {
        "threshold": float(threshold),
        "invert": bool(invert),
        "accuracy": float(accuracy_score(y[valid], pred[valid])),
        "balanced_accuracy": float(balanced_accuracy_score(y[valid], pred[valid])),
        "up_auc": _safe_auc((y[valid] == 2).astype(int), sig[valid]),
        "down_auc": _safe_auc((y[valid] == 1).astype(int), -sig[valid]),
        "large_move_auc": _safe_auc((y[valid] != 0).astype(int), width[valid]),
        "pred_counts": np.bincount(pred[valid], minlength=3).astype(int).tolist(),
        "label_counts": np.bincount(y[valid], minlength=3).astype(int).tolist(),
    }


def _select_threshold(val_frame: pd.DataFrame, q_val: pd.DataFrame) -> dict[str, Any]:
    sig0 = q_val["q50"].to_numpy(dtype=float)
    finite = np.isfinite(sig0)
    candidates = sorted(set([0.0] + [float(x) for x in np.nanquantile(np.abs(sig0[finite]), np.linspace(0.05, 0.70, 14))]))
    best: dict[str, Any] | None = None
    for invert in (False, True):
        for threshold in candidates:
            got = _metrics(val_frame, q_val, threshold=threshold, invert=invert)
            if best is None or float(got["balanced_accuracy"]) > float(best["balanced_accuracy"]):
                best = got
    assert best is not None
    return best


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: test_chronos_multiseries_standalone_20260530")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    val = _read_frame(args.val_csv, int(args.limit))
    oos = _read_frame(args.oos_csv, int(args.limit))
    series = [x.strip() for x in str(args.series).split(",") if x.strip()]
    results: dict[str, Any] = {
        "type": "chronos_multiseries_standalone_20260530",
        "contract": "Chronos standalone zero-shot only; no downstream CatBoost/meta head. Threshold/inversion selected on 2025 and fixed on 2026.",
        "series": {},
    }
    for name in series:
        val_values, mode = _build_series(val, name)
        oos_values, mode_oos = _build_series(oos, name)
        if mode_oos != mode:
            raise RuntimeError(f"mode mismatch for {name}: {mode} vs {mode_oos}")
        q_val = _chronos_quantiles(
            val_values,
            mode=mode,
            out_csv=args.out_dir / f"{name}_val2025_chronos.csv",
            timestamps=val["timestamp"],
            model_id=str(args.model_id),
            context_length=int(args.context_length),
            prediction_length=int(args.prediction_length),
            stride=int(args.stride),
            batch_size=int(args.batch_size),
        )
        q_oos = _chronos_quantiles(
            oos_values,
            mode=mode,
            out_csv=args.out_dir / f"{name}_oos2026_chronos.csv",
            timestamps=oos["timestamp"],
            model_id=str(args.model_id),
            context_length=int(args.context_length),
            prediction_length=int(args.prediction_length),
            stride=int(args.stride),
            batch_size=int(args.batch_size),
        )
        raw_val = _metrics(val, q_val, threshold=0.0, invert=False)
        raw_oos = _metrics(oos, q_oos, threshold=0.0, invert=False)
        selected = _select_threshold(val, q_val)
        tuned_oos = _metrics(oos, q_oos, threshold=float(selected["threshold"]), invert=bool(selected["invert"]))
        results["series"][name] = {
            "mode": mode,
            "raw_val": raw_val,
            "raw_oos": raw_oos,
            "selected_on_val": selected,
            "oos_with_val_selection": tuned_oos,
            "artifacts": {
                "val": str(args.out_dir / f"{name}_val2025_chronos.csv"),
                "oos": str(args.out_dir / f"{name}_oos2026_chronos.csv"),
            },
        }
        (args.out_dir / "summary.partial.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    best = sorted(
        ((k, v["oos_with_val_selection"]) for k, v in results["series"].items()),
        key=lambda kv: float(kv[1]["balanced_accuracy"]),
        reverse=True,
    )
    results["best_by_oos_bacc"] = [{"series": k, **v} for k, v in best]
    (args.out_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
