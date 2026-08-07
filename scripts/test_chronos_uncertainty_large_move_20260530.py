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
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.ensemble_router import _flow_pressure  # noqa: E402
from scripts.build_ai_patchmix_direction_core_20260530 import _read_frame  # noqa: E402

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Chronos as an uncertainty / large-move context model.")
    p.add_argument("--val-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--oos-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model-id", default="amazon/chronos-t5-tiny")
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--prediction-length", type=int, default=6)
    p.add_argument("--horizon", type=int, default=6)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument(
        "--series",
        default=(
            "log_close,ret6_z,flow_pressure,funding_pressure,oi_change_rate,crowding_pressure,"
            "long_squeeze_risk,funding_oi_divergence,cvd_288,price_cvd_divergence,vwap_dist_96,"
            "compression_score,bb_width_pct_rank_288,range_breakout,atr14_pct,realized_vol_24"
        ),
    )
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
        raise KeyError(f"missing required Chronos input column: {col}")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)


def _zscore(s: pd.Series, window: int = 288) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)
    mean = x.rolling(window, min_periods=max(12, window // 12)).mean()
    std = x.rolling(window, min_periods=max(12, window // 12)).std()
    return ((x - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _atr14_pct(frame: pd.DataFrame) -> pd.Series:
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return (tr.rolling(14, min_periods=3).mean() / close).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _realized_vol_24(frame: pd.DataFrame) -> pd.Series:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    ret = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return ret.rolling(24, min_periods=6).std().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _build_series(frame: pd.DataFrame, name: str) -> tuple[np.ndarray, str]:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    if name == "log_close":
        return np.log(close.to_numpy(dtype=np.float32)), "delta"
    if name == "ret6_z":
        ret6 = np.log(close / close.shift(6)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return _zscore(ret6, 288).to_numpy(dtype=np.float32), "level"
    if name == "flow_pressure":
        return pd.Series(_flow_pressure(frame), index=frame.index).fillna(0.0).clip(-1.0, 1.0).to_numpy(dtype=np.float32), "level"
    if name == "range_breakout":
        return _zscore(_num(frame, "range_contraction_breakout_dir"), 288).to_numpy(dtype=np.float32), "level"
    if name == "atr14_pct":
        return _zscore(_atr14_pct(frame), 288).to_numpy(dtype=np.float32), "level"
    if name == "realized_vol_24":
        return _zscore(_realized_vol_24(frame), 288).to_numpy(dtype=np.float32), "level"
    return _zscore(_num(frame, name), 288).to_numpy(dtype=np.float32), "level"


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
            quantiles, mean = pipe.predict_quantiles(
                windows,
                prediction_length=int(prediction_length),
                quantile_levels=qlevels,
            )
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


def _future_targets(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    fut_ret = np.log(close.shift(-horizon) / close).replace([np.inf, -np.inf], np.nan)
    fut_high = high.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).max().iloc[::-1]
    fut_low = low.shift(-1).iloc[::-1].rolling(horizon, min_periods=1).min().iloc[::-1]
    upside = ((fut_high / close) - 1.0).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
    downside = (1.0 - (fut_low / close)).clip(lower=0.0).replace([np.inf, -np.inf], np.nan)
    out = pd.DataFrame(
        {
            "future_abs_ret": fut_ret.abs(),
            "future_signed_ret": fut_ret,
            "future_upside": upside,
            "future_downside": downside,
        }
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _safe_auc(y: np.ndarray, score: np.ndarray) -> float | None:
    mask = np.isfinite(score) & np.isfinite(y)
    y2 = y[mask].astype(int)
    s2 = score[mask].astype(float)
    if len(y2) < 100 or len(np.unique(y2)) < 2:
        return None
    try:
        return float(roc_auc_score(y2, s2))
    except Exception:
        return None


def _lift(y: np.ndarray, score: np.ndarray, *, top_q: float) -> dict[str, float | None]:
    mask = np.isfinite(y) & np.isfinite(score)
    if int(mask.sum()) < 100:
        return {"rate": None, "lift": None, "coverage": None}
    y2 = y[mask].astype(float)
    s2 = score[mask].astype(float)
    cut = float(np.quantile(s2, top_q))
    top = s2 >= cut
    if int(top.sum()) == 0:
        return {"rate": None, "lift": None, "coverage": None}
    base = float(y2.mean())
    rate = float(y2[top].mean())
    return {
        "rate": rate,
        "lift": None if base <= 0.0 else float(rate / base),
        "coverage": float(top.mean()),
    }


def _score_table(frame: pd.DataFrame, q: pd.DataFrame, *, horizon: int) -> dict[str, Any]:
    targets = _future_targets(frame, horizon)
    valid = targets.notna().all(axis=1).to_numpy(dtype=bool)
    q10 = q["q10"].to_numpy(dtype=float)
    q50 = q["q50"].to_numpy(dtype=float)
    q90 = q["q90"].to_numpy(dtype=float)
    width = np.maximum(q["width"].to_numpy(dtype=float), 0.0)
    mean = q["mean"].to_numpy(dtype=float)
    expected_move = np.abs(q50)
    upside_band = np.maximum(q90, 0.0)
    downside_band = np.maximum(-q10, 0.0)
    asymmetry_abs = np.abs(upside_band - downside_band)
    large_move_score = width * (1.0 + expected_move)
    base_scores = {
        "width": width,
        "expected_move": expected_move,
        "large_move_score": large_move_score,
        "asymmetry_abs": asymmetry_abs,
        "downside_band": downside_band,
        "upside_band": upside_band,
        "mean_abs": np.abs(mean),
    }
    scores: dict[str, np.ndarray] = {}
    for name, score in base_scores.items():
        scores[name] = score
        # Live-safe smoothing: only current and past Chronos outputs are used.
        # This is useful for uncertainty context because volatility/risk regimes persist across several bars.
        score_ser = pd.Series(score).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        scores[f"{name}_ewm3"] = score_ser.ewm(span=3, adjust=False, min_periods=1).mean().to_numpy(dtype=float)
        scores[f"{name}_ewm6"] = score_ser.ewm(span=6, adjust=False, min_periods=1).mean().to_numpy(dtype=float)
    future_abs = targets["future_abs_ret"].to_numpy(dtype=float)
    future_downside = targets["future_downside"].to_numpy(dtype=float)
    future_upside = targets["future_upside"].to_numpy(dtype=float)
    large_cut = float(np.nanquantile(future_abs[valid], 0.70))
    downside_cut = float(np.nanquantile(future_downside[valid], 0.70))
    upside_cut = float(np.nanquantile(future_upside[valid], 0.70))
    large_y = ((future_abs >= large_cut) & valid).astype(int)
    downside_y = ((future_downside >= downside_cut) & valid).astype(int)
    upside_y = ((future_upside >= upside_cut) & valid).astype(int)
    by_score: dict[str, Any] = {}
    for name, score in scores.items():
        by_score[name] = {
            "large_move_auc": _safe_auc(large_y[valid], score[valid]),
            "large_top10": _lift(large_y[valid], score[valid], top_q=0.90),
            "large_top20": _lift(large_y[valid], score[valid], top_q=0.80),
            "downside_auc": _safe_auc(downside_y[valid], score[valid]),
            "downside_top10": _lift(downside_y[valid], score[valid], top_q=0.90),
            "upside_auc": _safe_auc(upside_y[valid], score[valid]),
            "mean_future_abs_top10": float(np.nanmean(future_abs[valid][score[valid] >= np.nanquantile(score[valid], 0.90)])),
            "mean_future_abs_base": float(np.nanmean(future_abs[valid])),
        }
    return {
        "n_valid": int(valid.sum()),
        "horizon": int(horizon),
        "large_move_cut": large_cut,
        "downside_cut": downside_cut,
        "upside_cut": upside_cut,
        "target_prevalence": {
            "large": float(large_y[valid].mean()),
            "downside": float(downside_y[valid].mean()),
            "upside": float(upside_y[valid].mean()),
        },
        "scores": by_score,
    }


def _best_rows(results: dict[str, Any], split: str, metric: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for series, payload in results["series"].items():
        table = payload[split]["scores"]
        for score_name, metrics in table.items():
            val = metrics.get(metric)
            if val is not None:
                rows.append({"series": series, "score": score_name, metric: float(val)})
    return sorted(rows, key=lambda r: r[metric], reverse=True)


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: test_chronos_uncertainty_large_move_20260530")
        return 0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    val = _read_frame(args.val_csv, int(args.limit))
    oos = _read_frame(args.oos_csv, int(args.limit))
    series = [x.strip() for x in str(args.series).split(",") if x.strip()]
    results: dict[str, Any] = {
        "type": "chronos_uncertainty_large_move_20260530",
        "contract": (
            "Chronos zero-shot only. Role is uncertainty / large-move / adverse-move context, "
            "not long-short direction ownership. No target fitting or val distribution thresholding is used."
        ),
        "model_id": str(args.model_id),
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
        results["series"][name] = {
            "mode": mode,
            "val2025": _score_table(val, q_val, horizon=int(args.horizon)),
            "oos2026": _score_table(oos, q_oos, horizon=int(args.horizon)),
            "artifacts": {
                "val": str(args.out_dir / f"{name}_val2025_chronos.csv"),
                "oos": str(args.out_dir / f"{name}_oos2026_chronos.csv"),
            },
        }
        (args.out_dir / "summary.partial.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )

    results["best"] = {
        "val_large_move_auc": _best_rows(results, "val2025", "large_move_auc")[:10],
        "oos_large_move_auc": _best_rows(results, "oos2026", "large_move_auc")[:10],
        "val_downside_auc": _best_rows(results, "val2025", "downside_auc")[:10],
        "oos_downside_auc": _best_rows(results, "oos2026", "downside_auc")[:10],
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps(results["best"], ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
