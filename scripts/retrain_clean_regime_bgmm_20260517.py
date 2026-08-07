#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.mixture import BayesianGaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.certified_teacher_regime_moe import (  # noqa: E402
    CLEAN_PREFIX,
    clean_regime_factors,
    clean_regime_fit_columns,
)


MODEL_ID = "clean_regime_2024_unsup_bgmm_v5_20260517"
DEFAULT_TRAIN_2024 = ROOT / "data/splits/year_oos/training_features_2024.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/clean_regime_bgmm_v5_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_regime_bgmm_v5_20260517_report.json"
DEFAULT_TRANSFORMS = (
    ROOT / "data/splits/year_oos/training_features_2024.csv",
    ROOT / "data/splits/year_oos/training_features_2025.csv",
)
NON_FEATURES = {
    "timestamp",
    "open",
    "high",
    "low",
    "close",
}
FORBIDDEN_FRAGMENTS = (
    "future",
    "target",
    "label",
    "realized",
    "trade_pnl",
    "cash_after",
    "legacy",
    "hdb",
    "hmm_",
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _read(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return df.reset_index(drop=True)


def _safe_numeric(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _safe_numeric(frame, c) for c in cols}, index=frame.index)


def _candidate_columns(frame: pd.DataFrame) -> list[str]:
    priority = clean_regime_fit_columns(frame)
    extra_hints = [
        "log_return",
        "rsi",
        "macd_hist",
        "bb_width_z",
        "hma_slope",
        "wick_ratio",
        "garman_klass_vol",
        "realized_vol_ratio",
        "rogers_satchell_vol",
        "parkinson_vol",
        "amihud_illiquidity_z",
        "btc_corr_60",
        "eth_btc_ratio_change",
        "fvg_dist",
        "chop_index",
        "cvp_poc_dist",
        "cvp_cluster_position",
        "cvp_volume_imbalance",
        "breakout_strength",
        "long_squeeze_risk",
        "funding_price_divergence",
        "ofi_acceleration",
        "kalman_velocity",
        "realized_skewness",
        "ofti",
        "kel",
        "garch_vol_z",
        "jump_z",
        "evt_excess_z",
        "liquidity_vacuum",
        "execution_quality",
        "crowding_pressure",
    ]
    selected: list[str] = []
    for col in priority + extra_hints:
        lower = col.lower()
        if col in selected or col in NON_FEATURES:
            continue
        if col.startswith(CLEAN_PREFIX):
            continue
        if any(fragment in lower for fragment in FORBIDDEN_FRAGMENTS):
            continue
        if col in frame.columns and pd.to_numeric(frame[col], errors="coerce").notna().any():
            selected.append(col)
    return selected


def _future_path_labels(frame: pd.DataFrame, horizon: int = 36) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=float)
    high = pd.to_numeric(frame.get("high", frame["close"]), errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=float)
    low = pd.to_numeric(frame.get("low", frame["close"]), errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().to_numpy(dtype=float)
    n = len(frame)
    fut_ret = np.full(n, np.nan)
    fut_mfe_long = np.full(n, np.nan)
    fut_mae_long = np.full(n, np.nan)
    fut_mfe_short = np.full(n, np.nan)
    fut_mae_short = np.full(n, np.nan)
    for i in range(0, max(n - horizon - 1, 0)):
        entry = close[i]
        if not np.isfinite(entry) or entry <= 0:
            continue
        hi = float(np.nanmax(high[i + 1 : i + horizon + 1]))
        lo = float(np.nanmin(low[i + 1 : i + horizon + 1]))
        last = float(close[i + horizon])
        fut_ret[i] = last / entry - 1.0
        fut_mfe_long[i] = hi / entry - 1.0
        fut_mae_long[i] = max(0.0, 1.0 - lo / entry)
        fut_mfe_short[i] = entry / max(lo, 1e-12) - 1.0
        fut_mae_short[i] = max(0.0, hi / entry - 1.0)
    label = np.full(n, "normal", dtype=object)
    abs_ret = np.abs(fut_ret)
    long_quality = fut_mfe_long - fut_mae_long
    short_quality = fut_mfe_short - fut_mae_short
    range_width = np.nan_to_num(fut_mfe_long + fut_mfe_short, nan=0.0)
    valid = np.isfinite(fut_ret)
    vol_hi = np.nanquantile(range_width[valid], 0.70) if valid.any() else 0.0
    vol_lo = np.nanquantile(range_width[valid], 0.35) if valid.any() else 0.0
    ret_lo = np.nanquantile(abs_ret[valid], 0.45) if valid.any() else 0.0
    label[(fut_ret > 0.0015) & (long_quality > 0.0010) & (fut_mfe_long > fut_mfe_short * 1.08)] = "bull"
    label[(fut_ret < -0.0015) & (short_quality > 0.0010) & (fut_mfe_short > fut_mfe_long * 1.08)] = "bear"
    label[(range_width >= vol_hi) & (abs_ret <= ret_lo) & (fut_mae_long > 0.0010) & (fut_mae_short > 0.0010)] = "whipsaw"
    label[(label == "normal") & (range_width <= vol_lo) & (abs_ret <= ret_lo)] = "chop"
    out = pd.DataFrame(
        {
            "_future_label": label,
            "_future_ret": np.nan_to_num(fut_ret, nan=0.0),
            "_future_range_width": np.nan_to_num(range_width, nan=0.0),
            "_future_long_quality": np.nan_to_num(long_quality, nan=0.0),
            "_future_short_quality": np.nan_to_num(short_quality, nan=0.0),
        },
        index=frame.index,
    )
    return out.iloc[: max(n - horizon - 1, 0)].copy()


def _fit_bgmm(train: pd.DataFrame, cols: list[str], *, components: int, seed: int) -> dict[str, Any]:
    x = _matrix(train, cols)
    preprocess = make_pipeline(
        SimpleImputer(strategy="median"),
        RobustScaler(quantile_range=(10.0, 90.0)),
        PCA(n_components=min(12, max(2, len(cols))), whiten=True, random_state=seed),
    )
    xz = preprocess.fit_transform(x)
    model = BayesianGaussianMixture(
        n_components=components,
        covariance_type="diag",
        weight_concentration_prior_type="dirichlet_distribution",
        weight_concentration_prior=2.0,
        reg_covar=1e-5,
        max_iter=600,
        n_init=3,
        init_params="k-means++",
        random_state=seed,
    )
    model.fit(xz)
    return {"feature_cols": cols, "preprocess": preprocess, "model": model, "components": components}


def _append_bgmm_clean(frame: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    out = frame.copy()
    factors = clean_regime_factors(out)
    for col in factors.columns:
        out[col] = factors[col].to_numpy(dtype=float)
    cols = list(bundle["feature_cols"])
    xz = bundle["preprocess"].transform(_matrix(out, cols))
    model = bundle["model"]
    labels = model.predict(xz).astype(int)
    prob = np.asarray(model.predict_proba(xz), dtype=float)
    prob /= np.clip(prob.sum(axis=1, keepdims=True), 1e-12, None)
    out[f"{CLEAN_PREFIX}cluster"] = labels
    out[f"{CLEAN_PREFIX}cluster_confidence"] = prob.max(axis=1)
    out[f"{CLEAN_PREFIX}cluster_entropy"] = -np.sum(prob * np.log(np.clip(prob, 1e-12, None)), axis=1) / math.log(prob.shape[1])
    for k in range(prob.shape[1]):
        out[f"{CLEAN_PREFIX}cluster_prob_{k}"] = prob[:, k]
    return out


def _clean_columns(frame: pd.DataFrame) -> list[str]:
    return ["timestamp"] + [c for c in frame.columns if c.startswith(CLEAN_PREFIX)]


def _cluster_report(frame: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    labels = pd.to_numeric(clean[f"{CLEAN_PREFIX}cluster"], errors="coerce").fillna(-1).astype(int)
    counts = labels.value_counts().sort_index()
    future = _future_path_labels(frame)
    joined = clean.loc[future.index].copy()
    joined = joined.join(future)
    rows: list[dict[str, Any]] = []
    for cluster in sorted(counts.index.tolist()):
        sub = joined[pd.to_numeric(joined[f"{CLEAN_PREFIX}cluster"], errors="coerce").fillna(-1).astype(int) == int(cluster)]
        label_mix = sub["_future_label"].value_counts(normalize=True).to_dict() if len(sub) else {}
        rows.append(
            {
                "cluster": int(cluster),
                "rows": int(counts.loc[cluster]),
                "share": float(counts.loc[cluster] / max(len(labels), 1)),
                "avg_confidence": float(pd.to_numeric(sub.get(f"{CLEAN_PREFIX}cluster_confidence", 0.0), errors="coerce").mean() if len(sub) else 0.0),
                "future_label_mix": {str(k): float(v) for k, v in label_mix.items()},
                "future_ret_mean": float(pd.to_numeric(sub.get("_future_ret", 0.0), errors="coerce").mean() if len(sub) else 0.0),
                "future_range_width_mean": float(pd.to_numeric(sub.get("_future_range_width", 0.0), errors="coerce").mean() if len(sub) else 0.0),
                "future_long_quality_mean": float(pd.to_numeric(sub.get("_future_long_quality", 0.0), errors="coerce").mean() if len(sub) else 0.0),
                "future_short_quality_mean": float(pd.to_numeric(sub.get("_future_short_quality", 0.0), errors="coerce").mean() if len(sub) else 0.0),
            }
        )
    return {
        "rows": int(len(clean)),
        "cluster_counts": {str(int(k)): int(v) for k, v in counts.items()},
        "avg_cluster_confidence": float(pd.to_numeric(clean[f"{CLEAN_PREFIX}cluster_confidence"], errors="coerce").mean()),
        "avg_cluster_entropy": float(pd.to_numeric(clean[f"{CLEAN_PREFIX}cluster_entropy"], errors="coerce").mean()),
        "clusters": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--transform", type=Path, action="append", default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1705)
    parser.add_argument("--write-augmented", action="store_true")
    args = parser.parse_args()

    transforms = list(args.transform or DEFAULT_TRANSFORMS)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    train = _read(args.train_2024)
    cols = _candidate_columns(train)
    if len(cols) < 8:
        raise ValueError(f"not enough clean regime columns: {len(cols)}")
    bundle = _fit_bgmm(train, cols, components=int(args.components), seed=int(args.seed))
    payload = {
        "model_id": MODEL_ID,
        "clean_prefix": CLEAN_PREFIX,
        "regime": bundle,
        "feature_cols": cols,
        "fit_source": str(args.train_2024),
        "fit_rows": int(len(train)),
        "fit_range": [str(train["timestamp"].iloc[0]), str(train["timestamp"].iloc[-1])],
        "notes": [
            "2024-only fit.",
            "BayesianGaussianMixture replaces MiniBatchKMeans for soft unsupervised state posteriors.",
            "BGMM input columns are raw causal market/model features only; clean_regime_* factors are outputs, not clustering inputs.",
            "Column prefix intentionally remains clean_regime_2024_unsup_v4_ for existing audit and downstream compatibility.",
        ],
    }
    model_path = args.out_dir / "clean_regime_bgmm_v5.joblib"
    joblib.dump(payload, model_path)

    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "clean_prefix": CLEAN_PREFIX,
        "fit_source": str(args.train_2024),
        "fit_rows": int(len(train)),
        "feature_cols": cols,
        "feature_count": int(len(cols)),
        "components": int(args.components),
        "outputs": {},
    }
    for src in transforms:
        frame = _read(src)
        clean = _append_bgmm_clean(frame, bundle)
        sidecar = args.out_dir / f"{src.stem}_clean_regime_bgmm_v5.csv"
        clean[_clean_columns(clean)].to_csv(sidecar, index=False)
        item = {
            "source": str(src),
            "rows": int(len(frame)),
            "range": [str(frame["timestamp"].iloc[0]), str(frame["timestamp"].iloc[-1])],
            "sidecar": str(sidecar),
            "clean_feature_count": int(len(_clean_columns(clean)) - 1),
            "diagnostics": _cluster_report(frame, clean),
        }
        if args.write_augmented:
            augmented = args.out_dir / f"{src.stem}_clean_regime_bgmm_v5_augmented.csv"
            clean.to_csv(augmented, index=False)
            item["augmented"] = str(augmented)
        report["outputs"][src.name] = item
        print(f"[{MODEL_ID}] wrote {sidecar} rows={len(frame)} clean_cols={item['clean_feature_count']}", flush=True)

    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] model={model_path}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
