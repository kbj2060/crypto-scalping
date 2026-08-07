#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


MODEL_ID = "ai_patchmix_direction_core_audit_v2_20260530"
DEFAULT_OUT = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
PATCH_MODEL_ID = "ibm/patchtsmixer-etth1-pretrain"
HORIZONS = (12, 24, 48)

BASE_CORE_FEATURES = (
    "ret_1",
    "ret_3",
    "ret_6",
    "ret_12",
    "ret_24",
    "atr14_pct",
    "realized_vol_24",
    "compression_ratio",
    "last_funding_rate",
    "funding_pressure",
    "oi_change_rate",
    "smart_money_flow",
    "ofi_acceleration",
    "net_taker_ratio",
    "taker_acceleration",
    "cvp_volume_imbalance",
    "vwap_dev_48",
    "lower_wick_ratio",
    "upper_wick_ratio",
)

AUDITED_FULL_FEATURES = (
    "funding_abs",
    "funding_roc_288",
    "funding_price_divergence",
    "mta_funding",
    "long_squeeze_risk",
    "crowding_pressure",
    "crowded_short_squeeze_risk",
    "crowded_long_unwind_risk",
    "compression_score",
    "atr_pct_rank_288",
    "bb_width_pct_rank_288",
    "vwap_dist_24",
    "vwap_dist_96",
    "anchored_vwap_session_dist",
    "cvd_288",
    "eth_btc_ret_spread_12",
    "btc_lead_eth_follow_gap_3",
    "btc_volume_impulse_z",
    "range_contraction_breakout_dir",
    "distance_to_day_high_low_pct",
    "price_cvd_divergence",
    "funding_oi_divergence",
    "hour_sin",
    "hour_cos",
    "session_us",
)

AUDITED_COMPACT_FEATURES = (
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
)

LOCAL_REGIME_FEATURES = (
    "cvp_regime",
    "regime_trending",
    "regime_persistence",
)

REGIME3_RISK_FEATURES = (
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
)

RAW_REQUIRED = (
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "last_funding_rate",
    "funding_pressure",
    "oi_change_rate",
    "smart_money_flow",
    "ofi_acceleration",
    "net_taker_ratio",
    "taker_acceleration",
    "cvp_volume_imbalance",
    *LOCAL_REGIME_FEATURES,
    *AUDITED_FULL_FEATURES,
)

CORE_FEATURES = (*BASE_CORE_FEATURES, *AUDITED_FULL_FEATURES)

PATCH_CHANNELS = (
    "patch_ch_momentum",
    "patch_ch_vol",
    "patch_ch_funding_oi",
    "patch_ch_flow",
    "patch_ch_cvp",
    "patch_ch_vwap_wick",
    "patch_ch_wick_balance",
    "patch_ch_squeeze_crowding",
    "patch_ch_compression_breakout",
    "patch_ch_cvd_btc",
    "patch_ch_session",
    "patch_ch_local_regime",
    "patch_ch_regime3_risk",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build PatchTSMixer direction-core AI features with a strict upstream contract.")
    p.add_argument("--train-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2024.csv")
    p.add_argument("--score-csv", type=Path, default=ROOT / "data/splits/year_oos/training_features_2025.csv")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT / "fit2024_score2025")
    p.add_argument("--out-csv", type=Path, default=DEFAULT_OUT / "fit2024_score2025" / "ai_patchmix_direction_core_2025.csv")
    p.add_argument("--patch-model-id", default=PATCH_MODEL_ID)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--stride", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=192)
    p.add_argument("--emb-dim", type=int, default=16)
    p.add_argument("--iterations", type=int, default=700)
    p.add_argument("--learning-rate", type=float, default=0.035)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--l2-leaf-reg", type=float, default=8.0)
    p.add_argument("--class-weight-power", type=float, default=0.5)
    p.add_argument("--random-seed", type=int, default=20260530)
    p.add_argument("--task-type", choices=("CPU", "GPU"), default="GPU")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--horizons", default="12,24,48", help="Comma-separated positive integer horizons, e.g. 6 or 6,12.")
    p.add_argument(
        "--input-profile",
        choices=(
            "audit_compact",
            "audit_full",
            "audit_compact_local_regime",
            "audit_compact_regime3_risk",
        ),
        default="audit_full",
    )
    p.add_argument("--train-regime3-risk-csv", type=Path, default=None)
    p.add_argument("--score-regime3-risk-csv", type=Path, default=None)
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


def _read_frame(path: Path, limit: int = 0) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if limit > 0:
        frame = frame.tail(int(limit)).reset_index(drop=True)
    missing = [c for c in RAW_REQUIRED if c not in frame.columns]
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="raise")
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    return frame.replace([np.inf, -np.inf], np.nan)


def _merge_regime3_risk_sidecar(frame: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    if path is None:
        raise ValueError("--train-regime3-risk-csv/--score-regime3-risk-csv is required for audit_compact_regime3_risk")
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path)
    missing = ["timestamp", *REGIME3_RISK_FEATURES]
    missing = [c for c in missing if c not in side.columns]
    if missing:
        raise KeyError(f"{path} missing required regime3 risk columns: {missing}")
    side["timestamp"] = pd.to_datetime(side["timestamp"], errors="raise")
    side = side[["timestamp", *REGIME3_RISK_FEATURES]].sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    before = len(frame)
    merged = frame.merge(side, on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"regime3 risk merge changed row count: before={before} after={len(merged)}")
    bad = [c for c in REGIME3_RISK_FEATURES if merged[c].replace([np.inf, -np.inf], np.nan).isna().any()]
    if bad:
        missing = merged[bad].replace([np.inf, -np.inf], np.nan).isna().any(axis=1).to_numpy()
        first_missing = int(np.argmax(missing))
        if not bool(missing[first_missing]) or not bool(missing[first_missing:].all()):
            raise RuntimeError(f"regime3 risk exact timestamp join produced non-tail missing values: {bad}")
        merged = merged.iloc[:first_missing].reset_index(drop=True)
    return merged


def _num(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        raise KeyError(col)
    return pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)


def _zscore(s: pd.Series, window: int = 288) -> pd.Series:
    mean = s.rolling(window, min_periods=max(8, window // 12)).mean()
    std = s.rolling(window, min_periods=max(8, window // 12)).std()
    return ((s - mean) / (std + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-8.0, 8.0)


def _core_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    open_ = _num(frame, "open").ffill().bfill()
    qv = _num(frame, "quote_volume").fillna(0.0).clip(lower=0.0)

    out = pd.DataFrame(index=frame.index)
    for n in (1, 3, 6, 12, 24):
        out[f"ret_{n}"] = close.pct_change(n).fillna(0.0).clip(-0.12, 0.12)

    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    out["atr14_pct"] = (tr.rolling(14, min_periods=3).mean() / close).fillna(0.0).clip(0.0, 0.2)
    out["realized_vol_24"] = out["ret_1"].rolling(24, min_periods=6).std().fillna(0.0).clip(0.0, 0.2)

    range_12 = (high.rolling(12, min_periods=3).max() / low.rolling(12, min_periods=3).min().clip(lower=1e-12) - 1.0)
    range_48 = (high.rolling(48, min_periods=12).max() / low.rolling(48, min_periods=12).min().clip(lower=1e-12) - 1.0)
    out["compression_ratio"] = (range_12 / (range_48 + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0.0, 5.0)

    for col in (
        "last_funding_rate",
        "funding_pressure",
        "oi_change_rate",
        "smart_money_flow",
        "ofi_acceleration",
        "net_taker_ratio",
        "taker_acceleration",
        "cvp_volume_imbalance",
    ):
        out[col] = _num(frame, col).fillna(0.0).clip(-10.0, 10.0)

    for col in AUDITED_FULL_FEATURES:
        out[col] = _num(frame, col).fillna(0.0).clip(-10.0, 10.0)

    for col in LOCAL_REGIME_FEATURES:
        out[col] = _num(frame, col).fillna(0.0).clip(-10.0, 10.0)

    for col in REGIME3_RISK_FEATURES:
        if col in frame.columns:
            out[col] = _num(frame, col).fillna(0.0).clip(-10.0, 10.0)

    tp = (high + low + close) / 3.0
    pv = (tp * qv).rolling(48, min_periods=12).sum()
    vv = qv.rolling(48, min_periods=12).sum().replace(0.0, np.nan)
    vwap = pv / (vv + 1e-12)
    out["vwap_dev_48"] = (close / (vwap + 1e-12) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-0.2, 0.2)

    candle_range = (high - low).abs().clip(lower=1e-12)
    body_high = pd.concat([open_, close], axis=1).max(axis=1)
    body_low = pd.concat([open_, close], axis=1).min(axis=1)
    out["upper_wick_ratio"] = ((high - body_high).clip(lower=0.0) / candle_range).fillna(0.0).clip(0.0, 1.0)
    out["lower_wick_ratio"] = ((body_low - low).clip(lower=0.0) / candle_range).fillna(0.0).clip(0.0, 1.0)
    return out.loc[:, CORE_FEATURES].astype("float32")


def _patch_channels(core: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=core.index)
    out["patch_ch_momentum"] = (
        0.40 * core["ret_3"] + 0.30 * core["ret_6"] + 0.20 * core["ret_12"] + 0.10 * core["ret_24"]
    )
    out["patch_ch_vol"] = _zscore(core["atr14_pct"] + core["realized_vol_24"] + 0.002 * core["compression_ratio"])
    out["patch_ch_funding_oi"] = np.tanh(
        40.0 * core["last_funding_rate"] + 1.2 * core["funding_pressure"] + 0.8 * core["oi_change_rate"]
    )
    out["patch_ch_flow"] = np.tanh(
        1.1 * core["smart_money_flow"] + 0.8 * core["ofi_acceleration"] + 0.7 * core["net_taker_ratio"] + 0.5 * core["taker_acceleration"]
    )
    out["patch_ch_cvp"] = np.tanh(core["cvp_volume_imbalance"])
    out["patch_ch_vwap_wick"] = np.tanh(8.0 * core["vwap_dev_48"] + core["lower_wick_ratio"] - core["upper_wick_ratio"])
    out["patch_ch_wick_balance"] = core["lower_wick_ratio"] - core["upper_wick_ratio"]
    out["patch_ch_squeeze_crowding"] = np.tanh(
        1.2 * core["long_squeeze_risk"]
        + 0.8 * core["crowding_pressure"]
        + 0.7 * core["crowded_short_squeeze_risk"]
        - 0.7 * core["crowded_long_unwind_risk"]
    )
    out["patch_ch_compression_breakout"] = np.tanh(
        1.1 * core["compression_score"]
        + 0.8 * core["range_contraction_breakout_dir"]
        + 0.5 * core["atr_pct_rank_288"]
        + 0.4 * core["bb_width_pct_rank_288"]
    )
    out["patch_ch_cvd_btc"] = np.tanh(
        1.0 * core["cvd_288"]
        + 0.8 * core["eth_btc_ret_spread_12"]
        + 0.8 * core["btc_lead_eth_follow_gap_3"]
        + 0.5 * core["btc_volume_impulse_z"]
        + 0.6 * core["price_cvd_divergence"]
    )
    out["patch_ch_session"] = np.tanh(
        0.5 * core["hour_sin"] + 0.5 * core["hour_cos"] + 0.8 * core["session_us"]
    )
    out["patch_ch_local_regime"] = np.tanh(
        0.8 * core.get("cvp_regime", 0.0)
        + 0.7 * core.get("regime_trending", 0.0)
        + 0.5 * core.get("regime_persistence", 0.0)
    )
    if "regime3_stability_h6_score" in core.columns:
        out["patch_ch_regime3_risk"] = np.tanh(
            0.8 * core["regime3_stability_h6_score"]
            - 0.9 * core["regime3_transition_h6_risk_prob"]
            - 0.7 * core["regime3_churn_h6_risk_score"]
            - 0.4 * core["regime3_transition_h6_risk_pred"]
        )
    else:
        out["patch_ch_regime3_risk"] = 0.0
    return out.loc[:, PATCH_CHANNELS].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")


def _refresh_indices(n: int, context_length: int, stride: int) -> np.ndarray:
    if n <= context_length:
        return np.array([], dtype=np.int64)
    idx = np.arange(context_length, n, max(1, int(stride)), dtype=np.int64)
    if idx.size == 0 or idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


def _patch_embeddings(
    frame: pd.DataFrame,
    *,
    model_id: str,
    context_length: int,
    stride: int,
    batch_size: int,
    emb_dim: int,
    device: torch.device,
) -> pd.DataFrame:
    from transformers import PatchTSMixerModel

    model = PatchTSMixerModel.from_pretrained(model_id, local_files_only=True).eval().to(device)
    core = _core_features(frame)
    channels = _patch_channels(core)
    values = channels.to_numpy(dtype=np.float32)
    indices = _refresh_indices(len(frame), context_length, stride)
    if indices.size == 0:
        raise ValueError(f"not enough rows for context_length={context_length}: rows={len(frame)}")

    emb_cols = [f"_patch_emb_{i:02d}" for i in range(int(emb_dim))]
    out = pd.DataFrame(np.nan, index=frame.index, columns=emb_cols, dtype="float32")
    with torch.no_grad():
        for start in range(0, len(indices), max(1, int(batch_size))):
            batch_idx = indices[start : start + int(batch_size)]
            windows = np.stack([values[i - context_length : i] for i in batch_idx], axis=0)
            x = torch.as_tensor(windows, dtype=torch.float32, device=device)
            pred = model(past_values=x, return_dict=True).last_hidden_state
            emb = pred.mean(dim=(1, 2)).detach().cpu().numpy()
            if emb.shape[1] < emb_dim:
                raise ValueError(f"embedding dim {emb.shape[1]} < requested {emb_dim}")
            out.loc[batch_idx, emb_cols] = emb[:, :emb_dim].astype("float32")
    out[emb_cols] = out[emb_cols].ffill().fillna(0.0)
    return pd.concat([core, out], axis=1)


def _future_extreme(s: pd.Series, horizon: int, mode: str) -> pd.Series:
    future = s.shift(-1)
    if mode == "max":
        return future[::-1].rolling(horizon, min_periods=1).max()[::-1]
    if mode == "min":
        return future[::-1].rolling(horizon, min_periods=1).min()[::-1]
    raise ValueError(mode)


def _labels(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    close = _num(frame, "close").ffill().bfill().clip(lower=1e-12)
    high = _num(frame, "high").ffill().bfill()
    low = _num(frame, "low").ffill().bfill()
    prev_close = close.shift(1).fillna(close)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_pct = (tr.rolling(14, min_periods=3).mean() / close).fillna(0.001)
    floor = np.maximum(0.0012, np.maximum(0.0011, atr_pct.to_numpy(dtype=np.float64) * 0.22))

    fut_high = _future_extreme(high, horizon, "max")
    fut_low = _future_extreme(low, horizon, "min")
    long_mfe = (fut_high / close - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    long_mae = (1.0 - fut_low / close).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float64)
    short_mfe = long_mae.copy()
    short_mae = long_mfe.copy()
    long_score = long_mfe - 0.55 * long_mae - 0.00055
    short_score = short_mfe - 0.55 * short_mae - 0.00055

    y = np.zeros(len(frame), dtype=np.int64)
    margin = 0.00035
    y[(short_score - long_score > margin) & (short_score > floor)] = 1
    y[(long_score - short_score > margin) & (long_score > floor)] = 2
    valid = np.ones(len(frame), dtype=bool)
    valid[-horizon:] = False
    return pd.DataFrame({"label": y, "valid": valid.astype(np.int8)}, index=frame.index)


def _fit_values(frame: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    out: dict[str, float] = {}
    for col in cols:
        x = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        val = float(x.median()) if x.notna().any() else 0.0
        out[col] = val if math.isfinite(val) else 0.0
    return out


def _matrix(frame: pd.DataFrame, cols: list[str], fill: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            col: pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(fill[col])
            for col in cols
        },
        index=frame.index,
    )


def _class_weights(y: np.ndarray, power: float = 0.5) -> list[float]:
    counts = np.maximum(np.bincount(y.astype(int), minlength=3).astype(float), 1.0)
    weights = (counts.sum() / (3.0 * counts)) ** float(power)
    return [float(v) for v in weights]


def _entropy(p: np.ndarray) -> np.ndarray:
    q = np.clip(p, 1e-8, 1.0)
    return -(q * np.log(q)).sum(axis=1) / np.log(float(q.shape[1]))


def _train_and_score(
    train: pd.DataFrame,
    score: pd.DataFrame,
    train_x: pd.DataFrame,
    score_x: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cols = list(train_x.columns)
    fill = _fit_values(train_x, cols)
    x_score = _matrix(score_x, cols, fill)
    out = pd.DataFrame(index=score.index)
    metrics: dict[str, Any] = {"feature_cols": cols, "heads": {}}
    model_dir = args.out_dir / "heads"
    model_dir.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        lab = _labels(train, horizon)
        data = pd.concat([train_x.reset_index(drop=True), lab.reset_index(drop=True)], axis=1)
        data = data[data["valid"] > 0].reset_index(drop=True)
        split = int(len(data) * 0.82)
        fit_df = data.iloc[:split].reset_index(drop=True)
        hold_df = data.iloc[split:].reset_index(drop=True)
        x_fit = _matrix(fit_df, cols, fill)
        y_fit = fit_df["label"].to_numpy(dtype=np.int64)
        x_hold = _matrix(hold_df, cols, fill)
        y_hold = hold_df["label"].to_numpy(dtype=np.int64)

        model = CatBoostClassifier(
            loss_function="MultiClass",
            eval_metric="TotalF1",
            iterations=int(args.iterations),
            learning_rate=float(args.learning_rate),
            depth=int(args.depth),
            l2_leaf_reg=float(args.l2_leaf_reg),
            random_seed=int(args.random_seed) + int(horizon),
            task_type=str(args.task_type),
            class_weights=_class_weights(y_fit, float(args.class_weight_power)),
            od_type="Iter",
            od_wait=80,
            verbose=False,
            allow_writing_files=False,
        )
        model.fit(Pool(x_fit, y_fit), eval_set=Pool(x_hold, y_hold), use_best_model=True)
        hold_p = np.asarray(model.predict_proba(x_hold), dtype=np.float64)
        score_p = np.asarray(model.predict_proba(x_score), dtype=np.float64)
        prefix = f"ai_patch_h{horizon}"
        out[f"{prefix}_p_flat"] = score_p[:, 0]
        out[f"{prefix}_p_down"] = score_p[:, 1]
        out[f"{prefix}_p_up"] = score_p[:, 2]
        out[f"{prefix}_edge"] = score_p[:, 2] - score_p[:, 1]
        out[f"{prefix}_conf"] = np.maximum(score_p[:, 1], score_p[:, 2]) - score_p[:, 0]
        out[f"{prefix}_entropy"] = _entropy(score_p)
        model.save_model(model_dir / f"{prefix}.cbm")
        head_metrics: dict[str, Any] = {
            "fit_rows": int(len(fit_df)),
            "hold_rows": int(len(hold_df)),
            "label_counts": {str(k): int(v) for k, v in zip(*np.unique(lab["label"].to_numpy(), return_counts=True))},
            "best_iteration": int(model.get_best_iteration() or 0),
            "hold_balanced_accuracy": float(balanced_accuracy_score(y_hold, np.argmax(hold_p, axis=1))),
        }
        try:
            head_metrics["hold_ovr_auc"] = float(roc_auc_score(y_hold, hold_p, multi_class="ovr"))
        except Exception:
            head_metrics["hold_ovr_auc"] = None
        metrics["heads"][f"h{horizon}"] = head_metrics

    edges = out[[f"ai_patch_h{h}_edge" for h in HORIZONS]].to_numpy(dtype=np.float64)
    ent = out[[f"ai_patch_h{h}_entropy" for h in HORIZONS]].to_numpy(dtype=np.float64)
    out["ai_patch_consensus"] = np.abs(np.sign(edges).sum(axis=1)) / float(len(HORIZONS))
    out["ai_patch_edge_mean"] = edges.mean(axis=1)
    out["ai_patch_risk_adj_edge"] = out["ai_patch_edge_mean"] * (1.0 - ent.mean(axis=1))
    return out.astype("float32"), metrics


def main() -> int:
    args = parse_args()
    if args.startup_check_only:
        print("startup check ok: build_ai_patchmix_direction_core_20260530")
        return 0

    global CORE_FEATURES, HORIZONS
    HORIZONS = tuple(int(x.strip()) for x in str(args.horizons).split(",") if x.strip())
    if not HORIZONS or any(h <= 0 for h in HORIZONS):
        raise ValueError(f"invalid horizons: {args.horizons}")

    if args.input_profile == "audit_compact":
        CORE_FEATURES = (*BASE_CORE_FEATURES, *AUDITED_COMPACT_FEATURES)
    elif args.input_profile == "audit_compact_local_regime":
        CORE_FEATURES = (*BASE_CORE_FEATURES, *AUDITED_COMPACT_FEATURES, *LOCAL_REGIME_FEATURES)
    elif args.input_profile == "audit_compact_regime3_risk":
        CORE_FEATURES = (
            *BASE_CORE_FEATURES,
            *AUDITED_COMPACT_FEATURES,
            *LOCAL_REGIME_FEATURES,
            *REGIME3_RISK_FEATURES,
        )
    else:
        CORE_FEATURES = (*BASE_CORE_FEATURES, *AUDITED_FULL_FEATURES)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() and args.task_type == "GPU" else "cpu")
    train = _read_frame(args.train_csv, int(args.limit))
    score = _read_frame(args.score_csv, int(args.limit))
    if args.input_profile == "audit_compact_regime3_risk":
        train = _merge_regime3_risk_sidecar(train, args.train_regime3_risk_csv)
        score = _merge_regime3_risk_sidecar(score, args.score_regime3_risk_csv)
    train_x = _patch_embeddings(
        train,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )
    score_x = _patch_embeddings(
        score,
        model_id=str(args.patch_model_id),
        context_length=int(args.context_length),
        stride=int(args.stride),
        batch_size=int(args.batch_size),
        emb_dim=int(args.emb_dim),
        device=device,
    )
    features, metrics = _train_and_score(train, score, train_x, score_x, args)
    out = pd.concat([score[["timestamp"]].reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    ai_cols = [c for c in out.columns if c.startswith("ai_patch_")]
    if out[ai_cols].replace([np.inf, -np.inf], np.nan).isna().any().any():
        bad = out[ai_cols].columns[out[ai_cols].replace([np.inf, -np.inf], np.nan).isna().any()].tolist()
        raise RuntimeError(f"non-finite output columns: {bad}")
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    manifest = {
        "model_id": MODEL_ID,
        "train_csv": str(args.train_csv),
        "score_csv": str(args.score_csv),
        "out_csv": str(args.out_csv),
        "patch_model_id": str(args.patch_model_id),
        "input_profile": str(args.input_profile),
        "hf_offline": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            "local_files_only": True,
        },
        "horizons": list(HORIZONS),
        "context_length": int(args.context_length),
        "stride": int(args.stride),
        "class_weight_power": float(args.class_weight_power),
        "core_features": list(CORE_FEATURES),
        "patch_channels": list(PATCH_CHANNELS),
        "output_columns": ai_cols,
        "metrics": metrics,
    }
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps({"out_csv": str(args.out_csv), "rows": len(out), "features": len(ai_cols)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
