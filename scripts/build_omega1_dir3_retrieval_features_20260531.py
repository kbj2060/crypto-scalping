#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_dir3_retrieval_20260531"
DEFAULT_SPLIT_DIR = ROOT / "data/splits/year_oos"
DEFAULT_LABEL_DIR = ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_20260531"
DEFAULT_REGIME3_CURRENT_DIR = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530"
DEFAULT_REGIME3_RISK_DIR = ROOT / "data/ensemble/supervised/regime3_stability_risk_h6_20260530"
DEFAULT_REGIME3_CMAMBA_DIR = ROOT / "data/ensemble/supervised/regime3_cryptomamba_pred_h6_nocurrent_20260531"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_retrieval_20260531"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_retrieval_20260531"

CLASS_NAMES = ["cash", "long", "short"]
OUTPUT_COLS = [
    "dir3_retrieval_h6_fl_prob",
    "dir3_retrieval_h6_up_prob",
    "dir3_retrieval_h6_dn_prob",
    "dir3_retrieval_h6_confidence",
    "dir3_retrieval_h6_side_edge",
    "dir3_retrieval_h6_trade_prob",
    "dir3_retrieval_h6_neighbor_edge_mean",
    "dir3_retrieval_h6_neighbor_edge_q25",
    "dir3_retrieval_h6_neighbor_edge_q75",
    "dir3_retrieval_h6_regime_consensus",
    "dir3_retrieval_h6_similarity_score",
]

BASE_FEATURES = [
    "log_return",
    "volatility_z",
    "rsi",
    "macd_hist",
    "bb_width_z",
    "hma_slope",
    "wick_ratio",
    "garman_klass_vol",
    "realized_vol_ratio",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "rogers_satchell_vol",
    "parkinson_vol",
    "amihud_illiquidity_z",
    "btc_corr_60",
    "eth_btc_ratio_change",
    "chop_index",
    "hour_sin",
    "hour_cos",
    "minute_sin",
    "minute_cos",
    "session_europe",
    "session_us",
    "is_hour_open",
    "cvp_poc_dist",
    "cvp_vah_val_width",
    "cvp_cluster_position",
    "cvp_volume_imbalance",
    "cvp_regime",
    "turtle_signal",
    "dual_momentum",
    "mean_reversion_z",
    "breakout_strength",
    "volume_profile_signal",
    "fibonacci_level",
    "funding_roc_12",
    "funding_roc_48",
    "funding_roc_288",
    "funding_z_score",
    "long_squeeze_risk",
    "short_squeeze_risk",
    "funding_price_divergence",
    "hurst_48",
    "hurst_288",
    "regime_trending",
    "ofi_acceleration",
    "kalman_velocity",
    "realized_skewness",
    "ofti",
    "kel",
    "mta_funding",
    "svps",
    "funding_abs",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_288",
    "cvd_slope_12",
    "cvd_slope_48",
    "price_cvd_divergence",
    "cvd_breakout_z",
    "btc_ret_1",
    "btc_ret_3",
    "btc_ret_6",
    "btc_ret_12",
    "btc_ret_z_48",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "eth_btc_beta_residual_z",
    "btc_lead_eth_follow_gap_3",
    "btc_breakout_eth_lag_dir",
    "btc_volume_impulse_z",
    "btc_eth_volume_rank_spread",
    "btc_impulse_x_eth_beta",
    "bb_width_pct_rank_288",
    "atr_pct_rank_288",
    "compression_score",
    "compression_release_up",
    "compression_release_down",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "vwap_dist_96",
    "vwap_dist_288",
    "anchored_vwap_session_dist",
    "vwap_reclaim_flag",
    "vwap_reject_flag",
    "distance_to_day_high_low_pct",
    "funding_oi_divergence",
    "funding_flip_signal",
    "oi_up_price_down",
    "oi_up_price_up",
    "crowded_long_unwind_risk",
    "crowded_short_squeeze_risk",
    "upper_wick_z",
    "lower_wick_z",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "garch_vol_z",
    "ou_funding_z",
    "ou_halflife",
    "jump_flag",
    "jump_z",
    "evt_tail_flag",
    "evt_excess_z",
    "sig_volume_confirm",
    "sig_liquidity_trap",
    "sig_trend_health",
    "regime_persistence",
    "cross_scale_curvature",
    "liquidity_vacuum",
    "crowding_pressure",
    "execution_quality",
]

REGIME3_CURRENT_FEATURES = [
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
    "regime3_current_sensitive_wide24_confidence",
    "regime3_current_sensitive_wide24_entropy",
    "regime3_current_sensitive_wide24_margin",
]

REGIME3_H6_CONTEXT_FEATURES = [
    "regime3_stability_h6_score",
    "regime3_transition_h6_risk_prob",
    "regime3_transition_h6_risk_pred",
    "regime3_churn_h6_risk_score",
    "regime3_cmamba_h6_future_bull_prob",
    "regime3_cmamba_h6_future_bear_prob",
    "regime3_cmamba_h6_future_chop_prob",
    "regime3_cmamba_h6_confidence",
    "regime3_cmamba_h6_transition_prob",
    "regime3_cmamba_h6_stability_score",
]

FORBIDDEN_PREFIXES = ("teacher_", "a5dir_", "clean_regime_", "clean_regime4_", "regime4_pred_", "regime3_pred_")
FORBIDDEN_SUBSTRINGS = ("label", "target", "pnl", "action_score")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)


def _split_file(split_dir: Path, year: int) -> Path:
    return split_dir / ("training_features_2026_rebuilt.csv" if int(year) == 2026 else f"training_features_{int(year)}.csv")


def _read_base(split_dir: Path, year: int) -> pd.DataFrame:
    path = _split_file(split_dir, year)
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{path} missing timestamp")
    return frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _exact_join(
    left: pd.DataFrame,
    right: pd.DataFrame,
    cols: list[str],
    source: str,
    *,
    allow_tail_drop: bool = False,
    allow_edge_drop: bool = False,
) -> pd.DataFrame:
    before = len(left)
    merged = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(merged) != before:
        raise RuntimeError(f"{source} changed row count: {before} -> {len(merged)}")
    miss = merged[cols].isna().any(axis=1).to_numpy()
    if miss.any():
        idx = np.flatnonzero(miss)
        tail_only = np.array_equal(idx, np.arange(len(merged) - idx.size, len(merged)))
        if allow_tail_drop and tail_only:
            return merged.iloc[: len(merged) - idx.size].reset_index(drop=True)
        head_only = np.array_equal(idx, np.arange(0, idx.size))
        if allow_edge_drop and head_only:
            return merged.iloc[idx.size :].reset_index(drop=True)
        if allow_edge_drop and tail_only:
            return merged.iloc[: len(merged) - idx.size].reset_index(drop=True)
        missing = {c: int(merged[c].isna().sum()) for c in cols if int(merged[c].isna().sum())}
        raise RuntimeError(f"{source} exact join missing values: {missing}")
    return merged


def _add_label(frame: pd.DataFrame, label_dir: Path, year: int) -> pd.DataFrame:
    path = label_dir / f"zigzag_action_labels_{int(year)}.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    labels = pd.read_csv(path, parse_dates=["timestamp"])
    required = ["zigzag_action", "zigzag_path_edge"]
    missing = sorted(set(required) - set(labels.columns))
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return _exact_join(frame, labels, required, f"ZigZag labels {year}")


def _add_regime3_current(frame: pd.DataFrame, regime_dir: Path, year: int) -> pd.DataFrame:
    name = "training_features_2026_rebuilt_regime3_current_sensitive_hmm_wide24.csv" if int(year) == 2026 else f"training_features_{int(year)}_regime3_current_sensitive_hmm_wide24.csv"
    path = regime_dir / name
    if not path.exists():
        raise FileNotFoundError(path)
    side = pd.read_csv(path, parse_dates=["timestamp"])
    missing = sorted(set(REGIME3_CURRENT_FEATURES) - set(side.columns))
    if missing:
        raise ValueError(f"{path} missing {missing}")
    return _exact_join(frame, side, REGIME3_CURRENT_FEATURES, f"Regime3 current {year}", allow_tail_drop=True)


def _add_regime3_h6(frame: pd.DataFrame, risk_dir: Path, cmamba_dir: Path, year: int) -> pd.DataFrame:
    risk_name = "training_features_2026_rebuilt_regime3_stability_risk_h6.csv" if int(year) == 2026 else f"training_features_{int(year)}_regime3_stability_risk_h6.csv"
    cm_name = "training_features_2026_rebuilt_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv" if int(year) == 2026 else f"training_features_{int(year)}_regime3_cryptomamba_pred_h6_nocurrent_20260531.csv"
    risk = pd.read_csv(risk_dir / risk_name, parse_dates=["timestamp"])
    cm = pd.read_csv(cmamba_dir / cm_name, parse_dates=["timestamp"])
    risk_cols = REGIME3_H6_CONTEXT_FEATURES[:4]
    cm_cols = REGIME3_H6_CONTEXT_FEATURES[4:]
    missing_risk = sorted(set(risk_cols) - set(risk.columns))
    missing_cm = sorted(set(cm_cols) - set(cm.columns))
    if missing_risk:
        raise ValueError(f"{risk_dir / risk_name} missing {missing_risk}")
    if missing_cm:
        raise ValueError(f"{cmamba_dir / cm_name} missing {missing_cm}")
    out = _exact_join(frame, risk, risk_cols, f"Regime3 h6 risk {year}", allow_tail_drop=True)
    return _exact_join(out, cm, cm_cols, f"Regime3 CryptoMamba h6 {year}", allow_tail_drop=True, allow_edge_drop=True)


def _check_feature_contract(cols: list[str]) -> None:
    bad: list[str] = []
    for col in cols:
        lower = col.lower()
        if col.startswith(FORBIDDEN_PREFIXES):
            bad.append(col)
        elif any(token in lower for token in FORBIDDEN_SUBSTRINGS):
            bad.append(col)
        elif "future" in lower and col not in REGIME3_H6_CONTEXT_FEATURES:
            bad.append(col)
    if bad:
        raise ValueError(f"forbidden dir3 retrieval inputs selected: {bad[:30]}")


def _feature_cols(frame: pd.DataFrame, variant: str) -> list[str]:
    cols = [c for c in BASE_FEATURES if c in frame.columns]
    if variant in {"base_regime3_current", "base_regime3_current_h6"}:
        cols.extend(REGIME3_CURRENT_FEATURES)
    if variant == "base_regime3_current_h6":
        cols.extend(REGIME3_H6_CONTEXT_FEATURES)
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise ValueError(f"{variant} missing inputs: {missing}")
    _check_feature_contract(cols)
    for col in cols:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise TypeError(f"dir3 retrieval input must be numeric: {col}")
    return cols


def _make_pipeline(n_components: int, seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler(quantile_range=(5.0, 95.0), unit_variance=True)),
            ("pca", PCA(n_components=int(n_components), random_state=int(seed), whiten=True)),
        ]
    )


def _prob_from_neighbors(labels: np.ndarray, edges: np.ndarray, distances: np.ndarray, indices: np.ndarray, weighted: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    neigh_labels = labels[indices]
    neigh_edges = edges[indices]
    if weighted:
        weights = 1.0 / (distances + 1e-6)
    else:
        weights = np.ones_like(distances)
    denom = weights.sum(axis=1, keepdims=True)
    probs = np.zeros((indices.shape[0], 3), dtype=np.float64)
    for cls in range(3):
        probs[:, cls] = ((neigh_labels == cls) * weights).sum(axis=1) / np.maximum(denom[:, 0], 1e-12)
    edge_mean = (neigh_edges * weights).sum(axis=1) / np.maximum(denom[:, 0], 1e-12)
    edge_q25 = np.quantile(neigh_edges, 0.25, axis=1)
    edge_q75 = np.quantile(neigh_edges, 0.75, axis=1)
    similarity = 1.0 / (1.0 + distances.mean(axis=1))
    consensus = probs.max(axis=1)
    return probs, edge_mean, edge_q25, edge_q75, similarity * consensus


def _score_frame(
    train: pd.DataFrame,
    target: pd.DataFrame,
    cols: list[str],
    *,
    k: int,
    n_components: int,
    weighted: bool,
    seed: int,
    max_train_samples: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    fit_train = train
    if int(max_train_samples) > 0 and len(train) > int(max_train_samples):
        rng = np.random.default_rng(int(seed))
        parts: list[pd.DataFrame] = []
        labels_for_sample = pd.to_numeric(train["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
        per_class = max(1, int(max_train_samples) // 3)
        for cls in [0, 1, 2]:
            idx = np.flatnonzero(labels_for_sample == cls)
            if idx.size:
                take = min(per_class, idx.size)
                parts.append(train.iloc[np.sort(rng.choice(idx, size=take, replace=False))])
        fit_train = pd.concat(parts, axis=0).sort_values("timestamp").reset_index(drop=True)
    pipeline = _make_pipeline(n_components, seed)
    x_train = pipeline.fit_transform(fit_train[cols])
    x_target = pipeline.transform(target[cols])
    nn = NearestNeighbors(n_neighbors=int(k), algorithm="auto", metric="euclidean", n_jobs=-1)
    nn.fit(x_train)
    distances, indices = nn.kneighbors(x_target, return_distance=True)
    labels = pd.to_numeric(fit_train["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    edges = pd.to_numeric(fit_train["zigzag_path_edge"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    probs, edge_mean, edge_q25, edge_q75, sim_cons = _prob_from_neighbors(labels, edges, distances, indices, weighted)
    out = pd.DataFrame(
        {
            "timestamp": target["timestamp"].to_numpy(),
            "dir3_retrieval_h6_fl_prob": probs[:, 0],
            "dir3_retrieval_h6_up_prob": probs[:, 1],
            "dir3_retrieval_h6_dn_prob": probs[:, 2],
            "dir3_retrieval_h6_confidence": probs.max(axis=1),
            "dir3_retrieval_h6_side_edge": probs[:, 1] - probs[:, 2],
            "dir3_retrieval_h6_trade_prob": probs[:, 1] + probs[:, 2],
            "dir3_retrieval_h6_neighbor_edge_mean": edge_mean,
            "dir3_retrieval_h6_neighbor_edge_q25": edge_q25,
            "dir3_retrieval_h6_neighbor_edge_q75": edge_q75,
            "dir3_retrieval_h6_regime_consensus": probs.max(axis=1),
            "dir3_retrieval_h6_similarity_score": sim_cons,
        }
    )
    if out[OUTPUT_COLS].isna().any().any():
        raise RuntimeError("dir3 retrieval output contains NaN")
    meta = {
        "feature_cols": cols,
        "pca_explained_variance_sum": float(pipeline.named_steps["pca"].explained_variance_ratio_.sum()),
        "k": int(k),
        "n_components": int(n_components),
        "weighted": bool(weighted),
        "fit_train_rows": int(len(fit_train)),
    }
    return out, {"pipeline": pipeline, "nearest_neighbors": nn, "meta": meta}


def _metrics(scored: pd.DataFrame, labels: pd.DataFrame) -> dict[str, Any]:
    df = scored.merge(labels[["timestamp", "zigzag_action"]], on="timestamp", how="inner", validate="one_to_one")
    if len(df) != len(scored):
        raise RuntimeError("metrics label join changed row count")
    y = pd.to_numeric(df["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    proba = df[["dir3_retrieval_h6_fl_prob", "dir3_retrieval_h6_up_prob", "dir3_retrieval_h6_dn_prob"]].to_numpy(float)
    pred = proba.argmax(axis=1)
    trade_mask = pred != 0
    trade_count = int(trade_mask.sum())
    long_trades = int((pred == 1).sum())
    short_trades = int((pred == 2).sum())
    proxy_wr = float((pred[trade_mask] == y[trade_mask]).mean()) if trade_count else None
    out = {
        "rows": int(len(df)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
        "proxy_trades": trade_count,
        "proxy_long_trades": long_trades,
        "proxy_short_trades": short_trades,
        "proxy_trade_rate": float(trade_count / len(df)) if len(df) else None,
        "proxy_wr": proxy_wr,
        "mean_confidence": float(proba.max(axis=1).mean()),
        "mean_trade_prob": float((proba[:, 1] + proba[:, 2]).mean()),
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(y, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    parser.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--regime3-risk-dir", type=Path, default=DEFAULT_REGIME3_RISK_DIR)
    parser.add_argument("--regime3-cmamba-dir", type=Path, default=DEFAULT_REGIME3_CMAMBA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--variants", type=str, default="base_only,base_regime3_current,base_regime3_current_h6")
    parser.add_argument("--pca-components", type=str, default="32")
    parser.add_argument("--neighbors", type=str, default="64,128")
    parser.add_argument("--weights", type=str, default="uniform,distance")
    parser.add_argument("--max-train-samples", type=int, default=30000)
    args = parser.parse_args()

    frames: dict[int, pd.DataFrame] = {}
    for year in [2024, 2025, 2026]:
        frame = _read_base(args.split_dir, year)
        frame = _add_label(frame, args.label_dir, year)
        frame = _add_regime3_current(frame, args.regime3_current_dir, year)
        frame = _add_regime3_h6(frame, args.regime3_risk_dir, args.regime3_cmamba_dir, year)
        frames[year] = frame

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    pca_grid = [int(v.strip()) for v in args.pca_components.split(",") if v.strip()]
    k_grid = [int(v.strip()) for v in args.neighbors.split(",") if v.strip()]
    weight_grid = [v.strip() for v in args.weights.split(",") if v.strip()]
    invalid_variants = sorted(set(variants) - {"base_only", "base_regime3_current", "base_regime3_current_h6"})
    invalid_weights = sorted(set(weight_grid) - {"uniform", "distance"})
    if invalid_variants:
        raise ValueError(f"invalid variants: {invalid_variants}")
    if invalid_weights:
        raise ValueError(f"invalid weights: {invalid_weights}")
    grid: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    best_artifacts: dict[str, Any] | None = None
    for variant in variants:
        cols = _feature_cols(frames[2024], variant)
        for n_components in pca_grid:
            if n_components >= len(cols):
                continue
            for k in k_grid:
                for weight_name in weight_grid:
                    weighted = weight_name == "distance"
                    scored_2025, artifacts = _score_frame(
                        frames[2024],
                        frames[2025],
                        cols,
                        k=k,
                        n_components=n_components,
                        weighted=weighted,
                        seed=args.seed,
                        max_train_samples=args.max_train_samples,
                    )
                    scored_2026, _ = _score_frame(
                        frames[2024],
                        frames[2026],
                        cols,
                        k=k,
                        n_components=n_components,
                        weighted=weighted,
                        seed=args.seed,
                        max_train_samples=args.max_train_samples,
                    )
                    m25 = _metrics(scored_2025, frames[2025])
                    m26 = _metrics(scored_2026, frames[2026])
                    row = {
                        "variant": variant,
                        "k": int(k),
                        "n_components": int(n_components),
                        "weighted": bool(weighted),
                        "feature_count": int(len(cols)),
                        "pca_explained_variance_sum": artifacts["meta"]["pca_explained_variance_sum"],
                        "selection_score_2025": float(m25["balanced_accuracy"] + 0.25 * (m25["ovr_auc"] or 0.0)),
                        "metrics_2025": m25,
                        "metrics_2026": m26,
                    }
                    grid.append(row)
                    if best is None or row["selection_score_2025"] > best["selection_score_2025"]:
                        best = row
                        best_artifacts = {"scored_2025": scored_2025, "scored_2026": scored_2026, "objects": artifacts}

    if best is None or best_artifacts is None:
        raise RuntimeError("no dir3 retrieval candidate was produced")

    selected = best
    selected_name = f"{MODEL_ID}_{selected['variant']}_pca{selected['n_components']}_k{selected['k']}_{'w' if selected['weighted'] else 'u'}"
    out_2025 = args.out_dir / f"training_features_2025_{MODEL_ID}.csv"
    out_2026 = args.out_dir / f"training_features_2026_rebuilt_{MODEL_ID}.csv"
    best_artifacts["scored_2025"].to_csv(out_2025, index=False)
    best_artifacts["scored_2026"].to_csv(out_2026, index=False)
    model_path = args.out_dir / f"{selected_name}.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "selected": selected,
            "feature_cols": best_artifacts["objects"]["meta"]["feature_cols"],
            "objects": best_artifacts["objects"],
        },
        model_path,
    )

    grid_sorted = sorted(grid, key=lambda r: r["selection_score_2025"], reverse=True)
    audit = {
        "model_id": MODEL_ID,
        "role": "Omega1 third-stage direction feature generator, same level as teacher but generated in parallel",
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "selected": selected,
        "top20": grid_sorted[:20],
        "artifacts": {
            "features_2025": str(out_2025),
            "features_2026": str(out_2026),
            "model": str(model_path),
        },
        "outputs": OUTPUT_COLS,
        "contract": {
            "allowed_inputs": "base current-bar features plus explicit Regime3 current/h6 context by exact timestamp join",
            "forbidden_inputs": [
                "teacher_*",
                "a5dir_*",
                "Regime4",
                "regime3_pred_*",
                "label/target/future/PnL/action_score",
                "same-level dir3 outputs",
            ],
            "notes": [
                "No teacher feedback.",
                "No broad prefix sweep.",
                "Regime3 CryptoMamba columns are exact allowed prediction-sidecar exceptions.",
            ],
        },
    }
    audit_path = args.report_dir / "dir3_retrieval_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "variant": r["variant"],
                "k": r["k"],
                "n_components": r["n_components"],
                "weighted": r["weighted"],
                "feature_count": r["feature_count"],
                "selection_score_2025": r["selection_score_2025"],
                "bacc_2025": r["metrics_2025"]["balanced_accuracy"],
                "auc_2025": r["metrics_2025"]["ovr_auc"],
                "bacc_2026": r["metrics_2026"]["balanced_accuracy"],
                "auc_2026": r["metrics_2026"]["ovr_auc"],
                "macro_f1_2026": r["metrics_2026"]["macro_f1"],
                "proxy_wr_2026": r["metrics_2026"]["proxy_wr"],
                "proxy_trades_2026": r["metrics_2026"]["proxy_trades"],
                "proxy_trade_rate_2026": r["metrics_2026"]["proxy_trade_rate"],
            }
            for r in grid_sorted
        ]
    ).to_csv(args.report_dir / "dir3_retrieval_grid.csv", index=False)
    print(json.dumps({"audit": str(audit_path), "selected": selected, "artifacts": audit["artifacts"]}, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
