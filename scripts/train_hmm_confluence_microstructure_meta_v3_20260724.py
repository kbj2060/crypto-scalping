#!/usr/bin/env python3
"""Walk-forward microstructure meta-filter for ETH HMM pullback labels.

All model/quantile selection uses chronological folds inside 2025. The 2026
files are opened only after the final 2025 policy has been fitted and locked.
Because earlier experiments already consumed 2026, its output is development
diagnostic only and cannot restore a formal OOS claim.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_hmm_confluence_meta_labels_20260724 as v1  # noqa: E402
import scripts.build_hmm_confluence_meta_labels_v2_20260724 as v2  # noqa: E402
import scripts.train_hmm_confluence_meta_filter_v2_20260724 as meta_v2  # noqa: E402


MODEL_ID = "eth_hmm_confluence_microstructure_meta_v3_20260724"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516" / v2.MODEL_ID
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MARKET_FEATURES = [
    "sum_toptrader_long_short_ratio",
    "count_long_short_ratio",
    "net_taker_ratio",
    "taker_acceleration",
    "volatility_z",
    "bb_width_z",
    "wick_ratio",
    "amihud_illiquidity_z",
    "dual_momentum",
    "breakout_strength",
    "volume_profile_signal",
    "funding_price_divergence",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_288",
    "cvd_slope_12",
    "cvd_slope_48",
    "price_cvd_divergence",
    "cvd_breakout_z",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "btc_breakout_eth_lag_dir",
    "btc_volume_impulse_z",
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
    "funding_oi_divergence",
    "funding_flip_signal",
    "oi_up_price_down",
    "oi_up_price_up",
    "sweep_prev_high_reclaim",
    "sweep_prev_low_reclaim",
    "failed_breakout_up",
    "failed_breakout_down",
    "liquidity_vacuum",
]

DIRECTIONAL_FEATURES = [
    "net_taker_ratio",
    "taker_acceleration",
    "dual_momentum",
    "breakout_strength",
    "volume_profile_signal",
    "funding_price_divergence",
    "funding_pressure",
    "cvd_12",
    "cvd_48",
    "cvd_288",
    "cvd_slope_12",
    "cvd_slope_48",
    "price_cvd_divergence",
    "cvd_breakout_z",
    "eth_btc_ret_spread_12",
    "eth_btc_ret_spread_48",
    "btc_breakout_eth_lag_dir",
    "compression_release_up",
    "compression_release_down",
    "range_contraction_breakout_dir",
    "vwap_dist_24",
    "vwap_dist_96",
    "vwap_dist_288",
    "anchored_vwap_session_dist",
    "vwap_reclaim_flag",
    "vwap_reject_flag",
    "funding_oi_divergence",
    "funding_flip_signal",
    "oi_up_price_down",
    "oi_up_price_up",
    "liquidity_vacuum",
]

FOLDS = [
    ("2025-05-01", "2025-06-30 23:55:00"),
    ("2025-07-01", "2025-08-31 23:55:00"),
    ("2025-09-01", "2025-10-31 23:55:00"),
    ("2025-11-01", "2025-12-31 23:55:00"),
]


def enrich(labels: pd.DataFrame, market_path: Path) -> pd.DataFrame:
    market = pd.read_csv(market_path, usecols=["timestamp", *MARKET_FEATURES], parse_dates=["timestamp"], low_memory=False)
    index = labels["decision_index"].to_numpy(int)
    if index.min(initial=0) < 0 or index.max(initial=0) >= len(market):
        raise RuntimeError("decision index is outside its source market frame")
    expected = pd.to_datetime(labels["decision_timestamp"]).reset_index(drop=True)
    actual = market["timestamp"].iloc[index].reset_index(drop=True)
    if not expected.equals(actual):
        raise RuntimeError("decision index/timestamp contract mismatch during market enrichment")
    out = labels.reset_index(drop=True).copy()
    selected_market = market.iloc[index].reset_index(drop=True)
    for column in MARKET_FEATURES:
        out[f"micro_{column}"] = pd.to_numeric(selected_market[column], errors="coerce").to_numpy(float)
    return out


def feature_frame(labels: pd.DataFrame) -> pd.DataFrame:
    base = meta_v2.feature_frame(labels).copy()
    side = labels["candidate_side"].to_numpy(float)
    for column in MARKET_FEATURES:
        source = labels[f"micro_{column}"].replace([np.inf, -np.inf], np.nan).astype(float)
        if column in DIRECTIONAL_FEATURES:
            base[f"aligned_{column}"] = side * source.to_numpy(float)
        else:
            base[f"micro_{column}"] = source.to_numpy(float)
    return base


def _model(alpha: float) -> Any:
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha))


def _fold_metrics(labels: pd.DataFrame, score: np.ndarray, cutoff: float) -> dict[str, Any]:
    selected = labels.loc[score >= cutoff].copy()
    trades = v1.replay_non_overlapping(selected)
    returns = trades["label_net_return_per_notional"].to_numpy(float) if len(trades) else np.empty(0)
    return {
        "selected_labels": int(len(selected)),
        "mean_net_return": float(selected["label_net_return_per_notional"].mean()) if len(selected) else -9.0,
        "policy_trades": int(len(trades)),
        "policy_compounded_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else -9.0,
    }


def walk_forward_select(labels_2025: pd.DataFrame) -> tuple[float, float, list[dict[str, Any]]]:
    timestamp = pd.to_datetime(labels_2025["decision_timestamp"])
    results: list[dict[str, Any]] = []
    for alpha in (10.0, 30.0, 100.0, 300.0):
        for quantile in (0.40, 0.50, 0.60, 0.70):
            folds: list[dict[str, Any]] = []
            for validation_start, validation_end in FOLDS:
                train_mask = timestamp < pd.Timestamp(validation_start)
                validation_mask = timestamp.between(validation_start, validation_end)
                train = labels_2025.loc[train_mask].reset_index(drop=True)
                validation = labels_2025.loc[validation_mask].reset_index(drop=True)
                if len(train) < 80 or len(validation) < 20:
                    raise RuntimeError(f"insufficient walk-forward rows for fold {validation_start}")
                model = _model(alpha)
                model.fit(feature_frame(train), train["label_net_r"].clip(-2.0, 2.0))
                train_score = np.asarray(model.predict(feature_frame(train)), dtype=float)
                validation_score = np.asarray(model.predict(feature_frame(validation)), dtype=float)
                cutoff = float(np.quantile(train_score, quantile))
                metrics = _fold_metrics(validation, validation_score, cutoff)
                folds.append({"start": validation_start, "end": validation_end, "cutoff": cutoff, **metrics})
            fold_means = np.asarray([fold["mean_net_return"] for fold in folds], dtype=float)
            fold_returns = np.asarray([fold["policy_compounded_return"] for fold in folds], dtype=float)
            selected_total = sum(fold["selected_labels"] for fold in folds)
            positive_folds = int(((fold_means > 0.0) & (fold_returns > 0.0)).sum())
            eligible = positive_folds >= 3 and selected_total >= 50 and float(fold_means.mean()) > 0.0
            results.append(
                {
                    "alpha": alpha,
                    "score_quantile": quantile,
                    "folds": folds,
                    "positive_folds": positive_folds,
                    "selected_total": selected_total,
                    "eligible": eligible,
                    "selection_score": float(np.quantile(fold_means, 0.25)) if eligible else -9.0,
                }
            )
    eligible_results = [row for row in results if row["eligible"]]
    if not eligible_results:
        raise RuntimeError("no microstructure meta-filter passed the 2025 walk-forward contract")
    winner = max(eligible_results, key=lambda row: (row["selection_score"], row["selected_total"]))
    return float(winner["alpha"]), float(winner["score_quantile"]), results


def apply(labels: pd.DataFrame, model: Any, cutoff: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = labels.copy()
    out["meta_score"] = np.asarray(model.predict(feature_frame(out)), dtype=float)
    out["meta_selected"] = (out["meta_score"] >= cutoff).astype(np.int8)
    selected = out.loc[out["meta_selected"] == 1].copy()
    trades = v1.replay_non_overlapping(selected)
    returns = trades["label_net_return_per_notional"].to_numpy(float) if len(trades) else np.empty(0)
    summary = {
        "candidate_labels": int(len(out)),
        "selected_labels": int(len(selected)),
        "mean_net_return": float(selected["label_net_return_per_notional"].mean()) if len(selected) else -9.0,
        "success_rate": float(selected["label_success"].mean()) if len(selected) else 0.0,
        "policy_trades": int(len(trades)),
        "policy_win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
        "policy_compounded_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else -9.0,
    }
    return out, trades, summary


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(LABEL_DIR / "train_meta_labels.parquet").query("label_valid == 1")
    validation = pd.read_parquet(LABEL_DIR / "validation_meta_labels.parquet").query("label_valid == 1")
    labels_2025 = enrich(pd.concat([train, validation], ignore_index=True).sort_values("decision_timestamp"), v1.MARKET_2025)
    alpha, quantile, search = walk_forward_select(labels_2025)
    final_model = _model(alpha)
    final_model.fit(feature_frame(labels_2025), labels_2025["label_net_r"].clip(-2.0, 2.0))
    score_2025 = np.asarray(final_model.predict(feature_frame(labels_2025)), dtype=float)
    cutoff = float(np.quantile(score_2025, quantile))
    print(
        json.dumps(
            {
                "stage": "v3_policy_locked_before_2026_development_diagnostic",
                "alpha": alpha,
                "score_quantile": quantile,
                "score_cutoff": cutoff,
            }
        ),
        flush=True,
    )

    oos = enrich(pd.read_parquet(LABEL_DIR / "oos_meta_labels.parquet").query("label_valid == 1"), v1.MARKET_2026)
    fresh = enrich(pd.read_parquet(LABEL_DIR / "fresh_meta_labels.parquet").query("label_valid == 1"), v1.MARKET_2026)
    summaries: dict[str, Any] = {}
    for split, labels in (("development_2025", labels_2025), ("consumed_2026_oos", oos), ("consumed_2026_fresh", fresh)):
        scored, trades, summary = apply(labels, final_model, cutoff)
        scored.to_parquet(OUT_DIR / f"{split}_scored_labels.parquet", index=False)
        trades.to_csv(OUT_DIR / f"{split}_diagnostic_trades.csv", index=False)
        summaries[split] = summary

    artifact_path = OUT_DIR / "microstructure_meta_filter.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "model": final_model,
            "alpha": alpha,
            "score_quantile": quantile,
            "score_cutoff": cutoff,
            "market_features": MARKET_FEATURES,
            "directional_features": DIRECTIONAL_FEATURES,
            "label_model_id": v2.MODEL_ID,
        },
        artifact_path,
    )
    search_path = OUT_DIR / "walk_forward_search_2025.json"
    search_path.write_text(json.dumps(search, indent=2, default=v1._json_default), encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "status": "development_diagnostic_only_2026_already_consumed",
        "alpha": alpha,
        "score_quantile": quantile,
        "score_cutoff": cutoff,
        "selection": {
            "walk_forward_year": 2025,
            "folds": FOLDS,
            "search_results": str(search_path),
            "2026_used_for_selection": False,
        },
        "summaries": summaries,
        "artifact": {"path": str(artifact_path), "sha256": v1.sha256(artifact_path)},
        "promotion_eligible": False,
        "promotion_blocker": "2026 periods were consumed by prior V2 experiments; a new untouched forward period is required",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=v1._json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summaries": summaries}, default=v1._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
