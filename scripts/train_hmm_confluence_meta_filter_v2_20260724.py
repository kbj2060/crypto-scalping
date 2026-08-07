#!/usr/bin/env python3
"""Train a causal meta-filter for V2 ETH HMM pullback labels.

Model family and score cutoff are selected on 2025 train/validation only. The
OOS and fresh parquet files are not opened until the policy has been locked.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.build_hmm_confluence_meta_labels_20260724 as v1  # noqa: E402
import scripts.build_hmm_confluence_meta_labels_v2_20260724 as v2  # noqa: E402


MODEL_ID = "eth_hmm_confluence_meta_filter_v2_20260724"
LABEL_DIR = ROOT / "tmp/causal_regen_20260516" / v2.MODEL_ID
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
MIN_TRAIN_SELECTED = 80
MIN_VALIDATION_SELECTED = 40

BASE_FEATURES = [
    "candidate_side",
    "planned_sl_price_move",
    "context_regime_confidence",
    "context_regime_margin",
    "context_regime_entropy",
    "context_regime_mean_probability",
    "context_regime_persistence",
    "context_transition_risk",
    "context_churn_risk",
    "context_sample_weight",
    "context_rsi",
    "context_vwma288_slope12",
    "context_atr192",
    "context_structural_clearance_r",
    "context_volume_confirm",
    "context_oi_change_rate",
    "context_volume_imbalance",
    "context_funding_z",
]


def feature_frame(labels: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(BASE_FEATURES) - set(labels.columns))
    if missing:
        raise RuntimeError(f"meta-filter input missing columns: {missing}")
    out = labels[BASE_FEATURES].replace([np.inf, -np.inf], np.nan).copy()
    out["slope_atr"] = out["context_vwma288_slope12"] / out["context_atr192"].replace(0.0, np.nan)
    out["aligned_volume"] = out["candidate_side"] * out["context_volume_confirm"]
    out["aligned_imbalance"] = out["candidate_side"] * out["context_volume_imbalance"]
    out["aligned_funding"] = -out["candidate_side"] * out["context_funding_z"]
    return out.drop(columns=["context_vwma288_slope12", "context_atr192"])


def model_grid(train: pd.DataFrame) -> list[tuple[str, Any, pd.Series]]:
    models: list[tuple[str, Any, pd.Series]] = []
    for regularization in (0.05, 0.20, 1.0):
        models.append(
            (
                f"logistic_success_c{regularization}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                    LogisticRegression(C=regularization, max_iter=2000, class_weight="balanced"),
                ),
                train["label_success"].astype(int),
            )
        )
    for alpha in (1.0, 10.0, 100.0):
        models.append(
            (
                f"ridge_net_r_a{alpha}",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge(alpha=alpha)),
                train["label_net_r"].clip(-2.0, 2.0),
            )
        )
    for leaf_size in (15, 30, 45):
        models.append(
            (
                f"hgb_net_r_leaf{leaf_size}",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    HistGradientBoostingRegressor(
                        max_iter=120,
                        max_leaf_nodes=7,
                        min_samples_leaf=leaf_size,
                        l2_regularization=1.0,
                        learning_rate=0.04,
                        random_state=724,
                    ),
                ),
                train["label_net_r"].clip(-2.0, 2.0),
            )
        )
    return models


def score_model(model: Any, features: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1].astype(float)
    return np.asarray(model.predict(features), dtype=float)


def selected_metrics(labels: pd.DataFrame, score: np.ndarray, cutoff: float) -> dict[str, Any]:
    selected = labels.loc[score >= cutoff].copy()
    trades = v1.replay_non_overlapping(selected)
    returns = trades["label_net_return_per_notional"].to_numpy(float) if len(trades) else np.empty(0)
    return {
        "selected_labels": int(len(selected)),
        "mean_net_return": float(selected["label_net_return_per_notional"].mean()) if len(selected) else -9.0,
        "success_rate": float(selected["label_success"].mean()) if len(selected) else 0.0,
        "policy_trades": int(len(trades)),
        "policy_compounded_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else -9.0,
        "policy_win_rate": float((returns > 0.0).mean()) if len(returns) else 0.0,
    }


def select_policy(train: pd.DataFrame, validation: pd.DataFrame) -> tuple[Any, float, dict[str, Any], list[dict[str, Any]]]:
    train_x = feature_frame(train)
    validation_x = feature_frame(validation)
    results: list[dict[str, Any]] = []
    fitted: dict[str, Any] = {}
    for name, model, target in model_grid(train):
        model.fit(train_x, target)
        fitted[name] = model
        train_score = score_model(model, train_x)
        validation_score = score_model(model, validation_x)
        for quantile in (0.30, 0.40, 0.50, 0.60, 0.70):
            cutoff = float(np.quantile(train_score, quantile))
            train_metrics = selected_metrics(train, train_score, cutoff)
            validation_metrics = selected_metrics(validation, validation_score, cutoff)
            eligible = (
                train_metrics["selected_labels"] >= MIN_TRAIN_SELECTED
                and validation_metrics["selected_labels"] >= MIN_VALIDATION_SELECTED
                and train_metrics["mean_net_return"] > 0.0
                and validation_metrics["mean_net_return"] > 0.0
                and train_metrics["policy_compounded_return"] > 0.0
                and validation_metrics["policy_compounded_return"] > 0.0
            )
            results.append(
                {
                    "model_name": name,
                    "train_score_quantile": quantile,
                    "score_cutoff": cutoff,
                    "train": train_metrics,
                    "validation": validation_metrics,
                    "eligible": eligible,
                    "selection_score": (
                        min(train_metrics["mean_net_return"], validation_metrics["mean_net_return"])
                        if eligible
                        else -9.0
                    ),
                }
            )
    eligible_results = [row for row in results if row["eligible"]]
    if not eligible_results:
        raise RuntimeError("no meta-filter passed the train/validation profitability contract")
    winner = max(
        eligible_results,
        key=lambda row: (
            row["selection_score"],
            min(row["train"]["selected_labels"], row["validation"]["selected_labels"]),
        ),
    )
    return fitted[winner["model_name"]], float(winner["score_cutoff"]), winner, results


def apply_policy(labels: pd.DataFrame, model: Any, cutoff: float) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    out = labels.copy()
    out["meta_score"] = score_model(model, feature_frame(out))
    out["meta_selected"] = (out["meta_score"] >= cutoff).astype(np.int8)
    selected = out.loc[out["meta_selected"] == 1].copy()
    trades = v1.replay_non_overlapping(selected)
    metrics = selected_metrics(out, out["meta_score"].to_numpy(float), cutoff)
    return out, trades, metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(LABEL_DIR / "train_meta_labels.parquet").query("label_valid == 1").reset_index(drop=True)
    validation = pd.read_parquet(LABEL_DIR / "validation_meta_labels.parquet").query("label_valid == 1").reset_index(drop=True)
    model, cutoff, winner, search = select_policy(train, validation)
    print(
        json.dumps(
            {
                "stage": "meta_policy_locked_before_oos",
                "model_name": winner["model_name"],
                "score_cutoff": cutoff,
                "train": winner["train"],
                "validation": winner["validation"],
            }
        ),
        flush=True,
    )

    split_frames = {"train": train, "validation": validation}
    split_frames["oos"] = pd.read_parquet(LABEL_DIR / "oos_meta_labels.parquet").query("label_valid == 1").reset_index(drop=True)
    split_frames["fresh"] = pd.read_parquet(LABEL_DIR / "fresh_meta_labels.parquet").query("label_valid == 1").reset_index(drop=True)
    summaries: dict[str, Any] = {}
    all_trades: list[pd.DataFrame] = []
    for split, labels in split_frames.items():
        scored, trades, metrics = apply_policy(labels, model, cutoff)
        scored.to_parquet(OUT_DIR / f"{split}_scored_meta_labels.parquet", index=False)
        trades.to_csv(OUT_DIR / f"{split}_selected_diagnostic_trades.csv", index=False)
        summaries[split] = metrics
        all_trades.append(trades)

    artifact_path = OUT_DIR / "meta_filter.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "model_name": winner["model_name"],
            "model": model,
            "score_cutoff": cutoff,
            "base_features": BASE_FEATURES,
            "derived_features": ["slope_atr", "aligned_volume", "aligned_imbalance", "aligned_funding"],
            "label_model_id": v2.MODEL_ID,
        },
        artifact_path,
    )
    search_path = OUT_DIR / "train_validation_search.json"
    search_path.write_text(json.dumps(search, indent=2, default=v1._json_default), encoding="utf-8")

    oos_trades = all_trades[2]
    market_2026 = pd.read_csv(v1.MARKET_2026, usecols=["timestamp", "close"], parse_dates=["timestamp"])
    chart_path = OUT_DIR / "oos_selected_trade_chart.png"
    if len(oos_trades):
        start, end = v1.choose_chart_window(oos_trades)
        view = market_2026.loc[market_2026["timestamp"].between(start, end)]
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (axis, equity_axis) = plt.subplots(2, 1, figsize=(16, 9), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
        axis.plot(view["timestamp"], view["close"], color="#334155", linewidth=1.0)
        shown = oos_trades.loc[pd.to_datetime(oos_trades["decision_timestamp"]).between(start, end)].sort_values("event_end_timestamp")
        for _, trade in shown.iterrows():
            color = "#16a34a" if float(trade["label_net_return_per_notional"]) > 0.0 else "#dc2626"
            marker = "^" if int(trade["candidate_side"]) > 0 else "v"
            axis.scatter(pd.Timestamp(trade["entry_timestamp"]), trade["entry_fill_price"], marker=marker, s=70, color=color, zorder=5)
            axis.scatter(pd.Timestamp(trade["event_end_timestamp"]), trade["exit_fill_price"], marker="x", s=55, color=color, zorder=5)
            axis.plot(
                [pd.Timestamp(trade["entry_timestamp"]), pd.Timestamp(trade["event_end_timestamp"])],
                [trade["entry_fill_price"], trade["exit_fill_price"]],
                color=color,
                linewidth=1.0,
                alpha=0.7,
            )
        equity = (1.0 + shown["label_net_return_per_notional"].astype(float)).cumprod()
        equity_axis.step(pd.to_datetime(shown["event_end_timestamp"]), equity, where="post", color="#0f766e", linewidth=1.5)
        axis.set_title(f"Meta-filtered OOS trades: {start.date()} to {end.date()}")
        axis.set_ylabel("ETHUSDT price")
        equity_axis.set_ylabel("Equity")
        equity_axis.set_xlabel("UTC")
        axis.grid(alpha=0.15)
        equity_axis.grid(alpha=0.15)
        fig.tight_layout()
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)

    report = {
        "model_id": MODEL_ID,
        "status": "meta_filter_trained_policy_locked_before_oos",
        "label_model_id": v2.MODEL_ID,
        "model_name": winner["model_name"],
        "score_cutoff": cutoff,
        "selection": {
            "train_validation_only": True,
            "oos_used_for_selection": False,
            "minimum_train_selected": MIN_TRAIN_SELECTED,
            "minimum_validation_selected": MIN_VALIDATION_SELECTED,
            "winner": winner,
            "search_results": str(search_path),
        },
        "summaries": summaries,
        "artifact": {"path": str(artifact_path), "sha256": v1.sha256(artifact_path)},
        "chart": str(chart_path),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "stored_trade_ledger_is_diagnostic_only": True,
    }
    report_path = OUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=v1._json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "summaries": summaries}, default=v1._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
