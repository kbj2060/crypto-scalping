#!/usr/bin/env python3
"""BTC v3 research candidate: causal HMM win-probability entry nowcast.

Parallel candidate to Stage 1 (docs/model_contracts/btc_v3_stage1_sparse_events_20260714.md),
NOT a replacement -- ts_action and the existing sparse_event_dataset remain untouched. Builds an
alternative entry signal by reusing GaussianStateModel unmodified from
scripts/retrain_clean_regime_hmm_20260517.py (same causal `filter_proba` contract used by the
live Regime3 "current" HMM), fit on the existing BTC 28-feature hourly contract instead of the ETH
RegimeEngine factors. Instead of mapping hidden states to a bull/bear/chop rule label, this maps
states to a win-rate/mean-return vector learned from Stage 1's own realized-outcome events
(sparse_event_dataset.parquet) -- so `entry_score(t) = filter_proba(t) @ state_win_rate` is a
causal "predicted win probability of entering now" nowcast.

Enforces docs/model_contracts/btc_v3_holdout_policy_20260714.md: refuses to fit or score using any
timestamp >= HOLDOUT_START.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from scripts.retrain_clean_regime_hmm_20260517 import GaussianStateModel, _json_default  # noqa: E402
import train_eval_btc_v2_regime_trendscan_20260714 as btc_v2  # noqa: E402

MODEL_ID = "btc_v3_hmm_entry_score_20260715"
HOLDOUT_START = pd.Timestamp("2026-07-14 00:00:00")
SPARSE_EVENTS_PATH = ROOT / "tmp/causal_regen_20260516/btc_v3_sparse_event_dataset_20260714/sparse_event_dataset.parquet"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/btc_v3_hmm_mamba_candidate_20260715"
VAL_START = pd.Timestamp("2025-10-01")

N_STATES = 12
STICKY = 0.93
PCA_COMPONENTS = 12
N_ITER = 22
SEED = 20260715


def _fit_observations(train_x: pd.DataFrame, all_x: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray, Any, pd.Series]:
    medians = train_x.median(numeric_only=True).fillna(0.0)
    train_filled = train_x.fillna(medians).fillna(0.0)
    all_filled = all_x.fillna(medians).fillna(0.0)
    preprocess = make_pipeline(
        RobustScaler(quantile_range=(5.0, 95.0)),
        PCA(n_components=min(PCA_COMPONENTS, train_filled.shape[1]), whiten=True, random_state=int(seed)),
    )
    train_obs = preprocess.fit_transform(train_filled)
    all_obs = preprocess.transform(all_filled)
    return train_obs, all_obs, preprocess, medians


def _state_outcome_vectors(state_prob: np.ndarray, win: np.ndarray, trade_return: np.ndarray, smoothing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Weighted-average win-rate / mean-return per hidden state, using each event's causal state
    posterior as a soft assignment weight (mirrors _state_class_matrix's approach in
    retrain_clean_regime_hmm_20260517.py, but regresses onto a realized-outcome target instead of a
    discrete rule-based class)."""
    weight_sum = state_prob.sum(axis=0) + smoothing
    win_rate = (state_prob * win[:, None]).sum(axis=0) / weight_sum
    mean_return = (state_prob * trade_return[:, None]).sum(axis=0) / weight_sum
    baseline_win = float(win.mean())
    win_rate = np.where(state_prob.sum(axis=0) > 0, win_rate, baseline_win)
    return win_rate, mean_return


def build(history_end: pd.Timestamp) -> dict[str, Any]:
    if history_end >= HOLDOUT_START:
        raise RuntimeError(
            f"history_end={history_end} >= HOLDOUT_START={HOLDOUT_START} -- refusing per "
            f"docs/model_contracts/btc_v3_holdout_policy_20260714.md"
        )
    print("stage=load_hourly_btc_features", flush=True)
    hourly, feature_columns = btc_v2._read_hourly()
    hourly = hourly.loc[hourly["timestamp"] <= history_end].reset_index(drop=True)

    print("stage=load_sparse_events", flush=True)
    if not SPARSE_EVENTS_PATH.exists():
        raise FileNotFoundError(f"{SPARSE_EVENTS_PATH} missing -- run build_btc_v3_sparse_event_dataset_20260714.py first")
    events = pd.read_parquet(SPARSE_EVENTS_PATH)
    events = events.loc[events["event_hour_timestamp"] <= history_end].reset_index(drop=True)
    if events["event_hour_timestamp"].max() >= HOLDOUT_START:
        raise RuntimeError("sparse events extend past HOLDOUT_START -- refusing")

    x_all = hourly[feature_columns].apply(pd.to_numeric, errors="coerce")
    train_mask = hourly["timestamp"].lt(VAL_START).to_numpy()
    x_train = x_all.loc[train_mask]

    print("stage=fit_causal_hmm", flush=True)
    train_obs, all_obs, preprocess, medians = _fit_observations(x_train, x_all, SEED)
    model = GaussianStateModel(N_STATES, N_ITER, SEED, sticky=STICKY).fit(train_obs)
    state_prob_all = model.filter_proba(all_obs)

    print("stage=map_states_to_realized_outcomes", flush=True)
    hourly_ts = hourly["timestamp"].to_numpy(dtype="datetime64[ns]")
    event_state_idx = np.searchsorted(hourly_ts, events["event_hour_timestamp"].to_numpy(dtype="datetime64[ns]"), side="left")
    valid = event_state_idx < len(hourly_ts)
    events = events.loc[valid].reset_index(drop=True)
    event_state_idx = event_state_idx[valid]
    event_train_mask = events["event_hour_timestamp"].lt(VAL_START).to_numpy()

    event_state_prob = state_prob_all[event_state_idx]
    win = events["win"].to_numpy(dtype=np.float64)
    trade_return = events["trade_return"].to_numpy(dtype=np.float64)
    state_win_rate, state_mean_return = _state_outcome_vectors(
        event_state_prob[event_train_mask], win[event_train_mask], trade_return[event_train_mask]
    )

    entry_score = state_prob_all @ state_win_rate
    entry_return_score = state_prob_all @ state_mean_return

    val_event_mask = ~event_train_mask
    val_entry_score = event_state_prob[val_event_mask] @ state_win_rate
    val_win = win[val_event_mask]
    val_auc = None
    if len(np.unique(val_win)) > 1:
        from sklearn.metrics import roc_auc_score
        val_auc = float(roc_auc_score(val_win, val_entry_score))

    out_dir = OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / f"{MODEL_ID}.joblib"
    joblib.dump(
        {
            "model_id": MODEL_ID,
            "feature_cols": feature_columns,
            "feature_medians": medians.to_dict(),
            "preprocess": preprocess,
            "model": model,
            "state_win_rate": state_win_rate,
            "state_mean_return": state_mean_return,
            "n_states": N_STATES,
            "sticky": STICKY,
            "pca_components": PCA_COMPONENTS,
        },
        model_path,
    )

    scores = pd.DataFrame({
        "timestamp": hourly["timestamp"],
        "btc_v3_hmm_entry_score": entry_score,
        "btc_v3_hmm_entry_return_score": entry_return_score,
    })
    scores_path = out_dir / "btc_v3_hmm_entry_score.parquet"
    scores.to_parquet(scores_path, index=False)

    report = {
        "model_id": MODEL_ID,
        "status": "research_candidate_not_live",
        "supersedes": "none (parallel to stage1_sparse_events)",
        "history_end": str(history_end),
        "holdout_start": str(HOLDOUT_START),
        "val_start": str(VAL_START),
        "n_states": N_STATES,
        "sticky": STICKY,
        "pca_components": PCA_COMPONENTS,
        "n_iter": N_ITER,
        "feature_count": len(feature_columns),
        "train_hourly_rows": int(train_mask.sum()),
        "n_events_total": int(len(events)),
        "n_events_train": int(event_train_mask.sum()),
        "n_events_val": int(val_event_mask.sum()),
        "state_win_rate": state_win_rate.tolist(),
        "state_mean_return": state_mean_return.tolist(),
        "validation": {
            "n_val_events": int(val_event_mask.sum()),
            "val_entry_score_auc_vs_win": val_auc,
            "val_win_rate_baseline": float(val_win.mean()) if len(val_win) else None,
        },
        "log_likelihood": model.log_likelihood_,
        "artifacts": {"model": str(model_path), "scores": str(scores_path)},
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": True,
        "trade_ledgers_used_as_input_note": (
            "Stage 1's sparse_event_dataset realized outcomes are used to fit the state->win-rate "
            "mapping (a training label source, exactly analogous to how retrain_clean_regime_hmm's "
            "state_class_matrix is fit from RegimeEngine labels) -- not as a live input feature; "
            "scoring itself only ever calls filter_proba on causal hourly features."
        ),
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
    }
    report_path = out_dir / "btc_v3_hmm_entry_score_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=_json_default))
    print(f"stage=done model={model_path} scores={scores_path} report={report_path}", flush=True)
    print(f"val_auc={val_auc} n_val_events={int(val_event_mask.sum())}", flush=True)
    return report


def main() -> int:
    history_end = pd.Timestamp("2026-07-12 23:59:59")
    build(history_end)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
