"""Red-team the frozen DeepScalp-PnL v1 checkpoint without retraining it."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_deepscalp_pnl_20260717 as ds  # noqa: E402


OUT_PATH = ROOT / f"data/ensemble/reports/{ds.MODEL_ID}_audit.json"
LOOKAHEAD_REPORT = ROOT / "data/ensemble/reports/scalp_1m_lookahead_check_20260717.json"


def compact(metrics: dict) -> dict:
    keys = (
        "bars", "days", "compounded_return_pct", "additive_net_return_pct",
        "additive_gross_return_pct", "max_drawdown_pct", "entries_or_reversals",
        "exposure_fraction", "turnover", "positive_day_fraction", "side_fraction",
    )
    return {key: metrics[key] for key in keys if key in metrics}


def replay(
    model: ds.DeepScalpPolicy,
    embeddings: torch.Tensor,
    returns: np.ndarray,
    timestamp_ns: np.ndarray,
    config: ds.Config,
    device: torch.device,
) -> dict:
    metrics, _ = ds.replay_from_embeddings(
        model, embeddings, returns, timestamp_ns, config.fee_per_notional, device,
    )
    return compact(metrics)


def rank_correlations(
    values: np.ndarray,
    names: list[str],
    target: np.ndarray,
    indices: np.ndarray,
    limit: int = 20,
) -> list[dict]:
    target_series = pd.Series(target[indices])
    rows = []
    for column, name in enumerate(names):
        feature = pd.Series(np.asarray(values[indices, column], dtype=float))
        valid = feature.notna() & target_series.notna()
        if valid.sum() < 100 or feature[valid].nunique() < 2:
            correlation = 0.0
        else:
            correlation = float(feature[valid].corr(target_series[valid], method="spearman"))
        rows.append({"feature": name, "spearman_next_1m_return": correlation})
    rows.sort(key=lambda row: abs(row["spearman_next_1m_return"]), reverse=True)
    return rows[:limit]


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    arrays, metadata = ds.build_or_load_cache(False)
    checkpoint = torch.load(ds.CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = ds.Config(**checkpoint["config"])
    if checkpoint["feature_contract_sha256"] != metadata["source_signature"]["contract_sha256"]:
        raise RuntimeError("checkpoint/cache feature contract mismatch")
    scalers = {key: np.asarray(value, dtype=np.float32) for key, value in checkpoint["scalers"].items()}
    raw_base = np.asarray(arrays["base"])
    raw_micro = np.asarray(arrays["micro"])
    targets = np.asarray(arrays["targets"])
    next_return = np.asarray(arrays["next_return"])
    timestamp_ns = np.asarray(arrays["timestamp_ns"])
    base = ds.apply_scaler(raw_base, scalers["base_center"], scalers["base_scale"])
    micro = ds.apply_scaler(raw_micro, scalers["micro_center"], scalers["micro_scale"])
    valid = ds.causal_window_end_indices(timestamp_ns, targets, next_return, config.window)
    validation = ds._select_indices(
        valid, timestamp_ns, str(pd.Timestamp(config.micro_train_end) + pd.Timedelta(minutes=1)), config.validation_end,
    )
    development = ds._select_indices(
        valid, timestamp_ns, str(pd.Timestamp(config.validation_end) + pd.Timedelta(minutes=1)), config.development_oos_end,
    )
    model = ds.DeepScalpPolicy(base.shape[1], micro.shape[1], config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    print("Encoding frozen policy inputs...", flush=True)
    validation_embeddings = ds.encode_end_indices(model, base, micro, validation, config.window, device)
    development_embeddings = ds.encode_end_indices(model, base, micro, development, config.window, device)
    correct_validation = replay(
        model, validation_embeddings, next_return[validation], timestamp_ns[validation], config, device,
    )
    correct_development = replay(
        model, development_embeddings, next_return[development], timestamp_ns[development], config, device,
    )

    rng = np.random.default_rng(config.seed)
    shuffled_return = next_return[development].copy()
    rng.shuffle(shuffled_return)
    timing_controls = {
        "correct_alignment": correct_development,
        "features_lagged_1m": replay(
            model, development_embeddings[:-1], next_return[development][1:], timestamp_ns[development][1:], config, device,
        ),
        "features_led_1m_noncausal_control": replay(
            model, development_embeddings[1:], next_return[development][:-1], timestamp_ns[development][:-1], config, device,
        ),
        "returns_randomly_shuffled": replay(
            model, development_embeddings, shuffled_return, timestamp_ns[development], config, device,
        ),
    }

    print("Running frozen-input ablations...", flush=True)
    zero_micro = np.zeros_like(micro)
    no_micro_embeddings = ds.encode_end_indices(model, base, zero_micro, development, config.window, device)
    no_micro = replay(model, no_micro_embeddings, next_return[development], timestamp_ns[development], config, device)
    no_book_matrix = micro.copy()
    for column, name in enumerate(metadata["micro_feature_names"]):
        if name.startswith("book_") or name in {"book_available", "book_age_min"}:
            no_book_matrix[:, column] = 0.0
    no_book_embeddings = ds.encode_end_indices(model, base, no_book_matrix, development, config.window, device)
    no_book = replay(model, no_book_embeddings, next_return[development], timestamp_ns[development], config, device)
    zero_base = np.zeros_like(base)
    no_base_embeddings = ds.encode_end_indices(model, zero_base, micro, development, config.window, device)
    no_base = replay(model, no_base_embeddings, next_return[development], timestamp_ns[development], config, device)

    micro_names = metadata["micro_feature_names"]
    age_checks = {}
    for name in ("micro_age_min", "book_age_min"):
        column = micro_names.index(name)
        finite = raw_micro[:, column][np.isfinite(raw_micro[:, column])]
        age_checks[name] = {
            "minimum": float(finite.min()) if len(finite) else None,
            "maximum": float(finite.max()) if len(finite) else None,
            "negative_count": int((finite < -1e-9).sum()),
        }
    top_base_ic = rank_correlations(raw_base, metadata["base_feature_names"], next_return, development)
    top_micro_ic = rank_correlations(raw_micro, micro_names, next_return, development)
    max_abs_raw_ic = max(
        max(abs(row["spearman_next_1m_return"]) for row in top_base_ic),
        max(abs(row["spearman_next_1m_return"]) for row in top_micro_ic),
    )
    ledger = pd.read_csv(ds.LEDGER_PATH, parse_dates=["timestamp"])
    ledger["date"] = ledger["timestamp"].dt.date
    daily_net = ledger.groupby("date")["net_account_return"].sum()
    positive_total = daily_net.clip(lower=0).sum()
    concentration = {
        "largest_positive_day_share": float(daily_net.max() / positive_total) if positive_total > 0 else None,
        "worst_day_additive_return_pct": float(daily_net.min() * 100.0),
        "best_day_additive_return_pct": float(daily_net.max() * 100.0),
    }
    gross_sum = float(ledger["gross_account_return"].sum())
    turnover_sum = float(ledger["turnover"].sum())
    break_even_cost = gross_sum / turnover_sum if turnover_sum > 0 else None

    lookahead = json.loads(LOOKAHEAD_REPORT.read_text()) if LOOKAHEAD_REPORT.exists() else {"verdict": "MISSING"}
    shuffled_ok = timing_controls["returns_randomly_shuffled"]["compounded_return_pct"] < 0
    lag_degrades = timing_controls["features_lagged_1m"]["compounded_return_pct"] < correct_development["compounded_return_pct"]
    age_ok = all(item["negative_count"] == 0 for item in age_checks.values())
    raw_ic_ok = max_abs_raw_ic < 0.25
    btc_semantic_leak_excluded = not any(
        "btc" in name.lower() for name in metadata["base_feature_names"] + metadata["micro_feature_names"]
    )
    audit_pass = (
        lookahead.get("verdict") == "PASS" and shuffled_ok and lag_degrades and age_ok
        and raw_ic_ok and btc_semantic_leak_excluded
    )
    report = {
        "model_id": ds.MODEL_ID,
        "checkpoint_frozen": True,
        "validation_replay": correct_validation,
        "development_oos_replay": correct_development,
        "timing_controls": timing_controls,
        "frozen_input_ablation": {
            "all_microstructure_replaced_by_training_median": no_micro,
            "orderbook_replaced_by_training_median": no_book,
            "all_base_market_features_replaced_by_training_median": no_base,
        },
        "top_base_raw_feature_ic": top_base_ic,
        "top_micro_raw_feature_ic": top_micro_ic,
        "max_absolute_raw_feature_ic": max_abs_raw_ic,
        "asof_age_checks": age_checks,
        "daily_pnl_concentration": concentration,
        "break_even_cost": {
            "per_notional_change": break_even_cost,
            "basis_points_per_notional_change": break_even_cost * 10_000.0 if break_even_cost is not None else None,
        },
        "existing_full_feature_truncation_audit": lookahead,
        "audit_checks": {
            "shuffled_returns_are_unprofitable": shuffled_ok,
            "one_minute_feature_lag_degrades_return": lag_degrades,
            "no_negative_asof_age": age_ok,
            "max_absolute_raw_feature_ic_below_0_25": raw_ic_ok,
            "truncated_feature_rebuild_passed": lookahead.get("verdict") == "PASS",
            "btc_5m_open_timestamp_semantic_leak_excluded": btc_semantic_leak_excluded,
        },
        "audit_pass": audit_pass,
        "promotion_pass": False,
        "promotion_note": "Audit pass only supports causal development status; July is consumed and the history is not promotion-length.",
    }
    OUT_PATH.write_text(json.dumps(report, indent=2, default=ds._json_default))
    print(json.dumps({"audit_pass": audit_pass, "timing_controls": timing_controls, "ablation": report["frozen_input_ablation"]}, indent=2))
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
