#!/usr/bin/env python3
"""Materialize the validation-selected 1h HMM into the live six-column 5m contract.

Values are joined only from completed hourly bars at or before each 5m timestamp.
The wide24 names are an explicit experiment contract replacement, not a runtime alias.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import chart_regime3_1h_fresh_forward_latest_week_20260728 as one_hour  # noqa: E402
import research_regime3_1h_deep_20260728 as deep  # noqa: E402


MODEL = ROOT / "tmp/causal_regen_20260516/regime3_1h_deep_research_20260728/selected_validation_only_regime3_1h_model.joblib"
SOURCE = {
    2024: ROOT / "data/splits/year_oos/training_features_2024.csv",
    2025: ROOT / "data/splits/year_oos/training_features_2025.csv",
    2026: ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
OUT_DIR = ROOT / "tmp/causal_regen_20260516/regime3_1h_as_5m_contract_20260728"
OUT_2024_2025 = OUT_DIR / "training_features_2024_2025_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
OUT_2026 = OUT_DIR / "training_features_2026_regime3_current_sensitive_hmm_1h_masked_wide24.csv"
AUDIT = OUT_DIR / "materialization_audit.json"
PREFIX = "regime3_current_sensitive_wide24_"


def _hourly_probability(payload: dict, features: pd.DataFrame) -> pd.DataFrame:
    cols = payload["feature_cols"]
    medians = pd.Series(payload["feature_medians"])
    observations = payload["scaler"].transform(features[cols].fillna(medians).fillna(0.0))
    model = payload["model"]
    log_emission = model._log_emission(observations)
    log_transition = np.log(model.A_ + 1e-300)
    state_probability = np.empty((len(features), model.n_states), dtype=np.float64)
    previous: np.ndarray | None = None
    for bar_index in range(len(features)):
        if previous is None:
            current = np.log(model.pi_ + 1e-300) + log_emission[bar_index]
        else:
            current = log_emission[bar_index] + model._logsumexp(
                previous[:, None] + log_transition, axis=0
            )
        current -= model._logsumexp(current, axis=0)
        state_probability[bar_index] = np.exp(current)
        previous = current
    probability = one_hour._class_probability(state_probability, payload["state_class_matrix"])
    sorted_probability = np.sort(probability, axis=1)
    out = pd.DataFrame({"hour_timestamp": features["timestamp"]})
    for class_index, name in enumerate(one_hour.CLASSES):
        out[f"{PREFIX}{name}_prob"] = probability[:, class_index]
    out[f"{PREFIX}confidence"] = sorted_probability[:, -1]
    out[f"{PREFIX}entropy"] = -np.sum(
        probability * np.log(np.clip(probability, 1e-12, None)), axis=1
    ) / np.log(len(one_hour.CLASSES))
    out[f"{PREFIX}margin"] = sorted_probability[:, -1] - sorted_probability[:, -2]
    return out


def _materialize(target: pd.DataFrame, hourly: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target_ts = target[["timestamp"]].sort_values("timestamp").reset_index(drop=True)
    joined = pd.merge_asof(
        target_ts,
        hourly.sort_values("hour_timestamp"),
        left_on="timestamp",
        right_on="hour_timestamp",
        direction="backward",
        allow_exact_matches=True,
    )
    missing = joined["hour_timestamp"].isna()
    lag_all = (joined["timestamp"] - joined["hour_timestamp"]).dt.total_seconds() / 60.0
    stale = lag_all.gt(55.0)
    available = joined.loc[~missing & ~stale].copy()
    lag_minutes = (available["timestamp"] - available["hour_timestamp"]).dt.total_seconds() / 60.0
    output = available.drop(columns=["hour_timestamp"])
    return output, {
        "target_rows": int(len(target_ts)),
        "materialized_rows": int(len(output)),
        "head_rows_without_completed_hour": int(missing.sum()),
        "stale_rows_removed": int(stale.sum()),
        "max_completed_hour_lag_minutes": float(lag_minutes.max()),
        "min_completed_hour_lag_minutes": float(lag_minutes.min()),
        "future_hour_join_count": int((lag_minutes < 0.0).sum()),
        "range": [str(output["timestamp"].iloc[0]), str(output["timestamp"].iloc[-1])],
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = joblib.load(MODEL)
    raw_parts = [one_hour._read_source(SOURCE[year]) for year in (2024, 2025, 2026)]
    continuous_raw = (
        pd.concat(raw_parts, ignore_index=True)
        .sort_values("timestamp")
        .drop_duplicates("timestamp", keep="last")
        .reset_index(drop=True)
    )
    hourly_features = one_hour._with_1h_features(one_hour._aggregate_completed_hours(continuous_raw))
    hourly = _hourly_probability(payload, hourly_features)

    targets = {
        "2024_2025": pd.concat(
            [part[["timestamp"]] for part in raw_parts[:2]], ignore_index=True
        ).sort_values("timestamp").reset_index(drop=True),
        "2026": raw_parts[2][["timestamp"]].copy(),
    }
    out_train, train_audit = _materialize(targets["2024_2025"], hourly)
    out_eval, eval_audit = _materialize(targets["2026"], hourly)
    out_train.to_csv(OUT_2024_2025, index=False)
    out_eval.to_csv(OUT_2026, index=False)
    report = {
        "contract": "validation-selected 1h HMM replaces the six existing 5m current-regime inputs",
        "model_path": str(MODEL),
        "model_id": payload["model_id"],
        "state_count": payload["state_count"],
        "feature_cols": payload["feature_cols"],
        "output_columns": [c for c in out_train.columns if c != "timestamp"],
        "fresh_forward_bar_by_bar": True,
        "completed_hour_asof_join": True,
        "future_rows_used": False,
        "saved_5m_hmm_features_used": False,
        "train_2024_2025": train_audit,
        "eval_2026": eval_audit,
        "outputs": {"train_2024_2025": str(OUT_2024_2025), "eval_2026": str(OUT_2026)},
    }
    AUDIT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
