#!/usr/bin/env python3
"""Build causal a5dir_v2 probabilities on the active BTC feature contract."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_eval_btc_shared_policy_state_conditioned_r5 import TRAIN_DATA, VAL_DATA, labels_for  # noqa: E402
from scripts.walkforward_btc_shared_policy_state_conditioned_r6 import read_window  # noqa: E402

SELECTION = ROOT / "docs/experiments/btc_shared_policy_v1_r3_head_features.json"
OUT = ROOT / "tmp/btc_a5dir_v2_causal"
OOF_FOLDS = (("2024-04-01", "2024-07-01"), ("2024-07-01", "2024-10-01"), ("2024-10-01", "2025-01-01"))


def train_labels(base: pd.DataFrame) -> pd.DataFrame:
    labels = labels_for(base)
    labels = labels.loc[np.isclose(labels["current_margin_fraction"], 0.0)].copy()
    target = np.sign(labels["teacher_best_target_margin_fraction"].to_numpy(dtype=float)).astype(int)
    labels["a5dir_v2_target"] = np.where(target > 0, 1, np.where(target < 0, 2, 0))
    return labels


def fit(base: pd.DataFrame, features: list[str]) -> LGBMClassifier:
    labels = train_labels(base)
    frame = base.merge(labels[["decision_timestamp", "a5dir_v2_target"]], left_on="timestamp", right_on="decision_timestamp", how="inner")
    x = frame[features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = frame["a5dir_v2_target"]
    model = LGBMClassifier(objective="multiclass", num_class=3, n_estimators=350, learning_rate=.04, num_leaves=31, min_child_samples=100, random_state=270705, verbosity=-1)
    model.fit(x, y)
    return model


def score(model: LGBMClassifier, base: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    probabilities = model.predict_proba(base[features].replace([np.inf, -np.inf], np.nan).fillna(0.0))
    return pd.DataFrame({
        "timestamp": base["timestamp"].to_numpy(),
        "a5dir_v2_available": 1.0,
        "a5dir_v2_flat_prob": probabilities[:, 0],
        "a5dir_v2_long_prob": probabilities[:, 1],
        "a5dir_v2_short_prob": probabilities[:, 2],
        "a5dir_v2_prob_max": probabilities.max(axis=1),
        "a5dir_v2_edge": probabilities[:, 1] - probabilities[:, 2],
        "a5dir_v2_margin": np.abs(probabilities[:, 1] - probabilities[:, 2]),
        "a5dir_v2_side": np.where(probabilities[:, 1] >= probabilities[:, 2], 1.0, -1.0),
    })


def main() -> int:
    features = json.loads(SELECTION.read_text())["action_features"]
    base_2024 = read_window(TRAIN_DATA, features, "2024-01-01", "2024-12-31 23:59:59+00:00")
    base_2025 = read_window(VAL_DATA, features, "2025-01-01", "2025-12-31 23:59:59+00:00")
    oof = []
    for score_start, score_end in OOF_FOLDS:
        cutoff = pd.to_datetime(score_start, utc=True)
        fit_base = base_2024.loc[base_2024["timestamp"] < cutoff].reset_index(drop=True)
        score_base = base_2024.loc[base_2024["timestamp"].between(cutoff, pd.to_datetime(score_end, utc=True), inclusive="left")].reset_index(drop=True)
        oof.append(score(fit(fit_base, features), score_base, features))
    final = fit(base_2024, features)
    OUT.mkdir(parents=True, exist_ok=True)
    pd.concat(oof, ignore_index=True).to_parquet(OUT / "a5dir_v2_2024_oof.parquet", index=False)
    score(final, base_2025, features).to_parquet(OUT / "a5dir_v2_2025_forward.parquet", index=False)
    report = {
        "artifact": "a5dir_v2",
        "input_contract": "active adaptive-squeeze BTC 24 action features only",
        "forbidden_inputs_absent": ["Regime4", "legacy a5dir", "teacher", "M7", "AI", "labels", "future", "PnL"],
        "label": "sign of state-conditioned cost-aware teacher best target at current margin 0; labels only",
        "oof_folds": OOF_FOLDS,
        "oof_rows": int(sum(len(x) for x in oof)),
        "forward_rows": int(len(base_2025)),
        "oof_source_row_never_seen_by_its_router": True,
    }
    (OUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
