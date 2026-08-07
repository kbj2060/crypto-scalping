#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard
import train_omega1_regime3_routed_expert_direction_quality_tabm_20260602 as tabm


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_1_tabm_head_routing_compare_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_1_tabm_head_routing_compare_20260602"

SPECS = [
    ("global", 0.00),
    ("hard", 0.00),
    ("soft", 0.00),
    ("soft", 0.05),
    ("soft", 0.10),
    ("soft", 0.20),
    ("hybrid", 0.05),
    ("hybrid", 0.10),
]


def _fit_head_models_compare(
    x: pd.DataFrame,
    y: np.ndarray,
    frame: pd.DataFrame,
    *,
    mode: str,
    floor: float,
    seed: int,
    epochs: int,
    model_dir: Path,
    suffix: str,
) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    route = hard._route_id(frame)
    probs = tabm._route_probs(frame)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}

    if mode == "global":
        classes = sorted(np.unique(y).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{mode}: missing zigzag_action classes: {classes}")
        path = model_dir / f"global_{suffix}.pt"
        payload = tabm._fit_tabm(x.reset_index(drop=True), y, sample_weight=None, seed=seed, epochs=epochs, model_path=path)
        for expert in hard.EXPERT_NAMES:
            models[expert] = payload
            summaries[expert] = {
                "rows": int(len(y)),
                "weight_sum": None,
                "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=3))},
                "model": str(path),
                "epochs_ran": int(payload["epochs_ran"]),
                "best_validation_loss": float(payload["best_validation_loss"]),
            }
        return {"models": models, "summaries": summaries}

    for idx, expert in enumerate(hard.EXPERT_NAMES):
        if mode == "hard":
            mask = route == idx
            if int(mask.sum()) < 1000:
                raise RuntimeError(f"{expert}: too few hard-routed rows: {int(mask.sum())}")
            x_fit = x.loc[mask].reset_index(drop=True)
            y_fit = y[mask]
            sample_weight = None
            effective_rows = int(mask.sum())
            weight_sum = None
        elif mode == "soft":
            x_fit = x.reset_index(drop=True)
            y_fit = y
            sample_weight = float(floor) + probs[:, idx]
            effective_rows = int(len(y))
            weight_sum = float(np.asarray(sample_weight, dtype=np.float64).sum())
        elif mode == "hybrid":
            x_fit = x.reset_index(drop=True)
            y_fit = y
            sample_weight = np.where(route == idx, 1.0, float(floor))
            effective_rows = int(len(y))
            weight_sum = float(np.asarray(sample_weight, dtype=np.float64).sum())
        else:
            raise ValueError(f"unknown mode: {mode}")

        classes = sorted(np.unique(y_fit).astype(int).tolist())
        if classes != [0, 1, 2]:
            raise RuntimeError(f"{mode}/{expert}: missing zigzag_action classes: {classes}")
        path = model_dir / f"{expert}_{suffix}.pt"
        payload = tabm._fit_tabm(x_fit, y_fit, sample_weight=sample_weight, seed=seed + idx, epochs=epochs, model_path=path)
        models[expert] = payload
        summaries[expert] = {
            "rows": effective_rows,
            "weight_sum": weight_sum,
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_fit, minlength=3))},
            "model": str(path),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }
    return {"models": models, "summaries": summaries}


def main() -> int:
    tabm.MODEL_ID = MODEL_ID
    tabm.OUT_DIR = OUT_DIR
    tabm.SPECS = SPECS
    tabm._fit_head_models = _fit_head_models_compare
    return tabm.main()


if __name__ == "__main__":
    raise SystemExit(main())
