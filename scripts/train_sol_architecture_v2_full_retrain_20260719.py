#!/usr/bin/env python3
"""Full-data retrain of the frozen SOL architecture-v2 research candidate.

Hyperparameters are copied from the completed validation search.  This script
uses every available 2025-01-01..2026-07-12 row and produces no validation or
test performance claim.
"""
from __future__ import annotations

import json
import pickle
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707 as sol  # noqa: E402
import train_eval_sol_architecture_v2_entry_20260719 as entry  # noqa: E402
import train_eval_sol_architecture_v2_pipeline_20260719 as pipeline  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


OUT_DIR = ROOT / "tmp/causal_regen_20260516/sol_architecture_v2_full_retrain_20260719"
EPOCHS = 8
QUALITY_ITERATIONS = {"long": 2, "short": 20}
OUTCOME_ITERATIONS = 600
EXIT_ITERATIONS = 193


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tb_all() -> pd.DataFrame:
    cols = ["timestamp", *entry._target_cols(24)]
    parts = [pd.read_csv(entry.TB_DIR / f"{split}_triple_barrier_labels.csv", usecols=cols, parse_dates=["timestamp"]) for split in ("train", "validation", "oos")]
    return pd.concat(parts, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last")


def _fit_tcn(
    frame: pd.DataFrame,
    base_cols: list[str],
    *,
    device: torch.device,
) -> dict[str, Any]:
    raw = frame[base_cols].to_numpy(dtype=np.float32)
    mean = raw.mean(axis=0).astype(np.float32)
    std = raw.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    x = ((raw - mean) / std).astype(np.float32)
    gates = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float32)
    gates /= np.clip(gates.sum(axis=1, keepdims=True), 1.0e-8, None)
    y_dir = frame["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = frame[["q_long", "q_short"]].to_numpy(dtype=np.float32)
    target_cols = entry._target_cols(24)
    outcome_raw = frame[target_cols[:6]].to_numpy(dtype=np.float32)
    outcome_mean = outcome_raw.mean(axis=0).astype(np.float32)
    outcome_std = outcome_raw.std(axis=0).astype(np.float32)
    outcome_std[outcome_std < 1.0e-6] = 1.0
    y_outcome = ((outcome_raw - outcome_mean) / outcome_std).astype(np.float32)
    variant = entry.Variant("tcn_l24_h32_full", "tcn", 24, 32, 0.15)
    ds = entry.SequenceRows(x, gates, y_dir, y_quality, y_outcome, seq_len=variant.seq_len, start=0, end=len(frame))
    loader = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0)
    model = entry.SoftResidualEntry(len(base_cols), variant).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4, weight_decay=1.0e-3)
    classes = np.array([0, 1, 2])
    dir_weights = torch.tensor(compute_class_weight("balanced", classes=classes, y=y_dir), dtype=torch.float32, device=device)
    quality_pos = torch.tensor(
        [(y_quality[:, i] == 0).sum() / max((y_quality[:, i] == 1).sum(), 1) for i in range(2)],
        dtype=torch.float32,
        device=device,
    )
    for epoch_idx in range(EPOCHS):
        model.train()
        losses = []
        for xb, gb, yd, yq, yo, _idx in loader:
            xb, gb, yd, yq, yo = xb.to(device), gb.to(device), yd.to(device), yq.to(device), yo.to(device)
            out = model(xb, gb)
            loss = (
                0.35 * nn.functional.cross_entropy(out["direction"], yd, weight=dir_weights)
                + nn.functional.binary_cross_entropy_with_logits(out["quality"], yq, pos_weight=quality_pos)
                + 0.10 * nn.functional.smooth_l1_loss(out["outcome"], yo)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        print(f"stage=tcn epoch={epoch_idx + 1} loss={np.mean(losses):.6f}", flush=True)
    payload = {
        "variant": {"name": variant.name, "encoder": variant.encoder, "seq_len": variant.seq_len, "hidden": variant.hidden, "dropout": variant.dropout},
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "feature_columns": base_cols,
        "feature_mean": mean,
        "feature_std": std,
        "outcome_mean": outcome_mean,
        "outcome_std": outcome_std,
        "epochs": EPOCHS,
        "loss_weights": {"direction": 0.35, "quality": 1.0, "outcome": 0.10},
    }
    torch.save(payload, OUT_DIR / "entry_tcn24_full.pt")
    return payload


def _fit_quality_outcome(frame: pd.DataFrame, base_cols: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    quality_models: dict[str, Any] = {}
    outcome_models: dict[str, Any] = {}
    for side in ("long", "short"):
        reason = f"tb_{side}_reason_h24_conservative"
        y = (frame[reason] == "tp").astype(np.int64)
        quality = CatBoostClassifier(
            iterations=QUALITY_ITERATIONS[side],
            depth=7,
            learning_rate=0.03,
            loss_function="Logloss",
            auto_class_weights="Balanced",
            random_seed=260719,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
        quality.fit(frame[base_cols], y)
        quality.save_model(str(OUT_DIR / f"quality_h24_{side}_full.cbm"))
        quality_models[side] = quality
        for target, loss, iterations in (
            ("return", "RMSE", OUTCOME_ITERATIONS),
            ("adverse", "Quantile:alpha=0.80", OUTCOME_ITERATIONS),
        ):
            model = CatBoostRegressor(
                iterations=iterations,
                depth=7,
                learning_rate=0.03,
                loss_function=loss,
                random_seed=260720,
                verbose=False,
                allow_writing_files=False,
                thread_count=-1,
            )
            values = frame[f"tb_{side}_{'ret' if target == 'return' else 'mae'}_h24_conservative"].to_numpy(dtype=np.float64)
            if target == "adverse":
                values = np.abs(values)
            model.fit(frame[base_cols], values)
            model.save_model(str(OUT_DIR / f"outcome_{side}_{target}_full.cbm"))
            outcome_models[f"{side}_{target}"] = model
    return quality_models, outcome_models


def _fit_exit(full_raw: pd.DataFrame, all_feature_cols: list[str], base_cols: list[str]) -> tuple[CatBoostClassifier, list[str], dict[str, Any]]:
    fee, slip = sol.omega._load_fee_slip()
    base_input = parent._base_input(full_raw, all_feature_cols)
    x_raw, y, _route, diag = sol._build_exit_dataset_entry_label_terminal_giveback(
        full_raw,
        base_input,
        risk_margin=None,
        risk_leverage=None,
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        max_samples=60000,
        terminal_window=3,
        adverse_unreal=-0.010,
        min_mfe_for_giveback=0.006,
        giveback_min=0.65,
    )
    x = parent._exit_input_from_position_rows(x_raw, base_cols)
    model = CatBoostClassifier(
        iterations=EXIT_ITERATIONS,
        depth=7,
        learning_rate=0.03,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=260721,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    model.fit(x, y)
    model.save_model(str(OUT_DIR / "separate_exit_full.cbm"))
    return model, list(x.columns), {"rows": len(x), "positive_rate": float(np.mean(y)), "label_diag": diag}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _seed(260719)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = sol._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=pipeline.LABEL_DIR,
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    full_raw = pd.concat([frames["train_raw"], frames["val_raw"], frames["oos_raw"]], ignore_index=True)
    full_raw["_split"] = "full"
    full, missing = entry._attach_targets(full_raw, _tb_all(), horizon=24)
    contract = torch.load(entry.ETH_CONTRACT, map_location="cpu", weights_only=False)
    base_cols = list(contract["base_cols"])
    tcn_payload = _fit_tcn(full, base_cols, device=device)
    quality_models, outcome_models = _fit_quality_outcome(full, base_cols)
    exit_model, exit_columns, exit_diag = _fit_exit(full_raw, list(frames["feature_cols"]), base_cols)
    selected = json.loads((pipeline.OUT_DIR / "report.json").read_text())["selected"]["config"]
    with (OUT_DIR / "full_candidate.pkl").open("wb") as handle:
        pickle.dump(
            {
                "entry_artifact": str(OUT_DIR / "entry_tcn24_full.pt"),
                "quality_models": quality_models,
                "outcome_models": outcome_models,
                "exit_model": exit_model,
                "exit_columns": exit_columns,
                "base_cols": base_cols,
                "selected_config": selected,
            },
            handle,
        )
    report = {
        "model_id": "sol_architecture_v2_full_retrain_20260719",
        "research_candidate_only": True,
        "live_path_changed": False,
        "performance_claim": False,
        "training_range": [str(full.timestamp.iloc[0]), str(full.timestamp.iloc[-1])],
        "rows": len(full),
        "excluded_missing_h24_target_rows": missing,
        "base_feature_count": len(base_cols),
        "architecture": "TCN24 direction + conditional H24 quality + outcome heads + separate exit + semi-Markov transitions",
        "frozen_training_config": {
            "tcn_epochs": EPOCHS,
            "quality_iterations": QUALITY_ITERATIONS,
            "outcome_iterations": OUTCOME_ITERATIONS,
            "exit_iterations": EXIT_ITERATIONS,
            "selected_policy": selected,
        },
        "exit": exit_diag,
        "artifacts": {
            "entry": str(OUT_DIR / "entry_tcn24_full.pt"),
            "bundle": str(OUT_DIR / "full_candidate.pkl"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
