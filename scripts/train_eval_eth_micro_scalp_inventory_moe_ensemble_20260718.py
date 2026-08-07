"""Train a three-seed, nine-expert duration-free ETH micro-scalp ensemble.

Each seed owns the dual causal encoder / three-regime-expert / inventory-Q
architecture from inventory_moe_v1. Mixed Q values are averaged across seeds;
all nine expert action heads remain available for consensus switching. Policy
selection still uses tune only. There is no fixed/max holding period.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_eth_micro_scalp_inventory_moe_20260718 as core  # noqa: E402


MODEL_ID = "eth_micro_scalp_inventory_moe_ensemble_v2_20260718"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "ensemble.pt"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "development_diagnostic_ledger.csv"
SEEDS = (18, 29, 41)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def aggregate_seed_predictions(seed_predictions: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not seed_predictions:
        raise ValueError("at least one seed prediction is required")
    q = np.mean(np.stack([row["q"] for row in seed_predictions], axis=0), axis=0)
    expert_q = np.concatenate([row["expert_q"] for row in seed_predictions], axis=1)
    gates = np.concatenate([row["gates"] for row in seed_predictions], axis=1)
    return {"q": q.astype(np.float32), "expert_q": expert_q.astype(np.float32), "gates": gates.astype(np.float32)}


def prepare_data(config: core.Config) -> dict[str, Any]:
    arrays, metadata = core.load_frozen_cache()
    frame = core.build_sequence_frame(arrays, metadata, config)
    timestamps = frame["timestamps"]
    masks = {
        "fit": core.purged_interval_mask(timestamps, config.fit_start, config.tune_start, config.forecast_horizon_min),
        "tune": core.purged_interval_mask(timestamps, config.tune_start, config.validation_start, config.forecast_horizon_min),
        "validation": core.purged_interval_mask(
            timestamps, config.validation_start, config.development_start, config.forecast_horizon_min
        ),
        "development": core.purged_interval_mask(
            timestamps, config.development_start, config.development_end, config.forecast_horizon_min
        ),
    }
    fit_mask = masks["fit"]
    base_center, base_scale = core.fit_robust_scaler(frame["base_raw"], fit_mask)
    micro_center, micro_scale = core.fit_robust_scaler(frame["micro_raw"], fit_mask & frame["available"])
    aux_valid = fit_mask & np.isfinite(frame["target_raw"]).all(axis=1)
    aux_center, aux_scale = core.fit_robust_scaler(frame["target_raw"], aux_valid)
    base = core.apply_scaler(frame["base_raw"], base_center, base_scale)
    micro = core.apply_scaler(frame["micro_raw"], micro_center, micro_scale)
    auxiliary = core.apply_scaler(frame["target_raw"], aux_center, aux_scale)

    volatility_idx = frame["base_names"].index("garman_klass_vol")
    fit_indices = np.flatnonzero(fit_mask)
    teacher_q_local, teacher_action_local = core.build_cost_aware_teacher(
        frame["next_return"][fit_indices], frame["available"][fit_indices],
        frame["base_raw"][fit_indices, volatility_idx], config.fee_per_notional_change,
        config.teacher_gamma, config.teacher_inventory_vol_weight, config.teacher_advantage_clip_bp,
    )
    teacher_q = np.full((len(timestamps), 3, 3), np.nan, dtype=np.float32)
    teacher_action = np.zeros((len(timestamps), 3), dtype=np.int64)
    teacher_q[fit_indices] = teacher_q_local
    teacher_action[fit_indices] = teacher_action_local
    train_indices = core.valid_window_end_indices(
        fit_mask & np.isfinite(frame["target_raw"]).all(axis=1), timestamps, config.window
    )
    split_indices = {
        name: core.valid_window_end_indices(mask, timestamps, config.window) for name, mask in masks.items()
    }
    return {
        "metadata": metadata,
        "frame": frame,
        "masks": masks,
        "base": base,
        "micro": micro,
        "auxiliary": auxiliary,
        "teacher_q": teacher_q,
        "teacher_action": teacher_action,
        "teacher_action_local": teacher_action_local,
        "train_indices": train_indices,
        "split_indices": split_indices,
        "scalers": {
            "base_center": base_center, "base_scale": base_scale,
            "micro_center": micro_center, "micro_scale": micro_scale,
            "aux_center": aux_center, "aux_scale": aux_scale,
        },
    }


def train_seed(
    seed: int,
    base_config: core.Config,
    prepared: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], list[dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    config = replace(base_config, seed=seed)
    core.seed_everything(seed)
    model = core.InventoryMoEQPolicy(
        prepared["base"].shape[1], prepared["micro"].shape[1], prepared["auxiliary"].shape[1], config
    ).to(device)
    print(f"seed={seed} training", flush=True)
    history = core.train_model(
        model, prepared["base"], prepared["micro"], prepared["teacher_q"],
        prepared["teacher_action"], prepared["auxiliary"], prepared["train_indices"], config, device,
    )
    predictions: dict[str, dict[str, np.ndarray]] = {}
    for name, indices in prepared["split_indices"].items():
        q, gates, expert_q = core.infer_q_tables(
            model, prepared["base"], prepared["micro"], indices,
            config.window, config.batch_size, device,
        )
        predictions[name] = {"q": q, "gates": gates, "expert_q": expert_q}
    state = {name: value.detach().cpu() for name, value in model.state_dict().items()}
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state, history, predictions


def run(config: core.Config, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prepared = prepare_data(config)
    seed_states: dict[str, dict[str, torch.Tensor]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    seed_predictions: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in prepared["split_indices"]
    }
    for seed in seeds:
        state, history, predictions = train_seed(seed, config, prepared, device)
        seed_states[str(seed)] = state
        histories[str(seed)] = history
        for name in seed_predictions:
            seed_predictions[name].append(predictions[name])

    ensemble = {name: aggregate_seed_predictions(rows) for name, rows in seed_predictions.items()}
    frame = prepared["frame"]
    timestamps = frame["timestamps"]
    tune_indices = prepared["split_indices"]["tune"]
    policy, candidates = core.select_q_policy(
        ensemble["fit"]["q"], ensemble["tune"]["q"], ensemble["tune"]["expert_q"],
        frame["available"][tune_indices], frame["next_return"][tune_indices],
        timestamps[tune_indices], config,
    )

    results: dict[str, Any] = {}
    stresses: dict[str, Any] = {}
    ledgers: dict[str, Any] = {}
    for name in ("tune", "validation", "development"):
        indices = prepared["split_indices"][name]
        metrics, ledger = core.replay_q_policy(
            ensemble[name]["q"], frame["available"][indices], frame["next_return"][indices],
            timestamps[indices], policy, config.fee_per_notional_change, ensemble[name]["expert_q"],
        )
        results[name] = metrics
        ledgers[name] = ledger
        stresses[name] = core.cost_stress(
            ledger["position"].to_numpy(dtype=np.int8), frame["next_return"][indices], timestamps[indices]
        )

    active_and_positive = (
        policy.enabled
        and results["validation"]["compounded_return_pct"] > 0.0
        and results["development"]["compounded_return_pct"] > 0.0
    )
    expert_count = len(seeds) * config.experts
    execution_policy = policy if active_and_positive else core.QPolicy(False, 0.0, expert_count)
    if not policy.enabled:
        reason = "No active nine-expert ensemble policy survived tune after modeled cost."
    elif results["validation"]["compounded_return_pct"] <= 0.0:
        reason = "The tune-selected ensemble failed locked validation; artifact execution is fail-safe CASH."
    elif results["development"]["compounded_return_pct"] <= 0.0:
        reason = "The tune-selected ensemble failed the development diagnostic; artifact execution is fail-safe CASH."
    else:
        reason = "Historical intervals are consumed development data; post-freeze fresh-forward evidence is required."

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "seeds": list(seeds),
        "seed_model_states": seed_states,
        "config": asdict(config),
        "base_feature_names": frame["base_names"],
        "micro_feature_names": frame["micro_names"],
        "scalers": prepared["scalers"],
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": active_and_positive,
        "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        "trainer_script_sha256": _sha256(Path(__file__)),
        "core_script_sha256": _sha256(Path(core.__file__)),
        "fixed_holding_period_used": False,
    }
    torch.save(checkpoint, MODEL_PATH)
    ledgers["validation"].to_csv(VALIDATION_LEDGER_PATH, index=False)
    ledgers["development"].to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    gate_means = {
        name: ensemble[name]["gates"].mean(axis=0).tolist() for name in ensemble
    }
    report = {
        "model_id": MODEL_ID,
        "status": "research_shadow_candidate" if active_and_positive else "research_no_viable_active_policy",
        "model_family": "three-seed ensemble of dual causal encoder / three-regime-expert inventory Q policies",
        "device": str(device),
        "seeds": list(seeds),
        "expert_count": expert_count,
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "switch_rule": "mean Q action plus tune-selected consensus across nine expert heads",
        },
        "config": asdict(config),
        "feature_contract": {
            "base_features": frame["base_names"],
            "micro_features": frame["micro_names"],
            "btc_features_used": False,
            "rule_outputs_used": False,
            "trade_ledgers_used": False,
            "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        },
        "training_histories": histories,
        "per_seed_regime_gate_mean_weights": gate_means,
        "selected_research_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": active_and_positive,
        "tune": results["tune"],
        "tune_cost_stress": stresses["tune"],
        "validation": results["validation"],
        "validation_cost_stress": stresses["validation"],
        "development": results["development"],
        "development_cost_stress": stresses["development"],
        "top_tune_candidates": candidates[:15],
        "artifacts": {
            "model": str(MODEL_PATH),
            "validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "core_script_sha256": _sha256(Path(core.__file__)),
            "metadata_sha256": _sha256(core.CACHE_DIR / "metadata.json"),
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "future_path_used_only_for_fit_teacher_target": True,
            "fixed_holding_period_used": False,
            "outer_results_used_for_policy_selection": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": reason,
            "next_untouched_start": "after this 2026-07-18 ensemble freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=core._json_default))
    print(json.dumps({
        "selected_policy": asdict(policy),
        "activation_allowed": active_and_positive,
        "tune": results["tune"],
        "validation": results["validation"],
        "development": results["development"],
    }, indent=2, default=core._json_default))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    config = replace(core.Config(), epochs=1, batch_size=1024) if args.smoke else core.Config()
    seeds = SEEDS[:2] if args.smoke else SEEDS
    run(config, seeds)
