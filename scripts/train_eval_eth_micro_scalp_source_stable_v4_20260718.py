"""Train a source-stable Opportunity-MoE v4 from the frozen v3 ensemble.

Feature removal is selected only from post-freeze source-parity evidence, never
from return diagnostics.  The retained v3 channels, all nine experts, and both
opportunity heads are warm-started exactly; the full network is then adapted on
the original fit interval.  Tune alone selects the research policy.  Historical
validation/development remain consumed diagnostics and activation stays blocked.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_eth_micro_scalp_inventory_moe_ensemble_20260718 as parent  # noqa: E402
import train_eval_eth_micro_scalp_opportunity_moe_20260718 as v3  # noqa: E402


MODEL_ID = "eth_micro_scalp_source_stable_opportunity_moe_v4_20260718"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "ensemble.pt"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "historical_validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "historical_development_diagnostic_ledger.csv"
CONTRACT_PATH = ROOT / "docs/model_contracts/eth_micro_scalp_source_stable_v4_20260718_contract.md"
SOURCE_PARITY_REPORT_PATH = (
    v3.ARTIFACT_DIR / "fresh_forward_observer/feature_stream_build.json"
)
SEEDS = v3.SEEDS

SOURCE_UNSTABLE_FEATURES = (
    "whale_retail_ratio",
    "whale_conviction",
    "smart_money_flow",
    "squeeze_power",
    "oi_change_rate",
    "long_squeeze_risk",
    "short_squeeze_risk",
)
SOURCE_STABLE_FEATURES = tuple(
    name for name in v3.core.BASE_FEATURES if name not in SOURCE_UNSTABLE_FEATURES
)


@dataclass(frozen=True)
class SourceStableConfig(v3.OpportunityConfig):
    epochs: int = 4
    learning_rate: float = 0.0001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_source_stability(path: Path = SOURCE_PARITY_REPORT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"source-parity report is required: {path}")
    report = json.loads(path.read_text())
    parity = report.get("parity") or {}
    thresholds = parity.get("thresholds") or {}
    worst = parity.get("worst_features") or []
    names = [str(row["name"]) for row in worst]
    if len(worst) <= len(SOURCE_UNSTABLE_FEATURES):
        raise RuntimeError("source-parity report does not expose a retained-feature boundary")
    missing_evidence = [name for name in SOURCE_UNSTABLE_FEATURES if name not in names]
    if missing_evidence:
        raise RuntimeError(f"unstable feature evidence is incomplete: {missing_evidence}")
    max_threshold = float(thresholds["max_scaled_error"])
    p99_threshold = float(thresholds["p99_scaled_error"])
    retained_rows = [row for row in worst if row["name"] not in SOURCE_UNSTABLE_FEATURES]
    retained_max = max(float(row["max_scaled_error"]) for row in retained_rows)
    retained_p99 = max(float(row["p99_scaled_error"]) for row in retained_rows)
    sorted_maxima = [float(row["max_scaled_error"]) for row in worst]
    if sorted_maxima != sorted(sorted_maxima, reverse=True):
        raise RuntimeError("source-parity worst-feature rows are not ordered")
    passed = retained_max <= max_threshold and retained_p99 <= p99_threshold
    if not passed:
        raise RuntimeError(
            "retained feature source parity failed: "
            f"max={retained_max}/{max_threshold}, p99={retained_p99}/{p99_threshold}"
        )
    return {
        "pass": True,
        "selection_uses_return_metrics": False,
        "source_report": str(path),
        "source_report_sha256": _sha256(path),
        "source_report_full_contract_pass": bool(report.get("stream_contract_pass")),
        "source_report_original_parity_pass": bool(parity.get("pass")),
        "excluded_features": list(SOURCE_UNSTABLE_FEATURES),
        "retained_feature_count": len(SOURCE_STABLE_FEATURES),
        "retained_worst_max_scaled_error": retained_max,
        "retained_worst_p99_scaled_error": retained_p99,
        "thresholds": {
            "max_scaled_error": max_threshold,
            "p99_scaled_error": p99_threshold,
        },
    }


def prepare_source_stable_data(config: SourceStableConfig) -> dict[str, Any]:
    prepared = parent.prepare_data(config)
    names = list(prepared["frame"]["base_names"])
    if tuple(names) != tuple(v3.core.BASE_FEATURES):
        raise RuntimeError("v3 frozen base feature contract changed")
    indices = [names.index(name) for name in SOURCE_STABLE_FEATURES]
    prepared["base"] = np.ascontiguousarray(prepared["base"][:, indices])
    prepared["frame"]["base_raw"] = np.ascontiguousarray(
        prepared["frame"]["base_raw"][:, indices]
    )
    prepared["frame"]["base_names"] = list(SOURCE_STABLE_FEATURES)
    prepared["scalers"]["base_center"] = np.ascontiguousarray(
        prepared["scalers"]["base_center"][indices]
    )
    prepared["scalers"]["base_scale"] = np.ascontiguousarray(
        prepared["scalers"]["base_scale"][indices]
    )
    return prepared


def prune_v3_state(
    state: dict[str, torch.Tensor],
    original_names: tuple[str, ...] | list[str],
    retained_names: tuple[str, ...] = SOURCE_STABLE_FEATURES,
) -> dict[str, torch.Tensor]:
    if tuple(original_names) != tuple(v3.core.BASE_FEATURES):
        raise RuntimeError("warm-start feature contract mismatch")
    indices = [original_names.index(name) for name in retained_names]
    result = {name: value.detach().clone() for name, value in state.items()}
    key = "base_encoder.projection.weight"
    weight = result[key]
    if weight.ndim != 3 or weight.shape[1] != len(original_names):
        raise RuntimeError(f"unexpected v3 input projection shape: {tuple(weight.shape)}")
    result[key] = weight[:, indices, :].contiguous()
    return result


def load_pruned_v3_weights(
    model: v3.OpportunityCostMoE,
    state: dict[str, torch.Tensor],
    original_names: tuple[str, ...] | list[str],
) -> None:
    pruned = prune_v3_state(state, original_names)
    model.load_state_dict(pruned, strict=True)


def train_joint_model(
    model: v3.OpportunityCostMoE,
    prepared: dict[str, Any],
    continuation_target: np.ndarray,
    exit_target: np.ndarray,
    config: SourceStableConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = v3.OpportunityDataset(
        prepared["base"], prepared["micro"], prepared["teacher_q"],
        prepared["teacher_action"], prepared["auxiliary"], continuation_target,
        exit_target, prepared["train_indices"], config.window,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=generator,
    )
    for parameter in model.parameters():
        parameter.requires_grad = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(config.epochs):
        totals = {
            "loss": 0.0, "q": 0.0, "expert_q": 0.0, "action": 0.0,
            "auxiliary": 0.0, "continuation": 0.0,
            "expert_continuation": 0.0, "hazard": 0.0,
            "gate_balance": 0.0, "batches": 0.0,
        }
        for xb, xm, yq, ya, yu, yc, ye in loader:
            xb, xm, yq, ya, yu, yc, ye = (
                tensor.to(device, non_blocking=True)
                for tensor in (xb, xm, yq, ya, yu, yc, ye)
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, xm)
            losses = {
                "q": F.smooth_l1_loss(output["q"], yq),
                "expert_q": F.smooth_l1_loss(
                    output["expert_q"], yq[:, None].expand_as(output["expert_q"])
                ),
                "action": F.cross_entropy(output["q"].reshape(-1, 3), ya.reshape(-1)),
                "auxiliary": F.smooth_l1_loss(output["auxiliary"], yu),
                "continuation": F.smooth_l1_loss(output["continuation"], yc),
                "expert_continuation": F.smooth_l1_loss(
                    output["expert_continuation"],
                    yc[:, None].expand_as(output["expert_continuation"]),
                ),
                "hazard": F.binary_cross_entropy_with_logits(output["exit_logit"], ye),
            }
            mean_gate = output["gate"].mean(dim=0)
            gate_balance = torch.sum(
                mean_gate * torch.log(mean_gate * config.experts + 1e-8)
            )
            loss = (
                config.q_loss_weight * losses["q"]
                + config.expert_q_loss_weight * losses["expert_q"]
                + config.action_loss_weight * losses["action"]
                + config.auxiliary_loss_weight * losses["auxiliary"]
                + config.continuation_loss_weight * losses["continuation"]
                + config.expert_continuation_loss_weight * losses["expert_continuation"]
                + config.exit_hazard_loss_weight * losses["hazard"]
                + config.gate_balance_weight * gate_balance
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            totals["loss"] += float(loss.detach())
            for name, value in losses.items():
                totals[name] += float(value.detach())
            totals["gate_balance"] += float(gate_balance.detach())
            totals["batches"] += 1.0
        row = {
            name: totals[name] / max(totals["batches"], 1.0)
            for name in totals if name != "batches"
        }
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print(
            f"epoch={epoch + 1} loss={row['loss']:.4f} q={row['q']:.4f} "
            f"action={row['action']:.4f} continuation={row['continuation']:.4f}",
            flush=True,
        )
    return history


def _diagnostic_comparison(v3_report: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    old_names = {
        "tune": "tune",
        "validation": "historical_validation_diagnostic",
        "development": "historical_development_diagnostic",
    }
    comparison: dict[str, Any] = {
        "evidence_class": "consumed historical diagnostics; not selection or promotion evidence"
    }
    for name, old_name in old_names.items():
        old = v3_report[old_name]
        new = results[name]
        comparison[name] = {
            "v3_return_pct": old["compounded_return_pct"],
            "v4_return_pct": new["compounded_return_pct"],
            "delta_pct_points": new["compounded_return_pct"] - old["compounded_return_pct"],
            "v3_max_drawdown_pct": old["max_drawdown_pct"],
            "v4_max_drawdown_pct": new["max_drawdown_pct"],
        }
    return comparison


def select_seed_subset(
    seed_predictions: dict[str, list[dict[str, np.ndarray]]],
    seeds: tuple[int, ...],
    prepared: dict[str, Any],
    config: SourceStableConfig,
) -> tuple[
    tuple[int, ...], dict[str, dict[str, np.ndarray]], v3.OpportunityPolicy,
    list[dict[str, Any]], list[dict[str, Any]],
]:
    if len(seeds) < 2:
        subsets = [tuple(seeds)]
    else:
        subsets = [
            subset
            for size in range(2, len(seeds) + 1)
            for subset in itertools.combinations(seeds, size)
        ]
    seed_to_index = {seed: index for index, seed in enumerate(seeds)}
    frame = prepared["frame"]
    tune_indices = prepared["split_indices"]["tune"]
    subset_rows: list[dict[str, Any]] = []
    evaluated: dict[
        tuple[int, ...],
        tuple[dict[str, dict[str, np.ndarray]], v3.OpportunityPolicy, list[dict[str, Any]]],
    ] = {}
    for subset in subsets:
        ensemble = {
            name: v3.aggregate_seed_predictions(
                [rows[seed_to_index[seed]] for seed in subset]
            )
            for name, rows in seed_predictions.items()
        }
        policy, candidates = v3.select_policy(
            ensemble["fit"], ensemble["tune"], frame["available"][tune_indices],
            frame["next_return"][tune_indices], frame["timestamps"][tune_indices], config,
        )
        metrics, _ = v3.replay_policy(
            ensemble["tune"], frame["available"][tune_indices],
            frame["next_return"][tune_indices], frame["timestamps"][tune_indices],
            policy, config.fee_per_notional_change,
        )
        net = metrics["compounded_return_pct"] / 100.0
        drawdown = metrics["max_drawdown_pct"] / 100.0
        eligible = bool(
            policy.enabled
            and metrics["entries_or_reversals"] >= config.min_tune_switches
            and net > 0.0
        )
        score = net - 0.25 * drawdown if eligible else float("-inf")
        subset_rows.append({
            "seeds": list(subset),
            "expert_count": len(subset) * config.experts,
            "eligible": eligible,
            "selection_score": score,
            "policy": asdict(policy),
            "tune_metrics": metrics,
        })
        evaluated[subset] = (ensemble, policy, candidates)
    subset_rows.sort(
        key=lambda row: (row["selection_score"], row["expert_count"]), reverse=True
    )
    if subset_rows and np.isfinite(subset_rows[0]["selection_score"]):
        selected = tuple(int(seed) for seed in subset_rows[0]["seeds"])
        ensemble, policy, candidates = evaluated[selected]
        return selected, ensemble, policy, candidates, subset_rows
    selected = tuple(seeds)
    ensemble, _, candidates = evaluated[selected]
    expert_count = len(selected) * config.experts
    disabled = v3.OpportunityPolicy(
        False, 0.0, expert_count, False, 0.0, expert_count
    )
    return selected, ensemble, disabled, candidates, subset_rows


def run(
    config: SourceStableConfig,
    seeds: tuple[int, ...] = SEEDS,
    reuse_checkpoint: bool = False,
) -> dict[str, Any]:
    source_audit = audit_source_stability()
    if not v3.MODEL_PATH.exists() or not v3.REPORT_PATH.exists():
        raise FileNotFoundError("frozen v3 model and report are required")
    checkpoint_v3 = torch.load(v3.MODEL_PATH, map_location="cpu", weights_only=False)
    if checkpoint_v3.get("model_id") != v3.MODEL_ID:
        raise RuntimeError("v3 model id mismatch")
    parent_seeds = tuple(checkpoint_v3.get("seeds", ()))
    if not seeds or tuple(seeds) != parent_seeds[: len(seeds)]:
        raise RuntimeError("v4 seeds must be an ordered prefix of the exact v3 seed set")
    if tuple(checkpoint_v3.get("base_feature_names", ())) != tuple(v3.core.BASE_FEATURES):
        raise RuntimeError("v3 artifact base feature contract mismatch")
    prepared = prepare_source_stable_data(config)
    continuation_target, exit_target = v3.build_opportunity_targets(prepared["teacher_q"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    existing_checkpoint: dict[str, Any] | None = None
    existing_report: dict[str, Any] | None = None
    if reuse_checkpoint:
        if not MODEL_PATH.exists() or not REPORT_PATH.exists():
            raise FileNotFoundError("v4 checkpoint and report are required for re-selection")
        existing_checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        existing_report = json.loads(REPORT_PATH.read_text())
        if tuple(existing_checkpoint.get("seeds", ())) != tuple(seeds):
            raise RuntimeError("re-selection requires the exact trained seed set")
    states: dict[str, dict[str, torch.Tensor]] = (
        existing_checkpoint["seed_model_states"] if existing_checkpoint else {}
    )
    histories: dict[str, list[dict[str, float]]] = (
        existing_report["training_histories"] if existing_report else {}
    )
    predictions: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in prepared["split_indices"]
    }
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        v3.core.seed_everything(seed)
        model = v3.OpportunityCostMoE(
            prepared["base"].shape[1], prepared["micro"].shape[1],
            prepared["auxiliary"].shape[1], seed_config,
        ).to(device)
        if existing_checkpoint:
            model.load_state_dict(existing_checkpoint["seed_model_states"][str(seed)], strict=True)
            print(f"seed={seed} source-stable re-selection inference", flush=True)
        else:
            load_pruned_v3_weights(
                model, checkpoint_v3["seed_model_states"][str(seed)],
                checkpoint_v3["base_feature_names"],
            )
            print(f"seed={seed} source-stable joint adaptation", flush=True)
            histories[str(seed)] = train_joint_model(
                model, prepared, continuation_target, exit_target, seed_config, device
            )
        for name, indices in prepared["split_indices"].items():
            predictions[name].append(
                v3.infer(model, prepared["base"], prepared["micro"], indices, seed_config, device)
            )
        if not existing_checkpoint:
            states[str(seed)] = {
                name: value.detach().cpu() for name, value in model.state_dict().items()
            }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    frame = prepared["frame"]
    timestamps = frame["timestamps"]
    selected_seeds, ensemble, policy, candidates, subset_candidates = select_seed_subset(
        predictions, seeds, prepared, config,
    )
    results: dict[str, Any] = {}
    stresses: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for name in ("tune", "validation", "development"):
        indices = prepared["split_indices"][name]
        metrics, ledger = v3.replay_policy(
            ensemble[name], frame["available"][indices], frame["next_return"][indices],
            timestamps[indices], policy, config.fee_per_notional_change,
        )
        results[name] = metrics
        ledgers[name] = ledger
        stresses[name] = v3._cost_stress(
            ledger["position"].to_numpy(dtype=np.int8),
            frame["next_return"][indices], timestamps[indices],
        )

    expert_count = len(selected_seeds) * config.experts
    execution_policy = v3.OpportunityPolicy(
        False, 0.0, expert_count, False, 0.0, expert_count
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "parent_model_id": v3.MODEL_ID,
        "seeds": list(seeds),
        "selected_ensemble_seeds": list(selected_seeds),
        "seed_model_states": states,
        "config": asdict(config),
        "base_feature_names": list(SOURCE_STABLE_FEATURES),
        "micro_feature_names": frame["micro_names"],
        "removed_source_unstable_features": list(SOURCE_UNSTABLE_FEATURES),
        "source_stability_audit": source_audit,
        "scalers": prepared["scalers"],
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": False,
        "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        "parent_artifact_sha256": _sha256(v3.MODEL_PATH),
        "trainer_script_sha256": _sha256(Path(__file__)),
        "fixed_holding_period_used": False,
    }
    torch.save(checkpoint, MODEL_PATH)
    ledgers["validation"].to_csv(VALIDATION_LEDGER_PATH, index=False)
    ledgers["development"].to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    report_v3 = json.loads(v3.REPORT_PATH.read_text())
    report = {
        "model_id": MODEL_ID,
        "status": "research_only_source_stable_consumed_outer_intervals",
        "model_family": "source-pruned nine-expert Opportunity-MoE with joint fit-only adaptation",
        "device": str(device),
        "seeds": list(seeds),
        "selected_ensemble_seeds": list(selected_seeds),
        "expert_count": expert_count,
        "parent": {
            "model_id": v3.MODEL_ID,
            "artifact": str(v3.MODEL_PATH),
            "artifact_sha256": _sha256(v3.MODEL_PATH),
            "warm_start_method": "exact retained-channel projection plus strict full-state load",
            "joint_adaptation_scope": "all parameters on fit only",
        },
        "source_stability": source_audit,
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "holding_duration_feature_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "exit_rule": "inventory Q plus learned continuation opportunity cost and expert consensus",
        },
        "config": asdict(config),
        "feature_contract": {
            "base_features": list(SOURCE_STABLE_FEATURES),
            "micro_features": frame["micro_names"],
            "removed_source_unstable_features": list(SOURCE_UNSTABLE_FEATURES),
            "removal_selected_from_return_metrics": False,
            "trade_ledgers_used": False,
            "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        },
        "data": {
            "splits": {
                "fit": [config.fit_start, config.tune_start],
                "tune": [config.tune_start, config.validation_start],
                "historical_validation_diagnostic": [config.validation_start, config.development_start],
                "historical_development_diagnostic": [config.development_start, config.development_end],
            },
            "outer_interval_evidence_class": "consumed-development; not promotion evidence",
            "split_window_counts": {
                name: int(len(indices)) for name, indices in prepared["split_indices"].items()
            },
        },
        "training_histories": histories,
        "tune_seed_subset_candidates": subset_candidates,
        "selected_research_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": False,
        "tune": results["tune"],
        "tune_cost_stress": stresses["tune"],
        "historical_validation_diagnostic": results["validation"],
        "historical_validation_cost_stress": stresses["validation"],
        "historical_development_diagnostic": results["development"],
        "historical_development_cost_stress": stresses["development"],
        "v3_comparison": _diagnostic_comparison(report_v3, results),
        "top_tune_candidates": candidates[:15],
        "artifacts": {
            "model": str(MODEL_PATH),
            "historical_validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "historical_development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
            "contract": str(CONTRACT_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "parent_artifact_sha256": _sha256(v3.MODEL_PATH),
            "source_parity_report_sha256": _sha256(SOURCE_PARITY_REPORT_PATH),
            "metadata_sha256": _sha256(v3.core.CACHE_DIR / "metadata.json"),
        },
        "compliance": {
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "future_path_used_only_for_fit_teacher_target": True,
            "fixed_holding_period_used": False,
            "outer_results_used_for_policy_selection": False,
            "source_feature_removal_used_return_results": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": "Historical outer intervals are consumed; a v4 exact-source post-freeze fresh-forward run is required.",
            "next_untouched_start": "after this v4 artifact freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=v3.core._json_default))
    print(json.dumps({
        "selected_research_policy": asdict(policy),
        "activation_allowed": False,
        "source_stability": source_audit,
        "tune": results["tune"],
        "historical_validation_diagnostic": results["validation"],
        "historical_development_diagnostic": results["development"],
    }, indent=2, default=v3.core._json_default))
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--reselect-only", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    if args.smoke and args.reselect_only:
        raise SystemExit("--smoke and --reselect-only are mutually exclusive")
    if args.reselect_only:
        saved = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        config = SourceStableConfig(**saved["config"])
    else:
        config = (
            replace(SourceStableConfig(), epochs=1, batch_size=1024)
            if args.smoke else SourceStableConfig()
        )
    seeds = SEEDS[:1] if args.smoke else SEEDS
    run(config, seeds, reuse_checkpoint=args.reselect_only)
