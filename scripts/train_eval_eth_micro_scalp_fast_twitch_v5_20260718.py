"""Train source-stable v5 with a causal fast-twitch action residual.

v4's fit teacher changes inventory quickly, but its 60-minute encoders smooth
those changes into multi-hour holds.  v5 freezes the two causal feature
encoders and adds a zero-initialized residual head over current,
one-minute-delta, and five-minute-delta inputs.  The downstream experts and
action heads adapt with a switch-weighted loss, without adding a fixed or
maximum holding period.
"""

from __future__ import annotations

import argparse
import hashlib
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

import train_eval_eth_micro_scalp_source_stable_v4_20260718 as v4  # noqa: E402


v3 = v4.v3
MODEL_ID = "eth_micro_scalp_source_stable_fast_twitch_v5_20260718"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "ensemble.pt"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "historical_validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "historical_development_diagnostic_ledger.csv"
CONTRACT_PATH = ROOT / "docs/model_contracts/eth_micro_scalp_fast_twitch_v5_20260718_contract.md"
SEEDS = v4.SEEDS


@dataclass(frozen=True)
class FastTwitchConfig(v4.SourceStableConfig):
    epochs: int = 6
    learning_rate: float = 0.0001
    action_loss_weight: float = 1.5
    switch_target_weight: float = 2.0
    fast_residual_l2_weight: float = 0.001


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FastTwitchOpportunityMoE(v3.OpportunityCostMoE):
    """Opportunity-MoE plus a shared causal short-horizon Q residual."""

    def __init__(self, n_base: int, n_micro: int, n_aux: int, config: FastTwitchConfig):
        super().__init__(n_base, n_micro, n_aux, config)
        fast_input = 3 * (n_base + n_micro)
        self.fast_q_head = nn.Sequential(
            nn.Linear(fast_input, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(64, 9),
        )
        nn.init.zeros_(self.fast_q_head[-1].weight)
        nn.init.zeros_(self.fast_q_head[-1].bias)

    @staticmethod
    def _fast_inputs(base: torch.Tensor, micro: torch.Tensor) -> torch.Tensor:
        if base.shape[1] < 6 or micro.shape[1] < 6:
            raise ValueError("fast-twitch head requires at least six causal rows")
        current = torch.cat([base[:, -1], micro[:, -1]], dim=-1)
        delta_1m = torch.cat(
            [base[:, -1] - base[:, -2], micro[:, -1] - micro[:, -2]], dim=-1
        )
        delta_5m = torch.cat(
            [base[:, -1] - base[:, -6], micro[:, -1] - micro[:, -6]], dim=-1
        )
        return torch.cat([current, delta_1m, delta_5m], dim=-1)

    def forward(self, base: torch.Tensor, micro: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(base, micro)
        residual = self.fast_q_head(self._fast_inputs(base, micro)).reshape(-1, 3, 3)
        output["slow_q"] = output["q"]
        output["fast_q_residual"] = residual
        output["q"] = output["q"] + residual
        output["expert_q"] = output["expert_q"] + residual[:, None]
        return output


def load_v4_weights(
    model: FastTwitchOpportunityMoE,
    state: dict[str, torch.Tensor],
) -> list[str]:
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    if unexpected or not missing or any(not name.startswith("fast_q_head.") for name in missing):
        raise RuntimeError(
            f"v4 warm-start contract mismatch: missing={missing}, unexpected={unexpected}"
        )
    return missing


def weighted_action_loss(
    q_values: torch.Tensor,
    action_target: torch.Tensor,
    switch_target_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    per_state = F.cross_entropy(
        q_values.reshape(-1, 3), action_target.reshape(-1), reduction="none"
    ).reshape_as(action_target)
    previous = torch.arange(3, device=action_target.device).reshape(1, 3)
    switch = action_target != previous
    weights = torch.where(
        switch,
        torch.full_like(per_state, switch_target_weight),
        torch.ones_like(per_state),
    )
    return torch.sum(per_state * weights) / torch.sum(weights), switch


ADAPTER_TRAINABLE_PREFIXES = (
    "regime_gate.",
    "experts.",
    "position_embedding.",
    "q_head.",
    "auxiliary_head.",
    "continuation_head.",
    "exit_hazard_head.",
    "fast_q_head.",
)


def configure_adapter_training(model: FastTwitchOpportunityMoE) -> list[nn.Parameter]:
    trainable: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(ADAPTER_TRAINABLE_PREFIXES)
        if parameter.requires_grad:
            trainable.append(parameter)
    if not trainable:
        raise RuntimeError("fast-twitch adapter has no trainable parameters")
    return trainable


def set_adapter_train_mode(model: FastTwitchOpportunityMoE) -> None:
    model.eval()
    for module in (
        model.regime_gate,
        model.experts,
        model.q_head,
        model.auxiliary_head,
        model.continuation_head,
        model.exit_hazard_head,
        model.fast_q_head,
    ):
        module.train()


def train_model(
    model: FastTwitchOpportunityMoE,
    prepared: dict[str, Any],
    continuation_target: np.ndarray,
    exit_target: np.ndarray,
    config: FastTwitchConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = v3.OpportunityDataset(
        prepared["base"], prepared["micro"], prepared["teacher_q"],
        prepared["teacher_action"], prepared["auxiliary"], continuation_target,
        exit_target, prepared["train_indices"], config.window,
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
    )
    trainable = configure_adapter_training(model)
    optimizer = torch.optim.AdamW(
        trainable, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    history: list[dict[str, float]] = []
    set_adapter_train_mode(model)
    for epoch in range(config.epochs):
        totals = {
            "loss": 0.0, "q": 0.0, "expert_q": 0.0, "action": 0.0,
            "auxiliary": 0.0, "continuation": 0.0,
            "expert_continuation": 0.0, "hazard": 0.0,
            "gate_balance": 0.0, "fast_residual_l2": 0.0,
            "switch_accuracy": 0.0, "batches": 0.0,
        }
        for xb, xm, yq, ya, yu, yc, ye in loader:
            xb, xm, yq, ya, yu, yc, ye = (
                tensor.to(device, non_blocking=True)
                for tensor in (xb, xm, yq, ya, yu, yc, ye)
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, xm)
            action_loss, switch_mask = weighted_action_loss(
                output["q"], ya, config.switch_target_weight
            )
            losses = {
                "q": F.smooth_l1_loss(output["q"], yq),
                "expert_q": F.smooth_l1_loss(
                    output["expert_q"], yq[:, None].expand_as(output["expert_q"])
                ),
                "action": action_loss,
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
            fast_l2 = torch.mean(output["fast_q_residual"] ** 2)
            loss = (
                config.q_loss_weight * losses["q"]
                + config.expert_q_loss_weight * losses["expert_q"]
                + config.action_loss_weight * losses["action"]
                + config.auxiliary_loss_weight * losses["auxiliary"]
                + config.continuation_loss_weight * losses["continuation"]
                + config.expert_continuation_loss_weight * losses["expert_continuation"]
                + config.exit_hazard_loss_weight * losses["hazard"]
                + config.gate_balance_weight * gate_balance
                + config.fast_residual_l2_weight * fast_l2
            )
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, config.grad_clip)
            optimizer.step()
            predicted = torch.argmax(output["q"], dim=-1)
            switch_correct = (predicted[switch_mask] == ya[switch_mask]).float().mean()
            totals["loss"] += float(loss.detach())
            for name, value in losses.items():
                totals[name] += float(value.detach())
            totals["gate_balance"] += float(gate_balance.detach())
            totals["fast_residual_l2"] += float(fast_l2.detach())
            totals["switch_accuracy"] += float(switch_correct.detach())
            totals["batches"] += 1.0
        row = {
            name: totals[name] / max(totals["batches"], 1.0)
            for name in totals if name != "batches"
        }
        row["epoch"] = float(epoch + 1)
        history.append(row)
        print(
            f"epoch={epoch + 1} loss={row['loss']:.4f} q={row['q']:.4f} "
            f"action={row['action']:.4f} switch_acc={row['switch_accuracy']:.4f} "
            f"fast_l2={row['fast_residual_l2']:.4f}",
            flush=True,
        )
    return history


@torch.no_grad()
def infer(
    model: FastTwitchOpportunityMoE,
    base: np.ndarray,
    micro: np.ndarray,
    end_indices: np.ndarray,
    config: FastTwitchConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    return v3.infer(model, base, micro, end_indices, config, device)


def _comparison(parent_report: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    old_keys = {
        "tune": "tune",
        "validation": "historical_validation_diagnostic",
        "development": "historical_development_diagnostic",
    }
    output: dict[str, Any] = {
        "evidence_class": "consumed diagnostics; never used for v5 selection"
    }
    for name, old_key in old_keys.items():
        old = parent_report[old_key]
        new = results[name]
        output[name] = {
            "v4_return_pct": old["compounded_return_pct"],
            "v5_return_pct": new["compounded_return_pct"],
            "return_delta_pct_points": new["compounded_return_pct"] - old["compounded_return_pct"],
            "v4_median_holding_minutes": old["holding_bars"]["median"],
            "v5_median_holding_minutes": new["holding_bars"]["median"],
            "v4_max_holding_minutes": old["holding_bars"]["max"],
            "v5_max_holding_minutes": new["holding_bars"]["max"],
        }
    return output


def run(config: FastTwitchConfig, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    source_audit = v4.audit_source_stability()
    parent_checkpoint = torch.load(v4.MODEL_PATH, map_location="cpu", weights_only=False)
    if parent_checkpoint.get("model_id") != v4.MODEL_ID:
        raise RuntimeError("v4 parent model id mismatch")
    if tuple(parent_checkpoint.get("seeds", ())) != tuple(seeds):
        raise RuntimeError("v5 requires all exact v4 trained seeds")
    if tuple(parent_checkpoint["base_feature_names"]) != v4.SOURCE_STABLE_FEATURES:
        raise RuntimeError("v4 source-stable feature contract mismatch")
    prepared = v4.prepare_source_stable_data(config)
    continuation_target, exit_target = v3.build_opportunity_targets(prepared["teacher_q"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states: dict[str, dict[str, torch.Tensor]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    predictions: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in prepared["split_indices"]
    }
    warm_start_missing: dict[str, list[str]] = {}
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        v3.core.seed_everything(seed)
        state = parent_checkpoint["seed_model_states"][str(seed)]
        model = FastTwitchOpportunityMoE(
            len(v4.SOURCE_STABLE_FEATURES), len(v3.core.MICRO_FEATURES),
            int(state["auxiliary_head.2.weight"].shape[0]), seed_config,
        ).to(device)
        warm_start_missing[str(seed)] = load_v4_weights(model, state)
        print(f"seed={seed} frozen-encoder fast-adapter training", flush=True)
        histories[str(seed)] = train_model(
            model, prepared, continuation_target, exit_target, seed_config, device
        )
        for name, indices in prepared["split_indices"].items():
            predictions[name].append(
                infer(model, prepared["base"], prepared["micro"], indices, seed_config, device)
            )
        states[str(seed)] = {
            name: value.detach().cpu() for name, value in model.state_dict().items()
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = {
        name: v3.aggregate_seed_predictions(rows) for name, rows in predictions.items()
    }
    frame = prepared["frame"]
    tune_indices = prepared["split_indices"]["tune"]
    policy, candidates = v3.select_policy(
        ensemble["fit"], ensemble["tune"], frame["available"][tune_indices],
        frame["next_return"][tune_indices], frame["timestamps"][tune_indices], config,
    )
    results: dict[str, Any] = {}
    stresses: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for name in ("tune", "validation", "development"):
        indices = prepared["split_indices"][name]
        metrics, ledger = v3.replay_policy(
            ensemble[name], frame["available"][indices], frame["next_return"][indices],
            frame["timestamps"][indices], policy, config.fee_per_notional_change,
        )
        results[name] = metrics
        ledgers[name] = ledger
        stresses[name] = v3._cost_stress(
            ledger["position"].to_numpy(dtype=np.int8), frame["next_return"][indices],
            frame["timestamps"][indices],
        )

    expert_count = len(seeds) * config.experts
    execution_policy = v3.OpportunityPolicy(
        False, 0.0, expert_count, False, 0.0, expert_count
    )
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "parent_model_id": v4.MODEL_ID,
        "seeds": list(seeds),
        "selected_ensemble_seeds": list(seeds),
        "seed_model_states": states,
        "config": asdict(config),
        "base_feature_names": list(v4.SOURCE_STABLE_FEATURES),
        "micro_feature_names": list(v3.core.MICRO_FEATURES),
        "source_stability_audit": source_audit,
        "scalers": prepared["scalers"],
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": False,
        "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        "parent_artifact_sha256": _sha256(v4.MODEL_PATH),
        "trainer_script_sha256": _sha256(Path(__file__)),
        "fixed_holding_period_used": False,
    }
    torch.save(checkpoint, MODEL_PATH)
    ledgers["validation"].to_csv(VALIDATION_LEDGER_PATH, index=False)
    ledgers["development"].to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    parent_report = json.loads(v4.REPORT_PATH.read_text())
    report = {
        "model_id": MODEL_ID,
        "status": "research_only_post_consumed_outer_architecture_freeze",
        "model_family": "frozen-encoder source-stable nine-expert Opportunity-MoE plus causal fast-twitch Q residual",
        "device": str(device),
        "seeds": list(seeds),
        "expert_count": expert_count,
        "parent": {
            "model_id": v4.MODEL_ID,
            "artifact": str(v4.MODEL_PATH),
            "artifact_sha256": _sha256(v4.MODEL_PATH),
            "warm_start_missing_new_head_keys": warm_start_missing,
        },
        "source_stability": source_audit,
        "architecture_change": {
            "fast_inputs": ["current", "one_minute_delta", "five_minute_delta"],
            "fast_input_width": 3 * (len(v4.SOURCE_STABLE_FEATURES) + len(v3.core.MICRO_FEATURES)),
            "fast_hidden_width": 64,
            "fast_outputs": "shared 3x3 inventory/action Q residual",
            "zero_initialized_parent_equivalence": True,
            "v4_base_and_micro_encoders_frozen": True,
            "adapter_trainable_prefixes": list(ADAPTER_TRAINABLE_PREFIXES),
            "switch_target_weight": config.switch_target_weight,
        },
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "holding_duration_feature_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "exit_rule": "slow inventory Q plus causal fast-twitch Q residual and expert consensus",
        },
        "config": asdict(config),
        "feature_contract": {
            "base_features": list(v4.SOURCE_STABLE_FEATURES),
            "micro_features": list(v3.core.MICRO_FEATURES),
            "trade_ledgers_used": False,
            "outer_results_used_for_training_or_selection": False,
        },
        "development_process": {
            "intermediate_joint_adaptation_outer_diagnostics_seen": True,
            "architecture_changed_after_intermediate_outer_diagnostics": True,
            "intermediate_fully_frozen_parent_failed_on_tune": True,
            "current_adapter_weights_trained_on_fit_only": True,
            "current_policy_selected_on_tune_only": True,
        },
        "training_histories": histories,
        "selected_research_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": False,
        "tune": results["tune"],
        "tune_cost_stress": stresses["tune"],
        "historical_validation_diagnostic": results["validation"],
        "historical_validation_cost_stress": stresses["validation"],
        "historical_development_diagnostic": results["development"],
        "historical_development_cost_stress": stresses["development"],
        "v4_comparison": _comparison(parent_report, results),
        "top_tune_candidates": candidates[:15],
        "artifacts": {
            "model": str(MODEL_PATH),
            "historical_validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "historical_development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
            "contract": str(CONTRACT_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "parent_artifact_sha256": _sha256(v4.MODEL_PATH),
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
            "outer_intervals_consumed_before_final_architecture_freeze": True,
            "all_three_seeds_used": True,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": "Historical outer intervals are consumed; a new post-v5-freeze fresh-forward run is required.",
            "next_untouched_start": "after this v5 artifact freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=v3.core._json_default))
    print(json.dumps({
        "selected_research_policy": asdict(policy),
        "activation_allowed": False,
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
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    config = (
        replace(FastTwitchConfig(), epochs=1, batch_size=1024)
        if args.smoke else FastTwitchConfig()
    )
    run(config, SEEDS)
