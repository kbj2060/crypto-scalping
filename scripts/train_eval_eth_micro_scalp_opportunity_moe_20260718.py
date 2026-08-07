"""Fine-tune Inventory-MoE with learned opportunity-cost exits.

The policy still decides SHORT/CASH/LONG on every completed minute without a
fixed or maximum holding period.  A dedicated head estimates the value of
continuing the current inventory relative to the best alternative.  A
tune-selected consensus overlay may therefore close an inventory even when the
base Q ensemble would otherwise keep holding it.

All available historical outer intervals have already influenced model
development.  They are diagnostics only; the saved execution policy is always
fail-safe CASH pending genuinely fresh-forward evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from torch import nn
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import train_eval_eth_micro_scalp_inventory_moe_20260718 as core  # noqa: E402
import train_eval_eth_micro_scalp_inventory_moe_ensemble_20260718 as parent  # noqa: E402


MODEL_ID = "eth_micro_scalp_opportunity_moe_v3_20260718"
ARTIFACT_DIR = ROOT / f"data/ensemble/{MODEL_ID}"
MODEL_PATH = ARTIFACT_DIR / "ensemble.pt"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}.json"
VALIDATION_LEDGER_PATH = ARTIFACT_DIR / "historical_validation_diagnostic_ledger.csv"
DEVELOPMENT_LEDGER_PATH = ARTIFACT_DIR / "historical_development_diagnostic_ledger.csv"
CONTRACT_PATH = ROOT / "docs/model_contracts/eth_micro_scalp_opportunity_moe_v3_20260718_contract.md"
PARENT_REPORT_PATH = parent.REPORT_PATH
SEEDS = parent.SEEDS


@dataclass(frozen=True)
class OpportunityConfig(core.Config):
    epochs: int = 8
    continuation_loss_weight: float = 0.35
    expert_continuation_loss_weight: float = 0.10
    exit_hazard_loss_weight: float = 0.15


@dataclass(frozen=True)
class OpportunityPolicy:
    enabled: bool
    switch_margin_bp: float
    min_switch_agreement: int
    exit_overlay_enabled: bool
    continuation_floor_bp: float
    min_exit_agreement: int
    uncertainty_penalty: float = 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_opportunity_targets(teacher_q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return hold-vs-alternative advantage and a binary exit target."""
    q = np.asarray(teacher_q, dtype=np.float32)
    if q.ndim != 3 or q.shape[1:] != (3, 3):
        raise ValueError(f"teacher Q must have shape [rows, 3, 3], got {q.shape}")
    continuation = np.empty((len(q), 3), dtype=np.float32)
    exit_target = np.empty((len(q), 3), dtype=np.float32)
    for previous_idx in range(3):
        alternatives = [idx for idx in range(3) if idx != previous_idx]
        best_alternative = np.max(q[:, previous_idx, alternatives], axis=1)
        continuation[:, previous_idx] = q[:, previous_idx, previous_idx] - best_alternative
        exit_target[:, previous_idx] = (
            np.argmax(q[:, previous_idx], axis=1) != previous_idx
        ).astype(np.float32)
    return continuation, exit_target


class OpportunityCostMoE(nn.Module):
    """Parent-compatible Inventory-MoE plus continuation and exit-hazard heads."""

    def __init__(self, n_base: int, n_micro: int, n_aux: int, config: OpportunityConfig):
        super().__init__()
        self.base_encoder = core.CausalBranchEncoder(
            n_base, config.base_channels, (1, 2, 4, 8), config.dropout
        )
        self.micro_encoder = core.CausalBranchEncoder(
            n_micro, config.micro_channels, (1, 2, 4), config.dropout
        )
        fused_dim = config.base_channels + config.micro_channels
        self.regime_gate = nn.Sequential(
            nn.Linear(fused_dim, 48), nn.GELU(), nn.Linear(48, config.experts)
        )
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(fused_dim, 96), nn.LayerNorm(96), nn.GELU(),
                    nn.Dropout(config.dropout), nn.Linear(96, config.latent_dim), nn.GELU(),
                )
                for _ in range(config.experts)
            ]
        )
        self.position_embedding = nn.Embedding(3, 12)
        state_dim = config.latent_dim + 12
        self.q_head = nn.Sequential(
            nn.Linear(state_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Linear(64, 3)
        )
        self.auxiliary_head = nn.Sequential(
            nn.Linear(config.latent_dim, 64), nn.GELU(), nn.Linear(64, n_aux)
        )
        self.continuation_head = nn.Sequential(
            nn.Linear(state_dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, 1)
        )
        self.exit_hazard_head = nn.Sequential(
            nn.Linear(state_dim, 48), nn.LayerNorm(48), nn.GELU(), nn.Linear(48, 1)
        )

    def forward(self, base: torch.Tensor, micro: torch.Tensor) -> dict[str, torch.Tensor]:
        fused = torch.cat([self.base_encoder(base), self.micro_encoder(micro)], dim=-1)
        gate = torch.softmax(self.regime_gate(fused), dim=-1)
        expert_values = torch.stack([expert(fused) for expert in self.experts], dim=1)
        latent = torch.sum(expert_values * gate.unsqueeze(-1), dim=1)
        position_ids = torch.arange(3, device=latent.device)
        position = self.position_embedding(position_ids).unsqueeze(0).expand(len(latent), -1, -1)
        expert_state = torch.cat(
            [
                expert_values.unsqueeze(2).expand(-1, -1, 3, -1),
                position.unsqueeze(1).expand(-1, len(self.experts), -1, -1),
            ],
            dim=-1,
        )
        expert_q = self.q_head(expert_state)
        expert_continuation = self.continuation_head(expert_state).squeeze(-1)
        expert_exit_logit = self.exit_hazard_head(expert_state).squeeze(-1)
        return {
            "q": torch.sum(expert_q * gate[:, :, None, None], dim=1),
            "auxiliary": self.auxiliary_head(latent),
            "gate": gate,
            "expert_q": expert_q,
            "continuation": torch.sum(expert_continuation * gate[:, :, None], dim=1),
            "expert_continuation": expert_continuation,
            "exit_logit": torch.sum(expert_exit_logit * gate[:, :, None], dim=1),
            "expert_exit_logit": expert_exit_logit,
        }


def load_parent_weights(model: OpportunityCostMoE, state: dict[str, torch.Tensor]) -> list[str]:
    incompatible = model.load_state_dict(state, strict=False)
    missing = sorted(incompatible.missing_keys)
    unexpected = sorted(incompatible.unexpected_keys)
    allowed_prefixes = ("continuation_head.", "exit_hazard_head.")
    if unexpected or not missing or any(not key.startswith(allowed_prefixes) for key in missing):
        raise RuntimeError(
            f"parent model contract mismatch: missing={missing}, unexpected={unexpected}"
        )
    return missing


def freeze_parent_parameters(model: OpportunityCostMoE) -> list[str]:
    trainable_prefixes = ("continuation_head.", "exit_hazard_head.")
    trainable: list[str] = []
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith(trainable_prefixes)
        if parameter.requires_grad:
            trainable.append(name)
    if not trainable or any(not name.startswith(trainable_prefixes) for name in trainable):
        raise RuntimeError(f"invalid opportunity-head parameter set: {trainable}")
    return trainable


class OpportunityDataset(Dataset):
    def __init__(
        self,
        base: np.ndarray,
        micro: np.ndarray,
        q_target: np.ndarray,
        action_target: np.ndarray,
        auxiliary: np.ndarray,
        continuation_target: np.ndarray,
        exit_target: np.ndarray,
        end_indices: np.ndarray,
        window: int,
    ):
        self.base_windows = sliding_window_view(base, window_shape=window, axis=0)
        self.micro_windows = sliding_window_view(micro, window_shape=window, axis=0)
        self.q_target = q_target
        self.action_target = action_target
        self.auxiliary = auxiliary
        self.continuation_target = continuation_target
        self.exit_target = exit_target
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.window = window

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, item: int) -> tuple[torch.Tensor, ...]:
        end = int(self.end_indices[item])
        start = end - self.window + 1
        return (
            torch.from_numpy(np.ascontiguousarray(self.base_windows[start].T)),
            torch.from_numpy(np.ascontiguousarray(self.micro_windows[start].T)),
            torch.from_numpy(np.asarray(self.q_target[end], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.action_target[end], dtype=np.int64)),
            torch.from_numpy(np.asarray(self.auxiliary[end], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.continuation_target[end], dtype=np.float32)),
            torch.from_numpy(np.asarray(self.exit_target[end], dtype=np.float32)),
        )


def train_model(
    model: OpportunityCostMoE,
    prepared: dict[str, Any],
    continuation_target: np.ndarray,
    exit_target: np.ndarray,
    config: OpportunityConfig,
    device: torch.device,
) -> list[dict[str, float]]:
    dataset = OpportunityDataset(
        prepared["base"], prepared["micro"], prepared["teacher_q"],
        prepared["teacher_action"], prepared["auxiliary"], continuation_target,
        exit_target, prepared["train_indices"], config.window,
    )
    generator = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=0,
        pin_memory=device.type == "cuda", generator=generator,
    )
    trainable_names = freeze_parent_parameters(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    model.eval()
    model.continuation_head.train()
    model.exit_hazard_head.train()
    for epoch in range(config.epochs):
        totals = {
            "loss": 0.0, "continuation": 0.0, "expert_continuation": 0.0,
            "hazard": 0.0, "batches": 0.0,
        }
        for xb, xm, yq, ya, yu, yc, ye in loader:
            xb, xm, yq, ya, yu, yc, ye = (
                tensor.to(device, non_blocking=True) for tensor in (xb, xm, yq, ya, yu, yc, ye)
            )
            optimizer.zero_grad(set_to_none=True)
            output = model(xb, xm)
            continuation_loss = F.smooth_l1_loss(output["continuation"], yc)
            expert_continuation_loss = F.smooth_l1_loss(
                output["expert_continuation"],
                yc[:, None].expand_as(output["expert_continuation"]),
            )
            hazard_loss = F.binary_cross_entropy_with_logits(output["exit_logit"], ye)
            loss = (
                config.continuation_loss_weight * continuation_loss
                + config.expert_continuation_loss_weight * expert_continuation_loss
                + config.exit_hazard_loss_weight * hazard_loss
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            values = {
                "loss": loss, "continuation": continuation_loss,
                "expert_continuation": expert_continuation_loss, "hazard": hazard_loss,
            }
            for key, value in values.items():
                totals[key] += float(value.detach())
            totals["batches"] += 1.0
        row = {
            key: totals[key] / max(totals["batches"], 1.0)
            for key in totals if key != "batches"
        }
        row["epoch"] = float(epoch + 1)
        history.append(row)
        row["trainable_parameter_names"] = trainable_names
        print(
            f"epoch={epoch + 1} loss={row['loss']:.4f} "
            f"continuation={row['continuation']:.4f} hazard={row['hazard']:.4f}",
            flush=True,
        )
    return history


@torch.no_grad()
def infer(
    model: OpportunityCostMoE,
    base: np.ndarray,
    micro: np.ndarray,
    end_indices: np.ndarray,
    config: OpportunityConfig,
    device: torch.device,
) -> dict[str, np.ndarray]:
    base_windows = sliding_window_view(base, window_shape=config.window, axis=0)
    micro_windows = sliding_window_view(micro, window_shape=config.window, axis=0)
    rows: dict[str, list[np.ndarray]] = {
        "q": [], "gate": [], "expert_q": [], "continuation": [],
        "expert_continuation": [], "exit_logit": [], "expert_exit_logit": [],
    }
    model.eval()
    for offset in range(0, len(end_indices), config.batch_size):
        indices = end_indices[offset : offset + config.batch_size]
        xb = np.stack(
            [base_windows[int(end) - config.window + 1].T for end in indices]
        ).astype(np.float32)
        xm = np.stack(
            [micro_windows[int(end) - config.window + 1].T for end in indices]
        ).astype(np.float32)
        output = model(torch.from_numpy(xb).to(device), torch.from_numpy(xm).to(device))
        for key in rows:
            rows[key].append(output[key].cpu().numpy())
    return {key: np.concatenate(values) for key, values in rows.items()}


def aggregate_seed_predictions(rows: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not rows:
        raise ValueError("at least one seed prediction is required")
    mean_keys = ("q", "continuation", "exit_logit")
    concat_keys = ("gate", "expert_q", "expert_continuation", "expert_exit_logit")
    result = {
        key: np.mean(np.stack([row[key] for row in rows]), axis=0).astype(np.float32)
        for key in mean_keys
    }
    result.update(
        {
            key: np.concatenate([row[key] for row in rows], axis=1).astype(np.float32)
            for key in concat_keys
        }
    )
    return result


def decide_positions(
    prediction: dict[str, np.ndarray],
    available: np.ndarray,
    policy: OpportunityPolicy,
) -> tuple[np.ndarray, np.ndarray]:
    q_values = np.asarray(prediction["q"], dtype=np.float64)
    expert_q = np.asarray(prediction["expert_q"], dtype=np.float64)
    continuation = np.asarray(prediction["continuation"], dtype=np.float64)
    expert_continuation = np.asarray(prediction["expert_continuation"], dtype=np.float64)
    usable = np.asarray(available, dtype=bool)
    if q_values.shape != (len(usable), 3, 3):
        raise ValueError("prediction and availability lengths must match")
    position = np.zeros(len(q_values), dtype=np.int8)
    opportunity_exit = np.zeros(len(q_values), dtype=bool)
    if not policy.enabled:
        return position, opportunity_exit
    previous_idx = 1
    for idx in range(len(q_values)):
        if not usable[idx] or not np.isfinite(q_values[idx]).all():
            action_idx = 1
        else:
            expert_state_q = expert_q[idx, :, previous_idx]
            state_q = (
                q_values[idx, previous_idx]
                - policy.uncertainty_penalty * np.std(expert_state_q, axis=0)
            )
            action_idx = int(np.argmax(state_q))
            improvement = float(state_q[action_idx] - state_q[previous_idx])
            if action_idx != previous_idx and improvement < policy.switch_margin_bp:
                action_idx = previous_idx
            if action_idx != previous_idx and policy.min_switch_agreement > 1:
                votes = np.argmax(expert_q[idx, :, previous_idx], axis=1)
                if int(np.sum(votes == action_idx)) < policy.min_switch_agreement:
                    action_idx = previous_idx
            if (
                policy.exit_overlay_enabled
                and previous_idx != 1
                and action_idx == previous_idx
                and np.isfinite(continuation[idx, previous_idx])
                and continuation[idx, previous_idx] < policy.continuation_floor_bp
            ):
                exit_votes = int(
                    np.sum(
                        expert_continuation[idx, :, previous_idx]
                        < policy.continuation_floor_bp
                    )
                )
                if exit_votes >= policy.min_exit_agreement:
                    alternatives = [candidate for candidate in range(3) if candidate != previous_idx]
                    action_idx = alternatives[int(np.argmax(state_q[alternatives]))]
                    opportunity_exit[idx] = True
        position[idx] = core.ACTIONS[action_idx]
        previous_idx = action_idx
    return position, opportunity_exit


def replay_policy(
    prediction: dict[str, np.ndarray],
    available: np.ndarray,
    next_return: np.ndarray,
    timestamps: pd.DatetimeIndex,
    policy: OpportunityPolicy,
    fee: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    position, opportunity_exit = decide_positions(prediction, available, policy)
    metrics, ledger = core.replay_positions(position, next_return, timestamps, fee)
    metrics["opportunity_exit_triggers"] = int(np.sum(opportunity_exit))
    ledger["available"] = np.asarray(available, dtype=bool)
    ledger["opportunity_exit_trigger"] = opportunity_exit
    return metrics, ledger


def select_policy(
    fit_prediction: dict[str, np.ndarray],
    tune_prediction: dict[str, np.ndarray],
    tune_available: np.ndarray,
    tune_returns: np.ndarray,
    tune_timestamps: pd.DatetimeIndex,
    config: OpportunityConfig,
) -> tuple[OpportunityPolicy, list[dict[str, Any]]]:
    base_policy, _ = core.select_q_policy(
        fit_prediction["q"], tune_prediction["q"], tune_prediction["expert_q"],
        tune_available, tune_returns, tune_timestamps, config,
    )
    expert_count = int(tune_prediction["expert_q"].shape[1])
    if not base_policy.enabled:
        return OpportunityPolicy(False, 0.0, expert_count, False, 0.0, expert_count), []
    baseline = OpportunityPolicy(
        True, base_policy.switch_margin_bp, base_policy.min_expert_agreement,
        False, 0.0, expert_count,
    )
    candidates: list[dict[str, Any]] = []

    def evaluate(policy: OpportunityPolicy) -> None:
        metrics, _ = replay_policy(
            tune_prediction, tune_available, tune_returns, tune_timestamps,
            policy, config.fee_per_notional_change,
        )
        net = metrics["compounded_return_pct"] / 100.0
        drawdown = metrics["max_drawdown_pct"] / 100.0
        eligible = metrics["entries_or_reversals"] >= config.min_tune_switches and net > 0.0
        score = net - 0.25 * drawdown if eligible else float("-inf")
        candidates.append(
            {"policy": asdict(policy), "eligible": bool(eligible), "selection_score": score, "metrics": metrics}
        )

    base_candidates: list[OpportunityPolicy] = [baseline]
    for uncertainty_penalty in (0.10, 0.25, 0.50, 1.00, 2.00):
        base_candidates.append(replace(baseline, uncertainty_penalty=uncertainty_penalty))
    for candidate in base_candidates:
        evaluate(candidate)
    eligible_base = [row for row in candidates if row["eligible"]]
    if not eligible_base:
        candidates.sort(key=lambda row: row["selection_score"], reverse=True)
        return OpportunityPolicy(False, 0.0, expert_count, False, 0.0, expert_count), candidates
    eligible_base.sort(key=lambda row: row["selection_score"], reverse=True)
    selected_base = OpportunityPolicy(**eligible_base[0]["policy"])
    fit_values = fit_prediction["continuation"][:, (0, 2)].reshape(-1)
    finite = fit_values[np.isfinite(fit_values)]
    if len(finite):
        floors = sorted(
            {
                round(float(value), 6)
                for value in np.r_[np.quantile(finite, (0.10, 0.25, 0.50, 0.75, 0.90)), 0.0]
            }
        )
        agreements = sorted(
            {
                max(1, math.ceil(expert_count * fraction))
                for fraction in (0.50, 2.0 / 3.0, 0.80, 1.0)
            }
        )
        for floor in floors:
            for agreement in agreements:
                evaluate(
                    OpportunityPolicy(
                        True, selected_base.switch_margin_bp, selected_base.min_switch_agreement,
                        True, floor, agreement, selected_base.uncertainty_penalty,
                    )
                )
    candidates.sort(key=lambda row: row["selection_score"], reverse=True)
    if not candidates or not np.isfinite(candidates[0]["selection_score"]):
        return OpportunityPolicy(False, 0.0, expert_count, False, 0.0, expert_count), candidates
    return OpportunityPolicy(**candidates[0]["policy"]), candidates


def _cost_stress(
    positions: np.ndarray, returns: np.ndarray, timestamps: pd.DatetimeIndex
) -> dict[str, Any]:
    return core.cost_stress(positions, returns, timestamps)


def _comparison(parent_report: dict[str, Any], results: dict[str, Any]) -> dict[str, Any]:
    comparison: dict[str, Any] = {"evidence_class": "consumed-development diagnostic only"}
    for name in ("tune", "validation", "development"):
        old = parent_report[name]
        new = results[name]
        comparison[name] = {
            "parent_return_pct": old["compounded_return_pct"],
            "v3_return_pct": new["compounded_return_pct"],
            "return_delta_pct_points": new["compounded_return_pct"] - old["compounded_return_pct"],
            "parent_median_holding_minutes": old["holding_bars"]["median"],
            "v3_median_holding_minutes": new["holding_bars"]["median"],
            "parent_max_holding_minutes": old["holding_bars"]["max"],
            "v3_max_holding_minutes": new["holding_bars"]["max"],
        }
    return comparison


def run(config: OpportunityConfig, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    if not parent.MODEL_PATH.exists() or not PARENT_REPORT_PATH.exists():
        raise FileNotFoundError("exact v2 parent model and report are required")
    parent_checkpoint = torch.load(parent.MODEL_PATH, map_location="cpu", weights_only=False)
    if parent_checkpoint.get("model_id") != parent.MODEL_ID:
        raise RuntimeError("parent model id mismatch")
    if tuple(parent_checkpoint.get("seeds", ())) != tuple(seeds):
        raise RuntimeError("v3 requires the exact parent seed set")
    parent_report = json.loads(PARENT_REPORT_PATH.read_text())
    prepared = parent.prepare_data(config)
    continuation_target, exit_target = build_opportunity_targets(prepared["teacher_q"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    states: dict[str, dict[str, torch.Tensor]] = {}
    histories: dict[str, list[dict[str, float]]] = {}
    seed_predictions: dict[str, list[dict[str, np.ndarray]]] = {
        name: [] for name in prepared["split_indices"]
    }
    warm_start_missing: dict[str, list[str]] = {}
    for seed in seeds:
        seed_config = replace(config, seed=seed)
        core.seed_everything(seed)
        model = OpportunityCostMoE(
            prepared["base"].shape[1], prepared["micro"].shape[1],
            prepared["auxiliary"].shape[1], seed_config,
        ).to(device)
        missing = load_parent_weights(model, parent_checkpoint["seed_model_states"][str(seed)])
        warm_start_missing[str(seed)] = missing
        print(f"seed={seed} opportunity fine-tuning", flush=True)
        histories[str(seed)] = train_model(
            model, prepared, continuation_target, exit_target, seed_config, device
        )
        for name, indices in prepared["split_indices"].items():
            seed_predictions[name].append(
                infer(model, prepared["base"], prepared["micro"], indices, seed_config, device)
            )
        states[str(seed)] = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ensemble = {
        name: aggregate_seed_predictions(rows) for name, rows in seed_predictions.items()
    }
    frame = prepared["frame"]
    timestamps = frame["timestamps"]
    tune_indices = prepared["split_indices"]["tune"]
    policy, candidates = select_policy(
        ensemble["fit"], ensemble["tune"], frame["available"][tune_indices],
        frame["next_return"][tune_indices], timestamps[tune_indices], config,
    )

    results: dict[str, Any] = {}
    stresses: dict[str, Any] = {}
    ledgers: dict[str, pd.DataFrame] = {}
    for name in ("tune", "validation", "development"):
        indices = prepared["split_indices"][name]
        metrics, ledger = replay_policy(
            ensemble[name], frame["available"][indices], frame["next_return"][indices],
            timestamps[indices], policy, config.fee_per_notional_change,
        )
        results[name] = metrics
        ledgers[name] = ledger
        stresses[name] = _cost_stress(
            ledger["position"].to_numpy(dtype=np.int8), frame["next_return"][indices], timestamps[indices]
        )

    expert_count = len(seeds) * config.experts
    execution_policy = OpportunityPolicy(False, 0.0, expert_count, False, 0.0, expert_count)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_id": MODEL_ID,
        "parent_model_id": parent.MODEL_ID,
        "seeds": list(seeds),
        "seed_model_states": states,
        "config": asdict(config),
        "base_feature_names": frame["base_names"],
        "micro_feature_names": frame["micro_names"],
        "scalers": prepared["scalers"],
        "policy": asdict(execution_policy),
        "selected_research_policy": asdict(policy),
        "activation_allowed": False,
        "cache_contract_sha256": prepared["metadata"]["source_signature"]["contract_sha256"],
        "parent_artifact_sha256": _sha256(parent.MODEL_PATH),
        "trainer_script_sha256": _sha256(Path(__file__)),
        "core_script_sha256": _sha256(Path(core.__file__)),
        "parent_trainer_script_sha256": _sha256(Path(parent.__file__)),
        "fixed_holding_period_used": False,
    }
    torch.save(checkpoint, MODEL_PATH)
    ledgers["validation"].to_csv(VALIDATION_LEDGER_PATH, index=False)
    ledgers["development"].to_csv(DEVELOPMENT_LEDGER_PATH, index=False)

    report = {
        "model_id": MODEL_ID,
        "status": "research_only_consumed_outer_intervals",
        "model_family": "warm-started nine-expert Inventory-MoE with opportunity-cost and exit-hazard heads",
        "device": str(device),
        "seeds": list(seeds),
        "expert_count": expert_count,
        "parent": {
            "model_id": parent.MODEL_ID,
            "artifact": str(parent.MODEL_PATH),
            "artifact_sha256": _sha256(parent.MODEL_PATH),
            "warm_start_missing_new_head_keys": warm_start_missing,
            "parent_parameters_frozen_during_v3_training": True,
        },
        "holding_contract": {
            "fixed_holding_period_used": False,
            "max_holding_period_used": False,
            "fixed_tp_sl_used": False,
            "cooldown_used": False,
            "holding_duration_feature_used": False,
            "decision_frequency": "every completed 1-minute bar",
            "exit_rule": "learned continuation opportunity cost plus expert consensus",
            "uncertainty_rule": "mixed Q minus tune-selected penalty times nine-expert Q standard deviation",
        },
        "teacher_contract": {
            "continuation_target": "Q(previous, hold) - max Q(previous, alternative)",
            "exit_hazard_target": "argmax action differs from previous inventory",
            "future_path_used_as_fit_target_only": True,
        },
        "config": asdict(config),
        "feature_contract": {
            "base_features": frame["base_names"],
            "micro_features": frame["micro_names"],
            "trade_ledgers_used": False,
            "holding_duration_feature_used": False,
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
        "selected_research_policy": asdict(policy),
        "artifact_execution_policy": asdict(execution_policy),
        "activation_allowed": False,
        "tune": results["tune"],
        "tune_cost_stress": stresses["tune"],
        "historical_validation_diagnostic": results["validation"],
        "historical_validation_cost_stress": stresses["validation"],
        "historical_development_diagnostic": results["development"],
        "historical_development_cost_stress": stresses["development"],
        "parent_comparison": _comparison(parent_report, results),
        "top_tune_candidates": candidates[:15],
        "artifacts": {
            "model": str(MODEL_PATH),
            "historical_validation_diagnostic_ledger": str(VALIDATION_LEDGER_PATH),
            "historical_development_diagnostic_ledger": str(DEVELOPMENT_LEDGER_PATH),
            "contract": str(CONTRACT_PATH),
        },
        "integrity": {
            "script_sha256": _sha256(Path(__file__)),
            "core_script_sha256": _sha256(Path(core.__file__)),
            "parent_trainer_script_sha256": _sha256(Path(parent.__file__)),
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
            "parent_outer_results_used_for_policy_selection": False,
        },
        "promotion": {
            "promotion_pass": False,
            "live_candidate": False,
            "reason": "All historical outer intervals are consumed development data; post-freeze fresh-forward evidence is required.",
            "next_untouched_start": "after this 2026-07-18 v3 freeze",
        },
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=core._json_default))
    print(
        json.dumps(
            {
                "selected_policy": asdict(policy),
                "activation_allowed": False,
                "tune": results["tune"],
                "historical_validation_diagnostic": results["validation"],
                "historical_development_diagnostic": results["development"],
            },
            indent=2,
            default=core._json_default,
        )
    )
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved report: {REPORT_PATH}")
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    config = replace(OpportunityConfig(), epochs=1, batch_size=1024) if args.smoke else OpportunityConfig()
    run(config, SEEDS)
