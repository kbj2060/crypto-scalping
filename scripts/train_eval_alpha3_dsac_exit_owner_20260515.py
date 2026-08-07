#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.train_rl_dsac_agent import DSACAgent  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_alpha3_rl_exit_owner_fulltrain_20260514 as base_exit  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_dsac_exit_owner_20260515"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_dsac_exit_owner_20260515"
MODEL_OUT = OUT_DIR / "dsac_exit_owner.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_20260515_grid.csv"
DATASET_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_20260515_dataset.json"
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-10-01")
VAL_START = pd.Timestamp("2025-10-01")


ACTION_SCALARS = {
    "hold": 0.0,
    "baseline_exit2_pen05": -0.85,
    "exit0_pen0": -0.50,
    "exit1_pen0": -0.22,
    "exit2_pen0": 0.35,
    "exit3_pen0": 0.62,
    "exit4_pen0": 0.88,
}


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _normalise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std, ((x - mean) / std).astype(np.float32)


def _scalar_to_action_idx(a: float, action_names: list[str], fallback_arm: str, *, force_exit: bool) -> int:
    x = float(np.clip(a, -1.0, 1.0))
    if -0.15 <= x <= 0.15:
        return action_names.index(fallback_arm) if force_exit else 0
    if x < -0.65:
        return action_names.index("baseline_exit2_pen05")
    if x < -0.35:
        return action_names.index("exit0_pen0")
    if x < -0.15:
        return action_names.index("exit1_pen0")
    if x < 0.45:
        return action_names.index("exit2_pen0")
    if x < 0.75:
        return action_names.index("exit3_pen0")
    return action_names.index("exit4_pen0")


def _select_action_dsac(
    agent: DSACAgent,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    policy: base_exit.OfflineRLPolicy,
    action_names: list[str],
    *,
    force_exit: bool,
) -> tuple[int, float, np.ndarray]:
    z = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    scalar = float(agent.act(z, deterministic=True))
    idx = _scalar_to_action_idx(scalar, action_names, policy.exit_fallback_arm, force_exit=force_exit)
    return int(idx), scalar, np.array([scalar], dtype=np.float64)


def _runtime_policies() -> list[base_exit.OfflineRLPolicy]:
    rows: list[base_exit.OfflineRLPolicy] = []
    for min_hold in (1, 3, 6, 12):
        for fallback in ("exit4_pen0", "baseline_exit2_pen05"):
            rows.append(
                base_exit.OfflineRLPolicy(
                    name=f"dsac_exit_owner_minhold{min_hold}_fb_{fallback}",
                    q_margin=0.0,
                    min_advantage_conf=0.0,
                    min_hold=int(min_hold),
                    exit_fallback_arm=fallback,
                    force_exit_mode="dsac_actor",
                )
            )
    rows.append(base_exit.OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", "fallback"))
    return rows


def _populate_dsac_replay(
    agent: DSACAgent,
    x_z: np.ndarray,
    y: np.ndarray,
    action_names: list[str],
    *,
    reward_scale: float = 12.0,
    reward_clip: float = 5.0,
) -> dict[str, Any]:
    pushed = 0
    action_counts: dict[str, int] = {}
    reward_sum: dict[str, float] = {}
    for i in range(len(x_z)):
        state = x_z[i].astype(np.float32)
        next_state = state
        for j, name in enumerate(action_names):
            q = float(y[i, j])
            if not np.isfinite(q) or q < -1e5:
                continue
            scalar = float(ACTION_SCALARS[name])
            reward = float(np.clip(q * reward_scale, -reward_clip, reward_clip))
            agent.memory.push(state, scalar, reward, next_state, 1.0)
            pushed += 1
            action_counts[name] = action_counts.get(name, 0) + 1
            reward_sum[name] = reward_sum.get(name, 0.0) + reward
    reward_mean = {k: float(reward_sum[k] / max(action_counts.get(k, 1), 1)) for k in action_counts}
    return {"pushed": int(pushed), "action_counts": action_counts, "reward_mean": reward_mean}


def _train_dsac(
    x: np.ndarray,
    y: np.ndarray,
    action_names: list[str],
    *,
    updates: int = 18000,
    batch_size: int = 512,
    seed: int = 20260515,
) -> tuple[DSACAgent, dict[str, Any], np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device()
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mean, std, x_z = _normalise(x)
    agent = DSACAgent(
        state_dim=x.shape[1],
        hidden_dim=256,
        lr_actor=2e-4,
        lr_critic=3e-4,
        lr_alpha=2e-4,
        gamma=0.995,
        tau=0.007,
        n_quantiles=48,
        cvar_frac=0.35,
        adaptive_pessimism=True,
        pessimism_min_weight=0.68,
        pessimism_weight_min=0.60,
        pessimism_weight_max=0.82,
        dynamic_entropy=True,
        entropy_min=-0.95,
        entropy_max=-0.35,
        entropy_std_low=0.12,
        entropy_std_high=0.38,
        entropy_step=0.035,
        critic_var_weight=True,
        critic_var_scale=0.50,
        primacy_soft_reset=True,
        primacy_window=80,
        cql_reg=True,
        cql_alpha=0.045,
        redo_enable=True,
        redo_interval=750,
        redo_tau=5e-3,
        redo_ratio=0.08,
        alpha_min=0.003,
        alpha_init=0.035,
        anti_flat_lambda=0.02,
        anti_flat_min_abs=0.08,
        anti_flat_anneal_updates=max(updates // 2, 1),
        direction_reg_lambda=0.02,
        side_balance_lambda=0.02,
        device=device,
    )
    replay_meta = _populate_dsac_replay(agent, x_z, y, action_names)
    history: list[dict[str, float]] = []
    acc: dict[str, float] = {}
    count = 0
    for step in range(1, int(updates) + 1):
        out = agent.update(batch_size=batch_size)
        if not out:
            continue
        count += 1
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step == 1 or step % 1000 == 0 or step == updates:
            row = {"step": float(step)}
            div = max(count, 1)
            for k, v in acc.items():
                row[k] = float(v / div)
            history.append(row)
            print(
                f"[{MODEL_ID}] update {step}/{updates} "
                f"critic={row.get('critic_loss', 0.0):.4f} actor={row.get('actor_loss', 0.0):.4f} "
                f"alpha={row.get('alpha', 0.0):.4f} cvar={row.get('cvar_q', 0.0):.4f}",
                flush=True,
            )
            acc = {}
            count = 0
    meta = {
        "device": device,
        "updates": int(updates),
        "batch_size": int(batch_size),
        "replay": replay_meta,
        "history": history,
    }
    return agent, meta, mean, std


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    arms = deep_exit._arm_configs()
    arm_by_name = {a.name: a for a in arms}
    entry_cfg = arm_by_name["baseline_exit2_pen05"]
    action_names = base_exit._action_names(arms)
    feature_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = deep_exit._feature_names(feature_cols)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < TRAIN_END)].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= VAL_START].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions and frozen V27 q", flush=True)
    train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    print(f"[{MODEL_ID}] collecting DP-labeled exit-owner replay", flush=True)
    x, y, dataset_meta = base_exit.collect_q_dataset(
        train_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        train_q,
        train_dec,
        stack["overlay"],
        entry_cfg,
        arms,
        feature_cols,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    label_counts = np.bincount(np.argmax(y, axis=1), minlength=len(action_names)).astype(int).tolist()
    dataset_summary = {
        **dataset_meta,
        "train_start": str(train_df["timestamp"].iloc[0]) if len(train_df) else None,
        "train_end": str(train_df["timestamp"].iloc[-1]) if len(train_df) else None,
        "target_argmax_counts": dict(zip(action_names, label_counts)),
        "target_mean_by_action": dict(zip(action_names, np.mean(y, axis=0).astype(float).tolist())),
        "state_dim": int(x.shape[1]),
        "action_scalars": ACTION_SCALARS,
    }
    DATASET_OUT.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"[{MODEL_ID}] training DSAC exit owner", flush=True)
    agent, train_meta, mean, std = _train_dsac(x, y, action_names)
    torch.save(
        {
            "model_id": MODEL_ID,
            "actor_state": agent.actor.state_dict(),
            "critic_state": agent.critic.state_dict(),
            "critic_target_state": agent.critic_target.state_dict(),
            "state_dim": int(x.shape[1]),
            "feature_names": feature_names,
            "feature_mean": mean,
            "feature_std": std,
            "actions": action_names,
            "action_scalars": ACTION_SCALARS,
            "train_meta": train_meta,
            "dataset": dataset_summary,
        },
        MODEL_OUT,
    )

    original_select = base_exit._select_action
    base_exit._select_action = _select_action_dsac
    try:
        print(f"[{MODEL_ID}] selecting DSAC runtime on 2025Q4", flush=True)
        rows: list[dict[str, Any]] = []
        best_dsac: tuple[float, base_exit.OfflineRLPolicy, dict[str, Any]] | None = None
        best_any: tuple[float, base_exit.OfflineRLPolicy, dict[str, Any]] | None = None
        for policy in _runtime_policies():
            metrics = base_exit._metrics_rl(val_df, stack, val_q, val_dec, entry_cfg, arms, feature_cols, agent, mean, std, policy)
            score = _score(metrics)
            rows.append(
                {
                    **asdict(policy),
                    "selection_score": score,
                    "val_cost1_pnl": metrics["cost1"]["pnl"],
                    "val_cost1_mdd": metrics["cost1"]["mdd"],
                    "val_cost1_trades": metrics["cost1"]["trades"],
                    "val_cost2_pnl": metrics["cost2"]["pnl"],
                    "val_cost3_pnl": metrics["cost3"]["pnl"],
                    "val_cost1_rl_action_counts": json.dumps(metrics["cost1"].get("rl_action_counts", {}), sort_keys=True),
                    "val_cost1_route_counts": json.dumps(metrics["cost1"].get("route_counts", {}), sort_keys=True),
                }
            )
            print(
                f"[{MODEL_ID}] {policy.name} val c1={metrics['cost1']['pnl']:.2f} "
                f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
                flush=True,
            )
            if best_any is None or score > best_any[0]:
                best_any = (score, policy, metrics)
            if policy.name.startswith("dsac_") and (best_dsac is None or score > best_dsac[0]):
                best_dsac = (score, policy, metrics)
        assert best_dsac is not None and best_any is not None
        selected_policy = best_dsac[1]
        pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

        print(f"[{MODEL_ID}] fixed current 2026 replay", flush=True)
        taker = alpha2._metrics(
            eval_df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            eval_q,
            eval_dec,
            l2._variants()[0],
            fee=stack["fee"],
            slip=stack["slip"],
        )
        old_l2 = alpha2._metrics(
            eval_df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            eval_q,
            eval_dec,
            stack["selected_l2_variant"],
            fee=stack["fee"],
            slip=stack["slip"],
        )
        baseline = alpha3._metrics_signal_limit(
            eval_df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            eval_q,
            eval_dec,
            stack["overlay"],
            entry_cfg,
            fee=stack["fee"],
            slip=stack["slip"],
        )
        fixed_front_policy = base_exit.OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", "fallback")
        fixed_front = base_exit._metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, agent, mean, std, fixed_front_policy)
        dsac_metrics = base_exit._metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, agent, mean, std, selected_policy)
        best_any_metrics = base_exit._metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, agent, mean, std, best_any[1])
    finally:
        base_exit._select_action = original_select

    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": "alpha3_baseline_exit2_pen05", "metrics": baseline, "score": _score(baseline)},
        {"name": "alpha3_fixed_front_run_exit4_pen0", "policy": asdict(fixed_front_policy), "metrics": fixed_front, "score": _score(fixed_front)},
        {"name": f"alpha3_dsac_exit_owner::{selected_policy.name}", "policy": asdict(selected_policy), "metrics": dsac_metrics, "score": _score(dsac_metrics)},
        {"name": f"alpha3_best_any_selection::{best_any[1].name}", "policy": asdict(best_any[1]), "metrics": best_any_metrics, "score": _score(best_any_metrics)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    report = {
        "model_id": MODEL_ID,
        "date": "2026-05-15",
        "design": {
            "algorithm": "DSAC actor + distributional twin quantile critic + CVaR policy objective + CQL regularization",
            "source": "Adapted from ensemble/train_rl_dsac_agent.py DSACAgent.",
            "scope": "Alpha3 entry stack frozen. DSAC exit owner can hold or close 100% with a reduce-only exit placement arm.",
            "limitations": "Replay is DP-labeled contextual offline DSAC over Alpha3 lifecycle states; partial close/TWAP and real L2 fills are not included yet.",
            "train_split": "2025-01-01..2025-09-30",
            "selection_split": "2025-10-01..2025-12-31",
            "selection_uses_2026": False,
        },
        "dataset": dataset_summary,
        "train_meta": train_meta,
        "selected_dsac_policy": asdict(selected_policy),
        "selected_any_policy": asdict(best_any[1]),
        "validation_best_dsac_score": float(best_dsac[0]),
        "validation_best_any_score": float(best_any[0]),
        "experiments": experiments,
        "artifacts": {
            "model": str(MODEL_OUT.relative_to(ROOT)),
            "summary": str(REPORT_OUT.relative_to(ROOT)),
            "grid": str(GRID_OUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUT.relative_to(ROOT)),
            "dataset": str(DATASET_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    audit = {
        "model_id": MODEL_ID,
        "selection_uses_2026": False,
        "status": "research_candidate",
        "base_contract": "docs/model_contracts/alpha3_teacher_l2_limit_fallback_20260514_contract.md",
        "notes": [
            "Uses DSACAgent architecture and training techniques from ensemble/train_rl_dsac_agent.py.",
            "State/reward replay is Alpha3 exit-owner specific, not the original DSAC trading env.",
            "No partial close, TWAP, true market-close action, or real L2 queue fill labels in this first DSAC exit-owner run.",
        ],
        "promotion_gate": [
            "Must beat Alpha3 baseline and fixed exit4_pen0 on same eval horizon.",
            "Must survive cost2/cost3 stress and real L2 shadow route/fallback audit.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{MODEL_ID}] wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
