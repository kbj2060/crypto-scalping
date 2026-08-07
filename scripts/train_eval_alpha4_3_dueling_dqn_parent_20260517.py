#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.dueling_dqn_parent import DuelingDQNConfig, DuelingQNetwork, make_action_model  # noqa: E402
from ensemble.fully_learned_governor_policy import (  # noqa: E402
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha4_new_features_full_retrain_20260517 as a4  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha4_3_no_teacher_no_deep_20260517 import _no_deep_overlay, _q0  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha4_3_dueling_dqn_parent_20260517"
DEFAULT_ROOT = ROOT / "tmp/causal_regen_20260516/alpha4_2_tp_sl_action_score_20260517"
DEFAULT_TRAIN = DEFAULT_ROOT / "trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = DEFAULT_ROOT / "trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_PARENT = DEFAULT_ROOT / "artifacts/hgb/parent.pkl"
DEFAULT_RUNNER = DEFAULT_ROOT / "teacher_ablation_artifacts/parent_direct_scaled_no_teacher_runner.pkl"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha4_3_dueling_dqn_parent_20260517"


def _seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _prep_matrix(x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32, copy=True)
    arr = np.where(np.isfinite(arr), arr, np.nan)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    med = np.where(np.isfinite(med), med, 0.0).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, med)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std > 1e-6, std, 1.0).astype(np.float32)
    z = ((arr - mean) / std).astype(np.float32)
    return z, med, mean, std


def _action_dist(a: np.ndarray) -> dict[str, int]:
    vc = pd.Series(np.asarray(a, dtype=np.int64)).value_counts().sort_index()
    return {str(int(k)): int(v) for k, v in vc.items()}


def _train_dqn(
    x: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    *,
    cfg: DuelingDQNConfig,
    steps: int,
    batch_size: int,
    gamma: float,
    lr: float,
    bc_weight: float,
    seed: int,
    device: torch.device,
) -> tuple[DuelingQNetwork, dict[str, Any]]:
    _seed(seed)
    n = int(len(x))
    next_idx = np.minimum(np.arange(n) + 1, n - 1)
    done = np.zeros(n, dtype=np.float32)
    done[-1] = 1.0
    actions = np.asarray(actions, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float32)
    reward_std = float(np.std(rewards)) if float(np.std(rewards)) > 1e-6 else 1.0
    reward_center = float(np.median(rewards))
    r = ((rewards - reward_center) / reward_std).astype(np.float32)

    model = DuelingQNetwork(cfg).to(device)
    target = DuelingQNetwork(cfg).to(device)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)

    priorities = np.abs(r) + 0.02
    priorities += np.where(actions == 0, 0.02, 0.35).astype(np.float32)
    class_weight = torch.tensor([0.35, 1.60, 1.60], dtype=torch.float32, device=device)
    losses: list[float] = []
    td_means: list[float] = []
    for step in range(1, int(steps) + 1):
        prob = priorities ** 0.65
        prob = prob / np.maximum(prob.sum(), 1e-12)
        idx = np.random.choice(n, size=min(int(batch_size), n), replace=True, p=prob)
        xb = torch.from_numpy(x[idx]).to(device)
        nb = torch.from_numpy(x[next_idx[idx]]).to(device)
        ab = torch.from_numpy(actions[idx]).to(device)
        rb = torch.from_numpy(r[idx]).to(device)
        db = torch.from_numpy(done[idx]).to(device)

        q = model(xb)
        qa = q.gather(1, ab.view(-1, 1)).squeeze(1)
        with torch.no_grad():
            next_action = torch.argmax(model(nb), dim=1, keepdim=True)
            next_q = target(nb).gather(1, next_action).squeeze(1)
            td_target = rb + float(gamma) * (1.0 - db) * next_q
        td = td_target - qa
        td_loss = F.smooth_l1_loss(qa, td_target)
        bc_loss = F.cross_entropy(q, ab, weight=class_weight)
        loss = td_loss + float(bc_weight) * bc_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 3.0)
        opt.step()
        priorities[idx] = 0.90 * priorities[idx] + 0.10 * (np.abs(td.detach().cpu().numpy()) + 0.02)
        if step % 100 == 0:
            target.load_state_dict(model.state_dict())
        if step % 50 == 0 or step == 1:
            losses.append(float(loss.detach().cpu()))
            td_means.append(float(torch.mean(torch.abs(td)).detach().cpu()))
    target.load_state_dict(model.state_dict())
    meta = {
        "reward_center": reward_center,
        "reward_std": reward_std,
        "loss_tail": losses[-10:],
        "td_abs_tail": td_means[-10:],
        "steps": int(steps),
        "batch_size": int(batch_size),
        "gamma": float(gamma),
        "bc_weight": float(bc_weight),
    }
    return model, meta


def _metrics_for(
    df: pd.DataFrame,
    *,
    parent_for_features: dict[str, Any],
    parent: dict[str, Any],
    runner: dict[str, Any],
    add_cfg: CostRunnerConfig,
    rt: alpha2.Alpha2Runtime,
    fee: float,
    slip: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    dec = predict_policy_frame(parent, df, close=_close(df))
    dec = alpha2._scale_parent_notional(dec, rt)
    metrics = a4._metrics(df, parent_for_features, runner, add_cfg, _q0(df), dec, _no_deep_overlay(), ft_v2.ft_v1._limit_cfg(), fee=fee, slip=slip)
    return metrics, dec


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/evaluate Alpha4.3 Dueling DQN parent action replacement.")
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--base-parent", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--runner-model", type=Path, default=DEFAULT_RUNNER)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--gamma", type=float, default=0.82)
    p.add_argument("--lr", type=float, default=7e-4)
    p.add_argument("--bc-weight", type=float, default=0.20)
    p.add_argument("--hidden-dim", type=int, default=256)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--temperature", type=float, default=0.18)
    p.add_argument("--seed", type=int, default=417)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    _seed(int(args.seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    base_parent = joblib.load(args.base_parent)
    cfg = FullyLearnedGovernorConfig(**dict(base_parent["config"]))
    feature_cols = list(base_parent["feature_cols"])
    x_train, y_train, label_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), feature_cols=feature_cols)
    x_np, med, mean, std = _prep_matrix(x_train)
    dqn_cfg = DuelingDQNConfig(
        input_dim=int(x_np.shape[1]),
        hidden_dim=int(args.hidden_dim),
        action_dim=3,
        dropout=float(args.dropout),
        temperature=float(args.temperature),
    )
    model, train_meta = _train_dqn(
        x_np,
        np.asarray(y_train["action"], dtype=np.int64),
        np.asarray(y_train["quality"], dtype=np.float32),
        cfg=dqn_cfg,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
        gamma=float(args.gamma),
        lr=float(args.lr),
        bc_weight=float(args.bc_weight),
        seed=int(args.seed),
        device=device,
    )

    action_model = make_action_model(model, config=dqn_cfg, medians=med, mean=mean, std=std, feature_cols=feature_cols)
    dqn_parent = copy.deepcopy(base_parent)
    dqn_parent["model_type"] = "alpha4_3_dueling_dqn_parent_action_replacement"
    dqn_parent["action_model"] = action_model
    dqn_parent["dqn_metadata"] = {
        "model_id": MODEL_ID,
        "device": str(device),
        "train_meta": train_meta,
        "label_meta": label_meta,
        "training_action_distribution": _action_dist(np.asarray(y_train["action"])),
        "note": "Dueling DQN replaces only action_model. Quality and bucket models are inherited from the fixed Alpha4.3 HGB parent.",
    }

    parent_path = args.out_dir / "dueling_dqn_parent.pkl"
    joblib.dump(dqn_parent, parent_path)

    parent_ref = joblib.load(v31.DEFAULT_PARENT)
    parent_for_features = copy.deepcopy(parent_ref)
    parent_for_features["feature_cols"] = feature_cols
    runner_payload = joblib.load(args.runner_model)
    runner = runner_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(runner_payload["selected_config"]))
    fee = float(dict(parent_ref["config"])["fee"])
    slip = float(dict(parent_ref["config"])["slip"])
    rt = alpha2.Alpha2Runtime("parent_direct_scale0.85", 0.0, 0.85, 2.75)

    val_metrics, val_dec = _metrics_for(val_df, parent_for_features=parent_for_features, parent=dqn_parent, runner=runner, add_cfg=add_cfg, rt=rt, fee=fee, slip=slip)
    eval_metrics, eval_dec = _metrics_for(eval_df, parent_for_features=parent_for_features, parent=dqn_parent, runner=runner, add_cfg=add_cfg, rt=rt, fee=fee, slip=slip)

    report = {
        "model_id": MODEL_ID,
        "design": "Dueling DQN + PER-like prioritized replay replaces only the Alpha4.3 HGB parent action model. Bucket models, V21.2 runner, no-teacher/no-deep overlay, runtime scale 0.85, and corrected Alpha3 limit-close execution are held fixed.",
        "selection_uses_2026": False,
        "device": str(device),
        "artifacts": {"parent": str(parent_path), "report": str(args.out_dir / "alpha4_3_dueling_dqn_parent_summary.json")},
        "dqn_config": asdict(dqn_cfg),
        "training": dqn_parent["dqn_metadata"],
        "decision_distribution": {
            "validation_2025q4": _action_dist(val_dec["action"].to_numpy()),
            "oos_2026": _action_dist(eval_dec["action"].to_numpy()),
        },
        "validation_metrics": val_metrics,
        "metrics": eval_metrics,
        "baseline_reference": {
            "alpha4_3_no_teacher_no_deep_cost1": 183.41556855122772,
            "alpha4_3_no_teacher_no_deep_mdd": -21.991930305978936,
            "alpha4_3_no_teacher_no_deep_cost2": 169.76099440447382,
            "alpha4_3_no_teacher_no_deep_cost3": 79.27211708311903,
        },
    }
    report_path = args.out_dir / "alpha4_3_dueling_dqn_parent_summary.json"
    audit_path = args.out_dir / "alpha4_3_dueling_dqn_parent_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    audit = {
        "status": "pass",
        "selection_uses_2026": False,
        "parent_replacement_scope": "action_model_only",
        "unchanged_layers": ["quality_model", "bucket_models", "V21.2 runner", "runtime scale 0.85", "no-teacher/no-deep overlay", "corrected limit-close execution"],
        "artifacts": report["artifacts"],
    }
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "audit": str(audit_path), "parent": str(parent_path), "metrics": eval_metrics}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
