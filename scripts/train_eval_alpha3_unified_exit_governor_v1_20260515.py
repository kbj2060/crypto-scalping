#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import numpy.random._pickle as np_random_pickle

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_alpha3_rl_exit_owner_fulltrain_20260514 as rl_exit  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_unified_exit_governor_v1_20260515"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_unified_exit_governor_v1_20260515"
MODEL_OUT = OUT_DIR / "unified_exit_governor.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_governor_v1_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_governor_v1_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_governor_v1_20260515_grid.csv"
DATASET_OUT = ROOT / "data/ensemble/reports/alpha3_unified_exit_governor_v1_20260515_dataset.json"
CONTRACT_OUT = ROOT / "docs/model_contracts/alpha3_unified_exit_governor_v1_20260515_contract.md"
TRAIN_START = pd.Timestamp("2025-01-01")
VAL_START = pd.Timestamp("2025-10-01")


_ORIG_BITGEN_CTOR = np_random_pickle.__bit_generator_ctor


def _compat_bitgen_ctor(bit_generator_name: Any = "MT19937") -> Any:
    # Some artifacts were pickled under a NumPy build that serialized the
    # BitGenerator class object instead of its string name.
    if isinstance(bit_generator_name, type):
        bit_generator_name = bit_generator_name.__name__
    return _ORIG_BITGEN_CTOR(bit_generator_name)


np_random_pickle.__bit_generator_ctor = _compat_bitgen_ctor


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _corrected_live_entry_cfg() -> alpha3.ImmediateLimitConfig:
    return alpha3.ImmediateLimitConfig(
        "alpha3_corrected_selected_touch0_skip_entry",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _unified_policy_grid() -> list[rl_exit.OfflineRLPolicy]:
    rows: list[rl_exit.OfflineRLPolicy] = []
    for fallback in ("exit4_pen0", "baseline_exit2_pen05"):
        for margin, conf, min_hold in ((0.000, 0.000, 1), (0.001, 0.001, 1), (0.002, 0.001, 3)):
            rows.append(
                rl_exit.OfflineRLPolicy(
                    name=f"unified_exit_m{margin:.3f}_c{conf:.4f}_h{min_hold}_fb_{fallback}",
                    q_margin=float(margin),
                    min_advantage_conf=float(conf),
                    min_hold=int(min_hold),
                    exit_fallback_arm=fallback,
                )
            )
    rows.extend(
        [
            rl_exit.OfflineRLPolicy("fixed_touch0_exit_fallback", 99.0, 99.0, 999, "baseline_exit2_pen05", "fallback"),
            rl_exit.OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", "fallback"),
            rl_exit.OfflineRLPolicy("strict_best_q_exit4", 0.0, 0.001, 1, "exit4_pen0", "q_or_fallback"),
        ]
    )
    return rows


def _install_unified_selector() -> None:
    original = rl_exit._select_action

    def _select_action_unified(
        model: rl_exit.ExitQNet,
        x: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        policy: rl_exit.OfflineRLPolicy,
        action_names: list[str],
        *,
        force_exit: bool,
    ) -> tuple[int, float, np.ndarray]:
        q = rl_exit._q_from_model(model, x, mean, std)
        hold_q = float(q[0])
        if force_exit or policy.force_exit_mode == "fallback":
            return original(model, x, mean, std, policy, action_names, force_exit=force_exit)

        exit_i = int(np.argmax(q[1:])) + 1
        exit_adv = float(q[exit_i] - hold_q)
        sorted_q = np.sort(q)
        conf = float(sorted_q[-1] - sorted_q[-2]) if len(sorted_q) > 1 else 0.0
        if exit_adv >= float(policy.q_margin) and conf >= float(policy.min_advantage_conf):
            return exit_i, exit_adv, q
        return 0, exit_adv, q

    rl_exit._select_action = _select_action_unified


def _serialise(obj: Any) -> Any:
    return json.loads(json.dumps(obj, ensure_ascii=False, default=_json_default))


def _write_contract(selected_policy: rl_exit.OfflineRLPolicy, entry_cfg: alpha3.ImmediateLimitConfig) -> None:
    CONTRACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_OUT.write_text(
        f"""# Alpha3 Unified Exit Governor v1 Contract

## Scope

- Entry stack is frozen Alpha3 corrected live contract.
- Entry contract: `{entry_cfg.name}`, next-open touch0 maker attempt, skipped if entry maker does not fill.
- This layer is reduce-only. It cannot open, flip, add, or increase a position.
- It governs both parent-owned `v21_2` and deep-scout-owned `deep_alpha` positions with one state/action interface.

## Decision Order

1. Active position state is built each bar after entry.
2. Unified exit governor may choose `hold` or a full reduce-only close with a selected maker-limit exit arm.
3. If the learned layer does not close, existing TP/SL/max-hold checks remain as fallback safety rails.
4. If a safety rail fires, the governor still selects the exit placement arm, but cannot veto the safety exit.

## Selected Runtime

```json
{json.dumps(asdict(selected_policy), indent=2, ensure_ascii=False)}
```

## Audit Boundaries

- Train: 2025-01-01 through 2025-09-30.
- Runtime selection: 2025-10-01 through 2025-12-31.
- Fixed OOS: 2026 only after selection.
- Backtest execution still uses 5m OHLC touch proxy, not real queue position.
""",
        encoding="utf-8",
    )


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    arms = deep_exit._arm_configs()
    entry_cfg = _corrected_live_entry_cfg()
    feature_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = deep_exit._feature_names(feature_cols)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < VAL_START)].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= VAL_START].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    need_train = not MODEL_OUT.exists()
    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions and frozen V27 q", flush=True)
    if need_train:
        train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    action_names = rl_exit._action_names(arms)
    if MODEL_OUT.exists():
        print(f"[{MODEL_ID}] loading cached unified fitted-Q exit governor", flush=True)
        payload = torch.load(MODEL_OUT, map_location="cpu", weights_only=False)
        q_model = rl_exit.ExitQNet(int(payload["input_dim"]), len(payload["actions"]))
        q_model.load_state_dict(payload["model_state"])
        q_model.eval()
        mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        std = np.asarray(payload["feature_std"], dtype=np.float32)
        train_meta = dict(payload.get("train_meta", {}))
        dataset_summary = dict(payload.get("dataset_meta", {}))
    else:
        print(f"[{MODEL_ID}] collecting corrected-contract unified exit states", flush=True)
        x, y, dataset_meta = rl_exit.collect_q_dataset(
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
            "entry_contract": asdict(entry_cfg),
            "target_argmax_counts": dict(zip(action_names, label_counts)),
            "target_mean_by_action": dict(zip(action_names, np.mean(y, axis=0).astype(float).tolist())),
        }
        DATASET_OUT.write_text(json.dumps(_serialise(dataset_summary), indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"[{MODEL_ID}] training unified fitted-Q exit governor", flush=True)
        q_model, train_meta = rl_exit._train_q_model(x, y, seed=20260515)
        mean = train_meta["feature_mean"]
        std = train_meta["feature_std"]
        torch.save(
            {
                "model_id": MODEL_ID,
                "model_state": q_model.state_dict(),
                "input_dim": len(feature_names),
                "actions": action_names,
                "arms": [asdict(cfg) for cfg in arms],
                "entry_contract": asdict(entry_cfg),
                "feature_names": feature_names,
                "feature_mean": mean,
                "feature_std": std,
                "train_meta": {k: v for k, v in train_meta.items() if k not in {"feature_mean", "feature_std"}},
                "dataset_meta": dataset_summary,
            },
            MODEL_OUT,
        )

    _install_unified_selector()
    print(f"[{MODEL_ID}] selecting unified exit runtime on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best: tuple[float, rl_exit.OfflineRLPolicy, dict[str, Any]] | None = None
    for policy in _unified_policy_grid():
        metrics = rl_exit._metrics_rl(val_df, stack, val_q, val_dec, entry_cfg, arms, feature_cols, q_model, mean, std, policy)
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
                "val_cost1_exits": json.dumps(metrics["cost1"].get("exits", {}), sort_keys=True),
                "val_cost1_rl_action_counts": json.dumps(metrics["cost1"].get("rl_action_counts", {}), sort_keys=True),
            }
        )
        if best is None or score > best[0]:
            best = (score, policy, metrics)
            print(
                f"[{MODEL_ID}] new best {policy.name} val c1={metrics['cost1']['pnl']:.2f} "
                f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
                flush=True,
            )
    assert best is not None
    selected_policy = best[1]
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
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
    fixed_front_policy = rl_exit.OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", "fallback")
    fixed_front = rl_exit._metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, fixed_front_policy)
    unified = rl_exit._metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, selected_policy)
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": "alpha3_corrected_touch0_skip_entry_baseline", "config": asdict(entry_cfg), "metrics": baseline, "score": _score(baseline)},
        {"name": "alpha3_corrected_touch0_skip_entry_fixed_front_run_exit4", "policy": asdict(fixed_front_policy), "metrics": fixed_front, "score": _score(fixed_front)},
        {"name": f"alpha3_unified_exit_governor::{selected_policy.name}", "policy": asdict(selected_policy), "metrics": unified, "score": _score(unified)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"trades={m['cost1']['trades']} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    _write_contract(selected_policy, entry_cfg)
    report = {
        "model_id": MODEL_ID,
        "design": {
            "algorithm": "unified reduce-only fitted-Q exit governor with aggressive validation-selected early-exit selector",
            "scope": "Alpha3 entry/size stack frozen. Parent-owned v21_2 and deep-owned deep_alpha positions share one exit layer.",
            "replaces": "fragmented parent_policy vs v31_effective_overlay timing exits",
            "safety": "TP/SL/max-hold remain fallback rails, not primary profit-taking logic.",
            "entry_contract": asdict(entry_cfg),
            "actions": action_names,
            "selection_uses_2026": False,
        },
        "dataset": dataset_summary,
        "train_meta": {k: v for k, v in train_meta.items() if k not in {"feature_mean", "feature_std", "history"}},
        "selected_policy": asdict(selected_policy),
        "validation_best_score": float(best[0]),
        "validation_best_metrics": best[2],
        "experiments": experiments,
        "artifacts": {
            "model": str(MODEL_OUT.relative_to(ROOT)),
            "grid": str(GRID_OUT.relative_to(ROOT)),
            "dataset": str(DATASET_OUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUT.relative_to(ROOT)),
            "contract": str(CONTRACT_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(_serialise(report), indent=2, ensure_ascii=False), encoding="utf-8")

    base_exits = baseline["cost1"].get("exits", {})
    unified_exits = unified["cost1"].get("exits", {})
    baseline_score = _score(baseline)
    unified_score = _score(unified)
    do_not_promote = unified_score < baseline_score
    audit = {
        "model_id": MODEL_ID,
        "status": "reject_do_not_promote" if do_not_promote else "shadow_candidate",
        "selection_uses_2026": False,
        "causality": [
            "Train: 2025-01-01..2025-09-30 only.",
            "Runtime selection: 2025-10-01..2025-12-31 only.",
            "2026 is fixed OOS and not used for fitting or selection.",
            "State uses current-bar causal features, open-position state, Alpha3 decision outputs, and frozen V27 q.",
        ],
        "exit_attribution_cost1": {
            "baseline": base_exits,
            "unified_governor": unified_exits,
            "baseline_stop_loss_plus_max_hold": int(sum(v for k, v in base_exits.items() if "stop_loss" in k or "max_hold" in k)),
            "unified_stop_loss_plus_max_hold": int(sum(v for k, v in unified_exits.items() if "stop_loss" in k or "max_hold" in k)),
        },
        "blocking": ["unified_exit_governor_underperforms_corrected_alpha3_baseline_on_2026_oos"] if do_not_promote else [],
        "warnings": [
            "Backtest execution uses 5m OHLC maker-touch proxy, not real queue position or partial fills.",
            "First v1 governor closes 100% only. Partial close/TWAP can be added after shadow attribution is stable.",
            "TP/SL/max-hold are still fallback rails by design; production should not remove hard safety rails before live shadow confirms exit latency.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(_serialise(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "model": str(MODEL_OUT), "contract": str(CONTRACT_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
