#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from features.elite import RegimeEngine  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import redteam_retest_alpha3_realistic_execution_20260515 as redteam  # noqa: E402
from scripts import retest_alpha3_entry_exit_timing_20260515 as timing_mod  # noqa: E402
from scripts.retest_alpha3_current_live_guard_20260515 import LIVE_TRAIL_ACTIVATION  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "redteam_alpha3_timing_realistic_execution_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/redteam_alpha3_timing_realistic_execution_20260515.json"
GRID_OUT = ROOT / "data/ensemble/reports/redteam_alpha3_timing_realistic_execution_20260515_grid.csv"


def _metrics_for_contract(
    df: pd.DataFrame,
    stack: dict[str, Any],
    q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    timing: timing_mod.TimingConfig,
    cfg: alpha3.ImmediateLimitConfig,
    try_limit_fn: redteam.TryLimitFn,
) -> dict[str, Any]:
    with redteam._patched_try_limit(try_limit_fn):
        return {
            f"cost{mult}": timing_mod.backtest_timing(
                df,
                stack["parent"],
                stack["jackpot_model"],
                stack["add_cfg"],
                q,
                decisions,
                overlay,
                cfg,
                timing,
                fee=stack["fee"],
                slip=stack["slip"],
                cost_mult=float(mult),
            )
            for mult in (1, 2, 3)
        }


def _score(metrics: dict[str, Any]) -> float:
    return redteam._score(metrics)


def _timing_variants() -> list[timing_mod.TimingConfig]:
    return [
        timing_mod.TimingConfig("baseline_live_guard"),
        timing_mod.TimingConfig("counter_regime_block", block_counter_regime=True),
        timing_mod.TimingConfig("counter_signal_antichase4", block_counter_regime=True, signal_exit_enable=True, chase_lookback=4, max_aligned_move=0.0045),
    ]


def _json_default_safe(obj: Any) -> Any:
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return _json_default(obj)


def main() -> int:
    print(f"[{MODEL_ID}] loading frozen Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_df = RegimeEngine().compute(_read(v31.DEFAULT_EVAL).copy())
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    overlay = replace(stack["overlay"], notional=2.0, trail_activation=LIVE_TRAIL_ACTIVATION)
    fill_prob = redteam._live_l2_compatible_ratio()

    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    for timing in _timing_variants():
        for contract in redteam._contract_rows(fill_prob):
            print(f"[{MODEL_ID}] {timing.name} x {contract['name']}", flush=True)
            metrics = _metrics_for_contract(
                eval_df,
                stack,
                eval_q,
                eval_dec,
                overlay,
                timing,
                contract["cfg"],
                contract["try_fn"],
            )
            score = _score(metrics)
            row = {
                "timing": timing.name,
                "contract": contract["name"],
                "production_eligible": bool(contract.get("production_eligible", False)),
                "score": score,
                "cost1_pnl": metrics["cost1"]["pnl"],
                "cost1_mdd": metrics["cost1"]["mdd"],
                "cost1_trades": metrics["cost1"]["trades"],
                "cost1_wr": metrics["cost1"]["wr"],
                "cost2_pnl": metrics["cost2"]["pnl"],
                "cost2_mdd": metrics["cost2"]["mdd"],
                "cost3_pnl": metrics["cost3"]["pnl"],
                "cost3_mdd": metrics["cost3"]["mdd"],
                "cost1_exits": json.dumps(metrics["cost1"].get("exits", {}), sort_keys=True),
                "cost1_guards": json.dumps(metrics["cost1"].get("guard_counts", {}), sort_keys=True),
            }
            rows.append(row)
            experiments.append({"timing": asdict(timing), "contract": contract["name"], "metrics": metrics, "score": score})
    grid = pd.DataFrame(rows).sort_values("score", ascending=False)
    grid.to_csv(GRID_OUT, index=False)
    production = grid[grid["production_eligible"].astype(bool)].copy()
    best_prod = production.iloc[0].to_dict() if len(production) else {}
    blocking: list[str] = []
    if not best_prod:
        blocking.append("no_production_eligible_contracts_evaluated")
    if float(best_prod.get("cost1_pnl", -1e18)) <= 0.0:
        blocking.append("no_positive_cost1_under_production_eligible_execution")
    if float(best_prod.get("cost3_pnl", -1e18)) <= 0.0:
        blocking.append("no_positive_cost3_under_production_eligible_execution")
    if float(best_prod.get("cost1_mdd", -100.0)) < -35.0:
        blocking.append("best_production_eligible_mdd_below_minus35")
    report = {
        "model_id": MODEL_ID,
        "selection_uses_2026": True,
        "purpose": "redteam selected timing variants under realistic execution contracts",
        "live_l2_fill_ratio_used": fill_prob,
        "best_production_eligible_by_score": best_prod,
        "grid": str(GRID_OUT),
        "experiments": experiments,
        "audit": {
            "status": "blocked" if blocking else "research_pass",
            "blocking": blocking,
            "warnings": [
                "timing variant selection uses 2026 OOS and must be revalidated on Mar-Apr or walk-forward before promotion",
                "l2_haircut is deterministic approximation, not true order book replay",
            ],
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default_safe), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "best_production": best_prod, "blocking": blocking}, ensure_ascii=False, default=_json_default_safe), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
