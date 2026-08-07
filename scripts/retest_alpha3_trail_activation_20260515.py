#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_trail_activation_retest_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_trail_activation_retest_20260515.json"
PREVIOUS_REPORT = ROOT / "data/ensemble/reports/alpha3_live_execution_contract_retest_20260515.json"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _corrected_cfg() -> alpha3.ImmediateLimitConfig:
    return alpha3.ImmediateLimitConfig(
        "alpha3_corrected_selected_touch0_skip_entry_trail_activation",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )


def _previous_corrected() -> dict[str, Any] | None:
    if not PREVIOUS_REPORT.exists():
        return None
    data = json.loads(PREVIOUS_REPORT.read_text(encoding="utf-8"))
    for row in data.get("results", []):
        cfg = dict(row.get("config", {}) or {})
        if cfg.get("name") == "alpha3_corrected_selected_touch0_skip_entry":
            return row
    return None


def _stop_max_count(metrics: dict[str, Any]) -> int:
    exits = dict(metrics.get("cost1", {}).get("exits", {}) or {})
    return int(sum(int(v) for k, v in exits.items() if "stop_loss" in k or "max_hold" in k))


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding 2026 decisions and V27 q", flush=True)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    cfg = _corrected_cfg()

    print(f"[{MODEL_ID}] evaluating patched trail activation contract", flush=True)
    metrics = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["overlay"],
        cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    previous = _previous_corrected()
    prev_metrics = dict(previous.get("metrics", {}) if previous else {})
    report = {
        "model_id": MODEL_ID,
        "change": "V31/deep_alpha trailing stop now supports overlay.trail_activation. Default 0.0 preserves historical backtest behavior; higher values are experimental and can reduce SL churn at the cost of lower OOS PnL.",
        "selection_uses_2026": False,
        "contract": asdict(cfg),
        "patched_overlay_config": asdict(stack["overlay"]),
        "previous_corrected_reference": previous,
        "metrics": metrics,
        "score": _score(metrics),
        "comparison_cost1": {
            "previous_pnl": prev_metrics.get("cost1", {}).get("pnl") if prev_metrics else None,
            "patched_pnl": metrics["cost1"]["pnl"],
            "previous_mdd": prev_metrics.get("cost1", {}).get("mdd") if prev_metrics else None,
            "patched_mdd": metrics["cost1"]["mdd"],
            "previous_trades": prev_metrics.get("cost1", {}).get("trades") if prev_metrics else None,
            "patched_trades": metrics["cost1"]["trades"],
            "previous_exits": prev_metrics.get("cost1", {}).get("exits") if prev_metrics else None,
            "patched_exits": metrics["cost1"].get("exits", {}),
            "previous_stop_loss_plus_max_hold": _stop_max_count(prev_metrics) if prev_metrics else None,
            "patched_stop_loss_plus_max_hold": _stop_max_count(metrics),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(REPORT_OUT),
                "cost1_pnl": metrics["cost1"]["pnl"],
                "cost1_mdd": metrics["cost1"]["mdd"],
                "cost1_trades": metrics["cost1"]["trades"],
                "cost1_exits": metrics["cost1"].get("exits", {}),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
