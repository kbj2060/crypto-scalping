#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_alpha3_rescue_exit_governor_v2_20260515 as v2  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_guarded_exit_selector_v3_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_guarded_exit_selector_v3_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_guarded_exit_selector_v3_20260515_audit.json"
GRID_IN = ROOT / "data/ensemble/reports/alpha3_rescue_exit_governor_v2_20260515_grid.csv"
CONTRACT_OUT = ROOT / "docs/model_contracts/alpha3_guarded_exit_selector_v3_20260515_contract.md"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _cfg_from_row(row: pd.Series) -> v2.RescueExitConfig:
    return v2.RescueExitConfig(
        name=str(row["name"]),
        min_hold=int(row["min_hold"]),
        sl_progress=float(row["sl_progress"]),
        adverse_q_margin=float(row["adverse_q_margin"]),
        min_mfe=float(row["min_mfe"]),
        giveback_frac=float(row["giveback_frac"]),
        time_frac=float(row["time_frac"]),
        exit_arm=str(row["exit_arm"]),
        maker_fee_mult=float(row.get("maker_fee_mult", 0.20)),
    )


def _select_guarded(grid: pd.DataFrame) -> tuple[v2.RescueExitConfig, dict[str, Any]]:
    base = grid.loc[grid["name"] == "disabled_baseline"].iloc[0]
    base_trades = int(base["val_cost1_trades"])
    base_score = float(base["selection_score"])
    eligible = grid.copy()
    eligible["trade_ratio"] = eligible["val_cost1_trades"].astype(float) / max(float(base_trades), 1.0)
    eligible = eligible[
        (eligible["selection_score"].astype(float) >= base_score + 10.0)
        & (eligible["trade_ratio"] <= 1.00)
        & (eligible["val_cost1_mdd"].astype(float) >= float(base["val_cost1_mdd"]) - 2.0)
    ]
    if eligible.empty:
        return _cfg_from_row(base), {
            "selected_mode": "fail_closed_to_baseline",
            "baseline_validation_score": base_score,
            "reason": "no_rescue_candidate_passed_trade_count_mdd_and_score_stability_guards",
            "guards": {
                "min_score_improvement": 10.0,
                "max_trade_ratio": 1.00,
                "max_mdd_degradation_pct": 2.0,
            },
        }
    row = eligible.sort_values("selection_score", ascending=False).iloc[0]
    return _cfg_from_row(row), {
        "selected_mode": "guarded_rescue",
        "baseline_validation_score": base_score,
        "selected_validation_score": float(row["selection_score"]),
        "selected_validation_trade_ratio": float(row["trade_ratio"]),
        "guards": {
            "min_score_improvement": 10.0,
            "max_trade_ratio": 1.00,
            "max_mdd_degradation_pct": 2.0,
        },
    }


def _write_contract(selected: v2.RescueExitConfig, meta: dict[str, Any]) -> None:
    CONTRACT_OUT.parent.mkdir(parents=True, exist_ok=True)
    CONTRACT_OUT.write_text(
        f"""# Alpha3 Guarded Exit Selector v3 Contract

## Purpose

v1 learned early exits and v2 rescue exits both showed that aggressive early-close logic can overfit 2025Q4 and damage 2026. v3 therefore makes the exit layer fail-closed: a rescue policy is allowed only if it beats baseline validation score while keeping trade count and MDD stable.

## Selected Runtime

```json
{json.dumps(asdict(selected), indent=2, ensure_ascii=False)}
```

## Selector Metadata

```json
{json.dumps(meta, indent=2, ensure_ascii=False)}
```

## Production Rule

If no candidate passes stability guards, the corrected Alpha3 baseline lifecycle remains active.
""",
        encoding="utf-8",
    )


def main() -> int:
    print(f"[{MODEL_ID}] selecting guarded runtime from v2 validation grid", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    grid = pd.read_csv(GRID_IN)
    selected, selector_meta = _select_guarded(grid)
    print(f"[{MODEL_ID}] selected {selected.name} mode={selector_meta['selected_mode']}", flush=True)

    print(f"[{MODEL_ID}] loading fixed Alpha3 stack and rebuilding 2026 decisions", flush=True)
    stack = front_run._load_fixed_stack()
    eval_df = _read(v31.DEFAULT_EVAL)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)
    entry_cfg = v2._entry_cfg()

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    taker = alpha2._metrics(eval_df, stack["parent"], stack["jackpot_model"], stack["add_cfg"], eval_q, eval_dec, l2._variants()[0], fee=stack["fee"], slip=stack["slip"])
    corrected_baseline = alpha3._metrics_signal_limit(
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
    guarded = v2._metrics_rescue(eval_df, stack, eval_q, eval_dec, selected)
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha3_corrected_touch0_skip_entry_baseline", "config": asdict(entry_cfg), "metrics": corrected_baseline, "score": _score(corrected_baseline)},
        {"name": f"alpha3_guarded_exit_selector::{selected.name}", "policy": asdict(selected), "selector": selector_meta, "metrics": guarded, "score": _score(guarded)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"trades={m['cost1']['trades']} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    base_exits = corrected_baseline["cost1"].get("exits", {})
    guarded_exits = guarded["cost1"].get("exits", {})
    fail_closed = selector_meta.get("selected_mode") == "fail_closed_to_baseline"
    audit = {
        "model_id": MODEL_ID,
        "status": "promote_guarded_baseline" if fail_closed else ("promote_shadow_candidate" if _score(guarded) >= _score(corrected_baseline) else "reject_do_not_promote"),
        "selection_uses_2026": False,
        "selected_config": asdict(selected),
        "selector_meta": selector_meta,
        "exit_attribution_cost1": {
            "baseline": base_exits,
            "guarded": guarded_exits,
            "baseline_stop_loss_plus_max_hold": int(sum(v for k, v in base_exits.items() if "stop_loss" in k or "max_hold" in k)),
            "guarded_stop_loss_plus_max_hold": int(sum(v for k, v in guarded_exits.items() if "stop_loss" in k or "max_hold" in k)),
        },
        "blocking": [] if _score(guarded) >= _score(corrected_baseline) else ["guarded_selector_underperforms_corrected_baseline"],
        "warnings": ["v3 is intentionally fail-closed; it preserves baseline when rescue candidates look unstable."],
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Guarded selector over v2 rescue candidates. It fails closed to corrected Alpha3 baseline unless validation improvement is stable under trade-count and MDD constraints.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"source_grid": str(GRID_IN.relative_to(ROOT)), "audit": str(AUDIT_OUT.relative_to(ROOT)), "contract": str(CONTRACT_OUT.relative_to(ROOT))},
    }
    _write_contract(selected, selector_meta)
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "contract": str(CONTRACT_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
