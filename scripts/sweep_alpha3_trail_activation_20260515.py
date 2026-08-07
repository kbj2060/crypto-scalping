#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_trail_activation_sweep_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_trail_activation_sweep_20260515_summary.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_trail_activation_sweep_20260515_grid.csv"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _cfg() -> alpha3.ImmediateLimitConfig:
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


def _metrics(df: pd.DataFrame, stack: dict[str, Any], q, dec, activation: float) -> dict[str, Any]:
    overlay = replace(stack["overlay"], trail_activation=float(activation))
    return alpha3._metrics_signal_limit(
        df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        q,
        dec,
        overlay,
        _cfg(),
        fee=stack["fee"],
        slip=stack["slip"],
    )


def _stop_count(metrics: dict[str, Any]) -> int:
    exits = dict(metrics["cost1"].get("exits", {}) or {})
    return int(sum(int(v) for k, v in exits.items() if "stop_loss" in k))


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    train_all = _read(v31.DEFAULT_TRAIN)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding decisions and V27 q", flush=True)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    activations = [0.0, 0.001, 0.002, 0.003, 0.004, 0.006, 0.009, 0.012, 0.018, 999.0]
    rows: list[dict[str, Any]] = []
    best: tuple[float, float, dict[str, Any]] | None = None
    print(f"[{MODEL_ID}] selecting activation on 2025Q4", flush=True)
    for activation in activations:
        m = _metrics(val_df, stack, val_q, val_dec, activation)
        score = _score(m)
        rows.append(
            {
                "trail_activation": float(activation),
                "selection_score": score,
                "val_cost1_pnl": m["cost1"]["pnl"],
                "val_cost1_mdd": m["cost1"]["mdd"],
                "val_cost1_trades": m["cost1"]["trades"],
                "val_cost1_stop_count": _stop_count(m),
                "val_cost2_pnl": m["cost2"]["pnl"],
                "val_cost3_pnl": m["cost3"]["pnl"],
                "val_cost1_exits": json.dumps(m["cost1"].get("exits", {}), sort_keys=True),
            }
        )
        print(
            f"[{MODEL_ID}] activation={activation:g} val c1={m['cost1']['pnl']:.2f} "
            f"mdd={m['cost1']['mdd']:.2f} stops={_stop_count(m)} c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )
        if best is None or score > best[0]:
            best = (score, float(activation), m)
    assert best is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    selected_activation = best[1]
    print(f"[{MODEL_ID}] fixed 2026 OOS selected_activation={selected_activation:g}", flush=True)
    selected_eval = _metrics(eval_df, stack, eval_q, eval_dec, selected_activation)
    live_default_eval = _metrics(eval_df, stack, eval_q, eval_dec, 0.009)
    old_like_eval = _metrics(eval_df, stack, eval_q, eval_dec, 0.0)
    disabled_eval = _metrics(eval_df, stack, eval_q, eval_dec, 999.0)

    experiments = [
        {"name": "old_like_activation_0", "trail_activation": 0.0, "metrics": old_like_eval, "score": _score(old_like_eval)},
        {"name": "selected_by_2025q4", "trail_activation": selected_activation, "metrics": selected_eval, "score": _score(selected_eval)},
        {"name": "live_default_activation_0_009", "trail_activation": 0.009, "metrics": live_default_eval, "score": _score(live_default_eval)},
        {"name": "trail_disabled_999", "trail_activation": 999.0, "metrics": disabled_eval, "score": _score(disabled_eval)},
    ]
    for e in experiments:
        m = e["metrics"]["cost1"]
        print(
            f"[{MODEL_ID}] {e['name']} act={e['trail_activation']:g} c1={m['pnl']:.2f} "
            f"mdd={m['mdd']:.2f} trades={m['trades']} stops={_stop_count(e['metrics'])} exits={m.get('exits', {})}",
            flush=True,
        )

    report = {
        "model_id": MODEL_ID,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS",
        "selected_trail_activation": selected_activation,
        "selected_validation_metrics": best[2],
        "experiments": experiments,
        "artifacts": {"grid": str(GRID_OUT.relative_to(ROOT))},
        "notes": [
            "activation=0 approximates previous immediate-trailing behavior.",
            "activation=999 disables trailing stop contraction.",
        ],
        "patched_overlay_base": asdict(stack["overlay"]),
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "grid": str(GRID_OUT), "selected_trail_activation": selected_activation}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
