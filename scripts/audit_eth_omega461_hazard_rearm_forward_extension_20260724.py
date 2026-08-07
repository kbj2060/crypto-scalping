#!/usr/bin/env python3
"""One-shot frozen-policy extension audit for ETH Omega4.6.1 hazard + re-arm."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import audit_eth_omega461_live_chop_hazard_composition_20260724 as composition  # noqa: E402
import research_eth_omega461_competing_risk_rescue_20260724 as hazard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_live_hazard_rearm_20260724 as rearm  # noqa: E402


MODEL_ID = "eth_omega461_hazard_rearm_forward_extension_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
EXTENSION_START = "2026-04-01"
EXTENSION_END = "2026-07-12 09:00:00"
FROZEN_REARM_MODE = "cooldown"
FROZEN_REARM_BARS = 96
COST_MULTIPLIERS = (1.0, 2.0, 3.0)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _prepare_extension():
    frame = sweep.load_frame(
        EXTENSION_START,
        EXTENSION_END,
        base_csv=sweep.BASE_2026,
        wide24_csv=sweep.WIDE24_2026,
    )
    components = {
        name: hazard.prepare_split(
            name,
            cfg,
            frame,
            hazard._prediction_path("oos", name, cfg),
            oof=False,
            pre_quality=False,
        )
        for name, cfg in sweep.COMPONENTS.items()
    }
    return frame, components


def _run(frame, components, bundles, *, candidate: bool, cost_mult: float):
    metrics, ledger = hazard.replay_router(
        frame,
        components,
        rescue_bundles=bundles if candidate else None,
        sl_probability_min=composition.SL_PROBABILITY_MIN if candidate else 1.0,
        value_margin=composition.VALUE_MARGIN if candidate else 1.0,
        persistence=composition.PERSISTENCE if candidate else 1,
        entry_notional_multiplier=composition.ENTRY_NOTIONAL_MULTIPLIER,
        max_entry_notional=composition.MAX_ENTRY_NOTIONAL,
        chop_soft_size_threshold=composition.CHOP_SOFT_SIZE_THRESHOLD,
        rearm_mode=FROZEN_REARM_MODE if candidate else "none",
        rearm_bars=FROZEN_REARM_BARS if candidate else 0,
        cost_mult=cost_mult,
    )
    return composition._augment(metrics, ledger), ledger


def _monthly(ledger: pd.DataFrame) -> list[dict[str, Any]]:
    work = ledger.copy()
    work["month"] = pd.to_datetime(work["exit_timestamp"]).dt.to_period("M").astype(str)
    rows = []
    for month, group in work.groupby("month", sort=True):
        returns = group["trade_return"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "month": month,
                "pnl": float((np.prod(1.0 + returns) - 1.0) * 100.0),
                "realized_mdd": composition._realized_mdd(returns),
                "trades": int(len(group)),
                "wr": float(np.mean(returns > 0.0)),
            }
        )
    return rows


def _passes(candidate: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        candidate["pnl"] >= 0.90 * baseline["pnl"]
        and candidate["close_mark_to_market_mdd"] >= baseline["close_mark_to_market_mdd"]
        and candidate["realized_mdd"] >= baseline["realized_mdd"]
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    selection_report = json.loads(
        (rearm.OUT_DIR / "report.json").read_text(encoding="utf-8")
    )
    selected = selection_report.get("selected") or {}
    if selected.get("rearm_mode") != FROZEN_REARM_MODE or int(selected.get("rearm_bars", -1)) != FROZEN_REARM_BARS:
        raise RuntimeError("frozen re-arm contract does not match the VAL-selected source report")

    frame, components = _prepare_extension()
    bundles = rearm._load_models()
    costs: dict[str, Any] = {}
    baseline_1x = candidate_1x = None
    for cost_mult in COST_MULTIPLIERS:
        baseline, baseline_ledger = _run(
            frame, components, bundles, candidate=False, cost_mult=cost_mult
        )
        candidate, candidate_ledger = _run(
            frame, components, bundles, candidate=True, cost_mult=cost_mult
        )
        tag = f"cost{int(cost_mult)}"
        costs[tag] = {
            "live_baseline": baseline,
            "live_plus_frozen_hazard_rearm": candidate,
            "passes_same_cost_baseline": _passes(candidate, baseline),
        }
        if cost_mult == 1.0:
            baseline_1x, candidate_1x = baseline, candidate
            baseline_ledger.to_csv(OUT_DIR / "extension_live_baseline_ledger.csv", index=False)
            candidate_ledger.to_csv(OUT_DIR / "extension_frozen_hazard_rearm_ledger.csv", index=False)
            costs[tag]["monthly_baseline"] = _monthly(baseline_ledger)
            costs[tag]["monthly_candidate"] = _monthly(candidate_ledger)

    assert baseline_1x is not None and candidate_1x is not None
    report = {
        "model_id": MODEL_ID,
        "status": "one_shot_extension_complete",
        "deployment_verdict": (
            "shadow_only_true_forward_confirmation_required"
            if _passes(candidate_1x, baseline_1x)
            else "do_not_apply_to_live"
        ),
        "frozen_policy": {
            "hazard_variant": composition.VARIANT,
            "sl_probability_min": composition.SL_PROBABILITY_MIN,
            "value_margin": composition.VALUE_MARGIN,
            "persistence": composition.PERSISTENCE,
            "rearm_mode": FROZEN_REARM_MODE,
            "rearm_bars": FROZEN_REARM_BARS,
            "models_retrained": False,
            "parameters_retuned_on_extension": False,
        },
        "protocol": {
            "extension": [EXTENSION_START, EXTENSION_END],
            "rows": int(len(frame)),
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "extension_used_for_hazard_or_rearm_selection": False,
            "fully_untouched_live_holdout": False,
            "limitation": (
                "Fresh relative to hazard/re-arm selection, but the underlying live chop sizing "
                "was researched using overlapping 2026 data. This cannot promote the policy."
            ),
        },
        "cost_sensitivity": costs,
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
