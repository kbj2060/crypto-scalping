#!/usr/bin/env python3
"""Compose the frozen ETH Omega4.6.1 live sizing contract with the fixed hazard exit.

This is a diagnostic composition audit, not a new model-selection run.  It keeps the
already-selected B_post_quality hazard configuration fixed and compares it with the
same live entry/sizing path: duration gate off, 1.5x notional, chop soft-size T=0.3.
The OOS window was already opened by the source hazard experiment, so its readout is
not an untouched promotion result.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_omega461_competing_risk_rescue_20260724 as hazard  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402


MODEL_ID = "eth_omega461_live_chop_hazard_composition_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SOURCE_DIR = ROOT / "tmp/causal_regen_20260516" / hazard.MODEL_ID
VARIANT = "B_post_quality"
SL_PROBABILITY_MIN = 0.60
VALUE_MARGIN = 0.0025
PERSISTENCE = 1
ENTRY_NOTIONAL_MULTIPLIER = 1.5
MAX_ENTRY_NOTIONAL = 1.5
CHOP_SOFT_SIZE_THRESHOLD = 0.3


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    raise TypeError(type(value).__name__)


def _realized_mdd(returns: np.ndarray) -> float:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    for value in returns:
        cash *= 1.0 + float(value)
        peak = max(peak, cash)
        mdd = min(mdd, cash / peak - 1.0)
    return float(mdd * 100.0)


def _augment(metrics: dict[str, Any], ledger) -> dict[str, Any]:
    result = dict(metrics)
    result["close_mark_to_market_mdd"] = float(result.pop("mdd"))
    result["realized_mdd"] = _realized_mdd(ledger["trade_return"].to_numpy(dtype=np.float64))
    result["avg_notional"] = float(ledger["notional"].mean())
    result["avg_entry_sizing_multiplier"] = float(ledger["entry_sizing_multiplier"].mean())
    return result


def _load_models() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for name in sweep.COMPONENTS:
        path = SOURCE_DIR / f"model_{name}_{VARIANT}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            bundles[name] = pickle.load(handle)
    return bundles


def _run_split(split: str, bundles: dict[str, dict[str, Any]]):
    if split == "validation":
        frame = sweep.load_frame(
            sweep.VAL_START,
            sweep.VAL_END,
            base_csv=sweep.BASE_2025,
            wide24_csv=sweep.WIDE24_2025,
        )
        prediction_split = "validation"
        oof = True
    else:
        frame = sweep.load_frame(
            sweep.OOS_START,
            sweep.OOS_END,
            base_csv=sweep.BASE_2026,
            wide24_csv=sweep.WIDE24_2026,
        )
        prediction_split = "oos"
        oof = False

    components = {
        name: hazard.prepare_split(
            name,
            cfg,
            frame,
            hazard._prediction_path(prediction_split, name, cfg),
            oof=oof,
            pre_quality=False,
        )
        for name, cfg in sweep.COMPONENTS.items()
    }
    common = {
        "entry_notional_multiplier": ENTRY_NOTIONAL_MULTIPLIER,
        "max_entry_notional": MAX_ENTRY_NOTIONAL,
        "chop_soft_size_threshold": CHOP_SOFT_SIZE_THRESHOLD,
    }
    baseline, baseline_ledger = hazard.replay_router(
        frame,
        components,
        rescue_bundles=None,
        sl_probability_min=1.0,
        value_margin=1.0,
        persistence=1,
        **common,
    )
    candidate, candidate_ledger = hazard.replay_router(
        frame,
        components,
        rescue_bundles=bundles,
        sl_probability_min=SL_PROBABILITY_MIN,
        value_margin=VALUE_MARGIN,
        persistence=PERSISTENCE,
        **common,
    )
    baseline_ledger.to_csv(OUT_DIR / f"{split}_live_baseline_ledger.csv", index=False)
    candidate_ledger.to_csv(OUT_DIR / f"{split}_live_plus_hazard_ledger.csv", index=False)
    return _augment(baseline, baseline_ledger), _augment(candidate, candidate_ledger)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = _load_models()
    validation_baseline, validation_candidate = _run_split("validation", bundles)
    oos_baseline, oos_candidate = _run_split("oos", bundles)
    validation_pass = bool(
        validation_candidate["pnl"] >= 0.90 * validation_baseline["pnl"]
        and validation_candidate["close_mark_to_market_mdd"]
        >= validation_baseline["close_mark_to_market_mdd"]
        and validation_candidate["realized_mdd"] >= validation_baseline["realized_mdd"]
    )
    report = {
        "model_id": MODEL_ID,
        "status": "rejected_on_validation_mdd",
        "deployment_verdict": "do_not_apply_to_live",
        "validation_pass": validation_pass,
        "live_contract": {
            "duration_gate_off": True,
            "entry_notional_multiplier": ENTRY_NOTIONAL_MULTIPLIER,
            "portfolio_total_notional_cap": 3.0,
            "portfolio_eth_share": 0.5,
            "max_entry_notional": MAX_ENTRY_NOTIONAL,
            "chop_soft_size_enabled": True,
            "chop_soft_size_threshold": CHOP_SOFT_SIZE_THRESHOLD,
            "router_priority": list(hazard.greedy.PRIORITY),
            "frozen_exit_head_used": False,
            "frozen_exit_head_omission_reason": "EXIT_THRESHOLD=0.95 is inert in the audited baseline.",
        },
        "hazard_contract": {
            "variant": VARIANT,
            "sl_probability_min": SL_PROBABILITY_MIN,
            "value_margin": VALUE_MARGIN,
            "persistence": PERSISTENCE,
            "models_retrained": False,
        },
        "protocol": {
            "validation": [sweep.VAL_START, sweep.VAL_END],
            "oos": [sweep.OOS_START, sweep.OOS_END],
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "oos_is_untouched": False,
            "oos_limitation": "The source hazard experiment already opened this OOS window; diagnostic only.",
            "barrier_observation": "close_based_frozen_replay_contract",
        },
        "validation": {"live_baseline": validation_baseline, "live_plus_hazard": validation_candidate},
        "oos_diagnostic": {"live_baseline": oos_baseline, "live_plus_hazard": oos_candidate},
    }
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
