#!/usr/bin/env python3
"""VAL-select a causal re-arm rule for the frozen ETH Omega4.6.1 hazard exit."""

from __future__ import annotations

import json
import pickle
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


MODEL_ID = "eth_omega461_live_hazard_rearm_20260724"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
REARM_CANDIDATES = (
    ("none", 0),
    ("cooldown", 12),
    ("cooldown", 48),
    ("cooldown", 96),
    ("cooldown", 192),
    ("cooldown", 384),
    ("signal_reset", 0),
    ("signal_reset", 12),
    ("signal_reset", 48),
    ("signal_reset", 96),
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    raise TypeError(type(value).__name__)


def _load_models() -> dict[str, dict[str, Any]]:
    bundles: dict[str, dict[str, Any]] = {}
    for name in sweep.COMPONENTS:
        path = composition.SOURCE_DIR / f"model_{name}_{composition.VARIANT}.pkl"
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("rb") as handle:
            bundles[name] = pickle.load(handle)
    return bundles


def _prepare(split: str):
    if split == "validation":
        frame = sweep.load_frame(
            sweep.VAL_START,
            sweep.VAL_END,
            base_csv=sweep.BASE_2025,
            wide24_csv=sweep.WIDE24_2025,
        )
        prediction_split, oof = "validation", True
    else:
        frame = sweep.load_frame(
            sweep.OOS_START,
            sweep.OOS_END,
            base_csv=sweep.BASE_2026,
            wide24_csv=sweep.WIDE24_2026,
        )
        prediction_split, oof = "oos", False
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
    return frame, components


def _replay(frame, components, bundles, *, rearm_mode: str, rearm_bars: int):
    metrics, ledger = hazard.replay_router(
        frame,
        components,
        rescue_bundles=bundles,
        sl_probability_min=composition.SL_PROBABILITY_MIN,
        value_margin=composition.VALUE_MARGIN,
        persistence=composition.PERSISTENCE,
        entry_notional_multiplier=composition.ENTRY_NOTIONAL_MULTIPLIER,
        max_entry_notional=composition.MAX_ENTRY_NOTIONAL,
        chop_soft_size_threshold=composition.CHOP_SOFT_SIZE_THRESHOLD,
        rearm_mode=rearm_mode,
        rearm_bars=rearm_bars,
    )
    return composition._augment(metrics, ledger), ledger


def _baseline(frame, components):
    metrics, ledger = hazard.replay_router(
        frame,
        components,
        rescue_bundles=None,
        sl_probability_min=1.0,
        value_margin=1.0,
        persistence=1,
        entry_notional_multiplier=composition.ENTRY_NOTIONAL_MULTIPLIER,
        max_entry_notional=composition.MAX_ENTRY_NOTIONAL,
        chop_soft_size_threshold=composition.CHOP_SOFT_SIZE_THRESHOLD,
    )
    return composition._augment(metrics, ledger), ledger


def _passes(metrics: dict[str, Any], baseline: dict[str, Any]) -> bool:
    return bool(
        metrics["pnl"] >= 0.90 * baseline["pnl"]
        and metrics["close_mark_to_market_mdd"] >= baseline["close_mark_to_market_mdd"]
        and metrics["realized_mdd"] >= baseline["realized_mdd"]
        and metrics["rescue_counterfactual_causes"].get("take_profit", 0)
        <= metrics["rescue_counterfactual_causes"].get("stop_loss", 0)
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bundles = _load_models()
    val_frame, val_components = _prepare("validation")
    val_baseline, val_baseline_ledger = _baseline(val_frame, val_components)
    val_baseline_ledger.to_csv(OUT_DIR / "validation_live_baseline_ledger.csv", index=False)

    rows: list[dict[str, Any]] = []
    ledgers = {}
    for rearm_mode, rearm_bars in REARM_CANDIDATES:
        metrics, ledger = _replay(
            val_frame,
            val_components,
            bundles,
            rearm_mode=rearm_mode,
            rearm_bars=rearm_bars,
        )
        row = {"rearm_mode": rearm_mode, "rearm_bars": rearm_bars, **metrics}
        row["validation_pass"] = _passes(row, val_baseline)
        rows.append(row)
        ledgers[(rearm_mode, rearm_bars)] = ledger

    ranking = pd.DataFrame(rows).sort_values(
        ["validation_pass", "log_risk_utility", "close_mark_to_market_mdd", "pnl"],
        ascending=[False, False, False, False],
    )
    ranking.to_csv(OUT_DIR / "validation_ranking.csv", index=False)
    winners = [row for row in rows if row["validation_pass"]]
    selected = (
        max(
            winners,
            key=lambda row: (
                row["log_risk_utility"],
                row["close_mark_to_market_mdd"],
                row["realized_mdd"],
                row["pnl"],
            ),
        )
        if winners
        else None
    )
    report: dict[str, Any] = {
        "model_id": MODEL_ID,
        "status": "val_rejected" if selected is None else "val_selected_oos_diagnostic_pending",
        "deployment_verdict": "do_not_apply_to_live",
        "live_contract": {
            "duration_gate_off": True,
            "entry_notional_multiplier": composition.ENTRY_NOTIONAL_MULTIPLIER,
            "max_entry_notional": composition.MAX_ENTRY_NOTIONAL,
            "chop_soft_size_threshold": composition.CHOP_SOFT_SIZE_THRESHOLD,
        },
        "hazard_contract": {
            "variant": composition.VARIANT,
            "sl_probability_min": composition.SL_PROBABILITY_MIN,
            "value_margin": composition.VALUE_MARGIN,
            "persistence": composition.PERSISTENCE,
            "models_retrained": False,
        },
        "protocol": {
            "selection_split": "validation_only",
            "validation": [sweep.VAL_START, sweep.VAL_END],
            "oos": [sweep.OOS_START, sweep.OOS_END],
            "fresh_forward_bar_by_bar": True,
            "trade_ledgers_used_as_input": False,
            "saved_parent_exit_timestamps_used": False,
            "future_rows_used_for_entry": False,
            "oos_is_untouched": False,
            "oos_limitation": "The source hazard experiment already opened this OOS window; diagnostic only.",
        },
        "validation_baseline": val_baseline,
        "selected": selected,
    }
    if selected is None:
        (OUT_DIR / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
        )
        print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
        return 0

    key = (str(selected["rearm_mode"]), int(selected["rearm_bars"]))
    ledgers[key].to_csv(OUT_DIR / "validation_selected_rearm_ledger.csv", index=False)
    oos_frame, oos_components = _prepare("oos")
    oos_baseline, oos_baseline_ledger = _baseline(oos_frame, oos_components)
    oos_candidate, oos_candidate_ledger = _replay(
        oos_frame,
        oos_components,
        bundles,
        rearm_mode=key[0],
        rearm_bars=key[1],
    )
    oos_baseline_ledger.to_csv(OUT_DIR / "oos_live_baseline_ledger.csv", index=False)
    oos_candidate_ledger.to_csv(OUT_DIR / "oos_selected_rearm_ledger.csv", index=False)
    oos_diagnostic_pass = _passes(oos_candidate, oos_baseline)
    report["status"] = "oos_diagnostic_complete" if oos_diagnostic_pass else "oos_diagnostic_rejected"
    report["oos_diagnostic"] = {"live_baseline": oos_baseline, "live_plus_hazard_rearm": oos_candidate}
    report["oos_diagnostic_pass"] = oos_diagnostic_pass
    report["deployment_verdict"] = (
        "shadow_only_new_untouched_forward_required" if oos_diagnostic_pass else "do_not_apply_to_live"
    )
    (OUT_DIR / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
