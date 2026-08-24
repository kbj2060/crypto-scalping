#!/usr/bin/env python3
"""RESEARCH ONLY -- ETH conformal veto cheap_gate: before training the conformal downside-LCB regressors
(docs/model_contracts/eth_candidate_conformal_downside_veto_contract_20260816.md), check whether the
FREE signal already computed by each component -- its own `quality_score` (`quality_for_action`,
already in `dec["quality_score"]` for every prepared component) -- already captures similar
downside-filtering value if the entry threshold is simply raised. No model, no new feature: this
post-filters the ALREADY-thresholded `dec["side"]` (originally gated at h48qual>=0.50, zig075>=0.75)
to a higher quality_score cutoff, which is a valid monotonic restriction (raising the bar can only
remove previously-accepted entries, never add new ones).

Two views of the same sweep, both required by the contract's cheap_gate section: (1) the absolute
quality_score threshold, and (2) the percentile of originally-accepted signal bars that threshold
would cut (the "확률 분위수 컷" framing) -- reported together per row so both readings are free.

VAL ONLY. OOS-Q1/OOS-Q2 not opened by this script.

fresh_forward_bar_by_bar=true (same causal replay, only difference is a static per-bar filter
applied to an already-causal decision column). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/omega4_6_1_live.py /
trading_bot_modules/runtime_config.py / .env. Does NOT modify any imported module. No retraining,
no GPU (DEVICE=cpu).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_omega461_zig075_short_entry_veto_sustained_uptrend_20260814 as o4  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_conformal_veto_cheap_gate_20260816"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05
WINDOW = "val"

G0_ODYSSEY4_VAL_WITH_GATE = {"pnl": 77.31, "mdd": -21.76, "trades": 26}
G0_ODYSSEY4_VAL_NO_GATE = {"pnl": 41.13, "mdd": -21.70, "trades": 35}

BASE_THRESHOLD = {"h48qual": 0.50, "zig075": 0.75}
THRESHOLD_GRID = {
    "h48qual": (0.50, 0.55, 0.60, 0.65, 0.70),
    "zig075": (0.75, 0.80, 0.85, 0.90),
}


def log(msg: str) -> None:
    print(f"[candidate_conformal_veto_cheap_gate] {msg}", flush=True)


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _metrics_pair(ledger: pd.DataFrame, aligned_frame: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any]]:
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return no_gate, with_gate


def _raise_quality_threshold(components: dict[str, Any], name: str, new_threshold: float) -> dict[str, Any]:
    """Shallow-copy `components` and `components[name]`, then re-gate `dec["side"]`/`dec["action"]`
    to CASH wherever `dec["quality_score"] < new_threshold`. Never mutates the input -- same pattern
    as o4._attach_veto_mask. Valid only as a RAISE vs the component's own baked-in threshold (the
    already-computed dec is gated at BASE_THRESHOLD[name]; asking for a lower value would be a
    no-op at best, not a real lower gate, since already-rejected bars were never scored)."""
    out = dict(components)
    comp = dict(out[name])
    dec = comp["dec"].copy()
    cut = dec["quality_score"].to_numpy(dtype=np.float64) < float(new_threshold)
    dec.loc[cut, "side"] = 0
    if "action" in dec.columns:
        dec.loc[cut, "action"] = 0
    comp["dec"] = dec
    out[name] = comp
    return out


def _signal_percentile_cut(components: dict[str, Any], name: str, threshold: float) -> float:
    """Of the bars the component's ORIGINAL (base-threshold) side already accepted, what fraction
    would this higher threshold additionally remove -- i.e. the percentile-of-signals-cut framing
    the contract's cheap_gate item 2 asks for, computed for free from the same quality_score column."""
    dec = components[name]["dec"]
    side = pd.to_numeric(dec["side"], errors="raise").to_numpy()
    active = side != 0
    if not active.any():
        return 0.0
    q = dec["quality_score"].to_numpy(dtype=np.float64)[active]
    return float(np.mean(q < float(threshold)))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": "ETH conformal veto cheap_gate -- raise quality_score threshold per component (free signal, no model), VAL only, vs Odyssey4 G0.",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window": WINDOW,
    }

    log("=== stage=detector_build ===")
    score_by_base, robustness_thresholds, threshold = guard.build_detector()

    log("=== stage=prepare_val ===")
    aligned_frame, components, prep_diag = guard.prepare_regime_aware_components(WINDOW, gate.load_all_windows(), score_by_base, threshold, OUT_DIR, device)
    mask, _ = guard._detector_mask_for_frame(aligned_frame, WINDOW, score_by_base, threshold)
    veto_components = o4._attach_veto_mask(components, mask)

    log("=== stage=G0_reproduce ===")
    diag0, ledger0 = o4.greedy_replay_entry_veto(aligned_frame, veto_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    no_gate0, with_gate0 = _metrics_pair(ledger0, aligned_frame)
    g0_ok = _close(no_gate0, G0_ODYSSEY4_VAL_NO_GATE) and _close(with_gate0, G0_ODYSSEY4_VAL_WITH_GATE)
    report["g0_reproduce"] = {"no_gate": no_gate0, "with_gate": with_gate0, "pass": g0_ok}
    log(f"  no_gate={no_gate0['pnl']:.2f}%/{no_gate0['mdd']:.2f}%/{no_gate0['trades']}  with_gate={with_gate0['pnl']:.2f}%/{with_gate0['mdd']:.2f}%/{with_gate0['trades']}  match={g0_ok}")
    if not g0_ok:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 reproduction failed -- aborting before trusting any threshold sweep."
        _write_report(report)
        log("stage=ABORT G0 failed")
        return 1

    for name in ("h48qual", "zig075"):
        log(f"=== stage=threshold_sweep_{name} ===")
        rows = []
        for thr in THRESHOLD_GRID[name]:
            comps = _raise_quality_threshold(veto_components, name, thr)
            diag, ledger = o4.greedy_replay_entry_veto(aligned_frame, comps, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
            no_gate, with_gate = _metrics_pair(ledger, aligned_frame)
            pct_cut = _signal_percentile_cut(veto_components, name, thr)
            row = {"threshold": thr, "pct_of_base_signals_cut": pct_cut, "no_gate": no_gate, "with_gate": with_gate}
            rows.append(row)
            log(f"  {name} thr={thr:.2f}  cuts {pct_cut*100:5.1f}% of base signals  "
                f"no_gate={no_gate['pnl']:7.2f}%/{no_gate['mdd']:7.2f}%/{no_gate['trades']:3d}  "
                f"with_gate={with_gate['pnl']:7.2f}%/{with_gate['mdd']:7.2f}%/{with_gate['trades']:3d}")
        report[f"threshold_sweep_{name}"] = rows

    report["stage_reached"] = "done"
    report["gate_pass"] = True
    _write_report(report)
    log("stage=done")
    return 0


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
