#!/usr/bin/env python3
"""RESEARCH ONLY -- ONE-SHOT OOS confirmation of the GBDT h48qual exit_head, GATED on
scripts/research_eth_omega461_gbdt_exit_head_val_20260813.py's VAL gate. This script reads that
script's report.json and REFUSES TO RUN (raises before loading any OOS data) if gate_pass is not
True -- per this project's "VAL 못 이기면 OOS는 절대 보지 말고 부정 결과로 문서화" discipline, this
is enforced in code, not left to operator discretion. No retuning after seeing this result: if run,
this script is not re-run with different parameters regardless of outcome.

Window: research_eth_omega461_exit_sweep_20260721.OOS_START/OOS_END = 2026-01-01..2026-03-31 (this
exit-head research line's standing OOS convention, matching
research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py).

*** MANDATORY CAVEAT (do not drop when reporting this script's numbers) ***
Inherited unchanged from research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py
(docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md): the deployed
quality_threshold values h48qual (0.50) and zig075 (0.75) both run live with -- shared identically
by ALL THREE variants compared here (baseline / TabM live-ATR / GBDT all freeze direction_head /
quality_head / quality_threshold, only exit_head differs) -- were themselves OOS-pnl-primary
selected against a frame spanning 2026-01-01..2026-02-28, the first two of this run's three OOS
months. Because that contamination applies EQUALLY to all three variants (identical frozen
direction/quality/quality_threshold), the RELATIVE comparison (does GBDT beat the TabM live-ATR
baseline within this run) remains meaningful. The ABSOLUTE OOS PnL/MDD figures must NOT be read as
clean, unbiased forward performance.

Portfolio-level only (matches the precedent script's scope -- no component-level OOS table exists
for the TabM live-ATR candidate either).

fresh_forward_bar_by_bar=true (greedy_replay: single causal forward pass). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. direction_head/
quality_head frozen (unchanged across all three variants); only h48qual's exit_head differs.
No duration-gate post-filter applied (matches the VAL-side comparison and current live's gate-off
setting).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or promote anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813 as oos_confirm  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_gbdt_exit_head_val_20260813 as gbdt_val  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_gbdt_exit_head_oos_20260813"
VAL_REPORT_PATH = gbdt_val.OUT_DIR / "report.json"

CAVEAT_TEXT = (
    "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by all three variants "
    "compared here (baseline / TabM live-ATR / GBDT), was OOS-pnl-primary selected against a frame "
    "spanning exactly 2026-01-01..2026-02-28 -- the first two of this run's three OOS months (see "
    "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
    "relative comparison (GBDT vs TabM live-ATR baseline within this run) remains meaningful "
    "because all variants share the identical contaminated entry-selection layer; the absolute OOS "
    "PnL/MDD figures below are not clean unbiased forward performance and must not be "
    "over-interpreted as such."
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _run_variant_oos_gbdt(
    oos_frame: pd.DataFrame, aligned_pred_paths: dict[str, Path], gbdt_models: dict[str, Any], *, fee: float, slip: float,
) -> dict[str, Any]:
    h48qual_cfg = portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE)
    zig075_cfg = portfolio._component_cfg("zig075")
    base_cols = list(torch.load(h48qual_cfg["bundle"], map_location="cpu", weights_only=False)["base_cols"])
    # greedy.prepare_component hardcodes oof=False, the correct convention for OOS predictions
    # (research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py's own pattern).
    h48qual_prepped = greedy.prepare_component(oos_frame, aligned_pred_paths["h48qual"], h48qual_cfg, portfolio.DEVICE)
    h48qual_gbdt = gbdt_val._inject_gbdt_exit_runtime(h48qual_prepped, gbdt_models, portfolio.DEVICE, base_cols)
    zig075_prepped = greedy.prepare_component(oos_frame, aligned_pred_paths["zig075"], zig075_cfg, portfolio.DEVICE)
    components = {"h48qual": h48qual_gbdt, "zig075": zig075_prepped}
    for cname, comp in components.items():
        print(f"  {cname}: nonzero_side={(comp['dec']['side'] != 0).mean():.3f}", flush=True)
    _diag, ledger = greedy.greedy_replay(oos_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE)
    ledger.to_csv(OUT_DIR / "portfolio_ledger_oos_asymmetric_h48qual_gbdt_zig075_original.csv", index=False)
    metrics = portfolio._ledger_metrics(ledger)
    print(f"  asymmetric_h48qual_gbdt_zig075_original: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if not VAL_REPORT_PATH.exists():
        raise RuntimeError(
            f"VAL report not found -- run research_eth_omega461_gbdt_exit_head_val_20260813.py first: {VAL_REPORT_PATH}"
        )
    val_report = json.loads(VAL_REPORT_PATH.read_text(encoding="utf-8"))
    if not bool(val_report.get("gate_pass")):
        raise RuntimeError(
            "VAL gate_pass=False -- GBDT did not beat the TabM live-ATR baseline on VAL "
            "(component+portfolio, PnL+MDD both non-worse). Per this project's methodology "
            "discipline, OOS must not be opened when the VAL gate fails. Aborting without loading "
            "any OOS data. Document the VAL result as a negative finding instead of running this "
            "script with different parameters."
        )
    print("stage=VAL_gate_check pass=True -- proceeding to OOS (one-shot, no retune)", flush=True)
    print("*** CAVEAT ***", flush=True)
    print(CAVEAT_TEXT, flush=True)

    val_results_portfolio = val_report["portfolio_level"]

    print("stage=load_gbdt_bundle", flush=True)
    gbdt_bundle = gbdt_val._load_gbdt_bundle(gbdt_val.GBDT_BUNDLE)
    gbdt_models = gbdt_bundle["models"]

    print("stage=load_oos_frame", flush=True)
    oos_frame_raw = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"  OOS frame rows={len(oos_frame_raw)} range=[{oos_frame_raw['timestamp'].min()}, {oos_frame_raw['timestamp'].max()}]", flush=True)
    fee, slip = omega._load_fee_slip()

    print("stage=align_frame_and_oos_predictions", flush=True)
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    # Reuses oos_confirm's own alignment helper unchanged (drops the WIDE24_2026 95-bar Regime3
    # route-probability coverage gap the same way that script already established as the causally
    # faithful fix; re-deriving that fix here would risk silently diverging from it).
    oos_frame, aligned_pred_paths = oos_confirm._align_frame_and_oos_predictions(oos_frame_raw, q_tags)
    print(f"  aligned rows={len(oos_frame)} (from raw {len(oos_frame_raw)})", flush=True)

    variants = {
        "baseline_both_original": {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")},
        "asymmetric_h48qual_liveatr_zig075_original": {
            "h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE),
            "zig075": portfolio._component_cfg("zig075"),
        },
    }
    oos_results: dict[str, Any] = {}
    for name, comp_cfgs in variants.items():
        print(f"stage=run_variant_oos name={name}", flush=True)
        oos_results[name] = oos_confirm.run_variant_oos(name, comp_cfgs, oos_frame, aligned_pred_paths, fee=fee, slip=slip)

    print("stage=run_variant_oos name=asymmetric_h48qual_gbdt_zig075_original", flush=True)
    oos_results["asymmetric_h48qual_gbdt_zig075_original"] = _run_variant_oos_gbdt(oos_frame, aligned_pred_paths, gbdt_models, fee=fee, slip=slip)

    print("\nstage=val_vs_oos_summary", flush=True)
    val_by_name = {
        "baseline_both_original": val_results_portfolio["baseline_both_original"],
        "asymmetric_h48qual_liveatr_zig075_original": val_results_portfolio["asymmetric_h48qual_liveatr_zig075_original"],
        "asymmetric_h48qual_gbdt_zig075_original": val_results_portfolio["asymmetric_h48qual_gbdt_zig075_original"],
    }
    for name in ("baseline_both_original", "asymmetric_h48qual_liveatr_zig075_original", "asymmetric_h48qual_gbdt_zig075_original"):
        v, o = val_by_name[name], oos_results[name]
        print(f"  {name}: VAL pnl={v['pnl']:.2f} mdd={v['mdd']:.2f} trades={v['trades']} | OOS pnl={o['pnl']:.2f} mdd={o['mdd']:.2f} trades={o['trades']}", flush=True)

    gbdt_oos = oos_results["asymmetric_h48qual_gbdt_zig075_original"]
    tabm_oos = oos_results["asymmetric_h48qual_liveatr_zig075_original"]
    oos_gbdt_nonworse = bool(float(gbdt_oos["pnl"]) >= float(tabm_oos["pnl"]) and float(gbdt_oos["mdd"]) >= float(tabm_oos["mdd"]))
    print(f"stage=oos_relative_result gbdt_nonworse_than_tabm_liveatr={oos_gbdt_nonworse}", flush=True)

    report = {
        "caveat_quality_threshold_oos_contamination": CAVEAT_TEXT,
        "caveat_source_doc": "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md",
        "one_shot_no_retune": True,
        "val_gate_pass_at_open_time": True,
        "oos_window": [sweep.OOS_START, sweep.OOS_END],
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "duration_gate_applied": False,
        "gbdt_bundle": str(gbdt_val.GBDT_BUNDLE),
        "val_results_source": str(VAL_REPORT_PATH),
        "val_results": val_by_name,
        "oos_results": oos_results,
        "oos_gbdt_nonworse_than_tabm_liveatr": oos_gbdt_nonworse,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"\nreport={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
