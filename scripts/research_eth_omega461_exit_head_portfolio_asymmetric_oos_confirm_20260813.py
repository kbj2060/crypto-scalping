#!/usr/bin/env python3
"""RESEARCH ONLY -- ONE-SHOT OOS confirmation of the asymmetric-adoption portfolio configuration
(h48qual = new live-ATR-relabeled exit head, zig075 = original frozen exit head) selected on VAL
by scripts/research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py (see that script's
already-saved report.json for the VAL numbers reused here, not recomputed). Coordinator-authorized
single OOS open -- per the "VAL-first, OOS opened once for the selected config" discipline this
whole exit-head research thread has followed all night. No retuning after seeing this result: this
script is not re-run with different parameters regardless of outcome.

Window: research_eth_omega461_exit_sweep_20260721.OOS_START/OOS_END = 2026-01-01..2026-03-31 (this
exit-head research line's standing OOS convention throughout tonight).

*** MANDATORY CAVEAT (do not drop when reporting this script's numbers) ***
docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md (a separate,
same-night investigation) found by direct code read that the deployed quality_threshold values
both h48qual (0.50) and zig075 (0.75) actually run live with -- which BOTH the baseline and this
asymmetric candidate depend on identically, since direction/quality heads and quality_threshold are
frozen and unchanged by this whole exit-head research line -- were themselves selected by
scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173 sorting candidates by
`(oos_pnl, validation_pnl)`, i.e. OOS PnL as the PRIMARY key. The exact "oos" frame that selection
optimized against is confirmed (by that document's own direct pandas read) to span 2026-01-01
through 2026-02-28 -- the first two of this script's three OOS months. So the entry-selection layer
shared by both configurations compared here was directly tuned to look good on most of this same
OOS window, before this script ever ran. Because that contamination applies EQUALLY to baseline and
candidate (same frozen quality_threshold/quality_head/direction_head in both), the RELATIVE
comparison below (does the new exit head beat baseline within this run) remains meaningful.
The ABSOLUTE OOS PnL/MDD figures themselves must NOT be read as clean, unbiased forward
performance -- they are inflated in ways not specific to the exit-head change being tested here.

fresh_forward_bar_by_bar=true (greedy_replay: single causal forward pass). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. direction/quality frozen
(unchanged in both variants); only h48qual's exit_head weights differ between variants. No
duration-gate post-filter applied (matches the VAL-side comparison and current live's gate-off
setting; keeps this run isolated to the one variable already isolated on VAL).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or promote anything.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813"
VAL_REPORT_PATH = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_portfolio_asymmetric_20260813/report.json"

CAVEAT_TEXT = (
    "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by baseline and candidate, "
    "was OOS-pnl-primary selected against a frame spanning exactly 2026-01-01..2026-02-28 -- the "
    "first two of this run's three OOS months (see "
    "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
    "relative comparison (candidate vs baseline within this run) remains meaningful because both "
    "share the identical contaminated entry-selection layer; the absolute OOS PnL/MDD figures "
    "below are not clean unbiased forward performance and must not be over-interpreted as such."
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _align_frame_and_oos_predictions(oos_frame: pd.DataFrame, q_tags: dict[str, str]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Same intersection-alignment as portfolio._align_frame_and_predictions, for
    oos_predictions_qXXX.csv instead of validation_predictions_qXXX.csv. Also drops bars with
    non-finite Regime3 route probabilities (WIDE24_2026 has a 95-row / 0.37% coverage gap on
    2026-02-28 16:05..23:55 -- hard._route_id would otherwise raise; a live system also cannot
    route a bar it has no regime probabilities for, so excluding those bars from the replay is the
    causally faithful choice, not a result-shaping one). Discovered and fixed while running this
    one-shot OOS confirmation, not a retune of the exit-head comparison itself."""
    n_route_bad = int((~np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)).sum())
    if n_route_bad:
        oos_frame = oos_frame[np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)].reset_index(drop=True)
        print(f"  dropped {n_route_bad} bars with non-finite Regime3 route probabilities (WIDE24_2026 coverage gap)", flush=True)
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(oos_frame["timestamp"])
    for cname, q_tag in q_tags.items():
        pred_csv = sweep.EXT_PRED_DIR / cname / f"oos_predictions_{q_tag}.csv"
        df = pd.read_csv(pred_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[cname] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = oos_frame[oos_frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    for cname, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{cname}: OOS alignment failed after timestamp intersection")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = OUT_DIR / f"_aligned_oos_{cname}_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        aligned_paths[cname] = out_path
    return aligned_frame, aligned_paths


def run_variant_oos(
    name: str,
    comp_cfgs: dict[str, dict[str, Any]],
    oos_frame: pd.DataFrame,
    aligned_pred_paths: dict[str, Path],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    components = {}
    for cname, cfg in comp_cfgs.items():
        # greedy.prepare_component hardcodes oof=False, which is the CORRECT convention for OOS
        # predictions (unlike VAL, which needed the oof=True local copy in the portfolio script).
        components[cname] = greedy.prepare_component(oos_frame, aligned_pred_paths[cname], cfg, portfolio.DEVICE)
        print(f"  {cname}: bundle={Path(cfg['bundle']).parent.name} nonzero_side={(components[cname]['dec']['side'] != 0).mean():.3f}", flush=True)
    _diag, ledger = greedy.greedy_replay(oos_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=portfolio.DEVICE)
    ledger.to_csv(OUT_DIR / f"portfolio_ledger_oos_{name}.csv", index=False)
    metrics = portfolio._ledger_metrics(ledger)
    print(f"  {name}: {json.dumps({k: v for k, v in metrics.items() if k not in ('reason_counts', 'source_component_counts')})}", flush=True)
    return metrics


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("*** CAVEAT ***", flush=True)
    print(CAVEAT_TEXT, flush=True)

    val_report = json.loads(VAL_REPORT_PATH.read_text(encoding="utf-8"))
    val_results = val_report["results"]

    print("stage=load_oos_frame", flush=True)
    oos_frame_raw = sweep.load_frame(sweep.OOS_START, sweep.OOS_END, base_csv=sweep.BASE_2026, wide24_csv=sweep.WIDE24_2026)
    print(f"  OOS frame rows={len(oos_frame_raw)} range=[{oos_frame_raw['timestamp'].min()}, {oos_frame_raw['timestamp'].max()}]", flush=True)
    fee, slip = omega._load_fee_slip()

    print("stage=align_frame_and_oos_predictions", flush=True)
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in ("h48qual", "zig075")}
    oos_frame, aligned_pred_paths = _align_frame_and_oos_predictions(oos_frame_raw, q_tags)
    print(f"  aligned rows={len(oos_frame)} (from raw {len(oos_frame_raw)})", flush=True)

    variants = {
        "baseline_both_original": {
            "h48qual": portfolio._component_cfg("h48qual"),
            "zig075": portfolio._component_cfg("zig075"),
        },
        "asymmetric_h48qual_liveatr_zig075_original": {
            "h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE),
            "zig075": portfolio._component_cfg("zig075"),
        },
    }

    oos_results: dict[str, Any] = {}
    for name, comp_cfgs in variants.items():
        print(f"stage=run_variant_oos name={name}", flush=True)
        oos_results[name] = run_variant_oos(name, comp_cfgs, oos_frame, aligned_pred_paths, fee=fee, slip=slip)

    print("\nstage=val_vs_oos_summary", flush=True)
    for name in variants:
        v, o = val_results[name], oos_results[name]
        print(f"  {name}: VAL pnl={v['pnl']:.2f} mdd={v['mdd']:.2f} trades={v['trades']} | OOS pnl={o['pnl']:.2f} mdd={o['mdd']:.2f} trades={o['trades']}", flush=True)

    report = {
        "caveat_quality_threshold_oos_contamination": CAVEAT_TEXT,
        "caveat_source_doc": "docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md",
        "one_shot_no_retune": True,
        "oos_window": [sweep.OOS_START, sweep.OOS_END],
        "val_window": [sweep.VAL_START, sweep.VAL_END],
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "duration_gate_applied": False,
        "new_h48qual_bundle": str(portfolio.NEW_H48QUAL_BUNDLE),
        "val_results_source": str(VAL_REPORT_PATH),
        "val_results": val_results,
        "oos_results": oos_results,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"\nreport={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
