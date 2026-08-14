#!/usr/bin/env python3
"""RESEARCH ONLY -- ONE-SHOT OOS confirmation of the asymmetric-adoption portfolio configuration
(h48qual = new quality-head live-ATR-relabeled bundle, zig075 = original frozen bundle) selected on
VAL by scripts/research_eth_omega461_quality_head_portfolio_asymmetric_20260813.py (see that
script's already-saved report.json for the VAL numbers reused here, not recomputed). Coordinator-
authorized single OOS open, gated on component-level AND portfolio-level VAL both showing genuine
improvement (both did -- see this session's report). No retuning after seeing this result: this
script is not re-run with different parameters regardless of outcome, matching tonight's standing
"VAL-first, one OOS look" discipline (same rule the exit-head-fix's own OOS confirmation followed,
scripts/research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py, reused as the
structural template here).

Window: research_eth_omega461_exit_sweep_20260721.OOS_START/OOS_END = 2026-01-01..2026-03-31.

*** MANDATORY CAVEAT (do not drop when reporting this script's numbers), inherited verbatim from the
exit-head fix's own OOS confirmation because it applies identically here ***
docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md found by direct
code read that the deployed quality_threshold values both h48qual (0.50) and zig075 (0.75) actually
run live with -- which BOTH the baseline and this quality-head candidate depend on IDENTICALLY,
since this experiment never changes quality_threshold, only quality_head's WEIGHTS -- were
themselves selected by
scripts/train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py:1173 sorting candidates by
`(oos_pnl, validation_pnl)`, i.e. OOS PnL as the PRIMARY key, against a frame spanning exactly
2026-01-01..2026-02-28 (the first two of this run's three OOS months). Because that contamination
applies EQUALLY to baseline and candidate (same frozen quality_threshold in both), the RELATIVE
comparison (does the new quality head beat baseline within this run) remains meaningful. The
ABSOLUTE OOS PnL/MDD figures must NOT be read as clean, unbiased forward performance.

ONE prediction file is regenerated here (permitted per the coordinator's explicit instruction,
CPU-only): OOS predictions for h48qual's NEW quality-head bundle, via
research_eth_omega461_quality_head_liveatr_relabel_20260813._fresh_predictions (imported and reused
unchanged, the same already-self-verified function used for the VAL-side component/portfolio
checks) applied to the OOS frame instead of VAL. h48qual's BASELINE and zig075 (both variants) use
the existing, already-established static oos_predictions_qXXX.csv files -- unchanged, untouched.

fresh_forward_bar_by_bar=true (greedy_replay: single causal forward pass). trade_ledgers_used_as_input=false.
saved_parent_exit_timestamps_used=false. future_rows_used_for_entry=false. direction_head/exit_head
unchanged for h48qual; zig075 and regime3 routing completely unchanged. No duration-gate post-filter
(matches the VAL-side comparison and current live's gate-off setting).

Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py, runtime_config.py, .env.
Does NOT overwrite any live checkpoint or promote anything.
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
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_quality_head_liveatr_relabel_20260813 as quality_relabel  # noqa: E402
import research_eth_omega461_quality_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_quality_head_portfolio_asymmetric_oos_confirm_20260813"
VAL_REPORT_PATH = ROOT / "tmp/causal_regen_20260516/eth_omega461_quality_head_portfolio_asymmetric_20260813/report.json"
DEVICE_CPU = torch.device("cpu")  # explicit CPU-only, per the coordinator's constraint on any regenerated prediction file

CAVEAT_TEXT = (
    "quality_threshold (h48qual=0.50, zig075=0.75), shared identically by baseline and candidate and "
    "never changed by this quality-head-relabel experiment, was OOS-pnl-primary selected against a "
    "frame spanning exactly 2026-01-01..2026-02-28 -- the first two of this run's three OOS months "
    "(see docs/experiments/eth_omega461_oos_selection_bias_scope_and_resolution_20260813.md). The "
    "relative comparison (candidate vs baseline within this run) remains meaningful because both "
    "share the identical contaminated entry-selection layer; the absolute OOS PnL/MDD figures below "
    "are not clean unbiased forward performance and must not be over-interpreted as such."
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _drop_route_coverage_gap(oos_frame: pd.DataFrame) -> pd.DataFrame:
    """WIDE24_2026 (Regime3 regime-probability overlay) has a known 95-row/0.37% coverage gap
    (2026-02-28 16:05..23:55, per docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md
    "후속 3" and research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813.py's own
    identical fix) -- hard._route_id raises on non-finite route probabilities, and a live system
    cannot route a bar it has no regime probabilities for either, so dropping those bars is the
    causally faithful choice. Must run BEFORE any prediction generation (hard._route_id is called
    inside quality_relabel._fresh_predictions too, not just inside greedy_replay's own routing)."""
    route_ok = np.isfinite(oos_frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    n_bad = int((~route_ok).sum())
    if n_bad:
        oos_frame = oos_frame[route_ok].reset_index(drop=True)
        print(f"  dropped {n_bad} bars with non-finite Regime3 route probabilities (WIDE24_2026 coverage gap)", flush=True)
    return oos_frame


def _generate_new_h48qual_oos_predictions(oos_frame: pd.DataFrame) -> Path:
    """quality_relabel._fresh_predictions always builds columns under the "omega1_regime3_expertdq_oof_"
    prefix (matching oof=True consumption -- that's what it was built for on the VAL side). The
    static oos_predictions_qXXX.csv files instead use "omega1_regime3_expertdq_" (no "_oof_"), which
    is what greedy.prepare_component's hardcoded `_to_decisions(pred, oof=False)` call expects --
    confirmed directly from train_eval_omega4_3head_parent72_loose_entry_quality_20260620.py's own
    main() (lines ~1128-1129), which does the exact same rename (oos_src_oof -> oos_src) before
    saving its own OOS CSVs; the underlying predicted values are identical, this is a column-naming
    convention only. Reproduced here rather than modifying `_fresh_predictions` itself (VAL callers
    still need the "_oof_" naming)."""
    cfg = dict(sweep.COMPONENTS["h48qual"])
    src_oof, q_tag = quality_relabel._fresh_predictions(cfg, oos_frame, portfolio.NEW_H48QUAL_BUNDLE, device=DEVICE_CPU)
    src = src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in src_oof.columns})
    out_path = OUT_DIR / f"oos_predictions_new_h48qual_{q_tag}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    src.to_csv(out_path, index=False)
    return out_path


def _align_frame_and_oos_predictions(oos_frame: pd.DataFrame, pred_paths: dict[str, Path]) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Same route-coverage-gap fix + intersection-alignment as
    research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813._align_frame_and_oos_predictions,
    generalized to explicit {label: pred_csv_path} (h48qual's path differs by variant here, unlike
    that predecessor). Caller is expected to have already run `_drop_route_coverage_gap`; this also
    re-applies it defensively (idempotent, a no-op if already clean) in case `oos_frame` here came
    from a different caller."""
    oos_frame = _drop_route_coverage_gap(oos_frame)
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(oos_frame["timestamp"])
    for label, path in pred_paths.items():
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[label] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = oos_frame[oos_frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    for label, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{label}: OOS alignment failed after timestamp intersection")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = OUT_DIR / f"_aligned_oos_{label}_predictions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        aligned_paths[label] = out_path
    return aligned_frame, aligned_paths


def run_variant_oos(name: str, comp_cfgs: dict[str, dict[str, Any]], oos_frame: pd.DataFrame,
                     pred_paths: dict[str, Path], *, fee: float, slip: float) -> dict[str, Any]:
    components = {}
    for cname, cfg in comp_cfgs.items():
        # greedy.prepare_component hardcodes oof=False, the correct convention for OOS predictions
        # (both the static baseline/zig075 CSVs and this session's freshly regenerated h48qual CSV
        # are built oof=False-shaped -- _fresh_predictions itself doesn't set an oof flag, and
        # parent._to_decisions(oof=False) is what greedy.prepare_component applies uniformly).
        components[cname] = greedy.prepare_component(oos_frame, pred_paths[cname], cfg, DEVICE_CPU)
        print(f"  {cname}: bundle={Path(cfg['bundle']).parent.name} nonzero_side={(components[cname]['dec']['side'] != 0).mean():.3f}", flush=True)
    _diag, ledger = greedy.greedy_replay(oos_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=DEVICE_CPU)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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
    oos_frame_raw = _drop_route_coverage_gap(oos_frame_raw)

    print("stage=generate_new_h48qual_oos_predictions (CPU-only fresh forward pass, not training)", flush=True)
    new_h48qual_oos_csv = _generate_new_h48qual_oos_predictions(oos_frame_raw)
    print(f"  saved {new_h48qual_oos_csv}", flush=True)

    h48qual_baseline_static = sweep.EXT_PRED_DIR / "h48qual" / f"oos_predictions_{sweep.COMPONENTS['h48qual']['q_tag']}.csv"
    zig075_static = sweep.EXT_PRED_DIR / "zig075" / f"oos_predictions_{sweep.COMPONENTS['zig075']['q_tag']}.csv"

    print("stage=align_frame_and_oos_predictions", flush=True)
    oos_frame, aligned = _align_frame_and_oos_predictions(oos_frame_raw, {
        "h48qual_baseline": h48qual_baseline_static,
        "h48qual_new": new_h48qual_oos_csv,
        "zig075": zig075_static,
    })
    print(f"  aligned rows={len(oos_frame)} (from raw {len(oos_frame_raw)})", flush=True)

    variants = {
        "baseline_both_original": {
            "comp_cfgs": {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")},
            "pred_paths": {"h48qual": aligned["h48qual_baseline"], "zig075": aligned["zig075"]},
        },
        "asymmetric_h48qual_quality_liveatr_zig075_original": {
            "comp_cfgs": {"h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE), "zig075": portfolio._component_cfg("zig075")},
            "pred_paths": {"h48qual": aligned["h48qual_new"], "zig075": aligned["zig075"]},
        },
    }

    oos_results: dict[str, Any] = {}
    for name, v in variants.items():
        print(f"stage=run_variant_oos name={name}", flush=True)
        oos_results[name] = run_variant_oos(name, v["comp_cfgs"], oos_frame, v["pred_paths"], fee=fee, slip=slip)

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
        "scale_map_applied": True,
        "new_h48qual_bundle": str(portfolio.NEW_H48QUAL_BUNDLE),
        "new_h48qual_oos_pred_csv": str(new_h48qual_oos_csv),
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
