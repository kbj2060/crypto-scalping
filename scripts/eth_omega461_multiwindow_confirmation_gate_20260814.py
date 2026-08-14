#!/usr/bin/env python3
"""Odyssey2 -- reusable multi-window confirmation gate infrastructure (2026-08-14).

Implements docs/experiments/eth_val_oos_regime_mismatch_investigation_20260813.md's practical
recommendation #2, verbatim: "3개월 단일 OOS 창도 그 자체로는 약한 증거임을 인정한다 ... 앞으로 새
후보는 최소 VAL+OOS-Q1+OOS-Q2, 가능하면 2025 Q1~Q3까지 포함한 4개 이상의, 상승·하락 방향이 섞인
독립 구간에서 부호 일치를 확인하기 전엔 '확인됨'이라고 쓰지 않는다." That document (independently,
the same night as 6 different post-entry candidates VAL-won-then-OOS-reversed) diagnosed the root
cause as a 3x-stacked selection-bias structure (risk-sizing -> quality_threshold -> new candidate,
all fit/selected against the SAME ~26,000-bar/22-38-trade VAL window) rather than a genuine regime
shift -- VAL and OOS-Q1 2026 were shown to have nearly IDENTICAL volatility/regime composition, so a
single VAL-then-OOS-Q1 pass cannot distinguish a real edge from VAL-selection noise. This module
turns that recommendation into an enforceable, reusable gate rather than a prose reminder.

=== The 6 pre-registered windows ===
  2025q1/2025q2/2025q3 (context, oof=True, train_predictions_*.csv)  -- diagnostic only, shown in
    every report table but NEVER gates pass/fail (in-sample: these bars were inside each bundle's
    own TRAIN split).
  val (2025-10-01..12-31, oof=True, validation_predictions_*.csv)    -- unchanged role: primary
    tuning/selection window, or whatever relaxed criterion a given candidate is judged under
    (docs/experiments/eth_omega461_relaxed_gate_rescoring_20260814.md). A VAL loss is still strong
    evidence against; a VAL win is still only "earns the right to open OOS", never "confirmed".
  oos_q1 (2026-01-01..03-31, oof=False, oos_predictions_*.csv)       -- OOS-confirm tier.
  oos_q2 (2026-04-01..06-30, oof=False, oos_predictions_*.csv)       -- OOS-confirm tier.

=== Pre-registered pass/fail criterion (fixed BEFORE this module was applied to any candidate;
see docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md) ===
  1. VAL role is unchanged (handled entirely by the CALLER, upstream of this module).
  2. The formal OOS confirmation is now a SINGLE TOUCH that opens oos_q1 AND oos_q2 TOGETHER in one
     run (never sequentially/iteratively) -- both windows must show with_gate PnL non-worse than
     baseline (optionally within a pre-registered MDD slack, e.g. the relaxed gate's 3pp allowance)
     for the candidate to be called "confirmed". If only one of the two passes, the candidate is
     REJECTED for sign mismatch -- exactly the pattern this module exists to catch (queue-pressure
     and risk-controlled both VAL-won, single-OOS-touch-reversed on oos_q1 alone; see below).
  3. 2025q1/q2/q3 are always computed and shown in the report table for context, but never enter
     the pass/fail decision and never block opening OOS.

=== What this module is and is NOT responsible for ===
It IS responsible for: loading the 6 windows correctly (load_all_windows), verifying that loading
(verify_windows), a generic frame<->prediction-CSV alignment utility for portfolio-level replay on
ANY of the 6 windows (align_frame_and_predictions -- a parameterized generalization of the
near-identical, split-hardcoded `_align_frame_and_predictions`/`_align_frame_and_oos_predictions`
helpers duplicated across research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py /
research_eth_omega461_queue_pressure_exit_threshold_20260814.py /
research_eth_omega461_risk_controlled_exit_fallback_20260814.py -- none of those are edited, this is
a new function), a baseline-shape-only portfolio replay runner for arbitrary component configs on
any window (run_portfolio_variant -- NO candidate-specific intervention logic, just
prepare+greedy_replay+metrics, reused by G0 and by both candidates below as their shared
"asymmetric_tabm_liveatr" comparator), and the pass/fail table/verdict builder
(summarize_multiwindow).

It is NOT a universal per-candidate evaluator. Each candidate's own intervention (final_action
rewrite, exit_threshold modulation, exit_prob risk-controlled switching, ...) lives in that
candidate's OWN script and is imported/reused as-is -- never reimplemented here. This module's
job stops at "hand the candidate script a correctly-loaded, correctly-aligned window"; the actual
evaluate-this-candidate-on-this-window glue is written per call site (see main() below for the two
worked examples: queue-pressure and risk-controlled).

=== Compliance (same standard as every Odyssey2 script this session) ===
fresh_forward_bar_by_bar=true (every replay this module drives is a single causal bar-by-bar
forward pass via unmodified greedy.greedy_replay / candidate-specific renamed copies of it -- this
module adds no new bar-by-bar simulation logic of its own, only window loading/alignment glue).
trade_ledgers_used_as_input=false (ledgers are written-only outputs). saved_parent_exit_timestamps_
used=false. future_rows_used_for_entry=false. Does NOT touch trading_bot.py /
trading_bot_modules/omega4_6_1_live.py / runtime_config.py / .env. Does NOT modify any imported
module -- research_eth_omega461_exit_sweep_20260721.py, replay_omega4_6_1_greedy_router_20260706.py,
research_eth_omega461_exit_head_portfolio_asymmetric_20260813.py,
research_eth_omega461_live_sltp_mfe_width_20260813.py,
research_eth_omega461_queue_pressure_exit_threshold_20260814.py, and
research_eth_omega461_risk_controlled_exit_fallback_20260814.py are all imported and read only.
No retraining, no GPU -- every prediction CSV this module reads already exists on disk.
"""
from __future__ import annotations

import json
import pickle
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
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import audit_omega4_6_1_phase1_robustness_20260707 as audit  # noqa: E402
import research_eth_omega461_queue_pressure_exit_threshold_20260814 as qp_mod  # noqa: E402
import research_eth_omega461_risk_controlled_exit_fallback_20260814 as rc_mod  # noqa: E402
from test_omega4_6_1_drop_h48qual_20260706 import _metrics as legacy_duration_gate_metrics  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_multiwindow_confirmation_gate_20260814"
DEVICE = portfolio.DEVICE
G0_TOLERANCE_PP = 0.05  # percentage points on pnl/mdd; trades must match exactly (deterministic replay)

# =====================================================================================================
# (a) Window definitions.
#
# End-boundary convention (verified empirically before writing this module, not assumed -- see
# docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md "창 경계" section):
#   - val/oos_q1 use sweep.VAL_START/VAL_END/OOS_START/OOS_END LITERALLY (no time-of-day suffix).
#     Every already-published reference number this module reproduces below (asymmetric_tabm_
#     liveatr 46.59/77.31/93.27/67.25 etc.) was computed via research_eth_omega461_exit_sweep_
#     20260721.load_frame(sweep.VAL_START, sweep.VAL_END, ...) called with those exact bare date
#     strings, which pandas compares against midnight of the end date -- i.e. the last calendar day
#     of each window is truncated to its 00:00 bar. Adding a time-of-day suffix would silently
#     change the frame and break reproduction.
#   - 2025q1/q2/q3 use "23:59:59"-suffixed ends, matching audit_omega4_6_1_phase1_robustness_
#     20260707.load_2025_quarter_components's own explicit choice (its `quarters` tuple literally
#     contains "2025-03-31 23:59:59" etc.) -- confirmed this is NOT cosmetic: using the bare date
#     instead drops 287 bars (the entire last day) for Q1 alone (25631 vs 25918 rows) and fails to
#     reproduce the published 28.54/-20.62/19 reference.
#   - oos_q2 has no prior published reference (first use of this window in the project). It uses
#     the bare-date convention (matching its sibling oos_q1, for single-touch-pair symmetry) rather
#     than "23:59:59": WIDE24_2026's overlay file was discovered (while building this module) to
#     end at exactly 2026-06-30 00:00:00, so a "23:59:59" end would produce 287 route-probability-
#     NaN bars on the last day (same coverage-limit class as the already-documented 95-bar Feb-28
#     gap, just at the tail instead of mid-file) that then have to be dropped anyway -- the bare-date
#     boundary reaches the identical final frame without ever materializing then discarding those
#     rows.
# =====================================================================================================
WINDOW_DEFS: dict[str, dict[str, Any]] = {
    "2025q1": {"start": "2025-01-01", "end": "2025-03-31 23:59:59", "split": "train", "oof": True, "tier": "context", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    "2025q2": {"start": "2025-04-01", "end": "2025-06-30 23:59:59", "split": "train", "oof": True, "tier": "context", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    "2025q3": {"start": "2025-07-01", "end": "2025-09-30 23:59:59", "split": "train", "oof": True, "tier": "context", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    "val":    {"start": sweep.VAL_START, "end": sweep.VAL_END, "split": "validation", "oof": True, "tier": "val", "base_csv": sweep.BASE_2025, "wide24_csv": sweep.WIDE24_2025},
    "oos_q1": {"start": sweep.OOS_START, "end": sweep.OOS_END, "split": "oos", "oof": False, "tier": "oos_confirm", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
    "oos_q2": {"start": "2026-04-01", "end": "2026-06-30", "split": "oos", "oof": False, "tier": "oos_confirm", "base_csv": sweep.BASE_2026, "wide24_csv": sweep.WIDE24_2026},
}
CONTEXT_WINDOWS = ("2025q1", "2025q2", "2025q3")
VAL_WINDOW = "val"
OOS_CONFIRM_WINDOWS = ("oos_q1", "oos_q2")  # single-touch: always opened together, never sequentially
ALL_WINDOWS = tuple(WINDOW_DEFS.keys())

_SPLIT_FILE_PREFIX = {"train": "train_predictions", "validation": "validation_predictions", "oos": "oos_predictions"}


def _pred_path(name: str, split: str) -> Path:
    cfg = sweep.COMPONENTS[name]
    return sweep.EXT_PRED_DIR / name / f"{_SPLIT_FILE_PREFIX[split]}_{cfg['q_tag']}.csv"


def _drop_route_nan(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Same WIDE24_*-coverage-gap fix used verbatim by every existing OOS-touching script in this
    lineage (research_eth_omega461_queue_pressure_exit_threshold_20260814._align_frame_and_oos_
    predictions / research_eth_omega461_risk_controlled_exit_fallback_20260814 main() /
    research_eth_omega461_exit_head_portfolio_asymmetric_oos_confirm_20260813) -- copied here, not
    reimplemented differently, so hard._route_id never raises 'non-finite Regime3 route
    probabilities' downstream. Applied uniformly to all 6 windows (harmless no-op where the count is
    0, verified true for every 2025-based window and for oos_q1/oos_q2 under the bare-date
    convention above -- see module docstring)."""
    ok = np.isfinite(frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)).all(axis=1)
    n_bad = int((~ok).sum())
    if n_bad:
        frame = frame[ok].reset_index(drop=True)
    return frame, n_bad


def load_all_windows() -> dict[str, dict[str, Any]]:
    """Load all 6 pre-registered windows. Returns {window_name: {"frame": DataFrame, "raw":
    {"h48qual": DataFrame, "zig075": DataFrame}, "raw_paths": {"h48qual": Path, "zig075": Path},
    "oof": bool, "tier": "context"|"val"|"oos_confirm", "route_nan_dropped": int}}.

    "raw" holds the FULL (unsliced) prediction CSV for each component as a DataFrame, for direct
    inspection/verification (see verify_windows) -- it is NOT pre-intersected with "frame". Every
    downstream evaluate_component-style function in this project's lineage
    (research_eth_omega461_exit_sweep_20260721.prep_component,
    research_eth_omega461_exit_head_portfolio_asymmetric_20260813._prepare_component_val via
    align_frame_and_predictions below) does its own timestamp intersection internally, so callers
    should pass "raw_paths" (or the original EXT_PRED_DIR path it points to) directly rather than
    writing a new pre-sliced CSV -- confirmed empirically before this module was written (see
    docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md).
    """
    windows: dict[str, dict[str, Any]] = {}
    for wname, wd in WINDOW_DEFS.items():
        frame = sweep.load_frame(wd["start"], wd["end"], base_csv=wd["base_csv"], wide24_csv=wd["wide24_csv"])
        frame, n_dropped = _drop_route_nan(frame)
        raw: dict[str, pd.DataFrame] = {}
        raw_paths: dict[str, Path] = {}
        for name in ("h48qual", "zig075"):
            path = _pred_path(name, wd["split"])
            raw_paths[name] = path
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            raw[name] = df
        windows[wname] = {
            "frame": frame, "raw": raw, "raw_paths": raw_paths, "oof": wd["oof"], "tier": wd["tier"],
            "split": wd["split"], "start": wd["start"], "end": wd["end"], "route_nan_dropped": n_dropped,
        }
    return windows


def verify_windows(windows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Reconfirms (does not assume) that each window's frame actually overlaps each component's raw
    predictions and that the overlap covers (nearly) the ENTIRE frame -- i.e. directly checks the
    "raw를 미리 자를 필요 없음" claim rather than taking it on faith.

    NOTE (discovered running this check, not assumed beforehand): train_predictions_q050/q075.csv
    (used by the 2025q1/2025q3 context windows) have a small number of rows genuinely absent versus
    the base+wide24 feature frame -- 66 rows scattered across 2025-01-01..01-23 for 2025q1, and one
    contiguous 30-row/2.5h gap on 2025-07-22 13:50..16:15 for 2025q3 (2025q2/val/oos_q1/oos_q2 all
    have 100% coverage). This is NOT a bug in this module's loading -- research_eth_omega461_exit_
    sweep_20260721.prep_component's own docstring already documents "predictions are the
    authoritative row set" (the original training pipeline's omega._align() trims rows for label
    warm-up/tail/NaN-feature reasons the raw feature frame doesn't reflect) and every downstream
    consumer (prep_component / align_frame_and_predictions above) already narrows the frame to the
    intersection by design, exactly as it does here. The check below therefore gates on a high
    coverage RATIO (>=99%, comfortably below the observed 99.75%/99.89%), not on exact 100% -- a
    materially lower ratio would indicate a real problem (wrong file, wrong date range, a genuine
    alignment bug) and should still abort.
    """
    diag: dict[str, Any] = {}
    for wname, w in windows.items():
        frame_ts = set(w["frame"]["timestamp"])
        row: dict[str, Any] = {
            "frame_rows": int(len(w["frame"])),
            "frame_range": [str(w["frame"]["timestamp"].min()), str(w["frame"]["timestamp"].max())],
            "route_nan_dropped": int(w["route_nan_dropped"]),
        }
        for name, raw_df in w["raw"].items():
            raw_ts = set(raw_df["timestamp"])
            inter = frame_ts & raw_ts
            ratio = (len(inter) / len(frame_ts)) if frame_ts else 0.0
            row[f"{name}_raw_rows"] = int(len(raw_df))
            row[f"{name}_intersection_rows"] = int(len(inter))
            row[f"{name}_intersection_ratio"] = float(ratio)
            row[f"{name}_intersection_covers_frame"] = bool(len(inter) == len(frame_ts) and len(frame_ts) > 0)
            row[f"{name}_intersection_high_coverage"] = bool(ratio >= 0.99)
        diag[wname] = row
    return diag


def align_frame_and_predictions(frame: pd.DataFrame, q_tags: dict[str, str], split: str, out_dir: Path) -> tuple[pd.DataFrame, dict[str, Path]]:
    """Portfolio-level alignment for ANY of the 6 windows. greedy.prepare_component /
    portfolio._prepare_component_val require `pred["timestamp"].equals(frame["timestamp"])` EXACTLY
    (no internal intersection, unlike sweep.prep_component) -- this pre-intersects frame timestamps
    against each component's prediction CSV and writes aligned copies both can consume.

    Generalizes research_eth_omega461_exit_head_portfolio_asymmetric_20260813._align_frame_and_
    predictions (hardcoded to "validation_predictions") and research_eth_omega461_queue_pressure_
    exit_threshold_20260814._align_frame_and_oos_predictions (hardcoded to "oos_predictions") into
    one function parameterized by `split` -- same intersect-then-reindex-then-fix-StringDtype logic,
    copied verbatim from those (neither is edited), only the filename prefix differs. Writes into
    `out_dir` (this module's own OUT_DIR by default in main() below) rather than any candidate
    script's OUT_DIR, so re-running this on oos_q2 never overwrites another experiment's already-
    published oos_q1-aligned CSVs.
    """
    fname_prefix = _SPLIT_FILE_PREFIX[split]
    raw_preds: dict[str, pd.DataFrame] = {}
    keep_ts = set(frame["timestamp"])
    for cname, q_tag in q_tags.items():
        pred_csv = sweep.EXT_PRED_DIR / cname / f"{fname_prefix}_{q_tag}.csv"
        df = pd.read_csv(pred_csv)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        raw_preds[cname] = df
        keep_ts &= set(df["timestamp"])
    aligned_frame = frame[frame["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
    aligned_paths: dict[str, Path] = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for cname, df in raw_preds.items():
        df = df[df["timestamp"].isin(keep_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(df) != len(aligned_frame) or not df["timestamp"].equals(aligned_frame["timestamp"]):
            raise RuntimeError(f"{cname}: alignment failed after timestamp intersection (split={split})")
        for c in df.columns:
            if str(df[c].dtype).lower().startswith("str"):
                df[c] = df[c].astype(object)
        out_path = out_dir / f"_aligned_{split}_{cname}_predictions.csv"
        df.to_csv(out_path, index=False)
        aligned_paths[cname] = out_path
    return aligned_frame, aligned_paths


def run_portfolio_variant(
    window_name: str, windows: dict[str, dict[str, Any]], comp_cfgs: dict[str, dict[str, Any]],
    *, fee: float, slip: float, device: torch.device, out_dir: Path, variant_label: str,
) -> dict[str, Any]:
    """Baseline-SHAPE-only portfolio replay: prepare components under `comp_cfgs` (plain h48qual/
    zig075 configs, e.g. baseline_both_original or asymmetric_tabm_liveatr -- no candidate-specific
    intervention hook of any kind) on `window_name`, run the UNMODIFIED greedy.greedy_replay, return
    no_gate + with_gate metrics. Reuses greedy.prepare_component / portfolio._prepare_component_val
    (chosen by the window's own oof flag, matching each function's own established convention) /
    greedy.greedy_replay / portfolio._ledger_metrics / mfe_width._duration_gated unchanged.

    This is NOT a per-candidate evaluator (see module docstring) -- it exists so (a) this module's
    own G0 self-check can reproduce the asymmetric_tabm_liveatr / baseline_both_original reference
    numbers on any window, and (b) every candidate's own comparator baseline (which is always this
    same asymmetric_tabm_liveatr config) does not need to be independently re-derived per call site.
    """
    w = windows[window_name]
    split = WINDOW_DEFS[window_name]["split"]
    q_tags = {name: sweep.COMPONENTS[name]["q_tag"] for name in comp_cfgs}
    aligned_frame, aligned_paths = align_frame_and_predictions(w["frame"], q_tags, split, out_dir)
    components: dict[str, Any] = {}
    for cname, cfg in comp_cfgs.items():
        if w["oof"]:
            components[cname] = portfolio._prepare_component_val(aligned_frame, aligned_paths[cname], cfg, device)
        else:
            components[cname] = greedy.prepare_component(aligned_frame, aligned_paths[cname], cfg, device)
    _diag, ledger = greedy.greedy_replay(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
    ledger_path = out_dir / f"portfolio_ledger_{window_name}_{variant_label}.csv"
    ledger.to_csv(ledger_path, index=False)
    no_gate = portfolio._ledger_metrics(ledger)
    with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
    return {"no_gate": no_gate, "with_gate": with_gate, "ledger_path": str(ledger_path), "aligned_frame": aligned_frame, "aligned_rows": int(len(aligned_frame))}


def summarize_multiwindow(
    baseline_results: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    candidate_results: dict[str, tuple[dict[str, Any], dict[str, Any]]],
    *, mdd_slack_pp: float = 0.0,
) -> dict[str, Any]:
    """Pre-registered table/verdict builder (see module docstring criterion). Each value is a
    (no_gate_metrics, with_gate_metrics) tuple -- computing those tuples per window/component-config
    is the CALLER's responsibility (each candidate's own intervention logic differs; see module
    docstring). Gate criterion: with_gate PnL non-worse AND with_gate MDD within `mdd_slack_pp`
    points of baseline, evaluated independently per window; final verdict requires ALL windows in
    OOS_CONFIRM_WINDOWS to pass simultaneously (single touch, not sequential/OR'd). context/val tier
    rows are always included in the table for reference but never enter the verdict.
    """
    rows: dict[str, Any] = {}
    for wname, wd in WINDOW_DEFS.items():
        if wname not in baseline_results or wname not in candidate_results:
            continue
        b_ng, b_wg = baseline_results[wname]
        c_ng, c_wg = candidate_results[wname]
        pnl_pass = float(c_wg["pnl"]) >= float(b_wg["pnl"])
        mdd_pass = (float(c_wg["mdd"]) - float(b_wg["mdd"])) >= -abs(mdd_slack_pp)
        rows[wname] = {
            "tier": wd["tier"],
            "baseline_no_gate": b_ng, "candidate_no_gate": c_ng,
            "baseline_with_gate": b_wg, "candidate_with_gate": c_wg,
            "with_gate_pnl_nonworse": bool(pnl_pass),
            "with_gate_mdd_nonworse_within_slack": bool(mdd_pass),
            "with_gate_pass": bool(pnl_pass and mdd_pass),
        }
    oos_rows = {k: v for k, v in rows.items() if v["tier"] == "oos_confirm"}
    oos_all_pass = bool(oos_rows) and all(v["with_gate_pass"] for v in oos_rows.values())
    return {
        "rows": rows,
        "mdd_slack_pp": float(mdd_slack_pp),
        "oos_confirm_windows_required": list(OOS_CONFIRM_WINDOWS),
        "oos_confirm_windows_present": list(oos_rows.keys()),
        "oos_confirm_per_window_pass": {k: v["with_gate_pass"] for k, v in oos_rows.items()},
        "oos_confirm_all_pass_single_touch": bool(oos_all_pass),
        "final_verdict": "CONFIRMED" if oos_all_pass else "REJECTED_SIGN_MISMATCH",
    }


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = G0_TOLERANCE_PP, check_trades: bool = True) -> bool:
    ok = bool(abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp)
    if check_trades and "trades" in expected:
        ok = ok and int(actual["trades"]) == int(expected["trades"])
    return ok


def _write_report(report: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)


# =====================================================================================================
# G0 reference values (all pre-existing/already-published, none derived by this module -- see
# docs/experiments/eth_omega461_multiwindow_confirmation_gate_20260814.md for exact provenance of
# each number).
# =====================================================================================================
COMP_CFGS_ASYMMETRIC_TABM_LIVEATR = {"h48qual": portfolio._component_cfg("h48qual", bundle_override=portfolio.NEW_H48QUAL_BUNDLE), "zig075": portfolio._component_cfg("zig075")}
COMP_CFGS_BASELINE_BOTH_ORIGINAL = {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")}

REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR = {
    # tmp/causal_regen_20260516/eth_omega461_risk_controlled_exit_fallback_20260814/report.json
    # g0.portfolio_asymmetric_tabm_liveatr / g0b.portfolio_tau_never_switch_with_gate_freshly_
    # established / oos_portfolio_baseline_{no,with}_gate -- reconfirmed by direct read before this
    # module was written.
    "val": ({"pnl": 46.59, "mdd": -21.70, "trades": 35}, {"pnl": 77.31, "mdd": -21.76, "trades": 26}),
    "oos_q1": ({"pnl": 93.27, "mdd": -15.48, "trades": 24}, {"pnl": 67.25, "mdd": -15.48, "trades": 19}),
}
REFERENCE_2025Q_BASELINE_BOTH_ORIGINAL_WITH_GATE = {
    # tmp/causal_regen_20260516/omega4_6_1_phase1_robustness_20260707/result.json
    # rolling_walk_forward_diagnostic (apply_gate=True) -- reconfirmed by direct read before this
    # module was written.
    "2025q1": {"pnl": 28.54, "mdd": -20.62, "trades": 19},
    "2025q2": {"pnl": 39.99, "mdd": -10.82, "trades": 15},
    "2025q3": {"pnl": -9.73, "mdd": -44.37, "trades": 19},
}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()
    report: dict[str, Any] = {
        "design": (
            "Reusable multi-window confirmation gate infra: load_all_windows/verify_windows/"
            "align_frame_and_predictions/run_portfolio_variant/summarize_multiwindow. main() runs "
            "(1) G0 self-check against known reference values, then (2) a retroactive stress test "
            "applying this module to the two already-decided candidates (queue-pressure, "
            "risk-controlled) across all 6 windows -- NOT a re-judgment (both were already "
            "decisively rejected on oos_q1 alone), a stress test of this module's own correctness."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window_defs": {k: {kk: (str(vv) if isinstance(vv, Path) else vv) for kk, vv in v.items()} for k, v in WINDOW_DEFS.items()},
    }

    # =================================================================================================
    # stage=load_and_verify_windows
    # =================================================================================================
    print("=== stage=load_and_verify_windows ===", flush=True)
    windows = load_all_windows()
    verify_diag = verify_windows(windows)
    for wname, row in verify_diag.items():
        print(f"  {wname}: frame_rows={row['frame_rows']} range={row['frame_range']} route_nan_dropped={row['route_nan_dropped']} "
              f"h48qual_cover={row['h48qual_intersection_ratio']:.4f} zig075_cover={row['zig075_intersection_ratio']:.4f}", flush=True)
    # Gate on high coverage (>=99%), not exact 100% -- 2025q1/2025q3 were found (this run) to have a
    # small number of rows genuinely absent from train_predictions_*.csv (66 rows scattered across
    # 2025-01-01..01-23 for 2025q1, one contiguous 30-row/2.5h gap on 2025-07-22 for 2025q3; 2025q2/
    # val/oos_q1/oos_q2 all have exactly 100%), which align_frame_and_predictions/prep_component
    # already handle correctly by design (intersection narrowing) -- see verify_windows docstring.
    high_coverage = all(row[f"{name}_intersection_high_coverage"] for row in verify_diag.values() for name in ("h48qual", "zig075"))
    context_val_route_nan_zero = all(verify_diag[w]["route_nan_dropped"] == 0 for w in CONTEXT_WINDOWS + (VAL_WINDOW,))
    window_verification_pass = bool(high_coverage and context_val_route_nan_zero)
    report["window_verification"] = verify_diag
    report["window_verification_pass"] = window_verification_pass
    print(f"stage=load_and_verify_windows_result pass={window_verification_pass}", flush=True)
    if not window_verification_pass:
        report["stage_reached"] = "load_and_verify_windows"
        report["gate_pass"] = False
        report["note"] = "Window verification failed -- a raw prediction file covers <99% of its frame, or an unexpected route-NaN gap appeared on a 2025/VAL window. Aborting before trusting any G0 number."
        _write_report(report)
        print("stage=ABORT window_verification failed", flush=True)
        return 1

    # =================================================================================================
    # stage=G0a -- asymmetric_tabm_liveatr on val + oos_q1 via this module's own window loader +
    # run_portfolio_variant, compared against already-published reference numbers.
    # =================================================================================================
    print("=== stage=G0a_asymmetric_tabm_liveatr_val_oosq1 ===", flush=True)
    g0a: dict[str, Any] = {}
    for wname in ("val", "oos_q1"):
        result = run_portfolio_variant(wname, windows, COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        ref_ng, ref_wg = REFERENCE_VAL_OOSQ1_ASYMMETRIC_TABM_LIVEATR[wname]
        ok_ng, ok_wg = _close(result["no_gate"], ref_ng), _close(result["with_gate"], ref_wg)
        g0a[wname] = {"no_gate": {"actual": result["no_gate"], "reference": ref_ng, "match": ok_ng},
                      "with_gate": {"actual": result["with_gate"], "reference": ref_wg, "match": ok_wg}}
        print(f"  {wname}: no_gate={result['no_gate']['pnl']:.2f}%/{result['no_gate']['mdd']:.2f}%/{result['no_gate']['trades']} match={ok_ng}  "
              f"with_gate={result['with_gate']['pnl']:.2f}%/{result['with_gate']['mdd']:.2f}%/{result['with_gate']['trades']} match={ok_wg}", flush=True)
    g0a_pass = all(g0a[w]["no_gate"]["match"] and g0a[w]["with_gate"]["match"] for w in ("val", "oos_q1"))
    report["g0a_val_oosq1"] = {"windows": g0a, "pass": g0a_pass}
    print(f"stage=G0a_result pass={g0a_pass}", flush=True)

    # =================================================================================================
    # stage=G0b -- 2025 Q1/Q2/Q3: (i) baseline_both_original via TWO independent code paths (this
    # module's own run_portfolio_variant, AND a direct call to audit_omega4_6_1_phase1_robustness_
    # 20260707.load_2025_quarter_components + the legacy test_omega4_6_1_drop_h48qual_20260706._
    # metrics(apply_gate=True) -- the exact code that produced the published reference), both checked
    # against the published reference; (ii) asymmetric_tabm_liveatr (new numbers, no prior reference,
    # reliability argued from (i)'s successful reproduction of the loading mechanism).
    # =================================================================================================
    print("=== stage=G0b_2025_quarters ===", flush=True)
    g0b_both_original: dict[str, Any] = {}
    g0b_both_original_legacy: dict[str, Any] = {}
    g0b_tabm_liveatr: dict[str, Any] = {}
    for wname in CONTEXT_WINDOWS:
        ref = REFERENCE_2025Q_BASELINE_BOTH_ORIGINAL_WITH_GATE[wname]

        result_new = run_portfolio_variant(wname, windows, COMP_CFGS_BASELINE_BOTH_ORIGINAL, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="baseline_both_original")
        ok_new = _close(result_new["with_gate"], ref)
        g0b_both_original[wname] = {"actual_with_gate": result_new["with_gate"], "actual_no_gate": result_new["no_gate"], "reference": ref, "match": ok_new}

        wd = WINDOW_DEFS[wname]
        legacy_frame, legacy_components = audit.load_2025_quarter_components(wd["start"], wd["end"])
        greedy.PRIORITY = ("h48qual", "zig075")
        _diag, legacy_ledger = greedy.greedy_replay(legacy_frame, legacy_components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device)
        legacy_m = legacy_duration_gate_metrics(legacy_ledger, legacy_frame, apply_gate=True)
        ok_legacy = _close(legacy_m, ref)
        g0b_both_original_legacy[wname] = {"actual": legacy_m, "reference": ref, "match": ok_legacy}
        print(f"  {wname} baseline_both_original: this_module={result_new['with_gate']['pnl']:.2f}%/{result_new['with_gate']['mdd']:.2f}%/{result_new['with_gate']['trades']} match={ok_new}  "
              f"independent_legacy_path={legacy_m['pnl']:.2f}%/{legacy_m['mdd']:.2f}%/{legacy_m['trades']} match={ok_legacy}", flush=True)

        result_tabm = run_portfolio_variant(wname, windows, COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")
        g0b_tabm_liveatr[wname] = {"no_gate": result_tabm["no_gate"], "with_gate": result_tabm["with_gate"], "reference": None, "note": "new number, no prior published reference -- reliability argued from this window's baseline_both_original reproduction above (same loading mechanism, only the h48qual bundle differs)"}
        print(f"  {wname} asymmetric_tabm_liveatr (NEW, no prior reference): no_gate={result_tabm['no_gate']['pnl']:.2f}%/{result_tabm['no_gate']['mdd']:.2f}%/{result_tabm['no_gate']['trades']} "
              f"with_gate={result_tabm['with_gate']['pnl']:.2f}%/{result_tabm['with_gate']['mdd']:.2f}%/{result_tabm['with_gate']['trades']}", flush=True)

    g0b_pass = all(g0b_both_original[w]["match"] and g0b_both_original_legacy[w]["match"] for w in CONTEXT_WINDOWS)
    report["g0b_2025_quarters"] = {
        "baseline_both_original_this_module": g0b_both_original,
        "baseline_both_original_independent_legacy_path": g0b_both_original_legacy,
        "asymmetric_tabm_liveatr_new_numbers": g0b_tabm_liveatr,
        "pass": g0b_pass,
    }
    print(f"stage=G0b_result pass={g0b_pass}", flush=True)

    # =================================================================================================
    # stage=G0c -- mathematical equivalence of test_omega4_6_1_drop_h48qual_20260706._metrics(
    # apply_gate=True) and research_eth_omega461_live_sltp_mfe_width_20260813._duration_gated on one
    # concrete ledger (the VAL asymmetric_tabm_liveatr ledger just produced in G0a).
    # =================================================================================================
    print("=== stage=G0c_metrics_duration_gated_equivalence ===", flush=True)
    val_ledger_path = OUT_DIR / "portfolio_ledger_val_asymmetric_tabm_liveatr.csv"
    val_ledger = pd.read_csv(val_ledger_path)
    val_aligned_frame = run_portfolio_variant("val", windows, COMP_CFGS_ASYMMETRIC_TABM_LIVEATR, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label="asymmetric_tabm_liveatr")["aligned_frame"]
    legacy_eq = legacy_duration_gate_metrics(val_ledger, val_aligned_frame, apply_gate=True)
    new_eq = mfe_width._duration_gated(val_ledger, val_aligned_frame, greedy.DURATION_THRESHOLD)
    equivalence_ok = bool(abs(legacy_eq["pnl"] - new_eq["pnl"]) < 1.0e-6 and abs(legacy_eq["mdd"] - new_eq["mdd"]) < 1.0e-6 and int(legacy_eq["trades"]) == int(new_eq["trades"]))
    report["g0c_metrics_duration_gated_equivalence"] = {"legacy__metrics_apply_gate_true": legacy_eq, "new__duration_gated": new_eq, "equivalent": equivalence_ok, "ledger_used": str(val_ledger_path), "n_trades_in_ledger": int(len(val_ledger))}
    print(f"  legacy={legacy_eq} new={new_eq} equivalent={equivalence_ok}", flush=True)

    g0_pass = bool(g0a_pass and g0b_pass and equivalence_ok)
    report["gate_pass_g0"] = g0_pass
    print(f"stage=G0_overall_result pass={g0_pass}", flush=True)
    if not g0_pass:
        report["stage_reached"] = "G0"
        report["gate_pass"] = False
        report["note"] = "G0 self-check failed -- this module does not faithfully reproduce already-published reference numbers. Aborting before stage 4 (retroactive stress test), per task instruction."
        _write_report(report)
        print("stage=ABORT G0 failed", flush=True)
        return 1

    # =================================================================================================
    # stage=step4 -- retroactive stress test: apply queue-pressure (threshold=0.80) and
    # risk-controlled (eps_frac=0.90, tau_hat=0.9995394945144653, FROZEN, no recalibration) to all 6
    # windows. Both candidates were ALREADY decisively rejected on oos_q1 alone (before this module
    # existed) -- this stress-tests whether this module's plumbing produces correct, internally
    # consistent numbers when driving real candidate logic, not a re-judgment.
    # =================================================================================================
    print("=== stage=step4_retroactive_stress_test ===", flush=True)
    with open(rc_mod.GBDT_BUNDLE, "rb") as f:
        gbdt_bundle = pickle.load(f)
    gbdt_models = gbdt_bundle["models"]
    base_cols = list(torch.load(COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"]["bundle"], map_location="cpu", weights_only=False)["base_cols"])

    def _prep_asymmetric_components(window_name: str) -> tuple[pd.DataFrame, dict[str, Path], dict[str, Any], dict[str, Any]]:
        w = windows[window_name]
        split = WINDOW_DEFS[window_name]["split"]
        q_tags = {"h48qual": sweep.COMPONENTS["h48qual"]["q_tag"], "zig075": sweep.COMPONENTS["zig075"]["q_tag"]}
        aligned_frame, aligned_paths = align_frame_and_predictions(w["frame"], q_tags, split, OUT_DIR)
        prep = portfolio._prepare_component_val if w["oof"] else greedy.prepare_component
        h48qual_prepped = prep(aligned_frame, aligned_paths["h48qual"], COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["h48qual"], device)
        zig075_prepped = prep(aligned_frame, aligned_paths["zig075"], COMP_CFGS_ASYMMETRIC_TABM_LIVEATR["zig075"], device)
        return aligned_frame, aligned_paths, h48qual_prepped, zig075_prepped

    def _queue_pressure_variant(window_name: str) -> dict[str, Any]:
        w = windows[window_name]
        aligned_frame, aligned_paths, h48qual_prepped, zig075_prepped = _prep_asymmetric_components(window_name)
        pressure_mask, mismatches = qp_mod._zig075_pressure_mask(aligned_paths["zig075"], zig075_prepped["dec"], oof=w["oof"], quality_threshold=sweep.COMPONENTS["zig075"]["quality_threshold"])
        if mismatches:
            raise RuntimeError(f"{window_name}: queue-pressure mask cross-check failed, {mismatches} mismatches")
        h48qual_qp = dict(h48qual_prepped)
        h48qual_qp["queue_pressure_mask"] = pressure_mask
        components = {"h48qual": h48qual_qp, "zig075": zig075_prepped}
        _diag_b, ledger_b = qp_mod.greedy_replay_queue_pressure(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, queue_pressure_component="h48qual", queue_pressure_threshold=0.95)
        _diag_c, ledger_c = qp_mod.greedy_replay_queue_pressure(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, queue_pressure_component="h48qual", queue_pressure_threshold=0.80)
        ledger_b.to_csv(OUT_DIR / f"portfolio_ledger_{window_name}_qp_baseline_degenerate095.csv", index=False)
        ledger_c.to_csv(OUT_DIR / f"portfolio_ledger_{window_name}_qp_candidate_thr080.csv", index=False)
        baseline = (portfolio._ledger_metrics(ledger_b), mfe_width._duration_gated(ledger_b, aligned_frame, greedy.DURATION_THRESHOLD))
        candidate = (portfolio._ledger_metrics(ledger_c), mfe_width._duration_gated(ledger_c, aligned_frame, greedy.DURATION_THRESHOLD))
        return {"baseline": baseline, "candidate": candidate, "pressure_bars": int(pressure_mask.sum()), "mismatches": int(mismatches)}

    def _risk_controlled_variant(window_name: str) -> dict[str, Any]:
        aligned_frame, aligned_paths, h48qual_prepped, zig075_prepped = _prep_asymmetric_components(window_name)
        h48qual_rc = rc_mod._gbdt_portfolio_fallback(dict(h48qual_prepped), base_cols, gbdt_models, device)
        components = {"h48qual": h48qual_rc, "zig075": zig075_prepped}
        _diag_b, ledger_b = rc_mod.greedy_replay_risk_controlled(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, risk_component="h48qual", tau=rc_mod.TAU_NEVER_SWITCH)
        _diag_c, ledger_c = rc_mod.greedy_replay_risk_controlled(aligned_frame, components, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device, risk_component="h48qual", tau=0.9995394945144653)
        ledger_b.to_csv(OUT_DIR / f"portfolio_ledger_{window_name}_rc_baseline_never_switch.csv", index=False)
        ledger_c.to_csv(OUT_DIR / f"portfolio_ledger_{window_name}_rc_candidate_eps090.csv", index=False)
        baseline = (portfolio._ledger_metrics(ledger_b), mfe_width._duration_gated(ledger_b, aligned_frame, greedy.DURATION_THRESHOLD))
        candidate = (portfolio._ledger_metrics(ledger_c), mfe_width._duration_gated(ledger_c, aligned_frame, greedy.DURATION_THRESHOLD))
        return {"baseline": baseline, "candidate": candidate, "switch_bars": int(_diag_c["rc_switch_bars"])}

    print("--- queue-pressure (threshold=0.80) across all 6 windows ---", flush=True)
    qp_by_window: dict[str, Any] = {}
    for wname in ALL_WINDOWS:
        qp_by_window[wname] = _queue_pressure_variant(wname)
        b_ng, b_wg = qp_by_window[wname]["baseline"]
        c_ng, c_wg = qp_by_window[wname]["candidate"]
        print(f"  {wname}: baseline no_gate={b_ng['pnl']:.2f}%/{b_ng['mdd']:.2f}%/{b_ng['trades']} with_gate={b_wg['pnl']:.2f}%/{b_wg['mdd']:.2f}%/{b_wg['trades']}  |  "
              f"candidate no_gate={c_ng['pnl']:.2f}%/{c_ng['mdd']:.2f}%/{c_ng['trades']} with_gate={c_wg['pnl']:.2f}%/{c_wg['mdd']:.2f}%/{c_wg['trades']}", flush=True)
    qp_val_baseline_check = _close(qp_by_window["val"]["baseline"][0], {"pnl": 46.59, "mdd": -21.70, "trades": 35})
    qp_oosq1_baseline_check = _close(qp_by_window["oos_q1"]["baseline"][0], {"pnl": 93.27, "mdd": -15.48, "trades": 24})
    qp_oosq1_candidate_check = _close(qp_by_window["oos_q1"]["candidate"][0], {"pnl": 59.08, "mdd": -15.48, "trades": 27})
    print(f"  cross-check vs already-published queue-pressure numbers: val_baseline={qp_val_baseline_check} oos_q1_baseline={qp_oosq1_baseline_check} oos_q1_candidate_no_gate={qp_oosq1_candidate_check}", flush=True)

    print("--- risk-controlled (eps_frac=0.90, tau_hat FROZEN=0.9995394945144653) across all 6 windows ---", flush=True)
    rc_by_window: dict[str, Any] = {}
    for wname in ALL_WINDOWS:
        rc_by_window[wname] = _risk_controlled_variant(wname)
        b_ng, b_wg = rc_by_window[wname]["baseline"]
        c_ng, c_wg = rc_by_window[wname]["candidate"]
        print(f"  {wname}: baseline no_gate={b_ng['pnl']:.2f}%/{b_ng['mdd']:.2f}%/{b_ng['trades']} with_gate={b_wg['pnl']:.2f}%/{b_wg['mdd']:.2f}%/{b_wg['trades']}  |  "
              f"candidate no_gate={c_ng['pnl']:.2f}%/{c_ng['mdd']:.2f}%/{c_ng['trades']} with_gate={c_wg['pnl']:.2f}%/{c_wg['mdd']:.2f}%/{c_wg['trades']} switch_bars={rc_by_window[wname]['switch_bars']}", flush=True)
    rc_val_baseline_check = _close(rc_by_window["val"]["baseline"][0], {"pnl": 46.59, "mdd": -21.70, "trades": 35})
    rc_oosq1_baseline_check = _close(rc_by_window["oos_q1"]["baseline"][0], {"pnl": 93.27, "mdd": -15.48, "trades": 24})
    rc_oosq1_candidate_check = _close(rc_by_window["oos_q1"]["candidate"][0], {"pnl": 21.18, "mdd": -28.70, "trades": 25})
    print(f"  cross-check vs already-published risk-controlled numbers: val_baseline={rc_val_baseline_check} oos_q1_baseline={rc_oosq1_baseline_check} oos_q1_candidate_no_gate={rc_oosq1_candidate_check}", flush=True)

    # ---- summarize_multiwindow verdicts (mdd_slack_pp=0.0 strict, and 3.0 relaxed-gate variant) ----
    qp_baseline_tuples = {w: qp_by_window[w]["baseline"] for w in ALL_WINDOWS}
    qp_candidate_tuples = {w: qp_by_window[w]["candidate"] for w in ALL_WINDOWS}
    rc_baseline_tuples = {w: rc_by_window[w]["baseline"] for w in ALL_WINDOWS}
    rc_candidate_tuples = {w: rc_by_window[w]["candidate"] for w in ALL_WINDOWS}
    qp_summary_strict = summarize_multiwindow(qp_baseline_tuples, qp_candidate_tuples, mdd_slack_pp=0.0)
    qp_summary_relaxed = summarize_multiwindow(qp_baseline_tuples, qp_candidate_tuples, mdd_slack_pp=3.0)
    rc_summary_strict = summarize_multiwindow(rc_baseline_tuples, rc_candidate_tuples, mdd_slack_pp=0.0)
    rc_summary_relaxed = summarize_multiwindow(rc_baseline_tuples, rc_candidate_tuples, mdd_slack_pp=3.0)
    print(f"queue_pressure verdict: strict={qp_summary_strict['final_verdict']} relaxed_mdd3pp={qp_summary_relaxed['final_verdict']}", flush=True)
    print(f"risk_controlled verdict: strict={rc_summary_strict['final_verdict']} relaxed_mdd3pp={rc_summary_relaxed['final_verdict']}", flush=True)

    already_decided_rejected = True  # both queue-pressure(0.80) and risk-controlled(eps=0.90) were ALREADY decisively rejected on oos_q1 alone, before this module existed (see docs/model_contracts/odyssey2_eth_live_injection_contract_20260813.md #7/#8)
    verdict_changed = bool((qp_summary_strict["final_verdict"] == "CONFIRMED") or (rc_summary_strict["final_verdict"] == "CONFIRMED"))

    report["step4_retroactive_stress_test"] = {
        "queue_pressure_threshold080": {
            "by_window": qp_by_window,
            "cross_checks_vs_already_published": {"val_baseline_no_gate": qp_val_baseline_check, "oos_q1_baseline_no_gate": qp_oosq1_baseline_check, "oos_q1_candidate_no_gate": qp_oosq1_candidate_check},
            "summary_strict_mdd0pp": qp_summary_strict, "summary_relaxed_mdd3pp": qp_summary_relaxed,
        },
        "risk_controlled_eps090": {
            "by_window": rc_by_window,
            "cross_checks_vs_already_published": {"val_baseline_no_gate": rc_val_baseline_check, "oos_q1_baseline_no_gate": rc_oosq1_baseline_check, "oos_q1_candidate_no_gate": rc_oosq1_candidate_check},
            "summary_strict_mdd0pp": rc_summary_strict, "summary_relaxed_mdd3pp": rc_summary_relaxed,
        },
        "prior_verdicts_were_rejected": already_decided_rejected,
        "this_module_changes_prior_verdict": verdict_changed,
        "note": "Both candidates were already decisively rejected (oos_q1 alone reversed) before this module existed -- this stage is a plumbing stress test, not a re-judgment. verdict_changed=false is the expected/required outcome.",
    }

    report["stage_reached"] = "step4_retroactive_stress_test"
    report["gate_pass"] = True
    _write_report(report)
    print(f"stage=done gate_pass=True verdict_changed={verdict_changed}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
