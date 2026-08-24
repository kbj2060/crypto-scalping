#!/usr/bin/env python3
"""RESEARCH ONLY -- Ilias correction session (2026-08-17): re-run of the pre-registered success/kill
evaluation using the SIDE-BLIND classifier (research_ilias_eth_adaptive_exit_signal_train_sideblind_
20260817.py, feature_columns = research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.
FEATURE_COLUMNS, pos_side/pos_leverage/pos_notional/pos_exposure excluded -- see that module's docstring
for the quasi-separation finding this corrects).

Documented copy of research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py (that original script
and its outputs, arm_eval_report.json etc., are left UNTOUCHED -- preserved as the research record of
the contaminated run). Changes vs the original, ALL mechanical (loading a different bundle file /
using the side-blind replay function / writing to different output filenames) -- the success/kill
criteria, windows, seeds, guardrail, and interpretation decision below are copied VERBATIM, unchanged,
per the correction task's explicit instruction not to alter the pre-registered bar:

=== Success/kill criteria (verbatim from the design doc, NOT re-derived or adjusted here) ===
1. The new signal's firing rate OR precision (SL-outcome prediction accuracy) differs significantly
   between always_long-type (wrong direction) and always_short-type (right direction) arms -- N=30
   independent seeds, t = gap / (std / sqrt(30)), |t| > 2.
2. For windows passing (1): a fresh-forward replay with the new signal substituted for h48qual's
   exit_head reduces MDD on the always_long-type arm vs that arm's own G0 (real-exit_head) baseline,
   while not significantly hurting always_short-type PnL (50%-relative-degradation / sign-flip
   guardrail, reused from research_eth_omega461_gittins_index_exit_head_20260814._guardrail_pass's
   convention -- see `_guardrail_pass` below, generalized here to a possibly-negative baseline since
   that script's own baseline was always positive).
Kill: if (1) fails, the axis is closed -- no post-hoc metric/threshold changes, no N increase chasing
significance (pre-registered, per the design doc's own kill-condition text).

=== Interpretation decision (verbatim, unchanged) ===
gap = firing_rate(always_long) - firing_rate(always_short) (or precision, same formula) -- a direct
point-estimate difference between the two structurally-fixed-direction arms; std = the standard
deviation of that SAME metric across the N=30 random-direction arms in the SAME window, used as the
noise-floor estimate. This interpretation was fixed BEFORE running anything in the ORIGINAL arm-eval
script -- reused here unchanged, not re-derived for this correction.

=== Method (only the classifier/scoring path is side-blind; everything else is identical) ===
"Firing rate"/"precision" are measured at TRADE level, via
research_ilias_eth_adaptive_exit_signal_common_20260817.simulate_private_barrier_trades (imported
UNCHANGED, re-exported by the sideblind common module -- label/feature construction does not depend on
which feature subset a downstream classifier uses). Criterion 1's `score_arm_trades` below selects
`bundle["feature_columns"]` (the side-blind 10-column list, read from the bundle itself rather than
hardcoded) instead of the original's fixed `common.FEATURE_COLUMNS` (14 columns). Criterion 2's replay
uses research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817.
greedy_replay_new_exit_signal_sideblind (documented copy of common.greedy_replay_new_exit_signal, ONE
inserted block: name->value dict feature selection instead of the original's fixed-order pos_values
list -- necessary because the side-blind bundle's feature_columns is a strict subset in a different
position than the original's assumed 14-column layout).

fresh_forward_bar_by_bar=true for every replay in this script. The counterfactual barrier simulation
used for criterion-1 ground truth is offline label-construction/measurement, not a live decision.
trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false.
future_rows_used_for_entry=false.

Does NOT touch trading_bot.py / trading_bot_modules/* / runtime_config.py / .env. Does NOT modify any
imported module (including the original, non-sideblind ilias scripts -- read-only). No retraining
inside this script, no GPU (DEVICE=cpu), conda env quant_ai.
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
import replay_omega4_6_1_greedy_router_20260706 as greedy  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import research_eth_omega461_live_sltp_mfe_width_20260813 as mfe_width  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402
import research_eth_omega461_regime_aware_exit_head_uptrend_guard_20260814 as guard  # noqa: E402
import research_eth_odyssey4_random_direction_risk_management_ablation_20260817 as abl  # noqa: E402
import research_eth_odyssey4_random_direction_large_n_reverification_20260817 as abl_large  # noqa: E402
import research_ilias_eth_adaptive_exit_signal_common_sideblind_20260817 as common_sb  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/ilias_eth_adaptive_exit_signal_baseline_20260817"
DEVICE = portfolio.DEVICE
N_SEEDS = 30
JUDGED_DOWNTREND_WINDOWS = ("val", "oos_q1", "oos_q2")
RANGING_WINDOW_KEYS = [c["key"] for c in abl_large.RANGING_CANDIDATES]
ALL_EVAL_WINDOWS = list(JUDGED_DOWNTREND_WINDOWS) + RANGING_WINDOW_KEYS
T_STAT_SIGNIFICANT = 2.0
GUARDRAIL_MAX_RELATIVE_DEGRADATION = 0.50  # reused convention, see module docstring


def log(msg: str) -> None:
    common_sb.log(msg)


def _guardrail_pass(baseline_pnl: float, candidate_pnl: float) -> bool:
    """Verbatim copy of the original arm-eval script's _guardrail_pass -- see that script's docstring
    for the full derivation. Unchanged here."""
    if baseline_pnl == 0.0:
        return bool(candidate_pnl >= 0.0)
    if baseline_pnl > 0.0 and candidate_pnl <= 0.0:
        return False
    relative_degradation = (baseline_pnl - candidate_pnl) / abs(baseline_pnl)
    return bool(relative_degradation <= GUARDRAIL_MAX_RELATIVE_DEGRADATION)


def load_windows() -> dict[str, Any]:
    windows = dict(gate.load_all_windows())
    for cand in abl_large.RANGING_CANDIDATES:
        windows[cand["key"]] = abl_large.load_custom_window(cand)
    return windows


def score_arm_trades(feat_df: pd.DataFrame, bundle: dict[str, Any]) -> dict[str, Any]:
    if feat_df.empty:
        return {"n_trades": 0, "n_fired_trades": 0, "firing_rate": float("nan"), "precision": float("nan"), "label_positive_rate": float("nan")}
    x = feat_df[bundle["feature_columns"]].to_numpy(dtype=np.float64)  # side-blind subset, read from bundle
    proba = bundle["model"].predict_proba(x)[:, 1]
    scored = feat_df.assign(proba=proba, fired=proba >= float(bundle["threshold"]))
    trade_level = scored.groupby("trade_id").agg(would_fire=("fired", "any"), label_sl=("label_sl", "first"))
    n_trades = int(len(trade_level))
    n_fired = int(trade_level["would_fire"].sum())
    firing_rate = float(trade_level["would_fire"].mean()) if n_trades else float("nan")
    precision = float(trade_level.loc[trade_level["would_fire"], "label_sl"].mean()) if n_fired else float("nan")
    return {
        "n_trades": n_trades, "n_fired_trades": n_fired, "firing_rate": firing_rate, "precision": precision,
        "label_positive_rate": float(trade_level["label_sl"].mean()) if n_trades else float("nan"),
    }


def gap_std_t(window_rows: pd.DataFrame, metric: str) -> dict[str, Any]:
    al = float(window_rows.loc[window_rows.arm == "always_long", metric].iloc[0])
    as_ = float(window_rows.loc[window_rows.arm == "always_short", metric].iloc[0])
    rnd = window_rows.loc[window_rows.arm.str.startswith("random_seed"), metric].dropna().to_numpy(dtype=np.float64)
    gap = al - as_
    n = int(len(rnd))
    std = float(np.std(rnd, ddof=1)) if n > 1 else float("nan")
    t = float(gap / (std / np.sqrt(n))) if (n > 1 and std > 0) else float("nan")
    return {
        "always_long": al, "always_short": as_, "gap": gap,
        "random_n_effective": n, "random_mean": float(np.mean(rnd)) if n else float("nan"),
        "random_std": std, "t_stat": t, "significant": bool(n > 1 and std > 0 and abs(t) > T_STAT_SIGNIFICANT),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = DEVICE
    fee, slip = omega._load_fee_slip()

    log("=== stage=load_frozen_new_exit_signal_bundle (SIDE-BLIND) ===")
    with open(OUT_DIR / "new_exit_signal_bundle_sideblind.pkl", "rb") as f:
        bundle = pickle.load(f)
    log(f"  model={bundle['model_name']} threshold={bundle['threshold']} n_train_trades={bundle['n_train_trades']} "
        f"feature_columns={bundle['feature_columns']}")

    log("=== stage=load_windows (3 downtrend judged + 3 ranging, reused from prior ablation) ===")
    windows = load_windows()

    log("=== stage=detector_build (reused, zero new free parameters) ===")
    score_by_base, _robustness, threshold = guard.build_detector()

    seed_sequence = np.random.SeedSequence(20260817)
    seeds = [int(s) for s in seed_sequence.generate_state(N_SEEDS)]
    log(f"  N_SEEDS={N_SEEDS} independently-spawned seeds (identical to original arm-eval, same SeedSequence(20260817)): {seeds}")

    arm_specs: list[tuple[str, Any]] = [
        ("always_long", lambda n: abl._side_selector_constant(n, 1)),
        ("always_short", lambda n: abl._side_selector_constant(n, -1)),
    ]
    arm_specs += [(f"random_seed{seed}", (lambda n, _s=seed: abl._side_selector_random(n, _s))) for seed in seeds]

    log("=== stage=criterion1_firing_rate_precision_by_arm (all 6 windows x 32 arms) ===")
    all_rows: list[dict[str, Any]] = []
    for window_name in ALL_EVAL_WINDOWS:
        log(f"  window={window_name}")
        for arm_label, selector in arm_specs:
            aligned_frame, components = abl.build_ablation_components(
                window_name, windows, score_by_base, threshold, OUT_DIR, device, side_selector=selector,
            )
            feat_df, sim_diag = common_sb.simulate_private_barrier_trades(
                aligned_frame, components["h48qual"], fee=fee, slip=slip, cost_mult=sweep.COST_MULT,
            )
            metrics = score_arm_trades(feat_df, bundle)
            all_rows.append({"window": window_name, "arm": arm_label, **metrics,
                              "n_trades_total_sim": sim_diag.get("n_trades_total"),
                              "n_trades_truncated": sim_diag.get("n_trades_truncated_open_at_frame_end")})
        log(f"    done ({len(arm_specs)} arms)")

    results_df = pd.DataFrame(all_rows)
    results_csv = OUT_DIR / "arm_eval_criterion1_by_arm_sideblind.csv"
    results_df.to_csv(results_csv, index=False)
    log(f"wrote {results_csv}")

    log("=== stage=criterion1_verdict_per_window ===")
    criterion1: dict[str, Any] = {}
    for window_name in ALL_EVAL_WINDOWS:
        wdf = results_df[results_df.window == window_name]
        firing = gap_std_t(wdf, "firing_rate")
        precision = gap_std_t(wdf, "precision")
        passed = bool(firing["significant"] or precision["significant"])
        criterion1[window_name] = {"firing_rate": firing, "precision": precision, "criterion1_pass": passed}
        log(f"  {window_name}: firing_rate t={firing['t_stat']:.3f} (AL={firing['always_long']:.4f} AS={firing['always_short']:.4f} "
            f"random={firing['random_mean']:.4f}+-{firing['random_std']:.4f} n={firing['random_n_effective']})  "
            f"precision t={precision['t_stat']:.3f} (AL={precision['always_long']:.4f} AS={precision['always_short']:.4f} "
            f"random={precision['random_mean']:.4f}+-{precision['random_std']:.4f} n={precision['random_n_effective']})  "
            f"PASS={passed}")

    passing_windows = [w for w, v in criterion1.items() if v["criterion1_pass"]]
    log(f"criterion1 passing windows: {passing_windows}")

    log("=== stage=criterion2_replay (only for criterion1-passing windows, always_long/always_short only) ===")
    criterion2: dict[str, Any] = {}
    for window_name in passing_windows:
        log(f"  window={window_name}")
        base_long = abl.run_arm("always_long", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                                 side_selector=lambda n: abl._side_selector_constant(n, 1))["with_gate"]
        base_short = abl.run_arm("always_short", window_name, windows, score_by_base, threshold, OUT_DIR, device, fee, slip,
                                  side_selector=lambda n: abl._side_selector_constant(n, -1))["with_gate"]
        if not (base_short["pnl"] >= base_long["pnl"]):
            log(f"    WARNING: always_short-type baseline PnL ({base_short['pnl']:.2f}%) is NOT >= "
                f"always_long-type baseline PnL ({base_long['pnl']:.2f}%) in this window -- interpreting "
                f"criterion 2 literally by arm name anyway, flagged for the report.")

        def _new_signal_arm(side_val: int) -> dict[str, Any]:
            aligned_frame, components = abl.build_ablation_components(
                window_name, windows, score_by_base, threshold, OUT_DIR, device,
                side_selector=lambda n, _s=side_val: abl._side_selector_constant(n, _s),
            )
            h48_new = dict(components["h48qual"])
            h48_new["new_exit_model"] = bundle
            h48_new["new_exit_threshold"] = float(bundle["threshold"])
            components_new = dict(components)
            components_new["h48qual"] = h48_new
            diag, ledger = common_sb.greedy_replay_new_exit_signal_sideblind(
                aligned_frame, components_new, fee=fee, slip=slip, cost_mult=sweep.COST_MULT, device=device,
            )
            with_gate = mfe_width._duration_gated(ledger, aligned_frame, greedy.DURATION_THRESHOLD)
            no_gate = portfolio._ledger_metrics(ledger)
            return {"no_gate": no_gate, "with_gate": with_gate,
                    "reason_counts": diag.get("reason_counts"), "veto_bars": diag.get("veto_bars")}

        new_long = _new_signal_arm(1)
        new_short = _new_signal_arm(-1)

        mdd_improves_always_long = bool(new_long["with_gate"]["mdd"] >= base_long["mdd"])
        guardrail_always_short = _guardrail_pass(base_short["pnl"], new_short["with_gate"]["pnl"])
        criterion2_pass = bool(mdd_improves_always_long and guardrail_always_short)

        criterion2[window_name] = {
            "baseline_always_long_with_gate": base_long, "baseline_always_short_with_gate": base_short,
            "new_signal_always_long": new_long, "new_signal_always_short": new_short,
            "mdd_improves_always_long": mdd_improves_always_long,
            "guardrail_pass_always_short_pnl": guardrail_always_short,
            "criterion2_pass": criterion2_pass,
        }
        log(f"    baseline: AL pnl={base_long['pnl']:+.2f}% mdd={base_long['mdd']:.2f}%  AS pnl={base_short['pnl']:+.2f}% mdd={base_short['mdd']:.2f}%")
        log(f"    new_signal: AL pnl={new_long['with_gate']['pnl']:+.2f}% mdd={new_long['with_gate']['mdd']:.2f}%  "
            f"AS pnl={new_short['with_gate']['pnl']:+.2f}% mdd={new_short['with_gate']['mdd']:.2f}%")
        log(f"    mdd_improves_always_long={mdd_improves_always_long}  guardrail_always_short={guardrail_always_short}  criterion2_pass={criterion2_pass}")

    overall_pass = bool(passing_windows) and any(v["criterion2_pass"] for v in criterion2.values())
    downtrend_pass = [w for w in passing_windows if w in JUDGED_DOWNTREND_WINDOWS]
    ranging_pass = [w for w in passing_windows if w in RANGING_WINDOW_KEYS]

    report = {
        "design": __doc__,
        "correction_of": "research_ilias_eth_adaptive_exit_signal_arm_eval_20260817.py (original, untouched, preserved)",
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "new_exit_signal_bundle": {"model_name": bundle["model_name"], "threshold": bundle["threshold"],
                                     "n_train_trades": bundle["n_train_trades"], "n_train_rows": bundle["n_train_rows"],
                                     "feature_columns": bundle["feature_columns"]},
        "seeds": seeds,
        "criterion1_by_window": criterion1,
        "criterion1_passing_windows": passing_windows,
        "criterion1_passing_downtrend_windows": downtrend_pass,
        "criterion1_passing_ranging_windows": ranging_pass,
        "criterion2_by_window": criterion2,
        "final_verdict": "SUCCESS" if overall_pass else "KILL",
        "kill_reason": None if overall_pass else (
            "criterion1_failed_all_windows" if not passing_windows else "criterion2_failed_all_criterion1_passing_windows"
        ),
        "results_csv": str(results_csv),
    }
    report_path = OUT_DIR / "arm_eval_report_sideblind.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating, np.bool_)) else str(o)),
        encoding="utf-8",
    )
    log(f"report={report_path}")
    log(f"FINAL_VERDICT={report['final_verdict']}  passing_windows={passing_windows}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
