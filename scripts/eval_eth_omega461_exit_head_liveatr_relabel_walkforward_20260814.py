#!/usr/bin/env python3
"""RESEARCH ONLY -- evaluation harness for the walk-forward retraining robustness folds built by
`train_eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814.py` (folds B/C/D) plus the
already-existing Fold A (the original, un-retrained live-ATR-relabel run, see
docs/experiments/eth_omega461_live_exit_head_liveatr_relabel_20260813.md).

For each fold, evaluates h48qual COMPONENT-level (baseline frozen exit_head vs this fold's newly
retrained exit_head -- encoder/direction_head/quality_head always frozen to the same original live
bundle in every fold) on that fold's own pre-registered confirm window, loaded via
`eth_omega461_multiwindow_confirmation_gate_20260814.load_all_windows` (imported as `gate`, never
reimplemented) -- the SAME 6-window loader every other Odyssey2/3 candidate uses:

  Fold A (not retrained -- reuses the original 2026-08-13 shadow-deployed bundle):
    train window 2025-01-01..2025-09-30 (original, hardcoded parent.SPLIT_TS) -> confirm window "val"
  Fold B: train window 2025-01-01..2025-06-30 -> confirm window "2025q3" (sustained uptrend)
  Fold C: train window 2025-01-01..2025-12-31 -> confirm window "oos_q1" (downtrend)
  Fold D: train window 2025-01-01..2026-03-31 -> confirm window "oos_q2" (whipsaw)

Component-level evaluation reuses `research_eth_omega461_exit_sweep_20260721.prep_component` /
`.replay_exit_variant` (imported as `sweep`) -- the SAME two primitives
`research_eth_omega461_exit_head_h48cons_relabel_20260813._evaluate_val` already uses, just no
longer hardcoded to the VAL window (that function only ever loads `sweep.VAL_START`/`VAL_END`; this
script's `_evaluate_component_on_window` takes a `gate.load_all_windows()` window entry instead, so
the SAME two-line "prep baseline bundle, prep candidate bundle, replay both" pattern works on any of
the 6 pre-registered windows). Portfolio-level (h48qual on this fold's new exit_head, zig075 always
on its fully original frozen exit_head -- the same "asymmetric" pairing already shadow-deployed)
reuses `gate.run_portfolio_variant` verbatim with a bundle-override component config, exactly the
pattern the gate module's own G0 self-check already exercises for the ORIGINAL (non-walk-forward)
liveatr-relabel bundle.

G0 self-check: Fold A's component-level numbers are recomputed here (bundle = the actual shadow-
deployed bundle, window = "val") and compared against the already-published reference (baseline PnL
+5.4545%/MDD -11.6196%/29 trades, relabel PnL +9.2289%/MDD -7.5940%/63 trades, read directly from
`tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json`
components.h48qual.val_metrics) -- if this script's generalized evaluator cannot reproduce
already-published numbers on the one window it CAN be checked against, its numbers for folds B/C/D
(which have no prior reference) should not be trusted either. Mirrors the gate module's own G0
self-check convention.

fresh_forward_bar_by_bar=true (every replay here is `sweep.replay_exit_variant` or
`gate.run_portfolio_variant`'s own `greedy.greedy_replay`, both unmodified single causal forward
passes). trade_ledgers_used_as_input=false. saved_parent_exit_timestamps_used=false. future_rows_
used_for_entry=false. Does NOT touch trading_bot_modules/omega4_6_1_live.py, trading_bot.py,
runtime_config.py, .env. Does NOT modify any imported module -- research_eth_omega461_exit_sweep_
20260721.py, eth_omega461_multiwindow_confirmation_gate_20260814.py, and research_eth_omega461_exit_
head_portfolio_asymmetric_20260813.py are imported and read only. No retraining, no GPU -- every
prediction CSV this script reads already exists on disk.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import research_eth_omega461_exit_sweep_20260721 as sweep  # noqa: E402
import research_eth_omega461_exit_head_portfolio_asymmetric_20260813 as portfolio  # noqa: E402
import eth_omega461_multiwindow_confirmation_gate_20260814 as gate  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_20260814"
DEVICE = torch.device("cpu")

FOLD_A_BUNDLE = (
    ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500"
    "/h48qual/true_3head_tabm_bundle.pt"
)
FOLD_A_REFERENCE = {
    # tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_20260813_full1500/report.json
    # components.h48qual.val_metrics -- reconfirmed by direct read before this script was written.
    "baseline": {"pnl": 5.454527242757146, "mdd": -11.619634060124607, "trades": 29},
    "new": {"pnl": 9.228934674546663, "mdd": -7.593974540850745, "trades": 63},
}

FOLDS: dict[str, dict[str, Any]] = {
    "A": {
        "window": "val",
        "train_start": "2025-01-01", "train_end": "2025-09-30 (original hardcoded parent.SPLIT_TS run, not re-run)",
        "bundle": FOLD_A_BUNDLE,
        "retrained": False,
        "note": "not retrained -- reuses the original 2026-08-13 shadow-deployed bundle as-is",
    },
    "B": {
        "window": "2025q3",
        "train_start": "2025-01-01", "train_end": "2025-07-01 (exclusive, i.e. through 2025-06-30)",
        "bundle": ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldB/h48qual/true_3head_tabm_bundle.pt",
        "retrained": True,
    },
    "C": {
        "window": "oos_q1",
        "train_start": "2025-01-01", "train_end": "2026-01-01 (exclusive, i.e. through 2025-12-31)",
        "bundle": ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldC/h48qual/true_3head_tabm_bundle.pt",
        "retrained": True,
    },
    "D": {
        "window": "oos_q2",
        "train_start": "2025-01-01", "train_end": "2026-04-01 (exclusive, i.e. through 2026-03-31)",
        "bundle": ROOT / "tmp/causal_regen_20260516/eth_omega461_exit_head_liveatr_relabel_walkforward_fold_20260814_foldD/h48qual/true_3head_tabm_bundle.pt",
        "retrained": True,
    },
}


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _evaluate_component_on_window(component: str, new_bundle_path: Path, window: dict[str, Any]) -> dict[str, Any]:
    """Generalization of `research_eth_omega461_exit_head_h48cons_relabel_20260813._evaluate_val`
    (which hardcodes `sweep.VAL_START`/`VAL_END`) to any of the 6 pre-registered windows returned by
    `gate.load_all_windows()`. Reuses `sweep.prep_component`/`sweep.replay_exit_variant` UNCHANGED --
    only the frame/pred_csv/oof source (window["frame"]/window["raw_paths"][component]/window["oof"]
    instead of a hardcoded VAL load) differs."""
    cfg = dict(sweep.COMPONENTS[component])
    frame = window["frame"]
    pred_csv = window["raw_paths"][component]
    oof = bool(window["oof"])

    baseline_prepped = sweep.prep_component(component, cfg, frame, pred_csv, oof=oof)
    m_baseline, _ledger_baseline = sweep.replay_exit_variant(
        baseline_prepped["frame"], baseline_prepped["x"], baseline_prepped["dec"], baseline_prepped["loaded"],
        risk_margin_fraction=baseline_prepped["margin"], risk_leverage=baseline_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=baseline_prepped["fee"], slip=baseline_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=baseline_prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )

    cfg_new = dict(cfg)
    cfg_new["bundle"] = new_bundle_path
    new_prepped = sweep.prep_component(component, cfg_new, frame, pred_csv, oof=oof)
    m_new, _ledger_new = sweep.replay_exit_variant(
        new_prepped["frame"], new_prepped["x"], new_prepped["dec"], new_prepped["loaded"],
        risk_margin_fraction=new_prepped["margin"], risk_leverage=new_prepped["leverage"],
        exit_threshold=sweep.BASELINE_EXIT_THRESHOLD, fee=new_prepped["fee"], slip=new_prepped["slip"],
        cost_mult=sweep.COST_MULT, notional_scaled_sltp=new_prepped["notional_scaled_sltp"], device=sweep.DEVICE,
    )
    return {"baseline": m_baseline, "new": m_new}


def _close(actual: dict[str, Any], expected: dict[str, Any], *, tol_pp: float = 0.01) -> bool:
    return bool(
        abs(float(actual["pnl"]) - float(expected["pnl"])) <= tol_pp
        and abs(float(actual["mdd"]) - float(expected["mdd"])) <= tol_pp
        and int(actual["trades"]) == int(expected["trades"])
    )


def _sign_verdict(baseline: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    pnl_up = bool(float(new["pnl"]) > float(baseline["pnl"]))
    mdd_improved = bool(float(new["mdd"]) >= float(baseline["mdd"]))  # less negative = better
    return {"pnl_up": pnl_up, "mdd_improved_or_equal": mdd_improved, "relabel_beats_original": bool(pnl_up and mdd_improved)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    device = DEVICE

    print("=== stage=load_all_windows (reused from eth_omega461_multiwindow_confirmation_gate_20260814) ===", flush=True)
    windows = gate.load_all_windows()
    verify_diag = gate.verify_windows(windows)
    high_coverage = all(row[f"{n}_intersection_high_coverage"] for row in verify_diag.values() for n in ("h48qual", "zig075"))
    print(f"  window_verification high_coverage_pass={high_coverage}", flush=True)

    report: dict[str, Any] = {
        "design": (
            "Walk-forward RETRAINING robustness evaluation: for each fold (A=original unretrained, "
            "B/C/D=retrained on a different training window via train_eth_omega461_exit_head_"
            "liveatr_relabel_walkforward_fold_20260814.py), evaluate h48qual component-level "
            "(baseline frozen exit_head vs this fold's exit_head) and portfolio-level (h48qual=fold "
            "exit_head, zig075=always original) on that fold's own pre-registered confirm window."
        ),
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "window_verification_high_coverage_pass": high_coverage,
        "folds": {},
    }
    if not high_coverage:
        report["note"] = "window verification failed -- aborting before trusting any fold number"
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print("stage=ABORT window_verification failed", flush=True)
        return 1

    print("=== stage=G0_self_check (Fold A component-level, must reproduce already-published numbers) ===", flush=True)
    g0 = _evaluate_component_on_window("h48qual", FOLD_A_BUNDLE, windows["val"])
    g0_baseline_ok = _close(g0["baseline"], FOLD_A_REFERENCE["baseline"])
    g0_new_ok = _close(g0["new"], FOLD_A_REFERENCE["new"])
    g0_pass = bool(g0_baseline_ok and g0_new_ok)
    print(f"  baseline actual={g0['baseline']['pnl']:.4f}%/{g0['baseline']['mdd']:.4f}%/{g0['baseline']['trades']}t "
          f"reference={FOLD_A_REFERENCE['baseline']} match={g0_baseline_ok}", flush=True)
    print(f"  new      actual={g0['new']['pnl']:.4f}%/{g0['new']['mdd']:.4f}%/{g0['new']['trades']}t "
          f"reference={FOLD_A_REFERENCE['new']} match={g0_new_ok}", flush=True)
    report["g0_self_check"] = {"actual": g0, "reference": FOLD_A_REFERENCE, "pass": g0_pass}
    print(f"stage=G0_result pass={g0_pass}", flush=True)
    if not g0_pass:
        report["note"] = "G0 self-check failed -- _evaluate_component_on_window does not reproduce already-published Fold A numbers on the VAL window. Aborting before trusting folds B/C/D (no prior reference exists for them)."
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print("stage=ABORT G0 self-check failed", flush=True)
        return 1

    comp_cfgs_baseline = {"h48qual": portfolio._component_cfg("h48qual"), "zig075": portfolio._component_cfg("zig075")}

    for fold_name, fold in FOLDS.items():
        window_name = fold["window"]
        bundle_path = Path(fold["bundle"])
        print(f"=== stage=fold_{fold_name} window={window_name} retrained={fold['retrained']} bundle_exists={bundle_path.exists()} ===", flush=True)
        if not bundle_path.exists():
            report["folds"][fold_name] = {**fold, "bundle": str(bundle_path), "error": "bundle not found -- fold training did not complete"}
            print(f"  SKIP fold {fold_name}: bundle not found at {bundle_path}", flush=True)
            continue

        if fold_name == "A":
            comp = g0  # already computed above (G0 self-check IS fold A's component-level number), avoid recomputing
        else:
            comp = _evaluate_component_on_window("h48qual", bundle_path, windows[window_name])
        comp_verdict = _sign_verdict(comp["baseline"], comp["new"])
        print(f"  component h48qual: baseline pnl={comp['baseline']['pnl']:.2f}% mdd={comp['baseline']['mdd']:.2f}% trades={comp['baseline']['trades']} | "
              f"new pnl={comp['new']['pnl']:.2f}% mdd={comp['new']['mdd']:.2f}% trades={comp['new']['trades']} | verdict={comp_verdict}", flush=True)

        comp_cfgs_new = {"h48qual": portfolio._component_cfg("h48qual", bundle_override=bundle_path), "zig075": portfolio._component_cfg("zig075")}
        port_baseline = gate.run_portfolio_variant(window_name, windows, comp_cfgs_baseline, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label=f"fold{fold_name}_baseline_both_original")
        port_new = gate.run_portfolio_variant(window_name, windows, comp_cfgs_new, fee=fee, slip=slip, device=device, out_dir=OUT_DIR, variant_label=f"fold{fold_name}_asymmetric_new")
        port_verdict_nogate = _sign_verdict(port_baseline["no_gate"], port_new["no_gate"])
        port_verdict_gate = _sign_verdict(port_baseline["with_gate"], port_new["with_gate"])
        print(f"  portfolio no_gate:   baseline pnl={port_baseline['no_gate']['pnl']:.2f}% mdd={port_baseline['no_gate']['mdd']:.2f}% | "
              f"new pnl={port_new['no_gate']['pnl']:.2f}% mdd={port_new['no_gate']['mdd']:.2f}% | verdict={port_verdict_nogate}", flush=True)
        print(f"  portfolio with_gate: baseline pnl={port_baseline['with_gate']['pnl']:.2f}% mdd={port_baseline['with_gate']['mdd']:.2f}% | "
              f"new pnl={port_new['with_gate']['pnl']:.2f}% mdd={port_new['with_gate']['mdd']:.2f}% | verdict={port_verdict_gate}", flush=True)

        report["folds"][fold_name] = {
            **{k: v for k, v in fold.items() if k != "bundle"},
            "bundle": str(bundle_path),
            "component_h48qual": {"baseline": comp["baseline"], "new": comp["new"], "verdict": comp_verdict},
            "portfolio_no_gate": {"baseline": port_baseline["no_gate"], "new": port_new["no_gate"], "verdict": port_verdict_nogate},
            "portfolio_with_gate": {"baseline": port_baseline["with_gate"], "new": port_new["with_gate"], "verdict": port_verdict_gate},
        }

    n_folds_evaluated = sum(1 for f in report["folds"].values() if "error" not in f)
    n_component_reproduced = sum(1 for f in report["folds"].values() if "error" not in f and f["component_h48qual"]["verdict"]["relabel_beats_original"])
    report["summary"] = {
        "n_folds_evaluated": n_folds_evaluated,
        "n_folds_component_relabel_beats_original": n_component_reproduced,
        "n_folds_total_designed": len(FOLDS),
    }
    print(f"=== SUMMARY: component-level relabel-beats-original in {n_component_reproduced}/{n_folds_evaluated} evaluated folds (of {len(FOLDS)} designed) ===", flush=True)

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(f"report={OUT_DIR / 'report.json'}", flush=True)
    print("stage=done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
