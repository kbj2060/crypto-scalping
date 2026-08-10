#!/usr/bin/env python3
"""SOL dual-component regime router: seed-ensemble-averaged probabilities + a genuinely
RE-TUNED (not frozen) risk-sidecar mapping and router structure grid.

Background: the 2026-07-29 dual-component router (ZIG075 x H24-wide, bull/bear/chop routing)
looked like the best SOL ML candidate ever -- VAL and OOS both beat the rule-baseline on PnL
and MDD -- but a 5-seed reproducibility test
(docs/sol_dual_h24wide_seed_stability_20260729.md) showed the original result was a seed-lucky
outlier: new-seed VAL mean -1.12%, OOS mean +3.35%, far below the original single-seed run.
That doc's own prescribed next step was never executed:

    "Do not select the best seed. The next candidate should reduce training variance before
    performance selection, preferably by averaging direction, quality, and exit probabilities
    across a fixed seed ensemble and then running a new, untouched forward test."

This script executes exactly that, reusing the 5 already-trained seed parent+sidecar artifacts
from 2026-07-29 (seeds 17/29/43/71/101, both components) -- no GPU retraining is needed for the
averaging step itself:

  1. Average each component's direction/quality probabilities across the 5 seeds
     (scripts/build_fixed_seed_prediction_ensemble_20260729.py, unmodified, reused as-is).
  2. Merge the averaged predictions with one reference seed's true_3head_tabm_bundle.pt. The
     exit head is architecturally NOT ensembled by that script (same documented limitation as
     its own manifest: exit_head_ensemble=false) -- REFERENCE_SEED below picks which seed's exit
     head + base-feature contract carries through.
  3. Retrain each component's risk sidecar from scratch on the averaged predictions WITHOUT
     --fixed-mapping-report -- i.e. a genuine margin/leverage sigmoid-mapping grid search on the
     smoothed ensemble scores, rather than reusing one (possibly overfit) seed's frozen mapping.
     This is the actual "retune" step: the 2026-07-29 seed-stability test explicitly froze the
     mapping across all 5 seeds and only this step was left untested.
  4. Re-run the full VAL router-structure grid search (bull/bear/chop component assignment +
     regime margin scale) and one frozen OOS read, via the unmodified
     scripts/eval_sol_dual_structure_router_20260729.py.

Every sub-step calls the exact tested script/module used in the original 2026-07-29 line -- this
file is an orchestrator, not a reimplementation. Cost model, VAL/OOS boundaries (2025-09-01 /
2026-01-01 / 2026-04-01), and the fresh-forward bar-by-bar contract are all inherited unchanged
from those scripts.

If VAL beats the rule-baseline here, this is still NOT promotion evidence -- the recommended next
step (not done by this script) is a fresh GPU retrain with N=5 genuinely NEW random seeds to
confirm the result is not an artifact of averaging these specific 5 seeds, per this project's
Seed-Diversity Ensemble Promotion Gate policy.

Usage:
    python scripts/eval_sol_dual_router_seed_ensemble_retune_20260810.py
    python scripts/eval_sol_dual_router_seed_ensemble_retune_20260810.py --skip-existing
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TMP = ROOT / "tmp" / "causal_regen_20260516"
SCRIPTS = ROOT / "scripts"

SEEDS = (17, 29, 43, 71, 101)
# Exit head + base-feature contract source for both components. Arbitrary/first of the 5 --
# documented limitation, matches build_fixed_seed_prediction_ensemble_20260729.py's own declared
# ensemble_scope ("entry_direction_and_quality_probabilities_only", exit_head_ensemble=false).
REFERENCE_SEED = 17

COMPONENTS = {
    "zig075": {
        "tag": "q060",
        "quality_threshold": 0.60,
        "parent_dir_pattern": (
            "sol_omega4_3head_parent72_loose_entry_quality_20260707_dualseed_zig075_exit30k_s{seed}_20260729"
        ),
        "label_dir": TMP / "sol_dual_zig075_h24wide_splitlocal_20260729" / "zig075",
    },
    "h24wide": {
        "tag": "q055",
        "quality_threshold": 0.55,
        "parent_dir_pattern": (
            "sol_omega4_3head_parent72_loose_entry_quality_20260707_dualseed_h24wide_exit30k_s{seed}_20260729"
        ),
        "label_dir": TMP / "sol_dual_zig075_h24wide_splitlocal_20260729" / "h24wide",
    },
}

RUN_TAG = "seedensemble_retune_20260810"

# Cited for the final comparison table only -- not recomputed by this script. Sources:
# docs/sol_dual_h24wide_final_20260729.md (rule_baseline, original_single_seed) and
# docs/sol_dual_h24wide_seed_stability_20260729.md (5seed_mean, across seeds 17/29/43/71/101 with
# the mapping frozen).
REFERENCE_NUMBERS = {
    "rule_baseline": {"val_pnl": 23.45, "val_mdd": -7.69, "oos_pnl": 7.66, "oos_mdd": -12.52},
    "original_single_seed": {"val_pnl": 25.08, "val_mdd": -7.49, "oos_pnl": 21.62, "oos_mdd": -9.88},
    "5seed_mean_frozen_mapping": {"val_pnl": -1.12, "val_mdd": -11.68, "oos_pnl": 3.35, "oos_mdd": -15.87},
}


def run(cmd: list) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"\n$ {printable}", flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def stage_ensemble(component: str, cfg: dict) -> Path:
    """Average the 5 seeds' direction/quality probabilities (unmodified 07-29 script)."""
    seed_dirs = {seed: TMP / cfg["parent_dir_pattern"].format(seed=seed) for seed in SEEDS}
    for seed, path in seed_dirs.items():
        if not path.is_dir():
            raise FileNotFoundError(f"missing seed {seed} parent dir for {component}: {path}")

    out_dir = TMP / f"sol_dual_seedensemble_{component}_{cfg['tag']}_{RUN_TAG}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [sys.executable, SCRIPTS / "build_fixed_seed_prediction_ensemble_20260729.py"]
    for seed, path in sorted(seed_dirs.items()):
        cmd += ["--seed-dir", f"{seed}={path}"]
    cmd += ["--quality-threshold", str(cfg["quality_threshold"]), "--out-dir", out_dir]
    run(cmd)

    # Merge in the reference seed's exit-head/base-feature bundle. Required because
    # eval_sol_dual_structure_router_20260729.py::prepare_component reads
    # {parent_dir}/true_3head_tabm_bundle.pt unconditionally (for the exit head and base feature
    # contract, which this ensemble step does not touch). The bundle is fully self-contained
    # (state_dict + scaler in memory -- see
    # train_eval_omega1_2_tabm_3head_20260603.py::_load_payloads), so copying just this one file
    # is sufficient; the models/ subdirectory on disk is redundant.
    ref_bundle = seed_dirs[REFERENCE_SEED] / "true_3head_tabm_bundle.pt"
    shutil.copy2(ref_bundle, out_dir / "true_3head_tabm_bundle.pt")
    print(f"  merged reference bundle from seed {REFERENCE_SEED} into {out_dir}")
    return out_dir


def stage_retrain_sidecar(component: str, cfg: dict, ensemble_dir: Path) -> Path:
    """Retrain the risk sidecar from scratch on the averaged predictions -- full mapping grid
    search, NOT a frozen mapping. Goes through run_sol_dual_sidecar_candidate_retune_20260810.py
    (a copy of the original 07-29 wrapper with --fixed-mapping-report made optional) rather than
    calling train_eval_omega4_2_risk_sidecar_sol_20260707.py directly, because that wrapper also
    monkeypatches parent.SPLIT_TS / omega.SPLIT_TS to 2025-09-01 before training -- both modules'
    own module-level default is 2025-10-01, and without this patch _prepare_frames() builds a
    train/validation boundary one month later than the one baked into every 07-29 prediction CSV,
    which fails the script's own precomputed-prediction timestamp-contract check. Confirmed via a
    direct diagnostic run against seed 17's own unmodified original predictions before this fix.
    Every other hyperparameter matches the original 07-29 sidecar exactly (read from that run's
    own report.json risk_model block: exit_threshold 0.95 / validation_only scope /
    full-replay-top-k 5)."""
    out_suffix = f"seedensemble_{component}_retune_20260810"
    out_dir = TMP / f"sol_omega4_2_trade_risk_sidecar_20260707_{out_suffix}"
    if out_dir.exists():
        shutil.rmtree(out_dir)

    cmd = [
        sys.executable, SCRIPTS / "run_sol_dual_sidecar_candidate_retune_20260810.py",
        "--parent-dir", ensemble_dir,
        "--tag", cfg["tag"],
        "--quality-threshold", str(cfg["quality_threshold"]),
        "--out-suffix", out_suffix,
        "--direction-label-dir", cfg["label_dir"],
        "--seed", "260810",
    ]
    run(cmd)
    return out_dir


def stage_router_eval(zig_dir: Path, zig_risk_dir: Path, h24_dir: Path, h24_risk_dir: Path) -> Path:
    cmd = [
        sys.executable, SCRIPTS / "eval_sol_dual_structure_router_20260729.py",
        "--zig-parent-dir", zig_dir, "--zig-risk-dir", zig_risk_dir,
        "--zig-tag", COMPONENTS["zig075"]["tag"],
        "--h24-parent-dir", h24_dir, "--h24-risk-dir", h24_risk_dir,
        "--h24-tag", COMPONENTS["h24wide"]["tag"],
        "--risk-mode", "sidecar",
        "--out-suffix", RUN_TAG,
        "--device", "cuda",
    ]
    run(cmd)
    zig_tag, h24_tag = COMPONENTS["zig075"]["tag"], COMPONENTS["h24wide"]["tag"]
    # eval_sol_dual_structure_router_20260729.py prefixes a non-empty --out-suffix with its own
    # "_" internally, so the resulting dirname has RUN_TAG appended with exactly one underscore.
    return TMP / f"sol_dual_structure_router_sidecar_{zig_tag}_{h24_tag}_20260729_{RUN_TAG}"


def print_comparison(report_dir: Path) -> dict:
    report_path = report_dir / "report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    val = report["selected_on_validation"]["validation"]
    oos = report["frozen_oos"]
    row = {"val_pnl": val["pnl"], "val_mdd": val["mdd"], "oos_pnl": oos["pnl"], "oos_mdd": oos["mdd"]}
    table = {**REFERENCE_NUMBERS, "seed_ensemble_retune (THIS RUN)": row}

    print("\n" + "=" * 82)
    print(f"{'candidate':34s} {'VAL pnl':>10s} {'VAL mdd':>10s} {'OOS pnl':>10s} {'OOS mdd':>10s}")
    for name, r in table.items():
        marker = "  <==" if "THIS RUN" in name else ""
        print(f"{name:34s} {r['val_pnl']:>9.2f}% {r['val_mdd']:>9.2f}% {r['oos_pnl']:>9.2f}% {r['oos_mdd']:>9.2f}%{marker}")
    print("=" * 82)
    print(f"trades: val={val.get('trades')} wr={val.get('wr', 0):.3f}  oos: trades={oos.get('trades')} wr={oos.get('wr', 0):.3f}")
    print(f"selected variant: {report['selected_on_validation']['variant']['name']}")
    print(f"\nfull report: {report_path}")

    beats_baseline = (
        row["val_pnl"] > REFERENCE_NUMBERS["rule_baseline"]["val_pnl"]
        and row["val_mdd"] > REFERENCE_NUMBERS["rule_baseline"]["val_mdd"]
        and row["oos_pnl"] > REFERENCE_NUMBERS["rule_baseline"]["oos_pnl"]
        and row["oos_mdd"] > REFERENCE_NUMBERS["rule_baseline"]["oos_mdd"]
    )
    print(f"\nbeats rule_baseline on PnL AND MDD in BOTH windows: {beats_baseline}")
    if beats_baseline:
        print("-> genuinely promising; next step is a fresh GPU retrain with N=5 NEW random seeds")
        print("   to confirm this isn't an artifact of averaging these specific 5 seeds.")
    else:
        print("-> does not clear the bar. Consistent with the 07-31 seed-ensemble-averaging finding")
        print("   on this same asset (ensembling reduces variance but cannot fix a negative mean).")
    return {"table": table, "beats_baseline": beats_baseline, "report_path": str(report_path)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="reuse ensemble/sidecar outputs from a prior run of this script if already present",
    )
    args = parser.parse_args()

    ensemble_dirs = {}
    for name, cfg in COMPONENTS.items():
        print(f"\n### stage 1/3 -- probability ensemble ({len(SEEDS)} seeds): {name} ###")
        out_dir = TMP / f"sol_dual_seedensemble_{name}_{cfg['tag']}_{RUN_TAG}"
        if args.skip_existing and out_dir.exists() and (out_dir / "true_3head_tabm_bundle.pt").exists():
            print(f"  skip-existing: reusing {out_dir}")
        else:
            out_dir = stage_ensemble(name, cfg)
        ensemble_dirs[name] = out_dir

    sidecar_dirs = {}
    for name, cfg in COMPONENTS.items():
        print(f"\n### stage 2/3 -- risk sidecar retrain (full mapping grid, NOT frozen): {name} ###")
        out_suffix = f"seedensemble_{name}_retune_20260810"
        out_dir = TMP / f"sol_omega4_2_trade_risk_sidecar_20260707_{out_suffix}"
        if args.skip_existing and out_dir.exists() and (out_dir / "report.json").exists():
            print(f"  skip-existing: reusing {out_dir}")
        else:
            out_dir = stage_retrain_sidecar(name, cfg, ensemble_dirs[name])
        sidecar_dirs[name] = out_dir

    print("\n### stage 3/3 -- router structure grid search (VAL) + frozen OOS read (H24wide x ZIG075) ###")
    report_dir = stage_router_eval(
        ensemble_dirs["zig075"], sidecar_dirs["zig075"],
        ensemble_dirs["h24wide"], sidecar_dirs["h24wide"],
    )

    summary = print_comparison(report_dir)
    summary_path = TMP / f"sol_dual_router_{RUN_TAG}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
