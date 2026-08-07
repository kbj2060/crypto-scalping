#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRAIN_SCRIPT = ROOT / "scripts/alpha6_catboost_entry_quality_exit_policy_20260522.py"
DEFAULT_PRESETS = (
    "current_quality",
    "density_balanced",
    "regime_conditional",
    "perturbation_robust",
    "adverse_conformal",
    "sam_conformal",
    "high_precision_robust",
    "turnover_balanced_robust",
)
HORIZON_REG_PRESETS = ("scalp_short_horizon", "short_horizon_robust", "pullback_entry")
FIXED_HORIZON_PRESETS = {
    "scalp_short_horizon": 12,
    "short_horizon_robust": 24,
    "pullback_entry": 24,
}


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-", "."} else "_" for ch in value)


def _auto_candidates(include_regime: bool = True) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for preset in DEFAULT_PRESETS:
        if preset == "regime_conditional" and not include_regime:
            continue
        out.append({"name": preset, "preset": preset, "mode": "bucket5", "fixed": 0})
    for preset in HORIZON_REG_PRESETS:
        out.append({"name": f"{preset}_hreg", "preset": preset, "mode": "horizon_reg", "fixed": 0})
    for preset, horizon in FIXED_HORIZON_PRESETS.items():
        out.append({"name": f"{preset}_fixed{horizon}", "preset": preset, "mode": "fixed", "fixed": int(horizon)})
    return out


def _preset_candidates(raw: str) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for token in [p.strip() for p in str(raw).split(",") if p.strip()]:
        parts = token.split(":")
        preset = parts[0]
        mode = parts[1] if len(parts) > 1 and parts[1] else "bucket5"
        fixed = int(parts[2]) if len(parts) > 2 and parts[2] else 0
        suffix = "" if mode == "bucket5" else f"_{mode}"
        if mode == "fixed":
            suffix = f"_fixed{fixed or FIXED_HORIZON_PRESETS.get(preset, 0)}"
        candidates.append({"name": f"{preset}{suffix}", "preset": preset, "mode": mode, "fixed": fixed})
    return candidates


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Alpha6 bucket5 CatBoost label-preset ensemble candidates.")
    ap.add_argument("--variant", default="current_tail111")
    ap.add_argument("--out-root", type=Path, default=ROOT / "tmp/causal_regen_20260516/alpha6_bucket5_label_ensemble_20260523")
    ap.add_argument("--plan", choices=["auto", "auto-no-regime", "presets"], default="auto")
    ap.add_argument(
        "--presets",
        default="",
        help="Used with --plan presets. Format: preset[:bucket5|horizon_reg|fixed[:fixed_horizon]],...",
    )
    ap.add_argument("--iterations", type=int, default=650)
    ap.add_argument("--exit-iterations", type=int, default=500)
    ap.add_argument("--entry-thresholds", type=int, default=50)
    ap.add_argument("--exit-max-trades", type=int, default=9000)
    ap.add_argument("--exit-step", type=int, default=2)
    ap.add_argument("--eval-costs", default="1,2,3")
    ap.add_argument("--exit-threshold-grid", default="0.35,0.45,0.55,0.70")
    ap.add_argument("--extra-args", default="", help="Extra raw args passed to every candidate.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    args = ap.parse_args()

    if args.plan == "auto":
        candidates = _auto_candidates(include_regime=True)
    elif args.plan == "auto-no-regime":
        candidates = _auto_candidates(include_regime=False)
    else:
        candidates = _preset_candidates(args.presets)
    args.out_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, object]] = []

    for candidate in candidates:
        preset = str(candidate["preset"])
        mode = str(candidate["mode"])
        fixed = int(candidate.get("fixed") or 0)
        out_dir = args.out_root / _safe_name(str(candidate["name"]))
        cmd = [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--variant",
            str(args.variant),
            "--out-dir",
            str(out_dir),
            "--label-preset",
            preset,
            "--target-head-mode",
            mode,
            "--iterations",
            str(args.iterations),
            "--exit-iterations",
            str(args.exit_iterations),
            "--entry-thresholds",
            str(args.entry_thresholds),
            "--exit-max-trades",
            str(args.exit_max_trades),
            "--exit-step",
            str(args.exit_step),
            "--eval-costs",
            str(args.eval_costs),
            "--exit-threshold-grid",
            str(args.exit_threshold_grid),
            "--verbose",
            "0",
        ]
        if fixed > 0:
            cmd.extend(["--fixed-target-horizon", str(fixed)])
        if preset == "pullback_entry":
            cmd.extend(["--entry-pullback-atr", "0.30"])
        if args.smoke:
            cmd.append("--smoke")
        if str(args.extra_args).strip():
            cmd.extend(str(args.extra_args).strip().split())

        status = "ok"
        try:
            print(f"[alpha6-ensemble] preset={preset} mode={mode} fixed={fixed} out={out_dir}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as exc:
            status = f"failed:{exc.returncode}"
            if not args.keep_going:
                raise
        manifest.append(
            {
                "preset": preset,
                "target_head_mode": mode,
                "fixed_target_horizon": fixed,
                "status": status,
                "out_dir": str(out_dir),
                "summary": str(out_dir / f"{args.variant}_summary.json"),
                "threshold_grid": str(out_dir / f"{args.variant}_threshold_grid.csv"),
                "train_labels": str(out_dir / f"{args.variant}_train_labels.csv"),
                "val_predictions": str(out_dir / f"{args.variant}_val_predictions.csv"),
                "bundle": str(out_dir / f"{args.variant}_bundle.joblib"),
            }
        )
        (args.out_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
