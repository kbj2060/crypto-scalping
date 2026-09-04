"""Step 2 of the BTC live pipeline for the redesigned-JM candidate: extended-window inference.

The live BTC stack is four stages, and the sidecar's own metrics are only stage 3:

    1. parent train on the standard frames                  -> true_3head_tabm_bundle.pt
    2. INFERENCE ONLY from the frozen bundle over the        -> oos_predictions_qXXX.csv
       fresh-forward EXTENDED window
    3. risk sidecar on the standard frames                   -> risk_sidecar.pkl
    4. final replay on the EXTENDED frames with the duration -> the number the promotion was
       gate, scale map and exit threshold applied               judged on (+10.76% OOS)

Stage 2 was missing, which is why the stage-4 replay rejected the candidate's predictions with
"oos: precomputed prediction timestamps do not match prepared frame" -- stage 4 prepares the
extended frame, and the parent only ever predicted the standard one.

This forks scripts/infer_btc_h48qual_predictions_freshforward_ext_swingtransition_20260806.py,
which has no CLI, by overriding its module constants: the candidate's bundle, the candidate's
quality threshold (0.50, VAL-selected, not the incumbent's 0.55), a per-seed output directory, and
the candidate regime overlay on the omega module. Nothing is retrained -- the bundle is frozen and
this is pure forward inference, same as the live stage 2.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import infer_btc_h48qual_predictions_freshforward_ext_swingtransition_20260806 as infer  # noqa: E402

TAG = "jmredesign_20260810"
SUP = ROOT / "data/ensemble/supervised"
RUN_ROOT = ROOT / "tmp/causal_regen_20260516"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--quality-threshold", type=float, default=0.50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--train-rows", type=int, default=30000)
    args = ap.parse_args()

    # the prefix matters: this script's own OUT_DIR ends with the same
    # _e{epochs}_r{rows}_s{seed} suffix, so a loose glob picks the output of a previous attempt as
    # the "parent" and then tries to load a bundle out of it
    hits = [p for p in sorted(RUN_ROOT.glob(
        f"btc_omega4_3head_parent72_*{TAG}_e{args.epochs}_r{args.train_rows}_s{args.seed}"))
        if p.is_dir()]
    if not hits:
        raise SystemExit(f"no parent dir for seed {args.seed}")
    parent_dir = hits[-1]
    qtag = f"q{int(round(args.quality_threshold * 100)):03d}"

    # _load_omega_frames() reads the BARE module globals REGIME3_CURRENT_2025/2026, so the
    # override has to land on that exact module object -- reaching it through whatever alias the
    # importing script happens to expose is not enough (and silently did nothing on the first try).
    import train_eval_omega1_2_tabm_diffusion_risk_btc_swingtransition_20260806 as omega_mod
    omega_mod.REGIME3_CURRENT_2025 = SUP / f"btc_regime3_current_hmm_{TAG}_2025_maskedname.csv"
    omega_mod.REGIME3_CURRENT_2026 = SUP / f"btc_regime3_current_hmm_{TAG}_2026_maskedname.csv"

    infer.BASELINE_BUNDLE = parent_dir / "true_3head_tabm_bundle.pt"
    infer.Q_THRESHOLD = float(args.quality_threshold)
    infer.Q_TAG = qtag
    infer.OUT_DIR = RUN_ROOT / f"btc_parent_ext_{TAG}_e{args.epochs}_r{args.train_rows}_s{args.seed}"

    print(f"[ext-infer] bundle={parent_dir.name}")
    print(f"[ext-infer] quality={args.quality_threshold:.2f} ({qtag}) -> {infer.OUT_DIR.name}")
    rc = infer.main()
    if rc:
        return rc

    # Only the OOS window is extended; VALIDATION stays on the standard 2025-10-01..12-31 frame.
    # The live extended directory reflects that exactly (its validation CSV is 26,496 rows of the
    # unchanged window), so the parent's own validation predictions are the right file to carry
    # over -- the stage-4 replay loads BOTH splits from this one directory.
    import shutil

    src = parent_dir / f"validation_predictions_{qtag}.csv"
    if not src.exists():
        raise SystemExit(f"parent is missing {src.name}; cannot complete the extended dir")
    dst = infer.OUT_DIR / src.name
    shutil.copy2(src, dst)
    print(f"[ext-infer] carried over {src.name} from the parent (validation window is not extended)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
