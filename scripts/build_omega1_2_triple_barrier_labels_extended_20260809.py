"""Regenerate the triple-barrier quality labels (h48_conservative etc.) with EVAL_CSV extended to
2026-01-01..06-30, fixing the same Feb-28 cutoff found in the direction labels. Module-level
TRAIN_CSV/EVAL_CSV in the base script are hardcoded (not omega-sourced) but read as plain globals at
call time, so overriding the module attribute before calling main() works the same way."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_omega1_2_triple_barrier_labels_20260619 as tb  # noqa: E402

tb.EVAL_CSV = ROOT / "data/ensemble/supervised/eth_eval_2026_extended_20260809/eth_eval_2026_extended_raw.csv"
tb.OUT_DIR = ROOT / "tmp/triple_barrier_labels_extended_20260809"

if __name__ == "__main__":
    if "--out-dir" not in sys.argv:
        sys.argv += ["--out-dir", str(tb.OUT_DIR)]
    raise SystemExit(tb.main())
