"""Test whether reducing TabM capacity (k=8->2, hidden=192->64) reduces seed-driven variance
for SOL's Omega4.6.1 parent, vs the full-capacity baseline already measured at q0.45/30k-row
training (VAL/OOS spread already on file from prior seed tests this session).

Monkeypatches the frozen parent.CFG dataclass before invoking the SOL training script's own
main() via a faked sys.argv -- reuses the exact same training/eval pipeline unmodified, only the
architecture config changes.
"""
from __future__ import annotations

import dataclasses
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [260620, 260728]


def run(seed: int) -> None:
    sol_main = importlib.import_module("train_eval_omega4_3head_parent72_loose_entry_quality_sol_20260707")
    sol_main.parent.CFG = dataclasses.replace(sol_main.parent.CFG, k=2, hidden=64)
    sys.argv = [
        "prog", "--seed", str(seed), "--device", "cuda",
        "--out-suffix", f"seedtest_{seed}_reducedcap",
    ]
    sol_main.main()


if __name__ == "__main__":
    for s in SEEDS:
        run(s)
