"""RESEARCH ONLY -- ablates the exit head's contribution to the shared TabM trunk entirely,
to directly test the user's question: does having an exit head (vs not) matter for ENTRY
(direction/quality) performance?

This is a DIFFERENT manipulation from everything tested earlier in this investigation:
  - Rounds 12/15/16/18/19 changed the exit LABEL (giveback_min, terminal_window) -- the exit
    head still exists and still receives gradient, just trained toward a different target. That
    is what caused entry to swing 10-45 VAL PnL points (round 18), later shown to be dominated by
    plain retraining variance rather than a systematic label effect (round 19).
  - This script instead sets `parent.CFG.exit_loss_weight = 0.0` before training
    (train_eval_omega1_2_tabm_3head_20260603.py:62 default is 1.15; the shared loss is
    `loss_dir + quality_loss_weight*loss_qual + exit_loss_weight*loss_exit`, line 502 of the
    2026-06-20 trainer). At weight 0 the exit loss contributes ZERO gradient to the shared
    encoder AND to the exit head's own weights -- the exit head's linear layer only decays
    toward zero via AdamW's weight_decay, never learns anything. This is the closest reasonable
    operationalization of "no exit head" without editing the model architecture class itself
    (removing the head cleanly would require a code change to ThreeHeadTabM, which is out of
    scope for a research ablation).

Layers on train_eval_omega4_3head_parent72_pinned102_2024tape_20260727.py (same 2024+2025 tape,
same live 102-col contract, same cmamba-warmup-row fix) so the ONLY difference from that script's
control run is exit_loss_weight. Compare three-way: live checkpoint vs pinned102_2024tape_control
(exit_loss_weight=1.15, same data/seed) vs this run (exit_loss_weight=0.0, same data/seed) --
isolates "exit head present vs absent" from both the exit-label axis and the retraining-variance
axis already quantified in round 19.

Does NOT touch trading_bot_modules/, trading_bot.py, runtime_config.py, .env, or any live
checkpoint. Research artifact only.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega4_3head_parent72_pinned102_2024tape_20260727 as tape2024  # noqa: E402

parent = tape2024.parent_script.parent  # train_eval_omega1_2_tabm_3head_20260603 module

# CFG is a frozen dataclass singleton (train_eval_omega1_2_tabm_3head_20260603.py:66); every
# consumer reads it via `parent.CFG.xxx` attribute lookup at call time, so reassigning the module
# attribute to a new instance (rather than mutating the frozen one) is picked up everywhere.
_ORIG_EXIT_LOSS_WEIGHT = parent.CFG.exit_loss_weight
parent.CFG = dataclasses.replace(parent.CFG, exit_loss_weight=0.0)
print(f"[pinned102_2024tape_noexithead] exit_loss_weight {_ORIG_EXIT_LOSS_WEIGHT} -> "
      f"{parent.CFG.exit_loss_weight} (exit head receives zero training gradient)", flush=True)

if __name__ == "__main__":
    raise SystemExit(tape2024.main())
