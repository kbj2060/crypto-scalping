#!/usr/bin/env python3
"""B1 diagnostic (Odyssey4 layer/parameter improvement proposal 20260816).

For each of the 3 regime experts (bull/bear/chop) trained inside
scripts/train_eval_omega1_2_tabm_3head_20260603.py's _fit_expert_3head, report:
  - route_w.sum() (effective/soft sample count, since route_w is a continuous
    Regime3 HMM probability, not a hard mask)
  - len(route_w) (raw row count each expert's loss loop iterates over)

No training happens here.

KNOWN BLOCKER (documented, not silently worked around): calling
canon._prepare_frames() literally, as the task first specified, fails on both
dev and server -- hard._build_frame(year) (imported as `hard` in the
canonical script) transitively requires two AI-context feature CSV families
that no longer exist on disk anywhere in this repo:
  - data/ensemble/supervised/omega1_dir3_vsnlstm_full_20260531/training_features_*.csv
  - tmp/causal_regen_20260516/chronos_uncertainty_large_move_20260530/*_chronos.csv
Only the vsnlstm *model checkpoint* (vsnlstm_full.pt + scaler) survives; the
chronos_uncertainty family requires amazon/chronos-t5-tiny, which is not in
the local HF cache (only chronos-2 / chronos-bolt-tiny are). The last
successful run of the canonical script's own report.json artifacts is dated
2026-06-08/06-18 -- i.e. this pipeline has been stale for ~2 months, unrelated
to this task. Regenerating chronos-t5-tiny outputs would need network access
this sandbox does not have; this is out of scope for a "cheap diagnostic".

Bypass used instead: this session already hit and solved the exact same
problem in scripts/research_eth_candidate_faithful_tabm_batchensemble_
cheap_gate_20260816.py's `_prepare_frames_light()` (confirmed there: live
h48qual/zig075 don't consume the LSTM/chronos context chain at all -- the
live feature engine is features/engineering.py's 102 base cols, per
docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md). That
helper substitutes hard._build_frame(year)[["timestamp","zigzag_action"]]
with label_base._add_labels(year) (same label CSV,
tmp/causal_regen_20260516/zigzag_action_labels_20260531/zigzag_action_labels_
<year>.csv, with no LSTM/chronos dependency) and is otherwise byte-identical
to canon._prepare_frames(). This script reuses that exact helper via import
rather than re-deriving a second copy.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import research_eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816 as gate  # noqa: E402

canon = gate.base
hard = gate.hard


def main() -> int:
    frames = gate._prepare_frames_light()
    train_fit_frame = frames["train_raw"]
    route_probs = canon._route_probs(train_fit_frame)

    results = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        route_w = route_probs[:, int(idx)].astype(np.float32)
        results[expert] = {
            "route_w_sum_effective_samples": float(route_w.sum()),
            "len_route_w_raw_count": int(len(route_w)),
            "effective_fraction": float(route_w.sum() / len(route_w)),
        }

    total_raw = len(train_fit_frame)
    hard_route_id = hard._route_id(train_fit_frame)
    hard_counts = {expert: int((hard_route_id == idx).sum()) for idx, expert in enumerate(hard.EXPERT_NAMES)}

    report = {
        "diagnostic": "odyssey4_expert_effective_sample_size_20260816 (Phase B1)",
        "blocker_found_and_bypassed": "see module docstring: canon._prepare_frames() itself is currently broken "
        "(missing vsnlstm/chronos_uncertainty AI-context CSVs, stale ~2 months, unrelated to this task); "
        "this script reproduces train_raw faithfully via a documented bypass instead.",
        "train_fit_frame_rows": int(total_raw),
        "per_expert": results,
        "hard_argmax_route_counts_for_comparison": hard_counts,
        "note": "route_w is Regime3 HMM soft probability per bar (not a hard split); "
        "route_w.sum() is the effective (soft) sample count each expert's weighted "
        "loss actually sees, vs len(route_w) which is the raw row count the "
        "DataLoader/loop iterates over regardless of weight.",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
