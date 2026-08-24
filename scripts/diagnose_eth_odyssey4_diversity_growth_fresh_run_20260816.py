#!/usr/bin/env python3
"""One fresh canonical-config (epochs=28 default, patience=8) _fit_expert_3head run for a single
expert, to check epochs_ran and BatchEnsemble diversity-parameter growth under REAL production
training conditions (as opposed to the --epochs 2 quick-test bundles found on disk, which turned out
to be an artificially capped budget, not genuine early stopping -- see
docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as base3head  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

SEED = 260603
EXPERT_IDX = 0  # bull


def main() -> int:
    device = torch.device("cpu")
    # _prepare_frames() hits the dead vsnlstm/chronos CSV chain via hard._build_frame() just to
    # fetch zigzag_action -- use the already-recovered dependency-free true-feature pipeline instead
    # (same real 102(+13pos)=115 live feature contract, confirmed working 2026-08-16).
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = base3head._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)

    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"],
        frames["s_train_label"],
        frames["train_fixed"],
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        exit_edge_min=0.0020,
        hold_offsets=[1, 2, 3, 6, 12, 24, 48, 96, 192, 384],
        max_samples=60000,
    )
    x_exit = base3head._exit_input_from_position_rows(x_exit_raw, base_cols)

    print(f"[diag] x_train shape={x_train.shape} x_exit shape={x_exit.shape}", file=sys.stderr)

    init_scale_std = None

    # Capture a fresh model's init-time diversity spread for comparison (same init distribution the
    # real training call below will start from).
    torch.manual_seed(int(SEED) + int(EXPERT_IDX))
    tmp_model = base3head.ThreeHeadTabM(x_train.shape[1] + len(base3head.POS_COLS), cfg=base3head.CFG)
    init_scale_std = tmp_model.input_scale.detach().std(dim=0).mean().item()
    print(f"[diag] fresh-init input_scale std_across_k={init_scale_std:.5f} (should match ~0.03 init spec)", file=sys.stderr)

    model_path = ROOT / "tmp/diag_fresh_run_20260816" / "bull_diag.pt"
    payload = base3head._fit_expert_3head(
        x_train,
        y_train,
        train_raw,
        x_exit,
        y_exit,
        frame_exit,
        expert_idx=EXPERT_IDX,
        seed=SEED,
        epochs=28,
        device=device,
        model_path=model_path,
    )

    sd = payload["state_dict"]
    isc = sd["input_scale"]
    print(f"[diag] === RESULT (expert={hard.EXPERT_NAMES[EXPERT_IDX]}, epochs budget=28, patience=8) ===")
    print(f"[diag] epochs_ran={payload['epochs_ran']} best_validation_loss={payload['best_validation_loss']:.5f}")
    print(
        f"[diag] trained input_scale std_across_k(mean over features)={isc.std(dim=0).mean().item():.5f} "
        f"(vs fresh-init {init_scale_std:.5f}) mean={isc.mean().item():.5f}"
    )
    for lidx in range(2):
        key = f"expert_scale.{lidx}"
        if key in sd:
            es = sd[key]
            print(f"[diag] trained expert_scale[{lidx}] std_across_k={es.std(dim=0).mean().item():.5f} mean={es.mean().item():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
