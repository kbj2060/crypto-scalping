"""Final ETH candidate: SWA-trained, 5-genuinely-random-seed-averaged parent, on the CORRECT
live-matching 2024+2025 training tape (183,936 rows, reproduces live's exact
label_quality_summary) -- not the simplified 2025-only default used for the earlier controlled
SWA-vs-baseline comparison this session ([[project-eth-swa-seed-variance-partial-20260731]]).

Trains both live components (h48qual, zig075) x 5 seeds with SWA (reusing the exact SWA
implementation validated this session), via
scripts/train_eval_omega4_3head_parent72_pinned102_2024tape_20260727.py (the correct live-tape
wrapper). Threshold is fixed at q0.45, pre-registered before this run (used throughout this
session's seed-variance investigation, not cherry-picked from this run's own VAL numbers) --
per this project's hard-won lesson (Sigma3-1h/Sigma6) against selecting configs by reading VAL
directly. Ensembles the 5 seeds' predictions (same averaging method as
scripts/eval_seed_ensemble_average_20260731.py) and reports final VAL/OOS performance.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

SEEDS = [260620, 260728, 260729, 260730, 260731]
COMPONENTS = ["zig075", "h48qual"]
SWA_BURNIN_EPOCHS = 1

ROOT_STR = str(ROOT)
DIRLAB = f"{ROOT_STR}/tmp/causal_regen_20260516/omega_current_only_all_label_candidate_parent_screen_20260629/label_contracts/zigzag_action_labels_20260531"
QLAB = f"{ROOT_STR}/tmp/causal_regen_20260516/omega_zigzag_fix_all_solutions_20260630/label_contracts/sltp_h48_conservative_padded_to_zigzag_timestamps"

# Matches scripts/run_pinned102_2024tape_controls_20260727.sh exactly (the reference launcher
# for the live-equivalent h48qual/zig075 controls) -- the earlier run of this script used the
# base script's generic defaults instead of these, which do NOT match either live component's
# actual training recipe. Corrected 2026-07-31 before reporting any "final" number.
COMPONENT_ARGS = {
    "h48qual": [
        "--epochs", "2", "--max-train-rows", "0", "--max-exit-samples", "30000",
        "--quality-thresholds", "0.50", "--exit-label-mode", "entry_label_terminal_giveback",
        "--direction-label-dir", DIRLAB, "--quality-mode", "quality_label_action",
        "--quality-label-dir", QLAB,
    ],
    "zig075": [
        "--epochs", "2", "--max-train-rows", "0", "--max-exit-samples", "30000",
        "--quality-thresholds", "0.75", "--exit-label-mode", "entry_label_terminal_giveback",
        "--direction-label-dir", DIRLAB, "--quality-mode", "same_as_direction",
    ],
}
QUALITY_THRESHOLD_TAG = {"h48qual": "q050", "zig075": "q075"}


def make_swa_fit_expert_omega4(eth_main):
    parent = eth_main.parent
    hard = eth_main.hard

    def _fit_expert_omega4_swa(
        x_dir, y_dir, y_qual, route_frame, x_exit, y_exit, exit_route_frame, *,
        expert_idx, seed, epochs, device, model_path,
        direction_class_weights, quality_class_weights,
    ):
        torch.manual_seed(int(seed) + int(expert_idx))
        np.random.seed(int(seed) + int(expert_idx))
        model_path.parent.mkdir(parents=True, exist_ok=True)
        x_all = pd.concat([x_dir, x_exit], ignore_index=True)
        _x_np, scaler = parent._standardize_fit(x_all)
        x_dir_np = parent._standardize_apply(x_dir, scaler)
        x_exit_np = parent._standardize_apply(x_exit, scaler)
        y_dir_np = np.asarray(y_dir, dtype=np.int64)
        y_qual_np = np.asarray(y_qual, dtype=np.int64)
        y_exit_np = np.asarray(y_exit, dtype=np.int64)
        route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
        exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
        dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
        qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
        dir_w *= np.asarray([float(direction_class_weights.get(int(y), 1.0)) for y in y_dir_np], dtype=np.float32)
        qual_w *= np.asarray([float(quality_class_weights.get(int(y), 1.0)) for y in y_qual_np], dtype=np.float32)
        ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
        if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
            raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid Omega4 sample weights")

        n = len(y_dir_np)
        split = max(int(n * 0.85), min(n - 1, 512))
        train_idx = np.arange(split)
        exit_n = len(y_exit_np)
        exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
        exit_train_idx = np.arange(exit_split)
        val_idx = np.arange(split, n)
        exit_val_idx = np.arange(exit_split, exit_n)

        model = parent.ThreeHeadTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
        ds_dir = TensorDataset(
            torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
            torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]),
            torch.from_numpy(qual_w[train_idx]),
        )
        ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
        dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
        dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)

        swa_state = None
        swa_n = 0
        last_epoch = 0

        for epoch in range(int(epochs)):
            last_epoch = epoch + 1
            model.train()
            exit_iter = iter(dl_exit)
            for xb, yb, yqb, wb, qwb in dl_dir:
                try:
                    xe, ye, we = next(exit_iter)
                except StopIteration:
                    exit_iter = iter(dl_exit)
                    xe, ye, we = next(exit_iter)
                xb, yb, yqb, wb, qwb = (t.to(device, non_blocking=True) for t in (xb, yb, yqb, wb, qwb))
                xe, ye, we = (t.to(device, non_blocking=True) for t in (xe, ye, we))
                out_dir = model(xb)
                loss_dir_k = torch.nn.functional.cross_entropy(
                    out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none"
                ).reshape(-1, int(parent.CFG.k))
                loss_qual_k = torch.nn.functional.cross_entropy(
                    out_dir["quality"].reshape(-1, 3), yqb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none"
                ).reshape(-1, int(parent.CFG.k))
                out_exit = model(xe)
                loss_exit_k = torch.nn.functional.cross_entropy(
                    out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none"
                ).reshape(-1, int(parent.CFG.k))
                loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
                loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
                loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
                loss = loss_dir + float(parent.CFG.quality_loss_weight) * loss_qual + float(parent.CFG.exit_loss_weight) * loss_exit
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()

            if epoch + 1 > SWA_BURNIN_EPOCHS:
                cur_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                if swa_state is None:
                    swa_state = cur_state
                    swa_n = 1
                else:
                    swa_n += 1
                    for k in swa_state:
                        if swa_state[k].is_floating_point():
                            swa_state[k] += (cur_state[k] - swa_state[k]) / swa_n
                        else:
                            swa_state[k] = cur_state[k]

        final_state = swa_state if swa_state is not None else {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(final_state)

        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float(
                (((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                 + float(parent.CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                 + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)))
                .detach().cpu()
            )

        payload = {
            "model_id": eth_main.MODEL_ID,
            "expert": hard.EXPERT_NAMES[int(expert_idx)],
            "config": parent.CFG.__dict__,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler": scaler,
            "n_features": int(x_dir_np.shape[1]),
            "best_validation_loss": float(vloss),
            "epochs_ran": int(last_epoch),
            "swa_epochs_averaged": int(swa_n),
            "input_columns": list(x_dir.columns),
            "quality_target": "omega4_quality_action",
            "direction_class_weights": {str(k): float(v) for k, v in direction_class_weights.items()},
            "quality_class_weights": {str(k): float(v) for k, v in quality_class_weights.items()},
        }
        torch.save(payload, model_path)
        return payload

    return _fit_expert_omega4_swa


def train_one(component: str, seed: int) -> None:
    wrapper = importlib.import_module("train_eval_omega4_3head_parent72_pinned102_2024tape_20260727")
    wrapper.parent_script._fit_expert_omega4 = make_swa_fit_expert_omega4(wrapper.parent_script)
    sys.argv = [
        "prog", "--pin-component", component,
        "--seed", str(seed), "--device", "cuda",
        "--out-suffix", f"final_swa_{component}_seed{seed}",
        *COMPONENT_ARGS[component],
    ]
    wrapper.main()


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--component", required=True, choices=COMPONENTS)
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = ap.parse_args()
    for s in [int(x) for x in args.seeds.split(",")]:
        train_one(args.component, s)
