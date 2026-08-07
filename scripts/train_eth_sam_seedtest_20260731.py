"""Test whether Sharpness-Aware Minimization (SAM, Foret et al. 2020) reduces ETH parent
seed-driven variance vs the plain-AdamW baseline already measured this session (seeds
260620/260728, q0.45, VAL/OOS spread ~11.25pp on OOS).

SAM perturbs weights adversarially (ascent step) before computing the real gradient step
(descent step), seeking flatter minima that are less sensitive to exactly which point training
converges to -- directly targeting the seed-sensitivity problem, unlike richer features/sidecar
gates/capacity changes already tried and refuted this session.

Monkeypatches the ETH parent script's module-level `_fit_expert_omega4` with a SAM-enabled copy
(same architecture/data/hyperparameters, only the optimizer step changes from single-step AdamW
to two-step SAM+AdamW) before invoking its own main() via a faked sys.argv. Everything else
(data loading, eval, backtest) is untouched original code.
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

SEEDS = [260620, 260728]
SAM_RHO = 0.05


class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho: float = 0.05, **kwargs):
        defaults = dict(rho=rho)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self):
        grad_norm = torch.norm(
            torch.stack([p.grad.norm(p=2) for g in self.param_groups for p in g["params"] if p.grad is not None]),
            p=2,
        )
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None:
                    continue
                e_w = p.grad * scale
                p.add_(e_w)
                self.state[p]["e_w"] = e_w

    @torch.no_grad()
    def second_step(self):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.sub_(self.state[p]["e_w"])
        self.base_optimizer.step()


def make_sam_fit_expert_omega4(eth_main):
    parent = eth_main.parent
    hard = eth_main.hard

    def _fit_expert_omega4_sam(
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
        val_idx = np.arange(split, n)
        exit_n = len(y_exit_np)
        exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
        exit_train_idx = np.arange(exit_split)
        exit_val_idx = np.arange(exit_split, exit_n)

        model = parent.ThreeHeadTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
        opt = SAM(model.parameters(), torch.optim.AdamW, rho=SAM_RHO,
                  lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
        ds_dir = TensorDataset(
            torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]),
            torch.from_numpy(y_qual_np[train_idx]), torch.from_numpy(dir_w[train_idx]),
            torch.from_numpy(qual_w[train_idx]),
        )
        ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
        dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
        dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
        best_state = None
        best_loss = float("inf")
        stale = 0
        last_epoch = 0

        def compute_loss(xb, yb, yqb, wb, qwb, xe, ye, we):
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
            return loss_dir + float(parent.CFG.quality_loss_weight) * loss_qual + float(parent.CFG.exit_loss_weight) * loss_exit

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

                loss = compute_loss(xb, yb, yqb, wb, qwb, xe, ye, we)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.first_step()

                opt.zero_grad(set_to_none=True)
                loss2 = compute_loss(xb, yb, yqb, wb, qwb, xe, ye, we)
                loss2.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.second_step()
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
            if vloss + 1.0e-6 < best_loss:
                best_loss = vloss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
                if stale >= int(parent.CFG.patience):
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        payload = {
            "model_id": eth_main.MODEL_ID,
            "expert": hard.EXPERT_NAMES[int(expert_idx)],
            "config": parent.CFG.__dict__,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "scaler": scaler,
            "n_features": int(x_dir_np.shape[1]),
            "best_validation_loss": float(best_loss),
            "epochs_ran": int(last_epoch),
            "input_columns": list(x_dir.columns),
            "quality_target": "omega4_quality_action",
            "direction_class_weights": {str(k): float(v) for k, v in direction_class_weights.items()},
            "quality_class_weights": {str(k): float(v) for k, v in quality_class_weights.items()},
        }
        torch.save(payload, model_path)
        return payload

    return _fit_expert_omega4_sam


def run(seed: int) -> None:
    eth_main = importlib.import_module("train_eval_omega4_3head_parent72_loose_entry_quality_20260620")
    eth_main._fit_expert_omega4 = make_sam_fit_expert_omega4(eth_main)
    sys.argv = ["prog", "--seed", str(seed), "--device", "cuda", "--out-suffix", f"seedtest_{seed}_sam"]
    eth_main.main()


if __name__ == "__main__":
    for s in SEEDS:
        run(s)
