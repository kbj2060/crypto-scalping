#!/usr/bin/env python3
"""RESEARCH ONLY -- cheap_gate Step A for the faithful-TabM candidate (docs/model_contracts/
eth_candidate_faithful_tabm_batchensemble_contract_20260816.md). The live h48qual/zig075
`ThreeHeadTabM` (scripts/train_eval_omega1_2_tabm_3head_20260603.py) only implements the R
(pre-multiply) BatchEnsemble adapter at every layer, missing the S (post-multiply) adapter and
per-layer bias B the TabM paper's default variant uses (arXiv:2410.24210:
`l_BE(X) = ((X (X-hadamard) R) W) (X-hadamard) S + B`, applied at every linear layer). This script
defines `ThreeHeadTabMFull` completing R+S+B at every layer (residual connections kept unchanged --
out of scope for this ablation; numerical embeddings deferred to Step B) and trains it against the
existing architecture on IDENTICAL data/seed/hyperparameters, comparing held-out validation
classification metrics only (no backtest yet) -- the contract's cheap_gate requirement.

Reuses `train_eval_omega1_2_tabm_3head_20260603`'s own data pipeline (`_base_input`,
`_standardize_fit/_apply`, `_route_probs`, exit-dataset construction) unmodified -- only the model
class, a copy of `_fit_expert_3head` (parameterized by model class instead of hardcoded to
`ThreeHeadTabM`), and `_prepare_frames_light` (see its docstring: swaps the zigzag_action label
fetch to bypass a broken, irrelevant LSTM/chronos context-feature dependency that base._prepare_
frames pulls in but Odyssey's live feature engine never consumes) are new. This does NOT reproduce the exact historical derived recipe
that trained the currently-deployed h48qual bundle (which has since evolved through further
zigzag/quality-relabeling scripts) -- it is a self-consistent architecture-only ablation on this
script's own base zigzag_action-label pipeline, which is a valid apples-to-apples comparison for
isolating the architecture variable, not a claim of bit-for-bit live-bundle reproduction.

Single seed only (cheap_gate stage) -- N>=5 seed reproduction is required before any adoption claim
per this repo's seed-diversity policy, and is NOT satisfied by this script.

fresh_forward_bar_by_bar=n/a (this trains a classifier, does not backtest). No portfolio ledger
touched. No GPU (CPU-only dev machine, verified via torch.cuda.is_available()==False).

Does NOT modify train_eval_omega1_2_tabm_3head_20260603.py -- only imports its functions.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as base  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_omega1_direction_head_direction_only_20260602 as label_base  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_cheap_gate_20260816"
CFG = base.CFG
SEED = 260816
EPOCHS = 28


def log(msg: str) -> None:
    print(f"[faithful_tabm_cheap_gate] {msg}", flush=True)


class ThreeHeadTabMFull(nn.Module):
    """Completes the BatchEnsemble adapters the live ThreeHeadTabM only has R for: adds S
    (post-multiply) and per-layer bias B at every layer, matching arXiv:2410.24210's
    l_BE(X) = ((X (had) R) W) (had) S + B. Residual connections kept (out of scope here)."""

    def __init__(self, n_features: int, *, cfg=CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        hidden = int(cfg.hidden)
        n_blocks = max(0, int(cfg.layers) - 1)
        self.input_r = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.in_proj = nn.Linear(self.n_features, hidden)
        self.input_s = nn.Parameter(torch.randn(self.k, hidden) * 0.03 + 1.0)
        self.input_b = nn.Parameter(torch.zeros(self.k, hidden))
        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_blocks))
        self.block_r = nn.ParameterList(nn.Parameter(torch.randn(self.k, hidden) * 0.03 + 1.0) for _ in range(n_blocks))
        self.block_s = nn.ParameterList(nn.Parameter(torch.randn(self.k, hidden) * 0.03 + 1.0) for _ in range(n_blocks))
        self.block_b = nn.ParameterList(nn.Parameter(torch.zeros(self.k, hidden)) for _ in range(n_blocks))
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(int(cfg.layers)))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 3)
        self.exit_head = nn.Linear(hidden, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_r.unsqueeze(0)
        h = self.in_proj(xk) * self.input_s.unsqueeze(0) + self.input_b.unsqueeze(0)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            pre = h * self.block_r[idx].unsqueeze(0)
            h = layer(pre) * self.block_s[idx].unsqueeze(0) + self.block_b[idx].unsqueeze(0)
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {"direction": self.direction_head(h), "quality": self.quality_head(h), "exit": self.exit_head(h)}


def _fit_one(
    model_cls,
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
) -> dict[str, Any]:
    """Copy of base._fit_expert_3head, parameterized by model_cls, returning validation metrics
    instead of saving a bundle to disk (cheap_gate -- no artifact persistence needed yet)."""
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = base._standardize_fit(x_all)
    x_dir_np = base._standardize_apply(x_dir, scaler)
    x_exit_np = base._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = base._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = base._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = dir_w.copy()
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 3-head sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = model_cls(x_dir_np.shape[1], cfg=CFG).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state = None
    best_loss = float("inf")
    best_components = None
    stale = 0
    last_epoch = 0
    t0 = time.time()
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, wb = xb.to(device), yb.to(device), wb.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            loss_qual_k = torch.nn.functional.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(CFG.quality_loss_weight) * vqual_loss + float(CFG.exit_loss_weight) * vex_loss
            dir_pred_k = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred_k))
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k2: v.detach().cpu().clone() for k2, v in model.state_dict().items()}
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": bacc}
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    return {
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "n_params": int(n_params),
        "epochs_ran": int(last_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "train_seconds": round(time.time() - t0, 1),
    }


def _prepare_frames_light() -> dict[str, Any]:
    """Copy of base._prepare_frames(disable_tp_sl=False) with ONE substitution: the zigzag_action
    label fetch. base._prepare_frames calls hard._build_frame(year)[["timestamp","zigzag_action"]],
    which internally rebuilds a much heavier context-feature chain (volpca/context_groups/tsfm_
    chronos, including an LSTM/chronos "vsnlstm" feature family) that Odyssey's live h48qual/zig075
    do NOT consume (confirmed: the live feature engine is features/engineering.py's 102 base cols,
    per docs/model_contracts/odyssey4_eth_full_stack_architecture_20260814.md) -- that chain's own
    intermediate CSV is missing on this dev machine and unrelated to the label itself. The label's
    true, standalone source is label_base._add_labels(year), which reads tmp/causal_regen_20260516/
    zigzag_action_labels_20260531/zigzag_action_labels_<year>.csv directly -- exactly the two
    columns base._prepare_frames actually uses, with no LSTM/chronos dependency. Everything else
    below is unchanged from base._prepare_frames."""
    base.omega.BASE_TEMPLATE["max_hold"] = 0
    base.omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = base.omega._load_omega_frames()
    feature_cols = base.omega._numeric_feature_cols(train_all, eval_df)
    label_2025 = label_base._add_labels(2025)
    label_2026 = label_base._add_labels(2026)
    train_all, train_labels = base.omega._align(train_all, label_2025, "omega train labels")
    eval_df, eval_labels = base.omega._align(eval_df, label_2026, "omega oos labels")
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    train_raw = train_all[train_all["timestamp"] < base.SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= base.SPLIT_TS].reset_index(drop=True)

    tabm_2025 = base.omega._read(base.omega.TABM_2025)
    train_df, train_src = base.omega._align(train_raw, tabm_2025, "train")
    train_fixed = base.omega._to_fixed_decisions(train_src, oof=True)
    s_train_label = base._base_input(train_df, feature_cols)
    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "train_df": train_df,
        "train_fixed": train_fixed,
        "s_train_label": s_train_label,
        "feature_cols": feature_cols,
        "overlay_report": overlay_report,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log("=== stage=prepare_frames (light -- bypasses the LSTM/chronos context chain Odyssey doesn't use) ===")
    frames = _prepare_frames_light()
    fee, slip = base.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = base._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=0,
    )
    x_exit = base._exit_input_from_position_rows(x_exit_raw, base_cols)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {"design": "faithful-TabM Step A cheap_gate -- BatchEnsemble R+S+B completion, single seed, classification metrics only.", "seed": SEED, "epochs_budget": EPOCHS, "architectures": {}}

    for arch_name, model_cls in (("baseline_R_only", base.ThreeHeadTabM), ("full_R_S_B", ThreeHeadTabMFull)):
        log(f"=== stage=train architecture={arch_name} ===")
        expert_results = []
        for idx, expert in enumerate(hard.EXPERT_NAMES):
            res = _fit_one(model_cls, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=idx, seed=SEED, epochs=EPOCHS, device=device)
            expert_results.append(res)
            log(f"  {arch_name} {expert}: n_params={res['n_params']} epochs_ran={res['epochs_ran']} "
                f"best_val_loss={res['best_validation_loss']:.4f} dir_bacc={res['best_components']['direction_balanced_accuracy']:.4f} "
                f"({res['train_seconds']}s)")
        report["architectures"][arch_name] = expert_results

    log("=== stage=summary ===")
    for expert in hard.EXPERT_NAMES:
        b = next(r for r in report["architectures"]["baseline_R_only"] if r["expert"] == expert)
        f = next(r for r in report["architectures"]["full_R_S_B"] if r["expert"] == expert)
        log(f"  {expert:6s}: baseline val_loss={b['best_validation_loss']:.4f} bacc={b['best_components']['direction_balanced_accuracy']:.4f}  "
            f"|  full val_loss={f['best_validation_loss']:.4f} bacc={f['best_components']['direction_balanced_accuracy']:.4f}  "
            f"|  delta_val_loss={f['best_validation_loss']-b['best_validation_loss']:+.4f} delta_bacc={f['best_components']['direction_balanced_accuracy']-b['best_components']['direction_balanced_accuracy']:+.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
