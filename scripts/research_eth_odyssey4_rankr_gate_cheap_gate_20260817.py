#!/usr/bin/env python3
"""RESEARCH ONLY -- implementation + cheap_gate for the rank-r BatchEnsemble gate candidate.

Follow-up to docs/experiments/eth_odyssey4_dl_reference_deep_analysis_20260816.md Section 6
(LoMETab, arXiv:2605.14365): the live ThreeHeadTabM gates (`input_scale`/`expert_scale`) are
R-only rank-1 BatchEnsemble (canonical formula `((X⊙R)W)⊙S+B`, S omitted -- confirmed by
`eth_candidate_faithful_tabm_batchensemble_20260816.md`). This means the effective per-member
weight is `W_k = W ⊙ (r_k ⊗ \U0001D7D9)` -- every output unit gets the SAME scalar-per-input-row
scaling, so widening the init spread of `r_k` (already tried, `eth_odyssey4_dl_reference_deep_
analysis_20260816.md` §2.4b) cannot enlarge the hypothesis class, only the magnitude of an
already-narrow perturbation family. LoMETab's rank-r generalization `W_k = W ⊙ (A_kB_k^T)`
(A_k: in x r, B_k: out x r) is proven to strictly enlarge the hypothesis class for r>=2.

Implementation note (deviation from the paper, disclosed): the paper's exact parameterization is
`1 + A_kB_k^T` (identity-residual). This script implements the simpler bare `A_kB_k^T` (summed
Hadamard-rank-r term, no separate shared identity term) because it lets rank=1 with B fixed at 1
reproduce the CURRENT live architecture exactly (A[:,:,0]=scale_k, B[:,:,0]=1 gives back
`(x⊙scale_k)@W` bit-for-bit) -- required for the mandatory sanity check below. The "rank-r
strictly generalizes rank-1" property is unaffected by this reparameterization choice: rank-1 is
still nested inside rank-r either way (extra columns are additive corrections on top of column 0).

Protocol matches the existing cheap_gates in this line exactly: single seed (260816), expert=bull,
plain CE, true 102(+13pos)=115-feature live pipeline
(`eth_odyssey4_true_feature_pipeline_20260816.py`), current live k=8 and same_as_direction quality
target held fixed. r=1 (B frozen at 1, i.e. bit-identical to the current architecture) is the
control point in the sweep, not just the sanity-check condition -- this isolates "does moving
beyond rank-1" from "does learning S at rank-1 help" (a narrower version of the already-CLOSED
R+S+B-completion axis, deliberately not re-tested here). r=2/3 have both A and B fully learned,
with the extra columns (c>=1) initialized near zero so training starts close to the r=1 baseline.
Also re-runs the §2.1 member-collapse diagnostic (pairwise top-confidence correlation, argmax
unanimity) at each best checkpoint, matching this document's own methodology for separating "did
diversity actually change" from "did accuracy follow."
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
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
from sklearn.utils.class_weight import compute_sample_weight
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eth_odyssey4_true_feature_pipeline_20260816 as truepipe  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as canon  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_odyssey4_rankr_gate_cheap_gate_20260817"
EXPERT = "bull"
SEED = 260816
EPOCHS = 28
RANKS = [1, 2, 3]  # r=1 == current live architecture (B frozen at 1); control point, not just sanity check


# =====================================================================================================
# Rank-r gate + model
# =====================================================================================================


class RankRGate(nn.Module):
    """Sum_{c=1..r} (x_k ⊙ A[:,:,c]) @ W.T ⊙ B[:,:,c] for a shared weight matrix W (out,in),
    across k ensemble members. At r=1 with B fixed to 1, reproduces `x_k ⊙ scale_k` then `@ W.T`
    exactly (the current live R-only rank-1 mechanism)."""

    def __init__(self, k: int, in_dim: int, out_dim: int, rank: int, *, fixed_b_at_r1: bool, seed_offset: int) -> None:
        super().__init__()
        self.k = int(k)
        self.rank = int(rank)
        g = torch.Generator().manual_seed(int(seed_offset))
        a0 = torch.randn(k, in_dim, 1, generator=g) * 0.03 + 1.0
        if rank > 1:
            a_extra = torch.randn(k, in_dim, rank - 1, generator=g) * 0.03
            a_init = torch.cat([a0, a_extra], dim=-1)
        else:
            a_init = a0
        self.A = nn.Parameter(a_init)

        b0 = torch.ones(k, out_dim, 1)
        if rank > 1:
            b_extra = torch.randn(k, out_dim, rank - 1, generator=g) * 0.03
            b_init = torch.cat([b0, b_extra], dim=-1)
        else:
            b_init = b0
        if rank == 1 and fixed_b_at_r1:
            self.register_buffer("B", b_init)
        else:
            self.B = nn.Parameter(b_init)

    def forward(self, x_k: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        # x_k: (batch,k,in_dim), weight: (out_dim,in_dim) [nn.Linear.weight convention]
        xa = x_k.unsqueeze(-1) * self.A.unsqueeze(0)  # (batch,k,in_dim,r)
        xa = xa.permute(0, 1, 3, 2)  # (batch,k,r,in_dim)
        proj = torch.matmul(xa, weight.t())  # (batch,k,r,out_dim)
        gated = proj * self.B.unsqueeze(0).permute(0, 1, 3, 2)  # (batch,k,r,out_dim)
        return gated.sum(dim=2)  # (batch,k,out_dim)


class RankRThreeHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: canon.ThreeHeadConfig, rank: int, fixed_b_at_r1: bool = True) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.rank = int(rank)
        self.n_features = int(n_features)
        hidden = int(cfg.hidden)
        n_blocks = max(0, int(cfg.layers) - 1)

        self.input_bias = nn.Parameter(torch.zeros(self.k, n_features))  # unchanged additive term (matches canon)
        self.in_proj = nn.Linear(n_features, hidden)
        self.input_gate = RankRGate(self.k, n_features, hidden, self.rank, fixed_b_at_r1=fixed_b_at_r1, seed_offset=9001)

        self.blocks = nn.ModuleList(nn.Linear(hidden, hidden) for _ in range(n_blocks))
        self.block_gates = nn.ModuleList(
            RankRGate(self.k, hidden, hidden, self.rank, fixed_b_at_r1=fixed_b_at_r1, seed_offset=9101 + idx)
            for idx in range(n_blocks)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(hidden) for _ in range(int(cfg.layers)))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 3)
        self.exit_head = nn.Linear(hidden, 2)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_k = x.unsqueeze(1).expand(-1, self.k, -1)  # (batch,k,n_features)
        gate_out = self.input_gate(x_k, self.in_proj.weight)  # (batch,k,hidden)
        bias_contrib = self.input_bias @ self.in_proj.weight.t()  # (k,hidden)
        h = gate_out + bias_contrib.unsqueeze(0) + self.in_proj.bias.view(1, 1, -1)
        h = self.dropout(F.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            gate_out2 = self.block_gates[idx](h, layer.weight)
            h2 = gate_out2 + layer.bias.view(1, 1, -1)
            h2 = self.dropout(F.silu(self.norms[idx + 1](h2)))
            h = h2 + residual
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {"direction": self.direction_head(h), "quality": self.quality_head(h), "exit": self.exit_head(h)}


def log(msg: str) -> None:
    print(f"[rankr_gate] {msg}", flush=True)


# =====================================================================================================
# Sanity check -- r=1 (B frozen) must reproduce canon.ThreeHeadTabM bit-for-bit
# =====================================================================================================


def sanity_check(n_features: int, device: torch.device) -> bool:
    torch.manual_seed(SEED)
    cfg = canon.ThreeHeadConfig()
    canon_model = canon.ThreeHeadTabM(n_features, cfg=cfg).to(device).eval()
    rankr_model = RankRThreeHeadTabM(n_features, cfg=cfg, rank=1, fixed_b_at_r1=True).to(device).eval()

    with torch.no_grad():
        rankr_model.input_gate.A[:, :, 0].copy_(canon_model.input_scale)
        rankr_model.input_bias.copy_(canon_model.input_bias)
        rankr_model.in_proj.weight.copy_(canon_model.in_proj.weight)
        rankr_model.in_proj.bias.copy_(canon_model.in_proj.bias)
        for idx in range(len(canon_model.blocks)):
            rankr_model.block_gates[idx].A[:, :, 0].copy_(canon_model.expert_scale[idx])
            rankr_model.blocks[idx].weight.copy_(canon_model.blocks[idx].weight)
            rankr_model.blocks[idx].bias.copy_(canon_model.blocks[idx].bias)
        for idx in range(len(canon_model.norms)):
            rankr_model.norms[idx].weight.copy_(canon_model.norms[idx].weight)
            rankr_model.norms[idx].bias.copy_(canon_model.norms[idx].bias)
        rankr_model.direction_head.weight.copy_(canon_model.direction_head.weight)
        rankr_model.direction_head.bias.copy_(canon_model.direction_head.bias)
        rankr_model.quality_head.weight.copy_(canon_model.quality_head.weight)
        rankr_model.quality_head.bias.copy_(canon_model.quality_head.bias)
        rankr_model.exit_head.weight.copy_(canon_model.exit_head.weight)
        rankr_model.exit_head.bias.copy_(canon_model.exit_head.bias)

    g = torch.Generator().manual_seed(4242)
    x = torch.randn(37, n_features, generator=g).to(device)
    with torch.no_grad():
        # dropout is stochastic -- disable it for the equivalence check (both models .eval() already
        # sets dropout to identity, but assert explicitly since this check is meaningless otherwise)
        assert not canon_model.training and not rankr_model.training
        out_canon = canon_model(x)
        out_rankr = rankr_model(x)

    ok = True
    for key in ("direction", "quality", "exit"):
        diff = (out_canon[key] - out_rankr[key]).abs().max().item()
        match = torch.allclose(out_canon[key], out_rankr[key], atol=1e-5, rtol=1e-4)
        log(f"  sanity[{key}]: max_abs_diff={diff:.3e} allclose={match}")
        ok = ok and match
    return ok


# =====================================================================================================
# Collapse diagnostic (same as §2.1 of the reference doc / wide_init cheap_gate)
# =====================================================================================================


def _collapse_stats(logits_k: torch.Tensor) -> dict[str, float]:
    probs_k = torch.softmax(logits_k, dim=-1)
    n, k, c = probs_k.shape
    top_prob = probs_k.max(dim=-1).values
    top_np = top_prob.detach().cpu().numpy()
    corr = np.corrcoef(top_np.T)
    iu = np.triu_indices(k, k=1)
    pred_np = probs_k.argmax(dim=-1).detach().cpu().numpy()
    unanimous = (pred_np == pred_np[:, [0]]).all(axis=1).mean()
    class_freq = np.stack([(pred_np == cls).mean(axis=0) for cls in range(c)], axis=0)
    mean_class_freq = class_freq.mean(axis=1)
    independent_unanimity = float((mean_class_freq**k).sum())
    return {
        "mean_pairwise_corr_of_top_confidence": float(np.mean(corr[iu])),
        "argmax_unanimity_rate": float(unanimous),
        "independent_baseline_unanimity_rate": independent_unanimity,
    }


# =====================================================================================================
# Training loop -- mirrors canon._fit_expert_3head / the wide_init cheap_gate's _fit_with_init_std
# =====================================================================================================


def _fit_rankr(x_dir, y_dir, route_frame, x_exit, y_exit, exit_route_frame, *, expert_idx: int, seed: int, epochs: int, device: torch.device, rank: int) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx) + int(rank) * 137)
    np.random.seed(int(seed) + int(expert_idx) + int(rank) * 137)
    cfg = canon.CFG
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = canon._standardize_fit(x_all)
    x_dir_np = canon._standardize_apply(x_dir, scaler)
    x_exit_np = canon._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = canon._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = canon._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = RankRThreeHeadTabM(x_dir_np.shape[1], cfg=cfg, rank=rank, fixed_b_at_r1=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_loss = float("inf")
    best_components = None
    best_epoch = 0
    best_collapse: dict[str, float] | None = None
    stale = 0
    last_epoch = 0
    curve: list[dict[str, Any]] = []
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
            loss_dir_k = F.cross_entropy(out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_qual_k = F.cross_entropy(out_dir["quality"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = F.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(cfg.quality_loss_weight) * loss_qual + float(cfg.exit_loss_weight) * loss_exit
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
            vdir = F.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = F.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = F.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vdir_loss = float(((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vqual_loss = float(((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
            vex_loss = float(((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0)).detach().cpu())
            vloss = vdir_loss + float(cfg.quality_loss_weight) * vqual_loss + float(cfg.exit_loss_weight) * vex_loss
            dir_pred = torch.softmax(vo["direction"], dim=-1).mean(dim=1).argmax(dim=-1).cpu().numpy()
            dir_bacc = float(balanced_accuracy_score(vy.cpu().numpy(), dir_pred))
            collapse = _collapse_stats(vo["direction"])
        curve.append({"epoch": epoch + 1, "val_loss": round(vloss, 5), "direction_bacc": round(dir_bacc, 5)})
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_components = {"direction_val_loss": vdir_loss, "quality_val_loss": vqual_loss, "exit_val_loss": vex_loss, "direction_balanced_accuracy": dir_bacc}
            best_epoch = epoch + 1
            best_collapse = collapse
            stale = 0
        else:
            stale += 1
            if stale >= int(cfg.patience):
                break
    return {
        "rank": int(rank),
        "n_params": int(n_params),
        "epochs_ran": int(last_epoch),
        "early_stop_epoch": int(best_epoch),
        "best_validation_loss": float(best_loss),
        "best_components": best_components,
        "collapse_at_best_checkpoint": best_collapse,
        "curve": curve,
        "train_seconds": round(time.time() - t0, 1),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = canon._device("cpu")
    canon._seed_everything(SEED)

    log(f"=== stage=prepare_frames (true 115-feature pipeline) expert={EXPERT} seed={SEED} ranks={RANKS} ===")
    frames = truepipe.prepare_frames_true(disable_tp_sl=False)
    fee, slip = canon.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    n_features = len(base_cols) + len(canon.POS_COLS)
    log(f"  n_features_expected={n_features} (base={len(base_cols)}+pos={len(canon.POS_COLS)})")

    log("=== stage=sanity_check (r=1, B frozen, must reproduce canon.ThreeHeadTabM bit-for-bit) ===")
    ok = sanity_check(n_features, device)
    if not ok:
        log("  SANITY CHECK FAILED -- aborting before spending any training compute.")
        report = {"sanity_check_passed": False}
        (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return 1
    log("  sanity check PASSED -- rank-1 (B frozen) forward output matches canon.ThreeHeadTabM exactly.")

    train_raw = frames["train_raw"]
    x_train = canon._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=60000,
    )
    x_exit = canon._exit_input_from_position_rows(x_exit_raw, base_cols)
    expert_idx = list(hard.EXPERT_NAMES).index(EXPERT)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "Rank-r BatchEnsemble gate cheap_gate (LoMETab, arXiv:2605.14365) -- r in {1(=current live, B frozen),2,3}, k=8 and same_as_direction quality held fixed, single seed/expert, plain-CE, true 115-feature pipeline.",
        "seed": SEED,
        "expert": EXPERT,
        "epochs_budget": EPOCHS,
        "ranks_tested": RANKS,
        "sanity_check_passed": True,
        "runs": {},
    }
    for rank in RANKS:
        res = _fit_rankr(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=expert_idx, seed=SEED, epochs=EPOCHS, device=device, rank=rank)
        report["runs"][f"rank_{rank}"] = res
        bc = res["best_components"]
        cs = res["collapse_at_best_checkpoint"]
        log(
            f"  rank={rank} n_params={res['n_params']} early_stop_epoch={res['early_stop_epoch']} best_val_loss={res['best_validation_loss']:.4f} "
            f"dir_bacc={bc['direction_balanced_accuracy']:.4f} pairwise_corr={cs['mean_pairwise_corr_of_top_confidence']:.4f} "
            f"unanimity={cs['argmax_unanimity_rate']:.4f} ({res['train_seconds']}s)"
        )

    baseline = report["runs"]["rank_1"]
    verdict = {}
    for rank in RANKS:
        if rank == 1:
            continue
        r = report["runs"][f"rank_{rank}"]
        verdict[f"rank_{rank}_vs_rank_1"] = {
            "dir_bacc_delta": r["best_components"]["direction_balanced_accuracy"] - baseline["best_components"]["direction_balanced_accuracy"],
            "val_loss_delta": r["best_validation_loss"] - baseline["best_validation_loss"],
            "pairwise_corr_delta": r["collapse_at_best_checkpoint"]["mean_pairwise_corr_of_top_confidence"] - baseline["collapse_at_best_checkpoint"]["mean_pairwise_corr_of_top_confidence"],
            "unanimity_delta": r["collapse_at_best_checkpoint"]["argmax_unanimity_rate"] - baseline["collapse_at_best_checkpoint"]["argmax_unanimity_rate"],
            "n_params_delta": r["n_params"] - baseline["n_params"],
        }
    report["verdict"] = verdict
    for k, v in verdict.items():
        log(f"=== VERDICT {k}: dir_bacc_delta={v['dir_bacc_delta']:+.4f} pairwise_corr_delta={v['pairwise_corr_delta']:+.4f} params_delta={v['n_params_delta']:+d} ===")

    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"report written to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
