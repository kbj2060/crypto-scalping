#!/usr/bin/env python3
"""RESEARCH ONLY -- Step B for the faithful-TabM candidate (docs/model_contracts/
eth_candidate_faithful_tabm_batchensemble_contract_20260816.md): adds per-feature
piecewise-linear numerical embeddings (Gorishniy et al. 2022, "On Embeddings for Numerical
Features in Tabular Deep Learning", arXiv:2203.05556 -- the embedding scheme the TabM paper's own
ablation credits with the LARGER share of its reported +2-3% relative improvement, separate from
the BatchEnsemble R+S+B adapters tested in Step A) on top of `ThreeHeadTabMFull` (Step A's
completed-BatchEnsemble class, imported unmodified from the nseed script).

User explicitly overrode Step A's N>=5-seed negative-result CLOSED verdict ("아니야. 닫지 말고
논문대로 우리 모델을 최적화해보자") and asked whether all layers were actually retrained. Answer:
yes -- both architectures are always freshly initialized and trained end-to-end from scratch, no
warm-starting. But every one of the 66 Step-A runs (single-seed cheap_gate + N-seed reproduction)
stopped at exactly epoch 9 with patience=8, meaning the best checkpoint was always found at epoch 1
for both architectures alike -- a separate patience/epoch-budget relaxation diagnostic
(research_eth_candidate_faithful_tabm_batchensemble_patience_diag_20260816.py) checks whether that
was starving Step A specifically. This script is the OTHER paper-fidelity gap: numerical embeddings,
which per the contract's "설계" section were deliberately deferred to a separate Step B to isolate
the two effects, not conflated with Step A's adapter completion.

PiecewiseLinearEmbedding: bin edges are per-feature quantile cutpoints computed from the training
data (non-trainable buffer, same convention as _standardize_fit's mean/std); the per-feature linear
projection to d_embed IS trainable. T=16 bins, d_embed=8 -- deliberately scaled down from the
paper's typical benchmark defaults (tens of bins, comparable embedding dims run on much larger
tabular benchmarks) to this project's modest 185-feature/78k-row scale, per the contract's
"미해결 이슈 2" note that exact hyperparameters need deciding at Step B time, not inherited
unchanged from the paper's own benchmark scale.

Single-seed cheap_gate only (three-way comparison: baseline_R_only, full_R_S_B [Step A alone],
full_R_S_B_embed [Step A+B]) -- isolates the embedding's own marginal effect on top of Step A,
consistent with the contract's "섞어서 한 번에 테스트하지 않는다" ablation design. N>=5 seed
reproduction is required before any adoption claim, same seed-diversity policy as Step A, and is
NOT satisfied by this script.
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

import research_eth_candidate_faithful_tabm_batchensemble_nseed_20260816 as nseed  # noqa: E402

gate = nseed.gate
base = gate.base
hard = gate.hard

OUT_DIR = ROOT / "tmp/causal_regen_20260516/eth_candidate_faithful_tabm_batchensemble_stepb_embed_20260816"
CFG = gate.CFG
SEED = 260816
EPOCHS = gate.EPOCHS
N_BINS = 16
D_EMBED = 8


def log(msg: str) -> None:
    print(f"[faithful_tabm_stepb_embed] {msg}", flush=True)


class PiecewiseLinearEmbedding(nn.Module):
    """Per-feature PLE numerical embedding. bin_edges: (n_features, n_bins+1), non-trainable."""

    def __init__(self, bin_edges: torch.Tensor, d_embed: int) -> None:
        super().__init__()
        n_features, n_edges = bin_edges.shape
        self.n_features = int(n_features)
        self.n_bins = int(n_edges - 1)
        self.d_embed = int(d_embed)
        self.register_buffer("bin_edges", bin_edges)
        self.weight = nn.Parameter(torch.randn(self.n_features, self.n_bins, self.d_embed) * (1.0 / self.n_bins**0.5))
        self.bias = nn.Parameter(torch.zeros(self.n_features, self.d_embed))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lo = self.bin_edges[:, :-1]
        hi = self.bin_edges[:, 1:]
        width = (hi - lo).clamp_min(1.0e-6)
        ple = ((x.unsqueeze(-1) - lo.unsqueeze(0)) / width.unsqueeze(0)).clamp(0.0, 1.0)
        emb = torch.einsum("bft,ftd->bfd", ple, self.weight) + self.bias.unsqueeze(0)
        return emb.reshape(x.shape[0], -1)


class ThreeHeadTabMFullEmbed(nn.Module):
    """Step A (complete BatchEnsemble R+S+B) + Step B (piecewise-linear numerical embeddings)."""

    def __init__(self, n_features: int, *, cfg=CFG, bin_edges: torch.Tensor, d_embed: int = D_EMBED) -> None:
        super().__init__()
        self.embed = PiecewiseLinearEmbedding(bin_edges, d_embed)
        self.backbone = gate.ThreeHeadTabMFull(n_features * d_embed, cfg=cfg)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.backbone(self.embed(x))


def _quantile_bin_edges(x: np.ndarray, n_bins: int) -> torch.Tensor:
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(x, qs, axis=0).astype(np.float32)  # (n_bins+1, n_features)
    return torch.from_numpy(edges.T.copy())  # (n_features, n_bins+1)


def _fit_one_embed(
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
    use_embed: bool,
) -> dict[str, Any]:
    """Copy of gate._fit_one, adding bin-edge computation + PLE-embedded model construction when
    use_embed=True; use_embed=False path reproduces plain full_R_S_B for the 3-way comparison."""
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

    if use_embed:
        bin_edges = _quantile_bin_edges(_x_np, N_BINS)
        model = ThreeHeadTabMFullEmbed(x_dir_np.shape[1], cfg=CFG, bin_edges=bin_edges, d_embed=D_EMBED).to(device)
    else:
        model = gate.ThreeHeadTabMFull(x_dir_np.shape[1], cfg=CFG).to(device)
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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = base._device("cpu")
    base._seed_everything(SEED)
    log(f"=== stage=prepare_frames (light, shared) n_bins={N_BINS} d_embed={D_EMBED} ===")
    frames = gate._prepare_frames_light()
    fee, slip = base.omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    x_train = base._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_offsets = [1, 2, 3, 6, 12, 24, 48, 96, 192, 384]
    x_exit_raw, y_exit, frame_exit, exit_diag = gate.exit_head._build_exit_dataset_independent(
        frames["train_df"], frames["s_train_label"], frames["train_fixed"],
        fee=fee, slip=slip, cost_mult=3.0, exit_edge_min=0.0020, hold_offsets=hold_offsets, max_samples=0,
    )
    x_exit = base._exit_input_from_position_rows(x_exit_raw, base_cols)
    log(f"  n_train={len(x_train)} n_exit={len(x_exit)} n_features={x_train.shape[1]}")

    report: dict[str, Any] = {
        "design": "faithful-TabM Step B cheap_gate -- piecewise-linear numerical embeddings on top of Step A (R+S+B). Single seed.",
        "seed": SEED, "epochs_budget": EPOCHS, "n_bins": N_BINS, "d_embed": D_EMBED, "architectures": {},
    }

    for arch_name, model_cls, use_embed in (
        ("baseline_R_only", base.ThreeHeadTabM, None),
        ("full_R_S_B", gate.ThreeHeadTabMFull, False),
        ("full_R_S_B_embed", None, True),
    ):
        log(f"=== stage=train architecture={arch_name} ===")
        expert_results = []
        for idx, expert in enumerate(hard.EXPERT_NAMES):
            if arch_name == "baseline_R_only":
                res = gate._fit_one(model_cls, x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=idx, seed=SEED, epochs=EPOCHS, device=device)
            else:
                res = _fit_one_embed(x_train, y_train, train_raw, x_exit, y_exit, frame_exit, expert_idx=idx, seed=SEED, epochs=EPOCHS, device=device, use_embed=bool(use_embed))
            expert_results.append(res)
            log(f"  {arch_name} {expert}: n_params={res['n_params']} epochs_ran={res['epochs_ran']} "
                f"best_val_loss={res['best_validation_loss']:.4f} dir_bacc={res['best_components']['direction_balanced_accuracy']:.4f} "
                f"({res['train_seconds']}s)")
        report["architectures"][arch_name] = expert_results

    log("=== stage=summary ===")
    for expert in hard.EXPERT_NAMES:
        rows = {a: next(r for r in report["architectures"][a] if r["expert"] == expert) for a in ("baseline_R_only", "full_R_S_B", "full_R_S_B_embed")}
        log(f"  {expert:6s}: " + " | ".join(f"{a}: val_loss={rows[a]['best_validation_loss']:.4f} bacc={rows[a]['best_components']['direction_balanced_accuracy']:.4f}" for a in rows))
        d1 = rows["full_R_S_B_embed"]["best_components"]["direction_balanced_accuracy"] - rows["full_R_S_B"]["best_components"]["direction_balanced_accuracy"]
        log(f"  {expert:6s}: delta_bacc(embed vs StepA-only)={d1:+.4f}")

    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
