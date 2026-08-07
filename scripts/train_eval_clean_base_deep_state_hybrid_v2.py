#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import train_eval_clean_base_deep_state_hybrid_v1 as base_v1  # noqa: E402


MODEL_ID = "clean_base_deep_state_hybrid_v2"
SEEDS = (42, 7, 13)
LOOKBACK = 72
HIDDEN_DIM = 48
PER_SEED_EMBED_DIM = 16
ENSEMBLE_EMBED_DIM = PER_SEED_EMBED_DIM * len(SEEDS)
N_CLUSTERS = 6
DEFAULT_EPOCHS = 40
PATIENCE = 7


class EnhancedGRUStateEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = HIDDEN_DIM, embed_dim: int = PER_SEED_EMBED_DIM) -> None:
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, dropout=0.0)
        self.norm = nn.LayerNorm(hidden_dim)
        self.embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, embed_dim),
            nn.Tanh(),
        )
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 3),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, h = self.gru(x)
        z = self.embed(self.norm(h[-1]))
        y = self.head(z)
        return y, z


class GRUSeedEnsemble(nn.Module):
    def __init__(self, models: list[EnhancedGRUStateEncoder]) -> None:
        super().__init__()
        self.models = nn.ModuleList(models)
        self.hidden_dim = HIDDEN_DIM
        self.output_embed_dim = ENSEMBLE_EMBED_DIM


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)


def _train_one_seed(
    seed: int,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[EnhancedGRUStateEncoder, dict[str, Any]]:
    _set_seed(seed)
    model = EnhancedGRUStateEncoder(input_dim=train_x.shape[2])
    opt = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in loader:
            opt.zero_grad(set_to_none=True)
            pred, _z = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            count += len(xb)
        train_loss = total / max(count, 1)
        model.eval()
        with torch.no_grad():
            val_pred, _val_z = model(val_x)
            val_loss = float(loss_fn(val_pred, val_y).detach().cpu())
        history.append({"epoch": float(epoch), "train_loss": float(train_loss), "val_loss": float(val_loss)})
        if val_loss < best_loss - 1e-5:
            best_loss = val_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model, {"seed": int(seed), "best_epoch": int(best_epoch), "best_val_loss": float(best_loss), "history": history}


def _train_deep_encoder_v2(
    seq: np.ndarray,
    labels,
    *,
    epochs: int,
    batch_size: int,
) -> tuple[GRUSeedEnsemble, dict[str, Any]]:
    x_all = torch.tensor(seq, dtype=torch.float32)
    y_raw = labels[["full", "adverse", "same"]].to_numpy(dtype=np.float32)
    n = len(x_all)
    split = max(1, int(n * 0.82))
    split = min(split, n - 1)
    y_mean = y_raw[:split].mean(axis=0)
    y_std = y_raw[:split].std(axis=0)
    y_std = np.where(y_std < 1e-6, 1.0, y_std)
    y_all = torch.tensor((y_raw - y_mean) / y_std, dtype=torch.float32)
    train_x, val_x = x_all[:split], x_all[split:]
    train_y, val_y = y_all[:split], y_all[split:]
    models: list[EnhancedGRUStateEncoder] = []
    seed_meta: list[dict[str, Any]] = []
    for seed in SEEDS:
        model, meta = _train_one_seed(
            seed,
            train_x,
            train_y,
            val_x,
            val_y,
            epochs=int(epochs),
            batch_size=int(batch_size),
        )
        models.append(model)
        seed_meta.append(meta)
    ensemble = GRUSeedEnsemble(models)
    return ensemble, {
        "epochs_requested": int(epochs),
        "batch_size": int(batch_size),
        "inner_train_rows": int(len(train_x)),
        "inner_val_rows": int(len(val_x)),
        "target_mean": y_mean.astype(float).tolist(),
        "target_std": y_std.astype(float).tolist(),
        "seeds": seed_meta,
        "ensemble": {"seeds": list(SEEDS), "hidden_dim": HIDDEN_DIM, "per_seed_embed_dim": PER_SEED_EMBED_DIM, "output_embed_dim": ENSEMBLE_EMBED_DIM},
    }


def _deep_predict_v2(model: GRUSeedEnsemble, seq: np.ndarray, target_mean: list[float], target_std: list[float]) -> dict[str, np.ndarray]:
    if len(seq) == 0:
        return {
            "full": np.asarray([], dtype=np.float64),
            "adverse": np.asarray([], dtype=np.float64),
            "same": np.asarray([], dtype=np.float64),
            "embedding": np.zeros((0, ENSEMBLE_EMBED_DIM), dtype=np.float64),
        }
    preds: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(seq), 512):
            xb = torch.tensor(seq[start : start + 512], dtype=torch.float32)
            chunk_preds: list[np.ndarray] = []
            chunk_embs: list[np.ndarray] = []
            for encoder in model.models:
                pred, z = encoder(xb)
                chunk_preds.append(pred.cpu().numpy())
                chunk_embs.append(z.cpu().numpy())
            preds.append(np.mean(np.stack(chunk_preds, axis=0), axis=0))
            embeddings.append(np.concatenate(chunk_embs, axis=1))
    pred_arr = np.vstack(preds).astype(np.float64)
    emb = np.vstack(embeddings).astype(np.float64)
    raw = pred_arr * np.asarray(target_std, dtype=np.float64) + np.asarray(target_mean, dtype=np.float64)
    return {
        "full": raw[:, 0],
        "adverse": np.maximum(raw[:, 1], 0.0),
        "same": raw[:, 2],
        "embedding": emb,
    }


def _contract_doc_v2(report: dict[str, Any] | None = None) -> str:
    return f"""# Clean Base Deep State Hybrid V2 Contract

Status: `experimental_challenger`

## Architecture

- Deep layer: 3-seed GRU ensemble over `{LOOKBACK}` bars.
- Per-seed hidden/embedding: `{HIDDEN_DIM}` / `{PER_SEED_EMBED_DIM}`.
- Ensemble embedding width: `{ENSEMBLE_EMBED_DIM}`.
- Early stopping: train-internal chronological holdout, patience `{PATIENCE}`.
- Unsupervised layer: KMeans with `{N_CLUSTERS}` clusters over ensemble embedding and deep heads.
- Supervised layer: HGB same-side utility and adverse-risk heads.
- Execution layer: deterministic same-side sleeve only.

## Runtime Invariants

- Clean base/Lifecycle core entries, sides, exits, notionals, and leverage are preserved.
- The hybrid layer can only add a same-side sleeve or abstain.
- No OOS threshold selection.
- Forbidden runtime fields: `{', '.join(base_v1.FORBIDDEN_RUNTIME_FEATURES)}`.
"""


def main() -> int:
    base_v1.MODEL_ID = MODEL_ID
    base_v1.LOOKBACK = LOOKBACK
    base_v1.HIDDEN_DIM = HIDDEN_DIM
    base_v1.EMBED_DIM = ENSEMBLE_EMBED_DIM
    base_v1.N_CLUSTERS = N_CLUSTERS
    base_v1.DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/clean_base_deep_state_hybrid_v2"
    base_v1.DEFAULT_REPORT = ROOT / "data/ensemble/reports/clean_base_deep_state_hybrid_v2_2026.json"
    base_v1.DEFAULT_GRID = ROOT / "data/ensemble/reports/clean_base_deep_state_hybrid_v2_grid.csv"
    base_v1.DEFAULT_LEDGER = ROOT / "data/ensemble/reports/clean_base_deep_state_hybrid_v2_ledger.csv"
    base_v1.DEFAULT_DOC = ROOT / "docs/experiments/clean_base_deep_state_hybrid_v2.md"
    base_v1.DEFAULT_CONTRACT = ROOT / "docs/model_contracts/clean_base_deep_state_hybrid_v2_contract.md"
    base_v1._train_deep_encoder = _train_deep_encoder_v2
    base_v1._deep_predict = _deep_predict_v2
    base_v1._contract_doc = _contract_doc_v2
    args = base_v1.parse_args()
    if int(args.deep_epochs) <= 5:
        args.deep_epochs = DEFAULT_EPOCHS
    args.deep_batch_size = 128
    report = base_v1.run(args)
    if isinstance(report, dict):
        summary = {
            "model_id": MODEL_ID,
            "report": report.get("artifacts", {}).get("report"),
            "verdict": report.get("verdict"),
            "selected": report.get("selected_config", {}).get("name"),
            "pnl": report.get("cost_1x", {}).get("pnl"),
            "mdd": report.get("cost_1x", {}).get("mdd"),
            "cost2": report.get("cost_2x", {}).get("pnl"),
            "cost3": report.get("cost_3x", {}).get("pnl"),
        }
        print(json.dumps({"v2_summary": summary}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
