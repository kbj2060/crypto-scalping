"""Train one of the three BTC deep-feature encoders (see
ensemble/deep_features/btc_deepfeat_encoders_20260806.py) supervised against the zigzag
risk-adjusted soft label (ensemble/deep_features/btc_deepfeat_dataset_20260806.py), and emit the
learned embeddings as a deep-feature parquet for downstream use.

Usage:
    python scripts/train_btc_deepfeat_encoders_20260806.py --arch cnn_seq
    python scripts/train_btc_deepfeat_encoders_20260806.py --arch cnn_category
    python scripts/train_btc_deepfeat_encoders_20260806.py --arch transformer

This trains the encoder itself (soft-label classification loss) and reports VAL/OOS soft-label
loss + hard-label top-1 agreement -- it does NOT run a trading backtest. That is deliberately out
of scope here: the point of this stage is to check whether any of the three architectures learns
a soft-label-predictive representation at all before spending time wiring one into a strategy.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from ensemble.deep_features.btc_deepfeat_dataset_20260806 import build_dataset  # noqa: E402
from ensemble.deep_features.btc_deepfeat_encoders_20260806 import ARCHES, build_model  # noqa: E402

OUT_ROOT = ROOT / "tmp/btc_deepfeat_encoders_20260806"


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _prepare_target(y_soft: torch.Tensor, y_hard: torch.Tensor, sharpen: float, cash_weight: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Diagnosed 2026-08-06: the base zigzag soft label is fully peaked (maxprob=1.0, entropy=0)
    on CASH bars but only moderately peaked on active LONG/SHORT bars (mean maxprob ~0.79,
    entropy ~0.49) -- and its own argmax disagrees with the hard direction label on ~9% of active
    bars (low-quality/near-boundary wave segments the risk-adjusted score correctly hedges away
    from). That gap is why minimizing soft-CE loss doesn't track hard-direction accuracy well.
    `sharpen<1.0` raises the soft label to power 1/sharpen (temperature scaling) to pull active-bar
    targets closer to their own argmax before computing loss; `cash_weight<1.0` down-weights the
    already-trivial (always fully-peaked) CASH-labeled samples' contribution to the loss so
    gradient signal focuses on the harder LONG/SHORT calibration. Both default to a no-op (1.0)."""
    if sharpen != 1.0:
        p = torch.clamp(y_soft, min=1e-8) ** (1.0 / sharpen)
        target = p / p.sum(dim=-1, keepdim=True)
    else:
        target = y_soft
    weight = torch.where(y_hard == 0, torch.full_like(y_hard, cash_weight, dtype=torch.float32), torch.ones_like(y_hard, dtype=torch.float32))
    return target, weight


def _soft_ce_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    per_sample = -(target * log_probs).sum(dim=-1)
    if weight is None:
        return per_sample.mean()
    return (per_sample * weight).sum() / weight.sum().clamp(min=1e-8)


def _iterate_batches(row_idx: np.ndarray, batch_size: int, *, shuffle: bool, rng: np.random.Generator) -> list[np.ndarray]:
    idx = row_idx.copy()
    if shuffle:
        rng.shuffle(idx)
    return [idx[i : i + batch_size] for i in range(0, len(idx), batch_size)]


@torch.no_grad()
def _evaluate(model: nn.Module, ds, split: str, device: torch.device, batch_size: int, sharpen: float, cash_weight: float, quality_loss_weight: float) -> dict:
    model.eval()
    row_idx = ds.end_idx[split]
    total_loss, total_correct, total_n = 0.0, 0, 0
    total_quality_se = 0.0
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        y_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
        y_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
        y_quality = torch.from_numpy(ds.y_quality_all[chunk]).to(device)
        target, weight = _prepare_target(y_soft, y_hard, sharpen, cash_weight)
        logits, quality_pred, _ = model(x)
        loss = _soft_ce_loss(logits, target, weight)
        if quality_loss_weight > 0.0:
            loss = loss + quality_loss_weight * F.mse_loss(quality_pred, y_quality)
        total_loss += float(loss.item()) * len(chunk)
        total_quality_se += float(F.mse_loss(quality_pred, y_quality, reduction="sum").item())
        total_correct += int((logits.argmax(dim=-1) == y_hard).sum().item())
        total_n += len(chunk)
    return {
        "n": total_n,
        "soft_ce_loss": total_loss / max(total_n, 1),
        "hard_top1_acc": total_correct / max(total_n, 1),
        "quality_mse": total_quality_se / max(total_n, 1),
    }


@torch.no_grad()
def _emit_embeddings(model: nn.Module, ds, split: str, device: torch.device, batch_size: int, out_path: Path) -> None:
    model.eval()
    row_idx = ds.end_idx[split]
    embs, quality_preds = [], []
    for i in range(0, len(row_idx), batch_size):
        chunk = row_idx[i : i + batch_size]
        x = torch.from_numpy(ds.get_batch(chunk)).to(device)
        _, quality_pred, emb = model(x)
        embs.append(emb.cpu().numpy())
        quality_preds.append(quality_pred.cpu().numpy())
    embs = np.concatenate(embs, axis=0) if embs else np.zeros((0, 0), dtype=np.float32)
    quality_preds = np.concatenate(quality_preds, axis=0) if quality_preds else np.zeros((0,), dtype=np.float32)
    cols = [f"deepfeat_{i}" for i in range(embs.shape[1])]
    out = pd.DataFrame(embs, columns=cols)
    out.insert(0, "timestamp", ds.timestamps_all[row_idx])
    out["quality_pred_log1p_calmar"] = quality_preds
    out.to_parquet(out_path, index=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arch", required=True, choices=ARCHES)
    p.add_argument("--window", type=int, default=48)
    p.add_argument("--embed-dim", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--train-stride", type=int, default=4, help="subsample train windows every Nth row to cut near-duplicate overlapping-window redundancy")
    p.add_argument("--seed", type=int, default=20260806)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--out-dir", type=Path, default=None)
    # transformer-only architecture hyperparameters (ignored by cnn_seq/cnn_category)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--ffn-mult", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.25)
    # soft-label target shaping (see _prepare_target docstring)
    p.add_argument("--label-sharpen", type=float, default=1.0, help="temperature<1.0 sharpens active-bar soft targets toward their own argmax; 1.0=no-op")
    p.add_argument("--cash-weight", type=float, default=1.0, help="loss weight applied to CASH-labeled (already fully-peaked) samples; <1.0 de-emphasizes them")
    # quality regression auxiliary head (predicts log1p(zigzag_path_calmar), used to filter low-quality entries at backtest time)
    p.add_argument("--quality-head", action="store_true", help="add a quality regression head alongside the direction classification head")
    p.add_argument("--quality-loss-weight", type=float, default=0.0, help="weight of the quality MSE loss added to the soft-CE loss; 0.0=head unused in training even if present")
    p.add_argument("--head-type", type=str, default="linear", choices=("linear", "tabm"), help="prediction head trained end-to-end with the encoder: plain nn.Linear, or a TabM-style ensemble MLP")
    # label source override (defaults to the zigzag wave label; pass the triple-barrier label to train against that instead)
    p.add_argument("--label-path", type=Path, default=None)
    p.add_argument("--hard-col", type=str, default=None)
    p.add_argument("--soft-cols", type=str, default=None, help="comma-separated, e.g. trade_outcome_soft_cash,trade_outcome_soft_long,trade_outcome_soft_short")
    p.add_argument("--quality-col", type=str, default=None)
    p.add_argument("--extra-feature-path", type=Path, default=None)
    p.add_argument("--extra-feature-cols", type=str, default=None, help="comma-separated extra causal feature columns to merge onto the standard 113")
    args = p.parse_args()

    _seed_everything(args.seed)
    device = _device(args.device)
    out_dir = args.out_dir or (OUT_ROOT / args.arch)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(
        window=args.window, train_stride=args.train_stride,
        label_path=args.label_path, hard_col=args.hard_col,
        soft_cols=args.soft_cols.split(",") if args.soft_cols else None,
        quality_col=args.quality_col,
        extra_feature_path=args.extra_feature_path,
        extra_feature_cols=args.extra_feature_cols.split(",") if args.extra_feature_cols else None,
    )
    model = build_model(
        args.arch, len(ds.feature_columns), ds.category_sizes, embed_dim=args.embed_dim,
        d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_mult=args.ffn_mult, dropout=args.dropout, quality_head=args.quality_head, head_type=args.head_type,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=2, min_lr=1e-6)

    rng = np.random.default_rng(args.seed)
    best_val_loss = float("inf")
    best_state = None
    epochs_since_best = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        batches = _iterate_batches(ds.end_idx["train"], args.batch_size, shuffle=True, rng=rng)
        train_loss_sum, train_n = 0.0, 0
        for chunk in batches:
            x = torch.from_numpy(ds.get_batch(chunk)).to(device)
            y_soft = torch.from_numpy(ds.y_soft_all[chunk]).to(device)
            y_hard = torch.from_numpy(ds.y_hard_all[chunk]).to(device)
            y_quality = torch.from_numpy(ds.y_quality_all[chunk]).to(device)
            target, weight = _prepare_target(y_soft, y_hard, args.label_sharpen, args.cash_weight)
            opt.zero_grad()
            logits, quality_pred, _ = model(x)
            loss = _soft_ce_loss(logits, target, weight)
            if args.quality_loss_weight > 0.0:
                loss = loss + args.quality_loss_weight * F.mse_loss(quality_pred, y_quality)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            opt.step()
            train_loss_sum += float(loss.item()) * len(chunk)
            train_n += len(chunk)

        val_metrics = _evaluate(model, ds, "val", device, args.batch_size, args.label_sharpen, args.cash_weight, args.quality_loss_weight)
        scheduler.step(val_metrics["soft_ce_loss"])
        epoch_row = {
            "epoch": epoch,
            "lr": opt.param_groups[0]["lr"],
            "train_soft_ce_loss": train_loss_sum / max(train_n, 1),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(epoch_row)
        print(json.dumps(epoch_row))

        if val_metrics["soft_ce_loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["soft_ce_loss"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if epochs_since_best >= args.patience:
                print(f"early stopping at epoch {epoch} (best val_soft_ce_loss={best_val_loss:.5f})")
                break

    if best_state is None:
        raise RuntimeError("training produced no valid checkpoint")
    model.load_state_dict(best_state)

    val_metrics = _evaluate(model, ds, "val", device, args.batch_size, args.label_sharpen, args.cash_weight, args.quality_loss_weight)
    oos_metrics = _evaluate(model, ds, "oos", device, args.batch_size, args.label_sharpen, args.cash_weight, args.quality_loss_weight)
    print(f"FINAL val={val_metrics} oos={oos_metrics}")

    config = {
        "arch": args.arch,
        "window": args.window,
        "embed_dim": args.embed_dim,
        "d_model": args.d_model,
        "n_heads": args.n_heads,
        "n_layers": args.n_layers,
        "ffn_mult": args.ffn_mult,
        "dropout": args.dropout,
        "label_sharpen": args.label_sharpen,
        "cash_weight": args.cash_weight,
        "quality_head": args.quality_head,
        "quality_loss_weight": args.quality_loss_weight,
        "head_type": args.head_type,
        "label_path": str(args.label_path) if args.label_path else None,
        "hard_col": args.hard_col,
        "soft_cols": args.soft_cols,
        "quality_col": args.quality_col,
        "n_features": len(ds.feature_columns),
        "category_order": ds.category_order,
        "category_sizes": ds.category_sizes,
        "feature_columns": ds.feature_columns,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "train_stride": args.train_stride,
        "grad_clip_norm": args.grad_clip_norm,
    }
    torch.save(
        {"model_state": best_state, "config": config, "mean": ds.mean, "std": ds.std},
        out_dir / "deepfeat_bundle.pt",
    )

    metrics = {
        "arch": args.arch,
        "val": val_metrics,
        "oos": oos_metrics,
        "history": history,
        "teacher_label": "zigzag_risk_adjusted_soft_action_label (build_btc_5m_zigzag_and_pivot_labels_20260806.py)",
        "note": "soft-label supervised loss + hard top-1 agreement only; no trading backtest run at this stage",
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    for split in ("train", "val", "oos"):
        _emit_embeddings(model, ds, split, device, args.batch_size, out_dir / f"deepfeat_embeddings_{split}.parquet")

    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
