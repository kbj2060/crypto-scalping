#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_omega1_mamba_teacher_20260531 import (
    DEFAULT_CANDIDATE_DIR,
    DEFAULT_CHRONOS_DIR,
    DEFAULT_REGIME3_CURRENT_DIR,
    DEFAULT_REGIME3_DIR,
    FORBIDDEN_PREFIXES,
    OMEGA1_SECONDARY_INPUTS,
    SeqDataset,
    _build_frame,
    _class_weights,
    _feature_columns,
    _is_forbidden,
    _json_default,
    _labels,
    _standardize,
    _valid_indices,
)


MODEL_ID = "omega1_transformer_teacher_20260531"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_transformer_teacher_current_chronos_20260531"

TEACHER_TRANSFORMER_OUTPUTS = [
    "teacher_transformer_p_cash",
    "teacher_transformer_p_long",
    "teacher_transformer_p_short",
    "teacher_transformer_confidence",
    "teacher_transformer_side_edge",
    "teacher_transformer_uncertainty",
    "teacher_transformer_risk_veto_score",
]


@dataclass(frozen=True)
class TrainConfig:
    seq_len: int = 72
    label_threshold: float = 0.08
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    ff_mult: int = 3
    dropout: float = 0.10
    batch_size: int = 512
    epochs: int = 6
    lr: float = 2e-4
    weight_decay: float = 1e-4
    val_fraction: float = 0.18
    seed: int = 20260531


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.shape[1], :]


class Omega1TransformerTeacher(nn.Module):
    def __init__(self, input_dim: int, cfg: TrainConfig):
        super().__init__()
        self.in_proj = nn.Linear(input_dim, cfg.d_model)
        self.pos = SinusoidalPositionEncoding(cfg.d_model, cfg.seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.heads,
            dim_feedforward=cfg.d_model * cfg.ff_mult,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 3),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.pos(self.in_proj(x))
        z = self.encoder(z)
        return self.head(self.norm(z[:, -1, :]))


def _evaluate_logits(y: np.ndarray, indices: np.ndarray, proba: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(proba, axis=1).astype(np.int64)
    yt = y[indices].astype(np.int64)
    out: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(yt, pred)),
        "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(yt, minlength=3))},
        "pred_counts": {str(i): int(v) for i, v in enumerate(np.bincount(pred, minlength=3))},
    }
    try:
        out["ovr_auc"] = float(roc_auc_score(yt, proba, multi_class="ovr", labels=[0, 1, 2]))
    except ValueError:
        out["ovr_auc"] = None
    return out


@torch.no_grad()
def _predict(model: nn.Module, x: np.ndarray, indices: np.ndarray, seq_len: int, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    ds = SeqDataset(x, np.zeros(len(x), dtype=np.int64), indices, seq_len)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device.type == "cuda"))
    outs: list[np.ndarray] = []
    for xb, _ in loader:
        logits = model(xb.to(device, non_blocking=True))
        outs.append(torch.softmax(logits, dim=-1).detach().cpu().numpy())
    return np.concatenate(outs, axis=0)


def _append_outputs(frame, indices: np.ndarray, proba: np.ndarray):
    out = frame.copy()
    for col in TEACHER_TRANSFORMER_OUTPUTS:
        out[col] = np.nan
    full = np.zeros((len(frame), 3), dtype=np.float32)
    full[indices] = proba.astype(np.float32)
    out.loc[indices, "teacher_transformer_p_cash"] = full[indices, 0]
    out.loc[indices, "teacher_transformer_p_long"] = full[indices, 1]
    out.loc[indices, "teacher_transformer_p_short"] = full[indices, 2]
    conf = np.max(full[indices], axis=1)
    out.loc[indices, "teacher_transformer_confidence"] = conf
    out.loc[indices, "teacher_transformer_side_edge"] = full[indices, 1] - full[indices, 2]
    out.loc[indices, "teacher_transformer_uncertainty"] = 1.0 - conf
    out.loc[indices, "teacher_transformer_risk_veto_score"] = np.clip(full[indices, 0] + (1.0 - conf), 0.0, 1.0)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument("--regime3-dir", type=Path, default=DEFAULT_REGIME3_DIR)
    parser.add_argument("--regime3-current-dir", type=Path, default=DEFAULT_REGIME3_CURRENT_DIR)
    parser.add_argument("--chronos-dir", type=Path, default=DEFAULT_CHRONOS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--label-threshold", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260531)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))

    cfg = TrainConfig(seq_len=int(args.seq_len), label_threshold=float(args.label_threshold), batch_size=int(args.batch_size), epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    train_name = "trade_candidates_2025_alpha6_current_tail111_exact.csv"
    oos_name = "trade_candidates_2026_alpha6_current_tail111_exact.csv"
    train = _build_frame(args.candidate_dir / train_name, year=2025, regime3_dir=args.regime3_dir, regime3_current_dir=args.regime3_current_dir, chronos_dir=args.chronos_dir)
    oos = _build_frame(args.candidate_dir / oos_name, year=2026, regime3_dir=args.regime3_dir, regime3_current_dir=args.regime3_current_dir, chronos_dir=args.chronos_dir)

    feature_cols, base_cols = _feature_columns(train, oos)
    x_train, x_oos, norm = _standardize(train, oos, feature_cols)
    y_train = _labels(train["tp_sl_action_score"], cfg.label_threshold)
    y_oos = _labels(oos["tp_sl_action_score"], cfg.label_threshold)

    all_idx = _valid_indices(len(train), cfg.seq_len)
    split_at = int(round(len(all_idx) * (1.0 - cfg.val_fraction)))
    train_idx = all_idx[:split_at]
    val_idx = all_idx[split_at:]
    oos_idx = _valid_indices(len(oos), cfg.seq_len)

    model = Omega1TransformerTeacher(x_train.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y_train, train_idx, device), label_smoothing=0.02)
    train_loader = DataLoader(
        SeqDataset(x_train, y_train, train_idx, cfg.seq_len),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_val = -1.0
    history: list[dict[str, Any]] = []
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_proba = _predict(model, x_train, val_idx, cfg.seq_len, cfg.batch_size, device)
        val_metrics = _evaluate_logits(y_train, val_idx, val_proba)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), "val": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
        score = float(val_metrics["balanced_accuracy"])
        if score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    train_proba = _predict(model, x_train, all_idx, cfg.seq_len, cfg.batch_size, device)
    val_proba = _predict(model, x_train, val_idx, cfg.seq_len, cfg.batch_size, device)
    oos_proba = _predict(model, x_oos, oos_idx, cfg.seq_len, cfg.batch_size, device)
    train_metrics = _evaluate_logits(y_train, all_idx, train_proba)
    val_metrics = _evaluate_logits(y_train, val_idx, val_proba)
    oos_metrics = _evaluate_logits(y_oos, oos_idx, oos_proba)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_out = _append_outputs(train, all_idx, train_proba)
    oos_out = _append_outputs(oos, oos_idx, oos_proba)
    train_path = args.out_dir / train_name
    oos_path = args.out_dir / oos_name
    train_out.to_csv(train_path, index=False)
    oos_out.to_csv(oos_path, index=False)
    model_path = args.out_dir / "omega1_transformer_teacher.pt"
    torch.save({"model_id": MODEL_ID, "config": asdict(cfg), "state_dict": model.state_dict(), "feature_cols": feature_cols, "base_cols": base_cols, "norm": norm}, model_path)

    audit = {
        "model_id": MODEL_ID,
        "config": asdict(cfg),
        "device": str(device),
        "candidate_dir": str(args.candidate_dir),
        "out_dir": str(args.out_dir),
        "feature_count": int(len(feature_cols)),
        "secondary_feature_count": int(len(OMEGA1_SECONDARY_INPUTS)),
        "base_feature_count": int(len(base_cols)),
        "feature_cols": feature_cols,
        "secondary_feature_cols": OMEGA1_SECONDARY_INPUTS,
        "base_feature_cols": base_cols,
        "outputs": TEACHER_TRANSFORMER_OUTPUTS,
        "history": history,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "oos_label_probe_metrics": oos_metrics,
        "contract": {
            "seq_len": int(cfg.seq_len),
            "teacher_feedback_inputs_forbidden": True,
            "regime4_forbidden": True,
            "regime3_pred_forbidden": True,
            "chronos_exact_join": True,
            "regime3_exact_join": True,
            "base_feature_policy": "numeric_current_context_after_explicit_forbidden_filters",
        },
        "artifacts": {"train_csv": str(train_path), "oos_csv": str(oos_path), "model": str(model_path), "audit": str(args.out_dir / "omega1_transformer_teacher_audit.json")},
    }
    forbidden_selected = [c for c in feature_cols if _is_forbidden(c)]
    if forbidden_selected:
        raise RuntimeError(f"forbidden columns selected: {forbidden_selected[:20]}")
    if any(c.startswith(FORBIDDEN_PREFIXES) for c in feature_cols):
        raise RuntimeError("forbidden prefix selected")
    (args.out_dir / "omega1_transformer_teacher_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default))
    print(json.dumps(audit, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
