#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_regime4_pred_tft_clean_target_20260517 import (  # noqa: E402
    CLEAN4_PREFIX,
    CLASSES4,
    DEFAULT_CLEAN_2024,
    DEFAULT_CLEAN_2025,
    DEFAULT_PREDICT_2025,
    DEFAULT_TRAIN_2024,
    PRED_PREFIX,
    _clean_future_labels,
    _feature_cols,
    _known_cov,
    _merge_clean4,
    _num,
    _output,
    _read,
)
from scripts.build_regime_pred_moe_20260517 import _json_default  # noqa: E402


MODEL_ID = "regime4_pred_tft_multihorizon_20260517"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/regime4_pred_tft_multihorizon_20260517"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/regime4_pred_tft_multihorizon_20260517_report.json"
DEFAULT_SELECTED_REPORT = ROOT / "data/ensemble/reports/regime4_pred_tft_vsn_select_20260517_report.json"
HORIZONS = [12, 36, 72]


def _matrix(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: _num(frame, c) for c in cols}, index=frame.index)


def _prepare(frame: pd.DataFrame, cols: list[str], scaler: StandardScaler | None = None, medians: pd.Series | None = None, fit_rows: np.ndarray | None = None):
    raw = _matrix(frame, cols)
    if medians is None:
        fit = raw if fit_rows is None else raw.iloc[np.asarray(fit_rows, dtype=np.int64)]
        medians = fit.median(numeric_only=True).fillna(0.0)
    filled = raw.fillna(medians).fillna(0.0)
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(filled if fit_rows is None else filled.iloc[np.asarray(fit_rows, dtype=np.int64)])
    x = scaler.transform(filled).astype(np.float32)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), scaler, medians


def _labels_multi(frame: pd.DataFrame, horizons: list[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    max_h = max(horizons)
    n = max(0, len(frame) - max_h)
    out = pd.DataFrame(index=frame.index[:n])
    meta: dict[str, Any] = {"horizons": horizons, "label_source": "argmax_clean_regime4_at_t_plus_horizon"}
    for h in horizons:
        labels, h_meta = _clean_future_labels(frame, h)
        out[f"_label_h{h}"] = labels.loc[out.index, "_label_id"].astype(int)
        counts = out[f"_label_h{h}"].value_counts().reindex(range(len(CLASSES4)), fill_value=0)
        meta[f"h{h}_label_counts"] = {CLASSES4[i]: int(counts.loc[i]) for i in range(len(CLASSES4))}
    return out, meta


class MultiDS(Dataset):
    def __init__(self, x: np.ndarray, known_by_h: dict[int, np.ndarray], idx: np.ndarray, y: np.ndarray | None, seq_len: int) -> None:
        self.x = x
        self.known_by_h = known_by_h
        self.idx = idx.astype(np.int64)
        self.y = y
        self.seq_len = int(seq_len)
        self.horizons = sorted(known_by_h)

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        end = int(self.idx[i])
        start = end - self.seq_len + 1
        if start < 0:
            seq = np.concatenate([np.repeat(self.x[[0]], -start, axis=0), self.x[: end + 1]], axis=0)
        else:
            seq = self.x[start : end + 1]
        known = np.concatenate([self.known_by_h[h][end] for h in self.horizons]).astype(np.float32)
        if self.y is None:
            return torch.from_numpy(seq), torch.from_numpy(known)
        return torch.from_numpy(seq), torch.from_numpy(known), torch.from_numpy(self.y[end].astype(np.int64))


class GRN(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.gate = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + torch.sigmoid(self.gate(x)) * self.fc2(self.drop(F.elu(self.fc1(x)))))


class MultiTFT(nn.Module):
    def __init__(self, n_features: int, n_known: int, d_model: int, heads: int, layers: int, dropout: float, seq_len: int, n_heads_out: int) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(nn.Linear(n_features, n_features), nn.Sigmoid())
        self.input_proj = nn.Linear(n_features, d_model)
        self.known_proj = nn.Linear(n_known, d_model)
        self.pos = nn.Parameter(torch.zeros(1, seq_len, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=heads, dim_feedforward=d_model * 4, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)
        self.context = GRN(d_model * 5, d_model * 4, dropout)
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(d_model * 5, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, len(CLASSES4))) for _ in range(n_heads_out)])

    @staticmethod
    def tail_mean(x: torch.Tensor, n: int) -> torch.Tensor:
        return x[:, -min(n, x.shape[1]) :, :].mean(dim=1)

    def forward(self, x: torch.Tensor, known: torch.Tensor) -> list[torch.Tensor]:
        x = x * self.feature_gate(x)
        h = self.encoder(self.input_proj(x) + self.pos[:, : x.shape[1], :])
        pooled = torch.cat([h[:, -1], self.tail_mean(h, 12), self.tail_mean(h, 36), self.tail_mean(h, 72), self.known_proj(known)], dim=1)
        ctx = self.context(pooled)
        return [head(ctx) for head in self.heads]


def _class_weights(y: np.ndarray, head: int) -> torch.Tensor:
    yy = y[:, head]
    counts = np.bincount(yy, minlength=len(CLASSES4)).astype(float)
    w = counts.sum() / np.clip(len(CLASSES4) * counts, 1.0, None)
    return torch.tensor(np.clip(w, 0.35, 4.0), dtype=torch.float32)


def _predict(model: nn.Module, loader: DataLoader, device: torch.device) -> list[np.ndarray]:
    model.eval()
    rows: list[list[np.ndarray]] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch[0].to(device), batch[1].to(device))
            probs = [torch.softmax(z, dim=1).cpu().numpy() for z in logits]
            rows.append(probs)
    out = []
    for i in range(len(HORIZONS)):
        p = np.vstack([batch[i] for batch in rows]).astype(float)
        p /= np.clip(p.sum(axis=1, keepdims=True), 1e-12, None)
        out.append(p)
    return out


def _eval(y: np.ndarray, p: np.ndarray) -> dict[str, Any]:
    pred = np.argmax(p, axis=1)
    return {
        "rows": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "log_loss": float(log_loss(y, p, labels=list(range(len(CLASSES4))))),
        "true_counts": {CLASSES4[i]: int((y == i).sum()) for i in range(len(CLASSES4))},
        "pred_counts": {CLASSES4[i]: int((pred == i).sum()) for i in range(len(CLASSES4))},
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(len(CLASSES4)))).tolist(),
    }


def _fit(x, known_by_h, y, train_idx, val_idx, args, seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    known_dim = len(HORIZONS) * next(iter(known_by_h.values())).shape[1]
    model = MultiTFT(x.shape[1], known_dim, args.d_model, args.heads, args.layers, args.dropout, args.seq_len, len(HORIZONS)).to(device)
    train_loader = DataLoader(MultiDS(x, known_by_h, train_idx, y, args.seq_len), batch_size=args.batch_size, shuffle=True)
    val_loader = None if val_idx is None else DataLoader(MultiDS(x, known_by_h, val_idx, y, args.seq_len), batch_size=args.batch_size * 2, shuffle=False)
    criteria = [nn.CrossEntropyLoss(weight=_class_weights(y[train_idx], i).to(device), label_smoothing=0.03) for i in range(len(HORIZONS))]
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    hist = []
    best = None
    best_score = float("inf")
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for seq, known, target in train_loader:
            seq, known, target = seq.to(device), known.to(device), target.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(seq, known)
            loss = sum(criteria[i](logits[i], target[:, i]) for i in range(len(HORIZONS))) / len(HORIZONS)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        row = {"epoch": float(epoch), "train_loss": float(np.mean(losses))}
        if val_loader is not None:
            probs = _predict(model, val_loader, device)
            losses_val = [log_loss(y[val_idx, i], probs[i], labels=list(range(len(CLASSES4)))) for i in range(len(HORIZONS))]
            row["val_log_loss_mean"] = float(np.mean(losses_val))
            for i, h in enumerate(HORIZONS):
                row[f"val_h{h}_log_loss"] = float(losses_val[i])
                row[f"val_h{h}_balanced_accuracy"] = float(balanced_accuracy_score(y[val_idx, i], np.argmax(probs[i], axis=1)))
            if row["val_log_loss_mean"] < best_score:
                best_score = row["val_log_loss_mean"]
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
        hist.append(row)
        print(f"[{MODEL_ID}] epoch={epoch} train_loss={row['train_loss']:.5f} val_log_loss_mean={row.get('val_log_loss_mean', float('nan')):.5f}", flush=True)
        if val_loader is not None and stale >= 2:
            break
    if best is not None:
        model.load_state_dict(best)
    return model, hist


def _selected_cols(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    obj = json.loads(path.read_text(encoding="utf-8"))
    cols = obj.get("selected_feature_cols")
    return list(cols) if cols else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-2024", type=Path, default=DEFAULT_TRAIN_2024)
    parser.add_argument("--predict-2025", type=Path, default=DEFAULT_PREDICT_2025)
    parser.add_argument("--clean-2024", type=Path, default=DEFAULT_CLEAN_2024)
    parser.add_argument("--clean-2025", type=Path, default=DEFAULT_CLEAN_2025)
    parser.add_argument("--selected-report", type=Path, default=DEFAULT_SELECTED_REPORT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--seq-len", type=int, default=72)
    parser.add_argument("--val-start", default="2024-10-01")
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--train-stride", type=int, default=2)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.12)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--max-features", type=int, default=96)
    parser.add_argument("--seed", type=int, default=4617)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_raw = _merge_clean4(_read(args.train_2024), args.clean_2024)
    pred_raw = _merge_clean4(_read(args.predict_2025), args.clean_2025)
    cols = _selected_cols(args.selected_report) or _feature_cols(train_raw, pred_raw, args.max_features)
    labels, label_meta = _labels_multi(train_raw, HORIZONS)
    labeled = train_raw.loc[labels.index].copy().join(labels)
    y = labeled[[f"_label_h{h}" for h in HORIZONS]].astype(int).to_numpy()
    ts = pd.to_datetime(labeled["timestamp"])
    first_val = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(pd.Timestamp(args.val_start))))
    train_end = max(args.seq_len - 1, first_val - max(HORIZONS))
    train_idx = np.arange(args.seq_len - 1, train_end, max(1, args.train_stride), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, first_val), len(labeled), dtype=np.int64)
    x, _, _ = _prepare(labeled, cols, fit_rows=train_idx)
    known_by_h = {h: _known_cov(labeled["timestamp"], h) for h in HORIZONS}
    val_model, val_hist = _fit(x, known_by_h, y, train_idx, val_idx, args, args.seed, device)
    val_probs = _predict(val_model, DataLoader(MultiDS(x, known_by_h, val_idx, y, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    val_reports = {f"h{h}": _eval(y[val_idx, i], val_probs[i]) for i, h in enumerate(HORIZONS)}
    val_avg = np.mean(val_probs, axis=0)
    val_consensus = _eval(y[val_idx, HORIZONS.index(36)], val_avg)
    best_epoch = int(min(val_hist, key=lambda r: r.get("val_log_loss_mean", float("inf")))["epoch"])

    full_labels, full_meta = _labels_multi(train_raw, HORIZONS)
    full = train_raw.loc[full_labels.index].copy().join(full_labels)
    y_full = full[[f"_label_h{h}" for h in HORIZONS]].astype(int).to_numpy()
    x_full, scaler, medians = _prepare(full, cols)
    known_full = {h: _known_cov(full["timestamp"], h) for h in HORIZONS}
    full_idx = np.arange(args.seq_len - 1, len(full), max(1, args.train_stride), dtype=np.int64)
    old_epochs = args.epochs
    args.epochs = max(1, best_epoch)
    final_model, final_hist = _fit(x_full, known_full, y_full, full_idx, None, args, args.seed + 101, device)
    args.epochs = old_epochs

    pred_x, _, _ = _prepare(pred_raw, cols, scaler=scaler, medians=medians)
    known_pred = {h: _known_cov(pred_raw["timestamp"], h) for h in HORIZONS}
    pred_probs = _predict(final_model, DataLoader(MultiDS(pred_x, known_pred, np.arange(len(pred_raw)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    pred_avg = np.mean(pred_probs, axis=0)
    full_probs = _predict(final_model, DataLoader(MultiDS(x_full, known_full, np.arange(len(full)), None, args.seq_len), batch_size=args.batch_size * 2, shuffle=False), device)
    full_avg = np.mean(full_probs, axis=0)

    pred_sidecar = args.out_dir / f"{args.predict_2025.stem}_regime4_pred_tft_multihorizon.csv"
    train_sidecar = args.out_dir / f"{args.train_2024.stem}_regime4_pred_tft_multihorizon.csv"
    model_path = args.out_dir / "regime4_pred_tft_multihorizon_2024.pt"
    _output(pred_raw["timestamp"], pred_avg).to_csv(pred_sidecar, index=False)
    _output(full["timestamp"], full_avg).to_csv(train_sidecar, index=False)
    torch.save({"model_id": MODEL_ID, "classes": CLASSES4, "horizons": HORIZONS, "feature_cols": cols, "feature_medians": medians.to_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_, "state_dict": {k: v.detach().cpu() for k, v in final_model.state_dict().items()}}, model_path)
    report = {
        "model_id": MODEL_ID,
        "model_path": str(model_path),
        "classes": CLASSES4,
        "horizons": HORIZONS,
        "feature_count": len(cols),
        "feature_cols": cols,
        "selected_report": str(args.selected_report),
        "validation_label_meta": {**label_meta, "split_policy": "Q4_2024_validation_with_max_horizon_embargo", "embargo_bars": max(HORIZONS)},
        "validation_training_history": val_hist,
        "validation_by_horizon": val_reports,
        "validation_avg_prob_against_h36": val_consensus,
        "selected_final_epochs": int(best_epoch),
        "final_label_meta": full_meta,
        "final_training_history": final_hist,
        "train_sidecar": str(train_sidecar),
        "predict_sidecar": str(pred_sidecar),
        "predict_probability_sum_min": float(pred_avg.sum(axis=1).min()),
        "predict_probability_sum_max": float(pred_avg.sum(axis=1).max()),
        "predict_counts": {CLASSES4[i]: int((np.argmax(pred_avg, axis=1) == i).sum()) for i in range(len(CLASSES4))},
        "predict_confidence_mean": float(_output(pred_raw["timestamp"], pred_avg)[f"{PRED_PREFIX}confidence"].mean()),
        "predict_entropy_mean": float(_output(pred_raw["timestamp"], pred_avg)[f"{PRED_PREFIX}entropy"].mean()),
        "notes": ["Sidecar emits averaged 12/36/72 probabilities; horizon-specific validation metrics are in this report."],
    }
    args.report.write_text(json.dumps(report, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] feature_count={len(cols)} horizons={HORIZONS}", flush=True)
    print(f"[{MODEL_ID}] predict_sidecar={pred_sidecar}", flush=True)
    print(f"[{MODEL_ID}] report={args.report}", flush=True)


if __name__ == "__main__":
    main()
