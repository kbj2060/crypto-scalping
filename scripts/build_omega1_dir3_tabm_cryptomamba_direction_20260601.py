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
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_omega1_dir3_cryptomamba_direction_20260531 import (
    CLASS_NAMES,
    DEFAULT_LABEL_DIR,
    DEFAULT_SPLIT_DIR,
    CMBlock,
    SeqDataset,
    _add_label,
    _add_rolling_features,
    _class_weights,
    _eval,
    _feature_cols,
    _is_forbidden,
    _json_default,
    _prepare,
    _predict,
    _read,
)


MODEL_ID = "omega1_dir3_tabm_cryptomamba_direction_20260601"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/omega1_dir3_tabm_cryptomamba_20260601"
DEFAULT_REPORT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_dir3_tabm_cryptomamba_20260601"


class TabMLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ensemble_size: int = 5) -> None:
        super().__init__()
        self.ensemble_size = int(ensemble_size)
        self.weight = nn.Parameter(torch.empty(int(out_features), int(in_features)))
        self.bias = nn.Parameter(torch.empty(int(out_features)))
        self.r = nn.Parameter(torch.empty(self.ensemble_size, int(in_features)))
        self.s = nn.Parameter(torch.empty(self.ensemble_size, int(out_features)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        nn.init.uniform_(self.bias, -bound, bound)
        nn.init.normal_(self.r, mean=1.0, std=0.05)
        nn.init.normal_(self.s, mean=1.0, std=0.05)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            xk = x.unsqueeze(0) * self.r[:, None, None, :]
            out = F.linear(xk, self.weight, self.bias) * self.s[:, None, None, :]
            return out
        if x.ndim == 4:
            out = F.linear(x * self.r[:, None, None, :], self.weight, self.bias) * self.s[:, None, None, :]
            return out
        raise ValueError(f"TabMLinear expects [B,T,D] or [K,B,T,D], got {tuple(x.shape)}")


class TabMFrontEnd(nn.Module):
    def __init__(self, n_features: int, d_model: int, ensemble_size: int, dropout: float) -> None:
        super().__init__()
        self.tabm1 = TabMLinear(int(n_features), int(d_model), int(ensemble_size))
        self.tabm2 = TabMLinear(int(d_model), int(d_model), int(ensemble_size))
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tabm1(x)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.tabm2(z)
        return z.mean(dim=0)


class CBlock(nn.Module):
    def __init__(self, d_model: int, n_cmblocks: int, seq_len_in: int, seq_len_out: int, d_state: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([CMBlock(d_model, d_state, d_conv=4, expand=2) for _ in range(int(n_cmblocks))])
        self.seq_proj = nn.Linear(int(seq_len_in), int(seq_len_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.seq_proj(x.permute(0, 2, 1)).permute(0, 2, 1)


class TabMCryptoMambaDirection(nn.Module):
    def __init__(
        self,
        n_features: int,
        seq_len: int,
        d_model: int,
        n_cblocks: int,
        n_cmblocks: int,
        d_state: int,
        dropout: float,
        ensemble_size: int,
    ) -> None:
        super().__init__()
        self.frontend = TabMFrontEnd(n_features, d_model, ensemble_size, dropout)
        seq_lens = [int(seq_len)]
        for _ in range(int(n_cblocks)):
            seq_lens.append(max(seq_lens[-1] * 3 // 4, 8))
        self.cblocks = nn.ModuleList(
            [CBlock(d_model, n_cmblocks, seq_lens[i], seq_lens[i + 1], d_state=d_state) for i in range(int(n_cblocks))]
        )
        self.merge = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model) * int(n_cblocks), int(d_model)),
            nn.GELU(),
            nn.LayerNorm(int(d_model)),
        )
        self.head = nn.Sequential(
            nn.Dropout(float(dropout)),
            nn.Linear(int(d_model), 64),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(64, len(CLASS_NAMES)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.frontend(x)
        outs: list[torch.Tensor] = []
        for block in self.cblocks:
            z = block(z)
            outs.append(z[:, -1, :])
        return self.head(self.merge(torch.cat(outs, dim=-1)))


def _fit(x: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray, args: argparse.Namespace) -> tuple[nn.Module, list[dict[str, Any]], torch.device]:
    if not torch.cuda.is_available() and not args.cpu:
        raise RuntimeError("CUDA is unavailable; pass --cpu explicitly for a slow CPU run")
    device = torch.device("cpu" if args.cpu else "cuda")
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    model = TabMCryptoMambaDirection(
        x.shape[1],
        args.seq_len,
        args.d_model,
        args.cblocks,
        args.cmblocks,
        args.d_state,
        args.dropout,
        args.ensemble_size,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loss_fn = nn.CrossEntropyLoss(weight=_class_weights(y, train_idx, device), label_smoothing=0.02)
    loader = DataLoader(SeqDataset(x, y, train_idx, args.seq_len), batch_size=int(args.batch_size), shuffle=True, num_workers=0, pin_memory=(device.type == "cuda"))
    best_state: dict[str, torch.Tensor] | None = None
    best = -1.0
    bad = 0
    history: list[dict[str, Any]] = []
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        val_proba = _predict(model, x, val_idx, args.seq_len, args.batch_size * 2, device)
        ev = _eval(y[val_idx], val_proba)
        row = {"epoch": int(epoch), "loss": float(np.mean(losses)), "val": ev}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False, default=_json_default), flush=True)
        score = float(ev["balanced_accuracy"])
        if score > best + 1e-4:
            best = score
            bad = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= int(args.patience):
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, device


def _output(frame: pd.DataFrame, idx: np.ndarray, proba: np.ndarray) -> pd.DataFrame:
    out = pd.DataFrame({"timestamp": frame["timestamp"].reset_index(drop=True)})
    cols = [
        "dir3_tabm_cmamba_h6_fl_prob",
        "dir3_tabm_cmamba_h6_up_prob",
        "dir3_tabm_cmamba_h6_dn_prob",
        "dir3_tabm_cmamba_h6_confidence",
        "dir3_tabm_cmamba_h6_side_edge",
        "dir3_tabm_cmamba_h6_trade_prob",
    ]
    for col in cols:
        out[col] = np.nan
    out.loc[idx, "dir3_tabm_cmamba_h6_fl_prob"] = proba[:, 0]
    out.loc[idx, "dir3_tabm_cmamba_h6_up_prob"] = proba[:, 1]
    out.loc[idx, "dir3_tabm_cmamba_h6_dn_prob"] = proba[:, 2]
    out.loc[idx, "dir3_tabm_cmamba_h6_confidence"] = proba.max(axis=1)
    out.loc[idx, "dir3_tabm_cmamba_h6_side_edge"] = proba[:, 1] - proba[:, 2]
    out.loc[idx, "dir3_tabm_cmamba_h6_trade_prob"] = proba[:, 1] + proba[:, 2]
    return out


def _write_features(out_dir: Path, year: int, out: pd.DataFrame) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    name = (
        f"training_features_{year}_omega1_dir3_tabm_cryptomamba_20260601.csv"
        if int(year) != 2026
        else "training_features_2026_rebuilt_omega1_dir3_tabm_cryptomamba_20260601.csv"
    )
    path = out_dir / name
    out.to_csv(path, index=False)
    return path


def main() -> int:
    p = argparse.ArgumentParser(description="Train TabM-fronted CryptoMamba direction sidecar on ZigZag action labels.")
    p.add_argument("--split-dir", type=Path, default=DEFAULT_SPLIT_DIR)
    p.add_argument("--label-dir", type=Path, default=DEFAULT_LABEL_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    p.add_argument("--seq-len", type=int, default=60)
    p.add_argument("--val-start", default="2024-10-01")
    p.add_argument("--train-stride", type=int, default=2)
    p.add_argument("--max-features", type=int, default=200)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--d-state", type=int, default=32)
    p.add_argument("--cblocks", type=int, default=4)
    p.add_argument("--cmblocks", type=int, default=2)
    p.add_argument("--ensemble-size", type=int, default=5)
    p.add_argument("--dropout", type=float, default=0.10)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=20260601)
    p.add_argument("--cpu", action="store_true")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    frames = {year: _add_label(_add_rolling_features(_read(args.split_dir, year)), args.label_dir, year) for year in [2024, 2025, 2026]}
    cols = _feature_cols([frames[2024], frames[2025], frames[2026]], args.max_features)
    y = frames[2024]["zigzag_action"].astype(int).to_numpy()
    ts = pd.to_datetime(frames[2024]["timestamp"])
    val_start = pd.Timestamp(args.val_start)
    val_start_idx = int(np.searchsorted(ts.to_numpy(dtype="datetime64[ns]"), np.datetime64(val_start)))
    train_idx = np.arange(args.seq_len - 1, val_start_idx, max(1, int(args.train_stride)), dtype=np.int64)
    val_idx = np.arange(max(args.seq_len - 1, val_start_idx), len(frames[2024]), dtype=np.int64)
    x_train, x_frames, scaler, medians = _prepare(frames[2024], [frames[2025], frames[2026]], cols, train_idx)
    model, history, device = _fit(x_train, y, train_idx, val_idx, args)
    torch.save(
        {
            "model_id": MODEL_ID,
            "classes": CLASS_NAMES,
            "feature_cols": cols,
            "feature_medians": medians.to_dict(),
            "scaler_mean": scaler.mean_,
            "scaler_scale": scaler.scale_,
            "seq_len": int(args.seq_len),
            "ensemble_size": int(args.ensemble_size),
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        },
        args.out_dir / "dir3_tabm_cryptomamba_direction.pt",
    )

    outputs: dict[str, Any] = {}
    for year, x in zip([2025, 2026], x_frames):
        frame = frames[year]
        idx = np.arange(args.seq_len - 1, len(frame), dtype=np.int64)
        proba = _predict(model, x, idx, args.seq_len, args.batch_size * 2, device)
        out = _output(frame, idx, proba)
        path = _write_features(args.out_dir, year, out)
        outputs[str(year)] = {
            "path": str(path),
            "metrics": _eval(frame["zigzag_action"].astype(int).to_numpy()[idx], proba),
        }
    val_proba = _predict(model, x_train, val_idx, args.seq_len, args.batch_size * 2, device)
    report = {
        "model_id": MODEL_ID,
        "architecture": {
            "type": "TabM-fronted CryptoMamba C-Block Merge",
            "tabm_frontend": "BatchEnsemble input projection with shared W and per-expert r/s vectors, averaged before Mamba",
            "seq_len": int(args.seq_len),
            "d_model": int(args.d_model),
            "d_state": int(args.d_state),
            "cblocks": int(args.cblocks),
            "cmblocks": int(args.cmblocks),
            "ensemble_size": int(args.ensemble_size),
        },
        "train_year": 2024,
        "selection_year": 2025,
        "oos_year": 2026,
        "label_source": "zigzag_action",
        "feature_count": int(len(cols)),
        "feature_cols": cols,
        "history": history,
        "internal_validation": _eval(y[val_idx], val_proba),
        "outputs": outputs,
        "contract": {
            "forbidden_inputs": ["teacher_*", "a5dir_*", "Regime4", "regime3_pred_*", "regime3_cmamba_*", "label/target/future/PnL/action_score", "same-level dir3 outputs"],
            "forbidden_feature_hits": [col for col in cols if _is_forbidden(col)],
            "notes": ["TabM is tested as a CryptoMamba frontend only; active HGB MoE is not replaced.", "Uses current/past rows only; scaler/median fitted on 2024 train split only."],
        },
    }
    report_path = args.report_dir / "dir3_tabm_cryptomamba_audit.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(report_path), "outputs": outputs, "internal_validation": report["internal_validation"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
