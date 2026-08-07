#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_supervised_risk_selector_20260604 as sup  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_tabm_risk_selector_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RiskTabMConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08


class RiskTabM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, cfg: RiskTabMConfig = RiskTabMConfig()) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.head = nn.Linear(int(cfg.hidden), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return self.head(h).mean(dim=1)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[~np.isfinite(std) | (std < 1e-6)] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite TabM risk training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("TabM risk feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite TabM risk inference matrix")
    return out.astype(np.float32)


def _train_tabm(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    device: torch.device,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> tuple[RiskTabM, dict[str, Any], dict[str, Any]]:
    _seed_everything(seed)
    x_np, scaler = _standardize_fit(x)
    y_np = np.asarray(y, dtype=np.int64)
    counts = np.bincount(y_np, minlength=len(sup.RISK_TEMPLATES)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / np.mean(weights[counts > 0])
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    ds = TensorDataset(torch.tensor(x_np, dtype=torch.float32), torch.tensor(y_np, dtype=torch.long))
    loader = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=False)
    model = RiskTabM(x_np.shape[1], len(sup.RISK_TEMPLATES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    diag: dict[str, Any] = {}
    for epoch in range(int(epochs)):
        model.train()
        losses: list[float] = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            loss = torch.nn.functional.cross_entropy(model(xb), yb, weight=weight_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == int(epochs) - 1 or (epoch + 1) % 10 == 0:
            diag = {"epoch": int(epoch + 1), "loss": float(np.mean(losses)), "label_counts": {str(i): int(v) for i, v in enumerate(counts.astype(int))}}
            print(json.dumps({"stage": "tabm_risk_train", **diag}, ensure_ascii=False), flush=True)
    return model, scaler, diag


@torch.no_grad()
def _predict(model: RiskTabM, x: pd.DataFrame, scaler: dict[str, Any], *, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    arr = _standardize_apply(x, scaler)
    outs: list[np.ndarray] = []
    for start in range(0, len(arr), int(batch_size)):
        xb = torch.tensor(arr[start : start + int(batch_size)], dtype=torch.float32, device=device)
        outs.append(torch.argmax(model(xb), dim=1).cpu().numpy().astype(np.int64))
    pred = np.concatenate(outs) if outs else np.zeros(0, dtype=np.int64)
    if not set(np.unique(pred)).issubset(set(range(len(sup.RISK_TEMPLATES)))):
        raise RuntimeError(f"unexpected risk classes: {sorted(np.unique(pred).tolist())}")
    return pred


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--candidate-deltas", default="-0.30,-0.25,-0.20,-0.15,-0.10")
    ap.add_argument("--candidate-delta-train", type=float, default=-0.25)
    ap.add_argument("--min-score", type=float, default=0.0010)
    ap.add_argument("--max-candidates", type=int, default=6000)
    ap.add_argument("--allow-non-tp-labels", action="store_true")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260605)
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    device = _device(args.device)
    _seed_everything(int(args.seed))
    out_dir = OUT_DIR if not args.out_suffix.strip() else OUT_DIR.parent / f"{MODEL_ID}_{args.out_suffix.strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = tabm._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(sup.ZIGZAG_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    train_x_base, train_src = sup._predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_src = sup._read_predictions(sup.ZIGZAG_DIR / "validation_predictions_2025_true3head.csv", frames["val_raw"])
    oos_src = sup._read_predictions(sup.ZIGZAG_DIR / "oos_predictions_2026_true3head.csv", frames["oos_raw"])
    val_x_base = tabm._base_input(frames["val_raw"], list(bundle["base_cols"]))
    oos_x_base = tabm._base_input(frames["oos_raw"], list(bundle["base_cols"]))
    train_x_risk = sup._risk_features(train_x_base, train_src, oof=True)
    val_x_risk = sup._risk_features(val_x_base, val_src, oof=True)
    oos_x_risk = sup._risk_features(oos_x_base, oos_src, oof=False)

    train_idx, y_risk, label_diag = sup._build_risk_labels(
        frames["train_raw"],
        train_src,
        oof=True,
        candidate_delta=float(args.candidate_delta_train),
        min_score=float(args.min_score),
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_candidates=int(args.max_candidates),
        require_take_profit=not bool(args.allow_non_tp_labels),
    )
    if len(train_idx) < 200:
        raise RuntimeError(f"not enough risk training candidates: {len(train_idx)}")
    x_train = train_x_risk.iloc[train_idx].reset_index(drop=True)
    model, scaler, train_diag = _train_tabm(
        x_train,
        y_risk,
        device=device,
        seed=int(args.seed),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "scaler": scaler,
            "n_features": int(x_train.shape[1]),
            "n_classes": len(sup.RISK_TEMPLATES),
            "risk_templates": sup.RISK_TEMPLATES,
        },
        out_dir / "tabm_risk_selector.pt",
    )
    label_diag.to_csv(out_dir / "risk_label_diagnostics_head.csv", index=False)

    rows: list[dict[str, Any]] = []
    for delta_s in str(args.candidate_deltas).split(","):
        delta = float(delta_s)
        val_action = sup._threshold_action(val_src, oof=True, thresholds=sup._candidate_thresholds(delta))
        oos_action = sup._threshold_action(oos_src, oof=False, thresholds=sup._candidate_thresholds(delta))
        val_candidate = val_action != omega.ACTION_CASH
        oos_candidate = oos_action != omega.ACTION_CASH
        val_risk = np.zeros(len(val_action), dtype=np.int64)
        oos_risk = np.zeros(len(oos_action), dtype=np.int64)
        if bool(val_candidate.any()):
            val_risk[val_candidate] = _predict(model, val_x_risk.loc[val_candidate].reset_index(drop=True), scaler, device=device, batch_size=int(args.batch_size))
        if bool(oos_candidate.any()):
            oos_risk[oos_candidate] = _predict(model, oos_x_risk.loc[oos_candidate].reset_index(drop=True), scaler, device=device, batch_size=int(args.batch_size))
        val_dec = sup._risk_decision(val_src, oof=True, action=val_action, risk_class=val_risk)
        oos_dec = sup._risk_decision(oos_src, oof=False, action=oos_action, risk_class=oos_risk)
        val = omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        rows.append(
            {
                "variant": "tabm_risk_selector",
                "candidate_delta": float(delta),
                "train_candidates": int(len(train_idx)),
                "train_label_cash_rate": float((y_risk == 0).mean()),
                "val_candidate_rows": int(val_candidate.sum()),
                "oos_candidate_rows": int(oos_candidate.sum()),
                "val_pnl": val["pnl"],
                "val_mdd": val["mdd"],
                "val_wr": val["wr"],
                "val_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "val_exit_reasons": val.get("exit_reasons", {}),
                "oos_exit_reasons": oos.get("exit_reasons", {}),
            }
        )
    ranking = pd.DataFrame(rows).sort_values(["val_pnl", "val_wr"], ascending=False)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "TabM finite risk-template selector. Parent fixed Direction/Quality stays frozen; this model selects CASH or TP/SL/notional bucket from the same counterfactual labels as the supervised risk selector.",
        "risk_templates": sup.RISK_TEMPLATES,
        "train": {
            "candidate_delta": float(args.candidate_delta_train),
            "candidates": int(len(train_idx)),
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_risk, minlength=len(sup.RISK_TEMPLATES)))},
            "min_score": float(args.min_score),
            "train_diag": train_diag,
        },
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "model": str(out_dir / "tabm_risk_selector.pt"),
            "label_diag": str(out_dir / "risk_label_diagnostics_head.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.to_string(index=False))
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": str(out_dir / "ranking.csv")}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
