#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_delta_risk_heads_20260611 as delta  # noqa: E402
import train_eval_omega1_2_1_independent_risk_heads_20260611 as indep  # noqa: E402
import train_eval_omega1_2_1_tabm_7head_risk_20260611 as seven  # noqa: E402


MODEL_ID = "omega1_2_1_two_backbone_risk_tabm_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class RiskTabMConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8


CFG = RiskTabMConfig()


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    z = (arr - mean) / std
    if not np.isfinite(z).all():
        raise RuntimeError("non-finite standardized risk TabM train matrix")
    return z.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    if list(x.columns) != list(scaler["columns"]):
        raise RuntimeError("risk TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    z = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(z).all():
        raise RuntimeError("non-finite standardized risk TabM inference matrix")
    return z.astype(np.float32)


class RiskTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: RiskTabMConfig = CFG) -> None:
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
        self.tp_head = nn.Linear(int(cfg.hidden), len(delta.TP_MULT))
        self.sl_head = nn.Linear(int(cfg.hidden), len(delta.SL_MULT))
        self.margin_head = nn.Linear(int(cfg.hidden), len(delta.MARGIN_MULT))
        self.leverage_head = nn.Linear(int(cfg.hidden), len(delta.LEVERAGE_MULT))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return h

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "tp": self.tp_head(h),
            "sl": self.sl_head(h),
            "margin": self.margin_head(h),
            "leverage": self.leverage_head(h),
        }


def _soft_targets(labels: dict[str, np.ndarray], n_rows: int) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rows = np.asarray(labels["rows"], dtype=np.int64)
    weights = np.asarray(labels["weight"], dtype=np.float64)
    uniq = np.unique(rows)
    pos = {int(r): i for i, r in enumerate(uniq)}
    targets = {
        "tp": np.zeros((len(uniq), len(delta.TP_MULT)), dtype=np.float32),
        "sl": np.zeros((len(uniq), len(delta.SL_MULT)), dtype=np.float32),
        "margin": np.zeros((len(uniq), len(delta.MARGIN_MULT)), dtype=np.float32),
        "leverage": np.zeros((len(uniq), len(delta.LEVERAGE_MULT)), dtype=np.float32),
    }
    sample_weight = np.zeros(len(uniq), dtype=np.float32)
    for j, row in enumerate(rows):
        i = pos[int(row)]
        w = float(weights[j])
        sample_weight[i] += w
        for head in targets:
            targets[head][i, int(labels[head][j])] += w
    for head, arr in targets.items():
        den = np.clip(arr.sum(axis=1, keepdims=True), 1e-12, None)
        targets[head] = (arr / den).astype(np.float32)
    sample_weight = sample_weight / max(float(sample_weight.mean()), 1e-8)
    return {**targets, "row_index": uniq.astype(np.int64)}, sample_weight.astype(np.float32)


def _soft_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    loss_k = -(target.unsqueeze(1) * logp).sum(dim=-1)
    loss = loss_k.mean(dim=1)
    return (loss * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _fit_risk_tabm(x: pd.DataFrame, targets: dict[str, np.ndarray], weights: np.ndarray, *, seed: int, epochs: int, device: torch.device) -> tuple[RiskTabM, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    x_np, scaler = _standardize_fit(x)
    n = len(x_np)
    split = max(int(n * 0.85), min(n - 1, 64))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    ds = TensorDataset(
        torch.from_numpy(x_np[train_idx]),
        torch.from_numpy(targets["tp"][train_idx]),
        torch.from_numpy(targets["sl"][train_idx]),
        torch.from_numpy(targets["margin"][train_idx]),
        torch.from_numpy(targets["leverage"][train_idx]),
        torch.from_numpy(weights[train_idx]),
    )
    dl = DataLoader(ds, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    model = RiskTabM(x_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, ytp, ysl, ym, yl, wb in dl:
            xb = xb.to(device, non_blocking=True)
            ytp = ytp.to(device, non_blocking=True)
            ysl = ysl.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            yl = yl.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            out = model(xb)
            loss = (
                _soft_ce(out["tp"], ytp, wb)
                + _soft_ce(out["sl"], ysl, wb)
                + _soft_ce(out["margin"], ym, wb)
                + _soft_ce(out["leverage"], yl, wb)
            ) / 4.0
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            wb = torch.from_numpy(weights[val_idx]).to(device)
            out = model(vx)
            vloss = (
                _soft_ce(out["tp"], torch.from_numpy(targets["tp"][val_idx]).to(device), wb)
                + _soft_ce(out["sl"], torch.from_numpy(targets["sl"][val_idx]).to(device), wb)
                + _soft_ce(out["margin"], torch.from_numpy(targets["margin"][val_idx]).to(device), wb)
                + _soft_ce(out["leverage"], torch.from_numpy(targets["leverage"][val_idx]).to(device), wb)
            ) / 4.0
            val_loss = float(vloss.detach().cpu())
        if val_loss + 1e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    meta = {"config": CFG.__dict__, "scaler": scaler, "epochs_ran": int(last_epoch), "best_validation_loss": float(best_loss), "n_features": int(x_np.shape[1])}
    return model, meta


@torch.no_grad()
def _predict_risk(model: RiskTabM, meta: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    x_np = _standardize_apply(x, meta["scaler"])
    chunks = {k: [] for k in ("tp", "sl", "margin", "leverage")}
    model.eval()
    for start in range(0, len(x_np), 8192):
        out = model(torch.from_numpy(x_np[start : start + 8192]).to(device))
        for key, logits in out.items():
            chunks[key].append(torch.softmax(logits, dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _apply_risk_tabm(dec: pd.DataFrame, probs: dict[str, np.ndarray]) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = pd.to_numeric(out["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    idx = np.flatnonzero(active)
    if len(idx) == 0:
        return out
    pred = {k: np.argmax(v[idx], axis=1).astype(np.int64) for k, v in probs.items()}
    for j, frame_idx in enumerate(idx):
        tp, sl, margin, lev, _hold = delta._risk_values(
            out.iloc[int(frame_idx)],
            (int(pred["tp"][j]), int(pred["sl"][j]), int(pred["margin"][j]), int(pred["leverage"][j]), 0),
        )
        out.loc[int(frame_idx), "take_profit"] = tp
        out.loc[int(frame_idx), "stop_loss"] = sl
        out.loc[int(frame_idx), "position_fraction"] = margin
        out.loc[int(frame_idx), "leverage"] = lev
        out.loc[int(frame_idx), "notional_exposure"] = margin * lev
        out.loc[int(frame_idx), "max_hold_bars"] = 0
        out.loc[int(frame_idx), "cooldown_bars"] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live-threshold", type=float, default=0.80)
    ap.add_argument("--candidate-threshold", type=float, default=0.50)
    ap.add_argument("--risk-label-max-rows", type=int, default=1200)
    ap.add_argument("--soft-top-k", type=int, default=5)
    ap.add_argument("--soft-temp", type=float, default=0.025)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    delta.MAX_HOLD_BUCKETS = np.asarray([0], dtype=np.int64)
    device = _device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = seven._prepare_frames()
    fee, slip = omega._load_fee_slip()
    bundle = indep._load_parent_bundle(device)
    base_cols = list(bundle["base_cols"])
    train = frames["train_raw"]
    val = frames["val_raw"]
    oos = frames["oos_raw"]
    prefix = "omega1_regime3_expertdq"

    train_cand_parent = indep._predict_parent(train, bundle, threshold=float(args.candidate_threshold), device=device, prefix=prefix)
    val_parent = indep._predict_parent(val, bundle, threshold=float(args.live_threshold), device=device, prefix=prefix)
    oos_parent = indep._predict_parent(oos, bundle, threshold=float(args.live_threshold), device=device, prefix=prefix)
    train_cand_dec = indep._parent_to_decisions(train_cand_parent, prefix=prefix)
    val_dec_base = indep._parent_to_decisions(val_parent, prefix=prefix)
    oos_dec_base = indep._parent_to_decisions(oos_parent, prefix=prefix)

    labels, label_diag = delta._delta_risk_soft_dataset(
        train,
        train_cand_dec,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.risk_label_max_rows),
        top_k=int(args.soft_top_k),
        temp=float(args.soft_temp),
    )
    x_train_all = indep._risk_feature_frame(train, base_cols, train_cand_parent, prefix=prefix)
    targets, weights = _soft_targets(labels, len(train))
    train_rows = np.asarray(targets["row_index"], dtype=np.int64)
    x_train = x_train_all.iloc[train_rows].reset_index(drop=True)
    target_only = {k: v for k, v in targets.items() if k != "row_index"}
    model, meta = _fit_risk_tabm(x_train, target_only, weights, seed=int(args.seed), epochs=int(args.epochs), device=device)
    x_val = indep._risk_feature_frame(val, base_cols, val_parent, prefix=prefix)
    x_oos = indep._risk_feature_frame(oos, base_cols, oos_parent, prefix=prefix)
    val_probs = _predict_risk(model, meta, x_val, device=device)
    oos_probs = _predict_risk(model, meta, x_oos, device=device)
    val_dec_risk = _apply_risk_tabm(val_dec_base, val_probs)
    oos_dec_risk = _apply_risk_tabm(oos_dec_base, oos_probs)

    base_val = omega._metrics(val, val_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    base_oos = omega._metrics(oos, oos_dec_base, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    risk_val = omega._metrics(val, val_dec_risk, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    risk_oos = omega._metrics(oos, oos_dec_risk, fee=fee, slip=slip, cost_mult=float(args.cost_mult))

    payload = {
        "model_id": MODEL_ID,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "meta": meta,
        "feature_cols": list(x_train_all.columns),
        "delta_buckets": {
            "tp_mult": delta.TP_MULT.tolist(),
            "sl_mult": delta.SL_MULT.tolist(),
            "margin_mult": delta.MARGIN_MULT.tolist(),
            "leverage_mult": delta.LEVERAGE_MULT.tolist(),
            "max_hold": [0],
        },
    }
    torch.save(payload, OUT_DIR / "risk_tabm_backbone.pt")
    joblib.dump({"feature_cols": list(x_train_all.columns), "label_diag": label_diag}, OUT_DIR / "risk_tabm_meta.joblib")
    val_dec_base.to_csv(OUT_DIR / "validation_base_decisions.csv", index=False)
    oos_dec_base.to_csv(OUT_DIR / "oos_base_decisions.csv", index=False)
    val_dec_risk.to_csv(OUT_DIR / "validation_two_backbone_risk_decisions.csv", index=False)
    oos_dec_risk.to_csv(OUT_DIR / "oos_two_backbone_risk_decisions.csv", index=False)
    ranking = pd.DataFrame(
        [
            {"variant": "fixed_template", "split": "validation", **base_val},
            {"variant": "fixed_template", "split": "oos", **base_oos},
            {"variant": "two_backbone_risk_tabm", "split": "validation", **risk_val},
            {"variant": "two_backbone_risk_tabm", "split": "oos", **risk_oos},
        ]
    )
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Two-backbone Omega1.2. Existing regime-routed D/Q TabM experts are frozen; global risk-only TabM backbone predicts baseline-relative TP/SL/margin/leverage delta buckets. Exit head is not used. max_hold fixed at 0.",
        "architecture": {
            "dq_backbones": "3 frozen expert TabM backbones: bull, bear, chop. Each has direction and quality heads from parent bundle.",
            "risk_backbone": "1 global TabM backbone conditioned on Omega features + D/Q parent outputs + router one-hot.",
            "risk_heads": ["tp_delta", "sl_delta", "margin_delta", "leverage_delta"],
        },
        "thresholds": {"live": float(args.live_threshold), "candidate": float(args.candidate_threshold)},
        "label_diag": label_diag,
        "risk_training": {"rows": int(len(x_train)), "epochs_ran": int(meta["epochs_ran"]), "best_validation_loss": float(meta["best_validation_loss"])},
        "results": {
            "fixed_template": {"validation": base_val, "oos": base_oos},
            "two_backbone_risk_tabm": {"validation": risk_val, "oos": risk_oos},
        },
        "bucket_summary": {
            "validation_base": indep._bucket_summary(val_dec_base),
            "oos_base": indep._bucket_summary(oos_dec_base),
            "validation_two_backbone": indep._bucket_summary(val_dec_risk),
            "oos_two_backbone": indep._bucket_summary(oos_dec_risk),
        },
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "model": str(OUT_DIR / "risk_tabm_backbone.pt"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "results": report["results"], "bucket_summary": report["bucket_summary"], "risk_training": report["risk_training"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
