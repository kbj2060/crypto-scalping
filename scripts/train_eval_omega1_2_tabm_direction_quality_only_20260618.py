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
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_true_2head_tabm_no_exit_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class TwoHeadConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8
    quality_loss_weight: float = 0.80


CFG = TwoHeadConfig()


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
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


class TwoHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: TwoHeadConfig = CFG) -> None:
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
        self.direction_head = nn.Linear(int(cfg.hidden), 3)
        self.quality_head = nn.Linear(int(cfg.hidden), 3)

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
            "direction": self.direction_head(h),
            "quality": self.quality_head(h),
        }


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 2-head training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("2-head TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 2-head inference matrix")
    return out.astype(np.float32)


def _fit_expert_2head(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_dir_np, scaler = _standardize_fit(x_dir)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    route_w = threehead._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    if float(dir_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 2-head sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)

    model = TwoHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in dl_dir:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            out = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out["direction"].reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_qual_k = torch.nn.functional.cross_entropy(
                out["quality"].reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vo = model(vx)
            vdir = torch.nn.functional.cross_entropy(
                vo["direction"].reshape(-1, 3),
                vy[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(
                vo["quality"].reshape(-1, 3),
                vy[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                )
                .detach()
                .cpu()
            )
        if vloss + 1.0e-6 < best_loss:
            best_loss = vloss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_dir.columns),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = TwoHeadTabM(int(payload["n_features"]), cfg=CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _metric_row(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidate": candidate,
        **{f"validation_{k}": v for k, v in val_m.items() if k in {"pnl", "mdd", "wr", "trades", "long_entries", "short_entries", "exit_reasons"}},
        **{f"oos_{k}": v for k, v in oos_m.items() if k in {"pnl", "mdd", "wr", "trades", "long_entries", "short_entries", "exit_reasons"}},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--quality-threshold", type=float, default=0.8)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260618)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_train = threehead._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_2head(
            x_train,
            y_train,
            train_raw,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_2head_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_2head_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_frame(frame: pd.DataFrame, *, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = threehead._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = threehead._routed(preds, route, "direction", 3)
        quality = threehead._routed(preds, route, "quality", 3)
        out = threehead._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        return x, out

    _x_val, val_src = predict_frame(val_raw, prefix="omega1_regime3_expertdq_oof")
    _x_oos, oos_src = predict_frame(oos_raw, prefix="omega1_regime3_expertdq")
    val_dec = threehead._to_decisions(val_src, oof=True)
    oos_dec = threehead._to_decisions(oos_src, oof=False)
    val_metrics = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_metrics = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    row = _metric_row("two_head_direction_quality_only_parent", val_metrics, oos_metrics)
    val_src.to_csv(out_dir / "validation_predictions_2025_2head.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_2head.csv", index=False)
    pd.DataFrame([row]).to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Direction+Quality-only TabM per bull/bear/chop expert. Exit head and exit loss are removed; parent-only backtest uses normal TP/SL/max-hold accounting.",
        "input_contract": {"base_feature_count": len(base_cols), "position_feature_count": len(threehead.POS_COLS), "total_features": len(base_cols) + len(threehead.POS_COLS), "position_cols": threehead.POS_COLS},
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
        "risk_template": {"max_hold_bars": omega.BASE_TEMPLATE["max_hold"], "cooldown_bars": omega.BASE_TEMPLATE["cooldown"], "tp_sl_disabled": False},
        "quality_threshold": float(args.quality_threshold),
        "summaries": summaries,
        "results": {"parent_only": {"validation": val_metrics, "oos": oos_metrics}},
        "ranking_by_validation_pnl": [row],
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv"), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": threehead.POS_COLS, "config": CFG.__dict__}, out_dir / "two_head_tabm_bundle.pt")
    print(json.dumps({"report": str(out_dir / "report.json"), "result": row}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
