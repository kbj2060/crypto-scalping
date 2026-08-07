#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_quality_net_return_regression_20260624"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
DEFAULT_DIRECTION_LABEL_DIR = (
    ROOT
    / "tmp/causal_regen_20260516/zigzag_multithreshold_horizon_20260624_t003_008_015_h24_48_96_q067/direction_medium"
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class QualityRegressorTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: parent.ThreeHeadConfig = parent.CFG) -> None:
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
        self.out = nn.Linear(int(cfg.hidden), 1)

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.encode(x)).squeeze(-1)


class ThreeHeadQualityRegTabM(QualityRegressorTabM):
    def __init__(self, n_features: int, *, cfg: parent.ThreeHeadConfig = parent.CFG) -> None:
        super().__init__(n_features, cfg=cfg)
        hidden = int(cfg.hidden)
        self.direction_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.exit_head = nn.Linear(hidden, 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # type: ignore[override]
        h = self.encode(x)
        return {
            "direction": self.direction_head(h),
            "quality_reg": self.quality_head(h).squeeze(-1),
            "exit": self.exit_head(h),
        }


def _target_distribution(name: str, target: np.ndarray, weight: np.ndarray) -> dict[str, Any]:
    active = np.asarray(weight, dtype=np.float64) > 0.0
    vals = np.asarray(target, dtype=np.float64)[active]
    if len(vals) == 0:
        return {"name": name, "active_rows": 0}
    return {
        "name": name,
        "active_rows": int(len(vals)),
        "positive_rate": float(np.mean(vals > 0.0)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
        "p01": float(np.quantile(vals, 0.01)),
        "p10": float(np.quantile(vals, 0.10)),
        "p50": float(np.quantile(vals, 0.50)),
        "p90": float(np.quantile(vals, 0.90)),
        "p99": float(np.quantile(vals, 0.99)),
    }


def _quality_net_return_target(frame: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    target = np.zeros(len(frame), dtype=np.float32)
    weight = np.zeros(len(frame), dtype=np.float32)
    reason_counts: dict[str, int] = {}
    active = 0
    filled = 0
    for i in range(0, len(frame) - 2):
        a = int(action[i])
        if a not in (1, 2):
            continue
        active += 1
        side = 1 if a == 1 else -1
        ok, entry_price, entry_fee, route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not ok:
            target[i] = -abs(float(stop_loss))
            weight[i] = 1.0
            reason_counts[str(route)] = reason_counts.get(str(route), 0) + 1
            continue
        filled += 1
        entry_i = min(int(i) + 1, len(frame) - 1)
        cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
        net, _final_i, reason = exit_head._continue_to_barrier_net(
            arrays,
            start_i=entry_i,
            side=side,
            entry_price=float(entry_price),
            cash_after_entry_fee=cash_after_entry_fee,
            notional=notional,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
        )
        target[i] = np.float32(float(net))
        weight[i] = 1.0
        reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    diag = {
        "active_rows": int(active),
        "filled_entries": int(filled),
        "fill_rate": float(filled / max(active, 1)),
        "exit_reasons": reason_counts,
        "target": _target_distribution("quality_net_return", target, weight),
        "risk_template": {
            "notional": float(notional),
            "take_profit": float(take_profit),
            "stop_loss": float(stop_loss),
            "cost_mult": float(cost_mult),
        },
    }
    return target, weight, diag


def _fit_quality_only(
    x: pd.DataFrame,
    y_pct: np.ndarray,
    w: np.ndarray,
    *,
    seed: int,
    epochs: int,
    huber_delta_pct: float,
    device: torch.device,
    out_path: Path,
) -> dict[str, Any]:
    active = np.asarray(w, dtype=np.float32) > 0.0
    if int(active.sum()) < 128:
        raise RuntimeError(f"quality-only regression has too few active rows: {int(active.sum())}")
    x_active = x.loc[active].reset_index(drop=True)
    y_active = np.asarray(y_pct, dtype=np.float32)[active]
    x_np, scaler = parent._standardize_fit(x_active)
    n = len(y_active)
    split = max(int(n * 0.85), min(n - 1, 128))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model = QualityRegressorTabM(x_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_active[train_idx]))
    dl = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb).mean(dim=1)
            loss = torch.nn.functional.huber_loss(pred, yb, reduction="mean", delta=float(huber_delta_pct))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vy = torch.from_numpy(y_active[val_idx]).to(device)
            vpred = model(vx).mean(dim=1)
            vloss = float(torch.nn.functional.huber_loss(vpred, vy, reduction="mean", delta=float(huber_delta_pct)).detach().cpu())
        if vloss + 1.0e-8 < best_loss:
            best_loss = vloss
            best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best is not None:
        model.load_state_dict(best)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "epochs_ran": int(last_epoch),
        "best_validation_loss": float(best_loss),
    }
    torch.save(payload, out_path)
    return payload


@torch.no_grad()
def _predict_quality_only(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> np.ndarray:
    model = QualityRegressorTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    pred = model(torch.from_numpy(x_np).to(device)).mean(dim=1)
    return pred.detach().cpu().numpy().astype(np.float64)


def _regression_metrics(name: str, y_true_pct: np.ndarray, y_pred_pct: np.ndarray, w: np.ndarray) -> dict[str, Any]:
    active = np.asarray(w, dtype=np.float64) > 0.0
    y = np.asarray(y_true_pct, dtype=np.float64)[active]
    p = np.asarray(y_pred_pct, dtype=np.float64)[active]
    if len(y) == 0:
        return {"name": name, "active_rows": 0}
    corr = float(pd.Series(y).corr(pd.Series(p))) if len(y) > 1 else 0.0
    rank_corr = float(pd.Series(y).corr(pd.Series(p), method="spearman")) if len(y) > 1 else 0.0
    return {
        "name": name,
        "active_rows": int(len(y)),
        "mae_pct": float(np.mean(np.abs(p - y))),
        "rmse_pct": float(np.sqrt(np.mean((p - y) ** 2))),
        "corr": corr if np.isfinite(corr) else 0.0,
        "spearman": rank_corr if np.isfinite(rank_corr) else 0.0,
        "pred_mean_pct": float(np.mean(p)),
        "true_mean_pct": float(np.mean(y)),
        "pred_positive_rate": float(np.mean(p > 0.0)),
        "true_positive_rate": float(np.mean(y > 0.0)),
    }


def _fit_expert_mtl(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_quality_pct: np.ndarray,
    quality_weight: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    seed: int,
    epochs: int,
    quality_loss_weight: float,
    exit_loss_weight: float,
    huber_delta_pct: float,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_quality_np = np.asarray(y_quality_pct, dtype=np.float32)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    q_w = np.asarray(quality_weight, dtype=np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(q_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid MTL sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = ThreeHeadQualityRegTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_quality_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(q_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    last_losses: dict[str, float] = {}
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        loss_parts: list[tuple[float, float, float]] = []
        for xb, yb, yqb, wb, qwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yqb = yqb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            qwb = qwb.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True)
            ye = ye.to(device, non_blocking=True)
            we = we.to(device, non_blocking=True)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3),
                yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            q_pred = out_dir["quality_reg"].mean(dim=1)
            loss_q = torch.nn.functional.huber_loss(q_pred, yqb, reduction="none", delta=float(huber_delta_pct))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(parent.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_q * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(quality_loss_weight) * loss_qual + float(exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            loss_parts.append((float(loss_dir.detach().cpu()), float(loss_qual.detach().cpu()), float(loss_exit.detach().cpu())))
        if loss_parts:
            arr = np.asarray(loss_parts, dtype=np.float64)
            last_losses = {"direction": float(arr[:, 0].mean()), "quality_reg": float(arr[:, 1].mean()), "exit": float(arr[:, 2].mean())}
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_quality_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(q_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vq = torch.nn.functional.huber_loss(vo["quality_reg"].mean(dim=1), vqy, reduction="none", delta=float(huber_delta_pct))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(quality_loss_weight) * ((vq * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                    + float(exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
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
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": parent.CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_dir.columns),
        "quality_target": "exit_replay_net_return_pct",
        "loss_weights": {"direction": 1.0, "quality_reg": float(quality_loss_weight), "exit": float(exit_loss_weight)},
        "last_train_losses": last_losses,
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_expert(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = ThreeHeadQualityRegTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    out = model(torch.from_numpy(x_np).to(device))
    return {
        "direction": torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy().astype(np.float64),
        "quality_reg_pct": out["quality_reg"].mean(dim=1).detach().cpu().numpy().astype(np.float64),
        "exit": torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy().astype(np.float64),
    }


def _routed_scalar(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray, key: str) -> np.ndarray:
    out = np.zeros(len(route), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][key][mask]
    return out


def _prediction_output_reg_quality(frame: pd.DataFrame, direction: np.ndarray, quality_pct: np.ndarray, *, threshold: float, prefix: str) -> pd.DataFrame:
    route = hard._route_id(frame)
    direction_action = np.argmax(direction, axis=1).astype(np.int64)
    quality_net = np.asarray(quality_pct, dtype=np.float64) / 100.0
    final_action = direction_action.copy()
    final_action[(direction_action != 0) & (quality_net < float(threshold))] = 0
    quality_cash = np.zeros(len(frame), dtype=np.float64)
    quality_long = np.where(direction_action == 1, quality_net, 0.0)
    quality_short = np.where(direction_action == 2, quality_net, 0.0)
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_router_expert": np.asarray(hard.EXPERT_NAMES, dtype=object)[route],
            f"{prefix}_router_confidence": hard._route_conf(frame),
            f"{prefix}_router_margin": pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64),
            f"{prefix}_dir_p_cash": direction[:, 0],
            f"{prefix}_dir_p_long": direction[:, 1],
            f"{prefix}_dir_p_short": direction[:, 2],
            f"{prefix}_dir_confidence": np.max(direction, axis=1),
            f"{prefix}_dir_side_edge": direction[:, 1] - direction[:, 2],
            f"{prefix}_dir_trade_prob": direction[:, 1] + direction[:, 2],
            f"{prefix}_dir_action": direction_action,
            f"{prefix}_quality_p_cash": quality_cash,
            f"{prefix}_quality_p_long": quality_long,
            f"{prefix}_quality_p_short": quality_short,
            f"{prefix}_quality_for_action": quality_net,
            f"{prefix}_quality_threshold": float(threshold),
            f"{prefix}_final_action": final_action,
            f"{prefix}_quality_reg_net_return": quality_net,
        }
    )


def _metric_row(split: str, metrics: dict[str, Any], threshold: float) -> dict[str, Any]:
    return {
        f"{split}_pnl": float(metrics["pnl"]),
        f"{split}_mdd": float(metrics["mdd"]),
        f"{split}_trades": int(metrics["trades"]),
        f"{split}_wr": float(metrics["wr"]),
        "quality_threshold": float(threshold),
    }


def _tag_for_threshold(q: float) -> str:
    if q < 0:
        return f"qneg{int(round(abs(q) * 1000)):03d}"
    return f"q{int(round(q * 1000)):03d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=DEFAULT_DIRECTION_LABEL_DIR)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--max-exit-samples", type=int, default=15000)
    ap.add_argument("--quality-thresholds", default="-0.002,0.000,0.001,0.002,0.004,0.006,0.008")
    ap.add_argument("--save-quality-threshold", type=float, default=0.0)
    ap.add_argument("--quality-loss-weight", type=float, default=0.20)
    ap.add_argument("--exit-loss-weight", type=float, default=1.15)
    ap.add_argument("--huber-delta-pct", type=float, default=1.0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260624)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--out-suffix", default="multithreshold_medium_clean_net_return_e2_train15k_exit15k")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}" if str(args.out_suffix).strip() else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    print("stage=build_quality_net_return_targets", flush=True)
    train_target, train_qw, train_target_diag = _quality_net_return_target(train_raw, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    val_target, val_qw, val_target_diag = _quality_net_return_target(val_raw, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_target, oos_qw, oos_target_diag = _quality_net_return_target(oos_raw, fee=fee, slip=slip, cost_mult=float(args.cost_mult))

    x_train_all = parent._base_input(train_raw, base_cols)
    y_dir_all = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train_all.iloc[:limit].reset_index(drop=True)
        y_dir = y_dir_all[:limit]
        y_quality_pct = train_target[:limit].astype(np.float32) * 100.0
        q_weight = train_qw[:limit].astype(np.float32)
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
    else:
        x_train = x_train_all
        y_dir = y_dir_all
        y_quality_pct = train_target.astype(np.float32) * 100.0
        q_weight = train_qw.astype(np.float32)
        train_fit_frame = train_raw

    print("stage=train_quality_only_regression", flush=True)
    quality_only = _fit_quality_only(
        x_train,
        y_quality_pct,
        q_weight,
        seed=int(args.seed),
        epochs=int(args.epochs),
        huber_delta_pct=float(args.huber_delta_pct),
        device=device,
        out_path=out_dir / "quality_only_tabm.pt",
    )
    val_qonly_pred = _predict_quality_only(quality_only, parent._base_input(val_raw, base_cols), device=device)
    oos_qonly_pred = _predict_quality_only(quality_only, parent._base_input(oos_raw, base_cols), device=device)
    quality_only_metrics = {
        "validation": _regression_metrics("quality_only_validation", val_target * 100.0, val_qonly_pred, val_qw),
        "oos": _regression_metrics("quality_only_oos", oos_target * 100.0, oos_qonly_pred, oos_qw),
    }

    print("stage=build_exit_dataset", flush=True)
    x_exit_raw, y_exit, frame_exit, exit_diag = omega4._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_samples=int(args.max_exit_samples),
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    print("stage=train_mtl_quality_regression", flush=True)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_mtl(
            x_train,
            y_dir,
            y_quality_pct,
            q_weight,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            quality_loss_weight=float(args.quality_loss_weight),
            exit_loss_weight=float(args.exit_loss_weight),
            huber_delta_pct=float(args.huber_delta_pct),
            device=device,
            model_path=out_dir / "models" / f"{expert}_3head_quality_reg_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_3head_quality_reg_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
            "last_train_losses": payload["last_train_losses"],
        }

    def predict_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_expert(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality_pct = _routed_scalar(preds, route, "quality_reg_pct")
        return x, direction, quality_pct

    x_val, val_direction, val_quality_pct = predict_all(val_raw)
    x_oos, oos_direction, oos_quality_pct = predict_all(oos_raw)
    mtl_quality_metrics = {
        "validation": _regression_metrics("mtl_quality_validation", val_target * 100.0, val_quality_pct, val_qw),
        "oos": _regression_metrics("mtl_quality_oos", oos_target * 100.0, oos_quality_pct, oos_qw),
    }

    q_values = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    saved_predictions: dict[str, str] = {}
    for q in q_values:
        tag = _tag_for_threshold(float(q))
        val_src = _prediction_output_reg_quality(val_raw, val_direction, val_quality_pct, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
        oos_src_oof = _prediction_output_reg_quality(oos_raw, oos_direction, oos_quality_pct, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
        oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
        val_dec = parent._to_decisions(val_src, oof=True)
        oos_dec = parent._to_decisions(oos_src, oof=False)
        val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        key = tag
        results[key] = {"validation": val_m, "oos": oos_m}
        row = {"variant": key}
        row.update(_metric_row("validation", val_m, q))
        row.update(_metric_row("oos", oos_m, q))
        rows.append(row)
        if abs(float(q) - float(args.save_quality_threshold)) < 1.0e-12:
            train_x, train_direction, train_quality_pct = predict_all(train_raw)
            train_src = _prediction_output_reg_quality(train_raw, train_direction, train_quality_pct, threshold=float(q), prefix="omega1_regime3_expertdq_oof")
            train_src.to_csv(out_dir / f"train_predictions_{tag}.csv", index=False)
            val_src.to_csv(out_dir / f"validation_predictions_{tag}.csv", index=False)
            oos_src.to_csv(out_dir / f"oos_predictions_{tag}.csv", index=False)
            train_dec = parent._to_decisions(train_src, oof=True)
            train_m = omega._metrics(train_raw, train_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
            results[key]["train"] = train_m
            saved_predictions = {
                f"train_{tag}": str(out_dir / f"train_predictions_{tag}.csv"),
                f"validation_{tag}": str(out_dir / f"validation_predictions_{tag}.csv"),
                f"oos_{tag}": str(out_dir / f"oos_predictions_{tag}.csv"),
            }
    ranking = pd.DataFrame(rows).sort_values(["validation_pnl", "validation_mdd", "oos_pnl"], ascending=[False, False, False])
    ranking.to_csv(out_dir / "quality_threshold_ranking.csv", index=False)
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__}, out_dir / "quality_regression_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "base_model": "omega4_4_topdown_style_experiment",
        "design": "Direction CE + clean exit-replay net-return Quality regression + Exit CE. No zigzag diagnostic/lookahead columns are used as model features.",
        "direction_label_dir": str(args.direction_label_dir),
        "input_contract": {
            "base_feature_count": len(base_cols),
            "position_feature_count": len(parent.POS_COLS),
            "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
            "no_label_diagnostic_features": True,
            "remaining_swing_room_feature_present": False,
        },
        "quality_target_contract": {
            "target": "exit_replay_net_return",
            "train_loss_units": "percentage_points",
            "huber_delta_pct": float(args.huber_delta_pct),
            "quality_loss_weight": float(args.quality_loss_weight),
            "cash_rows_quality_weight": 0.0,
            "active_rows_quality_weight": 1.0,
        },
        "target_distribution": {
            "train_raw": train_target_diag,
            "validation": val_target_diag,
            "oos": oos_target_diag,
            "train_fit": _target_distribution("train_fit", y_quality_pct / 100.0, q_weight),
        },
        "quality_only_metrics": quality_only_metrics,
        "mtl_quality_metrics": mtl_quality_metrics,
        "exit_label": {"mode": "entry_label_terminal_giveback", "diag": exit_diag},
        "summaries": summaries,
        "results": results,
        "ranking_by_validation_pnl": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "quality_threshold_ranking.csv"),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "quality_regression_bundle.pt"),
            **saved_predictions,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "top": ranking.head(5).to_dict(orient="records"), "quality_only": quality_only_metrics, "mtl_quality": mtl_quality_metrics}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
