#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_softfloor00_tabm_exit_head_nohold_20260603"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")


@dataclass(frozen=True)
class ExitTabMConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    epochs: int = 42
    patience: int = 8


CFG = ExitTabMConfig()


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    return torch.device("cuda" if (requested == "cuda" or (requested == "auto" and torch.cuda.is_available())) else "cpu")


class ExitTabMClassifier(nn.Module):
    def __init__(self, n_features: int, n_classes: int = 2, *, cfg: ExitTabMConfig = CFG) -> None:
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
        self.out = nn.Linear(int(cfg.hidden), int(n_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xk = x.unsqueeze(1) * self.input_scale.unsqueeze(0) + self.input_bias.unsqueeze(0)
        h = self.in_proj(xk)
        h = self.dropout(torch.nn.functional.silu(self.norms[0](h)))
        for idx, layer in enumerate(self.blocks):
            residual = h
            h = layer(h * self.expert_scale[idx].unsqueeze(0))
            h = self.dropout(torch.nn.functional.silu(self.norms[idx + 1](h)))
            h = h + residual
        return self.out(h)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized exit training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("Exit Head feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized exit inference matrix")
    return out.astype(np.float32)


def _fit_binary_tabm(
    x: pd.DataFrame,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_np, scaler = _standardize_fit(x)
    y_np = np.asarray(y, dtype=np.int64)
    classes = sorted(np.unique(y_np).astype(int).tolist())
    if classes != [0, 1]:
        raise RuntimeError(f"Exit Head needs both classes [0,1], got {classes}")
    weights = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32)
    if sample_weight is not None:
        weights *= np.asarray(sample_weight, dtype=np.float32)
    if not np.isfinite(weights).all() or float(weights.sum()) <= 0.0:
        raise RuntimeError("invalid Exit Head sample weights")

    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    if split >= n:
        split = n
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    model = ExitTabMClassifier(x_np.shape[1], 2, cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds = TensorDataset(torch.from_numpy(x_np[train_idx]), torch.from_numpy(y_np[train_idx]), torch.from_numpy(weights[train_idx]))
    loader = DataLoader(ds, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, wb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            logits = model(xb)
            loss_k = torch.nn.functional.cross_entropy(
                logits.reshape(-1, 2),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss = (loss_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        if len(val_idx):
            model.eval()
            with torch.no_grad():
                vx = torch.from_numpy(x_np[val_idx]).to(device)
                vy = torch.from_numpy(y_np[val_idx]).to(device)
                vw = torch.from_numpy(weights[val_idx]).to(device)
                logits = model(vx)
                loss_k = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, 2),
                    vy[:, None].expand(-1, int(CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(CFG.k))
                val_loss = float(((loss_k.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)).detach().cpu())
        else:
            val_loss = float(loss.detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
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
        "config": CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "n_classes": 2,
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_binary_tabm(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> np.ndarray:
    model = ExitTabMClassifier(int(payload["n_features"]), int(payload["n_classes"]), cfg=CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    out: list[np.ndarray] = []
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        probs = torch.softmax(model(xb), dim=-1).mean(dim=1)
        out.append(probs.detach().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float64)


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    values = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return values


def _fit_exit_heads(x: pd.DataFrame, y: np.ndarray, frame: pd.DataFrame, *, seed: int, epochs: int, device: torch.device, model_dir: Path) -> dict[str, Any]:
    model_dir.mkdir(parents=True, exist_ok=True)
    probs = _route_probs(frame)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        sample_weight = probs[:, idx]
        path = model_dir / f"{expert}_exit_head_tabm.pt"
        payload = _fit_binary_tabm(x.reset_index(drop=True), y, sample_weight=sample_weight, seed=seed + idx, epochs=epochs, device=device, model_path=path)
        models[expert] = payload
        summaries[expert] = {
            "rows": int(len(y)),
            "weight_sum": float(np.asarray(sample_weight, dtype=np.float64).sum()),
            "label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y, minlength=2))},
            "model": str(path),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }
    return {"models": models, "summaries": summaries}


def _routed_exit_prob(models: dict[str, dict[str, Any]], x: pd.DataFrame, frame: pd.DataFrame, *, device: torch.device) -> np.ndarray:
    route = hard._route_id(frame)
    out = np.zeros(len(frame), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if not bool(mask.any()):
            continue
        prob = _predict_binary_tabm(models[expert], x.loc[mask].reset_index(drop=True), device=device)
        out[mask] = prob[:, 1]
    return out


def _load_exit_heads(models: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[ExitTabMClassifier, dict[str, Any]]]:
    loaded: dict[str, tuple[ExitTabMClassifier, dict[str, Any]]] = {}
    for expert, payload in models.items():
        model = ExitTabMClassifier(int(payload["n_features"]), int(payload["n_classes"]), cfg=CFG).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[expert] = (model, payload["scaler"])
    return loaded


@torch.no_grad()
def _predict_loaded_exit_prob(
    loaded: dict[str, tuple[ExitTabMClassifier, dict[str, Any]]],
    x: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    device: torch.device,
) -> np.ndarray:
    route = hard._route_id(frame)
    out = np.zeros(len(frame), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if not bool(mask.any()):
            continue
        model, scaler = loaded[expert]
        x_np = _standardize_apply(x.loc[mask].reset_index(drop=True), scaler)
        probs = torch.softmax(model(torch.from_numpy(x_np).to(device)), dim=-1).mean(dim=1)
        out[mask] = probs[:, 1].detach().cpu().numpy()
    return out


def _exit_fill_net(
    arrays: dict[str, np.ndarray],
    *,
    signal_i: int,
    side: int,
    entry_price: float,
    cash_after_entry_fee: float,
    notional: float,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, int, str]:
    filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(signal_i), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash_after_entry_fee - 1.0, int(signal_i), "exit_unfilled"
    raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    cash = cash_after_entry_fee * (1.0 + raw_exit * notional)
    cash -= cash_after_entry_fee * exit_fee * notional
    return float(cash - 1.0), min(int(signal_i) + 1, len(arrays["open"]) - 1), "model_exit"


def _continue_to_barrier_net(
    arrays: dict[str, np.ndarray],
    *,
    start_i: int,
    side: int,
    entry_price: float,
    cash_after_entry_fee: float,
    notional: float,
    take_profit: float,
    stop_loss: float,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, int, str]:
    end_i = len(arrays["close"]) - 2
    tp = float(take_profit)
    sl = abs(float(stop_loss))
    exit_i = end_i
    reason = "forced_end"
    for j in range(max(0, int(start_i)), end_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        if sl > 0.0 and unreal <= -sl:
            exit_i = int(j)
            reason = "stop_loss"
            break
        if tp > 0.0 and unreal >= tp:
            exit_i = int(j)
            reason = "take_profit"
            break
    if reason == "forced_end":
        exit_px = omega._fill_price(arrays, min(exit_i + 1, len(arrays["open"]) - 1), side, slip_eff, entry=False)
        exit_fee = fee_eff
    else:
        _, exit_px, exit_fee, _route = omega._try_execution(arrays, exit_i, side, entry=False, fee_base=fee_eff, slip_base=slip_eff)
    raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    cash = cash_after_entry_fee * (1.0 + raw_exit * notional)
    cash -= cash_after_entry_fee * exit_fee * notional
    return float(cash - 1.0), int(exit_i), reason


def _position_feature_row(
    state: pd.DataFrame,
    entry_state: pd.Series,
    *,
    row_i: int,
    side: int,
    entry_price: float,
    entry_i: int,
    notional: float,
    leverage: float,
    take_profit: float,
    stop_loss: float,
    mfe: float,
    mae: float,
    unreal: float,
) -> dict[str, float]:
    cur = state.iloc[int(row_i)]
    out: dict[str, float] = {f"cur_{c}": float(cur[c]) for c in state.columns}
    entry_cols = [c for c in state.columns if c.startswith("tabm_") or c.startswith("fixed_")]
    for c in entry_cols:
        out[f"entry_{c}"] = float(entry_state[c])
        out[f"drift_{c}"] = float(cur[c]) - float(entry_state[c])
    hold = max(int(row_i) - int(entry_i), 0)
    giveback = (float(mfe) - float(unreal)) / max(abs(float(mfe)), 1e-8) if mfe > 0 else 0.0
    out.update(
        {
            "pos_side": float(side),
            "pos_hold_bars": float(hold),
            "pos_unrealized": float(unreal),
            "pos_mfe": float(mfe),
            "pos_mae": float(mae),
            "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
            "pos_dist_to_tp": float(take_profit - unreal),
            "pos_dist_to_sl": float(unreal + abs(stop_loss)),
            "pos_notional": float(notional),
            "pos_leverage": float(leverage),
            "pos_exposure": float(notional * leverage),
            "pos_tp": float(take_profit),
            "pos_sl": float(stop_loss),
        }
    )
    return out


def _build_exit_dataset(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    exit_edge_min: float,
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cooldown = 0
    pos = 0
    entry_price = 0.0
    entry_i = 0
    entry_signal_i = 0
    entry_state: pd.Series | None = None
    cash_after_entry_fee = 1.0
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    final_continue_net = 0.0
    final_continue_i = 0
    final_continue_reason = ""
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    reason_counts: dict[str, int] = {}
    exit_positive = 0
    exit_edges: list[float] = []
    for i in range(0, len(frame) - 2):
        if pos != 0 and entry_state is not None:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            row = _position_feature_row(
                state,
                entry_state,
                row_i=i,
                side=pos,
                entry_price=entry_price,
                entry_i=entry_i,
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mfe=mfe,
                mae=mae,
                unreal=unreal,
            )
            exit_now, _, _ = _exit_fill_net(
                arrays,
                signal_i=i,
                side=pos,
                entry_price=entry_price,
                cash_after_entry_fee=cash_after_entry_fee,
                notional=notional,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
            edge = float(exit_now - final_continue_net)
            label = int(edge >= float(exit_edge_min))
            exit_edges.append(edge)
            exit_positive += label
            rows.append(row)
            labels.append(label)
            frame_rows.append(frame.iloc[i])
            reason_counts[final_continue_reason] = reason_counts.get(final_continue_reason, 0) + 1
            if i >= final_continue_i:
                pos = 0
                entry_state = None
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if not bool(active[i]):
            continue
        drow = dec.iloc[i]
        side = int(drow.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = float(px)
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_signal_i = int(i)
        entry_state = state.iloc[int(i)]
        notional = float(drow.get("notional_exposure", 0.0) or 0.0)
        leverage = float(drow.get("leverage", 1.0) or 1.0)
        take_profit = float(drow.get("take_profit", 0.0) or 0.0)
        stop_loss = float(drow.get("stop_loss", 0.0) or 0.0)
        cash_after_entry_fee = 1.0 - 1.0 * entry_fee * notional
        final_continue_net, final_continue_i, final_continue_reason = _continue_to_barrier_net(
            arrays,
            start_i=entry_i,
            side=pos,
            entry_price=entry_price,
            cash_after_entry_fee=cash_after_entry_fee,
            notional=notional,
            take_profit=take_profit,
            stop_loss=stop_loss,
            fee_eff=fee_eff,
            slip_eff=slip_eff,
        )
        mfe = 0.0
        mae = 0.0
    if not rows:
        raise RuntimeError("empty Exit Head lifecycle dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(exit_positive),
        "negative_count": int(len(y) - exit_positive),
        "exit_edge_mean": float(np.mean(exit_edges)) if exit_edges else 0.0,
        "exit_edge_p50": float(np.quantile(exit_edges, 0.50)) if exit_edges else 0.0,
        "exit_edge_p90": float(np.quantile(exit_edges, 0.90)) if exit_edges else 0.0,
        "exit_edge_p99": float(np.quantile(exit_edges, 0.99)) if exit_edges else 0.0,
        "continued_exit_reasons": reason_counts,
        "entry_signal_i_last": int(entry_signal_i),
    }


def _build_exit_dataset_independent(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    exit_edge_min: float,
    hold_offsets: list[int],
    max_samples: int,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active_idx = np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - 3))
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    offsets = sorted({int(x) for x in hold_offsets if int(x) >= 1})
    if not offsets:
        raise RuntimeError("hold_offsets must contain at least one positive integer")
    rows: list[dict[str, float]] = []
    labels: list[int] = []
    frame_rows: list[pd.Series] = []
    reason_counts: dict[str, int] = {}
    exit_edges: list[float] = []
    exit_positive = 0
    used_entries = 0
    missed_entries = 0
    for signal_i in active_idx:
        drow = dec.iloc[int(signal_i)]
        side = int(drow.get("side", 0) or 0)
        if side == 0:
            continue
        filled, entry_price, entry_fee, _route = omega._try_execution(arrays, int(signal_i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            missed_entries += 1
            continue
        entry_i = min(int(signal_i) + 1, len(frame) - 1)
        entry_state = state.iloc[int(signal_i)]
        notional = float(drow.get("notional_exposure", 0.0) or 0.0)
        leverage = float(drow.get("leverage", 1.0) or 1.0)
        take_profit = float(drow.get("take_profit", 0.0) or 0.0)
        stop_loss = float(drow.get("stop_loss", 0.0) or 0.0)
        cash_after_entry_fee = 1.0 - 1.0 * entry_fee * notional
        final_continue_net, _final_continue_i, final_continue_reason = _continue_to_barrier_net(
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
        used_entries += 1
        mfe = 0.0
        mae = 0.0
        max_offset = max(offsets)
        end_i = min(entry_i + max_offset, len(frame) - 2)
        cursor = entry_i
        for offset in offsets:
            row_i = min(entry_i + int(offset), len(frame) - 2)
            if row_i < cursor:
                continue
            for j in range(cursor, row_i + 1):
                px = float(arrays["close"][j])
                raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
                unreal_j = raw * notional
                mfe = max(mfe, unreal_j)
                mae = min(mae, unreal_j)
            cursor = row_i + 1
            px = float(arrays["close"][row_i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            row = _position_feature_row(
                state,
                entry_state,
                row_i=row_i,
                side=side,
                entry_price=float(entry_price),
                entry_i=entry_i,
                notional=notional,
                leverage=leverage,
                take_profit=take_profit,
                stop_loss=stop_loss,
                mfe=mfe,
                mae=mae,
                unreal=unreal,
            )
            exit_now, _, _ = _exit_fill_net(
                arrays,
                signal_i=row_i,
                side=side,
                entry_price=float(entry_price),
                cash_after_entry_fee=cash_after_entry_fee,
                notional=notional,
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
            edge = float(exit_now - final_continue_net)
            label = int(edge >= float(exit_edge_min))
            exit_edges.append(edge)
            exit_positive += label
            rows.append(row)
            labels.append(label)
            frame_rows.append(frame.iloc[row_i])
            reason_counts[final_continue_reason] = reason_counts.get(final_continue_reason, 0) + 1
            if max_samples > 0 and len(rows) >= int(max_samples):
                break
            if row_i >= len(frame) - 2:
                break
        if max_samples > 0 and len(rows) >= int(max_samples):
            break
    if not rows:
        raise RuntimeError("empty independent Exit Head lifecycle dataset")
    x = pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = np.asarray(labels, dtype=np.int64)
    f = pd.DataFrame(frame_rows).reset_index(drop=True)
    return x, y, f, {
        "rows": int(len(y)),
        "positive_rate": float(np.mean(y)),
        "positive_count": int(exit_positive),
        "negative_count": int(len(y) - exit_positive),
        "exit_edge_mean": float(np.mean(exit_edges)) if exit_edges else 0.0,
        "exit_edge_p50": float(np.quantile(exit_edges, 0.50)) if exit_edges else 0.0,
        "exit_edge_p90": float(np.quantile(exit_edges, 0.90)) if exit_edges else 0.0,
        "exit_edge_p99": float(np.quantile(exit_edges, 0.99)) if exit_edges else 0.0,
        "continued_exit_reasons": reason_counts,
        "used_entries": int(used_entries),
        "missed_entries": int(missed_entries),
        "hold_offsets": offsets,
        "label_mode": "independent_entry_hold_offsets",
    }


def _metrics_with_exit_head(
    frame: pd.DataFrame,
    state: pd.DataFrame,
    dec: pd.DataFrame,
    models: dict[str, dict[str, Any]],
    *,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    loaded_heads = _load_exit_heads(models, device=device)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_state: pd.Series | None = None
    notional = 0.0
    leverage = 1.0
    take_profit = 0.0
    stop_loss = 0.0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    leverage_sum = 0.0
    reasons: dict[str, int] = {}
    exit_prob_sum = 0.0
    exit_prob_count = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0 and entry_state is not None:
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                x_exit = pd.DataFrame(
                    [
                        _position_feature_row(
                            state,
                            entry_state,
                            row_i=i,
                            side=pos,
                            entry_price=entry_price,
                            entry_i=entry_i,
                            notional=notional,
                            leverage=leverage,
                            take_profit=take_profit,
                            stop_loss=stop_loss,
                            mfe=mfe,
                            mae=mae,
                            unreal=unreal,
                        )
                    ]
                )
                prob = float(_predict_loaded_exit_prob(loaded_heads, x_exit, frame.iloc[[i]].reset_index(drop=True), device=device)[0])
                exit_prob_sum += prob
                exit_prob_count += 1
                if prob >= float(threshold):
                    reason = "exit_head"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                entry_state = None
                continue
        if pos != 0:
            continue
        if not bool(active[i]):
            continue
        row = dec.iloc[i]
        side = int(row.get("side", 0) or 0)
        if side == 0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = side
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        entry_state = state.iloc[int(i)]
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        fill_i = len(frame) - 1
        exit_px = omega._fill_price(arrays, fill_i, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    n_entries = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "trades_per_day": float(trades / max((pd.to_datetime(frame["timestamp"].iloc[-1]) - pd.to_datetime(frame["timestamp"].iloc[0])).total_seconds() / 86400.0, 1e-9)),
        "avg_notional": float(notional_sum / n_entries),
        "avg_leverage": float(leverage_sum / n_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_exit_prob_seen": float(exit_prob_sum / max(exit_prob_count, 1)),
        "exit_reasons": reasons,
    }


def _disable_tp_sl(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    out["take_profit"] = 0.0
    out["stop_loss"] = 0.0
    return out


def _prepare_frames(*, disable_tp_sl: bool) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    tabm_2025 = omega._read(omega.TABM_2025)
    tabm_2026 = omega._read(omega.TABM_2026)
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    train_df, train_src = omega._align(train_raw, tabm_2025, "train")
    val_df, val_src = omega._align(val_raw, tabm_2025, "validation")
    oos_df, oos_src = omega._align(eval_df, tabm_2026, "oos")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    val_fixed = omega._to_fixed_decisions(val_src, oof=True)
    oos_fixed = omega._to_fixed_decisions(oos_src, oof=False)
    if disable_tp_sl:
        train_fixed = _disable_tp_sl(train_fixed)
        val_fixed = _disable_tp_sl(val_fixed)
        oos_fixed = _disable_tp_sl(oos_fixed)
    s_train = omega._build_state_frame(train_df, train_src, train_fixed, oof=True, feature_cols=feature_cols)
    s_val = omega._build_state_frame(val_df, val_src, val_fixed, oof=True, feature_cols=feature_cols)
    s_oos = omega._build_state_frame(oos_df, oos_src, oos_fixed, oof=False, feature_cols=feature_cols)
    return {
        "train_df": train_df,
        "val_df": val_df,
        "oos_df": oos_df,
        "train_fixed": train_fixed,
        "val_fixed": val_fixed,
        "oos_fixed": oos_fixed,
        "s_train": s_train,
        "s_val": s_val,
        "s_oos": s_oos,
        "overlay_report": overlay_report,
        "feature_cols": feature_cols,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=42)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--max-train-samples", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260603)
    ap.add_argument("--disable-tp-sl", action="store_true")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames(disable_tp_sl=bool(args.disable_tp_sl))
    fee, slip = omega._load_fee_slip()
    hold_offsets = [int(x.strip()) for x in str(args.exit_hold_offsets).split(",") if x.strip()]
    if bool(args.disable_tp_sl):
        x_train, y_train, frame_train_exit, exit_data_diag = _build_exit_dataset_independent(
            frames["train_df"],
            frames["s_train"],
            frames["train_fixed"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            exit_edge_min=float(args.exit_edge_min),
            hold_offsets=hold_offsets,
            max_samples=int(args.max_train_samples),
        )
    else:
        x_train, y_train, frame_train_exit, exit_data_diag = _build_exit_dataset(
            frames["train_df"],
            frames["s_train"],
            frames["train_fixed"],
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            exit_edge_min=float(args.exit_edge_min),
            max_samples=int(args.max_train_samples),
        )
    print(
        json.dumps(
            {
                "stage": "exit_head_train_start",
                "model_id": MODEL_ID,
                "device": str(device),
                "exit_features": int(x_train.shape[1]),
                "exit_data_diag": exit_data_diag,
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    bundle = _fit_exit_heads(
        x_train,
        y_train,
        frame_train_exit,
        seed=int(args.seed),
        epochs=int(args.epochs),
        device=device,
        model_dir=out_dir / "exit_head",
    )
    val_base = omega._metrics(frames["val_df"], frames["val_fixed"], fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_base = omega._metrics(frames["oos_df"], frames["oos_fixed"], fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {
        "no_exit_head": {
            "validation": val_base,
            "oos": oos_base,
        }
    }
    for thr in thresholds:
        val = _metrics_with_exit_head(
            frames["val_df"],
            frames["s_val"],
            frames["val_fixed"],
            bundle["models"],
            threshold=thr,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        oos = _metrics_with_exit_head(
            frames["oos_df"],
            frames["s_oos"],
            frames["oos_fixed"],
            bundle["models"],
            threshold=thr,
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            device=device,
        )
        name = f"exit_head_thr_{thr:.2f}".replace(".", "p")
        reports[name] = {"validation": val, "oos": oos}
        rows.append(
            {
                "variant": name,
                "threshold": thr,
                "validation_pnl": val["pnl"],
                "validation_mdd": val["mdd"],
                "validation_wr": val["wr"],
                "validation_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
            }
        )
    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["validation_wr"])), reverse=True)
    ranking = pd.DataFrame(rows)
    ranking.to_csv(out_dir / "ranking.csv", index=False)
    torch.save(
        {
            "model_id": MODEL_ID,
            "feature_columns": list(x_train.columns),
            "exit_head_summaries": bundle["summaries"],
            "exit_head_models": bundle["models"],
            "config": CFG.__dict__,
        },
        out_dir / "exit_head_bundle.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Omega1.2 soft_floor_0p00 TabM ExpertDQ remains the entry Direction/Quality source. max_hold_bars and cooldown_bars are removed. A soft_floor_0p00 routed TabM binary Exit Head is trained on lifecycle rows and called every 5m while a position is open.",
        "tabm_source": {"variant": "soft_floor_0p00", "train_oof": str(omega.TABM_2025), "oos": str(omega.TABM_2026)},
        "risk_template": {
            "take_profit": omega.BASE_TEMPLATE["take_profit"],
            "stop_loss": omega.BASE_TEMPLATE["stop_loss"],
            "leverage": omega.BASE_TEMPLATE["leverage"],
            "notional": omega.BASE_TEMPLATE["notional"],
            "max_hold_bars": omega.BASE_TEMPLATE["max_hold"],
            "cooldown_bars": omega.BASE_TEMPLATE["cooldown"],
            "tp_sl_disabled": bool(args.disable_tp_sl),
            "note": "When tp_sl_disabled=True, TP/SL risk exits are removed and Exit Head is the lifecycle exit owner. Notional/leverage remain accounting exposure constants.",
        },
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS, "entry_feature_count": len(frames["feature_cols"]), "exit_feature_count": int(x_train.shape[1])},
        "exit_label": {
            "rule": "EXIT=1 only if exit_now_net - continue_to_TP_SL_or_end_net >= exit_edge_min",
            "exit_edge_min": float(args.exit_edge_min),
            "future_used_only_for_training_label": True,
            "intent": "cost-aware exit veto; ambiguous lifecycle rows are labeled HOLD to prevent micro-exit churn",
            "hold_offsets": hold_offsets if bool(args.disable_tp_sl) else None,
        },
        "exit_data_diag": exit_data_diag,
        "exit_head_summaries": bundle["summaries"],
        "cost_accounting": {"fee": fee, "slip": slip, "cost_mult": float(args.cost_mult), "entry_exit_notional_fee": True},
        "results": reports,
        "ranking_by_validation_pnl": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "ranking": str(out_dir / "ranking.csv"),
            "report": str(out_dir / "report.json"),
            "model": str(out_dir / "exit_head_bundle.pt"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": rows[:5]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
