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

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_omega1_regime3_routed_expert_direction_quality_20260602 as cat_dq  # noqa: E402


MODEL_ID = "omega1_2_true_3head_tabm_20260603"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")

POS_COLS = [
    "pos_side",
    "pos_hold_bars",
    "pos_unrealized",
    "pos_mfe",
    "pos_mae",
    "pos_giveback",
    "pos_dist_to_tp",
    "pos_dist_to_sl",
    "pos_notional",
    "pos_leverage",
    "pos_exposure",
    "pos_tp",
    "pos_sl",
]


@dataclass(frozen=True)
class ThreeHeadConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8
    exit_loss_weight: float = 1.15
    quality_loss_weight: float = 0.80


CFG = ThreeHeadConfig()


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


class ThreeHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: ThreeHeadConfig = CFG) -> None:
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
        self.exit_head = nn.Linear(int(cfg.hidden), 2)

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
            "exit": self.exit_head(h),
        }


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 3-head training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("3-head TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 3-head inference matrix")
    return out.astype(np.float32)


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    values = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return values


def _base_input(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    x = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in POS_COLS:
        x[col] = 0.0
    return x.astype(np.float32)


def _exit_input_from_position_rows(x_exit: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    data: dict[str, Any] = {}
    for col in base_cols:
        cur = f"cur_{col}"
        data[col] = pd.to_numeric(x_exit[cur], errors="coerce").to_numpy(dtype=np.float32) if cur in x_exit.columns else np.zeros(len(x_exit), dtype=np.float32)
    for col in POS_COLS:
        data[col] = pd.to_numeric(x_exit[col], errors="coerce").to_numpy(dtype=np.float32) if col in x_exit.columns else np.zeros(len(x_exit), dtype=np.float32)
    out = pd.DataFrame(data, index=x_exit.index)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _prepare_frames(*, disable_tp_sl: bool) -> dict[str, Any]:
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    train_all, eval_df, overlay_report = omega._load_omega_frames()
    feature_cols = omega._numeric_feature_cols(train_all, eval_df)
    label_2025 = hard._build_frame(2025)[["timestamp", "zigzag_action"]]
    label_2026 = hard._build_frame(2026)[["timestamp", "zigzag_action"]]
    train_all, train_labels = omega._align(train_all, label_2025, "omega train labels")
    eval_df, eval_labels = omega._align(eval_df, label_2026, "omega oos labels")
    train_all = train_all.copy()
    eval_df = eval_df.copy()
    train_all["zigzag_action"] = pd.to_numeric(train_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    eval_df["zigzag_action"] = pd.to_numeric(eval_labels["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    train_raw = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    # Build baseline decisions only for lifecycle exit-label generation.
    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, train_src = omega._align(train_raw, tabm_2025, "train")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    if disable_tp_sl:
        train_fixed = exit_head._disable_tp_sl(train_fixed)
    s_train_label = _base_input(train_df, feature_cols)
    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "train_df": train_df,
        "train_fixed": train_fixed,
        "s_train_label": s_train_label,
        "feature_cols": feature_cols,
        "overlay_report": overlay_report,
    }


def _fit_expert_3head(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
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
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = _standardize_fit(x_all)
    x_dir_np = _standardize_apply(x_dir, scaler)
    x_exit_np = _standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = _route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = _route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = dir_w.copy()
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 3-head sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = ThreeHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(torch.from_numpy(x_dir_np[train_idx]), torch.from_numpy(y_dir_np[train_idx]), torch.from_numpy(dir_w[train_idx]))
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, wb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True)
            ye = ye.to(device, non_blocking=True)
            we = we.to(device, non_blocking=True)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_qual_k = torch.nn.functional.cross_entropy(
                out_dir["quality"].reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
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
    model = ThreeHeadTabM(int(payload["n_features"]), cfg=CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": [], "exit": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _load_payloads(payloads: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[ThreeHeadTabM, dict[str, Any]]]:
    loaded: dict[str, tuple[ThreeHeadTabM, dict[str, Any]]] = {}
    for expert, payload in payloads.items():
        model = ThreeHeadTabM(int(payload["n_features"]), cfg=CFG).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[expert] = (model, payload["scaler"])
    return loaded


@torch.no_grad()
def _predict_loaded_exit(model: ThreeHeadTabM, scaler: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> np.ndarray:
    x_np = _standardize_apply(x, scaler)
    probs = torch.softmax(model(torch.from_numpy(x_np).to(device))["exit"], dim=-1).mean(dim=1)
    return probs.detach().cpu().numpy().astype(np.float64)


def _routed(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray, head: str, n_classes: int) -> np.ndarray:
    out = np.zeros((len(route), n_classes), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][head][mask]
    return out


def _prediction_output(frame: pd.DataFrame, direction: np.ndarray, quality: np.ndarray, *, threshold: float, prefix: str) -> pd.DataFrame:
    return cat_dq._prediction_output(frame, direction, quality, threshold=threshold, prefix=prefix)


def _to_decisions(src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    return omega._to_fixed_decisions(src, oof=oof)


def _metrics_with_shared_exit(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[ThreeHeadTabM, dict[str, Any]]],
    *,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    entry_base: pd.Series | None = None
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
    reasons: dict[str, int] = {}
    route = hard._route_id(frame)
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
        if pos != 0 and entry_base is not None:
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "stop_loss"
            else:
                xrow = base_x.iloc[[i]].copy().reset_index(drop=True)
                hold = max(int(i) - int(entry_i), 0)
                giveback = (float(mfe) - float(unreal)) / max(abs(float(mfe)), 1e-8) if mfe > 0 else 0.0
                vals = {
                    "pos_side": float(pos),
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
                for col, val in vals.items():
                    xrow[col] = val
                expert = hard.EXPERT_NAMES[int(route[i])]
                model, scaler = loaded_models[expert]
                prob = float(_predict_loaded_exit(model, scaler, xrow, device=device)[0, 1])
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
                entry_base = None
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
        entry_base = base_x.iloc[int(i)]
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        leverage = float(row.get("leverage", 1.0) or 1.0)
        take_profit = float(row.get("take_profit", 0.0) or 0.0)
        stop_loss = float(row.get("stop_loss", 0.0) or 0.0)
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        mfe = 0.0
        mae = 0.0
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--thresholds", default="0.45,0.50,0.60,0.70,0.80,0.90")
    ap.add_argument("--quality-threshold", type=float, default=0.45)
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--disable-tp-sl", action="store_true")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260603)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames(disable_tp_sl=bool(args.disable_tp_sl))
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_train = _base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    if int(args.max_train_rows) > 0:
        x_train = x_train.iloc[: int(args.max_train_rows)].reset_index(drop=True)
        y_train = y_train[: int(args.max_train_rows)]
        train_fit_frame = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True)
    else:
        train_fit_frame = train_raw

    hold_offsets = [int(x.strip()) for x in str(args.exit_hold_offsets).split(",") if x.strip()]
    x_exit_raw, y_exit, frame_exit, exit_diag = exit_head._build_exit_dataset_independent(
        frames["train_df"],
        frames["s_train_label"],
        frames["train_fixed"],
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        exit_edge_min=float(args.exit_edge_min),
        hold_offsets=hold_offsets,
        max_samples=int(args.max_exit_samples),
    )
    x_exit = _exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_3head(
            x_train,
            y_train,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_3head_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_3head_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }
    loaded_models = _load_payloads(models, device=device)

    def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, np.ndarray]]]:
        x = _base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = _routed(preds, route, "direction", 3)
        quality = _routed(preds, route, "quality", 3)
        out = _prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix="omega1_regime3_expertdq_oof")
        return x, out, preds

    x_val, val_src, _ = predict_frame(val_raw)
    x_oos, oos_src_oof_prefix, _ = predict_frame(oos_raw)
    oos_src = oos_src_oof_prefix.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof_prefix.columns})
    val_dec = _to_decisions(val_src, oof=True)
    oos_dec = _to_decisions(oos_src, oof=False)
    if bool(args.disable_tp_sl):
        val_dec = exit_head._disable_tp_sl(val_dec)
        oos_dec = exit_head._disable_tp_sl(oos_dec)
    thresholds = [float(x.strip()) for x in str(args.thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    base_val = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    base_oos = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    reports["no_exit_head"] = {"validation": base_val, "oos": base_oos}
    for thr in thresholds:
        val = _metrics_with_shared_exit(val_raw, x_val, val_dec, loaded_models, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        oos = _metrics_with_shared_exit(oos_raw, x_oos, oos_dec, loaded_models, threshold=thr, fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
        name = f"exit_thr_{thr:.2f}".replace(".", "p")
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
    val_src.to_csv(out_dir / "validation_predictions_2025_true3head.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_true3head.csv", index=False)
    pd.DataFrame(rows).to_csv(out_dir / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "True 3-head TabM per bull/bear/chop expert. Direction, Quality, and Exit share one BatchEnsemble TabM encoder. Direction/Quality rows use zero position features; Exit rows use the same market columns plus explicit position features.",
        "input_contract": {"base_feature_count": len(base_cols), "position_feature_count": len(POS_COLS), "total_features": len(base_cols) + len(POS_COLS), "position_cols": POS_COLS},
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
        "risk_template": {"max_hold_bars": omega.BASE_TEMPLATE["max_hold"], "cooldown_bars": omega.BASE_TEMPLATE["cooldown"], "tp_sl_disabled": bool(args.disable_tp_sl)},
        "quality_threshold": float(args.quality_threshold),
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "hold_offsets": hold_offsets, "diag": exit_diag},
        "summaries": summaries,
        "results": reports,
        "ranking_by_validation_pnl": rows,
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "ranking.csv"), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": POS_COLS, "config": CFG.__dict__}, out_dir / "true_3head_tabm_bundle.pt")
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": rows[:5], "no_exit_head": reports["no_exit_head"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
