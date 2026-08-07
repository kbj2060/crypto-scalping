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
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_omega1_regime3_routed_expert_direction_quality_20260602 as cat_dq  # noqa: E402


MODEL_ID = "omega1_2_1_tabm_maxhold_head_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")

CURRENT_THR_MAP = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
EXPERT_SCALES = {"bull": 0.75, "bear": 0.90, "chop": 0.90}
OVERLAY_SCALES = {"bull": 0.65, "bear": 0.90, "chop": 0.90}
BASE_NOTIONAL = 0.45
BASE_LEVERAGE = 2.0
BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
NOTIONAL_CAP = 0.90
MAX_HOLD_BUCKETS = np.asarray([0, 96, 192, 384, 768], dtype=np.int64)


@dataclass(frozen=True)
class FourHeadConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8
    quality_loss_weight: float = 0.80
    exit_loss_weight: float = 1.15
    max_hold_loss_weight: float = 0.45


CFG = FourHeadConfig()


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


class FourHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: FourHeadConfig = CFG) -> None:
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
        self.max_hold_head = nn.Linear(int(cfg.hidden), len(MAX_HOLD_BUCKETS))

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
            "max_hold": self.max_hold_head(h),
        }


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 4-head training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("4-head TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 4-head inference matrix")
    return out.astype(np.float32)


def _base_input(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    return threehead._base_input(frame, base_cols)


def _exit_input_from_position_rows(x_exit: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    return threehead._exit_input_from_position_rows(x_exit, base_cols)


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    return threehead._route_probs(frame)


def _weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, classes: int) -> torch.Tensor:
    loss_k = torch.nn.functional.cross_entropy(
        logits.reshape(-1, classes),
        target[:, None].expand(-1, int(CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(CFG.k))
    return (loss_k.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _prepare_frames(*, disable_tp_sl_for_exit_labels: bool) -> dict[str, Any]:
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

    tabm_2025 = omega._read(omega.TABM_2025)
    train_df, train_src = omega._align(train_raw, tabm_2025, "train")
    train_fixed = omega._to_fixed_decisions(train_src, oof=True)
    if disable_tp_sl_for_exit_labels:
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


def _aggressive_risk_for_expert(expert: str) -> tuple[float, float, float, float]:
    base = BASE_NOTIONAL * float(EXPERT_SCALES[expert]) * (float(OVERLAY_SCALES[expert]) / float(EXPERT_SCALES[expert]))
    margin = min(float(base) * COMPENSATED_SCALE, NOTIONAL_CAP)
    ratio = margin / max(float(base), 1e-12)
    leverage = float(BASE_LEVERAGE)
    exposure = margin * leverage
    barrier_scale = ratio * leverage
    return exposure, margin, BASE_TP * barrier_scale, BASE_SL * barrier_scale


def _simulate_fixed_risk_hold(
    arrays: dict[str, np.ndarray],
    i: int,
    side: int,
    *,
    expert: str,
    max_hold: int,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, dict[str, Any]]:
    exposure, _margin, tp, sl = _aggressive_risk_for_expert(expert)
    filled, entry, entry_fee, _route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled or exposure <= 0.0:
        return -1.0, {"exit_reason": "entry_miss", "net": -1.0}
    entry_i = min(int(i) + 1, len(arrays["close"]) - 1)
    end_i = min(int(i) + int(max_hold), len(arrays["close"]) - 2) if int(max_hold) > 0 else len(arrays["close"]) - 2
    cash = 1.0 - float(entry_fee) * exposure
    exit_fill: float | None = None
    exit_fee = fee_eff
    exit_reason = "forced_end" if int(max_hold) <= 0 else "max_hold"
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, end_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip_eff)) / max(entry, 1e-12)
        unreal = raw * exposure
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        if unreal <= -abs(sl):
            _, exit_fill, exit_fee, _ = omega._try_execution(arrays, int(j), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "stop_loss"
            end_i = j
            break
        if unreal >= float(tp):
            _, exit_fill, exit_fee, _ = omega._try_execution(arrays, int(j), int(side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
            exit_reason = "take_profit"
            end_i = j
            break
    if exit_fill is None:
        exit_fill = omega._fill_price(arrays, min(end_i + 1, len(arrays["close"]) - 1), int(side), slip_eff, entry=False)
    raw_exit = (exit_fill - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_fill) / max(entry, 1e-12)
    before_exit_fee = cash
    cash = cash * (1.0 + raw_exit * exposure)
    cash -= before_exit_fee * float(exit_fee) * exposure
    net = float(cash - 1.0)
    hold_penalty = 0.000010 * max(0, int(end_i) - int(i))
    tail_penalty = 0.16 * max(0.0, -mae - 0.035)
    return float(net - hold_penalty - tail_penalty), {"exit_reason": exit_reason, "net": net, "mfe": mfe, "mae": mae}


def _max_hold_labels(
    frame: pd.DataFrame,
    y: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    n = len(frame)
    labels = np.zeros(n, dtype=np.int64)
    weights = np.zeros(n, dtype=np.float32)
    scores = np.zeros(n, dtype=np.float32)
    route = hard._route_id(frame)
    active_idx = np.flatnonzero(np.asarray(y, dtype=np.int64) != 0)
    active_idx = active_idx[active_idx < n - 3]
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        pick = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[pick]
    reason_counts: dict[str, int] = {}
    for row_num, i in enumerate(active_idx):
        side = 1 if int(y[int(i)]) == 1 else -1
        expert = hard.EXPERT_NAMES[int(route[int(i)])]
        best_i = 0
        best_score = -1e18
        best_meta: dict[str, Any] = {}
        for hold_i, hold in enumerate(MAX_HOLD_BUCKETS):
            score, meta = _simulate_fixed_risk_hold(
                arrays,
                int(i),
                side,
                expert=expert,
                max_hold=int(hold),
                fee_eff=fee_eff,
                slip_eff=slip_eff,
            )
            if score > best_score:
                best_score = score
                best_i = hold_i
                best_meta = meta
        labels[int(i)] = int(best_i)
        scores[int(i)] = float(best_score)
        weights[int(i)] = float(np.clip(1.0 + max(best_score, -0.05) * 8.0, 0.20, 4.0))
        reason = str(best_meta.get("exit_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (row_num + 1) % 1000 == 0:
            print(json.dumps({"maxhold_label_progress": int(row_num + 1), "total": int(len(active_idx))}), flush=True)
    diag = {
        "active_labeled_rows": int(len(active_idx)),
        "label_exit_reasons": reason_counts,
        "max_hold_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels[active_idx], minlength=len(MAX_HOLD_BUCKETS)))},
        "max_hold_buckets": MAX_HOLD_BUCKETS.tolist(),
    }
    return {"max_hold": labels, "max_hold_weight": weights, "max_hold_score": scores}, diag


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    hold_labels: dict[str, np.ndarray],
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
    y_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = _route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = _route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32) * route_w
    qual_w = dir_w.copy()
    exit_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    hold_w = np.asarray(hold_labels["max_hold_weight"], dtype=np.float32) * route_w
    if float(dir_w.sum()) <= 0.0 or float(exit_w.sum()) <= 0.0 or float(hold_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 4-head sample weights")
    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_np[train_idx]),
        torch.from_numpy(np.asarray(hold_labels["max_hold"], dtype=np.int64)[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        torch.from_numpy(hold_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(exit_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    model = FourHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yh, wd, wq, wh in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            yh = yh.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)
            wq = wq.to(device, non_blocking=True)
            wh = wh.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True)
            ye = ye.to(device, non_blocking=True)
            we = we.to(device, non_blocking=True)
            out = model(xb)
            out_exit = model(xe)
            loss = (
                _weighted_ce(out["direction"], yb, wd, 3)
                + float(CFG.quality_loss_weight) * _weighted_ce(out["quality"], yb, wq, 3)
                + float(CFG.exit_loss_weight) * _weighted_ce(out_exit["exit"], ye, we, 2)
                + float(CFG.max_hold_loss_weight) * _weighted_ce(out["max_hold"], yh, wh, len(MAX_HOLD_BUCKETS))
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_np[val_idx]).to(device)
            vh = torch.from_numpy(np.asarray(hold_labels["max_hold"], dtype=np.int64)[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vhw = torch.from_numpy(hold_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(exit_w[exit_val_idx]).to(device)
            out = model(vx)
            out_exit = model(ve)
            vloss = (
                _weighted_ce(out["direction"], vy, vw, 3)
                + float(CFG.quality_loss_weight) * _weighted_ce(out["quality"], vy, vw, 3)
                + float(CFG.exit_loss_weight) * _weighted_ce(out_exit["exit"], vey, vew, 2)
                + float(CFG.max_hold_loss_weight) * _weighted_ce(out["max_hold"], vh, vhw, len(MAX_HOLD_BUCKETS))
            )
            val_loss = float(vloss.detach().cpu())
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
    model = FourHeadTabM(int(payload["n_features"]), cfg=FourHeadConfig(**dict(payload["config"]))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    chunks = {k: [] for k in ("direction", "quality", "exit", "max_hold")}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        for key, logits in out.items():
            chunks[key].append(torch.softmax(logits, dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _routed(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray, head: str, n_classes: int) -> np.ndarray:
    out = np.zeros((len(route), n_classes), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][head][mask]
    return out


def _prediction_output(frame: pd.DataFrame, direction: np.ndarray, quality: np.ndarray, hold_prob: np.ndarray, *, threshold: float, prefix: str) -> pd.DataFrame:
    out = cat_dq._prediction_output(frame, direction, quality, threshold=threshold, prefix=prefix)
    hold_id = np.argmax(hold_prob, axis=1)
    out[f"{prefix}_max_hold_id"] = hold_id
    out[f"{prefix}_max_hold_bars"] = MAX_HOLD_BUCKETS[hold_id]
    return out


def _to_decisions(src: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    base = src.rename(columns={c: c.replace(f"{prefix}_", "omega1_regime3_expertdq_") for c in src.columns})
    dec = omega._to_fixed_decisions(base, oof=False)
    active = pd.to_numeric(dec["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    dec["max_hold_bars"] = pd.to_numeric(src[f"{prefix}_max_hold_bars"], errors="raise").astype(int)
    dec.loc[~active, "max_hold_bars"] = 0
    return dec


def _apply_aggressive_template(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
    for expert in hard.EXPERT_NAMES:
        mask = (pd.to_numeric(out["action"], errors="raise") != 0) & (out["router_expert"].astype(str) == expert)
        exposure, margin, tp, sl = _aggressive_risk_for_expert(expert)
        out.loc[mask, "notional_exposure"] = float(exposure)
        out.loc[mask, "position_fraction"] = float(margin)
        out.loc[mask, "leverage"] = float(BASE_LEVERAGE)
        out.loc[mask, "take_profit"] = float(tp)
        out.loc[mask, "stop_loss"] = float(sl)
    inactive = pd.to_numeric(out["action"], errors="raise") == 0
    out.loc[inactive, ["notional_exposure", "position_fraction", "take_profit", "stop_loss"]] = 0.0
    out.loc[inactive, "leverage"] = 1.0
    out.loc[inactive, "max_hold_bars"] = 0
    out["cooldown_bars"] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--quality-thresholds", default="0.45,0.50,0.55,0.60,0.64,0.65,0.70,0.72")
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--max-hold-label-rows", type=int, default=12000)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames(disable_tp_sl_for_exit_labels=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_train = _base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    hold_labels, hold_diag = _max_hold_labels(
        train_raw,
        y_train,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.max_hold_label_rows),
    )
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
        payload = _fit_expert(
            x_train,
            y_train,
            train_raw,
            x_exit,
            y_exit,
            frame_exit,
            hold_labels,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_4head_maxhold_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_4head_maxhold_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_frame(frame: pd.DataFrame, *, threshold: float, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = _base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = _routed(preds, route, "direction", 3)
        quality = _routed(preds, route, "quality", 3)
        hold_prob = _routed(preds, route, "max_hold", len(MAX_HOLD_BUCKETS))
        out = _prediction_output(frame, direction, quality, hold_prob, threshold=threshold, prefix=prefix)
        return x, out

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    for thr in thresholds:
        prefix = "omega1_2_1_maxhold"
        _, val_src = predict_frame(val_raw, threshold=thr, prefix=prefix)
        _, oos_src = predict_frame(oos_raw, threshold=thr, prefix=prefix)
        val_dec = _apply_aggressive_template(_to_decisions(val_src, prefix=prefix))
        oos_dec = _apply_aggressive_template(_to_decisions(oos_src, prefix=prefix))
        val = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        name = f"q{thr:.2f}".replace(".", "p")
        reports[name] = {"validation": val, "oos": oos}
        val_src.to_csv(out_dir / f"validation_predictions_{name}.csv", index=False)
        oos_src.to_csv(out_dir / f"oos_predictions_{name}.csv", index=False)
        val_dec.to_csv(out_dir / f"validation_decisions_{name}.csv", index=False)
        oos_dec.to_csv(out_dir / f"oos_decisions_{name}.csv", index=False)
        rows.append(
            {
                "variant": name,
                "quality_threshold": float(thr),
                "validation_pnl": val["pnl"],
                "validation_mdd": val["mdd"],
                "validation_wr": val["wr"],
                "validation_trades": val["trades"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "oos_exit_reasons": json.dumps(oos.get("exit_reasons", {}), sort_keys=True),
            }
        )
    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["validation_wr"])), reverse=True)
    pd.DataFrame(rows).to_csv(out_dir / "ranking.csv", index=False)
    torch.save(
        {"models": models, "base_cols": base_cols, "pos_cols": threehead.POS_COLS, "config": CFG.__dict__, "max_hold_buckets": MAX_HOLD_BUCKETS},
        out_dir / "four_head_maxhold_tabm_bundle.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Regime-routed 4-head TabM per bull/bear/chop expert. Existing direction/quality/exit heads retained; only max_hold head added. TP/SL/margin/leverage stay fixed to Omega1.2.1 true-leverage aggressive template.",
        "input_contract": {
            "base_feature_count": len(base_cols),
            "position_feature_count": len(threehead.POS_COLS),
            "total_features": len(base_cols) + len(threehead.POS_COLS),
        },
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
        "fixed_risk_template": {
            "base_notional": BASE_NOTIONAL,
            "base_leverage": BASE_LEVERAGE,
            "base_take_profit": BASE_TP,
            "base_stop_loss": BASE_SL,
            "compensated_scale": COMPENSATED_SCALE,
            "notional_cap": NOTIONAL_CAP,
            "true_leverage_exposure": True,
            "preserve_price_barrier": True,
        },
        "max_hold_buckets": MAX_HOLD_BUCKETS.tolist(),
        "max_hold_label_diag": hold_diag,
        "exit_label_diag": exit_diag,
        "summaries": summaries,
        "results": reports,
        "ranking_by_validation_pnl": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "bundle": str(out_dir / "four_head_maxhold_tabm_bundle.pt"),
            "ranking": str(out_dir / "ranking.csv"),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": rows[:8]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
