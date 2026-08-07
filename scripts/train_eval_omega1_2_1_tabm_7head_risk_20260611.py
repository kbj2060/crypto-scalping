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
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_omega1_regime3_routed_expert_direction_quality_20260602 as cat_dq  # noqa: E402


MODEL_ID = "omega1_2_1_regime3_7head_tabm_risk_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SPLIT_TS = pd.Timestamp("2025-10-01")

TP_BUCKETS = np.asarray([0.078, 0.092, 0.104, 0.120, 0.140], dtype=np.float64)
SL_BUCKETS = np.asarray([0.042, 0.050, 0.056, 0.066, 0.080], dtype=np.float64)
MARGIN_BUCKETS = np.asarray([0.45, 0.60, 0.75, 0.81, 0.90], dtype=np.float64)
LEVERAGE_BUCKETS = np.asarray([1.0, 1.5, 2.0, 3.0], dtype=np.float64)
MAX_HOLD_BUCKETS = np.asarray([0, 96, 192, 384, 768], dtype=np.int64)


@dataclass(frozen=True)
class SevenHeadConfig:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 8
    quality_loss_weight: float = 0.80
    risk_loss_weight: float = 0.55


CFG = SevenHeadConfig()


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


class SevenHeadTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: SevenHeadConfig = CFG) -> None:
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
        self.tp_head = nn.Linear(int(cfg.hidden), len(TP_BUCKETS))
        self.sl_head = nn.Linear(int(cfg.hidden), len(SL_BUCKETS))
        self.margin_head = nn.Linear(int(cfg.hidden), len(MARGIN_BUCKETS))
        self.leverage_head = nn.Linear(int(cfg.hidden), len(LEVERAGE_BUCKETS))
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
            "tp": self.tp_head(h),
            "sl": self.sl_head(h),
            "margin": self.margin_head(h),
            "leverage": self.leverage_head(h),
            "max_hold": self.max_hold_head(h),
        }


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = (arr - mean) / std
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 7-head training matrix")
    return out.astype(np.float32), {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("7-head TabM feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - scaler["mean"]) / scaler["std"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized 7-head inference matrix")
    return out.astype(np.float32)


def _base_input(frame: pd.DataFrame, base_cols: list[str]) -> pd.DataFrame:
    x = frame.reindex(columns=base_cols).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for col in threehead.POS_COLS:
        x[col] = 0.0
    return x.astype(np.float32)


def _prepare_frames() -> dict[str, Any]:
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
    return {
        "train_raw": train_raw,
        "val_raw": val_raw,
        "oos_raw": eval_df.reset_index(drop=True),
        "feature_cols": feature_cols,
        "overlay_report": overlay_report,
    }


def _simulate_one_risk(
    arrays: dict[str, np.ndarray],
    i: int,
    side: int,
    *,
    tp: float,
    sl: float,
    margin: float,
    leverage: float,
    max_hold: int,
    fee_eff: float,
    slip_eff: float,
) -> tuple[float, dict[str, Any]]:
    effective_exposure = float(margin) * float(leverage)
    filled, entry, entry_fee, entry_route = omega._try_execution(arrays, int(i), int(side), entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled or effective_exposure <= 0.0:
        return -1.0, {"active": 0, "net": -1.0}
    entry_i = min(int(i) + 1, len(arrays["close"]) - 1)
    end_i = min(int(i) + int(max_hold), len(arrays["close"]) - 2) if int(max_hold) > 0 else len(arrays["close"]) - 2
    cash = 1.0 - float(entry_fee) * effective_exposure
    exit_fill: float | None = None
    exit_fee = fee_eff
    exit_reason = "forced_end" if int(max_hold) <= 0 else "max_hold"
    mfe = 0.0
    mae = 0.0
    for j in range(entry_i, end_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip_eff)) / max(entry, 1e-12)
        unreal = raw * effective_exposure
        mfe = max(mfe, unreal)
        mae = min(mae, unreal)
        if unreal <= -abs(float(sl)):
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
    cash = cash * (1.0 + raw_exit * effective_exposure)
    cash -= before_exit_fee * float(exit_fee) * effective_exposure
    net = float(cash - 1.0)
    risk_penalty = 0.18 * max(0.0, -mae - 0.035) + 0.010 * max(0.0, effective_exposure - 1.8)
    hold_penalty = 0.000010 * max(0, int(end_i) - int(i))
    score = float(net - risk_penalty - hold_penalty)
    return score, {
        "active": 1,
        "net": net,
        "mfe": float(mfe),
        "mae": float(mae),
        "exit_reason": exit_reason,
        "entry_route": entry_route,
        "exit_i": int(end_i),
    }


def _risk_labels(
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
    labels = {
        "tp": np.full(n, 2, dtype=np.int64),
        "sl": np.full(n, 2, dtype=np.int64),
        "margin": np.full(n, 3, dtype=np.int64),
        "leverage": np.full(n, 2, dtype=np.int64),
        "max_hold": np.full(n, 0, dtype=np.int64),
        "risk_weight": np.zeros(n, dtype=np.float32),
        "risk_score": np.zeros(n, dtype=np.float32),
    }
    active_idx = np.flatnonzero(np.asarray(y, dtype=np.int64) != 0)
    active_idx = active_idx[active_idx < n - 3]
    if int(max_rows) > 0 and len(active_idx) > int(max_rows):
        # Deterministic stratified thinning keeps full time span while bounding runtime.
        pick = np.linspace(0, len(active_idx) - 1, int(max_rows)).round().astype(np.int64)
        active_idx = active_idx[pick]
    reason_counts: dict[str, int] = {}
    for row_num, i in enumerate(active_idx):
        side = 1 if int(y[int(i)]) == 1 else -1
        best_score = -1e18
        best = (2, 2, 3, 2, 0)
        best_meta: dict[str, Any] = {}
        for tp_i, tp in enumerate(TP_BUCKETS):
            for sl_i, sl in enumerate(SL_BUCKETS):
                for margin_i, margin in enumerate(MARGIN_BUCKETS):
                    for lev_i, lev in enumerate(LEVERAGE_BUCKETS):
                        for hold_i, hold in enumerate(MAX_HOLD_BUCKETS):
                            score, meta = _simulate_one_risk(
                                arrays,
                                int(i),
                                side,
                                tp=float(tp),
                                sl=float(sl),
                                margin=float(margin),
                                leverage=float(lev),
                                max_hold=int(hold),
                                fee_eff=fee_eff,
                                slip_eff=slip_eff,
                            )
                            if score > best_score:
                                best_score = score
                                best = (tp_i, sl_i, margin_i, lev_i, hold_i)
                                best_meta = meta
        labels["tp"][int(i)], labels["sl"][int(i)], labels["margin"][int(i)], labels["leverage"][int(i)], labels["max_hold"][int(i)] = best
        labels["risk_weight"][int(i)] = float(np.clip(1.0 + max(best_score, -0.05) * 8.0, 0.20, 4.0))
        labels["risk_score"][int(i)] = float(best_score)
        reason = str(best_meta.get("exit_reason", "unknown"))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if (row_num + 1) % 1000 == 0:
            print(json.dumps({"risk_label_progress": int(row_num + 1), "total": int(len(active_idx))}), flush=True)
    diag = {
        "active_labeled_rows": int(len(active_idx)),
        "label_exit_reasons": reason_counts,
        "tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["tp"][active_idx], minlength=len(TP_BUCKETS)))},
        "sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["sl"][active_idx], minlength=len(SL_BUCKETS)))},
        "margin_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["margin"][active_idx], minlength=len(MARGIN_BUCKETS)))},
        "leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["leverage"][active_idx], minlength=len(LEVERAGE_BUCKETS)))},
        "max_hold_counts": {str(i): int(v) for i, v in enumerate(np.bincount(labels["max_hold"][active_idx], minlength=len(MAX_HOLD_BUCKETS)))},
    }
    return labels, diag


def _route_probs(frame: pd.DataFrame) -> np.ndarray:
    values = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise RuntimeError("non-finite Regime3 route probabilities")
    return values


def _weighted_ce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, classes: int) -> torch.Tensor:
    loss_k = torch.nn.functional.cross_entropy(
        logits.reshape(-1, classes),
        target[:, None].expand(-1, int(CFG.k)).reshape(-1),
        reduction="none",
    ).reshape(-1, int(CFG.k))
    return (loss_k.mean(dim=1) * weight).sum() / torch.clamp(weight.sum(), min=1.0)


def _fit_expert_7head(
    x: pd.DataFrame,
    y_dir: np.ndarray,
    risk_labels: dict[str, np.ndarray],
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
    x_np, scaler = _standardize_fit(x)
    y_np = np.asarray(y_dir, dtype=np.int64)
    route_w = _route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_np).astype(np.float32) * route_w
    qual_w = dir_w.copy()
    risk_w = np.asarray(risk_labels["risk_weight"], dtype=np.float32) * route_w
    if float(dir_w.sum()) <= 0.0 or float(risk_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 7-head sample weights")
    n = len(y_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    tensors = [
        torch.from_numpy(x_np[train_idx]),
        torch.from_numpy(y_np[train_idx]),
        torch.from_numpy(np.asarray(risk_labels["tp"], dtype=np.int64)[train_idx]),
        torch.from_numpy(np.asarray(risk_labels["sl"], dtype=np.int64)[train_idx]),
        torch.from_numpy(np.asarray(risk_labels["margin"], dtype=np.int64)[train_idx]),
        torch.from_numpy(np.asarray(risk_labels["leverage"], dtype=np.int64)[train_idx]),
        torch.from_numpy(np.asarray(risk_labels["max_hold"], dtype=np.int64)[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        torch.from_numpy(risk_w[train_idx]),
    ]
    dl = DataLoader(TensorDataset(*tensors), batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    model = SevenHeadTabM(x_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, yb, ytp, ysl, ym, yl, yh, wd, wq, wr in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            ytp = ytp.to(device, non_blocking=True)
            ysl = ysl.to(device, non_blocking=True)
            ym = ym.to(device, non_blocking=True)
            yl = yl.to(device, non_blocking=True)
            yh = yh.to(device, non_blocking=True)
            wd = wd.to(device, non_blocking=True)
            wq = wq.to(device, non_blocking=True)
            wr = wr.to(device, non_blocking=True)
            out = model(xb)
            loss_dir = _weighted_ce(out["direction"], yb, wd, 3)
            loss_qual = _weighted_ce(out["quality"], yb, wq, 3)
            risk_loss = (
                _weighted_ce(out["tp"], ytp, wr, len(TP_BUCKETS))
                + _weighted_ce(out["sl"], ysl, wr, len(SL_BUCKETS))
                + _weighted_ce(out["margin"], ym, wr, len(MARGIN_BUCKETS))
                + _weighted_ce(out["leverage"], yl, wr, len(LEVERAGE_BUCKETS))
                + _weighted_ce(out["max_hold"], yh, wr, len(MAX_HOLD_BUCKETS))
            ) / 5.0
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.risk_loss_weight) * risk_loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vy = torch.from_numpy(y_np[val_idx]).to(device)
            wd = torch.from_numpy(dir_w[val_idx]).to(device)
            wr = torch.from_numpy(risk_w[val_idx]).to(device)
            out = model(vx)
            vloss = _weighted_ce(out["direction"], vy, wd, 3)
            if float(wr.sum().detach().cpu()) > 0.0:
                vloss = vloss + float(CFG.risk_loss_weight) * (
                    _weighted_ce(out["tp"], torch.from_numpy(risk_labels["tp"][val_idx]).to(device), wr, len(TP_BUCKETS))
                    + _weighted_ce(out["sl"], torch.from_numpy(risk_labels["sl"][val_idx]).to(device), wr, len(SL_BUCKETS))
                    + _weighted_ce(out["margin"], torch.from_numpy(risk_labels["margin"][val_idx]).to(device), wr, len(MARGIN_BUCKETS))
                    + _weighted_ce(out["leverage"], torch.from_numpy(risk_labels["leverage"][val_idx]).to(device), wr, len(LEVERAGE_BUCKETS))
                    + _weighted_ce(out["max_hold"], torch.from_numpy(risk_labels["max_hold"][val_idx]).to(device), wr, len(MAX_HOLD_BUCKETS))
                ) / 5.0
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
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": CFG.__dict__,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x.columns),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = SevenHeadTabM(int(payload["n_features"]), cfg=SevenHeadConfig(**dict(payload["config"]))).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    chunks = {k: [] for k in ("direction", "quality", "tp", "sl", "margin", "leverage", "max_hold")}
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


def _prediction_output(
    frame: pd.DataFrame,
    direction: np.ndarray,
    quality: np.ndarray,
    risk_probs: dict[str, np.ndarray],
    *,
    threshold: float,
    prefix: str,
) -> pd.DataFrame:
    out = cat_dq._prediction_output(frame, direction, quality, threshold=threshold, prefix=prefix)
    tp_id = np.argmax(risk_probs["tp"], axis=1)
    sl_id = np.argmax(risk_probs["sl"], axis=1)
    margin_id = np.argmax(risk_probs["margin"], axis=1)
    lev_id = np.argmax(risk_probs["leverage"], axis=1)
    hold_id = np.argmax(risk_probs["max_hold"], axis=1)
    margin = MARGIN_BUCKETS[margin_id]
    lev = LEVERAGE_BUCKETS[lev_id]
    out[f"{prefix}_risk_tp_id"] = tp_id
    out[f"{prefix}_risk_sl_id"] = sl_id
    out[f"{prefix}_risk_margin_id"] = margin_id
    out[f"{prefix}_risk_leverage_id"] = lev_id
    out[f"{prefix}_risk_max_hold_id"] = hold_id
    out[f"{prefix}_take_profit"] = TP_BUCKETS[tp_id]
    out[f"{prefix}_stop_loss"] = SL_BUCKETS[sl_id]
    out[f"{prefix}_position_fraction"] = margin
    out[f"{prefix}_leverage"] = lev
    out[f"{prefix}_notional_exposure"] = margin * lev
    out[f"{prefix}_max_hold_bars"] = MAX_HOLD_BUCKETS[hold_id]
    return out


def _to_decisions(src: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    action = pd.to_numeric(src[f"{prefix}_final_action"], errors="raise").astype(int)
    side = np.where(action == 1, 1, np.where(action == 2, -1, 0)).astype(np.int64)
    out = pd.DataFrame(
        {
            "timestamp": src["timestamp"],
            "action": action,
            "side": side,
            "quality_score": pd.to_numeric(src[f"{prefix}_quality_for_action"], errors="raise"),
            "confidence": pd.to_numeric(src[f"{prefix}_dir_confidence"], errors="raise"),
            "notional_exposure": pd.to_numeric(src[f"{prefix}_notional_exposure"], errors="raise"),
            "position_fraction": pd.to_numeric(src[f"{prefix}_position_fraction"], errors="raise"),
            "leverage": pd.to_numeric(src[f"{prefix}_leverage"], errors="raise"),
            "take_profit": pd.to_numeric(src[f"{prefix}_take_profit"], errors="raise"),
            "stop_loss": pd.to_numeric(src[f"{prefix}_stop_loss"], errors="raise"),
            "max_hold_bars": pd.to_numeric(src[f"{prefix}_max_hold_bars"], errors="raise").astype(int),
            "cooldown_bars": np.zeros(len(src), dtype=np.int64),
            "router_expert": src[f"{prefix}_router_expert"].astype(str),
        }
    )
    active = pd.to_numeric(out["action"], errors="raise").to_numpy(dtype=np.int64) != 0
    for col in ("notional_exposure", "position_fraction", "take_profit", "stop_loss"):
        out.loc[~active, col] = 0.0
    out.loc[~active, "leverage"] = 1.0
    out.loc[~active, "max_hold_bars"] = 0
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--quality-thresholds", default="0.45,0.50,0.55,0.60,0.65,0.70")
    ap.add_argument("--risk-label-max-rows", type=int, default=12000)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260611)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--out-suffix", default="")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _prepare_frames()
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    x_train = _base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    risk_labels, risk_diag = _risk_labels(
        train_raw,
        y_train,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_rows=int(args.risk_label_max_rows),
    )
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_7head(
            x_train,
            y_train,
            risk_labels,
            train_raw,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_7head_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(out_dir / "models" / f"{expert}_7head_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_frame(frame: pd.DataFrame, *, threshold: float, prefix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        x = _base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = _routed(preds, route, "direction", 3)
        quality = _routed(preds, route, "quality", 3)
        risk_probs = {
            "tp": _routed(preds, route, "tp", len(TP_BUCKETS)),
            "sl": _routed(preds, route, "sl", len(SL_BUCKETS)),
            "margin": _routed(preds, route, "margin", len(MARGIN_BUCKETS)),
            "leverage": _routed(preds, route, "leverage", len(LEVERAGE_BUCKETS)),
            "max_hold": _routed(preds, route, "max_hold", len(MAX_HOLD_BUCKETS)),
        }
        out = _prediction_output(frame, direction, quality, risk_probs, threshold=threshold, prefix=prefix)
        return x, out

    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for thr in thresholds:
        prefix = "omega1_2_1_7head"
        _, val_src = predict_frame(val_raw, threshold=thr, prefix=prefix)
        _, oos_src = predict_frame(oos_raw, threshold=thr, prefix=prefix)
        val_dec = _to_decisions(val_src, prefix=prefix)
        oos_dec = _to_decisions(oos_src, prefix=prefix)
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
                "validation_avg_notional": val.get("avg_notional", 0.0),
                "validation_avg_leverage": val.get("avg_leverage", 0.0),
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "oos_avg_notional": oos.get("avg_notional", 0.0),
                "oos_avg_leverage": oos.get("avg_leverage", 0.0),
            }
        )
    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["validation_wr"])), reverse=True)
    pd.DataFrame(rows).to_csv(out_dir / "ranking.csv", index=False)
    bundle = {
        "models": models,
        "base_cols": base_cols,
        "pos_cols": threehead.POS_COLS,
        "config": CFG.__dict__,
        "buckets": {
            "tp": TP_BUCKETS,
            "sl": SL_BUCKETS,
            "margin": MARGIN_BUCKETS,
            "leverage": LEVERAGE_BUCKETS,
            "max_hold": MAX_HOLD_BUCKETS,
        },
    }
    torch.save(bundle, out_dir / "regime3_7head_tabm_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "design": "Regime-routed 7-head TabM per bull/bear/chop expert. Exit head removed. Heads: direction, quality, tp, sl, margin_notional, leverage, max_hold.",
        "risk_contract": {
            "notional_head_semantics": "margin_notional bucket",
            "final_notional_exposure": "margin_notional * leverage",
            "tp_sl_semantics": "equity-return barriers",
            "cooldown_bars": 0,
            "tp_buckets": TP_BUCKETS.tolist(),
            "sl_buckets": SL_BUCKETS.tolist(),
            "margin_buckets": MARGIN_BUCKETS.tolist(),
            "leverage_buckets": LEVERAGE_BUCKETS.tolist(),
            "max_hold_buckets": MAX_HOLD_BUCKETS.tolist(),
        },
        "input_contract": {"base_feature_count": len(base_cols), "position_feature_count": len(threehead.POS_COLS), "total_features": len(base_cols) + len(threehead.POS_COLS)},
        "forbidden_feature_policy": {"deny_prefixes": omega.DENY_PREFIXES, "deny_tokens": omega.DENY_TOKENS},
        "risk_label_diag": risk_diag,
        "summaries": summaries,
        "results": reports,
        "ranking_by_validation_pnl": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "bundle": str(out_dir / "regime3_7head_tabm_bundle.pt"),
            "ranking": str(out_dir / "ranking.csv"),
            "report": str(out_dir / "report.json"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "ranking": rows[:8]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
