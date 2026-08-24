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


MODEL_ID = "omega4_direction_conditioned_quality_20260621"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _log(message: str) -> None:
    print(f"[{MODEL_ID}] {message}", flush=True)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class DirectionConditionedQualityTabM(nn.Module):
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
        self.direction_head = nn.Linear(int(cfg.hidden), 3)
        self.side_quality_head = nn.Linear(int(cfg.hidden), 2)
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
            "side_quality": self.side_quality_head(h),
            "exit": self.exit_head(h),
        }


def _side_quality_targets(frame: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    target = np.zeros((len(frame), 2), dtype=np.float32)
    filled = {1: 0, -1: 0}
    positive = {1: 0, -1: 0}
    nets: dict[int, list[float]] = {1: [], -1: []}
    for i in range(0, len(frame) - 2):
        for side, col in ((1, 0), (-1, 1)):
            ok, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
            if not ok:
                nets[side].append(-1.0)
                continue
            filled[side] += 1
            entry_i = min(int(i) + 1, len(frame) - 1)
            cash_after_entry_fee = 1.0 - 1.0 * float(entry_fee) * notional
            net, _final_i, _reason = exit_head._continue_to_barrier_net(
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
            nets[side].append(float(net))
            if float(net) > 0.0:
                target[i, col] = 1.0
                positive[side] += 1
    diag: dict[str, Any] = {"mode": "direction_conditioned_binary_side_quality"}
    for side, name in ((1, "long"), (-1, "short")):
        arr = np.asarray(nets[side], dtype=np.float64) if nets[side] else np.asarray([0.0], dtype=np.float64)
        diag[name] = {
            "filled_entries": int(filled[side]),
            "positive_rows": int(positive[side]),
            "positive_rate_all": float(target[:, 0 if side == 1 else 1].mean()),
            "net_mean": float(arr.mean()),
            "net_p10": float(np.quantile(arr, 0.10)),
            "net_p50": float(np.quantile(arr, 0.50)),
            "net_p90": float(np.quantile(arr, 0.90)),
        }
    return target, diag


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_side_quality: np.ndarray,
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
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_sq_np = np.asarray(y_side_quality, dtype=np.float32)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    long_w = compute_sample_weight(class_weight="balanced", y=y_sq_np[:, 0].astype(np.int64)).astype(np.float32)
    short_w = compute_sample_weight(class_weight="balanced", y=y_sq_np[:, 1].astype(np.int64)).astype(np.float32)
    qual_w = np.stack([long_w, short_w], axis=1).astype(np.float32) * route_w[:, None]
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = DirectionConditionedQualityTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_sq_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yq, wb, qwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            yq = yq.to(device)
            wb = wb.to(device)
            qwb = qwb.to(device)
            xe = xe.to(device)
            ye = ye.to(device)
            we = we.to(device)
            out = model(xb)
            out_exit = model(xe)
            loss_dir_k = torch.nn.functional.cross_entropy(out["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            loss_qual_k = torch.nn.functional.binary_cross_entropy_with_logits(out["side_quality"], yq[:, None, :].expand(-1, int(parent.CFG.k), -1), reduction="none")
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = ((loss_qual_k.mean(dim=1) * qwb).sum(dim=1)).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(parent.CFG.quality_loss_weight) * loss_qual + float(parent.CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vq = torch.from_numpy(y_sq_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vqual = torch.nn.functional.binary_cross_entropy_with_logits(vo["side_quality"], vq[:, None, :].expand(-1, int(parent.CFG.k), -1), reduction="none")
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float((((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)) + float(parent.CFG.quality_loss_weight) * (((vqual.mean(dim=1) * vqw).sum(dim=1)).sum() / torch.clamp(vqw.sum(), min=1.0)) + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))).detach().cpu())
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
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = DirectionConditionedQualityTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    outs: dict[str, list[np.ndarray]] = {"direction": [], "side_quality": [], "exit": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        outs["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        outs["side_quality"].append(torch.sigmoid(out["side_quality"]).mean(dim=1).detach().cpu().numpy())
        outs["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in outs.items()}


def _routed(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray, head: str, n_classes: int) -> np.ndarray:
    out = np.zeros((len(route), n_classes), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][head][mask]
    return out


def _score_calibration(direction: np.ndarray, side_quality: np.ndarray, action: np.ndarray) -> dict[str, float]:
    mask = np.asarray(action, dtype=np.int64) != 0
    if not bool(mask.any()):
        mask = np.ones(len(direction), dtype=bool)
    values = {
        "long_dir": direction[mask, 1],
        "short_dir": direction[mask, 2],
        "long_quality": side_quality[mask, 0],
        "short_quality": side_quality[mask, 1],
    }
    out: dict[str, float] = {}
    for name, arr in values.items():
        arr = np.asarray(arr, dtype=np.float64)
        out[f"{name}_mean"] = float(np.mean(arr))
        std = float(np.std(arr))
        out[f"{name}_std"] = std if std >= 1.0e-6 else 1.0
    return out


def _side_scores(direction: np.ndarray, side_quality: np.ndarray, *, score_mode: str, calibration: dict[str, float] | None) -> tuple[np.ndarray, np.ndarray]:
    if str(score_mode) == "product":
        return direction[:, 1] * side_quality[:, 0], direction[:, 2] * side_quality[:, 1]
    if str(score_mode) != "zsum":
        raise RuntimeError(f"unknown direction-conditioned quality score mode: {score_mode}")
    if calibration is None:
        raise RuntimeError("zsum score mode requires calibration")
    long_score = (
        (direction[:, 1] - float(calibration["long_dir_mean"])) / float(calibration["long_dir_std"])
        + (side_quality[:, 0] - float(calibration["long_quality_mean"])) / float(calibration["long_quality_std"])
    )
    short_score = (
        (direction[:, 2] - float(calibration["short_dir_mean"])) / float(calibration["short_dir_std"])
        + (side_quality[:, 1] - float(calibration["short_quality_mean"])) / float(calibration["short_quality_std"])
    )
    return long_score, short_score


def _prediction_output(
    frame: pd.DataFrame,
    direction: np.ndarray,
    side_quality: np.ndarray,
    *,
    threshold: float,
    prefix: str,
    score_mode: str,
    calibration: dict[str, float] | None,
) -> pd.DataFrame:
    long_score, short_score = _side_scores(direction, side_quality, score_mode=score_mode, calibration=calibration)
    choose_long = long_score >= short_score
    action_score = np.maximum(long_score, short_score)
    final_action = np.where(action_score >= float(threshold), np.where(choose_long, 1, 2), 0).astype(np.int64)
    dir_action = np.argmax(direction, axis=1).astype(np.int64)
    route = hard._route_id(frame)
    route_probs = frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64)
    sorted_route = np.sort(route_probs, axis=1)
    out = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy()})
    out[f"{prefix}_router_expert"] = np.asarray([hard.EXPERT_NAMES[int(i)].replace("chop_expert", "chop") for i in route], dtype=object)
    out[f"{prefix}_router_confidence"] = sorted_route[:, -1]
    out[f"{prefix}_router_margin"] = sorted_route[:, -1] - sorted_route[:, -2]
    out[f"{prefix}_dir_p_cash"] = direction[:, 0]
    out[f"{prefix}_dir_p_long"] = direction[:, 1]
    out[f"{prefix}_dir_p_short"] = direction[:, 2]
    out[f"{prefix}_dir_confidence"] = np.max(direction, axis=1)
    out[f"{prefix}_dir_side_edge"] = direction[:, 1] - direction[:, 2]
    out[f"{prefix}_dir_trade_prob"] = 1.0 - direction[:, 0]
    out[f"{prefix}_dir_action"] = dir_action
    out[f"{prefix}_quality_p_cash"] = 1.0 - action_score
    out[f"{prefix}_quality_p_long"] = long_score
    out[f"{prefix}_quality_p_short"] = short_score
    out[f"{prefix}_quality_for_action"] = action_score
    out[f"{prefix}_quality_threshold"] = float(threshold)
    out[f"{prefix}_final_action"] = final_action
    return out


def _metrics_with_exit(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
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
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
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
        if pos != 0:
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
                prob = float(_predict_payload(models[expert], xrow, device=device)["exit"][0, 1])
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
                continue
        if pos != 0 or not bool(active[i]):
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
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--max-exit-samples", type=int, default=15000)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--score-mode", choices=["zsum", "product"], default="zsum")
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260621)
    ap.add_argument("--out-suffix", default="smoke_e2_train15k_exit15k_q70_exit70")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    _log("loading frames")
    frames = omega4._prepare_frames(disable_tp_sl=False, direction_label_dir=Path(args.direction_label_dir), quality_mode="same_as_direction", quality_label_dir=None)
    _log("frames loaded")
    fee, slip = omega._load_fee_slip()
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    base_cols = list(frames["feature_cols"])
    if int(args.max_train_rows) > 0:
        train_fit_frame = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True)
    else:
        train_fit_frame = train_raw
    _log(f"building side-quality labels rows={len(train_fit_frame)}")
    y_side_quality, side_quality_diag = _side_quality_targets(train_fit_frame, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    _log("building model inputs")
    x_train = parent._base_input(train_fit_frame, base_cols)
    y_train = train_fit_frame["zigzag_action"].to_numpy(dtype=np.int64)
    _log(f"building exit labels max_samples={int(args.max_exit_samples)}")
    x_exit_raw, y_exit, frame_exit, exit_diag = omega4._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        risk_margin=None,
        risk_leverage=None,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_samples=int(args.max_exit_samples),
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        _log(f"training expert={expert}")
        payload = _fit_expert(
            x_train,
            y_train,
            y_side_quality,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_dcq_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {"model": str(out_dir / "models" / f"{expert}_dcq_tabm.pt"), "epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}
        _log(f"trained expert={expert}")

    def predict_components(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = _routed(preds, route, "direction", 3)
        side_quality = _routed(preds, route, "side_quality", 2)
        return x, direction, side_quality

    train_x, train_direction, train_side_quality = predict_components(train_fit_frame)
    del train_x
    score_calibration = _score_calibration(train_direction, train_side_quality, y_train)
    train_long_score, train_short_score = _side_scores(train_direction, train_side_quality, score_mode=str(args.score_mode), calibration=score_calibration)
    train_action_score = np.maximum(train_long_score, train_short_score)
    train_active = y_train != 0
    if str(args.score_mode) == "zsum":
        quality_cutoff = float(np.quantile(train_action_score[train_active] if bool(train_active.any()) else train_action_score, float(args.quality_threshold)))
    else:
        quality_cutoff = float(args.quality_threshold)

    def predict(frame: pd.DataFrame, *, oof: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        x, direction, side_quality = predict_components(frame)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        return _prediction_output(
            frame,
            direction,
            side_quality,
            threshold=float(quality_cutoff),
            prefix=prefix,
            score_mode=str(args.score_mode),
            calibration=score_calibration,
        ), x

    val_src, x_val = predict(val_raw, oof=True)
    oos_src, x_oos = predict(oos_raw, oof=False)
    _log("computing no-exit metrics")
    val_dec = parent._to_decisions(val_src, oof=True)
    oos_dec = parent._to_decisions(oos_src, oof=False)
    val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    _log("computing exit-head metrics")
    val_exit = _metrics_with_exit(val_raw, x_val, val_dec, models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_exit = _metrics_with_exit(oos_raw, x_oos, oos_dec, models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_src.to_csv(out_dir / "validation_predictions_2025_dcq_q70.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_dcq_q70.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega4_1_exit_thr_0p70",
        "design": "Direction-conditioned quality: long_quality and short_quality are trained as side-specific binary barrier payoff labels and combined with direction probabilities.",
        "thresholds": {
            "quality_threshold": float(args.quality_threshold),
            "quality_cutoff": float(quality_cutoff),
            "score_mode": str(args.score_mode),
            "exit_threshold": float(args.exit_threshold),
        },
        "score_calibration": score_calibration,
        "label_contract": {
            "direction_label_dir": str(args.direction_label_dir),
            "direction_target": "zigzag_action",
            "long_quality_target": "barrier_replay_net_return(long) > 0",
            "short_quality_target": "barrier_replay_net_return(short) > 0",
            "exit_target": "entry_label_terminal_giveback",
        },
        "side_quality_diag": side_quality_diag,
        "exit_label": exit_diag,
        "results": {"validation_no_exit": val_m, "oos_no_exit": oos_m, "validation_exit_thr": val_exit, "oos_exit_thr": oos_exit},
        "summaries": summaries,
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "validation_predictions": str(out_dir / "validation_predictions_2025_dcq_q70.csv"), "oos_predictions": str(out_dir / "oos_predictions_2026_dcq_q70.csv")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_class": "DirectionConditionedQualityTabM"}, out_dir / "dcq_3head_tabm_bundle.pt")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
