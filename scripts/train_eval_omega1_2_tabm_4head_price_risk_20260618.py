#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
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

import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402
import train_eval_omega1_2_asymmetric_direction_cleanup_20260618 as fast_frames  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega1_2_true_4head_price_exit_notional_bucket_conservative_tabm_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

RISK_COLS = ["tp_price_move", "sl_price_move"]
RISK_BOUNDS = {
    "tp_price_move": (0.008, 0.050),
    "sl_price_move": (0.006, 0.035),
}
NOTIONAL_BUCKETS = np.asarray([0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=np.float32)

CURRENT_PARENT_VAL = {
    "pnl": 100.54272942091158,
    "mdd": -10.677652697162888,
    "trades": 33,
    "wr": 0.6363636363636364,
    "long_entries": 3,
    "short_entries": 30,
    "exit_reasons": {"take_profit": 21, "stop_loss": 12},
}
CURRENT_PARENT_OOS = {
    "pnl": 72.76004148106665,
    "mdd": -8.108170708968387,
    "trades": 18,
    "wr": 0.7222222222222222,
    "long_entries": 2,
    "short_entries": 16,
    "exit_reasons": {"take_profit": 13, "stop_loss": 5},
}


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
    risk_loss_weight: float = 0.65


CFG = FourHeadConfig()


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


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
        self.risk_head = nn.Linear(int(cfg.hidden), len(RISK_COLS))
        self.notional_head = nn.Linear(int(cfg.hidden), len(NOTIONAL_BUCKETS))

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
            "risk": self.risk_head(h),
            "notional": self.notional_head(h),
        }


def _risk_to_unit(risk: np.ndarray) -> np.ndarray:
    out = np.empty_like(risk, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = RISK_BOUNDS[col]
        out[:, j] = (np.clip(risk[:, j], lo, hi) - lo) / (hi - lo)
    return out


def _unit_to_risk(unit: np.ndarray) -> np.ndarray:
    u = np.clip(unit, 0.0, 1.0)
    out = np.empty_like(u, dtype=np.float32)
    for j, col in enumerate(RISK_COLS):
        lo, hi = RISK_BOUNDS[col]
        out[:, j] = lo + u[:, j] * (hi - lo)
    return out


def _build_price_risk_labels(frame: pd.DataFrame, y: np.ndarray, *, horizon: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    y = np.asarray(y, dtype=np.int64)
    risk = np.zeros((len(frame), len(RISK_COLS)), dtype=np.float32)
    risk[:, 0] = float(omega.BASE_TEMPLATE["take_profit"]) / max(float(omega.BASE_TEMPLATE["notional"]), 1.0e-12)
    risk[:, 1] = float(omega.BASE_TEMPLATE["stop_loss"]) / max(float(omega.BASE_TEMPLATE["notional"]), 1.0e-12)
    notional_bucket = np.full(len(frame), int(np.argmin(np.abs(NOTIONAL_BUCKETS - float(omega.BASE_TEMPLATE["notional"])))), dtype=np.int64)
    active = (y == omega.ACTION_LONG) | (y == omega.ACTION_SHORT)
    usable = np.zeros(len(frame), dtype=bool)
    favs: list[float] = []
    advs: list[float] = []
    notionals: list[float] = []
    for i in np.flatnonzero(active):
        entry_i = int(i) + 1
        end_i = min(len(frame) - 2, entry_i + int(horizon))
        if entry_i >= end_i:
            continue
        entry = float(arrays["open"][entry_i])
        if not np.isfinite(entry) or entry <= 0.0:
            continue
        high = arrays["high"][entry_i : end_i + 1]
        low = arrays["low"][entry_i : end_i + 1]
        if int(y[i]) == omega.ACTION_LONG:
            fav = float(np.nanmax(high) / entry - 1.0)
            adv = float(entry / max(np.nanmin(low), 1.0e-12) - 1.0)
        else:
            fav = float(entry / max(np.nanmin(low), 1.0e-12) - 1.0)
            adv = float(np.nanmax(high) / entry - 1.0)
        fav = max(fav, 0.0)
        adv = max(adv, 0.0)
        edge = fav - 1.25 * adv
        notional_raw = 0.5 + 2.5 / (1.0 + np.exp(-edge / 0.018))
        if adv >= 0.030:
            notional_raw = min(notional_raw, 1.5)
        elif adv >= 0.020:
            notional_raw = min(notional_raw, 2.0)
        elif adv >= fav:
            notional_raw = min(notional_raw, 2.0)
        bucket = int(np.argmin(np.abs(NOTIONAL_BUCKETS - float(notional_raw))))
        tp = np.clip(max(0.008, 0.70 * fav), RISK_BOUNDS["tp_price_move"][0], RISK_BOUNDS["tp_price_move"][1])
        sl = np.clip(max(0.006, 0.85 * adv), RISK_BOUNDS["sl_price_move"][0], RISK_BOUNDS["sl_price_move"][1])
        risk[int(i)] = np.asarray([tp, sl], dtype=np.float32)
        notional_bucket[int(i)] = bucket
        usable[int(i)] = True
        favs.append(fav)
        advs.append(adv)
        notionals.append(float(NOTIONAL_BUCKETS[bucket]))
    unit = _risk_to_unit(risk)
    diag = {
        "horizon": int(horizon),
        "active_rows": int(active.sum()),
        "usable_rows": int(usable.sum()),
        "tp_price_move_mean": float(np.mean(risk[usable, 0])) if bool(usable.any()) else 0.0,
        "sl_price_move_mean": float(np.mean(risk[usable, 1])) if bool(usable.any()) else 0.0,
        "notional_bucket_values": NOTIONAL_BUCKETS.tolist(),
        "notional_mean": float(np.mean([NOTIONAL_BUCKETS[int(x)] for x in notional_bucket[usable]])) if bool(usable.any()) else 0.0,
        "notional_bucket_counts": {str(int(k)): int(v) for k, v in pd.Series(notional_bucket[usable]).value_counts().sort_index().to_dict().items()} if bool(usable.any()) else {},
        "favorable_move_mean": float(np.mean(favs)) if favs else 0.0,
        "adverse_move_mean": float(np.mean(advs)) if advs else 0.0,
        "notional_q": np.quantile(np.asarray(notionals, dtype=np.float64), [0.1, 0.5, 0.9]).tolist() if notionals else [],
    }
    return unit.astype(np.float32), notional_bucket.astype(np.int64), usable.astype(np.float32), diag


def _fit_expert_4head(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_risk_unit: np.ndarray,
    y_notional_bucket: np.ndarray,
    risk_mask: np.ndarray,
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
    _x_np, scaler = threehead._standardize_fit(x_all)
    x_dir_np = threehead._standardize_apply(x_dir, scaler)
    x_exit_np = threehead._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    y_risk_np = np.asarray(y_risk_unit, dtype=np.float32)
    y_notional_np = np.asarray(y_notional_bucket, dtype=np.int64)
    route_w = threehead._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = threehead._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    risk_w = np.asarray(risk_mask, dtype=np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0 or float(risk_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 4-head sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = FourHeadTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_risk_np[train_idx]),
        torch.from_numpy(y_notional_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(risk_w[train_idx]),
    )
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
        for xb, yb, rb, nb, wb, rwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            rb = rb.to(device, non_blocking=True)
            nb = nb.to(device, non_blocking=True)
            wb = wb.to(device, non_blocking=True)
            rwb = rwb.to(device, non_blocking=True)
            xe = xe.to(device, non_blocking=True)
            ye = ye.to(device, non_blocking=True)
            we = we.to(device, non_blocking=True)
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
            risk_pred = torch.sigmoid(out["risk"]).mean(dim=1)
            loss_risk = (((risk_pred - rb) ** 2).mean(dim=1) * rwb).sum() / torch.clamp(rwb.sum(), min=1.0)
            loss_notional_k = torch.nn.functional.cross_entropy(
                out["notional"].reshape(-1, len(NOTIONAL_BUCKETS)),
                nb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_notional = (loss_notional_k.mean(dim=1) * rwb).sum() / torch.clamp(rwb.sum(), min=1.0)
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit + float(CFG.risk_loss_weight) * (loss_risk + loss_notional)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vr = torch.from_numpy(y_risk_np[val_idx]).to(device)
            vn = torch.from_numpy(y_notional_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vrw = torch.from_numpy(risk_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vrisk = (((torch.sigmoid(vo["risk"]).mean(dim=1) - vr) ** 2).mean(dim=1) * vrw).sum() / torch.clamp(vrw.sum(), min=1.0)
            vnotional_k = torch.nn.functional.cross_entropy(
                vo["notional"].reshape(-1, len(NOTIONAL_BUCKETS)),
                vn[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            vnotional = (vnotional_k.mean(dim=1) * vrw).sum() / torch.clamp(vrw.sum(), min=1.0)
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
                    + float(CFG.risk_loss_weight) * (vrisk + vnotional)
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
        "risk_cols": RISK_COLS,
        "risk_bounds": RISK_BOUNDS,
        "notional_buckets": NOTIONAL_BUCKETS.tolist(),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = FourHeadTabM(int(payload["n_features"]), cfg=CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = threehead._standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": [], "exit": [], "risk_unit": [], "notional": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["risk_unit"].append(torch.sigmoid(out["risk"]).mean(dim=1).detach().cpu().numpy())
        chunks["notional"].append(torch.softmax(out["notional"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _apply_price_risk(dec: pd.DataFrame, risk: np.ndarray, notional_bucket: np.ndarray) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    r = np.asarray(risk, dtype=np.float64)
    bucket_idx = np.clip(np.asarray(notional_bucket, dtype=np.int64), 0, len(NOTIONAL_BUCKETS) - 1)
    notional = NOTIONAL_BUCKETS[bucket_idx].astype(np.float64)
    tp_price = np.clip(r[:, 0], RISK_BOUNDS["tp_price_move"][0], RISK_BOUNDS["tp_price_move"][1])
    sl_price = np.clip(r[:, 1], RISK_BOUNDS["sl_price_move"][0], RISK_BOUNDS["sl_price_move"][1])
    out.loc[active, "notional_exposure"] = notional[active]
    out.loc[active, "position_fraction"] = notional[active]
    out.loc[active, "leverage"] = 1.0
    out.loc[active, "take_profit"] = 0.0
    out.loc[active, "stop_loss"] = 0.0
    out.loc[active, "tp_price_move"] = tp_price[active]
    out.loc[active, "sl_price_move"] = sl_price[active]
    out.loc[active, "notional_bucket"] = bucket_idx[active]
    out.loc[~active, "notional_exposure"] = 0.0
    out.loc[~active, "position_fraction"] = 0.0
    out.loc[~active, "leverage"] = 1.0
    out.loc[~active, "take_profit"] = 0.0
    out.loc[~active, "stop_loss"] = 0.0
    out.loc[~active, "tp_price_move"] = 0.0
    out.loc[~active, "sl_price_move"] = 0.0
    out.loc[~active, "notional_bucket"] = -1
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _metrics_price_move_exit(frame: pd.DataFrame, dec: pd.DataFrame, *, fee: float, slip: float, cost_mult: float) -> dict[str, Any]:
    required = {"tp_price_move", "sl_price_move", "notional_exposure", "side", "action"}
    missing = sorted(required - set(dec.columns))
    if missing:
        raise RuntimeError(f"price-move exit decision missing columns: {missing}")
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
    notional = 0.0
    tp_price_move = 0.0
    sl_price_move = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    notional_sum = 0.0
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            price_move = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            eq = cash * (1.0 + price_move * notional)
        else:
            price_move = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            reason = ""
            if price_move >= tp_price_move:
                reason = "take_profit"
            elif price_move <= -abs(sl_price_move):
                reason = "stop_loss"
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
        notional = float(row.get("notional_exposure", 0.0) or 0.0)
        tp_price_move = float(row.get("tp_price_move", 0.0) or 0.0)
        sl_price_move = float(row.get("sl_price_move", 0.0) or 0.0)
        if notional <= 0.0 or tp_price_move <= 0.0 or sl_price_move <= 0.0:
            raise RuntimeError("active price-move exit row has non-positive notional/tp/sl")
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    n_entries = long_entries + short_entries
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / trades) if trades else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "avg_notional": float(notional_sum / n_entries) if n_entries else 0.0,
        "avg_leverage": 1.0,
    }


def _metric_row(candidate: str, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {"candidate": candidate}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_vs_current"] = float(row["val_pnl"] - float(base_val["pnl"]))
    row["oos_delta_vs_current"] = float(row["oos_pnl"] - float(base_oos["pnl"]))
    val_reasons = row["val_reasons"] if isinstance(row["val_reasons"], dict) else {}
    row["val_stop_loss"] = int(val_reasons.get("stop_loss", 0))
    row["selection_score_val_only"] = (
        row["val_delta_vs_current"]
        + 10.0 * float(row["val_wr"])
        + 0.25 * float(row["val_mdd"])
        - 0.75 * float(row["val_stop_loss"])
        - 0.05 * max(0.0, float(row["val_trades"]) - 80.0)
    )
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--quality-thresholds", default="0.45,0.55,0.65,0.75,0.80,0.85,0.90")
    ap.add_argument("--risk-label-horizon", type=int, default=192)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192")
    ap.add_argument("--max-exit-samples", type=int, default=30000)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260618)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    threehead._seed_everything(int(args.seed))
    device = threehead._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = fast_frames._prepare_frames_fast(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = fast_frames._filter_to_parent_prediction_span(frames["val_raw"], "validation")
    oos_raw = fast_frames._filter_to_parent_prediction_span(frames["oos_raw"], "oos")
    x_train = threehead._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_risk, y_notional_bucket, risk_mask, risk_diag = _build_price_risk_labels(train_raw, y_train, horizon=int(args.risk_label_horizon))
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_risk = y_risk[:limit]
        y_notional_bucket = y_notional_bucket[:limit]
        risk_mask = risk_mask[:limit]
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
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
    x_exit = threehead._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(json.dumps({"stage": "fit_4head_expert", "expert": expert}, ensure_ascii=True), flush=True)
        payload = _fit_expert_4head(
            x_train,
            y_train,
            y_risk,
            y_notional_bucket,
            risk_mask,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=OUT_DIR / "models" / f"{expert}_4head_price_risk_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(OUT_DIR / "models" / f"{expert}_4head_price_risk_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_frame(frame: pd.DataFrame, *, threshold: float, oof: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        x = threehead._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = threehead._routed(preds, route, "direction", 3)
        quality = threehead._routed(preds, route, "quality", 3)
        risk_unit = threehead._routed(preds, route, "risk_unit", len(RISK_COLS))
        notional_prob = threehead._routed(preds, route, "notional", len(NOTIONAL_BUCKETS))
        notional_bucket = np.argmax(notional_prob, axis=1).astype(np.int64)
        risk = _unit_to_risk(risk_unit.astype(np.float32)).astype(np.float64)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = threehead._prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)
        dec = omega._to_fixed_decisions(src, oof=oof)
        return _apply_price_risk(dec, risk, notional_bucket), risk, notional_bucket

    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    for q in thresholds:
        val_dec, val_risk, val_notional_bucket = predict_frame(val_raw, threshold=q, oof=True)
        oos_dec, oos_risk, oos_notional_bucket = predict_frame(oos_raw, threshold=q, oof=False)
        val_m = _metrics_price_move_exit(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_m = _metrics_price_move_exit(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        name = f"4head_price_risk_q{q:.2f}".replace(".", "p")
        reports[name] = {
            "validation": val_m,
            "oos": oos_m,
            "validation_risk_distribution": {
                "tp_price_move_mean": float(np.mean(val_risk[:, 0])),
                "sl_price_move_mean": float(np.mean(val_risk[:, 1])),
                "notional_mean": float(np.mean(NOTIONAL_BUCKETS[val_notional_bucket])),
                "notional_bucket_counts": {str(int(k)): int(v) for k, v in pd.Series(val_notional_bucket).value_counts().sort_index().to_dict().items()},
            },
            "oos_risk_distribution": {
                "tp_price_move_mean": float(np.mean(oos_risk[:, 0])),
                "sl_price_move_mean": float(np.mean(oos_risk[:, 1])),
                "notional_mean": float(np.mean(NOTIONAL_BUCKETS[oos_notional_bucket])),
                "notional_bucket_counts": {str(int(k)): int(v) for k, v in pd.Series(oos_notional_bucket).value_counts().sort_index().to_dict().items()},
            },
        }
        rows.append(_metric_row(name, val_m, oos_m, CURRENT_PARENT_VAL, CURRENT_PARENT_OOS))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_delta_vs_current", "val_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_vs_current"], ascending=False).iloc[0].to_dict()
    torch.save(
        {
            "models": models,
            "base_cols": base_cols,
            "pos_cols": threehead.POS_COLS,
            "config": CFG.__dict__,
            "risk_cols": RISK_COLS,
            "risk_bounds": RISK_BOUNDS,
            "notional_buckets": NOTIONAL_BUCKETS.tolist(),
        },
        OUT_DIR / "true_4head_price_risk_tabm_bundle.pt",
    )
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_4head_price_risk_eval",
        "design": "Parent TabM with shared encoder and four heads: direction, quality, exit, risk. Risk branch outputs tp_price_move and sl_price_move as regressions plus a notional bucket classifier. Leverage and margin are removed from the learned risk contract. Backtest exits use the same price-move semantics as the model output.",
        "risk_contract": {
            "outputs": RISK_COLS + ["notional_bucket"],
            "bounds": RISK_BOUNDS,
            "notional_buckets": NOTIONAL_BUCKETS.tolist(),
            "take_profit": "exit when realized price_move >= tp_price_move",
            "stop_loss": "exit when realized price_move <= -sl_price_move",
            "notional": "selected bucket value replacing margin and leverage",
            "pnl": "account PnL = realized price_move * notional",
            "leverage": "fixed bookkeeping value 1.0; not learned; not multiplied into TP/SL",
        },
        "current_quality_gate_parent": {"validation": CURRENT_PARENT_VAL, "oos": CURRENT_PARENT_OOS},
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "results": reports,
        "top30": ranking.head(30).to_dict(orient="records"),
        "risk_label_diag": risk_diag,
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "hold_offsets": hold_offsets, "diag": exit_diag},
        "summaries": summaries,
        "artifacts": {"out_dir": str(OUT_DIR), "ranking": str(OUT_DIR / "ranking.csv"), "report": str(OUT_DIR / "report.json")},
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "best_oos": best_oos, "risk_label_diag": risk_diag}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
