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


MODEL_ID = "omega4_5head_margin_leverage_20260622"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MARGIN_BUCKETS = np.asarray([0.15, 0.225, 0.30, 0.45, 0.60], dtype=np.float64)
LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float64)
RISK_LOSS_WEIGHT = 0.20


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _nearest_idx(values: np.ndarray, value: float) -> int:
    return int(np.argmin(np.abs(np.asarray(values, dtype=np.float64) - float(value))))


class FiveHeadRiskTabM(nn.Module):
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
        self.quality_head = nn.Linear(int(cfg.hidden), 3)
        self.exit_head = nn.Linear(int(cfg.hidden), 2)
        self.margin_fraction_head = nn.Linear(int(cfg.hidden), len(MARGIN_BUCKETS))
        self.leverage_head = nn.Linear(int(cfg.hidden), len(LEVERAGE_BUCKETS))

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
            "margin_fraction": self.margin_fraction_head(h),
            "leverage": self.leverage_head(h),
        }


def _path_mae_mfe(
    arrays: dict[str, np.ndarray],
    *,
    entry_i: int,
    exit_i: int,
    side: int,
    entry_price: float,
    notional: float,
    slip_eff: float,
) -> tuple[float, float]:
    mae = 0.0
    mfe = 0.0
    for row_i in range(int(entry_i), max(int(entry_i), int(exit_i)) + 1):
        px = float(arrays["close"][int(row_i)])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1.0e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1.0e-12)
        unreal = float(raw) * float(notional)
        mae = min(mae, unreal)
        mfe = max(mfe, unreal)
    return float(mae), float(mfe)


def _build_margin_leverage_labels(
    frame: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    required = {"zigzag_action", "open", "high", "low", "close"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"margin/leverage labels missing columns: {missing}")
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_notional = float(omega.BASE_TEMPLATE["notional"])
    base_leverage = float(omega.BASE_TEMPLATE["leverage"])
    base_margin = base_notional / max(base_leverage, 1.0e-12)
    tp_price_move = float(omega.BASE_TEMPLATE["take_profit"])
    sl_price_move = float(omega.BASE_TEMPLATE["stop_loss"])
    default_margin_idx = _nearest_idx(MARGIN_BUCKETS, base_margin)
    default_leverage_idx = _nearest_idx(LEVERAGE_BUCKETS, base_leverage)
    margin_y = np.full(len(frame), default_margin_idx, dtype=np.int64)
    leverage_y = np.full(len(frame), default_leverage_idx, dtype=np.int64)
    risk_active = np.zeros(len(frame), dtype=np.float32)
    chosen_utility: list[float] = []
    chosen_net: list[float] = []
    chosen_mae: list[float] = []
    reason_counts: dict[str, int] = {}
    active_rows = 0
    filled_entries = 0
    for i in range(0, len(frame) - 2):
        a = int(action[i])
        if a not in (1, 2):
            continue
        active_rows += 1
        side = 1 if a == 1 else -1
        ok, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not ok:
            reason_counts["entry_not_filled"] = reason_counts.get("entry_not_filled", 0) + 1
            continue
        filled_entries += 1
        entry_i = min(int(i) + 1, len(frame) - 1)
        best: tuple[float, int, int, float, float, str] | None = None
        for mi, margin in enumerate(MARGIN_BUCKETS):
            for li, leverage in enumerate(LEVERAGE_BUCKETS):
                notional = float(margin) * float(leverage)
                take_profit = float(tp_price_move) * notional
                stop_loss = float(sl_price_move) * notional
                cash_after_entry_fee = 1.0 - float(entry_fee) * notional
                net, final_i, reason = exit_head._continue_to_barrier_net(
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
                mae, _mfe = _path_mae_mfe(arrays, entry_i=entry_i, exit_i=final_i, side=side, entry_price=float(entry_price), notional=notional, slip_eff=slip_eff)
                utility = (
                    float(net)
                    - 0.40 * abs(float(mae))
                    - 0.015 * float(margin)
                    - 0.0025 * max(float(leverage) - 2.0, 0.0)
                    - (0.015 if reason == "stop_loss" else 0.0)
                )
                if best is None or utility > best[0]:
                    best = (float(utility), int(mi), int(li), float(net), float(mae), str(reason))
        if best is None:
            reason_counts["no_candidate"] = reason_counts.get("no_candidate", 0) + 1
            continue
        utility, mi, li, net, mae, reason = best
        margin_y[i] = int(mi)
        leverage_y[i] = int(li)
        risk_active[i] = 1.0
        chosen_utility.append(float(utility))
        chosen_net.append(float(net))
        chosen_mae.append(float(mae))
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
    active_idx = risk_active > 0
    diag = {
        "rows": int(len(frame)),
        "active_rows": int(active_rows),
        "filled_entries": int(filled_entries),
        "risk_labeled_rows": int(active_idx.sum()),
        "margin_buckets": [float(x) for x in MARGIN_BUCKETS.tolist()],
        "leverage_buckets": [float(x) for x in LEVERAGE_BUCKETS.tolist()],
        "default_margin_fraction": float(MARGIN_BUCKETS[default_margin_idx]),
        "default_leverage": float(LEVERAGE_BUCKETS[default_leverage_idx]),
        "tp_price_move": float(tp_price_move),
        "sl_price_move": float(sl_price_move),
        "chosen_margin_counts": {str(float(MARGIN_BUCKETS[k])): int(v) for k, v in pd.Series(margin_y[active_idx]).value_counts().sort_index().items()},
        "chosen_leverage_counts": {str(float(LEVERAGE_BUCKETS[k])): int(v) for k, v in pd.Series(leverage_y[active_idx]).value_counts().sort_index().items()},
        "chosen_reason_counts": reason_counts,
        "utility_mean": float(np.mean(chosen_utility)) if chosen_utility else 0.0,
        "net_mean": float(np.mean(chosen_net)) if chosen_net else 0.0,
        "mae_p50": float(np.quantile(np.asarray(chosen_mae), 0.50)) if chosen_mae else 0.0,
    }
    return margin_y, leverage_y, risk_active, diag


def _risk_weights(y: np.ndarray, active: np.ndarray, route_w: np.ndarray) -> np.ndarray:
    out = np.zeros(len(y), dtype=np.float32)
    mask = np.asarray(active, dtype=np.float32) > 0.0
    if bool(mask.any()):
        cw = compute_sample_weight(class_weight="balanced", y=np.asarray(y, dtype=np.int64)[mask]).astype(np.float32)
        out[mask] = cw
    out *= np.asarray(route_w, dtype=np.float32)
    return out


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    y_margin: np.ndarray,
    y_leverage: np.ndarray,
    risk_active: np.ndarray,
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
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_margin_np = np.asarray(y_margin, dtype=np.int64)
    y_leverage_np = np.asarray(y_leverage, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    margin_w = _risk_weights(y_margin_np, risk_active, route_w)
    leverage_w = _risk_weights(y_leverage_np, risk_active, route_w)
    if min(float(dir_w.sum()), float(qual_w.sum()), float(ex_w.sum()), float(margin_w.sum()), float(leverage_w.sum())) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = FiveHeadRiskTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]),
        torch.from_numpy(y_margin_np[train_idx]),
        torch.from_numpy(y_leverage_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        torch.from_numpy(margin_w[train_idx]),
        torch.from_numpy(leverage_w[train_idx]),
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
        for xb, yd, yq, ym, yl, wd, wq, wm, wl in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yd, yq, ym, yl = xb.to(device), yd.to(device), yq.to(device), ym.to(device), yl.to(device)
            wd, wq, wm, wl = wd.to(device), wq.to(device), wm.to(device), wl.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            out = model(xb)
            out_exit = model(xe)

            def ce_head(logits: torch.Tensor, target: torch.Tensor, classes: int) -> torch.Tensor:
                return torch.nn.functional.cross_entropy(
                    logits.reshape(-1, classes),
                    target[:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(parent.CFG.k)).mean(dim=1)

            loss_dir = (ce_head(out["direction"], yd, 3) * wd).sum() / torch.clamp(wd.sum(), min=1.0)
            loss_qual = (ce_head(out["quality"], yq, 3) * wq).sum() / torch.clamp(wq.sum(), min=1.0)
            loss_margin = (ce_head(out["margin_fraction"], ym, len(MARGIN_BUCKETS)) * wm).sum() / torch.clamp(wm.sum(), min=1.0)
            loss_leverage = (ce_head(out["leverage"], yl, len(LEVERAGE_BUCKETS)) * wl).sum() / torch.clamp(wl.sum(), min=1.0)
            loss_exit = (ce_head(out_exit["exit"], ye, 2) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = (
                loss_dir
                + float(parent.CFG.quality_loss_weight) * loss_qual
                + float(parent.CFG.exit_loss_weight) * loss_exit
                + float(RISK_LOSS_WEIGHT) * loss_margin
                + float(RISK_LOSS_WEIGHT) * loss_leverage
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vq = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vm = torch.from_numpy(y_margin_np[val_idx]).to(device)
            vl = torch.from_numpy(y_leverage_np[val_idx]).to(device)
            vwd = torch.from_numpy(dir_w[val_idx]).to(device)
            vwq = torch.from_numpy(qual_w[val_idx]).to(device)
            vwm = torch.from_numpy(margin_w[val_idx]).to(device)
            vwl = torch.from_numpy(leverage_w[val_idx]).to(device)
            vye = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vwe = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vloss = (
                (ce_head(vo["direction"], vy, 3) * vwd).sum() / torch.clamp(vwd.sum(), min=1.0)
                + float(parent.CFG.quality_loss_weight) * (ce_head(vo["quality"], vq, 3) * vwq).sum() / torch.clamp(vwq.sum(), min=1.0)
                + float(parent.CFG.exit_loss_weight) * (ce_head(veo["exit"], vye, 2) * vwe).sum() / torch.clamp(vwe.sum(), min=1.0)
                + float(RISK_LOSS_WEIGHT) * (ce_head(vo["margin_fraction"], vm, len(MARGIN_BUCKETS)) * vwm).sum() / torch.clamp(vwm.sum(), min=1.0)
                + float(RISK_LOSS_WEIGHT) * (ce_head(vo["leverage"], vl, len(LEVERAGE_BUCKETS)) * vwl).sum() / torch.clamp(vwl.sum(), min=1.0)
            )
            val_loss = float(vloss.detach().cpu())
        if val_loss + 1.0e-6 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= int(parent.CFG.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
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
    model = FiveHeadRiskTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    out: dict[str, list[np.ndarray]] = {k: [] for k in ("direction", "quality", "exit", "margin_fraction", "leverage")}
    classes = {"direction": 3, "quality": 3, "exit": 2, "margin_fraction": len(MARGIN_BUCKETS), "leverage": len(LEVERAGE_BUCKETS)}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        pred = model(xb)
        for head, n_classes in classes.items():
            out[head].append(torch.softmax(pred[head], dim=-1).mean(dim=1).detach().cpu().numpy().reshape(-1, n_classes))
    return {head: np.concatenate(parts, axis=0).astype(np.float64) for head, parts in out.items()}


def _apply_learned_risk(dec: pd.DataFrame, margin: np.ndarray, leverage: np.ndarray) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    margin_idx = np.argmax(margin, axis=1)
    leverage_idx = np.argmax(leverage, axis=1)
    margin_value = MARGIN_BUCKETS[margin_idx].astype(np.float64)
    leverage_value = LEVERAGE_BUCKETS[leverage_idx].astype(np.float64)
    notional = margin_value * leverage_value
    tp_price_move = float(omega.BASE_TEMPLATE["take_profit"])
    sl_price_move = float(omega.BASE_TEMPLATE["stop_loss"])
    out["margin_fraction"] = 0.0
    out.loc[active, "margin_fraction"] = margin_value[active]
    out.loc[active, "leverage"] = leverage_value[active]
    out.loc[active, "notional_exposure"] = notional[active]
    out.loc[active, "position_fraction"] = margin_value[active]
    out.loc[active, "take_profit"] = tp_price_move * notional[active]
    out.loc[active, "stop_loss"] = sl_price_move * notional[active]
    out.loc[~active, ["margin_fraction", "notional_exposure", "position_fraction", "take_profit", "stop_loss"]] = 0.0
    out.loc[~active, "leverage"] = 1.0
    return out


def _risk_distribution(dec: pd.DataFrame) -> dict[str, Any]:
    active = omega._active(dec)
    if not bool(active.any()):
        return {"active_rows": 0}
    margin = pd.to_numeric(dec.loc[active, "margin_fraction"], errors="raise").to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(dec.loc[active, "leverage"], errors="raise").to_numpy(dtype=np.float64)
    notional = pd.to_numeric(dec.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    return {
        "active_rows": int(active.sum()),
        "margin_counts": {str(k): int(v) for k, v in pd.Series(margin).value_counts().sort_index().items()},
        "leverage_counts": {str(k): int(v) for k, v in pd.Series(leverage).value_counts().sort_index().items()},
        "avg_margin_fraction": float(np.mean(margin)),
        "avg_leverage": float(np.mean(leverage)),
        "avg_notional": float(np.mean(notional)),
        "max_notional": float(np.max(notional)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--max-exit-samples", type=int, default=15000)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260622)
    ap.add_argument("--out-suffix", default="e2_train15k_exit15k_q070")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=omega4.LABEL_DIR,
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
    if int(args.max_train_rows) > 0:
        train_fit_frame = train_raw.iloc[: int(args.max_train_rows)].reset_index(drop=True)
    else:
        train_fit_frame = train_raw.reset_index(drop=True)
    x_train = parent._base_input(train_fit_frame, base_cols)
    y_train = train_fit_frame["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_fit_frame["omega4_quality_action"].to_numpy(dtype=np.int64)
    print(f"stage=build_risk_labels rows={len(train_fit_frame)}", flush=True)
    y_margin, y_leverage, risk_active, risk_label_diag = _build_margin_leverage_labels(train_fit_frame, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    print(f"stage=build_exit_labels max_samples={int(args.max_exit_samples)}", flush=True)
    x_exit_raw, y_exit, frame_exit, exit_diag = omega4._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        risk_margin=None,
        risk_leverage=None,
        fee=fee,
        slip=slip,
        cost_mult=float(args.cost_mult),
        max_samples=int(args.max_exit_samples),
        terminal_window=3,
        adverse_unreal=-0.010,
        min_mfe_for_giveback=0.006,
        giveback_min=0.65,
    )
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        print(f"stage=train_expert expert={expert}", flush=True)
        payload = _fit_expert(
            x_train,
            y_train,
            y_quality,
            y_margin,
            y_leverage,
            risk_active,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_5head_margin_leverage_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {"best_validation_loss": payload["best_validation_loss"], "epochs_ran": payload["epochs_ran"]}

    def predict(frame: pd.DataFrame, *, oof: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        margin = parent._routed(preds, route, "margin_fraction", len(MARGIN_BUCKETS))
        leverage = parent._routed(preds, route, "leverage", len(LEVERAGE_BUCKETS))
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        fixed_dec = parent._to_decisions(src, oof=oof)
        learned_dec = _apply_learned_risk(fixed_dec, margin, leverage)
        return src, fixed_dec, learned_dec

    print("stage=predict_validation", flush=True)
    val_src, val_fixed_dec, val_learned_dec = predict(frames["val_raw"], oof=True)
    print("stage=predict_oos", flush=True)
    oos_src, oos_fixed_dec, oos_learned_dec = predict(frames["oos_raw"], oof=False)
    val_src.to_csv(out_dir / "validation_predictions_2025_q070.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_q070.csv", index=False)
    val_learned_dec.to_csv(out_dir / "validation_decisions_learned_margin_leverage_q070.csv", index=False)
    oos_learned_dec.to_csv(out_dir / "oos_decisions_learned_margin_leverage_q070.csv", index=False)
    fixed_results = {
        "validation": omega._metrics(frames["val_raw"], val_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        "oos": omega._metrics(frames["oos_raw"], oos_fixed_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    }
    learned_results = {
        "validation": omega._metrics(frames["val_raw"], val_learned_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
        "oos": omega._metrics(frames["oos_raw"], oos_learned_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult)),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Omega4 baseline TabM experts with added margin_fraction and leverage classification heads. Runtime is no-exit: direction+quality q0.70 owns entry; learned margin/leverage replace fixed notional/leverage; TP/SL preserve baseline price-move width and convert to account thresholds as price_move * notional.",
        "head_contract": ["direction", "quality", "exit", "margin_fraction", "leverage"],
        "risk_contract": {
            "margin_fraction_buckets": [float(x) for x in MARGIN_BUCKETS.tolist()],
            "leverage_buckets": [float(x) for x in LEVERAGE_BUCKETS.tolist()],
            "notional": "margin_fraction * leverage",
            "tp_sl": "BASE_TEMPLATE take_profit/stop_loss are price-move targets; account thresholds are price_move * notional",
        },
        "input_contract": {"base_feature_count": len(base_cols), "position_feature_count": len(parent.POS_COLS), "total_features": len(base_cols) + len(parent.POS_COLS)},
        "label_contract": frames["label_contract"],
        "risk_label_diag": risk_label_diag,
        "exit_label_diag": exit_diag,
        "summaries": summaries,
        "results": {"fixed_baseline_path_from_5head": fixed_results, "learned_margin_leverage_no_exit": learned_results},
        "risk_prediction_distribution": {"validation": _risk_distribution(val_learned_dec), "oos": _risk_distribution(oos_learned_dec)},
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "bundle": str(out_dir / "true_5head_margin_leverage_bundle.pt"),
            "validation_decisions": str(out_dir / "validation_decisions_learned_margin_leverage_q070.csv"),
            "oos_decisions": str(out_dir / "oos_decisions_learned_margin_leverage_q070.csv"),
        },
    }
    torch.save(
        {
            "models": models,
            "base_cols": base_cols,
            "pos_cols": parent.POS_COLS,
            "config": parent.CFG.__dict__,
            "model_class": "FiveHeadRiskTabM",
            "margin_fraction_buckets": MARGIN_BUCKETS,
            "leverage_buckets": LEVERAGE_BUCKETS,
        },
        out_dir / "true_5head_margin_leverage_bundle.pt",
    )
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"], "risk_distribution": report["risk_prediction_distribution"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
