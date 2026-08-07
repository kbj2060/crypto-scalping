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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_3head_cmamba_replacement_20260621"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


@dataclass(frozen=True)
class CMambaConfig:
    seq_len: int = 16
    d_model: int = 32
    blocks: int = 1
    kernel: int = 5
    dropout: float = 0.08
    batch_size: int = 1024
    lr: float = 1.5e-3
    weight_decay: float = 2.0e-4
    quality_loss_weight: float = 0.80
    exit_loss_weight: float = 1.15


CFG = CMambaConfig()


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


class CausalGatedConvBlock(nn.Module):
    def __init__(self, d_model: int, kernel: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(int(d_model))
        self.depthwise = nn.Conv1d(
            int(d_model),
            int(d_model),
            kernel_size=int(kernel),
            groups=int(d_model),
            padding=int(kernel) - 1,
        )
        self.mix = nn.Linear(int(d_model), int(d_model) * 2)
        self.dropout = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.norm(x)
        z = self.depthwise(z.transpose(1, 2))[:, :, : x.shape[1]].transpose(1, 2)
        value, gate = self.mix(z).chunk(2, dim=-1)
        z = torch.nn.functional.silu(value) * torch.sigmoid(gate)
        return x + self.dropout(z)


class ThreeHeadCMamba(nn.Module):
    def __init__(self, n_features: int, cfg: CMambaConfig = CFG) -> None:
        super().__init__()
        self.n_features = int(n_features)
        self.seq_len = int(cfg.seq_len)
        self.in_proj = nn.Linear(int(n_features), int(cfg.d_model))
        self.blocks = nn.ModuleList(CausalGatedConvBlock(int(cfg.d_model), int(cfg.kernel), float(cfg.dropout)) for _ in range(int(cfg.blocks)))
        self.norm = nn.LayerNorm(int(cfg.d_model))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_head = nn.Linear(int(cfg.d_model), 3)
        self.quality_head = nn.Linear(int(cfg.d_model), 3)
        self.exit_head = nn.Linear(int(cfg.d_model), 2)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.in_proj(x)
        for block in self.blocks:
            z = block(z)
        h = self.dropout(self.norm(z[:, -1, :]))
        return {"direction": self.direction_head(h), "quality": self.quality_head(h), "exit": self.exit_head(h)}


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    out = ((arr - mean) / std).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized CMamba matrix")
    return out, {"mean": mean, "std": std, "columns": list(x.columns)}


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    cols = list(scaler["columns"])
    if list(x.columns) != cols:
        raise RuntimeError("CMamba feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = ((arr - scaler["mean"]) / scaler["std"]).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite standardized CMamba inference matrix")
    return out


def _seq_tensor(x: np.ndarray, idx: np.ndarray, seq_len: int) -> np.ndarray:
    out = np.empty((len(idx), int(seq_len), x.shape[1]), dtype=np.float32)
    for j, end in enumerate(idx.astype(np.int64)):
        out[j] = x[end - int(seq_len) + 1 : end + 1]
    return out


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    route_frame: pd.DataFrame,
    x_exit: pd.DataFrame,
    y_exit: np.ndarray,
    exit_route_frame: pd.DataFrame,
    *,
    expert_idx: int,
    epochs: int,
    seed: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _, scaler = _standardize_fit(x_all)
    x_dir_np = _standardize_apply(x_dir, scaler)
    x_exit_np = _standardize_apply(x_exit, scaler)
    seq_len = int(CFG.seq_len)
    idx_dir = np.arange(seq_len - 1, len(x_dir_np), dtype=np.int64)
    idx_exit = np.arange(seq_len - 1, len(x_exit_np), dtype=np.int64)
    if len(idx_dir) < 512 or len(idx_exit) < 256:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} insufficient CMamba sequence rows")

    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w

    split = max(int(len(idx_dir) * 0.85), min(len(idx_dir) - 1, 512))
    exit_split = max(int(len(idx_exit) * 0.85), min(len(idx_exit) - 1, 256))
    tr_idx, va_idx = idx_dir[:split], idx_dir[split:]
    ex_tr_idx, ex_va_idx = idx_exit[:exit_split], idx_exit[exit_split:]

    ds_dir = TensorDataset(
        torch.from_numpy(_seq_tensor(x_dir_np, tr_idx, seq_len)),
        torch.from_numpy(y_dir_np[tr_idx]),
        torch.from_numpy(y_qual_np[tr_idx]),
        torch.from_numpy(dir_w[tr_idx]),
        torch.from_numpy(qual_w[tr_idx]),
    )
    ds_exit = TensorDataset(
        torch.from_numpy(_seq_tensor(x_exit_np, ex_tr_idx, seq_len)),
        torch.from_numpy(y_exit_np[ex_tr_idx]),
        torch.from_numpy(ex_w[ex_tr_idx]),
    )
    dl_dir = DataLoader(ds_dir, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(CFG.batch_size), shuffle=True, drop_last=False)
    model = ThreeHeadCMamba(x_dir_np.shape[1], CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    for epoch in range(int(epochs)):
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yd, yq, wd, wq in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yd, yq, wd, wq = xb.to(device), yd.to(device), yq.to(device), wd.to(device), wq.to(device)
            xe, ye, we = xe.to(device), ye.to(device), we.to(device)
            od = model(xb)
            oe = model(xe)
            loss_dir = (torch.nn.functional.cross_entropy(od["direction"], yd, reduction="none") * wd).sum() / torch.clamp(wd.sum(), min=1.0)
            loss_qual = (torch.nn.functional.cross_entropy(od["quality"], yq, reduction="none") * wq).sum() / torch.clamp(wq.sum(), min=1.0)
            loss_exit = (torch.nn.functional.cross_entropy(oe["exit"], ye, reduction="none") * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_qual + float(CFG.exit_loss_weight) * loss_exit
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(_seq_tensor(x_dir_np, va_idx, seq_len)).to(device)
            ve = torch.from_numpy(_seq_tensor(x_exit_np, ex_va_idx, seq_len)).to(device)
            vyd = torch.from_numpy(y_dir_np[va_idx]).to(device)
            vyq = torch.from_numpy(y_qual_np[va_idx]).to(device)
            vye = torch.from_numpy(y_exit_np[ex_va_idx]).to(device)
            vwd = torch.from_numpy(dir_w[va_idx]).to(device)
            vwq = torch.from_numpy(qual_w[va_idx]).to(device)
            vwe = torch.from_numpy(ex_w[ex_va_idx]).to(device)
            od = model(vx)
            oe = model(ve)
            vloss = (
                (torch.nn.functional.cross_entropy(od["direction"], vyd, reduction="none") * vwd).sum() / torch.clamp(vwd.sum(), min=1.0)
                + float(CFG.quality_loss_weight) * (torch.nn.functional.cross_entropy(od["quality"], vyq, reduction="none") * vwq).sum() / torch.clamp(vwq.sum(), min=1.0)
                + float(CFG.exit_loss_weight) * (torch.nn.functional.cross_entropy(oe["exit"], vye, reduction="none") * vwe).sum() / torch.clamp(vwe.sum(), min=1.0)
            )
            val_loss = float(vloss.detach().cpu())
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "scaler": scaler, "n_features": int(x_dir_np.shape[1]), "expert": hard.EXPERT_NAMES[int(expert_idx)]}, model_path)
    return {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_dir_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(epochs),
        "input_columns": list(x_dir.columns),
    }


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = ThreeHeadCMamba(int(payload["n_features"]), CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    seq_len = int(CFG.seq_len)
    out = {
        "direction": np.zeros((len(x), 3), dtype=np.float64),
        "quality": np.zeros((len(x), 3), dtype=np.float64),
        "exit": np.zeros((len(x), 2), dtype=np.float64),
    }
    out["direction"][:, 0] = 1.0
    out["quality"][:, 0] = 1.0
    out["exit"][:, 0] = 1.0
    if len(x_np) < seq_len:
        return out
    idx = np.arange(seq_len - 1, len(x_np), dtype=np.int64)
    for start in range(0, len(idx), 2048):
        part = idx[start : start + 2048]
        xb = torch.from_numpy(_seq_tensor(x_np, part, seq_len)).to(device)
        pred = model(xb)
        for head in out:
            out[head][part] = torch.softmax(pred[head], dim=-1).detach().cpu().numpy()
    return out


@torch.no_grad()
def _predict_exit_prob_sequence_loaded(model: ThreeHeadCMamba, scaler: dict[str, Any], xseq: pd.DataFrame, *, device: torch.device) -> float:
    seq_len = int(CFG.seq_len)
    if len(xseq) < seq_len:
        return 0.0
    x_np = _standardize_apply(xseq.iloc[-seq_len:].reset_index(drop=True), scaler)
    xb = torch.from_numpy(x_np[None, :, :]).to(device)
    return float(torch.softmax(model(xb)["exit"], dim=-1).detach().cpu().numpy()[0, 1])


def _load_runtime_models(models: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[ThreeHeadCMamba, dict[str, Any]]]:
    loaded: dict[str, tuple[ThreeHeadCMamba, dict[str, Any]]] = {}
    for expert, payload in models.items():
        model = ThreeHeadCMamba(int(payload["n_features"]), CFG).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[expert] = (model, payload["scaler"])
    return loaded


def _metrics_with_shared_exit_cmamba(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[ThreeHeadCMamba, dict[str, Any]]],
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
                start = max(0, int(i) - int(CFG.seq_len) + 1)
                xseq = base_x.iloc[start : int(i) + 1].copy().reset_index(drop=True)
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
                    xseq.loc[xseq.index[-1], col] = val
                expert = hard.EXPERT_NAMES[int(route[i])]
                model, scaler = loaded_models[expert]
                prob = _predict_exit_prob_sequence_loaded(model, scaler, xseq, device=device)
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
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-train-rows", type=int, default=5000)
    ap.add_argument("--max-exit-samples", type=int, default=5000)
    ap.add_argument("--quality-threshold", type=float, default=0.70)
    ap.add_argument("--exit-threshold", type=float, default=0.70)
    ap.add_argument("--seed", type=int, default=260621)
    ap.add_argument("--out-suffix", default="smoke_e1_train5k_exit5k_q070_exit070")
    ap.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = torch.device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix)}"
    out_dir.mkdir(parents=True, exist_ok=True)
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
    limit = int(args.max_train_rows)
    train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True) if limit > 0 else train_raw
    x_train = parent._base_input(train_fit_frame, base_cols)
    y_train = train_fit_frame["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_fit_frame["omega4_quality_action"].to_numpy(dtype=np.int64)
    x_exit_raw, y_exit, frame_exit, exit_diag = omega4._build_exit_dataset_entry_label_terminal_giveback(
        frames["train_df"],
        frames["s_train_label"],
        fee=fee,
        slip=slip,
        cost_mult=3.0,
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
        payload = _fit_expert(
            x_train,
            y_train,
            y_quality,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            epochs=int(args.epochs),
            seed=int(args.seed),
            device=device,
            model_path=out_dir / "models" / f"{expert}_3head_cmamba.pt",
        )
        models[expert] = payload
        summaries[expert] = {"best_validation_loss": payload["best_validation_loss"], "epochs_ran": payload["epochs_ran"]}

    def predict(frame: pd.DataFrame, *, oof: bool) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix=prefix)
        return x, src, parent._to_decisions(src, oof=oof)

    x_val, val_src, val_dec = predict(frames["val_raw"], oof=True)
    x_oos, oos_src, oos_dec = predict(frames["oos_raw"], oof=False)
    val_src.to_csv(out_dir / "validation_predictions_2025_cmamba_q070.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_cmamba_q070.csv", index=False)
    no_exit = {
        "validation": omega._metrics(frames["val_raw"], val_dec, fee=fee, slip=slip, cost_mult=3.0),
        "oos": omega._metrics(frames["oos_raw"], oos_dec, fee=fee, slip=slip, cost_mult=3.0),
    }
    loaded_exit_models = _load_runtime_models(models, device=device)
    with_exit = {
        "validation": _metrics_with_shared_exit_cmamba(frames["val_raw"], x_val, val_dec, loaded_exit_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=3.0, device=device),
        "oos": _metrics_with_shared_exit_cmamba(frames["oos_raw"], x_oos, oos_dec, loaded_exit_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=3.0, device=device),
    }

    report = {
        "model_id": MODEL_ID,
        "design": "CPU-compatible CMamba-style replacement for Omega4 expert TabM backbones. This does not use mamba_ssm CUDA kernels; it uses causal depthwise-conv gated C-blocks with the same bull/bear/chop 3-head contract.",
        "config": CFG.__dict__,
        "input_contract": {"base_feature_count": len(base_cols), "position_feature_count": len(parent.POS_COLS), "total_features": len(base_cols) + len(parent.POS_COLS)},
        "label_contract": frames["label_contract"],
        "exit_label": {"mode": "entry_label_terminal_giveback", "diag": exit_diag},
        "summaries": summaries,
        "results": {"q0p70_no_exit_head": no_exit, "q0p70_exit_thr_0p70": with_exit},
        "artifacts": {
            "out_dir": str(out_dir),
            "report": str(out_dir / "report.json"),
            "validation_predictions": str(out_dir / "validation_predictions_2025_cmamba_q070.csv"),
            "oos_predictions": str(out_dir / "oos_predictions_2026_cmamba_q070.csv"),
        },
    }
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": CFG.__dict__, "model_class": "ThreeHeadCMambaStyle"}, out_dir / "true_3head_cmamba_bundle.pt")
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
