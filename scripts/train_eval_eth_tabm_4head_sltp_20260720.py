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
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_safety  # noqa: E402


MODEL_ID = "eth_tabm_4head_sltp_20260720"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

# Safety envelope: identical to the live Omega4.6.1 ATR barrier bounds (docs/model_contracts/
# omega4_6_1_full_architecture_blueprint_20260706.md L3). The learned head can only choose WHERE
# inside this proven-safe range the barrier sits -- it can never widen the range itself.
MIN_TP = 0.075
MAX_TP = 0.22
MIN_SL = 0.040
MAX_SL = 0.12


@dataclass(frozen=True)
class FourHeadConfig(parent.ThreeHeadConfig):
    sltp_loss_weight: float = 0.35
    tp_capture_frac: float = 0.55
    sl_capture_frac: float = 0.90


class FourHeadTabM(parent.ThreeHeadTabM):
    """ThreeHeadTabM plus a 4th head predicting entry-time TP/SL price-move targets from the same
    shared encoder used by direction/quality, instead of the fixed atr_pct * const formula."""

    def __init__(self, n_features: int, *, cfg: FourHeadConfig) -> None:
        super().__init__(n_features, cfg=cfg)
        self.sltp_head = nn.Linear(int(cfg.hidden), 4)  # [tp_long_raw, sl_long_raw, tp_short_raw, sl_short_raw]

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        return {
            "direction": self.direction_head(h),
            "quality": self.quality_head(h),
            "exit": self.exit_head(h),
            "sltp": self.sltp_head(h),
        }


def _sltp_price_moves(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.sigmoid(raw)
    tp_long = MIN_TP + (MAX_TP - MIN_TP) * z[..., 0]
    sl_long = MIN_SL + (MAX_SL - MIN_SL) * z[..., 1]
    tp_short = MIN_TP + (MAX_TP - MIN_TP) * z[..., 2]
    sl_short = MIN_SL + (MAX_SL - MIN_SL) * z[..., 3]
    return tp_long, sl_long, tp_short, sl_short


def _sltp_targets(action: np.ndarray, mfe: np.ndarray, mae: np.ndarray, *, tp_capture_frac: float, sl_capture_frac: float) -> tuple[np.ndarray, np.ndarray]:
    tp = np.clip(np.abs(mfe) * float(tp_capture_frac), MIN_TP, MAX_TP)
    sl = np.clip(np.abs(mae) * float(sl_capture_frac), MIN_SL, MAX_SL)
    active = action != 0
    tp = np.where(active, tp, 0.0).astype(np.float32)
    sl = np.where(active, sl, 0.0).astype(np.float32)
    return tp, sl


def _load_path_diagnostics(direction_label_dir: Path, *, train_all: pd.DataFrame, eval_df: pd.DataFrame) -> dict[str, np.ndarray]:
    label_2025 = omega4._read_labels(direction_label_dir, 2025, require_diagnostics=True)
    label_2026 = omega4._read_labels(direction_label_dir, 2026, require_diagnostics=True)
    _train_all_aligned, train_diag = omega._align(train_all[["timestamp"]], label_2025, "sltp-head train diagnostics")
    _eval_df_aligned, eval_diag = omega._align(eval_df[["timestamp"]], label_2026, "sltp-head oos diagnostics")
    if len(train_diag) != len(train_all) or len(eval_diag) != len(eval_df):
        raise RuntimeError("sltp-head diagnostic alignment changed row count")
    return {
        "train_mfe": pd.to_numeric(train_diag["zigzag_path_mfe"], errors="raise").to_numpy(dtype=np.float64),
        "train_mae": pd.to_numeric(train_diag["zigzag_path_mae"], errors="raise").to_numpy(dtype=np.float64),
        "eval_mfe": pd.to_numeric(eval_diag["zigzag_path_mfe"], errors="raise").to_numpy(dtype=np.float64),
        "eval_mae": pd.to_numeric(eval_diag["zigzag_path_mae"], errors="raise").to_numpy(dtype=np.float64),
    }


def _fit_expert_4head(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    tp_target: np.ndarray,
    sl_target: np.ndarray,
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
    cfg: FourHeadConfig,
    direction_class_weights: dict[int, float],
    quality_class_weights: dict[int, float],
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_qual_np = np.asarray(y_qual, dtype=np.int64)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    tp_np = np.asarray(tp_target, dtype=np.float32)
    sl_np = np.asarray(sl_target, dtype=np.float32)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    dir_w *= np.asarray([float(direction_class_weights.get(int(y), 1.0)) for y in y_dir_np], dtype=np.float32)
    qual_w *= np.asarray([float(quality_class_weights.get(int(y), 1.0)) for y in y_qual_np], dtype=np.float32)
    sltp_w = route_w * (y_dir_np != 0).astype(np.float32)
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0 or float(sltp_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 4-head sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = FourHeadTabM(x_dir_np.shape[1], cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        torch.from_numpy(tp_np[train_idx]),
        torch.from_numpy(sl_np[train_idx]),
        torch.from_numpy(sltp_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0

    def _sltp_loss(out: dict[str, torch.Tensor], yb: torch.Tensor, tp_b: torch.Tensor, sl_b: torch.Tensor, w_b: torch.Tensor) -> torch.Tensor:
        tp_long, sl_long, tp_short, sl_short = _sltp_price_moves(out["sltp"])
        is_long = (yb == 1).float()[:, None]
        is_short = (yb == 2).float()[:, None]
        pred_tp = tp_long * is_long + tp_short * is_short
        pred_sl = sl_long * is_long + sl_short * is_short
        tgt_tp = tp_b[:, None].expand(-1, int(cfg.k))
        tgt_sl = sl_b[:, None].expand(-1, int(cfg.k))
        loss_tp = torch.nn.functional.smooth_l1_loss(pred_tp, tgt_tp, reduction="none").mean(dim=1)
        loss_sl = torch.nn.functional.smooth_l1_loss(pred_sl, tgt_sl, reduction="none").mean(dim=1)
        return ((loss_tp + loss_sl) * w_b).sum() / torch.clamp(w_b.sum(), min=1.0)

    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yqb, wb, qwb, tpb, slb, sw in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, yqb, wb, qwb, tpb, slb, sw = (t.to(device, non_blocking=True) for t in (xb, yb, yqb, wb, qwb, tpb, slb, sw))
            xe, ye, we = xe.to(device, non_blocking=True), ye.to(device, non_blocking=True), we.to(device, non_blocking=True)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3), yb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none"
            ).reshape(-1, int(cfg.k))
            loss_qual_k = torch.nn.functional.cross_entropy(
                out_dir["quality"].reshape(-1, 3), yqb[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none"
            ).reshape(-1, int(cfg.k))
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none"
            ).reshape(-1, int(cfg.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k.mean(dim=1) * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss_sltp = _sltp_loss(out_dir, yb, tpb, slb, sw)
            loss = loss_dir + float(cfg.quality_loss_weight) * loss_qual + float(cfg.exit_loss_weight) * loss_exit + float(cfg.sltp_loss_weight) * loss_sltp
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_dir_np[val_idx]).to(device)
            vy = torch.from_numpy(y_dir_np[val_idx]).to(device)
            vqy = torch.from_numpy(y_qual_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            vtp = torch.from_numpy(tp_np[val_idx]).to(device)
            vsl = torch.from_numpy(sl_np[val_idx]).to(device)
            vsw = torch.from_numpy(sltp_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vsltp = _sltp_loss(vo, vy, vtp, vsl, vsw)
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(cfg.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                    + float(cfg.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
                    + float(cfg.sltp_loss_weight) * vsltp
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
            if stale >= int(cfg.patience):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": cfg.__dict__,
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
def _predict_sltp(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    cfg = FourHeadConfig(**{k: v for k, v in payload["config"].items() if k in FourHeadConfig.__dataclass_fields__})
    model = FourHeadTabM(int(payload["n_features"]), cfg=cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": [], "tp_long": [], "sl_long": [], "tp_short": [], "sl_short": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
        tp_long, sl_long, tp_short, sl_short = _sltp_price_moves(out["sltp"])
        chunks["tp_long"].append(tp_long.mean(dim=1).detach().cpu().numpy())
        chunks["sl_long"].append(sl_long.mean(dim=1).detach().cpu().numpy())
        chunks["tp_short"].append(tp_short.mean(dim=1).detach().cpu().numpy())
        chunks["sl_short"].append(sl_short.mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _load_payloads_4head(payloads: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[FourHeadTabM, dict[str, Any]]]:
    loaded: dict[str, tuple[FourHeadTabM, dict[str, Any]]] = {}
    for expert, payload in payloads.items():
        cfg = FourHeadConfig(**{k: v for k, v in payload["config"].items() if k in FourHeadConfig.__dataclass_fields__})
        model = FourHeadTabM(int(payload["n_features"]), cfg=cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[expert] = (model, payload["scaler"])
    return loaded


def _apply_learned_sltp(dec: pd.DataFrame, sltp_by_expert: dict[str, dict[str, np.ndarray]], route: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    side = pd.to_numeric(out["side"], errors="coerce").fillna(0).to_numpy(dtype=np.int64) if "side" in out.columns else np.zeros(len(out), dtype=np.int64)
    tp = np.zeros(len(out), dtype=np.float64)
    sl = np.zeros(len(out), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if not bool(mask.any()):
            continue
        preds = sltp_by_expert[expert]
        long_mask = mask & (side == 1)
        short_mask = mask & (side == -1)
        tp[long_mask] = preds["tp_long"][long_mask]
        sl[long_mask] = preds["sl_long"][long_mask]
        tp[short_mask] = preds["tp_short"][short_mask]
        sl[short_mask] = preds["sl_short"][short_mask]
    out.loc[active, "take_profit"] = tp[active]
    out.loc[active, "stop_loss"] = sl[active]
    out.loc[~active, ["take_profit", "stop_loss"]] = 0.0
    active_tp = tp[active]
    active_sl = sl[active]
    diag = {
        "active_rows": int(active.sum()),
        "tp_p50": float(np.quantile(active_tp, 0.50)) if len(active_tp) else 0.0,
        "tp_p90": float(np.quantile(active_tp, 0.90)) if len(active_tp) else 0.0,
        "sl_p50": float(np.quantile(active_sl, 0.50)) if len(active_sl) else 0.0,
        "sl_p90": float(np.quantile(active_sl, 0.90)) if len(active_sl) else 0.0,
        "tp_at_min_floor_rate": float((np.isclose(active_tp, MIN_TP)).mean()) if len(active_tp) else 0.0,
        "sl_at_min_floor_rate": float((np.isclose(active_sl, MIN_SL)).mean()) if len(active_sl) else 0.0,
    }
    return out, diag


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620")
    ap.add_argument("--quality-mode", default="same_as_direction")
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--exit-threshold", type=float, default=0.95)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--max-exit-samples", type=int, default=0)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260720)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--tp-capture-frac", type=float, default=0.55)
    ap.add_argument("--sl-capture-frac", type=float, default=0.90)
    ap.add_argument("--sltp-loss-weight", type=float, default=0.35)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--atr-tp-mult", type=float, default=12.0)
    ap.add_argument("--atr-sl-mult", type=float, default=6.0)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    omega4._seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode=str(args.quality_mode),
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=1.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]

    train_all_for_diag = pd.concat([train_raw, val_raw], ignore_index=True)
    diag = _load_path_diagnostics(Path(args.direction_label_dir), train_all=train_all_for_diag, eval_df=oos_raw)
    train_mfe, val_mfe = diag["train_mfe"][: len(train_raw)], diag["train_mfe"][len(train_raw) :]
    train_mae, val_mae = diag["train_mae"][: len(train_raw)], diag["train_mae"][len(train_raw) :]

    x_train = parent._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_raw["omega4_quality_action"].to_numpy(dtype=np.int64)
    tp_target, sl_target = _sltp_targets(y_train, train_mfe, train_mae, tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_quality = y_quality[:limit]
        tp_target = tp_target[:limit]
        sl_target = sl_target[:limit]
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
    x_exit = parent._exit_input_from_position_rows(x_exit_raw, base_cols)

    cfg = FourHeadConfig(sltp_loss_weight=float(args.sltp_loss_weight), tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_4head(
            x_train, y_train, y_quality, tp_target, sl_target,
            train_fit_frame, x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(args.seed), epochs=int(args.epochs), device=device,
            model_path=out_dir / "models" / f"{expert}_4head_tabm.pt",
            cfg=cfg, direction_class_weights={0: 1.0, 1: 1.0, 2: 1.0}, quality_class_weights={0: 1.0, 1: 1.0, 2: 1.0},
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_sltp(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = parent._routed(preds, route, "direction", 3)
        quality = parent._routed(preds, route, "quality", 3)
        out = parent._prediction_output(frame, direction, quality, threshold=float(args.quality_threshold), prefix="omega1_regime3_expertdq_oof")
        return x, out, route, preds

    x_val, val_src, val_route, val_sltp_preds = predict_frame(val_raw)
    x_oos, oos_src_oof, oos_route, oos_sltp_preds = predict_frame(oos_raw)
    oos_src = oos_src_oof.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in oos_src_oof.columns})
    val_dec_base = parent._to_decisions(val_src, oof=True)
    oos_dec_base = parent._to_decisions(oos_src, oof=False)

    loaded_models = _load_payloads_4head(models, device=device)

    val_dec_atr, val_atr_diag = atr_safety._apply_atr_safety_sltp(
        val_dec_base, val_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    oos_dec_atr, oos_atr_diag = atr_safety._apply_atr_safety_sltp(
        oos_dec_base, oos_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    val_dec_learned, val_learned_diag = _apply_learned_sltp(val_dec_base, val_sltp_preds, val_route)
    oos_dec_learned, oos_learned_diag = _apply_learned_sltp(oos_dec_base, oos_sltp_preds, oos_route)

    val_m_atr = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_atr = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_m_learned = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_learned = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)

    report = {
        "model_id": MODEL_ID,
        "design": "FourHeadTabM: direction/quality/exit heads identical to omega4_3head_parent72_loose_entry_quality; a 4th sltp_head shares the same encoder and predicts entry-time TP/SL price-move targets, sigmoid-mapped into the exact same [0.075,0.22]/[0.04,0.12] safety envelope the live ATR formula already uses. Trained jointly (Huber loss) against tp_capture_frac*zigzag_path_mfe / sl_capture_frac*zigzag_path_mae, active rows only.",
        "caveats": [
            "fresh_forward_bar_by_bar=true for the backtest replay itself (causal walk, no saved-ledger reuse), but the train/val/oos split here is this script family's legacy convention (train<2025-10-01, val>=2025-10-01 within 2025, oos=full 2026), NOT the project's canonical val 2025-09-01..12-31 / oos 2026-01-01..03-31 split -- rerun on the canonical split before treating this as promotion evidence.",
            "trade_ledgers_used_as_input=false; sltp regression targets (zigzag_path_mfe/mae) are training labels only, never model inputs at inference.",
            "Prior 2026-06-16/17 sltp-head attempts in this repo (omega1_2_8d-8i) never beat baseline when isolated from the EV-veto family -- treat any win here as provisional until it also survives an ablation that isolates the sltp mechanism the same way.",
            "direction/quality/exit heads are retrained from scratch here (new architecture, can't warm-start from the live h48qual/zig075 bundles), so this is not a drop-in replacement for the live model even if the sltp mechanism itself wins -- it would need the live components retrained with this 4th head.",
        ],
        "quality_threshold": float(args.quality_threshold),
        "exit_threshold": float(args.exit_threshold),
        "exit_label_diag": exit_diag,
        "sltp_targets": {"tp_capture_frac": float(args.tp_capture_frac), "sl_capture_frac": float(args.sl_capture_frac), "sltp_loss_weight": float(args.sltp_loss_weight), "min_tp": MIN_TP, "max_tp": MAX_TP, "min_sl": MIN_SL, "max_sl": MAX_SL},
        "summaries": summaries,
        "results": {
            "baseline_atr_fixed_formula": {"validation": val_m_atr, "oos": oos_m_atr, "validation_atr_diag": val_atr_diag, "oos_atr_diag": oos_atr_diag},
            "learned_sltp_head": {"validation": val_m_learned, "oos": oos_m_learned, "validation_sltp_diag": val_learned_diag, "oos_sltp_diag": oos_learned_diag},
        },
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": cfg.__dict__}, out_dir / "eth_4head_tabm_bundle.pt")
    print(json.dumps({
        "report": str(out_dir / "report.json"),
        "baseline_atr": {"validation": val_m_atr, "oos": oos_m_atr},
        "learned_sltp": {"validation": val_m_learned, "oos": oos_m_learned},
    }, ensure_ascii=False, indent=2, default=omega._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
