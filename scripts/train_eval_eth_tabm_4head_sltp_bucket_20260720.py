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

import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import eval_omega4_1_atr_safety_sltp_20260622 as atr_safety  # noqa: E402
import train_eval_eth_tabm_4head_sltp_20260720 as reg_variant  # noqa: E402


MODEL_ID = "eth_tabm_4head_sltp_bucket_20260720"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

# Same live-proven safety envelope as the regression variant (reg_variant.MIN_TP/MAX_TP/MIN_SL/MAX_SL).
MIN_TP = reg_variant.MIN_TP
MAX_TP = reg_variant.MAX_TP
MIN_SL = reg_variant.MIN_SL
MAX_SL = reg_variant.MAX_SL
N_LEVELS = 5
TP_LEVELS = np.linspace(MIN_TP, MAX_TP, N_LEVELS).astype(np.float32)
SL_LEVELS = np.linspace(MIN_SL, MAX_SL, N_LEVELS).astype(np.float32)


@dataclass(frozen=True)
class FourHeadBucketConfig(parent.ThreeHeadConfig):
    sltp_loss_weight: float = 0.35
    tp_capture_frac: float = 0.55
    sl_capture_frac: float = 0.90


class FourHeadTabMBucket(parent.ThreeHeadTabM):
    """ThreeHeadTabM plus a 4th head that classifies entry-time TP/SL into discrete safety-envelope
    buckets (per side) instead of regressing a continuous value. Classification + balanced class
    weights lets the rare "genuinely large opportunity" bucket get real gradient signal even though
    most rows' realized mfe/mae are small -- the regression variant (train_eval_eth_tabm_4head_sltp_
    20260720.py) collapsed to always predicting near the floor because Huber loss is dominated by the
    bulk of small targets."""

    def __init__(self, n_features: int, *, cfg: FourHeadBucketConfig) -> None:
        super().__init__(n_features, cfg=cfg)
        self.sltp_head = nn.Linear(int(cfg.hidden), 4 * N_LEVELS)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encode(x)
        raw = self.sltp_head(h)
        return {
            "direction": self.direction_head(h),
            "quality": self.quality_head(h),
            "exit": self.exit_head(h),
            "sltp_tp_long": raw[..., 0 * N_LEVELS : 1 * N_LEVELS],
            "sltp_sl_long": raw[..., 1 * N_LEVELS : 2 * N_LEVELS],
            "sltp_tp_short": raw[..., 2 * N_LEVELS : 3 * N_LEVELS],
            "sltp_sl_short": raw[..., 3 * N_LEVELS : 4 * N_LEVELS],
        }


def _bucket_index(values: np.ndarray, levels: np.ndarray) -> np.ndarray:
    diffs = np.abs(values[:, None] - levels[None, :])
    return np.argmin(diffs, axis=1).astype(np.int64)


def _sltp_bucket_targets(action: np.ndarray, mfe: np.ndarray, mae: np.ndarray, *, tp_capture_frac: float, sl_capture_frac: float) -> tuple[np.ndarray, np.ndarray]:
    tp_cont = np.clip(np.abs(mfe) * float(tp_capture_frac), MIN_TP, MAX_TP)
    sl_cont = np.clip(np.abs(mae) * float(sl_capture_frac), MIN_SL, MAX_SL)
    tp_bucket = _bucket_index(tp_cont, TP_LEVELS)
    sl_bucket = _bucket_index(sl_cont, SL_LEVELS)
    del action
    return tp_bucket, sl_bucket


def _fit_expert_4head_bucket(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    tp_bucket: np.ndarray,
    sl_bucket: np.ndarray,
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
    cfg: FourHeadBucketConfig,
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
    tp_bucket_np = np.asarray(tp_bucket, dtype=np.int64)
    sl_bucket_np = np.asarray(sl_bucket, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    active = y_dir_np != 0
    tp_w = np.zeros(len(y_dir_np), dtype=np.float32)
    sl_w = np.zeros(len(y_dir_np), dtype=np.float32)
    if bool(active.any()):
        tp_w[active] = compute_sample_weight(class_weight="balanced", y=tp_bucket_np[active]).astype(np.float32)
        sl_w[active] = compute_sample_weight(class_weight="balanced", y=sl_bucket_np[active]).astype(np.float32)
    tp_w *= route_w
    sl_w *= route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0 or float(tp_w.sum()) <= 0.0 or float(sl_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 4-head bucket sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = FourHeadTabMBucket(x_dir_np.shape[1], cfg=cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg.lr), weight_decay=float(cfg.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        torch.from_numpy(tp_bucket_np[train_idx]),
        torch.from_numpy(sl_bucket_np[train_idx]),
        torch.from_numpy(tp_w[train_idx]),
        torch.from_numpy(sl_w[train_idx]),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0

    def _sltp_bucket_loss(out: dict[str, torch.Tensor], yb: torch.Tensor, tp_b: torch.Tensor, sl_b: torch.Tensor, tp_wb: torch.Tensor, sl_wb: torch.Tensor) -> torch.Tensor:
        is_long = (yb == 1)[:, None, None]
        tp_logits = torch.where(is_long, out["sltp_tp_long"], out["sltp_tp_short"])
        sl_logits = torch.where(is_long, out["sltp_sl_long"], out["sltp_sl_short"])
        tgt_tp = tp_b[:, None].expand(-1, int(cfg.k)).reshape(-1)
        tgt_sl = sl_b[:, None].expand(-1, int(cfg.k)).reshape(-1)
        loss_tp_k = torch.nn.functional.cross_entropy(tp_logits.reshape(-1, N_LEVELS), tgt_tp, reduction="none").reshape(-1, int(cfg.k))
        loss_sl_k = torch.nn.functional.cross_entropy(sl_logits.reshape(-1, N_LEVELS), tgt_sl, reduction="none").reshape(-1, int(cfg.k))
        loss_tp = (loss_tp_k.mean(dim=1) * tp_wb).sum() / torch.clamp(tp_wb.sum(), min=1.0)
        loss_sl = (loss_sl_k.mean(dim=1) * sl_wb).sum() / torch.clamp(sl_wb.sum(), min=1.0)
        return loss_tp + loss_sl

    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for xb, yb, yqb, wb, qwb, tpb, slb, tpwb, slwb in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, yqb, wb, qwb, tpb, slb, tpwb, slwb = (t.to(device, non_blocking=True) for t in (xb, yb, yqb, wb, qwb, tpb, slb, tpwb, slwb))
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
            loss_sltp = _sltp_bucket_loss(out_dir, yb, tpb, slb, tpwb, slwb)
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
            vtp = torch.from_numpy(tp_bucket_np[val_idx]).to(device)
            vsl = torch.from_numpy(sl_bucket_np[val_idx]).to(device)
            vtpw = torch.from_numpy(tp_w[val_idx]).to(device)
            vslw = torch.from_numpy(sl_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vsltp = _sltp_bucket_loss(vo, vy, vtp, vsl, vtpw, vslw)
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
def _predict_sltp_bucket(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    cfg = FourHeadBucketConfig(**{k: v for k, v in payload["config"].items() if k in FourHeadBucketConfig.__dataclass_fields__})
    model = FourHeadTabMBucket(int(payload["n_features"]), cfg=cfg).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": [], "tp_long": [], "sl_long": [], "tp_short": [], "sl_short": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append(torch.softmax(out["quality"], dim=-1).mean(dim=1).detach().cpu().numpy())
        for name, levels in (("tp_long", TP_LEVELS), ("sl_long", SL_LEVELS), ("tp_short", TP_LEVELS), ("sl_short", SL_LEVELS)):
            probs = torch.softmax(out[f"sltp_{name}"], dim=-1).mean(dim=1).detach().cpu().numpy()
            level_idx = np.argmax(probs, axis=-1)
            chunks[name].append(levels[level_idx].astype(np.float64))
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in chunks.items()}


def _load_payloads_4head_bucket(payloads: dict[str, dict[str, Any]], *, device: torch.device) -> dict[str, tuple[FourHeadTabMBucket, dict[str, Any]]]:
    loaded: dict[str, tuple[FourHeadTabMBucket, dict[str, Any]]] = {}
    for expert, payload in payloads.items():
        cfg = FourHeadBucketConfig(**{k: v for k, v in payload["config"].items() if k in FourHeadBucketConfig.__dataclass_fields__})
        model = FourHeadTabMBucket(int(payload["n_features"]), cfg=cfg).to(device)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        loaded[expert] = (model, payload["scaler"])
    return loaded


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=ROOT / "tmp/causal_regen_20260516/zigzag_action_labels_parent72_loose_20260620")
    ap.add_argument("--quality-mode", default="same_as_direction")
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--exit-threshold", type=float, default=0.97)
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
    diag = reg_variant._load_path_diagnostics(Path(args.direction_label_dir), train_all=train_all_for_diag, eval_df=oos_raw)
    train_mfe = diag["train_mfe"][: len(train_raw)]
    train_mae = diag["train_mae"][: len(train_raw)]

    x_train = parent._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_raw["omega4_quality_action"].to_numpy(dtype=np.int64)
    tp_bucket, sl_bucket = _sltp_bucket_targets(y_train, train_mfe, train_mae, tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_quality = y_quality[:limit]
        tp_bucket = tp_bucket[:limit]
        sl_bucket = sl_bucket[:limit]
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

    cfg = FourHeadBucketConfig(sltp_loss_weight=float(args.sltp_loss_weight), tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert_4head_bucket(
            x_train, y_train, y_quality, tp_bucket, sl_bucket,
            train_fit_frame, x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(args.seed), epochs=int(args.epochs), device=device,
            model_path=out_dir / "models" / f"{expert}_4head_bucket_tabm.pt",
            cfg=cfg,
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_sltp_bucket(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
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

    loaded_models = _load_payloads_4head_bucket(models, device=device)

    val_dec_atr, val_atr_diag = atr_safety._apply_atr_safety_sltp(
        val_dec_base, val_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    oos_dec_atr, oos_atr_diag = atr_safety._apply_atr_safety_sltp(
        oos_dec_base, oos_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    val_dec_learned, val_learned_diag = reg_variant._apply_learned_sltp(val_dec_base, val_sltp_preds, val_route)
    oos_dec_learned, oos_learned_diag = reg_variant._apply_learned_sltp(oos_dec_base, oos_sltp_preds, oos_route)

    val_m_atr = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_atr = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_m_learned = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_learned = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_learned, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)

    report = {
        "model_id": MODEL_ID,
        "design": "FourHeadTabMBucket: same direction/quality/exit heads as the regression sltp variant, but the 4th head classifies TP/SL into 5 discrete levels per side (cross-entropy + balanced class weights) spanning the same [0.075,0.22]/[0.04,0.12] safety envelope, instead of Huber-regressing a continuous value. Intended to fix the regression variant's collapse-to-floor failure mode.",
        "caveats": [
            "fresh_forward_bar_by_bar=true for the backtest replay itself, but train/val/oos split is this script family's legacy convention (train<2025-10-01, val>=2025-10-01 in 2025, oos=full 2026), NOT the project's canonical 2025-09-01..12-31 / 2026-01-01..03-31 split -- rerun on canonical split before treating as promotion evidence.",
            "trade_ledgers_used_as_input=false; bucket targets derived from zigzag_path_mfe/mae are training labels only, never inference inputs.",
            "Predecessor: train_eval_eth_tabm_4head_sltp_20260720.py (continuous regression) collapsed to predicting near MIN_TP/MIN_SL for both val and oos, statistically indistinguishable from the ATR floor -- this bucket variant exists specifically to test whether balanced-class-weight classification escapes that collapse.",
            "exit_threshold raised from the live default (0.95) in the predecessor run to 0.97 here: at 0.95 the exit head closed ~100% of trades before any TP/SL barrier could bind, making the sltp mechanism untestable; 0.97 is a compromise so barriers occasionally bind without fully reverting to that failure mode.",
            "direction/quality/exit heads are retrained from scratch (new architecture can't warm-start from live h48qual/zig075 bundles) -- not a drop-in live replacement even if the sltp mechanism wins here.",
        ],
        "quality_threshold": float(args.quality_threshold),
        "exit_threshold": float(args.exit_threshold),
        "exit_label_diag": exit_diag,
        "sltp_targets": {
            "tp_capture_frac": float(args.tp_capture_frac), "sl_capture_frac": float(args.sl_capture_frac), "sltp_loss_weight": float(args.sltp_loss_weight),
            "n_levels": N_LEVELS, "tp_levels": TP_LEVELS.tolist(), "sl_levels": SL_LEVELS.tolist(),
        },
        "summaries": summaries,
        "results": {
            "baseline_atr_fixed_formula": {"validation": val_m_atr, "oos": oos_m_atr, "validation_atr_diag": val_atr_diag, "oos_atr_diag": oos_atr_diag},
            "learned_sltp_bucket_head": {"validation": val_m_learned, "oos": oos_m_learned, "validation_sltp_diag": val_learned_diag, "oos_sltp_diag": oos_learned_diag},
        },
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": cfg.__dict__}, out_dir / "eth_4head_bucket_tabm_bundle.pt")
    print(json.dumps({
        "report": str(out_dir / "report.json"),
        "baseline_atr": {"validation": val_m_atr, "oos": oos_m_atr},
        "learned_sltp_bucket": {"validation": val_m_learned, "oos": oos_m_learned},
    }, ensure_ascii=False, indent=2, default=omega._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
