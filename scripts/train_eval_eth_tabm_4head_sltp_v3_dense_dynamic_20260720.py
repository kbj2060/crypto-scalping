#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
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
import train_eval_eth_tabm_4head_sltp_bucket_20260720 as bucket_v1  # noqa: E402


MODEL_ID = "eth_tabm_4head_sltp_v3_dense_dynamic_20260720"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

MIN_TP = bucket_v1.MIN_TP
MAX_TP = bucket_v1.MAX_TP
MIN_SL = bucket_v1.MIN_SL
MAX_SL = bucket_v1.MAX_SL
N_LEVELS = bucket_v1.N_LEVELS
TP_LEVELS = bucket_v1.TP_LEVELS
SL_LEVELS = bucket_v1.SL_LEVELS
FourHeadTabMBucket = bucket_v1.FourHeadTabMBucket
FourHeadBucketConfig = bucket_v1.FourHeadBucketConfig


def _dense_horizon_mfe_mae(frame: pd.DataFrame, *, horizon_bars: int) -> dict[str, np.ndarray]:
    """Unlike v1/v2 (targets only defined where zigzag_action already flags an active trade), this
    computes a hypothetical-long AND hypothetical-short forward-path MFE/MAE for EVERY bar, so the
    sltp head gets dense per-bar supervision about how much realistic opportunity currently exists
    for either side -- not just at the sparse moments a trade actually fires. The point (per the
    user's hypothesis): a head that has only ever seen entry-moment snapshots can't tell "conditions
    just turned bad mid-trade" from "conditions were never good" -- training it on every bar's own
    market state, for both sides, gives it the vocabulary to recognize a live regime shift and can be
    re-queried bar-by-bar (see _metrics_dynamic_ratchet_sltp) as a trailing/shrinking barrier instead
    of a value frozen at entry. Training-label only; never a model input at inference."""
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    n = len(close)
    mfe_long = np.zeros(n, dtype=np.float64)
    mae_long = np.zeros(n, dtype=np.float64)
    mfe_short = np.zeros(n, dtype=np.float64)
    mae_short = np.zeros(n, dtype=np.float64)
    for i in range(n - 1):
        entry = close[i]
        end = min(i + int(horizon_bars), n - 1)
        path = close[i + 1 : end + 1]
        if len(path) == 0:
            continue
        moves_long = (path - entry) / entry
        moves_short = (entry - path) / entry
        mfe_long[i] = float(moves_long.max())
        mae_long[i] = float(moves_long.min())
        mfe_short[i] = float(moves_short.max())
        mae_short[i] = float(moves_short.min())
    return {"mfe_long": mfe_long, "mae_long": mae_long, "mfe_short": mfe_short, "mae_short": mae_short}


def _dense_bucket_targets(dense: dict[str, np.ndarray], *, tp_capture_frac: float, sl_capture_frac: float) -> dict[str, np.ndarray]:
    tp_long = np.clip(np.abs(dense["mfe_long"]) * float(tp_capture_frac), MIN_TP, MAX_TP)
    sl_long = np.clip(np.abs(dense["mae_long"]) * float(sl_capture_frac), MIN_SL, MAX_SL)
    tp_short = np.clip(np.abs(dense["mfe_short"]) * float(tp_capture_frac), MIN_TP, MAX_TP)
    sl_short = np.clip(np.abs(dense["mae_short"]) * float(sl_capture_frac), MIN_SL, MAX_SL)
    return {
        "tp_long": bucket_v1._bucket_index(tp_long, TP_LEVELS),
        "sl_long": bucket_v1._bucket_index(sl_long, SL_LEVELS),
        "tp_short": bucket_v1._bucket_index(tp_short, TP_LEVELS),
        "sl_short": bucket_v1._bucket_index(sl_short, SL_LEVELS),
    }


def _fit_expert_4head_dense(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_qual: np.ndarray,
    buckets: dict[str, np.ndarray],
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
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    qual_w = compute_sample_weight(class_weight="balanced", y=y_qual_np).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w

    branch_labels = {k: np.asarray(v, dtype=np.int64) for k, v in buckets.items()}
    branch_weights = {k: compute_sample_weight(class_weight="balanced", y=v).astype(np.float32) * route_w for k, v in branch_labels.items()}
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0 or any(float(w.sum()) <= 0.0 for w in branch_weights.values()):
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid 4-head dense sample weights")

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
    branch_names = ("tp_long", "sl_long", "tp_short", "sl_short")
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_qual_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
        *(torch.from_numpy(branch_labels[k][train_idx]) for k in branch_names),
        *(torch.from_numpy(branch_weights[k][train_idx]) for k in branch_names),
    )
    ds_exit = TensorDataset(torch.from_numpy(x_exit_np[exit_train_idx]), torch.from_numpy(y_exit_np[exit_train_idx]), torch.from_numpy(ex_w[exit_train_idx]))
    dl_dir = DataLoader(ds_dir, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    dl_exit = DataLoader(ds_exit, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0

    def _dense_sltp_loss(out: dict[str, torch.Tensor], labels: dict[str, torch.Tensor], weights: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for name in branch_names:
            logits = out[f"sltp_{name}"].reshape(-1, N_LEVELS)
            tgt = labels[name][:, None].expand(-1, int(cfg.k)).reshape(-1)
            loss_k = torch.nn.functional.cross_entropy(logits, tgt, reduction="none").reshape(-1, int(cfg.k))
            w = weights[name]
            total = total + (loss_k.mean(dim=1) * w).sum() / torch.clamp(w.sum(), min=1.0)
        return total

    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        exit_iter = iter(dl_exit)
        for batch in dl_dir:
            xb, yb, yqb, wb, qwb = batch[:5]
            lbls = dict(zip(branch_names, batch[5:9]))
            wts = dict(zip(branch_names, batch[9:13]))
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb, yb, yqb, wb, qwb = (t.to(device, non_blocking=True) for t in (xb, yb, yqb, wb, qwb))
            lbls = {k: v.to(device, non_blocking=True) for k, v in lbls.items()}
            wts = {k: v.to(device, non_blocking=True) for k, v in wts.items()}
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
            loss_sltp = _dense_sltp_loss(out_dir, lbls, wts)
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
            v_lbls = {k: torch.from_numpy(branch_labels[k][val_idx]).to(device) for k in branch_names}
            v_wts = {k: torch.from_numpy(branch_weights[k][val_idx]).to(device) for k in branch_names}
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vqual = torch.nn.functional.cross_entropy(vo["quality"].reshape(-1, 3), vqy[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(cfg.k)).reshape(-1), reduction="none").reshape(-1, int(cfg.k))
            vsltp = _dense_sltp_loss(vo, v_lbls, v_wts)
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


def _metrics_dynamic_ratchet_sltp(
    frame: pd.DataFrame,
    base_x: pd.DataFrame,
    dec: pd.DataFrame,
    loaded_models: dict[str, tuple[Any, dict[str, Any]]],
    sltp_by_expert: dict[str, dict[str, np.ndarray]],
    route: np.ndarray,
    *,
    threshold: float,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    """Same bar-by-bar backtest as parent._metrics_with_shared_exit, but while a position is open the
    TP/SL barrier is re-tightened every bar from the sltp head's freshly re-evaluated opinion at that
    bar (ratchet-only: TP and SL can only shrink toward the current price over the life of the trade,
    never widen back out) -- this is the mechanism the user asked for: instead of freezing the barrier
    at entry-time market conditions, let it visibly react when the head's own read of "how much room is
    realistically left" narrows mid-trade."""
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
    ratchet_events = 0
    mfe = 0.0
    mae = 0.0
    trades = 0
    wins = 0
    long_entries = 0
    short_entries = 0
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos != 0:
            expert = hard.EXPERT_NAMES[int(route[i])]
            preds = sltp_by_expert[expert]
            fresh_tp = float(preds["tp_long"][i] if pos > 0 else preds["tp_short"][i])
            fresh_sl = float(preds["sl_long"][i] if pos > 0 else preds["sl_short"][i])
            if fresh_tp < take_profit - 1.0e-9 or fresh_sl < stop_loss - 1.0e-9:
                ratchet_events += 1
            take_profit = min(take_profit, fresh_tp)
            stop_loss = min(stop_loss, fresh_sl)
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
                    "pos_side": float(pos), "pos_hold_bars": float(hold), "pos_unrealized": float(unreal),
                    "pos_mfe": float(mfe), "pos_mae": float(mae), "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
                    "pos_dist_to_tp": float(take_profit - unreal), "pos_dist_to_sl": float(unreal + abs(stop_loss)),
                    "pos_notional": float(notional), "pos_leverage": float(leverage), "pos_exposure": float(notional * leverage),
                    "pos_tp": float(take_profit), "pos_sl": float(stop_loss),
                }
                for col, val in vals.items():
                    xrow[col] = val
                expert = hard.EXPERT_NAMES[int(route[i])]
                model, scaler = loaded_models[expert]
                prob = float(parent._predict_loaded_exit(model, scaler, xrow, device=device)[0, 1])
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
        "ratchet_events": int(ratchet_events),
    }


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
    ap.add_argument("--tp-capture-frac", type=float, default=0.70)
    ap.add_argument("--sl-capture-frac", type=float, default=0.90)
    ap.add_argument("--sltp-loss-weight", type=float, default=0.35)
    ap.add_argument("--label-horizon-bars", type=int, default=2016)
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

    x_train = parent._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = train_raw["omega4_quality_action"].to_numpy(dtype=np.int64)
    dense = _dense_horizon_mfe_mae(train_raw, horizon_bars=int(args.label_horizon_bars))
    buckets = _dense_bucket_targets(dense, tp_capture_frac=float(args.tp_capture_frac), sl_capture_frac=float(args.sl_capture_frac))
    label_diag = {
        "label_horizon_bars": int(args.label_horizon_bars),
        "rows": int(len(train_raw)),
        "bucket_counts": {k: {str(a): int(b) for a, b in zip(*np.unique(v, return_counts=True))} for k, v in buckets.items()},
        "mfe_long_p50": float(np.quantile(np.abs(dense["mfe_long"]), 0.50)),
        "mfe_long_p90": float(np.quantile(np.abs(dense["mfe_long"]), 0.90)),
        "mae_long_p50": float(np.quantile(np.abs(dense["mae_long"]), 0.50)),
        "mae_long_p90": float(np.quantile(np.abs(dense["mae_long"]), 0.90)),
    }

    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_train = x_train.iloc[:limit].reset_index(drop=True)
        y_train = y_train[:limit]
        y_quality = y_quality[:limit]
        buckets = {k: v[:limit] for k, v in buckets.items()}
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
        payload = _fit_expert_4head_dense(
            x_train, y_train, y_quality, buckets,
            train_fit_frame, x_exit, y_exit, frame_exit,
            expert_idx=idx, seed=int(args.seed), epochs=int(args.epochs), device=device,
            model_path=out_dir / "models" / f"{expert}_4head_dense_tabm.pt",
            cfg=cfg,
        )
        models[expert] = payload
        summaries[expert] = {"epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, dict[str, np.ndarray]]]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: bucket_v1._predict_sltp_bucket(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
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

    loaded_models = bucket_v1._load_payloads_4head_bucket(models, device=device)

    val_dec_atr, val_atr_diag = atr_safety._apply_atr_safety_sltp(
        val_dec_base, val_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    oos_dec_atr, oos_atr_diag = atr_safety._apply_atr_safety_sltp(
        oos_dec_base, oos_raw, atr_window=int(args.atr_window), tp_mult=float(args.atr_tp_mult), sl_mult=float(args.atr_sl_mult),
        min_tp=MIN_TP, min_sl=MIN_SL, max_tp=MAX_TP, max_sl=MAX_SL,
    )
    val_dec_static, val_static_diag = reg_variant._apply_learned_sltp(val_dec_base, val_sltp_preds, val_route)
    oos_dec_static, oos_static_diag = reg_variant._apply_learned_sltp(oos_dec_base, oos_sltp_preds, oos_route)

    val_m_atr = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_atr = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_atr, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_m_static = parent._metrics_with_shared_exit(val_raw, x_val, val_dec_static, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_static = parent._metrics_with_shared_exit(oos_raw, x_oos, oos_dec_static, loaded_models, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    val_m_dynamic = _metrics_dynamic_ratchet_sltp(val_raw, x_val, val_dec_static, loaded_models, val_sltp_preds, val_route, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)
    oos_m_dynamic = _metrics_dynamic_ratchet_sltp(oos_raw, x_oos, oos_dec_static, loaded_models, oos_sltp_preds, oos_route, threshold=float(args.exit_threshold), fee=fee, slip=slip, cost_mult=float(args.cost_mult), device=device)

    report = {
        "model_id": MODEL_ID,
        "design": "Same FourHeadTabMBucket architecture as bucket_v2, but (a) training labels are now dense: every bar gets a hypothetical-long AND hypothetical-short forward-horizon bucket target (not just bars where a trade actually fires), and (b) a new backtest variant re-queries the sltp head every bar WHILE a position is open and ratchets the TP/SL barrier tighter (never wider) toward whatever the head currently believes is realistic -- addressing the user's hypothesis that a barrier frozen at entry-time conditions doesn't react when the regime turns mid-trade.",
        "caveats": [
            "fresh_forward_bar_by_bar=true for the backtest replay itself (dynamic variant re-queries the model causally, only ever using that bar's own market state -- no future information), but train/val/oos split is this script family's legacy convention, NOT the project's canonical 2025-09-01..12-31 / 2026-01-01..03-31 split.",
            "trade_ledgers_used_as_input=false; dense bucket targets are training labels only, never inference inputs. The ratchet mechanism at inference/backtest time uses only the model's own live output at the current bar, never a saved ledger.",
            "static_learned_sltp here uses the SAME dense-trained model as dynamic_ratchet_sltp -- the comparison isolates the ratchet backtest mechanism, not a different model.",
            "v1 (regression) and v2 (bucket, sparse labels) both failed to beat the ATR baseline (v1: collapsed to floor; v2 non-dense: real differentiation but OOS PnL/MDD worse than baseline) -- this run tests two changes at once (dense labels + dynamic ratchet), so if it wins, a follow-up ablation (dense labels + STATIC use, no ratchet) is needed to attribute the win correctly.",
        ],
        "quality_threshold": float(args.quality_threshold),
        "exit_threshold": float(args.exit_threshold),
        "exit_label_diag": exit_diag,
        "sltp_label_diag": label_diag,
        "sltp_targets": {
            "tp_capture_frac": float(args.tp_capture_frac), "sl_capture_frac": float(args.sl_capture_frac), "sltp_loss_weight": float(args.sltp_loss_weight),
            "label_horizon_bars": int(args.label_horizon_bars), "n_levels": N_LEVELS, "tp_levels": TP_LEVELS.tolist(), "sl_levels": SL_LEVELS.tolist(),
        },
        "summaries": summaries,
        "results": {
            "baseline_atr_fixed_formula": {"validation": val_m_atr, "oos": oos_m_atr, "validation_atr_diag": val_atr_diag, "oos_atr_diag": oos_atr_diag},
            "learned_sltp_static_entry_only": {"validation": val_m_static, "oos": oos_m_static, "validation_sltp_diag": val_static_diag, "oos_sltp_diag": oos_static_diag},
            "learned_sltp_dynamic_ratchet": {"validation": val_m_dynamic, "oos": oos_m_dynamic},
        },
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=omega._json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": cfg.__dict__}, out_dir / "eth_4head_dense_tabm_bundle.pt")
    print(json.dumps({
        "report": str(out_dir / "report.json"),
        "label_diag": label_diag,
        "baseline_atr": {"validation": val_m_atr, "oos": oos_m_atr},
        "learned_sltp_static": {"validation": val_m_static, "oos": oos_m_static},
        "learned_sltp_dynamic": {"validation": val_m_dynamic, "oos": oos_m_dynamic},
    }, ensure_ascii=False, indent=2, default=omega._json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
