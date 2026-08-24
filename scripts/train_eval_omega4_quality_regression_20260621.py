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


MODEL_ID = "omega4_quality_regression_20260621"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class ThreeHeadQualityRegTabM(nn.Module):
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
        self.quality_head = nn.Linear(int(cfg.hidden), 1)
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
            "quality": self.quality_head(h).squeeze(-1),
            "exit": self.exit_head(h),
        }


def _barrier_quality_targets(frame: pd.DataFrame, *, fee: float, slip: float, cost_mult: float, mae_lambda: float, clip: float, mode: str) -> tuple[np.ndarray, dict[str, Any]]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    action = pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    notional = float(omega.BASE_TEMPLATE["notional"])
    take_profit = float(omega.BASE_TEMPLATE["take_profit"])
    stop_loss = float(omega.BASE_TEMPLATE["stop_loss"])
    target = np.zeros(len(frame), dtype=np.float32)
    raw_values: list[float] = []
    active = 0
    filled_count = 0
    for i in range(0, len(frame) - 2):
        a = int(action[i])
        if a not in (1, 2):
            continue
        active += 1
        side = 1 if a == 1 else -1
        filled, entry_price, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            raw = -float(clip)
            target[i] = 0.0 if str(mode) == "binary_meta" else -1.0
            raw_values.append(raw)
            continue
        filled_count += 1
        entry_i = min(int(i) + 1, len(frame) - 1)
        cash_after_entry_fee = 1.0 - 1.0 * entry_fee * notional
        net, final_i, _reason = exit_head._continue_to_barrier_net(
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
        mae = 0.0
        for j in range(entry_i, min(int(final_i), len(frame) - 1) + 1):
            px = float(arrays["close"][j])
            ret = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if side > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            mae = min(mae, ret * notional)
        raw_net = float(net)
        raw = raw_net - float(mae_lambda) * abs(float(mae))
        if str(mode) == "binary_meta":
            target[i] = 1.0 if raw_net > 0.0 else 0.0
        else:
            raw = float(np.clip(raw, -float(clip), float(clip)))
            target[i] = raw / float(clip)
        raw_values.append(raw)
    arr = np.asarray(raw_values, dtype=np.float64) if raw_values else np.asarray([0.0], dtype=np.float64)
    return target, {
        "active_rows": int(active),
        "filled_entries": int(filled_count),
        "target_clip": float(clip),
        "mae_lambda": float(mae_lambda),
        "raw_mean": float(arr.mean()),
        "raw_p10": float(np.quantile(arr, 0.10)),
        "raw_p50": float(np.quantile(arr, 0.50)),
        "raw_p90": float(np.quantile(arr, 0.90)),
        "scaled_mean": float(target.mean()),
        "scaled_p70": float(np.quantile(target, 0.70)),
        "target_mode": str(mode),
        "positive_rate": float(target.mean()) if str(mode) == "binary_meta" else None,
    }


def _fit_expert(
    x_dir: pd.DataFrame,
    y_dir: np.ndarray,
    y_quality: np.ndarray,
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
    quality_target_mode: str,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _x_np, scaler = parent._standardize_fit(x_all)
    x_dir_np = parent._standardize_apply(x_dir, scaler)
    x_exit_np = parent._standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_quality_np = np.asarray(y_quality, dtype=np.float32)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = parent._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    if str(quality_target_mode) == "binary_meta":
        qual_w = compute_sample_weight(class_weight="balanced", y=y_quality_np.astype(np.int64)).astype(np.float32) * route_w
    else:
        qual_w = (1.0 + (np.abs(y_quality_np) > 1.0e-6).astype(np.float32) * 2.0) * route_w
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

    model = ThreeHeadQualityRegTabM(x_dir_np.shape[1], cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_quality_np[train_idx]),
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
            pred_q = out["quality"].mean(dim=1)
            if str(quality_target_mode) == "binary_meta":
                loss_qual_k = torch.nn.functional.binary_cross_entropy_with_logits(pred_q, yq, reduction="none")
            else:
                loss_qual_k = torch.nn.functional.smooth_l1_loss(pred_q, yq, reduction="none")
            loss_exit_k = torch.nn.functional.cross_entropy(out_exit["exit"].reshape(-1, 2), ye[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_qual = (loss_qual_k * qwb).sum() / torch.clamp(qwb.sum(), min=1.0)
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
            vq = torch.from_numpy(y_quality_np[val_idx]).to(device)
            vw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            if str(quality_target_mode) == "binary_meta":
                vqual = torch.nn.functional.binary_cross_entropy_with_logits(vo["quality"].mean(dim=1), vq, reduction="none")
            else:
                vqual = torch.nn.functional.smooth_l1_loss(vo["quality"].mean(dim=1), vq, reduction="none")
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(parent.CFG.k)).reshape(-1), reduction="none").reshape(-1, int(parent.CFG.k))
            vloss = float((((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0)) + float(parent.CFG.quality_loss_weight) * ((vqual * vqw).sum() / torch.clamp(vqw.sum(), min=1.0)) + float(parent.CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))).detach().cpu())
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
        "quality_target": str(quality_target_mode),
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = ThreeHeadQualityRegTabM(int(payload["n_features"]), cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    outs: dict[str, list[np.ndarray]] = {"direction": [], "quality": [], "exit": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        outs["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        q = out["quality"].mean(dim=1)
        if str(payload.get("quality_target", "")) == "binary_meta":
            q = torch.sigmoid(q)
        outs["quality"].append(q.detach().cpu().numpy()[:, None])
        outs["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float64) for k, v in outs.items()}


def _routed(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray, head: str, n_classes: int) -> np.ndarray:
    out = np.zeros((len(route), n_classes), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][head][mask]
    return out


def _prediction_output_reg(frame: pd.DataFrame, direction: np.ndarray, quality_score: np.ndarray, *, threshold: float, prefix: str) -> pd.DataFrame:
    dir_action = np.argmax(direction, axis=1).astype(np.int64)
    final_action = np.where((dir_action != 0) & (quality_score >= float(threshold)), dir_action, 0).astype(np.int64)
    out = pd.DataFrame({"timestamp": frame["timestamp"].to_numpy()})
    experts = hard.EXPERT_NAMES
    route = hard._route_id(frame)
    out[f"{prefix}_router_expert"] = np.asarray([experts[int(i)].replace("chop_expert", "chop") for i in route], dtype=object)
    out[f"{prefix}_router_confidence"] = np.max(frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64), axis=1)
    out[f"{prefix}_router_margin"] = np.sort(frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64), axis=1)[:, -1] - np.sort(frame[hard.ROUTE_COLS].to_numpy(dtype=np.float64), axis=1)[:, -2]
    out[f"{prefix}_dir_p_cash"] = direction[:, 0]
    out[f"{prefix}_dir_p_long"] = direction[:, 1]
    out[f"{prefix}_dir_p_short"] = direction[:, 2]
    out[f"{prefix}_dir_confidence"] = np.max(direction, axis=1)
    out[f"{prefix}_dir_side_edge"] = direction[:, 1] - direction[:, 2]
    out[f"{prefix}_dir_trade_prob"] = 1.0 - direction[:, 0]
    out[f"{prefix}_dir_action"] = dir_action
    out[f"{prefix}_quality_p_cash"] = 0.0
    out[f"{prefix}_quality_p_long"] = np.where(dir_action == 1, quality_score, 0.0)
    out[f"{prefix}_quality_p_short"] = np.where(dir_action == 2, quality_score, 0.0)
    out[f"{prefix}_quality_for_action"] = quality_score
    out[f"{prefix}_quality_threshold"] = float(threshold)
    out[f"{prefix}_final_action"] = final_action
    return out


def _apply_compensated(dec: pd.DataFrame, *, scale: float, cap: float) -> pd.DataFrame:
    return omega4._apply_compensated(dec, scale=scale, cap=cap)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--max-exit-samples", type=int, default=15000)
    ap.add_argument("--quality-quantile", type=float, default=0.70)
    ap.add_argument("--quality-target-mode", choices=["regression", "binary_meta"], default="regression")
    ap.add_argument("--quality-mae-lambda", type=float, default=0.5)
    ap.add_argument("--quality-clip", type=float, default=0.03)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260621)
    ap.add_argument("--out-suffix", default="smoke_e2_train15k_exit15k_q70")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cpu")
    args = ap.parse_args()
    _seed_everything(int(args.seed))
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0
    device = parent._device(str(args.device))
    out_dir = OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0010,
        quality_max_mae=0.0100,
        quality_min_mfe_mae=1.20,
        quality_max_hold_bars=288,
    )
    fee, slip = omega._load_fee_slip()
    train_raw = frames["train_raw"]
    val_raw = frames["val_raw"]
    oos_raw = frames["oos_raw"]
    base_cols = list(frames["feature_cols"])
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
    else:
        train_fit_frame = train_raw
    q_target_all, q_diag = _barrier_quality_targets(train_fit_frame, fee=fee, slip=slip, cost_mult=float(args.cost_mult), mae_lambda=float(args.quality_mae_lambda), clip=float(args.quality_clip), mode=str(args.quality_target_mode))
    x_train = parent._base_input(train_fit_frame, base_cols)
    y_train = train_fit_frame["zigzag_action"].to_numpy(dtype=np.int64)
    y_quality = q_target_all.astype(np.float32)

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
        payload = _fit_expert(
            x_train,
            y_train,
            y_quality,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=out_dir / "models" / f"{expert}_quality_reg_tabm.pt",
            quality_target_mode=str(args.quality_target_mode),
        )
        models[expert] = payload
        summaries[expert] = {"model": str(out_dir / "models" / f"{expert}_quality_reg_tabm.pt"), "epochs_ran": int(payload["epochs_ran"]), "best_validation_loss": float(payload["best_validation_loss"])}

    def predict(frame: pd.DataFrame, *, oof: bool, threshold: float) -> tuple[pd.DataFrame, np.ndarray]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = _routed(preds, route, "direction", 3)
        quality = _routed(preds, route, "quality", 1)[:, 0]
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        return _prediction_output_reg(frame, direction, quality, threshold=float(threshold), prefix=prefix), quality

    if str(args.quality_target_mode) == "binary_meta":
        cutoff = float(args.quality_quantile)
        active_pred = np.asarray([], dtype=np.float64)
    else:
        train_src, train_quality_pred = predict(train_fit_frame, oof=True, threshold=-999.0)
        del train_src
        active_pred = train_quality_pred[y_train != 0]
        cutoff = float(np.quantile(active_pred, float(args.quality_quantile))) if len(active_pred) else float(np.quantile(train_quality_pred, float(args.quality_quantile)))
    val_src, val_q = predict(val_raw, oof=True, threshold=cutoff)
    oos_src, oos_q = predict(oos_raw, oof=False, threshold=cutoff)
    val_dec = parent._to_decisions(val_src, oof=True)
    oos_dec = parent._to_decisions(oos_src, oof=False)
    val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    val_aggr = omega._metrics(val_raw, _apply_compensated(val_dec, scale=2.0, cap=0.90), fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    oos_aggr = omega._metrics(oos_raw, _apply_compensated(oos_dec, scale=2.0, cap=0.90), fee=fee, slip=slip, cost_mult=float(args.cost_mult))
    val_src.to_csv(out_dir / "validation_predictions_2025_quality_reg_q70.csv", index=False)
    oos_src.to_csv(out_dir / "oos_predictions_2026_quality_reg_q70.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "baseline_model": "omega4_3head_parent72_loose_entry_quality_20260620",
        "design": "Direction classification and Exit classification are held; Quality head is changed to AFML-style barrier replay payoff regression.",
        "label_contract": {
            "direction_label_dir": str(args.direction_label_dir),
            "direction_target": "zigzag_action",
            "quality_target": "binary_meta_net_return_positive" if str(args.quality_target_mode) == "binary_meta" else "clip(barrier_replay_net_return - mae_lambda * abs(MAE), +/- quality_clip) / quality_clip",
            "exit_target": "entry_label_terminal_giveback",
        },
        "quality_target_diag": q_diag,
        "quality_cutoff": {"quantile": float(args.quality_quantile), "cutoff": cutoff, "train_active_pred_count": int(len(active_pred)), "mode": str(args.quality_target_mode)},
        "exit_label": exit_diag,
        "results": {"validation": val_m, "oos": oos_m, "validation_aggressive_scale200_cap090": val_aggr, "oos_aggressive_scale200_cap090": oos_aggr},
        "prediction_quality_summary": {
            "validation_mean": float(np.mean(val_q)),
            "validation_p70": float(np.quantile(val_q, 0.70)),
            "oos_mean": float(np.mean(oos_q)),
            "oos_p70": float(np.quantile(oos_q, 0.70)),
        },
        "summaries": summaries,
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "validation_predictions": str(out_dir / "validation_predictions_2025_quality_reg_q70.csv"), "oos_predictions": str(out_dir / "oos_predictions_2026_quality_reg_q70.csv")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": parent.POS_COLS, "config": parent.CFG.__dict__, "model_class": "ThreeHeadQualityRegTabM", "quality_cutoff": cutoff}, out_dir / "quality_reg_3head_tabm_bundle.pt")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"], "quality_cutoff": report["quality_cutoff"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
