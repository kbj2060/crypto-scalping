#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_sample_weight
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train_eval_omega1_2_tabm_3head_20260603 as base3
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega
import train_eval_omega1_2_tabm_exit_head_20260603 as exit_head
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_2_true_3head_tabm_quality_replay_regression_20260620"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

FEE_RATE = 0.0005
SLIP_RATE = 0.0002
BASE_NOTIONAL = 0.45
BASE_TP = 0.026
BASE_SL = 0.014
EXPERT_SCALE = {"bull": 0.75, "bear": 0.90, "chop": 0.90, "chop_expert": 0.90}
MAX_LABEL_HORIZON = 384
QUALITY_TARGET_SCALE = 100.0


@dataclass(frozen=True)
class Config:
    k: int = 8
    hidden: int = 192
    layers: int = 3
    dropout: float = 0.08
    batch_size: int = 2048
    lr: float = 2.0e-3
    weight_decay: float = 2.0e-4
    patience: int = 5
    quality_loss_weight: float = 0.80
    exit_loss_weight: float = 1.15


CFG = Config()


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


class ReplayQualityTabM(nn.Module):
    def __init__(self, n_features: int, *, cfg: Config = CFG) -> None:
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


def _notional(expert: str) -> float:
    return BASE_NOTIONAL * EXPERT_SCALE.get(str(expert).replace("chop_expert", "chop"), 0.90)


def _side(action: int) -> int:
    if int(action) == omega.ACTION_LONG:
        return 1
    if int(action) == omega.ACTION_SHORT:
        return -1
    return 0


def _replay_quality_target(frame: pd.DataFrame, action: np.ndarray, *, cost_mult: float) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().bfill().to_numpy(dtype=np.float64)
    route = np.asarray(hard.EXPERT_NAMES, dtype=object)[hard._route_id(frame)]
    fee = FEE_RATE * float(cost_mult)
    slip = SLIP_RATE * float(cost_mult)
    rows: list[dict[str, Any]] = []
    for i in range(len(frame)):
        side = _side(int(action[i]))
        notional = _notional(str(route[i]))
        if side == 0 or i + 2 >= len(frame):
            rows.append({"quality_target_net_return": 0.0, "quality_target_return_over_mae": 0.0, "quality_label_hold_bars": 0, "quality_label_reason": "cash"})
            continue
        entry_i = min(i + 1, len(frame) - 1)
        entry = float(close[entry_i]) * (1.0 + slip if side > 0 else 1.0 - slip)
        end_i = min(len(frame) - 1, entry_i + MAX_LABEL_HORIZON)
        exit_i = end_i
        exit_px = float(close[end_i])
        reason = "vertical"
        mae = 0.0
        for j in range(entry_i, end_i + 1):
            px = float(close[j])
            raw = (px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - px * (1.0 + slip)) / max(entry, 1e-12)
            pnl = raw * notional
            mae = min(mae, pnl)
            if pnl >= BASE_TP:
                exit_i = j
                exit_px = px
                reason = "take_profit"
                break
            if pnl <= -abs(BASE_SL):
                exit_i = j
                exit_px = px
                reason = "stop_loss"
                break
        raw_exit = (exit_px * (1.0 - slip) - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px * (1.0 + slip)) / max(entry, 1e-12)
        net = raw_exit * notional - 2.0 * fee * notional
        rows.append(
            {
                "quality_target_net_return": float(net),
                "quality_target_return_over_mae": float(net / max(abs(mae), 1e-6)),
                "quality_label_hold_bars": int(exit_i - entry_i),
                "quality_label_reason": reason,
            }
        )
    return pd.DataFrame(rows)


def _standardize_fit(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    return base3._standardize_fit(x)


def _standardize_apply(x: pd.DataFrame, scaler: dict[str, Any]) -> np.ndarray:
    return base3._standardize_apply(x, scaler)


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
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_all = pd.concat([x_dir, x_exit], ignore_index=True)
    _, scaler = _standardize_fit(x_all)
    x_dir_np = _standardize_apply(x_dir, scaler)
    x_exit_np = _standardize_apply(x_exit, scaler)
    y_dir_np = np.asarray(y_dir, dtype=np.int64)
    y_quality_np = (np.asarray(y_quality, dtype=np.float32) * float(QUALITY_TARGET_SCALE)).astype(np.float32)
    y_exit_np = np.asarray(y_exit, dtype=np.int64)
    route_w = base3._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    exit_w = base3._route_probs(exit_route_frame)[:, int(expert_idx)].astype(np.float32)
    dir_w = compute_sample_weight(class_weight="balanced", y=y_dir_np).astype(np.float32) * route_w
    quality_w = (0.35 + np.clip(np.abs(np.asarray(y_quality, dtype=np.float32)), 0.0, 0.03) * 25.0).astype(np.float32) * route_w
    ex_w = compute_sample_weight(class_weight="balanced", y=y_exit_np).astype(np.float32) * exit_w
    if float(dir_w.sum()) <= 0.0 or float(quality_w.sum()) <= 0.0 or float(ex_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid sample weights")

    n = len(y_dir_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    exit_n = len(y_exit_np)
    exit_split = max(int(exit_n * 0.85), min(exit_n - 1, 256))
    exit_train_idx = np.arange(exit_split)
    exit_val_idx = np.arange(exit_split, exit_n)

    model = ReplayQualityTabM(x_dir_np.shape[1], cfg=CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(CFG.lr), weight_decay=float(CFG.weight_decay))
    ds_dir = TensorDataset(
        torch.from_numpy(x_dir_np[train_idx]),
        torch.from_numpy(y_dir_np[train_idx]),
        torch.from_numpy(y_quality_np[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(quality_w[train_idx]),
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
        for xb, yb, yq, wb, wq in dl_dir:
            try:
                xe, ye, we = next(exit_iter)
            except StopIteration:
                exit_iter = iter(dl_exit)
                xe, ye, we = next(exit_iter)
            xb = xb.to(device)
            yb = yb.to(device)
            yq = yq.to(device)
            wb = wb.to(device)
            wq = wq.to(device)
            xe = xe.to(device)
            ye = ye.to(device)
            we = we.to(device)
            out_dir = model(xb)
            loss_dir_k = torch.nn.functional.cross_entropy(
                out_dir["direction"].reshape(-1, 3),
                yb[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_quality_k = torch.nn.functional.smooth_l1_loss(out_dir["quality"], yq[:, None].expand(-1, int(CFG.k)), reduction="none")
            out_exit = model(xe)
            loss_exit_k = torch.nn.functional.cross_entropy(
                out_exit["exit"].reshape(-1, 2),
                ye[:, None].expand(-1, int(CFG.k)).reshape(-1),
                reduction="none",
            ).reshape(-1, int(CFG.k))
            loss_dir = (loss_dir_k.mean(dim=1) * wb).sum() / torch.clamp(wb.sum(), min=1.0)
            loss_quality = (loss_quality_k.mean(dim=1) * wq).sum() / torch.clamp(wq.sum(), min=1.0)
            loss_exit = (loss_exit_k.mean(dim=1) * we).sum() / torch.clamp(we.sum(), min=1.0)
            loss = loss_dir + float(CFG.quality_loss_weight) * loss_quality + float(CFG.exit_loss_weight) * loss_exit
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
            vqw = torch.from_numpy(quality_w[val_idx]).to(device)
            ve = torch.from_numpy(x_exit_np[exit_val_idx]).to(device)
            vey = torch.from_numpy(y_exit_np[exit_val_idx]).to(device)
            vew = torch.from_numpy(ex_w[exit_val_idx]).to(device)
            vo = model(vx)
            veo = model(ve)
            vdir = torch.nn.functional.cross_entropy(vo["direction"].reshape(-1, 3), vy[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vqual = torch.nn.functional.smooth_l1_loss(vo["quality"], vq[:, None].expand(-1, int(CFG.k)), reduction="none")
            vex = torch.nn.functional.cross_entropy(veo["exit"].reshape(-1, 2), vey[:, None].expand(-1, int(CFG.k)).reshape(-1), reduction="none").reshape(-1, int(CFG.k))
            vloss = float(
                (
                    ((vdir.mean(dim=1) * vw).sum() / torch.clamp(vw.sum(), min=1.0))
                    + float(CFG.quality_loss_weight) * ((vqual.mean(dim=1) * vqw).sum() / torch.clamp(vqw.sum(), min=1.0))
                    + float(CFG.exit_loss_weight) * ((vex.mean(dim=1) * vew).sum() / torch.clamp(vew.sum(), min=1.0))
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
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, np.ndarray]:
    model = ReplayQualityTabM(int(payload["n_features"]), cfg=CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = _standardize_apply(x, payload["scaler"])
    chunks = {"direction": [], "quality": [], "exit": []}
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        chunks["direction"].append(torch.softmax(out["direction"], dim=-1).mean(dim=1).detach().cpu().numpy())
        chunks["quality"].append((out["quality"].mean(dim=1).detach().cpu().numpy() / float(QUALITY_TARGET_SCALE)))
        chunks["exit"].append(torch.softmax(out["exit"], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {
        "direction": np.concatenate(chunks["direction"], axis=0).astype(np.float64),
        "quality": np.concatenate(chunks["quality"], axis=0).astype(np.float64),
        "exit": np.concatenate(chunks["exit"], axis=0).astype(np.float64),
    }


def _routed_scalar(preds: dict[str, dict[str, np.ndarray]], route: np.ndarray) -> np.ndarray:
    out = np.zeros(len(route), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert]["quality"][mask]
    return out


def _prediction_output(frame: pd.DataFrame, direction: np.ndarray, quality: np.ndarray, *, threshold: float, prefix: str) -> pd.DataFrame:
    route = hard._route_id(frame)
    action = np.argmax(direction, axis=1).astype(np.int64)
    final_action = action.copy()
    final_action[(action != omega.ACTION_CASH) & (quality < float(threshold))] = omega.ACTION_CASH
    return pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"{prefix}_router_expert": np.asarray(hard.EXPERT_NAMES, dtype=object)[route],
            f"{prefix}_router_confidence": hard._route_conf(frame),
            f"{prefix}_router_margin": pd.to_numeric(frame["regime3_current_sensitive_wide24_margin"], errors="raise").to_numpy(dtype=np.float64),
            f"{prefix}_dir_p_cash": direction[:, 0],
            f"{prefix}_dir_p_long": direction[:, 1],
            f"{prefix}_dir_p_short": direction[:, 2],
            f"{prefix}_dir_confidence": np.max(direction, axis=1),
            f"{prefix}_dir_side_edge": direction[:, 1] - direction[:, 2],
            f"{prefix}_dir_trade_prob": direction[:, 1] + direction[:, 2],
            f"{prefix}_dir_action": action,
            f"{prefix}_quality_p_cash": quality,
            f"{prefix}_quality_p_long": quality,
            f"{prefix}_quality_p_short": quality,
            f"{prefix}_quality_for_action": quality,
            f"{prefix}_quality_threshold": float(threshold),
            f"{prefix}_final_action": final_action,
        }
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--quality-thresholds", default="0.0000,0.0010,0.0020,0.0030,0.0040,0.0050")
    ap.add_argument("--exit-edge-min", type=float, default=0.0020)
    ap.add_argument("--exit-hold-offsets", default="1,2,3,6,12,24,48,96,192,384")
    ap.add_argument("--max-exit-samples", type=int, default=30000)
    ap.add_argument("--max-train-rows", type=int, default=0)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260620)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()

    base3._seed_everything(int(args.seed))
    device = base3._device(str(args.device))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = base3._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    base_cols = list(frames["feature_cols"])
    train_raw = frames["train_raw"].reset_index(drop=True)
    val_raw = frames["val_raw"].reset_index(drop=True)
    oos_raw = frames["oos_raw"].reset_index(drop=True)
    x_train = base3._base_input(train_raw, base_cols)
    y_train = train_raw["zigzag_action"].to_numpy(dtype=np.int64)
    q_train_df = _replay_quality_target(train_raw, y_train, cost_mult=float(args.cost_mult))
    q_val_df = _replay_quality_target(val_raw, val_raw["zigzag_action"].to_numpy(dtype=np.int64), cost_mult=float(args.cost_mult))
    if int(args.max_train_rows) > 0:
        rows = int(args.max_train_rows)
        x_train = x_train.iloc[:rows].reset_index(drop=True)
        y_train = y_train[:rows]
        q_train = q_train_df["quality_target_net_return"].to_numpy(dtype=np.float32)[:rows]
        train_fit_frame = train_raw.iloc[:rows].reset_index(drop=True)
    else:
        q_train = q_train_df["quality_target_net_return"].to_numpy(dtype=np.float32)
        train_fit_frame = train_raw
    lo, hi = np.quantile(q_train, [0.01, 0.99])
    q_train = np.clip(q_train, lo, hi).astype(np.float32)

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
    x_exit = base3._exit_input_from_position_rows(x_exit_raw, base_cols)
    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        payload = _fit_expert(
            x_train,
            y_train,
            q_train,
            train_fit_frame,
            x_exit,
            y_exit,
            frame_exit,
            expert_idx=idx,
            seed=int(args.seed),
            epochs=int(args.epochs),
            device=device,
            model_path=OUT_DIR / "models" / f"{expert}_quality_replay_tabm.pt",
        )
        models[expert] = payload
        summaries[expert] = {
            "model": str(OUT_DIR / "models" / f"{expert}_quality_replay_tabm.pt"),
            "epochs_ran": int(payload["epochs_ran"]),
            "best_validation_loss": float(payload["best_validation_loss"]),
        }

    def predict_arrays(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        x = base3._base_input(frame, base_cols)
        preds = {expert: _predict_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction = base3._routed(preds, route, "direction", 3)
        quality = _routed_scalar(preds, route)
        return x, direction, quality

    def build_output(frame: pd.DataFrame, direction: np.ndarray, quality: np.ndarray, threshold: float, prefix: str) -> pd.DataFrame:
        out = _prediction_output(frame, direction, quality, threshold=float(threshold), prefix=prefix)
        return out

    thresholds = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    _, val_direction, val_quality = predict_arrays(val_raw)
    _, oos_direction, oos_quality = predict_arrays(oos_raw)
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    for threshold in thresholds:
        val_src = build_output(val_raw, val_direction, val_quality, threshold, "omega1_regime3_expertdq_oof")
        oos_src = build_output(oos_raw, oos_direction, oos_quality, threshold, "omega1_regime3_expertdq")
        val_dec = omega._to_fixed_decisions(val_src, oof=True)
        oos_dec = omega._to_fixed_decisions(oos_src, oof=False)
        val_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        name = f"quality_replay_thr_{threshold:.4f}".replace(".", "p").replace("-", "m")
        reports[name] = {"validation": val_m, "oos": oos_m}
        rows.append(
            {
                "variant": name,
                "threshold": threshold,
                "validation_pnl": val_m["pnl"],
                "validation_mdd": val_m["mdd"],
                "validation_trades": val_m["trades"],
                "validation_wr": val_m["wr"],
                "oos_pnl": oos_m["pnl"],
                "oos_mdd": oos_m["mdd"],
                "oos_trades": oos_m["trades"],
                "oos_wr": oos_m["wr"],
            }
        )
    rows.sort(key=lambda r: (float(r["validation_pnl"]), float(r["validation_mdd"])), reverse=True)
    best_threshold = float(rows[0]["threshold"])
    val_best = build_output(val_raw, val_direction, val_quality, best_threshold, "omega1_regime3_expertdq_oof")
    oos_best = build_output(oos_raw, oos_direction, oos_quality, best_threshold, "omega1_regime3_expertdq")
    val_best.to_csv(OUT_DIR / "validation_predictions_2025_quality_replay.csv", index=False)
    oos_best.to_csv(OUT_DIR / "oos_predictions_2026_quality_replay.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT_DIR / "ranking.csv", index=False)
    q_train_df.to_csv(OUT_DIR / "train_quality_replay_labels.csv", index=False)
    q_val_df.to_csv(OUT_DIR / "validation_quality_replay_label_audit.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Step-2 parent retrain. Direction and Exit heads remain classification heads; Quality head is replaced by barrier-replay cost-included net-return regression.",
        "quality_label_contract": {
            "side_source": "zigzag_action training label for this first parent-retrain probe",
            "target": "quality_target_net_return",
            "replay": "fixed TP/SL close-path barrier replay with fee/slippage cost included",
            "tp_account_pnl": BASE_TP,
            "sl_account_pnl": BASE_SL,
            "max_label_horizon_bars": MAX_LABEL_HORIZON,
            "clip_q01_q99": [float(lo), float(hi)],
            "training_target_scale": QUALITY_TARGET_SCALE,
        },
        "quality_label_audit": {
            "train_rows": int(len(q_train_df)),
            "train_positive_rate": float((q_train_df["quality_target_net_return"] > 0.0).mean()),
            "train_mean": float(q_train_df["quality_target_net_return"].mean()),
            "validation_positive_rate": float((q_val_df["quality_target_net_return"] > 0.0).mean()),
            "validation_mean": float(q_val_df["quality_target_net_return"].mean()),
        },
        "exit_label": {"exit_edge_min": float(args.exit_edge_min), "hold_offsets": hold_offsets, "diag": exit_diag},
        "summaries": summaries,
        "ranking_by_validation_pnl": rows,
        "results": reports,
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
            "validation_predictions": str(OUT_DIR / "validation_predictions_2025_quality_replay.csv"),
            "oos_predictions": str(OUT_DIR / "oos_predictions_2026_quality_replay.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    torch.save({"models": models, "base_cols": base_cols, "pos_cols": base3.POS_COLS, "config": CFG.__dict__}, OUT_DIR / "quality_replay_3head_tabm_bundle.pt")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": rows[:5], "quality_label_audit": report["quality_label_audit"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
