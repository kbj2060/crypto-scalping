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

import eval_omega4_1_atr_safety_sltp_20260622 as atr_eval  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as parent  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega4_2_risk_sidecar_20260622 as risk_sidecar  # noqa: E402
import train_eval_omega4_3head_parent72_loose_entry_quality_20260620 as omega4  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


MODEL_ID = "omega4_4_multihorizon_parent_20260624"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_BUNDLE = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_3head_parent72_loose_entry_quality_20260620_topdown_best_parent_e2_train15k_exit15k_q070_20260623"
    / "true_3head_tabm_bundle.pt"
)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


class MultiHorizonTabM(nn.Module):
    def __init__(self, n_features: int, horizons: list[int], *, cfg: parent.ThreeHeadConfig = parent.CFG) -> None:
        super().__init__()
        self.k = int(cfg.k)
        self.n_features = int(n_features)
        self.horizons = [int(h) for h in horizons]
        self.input_scale = nn.Parameter(torch.randn(self.k, self.n_features) * 0.03 + 1.0)
        self.input_bias = nn.Parameter(torch.zeros(self.k, self.n_features))
        self.in_proj = nn.Linear(self.n_features, int(cfg.hidden))
        self.blocks = nn.ModuleList(nn.Linear(int(cfg.hidden), int(cfg.hidden)) for _ in range(max(0, int(cfg.layers) - 1)))
        self.expert_scale = nn.ParameterList(
            nn.Parameter(torch.randn(self.k, int(cfg.hidden)) * 0.03 + 1.0) for _ in range(max(0, int(cfg.layers) - 1))
        )
        self.norms = nn.ModuleList(nn.LayerNorm(int(cfg.hidden)) for _ in range(max(0, int(cfg.layers))))
        self.dropout = nn.Dropout(float(cfg.dropout))
        self.direction_heads = nn.ModuleDict({str(h): nn.Linear(int(cfg.hidden), 3) for h in self.horizons})
        self.quality_heads = nn.ModuleDict({str(h): nn.Linear(int(cfg.hidden), 3) for h in self.horizons})

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

    def forward(self, x: torch.Tensor) -> dict[str, dict[str, torch.Tensor]]:
        h = self.encode(x)
        return {
            "direction": {str(k): head(h) for k, head in self.direction_heads.items()},
            "quality": {str(k): head(h) for k, head in self.quality_heads.items()},
        }


def _parse_ints(raw: str) -> list[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise RuntimeError("empty horizon list")
    if sorted(set(out)) != out:
        raise RuntimeError(f"horizons must be unique and ascending: {out}")
    return out


def _horizon_label_one(
    frame: pd.DataFrame,
    horizon: int,
    *,
    label_mode: str,
    smooth_window: int,
    atr_threshold_mult: float,
    zigzag_match_weight: float,
    zigzag_conflict_weight: float,
    zigzag_cash_weight: float,
    fee: float,
    slip: float,
    cost_mult: float,
    edge_min: float,
    edge_sqrt_scale: float,
    mae_penalty: float,
    mfe_bonus: float,
    quality_edge_mult: float,
    quality_mae_base: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if str(label_mode) not in {"future_path", "zigzag_gated_path", "tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"}:
        raise RuntimeError(f"unknown multi-horizon label mode: {label_mode}")
    if str(label_mode) in {"zigzag_gated_path", "tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"} and "zigzag_action" not in frame.columns:
        raise RuntimeError(f"{label_mode} requires zigzag_action column")
    is_tlob_smoothed = str(label_mode) in {"tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"}
    is_zigzag_gated = str(label_mode) in {"zigzag_gated_path", "tlob_smoothed_zigzag_gated"}
    high = pd.to_numeric(frame["high"], errors="raise").to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="raise").to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="raise").to_numpy(dtype=np.float64)
    base_action = (
        pd.to_numeric(frame["zigzag_action"], errors="raise").to_numpy(dtype=np.int64)
        if str(label_mode) in {"zigzag_gated_path", "tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"}
        else np.zeros(len(frame), dtype=np.int64)
    )
    n = len(frame)
    h = int(horizon)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    round_trip_cost = 2.0 * (fee_eff + slip_eff)
    min_edge = float(edge_min) + float(edge_sqrt_scale) * float(np.sqrt(max(h, 1) / 12.0))
    quality_edge = min_edge * float(quality_edge_mult)
    quality_mae_max = float(quality_mae_base) * float(np.sqrt(max(h, 1) / 12.0))
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([high - low, np.abs(high - prev_close), np.abs(low - prev_close)])
    atr_pct = pd.Series(tr).rolling(window=max(int(smooth_window), 1), min_periods=1).mean().to_numpy(dtype=np.float64) / np.maximum(close, 1.0e-12)
    action = np.zeros(n, dtype=np.int64)
    quality = np.zeros(n, dtype=np.int64)
    sample_weight = np.ones(n, dtype=np.float64)
    score_long = np.zeros(n, dtype=np.float64)
    score_short = np.zeros(n, dtype=np.float64)
    best_score = np.zeros(n, dtype=np.float64)
    path_mfe = np.zeros(n, dtype=np.float64)
    path_mae = np.zeros(n, dtype=np.float64)
    final_move = np.zeros(n, dtype=np.float64)
    tie_margin = max(min_edge * 0.15, 1.0e-6)
    last = n - 1
    k = max(int(smooth_window), 0)
    for i in range(n):
        end = min(i + h, last)
        if end <= i:
            continue
        if is_tlob_smoothed:
            cur_start = max(0, i - k)
            cur_end = i + 1
            fut_start = max(i + 1, end - k)
            fut_end = min(last + 1, end + k + 1)
            base = max(float(np.median(close[cur_start:cur_end])), 1.0e-12)
            final = float(np.median(close[fut_start:fut_end]))
            dynamic_edge = min_edge + float(atr_threshold_mult) * float(atr_pct[i]) * float(np.sqrt(max(h, 1) / 12.0))
        else:
            base = max(float(close[i]), 1.0e-12)
            final = float(close[end])
            dynamic_edge = min_edge
        hi = float(np.max(high[i + 1 : end + 1]))
        lo = float(np.min(low[i + 1 : end + 1]))
        long_final = final / base - 1.0
        long_mfe = hi / base - 1.0
        long_mae = lo / base - 1.0
        short_final = base / max(final, 1.0e-12) - 1.0
        short_mfe = base / max(lo, 1.0e-12) - 1.0
        short_mae = base / max(hi, 1.0e-12) - 1.0
        lscore = long_final + float(mfe_bonus) * max(long_mfe, 0.0) - float(mae_penalty) * abs(min(long_mae, 0.0)) - round_trip_cost
        sscore = short_final + float(mfe_bonus) * max(short_mfe, 0.0) - float(mae_penalty) * abs(min(short_mae, 0.0)) - round_trip_cost
        score_long[i] = lscore
        score_short[i] = sscore
        allow_long = (not is_zigzag_gated) or int(base_action[i]) == 1
        allow_short = (not is_zigzag_gated) or int(base_action[i]) == 2
        if allow_long and lscore >= dynamic_edge and lscore > sscore + tie_margin:
            action[i] = 1
            best_score[i] = lscore
            path_mfe[i] = max(long_mfe, 0.0)
            path_mae[i] = min(long_mae, 0.0)
            final_move[i] = long_final
            if lscore >= max(quality_edge, dynamic_edge * float(quality_edge_mult)) and abs(min(long_mae, 0.0)) <= quality_mae_max:
                quality[i] = 1
        elif allow_short and sscore >= dynamic_edge and sscore > lscore + tie_margin:
            action[i] = 2
            best_score[i] = sscore
            path_mfe[i] = max(short_mfe, 0.0)
            path_mae[i] = min(short_mae, 0.0)
            final_move[i] = short_final
            if sscore >= max(quality_edge, dynamic_edge * float(quality_edge_mult)) and abs(min(short_mae, 0.0)) <= quality_mae_max:
                quality[i] = 2
        if str(label_mode) == "tlob_smoothed_zigzag_weighted":
            if int(action[i]) == 0:
                sample_weight[i] = 1.0 if int(base_action[i]) == 0 else float(zigzag_cash_weight)
            elif int(action[i]) == int(base_action[i]):
                sample_weight[i] = float(zigzag_match_weight)
            elif int(base_action[i]) == 0:
                sample_weight[i] = float(zigzag_cash_weight)
            else:
                sample_weight[i] = float(zigzag_conflict_weight)
    out = pd.DataFrame(
        {
            "timestamp": frame["timestamp"].to_numpy(),
            f"mh{h}_action": action,
            f"mh{h}_quality_action": quality,
            f"mh{h}_sample_weight": sample_weight,
            f"mh{h}_score_long": score_long,
            f"mh{h}_score_short": score_short,
            f"mh{h}_best_score": best_score,
            f"mh{h}_final_move": final_move,
            f"mh{h}_path_mfe": path_mfe,
            f"mh{h}_path_mae": path_mae,
        }
    )
    diag = {
        "horizon": h,
        "label_mode": str(label_mode),
        "smooth_window": int(k),
        "atr_threshold_mult": float(atr_threshold_mult),
        "min_edge": float(min_edge),
        "quality_edge": float(quality_edge),
        "quality_mae_max": float(quality_mae_max),
        "round_trip_cost": float(round_trip_cost),
        "base_zigzag_counts": {str(k): int(v) for k, v in pd.Series(base_action).value_counts().sort_index().items()} if str(label_mode) in {"zigzag_gated_path", "tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"} else {},
        "direction_counts": {str(k): int(v) for k, v in pd.Series(action).value_counts().sort_index().items()},
        "quality_counts": {str(k): int(v) for k, v in pd.Series(quality).value_counts().sort_index().items()},
        "sample_weight_mean": float(np.mean(sample_weight)) if len(sample_weight) else 0.0,
        "direction_active_ratio": float(np.mean(action != 0)) if len(action) else 0.0,
        "quality_active_ratio": float(np.mean(quality != 0)) if len(quality) else 0.0,
        "best_score_p50": float(np.quantile(best_score[action != 0], 0.50)) if bool((action != 0).any()) else 0.0,
        "best_score_p90": float(np.quantile(best_score[action != 0], 0.90)) if bool((action != 0).any()) else 0.0,
    }
    return out, diag


def _attach_horizon_labels(
    frame: pd.DataFrame,
    horizons: list[int],
    *,
    label_mode: str,
    smooth_window: int,
    atr_threshold_mult: float,
    zigzag_match_weight: float,
    zigzag_conflict_weight: float,
    zigzag_cash_weight: float,
    fee: float,
    slip: float,
    cost_mult: float,
    edge_min: float,
    edge_sqrt_scale: float,
    mae_penalty: float,
    mfe_bonus: float,
    quality_edge_mult: float,
    quality_mae_base: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    out = frame.copy().reset_index(drop=True)
    diags: dict[str, Any] = {}
    for h in horizons:
        labels, diag = _horizon_label_one(
            out,
            int(h),
            label_mode=str(label_mode),
            smooth_window=int(smooth_window),
            atr_threshold_mult=float(atr_threshold_mult),
            zigzag_match_weight=float(zigzag_match_weight),
            zigzag_conflict_weight=float(zigzag_conflict_weight),
            zigzag_cash_weight=float(zigzag_cash_weight),
            fee=fee,
            slip=slip,
            cost_mult=cost_mult,
            edge_min=edge_min,
            edge_sqrt_scale=edge_sqrt_scale,
            mae_penalty=mae_penalty,
            mfe_bonus=mfe_bonus,
            quality_edge_mult=quality_edge_mult,
            quality_mae_base=quality_mae_base,
        )
        for col in labels.columns:
            if col != "timestamp":
                out[col] = labels[col].to_numpy()
        diags[f"h{int(h)}"] = diag
    return out, diags


def _write_label_frame(path: Path, frame: pd.DataFrame, horizons: list[int]) -> None:
    cols = ["timestamp", "zigzag_action"]
    for h in horizons:
        cols.extend(
            [
                f"mh{int(h)}_action",
                f"mh{int(h)}_quality_action",
                f"mh{int(h)}_sample_weight",
                f"mh{int(h)}_score_long",
                f"mh{int(h)}_score_short",
                f"mh{int(h)}_best_score",
                f"mh{int(h)}_final_move",
                f"mh{int(h)}_path_mfe",
                f"mh{int(h)}_path_mae",
            ]
        )
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise RuntimeError(f"multi-horizon label frame missing columns: {missing[:20]}")
    frame.loc[:, cols].to_csv(path, index=False)


def _fit_expert_multihorizon(
    x_train: pd.DataFrame,
    y_dir: dict[int, np.ndarray],
    y_quality: dict[int, np.ndarray],
    label_weight: dict[int, np.ndarray],
    route_frame: pd.DataFrame,
    *,
    horizons: list[int],
    expert_idx: int,
    seed: int,
    epochs: int,
    device: torch.device,
    model_path: Path,
) -> dict[str, Any]:
    torch.manual_seed(int(seed) + int(expert_idx))
    np.random.seed(int(seed) + int(expert_idx))
    model_path.parent.mkdir(parents=True, exist_ok=True)
    x_np, scaler = parent._standardize_fit(x_train)
    route_w = parent._route_probs(route_frame)[:, int(expert_idx)].astype(np.float32)
    n = len(x_np)
    split = max(int(n * 0.85), min(n - 1, 512))
    train_idx = np.arange(split)
    val_idx = np.arange(split, n)
    y_dir_matrix = np.column_stack([np.asarray(y_dir[int(h)], dtype=np.int64) for h in horizons])
    y_qual_matrix = np.column_stack([np.asarray(y_quality[int(h)], dtype=np.int64) for h in horizons])
    dir_w = np.zeros_like(y_dir_matrix, dtype=np.float32)
    qual_w = np.zeros_like(y_qual_matrix, dtype=np.float32)
    label_summary: dict[str, Any] = {}
    for j, h in enumerate(horizons):
        yd = y_dir_matrix[:, j]
        yq = y_qual_matrix[:, j]
        lw = np.asarray(label_weight[int(h)], dtype=np.float32)
        if len(lw) != len(yd):
            raise RuntimeError(f"h{h} label weight length mismatch")
        classes_d = sorted(np.unique(yd).astype(int).tolist())
        classes_q = sorted(np.unique(yq).astype(int).tolist())
        if classes_d != [0, 1, 2]:
            raise RuntimeError(f"h{h} direction labels need all classes [0,1,2], got {classes_d}")
        if classes_q != [0, 1, 2]:
            raise RuntimeError(f"h{h} quality labels need all classes [0,1,2], got {classes_q}")
        dir_w[:, j] = compute_sample_weight(class_weight="balanced", y=yd).astype(np.float32) * route_w * lw
        qual_w[:, j] = compute_sample_weight(class_weight="balanced", y=yq).astype(np.float32) * route_w * lw
        label_summary[f"h{h}"] = {
            "direction_counts": {str(k): int(v) for k, v in pd.Series(yd).value_counts().sort_index().items()},
            "quality_counts": {str(k): int(v) for k, v in pd.Series(yq).value_counts().sort_index().items()},
            "sample_weight_mean": float(np.mean(lw)),
        }
    if float(dir_w.sum()) <= 0.0 or float(qual_w.sum()) <= 0.0:
        raise RuntimeError(f"{hard.EXPERT_NAMES[expert_idx]} invalid multi-horizon sample weights")

    model = MultiHorizonTabM(x_np.shape[1], horizons=horizons, cfg=parent.CFG).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(parent.CFG.lr), weight_decay=float(parent.CFG.weight_decay))
    ds = TensorDataset(
        torch.from_numpy(x_np[train_idx]),
        torch.from_numpy(y_dir_matrix[train_idx]),
        torch.from_numpy(y_qual_matrix[train_idx]),
        torch.from_numpy(dir_w[train_idx]),
        torch.from_numpy(qual_w[train_idx]),
    )
    dl = DataLoader(ds, batch_size=int(parent.CFG.batch_size), shuffle=True, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    last_epoch = 0
    for epoch in range(int(epochs)):
        last_epoch = epoch + 1
        model.train()
        for xb, ydb, yqb, dwb, qwb in dl:
            xb = xb.to(device, non_blocking=True)
            ydb = ydb.to(device, non_blocking=True)
            yqb = yqb.to(device, non_blocking=True)
            dwb = dwb.to(device, non_blocking=True)
            qwb = qwb.to(device, non_blocking=True)
            out = model(xb)
            loss = torch.zeros((), dtype=torch.float32, device=device)
            for j, h in enumerate(horizons):
                key = str(int(h))
                dloss_k = torch.nn.functional.cross_entropy(
                    out["direction"][key].reshape(-1, 3),
                    ydb[:, j][:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(parent.CFG.k))
                qloss_k = torch.nn.functional.cross_entropy(
                    out["quality"][key].reshape(-1, 3),
                    yqb[:, j][:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(parent.CFG.k))
                loss = loss + (dloss_k.mean(dim=1) * dwb[:, j]).sum() / torch.clamp(dwb[:, j].sum(), min=1.0)
                loss = loss + float(parent.CFG.quality_loss_weight) * (qloss_k.mean(dim=1) * qwb[:, j]).sum() / torch.clamp(qwb[:, j].sum(), min=1.0)
            loss = loss / float(len(horizons))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vx = torch.from_numpy(x_np[val_idx]).to(device)
            vyd = torch.from_numpy(y_dir_matrix[val_idx]).to(device)
            vyq = torch.from_numpy(y_qual_matrix[val_idx]).to(device)
            vdw = torch.from_numpy(dir_w[val_idx]).to(device)
            vqw = torch.from_numpy(qual_w[val_idx]).to(device)
            vo = model(vx)
            vloss = torch.zeros((), dtype=torch.float32, device=device)
            for j, h in enumerate(horizons):
                key = str(int(h))
                vd = torch.nn.functional.cross_entropy(
                    vo["direction"][key].reshape(-1, 3),
                    vyd[:, j][:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(parent.CFG.k))
                vq = torch.nn.functional.cross_entropy(
                    vo["quality"][key].reshape(-1, 3),
                    vyq[:, j][:, None].expand(-1, int(parent.CFG.k)).reshape(-1),
                    reduction="none",
                ).reshape(-1, int(parent.CFG.k))
                vloss = vloss + (vd.mean(dim=1) * vdw[:, j]).sum() / torch.clamp(vdw[:, j].sum(), min=1.0)
                vloss = vloss + float(parent.CFG.quality_loss_weight) * (vq.mean(dim=1) * vqw[:, j]).sum() / torch.clamp(vqw[:, j].sum(), min=1.0)
            val_loss = float((vloss / float(len(horizons))).detach().cpu())
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
    payload = {
        "model_id": MODEL_ID,
        "expert": hard.EXPERT_NAMES[int(expert_idx)],
        "config": parent.CFG.__dict__,
        "horizons": [int(h) for h in horizons],
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "scaler": scaler,
        "n_features": int(x_np.shape[1]),
        "best_validation_loss": float(best_loss),
        "epochs_ran": int(last_epoch),
        "input_columns": list(x_train.columns),
        "label_summary": label_summary,
    }
    torch.save(payload, model_path)
    return payload


@torch.no_grad()
def _predict_multihorizon_payload(payload: dict[str, Any], x: pd.DataFrame, *, device: torch.device) -> dict[str, dict[int, np.ndarray]]:
    horizons = [int(h) for h in payload["horizons"]]
    model = MultiHorizonTabM(int(payload["n_features"]), horizons=horizons, cfg=parent.CFG).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    x_np = parent._standardize_apply(x, payload["scaler"])
    chunks: dict[str, dict[int, list[np.ndarray]]] = {
        "direction": {int(h): [] for h in horizons},
        "quality": {int(h): [] for h in horizons},
    }
    for start in range(0, len(x_np), 8192):
        xb = torch.from_numpy(x_np[start : start + 8192]).to(device)
        out = model(xb)
        for h in horizons:
            key = str(int(h))
            chunks["direction"][int(h)].append(torch.softmax(out["direction"][key], dim=-1).mean(dim=1).detach().cpu().numpy())
            chunks["quality"][int(h)].append(torch.softmax(out["quality"][key], dim=-1).mean(dim=1).detach().cpu().numpy())
    return {
        head: {int(h): np.concatenate(parts, axis=0).astype(np.float64) for h, parts in by_h.items()}
        for head, by_h in chunks.items()
    }


def _routed_horizon(
    preds: dict[str, dict[str, dict[int, np.ndarray]]],
    route: np.ndarray,
    *,
    head: str,
    horizon: int,
) -> np.ndarray:
    out = np.zeros((len(route), 3), dtype=np.float64)
    for idx, expert in enumerate(hard.EXPERT_NAMES):
        mask = route == idx
        if bool(mask.any()):
            out[mask] = preds[expert][head][int(horizon)][mask]
    return out


def _quality_for_action(quality: np.ndarray, action: np.ndarray) -> np.ndarray:
    return quality[np.arange(len(action)), action.astype(np.int64)]


def _select_horizon_prediction(
    frame: pd.DataFrame,
    direction_by_h: dict[int, np.ndarray],
    quality_by_h: dict[int, np.ndarray],
    horizons: list[int],
    *,
    threshold: float,
    score_edge_weight: float,
    score_trade_weight: float,
    prefix: str,
) -> pd.DataFrame:
    n = len(frame)
    h_scores = np.full((n, len(horizons)), -np.inf, dtype=np.float64)
    h_actions = np.zeros((n, len(horizons)), dtype=np.int64)
    h_quality = np.zeros((n, len(horizons)), dtype=np.float64)
    for j, h in enumerate(horizons):
        d = direction_by_h[int(h)]
        q = quality_by_h[int(h)]
        action = np.argmax(d, axis=1).astype(np.int64)
        q_for = _quality_for_action(q, action)
        score = q_for + float(score_edge_weight) * np.abs(d[:, 1] - d[:, 2]) + float(score_trade_weight) * (d[:, 1] + d[:, 2])
        valid = (action != 0) & (q_for >= float(threshold))
        h_scores[valid, j] = score[valid]
        h_actions[:, j] = action
        h_quality[:, j] = q_for
    any_valid = np.isfinite(h_scores).any(axis=1)
    selected_idx = np.argmax(h_scores, axis=1)
    selected_horizon = np.zeros(n, dtype=np.int64)
    selected_horizon[any_valid] = np.asarray(horizons, dtype=np.int64)[selected_idx[any_valid]]
    selected_direction = np.zeros((n, 3), dtype=np.float64)
    selected_quality = np.zeros((n, 3), dtype=np.float64)
    selected_direction[:, 0] = 1.0
    selected_quality[:, 0] = 1.0
    selected_score = np.zeros(n, dtype=np.float64)
    for j, h in enumerate(horizons):
        mask = any_valid & (selected_idx == j)
        if bool(mask.any()):
            selected_direction[mask] = direction_by_h[int(h)][mask]
            selected_quality[mask] = quality_by_h[int(h)][mask]
            selected_score[mask] = h_scores[mask, j]
    src = parent._prediction_output(frame, selected_direction, selected_quality, threshold=float(threshold), prefix=prefix)
    src[f"{prefix}_mh_selected_horizon"] = selected_horizon
    src[f"{prefix}_mh_selected_score"] = selected_score
    for j, h in enumerate(horizons):
        src[f"{prefix}_mh_score_h{int(h)}"] = np.where(np.isfinite(h_scores[:, j]), h_scores[:, j], 0.0)
        src[f"{prefix}_mh_action_h{int(h)}"] = h_actions[:, j]
        src[f"{prefix}_mh_quality_for_action_h{int(h)}"] = h_quality[:, j]
    return src


def _rename_oos_prefix(src: pd.DataFrame) -> pd.DataFrame:
    return src.rename(columns={c: c.replace("omega1_regime3_expertdq_oof_", "omega1_regime3_expertdq_") for c in src.columns})


def _selected_horizon_summary(src: pd.DataFrame, prefix: str) -> dict[str, Any]:
    col = f"{prefix}_mh_selected_horizon"
    if col not in src.columns:
        raise RuntimeError(f"missing selected horizon column: {col}")
    h = pd.to_numeric(src[col], errors="raise").to_numpy(dtype=np.int64)
    active = h > 0
    return {
        "active_rows": int(active.sum()),
        "active_ratio": float(active.mean()) if len(active) else 0.0,
        "counts": {str(k): int(v) for k, v in pd.Series(h[active]).value_counts().sort_index().items()},
    }


def _safe_ledger_log_metrics(frame: pd.DataFrame, ledger: pd.DataFrame) -> dict[str, float]:
    if ledger is None or len(ledger) == 0:
        return {
            "log_growth_sum": 0.0,
            "tail_excess_sum": 0.0,
            "liquidation_excess_sum": 0.0,
            "log_risk_utility": 0.0,
        }
    metrics, _ = risk_sidecar._ledger_metrics_with_margins(frame, ledger, None)
    return metrics


def _metric_row(name: str, metrics: dict[str, Any], q: float) -> dict[str, Any]:
    return {
        f"{name}_pnl": float(metrics["pnl"]),
        f"{name}_mdd": float(metrics["mdd"]),
        f"{name}_trades": int(metrics["trades"]),
        f"{name}_wr": float(metrics["wr"]),
        "quality_threshold": float(q),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-bundle", type=Path, default=BASELINE_BUNDLE)
    ap.add_argument("--direction-label-dir", type=Path, default=omega4.LABEL_DIR)
    ap.add_argument("--horizons", default="12,24,48,96,192")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train-rows", type=int, default=15000)
    ap.add_argument("--quality-thresholds", default="0.60,0.65,0.70,0.75,0.80")
    ap.add_argument("--save-quality-threshold", type=float, default=0.70)
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--label-mode", choices=["future_path", "zigzag_gated_path", "tlob_smoothed_zigzag_weighted", "tlob_smoothed_zigzag_gated"], default="future_path")
    ap.add_argument("--smooth-window", type=int, default=6)
    ap.add_argument("--atr-threshold-mult", type=float, default=0.15)
    ap.add_argument("--zigzag-match-weight", type=float, default=1.50)
    ap.add_argument("--zigzag-conflict-weight", type=float, default=0.40)
    ap.add_argument("--zigzag-cash-weight", type=float, default=0.70)
    ap.add_argument("--edge-min", type=float, default=0.0015)
    ap.add_argument("--edge-sqrt-scale", type=float, default=0.0010)
    ap.add_argument("--mae-penalty", type=float, default=0.85)
    ap.add_argument("--mfe-bonus", type=float, default=0.25)
    ap.add_argument("--quality-edge-mult", type=float, default=1.15)
    ap.add_argument("--quality-mae-base", type=float, default=0.010)
    ap.add_argument("--score-edge-weight", type=float, default=0.25)
    ap.add_argument("--score-trade-weight", type=float, default=0.10)
    ap.add_argument("--exit-threshold", type=float, default=0.75)
    ap.add_argument("--atr-window", type=int, default=192)
    ap.add_argument("--tp-mult", type=float, default=12.0)
    ap.add_argument("--sl-mult", type=float, default=6.0)
    ap.add_argument("--min-tp", type=float, default=0.075)
    ap.add_argument("--min-sl", type=float, default=0.040)
    ap.add_argument("--max-tp", type=float, default=0.22)
    ap.add_argument("--max-sl", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=260624)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="cuda")
    ap.add_argument("--out-suffix", default="h12_24_48_96_192_e2_train15k_q070_exit075")
    ap.add_argument("--existing-bundle", type=Path, default=None)
    args = ap.parse_args()

    _seed_everything(int(args.seed))
    device = parent._device(str(args.device))
    horizons = _parse_ints(str(args.horizons))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    omega.BASE_TEMPLATE["max_hold"] = 0
    omega.BASE_TEMPLATE["cooldown"] = 0

    print("stage=prepare_frames", flush=True)
    frames = omega4._prepare_frames(
        disable_tp_sl=False,
        direction_label_dir=Path(args.direction_label_dir),
        quality_mode="same_as_direction",
        quality_label_dir=None,
        quality_min_edge=0.0,
        quality_max_mae=0.0,
        quality_min_mfe_mae=0.0,
        quality_max_hold_bars=0,
    )
    fee, slip = omega._load_fee_slip()
    label_diag: dict[str, Any] = {}
    labeled_frames: dict[str, pd.DataFrame] = {}
    for split in ("train_raw", "val_raw", "oos_raw"):
        labeled, diag = _attach_horizon_labels(
            frames[split],
            horizons,
            label_mode=str(args.label_mode),
            smooth_window=int(args.smooth_window),
            atr_threshold_mult=float(args.atr_threshold_mult),
            zigzag_match_weight=float(args.zigzag_match_weight),
            zigzag_conflict_weight=float(args.zigzag_conflict_weight),
            zigzag_cash_weight=float(args.zigzag_cash_weight),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            edge_min=float(args.edge_min),
            edge_sqrt_scale=float(args.edge_sqrt_scale),
            mae_penalty=float(args.mae_penalty),
            mfe_bonus=float(args.mfe_bonus),
            quality_edge_mult=float(args.quality_edge_mult),
            quality_mae_base=float(args.quality_mae_base),
        )
        labeled_frames[split] = labeled
        label_diag[split] = diag
        _write_label_frame(out_dir / f"{split}_multihorizon_labels.csv", labeled, horizons)

    base_cols = list(frames["feature_cols"])
    train_raw = labeled_frames["train_raw"]
    val_raw = labeled_frames["val_raw"]
    oos_raw = labeled_frames["oos_raw"]
    x_train = parent._base_input(train_raw, base_cols)
    if int(args.max_train_rows) > 0:
        limit = int(args.max_train_rows)
        x_fit = x_train.iloc[:limit].reset_index(drop=True)
        train_fit_frame = train_raw.iloc[:limit].reset_index(drop=True)
    else:
        x_fit = x_train
        train_fit_frame = train_raw
    y_dir = {int(h): train_fit_frame[f"mh{int(h)}_action"].to_numpy(dtype=np.int64) for h in horizons}
    y_quality = {int(h): train_fit_frame[f"mh{int(h)}_quality_action"].to_numpy(dtype=np.int64) for h in horizons}
    label_weight = {int(h): train_fit_frame[f"mh{int(h)}_sample_weight"].to_numpy(dtype=np.float32) for h in horizons}

    models: dict[str, dict[str, Any]] = {}
    summaries: dict[str, Any] = {}
    if args.existing_bundle is not None:
        print("stage=load_existing_multihorizon_parent", flush=True)
        loaded_bundle = torch.load(Path(args.existing_bundle), map_location=device, weights_only=False)
        bundle_horizons = [int(h) for h in loaded_bundle["horizons"]]
        if bundle_horizons != horizons:
            raise RuntimeError(f"existing bundle horizons {bundle_horizons} do not match requested horizons {horizons}")
        bundle_cols = list(loaded_bundle["base_cols"])
        if bundle_cols != base_cols:
            raise RuntimeError("existing bundle base feature columns do not match prepared frames")
        models = loaded_bundle["models"]
        summaries = {
            expert: {
                "model": "loaded_from_existing_bundle",
                "epochs_ran": int(models[expert].get("epochs_ran", 0)),
                "best_validation_loss": float(models[expert].get("best_validation_loss", float("nan"))),
            }
            for expert in hard.EXPERT_NAMES
        }
    else:
        print("stage=train_multihorizon_parent", flush=True)
        for idx, expert in enumerate(hard.EXPERT_NAMES):
            payload = _fit_expert_multihorizon(
                x_fit,
                y_dir,
                y_quality,
                label_weight,
                train_fit_frame,
                horizons=horizons,
                expert_idx=idx,
                seed=int(args.seed),
                epochs=int(args.epochs),
                device=device,
                model_path=out_dir / "models" / f"{expert}_multihorizon_tabm.pt",
            )
            models[expert] = payload
            summaries[expert] = {
                "model": str(out_dir / "models" / f"{expert}_multihorizon_tabm.pt"),
                "epochs_ran": int(payload["epochs_ran"]),
                "best_validation_loss": float(payload["best_validation_loss"]),
            }

    def predict_all(frame: pd.DataFrame, *, oof: bool, q: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        x = parent._base_input(frame, base_cols)
        preds = {expert: _predict_multihorizon_payload(models[expert], x, device=device) for expert in hard.EXPERT_NAMES}
        route = hard._route_id(frame)
        direction_by_h = {int(h): _routed_horizon(preds, route, head="direction", horizon=int(h)) for h in horizons}
        quality_by_h = {int(h): _routed_horizon(preds, route, head="quality", horizon=int(h)) for h in horizons}
        prefix = "omega1_regime3_expertdq_oof" if oof else "omega1_regime3_expertdq"
        src = _select_horizon_prediction(
            frame,
            direction_by_h,
            quality_by_h,
            horizons,
            threshold=float(q),
            score_edge_weight=float(args.score_edge_weight),
            score_trade_weight=float(args.score_trade_weight),
            prefix=prefix,
        )
        dec = parent._to_decisions(src, oof=oof)
        dec_atr, _diag = atr_eval._apply_atr_safety_sltp(
            dec,
            frame,
            atr_window=int(args.atr_window),
            tp_mult=float(args.tp_mult),
            sl_mult=float(args.sl_mult),
            min_tp=float(args.min_tp),
            min_sl=float(args.min_sl),
            max_tp=float(args.max_tp),
            max_sl=float(args.max_sl),
        )
        return x, src, dec_atr

    print("stage=load_exit_bundle", flush=True)
    exit_bundle = torch.load(Path(args.baseline_bundle), map_location=device, weights_only=False)
    exit_loaded = parent._load_payloads(exit_bundle["models"], device=device)
    exit_base_cols = list(exit_bundle["base_cols"])
    if exit_base_cols != base_cols:
        raise RuntimeError("multi-horizon base columns differ from baseline exit bundle columns")

    q_values = [float(x.strip()) for x in str(args.quality_thresholds).split(",") if x.strip()]
    rows: list[dict[str, Any]] = []
    results: dict[str, Any] = {}
    saved_predictions: dict[str, str] = {}
    for q in q_values:
        print(f"stage=evaluate_q{q:.2f}", flush=True)
        x_val, val_src, val_dec = predict_all(val_raw, oof=True, q=float(q))
        x_oos, oos_src, oos_dec = predict_all(oos_raw, oof=False, q=float(q))
        val_parent_m = omega._metrics(val_raw, val_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        oos_parent_m = omega._metrics(oos_raw, oos_dec, fee=fee, slip=slip, cost_mult=float(args.cost_mult))
        val_exit_m, val_exit_ledger = risk_sidecar._replay_with_risk(
            val_raw,
            x_val,
            val_dec,
            exit_loaded,
            risk_margin_fraction=None,
            risk_leverage=None,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            notional_scaled_sltp=False,
            device=device,
        )
        oos_exit_m, oos_exit_ledger = risk_sidecar._replay_with_risk(
            oos_raw,
            x_oos,
            oos_dec,
            exit_loaded,
            risk_margin_fraction=None,
            risk_leverage=None,
            exit_threshold=float(args.exit_threshold),
            fee=fee,
            slip=slip,
            cost_mult=float(args.cost_mult),
            notional_scaled_sltp=False,
            device=device,
        )
        val_log_m = _safe_ledger_log_metrics(val_raw, val_exit_ledger)
        oos_log_m = _safe_ledger_log_metrics(oos_raw, oos_exit_ledger)
        for key in ("log_growth_sum", "tail_excess_sum", "liquidation_excess_sum", "log_risk_utility"):
            val_exit_m[key] = val_log_m[key]
            oos_exit_m[key] = oos_log_m[key]
        key = f"q{q:.2f}".replace(".", "p")
        results[key] = {
            "validation_parent_only": val_parent_m,
            "oos_parent_only": oos_parent_m,
            "validation_exit_replay": val_exit_m,
            "oos_exit_replay": oos_exit_m,
            "validation_horizon_summary": _selected_horizon_summary(val_src, "omega1_regime3_expertdq_oof"),
            "oos_horizon_summary": _selected_horizon_summary(oos_src, "omega1_regime3_expertdq"),
        }
        row = {"variant": key, "quality_threshold": float(q)}
        row.update({f"parent_{k}": v for k, v in _metric_row("validation", val_parent_m, q).items() if k != "quality_threshold"})
        row.update({f"parent_{k}": v for k, v in _metric_row("oos", oos_parent_m, q).items() if k != "quality_threshold"})
        row.update({f"exit_{k}": v for k, v in _metric_row("validation", val_exit_m, q).items() if k != "quality_threshold"})
        row.update({f"exit_{k}": v for k, v in _metric_row("oos", oos_exit_m, q).items() if k != "quality_threshold"})
        rows.append(row)
        if abs(float(q) - float(args.save_quality_threshold)) < 1.0e-12:
            save_tag = f"q{int(round(float(q) * 100.0)):03d}"
            x_train_full, train_src, train_dec = predict_all(train_raw, oof=True, q=float(q))
            train_exit_m, train_exit_ledger = risk_sidecar._replay_with_risk(
                train_raw,
                x_train_full,
                train_dec,
                exit_loaded,
                risk_margin_fraction=None,
                risk_leverage=None,
                exit_threshold=float(args.exit_threshold),
                fee=fee,
                slip=slip,
                cost_mult=float(args.cost_mult),
                notional_scaled_sltp=False,
                device=device,
            )
            train_src.to_csv(out_dir / f"train_predictions_{save_tag}.csv", index=False)
            val_src.to_csv(out_dir / f"validation_predictions_{save_tag}.csv", index=False)
            oos_src.to_csv(out_dir / f"oos_predictions_{save_tag}.csv", index=False)
            train_dec.to_csv(out_dir / f"train_decisions_{save_tag}_atr.csv", index=False)
            val_dec.to_csv(out_dir / f"validation_decisions_{save_tag}_atr.csv", index=False)
            oos_dec.to_csv(out_dir / f"oos_decisions_{save_tag}_atr.csv", index=False)
            train_exit_ledger.to_csv(out_dir / "train_baseline_trade_ledger.csv", index=False)
            val_exit_ledger.to_csv(out_dir / "validation_baseline_trade_ledger.csv", index=False)
            oos_exit_ledger.to_csv(out_dir / "oos_baseline_trade_ledger.csv", index=False)
            saved_predictions = {
                f"train_{save_tag}": str(out_dir / f"train_predictions_{save_tag}.csv"),
                f"validation_{save_tag}": str(out_dir / f"validation_predictions_{save_tag}.csv"),
                f"oos_{save_tag}": str(out_dir / f"oos_predictions_{save_tag}.csv"),
                "train_baseline_ledger": str(out_dir / "train_baseline_trade_ledger.csv"),
                "validation_baseline_ledger": str(out_dir / "validation_baseline_trade_ledger.csv"),
                "oos_baseline_ledger": str(out_dir / "oos_baseline_trade_ledger.csv"),
            }
            results[key]["train_exit_replay"] = train_exit_m

    ranking = pd.DataFrame(rows).sort_values(["exit_validation_pnl", "exit_validation_mdd", "exit_oos_pnl"], ascending=[False, False, False])
    ranking.to_csv(out_dir / "quality_threshold_ranking.csv", index=False)
    torch.save({"models": models, "base_cols": base_cols, "horizons": horizons, "config": parent.CFG.__dict__}, out_dir / "multihorizon_parent_bundle.pt")
    report = {
        "model_id": MODEL_ID,
        "base_model": "omega4_4_topdown_reproducible_architecture_baseline_20260623",
        "baseline_exit_bundle": str(args.baseline_bundle),
        "design": "Multi-horizon Omega4.4 parent experiment. Direction and quality heads are split by horizon on a shared TabM encoder. Existing Omega4.4 exit head and ATR safety replay are kept unchanged.",
        "horizons_bars": horizons,
        "bar_minutes": 5,
        "label_contract": {
            "method": str(args.label_mode),
            "base_label_dir": str(args.direction_label_dir),
            "base_label_gate": "zigzag_action must match horizon path direction" if str(args.label_mode) in {"zigzag_gated_path", "tlob_smoothed_zigzag_gated"} else "soft sample-weight prior" if str(args.label_mode) == "tlob_smoothed_zigzag_weighted" else "none",
            "smooth_window": int(args.smooth_window),
            "atr_threshold_mult": float(args.atr_threshold_mult),
            "zigzag_sample_weights": {
                "match": float(args.zigzag_match_weight),
                "conflict": float(args.zigzag_conflict_weight),
                "cash": float(args.zigzag_cash_weight),
            },
            "edge_min": float(args.edge_min),
            "edge_sqrt_scale": float(args.edge_sqrt_scale),
            "mae_penalty": float(args.mae_penalty),
            "mfe_bonus": float(args.mfe_bonus),
            "quality_edge_mult": float(args.quality_edge_mult),
            "quality_mae_base": float(args.quality_mae_base),
            "uses_future_only_for_offline_labeling": True,
        },
        "runtime_contract": {
            "horizon_selection": "argmax quality_for_action + edge/trade-prob bonus among horizons passing quality threshold",
            "exit_head": "Omega4.4 baseline exit head reused",
            "exit_threshold": float(args.exit_threshold),
            "atr_safety": {
                "atr_window": int(args.atr_window),
                "tp_mult": float(args.tp_mult),
                "sl_mult": float(args.sl_mult),
                "min_tp": float(args.min_tp),
                "min_sl": float(args.min_sl),
                "max_tp": float(args.max_tp),
                "max_sl": float(args.max_sl),
            },
        },
        "input_contract": {
            "base_feature_count": len(base_cols),
            "position_feature_count": len(parent.POS_COLS),
            "total_features_for_entry": len(base_cols) + len(parent.POS_COLS),
        },
        "label_diag": label_diag,
        "summaries": summaries,
        "results": results,
        "ranking": rows,
        "artifacts": {
            "out_dir": str(out_dir),
            "bundle": str(out_dir / "multihorizon_parent_bundle.pt"),
            "ranking": str(out_dir / "quality_threshold_ranking.csv"),
            "report": str(out_dir / "report.json"),
            "train_labels": str(out_dir / "train_raw_multihorizon_labels.csv"),
            "validation_labels": str(out_dir / "val_raw_multihorizon_labels.csv"),
            "oos_labels": str(out_dir / "oos_raw_multihorizon_labels.csv"),
            **saved_predictions,
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "top": ranking.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
