#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, prepare_features  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    EVAL_CSV,
    TRAIN_CSV,
    _active,
    _combine_primary_fallback,
    _empty_dec_like,
    _json_default,
    _load_best_scale_runtime,
    _predict_scaled,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha7_iqn_fallback_10x_cvar_20260527"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_iqn_fallback_10x_cvar_20260527"
DERIVED_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}


class IQNFallbackNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int = 3, hidden_dim: int = 256, n_cos: int = 64) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_cos = int(n_cos)
        self.state = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.quantile = nn.Sequential(
            nn.Linear(n_cos, hidden_dim),
            nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # x: [B, D], tau: [B, N]
        state = self.state(x).unsqueeze(1)
        basis_idx = torch.arange(1, self.n_cos + 1, device=x.device, dtype=x.dtype).view(1, 1, -1)
        tau_basis = torch.cos(math.pi * tau.unsqueeze(-1) * basis_idx)
        tau_emb = self.quantile(tau_basis)
        fused = state * tau_emb
        return self.head(fused)


def _require_alpha7_features(frame: pd.DataFrame, feature_cols: list[str], *, name: str) -> None:
    missing = [c for c in feature_cols if c not in frame.columns and c not in DERIVED_FEATURES]
    if missing:
        raise RuntimeError(f"{name}: alpha7 IQN feature contract missing columns: {missing[:40]}")
    legacy = [c for c in feature_cols if str(c).startswith("clean_regime4_2024_unsup_v1_")]
    if legacy:
        raise RuntimeError(f"{name}: legacy clean regime features are not allowed: {legacy[:20]}")


def _feature_matrix(frame: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    _require_alpha7_features(frame, feature_cols, name="feature_matrix")
    feat = prepare_features(frame, side_hint=0, feature_cols=feature_cols)
    return feat.replace([np.inf, -np.inf], np.nan)


def _fit_scaler(x: pd.DataFrame) -> dict[str, np.ndarray]:
    arr = x.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, med)
    mean = arr.mean(axis=0).astype(np.float32)
    std = arr.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return {"median": med, "mean": mean, "std": std}


def _apply_scaler(x: pd.DataFrame, scaler: dict[str, np.ndarray]) -> np.ndarray:
    arr = x.to_numpy(dtype=np.float32)
    med = scaler["median"].astype(np.float32)
    arr = np.where(np.isfinite(arr), arr, med)
    return ((arr - scaler["mean"]) / scaler["std"]).astype(np.float32)


def _simulate_action_targets(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    notional: float,
    tp: float,
    sl: float,
    max_hold: int,
    fee: float,
    slip: float,
    cost_mult: float,
    margin_limit: float,
    dd_lambda: float,
    liquidation_penalty: float,
    entry_hurdle: float,
    theta_penalty: float,
) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    targets = np.zeros((len(indices), 3), dtype=np.float32)
    horizon = int(max(max_hold, 1))
    for row_i, idx_raw in enumerate(indices):
        idx = int(idx_raw)
        entry_i = min(idx + 1, len(frame) - 1)
        if entry_i >= len(frame) - 1:
            continue
        for action, side in ((1, 1), (2, -1)):
            entry = float(open_px[entry_i])
            if entry <= 0.0:
                continue
            entry = entry * (1.0 + slip_eff if side > 0 else 1.0 - slip_eff)
            end_i = min(entry_i + horizon, len(frame) - 1)
            realized: float | None = None
            max_dd = 0.0
            liquidated = False
            exit_j = end_i
            for j in range(entry_i + 1, end_i + 1):
                exit_j = j
                if side > 0:
                    favorable = float(high[j] / max(entry, 1e-12) - 1.0) * float(notional)
                    adverse = float(low[j] / max(entry, 1e-12) - 1.0) * float(notional)
                else:
                    favorable = float(entry / max(low[j], 1e-12) - 1.0) * float(notional)
                    adverse = float(entry / max(high[j], 1e-12) - 1.0) * float(notional)
                max_dd = max(max_dd, max(0.0, -adverse))
                if adverse <= -float(margin_limit):
                    realized = -float(margin_limit)
                    liquidated = True
                    break
                if adverse <= -abs(float(sl)):
                    realized = -abs(float(sl))
                    break
                if favorable >= float(tp):
                    realized = float(tp)
                    break
            if realized is None:
                exit_px = float(close[end_i])
                exit_px = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
                raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
                realized = float(raw) * float(notional)
            hold_frac = float(max(exit_j - entry_i, 1) / max(horizon, 1))
            reward = float(realized) - 2.0 * fee_eff * float(notional) - float(entry_hurdle) - float(theta_penalty) * hold_frac
            if max_dd > float(margin_limit):
                reward -= float(dd_lambda) * (max_dd - float(margin_limit)) ** 2
            if liquidated:
                reward -= float(liquidation_penalty)
            targets[row_i, action] = float(np.clip(reward, -2.0, 2.0))
    return targets


def _quantile_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    tau: torch.Tensor,
    kappa: float = 1.0,
    sample_weight: torch.Tensor | None = None,
    action_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    # pred: [B, N, A], target: [B, A], tau: [B, N]
    td = target.unsqueeze(1) - pred
    abs_td = td.abs()
    huber = torch.where(abs_td <= kappa, 0.5 * td.pow(2), kappa * (abs_td - 0.5 * kappa))
    weight = (tau.unsqueeze(-1) - (td.detach() < 0.0).float()).abs()
    loss = weight * huber / kappa
    if action_weight is not None:
        loss = loss * action_weight.unsqueeze(1)
    loss = loss.mean(dim=(1, 2))
    if sample_weight is not None:
        loss = loss * sample_weight
    return loss.mean()


def _sample_weights_from_targets(
    y: np.ndarray,
    *,
    tail_weight: float,
    recent_mix_ratio: float,
    recent_window: int,
) -> np.ndarray:
    best_action = np.argmax(y, axis=1)
    counts = np.bincount(best_action, minlength=y.shape[1]).astype(np.float64)
    class_w = np.sqrt(max(float(len(y)), 1.0) / np.maximum(counts, 1.0))
    weights = class_w[best_action]

    risky_tail = np.maximum(0.0, -np.min(y[:, 1:3], axis=1))
    if np.nanmax(risky_tail) > 0.0:
        risky_tail = risky_tail / max(float(np.nanpercentile(risky_tail, 95)), 1e-8)
        weights *= 1.0 + float(tail_weight) * np.clip(risky_tail, 0.0, 3.0)

    if recent_mix_ratio > 0.0 and recent_window > 0:
        n_recent = min(int(recent_window), len(weights))
        if n_recent > 0:
            recent_boost = 1.0 + float(np.clip(recent_mix_ratio, 0.0, 2.0))
            weights[-n_recent:] *= recent_boost

    weights = np.where(np.isfinite(weights), weights, 1.0)
    return np.clip(weights, 1e-3, np.nanpercentile(weights, 99.5)).astype(np.float32)


def _action_weights_from_targets(y: np.ndarray, *, noncash_best_boost: float, tail_action_boost: float) -> np.ndarray:
    weights = np.ones_like(y, dtype=np.float32)
    best_action = np.argmax(y, axis=1)
    noncash_best = best_action > 0
    if np.any(noncash_best):
        weights[noncash_best, best_action[noncash_best]] += float(noncash_best_boost)
    tail = np.maximum(0.0, -y[:, 1:3])
    if np.nanmax(tail) > 0.0:
        tail = tail / max(float(np.nanpercentile(tail, 95)), 1e-8)
        weights[:, 1:3] += float(tail_action_boost) * np.clip(tail, 0.0, 3.0)
    return weights.astype(np.float32)


def _sample_tau(batch: int, n_tau: int, device: torch.device, dtype: torch.dtype, *, tail_mix: float, tail_max: float) -> torch.Tensor:
    n_tau = int(n_tau)
    n_tail = int(round(n_tau * float(np.clip(tail_mix, 0.0, 0.95))))
    n_base = max(1, n_tau - n_tail)
    base = torch.rand((batch, n_base), device=device, dtype=dtype)
    if n_tail <= 0:
        tau = base
    else:
        tail = torch.rand((batch, n_tail), device=device, dtype=dtype) * float(np.clip(tail_max, 0.01, 1.0))
        tau = torch.cat([tail, base], dim=1)
    return tau.clamp(0.001, 0.999)


def _redo_rejuvenate(model: nn.Module, *, tau: float, ratio: float) -> int:
    reset_count = 0
    max_ratio = float(np.clip(ratio, 0.0, 0.50))
    if max_ratio <= 0.0:
        return 0
    for module in model.modules():
        if not isinstance(module, nn.Linear) or module.out_features < 16:
            continue
        with torch.no_grad():
            row_norm = module.weight.detach().norm(dim=1)
            mean_norm = row_norm.mean().clamp_min(1e-12)
            weak = torch.nonzero(row_norm < float(tau) * mean_norm, as_tuple=False).flatten()
            if weak.numel() == 0:
                continue
            max_reset = max(1, int(module.out_features * max_ratio))
            weak = weak[:max_reset]
            fan_in = module.weight.shape[1]
            bound = math.sqrt(6.0 / max(fan_in + module.out_features, 1))
            module.weight[weak].uniform_(-bound, bound)
            if module.bias is not None:
                module.bias[weak].zero_()
            reset_count += int(weak.numel())
    return reset_count


def _train_iqn(
    x_train: np.ndarray,
    y_train: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    tau_samples: int,
    seed: int,
    tail_tau_mix: float,
    tail_tau_max: float,
    balanced_replay: bool,
    tail_sample_weight: float,
    recent_mix_ratio: float,
    recent_window: int,
    cql_alpha: float,
    anti_flat_lambda: float,
    anti_flat_edge: float,
    grad_clip: float,
    redo_enable: bool,
    redo_interval: int,
    redo_tau: float,
    redo_ratio: float,
) -> tuple[IQNFallbackNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = IQNFallbackNet(state_dim=x_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    sample_w = _sample_weights_from_targets(
        y_train,
        tail_weight=float(tail_sample_weight),
        recent_mix_ratio=float(recent_mix_ratio),
        recent_window=int(recent_window),
    )
    action_w = _action_weights_from_targets(y_train, noncash_best_boost=1.5, tail_action_boost=0.35)
    ds = TensorDataset(
        torch.from_numpy(x_train),
        torch.from_numpy(y_train.astype(np.float32)),
        torch.from_numpy(sample_w.astype(np.float32)),
        torch.from_numpy(action_w.astype(np.float32)),
    )
    sampler = (
        WeightedRandomSampler(weights=torch.from_numpy(sample_w.astype(np.float64)), num_samples=len(sample_w), replacement=True)
        if bool(balanced_replay)
        else None
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=(sampler is None), sampler=sampler, drop_last=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=2, min_lr=1e-5)
    losses: list[float] = []
    aux_losses: list[dict[str, float]] = []
    redo_count = 0
    for epoch in range(int(epochs)):
        model.train()
        epoch_loss = 0.0
        epoch_cql = 0.0
        epoch_af = 0.0
        n = 0
        for xb, yb, sw, aw in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            sw = sw.to(device)
            aw = aw.to(device)
            tau = _sample_tau(
                len(xb),
                int(tau_samples),
                device,
                xb.dtype,
                tail_mix=float(tail_tau_mix),
                tail_max=float(tail_tau_max),
            )
            pred = model(xb, tau)
            loss = _quantile_huber_loss(pred, yb, tau, sample_weight=sw, action_weight=aw)
            mean_q = pred.mean(dim=1)
            cql_pen = torch.zeros((), device=device)
            if float(cql_alpha) > 0.0:
                cash_best = (yb[:, 0] >= yb[:, 1:3].max(dim=1).values).float()
                risky_lse = torch.logsumexp(mean_q[:, 1:3], dim=1)
                cql_pen = (F.softplus(risky_lse - mean_q[:, 0]) * cash_best).mean()
                loss = loss + float(cql_alpha) * cql_pen
            anti_flat_pen = torch.zeros((), device=device)
            if float(anti_flat_lambda) > 0.0:
                target_edge = yb[:, 1:3].max(dim=1).values - yb[:, 0]
                mask = target_edge > float(anti_flat_edge)
                if bool(mask.any()):
                    pred_edge = mean_q[:, 1:3].max(dim=1).values - mean_q[:, 0]
                    anti_flat_pen = F.relu(float(anti_flat_edge) - pred_edge[mask]).mean()
                    loss = loss + float(anti_flat_lambda) * anti_flat_pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            epoch_loss += float(loss.detach().cpu()) * len(xb)
            epoch_cql += float(cql_pen.detach().cpu()) * len(xb)
            epoch_af += float(anti_flat_pen.detach().cpu()) * len(xb)
            n += len(xb)
        mean_loss = epoch_loss / max(n, 1)
        losses.append(mean_loss)
        aux_losses.append({"cql_pen": epoch_cql / max(n, 1), "anti_flat_pen": epoch_af / max(n, 1), "lr": float(opt.param_groups[0]["lr"])})
        scheduler.step(mean_loss)
        if bool(redo_enable) and int(redo_interval) > 0 and (epoch + 1) % int(redo_interval) == 0:
            redo_count += _redo_rejuvenate(model, tau=float(redo_tau), ratio=float(redo_ratio))
    best_counts = np.bincount(np.argmax(y_train, axis=1), minlength=y_train.shape[1]).astype(int).tolist()
    return model, {
        "device": str(device),
        "losses": losses,
        "aux_losses": aux_losses,
        "best_action_counts": best_counts,
        "sample_weight_mean": float(np.mean(sample_w)),
        "sample_weight_p95": float(np.percentile(sample_w, 95)),
        "balanced_replay": bool(balanced_replay),
        "tail_tau_mix": float(tail_tau_mix),
        "tail_tau_max": float(tail_tau_max),
        "cql_alpha": float(cql_alpha),
        "anti_flat_lambda": float(anti_flat_lambda),
        "anti_flat_edge": float(anti_flat_edge),
        "redo_reset_neurons": int(redo_count),
    }


def _iqn_scores(model: IQNFallbackNet, x: np.ndarray, *, risk_tau: float, num_tau: int, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    taus = torch.linspace(0.01, float(risk_tau), int(num_tau), device=device).view(1, -1)
    out = np.zeros((len(x), 3), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(x), int(batch_size)):
            end = min(start + int(batch_size), len(x))
            xb = torch.from_numpy(x[start:end]).to(device)
            tb = taus.repeat(len(xb), 1)
            q = model(xb, tb).mean(dim=1)
            out[start:end] = q.detach().cpu().numpy().astype(np.float32)
    return out


def _build_iqn_decisions(
    template: pd.DataFrame,
    primary_dec: pd.DataFrame,
    scores: np.ndarray,
    *,
    cvar_min: float,
    edge_min: float,
    notional: float,
    leverage: float,
    tp: float,
    sl: float,
    max_hold: int,
    cooldown: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_dec_like(template)
    primary_cash = ~_active(primary_dec)
    counts = {"cash": 0, "long": 0, "short": 0}
    for i in range(len(out)):
        if not primary_cash[i]:
            counts["cash"] += 1
            continue
        row = scores[i]
        action = int(np.argmax(row))
        best = float(row[action])
        cash_score = float(row[0])
        if action == 0 or best < float(cvar_min) or (best - cash_score) < float(edge_min):
            counts["cash"] += 1
            continue
        side = 1 if action == 1 else -1
        out.at[i, "action"] = int(action)
        out.at[i, "side"] = int(side)
        out.at[i, "notional_exposure"] = float(notional)
        out.at[i, "leverage"] = float(leverage)
        out.at[i, "position_fraction"] = float(min(float(notional) / max(float(leverage), 1e-12), 1.0))
        out.at[i, "take_profit"] = float(tp)
        out.at[i, "stop_loss"] = float(sl)
        out.at[i, "max_hold_bars"] = int(max_hold)
        out.at[i, "cooldown_bars"] = int(cooldown)
        out.at[i, "quality_score"] = float(best - cash_score)
        out.at[i, "confidence"] = float(1.0 / (1.0 + math.exp(-8.0 * (best - cash_score))))
        counts["long" if side > 0 else "short"] += 1
    return out, counts


def _eval_combo(
    frame: pd.DataFrame,
    primary_dec: pd.DataFrame,
    fallback_dec: pd.DataFrame,
    *,
    ref_parent: dict[str, Any],
    runner: dict[str, Any],
    runner_cfg: Any,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    combo = _combine_primary_fallback(primary_dec, fallback_dec)
    return _compact_costs(
        _metrics(
            frame,
            parent_for_features=ref_parent,
            runner=runner,
            runner_cfg=runner_cfg,
            dec=combo,
            fee=fee,
            slip=slip,
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train an Alpha7-feature IQN fallback for 10x CVaR action selection.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=52727)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--batch-size", type=int, default=768)
    ap.add_argument("--lr", type=float, default=8e-4)
    ap.add_argument("--tau-samples", type=int, default=32)
    ap.add_argument("--risk-tau", type=float, default=0.25)
    ap.add_argument("--tail-tau-mix", type=float, default=0.45)
    ap.add_argument("--tail-tau-max", type=float, default=0.25)
    ap.add_argument("--balanced-replay", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tail-sample-weight", type=float, default=1.2)
    ap.add_argument("--recent-mix-ratio", type=float, default=0.30)
    ap.add_argument("--recent-window", type=int, default=60000)
    ap.add_argument("--cql-alpha", type=float, default=0.025)
    ap.add_argument("--anti-flat-lambda", type=float, default=0.06)
    ap.add_argument("--anti-flat-edge", type=float, default=0.002)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--redo-enable", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--redo-interval", type=int, default=4)
    ap.add_argument("--redo-tau", type=float, default=5e-3)
    ap.add_argument("--redo-ratio", type=float, default=0.05)
    ap.add_argument("--notional", type=float, default=10.0)
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--take-profit", type=float, default=0.055)
    ap.add_argument("--stop-loss", type=float, default=0.050)
    ap.add_argument("--max-hold", type=int, default=12)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--margin-limit", type=float, default=0.065)
    ap.add_argument("--dd-lambda", type=float, default=4.0)
    ap.add_argument("--liquidation-penalty", type=float, default=0.75)
    ap.add_argument("--entry-hurdle", type=float, default=0.0)
    ap.add_argument("--theta-penalty", type=float, default=0.0)
    ap.add_argument("--min-val-fallback-trades", type=int, default=0)
    ap.add_argument("--max-val-fallback-trades", type=int, default=0)
    ap.add_argument("--fallback-trade-penalty", type=float, default=0.0)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(args.train_csv))
    eval_df = _rename_clean4_v2(_read(args.eval_csv))
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    primary_parent = joblib.load(baseline.primary_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_parent = joblib.load(baseline.fallback_parent)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)
    feature_cols = list(primary_parent["feature_cols"])
    for frame_name, frame in (("train", train_df), ("val", val_df), ("eval", eval_df)):
        _require_alpha7_features(frame, feature_cols, name=frame_name)

    primary_train = _predict_scaled(primary_parent, train_df, primary_rt)
    primary_val = _predict_scaled(primary_parent, val_df, primary_rt)
    primary_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    fallback_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    fallback_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    x_train_df = _feature_matrix(train_df, feature_cols)
    x_val_df = _feature_matrix(val_df, feature_cols)
    x_eval_df = _feature_matrix(eval_df, feature_cols)
    scaler = _fit_scaler(x_train_df.loc[~_active(primary_train)].reset_index(drop=True))
    x_train_all = _apply_scaler(x_train_df, scaler)
    x_val = _apply_scaler(x_val_df, scaler)
    x_eval = _apply_scaler(x_eval_df, scaler)

    train_cash = ~_active(primary_train)
    max_idx = len(train_df) - int(args.max_hold) - 3
    train_indices = np.flatnonzero(train_cash & (np.arange(len(train_df)) < max_idx)).astype(np.int64)
    if len(train_indices) < 1000:
        raise RuntimeError(f"too few Alpha7 primary-cash train rows for IQN: {len(train_indices)}")
    y_train = _simulate_action_targets(
        train_df,
        train_indices,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        fee=float(primary_parent.get("config", {}).get("fee", 0.0004)),
        slip=float(primary_parent.get("config", {}).get("slip", 0.00015)),
        cost_mult=3.0,
        margin_limit=float(args.margin_limit),
        dd_lambda=float(args.dd_lambda),
        liquidation_penalty=float(args.liquidation_penalty),
        entry_hurdle=float(args.entry_hurdle),
        theta_penalty=float(args.theta_penalty),
    )
    model, train_diag = _train_iqn(
        x_train_all[train_indices],
        y_train,
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        tau_samples=int(args.tau_samples),
        seed=int(args.seed),
        tail_tau_mix=float(args.tail_tau_mix),
        tail_tau_max=float(args.tail_tau_max),
        balanced_replay=bool(args.balanced_replay),
        tail_sample_weight=float(args.tail_sample_weight),
        recent_mix_ratio=float(args.recent_mix_ratio),
        recent_window=int(args.recent_window),
        cql_alpha=float(args.cql_alpha),
        anti_flat_lambda=float(args.anti_flat_lambda),
        anti_flat_edge=float(args.anti_flat_edge),
        grad_clip=float(args.grad_clip),
        redo_enable=bool(args.redo_enable),
        redo_interval=int(args.redo_interval),
        redo_tau=float(args.redo_tau),
        redo_ratio=float(args.redo_ratio),
    )

    val_scores = _iqn_scores(model, x_val, risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    eval_scores = _iqn_scores(model, x_eval, risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")
    baseline_val = _eval_combo(val_df, primary_val, fallback_val, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
    baseline_eval = _eval_combo(eval_df, primary_eval, fallback_eval, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)

    active_val = val_scores[~_active(primary_val)]
    action_best = np.max(active_val[:, 1:3], axis=1) - active_val[:, 0]
    quantiles = [q for q in (0.85, 0.92, 0.97, 0.99, 0.995, 0.999, 0.9995) if len(action_best) > 0]
    edge_grid = sorted({0.0, 0.0020, 0.0100, 0.0200, 0.0500, 0.1000, 0.2000, 0.5000, *[float(np.quantile(action_best, q)) for q in quantiles]})
    best_score_grid = active_val[:, 1:3].max(axis=1) if len(active_val) else np.zeros(0, dtype=np.float32)
    cvar_quantiles = [q for q in (0.85, 0.92, 0.97, 0.99, 0.995, 0.999) if len(best_score_grid) > 0]
    cvar_grid = sorted({-0.005, 0.0, 0.0050, 0.0100, 0.0200, 0.0500, 0.1000, *[float(np.quantile(best_score_grid, q)) for q in cvar_quantiles]})

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cvar_min in cvar_grid:
        for edge_min in edge_grid:
            val_fb, val_counts = _build_iqn_decisions(
                primary_val,
                primary_val,
                val_scores,
                cvar_min=float(cvar_min),
                edge_min=float(edge_min),
                notional=float(args.notional),
                leverage=float(args.leverage),
                tp=float(args.take_profit),
                sl=float(args.stop_loss),
                max_hold=int(args.max_hold),
                cooldown=int(args.cooldown),
            )
            val_metrics = _eval_combo(val_df, primary_val, val_fb, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
            val_fallback_trades = int(val_counts.get("long", 0) + val_counts.get("short", 0))
            eval_fb, eval_counts = _build_iqn_decisions(
                primary_eval,
                primary_eval,
                eval_scores,
                cvar_min=float(cvar_min),
                edge_min=float(edge_min),
                notional=float(args.notional),
                leverage=float(args.leverage),
                tp=float(args.take_profit),
                sl=float(args.stop_loss),
                max_hold=int(args.max_hold),
                cooldown=int(args.cooldown),
            )
            eval_metrics = _eval_combo(eval_df, primary_eval, eval_fb, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
            oos_fallback_trades = int(eval_counts.get("long", 0) + eval_counts.get("short", 0))
            raw_selection_score = float(_score(val_metrics))
            selection_score = raw_selection_score - float(args.fallback_trade_penalty) * float(val_fallback_trades)
            row = {
                "cvar_min": float(cvar_min),
                "edge_min": float(edge_min),
                "selection_score": float(selection_score),
                "raw_selection_score": float(raw_selection_score),
                "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                "oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(eval_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(eval_metrics["cost3"]["trades"]),
                "oos_cost3_wr": float(eval_metrics["cost3"]["wr"]),
                "delta_vs_baseline_oos_cost3_pnl": float(eval_metrics["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
                "val_counts": val_counts,
                "eval_counts": eval_counts,
                "val_fallback_trades": val_fallback_trades,
                "oos_fallback_trades": oos_fallback_trades,
            }
            rows.append(row)
            if int(row["val_fallback_trades"]) < int(args.min_val_fallback_trades):
                continue
            if int(args.max_val_fallback_trades) > 0 and int(row["val_fallback_trades"]) > int(args.max_val_fallback_trades):
                continue
            if best is None or row["selection_score"] > best["selection_score"]:
                best = row
    if best is None:
        constrained_rows = [
            r
            for r in rows
            if int(r["val_fallback_trades"]) >= int(args.min_val_fallback_trades)
        ]
        if int(args.max_val_fallback_trades) > 0:
            constrained_rows = [
                r
                for r in constrained_rows
                if int(r["val_fallback_trades"]) <= int(args.max_val_fallback_trades)
            ]
        if constrained_rows:
            best = max(constrained_rows, key=lambda r: r["selection_score"])
        else:
            best = min(rows, key=lambda r: (int(r["val_fallback_trades"]), -float(r["selection_score"])))

    best_val_fb, best_val_counts = _build_iqn_decisions(
        primary_val,
        primary_val,
        val_scores,
        cvar_min=float(best["cvar_min"]),
        edge_min=float(best["edge_min"]),
        notional=float(args.notional),
        leverage=float(args.leverage),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        cooldown=int(args.cooldown),
    )
    best_eval_fb, best_eval_counts = _build_iqn_decisions(
        primary_eval,
        primary_eval,
        eval_scores,
        cvar_min=float(best["cvar_min"]),
        edge_min=float(best["edge_min"]),
        notional=float(args.notional),
        leverage=float(args.leverage),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        cooldown=int(args.cooldown),
    )
    best_val_metrics = _eval_combo(val_df, primary_val, best_val_fb, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)
    best_eval_metrics = _eval_combo(eval_df, primary_eval, best_eval_fb, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee, slip=slip)

    model_path = args.out_dir / "iqn_fallback_10x.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler": scaler,
            "network": {"hidden_dim": 256, "n_cos": 64, "action_dim": 3},
            "runtime": {
                "risk_tau": float(args.risk_tau),
                "cvar_min": float(best["cvar_min"]),
                "edge_min": float(best["edge_min"]),
                "notional": float(args.notional),
                "leverage": float(args.leverage),
                "take_profit": float(args.take_profit),
                "stop_loss": float(args.stop_loss),
                "max_hold_bars": int(args.max_hold),
                "cooldown_bars": int(args.cooldown),
            },
            "train_diag": train_diag,
        },
        model_path,
    )

    ranking_path = args.out_dir / "ranking.csv"
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(ranking_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha7 feature contract IQN fallback. It trains counterfactual CASH/LONG10x/SHORT10x return distributions on Alpha7 primary-CASH rows and selects live actions using lower-tail CVaR instead of mean Q.",
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "feature_contract": {
            "source": "alpha7 primary_parent feature_cols",
            "feature_count": int(len(feature_cols)),
            "feature_cols": feature_cols,
        },
        "training": {
            "cash_train_rows": int(len(train_indices)),
            "target_reward_mean": y_train.mean(axis=0).tolist(),
            "target_reward_p05": np.quantile(y_train, 0.05, axis=0).tolist(),
            "target_reward_p50": np.quantile(y_train, 0.50, axis=0).tolist(),
            "target_reward_p95": np.quantile(y_train, 0.95, axis=0).tolist(),
            "train_diag": train_diag,
        },
        "risk_contract": {
            "notional": float(args.notional),
            "leverage": float(args.leverage),
            "take_profit": float(args.take_profit),
            "stop_loss": float(args.stop_loss),
            "max_hold_bars": int(args.max_hold),
            "margin_limit": float(args.margin_limit),
            "dd_lambda": float(args.dd_lambda),
            "liquidation_penalty": float(args.liquidation_penalty),
            "cvar_risk_tau": float(args.risk_tau),
            "tail_tau_mix": float(args.tail_tau_mix),
            "tail_tau_max": float(args.tail_tau_max),
            "entry_hurdle": float(args.entry_hurdle),
            "theta_penalty": float(args.theta_penalty),
            "min_val_fallback_trades": int(args.min_val_fallback_trades),
            "max_val_fallback_trades": int(args.max_val_fallback_trades),
            "fallback_trade_penalty": float(args.fallback_trade_penalty),
        },
        "baseline": {
            "val": baseline_val,
            "oos": baseline_eval,
        },
        "best_by_selection": {
            **best,
            "val_metrics": best_val_metrics,
            "oos_metrics": best_eval_metrics,
            "best_val_counts": best_val_counts,
            "best_eval_counts": best_eval_counts,
        },
        "artifacts": {
            "model": str(model_path),
            "ranking_csv": str(ranking_path),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "feature_contract_fail_fast": True,
            "legacy_clean_regime4_allowed": False,
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "model": str(model_path),
                "best_cvar_min": float(best["cvar_min"]),
                "best_edge_min": float(best["edge_min"]),
                "oos_cost3_pnl": float(best_eval_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(best_eval_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(best_eval_metrics["cost3"]["trades"]),
                "delta_vs_baseline_oos_cost3_pnl": float(best_eval_metrics["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
