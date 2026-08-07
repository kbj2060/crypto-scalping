#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FEATURE_COLS,
)
from scripts.eval_hf_entry_overlay_grid import _audit  # noqa: E402
from scripts.train_eval_alphazero_style_governor_2026 import (  # noqa: E402
    DEFAULT_MODEL_OUT as DEFAULT_AZ_MODEL,
    _predict_pv,
)
from scripts.train_eval_dsac_replacement_heads_2026 import (  # noqa: E402
    DEFAULT_EVAL_CSV,
    DEFAULT_EXIT_BUNDLE,
    DEFAULT_POLICY,
    DEFAULT_SELECTION,
    DEFAULT_TRAIN_CSV,
    _load_selected,
    _read,
)
from scripts.train_eval_hf_no_limit_exit_governor import (  # noqa: E402
    _base_frame,
    _compact,
    backtest_no_limit_exit,
    collect_exit_samples,
    train_exit_model,
)
from scripts.train_eval_muzero_style_exit_governor_2026 import DEFAULT_MODEL_OUT as DEFAULT_MZ_EXIT_MODEL  # noqa: E402
from scripts.train_eval_muzero_style_governor_2026 import (  # noqa: E402
    DEFAULT_MODEL_OUT as DEFAULT_MZ_ENTRY_MODEL,
    _load_az_exit,
)
from scripts.train_eval_zero_style_remaining_layers_2026 import _load_mz_exit, _load_mz_risk, _load_pv  # noqa: E402
from scripts.train_eval_zero_style_risk_overlay_2026 import (  # noqa: E402
    DEFAULT_AZ_RISK_OUT,
    DEFAULT_MZ_RISK_OUT,
    RISK_ACTIONS,
    RISK_SCALES,
    _apply_scale,
    _mz_entry_decisions,
    _predict_mz_risk,
    _state_frame,
)


DEFAULT_REPORT_OUT = ROOT / "data/ensemble/reports/dt_lifecycle_vs_muzero_az_2026.json"
DEFAULT_MODEL_DIR = ROOT / "data/ensemble/supervised/dt_lifecycle_iql_cvar"


def _close(df: pd.DataFrame) -> np.ndarray:
    return (
        pd.to_numeric(df["close"], errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
        .ffill()
        .to_numpy(dtype=np.float64)
    )


def _date_range(df: pd.DataFrame) -> list[str]:
    if "timestamp" not in df.columns or df.empty:
        return ["", ""]
    ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
    if ts.empty:
        return ["", ""]
    return [str(ts.min()), str(ts.max())]


def _standardize_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(x, dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    z = (np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0) - mean) / std
    return z.astype(np.float32), mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return ((arr - mean.astype(np.float32)) / np.maximum(std.astype(np.float32), 1e-6)).astype(np.float32)


def _score(bt: dict[str, Any], mdd_weight: float) -> float:
    trades_per_day = float(bt.get("trades_per_day", 0.0) or 0.0)
    trade_sparsity_penalty = 120.0 * max(0.0, 1.0 - trades_per_day)
    return float(bt.get("pnl", 0.0)) + float(mdd_weight) * float(bt.get("mdd", 0.0)) - trade_sparsity_penalty


def _clamp_decisions(dec: pd.DataFrame, *, max_notional: float, leverage_cap: float) -> pd.DataFrame:
    out = dec.copy()
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    notional = np.where(active, np.clip(notional, 0.0, float(max_notional)), 0.0)
    leverage = np.where(active, np.clip(leverage, 1.0, float(leverage_cap)), 1.0)
    flat = notional <= 0.05
    out.loc[:, "notional_exposure"] = notional
    out.loc[:, "leverage"] = leverage
    out.loc[:, "position_fraction"] = notional / np.maximum(leverage, 1e-12)
    out.loc[flat, ["action", "side", "notional_exposure", "position_fraction"]] = 0
    out.loc[flat, "leverage"] = 1.0
    return out


def _slice_precomputed(
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    mask: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    idx = np.flatnonzero(np.asarray(mask, dtype=bool))
    feat, dec, close, fill = precomputed
    return (
        feat.iloc[idx].reset_index(drop=True),
        dec.iloc[idx].reset_index(drop=True),
        close[idx],
        fill[idx],
    )


def _monthly(
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    if "timestamp" not in df.columns:
        return {}
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    months = sorted(ts.dropna().dt.to_period("M").unique())
    out: dict[str, Any] = {}
    for month in months:
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if not mask.any():
            continue
        sub = df.loc[mask].reset_index(drop=True)
        pre = _slice_precomputed(precomputed, mask)
        bt = backtest_no_limit_exit(
            sub,
            policy,
            exit_model,
            entry_config=entry_cfg,
            risk_config=risk_cfg,
            exit_threshold=float(exit_cfg["exit_threshold"]),
            min_exit_age=int(exit_cfg["min_exit_age"]),
            fee=float(fee),
            slip=float(slip),
            precomputed=pre,
        )
        out[str(month)] = _compact(bt)
    return out


@dataclass(frozen=True)
class DTLifecycleConfig:
    seq_len: int = 24
    horizon: int = 144
    d_model: int = 96
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.08
    batch_size: int = 512
    epochs: int = 3
    lr: float = 8e-4
    max_train_samples: int = 30000
    min_train_edge: float = 0.0012
    cvar_alpha: float = 0.10
    adverse_penalty: float = 0.55
    cvar_penalty: float = 0.45
    cql_alpha: float = 0.08
    conservative_penalty: float = 0.65
    min_notional: float = 0.35
    seed: int = 42


class DecisionTransformerLifecycle(nn.Module):
    def __init__(self, state_dim: int, cfg: DTLifecycleConfig):
        super().__init__()
        self.action_emb = nn.Embedding(3, cfg.d_model)
        self.state_proj = nn.Linear(int(state_dim), cfg.d_model)
        self.cond_proj = nn.Linear(3, cfg.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, int(cfg.seq_len), cfg.d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.action_head = nn.Linear(cfg.d_model, 3)

    def forward(self, states: torch.Tensor, prev_actions: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.state_proj(states) + self.action_emb(prev_actions.long()) + self.cond_proj(cond) + self.pos_emb[:, : states.shape[1]]
        mask = torch.triu(torch.ones(states.shape[1], states.shape[1], device=states.device, dtype=torch.bool), diagonal=1)
        z = self.encoder(h, mask=mask)
        return self.action_head(self.norm(z[:, -1]))


class ConservativeCritic(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(int(state_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.LayerNorm(int(hidden_dim)),
            nn.SiLU(),
        )
        self.q_head = nn.Linear(int(hidden_dim), 3)
        self.cvar_head = nn.Linear(int(hidden_dim), 3)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.q_head(h), self.cvar_head(h)


class SequencePolicyDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        cond: np.ndarray,
        indices: np.ndarray,
        *,
        seq_len: int,
    ):
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.cond = np.asarray(cond, dtype=np.float32)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        end = int(self.indices[int(i)])
        start = end - self.seq_len + 1
        x = self.features[start : end + 1]
        c = self.cond[start : end + 1]
        prev = np.zeros(self.seq_len, dtype=np.int64)
        if start > 0:
            prev[:] = self.labels[start - 1 : end]
        else:
            prev[1:] = self.labels[start:end]
        y = np.asarray(self.labels[end], dtype=np.int64)
        return torch.from_numpy(x), torch.from_numpy(prev), torch.from_numpy(c), torch.from_numpy(y.reshape(()))


def _future_targets(
    df: pd.DataFrame,
    *,
    horizon: int,
    fee: float,
    slip: float,
    max_notional: float,
    cvar_alpha: float,
    adverse_penalty: float,
    cvar_penalty: float,
    min_train_edge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    n = len(close)
    q = np.zeros((n, 3), dtype=np.float32)
    cvar = np.zeros((n, 3), dtype=np.float32)
    labels = np.zeros(n, dtype=np.int64)
    cond = np.zeros((n, 3), dtype=np.float32)
    cost = 2.0 * float(fee + slip) * float(max_notional)
    usable = max(0, n - int(horizon) - 2)
    for i in range(usable):
        base = max(float(close[i]), 1e-12)
        fut = close[i + 1 : i + 1 + int(horizon)]
        long_path = (fut / base - 1.0) * float(max_notional) - cost
        short_path = (base / np.maximum(fut, 1e-12) - 1.0) * float(max_notional) - cost
        for action, path in ((ACTION_LONG, long_path), (ACTION_SHORT, short_path)):
            run_min = np.minimum.accumulate(path)
            adverse = max(0.0, -float(np.min(run_min)))
            tail = float(np.quantile(path, float(cvar_alpha))) if len(path) else 0.0
            best = float(np.max(path)) if len(path) else 0.0
            q[i, action] = float(best - float(adverse_penalty) * adverse - float(cvar_penalty) * max(0.0, -tail))
            cvar[i, action] = tail
        conservative = q[i] - float(cvar_penalty) * np.maximum(0.0, -cvar[i])
        best_action = int(np.argmax(conservative))
        labels[i] = best_action if float(conservative[best_action]) >= float(min_train_edge) else ACTION_CASH
        cond[i] = np.asarray(
            [
                float(np.max(conservative)),
                float(np.min(cvar[i])),
                float(cost),
            ],
            dtype=np.float32,
        )
    meta = {
        "usable_rows": int(usable),
        "horizon": int(horizon),
        "fee": float(fee),
        "slip": float(slip),
        "max_notional": float(max_notional),
        "cost_round_trip_at_max_notional": float(cost),
        "label_counts": {
            "cash": int((labels == ACTION_CASH).sum()),
            "long": int((labels == ACTION_LONG).sum()),
            "short": int((labels == ACTION_SHORT).sum()),
        },
        "q_quantiles": np.quantile(q[:usable].max(axis=1), [0.0, 0.25, 0.5, 0.75, 0.95, 1.0]).round(8).tolist() if usable else [],
        "cvar_quantiles": np.quantile(cvar[:usable].min(axis=1), [0.0, 0.05, 0.25, 0.5, 0.75, 1.0]).round(8).tolist() if usable else [],
    }
    return q, cvar, labels, cond, meta


def _train_dt(
    xz: np.ndarray,
    labels: np.ndarray,
    cond: np.ndarray,
    cfg: DTLifecycleConfig,
    *,
    device: str,
) -> tuple[DecisionTransformerLifecycle, dict[str, Any]]:
    rng = np.random.default_rng(int(cfg.seed))
    upper = len(xz) - int(cfg.horizon) - 2
    idx = np.arange(int(cfg.seq_len) - 1, max(int(cfg.seq_len), upper), dtype=np.int64)
    if len(idx) > int(cfg.max_train_samples):
        idx = np.sort(rng.choice(idx, size=int(cfg.max_train_samples), replace=False))
    ds = SequencePolicyDataset(xz, labels, cond, idx, seq_len=int(cfg.seq_len))
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    net = DecisionTransformerLifecycle(xz.shape[1], cfg).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    class_counts = np.bincount(labels[idx], minlength=3).astype(np.float64)
    weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    weights = weights / np.mean(weights)
    weight_t = torch.tensor(weights, dtype=torch.float32, device=device)
    losses: list[float] = []
    for _ in range(int(cfg.epochs)):
        total = 0.0
        n = 0
        net.train()
        for xb, ab, cb, yb in loader:
            xb = xb.to(device)
            ab = ab.to(device)
            cb = cb.to(device)
            yb = yb.to(device)
            logits = net(xb, ab, cb)
            loss = F.cross_entropy(logits, yb, weight=weight_t)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
    meta = {
        "samples": int(len(idx)),
        "class_counts": {"cash": int(class_counts[0]), "long": int(class_counts[1]), "short": int(class_counts[2])},
        "epochs": int(cfg.epochs),
        "final_loss": float(losses[-1]) if losses else None,
    }
    return net, meta


def _train_critic(
    xz: np.ndarray,
    q_targets: np.ndarray,
    cvar_targets: np.ndarray,
    labels: np.ndarray,
    cfg: DTLifecycleConfig,
    *,
    device: str,
) -> tuple[ConservativeCritic, dict[str, Any]]:
    upper = len(xz) - int(cfg.horizon) - 2
    x = xz[:upper]
    q = q_targets[:upper]
    cvar = cvar_targets[:upper]
    y = labels[:upper]
    if len(x) > int(cfg.max_train_samples):
        rng = np.random.default_rng(int(cfg.seed) + 7)
        take = np.sort(rng.choice(len(x), size=int(cfg.max_train_samples), replace=False))
        x, q, cvar, y = x[take], q[take], cvar[take], y[take]
    ds = TensorDataset(
        torch.from_numpy(x.astype(np.float32)),
        torch.from_numpy(q.astype(np.float32)),
        torch.from_numpy(cvar.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    loader = DataLoader(ds, batch_size=int(cfg.batch_size), shuffle=True, drop_last=False)
    net = ConservativeCritic(xz.shape[1], hidden_dim=max(96, int(cfg.d_model))).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.lr), weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(int(cfg.epochs)):
        total = 0.0
        n = 0
        net.train()
        for xb, qb, cb, yb in loader:
            xb = xb.to(device)
            qb = qb.to(device)
            cb = cb.to(device)
            yb = yb.to(device)
            q_pred, c_pred = net(xb)
            q_loss = F.smooth_l1_loss(q_pred, qb)
            c_loss = F.smooth_l1_loss(c_pred, cb)
            chosen = q_pred.gather(1, yb[:, None]).squeeze(1)
            cql = (torch.logsumexp(q_pred, dim=1) - chosen).mean()
            loss = q_loss + 0.5 * c_loss + float(cfg.cql_alpha) * cql
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
    meta = {
        "samples": int(len(x)),
        "epochs": int(cfg.epochs),
        "final_loss": float(losses[-1]) if losses else None,
        "q_target_mean": float(np.mean(q)) if len(q) else 0.0,
        "cvar_target_mean": float(np.mean(cvar)) if len(cvar) else 0.0,
    }
    return net, meta


def _predict_dt(
    net: DecisionTransformerLifecycle,
    xz: np.ndarray,
    cond: np.ndarray,
    cfg: DTLifecycleConfig,
    *,
    device: str,
    batch_size: int,
) -> np.ndarray:
    n = len(xz)
    probs = np.zeros((n, 3), dtype=np.float32)
    if n == 0:
        return probs
    seq_len = int(cfg.seq_len)
    padded_x = np.vstack([np.repeat(xz[:1], seq_len - 1, axis=0), xz]).astype(np.float32)
    inference_cond = np.tile(
        np.asarray([max(float(cfg.min_train_edge) * 2.0, 0.002), -0.02, 0.0], dtype=np.float32),
        (n + seq_len - 1, 1),
    )
    if len(cond):
        inference_cond[seq_len - 1 :] = np.asarray(cond, dtype=np.float32)
    net.eval()
    with torch.no_grad():
        for start in range(0, n, int(batch_size)):
            end = min(n, start + int(batch_size))
            seqs = np.stack([padded_x[i : i + seq_len] for i in range(start, end)], axis=0)
            cseq = np.stack([inference_cond[i : i + seq_len] for i in range(start, end)], axis=0)
            prev = np.zeros((end - start, seq_len), dtype=np.int64)
            logits = net(
                torch.from_numpy(seqs).to(device),
                torch.from_numpy(prev).to(device),
                torch.from_numpy(cseq).to(device),
            )
            probs[start:end] = torch.softmax(logits, dim=-1).detach().cpu().numpy().astype(np.float32)
    return probs


def _predict_critic(
    net: ConservativeCritic,
    xz: np.ndarray,
    *,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    q_rows: list[np.ndarray] = []
    c_rows: list[np.ndarray] = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(xz), int(batch_size)):
            xb = torch.from_numpy(xz[start : start + int(batch_size)].astype(np.float32)).to(device)
            q, c = net(xb)
            q_rows.append(q.detach().cpu().numpy().astype(np.float32))
            c_rows.append(c.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(q_rows) if q_rows else np.zeros((0, 3), dtype=np.float32), np.concatenate(c_rows) if c_rows else np.zeros((0, 3), dtype=np.float32)


class ExitProbabilityModel:
    classes_ = np.asarray([0, 1], dtype=np.int64)

    def __init__(self, model: Any):
        self.model = model

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(x)


def _candidate_decisions(
    template_dec: pd.DataFrame,
    probs: np.ndarray,
    q: np.ndarray,
    cvar: np.ndarray,
    cfg: DTLifecycleConfig,
    *,
    min_lower_edge: float,
    max_cvar_loss: float,
    max_notional: float,
    leverage_cap: float,
) -> pd.DataFrame:
    out = template_dec.copy()
    row = np.arange(len(out))
    lower_all = q - float(cfg.conservative_penalty) * np.maximum(0.0, -cvar)
    utility = np.log(np.maximum(probs, 1e-8)) + lower_all / 0.025
    utility[:, ACTION_CASH] = np.log(np.maximum(probs[:, ACTION_CASH], 1e-8))
    utility[:, ACTION_LONG] = np.where(cvar[:, ACTION_LONG] >= -abs(float(max_cvar_loss)), utility[:, ACTION_LONG], -1e9)
    utility[:, ACTION_SHORT] = np.where(cvar[:, ACTION_SHORT] >= -abs(float(max_cvar_loss)), utility[:, ACTION_SHORT], -1e9)
    action = np.argmax(utility, axis=1).astype(np.int64)
    confidence = probs[row, action]
    q_chosen = q[row, action]
    cvar_chosen = cvar[row, action]
    lower = lower_all[row, action]
    active = (action != ACTION_CASH) & (lower >= float(min_lower_edge)) & (cvar_chosen >= -abs(float(max_cvar_loss)))
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0))
    side = np.where(active, side, 0)
    sizing_denominator = max(0.06, abs(float(max_cvar_loss)) * 3.0)
    scale = np.clip((lower - float(min_lower_edge)) / sizing_denominator, 0.0, 1.0)
    notional = np.where(active, float(cfg.min_notional) + (float(max_notional) - float(cfg.min_notional)) * scale, 0.0)
    base_leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    leverage = np.where(active, np.clip(np.maximum(base_leverage, notional), 1.0, float(leverage_cap)), 1.0)
    out.loc[:, "action"] = np.where(side > 0, ACTION_LONG, np.where(side < 0, ACTION_SHORT, ACTION_CASH)).astype(int)
    out.loc[:, "side"] = side.astype(int)
    out.loc[:, "notional_exposure"] = np.clip(notional, 0.0, float(max_notional))
    out.loc[:, "leverage"] = leverage
    out.loc[:, "position_fraction"] = out["notional_exposure"] / np.maximum(out["leverage"], 1e-12)
    out.loc[:, "quality_score"] = lower.astype(np.float64)
    out.loc[:, "confidence"] = confidence.astype(np.float64)
    return _clamp_decisions(out, max_notional=float(max_notional), leverage_cap=float(leverage_cap))


def _build_zero_style_current(
    df: pd.DataFrame,
    policy: dict[str, Any],
    entry_cfg: dict[str, Any],
    *,
    mz_entry: Any,
    az_risk: Any,
    mz_risk: Any,
    device: str,
    max_notional: float,
    leverage_cap: float,
    stage2_gamma: float,
    stage2_prior: float,
    stage2_depth: int,
    stage2_score_floor: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    feat, dec0, close, fill, scores, probs, vals = _mz_entry_decisions(df, policy, entry_cfg, mz_entry, device=device)
    state0 = _state_frame(feat, dec0, scores, probs, vals)
    az_x = state0.reindex(columns=az_risk.feature_cols).to_numpy(dtype=np.float32)
    az_probs, az_values = _predict_pv(az_risk, az_x, device)
    keep_idx = int(np.flatnonzero(np.isclose(RISK_SCALES, 1.0))[0])
    az_idx = np.argmax(az_probs, axis=1)
    az_idx = np.where(az_values < -0.15, keep_idx, az_idx)
    dec1 = _apply_scale(dec0, az_idx)
    state2 = _state_frame(feat, dec1, scores, probs, vals)
    risk_x = state2.reindex(columns=mz_risk.feature_cols).to_numpy(dtype=np.float32)
    risk_scores, _, _ = _predict_mz_risk(
        mz_risk,
        risk_x,
        device=device,
        gamma=float(stage2_gamma),
        prior_weight=float(stage2_prior),
        depth=int(stage2_depth),
    )
    stage_idx = np.where(risk_scores.max(axis=1) < float(stage2_score_floor), keep_idx, np.argmax(risk_scores, axis=1))
    dec2 = _apply_scale(dec1, stage_idx)
    return feat, _clamp_decisions(dec2, max_notional=float(max_notional), leverage_cap=float(leverage_cap)), close, fill


def _run(
    name: str,
    df: pd.DataFrame,
    policy: dict[str, Any],
    exit_model: Any,
    entry_cfg: dict[str, Any],
    risk_cfg: dict[str, Any],
    exit_cfg: dict[str, Any],
    precomputed: tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray],
    *,
    fee: float,
    slip: float,
    monthly: bool = False,
    mdd_weight: float = 3.0,
) -> dict[str, Any]:
    bt = backtest_no_limit_exit(
        df,
        policy,
        exit_model,
        entry_config=entry_cfg,
        risk_config=risk_cfg,
        exit_threshold=float(exit_cfg["exit_threshold"]),
        min_exit_age=int(exit_cfg["min_exit_age"]),
        fee=float(fee),
        slip=float(slip),
        precomputed=precomputed,
    )
    row = {"name": name, "eval": _compact(bt), "score": _score(bt, mdd_weight)}
    if monthly:
        row["monthly"] = _monthly(df, policy, exit_model, entry_cfg, risk_cfg, exit_cfg, precomputed, fee=float(fee), slip=float(slip))
    return row


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare current MuZero/AZ zero-style stack with DT Lifecycle + IQL/CQL + CVaR candidate.")
    p.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    p.add_argument("--exit-bundle", type=Path, default=DEFAULT_EXIT_BUNDLE)
    p.add_argument("--selection-report", type=Path, default=DEFAULT_SELECTION)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN_CSV)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    p.add_argument("--mz-entry-model", type=Path, default=DEFAULT_MZ_ENTRY_MODEL)
    p.add_argument("--az-model", type=Path, default=DEFAULT_AZ_MODEL)
    p.add_argument("--az-risk-model", type=Path, default=DEFAULT_AZ_RISK_OUT)
    p.add_argument("--mz-risk-model", type=Path, default=DEFAULT_MZ_RISK_OUT)
    p.add_argument("--mz-exit-model", type=Path, default=DEFAULT_MZ_EXIT_MODEL)
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    p.add_argument("--validation-start", default="2025-11-01")
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--max-notional", type=float, default=None)
    p.add_argument("--leverage-cap", type=float, default=5.0)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seq-len", type=int, default=24)
    p.add_argument("--horizon", type=int, default=144)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--d-model", type=int, default=96)
    p.add_argument("--max-train-samples", type=int, default=30000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--exit-samples", type=int, default=80000)
    p.add_argument("--mdd-weight", type=float, default=3.0)
    p.add_argument("--stage2-gamma", type=float, default=0.55)
    p.add_argument("--stage2-prior", type=float, default=0.0)
    p.add_argument("--stage2-depth", type=int, default=1)
    p.add_argument("--stage2-score-floor", type=float, default=0.12)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    warnings.filterwarnings("ignore", category=UserWarning)
    device = "cuda" if args.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    torch.manual_seed(int(args.seed))

    policy = joblib.load(args.policy)
    entry_cfg, risk_cfg, exit_cfg = _load_selected(args.selection_report)
    max_notional = float(args.max_notional if args.max_notional is not None else risk_cfg.get("max_notional", entry_cfg.get("max_notional", 3.6)))
    entry_cfg = dict(entry_cfg)
    risk_cfg = dict(risk_cfg)
    exit_cfg = dict(exit_cfg)
    entry_cfg["max_notional"] = max_notional
    risk_cfg["max_notional"] = max_notional

    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    split_ts = pd.Timestamp(args.validation_start)
    ts = pd.to_datetime(train_all["timestamp"], errors="coerce") if "timestamp" in train_all.columns else pd.Series(np.arange(len(train_all)))
    train_df = train_all.loc[ts < split_ts].reset_index(drop=True)
    val_df = train_all.loc[ts >= split_ts].reset_index(drop=True)

    mz_entry = __import__("scripts.train_eval_zero_style_risk_overlay_2026", fromlist=["_load_mz_entry"])._load_mz_entry(args.mz_entry_model, device)
    az_risk = _load_pv(args.az_risk_model, len(RISK_ACTIONS), RISK_ACTIONS, device)
    mz_risk = _load_mz_risk(args.mz_risk_model, device)
    az_exit = _load_az_exit(args.az_model, device)
    if az_exit is None:
        raise FileNotFoundError(f"AZ exit model not found: {args.az_model}")
    # Load once to fail early when the MuZero exit artifact is missing; this comparison uses AZ threshold 0.45.
    _ = _load_mz_exit(args.mz_exit_model, device)

    val_zero_pre = _build_zero_style_current(
        val_df,
        policy,
        entry_cfg,
        mz_entry=mz_entry,
        az_risk=az_risk,
        mz_risk=mz_risk,
        device=device,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
        stage2_gamma=float(args.stage2_gamma),
        stage2_prior=float(args.stage2_prior),
        stage2_depth=int(args.stage2_depth),
        stage2_score_floor=float(args.stage2_score_floor),
    )
    eval_zero_pre = _build_zero_style_current(
        eval_df,
        policy,
        entry_cfg,
        mz_entry=mz_entry,
        az_risk=az_risk,
        mz_risk=mz_risk,
        device=device,
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
        stage2_gamma=float(args.stage2_gamma),
        stage2_prior=float(args.stage2_prior),
        stage2_depth=int(args.stage2_depth),
        stage2_score_floor=float(args.stage2_score_floor),
    )
    zero_exit_cfg = {"exit_threshold": 0.45, "min_exit_age": int(exit_cfg["min_exit_age"])}
    zero_val = _run("current_muzero_az_val", val_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, val_zero_pre, fee=args.fee, slip=args.slip, mdd_weight=args.mdd_weight)
    zero_eval = _run("current_muzero_az_eval", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_zero_pre, fee=args.fee, slip=args.slip, monthly=True, mdd_weight=args.mdd_weight)

    cfg = DTLifecycleConfig(
        seq_len=int(args.seq_len),
        horizon=int(args.horizon),
        d_model=int(args.d_model),
        batch_size=int(args.batch_size),
        epochs=int(args.epochs),
        max_train_samples=int(args.max_train_samples),
        seed=int(args.seed),
    )
    train_feat, train_template_dec, _, _ = _base_frame(train_df, policy, entry_cfg)
    val_feat, val_template_dec, val_close, val_fill = _base_frame(val_df, policy, entry_cfg)
    eval_feat, eval_template_dec, eval_close, eval_fill = _base_frame(eval_df, policy, entry_cfg)
    train_x = train_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    val_x = val_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    eval_x = eval_feat.reindex(columns=FEATURE_COLS).replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    train_z, mean, std = _standardize_fit(train_x)
    val_z = _standardize_apply(val_x, mean, std)
    eval_z = _standardize_apply(eval_x, mean, std)

    q_train, cvar_train, labels_train, cond_train, target_meta = _future_targets(
        train_df,
        horizon=int(cfg.horizon),
        fee=float(args.fee),
        slip=float(args.slip),
        max_notional=max_notional,
        cvar_alpha=float(cfg.cvar_alpha),
        adverse_penalty=float(cfg.adverse_penalty),
        cvar_penalty=float(cfg.cvar_penalty),
        min_train_edge=float(cfg.min_train_edge),
    )
    dt_net, dt_meta = _train_dt(train_z, labels_train, cond_train, cfg, device=device)
    critic_net, critic_meta = _train_critic(train_z, q_train, cvar_train, labels_train, cfg, device=device)

    x_exit, y_exit, exit_sample_meta = collect_exit_samples(
        train_df,
        policy,
        entry_config=entry_cfg,
        fee=float(args.fee),
        slip=float(args.slip),
        entry_stride=24,
        min_age=3,
        max_age=288,
        age_stride=12,
        future_horizon=int(args.horizon),
        exit_edge=0.0015,
        adverse_gap=0.012,
        max_samples=int(args.exit_samples),
        seed=int(args.seed),
    )
    exit_model = ExitProbabilityModel(train_exit_model(x_exit, y_exit, seed=int(args.seed)))

    val_probs = _predict_dt(dt_net, val_z, np.zeros((len(val_z), 3), dtype=np.float32), cfg, device=device, batch_size=int(args.batch_size))
    eval_probs = _predict_dt(dt_net, eval_z, np.zeros((len(eval_z), 3), dtype=np.float32), cfg, device=device, batch_size=int(args.batch_size))
    val_q, val_cvar = _predict_critic(critic_net, val_z, device=device, batch_size=int(args.batch_size))
    eval_q, eval_cvar = _predict_critic(critic_net, eval_z, device=device, batch_size=int(args.batch_size))

    grid: list[dict[str, Any]] = []
    for min_lower_edge in (-0.0060, -0.0020, 0.0000, 0.0006, 0.0012, 0.0024):
        for max_cvar_loss in (0.012, 0.020, 0.035, 0.060, 0.100):
            val_dec = _candidate_decisions(
                val_template_dec,
                val_probs,
                val_q,
                val_cvar,
                cfg,
                min_lower_edge=min_lower_edge,
                max_cvar_loss=max_cvar_loss,
                max_notional=max_notional,
                leverage_cap=float(args.leverage_cap),
            )
            val_pre = (val_feat, val_dec, val_close, val_fill)
            for exit_threshold in (0.10, 0.20, 0.35, 0.45, 0.55, 0.65):
                cand_exit_cfg = {"exit_threshold": float(exit_threshold), "min_exit_age": int(exit_cfg["min_exit_age"])}
                row = _run(
                    f"dt_iql_cvar_edge{min_lower_edge:.4f}_cvar{max_cvar_loss:.3f}_exit{exit_threshold:.2f}_val",
                    val_df,
                    policy,
                    exit_model,
                    entry_cfg,
                    risk_cfg,
                    cand_exit_cfg,
                    val_pre,
                    fee=float(args.fee),
                    slip=float(args.slip),
                    mdd_weight=float(args.mdd_weight),
                )
                row["config"] = {
                    "min_lower_edge": float(min_lower_edge),
                    "max_cvar_loss": float(max_cvar_loss),
                    "exit_threshold": float(exit_threshold),
                }
                grid.append(row)
    selected = sorted(grid, key=lambda r: float(r["score"]), reverse=True)[0]
    selected_cfg = dict(selected["config"])
    eval_dec = _candidate_decisions(
        eval_template_dec,
        eval_probs,
        eval_q,
        eval_cvar,
        cfg,
        min_lower_edge=float(selected_cfg["min_lower_edge"]),
        max_cvar_loss=float(selected_cfg["max_cvar_loss"]),
        max_notional=max_notional,
        leverage_cap=float(args.leverage_cap),
    )
    eval_pre = (eval_feat, eval_dec, eval_close, eval_fill)
    cand_exit_cfg = {"exit_threshold": float(selected_cfg["exit_threshold"]), "min_exit_age": int(exit_cfg["min_exit_age"])}
    candidate_eval = _run("dt_lifecycle_iql_cql_cvar_eval", eval_df, policy, exit_model, entry_cfg, risk_cfg, cand_exit_cfg, eval_pre, fee=args.fee, slip=args.slip, monthly=True, mdd_weight=args.mdd_weight)

    cost_stress: dict[str, list[dict[str, Any]]] = {}
    for mult in (1.0, 2.0, 3.0):
        cost_stress[f"cost_{mult:g}x"] = [
            _run("current_muzero_az", eval_df, policy, az_exit, entry_cfg, risk_cfg, zero_exit_cfg, eval_zero_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult, mdd_weight=args.mdd_weight),
            _run("dt_lifecycle_iql_cql_cvar", eval_df, policy, exit_model, entry_cfg, risk_cfg, cand_exit_cfg, eval_pre, fee=float(args.fee) * mult, slip=float(args.slip) * mult, mdd_weight=args.mdd_weight),
        ]

    args.model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "type": "dt_lifecycle_policy_v0",
            "state_dict": dt_net.state_dict(),
            "feature_cols": list(FEATURE_COLS),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "config": asdict(cfg),
            "train_meta": dt_meta,
            "target_meta": target_meta,
        },
        args.model_dir / "dt_lifecycle_policy.pt",
    )
    torch.save(
        {
            "type": "iql_cql_cvar_critic_v0",
            "state_dict": critic_net.state_dict(),
            "feature_cols": list(FEATURE_COLS),
            "mean": mean.astype(np.float32),
            "std": std.astype(np.float32),
            "config": asdict(cfg),
            "train_meta": critic_meta,
        },
        args.model_dir / "iql_cql_cvar_critic.pt",
    )
    joblib.dump(
        {
            "type": "dt_lifecycle_exit_governor_v0",
            "model": exit_model.model,
            "sample_meta": exit_sample_meta,
        },
        args.model_dir / "dt_lifecycle_exit_governor.pkl",
    )

    report = {
        "type": "dt_lifecycle_vs_muzero_az_2026",
        "note": "New candidate is isolated from existing MuZero/AZ artifacts. It uses a small Decision-Transformer-style lifecycle policy plus conservative IQL/CQL-inspired critic and CVaR tail gate. This is an experimental harness, not a live promotion.",
        "policy": str(args.policy),
        "train_csv": str(args.train_csv),
        "eval_csv": str(args.eval_csv),
        "model_dir": str(args.model_dir),
        "audit": {
            "source_audit": _audit(args.train_csv, args.eval_csv, policy),
            "train_range": _date_range(train_df),
            "validation_range": _date_range(val_df),
            "eval_range": _date_range(eval_df),
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "eval_rows": int(len(eval_df)),
        },
        "cost_and_caps": {
            "fee": float(args.fee),
            "slip": float(args.slip),
            "max_notional": float(max_notional),
            "leverage_cap": float(args.leverage_cap),
            "risk_config_max_notional": float(risk_cfg.get("max_notional", max_notional)),
        },
        "zero_style_current_config": {
            "entry": "MuZero entry planner",
            "risk": "AZ risk overlay",
            "stage2": {
                "model": "MuZero sleeve overlay",
                "gamma": float(args.stage2_gamma),
                "prior": float(args.stage2_prior),
                "depth": int(args.stage2_depth),
                "score_floor": float(args.stage2_score_floor),
            },
            "exit": {"model": "AZ exit governor", "threshold": 0.45},
        },
        "candidate_config": {
            "dt_lifecycle": asdict(cfg),
            "selected_gate": selected_cfg,
        },
        "target_meta": target_meta,
        "train_meta": {
            "dt": dt_meta,
            "critic": critic_meta,
            "exit": exit_sample_meta,
        },
        "validation": {
            "current_muzero_az": zero_val,
            "candidate_grid_ranked": sorted(grid, key=lambda r: float(r["score"]), reverse=True)[:20],
            "selected_candidate": selected,
        },
        "eval": {
            "current_muzero_az": zero_eval,
            "dt_lifecycle_iql_cql_cvar": candidate_eval,
            "delta": {
                "pnl": float(candidate_eval["eval"]["pnl"] - zero_eval["eval"]["pnl"]),
                "mdd": float(candidate_eval["eval"]["mdd"] - zero_eval["eval"]["mdd"]),
                "trades": int(candidate_eval["eval"]["trades"] - zero_eval["eval"]["trades"]),
                "trades_per_day": float(candidate_eval["eval"]["trades_per_day"] - zero_eval["eval"]["trades_per_day"]),
            },
        },
        "cost_stress": cost_stress,
        "red_team_required": [
            "This candidate uses future-window labels for training and must pass OOF/embargo leakage audit before promotion.",
            "Backtest uses same no-limit accounting path, but funding and liquidation proximity are still approximations.",
            "Leverage cap is enforced on decision records; PnL path is primarily notional-driven in this accounting engine.",
            "Run weekly/monthly walk-forward before considering any live shadow deployment.",
        ],
    }
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "model_dir": str(args.model_dir),
                "current": zero_eval["eval"],
                "candidate": candidate_eval["eval"],
                "delta": report["eval"]["delta"],
                "selected_gate": selected_cfg,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
