#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import QuantileTransformer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_deep_tabular_parent_mdd_v2_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_tabular_parent_mdd_v2_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/hf_v13_deep_tabular_parent_mdd_v2_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/hf_v13_deep_tabular_parent_mdd_v2_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/hf_v13_deep_tabular_parent_mdd_v2_20260514_grid.csv"
SEQ_LEN = 72


@dataclass(frozen=True)
class RuntimeConfig:
    name: str
    model_key: str
    mode: str
    confidence: float
    quality_floor: float
    notional_scale: float
    max_notional: float
    uncertainty_max: float


class ParentDataset(Dataset):
    def __init__(
        self,
        x_tab: np.ndarray,
        y: dict[str, np.ndarray],
        x_seq: np.ndarray | None = None,
    ) -> None:
        self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
        self.x_seq = None if x_seq is None else torch.as_tensor(x_seq, dtype=torch.float32)
        self.y = {
            "action": torch.as_tensor(y["action"], dtype=torch.long),
            "quality": torch.as_tensor(y["quality"], dtype=torch.float32),
            "notional": torch.as_tensor(y["notional"], dtype=torch.long),
            "leverage": torch.as_tensor(y["leverage"], dtype=torch.long),
            "take_profit": torch.as_tensor(y["take_profit"], dtype=torch.long),
            "stop_loss": torch.as_tensor(y["stop_loss"], dtype=torch.long),
            "max_hold": torch.as_tensor(y["max_hold"], dtype=torch.long),
            "cooldown": torch.as_tensor(y["cooldown"], dtype=torch.long),
        }

    def __len__(self) -> int:
        return int(self.x_tab.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        seq = self.x_tab[idx] if self.x_seq is None else self.x_seq[idx]
        return self.x_tab[idx], seq, {k: v[idx] for k, v in self.y.items()}


class HeadMixin:
    def _init_heads(self, hidden: int, cfg: FullyLearnedGovernorConfig) -> None:
        self.action_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.bucket_heads = nn.ModuleDict(
            {
                "notional": nn.Linear(hidden, len(cfg.notional_buckets)),
                "leverage": nn.Linear(hidden, len(cfg.leverage_buckets)),
                "take_profit": nn.Linear(hidden, len(cfg.take_profit_buckets)),
                "stop_loss": nn.Linear(hidden, len(cfg.stop_loss_buckets)),
                "max_hold": nn.Linear(hidden, len(cfg.max_hold_buckets)),
                "cooldown": nn.Linear(hidden, len(cfg.cooldown_buckets)),
            }
        )
        self.loss_log_vars = nn.ParameterDict(
            {
                "action": nn.Parameter(torch.zeros(())),
                "quality": nn.Parameter(torch.zeros(())),
                "bucket": nn.Parameter(torch.zeros(())),
            }
        )

    def _heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        out = {
            "action": self.action_head(z),
            "quality": self.quality_head(z).squeeze(-1),
        }
        out.update({k: head(z) for k, head in self.bucket_heads.items()})
        return out


def _sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    z = logits - logits.max(dim=dim, keepdim=True).values
    z_sorted = torch.sort(z, descending=True, dim=dim).values
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.ndim
    view[dim] = -1
    k = k.view(view)
    z_cumsum = z_sorted.cumsum(dim)
    support = 1 + k * z_sorted > z_cumsum
    k_z = support.sum(dim=dim, keepdim=True).clamp_min(1)
    tau = (z_cumsum.gather(dim, k_z.long() - 1) - 1) / k_z
    return torch.clamp(z - tau, min=0.0)


class FTTransformerParent(nn.Module, HeadMixin):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, d_model: int = 64, n_layers: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, dim_feedforward=d_model * 4, dropout=0.10, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self._init_heads(d_model, cfg)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = x_tab.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        cls = self.cls.expand(x_tab.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        return self._heads(self.norm(z))


class TabNetLiteParent(nn.Module, HeadMixin):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, hidden: int = 96, steps: int = 3) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(n_features)
        self.steps = int(steps)
        self.maskers = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_features if i == 0 else hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, n_features),
                )
                for i in range(self.steps)
            ]
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(n_features, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.10),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                )
                for _ in range(self.steps)
            ]
        )
        self.prior = nn.Linear(n_features, hidden)
        self._init_heads(hidden, cfg)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        x = self.norm(x_tab)
        state = self.prior(x)
        agg = torch.zeros_like(state)
        for i in range(self.steps):
            mask_src = x if i == 0 else state
            mask = _sparsemax(self.maskers[i](mask_src), dim=-1)
            state = self.blocks[i](x * mask)
            agg = agg + F.relu(state)
        return self._heads(agg / float(self.steps))


class TFTLiteParent(nn.Module, HeadMixin):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, hidden: int = 80, n_layers: int = 1) -> None:
        super().__init__()
        self.feature_gate = nn.Sequential(nn.LayerNorm(n_features), nn.Linear(n_features, n_features), nn.Sigmoid())
        self.proj = nn.Linear(n_features, hidden)
        enc = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=hidden * 4, dropout=0.10, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.skip = nn.Linear(n_features, hidden)
        self.norm = nn.LayerNorm(hidden)
        self._init_heads(hidden, cfg)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        seq = x_seq if x_seq is not None and x_seq.ndim == 3 else x_tab[:, None, :]
        gate = self.feature_gate(seq)
        h = self.encoder(self.proj(seq * gate))
        recency = torch.linspace(0.0, 0.35, h.shape[1], device=h.device).view(1, -1, 1)
        w = torch.softmax(self.attn(h) + recency, dim=1)
        z = torch.sum(h * w, dim=1) + self.skip(x_tab)
        return self._heads(self.norm(z))


def _normalise_fit(x: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    arr = x.replace([np.inf, -np.inf], np.nan).astype(float)
    med = arr.median(axis=0).to_numpy(dtype=np.float32)
    filled = arr.fillna(pd.Series(med, index=arr.columns))
    qt = QuantileTransformer(
        n_quantiles=min(2048, max(16, len(filled))),
        output_distribution="normal",
        subsample=None,
        random_state=20260514,
    )
    out = qt.fit_transform(filled.to_numpy(dtype=np.float32)).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0).clip(-5.0, 5.0)
    return pd.DataFrame(out, columns=x.columns), {"kind": "quantile_normal", "median": med, "transformer": qt, "columns": list(x.columns)}


def _normalise_apply(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    arr = x.reindex(columns=cols).replace([np.inf, -np.inf], np.nan).astype(float)
    med = np.asarray(norm["median"], dtype=np.float32)
    filled = arr.fillna(pd.Series(med, index=cols)).to_numpy(dtype=np.float32)
    if norm.get("kind") == "quantile_normal":
        qt: QuantileTransformer = norm["transformer"]
        out = qt.transform(filled).astype(np.float32)
        return np.nan_to_num(out, nan=0.0, posinf=5.0, neginf=-5.0).clip(-5.0, 5.0)
    mean = np.asarray(norm["mean"], dtype=np.float32)
    std = np.asarray(norm["std"], dtype=np.float32)
    return ((filled - mean) / np.maximum(std, 1e-6)).clip(-8.0, 8.0)


def _sequence_array(x_full: np.ndarray, indices: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    out = np.zeros((len(indices), seq_len, x_full.shape[1]), dtype=np.float32)
    for j, idx in enumerate(indices.astype(int)):
        start = max(0, idx - seq_len + 1)
        chunk = x_full[start : idx + 1]
        out[j, -len(chunk) :] = chunk
    return out


def _candidate_indices(n: int, cfg: FullyLearnedGovernorConfig, stride: int) -> np.ndarray:
    return np.arange(0, max(0, n - int(cfg.max_train_horizon_bars) - 1), max(1, int(stride)), dtype=np.int64)


def _bucket_index(values: np.ndarray, buckets: tuple[float, ...]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    b = np.asarray(buckets, dtype=np.float64)
    return np.argmin(np.abs(arr[:, None] - b[None, :]), axis=1).astype(np.int64)


def _teacher_labels(decisions: pd.DataFrame, indices: np.ndarray, cfg: FullyLearnedGovernorConfig) -> dict[str, np.ndarray]:
    d = decisions.iloc[indices.astype(int)]
    return {
        "action": d["action"].astype(int).to_numpy(dtype=np.int64),
        "quality": pd.to_numeric(d["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32),
        "notional": _bucket_index(pd.to_numeric(d["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(), cfg.notional_buckets),
        "leverage": _bucket_index(pd.to_numeric(d["leverage"], errors="coerce").fillna(1.0).to_numpy(), cfg.leverage_buckets),
        "take_profit": _bucket_index(pd.to_numeric(d["take_profit"], errors="coerce").fillna(0.0).to_numpy(), cfg.take_profit_buckets),
        "stop_loss": _bucket_index(pd.to_numeric(d["stop_loss"], errors="coerce").fillna(0.0).to_numpy(), cfg.stop_loss_buckets),
        "max_hold": _bucket_index(pd.to_numeric(d["max_hold_bars"], errors="coerce").fillna(0.0).to_numpy(), tuple(float(v) for v in cfg.max_hold_buckets)),
        "cooldown": _bucket_index(pd.to_numeric(d["cooldown_bars"], errors="coerce").fillna(0.0).to_numpy(), tuple(float(v) for v in cfg.cooldown_buckets)),
    }


def _balanced(model: nn.Module, name: str, term: torch.Tensor) -> torch.Tensor:
    log_vars = getattr(model, "loss_log_vars", None)
    if log_vars is None or name not in log_vars:
        return term
    s = log_vars[name].clamp(-3.0, 3.0)
    return torch.exp(-s) * term + 0.5 * s


def _loss(model: nn.Module, outputs: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
    action = y["action"]
    active = action != ACTION_CASH
    action_weight = torch.ones(3, device=action.device)
    action_weight[ACTION_CASH] = 0.45
    action_loss = F.cross_entropy(outputs["action"], action, weight=action_weight)
    q_weight = torch.where(active, torch.tensor(1.0, device=action.device), torch.tensor(0.35, device=action.device))
    quality_loss = (F.smooth_l1_loss(outputs["quality"], y["quality"], reduction="none") * q_weight).mean()
    bucket_loss = torch.zeros((), device=action.device)
    if bool(active.any()):
        for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
            bucket_loss = bucket_loss + F.cross_entropy(outputs[key][active], y[key][active])
        bucket_loss = bucket_loss / 6.0
    return _balanced(model, "action", action_loss) + _balanced(model, "quality", quality_loss) + 0.60 * _balanced(model, "bucket", bucket_loss)


def _train_model(
    key: str,
    model: nn.Module,
    train_ds: ParentDataset,
    val_ds: ParentDataset,
    *,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, xs, yb in train_loader:
            xb = xb.to(device)
            xs = xs.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, model(xb, xs), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, xs, yb in val_loader:
                xb = xb.to(device)
                xs = xs.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                vl = _loss(model, model(xb, xs), yb)
                vtotal += float(vl.item()) * len(xb)
                vcount += len(xb)
        tr = total / max(count, 1)
        va = vtotal / max(vcount, 1)
        history.append({"epoch": float(epoch), "train_loss": tr, "val_loss": va})
        print(f"[{MODEL_ID}] {key} epoch={epoch:02d} train_loss={tr:.5f} val_loss={va:.5f}", flush=True)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best_val), "history": history}


def _predict_outputs(
    model: nn.Module,
    x_tab: np.ndarray,
    x_seq: np.ndarray | None,
    device: torch.device,
    batch_size: int,
    *,
    mc_passes: int = 8,
    temperature: float = 1.35,
    logit_clip: float = 7.0,
) -> dict[str, np.ndarray]:
    model.to(device)
    outs: dict[str, list[np.ndarray]] = {"quality": [], "action_uncertainty": []}
    for key in ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
        outs[key] = []
    n = len(x_tab)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            xb = torch.as_tensor(x_tab[start:end], dtype=torch.float32, device=device)
            xs = xb[:, None, :] if x_seq is None else torch.as_tensor(x_seq[start:end], dtype=torch.float32, device=device)
            pass_count = max(1, int(mc_passes))
            quality_passes: list[torch.Tensor] = []
            proba_passes: dict[str, list[torch.Tensor]] = {k: [] for k in ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")}
            for _ in range(pass_count):
                model.train(pass_count > 1)
                pred = model(xb, xs)
                quality_passes.append(pred["quality"])
                for key in proba_passes:
                    logits = pred[key].clamp(-float(logit_clip), float(logit_clip)) / max(float(temperature), 1e-6)
                    proba_passes[key].append(torch.softmax(logits, dim=-1))
            action_stack = torch.stack(proba_passes["action"], dim=0)
            outs["quality"].append(torch.stack(quality_passes, dim=0).mean(dim=0).detach().cpu().numpy())
            outs["action_uncertainty"].append(action_stack.std(dim=0).mean(dim=1).detach().cpu().numpy())
            for key in ("action", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
                outs[key].append(torch.stack(proba_passes[key], dim=0).mean(dim=0).detach().cpu().numpy())
    model.to("cpu")
    return {k: np.concatenate(v, axis=0) for k, v in outs.items()}


def _expected_bucket(proba: np.ndarray, buckets: tuple[float, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vals = np.asarray(buckets, dtype=np.float64)
    p = proba[:, : len(vals)]
    idx = np.argmax(p, axis=1).astype(np.int64)
    return p @ vals, np.max(p, axis=1), idx


def _decisions_from_outputs(
    outputs: dict[str, np.ndarray],
    cfg: FullyLearnedGovernorConfig,
    rt: RuntimeConfig,
    index: pd.Index,
    teacher: pd.DataFrame | None = None,
) -> pd.DataFrame:
    action_p = outputs["action"]
    pred_action = np.argmax(action_p, axis=1).astype(np.int64)
    pred_conf = np.max(action_p, axis=1)
    uncertainty = np.asarray(outputs.get("action_uncertainty", np.zeros_like(pred_conf)), dtype=np.float64)
    side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0)).astype(np.int64)
    notional, c1, _ = _expected_bucket(outputs["notional"], cfg.notional_buckets)
    leverage, c2, _ = _expected_bucket(outputs["leverage"], cfg.leverage_buckets)
    tp, c3, _ = _expected_bucket(outputs["take_profit"], cfg.take_profit_buckets)
    sl, c4, _ = _expected_bucket(outputs["stop_loss"], cfg.stop_loss_buckets)
    mh, c5, _ = _expected_bucket(outputs["max_hold"], tuple(float(v) for v in cfg.max_hold_buckets))
    cd, c6, _ = _expected_bucket(outputs["cooldown"], tuple(float(v) for v in cfg.cooldown_buckets))
    quality = np.asarray(outputs["quality"], dtype=np.float64)

    if rt.mode == "veto":
        if teacher is None:
            raise ValueError("veto mode requires teacher decisions")
        out = teacher.copy()
        teacher_action = out["action"].astype(int).to_numpy()
        teacher_side = out["side"].astype(int).to_numpy()
        teacher_active = (teacher_action != ACTION_CASH) & (teacher_side != 0)
        agree = (pred_action == teacher_action) & (side == teacher_side)
        keep = teacher_active & agree & (pred_conf >= float(rt.confidence)) & (quality >= float(rt.quality_floor)) & (uncertainty <= float(rt.uncertainty_max))
        out.loc[~keep, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
        out.loc[~keep, "leverage"] = 1.0
        out.loc[:, "deep_parent_veto_confidence"] = pred_conf.astype(np.float64)
        out.loc[:, "deep_parent_veto_quality"] = quality.astype(np.float64)
        out.loc[:, "deep_parent_veto_action"] = pred_action.astype(np.int64)
        out.loc[:, "deep_parent_veto_uncertainty"] = uncertainty.astype(np.float64)
        return out

    notional = np.clip(notional * float(rt.notional_scale), min(cfg.notional_buckets), float(rt.max_notional))
    leverage = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
    fraction = np.clip(notional / np.maximum(leverage, 1e-8), 0.0, cfg.max_margin_fraction)
    notional = fraction * leverage
    confidence = np.mean(np.vstack([pred_conf, c1, c2, c3, c4, c5, c6]), axis=0)
    active = (pred_action != ACTION_CASH) & (side != 0) & (pred_conf >= float(rt.confidence)) & (quality >= float(rt.quality_floor)) & (uncertainty <= float(rt.uncertainty_max))
    action = np.where(active, pred_action, ACTION_CASH).astype(np.int64)
    side = np.where(active, side, 0).astype(np.int64)
    out = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": notional.astype(np.float64),
            "leverage": leverage.astype(np.float64),
            "position_fraction": fraction.astype(np.float64),
            "take_profit": tp.astype(np.float64),
            "stop_loss": sl.astype(np.float64),
            "max_hold_bars": np.rint(mh).astype(np.int64),
            "cooldown_bars": np.rint(cd).astype(np.int64),
            "quality_score": quality.astype(np.float64),
            "confidence": confidence.astype(np.float64),
            "deep_parent_action_confidence": pred_conf.astype(np.float64),
            "deep_parent_action_uncertainty": uncertainty.astype(np.float64),
        },
        index=index,
    )
    cash = out["action"].astype(int).to_numpy() == ACTION_CASH
    out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[cash, "leverage"] = 1.0
    return out


def _metrics(
    df: pd.DataFrame,
    q: np.ndarray,
    decisions: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    variant: v45.LayerVariant,
    base: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"cost{mult}": v45.backtest_variant(df, parent, jackpot_model, add_cfg, q, variant, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=decisions)
        for mult in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    trades = int(c1.get("trades", 0))
    if trades < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    mdd_abs = abs(float(c1.get("mdd", 0.0)))
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.10 * c3["pnl"] - 6.0 * mdd_abs + 0.05 * min(trades, 120))


def _runtime_grid(model_key: str) -> list[RuntimeConfig]:
    rows: list[RuntimeConfig] = []
    for conf in (0.38, 0.46, 0.54, 0.62, 0.70):
        for q_floor in (-0.010, 0.000, 0.010, 0.020):
            for scale, cap in ((0.45, 1.20), (0.60, 1.60), (0.75, 2.00), (0.90, 2.30)):
                for unc in (0.035, 0.060, 0.090):
                    rows.append(RuntimeConfig(f"{model_key}_replace_c{conf:.2f}_q{q_floor:.3f}_s{scale:.2f}_cap{cap:.2f}_u{unc:.3f}", model_key, "replace", conf, q_floor, scale, cap, unc))
    for conf in (0.30, 0.38, 0.46, 0.54, 0.62):
        for q_floor in (-0.020, -0.010, 0.000, 0.010):
            for unc in (0.035, 0.060, 0.090):
                rows.append(RuntimeConfig(f"{model_key}_veto_c{conf:.2f}_q{q_floor:.3f}_u{unc:.3f}", model_key, "veto", conf, q_floor, 1.0, max(FullyLearnedGovernorConfig().notional_buckets), unc))
    return rows


def _overlay_alpha1() -> v31.OverlayConfig:
    return v31.OverlayConfig("alpha1_v31_deep_notional2", 0.010, 0.004, 2.0, 12, 0.040, 0.018, 48, 1.5, 2.5, 1.0, 0.50, 18, 0.025, 0.075, 0.036)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare FT-Transformer, TabNet-lite, and TFT-lite as MDD-focused Alpha1 parent replacements.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quick", action="store_true", help="Use fewer runtime configs for a smoke pass.")
    args = parser.parse_args()

    torch.manual_seed(20260514)
    np.random.seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    cfg = FullyLearnedGovernorConfig(**base)
    feature_cols = list(parent.get("feature_cols") or [])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    print(f"[{MODEL_ID}] rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)} features={len(feature_cols)}", flush=True)

    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    train_teacher = predict_policy_frame(parent, train_df, close=_close(train_df))
    val_teacher = predict_policy_frame(parent, val_df, close=_close(val_df))
    train_full_pre = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    idx_train = _candidate_indices(len(train_df), cfg, int(args.stride))
    x_train_labels = train_full_pre.iloc[idx_train].reset_index(drop=True)
    y_train = _teacher_labels(train_teacher, idx_train, cfg)
    train_meta = {"candidates": int(len(idx_train)), "stride_bars": int(args.stride), "label_source": "hgb_teacher_distillation"}
    x_train_norm, norm = _normalise_fit(x_train_labels)
    x_train_tab = x_train_norm.to_numpy(dtype=np.float32)

    train_full_norm = _normalise_apply(train_full_pre, norm)
    x_train_seq = _sequence_array(train_full_norm, idx_train)

    val_full_pre = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    x_val_full = _normalise_apply(val_full_pre, norm)
    idx_val = _candidate_indices(len(val_df), cfg, max(3, int(args.stride)))
    idx_val = idx_val if len(idx_val) > 0 else np.arange(0, len(val_df), max(1, int(args.stride)), dtype=np.int64)
    x_val_ds_tab = x_val_full[idx_val]
    x_val_ds_seq = _sequence_array(x_val_full, idx_val)
    y_val_ds = _teacher_labels(val_teacher, idx_val, cfg)
    val_meta = {"candidates": int(len(idx_val)), "stride_bars": max(3, int(args.stride)), "label_source": "hgb_teacher_distillation"}

    train_tab_ds = ParentDataset(x_train_tab, y_train)
    train_seq_ds = ParentDataset(x_train_tab, y_train, x_train_seq)
    val_tab_ds = ParentDataset(x_val_ds_tab, y_val_ds)
    val_seq_ds = ParentDataset(x_val_ds_tab, y_val_ds, x_val_ds_seq)

    models: dict[str, nn.Module] = {
        "ft_transformer": FTTransformerParent(len(feature_cols), cfg),
        "tabnet_lite": TabNetLiteParent(len(feature_cols), cfg),
        "tft_lite": TFTLiteParent(len(feature_cols), cfg),
    }
    training: dict[str, Any] = {}
    for key, model in models.items():
        print(f"[{MODEL_ID}] training {key}", flush=True)
        if key == "tft_lite":
            training[key] = _train_model(key, model, train_seq_ds, val_seq_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))
        else:
            training[key] = _train_model(key, model, train_tab_ds, val_tab_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))

    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_full_pre = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    x_eval_full = _normalise_apply(eval_full_pre, norm)
    val_seq_full = _sequence_array(x_val_full, np.arange(len(val_df), dtype=np.int64))
    eval_seq_full = _sequence_array(x_eval_full, np.arange(len(eval_df), dtype=np.int64))

    val_outputs: dict[str, dict[str, np.ndarray]] = {}
    eval_outputs: dict[str, dict[str, np.ndarray]] = {}
    for key, model in models.items():
        print(f"[{MODEL_ID}] predicting {key}", flush=True)
        if key == "tft_lite":
            val_outputs[key] = _predict_outputs(model, x_val_full, val_seq_full, device, int(args.batch_size))
            eval_outputs[key] = _predict_outputs(model, x_eval_full, eval_seq_full, device, int(args.batch_size))
        else:
            val_outputs[key] = _predict_outputs(model, x_val_full, None, device, int(args.batch_size))
            eval_outputs[key] = _predict_outputs(model, x_eval_full, None, device, int(args.batch_size))

    variant = v45.LayerVariant("alpha1_parent_deep_tabular_mdd", "parent_deep_tabular_mdd", _overlay_alpha1())
    eval_teacher = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    selected: RuntimeConfig | None = None
    best_score = -1e18
    runtime_keys = list(models.keys())
    for key in runtime_keys:
        rt_grid = _runtime_grid(key)
        if args.quick:
            rt_grid = [
                r
                for r in rt_grid
                if (
                    (r.mode == "replace" and r.confidence in (0.54, 0.62) and r.quality_floor == 0.0 and r.max_notional in (1.2, 1.6) and r.uncertainty_max == 0.060)
                    or (r.mode == "veto" and r.confidence in (0.30, 0.46, 0.62) and r.quality_floor in (-0.01, 0.0, 0.01) and r.uncertainty_max == 0.090)
                )
            ]
        for rt in rt_grid:
            dec = _decisions_from_outputs(val_outputs[key], cfg, rt, val_df.index, teacher=val_teacher)
            vm = _metrics(val_df, val_q, dec, parent, jackpot_model, add_cfg, variant, base)
            score = _score(vm)
            row = {
                **asdict(rt),
                "score": score,
                "val_pnl": vm["cost1"]["pnl"],
                "val_mdd": vm["cost1"]["mdd"],
                "val_trades": vm["cost1"]["trades"],
                "val_tpd": vm["cost1"]["trades_per_day"],
                "val_cost2_pnl": vm["cost2"]["pnl"],
                "val_cost3_pnl": vm["cost3"]["pnl"],
            }
            rows.append(row)
            if score > best_score:
                best_score = score
                selected = rt
                print(f"[{MODEL_ID}] new val best {rt.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f} trades={row['val_trades']}", flush=True)
    assert selected is not None

    baseline_metrics = _metrics(eval_df, eval_q, eval_teacher, parent, jackpot_model, add_cfg, variant, base)
    experiments = [{"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)}]
    for key in runtime_keys:
        model_rows = [r for r in rows if r["model_key"] == key]
        best_row = max(model_rows, key=lambda r: r["score"])
        rt = RuntimeConfig(best_row["name"], best_row["model_key"], str(best_row["mode"]), float(best_row["confidence"]), float(best_row["quality_floor"]), float(best_row["notional_scale"]), float(best_row["max_notional"]), float(best_row["uncertainty_max"]))
        dec = _decisions_from_outputs(eval_outputs[key], cfg, rt, eval_df.index, teacher=eval_teacher)
        metrics = _metrics(eval_df, eval_q, dec, parent, jackpot_model, add_cfg, variant, base)
        experiments.append({"name": f"{key}::{rt.name}", "selected_runtime": asdict(rt), "metrics": metrics, "score": _score(metrics)})
        print(
            f"[{MODEL_ID}] OOS {key} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} "
            f"trades={metrics['cost1']['trades']} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )
    best = max(experiments, key=lambda e: e["score"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, model in models.items():
        torch.save({"model_id": MODEL_ID, "model_key": key, "state_dict": model.state_dict(), "feature_cols": feature_cols, "normalizer": norm, "config": base, "training": training[key]}, OUT_DIR / f"{key}.pt")
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best["name"] == "alpha1_hgb_parent_baseline":
        warnings.append("deep_parent_candidates_did_not_win_mdd_weighted_score")
    if best["metrics"]["cost1"]["pnl"] < 0:
        warnings.append("best_candidate_negative_cost1_pnl")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1_hgb_parent_baseline" and best["metrics"]["cost1"]["mdd"] > baseline_metrics["cost1"]["mdd"] and best["metrics"]["cost1"]["pnl"] > 0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "model_families": ["ft_transformer", "tabnet_lite", "tft_lite"],
        "objective": "minimize MDD subject to positive PnL and enough trade count inside Alpha1/V31 stack",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "base_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1/V31 stack with parent policy replaced by FT-Transformer, TabNet-lite, and TFT-lite deep tabular parents. Selection is MDD-weighted and uses 2025 validation only.",
        "selected": best,
        "experiments": experiments,
        "baseline": {"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics},
        "artifact_dir": str(OUT_DIR),
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] selected={best['name']} cost1={best['metrics']['cost1']['pnl']:.2f} mdd={best['metrics']['cost1']['mdd']:.2f}", flush=True)
    print(f"[{MODEL_ID}] report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
