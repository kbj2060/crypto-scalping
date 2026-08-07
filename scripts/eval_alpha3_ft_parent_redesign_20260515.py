#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
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
from torch.utils.data import DataLoader, Dataset

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
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_v2_20260515 as ft_v2  # noqa: E402
from scripts import eval_alpha3_gbdt_parent_full_retrain_20260515 as gbdt  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_alpha3_ft_transformer_mtl_parent_v2_20260515 import FeatureGRNTokenizer  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import _normalise_apply, _normalise_fit  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_ft_parent_redesign_20260515"
BASE_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_repro_20260511/v13_clean_regime_h288.pkl"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_ft_parent_redesign_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_parent_redesign_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_parent_redesign_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_ft_parent_redesign_20260515_grid.csv"
MODEL_OUT = OUT_DIR / "ft_grouped_hgb_surrogate.pt"
BASELINE_EXPECTED = {
    "cost1": {"pnl": 654.9174150098765, "mdd": -29.61731295277763, "trades": 195},
    "cost2": {"pnl": 602.2624624847589, "mdd": -30.093378120960466, "trades": 195},
    "cost3": {"pnl": 456.48201847894717, "mdd": -31.397871677089583, "trades": 198},
}


class DistillDataset(Dataset):
    def __init__(self, x: np.ndarray, y: dict[str, np.ndarray], soft: dict[str, np.ndarray]) -> None:
        self.x = torch.as_tensor(x, dtype=torch.float32)
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
        self.soft = {k: torch.as_tensor(v, dtype=torch.float32) for k, v in soft.items()}

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.x[idx], {k: v[idx] for k, v in self.y.items()}, {k: v[idx] for k, v in self.soft.items()}


class GroupedFTParent(nn.Module):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, d_model: int = 80, n_layers: int = 3) -> None:
        super().__init__()
        self.tokenizer = FeatureGRNTokenizer(n_features, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=4,
            dim_feedforward=d_model * 4,
            dropout=0.12,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.action_quality_tower = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.10))
        self.exposure_tower = nn.Sequential(nn.Linear(d_model + 3, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.10))
        self.exit_tower = nn.Sequential(nn.Linear(d_model + 3, d_model), nn.LayerNorm(d_model), nn.GELU(), nn.Dropout(0.10))
        self.action_head = nn.Linear(d_model, 3)
        self.quality_head = nn.Linear(d_model, 1)
        self.bucket_heads = nn.ModuleDict(
            {
                "notional": nn.Linear(d_model, len(cfg.notional_buckets)),
                "leverage": nn.Linear(d_model, len(cfg.leverage_buckets)),
                "take_profit": nn.Linear(d_model, len(cfg.take_profit_buckets)),
                "stop_loss": nn.Linear(d_model, len(cfg.stop_loss_buckets)),
                "max_hold": nn.Linear(d_model, len(cfg.max_hold_buckets)),
                "cooldown": nn.Linear(d_model, len(cfg.cooldown_buckets)),
            }
        )
        self.loss_log_vars = nn.ParameterDict(
            {name: nn.Parameter(torch.zeros(())) for name in ("action", "quality", "exposure", "exit", "distill")}
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x)
        cls = self.cls.expand(x.shape[0], -1, -1)
        z = self.norm(self.encoder(torch.cat([cls, tokens], dim=1))[:, 0])
        aq = self.action_quality_tower(z)
        action_logits = self.action_head(aq)
        action_prob = torch.softmax(action_logits.clamp(-8.0, 8.0), dim=-1)
        exp = self.exposure_tower(torch.cat([z, action_prob], dim=-1))
        ext = self.exit_tower(torch.cat([z, action_prob], dim=-1))
        return {
            "action": action_logits,
            "quality": self.quality_head(aq).squeeze(-1),
            "notional": self.bucket_heads["notional"](exp),
            "leverage": self.bucket_heads["leverage"](exp),
            "take_profit": self.bucket_heads["take_profit"](ext),
            "stop_loss": self.bucket_heads["stop_loss"](ext),
            "max_hold": self.bucket_heads["max_hold"](ext),
            "cooldown": self.bucket_heads["cooldown"](ext),
        }


def _balanced(model: nn.Module, name: str, term: torch.Tensor) -> torch.Tensor:
    s = model.loss_log_vars[name].clamp(-3.0, 3.0)
    return torch.exp(-s) * term + 0.5 * s


def _full_proba(model: Any, x: pd.DataFrame, n: int) -> np.ndarray:
    out = np.zeros((len(x), n), dtype=np.float32)
    if model is None:
        out[:, 0] = 1.0
        return out
    p = np.asarray(model.predict_proba(x), dtype=np.float32)
    classes = np.asarray(model.classes_, dtype=int)
    for j, c in enumerate(classes):
        if 0 <= int(c) < n:
            out[:, int(c)] = p[:, j]
    row_sum = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row_sum, 1e-8)


def _one_hot(values: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros((len(values), n), dtype=np.float32)
    out[np.arange(len(values)), np.clip(values.astype(int), 0, n - 1)] = 1.0
    return out


def _hgb_soft_targets(parent: dict[str, Any], x: pd.DataFrame, y: dict[str, np.ndarray], cfg: FullyLearnedGovernorConfig) -> dict[str, np.ndarray]:
    action_p = _full_proba(parent.get("action_model"), x, 3)
    action = np.argmax(action_p, axis=1)
    side = np.where(action == ACTION_LONG, 1.0, np.where(action == ACTION_SHORT, -1.0, 0.0))
    x_side = x.copy()
    if "side_hint" in x_side.columns:
        x_side["side_hint"] = side
    soft: dict[str, np.ndarray] = {
        "action": action_p,
        "quality": np.asarray(parent["quality_model"].predict(x), dtype=np.float32) if "quality_model" in parent else np.zeros(len(x), dtype=np.float32),
    }
    sizes = {
        "notional": len(cfg.notional_buckets),
        "leverage": len(cfg.leverage_buckets),
        "take_profit": len(cfg.take_profit_buckets),
        "stop_loss": len(cfg.stop_loss_buckets),
        "max_hold": len(cfg.max_hold_buckets),
        "cooldown": len(cfg.cooldown_buckets),
    }
    for key, n in sizes.items():
        model = parent.get(f"{key}_model")
        soft[key] = _full_proba(model, x_side, n) if model is not None else _one_hot(np.asarray(y[key]), n)
    return soft


def _loss(model: GroupedFTParent, out: dict[str, torch.Tensor], y: dict[str, torch.Tensor], soft: dict[str, torch.Tensor]) -> torch.Tensor:
    action = y["action"]
    active = action != ACTION_CASH
    action_weight = torch.ones(3, device=action.device)
    action_weight[ACTION_CASH] = 0.45
    supervised_action = F.cross_entropy(out["action"], action, weight=action_weight)
    distill_action = F.kl_div(F.log_softmax(out["action"] / 1.35, dim=-1), soft["action"], reduction="batchmean") * (1.35**2)
    q_weight = torch.where(active, torch.tensor(1.0, device=action.device), torch.tensor(0.35, device=action.device))
    quality = (F.smooth_l1_loss(out["quality"], y["quality"], reduction="none") * q_weight).mean()
    quality = quality + 0.50 * F.smooth_l1_loss(out["quality"], soft["quality"])
    exposure = torch.zeros((), device=action.device)
    exits = torch.zeros((), device=action.device)
    distill = distill_action
    if bool(active.any()):
        for key in ("notional", "leverage"):
            exposure = exposure + F.cross_entropy(out[key][active], y[key][active])
            distill = distill + 0.35 * F.kl_div(F.log_softmax(out[key][active] / 1.35, dim=-1), soft[key][active], reduction="batchmean") * (1.35**2)
        exposure = exposure / 2.0
        for key in ("take_profit", "stop_loss", "max_hold", "cooldown"):
            exits = exits + F.cross_entropy(out[key][active], y[key][active])
            distill = distill + 0.35 * F.kl_div(F.log_softmax(out[key][active] / 1.35, dim=-1), soft[key][active], reduction="batchmean") * (1.35**2)
        exits = exits / 4.0
    return (
        _balanced(model, "action", supervised_action)
        + _balanced(model, "quality", quality)
        + 0.70 * _balanced(model, "exposure", exposure)
        + 0.70 * _balanced(model, "exit", exits)
        + 0.60 * _balanced(model, "distill", distill)
    )


def _train_model(model: GroupedFTParent, train_ds: DistillDataset, val_ds: DistillDataset, *, epochs: int, batch_size: int, device: torch.device) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1.5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb, sb in train_loader:
            xb = xb.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            sb = {k: v.to(device) for k, v in sb.items()}
            opt.zero_grad(set_to_none=True)
            loss = _loss(model, model(xb), yb, sb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, yb, sb in val_loader:
                xb = xb.to(device)
                yb = {k: v.to(device) for k, v in yb.items()}
                sb = {k: v.to(device) for k, v in sb.items()}
                vl = _loss(model, model(xb), yb, sb)
                vtotal += float(vl.item()) * len(xb)
                vcount += len(xb)
        tr = total / max(count, 1)
        va = vtotal / max(vcount, 1)
        scheduler.step(va)
        lr = float(opt.param_groups[0]["lr"])
        history.append({"epoch": float(epoch), "train_loss": tr, "val_loss": va, "lr": lr})
        print(f"[{MODEL_ID}] epoch={epoch:02d} train_loss={tr:.5f} val_loss={va:.5f} lr={lr:.2e}", flush=True)
        if va < best_val - 1e-5:
            best_val = va
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= 8:
            print(f"[{MODEL_ID}] early_stop epoch={epoch} best_val={best_val:.5f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best_val), "history": history}


def _predict_head(model: GroupedFTParent, x: pd.DataFrame, norm: dict[str, Any], head: str, device: torch.device, batch_size: int) -> np.ndarray:
    arr = _normalise_apply(x, norm)
    model.to(device)
    model.eval()
    outs: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(arr), batch_size):
            xb = torch.as_tensor(arr[start : start + batch_size], dtype=torch.float32, device=device)
            pred = model(xb)[head]
            if head == "quality":
                outs.append(pred.detach().cpu().numpy().astype(np.float64))
            else:
                outs.append(torch.softmax(pred.clamp(-8.0, 8.0) / 1.15, dim=-1).detach().cpu().numpy().astype(np.float64))
    model.to("cpu")
    return np.concatenate(outs, axis=0)


class TorchClassifierHead:
    def __init__(self, model: GroupedFTParent, norm: dict[str, Any], head: str, n_classes: int, device: str, batch_size: int) -> None:
        self.model = model
        self.norm = norm
        self.head = head
        self.classes_ = np.arange(n_classes, dtype=int)
        self.device = device
        self.batch_size = int(batch_size)

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        device = torch.device(self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu")
        p = _predict_head(self.model, x, self.norm, self.head, device, self.batch_size)
        return p / np.maximum(p.sum(axis=1, keepdims=True), 1e-12)


class TorchQualityHead:
    def __init__(self, model: GroupedFTParent, norm: dict[str, Any], device: str, batch_size: int) -> None:
        self.model = model
        self.norm = norm
        self.device = device
        self.batch_size = int(batch_size)

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        device = torch.device(self.device if self.device == "cpu" or torch.cuda.is_available() else "cpu")
        return _predict_head(self.model, x, self.norm, "quality", device, self.batch_size)


def _parent_bundle(model: GroupedFTParent, norm: dict[str, Any], cfg: FullyLearnedGovernorConfig, feature_cols: list[str], device: str, batch_size: int, y: dict[str, np.ndarray]) -> dict[str, Any]:
    trade_mask = np.asarray(y["action"]) != ACTION_CASH
    out: dict[str, Any] = {
        "model_type": "alpha3_ft_grouped_hgb_surrogate_20260515",
        "feature_cols": list(feature_cols),
        "config": asdict(cfg),
        "action_model": TorchClassifierHead(model, norm, "action", 3, device, batch_size),
        "quality_model": TorchQualityHead(model, norm, device, batch_size),
        "default_bucket_indexes": {
            key: int(pd.Series(np.asarray(y[key])[trade_mask]).mode().iloc[0]) if np.any(trade_mask) else 0
            for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")
        },
        "label_distribution": {
            key: pd.Series(vals).value_counts().sort_index().to_dict()
            for key, vals in y.items()
            if key != "quality"
        },
    }
    sizes = {
        "notional": len(cfg.notional_buckets),
        "leverage": len(cfg.leverage_buckets),
        "take_profit": len(cfg.take_profit_buckets),
        "stop_loss": len(cfg.stop_loss_buckets),
        "max_hold": len(cfg.max_hold_buckets),
        "cooldown": len(cfg.cooldown_buckets),
    }
    for key, n in sizes.items():
        out[f"{key}_model"] = TorchClassifierHead(model, norm, key, n, device, batch_size)
    out["label_distribution"]["quality_mean"] = float(np.mean(y["quality"]))
    out["label_distribution"]["quality_p95"] = float(np.quantile(y["quality"], 0.95))
    return out


def _with_runtime_overlay(bundle: dict[str, Any], runtime_parent: dict[str, Any]) -> dict[str, Any]:
    out = copy.copy(bundle)
    out["training_config"] = dict(out.get("config", {}))
    out["config"] = dict(runtime_parent["config"])
    out["runtime_overlay_source"] = str(v31.DEFAULT_PARENT)
    out["runtime_overlay_note"] = "FT surrogate is trained on base h288 labels and HGB soft outputs; Alpha3 margin110 is applied only as runtime exposure overlay."
    return out


def _baseline(parent_ref: dict[str, Any], eval_df: pd.DataFrame, v27_model: Any, v27_payload: dict[str, Any], overlay: Any, limit_cfg: Any, fee: float, slip: float) -> dict[str, Any]:
    runner_payload = joblib.load(v31.DEFAULT_JACKPOT)
    runner = runner_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(runner_payload["selected_config"]))
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = ft_v2.ft_v1._load_teacher()
    runtime = ft_v2.ft_v1._selected_alpha3_runtime()
    hgb_dec = predict_policy_frame(parent_ref, eval_df, close=_close(eval_df))
    teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    teacher_pred = teacher._predict_deep(teacher_model, teacher_features, teacher_cols, teacher_norm)
    decisions = alpha2._decisions(hgb_dec, teacher_pred, teacher_buckets, runtime)
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    metrics = gbdt._metrics(eval_df, parent_ref, runner, add_cfg, eval_q, decisions, overlay, limit_cfg, fee=fee, slip=slip)
    return {"name": "alpha3_current_hgb_parent_teacher_downstream", "metrics": metrics, "score": gbdt._score(metrics)}


def _baseline_reproduced(metrics: dict[str, Any]) -> tuple[bool, list[str]]:
    errors: list[str] = []
    for cost, expected in BASELINE_EXPECTED.items():
        got = metrics.get(cost, {})
        if abs(float(got.get("pnl", 0.0)) - float(expected["pnl"])) > 0.05:
            errors.append(f"{cost}_pnl_mismatch")
        if abs(float(got.get("mdd", 0.0)) - float(expected["mdd"])) > 0.05:
            errors.append(f"{cost}_mdd_mismatch")
        if int(got.get("trades", -1)) != int(expected["trades"]):
            errors.append(f"{cost}_trades_mismatch")
    return not errors, errors


def _decision_audit(name: str, decisions: pd.DataFrame) -> dict[str, Any]:
    active = decisions["action"].astype(int).to_numpy() != ACTION_CASH
    return {
        "name": name,
        "rows": int(len(decisions)),
        "active_rate": float(np.mean(active)),
        "action_counts": {str(k): int(v) for k, v in decisions["action"].astype(int).value_counts().sort_index().to_dict().items()},
        "avg_notional_active": float(pd.to_numeric(decisions.loc[active, "notional_exposure"], errors="coerce").mean()) if np.any(active) else 0.0,
        "avg_leverage_active": float(pd.to_numeric(decisions.loc[active, "leverage"], errors="coerce").mean()) if np.any(active) else 0.0,
        "tp_p50_active": float(pd.to_numeric(decisions.loc[active, "take_profit"], errors="coerce").median()) if np.any(active) else 0.0,
        "sl_p50_active": float(pd.to_numeric(decisions.loc[active, "stop_loss"], errors="coerce").median()) if np.any(active) else 0.0,
        "max_hold_p50_active": float(pd.to_numeric(decisions.loc[active, "max_hold_bars"], errors="coerce").median()) if np.any(active) else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Contract-compliant Alpha3 FT parent redesign backtest.")
    parser.add_argument("--epochs", type=int, default=28)
    parser.add_argument("--teacher-epochs", type=int, default=35)
    parser.add_argument("--stride", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    torch.manual_seed(20260515)
    np.random.seed(20260515)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    gbdt.OUT_DIR = OUT_DIR
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} teacher_epochs={args.teacher_epochs} stride={args.stride}", flush=True)

    runtime_parent = joblib.load(v31.DEFAULT_PARENT)
    base_parent = joblib.load(BASE_PARENT)
    label_cfg = FullyLearnedGovernorConfig(**dict(base_parent["config"]))
    runtime_cfg = FullyLearnedGovernorConfig(**dict(runtime_parent["config"]))
    feature_cols = list(runtime_parent["feature_cols"])
    fee = float(dict(runtime_parent["config"])["fee"])
    slip = float(dict(runtime_parent["config"])["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    print(f"[{MODEL_ID}] building base h288 labels; margin110 reserved for runtime overlay", flush=True)
    x_train, y_train, train_meta = build_training_set(train_df, cfg=label_cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_val, y_val, val_meta = build_training_set(val_df, cfg=label_cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_train_norm, norm = _normalise_fit(x_train)
    x_val_norm = _normalise_apply(x_val, norm)
    train_soft = _hgb_soft_targets(runtime_parent, x_train, y_train, runtime_cfg)
    val_soft = _hgb_soft_targets(runtime_parent, x_val, y_val, runtime_cfg)
    train_ds = DistillDataset(x_train_norm.to_numpy(dtype=np.float32), y_train, train_soft)
    val_ds = DistillDataset(x_val_norm.astype(np.float32), y_val, val_soft)

    model = GroupedFTParent(len(feature_cols), label_cfg, d_model=80, n_layers=3)
    train_log = _train_model(model, train_ds, val_ds, epochs=int(args.epochs), batch_size=int(args.batch_size), device=device)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "normalizer": norm,
            "label_config": asdict(label_cfg),
            "runtime_config": asdict(runtime_cfg),
            "train_log": train_log,
        },
        MODEL_OUT,
    )

    parent_bundle = _parent_bundle(model, norm, label_cfg, feature_cols, str(device), int(args.batch_size), y_train)
    parent_bundle = _with_runtime_overlay(parent_bundle, runtime_parent)
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = ft_v2.ft_v1._limit_cfg()
    baseline = _baseline(runtime_parent, eval_df, v27_model, v27_payload, overlay, limit_cfg, fee, slip)
    ok, baseline_errors = _baseline_reproduced(baseline["metrics"])
    if not ok:
        raise RuntimeError(f"baseline reproduction failed: {baseline_errors}")
    print(
        f"[{MODEL_ID}] baseline c1={baseline['metrics']['cost1']['pnl']:.2f} "
        f"mdd={baseline['metrics']['cost1']['mdd']:.2f} c2={baseline['metrics']['cost2']['pnl']:.2f} c3={baseline['metrics']['cost3']['pnl']:.2f}",
        flush=True,
    )

    existing_runner_payload = joblib.load(v31.DEFAULT_JACKPOT)
    existing_runner = existing_runner_payload["cost_runner"]
    existing_add_cfg = CostRunnerConfig(**dict(existing_runner_payload["selected_config"]))
    contract_cols = _feature_cols(train_all, eval_df)
    result, rows, selected = gbdt._train_downstream(
        name="ft_grouped_hgb_surrogate",
        parent_for_features=runtime_parent,
        parent_bundle=parent_bundle,
        train_df=train_df,
        val_df=val_df,
        eval_df=eval_df,
        contract_cols=contract_cols,
        v27_model=v27_model,
        v27_payload=v27_payload,
        overlay=overlay,
        limit_cfg=limit_cfg,
        existing_runner=existing_runner,
        existing_add_cfg=existing_add_cfg,
        fee=fee,
        slip=slip,
        teacher_epochs=int(args.teacher_epochs),
    )
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)
    candidate = result
    delta = {
        cost: {
            "pnl": float(candidate["metrics"][cost]["pnl"] - baseline["metrics"][cost]["pnl"]),
            "mdd": float(candidate["metrics"][cost]["mdd"] - baseline["metrics"][cost]["mdd"]),
            "trades": int(candidate["metrics"][cost]["trades"] - baseline["metrics"][cost]["trades"]),
        }
        for cost in ("cost1", "cost2", "cost3")
    }
    train_dec_audit = _decision_audit("ft_train", predict_policy_frame(parent_bundle, train_df, close=_close(train_df)))
    val_dec_audit = _decision_audit("ft_val", predict_policy_frame(parent_bundle, val_df, close=_close(val_df)))
    eval_dec_audit = _decision_audit("ft_eval", predict_policy_frame(parent_bundle, eval_df, close=_close(eval_df)))
    warnings = list(audit_base.get("warnings", []))
    blockers = list(audit_base.get("blocking", []))
    if candidate["score"] <= baseline["score"]:
        warnings.append("ft_grouped_hgb_surrogate_retrained_downstream_did_not_beat_alpha3_baseline")
    if candidate["metrics"]["cost1"]["pnl"] <= 0:
        warnings.append("ft_grouped_hgb_surrogate_cost1_not_survived")
    audit = {
        "status": "pass" if not blockers else "fail",
        "verdict": "promote" if not blockers and candidate["score"] > baseline["score"] else "iterate",
        "blocking": blockers,
        "warnings": warnings,
        "baseline_reproduced": ok,
        "baseline_reproduction_errors": baseline_errors,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after FT parent training and downstream validation selection",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "label_training_config": asdict(label_cfg),
        "runtime_config": asdict(runtime_cfg),
        "runtime_overlay_source": str(v31.DEFAULT_PARENT),
        "selected_teacher_runtime": selected["teacher_runtime"],
        "selected_runner_config": selected["runner_config"],
        "parent_decision_audit": [train_dec_audit, val_dec_audit, eval_dec_audit],
        "base_feature_audit": audit_base,
        "alpha3_execution_contract": asdict(limit_cfg),
    }
    report = {
        "model_id": MODEL_ID,
        "base_model_alias": "alpha3",
        "frozen_protocol": "alpha3_frozen_backtest_protocol_20260515",
        "redesign_contract": str(ROOT / "docs/model_contracts/alpha3_ft_transformer_parent_replacement_redesign_20260515_contract.md"),
        "primary_mutable_surface": "parent_plus_downstream_retune",
        "changed_layers": ["parent", "teacher_gate", "v21_2_runner"],
        "frozen_layers": ["v27_deep_scout", "v31_exit", "execution", "accounting", "data"],
        "baseline_reproduced": ok,
        "baseline_metrics": baseline["metrics"],
        "candidate_metrics": candidate["metrics"],
        "delta_vs_baseline": delta,
        "selection_uses_2026": False,
        "selection_window": "2025Q4",
        "oos_window": "2026 fixed OOS",
        "route_counts": candidate["metrics"]["cost1"].get("route_counts", {}),
        "warnings": warnings,
        "red_team_blockers": blockers,
        "design": "Grouped-head FT-Transformer HGB-compatible neural surrogate. Labels are generated only with base h288. HGB margin110 is used as runtime overlay only. The surrogate distills HGB soft outputs plus base h288 supervised labels, then Alpha3 teacher gate and V21.2 runner are retrained and selected on 2025Q4 before one fixed 2026 OOS evaluation.",
        "train_log": train_log,
        "experiments": [baseline, candidate],
        "audit": audit,
        "artifacts": {
            "ft_model": str(MODEL_OUT),
            "out_dir": str(OUT_DIR),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
            **candidate.get("artifacts", {}),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "candidate": candidate["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
