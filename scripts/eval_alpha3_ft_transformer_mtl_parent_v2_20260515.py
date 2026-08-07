#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    FullyLearnedGovernorConfig,
    build_training_set,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3_exec  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_ft_transformer_mtl_parent_20260515 as ft_v1  # noqa: E402
from scripts import eval_alpha3_limit_close_fallback_20260514 as alpha3_close  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.eval_hf_v13_deep_tabular_parent_mdd_20260514 import (  # noqa: E402
    ParentDataset,
    RuntimeConfig,
    _decisions_from_outputs,
    _normalise_apply,
    _normalise_fit,
    _predict_outputs,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _json_default, _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha3_ft_transformer_mtl_parent_v2_20260515"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_ft_transformer_mtl_parent_v2_20260515"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_v2_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_v2_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_ft_transformer_mtl_parent_v2_20260515_grid.csv"


class FeatureGRNTokenizer(nn.Module):
    """Non-linear scalar feature tokenizer with a gated residual branch."""

    def __init__(self, n_features: int, d_model: int) -> None:
        super().__init__()
        self.base_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.base_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.hidden_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.hidden_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.gate_weight = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        self.gate_bias = nn.Parameter(torch.zeros(n_features, d_model))
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = x.unsqueeze(-1)
        base = v * self.base_weight.unsqueeze(0) + self.base_bias.unsqueeze(0)
        hidden = F.gelu(v * self.hidden_weight.unsqueeze(0) + self.hidden_bias.unsqueeze(0))
        gate = torch.sigmoid(v * self.gate_weight.unsqueeze(0) + self.gate_bias.unsqueeze(0))
        return self.norm(base + gate * hidden)


class AutoRegressiveHeadMixin:
    def _init_autoreg_heads(self, hidden: int, cfg: FullyLearnedGovernorConfig) -> None:
        self.action_trunk = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
        )
        self.action_head = nn.Linear(hidden, 3)
        self.param_trunk = nn.Sequential(
            nn.Linear(hidden * 2 + 3, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(hidden, hidden),
            nn.GELU(),
        )
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
            {name: nn.Parameter(torch.zeros(())) for name in ("action", "quality", "notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown")}
        )

    def _heads(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        action_hidden = self.action_trunk(z)
        action_logits = self.action_head(action_hidden)
        action_prob = torch.softmax(action_logits.clamp(-8.0, 8.0), dim=-1)
        param_hidden = self.param_trunk(torch.cat([z, action_hidden, action_prob], dim=-1))
        out = {
            "action": action_logits,
            "quality": self.quality_head(param_hidden).squeeze(-1),
        }
        out.update({k: head(param_hidden) for k, head in self.bucket_heads.items()})
        return out


class FTTransformerParentV2(nn.Module, AutoRegressiveHeadMixin):
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
        self._init_autoreg_heads(d_model, cfg)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        tokens = self.tokenizer(x_tab)
        cls = self.cls.expand(x_tab.shape[0], -1, -1)
        z = self.encoder(torch.cat([cls, tokens], dim=1))[:, 0]
        return self._heads(self.norm(z))


def _balanced(model: nn.Module, name: str, term: torch.Tensor) -> torch.Tensor:
    s = model.loss_log_vars[name].clamp(-3.0, 3.0)
    return torch.exp(-s) * term + 0.5 * s


def _loss_v2(model: nn.Module, outputs: dict[str, torch.Tensor], y: dict[str, torch.Tensor]) -> torch.Tensor:
    action = y["action"]
    active = action != ACTION_CASH
    action_weight = torch.ones(3, device=action.device)
    action_weight[ACTION_CASH] = 0.45
    losses: dict[str, torch.Tensor] = {
        "action": F.cross_entropy(outputs["action"], action, weight=action_weight)
    }
    q_weight = torch.where(active, torch.tensor(1.0, device=action.device), torch.tensor(0.35, device=action.device))
    losses["quality"] = (F.smooth_l1_loss(outputs["quality"], y["quality"], reduction="none") * q_weight).mean()
    if bool(active.any()):
        for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
            losses[key] = F.cross_entropy(outputs[key][active], y[key][active])
    else:
        zero = torch.zeros((), device=action.device)
        for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
            losses[key] = zero
    return sum(_balanced(model, key, val) for key, val in losses.items())


def _train_model_v2(
    model: nn.Module,
    train_ds: ParentDataset,
    val_ds: ParentDataset,
    *,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
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
        for xb, xs, yb in train_loader:
            xb = xb.to(device)
            xs = xs.to(device)
            yb = {k: v.to(device) for k, v in yb.items()}
            opt.zero_grad(set_to_none=True)
            loss = _loss_v2(model, model(xb, xs), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.5)
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
                vl = _loss_v2(model, model(xb, xs), yb)
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
        if stale >= 9:
            print(f"[{MODEL_ID}] early_stop epoch={epoch} best_val={best_val:.5f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best_val), "history": history}


def main() -> int:
    p = argparse.ArgumentParser(description="Train FT-Transformer v2 MTL parent replacement and backtest inside Alpha3 corrected stack.")
    p.add_argument("--epochs", type=int, default=32)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()

    torch.manual_seed(20260515)
    np.random.seed(20260515)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    original_parent = joblib.load(v31.DEFAULT_PARENT)
    cfg = FullyLearnedGovernorConfig(**dict(original_parent["config"]))
    feature_cols = list(original_parent["feature_cols"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_base = _audit_contract(train_all, eval_df, feature_cols)

    print(f"[{MODEL_ID}] building original MTL labels", flush=True)
    x_train, y_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)
    x_val, y_val, val_meta = build_training_set(val_df, cfg=cfg, stride_bars=max(3, int(args.stride)), batch_size=512, feature_cols=feature_cols)
    x_train_norm, norm = _normalise_fit(x_train)
    x_val_norm = _normalise_apply(x_val, norm)

    model = FTTransformerParentV2(len(feature_cols), cfg, d_model=80, n_layers=3)
    train_ds = ParentDataset(x_train_norm.to_numpy(dtype=np.float32), y_train)
    val_ds = ParentDataset(x_val_norm.astype(np.float32), y_val)
    print(f"[{MODEL_ID}] training FT-Transformer v2 MTL parent", flush=True)
    training = _train_model_v2(model, train_ds, val_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))

    torch.save(
        {
            "model_id": MODEL_ID,
            "architecture": "FTTransformerParentV2(GRN tokenizer, autoregressive heads, per-head uncertainty loss)",
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "normalizer": norm,
            "config": dict(original_parent["config"]),
            "training": training,
        },
        OUT_DIR / "ft_transformer_mtl_parent_v2.pt",
    )

    print(f"[{MODEL_ID}] loading fixed Alpha3 downstream layers", flush=True)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    _, v27_model = v31._load_v27(v31.DEFAULT_V27)
    v27_payload = torch.load(v31.DEFAULT_V27, map_location="cpu", weights_only=False)
    teacher_model, teacher_cols, teacher_norm, teacher_buckets = ft_v1._load_teacher()
    alpha3_runtime = ft_v1._selected_alpha3_runtime()
    overlay = next(v.overlay for v in l2._variants() if v.name == "alpha1_l2_conservative_fee20")
    limit_cfg = ft_v1._limit_cfg()
    fee = float(dict(original_parent["config"])["fee"])
    slip = float(dict(original_parent["config"])["slip"])

    print(f"[{MODEL_ID}] predicting validation/eval outputs", flush=True)
    val_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    val_x = _normalise_apply(val_features, norm)
    eval_x = _normalise_apply(eval_features, norm)
    val_out = _predict_outputs(model, val_x, None, device, int(args.batch_size), mc_passes=5)
    eval_out = _predict_outputs(model, eval_x, None, device, int(args.batch_size), mc_passes=5)
    val_teacher_features = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=teacher_cols)
    eval_teacher_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=teacher_cols)
    val_teacher_pred = teacher._predict_deep(teacher_model, val_teacher_features, teacher_cols, teacher_norm)
    eval_teacher_pred = teacher._predict_deep(teacher_model, eval_teacher_features, teacher_cols, teacher_norm)
    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])

    print(f"[{MODEL_ID}] selecting runtime on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_rt: RuntimeConfig | None = None
    best_score = -1e18
    rt_grid = ft_v1._runtime_grid()
    if args.quick:
        rt_grid = [r for r in rt_grid if r.confidence in (0.38, 0.54, 0.70) and r.quality_floor in (-0.01, 0.0) and r.uncertainty_max == 0.070]
    for rt in rt_grid:
        val_dec = _decisions_from_outputs(val_out, cfg, rt, val_df.index)
        metrics = ft_v1._alpha3_metrics(
            df=val_df,
            original_parent=original_parent,
            decision_frame=val_dec,
            teacher_pred=val_teacher_pred,
            teacher_buckets=teacher_buckets,
            alpha3_runtime=alpha3_runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            q=val_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        score = ft_v1._score(metrics)
        rows.append(
            {
                **asdict(rt),
                "score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost1_deep_entries": metrics["cost1"].get("deep_entries", 0),
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
            }
        )
        if score > best_score:
            best_score = float(score)
            best_rt = rt
            print(f"[{MODEL_ID}] new best {rt.name} score={score:.2f} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f}", flush=True)
    assert best_rt is not None
    pd.DataFrame(rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    eval_ft_dec = _decisions_from_outputs(eval_out, cfg, best_rt, eval_df.index)
    eval_hgb_dec = predict_policy_frame(original_parent, eval_df, close=_close(eval_df))
    experiments: list[dict[str, Any]] = []
    for name, dec in (
        ("alpha3_original_hgb_parent", eval_hgb_dec),
        (f"alpha3_ft_transformer_mtl_parent_v2::{best_rt.name}", eval_ft_dec),
    ):
        metrics = ft_v1._alpha3_metrics(
            df=eval_df,
            original_parent=original_parent,
            decision_frame=dec,
            teacher_pred=eval_teacher_pred,
            teacher_buckets=teacher_buckets,
            alpha3_runtime=alpha3_runtime,
            jackpot_model=jackpot_model,
            add_cfg=add_cfg,
            q=eval_q,
            overlay=overlay,
            limit_cfg=limit_cfg,
            fee=fee,
            slip=slip,
        )
        experiments.append({"name": name, "runtime": asdict(best_rt) if name.startswith("alpha3_ft") else None, "metrics": metrics, "score": ft_v1._score(metrics)})
        print(f"[{MODEL_ID}] {name} c1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}", flush=True)

    baseline = experiments[0]
    candidate = experiments[1]
    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if candidate["score"] <= baseline["score"]:
        warnings.append("ft_transformer_mtl_parent_v2_did_not_beat_alpha3_hgb_parent")
    if candidate["metrics"]["cost1"]["pnl"] <= 0:
        warnings.append("ft_transformer_mtl_parent_v2_cost1_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and candidate["score"] > baseline["score"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after runtime selection",
        "alpha3_execution_contract": asdict(limit_cfg),
        "alpha3_teacher_runtime": asdict(alpha3_runtime),
        "ft_selected_runtime": asdict(best_rt),
        "train_meta": train_meta,
        "val_meta": val_meta,
        "base_feature_audit": audit_base,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 parent replacement with FT-Transformer v2: GRN tokenizer, autoregressive action-conditioned parameter heads, and per-head homoscedastic uncertainty loss. Downstream Alpha3 teacher gate, V27 scout, V21.2 runner, L2/V31 overlay, and corrected next_open_limit_touch0_fee20 execution are fixed.",
        "architecture": {
            "backbone": "FTTransformerParentV2",
            "tokenization": "FeatureGRNTokenizer: learned linear token + GELU nonlinear branch gated by sigmoid per feature",
            "transformer": "3 encoder layers, d_model=80, 4 attention heads, GELU FFN",
            "heads": "action trunk first; action probabilities and action hidden state condition quality/notional/leverage/TP/SL/max_hold/cooldown",
            "loss": "per-task homoscedastic uncertainty variables for action, quality, notional, leverage, take_profit, stop_loss, max_hold, cooldown",
            "normalization": "train-only QuantileTransformer normal distribution fitted on 2025 Jan-Sep training candidates",
        },
        "training": training,
        "experiments": experiments,
        "audit": audit,
        "artifacts": {
            "model": str(OUT_DIR / "ft_transformer_mtl_parent_v2.pt"),
            "report": str(REPORT_OUT),
            "audit": str(AUDIT_OUT),
            "grid": str(GRID_OUT),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "candidate": candidate}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
