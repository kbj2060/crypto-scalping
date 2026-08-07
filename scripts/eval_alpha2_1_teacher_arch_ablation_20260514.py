#!/usr/bin/env python3
from __future__ import annotations

import gc
import json
import math
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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import (  # noqa: E402
    ACTION_CASH,
    ACTION_LONG,
    ACTION_SHORT,
    predict_policy_frame,
    prepare_features,
)
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha1_teacher_constrained_deep_parent_20260513 as teacher  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _feature_cols, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_deep_entry_parent_lite_v38 import DeepEntryParentLite, SEQ_LEN, _apply_norm, _normalizer, _seq_tensor  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha2_1_teacher_arch_ablation_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha2_1_teacher_arch_ablation_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha2_1_teacher_arch_ablation_20260514_grid.csv"
BASE_TEACHER = ROOT / "data/ensemble/supervised/alpha1_l2_teacher_deep_parent_20260514/teacher_deep_parent_l2_replay.pt"


@dataclass(frozen=True)
class ArchVariant:
    name: str
    model: str
    hgb_meta: bool = False
    focal_loss: bool = True
    dynamic_threshold: bool = False
    mc_dropout: bool = False
    uncertainty_cap: float = 0.030
    train_epochs: int = 48


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fixed_runtime() -> alpha2.Alpha2Runtime:
    return alpha2.Alpha2Runtime("noflip_c0.56_parent_scale1.10", 0.56, 1.10, 2.75)


def _l2_variant() -> Any:
    for variant in l2._variants():
        if variant.name == "alpha1_l2_conservative_fee20":
            return variant
    raise RuntimeError("alpha1_l2_conservative_fee20_not_found")


class TaskAttentionTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 96, notional_classes: int = 9, *, grn: bool = False, rope: bool = False) -> None:
        super().__init__()
        self.hidden = int(hidden)
        self.use_grn = bool(grn)
        self.use_rope = bool(rope)
        self.proj = nn.Linear(input_dim, hidden)
        if self.use_grn:
            self.grn_fc1 = nn.Linear(input_dim, hidden)
            self.grn_fc2 = nn.Linear(hidden, hidden * 2)
            self.grn_skip = nn.Linear(input_dim, hidden)
            self.grn_norm = nn.LayerNorm(hidden)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 3,
            dropout=0.10,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, hidden))
        self.action_attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.quality_attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.notional_attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.recency_bias = nn.Parameter(torch.linspace(-0.25, 0.35, SEQ_LEN).view(1, SEQ_LEN, 1))
        self.action_head = nn.Linear(hidden, 3)
        self.quality_head = nn.Linear(hidden, 1)
        self.notional_head = nn.Linear(hidden, notional_classes)

    def _front(self, seq: torch.Tensor) -> torch.Tensor:
        if not self.use_grn:
            return self.proj(seq)
        z = F.elu(self.grn_fc1(seq))
        value, gate = self.grn_fc2(z).chunk(2, dim=-1)
        return self.grn_norm(self.grn_skip(seq) + value * torch.sigmoid(gate))

    def _apply_rope(self, h: torch.Tensor) -> torch.Tensor:
        # RoPE-style relative rotation on projected temporal states. Keeping this
        # outside MultiheadAttention lets us compare it without rewriting PyTorch's
        # encoder internals.
        d = h.shape[-1]
        d2 = d - (d % 2)
        if d2 <= 0:
            return h
        pos = torch.arange(h.shape[1], device=h.device, dtype=h.dtype)
        inv = 1.0 / (10000 ** (torch.arange(0, d2, 2, device=h.device, dtype=h.dtype) / max(d2, 1)))
        theta = torch.einsum("s,d->sd", pos, inv).unsqueeze(0)
        x = h[..., :d2]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot = torch.stack((x1 * theta.cos() - x2 * theta.sin(), x1 * theta.sin() + x2 * theta.cos()), dim=-1).flatten(-2)
        if d2 == d:
            return rot
        return torch.cat([rot, h[..., d2:]], dim=-1)

    def _pool(self, h: torch.Tensor, attn: nn.Module) -> torch.Tensor:
        w = torch.softmax(attn(h) + self.recency_bias[:, -h.shape[1] :, :], dim=1)
        return torch.sum(h * w, dim=1)

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self._front(seq) + self.pos[:, -seq.shape[1] :, :]
        if self.use_rope:
            h = self._apply_rope(h)
        h = self.encoder(h)
        return (
            self.action_head(self._pool(h, self.action_attn)),
            self.quality_head(self._pool(h, self.quality_attn)).squeeze(-1),
            self.notional_head(self._pool(h, self.notional_attn)),
        )


def _new_model(name: str, input_dim: int, n_buckets: int) -> nn.Module:
    if name == "baseline":
        return DeepEntryParentLite(input_dim, notional_classes=n_buckets)
    if name == "task_attn":
        return TaskAttentionTeacher(input_dim, notional_classes=n_buckets)
    if name == "rope_task_attn":
        return TaskAttentionTeacher(input_dim, notional_classes=n_buckets, rope=True)
    if name == "grn_task_attn":
        return TaskAttentionTeacher(input_dim, notional_classes=n_buckets, grn=True)
    if name == "rope_grn_task_attn":
        return TaskAttentionTeacher(input_dim, notional_classes=n_buckets, grn=True, rope=True)
    raise ValueError(f"unknown_model:{name}")


def _bucket_labels(dec: pd.DataFrame, buckets: tuple[float, ...]) -> np.ndarray:
    vals = pd.to_numeric(dec["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    b = np.asarray(buckets, dtype=np.float64)
    return np.argmin(np.abs(vals[:, None] - b[None, :]), axis=1).astype(np.int64)


def _augment_hgb_meta(features: pd.DataFrame, decisions: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = features.copy()
    action = decisions["action"].astype(int).to_numpy()
    side = decisions["side"].astype(int).to_numpy()
    out["hgb_action_cash"] = (action == ACTION_CASH).astype(np.float32)
    out["hgb_action_long"] = (action == ACTION_LONG).astype(np.float32)
    out["hgb_action_short"] = (action == ACTION_SHORT).astype(np.float32)
    out["hgb_side"] = side.astype(np.float32)
    for col in ("quality_score", "confidence", "notional_exposure", "leverage", "take_profit", "stop_loss", "max_hold_bars"):
        out[f"hgb_{col}"] = pd.to_numeric(decisions[col], errors="coerce").fillna(0.0).astype(np.float32)
    return out, list(out.columns)


def _focal_loss(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    ce = F.nll_loss(logp, target, weight=weight, reduction="none")
    pt = p.gather(1, target.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
    return torch.mean(((1.0 - pt) ** gamma) * ce)


def _train_model(
    variant: ArchVariant,
    train_seq: np.ndarray,
    val_seq: np.ndarray,
    action: np.ndarray,
    val_action: np.ndarray,
    quality: np.ndarray,
    val_quality: np.ndarray,
    notional: np.ndarray,
    val_notional: np.ndarray,
    *,
    n_buckets: int,
) -> tuple[nn.Module, dict[str, Any]]:
    torch.manual_seed(20260514)
    norm = _normalizer(train_seq)
    x = _apply_norm(train_seq, norm)
    vx = _apply_norm(val_seq, norm)
    device = _device()
    model = _new_model(variant.model, x.shape[-1], n_buckets).to(device)
    counts = np.bincount(action, minlength=3).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights[0] *= 0.25
    weights = weights / max(float(weights.mean()), 1e-6)
    weight_t = torch.from_numpy(weights).to(device)
    ce_size = nn.CrossEntropyLoss()
    huber = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x),
            torch.from_numpy(action.astype(np.int64)),
            torch.from_numpy(quality.astype(np.float32)),
            torch.from_numpy(notional.astype(np.int64)),
        ),
        batch_size=256,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
    )
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.55, patience=3)
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    print(f"[{MODEL_ID}] train {variant.name} epochs<={variant.train_epochs}", flush=True)
    for ep in range(int(variant.train_epochs)):
        model.train()
        loss_sum = 0.0
        for xb, ab, qb, nb in loader:
            xb, ab, qb, nb = xb.to(device, non_blocking=True), ab.to(device, non_blocking=True), qb.to(device, non_blocking=True), nb.to(device, non_blocking=True)
            logits, qhat, nlogits = model(xb)
            active = ab != ACTION_CASH
            action_loss = _focal_loss(logits, ab, weight_t) if variant.focal_loss else F.cross_entropy(logits, ab, weight=weight_t)
            loss = action_loss + 1.2 * huber(qhat, qb)
            if torch.any(active):
                loss = loss + 0.25 * ce_size(nlogits[active], nb[active])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.detach().cpu())
        model.eval()
        val_losses: list[float] = []
        with torch.no_grad():
            for start in range(0, len(vx), 4096):
                xb = torch.from_numpy(vx[start : start + 4096]).to(device)
                ab = torch.from_numpy(val_action[start : start + 4096].astype(np.int64)).to(device)
                qb = torch.from_numpy(val_quality[start : start + 4096].astype(np.float32)).to(device)
                nb = torch.from_numpy(val_notional[start : start + 4096].astype(np.int64)).to(device)
                logits, qhat, nlogits = model(xb)
                active = ab != ACTION_CASH
                vloss = F.cross_entropy(logits, ab, weight=weight_t) + 1.2 * huber(qhat, qb)
                if torch.any(active):
                    vloss = vloss + 0.25 * ce_size(nlogits[active], nb[active])
                val_losses.append(float(vloss.detach().cpu()))
        val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
        if ep in {0, int(variant.train_epochs) - 1} or (ep + 1) % 8 == 0:
            lr = float(opt.param_groups[0]["lr"])
            print(f"[{MODEL_ID}] {variant.name} epoch={ep+1} train={loss_sum/max(len(loader),1):.5f} val={val_loss:.5f} lr={lr:.2e}", flush=True)
        if bad_epochs >= 8:
            print(f"[{MODEL_ID}] {variant.name} early_stop epoch={ep+1} best_val={best_val:.5f}", flush=True)
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model.cpu().eval(), {"norm": norm, "best_val_loss": best_val, "epochs_ran": ep + 1, "label_counts": {str(i): int(v) for i, v in enumerate(counts)}}


def _predict_model(model: nn.Module, features: pd.DataFrame, cols: list[str], norm: dict[str, np.ndarray], *, mc: bool = False, repeats: int = 8) -> dict[str, np.ndarray]:
    indices = np.arange(len(features), dtype=np.int64)
    seq = _seq_tensor(features, indices, cols)
    x = _apply_norm(seq, norm)
    device = _device()
    model = model.to(device)
    if not mc:
        model.eval()
        probs: list[np.ndarray] = []
        qvals: list[np.ndarray] = []
        nprobs: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(x), 4096):
                logits, qhat, nlogits = model(torch.from_numpy(x[start : start + 4096]).to(device))
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                qvals.append(qhat.detach().cpu().numpy())
                nprobs.append(torch.softmax(nlogits, dim=1).detach().cpu().numpy())
        return {"action_proba": np.vstack(probs), "quality": np.concatenate(qvals), "notional_proba": np.vstack(nprobs)}
    # MC dropout: keep dropout active, then use the predictive mean and max class variance.
    model.train()
    all_p: list[np.ndarray] = []
    all_q: list[np.ndarray] = []
    all_n: list[np.ndarray] = []
    with torch.no_grad():
        for _ in range(int(repeats)):
            probs: list[np.ndarray] = []
            qvals: list[np.ndarray] = []
            nprobs: list[np.ndarray] = []
            for start in range(0, len(x), 4096):
                logits, qhat, nlogits = model(torch.from_numpy(x[start : start + 4096]).to(device))
                probs.append(torch.softmax(logits, dim=1).detach().cpu().numpy())
                qvals.append(qhat.detach().cpu().numpy())
                nprobs.append(torch.softmax(nlogits, dim=1).detach().cpu().numpy())
            all_p.append(np.vstack(probs))
            all_q.append(np.concatenate(qvals))
            all_n.append(np.vstack(nprobs))
    stack_p = np.stack(all_p, axis=0)
    return {
        "action_proba": stack_p.mean(axis=0),
        "quality": np.stack(all_q, axis=0).mean(axis=0),
        "notional_proba": np.stack(all_n, axis=0).mean(axis=0),
        "uncertainty": stack_p.var(axis=0).max(axis=1),
    }


def _dynamic_decisions(base_dec: pd.DataFrame, pred: dict[str, np.ndarray], buckets: tuple[float, ...], *, variant: ArchVariant) -> pd.DataFrame:
    out = base_dec.copy()
    p = np.asarray(pred["action_proba"], dtype=np.float64)
    pred_action = np.argmax(p, axis=1).astype(np.int64)
    conf = np.max(p, axis=1)
    q = np.asarray(pred["quality"], dtype=np.float64)
    q_norm = np.tanh(q / max(float(np.nanstd(q) or 1.0), 1e-6))
    threshold = np.full(len(out), 0.56, dtype=np.float64)
    if variant.dynamic_threshold:
        threshold = np.clip(0.56 - 0.05 * q_norm, 0.50, 0.62)
    active = (out["action"].astype(int).to_numpy() != ACTION_CASH) & (out["side"].astype(int).to_numpy() != 0)
    active &= conf >= threshold
    active &= pred_action != ACTION_CASH
    if variant.mc_dropout and "uncertainty" in pred:
        active &= np.asarray(pred["uncertainty"], dtype=np.float64) <= float(variant.uncertainty_cap)
    out.loc[~active, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[~active, "leverage"] = 1.0
    notional = pd.to_numeric(out["notional_exposure"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
    leverage = pd.to_numeric(out["leverage"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64)
    scaled = np.minimum(notional * 1.10, 2.75)
    out.loc[active, "notional_exposure"] = scaled[active]
    out.loc[active, "position_fraction"] = scaled[active] / np.maximum(leverage[active], 1e-12)
    out.loc[:, "quality_score"] = q
    out.loc[:, "confidence"] = conf
    out.loc[:, "teacher_threshold"] = threshold
    if "uncertainty" in pred:
        out.loc[:, "teacher_uncertainty"] = np.asarray(pred["uncertainty"], dtype=np.float64)
    return out


def _metrics(df, parent, jackpot_model, add_cfg, q, decisions, variant, fee: float, slip: float) -> dict[str, Any]:
    return alpha2._metrics(df, parent, jackpot_model, add_cfg, q, decisions, variant, fee=fee, slip=slip)


def _eval_decisions(name: str, val_dec: pd.DataFrame, eval_dec: pd.DataFrame, val_pred: dict[str, np.ndarray], eval_pred: dict[str, np.ndarray], buckets: tuple[float, ...], variant: ArchVariant, stack: dict[str, Any]) -> dict[str, Any]:
    l2_variant = stack["l2_variant"]
    if variant.dynamic_threshold or variant.mc_dropout:
        val_out = _dynamic_decisions(val_dec, val_pred, buckets, variant=variant)
        eval_out = _dynamic_decisions(eval_dec, eval_pred, buckets, variant=variant)
    else:
        rt = _fixed_runtime()
        val_out = alpha2._decisions(val_dec, val_pred, buckets, rt)
        eval_out = alpha2._decisions(eval_dec, eval_pred, buckets, rt)
    val_metrics = _metrics(stack["val"], stack["parent"], stack["jackpot_model"], stack["add_cfg"], stack["val_q"], val_out, l2_variant, stack["fee"], stack["slip"])
    eval_metrics = _metrics(stack["eval"], stack["parent"], stack["jackpot_model"], stack["add_cfg"], stack["eval_q"], eval_out, l2_variant, stack["fee"], stack["slip"])
    return {
        "name": name,
        "variant": asdict(variant),
        "selection_score": alpha2._score(val_metrics["cost1"], val_metrics["cost2"], val_metrics["cost3"]),
        "val_metrics": val_metrics,
        "metrics": eval_metrics,
        "score": alpha2._score(eval_metrics["cost1"], eval_metrics["cost2"], eval_metrics["cost3"]),
    }


def _load_stack() -> dict[str, Any]:
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_cols = _feature_cols(train_all, eval_df)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    print(f"[{MODEL_ID}] parent decisions and V27 q", flush=True)
    train_dec = predict_policy_frame(parent, train, close=_close(train))
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    train_features = prepare_features(train, side_hint=0, close=_close(train), feature_cols=feature_cols)
    val_features = prepare_features(val, side_hint=0, close=_close(val), feature_cols=feature_cols)
    eval_features = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    return {
        "parent": parent,
        "jackpot_model": jackpot_model,
        "add_cfg": add_cfg,
        "train": train,
        "val": val,
        "eval": eval_df,
        "train_dec": train_dec,
        "val_dec": val_dec,
        "eval_dec": eval_dec,
        "train_features": train_features,
        "val_features": val_features,
        "eval_features": eval_features,
        "feature_cols": feature_cols,
        "val_q": val_q,
        "eval_q": eval_q,
        "fee": float(base["fee"]),
        "slip": float(base["slip"]),
        "buckets": tuple(float(x) for x in base.get("notional_buckets", (0.23, 0.368, 0.575, 0.8625, 1.2075, 1.6675, 2.3, 3.105, 4.14))),
        "audit": parent_audit,
        "l2_variant": _l2_variant(),
    }


def _load_base_teacher() -> tuple[nn.Module, list[str], dict[str, np.ndarray], tuple[float, ...]]:
    payload = torch.load(BASE_TEACHER, map_location="cpu", weights_only=False)
    cols = list(payload["feature_cols"])
    buckets = tuple(float(x) for x in payload["buckets"])
    model = DeepEntryParentLite(len(cols), notional_classes=len(buckets))
    model.load_state_dict(payload["state_dict"])
    return model.cpu().eval(), cols, dict(payload["train_meta"]["norm"]), buckets


def main() -> int:
    print(f"[{MODEL_ID}] start", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = _load_stack()
    buckets = stack["buckets"]
    experiments: list[dict[str, Any]] = []

    base_variant = ArchVariant("alpha2_1_reference_saved_teacher", "baseline", focal_loss=False)
    base_model, base_cols, base_norm, _base_buckets = _load_base_teacher()
    base_val_pred = _predict_model(base_model, stack["val_features"].reindex(columns=base_cols, fill_value=0.0), base_cols, base_norm)
    base_eval_pred = _predict_model(base_model, stack["eval_features"].reindex(columns=base_cols, fill_value=0.0), base_cols, base_norm)
    experiments.append(_eval_decisions(base_variant.name, stack["val_dec"], stack["eval_dec"], base_val_pred, base_eval_pred, buckets, base_variant, stack))

    dynamic_variant = ArchVariant("quality_dynamic_threshold_saved_teacher", "baseline", focal_loss=False, dynamic_threshold=True)
    experiments.append(_eval_decisions(dynamic_variant.name, stack["val_dec"], stack["eval_dec"], base_val_pred, base_eval_pred, buckets, dynamic_variant, stack))

    mc_variant = ArchVariant("mc_dropout_uncertainty_saved_teacher", "baseline", focal_loss=False, dynamic_threshold=True, mc_dropout=True, uncertainty_cap=0.020)
    mc_val_pred = _predict_model(base_model, stack["val_features"].reindex(columns=base_cols, fill_value=0.0), base_cols, base_norm, mc=True, repeats=8)
    mc_eval_pred = _predict_model(base_model, stack["eval_features"].reindex(columns=base_cols, fill_value=0.0), base_cols, base_norm, mc=True, repeats=8)
    experiments.append(_eval_decisions(mc_variant.name, stack["val_dec"], stack["eval_dec"], mc_val_pred, mc_eval_pred, buckets, mc_variant, stack))
    del base_model, base_val_pred, base_eval_pred, mc_val_pred, mc_eval_pred
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    train_y_action = stack["train_dec"]["action"].astype(int).to_numpy(dtype=np.int64)
    val_y_action = stack["val_dec"]["action"].astype(int).to_numpy(dtype=np.int64)
    train_y_quality = pd.to_numeric(stack["train_dec"]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    val_y_quality = pd.to_numeric(stack["val_dec"]["quality_score"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
    train_y_notional = _bucket_labels(stack["train_dec"], buckets)
    val_y_notional = _bucket_labels(stack["val_dec"], buckets)

    train_variants = [
        ArchVariant("task_specific_attention_focal", "task_attn", train_epochs=48),
        ArchVariant("grn_task_attention_focal", "grn_task_attn", train_epochs=48),
        ArchVariant("rope_grn_task_attention_focal", "rope_grn_task_attn", train_epochs=48),
        ArchVariant("hgb_meta_grn_task_attention_focal", "grn_task_attn", hgb_meta=True, train_epochs=48),
    ]
    for variant in train_variants:
        print(f"[{MODEL_ID}] prepare {variant.name}", flush=True)
        if variant.hgb_meta:
            train_features, cols = _augment_hgb_meta(stack["train_features"], stack["train_dec"])
            val_features, _ = _augment_hgb_meta(stack["val_features"], stack["val_dec"])
            eval_features, _ = _augment_hgb_meta(stack["eval_features"], stack["eval_dec"])
        else:
            cols = list(stack["feature_cols"])
            train_features = stack["train_features"].reindex(columns=cols, fill_value=0.0)
            val_features = stack["val_features"].reindex(columns=cols, fill_value=0.0)
            eval_features = stack["eval_features"].reindex(columns=cols, fill_value=0.0)
        train_seq = _seq_tensor(train_features, np.arange(len(train_features), dtype=np.int64), cols)
        val_seq = _seq_tensor(val_features, np.arange(len(val_features), dtype=np.int64), cols)
        model, meta = _train_model(
            variant,
            train_seq,
            val_seq,
            train_y_action,
            val_y_action,
            train_y_quality,
            val_y_quality,
            train_y_notional,
            val_y_notional,
            n_buckets=len(buckets),
        )
        val_pred = _predict_model(model, val_features, cols, meta["norm"])
        eval_pred = _predict_model(model, eval_features, cols, meta["norm"])
        row = _eval_decisions(variant.name, stack["val_dec"], stack["eval_dec"], val_pred, eval_pred, buckets, variant, stack)
        row["train_meta"] = {k: v for k, v in meta.items() if k != "norm"}
        experiments.append(row)
        torch.save(
            {
                "model_id": MODEL_ID,
                "variant": asdict(variant),
                "state_dict": model.state_dict(),
                "feature_cols": cols,
                "train_meta": meta,
                "buckets": buckets,
            },
            OUT_DIR / f"{variant.name}.pt",
        )
        print(
            f"[{MODEL_ID}] {variant.name} OOS cost1={row['metrics']['cost1']['pnl']:.2f} "
            f"mdd={row['metrics']['cost1']['mdd']:.2f} cost2={row['metrics']['cost2']['pnl']:.2f} "
            f"cost3={row['metrics']['cost3']['pnl']:.2f}",
            flush=True,
        )
        del train_seq, val_seq, model, val_pred, eval_pred
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    grid_rows = []
    for exp in experiments:
        grid_rows.append(
            {
                "name": exp["name"],
                "selection_score": exp["selection_score"],
                "score": exp["score"],
                "val_cost1_pnl": exp["val_metrics"]["cost1"]["pnl"],
                "val_cost1_mdd": exp["val_metrics"]["cost1"]["mdd"],
                "val_cost2_pnl": exp["val_metrics"]["cost2"]["pnl"],
                "val_cost3_pnl": exp["val_metrics"]["cost3"]["pnl"],
                "cost1_pnl": exp["metrics"]["cost1"]["pnl"],
                "cost1_mdd": exp["metrics"]["cost1"]["mdd"],
                "cost1_trades": exp["metrics"]["cost1"]["trades"],
                "cost2_pnl": exp["metrics"]["cost2"]["pnl"],
                "cost3_pnl": exp["metrics"]["cost3"]["pnl"],
            }
        )
    pd.DataFrame(grid_rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    selected = max(experiments, key=lambda x: float(x["selection_score"]))
    blocking = list(stack["audit"].get("blocking", []))
    warnings = list(stack["audit"].get("warnings", []))
    if selected["name"] == "alpha2_1_reference_saved_teacher":
        warnings.append("no_teacher_arch_variant_beat_reference_on_selection")
    if selected["metrics"]["cost1"]["pnl"] <= experiments[0]["metrics"]["cost1"]["pnl"]:
        warnings.append("selected_variant_did_not_beat_reference_oos_cost1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote_candidate" if not blocking and selected["name"] != "alpha2_1_reference_saved_teacher" else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection only",
        "selected": selected["name"],
        "l2_variant": asdict(stack["l2_variant"]),
    }
    report = {
        "model_id": MODEL_ID,
        "design": {
            "tested": [
                "saved teacher + fixed Alpha2.1 runtime",
                "Quality-head dynamic confidence threshold",
                "MC-dropout uncertainty guard",
                "Task-specific attention pooling",
                "GRN feature gate + task-specific attention",
                "RoPE-style temporal rotation + GRN + task-specific attention",
                "HGB meta-state features + GRN + task-specific attention",
            ],
            "deferred": [
                "Cost3 asymmetric quality relabeling requires trade-attribution labels before safe use",
                "Decision Transformer migration requires trajectory dataset and is not a layer-local ablation",
                "Contrastive loss requires paired bad-trade mining; run after identifying stable best backbone",
            ],
        },
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "out_dir": str(OUT_DIR)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "selected": selected["name"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
