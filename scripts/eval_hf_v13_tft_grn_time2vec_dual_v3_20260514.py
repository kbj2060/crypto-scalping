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
from scripts import eval_hf_v13_deep_tabular_parent_mdd_20260514 as base  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "hf_v13_tft_grn_time2vec_dual_v3_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_tft_grn_time2vec_dual_v3_20260514"
REPORT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_grn_time2vec_dual_v3_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_grn_time2vec_dual_v3_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/hf_v13_tft_grn_time2vec_dual_v3_20260514_grid.csv"
SEQ_LEN = 72


@dataclass(frozen=True)
class RuntimeV3:
    name: str
    mode: str
    veto_conf: float
    veto_q: float
    veto_unc: float
    override_conf: float
    override_q: float
    override_unc: float
    override_notional_scale: float
    override_max_notional: float


class DualDataset(Dataset):
    def __init__(self, x_tab: np.ndarray, x_seq: np.ndarray, y_teacher: dict[str, np.ndarray], y_gt: dict[str, np.ndarray]) -> None:
        self.x_tab = torch.as_tensor(x_tab, dtype=torch.float32)
        self.x_seq = torch.as_tensor(x_seq, dtype=torch.float32)
        self.y_teacher = self._tensorise(y_teacher)
        self.y_gt = self._tensorise(y_gt)

    @staticmethod
    def _tensorise(y: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
        return {
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return self.x_tab[idx], self.x_seq[idx], {k: v[idx] for k, v in self.y_teacher.items()}, {k: v[idx] for k, v in self.y_gt.items()}


class GRN(nn.Module):
    def __init__(self, dim: int, hidden: int | None = None, dropout: float = 0.10) -> None:
        super().__init__()
        hidden = int(hidden or dim * 2)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.elu(self.fc1(x))
        y = self.dropout(self.fc2(y))
        val, gate = y.chunk(2, dim=-1)
        return self.norm(x + val * torch.sigmoid(gate))


class Time2Vec(nn.Module):
    def __init__(self, hidden: int, seq_len: int = SEQ_LEN) -> None:
        super().__init__()
        self.seq_len = int(seq_len)
        self.linear = nn.Parameter(torch.randn(1) * 0.01)
        self.linear_bias = nn.Parameter(torch.zeros(1))
        self.freq = nn.Parameter(torch.randn(hidden - 1) * 0.02)
        self.phase = nn.Parameter(torch.zeros(hidden - 1))

    def forward(self, batch: int, device: torch.device) -> torch.Tensor:
        t = torch.linspace(0.0, 1.0, self.seq_len, device=device).view(1, self.seq_len, 1)
        lin = self.linear * t + self.linear_bias
        per = torch.sin(t * self.freq.view(1, 1, -1) + self.phase.view(1, 1, -1))
        return torch.cat([lin, per], dim=-1).expand(batch, -1, -1)


class TFTV3Parent(nn.Module, base.HeadMixin):
    def __init__(self, n_features: int, cfg: FullyLearnedGovernorConfig, hidden: int = 96, n_layers: int = 2) -> None:
        super().__init__()
        self.input_grn = GRN(n_features, hidden=n_features * 2)
        self.proj = nn.Linear(n_features, hidden)
        self.time2vec = Time2Vec(hidden)
        enc = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=4,
            dim_feedforward=hidden * 4,
            dropout=0.12,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=n_layers)
        self.attn = nn.Sequential(nn.Linear(hidden, hidden // 2), nn.Tanh(), nn.Linear(hidden // 2, 1))
        self.skip = nn.Sequential(nn.Linear(n_features, hidden), nn.GELU(), nn.Dropout(0.08))
        self.norm = nn.LayerNorm(hidden)
        self._init_heads(hidden, cfg)

    def forward(self, x_tab: torch.Tensor, x_seq: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        seq = x_seq if x_seq is not None and x_seq.ndim == 3 else x_tab[:, None, :]
        seq = self.input_grn(seq)
        h = self.proj(seq) + self.time2vec(seq.shape[0], seq.device)
        h = self.encoder(h)
        recency = torch.linspace(0.0, 0.45, h.shape[1], device=h.device).view(1, -1, 1)
        w = torch.softmax(self.attn(h) + recency, dim=1)
        z = torch.sum(h * w, dim=1) + self.skip(x_tab)
        return self._heads(self.norm(z))


def _dual_loss(
    model: nn.Module,
    outputs: dict[str, torch.Tensor],
    y_teacher: dict[str, torch.Tensor],
    y_gt: dict[str, torch.Tensor],
    *,
    teacher_weight: float,
) -> torch.Tensor:
    def one(y: dict[str, torch.Tensor], quality_focal: bool) -> torch.Tensor:
        action = y["action"]
        active = action != ACTION_CASH
        action_weight = torch.ones(3, device=action.device)
        action_weight[ACTION_CASH] = 0.45
        action_loss = F.cross_entropy(outputs["action"], action, weight=action_weight)
        q_err = F.smooth_l1_loss(outputs["quality"], y["quality"], reduction="none")
        q_mag = y["quality"].abs().clamp(0.0, 0.15)
        q_weight = torch.where(active, torch.tensor(1.0, device=action.device), torch.tensor(0.35, device=action.device))
        if quality_focal:
            q_weight = q_weight * (1.0 + 45.0 * q_mag.pow(1.35))
        quality_loss = (q_err * q_weight).mean()
        bucket_loss = torch.zeros((), device=action.device)
        if bool(active.any()):
            for key in ("notional", "leverage", "take_profit", "stop_loss", "max_hold", "cooldown"):
                bucket_loss = bucket_loss + F.cross_entropy(outputs[key][active], y[key][active])
            bucket_loss = bucket_loss / 6.0
        return base._balanced(model, "action", action_loss) + base._balanced(model, "quality", quality_loss) + 0.60 * base._balanced(model, "bucket", bucket_loss)

    tw = float(np.clip(teacher_weight, 0.0, 1.0))
    return tw * one(y_teacher, quality_focal=False) + (1.0 - tw) * one(y_gt, quality_focal=True)


def _train_tft_v3(
    model: TFTV3Parent,
    train_ds: DualDataset,
    val_ds: DualDataset,
    *,
    epochs: int,
    device: torch.device,
    batch_size: int,
) -> dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1.5e-4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        teacher_w = max(0.58, 0.86 - 0.28 * (epoch - 1) / max(epochs - 1, 1))
        model.train()
        total = 0.0
        count = 0
        for xb, xs, yt, yg in train_loader:
            xb = xb.to(device)
            xs = xs.to(device)
            yt = {k: v.to(device) for k, v in yt.items()}
            yg = {k: v.to(device) for k, v in yg.items()}
            opt.zero_grad(set_to_none=True)
            loss = _dual_loss(model, model(xb, xs), yt, yg, teacher_weight=teacher_w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.item()) * len(xb)
            count += len(xb)
        model.eval()
        vtotal = 0.0
        vcount = 0
        with torch.no_grad():
            for xb, xs, yt, yg in val_loader:
                xb = xb.to(device)
                xs = xs.to(device)
                yt = {k: v.to(device) for k, v in yt.items()}
                yg = {k: v.to(device) for k, v in yg.items()}
                loss = _dual_loss(model, model(xb, xs), yt, yg, teacher_weight=teacher_w)
                vtotal += float(loss.item()) * len(xb)
                vcount += len(xb)
        tr = total / max(count, 1)
        va = vtotal / max(vcount, 1)
        history.append({"epoch": float(epoch), "teacher_weight": float(teacher_w), "train_loss": tr, "val_loss": va})
        print(f"[{MODEL_ID}] epoch={epoch:02d} teacher_w={teacher_w:.3f} train_loss={tr:.5f} val_loss={va:.5f}", flush=True)
        if va < best_val:
            best_val = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.to("cpu")
    return {"best_val_loss": float(best_val), "history": history}


def _runtime_grid() -> list[RuntimeV3]:
    rows: list[RuntimeV3] = []
    for conf in (0.30, 0.38, 0.46):
        for q in (-0.015, -0.005, 0.005):
            for unc in (0.070, 0.095):
                rows.append(RuntimeV3(f"veto_c{conf:.2f}_q{q:.3f}_u{unc:.3f}", "veto", conf, q, unc, 9.0, 9.0, -1.0, 1.0, 3.0))
    for conf in (0.62, 0.70, 0.78):
        for unc in (0.030, 0.050):
            for scale, cap in ((0.55, 1.20), (0.75, 1.60)):
                rows.append(RuntimeV3(f"hybrid_ovr_c{conf:.2f}_u{unc:.3f}_s{scale:.2f}_cap{cap:.2f}", "hybrid", 0.38, -0.010, 0.095, conf, 0.0, unc, scale, cap))
    for conf in (0.38, 0.46, 0.54, 0.62, 0.70):
        for q in (-0.015, -0.005, 0.005):
            for unc in (0.050, 0.070, 0.095):
                for scale, cap in ((0.45, 1.20), (0.60, 1.60), (0.75, 2.00)):
                    rows.append(RuntimeV3(f"replace_c{conf:.2f}_q{q:.3f}_u{unc:.3f}_s{scale:.2f}_cap{cap:.2f}", "replace", conf, q, unc, 9.0, 9.0, -1.0, scale, cap))
    return rows


def _decision_frame(
    outputs: dict[str, np.ndarray],
    teacher: pd.DataFrame,
    cfg: FullyLearnedGovernorConfig,
    rt: RuntimeV3,
    index: pd.Index,
) -> pd.DataFrame:
    action_p = outputs["action"]
    pred_action = np.argmax(action_p, axis=1).astype(np.int64)
    pred_conf = np.max(action_p, axis=1)
    side = np.where(pred_action == ACTION_LONG, 1, np.where(pred_action == ACTION_SHORT, -1, 0)).astype(np.int64)
    uncertainty = np.asarray(outputs.get("action_uncertainty", np.zeros_like(pred_conf)), dtype=np.float64)
    quality = np.asarray(outputs["quality"], dtype=np.float64)
    notional, _, _ = base._expected_bucket(outputs["notional"], cfg.notional_buckets)
    leverage, _, _ = base._expected_bucket(outputs["leverage"], cfg.leverage_buckets)
    tp, _, _ = base._expected_bucket(outputs["take_profit"], cfg.take_profit_buckets)
    sl, _, _ = base._expected_bucket(outputs["stop_loss"], cfg.stop_loss_buckets)
    mh, _, _ = base._expected_bucket(outputs["max_hold"], tuple(float(v) for v in cfg.max_hold_buckets))
    cd, _, _ = base._expected_bucket(outputs["cooldown"], tuple(float(v) for v in cfg.cooldown_buckets))

    if rt.mode == "replace":
        lev = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
        noz = np.clip(notional * rt.override_notional_scale, min(cfg.notional_buckets), rt.override_max_notional)
        frac = np.clip(noz / np.maximum(lev, 1e-8), 0.0, cfg.max_margin_fraction)
        noz = frac * lev
        active = (pred_action != ACTION_CASH) & (side != 0) & (pred_conf >= rt.veto_conf) & (quality >= rt.veto_q) & (uncertainty <= rt.veto_unc)
        out = pd.DataFrame(
            {
                "action": np.where(active, pred_action, ACTION_CASH).astype(np.int64),
                "side": np.where(active, side, 0).astype(np.int64),
                "notional_exposure": noz.astype(np.float64),
                "leverage": lev.astype(np.float64),
                "position_fraction": frac.astype(np.float64),
                "take_profit": tp.astype(np.float64),
                "stop_loss": sl.astype(np.float64),
                "max_hold_bars": np.rint(mh).astype(np.int64),
                "cooldown_bars": np.rint(cd).astype(np.int64),
                "quality_score": quality.astype(np.float64),
                "confidence": pred_conf.astype(np.float64),
                "tft_v3_action": pred_action,
                "tft_v3_confidence": pred_conf,
                "tft_v3_quality": quality,
                "tft_v3_uncertainty": uncertainty,
            },
            index=index,
        )
        cash = out["action"].astype(int).to_numpy() == ACTION_CASH
        out.loc[cash, ["side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
        out.loc[cash, "leverage"] = 1.0
        return out

    out = teacher.copy()
    teacher_action = out["action"].astype(int).to_numpy()
    teacher_side = out["side"].astype(int).to_numpy()
    teacher_active = (teacher_action != ACTION_CASH) & (teacher_side != 0)
    agree = (pred_action == teacher_action) & (side == teacher_side)
    keep_teacher = teacher_active & agree & (pred_conf >= rt.veto_conf) & (quality >= rt.veto_q) & (uncertainty <= rt.veto_unc)
    out.loc[teacher_active & ~keep_teacher, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
    out.loc[teacher_active & ~keep_teacher, "leverage"] = 1.0

    if rt.mode == "hybrid":
        teacher_cash = ~teacher_active
        override = teacher_cash & (pred_action != ACTION_CASH) & (side != 0) & (pred_conf >= rt.override_conf) & (quality >= rt.override_q) & (uncertainty <= rt.override_unc)
        lev = np.clip(leverage, min(cfg.leverage_buckets), max(cfg.leverage_buckets))
        noz = np.clip(notional * rt.override_notional_scale, min(cfg.notional_buckets), rt.override_max_notional)
        frac = np.clip(noz / np.maximum(lev, 1e-8), 0.0, cfg.max_margin_fraction)
        noz = frac * lev
        out.loc[override, "action"] = pred_action[override]
        out.loc[override, "side"] = side[override]
        out.loc[override, "notional_exposure"] = noz[override]
        out.loc[override, "leverage"] = lev[override]
        out.loc[override, "position_fraction"] = frac[override]
        out.loc[override, "take_profit"] = tp[override]
        out.loc[override, "stop_loss"] = sl[override]
        out.loc[override, "max_hold_bars"] = np.rint(mh[override]).astype(np.int64)
        out.loc[override, "cooldown_bars"] = np.rint(cd[override]).astype(np.int64)

    out.loc[:, "tft_v3_action"] = pred_action
    out.loc[:, "tft_v3_confidence"] = pred_conf
    out.loc[:, "tft_v3_quality"] = quality
    out.loc[:, "tft_v3_uncertainty"] = uncertainty
    return out


def _metrics(
    df: pd.DataFrame,
    q: np.ndarray,
    decisions: pd.DataFrame,
    parent: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    variant: v45.LayerVariant,
    base_cfg: dict[str, Any],
) -> dict[str, Any]:
    return {
        f"cost{m}": v45.backtest_variant(df, parent, jackpot_model, add_cfg, q, variant, fee=float(base_cfg["fee"]), slip=float(base_cfg["slip"]), cost_mult=float(m), decisions=decisions)
        for m in (1, 2, 3)
    }


def _score(metrics: dict[str, Any]) -> float:
    c1, c2, c3 = metrics["cost1"], metrics["cost2"], metrics["cost3"]
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.12 * c3["pnl"] - 4.5 * abs(c1["mdd"]))


def main() -> int:
    p = argparse.ArgumentParser(description="TFT v3 GRN+Time2Vec+dual objective parent guard/override backtest.")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=768)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(20260514)
    np.random.seed(20260514)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_ID}] device={device} epochs={args.epochs} stride={args.stride}", flush=True)

    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    cfg = FullyLearnedGovernorConfig(**dict(parent["config"]))
    feature_cols = list(parent.get("feature_cols") or [])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    print(f"[{MODEL_ID}] rows train={len(train_df)} val={len(val_df)} eval={len(eval_df)} features={len(feature_cols)}", flush=True)

    audit_base = _audit_contract(train_all, eval_df, feature_cols)
    train_teacher_dec = predict_policy_frame(parent, train_df, close=_close(train_df))
    val_teacher_dec = predict_policy_frame(parent, val_df, close=_close(val_df))
    eval_teacher_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))

    idx_train = base._candidate_indices(len(train_df), cfg, int(args.stride))
    train_pre = prepare_features(train_df, side_hint=0, close=_close(train_df), feature_cols=feature_cols)
    x_train_labels = train_pre.iloc[idx_train].reset_index(drop=True)
    x_train_norm, norm = base._normalise_fit(x_train_labels)
    train_full_norm = base._normalise_apply(train_pre, norm)
    x_train_seq = base._sequence_array(train_full_norm, idx_train)
    y_teacher_train = base._teacher_labels(train_teacher_dec, idx_train, cfg)
    _, y_gt_train, train_meta = build_training_set(train_df, cfg=cfg, stride_bars=int(args.stride), batch_size=512, feature_cols=feature_cols)

    idx_val = base._candidate_indices(len(val_df), cfg, max(3, int(args.stride)))
    val_pre = prepare_features(val_df, side_hint=0, close=_close(val_df), feature_cols=feature_cols)
    val_full_norm = base._normalise_apply(val_pre, norm)
    x_val_tab_ds = val_full_norm[idx_val]
    x_val_seq_ds = base._sequence_array(val_full_norm, idx_val)
    y_teacher_val = base._teacher_labels(val_teacher_dec, idx_val, cfg)
    _, y_gt_val, val_meta = build_training_set(val_df, cfg=cfg, stride_bars=max(3, int(args.stride)), batch_size=512, feature_cols=feature_cols)
    if len(y_gt_val["action"]) != len(x_val_tab_ds):
        keep = min(len(y_gt_val["action"]), len(x_val_tab_ds))
        x_val_tab_ds = x_val_tab_ds[:keep]
        x_val_seq_ds = x_val_seq_ds[:keep]
        y_teacher_val = {k: v[:keep] for k, v in y_teacher_val.items()}
        y_gt_val = {k: v[:keep] for k, v in y_gt_val.items()}

    train_ds = DualDataset(x_train_norm.to_numpy(dtype=np.float32), x_train_seq, y_teacher_train, y_gt_train)
    val_ds = DualDataset(x_val_tab_ds, x_val_seq_ds, y_teacher_val, y_gt_val)
    model = TFTV3Parent(len(feature_cols), cfg)
    training = _train_tft_v3(model, train_ds, val_ds, epochs=int(args.epochs), device=device, batch_size=int(args.batch_size))

    val_q = v31._predict_all(v27_model, val_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_pre = prepare_features(eval_df, side_hint=0, close=_close(eval_df), feature_cols=feature_cols)
    x_val_full = val_full_norm
    x_eval_full = base._normalise_apply(eval_pre, norm)
    val_outputs = base._predict_outputs(model, x_val_full, base._sequence_array(x_val_full, np.arange(len(val_df), dtype=np.int64)), device, int(args.batch_size), mc_passes=8)
    eval_outputs = base._predict_outputs(model, x_eval_full, base._sequence_array(x_eval_full, np.arange(len(eval_df), dtype=np.int64)), device, int(args.batch_size), mc_passes=8)

    variant = v45.LayerVariant("alpha1_tft_v3", "parent_tft_v3", base._overlay_alpha1())
    base_cfg = dict(parent["config"])
    val_rows: list[dict[str, Any]] = []
    selected: RuntimeV3 | None = None
    best_score = -1e18
    grid = _runtime_grid()
    if args.quick:
        grid = [
            r
            for r in grid
            if (r.mode == "veto" and r.veto_conf in (0.30, 0.46) and r.veto_q in (-0.015, 0.005))
            or (r.mode == "hybrid" and r.override_conf in (0.70,) and r.override_unc in (0.05,))
            or (r.mode == "replace" and r.veto_conf in (0.46, 0.62) and r.veto_q in (-0.005, 0.005) and r.veto_unc in (0.070,) and r.override_max_notional in (1.2, 2.0))
        ]
    for rt in grid:
        dec = _decision_frame(val_outputs, val_teacher_dec, cfg, rt, val_df.index)
        vm = _metrics(val_df, val_q, dec, parent, jackpot_model, add_cfg, variant, base_cfg)
        score = _score(vm)
        row = {
            **asdict(rt),
            "score": score,
            "val_pnl": vm["cost1"]["pnl"],
            "val_mdd": vm["cost1"]["mdd"],
            "val_trades": vm["cost1"]["trades"],
            "val_cost2_pnl": vm["cost2"]["pnl"],
            "val_cost3_pnl": vm["cost3"]["pnl"],
        }
        val_rows.append(row)
        if score > best_score:
            best_score = score
            selected = rt
            print(f"[{MODEL_ID}] new val best {rt.name} score={score:.2f} pnl={row['val_pnl']:.2f} mdd={row['val_mdd']:.2f}", flush=True)
    assert selected is not None

    baseline_metrics = _metrics(eval_df, eval_q, eval_teacher_dec, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments: list[dict[str, Any]] = [{"name": "alpha1_hgb_parent_baseline", "metrics": baseline_metrics, "score": _score(baseline_metrics)}]
    best_dec = _decision_frame(eval_outputs, eval_teacher_dec, cfg, selected, eval_df.index)
    best_metrics = _metrics(eval_df, eval_q, best_dec, parent, jackpot_model, add_cfg, variant, base_cfg)
    experiments.append({"name": f"tft_v3::{selected.name}", "selected_runtime": asdict(selected), "metrics": best_metrics, "score": _score(best_metrics)})

    print(f"[{MODEL_ID}] OOS selected={selected.name} cost1={best_metrics['cost1']['pnl']:.2f} mdd={best_metrics['cost1']['mdd']:.2f} cost2={best_metrics['cost2']['pnl']:.2f} cost3={best_metrics['cost3']['pnl']:.2f}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "feature_cols": feature_cols, "normalizer": norm, "config": base_cfg, "training": training, "selected_runtime": asdict(selected)}, OUT_DIR / "tft_v3_parent.pt")
    pd.DataFrame(val_rows).sort_values("score", ascending=False).to_csv(GRID_OUT, index=False)

    blocking = list(audit_base.get("blocking", []))
    warnings = list(audit_base.get("warnings", []))
    if best_metrics["cost1"]["pnl"] < baseline_metrics["cost1"]["pnl"]:
        warnings.append("tft_v3_cost1_below_alpha1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_metrics["cost1"]["mdd"] > baseline_metrics["cost1"]["mdd"] and best_metrics["cost1"]["pnl"] >= 0.75 * baseline_metrics["cost1"]["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS after selection",
        "design": "TFT v3 with GRN feature gate, Time2Vec positional encoding, dual HGB teacher + ground-truth objective, focal quality loss, MC-dropout uncertainty, and optional high-confidence override.",
        "train_meta": train_meta,
        "val_meta": val_meta,
        "base_audit": audit_base,
    }
    best = max(experiments, key=lambda e: e["score"])
    report = {
        "model_id": MODEL_ID,
        "selected": best,
        "experiments": experiments,
        "artifact_dir": str(OUT_DIR),
        "grid_path": str(GRID_OUT),
        "audit_path": str(AUDIT_OUT),
        "audit": audit,
    }
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(f"[{MODEL_ID}] report={REPORT_OUT}", flush=True)
    print(f"[{MODEL_ID}] audit={AUDIT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
