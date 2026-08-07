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
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from catboost import CatBoostClassifier
except Exception as exc:  # pragma: no cover - fail-fast dependency contract
    raise RuntimeError("CatBoost is required for this experiment; install it in quant_ai.") from exc

from ensemble.fully_learned_governor_policy import ACTION_CASH, prepare_features  # noqa: E402
from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo_loop  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import backtest_signal_limit_exit_guard  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    _active,
    _apply_decision_mods,
    _augment_with_alpha7_features,
    _decision_sources,
    _default_limit_cfg,
    _guard,
    _load_stack,
    _merge_state24,
    _overlay,
    _score,
    _sl_ratio,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.train_eval_alpha7_iqn_fallback_20260527 import (  # noqa: E402
    _action_weights_from_targets,
    _apply_scaler,
    _fit_scaler,
    _quantile_huber_loss,
    _sample_tau,
    _sample_weights_from_targets,
    _simulate_action_targets,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "01965_tcn_iqn_catboost_fallback_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
RANKING_OUT = OUT_DIR / "ranking.csv"
DECONTAM_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"
DECONTAM_TRAIN_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2025_alpha6_current_tail111_exact.csv"
)
DECONTAM_EVAL_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2026_alpha6_current_tail111_exact.csv"
)

DERIVED_FEATURES = {
    "side_hint",
    "mom_21d",
    "abs_mom_21d",
    "mom_3d",
    "abs_mom_3d",
    "mom_1d",
    "abs_mom_1d",
}


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, scaled: np.ndarray, indices: np.ndarray, targets: np.ndarray, sample_w: np.ndarray, action_w: np.ndarray, seq_len: int) -> None:
        self.scaled = scaled.astype(np.float32, copy=False)
        self.indices = indices.astype(np.int64, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.sample_w = sample_w.astype(np.float32, copy=False)
        self.action_w = action_w.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        end = int(self.indices[idx]) + 1
        start = end - self.seq_len
        seq = self.scaled[start:end]
        return (
            torch.from_numpy(seq),
            torch.from_numpy(self.targets[idx]),
            torch.tensor(self.sample_w[idx], dtype=torch.float32),
            torch.from_numpy(self.action_w[idx]),
        )


class CausalConv1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_pad = int((kernel_size - 1) * dilation)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(x, (self.left_pad, 0)))


class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.norm1 = nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch)
        self.norm2 = nn.GroupNorm(8 if out_ch % 8 == 0 else 1, out_ch)
        self.drop = nn.Dropout(float(dropout))
        self.proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.proj(x)
        out = self.drop(F.silu(self.norm1(self.conv1(x))))
        out = self.drop(F.silu(self.norm2(self.conv2(out))))
        return F.silu(out + residual)


class TCNIQNNet(nn.Module):
    def __init__(self, feature_dim: int, action_dim: int = 3, hidden_dim: int = 256, n_cos: int = 64, dropout: float = 0.08) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.n_cos = int(n_cos)
        self.tcn = nn.Sequential(
            TCNBlock(feature_dim, 128, kernel_size=3, dilation=1, dropout=dropout),
            TCNBlock(128, 128, kernel_size=3, dilation=2, dropout=dropout),
            TCNBlock(128, hidden_dim, kernel_size=3, dilation=4, dropout=dropout),
        )
        self.state_norm = nn.LayerNorm(hidden_dim)
        self.quantile = nn.Sequential(nn.Linear(n_cos, hidden_dim), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def encode(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: [B, L, D] -> Conv1d: [B, D, L]. Latest closed bar is the rightmost timestep.
        x = seq.transpose(1, 2)
        h = self.tcn(x)[:, :, -1]
        return self.state_norm(h)

    def forward(self, seq: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        state = self.encode(seq).unsqueeze(1)
        basis_idx = torch.arange(1, self.n_cos + 1, device=seq.device, dtype=seq.dtype).view(1, 1, -1)
        tau_basis = torch.cos(math.pi * tau.unsqueeze(-1) * basis_idx)
        tau_emb = self.quantile(tau_basis)
        return self.head(state * tau_emb)


def _load_train_val_eval() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = _merge_state24(_read(v31.DEFAULT_TRAIN), alpha3_full.SIDE_CLEAN4_2025)
    eval_df = _merge_state24(_read(v31.DEFAULT_EVAL), alpha3_full.SIDE_CLEAN4_2026)
    a7_train = _rename_clean4_v2(_read(DECONTAM_TRAIN_CSV))
    a7_eval = _rename_clean4_v2(_read(DECONTAM_EVAL_CSV))
    train_all = _augment_with_alpha7_features(train_all, a7_train)
    eval_df = _augment_with_alpha7_features(eval_df, a7_eval)
    train_all["timestamp"] = pd.to_datetime(train_all["timestamp"], errors="raise")
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return train, val, eval_df.reset_index(drop=True)


def _patch_decontam_sources() -> None:
    combo_loop.ALPHA7_LIVE_DIR = DECONTAM_DIR
    combo_loop.PRIMARY_TRAIN_CSV = DECONTAM_TRAIN_CSV
    combo_loop.PRIMARY_EVAL_CSV = DECONTAM_EVAL_CSV
    combo_loop.PRIMARY_SUMMARY = DECONTAM_DIR / "primary_summary.json"
    combo_loop.FALLBACK_SUMMARY = DECONTAM_DIR / "fallback_alpha43_no_legacy_summary.json"


def _require_feature_contract(frame: pd.DataFrame, feature_cols: list[str], *, name: str) -> None:
    missing = [c for c in feature_cols if c not in frame.columns and c not in DERIVED_FEATURES]
    if missing:
        raise RuntimeError(f"{name}: TCN-IQN feature contract missing columns: {missing[:40]}")
    legacy = [c for c in feature_cols if str(c).startswith("clean_regime4_2024_unsup_v1_")]
    if legacy:
        raise RuntimeError(f"{name}: legacy clean regime features are not allowed: {legacy[:20]}")


def _feature_matrix(frame: pd.DataFrame, feature_cols: list[str], *, name: str) -> pd.DataFrame:
    _require_feature_contract(frame, feature_cols, name=name)
    feat = prepare_features(frame, side_hint=0, feature_cols=feature_cols)
    if list(feat.columns) != list(feature_cols):
        raise RuntimeError(f"{name}: prepare_features changed feature order/contract")
    return feat.replace([np.inf, -np.inf], np.nan)


def _eligible_primary_cash(primary_final: pd.DataFrame, seq_len: int, max_hold: int) -> np.ndarray:
    active = _active(primary_final).to_numpy(dtype=bool)
    idx = np.arange(len(primary_final), dtype=np.int64)
    return (~active) & (idx >= int(seq_len) - 1) & (idx < len(primary_final) - int(max_hold) - 3)


def _train_tcn_iqn(
    scaled_train: np.ndarray,
    train_indices: np.ndarray,
    y_train: np.ndarray,
    *,
    seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tau_samples: int,
    seed: int,
) -> tuple[TCNIQNNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sample_w = _sample_weights_from_targets(y_train, tail_weight=1.25, recent_mix_ratio=0.25, recent_window=60000)
    action_w = _action_weights_from_targets(y_train, noncash_best_boost=1.5, tail_action_boost=0.35)
    ds = SequenceDataset(scaled_train, train_indices, y_train, sample_w, action_w, seq_len)
    sampler = WeightedRandomSampler(torch.from_numpy(sample_w.astype(np.float64)), num_samples=len(sample_w), replacement=True)
    dl = DataLoader(ds, batch_size=int(batch_size), sampler=sampler, drop_last=False, num_workers=0, pin_memory=torch.cuda.is_available())
    model = TCNIQNNet(feature_dim=scaled_train.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.6, patience=2, min_lr=1e-5)
    losses: list[float] = []
    aux_losses: list[dict[str, float]] = []
    redo_reset_neurons = 0
    for _epoch in range(int(epochs)):
        model.train()
        total = 0.0
        total_cql = 0.0
        total_af = 0.0
        count = 0
        for seq, yb, sw, aw in dl:
            seq = seq.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            sw = sw.to(device, non_blocking=True)
            aw = aw.to(device, non_blocking=True)
            tau = _sample_tau(len(seq), int(tau_samples), device, seq.dtype, tail_mix=0.50, tail_max=0.25)
            pred = model(seq, tau)
            loss = _quantile_huber_loss(pred, yb, tau, sample_weight=sw, action_weight=aw)
            mean_q = pred.mean(dim=1)
            cash_best = (yb[:, 0] >= yb[:, 1:3].max(dim=1).values).float()
            cql_pen = (F.softplus(torch.logsumexp(mean_q[:, 1:3], dim=1) - mean_q[:, 0]) * cash_best).mean()
            loss = loss + 0.025 * cql_pen
            target_edge = yb[:, 1:3].max(dim=1).values - yb[:, 0]
            edge_mask = target_edge > 0.002
            anti_flat_pen = torch.zeros((), device=device)
            if bool(edge_mask.any()):
                pred_edge = mean_q[:, 1:3].max(dim=1).values - mean_q[:, 0]
                anti_flat_pen = F.relu(0.002 - pred_edge[edge_mask]).mean()
                loss = loss + 0.06 * anti_flat_pen
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(seq)
            total_cql += float(cql_pen.detach().cpu()) * len(seq)
            total_af += float(anti_flat_pen.detach().cpu()) * len(seq)
            count += len(seq)
        epoch_loss = total / max(count, 1)
        losses.append(epoch_loss)
        aux_losses.append(
            {
                "cql_pen": total_cql / max(count, 1),
                "anti_flat_pen": total_af / max(count, 1),
                "lr": float(opt.param_groups[0]["lr"]),
            }
        )
        scheduler.step(epoch_loss)
        if (_epoch + 1) % 4 == 0:
            redo_reset_neurons += _redo_rejuvenate_linear(model, tau=5e-3, ratio=0.05)
    return model, {
        "device": str(device),
        "losses": losses,
        "aux_losses": aux_losses,
        "best_action_counts": np.bincount(np.argmax(y_train, axis=1), minlength=3).astype(int).tolist(),
        "sample_weight_mean": float(np.mean(sample_w)),
        "sample_weight_p95": float(np.percentile(sample_w, 95)),
        "techniques": [
            "weighted balanced replay",
            "tail-weighted samples",
            "action weighting",
            "tail-biased tau sampling",
            "CQL cash conservatism",
            "anti-flat edge penalty",
            "linear REDO rejuvenation",
        ],
        "redo_reset_neurons": int(redo_reset_neurons),
    }


def _redo_rejuvenate_linear(model: nn.Module, *, tau: float, ratio: float) -> int:
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


def _iqn_scores(model: TCNIQNNet, scaled: np.ndarray, *, seq_len: int, risk_tau: float, num_tau: int, batch_size: int) -> np.ndarray:
    device = next(model.parameters()).device
    out = np.zeros((len(scaled), 3), dtype=np.float32)
    taus = torch.linspace(0.01, float(risk_tau), int(num_tau), device=device).view(1, -1)
    model.eval()
    with torch.no_grad():
        for start in range(int(seq_len) - 1, len(scaled), int(batch_size)):
            end = min(len(scaled), start + int(batch_size))
            indices = np.arange(start, end, dtype=np.int64)
            seq = np.stack([scaled[i - int(seq_len) + 1 : i + 1] for i in indices]).astype(np.float32)
            xb = torch.from_numpy(seq).to(device)
            tb = taus.repeat(len(xb), 1)
            out[start:end] = model(xb, tb).mean(dim=1).detach().cpu().numpy().astype(np.float32)
    return out


def _success_label_for_side(frame: pd.DataFrame, indices: np.ndarray, *, side: int, notional: float, tp: float, sl: float, max_hold: int, margin_limit: float) -> np.ndarray:
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    labels = np.zeros(len(indices), dtype=np.int64)
    for row_i, idx_raw in enumerate(indices):
        entry_i = min(int(idx_raw) + 1, len(frame) - 1)
        if entry_i >= len(frame) - 1:
            continue
        entry = float(open_px[entry_i])
        end_i = min(entry_i + int(max_hold), len(frame) - 1)
        for j in range(entry_i + 1, end_i + 1):
            if side > 0:
                favorable = float(high[j] / max(entry, 1e-12) - 1.0) * float(notional)
                adverse = float(low[j] / max(entry, 1e-12) - 1.0) * float(notional)
            else:
                favorable = float(entry / max(low[j], 1e-12) - 1.0) * float(notional)
                adverse = float(entry / max(high[j], 1e-12) - 1.0) * float(notional)
            if adverse <= -float(margin_limit) or adverse <= -abs(float(sl)):
                break
            if favorable >= float(tp):
                labels[row_i] = 1
                break
    return labels


def _fit_side_catboost(x_raw: pd.DataFrame, indices: np.ndarray, labels: np.ndarray, *, seed: int, task_type: str) -> CatBoostClassifier:
    x = x_raw.iloc[indices].copy()
    if len(np.unique(labels)) < 2:
        raise RuntimeError("CatBoost meta-label has a single class; adjust risk contract or train window")
    model = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="Logloss",
        iterations=220,
        depth=5,
        learning_rate=0.045,
        l2_leaf_reg=8.0,
        random_seed=int(seed),
        verbose=False,
        allow_writing_files=False,
        task_type=str(task_type),
    )
    model.fit(x, labels.astype(int))
    return model


def _cat_probs(model: CatBoostClassifier, x_raw: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(x_raw)[:, 1].astype(np.float32)


def _empty_dec_like(template: pd.DataFrame) -> pd.DataFrame:
    out = template.copy().reset_index(drop=True)
    out.loc[:, ["action", "side"]] = 0
    for col in ["notional_exposure", "position_fraction", "quality_score", "confidence"]:
        if col in out.columns:
            out[col] = 0.0
    return out


def _build_replacement_decisions(
    primary_final: pd.DataFrame,
    fallback_template: pd.DataFrame,
    scores: np.ndarray,
    long_p: np.ndarray,
    short_p: np.ndarray,
    *,
    allowed_mask: np.ndarray,
    seq_len: int,
    cvar_min: float,
    edge_min: float,
    cat_min: float,
    notional: float,
    leverage: float,
    tp: float,
    sl: float,
    max_hold: int,
    cooldown: int,
    risk_source: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_dec_like(primary_final)
    primary_cash = ~_active(primary_final).to_numpy(dtype=bool)
    allowed = np.asarray(allowed_mask, dtype=bool)
    if len(allowed) != len(out):
        raise RuntimeError(f"allowed_mask length mismatch: {len(allowed)} vs {len(out)}")
    fallback_template = fallback_template.reset_index(drop=True)
    counts = {"cash": 0, "long": 0, "short": 0, "cat_veto": 0, "iqn_veto": 0, "seq_warmup": 0, "outside_scope": 0}
    for i in range(len(out)):
        if not primary_cash[i]:
            counts["cash"] += 1
            continue
        if not allowed[i]:
            counts["outside_scope"] += 1
            continue
        if i < int(seq_len) - 1:
            counts["seq_warmup"] += 1
            continue
        row = scores[i]
        action = int(np.argmax(row))
        best = float(row[action])
        cash_score = float(row[0])
        if action == 0 or best < float(cvar_min) or (best - cash_score) < float(edge_min):
            counts["iqn_veto"] += 1
            continue
        side = 1 if action == 1 else -1
        p_success = float(long_p[i] if side > 0 else short_p[i])
        if p_success < float(cat_min):
            counts["cat_veto"] += 1
            continue
        out.at[i, "action"] = int(action)
        out.at[i, "side"] = int(side)
        if str(risk_source) == "existing_fallback":
            for col in ["notional_exposure", "leverage", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]:
                if col not in fallback_template.columns:
                    raise RuntimeError(f"fallback risk template missing required column: {col}")
                out.at[i, col] = fallback_template.at[i, col]
        elif str(risk_source) == "fixed":
            out.at[i, "notional_exposure"] = float(notional)
            out.at[i, "leverage"] = float(leverage)
            out.at[i, "position_fraction"] = float(min(float(notional) / max(float(leverage), 1e-12), 1.0))
            out.at[i, "take_profit"] = float(tp)
            out.at[i, "stop_loss"] = float(sl)
            out.at[i, "max_hold_bars"] = int(max_hold)
            out.at[i, "cooldown_bars"] = int(cooldown)
        else:
            raise RuntimeError(f"unknown risk_source: {risk_source}")
        out.at[i, "quality_score"] = float(best - cash_score)
        out.at[i, "confidence"] = float(p_success)
        counts["long" if side > 0 else "short"] += 1
    return out, counts


def _combine_primary_with_replacement(primary_final: pd.DataFrame, replacement: pd.DataFrame) -> pd.DataFrame:
    out = primary_final.copy().reset_index(drop=True)
    replacement = replacement.reset_index(drop=True)
    mask = (~_active(out)) & _active(replacement)
    for col in replacement.columns:
        if col in out.columns:
            out.loc[mask, col] = replacement.loc[mask, col].to_numpy()
    return out


def _eval_final_dec(df: pd.DataFrame, q: np.ndarray, dec: pd.DataFrame, stack: dict[str, Any], cfg: dict[str, Any], *, cost_mult: int) -> dict[str, Any]:
    return backtest_signal_limit_exit_guard(
        df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        dec.reset_index(drop=True),
        _overlay(stack["overlay"], cfg),
        _default_limit_cfg(),
        _guard(cfg),
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=float(cost_mult),
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Test TCN-IQN-CatBoost as 01965 fallback replacement.")
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--seed", type=int, default=20260528)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--tau-samples", type=int, default=32)
    ap.add_argument("--risk-tau", type=float, default=0.25)
    ap.add_argument("--notional", type=float, default=2.0)
    ap.add_argument("--leverage", type=float, default=2.0)
    ap.add_argument("--take-profit", type=float, default=0.060)
    ap.add_argument("--stop-loss", type=float, default=0.045)
    ap.add_argument("--max-hold", type=int, default=96)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--margin-limit", type=float, default=0.12)
    ap.add_argument("--min-val-replacement-rows", type=int, default=40)
    ap.add_argument("--catboost-task-type", choices=["CPU", "GPU"], default="CPU")
    ap.add_argument("--grid-profile", choices=["smoke", "standard", "full"], default="standard")
    ap.add_argument("--replacement-scope", choices=["existing_fallback_active", "primary_cash_region"], default="existing_fallback_active")
    ap.add_argument("--risk-source", choices=["existing_fallback", "fixed"], default="existing_fallback")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))

    cfg = _cfg_from_results()
    if cfg.get("source") != "alpha7_combo_primary_fallback":
        raise RuntimeError(f"01965 source contract changed: {cfg.get('source')}")

    _patch_decontam_sources()
    stack = _load_stack()
    train_df, val_df, eval_df = _load_train_val_eval()
    print(json.dumps({"stage": "frames_loaded", "train_rows": len(train_df), "val_rows": len(val_df), "oos_rows": len(eval_df)}), flush=True)
    sources_train = _decision_sources(train_df, train_df, stack["parent"])
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    print(json.dumps({"stage": "decision_sources_ready"}), flush=True)

    primary_train_final = _apply_decision_mods(sources_train["alpha7_primary"][0], cfg)
    primary_val_final = _apply_decision_mods(sources["alpha7_primary"][0], cfg)
    primary_eval_final = _apply_decision_mods(sources["alpha7_primary"][1], cfg)
    baseline_val_final = _apply_decision_mods(sources["alpha7_combo_primary_fallback"][0], cfg)
    baseline_eval_final = _apply_decision_mods(sources["alpha7_combo_primary_fallback"][1], cfg)
    if str(args.replacement_scope) == "existing_fallback_active":
        allowed_val = ((~_active(primary_val_final)) & _active(baseline_val_final)).to_numpy(dtype=bool)
        allowed_eval = ((~_active(primary_eval_final)) & _active(baseline_eval_final)).to_numpy(dtype=bool)
    elif str(args.replacement_scope) == "primary_cash_region":
        allowed_val = (~_active(primary_val_final)).to_numpy(dtype=bool)
        allowed_eval = (~_active(primary_eval_final)).to_numpy(dtype=bool)
    else:
        raise RuntimeError(f"unknown replacement_scope: {args.replacement_scope}")

    feature_cols = list(joblib.load(DECONTAM_DIR / "primary_parent.pkl")["feature_cols"])
    for name, frame in (("train", train_df), ("val", val_df), ("eval", eval_df)):
        _require_feature_contract(frame, feature_cols, name=name)

    x_train_df = _feature_matrix(train_df, feature_cols, name="train")
    x_val_df = _feature_matrix(val_df, feature_cols, name="val")
    x_eval_df = _feature_matrix(eval_df, feature_cols, name="eval")
    train_eligible = _eligible_primary_cash(primary_train_final, int(args.seq_len), int(args.max_hold))
    train_indices = np.flatnonzero(train_eligible).astype(np.int64)
    if len(train_indices) < 1000:
        raise RuntimeError(f"too few primary-CASH training rows: {len(train_indices)}")
    print(json.dumps({"stage": "training_set_ready", "primary_cash_train_rows": int(len(train_indices)), "feature_count": int(len(feature_cols))}), flush=True)
    scaler = _fit_scaler(x_train_df.iloc[train_indices].reset_index(drop=True))
    x_train = _apply_scaler(x_train_df, scaler)
    x_val = _apply_scaler(x_val_df, scaler)
    x_eval = _apply_scaler(x_eval_df, scaler)

    y_train = _simulate_action_targets(
        train_df,
        train_indices,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        fee=float(stack["fee"]),
        slip=float(stack["slip"]),
        cost_mult=3.0,
        margin_limit=float(args.margin_limit),
        dd_lambda=4.0,
        liquidation_penalty=0.75,
        entry_hurdle=0.0,
        theta_penalty=0.0005,
    )
    model, train_diag = _train_tcn_iqn(
        x_train,
        train_indices,
        y_train,
        seq_len=int(args.seq_len),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        tau_samples=int(args.tau_samples),
        seed=int(args.seed),
    )
    print(json.dumps({"stage": "tcn_iqn_trained", "device": train_diag["device"], "last_loss": train_diag["losses"][-1]}), flush=True)

    long_labels = _success_label_for_side(
        train_df,
        train_indices,
        side=1,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        margin_limit=float(args.margin_limit),
    )
    short_labels = _success_label_for_side(
        train_df,
        train_indices,
        side=-1,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        margin_limit=float(args.margin_limit),
    )
    cat_long = _fit_side_catboost(x_train_df, train_indices, long_labels, seed=int(args.seed) + 11, task_type=str(args.catboost_task_type))
    cat_short = _fit_side_catboost(x_train_df, train_indices, short_labels, seed=int(args.seed) + 29, task_type=str(args.catboost_task_type))
    print(json.dumps({"stage": "catboost_trained", "long_success_rate": float(np.mean(long_labels)), "short_success_rate": float(np.mean(short_labels))}), flush=True)

    val_scores = _iqn_scores(model, x_val, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    eval_scores = _iqn_scores(model, x_eval, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    val_long_p = _cat_probs(cat_long, x_val_df)
    val_short_p = _cat_probs(cat_short, x_val_df)
    eval_long_p = _cat_probs(cat_long, x_eval_df)
    eval_short_p = _cat_probs(cat_short, x_eval_df)

    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    baseline_val = {f"cost{c}": _eval_final_dec(val_df, val_q, baseline_val_final, stack, cfg, cost_mult=c) for c in (1, 2, 3)}
    baseline_eval = {f"cost{c}": _eval_final_dec(eval_df, eval_q, baseline_eval_final, stack, cfg, cost_mult=c) for c in (1, 2, 3)}
    print(json.dumps({"stage": "baseline_evaluated", "baseline_oos_cost3_pnl": baseline_eval["cost3"]["pnl"]}), flush=True)

    active_val = val_scores[(~_active(primary_val_final).to_numpy(dtype=bool)) & (np.arange(len(val_scores)) >= int(args.seq_len) - 1)]
    edge_basis = active_val[:, 1:3].max(axis=1) - active_val[:, 0] if len(active_val) else np.zeros(0, dtype=np.float32)
    best_basis = active_val[:, 1:3].max(axis=1) if len(active_val) else np.zeros(0, dtype=np.float32)
    if str(args.grid_profile) == "smoke":
        edge_grid = sorted({0.0, 0.005, *[float(np.quantile(edge_basis, q)) for q in (0.85,) if len(edge_basis)]})
        cvar_grid = sorted({0.0, 0.005, *[float(np.quantile(best_basis, q)) for q in (0.85,) if len(best_basis)]})
        cat_grid = [0.55, 0.62]
    elif str(args.grid_profile) == "standard":
        edge_grid = sorted({0.0, 0.002, 0.010, *[float(np.quantile(edge_basis, q)) for q in (0.80, 0.90, 0.96) if len(edge_basis)]})
        cvar_grid = sorted({-0.005, 0.0, 0.010, *[float(np.quantile(best_basis, q)) for q in (0.80, 0.90, 0.96) if len(best_basis)]})
        cat_grid = [0.52, 0.58, 0.64, 0.70]
    else:
        edge_grid = sorted({0.0, 0.002, 0.005, 0.010, *[float(np.quantile(edge_basis, q)) for q in (0.70, 0.80, 0.90, 0.95, 0.98) if len(edge_basis)]})
        cvar_grid = sorted({-0.01, 0.0, 0.005, 0.010, *[float(np.quantile(best_basis, q)) for q in (0.70, 0.80, 0.90, 0.95, 0.98) if len(best_basis)]})
        cat_grid = [0.50, 0.55, 0.58, 0.62, 0.66, 0.70]
    print(json.dumps({"stage": "grid_ready", "profile": str(args.grid_profile), "variants": int(len(edge_grid) * len(cvar_grid) * len(cat_grid))}), flush=True)

    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_payload: tuple[pd.DataFrame, dict[str, int]] | None = None
    for cvar_min in cvar_grid:
        for edge_min in edge_grid:
            for cat_min in cat_grid:
                val_fb, val_counts = _build_replacement_decisions(
                    primary_val_final,
                    baseline_val_final,
                    val_scores,
                    val_long_p,
                    val_short_p,
                    allowed_mask=allowed_val,
                    seq_len=int(args.seq_len),
                    cvar_min=float(cvar_min),
                    edge_min=float(edge_min),
                    cat_min=float(cat_min),
                    notional=float(args.notional),
                    leverage=float(args.leverage),
                    tp=float(args.take_profit),
                    sl=float(args.stop_loss),
                    max_hold=int(args.max_hold),
                    cooldown=int(args.cooldown),
                    risk_source=str(args.risk_source),
                )
                val_dec = _combine_primary_with_replacement(primary_val_final, val_fb)
                val_c3 = _eval_final_dec(val_df, val_q, val_dec, stack, cfg, cost_mult=3)
                replaced_val_rows = int(val_counts["long"] + val_counts["short"])
                if replaced_val_rows < int(args.min_val_replacement_rows):
                    selection_score = -1e9 + float(val_c3["pnl"])
                else:
                    selection_score = _score(val_c3)

                row = {
                    "cvar_min": float(cvar_min),
                    "edge_min": float(edge_min),
                    "cat_min": float(cat_min),
                    "selection_score": float(selection_score),
                    "val_cost3_pnl": float(val_c3["pnl"]),
                    "val_cost3_mdd": float(val_c3["mdd"]),
                    "val_cost3_wr": float(val_c3["wr"]),
                    "val_cost3_trades": int(val_c3["trades"]),
                    "val_sl_ratio": float(_sl_ratio(val_c3)),
                    "val_replacement_rows": replaced_val_rows,
                    "val_counts": json.dumps(val_counts, sort_keys=True),
                }
                rows.append(row)
                if best_row is None or float(row["selection_score"]) > float(best_row["selection_score"]):
                    best_row = row
                    best_payload = (val_dec, val_counts)

    viable_rows = [r for r in rows if int(r["val_replacement_rows"]) >= int(args.min_val_replacement_rows)]
    if not viable_rows:
        ranking_out = args.out_dir / "ranking.csv"
        pd.DataFrame(rows).sort_values(["selection_score", "val_cost3_pnl"], ascending=[False, False]).to_csv(ranking_out, index=False)
        raise RuntimeError(
            f"no viable TCN-IQN-CatBoost fallback variants met min_val_replacement_rows="
            f"{int(args.min_val_replacement_rows)}; ranking written to {ranking_out}"
        )
    if best_row is None or best_payload is None:
        raise RuntimeError("no TCN-IQN-CatBoost variant evaluated")
    print(json.dumps({"stage": "validation_grid_done", "variants": len(rows), "best": {k: best_row[k] for k in ["cvar_min", "edge_min", "cat_min", "val_cost3_pnl"]}}), flush=True)
    best_val_dec, best_val_counts = best_payload
    best_eval_fb, best_eval_counts = _build_replacement_decisions(
        primary_eval_final,
        baseline_eval_final,
        eval_scores,
        eval_long_p,
        eval_short_p,
        allowed_mask=allowed_eval,
        seq_len=int(args.seq_len),
        cvar_min=float(best_row["cvar_min"]),
        edge_min=float(best_row["edge_min"]),
        cat_min=float(best_row["cat_min"]),
        notional=float(args.notional),
        leverage=float(args.leverage),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        cooldown=int(args.cooldown),
        risk_source=str(args.risk_source),
    )
    best_eval_dec = _combine_primary_with_replacement(primary_eval_final, best_eval_fb)
    best_val = {f"cost{c}": _eval_final_dec(val_df, val_q, best_val_dec, stack, cfg, cost_mult=c) for c in (1, 2, 3)}
    best_eval = {f"cost{c}": _eval_final_dec(eval_df, eval_q, best_eval_dec, stack, cfg, cost_mult=c) for c in (1, 2, 3)}
    best_row.update(
        {
            "oos_cost3_pnl": float(best_eval["cost3"]["pnl"]),
            "oos_cost3_mdd": float(best_eval["cost3"]["mdd"]),
            "oos_cost3_wr": float(best_eval["cost3"]["wr"]),
            "oos_cost3_trades": int(best_eval["cost3"]["trades"]),
            "oos_sl_ratio": float(_sl_ratio(best_eval["cost3"])),
            "oos_replacement_rows": int(best_eval_counts["long"] + best_eval_counts["short"]),
            "delta_vs_01965_oos_cost3_pnl": float(best_eval["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
            "oos_counts": json.dumps(best_eval_counts, sort_keys=True),
        }
    )

    model_path = args.out_dir / "tcn_iqn_fallback.pt"
    cat_long_path = args.out_dir / "catboost_long_success.cbm"
    cat_short_path = args.out_dir / "catboost_short_success.cbm"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler": scaler,
            "network": {"seq_len": int(args.seq_len), "feature_dim": int(len(feature_cols)), "hidden_dim": 256, "n_cos": 64, "action_dim": 3},
            "runtime": {
                "internal_action_contract": {"0": "CASH", "1": "LONG", "2": "SHORT"},
                "risk_tau": float(args.risk_tau),
                "cvar_min": float(best_row["cvar_min"]),
                "edge_min": float(best_row["edge_min"]),
                "cat_min": float(best_row["cat_min"]),
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
    cat_long.save_model(cat_long_path)
    cat_short.save_model(cat_short_path)

    ranking_out = args.out_dir / "ranking.csv"
    summary_out = args.out_dir / "summary.json"
    pd.DataFrame(rows).sort_values(["selection_score", "val_cost3_pnl"], ascending=[False, False]).to_csv(ranking_out, index=False)
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "design": (
            "TCN-IQN-CatBoost fallback replacement for 01965. Alpha7 primary rows are unchanged. "
            "When primary is CASH, TCN-IQN chooses CASH/LONG/SHORT by lower-tail CVaR and side-specific CatBoost success models can veto the entry."
        ),
        "contract": {
            "replacement_point": "alpha7_combo_primary_fallback fallback leg only",
            "replacement_scope": str(args.replacement_scope),
            "risk_source": str(args.risk_source),
            "primary_unchanged": True,
            "deep_scout_unchanged": True,
            "feature_source": "alpha7_v2_only_live primary_parent feature_cols",
            "feature_count": int(len(feature_cols)),
            "seq_shape": [int(args.seq_len), int(len(feature_cols))],
            "internal_action_contract": {"0": "CASH", "1": "LONG", "2": "SHORT"},
            "catboost_contract": "two side-specific static 93-feature classifiers: long_success, short_success",
            "feature_contract_fail_fast": True,
        },
        "risk_contract": {
            "notional": float(args.notional),
            "leverage": float(args.leverage),
            "take_profit": float(args.take_profit),
            "stop_loss": float(args.stop_loss),
            "max_hold_bars": int(args.max_hold),
            "margin_limit": float(args.margin_limit),
            "risk_tau": float(args.risk_tau),
        },
        "training": {
            "train_rows": int(len(train_df)),
            "primary_cash_train_rows": int(len(train_indices)),
            "target_reward_mean": y_train.mean(axis=0).tolist(),
            "target_reward_p05": np.quantile(y_train, 0.05, axis=0).tolist(),
            "target_reward_p50": np.quantile(y_train, 0.50, axis=0).tolist(),
            "target_reward_p95": np.quantile(y_train, 0.95, axis=0).tolist(),
            "long_success_rate": float(np.mean(long_labels)),
            "short_success_rate": float(np.mean(short_labels)),
            "train_diag": train_diag,
        },
        "baseline_01965": {"val": baseline_val, "oos": baseline_eval},
        "best_by_validation": {
            **best_row,
            "val_metrics": best_val,
            "oos_metrics": best_eval,
            "val_counts": best_val_counts,
            "oos_counts": best_eval_counts,
        },
        "artifacts": {
            "tcn_iqn": str(model_path),
            "catboost_long": str(cat_long_path),
            "catboost_short": str(cat_short_path),
            "ranking_csv": str(ranking_out),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "legacy_clean_regime4_allowed": False,
            "live_path_modified": False,
        },
    }
    summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_out),
                "ranking": str(ranking_out),
                "baseline_oos_cost3": baseline_eval["cost3"],
                "best_oos_cost3": best_eval["cost3"],
                "delta_oos_cost3_pnl": float(best_eval["cost3"]["pnl"]) - float(baseline_eval["cost3"]["pnl"]),
                "best_cfg": {k: best_row[k] for k in ["cvar_min", "edge_min", "cat_min"]},
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
