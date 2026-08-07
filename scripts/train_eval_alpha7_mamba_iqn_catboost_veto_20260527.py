#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from catboost import CatBoostClassifier
from mamba_ssm import Mamba
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.rebuild_alpha7_v2_only_live_20260526 import _rename_clean4_v2  # noqa: E402
from scripts.research_alpha_model_synergy_oos_20260525 import _parent_for_features  # noqa: E402
from scripts.train_eval_alpha5_regime4_tp18_sl10_no_teacher_no_deep_20260517 import (  # noqa: E402
    _compact_costs,
    _metrics,
    _score,
)
from scripts.train_eval_alpha7_iqn_fallback_20260527 import (  # noqa: E402
    _action_weights_from_targets,
    _apply_scaler,
    _feature_matrix,
    _fit_scaler,
    _quantile_huber_loss,
    _require_alpha7_features,
    _sample_tau,
    _sample_weights_from_targets,
)
from scripts.train_eval_alpha7_meta_fallback_cash_router_20260526 import (  # noqa: E402
    EVAL_CSV,
    TRAIN_CSV,
    _empty_dec_like,
    _json_default,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import _grid as _runner_grid  # noqa: E402


MODEL_ID = "alpha7_mamba_iqn_catboost_veto_20260527"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_mamba_iqn_catboost_veto_20260527"
ALPHA6_CURRENT_TAIL111_BUNDLE = (
    ROOT / "data/ensemble/supervised/alpha6_entry_quality_exit_5bucket_main_20260522/current_tail111_bundle.joblib"
)
ALPHA3_STATE24_V2_PARENT = (
    ROOT / "data/ensemble/supervised/alpha3_regime4_state24_v2_full_retrain_20260526/parent_state24_v2.pkl"
)
ALPHA3_STATE24_V2_PLUS_PRED_PARENT = (
    ROOT / "data/ensemble/supervised/alpha3_regime4_state24_v2_plus_pred_full_retrain_20260526/parent_state24_v2.pkl"
)
ALPHA3_UNAVAILABLE_AI_FEATURES = {
    "patchtst_pred",
    "patchtst_confidence",
    "ai_anchor_revert_prob",
    "ai_anchor_overheat",
    "ai_anchor_trend_escape_prob",
    "timesnet_cycle_sin",
    "timesnet_cycle_cos",
    "timesnet_cycle_delta",
}


def _load_feature_contract(source: str) -> tuple[list[str], dict[str, Any]]:
    if source == "alpha7_live":
        baseline = get_live_baseline()
        parent = joblib.load(baseline.primary_parent)
        return list(parent["feature_cols"]), {"source": source, "artifact": str(baseline.primary_parent)}
    if source == "alpha6_current_tail111":
        obj = joblib.load(ALPHA6_CURRENT_TAIL111_BUNDLE)
        return list(obj["feature_cols"]), {"source": source, "artifact": str(ALPHA6_CURRENT_TAIL111_BUNDLE)}
    if source == "alpha3_state24_v2":
        obj = joblib.load(ALPHA3_STATE24_V2_PARENT)
        return list(obj["feature_cols"]), {"source": source, "artifact": str(ALPHA3_STATE24_V2_PARENT)}
    if source == "alpha3_state24_v2_plus_pred":
        obj = joblib.load(ALPHA3_STATE24_V2_PLUS_PRED_PARENT)
        return list(obj["feature_cols"]), {"source": source, "artifact": str(ALPHA3_STATE24_V2_PLUS_PRED_PARENT)}
    if source == "alpha3_plus_pred_available94":
        obj = joblib.load(ALPHA3_STATE24_V2_PLUS_PRED_PARENT)
        cols = [c for c in list(obj["feature_cols"]) if c not in ALPHA3_UNAVAILABLE_AI_FEATURES]
        return cols, {
            "source": source,
            "artifact": str(ALPHA3_STATE24_V2_PLUS_PRED_PARENT),
            "dropped_from_alpha3_exact": sorted(ALPHA3_UNAVAILABLE_AI_FEATURES),
        }
    raise ValueError(f"unknown feature contract: {source!r}")


class SequenceTargetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        features: np.ndarray,
        indices: np.ndarray,
        targets: np.ndarray,
        sample_weights: np.ndarray,
        action_weights: np.ndarray,
        seq_len: int,
    ) -> None:
        self.features = features.astype(np.float32, copy=False)
        self.indices = indices.astype(np.int64, copy=False)
        self.targets = targets.astype(np.float32, copy=False)
        self.sample_weights = sample_weights.astype(np.float32, copy=False)
        self.action_weights = action_weights.astype(np.float32, copy=False)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return int(len(self.indices))

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        idx = int(self.indices[i])
        start = idx - self.seq_len + 1
        x = self.features[start : idx + 1]
        return (
            torch.from_numpy(x),
            torch.from_numpy(self.targets[i]),
            torch.tensor(self.sample_weights[i], dtype=torch.float32),
            torch.from_numpy(self.action_weights[i]),
        )


class MambaIQNNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 2,
        n_cos: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.hidden_dim = int(hidden_dim)
        self.n_cos = int(n_cos)
        self.proj = nn.Linear(int(input_dim), int(hidden_dim))
        self.blocks = nn.ModuleList(
            [
                Mamba(
                    d_model=int(hidden_dim),
                    d_state=int(d_state),
                    d_conv=int(d_conv),
                    expand=2,
                )
                for _ in range(int(n_layers))
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(int(hidden_dim)) for _ in range(int(n_layers))])
        self.state_norm = nn.LayerNorm(int(hidden_dim))
        self.quantile = nn.Sequential(nn.Linear(int(n_cos), int(hidden_dim)), nn.SiLU())
        self.head = nn.Sequential(
            nn.Linear(int(hidden_dim), 128),
            nn.ReLU(),
            nn.Linear(128, int(action_dim)),
        )

    def encode(self, x_seq: torch.Tensor) -> torch.Tensor:
        x = self.proj(x_seq)
        for block, norm in zip(self.blocks, self.norms):
            x = norm(x + block(x))
        return self.state_norm(x[:, -1, :])

    def forward(self, x_seq: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        state = self.encode(x_seq).unsqueeze(1)
        basis_idx = torch.arange(1, self.n_cos + 1, device=x_seq.device, dtype=x_seq.dtype).view(1, 1, -1)
        tau_basis = torch.cos(math.pi * tau.unsqueeze(-1) * basis_idx)
        tau_emb = self.quantile(tau_basis)
        return self.head(state * tau_emb)


def _simulate_targets_and_success(
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
) -> tuple[np.ndarray, np.ndarray]:
    open_px = pd.to_numeric(frame["open"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    high = pd.to_numeric(frame["high"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    low = pd.to_numeric(frame["low"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    close = pd.to_numeric(frame["close"], errors="coerce").ffill().to_numpy(dtype=np.float64)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    targets = np.zeros((len(indices), 3), dtype=np.float32)
    success = np.zeros((len(indices), 3), dtype=np.int8)
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
            tp_first = False
            for j in range(entry_i + 1, end_i + 1):
                if side > 0:
                    favorable_price = float(high[j] / max(entry, 1e-12) - 1.0)
                    adverse_price = float(low[j] / max(entry, 1e-12) - 1.0)
                else:
                    favorable_price = float(entry / max(low[j], 1e-12) - 1.0)
                    adverse_price = float(entry / max(high[j], 1e-12) - 1.0)
                max_dd = max(max_dd, max(0.0, -adverse_price * float(notional)))
                if adverse_price <= -float(margin_limit):
                    realized = -float(margin_limit) * float(notional)
                    liquidated = True
                    break
                if adverse_price <= -abs(float(sl)):
                    realized = -abs(float(sl)) * float(notional)
                    break
                if favorable_price >= float(tp):
                    realized = float(tp) * float(notional)
                    tp_first = True
                    break
            if realized is None:
                exit_px = float(close[end_i])
                exit_px = exit_px * (1.0 - slip_eff if side > 0 else 1.0 + slip_eff)
                raw = (exit_px - entry) / max(entry, 1e-12) if side > 0 else (entry - exit_px) / max(entry, 1e-12)
                realized = float(raw) * float(notional)
            reward = float(realized) - 2.0 * fee_eff * float(notional)
            if max_dd > float(margin_limit):
                reward -= float(dd_lambda) * (max_dd - float(margin_limit)) ** 2
            if liquidated:
                reward -= float(liquidation_penalty)
            targets[row_i, action] = float(np.clip(reward, -2.0, 2.0))
            success[row_i, action] = int((not liquidated) and reward > 2.0 * fee_eff * float(notional))
    return targets, success


def _train_mamba_iqn(
    x_all: np.ndarray,
    indices: np.ndarray,
    y_train: np.ndarray,
    *,
    seq_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    tau_samples: int,
    seed: int,
    tail_tau_mix: float,
    tail_tau_max: float,
    balanced_replay: bool,
    tail_sample_weight: float,
    cql_alpha: float,
    grad_clip: float,
) -> tuple[MambaIQNNet, dict[str, Any]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MambaIQNNet(input_dim=x_all.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    sample_w = _sample_weights_from_targets(
        y_train,
        tail_weight=float(tail_sample_weight),
        recent_mix_ratio=0.0,
        recent_window=0,
    )
    action_w = _action_weights_from_targets(y_train, noncash_best_boost=1.25, tail_action_boost=0.25)
    ds = SequenceTargetDataset(x_all, indices, y_train, sample_w, action_w, seq_len)
    sampler = (
        WeightedRandomSampler(weights=torch.from_numpy(sample_w.astype(np.float64)), num_samples=len(sample_w), replacement=True)
        if bool(balanced_replay)
        else None
    )
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=(sampler is None), sampler=sampler, drop_last=False, num_workers=0)
    losses: list[float] = []
    for epoch in range(int(epochs)):
        model.train()
        total = 0.0
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
            if float(cql_alpha) > 0.0:
                mean_q = pred.mean(dim=1)
                cash_best = (yb[:, 0] >= yb[:, 1:3].max(dim=1).values).float()
                risky_lse = torch.logsumexp(mean_q[:, 1:3], dim=1)
                loss = loss + float(cql_alpha) * (F.softplus(risky_lse - mean_q[:, 0]) * cash_best).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
            opt.step()
            total += float(loss.detach().cpu()) * len(xb)
            n += len(xb)
        losses.append(total / max(n, 1))
        print(f"[{MODEL_ID}] iqn epoch={epoch + 1}/{epochs} loss={losses[-1]:.6f}", flush=True)
    return model, {
        "device": str(device),
        "losses": losses,
        "best_action_counts": np.bincount(np.argmax(y_train, axis=1), minlength=y_train.shape[1]).astype(int).tolist(),
        "sample_weight_mean": float(np.mean(sample_w)),
        "sample_weight_p95": float(np.percentile(sample_w, 95)),
    }


def _mamba_iqn_scores(
    model: MambaIQNNet,
    x_all: np.ndarray,
    *,
    seq_len: int,
    risk_tau: float,
    num_tau: int,
    batch_size: int,
) -> np.ndarray:
    device = next(model.parameters()).device
    taus = torch.linspace(0.01, float(risk_tau), int(num_tau), device=device).view(1, -1)
    out = np.zeros((len(x_all), 3), dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(x_all, int(seq_len), axis=0)
    model.eval()
    with torch.no_grad():
        for start in range(int(seq_len) - 1, len(x_all), int(batch_size)):
            end = min(start + int(batch_size), len(x_all))
            w = windows[start - int(seq_len) + 1 : end - int(seq_len) + 1]
            seq = np.moveaxis(w, -1, 1).astype(np.float32, copy=True)
            xb = torch.from_numpy(seq).to(device)
            tb = taus.repeat(len(xb), 1)
            q = model(xb, tb).mean(dim=1)
            out[start:end] = q.detach().cpu().numpy().astype(np.float32)
    return out


def _build_decisions(
    template: pd.DataFrame,
    scores: np.ndarray,
    veto_prob: np.ndarray | None,
    *,
    seq_len: int,
    cvar_min: float,
    edge_min: float,
    veto_threshold: float,
    notional: float,
    leverage: float,
    tp: float,
    sl: float,
    max_hold: int,
    cooldown: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    out = _empty_dec_like(template)
    counts = {"cash": 0, "long": 0, "short": 0, "veto": 0}
    for i in range(int(seq_len) - 1, len(out)):
        row = scores[i]
        action = int(np.argmax(row))
        best = float(row[action])
        cash_score = float(row[0])
        if action == 0 or best < float(cvar_min) or (best - cash_score) < float(edge_min):
            counts["cash"] += 1
            continue
        if veto_prob is not None:
            veto_p = float(veto_prob[i, action]) if getattr(veto_prob, "ndim", 1) == 2 else float(veto_prob[i])
            if veto_p < float(veto_threshold):
                counts["veto"] += 1
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


def _selection_score(metrics: dict[str, Any]) -> float:
    c3 = metrics["cost3"]
    trades = int(c3.get("trades", 0))
    mdd = abs(float(c3.get("mdd", 0.0)))
    pnl = float(c3.get("pnl", 0.0))
    base = pnl / max(mdd, 1e-6)
    base += 0.05 * pnl
    if trades < 12:
        base -= (12 - trades) * 8.0
    if trades > 120:
        base -= (trades - 120) * 1.5
    if mdd > 35.0:
        base -= (mdd - 35.0) * 3.0
    if pnl < -20.0:
        base -= abs(pnl + 20.0) * 1.2
    return base


def _with_veto_context(base: pd.DataFrame, scores: np.ndarray, action: int | np.ndarray) -> pd.DataFrame:
    out = base.copy()
    action_arr = np.asarray(action, dtype=np.int64)
    if action_arr.ndim == 0:
        action_arr = np.full(len(out), int(action_arr), dtype=np.int64)
    if len(action_arr) != len(out):
        raise ValueError(f"veto action length mismatch: {len(action_arr)} != {len(out)}")
    idx = np.arange(len(out), dtype=np.int64)
    out["veto_side"] = np.where(action_arr == 1, 1.0, np.where(action_arr == 2, -1.0, 0.0))
    out["veto_is_long"] = (action_arr == 1).astype(float)
    out["veto_is_short"] = (action_arr == 2).astype(float)
    out["iqn_action_q"] = scores[idx, action_arr].astype(float)
    out["iqn_cash_q"] = scores[:, 0].astype(float)
    out["iqn_edge_q"] = (scores[idx, action_arr] - scores[:, 0]).astype(float)
    return out


def _side_veto_probs(cat: CatBoostClassifier, base: pd.DataFrame, scores: np.ndarray) -> np.ndarray:
    probs = np.zeros((len(base), 3), dtype=np.float32)
    for action in (1, 2):
        ctx = _with_veto_context(base, scores, action)
        probs[:, action] = cat.predict_proba(ctx)[:, 1].astype(np.float32)
    return probs


def _parse_float_list(raw: str) -> list[float]:
    vals = [float(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not vals:
        raise ValueError(f"empty float list: {raw!r}")
    return vals


def _eval_decisions(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    *,
    ref_parent: dict[str, Any],
    runner: dict[str, Any],
    runner_cfg: Any,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    return _compact_costs(
        _metrics(
            frame,
            parent_for_features=ref_parent,
            runner=runner,
            runner_cfg=runner_cfg,
            dec=dec,
            fee=fee,
            slip=slip,
        )
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Train Alpha7-feature Mamba-IQN policy with CatBoost veto meta-labeler.")
    ap.add_argument("--train-csv", type=Path, default=TRAIN_CSV)
    ap.add_argument("--eval-csv", type=Path, default=EVAL_CSV)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=52770)
    ap.add_argument("--seq-len", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=384)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--tau-samples", type=int, default=32)
    ap.add_argument("--risk-tau", type=float, default=0.25)
    ap.add_argument("--tail-tau-mix", type=float, default=0.50)
    ap.add_argument("--tail-tau-max", type=float, default=0.25)
    ap.add_argument("--balanced-replay", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--tail-sample-weight", type=float, default=1.0)
    ap.add_argument("--cql-alpha", type=float, default=0.02)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--notional", type=float, default=10.0)
    ap.add_argument("--notional-grid", default="0.25,0.5,1.0,2.0")
    ap.add_argument("--leverage", type=float, default=10.0)
    ap.add_argument("--take-profit", type=float, default=0.055)
    ap.add_argument("--stop-loss", type=float, default=0.050)
    ap.add_argument("--max-hold", type=int, default=12)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--margin-limit", type=float, default=0.065)
    ap.add_argument("--dd-lambda", type=float, default=4.0)
    ap.add_argument("--liquidation-penalty", type=float, default=0.75)
    ap.add_argument("--catboost-iterations", type=int, default=1000)
    ap.add_argument("--catboost-depth", type=int, default=6)
    ap.add_argument("--catboost-lr", type=float, default=0.03)
    ap.add_argument("--catboost-task-type", choices=["GPU", "CPU"], default="GPU")
    ap.add_argument("--veto-grid", default="0.55,0.65,0.75,0.85,0.90")
    ap.add_argument(
        "--feature-contract",
        choices=[
            "alpha7_live",
            "alpha6_current_tail111",
            "alpha3_state24_v2",
            "alpha3_state24_v2_plus_pred",
            "alpha3_plus_pred_available94",
        ],
        default="alpha7_live",
    )
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    random.seed(int(args.seed))

    if bool(args.smoke):
        args.epochs = min(int(args.epochs), 1)
        args.catboost_iterations = min(int(args.catboost_iterations), 80)

    train_all = _rename_clean4_v2(_read(args.train_csv))
    eval_df = _rename_clean4_v2(_read(args.eval_csv))
    train_df = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)

    baseline = get_live_baseline()
    primary_parent = joblib.load(baseline.primary_parent)
    feature_cols, feature_contract = _load_feature_contract(str(args.feature_contract))
    for frame_name, frame in (("train", train_df), ("val", val_df), ("eval", eval_df)):
        _require_alpha7_features(frame, feature_cols, name=frame_name)

    x_train_df = _feature_matrix(train_df, feature_cols)
    x_val_df = _feature_matrix(val_df, feature_cols)
    x_eval_df = _feature_matrix(eval_df, feature_cols)
    scaler = _fit_scaler(x_train_df)
    x_train = _apply_scaler(x_train_df, scaler)
    x_val = _apply_scaler(x_val_df, scaler)
    x_eval = _apply_scaler(x_eval_df, scaler)

    max_idx = len(train_df) - int(args.max_hold) - 3
    train_indices = np.arange(int(args.seq_len) - 1, max_idx, dtype=np.int64)
    if bool(args.smoke):
        train_indices = train_indices[:: max(1, len(train_indices) // 3000)]
    if len(train_indices) < 1000:
        raise RuntimeError(f"too few Mamba-IQN train rows: {len(train_indices)}")

    fee = float(primary_parent.get("config", {}).get("fee", 0.0004))
    slip = float(primary_parent.get("config", {}).get("slip", 0.00015))
    y_train, success_train = _simulate_targets_and_success(
        train_df,
        train_indices,
        notional=float(args.notional),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        margin_limit=float(args.margin_limit),
        dd_lambda=float(args.dd_lambda),
        liquidation_penalty=float(args.liquidation_penalty),
    )

    model, train_diag = _train_mamba_iqn(
        x_train,
        train_indices,
        y_train,
        seq_len=int(args.seq_len),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        tau_samples=int(args.tau_samples),
        seed=int(args.seed),
        tail_tau_mix=float(args.tail_tau_mix),
        tail_tau_max=float(args.tail_tau_max),
        balanced_replay=bool(args.balanced_replay),
        tail_sample_weight=float(args.tail_sample_weight),
        cql_alpha=float(args.cql_alpha),
        grad_clip=float(args.grad_clip),
    )

    train_scores = _mamba_iqn_scores(model, x_train, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    val_scores = _mamba_iqn_scores(model, x_val, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)
    eval_scores = _mamba_iqn_scores(model, x_eval, seq_len=int(args.seq_len), risk_tau=float(args.risk_tau), num_tau=32, batch_size=2048)

    train_action = np.argmax(train_scores[train_indices], axis=1).astype(np.int64)
    train_edge = np.max(train_scores[train_indices][:, 1:3], axis=1) - train_scores[train_indices][:, 0]
    min_veto_rows = 20 if bool(args.smoke) else 200
    veto_mask = np.zeros_like(train_action, dtype=bool)
    for q in (0.35, 0.20, 0.05, 0.0):
        edge_floor = float(np.quantile(train_edge, q))
        veto_mask = (train_action > 0) & (train_edge >= edge_floor)
        if int(veto_mask.sum()) >= min_veto_rows:
            break
    veto_indices = train_indices[veto_mask]
    veto_action = train_action[veto_mask]
    if len(veto_indices) < min_veto_rows:
        raise RuntimeError(f"too few CatBoost veto rows: {len(veto_indices)}")
    veto_y = success_train[veto_mask, veto_action].astype(np.int64)
    if len(np.unique(veto_y)) < 2:
        raise RuntimeError(f"CatBoost veto target is single class: {np.bincount(veto_y, minlength=2).tolist()}")

    static_train = _with_veto_context(
        x_train_df.iloc[veto_indices].reset_index(drop=True),
        train_scores[veto_indices],
        veto_action,
    )
    static_val = x_val_df.copy()
    static_eval = x_eval_df.copy()
    cat = CatBoostClassifier(
        loss_function="Logloss",
        iterations=int(args.catboost_iterations),
        depth=int(args.catboost_depth),
        learning_rate=float(args.catboost_lr),
        l2_leaf_reg=3.0,
        random_seed=int(args.seed),
        task_type=str(args.catboost_task_type),
        verbose=False,
        allow_writing_files=False,
    )
    try:
        cat.fit(static_train, veto_y)
    except Exception as exc:
        if str(args.catboost_task_type) != "GPU":
            raise
        print(f"[{MODEL_ID}] CatBoost GPU failed; retrying CPU: {exc}", flush=True)
        cat.set_params(task_type="CPU")
        cat.fit(static_train, veto_y)
    val_veto_prob = _side_veto_probs(cat, static_val, val_scores)
    eval_veto_prob = _side_veto_probs(cat, static_eval, eval_scores)

    ref_parent = _parent_for_features(list(joblib.load(v31.DEFAULT_PARENT)["feature_cols"]))
    fee_ref = float(joblib.load(v31.DEFAULT_PARENT)["config"]["fee"])
    slip_ref = float(joblib.load(v31.DEFAULT_PARENT)["config"]["slip"])
    noop_runner = joblib.load(v31.DEFAULT_JACKPOT)["cost_runner"]
    noop_cfg = next(c for c in _runner_grid() if c.name == "v21_2_parent_noop")

    edge_base = np.max(val_scores[int(args.seq_len) - 1 :, 1:3], axis=1) - val_scores[int(args.seq_len) - 1 :, 0]
    if bool(args.smoke):
        edge_grid = [0.0, float(np.quantile(edge_base, 0.70))]
        cvar_grid = [0.0]
        veto_grid = [0.55]
        notional_grid = [float(args.notional)]
    else:
        edge_grid = sorted({
            0.0,
            0.0020,
            max(0.0, float(np.quantile(edge_base, 0.97))),
        })
        cvar_grid = [-0.0100, -0.0050, 0.0]
        veto_grid = _parse_float_list(args.veto_grid)
        notional_grid = _parse_float_list(args.notional_grid)
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cvar_min in cvar_grid:
        for edge_min in edge_grid:
            for veto_threshold in veto_grid:
                for grid_notional in notional_grid:
                    val_dec, val_counts = _build_decisions(
                        val_df,
                        val_scores,
                        val_veto_prob,
                        seq_len=int(args.seq_len),
                        cvar_min=float(cvar_min),
                        edge_min=float(edge_min),
                        veto_threshold=float(veto_threshold),
                        notional=float(grid_notional),
                        leverage=float(args.leverage),
                        tp=float(args.take_profit),
                        sl=float(args.stop_loss),
                        max_hold=int(args.max_hold),
                        cooldown=int(args.cooldown),
                    )
                    val_metrics = _eval_decisions(val_df, val_dec, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee_ref, slip=slip_ref)
                    row = {
                        "cvar_min": float(cvar_min),
                        "edge_min": float(edge_min),
                        "veto_threshold": float(veto_threshold),
                        "notional": float(grid_notional),
                        "selection_score": float(_selection_score(val_metrics)),
                        "raw_selection_score": float(_score(val_metrics)),
                        "val_cost3_pnl": float(val_metrics["cost3"]["pnl"]),
                        "val_cost3_mdd": float(val_metrics["cost3"]["mdd"]),
                        "val_cost3_trades": int(val_metrics["cost3"]["trades"]),
                        "oos_cost3_pnl": np.nan,
                        "oos_cost3_mdd": np.nan,
                        "oos_cost3_trades": -1,
                        "oos_cost3_wr": np.nan,
                        "val_counts": val_counts,
                    }
                    rows.append(row)
                    if best is None or row["selection_score"] > best["selection_score"]:
                        best = row
    assert best is not None

    best_val_dec, best_val_counts = _build_decisions(
        val_df,
        val_scores,
        val_veto_prob,
        seq_len=int(args.seq_len),
        cvar_min=float(best["cvar_min"]),
        edge_min=float(best["edge_min"]),
        veto_threshold=float(best["veto_threshold"]),
        notional=float(best["notional"]),
        leverage=float(args.leverage),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        cooldown=int(args.cooldown),
    )
    best_eval_dec, best_eval_counts = _build_decisions(
        eval_df,
        eval_scores,
        eval_veto_prob,
        seq_len=int(args.seq_len),
        cvar_min=float(best["cvar_min"]),
        edge_min=float(best["edge_min"]),
        veto_threshold=float(best["veto_threshold"]),
        notional=float(best["notional"]),
        leverage=float(args.leverage),
        tp=float(args.take_profit),
        sl=float(args.stop_loss),
        max_hold=int(args.max_hold),
        cooldown=int(args.cooldown),
    )
    best_val_metrics = _eval_decisions(val_df, best_val_dec, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee_ref, slip=slip_ref)
    best_eval_metrics = _eval_decisions(eval_df, best_eval_dec, ref_parent=ref_parent, runner=noop_runner, runner_cfg=noop_cfg, fee=fee_ref, slip=slip_ref)

    model_path = args.out_dir / "mamba_iqn.pt"
    cat_path = args.out_dir / "catboost_veto.cbm"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "scaler": scaler,
            "network": {
                "input_dim": int(len(feature_cols)),
                "hidden_dim": 256,
                "n_layers": 2,
                "n_cos": 64,
                "action_dim": 3,
                "seq_len": int(args.seq_len),
            },
            "runtime": {
                "risk_tau": float(args.risk_tau),
                "cvar_min": float(best["cvar_min"]),
                "edge_min": float(best["edge_min"]),
                "veto_threshold": float(best["veto_threshold"]),
                "notional": float(best["notional"]),
                "leverage": float(args.leverage),
                "take_profit": float(args.take_profit),
                "stop_loss": float(args.stop_loss),
                "max_hold_bars": int(args.max_hold),
                "cooldown_bars": int(args.cooldown),
            },
        },
        model_path,
    )
    cat.save_model(str(cat_path))
    ranking_path = args.out_dir / "ranking.csv"
    pd.DataFrame(rows).sort_values(["selection_score", "oos_cost3_pnl"], ascending=[False, False]).to_csv(ranking_path, index=False)
    report = {
        "model_id": MODEL_ID,
        "design": (
            "Alpha7 93-feature sequence Mamba backbone + IQN lower-tail CVaR action selector + "
            "CatBoost current-bar veto meta-labeler. Selection uses 2025 Q4 only; 2026 is fixed OOS."
        ),
        "feature_contract": {
            **feature_contract,
            "feature_count": int(len(feature_cols)),
            "sequence_shape": [int(args.seq_len), int(len(feature_cols))],
            "static_shape": [int(len(feature_cols))],
            "feature_cols": feature_cols,
        },
        "training": {
            "train_rows": int(len(train_indices)),
            "veto_rows": int(len(veto_indices)),
            "veto_label_distribution": np.bincount(veto_y, minlength=2).astype(int).tolist(),
            "iqn_diag": train_diag,
        },
        "risk_contract": {
            "notional": float(args.notional),
            "notional_grid": [float(x) for x in notional_grid],
            "leverage": float(args.leverage),
            "take_profit": float(args.take_profit),
            "stop_loss": float(args.stop_loss),
            "max_hold_bars": int(args.max_hold),
            "margin_limit": float(args.margin_limit),
            "iqn_cvar_tau": float(args.risk_tau),
        },
        "best_by_selection": {
            **best,
            "val_metrics": best_val_metrics,
            "oos_metrics": best_eval_metrics,
            "best_val_counts": best_val_counts,
            "best_eval_counts": best_eval_counts,
        },
        "artifacts": {
            "mamba_iqn": str(model_path),
            "catboost_veto": str(cat_path),
            "ranking_csv": str(ranking_path),
        },
        "audit": {
            "selection_uses_2026": False,
            "selection_window": "2025-10-01..2025-12-31",
            "oos_window": "2026 fixed OOS",
            "legacy_clean_regime4_allowed": False,
            "catboost_static_current_bar_only": True,
        },
    }
    report_path = args.out_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "mamba_iqn": str(model_path),
                "catboost_veto": str(cat_path),
                "best_cvar_min": float(best["cvar_min"]),
                "best_edge_min": float(best["edge_min"]),
                "best_veto_threshold": float(best["veto_threshold"]),
                "best_notional": float(best["notional"]),
                "oos_cost3_pnl": float(best_eval_metrics["cost3"]["pnl"]),
                "oos_cost3_mdd": float(best_eval_metrics["cost3"]["mdd"]),
                "oos_cost3_trades": int(best_eval_metrics["cost3"]["trades"]),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
