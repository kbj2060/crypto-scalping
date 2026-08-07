#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_contextual_dynamic_exit_scout_v30_20260511"
DEFAULT_PARENT = v23.DEFAULT_PARENT
DEFAULT_JACKPOT = v23.DEFAULT_JACKPOT
DEFAULT_TRAIN = v23.DEFAULT_TRAIN
DEFAULT_EVAL = v23.DEFAULT_EVAL
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_contextual_dynamic_exit_scout_v30_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_contextual_dynamic_exit_scout_v30_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_contextual_dynamic_exit_scout_v30_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_contextual_dynamic_exit_scout_v30_20260511_grid.csv"
SEQ_LEN = 72
HOLD_BUCKETS = (12, 24, 48)
V27_COST1 = 226.82447187089713
V27_COST2 = 123.11659362616143
V27_COST3 = 14.22783363158393


@dataclass(frozen=True)
class DeepAlphaConfig:
    name: str
    edge_th: float
    margin_th: float
    notional: float
    cooldown: int
    exit_mode: str
    static_take_profit: float
    static_stop_loss: float
    static_max_hold: int
    tp_floor: float
    sl_floor: float
    tp_cap: float
    sl_cap: float


class ContextualScoutNet(nn.Module):
    def __init__(self, seq_dim: int, ctx_dim: int, hidden: int = 96, ctx_hidden: int = 64) -> None:
        super().__init__()
        self.hidden = hidden
        self.tcn = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, 3, padding=8, dilation=8),
            nn.GELU(),
            nn.Dropout(0.05),
        )
        self.q_proj = nn.Linear(hidden, hidden)
        self.k_proj = nn.Linear(hidden, hidden)
        self.v_proj = nn.Linear(hidden, hidden)
        self.ctx_mlp = nn.Sequential(
            nn.LayerNorm(ctx_dim),
            nn.Linear(ctx_dim, ctx_hidden),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(ctx_hidden, ctx_hidden),
            nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Linear(hidden * 2 + ctx_hidden, 192),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(192, 128),
            nn.GELU(),
        )
        self.edge_head = nn.Linear(128, 2)
        self.tp_head = nn.Linear(128, 2)
        self.sl_head = nn.Linear(128, 2)
        self.hold_head = nn.Linear(128, 2 * len(HOLD_BUCKETS))
        bias = torch.linspace(-0.75, 0.75, SEQ_LEN, dtype=torch.float32)
        self.register_buffer("recency_bias", bias, persistent=False)

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.tcn(seq.transpose(1, 2)).transpose(1, 2)
        last = h[:, -1, :]
        q = self.q_proj(last).unsqueeze(1)
        k = self.k_proj(h)
        v = self.v_proj(h)
        scores = (q * k).sum(dim=-1) / math.sqrt(self.hidden)
        scores = scores + self.recency_bias.unsqueeze(0)
        weights = torch.softmax(scores, dim=-1)
        attn = (weights.unsqueeze(-1) * v).sum(dim=1)
        ctx_z = self.ctx_mlp(ctx)
        z = self.fuse(torch.cat([last, attn, ctx_z], dim=-1))
        return {
            "edge": self.edge_head(z),
            "tp_mult": F.softplus(self.tp_head(z)) + 0.25,
            "sl_mult": F.softplus(self.sl_head(z)) + 0.20,
            "hold_logits": self.hold_head(z).view(-1, 2, len(HOLD_BUCKETS)),
        }


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _grid() -> list[DeepAlphaConfig]:
    return [
        DeepAlphaConfig("v30_static_ref", 0.010, 0.0040, 1.2, 12, "static", 0.045, 0.022, 48, 0.012, 0.008, 0.060, 0.032),
        DeepAlphaConfig("v30_static_tight", 0.012, 0.0040, 1.0, 12, "static", 0.040, 0.020, 48, 0.012, 0.008, 0.060, 0.032),
        DeepAlphaConfig("v30_dynamic_balanced", 0.010, 0.0040, 1.0, 12, "dynamic", 0.045, 0.022, 48, 0.012, 0.008, 0.060, 0.032),
        DeepAlphaConfig("v30_dynamic_full", 0.010, 0.0040, 1.2, 12, "dynamic", 0.045, 0.022, 48, 0.012, 0.008, 0.060, 0.032),
        DeepAlphaConfig("v30_dynamic_precision", 0.012, 0.0050, 1.0, 12, "dynamic", 0.045, 0.022, 48, 0.012, 0.008, 0.060, 0.030),
        DeepAlphaConfig("v30_dynamic_aggressive", 0.008, 0.0030, 1.2, 10, "dynamic", 0.045, 0.022, 48, 0.010, 0.007, 0.070, 0.035),
    ]


def _feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    cols = v23._select_seq_cols(df)
    extra = [
        c
        for c in df.columns
        if c.startswith("m7_") or c.startswith("teacher_") or c.startswith("clean_regime_2024_unsup_v4_") or c.startswith("ai_")
    ]
    merged: list[str] = []
    for c in cols + extra:
        lc = c.lower()
        if c in merged or c not in df.columns:
            continue
        if any(tok in lc for tok in v23.FORBIDDEN):
            continue
        if any(tok in lc for tok in ("target", "label", "future", "cash_after")):
            continue
        merged.append(c)

    ctx_cols: list[str] = []
    seq_cols: list[str] = []
    for c in merged:
        if c.startswith("clean_regime_2024_unsup_v4_") or c.startswith("teacher_") or c.startswith("m7_") or c.startswith("ai_"):
            ctx_cols.append(c)
        else:
            seq_cols.append(c)
    if not seq_cols or not ctx_cols:
        raise RuntimeError("failed to build seq/context feature groups")
    return seq_cols, ctx_cols


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str]) -> np.ndarray:
    start = max(0, idx - SEQ_LEN + 1)
    arr = (
        df.loc[start:idx, cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if len(arr) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(arr), len(cols)), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr[-SEQ_LEN:]


def _ctx_at(df: pd.DataFrame, idx: int, cols: list[str]) -> np.ndarray:
    return (
        df.loc[idx, cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_row_float(row, "bb_width", 0.0))
    gk = abs(_safe_row_float(row, "garman_klass_vol", 0.0))
    rs = abs(_safe_row_float(row, "rogers_satchell_vol", 0.0))
    pk = abs(_safe_row_float(row, "parkinson_vol", 0.0))
    volz = abs(_safe_row_float(row, "volatility_z", 0.0))
    rv = abs(_safe_row_float(row, "realized_vol_ratio", 1.0))
    base = max(0.0015, bbw * 0.15, gk * 2.5, rs * 2.5, pk * 2.5)
    scale = base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0))
    return _clip(scale, 0.0015, 0.030)


def _normalizer(arr: np.ndarray, axes: tuple[int, ...]) -> dict[str, np.ndarray]:
    return {
        "mean": np.nanmean(arr, axis=axes).astype(np.float32),
        "std": (np.nanstd(arr, axis=axes) + 1e-6).astype(np.float32),
    }


def _apply_norm(arr: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    if arr.ndim == 3:
        return ((arr - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)
    return ((arr - norm["mean"][None, :]) / norm["std"][None, :]).astype(np.float32)


def _raw_exit_return(df: pd.DataFrame, entry_px: float, exit_i: int, side: int, slip_eff: float) -> float:
    exit_px = _fill_price(df, exit_i, side, slip_eff, entry=False)
    if side > 0:
        return float((exit_px - entry_px) / max(entry_px, 1e-12))
    return float((entry_px - exit_px) / max(entry_px, 1e-12))


def _build_train_set(df: pd.DataFrame, seq_cols: list[str], ctx_cols: list[str], *, fee: float, slip: float, stride: int = 3) -> dict[str, np.ndarray]:
    seqs: list[np.ndarray] = []
    ctxs: list[np.ndarray] = []
    edge_targets: list[list[float]] = []
    aux_targets: list[list[float]] = []
    hold_targets: list[list[int]] = []

    for i in range(SEQ_LEN, len(df) - max(HOLD_BUCKETS) - 2, stride):
        entry_i = min(i + 1, len(df) - 1)
        row = df.iloc[i]
        vol_anchor = _vol_anchor(row)
        side_edges: list[float] = []
        side_tp_mult: list[float] = []
        side_sl_mult: list[float] = []
        side_hold_idx: list[int] = []
        for side in (1, -1):
            fee_eff = fee * 2.0
            slip_eff = slip * 2.0
            entry_px = _fill_price(df, entry_i, side, slip_eff, entry=True)
            rewards: list[float] = []
            paths: list[list[float]] = []
            for hold in HOLD_BUCKETS:
                exit_i = min(i + hold, len(df) - 1)
                raw_path = [
                    _raw_exit_return(df, entry_px, j, side, slip_eff)
                    for j in range(entry_i, exit_i + 1)
                ]
                reward = raw_path[-1] - fee_eff * 2.0
                rewards.append(float(reward))
                paths.append(raw_path)
            best_idx = int(np.argmax(rewards))
            best_path = paths[best_idx]
            mfe = max(best_path) if best_path else vol_anchor
            mae = abs(min(best_path)) if best_path else vol_anchor
            tp_mult = _clip(max(mfe * 0.85, vol_anchor * 0.75) / max(vol_anchor, 1e-8), 0.75, 8.0)
            sl_mult = _clip(max(mae * 0.90, vol_anchor * 0.60) / max(vol_anchor, 1e-8), 0.40, 6.0)
            side_edges.append(float(rewards[best_idx]))
            side_tp_mult.append(tp_mult)
            side_sl_mult.append(sl_mult)
            side_hold_idx.append(best_idx)
        seqs.append(_seq_at(df, i, seq_cols))
        ctxs.append(_ctx_at(df, i, ctx_cols))
        edge_targets.append(side_edges)
        aux_targets.append([side_tp_mult[0], side_tp_mult[1], side_sl_mult[0], side_sl_mult[1]])
        hold_targets.append(side_hold_idx)

    if not seqs:
        raise RuntimeError("no contextual deep scout train sequences")
    return {
        "seq": np.stack(seqs).astype(np.float32),
        "ctx": np.stack(ctxs).astype(np.float32),
        "edge": np.asarray(edge_targets, dtype=np.float32),
        "aux": np.asarray(aux_targets, dtype=np.float32),
        "hold": np.asarray(hold_targets, dtype=np.int64),
    }


def _train_model(ds: dict[str, np.ndarray], seq_norm: dict[str, np.ndarray], ctx_norm: dict[str, np.ndarray], *, epochs: int) -> ContextualScoutNet:
    x_seq = _apply_norm(ds["seq"], seq_norm)
    x_ctx = _apply_norm(ds["ctx"], ctx_norm)
    y_edge = ds["edge"].astype(np.float32)
    y_aux = ds["aux"].astype(np.float32)
    y_hold = ds["hold"].astype(np.int64)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ContextualScoutNet(x_seq.shape[-1], x_ctx.shape[-1]).to(device)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(x_seq),
            torch.from_numpy(x_ctx),
            torch.from_numpy(y_edge),
            torch.from_numpy(y_aux),
            torch.from_numpy(y_hold),
        ),
        batch_size=128,
        shuffle=True,
    )
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    reg_loss = nn.SmoothL1Loss()
    model.train()
    for _ in range(epochs):
        for xb_seq, xb_ctx, yb_edge, yb_aux, yb_hold in loader:
            xb_seq = xb_seq.to(device)
            xb_ctx = xb_ctx.to(device)
            yb_edge = yb_edge.to(device)
            yb_aux = yb_aux.to(device)
            yb_hold = yb_hold.to(device)
            pred = model(xb_seq, xb_ctx)
            hold_long = pred["hold_logits"][:, 0, :]
            hold_short = pred["hold_logits"][:, 1, :]
            loss = reg_loss(pred["edge"], yb_edge)
            loss = loss + 0.35 * reg_loss(torch.cat([pred["tp_mult"], pred["sl_mult"]], dim=1), yb_aux)
            loss = loss + 0.20 * F.cross_entropy(hold_long, yb_hold[:, 0])
            loss = loss + 0.20 * F.cross_entropy(hold_short, yb_hold[:, 1])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict_all(model: ContextualScoutNet, df: pd.DataFrame, seq_cols: list[str], ctx_cols: list[str], seq_norm: dict[str, np.ndarray], ctx_norm: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    seqs = np.stack([_seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    ctx = np.stack([_ctx_at(df, i, ctx_cols) for i in range(len(df))]).astype(np.float32)
    x_seq = _apply_norm(seqs, seq_norm)
    x_ctx = _apply_norm(ctx, ctx_norm)
    out = {"edge": [], "tp_mult": [], "sl_mult": [], "hold_logits": []}
    with torch.no_grad():
        for start in range(0, len(x_seq), 512):
            pred = model(
                torch.from_numpy(x_seq[start : start + 512]),
                torch.from_numpy(x_ctx[start : start + 512]),
            )
            for k in out:
                out[k].append(pred[k].numpy())
    return {k: np.concatenate(v, axis=0).astype(np.float32) for k, v in out.items()}


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_pred: dict[str, np.ndarray],
    cfg: DeepAlphaConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    owner = ""
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = f"{owner}_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {
                    "parent_notional": parent_notional,
                    "notional": notional,
                    "bars_since_entry": hold,
                    "unrealized": unreal,
                    "mfe": mfe,
                    "mae": mae,
                    "drawdown_abs": dd_abs,
                    "take_profit": take_profit,
                    "stop_loss": stop_loss,
                    "max_hold": max_hold,
                }
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update(
                        {
                            "exit_signal_timestamp": str(df["timestamp"].iloc[i]),
                            "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                            "exit_reason": reason,
                            "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0),
                            "final_notional_exposure": float(notional),
                            "mfe_pct": float(mfe * 100.0),
                            "mae_pct": float(mae * 100.0),
                            "fee_exit_pct": float(fee_eff * notional * 100.0),
                            "cash_after": float(cash),
                        }
                    )
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(cfg.cooldown))
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != ACTION_CASH and int(dec.side) != 0:
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {
                    "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                    "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                    "owner": owner,
                    "side": "LONG" if pos > 0 else "SHORT",
                    "entry_price": float(entry_price),
                    "notional_exposure": float(notional),
                    "leverage": float(dec.leverage),
                    "take_profit": float(take_profit),
                    "stop_loss": float(stop_loss),
                    "max_hold_bars": int(max_hold),
                    "fee_entry_pct": float(fee_eff * notional * 100.0),
                }
            continue
        if deep_cooldown <= 0 and i >= SEQ_LEN:
            ql, qs = float(deep_pred["edge"][i, 0]), float(deep_pred["edge"][i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= cfg.edge_th and margin >= cfg.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(cfg.notional)
                notional = float(cfg.notional)
                next_cooldown = int(cfg.cooldown)
                if cfg.exit_mode == "dynamic":
                    vol_anchor = _vol_anchor(df.iloc[i])
                    side_idx = 0 if side > 0 else 1
                    tp_mult = float(deep_pred["tp_mult"][i, side_idx])
                    sl_mult = float(deep_pred["sl_mult"][i, side_idx])
                    hold_idx = int(np.argmax(deep_pred["hold_logits"][i, side_idx]))
                    tp_raw = _clip(tp_mult * vol_anchor, cfg.tp_floor, cfg.tp_cap)
                    sl_raw = _clip(sl_mult * vol_anchor, cfg.sl_floor, cfg.sl_cap)
                    take_profit = float(tp_raw * notional)
                    stop_loss = float(sl_raw * notional)
                    max_hold = int(HOLD_BUCKETS[hold_idx])
                else:
                    take_profit = float(cfg.static_take_profit)
                    stop_loss = float(cfg.static_stop_loss)
                    max_hold = int(cfg.static_max_hold)
                    tp_mult = float("nan")
                    sl_mult = float("nan")
                    hold_idx = -1
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if record:
                    open_record = {
                        "entry_signal_timestamp": str(df["timestamp"].iloc[i]),
                        "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]),
                        "owner": owner,
                        "side": "LONG" if pos > 0 else "SHORT",
                        "entry_price": float(entry_price),
                        "notional_exposure": float(notional),
                        "deep_q_long": ql,
                        "deep_q_short": qs,
                        "deep_tp_mult": tp_mult,
                        "deep_sl_mult": sl_mult,
                        "deep_hold_bucket_idx": hold_idx,
                        "take_profit": float(take_profit),
                        "stop_loss": float(stop_loss),
                        "max_hold_bars": int(max_hold),
                        "fee_entry_pct": float(fee_eff * notional * 100.0),
                    }
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / _days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": actions,
    }
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.40 * c2["pnl"] + 0.25 * c3["pnl"] - 0.38 * abs(c1["mdd"]) + 0.20 * min(c1.get("deep_entries", 0), 90))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V30 contextual deep scout with recent-step readout and dynamic exits.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=110)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    train = train_all[train_all["timestamp"] < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    seq_cols, ctx_cols = _feature_groups(train_all)
    forbidden_cols = [c for c in seq_cols + ctx_cols if any(tok in c.lower() for tok in v23.FORBIDDEN)]
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    train_ds = _build_train_set(train, seq_cols, ctx_cols, fee=float(base["fee"]), slip=float(base["slip"]), stride=3)
    seq_norm = _normalizer(train_ds["seq"], axes=(0, 1))
    ctx_norm = _normalizer(train_ds["ctx"], axes=(0,))
    model = _train_model(train_ds, seq_norm, ctx_norm, epochs=args.epochs)
    val_pred = _predict_all(model, val, seq_cols, ctx_cols, seq_norm, ctx_norm)
    eval_pred = _predict_all(model, eval_df, seq_cols, ctx_cols, seq_norm, ctx_norm)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, add_cfg, val_pred, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = backtest(val, bundle, jackpot_model, add_cfg, val_pred, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = backtest(val, bundle, jackpot_model, add_cfg, val_pred, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        row = {
            "config": asdict(cfg),
            "validation_cost1": v1,
            "validation_cost2": v2,
            "validation_cost3": v3,
            "selection_score": _score(v1, v2, v3),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = DeepAlphaConfig(**best["config"])

    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(
            eval_df,
            bundle,
            jackpot_model,
            add_cfg,
            eval_pred,
            selected,
            fee=float(base["fee"]),
            slip=float(base["slip"]),
            cost_mult=float(mult),
            decisions=eval_dec,
            record=(mult == 1),
        )
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r

    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v30_contextual_dynamic_exit_scout.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "seq_cols": seq_cols,
            "ctx_cols": ctx_cols,
            "seq_norm": seq_norm,
            "ctx_norm": ctx_norm,
            "selected_config": asdict(selected),
            "parent_model": str(args.parent_model),
            "jackpot_model": str(args.jackpot_model),
        },
        model_path,
    )
    manifest_path = args.out_dir / "feature_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "seq_cols": seq_cols,
                "ctx_cols": ctx_cols,
                "seq_len": SEQ_LEN,
                "hold_buckets": HOLD_BUCKETS,
                "forbidden_cols": forbidden_cols,
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )

    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "score": r["selection_score"],
                "val_pnl": r["validation_cost1"]["pnl"],
                "val_mdd": r["validation_cost1"]["mdd"],
                "val_trades": r["validation_cost1"]["trades"],
                "val_deep_entries": r["validation_cost1"].get("deep_entries", 0),
                "val_c2_pnl": r["validation_cost2"]["pnl"],
                "val_c3_pnl": r["validation_cost3"]["pnl"],
            }
            for r in rows
        ]
    ).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={sorted(set(forbidden_cols))}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V27_COST1:
        warnings.append("oos_cost1_did_not_beat_v27")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")

    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V27_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": "contextual_dynamic_exit_scout_v30",
        "v21_2_preserved": True,
        "deep_sleeve_only_when_parent_cash": True,
        "feature_audit": feature_audit,
        "selected_config": asdict(selected),
        "metrics": metrics,
        "baselines": {"v27_cost1": V27_COST1, "v27_cost2": V27_COST2, "v27_cost3": V27_COST3},
    }
    report = {
        "model_id": MODEL_ID,
        "design": "V30 replaces global average pooling with recent-step readout plus recency-weighted attention, separates clean regime / teacher / m7 context into an explicit fusion path, and predicts dynamic deep-scout TP/SL/hold heads. V21.2 parent entries and jackpot add-ons are preserved; the deep sleeve may open only when the parent is CASH.",
        "parent_model": str(args.parent_model),
        "jackpot_model": str(args.jackpot_model),
        "model": str(model_path),
        "feature_manifest": str(manifest_path),
        "split_policy": "Train 2025 Jan-Sep; select thresholds/notional/exit_mode on 2025 Oct-Dec; evaluate fixed 2026 OOS only after selection.",
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {
            "model": str(model_path),
            "manifest": str(manifest_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(args.report_out),
                "audit": str(args.audit_out),
                "model": str(model_path),
                "selected": asdict(selected),
                "metrics": metrics,
                "verdict": verdict,
            },
            ensure_ascii=False,
            default=_json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
