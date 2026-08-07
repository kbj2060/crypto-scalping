#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import random
import sys
import argparse
from dataclasses import dataclass
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

from scripts.alpha7_experiment_config import get_live_baseline  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combo_metrics,
    _combine_primary_fallback,
    _load_best_scale_runtime,
    _predict_scaled,
    _read,
)
from scripts.rebuild_alpha7_v2_only_high_turnover_20260526 import _rename_clean4_v2  # noqa: E402


MODEL_ID = "alpha7_directional_dsac_router_20260529"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TRAIN_CSV = (
    Path(os.environ.get(
        "ALPHA7_ROUTER_TRAIN_CSV",
        str(ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv"),
    ))
)
EVAL_CSV = (
    Path(os.environ.get(
        "ALPHA7_ROUTER_EVAL_CSV",
        str(ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv"),
    ))
)
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")

ACTION_SKIP = 0
ACTION_PRIMARY = 1
ACTION_FALLBACK = 2
ACTION_DIM = 3

SOURCE_COLS = [
    "net_taker_ratio",
    "taker_acceleration",
    "ofi_acceleration",
    "smart_money_flow",
    "funding_price_divergence",
    "hurst_48",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "breakout_strength",
    "rsi",
    "tp_sl_action_score",
    "ai_dir_edge",
    "ai_flow_pressure",
    "m7_expected_ret",
    "m7_q50",
    "m7_quality_pred",
    "clean_regime4_state24_sticky090_v2_bull_prob",
    "clean_regime4_state24_sticky090_v2_bear_prob",
    "clean_regime4_state24_sticky090_v2_chop_prob",
    "clean_regime4_state24_sticky090_v2_whipsaw_prob",
    "clean_regime4_state24_sticky090_v2_confidence",
    "clean_regime4_state24_sticky090_v2_entropy",
    "regime4_pred_bull_prob",
    "regime4_pred_bear_prob",
    "regime4_pred_chop_prob",
    "regime4_pred_whipsaw_prob",
    "regime4_pred_confidence",
]

DECISION_COLS = [
    "action",
    "side",
    "quality_score",
    "confidence",
    "notional_exposure",
    "leverage",
    "take_profit",
    "stop_loss",
    "max_hold_bars",
    "cooldown_bars",
]


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _audit_frame_contract(df: pd.DataFrame, *, name: str) -> None:
    required = ["timestamp", "open", "high", "low", "close", "volume", *SOURCE_COLS]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name} missing required directional DSAC columns: {missing}")
    bad = [c for c in df.columns if c.startswith(FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")


def _safe_num(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise RuntimeError(f"feature contract violation: missing column {col}")
    return pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _decision_num(dec: pd.DataFrame, col: str) -> pd.Series:
    if col not in dec.columns:
        raise RuntimeError(f"decision contract violation: missing column {col}")
    return pd.to_numeric(dec[col], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _linear_slope(close: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=np.float64)
    x = x - x.mean()
    denom = float(np.sum(x * x))

    def slope(values: np.ndarray) -> float:
        y = np.asarray(values, dtype=np.float64)
        if y.size != window:
            return 0.0
        y = y - y.mean()
        return float(np.sum(x * y) / max(denom, 1e-12))

    return close.rolling(window, min_periods=window).apply(slope, raw=True).fillna(0.0)


def _directional_features(df: pd.DataFrame) -> pd.DataFrame:
    close = _safe_num(df, "close").replace(0.0, np.nan).ffill().bfill().fillna(1.0)
    high = _safe_num(df, "high").fillna(close)
    low = _safe_num(df, "low").fillna(close)
    volume = _safe_num(df, "volume")

    ret = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    atr_proxy = ((high - low).abs() / close.abs().clip(lower=1e-12)).rolling(24, min_periods=4).mean().fillna(0.0)
    vol = ret.rolling(24, min_periods=4).std(ddof=0).fillna(ret.std(ddof=0) or 1e-6).abs().clip(lower=1e-6)

    ema9 = close.ewm(span=9, adjust=False, min_periods=1).mean()
    ema21 = close.ewm(span=21, adjust=False, min_periods=1).mean()
    rolling_high_prev = high.rolling(12, min_periods=2).max().shift(1)
    rolling_low_prev = low.rolling(12, min_periods=2).min().shift(1)

    out = pd.DataFrame(index=df.index)
    out["logret_1"] = np.log(close / close.shift(1)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for n in (3, 6, 12, 24):
        out[f"price_momentum_{n}b"] = (close / close.shift(n) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["ema_cross_signal"] = ((ema9 - ema21) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["linear_slope_12b"] = (_linear_slope(close, 12) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["linear_slope_24b"] = (_linear_slope(close, 24) / close.abs().clip(lower=1e-12) / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out["higher_high_12b"] = (high > rolling_high_prev).fillna(False).astype(float)
    out["lower_low_12b"] = (low < rolling_low_prev).fillna(False).astype(float)
    out["range_atr_proxy"] = atr_proxy
    out["volume_momentum_12b"] = (volume / volume.rolling(12, min_periods=2).mean().replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def _state_frame(frame: pd.DataFrame, primary: pd.DataFrame, fallback: pd.DataFrame) -> pd.DataFrame:
    _audit_frame_contract(frame, name="state_frame")
    parts: list[pd.DataFrame] = [_directional_features(frame)]
    parts.append(pd.DataFrame({c: _safe_num(frame, c) for c in SOURCE_COLS}, index=frame.index))

    for prefix, dec in (("primary", primary), ("fallback", fallback)):
        d = pd.DataFrame(index=frame.index)
        for col in DECISION_COLS:
            d[f"{prefix}_{col}"] = _decision_num(dec, col).to_numpy(dtype=np.float64)
        parts.append(d)

    pa = _decision_num(primary, "action").astype(int)
    ps = _decision_num(primary, "side").astype(int)
    fa = _decision_num(fallback, "action").astype(int)
    fs = _decision_num(fallback, "side").astype(int)
    pq = _decision_num(primary, "quality_score")
    fq = _decision_num(fallback, "quality_score")
    pc = _decision_num(primary, "confidence")
    fc = _decision_num(fallback, "confidence")
    meta = pd.DataFrame(index=frame.index)
    meta["primary_active"] = ((pa != 0) & (ps != 0)).astype(float)
    meta["fallback_active"] = ((fa != 0) & (fs != 0)).astype(float)
    meta["side_agree"] = ((ps == fs) & (ps != 0)).astype(float)
    meta["side_disagree"] = ((ps != fs) & (ps != 0) & (fs != 0)).astype(float)
    meta["quality_diff_primary_fallback"] = pq - fq
    meta["confidence_diff_primary_fallback"] = pc - fc
    meta["quality_max"] = np.maximum(pq, fq)
    meta["confidence_max"] = np.maximum(pc, fc)
    parts.append(meta)

    out = pd.concat(parts, axis=1)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if out.columns.duplicated().any():
        dup = out.columns[out.columns.duplicated()].tolist()
        raise RuntimeError(f"duplicate DSAC state columns: {dup[:20]}")
    return out


def _fit_norm(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": list(x.columns), "median": med.tolist(), "scale": scale.tolist()}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"normalizer/state column mismatch: missing={missing[:20]}")
    extra = [c for c in x.columns if c not in cols]
    if extra:
        raise RuntimeError(f"normalizer/state column mismatch: extra={extra[:20]}")
    arr = x[cols].to_numpy(dtype=np.float64)
    med = np.asarray(norm["median"], dtype=np.float64)
    scale = np.asarray(norm["scale"], dtype=np.float64)
    z = (arr - med) / scale
    return np.tanh(np.nan_to_num(z, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _first_hit(path: np.ndarray, tp: float, sl: float, hold: int) -> int:
    m = min(int(max(1, hold)), len(path))
    if m <= 1:
        return 0
    p = path[:m]
    hit = np.flatnonzero((p >= float(tp)) | (p <= -abs(float(sl))))
    return int(hit[0]) if hit.size else int(m - 1)


def _candidate_reward(close: np.ndarray, i: int, dec_row: pd.Series, *, fee: float, slip: float) -> tuple[float, dict[str, Any]]:
    action = int(dec_row.get("action", 0) or 0)
    side = int(dec_row.get("side", 0) or 0)
    if action == 0 or side == 0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    notional = float(dec_row.get("notional_exposure", dec_row.get("notional", 0.0)) or 0.0)
    tp = float(dec_row.get("take_profit", 0.0) or 0.0)
    sl = float(dec_row.get("stop_loss", 0.0) or 0.0)
    hold = int(dec_row.get("max_hold_bars", 0) or 0)
    if notional <= 0.0 or hold <= 0:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    entry_i = min(int(i) + 1, len(close) - 1)
    end = min(len(close), entry_i + hold + 1)
    if end <= entry_i + 1:
        return 0.0, {"active": 0, "net": 0.0, "win": 0}
    entry = max(float(close[entry_i]), 1e-12)
    fut = close[entry_i + 1 : end]
    side_ret = ((fut / entry) - 1.0) * float(side)
    path = side_ret * notional
    exit_i = _first_hit(path, tp, sl, hold)
    gross = float(path[exit_i])
    net = gross - 2.0 * (fee + slip) * notional
    win = int(net > 0.0)
    # No trade-count penalty: reward is strictly outcome-driven.
    reward = 140.0 * net + (0.35 if win else -0.18)
    if net > 0.0:
        reward += 45.0 * net
    else:
        reward += 25.0 * net
    return float(reward), {"active": 1, "net": float(net), "win": win}


@dataclass
class DatasetBundle:
    states: np.ndarray
    next_states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray


def _build_counterfactual_dataset(
    frame: pd.DataFrame,
    states: np.ndarray,
    primary: pd.DataFrame,
    fallback: pd.DataFrame,
    *,
    fee: float,
    slip: float,
) -> tuple[DatasetBundle, dict[str, Any]]:
    close = _safe_num(frame, "close").to_numpy(dtype=np.float64)
    s_list: list[np.ndarray] = []
    sp_list: list[np.ndarray] = []
    a_list: list[int] = []
    r_list: list[float] = []
    d_list: list[float] = []
    reward_stats: dict[int, list[float]] = {ACTION_SKIP: [], ACTION_PRIMARY: [], ACTION_FALLBACK: []}
    win_stats: dict[int, list[int]] = {ACTION_SKIP: [], ACTION_PRIMARY: [], ACTION_FALLBACK: []}

    for i in range(len(frame) - 2):
        rewards = {
            ACTION_SKIP: (0.0, {"active": 0, "net": 0.0, "win": 0}),
            ACTION_PRIMARY: _candidate_reward(close, i, primary.iloc[i], fee=fee, slip=slip),
            ACTION_FALLBACK: _candidate_reward(close, i, fallback.iloc[i], fee=fee, slip=slip),
        }
        for action, (reward, meta) in rewards.items():
            s_list.append(states[i])
            sp_list.append(states[i + 1])
            a_list.append(int(action))
            r_list.append(float(reward))
            d_list.append(1.0 if i == len(frame) - 3 else 0.0)
            if int(meta["active"]) == 1:
                reward_stats[int(action)].append(float(meta["net"]))
                win_stats[int(action)].append(int(meta["win"]))

    rewards_np = np.asarray(r_list, dtype=np.float32)
    scale = float(np.nanstd(rewards_np))
    if not np.isfinite(scale) or scale < 1e-6:
        scale = 1.0
    rewards_np = np.clip(rewards_np / scale, -8.0, 8.0).astype(np.float32)
    diagnostics = {
        "reward_scale": scale,
        "candidate_net_mean": {str(k): float(np.mean(v)) if v else 0.0 for k, v in reward_stats.items()},
        "candidate_win_rate": {str(k): float(np.mean(v)) if v else 0.0 for k, v in win_stats.items()},
        "candidate_active_count": {str(k): int(len(v)) for k, v in reward_stats.items()},
    }
    return (
        DatasetBundle(
            states=np.asarray(s_list, dtype=np.float32),
            next_states=np.asarray(sp_list, dtype=np.float32),
            actions=np.asarray(a_list, dtype=np.int64),
            rewards=rewards_np,
            dones=np.asarray(d_list, dtype=np.float32),
        ),
        diagnostics,
    )


class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 192),
            nn.SiLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(256, 192),
            nn.SiLU(),
            nn.Linear(192, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _train_dsac_offline(
    data: DatasetBundle,
    *,
    state_dim: int,
    action_dim: int,
    device: torch.device,
    steps: int,
    batch_size: int,
    gamma: float = 0.995,
    tau: float = 0.01,
    lr: float = 2.5e-4,
) -> dict[str, Any]:
    actor = Actor(state_dim, action_dim).to(device)
    q1 = Critic(state_dim, action_dim).to(device)
    q2 = Critic(state_dim, action_dim).to(device)
    tq1 = Critic(state_dim, action_dim).to(device)
    tq2 = Critic(state_dim, action_dim).to(device)
    tq1.load_state_dict(q1.state_dict())
    tq2.load_state_dict(q2.state_dict())
    log_alpha = torch.tensor(math.log(0.15), device=device, requires_grad=True)
    target_entropy = 0.75 * math.log(float(action_dim))

    opt_actor = torch.optim.AdamW(actor.parameters(), lr=lr, weight_decay=1e-5)
    opt_q1 = torch.optim.AdamW(q1.parameters(), lr=lr, weight_decay=1e-5)
    opt_q2 = torch.optim.AdamW(q2.parameters(), lr=lr, weight_decay=1e-5)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)

    ds = TensorDataset(
        torch.from_numpy(data.states),
        torch.from_numpy(data.next_states),
        torch.from_numpy(data.actions),
        torch.from_numpy(data.rewards),
        torch.from_numpy(data.dones),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    it = iter(dl)
    last = {"q_loss": 0.0, "actor_loss": 0.0, "alpha": 0.0, "entropy": 0.0}

    for step in range(1, int(steps) + 1):
        try:
            s, sp, a, r, d = next(it)
        except StopIteration:
            it = iter(dl)
            s, sp, a, r, d = next(it)
        s = s.to(device)
        sp = sp.to(device)
        a = a.to(device)
        r = r.to(device)
        d = d.to(device)

        with torch.no_grad():
            next_logits = actor(sp)
            next_logp = F.log_softmax(next_logits, dim=-1)
            next_pi = next_logp.exp()
            alpha = log_alpha.exp()
            next_q = torch.min(tq1(sp), tq2(sp))
            v_next = (next_pi * (next_q - alpha * next_logp)).sum(dim=-1)
            y = r + (1.0 - d) * gamma * v_next

        qa1 = q1(s).gather(1, a.view(-1, 1)).squeeze(1)
        qa2 = q2(s).gather(1, a.view(-1, 1)).squeeze(1)
        q_loss = F.smooth_l1_loss(qa1, y) + F.smooth_l1_loss(qa2, y)
        opt_q1.zero_grad(set_to_none=True)
        opt_q2.zero_grad(set_to_none=True)
        q_loss.backward()
        nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), 5.0)
        opt_q1.step()
        opt_q2.step()

        logits = actor(s)
        logp = F.log_softmax(logits, dim=-1)
        pi = logp.exp()
        alpha = log_alpha.exp()
        q_min = torch.min(q1(s), q2(s))
        actor_loss = (pi * (alpha * logp - q_min)).sum(dim=-1).mean()
        opt_actor.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
        opt_actor.step()

        entropy = -(pi * logp).sum(dim=-1).mean().detach()
        alpha_loss = -(log_alpha * (entropy - target_entropy)).mean()
        opt_alpha.zero_grad(set_to_none=True)
        alpha_loss.backward()
        opt_alpha.step()
        log_alpha.data.clamp_(math.log(1e-4), math.log(3.0))

        with torch.no_grad():
            for p, tp in zip(q1.parameters(), tq1.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)
            for p, tp in zip(q2.parameters(), tq2.parameters()):
                tp.data.mul_(1.0 - tau).add_(tau * p.data)

        if step % 250 == 0:
            last = {
                "q_loss": float(q_loss.item()),
                "actor_loss": float(actor_loss.item()),
                "alpha": float(log_alpha.exp().item()),
                "entropy": float(entropy.item()),
                "step": int(step),
            }
        if step % 1000 == 0:
            print(json.dumps({"stage": "train_progress", **last}, ensure_ascii=False), flush=True)
    return {"actor": actor.cpu(), "q1": q1.cpu(), "q2": q2.cpu(), "train_diag": last}


def _policy_action(actor: nn.Module, states: np.ndarray, *, device: torch.device) -> np.ndarray:
    actor = actor.to(device)
    actor.eval()
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(states), 8192):
            x = torch.from_numpy(states[start : start + 8192]).to(device)
            logits = actor(x)
            out.append(torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int64))
    return np.concatenate(out) if out else np.zeros(0, dtype=np.int64)


def _compose_decisions(primary: pd.DataFrame, fallback: pd.DataFrame, actions: np.ndarray) -> pd.DataFrame:
    out = primary.copy().reset_index(drop=True)
    fallback = fallback.reset_index(drop=True)
    for i in range(len(out)):
        a = int(actions[i])
        if a == ACTION_SKIP:
            out.loc[i, ["action", "side", "notional_exposure"]] = [0, 0, 0.0]
        elif a == ACTION_FALLBACK:
            out.iloc[i] = fallback.iloc[i]
        elif a == ACTION_PRIMARY:
            continue
        else:
            raise RuntimeError(f"invalid DSAC action: {a}")
    return out


def _usage(actions: np.ndarray) -> dict[str, int]:
    return {
        "skip": int(np.sum(actions == ACTION_SKIP)),
        "primary": int(np.sum(actions == ACTION_PRIMARY)),
        "fallback": int(np.sum(actions == ACTION_FALLBACK)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=768)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()

    _seed_everything(290529)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")

    baseline = get_live_baseline()
    train_all = _rename_clean4_v2(_read(TRAIN_CSV))
    eval_df = _rename_clean4_v2(_read(EVAL_CSV))
    _audit_frame_contract(train_all, name="train_all")
    _audit_frame_contract(eval_df, name="eval")
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    primary_parent = joblib.load(baseline.primary_parent)
    fallback_parent = joblib.load(baseline.fallback_parent)
    primary_rt = _load_best_scale_runtime(baseline.primary_summary)
    fallback_rt = _load_best_scale_runtime(baseline.fallback_summary)

    p_train = _predict_scaled(primary_parent, train_df, primary_rt)
    p_val = _predict_scaled(primary_parent, val_df, primary_rt)
    p_eval = _predict_scaled(primary_parent, eval_df, primary_rt)
    f_train = _predict_scaled(fallback_parent, train_df, fallback_rt)
    f_val = _predict_scaled(fallback_parent, val_df, fallback_rt)
    f_eval = _predict_scaled(fallback_parent, eval_df, fallback_rt)

    state_train_df = _state_frame(train_df, p_train, f_train)
    state_val_df = _state_frame(val_df, p_val, f_val)
    state_eval_df = _state_frame(eval_df, p_eval, f_eval)
    norm = _fit_norm(state_train_df)
    x_train = _apply_norm(state_train_df, norm)
    x_val = _apply_norm(state_val_df, norm)
    x_eval = _apply_norm(state_eval_df, norm)

    fee = 0.0005
    slip = 0.0002
    data, data_diag = _build_counterfactual_dataset(train_df, x_train, p_train, f_train, fee=fee, slip=slip)
    print(
        json.dumps(
            {
                "stage": "train_start",
                "device": str(device),
                "state_dim": int(x_train.shape[1]),
                "samples": int(len(data.states)),
                "steps": int(args.steps),
                "batch_size": int(args.batch_size),
                "dataset_diagnostics": data_diag,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    trained = _train_dsac_offline(
        data,
        state_dim=int(x_train.shape[1]),
        action_dim=ACTION_DIM,
        device=device,
        steps=int(args.steps),
        batch_size=int(args.batch_size),
    )
    actor: nn.Module = trained["actor"]

    a_train = _policy_action(actor, x_train, device=device)
    a_val = _policy_action(actor, x_val, device=device)
    a_eval = _policy_action(actor, x_eval, device=device)

    dsac_train = _compose_decisions(p_train, f_train, a_train)
    dsac_val = _compose_decisions(p_val, f_val, a_val)
    dsac_eval = _compose_decisions(p_eval, f_eval, a_eval)
    base_train = _combine_primary_fallback(p_train, f_train)
    base_val = _combine_primary_fallback(p_val, f_val)
    base_eval = _combine_primary_fallback(p_eval, f_eval)

    rows: list[dict[str, Any]] = []
    for split, df, base_dec, dsac_dec in [
        ("train", train_df, base_train, dsac_train),
        ("val", val_df, base_val, dsac_val),
        ("oos", eval_df, base_eval, dsac_eval),
    ]:
        for name, dec in [("baseline_combo", base_dec), ("directional_dsac", dsac_dec)]:
            metrics = _combo_metrics(df, dec)
            for cost, vals in metrics.items():
                rows.append({"split": split, "variant": name, "cost": cost, **vals})
    grid = pd.DataFrame(rows)
    grid_path = OUT_DIR / "grid.csv"
    grid.to_csv(grid_path, index=False)

    ckpt_path = OUT_DIR / "directional_dsac_router.pt"
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dim": int(x_train.shape[1]),
            "action_dim": ACTION_DIM,
            "state_columns": list(norm["columns"]),
            "normalizer": norm,
            "actor_state_dict": actor.state_dict(),
            "train_diag": trained["train_diag"],
        },
        ckpt_path,
    )
    (OUT_DIR / "state_columns.json").write_text(json.dumps(list(norm["columns"]), indent=2) + "\n", encoding="utf-8")

    def pick(split: str, variant: str, cost: str) -> dict[str, Any]:
        row = grid[(grid["split"].eq(split)) & (grid["variant"].eq(variant)) & (grid["cost"].eq(cost))]
        if row.empty:
            return {}
        return row.iloc[0].to_dict()

    summary = {
        "model_id": MODEL_ID,
        "design": "Directional-feature discrete SAC router over Alpha7 primary/fallback/skip. Reward removes trade-count penalty and directly rewards net PnL plus winning outcomes.",
        "baseline_model_id": baseline.model_id,
        "train_csv": str(TRAIN_CSV),
        "eval_csv": str(EVAL_CSV),
        "device": str(device),
        "state_dim": int(x_train.shape[1]),
        "action_dim": ACTION_DIM,
        "reward": "reward = net_pnl_scaled + win_bonus - loss_penalty; no trade-count penalty",
        "dataset_diagnostics": data_diag,
        "train_diag": trained["train_diag"],
        "action_usage": {
            "train": _usage(a_train),
            "val": _usage(a_val),
            "oos": _usage(a_eval),
        },
        "cost3": {
            "train_baseline": pick("train", "baseline_combo", "cost3"),
            "train_dsac": pick("train", "directional_dsac", "cost3"),
            "val_baseline": pick("val", "baseline_combo", "cost3"),
            "val_dsac": pick("val", "directional_dsac", "cost3"),
            "oos_baseline": pick("oos", "baseline_combo", "cost3"),
            "oos_dsac": pick("oos", "directional_dsac", "cost3"),
        },
        "artifacts": {
            "grid": str(grid_path),
            "ckpt": str(ckpt_path),
            "state_columns": str(OUT_DIR / "state_columns.json"),
        },
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "grid": str(grid_path), "cost3": summary["cost3"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
