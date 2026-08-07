#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import eval_alpha1_l2_execution_replay_20260513 as l2  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_deep_exit_oracle_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_deep_exit_oracle_20260514"
MODEL_OUT = OUT_DIR / "deep_exit_oracle.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_deep_exit_oracle_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_deep_exit_oracle_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_deep_exit_oracle_20260514_grid.csv"
EVENTS_OUT = ROOT / "data/ensemble/reports/alpha3_deep_exit_oracle_20260514_events.json"
TRAIN_START = pd.Timestamp("2025-07-01")


@dataclass(frozen=True)
class DeepExitPolicy:
    name: str
    mode: str
    min_confidence: float
    fallback_arm: str
    fixed_arm: str = ""


class DeepExitOracleNet(nn.Module):
    def __init__(self, input_dim: int, n_arms: int, hidden: int = 128, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_arms),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _arm_configs() -> list[alpha3.ImmediateLimitConfig]:
    arms = [
        alpha3.ImmediateLimitConfig(
            "baseline_exit2_pen05",
            "next_open",
            2.0,
            2.0,
            0.5,
            0.20,
            entry_miss="market_fallback",
            exit_miss="market_fallback",
        )
    ]
    for exit_offset in (0.0, 1.0, 2.0, 3.0, 4.0):
        arms.append(
            alpha3.ImmediateLimitConfig(
                f"exit{exit_offset:g}_pen0",
                "next_open",
                2.0,
                float(exit_offset),
                0.0,
                0.20,
                entry_miss="market_fallback",
                exit_miss="market_fallback",
            )
        )
    return arms


def _policies() -> list[DeepExitPolicy]:
    rows = [
        DeepExitPolicy("fixed_baseline_exit2_pen05", "fixed", 1.0, "baseline_exit2_pen05", fixed_arm="baseline_exit2_pen05"),
        DeepExitPolicy("fixed_front_run_exit4_pen0", "fixed", 1.0, "exit4_pen0", fixed_arm="exit4_pen0"),
        DeepExitPolicy("deep_argmax", "deep", 0.0, "exit4_pen0"),
    ]
    for fallback in ("exit4_pen0", "baseline_exit2_pen05"):
        for conf in (0.35, 0.45, 0.55, 0.65, 0.75):
            rows.append(DeepExitPolicy(f"deep_conf{conf:.2f}_fallback_{fallback}", "deep", conf, fallback))
    return rows


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _feature_names(base_cols: list[str]) -> list[str]:
    state_cols = [
        "pos",
        "owner_deep",
        "owner_parent",
        "hold_norm",
        "unreal",
        "mfe",
        "mae",
        "giveback",
        "notional",
        "parent_notional",
        "take_profit",
        "stop_loss",
        "max_hold_norm",
        "entry_edge",
        "entry_vol_anchor",
        "effective_tp",
        "effective_sl",
        "dec_action",
        "dec_side",
        "dec_confidence",
        "dec_quality",
        "dec_notional",
        "dec_tp",
        "dec_sl",
        "q_long",
        "q_short",
        "q_same",
        "q_opp",
        "q_margin",
        "row_vol_anchor",
    ]
    return list(base_cols) + state_cols


def _feature_vector(
    df: pd.DataFrame,
    decisions: pd.DataFrame,
    deep_q: np.ndarray,
    base_cols: list[str],
    idx: int,
    *,
    pos: int,
    owner: str,
    hold: int,
    unreal: float,
    mfe: float,
    mae: float,
    notional: float,
    parent_notional: float,
    take_profit: float,
    stop_loss: float,
    max_hold: int,
    entry_edge: float,
    entry_vol_anchor: float,
    effective_tp: float,
    effective_sl: float,
) -> np.ndarray:
    row = df.iloc[int(idx)]
    dec = decisions.iloc[int(idx)]
    q_long = float(deep_q[int(idx), 0])
    q_short = float(deep_q[int(idx), 1])
    q_same = q_long if pos > 0 else q_short
    q_opp = q_short if pos > 0 else q_long
    vals = [_safe(row, col, 0.0) for col in base_cols]
    vals.extend(
        [
            float(pos),
            float(owner == "deep_alpha"),
            float(owner == "v21_2"),
            float(hold) / 64.0,
            float(unreal),
            float(mfe),
            float(mae),
            float(mfe - unreal),
            float(notional),
            float(parent_notional),
            float(take_profit),
            float(stop_loss),
            float(max_hold) / 64.0,
            float(entry_edge),
            float(entry_vol_anchor),
            float(effective_tp),
            float(effective_sl),
            float(dec.get("action", 0)),
            float(dec.get("side", 0)),
            float(dec.get("confidence", 0.0) or 0.0),
            float(dec.get("quality_score", 0.0) or 0.0),
            float(dec.get("notional_exposure", 0.0) or 0.0),
            float(dec.get("take_profit", 0.0) or 0.0),
            float(dec.get("stop_loss", 0.0) or 0.0),
            q_long,
            q_short,
            q_same,
            q_opp,
            float(q_opp - q_same),
            float(v31._vol_anchor(row)),
        ]
    )
    arr = np.asarray(vals, dtype=np.float32)
    arr[~np.isfinite(arr)] = 0.0
    return arr


def _exit_reward(
    df: pd.DataFrame,
    idx: int,
    pos: int,
    entry_price: float,
    notional: float,
    cfg: alpha3.ImmediateLimitConfig,
    *,
    fee_base: float,
    slip_base: float,
) -> tuple[float, str]:
    filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(
        df,
        idx,
        pos,
        cfg,
        entry=False,
        fee=fee_base,
        slip=slip_base,
    )
    if not filled:
        return -1e9, route
    raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    return float(raw * notional - exit_fee * notional), route


def _effective_deep_exits(
    owner: str,
    overlay: v31.OverlayConfig,
    take_profit: float,
    stop_loss: float,
    entry_edge: float,
    entry_vol_anchor: float,
    hold: int,
    mfe: float,
) -> tuple[float, float]:
    effective_tp = float(take_profit)
    effective_sl = float(stop_loss)
    if owner == "deep_alpha":
        if overlay.tp_util_mult > 0.0:
            util_gain = 1.0 + overlay.tp_util_mult * max(entry_edge - overlay.edge_th, 0.0) / max(0.02, overlay.edge_th)
            effective_tp = v31._clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap)
        if overlay.sl_vol_mult > 0.0:
            effective_sl = v31._clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap)
        if mfe > 0.0 and mfe >= float(getattr(overlay, "trail_activation", 0.009)) and overlay.trail_gap_mult > 0.0:
            trail_gap = entry_vol_anchor * overlay.trail_gap_mult
            if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
                trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * (hold - overlay.hold_decay_start) * entry_vol_anchor)
            trail_stop = max(-effective_sl, mfe - trail_gap)
            effective_sl = min(effective_sl, max(0.001, trail_stop))
    return float(effective_tp), float(effective_sl)


def collect_exit_events(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    path_cfg: alpha3.ImmediateLimitConfig,
    arms: list[alpha3.ImmediateLimitConfig],
    base_cols: list[str],
    *,
    fee: float,
    slip: float,
) -> list[dict[str, Any]]:
    close = _close(df)
    fee_base = float(fee)
    slip_base = float(slip)
    cash = peak = 1.0
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
    entry_edge = 0.0
    entry_vol_anchor = 0.0
    events: list[dict[str, Any]] = []

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))

        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            effective_tp, effective_sl = _effective_deep_exits(owner, overlay, take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            reason = ""
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
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
                x_runner = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x_runner)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, _ = alpha3._try_immediate_limit(df, i, pos, path_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
                add_done = True

            if reason:
                x = _feature_vector(
                    df,
                    decisions,
                    deep_q,
                    base_cols,
                    i,
                    pos=pos,
                    owner=owner,
                    hold=hold,
                    unreal=unreal,
                    mfe=mfe,
                    mae=mae,
                    notional=notional,
                    parent_notional=parent_notional,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    max_hold=max_hold,
                    entry_edge=entry_edge,
                    entry_vol_anchor=entry_vol_anchor,
                    effective_tp=effective_tp,
                    effective_sl=effective_sl,
                )
                rewards = [
                    _exit_reward(df, i, pos, entry_price, notional, cfg, fee_base=fee_base, slip_base=slip_base)[0]
                    for cfg in arms
                ]
                label = int(np.argmax(np.asarray(rewards, dtype=np.float64)))
                events.append(
                    {
                        "x": x,
                        "label": label,
                        "rewards": np.asarray(rewards, dtype=np.float32),
                        "reason": reason,
                        "owner": owner,
                        "timestamp": str(df["timestamp"].iloc[i]),
                    }
                )
                filled, exit_px, exit_fee, _, _ = alpha3._try_immediate_limit(df, i, pos, path_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                add_done = False
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
            filled, px, entry_fee, _, _ = alpha3._try_immediate_limit(df, i, int(dec.side), path_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
            cash -= cash * entry_fee * notional
            mfe = mae = 0.0
            add_done = False
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, _ = alpha3._try_immediate_limit(df, i, side, path_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                mfe = mae = 0.0
                add_done = True
    return events


def _normalise_arrays(train_x: np.ndarray, x: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    train_z = ((train_x - mean) / std).astype(np.float32)
    if x is None:
        return mean, std, train_z
    return mean, std, ((x - mean) / std).astype(np.float32)


def _train_model(events: list[dict[str, Any]], n_arms: int, seed: int = 20260514) -> tuple[DeepExitOracleNet, dict[str, Any]]:
    if len(events) < 40:
        raise RuntimeError(f"not enough exit events to train deep exit oracle: {len(events)}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    x_all = np.stack([e["x"] for e in events]).astype(np.float32)
    y_all = np.asarray([int(e["label"]) for e in events], dtype=np.int64)
    split = max(8, int(len(events) * 0.80))
    x_fit, y_fit = x_all[:split], y_all[:split]
    x_hold, y_hold = x_all[split:], y_all[split:]
    mean, std, x_fit_z = _normalise_arrays(x_fit)
    x_hold_z = ((x_hold - mean) / std).astype(np.float32)
    counts = np.bincount(y_fit, minlength=n_arms).astype(np.float32)
    weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    weights = weights / np.mean(weights)
    model = DeepExitOracleNet(x_fit_z.shape[1], n_arms)
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    x_fit_t = torch.from_numpy(x_fit_z)
    y_fit_t = torch.from_numpy(y_fit)
    x_hold_t = torch.from_numpy(x_hold_z)
    y_hold_t = torch.from_numpy(y_hold)
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    stale = 0
    history: list[dict[str, float]] = []
    batch = min(64, max(16, len(x_fit_z) // 4))
    for epoch in range(1, 121):
        model.train()
        order = torch.randperm(len(x_fit_t))
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            logits = model(x_fit_t[idx])
            loss = loss_fn(logits, y_fit_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / max(len(x_fit_t), 1)
        model.eval()
        with torch.no_grad():
            hold_loss = float(loss_fn(model(x_hold_t), y_hold_t).detach()) if len(x_hold_t) else train_loss
            hold_acc = float((model(x_hold_t).argmax(dim=1) == y_hold_t).float().mean()) if len(x_hold_t) else 0.0
        history.append({"epoch": float(epoch), "train_loss": train_loss, "holdout_loss": hold_loss, "holdout_acc": hold_acc})
        if hold_loss < best_loss - 1e-5:
            best_loss = hold_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 15:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    meta = {
        "feature_mean": mean,
        "feature_std": std,
        "train_events": len(events),
        "fit_events": int(len(x_fit)),
        "holdout_events": int(len(x_hold)),
        "label_counts": counts.astype(int).tolist(),
        "best_holdout_loss": float(best_loss),
        "history": history,
    }
    return model.eval(), meta


def _select_arm(
    model: DeepExitOracleNet,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    policy: DeepExitPolicy,
    arm_index: dict[str, int],
) -> tuple[int, float]:
    if policy.mode == "fixed":
        return int(arm_index[policy.fixed_arm]), 1.0
    z = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(z[None, :])), dim=1).cpu().numpy()[0]
    idx = int(np.argmax(probs))
    conf = float(probs[idx])
    if conf < float(policy.min_confidence):
        return int(arm_index[policy.fallback_arm]), conf
    return idx, conf


def backtest_deep_exit(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    overlay: v31.OverlayConfig,
    entry_cfg: alpha3.ImmediateLimitConfig,
    arms: list[alpha3.ImmediateLimitConfig],
    base_cols: list[str],
    model: DeepExitOracleNet,
    mean: np.ndarray,
    std: np.ndarray,
    policy: DeepExitPolicy,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
    arm_index = {cfg.name: i for i, cfg in enumerate(arms)}
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
    entry_edge = 0.0
    entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    arm_counts: dict[str, int] = {}
    conf_sum = 0.0

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip_base) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_base)) / max(entry_price, 1e-12)
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
            effective_tp, effective_sl = _effective_deep_exits(owner, overlay, take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            reason = ""
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
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
                x_runner = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x_runner)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    filled, add_px, add_fee, _, route = alpha3._try_immediate_limit(df, i, pos, entry_cfg, entry=True, fee=fee_base, slip=slip_base)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
                        actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                    else:
                        actions["v21_add_on_limit_miss"] = actions.get("v21_add_on_limit_miss", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True

            if reason:
                x = _feature_vector(
                    df,
                    decisions,
                    deep_q,
                    base_cols,
                    i,
                    pos=pos,
                    owner=owner,
                    hold=hold,
                    unreal=unreal,
                    mfe=mfe,
                    mae=mae,
                    notional=notional,
                    parent_notional=parent_notional,
                    take_profit=take_profit,
                    stop_loss=stop_loss,
                    max_hold=max_hold,
                    entry_edge=entry_edge,
                    entry_vol_anchor=entry_vol_anchor,
                    effective_tp=effective_tp,
                    effective_sl=effective_sl,
                )
                arm_i, conf = _select_arm(model, x, mean, std, policy, arm_index)
                exit_cfg = arms[arm_i]
                arm_counts[exit_cfg.name] = arm_counts.get(exit_cfg.name, 0) + 1
                conf_sum += float(conf)
                filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(df, i, pos, exit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["exit_limit_miss_hold"] = actions.get("exit_limit_miss_hold", 0) + 1
                    continue
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
                add_done = False
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
            filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, int(dec.side), entry_cfg, entry=True, fee=fee_base, slip=slip_base)
            if not filled:
                actions["parent_entry_limit_miss"] = actions.get("parent_entry_limit_miss", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_counts[route] = route_counts.get(route, 0) + 1
            continue

        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, side, entry_cfg, entry=True, fee=fee_base, slip=slip_base)
                if not filled:
                    actions["deep_entry_limit_miss"] = actions.get("deep_entry_limit_miss", 0) + 1
                    route_counts[route] = route_counts.get(route, 0) + 1
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * entry_fee * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                route_counts[route] = route_counts.get(route, 0) + 1

    if pos != 0:
        exit_px = _fill_price(df, len(df) - 1, pos, slip_base, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_base * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        route_counts["forced_end_market"] = route_counts.get("forced_end_market", 0) + 1

    n = max(long_entries + short_entries, 1)
    return {
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
        "route_counts": route_counts,
        "arm_counts": arm_counts,
        "avg_oracle_confidence": float(conf_sum / max(sum(arm_counts.values()), 1)),
    }


def _metrics_deep(
    df: pd.DataFrame,
    stack: dict[str, Any],
    q: np.ndarray,
    decisions: pd.DataFrame,
    entry_cfg: alpha3.ImmediateLimitConfig,
    arms: list[alpha3.ImmediateLimitConfig],
    base_cols: list[str],
    model: DeepExitOracleNet,
    mean: np.ndarray,
    std: np.ndarray,
    policy: DeepExitPolicy,
) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_deep_exit(
            df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            q,
            decisions,
            stack["overlay"],
            entry_cfg,
            arms,
            base_cols,
            model,
            mean,
            std,
            policy,
            fee=stack["fee"],
            slip=stack["slip"],
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    arms = _arm_configs()
    arm_index = {cfg.name: i for i, cfg in enumerate(arms)}
    entry_cfg = arms[arm_index["baseline_exit2_pen05"]]
    base_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = _feature_names(base_cols)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < pd.Timestamp("2025-10-01"))].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding train/validation/eval decisions and V27 q", flush=True)
    train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    print(f"[{MODEL_ID}] collecting counterfactual exit-placement labels", flush=True)
    train_events = collect_exit_events(
        train_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        train_q,
        train_dec,
        stack["overlay"],
        entry_cfg,
        arms,
        base_cols,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    val_events = collect_exit_events(
        val_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        val_q,
        val_dec,
        stack["overlay"],
        entry_cfg,
        arms,
        base_cols,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    print(f"[{MODEL_ID}] train_events={len(train_events)} val_events={len(val_events)}", flush=True)
    model, train_meta = _train_model(train_events, len(arms))
    mean = np.asarray(train_meta["feature_mean"], dtype=np.float32)
    std = np.asarray(train_meta["feature_std"], dtype=np.float32)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "arms": [asdict(cfg) for cfg in arms],
            "feature_names": feature_names,
            "feature_mean": mean,
            "feature_std": std,
            "train_meta": {k: v for k, v in train_meta.items() if k not in {"feature_mean", "feature_std"}},
        },
        MODEL_OUT,
    )

    event_summary = {
        "train_events": len(train_events),
        "val_events": len(val_events),
        "train_label_counts": np.bincount([e["label"] for e in train_events], minlength=len(arms)).astype(int).tolist(),
        "val_label_counts": np.bincount([e["label"] for e in val_events], minlength=len(arms)).astype(int).tolist(),
        "arms": [cfg.name for cfg in arms],
    }
    EVENTS_OUT.write_text(json.dumps(event_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[{MODEL_ID}] selecting deep policy on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_deep: tuple[float, DeepExitPolicy, dict[str, Any]] | None = None
    best_any: tuple[float, DeepExitPolicy, dict[str, Any]] | None = None
    for policy in _policies():
        metrics = _metrics_deep(val_df, stack, val_q, val_dec, entry_cfg, arms, base_cols, model, mean, std, policy)
        score = _score(metrics)
        rows.append(
            {
                **asdict(policy),
                "selection_score": score,
                "val_cost1_pnl": metrics["cost1"]["pnl"],
                "val_cost1_mdd": metrics["cost1"]["mdd"],
                "val_cost1_trades": metrics["cost1"]["trades"],
                "val_cost2_pnl": metrics["cost2"]["pnl"],
                "val_cost3_pnl": metrics["cost3"]["pnl"],
                "val_cost1_arm_counts": json.dumps(metrics["cost1"].get("arm_counts", {}), sort_keys=True),
            }
        )
        print(
            f"[{MODEL_ID}] {policy.name} val c1={metrics['cost1']['pnl']:.2f} "
            f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )
        if best_any is None or score > best_any[0]:
            best_any = (score, policy, metrics)
        if policy.mode == "deep" and (best_deep is None or score > best_deep[0]):
            best_deep = (score, policy, metrics)
    assert best_deep is not None and best_any is not None
    selected_policy = best_deep[1]
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)

    print(f"[{MODEL_ID}] fixed 2026 OOS", flush=True)
    taker = alpha2._metrics(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        l2._variants()[0],
        fee=stack["fee"],
        slip=stack["slip"],
    )
    old_l2 = alpha2._metrics(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["selected_l2_variant"],
        fee=stack["fee"],
        slip=stack["slip"],
    )
    baseline = alpha3._metrics_signal_limit(
        eval_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        eval_q,
        eval_dec,
        stack["overlay"],
        entry_cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    front_policy = DeepExitPolicy("fixed_front_run_exit4_pen0", "fixed", 1.0, "exit4_pen0", fixed_arm="exit4_pen0")
    front_metrics = _metrics_deep(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, base_cols, model, mean, std, front_policy)
    deep_metrics = _metrics_deep(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, base_cols, model, mean, std, selected_policy)
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": "alpha3_baseline_exit2_pen05", "metrics": baseline, "score": _score(baseline)},
        {"name": "alpha3_fixed_front_run_exit4_pen0", "policy": asdict(front_policy), "metrics": front_metrics, "score": _score(front_metrics)},
        {"name": f"alpha3_deep_exit_oracle::{selected_policy.name}", "policy": asdict(selected_policy), "metrics": deep_metrics, "score": _score(deep_metrics)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    payload = torch.load(MODEL_OUT, map_location="cpu", weights_only=False)
    payload["selected_policy"] = asdict(selected_policy)
    payload["validation_best_any_policy"] = asdict(best_any[1])
    torch.save(payload, MODEL_OUT)

    warnings = [
        "signal_limit_fill_uses_5m_high_low_touch_proxy_not_queue_fill",
        "real_l2_queue_and_partial_fill_require_forward_shadow_validation",
        "deep_oracle_labels_are_counterfactual_to_ohlc_touch_model_not_real_queue_fills",
    ]
    if deep_metrics["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("deep_exit_oracle_did_not_improve_alpha3_cost1_pnl")
    if deep_metrics["cost1"]["mdd"] < baseline["cost1"]["mdd"]:
        warnings.append("deep_exit_oracle_worsened_alpha3_cost1_mdd")
    audit = {
        "status": "pass",
        "verdict": "shadow_candidate" if not any(w.startswith("deep_exit_oracle_did_not") or w.startswith("deep_exit_oracle_worsened") for w in warnings) else "iterate",
        "blocking": [],
        "warnings": warnings,
        "selection_uses_2026": False,
        "train_window": f"{TRAIN_START.date()}..2025-09-30",
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "selected_policy": asdict(selected_policy),
        "validation_best_any_policy": asdict(best_any[1]),
        "event_summary": event_summary,
        "model_path": str(MODEL_OUT),
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha3 Deep Exit Oracle. The Alpha3 decision stack and exit reasons are frozen. A small gated MLP learns a counterfactual reduce-only exit-placement policy from 2025 train exit events, then 2025Q4 selects a confidence/fallback policy. The oracle chooses among baseline 2bps/0.5-penetration and 0/1/2/3/4bps zero-penetration exit placements at each model-wide exit.",
        "papers_used": [
            "Deep optimal stopping: stop/continue as learned exercise policy.",
            "DeepHit/DeepSurv: nonlinear time-to-event hazard representation.",
            "Temporal Fusion Transformer: time-varying covariate representation inspiration; implemented as compact tabular-state MLP due small event count.",
            "Conformal risk control: confidence/fallback selection on validation rather than unconditional neural action.",
        ],
        "experiments": experiments,
        "selection_grid": str(GRID_OUT),
        "events": str(EVENTS_OUT),
        "model_path": str(MODEL_OUT),
        "audit": audit,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT), "events": str(EVENTS_OUT), "model": str(MODEL_OUT), "selected": selected_policy.name}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
