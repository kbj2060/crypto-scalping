#!/usr/bin/env python3
from __future__ import annotations

import json
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
from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha3_offline_rl_exit_layer_20260514"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_offline_rl_exit_layer_20260514"
MODEL_OUT = OUT_DIR / "offline_rl_exit_q.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_grid.csv"
DATASET_OUT = ROOT / "data/ensemble/reports/alpha3_offline_rl_exit_layer_20260514_dataset.json"
TRAIN_START = pd.Timestamp("2025-07-01")


@dataclass(frozen=True)
class OfflineRLPolicy:
    name: str
    q_margin: float
    min_advantage_conf: float
    min_hold: int
    exit_fallback_arm: str
    force_exit_mode: str = "q_or_fallback"


class ExitQNet(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, hidden: int = 160, dropout: float = 0.10) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _action_names(arms: list[alpha3.ImmediateLimitConfig]) -> list[str]:
    return ["hold"] + [a.name for a in arms]


def _policies() -> list[OfflineRLPolicy]:
    rows: list[OfflineRLPolicy] = []
    for fallback in ("exit4_pen0", "baseline_exit2_pen05"):
        for margin in (0.000, 0.001, 0.002, 0.004, 0.008):
            for conf in (0.000, 0.001, 0.002):
                rows.append(
                    OfflineRLPolicy(
                        name=f"fq_exit_m{margin:.3f}_c{conf:.3f}_fb_{fallback}",
                        q_margin=float(margin),
                        min_advantage_conf=float(conf),
                        min_hold=1,
                        exit_fallback_arm=fallback,
                    )
                )
    rows.append(OfflineRLPolicy("placement_only_q_exit4_fallback", 99.0, 99.0, 999, "exit4_pen0"))
    rows.append(OfflineRLPolicy("placement_only_q_baseline_fallback", 99.0, 99.0, 999, "baseline_exit2_pen05"))
    rows.append(OfflineRLPolicy("fixed_baseline_exit2_pen05", 99.0, 99.0, 999, "baseline_exit2_pen05", force_exit_mode="fallback"))
    rows.append(OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", force_exit_mode="fallback"))
    return rows


def _normalise_fit(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std, ((x - mean) / std).astype(np.float32)


def _q_from_model(model: ExitQNet, x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    z = ((x.astype(np.float32) - mean) / std).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(z[None, :])).cpu().numpy()[0].astype(np.float64)


def _select_action(
    model: ExitQNet,
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    policy: OfflineRLPolicy,
    action_names: list[str],
    *,
    force_exit: bool,
) -> tuple[int, float, np.ndarray]:
    q = _q_from_model(model, x, mean, std)
    hold_q = float(q[0])
    if force_exit:
        exit_slice = q[1:]
        arm_i = int(np.argmax(exit_slice)) + 1
        best_exit_q = float(q[arm_i])
        fallback_i = action_names.index(policy.exit_fallback_arm)
        if policy.force_exit_mode == "fallback":
            return fallback_i, best_exit_q - hold_q, q
        if best_exit_q < float(q[fallback_i]) - 0.002:
            return fallback_i, best_exit_q - hold_q, q
        return arm_i, best_exit_q - hold_q, q
    best_i = int(np.argmax(q))
    best_q = float(q[best_i])
    exit_i = int(np.argmax(q[1:])) + 1
    exit_adv = float(q[exit_i] - hold_q)
    sorted_q = np.sort(q)
    conf = float(sorted_q[-1] - sorted_q[-2]) if len(sorted_q) > 1 else 0.0
    if best_i != 0 and exit_adv >= float(policy.q_margin) and conf >= float(policy.min_advantage_conf):
        return exit_i, exit_adv, q
    return 0, exit_adv, q


def _state_record(
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
    entry_price: float,
) -> dict[str, Any]:
    return {
        "idx": int(idx),
        "pos": int(pos),
        "entry_price": float(entry_price),
        "notional": float(notional),
        "x": deep_exit._feature_vector(
            df,
            decisions,
            deep_q,
            base_cols,
            idx,
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
        ),
    }


def _episode_q_targets(
    df: pd.DataFrame,
    episode: list[dict[str, Any]],
    arms: list[alpha3.ImmediateLimitConfig],
    *,
    fee: float,
    slip: float,
    gamma: float = 0.995,
    step_penalty: float = 0.00004,
    conservative_penalty: float = 0.0015,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    if not episode:
        return [], []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    next_v = -1e6
    n_actions = len(arms) + 1
    for k in range(len(episode) - 1, -1, -1):
        st = episode[k]
        q = np.full(n_actions, -1e6, dtype=np.float32)
        if k < len(episode) - 1:
            q[0] = float(gamma * next_v - step_penalty)
        for j, cfg in enumerate(arms, start=1):
            r, _ = deep_exit._exit_reward(
                df,
                int(st["idx"]),
                int(st["pos"]),
                float(st["entry_price"]),
                float(st["notional"]),
                cfg,
                fee_base=float(fee),
                slip_base=float(slip),
            )
            if cfg.name != "baseline_exit2_pen05":
                r -= conservative_penalty
            q[j] = float(r)
        v = float(np.max(q))
        xs.append(np.asarray(st["x"], dtype=np.float32))
        ys.append(q)
        next_v = v
    xs.reverse()
    ys.reverse()
    return xs, ys


def collect_q_dataset(
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
    *,
    fee: float,
    slip: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    close = _close(df)
    cash = peak = 1.0
    pos = 0
    owner = ""
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = deep_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    entry_edge = 0.0
    entry_vol_anchor = 0.0
    episode: list[dict[str, Any]] = []
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    episodes = 0
    state_owners: dict[str, int] = {}

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
        unreal = raw * notional
        return cash * (1.0 + unreal), unreal

    def flush() -> None:
        nonlocal episode, episodes
        if not episode:
            return
        ex, ey = _episode_q_targets(df, episode, arms, fee=fee, slip=slip)
        xs.extend(ex)
        ys.extend(ey)
        episodes += 1
        episode = []

    for i in range(0, len(df) - 2):
        eq, unreal = mark(i)
        peak = max(peak, eq)
        dd_abs = max(0.0, 1.0 - eq / max(peak, 1e-12))
        if pos != 0:
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            hold = i - entry_idx
            effective_tp, effective_sl = deep_exit._effective_deep_exits(owner, overlay, take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            reason = ""
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
            if hold >= 1:
                episode.append(
                    _state_record(
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
                        entry_price=entry_price,
                    )
                )
                state_owners[owner] = state_owners.get(owner, 0) + 1
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
                    filled, add_px, add_fee, _, _ = alpha3._try_immediate_limit(df, i, pos, entry_cfg, entry=True, fee=fee, slip=slip)
                    if filled and delta > 0.0:
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        cash -= cash * add_fee * delta
                        notional = new_notional
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
                add_done = True
            if reason:
                flush()
                filled, exit_px, exit_fee, _, _ = alpha3._try_immediate_limit(df, i, pos, entry_cfg, entry=False, fee=fee, slip=slip)
                if filled:
                    raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                    cash = cash * (1.0 + raw * notional)
                    cash -= cash * exit_fee * notional
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
            filled, px, entry_fee, _, _ = alpha3._try_immediate_limit(df, i, int(dec.side), entry_cfg, entry=True, fee=fee, slip=slip)
            if not filled:
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
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
                filled, px, entry_fee, _, _ = alpha3._try_immediate_limit(df, i, side, entry_cfg, entry=True, fee=fee, slip=slip)
                if not filled:
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
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
    flush()
    if not xs:
        raise RuntimeError("no offline RL states collected")
    meta = {
        "episodes": int(episodes),
        "states": int(len(xs)),
        "state_owners": state_owners,
        "actions": _action_names(arms),
    }
    return np.stack(xs).astype(np.float32), np.stack(ys).astype(np.float32), meta


def _train_q_model(x: np.ndarray, y: np.ndarray, seed: int = 20260514) -> tuple[ExitQNet, dict[str, Any]]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    split = max(32, int(len(x) * 0.80))
    x_fit, y_fit = x[:split], y[:split]
    x_hold, y_hold = x[split:], y[split:]
    mean, std, x_fit_z = _normalise_fit(x_fit)
    x_hold_z = ((x_hold - mean) / std).astype(np.float32)
    model = ExitQNet(x.shape[1], y.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=7e-4, weight_decay=2e-4)
    loss_fn = nn.SmoothL1Loss()
    x_fit_t = torch.from_numpy(x_fit_z)
    y_fit_t = torch.from_numpy(y_fit)
    x_hold_t = torch.from_numpy(x_hold_z)
    y_hold_t = torch.from_numpy(y_hold)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    batch = min(128, max(32, len(x_fit) // 5))
    for epoch in range(1, 161):
        model.train()
        order = torch.randperm(len(x_fit_t))
        total = 0.0
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            pred = model(x_fit_t[idx])
            loss = loss_fn(pred, y_fit_t[idx])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
            total += float(loss.detach()) * len(idx)
        train_loss = total / max(len(x_fit_t), 1)
        model.eval()
        with torch.no_grad():
            hold_loss = float(loss_fn(model(x_hold_t), y_hold_t).detach()) if len(x_hold_t) else train_loss
            hold_acc = float((model(x_hold_t).argmax(1) == y_hold_t.argmax(1)).float().mean()) if len(x_hold_t) else 0.0
        history.append({"epoch": float(epoch), "train_loss": train_loss, "holdout_loss": hold_loss, "holdout_action_acc": hold_acc})
        if hold_loss < best_loss - 1e-6:
            best_loss = hold_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= 20:
            break
    assert best_state is not None
    model.load_state_dict(best_state)
    return model.eval(), {
        "feature_mean": mean,
        "feature_std": std,
        "fit_states": int(len(x_fit)),
        "holdout_states": int(len(x_hold)),
        "best_holdout_loss": float(best_loss),
        "history": history,
    }


def backtest_rl_exit(
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
    model: ExitQNet,
    mean: np.ndarray,
    std: np.ndarray,
    policy: OfflineRLPolicy,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    close = _close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
    actions = _action_names(arms)
    arm_by_name = {a.name: a for a in arms}
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
    runner_actions: dict[str, int] = {}
    route_counts: dict[str, int] = {}
    rl_action_counts: dict[str, int] = {}
    adv_sum = 0.0

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
            effective_tp, effective_sl = deep_exit._effective_deep_exits(owner, overlay, take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            base_reason = ""
            if effective_tp > 0.0 and unreal >= effective_tp:
                base_reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                base_reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                base_reason = f"{owner}_max_hold"
            reason = base_reason
            selected_exit_cfg: alpha3.ImmediateLimitConfig | None = None
            if base_reason or hold >= int(policy.min_hold):
                x = deep_exit._feature_vector(
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
                action_i, adv, _ = _select_action(model, x, mean, std, policy, actions, force_exit=bool(base_reason))
                adv_sum += float(adv)
                act_name = actions[action_i]
                rl_action_counts[act_name] = rl_action_counts.get(act_name, 0) + 1
                if action_i > 0:
                    selected_exit_cfg = arm_by_name[act_name]
                    if not reason:
                        reason = f"{owner}_offline_rl_exit"
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
                        cash -= cash * add_fee * delta
                        notional = new_notional
                        entry_vol_anchor = max(entry_vol_anchor, v31._vol_anchor(df.iloc[i]) * notional)
                        runner_actions["v21_add_on"] = runner_actions.get("v21_add_on", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                    else:
                        runner_actions["v21_add_on_limit_miss"] = runner_actions.get("v21_add_on_limit_miss", 0) + 1
                        route_counts[route] = route_counts.get(route, 0) + 1
                else:
                    runner_actions["v21_reject"] = runner_actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                if selected_exit_cfg is None:
                    selected_exit_cfg = arm_by_name[policy.exit_fallback_arm]
                filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(df, i, pos, selected_exit_cfg, entry=False, fee=fee_base, slip=slip_base)
                if not filled:
                    runner_actions["exit_limit_miss_hold"] = runner_actions.get("exit_limit_miss_hold", 0) + 1
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
                runner_actions["parent_entry_limit_miss"] = runner_actions.get("parent_entry_limit_miss", 0) + 1
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
            runner_actions["v21_entry"] = runner_actions.get("v21_entry", 0) + 1
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
                    runner_actions["deep_entry_limit_miss"] = runner_actions.get("deep_entry_limit_miss", 0) + 1
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
                runner_actions["deep_entry"] = runner_actions.get("deep_entry", 0) + 1
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
    rl_calls = max(sum(rl_action_counts.values()), 1)
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
        "runner_actions": runner_actions,
        "route_counts": route_counts,
        "rl_action_counts": rl_action_counts,
        "avg_rl_advantage": float(adv_sum / rl_calls),
    }


def _metrics_rl(
    df: pd.DataFrame,
    stack: dict[str, Any],
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    entry_cfg: alpha3.ImmediateLimitConfig,
    arms: list[alpha3.ImmediateLimitConfig],
    base_cols: list[str],
    model: ExitQNet,
    mean: np.ndarray,
    std: np.ndarray,
    policy: OfflineRLPolicy,
) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_rl_exit(
            df,
            stack["parent"],
            stack["jackpot_model"],
            stack["add_cfg"],
            deep_q,
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


def _serialise_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(metrics, default=_json_default))


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    arms = deep_exit._arm_configs()
    arm_by_name = {a.name: a for a in arms}
    entry_cfg = arm_by_name["baseline_exit2_pen05"]
    feature_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = deep_exit._feature_names(feature_cols)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[
        (train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < pd.Timestamp("2025-10-01"))
    ].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions and frozen V27 q", flush=True)
    train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    print(f"[{MODEL_ID}] collecting counterfactual exit-placement Q targets", flush=True)
    x, y, dataset_meta = collect_q_dataset(
        train_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        train_q,
        train_dec,
        stack["overlay"],
        entry_cfg,
        arms,
        feature_cols,
        fee=stack["fee"],
        slip=stack["slip"],
    )
    label_counts = np.bincount(np.argmax(y, axis=1), minlength=len(arms) + 1).astype(int).tolist()
    dataset_summary = {
        **dataset_meta,
        "train_start": str(train_df["timestamp"].iloc[0]) if len(train_df) else None,
        "train_end": str(train_df["timestamp"].iloc[-1]) if len(train_df) else None,
        "target_argmax_counts": dict(zip(_action_names(arms), label_counts)),
        "target_mean_by_action": dict(zip(_action_names(arms), np.mean(y, axis=0).astype(float).tolist())),
        "conservative_penalty_for_non_baseline_exit": 0.0015,
    }
    DATASET_OUT.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"[{MODEL_ID}] training compact fitted-Q network", flush=True)
    q_model, train_meta = _train_q_model(x, y)
    mean = train_meta["feature_mean"]
    std = train_meta["feature_std"]
    torch.save(
        {
            "model_id": MODEL_ID,
            "model_state": q_model.state_dict(),
            "input_dim": len(feature_names),
            "actions": _action_names(arms),
            "arms": [asdict(cfg) for cfg in arms],
            "feature_names": feature_names,
            "feature_mean": mean,
            "feature_std": std,
            "train_meta": {k: v for k, v in train_meta.items() if k not in {"feature_mean", "feature_std"}},
            "dataset_meta": dataset_summary,
        },
        MODEL_OUT,
    )

    print(f"[{MODEL_ID}] selecting RL policy on 2025Q4", flush=True)
    rows: list[dict[str, Any]] = []
    best_rl: tuple[float, OfflineRLPolicy, dict[str, Any]] | None = None
    best_placement: tuple[float, OfflineRLPolicy, dict[str, Any]] | None = None
    best_any: tuple[float, OfflineRLPolicy, dict[str, Any]] | None = None
    for policy in _policies():
        metrics = _metrics_rl(val_df, stack, val_q, val_dec, entry_cfg, arms, feature_cols, q_model, mean, std, policy)
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
                "val_cost1_rl_action_counts": json.dumps(metrics["cost1"].get("rl_action_counts", {}), sort_keys=True),
                "val_cost1_route_counts": json.dumps(metrics["cost1"].get("route_counts", {}), sort_keys=True),
            }
        )
        print(
            f"[{MODEL_ID}] {policy.name} val c1={metrics['cost1']['pnl']:.2f} "
            f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )
        if best_any is None or score > best_any[0]:
            best_any = (score, policy, metrics)
        if policy.name.startswith("fq_exit_") and (best_rl is None or score > best_rl[0]):
            best_rl = (score, policy, metrics)
        if policy.name.startswith("placement_only_") and (best_placement is None or score > best_placement[0]):
            best_placement = (score, policy, metrics)
    assert best_rl is not None and best_placement is not None and best_any is not None
    selected_policy = best_placement[1]
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
    fixed_baseline_policy = OfflineRLPolicy("fixed_baseline_exit2_pen05", 99.0, 99.0, 999, "baseline_exit2_pen05", "fallback")
    fixed_front_policy = OfflineRLPolicy("fixed_front_run_exit4_pen0", 99.0, 99.0, 999, "exit4_pen0", "fallback")
    fixed_baseline = _metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, fixed_baseline_policy)
    fixed_front = _metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, fixed_front_policy)
    rl_metrics = _metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, selected_policy)
    best_any_metrics = _metrics_rl(eval_df, stack, eval_q, eval_dec, entry_cfg, arms, feature_cols, q_model, mean, std, best_any[1])
    experiments = [
        {"name": "alpha2_1_next_open_taker_control", "metrics": taker, "score": _score(taker)},
        {"name": "alpha2_1_old_l2_replay_fee20_control", "metrics": old_l2, "score": _score(old_l2)},
        {"name": "alpha3_baseline_exit2_pen05", "metrics": baseline, "score": _score(baseline)},
        {"name": "alpha3_fixed_baseline_noearly_rl_path", "policy": asdict(fixed_baseline_policy), "metrics": fixed_baseline, "score": _score(fixed_baseline)},
        {"name": "alpha3_fixed_front_run_exit4_pen0", "policy": asdict(fixed_front_policy), "metrics": fixed_front, "score": _score(fixed_front)},
        {"name": f"alpha3_offline_rl_exit_layer::{selected_policy.name}", "policy": asdict(selected_policy), "metrics": rl_metrics, "score": _score(rl_metrics)},
        {"name": f"alpha3_best_any_selection::{best_any[1].name}", "policy": asdict(best_any[1]), "metrics": best_any_metrics, "score": _score(best_any_metrics)},
    ]
    for exp in experiments:
        m = exp["metrics"]
        print(
            f"[{MODEL_ID}] {exp['name']} c1={m['cost1']['pnl']:.2f} mdd={m['cost1']['mdd']:.2f} "
            f"c2={m['cost2']['pnl']:.2f} c3={m['cost3']['pnl']:.2f}",
            flush=True,
        )

    report = {
        "model_id": MODEL_ID,
        "date": "2026-05-14",
        "design": {
            "algorithm": "compact discrete fitted-Q / CQL-style conservative offline RL",
            "scope": "Alpha3 stack frozen; RL only selects reduce-only exit placement arm at existing exit events unless validation-selected policy explicitly fires early.",
            "why_not_ppo": "Historical fixed OHLCV/L2-proxy logs provide offline counterfactual arm rewards; PPO needs online/on-policy interaction and is high-overfit here.",
            "entry_contract": asdict(entry_cfg),
            "actions": _action_names(arms),
            "selection_split": "2025Q4",
            "oos_split": "2026 full eval set",
            "selection_uses_2026": False,
        },
        "dataset": dataset_summary,
        "train_meta": {k: v for k, v in train_meta.items() if k not in {"feature_mean", "feature_std", "history"}},
        "selected_rl_policy": asdict(selected_policy),
        "selected_any_policy": asdict(best_any[1]),
        "validation_best_early_rl_policy": asdict(best_rl[1]),
        "validation_best_early_rl_score": float(best_rl[0]),
        "validation_best_placement_score": float(best_placement[0]),
        "validation_best_any_score": float(best_any[0]),
        "experiments": experiments,
        "artifacts": {
            "model": str(MODEL_OUT.relative_to(ROOT)),
            "grid": str(GRID_OUT.relative_to(ROOT)),
            "dataset": str(DATASET_OUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(_serialise_metrics(report), indent=2, ensure_ascii=False), encoding="utf-8")

    audit = {
        "model_id": MODEL_ID,
        "status": "shadow_candidate",
        "selection_uses_2026": False,
        "base_contract": "docs/model_contracts/alpha3_teacher_l2_limit_fallback_20260514_contract.md",
        "front_run_contract": "docs/model_contracts/alpha3_exit_front_run_layer_20260514_contract.md",
        "causality": [
            "Train: 2025-07-01..2025-09-30 only.",
            "Selection: 2025-10-01..2025-12-31 only.",
            "2026 used only once for fixed OOS reporting.",
            "RL state uses live-available current position, current bar features, Alpha3 decision outputs, and frozen V27 q only.",
        ],
        "limitations": [
            "Backtest still uses 5m high/low immediate-limit touch proxy.",
            "No real queue position, partial fill, post-only reject, or L2 replay validation yet.",
            "The selected RL policy may overfit because Alpha3 exit events are sparse and target labels are dominated by exit4_pen0.",
            "Production promotion requires live shadow route/fallback audit against fixed exit4_pen0.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{MODEL_ID}] wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
