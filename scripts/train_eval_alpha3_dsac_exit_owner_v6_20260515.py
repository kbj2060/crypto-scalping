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
from sklearn.cross_decomposition import PLSRegression
from sklearn.impute import SimpleImputer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.train_rl_dsac_agent import DSACAgent  # noqa: E402
from scripts import eval_alpha2_1_signal_immediate_limit_20260514 as alpha3  # noqa: E402
from scripts import eval_alpha2_teacher_l2_runtime_sweep_20260514 as alpha2  # noqa: E402
from scripts import eval_alpha3_deep_exit_oracle_20260514 as deep_exit  # noqa: E402
from scripts import eval_alpha3_exit_front_run_layer_20260514 as front_run  # noqa: E402
from scripts import eval_alpha3_rl_exit_owner_fulltrain_20260514 as base_exit  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402


MODEL_ID = "alpha3_dsac_exit_owner_v6_20260515"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha3_dsac_exit_owner_v6_20260515"
MODEL_OUT = OUT_DIR / "dsac_exit_owner_v6.pt"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_v6_20260515_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_v6_20260515_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_v6_20260515_grid.csv"
DATASET_OUT = ROOT / "data/ensemble/reports/alpha3_dsac_exit_owner_v6_20260515_dataset.json"
FEATURE_ANALYSIS_REPORT = ROOT / "data/ensemble/reports/alpha3_exit_owner_feature_analysis_20260515.json"
TRAIN_START = pd.Timestamp("2025-01-01")
TRAIN_END = pd.Timestamp("2025-10-01")
VAL_START = pd.Timestamp("2025-10-01")
TRAIN_UPDATES = 18000


EXPANDED_ACTIONS = ["hold", "close_50", "close_100"]

ACTION_SCALARS = {
    "hold": 0.0,
    "close_50": 0.65,
    "close_100": -0.75,
}

STATE16 = [
    "effective_tp",
    "notional",
    "parent_notional",
    "q_margin",
    "owner_deep",
    "q_same",
    "mfe",
    "owner_parent",
    "hold_norm",
    "q_opp",
    "unreal",
    "giveback",
    "mae",
    "effective_sl",
    "row_vol_anchor",
    "pos",
]

TOP_BASE20 = [
    "volatility_z",
    "long_squeeze_risk",
    "clean_regime_2024_unsup_v4_confidence",
    "rogers_satchell_vol",
    "clean_regime_2024_unsup_v4_entropy",
    "whale_retail_ratio",
    "trade_intensity",
    "clean_regime_2024_unsup_v4_cluster",
    "m7_qwidth",
    "ai_anchor_trend_escape_prob",
    "clean_regime_2024_unsup_v4_whipsaw_prob",
    "ai_adverse_risk",
    "tide_vol_raw",
    "ai_anchor_revert_prob",
    "squeeze_power",
    "clean_regime_2024_unsup_v4_cluster_prob_2",
    "ai_flow_flip_prob",
    "clean_regime_2024_unsup_v4_factor_vol",
    "clean_regime_2024_unsup_v4_cluster_prob_3",
    "big_trade_ratio",
]

FEATURE_MODES = ["state16_topbase20"]


def _resolve_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _score(metrics: dict[str, Any]) -> float:
    return alpha2._score(metrics["cost1"], metrics["cost2"], metrics["cost3"])


def _normalise(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0).astype(np.float32)
    std = x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std, ((x - mean) / std).astype(np.float32)


@dataclass
class FeatureEncoder:
    mode: str
    feature_names: list[str]
    raw_feature_names: list[str]
    raw_indices: list[int]
    base_indices: list[int]
    pls_feature_names: list[str]
    imputer: SimpleImputer | None = None
    pls: PLSRegression | None = None

    def transform(self, x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        one_dim = arr.ndim == 1
        if one_dim:
            arr = arr[None, :]
        chunks: list[np.ndarray] = []
        if self.raw_indices:
            chunks.append(arr[:, self.raw_indices].astype(np.float32))
        if self.pls is not None and self.imputer is not None:
            base = self.imputer.transform(arr[:, self.base_indices].astype(np.float64))
            chunks.append(self.pls.transform(base).astype(np.float32))
        if not chunks:
            out = np.empty((arr.shape[0], 0), dtype=np.float32)
        else:
            out = np.concatenate(chunks, axis=1).astype(np.float32)
        out[~np.isfinite(out)] = 0.0
        return out[0] if one_dim else out

    @property
    def out_dim(self) -> int:
        return len(self.raw_feature_names) + len(self.pls_feature_names)


def _projection_targets(y: np.ndarray) -> np.ndarray:
    best_i = np.argmax(y, axis=1)
    close_label = (best_i != 0).astype(np.float64)
    best_exit_q = np.max(y[:, 1:], axis=1)
    adv = np.clip(best_exit_q - y[:, 0], -0.25, 0.25)
    best_q = np.clip(np.max(y, axis=1), -0.25, 0.25)
    return np.column_stack([close_label, adv, best_q]).astype(np.float64)


def _build_feature_encoder(mode: str, x_full: np.ndarray, y: np.ndarray, feature_names: list[str]) -> FeatureEncoder:
    idx = {name: i for i, name in enumerate(feature_names)}
    missing = [name for name in STATE16 + TOP_BASE20 if name not in idx]
    if missing:
        raise RuntimeError(f"feature mode {mode} missing features: {missing}")
    raw_names = list(STATE16)
    if "topbase20" in mode:
        raw_names.extend(TOP_BASE20)
    raw_names = list(dict.fromkeys(raw_names))
    raw_indices = [idx[name] for name in raw_names]
    base_indices = list(range(len(feature_names) - 30))
    pls_names: list[str] = []
    imputer: SimpleImputer | None = None
    pls: PLSRegression | None = None
    if "base_pls8" in mode:
        imputer = SimpleImputer(strategy="median")
        base_x = imputer.fit_transform(x_full[:, base_indices].astype(np.float64))
        pls = PLSRegression(n_components=8, scale=True)
        pls.fit(base_x, _projection_targets(y))
        pls_names = [f"base_pls_{i}" for i in range(8)]
    return FeatureEncoder(
        mode=mode,
        feature_names=raw_names + pls_names,
        raw_feature_names=raw_names,
        raw_indices=raw_indices,
        base_indices=base_indices,
        pls_feature_names=pls_names,
        imputer=imputer,
        pls=pls,
    )


def _feature_meta(encoder: FeatureEncoder) -> dict[str, Any]:
    return {
        "mode": encoder.mode,
        "out_dim": encoder.out_dim,
        "raw_feature_names": encoder.raw_feature_names,
        "pls_feature_names": encoder.pls_feature_names,
        "feature_names": encoder.feature_names,
        "source": "Alpha3 exit-owner feature analysis 20260515: keep state16 raw; optionally add top_base20 raw and/or target-aware base PLS-8.",
    }


def _encoder_payload(encoder: FeatureEncoder) -> dict[str, Any]:
    return {
        **_feature_meta(encoder),
        "raw_indices": encoder.raw_indices,
        "base_indices": encoder.base_indices,
        "imputer": encoder.imputer,
        "pls": encoder.pls,
    }


def _scalar_to_action_name(a: float, *, force_exit: bool) -> str:
    x = float(np.clip(a, -1.0, 1.0))
    if -0.25 <= x <= 0.25:
        return "close_100" if force_exit else "hold"
    if x < -0.25:
        return "close_100"
    return "close_50"


def _select_action_dsac(
    agent: DSACAgent,
    x: np.ndarray,
    encoder: FeatureEncoder,
    mean: np.ndarray,
    std: np.ndarray,
    policy: base_exit.OfflineRLPolicy,
    *,
    force_exit: bool,
) -> tuple[str, float, np.ndarray]:
    x_enc = encoder.transform(x)
    z = ((x_enc.astype(np.float32) - mean) / std).astype(np.float32)
    scalar = float(agent.act(z, deterministic=True))
    name = _scalar_to_action_name(scalar, force_exit=force_exit)
    return str(name), scalar, np.array([scalar], dtype=np.float64)


def _runtime_policies() -> list[base_exit.OfflineRLPolicy]:
    rows: list[base_exit.OfflineRLPolicy] = []
    for min_hold in (6, 12, 18, 24, 32):
        for fallback in ("next_open_limit_touch0_fee20",):
            rows.append(
                base_exit.OfflineRLPolicy(
                    name=f"dsac_exit_owner_minhold{min_hold}_fb_{fallback}",
                    q_margin=0.0,
                    min_advantage_conf=0.0,
                    min_hold=int(min_hold),
                    exit_fallback_arm=fallback,
                    force_exit_mode="dsac_actor",
                )
            )
    return rows


def _expanded_targets(base_y: np.ndarray, base_actions: list[str], x_full: np.ndarray, feature_names: list[str]) -> np.ndarray:
    idx = {name: i for i, name in enumerate(base_actions)}
    fidx = {name: i for i, name in enumerate(feature_names)}
    hold = base_y[:, idx["hold"]]
    close = base_y[:, idx["next_open_limit_touch0_fee20"]]
    giveback = np.clip(x_full[:, fidx["giveback"]], 0.0, 0.25)
    mae = np.clip(-x_full[:, fidx["mae"]], 0.0, 0.25)
    hold_norm = np.clip(x_full[:, fidx["hold_norm"]], 0.0, 4.0)
    unreal = x_full[:, fidx["unreal"]]
    mfe = np.clip(x_full[:, fidx["mfe"]], 0.0, 0.50)
    effective_sl = np.abs(x_full[:, fidx["effective_sl"]])
    exit_adv = close - hold
    near_sl = (effective_sl > 1e-6) & ((-unreal) >= 0.55 * effective_sl)
    profit_lock = (unreal >= 0.010) & (giveback >= np.maximum(0.0035, 0.24 * np.maximum(mfe, 0.0)))
    strong_profit = (unreal >= 0.018) & (mfe >= 0.024)
    mature = hold_norm >= (3.0 / 64.0)
    exit_context = near_sl | profit_lock | ((exit_adv >= 0.0040) & (giveback >= 0.0040))
    partial_context = strong_profit & (giveback >= 0.0030) & mature

    close100 = close - 0.0040 - 0.015 * np.maximum(unreal, 0.0)
    close50 = 0.50 * close + 0.50 * hold - 0.0040 - 0.050 * mae
    hold_guard = hold - np.maximum(0.0, close - hold) * 0.025 - 0.010 * giveback - 0.002 * hold_norm

    invalid = np.full_like(hold, -1e6, dtype=np.float32)
    close100 = np.where(exit_context, close100, invalid)
    close50 = np.where(partial_context, close50, invalid)
    y = np.stack(
        [
            hold_guard,
            close50,
            close100,
        ],
        axis=1,
    ).astype(np.float32)
    return y


def _populate_dsac_replay(
    agent: DSACAgent,
    x_z: np.ndarray,
    y: np.ndarray,
    action_names: list[str],
    *,
    reward_scale: float = 12.0,
    reward_clip: float = 5.0,
) -> dict[str, Any]:
    pushed = 0
    action_counts: dict[str, int] = {}
    reward_sum: dict[str, float] = {}
    for i in range(len(x_z)):
        state = x_z[i].astype(np.float32)
        next_state = state
        for j, name in enumerate(action_names):
            q = float(y[i, j])
            if not np.isfinite(q) or q < -1e5:
                continue
            scalar = float(ACTION_SCALARS[name])
            reward = float(np.clip(q * reward_scale, -reward_clip, reward_clip))
            agent.memory.push(state, scalar, reward, next_state, 1.0)
            pushed += 1
            action_counts[name] = action_counts.get(name, 0) + 1
            reward_sum[name] = reward_sum.get(name, 0.0) + reward
    reward_mean = {k: float(reward_sum[k] / max(action_counts.get(k, 1), 1)) for k in action_counts}
    return {"pushed": int(pushed), "action_counts": action_counts, "reward_mean": reward_mean}


def _market_exit_price(df: pd.DataFrame, i: int, pos: int, slip: float) -> float:
    fill_i = min(int(i) + 1, len(df) - 1)
    return alpha3._fallback_close_price(df, fill_i, pos, slip, entry=False)


def _apply_exit_fill(
    df: pd.DataFrame,
    i: int,
    *,
    pos: int,
    entry_price: float,
    cash: float,
    notional: float,
    fraction: float,
    action_name: str,
    arm_by_name: dict[str, alpha3.ImmediateLimitConfig],
    fee_base: float,
    slip_base: float,
) -> tuple[bool, float, float, float, str]:
    frac = float(np.clip(fraction, 0.0, 1.0))
    exit_notional = float(notional * frac)
    if exit_notional <= 1e-12:
        return False, cash, notional, 0.0, "zero_exit_notional"
    filled, exit_px, exit_fee, _, route = alpha3._try_immediate_limit(
        df,
        i,
        pos,
        arm_by_name["next_open_limit_touch0_fee20"],
        entry=False,
        fee=fee_base,
        slip=slip_base,
    )
    if not filled:
        return False, cash, notional, 0.0, route
    raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
    before = cash
    cash = cash * (1.0 + raw * exit_notional)
    cash -= before * exit_fee * exit_notional
    new_notional = max(0.0, notional - exit_notional)
    return True, float(cash), float(new_notional), float(raw * exit_notional), route


def _guard_action(
    action_name: str,
    *,
    force_exit: bool,
    hold: int,
    unreal: float,
    mfe: float,
    mae: float,
    effective_sl: float,
    reduce_steps: int,
    last_reduce_hold: int,
) -> str:
    if force_exit:
        return "close_100"
    giveback = max(0.0, float(mfe - unreal))
    near_sl = effective_sl > 0.0 and unreal <= -0.55 * abs(effective_sl)
    profit_lock = unreal >= 0.010 and giveback >= max(0.0035, 0.24 * max(mfe, 0.0))
    if action_name == "close_100":
        if near_sl or profit_lock or mae <= -0.016 or unreal <= -0.014:
            return action_name
        return "hold"
    if action_name == "close_50":
        if reduce_steps >= 1 or hold - last_reduce_hold < 8:
            return "hold"
        if unreal < 0.018 or mfe < 0.024 or giveback < 0.0040 or hold < 3:
            return "hold"
        return action_name
    return action_name


def backtest_dsac_v2(
    df: pd.DataFrame,
    stack: dict[str, Any],
    deep_q: np.ndarray,
    decisions: pd.DataFrame,
    entry_cfg: alpha3.ImmediateLimitConfig,
    arms: list[alpha3.ImmediateLimitConfig],
    base_cols: list[str],
    agent: DSACAgent,
    encoder: FeatureEncoder,
    mean: np.ndarray,
    std: np.ndarray,
    policy: base_exit.OfflineRLPolicy,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
) -> dict[str, Any]:
    close = base_exit._close(df)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
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
    partial_exits = 0
    reduce_steps = 0
    last_reduce_hold = -9999

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
            effective_tp, effective_sl = deep_exit._effective_deep_exits(owner, stack["overlay"], take_profit, stop_loss, entry_edge, entry_vol_anchor, hold, mfe)
            base_reason = ""
            if effective_sl > 0.0 and unreal <= -abs(effective_sl):
                base_reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                base_reason = f"{owner}_max_hold"
            elif effective_tp > 0.0 and unreal >= effective_tp and (mfe - unreal) >= max(0.0015, 0.18 * max(mfe, 0.0)):
                base_reason = f"{owner}_profit_lock"
            reason = base_reason
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
                action_name, _, _ = _select_action_dsac(agent, x, encoder, mean, std, policy, force_exit=bool(base_reason))
                action_name = _guard_action(
                    action_name,
                    force_exit=bool(base_reason),
                    hold=hold,
                    unreal=unreal,
                    mfe=mfe,
                    mae=mae,
                    effective_sl=effective_sl,
                    reduce_steps=reduce_steps,
                    last_reduce_hold=last_reduce_hold,
                )
                if base_reason and action_name == "hold":
                    action_name = "close_100"
                if effective_sl > 0.0 and unreal <= -abs(effective_sl):
                    action_name = "close_100"
                if max_hold > 0 and hold >= max_hold and action_name == "hold":
                    action_name = "close_100"
                rl_action_counts[action_name] = rl_action_counts.get(action_name, 0) + 1
                if action_name != "hold":
                    frac = 1.0
                    if action_name == "close_50":
                        frac = 0.5
                    filled, cash, notional, _, route = _apply_exit_fill(
                        df,
                        i,
                        pos=pos,
                        entry_price=entry_price,
                        cash=cash,
                        notional=notional,
                        fraction=frac,
                        action_name=action_name,
                        arm_by_name=arm_by_name,
                        fee_base=fee_base,
                        slip_base=slip_base,
                    )
                    route_counts[route] = route_counts.get(route, 0) + 1
                    if not filled:
                        runner_actions["exit_action_miss_hold"] = runner_actions.get("exit_action_miss_hold", 0) + 1
                    elif notional <= 1e-9:
                        trades += 1
                        wins += int(cash > entry_equity)
                        exits[reason or f"{owner}_{action_name}"] = exits.get(reason or f"{owner}_{action_name}", 0) + 1
                        pos = 0
                        owner = ""
                        cooldown = int(next_cooldown)
                        next_cooldown = 0
                        deep_cooldown = max(deep_cooldown, int(stack["overlay"].cooldown))
                        add_done = False
                        reduce_steps = 0
                        last_reduce_hold = -9999
                        continue
                    else:
                        reduce_steps += 1
                        last_reduce_hold = hold
                        if action_name == "close_50":
                            partial_exits += 1
            if owner == "v21_2" and not reason and not add_done and stack["add_cfg"].full_add_frac > 0.0 and unreal >= stack["add_cfg"].min_unrealized and hold >= stack["add_cfg"].min_bars_since_entry and dd_abs <= stack["add_cfg"].dd_block:
                add_done = True
            if pos != 0:
                continue
        if cooldown > 0:
            cooldown -= 1
            continue
        if deep_cooldown > 0:
            deep_cooldown -= 1
        dec = decisions.iloc[i]
        if int(dec.action) != base_exit.ACTION_CASH and int(dec.side) != 0:
            filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, int(dec.side), entry_cfg, entry=True, fee=fee_base, slip=slip_base)
            route_counts[route] = route_counts.get(route, 0) + 1
            if not filled:
                runner_actions["parent_entry_limit_miss"] = runner_actions.get("parent_entry_limit_miss", 0) + 1
                continue
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = px
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), stack["add_cfg"].max_entry_notional)
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
            reduce_steps = 0
            last_reduce_hold = -9999
            runner_actions["v21_entry"] = runner_actions.get("v21_entry", 0) + 1
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= stack["overlay"].edge_th and margin >= stack["overlay"].margin_th:
                filled, px, entry_fee, _, route = alpha3._try_immediate_limit(df, i, side, entry_cfg, entry=True, fee=fee_base, slip=slip_base)
                route_counts[route] = route_counts.get(route, 0) + 1
                if not filled:
                    runner_actions["deep_entry_limit_miss"] = runner_actions.get("deep_entry_limit_miss", 0) + 1
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = px
                entry_equity = cash
                entry_idx = i
                parent_notional = notional = float(stack["overlay"].notional)
                take_profit = float(stack["overlay"].base_tp)
                stop_loss = float(stack["overlay"].base_sl)
                max_hold = int(stack["overlay"].base_hold)
                next_cooldown = int(stack["overlay"].cooldown)
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
                reduce_steps = 0
                last_reduce_hold = -9999
                runner_actions["deep_entry"] = runner_actions.get("deep_entry", 0) + 1
    if pos != 0:
        exit_px = base_exit._fill_price(df, len(df) - 1, pos, slip_base, entry=False)
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
        "trades_per_day": float(trades / base_exit._days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": runner_actions,
        "route_counts": route_counts,
        "rl_action_counts": rl_action_counts,
        "partial_exits": int(partial_exits),
    }


def _metrics_dsac_v2(df, stack, deep_q, decisions, entry_cfg, arms, base_cols, agent, encoder, mean, std, policy):
    return {
        f"cost{mult}": backtest_dsac_v2(
            df,
            stack,
            deep_q,
            decisions,
            entry_cfg,
            arms,
            base_cols,
            agent,
            encoder,
            mean,
            std,
            policy,
            fee=stack["fee"],
            slip=stack["slip"],
            cost_mult=float(mult),
        )
        for mult in (1, 2, 3)
    }


def _train_dsac(
    x: np.ndarray,
    y: np.ndarray,
    action_names: list[str],
    *,
    updates: int = 18000,
    batch_size: int = 512,
    seed: int = 20260515,
) -> tuple[DSACAgent, dict[str, Any], np.ndarray, np.ndarray]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = _resolve_device()
    if device == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    mean, std, x_z = _normalise(x)
    agent = DSACAgent(
        state_dim=x.shape[1],
        hidden_dim=256,
        lr_actor=2e-4,
        lr_critic=3e-4,
        lr_alpha=2e-4,
        gamma=0.995,
        tau=0.007,
        n_quantiles=48,
        cvar_frac=0.35,
        adaptive_pessimism=True,
        pessimism_min_weight=0.68,
        pessimism_weight_min=0.60,
        pessimism_weight_max=0.82,
        dynamic_entropy=True,
        entropy_min=-0.95,
        entropy_max=-0.35,
        entropy_std_low=0.12,
        entropy_std_high=0.38,
        entropy_step=0.035,
        critic_var_weight=True,
        critic_var_scale=0.50,
        primacy_soft_reset=True,
        primacy_window=80,
        cql_reg=True,
        cql_alpha=0.045,
        redo_enable=True,
        redo_interval=750,
        redo_tau=5e-3,
        redo_ratio=0.08,
        alpha_min=0.003,
        alpha_init=0.035,
        anti_flat_lambda=0.02,
        anti_flat_min_abs=0.08,
        anti_flat_anneal_updates=max(updates // 2, 1),
        direction_reg_lambda=0.02,
        side_balance_lambda=0.02,
        device=device,
    )
    replay_meta = _populate_dsac_replay(agent, x_z, y, action_names)
    history: list[dict[str, float]] = []
    acc: dict[str, float] = {}
    count = 0
    for step in range(1, int(updates) + 1):
        out = agent.update(batch_size=batch_size)
        if not out:
            continue
        count += 1
        for k, v in out.items():
            acc[k] = acc.get(k, 0.0) + float(v)
        if step == 1 or step % 1000 == 0 or step == updates:
            row = {"step": float(step)}
            div = max(count, 1)
            for k, v in acc.items():
                row[k] = float(v / div)
            history.append(row)
            print(
                f"[{MODEL_ID}] update {step}/{updates} "
                f"critic={row.get('critic_loss', 0.0):.4f} actor={row.get('actor_loss', 0.0):.4f} "
                f"alpha={row.get('alpha', 0.0):.4f} cvar={row.get('cvar_q', 0.0):.4f}",
                flush=True,
            )
            acc = {}
            count = 0
    meta = {
        "device": device,
        "updates": int(updates),
        "batch_size": int(batch_size),
        "replay": replay_meta,
        "history": history,
    }
    return agent, meta, mean, std


def main() -> int:
    print(f"[{MODEL_ID}] loading fixed Alpha3 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    stack = front_run._load_fixed_stack()
    corrected_cfg = alpha3.ImmediateLimitConfig(
        "next_open_limit_touch0_fee20",
        "next_open",
        0.0,
        0.0,
        0.0,
        0.20,
        entry_miss="skip",
        exit_miss="market_fallback",
    )
    arms = [corrected_cfg]
    arm_by_name = {a.name: a for a in arms}
    entry_cfg = corrected_cfg
    base_action_names = base_exit._action_names(arms)
    action_names = list(EXPANDED_ACTIONS)
    feature_cols = list(stack["teacher_payload"]["feature_cols"])
    feature_names = deep_exit._feature_names(feature_cols)

    train_all = _read(v31.DEFAULT_TRAIN)
    train_df = train_all[(train_all["timestamp"] >= TRAIN_START) & (train_all["timestamp"] < TRAIN_END)].reset_index(drop=True)
    val_df = train_all[train_all["timestamp"] >= VAL_START].reset_index(drop=True)
    eval_df = _read(v31.DEFAULT_EVAL)

    print(f"[{MODEL_ID}] rebuilding Alpha3 decisions and frozen V27 q", flush=True)
    train_dec, train_q = front_run._decisions_and_q(train_df, stack)
    val_dec, val_q = front_run._decisions_and_q(val_df, stack)
    eval_dec, eval_q = front_run._decisions_and_q(eval_df, stack)

    print(f"[{MODEL_ID}] collecting DP-labeled exit-owner replay", flush=True)
    x_full, base_y, dataset_meta = base_exit.collect_q_dataset(
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
    y = _expanded_targets(base_y, base_action_names, x_full, feature_names)
    base_label_counts = np.bincount(np.argmax(base_y, axis=1), minlength=len(base_action_names)).astype(int).tolist()
    label_counts = np.bincount(np.argmax(y, axis=1), minlength=len(action_names)).astype(int).tolist()
    dataset_summary = {
        **dataset_meta,
        "train_start": str(train_df["timestamp"].iloc[0]) if len(train_df) else None,
        "train_end": str(train_df["timestamp"].iloc[-1]) if len(train_df) else None,
        "base_actions": base_action_names,
        "base_target_argmax_counts": dict(zip(base_action_names, base_label_counts)),
        "base_target_mean_by_action": dict(zip(base_action_names, np.mean(base_y, axis=0).astype(float).tolist())),
        "actions": action_names,
        "target_argmax_counts": dict(zip(action_names, label_counts)),
        "target_mean_by_action": dict(zip(action_names, np.mean(y, axis=0).astype(float).tolist())),
        "full_state_dim": int(x_full.shape[1]),
        "feature_modes": FEATURE_MODES,
        "action_scalars": ACTION_SCALARS,
        "reward_design": {
            "hold": "Continuation value with mild exit-advantage/giveback/age penalty.",
            "close_50": "Masked unless profitable and mature; 50% corrected close value plus 50% hold continuation with MAE/action penalty.",
            "close_100": "Corrected Alpha3 close value, masked unless near-SL, profit-lock, or clear exit advantage/giveback context.",
        },
    }
    DATASET_OUT.write_text(json.dumps(dataset_summary, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")

    print(f"[{MODEL_ID}] selecting controls on 2025Q4", flush=True)
    val_baseline = alpha3._metrics_signal_limit(
        val_df,
        stack["parent"],
        stack["jackpot_model"],
        stack["add_cfg"],
        val_q,
        val_dec,
        stack["overlay"],
        entry_cfg,
        fee=stack["fee"],
        slip=stack["slip"],
    )

    rows: list[dict[str, Any]] = [
        {
            "feature_mode": "control",
            "name": "alpha3_corrected_next_open_limit_touch0_fee20",
            "selection_score": _score(val_baseline),
            "val_cost1_pnl": val_baseline["cost1"]["pnl"],
            "val_cost1_mdd": val_baseline["cost1"]["mdd"],
            "val_cost1_trades": val_baseline["cost1"]["trades"],
            "val_cost2_pnl": val_baseline["cost2"]["pnl"],
            "val_cost3_pnl": val_baseline["cost3"]["pnl"],
        },
    ]
    best_dsac: tuple[float, str, base_exit.OfflineRLPolicy, dict[str, Any], DSACAgent, FeatureEncoder, np.ndarray, np.ndarray] | None = None
    mode_results: list[dict[str, Any]] = []

    for mode_i, mode in enumerate(FEATURE_MODES):
        print(f"[{MODEL_ID}] training feature_mode={mode}", flush=True)
        encoder = _build_feature_encoder(mode, x_full, y, feature_names)
        x_mode = encoder.transform(x_full)
        agent, train_meta, mean, std = _train_dsac(
            x_mode,
            y,
            action_names,
            updates=TRAIN_UPDATES,
            seed=20260515 + mode_i,
        )
        model_out = OUT_DIR / f"dsac_exit_owner_v6_{mode}.pt"
        encoder_out = OUT_DIR / f"feature_encoder_{mode}.joblib"
        joblib.dump(_encoder_payload(encoder), encoder_out)
        torch.save(
            {
                "model_id": MODEL_ID,
                "feature_mode": mode,
                "actor_state": agent.actor.state_dict(),
                "critic_state": agent.critic.state_dict(),
                "critic_target_state": agent.critic_target.state_dict(),
                "state_dim": int(x_mode.shape[1]),
                "full_feature_names": feature_names,
                "feature_encoder": _feature_meta(encoder),
                "feature_mean": mean,
                "feature_std": std,
                "actions": action_names,
                "action_scalars": ACTION_SCALARS,
                "train_meta": train_meta,
                "dataset": dataset_summary,
            },
            model_out,
        )
        best_mode: tuple[float, base_exit.OfflineRLPolicy, dict[str, Any]] | None = None
        for policy in [p for p in _runtime_policies() if p.name.startswith("dsac_")]:
            metrics = _metrics_dsac_v2(val_df, stack, val_q, val_dec, entry_cfg, arms, feature_cols, agent, encoder, mean, std, policy)
            score = _score(metrics)
            rows.append(
                {
                    "feature_mode": mode,
                    **asdict(policy),
                    "selection_score": score,
                    "val_cost1_pnl": metrics["cost1"]["pnl"],
                    "val_cost1_mdd": metrics["cost1"]["mdd"],
                    "val_cost1_trades": metrics["cost1"]["trades"],
                    "val_cost2_pnl": metrics["cost2"]["pnl"],
                    "val_cost3_pnl": metrics["cost3"]["pnl"],
                    "val_cost1_rl_action_counts": json.dumps(metrics["cost1"].get("rl_action_counts", {}), sort_keys=True),
                    "val_cost1_route_counts": json.dumps(metrics["cost1"].get("route_counts", {}), sort_keys=True),
                    "val_cost1_partial_exits": metrics["cost1"].get("partial_exits", 0),
                    "model": str(model_out.relative_to(ROOT)),
                    "encoder": str(encoder_out.relative_to(ROOT)),
                }
            )
            print(
                f"[{MODEL_ID}] {mode}::{policy.name} val c1={metrics['cost1']['pnl']:.2f} "
                f"mdd={metrics['cost1']['mdd']:.2f} c2={metrics['cost2']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f}",
                flush=True,
            )
            if best_mode is None or score > best_mode[0]:
                best_mode = (score, policy, metrics)
            if best_dsac is None or score > best_dsac[0]:
                best_dsac = (score, mode, policy, metrics, agent, encoder, mean, std)
        assert best_mode is not None
        mode_results.append(
            {
                "feature_mode": mode,
                "feature_encoder": _feature_meta(encoder),
                "model": str(model_out.relative_to(ROOT)),
                "encoder": str(encoder_out.relative_to(ROOT)),
                "train_meta": train_meta,
                "best_validation_score": float(best_mode[0]),
                "best_validation_policy": asdict(best_mode[1]),
                "best_validation_metrics": best_mode[2],
            }
        )

    assert best_dsac is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    selected_score, selected_mode, selected_policy, selected_val_metrics, selected_agent, selected_encoder, selected_mean, selected_std = best_dsac
    print(f"[{MODEL_ID}] selected {selected_mode}::{selected_policy.name} score={selected_score:.2f}", flush=True)

    print(f"[{MODEL_ID}] fixed current 2026 replay", flush=True)
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
    dsac_metrics = _metrics_dsac_v2(
        eval_df,
        stack,
        eval_q,
        eval_dec,
        entry_cfg,
        arms,
        feature_cols,
        selected_agent,
        selected_encoder,
        selected_mean,
        selected_std,
        selected_policy,
    )

    experiments = [
        {"name": "alpha3_corrected_next_open_limit_touch0_fee20", "config": asdict(corrected_cfg), "metrics": baseline, "score": _score(baseline)},
        {
            "name": f"alpha3_dsac_exit_owner_v6::{selected_mode}::{selected_policy.name}",
            "feature_mode": selected_mode,
            "policy": asdict(selected_policy),
            "metrics": dsac_metrics,
            "score": _score(dsac_metrics),
        },
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
        "date": "2026-05-15",
        "design": {
            "algorithm": "DSAC actor + distributional twin quantile critic + CVaR policy objective + CQL regularization",
            "source": "Adapted from ensemble/train_rl_dsac_agent.py DSACAgent.",
            "scope": "Alpha3 corrected entry stack frozen. DSAC exit owner chooses only hold, close_50, or close_100; execution price/route remains the corrected Alpha3 post-only limit + close fallback contract.",
            "feature_design": "Feature mode is fixed to state16_topbase20 from Alpha3 exit-owner feature analysis 20260515.",
            "limitations": "No L2/orderbook/queue inputs or labels by request. Maker fills use existing OHLC immediate-limit proxy. close_50 labels are synthetic approximations from close/hold DP targets and are event-gated.",
            "train_split": "2025-01-01..2025-09-30",
            "selection_split": "2025-10-01..2025-12-31",
            "selection_uses_2026": False,
            "train_updates_per_feature_mode": TRAIN_UPDATES,
        },
        "dataset": dataset_summary,
        "feature_analysis_source": str(FEATURE_ANALYSIS_REPORT.relative_to(ROOT)),
        "mode_results": mode_results,
        "validation_controls": {
            "baseline": val_baseline,
        },
        "selected_dsac_policy": asdict(selected_policy),
        "selected_feature_mode": selected_mode,
        "selected_feature_encoder": _feature_meta(selected_encoder),
        "selected_validation_metrics": selected_val_metrics,
        "validation_best_dsac_score": float(selected_score),
        "experiments": experiments,
        "artifacts": {
            "out_dir": str(OUT_DIR.relative_to(ROOT)),
            "summary": str(REPORT_OUT.relative_to(ROOT)),
            "grid": str(GRID_OUT.relative_to(ROOT)),
            "audit": str(AUDIT_OUT.relative_to(ROOT)),
            "dataset": str(DATASET_OUT.relative_to(ROOT)),
        },
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    audit = {
        "model_id": MODEL_ID,
        "selection_uses_2026": False,
        "status": "research_candidate",
        "base_contract": "docs/model_contracts/alpha3_teacher_l2_limit_fallback_20260514_contract.md",
        "notes": [
            "Uses DSACAgent architecture and training techniques from ensemble/train_rl_dsac_agent.py.",
            "State/reward replay is Alpha3 exit-owner specific, not the original DSAC trading env.",
            "L2/orderbook/queue data is intentionally removed.",
            "Only hold, close_50, and close_100 actions are included. Execution placement is not learned by DSAC.",
            "Feature sweep is based on alpha3_exit_owner_feature_analysis_20260515: market features are reduced by top-k or target-aware PLS.",
        ],
        "promotion_gate": [
            "Must beat Alpha3 corrected next_open_limit_touch0_fee20 on the same eval horizon.",
            "Must survive cost2/cost3 stress and later real L2 shadow route/fallback audit if orderbook data is reintroduced.",
        ],
    }
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[{MODEL_ID}] wrote {REPORT_OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
