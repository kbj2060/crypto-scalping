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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts import train_eval_hf_v13_deep_jackpot_sequence_verifier_v23 as v23
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_offline_addon_lifecycle_rl_v25_20260511"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_offline_addon_lifecycle_rl_v25_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_offline_addon_lifecycle_rl_v25_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_offline_addon_lifecycle_rl_v25_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_offline_addon_lifecycle_rl_v25_20260511_grid.csv"
SEQ_LEN = 72
FORBIDDEN = v23.FORBIDDEN


class SleeveQ(nn.Module):
    def __init__(self, seq_dim: int, ctx_dim: int, hidden: int = 64) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, 3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Conv1d(hidden, hidden, 3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, 64), nn.GELU())
        self.head = nn.Sequential(nn.Linear(hidden + 64, 64), nn.GELU(), nn.Linear(64, 3))

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        h = self.seq(seq.transpose(1, 2)).squeeze(-1)
        return self.head(torch.cat([h, self.ctx(ctx)], dim=1))


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


def _grid() -> list[v23.VerifierConfig]:
    return [
        v23.VerifierConfig("v25_hold_bias_000", 0.000, 0.000, 0.50, -0.006),
        v23.VerifierConfig("v25_hold_bias_002", 0.002, 0.000, 0.50, -0.006),
        v23.VerifierConfig("v25_trim_margin_002", 0.000, 0.002, 0.50, -0.006),
        v23.VerifierConfig("v25_close_margin_004", 0.000, 0.004, 0.50, -0.006),
    ]


def _select_seq_cols(df: pd.DataFrame) -> list[str]:
    return v23._select_seq_cols(df)


def _raw(side: int, entry: float, px: float) -> float:
    return (px - entry) / max(entry, 1e-12) if side > 0 else (entry - px) / max(entry, 1e-12)


def _sleeve_value(frame: pd.DataFrame, *, side: int, addon_entry: float, addon_notional: float, snapshot_idx: int, parent_exit_i: int, fee_eff: float, slip_eff: float, action: int) -> float:
    close_i = min(snapshot_idx + 1, len(frame) - 1)
    final_i = min(parent_exit_i, len(frame) - 1)
    close_px = _fill_price(frame, close_i, side, slip_eff, entry=False)
    final_px = _fill_price(frame, final_i, side, slip_eff, entry=False)
    close_u = _raw(side, addon_entry, close_px) * addon_notional - fee_eff * addon_notional
    final_u = _raw(side, addon_entry, final_px) * addon_notional - fee_eff * addon_notional
    if action == 0:
        return float(final_u)
    if action == 1:
        half = addon_notional * 0.5
        return float(_raw(side, addon_entry, close_px) * half - fee_eff * half + _raw(side, addon_entry, final_px) * half - fee_eff * half)
    return float(close_u)


def _parent_exit_index(frame: pd.DataFrame, close: np.ndarray, *, side: int, entry_idx: int, entry_price: float, total_notional: float, take_profit: float, stop_loss: float, max_hold: int, slip_eff: float) -> int:
    exit_i = min(entry_idx + max(max_hold, 1) + 1, len(frame) - 1)
    for j in range(entry_idx, min(entry_idx + max_hold + 1, len(frame) - 1)):
        px = float(close[j])
        mark = px * (1.0 - slip_eff) if side > 0 else px * (1.0 + slip_eff)
        unreal = _raw(side, entry_price, mark) * total_notional
        if (take_profit > 0.0 and unreal >= take_profit) or (stop_loss > 0.0 and unreal <= -abs(stop_loss)) or (max_hold > 0 and j - entry_idx >= max_hold):
            exit_i = min(j + 1, len(frame) - 1)
            break
    return int(exit_i)


def _collect_dataset(frame: pd.DataFrame, bundle: dict[str, Any], jackpot_model: dict[str, Any], add_cfg: CostRunnerConfig, seq_cols: list[str], *, fee: float, slip: float) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    seqs: list[np.ndarray] = []
    ctxs: list[list[float]] = []
    rewards: list[list[float]] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = main_notional = addon_notional = 0.0
    addon_entry = 0.0
    addon_idx = -1
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = addon_mfe = addon_mae = 0.0
    add_done = False
    parent_exit_i = 0
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            main_unreal = _raw(pos, entry_price, px * (1.0 - slip if pos > 0 else 1.0 + slip)) * main_notional
            addon_unreal = _raw(pos, addon_entry, px * (1.0 - slip if pos > 0 else 1.0 + slip)) * addon_notional if addon_notional > 0 else 0.0
            unreal = main_unreal + addon_unreal
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if addon_notional > 0:
                addon_mfe = max(addon_mfe, addon_unreal)
                addon_mae = min(addon_mae, addon_unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            hold = i - entry_idx
            state = {"parent_notional": parent_notional, "notional": main_notional + addon_notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)), "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and state["drawdown_abs"] <= add_cfg.dd_block:
                x = _feature_frame(frame, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                is_jackpot = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                if is_jackpot:
                    fill_i = min(i + 1, len(frame) - 1)
                    addon_notional = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - main_notional))
                    addon_entry = _fill_price(frame, fill_i, pos, slip, entry=True)
                    addon_idx = i
                    parent_exit_i = _parent_exit_index(frame, close, side=pos, entry_idx=i, entry_price=(entry_price * main_notional + addon_entry * addon_notional) / max(main_notional + addon_notional, 1e-12), total_notional=main_notional + addon_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold - hold if max_hold > hold else 1, slip_eff=slip)
                    addon_mfe = addon_mae = 0.0
                add_done = True
            if addon_notional > 0 and addon_idx >= 0 and (i - addon_idx) >= 3 and (i - addon_idx) % 3 == 0 and i < parent_exit_i - 1:
                px_adj = px * (1.0 - slip if pos > 0 else 1.0 + slip)
                addon_unreal = _raw(pos, addon_entry, px_adj) * addon_notional
                ctx = [
                    float(pos),
                    float(parent_notional),
                    float(main_notional),
                    float(addon_notional),
                    float(i - entry_idx),
                    float(i - addon_idx),
                    float(unreal),
                    float(addon_unreal),
                    float(mfe - unreal),
                    float(addon_mfe - addon_unreal),
                    float(addon_unreal - addon_mae),
                    float(state["drawdown_abs"]),
                    float(take_profit),
                    float(stop_loss),
                    float(max_hold),
                ]
                seqs.append(v23._seq_at(frame, i, seq_cols))
                ctxs.append(ctx)
                rewards.append([
                    _sleeve_value(frame, side=pos, addon_entry=addon_entry, addon_notional=addon_notional, snapshot_idx=i, parent_exit_i=parent_exit_i, fee_eff=fee, slip_eff=slip, action=0),
                    _sleeve_value(frame, side=pos, addon_entry=addon_entry, addon_notional=addon_notional, snapshot_idx=i, parent_exit_i=parent_exit_i, fee_eff=fee, slip_eff=slip, action=1),
                    _sleeve_value(frame, side=pos, addon_entry=addon_entry, addon_notional=addon_notional, snapshot_idx=i, parent_exit_i=parent_exit_i, fee_eff=fee, slip_eff=slip, action=2),
                ])
            if reason:
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_main = _raw(pos, entry_price, exit_px) * main_notional
                raw_add = _raw(pos, addon_entry, exit_px) * addon_notional if addon_notional > 0 else 0.0
                before = cash
                cash = cash * (1.0 + raw_main + raw_add)
                cash -= before * fee * (main_notional + addon_notional)
                pos = 0
                addon_notional = 0.0
                add_done = False
                continue
        if pos == 0:
            dec = decisions.iloc[i]
            if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
                continue
            fill_i = min(i + 1, len(frame) - 1)
            pos = int(dec.side)
            entry_price = _fill_price(frame, fill_i, pos, slip, entry=True)
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            main_notional = parent_notional
            addon_notional = 0.0
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * main_notional
            mfe = mae = 0.0
            add_done = False
    if not rewards:
        raise RuntimeError("no add-on lifecycle snapshots")
    return {"seq": np.stack(seqs).astype(np.float32), "ctx": np.asarray(ctxs, dtype=np.float32), "reward": np.asarray(rewards, dtype=np.float32)}


def _norm(seq: np.ndarray, ctx: np.ndarray) -> dict[str, np.ndarray]:
    return {"seq_mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32), "seq_std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32), "ctx_mean": np.nanmean(ctx, axis=0).astype(np.float32), "ctx_std": (np.nanstd(ctx, axis=0) + 1e-6).astype(np.float32)}


def _apply(seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return ((seq - norm["seq_mean"][None, None, :]) / norm["seq_std"][None, None, :]).astype(np.float32), ((ctx - norm["ctx_mean"][None, :]) / norm["ctx_std"][None, :]).astype(np.float32)


def _train(ds: dict[str, Any], norm: dict[str, np.ndarray], *, epochs: int, cql_alpha: float) -> SleeveQ:
    seq, ctx = _apply(ds["seq"], ds["ctx"], norm)
    reward = ds["reward"].astype(np.float32)
    model = SleeveQ(seq.shape[-1], ctx.shape[-1]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    device = next(model.parameters()).device
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(ctx), torch.from_numpy(reward)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    for _ in range(epochs):
        for xb, cb, rb in loader:
            xb, cb, rb = xb.to(device), cb.to(device), rb.to(device)
            q = model(xb, cb)
            loss = loss_fn(q, rb) + cql_alpha * 0.01 * (torch.logsumexp(q, dim=1).mean() - q[:, 0].mean())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict(model: SleeveQ, seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    s, c = _apply(seq[None].astype(np.float32), ctx[None].astype(np.float32), norm)
    with torch.no_grad():
        return model(torch.from_numpy(s), torch.from_numpy(c))[0].numpy()


def backtest(df: pd.DataFrame, bundle: dict[str, Any], jackpot_model: dict[str, Any], model: SleeveQ, norm: dict[str, np.ndarray], add_cfg: CostRunnerConfig, cfg: v23.VerifierConfig, seq_cols: list[str], *, fee: float, slip: float, cost_mult: float = 1.0, decisions: pd.DataFrame | None = None, record: bool = False) -> dict[str, Any]:
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
    cash = peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = main_notional = addon_notional = 0.0
    addon_entry = 0.0
    addon_idx = -1
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = 0
    add_done = False
    mfe = mae = addon_mfe = addon_mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None
    for i in range(0, len(df) - 2):
        if pos != 0:
            px = float(close[i])
            px_mark = px * (1.0 - slip_eff) if pos > 0 else px * (1.0 + slip_eff)
            main_unreal = _raw(pos, entry_price, px_mark) * main_notional
            addon_unreal = _raw(pos, addon_entry, px_mark) * addon_notional if addon_notional > 0 else 0.0
            unreal = main_unreal + addon_unreal
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            if addon_notional > 0:
                addon_mfe = max(addon_mfe, addon_unreal)
                addon_mae = min(addon_mae, addon_unreal)
            hold = i - entry_idx
            state = {"parent_notional": parent_notional, "notional": main_notional + addon_notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)), "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and state["drawdown_abs"] <= add_cfg.dd_block:
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                is_jackpot = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                if is_jackpot:
                    fill_i = min(i + 1, len(df) - 1)
                    addon_notional = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - main_notional))
                    addon_entry = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    addon_idx = i
                    addon_mfe = addon_mae = 0.0
                    before = cash
                    cash -= before * fee_eff * addon_notional
                    actions["add_on"] = actions.get("add_on", 0) + 1
                    if record and open_record is not None:
                        open_record.update({"add_on_timestamp": str(df["timestamp"].iloc[fill_i]), "add_on_delta_notional": float(addon_notional), "add_on_price": float(addon_entry), "add_fee_pct": float(fee_eff * addon_notional * 100.0)})
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if not reason and addon_notional > 0 and addon_idx >= 0 and (i - addon_idx) >= 3 and (i - addon_idx) % 3 == 0:
                ctx = np.asarray([float(pos), float(parent_notional), float(main_notional), float(addon_notional), float(hold), float(i - addon_idx), float(unreal), float(addon_unreal), float(mfe - unreal), float(addon_mfe - addon_unreal), float(addon_unreal - addon_mae), float(state["drawdown_abs"]), float(take_profit), float(stop_loss), float(max_hold)], dtype=np.float32)
                q = _predict(model, v23._seq_at(df, i, seq_cols), ctx, norm)
                q = q.copy()
                q[0] += cfg.fragile_th
                q[1] -= cfg.edge_th
                act = int(np.argmax(q))
                if act == 1 and addon_notional > 0:
                    fill_i = min(i + 1, len(df) - 1)
                    trim_notional = addon_notional * cfg.reduce_frac
                    exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                    before = cash
                    cash = cash * (1.0 + _raw(pos, addon_entry, exit_px) * trim_notional)
                    cash -= before * fee_eff * trim_notional
                    addon_notional -= trim_notional
                    actions["rl_trim"] = actions.get("rl_trim", 0) + 1
                elif act == 2 and addon_notional > 0:
                    fill_i = min(i + 1, len(df) - 1)
                    exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                    before = cash
                    cash = cash * (1.0 + _raw(pos, addon_entry, exit_px) * addon_notional)
                    cash -= before * fee_eff * addon_notional
                    addon_notional = 0.0
                    actions["rl_close"] = actions.get("rl_close", 0) + 1
                else:
                    actions["rl_hold"] = actions.get("rl_hold", 0) + 1
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                before = cash
                cash = cash * (1.0 + _raw(pos, entry_price, exit_px) * main_notional + (_raw(pos, addon_entry, exit_px) * addon_notional if addon_notional > 0 else 0.0))
                cash -= before * fee_eff * (main_notional + addon_notional)
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(main_notional + addon_notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * (main_notional + addon_notional) * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                cooldown = int(next_cooldown)
                next_cooldown = 0
                add_done = False
                open_record = None
                continue
        if pos != 0:
            continue
        if cooldown > 0:
            cooldown -= 1
            continue
        dec = decisions.iloc[i]
        if int(dec.action) == ACTION_CASH or int(dec.side) == 0:
            continue
        fill_i = min(i + 1, len(df) - 1)
        pos = int(dec.side)
        entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
        entry_equity = cash
        entry_idx = i
        parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
        main_notional = parent_notional
        addon_notional = 0.0
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        leverage = float(dec.leverage)
        cash -= cash * fee_eff * main_notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += main_notional
        leverage_sum += leverage
        mfe = mae = 0.0
        add_done = False
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "parent_notional_exposure": float(dec.notional_exposure), "notional_exposure": float(main_notional), "leverage": float(leverage), "position_fraction": float(main_notional / max(leverage, 1e-12)), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * main_notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        before = cash
        cash = cash * (1.0 + _raw(pos, entry_price, exit_px) * main_notional + (_raw(pos, addon_entry, exit_px) * addon_notional if addon_notional > 0 else 0.0))
        cash -= before * fee_eff * (main_notional + addon_notional)
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.45 * float(c2["pnl"]) + 0.25 * float(c3["pnl"]) - 0.25 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Offline conservative RL for V21.2 add-on sleeve lifecycle.")
    p.add_argument("--parent-model", type=Path, default=v23.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v23.DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=v23.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v23.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=240)
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
    seq_cols = _select_seq_cols(train_all)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    forbidden_cols = [c for c in seq_cols if any(tok in c.lower() for tok in FORBIDDEN)]
    train_ds = _collect_dataset(train, bundle, jackpot_model, add_cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]))
    norm = _norm(train_ds["seq"], train_ds["ctx"])
    model = _train(train_ds, norm, epochs=args.epochs, cql_alpha=0.7)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, jackpot_model, model, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = v23.VerifierConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, model, norm, add_cfg, selected, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v25_offline_addon_lifecycle_rl.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": model.state_dict(), "seq_cols": seq_cols, "ctx_cols": ["side", "parent_notional", "main_notional", "addon_notional", "bars_since_entry", "bars_since_addon", "unrealized", "addon_unrealized", "giveback", "addon_giveback", "addon_recovery", "drawdown_abs", "take_profit", "stop_loss", "max_hold"], "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg)}, model_path)
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"], "val_actions": json.dumps(r["validation_cost1"].get("runner_actions", {}), ensure_ascii=False)} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= v23.V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > v23.V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "offline_cql_addon_sleeve_lifecycle", "allowed_actions": ["hold_addon", "trim_addon_50pct", "close_addon_only"], "parent_entry_exit_preserved": True, "forbidden_sequence_columns": forbidden_cols, "train_snapshot_count": int(len(train_ds["reward"])), "reward_mean_by_action": np.mean(train_ds["reward"], axis=0), "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "Offline conservative RL controls only the already-opened V21.2 jackpot add-on sleeve. It can hold, trim 50%, or close add-on only; it cannot change parent entry/exit, side, leverage, or open new add-ons.", "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "model": str(model_path), "split_policy": "Train on 2025 Jan-Sep; select lifecycle policy on 2025 Oct-Dec; evaluate fixed 2026 OOS after selection only.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"model": str(model_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
