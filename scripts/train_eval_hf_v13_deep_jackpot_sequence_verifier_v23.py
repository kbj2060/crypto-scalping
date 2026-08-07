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
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, ACTION_LONG, predict_policy_frame  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _addon_utility, _predict_cost_runner  # noqa: E402


MODEL_ID = "hf_v13_deep_jackpot_sequence_verifier_v23_20260511"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_deep_jackpot_sequence_verifier_v23_20260511"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_deep_jackpot_sequence_verifier_v23_20260511_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_deep_jackpot_sequence_verifier_v23_20260511_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_deep_jackpot_sequence_verifier_v23_20260511_grid.csv"
SEQ_LEN = 72
V21_2_COST1 = 199.5442148936891
V21_2_COST2 = 113.24305052028865
V21_2_COST3 = 24.714228358176072
FORBIDDEN = ("regime_v2", "hdb", "hmm")


SEQ_CANDIDATES = [
    "log_return",
    "mtf_trend_1h",
    "mtf_trend_4h",
    "bb_width_z",
    "garch_vol_z",
    "amihud_illiquidity_z",
    "net_taker_ratio",
    "taker_acceleration",
    "trade_intensity",
    "oi_change_rate",
    "last_funding_rate",
    "long_squeeze_risk",
    "funding_price_divergence",
    "volatility_z",
    "rsi",
    "macd_hist",
    "realized_vol_ratio",
    "chop_index",
    "ai_dir_edge",
    "ai_dir_entropy",
    "ai_adverse_risk",
    "ai_reward_risk",
    "ai_flow_pressure",
    "ai_flow_exhaustion",
    "dlinear_smf_slope",
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_uncertainty",
    "teacher_tail_warning",
    "clean_regime_2024_unsup_v4_factor_trend",
    "clean_regime_2024_unsup_v4_factor_flow",
    "clean_regime_2024_unsup_v4_factor_vol",
    "clean_regime_2024_unsup_v4_factor_crowding",
    "clean_regime_2024_unsup_v4_factor_liquidity",
    "clean_regime_2024_unsup_v4_transition_risk",
    "clean_regime_2024_unsup_v4_confidence",
    "clean_regime_2024_unsup_v4_entropy",
]


CTX_COLS = [
    "side",
    "parent_notional",
    "current_notional",
    "bars_since_entry",
    "unrealized",
    "mfe",
    "mae",
    "giveback",
    "recovery",
    "drawdown_abs",
    "take_profit",
    "stop_loss",
    "max_hold",
    "v21_edge",
    "v21_prob",
    "v21_q10",
    "v21_q90",
    "v21_p_jackpot",
    "v21_p_bad",
    "v21_p_cost3",
]


@dataclass(frozen=True)
class VerifierConfig:
    name: str
    fragile_th: float
    edge_th: float
    reduce_frac: float
    q10_floor: float


class DeepVerifier(nn.Module):
    def __init__(self, seq_dim: int, ctx_dim: int, hidden: int = 48) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv1d(seq_dim, hidden, kernel_size=3, padding=2, dilation=2),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Conv1d(hidden, hidden, kernel_size=3, padding=4, dilation=4),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ctx = nn.Sequential(nn.Linear(ctx_dim, 48), nn.GELU(), nn.Dropout(0.10))
        self.head = nn.Sequential(nn.Linear(hidden + 48, 64), nn.GELU(), nn.Dropout(0.10), nn.Linear(64, 5))

    def forward(self, seq: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # seq: [B,T,F]
        h = self.seq(seq.transpose(1, 2)).squeeze(-1)
        c = self.ctx(ctx)
        return self.head(torch.cat([h, c], dim=1))


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


def _grid() -> list[VerifierConfig]:
    rows: list[VerifierConfig] = []
    i = 0
    for fragile_th in (0.55, 0.65):
        for edge_th in (0.001, 0.002):
            rows.append(VerifierConfig(f"v23_f{fragile_th:.2f}_e{edge_th:.3f}", fragile_th, edge_th, 0.10, -0.006))
            i += 1
    return rows


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _select_seq_cols(df: pd.DataFrame) -> list[str]:
    cols = [c for c in SEQ_CANDIDATES if c in df.columns]
    bad = [c for c in cols if any(tok in c.lower() for tok in FORBIDDEN)]
    if bad:
        raise RuntimeError(f"forbidden sequence columns selected: {bad}")
    return cols


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str]) -> np.ndarray:
    start = max(0, idx - SEQ_LEN + 1)
    arr = df.iloc[start : idx + 1][cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=np.float32)
    if len(arr) < SEQ_LEN:
        pad = np.zeros((SEQ_LEN - len(arr), len(cols)), dtype=np.float32)
        arr = np.vstack([pad, arr])
    return arr[-SEQ_LEN:]


def _normalizers(seq: np.ndarray, ctx: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "seq_mean": np.nanmean(seq, axis=(0, 1)).astype(np.float32),
        "seq_std": (np.nanstd(seq, axis=(0, 1)) + 1e-6).astype(np.float32),
        "ctx_mean": np.nanmean(ctx, axis=0).astype(np.float32),
        "ctx_std": (np.nanstd(ctx, axis=0) + 1e-6).astype(np.float32),
    }


def _apply_norm(seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    return (
        ((seq - norm["seq_mean"][None, None, :]) / norm["seq_std"][None, None, :]).astype(np.float32),
        ((ctx - norm["ctx_mean"][None, :]) / norm["ctx_std"][None, :]).astype(np.float32),
    )


def _collect_snapshots(
    frame: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    seq_cols: list[str],
    *,
    fee: float,
    slip: float,
) -> dict[str, Any]:
    decisions = predict_policy_frame(bundle, frame, close=_close(frame))
    close = _close(frame)
    seq_rows: list[np.ndarray] = []
    ctx_rows: list[list[float]] = []
    targets: list[list[float]] = []
    meta: list[dict[str, Any]] = []
    pos = 0
    entry_price = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cash = peak = 1.0
    mfe = mae = 0.0
    add_done = False
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(close[i])
            raw = (px * (1.0 - slip) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            hold = i - entry_idx
            state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": max(0.0, 1.0 - eq / max(peak, 1e-12)), "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
            reason = ""
            if take_profit > 0.0 and unreal >= take_profit:
                reason = "tp"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "sl"
            elif max_hold > 0 and hold >= max_hold:
                reason = "hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and state["drawdown_abs"] <= add_cfg.dd_block:
                x = _feature_frame(frame, bundle, decisions, i, state)
                edge, p, q10, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                is_jackpot = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                if is_jackpot:
                    u1 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=add_cfg.full_add_frac, fee=fee, slip=slip, cost_mult=1.0)
                    u2 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=add_cfg.full_add_frac, fee=fee, slip=slip, cost_mult=2.0)
                    u3 = _addon_utility(frame, close, pos=pos, entry_idx=entry_idx, snapshot_idx=i, entry_price=entry_price, current_notional=notional, parent_notional=parent_notional, take_profit=take_profit, stop_loss=stop_loss, max_hold=max_hold, add_frac=add_cfg.full_add_frac, fee=fee, slip=slip, cost_mult=3.0)
                    ctx = [
                        float(pos),
                        float(parent_notional),
                        float(notional),
                        float(hold),
                        float(unreal),
                        float(mfe),
                        float(mae),
                        float(mfe - unreal),
                        float(unreal - mae),
                        float(state["drawdown_abs"]),
                        float(take_profit),
                        float(stop_loss),
                        float(max_hold),
                        float(edge),
                        float(p),
                        float(q10),
                        float(q90),
                        float(p_jackpot),
                        float(p_bad),
                        float(p_cost3),
                    ]
                    seq_rows.append(_seq_at(frame, i, seq_cols))
                    ctx_rows.append(ctx)
                    targets.append([float(u2), float(u3), float(u1 >= 0.015), float(u1 > 0.0 and u3 <= 0.0), float(u1)])
                    meta.append({"idx": int(i), "timestamp": str(frame["timestamp"].iloc[i]), "side": int(pos), "u1": float(u1), "u2": float(u2), "u3": float(u3), "p_jackpot": float(p_jackpot), "q90": float(q90)})
                add_done = True
            if reason:
                exit_i = min(i + 1, len(frame) - 1)
                exit_px = _fill_price(frame, exit_i, pos, slip, entry=False)
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * fee * notional
                pos = 0
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
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            cash -= cash * fee * notional
            mfe = mae = 0.0
            add_done = False
    if not seq_rows:
        raise RuntimeError("no V21.2 jackpot add-on snapshots")
    return {
        "seq": np.stack(seq_rows).astype(np.float32),
        "ctx": np.asarray(ctx_rows, dtype=np.float32),
        "target": np.asarray(targets, dtype=np.float32),
        "meta": meta,
        "decisions": decisions,
    }


def _train_model(train_ds: dict[str, Any], norm: dict[str, np.ndarray], *, epochs: int) -> DeepVerifier:
    seq, ctx = _apply_norm(train_ds["seq"], train_ds["ctx"], norm)
    y = train_ds["target"].astype(np.float32)
    device = _device()
    model = DeepVerifier(seq.shape[-1], ctx.shape[-1]).to(device)
    loader = DataLoader(TensorDataset(torch.from_numpy(seq), torch.from_numpy(ctx), torch.from_numpy(y)), batch_size=64, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    huber = nn.SmoothL1Loss()
    bce = nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(epochs):
        for xb, cb, yb in loader:
            xb, cb, yb = xb.to(device), cb.to(device), yb.to(device)
            out = model(xb, cb)
            loss = huber(out[:, 0], yb[:, 0]) + 0.70 * huber(out[:, 1], yb[:, 1])
            loss = loss + bce(out[:, 2], yb[:, 2]) + bce(out[:, 3], yb[:, 3]) + 0.25 * huber(out[:, 4], yb[:, 4])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model.cpu().eval()


def _predict_one(model: DeepVerifier, seq: np.ndarray, ctx: np.ndarray, norm: dict[str, np.ndarray]) -> dict[str, float]:
    s, c = _apply_norm(seq[None, ...].astype(np.float32), ctx[None, ...].astype(np.float32), norm)
    with torch.no_grad():
        out = model(torch.from_numpy(s), torch.from_numpy(c))[0]
    return {
        "delta_cost2": float(out[0].item()),
        "delta_cost3": float(out[1].item()),
        "p_jackpot_preserve": float(torch.sigmoid(out[2]).item()),
        "p_cost3_fragile": float(torch.sigmoid(out[3]).item()),
        "delta_cost1": float(out[4].item()),
    }


def _verifier_action(pred: dict[str, float], cfg: VerifierConfig) -> tuple[str, float]:
    if pred["p_cost3_fragile"] >= cfg.fragile_th and pred["delta_cost3"] < 0.0:
        return "reject", 0.0
    if pred["p_cost3_fragile"] >= cfg.fragile_th:
        return "reduce", cfg.reduce_frac
    if pred["delta_cost2"] >= cfg.edge_th and pred["p_jackpot_preserve"] >= 0.20:
        return "full", 0.20
    return "reduce", cfg.reduce_frac


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    verifier: DeepVerifier,
    norm: dict[str, np.ndarray],
    add_cfg: CostRunnerConfig,
    verify_cfg: VerifierConfig,
    seq_cols: list[str],
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
    entry_price = entry_equity = 0.0
    entry_idx = 0
    parent_notional = notional = 0.0
    leverage = 1.0
    take_profit = stop_loss = 0.0
    max_hold = 0
    cooldown = next_cooldown = 0
    add_done = False
    mfe = mae = 0.0
    trades = wins = long_entries = short_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    runner_actions: dict[str, int] = {}
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
                reason = "learned_take_profit"
            elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                reason = "learned_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
                reason = "learned_max_hold"
            if not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                edge, p, q10, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                is_jackpot = p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40
                if is_jackpot:
                    ctx = np.asarray([float(pos), float(parent_notional), float(notional), float(hold), float(unreal), float(mfe), float(mae), float(mfe - unreal), float(unreal - mae), float(dd_abs), float(take_profit), float(stop_loss), float(max_hold), float(edge), float(p), float(q10), float(q90), float(p_jackpot), float(p_bad), float(p_cost3)], dtype=np.float32)
                    pred = _predict_one(verifier, _seq_at(df, i, seq_cols), ctx, norm)
                    action, frac = _verifier_action(pred, verify_cfg)
                    cap = parent_notional * add_cfg.max_total_mult
                    delta = max(0.0, min(parent_notional * frac, cap - notional))
                    runner_actions[f"v23_{action}"] = runner_actions.get(f"v23_{action}", 0) + 1
                    if delta > 1e-12:
                        fill_i = min(i + 1, len(df) - 1)
                        add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * fee_eff * delta
                        notional = new_notional
                        runner_actions["add_on"] = runner_actions.get("add_on", 0) + 1
                        if record and open_record is not None:
                            open_record.update({"add_on_timestamp": str(df["timestamp"].iloc[fill_i]), "add_on_delta_notional": float(delta), "add_on_price": float(add_px), "v23_action": action, **{f"v23_{k}": v for k, v in pred.items()}, "add_fee_pct": float(fee_eff * delta * 100.0)})
                else:
                    runner_actions["v21_reject"] = runner_actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
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
        notional = parent_notional
        leverage = float(dec.leverage)
        take_profit = float(dec.take_profit)
        stop_loss = float(dec.stop_loss)
        max_hold = int(dec.max_hold_bars)
        next_cooldown = int(dec.cooldown_bars)
        cash -= cash * fee_eff * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        notional_sum += notional
        leverage_sum += leverage
        mfe = mae = 0.0
        add_done = False
        if record:
            open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "parent_notional_exposure": float(dec.notional_exposure), "notional_exposure": float(notional), "leverage": float(leverage), "position_fraction": float(notional / max(leverage, 1e-12)), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_price) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": runner_actions}
    if record:
        out["trade_records"] = records
    return out


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.45 * float(c2["pnl"]) + 0.20 * float(c3["pnl"]) - 0.25 * abs(float(c1["mdd"])))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V23 deep sequence verifier for V21.2 jackpot add-ons.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    p.add_argument("--epochs", type=int, default=80)
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
    train_ds = _collect_snapshots(train, bundle, jackpot_model, add_cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]))
    norm = _normalizers(train_ds["seq"], train_ds["ctx"])
    verifier = _train_model(train_ds, norm, epochs=args.epochs)
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=1.0)
        v2 = backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=2.0)
        v3 = backtest(val, bundle, jackpot_model, verifier, norm, add_cfg, cfg, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=val_dec, cost_mult=3.0)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    selected = VerifierConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, verifier, norm, add_cfg, selected, seq_cols, fee=float(base["fee"]), slip=float(base["slip"]), decisions=eval_dec, cost_mult=float(mult), record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.out_dir / "v23_deep_jackpot_sequence_verifier.pt"
    torch.save({"model_id": MODEL_ID, "state_dict": verifier.state_dict(), "seq_cols": seq_cols, "ctx_cols": CTX_COLS, "norm": norm, "selected_config": asdict(selected), "add_config": asdict(add_cfg), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model)}, model_path)
    manifest_path = args.out_dir / "feature_manifest.json"
    manifest_path.write_text(json.dumps({"seq_cols": seq_cols, "ctx_cols": CTX_COLS, "seq_len": SEQ_LEN, "forbidden_cols": forbidden_cols}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.grid_out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"], "val_actions": json.dumps(r["validation_cost1"].get("runner_actions", {}), ensure_ascii=False)} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit["blocking"])
    if forbidden_cols:
        blocking.append(f"forbidden_sequence_columns={forbidden_cols}")
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V21_2_COST1:
        warnings.append("oos_cost1_did_not_beat_v21_2")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V21_2_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "forbidden_sequence_columns": forbidden_cols,
        "train_snapshot_count": int(len(train_ds["target"])),
        "feature_audit": feature_audit,
        "selected_config": asdict(selected),
        "metrics": metrics,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Deep TCN sequence verifier attached only to V21.2 jackpot add-on proposals. Parent entry/exit are preserved; verifier can pass, reduce, or reject add-on notional.",
        "parent_model": str(args.parent_model),
        "jackpot_model": str(args.jackpot_model),
        "model": str(model_path),
        "feature_manifest": str(manifest_path),
        "split_policy": "Verifier trained on 2025 Jan-Sep; policy selected on 2025 Oct-Dec; 2026 fixed OOS after selection only.",
        "selected_config": asdict(selected),
        "selection_result": best,
        "metrics": metrics,
        "audit": audit,
        "artifacts": {"model": str(model_path), "manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers},
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "model": str(model_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
