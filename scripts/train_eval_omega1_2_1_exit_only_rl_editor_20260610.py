#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_exit_only_rl_editor_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
MARGIN_CAP = 0.90
TRUE_LEVERAGE = 2.0

HIGH_THRESHOLDS = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
LOW_THRESHOLDS = {"bull": 0.58, "bear": 0.52, "chop": 0.52}
SCALE_MAP = {"bull": 0.65, "bear": 0.90, "chop": 0.90}
BASE_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90, "chop": 0.90}

ACTION_NAMES = ["hold", "tighten_sl", "reduce50", "full_exit"]
HOLD = 0
TIGHTEN_SL = 1
REDUCE50 = 2
FULL_EXIT = 3

FORBIDDEN_PREFIXES = ("clean_regime4_", "regime4_pred_", "teacher_")
FORBIDDEN_EXACT = {"tp_sl_action_score"}


@dataclass
class Position:
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    entry_signal_i: int = 0
    entry_equity: float = 1.0
    notional: float = 0.0
    margin_notional: float = 0.0
    leverage: float = 1.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    floor_unreal: float = -1.0
    reduced: int = 0
    tightened: int = 0


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _reject_forbidden(cols: list[str], tag: str) -> None:
    bad = [c for c in cols if c in FORBIDDEN_EXACT or any(c.startswith(p) for p in FORBIDDEN_PREFIXES)]
    if bad:
        raise RuntimeError(f"{tag} forbidden feature columns: {bad[:40]}")


def _thresholded_source(src: pd.DataFrame, prefix: str, thresholds: dict[str, float]) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    expert = out[f"{prefix}router_expert"].astype(str).to_numpy()
    thr = np.asarray([thresholds.get(str(x), thresholds["chop"]) for x in expert], dtype=np.float64)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return out


def _to_decisions(src: pd.DataFrame, prefix: str, *, oof: bool, thresholds: dict[str, float]) -> pd.DataFrame:
    dec = omega._to_fixed_decisions(_thresholded_source(src, prefix, thresholds), oof=oof)
    active = omega._active(dec)
    for expert, scale in SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & dec["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(BASE_SCALES[key])
        dec.loc[mask, "notional_exposure"] = pd.to_numeric(dec.loc[mask, "notional_exposure"], errors="raise") * ratio
        dec.loc[mask, "position_fraction"] = pd.to_numeric(dec.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(dec)
    dec.loc[active, "take_profit"] = BASE_TP
    dec.loc[active, "stop_loss"] = BASE_SL
    dec.loc[active, "max_hold_bars"] = 0
    dec.loc[active, "cooldown_bars"] = 0
    return _apply_true_leverage_price_barrier(dec)


def _apply_true_leverage_price_barrier(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active_idx = np.flatnonzero(omega._active(out))
    if len(active_idx) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active_idx, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    margin_notional = np.minimum(base_notional * COMPENSATED_SCALE, MARGIN_CAP)
    ratio = margin_notional / np.maximum(base_notional, 1e-12)
    effective_exposure = margin_notional * TRUE_LEVERAGE
    barrier_scale = ratio * TRUE_LEVERAGE
    out.loc[active_idx, "notional_exposure"] = effective_exposure
    out.loc[active_idx, "position_fraction"] = margin_notional
    out.loc[active_idx, "leverage"] = TRUE_LEVERAGE
    out.loc[active_idx, "take_profit"] = BASE_TP * barrier_scale
    out.loc[active_idx, "stop_loss"] = BASE_SL * barrier_scale
    out.loc[active_idx, "max_hold_bars"] = 0
    out.loc[active_idx, "cooldown_bars"] = 0
    return out


def _build_splits() -> dict[str, dict[str, Any]]:
    frames = exposure.th._prepare_frames(disable_tp_sl=False)
    val_frame, val_src, _val_dec, val_prefix = exposure._build_split(frames, "validation")
    oos_frame, oos_src, _oos_dec, oos_prefix = exposure._build_split(frames, "oos")
    return {
        "validation": {
            "frame": val_frame.reset_index(drop=True),
            "src": val_src.reset_index(drop=True),
            "prefix": val_prefix,
            "oof": True,
        },
        "oos": {
            "frame": oos_frame.reset_index(drop=True),
            "src": oos_src.reset_index(drop=True),
            "prefix": oos_prefix,
            "oof": False,
        },
    }


def _rolling_features(frame: pd.DataFrame) -> pd.DataFrame:
    close = pd.to_numeric(frame["close"], errors="raise")
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    open_ = pd.to_numeric(frame["open"], errors="raise")
    ret = close.pct_change().replace([np.inf, -np.inf], np.nan)
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    out = pd.DataFrame(index=frame.index)
    out["bar_range_pct"] = ((high - low) / close).replace([np.inf, -np.inf], np.nan)
    out["body_pct"] = ((close - open_) / close).replace([np.inf, -np.inf], np.nan)
    out["atr14_pct"] = (atr / close).replace([np.inf, -np.inf], np.nan)
    for lag in (1, 3, 6, 12, 24):
        out[f"ret_{lag}"] = close.pct_change(lag).replace([np.inf, -np.inf], np.nan)
    for win in (6, 12, 24, 48):
        out[f"ret_vol_{win}"] = ret.rolling(win, min_periods=max(3, win // 3)).std()
        out[f"range_mean_{win}"] = out["bar_range_pct"].rolling(win, min_periods=max(3, win // 3)).mean()
    ema9 = close.ewm(span=9, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    out["ema9_21_gap"] = ((ema9 - ema21) / close).replace([np.inf, -np.inf], np.nan)
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    minute = ts.dt.hour * 60 + ts.dt.minute
    out["tod_sin"] = np.sin(2.0 * np.pi * minute / 1440.0)
    out["tod_cos"] = np.cos(2.0 * np.pi * minute / 1440.0)
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _state_base(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = _rolling_features(frame)
    cols = [
        "router_confidence",
        "router_margin",
        "dir_p_cash",
        "dir_p_long",
        "dir_p_short",
        "dir_confidence",
        "dir_side_edge",
        "dir_trade_prob",
        "quality_p_cash",
        "quality_p_long",
        "quality_p_short",
        "quality_for_action",
    ]
    for col in cols:
        out[f"tabm_{col}"] = pd.to_numeric(src[f"{prefix}{col}"], errors="raise").to_numpy(dtype=np.float64)
    expert = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})
    for name in ("bull", "bear", "chop_expert"):
        out[f"tabm_router_{name}"] = expert.eq(name).astype(float).to_numpy()
    for col in ("action", "side", "quality_score", "confidence", "notional_exposure", "position_fraction", "leverage", "take_profit", "stop_loss"):
        out[f"dec_{col}"] = pd.to_numeric(dec[col], errors="raise").to_numpy(dtype=np.float64)
    out["dec_rr"] = out["dec_take_profit"] / np.maximum(np.abs(out["dec_stop_loss"]), 1e-8)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    _reject_forbidden(list(out.columns), "exit_rl_state_base")
    return out


def _unreal(arrays: dict[str, np.ndarray], pos: Position, i: int, slip_eff: float) -> float:
    if pos.side == 0 or pos.notional <= 0.0:
        return 0.0
    px = float(arrays["close"][int(i)])
    if pos.side > 0:
        raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12)
    else:
        raw = (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
    return float(raw * pos.notional)


def _pos_features(base_state: pd.DataFrame, pos: Position, unreal: float, i: int) -> pd.DataFrame:
    row = base_state.iloc[[int(i)]].copy().reset_index(drop=True)
    mfe = max(pos.mfe, unreal)
    mae = min(pos.mae, unreal)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    vals = {
        "pos_side": float(pos.side),
        "pos_notional": float(pos.notional),
        "pos_margin_notional": float(pos.margin_notional),
        "pos_leverage": float(pos.leverage),
        "pos_unrealized": float(unreal),
        "pos_mfe": float(mfe),
        "pos_mae": float(mae),
        "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "pos_hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "pos_dist_tp": float(pos.take_profit - unreal),
        "pos_dist_sl": float(unreal + abs(pos.stop_loss)),
        "pos_tp_progress": float(unreal / max(pos.take_profit, 1e-8)),
        "pos_sl_progress": float(-unreal / max(abs(pos.stop_loss), 1e-8)) if pos.stop_loss > 0 else 0.0,
        "pos_floor_unreal": float(pos.floor_unreal),
        "pos_reduced": float(pos.reduced),
        "pos_tightened": float(pos.tightened),
    }
    for k, v in vals.items():
        row[k] = v
    return row.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _enter(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, fee_eff: float, slip_eff: float) -> tuple[float, Position, bool]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == omega.ACTION_CASH:
        return cash, Position(), False
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), False
    notional = float(row.get("notional_exposure", 0.0) or 0.0)
    if notional <= 0.0:
        return cash, Position(), False
    cash -= cash * float(entry_fee) * notional
    return (
        cash,
        Position(
            side=side,
            entry_price=float(entry_px),
            entry_i=min(int(i) + 1, len(arrays["open"]) - 1),
            entry_signal_i=int(i),
            entry_equity=float(cash),
            notional=notional,
            margin_notional=float(row.get("position_fraction", 0.0) or 0.0),
            leverage=float(row.get("leverage", 1.0) or 1.0),
            take_profit=float(row.get("take_profit", 0.0) or 0.0),
            stop_loss=abs(float(row.get("stop_loss", 0.0) or 0.0)),
            floor_unreal=-abs(float(row.get("stop_loss", 0.0) or 0.0)),
        ),
        True,
    )


def _close_fraction(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, frac: float, fee_eff: float, slip_eff: float) -> tuple[float, Position, float]:
    if pos.side == 0 or pos.notional <= 0.0 or frac <= 0.0:
        return cash, pos, 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    _filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), int(pos.side), entry=False, fee_base=fee_eff, slip_base=slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1e-12)
    reduce_notional = pos.notional * frac
    before = cash
    cash = cash * (1.0 + raw * reduce_notional)
    cash -= before * exit_fee * reduce_notional
    out = Position(**pos.__dict__)
    out.notional = max(0.0, pos.notional - reduce_notional)
    if out.notional <= 1e-9:
        out = Position()
    return cash, out, float(raw * reduce_notional)


def _hit_reason(unreal: float, pos: Position) -> str:
    if pos.take_profit > 0.0 and unreal >= pos.take_profit:
        return "take_profit"
    if pos.floor_unreal > -abs(pos.stop_loss) and unreal <= pos.floor_unreal:
        return "tightened_sl_exit"
    if pos.stop_loss > 0.0 and unreal <= -abs(pos.stop_loss):
        return "stop_loss"
    return ""


def _apply_action(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, action: int, unreal: float, fee_eff: float, slip_eff: float) -> tuple[float, Position, str]:
    out = Position(**pos.__dict__)
    if int(action) == FULL_EXIT:
        cash, out, _ = _close_fraction(cash, arrays, out, i, 1.0, fee_eff, slip_eff)
        return cash, out, "full_exit"
    if int(action) == REDUCE50 and out.reduced == 0 and out.notional > 0.10:
        cash, out, _ = _close_fraction(cash, arrays, out, i, 0.50, fee_eff, slip_eff)
        out.reduced = 1
        return cash, out, "reduce50"
    if int(action) == TIGHTEN_SL and out.tightened == 0:
        if unreal > 0.004:
            out.floor_unreal = max(out.floor_unreal, 0.001)
        else:
            out.floor_unreal = max(out.floor_unreal, -0.45 * abs(out.stop_loss))
        out.tightened = 1
        return cash, out, "tighten_sl"
    return cash, out, "hold"


def _simulate_first_action(
    cash: float,
    arrays: dict[str, np.ndarray],
    pos: Position,
    i: int,
    action: int,
    *,
    fee_eff: float,
    slip_eff: float,
    max_forward_bars: int,
) -> float:
    pos = Position(**pos.__dict__)
    unreal0 = _unreal(arrays, pos, i, slip_eff)
    base_equity = max(cash * (1.0 + unreal0), 1e-12)
    cash, pos, _ = _apply_action(cash, arrays, pos, i, action, unreal0, fee_eff, slip_eff)
    if pos.side == 0:
        return float(cash / base_equity - 1.0)
    min_eq = cash * (1.0 + unreal0)
    peak_mfe = max(pos.mfe, unreal0)
    last = min(len(arrays["close"]) - 2, int(i) + int(max_forward_bars))
    for j in range(int(i), last + 1):
        unreal = _unreal(arrays, pos, j, slip_eff)
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        peak_mfe = max(peak_mfe, unreal)
        min_eq = min(min_eq, cash * (1.0 + unreal))
        reason = _hit_reason(unreal, pos)
        if reason:
            cash, pos, _ = _close_fraction(cash, arrays, pos, j, 1.0, fee_eff, slip_eff)
            break
    if pos.side != 0:
        cash, pos, _ = _close_fraction(cash, arrays, pos, min(last + 1, len(arrays["close"]) - 1), 1.0, fee_eff, slip_eff)
    ret = cash / base_equity - 1.0
    dd = min(0.0, min_eq / base_equity - 1.0)
    giveback_penalty = max(0.0, peak_mfe - unreal0) * 0.03 if int(action) == HOLD and peak_mfe > 0.0 else 0.0
    return float(ret - 0.35 * max(0.0, -dd - 0.025) - giveback_penalty)


def _collect_dataset(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    stride: int,
    max_states: int,
    max_forward_bars: int,
) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    arrays = _arrays(frame)
    active = np.asarray(omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    pos = Position()
    rows: list[pd.DataFrame] = []
    rewards: list[np.ndarray] = []
    entry_count = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = _unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            reason = _hit_reason(unreal, pos)
            if reason:
                cash, pos, _ = _close_fraction(cash, arrays, pos, i, 1.0, fee_eff, slip_eff)
                continue
            hold = int(i) - int(pos.entry_i)
            near = bool(
                (pos.take_profit > 0 and unreal >= 0.40 * pos.take_profit)
                or (pos.stop_loss > 0 and unreal <= -0.45 * pos.stop_loss)
                or (pos.mfe > 0 and (pos.mfe - unreal) / max(abs(pos.mfe), 1e-8) > 0.35)
            )
            if hold >= 1 and (hold % int(stride) == 0 or near):
                rows.append(_pos_features(state, pos, unreal, i))
                rewards.append(
                    np.asarray(
                        [
                            _simulate_first_action(cash, arrays, pos, i, a, fee_eff=fee_eff, slip_eff=slip_eff, max_forward_bars=max_forward_bars)
                            for a in range(len(ACTION_NAMES))
                        ],
                        dtype=np.float32,
                    )
                )
                if len(rows) >= int(max_states):
                    break
            continue
        if bool(active[i]):
            cash, pos, entered = _enter(cash, arrays, dec, i, fee_eff, slip_eff)
            entry_count += int(entered)
    if not rows:
        raise RuntimeError("empty exit-only RL dataset")
    x = pd.concat(rows, ignore_index=True)
    r = np.stack(rewards, axis=0).astype(np.float32)
    best = np.argmax(r, axis=1)
    diag = {
        "states": int(len(x)),
        "entries_seen": int(entry_count),
        "best_action_counts": {ACTION_NAMES[i]: int(np.sum(best == i)) for i in range(len(ACTION_NAMES))},
        "mean_reward_by_action": {ACTION_NAMES[i]: float(np.mean(r[:, i])) for i in range(len(ACTION_NAMES))},
    }
    _reject_forbidden(list(x.columns), "exit_rl_dataset")
    return x, r, diag


class QNet(nn.Module):
    def __init__(self, state_dim: int, hidden: int = 256, actions: int = 4) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fit_norm(x: pd.DataFrame) -> dict[str, Any]:
    arr = x.to_numpy(dtype=np.float32)
    mean = np.nanmean(arr, axis=0).astype(np.float32)
    std = np.nanstd(arr, axis=0).astype(np.float32)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
    mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
    return {"columns": list(x.columns), "mean": mean, "std": std}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in x.columns]
    if missing:
        raise RuntimeError(f"missing normalized columns: {missing[:20]}")
    arr = x.reindex(columns=cols).to_numpy(dtype=np.float32)
    return np.nan_to_num((arr - norm["mean"]) / norm["std"], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _train_q(x: pd.DataFrame, rewards: np.ndarray, *, epochs: int, seed: int, cql_weight: float) -> tuple[QNet, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    norm = _fit_norm(x)
    xt = torch.from_numpy(_apply_norm(x, norm)).to(device)
    rt = torch.from_numpy(rewards.astype(np.float32)).to(device)
    model = QNet(xt.shape[1], hidden=256, actions=rt.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=2e-4)
    batch = min(512, len(xt))
    losses: list[float] = []
    for epoch in range(int(epochs)):
        perm = torch.randperm(len(xt), device=device)
        total = 0.0
        seen = 0
        for start in range(0, len(xt), batch):
            idx = perm[start : start + batch]
            q = model(xt[idx])
            target = rt[idx]
            mse = F.smooth_l1_loss(q, target)
            # Conservative penalty: non-best actions should not be overestimated.
            best = torch.argmax(target, dim=1)
            cql = (torch.logsumexp(q, dim=1) - q.gather(1, best[:, None]).squeeze(1)).mean()
            loss = mse + float(cql_weight) * cql
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
            total += float(loss.detach().cpu()) * len(idx)
            seen += len(idx)
        losses.append(total / max(seen, 1))
    model.eval()
    model.norm = norm  # type: ignore[attr-defined]
    diag = {"device": str(device), "epochs": int(epochs), "final_loss": float(losses[-1]), "loss_head": losses[:3], "loss_tail": losses[-3:]}
    return model, diag


@torch.no_grad()
def _q_action(model: QNet, x: pd.DataFrame, *, min_adv: float, allowed_full_exit: bool) -> int:
    device = next(model.parameters()).device
    arr = torch.from_numpy(_apply_norm(x, model.norm)).to(device)  # type: ignore[attr-defined]
    q = model(arr)[0].detach().cpu().numpy().astype(np.float64)
    if not bool(allowed_full_exit):
        q[FULL_EXIT] = -1e9
    best = int(np.argmax(q))
    if best == HOLD:
        return HOLD
    if float(q[best] - q[HOLD]) < float(min_adv):
        return HOLD
    return best


def _simulate_policy(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    state: pd.DataFrame,
    *,
    model: QNet | None,
    min_adv: float,
    fee: float,
    slip: float,
    cost_mult: float,
    allowed_full_exit: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    arrays = _arrays(frame)
    active = np.asarray(omega._active(dec), dtype=bool)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Position()
    trades: list[float] = []
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    action_counts = {name: 0 for name in ACTION_NAMES}
    long_entries = short_entries = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            unreal = _unreal(arrays, pos, i, slip_eff)
            pos.mfe = max(pos.mfe, unreal)
            pos.mae = min(pos.mae, unreal)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = _hit_reason(unreal, pos)
            if not reason and model is not None:
                action = _q_action(model, _pos_features(state, pos, unreal, i), min_adv=min_adv, allowed_full_exit=allowed_full_exit)
                before_pos = pos
                before_cash = cash
                cash, pos, action_name = _apply_action(cash, arrays, pos, i, action, unreal, fee_eff, slip_eff)
                action_counts[action_name] = action_counts.get(action_name, 0) + 1
                if before_pos.side != 0 and pos.side == 0:
                    reason = "rl_full_exit"
                    net_pct = float((cash / max(before_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                    trades.append(net_pct)
                    reasons[reason] = reasons.get(reason, 0) + 1
                    rows.append(
                        {
                            "trade_id": len(rows) + 1,
                            "side": "LONG" if before_pos.side > 0 else "SHORT",
                            "entry_signal_i": int(before_pos.entry_signal_i),
                            "exit_i": int(i),
                            "entry_time": str(frame["timestamp"].iloc[int(before_pos.entry_signal_i)]),
                            "exit_time": str(frame["timestamp"].iloc[int(i)]),
                            "entry_price": float(before_pos.entry_price),
                            "exit_price": float(arrays["close"][i]),
                            "effective_exposure": float(before_pos.notional),
                            "margin_notional": float(before_pos.margin_notional),
                            "leverage": float(before_pos.leverage),
                            "tp_equity_ret": float(before_pos.take_profit),
                            "sl_equity_ret": float(before_pos.stop_loss),
                            "net_trade_return_pct": net_pct,
                            "mfe_pct": float(before_pos.mfe * 100.0),
                            "mae_pct": float(before_pos.mae * 100.0),
                            "exit_reason": reason,
                            "cash_after": float(cash),
                        }
                    )
                    continue
                if action_name == "reduce50":
                    # Partial close changes cash and remaining notional but does not end the trade.
                    _ = before_cash
            if reason:
                close_pos = pos
                cash, pos, _ = _close_fraction(cash, arrays, close_pos, i, 1.0, fee_eff, slip_eff)
                net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append(
                    {
                        "trade_id": len(rows) + 1,
                        "side": "LONG" if close_pos.side > 0 else "SHORT",
                        "entry_signal_i": int(close_pos.entry_signal_i),
                        "exit_i": int(i),
                        "entry_time": str(frame["timestamp"].iloc[int(close_pos.entry_signal_i)]),
                        "exit_time": str(frame["timestamp"].iloc[int(i)]),
                        "entry_price": float(close_pos.entry_price),
                        "exit_price": float(arrays["close"][i]),
                        "effective_exposure": float(close_pos.notional),
                        "margin_notional": float(close_pos.margin_notional),
                        "leverage": float(close_pos.leverage),
                        "tp_equity_ret": float(close_pos.take_profit),
                        "sl_equity_ret": float(close_pos.stop_loss),
                        "net_trade_return_pct": net_pct,
                        "mfe_pct": float(close_pos.mfe * 100.0),
                        "mae_pct": float(close_pos.mae * 100.0),
                        "exit_reason": reason,
                        "cash_after": float(cash),
                    }
                )
            continue

        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if not bool(active[i]):
            continue
        before_side = int(dec.iloc[int(i)].get("side", 0) or 0)
        cash, pos, entered = _enter(cash, arrays, dec, i, fee_eff, slip_eff)
        if entered:
            long_entries += int(before_side > 0)
            short_entries += int(before_side < 0)

    if pos.side != 0:
        close_pos = pos
        cash, pos, _ = _close_fraction(cash, arrays, close_pos, len(frame) - 1, 1.0, fee_eff, slip_eff)
        net_pct = float((cash / max(close_pos.entry_equity, 1e-12) - 1.0) * 100.0)
        trades.append(net_pct)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    arr = np.asarray(trades, dtype=np.float64)
    metrics = {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(trades)),
        "wr": float(np.mean(arr > 0.0)) if len(arr) else 0.0,
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "exit_reasons": reasons,
        "adapter_actions": action_counts,
    }
    return metrics, pd.DataFrame(rows)


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_reasons": metrics["exit_reasons"],
        f"{prefix}_adapter_actions": metrics.get("adapter_actions", {}),
    }


def _entry_audit(base: pd.DataFrame, candidate: pd.DataFrame) -> dict[str, Any]:
    b = set(pd.to_numeric(base.get("entry_signal_i", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    c = set(pd.to_numeric(candidate.get("entry_signal_i", pd.Series(dtype=int)), errors="coerce").dropna().astype(int).tolist())
    return {
        "base_entries": int(len(b)),
        "candidate_entries": int(len(c)),
        "shared_entries": int(len(b & c)),
        "added_entries_after_earlier_exit": int(len(c - b)),
        "dropped_entries": int(len(b - c)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--stride", type=int, default=6)
    ap.add_argument("--max-states", type=int, default=2400)
    ap.add_argument("--max-forward-bars", type=int, default=288)
    ap.add_argument("--cql-weight", type=float, default=0.03)
    ap.add_argument("--seed", type=int, default=260610)
    ap.add_argument("--generators", default="high,low", help="Comma-separated candidate generators: high,low")
    ap.add_argument("--min-advs", default="0,0.001,0.0025,0.005,0.01")
    ap.add_argument("--full-exit-modes", default="0,1", help="0 defensive only, 1 allow full_exit")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    splits = _build_splits()
    built: dict[str, dict[str, Any]] = {}
    for threshold_name, thresholds in (("high", HIGH_THRESHOLDS), ("low", LOW_THRESHOLDS)):
        built[threshold_name] = {}
        for split, payload in splits.items():
            dec = _to_decisions(payload["src"], payload["prefix"], oof=payload["oof"], thresholds=thresholds)
            state = _state_base(payload["frame"], payload["src"], dec, payload["prefix"])
            built[threshold_name][split] = {"frame": payload["frame"], "dec": dec, "state": state}

    # Train one exit editor per candidate-generator policy. OOS is never used for training.
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}
    requested = tuple(x.strip() for x in str(args.generators).split(",") if x.strip())
    unknown = [x for x in requested if x not in {"high", "low"}]
    if unknown:
        raise RuntimeError(f"unknown generators: {unknown}")
    for threshold_name in requested:
        val = built[threshold_name]["validation"]
        oos = built[threshold_name]["oos"]
        x_train, rewards, data_diag = _collect_dataset(
            val["frame"],
            val["dec"],
            val["state"],
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            stride=int(args.stride),
            max_states=int(args.max_states),
            max_forward_bars=int(args.max_forward_bars),
        )
        model, train_diag = _train_q(x_train, rewards, epochs=int(args.epochs), seed=int(args.seed), cql_weight=float(args.cql_weight))
        torch.save({"state_dict": model.state_dict(), "norm": model.norm, "actions": ACTION_NAMES, "train_diag": train_diag}, OUT_DIR / f"{threshold_name}_exit_q_editor.pt")  # type: ignore[attr-defined]
        x_train.to_csv(OUT_DIR / f"{threshold_name}_train_states.csv", index=False)
        pd.DataFrame(rewards, columns=[f"reward_{a}" for a in ACTION_NAMES]).to_csv(OUT_DIR / f"{threshold_name}_train_rewards.csv", index=False)

        val_base, val_base_ledger = _simulate_policy(val["frame"], val["dec"], val["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        oos_base, oos_base_ledger = _simulate_policy(oos["frame"], oos["dec"], oos["state"], model=None, min_adv=1.0, fee=fee, slip=slip, cost_mult=3.0, allowed_full_exit=False)
        val_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_validation_baseline_ledger.csv", index=False)
        oos_base_ledger.to_csv(OUT_DIR / f"{threshold_name}_oos_baseline_ledger.csv", index=False)
        rows.append({"candidate_generator": threshold_name, "policy": "baseline_no_rl_exit", "min_adv": None, **_row("val", val_base), **_row("oos", oos_base)})

        full_exit_modes = tuple(bool(int(x.strip())) for x in str(args.full_exit_modes).split(",") if x.strip())
        min_advs = tuple(float(x.strip()) for x in str(args.min_advs).split(",") if x.strip())
        for allowed_full_exit in full_exit_modes:
            for min_adv in min_advs:
                val_m, val_ledger = _simulate_policy(
                    val["frame"],
                    val["dec"],
                    val["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                oos_m, oos_ledger = _simulate_policy(
                    oos["frame"],
                    oos["dec"],
                    oos["state"],
                    model=model,
                    min_adv=float(min_adv),
                    fee=fee,
                    slip=slip,
                    cost_mult=3.0,
                    allowed_full_exit=bool(allowed_full_exit),
                )
                row = {
                    "candidate_generator": threshold_name,
                    "policy": "exit_q_editor_full_exit" if allowed_full_exit else "exit_q_editor_defensive",
                    "min_adv": float(min_adv),
                    **_row("val", val_m),
                    **_row("oos", oos_m),
                    "val_entry_audit": _entry_audit(val_base_ledger, val_ledger),
                    "oos_entry_audit": _entry_audit(oos_base_ledger, oos_ledger),
                }
                rows.append(row)
                tag = f"{threshold_name}_{'full' if allowed_full_exit else 'def'}_adv{str(min_adv).replace('.', 'p')}"
                val_ledger.to_csv(OUT_DIR / f"validation_{tag}_ledger.csv", index=False)
                oos_ledger.to_csv(OUT_DIR / f"oos_{tag}_ledger.csv", index=False)
        reports[threshold_name] = {"dataset": data_diag, "training": train_diag, "thresholds": HIGH_THRESHOLDS if threshold_name == "high" else LOW_THRESHOLDS}

    ranking = pd.DataFrame(rows)
    if "high" in set(ranking["candidate_generator"].astype(str)):
        high_base = ranking[(ranking["candidate_generator"] == "high") & (ranking["policy"] == "baseline_no_rl_exit")].iloc[0]
    else:
        high_base = ranking[ranking["policy"] == "baseline_no_rl_exit"].iloc[0]
    ranking["delta_vs_high_base_oos_pnl"] = ranking["oos_pnl"] - float(high_base["oos_pnl"])
    ranking["delta_vs_high_base_val_pnl"] = ranking["val_pnl"] - float(high_base["val_pnl"])
    ranking["score"] = ranking["oos_pnl"] + 0.45 * ranking["val_pnl"] + 0.35 * ranking["oos_mdd"] + 0.25 * ranking["val_mdd"]
    ranking = ranking.sort_values(["oos_pnl", "val_pnl", "score"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "exit_only_rl_editor_ranking.csv", index=False)
    promotable = ranking[
        (ranking["policy"] != "baseline_no_rl_exit")
        & (ranking["oos_pnl"] > float(high_base["oos_pnl"]))
        & (ranking["val_pnl"] > float(high_base["val_pnl"]) * 0.80)
        & (ranking["oos_mdd"] >= float(high_base["oos_mdd"]) * 1.35)
    ].copy()
    promotable.to_csv(OUT_DIR / "exit_only_rl_editor_promotable.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "architect_review": {
            "decision": "entry creation remains Omega1.2.1; RL is restricted to in-position exit editing",
            "actions": ACTION_NAMES,
            "safety_net": "true-leverage TP/SL stays active before RL action each bar",
            "oos_protocol": "train on 2025 validation OOF only, evaluate 2026 OOS once; high and low entry-threshold generators are reported separately",
        },
        "baseline_high_threshold": high_base.to_dict(),
        "threshold_reports": reports,
        "promotable_count": int(len(promotable)),
        "top": ranking.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "exit_only_rl_editor_ranking.csv"),
            "promotable": str(OUT_DIR / "exit_only_rl_editor_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top5": ranking.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
