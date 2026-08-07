#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_stop_loss_hazard_veto_20260604 as hazard_base  # noqa: E402
import eval_omega1_2_true3head_overlays_20260604 as overlay  # noqa: E402
import train_eval_omega1_2_1_exposure_selector_20260606 as exposure  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as tabm  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_1_exit_rl_editor_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"

BASE_TP = 0.026
BASE_SL = 0.014
COMPENSATED_SCALE = 2.0
MARGIN_CAP = 0.90
TRUE_LEVERAGE = 2.0

ACTION_NAMES = ["hold", "tighten_sl", "reduce50", "full_exit"]
HOLD = 0
TIGHTEN_SL = 1
REDUCE50 = 2
FULL_EXIT = 3


@dataclass
class Pos:
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    entry_equity: float = 1.0
    notional: float = 0.0
    margin: float = 0.0
    leverage: float = 1.0
    tp: float = 0.0
    sl: float = 0.0
    sl_floor: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0


class ExitEditorNet(nn.Module):
    def __init__(self, input_dim: int, hidden: int = 192) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
        )
        self.q = nn.Linear(hidden, len(ACTION_NAMES))
        self.policy = nn.Linear(hidden, len(ACTION_NAMES))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(x)
        return self.q(h), self.policy(h)


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


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _set_thresholds(src: pd.DataFrame, prefix: str, thr_map: dict[str, float]) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    expert = out[f"{prefix}router_expert"].astype(str).replace({"chop_expert": "chop"}).to_numpy()
    thr = np.asarray([thr_map.get(str(x), thr_map["chop"]) for x in expert], dtype=np.float64)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return out


def _build_dec(src: pd.DataFrame, prefix: str, *, oof: bool, thr_map: dict[str, float]) -> pd.DataFrame:
    dec = omega._to_fixed_decisions(_set_thresholds(src, prefix, thr_map), oof=oof)
    active = omega._active(dec)
    for expert, scale in overlay.SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & dec["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(overlay.BASE_SCALES[key])
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
    active = np.flatnonzero(omega._active(out))
    if len(active) == 0:
        return out
    base_notional = pd.to_numeric(out.loc[active, "notional_exposure"], errors="raise").to_numpy(dtype=np.float64)
    margin = np.minimum(base_notional * COMPENSATED_SCALE, MARGIN_CAP)
    ratio = margin / np.maximum(base_notional, 1e-12)
    out.loc[active, "notional_exposure"] = margin * TRUE_LEVERAGE
    out.loc[active, "position_fraction"] = margin
    out.loc[active, "leverage"] = TRUE_LEVERAGE
    out.loc[active, "take_profit"] = BASE_TP * ratio * TRUE_LEVERAGE
    out.loc[active, "stop_loss"] = BASE_SL * ratio * TRUE_LEVERAGE
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        bad = out.loc[out.isna().any(axis=1), "timestamp"].head(10).tolist()
        raise RuntimeError(f"prediction alignment produced NaN: {bad}")
    return out


def _load_frames(device: torch.device, *, low_delta: float) -> dict[str, Any]:
    frames = tabm._prepare_frames(disable_tp_sl=False)
    bundle = torch.load(BASE_DIR / "true_3head_tabm_bundle.pt", map_location=device, weights_only=False)
    _train_x, train_src = hazard_base._predict_frame(frames["train_raw"], bundle, oof=True, device=device)
    val_pred = pd.read_csv(BASE_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"], low_memory=False)
    oos_pred = pd.read_csv(BASE_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"], low_memory=False)
    val_src = _align(frames["val_raw"], val_pred)
    oos_src = _align(frames["oos_raw"], oos_pred)

    high_thr = dict(overlay.THR_MAP)
    low_thr = {k: max(0.05, float(v) - float(low_delta)) for k, v in high_thr.items()}

    train_high = _build_dec(train_src, "omega1_regime3_expertdq_oof_", oof=True, thr_map=high_thr)
    train_low = _build_dec(train_src, "omega1_regime3_expertdq_oof_", oof=True, thr_map=low_thr)
    val_high = _build_dec(val_src, "omega1_regime3_expertdq_oof_", oof=True, thr_map=high_thr)
    val_low = _build_dec(val_src, "omega1_regime3_expertdq_oof_", oof=True, thr_map=low_thr)
    oos_high = _build_dec(oos_src, "omega1_regime3_expertdq_", oof=False, thr_map=high_thr)
    oos_low = _build_dec(oos_src, "omega1_regime3_expertdq_", oof=False, thr_map=low_thr)

    return {
        "train_frame": frames["train_raw"].reset_index(drop=True),
        "val_frame": frames["val_raw"].reset_index(drop=True),
        "oos_frame": frames["oos_raw"].reset_index(drop=True),
        "train_src": train_src.reset_index(drop=True),
        "val_src": val_src.reset_index(drop=True),
        "oos_src": oos_src.reset_index(drop=True),
        "train_high": train_high,
        "train_low": train_low,
        "val_high": val_high,
        "val_low": val_low,
        "oos_high": oos_high,
        "oos_low": oos_low,
        "high_thr": high_thr,
        "low_thr": low_thr,
    }


def _base_features(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = exposure._feature_frame(frame, src, dec, prefix)
    bad = [c for c in out.columns if c.startswith(("clean_regime4_", "regime4_pred_", "teacher_")) or c == "tp_sl_action_score"]
    if bad:
        raise RuntimeError(f"forbidden exit RL feature columns: {bad[:30]}")
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).reset_index(drop=True)


def _pos_values(arrays: dict[str, np.ndarray], pos: Pos, i: int, *, slip_eff: float) -> dict[str, float]:
    if pos.side == 0:
        return {
            "pos_side": 0.0,
            "pos_unreal": 0.0,
            "pos_mfe": 0.0,
            "pos_mae": 0.0,
            "pos_giveback": 0.0,
            "pos_hold_bars": 0.0,
            "pos_dist_tp": 0.0,
            "pos_dist_sl": 0.0,
            "pos_notional": 0.0,
            "pos_margin": 0.0,
            "pos_leverage": 0.0,
        }
    px = float(arrays["close"][int(i)])
    raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
    unreal = raw * pos.notional
    mfe = max(pos.mfe, unreal)
    mae = min(pos.mae, unreal)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0.0 else 0.0
    return {
        "pos_side": float(pos.side),
        "pos_unreal": float(unreal),
        "pos_mfe": float(mfe),
        "pos_mae": float(mae),
        "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "pos_hold_bars": float(max(0, int(i) - int(pos.entry_i))),
        "pos_dist_tp": float(pos.tp - unreal),
        "pos_dist_sl": float(unreal - pos.sl_floor),
        "pos_notional": float(pos.notional),
        "pos_margin": float(pos.margin),
        "pos_leverage": float(pos.leverage),
    }


def _state_row(base: pd.DataFrame, arrays: dict[str, np.ndarray], pos: Pos, i: int, *, slip_eff: float) -> pd.Series:
    row = base.iloc[int(i)].copy()
    for k, v in _pos_values(arrays, pos, i, slip_eff=slip_eff).items():
        row[k] = v
    return row


def _enter(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, *, fee_eff: float, slip_eff: float) -> tuple[float, Pos, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0:
        return cash, Pos(), "no_signal"
    filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Pos(), "entry_miss"
    notional = float(row.get("notional_exposure", 0.0) or 0.0)
    if notional <= 0.0:
        return cash, Pos(), "zero_notional"
    cash -= cash * float(entry_fee) * notional
    sl = abs(float(row.get("stop_loss", 0.0) or 0.0))
    return (
        cash,
        Pos(
            side=side,
            entry_price=float(px),
            entry_i=min(int(i) + 1, len(arrays["close"]) - 1),
            entry_equity=float(cash),
            notional=notional,
            margin=float(row.get("position_fraction", 0.0) or 0.0),
            leverage=float(row.get("leverage", 1.0) or 1.0),
            tp=float(row.get("take_profit", 0.0) or 0.0),
            sl=sl,
            sl_floor=-sl,
        ),
        "entry",
    )


def _realize(cash: float, arrays: dict[str, np.ndarray], pos: Pos, i: int, frac: float, *, fee_eff: float, slip_eff: float) -> tuple[float, Pos, float]:
    if pos.side == 0:
        return cash, pos, 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    exit_px = omega._fill_price(arrays, int(i), pos.side, slip_eff, entry=False)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1e-12)
    reduce_notional = pos.notional * frac
    before = cash
    cash = cash * (1.0 + raw * reduce_notional)
    cash -= before * float(fee_eff) * reduce_notional
    new = Pos(**pos.__dict__)
    new.notional = max(0.0, pos.notional - reduce_notional)
    if new.notional <= 1e-9:
        new = Pos()
    return cash, new, raw * reduce_notional


def _update_pos(arrays: dict[str, np.ndarray], pos: Pos, i: int, *, slip_eff: float) -> tuple[Pos, float]:
    vals = _pos_values(arrays, pos, i, slip_eff=slip_eff)
    pos.mfe = max(pos.mfe, vals["pos_unreal"])
    pos.mae = min(pos.mae, vals["pos_unreal"])
    return pos, float(vals["pos_unreal"])


def _continue(cash: float, arrays: dict[str, np.ndarray], pos: Pos, start_i: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, str]:
    if pos.side == 0:
        return cash, "flat"
    end_i = min(len(arrays["close"]) - 2, int(start_i) + int(max_bars))
    reason = "sim_horizon"
    for j in range(int(start_i), end_i + 1):
        pos, unreal = _update_pos(arrays, pos, j, slip_eff=slip_eff)
        if unreal >= pos.tp > 0.0:
            cash, _pos, _ = _realize(cash, arrays, pos, j, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
            return cash, "take_profit"
        if unreal <= pos.sl_floor:
            cash, _pos, _ = _realize(cash, arrays, pos, j, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
            return cash, "stop_or_floor"
    cash, _pos, _ = _realize(cash, arrays, pos, end_i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
    return cash, reason


def _action_reward(arrays: dict[str, np.ndarray], pos: Pos, i: int, action: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, str]:
    cash = 1.0
    p = Pos(**pos.__dict__)
    if action == TIGHTEN_SL:
        vals = _pos_values(arrays, p, i, slip_eff=slip_eff)
        p.sl_floor = max(p.sl_floor, min(float(vals["pos_unreal"]), 0.0))
        cash, reason = _continue(cash, arrays, p, int(i) + 1, fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
        return float(cash - 1.0) - 0.0002, f"tighten_{reason}"
    if action == REDUCE50:
        cash, p, _ = _realize(cash, arrays, p, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
        cash, reason = _continue(cash, arrays, p, int(i) + 1, fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
        return float(cash - 1.0) - 0.0005, f"reduce_{reason}"
    if action == FULL_EXIT:
        cash, _p, _ = _realize(cash, arrays, p, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        return float(cash - 1.0) - 0.0008, "full_exit"
    cash, reason = _continue(cash, arrays, p, int(i) + 1, fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
    return float(cash - 1.0), reason


def _fit_norm(df: pd.DataFrame) -> dict[str, Any]:
    arr = df.to_numpy(dtype=np.float64)
    med = np.nanmedian(arr, axis=0)
    q25 = np.nanpercentile(arr, 25, axis=0)
    q75 = np.nanpercentile(arr, 75, axis=0)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-8)] = 1.0
    return {"columns": list(df.columns), "median": med.astype(np.float32), "scale": scale.astype(np.float32)}


def _apply_norm(df: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    cols = list(norm["columns"])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"exit RL state missing columns: {missing[:20]}")
    arr = df[cols].to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    return np.tanh(np.nan_to_num(out, nan=0.0, posinf=8.0, neginf=-8.0) / 3.0).astype(np.float32)


def _build_dataset(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str, *, max_entries: int, samples_per_entry: int, seed: int, fee: float, slip: float, cost_mult: float, max_bars: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    arrays = _arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base = _base_features(frame, src, dec, prefix)
    active = np.flatnonzero(omega._active(dec) & (np.arange(len(dec)) < len(dec) - max(8, int(max_bars) + 2)))
    total_active = int(len(active))
    if max_entries > 0 and len(active) > max_entries:
        active = np.sort(rng.choice(active, size=int(max_entries), replace=False))
    states: list[pd.Series] = []
    q_targets: list[np.ndarray] = []
    best_actions: list[int] = []
    reason_counts: dict[str, int] = {}
    for entry_idx in active:
        cash, pos, reason = _enter(1.0, arrays, dec, int(entry_idx), fee_eff=fee_eff, slip_eff=slip_eff)
        if reason != "entry" or pos.side == 0:
            continue
        offsets = [1, 2, 3, 6, 12, 24, 48, 96][: max(1, int(samples_per_entry))]
        for off in offsets:
            j = min(pos.entry_i + int(off), len(frame) - 3)
            p = Pos(**pos.__dict__)
            alive = True
            for k in range(pos.entry_i, j + 1):
                p, unreal = _update_pos(arrays, p, k, slip_eff=slip_eff)
                if unreal >= p.tp > 0.0 or unreal <= p.sl_floor:
                    alive = False
                    break
            if not alive:
                break
            rewards = np.zeros(len(ACTION_NAMES), dtype=np.float32)
            for action in range(len(ACTION_NAMES)):
                reward, r = _action_reward(arrays, p, j, action, fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
                rewards[action] = float(reward)
                reason_counts[r] = reason_counts.get(r, 0) + 1
            best = int(np.argmax(rewards))
            states.append(_state_row(base, arrays, p, j, slip_eff=slip_eff))
            q_targets.append(rewards)
            best_actions.append(best)
    if not states:
        raise RuntimeError("empty exit RL dataset")
    state_df = pd.DataFrame(states).reset_index(drop=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    q = np.asarray(q_targets, dtype=np.float32)
    a = np.asarray(best_actions, dtype=np.int64)
    diag = {
        "total_low_threshold_active": total_active,
        "used_entries": int(len(active)),
        "samples": int(len(state_df)),
        "best_action_counts": {ACTION_NAMES[i]: int(v) for i, v in enumerate(np.bincount(a, minlength=len(ACTION_NAMES)))},
        "q_mean": float(np.mean(q)),
        "q_best_mean": float(np.mean(q.max(axis=1))),
        "counterfactual_reasons": reason_counts,
    }
    return state_df, q, a, np.ones(len(a), dtype=np.float32), diag


def _train(x: np.ndarray, q: np.ndarray, a: np.ndarray, w: np.ndarray, *, device: torch.device, steps: int, batch_size: int, lr: float) -> tuple[ExitEditorNet, dict[str, Any]]:
    model = ExitEditorNet(x.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    counts = np.bincount(a, minlength=len(ACTION_NAMES)).astype(np.float32)
    counts[counts < 1.0] = 1.0
    class_w = torch.from_numpy(np.clip(counts.sum() / (len(ACTION_NAMES) * counts), 0.25, 8.0).astype(np.float32)).to(device)
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(q), torch.from_numpy(a), torch.from_numpy(w))
    dl = DataLoader(ds, batch_size=int(batch_size), shuffle=True, drop_last=len(ds) >= int(batch_size))
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        for xb, qb, ab, wb in dl:
            xb = xb.to(device)
            qb = qb.to(device)
            ab = ab.to(device)
            wb = wb.to(device)
            q_pred, logits = model(xb)
            critic_loss = torch.nn.functional.smooth_l1_loss(q_pred, qb)
            ce = torch.nn.functional.cross_entropy(logits, ab, reduction="none")
            actor_w = wb * class_w[ab]
            actor_loss = (ce * actor_w).sum() / torch.clamp(actor_w.sum(), min=1.0)
            probs = torch.softmax(logits, dim=1)
            policy_q = (probs * q_pred.detach()).sum(dim=1).mean()
            loss = critic_loss + actor_loss - 0.15 * policy_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        if step % 50 == 0:
            last = {
                "epoch": int(step),
                "critic_loss": float(critic_loss.detach().cpu()),
                "actor_loss": float(actor_loss.detach().cpu()),
                "policy_q": float(policy_q.detach().cpu()),
            }
    return model.cpu(), last


def _allowed_exit_actions(state: pd.Series) -> list[int]:
    unreal = float(state.get("pos_unreal", 0.0) or 0.0)
    mfe = float(state.get("pos_mfe", 0.0) or 0.0)
    giveback = float(state.get("pos_giveback", 0.0) or 0.0)
    hold_bars = float(state.get("pos_hold_bars", 0.0) or 0.0)
    allowed = [HOLD]
    if hold_bars >= 2.0 and mfe >= 0.025:
        allowed.append(TIGHTEN_SL)
    if hold_bars >= 2.0 and unreal >= 0.035:
        allowed.append(REDUCE50)
    if hold_bars >= 2.0 and ((mfe >= 0.04 and giveback >= 0.65) or unreal <= -0.045):
        allowed.append(FULL_EXIT)
    return allowed


@torch.no_grad()
def _select(model: ExitEditorNet, state: pd.Series, norm: dict[str, Any], *, device: torch.device, min_q_edge: float, mode: str) -> int:
    x = _apply_norm(pd.DataFrame([state]).reset_index(drop=True), norm)
    q, logits = model(torch.from_numpy(x).to(device))
    if mode == "q":
        scores = q[0]
    else:
        scores = torch.softmax(logits[0], dim=0) * torch.clamp(q[0], min=-0.05, max=0.05).add(0.05)
    allowed = _allowed_exit_actions(state)
    mask = torch.full_like(scores, -1e9)
    mask[[int(a) for a in allowed]] = 0.0
    scores = scores + mask
    action = int(torch.argmax(scores).detach().cpu().item())
    hold_score = float(scores[HOLD].detach().cpu())
    act_score = float(scores[action].detach().cpu())
    if action != HOLD and (act_score - hold_score) < float(min_q_edge):
        return HOLD
    return action


def _replay(frame: pd.DataFrame, src: pd.DataFrame, dec: pd.DataFrame, prefix: str, model: ExitEditorNet, norm: dict[str, Any], *, device: torch.device, fee: float, slip: float, cost_mult: float, min_q_edge: float, mode: str) -> tuple[dict[str, Any], pd.DataFrame]:
    model = model.to(device)
    model.eval()
    arrays = _arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base = _base_features(frame, src, dec, prefix)
    active = omega._active(dec)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = Pos()
    rows: list[dict[str, Any]] = []
    reasons: dict[str, int] = {}
    trades: list[float] = []
    long_entries = 0
    short_entries = 0
    partials = 0
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            state = _state_row(base, arrays, pos, i, slip_eff=slip_eff)
            pos, unreal = _update_pos(arrays, pos, i, slip_eff=slip_eff)
            eq = cash * (1.0 + unreal)
            peak = max(peak, eq)
            mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
            reason = ""
            action = HOLD
            if unreal >= pos.tp > 0.0:
                reason = "take_profit"
            elif unreal <= pos.sl_floor:
                reason = "stop_loss" if pos.sl_floor <= -abs(pos.sl) + 1e-12 else "tightened_sl_exit"
            else:
                action = _select(model, state, norm, device=device, min_q_edge=min_q_edge, mode=mode)
                if action == TIGHTEN_SL:
                    pos.sl_floor = max(pos.sl_floor, min(unreal, 0.0))
                    reasons["tighten_sl"] = reasons.get("tighten_sl", 0) + 1
                elif action == REDUCE50:
                    before = cash
                    cash, pos, _ = _realize(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
                    partials += 1
                    reasons["reduce50"] = reasons.get("reduce50", 0) + 1
                    if pos.side == 0:
                        reason = "reduce50_full_exit"
                        trades.append(float((cash / max(before, 1e-12) - 1.0) * 100.0))
                elif action == FULL_EXIT:
                    reason = "rl_full_exit"
            if reason:
                before_trade = pos.entry_equity
                cash, _pos, _ = _realize(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                net_pct = float((cash / max(before_trade, 1e-12) - 1.0) * 100.0)
                trades.append(net_pct)
                reasons[reason] = reasons.get(reason, 0) + 1
                rows.append({"trade_id": len(rows) + 1, "side": "LONG" if pos.side > 0 else "SHORT", "entry_i": int(pos.entry_i), "exit_i": int(i), "entry_time": str(frame["timestamp"].iloc[int(pos.entry_i)]), "exit_time": str(frame["timestamp"].iloc[int(i)]), "net_trade_return_pct": net_pct, "exit_reason": reason, "action": ACTION_NAMES[action], "cash_after": float(cash)})
                pos = Pos()
            continue
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
        if not bool(active[i]):
            continue
        cash, new_pos, reason = _enter(cash, arrays, dec, i, fee_eff=fee_eff, slip_eff=slip_eff)
        if reason == "entry":
            pos = new_pos
            long_entries += int(pos.side > 0)
            short_entries += int(pos.side < 0)
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
    if pos.side != 0:
        before_trade = pos.entry_equity
        cash, _pos, _ = _realize(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        trades.append(float((cash / max(before_trade, 1e-12) - 1.0) * 100.0))
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
        rows.append({"trade_id": len(rows) + 1, "side": "LONG" if pos.side > 0 else "SHORT", "entry_i": int(pos.entry_i), "exit_i": int(len(frame) - 1), "entry_time": str(frame["timestamp"].iloc[int(pos.entry_i)]), "exit_time": str(frame["timestamp"].iloc[-1]), "net_trade_return_pct": trades[-1], "exit_reason": "forced_end", "action": "forced_end", "cash_after": float(cash)})
    return (
        {
            "pnl": float((cash - 1.0) * 100.0),
            "mdd": float(mdd * 100.0),
            "trades": int(len(trades)),
            "wr": float(np.mean(np.asarray(trades) > 0.0)) if trades else 0.0,
            "long_entries": int(long_entries),
            "short_entries": int(short_entries),
            "partials": int(partials),
            "reasons": reasons,
        },
        pd.DataFrame(rows),
    )


def _metric_row(prefix: str, m: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(m["pnl"]),
        f"{prefix}_mdd": float(m["mdd"]),
        f"{prefix}_wr": float(m["wr"]),
        f"{prefix}_trades": int(m["trades"]),
        f"{prefix}_long": int(m["long_entries"]),
        f"{prefix}_short": int(m["short_entries"]),
        f"{prefix}_partials": int(m.get("partials", 0)),
        f"{prefix}_reasons": m["reasons"],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--low-threshold-delta", type=float, default=0.14)
    ap.add_argument("--max-train-entries", type=int, default=1600)
    ap.add_argument("--samples-per-entry", type=int, default=8)
    ap.add_argument("--max-bars", type=int, default=288)
    ap.add_argument("--epochs", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--seed", type=int, default=260610)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--eval-modes", default="q,actor_q")
    ap.add_argument("--eval-edges", default="0.0,0.001,0.003,0.006")
    args = ap.parse_args()

    _seed(int(args.seed))
    device = _device(args.device)
    out_dir = OUT_DIR / f"delta{args.low_threshold_delta:.2f}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    fee, slip = omega._load_fee_slip()
    data = _load_frames(device, low_delta=float(args.low_threshold_delta))
    state_df, q, best, w, data_diag = _build_dataset(
        data["train_frame"],
        data["train_src"],
        data["train_low"],
        "omega1_regime3_expertdq_oof_",
        max_entries=int(args.max_train_entries),
        samples_per_entry=int(args.samples_per_entry),
        seed=int(args.seed),
        fee=fee,
        slip=slip,
        cost_mult=3.0,
        max_bars=int(args.max_bars),
    )
    norm = _fit_norm(state_df)
    x = _apply_norm(state_df, norm)
    model, train_diag = _train(x, q, best, w, device=device, steps=int(args.epochs), batch_size=int(args.batch_size), lr=float(args.lr))
    torch.save({"model_state": model.state_dict(), "norm": norm, "input_dim": int(x.shape[1]), "action_names": ACTION_NAMES, "config": vars(args)}, out_dir / "exit_rl_editor.pt")

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    eval_modes = [x.strip() for x in str(args.eval_modes).split(",") if x.strip()]
    eval_edges = [float(x.strip()) for x in str(args.eval_edges).split(",") if x.strip()]
    for entry_mode in ("high_entry", "low_entry"):
        val_dec = data["val_high"] if entry_mode == "high_entry" else data["val_low"]
        oos_dec = data["oos_high"] if entry_mode == "high_entry" else data["oos_low"]
        for select_mode in eval_modes:
            for min_q_edge in eval_edges:
                val_m, val_ledger = _replay(data["val_frame"], data["val_src"], val_dec, "omega1_regime3_expertdq_oof_", model, norm, device=device, fee=fee, slip=slip, cost_mult=3.0, min_q_edge=min_q_edge, mode=select_mode)
                oos_m, oos_ledger = _replay(data["oos_frame"], data["oos_src"], oos_dec, "omega1_regime3_expertdq_", model, norm, device=device, fee=fee, slip=slip, cost_mult=3.0, min_q_edge=min_q_edge, mode=select_mode)
                row = {"entry_mode": entry_mode, "select_mode": select_mode, "min_q_edge": float(min_q_edge)}
                row.update(_metric_row("validation", val_m))
                row.update(_metric_row("oos", oos_m))
                rows.append(row)
                key = f"{entry_mode}_{select_mode}_edge{min_q_edge:.3f}"
                ledgers[f"validation_{key}"] = val_ledger
                ledgers[f"oos_{key}"] = oos_ledger
    ranking = pd.DataFrame(rows)
    ranking = ranking.sort_values(["validation_pnl", "oos_pnl"], ascending=False).reset_index(drop=True)
    ranking.to_csv(out_dir / "exit_rl_ranking.csv", index=False)
    for key in list(ledgers)[:0]:
        ledgers[key].to_csv(out_dir / f"{key}.csv", index=False)
    for _, r in ranking.head(4).iterrows():
        key = f"{r['entry_mode']}_{r['select_mode']}_edge{float(r['min_q_edge']):.3f}"
        ledgers[f"validation_{key}"].to_csv(out_dir / f"validation_{key}_ledger.csv", index=False)
        ledgers[f"oos_{key}"].to_csv(out_dir / f"oos_{key}_ledger.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "method": "Low-threshold candidate generation for offline exit-only RL. Entry/risk contract is frozen at Omega1.2.1 true leverage. RL actions are hold/tighten_sl/reduce50/full_exit.",
        "thresholds": {"high": data["high_thr"], "low": data["low_thr"]},
        "data_diag": data_diag,
        "train_diag": train_diag,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {"out_dir": str(out_dir), "ranking": str(out_dir / "exit_rl_ranking.csv"), "model": str(out_dir / "exit_rl_editor.pt")},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(ranking.head(12).to_string(index=False))
    print(json.dumps(report["artifacts"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
