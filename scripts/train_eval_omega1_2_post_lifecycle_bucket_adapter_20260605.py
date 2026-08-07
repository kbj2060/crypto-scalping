#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_mamba_sac_3head_feature_coordinator_20260604 as feat_coord  # noqa: E402
import train_eval_omega1_2_mamba_sac_lifecycle_controller_20260604 as lifecycle  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402


MODEL_ID = "omega1_2_post_lifecycle_bucket_adapter_20260605"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASELINE_LIFECYCLE_DIR = ROOT / "data/ensemble/supervised/omega1_2_exit_feature_lifecycle_baseline_20260604"

TP_BUCKETS = np.asarray([0.018, 0.022, 0.026, 0.030, 0.034], dtype=np.float32)
SL_BUCKETS = np.asarray([0.008, 0.010, 0.012, 0.014, 0.018], dtype=np.float32)
NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
ENTRY_BUCKETS = np.asarray(
    [(t, s, n, l) for t in range(len(TP_BUCKETS)) for s in range(len(SL_BUCKETS)) for n in range(len(NOTIONAL_BUCKETS)) for l in range(len(LEVERAGE_BUCKETS))],
    dtype=np.int64,
)
BASE_IDS = (2, 3, 3, 1)
NOTIONAL_CAP = 1.20
NOTIONAL_MULT = 1.00
USE_LEVERAGE_EXPOSURE = False
COMPENSATE_SLTP_BY_NOTIONAL = False
COMPENSATE_REF_NOTIONAL = 0.45
USE_ATR_RISK = False
ATR_TP_CLIP = (0.008, 0.055)
ATR_SL_CLIP = (0.006, 0.040)


def _set_bucket_preset(name: str) -> None:
    global TP_BUCKETS, SL_BUCKETS, NOTIONAL_BUCKETS, LEVERAGE_BUCKETS, ENTRY_BUCKETS, BASE_IDS, NOTIONAL_CAP, USE_ATR_RISK
    USE_ATR_RISK = False
    if str(name) == "base":
        TP_BUCKETS = np.asarray([0.018, 0.022, 0.026, 0.030, 0.034], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.008, 0.010, 0.012, 0.014, 0.018], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 3, 3, 1)
        NOTIONAL_CAP = 1.20
    elif str(name) == "aggressive":
        TP_BUCKETS = np.asarray([0.022, 0.026, 0.030, 0.034, 0.042], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.010, 0.012, 0.014, 0.018, 0.024], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.45, 0.65, 0.85, 1.10, 1.35], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (1, 2, 0, 1)
        NOTIONAL_CAP = 1.35
    elif str(name) == "side_asym":
        TP_BUCKETS = np.asarray([0.020, 0.026, 0.030, 0.036, 0.045], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.009, 0.012, 0.014, 0.018, 0.026], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.35, 0.50, 0.70, 0.95, 1.20], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (1, 2, 1, 1)
        NOTIONAL_CAP = 1.20
    elif str(name) == "atr_aggressive":
        USE_ATR_RISK = True
        TP_BUCKETS = np.asarray([1.4, 1.8, 2.2, 2.8, 3.4], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.7, 0.9, 1.1, 1.4, 1.8], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.45, 0.65, 0.85, 1.10, 1.35], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 2, 0, 1)
        NOTIONAL_CAP = 1.35
    elif str(name) == "atr_wide":
        USE_ATR_RISK = True
        TP_BUCKETS = np.asarray([3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float32)
        SL_BUCKETS = np.asarray([1.5, 2.0, 2.5, 3.0, 3.5], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.45, 0.55, 0.65, 0.80, 1.00], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 2, 0, 1)
        NOTIONAL_CAP = 1.00
    elif str(name) == "fixed_wide":
        TP_BUCKETS = np.asarray([0.026, 0.034, 0.045, 0.060, 0.080], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.018, 0.024, 0.032, 0.045, 0.060], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 2, 3, 1)
        NOTIONAL_CAP = 1.20
    elif str(name) == "fixed_ultra_wide":
        TP_BUCKETS = np.asarray([0.040, 0.060, 0.080, 0.120, 0.160], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.025, 0.040, 0.060, 0.080, 0.100], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 2, 3, 1)
        NOTIONAL_CAP = 1.20
    elif str(name) == "base_expanded_notional":
        TP_BUCKETS = np.asarray([0.018, 0.022, 0.026, 0.030, 0.034], dtype=np.float32)
        SL_BUCKETS = np.asarray([0.008, 0.010, 0.012, 0.014, 0.018], dtype=np.float32)
        NOTIONAL_BUCKETS = np.asarray([0.25, 0.3375, 0.405, 0.45, 0.55, 0.65, 0.80, 1.00], dtype=np.float32)
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
        BASE_IDS = (2, 3, 3, 1)
        NOTIONAL_CAP = 1.20
    else:
        raise RuntimeError(f"unknown bucket preset: {name}")
    ENTRY_BUCKETS = np.asarray(
        [(t, s, n, l) for t in range(len(TP_BUCKETS)) for s in range(len(SL_BUCKETS)) for n in range(len(NOTIONAL_BUCKETS)) for l in range(len(LEVERAGE_BUCKETS))],
        dtype=np.int64,
    )


@dataclass
class Position:
    side: int = 0
    entry_price: float = 0.0
    entry_i: int = 0
    notional: float = 0.0
    take_profit: float = 0.0
    stop_loss: float = 0.0
    mfe: float = 0.0
    mae: float = 0.0
    tp_upshifts: int = 0


def _to_lifecycle_position(pos: Position) -> lifecycle.Position:
    fields = getattr(lifecycle.Position, "__dataclass_fields__", {})
    allowed = set(fields.keys()) if fields else {"side", "entry_price", "entry_i", "notional", "take_profit", "stop_loss", "mfe", "mae"}
    return lifecycle.Position(**{k: v for k, v in pos.__dict__.items() if k in allowed})


@dataclass
class AdapterData:
    seq: np.ndarray
    q_targets: np.ndarray
    best_actions: np.ndarray
    weights: np.ndarray


class MambaBucketAdapter(nn.Module):
    def __init__(self, input_dim: int, n_actions: int, d_model: int = 96, emb_dim: int = 96) -> None:
        super().__init__()
        self.enc = feat_coord.old_coord.MambaEncoder(input_dim, d_model, emb_dim)
        self.actor = nn.Sequential(nn.Linear(emb_dim, 128), nn.SiLU(), nn.Linear(128, n_actions))
        self.critic = nn.Sequential(nn.Linear(emb_dim, 128), nn.SiLU(), nn.Linear(128, n_actions))

    def forward(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.enc(seq)
        return self.actor(h), self.critic(h)


def _json_default(obj: Any) -> Any:
    return omega._json_default(obj)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    return torch.device("cuda" if (name == "cuda" or (name == "auto" and torch.cuda.is_available())) else "cpu")


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _atr_pct(frame: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise").astype(float)
    low = pd.to_numeric(frame["low"], errors="raise").astype(float)
    close = pd.to_numeric(frame["close"], errors="raise").astype(float)
    tr = pd.concat([(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=int(period), adjust=False).mean()
    out = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan).ffill().fillna(0.005)
    return np.clip(out.to_numpy(dtype=np.float64), 0.0005, 0.05)


def _risk_from_ids(ids: tuple[int, int, int, int] | np.ndarray, atr_pct: float | None = None) -> dict[str, float]:
    t, s, n, l = [int(x) for x in ids]
    leverage = float(LEVERAGE_BUCKETS[l])
    margin_notional = float(np.clip(float(NOTIONAL_BUCKETS[n]) * float(NOTIONAL_MULT), 0.0, float(NOTIONAL_CAP)))
    effective_notional = float(np.clip(margin_notional * leverage, 0.0, float(NOTIONAL_CAP))) if bool(USE_LEVERAGE_EXPOSURE) else margin_notional
    if bool(USE_ATR_RISK):
        atr = float(atr_pct if atr_pct is not None else 0.005)
        tp = float(np.clip(atr * float(TP_BUCKETS[t]), *ATR_TP_CLIP))
        sl = float(np.clip(atr * float(SL_BUCKETS[s]), *ATR_SL_CLIP))
    else:
        tp = float(TP_BUCKETS[t])
        sl = float(SL_BUCKETS[s])
        if bool(COMPENSATE_SLTP_BY_NOTIONAL):
            ref = max(float(COMPENSATE_REF_NOTIONAL), 1e-8)
            tp = float(tp / ref * effective_notional)
            sl = float(sl / ref * effective_notional)
    return {"tp": tp, "sl": sl, "notional": effective_notional, "margin_notional": margin_notional, "leverage": leverage}


def _single_dec_row(action: int, side: int, ids: tuple[int, int, int, int] | np.ndarray, atr_pct: float | None = None) -> pd.Series:
    r = _risk_from_ids(ids, atr_pct=atr_pct)
    return pd.Series(
        {
            "action": int(action),
            "side": int(side),
            "quality_score": 1.0,
            "confidence": 1.0,
            "notional_exposure": r["notional"],
            "position_fraction": r["notional"],
            "leverage": r["leverage"],
            "max_hold_bars": 0,
            "cooldown_bars": 0,
            "take_profit": r["tp"],
            "stop_loss": r["sl"],
        }
    )


def _fit_norm(x: pd.DataFrame) -> tuple[np.ndarray, dict[str, Any]]:
    arr = x.to_numpy(dtype=np.float32)
    med = np.nanmedian(arr, axis=0).astype(np.float32)
    q25 = np.nanpercentile(arr, 25, axis=0).astype(np.float32)
    q75 = np.nanpercentile(arr, 75, axis=0).astype(np.float32)
    scale = q75 - q25
    scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
    out = (arr - med) / scale
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite post adapter matrix")
    return np.tanh(out / 3.0).astype(np.float32), {"columns": list(x.columns), "median": med, "scale": scale}


def _apply_norm(x: pd.DataFrame, norm: dict[str, Any]) -> np.ndarray:
    if list(x.columns) != list(norm["columns"]):
        raise RuntimeError("post adapter feature column contract mismatch")
    arr = x.to_numpy(dtype=np.float32)
    out = (arr - norm["median"]) / norm["scale"]
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite post adapter inference matrix")
    return np.tanh(out / 3.0).astype(np.float32)


def _load_baseline_lifecycle(path: Path) -> tuple[lifecycle.MambaDiscreteActorCritic, dict[str, Any]]:
    ckpt = torch.load(path / "lifecycle_controller.pt", map_location="cpu", weights_only=False)
    model = lifecycle.MambaDiscreteActorCritic(len(ckpt["state_columns"]), len(lifecycle.ACTION_NAMES))
    model.load_state_dict(ckpt["model_state_dict"])
    return model, ckpt


def _base_frames(threehead_dir: Path, quality_threshold: float, device: torch.device) -> dict[str, Any]:
    return feat_coord._prepare_frames(threehead_dir, quality_threshold=float(quality_threshold), device=device)


def _position_values(arrays: dict[str, np.ndarray], pos: Position, i: int, *, slip_eff: float) -> dict[str, float]:
    if pos.side == 0 or pos.notional <= 0.0:
        return {c: 0.0 for c in lifecycle.POS_COLS}
    px = float(arrays["close"][int(i)])
    raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
    unreal = raw * pos.notional
    mfe = max(float(pos.mfe), unreal)
    mae = min(float(pos.mae), unreal)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0 else 0.0
    return {
        "lc_pos_side": float(pos.side),
        "lc_pos_notional": float(pos.notional),
        "lc_pos_entry_raw": float((px - pos.entry_price) / max(pos.entry_price, 1e-12)),
        "lc_pos_unrealized": float(unreal),
        "lc_pos_mfe": float(mfe),
        "lc_pos_mae": float(mae),
        "lc_pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "lc_pos_hold_bars": float(max(int(i) - int(pos.entry_i), 0)),
        "lc_pos_dist_tp": float(pos.take_profit - unreal),
        "lc_pos_dist_sl": float(unreal + abs(pos.stop_loss)),
    }


def _state_row(state: pd.DataFrame, arrays: dict[str, np.ndarray], pos: Position, i: int, *, slip_eff: float) -> pd.DataFrame:
    row = state.iloc[[int(i)]].copy().reset_index(drop=True)
    for k, v in _position_values(arrays, pos, i, slip_eff=slip_eff).items():
        row[k] = v
    return row


def _adapter_feature_row(state_row: pd.DataFrame, lifecycle_action: int) -> pd.DataFrame:
    out = state_row.copy().reset_index(drop=True)
    out["post_lifecycle_enter_base"] = float(int(lifecycle_action) == lifecycle.ENTER_BASE)
    out["post_lifecycle_enter_aggressive"] = float(int(lifecycle_action) == lifecycle.ENTER_AGGRESSIVE)
    out["post_lifecycle_action_id"] = float(int(lifecycle_action))
    bad = [c for c in out.columns if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden post adapter features passed audit: {bad[:20]}")
    return out.drop(columns=["timestamp"], errors="ignore").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)


def _fill_exit(arrays: dict[str, np.ndarray], i: int, side: int, slip_eff: float) -> float:
    return omega._fill_price(arrays, int(i), int(side), float(slip_eff), entry=False)


def _realize_fraction(cash: float, arrays: dict[str, np.ndarray], pos: Position, i: int, frac: float, *, fee_eff: float, slip_eff: float) -> tuple[float, Position, float]:
    if pos.side == 0 or pos.notional <= 0.0 or frac <= 0.0:
        return cash, pos, 0.0
    frac = float(np.clip(frac, 0.0, 1.0))
    exit_px = _fill_exit(arrays, int(i), pos.side, slip_eff)
    raw = (exit_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1e-12)
    reduce_notional = pos.notional * frac
    before = cash
    cash = cash * (1.0 + raw * reduce_notional)
    cash -= before * float(fee_eff) * reduce_notional
    new_pos = Position(**pos.__dict__)
    new_pos.notional = max(0.0, pos.notional - reduce_notional)
    if new_pos.notional <= 1e-9:
        new_pos = Position()
    return cash, new_pos, raw * reduce_notional


def _enter_with_risk(cash: float, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, ids: tuple[int, int, int, int] | np.ndarray, *, fee_eff: float, slip_eff: float, atr_pct: float | None = None) -> tuple[float, Position, str]:
    row = dec.iloc[int(i)]
    side = int(row.get("side", 0) or 0)
    if side == 0 or int(row.get("action", 0) or 0) == omega.ACTION_CASH:
        return cash, Position(), "no_signal"
    filled, entry_px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
    if not filled:
        return cash, Position(), "entry_miss"
    r = _risk_from_ids(ids, atr_pct=atr_pct)
    notional = float(np.clip(r["notional"], 0.0, float(NOTIONAL_CAP)))
    cash -= cash * float(entry_fee) * notional
    return cash, Position(side=side, entry_price=float(entry_px), entry_i=min(int(i) + 1, len(arrays["close"]) - 1), notional=notional, take_profit=float(r["tp"]), stop_loss=abs(float(r["sl"]))), "entry"


def _continue_to_end(cash: float, arrays: dict[str, np.ndarray], pos: Position, start_i: int, *, fee_eff: float, slip_eff: float, max_bars: int) -> tuple[float, str]:
    if pos.side == 0:
        return cash, "flat"
    last_i = len(arrays["close"]) - 2 if int(max_bars) <= 0 else min(len(arrays["close"]) - 2, int(start_i) + int(max_bars))
    reason = "sim_horizon" if int(max_bars) > 0 else "forced_end"
    exit_i = last_i
    for j in range(int(start_i), last_i + 1):
        px = float(arrays["close"][j])
        raw = (px * (1.0 - slip_eff) - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - px * (1.0 + slip_eff)) / max(pos.entry_price, 1e-12)
        unreal = raw * pos.notional
        pos.mfe = max(pos.mfe, unreal)
        pos.mae = min(pos.mae, unreal)
        if pos.stop_loss > 0.0 and unreal <= -pos.stop_loss:
            reason = "stop_loss"
            exit_i = int(j)
            break
        if pos.take_profit > 0.0 and unreal >= pos.take_profit:
            reason = "take_profit"
            exit_i = int(j)
            break
    cash, _pos, _ = _realize_fraction(cash, arrays, pos, exit_i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
    return cash, reason


def _simulate_entry_bucket(frame: pd.DataFrame, arrays: dict[str, np.ndarray], dec: pd.DataFrame, i: int, ids: tuple[int, int, int, int] | np.ndarray, *, fee: float, slip: float, cost_mult: float, max_bars: int, atr_pct: float | None = None) -> tuple[float, str]:
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    cash, pos, reason = _enter_with_risk(1.0, arrays, dec, int(i), ids, fee_eff=fee_eff, slip_eff=slip_eff, atr_pct=atr_pct)
    if pos.side != 0:
        cash, reason = _continue_to_end(cash, arrays, pos, max(int(i) + 1, pos.entry_i), fee_eff=fee_eff, slip_eff=slip_eff, max_bars=max_bars)
    return float(cash - 1.0), reason


def _sample_bucket_ids(rng: np.random.Generator, n: int) -> np.ndarray:
    ids = [BASE_IDS, (1, 2, 2, 1), (3, 3, 3, 1), (4, 4, 2, 1), (0, 1, 1, 0), (2, 2, 4, 2)]
    while len(ids) < int(n):
        ids.append(
            (
                int(rng.integers(0, len(TP_BUCKETS))),
                int(rng.integers(0, len(SL_BUCKETS))),
                int(rng.integers(0, len(NOTIONAL_BUCKETS))),
                int(rng.integers(0, len(LEVERAGE_BUCKETS))),
            )
        )
    return np.asarray(ids[: int(n)], dtype=np.int64)


@torch.no_grad()
def _select_lifecycle_action(model: lifecycle.MambaDiscreteActorCritic, ckpt: dict[str, Any], base_seq: np.ndarray, row: pd.DataFrame, allowed: list[int], i: int, *, device: torch.device, select_mode: str) -> int:
    seq = lifecycle._seq_for_state(base_seq, lifecycle._apply_norm(row, ckpt["normalizer"]), int(i))
    return lifecycle._select_action(model, seq, allowed, device=device, select_mode=str(select_mode))


@torch.no_grad()
def _lifecycle_scores(model: lifecycle.MambaDiscreteActorCritic, ckpt: dict[str, Any], base_seq: np.ndarray, row: pd.DataFrame, allowed: list[int], i: int, *, device: torch.device, select_mode: str) -> np.ndarray:
    seq = lifecycle._seq_for_state(base_seq, lifecycle._apply_norm(row, ckpt["normalizer"]), int(i))
    model = model.to(device)
    model.eval()
    logits, q = model(torch.from_numpy(seq[None, :, :]).to(device))
    if str(select_mode) == "q_only":
        score = q
    else:
        score = torch.softmax(logits, dim=1) * torch.clamp(q, min=-0.05, max=0.05).add(0.05)
    arr = score.detach().cpu().numpy().reshape(-1)
    mask = np.full_like(arr, -1e9, dtype=np.float32)
    mask[[int(a) for a in allowed]] = 0.0
    return arr + mask


def _collect_train_entries(
    frames: dict[str, Any],
    model: lifecycle.MambaDiscreteActorCritic,
    ckpt: dict[str, Any],
    *,
    device: torch.device,
    select_mode: str,
    fee: float,
    slip: float,
    cost_mult: float,
    max_rows: int,
    enter_topk: int,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    frame = frames["train_df"]
    state = lifecycle._base_state(frames["s_train"])
    dec = frames["train_dec"]
    arrays = _arrays(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = lifecycle._apply_norm(state, ckpt["normalizer"])
    base_seq = lifecycle._rolling_sequences(base_norm, int(ckpt["seq_len"]))
    pos = lifecycle.Position()
    xs: list[pd.DataFrame] = []
    idxs: list[int] = []
    actions: list[int] = []
    active = omega._active(dec)
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            vals = lifecycle._position_values(arrays, pos, i, slip_eff=slip_eff)
            pos.mfe = max(pos.mfe, vals["lc_pos_unrealized"])
            pos.mae = min(pos.mae, vals["lc_pos_unrealized"])
            if pos.stop_loss > 0.0 and vals["lc_pos_unrealized"] <= -pos.stop_loss:
                _cash, pos, _ = lifecycle._realize_fraction(1.0, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                reasons["stop_loss"] = reasons.get("stop_loss", 0) + 1
                continue
            if pos.take_profit > 0.0 and vals["lc_pos_unrealized"] >= pos.take_profit:
                _cash, pos, _ = lifecycle._realize_fraction(1.0, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                reasons["take_profit"] = reasons.get("take_profit", 0) + 1
                continue
        if pos.side == 0 and not bool(active[i]):
            continue
        row = lifecycle._state_row(state, arrays, pos, i, slip_eff=slip_eff)
        allowed = lifecycle._allowed_actions(arrays, dec, pos, i, slip_eff=slip_eff, disable_resize=True, disable_reverse=True)
        scores = _lifecycle_scores(model, ckpt, base_seq, row, allowed, i, device=device, select_mode=select_mode)
        action = int(np.argmax(scores))
        enter_scores = [(lifecycle.ENTER_BASE, float(scores[lifecycle.ENTER_BASE])), (lifecycle.ENTER_AGGRESSIVE, float(scores[lifecycle.ENTER_AGGRESSIVE]))]
        enter_scores = [(a, s) for a, s in enter_scores if np.isfinite(s) and s > -1e8]
        top_actions = np.argsort(scores)[::-1][: max(int(enter_topk), 1)]
        top_enter = [a for a, _s in sorted(enter_scores, key=lambda x: x[1], reverse=True) if int(a) in set(int(x) for x in top_actions)]
        should_label_enter = pos.side == 0 and (action in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE) or bool(top_enter))
        label_action = int(action) if action in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE) else int(top_enter[0]) if top_enter else int(action)
        if should_label_enter:
            xs.append(_adapter_feature_row(row, label_action))
            idxs.append(int(i))
            actions.append(int(label_action))
            reasons["label_enter_actual" if action in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE) else "label_enter_topk"] = reasons.get("label_enter_actual" if action in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE) else "label_enter_topk", 0) + 1
            if int(max_rows) > 0 and len(idxs) >= int(max_rows) and action not in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE):
                break
        if pos.side == 0 and action in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE):
            _cash, pos, reason = lifecycle._apply_action(1.0, arrays, dec, pos, i, action, fee_eff=fee_eff, slip_eff=slip_eff)
            reasons[reason] = reasons.get(reason, 0) + 1
            if int(max_rows) > 0 and len(idxs) >= int(max_rows):
                break
        elif pos.side != 0:
            _cash, pos, reason = lifecycle._apply_action(1.0, arrays, dec, pos, i, action, fee_eff=fee_eff, slip_eff=slip_eff)
            reasons[reason] = reasons.get(reason, 0) + 1
    if not xs:
        raise RuntimeError("no post-lifecycle entry events collected")
    return pd.concat(xs, axis=0, ignore_index=True), np.asarray(idxs, dtype=np.int64), np.asarray(actions, dtype=np.int64), {"entry_rows": int(len(idxs)), "lifecycle_reasons": reasons}


def _build_labels(
    frame: pd.DataFrame,
    dec: pd.DataFrame,
    entry_idx: np.ndarray,
    *,
    seed: int,
    candidates_per_row: int,
    fee: float,
    slip: float,
    cost_mult: float,
    max_bars: int,
    min_score: float,
    exit_aware_label: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(int(seed))
    arrays = _arrays(frame)
    atr_arr = _atr_pct(frame)
    y: list[tuple[int, int, int, int]] = []
    q_targets: list[np.ndarray] = []
    best_flat: list[int] = []
    enter_labels: list[int] = []
    weights: list[float] = []
    reasons: dict[str, int] = {}
    for i in entry_idx:
        candidates = _sample_bucket_ids(rng, int(candidates_per_row))
        scores = np.full(len(ENTRY_BUCKETS), -0.02, dtype=np.float32)
        local_scores = []
        local_flat = []
        for ids in candidates:
            score, reason = _simulate_entry_bucket(frame, arrays, dec, int(i), ids, fee=fee, slip=slip, cost_mult=cost_mult, max_bars=max_bars, atr_pct=float(atr_arr[int(i)]))
            adjusted_score = float(score)
            if bool(exit_aware_label):
                if str(reason) == "take_profit":
                    adjusted_score += 0.0010
                elif str(reason) == "full_exit" and float(score) > 0.0:
                    adjusted_score += 0.0007
                elif str(reason) == "stop_loss":
                    adjusted_score -= 0.0020
                elif str(reason) == "full_exit" and float(score) <= 0.0:
                    adjusted_score -= 0.0007
            flat = int(np.flatnonzero(np.all(ENTRY_BUCKETS == ids, axis=1))[0])
            scores[flat] = float(adjusted_score)
            local_scores.append(float(adjusted_score))
            local_flat.append(flat)
            reasons[reason] = reasons.get(reason, 0) + 1
        best_i = int(np.argmax(local_scores))
        best_score = float(local_scores[best_i])
        chosen_ids = tuple(int(x) for x in candidates[best_i])
        enter_label = int(best_score >= float(min_score))
        if not bool(enter_label):
            chosen_ids = BASE_IDS
            best = int(np.flatnonzero(np.all(ENTRY_BUCKETS == np.asarray(BASE_IDS), axis=1))[0])
        else:
            best = int(local_flat[best_i])
        scale = max(float(np.std(local_scores)), 1e-4)
        y.append(chosen_ids)
        q_targets.append(scores)
        best_flat.append(best)
        enter_labels.append(enter_label)
        weights.append(float(np.exp(np.clip((float(scores[best]) - float(np.median(scores))) / scale, -4.0, 4.0))))
    y_np = np.asarray(y, dtype=np.int64)
    return y_np, np.asarray(q_targets, dtype=np.float32), np.asarray(best_flat, dtype=np.int64), np.asarray(enter_labels, dtype=np.int64), np.asarray(weights, dtype=np.float32), {
        "rows": int(len(y_np)),
        "enter_label_counts": {str(i): int(v) for i, v in enumerate(np.bincount(np.asarray(enter_labels, dtype=np.int64), minlength=2))},
        "tp_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 0], minlength=len(TP_BUCKETS)))},
        "sl_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 1], minlength=len(SL_BUCKETS)))},
        "notional_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 2], minlength=len(NOTIONAL_BUCKETS)))},
        "leverage_counts": {str(i): int(v) for i, v in enumerate(np.bincount(y_np[:, 3], minlength=len(LEVERAGE_BUCKETS)))},
        "best_exit_reasons": reasons,
    }


def _make_model(kind: str, seed: int) -> Any:
    if kind == "hgb":
        return HistGradientBoostingClassifier(max_iter=180, learning_rate=0.04, max_leaf_nodes=7, l2_regularization=1.0, min_samples_leaf=35, random_state=int(seed))
    if kind == "extratrees":
        return ExtraTreesClassifier(n_estimators=240, max_depth=8, min_samples_leaf=20, random_state=int(seed), n_jobs=-1)
    raise RuntimeError(f"unknown selector kind: {kind}")


def _train_hgb_heads(x: np.ndarray, y: np.ndarray, y_enter: np.ndarray, w: np.ndarray, *, kind: str, seed: int, entry_gate: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    names = ["tp", "sl", "notional", "leverage"]
    models: dict[str, Any] = {}
    diag: dict[str, Any] = {}
    if bool(entry_gate):
        gate = _make_model(kind, int(seed) + 100)
        gate.fit(x, y_enter, sample_weight=w)
        gate_pred = np.asarray(gate.predict(x), dtype=np.int64).reshape(-1)
        models["enter_gate"] = gate
        diag["enter_gate_train_acc"] = float(np.mean(gate_pred == y_enter))
    for j, name in enumerate(names):
        model = _make_model(kind, int(seed) + j)
        model.fit(x, y[:, j], sample_weight=w)
        pred = np.asarray(model.predict(x), dtype=np.int64).reshape(-1)
        models[name] = model
        diag[f"{name}_train_acc"] = float(np.mean(pred == y[:, j]))
    return models, {"kind": kind, **diag}


def _predict_hgb_enter(models: dict[str, Any], x: pd.DataFrame, norm: dict[str, Any], *, entry_gate: bool) -> bool:
    if not bool(entry_gate):
        return True
    if "enter_gate" not in models:
        raise RuntimeError("entry_gate enabled but model missing enter_gate")
    xn = _apply_norm(x, norm)
    return bool(int(np.asarray(models["enter_gate"].predict(xn), dtype=np.int64).reshape(-1)[0]) == 1)


def _predict_hgb_ids(
    models: dict[str, Any],
    x: pd.DataFrame,
    norm: dict[str, Any],
    *,
    max_safe_notional_id: int | None = None,
    boost_min_prob: float = 0.0,
    boost_min_margin: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    xn = _apply_norm(x, norm)
    ids = np.column_stack([
        np.asarray(models["tp"].predict(xn), dtype=np.int64).reshape(-1),
        np.asarray(models["sl"].predict(xn), dtype=np.int64).reshape(-1),
        np.asarray(models["notional"].predict(xn), dtype=np.int64).reshape(-1),
        np.asarray(models["leverage"].predict(xn), dtype=np.int64).reshape(-1),
    ])
    meta: dict[str, Any] = {"notional_clamped": 0, "notional_conf": None, "notional_margin": None}
    if max_safe_notional_id is None or int(max_safe_notional_id) < 0 or "notional" not in models:
        return ids, meta
    model = models["notional"]
    if not hasattr(model, "predict_proba") or not hasattr(model, "classes_"):
        return ids, meta
    probs = np.asarray(model.predict_proba(xn), dtype=np.float64)
    classes = np.asarray(model.classes_, dtype=np.int64)
    safe = classes <= int(max_safe_notional_id)
    if not bool(np.any(safe)):
        raise RuntimeError("max_safe_notional_id excludes every notional class")
    for i in range(ids.shape[0]):
        row_prob = probs[i]
        top = np.sort(row_prob)[::-1]
        conf = float(top[0]) if len(top) else 0.0
        margin = float(top[0] - top[1]) if len(top) > 1 else conf
        meta["notional_conf"] = conf
        meta["notional_margin"] = margin
        if int(ids[i, 2]) <= int(max_safe_notional_id):
            continue
        if conf >= float(boost_min_prob) and margin >= float(boost_min_margin):
            continue
        safe_probs = np.where(safe, row_prob, -np.inf)
        ids[i, 2] = int(classes[int(np.argmax(safe_probs))])
        meta["notional_clamped"] = int(meta["notional_clamped"]) + 1
    return ids, meta


def _apply_vol_leverage_cap(
    ids: tuple[int, int, int, int] | np.ndarray,
    atr_pct: float,
    *,
    enabled: bool,
    high: float,
    medium: float,
) -> tuple[np.ndarray, str | None]:
    out = np.asarray(ids, dtype=np.int64).copy()
    if not bool(enabled):
        return out, None
    atr = float(atr_pct)
    if atr >= float(high):
        max_lev_id = 0
        reason = "vol_cap_high"
    elif atr >= float(medium):
        max_lev_id = min(1, len(LEVERAGE_BUCKETS) - 1)
        reason = "vol_cap_medium"
    else:
        return out, None
    if int(out[3]) > int(max_lev_id):
        out[3] = int(max_lev_id)
        return out, reason
    return out, None


def _train_mamba_adapter(x_seq: np.ndarray, q: np.ndarray, best: np.ndarray, w: np.ndarray, *, device: torch.device, steps: int, batch_size: int, lr: float) -> tuple[MambaBucketAdapter, dict[str, Any]]:
    model = MambaBucketAdapter(x_seq.shape[-1], len(ENTRY_BUCKETS)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=2e-5)
    ds = TensorDataset(torch.from_numpy(x_seq), torch.from_numpy(q), torch.from_numpy(best), torch.from_numpy(w))
    dl = DataLoader(ds, batch_size=min(int(batch_size), max(len(ds), 1)), shuffle=True, drop_last=False)
    last: dict[str, Any] = {}
    for step in range(1, int(steps) + 1):
        for seq_b, q_b, a_b, w_b in dl:
            seq_b, q_b, a_b, w_b = seq_b.to(device), q_b.to(device), a_b.to(device), w_b.to(device)
            logits, q_pred = model(seq_b)
            critic_loss = torch.nn.functional.smooth_l1_loss(q_pred, q_b)
            ce = torch.nn.functional.cross_entropy(logits, a_b, reduction="none")
            actor_loss = (ce * w_b).sum() / torch.clamp(w_b.sum(), min=1.0)
            probs = torch.softmax(logits, dim=1)
            policy_q = (probs * q_pred.detach()).sum(dim=1).mean()
            loss = critic_loss + actor_loss - 0.20 * policy_q
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            opt.step()
        if step % 100 == 0:
            last = {"step": int(step), "critic_loss": float(critic_loss.detach().cpu()), "actor_loss": float(actor_loss.detach().cpu()), "policy_q": float(policy_q.detach().cpu())}
    return model.cpu(), last


@torch.no_grad()
def _predict_mamba_id(model: MambaBucketAdapter, seq: np.ndarray, *, device: torch.device) -> np.ndarray:
    model = model.to(device)
    model.eval()
    logits, q = model(torch.from_numpy(seq[None, :, :]).to(device))
    score = torch.softmax(logits, dim=1) * torch.clamp(q, min=-0.05, max=0.05).add(0.05)
    flat = int(torch.argmax(score, dim=1).detach().cpu().item())
    return ENTRY_BUCKETS[flat]


def _replay_post_adapter(
    frames: dict[str, Any],
    split: str,
    lifecycle_model: lifecycle.MambaDiscreteActorCritic,
    lifecycle_ckpt: dict[str, Any],
    adapter: Any,
    adapter_norm: dict[str, Any],
    *,
    adapter_kind: str,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
    select_mode: str,
    replay_enter_topk: int,
    entry_gate: bool,
    max_safe_notional_id: int | None,
    boost_min_prob: float,
    boost_min_margin: float,
    vol_target_leverage: bool,
    vol_target_high_atr: float,
    vol_target_medium_atr: float,
    tp_upshift: bool,
    tp_upshift_mult: float,
    tp_upshift_max: int,
) -> dict[str, Any]:
    frame = frames[f"{split}_df"]
    state = lifecycle._base_state(frames[f"s_{split}"])
    dec = frames[f"{split}_dec"]
    arrays = _arrays(frame)
    atr_arr = _atr_pct(frame)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    base_norm = lifecycle._apply_norm(state, lifecycle_ckpt["normalizer"])
    base_seq = lifecycle._rolling_sequences(base_norm, int(lifecycle_ckpt["seq_len"]))
    cash = peak = 1.0
    mdd = 0.0
    pos = Position()
    lifecycle_pos = lifecycle.Position()
    trades = wins = long_entries = short_entries = partials = 0
    reasons: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    active = omega._active(dec)
    for i in range(0, len(frame) - 2):
        if pos.side != 0:
            vals = _position_values(arrays, pos, i, slip_eff=slip_eff)
            pos.mfe = max(pos.mfe, vals["lc_pos_unrealized"])
            pos.mae = min(pos.mae, vals["lc_pos_unrealized"])
            eq = cash * (1.0 + vals["lc_pos_unrealized"])
            if pos.stop_loss > 0.0 and vals["lc_pos_unrealized"] <= -pos.stop_loss:
                before = cash
                cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                lifecycle_pos = lifecycle.Position()
                trades += 1
                wins += int(cash > before)
                reasons["stop_loss"] = reasons.get("stop_loss", 0) + 1
                continue
            if pos.take_profit > 0.0 and vals["lc_pos_unrealized"] >= pos.take_profit:
                if bool(tp_upshift) and int(pos.tp_upshifts) < int(tp_upshift_max):
                    pos.take_profit = float(pos.take_profit) * float(tp_upshift_mult)
                    pos.tp_upshifts += 1
                    lifecycle_pos = _to_lifecycle_position(pos)
                    reasons["tp_upshift"] = reasons.get("tp_upshift", 0) + 1
                else:
                    before = cash
                    cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
                    lifecycle_pos = lifecycle.Position()
                    trades += 1
                    wins += int(cash > before)
                    reasons["take_profit"] = reasons.get("take_profit", 0) + 1
                continue
        else:
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos.side == 0 and not bool(active[i]):
            continue
        lc_row = lifecycle._state_row(state, arrays, lifecycle_pos, i, slip_eff=slip_eff)
        allowed = lifecycle._allowed_actions(arrays, dec, lifecycle_pos, i, slip_eff=slip_eff, disable_resize=True, disable_reverse=True)
        scores = _lifecycle_scores(lifecycle_model, lifecycle_ckpt, base_seq, lc_row, allowed, i, device=device, select_mode=select_mode)
        lc_action = int(np.argmax(scores))
        if lifecycle_pos.side == 0 and lc_action not in (lifecycle.ENTER_BASE, lifecycle.ENTER_AGGRESSIVE):
            top_actions = np.argsort(scores)[::-1][: max(int(replay_enter_topk), 1)]
            enter_scores = [(lifecycle.ENTER_BASE, float(scores[lifecycle.ENTER_BASE])), (lifecycle.ENTER_AGGRESSIVE, float(scores[lifecycle.ENTER_AGGRESSIVE]))]
            enter_scores = [(a, s) for a, s in enter_scores if np.isfinite(s) and s > -1e8 and int(a) in set(int(x) for x in top_actions)]
            if not enter_scores:
                reasons["skip"] = reasons.get("skip", 0) + 1
                continue
            lc_action = int(max(enter_scores, key=lambda x: x[1])[0])
            reasons["topk_enter_candidate"] = reasons.get("topk_enter_candidate", 0) + 1
        if lifecycle_pos.side == 0:
            feat = _adapter_feature_row(lc_row, lc_action)
            if str(adapter_kind) == "mamba":
                # Strict post-lifecycle adapter only observes sparse enter events, not every bar.
                # Use the current enter context as a short repeated sequence instead of adding
                # implicit compatibility columns or fabricating unavailable historical actions.
                cur = _apply_norm(feat, adapter_norm)
                seq = np.repeat(cur, int(lifecycle_ckpt["seq_len"]), axis=0).astype(np.float32)
                ids = _predict_mamba_id(adapter, seq, device=device)
            else:
                if not _predict_hgb_enter(adapter, feat, adapter_norm, entry_gate=bool(entry_gate)):
                    reasons["adapter_veto"] = reasons.get("adapter_veto", 0) + 1
                    continue
                ids_arr, pred_meta = _predict_hgb_ids(
                    adapter,
                    feat,
                    adapter_norm,
                    max_safe_notional_id=max_safe_notional_id,
                    boost_min_prob=float(boost_min_prob),
                    boost_min_margin=float(boost_min_margin),
                )
                if int(pred_meta.get("notional_clamped", 0)):
                    reasons["notional_conf_clamp"] = reasons.get("notional_conf_clamp", 0) + int(pred_meta["notional_clamped"])
                ids = ids_arr[0]
            ids, vol_reason = _apply_vol_leverage_cap(
                ids,
                float(atr_arr[int(i)]),
                enabled=bool(vol_target_leverage),
                high=float(vol_target_high_atr),
                medium=float(vol_target_medium_atr),
            )
            if vol_reason:
                reasons[vol_reason] = reasons.get(vol_reason, 0) + 1
            before = cash
            cash, pos, reason = _enter_with_risk(cash, arrays, dec, i, ids, fee_eff=fee_eff, slip_eff=slip_eff, atr_pct=float(atr_arr[int(i)]))
            lifecycle_pos = _to_lifecycle_position(pos)
            reasons[reason] = reasons.get(reason, 0) + 1
            if reason == "entry":
                long_entries += int(pos.side > 0)
                short_entries += int(pos.side < 0)
                bucket_counts[str(tuple(int(x) for x in ids))] = bucket_counts.get(str(tuple(int(x) for x in ids)), 0) + 1
            continue
        if lc_action == lifecycle.FULL_EXIT:
            before = cash
            cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = lifecycle.Position()
            trades += 1
            wins += int(cash > before)
            reasons["full_exit"] = reasons.get("full_exit", 0) + 1
        elif lc_action == lifecycle.REDUCE50:
            cash, pos, _ = _realize_fraction(cash, arrays, pos, i, 0.5, fee_eff=fee_eff, slip_eff=slip_eff)
            lifecycle_pos = _to_lifecycle_position(pos)
            partials += 1
            reasons["reduce50"] = reasons.get("reduce50", 0) + 1
        else:
            reasons["hold"] = reasons.get("hold", 0) + 1
    if pos.side != 0:
        before = cash
        cash, pos, _ = _realize_fraction(cash, arrays, pos, len(frame) - 1, 1.0, fee_eff=fee_eff, slip_eff=slip_eff)
        trades += 1
        wins += int(cash > before)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / trades) if trades else 0.0, "long_entries": int(long_entries), "short_entries": int(short_entries), "partials": int(partials), "reasons": reasons, "top_buckets": dict(sorted(bucket_counts.items(), key=lambda x: x[1], reverse=True)[:10])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threehead-dir", type=Path, default=feat_coord.DEFAULT_3HEAD_DIR)
    ap.add_argument("--baseline-lifecycle-dir", type=Path, default=BASELINE_LIFECYCLE_DIR)
    ap.add_argument("--adapter-kind", choices=["hgb", "extratrees", "mamba"], default="hgb")
    ap.add_argument("--bucket-preset", choices=["base", "aggressive", "side_asym", "atr_aggressive", "atr_wide", "fixed_wide", "fixed_ultra_wide", "base_expanded_notional"], default="base")
    ap.add_argument("--notional-mult", type=float, default=1.0)
    ap.add_argument("--notional-cap", type=float, default=0.0)
    ap.add_argument("--max-leverage", choices=["3", "5"], default="3")
    ap.add_argument("--use-leverage-exposure", action="store_true")
    ap.add_argument("--compensate-sltp-by-notional", action="store_true")
    ap.add_argument("--compensate-ref-notional", type=float, default=0.45)
    ap.add_argument("--quality-threshold", type=float, default=0.75)
    ap.add_argument("--max-label-rows", type=int, default=700)
    ap.add_argument("--candidates-per-row", type=int, default=64)
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--enter-topk", type=int, default=1)
    ap.add_argument("--replay-enter-topk", type=int, default=1)
    ap.add_argument("--entry-gate", action="store_true")
    ap.add_argument("--max-safe-notional-id", type=int, default=-1)
    ap.add_argument("--boost-min-prob", type=float, default=0.0)
    ap.add_argument("--boost-min-margin", type=float, default=0.0)
    ap.add_argument("--vol-target-leverage", action="store_true")
    ap.add_argument("--vol-target-high-atr", type=float, default=0.008)
    ap.add_argument("--vol-target-medium-atr", type=float, default=0.005)
    ap.add_argument("--tp-upshift", action="store_true")
    ap.add_argument("--tp-upshift-mult", type=float, default=1.35)
    ap.add_argument("--tp-upshift-max", type=int, default=1)
    ap.add_argument("--exit-aware-label", action="store_true")
    ap.add_argument("--train-max-sim-bars", type=int, default=96)
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--select-mode", choices=["actor_q", "q_only"], default="actor_q")
    ap.add_argument("--cost-mult", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=260670)
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = ap.parse_args()
    if bool(args.entry_gate) and str(args.adapter_kind) == "mamba":
        raise RuntimeError("entry_gate currently supports hgb/extratrees adapters only")
    _set_bucket_preset(str(args.bucket_preset))
    global NOTIONAL_MULT, NOTIONAL_CAP, LEVERAGE_BUCKETS, ENTRY_BUCKETS, USE_LEVERAGE_EXPOSURE, COMPENSATE_SLTP_BY_NOTIONAL, COMPENSATE_REF_NOTIONAL
    NOTIONAL_MULT = float(args.notional_mult)
    if float(args.notional_cap) > 0.0:
        NOTIONAL_CAP = float(args.notional_cap)
    if str(args.max_leverage) == "5":
        LEVERAGE_BUCKETS = np.asarray([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
        ENTRY_BUCKETS = np.asarray(
            [(t, s, n, l) for t in range(len(TP_BUCKETS)) for s in range(len(SL_BUCKETS)) for n in range(len(NOTIONAL_BUCKETS)) for l in range(len(LEVERAGE_BUCKETS))],
            dtype=np.int64,
        )
    USE_LEVERAGE_EXPOSURE = bool(args.use_leverage_exposure)
    COMPENSATE_SLTP_BY_NOTIONAL = bool(args.compensate_sltp_by_notional)
    COMPENSATE_REF_NOTIONAL = float(args.compensate_ref_notional)
    _seed_everything(int(args.seed))
    device = _device(str(args.device))
    out_dir = OUT_DIR if not str(args.out_suffix).strip() else OUT_DIR.parent / f"{MODEL_ID}_{str(args.out_suffix).strip()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = _base_frames(Path(args.threehead_dir), float(args.quality_threshold), device)
    lifecycle_model, lifecycle_ckpt = _load_baseline_lifecycle(Path(args.baseline_lifecycle_dir))
    bad = [c for c in lifecycle_ckpt["state_columns"] if "clean_regime4" in c or "regime4_pred" in c or "tp_sl_action_score" in c or str(c).startswith("teacher_")]
    if bad:
        raise RuntimeError(f"forbidden lifecycle state columns passed audit: {bad[:20]}")
    fee, slip = omega._load_fee_slip()
    x_train, entry_idx, _lc_actions, collect_diag = _collect_train_entries(frames, lifecycle_model, lifecycle_ckpt, device=device, select_mode=str(args.select_mode), fee=fee, slip=slip, cost_mult=float(args.cost_mult), max_rows=int(args.max_label_rows), enter_topk=int(args.enter_topk))
    y, q, best, y_enter, w, label_diag = _build_labels(frames["train_df"], frames["train_dec"], entry_idx, seed=int(args.seed), candidates_per_row=int(args.candidates_per_row), fee=fee, slip=slip, cost_mult=float(args.cost_mult), max_bars=int(args.train_max_sim_bars), min_score=float(args.min_score), exit_aware_label=bool(args.exit_aware_label))
    if str(args.adapter_kind) == "mamba":
        x_norm_arr, adapter_norm = _fit_norm(x_train)
        seq = lifecycle._rolling_sequences(x_norm_arr, int(lifecycle_ckpt["seq_len"]))
        adapter, train_diag = _train_mamba_adapter(seq, q, best, w, device=device, steps=int(args.steps), batch_size=int(args.batch_size), lr=float(args.lr))
        torch.save({"model_state_dict": adapter.state_dict(), "normalizer": adapter_norm, "entry_buckets": ENTRY_BUCKETS, "state_columns": adapter_norm["columns"], "seq_len": int(lifecycle_ckpt["seq_len"])}, out_dir / "post_mamba_bucket_adapter.pt")
        adapter_artifact = str(out_dir / "post_mamba_bucket_adapter.pt")
    else:
        x_arr, adapter_norm = _fit_norm(x_train)
        adapter, train_diag = _train_hgb_heads(x_arr, y, y_enter, w, kind=str(args.adapter_kind), seed=int(args.seed), entry_gate=bool(args.entry_gate))
        with (out_dir / "post_bucket_adapter.pkl").open("wb") as f:
            pickle.dump({"models": adapter, "normalizer": adapter_norm, "entry_buckets": ENTRY_BUCKETS, "tp_buckets": TP_BUCKETS, "sl_buckets": SL_BUCKETS, "notional_buckets": NOTIONAL_BUCKETS, "leverage_buckets": LEVERAGE_BUCKETS}, f)
        adapter_artifact = str(out_dir / "post_bucket_adapter.pkl")
    replay_kwargs = {
        "adapter_kind": str(args.adapter_kind),
        "fee": fee,
        "slip": slip,
        "cost_mult": float(args.cost_mult),
        "device": device,
        "select_mode": str(args.select_mode),
        "replay_enter_topk": int(args.replay_enter_topk),
        "entry_gate": bool(args.entry_gate),
        "max_safe_notional_id": None if int(args.max_safe_notional_id) < 0 else int(args.max_safe_notional_id),
        "boost_min_prob": float(args.boost_min_prob),
        "boost_min_margin": float(args.boost_min_margin),
        "vol_target_leverage": bool(args.vol_target_leverage),
        "vol_target_high_atr": float(args.vol_target_high_atr),
        "vol_target_medium_atr": float(args.vol_target_medium_atr),
        "tp_upshift": bool(args.tp_upshift),
        "tp_upshift_mult": float(args.tp_upshift_mult),
        "tp_upshift_max": int(args.tp_upshift_max),
    }
    val = _replay_post_adapter(frames, "val", lifecycle_model, lifecycle_ckpt, adapter, adapter_norm, **replay_kwargs)
    oos = _replay_post_adapter(frames, "oos", lifecycle_model, lifecycle_ckpt, adapter, adapter_norm, **replay_kwargs)
    report = {
        "model_id": MODEL_ID,
        "design": "Post-Lifecycle Risk Adapter. Frozen Mamba first selects enter/hold/exit; optional top-k enter replay exposes near-enter candidates, then this adapter can veto/enter and choose TP/SL/notional/leverage bucket ids.",
        "adapter_kind": str(args.adapter_kind),
        "bucket_preset": str(args.bucket_preset),
        "atr_risk": bool(USE_ATR_RISK),
        "notional_mult": float(args.notional_mult),
        "notional_cap": float(NOTIONAL_CAP),
        "accounting_note": "If use_leverage_exposure=false, replay uses notional_exposure as effective account exposure and leverage is metadata. If true, effective exposure is clipped(notional_bucket * leverage_bucket, notional_cap), and fees/PnL/TP/SL checks use that effective exposure.",
        "quality_threshold": float(args.quality_threshold),
        "use_leverage_exposure": bool(args.use_leverage_exposure),
        "compensate_sltp_by_notional": bool(args.compensate_sltp_by_notional),
        "compensate_ref_notional": float(args.compensate_ref_notional),
        "confidence_notional_clamp": {
            "max_safe_notional_id": None if int(args.max_safe_notional_id) < 0 else int(args.max_safe_notional_id),
            "boost_min_prob": float(args.boost_min_prob),
            "boost_min_margin": float(args.boost_min_margin),
        },
        "vol_target_leverage": {
            "enabled": bool(args.vol_target_leverage),
            "high_atr": float(args.vol_target_high_atr),
            "medium_atr": float(args.vol_target_medium_atr),
        },
        "tp_upshift": {
            "enabled": bool(args.tp_upshift),
            "mult": float(args.tp_upshift_mult),
            "max": int(args.tp_upshift_max),
        },
        "bucket_space": {"tp": TP_BUCKETS.tolist(), "sl": SL_BUCKETS.tolist(), "notional": NOTIONAL_BUCKETS.tolist(), "leverage": LEVERAGE_BUCKETS.tolist(), "n_entry_combos": int(len(ENTRY_BUCKETS))},
        "training": {"collect_diag": collect_diag, "label_diag": label_diag, "train_diag": train_diag, "steps": int(args.steps), "min_score": float(args.min_score), "enter_topk": int(args.enter_topk), "replay_enter_topk": int(args.replay_enter_topk), "entry_gate": bool(args.entry_gate), "exit_aware_label": bool(args.exit_aware_label)},
        "results": {"validation": val, "oos": oos},
        "artifacts": {"out_dir": str(out_dir), "report": str(out_dir / "report.json"), "adapter": adapter_artifact},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(out_dir / "report.json"), "results": report["results"]}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
