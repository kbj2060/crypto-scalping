#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_3head_20260603 as th  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402


ARTIFACT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_true_3head_tabm_20260603_final_tp_sl_on_e28_exit30k_q080"
OUT = ARTIFACT_DIR / "heartbeat_overlay_tests_20260604.csv"

THR_MAP = {"bull": 0.72, "bear": 0.64, "chop": 0.65}
SCALE_MAP = {"bull": 0.65, "bear": 0.90, "chop": 0.90}
BASE_SCALES = {"bull": 0.75, "bear": 0.90, "chop_expert": 0.90, "chop": 0.90}
TP = 0.026
SL = 0.014


@dataclass(frozen=True)
class OverlayCfg:
    name: str
    exit_veto_thr: float | None = None
    entry_veto_thr: float | None = None
    defense: str = "none"
    short_penalty_after: int = 0
    short_penalty: float = 0.0
    vol_scale: str = "none"
    cash_sleeve: str = "none"


def _align(frame: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    out = frame[["timestamp"]].merge(pred, on="timestamp", how="left", validate="one_to_one")
    if out.isna().any().any():
        raise RuntimeError("prediction alignment produced NaN")
    return out


def _set_expert_thresholds(src: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = src.copy()
    q = pd.to_numeric(out[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64)
    action = pd.to_numeric(out[f"{prefix}dir_action"], errors="raise").to_numpy(dtype=np.int64)
    expert = out[f"{prefix}router_expert"].astype(str).to_numpy()
    thr = np.asarray([THR_MAP.get(x, THR_MAP["chop"]) for x in expert], dtype=np.float64)
    out[f"{prefix}quality_threshold"] = thr
    out[f"{prefix}final_action"] = np.where(q >= thr, action, omega.ACTION_CASH).astype(np.int64)
    return out


def _build_dec(src: pd.DataFrame, prefix: str, *, oof: bool) -> pd.DataFrame:
    dec = omega._to_fixed_decisions(_set_expert_thresholds(src, prefix), oof=oof)
    active = omega._active(dec)
    for expert, scale in SCALE_MAP.items():
        key = "chop_expert" if expert == "chop" else expert
        mask = active & dec["router_expert"].astype(str).eq(key)
        ratio = float(scale) / float(BASE_SCALES[key])
        dec.loc[mask, "notional_exposure"] = pd.to_numeric(dec.loc[mask, "notional_exposure"], errors="raise") * ratio
        dec.loc[mask, "position_fraction"] = pd.to_numeric(dec.loc[mask, "position_fraction"], errors="raise") * ratio
    active = omega._active(dec)
    dec.loc[active, "take_profit"] = TP
    dec.loc[active, "stop_loss"] = SL
    dec.loc[active, "max_hold_bars"] = 0
    dec.loc[active, "cooldown_bars"] = 0
    return dec


def _atr_ratio(frame: pd.DataFrame) -> np.ndarray:
    high = pd.to_numeric(frame["high"], errors="raise")
    low = pd.to_numeric(frame["low"], errors="raise")
    close = pd.to_numeric(frame["close"], errors="raise")
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(span=14, adjust=False).mean()
    ratio = (atr / close).replace([np.inf, -np.inf], np.nan)
    base = ratio.rolling(288, min_periods=50).median()
    out = (ratio / base).replace([np.inf, -np.inf], np.nan).fillna(1.0).to_numpy(dtype=np.float64)
    return np.clip(out, 0.2, 5.0)


def _cash_sleeve_decision(frame: pd.DataFrame, i: int, mode: str) -> tuple[int, float, float, float] | None:
    if mode == "none" or i < 30:
        return None
    close = pd.to_numeric(frame["close"], errors="raise")
    window = close.iloc[max(0, i - 40) : i + 1]
    mean = float(window.mean())
    std = float(window.std(ddof=0))
    if not np.isfinite(std) or std <= 0:
        return None
    z = (float(close.iloc[i]) - mean) / std
    if mode == "mr_tiny":
        if z <= -2.2:
            return 1, 0.006, 0.006, 0.10
        if z >= 2.2:
            return -1, 0.006, 0.006, 0.10
    if mode == "mr_small":
        if z <= -2.5:
            return 1, 0.008, 0.006, 0.12
        if z >= 2.5:
            return -1, 0.008, 0.006, 0.12
    return None


def _pos_features(
    base_x: pd.DataFrame,
    i: int,
    *,
    pos: int,
    hold: int,
    unreal: float,
    mfe: float,
    mae: float,
    notional: float,
    leverage: float,
    tp: float,
    sl: float,
) -> pd.DataFrame:
    xrow = base_x.iloc[[int(i)]].copy().reset_index(drop=True)
    giveback = (mfe - unreal) / max(abs(mfe), 1e-8) if mfe > 0 else 0.0
    vals = {
        "pos_side": float(pos),
        "pos_hold_bars": float(hold),
        "pos_unrealized": float(unreal),
        "pos_mfe": float(mfe),
        "pos_mae": float(mae),
        "pos_giveback": float(np.clip(giveback, 0.0, 10.0)),
        "pos_dist_to_tp": float(tp - unreal),
        "pos_dist_to_sl": float(unreal + abs(sl)),
        "pos_notional": float(notional),
        "pos_leverage": float(leverage),
        "pos_exposure": float(notional * leverage),
        "pos_tp": float(tp),
        "pos_sl": float(sl),
    }
    for col, val in vals.items():
        xrow[col] = val
    return xrow


@torch.no_grad()
def _exit_prob(
    loaded_models: dict[str, tuple[th.ThreeHeadTabM, dict[str, Any]]],
    expert: str,
    xrow: pd.DataFrame,
    *,
    device: torch.device,
) -> float:
    key = expert if expert in loaded_models else "chop"
    model, scaler = loaded_models[key]
    return float(th._predict_loaded_exit(model, scaler, xrow, device=device)[0, 1])


def _metrics_overlay(
    frame: pd.DataFrame,
    src: pd.DataFrame,
    dec: pd.DataFrame,
    base_x: pd.DataFrame,
    loaded_models: dict[str, tuple[th.ThreeHeadTabM, dict[str, Any]]],
    prefix: str,
    cfg: OverlayCfg,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    device: torch.device,
) -> dict[str, Any]:
    arrays = {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}
    active = omega._active(dec)
    fee_eff = float(fee) * float(cost_mult)
    slip_eff = float(slip) * float(cost_mult)
    vol = _atr_ratio(frame)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    pos = 0
    entry_price = 0.0
    entry_equity = 1.0
    entry_i = 0
    notional = 0.0
    leverage = 1.0
    tp = 0.0
    sl = 0.0
    mfe = 0.0
    mae = 0.0
    trades = wins = long_entries = short_entries = 0
    short_streak = 0
    reasons: dict[str, int] = {}
    for i in range(0, len(frame) - 2):
        if pos != 0:
            px = float(arrays["close"][i])
            raw = (px * (1.0 - slip_eff) - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - px * (1.0 + slip_eff)) / max(entry_price, 1e-12)
            unreal = raw * notional
            mfe = max(mfe, unreal)
            mae = min(mae, unreal)
            eq = cash * (1.0 + unreal)
        else:
            unreal = 0.0
            eq = cash
        peak = max(peak, eq)
        mdd = min(mdd, eq / max(peak, 1e-12) - 1.0)
        if pos != 0:
            hold = int(i) - int(entry_i)
            expert = str(dec.iloc[int(entry_i)].get("router_expert", "chop")).replace("chop_expert", "chop")
            if cfg.exit_veto_thr is not None:
                p_exit = _exit_prob(
                    loaded_models,
                    expert,
                    _pos_features(base_x, i, pos=pos, hold=hold, unreal=unreal, mfe=mfe, mae=mae, notional=notional, leverage=leverage, tp=tp, sl=sl),
                    device=device,
                )
                if p_exit >= float(cfg.exit_veto_thr):
                    if cfg.defense == "half_tp":
                        tp = min(tp, max(0.002, TP * 0.5))
                        reasons["exit_head_half_tp"] = reasons.get("exit_head_half_tp", 0) + 1
                    elif cfg.defense == "breakeven_stop" and unreal > 0:
                        sl = 0.0
                        reasons["exit_head_breakeven"] = reasons.get("exit_head_breakeven", 0) + 1
            reason = ""
            if tp > 0.0 and unreal >= tp:
                reason = "take_profit"
            elif sl >= 0.0 and unreal <= -abs(sl):
                reason = "stop_loss"
            if reason:
                filled, exit_px, exit_fee, _route = omega._try_execution(arrays, int(i), pos, entry=False, fee_base=fee_eff, slip_base=slip_eff)
                if not filled:
                    continue
                raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw_exit * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                reasons[reason] = reasons.get(reason, 0) + 1
                pos = 0
                continue
        if pos != 0:
            continue
        if not bool(active[i]):
            sleeve = _cash_sleeve_decision(frame, i, cfg.cash_sleeve)
            if sleeve is None:
                continue
            side, s_tp, s_sl, s_notional = sleeve
            row = None
        else:
            row = dec.iloc[i]
            side = int(row.get("side", 0) or 0)
            if side == 0:
                continue
            if cfg.short_penalty_after > 0 and side < 0 and short_streak >= int(cfg.short_penalty_after):
                q = float(src.iloc[i][f"{prefix}quality_for_action"])
                expert = str(src.iloc[i][f"{prefix}router_expert"])
                if q < THR_MAP.get(expert, THR_MAP["chop"]) + float(cfg.short_penalty):
                    reasons["short_penalty_veto"] = reasons.get("short_penalty_veto", 0) + 1
                    continue
            if cfg.entry_veto_thr is not None:
                expert = str(row.get("router_expert", "chop")).replace("chop_expert", "chop")
                p_exit = _exit_prob(
                    loaded_models,
                    expert,
                    _pos_features(base_x, i, pos=0, hold=0, unreal=0.0, mfe=0.0, mae=0.0, notional=0.0, leverage=1.0, tp=0.0, sl=0.0),
                    device=device,
                )
                if p_exit >= float(cfg.entry_veto_thr):
                    reasons["entry_exit_veto"] = reasons.get("entry_exit_veto", 0) + 1
                    continue
            s_tp = float(row.get("take_profit", TP) or TP)
            s_sl = float(row.get("stop_loss", SL) or SL)
            s_notional = float(row.get("notional_exposure", 0.0) or 0.0)
            if cfg.vol_scale == "atr_0p8_1p2":
                mult = 0.8 if vol[i] < 0.8 else 1.2 if vol[i] > 1.2 else 1.0
                s_tp *= mult
                s_sl *= mult
        if side == 0 or s_notional <= 0.0:
            continue
        filled, px, entry_fee, _route = omega._try_execution(arrays, int(i), side, entry=True, fee_base=fee_eff, slip_base=slip_eff)
        if not filled:
            continue
        pos = int(side)
        entry_price = float(px)
        entry_equity = cash
        entry_i = min(int(i) + 1, len(frame) - 1)
        notional = float(s_notional)
        leverage = 2.0
        tp = float(s_tp)
        sl = abs(float(s_sl))
        cash -= cash * entry_fee * notional
        long_entries += int(pos > 0)
        short_entries += int(pos < 0)
        short_streak = short_streak + 1 if pos < 0 else 0
        mfe = mae = 0.0
        reasons["entry" if row is not None else "cash_sleeve_entry"] = reasons.get("entry" if row is not None else "cash_sleeve_entry", 0) + 1
    if pos != 0:
        exit_px = omega._fill_price(arrays, len(frame) - 1, pos, slip_eff, entry=False)
        raw_exit = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw_exit * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        reasons["forced_end"] = reasons.get("forced_end", 0) + 1
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "wr": float(wins / trades) if trades else 0.0,
        "trades": int(trades),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "reasons": reasons,
    }


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = th._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    bundle = torch.load(ARTIFACT_DIR / "true_3head_tabm_bundle.pt", map_location="cpu", weights_only=False)
    loaded = th._load_payloads(bundle["models"], device=device)
    val_raw = frames["val_raw"].reset_index(drop=True)
    oos_raw = frames["oos_raw"].reset_index(drop=True)
    base_cols = list(bundle["base_cols"])
    x_val = th._base_input(val_raw, base_cols)
    x_oos = th._base_input(oos_raw, base_cols)
    val_src = _align(val_raw, pd.read_csv(ARTIFACT_DIR / "validation_predictions_2025_true3head.csv", parse_dates=["timestamp"]))
    oos_src = _align(oos_raw, pd.read_csv(ARTIFACT_DIR / "oos_predictions_2026_true3head.csv", parse_dates=["timestamp"]))
    val_dec = _build_dec(val_src, "omega1_regime3_expertdq_oof_", oof=True)
    oos_dec = _build_dec(oos_src, "omega1_regime3_expertdq_", oof=False)
    configs = [
        OverlayCfg("baseline"),
        OverlayCfg("entry_veto_0p95", entry_veto_thr=0.95),
        OverlayCfg("entry_veto_0p98", entry_veto_thr=0.98),
        OverlayCfg("exit_half_tp_0p95", exit_veto_thr=0.95, defense="half_tp"),
        OverlayCfg("exit_breakeven_0p95", exit_veto_thr=0.95, defense="breakeven_stop"),
        OverlayCfg("short_penalty_3_0p05", short_penalty_after=3, short_penalty=0.05),
        OverlayCfg("vol_scale_atr_0p8_1p2", vol_scale="atr_0p8_1p2"),
        OverlayCfg("cash_sleeve_mr_tiny", cash_sleeve="mr_tiny"),
        OverlayCfg("cash_sleeve_mr_small", cash_sleeve="mr_small"),
    ]
    rows: list[dict[str, Any]] = []
    for cfg in configs:
        val = _metrics_overlay(
            val_raw,
            val_src,
            val_dec,
            x_val,
            loaded,
            "omega1_regime3_expertdq_oof_",
            cfg,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            device=device,
        )
        oos = _metrics_overlay(
            oos_raw,
            oos_src,
            oos_dec,
            x_oos,
            loaded,
            "omega1_regime3_expertdq_",
            cfg,
            fee=fee,
            slip=slip,
            cost_mult=3.0,
            device=device,
        )
        rows.append(
            {
                "variant": cfg.name,
                "val_pnl": val["pnl"],
                "val_mdd": val["mdd"],
                "val_wr": val["wr"],
                "val_trades": val["trades"],
                "val_long": val["long_entries"],
                "val_short": val["short_entries"],
                "oos_pnl": oos["pnl"],
                "oos_mdd": oos["mdd"],
                "oos_wr": oos["wr"],
                "oos_trades": oos["trades"],
                "oos_long": oos["long_entries"],
                "oos_short": oos["short_entries"],
                "val_reasons": val["reasons"],
                "oos_reasons": oos["reasons"],
            }
        )
    out = pd.DataFrame(rows).sort_values("val_pnl", ascending=False)
    out.to_csv(OUT, index=False)
    print(out[["variant", "val_pnl", "val_mdd", "val_wr", "val_trades", "oos_pnl", "oos_mdd", "oos_wr", "oos_trades", "oos_long", "oos_short"]].to_string(index=False))
    print(f"saved {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
