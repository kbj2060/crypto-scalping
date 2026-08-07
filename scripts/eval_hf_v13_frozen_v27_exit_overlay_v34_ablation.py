#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import DeepAlphaConfig, DeepAlphaTCN, _json_default
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_frozen_v27_exit_overlay_v34_ablation_20260512"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_frozen_v27_exit_overlay_v34_ablation_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_exit_overlay_v34_ablation_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_exit_overlay_v34_ablation_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_exit_overlay_v34_ablation_20260512_grid.csv"
SEQ_LEN = 72
V27_COST1 = 226.82447187089713
V27_COST2 = 123.11659362616143
V27_COST3 = 14.22783363158393


@dataclass(frozen=True)
class OverlayConfig:
    name: str
    edge_th: float
    margin_th: float
    notional: float
    cooldown: int
    base_tp: float
    base_sl: float
    base_hold: int
    tp_util_mult: float
    sl_vol_mult: float
    trail_gap_mult: float
    trail_decay: float
    hold_decay_start: int
    hold_decay_rate: float
    tp_cap: float
    sl_cap: float


@dataclass(frozen=True)
class AblationConfig:
    name: str
    micro_decay: bool = False
    shadow_maker: bool = False
    scale_out: bool = False
    soft_blend: bool = False
    maker_arm_ratio: float = 0.85
    maker_fee_mult: float = 0.20
    scale_out_frac: float = 0.50
    scale_out_trigger: float = 0.70
    scale_out_edge_bonus: float = 0.006
    soft_blend_edge_mult: float = 1.25
    soft_blend_margin_mult: float = 1.50
    soft_blend_quality_ceiling: float = 0.04


V31_SELECTED = OverlayConfig(
    "v31_notional1_time_decay",
    0.010,
    0.0040,
    1.0,
    12,
    0.040,
    0.018,
    48,
    1.5,
    2.5,
    1.0,
    0.50,
    18,
    0.025,
    0.075,
    0.036,
)


def _ablations() -> list[AblationConfig]:
    return [
        AblationConfig("v31_baseline"),
        AblationConfig("v34_1_microstructure_decay", micro_decay=True),
        AblationConfig("v34_2_shadow_maker_execution", shadow_maker=True),
        AblationConfig("v34_3_dynamic_scale_out", scale_out=True),
        AblationConfig("v34_4_soft_parent_deep_blending", soft_blend=True),
    ]


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _safe_row_float(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _vol_anchor(row: pd.Series) -> float:
    bbw = abs(_safe_row_float(row, "bb_width", 0.0))
    gk = abs(_safe_row_float(row, "garman_klass_vol", 0.0))
    rs = abs(_safe_row_float(row, "rogers_satchell_vol", 0.0))
    pk = abs(_safe_row_float(row, "parkinson_vol", 0.0))
    volz = abs(_safe_row_float(row, "volatility_z", 0.0))
    rv = abs(_safe_row_float(row, "realized_vol_ratio", 1.0))
    base = max(0.0015, bbw * 0.15, gk * 2.5, rs * 2.5, pk * 2.5)
    scale = base * (1.0 + 0.08 * min(volz, 3.0) + 0.05 * max(rv - 1.0, 0.0))
    return _clip(scale, 0.0015, 0.030)


def _flow_alignment(row: pd.Series, side: int) -> tuple[float, float]:
    signed = float(1 if side > 0 else -1)
    net_taker = _safe_row_float(row, "net_taker_ratio", 0.0)
    taker_accel = _safe_row_float(row, "taker_acceleration", 0.0)
    flow_pressure = _safe_row_float(row, "ai_flow_pressure", 0.0)
    flow_slope = _safe_row_float(row, "dlinear_smf_slope", 0.0)
    flip_prob = _safe_row_float(row, "ai_flow_flip_prob", 0.0)
    exhaustion = _safe_row_float(row, "ai_flow_exhaustion", 0.0)
    aligned = signed * (0.45 * net_taker + 0.35 * taker_accel + 0.25 * flow_pressure + 0.20 * flow_slope)
    stress = max(exhaustion, flip_prob, -aligned)
    return float(aligned), float(stress)


def _limit_hit(row: pd.Series, side: int, limit_px: float, close_px: float) -> bool:
    high = _safe_row_float(row, "high", close_px)
    low = _safe_row_float(row, "low", close_px)
    if side > 0:
        return high >= limit_px
    return low <= limit_px


def _grid() -> list[OverlayConfig]:
    return [
        OverlayConfig("v31_ref", 0.010, 0.004, 1.2, 12, 0.045, 0.022, 48, 0.0, 1.0, 1.0, 0.0, 999, 0.0, 0.070, 0.035),
        OverlayConfig("v31_util_tp_vol_sl", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 2.0, 2.8, 1.2, 0.0, 999, 0.0, 0.080, 0.040),
        OverlayConfig("v31_trailing_time_decay", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 1.5, 2.6, 1.0, 0.45, 12, 0.020, 0.080, 0.040),
        OverlayConfig("v31_tight_after_24", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 1.5, 2.4, 0.9, 0.60, 24, 0.030, 0.080, 0.040),
        OverlayConfig("v31_notional1_time_decay", 0.010, 0.004, 1.0, 12, 0.040, 0.018, 48, 1.5, 2.5, 1.0, 0.50, 18, 0.025, 0.075, 0.036),
        OverlayConfig("v31_precision", 0.012, 0.005, 1.0, 12, 0.038, 0.017, 48, 1.2, 2.3, 0.8, 0.70, 18, 0.030, 0.070, 0.032),
    ]


def _seq_at(df: pd.DataFrame, idx: int, cols: list[str]) -> np.ndarray:
    start = max(0, idx - SEQ_LEN + 1)
    arr = (
        df.loc[start:idx, cols]
        .astype(float)
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0.0)
        .to_numpy(dtype=np.float32)
    )
    if len(arr) < SEQ_LEN:
        arr = np.vstack([np.zeros((SEQ_LEN - len(arr), len(cols)), dtype=np.float32), arr])
    return arr[-SEQ_LEN:]


def _apply_norm(seqs: np.ndarray, norm: dict[str, np.ndarray]) -> np.ndarray:
    return ((seqs - norm["mean"][None, None, :]) / norm["std"][None, None, :]).astype(np.float32)


def _predict_all(model: DeepAlphaTCN, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    seqs = np.stack([_seq_at(df, i, seq_cols) for i in range(len(df))]).astype(np.float32)
    x = _apply_norm(seqs, norm)
    out: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), 512):
            out.append(model(torch.from_numpy(x[start : start + 512])).numpy())
    return np.vstack(out).astype(np.float32)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1["trades"]) < 20:
        return -1e9 + float(c1["pnl"])
    return float(c1["pnl"] + 0.35 * c2["pnl"] + 0.20 * c3["pnl"] - 0.35 * abs(c1["mdd"]) + 0.20 * min(c1.get("deep_entries", 0), 90))


def _load_v27(path: Path) -> tuple[dict[str, Any], DeepAlphaTCN]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = DeepAlphaTCN(len(payload["seq_cols"]))
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return payload, model


def backtest(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    cfg: OverlayConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float = 1.0,
    decisions: pd.DataFrame | None = None,
    variant: AblationConfig | None = None,
    record: bool = False,
) -> dict[str, Any]:
    variant = variant or AblationConfig("v31_baseline")
    close = _close(df)
    if decisions is None:
        decisions = predict_policy_frame(bundle, df, close=close)
    fee_eff = fee * cost_mult
    slip_eff = slip * cost_mult
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
    entry_margin = 0.0
    entry_vol_anchor = 0.0
    maker_armed = False
    maker_limit = 0.0
    scale_out_done = False
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
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
            custom_exit_px: float | None = None
            exit_fee_mult = 1.0
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha" and variant.shadow_maker and maker_armed:
                row_now = df.iloc[i]
                close_px = float(close[i])
                if _limit_hit(row_now, pos, maker_limit, close_px):
                    reason = "deep_alpha_shadow_maker_exit"
                    custom_exit_px = float(maker_limit)
                    exit_fee_mult = float(variant.maker_fee_mult)
            if owner == "deep_alpha":
                if cfg.tp_util_mult > 0.0:
                    util_gain = 1.0 + cfg.tp_util_mult * max(entry_edge - cfg.edge_th, 0.0) / max(0.02, cfg.edge_th)
                    effective_tp = _clip(cfg.base_tp * util_gain, cfg.base_tp * 0.8, cfg.tp_cap)
                if cfg.sl_vol_mult > 0.0:
                    vol_sl = _clip(entry_vol_anchor * cfg.sl_vol_mult, cfg.base_sl * 0.6, cfg.sl_cap)
                    effective_sl = vol_sl
                if mfe > 0.0 and cfg.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * cfg.trail_gap_mult
                    if variant.micro_decay:
                        flow_align, flow_stress = _flow_alignment(df.iloc[i], pos)
                        if flow_stress >= 0.65 or flow_align <= -0.12:
                            trail_gap *= 0.20 if hold >= 3 else 0.35
                        elif flow_align >= 0.18 and flow_stress < 0.35:
                            trail_gap = min(trail_gap * 1.35, entry_vol_anchor * 1.80)
                    if cfg.hold_decay_start < 999 and hold >= cfg.hold_decay_start:
                        decay_bars = hold - cfg.hold_decay_start
                        decay = cfg.hold_decay_rate * decay_bars * entry_vol_anchor
                        if variant.micro_decay:
                            flow_align, flow_stress = _flow_alignment(df.iloc[i], pos)
                            if flow_align >= 0.18 and flow_stress < 0.35:
                                decay = 0.0
                            elif flow_stress >= 0.65 or flow_align <= -0.12:
                                decay *= 2.0
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - decay)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
                if variant.scale_out and not scale_out_done and entry_edge >= cfg.edge_th + variant.scale_out_edge_bonus and unreal >= effective_tp * variant.scale_out_trigger:
                    fill_i = min(i + 1, len(df) - 1)
                    exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                    raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                    delta = min(notional * float(variant.scale_out_frac), notional)
                    before = cash
                    cash = cash * (1.0 + raw * delta)
                    cash -= before * fee_eff * delta
                    notional = max(0.0, notional - delta)
                    scale_out_done = True
                    actions["deep_alpha_scale_out"] = actions.get("deep_alpha_scale_out", 0) + 1
                    if notional <= 1e-9:
                        reason = "deep_alpha_scale_out_full"
                        custom_exit_px = float(exit_px)
                if variant.shadow_maker and not maker_armed and effective_tp > 0.0 and unreal >= effective_tp * variant.maker_arm_ratio:
                    target_raw = effective_tp / max(notional, 1e-12)
                    maker_limit = entry_price * (1.0 + target_raw) if pos > 0 else entry_price * (1.0 - target_raw)
                    maker_armed = True
                    actions["deep_alpha_shadow_maker_armed"] = actions.get("deep_alpha_shadow_maker_armed", 0) + 1
            if owner == "v21_2" and not reason and variant.soft_blend and i >= SEQ_LEN:
                ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
                own_q = ql if pos > 0 else qs
                opp_q = qs if pos > 0 else ql
                opp_margin = opp_q - own_q
                q = decisions.iloc[i]
                parent_quality = _safe_row_float(q, "quality_score", 0.0)
                if opp_q >= cfg.edge_th * variant.soft_blend_edge_mult and opp_margin >= cfg.margin_th * variant.soft_blend_margin_mult and (parent_quality <= variant.soft_blend_quality_ceiling or unreal < 0.0):
                    reason = "v21_2_deep_cross_early_exit"
            tp_for_exit = 0.0 if owner == "deep_alpha" and variant.scale_out and scale_out_done else effective_tp
            if not reason and tp_for_exit > 0.0 and unreal >= tp_for_exit:
                reason = f"{owner}_take_profit"
            elif not reason and effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif not reason and max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = float(custom_exit_px) if custom_exit_px is not None else _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * exit_fee_mult * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "exit_reason": reason, "effective_tp": float(effective_tp), "effective_sl": float(effective_sl), "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fee_eff * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(cfg.cooldown))
                add_done = False
                maker_armed = False
                maker_limit = 0.0
                scale_out_done = False
                open_record = None
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
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            cash -= cash * fee_eff * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            maker_armed = False
            maker_limit = 0.0
            scale_out_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= cfg.edge_th and margin >= cfg.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(cfg.notional)
                notional = float(cfg.notional)
                take_profit = float(cfg.base_tp)
                stop_loss = float(cfg.base_sl)
                max_hold = int(cfg.base_hold)
                next_cooldown = int(cfg.cooldown)
                entry_edge = edge
                entry_margin = margin
                entry_vol_anchor = _vol_anchor(df.iloc[i]) * notional
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                maker_armed = False
                maker_limit = 0.0
                scale_out_done = False
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if record:
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "deep_q_long": ql, "deep_q_short": qs, "deep_edge": float(edge), "deep_margin": float(margin), "deep_vol_anchor": float(entry_vol_anchor), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
    if pos != 0:
        fill_i = len(df) - 1
        exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * fee_eff * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions}
    if record:
        out["trade_records"] = records
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V31 frozen V27 with rule-based dynamic exit overlay.")
    p.add_argument("--parent-model", type=Path, default=DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = _load_v27(args.v27_model)
    base = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    eval_q = _predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metrics_by_variant: dict[str, Any] = {}
    ledgers: dict[str, str] = {}

    for variant in _ablations():
        variant_metrics: dict[str, Any] = {}
        for mult in (1, 2, 3):
            r = backtest(
                eval_df,
                bundle,
                jackpot_model,
                add_cfg,
                eval_q,
                V31_SELECTED,
                fee=float(base["fee"]),
                slip=float(base["slip"]),
                cost_mult=float(mult),
                decisions=eval_dec,
                variant=variant,
                record=(mult == 1),
            )
            if mult == 1:
                ledger = pd.DataFrame(r.pop("trade_records", []))
                lp = args.report_out.with_name(f"{args.report_out.stem}_{variant.name}_cost1_ledger.csv")
                ledger.to_csv(lp, index=False)
                ledgers[variant.name] = str(lp)
            variant_metrics[f"cost{mult}"] = r
        metrics_by_variant[variant.name] = variant_metrics
        rows.append(
            {
                "variant": variant.name,
                "cost1_pnl": variant_metrics["cost1"]["pnl"],
                "cost1_mdd": variant_metrics["cost1"]["mdd"],
                "cost1_trades": variant_metrics["cost1"]["trades"],
                "cost1_trades_per_day": variant_metrics["cost1"]["trades_per_day"],
                "cost1_deep_entries": variant_metrics["cost1"].get("deep_entries", 0),
                "cost2_pnl": variant_metrics["cost2"]["pnl"],
                "cost2_mdd": variant_metrics["cost2"]["mdd"],
                "cost3_pnl": variant_metrics["cost3"]["pnl"],
                "cost3_mdd": variant_metrics["cost3"]["mdd"],
                "runner_actions": json.dumps(variant_metrics["cost1"].get("runner_actions", {}), ensure_ascii=False, default=_json_default),
                "exits": json.dumps(variant_metrics["cost1"].get("exits", {}), ensure_ascii=False, default=_json_default),
            }
        )

    baseline = metrics_by_variant["v31_baseline"]["cost1"]
    best_cost1 = max(metrics_by_variant.items(), key=lambda kv: kv[1]["cost1"]["pnl"])
    best_cost3 = max(metrics_by_variant.items(), key=lambda kv: kv[1]["cost3"]["pnl"])
    pd.DataFrame(rows).to_csv(args.grid_out, index=False)

    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    warnings.append("shadow_maker_execution_is_ohlc_approximation_without_orderbook_queue")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_cost1[1]["cost1"]["pnl"] > baseline["pnl"] else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "V31 selected config reused; no new 2026 selection",
        "oos_window": "2026 fixed OOS only",
        "policy": "frozen_v27_exit_overlay_v34_ablation",
        "v27_entry_frozen": True,
        "v21_2_preserved": True,
        "parent_policy_preserved": True,
        "feature_audit": feature_audit,
        "selected_config": asdict(V31_SELECTED),
        "ablations": [asdict(v) for v in _ablations()],
        "baseline_v31_cost1": baseline,
        "best_cost1_variant": best_cost1[0],
        "best_cost3_variant": best_cost3[0],
        "metrics": metrics_by_variant,
    }
    manifest_path = args.out_dir / "v34_ablation_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "model_id": MODEL_ID,
                "parent_model": str(args.parent_model),
                "jackpot_model": str(args.jackpot_model),
                "v27_model": str(args.v27_model),
                "selected_v31_config": asdict(V31_SELECTED),
                "ablations": [asdict(v) for v in _ablations()],
            },
            indent=2,
            ensure_ascii=False,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    report = {
        "model_id": MODEL_ID,
        "design": "Ablation study over promoted V31. The parent policy, V21.2 jackpot add-on, and frozen V27 entry model are preserved. Each run toggles exactly one proposed exit/execution interaction layer.",
        "metrics": metrics_by_variant,
        "summary_table": rows,
        "audit": audit,
        "artifacts": {
            "manifest": str(manifest_path),
            "report": str(args.report_out),
            "audit": str(args.audit_out),
            "grid": str(args.grid_out),
            "ledgers": ledgers,
        },
    }
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "best_cost1": best_cost1[0], "best_cost3": best_cost3[0], "rows": rows}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
