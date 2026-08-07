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


MODEL_ID = "hf_v13_frozen_v27_clean_regime_moe_v37_20260512"
DEFAULT_PARENT = ROOT / "data/ensemble/supervised/hf_v13_clean_regime_margin110_20260511/v13_clean_regime_margin110.pkl"
DEFAULT_JACKPOT = ROOT / "data/ensemble/supervised/hf_v13_jackpot_runner_v21_2_20260511/v21_2_jackpot_runner.pkl"
DEFAULT_V27 = ROOT / "data/ensemble/supervised/hf_v13_deep_alpha_candidate_expansion_v27_20260511/v27_deep_alpha_candidate_expansion.pt"
DEFAULT_TRAIN = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2025_patchtst__tide__dlinear.csv"
DEFAULT_EVAL = ROOT / "tmp/ai_feature_combo_grid/trade_candidates_2026_patchtst__tide__dlinear.csv"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_frozen_v27_clean_regime_moe_v37_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_clean_regime_moe_v37_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_clean_regime_moe_v37_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_frozen_v27_clean_regime_moe_v37_20260512_grid.csv"
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


def _grid() -> list[OverlayConfig]:
    return [
        OverlayConfig("v31_ref", 0.010, 0.004, 1.2, 12, 0.045, 0.022, 48, 0.0, 1.0, 1.0, 0.0, 999, 0.0, 0.070, 0.035),
        OverlayConfig("v31_util_tp_vol_sl", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 2.0, 2.8, 1.2, 0.0, 999, 0.0, 0.080, 0.040),
        OverlayConfig("v31_trailing_time_decay", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 1.5, 2.6, 1.0, 0.45, 12, 0.020, 0.080, 0.040),
        OverlayConfig("v31_tight_after_24", 0.010, 0.004, 1.2, 12, 0.040, 0.018, 48, 1.5, 2.4, 0.9, 0.60, 24, 0.030, 0.080, 0.040),
        OverlayConfig("v31_notional1_time_decay", 0.010, 0.004, 1.0, 12, 0.040, 0.018, 48, 1.5, 2.5, 1.0, 0.50, 18, 0.025, 0.075, 0.036),
        OverlayConfig("v31_precision", 0.012, 0.005, 1.0, 12, 0.038, 0.017, 48, 1.2, 2.3, 0.8, 0.70, 18, 0.030, 0.070, 0.032),
    ]


def _v31_selected() -> OverlayConfig:
    return OverlayConfig("v37_normal_v31", 0.010, 0.0040, 1.0, 12, 0.040, 0.018, 48, 1.5, 2.5, 1.0, 0.50, 18, 0.025, 0.075, 0.036)


def _trend_expert() -> OverlayConfig:
    return OverlayConfig("v37_trend_runner", 0.009, 0.0035, 1.2, 12, 0.046, 0.020, 60, 2.0, 2.8, 1.2, 0.35, 24, 0.015, 0.090, 0.040)


def _chop_expert() -> OverlayConfig:
    return OverlayConfig("v37_chop_precision", 0.013, 0.0060, 0.8, 18, 0.034, 0.015, 30, 1.0, 2.1, 0.7, 0.75, 12, 0.040, 0.060, 0.028)


def _risk_expert() -> OverlayConfig:
    return OverlayConfig("v37_risk_off_defensive", 0.016, 0.0080, 0.6, 24, 0.030, 0.012, 24, 0.8, 1.8, 0.55, 0.90, 6, 0.055, 0.050, 0.024)


def _route_cfg(row: pd.Series) -> OverlayConfig:
    risk = max(
        _safe_row_float(row, "clean_regime_2024_unsup_v4_risk_off_prob", 0.0),
        _safe_row_float(row, "clean_regime_2024_unsup_v4_transition_risk", 0.0),
        _safe_row_float(row, "clean_regime_2024_unsup_v4_whipsaw_prob", 0.0),
    )
    chop = _safe_row_float(row, "clean_regime_2024_unsup_v4_chop_prob", 0.0)
    trend = abs(_safe_row_float(row, "clean_regime_2024_unsup_v4_trend_bias", 0.0)) + max(
        _safe_row_float(row, "clean_regime_2024_unsup_v4_bull_prob", 0.0),
        _safe_row_float(row, "clean_regime_2024_unsup_v4_bear_prob", 0.0),
    )
    if risk >= 0.55:
        return _risk_expert()
    if chop >= 0.50:
        return _chop_expert()
    if trend >= 0.75:
        return _trend_expert()
    return _v31_selected()


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
    active_cfg = cfg
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
            ecfg = active_cfg if owner == "deep_alpha" else cfg
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha":
                if ecfg.tp_util_mult > 0.0:
                    util_gain = 1.0 + ecfg.tp_util_mult * max(entry_edge - ecfg.edge_th, 0.0) / max(0.02, ecfg.edge_th)
                    effective_tp = _clip(ecfg.base_tp * util_gain, ecfg.base_tp * 0.8, ecfg.tp_cap)
                if ecfg.sl_vol_mult > 0.0:
                    vol_sl = _clip(entry_vol_anchor * ecfg.sl_vol_mult, ecfg.base_sl * 0.6, ecfg.sl_cap)
                    effective_sl = vol_sl
                if mfe > 0.0 and ecfg.trail_gap_mult > 0.0:
                    trail_gap = entry_vol_anchor * ecfg.trail_gap_mult
                    if ecfg.hold_decay_start < 999 and hold >= ecfg.hold_decay_start:
                        decay_bars = hold - ecfg.hold_decay_start
                        trail_gap = max(entry_vol_anchor * 0.35, trail_gap - ecfg.hold_decay_rate * decay_bars * entry_vol_anchor)
                    trail_stop = max(-effective_sl, mfe - trail_gap)
                    effective_sl = min(effective_sl, max(0.001, trail_stop))
            if effective_tp > 0.0 and unreal >= effective_tp:
                reason = f"{owner}_take_profit"
            elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                reason = f"{owner}_stop_loss"
            elif max_hold > 0 and hold >= max_hold:
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
                closing_owner = owner
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
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
                deep_cooldown = max(deep_cooldown, int(active_cfg.cooldown if closing_owner == "deep_alpha" else cfg.cooldown))
                add_done = False
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
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if record:
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= SEQ_LEN:
            dc = _route_cfg(df.iloc[i])
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= dc.edge_th and margin >= dc.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = i
                active_cfg = dc
                parent_notional = float(dc.notional)
                notional = float(dc.notional)
                take_profit = float(dc.base_tp)
                stop_loss = float(dc.base_sl)
                max_hold = int(dc.base_hold)
                next_cooldown = int(dc.cooldown)
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
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    val_q = _predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = _predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _grid():
        v1 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=1.0, decisions=val_dec)
        v2 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=2.0, decisions=val_dec)
        v3 = backtest(val, bundle, jackpot_model, add_cfg, val_q, cfg, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=3.0, decisions=val_dec)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = OverlayConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        r = backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, selected, fee=float(base["fee"]), slip=float(base["slip"]), cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(r.pop("trade_records", []))
            lp = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            lp.parent.mkdir(parents=True, exist_ok=True)
            ledger.to_csv(lp, index=False)
            ledgers["cost1"] = str(lp)
        metrics[f"cost{mult}"] = r
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "v31_rule_exit_overlay_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "v27_model": str(args.v27_model), "selected_config": asdict(selected), "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model)}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame([{**{f"cfg_{k}": v for k, v in r["config"].items()}, "score": r["selection_score"], "val_pnl": r["validation_cost1"]["pnl"], "val_mdd": r["validation_cost1"]["mdd"], "val_trades": r["validation_cost1"]["trades"], "val_deep_entries": r["validation_cost1"].get("deep_entries", 0), "val_c2_pnl": r["validation_cost2"]["pnl"], "val_c3_pnl": r["validation_cost3"]["pnl"]} for r in rows]).to_csv(args.grid_out, index=False)
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    if metrics["cost1"]["pnl"] <= V27_COST1:
        warnings.append("oos_cost1_did_not_beat_v27")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > V27_COST1 and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "frozen_v27_rule_exit_overlay_v31", "v27_entry_frozen": True, "v21_2_preserved": True, "deep_sleeve_only_when_parent_cash": True, "feature_audit": feature_audit, "selected_config": asdict(selected), "metrics": metrics, "baseline_v27": {"cost1": V27_COST1, "cost2": V27_COST2, "cost3": V27_COST3}}
    report = {"model_id": MODEL_ID, "design": "V31 freezes the trained V27 entry model and V21.2 jackpot parent. Only deep_alpha exits are post-processed with utility-scaled TP, volatility-scaled SL, and time-decay trailing stop rules.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "audit": audit, "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "manifest": str(manifest_path), "selected": asdict(selected), "metrics": metrics, "verdict": verdict}, ensure_ascii=False, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
