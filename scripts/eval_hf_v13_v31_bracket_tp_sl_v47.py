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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31
from scripts.eval_hf_v13_v31_frozen_parent_layer_ablation_v45 import _base_overlay, _score
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _days, _fill_price, _read
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner


MODEL_ID = "hf_v13_v31_bracket_tp_sl_v47_20260512"
DEFAULT_OUT_DIR = ROOT / "data/ensemble/supervised/hf_v13_v31_bracket_tp_sl_v47_20260512"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/hf_v13_v31_bracket_tp_sl_v47_20260512_summary.json"
DEFAULT_AUDIT = ROOT / "data/ensemble/reports/hf_v13_v31_bracket_tp_sl_v47_20260512_audit.json"
DEFAULT_GRID = ROOT / "data/ensemble/reports/hf_v13_v31_bracket_tp_sl_v47_20260512_grid.csv"


@dataclass(frozen=True)
class BracketConfig:
    name: str
    owners: str
    maker_fee_mult: float
    tp_penetration: float
    stop_slip_mult: float
    both_hit: str = "stop_first"


def _num(df: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
    if col not in df.columns:
        return float(default)
    try:
        x = float(pd.to_numeric(df[col], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _open(df: pd.DataFrame, idx: int) -> float:
    return _num(df, "open", idx, 0.0)


def _high(df: pd.DataFrame, idx: int) -> float:
    return _num(df, "high", idx, _open(df, idx))


def _low(df: pd.DataFrame, idx: int) -> float:
    return _num(df, "low", idx, _open(df, idx))


def _owner_bracketed(owner: str, cfg: BracketConfig) -> bool:
    if cfg.owners == "all":
        return True
    if cfg.owners == "parent":
        return owner == "v21_2"
    if cfg.owners == "deep":
        return owner == "deep_alpha"
    return False


def _bracket_prices(side: int, entry_price: float, notional: float, effective_tp: float, effective_sl: float) -> tuple[float, float]:
    n = max(float(notional), 1e-12)
    tp_ret = max(float(effective_tp), 0.0) / n
    sl_ret = max(float(effective_sl), 0.0) / n
    if side > 0:
        return float(entry_price * (1.0 + tp_ret)), float(entry_price * (1.0 - sl_ret))
    return float(entry_price * (1.0 - tp_ret)), float(entry_price * (1.0 + sl_ret))


def _stop_fill_price(df: pd.DataFrame, idx: int, side: int, stop_price: float, slip: float) -> float:
    op = _open(df, idx)
    if side > 0:
        base = min(float(stop_price), op) if op > 0 else float(stop_price)
        return float(base * (1.0 - slip))
    base = max(float(stop_price), op)
    return float(base * (1.0 + slip))


def _bracket_hit(
    df: pd.DataFrame,
    idx: int,
    side: int,
    entry_price: float,
    notional: float,
    effective_tp: float,
    effective_sl: float,
    cfg: BracketConfig,
    *,
    fee: float,
    slip: float,
) -> tuple[bool, str, float, float, float, float]:
    tp_price, sl_price = _bracket_prices(side, entry_price, notional, effective_tp, effective_sl)
    hi = _high(df, idx)
    lo = _low(df, idx)
    if side > 0:
        tp_hit = effective_tp > 0.0 and hi >= tp_price * (1.0 + cfg.tp_penetration)
        sl_hit = effective_sl > 0.0 and lo <= sl_price
    else:
        tp_hit = effective_tp > 0.0 and lo <= tp_price * (1.0 - cfg.tp_penetration)
        sl_hit = effective_sl > 0.0 and hi >= sl_price
    if tp_hit and sl_hit and cfg.both_hit == "stop_first":
        fill = _stop_fill_price(df, idx, side, sl_price, slip * cfg.stop_slip_mult)
        return True, "stop_market", fill, fee, tp_price, sl_price
    if sl_hit:
        fill = _stop_fill_price(df, idx, side, sl_price, slip * cfg.stop_slip_mult)
        return True, "stop_market", fill, fee, tp_price, sl_price
    if tp_hit:
        return True, "tp_maker_limit", tp_price, fee * cfg.maker_fee_mult, tp_price, sl_price
    return False, "", 0.0, 0.0, tp_price, sl_price


def _deep_effective_levels(row: pd.Series, overlay: v31.OverlayConfig, entry_edge: float, entry_vol_anchor: float, mfe: float, hold: int) -> tuple[float, float]:
    effective_tp = float(overlay.base_tp)
    effective_sl = float(overlay.base_sl)
    if overlay.tp_util_mult > 0.0:
        util_gain = 1.0 + overlay.tp_util_mult * max(entry_edge - overlay.edge_th, 0.0) / max(0.02, overlay.edge_th)
        effective_tp = v31._clip(overlay.base_tp * util_gain, overlay.base_tp * 0.8, overlay.tp_cap)
    if overlay.sl_vol_mult > 0.0:
        effective_sl = v31._clip(entry_vol_anchor * overlay.sl_vol_mult, overlay.base_sl * 0.6, overlay.sl_cap)
    if mfe > 0.0 and overlay.trail_gap_mult > 0.0:
        trail_gap = entry_vol_anchor * overlay.trail_gap_mult
        if overlay.hold_decay_start < 999 and hold >= overlay.hold_decay_start:
            decay_bars = hold - overlay.hold_decay_start
            trail_gap = max(entry_vol_anchor * 0.35, trail_gap - overlay.hold_decay_rate * decay_bars * entry_vol_anchor)
        trail_stop = max(-effective_sl, mfe - trail_gap)
        effective_sl = min(effective_sl, max(0.001, trail_stop))
    return float(effective_tp), float(effective_sl)


def backtest_bracket(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    overlay: v31.OverlayConfig,
    bracket: BracketConfig,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    decisions: pd.DataFrame,
    record: bool = False,
) -> dict[str, Any]:
    close = _close(df)
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
    entry_edge = entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    routes: dict[str, int] = {}
    records: list[dict[str, Any]] = []
    open_record: dict[str, Any] | None = None

    def route(name: str) -> None:
        routes[name] = routes.get(name, 0) + 1

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
            effective_tp = take_profit
            effective_sl = stop_loss
            if owner == "deep_alpha":
                effective_tp, effective_sl = _deep_effective_levels(df.iloc[i], overlay, entry_edge, entry_vol_anchor, mfe, hold)
            reason = ""
            fill_price = fill_fee = 0.0
            tp_price = sl_price = 0.0
            if _owner_bracketed(owner, bracket):
                hit, route_name, px, route_fee, tp_price, sl_price = _bracket_hit(
                    df,
                    i,
                    pos,
                    entry_price,
                    notional,
                    effective_tp,
                    effective_sl,
                    bracket,
                    fee=fee_eff,
                    slip=slip_eff,
                )
                if hit:
                    reason = f"{owner}_{route_name}"
                    fill_price = px
                    fill_fee = route_fee
                    route(route_name)
            else:
                if effective_tp > 0.0 and unreal >= effective_tp:
                    reason = f"{owner}_take_profit"
                elif effective_sl > 0.0 and unreal <= -abs(effective_sl):
                    reason = f"{owner}_stop_loss"
                if reason:
                    fill_i = min(i + 1, len(df) - 1)
                    fill_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                    fill_fee = fee_eff
                    route("legacy_next_open")
            if not reason and max_hold > 0 and hold >= max_hold:
                reason = f"{owner}_max_hold"
                fill_i = min(i + 1, len(df) - 1)
                fill_price = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                fill_fee = fee_eff
                route("max_hold_taker")
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
                    cash -= cash * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                    route("addon_taker_rebracket")
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                raw = (fill_price - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - fill_price) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fill_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                if record and open_record is not None:
                    fill_i = min(i + 1, len(df) - 1)
                    out = dict(open_record)
                    out.update({"exit_signal_timestamp": str(df["timestamp"].iloc[i]), "exit_fill_timestamp": str(df["timestamp"].iloc[i if "maker" in reason or "stop_market" in reason else fill_i]), "exit_reason": reason, "exit_price": float(fill_price), "tp_price": float(tp_price), "sl_price": float(sl_price), "effective_tp": float(effective_tp), "effective_sl": float(effective_sl), "realized_net_pct": float((cash / max(entry_equity, 1e-12) - 1.0) * 100.0), "final_notional_exposure": float(notional), "mfe_pct": float(mfe * 100.0), "mae_pct": float(mae * 100.0), "fee_exit_pct": float(fill_fee * notional * 100.0), "cash_after": float(cash)})
                    records.append(out)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(overlay.cooldown))
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
            entry_idx = fill_i
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
            route("entry_taker")
            if record:
                tp_price, sl_price = _bracket_prices(pos, entry_price, notional, take_profit, stop_loss)
                open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "leverage": float(dec.leverage), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "tp_price_at_entry": float(tp_price), "sl_price_at_entry": float(sl_price), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= overlay.edge_th and margin >= overlay.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                entry_equity = cash
                entry_idx = fill_i
                parent_notional = float(overlay.notional)
                notional = float(overlay.notional)
                take_profit = float(overlay.base_tp)
                stop_loss = float(overlay.base_sl)
                max_hold = int(overlay.base_hold)
                next_cooldown = int(overlay.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * notional
                cash -= cash * fee_eff * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                route("entry_taker")
                if record:
                    effective_tp, effective_sl = _deep_effective_levels(df.iloc[i], overlay, entry_edge, entry_vol_anchor, mfe, 0)
                    tp_price, sl_price = _bracket_prices(pos, entry_price, notional, effective_tp, effective_sl)
                    open_record = {"entry_signal_timestamp": str(df["timestamp"].iloc[i]), "entry_fill_timestamp": str(df["timestamp"].iloc[fill_i]), "owner": owner, "side": "LONG" if pos > 0 else "SHORT", "entry_price": float(entry_price), "notional_exposure": float(notional), "deep_q_long": ql, "deep_q_short": qs, "deep_edge": float(edge), "deep_margin": float(margin), "deep_vol_anchor": float(entry_vol_anchor), "take_profit": float(take_profit), "stop_loss": float(stop_loss), "tp_price_at_entry": float(tp_price), "sl_price_at_entry": float(sl_price), "max_hold_bars": int(max_hold), "fee_entry_pct": float(fee_eff * notional * 100.0)}
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
        route("forced_taker")
    n = max(long_entries + short_entries, 1)
    out = {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(trades), "wr": float(wins / max(trades, 1)), "trades_per_day": float(trades / _days(df)), "deep_entries": int(deep_entries), "long_entries": int(long_entries), "short_entries": int(short_entries), "avg_notional": float(notional_sum / n), "avg_leverage": float(leverage_sum / n), "exits": exits, "runner_actions": actions, "route_counts": routes}
    if record:
        out["trade_records"] = records
    return out


def _variants() -> list[BracketConfig]:
    return [
        BracketConfig("v47_parent_bracket_maker045_pen0", "parent", 0.45, 0.0, 1.0),
        BracketConfig("v47_parent_bracket_maker045_pen1bp", "parent", 0.45, 0.0001, 1.0),
        BracketConfig("v47_deep_bracket_maker045_pen1bp", "deep", 0.45, 0.0001, 1.0),
        BracketConfig("v47_all_bracket_maker045_pen0", "all", 0.45, 0.0, 1.0),
        BracketConfig("v47_all_bracket_maker045_pen1bp", "all", 0.45, 0.0001, 1.0),
        BracketConfig("v47_all_bracket_maker070_pen1bp", "all", 0.70, 0.0001, 1.0),
        BracketConfig("v47_all_bracket_maker045_pen2bp", "all", 0.45, 0.0002, 1.2),
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="V31 bracket order test: parent/V27 TP as maker limit and SL as stop-market.")
    p.add_argument("--parent-model", type=Path, default=v31.DEFAULT_PARENT)
    p.add_argument("--jackpot-model", type=Path, default=v31.DEFAULT_JACKPOT)
    p.add_argument("--v27-model", type=Path, default=v31.DEFAULT_V27)
    p.add_argument("--train-csv", type=Path, default=v31.DEFAULT_TRAIN)
    p.add_argument("--eval-csv", type=Path, default=v31.DEFAULT_EVAL)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    p.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    p.add_argument("--grid-out", type=Path, default=DEFAULT_GRID)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(args.parent_model)
    jackpot_payload = joblib.load(args.jackpot_model)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(args.v27_model)
    base_cfg = dict(bundle["config"])
    train_all = _read(args.train_csv)
    eval_df = _read(args.eval_csv)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    feature_audit = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    overlay = _base_overlay()
    print(f"[{MODEL_ID}] predicting frozen parent and V27", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    print(f"[{MODEL_ID}] selecting bracket config on 2025 Q4", flush=True)
    for cfg in _variants():
        v1 = backtest_bracket(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=1.0, decisions=val_dec)
        v2 = backtest_bracket(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=2.0, decisions=val_dec)
        v3 = backtest_bracket(val, bundle, jackpot_model, add_cfg, val_q, overlay, cfg, fee=fee, slip=slip, cost_mult=3.0, decisions=val_dec)
        row = {"config": asdict(cfg), "validation_cost1": v1, "validation_cost2": v2, "validation_cost3": v3, "selection_score": _score(v1, v2, v3)}
        rows.append(row)
        print(f"[{MODEL_ID}] val {cfg.name} score={row['selection_score']:.4f} c1={v1['pnl']:.2f} c3={v3['pnl']:.2f} routes={v1['route_counts']}", flush=True)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = BracketConfig(**best["config"])
    metrics: dict[str, Any] = {}
    ledgers: dict[str, str] = {}
    for mult in (1, 2, 3):
        result = backtest_bracket(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, selected, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, record=(mult == 1))
        if mult == 1:
            ledger = pd.DataFrame(result.pop("trade_records", []))
            ledger_path = args.report_out.with_name(args.report_out.stem + "_cost1_ledger.csv")
            ledger.to_csv(ledger_path, index=False)
            ledgers["cost1"] = str(ledger_path)
        metrics[f"cost{mult}"] = result
    oos_all_variants = {
        cfg.name: {
            f"cost{mult}": backtest_bracket(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, cfg, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec)
            for mult in (1, 2, 3)
        }
        for cfg in _variants()
    }
    baseline = {
        f"cost{mult}": v31.backtest(eval_df, bundle, jackpot_model, add_cfg, eval_q, overlay, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec)
        for mult in (1, 2, 3)
    }
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in row["config"].items()},
                "selection_score": row["selection_score"],
                "val_cost1_pnl": row["validation_cost1"]["pnl"],
                "val_cost1_mdd": row["validation_cost1"]["mdd"],
                "val_cost1_trades": row["validation_cost1"]["trades"],
                "val_cost2_pnl": row["validation_cost2"]["pnl"],
                "val_cost3_pnl": row["validation_cost3"]["pnl"],
                "val_cost1_routes": json.dumps(row["validation_cost1"].get("route_counts", {}), ensure_ascii=False),
            }
            for row in rows
        ]
    ).to_csv(args.grid_out, index=False)
    manifest_path = args.out_dir / "v47_bracket_tp_sl_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "parent_model": str(args.parent_model), "jackpot_model": str(args.jackpot_model), "v27_model": str(args.v27_model), "selected_config": asdict(selected), "overlay": asdict(overlay)}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    blocking: list[str] = []
    warnings: list[str] = []
    if feature_audit["status"] != "pass":
        blocking.extend(feature_audit.get("blocking", []))
    warnings.extend(feature_audit.get("warnings", []))
    warnings.append("bracket_intrabar_order_uses_ohlc_stop_first_when_tp_and_sl_touch_same_bar")
    warnings.append("tp_maker_fill_uses_ohlc_penetration_not_live_l2_queue")
    oracle_best = max(oos_all_variants, key=lambda name: float(oos_all_variants[name]["cost1"]["pnl"]))
    if oracle_best != selected.name:
        warnings.append(f"validation_selected_config_differs_from_oos_best:{selected.name}!={oracle_best}")
    if metrics["cost1"]["pnl"] <= baseline["cost1"]["pnl"]:
        warnings.append("oos_cost1_did_not_beat_v31_baseline")
    if metrics["cost2"]["pnl"] <= 0.0:
        warnings.append("cost2_not_survived")
    if metrics["cost3"]["pnl"] <= 0.0:
        warnings.append("cost3_not_survived")
    verdict = "promote" if not blocking and metrics["cost1"]["pnl"] > baseline["cost1"]["pnl"] and metrics["cost2"]["pnl"] > 0.0 and metrics["cost3"]["pnl"] > 0.0 else "iterate"
    audit = {"status": "pass" if not blocking else "fail", "verdict": verdict, "blocking": blocking, "warnings": warnings, "selection_uses_2026": False, "selection_window": "2025-10-01..2025-12-31", "oos_window": "2026 fixed OOS only after selection", "policy": "v31_bracket_tp_sl", "parent_frozen": True, "v27_entry_frozen": True, "v21_2_model_frozen": True, "selected_config": asdict(selected), "oracle_best_oos_config": oracle_best, "feature_audit": feature_audit, "baseline_v31": baseline, "metrics": metrics}
    report = {"model_id": MODEL_ID, "design": "V31 bracket order model. After entry fill, TP/SL thresholds are converted from equity thresholds to price levels using threshold/notional. TP is simulated as reduce-only maker limit; SL is stop-market/taker. Add-ons reblend entry and reprice the bracket.", "selected_config": asdict(selected), "selection_result": best, "metrics": metrics, "oos_all_variants": oos_all_variants, "oracle_best_oos_config": oracle_best, "baseline_v31": baseline, "audit": audit, "artifacts": {"manifest": str(manifest_path), "report": str(args.report_out), "audit": str(args.audit_out), "grid": str(args.grid_out), "ledgers": ledgers}}
    args.report_out.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    args.audit_out.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(args.report_out), "audit": str(args.audit_out), "selected": asdict(selected), "metrics": metrics, "baseline": baseline, "verdict": verdict}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
