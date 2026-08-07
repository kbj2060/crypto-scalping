#!/usr/bin/env python3
from __future__ import annotations

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

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_mdd_pyramid_layer_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_mdd_pyramid_layer_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_mdd_pyramid_layer_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_mdd_pyramid_layer_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_mdd_pyramid_layer_20260513_grid.csv"


@dataclass(frozen=True)
class Runtime:
    name: str
    pyramid_scope: str
    add_frac: float
    max_total_mult: float
    min_unreal: float
    min_mfe: float
    max_mae_abs: float
    opp_utility_block: float
    dd_entry_block: float
    dd_pyramid_block: float
    mdd_guard_scope: str
    flow_th: float = 0.10
    fee_mult: float = 0.60
    slip_mult: float = 0.35


def _configs() -> list[Runtime]:
    rows = [
        Runtime("alpha1_4_execution_only", "none", 0.0, 1.0, 99.0, 99.0, 0.0, -99.0, 1.0, 1.0, "none"),
    ]
    for dd_scope, dd_entry in (("deep", 0.22), ("deep", 0.28), ("all", 0.28), ("none", 1.0)):
        rows.append(Runtime(f"mdd_guard_{dd_scope}_{dd_entry:.2f}", "none", 0.0, 1.0, 99.0, 99.0, 0.0, -99.0, dd_entry, 1.0, dd_scope))
    i = 0
    for scope in ("deep", "all"):
        for frac, total in ((0.08, 1.12), (0.12, 1.18), (0.18, 1.25)):
            for unreal, mfe in ((0.014, 0.018), (0.020, 0.026), (0.030, 0.036)):
                for dd_block in (0.16, 0.22, 0.30):
                    rows.append(Runtime(f"pyramid_{scope}_{i}", scope, frac, total, unreal, mfe, 0.006, 0.012, 1.0, dd_block, "none"))
                    rows.append(Runtime(f"pyramid_mdd_{scope}_{i}", scope, frac, total, unreal, mfe, 0.006, 0.012, 0.28, dd_block, "deep"))
                    i += 1
    return rows


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    mdd = abs(float(c1["mdd"]))
    mdd_penalty = 1.45 * max(mdd - 28.0, 0.0) + 0.35 * mdd
    cost3_bonus = 0.35 * max(float(c3["pnl"]), -25.0)
    return float(c1["pnl"] + 0.35 * c2["pnl"] + cost3_bonus - mdd_penalty)


def _route_variant(rt: Runtime) -> v45.LayerVariant:
    return v45.LayerVariant(
        rt.name,
        "alpha1_5_mdd_pyramid",
        alpha1.ALPHA1_CFG,
        execution_sniper=True,
        sniper_flow_th=rt.flow_th,
        sniper_fee_mult=rt.fee_mult,
        sniper_slip_mult=rt.slip_mult,
    )


def _should_scope(owner: str, scope: str) -> bool:
    if scope == "deep":
        return owner == "deep_alpha"
    if scope == "all":
        return owner in {"deep_alpha", "v21_2"}
    return False


def _deep_blocked(owner: str, dd_abs: float, rt: Runtime) -> bool:
    if rt.mdd_guard_scope == "none":
        return False
    if rt.mdd_guard_scope == "deep" and owner == "deep_alpha":
        return dd_abs >= rt.dd_entry_block
    if rt.mdd_guard_scope == "all":
        return dd_abs >= rt.dd_entry_block
    return False


def backtest_alpha15(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    rt: Runtime,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    decisions: pd.DataFrame,
) -> dict[str, Any]:
    close = _close(df)
    route_variant = _route_variant(rt)
    fee_base = float(fee) * float(cost_mult)
    slip_base = float(slip) * float(cost_mult)
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
    pyramid_done = False
    mfe = mae = 0.0
    entry_edge = entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    routes: dict[str, int] = {}

    def route_count(name: str) -> None:
        routes[name] = routes.get(name, 0) + 1

    def mark(i: int) -> tuple[float, float]:
        if pos == 0:
            return cash, 0.0
        px = float(close[int(np.clip(i, 0, len(close) - 1))])
        _, slip_eff, _ = v45._route_cost(df.iloc[int(np.clip(i, 0, len(df) - 1))], pos, fee_base, slip_base, route_variant)
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
            if owner == "deep_alpha":
                tp, sl = alpha1._effective_v31_thresholds(alpha1.ALPHA1_CFG, entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                if unreal >= tp:
                    reason = "deep_alpha_take_profit"
                elif unreal <= -abs(sl):
                    reason = "deep_alpha_stop_loss"
                elif hold >= int(alpha1.ALPHA1_CFG.base_hold):
                    reason = "deep_alpha_max_hold"
            elif owner == "v21_2":
                if take_profit > 0.0 and unreal >= take_profit:
                    reason = "v21_2_take_profit"
                elif stop_loss > 0.0 and unreal <= -abs(stop_loss):
                    reason = "v21_2_stop_loss"
                elif max_hold > 0 and hold >= max_hold:
                    reason = "v21_2_max_hold"
            if (
                not reason
                and not pyramid_done
                and _should_scope(owner, rt.pyramid_scope)
                and unreal >= rt.min_unreal
                and mfe >= rt.min_mfe
                and abs(mae) <= rt.max_mae_abs
                and dd_abs <= rt.dd_pyramid_block
            ):
                ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
                opposite = qs if pos > 0 else ql
                if opposite <= rt.opp_utility_block:
                    room = parent_notional * rt.max_total_mult - notional
                    delta = max(0.0, min(parent_notional * rt.add_frac, room, 4.14 - notional))
                    if delta > 1e-9:
                        fill_i = min(i + 1, len(df) - 1)
                        add_px, add_fee, _, add_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
                        new_notional = notional + delta
                        entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                        before = cash
                        cash -= before * add_fee * delta
                        notional = new_notional
                        actions["alpha15_profit_pyramid"] = actions.get("alpha15_profit_pyramid", 0) + 1
                        route_count(add_route)
                else:
                    actions["alpha15_opp_block"] = actions.get("alpha15_opp_block", 0) + 1
                pyramid_done = True
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px, add_fee, _, add_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * add_fee * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                    route_count(add_route)
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px, exit_fee, _, exit_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * exit_fee * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                route_count(exit_route)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(alpha1.ALPHA1_CFG.cooldown))
                add_done = False
                pyramid_done = False
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
            if _deep_blocked("v21_2", dd_abs, rt):
                actions["mdd_block_parent"] = actions.get("mdd_block_parent", 0) + 1
                continue
            fill_i = min(i + 1, len(df) - 1)
            pos = int(dec.side)
            owner = "v21_2"
            entry_price, entry_fee, _, entry_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
            entry_equity = cash
            entry_idx = i
            parent_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional)
            notional = parent_notional
            take_profit = float(dec.take_profit)
            stop_loss = float(dec.stop_loss)
            max_hold = int(dec.max_hold_bars)
            next_cooldown = int(dec.cooldown_bars)
            entry_edge = 0.0
            entry_vol_anchor = 0.0
            cash -= cash * entry_fee * notional
            long_entries += int(pos > 0)
            short_entries += int(pos < 0)
            notional_sum += notional
            leverage_sum += float(dec.leverage)
            mfe = mae = 0.0
            add_done = False
            pyramid_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            route_count(entry_route)
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            if _deep_blocked("deep_alpha", dd_abs, rt):
                actions["mdd_block_deep"] = actions.get("mdd_block_deep", 0) + 1
                continue
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= alpha1.ALPHA1_CFG.edge_th and margin >= alpha1.ALPHA1_CFG.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                entry_price, entry_fee, _, entry_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(alpha1.ALPHA1_CFG.notional)
                notional = float(alpha1.ALPHA1_CFG.notional)
                take_profit = float(alpha1.ALPHA1_CFG.base_tp)
                stop_loss = float(alpha1.ALPHA1_CFG.base_sl)
                max_hold = int(alpha1.ALPHA1_CFG.base_hold)
                next_cooldown = int(alpha1.ALPHA1_CFG.cooldown)
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
                pyramid_done = False
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                route_count(entry_route)
    if pos != 0:
        fill_i = len(df) - 1
        exit_px, exit_fee, _, exit_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=False)
        raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
        before = cash
        cash = cash * (1.0 + raw * notional)
        cash -= before * exit_fee * notional
        trades += 1
        wins += int(cash > entry_equity)
        exits["forced_end"] = exits.get("forced_end", 0) + 1
        route_count(exit_route)
    n = max(long_entries + short_entries, 1)
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(trades),
        "wr": float(wins / max(trades, 1)),
        "trades_per_day": float(trades / v31._days(df)),
        "deep_entries": int(deep_entries),
        "long_entries": int(long_entries),
        "short_entries": int(short_entries),
        "avg_notional": float(notional_sum / n),
        "avg_leverage": float(leverage_sum / n),
        "exits": exits,
        "runner_actions": actions,
        "route_counts": routes,
    }


def _metrics(df, bundle, jackpot_model, add_cfg, q, dec, rt: Runtime, fee: float, slip: float) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_alpha15(df, bundle, jackpot_model, add_cfg, q, rt, fee=fee, slip=slip, cost_mult=float(mult), decisions=dec)
        for mult in (1, 2, 3)
    }


def main() -> int:
    print(f"[{MODEL_ID}] loading alpha1 stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base_cfg = dict(bundle["config"])
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_contract = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    print(f"[{MODEL_ID}] predicting frozen parent/V27", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))
    rows: list[dict[str, Any]] = []
    selected = None
    best_score = -1e18
    for rt in _configs():
        vm = _metrics(val, bundle, jackpot_model, add_cfg, val_q, val_dec, rt, fee, slip)
        score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
        rows.append({**asdict(rt), "selection_score": score, "val_cost1_pnl": vm["cost1"]["pnl"], "val_cost1_mdd": vm["cost1"]["mdd"], "val_cost1_trades": vm["cost1"]["trades"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"], "val_pyramid_actions": vm["cost1"]["runner_actions"].get("alpha15_profit_pyramid", 0), "val_mdd_blocks": vm["cost1"]["runner_actions"].get("mdd_block_deep", 0) + vm["cost1"]["runner_actions"].get("mdd_block_parent", 0)})
        if score > best_score:
            best_score = score
            selected = rt
    assert selected is not None
    pd.DataFrame(rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    print(f"[{MODEL_ID}] selected {selected.name}", flush=True)
    baseline = _configs()[0]
    experiments = []
    for name, rt in (("alpha1.4", baseline), (f"alpha1.5::{selected.name}", selected)):
        metrics = _metrics(eval_df, bundle, jackpot_model, add_cfg, eval_q, eval_dec, rt, fee, slip)
        experiments.append({"name": name, "config": asdict(rt), "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)
    manifest_path = OUT_DIR / "alpha1_5_mdd_pyramid_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "selected_config": asdict(selected), "parent_frozen": True, "v27_frozen": True, "v31_exit_frozen": True, "v21_2_frozen": True}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    best = max(experiments, key=lambda e: e["score"])
    alpha14 = experiments[0]["metrics"]
    blocking = list(audit_contract.get("blocking", []))
    warnings = list(audit_contract.get("warnings", []))
    warnings.append("execution_component_is_ohlcv_proxy_not_live_l2_orderbook")
    if best["name"] != "alpha1.4" and best["metrics"]["cost1"]["mdd"] <= alpha14["cost1"]["mdd"]:
        warnings.append("selected_did_not_reduce_cost1_mdd_vs_alpha1_4")
    if best["name"] != "alpha1.4" and best["metrics"]["cost1"]["pnl"] <= alpha14["cost1"]["pnl"]:
        warnings.append("selected_did_not_beat_alpha1_4_cost1")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1.4" and best["metrics"]["cost1"]["mdd"] > alpha14["cost1"]["mdd"] and best["metrics"]["cost1"]["pnl"] >= alpha14["cost1"]["pnl"] * 0.97 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "parent_frozen": True,
        "v27_entry_frozen": True,
        "v31_exit_frozen": True,
        "v21_2_model_frozen": True,
        "selected_config": asdict(selected),
        "feature_audit": audit_contract,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1.5 MDD research: alpha1.4 execution proxy plus optional drawdown entry guard and tightly constrained profit-state pyramid. Parent, V27, V31, and V21.2 model are frozen.",
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
