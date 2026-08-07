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
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_cost3_cvar_notional_guard_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_cost3_cvar_notional_guard_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_cost3_cvar_notional_guard_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_cost3_cvar_notional_guard_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_cost3_cvar_notional_guard_20260513_grid.csv"


@dataclass(frozen=True)
class Runtime:
    name: str
    scope: str
    risk_th: float
    high_risk_th: float
    scale: float
    high_scale: float
    min_notional: float
    parent_cap: float
    add_cap_mult: float
    flow_th: float = 0.10
    fee_mult: float = 0.60
    slip_mult: float = 0.35


def _configs() -> list[Runtime]:
    rows = [Runtime("alpha1_4_identity", "none", 99.0, 199.0, 1.0, 1.0, 2.0, 99.0, 1.35)]
    for risk_th, high_th in ((0.15, 0.35), (0.25, 0.45), (0.35, 0.55), (0.45, 0.65), (0.55, 0.75)):
        for scale, high_scale in ((0.75, 0.50), (0.75, 0.00), (0.50, 0.00), (0.90, 0.50)):
            rows.append(Runtime(f"cvar_deep_r{risk_th:.2f}_h{high_th:.2f}_s{scale:.2f}_{high_scale:.2f}", "deep", risk_th, high_th, scale, high_scale, 0.0, 99.0, 1.35))
    return rows


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.38 * c2["pnl"] + 0.62 * c3["pnl"] - 0.35 * abs(c1["mdd"]) - 0.45 * max(abs(float(c3["mdd"])) - 45.0, 0.0))


def _route_variant(rt: Runtime) -> v45.LayerVariant:
    return v45.LayerVariant(rt.name, "alpha1_cost3_cvar_guard", alpha1.ALPHA1_CFG, execution_sniper=True, sniper_flow_th=rt.flow_th, sniper_fee_mult=rt.fee_mult, sniper_slip_mult=rt.slip_mult)


def _num(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        x = float(row.get(col, default))
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _risk_score(row: pd.Series, deep_q: np.ndarray, i: int, side: int) -> float:
    direction = float(np.sign(side))
    adverse_flow = -direction * (0.50 * _num(row, "net_taker_ratio") + 0.25 * _num(row, "taker_acceleration") + 0.15 * _num(row, "ofi_acceleration") + 0.10 * _num(row, "smart_money_flow"))
    opp_utility = float(deep_q[i, 1] if side > 0 else deep_q[i, 0])
    same_utility = float(deep_q[i, 0] if side > 0 else deep_q[i, 1])
    vol = max(0.0, _num(row, "volatility_z"), _num(row, "garch_vol_z"), _num(row, "realized_vol_ratio") - 1.0)
    entropy = max(0.0, _num(row, "ai_dir_entropy"), _num(row, "clean_regime_2024_unsup_v4_entropy"))
    adverse_ai = max(0.0, _num(row, "ai_adverse_risk"), _num(row, "m7_tail_risk"))
    trans = max(0.0, _num(row, "clean_regime_2024_unsup_v4_transition_risk"))
    opp_pressure = max(0.0, opp_utility - same_utility + 0.006) * 18.0
    return float(0.34 * max(0.0, adverse_flow) + 0.22 * min(vol, 3.0) / 3.0 + 0.16 * min(entropy, 2.5) / 2.5 + 0.14 * min(adverse_ai, 2.5) / 2.5 + 0.08 * min(trans, 1.0) + 0.06 * min(opp_pressure, 1.0))


def _scale_for(row: pd.Series, deep_q: np.ndarray, i: int, side: int, owner: str, rt: Runtime) -> tuple[float, float]:
    if rt.scope == "none":
        return 1.0, 0.0
    if rt.scope == "deep" and owner != "deep_alpha":
        return 1.0, 0.0
    risk = _risk_score(row, deep_q, i, side)
    if risk >= rt.high_risk_th:
        return float(rt.high_scale), risk
    if risk >= rt.risk_th:
        return float(rt.scale), risk
    return 1.0, risk


def backtest_guard(
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
    mfe = mae = 0.0
    entry_edge = entry_vol_anchor = 0.0
    trades = wins = long_entries = short_entries = deep_entries = 0
    notional_sum = leverage_sum = 0.0
    exits: dict[str, int] = {}
    actions: dict[str, int] = {}
    routes: dict[str, int] = {}
    risk_sum = risk_n = 0

    def rc(name: str) -> None:
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
            if owner == "v21_2" and not reason and not add_done and add_cfg.full_add_frac > 0.0 and unreal >= add_cfg.min_unrealized and hold >= add_cfg.min_bars_since_entry and dd_abs <= add_cfg.dd_block:
                state = {"parent_notional": parent_notional, "notional": notional, "bars_since_entry": hold, "unrealized": unreal, "mfe": mfe, "mae": mae, "drawdown_abs": dd_abs, "take_profit": take_profit, "stop_loss": stop_loss, "max_hold": max_hold}
                x = _feature_frame(df, bundle, decisions, i, state)
                _, _, _, q90, p_jackpot, p_bad, p_cost3 = _predict_cost_runner(jackpot_model, x)
                if p_jackpot >= add_cfg.jackpot_p and q90 >= add_cfg.jackpot_q90 and p_bad <= add_cfg.bad_cap and p_cost3 >= 0.40:
                    fill_i = min(i + 1, len(df) - 1)
                    room_mult = min(add_cfg.max_total_mult, rt.add_cap_mult if rt.scope == "all" else add_cfg.max_total_mult)
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * room_mult - notional))
                    add_px, add_fee, _, add_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    before = cash
                    cash -= before * add_fee * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                    rc(add_route)
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
                rc(exit_route)
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(alpha1.ALPHA1_CFG.cooldown))
                add_done = False
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
            scale, risk = _scale_for(df.iloc[i], deep_q, i, pos, owner, rt)
            risk_sum += risk
            risk_n += 1
            parent_cap = float(rt.parent_cap) if rt.scope == "all" else 99.0
            base_notional = min(float(dec.notional_exposure), add_cfg.max_entry_notional, parent_cap)
            parent_notional = max(0.0, base_notional * scale)
            notional = parent_notional
            entry_price, entry_fee, _, entry_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
            entry_equity = cash
            entry_idx = i
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
            leverage_sum += float(dec.leverage) * max(scale, 0.0)
            mfe = mae = 0.0
            add_done = False
            actions["v21_entry"] = actions.get("v21_entry", 0) + 1
            if scale < 0.999:
                actions["cvar_scaled_parent"] = actions.get("cvar_scaled_parent", 0) + 1
            rc(entry_route)
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= alpha1.ALPHA1_CFG.edge_th and margin >= alpha1.ALPHA1_CFG.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                pos = side
                owner = "deep_alpha"
                scale, risk = _scale_for(df.iloc[i], deep_q, i, pos, owner, rt)
                risk_sum += risk
                risk_n += 1
                if scale <= 1e-9:
                    pos = 0
                    owner = ""
                    actions["cvar_skip_deep"] = actions.get("cvar_skip_deep", 0) + 1
                    deep_cooldown = max(deep_cooldown, int(alpha1.ALPHA1_CFG.cooldown))
                    continue
                parent_notional = max(float(rt.min_notional), float(alpha1.ALPHA1_CFG.notional) * scale)
                notional = parent_notional
                entry_price, entry_fee, _, entry_route = v45._fill_with_route(df, fill_i, pos, fee_base, slip_base, route_variant, entry=True)
                entry_equity = cash
                entry_idx = i
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
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
                if scale < 0.999:
                    actions["cvar_scaled_deep"] = actions.get("cvar_scaled_deep", 0) + 1
                rc(entry_route)
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
        rc(exit_route)
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
        "avg_cvar_risk": float(risk_sum / max(risk_n, 1)),
        "exits": exits,
        "runner_actions": actions,
        "route_counts": routes,
    }


def _metrics(df, bundle, jackpot_model, add_cfg, q, dec, rt: Runtime, fee: float, slip: float) -> dict[str, Any]:
    return {
        f"cost{mult}": backtest_guard(df, bundle, jackpot_model, add_cfg, q, rt, fee=fee, slip=slip, cost_mult=float(mult), decisions=dec)
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
        rows.append({**asdict(rt), "selection_score": score, "val_cost1_pnl": vm["cost1"]["pnl"], "val_cost1_mdd": vm["cost1"]["mdd"], "val_cost2_pnl": vm["cost2"]["pnl"], "val_cost3_pnl": vm["cost3"]["pnl"], "val_scaled_deep": vm["cost1"]["runner_actions"].get("cvar_scaled_deep", 0), "val_scaled_parent": vm["cost1"]["runner_actions"].get("cvar_scaled_parent", 0), "val_avg_notional": vm["cost1"]["avg_notional"]})
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
    manifest_path = OUT_DIR / "alpha1_5_cost3_cvar_notional_guard_manifest.json"
    manifest_path.write_text(json.dumps({"model_id": MODEL_ID, "selected_config": asdict(selected), "parent_frozen": True, "v27_frozen": True, "v31_exit_frozen": True, "v21_2_frozen": True}, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    best = max(experiments, key=lambda e: e["score"])
    alpha14 = experiments[0]["metrics"]
    blocking = list(audit_contract.get("blocking", []))
    warnings = list(audit_contract.get("warnings", []))
    warnings.append("execution_component_is_ohlcv_proxy_not_live_l2_orderbook")
    if best["name"] != "alpha1.4" and best["metrics"]["cost1"]["pnl"] < alpha14["cost1"]["pnl"] * 0.90:
        warnings.append("selected_cost1_pnl_drop_gt_10pct_vs_alpha1_4")
    if best["name"] != "alpha1.4" and best["metrics"]["cost3"]["pnl"] <= alpha14["cost3"]["pnl"]:
        warnings.append("selected_did_not_improve_cost3_vs_alpha1_4")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best["name"] != "alpha1.4" and best["metrics"]["cost3"]["pnl"] > alpha14["cost3"]["pnl"] and best["metrics"]["cost1"]["pnl"] >= alpha14["cost1"]["pnl"] * 0.90 else "iterate",
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
        "design": "Alpha1.5 cost3/CVaR notional guard. Parent, V27, V31, and V21.2 are frozen. The guard does not block entries; it only scales notional down on high cost3/CVaR risk states while preserving alpha1.4 soft execution.",
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
