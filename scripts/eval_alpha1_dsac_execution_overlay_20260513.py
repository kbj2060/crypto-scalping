#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
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

from ensemble.fully_learned_governor_policy import ACTION_CASH, predict_policy_frame  # noqa: E402
from ensemble.train_rl_dsac_agent import DSACRouter, DSAC_STATE_DIM, GaussianActor  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1_prev  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_convex_runner_pyramid_v18 import _feature_frame  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig, _predict_cost_runner  # noqa: E402


MODEL_ID = "alpha1_dsac_execution_overlay_20260513"
DEFAULT_CKPT = ROOT / "data/ensemble/ckpt/best_dsac_agents.pth"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_dsac_execution_overlay_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_dsac_execution_overlay_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_dsac_execution_overlay_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_dsac_execution_overlay_20260513_grid.csv"

ALPHA1_CFG = alpha1_prev.ALPHA1_CFG
ALPHA1_BASELINE = alpha1_prev.ALPHA1_BASELINE


@dataclass(frozen=True)
class DsacExecConfig:
    name: str
    taker_agree_th: float
    maker_agree_th: float
    maker_offset: float
    maker_fee_mult: float = 0.45
    penetration: float = 0.0001
    opposite_skip_th: float = 0.08
    fallback_to_taker: bool = False


def _configs() -> list[DsacExecConfig]:
    return [
        DsacExecConfig("dsac_agree_taker_weak_maker_2bp", 0.24, 0.08, 0.0002, fallback_to_taker=False),
        DsacExecConfig("dsac_agree_taker_weak_maker_5bp", 0.24, 0.08, 0.0005, fallback_to_taker=False),
        DsacExecConfig("dsac_strict_taker_maker_5bp", 0.34, 0.12, 0.0005, fallback_to_taker=False),
        DsacExecConfig("dsac_strict_taker_maker_10bp", 0.34, 0.12, 0.0010, fallback_to_taker=False),
        DsacExecConfig("dsac_soft_fallback_2bp", 0.24, 0.08, 0.0002, fallback_to_taker=True),
        DsacExecConfig("dsac_strict_fallback_5bp", 0.34, 0.12, 0.0005, fallback_to_taker=True),
    ]


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.25 * c3["pnl"] - 0.35 * abs(c1["mdd"]))


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_dsac_actor(path: Path, device: str) -> tuple[GaussianActor, dict[str, Any]]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state_dim = int(ckpt.get("state_dim", DSAC_STATE_DIM) or DSAC_STATE_DIM)
    actor = GaussianActor(state_dim=state_dim).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, {"checkpoint": str(path), "state_dim": state_dim, "keys": sorted(list(ckpt.keys()))}


def _num(df: pd.DataFrame, col: str, idx: int, default: float = 0.0) -> float:
    if col not in df.columns:
        return float(default)
    i = int(np.clip(idx, 0, len(df) - 1))
    try:
        x = float(pd.to_numeric(df[col], errors="coerce").ffill().iloc[i])
    except Exception:
        return float(default)
    return float(x) if np.isfinite(x) else float(default)


def _maker_entry(df: pd.DataFrame, idx: int, side: int, fee: float, cfg: DsacExecConfig) -> tuple[bool, float, float, str]:
    op = _num(df, "open", idx)
    hi = _num(df, "high", idx, op)
    lo = _num(df, "low", idx, op)
    if op <= 0.0:
        return False, 0.0, 0.0, "maker_bad_open"
    if side > 0:
        px = op * (1.0 - cfg.maker_offset)
        filled = lo <= px * (1.0 - cfg.penetration)
    else:
        px = op * (1.0 + cfg.maker_offset)
        filled = hi >= px * (1.0 + cfg.penetration)
    return bool(filled), float(px), float(fee * cfg.maker_fee_mult), "maker_fill" if filled else "maker_miss"


def _feature_dict(df: pd.DataFrame, i: int, side: int, edge: float, margin: float) -> dict[str, Any]:
    row = df.iloc[int(np.clip(i, 0, len(df) - 1))]
    out: dict[str, Any] = {}
    for k, v in row.items():
        if k == "timestamp":
            continue
        try:
            out[k] = float(v)
        except Exception:
            continue
    out["deep_edge"] = float(edge)
    out["deep_margin"] = float(margin)
    out["deep_side"] = float(side)
    return out


def _dsac_route(router: DSACRouter, df: pd.DataFrame, signal_i: int, side: int, edge: float, margin: float, cfg: DsacExecConfig, *, fee: float, slip: float) -> tuple[bool, float, float, str, dict[str, Any]]:
    features = _feature_dict(df, signal_i, side, edge, margin)
    action, lev, info = router.decide(features, {})
    raw = float(info.get("raw_action", 0.0))
    agree = raw * float(side)
    fill_i = min(signal_i + 1, len(df) - 1)
    if agree <= -abs(cfg.opposite_skip_th) or int(action) == 0:
        return False, 0.0, 0.0, "dsac_skip", info
    if agree >= cfg.taker_agree_th:
        return True, float(_fill_price(df, fill_i, side, slip, entry=True)), float(fee), "dsac_taker", info
    if agree >= cfg.maker_agree_th:
        filled, px, maker_fee, route = _maker_entry(df, fill_i, side, fee, cfg)
        if filled:
            return filled, px, maker_fee, "dsac_" + route, info
        if cfg.fallback_to_taker:
            return True, float(_fill_price(df, fill_i, side, slip, entry=True)), float(fee), "dsac_maker_miss_taker", info
        return False, px, maker_fee, "dsac_" + route, info
    return False, 0.0, 0.0, "dsac_weak_skip", info


def backtest_dsac_exec(
    df: pd.DataFrame,
    bundle: dict[str, Any],
    jackpot_model: dict[str, Any],
    add_cfg: CostRunnerConfig,
    deep_q: np.ndarray,
    *,
    fee: float,
    slip: float,
    cost_mult: float,
    decisions: pd.DataFrame,
    router: DSACRouter | None,
    cfg: DsacExecConfig | None,
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
    raw_sum = raw_n = 0.0

    def add_route(name: str) -> None:
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
            reason = ""
            if owner == "deep_alpha":
                tp, sl = alpha1_prev._effective_v31_thresholds(ALPHA1_CFG, entry_edge=entry_edge, entry_vol_anchor=entry_vol_anchor, mfe=mfe, hold=hold)
                if unreal >= tp:
                    reason = "deep_alpha_take_profit"
                elif unreal <= -abs(sl):
                    reason = "deep_alpha_stop_loss"
                elif hold >= int(ALPHA1_CFG.base_hold):
                    reason = "deep_alpha_max_hold"
            if owner == "v21_2" and not reason:
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
                    delta = max(0.0, min(parent_notional * add_cfg.full_add_frac, parent_notional * add_cfg.max_total_mult - notional))
                    add_px = _fill_price(df, fill_i, pos, slip_eff, entry=True)
                    new_notional = notional + delta
                    entry_price = (entry_price * notional + add_px * delta) / max(new_notional, 1e-12)
                    cash -= cash * fee_eff * delta
                    notional = new_notional
                    actions["v21_add_on"] = actions.get("v21_add_on", 0) + 1
                else:
                    actions["v21_reject"] = actions.get("v21_reject", 0) + 1
                add_done = True
            if reason:
                fill_i = min(i + 1, len(df) - 1)
                exit_px = _fill_price(df, fill_i, pos, slip_eff, entry=False)
                raw = (exit_px - entry_price) / max(entry_price, 1e-12) if pos > 0 else (entry_price - exit_px) / max(entry_price, 1e-12)
                before = cash
                cash = cash * (1.0 + raw * notional)
                cash -= before * fee_eff * notional
                trades += 1
                wins += int(cash > entry_equity)
                exits[reason] = exits.get(reason, 0) + 1
                pos = 0
                owner = ""
                cooldown = int(next_cooldown)
                next_cooldown = 0
                deep_cooldown = max(deep_cooldown, int(ALPHA1_CFG.cooldown))
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
            add_route("parent_taker")
            continue
        if deep_cooldown <= 0 and i >= v31.SEQ_LEN:
            ql, qs = float(deep_q[i, 0]), float(deep_q[i, 1])
            side = 1 if ql > qs else -1
            edge = max(ql, qs)
            margin = abs(ql - qs)
            if edge >= ALPHA1_CFG.edge_th and margin >= ALPHA1_CFG.margin_th:
                fill_i = min(i + 1, len(df) - 1)
                if router is None or cfg is None:
                    filled = True
                    fill_px = _fill_price(df, fill_i, side, slip_eff, entry=True)
                    fill_fee = fee_eff
                    route = "alpha1_taker"
                    info = {"raw_action": float(side)}
                else:
                    filled, fill_px, fill_fee, route, info = _dsac_route(router, df, i, side, edge, margin, cfg, fee=fee_eff, slip=slip_eff)
                add_route(route)
                raw_sum += float(info.get("raw_action", 0.0))
                raw_n += 1.0
                if not filled:
                    actions["deep_entry_miss"] = actions.get("deep_entry_miss", 0) + 1
                    deep_cooldown = max(deep_cooldown, int(ALPHA1_CFG.cooldown // 2))
                    continue
                pos = side
                owner = "deep_alpha"
                entry_price = float(fill_px)
                entry_equity = cash
                entry_idx = i
                parent_notional = float(ALPHA1_CFG.notional)
                notional = float(ALPHA1_CFG.notional)
                take_profit = float(ALPHA1_CFG.base_tp)
                stop_loss = float(ALPHA1_CFG.base_sl)
                max_hold = int(ALPHA1_CFG.base_hold)
                next_cooldown = int(ALPHA1_CFG.cooldown)
                entry_edge = edge
                entry_vol_anchor = v31._vol_anchor(df.iloc[i]) * float(notional)
                cash -= cash * float(fill_fee) * notional
                long_entries += int(pos > 0)
                short_entries += int(pos < 0)
                deep_entries += 1
                notional_sum += notional
                leverage_sum += max(notional, 1.0)
                mfe = mae = 0.0
                add_done = True
                actions["deep_entry"] = actions.get("deep_entry", 0) + 1
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
        "avg_dsac_raw_action": float(raw_sum / max(raw_n, 1.0)),
        "exits": exits,
        "runner_actions": actions,
        "route_counts": routes,
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    device = _device()
    actor, ckpt_meta = _load_dsac_actor(DEFAULT_CKPT, device)
    parent = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base = dict(parent["config"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    parent_audit = _audit_contract(train_all, eval_df, list(parent.get("feature_cols") or []))
    feature_audit = {"status": "pass", "blocking": [], "warnings": ["dsac_checkpoint_was_not_trained_as_alpha1_execution_only_agent"], "dsac_checkpoint": ckpt_meta}
    print(f"[{MODEL_ID}] audits parent={parent_audit.get('status')} dsac_state_dim={ckpt_meta['state_dim']} device={device}", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(parent, val, close=_close(val))
    eval_dec = predict_policy_frame(parent, eval_df, close=_close(eval_df))
    fee = float(base["fee"])
    slip = float(base["slip"])

    grid_rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for cfg in _configs():
        metrics = {}
        for mult in (1, 2, 3):
            router = DSACRouter(copy.deepcopy(actor), device=device)
            metrics[f"cost{mult}"] = backtest_dsac_exec(val, parent, jackpot_model, add_cfg, val_q, fee=fee, slip=slip, cost_mult=float(mult), decisions=val_dec, router=router, cfg=cfg)
        row = {"config": asdict(cfg), "validation": metrics, "selection_score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])}
        grid_rows.append(row)
        print(f"[{MODEL_ID}] val {cfg.name} score={row['selection_score']:.2f} c1={metrics['cost1']['pnl']:.2f} c3={metrics['cost3']['pnl']:.2f} routes={metrics['cost1'].get('route_counts', {})}", flush=True)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row
    assert best is not None
    selected = DsacExecConfig(**best["config"])

    experiments: list[dict[str, Any]] = []
    for name, cfg in [("alpha1", None), ("alpha1.4_dsac_execution_overlay", selected)]:
        metrics = {}
        for mult in (1, 2, 3):
            router = None if cfg is None else DSACRouter(copy.deepcopy(actor), device=device)
            metrics[f"cost{mult}"] = backtest_dsac_exec(eval_df, parent, jackpot_model, add_cfg, eval_q, fee=fee, slip=slip, cost_mult=float(mult), decisions=eval_dec, router=router, cfg=cfg)
        experiments.append({"name": name, "metrics": metrics, "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"])})
        print(f"[{MODEL_ID}] {name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}", flush=True)

    manifest = {"model_id": MODEL_ID, "selected_config": asdict(selected), "checkpoint": ckpt_meta, "note": "Existing project DSAC actor is used as execution agreement/skip router for alpha1 deep_alpha entries only."}
    (OUT_DIR / "alpha1_dsac_execution_overlay_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    pd.DataFrame(
        [
            {
                **{f"cfg_{k}": v for k, v in r["config"].items()},
                "score": r["selection_score"],
                "val_pnl": r["validation"]["cost1"]["pnl"],
                "val_mdd": r["validation"]["cost1"]["mdd"],
                "val_c2_pnl": r["validation"]["cost2"]["pnl"],
                "val_c3_pnl": r["validation"]["cost3"]["pnl"],
                "val_routes": json.dumps(r["validation"]["cost1"].get("route_counts", {}), ensure_ascii=False),
            }
            for r in grid_rows
        ]
    ).to_csv(GRID_OUT, index=False)
    best_exp = max(experiments, key=lambda x: x["score"])
    blocking = list(parent_audit.get("blocking", [])) + list(feature_audit.get("blocking", []))
    warnings = list(parent_audit.get("warnings", [])) + list(feature_audit.get("warnings", []))
    warnings.append("maker_fill_simulation_uses_next_bar_ohlc_not_live_orderbook_queue")
    if best_exp["metrics"]["cost1"]["pnl"] <= ALPHA1_BASELINE["cost1"]["pnl"]:
        warnings.append("best_did_not_beat_alpha1_cost1")
    if best_exp["metrics"]["cost2"]["pnl"] <= 0.0:
        warnings.append("best_cost2_not_survived")
    if best_exp["metrics"]["cost3"]["pnl"] <= 0.0:
        warnings.append("best_cost3_not_survived")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "promote" if not blocking and best_exp["name"] != "alpha1" and best_exp["metrics"]["cost1"]["pnl"] > ALPHA1_BASELINE["cost1"]["pnl"] and best_exp["metrics"]["cost2"]["pnl"] > 0.0 else "iterate",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "policy": MODEL_ID,
        "alpha1_parent_preserved": True,
        "alpha1_v27_entry_preserved": True,
        "alpha1_exit_preserved": True,
        "selected_config": asdict(selected),
        "dsac_checkpoint": ckpt_meta,
        "parent_audit": parent_audit,
        "feature_audit": feature_audit,
        "metrics": {e["name"]: e["metrics"] for e in experiments},
        "baseline_alpha1": ALPHA1_BASELINE,
    }
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1.4 uses the existing project DSAC actor as an execution-only overlay. DSAC does not choose trade direction; it gates deep_alpha entries into taker, maker, or skip based on agreement with the frozen V27 side.",
        "selected_config": asdict(selected),
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"manifest": str(OUT_DIR / "alpha1_dsac_execution_overlay_manifest.json"), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best_exp}, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
