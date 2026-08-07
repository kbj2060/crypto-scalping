#!/usr/bin/env python3
"""Native bar-by-bar portfolio replay for the frozen 2-action RL gate.

This evaluates the frozen policy learned by
train_portfolio_rl_gate_2action_20260708.py without using saved per-asset trade
ledgers or saved exit timestamps during replay. At each common 5m timestamp:

1. Generate ETH/SOL/BTC candidates from per-bar parent/risk artifacts.
2. Rule-select a single top candidate.
3. Let the frozen 2-action policy SKIP or TAKE_TOP.
4. If taken, advance bar-by-bar until the selected asset's own TP/SL/exit-head
   contract closes the position.

The frozen policy itself is still trained from the event-level prototype, so
this is a native evaluation step, not a complete promotion-grade retrain.
"""
from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import replay_omega4_6_1_greedy_router_20260706 as eth_greedy  # noqa: E402
import replay_omega4_6_1_greedy_val_20260706 as eth_valmod  # noqa: E402
import replay_omega4_6_1_two_component_router_assets_20260708 as asset_router  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as eth_retest  # noqa: E402
import train_omega1_regime3_expert_direction_head_volpca_20260602 as hard  # noqa: E402
import train_portfolio_rl_gate_2action_20260708 as proto  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_rl_gate_2action_native_20260708"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_rl_gate_2action_native_20260708.md"
POLICY_REPORT = ROOT / "tmp/causal_regen_20260516/portfolio_rl_gate_2action_20260708/report.json"

ASSET_SCORES = {"eth": 1.7639815967368822, "sol": 5.11558499257149, "btc": 1.909145364083251}
DURATION_THRESHOLDS = {"eth": 0.005417, "sol": 0.0055208323, "btc": 0.00541154875}
LEVERAGE_CAP = 5.0
NOTIONAL_CAP = 1.8
COST_MULT = 3.0

SOL_CFG = {
    "parent_dir": "tmp/causal_regen_20260516/sol_omega4_3head_parent72_loose_entry_quality_20260707_zig075_20260707",
    "risk_dir": "tmp/causal_regen_20260516/sol_omega4_2_trade_risk_sidecar_20260707_zig075_q070_20260707",
    "tag": "q070",
    "long_scale": 0.5,
    "short_scale": 1.75,
}
BTC_CFG = {
    "parent_dir": "tmp/causal_regen_20260516/btc_omega4_3head_parent72_loose_entry_quality_20260708_h48qual_20260708",
    "risk_dir": "tmp/causal_regen_20260516/btc_omega4_2_trade_risk_sidecar_20260708_h48qual_q055_20260708",
    "tag": "q055",
    "long_scale": 0.5,
    "short_scale": 2.5,
}


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


def _compound_metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    for ret in ledger["trade_return"].to_numpy(dtype=np.float64):
        cash *= 1.0 + float(ret)
        wins += int(ret > 0.0)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    return {"pnl": float((cash - 1.0) * 100.0), "mdd": float(mdd * 100.0), "trades": int(len(ledger)), "wr": float(wins / len(ledger))}


@dataclass
class Candidate:
    asset: str
    component: str
    local_i: int
    timestamp: pd.Timestamp
    side: int
    margin: float
    leverage: float
    notional: float
    take_profit: float
    stop_loss: float


@dataclass
class Position:
    candidate: Candidate
    entry_signal_i: int
    entry_i: int
    entry_timestamp: pd.Timestamp
    entry_price: float
    entry_equity: float
    side: int
    margin: float
    leverage: float
    notional: float
    take_profit: float
    stop_loss: float
    mfe: float = 0.0
    mae: float = 0.0


def _arrays(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    return {c: pd.to_numeric(frame[c], errors="raise").to_numpy(dtype=np.float64) for c in ("open", "high", "low", "close")}


def _load_policy() -> proto.QPolicy:
    report = json.loads(POLICY_REPORT.read_text(encoding="utf-8"))
    return proto.QPolicy(weights=np.asarray(report["policy_weights"], dtype=np.float64))


def _eth_components(split: str, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], tuple[float, float]]:
    if split == "validation":
        frame = eth_valmod.load_val_frame()
        components = {}
        for name, cfg in eth_retest.COMPONENTS.items():
            pred = pd.read_csv(eth_valmod.VAL_PRED[name])
            pred = pred.rename(columns={c: c.replace("_expertdq_oof_", "_expertdq_") for c in pred.columns})
            pred["timestamp"] = pd.to_datetime(pred["timestamp"])
            pred = pred[(pred["timestamp"] >= eth_valmod.START) & (pred["timestamp"] <= eth_valmod.END)].reset_index(drop=True)
            common = frame["timestamp"].isin(pred["timestamp"])
            frame_c = frame[common].reset_index(drop=True) if not common.all() else frame
            pred = pred[pred["timestamp"].isin(frame_c["timestamp"])].reset_index(drop=True)
            tmp_pred = OUT_DIR / f"_eth_val_{name}_aligned.csv"
            tmp_pred.parent.mkdir(parents=True, exist_ok=True)
            pred.to_csv(tmp_pred, index=False)
            components[name] = eth_greedy.prepare_component(frame_c, tmp_pred, cfg, device)
            components[name]["sidecar"] = eth_greedy.sidecar
            components[name]["long_scale"] = eth_greedy.SCALE_MAP[f"{name}_L"]
            components[name]["short_scale"] = eth_greedy.SCALE_MAP[f"{name}_S"]
            frame = frame_c
    else:
        frame = eth_retest.load_frame_current("2026-01-01", "2026-06-30")
        components = {}
        for name, cfg in eth_retest.COMPONENTS.items():
            pred_csv = eth_greedy.OUT_DIR / name / f"oos_predictions_{cfg['q_tag']}.csv"
            components[name] = eth_greedy.prepare_component(frame, pred_csv, cfg, device)
            components[name]["sidecar"] = eth_greedy.sidecar
            components[name]["long_scale"] = eth_greedy.SCALE_MAP[f"{name}_L"]
            components[name]["short_scale"] = eth_greedy.SCALE_MAP[f"{name}_S"]
    fee, slip = eth_greedy.omega._load_fee_slip()
    return frame, components, (float(fee), float(slip))


def _asset_component(asset: str, split: str, device: torch.device) -> tuple[pd.DataFrame, dict[str, Any], tuple[float, float]]:
    frames = asset_router._load_frames(asset)
    frame = frames["val_raw" if split == "validation" else "oos_raw"]
    cfg = SOL_CFG if asset == "sol" else BTC_CFG
    comp_name = "zig075" if asset == "sol" else "h48qual"
    comp = asset_router._prepare_component(asset, "validation" if split == "validation" else "oos", frame, cfg, device=device)
    omega = importlib.import_module(f"train_eval_omega1_2_tabm_diffusion_risk_{asset}_{asset_router.ASSET_DATES[asset]}")
    fee, slip = omega._load_fee_slip()
    return frame, {comp_name: comp}, (float(fee), float(slip))


def _build_world(split: str, device: torch.device) -> dict[str, Any]:
    eth_frame, eth_comps, eth_fee = _eth_components(split, device)
    sol_frame, sol_comps, sol_fee = _asset_component("sol", split, device)
    btc_frame, btc_comps, btc_fee = _asset_component("btc", split, device)
    world = {
        "eth": {"frame": eth_frame, "components": eth_comps, "fee_slip": eth_fee, "arrays": _arrays(eth_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(eth_frame["timestamp"])}},
        "sol": {"frame": sol_frame, "components": sol_comps, "fee_slip": sol_fee, "arrays": _arrays(sol_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(sol_frame["timestamp"])}},
        "btc": {"frame": btc_frame, "components": btc_comps, "fee_slip": btc_fee, "arrays": _arrays(btc_frame), "ts_to_i": {pd.Timestamp(ts): i for i, ts in enumerate(btc_frame["timestamp"])}},
    }
    common = set(world["eth"]["ts_to_i"]).intersection(world["sol"]["ts_to_i"]).intersection(world["btc"]["ts_to_i"])
    world["timestamps"] = sorted(common)
    return world


def _active_side(comp: dict[str, Any], i: int) -> int:
    if i < 0 or i >= len(comp["dec"]):
        return 0
    return int(comp["dec"]["side"].iloc[i])


def _candidate_for_asset(world: dict[str, Any], asset: str, ts: pd.Timestamp) -> Candidate | None:
    aw = world[asset]
    i = aw["ts_to_i"].get(ts)
    if i is None or i >= len(aw["frame"]) - 2:
        return None
    if float(aw["frame"]["ou_halflife"].iloc[i]) <= DURATION_THRESHOLDS[asset]:
        return None
    priority = ("h48qual", "zig075") if asset == "eth" else tuple(aw["components"].keys())
    for comp_name in priority:
        if comp_name not in aw["components"]:
            continue
        comp = aw["components"][comp_name]
        side = _active_side(comp, i)
        if side == 0:
            continue
        margin = float(comp["margin"][i])
        leverage = float(comp["leverage"][i])
        if margin <= 0.0 or leverage <= 0.0:
            continue
        scale = float(comp["long_scale"] if side > 0 else comp["short_scale"])
        leverage = min(leverage * scale, LEVERAGE_CAP)
        notional = min(margin * leverage, NOTIONAL_CAP)
        leverage = notional / max(margin, 1e-12)
        if notional <= 0.0:
            continue
        row = comp["dec"].iloc[i]
        return Candidate(
            asset=asset,
            component=comp_name,
            local_i=int(i),
            timestamp=ts,
            side=int(side),
            margin=float(margin),
            leverage=float(leverage),
            notional=float(notional),
            take_profit=float(row.get("take_profit", 0.0) or 0.0),
            stop_loss=float(row.get("stop_loss", 0.0) or 0.0),
        )
    return None


def _candidate_features(c: Candidate, ts: pd.Timestamp, lag_returns: dict[str, list[float]]) -> pd.DataFrame:
    lags = lag_returns[c.asset]
    row = {
        "is_eth": float(c.asset == "eth"),
        "is_sol": float(c.asset == "sol"),
        "is_btc": float(c.asset == "btc"),
        "is_long": float(c.side > 0),
        "is_short": float(c.side < 0),
        "notional": float(c.notional),
        "margin_fraction": float(c.margin),
        "leverage": float(c.leverage),
        "ou_halflife": np.nan,
        "asset_score": float(ASSET_SCORES[c.asset]),
        "ret_lag1_asset": float(lags[-1]) if lags else 0.0,
        "ret_lag3_asset": float(sum(lags[-3:])) if lags else 0.0,
        "hour": float(ts.hour),
        "month": float(ts.month),
    }
    return pd.DataFrame([row], columns=proto.FEATURE_COLS)


def _open_position(world: dict[str, Any], c: Candidate, cash: float) -> tuple[Position, float]:
    aw = world[c.asset]
    fee, slip = aw["fee_slip"]
    fee_eff = fee * COST_MULT
    slip_eff = slip * COST_MULT
    arrays = aw["arrays"]
    entry_i = min(c.local_i + 1, len(aw["frame"]) - 1)
    entry_px = arrays["open"][entry_i] * (1 + slip_eff if c.side > 0 else 1 - slip_eff)
    pos = Position(
        candidate=c,
        entry_signal_i=c.local_i,
        entry_i=entry_i,
        entry_timestamp=pd.Timestamp(aw["frame"]["timestamp"].iloc[c.local_i]),
        entry_price=float(entry_px),
        entry_equity=float(cash),
        side=c.side,
        margin=c.margin,
        leverage=c.leverage,
        notional=c.notional,
        take_profit=c.take_profit,
        stop_loss=c.stop_loss,
    )
    cash = cash - cash * fee_eff * c.notional
    return pos, cash


def _try_close(world: dict[str, Any], pos: Position, ts: pd.Timestamp, cash: float, device: torch.device) -> tuple[Position | None, float, dict[str, Any] | None, float]:
    aw = world[pos.candidate.asset]
    i = aw["ts_to_i"].get(ts)
    if i is None:
        return pos, cash, None, cash
    comp = aw["components"][pos.candidate.component]
    arrays = aw["arrays"]
    fee, slip = aw["fee_slip"]
    fee_eff = fee * COST_MULT
    slip_eff = slip * COST_MULT
    close_px = arrays["close"][i] * (1 - slip_eff if pos.side > 0 else 1 + slip_eff)
    move = (close_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - close_px) / max(pos.entry_price, 1e-12)
    pos.mfe = max(pos.mfe, move)
    pos.mae = min(pos.mae, move)
    reason = ""
    exit_prob = 0.0
    if pos.take_profit > 0.0 and move >= pos.take_profit:
        reason = "take_profit"
    elif pos.stop_loss > 0.0 and move <= -abs(pos.stop_loss):
        reason = "stop_loss"
    else:
        hold = max(int(i) - int(pos.entry_i), 0)
        giveback = (pos.mfe - move) / max(abs(pos.mfe), 1e-8) if pos.mfe > 0.0 else 0.0
        expert = hard.EXPERT_NAMES[int(comp["route"][i])]
        sidecar = comp["sidecar"]
        exit_prob = float(sidecar._predict_exit_prob_one(
            comp["base_np"],
            comp["exit_runtime"],
            comp["pos_idx"],
            row_i=int(i),
            expert=expert,
            pos_values=[
                float(pos.side), float(hold), float(move), float(pos.mfe), float(pos.mae),
                float(np.clip(giveback, 0.0, 10.0)), float(pos.take_profit - move),
                float(move + abs(pos.stop_loss)), float(pos.notional), float(pos.leverage),
                float(pos.notional * pos.leverage), float(pos.take_profit), float(pos.stop_loss),
            ],
            device=device,
        ))
        if exit_prob >= float(comp["exit_threshold"]):
            reason = "exit_head"
    equity_mark = cash * (1.0 + move * pos.notional)
    if not reason:
        return pos, cash, None, equity_mark
    before = cash
    cash = cash * (1.0 + move * pos.notional)
    cash -= before * fee_eff * pos.notional
    trade_return = cash / max(pos.entry_equity, 1e-12) - 1.0
    row = {
        "asset": pos.candidate.asset,
        "component": pos.candidate.component,
        "entry_signal_i": int(pos.entry_signal_i),
        "entry_i": int(pos.entry_i),
        "exit_i": int(i),
        "entry_timestamp": str(pos.entry_timestamp),
        "exit_timestamp": str(aw["frame"]["timestamp"].iloc[i]),
        "side": int(pos.side),
        "reason": reason,
        "win": int(cash > pos.entry_equity),
        "raw_exit_price_move": float(move),
        "mfe_price_move": float(pos.mfe),
        "mae_price_move": float(pos.mae),
        "trade_return": float(trade_return),
        "net_per_notional": float(trade_return / max(pos.notional, 1e-12)),
        "notional": float(pos.notional),
        "margin_fraction": float(pos.margin),
        "leverage": float(pos.leverage),
        "exit_prob": float(exit_prob),
        "take_profit": float(pos.take_profit),
        "stop_loss": float(pos.stop_loss),
    }
    return None, cash, row, cash


def _force_close(world: dict[str, Any], pos: Position, cash: float) -> tuple[float, dict[str, Any]]:
    aw = world[pos.candidate.asset]
    fee, slip = aw["fee_slip"]
    fee_eff = fee * COST_MULT
    slip_eff = slip * COST_MULT
    arrays = aw["arrays"]
    i = len(aw["frame"]) - 1
    exit_px = arrays["close"][i] * (1 - slip_eff if pos.side > 0 else 1 + slip_eff)
    move = (exit_px - pos.entry_price) / max(pos.entry_price, 1e-12) if pos.side > 0 else (pos.entry_price - exit_px) / max(pos.entry_price, 1e-12)
    before = cash
    cash = cash * (1.0 + move * pos.notional)
    cash -= before * fee_eff * pos.notional
    trade_return = cash / max(pos.entry_equity, 1e-12) - 1.0
    row = {
        "asset": pos.candidate.asset, "component": pos.candidate.component,
        "entry_signal_i": int(pos.entry_signal_i), "entry_i": int(pos.entry_i), "exit_i": int(i),
        "entry_timestamp": str(pos.entry_timestamp), "exit_timestamp": str(aw["frame"]["timestamp"].iloc[i]),
        "side": int(pos.side), "reason": "forced_end", "win": int(cash > pos.entry_equity),
        "raw_exit_price_move": float(move), "mfe_price_move": float(pos.mfe), "mae_price_move": float(pos.mae),
        "trade_return": float(trade_return), "net_per_notional": float(trade_return / max(pos.notional, 1e-12)),
        "notional": float(pos.notional), "margin_fraction": float(pos.margin), "leverage": float(pos.leverage),
        "exit_prob": 0.0, "take_profit": float(pos.take_profit), "stop_loss": float(pos.stop_loss),
    }
    return cash, row


def _replay(world: dict[str, Any], policy: proto.QPolicy | None, *, take_all: bool, device: torch.device) -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: Position | None = None
    rows: list[dict[str, Any]] = []
    skips = 0
    candidate_events = 0
    lag_returns = {"eth": [], "sol": [], "btc": []}
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = _try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                lag_returns[closed["asset"]].append(float(closed["trade_return"]))
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        candidates = [c for c in (_candidate_for_asset(world, asset, ts) for asset in ("eth", "sol", "btc")) if c is not None]
        if not candidates:
            continue
        candidate_events += 1
        candidates.sort(key=lambda c: (ASSET_SCORES[c.asset], c.notional), reverse=True)
        top = candidates[0]
        take = True
        q_skip = q_take = np.nan
        if not take_all:
            feat = _candidate_features(top, ts, lag_returns)
            feat["ou_halflife"] = float(world[top.asset]["frame"]["ou_halflife"].iloc[top.local_i])
            q_skip = float(policy.q(feat, 0)[0]) if policy else 0.0
            q_take = float(policy.q(feat, 1)[0]) if policy else 0.0
            take = q_take > q_skip
        if not take:
            skips += 1
            continue
        position, cash = _open_position(world, top, cash)
        position.candidate = Candidate(**{**position.candidate.__dict__})
        # keep q diagnostics on the position candidate via dynamic attrs
        setattr(position, "q_skip", q_skip)
        setattr(position, "q_take", q_take)
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = _force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = _compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    diag = {"candidate_events": int(candidate_events), "skips": int(skips), "final_cash": float(cash)}
    return metrics, ledger, diag


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio RL Gate 2-Action Native Replay - 2026-07-08",
        "",
        "Native evaluation of the frozen 2-action RL gate. Replay does not read saved trade ledgers or saved exit timestamps.",
        "",
        "| policy | split | PnL | MDD | MTM MDD | trades | WR |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for policy_name in ("rule_take_all", "rl_gate"):
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            m = report["results"][policy_name][split]
            lines.append(f"| {policy_name} | {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} |")
    lines.extend([
        "",
        "Caveat: the policy weights were trained by the earlier event-level prototype, so this is not a complete promotion-grade retrain.",
        "",
    ])
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = eth_retest.DEVICE
    policy = _load_policy()
    results: dict[str, dict[str, Any]] = {"rule_take_all": {}, "rl_gate": {}}
    diagnostics: dict[str, Any] = {}
    ledgers: dict[tuple[str, str], pd.DataFrame] = {}
    for split in ("validation", "oos"):
        print(f"stage=build_world split={split}", flush=True)
        world = _build_world(split, device)
        for policy_name, take_all in (("rule_take_all", True), ("rl_gate", False)):
            print(f"stage=replay split={split} policy={policy_name}", flush=True)
            metrics, ledger, diag = _replay(world, policy, take_all=take_all, device=device)
            key = "validation" if split == "validation" else "oos_extended"
            results[policy_name][key] = metrics
            diagnostics[f"{policy_name}_{key}"] = diag
            ledgers[(policy_name, key)] = ledger
            if split == "oos":
                q1 = ledger.loc[pd.to_datetime(ledger["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True) if not ledger.empty else ledger
                q1_metrics = _compound_metrics(q1)
                q1_metrics["mark_to_market_mdd"] = q1_metrics["mdd"]
                results[policy_name]["oos_frozen_q1_2026"] = q1_metrics
                ledgers[(policy_name, "oos_frozen_q1_2026")] = q1
    for (policy_name, split), ledger in ledgers.items():
        ledger.to_csv(OUT_DIR / f"{split}_{policy_name}_ledger.csv", index=False)
    report = {
        "method": "portfolio_rl_gate_2action_native_bar_by_bar_replay",
        "policy_source": str(POLICY_REPORT),
        "policy_training_used_event_ledger": True,
        "training_data": "frozen_policy_from_validation_event_level_prototype",
        "evaluation_data": "native_bar_by_bar",
        "action_space": {"0": "SKIP", "1": "TAKE_TOP"},
        "asset_scores_validation_only": ASSET_SCORES,
        "results": results,
        "diagnostics": diagnostics,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "results": results}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
