#!/usr/bin/env python3
"""Causal native TAKE-only portfolio asset switch router.

This removes the degenerate SKIP action from the online router. If one or more
asset candidates exist at a timestamp, the router must choose exactly one valid
asset action:

- TAKE_ETH
- TAKE_SOL
- TAKE_BTC

The model receives all ETH/SOL/BTC candidate features and updates only from
trades closed before the current decision timestamp.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import replay_portfolio_rl_gate_2action_native_20260708 as native

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_online_asset_switch_native_20260709"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_online_asset_switch_native_20260709.md"
ASSETS = ("eth", "sol", "btc")
ACTIONS = ("TAKE_ETH", "TAKE_SOL", "TAKE_BTC")


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


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * float(l2)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


def _candidate_map(world: dict[str, Any], ts: pd.Timestamp) -> dict[str, native.Candidate | None]:
    return {asset: native._candidate_for_asset(world, asset, ts) for asset in ASSETS}


def _state_features(candidates: dict[str, native.Candidate | None], ts: pd.Timestamp, lag_returns: dict[str, list[float]], drawdown: float, closed_count: int) -> np.ndarray:
    vals: list[float] = [1.0, float(drawdown), float(closed_count) / 100.0, np.sin(2 * np.pi * ts.hour / 24.0), (float(ts.month) - 6.5) / 6.0]
    for asset in ASSETS:
        c = candidates.get(asset)
        lags = lag_returns[asset]
        if c is None:
            vals.extend([0.0] * 12)
            continue
        vals.extend(
            [
                1.0,
                float(c.side > 0),
                float(c.side < 0),
                float(c.notional),
                float(c.margin),
                float(c.leverage),
                float(c.take_profit),
                float(c.stop_loss),
                float(native.ASSET_SCORES[asset]),
                float(lags[-1]) if lags else 0.0,
                float(sum(lags[-3:])) if lags else 0.0,
                float(len(lags)) / 100.0,
            ]
        )
    return np.asarray(vals, dtype=np.float64)


def _action_features(state: np.ndarray, action_idx: int) -> np.ndarray:
    onehot = np.zeros(len(ACTIONS), dtype=np.float64)
    onehot[int(action_idx)] = 1.0
    return np.concatenate([state, onehot, state * onehot[int(action_idx)]])


class OnlineAssetSwitch:
    def __init__(self, *, min_samples: int = 8, l2: float = 25.0, explore_until: int = 8) -> None:
        self.min_samples = int(min_samples)
        self.explore_until = int(explore_until)
        self.l2 = float(l2)
        self.xs: list[np.ndarray] = []
        self.ys: list[float] = []
        self.weights: np.ndarray | None = None

    def update(self, state: np.ndarray, action_idx: int, trade_return: float, notional: float) -> None:
        # Keep reward close to realized return. Heavy penalties made earlier
        # gates collapse to SKIP-like behavior; this router has no SKIP action.
        tail_penalty = max(0.0, -float(trade_return) - 0.04)
        reward = float(trade_return) - 0.10 * tail_penalty - 0.0005 * float(notional)
        self.xs.append(_action_features(state, action_idx))
        self.ys.append(float(reward))
        if len(self.ys) >= self.min_samples:
            self.weights = _fit_ridge(np.vstack(self.xs), np.asarray(self.ys, dtype=np.float64), self.l2)

    def q(self, state: np.ndarray, action_idx: int) -> float:
        if self.weights is None:
            return 0.0
        return float(_action_features(state, action_idx) @ self.weights)

    def choose(self, state: np.ndarray, valid_actions: list[int]) -> tuple[int, dict[str, float]]:
        if self.weights is None or len(self.ys) < self.explore_until:
            best = max(valid_actions, key=lambda a: native.ASSET_SCORES[ASSETS[a]])
            return int(best), {ACTIONS[a]: 0.0 for a in valid_actions}
        qs = {ACTIONS[a]: self.q(state, a) for a in valid_actions}
        best = max(valid_actions, key=lambda a: qs[ACTIONS[a]])
        return int(best), qs


def _replay(world: dict[str, Any], *, device: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    model = OnlineAssetSwitch()
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: native.Position | None = None
    position_state: np.ndarray | None = None
    position_action: int | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    lag_returns = {asset: [] for asset in ASSETS}
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                rows.append(closed)
                lag_returns[closed["asset"]].append(float(closed["trade_return"]))
                if position_state is not None and position_action is not None:
                    model.update(position_state, position_action, float(closed["trade_return"]), float(closed["notional"]))
                position_state = None
                position_action = None
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        candidates = _candidate_map(world, ts)
        valid_actions = [i for i, asset in enumerate(ASSETS) if candidates[asset] is not None]
        if not valid_actions:
            continue
        drawdown = cash / max(peak, 1e-12) - 1.0
        state = _state_features(candidates, ts, lag_returns, drawdown, len(rows))
        action_idx, qs = model.choose(state, valid_actions)
        asset = ASSETS[action_idx]
        c = candidates[asset]
        if c is None:
            raise RuntimeError(f"invalid masked action selected: {ACTIONS[action_idx]}")
        decisions.append(
            {
                "timestamp": ts,
                "action": ACTIONS[action_idx],
                "valid_actions": ",".join(ACTIONS[a] for a in valid_actions),
                "closed_samples_before_decision": int(len(model.ys)),
                **{f"q_{k}": v for k, v in qs.items()},
            }
        )
        position, cash = native._open_position(world, c, cash)
        position_state = state
        position_action = action_idx
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["closed_samples_final"] = int(len(model.ys))
    return metrics, ledger, pd.DataFrame(decisions)


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Online Asset Switch Native - 2026-07-09",
        "",
        "TAKE-only causal native router. State receives all ETH/SOL/BTC candidates; action selects one valid asset.",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        m = report["results"][split]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m.get('decisions', 0)} |")
    lines.append("")
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = native.eth_retest.DEVICE
    results: dict[str, Any] = {}
    for split in ("validation", "oos"):
        print(f"stage=build_world split={split}", flush=True)
        world = native._build_world("validation" if split == "validation" else "oos", device)
        print(f"stage=replay_asset_switch split={split}", flush=True)
        metrics, ledger, decisions = _replay(world, device=device)
        key = "validation" if split == "validation" else "oos_extended"
        results[key] = metrics
        ledger.to_csv(OUT_DIR / f"{key}_ledger.csv", index=False)
        decisions.to_csv(OUT_DIR / f"{key}_decisions.csv", index=False)
        if split == "oos":
            q1 = ledger.loc[pd.to_datetime(ledger["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True) if not ledger.empty else ledger
            q1m = native._compound_metrics(q1)
            q1m["mark_to_market_mdd"] = q1m["mdd"]
            q1m["decisions"] = int((decisions["timestamp"] < pd.Timestamp("2026-04-01")).sum()) if not decisions.empty else 0
            results["oos_frozen_q1_2026"] = q1m
            q1.to_csv(OUT_DIR / "oos_frozen_q1_2026_ledger.csv", index=False)
    report = {
        "method": "portfolio_online_asset_switch_native",
        "training_mode": "causal_online_from_previously_closed_trades_only",
        "action_space": {str(i): a for i, a in enumerate(ACTIONS)},
        "state_receives_all_asset_candidates": True,
        "action_masking": True,
        "skip_action_removed": True,
        "results": results,
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "uses_only_past_closed_trades_for_learning": True,
        "promotion_grade": False,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "results": results}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
