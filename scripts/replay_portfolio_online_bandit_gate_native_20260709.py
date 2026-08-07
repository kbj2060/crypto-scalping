#!/usr/bin/env python3
"""Causal online-style 2-action portfolio gate in the native environment.

This is intentionally not offline RL. It never trains on a completed validation
ledger ahead of time. During replay, the model can update only from trades that
have already closed before the current decision timestamp.

Action space:
- SKIP
- TAKE_TOP

Default action is TAKE_TOP until enough closed trades exist. The learned gate
is a conservative ridge contextual bandit: it skips only when predicted
risk-adjusted reward is below a negative margin.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import replay_portfolio_rl_gate_2action_native_20260708 as native
import train_portfolio_rl_gate_2action_20260708 as proto

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_online_bandit_gate_native_20260709"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_online_bandit_gate_native_20260709.md"


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


def _features(c: native.Candidate, ts: pd.Timestamp, world: dict[str, Any], lag_returns: dict[str, list[float]]) -> pd.DataFrame:
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
        "ou_halflife": float(world[c.asset]["frame"]["ou_halflife"].iloc[c.local_i]),
        "asset_score": float(native.ASSET_SCORES[c.asset]),
        "ret_lag1_asset": float(lags[-1]) if lags else 0.0,
        "ret_lag3_asset": float(sum(lags[-3:])) if lags else 0.0,
        "hour": float(ts.hour),
        "month": float(ts.month),
    }
    return pd.DataFrame([row], columns=proto.FEATURE_COLS)


def _x_take(feat: pd.DataFrame) -> np.ndarray:
    # Use the same featurization as the prior 2-action policy, taking the TAKE
    # action row. This gives action-interaction terms while fitting only a
    # reward model for TAKE outcomes.
    return proto._design_matrix(feat, 1)


class OnlineBandit:
    def __init__(self, *, min_samples: int, l2: float, skip_margin: float) -> None:
        self.min_samples = int(min_samples)
        self.l2 = float(l2)
        self.skip_margin = float(skip_margin)
        self.xs: list[np.ndarray] = []
        self.ys: list[float] = []
        self.weights: np.ndarray | None = None

    def update(self, feat: pd.DataFrame, trade_return: float, notional: float) -> None:
        tail_penalty = max(0.0, -float(trade_return) - 0.03)
        reward = float(trade_return) - 0.50 * tail_penalty - 0.005 * float(notional)
        self.xs.append(_x_take(feat)[0])
        self.ys.append(float(reward))
        if len(self.ys) >= self.min_samples:
            self.weights = _fit_ridge(np.vstack(self.xs), np.asarray(self.ys, dtype=np.float64), self.l2)

    def predict(self, feat: pd.DataFrame) -> float:
        if self.weights is None:
            return 0.0
        return float(_x_take(feat)[0] @ self.weights)

    def take(self, feat: pd.DataFrame) -> tuple[bool, float]:
        pred = self.predict(feat)
        if self.weights is None:
            return True, pred
        return bool(pred >= self.skip_margin), pred


def _select_top(world: dict[str, Any], ts: pd.Timestamp) -> native.Candidate | None:
    candidates = [c for c in (native._candidate_for_asset(world, asset, ts) for asset in ("eth", "sol", "btc")) if c is not None]
    if not candidates:
        return None
    candidates.sort(key=lambda c: (native.ASSET_SCORES[c.asset], c.notional), reverse=True)
    return candidates[0]


def _replay(world: dict[str, Any], *, min_samples: int, l2: float, skip_margin: float, device: Any) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    learner = OnlineBandit(min_samples=min_samples, l2=l2, skip_margin=skip_margin)
    cash = 1.0
    peak = 1.0
    mdd = 0.0
    position: native.Position | None = None
    position_feat: pd.DataFrame | None = None
    rows: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []
    lag_returns = {"eth": [], "sol": [], "btc": []}
    for ts in world["timestamps"]:
        if position is not None:
            position, cash, closed, mark_equity = native._try_close(world, position, ts, cash, device)
            peak = max(peak, mark_equity)
            mdd = min(mdd, mark_equity / max(peak, 1e-12) - 1.0)
            if closed is not None:
                closed["learned_from_past_only"] = True
                rows.append(closed)
                lag_returns[closed["asset"]].append(float(closed["trade_return"]))
                if position_feat is not None:
                    learner.update(position_feat, float(closed["trade_return"]), float(closed["notional"]))
                position_feat = None
                peak = max(peak, cash)
                mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
            continue
        top = _select_top(world, ts)
        if top is None:
            continue
        feat = _features(top, ts, world, lag_returns)
        take, pred = learner.take(feat)
        decisions.append(
            {
                "timestamp": ts,
                "asset": top.asset,
                "component": top.component,
                "side": int(top.side),
                "action": "TAKE_TOP" if take else "SKIP",
                "pred_reward": float(pred),
                "closed_samples_before_decision": int(len(learner.ys)),
                "skip_margin": float(skip_margin),
            }
        )
        if not take:
            continue
        position, cash = native._open_position(world, top, cash)
        position_feat = feat
        peak = max(peak, cash)
        mdd = min(mdd, cash / max(peak, 1e-12) - 1.0)
    if position is not None:
        cash, closed = native._force_close(world, position, cash)
        closed["learned_from_past_only"] = True
        rows.append(closed)
    ledger = pd.DataFrame(rows)
    metrics = native._compound_metrics(ledger)
    metrics["mark_to_market_mdd"] = float(mdd * 100.0)
    metrics["decisions"] = int(len(decisions))
    metrics["skips"] = int(sum(d["action"] == "SKIP" for d in decisions))
    metrics["closed_samples_final"] = int(len(learner.ys))
    return metrics, ledger, pd.DataFrame(decisions)


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio Online Bandit Gate Native - 2026-07-09",
        "",
        "Causal online-style gate. Each decision can learn only from trades closed before that timestamp.",
        "",
        "| split | PnL | MDD | MTM MDD | trades | WR | decisions | skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
        m = report["results"][split]
        lines.append(f"| {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['mark_to_market_mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} | {m.get('decisions', 0)} | {m.get('skips', 0)} |")
    lines.extend([
        "",
        "Paper-informed simplification: this uses a conservative contextual bandit rather than DT/CQL/IQL because the action space is binary and the available trade count is small.",
        "",
        "HF papers referenced: Contextual Conservative Q-Learning (2301.01298), IQL (2110.06169), Decision Transformer comparison (2305.14550), PAC-Bayesian Offline Contextual Bandits (2210.13132).",
        "",
    ])
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = native.eth_retest.DEVICE
    config = {"min_samples": 8, "l2": 10.0, "skip_margin": -0.015}
    results: dict[str, Any] = {}
    for split in ("validation", "oos"):
        print(f"stage=build_world split={split}", flush=True)
        world = native._build_world("validation" if split == "validation" else "oos", device)
        print(f"stage=online_bandit_replay split={split}", flush=True)
        metrics, ledger, decisions = _replay(world, device=device, **config)
        key = "validation" if split == "validation" else "oos_extended"
        results[key] = metrics
        ledger.to_csv(OUT_DIR / f"{key}_ledger.csv", index=False)
        decisions.to_csv(OUT_DIR / f"{key}_decisions.csv", index=False)
        if split == "oos":
            q1 = ledger.loc[pd.to_datetime(ledger["entry_timestamp"]) < pd.Timestamp("2026-04-01")].reset_index(drop=True) if not ledger.empty else ledger
            q1m = native._compound_metrics(q1)
            q1m["mark_to_market_mdd"] = q1m["mdd"]
            q1m["decisions"] = int((decisions["timestamp"] < pd.Timestamp("2026-04-01")).sum()) if not decisions.empty else 0
            q1m["skips"] = int(((decisions["timestamp"] < pd.Timestamp("2026-04-01")) & (decisions["action"] == "SKIP")).sum()) if not decisions.empty else 0
            results["oos_frozen_q1_2026"] = q1m
            q1.to_csv(OUT_DIR / "oos_frozen_q1_2026_ledger.csv", index=False)
    report = {
        "method": "portfolio_online_contextual_bandit_gate_native",
        "training_mode": "causal_online_from_previously_closed_trades_only",
        "paper_guidance": [
            {"paper": "Contextual Conservative Q-Learning for Offline Reinforcement Learning", "hf": "https://hf.co/papers/2301.01298"},
            {"paper": "Offline Reinforcement Learning with Implicit Q-Learning", "hf": "https://hf.co/papers/2110.06169"},
            {"paper": "When should we prefer Decision Transformers for Offline Reinforcement Learning?", "hf": "https://hf.co/papers/2305.14550"},
            {"paper": "PAC-Bayesian Offline Contextual Bandits With Guarantees", "hf": "https://hf.co/papers/2210.13132"},
        ],
        "config": config,
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
