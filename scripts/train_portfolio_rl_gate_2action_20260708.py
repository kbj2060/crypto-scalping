#!/usr/bin/env python3
"""2-action offline RL portfolio gate for ETH/SOL/BTC Omega4.6.1 candidates.

The router action space is deliberately minimal:
- 0: SKIP the current rule-selected top candidate
- 1: TAKE_TOP using the candidate model's own sizing and exit

Selection/training uses validation only. OOS is evaluated once after the fixed
policy is trained.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516/portfolio_rl_gate_2action_20260708"
DOC_PATH = ROOT / "docs/model_contracts/portfolio_rl_gate_2action_20260708.md"

ETH_VAL = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_VAL.csv"
ETH_OOS = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv"
SOL_VAL = ROOT / "tmp/causal_regen_20260516/sol_val_stability_exact_20260708/validation_ledger.csv"
SOL_OOS = ROOT / "tmp/causal_regen_20260516/sol_val_stability_exact_20260708/oos_ledger.csv"
BTC_VAL = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708/validation_ledger.csv"
BTC_OOS = ROOT / "tmp/causal_regen_20260516/btc_final_scale_map_20260708/oos_ledger.csv"

ETH_FEATURES_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
ETH_FEATURES_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
SOL_FEATURES_2025 = ROOT / "data/splits/year_oos/sol_features_2025.csv"
SOL_FEATURES_2026 = ROOT / "data/splits/year_oos/sol_features_2026.csv"
BTC_FEATURES_2025 = ROOT / "data/splits/year_oos/btc_features_2025.csv"
BTC_FEATURES_2026 = ROOT / "data/splits/year_oos/btc_features_2026.csv"

DURATION_THRESHOLDS = {
    "eth": 0.005417,
    "sol": 0.0055208323,
    "btc": 0.00541154875,
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
    return {
        "pnl": float((cash - 1.0) * 100.0),
        "mdd": float(mdd * 100.0),
        "trades": int(len(ledger)),
        "wr": float(wins / len(ledger)),
    }


def _load_asset(path: Path, features_path: Path, asset: str, component: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["entry_timestamp", "exit_timestamp"])
    if "ou_halflife" not in df.columns:
        feats = pd.read_csv(features_path, usecols=["timestamp", "ou_halflife"], parse_dates=["timestamp"])
        df = df.merge(feats.rename(columns={"timestamp": "entry_timestamp"}), on="entry_timestamp", how="left", validate="one_to_one")
    if df["ou_halflife"].isna().any():
        raise RuntimeError(f"{path}: ou_halflife merge produced NaN")
    df = df.loc[df["ou_halflife"] > DURATION_THRESHOLDS[asset]].reset_index(drop=True)
    df["asset"] = asset
    df["component"] = component
    if "source_component" in df.columns:
        df["component"] = df["source_component"].astype(str)
    keep = [
        "asset", "component", "entry_timestamp", "exit_timestamp", "side", "reason", "win",
        "trade_return", "notional", "margin_fraction", "leverage", "ou_halflife",
    ]
    out = df[keep].copy()
    out["side"] = pd.to_numeric(out["side"], errors="raise").astype(int)
    for col in ("trade_return", "notional", "margin_fraction", "leverage", "ou_halflife"):
        out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
    return out


def _asset_scores(val_assets: dict[str, pd.DataFrame]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for asset, df in val_assets.items():
        m = _compound_metrics(df)
        scores[asset] = float(m["pnl"]) / max(abs(float(m["mdd"])), 1.0)
    return scores


def _top_stream(assets: dict[str, pd.DataFrame], scores: dict[str, float]) -> pd.DataFrame:
    all_df = pd.concat(list(assets.values()), ignore_index=True).sort_values(["entry_timestamp", "asset"]).reset_index(drop=True)
    rows: list[pd.Series] = []
    for _, group in all_df.groupby("entry_timestamp", sort=True):
        ranked = group.copy()
        ranked["_rule_score"] = ranked["asset"].map(scores).astype(float)
        ranked = ranked.sort_values(["_rule_score", "notional"], ascending=[False, False])
        rows.append(ranked.iloc[0])
    stream = pd.DataFrame(rows).sort_values(["entry_timestamp", "asset"]).reset_index(drop=True)
    stream["event_i"] = np.arange(len(stream), dtype=int)
    stream["hour"] = stream["entry_timestamp"].dt.hour.astype(float)
    stream["month"] = stream["entry_timestamp"].dt.month.astype(float)
    stream["asset_score"] = stream["asset"].map(scores).astype(float)
    for asset in ("eth", "sol", "btc"):
        stream[f"is_{asset}"] = (stream["asset"] == asset).astype(float)
    stream["is_short"] = (stream["side"] < 0).astype(float)
    stream["is_long"] = (stream["side"] > 0).astype(float)
    stream["ret_lag1_asset"] = 0.0
    stream["ret_lag3_asset"] = 0.0
    for asset, idx in stream.groupby("asset").groups.items():
        idx_list = list(idx)
        rets = stream.loc[idx_list, "trade_return"].shift(1)
        stream.loc[idx_list, "ret_lag1_asset"] = rets.fillna(0.0).to_numpy(dtype=np.float64)
        stream.loc[idx_list, "ret_lag3_asset"] = stream.loc[idx_list, "trade_return"].shift(1).rolling(3, min_periods=1).sum().fillna(0.0).to_numpy(dtype=np.float64)
    return stream


def _next_after_exit(stream: pd.DataFrame) -> np.ndarray:
    entries = stream["entry_timestamp"].to_numpy()
    exits = stream["exit_timestamp"].to_numpy()
    out = np.empty(len(stream), dtype=int)
    for i, ts in enumerate(exits):
        out[i] = int(np.searchsorted(entries, ts, side="right"))
    return out


FEATURE_COLS = [
    "is_eth", "is_sol", "is_btc", "is_long", "is_short", "notional", "margin_fraction",
    "leverage", "ou_halflife", "asset_score", "ret_lag1_asset", "ret_lag3_asset",
    "hour", "month",
]


def _design_matrix(stream: pd.DataFrame, action: int) -> np.ndarray:
    x = stream[FEATURE_COLS].to_numpy(dtype=np.float64)
    hour = x[:, FEATURE_COLS.index("hour")]
    month = x[:, FEATURE_COLS.index("month")]
    x[:, FEATURE_COLS.index("hour")] = np.sin(2 * np.pi * hour / 24.0)
    x[:, FEATURE_COLS.index("month")] = (month - 6.5) / 6.0
    a = np.full((len(x), 1), float(action))
    return np.column_stack([np.ones(len(x)), x, a, x * float(action)])


def _fit_ridge(x: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    xtx = x.T @ x
    penalty = np.eye(xtx.shape[0]) * float(l2)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + penalty, x.T @ y)


@dataclass
class QPolicy:
    weights: np.ndarray

    def q(self, rows: pd.DataFrame, action: int) -> np.ndarray:
        return _design_matrix(rows, action) @ self.weights

    def action(self, row: pd.DataFrame) -> int:
        q_skip = float(self.q(row, 0)[0])
        q_take = float(self.q(row, 1)[0])
        return int(q_take > q_skip)


def _train_fqi(stream: pd.DataFrame, *, gamma: float = 0.97, l2: float = 1.0, iterations: int = 25) -> QPolicy:
    n = len(stream)
    next_skip = np.minimum(np.arange(n, dtype=int) + 1, n)
    next_take = _next_after_exit(stream)
    rewards_skip = np.zeros(n, dtype=np.float64)
    raw_take = stream["trade_return"].to_numpy(dtype=np.float64)
    tail_penalty = np.maximum(0.0, -raw_take - 0.03)
    rewards_take = raw_take - 0.50 * tail_penalty - 0.005 * stream["notional"].to_numpy(dtype=np.float64)
    x = np.vstack([_design_matrix(stream, 0), _design_matrix(stream, 1)])
    weights = np.zeros(x.shape[1], dtype=np.float64)
    for _ in range(int(iterations)):
        q0 = _design_matrix(stream, 0) @ weights
        q1 = _design_matrix(stream, 1) @ weights
        vmax = np.maximum(q0, q1)
        cont_skip = np.zeros(n, dtype=np.float64)
        cont_take = np.zeros(n, dtype=np.float64)
        skip_mask = next_skip < n
        take_mask = next_take < n
        cont_skip[skip_mask] = vmax[next_skip[skip_mask]]
        cont_take[take_mask] = vmax[next_take[take_mask]]
        y_skip = rewards_skip + float(gamma) * cont_skip
        y_take = rewards_take + float(gamma) * cont_take
        y = np.concatenate([y_skip, y_take])
        weights = _fit_ridge(x, y, l2=float(l2))
    return QPolicy(weights=weights)


def _replay(stream: pd.DataFrame, policy: QPolicy | None, *, take_all: bool) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    i = 0
    n = len(stream)
    next_take = _next_after_exit(stream)
    while i < n:
        row = stream.iloc[[i]]
        take = True if take_all else bool(policy and policy.action(row))
        if take:
            rec = stream.iloc[i].to_dict()
            q_skip = float(policy.q(row, 0)[0]) if policy else np.nan
            q_take = float(policy.q(row, 1)[0]) if policy else np.nan
            rec.update({"router_action": "TAKE_TOP", "q_skip": q_skip, "q_take": q_take})
            rows.append(rec)
            i = int(next_take[i])
        else:
            i += 1
    ledger = pd.DataFrame(rows)
    return _compound_metrics(ledger), ledger


def _load_split(split: str) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    if split == "validation":
        assets = {
            "eth": _load_asset(ETH_VAL, ETH_FEATURES_2025, "eth", "greedy"),
            "sol": _load_asset(SOL_VAL, SOL_FEATURES_2025, "sol", "zig075"),
            "btc": _load_asset(BTC_VAL, BTC_FEATURES_2025, "btc", "h48qual"),
        }
    elif split == "oos":
        assets = {
            "eth": _load_asset(ETH_OOS, ETH_FEATURES_2026, "eth", "greedy"),
            "sol": _load_asset(SOL_OOS, SOL_FEATURES_2026, "sol", "zig075"),
            "btc": _load_asset(BTC_OOS, BTC_FEATURES_2026, "btc", "h48qual"),
        }
    else:
        raise ValueError(split)
    return assets, {k: v.copy() for k, v in assets.items()}


def _write_doc(report: dict[str, Any]) -> None:
    lines = [
        "# Portfolio RL Gate 2-Action - 2026-07-08",
        "",
        "## Contract",
        "",
        "- Router action space: `0=SKIP`, `1=TAKE_TOP`.",
        "- Rule router first selects one top candidate per timestamp from ETH/SOL/BTC.",
        "- RL does not create entries, change exits, or change sizing.",
        "- Training uses validation only; OOS is reported once after training.",
        "- Implementation is offline Fitted Q Iteration with a ridge linear Q-function.",
        "",
        "## Results",
        "",
        "| policy | split | PnL | MDD | trades | WR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for policy_name in ("rule_take_all", "rl_gate"):
        for split in ("validation", "oos_extended", "oos_frozen_q1_2026"):
            m = report["results"][policy_name][split]
            lines.append(f"| {policy_name} | {split} | {m['pnl']:.2f}% | {m['mdd']:.2f}% | {m['trades']} | {m['wr']:.2%} |")
    lines.extend([
        "",
        "## Notes",
        "",
        f"- Validation events: `{report['event_counts']['validation_top_events']}`.",
        f"- OOS events: `{report['event_counts']['oos_top_events']}`.",
        f"- RL validation selected trades: `{report['results']['rl_gate']['validation']['trades']}`.",
        f"- RL OOS selected trades: `{report['results']['rl_gate']['oos_extended']['trades']}`.",
        "",
        "This is a research prototype, not a promotion-grade live router. The dataset is small, so the policy must be red-teamed before any live use.",
        "",
    ])
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    DOC_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    val_assets, _ = _load_split("validation")
    scores = _asset_scores(val_assets)
    val_stream = _top_stream(val_assets, scores)
    oos_assets, _ = _load_split("oos")
    oos_stream = _top_stream(oos_assets, scores)

    policy = _train_fqi(val_stream)
    val_rule_m, val_rule_l = _replay(val_stream, None, take_all=True)
    oos_rule_m, oos_rule_l = _replay(oos_stream, None, take_all=True)
    val_rl_m, val_rl_l = _replay(val_stream, policy, take_all=False)
    oos_rl_m, oos_rl_l = _replay(oos_stream, policy, take_all=False)
    oos_q1_stream = oos_stream.loc[oos_stream["entry_timestamp"] < pd.Timestamp("2026-04-01")].reset_index(drop=True)
    oos_rule_q1_m, _ = _replay(oos_q1_stream, None, take_all=True)
    oos_rl_q1_m, oos_rl_q1_l = _replay(oos_q1_stream, policy, take_all=False)

    report = {
        "method": "portfolio_rl_gate_2action_fitted_q_iteration",
        "training_data": "validation_only",
        "oos_usage": "reported_once_after_policy_training",
        "action_space": {"0": "SKIP", "1": "TAKE_TOP"},
        "asset_scores_validation_only": scores,
        "features": FEATURE_COLS,
        "rl_algorithm": {"name": "fitted_q_iteration_ridge_linear", "gamma": 0.97, "l2": 1.0, "iterations": 25},
        "event_counts": {"validation_top_events": int(len(val_stream)), "oos_top_events": int(len(oos_stream))},
        "results": {
            "rule_take_all": {
                "validation": val_rule_m,
                "oos_extended": oos_rule_m,
                "oos_frozen_q1_2026": oos_rule_q1_m,
            },
            "rl_gate": {
                "validation": val_rl_m,
                "oos_extended": oos_rl_m,
                "oos_frozen_q1_2026": oos_rl_q1_m,
            },
        },
        "policy_weights": policy.weights,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": True,
        "saved_parent_exit_timestamps_used": True,
        "future_rows_used_for_entry": False,
        "promotion_grade": False,
    }
    val_stream.to_csv(OUT_DIR / "validation_top_stream.csv", index=False)
    oos_stream.to_csv(OUT_DIR / "oos_top_stream.csv", index=False)
    val_rule_l.to_csv(OUT_DIR / "validation_rule_take_all_ledger.csv", index=False)
    oos_rule_l.to_csv(OUT_DIR / "oos_rule_take_all_ledger.csv", index=False)
    val_rl_l.to_csv(OUT_DIR / "validation_rl_gate_ledger.csv", index=False)
    oos_rl_l.to_csv(OUT_DIR / "oos_rl_gate_ledger.csv", index=False)
    oos_rl_q1_l.to_csv(OUT_DIR / "oos_q1_rl_gate_ledger.csv", index=False)
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_doc(report)
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "doc": str(DOC_PATH), "results": report["results"]}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
