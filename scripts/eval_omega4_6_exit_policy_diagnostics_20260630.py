#!/usr/bin/env python3
"""Validation-only exit-policy diagnostics for Omega 4.6."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE_REPORT = ROOT / (
    "tmp/causal_regen_20260516/omega4_6_plus_t12_nohold_risk1_20260630/runtime_contract.json"
)
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_exit_policy_diagnostics_20260630"


@dataclass(frozen=True)
class Policy:
    name: str
    min_age_hours: float
    profit_lock_net: float
    trail_activation: float
    trail_giveback: float


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, usecols=["timestamp", "high", "low", "close"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    return df.sort_values("timestamp").reset_index(drop=True)


def component_train_eval_paths(runtime: dict[str, Any]) -> tuple[Path, Path]:
    comp = runtime["components"]["h48qual"]
    risk_report = read_json(resolve_path(comp["report"]))
    return resolve_path(risk_report["risk_model"]["train_csv"]), resolve_path(risk_report["risk_model"]["eval_csv"])


def source_dir(runtime: dict[str, Any]) -> Path:
    return resolve_path(runtime["source_report"]).parent


def price_move(side: int, entry_price: float, exit_price: float) -> float:
    return float(side) * (float(exit_price) / float(entry_price) - 1.0)


def replay_trade(row: pd.Series, market: pd.DataFrame, policy: Policy) -> pd.Series:
    out = row.copy()
    if policy.name == "baseline":
        out["exit_policy"] = policy.name
        out["policy_exit_applied"] = False
        return out

    entry_ts = pd.Timestamp(row["entry_timestamp"])
    original_exit_ts = pd.Timestamp(row["exit_timestamp"])
    path = market[(market["timestamp"] >= entry_ts) & (market["timestamp"] <= original_exit_ts)].copy()
    if path.empty:
        raise ValueError(f"empty path for {entry_ts}..{original_exit_ts}")
    entry_price = float(path.iloc[0]["close"])
    side = int(row["side"])
    old_cost = float(row["raw_exit_price_move"]) - float(row["net_per_notional"])
    min_ts = entry_ts + pd.Timedelta(hours=float(policy.min_age_hours))
    eligible = path[path["timestamp"] >= min_ts].copy()
    if eligible.empty:
        out["exit_policy"] = policy.name
        out["policy_exit_applied"] = False
        return out

    exit_idx: int | None = None
    exit_reason: str | None = None
    raw_close = np.array([price_move(side, entry_price, px) for px in eligible["close"].astype(float)], dtype=float)
    net_close = raw_close - old_cost

    if policy.profit_lock_net > -999.0:
        hit = np.flatnonzero(net_close >= float(policy.profit_lock_net))
        if len(hit):
            exit_idx = int(eligible.index[int(hit[0])])
            exit_reason = f"profit_lock_{policy.min_age_hours:g}h_{policy.profit_lock_net:.3f}"

    if policy.trail_activation > 0.0 and policy.trail_giveback > 0.0:
        peak = -1.0e9
        for local_i, raw in enumerate(raw_close):
            peak = max(peak, float(raw))
            if peak >= float(policy.trail_activation) and peak - float(raw) >= float(policy.trail_giveback):
                trail_idx = int(eligible.index[local_i])
                if exit_idx is None or trail_idx < exit_idx:
                    exit_idx = trail_idx
                    exit_reason = (
                        f"trail_{policy.min_age_hours:g}h_"
                        f"a{policy.trail_activation:.3f}_g{policy.trail_giveback:.3f}"
                    )
                break

    if exit_idx is None:
        out["exit_policy"] = policy.name
        out["policy_exit_applied"] = False
        return out

    exit_row = market.loc[exit_idx]
    exit_ts = pd.Timestamp(exit_row["timestamp"])
    capped_path = market[(market["timestamp"] >= entry_ts) & (market["timestamp"] <= exit_ts)]
    raw_move = price_move(side, entry_price, float(exit_row["close"]))
    net_per_notional = raw_move - old_cost
    notional = float(row["notional"])
    if side > 0:
        mfe = float(capped_path["high"].max() / entry_price - 1.0)
        mae = float(capped_path["low"].min() / entry_price - 1.0)
    else:
        mfe = float(entry_price / capped_path["low"].min() - 1.0)
        mae = float(entry_price / capped_path["high"].max() - 1.0)
    hold_bars = int(round((exit_ts - entry_ts).total_seconds() / 300.0))

    out["exit_i"] = int(row["entry_i"]) + hold_bars
    out["exit_timestamp"] = exit_ts.strftime("%Y-%m-%d %H:%M:%S")
    out["reason"] = exit_reason
    out["raw_exit_price_move"] = raw_move
    out["net_per_notional"] = net_per_notional
    out["trade_return"] = net_per_notional * notional
    out["win"] = int(net_per_notional > 0.0)
    out["mfe_price_move"] = mfe
    out["mae_price_move"] = mae
    out["exit_policy"] = policy.name
    out["policy_exit_applied"] = True
    return out


def replay_ledger(ledger: pd.DataFrame, market: pd.DataFrame, policy: Policy) -> pd.DataFrame:
    rows = [replay_trade(row, market, policy) for _, row in ledger.iterrows()]
    return pd.DataFrame(rows)


def metrics(ledger: pd.DataFrame) -> dict[str, Any]:
    returns = ledger["trade_return"].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, 1.0e-12) - 1.0
    hold_hours = (
        pd.to_datetime(ledger["exit_timestamp"], errors="raise")
        - pd.to_datetime(ledger["entry_timestamp"], errors="raise")
    ).dt.total_seconds() / 3600.0
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(ledger)),
        "wr": float(ledger["win"].astype(float).mean()),
        "max_hold_hours": float(hold_hours.max()),
        "avg_hold_hours": float(hold_hours.mean()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "policy_exit_count": int(ledger.get("policy_exit_applied", pd.Series(False, index=ledger.index)).astype(bool).sum()),
        "max_leverage": float(ledger["leverage"].astype(float).max()),
        "max_notional": float(ledger["notional"].astype(float).max()),
    }


def policies() -> list[Policy]:
    out = [Policy("baseline", 0.0, -1000.0, 0.0, 0.0)]
    for h in [24.0, 36.0, 48.0, 72.0]:
        for lock in [0.0, 0.005, 0.010, 0.020, 0.030, 0.050]:
            out.append(Policy(f"profit_lock_h{int(h)}_n{lock:.3f}", h, lock, 0.0, 0.0))
        for activation in [0.030, 0.050, 0.075, 0.100]:
            for giveback in [0.015, 0.025, 0.040, 0.060]:
                out.append(Policy(f"trail_h{int(h)}_a{activation:.3f}_g{giveback:.3f}", h, -1000.0, activation, giveback))
        for lock in [0.010, 0.020, 0.030]:
            for activation in [0.050, 0.075]:
                for giveback in [0.025, 0.040]:
                    out.append(Policy(f"locktrail_h{int(h)}_n{lock:.3f}_a{activation:.3f}_g{giveback:.3f}", h, lock, activation, giveback))
    return out


def selection_score(val: dict[str, Any], baseline: dict[str, Any]) -> float:
    pnl = float(val["pnl"])
    mdd_penalty = 8.0 * max(0.0, abs(float(val["mdd"])) - 20.0)
    hold_improvement = max(0.0, float(baseline["avg_hold_hours"]) - float(val["avg_hold_hours"]))
    max_hold_improvement = max(0.0, float(baseline["max_hold_hours"]) - float(val["max_hold_hours"]))
    pnl_loss = max(0.0, float(baseline["pnl"]) - pnl)
    return pnl + 0.10 * hold_improvement + 0.03 * max_hold_improvement - 0.15 * pnl_loss - mdd_penalty


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-contract", type=Path, default=DEFAULT_BASELINE_REPORT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    runtime = read_json(resolve_path(args.runtime_contract))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = component_train_eval_paths(runtime)
    train_market = load_market(train_csv)
    eval_market = load_market(eval_csv)
    src = source_dir(runtime)
    val_ledger = pd.read_csv(src / "validation_scaled_trade_ledger.csv")
    oos_ledger = pd.read_csv(src / "oos_scaled_trade_ledger.csv")

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    baseline_metrics: dict[str, Any] | None = None
    for policy in policies():
        val_replay = replay_ledger(val_ledger, train_market, policy)
        oos_replay = replay_ledger(oos_ledger, eval_market, policy)
        val = metrics(val_replay)
        oos = metrics(oos_replay)
        if policy.name == "baseline":
            baseline_metrics = val
        if baseline_metrics is None:
            raise RuntimeError("baseline policy must be evaluated first")
        row = {
            **{f"policy_{k}": v for k, v in asdict(policy).items()},
            "selection_score": selection_score(val, baseline_metrics),
            **{f"validation_{k}": v for k, v in val.items()},
            **{f"oos_{k}": v for k, v in oos.items()},
        }
        rows.append(row)
        ledgers[policy.name] = (val_replay, oos_replay)

    ranking = pd.DataFrame(rows).sort_values(
        ["selection_score", "validation_pnl", "validation_mdd"],
        ascending=[False, False, False],
    )
    ranking_path = out_dir / "exit_policy_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    best = ranking.iloc[0].to_dict()
    best_name = str(best["policy_name"])
    best_val, best_oos = ledgers[best_name]
    best_val.to_csv(out_dir / "validation_best_exit_policy_ledger.csv", index=False)
    best_oos.to_csv(out_dir / "oos_best_exit_policy_ledger.csv", index=False)

    report = {
        "model_id": "omega4_6_exit_policy_diagnostics_20260630",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "baseline_runtime_contract": str(resolve_path(args.runtime_contract)),
        "selection_scope": "validation_only; OOS is readout only",
        "ranking_csv": str(ranking_path),
        "best_policy": best,
        "baseline_row": ranking[ranking["policy_name"] == "baseline"].iloc[0].to_dict(),
        "artifacts": {
            "validation_best_exit_policy_ledger": str(out_dir / "validation_best_exit_policy_ledger.csv"),
            "oos_best_exit_policy_ledger": str(out_dir / "oos_best_exit_policy_ledger.csv"),
            "ranking": str(ranking_path),
        },
        "redteam_note": "Diagnostic only. Intrabar order is not proven; do not promote without runtime-native replay and artifact audit.",
    }
    write_json(out_dir / "report.json", report)
    print(json.dumps({"report": str(out_dir / "report.json"), "best_policy": best_name, "best": best}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
