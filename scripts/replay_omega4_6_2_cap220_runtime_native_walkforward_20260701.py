#!/usr/bin/env python3
"""Runtime accounting shadow and holdout audit for Omega 4.6.2 cap220.

This script intentionally does not promote the candidate. It separates:

1. frozen ledger-contract parity,
2. GovernorPositionRouter accounting shadow replay,
3. fresh-holdout / fixed-candidate monthly readout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega4_6_2_cap220_short_boost125_time_stop120h_20260630"
CAP220_VARIANT = "short_rsi_skip_ge_56p656189__short_boost125_cap220__time_stop_120h"
VALIDATION_WINNER_VARIANT = "short_rsi_skip_ge_56p656189__none__time_stop_120h"
DEFAULT_RUNTIME_CONTRACT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / MODEL_ID
    / "runtime_contract.json"
)
DEFAULT_OUT_DIR = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_cap220_runtime_native_walkforward_20260701"
)
DEFAULT_AUDIT_JSON = (
    ROOT
    / "docs/audits"
    / "omega4_6_2_cap220_runtime_native_walkforward_20260701.json"
)
DEFAULT_AUDIT_MD = (
    ROOT
    / "docs/audits"
    / "omega4_6_2_cap220_runtime_native_walkforward_20260701.md"
)
EPS = 1.0e-12


def json_default(obj: Any) -> Any:
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_market(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="raise")
    required = {"timestamp", "open", "high", "low", "close"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise RuntimeError(f"market file missing required columns {missing}: {path}")
    return df.sort_values("timestamp").reset_index(drop=True)


def component_train_eval_paths(runtime: dict[str, Any]) -> tuple[Path, Path]:
    risk_report = read_json(resolve_path(runtime["components"]["h48qual"]["report"]))
    risk_model = risk_report["risk_model"]
    return resolve_path(risk_model["train_csv"]), resolve_path(risk_model["eval_csv"])


def active_ledger(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["notional"].astype(float) > EPS].copy()


def overlap_count(df: pd.DataFrame) -> int:
    active = active_ledger(df)
    if len(active) <= 1:
        return 0
    ordered = active.sort_values(["entry_i", "exit_i"]).reset_index(drop=True)
    prev_exit = -1
    overlaps = 0
    for _, row in ordered.iterrows():
        entry_i = int(row["entry_i"])
        exit_i = int(row["exit_i"])
        if entry_i <= prev_exit:
            overlaps += 1
        prev_exit = max(prev_exit, exit_i)
    return overlaps


def ensure_time_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["entry_timestamp_dt"] = pd.to_datetime(out["entry_timestamp"], errors="raise")
    out["exit_timestamp_dt"] = pd.to_datetime(out["exit_timestamp"], errors="raise")
    if "hold_hours" not in out.columns:
        out["hold_hours"] = (
            out["exit_timestamp_dt"] - out["entry_timestamp_dt"]
        ).dt.total_seconds() / 3600.0
    out["entry_month"] = out["entry_timestamp_dt"].dt.to_period("M").astype(str)
    return out


def metrics(df: pd.DataFrame, *, return_col: str = "trade_return") -> dict[str, Any]:
    df = ensure_time_columns(df)
    active = active_ledger(df)
    if active.empty:
        return {
            "pnl": 0.0,
            "mdd": 0.0,
            "trades": 0,
            "wr": 0.0,
            "max_hold_hours": 0.0,
            "hold_over_24h_count": 0,
            "avg_hold_hours": 0.0,
            "max_leverage": 0.0,
            "avg_notional": 0.0,
            "max_notional": 0.0,
            "skipped": int(len(df)),
            "overlap_count": 0,
            "accounting_error_max_abs": 0.0,
            "notional_contract_error_max_abs": 0.0,
            "long_trades": 0,
            "short_trades": 0,
            "reason_counts": {},
            "source_counts": {},
        }
    returns = active[return_col].astype(float).to_numpy()
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    drawdown = curve / np.maximum(peak, EPS) - 1.0
    accounting_error = (
        active["trade_return"].astype(float)
        - active["net_per_notional"].astype(float) * active["notional"].astype(float)
    ).abs()
    notional_contract_error = (
        active["notional"].astype(float)
        - active["margin_fraction"].astype(float) * active["leverage"].astype(float)
    ).abs()
    hold_hours = active["hold_hours"].astype(float)
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(np.min(drawdown) * 100.0),
        "trades": int(len(active)),
        "wr": float((active[return_col].astype(float) > 0.0).mean()),
        "max_hold_hours": float(hold_hours.max()),
        "hold_over_24h_count": int((hold_hours > 24.0 + 1.0e-9).sum()),
        "avg_hold_hours": float(hold_hours.mean()),
        "max_leverage": float(active["leverage"].astype(float).max()),
        "avg_notional": float(active["notional"].astype(float).mean()),
        "max_notional": float(active["notional"].astype(float).max()),
        "skipped": int((df["notional"].astype(float) <= EPS).sum()),
        "overlap_count": int(overlap_count(df)),
        "accounting_error_max_abs": float(accounting_error.max()),
        "notional_contract_error_max_abs": float(notional_contract_error.max()),
        "long_trades": int((active["side"].astype(int) > 0).sum()),
        "short_trades": int((active["side"].astype(int) < 0).sum()),
        "reason_counts": {
            str(k): int(v)
            for k, v in active["reason"].value_counts().sort_index().to_dict().items()
        },
        "source_counts": {
            str(k): int(v)
            for k, v in active["source_alias"].value_counts().sort_index().to_dict().items()
        },
    }


def monthly_rows(df: pd.DataFrame, *, return_col: str = "trade_return") -> list[dict[str, Any]]:
    df = ensure_time_columns(df)
    rows: list[dict[str, Any]] = []
    for month, group in df.groupby("entry_month", sort=True):
        row = metrics(group, return_col=return_col)
        row["month"] = str(month)
        rows.append(row)
    return rows


def compare_contract_metrics(
    split: str,
    observed: dict[str, Any],
    runtime: dict[str, Any],
    *,
    tolerance: float = 1.0e-9,
) -> dict[str, Any]:
    expected = runtime[split]
    key_map = {
        "pnl": "pnl",
        "mdd": "mdd",
        "trades": "trades",
        "wr": "wr",
        "avg_hold_hours": "avg_hold_hours",
        "max_hold_hours": "max_hold_hours",
        "hold_over_24h_count": "hold_over_24h_count",
        "max_leverage": "max_leverage",
        "max_notional": "max_notional",
        "avg_notional": "avg_notional",
        "skipped": "skipped",
        "overlap_count": "overlap_count",
        "accounting_error_max_abs": "accounting_error_max_abs",
        "notional_contract_error_max_abs": "notional_contract_error_max_abs",
    }
    rows: dict[str, dict[str, float | int | bool]] = {}
    passed = True
    for obs_key, exp_key in key_map.items():
        obs = observed[obs_key]
        exp = expected[exp_key]
        diff = float(obs) - float(exp)
        ok = abs(diff) <= tolerance
        passed = passed and ok
        rows[obs_key] = {
            "observed": obs,
            "expected": exp,
            "diff": diff,
            "pass": bool(ok),
        }
    return {"pass": bool(passed), "checks": rows, "tolerance": tolerance}


def lookup_prices(ledger: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    close_by_ts = dict(zip(market["timestamp"], market["close"].astype(float)))
    out = ensure_time_columns(ledger)
    entry_prices: list[float] = []
    exit_prices: list[float] = []
    raw_close_moves: list[float] = []
    for _, row in out.iterrows():
        entry_ts = pd.Timestamp(row["entry_timestamp_dt"])
        exit_ts = pd.Timestamp(row["exit_timestamp_dt"])
        entry_price = close_by_ts.get(entry_ts)
        exit_price = close_by_ts.get(exit_ts)
        if entry_price is None or exit_price is None:
            raise RuntimeError(f"missing close price for {entry_ts} -> {exit_ts}")
        side = int(row["side"])
        raw_move = float(side) * (float(exit_price) / float(entry_price) - 1.0)
        entry_prices.append(float(entry_price))
        exit_prices.append(float(exit_price))
        raw_close_moves.append(raw_move)
    out["entry_close"] = entry_prices
    out["exit_close"] = exit_prices
    out["raw_close_price_move"] = raw_close_moves
    out["raw_close_vs_ledger_move_diff"] = (
        out["raw_close_price_move"].astype(float) - out["raw_exit_price_move"].astype(float)
    )
    return out


class TradeMathShadow:
    """Source-equivalent shadow of GovernorPositionRouter._trade_math."""

    def __init__(
        self,
        *,
        trade_fee: float = 0.0005,
        trade_slip: float = 0.0002,
        taker_fee: float = 0.0005,
        maker_fee: float = 0.0001,
        exposure_cap: float = 5.0,
    ) -> None:
        self.trade_fee = float(trade_fee)
        self.trade_slip = float(trade_slip)
        self.taker_fee = float(taker_fee)
        self.maker_fee = float(maker_fee)
        self.exposure_cap = float(exposure_cap)

    @staticmethod
    def _is_real_execution_liquidity(liquidity: str | None) -> bool:
        s = str(liquidity or "").strip().lower()
        return bool(s) and "synthetic" not in s and s != "dry_run"

    def _fee_rate_for_liquidity(
        self,
        liquidity: str | None,
        *,
        default_synthetic: bool = True,
    ) -> tuple[float, str]:
        s = str(liquidity or "").strip().lower()
        if self._is_real_execution_liquidity(s):
            if "maker" in s and "taker" not in s:
                return float(self.maker_fee), "maker"
            if "maker" in s and "taker" in s:
                return float(self.taker_fee), "maker_taker_conservative_taker"
            if "taker" in s or "market" in s:
                return float(self.taker_fee), "taker"
        if default_synthetic:
            return float(self.trade_fee), "synthetic_default"
        return 0.0, "none"

    def _trade_math(
        self,
        side: str,
        entry_price: float,
        exit_price: float,
        exposure: float,
        *,
        entry_liquidity: str | None = None,
        exit_liquidity: str | None = None,
    ) -> dict[str, Any]:
        side_u = str(side or "").upper()
        entry = float(entry_price or 0.0)
        exit_raw = float(exit_price or 0.0)
        lev = float(np.clip(float(exposure or 0.0), 0.0, self.exposure_cap))
        if side_u not in {"LONG", "SHORT"} or entry <= 0.0 or exit_raw <= 0.0:
            return {
                "entry_exec_price": 0.0,
                "exit_exec_price": 0.0,
                "gross_return_frac": 0.0,
                "entry_fee_rate": 0.0,
                "exit_fee_rate": 0.0,
                "roundtrip_fee_rate": 0.0,
                "fee_model": "invalid",
                "pnl_frac": 0.0,
                "pnl_pct": 0.0,
            }
        entry_fee, entry_fee_model = self._fee_rate_for_liquidity(entry_liquidity)
        exit_fee, exit_fee_model = self._fee_rate_for_liquidity(exit_liquidity)
        entry_is_real = self._is_real_execution_liquidity(entry_liquidity)
        exit_is_real = self._is_real_execution_liquidity(exit_liquidity)
        if side_u == "LONG":
            entry_exec = entry if entry_is_real else entry * (1.0 + self.trade_slip)
            exit_exec = exit_raw if exit_is_real else exit_raw * (1.0 - self.trade_slip)
            gross = (exit_exec - entry_exec) / max(entry_exec, 1.0e-8)
        else:
            entry_exec = entry if entry_is_real else entry * (1.0 - self.trade_slip)
            exit_exec = exit_raw if exit_is_real else exit_raw * (1.0 + self.trade_slip)
            gross = (entry_exec - exit_exec) / max(abs(entry_exec), 1.0e-8)
        pnl_frac = float(gross * lev - ((entry_fee + exit_fee) * lev))
        return {
            "entry_exec_price": float(entry_exec),
            "exit_exec_price": float(exit_exec),
            "gross_return_frac": float(gross),
            "entry_fee_rate": float(entry_fee),
            "exit_fee_rate": float(exit_fee),
            "roundtrip_fee_rate": float(entry_fee + exit_fee),
            "entry_fee_model": str(entry_fee_model),
            "exit_fee_model": str(exit_fee_model),
            "fee_model": f"{entry_fee_model}+{exit_fee_model}",
            "fee_cost_frac": float((entry_fee + exit_fee) * lev),
            "pnl_frac": pnl_frac,
            "pnl_pct": float(pnl_frac * 100.0),
        }


def isolated_governor_router(out_dir: Path, *, import_trading_bot: bool) -> tuple[Any | None, dict[str, Any]]:
    if not import_trading_bot:
        router = TradeMathShadow(
            trade_fee=float(os.getenv("LIVE_FEE_RATE", "0.0005")),
            trade_slip=float(os.getenv("LIVE_SLIP_RATE", "0.0002")),
            taker_fee=float(os.getenv("LIVE_TAKER_FEE_RATE", "0.0005")),
            maker_fee=float(os.getenv("LIVE_MAKER_FEE_RATE", str(0.0005 * float(os.getenv("LIVE_MAKER_FEE_MULT", "0.20"))))),
            exposure_cap=5.0,
        )
        return router, {
            "available": True,
            "implementation": "source_equivalent_trade_math_shadow",
            "trading_bot_imported": False,
            "reason": "Default audit path avoids importing trading_bot because it initializes live/runtime modules.",
            "fee_rate": float(router.trade_fee),
            "slip_rate": float(router.trade_slip),
            "taker_fee": float(router.taker_fee),
            "maker_fee": float(router.maker_fee),
            "exposure_cap": float(router.exposure_cap),
        }

    os.environ.setdefault("CONSOLE_LOG_COMPACT", "1")
    os.environ["TRADE_JOURNAL_PATH"] = str(out_dir / "isolated_live" / "trade_journal.jsonl")
    os.environ["TRADING_BOT_PROCESS_LOCK_PATH"] = str(out_dir / "isolated_live" / "trade_journal.lock")
    os.environ["GOVERNOR_LIVE_STATE_PATH"] = str(out_dir / "isolated_governor_live_state.json")
    os.environ.setdefault("LIVE_FEE_RATE", "0.0005")
    os.environ.setdefault("LIVE_SLIP_RATE", "0.0002")
    os.environ.setdefault("LIVE_TAKER_FEE_RATE", "0.0005")
    os.environ.setdefault("LIVE_MAKER_FEE_MULT", "0.20")
    sys.path.insert(0, str(ROOT))
    try:
        with tempfile.TemporaryDirectory(prefix="omega462_import_"):
            module = importlib.import_module("trading_bot")
        router = module.GovernorPositionRouter()
        router.exposure_cap = 5.0
        return router, {
            "available": True,
            "implementation": "trading_bot.GovernorPositionRouter",
            "trading_bot_imported": True,
            "fee_rate": float(router.trade_fee),
            "slip_rate": float(router.trade_slip),
            "taker_fee": float(router.taker_fee),
            "maker_fee": float(router.maker_fee),
            "exposure_cap": float(router.exposure_cap),
        }
    except Exception as exc:  # pragma: no cover - audit should keep going.
        return None, {"available": False, "error": repr(exc)}


def replay_accounting_shadow(
    split: str,
    ledger: pd.DataFrame,
    market: pd.DataFrame,
    out_dir: Path,
    router: Any | None,
) -> dict[str, Any]:
    priced = lookup_prices(ledger, market)
    decisions = pd.DataFrame(
        {
            "model_id": MODEL_ID,
            "variant": CAP220_VARIANT,
            "split": split,
            "entry_i": priced["entry_i"].astype(int),
            "exit_i": priced["exit_i"].astype(int),
            "entry_timestamp": priced["entry_timestamp"],
            "exit_timestamp": priced["exit_timestamp"],
            "source_alias": priced["source_alias"].astype(str),
            "side": priced["side"].astype(int),
            "decision": np.where(
                priced["notional"].astype(float) > EPS,
                np.where(priced["side"].astype(int) > 0, "ENTER_LONG", "ENTER_SHORT"),
                "SKIP",
            ),
            "reason": priced["reason"].astype(str),
            "skip_reason": np.where(
                priced["notional"].astype(float) > EPS,
                "",
                "entry_gate_or_zero_notional",
            ),
            "notional": priced["notional"].astype(float),
            "margin_fraction": priced["margin_fraction"].astype(float),
            "leverage": priced["leverage"].astype(float),
            "entry_close": priced["entry_close"].astype(float),
            "exit_close": priced["exit_close"].astype(float),
            "raw_exit_price_move": priced["raw_exit_price_move"].astype(float),
            "raw_close_price_move": priced["raw_close_price_move"].astype(float),
            "net_per_notional": priced["net_per_notional"].astype(float),
            "trade_return": priced["trade_return"].astype(float),
        }
    )
    active = active_ledger(priced)
    journal = active.copy()
    if router is not None:
        native_rows: list[dict[str, Any]] = []
        for _, row in journal.iterrows():
            side_name = "LONG" if int(row["side"]) > 0 else "SHORT"
            math = router._trade_math(
                side_name,
                float(row["entry_close"]),
                float(row["exit_close"]),
                float(row["notional"]),
            )
            native_rows.append(
                {
                    "native_entry_exec_price": float(math.get("entry_exec_price", 0.0)),
                    "native_exit_exec_price": float(math.get("exit_exec_price", 0.0)),
                    "native_gross_return_frac": float(math.get("gross_return_frac", 0.0)),
                    "native_fee_model": str(math.get("fee_model", "")),
                    "native_fee_cost_frac": float(math.get("fee_cost_frac", 0.0)),
                    "native_trade_return": float(math.get("pnl_frac", 0.0)),
                    "native_trade_return_diff": float(math.get("pnl_frac", 0.0))
                    - float(row["trade_return"]),
                }
            )
        native = pd.DataFrame(native_rows, index=journal.index)
        journal = pd.concat([journal.reset_index(drop=True), native.reset_index(drop=True)], axis=1)
        native_metrics = metrics(journal, return_col="native_trade_return")
        max_abs_diff = float(journal["native_trade_return_diff"].abs().max()) if not journal.empty else 0.0
        mean_abs_diff = float(journal["native_trade_return_diff"].abs().mean()) if not journal.empty else 0.0
        native_available = True
    else:
        journal["native_trade_return"] = np.nan
        journal["native_trade_return_diff"] = np.nan
        native_metrics = {}
        max_abs_diff = float("nan")
        mean_abs_diff = float("nan")
        native_available = False

    decisions_path = out_dir / f"{split}_runtime_native_decisions.csv"
    journal_path = out_dir / f"{split}_runtime_native_trade_journal.csv"
    closes_path = out_dir / f"{split}_runtime_native_closes.csv"
    decisions.to_csv(decisions_path, index=False)
    journal.to_csv(journal_path, index=False)
    priced[
        [
            "entry_i",
            "exit_i",
            "entry_timestamp",
            "exit_timestamp",
            "side",
            "entry_close",
            "exit_close",
            "raw_exit_price_move",
            "raw_close_price_move",
            "raw_close_vs_ledger_move_diff",
            "net_per_notional",
            "trade_return",
            "notional",
        ]
    ].to_csv(closes_path, index=False)

    return {
        "native_accounting_shadow_available": bool(native_available),
        "native_accounting_shadow_parity_pass": bool(native_available and max_abs_diff <= 1.0e-9),
        "native_trade_return_diff_max_abs": max_abs_diff,
        "native_trade_return_diff_mean_abs": mean_abs_diff,
        "ledger_metrics": metrics(priced),
        "native_shadow_metrics": native_metrics,
        "artifacts": {
            "decisions": str(decisions_path),
            "trade_journal": str(journal_path),
            "closes": str(closes_path),
        },
    }


def source_variant_ledgers(source_dir: Path, variant: str) -> tuple[Path, Path]:
    safe = variant.replace(".", "p").replace("/", "_")
    return (
        source_dir / f"validation_{safe}_ledger.csv",
        source_dir / f"oos_{safe}_ledger.csv",
    )


def monthly_readout_frame(variant: str, split: str, ledger: pd.DataFrame) -> pd.DataFrame:
    rows = monthly_rows(ledger)
    for row in rows:
        row["variant"] = variant
        row["split"] = split
    columns = [
        "variant",
        "split",
        "month",
        "pnl",
        "mdd",
        "trades",
        "wr",
        "avg_hold_hours",
        "max_hold_hours",
        "max_notional",
        "skipped",
        "long_trades",
        "short_trades",
    ]
    return pd.DataFrame(rows)[columns]


def market_span(path: Path) -> dict[str, Any]:
    df = load_market(path)
    return {
        "path": str(path),
        "start": str(df["timestamp"].min()),
        "end": str(df["timestamp"].max()),
        "rows": int(len(df)),
    }


def write_audit_markdown(path: Path, report: dict[str, Any]) -> None:
    val = report["runtime_replay"]["validation"]
    oos = report["runtime_replay"]["oos"]
    holdout = report["fresh_holdout_walkforward"]
    text = f"""# Omega 4.6.2 Runtime Shadow / Holdout Audit - 2026-07-01

## Scope

- Model id: `{MODEL_ID}`
- Variant: `{CAP220_VARIANT}`
- Output dir: `{report["artifacts"]["out_dir"]}`

## Runtime Replay

| Check | Result |
| --- | --- |
| Ledger-contract parity | `{report["runtime_replay"]["ledger_contract_parity_pass"]}` |
| GovernorPositionRouter accounting shadow available | `{report["runtime_replay"]["governor_position_router_shadow_available"]}` |
| trading_bot imported for shadow | `{report["runtime_replay"]["governor_position_router_shadow_info"].get("trading_bot_imported")}` |
| Accounting shadow parity | `{report["runtime_replay"]["governor_position_router_shadow_parity_pass"]}` |
| FinalGovernorRuntime.decide replay available | `{report["runtime_replay"]["final_governor_runtime_decide_replay_available"]}` |
| Full runtime-native promotion pass | `{report["runtime_replay"]["full_runtime_native_promotion_pass"]}` |

Validation ledger PnL: `{val["ledger_metrics"]["pnl"]:.6f}%`, native-shadow PnL: `{val.get("native_shadow_metrics", {}).get("pnl", float("nan")):.6f}%`

OOS ledger PnL: `{oos["ledger_metrics"]["pnl"]:.6f}%`, native-shadow PnL: `{oos.get("native_shadow_metrics", {}).get("pnl", float("nan")):.6f}%`

The accounting shadow uses `GovernorPositionRouter._trade_math()` only. It is not a policy replay through `FinalGovernorRuntime.decide()`, so it cannot satisfy the full runtime-native replay gate.

## Fresh Holdout / Walk-Forward

| Check | Result |
| --- | --- |
| Exact fresh holdout available | `{holdout["fresh_holdout_available"]}` |
| Clean-OOS promotion claim allowed | `{holdout["clean_oos_promotion_claim_allowed"]}` |
| Fixed candidate OOS monthly positive | `{holdout["fixed_candidate_oos_monthly_positive"]}` |
| Fixed candidate OOS monthly count | `{holdout["fixed_candidate_oos_monthly_positive_count"]}/{holdout["fixed_candidate_oos_monthly_count"]}` |

Reason fresh holdout is unavailable: {holdout["fresh_holdout_unavailable_reason"]}

## Monthly Readout

CSV: `{report["artifacts"]["monthly_walkforward_readout"]}`

## Overall

- Runtime-native replay status: `{report["overall"]["runtime_native_replay_status"]}`
- Fresh holdout status: `{report["overall"]["fresh_holdout_status"]}`
- Promotion status: `{report["overall"]["promotion_status"]}`
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-contract", type=Path, default=DEFAULT_RUNTIME_CONTRACT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--audit-json", type=Path, default=DEFAULT_AUDIT_JSON)
    parser.add_argument("--audit-md", type=Path, default=DEFAULT_AUDIT_MD)
    parser.add_argument(
        "--import-trading-bot",
        action="store_true",
        help="Try importing trading_bot.GovernorPositionRouter. Default uses source-equivalent _trade_math shadow.",
    )
    args = parser.parse_args()

    runtime = read_json(resolve_path(args.runtime_contract))
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_csv, eval_csv = component_train_eval_paths(runtime)
    train_market = load_market(train_csv)
    eval_market = load_market(eval_csv)
    source_dir = resolve_path(runtime["source_report"]).parent

    val_path, oos_path = source_variant_ledgers(source_dir, CAP220_VARIANT)
    if not val_path.exists() or not oos_path.exists():
        raise RuntimeError(f"missing cap220 source ledgers: {val_path}, {oos_path}")
    val = ensure_time_columns(pd.read_csv(val_path))
    oos = ensure_time_columns(pd.read_csv(oos_path))

    winner_val_path, winner_oos_path = source_variant_ledgers(source_dir, VALIDATION_WINNER_VARIANT)
    monthly_frames = [
        monthly_readout_frame(CAP220_VARIANT, "validation", val),
        monthly_readout_frame(CAP220_VARIANT, "oos", oos),
    ]
    if winner_val_path.exists() and winner_oos_path.exists():
        monthly_frames.extend(
            [
                monthly_readout_frame(
                    VALIDATION_WINNER_VARIANT,
                    "validation",
                    ensure_time_columns(pd.read_csv(winner_val_path)),
                ),
                monthly_readout_frame(
                    VALIDATION_WINNER_VARIANT,
                    "oos",
                    ensure_time_columns(pd.read_csv(winner_oos_path)),
                ),
            ]
        )
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly_path = out_dir / "monthly_walkforward_readout.csv"
    monthly.to_csv(monthly_path, index=False)

    router, router_info = isolated_governor_router(out_dir, import_trading_bot=bool(args.import_trading_bot))
    validation_replay = replay_accounting_shadow("validation", val, train_market, out_dir, router)
    oos_replay = replay_accounting_shadow("oos", oos, eval_market, out_dir, router)

    val_parity = compare_contract_metrics("validation", validation_replay["ledger_metrics"], runtime)
    oos_parity = compare_contract_metrics("oos", oos_replay["ledger_metrics"], runtime)
    ledger_contract_parity_pass = bool(val_parity["pass"] and oos_parity["pass"])
    shadow_available = bool(
        validation_replay["native_accounting_shadow_available"]
        and oos_replay["native_accounting_shadow_available"]
    )
    shadow_parity_pass = bool(
        validation_replay["native_accounting_shadow_parity_pass"]
        and oos_replay["native_accounting_shadow_parity_pass"]
    )

    cap220_oos_monthly = monthly[
        (monthly["variant"].eq(CAP220_VARIANT)) & (monthly["split"].eq("oos"))
    ].copy()
    positive_count = int((cap220_oos_monthly["pnl"].astype(float) > 0.0).sum())
    month_count = int(len(cap220_oos_monthly))
    oos_exit_max = pd.to_datetime(oos["exit_timestamp"], errors="raise").max()
    eval_max = eval_market["timestamp"].max()
    eval_has_material_after_oos = bool(eval_max > oos_exit_max + pd.Timedelta(days=7))

    report = {
        "audit_id": "omega4_6_2_cap220_runtime_native_walkforward_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_ID,
        "variant": CAP220_VARIANT,
        "runtime_contract": str(resolve_path(args.runtime_contract)),
        "source_dir": str(source_dir),
        "runtime_replay": {
            "ledger_contract_parity_pass": ledger_contract_parity_pass,
            "ledger_contract_parity": {
                "validation": val_parity,
                "oos": oos_parity,
            },
            "governor_position_router_shadow_available": shadow_available,
            "governor_position_router_shadow_info": router_info,
            "governor_position_router_shadow_parity_pass": shadow_parity_pass,
            "final_governor_runtime_decide_replay_available": False,
            "final_governor_runtime_decide_replay_unavailable_reason": (
                "No Omega4.6.2 cap220 adapter or exact short_boost125_cap220 "
                "policy wiring exists in trading_bot.FinalGovernorRuntime.decide()."
            ),
            "full_runtime_native_promotion_pass": False,
            "validation": validation_replay,
            "oos": oos_replay,
        },
        "fresh_holdout_walkforward": {
            "fresh_holdout_available": False,
            "fresh_holdout_unavailable_reason": (
                "Exact candidate artifacts expose validation ledgers for 2025-10..2025-12 "
                "and OOS ledgers for 2026-01..2026-02 only. The component eval market "
                "ends at or near the OOS window, and no exact post-OOS prediction/ledger "
                "artifact is present for this model."
            ),
            "eval_has_material_bars_after_oos": eval_has_material_after_oos,
            "oos_exit_max": str(oos_exit_max),
            "eval_market_max": str(eval_max),
            "market_spans": {
                "train": market_span(train_csv),
                "eval": market_span(eval_csv),
            },
            "fixed_candidate_monthly_readout": str(monthly_path),
            "fixed_candidate_oos_monthly_positive": bool(positive_count == month_count and month_count > 0),
            "fixed_candidate_oos_monthly_positive_count": positive_count,
            "fixed_candidate_oos_monthly_count": month_count,
            "clean_oos_promotion_claim_allowed": False,
        },
        "artifacts": {
            "out_dir": str(out_dir),
            "monthly_walkforward_readout": str(monthly_path),
            "audit_json": str(resolve_path(args.audit_json)),
            "audit_md": str(resolve_path(args.audit_md)),
        },
        "overall": {
            "runtime_native_replay_status": "FAIL_FINAL_GOVERNOR_RUNTIME_DECIDE_NOT_AVAILABLE",
            "fresh_holdout_status": "FAIL_NO_EXACT_FRESH_HOLDOUT_AVAILABLE",
            "promotion_status": "BLOCKED_FOR_FULL_PROMOTION",
        },
    }
    write_json(out_dir / "report.json", report)
    write_json(resolve_path(args.audit_json), report)
    write_audit_markdown(resolve_path(args.audit_md), report)

    print(
        json.dumps(
            {
                "report": str(out_dir / "report.json"),
                "audit_json": str(resolve_path(args.audit_json)),
                "audit_md": str(resolve_path(args.audit_md)),
                "ledger_contract_parity_pass": ledger_contract_parity_pass,
                "governor_position_router_shadow_available": shadow_available,
                "governor_position_router_shadow_parity_pass": shadow_parity_pass,
                "final_governor_runtime_decide_replay_available": False,
                "fresh_holdout_available": False,
                "fixed_candidate_oos_monthly_positive": report["fresh_holdout_walkforward"][
                    "fixed_candidate_oos_monthly_positive"
                ],
            },
            ensure_ascii=False,
            indent=2,
            default=json_default,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
