#!/usr/bin/env python3
"""Runtime-native Omega5 walk-forward proof harness.

This script feeds historical bars through trading_bot.FinalGovernorRuntime.decide()
with Omega5 and its Omega4.6.2 source parent enabled. It does not use the
Omega4.6.2 ledger replay adapter for decisions.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / "omega5_source_parent_runtime_native_walkforward_20260701"
STATE_DIR = OUT_DIR / "isolated_live_state"
REPORT_PATH = OUT_DIR / "report.json"
REPORT_MD = ROOT / "docs/audits/omega5_source_parent_runtime_native_walkforward_20260701.md"
SOURCE_REPORT = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
    / "report.json"
)
VALIDATION_LEDGER = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
    / "validation_lf0p900_sf1p050_cap4p40_ledger.csv"
)
OOS_LEDGER = (
    ROOT
    / "tmp/causal_regen_20260516"
    / "omega4_6_2_v5_roll8_side_specific_two_stage_exposure_validation_only_20260701"
    / "oos_lf0p900_sf1p050_cap4p40_ledger.csv"
)
TRADE_CANDIDATES = {
    "validation": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2025_alpha6_current_tail111_exact.csv",
    "oos": ROOT / "tmp/causal_regen_20260516/alpha7_01965_cleanfunding_candidates_20260529/trade_candidates_2026_alpha6_current_tail111_exact.csv",
}
BASE_FEATURES = {
    "validation": ROOT / "data/splits/year_oos/training_features_2025.csv",
    "oos": ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv",
}
EXPECTED_LEDGERS = {"validation": VALIDATION_LEDGER, "oos": OOS_LEDGER}
SPLIT_WINDOWS = {
    "validation": ("2025-10-01 00:00:00", "2025-12-31 23:25:00"),
    "oos": ("2026-01-01 00:00:00", "2026-02-28 15:30:00"),
}
REPLAY_EXECUTION_LIQUIDITY = "signal_immediate_maker_limit"
REPLAY_EXECUTION_ROUTE = "runtime_native_signal_immediate_maker_limit"
REPLAY_EXECUTION_ORDER_TYPE = "LIMIT_MAKER"
EPS = 1.0e-12


def _set_isolated_live_env(*, window_bars: int) -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env_paths = {
        "TRADING_BOT_PROCESS_LOCK_PATH": STATE_DIR / "trading_bot_replay.lock",
        "GOVERNOR_LIVE_STATE_PATH": STATE_DIR / "governor_live_state.json",
        "FINAL_GOVERNOR_RUNTIME_STATE_PATH": STATE_DIR / "final_governor_runtime_state.json",
        "TRADE_JOURNAL_PATH": STATE_DIR / "trade_journal.jsonl",
        "POSITION_ACCOUNTING_AUDIT_PATH": STATE_DIR / "position_accounting_audit.jsonl",
        "DASHBOARD_EVENTS_PATH": STATE_DIR / "dashboard_events.jsonl",
        "DASHBOARD_STATE_PATH": STATE_DIR / "dashboard_state.json",
        "COMPACT_DASHBOARD_STATE_PATH": STATE_DIR / "dashboard_state_governor.json",
        "TP_RUNNER_SHADOW_PARITY_PATH": STATE_DIR / "tp_runner_shadow_parity.jsonl",
        "FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH": STATE_DIR / "pending_next_open_intent.json",
        "OMEGA123_CASH_SLEEVE_TELEMETRY_PATH": STATE_DIR / "omega_cash_sleeve_decisions.jsonl",
    }
    for key, path in env_paths.items():
        os.environ[key] = str(path)

    os.environ["FINAL_GOVERNOR_WINDOW_BARS"] = str(int(window_bars))
    os.environ["FINAL_GOVERNOR_OMEGA5_ENABLE"] = "1"
    os.environ["FINAL_GOVERNOR_OMEGA5_SOURCE_PARENT_ENABLE"] = "1"
    os.environ["FINAL_GOVERNOR_OMEGA1_2_1_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_FULLY_LEARNED_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_FULLY_LEARNED_FALLBACK_EXIT_SUBMODEL_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_ALPHA3_CANONICAL_DECISION_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_ALPHA2_1_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_V21_2_JACKPOT_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_V21_2_JACKPOT_REQUIRED"] = "0"
    os.environ["FINAL_GOVERNOR_V31_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_V31_REQUIRED"] = "0"
    os.environ["FINAL_GOVERNOR_CLEAN_REGIME4_STICKY_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_REGIME4_PRED_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_REGIME_PREDICTOR_ENABLE"] = "0"
    os.environ["TP_RUNNER_ONLY_ENABLE"] = "0"
    os.environ["TP_RUNNER_SHADOW_ENABLE"] = "0"
    os.environ["OMEGA123_CASH_SLEEVE_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE"] = "0"
    os.environ["FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION"] = "1"


def json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
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
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=json_default) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=json_default) + "\n")


def reset_state() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for path in STATE_DIR.glob("*"):
        if path.is_file() and path.name != "trading_bot_replay.lock":
            path.unlink()
    Path(os.environ["TRADE_JOURNAL_PATH"]).write_text("", encoding="utf-8")


def load_enriched_frame(split: str) -> pd.DataFrame:
    candidate = pd.read_csv(TRADE_CANDIDATES[split])
    base = pd.read_csv(BASE_FEATURES[split])
    candidate["timestamp"] = pd.to_datetime(candidate["timestamp"], errors="raise")
    base["timestamp"] = pd.to_datetime(base["timestamp"], errors="raise")
    missing = [c for c in base.columns if c not in candidate.columns]
    if missing:
        candidate = candidate.merge(base[["timestamp"] + missing], on="timestamp", how="left")
    candidate = candidate.sort_values("timestamp").reset_index(drop=True)
    return candidate


def prepare_replay_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from features.engineering import FeatureEngineer
    from features.high_order_state import add_high_order_state_features

    out = FeatureEngineer(keep_only_active=False)._create_directional_alpha_features(frame.copy())
    if "funding_z_score" not in out.columns:
        funding = pd.to_numeric(out["last_funding_rate"], errors="coerce").fillna(0.0)
        roll_mean = funding.rolling(window=288, min_periods=20).mean()
        roll_std = funding.rolling(window=288, min_periods=20).std().replace(0, np.nan)
        out["funding_z_score"] = ((funding - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "regime_persistence" not in out.columns:
        out = add_high_order_state_features(out)
    for col in ("regime_bull", "regime_bear", "regime_chop", "regime_whipsaw"):
        if col not in out.columns:
            out[col] = 0.0
    if "regime_normal" not in out.columns:
        out["regime_normal"] = 1.0
    return out.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0).reset_index(drop=True)


def split_index_bounds(frame: pd.DataFrame, split: str, max_bars: int | None) -> tuple[int, int]:
    start_raw, end_raw = SPLIT_WINDOWS[split]
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    start_ts = pd.Timestamp(start_raw)
    end_ts = pd.Timestamp(end_raw)
    idx = np.flatnonzero((ts >= start_ts).to_numpy() & (ts <= end_ts).to_numpy())
    if len(idx) == 0:
        raise RuntimeError(f"{split}: empty replay window {start_raw}..{end_raw}")
    start = max(int(idx[0]), 49)
    end = int(idx[-1])
    if max_bars is not None and max_bars > 0:
        end = min(end, start + int(max_bars))
    if end <= start:
        raise RuntimeError(f"{split}: invalid replay bounds start={start} end={end}")
    return start, end


def bar_time_kst(row: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(row["timestamp"]) + pd.Timedelta(hours=9)


def bar_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    val = row.get(key, default)
    try:
        out = float(val)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def transition_label(tb: Any, prev_pos: str | None, cur_pos: str | None) -> str:
    return tb._pos_transition_label(prev_pos, cur_pos)


def apply_close_accounting(router: Any, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    side = str(out.get("side", "") or "").upper()
    entry_price = float(out.get("entry_price", 0.0) or 0.0)
    exit_price = float(out.get("exit_price", out.get("current_price", 0.0)) or 0.0)
    exposure = float(out.get("notional_exposure", out.get("total_exposure", 0.0)) or 0.0)
    source = str(out.get("source", "") or "")
    model_id = str(out.get("model_id", out.get("close_decision_model_id", "")) or "")
    if (source.startswith("omega5|") or model_id == "omega5_event_risk_governor_20260702") and side in {"LONG", "SHORT"} and entry_price > 0.0 and exit_price > 0.0:
        roundtrip_cost = float(out.get("omega5_source_roundtrip_cost", 0.0) or 0.0)
        if roundtrip_cost <= 0.0:
            roundtrip_cost = 0.000612
        raw = (exit_price - entry_price) / entry_price if side == "LONG" else (entry_price - exit_price) / entry_price
        pnl_frac = float((raw - roundtrip_cost) * exposure)
        out.update(
            {
                "entry_exec_price": float(entry_price),
                "exit_exec_price": float(exit_price),
                "gross_return_frac": float(raw),
                "entry_fee_rate": float(roundtrip_cost / 2.0),
                "exit_fee_rate": float(roundtrip_cost / 2.0),
                "roundtrip_fee_rate": float(roundtrip_cost),
                "entry_fee_model": "omega5_source_cost",
                "exit_fee_model": "omega5_source_cost",
                "fee_model": "omega5_source_roundtrip_cost",
                "fee_cost_frac": float(roundtrip_cost * exposure),
                "pnl_frac": pnl_frac,
                "pnl_pct": float(pnl_frac * 100.0),
            }
        )
        return out
    math = router._trade_math(
        side,
        entry_price,
        exit_price,
        exposure,
        entry_liquidity=REPLAY_EXECUTION_LIQUIDITY,
        exit_liquidity=REPLAY_EXECUTION_LIQUIDITY,
    )
    out.update(math)
    out["entry_execution_liquidity"] = REPLAY_EXECUTION_LIQUIDITY
    out["entry_execution_route"] = REPLAY_EXECUTION_ROUTE
    out["entry_execution_order_type"] = REPLAY_EXECUTION_ORDER_TYPE
    out["exit_execution_liquidity"] = REPLAY_EXECUTION_LIQUIDITY
    out["exit_execution_route"] = REPLAY_EXECUTION_ROUTE
    out["exit_execution_order_type"] = REPLAY_EXECUTION_ORDER_TYPE
    return out


def audit_context(
    *,
    decision_row: pd.Series,
    execution_row: pd.Series,
    decision_time_kst: pd.Timestamp,
    execution_time_kst: pd.Timestamp,
    decision_price: float,
    execution_price: float,
    info: dict[str, Any],
    risk: dict[str, Any],
) -> dict[str, Any]:
    trace = dict((info or {}).get("sleeve_trace", {}) or {})
    return {
        "ledger_ts_kind": "next_bar_open_execution",
        "decision_made_at_kst": str(decision_time_kst),
        "decision_bar_ts": str(decision_time_kst),
        "decision_bar_utc": str(decision_row.get("timestamp", "")),
        "decision_bar_open": bar_float(decision_row, "open", decision_price),
        "decision_bar_high": bar_float(decision_row, "high", decision_price),
        "decision_bar_low": bar_float(decision_row, "low", decision_price),
        "decision_bar_close": bar_float(decision_row, "close", decision_price),
        "execution_bar_ts": str(execution_time_kst),
        "execution_bar_utc": str(execution_row.get("timestamp", "")),
        "execution_bar_open": bar_float(execution_row, "open", execution_price),
        "execution_bar_high": bar_float(execution_row, "high", execution_price),
        "execution_bar_low": bar_float(execution_row, "low", execution_price),
        "execution_bar_close": bar_float(execution_row, "close", execution_price),
        "decision_price": float(decision_price),
        "decision_price_source": "historical.close",
        "execution_price": float(execution_price),
        "execution_price_source": "historical.next_bar_open",
        "entry_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
        "entry_execution_route": REPLAY_EXECUTION_ROUTE,
        "entry_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
        "exit_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
        "exit_execution_route": REPLAY_EXECUTION_ROUTE,
        "exit_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
        "sleeve_trace": dict(trace),
        "model_version": str(info.get("model_version", trace.get("model_version", "")) or ""),
        "model_id": str(info.get("model_id", trace.get("model_id", "")) or ""),
        "model_sleeve": str(info.get("model_sleeve", "")),
        **risk,
    }


def risk_fields(governor: Any, snapshot: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    return {
        "take_profit": float((info or {}).get("take_profit", getattr(governor, "active_omega5_take_profit", 0.0)) or 0.0),
        "stop_loss": float((info or {}).get("stop_loss", getattr(governor, "active_omega5_stop_loss", 0.0)) or 0.0),
        "max_hold_bars": int((info or {}).get("max_hold_bars", getattr(governor, "active_omega5_max_hold_bars", 0)) or 0),
        "effective_take_profit": float(getattr(governor, "active_omega5_take_profit", 0.0) or 0.0),
        "effective_stop_loss": float(getattr(governor, "active_omega5_stop_loss", 0.0) or 0.0),
        "omega5_source_roundtrip_cost": float(
            (info or {}).get("omega5_source_roundtrip_cost", getattr(governor, "active_omega5_roundtrip_cost", 0.0)) or 0.0
        ),
        "omega5_source_exit_reason": str(
            (info or {}).get("omega5_source_exit_reason", getattr(governor, "active_omega5_source_exit_reason", "")) or ""
        ),
        "omega5_source_exit_price_move": float(
            (info or {}).get(
                "omega5_source_exit_price_move",
                getattr(governor, "active_omega5_source_exit_price_move", 0.0),
            )
            or 0.0
        ),
        "risk_source": "omega5_active_risk" if str((info or {}).get("owner", "")).lower() == "omega5" else "",
    }


def metrics_from_returns(returns: list[float]) -> dict[str, Any]:
    arr = np.asarray(returns, dtype=np.float64)
    if len(arr) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "trades": 0, "wr": 0.0}
    curve = np.concatenate([[1.0], np.cumprod(1.0 + arr)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, EPS) - 1.0
    return {
        "pnl": float((curve[-1] - 1.0) * 100.0),
        "mdd": float(dd.min() * 100.0),
        "trades": int(len(arr)),
        "wr": float((arr > 0.0).mean()),
    }


def normalize_reason(reason: Any) -> str:
    text = str(reason or "")
    for prefix in ("omega5_", "omega462_"):
        if text.startswith(prefix):
            return text[len(prefix) :]
    if text in {"omega5_max_hold", "max_hold", "omega5_parent_final", "parent_final", "roll8_time_exit", "roll8_final"}:
        return "time_exit"
    if text in {"omega5_take_profit", "take_profit", "roll8_bracket_tp"}:
        return "take_profit"
    if text in {"omega5_stop_loss", "stop_loss", "roll8_bracket_sl"}:
        return "stop_loss"
    return text


def compare_to_expected(split: str, observed: pd.DataFrame, *, window_start: pd.Timestamp, window_end: pd.Timestamp) -> dict[str, Any]:
    expected = pd.read_csv(EXPECTED_LEDGERS[split])
    expected = expected[pd.to_numeric(expected["notional"], errors="coerce").fillna(0.0) > EPS].reset_index(drop=True)
    expected_entry_ts = pd.to_datetime(expected["entry_timestamp"], errors="coerce")
    expected_exit_ts = pd.to_datetime(expected["exit_timestamp"], errors="coerce")
    expected = expected[
        (expected_entry_ts >= window_start)
        & (expected_exit_ts <= window_end)
    ].reset_index(drop=True)
    obs = observed.reset_index(drop=True)
    n = min(len(expected), len(obs))
    checks: list[dict[str, Any]] = []
    checks.append({"name": "active_trade_count", "pass": len(expected) == len(obs), "details": {"expected": len(expected), "observed": len(obs)}})
    if n == 0:
        return {"pass": len(expected) == len(obs), "checks": checks}

    exp_side = pd.to_numeric(expected["side"], errors="coerce").astype(int).iloc[:n].to_numpy()
    obs_side = pd.to_numeric(obs["side"], errors="coerce").astype(int).iloc[:n].to_numpy()
    checks.append({"name": "side_sequence", "pass": bool(np.array_equal(exp_side, obs_side)), "details": {"mismatch_count": int((exp_side != obs_side).sum())}})

    for col in ("entry_timestamp", "exit_timestamp"):
        left = pd.to_datetime(expected[col].iloc[:n], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        right = pd.to_datetime(obs[col].iloc[:n], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
        checks.append({"name": f"{col}_sequence", "pass": bool((left == right).all()), "details": {"mismatch_count": int((left != right).sum())}})

    exp_reason = expected["reason"].map(normalize_reason).astype(str).iloc[:n].reset_index(drop=True)
    obs_reason = obs["reason"].map(normalize_reason).astype(str).iloc[:n].reset_index(drop=True)
    checks.append({"name": "reason_sequence", "pass": bool((exp_reason == obs_reason).all()), "details": {"mismatch_count": int((exp_reason != obs_reason).sum())}})

    numeric_diffs: dict[str, float] = {}
    for col in ("notional", "margin_fraction", "leverage", "trade_return"):
        left = pd.to_numeric(expected[col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        right = pd.to_numeric(obs[col].iloc[:n], errors="coerce").to_numpy(dtype=np.float64)
        numeric_diffs[col] = float(np.nanmax(np.abs(left - right))) if n else 0.0
    checks.append({"name": "numeric_sequence", "pass": all(v <= 5.0e-6 for v in numeric_diffs.values()), "details": numeric_diffs})
    return {"pass": all(c["pass"] for c in checks), "checks": checks}


def run_split(tb: Any, split: str, *, max_bars: int | None, window_bars: int, progress_every: int) -> dict[str, Any]:
    reset_state()
    split_dir = OUT_DIR / split
    split_dir.mkdir(parents=True, exist_ok=True)
    frame = prepare_replay_frame(load_enriched_frame(split))
    start, end = split_index_bounds(frame, split, max_bars)
    router = tb.GovernorPositionRouter()
    governor = tb.FinalGovernorRuntime()
    decisions: list[dict[str, Any]] = []
    normalized_closes: list[dict[str, Any]] = []
    journal_rows: list[dict[str, Any]] = []
    returns: list[float] = []
    current_trade: dict[str, Any] | None = None
    journal_path = Path(os.environ["TRADE_JOURNAL_PATH"])

    for i in range(start, end):
        if progress_every and (i - start) % progress_every == 0:
            print(json.dumps({"split": split, "bar": int(i), "done": int(i - start), "total": int(end - start), "closed": len(normalized_closes)}, ensure_ascii=False), flush=True)
        router.decrement_cooldown()
        signal_row = frame.iloc[i]
        execution_row = frame.iloc[i]
        model_start = max(0, i + 1 - int(window_bars))
        processed_df = frame.iloc[model_start : i + 1].copy()
        decision_price = bar_float(signal_row, "close", 0.0)
        execution_price = bar_float(execution_row, "close", decision_price)
        decision_time_kst = bar_time_kst(signal_row)
        execution_time_kst = bar_time_kst(execution_row)
        prev_pos = router.pos
        prev_snapshot = router.position_snapshot()
        action, exposure, fraction, exec_lev, info, regime = governor.decide(
            processed_df=processed_df,
            meta_router=router,
            current_price=decision_price,
            m7_last=None,
            trend_signal=None,
        )
        info = dict(info or {})
        source = str(info.get("source", "FINAL_GOVERNOR") or "FINAL_GOVERNOR")
        reason = str(info.get("position_reason", info.get("reason", source)) or source)
        if int(action) == 0 and prev_pos is not None:
            override_price = float(info.get("execution_price_override", 0.0) or 0.0)
            if override_price > 0.0 and np.isfinite(override_price):
                execution_price = override_price

        router._update_pos(
            int(action),
            float(execution_price),
            execution_time_kst,
            float(exposure),
            fraction=float(fraction),
            leverage_mult=float(exec_lev),
            trend_signal=None,
            entry_price_source_override="next_bar_open",
            entry_decision_price_override=float(decision_price),
            entry_execution_liquidity_override=REPLAY_EXECUTION_LIQUIDITY,
            entry_execution_route_override=REPLAY_EXECUTION_ROUTE,
            entry_execution_order_type_override=REPLAY_EXECUTION_ORDER_TYPE,
        )
        router.update_adaptive_gate(final_action=int(action), in_position=(router.pos is not None))
        new_pos = router.pos
        new_snapshot = router.position_snapshot()
        prev_snapshot.update(risk_fields(governor, prev_snapshot, info))
        new_snapshot.update(risk_fields(governor, new_snapshot, info))
        risk = new_snapshot if new_pos is not None else prev_snapshot
        audit = audit_context(
            decision_row=signal_row,
            execution_row=execution_row,
            decision_time_kst=decision_time_kst,
            execution_time_kst=execution_time_kst,
            decision_price=decision_price,
            execution_price=execution_price,
            info=info,
            risk=dict(risk),
        )
        transition = transition_label(tb, prev_pos, new_pos)

        if prev_pos is not None and new_pos != prev_pos:
            close_payload = router.build_close_trade_payload(
                snapshot=prev_snapshot,
                current_price=float(execution_price),
                timestamp_kst=execution_time_kst,
                event=transition,
                regime_name=str(regime),
                source=source,
                reason=reason,
                next_side=new_pos,
                audit_context=audit,
            )
            close_payload = apply_close_accounting(router, close_payload)
            realized = float(close_payload.get("pnl_frac", 0.0) or 0.0)
            returns.append(realized)
            router.record_outcome(realized)
            router.append_trade_history(execution_time_kst, realized, payload=close_payload)
            journal_rows.append(dict(close_payload))
            append_jsonl(journal_path, close_payload)
            normalized_closes.append(
                {
                    "entry_timestamp": str((current_trade or {}).get("entry_timestamp", "")),
                    "exit_timestamp": str(execution_row.get("timestamp", "")),
                    "side": 1 if str(prev_pos).upper() == "LONG" else -1,
                    "reason": normalize_reason(reason),
                    "notional": float(prev_snapshot.get("notional_exposure", prev_snapshot.get("total_exposure", 0.0)) or 0.0),
                    "margin_fraction": float(prev_snapshot.get("position_fraction", 0.0) or 0.0),
                    "leverage": float(prev_snapshot.get("execution_leverage", prev_snapshot.get("leverage_mult", 0.0)) or 0.0),
                    "trade_return": realized,
                    "entry_price": float((current_trade or {}).get("entry_price", 0.0) or 0.0),
                    "exit_price": float(execution_price),
                    "source": source,
                }
            )
            current_trade = None

        if new_pos is not None and new_pos != prev_pos:
            open_payload = router.build_open_trade_payload(
                snapshot=new_snapshot,
                timestamp_kst=execution_time_kst,
                event=transition,
                regime_name=str(regime),
                source=source,
                reason=reason,
                audit_context=audit,
            )
            journal_rows.append(dict(open_payload))
            append_jsonl(journal_path, open_payload)
            current_trade = {
                "entry_timestamp": str(execution_row.get("timestamp", "")),
                "entry_price": float(execution_price),
                "side": str(new_pos),
            }

        decisions.append(
            {
                "row": int(i),
                "decision_timestamp": str(signal_row.get("timestamp", "")),
                "execution_timestamp": str(execution_row.get("timestamp", "")),
                "action": int(action),
                "target_exposure": float(exposure),
                "target_fraction": float(fraction),
                "target_exec_leverage": float(exec_lev),
                "prev_pos": prev_pos,
                "new_pos": new_pos,
                "source": source,
                "reason": reason,
                "regime": str(regime),
            }
        )

    decisions_df = pd.DataFrame(decisions)
    closes_df = pd.DataFrame(normalized_closes)
    journal_df = pd.DataFrame(journal_rows)
    decisions_path = split_dir / "runtime_native_decisions.csv"
    closes_path = split_dir / "runtime_native_closes_normalized.csv"
    journal_path_csv = split_dir / "runtime_native_journal.csv"
    decisions_df.to_csv(decisions_path, index=False)
    closes_df.to_csv(closes_path, index=False)
    journal_df.to_csv(journal_path_csv, index=False)
    comparison = compare_to_expected(
        split,
        closes_df,
        window_start=pd.Timestamp(frame.iloc[start]["timestamp"]),
        window_end=pd.Timestamp(frame.iloc[end]["timestamp"]),
    )
    return {
        "split": split,
        "replay_bounds": {"start": int(start), "end": int(end), "bars": int(end - start)},
        "metrics": metrics_from_returns(returns),
        "comparison": comparison,
        "artifacts": {
            "decisions": str(decisions_path),
            "closes": str(closes_path),
            "journal_csv": str(journal_path_csv),
            "journal_jsonl": str(Path(os.environ["TRADE_JOURNAL_PATH"])),
            "expected_ledger": str(EXPECTED_LEDGERS[split]),
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Omega5 Source Parent Runtime-Native Walk-Forward Proof - 2026-07-01",
        "",
        f"- Verdict: `{payload['verdict']}`",
        f"- Runtime-native proof pass: `{payload['runtime_native_proof_pass']}`",
        f"- Source report: `{SOURCE_REPORT}`",
        f"- Isolated state dir: `{STATE_DIR}`",
        "",
        "## Splits",
        "",
        "| Split | Pass | Bars | Trades | PnL | MDD | WR |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in payload.get("splits", []):
        m = row["metrics"]
        lines.append(
            f"| `{row['split']}` | `{row['comparison']['pass']}` | `{row['replay_bounds']['bars']}` | "
            f"`{m['trades']}` | `{m['pnl']:.4f}%` | `{m['mdd']:.4f}%` | `{m['wr']:.4f}` |"
        )
    lines.extend(["", "## Failed Checks", ""])
    failed: list[str] = []
    for split in payload.get("splits", []):
        for check in split.get("comparison", {}).get("checks", []):
            if not check.get("pass"):
                failed.append(f"- `{split['split']}` / `{check['name']}`: {check.get('details')}")
    lines.extend(failed or ["- None."])
    if payload.get("error"):
        lines.extend(["", "## Error", "", f"- `{payload.get('error_type')}`: {payload.get('error')}"])
    lines.extend(["", "## Artifacts", "", f"- JSON: `{REPORT_PATH}`"])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["validation", "oos", "both"], default="both")
    p.add_argument("--max-bars", type=int, default=0, help="0 means full split")
    p.add_argument("--window-bars", type=int, default=7000)
    p.add_argument("--progress-every", type=int, default=1000)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    _set_isolated_live_env(window_bars=int(args.window_bars))
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

    import trading_bot as tb  # noqa: WPS433

    selected = ["validation", "oos"] if args.split == "both" else [args.split]
    payload: dict[str, Any] = {
        "audit_id": "omega5_source_parent_runtime_native_walkforward_20260701",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_report": str(SOURCE_REPORT),
        "source_model": read_json(SOURCE_REPORT).get("model_id"),
        "decision_entrypoint": "trading_bot.FinalGovernorRuntime.decide",
        "ledger_replay_used_for_decisions": False,
        "isolated_state_dir": str(STATE_DIR),
        "args": {
            "split": args.split,
            "max_bars": int(args.max_bars),
            "window_bars": int(args.window_bars),
            "progress_every": int(args.progress_every),
        },
        "splits": [],
    }
    try:
        for split in selected:
            result = run_split(
                tb,
                split,
                max_bars=None if int(args.max_bars) <= 0 else int(args.max_bars),
                window_bars=int(args.window_bars),
                progress_every=int(args.progress_every),
            )
            payload["splits"].append(result)
        proof_pass = bool(payload["splits"] and all(row["comparison"]["pass"] for row in payload["splits"]))
        payload["runtime_native_proof_pass"] = proof_pass
        payload["verdict"] = "OMEGA5_RUNTIME_NATIVE_PROOF_PASS" if proof_pass else "OMEGA5_RUNTIME_NATIVE_PROOF_FAIL"
        write_json(REPORT_PATH, payload)
        REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
        print(json.dumps({"report": str(REPORT_PATH), "markdown": str(REPORT_MD), "verdict": payload["verdict"]}, ensure_ascii=False))
        return 0 if proof_pass else 1
    except Exception as exc:
        payload.update(
            {
                "runtime_native_proof_pass": False,
                "verdict": "OMEGA5_RUNTIME_NATIVE_PROOF_ERROR",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        write_json(REPORT_PATH, payload)
        REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
        REPORT_MD.write_text(render_markdown(payload), encoding="utf-8")
        print(json.dumps({"report": str(REPORT_PATH), "markdown": str(REPORT_MD), "verdict": payload["verdict"], "error": str(exc)}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
