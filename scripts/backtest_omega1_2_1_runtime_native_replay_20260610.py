#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

MODEL_ID = "omega1_2_1_runtime_native_replay_20260610"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
STATE_DIR = OUT_DIR / "isolated_live_state"
REPLAY_EXECUTION_LIQUIDITY = "signal_immediate_maker_limit"
REPLAY_EXECUTION_ROUTE = "runtime_native_next_open_maker_limit"
REPLAY_EXECUTION_ORDER_TYPE = "LIMIT_MAKER"


def _initial_out_dir_from_argv() -> Path | None:
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg == "--out-dir" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
        if arg.startswith("--out-dir="):
            return Path(arg.split("=", 1)[1]).resolve()
    return None


_ARGV_OUT_DIR = _initial_out_dir_from_argv()
if _ARGV_OUT_DIR is not None:
    OUT_DIR = _ARGV_OUT_DIR
    STATE_DIR = OUT_DIR / "isolated_live_state"


def _set_isolated_live_env() -> None:
    global STATE_DIR
    out_dir_env = os.getenv("OMEGA121_RUNTIME_NATIVE_OUT_DIR")
    if out_dir_env:
        STATE_DIR = Path(out_dir_env) / "isolated_live_state"
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    env_paths = {
        "TRADING_BOT_PROCESS_LOCK_PATH": STATE_DIR / "trading_bot_replay.lock",
        "GOVERNOR_LIVE_STATE_PATH": STATE_DIR / "governor_live_state.json",
        "FINAL_GOVERNOR_RUNTIME_STATE_PATH": STATE_DIR / "final_governor_runtime_state.json",
        "TRADE_JOURNAL_PATH": STATE_DIR / "trade_journal.jsonl",
        "POSITION_ACCOUNTING_AUDIT_PATH": STATE_DIR / "position_accounting_audit.jsonl",
        "DASHBOARD_EVENTS_PATH": STATE_DIR / "dashboard_events.jsonl",
        "TP_RUNNER_SHADOW_PARITY_PATH": STATE_DIR / "tp_runner_shadow_parity.jsonl",
        "FINAL_GOVERNOR_PENDING_NEXT_OPEN_PATH": STATE_DIR / "pending_next_open_intent.json",
        "OMEGA123_CASH_SLEEVE_TELEMETRY_PATH": STATE_DIR / "omega_cash_sleeve_decisions.jsonl",
    }
    for key, path in env_paths.items():
        os.environ[key] = str(path)
    os.environ.setdefault("FINAL_GOVERNOR_NEXT_OPEN_EXECUTION_ENABLE", "1")
    os.environ.setdefault("FINAL_GOVERNOR_ALLOW_LATE_NEXT_OPEN_SHADOW_EXECUTION", "1")


_set_isolated_live_env()

import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as split_builder  # noqa: E402
import trading_bot as tb  # noqa: E402
from features.engineering import FeatureEngineer  # noqa: E402
from features.high_order_state import add_high_order_state_features  # noqa: E402


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


def _reset_isolated_state() -> None:
    for path in STATE_DIR.glob("*"):
        if path.is_file() and path.name != "trading_bot_replay.lock":
            path.unlink()


def _bar_time_kst(row: pd.Series) -> pd.Timestamp:
    return pd.Timestamp(row.get("timestamp")) + pd.Timedelta(hours=9)


def _bar_float(row: pd.Series, key: str, default: float = 0.0) -> float:
    return float(tb._safe_float(row.get(key, default), default))


def _transition_label(prev_pos: str | None, cur_pos: str | None) -> str:
    return tb._pos_transition_label(prev_pos, cur_pos)


def _inject_elite_signals(processed_df: pd.DataFrame, elite_runtime: Any) -> pd.DataFrame:
    if len(processed_df) == 0:
        return processed_df
    out = processed_df.copy()
    last_idx = out.index[-1]
    try:
        last = out.iloc[-1]
        prev = out.iloc[-2] if len(out) >= 2 else last
        smf_std = out["smart_money_flow"].std() if "smart_money_flow" in out.columns else 1.0
        cur = tb.row_to_market_row(last)
        prev_mkt = tb.row_to_market_row(prev)
        sigs = elite_runtime.compute_all(current=cur, prev=prev_mkt, smf_std=smf_std)
        for col, val in sigs.items():
            if isinstance(col, str) and col.startswith("sig_"):
                out.at[last_idx, col] = float(val)
    except Exception:
        # Live bot only logs this failure and continues; keep replay behavior aligned.
        return out
    return out


def _complete_replay_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "funding_z_score" not in out.columns:
        if "last_funding_rate" not in out.columns:
            raise RuntimeError("runtime-native replay cannot build funding_z_score: missing last_funding_rate")
        funding = pd.to_numeric(out["last_funding_rate"], errors="coerce").fillna(0.0)
        roll_mean = funding.rolling(window=288, min_periods=20).mean()
        roll_std = funding.rolling(window=288, min_periods=20).std().replace(0, np.nan)
        out["funding_z_score"] = ((funding - roll_mean) / roll_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if "regime_persistence" not in out.columns:
        out = add_high_order_state_features(out)
    return out


def _risk_fields(governor: Any, snapshot: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    side = str(snapshot.get("pos") or "").upper()
    entry = float(snapshot.get("entry_price", 0.0) or 0.0)
    exposure = float(snapshot.get("notional_exposure", snapshot.get("total_exposure", 0.0)) or 0.0)
    owner = str((info or {}).get("owner", getattr(governor, "owner", "") or "")).lower()
    source = str((info or {}).get("source", "")).lower()
    is_omega = owner == tb.OMEGA121_OWNER or source.startswith("omega1_2_1|")
    tp = float((info or {}).get("take_profit", 0.0) or 0.0)
    sl = float((info or {}).get("stop_loss", 0.0) or 0.0)
    if is_omega and tp <= 0.0:
        tp = float(getattr(governor, "active_omega1_2_1_take_profit", 0.0) or 0.0)
    if is_omega and sl <= 0.0:
        sl = float(getattr(governor, "active_omega1_2_1_stop_loss", 0.0) or 0.0)

    def price_from_threshold(threshold: float, take_profit: bool) -> float:
        if side not in {"LONG", "SHORT"} or entry <= 0.0 or exposure <= 0.0 or threshold <= 0.0:
            return 0.0
        raw_move = threshold / max(exposure, 1e-8)
        if side == "LONG":
            return float(entry * (1.0 + raw_move) if take_profit else entry * max(0.0, 1.0 - raw_move))
        return float(entry * max(0.0, 1.0 - raw_move) if take_profit else entry * (1.0 + raw_move))

    return {
        "take_profit": tp,
        "stop_loss": sl,
        "max_hold_bars": 0,
        "max_hold_remaining_bars": 0,
        "take_profit_price": price_from_threshold(tp, True),
        "tp_price": price_from_threshold(tp, True),
        "stop_price": price_from_threshold(sl, False),
        "sl_price": price_from_threshold(sl, False),
        "effective_take_profit": tp,
        "effective_stop_loss": sl,
        "risk_source": "omega1_2_1_active_risk" if is_omega else "",
    }


def _audit_context(
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
        "decision_bar_open": _bar_float(decision_row, "open", decision_price),
        "decision_bar_high": _bar_float(decision_row, "high", decision_price),
        "decision_bar_low": _bar_float(decision_row, "low", decision_price),
        "decision_bar_close": _bar_float(decision_row, "close", decision_price),
        "decision_bar_volume": _bar_float(decision_row, "volume", 0.0),
        "decision_bar_is_complete": True,
        "decision_price": float(decision_price),
        "decision_price_source": "eth_buffer.close[-2]",
        "execution_bar_ts": str(execution_time_kst),
        "execution_bar_utc": str(execution_row.get("timestamp", "")),
        "execution_bar_open": _bar_float(execution_row, "open", execution_price),
        "execution_bar_high": _bar_float(execution_row, "high", execution_price),
        "execution_bar_low": _bar_float(execution_row, "low", execution_price),
        "execution_bar_close": _bar_float(execution_row, "close", execution_price),
        "execution_bar_volume": _bar_float(execution_row, "volume", 0.0),
        "execution_bar_is_current": False,
        "execution_price": float(execution_price),
        "execution_price_source": "eth_buffer.open[-1]",
        "execution_delay_sec": 0.0,
        "execution_delay_late": False,
        "execution_delay_mode": "runtime_native_replay",
        "entry_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
        "entry_execution_route": REPLAY_EXECUTION_ROUTE,
        "entry_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
        "exit_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
        "exit_execution_route": REPLAY_EXECUTION_ROUTE,
        "exit_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
        "ai_timing": dict(trace.get("ai_timing", {}) or {}),
        "model_version": str(info.get("model_version", trace.get("model_version", "")) or ""),
        "model_id": str(info.get("model_id", trace.get("decision_logic", "")) or ""),
        "model_path": str(info.get("model_path", trace.get("model_path", "")) or ""),
        "model_sleeve": str(info.get("model_sleeve", trace.get("v21_sleeve", "")) or ""),
        **risk,
    }


def _metrics(equity_curve: list[float], closes: list[dict[str, Any]]) -> dict[str, Any]:
    eq = np.asarray(equity_curve or [1.0], dtype=np.float64)
    peak = np.maximum.accumulate(eq)
    dd = (eq / np.maximum(peak, 1e-12) - 1.0) * 100.0
    pnl = (float(eq[-1]) - 1.0) * 100.0
    rets = np.asarray([float(r.get("pnl_frac", 0.0) or 0.0) for r in closes], dtype=np.float64)
    return {
        "pnl": pnl,
        "mdd": float(dd.min()) if len(dd) else 0.0,
        "wr": float(np.mean(rets > 0.0)) if len(rets) else 0.0,
        "trades": int(len(rets)),
        "long_entries": int(sum(1 for r in closes if str(r.get("side", "")).upper() == "LONG")),
        "short_entries": int(sum(1 for r in closes if str(r.get("side", "")).upper() == "SHORT")),
    }


def _apply_replay_close_accounting(router: Any, payload: dict[str, Any]) -> dict[str, Any]:
    out = dict(payload or {})
    side = str(out.get("side", "") or "").upper()
    entry_price = float(out.get("entry_price", 0.0) or 0.0)
    exit_price = float(out.get("exit_price", out.get("current_price", 0.0)) or 0.0)
    exposure = float(out.get("notional_exposure", out.get("total_exposure", 0.0)) or 0.0)
    math = router._trade_math(
        side,
        entry_price,
        exit_price,
        exposure,
        entry_liquidity=REPLAY_EXECUTION_LIQUIDITY,
        exit_liquidity=REPLAY_EXECUTION_LIQUIDITY,
    )
    for key in (
        "entry_exec_price",
        "exit_exec_price",
        "gross_return_frac",
        "entry_fee_rate",
        "exit_fee_rate",
        "roundtrip_fee_rate",
        "entry_fee_model",
        "exit_fee_model",
        "fee_model",
        "fee_cost_frac",
        "pnl_frac",
        "pnl_pct",
    ):
        if key in math:
            out[key] = math[key]
    out["entry_execution_liquidity"] = REPLAY_EXECUTION_LIQUIDITY
    out["entry_execution_route"] = REPLAY_EXECUTION_ROUTE
    out["entry_execution_order_type"] = REPLAY_EXECUTION_ORDER_TYPE
    out["exit_execution_liquidity"] = REPLAY_EXECUTION_LIQUIDITY
    out["exit_execution_route"] = REPLAY_EXECUTION_ROUTE
    out["exit_execution_order_type"] = REPLAY_EXECUTION_ORDER_TYPE
    return out


def _run_split(
    name: str,
    frame: pd.DataFrame,
    *,
    warmup: int,
    max_bars: int | None,
    model_bars: int,
    progress_every: int,
    prehistory: pd.DataFrame | None = None,
) -> dict[str, Any]:
    _reset_isolated_state()
    split_dir = OUT_DIR / name
    split_dir.mkdir(parents=True, exist_ok=True)

    router = tb.GovernorPositionRouter()
    governor = tb.FinalGovernorRuntime()
    elite_runtime = tb.EliteSignals()
    trend_hub = tb.SevenModelEnsemble(strict=False)

    eval_rows = frame.reset_index(drop=True).copy()
    pre_rows = pd.DataFrame() if prehistory is None else prehistory.reset_index(drop=True).copy()
    if len(pre_rows):
        rows = pd.concat([pre_rows, eval_rows], ignore_index=True)
        eval_start = int(len(pre_rows))
    else:
        rows = eval_rows
        eval_start = int(warmup)
    rows = FeatureEngineer(keep_only_active=False)._create_directional_alpha_features(rows)
    rows = _complete_replay_features(rows)
    rows = rows.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
    end = len(rows) - 1
    if max_bars is not None and max_bars > 0:
        end = min(end, eval_start + int(max_bars))
    if end <= eval_start:
        raise RuntimeError(f"{name}: insufficient bars for runtime-native replay: len={len(rows)} eval_start={eval_start}")

    equity = 1.0
    equity_curve: list[float] = [equity]
    decisions: list[dict[str, Any]] = []
    journal_rows: list[dict[str, Any]] = []
    close_rows: list[dict[str, Any]] = []

    for i in range(int(eval_start), int(end)):
        if progress_every > 0 and (i - int(eval_start)) % int(progress_every) == 0:
            print(
                json.dumps(
                    {
                        "progress": name,
                        "row": int(i),
                        "eval_row": int(i - int(eval_start)),
                        "done": int(i - int(eval_start)),
                        "total": int(end - eval_start),
                        "closed_trades": int(len(close_rows)),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
        router.decrement_cooldown()
        signal_row = rows.iloc[i]
        execution_row = rows.iloc[i + 1]
        model_start = max(0, i + 1 - int(max(600, model_bars)))
        processed_df = rows.iloc[model_start : i + 1].copy()
        processed_df = _inject_elite_signals(processed_df, elite_runtime)

        decision_time_kst = _bar_time_kst(signal_row)
        execution_time_kst = _bar_time_kst(execution_row)
        decision_price = _bar_float(signal_row, "close", 0.0)
        execution_price = _bar_float(execution_row, "open", decision_price)
        if decision_price <= 0.0:
            raise RuntimeError(f"{name}: invalid decision price at row={i}")
        if execution_price <= 0.0:
            execution_price = decision_price

        m7_last = trend_hub.predict_last(processed_df)
        trend_signal = tb._trend_signal_from_m7(m7_last)
        prev_pos = router.pos
        prev_snapshot = router.position_snapshot()
        action, exposure, fraction, exec_lev, info, regime = governor.decide(
            processed_df=processed_df,
            meta_router=router,
            current_price=decision_price,
            m7_last=m7_last,
            trend_signal=trend_signal,
        )
        info = dict(info or {})
        info.setdefault("source", "FINAL_GOVERNOR")
        info.setdefault("position_reason", str(info.get("reason", "")))

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
        prev_snapshot.update(_risk_fields(governor, prev_snapshot, info))
        new_snapshot.update(_risk_fields(governor, new_snapshot, info))
        risk = new_snapshot if new_pos is not None else prev_snapshot
        audit = _audit_context(
            decision_row=signal_row,
            execution_row=execution_row,
            decision_time_kst=decision_time_kst,
            execution_time_kst=execution_time_kst,
            decision_price=decision_price,
            execution_price=execution_price,
            info=info,
            risk=dict(risk),
        )
        transition = _transition_label(prev_pos, new_pos)
        source = str(info.get("source", "FINAL_GOVERNOR") or "FINAL_GOVERNOR")
        reason = str(info.get("position_reason", info.get("reason", source)) or source)

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
            close_payload = _apply_replay_close_accounting(router, close_payload)
            realized = float(close_payload.get("pnl_frac", 0.0) or 0.0)
            equity *= max(0.0, 1.0 + realized)
            router.record_outcome(realized)
            router.append_trade_history(execution_time_kst, realized, payload=close_payload)
            journal_rows.append(dict(close_payload))
            close_rows.append(dict(close_payload))

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

        mark = equity
        if router.pos is not None:
            mark *= max(0.0, 1.0 + float(router._net_pnl_frac(float(execution_price))))
        equity_curve.append(float(mark))
        decisions.append(
            {
                "row": int(i),
                "eval_row": int(i - int(eval_start)),
                "decision_timestamp": str(signal_row.get("timestamp", "")),
                "execution_timestamp": str(execution_row.get("timestamp", "")),
                "decision_time_kst": str(decision_time_kst),
                "execution_time_kst": str(execution_time_kst),
                "decision_price": float(decision_price),
                "execution_price": float(execution_price),
                "action": int(action),
                "target_exposure": float(exposure),
                "target_fraction": float(fraction),
                "target_exec_leverage": float(exec_lev),
                "prev_pos": prev_pos,
                "new_pos": new_pos,
                "source": source,
                "reason": reason,
                "regime": str(regime),
                "equity_mark": float(mark),
            }
        )

    if router.pos is not None:
        last_row = rows.iloc[int(end)]
        ts_kst = _bar_time_kst(last_row)
        price = _bar_float(last_row, "close", 0.0)
        prev_pos = router.pos
        prev_snapshot = router.position_snapshot()
        router._update_pos(0, price, ts_kst, 0.0, fraction=0.0, leverage_mult=1.0)
        close_payload = router.build_close_trade_payload(
            snapshot=prev_snapshot,
            current_price=float(price),
            timestamp_kst=ts_kst,
            event=_transition_label(prev_pos, None),
            regime_name="FORCE_END",
            source="runtime_native_replay",
            reason="force_end",
            next_side=None,
            audit_context={
                "execution_price": float(price),
                "execution_price_source": "final_close",
                "decision_price": float(price),
                "entry_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
                "entry_execution_route": REPLAY_EXECUTION_ROUTE,
                "entry_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
                "exit_execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
                "exit_execution_route": REPLAY_EXECUTION_ROUTE,
                "exit_execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
            },
        )
        close_payload = _apply_replay_close_accounting(router, close_payload)
        realized = float(close_payload.get("pnl_frac", 0.0) or 0.0)
        equity *= max(0.0, 1.0 + realized)
        router.record_outcome(realized)
        router.append_trade_history(ts_kst, realized, payload=close_payload)
        journal_rows.append(dict(close_payload))
        close_rows.append(dict(close_payload))
        equity_curve.append(float(equity))

    decisions_df = pd.DataFrame(decisions)
    journal_df = pd.DataFrame(journal_rows)
    closes_df = pd.DataFrame(close_rows)
    decisions_path = split_dir / "runtime_native_decisions.csv"
    journal_path = split_dir / "runtime_native_trade_journal.csv"
    closes_path = split_dir / "runtime_native_closes.csv"
    decisions_df.to_csv(decisions_path, index=False)
    journal_df.to_csv(journal_path, index=False)
    closes_df.to_csv(closes_path, index=False)

    shadow_path = Path(os.environ["TP_RUNNER_SHADOW_PARITY_PATH"])
    split_shadow_path = split_dir / "tp_runner_shadow_parity.jsonl"
    shadow_rows = 0
    if shadow_path.exists():
        text = shadow_path.read_text(encoding="utf-8")
        split_shadow_path.write_text(text, encoding="utf-8")
        shadow_rows = sum(1 for line in text.splitlines() if line.strip())
    else:
        split_shadow_path.write_text("", encoding="utf-8")

    metrics = _metrics(equity_curve, close_rows)
    return {
        "split": name,
        "metrics": metrics,
        "bars_processed": int(end - eval_start),
        "prehistory_bars": int(len(pre_rows)),
        "decisions": str(decisions_path),
        "journal": str(journal_path),
        "closes": str(closes_path),
        "tp_runner_shadow_parity": str(split_shadow_path),
        "tp_runner_shadow_rows": int(shadow_rows),
    }


def main() -> int:
    global OUT_DIR, STATE_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["validation", "oos", "both"], default="both")
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--max-bars", type=int, default=0, help="0 means full split")
    parser.add_argument("--model-bars", type=int, default=int(tb.FINAL_GOVERNOR_LIVE_MODEL_BARS))
    parser.add_argument("--progress-every", type=int, default=500)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    if args.out_dir is not None:
        OUT_DIR = args.out_dir.resolve()
        STATE_DIR = OUT_DIR / "isolated_live_state"
        _set_isolated_live_env()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUT_DIR / "summary.json"
    failure_path = OUT_DIR / "failure.json"
    results: list[dict[str, Any]] = []
    summary_base = {
        "model_id": MODEL_ID,
        "live_model_id": tb.OMEGA121_MODEL_ID,
        "live_owner": tb.OMEGA121_OWNER,
        "contract": "runtime_native_signal_close_next_open_replay",
        "accounting_alignment": {
            "execution_liquidity": REPLAY_EXECUTION_LIQUIDITY,
            "execution_route": REPLAY_EXECUTION_ROUTE,
            "execution_order_type": REPLAY_EXECUTION_ORDER_TYPE,
            "reason": "match research backtest next-open maker-limit route while preserving live FinalGovernorRuntime decision path",
        },
        "isolated_state_dir": str(STATE_DIR),
        "args": {
            "split": str(args.split),
            "warmup": int(args.warmup),
            "max_bars": int(args.max_bars),
            "model_bars": int(args.model_bars),
            "progress_every": int(args.progress_every),
            "out_dir": str(OUT_DIR),
        },
    }
    try:
        splits = split_builder._build_splits()
        selected = ["validation", "oos"] if args.split == "both" else [args.split]
        for name in selected:
            prehistory = None
            if name == "oos":
                prehistory = splits["validation"]["frame"].tail(int(max(600, args.model_bars))).copy()
            result = _run_split(
                name,
                splits[name]["frame"],
                warmup=int(args.warmup),
                max_bars=(None if int(args.max_bars) <= 0 else int(args.max_bars)),
                model_bars=int(args.model_bars),
                progress_every=int(args.progress_every),
                prehistory=prehistory,
            )
            results.append(result)
            print(json.dumps(result, ensure_ascii=False, default=_json_default))
        summary = {**summary_base, "status": "ok", "results": results}
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(json.dumps({"summary": str(summary_path), "results": results}, ensure_ascii=False, default=_json_default))
        return 0
    except Exception as exc:
        failure = {
            **summary_base,
            "status": "failed",
            "results": results,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        failure_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        summary_path.write_text(json.dumps(failure, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
        print(json.dumps({"failure": str(failure_path), "error": str(exc)}, ensure_ascii=False, default=_json_default))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
