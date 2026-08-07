#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega5_event_risk_governor_20260702"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega5_live_duckdb_forensics_20260702"
REPORT_JSON = OUT_DIR / "report.json"
EVENT_CSV = OUT_DIR / "omega5_live_duckdb_forensics_events.csv"
REPORT_MD = ROOT / "docs/audits/omega5_live_duckdb_forensics_20260702.md"

CORE_DECISION_FEATURES = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "bb_width",
    "m7_prob_up",
    "m7_prob_dn",
    "m7_confidence",
    "m7_action",
    "ai_dir_edge",
    "pred_patchtst",
    "conf_patchtst",
    "rsi",
    "net_taker_ratio",
    "ofi_acceleration",
    "liquidity_vacuum",
    "execution_quality",
    "regime3_current_sensitive_wide24_bull_prob",
    "regime3_current_sensitive_wide24_bear_prob",
    "regime3_current_sensitive_wide24_chop_prob",
]

MICRO_COLS = [
    "ts",
    "obi",
    "taker_buy_ratio",
    "nif_whale",
    "nif_retail",
    "eai",
    "oi_delta_pct",
    "funding_rate",
    "kelly_mult",
    "signal_bias",
    "shadow_toxicity_score",
    "shadow_queue_collapse",
    "shadow_absorption_score",
    "shadow_queue_bias",
    "shadow_regime_tag",
    "shadow_regime_conf",
    "data_stale",
    "depth_connected",
    "trade_connected",
    "poll_connected",
    "depth_age_sec",
    "trade_age_sec",
    "poll_age_sec",
    "recent_trade_count_5m",
    "recent_trade_notional_5m",
    "recent_whale_count_5m",
    "valid_taker_flow",
    "valid_nif",
    "warmup_30m_ready",
    "schema_version",
]

TAIL_COLS = [
    "ts",
    "long_usd_1m",
    "short_usd_1m",
    "mu_long",
    "sigma_long",
    "mu_short",
    "sigma_short",
    "shadow_aftershock_prob",
    "shadow_decay_half_life",
    "shadow_risk_bucket",
    "ws_connected",
    "ws_stale",
    "ws_age_sec",
    "liq_event_count_1m",
    "valid_liq_stream",
    "schema_version",
]


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except Exception:
        return int(default)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _parse_kst(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or value == "":
        return pd.NaT
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        return pd.NaT
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Seoul")
    return ts.tz_convert("Asia/Seoul")


def _fmt_ts(ts: Any) -> str:
    if ts is None or pd.isna(ts):
        return ""
    return str(pd.Timestamp(ts).tz_convert("Asia/Seoul") if pd.Timestamp(ts).tzinfo else pd.Timestamp(ts))


def _is_omega5(row: dict[str, Any]) -> bool:
    model_id = str(row.get("model_id", row.get("open_model_id", "")) or "")
    open_model_id = str(row.get("open_model_id", "") or "")
    source = str(row.get("source", row.get("open_source", "")) or "")
    return model_id == MODEL_ID or open_model_id == MODEL_ID or source.startswith("omega5|")


def _load_omega5_journal(path: Path) -> pd.DataFrame:
    rows = [r for r in _read_jsonl(path) if _is_omega5(r)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in ("ts", "decision_bar_ts", "execution_bar_ts", "event_recorded_at", "decision_at"):
        if col in df.columns:
            df[f"_{col}_kst"] = df[col].map(_parse_kst)
    if "_decision_bar_ts_kst" not in df.columns:
        df["_decision_bar_ts_kst"] = pd.NaT
    return df.sort_values(["_decision_bar_ts_kst", "kind"], na_position="last").reset_index(drop=True)


def _extract_snapshot(row: dict[str, Any]) -> dict[str, Any]:
    values = dict(row.get("values") or {})
    decision = dict(row.get("decision") or {})
    trace = dict(row.get("sleeve_trace") or {})
    parent_trace = dict(trace.get("parent_trace") or {})
    components = parent_trace.get("component_predictions")
    if not isinstance(components, list):
        components = []
    best_component = next((c for c in components if int(c.get("side", 0) or 0) != 0), components[0] if components else {})
    out: dict[str, Any] = {
        "snapshot_created_at": row.get("created_at", ""),
        "snapshot_timestamp": row.get("timestamp", values.get("timestamp", "")),
        "snapshot_feature_hash": row.get("feature_hash_sha256", ""),
        "snapshot_decision_action": decision.get("action"),
        "snapshot_decision_source": decision.get("source", ""),
        "snapshot_position_signal": decision.get("position_signal", ""),
        "snapshot_position_reason": decision.get("position_reason", ""),
        "snapshot_omega5_reason": trace.get("omega5_reason", ""),
        "snapshot_parent_action": trace.get("parent_action"),
        "snapshot_parent_side": trace.get("parent_side"),
        "snapshot_parent_notional": trace.get("parent_notional_exposure"),
        "snapshot_parent_quality": trace.get("parent_quality_score"),
        "snapshot_parent_confidence": trace.get("parent_confidence"),
        "snapshot_parent_router_expert": trace.get("parent_router_expert", ""),
        "snapshot_parent_ledger_replay_used": parent_trace.get("ledger_replay_used"),
        "snapshot_source_parent_live_native_adapter": parent_trace.get("source_parent_live_native_adapter"),
        "snapshot_source_parent_predictive_artifact": parent_trace.get("source_parent_predictive_artifact", ""),
        "snapshot_component_alias": best_component.get("alias", ""),
        "snapshot_component_expert": best_component.get("expert", ""),
        "snapshot_component_final_action": best_component.get("final_action"),
        "snapshot_component_side": best_component.get("side"),
        "snapshot_component_quality": best_component.get("quality_for_action"),
        "snapshot_component_confidence": best_component.get("confidence"),
        "snapshot_component_notional": best_component.get("notional"),
        "snapshot_component_margin": best_component.get("margin_fraction"),
        "snapshot_component_leverage": best_component.get("leverage"),
        "snapshot_component_sidecar_score": best_component.get("sidecar_score"),
    }
    for col in CORE_DECISION_FEATURES:
        out[f"feat_{col}"] = values.get(col)
    return out


def _load_decision_snapshots(path: Path) -> pd.DataFrame:
    rows = [_extract_snapshot(r) for r in _read_jsonl(path)]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["_snapshot_ts_kst"] = df["snapshot_timestamp"].map(_parse_kst)
    df = df.dropna(subset=["_snapshot_ts_kst"]).sort_values("_snapshot_ts_kst").reset_index(drop=True)
    return df


def _copy_or_connect_db(db_path: Path, out_dir: Path, max_copy_mb: float) -> tuple[Path | None, dict[str, Any]]:
    meta: dict[str, Any] = {"path": str(db_path), "exists": db_path.exists(), "copied": False, "copy_path": ""}
    if not db_path.exists():
        return None, meta
    size_mb = db_path.stat().st_size / (1024 * 1024)
    meta["size_mb"] = size_mb
    try:
        con = duckdb.connect(str(db_path), read_only=True)
        con.close()
        meta["read_mode"] = "direct_read_only"
        return db_path, meta
    except Exception as exc:
        meta["direct_read_error"] = str(exc)
    if size_mb > max_copy_mb:
        meta["read_mode"] = "skipped_large_locked_db"
        return None, meta
    copy_dir = out_dir / "duckdb_snapshots"
    copy_dir.mkdir(parents=True, exist_ok=True)
    copy_path = copy_dir / db_path.name
    shutil.copy2(db_path, copy_path)
    meta["copied"] = True
    meta["copy_path"] = str(copy_path)
    meta["read_mode"] = "copied_after_lock"
    return copy_path, meta


def _table_exists(con: duckdb.DuckDBPyConnection, table: str) -> bool:
    return any(str(r[0]) == table for r in con.execute("SHOW TABLES").fetchall())


def _load_table(db_path: Path, table: str, wanted_cols: list[str], prefix: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"table": table, "available": False, "rows": 0, "columns": []}
    if not db_path.exists():
        meta["error"] = "db_missing"
        return pd.DataFrame(), meta
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        if not _table_exists(con, table):
            meta["error"] = "table_missing"
            return pd.DataFrame(), meta
        info = con.execute(f"PRAGMA table_info('{table}')").fetchall()
        cols = [str(r[1]) for r in info]
        selected = [c for c in wanted_cols if c in cols]
        if "ts" not in selected:
            meta["error"] = "ts_missing"
            return pd.DataFrame(), meta
        sql_cols = ", ".join(selected)
        df = con.execute(f"SELECT {sql_cols} FROM {table} ORDER BY ts").fetchdf()
    finally:
        con.close()
    if df.empty:
        meta.update({"available": True, "rows": 0, "columns": selected})
        return df, meta
    df["_ts_kst"] = pd.to_datetime(df["ts"], errors="coerce", utc=True).dt.tz_convert("Asia/Seoul")
    df = df.dropna(subset=["_ts_kst"]).sort_values("_ts_kst").reset_index(drop=True)
    rename = {c: f"{prefix}_{c}" for c in df.columns if c not in {"ts", "_ts_kst"}}
    df = df.rename(columns=rename)
    meta.update(
        {
            "available": True,
            "rows": int(len(df)),
            "columns": selected,
            "first_ts": _fmt_ts(df["_ts_kst"].iloc[0]),
            "last_ts": _fmt_ts(df["_ts_kst"].iloc[-1]),
        }
    )
    return df, meta


def _asof_lookup(source: pd.DataFrame, ts: pd.Timestamp, ts_col: str, tolerance: pd.Timedelta) -> dict[str, Any] | None:
    if source.empty or pd.isna(ts):
        return None
    times = source[ts_col]
    pos = times.searchsorted(ts, side="right") - 1
    if pos < 0:
        return None
    row_ts = times.iloc[pos]
    age = ts - row_ts
    if age < pd.Timedelta(0) or age > tolerance:
        return None
    out = dict(source.iloc[pos])
    out["_match_age_sec"] = float(age.total_seconds())
    return out


def _flatten_event(
    row: pd.Series,
    *,
    snapshots: pd.DataFrame,
    micro: pd.DataFrame,
    tail: pd.DataFrame,
    tolerance: pd.Timedelta,
) -> dict[str, Any]:
    decision_ts = row.get("_decision_bar_ts_kst")
    if pd.isna(decision_ts):
        decision_ts = row.get("_decision_at_kst")
    if pd.isna(decision_ts):
        decision_ts = row.get("_ts_kst")

    snap = _asof_lookup(snapshots, decision_ts, "_snapshot_ts_kst", tolerance) if not snapshots.empty else None
    micro_row = _asof_lookup(micro, decision_ts, "_ts_kst", tolerance) if not micro.empty else None
    tail_row = _asof_lookup(tail, decision_ts, "_ts_kst", tolerance) if not tail.empty else None

    trace = row.get("omega5_sizing_trace")
    if not isinstance(trace, dict):
        trace = {}
    parent_trace = trace.get("parent_trace") if isinstance(trace.get("parent_trace"), dict) else {}
    source = str(row.get("source", "") or "")
    kind = str(row.get("kind", "") or "")
    out: dict[str, Any] = {
        "kind": kind,
        "trade_id": str(row.get("trade_id", "") or ""),
        "side": str(row.get("side", "") or ""),
        "source": source,
        "reason": str(row.get("reason", "") or ""),
        "model_id": str(row.get("model_id", row.get("open_model_id", "")) or ""),
        "decision_ts": _fmt_ts(decision_ts),
        "event_ts": str(row.get("ts", "") or ""),
        "event_recorded_at": str(row.get("event_recorded_at", "") or ""),
        "entry_price": _safe_float(row.get("entry_price", 0.0)),
        "exit_price": _safe_float(row.get("exit_price", 0.0)),
        "entry_exec_price": _safe_float(row.get("entry_exec_price", 0.0)),
        "exit_exec_price": _safe_float(row.get("exit_exec_price", 0.0)),
        "pnl_pct": _safe_float(row.get("pnl_pct", 0.0)),
        "gross_return_frac": _safe_float(row.get("gross_return_frac", 0.0)),
        "fee_cost_frac": _safe_float(row.get("fee_cost_frac", 0.0)),
        "notional_exposure": _safe_float(row.get("notional_exposure", 0.0)),
        "margin_fraction": _safe_float(row.get("margin_fraction", row.get("position_fraction", 0.0))),
        "execution_leverage": _safe_float(row.get("execution_leverage", 0.0)),
        "take_profit": _safe_float(row.get("effective_take_profit", row.get("take_profit", 0.0))),
        "stop_loss": _safe_float(row.get("effective_stop_loss", row.get("stop_loss", 0.0))),
        "max_hold_bars": _safe_int(row.get("max_hold_bars", 0)),
        "exchange_execution_enabled": bool(row.get("exchange_execution_enabled", False)),
        "exchange_execution_dry_run": bool(row.get("exchange_execution_dry_run", False)),
        "exchange_order_count": _safe_int(row.get("exchange_order_count", 0)),
        "execution_delay_sec": _safe_float(row.get("execution_delay_sec", 0.0)),
        "execution_delay_late": bool(row.get("execution_delay_late", False)),
        "omega5_trace_present": bool(trace),
        "omega5_parent_trace_present": bool(parent_trace),
        "omega5_parent_ledger_replay_used": parent_trace.get("ledger_replay_used"),
        "omega5_source_parent_live_native_adapter": parent_trace.get("source_parent_live_native_adapter"),
        "omega5_source_parent_predictive_artifact": parent_trace.get("source_parent_predictive_artifact", ""),
        "omega5_reason_from_trace": trace.get("omega5_reason", ""),
        "omega5_parent_action": trace.get("parent_action"),
        "omega5_parent_side": trace.get("parent_side"),
        "omega5_parent_notional": trace.get("parent_notional_exposure"),
        "matched_decision_snapshot": bool(snap),
        "matched_micro_duckdb": bool(micro_row),
        "matched_tail_duckdb": bool(tail_row),
    }
    if snap:
        out["decision_snapshot_age_sec"] = _safe_float(snap.pop("_match_age_sec", 0.0))
        for key, val in snap.items():
            if key not in {"_snapshot_ts_kst"}:
                out[key] = val
    if micro_row:
        out["micro_age_sec"] = _safe_float(micro_row.pop("_match_age_sec", 0.0))
        for key, val in micro_row.items():
            if key not in {"ts", "_ts_kst"}:
                out[key] = val
    if tail_row:
        out["tail_age_sec"] = _safe_float(tail_row.pop("_match_age_sec", 0.0))
        for key, val in tail_row.items():
            if key not in {"ts", "_ts_kst"}:
                out[key] = val
    return out


def _summarize_events(events: pd.DataFrame) -> dict[str, Any]:
    if events.empty:
        return {"rows": 0}
    opens = events[events["kind"].astype(str).str.upper() == "OPEN"].copy()
    closes = events[events["kind"].astype(str).str.upper() == "CLOSE"].copy()
    active = opens
    if "trade_id" in opens.columns and "trade_id" in closes.columns:
        closed_ids = set(closes["trade_id"].astype(str))
        active = opens[~opens["trade_id"].astype(str).isin(closed_ids)]
    risk_missing = opens[
        (pd.to_numeric(opens.get("take_profit", 0), errors="coerce").fillna(0.0) <= 0.0)
        | (pd.to_numeric(opens.get("stop_loss", 0), errors="coerce").fillna(0.0) <= 0.0)
        | (pd.to_numeric(opens.get("max_hold_bars", 0), errors="coerce").fillna(0).astype(int) <= 0)
    ]
    trace_missing = opens[~opens.get("omega5_trace_present", pd.Series(False, index=opens.index)).astype(bool)]
    ledger_replay = opens[opens.get("omega5_parent_ledger_replay_used", pd.Series(False, index=opens.index)).fillna(False).astype(bool)]
    return {
        "rows": int(len(events)),
        "open_count": int(len(opens)),
        "close_count": int(len(closes)),
        "active_open_count": int(len(active)),
        "quarantine_count": int(events["source"].astype(str).str.contains("contract_quarantine", na=False).sum()),
        "reconcile_close_count": int(events["source"].astype(str).str.contains("reconcile_close", na=False).sum()),
        "late_execution_count": int(events.get("execution_delay_late", pd.Series(False, index=events.index)).fillna(False).astype(bool).sum()),
        "missing_risk_contract_open_count": int(len(risk_missing)),
        "missing_trace_open_count": int(len(trace_missing)),
        "parent_ledger_replay_open_count": int(len(ledger_replay)),
        "decision_snapshot_match_count": int(events.get("matched_decision_snapshot", pd.Series(False, index=events.index)).fillna(False).astype(bool).sum()),
        "micro_match_count": int(events.get("matched_micro_duckdb", pd.Series(False, index=events.index)).fillna(False).astype(bool).sum()),
        "tail_match_count": int(events.get("matched_tail_duckdb", pd.Series(False, index=events.index)).fillna(False).astype(bool).sum()),
        "pnl_pct_sum": float(pd.to_numeric(closes.get("pnl_pct", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum()),
        "pnl_pct_mean_close": float(pd.to_numeric(closes.get("pnl_pct", pd.Series(dtype=float)), errors="coerce").mean()) if len(closes) else 0.0,
        "notional_avg_open": float(pd.to_numeric(opens.get("notional_exposure", pd.Series(dtype=float)), errors="coerce").mean()) if len(opens) else 0.0,
        "leverage_max_open": float(pd.to_numeric(opens.get("execution_leverage", pd.Series(dtype=float)), errors="coerce").max()) if len(opens) else 0.0,
    }


def _build_markdown(report: dict[str, Any], recent: list[dict[str, Any]]) -> str:
    s = report["summary"]
    lines = [
        "# Omega5 Live DuckDB Forensics - 2026-07-02",
        "",
        f"- Status: `{report['status']}`",
        f"- Verdict: `{report['verdict']}`",
        f"- Omega5 events: `{s.get('rows', 0)}` open `{s.get('open_count', 0)}` close `{s.get('close_count', 0)}`",
        f"- Decision snapshot matches: `{s.get('decision_snapshot_match_count', 0)}`",
        f"- DuckDB matches: micro `{s.get('micro_match_count', 0)}`, tail `{s.get('tail_match_count', 0)}`",
        f"- Missing risk-contract opens: `{s.get('missing_risk_contract_open_count', 0)}`",
        f"- Missing trace opens: `{s.get('missing_trace_open_count', 0)}`",
        f"- Quarantine events: `{s.get('quarantine_count', 0)}`",
        f"- Reconcile close events: `{s.get('reconcile_close_count', 0)}`",
        "",
        "## Blocking",
    ]
    if report["blocking"]:
        lines.extend(f"- `{x}`" for x in report["blocking"])
    else:
        lines.append("- none")
    lines.append("")
    lines.append("## Warnings")
    if report["warnings"]:
        lines.extend(f"- `{x}`" for x in report["warnings"])
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            "## Data Sources",
            f"- Journal: `{report['inputs']['trade_journal']}`",
            f"- Decision snapshots: `{report['inputs']['decision_snapshot_jsonl']}`",
            f"- Micro DuckDB: `{report['duckdb']['micro'].get('read_mode', 'unavailable')}` rows `{report['tables']['micro'].get('rows', 0)}`",
            f"- Tail DuckDB: `{report['duckdb']['tail'].get('read_mode', 'unavailable')}` rows `{report['tables']['tail'].get('rows', 0)}`",
            "",
            "## Recent Omega5 Events",
            "",
            "| kind | decision_ts | source | side | notional | lev | pnl_pct | snapshot | micro | tail | reason |",
            "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- |",
        ]
    )
    for row in recent:
        lines.append(
            "| {kind} | {decision_ts} | {source} | {side} | {notional:.4f} | {lev:.2f} | {pnl:.4f} | {snap} | {micro} | {tail} | {reason} |".format(
                kind=str(row.get("kind", "")),
                decision_ts=str(row.get("decision_ts", "")),
                source=str(row.get("source", "")),
                side=str(row.get("side", "")),
                notional=_safe_float(row.get("notional_exposure", 0.0)),
                lev=_safe_float(row.get("execution_leverage", 0.0)),
                pnl=_safe_float(row.get("pnl_pct", 0.0)),
                snap=bool(row.get("matched_decision_snapshot", False)),
                micro=bool(row.get("matched_micro_duckdb", False)),
                tail=bool(row.get("matched_tail_duckdb", False)),
                reason=str(row.get("reason", "")),
            )
        )
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = Path(args.trade_journal)
    snapshot_path = Path(args.decision_snapshot_jsonl)
    tolerance = pd.Timedelta(minutes=float(args.nearest_minutes))

    journal = _load_omega5_journal(journal_path)
    snapshots = _load_decision_snapshots(snapshot_path)

    micro_db, micro_db_meta = _copy_or_connect_db(Path(args.micro_db), out_dir, float(args.max_copy_mb))
    tail_db, tail_db_meta = _copy_or_connect_db(Path(args.tail_db), out_dir, float(args.max_copy_mb))
    micro, micro_meta = _load_table(micro_db, "microstructure_1m", MICRO_COLS, "micro") if micro_db else (pd.DataFrame(), {"rows": 0, "available": False})
    tail, tail_meta = _load_table(tail_db, "tail_risk_1m", TAIL_COLS, "tail") if tail_db else (pd.DataFrame(), {"rows": 0, "available": False})

    events: list[dict[str, Any]] = []
    if not journal.empty:
        for _, row in journal.iterrows():
            events.append(_flatten_event(row, snapshots=snapshots, micro=micro, tail=tail, tolerance=tolerance))
    event_df = pd.DataFrame(events)
    if not event_df.empty:
        csv_df = event_df.copy()
        for col in csv_df.columns:
            if csv_df[col].map(lambda x: isinstance(x, (dict, list))).any():
                csv_df[col] = csv_df[col].map(lambda x: json.dumps(x, ensure_ascii=False, default=_json_default) if isinstance(x, (dict, list)) else x)
        csv_df.to_csv(EVENT_CSV, index=False)
    else:
        pd.DataFrame().to_csv(EVENT_CSV, index=False)

    summary = _summarize_events(event_df)
    blocking: list[str] = []
    warnings: list[str] = []
    if journal.empty:
        blocking.append("omega5_trade_journal_events_missing")
    if snapshots.empty:
        warnings.append("decision_feature_snapshots_missing_or_empty")
    if summary.get("open_count", 0) and summary.get("missing_trace_open_count", 0):
        warnings.append("some_omega5_open_events_missing_sizing_trace")
    if summary.get("missing_risk_contract_open_count", 0):
        warnings.append("some_omega5_open_events_missing_or_zero_risk_contract")
    if summary.get("parent_ledger_replay_open_count", 0):
        blocking.append("omega5_parent_ledger_replay_detected")
    if summary.get("reconcile_close_count", 0):
        warnings.append("omega5_reconcile_close_events_present")
    if not micro_meta.get("available"):
        warnings.append("micro_duckdb_unavailable")
    if not tail_meta.get("available"):
        warnings.append("tail_duckdb_unavailable")

    status = "pass" if not blocking else "fail"
    verdict = "OMEGA5_LIVE_DUCKDB_FORENSICS_READY" if status == "pass" else "OMEGA5_LIVE_DUCKDB_FORENSICS_BLOCKED"
    recent = event_df.tail(int(args.recent_events)).to_dict(orient="records") if not event_df.empty else []
    report = {
        "audit_id": "omega5_live_duckdb_forensics_20260702",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "verdict": verdict,
        "blocking": blocking,
        "warnings": warnings,
        "inputs": {
            "trade_journal": str(journal_path),
            "decision_snapshot_jsonl": str(snapshot_path),
            "out_dir": str(out_dir),
            "nearest_minutes": float(args.nearest_minutes),
        },
        "duckdb": {
            "micro": micro_db_meta,
            "tail": tail_db_meta,
        },
        "tables": {
            "micro": micro_meta,
            "tail": tail_meta,
        },
        "summary": summary,
        "artifacts": {
            "event_csv": str(EVENT_CSV),
            "report_json": str(REPORT_JSON),
            "report_md": str(REPORT_MD),
        },
        "recent_events": recent,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    REPORT_MD.write_text(_build_markdown(report, recent), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Join Omega5 live journal events with decision snapshots and DuckDB live features.")
    parser.add_argument("--trade-journal", type=Path, default=ROOT / "data/live/trade_journal.jsonl")
    parser.add_argument("--decision-snapshot-jsonl", type=Path, default=ROOT / "data/live/decision_feature_snapshot.jsonl")
    parser.add_argument("--micro-db", type=Path, default=ROOT / "data/live/microstructure.duckdb")
    parser.add_argument("--tail-db", type=Path, default=ROOT / "data/live/tail_risk.duckdb")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--nearest-minutes", type=float, default=15.0)
    parser.add_argument("--max-copy-mb", type=float, default=512.0)
    parser.add_argument("--recent-events", type=int, default=20)
    args = parser.parse_args()
    report = run(args)
    print(json.dumps({k: report[k] for k in ("status", "verdict", "summary", "blocking", "warnings", "artifacts")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
