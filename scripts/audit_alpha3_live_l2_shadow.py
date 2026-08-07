#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / "data/live/microstructure.duckdb"
DEFAULT_JOURNAL = ROOT / "data/live/trade_journal.jsonl"
DEFAULT_REPORT = ROOT / "data/ensemble/reports/alpha3_live_l2_shadow_audit_latest.json"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _load_orderbook(db_path: Path, table: str) -> pd.DataFrame:
    if not db_path.exists():
        return pd.DataFrame()
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        tables = {str(r[0]) for r in con.execute("SHOW TABLES").fetchall()}
        if table not in tables:
            return pd.DataFrame()
        df = con.execute(f"SELECT * FROM {table} ORDER BY recorded_at_kst").fetchdf()
    finally:
        con.close()
    if df.empty:
        return df
    contexts: list[dict[str, Any]] = []
    for raw in df.get("context_json", pd.Series([""] * len(df))).fillna("").astype(str):
        try:
            contexts.append(json.loads(raw) if raw else {})
        except Exception:
            contexts.append({})
    for key in (
        "record_reason",
        "live_bar_contract",
        "decision_bar_ts",
        "execution_bar_ts",
        "decision_price",
        "execution_price",
        "final_action",
        "target_exposure",
        "target_exec_leverage",
        "source",
        "model_version",
        "position_reason",
    ):
        df[f"ctx_{key}"] = [c.get(key) for c in contexts]
    return df


def _load_journal(path: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return pd.DataFrame()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return pd.DataFrame(rows)


def _order_side(kind: str, pos_side: str) -> str:
    kind_u = str(kind or "").upper()
    side_u = str(pos_side or "").upper()
    if kind_u == "OPEN":
        return "buy" if side_u == "LONG" else "sell"
    if kind_u == "CLOSE":
        return "sell" if side_u == "LONG" else "buy"
    return ""


def _post_only_eval(order_side: str, limit_px: float, best_bid: float, best_ask: float) -> dict[str, Any]:
    if limit_px <= 0.0 or best_bid <= 0.0 or best_ask <= 0.0 or best_bid >= best_ask:
        return {"post_only_compatible": False, "risk": "bad_book", "distance_to_touch_bps": None}
    if order_side == "buy":
        if limit_px >= best_ask:
            return {
                "post_only_compatible": False,
                "risk": "would_cross_ask_or_post_only_reject",
                "distance_to_touch_bps": (best_ask - limit_px) / limit_px * 10000.0,
            }
        return {
            "post_only_compatible": True,
            "risk": "maker_resting",
            "distance_to_touch_bps": (best_ask - limit_px) / limit_px * 10000.0,
        }
    if order_side == "sell":
        if limit_px <= best_bid:
            return {
                "post_only_compatible": False,
                "risk": "would_cross_bid_or_post_only_reject",
                "distance_to_touch_bps": (limit_px - best_bid) / limit_px * 10000.0,
            }
        return {
            "post_only_compatible": True,
            "risk": "maker_resting",
            "distance_to_touch_bps": (limit_px - best_bid) / limit_px * 10000.0,
        }
    return {"post_only_compatible": False, "risk": "unknown_side", "distance_to_touch_bps": None}


def _nearest_snapshot(snaps: pd.DataFrame, decision_bar_ts: str, execution_bar_ts: str) -> dict[str, Any] | None:
    if snaps.empty:
        return None
    m = snaps[
        (snaps["ctx_decision_bar_ts"].astype(str) == str(decision_bar_ts))
        & (snaps["ctx_execution_bar_ts"].astype(str) == str(execution_bar_ts))
    ]
    if m.empty:
        m = snaps[snaps["ctx_execution_bar_ts"].astype(str) == str(execution_bar_ts)]
    if m.empty:
        return None
    return dict(m.iloc[-1])


def audit(db_path: Path, journal_path: Path, table: str) -> dict[str, Any]:
    ob = _load_orderbook(db_path, table)
    journal = _load_journal(journal_path)
    blocking: list[str] = []
    warnings: list[str] = []
    if ob.empty:
        blocking.append("orderbook_decision_snapshots_missing_or_empty")
    if journal.empty:
        blocking.append("trade_journal_missing_or_empty")

    context_counts: dict[str, int] = {}
    action_counts: dict[str, int] = {}
    if not ob.empty:
        context_counts = ob["ctx_model_version"].fillna("").astype(str).value_counts().to_dict()
        action_counts = ob["ctx_final_action"].fillna("").astype(str).value_counts().to_dict()
        if len(ob) < 500:
            warnings.append(f"insufficient_l2_snapshots_for_statistical_queue_validation:{len(ob)}<500")
        if int((pd.to_numeric(ob["spread_bps"], errors="coerce") > 4.0).sum()) > 0:
            warnings.append("wide_spread_snapshots_present")

    trade_rows: list[dict[str, Any]] = []
    if not journal.empty and not ob.empty:
        first_ob_ts = pd.to_datetime(ob["recorded_at_kst"], errors="coerce").min()
        j = journal[journal.get("kind", "").isin(["OPEN", "CLOSE"])].copy()
        if "event_recorded_at" in j.columns:
            j["_event_recorded_at"] = pd.to_datetime(j["event_recorded_at"], errors="coerce", utc=True)
            if pd.notna(first_ob_ts):
                first_cmp = pd.Timestamp(first_ob_ts)
                if first_cmp.tzinfo is None:
                    first_cmp = first_cmp.tz_localize("Asia/Seoul").tz_convert("UTC")
                else:
                    first_cmp = first_cmp.tz_convert("UTC")
                j = j[j["_event_recorded_at"] >= first_cmp]
        for _, row in j.iterrows():
            snap = _nearest_snapshot(
                ob,
                str(row.get("decision_bar_ts", "")),
                str(row.get("execution_bar_ts", "")),
            )
            order_side = _order_side(str(row.get("kind", "")), str(row.get("side", "")))
            limit_px = _safe_float(row.get("execution_price", row.get("entry_price", row.get("exit_price", 0.0))), 0.0)
            rec = {
                "kind": str(row.get("kind", "")),
                "side": str(row.get("side", "")),
                "trade_id": str(row.get("trade_id", "")),
                "decision_bar_ts": str(row.get("decision_bar_ts", "")),
                "execution_bar_ts": str(row.get("execution_bar_ts", "")),
                "event_recorded_at": str(row.get("event_recorded_at", "")),
                "order_side": order_side,
                "limit_price_touch0": float(limit_px),
                "source": str(row.get("source", "")),
                "model_version": str(row.get("model_version", "")),
                "exchange_execution_enabled": bool(row.get("exchange_execution_enabled", False)),
                "exchange_order_count": int(_safe_float(row.get("exchange_order_count", 0), 0.0)),
                "matched_l2_snapshot": bool(snap),
            }
            if snap:
                best_bid = _safe_float(snap.get("best_bid", 0.0), 0.0)
                best_ask = _safe_float(snap.get("best_ask", 0.0), 0.0)
                eval_out = _post_only_eval(order_side, limit_px, best_bid, best_ask)
                top_notional = _safe_float(snap.get("bid_notional_1" if order_side == "buy" else "ask_notional_1", 0.0), 0.0)
                depth20 = _safe_float(snap.get("bid_notional_20" if order_side == "buy" else "ask_notional_20", 0.0), 0.0)
                rec.update(
                    {
                        "recorded_at_kst": str(snap.get("recorded_at_kst", "")),
                        "best_bid": best_bid,
                        "best_ask": best_ask,
                        "mid": _safe_float(snap.get("mid", 0.0), 0.0),
                        "spread_bps": _safe_float(snap.get("spread_bps", 0.0), 0.0),
                        "imbalance_5": _safe_float(snap.get("imbalance_5", 0.0), 0.0),
                        "microprice_edge_bps": _safe_float(snap.get("microprice_edge_bps", 0.0), 0.0),
                        "top_notional_same_side": top_notional,
                        "depth20_notional_same_side": depth20,
                        **eval_out,
                    }
                )
            trade_rows.append(rec)

    matched = [r for r in trade_rows if r.get("matched_l2_snapshot")]
    compatible = [r for r in matched if r.get("post_only_compatible")]
    reject = [r for r in matched if not r.get("post_only_compatible")]
    actual_orders = [r for r in trade_rows if int(r.get("exchange_order_count", 0)) > 0]
    if not actual_orders:
        warnings.append("no_real_or_dry_run_exchange_orders_in_trade_journal_only_shadow_book_snapshots")
    if reject:
        warnings.append(f"post_only_cross_or_reject_risk_events:{len(reject)}")

    report = {
        "audit_id": "alpha3_live_l2_shadow_audit",
        "status": "pass" if not blocking else "fail",
        "verdict": "insufficient_live_l2_for_promotion" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "contract_under_test": {
            "model": "Alpha3 corrected selected",
            "execution": "next_open_limit_touch0_fee20",
            "entry_offset_bps": 0.0,
            "exit_offset_bps": 0.0,
            "penetration_bps": 0.0,
            "entry_miss": "skip",
            "exit_miss": "market_fallback_close",
        },
        "l2_snapshot_summary": {
            "rows": int(len(ob)),
            "first_recorded_at_kst": str(ob["recorded_at_kst"].iloc[0]) if not ob.empty else "",
            "last_recorded_at_kst": str(ob["recorded_at_kst"].iloc[-1]) if not ob.empty else "",
            "model_version_counts": context_counts,
            "final_action_counts": action_counts,
            "avg_spread_bps": float(pd.to_numeric(ob.get("spread_bps", pd.Series(dtype=float)), errors="coerce").mean()) if not ob.empty else 0.0,
            "p95_spread_bps": float(pd.to_numeric(ob.get("spread_bps", pd.Series(dtype=float)), errors="coerce").quantile(0.95)) if not ob.empty else 0.0,
        },
        "trade_shadow_summary": {
            "journal_events_after_l2_start": int(len(trade_rows)),
            "matched_events": int(len(matched)),
            "post_only_compatible_events": int(len(compatible)),
            "post_only_reject_risk_events": int(len(reject)),
            "actual_exchange_order_events": int(len(actual_orders)),
            "compatible_ratio_on_matched": float(len(compatible) / max(len(matched), 1)),
        },
        "matched_trade_events": trade_rows[-50:],
        "red_team_note": (
            "This audit checks post-only compatibility and book quality from live L2 summaries. "
            "It cannot prove queue fill because raw book levels, order acknowledgements, partial fills, "
            "and cancel/fallback events are not present while Binance execution is disabled."
        ),
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit Alpha3 corrected limit execution against live L2 snapshots.")
    ap.add_argument("--db", type=Path, default=DEFAULT_DB)
    ap.add_argument("--journal", type=Path, default=DEFAULT_JOURNAL)
    ap.add_argument("--table", default="orderbook_decision_snapshots")
    ap.add_argument("--out", type=Path, default=DEFAULT_REPORT)
    args = ap.parse_args()
    report = audit(args.db, args.journal, args.table)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": report["status"], "verdict": report["verdict"]}, ensure_ascii=False))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
