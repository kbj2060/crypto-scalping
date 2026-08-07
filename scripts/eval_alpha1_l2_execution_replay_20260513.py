#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import predict_policy_frame  # noqa: E402
from scripts import eval_alpha1_rl_exit_and_sizing_20260513 as alpha1  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import eval_hf_v13_v31_frozen_parent_layer_ablation_v45 as v45  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _audit_contract, _close, _fill_price, _read  # noqa: E402
from scripts.train_eval_hf_v13_deep_alpha_candidate_expansion_v27 import _json_default  # noqa: E402
from scripts.train_eval_hf_v13_jackpot_runner_v21_2 import CostRunnerConfig  # noqa: E402


MODEL_ID = "alpha1_l2_execution_replay_20260513"
OUT_DIR = ROOT / "data/ensemble/supervised/alpha1_l2_execution_replay_20260513"
REPORT_OUT = ROOT / "data/ensemble/reports/alpha1_l2_execution_replay_20260513_summary.json"
AUDIT_OUT = ROOT / "data/ensemble/reports/alpha1_l2_execution_replay_20260513_audit.json"
GRID_OUT = ROOT / "data/ensemble/reports/alpha1_l2_execution_replay_20260513_grid.csv"
LIVE_L2_PATH = ROOT / "data/live/orderbook_snapshots.jsonl"
LIVE_L2_DUCKDB = ROOT / "data/live/microstructure.duckdb"
LIVE_L2_TABLE = "orderbook_decision_snapshots"


def _safe(row: pd.Series, col: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(col, default))
    except Exception:
        return float(default)
    return value if np.isfinite(value) else float(default)


def _score(c1: dict[str, Any], c2: dict[str, Any], c3: dict[str, Any]) -> float:
    if int(c1.get("trades", 0)) < 20:
        return -1e9 + float(c1.get("pnl", 0.0))
    return float(c1["pnl"] + 0.45 * c2["pnl"] + 0.30 * c3["pnl"] - 0.30 * abs(c1["mdd"]))


def _live_l2_stats(path: Path = LIVE_L2_PATH, db_path: Path = LIVE_L2_DUCKDB, table: str = LIVE_L2_TABLE) -> dict[str, Any]:
    if db_path.exists():
        try:
            import duckdb

            con = duckdb.connect(str(db_path), read_only=True)
            tables = {str(r[0]) for r in con.execute("SHOW TABLES").fetchall()}
            if table in tables:
                row = con.execute(
                    f"SELECT count(*) AS n, min(timestamp_kst) AS first_ts, max(timestamp_kst) AS last_ts FROM {table}"
                ).fetchone()
                con.close()
                rows = int(row[0] or 0)
                return {
                    "storage": "duckdb",
                    "db_path": str(db_path),
                    "table": str(table),
                    "exists": True,
                    "rows": rows,
                    "first_timestamp_kst": str(row[1] or ""),
                    "last_timestamp_kst": str(row[2] or ""),
                    "usable_for_replay": bool(rows >= 500),
                }
            con.close()
        except Exception as exc:
            return {
                "storage": "duckdb",
                "db_path": str(db_path),
                "table": str(table),
                "exists": True,
                "rows": 0,
                "usable_for_replay": False,
                "error": str(exc),
            }
    if not path.exists():
        return {
            "storage": "duckdb",
            "db_path": str(db_path),
            "table": str(table),
            "jsonl_path": str(path),
            "exists": False,
            "rows": 0,
            "usable_for_replay": False,
        }
    rows = 0
    first = ""
    last = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows += 1
            if rows == 1:
                try:
                    first = str(json.loads(line).get("timestamp_kst", ""))
                except Exception:
                    first = ""
            try:
                last = str(json.loads(line).get("timestamp_kst", ""))
            except Exception:
                pass
    return {
        "storage": "jsonl",
        "path": str(path),
        "exists": True,
        "rows": int(rows),
        "first_timestamp_kst": first,
        "last_timestamp_kst": last,
        "usable_for_replay": bool(rows >= 500),
    }


def _spread_proxy_bps(row: pd.Series) -> float:
    # Conservative synthetic spread until enough live L2 snapshots are available.
    liq_vac = abs(_safe(row, "liquidity_vacuum", 0.0))
    exec_q = _safe(row, "execution_quality", 0.0)
    amihud = abs(_safe(row, "amihud_illiquidity_z", 0.0))
    return float(np.clip(2.0 + 1.75 * liq_vac + 0.75 * amihud - 0.50 * exec_q, 2.0, 12.0))


def _maker_fill_price(row: pd.Series, side: int, *, entry: bool, offset_bps: float, penetration_bps: float) -> tuple[bool, float]:
    open_px = _safe(row, "open", _safe(row, "close", 0.0))
    high = _safe(row, "high", open_px)
    low = _safe(row, "low", open_px)
    if open_px <= 0.0:
        return False, 0.0
    # Buying happens on long entries and short exits; selling is the opposite.
    is_buy = (side > 0 and entry) or (side < 0 and not entry)
    if is_buy:
        px = open_px * (1.0 - offset_bps / 10000.0)
        filled = bool(low <= px * (1.0 - penetration_bps / 10000.0))
    else:
        px = open_px * (1.0 + offset_bps / 10000.0)
        filled = bool(high >= px * (1.0 + penetration_bps / 10000.0))
    return filled, float(px)


def _fallback_close_price(df: pd.DataFrame, idx: int, side: int, slip: float, *, entry: bool) -> float:
    px = float(pd.to_numeric(df["close"], errors="coerce").ffill().iloc[int(np.clip(idx, 0, len(df) - 1))])
    if side > 0:
        return px * (1.0 + slip if entry else 1.0 - slip)
    return px * (1.0 - slip if entry else 1.0 + slip)


def _run_with_l2_proxy(df, bundle, jackpot_model, add_cfg, q, dec, variant, fee: float, slip: float, *, cost_mult: float, record: bool = False) -> dict[str, Any]:
    orig_route_cost = v45._route_cost
    orig_fill_with_route = v45._fill_with_route

    def route_cost(row: pd.Series, side: int, fee_in: float, slip_in: float, var: v45.LayerVariant) -> tuple[float, float, str]:
        if str(var.layer) != "conservative_l2_replay":
            return orig_route_cost(row, side, fee_in, slip_in, var)
        return float(fee_in), float(slip_in), "l2_replay_mark"

    def fill_with_route(df_in: pd.DataFrame, idx: int, side: int, fee_in: float, slip_in: float, var: v45.LayerVariant, *, entry: bool) -> tuple[float, float, float, str]:
        idx = int(np.clip(idx, 0, len(df_in) - 1))
        if str(var.layer) != "conservative_l2_replay":
            return orig_fill_with_route(df_in, idx, side, fee_in, slip_in, var, entry=entry)
        row = df_in.iloc[idx]
        offset_bps = _spread_proxy_bps(row)
        filled, maker_px = _maker_fill_price(row, side, entry=entry, offset_bps=offset_bps, penetration_bps=1.5)
        if filled:
            return float(maker_px), float(fee_in * var.sniper_fee_mult), 0.0, "conservative_maker_replay"
        fallback_px = _fallback_close_price(df_in, idx, side, slip_in, entry=entry)
        return float(fallback_px), float(fee_in), float(slip_in), "l2_replay_taker_fallback_close"

    try:
        v45._route_cost = route_cost
        v45._fill_with_route = fill_with_route
        return v45.backtest_variant(
            df,
            bundle,
            jackpot_model,
            add_cfg,
            q,
            variant,
            fee=fee,
            slip=slip,
            cost_mult=float(cost_mult),
            decisions=dec,
            record=record,
        )
    finally:
        v45._route_cost = orig_route_cost
        v45._fill_with_route = orig_fill_with_route


def _run_all(df, bundle, jackpot_model, add_cfg, q, dec, variant, fee: float, slip: float) -> dict[str, Any]:
    return {
        f"cost{mult}": _run_with_l2_proxy(df, bundle, jackpot_model, add_cfg, q, dec, variant, fee, slip, cost_mult=float(mult))
        for mult in (1, 2, 3)
    }


def _variants() -> list[v45.LayerVariant]:
    base = alpha1.ALPHA1_CFG
    return [
        v45.LayerVariant("alpha1_taker_baseline", "baseline", base),
        v45.LayerVariant("alpha1_l2_conservative_fee50", "conservative_l2_replay", replace(base, name="alpha1_l2_conservative_fee50"), execution_sniper=True, sniper_fee_mult=0.50, sniper_slip_mult=0.0),
        v45.LayerVariant("alpha1_l2_conservative_fee35", "conservative_l2_replay", replace(base, name="alpha1_l2_conservative_fee35"), execution_sniper=True, sniper_fee_mult=0.35, sniper_slip_mult=0.0),
        v45.LayerVariant("alpha1_l2_conservative_fee20", "conservative_l2_replay", replace(base, name="alpha1_l2_conservative_fee20"), execution_sniper=True, sniper_fee_mult=0.20, sniper_slip_mult=0.0),
    ]


def main() -> int:
    print(f"[{MODEL_ID}] loading stack", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUT.parent.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(v31.DEFAULT_PARENT)
    jackpot_payload = joblib.load(v31.DEFAULT_JACKPOT)
    jackpot_model = jackpot_payload["cost_runner"]
    add_cfg = CostRunnerConfig(**dict(jackpot_payload["selected_config"]))
    v27_payload, v27_model = v31._load_v27(v31.DEFAULT_V27)
    base_cfg = dict(bundle["config"])
    fee = float(base_cfg["fee"])
    slip = float(base_cfg["slip"])
    train_all = _read(v31.DEFAULT_TRAIN)
    eval_df = _read(v31.DEFAULT_EVAL)
    val = train_all[train_all["timestamp"] >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    audit_contract = _audit_contract(train_all, eval_df, list(bundle.get("feature_cols") or []))
    l2_stats = _live_l2_stats()

    print(f"[{MODEL_ID}] predicting parent and V27", flush=True)
    val_q = v31._predict_all(v27_model, val, v27_payload["seq_cols"], v27_payload["norm"])
    eval_q = v31._predict_all(v27_model, eval_df, v27_payload["seq_cols"], v27_payload["norm"])
    val_dec = predict_policy_frame(bundle, val, close=_close(val))
    eval_dec = predict_policy_frame(bundle, eval_df, close=_close(eval_df))

    selection_rows: list[dict[str, Any]] = []
    best_variant = _variants()[0]
    best_score = -1e18
    for variant in _variants():
        vm = _run_all(val, bundle, jackpot_model, add_cfg, val_q, val_dec, variant, fee, slip)
        score = _score(vm["cost1"], vm["cost2"], vm["cost3"])
        selection_rows.append({
            "name": variant.name,
            "layer": variant.layer,
            "selection_score": score,
            "val_cost1_pnl": vm["cost1"]["pnl"],
            "val_cost1_mdd": vm["cost1"]["mdd"],
            "val_cost2_pnl": vm["cost2"]["pnl"],
            "val_cost3_pnl": vm["cost3"]["pnl"],
            "maker_fee_mult": variant.sniper_fee_mult,
        })
        if score > best_score:
            best_score = score
            best_variant = variant
    pd.DataFrame(selection_rows).sort_values("selection_score", ascending=False).to_csv(GRID_OUT, index=False)
    print(f"[{MODEL_ID}] selected {best_variant.name}", flush=True)

    experiments: list[dict[str, Any]] = []
    for variant in (_variants()[0], best_variant):
        metrics = _run_all(eval_df, bundle, jackpot_model, add_cfg, eval_q, eval_dec, variant, fee, slip)
        experiments.append({
            "name": "alpha1_taker_baseline" if variant.name == "alpha1_taker_baseline" else f"alpha1_l2_replay::{variant.name}",
            "variant": asdict(variant),
            "metrics": metrics,
            "score": _score(metrics["cost1"], metrics["cost2"], metrics["cost3"]),
        })
        print(
            f"[{MODEL_ID}] {variant.name} cost1={metrics['cost1']['pnl']:.2f} mdd={metrics['cost1']['mdd']:.2f} cost2={metrics['cost2']['pnl']:.2f} cost3={metrics['cost3']['pnl']:.2f}",
            flush=True,
        )

    best = max(experiments, key=lambda x: x["score"])
    baseline = experiments[0]
    blocking = list(audit_contract.get("blocking", []))
    warnings = list(audit_contract.get("warnings", []))
    if not l2_stats.get("usable_for_replay", False):
        warnings.append("historical_l2_snapshots_insufficient_conservative_ohlc_replay_only")
    warnings.append("real_live_l2_fill_model_requires_forward_shadow_collection")
    warnings.append("l2_replay_taker_fallback_uses_same_bar_close_after_high_low_touch_window")
    audit = {
        "status": "pass" if not blocking else "fail",
        "verdict": "shadow_collect_l2" if not blocking else "fail",
        "blocking": blocking,
        "warnings": warnings,
        "selection_uses_2026": False,
        "selection_window": "2025-10-01..2025-12-31",
        "oos_window": "2026 fixed OOS only after selection",
        "live_l2_stats": l2_stats,
        "feature_audit": audit_contract,
        "red_team_note": "This resolves live data collection for alpha1.4 execution warnings. It does not claim historical L2 validation until enough orderbook_snapshots.jsonl rows exist.",
    }
    manifest = {
        "model_id": MODEL_ID,
        "selected_variant": asdict(best_variant),
        "parent_frozen": True,
        "v27_frozen": True,
        "v21_2_frozen": True,
        "execution_validation": "conservative_ohlc_replay_until_live_l2_history_available",
        "live_l2_path": str(LIVE_L2_PATH),
        "live_l2_duckdb": str(LIVE_L2_DUCKDB),
        "live_l2_table": str(LIVE_L2_TABLE),
    }
    manifest_path = OUT_DIR / "l2_execution_replay_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    report = {
        "model_id": MODEL_ID,
        "design": "Alpha1 execution-warning remediation harness. Live bot records L2 snapshots; this script replays alpha1 with a conservative maker-fill proxy until enough live L2 history exists for true L2 fill validation. Maker miss fallback now uses the same bar close after the high/low touch window, not that bar open.",
        "baseline_delta": {
            "cost1_pnl_delta": float(best["metrics"]["cost1"]["pnl"] - baseline["metrics"]["cost1"]["pnl"]),
            "cost2_pnl_delta": float(best["metrics"]["cost2"]["pnl"] - baseline["metrics"]["cost2"]["pnl"]),
            "cost3_pnl_delta": float(best["metrics"]["cost3"]["pnl"] - baseline["metrics"]["cost3"]["pnl"]),
        },
        "experiments": experiments,
        "audit": audit,
        "artifacts": {"manifest": str(manifest_path), "report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "grid": str(GRID_OUT)},
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    AUDIT_OUT.write_text(json.dumps(audit, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")
    print(json.dumps({"report": str(REPORT_OUT), "audit": str(AUDIT_OUT), "best": best["name"], "verdict": audit["verdict"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
