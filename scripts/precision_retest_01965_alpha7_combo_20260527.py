#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ensemble.fully_learned_governor_policy import ACTION_CASH  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import backtest_signal_limit_exit_guard  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    RESULTS_CSV,
    _apply_decision_mods,
    _decision_sources,
    _default_limit_cfg,
    _guard,
    _load_frames,
    _load_stack,
    _overlay,
    _score,
    _sl_ratio,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "precision_retest_01965_alpha7_combo_20260527"
CANDIDATE = "01965_random_alpha7_combo_primary_fallback"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
PERIOD_GRID_OUT = OUT_DIR / "cost_period_grid.csv"
SOURCE_AUDIT_OUT = OUT_DIR / "source_decision_audit.json"
OOS_LEDGER_OUT = OUT_DIR / "oos_cost3_ledger.csv"
VAL_LEDGER_OUT = OUT_DIR / "val_cost3_ledger.csv"


def _active(dec: pd.DataFrame) -> pd.Series:
    return (pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != ACTION_CASH) & (
        pd.to_numeric(dec["side"], errors="coerce").fillna(0).astype(int) != 0
    )


def _cfg_from_results() -> dict[str, Any]:
    results = pd.read_csv(RESULTS_CSV)
    row = results.loc[results["name"].eq(CANDIDATE)]
    if row.empty:
        raise RuntimeError(f"candidate not found in {RESULTS_CSV}: {CANDIDATE}")
    r = row.iloc[0]
    cfg: dict[str, Any] = {"name": CANDIDATE, "source": str(r["source"]), "tag": str(r["tag"])}
    for col, value in r.items():
        if not str(col).startswith("cfg_"):
            continue
        key = str(col)[4:]
        if isinstance(value, str) and value in {"True", "False"}:
            cfg[key] = value == "True"
        elif isinstance(value, (np.bool_, bool)):
            cfg[key] = bool(value)
        elif pd.isna(value):
            cfg[key] = None
        else:
            cfg[key] = value.item() if hasattr(value, "item") else value
    return cfg


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    periods: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            periods.append((f"{prefix}_{month}", mask))
    return periods


def _source_audit(raw_sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]], cfg: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for split_name, side in (("val", 0), ("oos", 1)):
        primary = raw_sources["alpha7_primary"][side].reset_index(drop=True)
        combo = raw_sources["alpha7_combo_primary_fallback"][side].reset_index(drop=True)
        primary_active = _active(primary)
        combo_active = _active(combo)
        fallback_used = (~primary_active) & combo_active
        mod = _apply_decision_mods(combo, cfg)
        mod_active = _active(mod)
        long = (pd.to_numeric(mod["side"], errors="coerce").fillna(0).astype(int) > 0) & mod_active
        short = (pd.to_numeric(mod["side"], errors="coerce").fillna(0).astype(int) < 0) & mod_active
        out[split_name] = {
            "rows": int(len(combo)),
            "primary_active_rows": int(primary_active.sum()),
            "fallback_used_rows": int(fallback_used.sum()),
            "combo_active_rows_before_mods": int(combo_active.sum()),
            "combo_active_rows_after_mods": int(mod_active.sum()),
            "blocked_by_quality_or_conf": int((combo_active & ~mod_active).sum()),
            "after_mod_long_rows": int(long.sum()),
            "after_mod_short_rows": int(short.sum()),
            "avg_notional_after_mods": float(pd.to_numeric(mod.loc[mod_active, "notional_exposure"], errors="coerce").mean()) if mod_active.any() else 0.0,
            "avg_tp_after_mods": float(pd.to_numeric(mod.loc[mod_active, "take_profit"], errors="coerce").mean()) if mod_active.any() else 0.0,
            "avg_sl_after_mods": float(pd.to_numeric(mod.loc[mod_active, "stop_loss"], errors="coerce").mean()) if mod_active.any() else 0.0,
            "avg_hold_after_mods": float(pd.to_numeric(mod.loc[mod_active, "max_hold_bars"], errors="coerce").mean()) if mod_active.any() else 0.0,
        }
    return out


def _eval(
    *,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    period: str,
    cost_mult: int,
    record: bool = False,
) -> dict[str, Any]:
    res = backtest_signal_limit_exit_guard(
        df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        _apply_decision_mods(dec, cfg).reset_index(drop=True),
        _overlay(stack["overlay"], cfg),
        _default_limit_cfg(),
        _guard(cfg),
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=float(cost_mult),
        record=record,
    )
    row = {
        "candidate": CANDIDATE,
        "period": period,
        "cost": int(cost_mult),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "sl_ratio": float(_sl_ratio(res)),
        "score": float(_score(res)),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "avg_notional": float(res.get("avg_notional", 0.0)),
        "avg_leverage": float(res.get("avg_leverage", 0.0)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }
    if record:
        row["_records"] = res.get("trade_records", [])
    return row


def _ledger_stats(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path)
    if df.empty:
        return {"rows": 0}
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    winners = ret[ret > 0].sort_values(ascending=False)
    losers = ret[ret <= 0].sort_values()
    return {
        "rows": int(len(df)),
        "gross_trade_return_sum": float(ret.sum()),
        "gross_trade_return_mean": float(ret.mean()),
        "gross_trade_return_median": float(ret.median()),
        "top1_trade_return": float(winners.iloc[0]) if len(winners) else 0.0,
        "top5_trade_return_sum": float(winners.head(5).sum()) if len(winners) else 0.0,
        "top10_trade_return_sum": float(winners.head(10).sum()) if len(winners) else 0.0,
        "bottom5_trade_return_sum": float(losers.head(5).sum()) if len(losers) else 0.0,
        "win_count": int((ret > 0).sum()),
        "loss_count": int((ret <= 0).sum()),
        "final_cash_after": float(pd.to_numeric(df["cash_after"], errors="coerce").iloc[-1]),
        "max_hold_bars": int(pd.to_numeric(df["hold_bars"], errors="coerce").max()),
        "median_hold_bars": float(pd.to_numeric(df["hold_bars"], errors="coerce").median()),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    stack = _load_stack()
    val_df, eval_df = _load_frames()
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    audit = _source_audit(sources, cfg)
    SOURCE_AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")

    rows: list[dict[str, Any]] = []
    ledger_records: dict[str, list[dict[str, Any]]] = {}
    for split_name, df, q, dec_side in (
        ("val", val_df, val_q, 0),
        ("oos", eval_df, eval_q, 1),
    ):
        base_dec = sources[str(cfg["source"])][dec_side]
        for period_name, mask in _periods(df, split_name):
            sub_df = df.loc[mask].reset_index(drop=True)
            sub_q = q[mask]
            sub_dec = base_dec.loc[mask].reset_index(drop=True)
            for cost in (1, 2, 3):
                record = period_name in {"val_full", "oos_full"} and cost == 3
                row = _eval(df=sub_df, q=sub_q, dec=sub_dec, stack=stack, cfg=cfg, period=period_name, cost_mult=cost, record=record)
                if record:
                    ledger_records[period_name] = list(row.pop("_records", []))
                rows.append(row)

    grid = pd.DataFrame(rows)
    grid.to_csv(PERIOD_GRID_OUT, index=False)
    pd.DataFrame(ledger_records.get("oos_full", [])).to_csv(OOS_LEDGER_OUT, index=False)
    pd.DataFrame(ledger_records.get("val_full", [])).to_csv(VAL_LEDGER_OUT, index=False)

    cost3 = grid[grid["cost"].eq(3)].copy()
    full = cost3[cost3["period"].isin(["val_full", "oos_full"])]
    month = cost3[~cost3["period"].isin(["val_full", "oos_full"])]
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "candidate_type": "runtime_combo_not_new_model_artifact",
        "config": cfg,
        "source_decision_audit": str(SOURCE_AUDIT_OUT),
        "period_grid": str(PERIOD_GRID_OUT),
        "oos_cost3_ledger": str(OOS_LEDGER_OUT),
        "val_cost3_ledger": str(VAL_LEDGER_OUT),
        "cost3_full": full.to_dict(orient="records"),
        "cost3_monthly": month.to_dict(orient="records"),
        "oos_ledger_stats": _ledger_stats(OOS_LEDGER_OUT),
        "val_ledger_stats": _ledger_stats(VAL_LEDGER_OUT),
        "warning": "This is a precision retest on available 2025Q4 and 2026 Jan-Feb frames. It is not a post-loop untouched OOS test.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "period_grid": str(PERIOD_GRID_OUT), "oos_ledger": str(OOS_LEDGER_OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
