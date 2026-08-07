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

from scripts import precision_retest_alpha7_parent_soft_regime_veto_20260528 as parent_soft  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_alpha7_parent_soft_meta_router_20260528 as meta_router  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts import walk_forward_alpha7_parent_soft_regime_veto_20260528 as walk  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_prev_block_ratio_router_precision_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
MONTHLY_OUT = OUT_DIR / "monthly_cost3.csv"
LEDGER_DIFF_OUT = OUT_DIR / "ledger_diff_summary.csv"
VAL_BASE_LEDGER = OUT_DIR / "val_baseline_cost3_ledger.csv"
VAL_ROUTE_LEDGER = OUT_DIR / "val_prev_block_ratio_cost3_ledger.csv"
OOS_BASE_LEDGER = OUT_DIR / "oos_baseline_cost3_ledger.csv"
OOS_ROUTE_LEDGER = OUT_DIR / "oos_prev_block_ratio_cost3_ledger.csv"

THRESHOLD = 0.02
VAL_START = pd.Timestamp("2025-10-01 00:00:00")


def _to_day(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["timestamp"], errors="coerce").dt.to_period("D").astype(str)


def _combine_decisions(df: pd.DataFrame, base_dec: pd.DataFrame, soft_dec: pd.DataFrame, route_days: set[str]) -> pd.DataFrame:
    out = base_dec.copy().reset_index(drop=True)
    soft = soft_dec.reset_index(drop=True)
    days = _to_day(df.reset_index(drop=True))
    for day in sorted(route_days):
        mask = days.eq(day).to_numpy(dtype=bool)
        if int(mask.sum()):
            out.loc[mask, out.columns] = soft.loc[mask, out.columns].to_numpy()
    return out


def _periods(df: pd.DataFrame, split: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    rows = [(f"{split}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            rows.append((f"{split}_{month}", mask))
    return rows


def _eval(
    *,
    name: str,
    split: str,
    period: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
    record: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    res = sweep._backtest_variant(
        df=df.reset_index(drop=True),
        q=q,
        dec=dec.reset_index(drop=True),
        stack=stack,
        cfg=cfg,
        variant=variant,
        cost_mult=3,
        record=record,
        deep_gate=None,
    )
    records = list(res.pop("trade_records", [])) if record else []
    return parent_soft._row(name, split, period, res), records


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    return parent_soft._ledger_stats(records)


def _ledger_diff(split: str, base_records: list[dict[str, Any]], route_records: list[dict[str, Any]]) -> dict[str, Any]:
    base = parent_soft._trade_key_frame(base_records, "base")
    route = parent_soft._trade_key_frame(route_records, "route")
    merged = base.merge(route, on="trade_key", how="outer", indicator=True)
    base_ret = pd.to_numeric(merged.get("base_trade_return"), errors="coerce").fillna(0.0)
    route_ret = pd.to_numeric(merged.get("route_trade_return"), errors="coerce").fillna(0.0)
    common = merged["_merge"].eq("both")
    base_only = merged["_merge"].eq("left_only")
    route_only = merged["_merge"].eq("right_only")
    detail_path = OUT_DIR / f"{split}_ledger_trade_key_diff.csv"
    merged.to_csv(detail_path, index=False)
    return {
        "split": split,
        "common_trades": int(common.sum()),
        "baseline_only_trades": int(base_only.sum()),
        "router_only_trades": int(route_only.sum()),
        "common_return_delta_sum": float((route_ret[common] - base_ret[common]).sum()),
        "baseline_only_return_sum": float(base_ret[base_only].sum()),
        "router_only_return_sum": float(route_ret[route_only].sum()),
        "gross_return_delta_sum": float(route_ret.sum() - base_ret.sum()),
        "detail_path": str(detail_path),
    }


def _slice_from(df: pd.DataFrame, q: np.ndarray, *frames: pd.DataFrame, start: pd.Timestamp) -> tuple[pd.DataFrame, np.ndarray, list[pd.DataFrame]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    mask = ts.ge(start).to_numpy(dtype=bool)
    return df.loc[mask].reset_index(drop=True), q[mask], [f.loc[mask].reset_index(drop=True) for f in frames]


def _route_days_from_daily(daily: pd.DataFrame, split: str, start: pd.Timestamp | None = None) -> set[str]:
    frame = daily[daily["split"].eq(split)].copy()
    if start is not None:
        frame = frame[pd.to_datetime(frame["day_ts"]) >= start]
    return set(frame.loc[pd.to_numeric(frame["prev_blocked_active_ratio"], errors="coerce").fillna(0.0) >= THRESHOLD, "day"].astype(str))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = walk.precision._cfg_from_results()
    stack = walk.precision._load_stack()
    train_df = walk._load_full_train_frame()
    _, eval_df = walk.precision._load_frames()
    sources = walk.precision._decision_sources(train_df, eval_df, stack["parent"])
    train_dec = sources[str(cfg["source"])][0].reset_index(drop=True)
    eval_dec = sources[str(cfg["source"])][1].reset_index(drop=True)
    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)

    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    train_soft_dec, _, train_block_summary = parent_soft._parent_soft_veto(train_df, train_dec, split="train2025")
    eval_soft_dec, _, eval_block_summary = parent_soft._parent_soft_veto(eval_df, eval_dec, split="oos2026")

    daily_path = meta_router.DAILY_OUT
    if not daily_path.exists():
        raise FileNotFoundError(f"missing daily meta-router frame: {daily_path}")
    daily = pd.read_csv(daily_path)
    val_route_days = _route_days_from_daily(daily, "train2025", VAL_START)
    oos_route_days = _route_days_from_daily(daily, "oos2026", None)

    train_route_dec = _combine_decisions(train_df, train_dec, train_soft_dec, val_route_days)
    eval_route_dec = _combine_decisions(eval_df, eval_dec, eval_soft_dec, oos_route_days)

    val_df, val_q, [val_dec, val_soft_dec, val_route_dec] = _slice_from(
        train_df,
        train_q,
        train_dec,
        train_soft_dec,
        train_route_dec,
        start=VAL_START,
    )

    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for split, df, q, base_dec, soft_dec, route_dec in [
        ("val", val_df, val_q, val_dec, val_soft_dec, val_route_dec),
        ("oos", eval_df, eval_q, eval_dec, eval_soft_dec, eval_route_dec),
    ]:
        for variant_name, dec in [
            ("baseline", base_dec),
            ("parent_soft_static", soft_dec),
            (f"prev_block_ratio_t{THRESHOLD:.2f}", route_dec),
        ]:
            row, records = _eval(
                name=variant_name,
                split=split,
                period=f"{split}_full",
                df=df,
                q=q,
                dec=dec,
                stack=stack,
                cfg=cfg,
                variant=variant,
                record=variant_name in {"baseline", f"prev_block_ratio_t{THRESHOLD:.2f}"},
            )
            rows.append(row)
            if records:
                ledgers[f"{split}_{variant_name}"] = records
            for period, mask in _periods(df, split):
                if period == f"{split}_full":
                    continue
                period_row, _ = _eval(
                    name=variant_name,
                    split=split,
                    period=period,
                    df=df.loc[mask],
                    q=q[mask],
                    dec=dec.loc[mask],
                    stack=stack,
                    cfg=cfg,
                    variant=variant,
                    record=False,
                )
                monthly_rows.append(period_row)

    pd.DataFrame(rows).to_csv(GRID_OUT, index=False)
    pd.DataFrame(monthly_rows).to_csv(MONTHLY_OUT, index=False)

    pd.DataFrame(ledgers.get("val_baseline", [])).to_csv(VAL_BASE_LEDGER, index=False)
    pd.DataFrame(ledgers.get(f"val_prev_block_ratio_t{THRESHOLD:.2f}", [])).to_csv(VAL_ROUTE_LEDGER, index=False)
    pd.DataFrame(ledgers.get("oos_baseline", [])).to_csv(OOS_BASE_LEDGER, index=False)
    pd.DataFrame(ledgers.get(f"oos_prev_block_ratio_t{THRESHOLD:.2f}", [])).to_csv(OOS_ROUTE_LEDGER, index=False)
    ledger_diff = [
        _ledger_diff("val", ledgers.get("val_baseline", []), ledgers.get(f"val_prev_block_ratio_t{THRESHOLD:.2f}", [])),
        _ledger_diff("oos", ledgers.get("oos_baseline", []), ledgers.get(f"oos_prev_block_ratio_t{THRESHOLD:.2f}", [])),
    ]
    pd.DataFrame(ledger_diff).to_csv(LEDGER_DIFF_OUT, index=False)

    summary = {
        "model_id": MODEL_ID,
        "scope": "Precision retest for previous-day blocked-active-ratio router. Research/shadow only.",
        "threshold": THRESHOLD,
        "artifacts": {
            "grid": str(GRID_OUT),
            "monthly": str(MONTHLY_OUT),
            "ledger_diff": str(LEDGER_DIFF_OUT),
            "val_baseline_ledger": str(VAL_BASE_LEDGER),
            "val_router_ledger": str(VAL_ROUTE_LEDGER),
            "oos_baseline_ledger": str(OOS_BASE_LEDGER),
            "oos_router_ledger": str(OOS_ROUTE_LEDGER),
        },
        "route_days": {
            "val": sorted(val_route_days),
            "oos": sorted(oos_route_days),
            "val_count": int(len(val_route_days)),
            "oos_count": int(len(oos_route_days)),
        },
        "block_summary": {"train2025": train_block_summary, "oos2026": eval_block_summary},
        "ledger_stats": {
            "val_baseline": _ledger_stats(ledgers.get("val_baseline", [])),
            "val_router": _ledger_stats(ledgers.get(f"val_prev_block_ratio_t{THRESHOLD:.2f}", [])),
            "oos_baseline": _ledger_stats(ledgers.get("oos_baseline", [])),
            "oos_router": _ledger_stats(ledgers.get(f"oos_prev_block_ratio_t{THRESHOLD:.2f}", [])),
        },
        "ledger_diff": ledger_diff,
        "decision": "Do not promote until more folds confirm this sparse path rule. Current result is shadow-only.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "monthly": str(MONTHLY_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
