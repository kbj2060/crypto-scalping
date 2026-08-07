#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_alpha7_01965_full_long_short_router_20260528 as ls  # noqa: E402
from scripts import train_eval_alpha7_01965_router_overlay_refinements_20260528 as overlay_refine  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _json_default,
    _read,
)
from scripts.retrain_alpha7_1_01965_tp_sl_decontam_20260528 import TRAIN_CSV  # noqa: E402


MODEL_ID = "alpha7_01965_short_override_precision_retest_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
BASE_LS_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_full_long_short_router_20260528"
BASE_OVERLAY_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_01965_router_overlay_refinements_20260528"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "threshold_period_grid.csv"
MONTHLY_OUT = OUT_DIR / "monthly_cost3.csv"
SELECTED_OOS_LEDGER_OUT = OUT_DIR / "selected_short_override_oos_cost3_ledger.csv"
BASELINE_OOS_LEDGER_OUT = OUT_DIR / "baseline_oos_cost3_ledger.csv"


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            out.append((f"{prefix}_{month}", mask))
    return out


def _row(variant: str, split: str, period: str, threshold: float | None, res: dict[str, Any], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    row = {
        "variant": variant,
        "split": split,
        "period": period,
        "threshold": threshold,
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res.get("trades_per_day", 0.0)),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "exits": json.dumps(res.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }
    if extra:
        row.update(extra)
    return row


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    winners = ret[ret > 0].sort_values(ascending=False)
    losers = ret[ret <= 0].sort_values()
    return {
        "rows": int(len(df)),
        "return_sum": float(ret.sum()),
        "return_mean": float(ret.mean()),
        "return_median": float(ret.median()),
        "top5_return_sum": float(winners.head(5).sum()) if len(winners) else 0.0,
        "bottom5_return_sum": float(losers.head(5).sum()) if len(losers) else 0.0,
        "win_count": int((ret > 0).sum()),
        "loss_count": int((ret <= 0).sum()),
        "median_hold_bars": float(pd.to_numeric(df["hold_bars"], errors="coerce").median()),
        "max_hold_bars": int(pd.to_numeric(df["hold_bars"], errors="coerce").max()),
    }


def _eval_periods(
    *,
    variant_name: str,
    split: str,
    df: pd.DataFrame,
    q: np.ndarray,
    dec: pd.DataFrame,
    stack: dict[str, Any],
    cfg: dict[str, Any],
    variant: sweep.Variant,
    threshold: float | None,
    record_full: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    full_records: list[dict[str, Any]] = []
    for period_name, mask in _periods(df, split):
        sub_df = df.loc[mask].reset_index(drop=True)
        sub_q = q[mask]
        sub_dec = dec.loc[mask].reset_index(drop=True)
        record = bool(record_full and period_name == f"{split}_full")
        res = sweep._backtest_variant(
            df=sub_df,
            q=sub_q,
            dec=sub_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=record,
            deep_gate=None,
        )
        if record:
            full_records = list(res.pop("trade_records", []))
        rows.append(_row(variant_name, split, period_name, threshold, res))
    return rows, full_records


def _capped_cfg_variant(cfg: dict[str, Any], stack: dict[str, Any], cap: float) -> tuple[dict[str, Any], sweep.Variant]:
    capped_cfg = dict(cfg)
    capped_cfg["parent_notional_cap"] = min(float(capped_cfg["parent_notional_cap"]), float(cap))
    base_overlay = precision._overlay(stack["overlay"], cfg)
    deep_mult = min(1.0, float(cap) / max(float(base_overlay.notional), 1e-12))
    return capped_cfg, sweep.Variant(f"deep_stop_cd18_cap{cap:.2f}", deep_stop_cooldown_extra=18, deep_notional_mult=deep_mult)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    val_df, eval_df = precision._load_frames()
    sources = precision._decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    base_val_dec = sources[str(cfg["source"])][0]
    base_eval_dec = sources[str(cfg["source"])][1]

    train_all = _read(TRAIN_CSV)
    train_df = train_all[train_all["timestamp"] < SPLIT_TS].reset_index(drop=True)
    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    primary_cols = list(joblib.load(ls.PRIMARY_PARENT)["feature_cols"])
    fallback_cols = list(joblib.load(ls.FALLBACK_PARENT)["feature_cols"])
    long_primary, _ = ls._train_side_parent(train_df=train_df, feature_cols=primary_cols, action_keep=1, seed=5289801, out_dir=BASE_LS_DIR / "long_primary")
    long_fallback, _ = ls._train_side_parent(train_df=train_df, feature_cols=fallback_cols, action_keep=1, seed=5289802, out_dir=BASE_LS_DIR / "long_fallback")
    short_primary, _ = ls._train_side_parent(train_df=train_df, feature_cols=primary_cols, action_keep=2, seed=5289901, out_dir=BASE_LS_DIR / "short_primary")
    short_fallback, _ = ls._train_side_parent(train_df=train_df, feature_cols=fallback_cols, action_keep=2, seed=5289902, out_dir=BASE_LS_DIR / "short_fallback")

    train_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=train_df, side_keep=1)
    train_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=train_df, side_keep=-1)
    val_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=val_df, side_keep=1)
    val_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=val_df, side_keep=-1)
    eval_long_dec = ls._predict_parent_pair(primary=long_primary, fallback=long_fallback, df=eval_df, side_keep=1)
    eval_short_dec = ls._predict_parent_pair(primary=short_primary, fallback=short_fallback, df=eval_df, side_keep=-1)

    opp_model, opp_cols, opp_summary = overlay_refine._train_opportunity_router(
        train_df=train_df,
        long_dec=train_long_dec,
        short_dec=train_short_dec,
        q=train_q,
        stack=stack,
        cfg=cfg,
        out_dir=BASE_OVERLAY_DIR / "opportunity_router",
    )

    baseline_variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    baseline_val_rows, _ = _eval_periods(
        variant_name="deep_stop_cd18_baseline",
        split="val",
        df=val_df,
        q=val_q,
        dec=base_val_dec,
        stack=stack,
        cfg=cfg,
        variant=baseline_variant,
        threshold=None,
    )
    baseline_oos_rows, baseline_oos_records = _eval_periods(
        variant_name="deep_stop_cd18_baseline",
        split="oos",
        df=eval_df,
        q=eval_q,
        dec=base_eval_dec,
        stack=stack,
        cfg=cfg,
        variant=baseline_variant,
        threshold=None,
        record_full=True,
    )
    rows.extend([r for r in baseline_val_rows + baseline_oos_rows if r["period"].endswith("_full")])
    monthly_rows.extend([r for r in baseline_val_rows + baseline_oos_rows if not r["period"].endswith("_full")])

    threshold_summaries: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    thresholds = [0.50, 0.55, 0.60, 0.625, 0.65, 0.675, 0.70, 0.75, 0.80]
    selected_oos_records: list[dict[str, Any]] = []
    selected_decisions: tuple[pd.DataFrame, pd.DataFrame] | None = None
    for th in thresholds:
        val_pred = overlay_refine._short_route_by_probability(
            model=opp_model,
            feature_cols=opp_cols,
            df=val_df,
            long_dec=val_long_dec,
            short_dec=val_short_dec,
            q=val_q,
            threshold=th,
        )
        oos_pred = overlay_refine._short_route_by_probability(
            model=opp_model,
            feature_cols=opp_cols,
            df=eval_df,
            long_dec=eval_long_dec,
            short_dec=eval_short_dec,
            q=eval_q,
            threshold=th,
        )
        val_dec = overlay_refine._short_override_decisions(base_val_dec, val_short_dec, val_pred)
        oos_dec = overlay_refine._short_override_decisions(base_eval_dec, eval_short_dec, oos_pred)
        val_rows, _ = _eval_periods(
            variant_name="short_override_overlay",
            split="val",
            df=val_df,
            q=val_q,
            dec=val_dec,
            stack=stack,
            cfg=cfg,
            variant=baseline_variant,
            threshold=th,
        )
        oos_rows, oos_records = _eval_periods(
            variant_name="short_override_overlay",
            split="oos",
            df=eval_df,
            q=eval_q,
            dec=oos_dec,
            stack=stack,
            cfg=cfg,
            variant=baseline_variant,
            threshold=th,
            record_full=True,
        )
        full_rows = [r for r in val_rows + oos_rows if r["period"].endswith("_full")]
        month_rows = [r for r in val_rows + oos_rows if not r["period"].endswith("_full")]
        rows.extend(full_rows)
        monthly_rows.extend(month_rows)
        val_full = next(r for r in full_rows if r["period"] == "val_full")
        oos_full = next(r for r in full_rows if r["period"] == "oos_full")
        threshold_summary = {
            "threshold": float(th),
            "val_full": val_full,
            "oos_full": oos_full,
            "val_route_distribution": pd.Series(val_pred).value_counts().sort_index().to_dict(),
            "oos_route_distribution": pd.Series(oos_pred).value_counts().sort_index().to_dict(),
        }
        threshold_summaries.append(threshold_summary)
        if selected is None or float(val_full["score"]) > float(selected["val_full"]["score"]):
            selected = threshold_summary
            selected_oos_records = oos_records
            selected_decisions = (val_dec, oos_dec)

    if selected is None or selected_decisions is None:
        raise RuntimeError("no threshold selected")

    capped_rows: list[dict[str, Any]] = []
    selected_val_dec, selected_oos_dec = selected_decisions
    for cap in (0.25, 0.50):
        cap_cfg, cap_variant = _capped_cfg_variant(cfg, stack, cap)
        for variant_name, val_dec, oos_dec in (
            (f"baseline_cap{cap:.2f}", base_val_dec, base_eval_dec),
            (f"short_override_cap{cap:.2f}", selected_val_dec, selected_oos_dec),
        ):
            cap_val_rows, _ = _eval_periods(
                variant_name=variant_name,
                split="val",
                df=val_df,
                q=val_q,
                dec=val_dec,
                stack=stack,
                cfg=cap_cfg,
                variant=cap_variant,
                threshold=float(selected["threshold"]) if "short_override" in variant_name else None,
            )
            cap_oos_rows, _ = _eval_periods(
                variant_name=variant_name,
                split="oos",
                df=eval_df,
                q=eval_q,
                dec=oos_dec,
                stack=stack,
                cfg=cap_cfg,
                variant=cap_variant,
                threshold=float(selected["threshold"]) if "short_override" in variant_name else None,
            )
            capped_rows.extend([r for r in cap_val_rows + cap_oos_rows if r["period"].endswith("_full")])

    pd.DataFrame(rows + capped_rows).to_csv(GRID_OUT, index=False)
    pd.DataFrame(monthly_rows).to_csv(MONTHLY_OUT, index=False)
    pd.DataFrame(selected_oos_records).to_csv(SELECTED_OOS_LEDGER_OUT, index=False)
    pd.DataFrame(baseline_oos_records).to_csv(BASELINE_OOS_LEDGER_OUT, index=False)

    selected_threshold = float(selected["threshold"])
    selected_monthly = [r for r in monthly_rows if r["variant"] == "short_override_overlay" and r["threshold"] == selected_threshold]
    baseline_monthly = [r for r in monthly_rows if r["variant"] == "deep_stop_cd18_baseline"]
    summary = {
        "model_id": MODEL_ID,
        "scope": "Precision retest for short_override_overlay only. Active/live artifacts unchanged.",
        "threshold_selection_rule": "highest validation full-period score",
        "selected_threshold": selected_threshold,
        "selected": selected,
        "baseline_full": [r for r in rows if r["variant"] == "deep_stop_cd18_baseline"],
        "threshold_summaries": threshold_summaries,
        "selected_monthly_cost3": selected_monthly,
        "baseline_monthly_cost3": baseline_monthly,
        "capped_full_cost3": capped_rows,
        "baseline_oos_ledger": str(BASELINE_OOS_LEDGER_OUT),
        "selected_oos_ledger": str(SELECTED_OOS_LEDGER_OUT),
        "baseline_oos_ledger_stats": _ledger_stats(baseline_oos_records),
        "selected_oos_ledger_stats": _ledger_stats(selected_oos_records),
        "opportunity_router_summary": opp_summary,
        "grid": str(GRID_OUT),
        "monthly_grid": str(MONTHLY_OUT),
        "warning": "Available OOS is 2026-01-01 to 2026-02-28 16:00 only. This is a stability retest, not an untouched deployment validation.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "monthly": str(MONTHLY_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
