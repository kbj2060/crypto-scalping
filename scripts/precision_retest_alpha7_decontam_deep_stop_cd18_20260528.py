#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_decontam_deep_stop_cd18_precision_20260528"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
GRID_OUT = OUT_DIR / "cost_period_grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"
OOS_LEDGER_OUT = OUT_DIR / "oos_cost3_ledger.csv"
VAL_LEDGER_OUT = OUT_DIR / "val_cost3_ledger.csv"
BASELINE_OOS_LEDGER_OUT = OUT_DIR / "baseline_oos_cost3_ledger.csv"


def _ledger_stats(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"rows": 0}
    df = pd.DataFrame(records)
    ret = pd.to_numeric(df["trade_return"], errors="coerce").fillna(0.0)
    cash = pd.to_numeric(df["cash_after"], errors="coerce")
    by_owner: dict[str, Any] = {}
    for owner, g in df.groupby("owner"):
        gr = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        by_owner[str(owner)] = {
            "trades": int(len(g)),
            "sum": float(gr.sum()),
            "mean": float(gr.mean()),
            "raw_wr": float((gr > 0).mean()),
        }
    by_side: dict[str, Any] = {}
    for side, g in df.groupby("side"):
        gr = pd.to_numeric(g["trade_return"], errors="coerce").fillna(0.0)
        by_side[str(side)] = {
            "trades": int(len(g)),
            "sum": float(gr.sum()),
            "mean": float(gr.mean()),
            "raw_wr": float((gr > 0).mean()),
        }
    peak = cash.cummax()
    dd = cash / peak - 1.0
    return {
        "rows": int(len(df)),
        "raw_sum": float(ret.sum()),
        "raw_mean": float(ret.mean()),
        "raw_median": float(ret.median()),
        "raw_wr": float((ret > 0).mean()),
        "final_cash_after": float(cash.iloc[-1]),
        "closed_trade_mdd": float(dd.min()),
        "top5_sum": float(ret[ret > 0].sort_values(ascending=False).head(5).sum()),
        "bottom5_sum": float(ret.sort_values().head(5).sum()),
        "by_owner": by_owner,
        "by_side": by_side,
    }


def _exits_dict(res: dict[str, Any]) -> dict[str, int]:
    return {str(k): int(v) for k, v in dict(res.get("exits", {})).items()}


def _row(
    *,
    variant: sweep.Variant,
    split: str,
    period: str,
    cost: int,
    res: dict[str, Any],
) -> dict[str, Any]:
    return {
        "model": MODEL_ID,
        "variant": variant.name,
        "split": split,
        "period": period,
        "cost": int(cost),
        "pnl": float(res["pnl"]),
        "mdd": float(res["mdd"]),
        "wr": float(res["wr"]),
        "trades": int(res["trades"]),
        "trades_per_day": float(res["trades_per_day"]),
        "deep_entries": int(res.get("deep_entries", 0)),
        "long_entries": int(res.get("long_entries", 0)),
        "short_entries": int(res.get("short_entries", 0)),
        "avg_notional": float(res.get("avg_notional", 0.0)),
        "avg_leverage": float(res.get("avg_leverage", 0.0)),
        "sl_ratio": float(sweep._sl_ratio(res)),
        "score": float(sweep._score(res)),
        "exits": json.dumps(_exits_dict(res), ensure_ascii=False, sort_keys=True),
    }


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
    val_dec = sources[str(cfg["source"])][0]
    eval_dec = sources[str(cfg["source"])][1]

    baseline = sweep.Variant("baseline")
    selected = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    rows: list[dict[str, Any]] = []
    ledger_records: dict[str, list[dict[str, Any]]] = {}

    for variant in (baseline, selected):
        for split_name, df, q, dec in (
            ("val", val_df, val_q, val_dec),
            ("oos", eval_df, eval_q, eval_dec),
        ):
            for period_name, mask in precision._periods(df, split_name):
                sub_df = df.loc[mask].reset_index(drop=True)
                sub_q = q[mask]
                sub_dec = dec.loc[mask].reset_index(drop=True)
                for cost in (1, 2, 3):
                    record = cost == 3 and period_name in {"val_full", "oos_full"}
                    res = sweep._backtest_variant(
                        df=sub_df,
                        q=sub_q,
                        dec=sub_dec,
                        stack=stack,
                        cfg=cfg,
                        variant=variant,
                        cost_mult=cost,
                        record=record,
                    )
                    if record:
                        ledger_records[f"{variant.name}_{period_name}"] = list(res.pop("trade_records", []))
                    rows.append(_row(variant=variant, split=split_name, period=period_name, cost=cost, res=res))

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    pd.DataFrame(ledger_records.get("deep_stop_cd18_oos_full", [])).to_csv(OOS_LEDGER_OUT, index=False)
    pd.DataFrame(ledger_records.get("deep_stop_cd18_val_full", [])).to_csv(VAL_LEDGER_OUT, index=False)
    pd.DataFrame(ledger_records.get("baseline_oos_full", [])).to_csv(BASELINE_OOS_LEDGER_OUT, index=False)

    cost3_full = grid[grid["cost"].eq(3) & grid["period"].isin(["val_full", "oos_full"])].copy()
    cost3_month = grid[grid["cost"].eq(3) & ~grid["period"].isin(["val_full", "oos_full"])].copy()
    summary = {
        "model_id": MODEL_ID,
        "base_model": "alpha7_submodel_01965_decontam_v2_tp_20260528",
        "selected_variant": "deep_stop_cd18",
        "variant_logic": "After a deep_alpha hard/soft stop-loss exit, set deep-only cooldown to at least 18 bars. Parent/v21_2 path remains unchanged.",
        "candidate_dir": str(decontam.CANDIDATE_DIR),
        "grid": str(GRID_OUT),
        "oos_cost3_ledger": str(OOS_LEDGER_OUT),
        "val_cost3_ledger": str(VAL_LEDGER_OUT),
        "baseline_oos_cost3_ledger": str(BASELINE_OOS_LEDGER_OUT),
        "cost3_full": cost3_full.to_dict(orient="records"),
        "cost3_monthly": cost3_month.to_dict(orient="records"),
        "oos_ledger_stats": _ledger_stats(ledger_records.get("deep_stop_cd18_oos_full", [])),
        "val_ledger_stats": _ledger_stats(ledger_records.get("deep_stop_cd18_val_full", [])),
        "baseline_oos_ledger_stats": _ledger_stats(ledger_records.get("baseline_oos_full", [])),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "oos_ledger": str(OOS_LEDGER_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
