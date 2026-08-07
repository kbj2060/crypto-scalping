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

from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    BEST_JSON,
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
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.backtest_alpha3_exit_guard_persistence_20260527 import backtest_signal_limit_exit_guard  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha3_1_alpha6_alpha7_combo_stability_retest_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
GRID_OUT = OUT_DIR / "period_grid.csv"
SUMMARY_OUT = OUT_DIR / "summary.json"


def _cfg_from_row(row: pd.Series) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "name": str(row["name"]),
        "source": str(row["source"]),
        "tag": str(row.get("tag", "selected")),
    }
    for col, value in row.items():
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


def _select_candidates(results: pd.DataFrame) -> list[dict[str, Any]]:
    picks: list[pd.Series] = []
    names: set[str] = set()

    def add(rows: pd.DataFrame) -> None:
        for _, row in rows.iterrows():
            name = str(row["name"])
            if name in names:
                continue
            names.add(name)
            picks.append(row)

    add(results.sort_values("oos_pnl", ascending=False).head(12))
    add(results.sort_values("val_score", ascending=False).head(12))
    robust = results[
        (pd.to_numeric(results["val_pnl"], errors="coerce") > 0.0)
        & (pd.to_numeric(results["oos_pnl"], errors="coerce") > 100.0)
        & (pd.to_numeric(results["oos_mdd"], errors="coerce") > -30.0)
    ].sort_values(["oos_pnl", "val_score"], ascending=[False, False])
    add(robust.head(12))
    return [_cfg_from_row(row) for row in picks]


def _periods(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    out: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    months = sorted(ts.dt.to_period("M").dropna().unique())
    for month in months:
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if mask.sum() >= 500:
            out.append((f"{prefix}_{month}", mask))
    return out


def _eval_period(
    *,
    period_name: str,
    df: pd.DataFrame,
    q: np.ndarray,
    sources: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    source_side: int,
    stack: dict[str, Any],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    src = str(cfg["source"])
    dec = sources[src][source_side].reset_index(drop=True)
    mod_dec = _apply_decision_mods(dec, cfg)
    guard = _guard(cfg)
    overlay = _overlay(stack["overlay"], cfg)
    limit_cfg = _default_limit_cfg()
    c3 = backtest_signal_limit_exit_guard(
        df.reset_index(drop=True),
        stack["parent"],
        stack["runner"],
        stack["add_cfg"],
        q,
        mod_dec.reset_index(drop=True),
        overlay,
        limit_cfg,
        guard,
        fee=stack["fee"],
        slip=stack["slip"],
        cost_mult=3.0,
    )
    return {
        "candidate": str(cfg["name"]),
        "source": src,
        "period": period_name,
        "pnl": float(c3["pnl"]),
        "mdd": float(c3["mdd"]),
        "wr": float(c3["wr"]),
        "trades": int(c3["trades"]),
        "sl_ratio": float(_sl_ratio(c3)),
        "score": float(_score(c3)),
        "deep_entries": int(c3.get("deep_entries", 0)),
        "long_entries": int(c3.get("long_entries", 0)),
        "short_entries": int(c3.get("short_entries", 0)),
        "exits": json.dumps(c3.get("exits", {}), ensure_ascii=False, sort_keys=True),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = pd.read_csv(RESULTS_CSV)
    candidates = _select_candidates(results)
    if not candidates:
        raise RuntimeError(f"no candidates selected from {RESULTS_CSV}")

    stack = _load_stack()
    val_df, eval_df = _load_frames()
    sources = _decision_sources(val_df, eval_df, stack["parent"])
    val_q = v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    rows: list[dict[str, Any]] = []
    for cfg in candidates:
        for period_name, mask in _periods(val_df, "val"):
            rows.append(
                _eval_period(
                    period_name=period_name,
                    df=val_df.loc[mask].reset_index(drop=True),
                    q=val_q[mask],
                    sources={k: (v[0].loc[mask].reset_index(drop=True), v[1]) for k, v in sources.items()},
                    source_side=0,
                    stack=stack,
                    cfg=cfg,
                )
            )
        for period_name, mask in _periods(eval_df, "oos"):
            rows.append(
                _eval_period(
                    period_name=period_name,
                    df=eval_df.loc[mask].reset_index(drop=True),
                    q=eval_q[mask],
                    sources={k: (v[0], v[1].loc[mask].reset_index(drop=True)) for k, v in sources.items()},
                    source_side=1,
                    stack=stack,
                    cfg=cfg,
                )
            )

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)

    summary_rows: list[dict[str, Any]] = []
    for name, g in grid.groupby("candidate"):
        val = g[g["period"].astype(str).str.startswith("val_")]
        oos = g[g["period"].astype(str).str.startswith("oos_")]
        val_month = val[val["period"] != "val_full"]
        oos_month = oos[oos["period"] != "oos_full"]
        full_val = val[val["period"] == "val_full"].iloc[0].to_dict()
        full_oos = oos[oos["period"] == "oos_full"].iloc[0].to_dict()
        summary_rows.append(
            {
                "candidate": name,
                "source": str(g["source"].iloc[0]),
                "val_full_pnl": float(full_val["pnl"]),
                "val_full_mdd": float(full_val["mdd"]),
                "val_full_wr": float(full_val["wr"]),
                "oos_full_pnl": float(full_oos["pnl"]),
                "oos_full_mdd": float(full_oos["mdd"]),
                "oos_full_wr": float(full_oos["wr"]),
                "oos_full_trades": int(full_oos["trades"]),
                "val_month_min_pnl": float(val_month["pnl"].min()),
                "val_month_positive_rate": float((val_month["pnl"] > 0.0).mean()),
                "oos_month_min_pnl": float(oos_month["pnl"].min()),
                "oos_month_positive_rate": float((oos_month["pnl"] > 0.0).mean()),
                "combined_score": float(
                    full_oos["pnl"]
                    + 0.50 * full_val["pnl"]
                    + 20.0 * full_oos["wr"]
                    + 10.0 * full_val["wr"]
                    + 2.0 * full_oos["mdd"]
                    + 1.0 * full_val["mdd"]
                    + 25.0 * (oos_month["pnl"] > 0.0).mean()
                    + 15.0 * (val_month["pnl"] > 0.0).mean()
                    + min(0.0, float(oos_month["pnl"].min()))
                    + min(0.0, float(val_month["pnl"].min()))
                ),
            }
        )

    ranking = pd.DataFrame(summary_rows).sort_values("combined_score", ascending=False).reset_index(drop=True)
    ranking_path = OUT_DIR / "candidate_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    summary = {
        "model_id": MODEL_ID,
        "input_loop_results": str(RESULTS_CSV),
        "candidate_count": int(len(candidates)),
        "period_grid": str(GRID_OUT),
        "ranking": str(ranking_path),
        "best_combined": ranking.iloc[0].to_dict() if len(ranking) else None,
        "best_oos_from_loop": json.loads(BEST_JSON.read_text(encoding="utf-8")).get("best_oos_pnl_observed"),
        "best_validation_from_loop": json.loads(BEST_JSON.read_text(encoding="utf-8")).get("best_validation_selected"),
        "warning": "No post-2026-02-28 untouched common feature CSV was available. This is a month-split stability retest, not a fresh OOS promotion test.",
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "ranking": str(ranking_path), "period_grid": str(GRID_OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
