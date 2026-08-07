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

from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts import precision_retest_alpha7_parent_soft_regime_veto_20260528 as parent_soft  # noqa: E402
from scripts import runtime_retest_alpha7_1_01965_decontam_20260528 as decontam  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import sweep_decontam_deep_alpha_controls_20260528 as sweep  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha7_parent_soft_regime_veto_walk_forward_20260528"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
BLOCKED_ROWS_OUT = OUT_DIR / "blocked_parent_rows.csv"


def _period_masks(df: pd.DataFrame, prefix: str) -> list[tuple[str, np.ndarray]]:
    ts = pd.to_datetime(df["timestamp"], errors="coerce")
    periods: list[tuple[str, np.ndarray]] = [(f"{prefix}_full", np.ones(len(df), dtype=bool))]
    for month in sorted(ts.dt.to_period("M").dropna().unique()):
        mask = (ts.dt.to_period("M") == month).to_numpy(dtype=bool)
        if int(mask.sum()) >= 500:
            periods.append((f"{prefix}_{month}", mask))
    for quarter in sorted(ts.dt.to_period("Q").dropna().unique()):
        mask = (ts.dt.to_period("Q") == quarter).to_numpy(dtype=bool)
        if int(mask.sum()) >= 1500:
            periods.append((f"{prefix}_{quarter}", mask))
    return periods


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
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for period, mask in _period_masks(df, split):
        res = sweep._backtest_variant(
            df=df.loc[mask].reset_index(drop=True),
            q=q[mask],
            dec=dec.loc[mask].reset_index(drop=True),
            stack=stack,
            cfg=cfg,
            variant=variant,
            cost_mult=3,
            record=False,
            deep_gate=None,
        )
        row = parent_soft._row(variant_name, split, period, res)
        rows.append(row)
    return rows


def _summary_by_period(grid: pd.DataFrame) -> list[dict[str, Any]]:
    base = grid[grid["variant"].eq("baseline")].copy()
    cand = grid[grid["variant"].eq("parent_soft_c65_conf70_q040_any")].copy()
    merged = base.merge(cand, on=["split", "period"], suffixes=("_baseline", "_candidate"))
    out = []
    for _, row in merged.iterrows():
        out.append(
            {
                "split": str(row["split"]),
                "period": str(row["period"]),
                "pnl_delta": float(row["pnl_candidate"] - row["pnl_baseline"]),
                "mdd_delta": float(row["mdd_candidate"] - row["mdd_baseline"]),
                "wr_delta": float(row["wr_candidate"] - row["wr_baseline"]),
                "score_delta": float(row["score_candidate"] - row["score_baseline"]),
                "baseline_pnl": float(row["pnl_baseline"]),
                "candidate_pnl": float(row["pnl_candidate"]),
                "baseline_mdd": float(row["mdd_baseline"]),
                "candidate_mdd": float(row["mdd_candidate"]),
                "baseline_score": float(row["score_baseline"]),
                "candidate_score": float(row["score_candidate"]),
            }
        )
    return out


def _aggregate_deltas(period_summary: list[dict[str, Any]], prefix: str) -> dict[str, Any]:
    rows = [r for r in period_summary if str(r["period"]).startswith(prefix) and not str(r["period"]).endswith("_full")]
    if not rows:
        return {"rows": 0}
    pnl = np.asarray([float(r["pnl_delta"]) for r in rows], dtype=np.float64)
    score = np.asarray([float(r["score_delta"]) for r in rows], dtype=np.float64)
    return {
        "rows": int(len(rows)),
        "pnl_delta_mean": float(pnl.mean()),
        "pnl_delta_median": float(np.median(pnl)),
        "pnl_delta_positive_count": int((pnl > 0).sum()),
        "score_delta_mean": float(score.mean()),
        "score_delta_median": float(np.median(score)),
        "score_delta_positive_count": int((score > 0).sum()),
    }


def _load_full_train_frame() -> pd.DataFrame:
    train_all = loop._merge_state24(loop._read(loop.v31.DEFAULT_TRAIN), loop.alpha3_full.SIDE_CLEAN4_2025)
    a7_train = loop._rename_clean4_v2(loop._read(loop.PRIMARY_TRAIN_CSV))
    return loop._augment_with_alpha7_features(train_all, a7_train).reset_index(drop=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    decontam._assert_clean_frame(decontam.TRAIN_CSV, name="train")
    decontam._assert_clean_frame(decontam.EVAL_CSV, name="eval")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    decontam._assert_clean_parent(decontam.CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    decontam._patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = precision._load_stack()
    train_df = _load_full_train_frame()
    _, eval_df = precision._load_frames()
    sources = precision._decision_sources(train_df, eval_df, stack["parent"])
    train_dec = sources[str(cfg["source"])][0].reset_index(drop=True)
    eval_dec = sources[str(cfg["source"])][1].reset_index(drop=True)

    train_q = v27._predict_all(stack["deep_model"], train_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    train_soft_dec, train_blocked, train_block_summary = parent_soft._parent_soft_veto(train_df, train_dec, split="train2025")
    eval_soft_dec, eval_blocked, eval_block_summary = parent_soft._parent_soft_veto(eval_df, eval_dec, split="oos2026")
    pd.concat([train_blocked, eval_blocked], ignore_index=True).to_csv(BLOCKED_ROWS_OUT, index=False)

    variant = sweep.Variant("deep_stop_cd18", deep_stop_cooldown_extra=18)
    rows: list[dict[str, Any]] = []
    rows.extend(
        _eval_periods(
            variant_name="baseline",
            split="train2025",
            df=train_df,
            q=train_q,
            dec=train_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.extend(
        _eval_periods(
            variant_name="parent_soft_c65_conf70_q040_any",
            split="train2025",
            df=train_df,
            q=train_q,
            dec=train_soft_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.extend(
        _eval_periods(
            variant_name="baseline",
            split="oos2026",
            df=eval_df,
            q=eval_q,
            dec=eval_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    rows.extend(
        _eval_periods(
            variant_name="parent_soft_c65_conf70_q040_any",
            split="oos2026",
            df=eval_df,
            q=eval_q,
            dec=eval_soft_dec,
            stack=stack,
            cfg=cfg,
            variant=variant,
        )
    )
    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    period_summary = _summary_by_period(grid)
    summary = {
        "model_id": MODEL_ID,
        "scope": "Wider walk-forward stability check for parent_soft_c65_conf70_q040_any on 2025 train frame and 2026 OOS frame. This is not untouched validation.",
        "grid": str(GRID_OUT),
        "blocked_rows": str(BLOCKED_ROWS_OUT),
        "block_summary": {"train2025": train_block_summary, "oos2026": eval_block_summary},
        "period_summary": period_summary,
        "aggregate": {
            "train2025_month_or_quarter": _aggregate_deltas(period_summary, "train2025_"),
            "oos2026_month_or_quarter": _aggregate_deltas(period_summary, "oos2026_"),
        },
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
