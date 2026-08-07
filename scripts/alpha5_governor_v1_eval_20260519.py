#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.alpha5_governor_v1_infer_20260519 import compose_rank_veto, load_governor, predict_heads  # noqa: E402
from scripts.train_eval_alpha5_23_hgb_direction_refined_20260519 import _eval  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_governor_v1_eval_20260519"
DEFAULT_MODEL_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_contracts_20260519"
DEFAULT_DATA_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable005_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_governor_v1_eval_20260519"


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate alpha5 governor v1 rank/veto composition on validation and 2026 OOS.")
    p.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--fee", type=float, default=0.0005)
    p.add_argument("--slip", type=float, default=0.0002)
    p.add_argument("--exposure", type=float, default=1.0)
    p.add_argument("--max-hold-bars", type=int, default=96)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    governor = load_governor(args.model_dir)
    val_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_val.parquet")
    oos_df = pd.read_parquet(args.data_dir / "alpha5_30_direction_learnable_oos.parquet")
    labels_val = pd.to_numeric(val_df["label_action"], errors="coerce").fillna(0.0).to_numpy(np.int64)
    labels_oos = pd.to_numeric(oos_df["label_action"], errors="coerce").fillna(0.0).to_numpy(np.int64)

    pred_val = predict_heads(governor, val_df)
    pred_oos = predict_heads(governor, oos_df)

    rows: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for trade_score_min, ambiguous_struct_cap, ambiguous_trade_like_cap, quality_min, side_threshold, side_margin_min, score_abs_min in product(
        [0.00, 0.02, 0.05, 0.08],
        [0.30, 0.40, 0.50, 0.60],
        [0.55, 0.70, 0.85, 1.00],
        [-0.10, -0.05, 0.00, 0.05],
        [0.48, 0.50, 0.55],
        [0.00, 0.03, 0.05],
        [0.05, 0.08, 0.10],
    ):
        actions_val, _ = compose_rank_veto(
            val_df,
            pred_val,
            trade_score_min=trade_score_min,
            ambiguous_struct_cap=ambiguous_struct_cap,
            ambiguous_trade_like_cap=ambiguous_trade_like_cap,
            quality_min=quality_min,
            side_threshold=side_threshold,
            side_margin_min=side_margin_min,
            score_abs_min=score_abs_min,
        )
        val_eval = _eval(
            val_df,
            actions_val,
            labels_val,
            fee=args.fee,
            slip=args.slip,
            exposure=args.exposure,
            max_hold=args.max_hold_bars,
        )
        val_trades = int(val_eval["backtest"]["cost1"]["trades"])
        trade_bonus = min(val_trades, 180) * 0.08 - max(0, val_trades - 220) * 0.12
        selection_score = float(val_eval["score"]) + float(trade_bonus)
        row = {
            "trade_score_min": float(trade_score_min),
            "ambiguous_struct_cap": float(ambiguous_struct_cap),
            "ambiguous_trade_like_cap": float(ambiguous_trade_like_cap),
            "quality_min": float(quality_min),
            "side_threshold": float(side_threshold),
            "side_margin_min": float(side_margin_min),
            "score_abs_min": float(score_abs_min),
            "val_score": float(val_eval["score"]),
            "selection_score": float(selection_score),
            "val_cost1_pnl": float(val_eval["backtest"]["cost1"]["pnl"]),
            "val_cost1_mdd": float(val_eval["backtest"]["cost1"]["mdd"]),
            "val_trades": val_trades,
            "val_trade_precision": float(val_eval["direction"]["trade_precision"]),
            "val_balanced_trade_precision": float(val_eval["direction"]["balanced_trade_precision"]),
            "val_coverage": float(val_eval["direction"]["coverage"]),
        }
        rows.append(row)
        if best is None or row["selection_score"] > best["selection_score"]:
            best = row

    assert best is not None
    actions_oos, _ = compose_rank_veto(
        oos_df,
        pred_oos,
        trade_score_min=float(best["trade_score_min"]),
        ambiguous_struct_cap=float(best["ambiguous_struct_cap"]),
        ambiguous_trade_like_cap=float(best["ambiguous_trade_like_cap"]),
        quality_min=float(best["quality_min"]),
        side_threshold=float(best["side_threshold"]),
        side_margin_min=float(best["side_margin_min"]),
        score_abs_min=float(best["score_abs_min"]),
    )
    oos_eval = _eval(
        oos_df,
        actions_oos,
        labels_oos,
        fee=args.fee,
        slip=args.slip,
        exposure=args.exposure,
        max_hold=args.max_hold_bars,
    )
    summary = {
        "model_id": MODEL_ID,
        "selection": best,
        "validation": best,
        "oos": {
            "cost1": oos_eval["backtest"]["cost1"],
            "cost2": oos_eval["backtest"]["cost2"],
            "cost3": oos_eval["backtest"]["cost3"],
            "direction": oos_eval["direction"],
            "score": float(oos_eval["score"]),
        },
    }
    (args.out_dir / "alpha5_governor_v1_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    pd.DataFrame(rows).sort_values("val_score", ascending=False).to_csv(
        args.out_dir / "alpha5_governor_v1_eval_grid.csv",
        index=False,
    )
    print(
        json.dumps(
            {
                "stage": "alpha5_governor_v1_eval_done",
                "summary_path": str(args.out_dir / "alpha5_governor_v1_eval_summary.json"),
                "val_trades": int(best["val_trades"]),
                "oos_pnl": float(oos_eval["backtest"]["cost1"]["pnl"]),
                "oos_trades": int(oos_eval["backtest"]["cost1"]["trades"]),
            },
            ensure_ascii=False,
            default=_json_default,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
