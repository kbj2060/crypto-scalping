#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as loop  # noqa: E402
from scripts import precision_retest_01965_alpha7_combo_20260527 as precision  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402

CANDIDATE_DIR = ROOT / "data/ensemble/supervised/alpha7_submodel_01965_decontam_v2_tp_20260528"
TRAIN_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2025_alpha6_current_tail111_exact.csv"
)
EVAL_CSV = (
    ROOT
    / "tmp/causal_regen_20260516/alpha7_1_01965_v2only_tp_sl_action_score_20260528/"
    "trade_candidates_2026_alpha6_current_tail111_exact.csv"
)
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_1_01965_decontam_runtime_retest_20260528"
FORBIDDEN_PREFIXES = ("clean_regime_2024_unsup_v4_", "clean_regime4_2024_unsup_v1_")


def _forbidden(cols: list[str]) -> list[str]:
    return [c for c in cols if c.startswith(FORBIDDEN_PREFIXES)]


def _assert_clean_parent(path: Path, *, name: str) -> None:
    parent = joblib.load(path)
    bad = _forbidden(list(parent["feature_cols"]))
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime features: {bad[:20]}")


def _assert_clean_frame(path: Path, *, name: str) -> None:
    cols = list(pd.read_csv(path, nrows=0).columns)
    bad = _forbidden(cols)
    if bad:
        raise RuntimeError(f"{name} contains forbidden legacy regime columns: {bad[:20]}")
    if "tp_sl_action_score" not in cols:
        raise RuntimeError(f"{name} missing tp_sl_action_score")


def _patch_runtime_sources() -> None:
    loop.ALPHA7_LIVE_DIR = CANDIDATE_DIR
    loop.PRIMARY_TRAIN_CSV = TRAIN_CSV
    loop.PRIMARY_EVAL_CSV = EVAL_CSV
    loop.PRIMARY_SUMMARY = CANDIDATE_DIR / "primary_summary.json"
    loop.FALLBACK_SUMMARY = CANDIDATE_DIR / "fallback_alpha43_no_legacy_summary.json"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _assert_clean_frame(TRAIN_CSV, name="train")
    _assert_clean_frame(EVAL_CSV, name="eval")
    _assert_clean_parent(CANDIDATE_DIR / "primary_parent.pkl", name="primary")
    _assert_clean_parent(CANDIDATE_DIR / "fallback_alpha43_no_legacy_parent.pkl", name="fallback")
    _patch_runtime_sources()

    cfg = precision._cfg_from_results()
    stack = loop._load_stack()
    val_df, eval_df = loop._load_frames()
    sources = loop._decision_sources(val_df, eval_df, stack["parent"])
    val_q = precision.v27._predict_all(stack["deep_model"], val_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])
    eval_q = precision.v27._predict_all(stack["deep_model"], eval_df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, list[dict[str, Any]]] = {}
    for split_name, df, q, side in (("val", val_df, val_q, 0), ("oos", eval_df, eval_q, 1)):
        dec = sources[str(cfg["source"])][side]
        for cost in (1, 2, 3):
            record = cost == 3
            row = precision._eval(df=df, q=q, dec=dec, stack=stack, cfg=cfg, period=f"{split_name}_full", cost_mult=cost, record=record)
            if record:
                ledgers[split_name] = list(row.pop("_records", []))
            rows.append(row)

    grid = pd.DataFrame(rows)
    grid_path = OUT_DIR / "cost_grid.csv"
    grid.to_csv(grid_path, index=False)
    oos_ledger_path = OUT_DIR / "oos_cost3_ledger.csv"
    val_ledger_path = OUT_DIR / "val_cost3_ledger.csv"
    pd.DataFrame(ledgers.get("oos", [])).to_csv(oos_ledger_path, index=False)
    pd.DataFrame(ledgers.get("val", [])).to_csv(val_ledger_path, index=False)
    summary = {
        "model_id": "alpha7_1_01965_decontam_runtime_retest_20260528",
        "candidate_dir": str(CANDIDATE_DIR),
        "train_csv": str(TRAIN_CSV),
        "eval_csv": str(EVAL_CSV),
        "config": cfg,
        "cost_grid": str(grid_path),
        "oos_cost3_ledger": str(oos_ledger_path),
        "val_cost3_ledger": str(val_ledger_path),
        "cost3_full": grid[grid["cost"].eq(3)].to_dict(orient="records"),
        "oos_ledger_stats": precision._ledger_stats(oos_ledger_path),
        "val_ledger_stats": precision._ledger_stats(val_ledger_path),
    }
    summary_path = OUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "cost_grid": str(grid_path)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
