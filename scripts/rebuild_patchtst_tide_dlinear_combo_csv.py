#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_governor_ai_feature_combo_grid import (  # noqa: E402
    AI_EVAL_CSV,
    AI_TRAIN_CSV,
    BASE_EVAL_CSV,
    BASE_TRAIN_CSV,
    DEFAULT_OUT_DIR,
    _build_combo_csv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rebuild only the PatchTST+TiDE+DLinear candidate CSVs.")
    p.add_argument("--base-train-csv", type=Path, default=BASE_TRAIN_CSV)
    p.add_argument("--base-eval-csv", type=Path, default=BASE_EVAL_CSV)
    p.add_argument("--ai-train-csv", type=Path, default=AI_TRAIN_CSV)
    p.add_argument("--ai-eval-csv", type=Path, default=AI_EVAL_CSV)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument("--summary-out", type=Path, default=ROOT / "data/ensemble/reports/patchtst_tide_dlinear_combo_rebuild_2026.json")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    groups = ("patchtst", "tide", "dlinear")
    name = "__".join(groups)
    train_out = args.out_dir / f"trade_candidates_2025_{name}.csv"
    eval_out = args.out_dir / f"trade_candidates_2026_{name}.csv"
    train_info = _build_combo_csv(args.base_train_csv, args.ai_train_csv, groups, train_out)
    eval_info = _build_combo_csv(args.base_eval_csv, args.ai_eval_csv, groups, eval_out)
    summary = {
        "type": "patchtst_tide_dlinear_combo_rebuild_2026",
        "groups": list(groups),
        "train": train_info,
        "eval": eval_info,
    }
    args.summary_out.parent.mkdir(parents=True, exist_ok=True)
    args.summary_out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
