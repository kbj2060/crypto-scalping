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

from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 import (  # noqa: E402
    _decision_sources,
    _load_frames,
    _load_stack,
)
from scripts.precision_retest_01965_alpha7_combo_20260527 import (  # noqa: E402
    CANDIDATE,
    _cfg_from_results,
    _eval,
)
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "ablate_01965_deep_teacher_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
AUDIT_OUT = OUT_DIR / "audit.json"
TEACHER_COLS = [
    "teacher_side_margin",
    "teacher_side_disagreement",
    "teacher_uncertainty",
    "teacher_tail_warning",
]


def _zero_teacher_for_deep(df: pd.DataFrame, seq_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    used = [c for c in TEACHER_COLS if c in seq_cols]
    missing = [c for c in used if c not in out.columns]
    if missing:
        raise RuntimeError(f"teacher columns used by deep scout but missing from frame: {missing}")
    for c in used:
        out[c] = 0.0
    return out


def _predict_deep(stack: dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    return v27._predict_all(stack["deep_model"], df, stack["deep_payload"]["seq_cols"], stack["deep_payload"]["norm"])


def _deep_disabled_q(n: int) -> np.ndarray:
    return np.full((int(n), 2), -1.0, dtype=np.float32)


def _teacher_audit(df: pd.DataFrame, seq_cols: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in TEACHER_COLS:
        if c not in seq_cols:
            out[c] = {"in_seq_cols": False}
            continue
        if c not in df.columns:
            out[c] = {"in_seq_cols": True, "present": False}
            continue
        s = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out[c] = {
            "in_seq_cols": True,
            "present": True,
            "na": int(s.isna().sum()),
            "nonzero": int((s.fillna(0.0) != 0.0).sum()),
            "mean": float(s.fillna(0.0).mean()),
            "std": float(s.fillna(0.0).std()),
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = _cfg_from_results()
    stack = _load_stack()
    val_df, eval_df = _load_frames()
    seq_cols = list(stack["deep_payload"]["seq_cols"])
    teacher_in_deep = [c for c in TEACHER_COLS if c in seq_cols]
    if not teacher_in_deep:
        raise RuntimeError("Deep Alpha Scout already has no teacher columns; ablation is not applicable")

    sources = _decision_sources(val_df, eval_df, stack["parent"])
    source = str(cfg["source"])
    base_val_dec = sources[source][0]
    base_eval_dec = sources[source][1]

    val_teacher_zero = _zero_teacher_for_deep(val_df, seq_cols)
    eval_teacher_zero = _zero_teacher_for_deep(eval_df, seq_cols)
    q_variants: dict[str, tuple[np.ndarray, np.ndarray]] = {
        "baseline_deep": (_predict_deep(stack, val_df), _predict_deep(stack, eval_df)),
        "teacher_zero_deep": (_predict_deep(stack, val_teacher_zero), _predict_deep(stack, eval_teacher_zero)),
        "deep_disabled": (_deep_disabled_q(len(val_df)), _deep_disabled_q(len(eval_df))),
    }

    rows: list[dict[str, Any]] = []
    for variant, (val_q, eval_q) in q_variants.items():
        for split, df, q, dec in (
            ("val", val_df, val_q, base_val_dec),
            ("oos", eval_df, eval_q, base_eval_dec),
        ):
            for cost in (1, 2, 3):
                row = _eval(df=df, q=q, dec=dec, stack=stack, cfg=cfg, period=split, cost_mult=cost, record=False)
                row["variant"] = variant
                rows.append(row)

    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    audit = {
        "candidate": CANDIDATE,
        "source": source,
        "teacher_cols_in_deep": teacher_in_deep,
        "seq_cols_count": int(len(seq_cols)),
        "val_teacher_audit": _teacher_audit(val_df, seq_cols),
        "oos_teacher_audit": _teacher_audit(eval_df, seq_cols),
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    cost3 = grid[grid["cost"].eq(3)].copy()
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "teacher_cols_in_deep": teacher_in_deep,
        "cost3": cost3.to_dict(orient="records"),
        "grid": str(GRID_OUT),
        "audit": str(AUDIT_OUT),
    }
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "audit": str(AUDIT_OUT)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
