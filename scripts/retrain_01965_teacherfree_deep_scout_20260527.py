#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import eval_alpha3_regime4_state24_v2_full_retrain_20260526 as alpha3_full  # noqa: E402
from scripts import eval_hf_v13_frozen_v27_rule_exit_overlay_v31 as v31  # noqa: E402
from scripts import loop_alpha3_1_alpha6_alpha7_combo_search_until_0800_20260527 as combo  # noqa: E402
from scripts import train_eval_hf_v13_deep_alpha_candidate_expansion_v27 as v27  # noqa: E402
from scripts.precision_retest_01965_alpha7_combo_20260527 import CANDIDATE, _cfg_from_results, _eval  # noqa: E402
from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default, _read  # noqa: E402


MODEL_ID = "retrain_01965_teacherfree_deep_scout_20260527"
OUT_DIR = ROOT / f"tmp/causal_regen_20260516/{MODEL_ID}"
SUMMARY_OUT = OUT_DIR / "summary.json"
GRID_OUT = OUT_DIR / "grid.csv"
SCOUT_OUT = OUT_DIR / "deep_scout_teacherfree.pt"
AUDIT_OUT = OUT_DIR / "audit.json"
TEACHER_PREFIX = "teacher_"


def _load_train_val_eval_frames() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_all = combo._merge_state24(_read(v31.DEFAULT_TRAIN), alpha3_full.SIDE_CLEAN4_2025)
    eval_df = combo._merge_state24(_read(v31.DEFAULT_EVAL), alpha3_full.SIDE_CLEAN4_2026)
    a7_train = combo._rename_clean4_v2(_read(combo.PRIMARY_TRAIN_CSV))
    a7_eval = combo._rename_clean4_v2(_read(combo.PRIMARY_EVAL_CSV))
    train_all = combo._augment_with_alpha7_features(train_all, a7_train)
    eval_df = combo._augment_with_alpha7_features(eval_df, a7_eval)
    ts = pd.to_datetime(train_all["timestamp"], errors="coerce")
    train_df = train_all[ts < pd.Timestamp("2025-10-01")].reset_index(drop=True)
    val_df = train_all[ts >= pd.Timestamp("2025-10-01")].reset_index(drop=True)
    return train_df, val_df, eval_df.reset_index(drop=True)


def _teacherfree_seq_cols(base_cols: list[str]) -> tuple[list[str], list[str]]:
    removed = [c for c in base_cols if str(c).startswith(TEACHER_PREFIX)]
    kept = [c for c in base_cols if not str(c).startswith(TEACHER_PREFIX)]
    if not removed:
        raise RuntimeError("current deep scout has no teacher columns to remove")
    if len(kept) >= len(base_cols):
        raise RuntimeError("teacher-free seq contract did not shrink")
    return kept, removed


def _assert_seq_contract(df: pd.DataFrame, seq_cols: list[str], *, name: str) -> None:
    missing = [c for c in seq_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"{name}: missing teacher-free deep scout columns: {missing[:20]}")
    forbidden = [c for c in seq_cols if str(c).startswith(TEACHER_PREFIX)]
    if forbidden:
        raise RuntimeError(f"{name}: teacher columns still present in seq contract: {forbidden}")


def _predict_deep(model: v27.DeepAlphaTCN, df: pd.DataFrame, seq_cols: list[str], norm: dict[str, np.ndarray]) -> np.ndarray:
    _assert_seq_contract(df, seq_cols, name="predict_deep")
    return v27._predict_all(model, df, seq_cols, norm)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.random.seed(20260527)
    torch.manual_seed(20260527)
    cfg = _cfg_from_results()
    stack = combo._load_stack()
    train_df, val_df, eval_df = _load_train_val_eval_frames()
    base_seq_cols = list(stack["deep_payload"]["seq_cols"])
    seq_cols, removed = _teacherfree_seq_cols(base_seq_cols)
    for name, df in (("train", train_df), ("val", val_df), ("oos", eval_df)):
        _assert_seq_contract(df, seq_cols, name=name)

    print(json.dumps({"stage": "build_teacherfree_deep_train_set", "seq_cols": len(seq_cols), "removed": removed}, ensure_ascii=False), flush=True)
    train_ds = v27._build_train_set(train_df, seq_cols, fee=float(stack["fee"]), slip=float(stack["slip"]), stride=3)
    norm = v27._normalizer(train_ds["seq"])
    print(json.dumps({"stage": "train_teacherfree_deep_scout", "epochs": 25, "samples": int(len(train_ds["target"]))}, ensure_ascii=False), flush=True)
    model = v27._train_model(train_ds, norm, epochs=25)
    torch.save(
        {
            "model_id": MODEL_ID,
            "state_dict": model.state_dict(),
            "seq_cols": seq_cols,
            "norm": norm,
            "removed_teacher_cols": removed,
            "base_scout": str(stack["deep_payload"].get("model_id", "unknown")),
        },
        SCOUT_OUT,
    )

    print(json.dumps({"stage": "predict_q"}, ensure_ascii=False), flush=True)
    baseline_val_q = v27._predict_all(stack["deep_model"], val_df, base_seq_cols, stack["deep_payload"]["norm"])
    baseline_eval_q = v27._predict_all(stack["deep_model"], eval_df, base_seq_cols, stack["deep_payload"]["norm"])
    teacherfree_val_q = _predict_deep(model, val_df, seq_cols, norm)
    teacherfree_eval_q = _predict_deep(model, eval_df, seq_cols, norm)
    disabled_val_q = np.full((len(val_df), 2), -1.0, dtype=np.float32)
    disabled_eval_q = np.full((len(eval_df), 2), -1.0, dtype=np.float32)

    print(json.dumps({"stage": "evaluate_01965"}, ensure_ascii=False), flush=True)
    sources = combo._decision_sources(val_df, eval_df, stack["parent"])
    source = str(cfg["source"])
    rows: list[dict[str, Any]] = []
    q_variants = {
        "baseline_deep": (baseline_val_q, baseline_eval_q),
        "teacherfree_retrained_deep": (teacherfree_val_q, teacherfree_eval_q),
        "deep_disabled": (disabled_val_q, disabled_eval_q),
    }
    for variant, (val_q, eval_q) in q_variants.items():
        for split, df, q, side in (("val", val_df, val_q, 0), ("oos", eval_df, eval_q, 1)):
            dec = sources[source][side]
            for cost in (1, 2, 3):
                row = _eval(df=df, q=q, dec=dec, stack=stack, cfg=cfg, period=split, cost_mult=cost, record=False)
                row["variant"] = variant
                rows.append(row)
    grid = pd.DataFrame(rows)
    grid.to_csv(GRID_OUT, index=False)
    cost3 = grid[grid["cost"].eq(3)].copy()
    summary = {
        "model_id": MODEL_ID,
        "candidate": CANDIDATE,
        "removed_teacher_cols": removed,
        "base_seq_col_count": int(len(base_seq_cols)),
        "teacherfree_seq_col_count": int(len(seq_cols)),
        "train_samples": int(len(train_ds["target"])),
        "scout": str(SCOUT_OUT),
        "cost3": cost3.to_dict(orient="records"),
        "grid": str(GRID_OUT),
        "audit": str(AUDIT_OUT),
    }
    audit = {
        "selection_uses_2026": False,
        "train_window": [str(train_df["timestamp"].iloc[0]), str(train_df["timestamp"].iloc[-1])],
        "val_window": [str(val_df["timestamp"].iloc[0]), str(val_df["timestamp"].iloc[-1])],
        "oos_window": [str(eval_df["timestamp"].iloc[0]), str(eval_df["timestamp"].iloc[-1])],
        "base_seq_cols": base_seq_cols,
        "teacherfree_seq_cols": seq_cols,
        "removed_teacher_cols": removed,
    }
    AUDIT_OUT.write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    SUMMARY_OUT.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(SUMMARY_OUT), "grid": str(GRID_OUT), "scout": str(SCOUT_OUT)}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
