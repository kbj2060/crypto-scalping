#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ZOO = ROOT / "tmp/causal_regen_20260516/zigzag_action_model_zoo_20260531"
DEFAULT_DATA_DIR = ROOT / "data/splits/year_oos"

ZIGZAG_M7_COLS = [
    "m7_zigzag_cat_fl",
    "m7_zigzag_cat_up",
    "m7_zigzag_cat_dn",
    "m7_zigzag_cat_action",
    "m7_zigzag_cat_confidence",
    "m7_zigzag_cat_side_edge",
    "m7_zigzag_cat_trade_prob",
    "m7_zigzag_xgb_fl",
    "m7_zigzag_xgb_up",
    "m7_zigzag_xgb_dn",
    "m7_zigzag_xgb_action",
    "m7_zigzag_xgb_confidence",
    "m7_zigzag_xgb_side_edge",
    "m7_zigzag_xgb_trade_prob",
]


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return str(obj)


def _read_timestamped(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in frame.columns:
        raise ValueError(f"{source} missing timestamp: {path}")
    before = len(frame)
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    if len(frame) != before:
        raise RuntimeError(f"{source} timestamp cleanup changed rows: {before}->{len(frame)}")
    return frame.replace([np.inf, -np.inf], np.nan)


def _score_path(zoo_dir: Path, model: str, train_year: int, score_year: int) -> Path:
    return zoo_dir / model / f"{model}_train{train_year}_score{score_year}.csv"


def _rename_score(score: pd.DataFrame, *, source_prefix: str, target_prefix: str) -> pd.DataFrame:
    required = [
        "timestamp",
        f"{source_prefix}_p_cash",
        f"{source_prefix}_p_long",
        f"{source_prefix}_p_short",
        f"{source_prefix}_action",
        f"{source_prefix}_confidence",
        f"{source_prefix}_side_edge",
        f"{source_prefix}_trade_prob",
    ]
    missing = [col for col in required if col not in score.columns]
    if missing:
        raise ValueError(f"{source_prefix} score missing columns: {missing}")
    out = pd.DataFrame({"timestamp": score["timestamp"]})
    out[f"{target_prefix}_fl"] = pd.to_numeric(score[f"{source_prefix}_p_cash"], errors="raise").astype("float32")
    out[f"{target_prefix}_up"] = pd.to_numeric(score[f"{source_prefix}_p_long"], errors="raise").astype("float32")
    out[f"{target_prefix}_dn"] = pd.to_numeric(score[f"{source_prefix}_p_short"], errors="raise").astype("float32")
    out[f"{target_prefix}_action"] = pd.to_numeric(score[f"{source_prefix}_action"], errors="raise").astype("int8")
    out[f"{target_prefix}_confidence"] = pd.to_numeric(score[f"{source_prefix}_confidence"], errors="raise").astype("float32")
    out[f"{target_prefix}_side_edge"] = pd.to_numeric(score[f"{source_prefix}_side_edge"], errors="raise").astype("float32")
    out[f"{target_prefix}_trade_prob"] = pd.to_numeric(score[f"{source_prefix}_trade_prob"], errors="raise").astype("float32")
    probs = out[[f"{target_prefix}_fl", f"{target_prefix}_up", f"{target_prefix}_dn"]].sum(axis=1).to_numpy(dtype=np.float64)
    if not np.allclose(probs, 1.0, atol=1e-4):
        raise RuntimeError(f"{target_prefix} probability sum guard failed: max_abs={float(np.max(np.abs(probs - 1.0)))}")
    return out


def _join_exact(left: pd.DataFrame, right: pd.DataFrame, cols: list[str], source: str) -> pd.DataFrame:
    before = len(left)
    collision = sorted(set(cols) & set(left.columns))
    if collision:
        raise RuntimeError(f"{source} would overwrite existing M7 columns: {collision}")
    out = left.merge(right[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{source} changed rows: {before}->{len(out)}")
    missing = {col: int(out[col].isna().sum()) for col in cols if int(out[col].isna().sum()) > 0}
    if missing:
        raise RuntimeError(f"{source} exact timestamp join missing values: {missing}")
    return out


def _load_meta(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _write_meta(base_meta_path: Path, out_meta_path: Path, *, rows: int, cols: int, source_files: dict[str, str]) -> None:
    meta = _load_meta(base_meta_path)
    meta["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    meta["dataset_role"] = "rl_train_with_m7_plus_zigzag_direction_candidates"
    meta["rows"] = int(rows)
    meta["cols"] = int(cols)
    meta.setdefault("sources", {})
    meta["sources"].update(source_files)
    m7 = meta.setdefault("m7", {})
    generated = list(dict.fromkeys(list(m7.get("generated_cols", [])) + ZIGZAG_M7_COLS))
    m7["generated_cols"] = generated
    m7["zigzag_direction_cols"] = ZIGZAG_M7_COLS
    m7["zigzag_direction_contract"] = {
        "label": "zigzag_action",
        "classes": {"0": "cash/fl", "1": "long/up", "2": "short/dn"},
        "cat_model": "alpha_catboost_action_master_like",
        "xgb_model": "trend_xgb_like_xgb",
        "note": "Candidate direction features. Do not feed these back into the model-zoo training jobs or teacher generation without a separate OOF/no-leak stacking contract.",
    }
    with out_meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2, default=_json_default)


def _integrate_pair(
    *,
    m7_path: Path,
    out_path: Path,
    train_year: int,
    score_year: int,
    zoo_dir: Path,
) -> dict[str, Any]:
    m7 = _read_timestamped(m7_path, f"M7 {score_year}")
    cat_score = _read_timestamped(_score_path(zoo_dir, "alpha_catboost_action_master_like", train_year, score_year), "catboost zigzag")
    xgb_score = _read_timestamped(_score_path(zoo_dir, "trend_xgb_like_xgb", train_year, score_year), "xgb zigzag")
    cat = _rename_score(
        cat_score,
        source_prefix="zigzag_alpha_catboost_action_master_like",
        target_prefix="m7_zigzag_cat",
    )
    xgb = _rename_score(
        xgb_score,
        source_prefix="zigzag_trend_xgb_like_xgb",
        target_prefix="m7_zigzag_xgb",
    )
    out = _join_exact(m7, cat, [c for c in cat.columns if c != "timestamp"], "catboost zigzag M7 integration")
    out = _join_exact(out, xgb, [c for c in xgb.columns if c != "timestamp"], "xgb zigzag M7 integration")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)
    out_meta = out_path.with_suffix(out_path.suffix + ".meta.json")
    _write_meta(
        m7_path.with_suffix(m7_path.suffix + ".meta.json"),
        out_meta,
        rows=len(out),
        cols=len(out.columns),
        source_files={
            "base_m7_path": str(m7_path),
            "catboost_zigzag_score": str(_score_path(zoo_dir, "alpha_catboost_action_master_like", train_year, score_year)),
            "xgb_zigzag_score": str(_score_path(zoo_dir, "trend_xgb_like_xgb", train_year, score_year)),
        },
    )
    return {
        "score_year": int(score_year),
        "rows": int(len(out)),
        "cols": int(len(out.columns)),
        "out_path": str(out_path),
        "meta_path": str(out_meta),
        "added_cols": ZIGZAG_M7_COLS,
        "cat_action_counts": {str(k): int(v) for k, v in out["m7_zigzag_cat_action"].value_counts().sort_index().items()},
        "xgb_action_counts": {str(k): int(v) for k, v in out["m7_zigzag_xgb_action"].value_counts().sort_index().items()},
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Integrate top ZigZag direction model outputs into M7-named feature files.")
    p.add_argument("--zoo-dir", type=Path, default=DEFAULT_ZOO)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-2025", type=Path, default=DEFAULT_DATA_DIR / "rl_training_2025_m7_zigzag_direction.csv")
    p.add_argument("--out-2026", type=Path, default=DEFAULT_DATA_DIR / "rl_training_2026_m7_zigzag_direction.csv")
    p.add_argument("--summary", type=Path, default=ROOT / "tmp/causal_regen_20260516/zigzag_m7_direction_integration_20260531/summary.json")
    args = p.parse_args()

    results = [
        _integrate_pair(
            m7_path=args.data_dir / "rl_training_2025_m7.csv",
            out_path=args.out_2025,
            train_year=2024,
            score_year=2025,
            zoo_dir=args.zoo_dir,
        ),
        _integrate_pair(
            m7_path=args.data_dir / "rl_training_2026_m7_supervised_redesign_clean.csv",
            out_path=args.out_2026,
            train_year=2025,
            score_year=2026,
            zoo_dir=args.zoo_dir,
        ),
    ]
    summary = {
        "contract": "m7_zigzag_direction_candidates",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "added_cols": ZIGZAG_M7_COLS,
        "results": results,
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
