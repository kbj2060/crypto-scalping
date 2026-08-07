#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_eval_hf_clean_regime_core_loop_20260511 import _json_default  # noqa: E402


MODEL_ID = "alpha5_31_direction_styles_20260519"
DEFAULT_IN_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_30_direction_learnable_20260519"
DEFAULT_OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha5_31_direction_styles_20260519"
STYLE_MAP = {
    0: "none",
    1: "continuation",
    2: "squeeze_breakout",
    3: "flow_supported",
}


def _num(frame: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    return pd.to_numeric(frame.get(col, default), errors="coerce").fillna(default)


def _robust_z(train: pd.Series, other: pd.Series) -> pd.Series:
    med = float(train.median())
    mad = float((train - med).abs().median())
    scale = max(mad * 1.4826, 1e-6)
    return ((other - med) / scale).clip(-6.0, 6.0)


def _signed(df: pd.DataFrame, col: str) -> pd.Series:
    y = _num(df, "direction_label", 0.0).astype(np.int64)
    s = _num(df, col, 0.0)
    return pd.Series(np.where(y == 1, s, -s), index=df.index)


def _augment(train_ref: pd.DataFrame, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    keep = _num(out, "direction_train_keep30", 0.0).astype(np.int8) == 1

    s_dirbias = _signed(out, "clean_regime4_2024_unsup_v1_directional_bias")
    s_trend = _signed(out, "clean_regime4_2024_unsup_v1_factor_trend")
    s_trendbias = _signed(out, "clean_regime4_2024_unsup_v1_trend_bias")
    s_mtf4 = _signed(out, "mtf_trend_4h")
    s_breakout = _signed(out, "breakout_strength")
    s_flow = _signed(out, "smart_money_flow")
    s_whale = _signed(out, "whale_retail_ratio")
    s_funding = _signed(out, "funding_pressure")

    sq_train = _num(train_ref, "squeeze_power", 0.0)
    sq_here = _num(out, "squeeze_power", 0.0)
    sq_z = _robust_z(sq_train, sq_here).abs()

    continuation = keep & (
        (s_dirbias >= 0.42)
        & (s_trend >= 0.12)
        & (s_trendbias >= 0.05)
        & (s_mtf4 >= 0.0)
        & (s_breakout >= 0.18)
    )
    squeeze_breakout = keep & (~continuation) & (
        (sq_z >= 1.25)
        & (s_breakout >= 0.22)
        & (s_dirbias >= 0.40)
    )
    flow_supported = keep & (~continuation) & (~squeeze_breakout) & (
        (s_whale >= 0.40)
        & (s_flow >= 0.0002)
        & (s_funding >= -0.02)
    )

    style = np.select([continuation, squeeze_breakout, flow_supported], [1, 2, 3], default=0).astype(np.int8)
    out["direction_style"] = style
    out["direction_style_name"] = np.asarray([STYLE_MAP[int(v)] for v in style], dtype=object)
    out["direction_style_keep"] = (style != 0).astype(np.int8)
    return out


def _report(frame: pd.DataFrame) -> dict[str, Any]:
    work = frame[_num(frame, "split_keep", 0.0).astype(np.int8) == 1].copy()
    keep30 = _num(work, "direction_train_keep30", 0.0).astype(np.int8) == 1
    style = _num(work, "direction_style", 0.0).astype(np.int8)
    counts = {STYLE_MAP[int(k)]: int(v) for k, v in pd.Series(style).value_counts().sort_index().to_dict().items()}
    by_style = {}
    for code, name in STYLE_MAP.items():
        if code == 0:
            continue
        grp = work[style == code]
        by_style[name] = {
            "rows": int(len(grp)),
            "share_within_keep30": float(len(grp) / max(int(np.sum(keep30)), 1)),
            "event_return_mean": float(_num(grp, "meta_event_return", 0.0).mean()) if len(grp) else 0.0,
            "quality_mean": float(_num(grp, "quality_score", 0.0).mean()) if len(grp) else 0.0,
            "long_ratio": float(np.mean(_num(grp, "direction_label", 0.0) == 1)) if len(grp) else 0.0,
        }
    return {
        "rows": int(len(work)),
        "direction_keep30_rows": int(np.sum(keep30)),
        "direction_style_counts": counts,
        "styled_share_within_keep30": float(np.mean(style[keep30] != 0)) if np.any(keep30) else 0.0,
        "by_style": by_style,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Assign semantic direction styles within alpha5_30 learnable direction subset.")
    p.add_argument("--in-dir", type=Path, default=DEFAULT_IN_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    train_ref = pd.read_parquet(args.in_dir / "alpha5_30_direction_learnable_train.parquet")
    report: dict[str, Any] = {"model_id": MODEL_ID}
    for split in ("train", "val", "oos"):
        df = pd.read_parquet(args.in_dir / f"alpha5_30_direction_learnable_{split}.parquet")
        out = _augment(train_ref, df)
        out.to_parquet(args.out_dir / f"alpha5_31_direction_styles_{split}.parquet", index=False)
        report[split] = _report(out)
    (args.out_dir / "alpha5_31_direction_styles_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    print(json.dumps({
        "stage": "alpha5_31_done",
        "report_path": str(args.out_dir / "alpha5_31_direction_styles_report.json"),
        "train_styled_share": report["train"]["styled_share_within_keep30"],
    }, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
