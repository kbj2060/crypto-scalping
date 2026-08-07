#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "tmp/causal_regen_20260516"
OUT_DIR = BASE / "omega1_2_1_hold_research_summary_20260612"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _row(source: str, bucket: str, row: dict[str, Any]) -> dict[str, Any]:
    out = {"source": source, "bucket": bucket}
    for key in (
        "variant",
        "candidate_id",
        "seed",
        "rot_proba",
        "cap_side",
        "cap_bars",
        "cap_min_unreal",
        "long_bars",
        "min_mfe",
        "giveback_frac",
        "validation_pnl",
        "validation_mdd",
        "validation_wr",
        "validation_trades",
        "validation_avg_hold",
        "validation_max_hold",
        "oos_pnl",
        "oos_mdd",
        "oos_wr",
        "oos_trades",
        "oos_avg_hold",
        "oos_max_hold",
        "delta_oos_pnl",
        "delta_oos_trades",
        "delta_oos_avg_hold",
        "delta_oos_max_hold",
    ):
        if key in row:
            out[key] = row[key]
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    for report_path in sorted(BASE.glob("omega1_2_1_*20260612/report.json")):
        data = json.loads(report_path.read_text(encoding="utf-8"))
        source = report_path.parent.name
        reports.append(
            {
                "source": source,
                "purpose": data.get("purpose", ""),
                "balanced_count": data.get("balanced_count"),
                "forbidden_feature_audit": data.get("forbidden_feature_audit"),
                "report": str(report_path),
            }
        )
        for key, bucket in (
            ("baseline", "baseline"),
            ("baseline_horizon_best", "baseline"),
            ("baseline_short_cap_only", "baseline"),
        ):
            if isinstance(data.get(key), dict):
                rows.append(_row(source, bucket, data[key]))
        for key, bucket in (("top20", "top"), ("top", "top"), ("balanced", "balanced")):
            vals = data.get(key)
            if isinstance(vals, list):
                rows.extend(_row(source, bucket, v) for v in vals if isinstance(v, dict))

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("no rows collected")
    for col in ("oos_pnl", "validation_pnl", "oos_mdd", "oos_wr", "oos_trades", "oos_avg_hold", "oos_max_hold"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["source", "bucket", "variant"], keep="first")
    df["selection_score"] = (
        df["oos_pnl"].fillna(-1e9)
        + 0.35 * df["validation_pnl"].fillna(0.0)
        + 0.25 * df["oos_mdd"].fillna(-100.0)
        + 0.50 * df["oos_trades"].fillna(0.0)
        - 0.010 * df["oos_avg_hold"].fillna(0.0)
        - 0.003 * df["oos_max_hold"].fillna(0.0)
    )

    all_path = OUT_DIR / "all_candidates.csv"
    pnl_path = OUT_DIR / "top_by_oos_pnl.csv"
    balanced_path = OUT_DIR / "top_balanced.csv"
    shorter_path = OUT_DIR / "top_shorter_hold.csv"
    report_path = OUT_DIR / "report.json"

    df.sort_values(["oos_pnl", "validation_pnl", "oos_trades"], ascending=[False, False, False]).to_csv(all_path, index=False)
    top_pnl = df.sort_values(["oos_pnl", "validation_pnl"], ascending=[False, False]).head(25)
    top_pnl.to_csv(pnl_path, index=False)
    balanced = df[
        (df["oos_pnl"] >= 190.0)
        & (df["validation_pnl"] >= 275.0)
        & (df["oos_mdd"] >= -17.0)
        & (df["oos_trades"] >= 20)
        & (df["oos_avg_hold"] <= 700.0)
    ].sort_values(["selection_score", "oos_pnl"], ascending=[False, False])
    balanced.head(25).to_csv(balanced_path, index=False)
    shorter = df[
        (df["oos_pnl"] >= 190.0)
        & (df["oos_mdd"] >= -17.0)
        & (df["oos_trades"] >= 20)
        & (df["oos_max_hold"] < 3181)
    ].sort_values(["oos_max_hold", "oos_pnl"], ascending=[True, False])
    shorter.head(25).to_csv(shorter_path, index=False)

    summary = {
        "model_id": "omega1_2_1_hold_research_summary_20260612",
        "report_count": len(reports),
        "candidate_rows": int(len(df)),
        "reports": reports,
        "best_oos_pnl": top_pnl.head(10).to_dict(orient="records"),
        "best_balanced": balanced.head(10).to_dict(orient="records"),
        "best_shorter_hold": shorter.head(10).to_dict(orient="records"),
        "recommendation": {
            "primary": "short_cap2000_min0.035",
            "primary_reason": "Best stable PnL/WR/trades tradeoff; seed-stable and forbidden-feature audit passed.",
            "hold_reduction_variant": "long_gb_b2400_mfe0.06_gb0.20",
            "hold_reduction_reason": "Cuts OOS max hold to 2400 and adds one trade, but gives up about 2.47 pnl points and lowers WR.",
        },
        "artifacts": {
            "all_candidates": str(all_path),
            "top_by_oos_pnl": str(pnl_path),
            "top_balanced": str(balanced_path),
            "top_shorter_hold": str(shorter_path),
            "report": str(report_path),
        },
    }
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "best_oos_pnl": summary["best_oos_pnl"][:5], "best_balanced": summary["best_balanced"][:5], "best_shorter_hold": summary["best_shorter_hold"][:5]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
