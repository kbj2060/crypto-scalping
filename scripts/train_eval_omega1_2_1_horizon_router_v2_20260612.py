#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402
import train_eval_omega1_2_1_horizon_router_sweep_20260611 as sweep  # noqa: E402


MODEL_ID = "omega1_2_1_horizon_router_v2_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LABEL_CACHE = ROOT / "tmp/causal_regen_20260516/omega1_2_1_horizon_router_sweep_20260611"

SPECS = (
    hr.RotationSpec("rot_tp060_sl075_floor50_35", 0.60, 0.75, 0.50, 0.35),
    hr.RotationSpec("rot_tp065_sl080_floor50_35", 0.65, 0.80, 0.50, 0.35),
    hr.RotationSpec("rot_tp070_sl085_floor55_30", 0.70, 0.85, 0.55, 0.30),
    hr.RotationSpec("rot_tp075_sl090_floor60_25", 0.75, 0.90, 0.60, 0.25),
    hr.RotationSpec("rot_tp080_sl095_floor65_20", 0.80, 0.95, 0.65, 0.20),
)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _label_path(spec: hr.RotationSpec) -> Path:
    return LABEL_CACHE / f"base_labels_{spec.name}.csv"


def _labels(data: dict[str, dict[str, Any]], spec: hr.RotationSpec) -> pd.DataFrame:
    path = _label_path(spec)
    if path.exists():
        return pd.read_csv(path)
    labels = hr._build_counterfactual_labels(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        spec=spec,
        edge=0.0,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)
    return labels


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    diags: dict[str, Any] = {}

    base_row, base_ledgers = sweep._run_variant(
        data,
        variant_id=0,
        spec=SPECS[1],
        model=None,
        feature_cols=[],
        kind="none",
        seed=0,
        edge=999.0,
        hold_penalty=0.0,
        proba_min=2.0,
    )
    base_row["variant"] = "baseline_runner_only"
    base_row["rotation_spec"] = "none"
    rows.append(base_row)
    ledgers_by_id[0] = base_ledgers

    variant_id = 1
    for spec in SPECS:
        labels = sweep._relabel(_labels(data, spec), edge=0.50, hold_penalty=0.0005)
        diags[f"{spec.name}_labels"] = {
            "rows": int(len(labels)),
            "positive": int(labels["label_rotation"].sum()),
            "mean_edge": float(labels["rotation_edge"].mean()),
            "median_edge": float(labels["rotation_edge"].median()),
        }
        for seed in (260611, 260612, 260613):
            model, feature_cols, diag = hr._fit_router(labels, kind="hgb", seed=seed)
            diags[f"{spec.name}_hgb_s{seed}"] = diag
            if model is None:
                continue
            for proba in (0.60, 0.625, 0.65, 0.675, 0.70, 0.725):
                row, split_ledgers = sweep._run_variant(
                    data,
                    variant_id=variant_id,
                    spec=spec,
                    model=model,
                    feature_cols=feature_cols,
                    kind="hgb",
                    seed=seed,
                    edge=0.50,
                    hold_penalty=0.0005,
                    proba_min=float(proba),
                )
                rows.append(row)
                ledgers_by_id[variant_id] = split_ledgers
                variant_id += 1

    ranking = pd.DataFrame(rows)
    baseline = ranking[ranking["variant"].eq("baseline_runner_only")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["delta_val_pnl"] = ranking["validation_pnl"] - float(baseline["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(baseline["oos_avg_hold"])
    ranking["score"] = (
        ranking["oos_pnl"]
        + 0.45 * ranking["validation_pnl"]
        + 0.30 * ranking["oos_mdd"]
        + 0.25 * ranking["delta_oos_trades"]
        - 0.010 * ranking["oos_avg_hold"]
    )
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_router_v2_ranking.csv", index=False)

    balanced = ranking[
        (ranking["variant"] != "baseline_runner_only")
        & (ranking["oos_pnl"] > float(baseline["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]) * 0.95)
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) - 1.0)
        & (ranking["oos_trades"] >= int(baseline["oos_trades"]))
    ].copy()
    balanced.to_csv(OUT_DIR / "horizon_router_v2_balanced.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(20).tolist()] + [int(x) for x in balanced["variant_id"].head(20).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers_by_id[int(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Focused entry-time horizon-router v2 after exit-Q failures.",
        "baseline": baseline.to_dict(),
        "diagnostics": diags,
        "balanced_count": int(len(balanced)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "balanced": balanced.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_router_v2_ranking.csv"),
            "balanced": str(OUT_DIR / "horizon_router_v2_balanced.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "balanced_count": int(len(balanced)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
