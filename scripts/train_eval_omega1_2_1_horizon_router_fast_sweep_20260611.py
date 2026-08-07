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


MODEL_ID = "omega1_2_1_horizon_router_fast_sweep_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LABEL_DIR = ROOT / "tmp/causal_regen_20260516/omega1_2_1_horizon_router_sweep_20260611"

SPECS = (
    hr.RotationSpec("rot_tp060_sl080_floor50_35", 0.60, 0.80, 0.50, 0.35),
    hr.RotationSpec("rot_tp065_sl080_floor50_35", 0.65, 0.80, 0.50, 0.35),
    hr.RotationSpec("rot_tp070_sl085_floor55_30", 0.70, 0.85, 0.55, 0.30),
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


def _load_or_build_labels(data: dict[str, dict[str, Any]], spec: hr.RotationSpec) -> pd.DataFrame:
    path = LABEL_DIR / f"base_labels_{spec.name}.csv"
    if path.exists():
        return pd.read_csv(path)
    return hr._build_counterfactual_labels(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        spec=spec,
        edge=0.0,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    label_diags: dict[str, Any] = {}
    model_diags: dict[str, Any] = {}

    baseline_row, baseline_ledgers = sweep._run_variant(
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
    baseline_row["variant"] = "baseline_runner_only"
    baseline_row["rotation_spec"] = "none"
    rows.append(baseline_row)
    ledgers_by_id[0] = baseline_ledgers

    variant_id = 1
    for spec in SPECS:
        base_labels = _load_or_build_labels(data, spec)
        for hold_penalty in (0.0, 0.0005, 0.0010):
            for edge in (0.0, 0.10, 0.25, 0.50):
                labels = sweep._relabel(base_labels, edge=float(edge), hold_penalty=float(hold_penalty))
                label_key = f"{spec.name}_e{edge:g}_hp{hold_penalty:g}"
                label_diags[label_key] = {
                    "rows": int(len(labels)),
                    "positive": int(labels["label_rotation"].sum()),
                    "mean_edge": float(labels["rotation_edge"].mean()),
                    "median_edge": float(labels["rotation_edge"].median()),
                }
                for kind in ("hgb", "et"):
                    for seed in (260611, 260612):
                        model, feature_cols, diag = hr._fit_router(labels, kind=kind, seed=seed)
                        model_diags[f"{label_key}_{kind}_s{seed}"] = diag
                        if model is None:
                            continue
                        for proba_min in (0.50, 0.55, 0.60, 0.65, 0.70):
                            row, split_ledgers = sweep._run_variant(
                                data,
                                variant_id=variant_id,
                                spec=spec,
                                model=model,
                                feature_cols=feature_cols,
                                kind=kind,
                                seed=seed,
                                edge=float(edge),
                                hold_penalty=float(hold_penalty),
                                proba_min=float(proba_min),
                            )
                            rows.append(row)
                            ledgers_by_id[variant_id] = split_ledgers
                            variant_id += 1

    ranking = pd.DataFrame(rows)
    base_row = ranking.loc[ranking["variant"].eq("baseline_runner_only")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(base_row["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(base_row["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(base_row["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(base_row["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(base_row["oos_max_hold"])
    ranking["score"] = (
        ranking["oos_pnl"]
        + 0.35 * ranking["validation_pnl"]
        + 0.20 * ranking["oos_mdd"]
        + 0.35 * ranking["delta_oos_trades"]
        - 0.012 * ranking["oos_avg_hold"]
    )
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_router_fast_sweep_ranking.csv", index=False)

    promotable = ranking[
        (ranking["variant"] != "baseline_runner_only")
        & (ranking["oos_pnl"] >= float(base_row["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(base_row["validation_pnl"]) * 0.70)
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) - 2.0)
        & (ranking["oos_trades"] >= int(base_row["oos_trades"]))
    ].copy()
    promotable.to_csv(OUT_DIR / "horizon_router_fast_sweep_promotable.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(20).tolist()] + [int(x) for x in promotable["variant_id"].head(20).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers_by_id[int(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Focused horizon-router threshold/relabel sweep around the only OOS-positive signal.",
        "baseline": base_row.to_dict(),
        "label_diags": label_diags,
        "model_diags": model_diags,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_router_fast_sweep_ranking.csv"),
            "promotable": str(OUT_DIR / "horizon_router_fast_sweep_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
