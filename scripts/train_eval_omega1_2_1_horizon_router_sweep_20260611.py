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


MODEL_ID = "omega1_2_1_horizon_router_sweep_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID

SPECS = (
    hr.RotationSpec("rot_tp060_sl080_floor50_35", 0.60, 0.80, 0.50, 0.35),
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


def _relabel(labels: pd.DataFrame, *, edge: float, hold_penalty: float) -> pd.DataFrame:
    out = labels.copy()
    runner_utility = pd.to_numeric(out["runner_ret"], errors="raise") - float(hold_penalty) * pd.to_numeric(out["runner_hold"], errors="raise")
    rotation_utility = pd.to_numeric(out["rotation_ret"], errors="raise") - float(hold_penalty) * pd.to_numeric(out["rotation_hold"], errors="raise")
    out["rotation_edge"] = rotation_utility - runner_utility
    out["label_rotation"] = (out["rotation_edge"] > float(edge)).astype(int)
    return out


def _run_variant(
    data: dict[str, dict[str, Any]],
    *,
    variant_id: int,
    spec: hr.RotationSpec,
    model: Any,
    feature_cols: list[str],
    kind: str,
    seed: int,
    edge: float,
    hold_penalty: float,
    proba_min: float,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    row: dict[str, Any] = {
        "variant_id": int(variant_id),
        "variant": f"{spec.name}_{kind}_s{seed}_e{edge:g}_hp{hold_penalty:g}_p{str(proba_min).replace('.', '')}",
        "rotation_spec": spec.name,
        "model_kind": kind,
        "seed": int(seed),
        "label_edge": float(edge),
        "hold_penalty": float(hold_penalty),
        "proba_min": float(proba_min),
    }
    ledgers: dict[str, pd.DataFrame] = {}
    for split in ("validation", "oos"):
        metrics, ledger = hr._simulate_router(
            data[split]["frame"],
            data[split]["dec"],
            data[split]["state"],
            fee=float(data[split]["fee"]),
            slip=float(data[split]["slip"]),
            cost_mult=3.0,
            spec=spec,
            model=model,
            feature_cols=feature_cols,
            proba_min=float(proba_min),
        )
        row.update(hr._row(split, metrics))
        ledgers[split] = ledger
    return row, ledgers


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    label_diags: dict[str, Any] = {}
    model_diags: dict[str, Any] = {}

    baseline_row, baseline_ledgers = _run_variant(
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

    base_labels_by_spec: dict[str, pd.DataFrame] = {}
    for spec in SPECS:
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
        labels.to_csv(OUT_DIR / f"base_labels_{spec.name}.csv", index=False)
        base_labels_by_spec[spec.name] = labels
        label_diags[f"{spec.name}_base"] = {
            "rows": int(len(labels)),
            "mean_edge": float(labels["rotation_edge"].mean()),
            "median_edge": float(labels["rotation_edge"].median()),
            "positive_edge_gt0": int((labels["rotation_edge"] > 0).sum()),
        }

    variant_id = 1
    for spec in SPECS:
        base_labels = base_labels_by_spec[spec.name]
        for hold_penalty in (0.0, 0.0005, 0.0010, 0.0020):
            for edge in (-0.25, 0.0, 0.10, 0.25, 0.50, 1.00):
                labels = _relabel(base_labels, edge=float(edge), hold_penalty=float(hold_penalty))
                key = f"{spec.name}_e{edge:g}_hp{hold_penalty:g}"
                label_diags[key] = {
                    "rows": int(len(labels)),
                    "positive": int(labels["label_rotation"].sum()),
                    "mean_edge": float(labels["rotation_edge"].mean()),
                    "median_edge": float(labels["rotation_edge"].median()),
                }
                for kind in ("hgb", "et"):
                    for seed in (260611, 260612):
                        model, feature_cols, diag = hr._fit_router(labels, kind=kind, seed=seed)
                        model_diags[f"{key}_{kind}_s{seed}"] = diag
                        if model is None:
                            continue
                        for proba_min in (0.35, 0.45, 0.55, 0.60, 0.65, 0.70):
                            row, split_ledgers = _run_variant(
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
        + 0.20 * ranking["delta_oos_trades"]
        - 0.012 * ranking["oos_avg_hold"]
        - 0.004 * ranking["oos_max_hold"]
    )
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_router_sweep_ranking.csv", index=False)

    promotable = ranking[
        (ranking["variant"] != "baseline_runner_only")
        & (ranking["oos_pnl"] >= float(base_row["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(base_row["validation_pnl"]) * 0.70)
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) - 2.0)
        & (ranking["oos_trades"] >= int(base_row["oos_trades"]))
    ].copy()
    promotable.to_csv(OUT_DIR / "horizon_router_sweep_promotable.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(20).tolist()] + [int(x) for x in promotable["variant_id"].head(20).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers_by_id[int(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Sweep horizon-router relabeling and thresholds to increase trade count without sacrificing PnL.",
        "baseline": base_row.to_dict(),
        "label_diags": label_diags,
        "model_diags": model_diags,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_router_sweep_ranking.csv"),
            "promotable": str(OUT_DIR / "horizon_router_sweep_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
