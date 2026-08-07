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


MODEL_ID = "omega1_2_1_horizon_router_micro_sweep_20260611"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LABEL_PATH = ROOT / "tmp/causal_regen_20260516/omega1_2_1_horizon_router_sweep_20260611/base_labels_rot_tp065_sl080_floor50_35.csv"
SPEC = hr.RotationSpec("rot_tp065_sl080_floor50_35", 0.65, 0.80, 0.50, 0.35)


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


def _labels(data: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if LABEL_PATH.exists():
        return pd.read_csv(LABEL_PATH)
    return hr._build_counterfactual_labels(
        data["validation"]["frame"],
        data["validation"]["dec"],
        data["validation"]["state"],
        fee=float(data["validation"]["fee"]),
        slip=float(data["validation"]["slip"]),
        cost_mult=3.0,
        spec=SPEC,
        edge=0.0,
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    base_labels = _labels(data)
    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    diags: dict[str, Any] = {}

    baseline_row, baseline_ledgers = sweep._run_variant(
        data,
        variant_id=0,
        spec=SPEC,
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
    for hold_penalty in (0.0, 0.0005):
        for edge in (-0.10, 0.0, 0.10, 0.25, 0.50):
            labels = sweep._relabel(base_labels, edge=float(edge), hold_penalty=float(hold_penalty))
            for seed in (260611, 260612, 260613):
                model, feature_cols, diag = hr._fit_router(labels, kind="hgb", seed=seed)
                diags[f"e{edge:g}_hp{hold_penalty:g}_s{seed}"] = diag
                if model is None:
                    continue
                for proba_min in (0.55, 0.60, 0.625, 0.65, 0.675, 0.70):
                    row, split_ledgers = sweep._run_variant(
                        data,
                        variant_id=variant_id,
                        spec=SPEC,
                        model=model,
                        feature_cols=feature_cols,
                        kind="hgb",
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
    ranking["score"] = ranking["oos_pnl"] + 0.35 * ranking["validation_pnl"] + 0.25 * ranking["delta_oos_trades"] - 0.012 * ranking["oos_avg_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_router_micro_sweep_ranking.csv", index=False)

    promotable = ranking[
        (ranking["variant"] != "baseline_runner_only")
        & (ranking["oos_pnl"] >= float(base_row["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(base_row["validation_pnl"]) * 0.70)
        & (ranking["oos_mdd"] >= float(base_row["oos_mdd"]) - 2.0)
        & (ranking["oos_trades"] >= int(base_row["oos_trades"]))
    ].copy()
    promotable.to_csv(OUT_DIR / "horizon_router_micro_sweep_promotable.csv", index=False)

    save_ids = sorted(set([0] + [int(x) for x in ranking["variant_id"].head(20).tolist()] + [int(x) for x in promotable["variant_id"].head(20).tolist()]))
    for sid in save_ids:
        for split, ledger in ledgers_by_id[int(sid)].items():
            ledger.to_csv(OUT_DIR / f"{split}_variant{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Micro sweep for rot_tp065/sl080 horizon router to increase trades and PnL.",
        "baseline": base_row.to_dict(),
        "diagnostics": diags,
        "promotable_count": int(len(promotable)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "promotable": promotable.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_router_micro_sweep_ranking.csv"),
            "promotable": str(OUT_DIR / "horizon_router_micro_sweep_promotable.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "promotable_count": int(len(promotable)), "top10": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
