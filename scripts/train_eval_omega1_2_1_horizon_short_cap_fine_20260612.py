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

import train_eval_omega1_2_1_horizon_long_cap_sweep_20260612 as sidecap  # noqa: E402
import eval_omega1_2_1_tp_runner_20260610 as runner  # noqa: E402
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402
import train_eval_omega1_2_1_horizon_router_sweep_20260611 as sweep  # noqa: E402


MODEL_ID = "omega1_2_1_horizon_short_cap_fine_20260612"
OUT_DIR = Path(__file__).resolve().parents[1] / "tmp/causal_regen_20260516" / MODEL_ID


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


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = runner._build()
    labels = sweep._relabel(
        hr._build_counterfactual_labels(
            data["validation"]["frame"],
            data["validation"]["dec"],
            data["validation"]["state"],
            fee=float(data["validation"]["fee"]),
            slip=float(data["validation"]["slip"]),
            cost_mult=3.0,
            spec=sidecap.ROT_SPEC,
            edge=0.0,
        ),
        edge=0.50,
        hold_penalty=0.0005,
    )
    rot_model, rot_cols, diag = hr._fit_router(labels, kind="hgb", seed=260611)
    if rot_model is None:
        raise RuntimeError(f"rotation router fit failed: {diag}")

    configs: list[dict[str, Any]] = [{"variant": "horizon_best_no_cap", "cap_bars": 0, "cap_min_unreal": 0.0}]
    for cap_bars in (1760, 1850, 1920, 2000, 2048, 2120, 2200, 2280):
        for min_unreal in (0.025, 0.030, 0.035, 0.040, 0.045, 0.050):
            configs.append({"variant": f"short_cap{cap_bars}_min{min_unreal:.3f}", "cap_bars": int(cap_bars), "cap_min_unreal": float(min_unreal)})

    rows: list[dict[str, Any]] = []
    ledgers_by_id: dict[int, dict[str, pd.DataFrame]] = {}
    for candidate_id, cfg in enumerate(configs):
        row: dict[str, Any] = {"candidate_id": int(candidate_id), "rot_proba": 0.65, "cap_side": -1, **cfg}
        split_ledgers: dict[str, pd.DataFrame] = {}
        for split in ("validation", "oos"):
            payload = data[split]
            metrics, ledger = sidecap._simulate(
                payload["frame"],
                payload["dec"],
                payload["state"],
                fee=float(payload["fee"]),
                slip=float(payload["slip"]),
                cost_mult=3.0,
                rot_model=rot_model,
                rot_cols=rot_cols,
                rot_proba=0.65,
                cap_bars=int(cfg["cap_bars"]),
                cap_min_unreal=float(cfg["cap_min_unreal"]),
                cap_side=-1,
            )
            row.update(sidecap._row(split, metrics))
            split_ledgers[split] = ledger
        rows.append(row)
        ledgers_by_id[candidate_id] = split_ledgers

    ranking = pd.DataFrame(rows)
    baseline = ranking.loc[ranking["variant"].eq("horizon_best_no_cap")].iloc[0]
    ranking["delta_oos_pnl"] = ranking["oos_pnl"] - float(baseline["oos_pnl"])
    ranking["delta_validation_pnl"] = ranking["validation_pnl"] - float(baseline["validation_pnl"])
    ranking["delta_oos_trades"] = ranking["oos_trades"] - int(baseline["oos_trades"])
    ranking["delta_oos_avg_hold"] = ranking["oos_avg_hold"] - float(baseline["oos_avg_hold"])
    ranking["delta_oos_max_hold"] = ranking["oos_max_hold"] - int(baseline["oos_max_hold"])
    ranking["score"] = ranking["oos_pnl"] + 0.55 * ranking["validation_pnl"] + 0.35 * ranking["oos_mdd"] + 0.35 * ranking["delta_oos_trades"] - 0.020 * ranking["oos_avg_hold"] - 0.008 * ranking["oos_max_hold"]
    ranking = ranking.sort_values(["oos_pnl", "validation_pnl", "oos_trades", "oos_mdd"], ascending=[False, False, False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "horizon_short_cap_fine_ranking.csv", index=False)

    balanced = ranking[
        (ranking["oos_pnl"] >= float(baseline["oos_pnl"]))
        & (ranking["validation_pnl"] >= float(baseline["validation_pnl"]))
        & (ranking["oos_mdd"] >= float(baseline["oos_mdd"]) - 1.0)
        & (ranking["oos_trades"] >= int(baseline["oos_trades"]))
        & (ranking["oos_avg_hold"] < float(baseline["oos_avg_hold"]))
    ].copy()
    balanced.to_csv(OUT_DIR / "horizon_short_cap_fine_balanced.csv", index=False)

    for sid in sorted(set([0] + [int(x) for x in ranking["candidate_id"].head(16).tolist()] + [int(x) for x in balanced["candidate_id"].head(16).tolist()])):
        for split, ledger in ledgers_by_id[sid].items():
            ledger.to_csv(OUT_DIR / f"{split}_candidate{sid}_ledger.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "purpose": "Fine sweep around profitable short static cap after current best horizon-router.",
        "rot_diag": diag,
        "baseline_horizon_best": baseline.to_dict(),
        "balanced_count": int(len(balanced)),
        "top20": ranking.head(20).to_dict(orient="records"),
        "balanced": balanced.head(20).to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "horizon_short_cap_fine_ranking.csv"),
            "balanced": str(OUT_DIR / "horizon_short_cap_fine_balanced.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "balanced_count": int(len(balanced)), "top": ranking.head(10).to_dict(orient="records"), "balanced": balanced.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
