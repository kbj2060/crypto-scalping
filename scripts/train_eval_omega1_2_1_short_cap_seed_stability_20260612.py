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
import train_eval_omega1_2_1_exit_only_rl_editor_20260610 as base  # noqa: E402
import train_eval_omega1_2_1_horizon_long_cap_sweep_20260612 as sidecap  # noqa: E402
import train_eval_omega1_2_1_horizon_router_20260611 as hr  # noqa: E402
import train_eval_omega1_2_1_horizon_router_sweep_20260611 as sweep  # noqa: E402


MODEL_ID = "omega1_2_1_short_cap_seed_stability_20260612"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
ROT_SPEC = sidecap.ROT_SPEC
CAP_BARS = 2000
CAP_MIN_UNREAL = 0.035
ROT_PROBA = 0.65


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


def _row(prefix: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        f"{prefix}_pnl": float(metrics["pnl"]),
        f"{prefix}_mdd": float(metrics["mdd"]),
        f"{prefix}_wr": float(metrics["wr"]),
        f"{prefix}_trades": int(metrics["trades"]),
        f"{prefix}_long": int(metrics["long_entries"]),
        f"{prefix}_short": int(metrics["short_entries"]),
        f"{prefix}_avg_hold": float(metrics["avg_hold_bars"]),
        f"{prefix}_median_hold": float(metrics["median_hold_bars"]),
        f"{prefix}_max_hold": int(metrics["max_hold_bars"]),
        f"{prefix}_route_counts": metrics["route_counts"],
        f"{prefix}_reasons": metrics["exit_reasons"],
    }


def _ledger_audit(ledger: pd.DataFrame) -> dict[str, Any]:
    if ledger.empty:
        return {"rows": 0, "bad_hold": 0, "bad_side": 0, "duplicate_entries": 0}
    entry_col = "entry_timestamp" if "entry_timestamp" in ledger.columns else "entry_time"
    hold = pd.to_numeric(ledger["hold_bars"], errors="coerce") if "hold_bars" in ledger.columns else pd.Series([], dtype=float)
    if "side" in ledger.columns:
        side = ledger["side"].astype(str).str.upper()
        bad_side = int((~side.isin(["LONG", "SHORT", "1", "-1", "1.0", "-1.0"])).sum())
    else:
        bad_side = 0
    return {
        "rows": int(len(ledger)),
        "bad_hold": int(hold.lt(0).sum()) if len(hold) else 0,
        "bad_side": bad_side,
        "duplicate_entries": int(ledger[entry_col].duplicated().sum()) if entry_col in ledger.columns else 0,
    }


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
            spec=ROT_SPEC,
            edge=0.0,
        ),
        edge=0.50,
        hold_penalty=0.0005,
    )
    feature_cols = hr._feature_cols(labels)
    base._reject_forbidden(feature_cols, "short_cap_seed_stability_router")

    rows: list[dict[str, Any]] = []
    ledgers: dict[str, pd.DataFrame] = {}
    diags: dict[str, Any] = {}
    for seed in (260601, 260602, 260603, 260611, 260612, 260613, 260621, 260622, 260623):
        rot_model, rot_cols, diag = hr._fit_router(labels, kind="hgb", seed=seed)
        diags[f"hgb_s{seed}"] = diag
        base._reject_forbidden(rot_cols, f"short_cap_seed_stability_router_s{seed}")
        if rot_model is None:
            continue
        row: dict[str, Any] = {
            "variant": f"short_cap{CAP_BARS}_min{CAP_MIN_UNREAL:.3f}_hgb_s{seed}",
            "seed": int(seed),
            "rot_proba": float(ROT_PROBA),
            "cap_side": -1,
            "cap_bars": int(CAP_BARS),
            "cap_min_unreal": float(CAP_MIN_UNREAL),
        }
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
                rot_proba=float(ROT_PROBA),
                cap_bars=int(CAP_BARS),
                cap_min_unreal=float(CAP_MIN_UNREAL),
                cap_side=-1,
            )
            row.update(_row(split, metrics))
            row[f"{split}_ledger_audit"] = _ledger_audit(ledger)
            ledgers[f"{split}_s{seed}"] = ledger
        rows.append(row)

    ranking = pd.DataFrame(rows).sort_values(["oos_pnl", "validation_pnl"], ascending=[False, False]).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "short_cap_seed_stability_ranking.csv", index=False)
    for key, ledger in ledgers.items():
        ledger.to_csv(OUT_DIR / f"{key}_ledger.csv", index=False)

    metric_cols = ["validation_pnl", "validation_mdd", "validation_wr", "validation_trades", "validation_avg_hold", "oos_pnl", "oos_mdd", "oos_wr", "oos_trades", "oos_avg_hold"]
    summary = {
        "model_id": MODEL_ID,
        "purpose": "Seed stability and simple ledger/forbidden-feature audit for the current short-cap horizon-router candidate.",
        "feature_count": int(len(feature_cols)),
        "forbidden_feature_audit": "pass",
        "diagnostics": diags,
        "top": ranking.head(20).to_dict(orient="records"),
        "metric_min": {c: float(ranking[c].min()) for c in metric_cols if c in ranking},
        "metric_max": {c: float(ranking[c].max()) for c in metric_cols if c in ranking},
        "metric_mean": {c: float(ranking[c].mean()) for c in metric_cols if c in ranking},
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "short_cap_seed_stability_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "top": ranking.head(10).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
