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

import search_omega_live_omega44_side_specialist_ensemble_20260629 as base  # noqa: E402


MODEL_ID = "omega_live_omega44_side_specialist_constrained_ensemble_20260629"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
LEDGER_DIR = OUT_DIR / "ledgers"
SOURCE_GRID = ROOT / "tmp/causal_regen_20260516/omega_live_omega44_side_specialist_ensemble_20260629/side_specialist_validation_grid.csv"


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


def _spec_from_row(row: pd.Series) -> base.SideSpec:
    return base.SideSpec(
        variant=str(row.variant),
        side=int(row.side),
        tp=float(row.tp),
        sl=float(row.sl),
        top_frac=float(row.top_frac),
        min_edge=float(row.min_edge),
        side_margin=float(row.side_margin),
        notional=float(row.notional),
        dd_governor=bool(row.dd_governor),
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    if not SOURCE_GRID.exists():
        raise RuntimeError(f"missing source grid: {SOURCE_GRID}")

    report = base._load_base_report()
    device = base.meta.parent._device("cuda")
    split_data, align_diag, feature_cols = base._prepare_split_data(device)
    models = base._load_models(report)
    preds = base._predict_all(models, split_data)
    fee, slip = base.meta.omega._load_fee_slip()

    side_grid = pd.read_csv(SOURCE_GRID)
    long_pool_df = (
        side_grid[(side_grid["side"] == 1) & (side_grid["validation_mdd"] >= base.TARGET_MDD) & (side_grid["validation_trades"] >= 5)]
        .sort_values(["validation_pnl", "validation_score"], ascending=False)
        .head(30)
    )
    short_pool_df = (
        side_grid[(side_grid["side"] == -1) & (side_grid["validation_mdd"] >= base.TARGET_MDD) & (side_grid["validation_trades"] >= 5)]
        .sort_values(["validation_pnl", "validation_score"], ascending=False)
        .head(30)
    )
    side_specs: dict[str, base.SideSpec] = {}
    pool_readout_rows: list[dict[str, Any]] = []
    for _, row in pd.concat([long_pool_df, short_pool_df], ignore_index=True).iterrows():
        spec = _spec_from_row(row)
        side_specs[spec.variant] = spec
        pool_readout_rows.append(base._eval_side_spec(spec, preds, split_data, fee=fee, slip=slip))
    pool_readout = pd.DataFrame(pool_readout_rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    pool_readout.to_csv(OUT_DIR / "constrained_side_pool_readout.csv", index=False)

    routers = ("edge_score", "none_on_conflict", "short_priority", "long_priority")
    rows: list[dict[str, Any]] = []
    ledger_cache: dict[str, dict[str, pd.DataFrame]] = {}
    total = int(len(long_pool_df) * len(short_pool_df) * len(routers))
    idx = 0
    for _, long_row in long_pool_df.iterrows():
        long_spec = side_specs[str(long_row.variant)]
        for _, short_row in short_pool_df.iterrows():
            short_spec = side_specs[str(short_row.variant)]
            for router in routers:
                idx += 1
                spec = base.EnsembleSpec(
                    variant=f"ens_{router}__{long_spec.variant}__{short_spec.variant}",
                    long_spec=long_spec,
                    short_spec=short_spec,
                    router=router,
                )
                rec, ledgers = base._eval_ensemble_spec(spec, preds, split_data, fee=fee, slip=slip)
                rows.append(rec)
                if idx <= 1 or idx % 300 == 0:
                    print(json.dumps({"stage": "constrained_ensemble_progress", "idx": idx, "total": total, "val": rec["validation_pnl"], "val_mdd": rec["validation_mdd"], "oos": rec["oos_pnl"], "oos_mdd": rec["oos_mdd"]}, ensure_ascii=False), flush=True)
                ledger_cache[spec.variant] = ledgers

    grid = pd.DataFrame(rows).sort_values(["pass_target", "target_score", "oos_pnl", "validation_pnl"], ascending=False).reset_index(drop=True)
    grid.to_csv(OUT_DIR / "constrained_ensemble_grid.csv", index=False)
    grid[grid["pass_target"]].to_csv(OUT_DIR / "target_pass.csv", index=False)

    saved_ledgers: list[str] = []
    for variant in set(grid.head(12)["variant"].astype(str).tolist()):
        for split, ledger in ledger_cache.get(variant, {}).items():
            path = LEDGER_DIR / f"{split}_{variant[:180]}_ledger.csv"
            ledger.to_csv(path, index=False)
            saved_ledgers.append(str(path))

    out_report = {
        "model_id": MODEL_ID,
        "source_model_id": base.MODEL_ID,
        "method": "Re-run the side-specialist ensemble using only long/short specialist candidates whose validation MDD is inside the -20% target floor.",
        "alignment": align_diag,
        "feature_count": int(len(feature_cols)),
        "long_pool": long_pool_df.head(20).to_dict(orient="records"),
        "short_pool": short_pool_df.head(20).to_dict(orient="records"),
        "pool_readout_top20": pool_readout.head(20).to_dict(orient="records"),
        "ensemble_top20": grid.head(20).to_dict(orient="records"),
        "pass_count": int(grid["pass_target"].sum()),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "pool_readout": str(OUT_DIR / "constrained_side_pool_readout.csv"),
            "ensemble_grid": str(OUT_DIR / "constrained_ensemble_grid.csv"),
            "target_pass": str(OUT_DIR / "target_pass.csv"),
            "ledgers": str(LEDGER_DIR),
            "saved_ledgers": saved_ledgers,
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(out_report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "pass_count": int(out_report["pass_count"]), "top": grid.head(5).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
