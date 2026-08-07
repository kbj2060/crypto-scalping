#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

import eval_omega1_regime3_expertdq_risk_replay_20260602 as base


ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "omega1_regime3_expertdq_tabm_risk_replay_20260602"
EXPERTDQ_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_tabm_20260602"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_expertdq_tabm_risk_replay_20260602"


def _expertdq_paths(variant: str) -> tuple[Path, Path]:
    vdir = EXPERTDQ_DIR / variant
    return (
        vdir / f"training_features_2025_{variant}_omega1_regime3_expertdq_oof_20260602.csv",
        vdir / f"training_features_2026_rebuilt_{variant}_omega1_regime3_expertdq_20260602.csv",
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = base._load_frames_max()
    val_df = train_all[train_all["timestamp"] >= base.SPLIT_TS].reset_index(drop=True)

    active_val_dec = base._load_csv(base.ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    active_oos_dec = base._load_csv(base.ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    if len(active_val_dec) != len(val_df) or len(active_oos_dec) != len(eval_df):
        raise RuntimeError(
            f"active decision length mismatch: val {len(active_val_dec)} vs {len(val_df)}, "
            f"oos {len(active_oos_dec)} vs {len(eval_df)}"
        )

    active_report = json.loads((base.ACTIVE_DIR / "report.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for variant_dir in sorted(p for p in EXPERTDQ_DIR.iterdir() if p.is_dir()):
        variant = variant_dir.name
        val_path, oos_path = _expertdq_paths(variant)
        if not val_path.exists() or not oos_path.exists():
            continue
        val_src = base._load_csv(val_path)
        oos_src = base._load_csv(oos_path)
        base._require_unique_timestamps(val_src, f"{variant} val")
        base._require_unique_timestamps(oos_src, f"{variant} oos")

        val_frame_common, active_val_common, val_src_common = base._align_source_to_frame(val_df, active_val_dec, val_src)
        oos_frame_common, active_oos_common, oos_src_common = base._align_source_to_frame(eval_df, active_oos_dec, oos_src)
        val_dec = base._to_decisions(val_src_common, oof=True)
        oos_dec = base._to_decisions(oos_src_common, oof=False)
        if len(val_dec) != len(val_frame_common) or len(oos_dec) != len(oos_frame_common):
            raise RuntimeError(f"{variant}: common frame/decision length mismatch")

        val_costs = base._combo_metrics(val_frame_common, val_dec)
        oos_costs = base._combo_metrics(oos_frame_common, oos_dec)
        active_val_costs = base._combo_metrics(val_frame_common, active_val_common)
        active_oos_costs = base._combo_metrics(oos_frame_common, active_oos_common)

        report = {
            "variant": variant,
            "validation_rows_common": int(len(val_frame_common)),
            "oos_rows_common": int(len(oos_frame_common)),
            "validation": val_costs,
            "oos": oos_costs,
            "active_validation_common": active_val_costs,
            "active_oos_common": active_oos_costs,
            "delta_vs_active_common": {
                f"cost{mult}": base._compact_delta(oos_costs[f"cost{mult}"], active_oos_costs[f"cost{mult}"])
                for mult in (1, 2, 3)
            },
            "policy_counts": {
                "validation": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                "oos": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
            },
        }
        reports[variant] = report
        row = {
            "variant": variant,
            "validation_rows_common": int(len(val_frame_common)),
            "oos_rows_common": int(len(oos_frame_common)),
        }
        for period, costs in [("val", val_costs), ("oos", oos_costs), ("active_val_common", active_val_costs), ("active_oos_common", active_oos_costs)]:
            for mult in (1, 2, 3):
                c = costs[f"cost{mult}"]
                row[f"{period}_cost{mult}_pnl"] = float(c["pnl"])
                row[f"{period}_cost{mult}_mdd"] = float(c["mdd"])
                row[f"{period}_cost{mult}_trades"] = int(c["trades"])
                row[f"{period}_cost{mult}_wr"] = float(c["wr"])
        for mult in (1, 2, 3):
            delta = report["delta_vs_active_common"][f"cost{mult}"]
            row[f"delta_oos_cost{mult}_pnl"] = float(delta["pnl"])
            row[f"delta_oos_cost{mult}_mdd"] = float(delta["mdd"])
            row[f"delta_oos_cost{mult}_trades"] = float(delta["trades"])
            row[f"delta_oos_cost{mult}_wr"] = float(delta["wr"])
        rows.append(row)

    if not rows:
        raise RuntimeError("no TabM expert-DQ variants found")

    ranking = pd.DataFrame(rows).sort_values("oos_cost3_pnl", ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected_variant = str(ranking.iloc[0]["variant"])
    report = {
        "model_id": MODEL_ID,
        "design": "Replay TabM expert-local Regime3 Direction+Quality final_action through the unchanged Omega1 ZigZag risk template and expert scales. Regime router, risk template, and Cost accounting are unchanged from the CatBoost baseline.",
        "active_reference_dir": str(base.ACTIVE_DIR),
        "expert_dq_dir": str(EXPERTDQ_DIR),
        "risk_template": base.ACTIVE_TEMPLATE,
        "expert_scales": base.ACTIVE_SCALES,
        "overlay": overlay,
        "active_full_period_reference": active_report.get("selected", {}),
        "selected_by_oos_cost3_pnl": selected_variant,
        "variants": reports,
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=base._json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected_variant, "top": ranking.head(8).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=base._json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
