#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_stability_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"


def _monthly_metrics(frame: pd.DataFrame, dec: pd.DataFrame) -> list[dict[str, Any]]:
    months = pd.to_datetime(frame["timestamp"], errors="raise").dt.to_period("M").astype(str)
    out: list[dict[str, Any]] = []
    for month in sorted(months.unique()):
        mask = months.eq(month).to_numpy()
        costs = _combo_metrics(frame.loc[mask].reset_index(drop=True), dec.loc[mask].reset_index(drop=True))
        out.append({"month": month, **costs})
    return out


def _stable_score(full_costs: dict[str, Any], months: list[dict[str, Any]]) -> float:
    base = float(_score(full_costs))
    month_scores = np.asarray([float(_score({k: m[k] for k in ("cost1", "cost2", "cost3")})) for m in months], dtype=float)
    cost3_pnls = np.asarray([float(m["cost3"]["pnl"]) for m in months], dtype=float)
    cost3_mdds = np.asarray([abs(float(m["cost3"]["mdd"])) for m in months], dtype=float)
    if month_scores.size == 0:
        return base
    negative_month_pen = float(np.maximum(0.0, -cost3_pnls).sum() * 0.10)
    dispersion_pen = float(np.std(month_scores) * 0.20)
    tail_mdd_pen = float(np.maximum(0.0, np.max(cost3_mdds) - abs(float(full_costs["cost3"]["mdd"]))) * 0.03)
    return base + float(np.min(month_scores) * 0.25) - negative_month_pen - dispersion_pen - tail_mdd_pen


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_base = pd.read_csv(VAL_DEC).reset_index(drop=True)
    oos_base = pd.read_csv(OOS_DEC).reset_index(drop=True)
    if len(val_df) != len(val_base) or len(eval_df) != len(oos_base):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_base)} oos {len(eval_df)} {len(oos_base)}")

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for bull, bear, chop in itertools.product([0.80, 0.85], [1.00, 1.15, 1.30], [1.10, 1.25]):
        val_dec = _apply_scale(val_base, bull=bull, bear=bear, chop=chop)
        val_costs = _combo_metrics(val_df, val_dec)
        val_months = _monthly_metrics(val_df, val_dec)
        key = f"bull{bull:.2f}_bear{bear:.2f}_chop{chop:.2f}"
        payload[key] = (val_dec, pd.DataFrame())
        rows.append({
            "candidate": key,
            "bull_scale": float(bull),
            "bear_scale": float(bear),
            "chop_scale": float(chop),
            "score": float(_score(val_costs)),
            "stable_score": float(_stable_score(val_costs, val_months)),
            "validation": val_costs,
            "validation_monthly": val_months,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })

    rows.sort(key=lambda r: float(r["stable_score"]), reverse=True)
    selected = rows[0]
    selected_val_dec, _ = payload[str(selected["candidate"])]
    selected_oos_dec = _apply_scale(
        oos_base,
        bull=float(selected["bull_scale"]),
        bear=float(selected["bear_scale"]),
        chop=float(selected["chop_scale"]),
    )
    selected["oos"] = _combo_metrics(eval_df, selected_oos_dec)
    selected["oos_monthly"] = _monthly_metrics(eval_df, selected_oos_dec)
    selected["oos_policy_counts"] = {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()}
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "bull_scale": r["bull_scale"],
            "bear_scale": r["bear_scale"],
            "chop_scale": r["chop_scale"],
            "score": r["score"],
            "stable_score": r["stable_score"],
            **_flatten("val", r["validation"]),
            "val_month_cost3_pnls": json.dumps([m["cost3"]["pnl"] for m in r["validation_monthly"]]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Validation-month stability selection for per-expert scale. Current-Regime3 bull/bear/chop MoE routing and expert models are unchanged; only candidate selection score changes.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bull_scale": selected["bull_scale"],
            "bear_scale": selected["bear_scale"],
            "chop_scale": selected["chop_scale"],
            "score": selected["score"],
            "stable_score": selected["stable_score"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_monthly": selected["validation_monthly"],
            "oos_monthly": selected["oos_monthly"],
            "validation_policy_counts": selected["validation_policy_counts"],
            "oos_policy_counts": selected["oos_policy_counts"],
        },
        "top_grid": rows[:12],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": report["selected"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
