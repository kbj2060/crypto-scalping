#!/usr/bin/env python3
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_wr_guard_filter_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_wr_guard_filter_20260601"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"


def _guard(dec: pd.DataFrame, *, q_min: float, conf_min: float, lowconf_q_min: float, bull_q_min: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    q = pd.to_numeric(out["quality_score"], errors="raise")
    conf = pd.to_numeric(out["confidence"], errors="raise")
    expert = out["router_expert"].astype(str)
    veto = active & ((q < float(q_min)) | (conf < float(conf_min)))
    veto |= active & expert.eq("lowconf_baseline") & (q < float(lowconf_q_min))
    veto |= active & expert.eq("bull") & (q < float(bull_q_min))
    for col, value in {
        "action": 0,
        "side": 0,
        "notional_exposure": 0.0,
        "position_fraction": 0.0,
        "leverage": 1.0,
        "take_profit": 0.0,
        "stop_loss": 0.0,
        "max_hold_bars": 0,
        "cooldown_bars": 0,
    }.items():
        if col in out.columns:
            out.loc[veto, col] = value
    out["wr_guard_q_min"] = float(q_min)
    out["wr_guard_conf_min"] = float(conf_min)
    out["wr_guard_lowconf_q_min"] = float(lowconf_q_min)
    out["wr_guard_bull_q_min"] = float(bull_q_min)
    out["wr_guard_veto"] = veto.astype(int)
    return out


def _select(rows: list[dict[str, Any]], active_val_cost3_pnl: float) -> dict[str, Any]:
    floor = float(active_val_cost3_pnl) * 0.85
    feasible = [
        r for r in rows
        if float(r["validation"]["cost1"]["pnl"]) > 0.0
        and float(r["validation"]["cost2"]["pnl"]) > 0.0
        and float(r["validation"]["cost3"]["pnl"]) >= floor
        and int(r["validation"]["cost3"]["trades"]) >= 80
    ]
    pool = feasible or rows
    return sorted(
        pool,
        key=lambda r: (
            float(r["validation"]["cost3"]["wr"]),
            float(r["validation"]["cost3"]["pnl"]),
            float(r["score"]),
        ),
        reverse=True,
    )[0]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_base = pd.read_csv(ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    oos_base = pd.read_csv(ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    if len(val_df) != len(val_base) or len(eval_df) != len(oos_base):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_base)} oos {len(eval_df)} {len(oos_base)}")

    active_costs = _combo_metrics(val_df, val_base)
    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    candidates = [
        (0.00, 0.00, 0.00, 0.00),
        (0.03, 0.00, 0.04, 0.12),
        (0.05, 0.00, 0.06, 0.12),
        (0.05, 0.00, 0.08, 0.20),
        (0.07, 0.00, 0.08, 0.20),
        (0.07, 0.68, 0.08, 0.20),
        (0.10, 0.00, 0.10, 0.28),
        (0.10, 0.68, 0.10, 0.28),
        (0.05, 0.72, 0.06, 0.20),
        (0.07, 0.72, 0.08, 0.28),
        (0.00, 0.68, 0.06, 0.20),
        (0.00, 0.72, 0.08, 0.28),
    ]
    for q_min, conf_min, lowconf_q_min, bull_q_min in candidates:
        val_dec = _guard(
            val_base,
            q_min=q_min,
            conf_min=conf_min,
            lowconf_q_min=lowconf_q_min,
            bull_q_min=bull_q_min,
        )
        val_costs = _combo_metrics(val_df, val_dec)
        candidate = f"q{q_min:.2f}_c{conf_min:.2f}_lq{lowconf_q_min:.2f}_bq{bull_q_min:.2f}"
        rows.append({
            "candidate": candidate,
            "q_min": float(q_min),
            "conf_min": float(conf_min),
            "lowconf_q_min": float(lowconf_q_min),
            "bull_q_min": float(bull_q_min),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_veto_rows": int(val_dec["wr_guard_veto"].sum()),
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
        payload[candidate] = val_dec

    selected = _select(rows, float(active_costs["cost3"]["pnl"]))
    selected_val_dec = payload[str(selected["candidate"])]
    selected_oos_dec = _guard(
        oos_base,
        q_min=float(selected["q_min"]),
        conf_min=float(selected["conf_min"]),
        lowconf_q_min=float(selected["lowconf_q_min"]),
        bull_q_min=float(selected["bull_q_min"]),
    )
    selected["oos"] = _combo_metrics(eval_df, selected_oos_dec)
    selected["oos_veto_rows"] = int(selected_oos_dec["wr_guard_veto"].sum())
    selected["oos_policy_counts"] = {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()}
    rows.sort(
        key=lambda r: (
            float(r["validation"]["cost3"]["wr"]),
            float(r["validation"]["cost3"]["pnl"]),
            float(r["score"]),
        ),
        reverse=True,
    )
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "q_min": r["q_min"],
            "conf_min": r["conf_min"],
            "lowconf_q_min": r["lowconf_q_min"],
            "bull_q_min": r["bull_q_min"],
            "score": r["score"],
            "validation_veto_rows": r["validation_veto_rows"],
            **_flatten("val", r["validation"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Validation-selected WR guard on the active current-Regime3 MoE. It can veto low quality/confidence entry rows but keeps the bull/bear/chop MoE frame and existing experts.",
        "selection_rule": "Maximize validation Cost3 WR among candidates with Cost3 PnL >= 85% of active validation Cost3 PnL, Cost1/2 positive, and Cost3 trades >= 80. 2026 OOS is evaluated only after selection.",
        "active_validation": active_costs,
        "active_oos_reference": json.loads((ACTIVE_DIR / "report.json").read_text(encoding="utf-8"))["selected"]["oos"],
        "overlay": overlay,
        "selected": selected,
        "top_grid": rows[:20],
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
            "validation_decisions": str(OUT_DIR / "validation_decisions.csv"),
            "oos_decisions": str(OUT_DIR / "oos_2026_decisions.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
