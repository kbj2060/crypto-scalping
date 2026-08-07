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


MODEL_ID = "alpha7_regime3_current_moe_active_scaled_exit_shape_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_exit_shape_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"


def _apply_exit_shape(
    dec: pd.DataFrame,
    *,
    bear_tp: float,
    chop_tp: float,
    chop_sl: float,
    chop_hold: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    bear = active & out["router_expert"].astype(str).eq("bear")
    chop = active & out["router_expert"].astype(str).eq("chop_expert")
    out.loc[bear, "take_profit"] = pd.to_numeric(out.loc[bear, "take_profit"], errors="raise") * float(bear_tp)
    out.loc[chop, "take_profit"] = pd.to_numeric(out.loc[chop, "take_profit"], errors="raise") * float(chop_tp)
    out.loc[chop, "stop_loss"] = pd.to_numeric(out.loc[chop, "stop_loss"], errors="raise") * float(chop_sl)
    hold = pd.to_numeric(out.loc[chop, "max_hold_bars"], errors="raise")
    out.loc[chop, "max_hold_bars"] = hold.mul(float(chop_hold)).round().clip(lower=1).astype(int)
    out["exit_shape_bear_tp"] = float(bear_tp)
    out["exit_shape_chop_tp"] = float(chop_tp)
    out["exit_shape_chop_sl"] = float(chop_sl)
    out["exit_shape_chop_hold"] = float(chop_hold)
    return out


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
    # Keep the search deliberately low-dimensional: prior experiments showed
    # bear risk context and chop scaling drive most of the edge.
    for bear_tp, chop_tp, chop_sl, chop_hold in itertools.product(
        [1.0, 1.10],
        [0.90, 1.0, 1.10],
        [0.85, 1.0],
        [0.75, 1.0],
    ):
        val_dec = _apply_exit_shape(val_base, bear_tp=bear_tp, chop_tp=chop_tp, chop_sl=chop_sl, chop_hold=chop_hold)
        oos_dec = _apply_exit_shape(oos_base, bear_tp=bear_tp, chop_tp=chop_tp, chop_sl=chop_sl, chop_hold=chop_hold)
        val_costs = _combo_metrics(val_df, val_dec)
        oos_costs = _combo_metrics(eval_df, oos_dec)
        key = f"btp{bear_tp:.2f}_ctp{chop_tp:.2f}_csl{chop_sl:.2f}_ch{chop_hold:.2f}"
        payload[key] = (val_dec, oos_dec)
        rows.append({
            "candidate": key,
            "bear_tp": float(bear_tp),
            "chop_tp": float(chop_tp),
            "chop_sl": float(chop_sl),
            "chop_hold": float(chop_hold),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "oos": oos_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
        })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "bear_tp": r["bear_tp"],
            "chop_tp": r["chop_tp"],
            "chop_sl": r["chop_sl"],
            "chop_hold": r["chop_hold"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Exit-shape overlay on top of active scaled current-Regime3 MoE. Entry, routing, expert models, and per-expert notional scales are unchanged.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bear_tp": selected["bear_tp"],
            "chop_tp": selected["chop_tp"],
            "chop_sl": selected["chop_sl"],
            "chop_hold": selected["chop_hold"],
            "validation": selected["validation"],
            "oos": selected["oos"],
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
