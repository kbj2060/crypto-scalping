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
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_expert_source_mix_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"


def _apply_scale(dec: pd.DataFrame, *, bull: float, bear: float, chop: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    scales = {
        "bull": float(bull),
        "bear": float(bear),
        "chop_expert": float(chop),
    }
    for name, scale in scales.items():
        mask = active & out["router_expert"].astype(str).eq(name)
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * scale
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * scale
    out["expert_scale_bull"] = float(bull)
    out["expert_scale_bear"] = float(bear)
    out["expert_scale_chop"] = float(chop)
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
    for bull, bear, chop in itertools.product([0.70, 0.80, 0.85], [1.15, 1.30, 1.45], [1.10, 1.25]):
        val_dec = _apply_scale(val_base, bull=bull, bear=bear, chop=chop)
        oos_dec = _apply_scale(oos_base, bull=bull, bear=bear, chop=chop)
        val_costs = _combo_metrics(val_df, val_dec)
        oos_costs = _combo_metrics(eval_df, oos_dec)
        key = f"bull{bull:.2f}_bear{bear:.2f}_chop{chop:.2f}"
        payload[key] = (val_dec, oos_dec)
        rows.append({
            "candidate": key,
            "bull_scale": float(bull),
            "bear_scale": float(bear),
            "chop_scale": float(chop),
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
            "bull_scale": r["bull_scale"],
            "bear_scale": r["bear_scale"],
            "chop_scale": r["chop_scale"],
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
        "design": "Refined per-expert notional/position scaling around the validation-selected active current-Regime3 MoE expert-source mix. Routing and expert models are unchanged.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "bull_scale": selected["bull_scale"],
            "bear_scale": selected["bear_scale"],
            "chop_scale": selected["chop_scale"],
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
