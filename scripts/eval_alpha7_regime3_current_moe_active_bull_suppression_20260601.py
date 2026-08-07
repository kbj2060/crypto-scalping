#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_active_bull_suppression_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_bull_suppression_20260601"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"


def _cash_bull(dec: pd.DataFrame, *, bull_cash: bool) -> pd.DataFrame:
    if not bull_cash:
        return dec
    out = dec.copy()
    mask = out["router_expert"].astype(str).eq("bull")
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
            out.loc[mask, col] = value
    out["bull_cash_veto"] = bool(bull_cash)
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_base = pd.read_csv(ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    oos_base = pd.read_csv(ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    if len(val_df) != len(val_base) or len(eval_df) != len(oos_base):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_base)} oos {len(eval_df)} {len(oos_base)}")

    rows: list[dict[str, Any]] = []
    payload: dict[str, pd.DataFrame] = {}
    for bull_scale, bull_cash in [(0.00, False), (0.25, False), (0.50, False), (0.70, False), (0.85, False), (0.00, True)]:
        val_dec = _apply_scale(val_base, bull=bull_scale, bear=1.15, chop=1.25)
        val_dec = _cash_bull(val_dec, bull_cash=bull_cash)
        val_costs = _combo_metrics(val_df, val_dec)
        candidate = f"bull{'cash' if bull_cash else f'{bull_scale:.2f}'}_bear1.15_chop1.25"
        rows.append({
            "candidate": candidate,
            "bull_scale": float(bull_scale),
            "bull_cash": bool(bull_cash),
            "score": float(_score(val_costs)),
            "validation": val_costs,
            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
        })
        payload[candidate] = val_dec

    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    selected_val_dec = payload[str(selected["candidate"])]
    selected_oos_dec = _apply_scale(oos_base, bull=float(selected["bull_scale"]), bear=1.15, chop=1.25)
    selected_oos_dec = _cash_bull(selected_oos_dec, bull_cash=bool(selected["bull_cash"]))
    selected["oos"] = _combo_metrics(eval_df, selected_oos_dec)
    selected["oos_policy_counts"] = {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()}

    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "bull_scale": r["bull_scale"],
            "bull_cash": r["bull_cash"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)

    report = {
        "model_id": MODEL_ID,
        "design": "Validation-selected bull expert suppression on the active current-Regime3 MoE. Bear and chop scales stay active at 1.15/1.25; routing and expert models are unchanged.",
        "diagnostic_source": str(ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_expert_attribution_20260601/report.json"),
        "overlay": overlay,
        "selected": selected,
        "top_grid": rows,
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
