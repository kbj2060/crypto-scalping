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
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_active_scaled_route_quality_scale_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_scaled_route_quality_scale_20260601"
BASE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"
VAL_DEC = BASE_DIR / "validation_decisions.csv"
OOS_DEC = BASE_DIR / "oos_2026_decisions.csv"
MARGIN_COL = "regime3_current_sensitive_wide24_margin"
ENTROPY_COL = "regime3_current_sensitive_wide24_entropy"


def _apply_route_quality(
    dec: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    margin_hi: float,
    margin_lo: float,
    entropy_hi: float,
    up_scale: float,
    down_scale: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    expert = out["router_expert"].astype(str).isin(["bull", "bear", "chop_expert"])
    margin = pd.to_numeric(frame[MARGIN_COL], errors="raise")
    entropy = pd.to_numeric(frame[ENTROPY_COL], errors="raise")
    high_quality = active & expert & (margin >= float(margin_hi)) & (entropy <= float(entropy_hi))
    low_quality = active & expert & (margin < float(margin_lo))
    # If a row is both high and low due to a bad grid, fail loudly.
    if bool((high_quality & low_quality).any()):
        raise RuntimeError("route-quality masks overlap; invalid grid")
    for mask, scale in [(high_quality, up_scale), (low_quality, down_scale)]:
        out.loc[mask, "notional_exposure"] = pd.to_numeric(out.loc[mask, "notional_exposure"], errors="raise") * float(scale)
        out.loc[mask, "position_fraction"] = pd.to_numeric(out.loc[mask, "position_fraction"], errors="raise") * float(scale)
    out["route_quality_margin_hi"] = float(margin_hi)
    out["route_quality_margin_lo"] = float(margin_lo)
    out["route_quality_entropy_hi"] = float(entropy_hi)
    out["route_quality_up_scale"] = float(up_scale)
    out["route_quality_down_scale"] = float(down_scale)
    out["route_quality_high"] = high_quality
    out["route_quality_low"] = low_quality
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    for col in [MARGIN_COL, ENTROPY_COL]:
        if col not in val_df.columns or col not in eval_df.columns:
            raise RuntimeError(f"missing required current-regime quality column: {col}")
    val_base = pd.read_csv(VAL_DEC).reset_index(drop=True)
    oos_base = pd.read_csv(OOS_DEC).reset_index(drop=True)
    if len(val_df) != len(val_base) or len(eval_df) != len(oos_base):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_base)} oos {len(eval_df)} {len(oos_base)}")

    rows: list[dict[str, Any]] = []
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for margin_hi in [0.35, 0.45]:
        for margin_lo in [0.15, 0.20]:
            for entropy_hi in [0.95, 1.05]:
                for up_scale in [1.05, 1.10]:
                    for down_scale in [0.80, 0.90]:
                        val_dec = _apply_route_quality(
                            val_base,
                            val_df,
                            margin_hi=margin_hi,
                            margin_lo=margin_lo,
                            entropy_hi=entropy_hi,
                            up_scale=up_scale,
                            down_scale=down_scale,
                        )
                        oos_dec = _apply_route_quality(
                            oos_base,
                            eval_df,
                            margin_hi=margin_hi,
                            margin_lo=margin_lo,
                            entropy_hi=entropy_hi,
                            up_scale=up_scale,
                            down_scale=down_scale,
                        )
                        val_costs = _combo_metrics(val_df, val_dec)
                        oos_costs = _combo_metrics(eval_df, oos_dec)
                        key = f"mhi{margin_hi:.2f}_mlo{margin_lo:.2f}_e{entropy_hi:.2f}_up{up_scale:.2f}_dn{down_scale:.2f}"
                        payload[key] = (val_dec, oos_dec)
                        rows.append({
                            "candidate": key,
                            "margin_hi": float(margin_hi),
                            "margin_lo": float(margin_lo),
                            "entropy_hi": float(entropy_hi),
                            "up_scale": float(up_scale),
                            "down_scale": float(down_scale),
                            "score": float(_score(val_costs)),
                            "validation": val_costs,
                            "oos": oos_costs,
                            "validation_high": int(val_dec["route_quality_high"].sum()),
                            "validation_low": int(val_dec["route_quality_low"].sum()),
                            "oos_high": int(oos_dec["route_quality_high"].sum()),
                            "oos_low": int(oos_dec["route_quality_low"].sum()),
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
            "margin_hi": r["margin_hi"],
            "margin_lo": r["margin_lo"],
            "entropy_hi": r["entropy_hi"],
            "up_scale": r["up_scale"],
            "down_scale": r["down_scale"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_high": r["validation_high"],
            "validation_low": r["validation_low"],
            "oos_high": r["oos_high"],
            "oos_low": r["oos_low"],
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Route-quality notional scale on top of active scaled current-Regime3 MoE. Expert ownership and entry/exit shape are unchanged; only current-regime margin/entropy adjusts expert-row exposure.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "margin_hi": selected["margin_hi"],
            "margin_lo": selected["margin_lo"],
            "entropy_hi": selected["entropy_hi"],
            "up_scale": selected["up_scale"],
            "down_scale": selected["down_scale"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_high": selected["validation_high"],
            "validation_low": selected["validation_low"],
            "oos_high": selected["oos_high"],
            "oos_low": selected["oos_low"],
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
