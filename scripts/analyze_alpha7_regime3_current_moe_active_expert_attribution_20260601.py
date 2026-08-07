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


MODEL_ID = "alpha7_regime3_current_moe_active_expert_attribution_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_expert_attribution_20260601"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601"


def _cash_decisions(dec: pd.DataFrame) -> pd.DataFrame:
    out = dec.copy()
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
            out[col] = value
    return out


def _subset_decisions(dec: pd.DataFrame, expert: str) -> pd.DataFrame:
    out = _cash_decisions(dec)
    mask = dec["router_expert"].astype(str).eq(expert)
    out.loc[mask, dec.columns] = dec.loc[mask, dec.columns].to_numpy()
    return out


def _period_metrics(frame: pd.DataFrame, dec: pd.DataFrame) -> list[dict[str, Any]]:
    periods = pd.to_datetime(frame["timestamp"], errors="raise").dt.to_period("M").astype(str)
    out: list[dict[str, Any]] = []
    for period in sorted(periods.unique()):
        mask = periods.eq(period).to_numpy()
        out.append({
            "period": period,
            "metrics": _combo_metrics(frame.loc[mask].reset_index(drop=True), dec.loc[mask].reset_index(drop=True)),
        })
    return out


def _attribution(frame: pd.DataFrame, dec: pd.DataFrame) -> dict[str, Any]:
    experts = [str(x) for x in dec["router_expert"].dropna().astype(str).unique().tolist()]
    experts = sorted(experts)
    full = _combo_metrics(frame, dec)
    pieces: dict[str, Any] = {}
    for expert in experts:
        sub = _subset_decisions(dec, expert)
        pieces[expert] = {
            "rows": int(dec["router_expert"].astype(str).eq(expert).sum()),
            "active_rows": int(((pd.to_numeric(dec["action"], errors="coerce").fillna(0).astype(int) != 0) & dec["router_expert"].astype(str).eq(expert)).sum()),
            "metrics": _combo_metrics(frame, sub),
            "monthly": _period_metrics(frame, sub),
        }
    return {"full": full, "pieces": pieces, "monthly": _period_metrics(frame, dec)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_dec = pd.read_csv(ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    oos_dec = pd.read_csv(ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    if len(val_df) != len(val_dec) or len(eval_df) != len(oos_dec):
        raise RuntimeError(f"frame/decision mismatch: val {len(val_df)} {len(val_dec)} oos {len(eval_df)} {len(oos_dec)}")
    report = {
        "model_id": MODEL_ID,
        "design": "Diagnostic only. Decomposes active current-Regime3 MoE decisions by router_expert by cashing out all other rows and rerunning the same backtest metric.",
        "active_candidate": "bull0.85_bear1.15_chop1.25",
        "overlay": overlay,
        "validation": _attribution(val_df, val_dec),
        "oos": _attribution(eval_df, oos_dec),
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "validation_full": report["validation"]["full"], "oos_full": report["oos"]["full"]}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
