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

import train_eval_omega_parent_quality_gate_upgrade_20260618 as qgate  # noqa: E402
import train_eval_omega1_2_tabm_3head_20260603 as threehead  # noqa: E402
import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import train_eval_omega1_2_1_cash_fallback_sleeve_20260606 as sleeve  # noqa: E402


MODEL_ID = "omega_parent_risk_template_sweep_20260618"
OUT_DIR = ROOT / "tmp/causal_regen_20260516" / MODEL_ID
TP_GRID = (0.042, 0.052, 0.062, 0.072)
SL_GRID = (0.022, 0.028, 0.035, 0.040)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(type(obj).__name__)


def _apply_risk(dec: pd.DataFrame, tp: float, sl: float) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = omega._active(out)
    out.loc[active, "take_profit"] = float(tp)
    out.loc[active, "stop_loss"] = float(sl)
    out.loc[active, "max_hold_bars"] = 0
    out.loc[active, "cooldown_bars"] = 0
    return out


def _row(candidate: str, tp: float, sl: float, val_m: dict[str, Any], oos_m: dict[str, Any], base_val: dict[str, Any], base_oos: dict[str, Any]) -> dict[str, Any]:
    row = {"candidate": candidate, "take_profit": float(tp), "stop_loss": float(sl), "rr": float(tp / max(sl, 1.0e-12))}
    row.update(sleeve._metric_row("val", {**val_m, "primary_entries": val_m["long_entries"] + val_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row.update(sleeve._metric_row("oos", {**oos_m, "primary_entries": oos_m["long_entries"] + oos_m["short_entries"], "fallback_entries": 0, "primary_takeovers": 0}))
    row["val_delta_pnl_vs_current"] = float(row["val_pnl"] - base_val["pnl"])
    row["oos_delta_pnl_vs_current"] = float(row["oos_pnl"] - base_oos["pnl"])
    row["selection_score_val_only"] = float(row["val_pnl"]) + 0.20 * float(row["val_mdd"]) + 8.0 * float(row["val_wr"])
    return row


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"stage": "load", "model_id": MODEL_ID}, ensure_ascii=True), flush=True)
    frames = threehead._prepare_frames(disable_tp_sl=False)
    fee, slip = omega._load_fee_slip()
    val_frame, val_src, val_prefix, val_oof = qgate._load_split(frames, "validation")
    oos_frame, oos_src, oos_prefix, oos_oof = qgate._load_split(frames, "oos")
    current_cfg = {"family": "current_control"}
    val_dec0 = qgate._build_dec(val_src, val_prefix, oof=val_oof, cfg=current_cfg)
    oos_dec0 = qgate._build_dec(oos_src, oos_prefix, oof=oos_oof, cfg=current_cfg)
    base_val = omega._metrics(val_frame, val_dec0, fee=fee, slip=slip, cost_mult=3.0)
    base_oos = omega._metrics(oos_frame, oos_dec0, fee=fee, slip=slip, cost_mult=3.0)

    rows: list[dict[str, Any]] = []
    for tp in TP_GRID:
        for sl in SL_GRID:
            val_dec = _apply_risk(val_dec0, tp, sl)
            oos_dec = _apply_risk(oos_dec0, tp, sl)
            val_m = omega._metrics(val_frame, val_dec, fee=fee, slip=slip, cost_mult=3.0)
            oos_m = omega._metrics(oos_frame, oos_dec, fee=fee, slip=slip, cost_mult=3.0)
            rows.append(_row(f"tp{tp:.3f}_sl{sl:.3f}", tp, sl, val_m, oos_m, base_val, base_oos))

    ranking = pd.DataFrame(rows).sort_values(["selection_score_val_only", "val_pnl", "val_mdd"], ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "risk_template_ranking.csv", index=False)
    selected = ranking.iloc[0].to_dict()
    best_oos = ranking.sort_values(["oos_pnl", "oos_delta_pnl_vs_current"], ascending=False).iloc[0].to_dict()
    report = {
        "model_id": MODEL_ID,
        "status": "redteam_pass_parent_risk_template_sweep",
        "method": "Replay current parent entries and change only effective applied TP/SL. Notional/leverage/actions are unchanged. Select by validation only; OOS diagnostic only.",
        "current_effective_risk": {"take_profit": 0.052, "stop_loss": 0.028, "leverage": 2.0},
        "current_control": _row("current_control", 0.052, 0.028, base_val, base_oos, base_val, base_oos),
        "selected_by_validation": selected,
        "best_by_oos_diagnostic": best_oos,
        "ranking": ranking.to_dict(orient="records"),
        "artifacts": {
            "out_dir": str(OUT_DIR),
            "ranking": str(OUT_DIR / "risk_template_ranking.csv"),
            "report": str(OUT_DIR / "report.json"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected, "current": report["current_control"], "best_oos": best_oos}, indent=2, ensure_ascii=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
