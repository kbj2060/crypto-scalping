#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default
from scripts.eval_alpha7_regime3_current_moe_active_mix_expert_scale_refine_20260601 import _apply_scale
from scripts.retrain_alpha7_active_max_feature_contract_moe_20260601 import _load_frames_max


MODEL_ID = "omega1_regime3_expertdq_risk_replay_20260602"
EXPERTDQ_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_routed_expert_direction_quality_20260602"
ACTIVE_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_active_max_feature_zigzag_moe_risk_redesign_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega1_regime3_expertdq_risk_replay_20260602"

ACTION_CASH = 0
ACTION_LONG = 1
ACTION_SHORT = 2

ACTIVE_TEMPLATE = {
    "notional": 0.45,
    "leverage": 2.0,
    "take_profit": 0.026,
    "stop_loss": 0.014,
    "max_hold": 72,
    "cooldown": 6,
}
ACTIVE_SCALES = {"bull": 0.75, "bear": 0.90, "chop": 0.90}


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _require_unique_timestamps(frame: pd.DataFrame, name: str) -> None:
    if "timestamp" not in frame.columns:
        raise RuntimeError(f"{name}: missing timestamp")
    ts = pd.to_datetime(frame["timestamp"], errors="raise")
    if ts.duplicated().any():
        dup = frame.loc[ts.duplicated(), "timestamp"].head(5).tolist()
        raise RuntimeError(f"{name}: duplicate timestamps: {dup}")


def _align_source_to_frame(frame: pd.DataFrame, dec: pd.DataFrame, src: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame_ts = pd.to_datetime(frame["timestamp"], errors="raise")
    src_ts = pd.to_datetime(src["timestamp"], errors="raise")
    src_lookup = pd.Series(np.arange(len(src), dtype=np.int64), index=src_ts)
    mask = frame_ts.isin(set(src_ts))
    out_frame = frame.loc[mask].reset_index(drop=True)
    out_dec = dec.loc[mask].reset_index(drop=True)
    out_src_idx = src_lookup.loc[pd.to_datetime(out_frame["timestamp"], errors="raise")].to_numpy(dtype=np.int64)
    out_src = src.iloc[out_src_idx].reset_index(drop=True)
    if len(out_frame) == 0:
        raise RuntimeError("empty timestamp intersection")
    if not out_frame["timestamp"].astype(str).reset_index(drop=True).equals(out_src["timestamp"].astype(str).reset_index(drop=True)):
        raise RuntimeError("source/frame timestamp order mismatch")
    return out_frame, out_dec, out_src


def _expertdq_paths(variant: str) -> tuple[Path, Path]:
    vdir = EXPERTDQ_DIR / variant
    return (
        vdir / f"training_features_2025_{variant}_omega1_regime3_expertdq_oof_20260602.csv",
        vdir / f"training_features_2026_rebuilt_{variant}_omega1_regime3_expertdq_20260602.csv",
    )


def _to_decisions(src: pd.DataFrame, *, oof: bool) -> pd.DataFrame:
    prefix = "omega1_regime3_expertdq_oof_" if oof else "omega1_regime3_expertdq_"
    required = [
        "timestamp",
        f"{prefix}router_expert",
        f"{prefix}final_action",
        f"{prefix}quality_for_action",
        f"{prefix}dir_confidence",
    ]
    missing = [c for c in required if c not in src.columns]
    if missing:
        raise RuntimeError(f"missing expert DQ columns: {missing}")

    action = pd.to_numeric(src[f"{prefix}final_action"], errors="raise").to_numpy(dtype=np.int64)
    if not set(np.unique(action)).issubset({ACTION_CASH, ACTION_LONG, ACTION_SHORT}):
        raise RuntimeError(f"unexpected final_action values: {sorted(np.unique(action).tolist())}")
    active = action != ACTION_CASH
    side = np.where(action == ACTION_LONG, 1, np.where(action == ACTION_SHORT, -1, 0)).astype(np.int64)
    router = src[f"{prefix}router_expert"].astype(str).replace({"chop": "chop_expert"})

    dec = pd.DataFrame(
        {
            "action": action,
            "side": side,
            "notional_exposure": np.where(active, float(ACTIVE_TEMPLATE["notional"]), 0.0),
            "leverage": np.where(active, float(ACTIVE_TEMPLATE["leverage"]), 1.0),
            "position_fraction": np.where(active, float(ACTIVE_TEMPLATE["notional"]), 0.0),
            "take_profit": np.where(active, float(ACTIVE_TEMPLATE["take_profit"]), 0.0),
            "stop_loss": np.where(active, float(ACTIVE_TEMPLATE["stop_loss"]), 0.0),
            "max_hold_bars": np.where(active, int(ACTIVE_TEMPLATE["max_hold"]), 0).astype(np.int64),
            "cooldown_bars": np.where(active, int(ACTIVE_TEMPLATE["cooldown"]), 0).astype(np.int64),
            "quality_score": pd.to_numeric(src[f"{prefix}quality_for_action"], errors="raise").to_numpy(dtype=np.float64),
            "confidence": pd.to_numeric(src[f"{prefix}dir_confidence"], errors="raise").to_numpy(dtype=np.float64),
            "router_expert": router.to_numpy(),
        }
    )
    return _apply_scale(dec, **ACTIVE_SCALES)


def _compact_delta(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, float]:
    return {
        "pnl": float(candidate["pnl"]) - float(baseline["pnl"]),
        "mdd": float(candidate["mdd"]) - float(baseline["mdd"]),
        "trades": float(candidate["trades"]) - float(baseline["trades"]),
        "wr": float(candidate["wr"]) - float(baseline["wr"]),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay = _load_frames_max()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)

    active_val_dec = _load_csv(ACTIVE_DIR / "validation_decisions.csv").reset_index(drop=True)
    active_oos_dec = _load_csv(ACTIVE_DIR / "oos_2026_decisions.csv").reset_index(drop=True)
    if len(active_val_dec) != len(val_df) or len(active_oos_dec) != len(eval_df):
        raise RuntimeError(
            f"active decision length mismatch: val {len(active_val_dec)} vs {len(val_df)}, "
            f"oos {len(active_oos_dec)} vs {len(eval_df)}"
        )

    active_report = json.loads((ACTIVE_DIR / "report.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    reports: dict[str, Any] = {}

    for variant_dir in sorted(p for p in EXPERTDQ_DIR.iterdir() if p.is_dir()):
        variant = variant_dir.name
        val_path, oos_path = _expertdq_paths(variant)
        if not val_path.exists() or not oos_path.exists():
            continue
        val_src = _load_csv(val_path)
        oos_src = _load_csv(oos_path)
        _require_unique_timestamps(val_src, f"{variant} val")
        _require_unique_timestamps(oos_src, f"{variant} oos")

        val_frame_common, active_val_common, val_src_common = _align_source_to_frame(val_df, active_val_dec, val_src)
        oos_frame_common, active_oos_common, oos_src_common = _align_source_to_frame(eval_df, active_oos_dec, oos_src)
        val_dec = _to_decisions(val_src_common, oof=True)
        oos_dec = _to_decisions(oos_src_common, oof=False)
        if len(val_dec) != len(val_frame_common) or len(oos_dec) != len(oos_frame_common):
            raise RuntimeError(f"{variant}: common frame/decision length mismatch")

        val_costs = _combo_metrics(val_frame_common, val_dec)
        oos_costs = _combo_metrics(oos_frame_common, oos_dec)
        active_val_costs = _combo_metrics(val_frame_common, active_val_common)
        active_oos_costs = _combo_metrics(oos_frame_common, active_oos_common)

        report = {
            "variant": variant,
            "validation_rows_common": int(len(val_frame_common)),
            "oos_rows_common": int(len(oos_frame_common)),
            "validation": val_costs,
            "oos": oos_costs,
            "active_validation_common": active_val_costs,
            "active_oos_common": active_oos_costs,
            "delta_vs_active_common": {
                f"cost{mult}": _compact_delta(oos_costs[f"cost{mult}"], active_oos_costs[f"cost{mult}"])
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
        raise RuntimeError("no expert DQ variants found")

    ranking = pd.DataFrame(rows).sort_values("oos_cost3_pnl", ascending=False).reset_index(drop=True)
    ranking.to_csv(OUT_DIR / "ranking.csv", index=False)
    selected_variant = str(ranking.iloc[0]["variant"])
    report = {
        "model_id": MODEL_ID,
        "design": "Replay expert-local Regime3 Direction+Quality final_action through the current Omega1 ZigZag risk template and expert scales. No threshold or OOS selection is performed here; this is a post-hoc PnL diagnostic.",
        "active_reference_dir": str(ACTIVE_DIR),
        "expert_dq_dir": str(EXPERTDQ_DIR),
        "risk_template": ACTIVE_TEMPLATE,
        "expert_scales": ACTIVE_SCALES,
        "overlay": overlay,
        "active_full_period_reference": active_report.get("selected", {}),
        "selected_by_oos_cost3_pnl": selected_variant,
        "variants": reports,
        "artifacts": {
            "report": str(OUT_DIR / "report.json"),
            "ranking": str(OUT_DIR / "ranking.csv"),
        },
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_DIR / "report.json"), "selected": selected_variant, "top": ranking.head(8).to_dict(orient="records")}, ensure_ascii=False, indent=2, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
