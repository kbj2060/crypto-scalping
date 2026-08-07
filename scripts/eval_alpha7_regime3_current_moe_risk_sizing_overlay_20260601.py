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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import SPLIT_TS, _combo_metrics, _json_default  # noqa: E402
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import (  # noqa: E402
    RISK_2025,
    RISK_2026,
    RISK_COLS,
    ROUTER_NAME,
    _load_router_frames,
)
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_risk_sizing_overlay_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_risk_sizing_overlay_20260601"
VAL_DEC = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601/practical_validation_decisions.csv"
OOS_DEC = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601/practical_oos_2026_decisions.csv"


def _read_overlay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["timestamp"], low_memory=False)
    if "timestamp" not in df.columns:
        raise ValueError(f"{path} missing timestamp")
    return df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)


def _edge_name(mask: pd.Series) -> str | None:
    idx = np.flatnonzero(mask.to_numpy())
    if len(idx) == 0:
        return None
    if np.array_equal(idx, np.arange(len(idx))):
        return "head"
    if np.array_equal(idx, np.arange(len(mask) - len(idx), len(mask))):
        return "tail"
    return None


def _overlay_risk_with_pos(base: pd.DataFrame, source: Path, *, tag: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    src = _read_overlay(source)
    missing = [c for c in RISK_COLS if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing risk columns: {missing}")
    out = base.copy().reset_index(drop=True)
    out["_orig_pos"] = np.arange(len(out), dtype=np.int64)
    missing_ts = out.loc[~out["timestamp"].isin(set(src["timestamp"])), "timestamp"]
    dropped: list[dict[str, Any]] = []
    if len(missing_ts) > 0:
        miss = missing_ts.reset_index(drop=True)
        head = out["timestamp"].head(len(missing_ts)).reset_index(drop=True)
        tail = out["timestamp"].tail(len(missing_ts)).reset_index(drop=True)
        if miss.equals(head):
            edge = "head"
        elif miss.equals(tail):
            edge = "tail"
        else:
            raise RuntimeError(f"{tag}: risk missing non-edge timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1]), "path": str(source)})
        out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *RISK_COLS]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after risk overlay")
    nan_mask = out[RISK_COLS].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        raise RuntimeError(f"{tag}: risk non-edge NaN rows: {out.loc[nan_mask, 'timestamp'].head(20).tolist()}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "path": str(source), "reason": "risk_edge_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    pos = out["_orig_pos"].to_numpy(dtype=np.int64)
    out = out.drop(columns=["_orig_pos"])
    return out, pos, {"path": str(source), "cols": RISK_COLS, "dropped_edge_rows": dropped}


def _apply_sizing(
    dec: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    risk_thr: float,
    churn_thr: float,
    router_conf_thr: float,
    scale: float,
    tighten: bool,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    risk = pd.to_numeric(frame["regime3_transition_h6_risk_prob"], errors="raise").to_numpy(dtype=np.float64)
    churn = pd.to_numeric(frame["regime3_churn_h6_risk_score"], errors="raise").to_numpy(dtype=np.float64)
    router_conf = pd.to_numeric(out["router_confidence"], errors="raise").to_numpy(dtype=np.float64)
    trigger = active & ((risk >= float(risk_thr)) | (churn >= float(churn_thr)) | (router_conf < float(router_conf_thr)))
    out.loc[trigger, "notional_exposure"] = pd.to_numeric(out.loc[trigger, "notional_exposure"], errors="raise") * float(scale)
    out.loc[trigger, "position_fraction"] = pd.to_numeric(out.loc[trigger, "position_fraction"], errors="raise") * float(scale)
    if tighten:
        out.loc[trigger, "stop_loss"] = pd.to_numeric(out.loc[trigger, "stop_loss"], errors="raise") * 0.80
        hold = pd.to_numeric(out.loc[trigger, "max_hold_bars"], errors="raise").to_numpy(dtype=np.float64)
        out.loc[trigger, "max_hold_bars"] = np.maximum(1, np.ceil(hold * 0.70)).astype(int)
    out["risk_sizing_trigger"] = trigger
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_raw, router_overlay = _load_router_frames(ROUTER_NAME)
    val_raw = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_df, val_pos, val_risk = _overlay_risk_with_pos(val_raw, RISK_2025, tag="validation_risk")
    eval_df, oos_pos, oos_risk = _overlay_risk_with_pos(eval_raw, RISK_2026, tag="oos_risk")
    val_base = pd.read_csv(VAL_DEC).iloc[val_pos].reset_index(drop=True)
    oos_base = pd.read_csv(OOS_DEC).iloc[oos_pos].reset_index(drop=True)
    overlay = {"router": router_overlay, "validation_risk": val_risk, "oos_risk": oos_risk}
    baseline_val = _combo_metrics(val_df, val_base)
    baseline_oos = _combo_metrics(eval_df, oos_base)
    rows: list[dict[str, Any]] = [{
        "candidate": "practical_moe_baseline",
        "risk_thr": None,
        "churn_thr": None,
        "router_conf_thr": None,
        "scale": None,
        "tighten": None,
        "score": float(_score(baseline_val)),
        "validation": baseline_val,
        "oos": baseline_oos,
        "validation_triggered": None,
        "oos_triggered": None,
        "validation_policy_counts": {str(k): int(v) for k, v in val_base["router_expert"].value_counts().to_dict().items()},
        "oos_policy_counts": {str(k): int(v) for k, v in oos_base["router_expert"].value_counts().to_dict().items()},
    }]
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for risk_thr in [0.55, 0.65]:
        for churn_thr in [0.55]:
            for router_conf_thr in [0.80]:
                for scale in [0.65, 0.80]:
                    for tighten in [False, True]:
                        val_dec = _apply_sizing(val_base, val_df, risk_thr=risk_thr, churn_thr=churn_thr, router_conf_thr=router_conf_thr, scale=scale, tighten=tighten)
                        oos_dec = _apply_sizing(oos_base, eval_df, risk_thr=risk_thr, churn_thr=churn_thr, router_conf_thr=router_conf_thr, scale=scale, tighten=tighten)
                        val_costs = _combo_metrics(val_df, val_dec)
                        oos_costs = _combo_metrics(eval_df, oos_dec)
                        key = f"r{risk_thr:.2f}_c{churn_thr:.2f}_q{router_conf_thr:.2f}_s{scale:.2f}_t{int(tighten)}"
                        payload[key] = (val_dec, oos_dec)
                        rows.append({
                            "candidate": key,
                            "risk_thr": float(risk_thr),
                            "churn_thr": float(churn_thr),
                            "router_conf_thr": float(router_conf_thr),
                            "scale": float(scale),
                            "tighten": bool(tighten),
                            "score": float(_score(val_costs)),
                            "validation": val_costs,
                            "oos": oos_costs,
                            "validation_triggered": int(val_dec["risk_sizing_trigger"].sum()),
                            "oos_triggered": int(oos_dec["risk_sizing_trigger"].sum()),
                            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                            "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                        })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    if selected["candidate"] == "practical_moe_baseline":
        selected_val_dec = val_base.copy()
        selected_oos_dec = oos_base.copy()
    else:
        selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "risk_thr": r["risk_thr"],
            "churn_thr": r["churn_thr"],
            "router_conf_thr": r["router_conf_thr"],
            "scale": r["scale"],
            "tighten": r["tighten"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_triggered": r["validation_triggered"],
            "oos_triggered": r["oos_triggered"],
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "Risk/churn/router-confidence sizing overlay on top of Regime3 current-context practical MoE. Bull/bear/chop expert frame is unchanged.",
        "overlay": overlay,
        "selected": {
            "candidate": selected["candidate"],
            "risk_thr": selected["risk_thr"],
            "churn_thr": selected["churn_thr"],
            "router_conf_thr": selected["router_conf_thr"],
            "scale": selected["scale"],
            "tighten": selected["tighten"],
            "validation": selected["validation"],
            "oos": selected["oos"],
            "validation_triggered": selected["validation_triggered"],
            "oos_triggered": selected["oos_triggered"],
            "validation_policy_counts": {str(k): int(v) for k, v in selected_val_dec["router_expert"].value_counts().to_dict().items()},
            "oos_policy_counts": {str(k): int(v) for k, v in selected_oos_dec["router_expert"].value_counts().to_dict().items()},
        },
        "top_grid": rows[:20],
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
