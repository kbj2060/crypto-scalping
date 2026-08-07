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

from scripts.analyze_alpha7_tp_sl_action_score_20260526 import (  # noqa: E402
    SPLIT_TS,
    _combo_metrics,
    _json_default,
)
from scripts.train_alpha7_regime3_current_moe_feature_variants_20260601 import _load_frames_with_risk  # noqa: E402
from scripts.train_alpha7_regime3_expert_moe_20260601 import _active, _flatten, _score  # noqa: E402


MODEL_ID = "alpha7_regime3_current_moe_retrieval_overlay_20260601"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_moe_retrieval_overlay_20260601"
VAL_DEC = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601/practical_validation_decisions.csv"
OOS_DEC = ROOT / "tmp/causal_regen_20260516/alpha7_regime3_current_practical_moe_20260601/practical_oos_2026_decisions.csv"
RETRIEVAL_2025 = ROOT / "data/ensemble/supervised/omega1_dir3_retrieval_20260531/training_features_2025_omega1_dir3_retrieval_20260531.csv"
RETRIEVAL_2026 = ROOT / "data/ensemble/supervised/omega1_dir3_retrieval_20260531/training_features_2026_rebuilt_omega1_dir3_retrieval_20260531.csv"
RETRIEVAL_COLS = [
    "dir3_retrieval_h6_fl_prob",
    "dir3_retrieval_h6_up_prob",
    "dir3_retrieval_h6_dn_prob",
    "dir3_retrieval_h6_confidence",
    "dir3_retrieval_h6_side_edge",
    "dir3_retrieval_h6_trade_prob",
    "dir3_retrieval_h6_neighbor_edge_mean",
    "dir3_retrieval_h6_neighbor_edge_q25",
    "dir3_retrieval_h6_neighbor_edge_q75",
    "dir3_retrieval_h6_regime_consensus",
    "dir3_retrieval_h6_similarity_score",
]


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


def _overlay_retrieval(base: pd.DataFrame, source: Path, *, tag: str) -> tuple[pd.DataFrame, np.ndarray, dict[str, Any]]:
    src = _read_overlay(source)
    missing = [c for c in RETRIEVAL_COLS if c not in src.columns]
    if missing:
        raise RuntimeError(f"{tag}: missing retrieval columns: {missing}")
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
            raise RuntimeError(f"{tag}: retrieval missing non-edge timestamps: {missing_ts.head(20).tolist()}")
        dropped.append({"edge": edge, "rows": int(len(missing_ts)), "first": str(missing_ts.iloc[0]), "last": str(missing_ts.iloc[-1]), "path": str(source)})
        out = out.loc[out["timestamp"].isin(set(src["timestamp"]))].reset_index(drop=True)
    before = len(out)
    out = out.merge(src[["timestamp", *RETRIEVAL_COLS]], on="timestamp", how="left", validate="one_to_one")
    if len(out) != before:
        raise RuntimeError(f"{tag}: row count changed after retrieval overlay")
    nan_mask = out[RETRIEVAL_COLS].isna().any(axis=1)
    edge = _edge_name(nan_mask)
    if edge is None and bool(nan_mask.any()):
        raise RuntimeError(f"{tag}: retrieval non-edge NaN rows: {out.loc[nan_mask, 'timestamp'].head(20).tolist()}")
    if edge is not None:
        bad = out.loc[nan_mask, "timestamp"]
        dropped.append({"edge": edge, "rows": int(len(bad)), "first": str(bad.iloc[0]), "last": str(bad.iloc[-1]), "path": str(source), "reason": "retrieval_edge_nan"})
        out = out.loc[~nan_mask].reset_index(drop=True)
    pos = out["_orig_pos"].to_numpy(dtype=np.int64)
    out = out.drop(columns=["_orig_pos"])
    return out, pos, {"path": str(source), "dropped_edge_rows": dropped, "cols": RETRIEVAL_COLS}


def _apply_overlay(
    dec: pd.DataFrame,
    frame: pd.DataFrame,
    *,
    mode: str,
    edge_thr: float,
    trade_min: float,
    edge_mean_min: float,
    sim_min: float,
    consensus_min: float,
) -> pd.DataFrame:
    out = dec.copy().reset_index(drop=True)
    active = _active(out)
    side = pd.to_numeric(out["side"], errors="raise").to_numpy(dtype=np.int64)
    side_edge = pd.to_numeric(frame["dir3_retrieval_h6_side_edge"], errors="raise").to_numpy(dtype=np.float64)
    trade_prob = pd.to_numeric(frame["dir3_retrieval_h6_trade_prob"], errors="raise").to_numpy(dtype=np.float64)
    edge_mean = pd.to_numeric(frame["dir3_retrieval_h6_neighbor_edge_mean"], errors="raise").to_numpy(dtype=np.float64)
    sim = pd.to_numeric(frame["dir3_retrieval_h6_similarity_score"], errors="raise").to_numpy(dtype=np.float64)
    consensus = pd.to_numeric(frame["dir3_retrieval_h6_regime_consensus"], errors="raise").to_numpy(dtype=np.float64)
    conflict = active & ((side.astype(np.float64) * side_edge) < -float(edge_thr))
    weak = active & (
        (trade_prob < float(trade_min))
        | (edge_mean < float(edge_mean_min))
        | (sim < float(sim_min))
        | (consensus < float(consensus_min))
    )
    trigger = conflict | weak
    if mode == "veto":
        out.loc[trigger, ["action", "side", "notional_exposure", "position_fraction", "take_profit", "stop_loss", "max_hold_bars", "cooldown_bars"]] = 0
        out.loc[trigger, "leverage"] = 1.0
    elif mode == "resize":
        out.loc[trigger, "notional_exposure"] = pd.to_numeric(out.loc[trigger, "notional_exposure"], errors="raise") * 0.50
        out.loc[trigger, "position_fraction"] = pd.to_numeric(out.loc[trigger, "position_fraction"], errors="raise") * 0.50
    else:
        raise ValueError(f"unknown mode={mode}")
    out["retrieval_overlay_mode"] = mode
    out["retrieval_conflict"] = conflict
    out["retrieval_weak"] = weak
    out["retrieval_trigger"] = trigger
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train_all, eval_df, overlay_base = _load_frames_with_risk()
    val_df = train_all[train_all["timestamp"] >= SPLIT_TS].reset_index(drop=True)
    val_df, val_pos, val_ret = _overlay_retrieval(val_df, RETRIEVAL_2025, tag="validation")
    eval_df, oos_pos, oos_ret = _overlay_retrieval(eval_df, RETRIEVAL_2026, tag="oos_2026")
    val_base_dec = pd.read_csv(VAL_DEC).iloc[val_pos].reset_index(drop=True)
    oos_base_dec = pd.read_csv(OOS_DEC).iloc[oos_pos].reset_index(drop=True)
    baseline_val = _combo_metrics(val_df, val_base_dec)
    baseline_oos = _combo_metrics(eval_df, oos_base_dec)
    rows: list[dict[str, Any]] = [{
        "candidate": "practical_moe_baseline",
        "mode": None,
        "edge_thr": None,
        "trade_min": None,
        "edge_mean_min": None,
        "sim_min": None,
        "consensus_min": None,
        "score": float(_score(baseline_val)),
        "validation": baseline_val,
        "oos": baseline_oos,
        "validation_policy_counts": {str(k): int(v) for k, v in val_base_dec["router_expert"].value_counts().to_dict().items()},
        "oos_policy_counts": {str(k): int(v) for k, v in oos_base_dec["router_expert"].value_counts().to_dict().items()},
    }]
    payload: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for mode in ["veto", "resize"]:
        for edge_thr in [0.10, 0.20]:
            for trade_min in [0.65]:
                for edge_mean_min in [0.0, 0.0025]:
                    for sim_min in [0.08]:
                        consensus_min = 0.35
                        val_dec = _apply_overlay(val_base_dec, val_df, mode=mode, edge_thr=edge_thr, trade_min=trade_min, edge_mean_min=edge_mean_min, sim_min=sim_min, consensus_min=consensus_min)
                        oos_dec = _apply_overlay(oos_base_dec, eval_df, mode=mode, edge_thr=edge_thr, trade_min=trade_min, edge_mean_min=edge_mean_min, sim_min=sim_min, consensus_min=consensus_min)
                        val_costs = _combo_metrics(val_df, val_dec)
                        oos_costs = _combo_metrics(eval_df, oos_dec)
                        key = f"{mode}_e{edge_thr:.2f}_t{trade_min:.2f}_m{edge_mean_min:.4f}_s{sim_min:.2f}"
                        payload[key] = (val_dec, oos_dec)
                        rows.append({
                            "candidate": key,
                            "mode": mode,
                            "edge_thr": float(edge_thr),
                            "trade_min": float(trade_min),
                            "edge_mean_min": float(edge_mean_min),
                            "sim_min": float(sim_min),
                            "consensus_min": float(consensus_min),
                            "score": float(_score(val_costs)),
                            "validation": val_costs,
                            "oos": oos_costs,
                            "validation_triggered": int(val_dec["retrieval_trigger"].sum()),
                            "oos_triggered": int(oos_dec["retrieval_trigger"].sum()),
                            "validation_policy_counts": {str(k): int(v) for k, v in val_dec["router_expert"].value_counts().to_dict().items()},
                            "oos_policy_counts": {str(k): int(v) for k, v in oos_dec["router_expert"].value_counts().to_dict().items()},
                        })
    rows.sort(key=lambda r: float(r["score"]), reverse=True)
    selected = rows[0]
    if selected["candidate"] == "practical_moe_baseline":
        selected_val_dec = val_base_dec.copy()
        selected_oos_dec = oos_base_dec.copy()
    else:
        selected_val_dec, selected_oos_dec = payload[str(selected["candidate"])]
    selected_val_dec.to_csv(OUT_DIR / "validation_decisions.csv", index=False)
    selected_oos_dec.to_csv(OUT_DIR / "oos_2026_decisions.csv", index=False)
    pd.DataFrame([
        {
            "candidate": r["candidate"],
            "mode": r["mode"],
            "edge_thr": r["edge_thr"],
            "trade_min": r["trade_min"],
            "edge_mean_min": r["edge_mean_min"],
            "sim_min": r["sim_min"],
            "consensus_min": r["consensus_min"],
            "score": r["score"],
            **_flatten("val", r["validation"]),
            **_flatten("oos", r["oos"]),
            "validation_triggered": r.get("validation_triggered"),
            "oos_triggered": r.get("oos_triggered"),
            "validation_policy_counts": json.dumps(r["validation_policy_counts"], ensure_ascii=False),
            "oos_policy_counts": json.dumps(r["oos_policy_counts"], ensure_ascii=False),
        }
        for r in rows
    ]).to_csv(OUT_DIR / "ranking.csv", index=False)
    report = {
        "model_id": MODEL_ID,
        "design": "RAFT-style retrieval confirmation/veto overlay on top of Regime3 current-context practical MoE. The bull/bear/chop expert frame is unchanged.",
        "overlay": {"base": overlay_base, "validation_retrieval": val_ret, "oos_retrieval": oos_ret},
        "selected": {
            "candidate": selected["candidate"],
            "mode": selected["mode"],
            "edge_thr": selected["edge_thr"],
            "trade_min": selected["trade_min"],
            "edge_mean_min": selected["edge_mean_min"],
            "sim_min": selected["sim_min"],
            "consensus_min": selected["consensus_min"],
            "validation": selected["validation"],
            "oos": selected["oos"],
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
