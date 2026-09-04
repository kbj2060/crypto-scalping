#!/usr/bin/env python3
"""User request: check whether the liquidation cascade switching/sustain indicator (dashboard
"OI 급변" successor, see eth_liquidation_cascade_sweep_vs_trend_pilot_20260828 memory) could
improve the V_REBOUND model. DIAGNOSTIC ONLY -- not promotion-eligible, per two blockers already
flagged to the user: (1) tail_risk_1m/microstructure_1m only have valid history from 2026-07-18,
entirely after TRAIN/VAL/OOS end, overlapping only the tail of the already-spent RESERVED holdout;
(2) the indicator's own confirmed rule is based on holdout n=8-9 with a documented small-N sign
flip (18->121). This script does NOT touch the reserved-holdout evaluation protocol or make any
promotion decision -- it only sizes up whether the idea is worth a real revisit once the data has
enough history (this repo's convention: 09-15 gate for liquidation-feed-based signals).

Reuses the pilot's own causal hawkes replay and feature formulas verbatim (import, not
reimplement) from research_eth_liquidation_cascade_sweep_vs_trend_pilot_20260828.py and its
_hybrid.py sibling, against freshly re-extracted tail_risk_1m/microstructure_1m (through
2026-08-28, re-pulled from the server today -- the original pilot snapshot stopped 2026-08-27).

Join logic: a cascade's t0 5m-bar and one of our 14,259 V_REBOUND sweep events are the SAME
population when they land on the same bar timestamp with matching direction (cascade 'down' ==
our 'downside', 'up' == 'upside') -- matching on exact timestamp+side implicitly requires both
genuine_breach AND the close-back-inside reclaim (a sweep event can't exist in our label file
without both), so no separate genuine_breach filter is needed on top of the join itself.
"""
from __future__ import annotations

import importlib.util
import json
import time as time_module
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data/research/eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"
VREBOUND_LABELS = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829/eth_5m_sweep_v_rebound_labels.csv"

NIF_FEATURE_WINDOW_MIN = 15  # matches _hybrid.py's FEATURE_WINDOW_MIN


def load_module(rel_path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    pilot = load_module("scripts/research_eth_liquidation_cascade_sweep_vs_trend_pilot_20260828.py", "pilot_core_20260829")
    hybrid = load_module("scripts/research_eth_liquidation_cascade_sweep_vs_trend_pilot_20260828_hybrid.py", "pilot_hybrid_20260829")

    tail_risk = pd.read_csv(DATA_DIR / "tail_risk_1m.csv", parse_dates=["ts"])
    tail_risk["ts"] = pd.to_datetime(tail_risk["ts"], utc=True)
    n_before = len(tail_risk)
    tail_risk = tail_risk[(tail_risk["valid_liq_stream"] == True) & (tail_risk["ws_stale"] != True)]  # noqa: E712
    tail_risk = tail_risk.sort_values("ts").reset_index(drop=True)
    print(f"tail_risk_1m: {n_before} rows -> {len(tail_risk)} after valid_liq_stream/ws_stale filter, "
          f"{tail_risk['ts'].min()} -> {tail_risk['ts'].max()}")

    micro_df = pd.read_csv(DATA_DIR / "microstructure_1m.csv", parse_dates=["ts"])
    micro_df["ts"] = pd.to_datetime(micro_df["ts"], utc=True)
    micro_df = micro_df.sort_values("ts").reset_index(drop=True)

    start_ms = int(pilot.WINDOW_START_UTC.timestamp() * 1000)
    end_ms = int(pd.Timestamp.utcnow().timestamp() * 1000)
    print("fetching futures 5m klines (public Binance API, same source as the core pilot)...")
    fut_kl = pilot.fetch_klines("https://fapi.binance.com/fapi/v1/klines", "ETHUSDT", start_ms, end_ms)
    print(f"  {len(fut_kl)} bars, {fut_kl['timestamp'].min()} -> {fut_kl['timestamp'].max()}")

    print("replaying Definition A hawkes cascade (causal, real tail_risk_interceptor.py)...")
    events_a = pilot.replay_definition_a(tail_risk)
    print(f"  {len(events_a)} onset events")

    labeled = pilot.label_events(events_a, fut_kl)
    print(f"  {len(labeled)} labeled (t0 fell inside the 5m-kline window with full lookback)")

    # only need wick_body_ratio -- pass real dfs for the other extract_features args so nothing
    # crashes, but their outputs (oi/orderbook/shadow) are unused below.
    oi_df = pd.read_csv(DATA_DIR / "oi_lsratio_5m.csv", parse_dates=["ts"])
    oi_df["ts"] = pd.to_datetime(oi_df["ts"], utc=True)
    ob_df = pd.read_csv(DATA_DIR / "orderbook_decision_snapshots.csv", parse_dates=["recorded_at_kst"])
    ob_df["recorded_at_kst"] = pd.to_datetime(ob_df["recorded_at_kst"], utc=True)
    full = pilot.extract_features(labeled, fut_kl, fut_kl, oi_df, ob_df, micro_df)

    full["genuine_breach"] = (
        ((full["direction"] == "down") & (full["cascade_extreme"] < full["swept_level"]))
        | ((full["direction"] == "up") & (full["cascade_extreme"] > full["swept_level"]))
    )
    genuine = full[full["genuine_breach"]].copy()
    print(f"genuine_breach cascade events: {len(genuine)}/{len(full)}")

    # nif_whale/nif_retail raw means, [t0, t0+15min] -- exact reuse of _hybrid.py's windowing
    nif = hybrid.add_new_features(genuine[["event_id", "t0"]], micro_df)[["event_id", "nif_whale", "nif_retail"]]
    genuine = genuine.merge(nif, on="event_id", how="left")

    # direction-relative sign flip (memory: "하락캐스케이드는 그대로, 상승캐스케이드는 부호반전" --
    # down cascades unchanged, up cascades sign-flipped, so positive always means "contrarian/switching-favoring")
    sign = np.where(genuine["direction"] == "down", 1.0, -1.0)
    genuine["nif_whale_rel"] = genuine["nif_whale"] * sign
    genuine["nif_retail_rel"] = genuine["nif_retail"] * sign

    # map cascade t0-bar timestamp -> our V_REBOUND sweep event (same population when matched:
    # requires both genuine_breach AND close-back-inside reclaim, which is exactly what being in
    # our label file already means)
    fut_kl_idx = fut_kl.reset_index(drop=True)
    genuine["bar_timestamp"] = fut_kl_idx["timestamp"].iloc[genuine["t0_idx"]].to_numpy()
    genuine["side"] = np.where(genuine["direction"] == "down", "downside", "upside")

    vrebound = pd.read_csv(VREBOUND_LABELS)
    vrebound["timestamp"] = pd.to_datetime(vrebound["timestamp"], utc=True)

    matched = genuine.merge(
        vrebound[["timestamp", "side", "label"]],
        left_on=["bar_timestamp", "side"], right_on=["timestamp", "side"], how="inner",
    )
    print(f"matched to a real V_REBOUND sweep event (same bar+side): {len(matched)}/{len(genuine)}")

    def naive_rate(d: pd.DataFrame) -> float:
        return float(d["label"].mean()) if len(d) else float("nan")

    result = {
        "n_hawkes_onsets": int(len(events_a)),
        "n_genuine_breach": int(len(genuine)),
        "n_matched_to_vrebound_sweep": int(len(matched)),
        "matched_label_rate_V_REBOUND": naive_rate(matched),
        "note": "matched_label_rate is this tiny population's own V_REBOUND=1 base rate, for "
                "comparison against each rule's precision below",
    }

    # rule 1 (pilot's "sustain/지속" call): wick_body_ratio<0.5 AND nif_whale_rel<=0 -> predict
    # V_REBOUND=0 (NO_V_REBOUND / continuation, "sustain" in the pilot's own vocabulary)
    m1 = matched.dropna(subset=["wick_body_ratio", "nif_whale_rel"])
    pred1 = (m1["wick_body_ratio"] < 0.5) & (m1["nif_whale_rel"] <= 0)
    n1 = int(pred1.sum())
    prec1 = float((m1.loc[pred1, "label"] == 0).mean()) if n1 else None
    result["rule_sustain_wick<0.5_and_nifwhalerel<=0_predicts_label0"] = {
        "n_eligible": int(len(m1)), "n_fired": n1, "precision": prec1,
        "vs_naive_predict_label0_always": float((m1["label"] == 0).mean()) if len(m1) else None,
    }

    # rule 2 (pilot's "switching/스위칭" call): wick_body_ratio>2.0 alone -> predict V_REBOUND=1
    m2 = matched.dropna(subset=["wick_body_ratio"])
    pred2 = m2["wick_body_ratio"] > 2.0
    n2 = int(pred2.sum())
    prec2 = float((m2.loc[pred2, "label"] == 1).mean()) if n2 else None
    result["rule_switching_wick>2.0_predicts_label1"] = {
        "n_eligible": int(len(m2)), "n_fired": n2, "precision": prec2,
        "vs_naive_predict_label1_always": float((m2["label"] == 1).mean()) if len(m2) else None,
    }

    print(json.dumps(result, ensure_ascii=False, indent=2))
    out_path = DATA_DIR / "vrebound_diagnostic_report.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    matched.to_csv(DATA_DIR / "vrebound_matched_events.csv", index=False)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
