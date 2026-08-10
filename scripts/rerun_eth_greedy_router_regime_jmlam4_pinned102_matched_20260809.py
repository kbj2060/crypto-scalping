"""CORRECTED final router comparison: both components use the LIVE 102-column feature contract
(pinned via train_eval_omega4_3head_parent72_pinned102_20260727.py), regime3 swapped to JM lambda=4,
thresholds chosen to match original wide24 activation rate (h48qual q070 ~0.67%, zig075 q080
~8.19%). Supersedes rerun_eth_greedy_router_regime_jmlam4_matched_20260809.py, which used the
buggy 172-feature bundles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import train_eval_omega1_2_tabm_diffusion_risk_20260603 as omega  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402
from replay_omega4_6_1_greedy_router_20260706 import (  # noqa: E402
    DURATION_THRESHOLD, greedy_replay, prepare_component,
)

OUT_DIR = ROOT / "tmp/eth_greedy_router_regime_jmlam4_pinned102_matched_20260809"
START, END = "2026-01-01", "2026-02-28"

JM_H48QUAL_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_h48qual_ext"
JM_H48QUAL_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q070_matched_20260809/risk_sidecar.pkl"
JM_ZIG075_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_pinned102_regime_jmlam4_20260809_zig075"
JM_ZIG075_SIDECAR = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_pinned102_jmlam4_q080_matched_20260809/risk_sidecar.pkl"
JM_REGIME3_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"

BASELINE_H48QUAL_PRED = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/oos_predictions_q050.csv"
BASELINE_ZIG075_PRED = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/zig075/oos_predictions_q075.csv"


def curve_metrics(returns: np.ndarray) -> dict:
    curve = np.concatenate([[1.0], np.cumprod(1.0 + returns)])
    peak = np.maximum.accumulate(curve)
    dd = curve / np.maximum(peak, 1e-12) - 1.0
    return {"pnl": round(float((curve[-1] - 1.0) * 100.0), 4),
            "mdd": round(float(dd.min() * 100.0), 4),
            "trades": int(len(returns)),
            "wr": round(float((returns > 0).mean()), 4) if len(returns) else 0.0}


def run_router(tag: str, use_jm: bool, common_ts: pd.Series) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = retest.DEVICE

    if use_jm:
        retest.WIDE24_2026 = JM_REGIME3_2026
        retest.COMPONENTS["h48qual"]["bundle"] = JM_H48QUAL_DIR / "true_3head_tabm_bundle.pt"
        retest.COMPONENTS["h48qual"]["sidecar_pkl"] = JM_H48QUAL_SIDECAR
        retest.COMPONENTS["zig075"]["bundle"] = JM_ZIG075_DIR / "true_3head_tabm_bundle.pt"
        retest.COMPONENTS["zig075"]["sidecar_pkl"] = JM_ZIG075_SIDECAR
        h48qual_pred_csv = JM_H48QUAL_DIR / "oos_predictions_q070.csv"
        zig075_pred_csv = JM_ZIG075_DIR / "oos_predictions_q080.csv"
    else:
        h48qual_pred_csv = BASELINE_H48QUAL_PRED
        zig075_pred_csv = BASELINE_ZIG075_PRED

    print(f"[{tag}] stage=load_frame_current", flush=True)
    frame = retest.load_frame_current(START, END)
    frame = frame[frame["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    fee, slip = omega._load_fee_slip()
    print(f"[{tag}] rows={len(frame)}", flush=True)

    components = {}
    for name, cfg, pred_csv in (("h48qual", retest.COMPONENTS["h48qual"], h48qual_pred_csv),
                                 ("zig075", retest.COMPONENTS["zig075"], zig075_pred_csv)):
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[pred["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(pred) != len(frame) or not pred["timestamp"].equals(frame["timestamp"]):
            raise RuntimeError(f"[{tag}] {name}: prediction rows = {len(pred)} != frame rows {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        pred.to_csv(tmp, index=False)
        components[name] = prepare_component(frame, tmp, cfg, device)
        print(f"[{tag}] {name}: prepared nonzero_side={(components[name]['dec']['side'] != 0).mean():.4f}", flush=True)

    print(f"[{tag}] stage=greedy_replay", flush=True)
    _, ledger = greedy_replay(frame, components, fee=fee, slip=slip, cost_mult=retest.COST_MULT, device=device)
    returns = ledger["trade_return"].to_numpy(dtype=float)
    no_gate = curve_metrics(returns)

    market = frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"})
    led = ledger.copy()
    led["entry_timestamp_dt"] = pd.to_datetime(led["entry_timestamp"])
    led = led.merge(market, on="entry_timestamp_dt", how="left")
    hit = led["ou_halflife"] <= DURATION_THRESHOLD
    gated = curve_metrics(led.loc[~hit, "trade_return"].to_numpy(dtype=float))
    gated["skipped"] = int(hit.sum())

    led.to_csv(OUT_DIR / f"ledger_{tag}.csv", index=False)
    return {"no_gate": no_gate, "with_gate": gated,
            "source_component_counts": ledger["source_component"].value_counts().to_dict()}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    frame_all = retest.load_frame_current(START, END)
    frame_ts = set(frame_all["timestamp"])
    jm_h48_ts = set(pd.to_datetime(pd.read_csv(JM_H48QUAL_DIR / "oos_predictions_q070.csv", usecols=["timestamp"])["timestamp"]))
    jm_zig_ts = set(pd.to_datetime(pd.read_csv(JM_ZIG075_DIR / "oos_predictions_q080.csv", usecols=["timestamp"])["timestamp"]))
    base_h48_ts = set(pd.to_datetime(pd.read_csv(BASELINE_H48QUAL_PRED, usecols=["timestamp"])["timestamp"]))
    base_zig_ts = set(pd.to_datetime(pd.read_csv(BASELINE_ZIG075_PRED, usecols=["timestamp"])["timestamp"]))
    common_ts = frame_ts & jm_h48_ts & jm_zig_ts & base_h48_ts & base_zig_ts
    print(f"common_ts={len(common_ts)} (frame={len(frame_ts)})", flush=True)
    common_ts_series = pd.Series(sorted(common_ts))

    result_baseline = run_router("baseline_wide24", use_jm=False, common_ts=common_ts_series)
    result_jm = run_router("jmlam4_pinned102_matched", use_jm=True, common_ts=common_ts_series)
    out = {
        "window": [START, END],
        "common_bars": len(common_ts),
        "note": "CORRECTED: both components pinned to live's 102-feature contract, regime3->JM, thresholds activation-matched (h48qual q070, zig075 q080).",
        "baseline_wide24_full_router": result_baseline,
        "jmlam4_pinned102_matched_full_router": result_jm,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
