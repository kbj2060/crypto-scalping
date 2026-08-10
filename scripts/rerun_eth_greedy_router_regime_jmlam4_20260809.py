"""ETH Omega4.6.1 FULL greedy router (h48qual + zig075) with h48qual's regime3-current input swapped
to the Statistical Jump Model (k=3, lambda=4) built in scripts/build_eth_regime3_jm_lam4_20260809.py,
h48qual's parent+sidecar retrained on it (train_eval_omega4_3head_..._eth_regime_jmlam4_20260809.py +
train_eval_omega4_2_risk_sidecar_eth_regime_jmlam4_20260809.py). zig075 is architecturally untouched
(its training script has zero "regime" references -- confirmed by grep -- so there is nothing to
swap there; it is included unmodified so the comparison is at the full Omega4.6.1 router level, not
just the h48qual component level docs42's 2026-07-21 test stopped at).

Window is capped at 2026-01-01..2026-02-28, NOT the full 2026-06-30 the honest-live rerun uses,
because h48qual's EVAL_CSV (train_eval_omega1_2_tabm_diffusion_risk_20260603.EVAL_CSV) only covers
that range -- discovered this session, not a choice. To keep the comparison fair, this script ALSO
runs the unmodified (live wide24) router over the IDENTICAL restricted window, so both rows are
apples-to-apples (neither is the +77.11%/6-month number, which is a different window).
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

OUT_DIR = ROOT / "tmp/eth_greedy_router_regime_jmlam4_20260809"
START, END = "2026-01-01", "2026-02-28"

JM_H48QUAL_DIR = ROOT / "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_regime_jmlam4_20260809"
JM_SIDECAR_PKL = ROOT / "tmp/causal_regen_20260516/omega4_2_trade_risk_sidecar_20260622_regime_jmlam4_q050_20260809/risk_sidecar.pkl"
JM_REGIME3_2026 = ROOT / "data/ensemble/supervised/eth_regime3_current_hmm_jmlam4_20260809_2026_maskedname.csv"


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
        retest.COMPONENTS["h48qual"]["sidecar_pkl"] = JM_SIDECAR_PKL
        h48qual_pred_csv = JM_H48QUAL_DIR / "oos_predictions_q050.csv"
    else:
        h48qual_pred_csv = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/oos_predictions_q050.csv"

    print(f"[{tag}] stage=load_frame_current", flush=True)
    frame = retest.load_frame_current(START, END)
    frame = frame[frame["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
    fee, slip = omega._load_fee_slip()
    print(f"[{tag}] rows={len(frame)} range=({frame['timestamp'].iloc[0]}, {frame['timestamp'].iloc[-1]})", flush=True)

    zig_cfg = retest.COMPONENTS["zig075"]
    zig_pred_csv = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/zig075/oos_predictions_q075.csv"

    components = {}
    for name, cfg, pred_csv in (("h48qual", retest.COMPONENTS["h48qual"], h48qual_pred_csv),
                                 ("zig075", zig_cfg, zig_pred_csv)):
        pred = pd.read_csv(pred_csv)
        pred["timestamp"] = pd.to_datetime(pred["timestamp"])
        pred = pred[pred["timestamp"].isin(common_ts)].sort_values("timestamp").reset_index(drop=True)
        if len(pred) != len(frame) or not pred["timestamp"].equals(frame["timestamp"]):
            raise RuntimeError(f"[{tag}] {name}: prediction rows covering window = {len(pred)} != frame rows {len(frame)}")
        tmp = OUT_DIR / f"_aligned_{tag}_{name}.csv"
        pred.to_csv(tmp, index=False)
        components[name] = prepare_component(frame, tmp, cfg, device)
        print(f"[{tag}] {name}: prepared nonzero_side={(components[name]['dec']['side'] != 0).mean():.3f}", flush=True)

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
    jm_pred_ts = set(pd.to_datetime(pd.read_csv(JM_H48QUAL_DIR / "oos_predictions_q050.csv", usecols=["timestamp"])["timestamp"]))
    baseline_h48_ts = set(pd.to_datetime(pd.read_csv(
        ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/h48qual/oos_predictions_q050.csv",
        usecols=["timestamp"])["timestamp"]))
    zig_ts = set(pd.to_datetime(pd.read_csv(
        ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/zig075/oos_predictions_q075.csv",
        usecols=["timestamp"])["timestamp"]))
    common_ts = frame_ts & jm_pred_ts & baseline_h48_ts & zig_ts
    print(f"common_ts={len(common_ts)} (frame={len(frame_ts)}, jm_missing={len(frame_ts - jm_pred_ts)})", flush=True)
    common_ts_series = pd.Series(sorted(common_ts))

    result_baseline = run_router("baseline_wide24", use_jm=False, common_ts=common_ts_series)
    result_jm = run_router("jmlam4", use_jm=True, common_ts=common_ts_series)
    out = {
        "window": [START, END],
        "common_bars": len(common_ts),
        "note": "window capped by h48qual's EVAL_CSV coverage; NOT comparable to the +77.11% 6-month figure. Both rows use the IDENTICAL common bar set.",
        "baseline_wide24_full_router": result_baseline,
        "jmlam4_h48qual_full_router": result_jm,
    }
    (OUT_DIR / "result.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
