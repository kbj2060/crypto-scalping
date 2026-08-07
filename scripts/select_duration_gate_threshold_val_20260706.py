#!/usr/bin/env python3
"""Re-select the Omega4.6.1 duration-gate ou_halflife threshold on VALIDATION data using the
CURRENT (post-fix) ou_halflife formula, since the frozen threshold (0.005415348) was implicitly
calibrated against the OLD funding-rate-based formula that features/elite.py's own code comment
says was buggy (constant ~1.0 halflife) and has since been fixed to use funding_roc_12 with a
5-day AR(1) window. Selection is VALIDATION-ONLY (2025-10-01..12-31, current-vintage
training_features_2025.csv + regime3 overlay), matching AGENTS.md's requirement that OOS remain a
readout, not a selection input. Reuses score_component() from
retest_omega4_6_1_extended_oos_20260706.py and the router-combine from
combine_omega4_6_1_extended_oos_20260706.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import build_omega_plus_t12_livepass_candidate_20260630 as builder  # noqa: E402
import eval_omega4_6_duration_aware_risk_layer_20260630 as duration  # noqa: E402
import retest_omega4_6_1_extended_oos_20260706 as retest  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
WIDE24_2025 = ROOT / "data/ensemble/supervised/regime3_current_hmm_sensitive_balancedish_20260530/training_features_2025_regime3_current_sensitive_hmm_wide24.csv"
BASE_2025 = ROOT / "data/splits/year_oos/training_features_2025.csv"
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ["h48qual", "zig075"]
LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE = 5.0, 1.8, 1.0
VAL_START, VAL_END = "2025-10-01", "2025-12-31"


def load_val_frame() -> pd.DataFrame:
    frame = pd.read_csv(BASE_2025, low_memory=False)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])
    frame = frame.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    overlay = pd.read_csv(WIDE24_2025, low_memory=False)
    overlay["timestamp"] = pd.to_datetime(overlay["timestamp"])
    cols = [c for c in overlay.columns if c != "timestamp"]
    frame = frame.merge(overlay[["timestamp", *cols]], on="timestamp", how="left", validate="one_to_one")
    frame = frame[(frame["timestamp"] >= VAL_START) & (frame["timestamp"] <= VAL_END)].reset_index(drop=True)
    return frame


def main() -> int:
    val_frame = load_val_frame()
    print(f"VAL frame: {len(val_frame)} rows {val_frame['timestamp'].min()}..{val_frame['timestamp'].max()}", flush=True)

    ledgers = []
    for name, cfg in retest.COMPONENTS.items():
        orig_dir = ROOT / (
            "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_zigzagfix_06_h48_quality_noctx_padded_e2_fulltrain_exit30k_20260630"
            if name == "h48qual" else
            "tmp/causal_regen_20260516/omega4_3head_parent72_loose_entry_quality_20260620_current_only_alllabels_01_zigzag_action_labels_20260531_e2_fulltrain_exit30k_20260629"
        )
        val_pred = orig_dir / f"validation_predictions_{cfg['q_tag']}.csv"
        m, led = retest.score_component(val_frame, val_pred, cfg, prefix=f"{name}_val", oof=True)
        led["source_alias"] = name
        ledgers.append(led)
        print(f"{name} VAL component: pnl={m['pnl']:.2f}% mdd={m['mdd']:.2f}% trades={m['trades']} wr={m['wr']:.3f}", flush=True)

    raw = pd.concat(ledgers, ignore_index=True)
    routed = builder.priority_route(raw, PRIORITY)
    scaled = builder.scale_ledger(routed, SCALE_MAP, LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE)
    combined = builder.apply_max_hold_time_stop(scaled, val_frame[["timestamp", "open", "high", "low", "close"]], 0.0)
    combined_m = builder.metrics(combined)
    print(f"\nVAL combined router (no gate): pnl={combined_m['pnl']:.2f}% mdd={combined_m['mdd']:.2f}% trades={combined_m['trades']} wr={combined_m['wr']:.3f}", flush=True)
    combined.to_csv(OUT_DIR / "combined_router_ledger_VAL.csv", index=False)

    # grid search ou_halflife thresholds on VAL using duration_priority_score-style objective
    active = combined[combined["notional"].astype(float) > 1e-12].copy()
    active["entry_timestamp_dt"] = pd.to_datetime(active["entry_timestamp"])
    active = active.merge(val_frame[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"}), on="entry_timestamp_dt", how="left")
    if active["ou_halflife"].isna().any():
        raise RuntimeError("missing ou_halflife join")
    active["hold_hours"] = (pd.to_datetime(active["exit_timestamp"]) - active["entry_timestamp_dt"]).dt.total_seconds() / 3600.0
    active["entry_month"] = active["entry_timestamp_dt"].dt.to_period("M").astype(str)

    baseline_m = duration.metrics(active.assign(duration_risk_hit=False, duration_risk_skipped=False))
    print(f"\nVAL baseline (no duration filter): pnl={baseline_m['pnl']:.2f}% mdd={baseline_m['mdd']:.2f}% trades={baseline_m['trades']}", flush=True)

    quantiles = np.arange(0.05, 0.85, 0.05)
    thresholds = sorted(set(float(active["ou_halflife"].quantile(q)) for q in quantiles))
    rows = []
    for thr in thresholds:
        hit = active["ou_halflife"] <= thr
        gated = active.copy()
        gated.loc[hit, "notional"] = 0.0
        gated["duration_risk_hit"] = hit
        gated["duration_risk_skipped"] = hit
        gated["trade_return"] = np.where(hit, 0.0, gated["trade_return"])
        m = duration.metrics(gated)
        month = duration.monthly_summary(gated)
        score = duration.duration_priority_score(m, month, baseline_m)
        rows.append({"threshold": thr, "pnl": m["pnl"], "mdd": m["mdd"], "trades": m["trades"], "wr": m["wr"],
                     "hit_count": int(hit.sum()), "monthly_min_pnl": month["monthly_min_pnl"], "score": score})
    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(OUT_DIR / "duration_threshold_val_selection.csv", index=False)
    print("\n=== VAL threshold grid (top 10 by duration_priority_score) ===", flush=True)
    print(df.head(10).to_string(index=False), flush=True)

    valid = df[(df["mdd"] >= -20.0) & (df["trades"] >= int(0.65 * baseline_m["trades"]))]
    if valid.empty:
        print("\nNo threshold passes the MDD/trade-count gate; falling back to baseline (no gate).", flush=True)
        selected = None
    else:
        selected = valid.sort_values("score", ascending=False).iloc[0]
        print(f"\nSELECTED threshold: {selected['threshold']:.6f} (VAL pnl={selected['pnl']:.2f}%, mdd={selected['mdd']:.2f}%, trades={selected['trades']}, hit_count={selected['hit_count']})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
