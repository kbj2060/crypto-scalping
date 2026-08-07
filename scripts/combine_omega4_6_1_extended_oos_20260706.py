#!/usr/bin/env python3
"""Combine the extended h48qual+zig075 sized ledgers (from retest_omega4_6_1_extended_oos_20260706.py)
via the frozen Omega4.6 router (h48qual > zig075 priority, scale_map, leverage_cap=5.0,
notional_cap=1.8, live_risk_scale=1.0, NO max-hold time-stop -- matching
omega4_6_plus_t12_nohold_risk1_20260630's runtime_contract exactly), then apply the frozen
Omega4.6.1 duration gate rule (ou_halflife <= 0.005415348 -> notional/leverage scaled to 0).
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

OUT_DIR = ROOT / "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706"
BASE_2026 = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
SCALE_MAP = {"h48qual_L": 0.38, "h48qual_S": 2.499, "zig075_L": 2.446, "zig075_S": 2.478}
PRIORITY = ["h48qual", "zig075"]
LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE = 5.0, 1.8, 1.0
OU_HALFLIFE_THRESHOLD = 0.005415348


def load_market() -> pd.DataFrame:
    df = pd.read_csv(BASE_2026, usecols=["timestamp", "open", "high", "low", "close", "ou_halflife"], low_memory=False)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp").reset_index(drop=True)


def main() -> int:
    market = load_market()
    ledgers = []
    for name in PRIORITY:
        led = pd.read_csv(OUT_DIR / f"{name}_extended_sized_ledger.csv")
        led["source_alias"] = name
        ledgers.append(led)
    raw = pd.concat(ledgers, ignore_index=True)
    routed = builder.priority_route(raw, PRIORITY)
    scaled = builder.scale_ledger(routed, SCALE_MAP, LEVERAGE_CAP, NOTIONAL_CAP, LIVE_RISK_SCALE)
    # NO max-hold time stop applied (max_hold_hours=0 -> apply_max_hold_time_stop is a no-op),
    # matching omega4_6_plus_t12_nohold_risk1_20260630's runtime_contract (max_hold_hours: 0.0).
    combined = builder.apply_max_hold_time_stop(scaled, market[["timestamp", "open", "high", "low", "close"]], 0.0)
    combined_m = builder.metrics(combined)
    print("=== Combined router (h48qual>zig075), NO duration gate, extended Jan-Jun 2026 OOS ===", flush=True)
    print({k: v for k, v in combined_m.items() if k not in ("source_counts", "side_counts", "reason_counts")}, flush=True)

    # apply frozen duration gate: join ou_halflife at entry_timestamp
    combined["entry_timestamp_dt"] = pd.to_datetime(combined["entry_timestamp"])
    combined = combined.merge(
        market[["timestamp", "ou_halflife"]].rename(columns={"timestamp": "entry_timestamp_dt"}),
        on="entry_timestamp_dt", how="left",
    )
    if combined["ou_halflife"].isna().any():
        raise RuntimeError("missing ou_halflife join for some entries")
    combined["side"] = combined["side"].astype(int)
    combined["hold_hours"] = (pd.to_datetime(combined["exit_timestamp"]) - combined["entry_timestamp_dt"]).dt.total_seconds() / 3600.0
    rule = duration.Rule("duration_ou_halflife", "ou_halflife", "le", OU_HALFLIFE_THRESHOLD, "all", 0.0)
    gated = duration.apply_rule(combined, rule, leverage_cap=LEVERAGE_CAP, notional_cap=NOTIONAL_CAP)
    gated_m = duration.metrics(gated)
    print("\n=== + frozen duration gate (ou_halflife<=0.005415348 -> skip), extended Jan-Jun 2026 OOS ===", flush=True)
    print({k: v for k, v in gated_m.items()}, flush=True)

    combined.to_csv(OUT_DIR / "combined_router_ledger_extended.csv", index=False)
    gated.to_csv(OUT_DIR / "combined_router_duration_gated_ledger_extended.csv", index=False)

    # monthly breakdown for the final (gated) result
    active = gated[gated["notional"].astype(float) > 1e-12].copy()
    active["month"] = pd.to_datetime(active["entry_timestamp"]).dt.to_period("M").astype(str)
    print("\nmonthly (gated):", flush=True)
    for month, grp in active.groupby("month"):
        print(f"  {month}: trades={len(grp)} pnl_sum={grp['trade_return'].sum()*100:.1f}% wr={(grp['win']>0).mean():.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
