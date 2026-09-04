#!/usr/bin/env python3
"""Does the OOS support-reversal / resistance paired-vs-aggregate contradiction from the dwell test
(eth_liquidation_map_dwell_duration_metric_rejected_20260825) concentrate in high-체결강도
(execution-strength / taker buy-sell skew) episodes -- 2026-08-25 follow-up to user's hypothesis:
"oos에 체결강도가 세서 지지/저항이 의미 없었던 날들 아니야?" (weren't there OOS days with strong
execution/order-flow imbalance where support/resistance was just meaningless?).

This is a genuinely different check from anything run today: it doesn't change the level formula
or the judgment metric again -- it partitions the SAME OOS snapshots (identical simulation, identical
dwell/placebo machinery, reused not reimplemented) by a REGIME tag and re-runs the identical
comparison within each partition. If the user's hypothesis is right, the OOS reversal should be
concentrated in the high-taker-skew half and weak/absent/reversed-back in the low half; if the
reversal shows up in BOTH halves about equally, regime isn't the explanation.

체결강도 operationalized directly: taker_skew = |taker_buy_base/volume - 0.5| per hourly bar (0 =
perfectly balanced execution, ->0.5 = one-sided). Each OOS snapshot is tagged by the MEAN taker_skew
over its own evaluation window [t0, t0+24) (1 day -- long enough to be representative, short enough
to stay local to the episode actually being scored; the dwell test's own median outcomes are 1-7h so
a 24h window comfortably covers where the actual touches/dwells happen), then split at the OOS
median into "high" vs "low" execution-strength halves.

Price/level data: base.load_hourly() + ed.simulate() called IDENTICALLY to the dwell test (same
function calls, same arguments) so the snapshot set is byte-for-byte the same population being
partitioned, not a fresh/possibly-drifted resimulation. Taker data: a SEPARATE fetch (research_eth_
liquidation_map_v2_cohort_ab_backtest_20260825.load_hourly_with_taker(), already built and validated
earlier today for this exact repo's OI-cohort work) joined onto df by exact timestamp match, not
assumed positionally aligned, since it comes from an independent fetch call.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_v2_cohort_ab_backtest_20260825 as v2ab
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell

TRAIN_FRACTION = 0.8
SEED = 20260825
REGIME_WINDOW_HOURS = 24


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    snapshots = ed.simulate(df)
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
          f"TRAIN={len(train_snaps)} OOS={len(oos_snaps)}", flush=True)

    taker_df = v2ab.load_hourly_with_taker()
    taker_df["taker_skew"] = (taker_df["taker_buy_base"] / taker_df["volume"] - 0.5).abs()
    skew = taker_df.set_index("timestamp")["taker_skew"].sort_index()

    def regime_tag(t0: int) -> float:
        start_ts = df["timestamp"].iloc[t0]
        end_i = min(n - 1, t0 + REGIME_WINDOW_HOURS)
        end_ts = df["timestamp"].iloc[end_i]
        window = skew.loc[(skew.index >= start_ts) & (skew.index < end_ts)]
        return float(window.mean()) if len(window) else float("nan")

    def stratify_and_report(split_name: str, snaps: list[dict], seed_base: int) -> None:
        tags = {s["t0"]: regime_tag(s["t0"]) for s in snaps}
        valid_tags = np.array([v for v in tags.values() if not np.isnan(v)])
        med = float(np.median(valid_tags))
        print(f"\n### {split_name}: {len(valid_tags)}/{len(snaps)} tagged, median taker_skew={med:.4f}", flush=True)
        high_snaps = [s for s in snaps if not np.isnan(tags[s["t0"]]) and tags[s["t0"]] >= med]
        low_snaps = [s for s in snaps if not np.isnan(tags[s["t0"]]) and tags[s["t0"]] < med]
        print(f"    high half: {len(high_snaps)}, low half: {len(low_snaps)}", flush=True)

        for label, sub_snaps, seed_off in ((f"{split_name} HIGH 체결강도 (top 50%)", high_snaps, seed_base + 1),
                                           (f"{split_name} LOW 체결강도 (bottom 50%)", low_snaps, seed_base + 2)):
            rng = np.random.default_rng(SEED + seed_off)
            res = dwell.evaluate_dwell(df, sub_snaps, rng)
            print(f"\n{'='*100}\n{label} -- n_snapshots={len(sub_snaps)}\n{'='*100}")
            for side in ("support", "resistance"):
                for buf in ("0.005", "0.001"):
                    d = res[side][buf]
                    rr, pp, pw = d["real"], d["placebo"], d["paired_outdwell"]
                    if not rr.get("n"):
                        print(f"  [{side} buf={float(buf)*100:.1f}%] n=0, skipped")
                        continue
                    print(f"  [{side} buf={float(buf)*100:.1f}%] "
                          f"real mean={rr['mean_dwell']:.2f}h placebo mean={pp['mean_dwell']:.2f}h  |  "
                          f"real surv1h={rr['survival_pct']['1']:.0f}% placebo surv1h={pp['survival_pct']['1']:.0f}%  |  "
                          f"paired winrate={pw['winrate']} ({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")

    stratify_and_report("TRAIN", train_snaps, 100)
    stratify_and_report("OOS", oos_snaps, 200)


if __name__ == "__main__":
    main()
