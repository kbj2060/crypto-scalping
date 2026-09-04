#!/usr/bin/env python3
"""Same dwell-duration test as research_eth_liquidation_map_dwell_duration_test_20260825.py, but
with the BREAK check switched from close-based to INTRABAR (high/low) -- 2026-08-25 user
correction: "종가 기준으로 청산을 잡지 말고 가격이 청산 라인만 닿아도 무조건 청산이잖아."

Real exchange liquidation engines trigger the instant mark price touches the liquidation price
(a resting order), not on candle close -- CLAUDE.md's own barrier-convention section makes exactly
this point for the live bot's own TP/SL: "실제 라이브 청산은 intrabar 고가/저가 기준... 이미
확정된 bar만 쓰므로 lookahead 아님." The ORIGINAL dwell test's close-based break check was
inherited from this session's earlier "근거리 스윕=진입타이밍, 종가이탈=무효화/반전" convention --
a discretionary CHART-READING framing (a wick sweep-and-reverse still counts as "the level held"),
which is a different question from "was this hypothetical liquidation actually triggered." This
script answers the latter, matching how compute_raw_bins() ALREADY drops a hypothetical position
once intrabar high/low crosses its liquidation price (future_min_low/future_max_high) -- so
generation and evaluation now share one consistent convention instead of two.

Only the break check changes (closes[i] -> lows[i]/highs[i] vs level_price*(1+/-buffer_pct));
touch detection was already intrabar (unchanged), placebo construction, paired/aggregate reporting,
TRAIN/OOS split, and event-driven level generation (ed.simulate(), unmodified) are all identical to
today's original dwell test, so results are directly comparable side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import scripts.research_eth_liquidation_map_support_resistance_backtest_20260824 as base
import scripts.research_eth_liquidation_map_event_driven_reset_20260824 as ed
import scripts.research_eth_liquidation_map_dwell_duration_test_20260825 as dwell

TRAIN_FRACTION = 0.8
SEED = 20260825


def dwell_bars_intrabar(lows: np.ndarray, highs: np.ndarray, touch_i: int, level_price: float,
                        side: str, buffer_pct: float) -> tuple[int, bool]:
    """Same offset/censoring convention as dwell.dwell_bars() (offset 0 = touch bar itself,
    inclusive; censored at dwell.DWELL_CAP_HOURS) -- only the break test uses intrabar low/high
    instead of close."""
    n = len(lows)
    end = min(n, touch_i + 1 + dwell.DWELL_CAP_HOURS)
    for i in range(touch_i, end):
        if side == "support" and lows[i] < level_price * (1 - buffer_pct):
            return i - touch_i, True
        if side == "resistance" and highs[i] > level_price * (1 + buffer_pct):
            return i - touch_i, True
    return end - touch_i, False


def evaluate_dwell_intrabar(df: pd.DataFrame, snapshots: list[dict], rng: np.random.Generator) -> dict:
    """Verbatim structure of dwell.evaluate_dwell(), with dwell_bars_intrabar substituted for
    dwell.dwell_bars() -- touch detection, placebo pool, agg()/paired() unchanged."""
    closes, lows, highs = df["close"].to_numpy(), df["low"].to_numpy(), df["high"].to_numpy()
    n = len(df)

    def find_touch(t0, level_price, side):
        fwd_end = min(n, t0 + 1 + base.FORWARD_HOURS)
        for i in range(t0 + 1, fwd_end):
            if side == "support" and lows[i] <= level_price:
                return i
            if side == "resistance" and highs[i] >= level_price:
                return i
        return None

    out = {}
    for side, key in (("support", "support_levels"), ("resistance", "resistance_levels")):
        pool = np.array([lv["distance_pct"] for s in snapshots for lv in s[key]])
        if not len(pool):
            pool = np.array([2.0, -2.0])

        real_by_buf = {b: [] for b in base.BUFFER_PCTS}
        placebo_by_buf = {b: [] for b in base.BUFFER_PCTS}

        for s in snapshots:
            cp = s["current_price"]
            for lv in s[key]:
                ti = find_touch(s["t0"], lv["price"], side)
                if ti is not None:
                    for buf in base.BUFFER_PCTS:
                        d, broke = dwell_bars_intrabar(lows, highs, ti, lv["price"], side, buf)
                        real_by_buf[buf].append((s["t0"], d, broke))
                pd_ = rng.choice(pool)
                pp = cp * (1 + pd_ / 100.0)
                ti2 = find_touch(s["t0"], pp, side)
                if ti2 is not None:
                    for buf in base.BUFFER_PCTS:
                        d, broke = dwell_bars_intrabar(lows, highs, ti2, pp, side, buf)
                        placebo_by_buf[buf].append((s["t0"], d, broke))

        def summarize(rows):
            if not rows:
                return {"n": 0}
            dwells = np.array([d for _, d, _ in rows])
            broke = np.array([b for _, _, b in rows])
            return {"n": len(rows), "mean_dwell": float(dwells.mean()), "median_dwell": float(np.median(dwells)),
                    "censored_pct": float((~broke).mean() * 100),
                    "survival_pct": {str(k): float((dwells >= k).mean() * 100) for k in dwell.SURVIVAL_CHECKPOINTS}}

        def paired(real_rows, placebo_rows):
            by_r, by_p = {}, {}
            for t0, d, _ in real_rows:
                by_r.setdefault(t0, []).append(d)
            for t0, d, _ in placebo_rows:
                by_p.setdefault(t0, []).append(d)
            fr = fp = tie = 0
            for t0 in set(by_r) & set(by_p):
                r, p = np.mean(by_r[t0]), np.mean(by_p[t0])
                fr += r > p
                fp += r < p
                tie += r == p
            return {"n_favor_real": fr, "n_favor_placebo": fp, "n_tie": tie,
                    "winrate": fr / (fr + fp) if (fr + fp) else None}

        out[side] = {str(buf): {"real": summarize(real_by_buf[buf]), "placebo": summarize(placebo_by_buf[buf]),
                                "paired_outdwell": paired(real_by_buf[buf], placebo_by_buf[buf])}
                    for buf in base.BUFFER_PCTS}
    return out


def summarize_split(split: str, snaps: list[dict], df: pd.DataFrame, seed_off: int) -> dict:
    rng = np.random.default_rng(SEED + seed_off)
    return {"split": split, "n_snapshots": len(snaps), "eval": evaluate_dwell_intrabar(df, snaps, rng)}


def main() -> None:
    df = base.load_hourly()
    print(f"hourly bars: {len(df)}, {df['timestamp'].iloc[0]} .. {df['timestamp'].iloc[-1]}", flush=True)
    print(f"break check: INTRABAR (low/high) vs level*(1+/-buffer) -- close-based comparison "
          f"available in research_eth_liquidation_map_dwell_duration_test_20260825.py", flush=True)

    snapshots = ed.simulate(df)
    n = len(df)
    split_i = int(n * TRAIN_FRACTION)
    train_snaps = [s for s in snapshots if s["t0"] < split_i]
    oos_snaps = [s for s in snapshots if s["t0"] >= split_i]
    print(f"split at bar {split_i} ({df['timestamp'].iloc[split_i]}) -- "
          f"TRAIN={len(train_snaps)}, OOS={len(oos_snaps)}", flush=True)

    results = [summarize_split("TRAIN", train_snaps, df, 0), summarize_split("OOS", oos_snaps, df, 1)]

    for r in results:
        print(f"\n{'='*100}\n{r['split']} (n_snapshots={r['n_snapshots']})\n{'='*100}")
        for side in ("support", "resistance"):
            for buf in ("0.005", "0.001"):
                d = r["eval"][side][buf]
                rr, pp, pw = d["real"], d["placebo"], d["paired_outdwell"]
                if not rr.get("n"):
                    print(f"  [{side} buf={float(buf)*100:.1f}%] n=0, skipped")
                    continue
                print(f"\n[{side} buf={float(buf)*100:.1f}%]")
                print(f"  real:    n={rr['n']:4d} mean_dwell={rr['mean_dwell']:5.2f}h "
                      f"median={rr['median_dwell']:4.1f}h censored={rr['censored_pct']:5.1f}%")
                print(f"  placebo: n={pp['n']:4d} mean_dwell={pp['mean_dwell']:5.2f}h "
                      f"median={pp['median_dwell']:4.1f}h censored={pp['censored_pct']:5.1f}%")
                surv_r = "  ".join(f"{k}h:{rr['survival_pct'][k]:.0f}%" for k in map(str, dwell.SURVIVAL_CHECKPOINTS))
                surv_p = "  ".join(f"{k}h:{pp['survival_pct'][k]:.0f}%" for k in map(str, dwell.SURVIVAL_CHECKPOINTS))
                print(f"  survival% real:    {surv_r}")
                print(f"  survival% placebo: {surv_p}")
                print(f"  paired out-dwell winrate: {pw['winrate']} ({pw['n_favor_real']}:{pw['n_favor_placebo']}, tie={pw['n_tie']})")


if __name__ == "__main__":
    main()
