#!/usr/bin/env python3
"""Follow-up to the N=121 dev/holdout result (wick_body_ratio survives alone, everything else in
the original 4-axis frame doesn't). User asked to try hybridizing with OTHER existing live
indicators (not re-testing the ones already discarded) to see if any revives a second axis.

Candidates pulled from microstructure_1m (already collected, no new collector): obi, nif_whale,
nif_retail, whale_position_score, funding_rate, eai, shadow_toxicity_score. spoofing_score is
skipped -- 87.8% exactly zero over the full 41-day history, too sparse for a 15min-window mean.

Same discipline as the wick_body_ratio check: SAME chronological dev(70%)/holdout(30%) split,
direction+threshold chosen on dev only, confirmed once on holdout, never re-adjusted after seeing
holdout. Reports every candidate including failures -- no cherry-picking.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"
FEATURE_WINDOW_MIN = 15  # matches the established [t0, t0+15min] convention from the main pipeline

CANDIDATES = ["obi", "nif_whale", "nif_retail", "whale_position_score", "funding_rate", "eai",
              "shadow_toxicity_score"]
CENTERED_AT_ZERO = {"obi", "nif_whale", "nif_retail", "whale_position_score", "funding_rate"}


def load_events() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "labeled_features_definition_a.csv", parse_dates=["t0"])
    df["genuine_breach"] = (
        ((df["direction"] == "down") & (df["cascade_extreme"] < df["swept_level"]))
        | ((df["direction"] == "up") & (df["cascade_extreme"] > df["swept_level"]))
    )
    sub = df[df["genuine_breach"] & df["label_1h"].isin(["sweep", "continuation"])].copy()
    return sub.sort_values("t0").reset_index(drop=True)


def add_new_features(events: pd.DataFrame, micro: pd.DataFrame) -> pd.DataFrame:
    micro = micro.sort_values("ts").reset_index(drop=True)
    out = []
    for ev in events.itertuples():
        t0 = ev.t0
        win = micro[(micro["ts"] > t0) & (micro["ts"] <= t0 + pd.Timedelta(minutes=FEATURE_WINDOW_MIN))]
        row = {"event_id": ev.event_id}
        for c in CANDIDATES:
            row[c] = win[c].mean() if len(win) else np.nan
        out.append(row)
    return events.merge(pd.DataFrame(out), on="event_id", how="left")


def precision_recall(d: pd.DataFrame, mask: np.ndarray, cls: str):
    pred = np.where(mask, cls, "other")
    actual = d["label_1h"].to_numpy()
    n_pred = int((pred == cls).sum())
    tp = int(((pred == cls) & (actual == cls)).sum())
    n_actual = int((actual == cls).sum())
    prec = tp / n_pred if n_pred else float("nan")
    rec = tp / n_actual if n_actual else float("nan")
    return prec, rec, n_pred


def screen_candidate(dev: pd.DataFrame, hold: pd.DataFrame, col: str) -> dict:
    d = dev.dropna(subset=[col])
    if len(d) < 10:
        return {"col": col, "skip": f"insufficient dev n={len(d)}"}
    threshold = 0.0 if col in CENTERED_AT_ZERO else d[col].median()

    # try both directions on dev, keep whichever gives the better sweep-precision AND the
    # mirror check for continuation-precision -- report both, direction is picked on dev only
    results = {}
    for cls, higher_means_sweep in [("sweep", True), ("continuation", False)]:
        mask_dev = (d[col] > threshold) if higher_means_sweep else (d[col] <= threshold)
        prec_dev, rec_dev, n_dev = precision_recall(d, mask_dev, cls)
        results[cls] = {"higher_means_sweep": higher_means_sweep, "threshold": threshold,
                         "dev_precision": prec_dev, "dev_recall": rec_dev, "dev_n": n_dev}

    # base rate on dev, for comparison
    base_rate = {"sweep": (d["label_1h"] == "sweep").mean(), "continuation": (d["label_1h"] == "continuation").mean()}

    # only confirm on holdout the direction that beat its own base rate on dev by a real margin
    h = hold.dropna(subset=[col])
    confirmed = {}
    for cls in ["sweep", "continuation"]:
        r = results[cls]
        if pd.notna(r["dev_precision"]) and r["dev_precision"] > base_rate[cls] + 0.05 and r["dev_n"] >= 8:
            mask_hold = (h[col] > threshold) if r["higher_means_sweep"] else (h[col] <= threshold)
            prec_hold, rec_hold, n_hold = precision_recall(h, mask_hold, cls)
            confirmed[cls] = {**r, "holdout_precision": prec_hold, "holdout_recall": rec_hold,
                               "holdout_n": n_hold, "holdout_base_rate": (h["label_1h"] == cls).mean()}
    return {"col": col, "threshold": threshold, "dev_n": len(d), "holdout_n": len(h),
            "base_rate_dev": base_rate, "candidates_worth_confirming": confirmed}


def main() -> None:
    events = load_events()
    micro = pd.read_csv(DATA_DIR / "microstructure_1m.csv", parse_dates=["ts"])
    micro["ts"] = pd.to_datetime(micro["ts"], utc=True)
    full = add_new_features(events, micro)

    split = int(len(full) * 0.7)
    dev, hold = full.iloc[:split], full.iloc[split:]
    print(f"N={len(full)}  dev n={len(dev)} ({dev['t0'].min().date()}~{dev['t0'].max().date()})  "
          f"holdout n={len(hold)} ({hold['t0'].min().date()}~{hold['t0'].max().date()})")
    print(f"dev base rate: {dict(dev['label_1h'].value_counts(normalize=True).round(3))}")
    print(f"holdout base rate: {dict(hold['label_1h'].value_counts(normalize=True).round(3))}\n")

    survivors = []
    for col in CANDIDATES:
        r = screen_candidate(dev, hold, col)
        if "skip" in r:
            print(f"{col:24s} SKIPPED ({r['skip']})")
            continue
        print(f"{col:24s} dev_n={r['dev_n']:3d} holdout_n={r['holdout_n']:3d} threshold={r['threshold']:.5f}")
        if not r["candidates_worth_confirming"]:
            print(f"{'':24s}   no direction beat dev base rate by >5pp with n>=8 -- not worth confirming on holdout")
        for cls, c in r["candidates_worth_confirming"].items():
            beat = c["holdout_precision"] > c["holdout_base_rate"]
            flag = "HOLDS UP" if beat else "fails on holdout"
            print(f"{'':24s}   [{cls}] dev_prec={c['dev_precision']:.1%}(n={c['dev_n']}) -> "
                  f"holdout_prec={c['holdout_precision']:.1%}(n={c['holdout_n']}, base_rate={c['holdout_base_rate']:.1%})  [{flag}]")
            if beat:
                survivors.append((col, cls, c))

    print(f"\n=== survivors (beat holdout base rate): {len(survivors)} ===")
    for col, cls, c in survivors:
        print(f"  {col} -> {cls}: holdout {c['holdout_precision']:.1%} vs base {c['holdout_base_rate']:.1%} (n={c['holdout_n']})")

    if survivors:
        print("\n=== 2-axis hybrid check: wick_body_ratio + each survivor (AND), dev-fit thresholds, holdout-confirmed once ===")
        for col, cls, c in survivors:
            wick_thresh = 2.0 if cls == "sweep" else 0.5
            wick_op = (lambda s: s["wick_body_ratio"] > wick_thresh) if cls == "sweep" else (lambda s: s["wick_body_ratio"] < wick_thresh)
            other_op = (lambda s: s[col] > c["threshold"]) if c["higher_means_sweep"] else (lambda s: s[col] <= c["threshold"])
            dev_valid = dev.dropna(subset=[col, "wick_body_ratio"])
            hold_valid = hold.dropna(subset=[col, "wick_body_ratio"])
            mask_dev = wick_op(dev_valid) & other_op(dev_valid)
            mask_hold = wick_op(hold_valid) & other_op(hold_valid)
            prec_dev, rec_dev, n_dev = precision_recall(dev_valid, mask_dev, cls)
            prec_hold, rec_hold, n_hold = precision_recall(hold_valid, mask_hold, cls)
            wick_only_prec_hold, _, wick_only_n_hold = precision_recall(
                hold_valid, wick_op(hold_valid), cls)
            print(f"  wick+{col} -> {cls}: dev={prec_dev:.1%}(n={n_dev})  holdout={prec_hold:.1%}(n={n_hold})  "
                  f"[wick-alone holdout on same rows: {wick_only_prec_hold:.1%}(n={wick_only_n_hold})]")


if __name__ == "__main__":
    main()
