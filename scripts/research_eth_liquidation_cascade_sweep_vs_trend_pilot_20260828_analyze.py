#!/usr/bin/env python3
"""Class-wise (sweep vs continuation) comparison of the 4 feature axes, both event definitions,
across all 5 horizons. Descriptive pilot analysis -- no significance gate, per the design doc's
§7 (report direction-consistency across horizons, not p-values). Also renders 4 example charts
(2 sweep, 2 continuation) for eyeball verification per feedback_show_chart_before_parameter_decisions.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "research" / "eth_liquidation_cascade_sweep_vs_trend_pilot_20260828"

HORIZONS = ["15m", "30m", "1h", "2h", "4h"]
PRIMARY = "1h"

# feature -> (hypothesis direction if TRUE, "sweep expected higher" or "continuation expected higher")
NUMERIC_FEATURES = {
    "wick_body_ratio": "sweep_higher",
    "oi_pct_change_15m": "continuation_higher",
    "ls_ratio_shift_15m": None,  # sign depends on direction, handled separately via oi_same_dir_expansion
    "cvd_divergence_15m": "sweep_higher",
    "book_notional10_pct_change": "sweep_higher",
    "shadow_queue_collapse_max_15m": "continuation_higher",
    "shadow_absorption_score_mean_15m": "sweep_higher",
}
BOOL_FEATURES = {
    "oi_same_dir_expansion": "continuation_higher",
    "cvd_sign_agree_15m": "continuation_higher",
}


def compare_at_horizon(df: pd.DataFrame, horizon: str) -> dict:
    label_col = f"label_{horizon}"
    sub = df[df[label_col].isin(["sweep", "continuation"])]
    out = {"horizon": horizon, "n_sweep": int((sub[label_col] == "sweep").sum()),
           "n_continuation": int((sub[label_col] == "continuation").sum()), "features": {}}
    for feat, hyp in NUMERIC_FEATURES.items():
        g = sub.dropna(subset=[feat]).groupby(label_col)[feat]
        if "sweep" not in g.groups or "continuation" not in g.groups:
            continue
        sweep_mean, cont_mean = g.mean().get("sweep"), g.mean().get("continuation")
        sweep_med, cont_med = g.median().get("sweep"), g.median().get("continuation")
        n_s, n_c = g.count().get("sweep", 0), g.count().get("continuation", 0)
        direction_matches = None
        if hyp == "sweep_higher":
            direction_matches = bool(sweep_mean > cont_mean)
        elif hyp == "continuation_higher":
            direction_matches = bool(cont_mean > sweep_mean)
        out["features"][feat] = {
            "sweep_mean": round(float(sweep_mean), 5), "continuation_mean": round(float(cont_mean), 5),
            "sweep_median": round(float(sweep_med), 5), "continuation_median": round(float(cont_med), 5),
            "n_sweep": int(n_s), "n_continuation": int(n_c), "hypothesis": hyp,
            "direction_matches_hypothesis": direction_matches,
        }
    for feat, hyp in BOOL_FEATURES.items():
        g = sub.dropna(subset=[feat]).groupby(label_col)[feat]
        if "sweep" not in g.groups or "continuation" not in g.groups:
            continue
        sweep_rate, cont_rate = g.mean().get("sweep"), g.mean().get("continuation")
        direction_matches = bool(cont_rate > sweep_rate) if hyp == "continuation_higher" else bool(sweep_rate > cont_rate)
        out["features"][feat] = {
            "sweep_true_rate": round(float(sweep_rate), 4), "continuation_true_rate": round(float(cont_rate), 4),
            "n_sweep": int(g.count().get("sweep", 0)), "n_continuation": int(g.count().get("continuation", 0)),
            "hypothesis": hyp, "direction_matches_hypothesis": direction_matches,
        }
    return out


def consistency_summary(per_horizon: list[dict]) -> dict:
    all_feats = set()
    for h in per_horizon:
        all_feats.update(h["features"].keys())
    summary = {}
    for feat in sorted(all_feats):
        matches = [h["features"][feat]["direction_matches_hypothesis"] for h in per_horizon if feat in h["features"]]
        matches = [m for m in matches if m is not None]
        summary[feat] = {"horizons_matching": sum(matches), "horizons_total": len(matches),
                          "majority_consistent": bool(matches) and sum(matches) > len(matches) / 2}
    return summary


def plot_example(fut_kl: pd.DataFrame, oi_df: pd.DataFrame, ob_df: pd.DataFrame, event: pd.Series,
                  out_path: Path) -> None:
    t0 = pd.Timestamp(event["t0"])
    window = (fut_kl["timestamp"] >= t0 - pd.Timedelta(hours=4)) & (fut_kl["timestamp"] <= t0 + pd.Timedelta(hours=4))
    kl = fut_kl[window]
    fig, axes = plt.subplots(3, 1, figsize=(16, 11), sharex=True, dpi=140,
                              gridspec_kw={"height_ratios": [3, 1, 1]})
    ax = axes[0]
    ax.plot(kl["timestamp"], kl["close"], color="#2563eb", linewidth=1.3, label="close")
    ax.fill_between(kl["timestamp"], kl["low"], kl["high"], color="#93c5fd", alpha=0.35, label="high-low range")
    ax.axvline(t0, color="#dc2626", linestyle="--", linewidth=1.6, label="t0 (cascade)")
    ax.axhline(event["swept_level"], color="#16a34a", linestyle=":", linewidth=1.4, label="swept_level")
    ax.axhline(event["cascade_extreme"], color="#ea580c", linestyle=":", linewidth=1.4, label="cascade_extreme")
    ax.set_title(f"{event['event_id']} | {event['definition']} | direction={event['direction']} | "
                 f"label_1h={event['label_1h']} | z_peak={event['z_peak']:.2f}", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    ax.tick_params(labelsize=9)

    ax2 = axes[1]
    oi_win = oi_df[(oi_df["ts"] >= t0 - pd.Timedelta(hours=4)) & (oi_df["ts"] <= t0 + pd.Timedelta(hours=4))]
    ax2.plot(oi_win["ts"], oi_win["sum_open_interest"], color="#7c3aed", linewidth=1.3)
    ax2.axvline(t0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax2.set_ylabel("OI (contracts)", fontsize=9)
    ax2.tick_params(labelsize=9)

    ax3 = axes[2]
    ob_win = ob_df[(ob_df["recorded_at_kst"] >= t0 - pd.Timedelta(hours=4)) & (ob_df["recorded_at_kst"] <= t0 + pd.Timedelta(hours=4))]
    total_notional = ob_win["bid_notional_10"] + ob_win["ask_notional_10"]
    ax3.plot(ob_win["recorded_at_kst"], total_notional, color="#0891b2", linewidth=1.3, marker="o", markersize=3)
    ax3.axvline(t0, color="#dc2626", linestyle="--", linewidth=1.2)
    ax3.set_ylabel("top-10 book notional ($)", fontsize=9)
    ax3.tick_params(labelsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"saved chart: {out_path}")


def add_genuine_breach(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["genuine_breach"] = (
        ((df["direction"] == "down") & (df["cascade_extreme"] < df["swept_level"]))
        | ((df["direction"] == "up") & (df["cascade_extreme"] > df["swept_level"]))
    )
    return df


def main() -> None:
    df_a = pd.read_csv(DATA_DIR / "labeled_features_definition_a.csv", parse_dates=["t0"])
    df_b = pd.read_csv(DATA_DIR / "labeled_features_definition_b.csv", parse_dates=["t0"])
    df_a = add_genuine_breach(df_a)
    df_b = add_genuine_breach(df_b)

    report = {"definition_a": {}, "definition_b": {},
              "definition_a_genuine_breach_only": {}, "definition_b_genuine_breach_only": {}}
    datasets = {
        "definition_a": df_a, "definition_b": df_b,
        "definition_a_genuine_breach_only": df_a[df_a["genuine_breach"]],
        "definition_b_genuine_breach_only": df_b[df_b["genuine_breach"]],
    }
    for name, df in datasets.items():
        per_horizon = [compare_at_horizon(df, h) for h in HORIZONS]
        report[name]["per_horizon"] = per_horizon
        report[name]["consistency_summary"] = consistency_summary(per_horizon)
        report[name]["label_distribution_primary"] = df[f"label_{PRIMARY}"].value_counts().to_dict()
    report["definition_a_genuine_breach_rate"] = float(df_a["genuine_breach"].mean())
    report["definition_b_genuine_breach_rate"] = float(df_b["genuine_breach"].mean())

    if "liquidity_withdrawal_matched" in df_b.columns:
        sub = df_b[df_b[f"label_{PRIMARY}"].isin(["sweep", "continuation"])]
        report["definition_b"]["liquidity_withdrawal_matched_by_label"] = (
            sub.groupby(f"label_{PRIMARY}")["liquidity_withdrawal_matched"].mean().to_dict()
        )

    report_path = DATA_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"saved: {report_path}")

    print(f"\ngenuine_breach rate: definition_a={report['definition_a_genuine_breach_rate']:.1%}  "
          f"definition_b={report['definition_b_genuine_breach_rate']:.1%}")

    for name in ["definition_a", "definition_b", "definition_a_genuine_breach_only", "definition_b_genuine_breach_only"]:
        dist = report[name]["label_distribution_primary"]
        print(f"\n=== consistency summary, {name} (primary N: sweep={dist.get('sweep', 0)}, "
              f"continuation={dist.get('continuation', 0)}) ===")
        for feat, s in report[name]["consistency_summary"].items():
            flag = "MATCH" if s["majority_consistent"] else "no-match"
            print(f"  {feat:35s} {s['horizons_matching']}/{s['horizons_total']} horizons  [{flag}]")

    # example charts
    fut_kl = pd.read_csv(DATA_DIR / "futures_5m_klines.csv", parse_dates=["timestamp"])
    oi_df = pd.read_csv(DATA_DIR / "oi_lsratio_5m.csv", parse_dates=["ts"])
    oi_df["ts"] = pd.to_datetime(oi_df["ts"], utc=True)
    ob_df = pd.read_csv(DATA_DIR / "orderbook_decision_snapshots.csv", parse_dates=["recorded_at_kst"])
    ob_df["recorded_at_kst"] = pd.to_datetime(ob_df["recorded_at_kst"], utc=True)

    charts_dir = DATA_DIR / "charts"
    charts_dir.mkdir(exist_ok=True)
    genuine_a = df_a[df_a["genuine_breach"]]
    sweep_examples = genuine_a[genuine_a["label_1h"] == "sweep"].sort_values("z_peak", ascending=False).head(2)
    cont_examples = genuine_a[genuine_a["label_1h"] == "continuation"].sort_values("z_peak", ascending=False).head(2)
    for i, (_, ev) in enumerate(pd.concat([sweep_examples, cont_examples]).iterrows()):
        plot_example(fut_kl, oi_df, ob_df, ev, charts_dir / f"example_{i}_{ev['label_1h']}.png")


if __name__ == "__main__":
    main()
