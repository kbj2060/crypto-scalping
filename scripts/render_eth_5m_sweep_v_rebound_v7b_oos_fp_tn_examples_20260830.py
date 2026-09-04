#!/usr/bin/env python3
"""Companion to render_eth_5m_sweep_v_rebound_v7b_oos_correct_calls_20260830.py (which showed 10
true-positive OOS examples): now the other two quadrants the user asked to compare --
- FP (콜했는데 틀린 것): model called V자반등 (proba>=0.5) but ground truth is 지지/횡보 (label==0).
- TN (지지횡보로 콜하고 맞은 것): model called 지지/횡보 (proba<0.5) and ground truth agrees (label==0).

The saved v7b_costgate_candidates.pkl only has the "called" (proba>=0.5) subset, so it covers FP
but not TN. Re-scores the FULL OOS labeled population (both classes) against the same frozen v7b
TRAIN context to get proba for TN too. Uses 2 seeds (not the formal 4) since this is for visual
example-selection, not a new precision/AUC claim; the existing 4-seed VAL/OOS/holdout AUC numbers
are unaffected either way. Runs on the server GPU via handoff.sh (dev has no CUDA).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from tabpfn import TabPFNClassifier

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
LABEL_DIR = ROOT / "data/labels/eth_5m_sweep_v_rebound_20260829"
LABELS_CSV = LABEL_DIR / "eth_5m_sweep_v_rebound_labels.csv"
KLINES = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_sweep_v_rebound_label_v6_binary_20260830"
WINDOW_BARS = 12
OOS_START, OOS_END = pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-03-31 23:59:59", tz="UTC")
SEEDS = [20260829, 141592]  # reduced from the formal 4 -- example selection only, CPU box

TIER0 = [
    "is_downside", "sweep_penetration_atr", "atr", "atr_percentile_864",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "flow_aligned_delta_z",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z",
    "lower_wick_ratio", "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile",
]
FEATURES = TIER0 + ["rsi"]


def load_rsi() -> pd.DataFrame:
    frames = []
    for y in ("2024", "2025", "2026_rebuilt"):
        f = pd.read_csv(ROOT / f"data/splits/year_oos/training_features_{y}.csv", usecols=["timestamp", "rsi"])
        f["timestamp"] = pd.to_datetime(f["timestamp"], utc=True)
        frames.append(f)
    return pd.concat(frames, ignore_index=True).drop_duplicates("timestamp")


def draw_candles(ax, sub: pd.DataFrame, sweep_pos: int, sweep_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.3, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(sweep_level, color="dimgray", linestyle="--", linewidth=1.3, zorder=1)
    ax.axvline(sweep_pos, color="dimgray", linestyle=":", linewidth=1.3, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def render(sample: pd.DataFrame, labels: pd.DataFrame, kl: pd.DataFrame, ts_to_idx: pd.Series,
           title: str, out_path: Path) -> None:
    side_to_sweep = {"long": "downside", "short": "upside"}
    plt.rcParams.update({"font.size": 13})
    fig, axes = plt.subplots(2, 5, figsize=(32, 14), dpi=145)
    fig.suptitle(title, fontsize=20, y=1.0)

    for ax, (_, ev) in zip(axes.flatten(), sample.iterrows()):
        sweep_side = side_to_sweep[ev["side"]]
        row = labels[(labels["candidate_index"] == ev["candidate_index"]) & (labels["side"] == sweep_side)]
        sweep_level = float(row["sweep_level"].iloc[0])
        idx = int(ts_to_idx[ev["timestamp"]])
        sub = kl.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
        draw_candles(ax, sub, WINDOW_BARS, sweep_level)
        ax.set_facecolor("#fdf1ef")
        ticks = list(range(0, len(sub), 4))
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=10)
        ax.tick_params(axis="y", labelsize=10)
        ax.set_title(f"{ev['side']} | proba={ev['model_proba']:.2f}", fontsize=13)
        ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"saved: {out_path}")


def pick_by_percentile(pool: pd.DataFrame, pcts: list[float]) -> pd.DataFrame:
    sorted_pool = pool.sort_values("model_proba").reset_index(drop=True)
    idxs = sorted({min(int(p * (len(sorted_pool) - 1)), len(sorted_pool) - 1) for p in pcts})
    return sorted_pool.iloc[idxs]


def main() -> int:
    train = pd.read_csv(LABEL_DIR / "tabpfn_train_context_frozen_v7b_20260830.csv")
    tier0 = pd.read_csv(LABEL_DIR / "eth_5m_sweep_v_rebound_features_tier0_v7b_20260830.csv")
    tier0["timestamp"] = pd.to_datetime(tier0["timestamp"], utc=True)
    rsi = load_rsi()
    df = tier0.merge(rsi, on="timestamp", how="left").dropna(subset=FEATURES + ["label"]).reset_index(drop=True)
    ts = df["timestamp"]
    oos = df.loc[(ts >= OOS_START) & (ts <= OOS_END)].copy()
    print(f"OOS 전체 라벨población: {len(oos)}건 (label0={int((oos['label']==0).sum())}, label1={int((oos['label']==1).sum())})")

    probas = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[FEATURES], train["label"].to_numpy())
        probas.append(clf.predict_proba(oos[FEATURES])[:, 1])
        print(f"  seed={seed} done")
    oos["model_proba"] = np.mean(probas, axis=0)
    oos["side"] = np.where(oos["is_downside"] == 1, "long", "short")

    fp = oos[(oos["model_proba"] >= 0.5) & (oos["label"] == 0)].copy()
    tn = oos[(oos["model_proba"] < 0.5) & (oos["label"] == 0)].copy()
    print(f"FP(콜했는데 틀림, proba>=0.5&label==0): {len(fp)}건")
    print(f"TN(지지횡보콜 맞음, proba<0.5&label==0): {len(tn)}건")

    pcts = [0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    fp_sample = pick_by_percentile(fp, pcts)
    tn_sample = pick_by_percentile(tn, pcts)

    labels = pd.read_csv(LABELS_CSV, usecols=["candidate_index", "side", "sweep_level"])
    kl = pd.read_csv(KLINES, usecols=["timestamp", "open", "high", "low", "close"])
    kl["timestamp"] = pd.to_datetime(kl["timestamp"], utc=True)
    kl = kl.dropna().sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    ts_to_idx = pd.Series(kl.index.to_numpy(), index=kl["timestamp"].to_numpy())

    render(fp_sample, labels, kl, ts_to_idx,
           "v7b OOS 콜했는데 틀린 것(거짓양성: 모델 V자반등 콜 proba≥0.5 BUT 실제 라벨은 지지/횡보) 10건 -- "
           "확신도(proba) 5~95 percentile 순",
           OUT_DIR / "v7b_oos_false_positive_10.png")
    render(tn_sample, labels, kl, ts_to_idx,
           "v7b OOS 지지/횡보로 콜하고 맞은 것(진짜음성: 모델 proba<0.5 AND 실제 라벨도 지지/횡보) 10건 -- "
           "확신도(proba 낮은순=강한 지지횡보 확신) 5~95 percentile 순",
           OUT_DIR / "v7b_oos_true_negative_10.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
