#!/usr/bin/env python3
"""Visual verification for the FEEDER-ROLE-OPTIMIZED 8-trigger V자반등 label (data/labels/
eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/eth_5m_v_rebound_multitrigger_feeder_
optimized_labels.csv, built by build_eth_5m_v_rebound_multitrigger_labels_feeder_optimized_
20260901.py: orthogonal_combo dropped -- zero net-new contribution, see that script's docstring --
kalman_deviation_meanrev/demarker_extreme now cluster-deduped GAP=12 before unioning). Per project
convention (feedback_visual_verification_chart_gate_explain_before_proceed), this MUST be reviewed
and explicitly approved by the user before any feature-building/TabPFN-retraining step proceeds.

draw_candles/pick_per_trigger reused VERBATIM from render_eth_5m_v_rebound_multitrigger_20examples_
20260831.py -- only TRIGGER_ORDER (drops orthogonal_combo, puts the two actually-changed signals
first since those are what this pass needs to re-verify) and the I/O paths change. The other 6
triggers' own label formula/candidates are byte-identical to the already-approved 08-31 chart (not
re-verified here in depth) -- this pass exists specifically to check that deduping kalman/demarker
didn't introduce any spurious V자반등 calls.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
ETH_CSV = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
LABELS_CSV = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/eth_5m_v_rebound_multitrigger_feeder_optimized_labels.csv"
OUT_PATH = ROOT / "data/labels/eth_5m_v_rebound_multitrigger_feeder_optimized_20260901/multitrigger_feeder_optimized_24examples.png"
WINDOW_BARS = 12

TRIGGER_ORDER = ["kalman_deviation_meanrev", "demarker_extreme", "liquidity_sweep",
                 "taker_delta_z_climax", "short_term_return_z", "smt_divergence",
                 "fib_extension_exhaustion", "local_extreme"]
TRIGGER_LABEL_KR = {
    "kalman_deviation_meanrev": "kalman_dev_meanrev (GAP=12 dedup 신규적용)",
    "demarker_extreme": "demarker_extreme (GAP=12 dedup 신규적용)",
    "liquidity_sweep": "liquidity_sweep (변경없음)",
    "taker_delta_z_climax": "taker_delta_z_climax (변경없음)",
    "short_term_return_z": "short_term_return_z (변경없음)",
    "smt_divergence": "smt_divergence (변경없음)",
    "fib_extension_exhaustion": "fib_extension_exhaustion (변경없음)",
    "local_extreme": "local_extreme (변경없음)",
}


def draw_candles(ax, sub: pd.DataFrame, event_pos: int, event_level: float) -> None:
    lows, highs = [], []
    for i, (_, bar) in enumerate(sub.iterrows()):
        color = "#2E86AB" if bar["close"] >= bar["open"] else "#C73E1D"
        ax.plot([i, i], [bar["low"], bar["high"]], color=color, linewidth=1.1, zorder=3)
        body_low, body_high = sorted([bar["open"], bar["close"]])
        height = max(body_high - body_low, (bar["high"] - bar["low"]) * 0.03)
        ax.add_patch(Rectangle((i - 0.32, body_low), 0.64, height, facecolor=color, edgecolor=color, zorder=4))
        lows.append(bar["low"]); highs.append(bar["high"])
    ax.axhline(event_level, color="dimgray", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(event_pos, color="dimgray", linestyle=":", linewidth=1.1, zorder=1)
    pad = (max(highs) - min(lows)) * 0.08 or 1.0
    ax.set_ylim(min(lows) - pad, max(highs) + pad)
    ax.set_xlim(-0.6, len(sub) - 0.4)


def pick_per_trigger(labels: pd.DataFrame, trigger: str, n: int = 3) -> pd.DataFrame:
    pool = labels[(labels["triggers"].str.contains(trigger)) & (labels["outcome"] == "V자반등")]
    pool = pool.sort_values("giveback_ratio").reset_index(drop=True)
    if len(pool) <= n:
        return pool
    pcts = [0.15, 0.5, 0.85][:n]
    idxs = sorted({min(int(p * (len(pool) - 1)), len(pool) - 1) for p in pcts})
    return pool.iloc[idxs]


def main() -> int:
    frame = pd.read_csv(ETH_CSV, usecols=["timestamp", "open", "high", "low", "close"])
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    labels = pd.read_csv(LABELS_CSV)

    samples = []
    for trig in TRIGGER_ORDER:
        picked = pick_per_trigger(labels, trig, n=3)
        for _, row in picked.iterrows():
            samples.append((trig, row))

    n_cols = 3
    n_rows = len(TRIGGER_ORDER)
    plt.rcParams.update({"font.size": 11})
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 6.2 * n_rows), dpi=145)
    fig.suptitle(
        "V자반등 8트리거 feeder-role 최적화판 육안검증 (트리거별 3건, giveback percentile 15/50/85) — "
        "라벨식(v7b)은 그대로, orthogonal_combo 제거(순증분0건) + kalman/demarker GAP=12 dedup 신규적용. "
        "점선=이벤트시점, 회색가로선=이벤트 극값. 위 2행(kalman/demarker)이 이번에 실제로 바뀐 부분",
        fontsize=16, y=0.998,
    )

    for r, trig in enumerate(TRIGGER_ORDER):
        row_samples = [s for t, s in samples if t == trig]
        for c in range(n_cols):
            ax = axes[r][c]
            if c >= len(row_samples):
                ax.axis("off")
                continue
            event = row_samples[c]
            idx = int(event["idx"])
            is_down = event["direction"] == "downside"
            sub = frame.iloc[idx - WINDOW_BARS: idx + WINDOW_BARS + 1].reset_index(drop=True)
            level = float(frame["low"].iloc[idx]) if is_down else float(frame["high"].iloc[idx])
            draw_candles(ax, sub, WINDOW_BARS, level)
            ax.set_facecolor("#eef8f0")
            ticks = list(range(0, len(sub), 4))
            ax.set_xticks(ticks)
            ax.set_xticklabels([f"{(t - WINDOW_BARS) * 5:+d}" for t in ticks], fontsize=9)
            ax.tick_params(axis="y", labelsize=9)
            dir_kr = "상승반등(downside sweep류)" if is_down else "하락반전(upside류)"
            title = (f"{TRIGGER_LABEL_KR[trig] if c == 0 else trig}\n"
                     f"{dir_kr} | fast={event['fast_move_atr_mult']:.2f}x | giveback={event['giveback_ratio']:.3f}\n"
                     f"other_triggers={event['triggers']}")
            ax.set_title(title, fontsize=9.5)
            ax.grid(alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(OUT_PATH)
    print(f"saved: {OUT_PATH}  ({len(samples)} examples)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
