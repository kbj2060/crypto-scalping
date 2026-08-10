"""Diagnostic equity comparison chart for the regime-sizing overlay OOS read (2026-08-08).

Rebuilds gated equity curves from the saved OOS ledgers (identity vs czz_trend) written by
research_btc_swingtransition_regime_sizing_overlay_20260808.py.  Visualization of an
already-taken OOS read -- NOT evidence for any further selection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import LIVE_DURATION_THRESHOLD  # noqa: E402

OUT_DIR = ROOT / "tmp/btc_regime_sizing_overlay_20260808"
INK = "#1F2430"


def gated_equity(path: Path):
    led = pd.read_csv(path)
    led["entry_timestamp"] = pd.to_datetime(led["entry_timestamp"])
    led = led.loc[led["ou_halflife"] > LIVE_DURATION_THRESHOLD] if "ou_halflife" in led.columns else led
    led = led.sort_values("exit_timestamp" if "exit_timestamp" in led.columns else "entry_timestamp")
    eq = (1.0 + led["trade_return"]).cumprod()
    t = pd.to_datetime(led["exit_timestamp" if "exit_timestamp" in led.columns else "entry_timestamp"])
    return t, eq.to_numpy()


def main() -> int:
    # ledgers lack ou; regenerate gate by merging with states? ledgers were saved pre-gate.
    # The overlay script saved raw ledgers; recompute gate via ou merge is unavailable here,
    # so plot UNGATED equity (close to gated: 41/44 trades pass the gate) and label as such.
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, color in (("identity", "#9AA0A6"), ("czz_trend", "#2563EB")):
        led = pd.read_csv(OUT_DIR / f"oos_ledger_{name}.csv").sort_values("entry_timestamp")
        t = pd.to_datetime(led["entry_timestamp"])
        eq = (1.0 + led["trade_return"].astype(float)).cumprod()
        dd = (eq / eq.cummax() - 1.0).min() * 100
        ax.plot(t, eq, color=color, linewidth=1.6,
                label=f"{name}: {((eq.iloc[-1] - 1) * 100):+.1f}% (MDD {dd:.1f}%)")
    ax.axhline(1.0, color="#D0D3D8", linewidth=0.8)
    ax.set_title("BTC swingtransition OOS (2026) — parent sizing vs czz_trend regime overlay (ungated ledgers, diagnostic)",
                 loc="left", fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    out = OUT_DIR / "oos_equity_comparison.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
