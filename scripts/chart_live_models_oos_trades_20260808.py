"""OOS trade charts for every model currently wired into the live/shadow stack (2026-08-08).

One panel per model: cumulative gated equity over the OOS window, with each trade marked at its
ENTRY timestamp (blue = long, red = short, hollow = losing trade).  Below each, the per-trade
return bars so trade-level dispersion is visible rather than hidden inside the equity curve.

Ledger sources are the artifacts the live manifest actually points at, not re-derived numbers:
  ETH  omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv
       (the +145.34% headline's own ledger -- see the audit note: this one carries NO
        fresh-forward flags and is a LEDGER-ROUTING composition, and the number no longer
        reproduces on current data)
  SOL  sol_final_scale_map_adaptive_squeeze_20260720/oos_ledger.csv
  BTC  btc_final_scale_map_swingtransition_freshforward_ext_20260806/oos_ledger.csv
  BTC multislot shadow (N=3 x1.5) and the same + czz_trend regime sizing overlay, from
       tmp/btc_multislot_shadow_regime_sizing_20260808/ (today's full causal re-replay)

The duration gate is applied where the ledger carries ou_halflife, matching how each model's
headline metric is computed; the applied convention is printed per panel.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from research_btc_swingtransition_trailing_stop_val_oos_20260807 import LIVE_DURATION_THRESHOLD  # noqa: E402

OUT_DIR = ROOT / "tmp/live_models_oos_20260808"
INK, C_LONG, C_SHORT, C_EQ = "#1F2430", "#2563EB", "#D9542B", "#0E7C66"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False

MODELS = [
    ("ETH Omega4.6.1 (greedy router)",
     "tmp/causal_regen_20260516/omega4_6_1_extended_oos_20260706/greedy_router_ledger_extended.csv", None),
    ("SOL zig075 v2 (adaptive_squeeze)",
     "tmp/causal_regen_20260516/sol_final_scale_map_adaptive_squeeze_20260720/oos_ledger.csv", "sol"),
    ("BTC h48qual+swingtransition (승격 라이브)",
     "tmp/causal_regen_20260516/btc_final_scale_map_swingtransition_freshforward_ext_20260806/oos_ledger.csv", "btc"),
    ("BTC 멀티슬롯 섀도 N=3 x1.5",
     "tmp/btc_multislot_shadow_regime_sizing_20260808/oos_ledger_n3_identity.csv", "btc"),
    ("BTC 멀티슬롯 + czz_trend 리스크 오버레이",
     "tmp/btc_multislot_shadow_regime_sizing_20260808/oos_ledger_n3_czz_trend.csv", "btc"),
]


def load(path: Path, gate_asset: str | None):
    d = pd.read_csv(path)
    d["entry_timestamp"] = pd.to_datetime(d["entry_timestamp"])
    note = "게이트 없음"
    if gate_asset and "ou_halflife" in d.columns:
        before = len(d)
        d = d.loc[d["ou_halflife"] > LIVE_DURATION_THRESHOLD]
        note = f"duration gate ou>{LIVE_DURATION_THRESHOLD:.6f} ({before}→{len(d)})"
    elif gate_asset:
        note = "원장에 ou_halflife 없음 (게이트 미적용)"
    return d.sort_values("entry_timestamp").reset_index(drop=True), note


def metrics(r: np.ndarray):
    if len(r) == 0:
        return 0.0, 0.0, None
    eq = np.cumprod(1.0 + r)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)
    return float((eq[-1] - 1.0) * 100), mdd, eq


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    fig, axes = plt.subplots(len(MODELS), 2, figsize=(16, 3.0 * len(MODELS)),
                             gridspec_kw={"width_ratios": [3, 1.5], "hspace": 0.55, "wspace": 0.18})
    for i, (name, rel, gate) in enumerate(MODELS):
        p = ROOT / rel
        if not p.exists():
            axes[i, 0].text(0.5, 0.5, f"{name}\n원장 없음: {rel}", ha="center", fontsize=9)
            axes[i, 0].axis("off"); axes[i, 1].axis("off")
            continue
        d, note = load(p, gate)
        r = d["trade_return"].to_numpy(dtype=float)
        pnl, mdd, eq = metrics(r)
        ts = d["entry_timestamp"]
        rows.append({"model": name, "ledger": rel, "gate": note, "trades": int(len(r)),
                     "pnl_pct": round(pnl, 2), "mdd_pct": round(mdd, 2),
                     "win_rate": round(float((r > 0).mean()), 3) if len(r) else None,
                     "window": [str(ts.min()), str(ts.max())] if len(r) else None})

        ax = axes[i, 0]
        if eq is not None:
            ax.plot(ts, eq, color=C_EQ, linewidth=1.6, zorder=2)
            for t, ret, side in zip(ts, r, d.get("side", pd.Series([1] * len(d)))):
                c = C_LONG if int(side) > 0 else C_SHORT
                ax.axvline(t, color=c, alpha=0.30 if ret > 0 else 0.55,
                           linewidth=1.6 if ret <= 0 else 0.9, zorder=1)
            ax.axhline(1.0, color="#9AA0A6", linewidth=0.9, linestyle="--")
        ax.set_title(f"{name}  —  OOS {len(r)}거래  PnL {pnl:+.2f}%  MDD {mdd:.2f}%   [{note}]",
                     loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel("누적 equity", fontsize=8.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)

        axb = axes[i, 1]
        cols = [C_LONG if v > 0 else C_SHORT for v in r]
        axb.bar(np.arange(len(r)), r * 100, color=cols)
        axb.axhline(0, color=INK, linewidth=0.8)
        axb.set_title("거래별 수익률 %", loc="left", fontsize=9, color=INK)
        axb.set_xlabel("거래 순번", fontsize=8)
        for s in ("top", "right"):
            axb.spines[s].set_visible(False)
        axb.grid(axis="y", color="#000000", alpha=0.07, linewidth=0.8)

    fig.suptitle("라이브/섀도 모델별 OOS 거래 (파랑=롱, 빨강=숏, 진한 선=손실 거래)",
                 fontsize=13, y=0.995)
    out = OUT_DIR / "live_models_oos_trades.png"
    fig.savefig(out, dpi=125, bbox_inches="tight", facecolor="white")
    (OUT_DIR / "summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
