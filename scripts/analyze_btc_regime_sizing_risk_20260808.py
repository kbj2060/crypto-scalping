"""Risk-first evaluation of the regime sizing overlay maps (2026-08-08).

Contract: docs/experiments/btc_regime_sizing_risk_first_20260808.json.
Reads the ledgers written by research_btc_swingtransition_regime_sizing_overlay_20260808.py
--stage risk (every map on both splits) and applies the pre-registered risk gates.

Evidence is deliberately NOT a single point estimate: with ~51 gated trades in total, one window's
drawdown is noise.  Two views are combined instead --
  consistency  MDD per natural calendar period (VAL 2025Q4, OOS 2026Q1, Q2, Q3-partial)
  bootstrap    stationary block bootstrap over the trade sequence (B=2000, mean block 5) giving
               P(map MDD better than identity) and the Calmar distribution
Neither manufactures new out-of-sample evidence; they measure how much of the observed risk
difference survives resampling and how consistently it shows up across periods.
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

LEDGER_DIR = ROOT / "tmp/btc_regime_sizing_overlay_20260808"
OUT_DIR = ROOT / "tmp/btc_regime_sizing_risk_20260808"
CONTRACT = ROOT / "docs/experiments/btc_regime_sizing_risk_first_20260808.json"
MAPS = ["identity", "jm_trend", "jm_skip_bear", "jm_contra", "czz_trend", "czz_consensus", "jm_consensus"]
B_BOOT, BLOCK_DAYS = 2000, 21
SEED = 903174
GATE_PERIODS, GATE_BOOT, GATE_PNL_RETAIN, GATE_TRADE_TOL = 3, 0.70, 0.60, 0.20
INK, C_BASE, C_GOOD, C_BAD = "#1F2430", "#9AA0A6", "#0E7C66", "#D9542B"

KOREAN_FONT = Path("/mnt/c/Windows/Fonts/malgun.ttf")
if KOREAN_FONT.exists():
    fm.fontManager.addfont(str(KOREAN_FONT))
    plt.rcParams["font.family"] = fm.FontProperties(fname=str(KOREAN_FONT)).get_name()
plt.rcParams["axes.unicode_minus"] = False


def load_ledger(name: str) -> pd.DataFrame:
    parts = []
    for split, fn in (("validation", f"validation_ledger_{name}.csv"), ("oos", f"oos_ledger_{name}.csv")):
        p = LEDGER_DIR / fn
        if not p.exists():
            raise FileNotFoundError(p)
        d = pd.read_csv(p)
        d["split"] = split
        parts.append(d)
    led = pd.concat(parts, ignore_index=True)
    led["entry_timestamp"] = pd.to_datetime(led["entry_timestamp"])
    if "ou_halflife" in led.columns:
        led = led.loc[led["ou_halflife"] > LIVE_DURATION_THRESHOLD]
    return led.sort_values("entry_timestamp").reset_index(drop=True)


def path_metrics(rets: np.ndarray) -> dict:
    if len(rets) == 0:
        return {"pnl": 0.0, "mdd": 0.0, "calmar": 0.0, "n": 0}
    eq = np.cumprod(1.0 + rets)
    mdd = float((eq / np.maximum.accumulate(eq) - 1.0).min() * 100)
    pnl = float((eq[-1] - 1.0) * 100)
    return {"pnl": round(pnl, 2), "mdd": round(mdd, 2),
            "calmar": round(pnl / abs(mdd), 2) if mdd < -1e-9 else None, "n": int(len(rets))}


def paired_time_block_bootstrap(ledgers: dict, block_days: int, rng, b: int = B_BOOT):
    """PAIRED bootstrap: resample contiguous CALENDAR blocks and evaluate every map on the SAME
    block draw.  Pairing by time is what makes the comparison powerful here -- the maps share
    entries and differ only in sizing, so an unpaired comparison of two independent resamples
    throws away exactly the information that matters.  Trade counts differ slightly between maps
    (sizing feeds the exit head), which is why blocks are calendar-based rather than trade-based."""
    all_ts = pd.concat([d["entry_timestamp"] for d in ledgers.values()])
    t0, t1 = all_ts.min(), all_ts.max()
    edges = pd.date_range(t0, t1 + pd.Timedelta(days=block_days), freq=f"{block_days}D")
    blocks = list(zip(edges[:-1], edges[1:]))
    per_map_blocks = {m: [d.loc[(d["entry_timestamp"] >= a) & (d["entry_timestamp"] < z),
                                "trade_return"].to_numpy(dtype=float) for a, z in blocks]
                      for m, d in ledgers.items()}
    n_blocks = len(blocks)
    out = {m: np.empty(b) for m in ledgers}
    for i in range(b):
        pick = rng.integers(n_blocks, size=n_blocks)
        for m in ledgers:
            r = np.concatenate([per_map_blocks[m][k] for k in pick]) if n_blocks else np.array([])
            if len(r) == 0:
                out[m][i] = 0.0
                continue
            eq = np.cumprod(1.0 + r)
            out[m][i] = (eq / np.maximum.accumulate(eq) - 1.0).min() * 100
    return out, n_blocks


def main() -> int:
    argparse.ArgumentParser().parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    contract = json.loads(CONTRACT.read_text())
    rng = np.random.default_rng(SEED)

    ledgers = {m: load_ledger(m) for m in MAPS}
    base = ledgers["identity"]
    base["period"] = base["entry_timestamp"].dt.to_period("Q").astype(str)
    periods = sorted(base["period"].unique())
    print(json.dumps({"periods": periods, "identity_trades": int(len(base))}), flush=True)

    boot, n_blocks = paired_time_block_bootstrap(ledgers, BLOCK_DAYS, rng)
    base_boot = boot["identity"]
    print(json.dumps({"paired_bootstrap": {"block_days": BLOCK_DAYS, "n_blocks": n_blocks,
                                           "B": B_BOOT}}), flush=True)

    report = {"contract": str(CONTRACT.relative_to(ROOT)), "periods": periods, "maps": {}}
    for m in MAPS:
        led = ledgers[m].copy()
        led["period"] = led["entry_timestamp"].dt.to_period("Q").astype(str)
        full = path_metrics(led["trade_return"].to_numpy(dtype=float))
        per = {}
        for p in periods:
            per[p] = path_metrics(led.loc[led["period"] == p, "trade_return"].to_numpy(dtype=float))
        worst5 = float(np.sort(led["trade_return"].to_numpy(dtype=float))[:5].sum() * 100)
        dd = led["trade_return"].to_numpy(dtype=float)
        report["maps"][m] = {"full": full, "by_period": per,
                             "worst5_trades_sum_pct": round(worst5, 2),
                             "downside_dev_pct": round(float(np.sqrt(np.mean(np.clip(dd, None, 0) ** 2)) * 100), 3),
                             "bootstrap_mdd_median": round(float(np.median(boot[m])), 2),
                             "p_mdd_better_than_identity": round(float(np.mean(boot[m] > base_boot)), 3)}

    ident = report["maps"]["identity"]
    results = {}
    for m in MAPS:
        if m == "identity":
            continue
        r = report["maps"][m]
        n_better = sum(1 for p in periods
                       if (r["by_period"][p]["mdd"] or 0) > (ident["by_period"][p]["mdd"] or 0))
        gates = {
            "gate_1_consistency": {"bar": GATE_PERIODS, "value": n_better, "pass": bool(n_better >= GATE_PERIODS)},
            "gate_2_bootstrap": {"bar": GATE_BOOT, "value": r["p_mdd_better_than_identity"],
                                 "pass": bool(r["p_mdd_better_than_identity"] >= GATE_BOOT)},
            "gate_3_pnl_guardrail": {"bar": round(GATE_PNL_RETAIN * ident["full"]["pnl"], 2),
                                     "value": r["full"]["pnl"],
                                     "pass": bool(r["full"]["pnl"] >= GATE_PNL_RETAIN * ident["full"]["pnl"])},
            "gate_4_not_by_starvation": {"bar": f"within {int(GATE_TRADE_TOL * 100)}% of {ident['full']['n']}",
                                         "value": r["full"]["n"],
                                         "pass": bool(abs(r["full"]["n"] - ident["full"]["n"])
                                                      <= GATE_TRADE_TOL * ident["full"]["n"])},
        }
        results[m] = {"gates": gates, "all_pass": all(g["pass"] for g in gates.values())}
    passing = [m for m, v in results.items() if v["all_pass"]]
    selected = min(passing, key=lambda m: (abs(report["maps"][m]["full"]["mdd"]),
                                           -(report["maps"][m]["full"]["calmar"] or 0))) if passing else None
    report["gate_results"] = results
    report["passing"] = passing
    report["selected"] = selected
    (OUT_DIR / "risk_evaluation.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))

    print("=== full-period (gated)", flush=True)
    for m in MAPS:
        r = report["maps"][m]["full"]
        b = report["maps"][m]
        print(f"  {m:15} pnl {r['pnl']:7.2f}  mdd {r['mdd']:7.2f}  calmar {str(r['calmar']):>6}  "
              f"n {r['n']:3d}  P(mdd better) {b['p_mdd_better_than_identity']:.3f}  "
              f"worst5 {b['worst5_trades_sum_pct']:7.2f}", flush=True)
    print("=== per-period MDD", flush=True)
    for m in MAPS:
        print(f"  {m:15} " + "  ".join(f"{p}:{report['maps'][m]['by_period'][p]['mdd']:7.2f}" for p in periods),
              flush=True)
    print(json.dumps({"passing": passing, "SELECTED": selected}, indent=2), flush=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    names = [m for m in MAPS]
    xs = np.arange(len(names))
    ax = axes[0]
    mdds = [report["maps"][m]["full"]["mdd"] for m in names]
    cols = [C_BASE if m == "identity" else (C_GOOD if mdds[i] > mdds[0] else C_BAD) for i, m in enumerate(names)]
    ax.bar(xs, mdds, color=cols)
    ax.axhline(mdds[0], color=INK, linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("전 구간 MDD %", fontsize=9)
    ax.set_title("① 드로다운 (0에 가까울수록 좋음)\n점선 = 베이스라인", loc="left", fontsize=11, color=INK)

    ax = axes[1]
    for i, m in enumerate(names):
        ax.scatter(report["maps"][m]["full"]["mdd"], report["maps"][m]["full"]["pnl"],
                   s=70, color=C_BASE if m == "identity" else C_GOOD, zorder=3)
        ax.annotate(m, (report["maps"][m]["full"]["mdd"], report["maps"][m]["full"]["pnl"]),
                    fontsize=7.5, xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("MDD %", fontsize=9)
    ax.set_ylabel("전 구간 PnL %", fontsize=9)
    ax.set_title("② 리스크-수익 평면\n왼쪽 위가 좋음", loc="left", fontsize=11, color=INK)

    ax = axes[2]
    ps = [report["maps"][m]["p_mdd_better_than_identity"] for m in names]
    ax.bar(xs, ps, color=[C_BASE if m == "identity" else (C_GOOD if ps[i] >= GATE_BOOT else C_BAD)
                          for i, m in enumerate(names)])
    ax.axhline(GATE_BOOT, color=INK, linewidth=1.0, linestyle="--")
    ax.set_xticks(xs)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("P(MDD가 베이스라인보다 좋음)", fontsize=9)
    ax.set_title(f"③ 블록 부트스트랩 B={B_BOOT}\n점선 = 게이트 {GATE_BOOT}", loc="left", fontsize=11, color=INK)

    for a in axes:
        a.grid(axis="y", color="#000000", alpha=0.08, linewidth=0.8)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig.suptitle("레짐 사이징 오버레이 — 리스크 우선 평가 (게이트 트레이드 기준)", fontsize=13, y=1.02)
    out = OUT_DIR / "risk_evaluation.png"
    fig.savefig(out, dpi=130, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
