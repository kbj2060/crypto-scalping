"""Final OOS one-week regime charts for the rule-selected JM detectors, per asset.

Draws every candidate against the model it would replace over the same OOS week, with the
retrospective zigzag oracle's wave turns as vertical lines. Those lines are the point: criterion 3
is whether a detector changes colour BEFORE the turn or after it, which no amount of regime
shading shows on its own. The oracle uses future information by construction and is a scoring
reference only, never an input.

BTC carries two candidates because the pre-registered rule's tie-break (OOS balanced accuracy) and
the three stated criteria disagree. 41 cells passed all three criteria; among them greedy_fwd8 wins
the tie-break on accuracy alone, while mrmr_top8 matches it exactly on timeliness (detection lag
10.0, wave-Q1 ~0.66) and is clearly steadier (whipsaw 0.20 vs 0.29). Accuracy was never one of the
three criteria, so both are drawn rather than silently resolving it.

Feature sets are passed as explicit column lists, because the greedy forward-selected sets are not
prefixes of any ranking and cannot be reconstructed from rankings_for().

Ad-hoc visualization, not a project artifact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM, SOURCES, _class_proba, _num,
    _read, _state_class_matrix, causal_decode_V, fit_jm, labels_for,
    quantile_matched_label_config, reference_label_quantiles, softmax_states,
)
from scripts.prep_jm_regime_redesign_inputs_20260810 import OUT_DIR  # noqa: E402
from scripts.ranked_jm_feature_selection_20260810 import load_pool  # noqa: E402
from scripts.scorecard_jm_regime_decision_20260810 import (  # noqa: E402
    ORACLE_THETA, SUP, detection_lag, to_direction,
)
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

COLORS = {"bull": "#2ca02c", "bear": "#d62728", "chop": "#9AA0A6"}
SEED = 7529
OUT_PNG = Path("/mnt/c/Users/kbj20/AppData/Local/Temp/claude/"
               "--wsl-localhost-ubuntu-home-llewyn-crypto-scalping/"
               "543afe45-c3db-4c32-a239-e5ba56172716/scratchpad")

INCUMBENTS = {
    "btc": [("ANCHOR  live HMM wide24 (the model being replaced)",
             SUP / "btc_regime3_current_hmm_sensitive_wide24_20260708/btc_features_{y}_regime3_current_sensitive_hmm_wide24.csv")],
    "eth": [("ANCHOR  JM lambda=4 wide24 (live ETH shadow)",
             SUP / "eth_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv")],
}


def predictions(asset: str, cols: list[str], scaler: str, k: int, lpd: float, temp: float,
                basis: str) -> np.ndarray:
    pool = load_pool(asset, scaler)
    idx = [pool["cols"].index(c) for c in cols]
    lam = lpd * len(idx)
    mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=k, lam=lam, seed=SEED, n_init=5, n_iter=15)
    V = {y: causal_decode_V(pool[f"x_{y}"][:, idx], mu, lam) for y in (FIT_YEAR, "2026")}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    sp = {y: softmax_states(v, temp * spread) for y, v in V.items()}
    sc = _state_class_matrix(sp[FIT_YEAR], pool[f"y_{basis}_{FIT_YEAR}"])
    return np.argmax(_class_proba(sp["2026"], sc), axis=1).astype(np.int64)


def csv_predictions(tmpl: Path, frame: pd.DataFrame) -> np.ndarray | None:
    p = Path(str(tmpl).replace("{y}", "2026"))
    if not p.exists():
        return None
    pf = pd.read_csv(p)
    cols = None
    for prefix in (f"{PREFIX_STEM}_wide24_", f"{PREFIX_STEM}_hmm_wide24_"):
        c = [f"{prefix}{n}_prob" for n in CLASSES3]
        if all(x in pf.columns for x in c):
            cols = c
            break
    if cols is None:
        return None
    pf["timestamp"] = pd.to_datetime(pf["timestamp"])
    merged = frame[["timestamp"]].merge(pf[["timestamp"] + cols], on="timestamp", how="left")
    proba = merged[cols].to_numpy()
    pred = np.full(len(frame), -1, dtype=np.int64)
    ok = np.isfinite(proba).all(axis=1)
    pred[ok] = np.argmax(proba[ok], axis=1)
    return pred


def densest_pivot_week(ts: pd.Series, pivots, lo, hi):
    best, best_n = None, -1
    start = lo
    while start + pd.Timedelta(days=7) <= hi:
        end = start + pd.Timedelta(days=7)
        n = sum(1 for p in pivots if start <= ts.iloc[p] < end)
        if n > best_n:
            best, best_n = (start, end), n
        start += pd.Timedelta(days=1)
    return best


def shade(ax, ts, pred) -> int:
    names = np.array(CLASSES3)
    start, runs = 0, 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[start]:
            t1 = ts[i] if i < len(pred) else ts[-1] + np.timedelta64(5, "m")
            if pred[start] >= 0:
                ax.axvspan(ts[start], t1, color=COLORS[names[pred[start]]], alpha=0.25, linewidth=0)
            runs += 1
            start = i
    return runs


def candidates_for(asset: str) -> list[dict]:
    """Rule winner plus, on BTC, the criteria-first runner-up the tie-break passed over."""
    dec = json.loads((OUT_DIR / "final_decision_v2.json").read_text())[asset]
    fsets, cells = dec["feature_sets"], dec["cells"]
    passing = [c for c in cells if c["passes_rule"]]
    out = []
    w = max(passing, key=lambda c: c["oos_balanced_accuracy"])
    out.append({"tag": "RULE WINNER (tie-break: OOS balanced accuracy)", **w})
    steady = min(passing, key=lambda c: (c["oos_whipsaw_share"], -c["oos_median_run_bars"]))
    if (steady["feature_set"], steady["scaler"], steady["k"], steady["lambda_per_dim"],
            steady["temperature_ratio"], steady["label_basis"]) != (
            w["feature_set"], w["scaler"], w["k"], w["lambda_per_dim"],
            w["temperature_ratio"], w["label_basis"]):
        out.append({"tag": "CRITERIA-FIRST (steadiest among rule-passing cells)", **steady})
    for c in out:
        c["cols"] = fsets[c["feature_set"]]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    args = ap.parse_args()
    OUT_PNG.mkdir(parents=True, exist_ok=True)

    for asset in args.assets:
        cands = candidates_for(asset)
        frame = _read(SOURCES[asset]["2026"])
        close = _num(frame, "close").ffill().bfill().to_numpy()
        oracle, pivots = zigzag_oracle(close, ORACLE_THETA)
        ts = frame["timestamp"]
        wk_lo, wk_hi = densest_pivot_week(ts, pivots, pd.Timestamp("2026-01-01"),
                                          pd.Timestamp("2026-04-01"))
        wmask = ((ts >= wk_lo) & (ts < wk_hi)).to_numpy()
        widx = np.flatnonzero(wmask)
        wpiv = [p for p in pivots if wk_lo <= ts.iloc[p] < wk_hi]
        print(f"{asset}: week {wk_lo:%Y-%m-%d}..{wk_hi:%Y-%m-%d}, {len(wpiv)} oracle pivots")

        panels = []
        for c in cands:
            pred = predictions(asset, c["cols"], c["scaler"], c["k"], c["lambda_per_dim"],
                               c["temperature_ratio"], c["label_basis"])
            title = (f"{c['tag']}\n{c['feature_set']} m={len(c['cols'])} {c['scaler']} "
                     f"K={c['k']} lpd={c['lambda_per_dim']:g} T={c['temperature_ratio']:g} "
                     f"[{c['label_basis']}]   OOS bal={c['oos_balanced_accuracy']:.4f}  "
                     f"detlag={c['oos_detection_lag_median']:.0f}  Q1={c['oos_wave_Q1']:.3f}  "
                     f"whip={c['oos_whipsaw_share']:.2f}  sep_t={c['oos_economic_separation_tstat']:+.2f}")
            panels.append((title, pred, c["cols"]))
        for name, tmpl in INCUMBENTS[asset]:
            p = csv_predictions(tmpl, frame)
            if p is not None:
                panels.append((name, p, None))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(15, 4.2 * n), sharex=True)
        axes = np.atleast_1d(axes)
        wts, wclose = ts.to_numpy()[wmask], close[wmask]
        for ax, (title, pred, _c) in zip(axes, panels):
            runs = shade(ax, wts, pred[wmask])
            ax.plot(wts, wclose, color="black", linewidth=1.1)
            for p in wpiv:
                ax.axvline(ts.iloc[p], color="#111", linestyle="--", linewidth=1.2, alpha=0.75)
            d = to_direction(pred)
            d[~wmask] = 0
            dl = detection_lag(d, oracle, pivots, int(widx[0]), int(widx[-1]))
            txt = ("no pivot with enough lead-in" if dl["median_bars"] is None else
                   f"in-week detection lag: median {dl['median_bars']:.0f} bars "
                   f"({dl['median_bars'] * 5:.0f} min) over {dl['n_pivots']} pivots")
            ax.set_title(title, fontsize=9.5, loc="left")
            ax.set_ylabel(f"{asset.upper()} close")
            ax.grid(alpha=0.15)
            ax.text(0.995, 0.05, f"{runs} regime runs   |   {txt}", transform=ax.transAxes,
                    ha="right", fontsize=9, color="#333")
        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        handles = [Patch(facecolor=COLORS[c], alpha=0.45, label=c) for c in CLASSES3]
        handles.append(Line2D([0], [0], color="#111", linestyle="--",
                              label=f"zigzag oracle wave turn ({ORACLE_THETA:.0%})"))
        axes[0].legend(handles=handles, loc="upper left", ncol=4, fontsize=9, framealpha=0.9)
        fig.suptitle(f"{asset.upper()} 5m regime, OOS week {wk_lo:%Y-%m-%d} to {wk_hi:%Y-%m-%d} "
                     f"(fit on 2024 only, causal forward decode)", fontsize=13)
        fig.text(0.01, 0.004, "features: " + ", ".join(panels[0][2]), fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.02, 1, 0.975))
        out = OUT_PNG / f"{asset}_jm_final_oos_week.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
