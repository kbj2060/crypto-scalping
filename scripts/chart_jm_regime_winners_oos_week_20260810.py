"""OOS one-week regime charts for the scorecard-selected JM detectors, per asset.

Shows the selected redesign against the model it would replace, over the same OOS week, with the
retrospective zigzag oracle's wave turns drawn as vertical lines. Those turns are the point of the
chart: criterion 3 is whether the detector changes colour BEFORE the pivot line or after it, and
that is invisible on a plain regime-shaded chart. The oracle is retrospective by construction and
is drawn as a scoring reference only -- it is never an input to any detector here.

Selected by the pre-registered rule (consistency / persistence / timeliness, each gated against
the model being replaced):
  BTC  m=8  mRMR,    robust,   K=3, lambda_per_dim=0.1, T=0.25
  ETH  m=6  ANOVA-F, standard, K=3, lambda_per_dim=8,   T=0.5

The week is chosen as the densest OOS week in oracle pivots rather than the calendar-last one:
a week with no wave turn cannot show timeliness either way, so picking by pivot count is what
makes the comparison legible. Ad-hoc visualization, not a project artifact.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.jm_regime_redesign_lib_20260810 import (  # noqa: E402
    CLASSES3, FIT_YEAR, LABEL_CONFIGS, LABEL_MODE, PREFIX_STEM, SOURCES, _class_proba, _num,
    _read, _state_class_matrix, causal_decode_V, fit_jm, labels_for,
    quantile_matched_label_config, reference_label_quantiles, run_lengths, softmax_states,
)
from scripts.ranked_jm_feature_selection_20260810 import load_pool, rankings_for  # noqa: E402
from scripts.scorecard_jm_regime_decision_20260810 import (  # noqa: E402
    ORACLE_THETA, SUP, detection_lag, to_direction,
)
from scripts.test_statistical_jump_model_regimes_20260808 import zigzag_oracle  # noqa: E402

SELECTED = {
    "btc": {
        "spec": ("mrmr", 8, "robust", 3, 0.1, 0.25),
        "title": "SELECTED  m=8 mRMR, robust, K=3, lambda_per_dim=0.1, T=0.25",
        "label_basis": "qmatched",
        "incumbents": [
            ("INCUMBENT  live HMM wide24",
             SUP / "btc_regime3_current_hmm_sensitive_wide24_20260708/btc_features_{y}_regime3_current_sensitive_hmm_wide24.csv"),
            ("SHADOW CANDIDATE  JM lambda=2 wide24",
             SUP / "btc_regime3_current_hmm_jmlam2_20260810_{y}_maskedname.csv"),
        ],
    },
    "eth": {
        "spec": ("f_rank", 6, "standard", 3, 8.0, 0.5),
        "title": "SELECTED  m=6 ANOVA-F, standard, K=3, lambda_per_dim=8, T=0.5",
        "label_basis": "frozen",
        "incumbents": [
            ("INCUMBENT  JM lambda=4 wide24 (live ETH shadow)",
             SUP / "eth_regime3_current_hmm_jmlam4_20260809_{y}_maskedname.csv"),
        ],
    },
}
COLORS = {"bull": "#2ca02c", "bear": "#d62728", "chop": "#9AA0A6"}
SEED = 7529
OUT_DIR = Path("/mnt/c/Users/kbj20/AppData/Local/Temp/claude/"
               "--wsl-localhost-ubuntu-home-llewyn-crypto-scalping/"
               "543afe45-c3db-4c32-a239-e5ba56172716/scratchpad")


def selected_predictions(asset: str) -> tuple[np.ndarray, list[str]]:
    cfg = SELECTED[asset]
    ranking, m, scaler, k, lpd, temp_ratio = cfg["spec"]
    pool = load_pool(asset, scaler)
    idx = [int(i) for i in rankings_for(asset, scaler)[ranking][:m]]
    cols = [pool["cols"][i] for i in idx]
    lam = lpd * m
    mu, _ = fit_jm(pool[f"x_{FIT_YEAR}"][:, idx], k=k, lam=lam, seed=SEED, n_init=5, n_iter=15)
    V = {y: causal_decode_V(pool[f"x_{y}"][:, idx], mu, lam) for y in (FIT_YEAR, "2026")}
    spread = max(float(np.median(V[FIT_YEAR].max(axis=1) - V[FIT_YEAR].min(axis=1))), 1e-9)
    sp = {y: softmax_states(v, temp_ratio * spread) for y, v in V.items()}
    state_class = _state_class_matrix(sp[FIT_YEAR], pool[f"y_{cfg['label_basis']}_{FIT_YEAR}"])
    return np.argmax(_class_proba(sp["2026"], state_class), axis=1).astype(np.int64), cols


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


def densest_pivot_week(ts: pd.Series, pivots: list[int], lo: pd.Timestamp,
                       hi: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
    best, best_n = None, -1
    start = lo
    while start + pd.Timedelta(days=7) <= hi:
        end = start + pd.Timedelta(days=7)
        n = sum(1 for p in pivots if start <= ts.iloc[p] < end)
        if n > best_n:
            best, best_n = (start, end), n
        start += pd.Timedelta(days=1)
    return best


def shade(ax, ts: np.ndarray, pred: np.ndarray) -> int:
    names = np.array(CLASSES3)
    start = 0
    runs = 0
    for i in range(1, len(pred) + 1):
        if i == len(pred) or pred[i] != pred[start]:
            t1 = ts[i] if i < len(pred) else ts[-1] + np.timedelta64(5, "m")
            if pred[start] >= 0:
                ax.axvspan(ts[start], t1, color=COLORS[names[pred[start]]], alpha=0.25, linewidth=0)
            runs += 1
            start = i
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["btc", "eth"])
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for asset in args.assets:
        cfg = SELECTED[asset]
        frame = _read(SOURCES[asset]["2026"])
        close = _num(frame, "close").ffill().bfill().to_numpy()
        oracle, pivots = zigzag_oracle(close, ORACLE_THETA)
        ts = frame["timestamp"]

        oos_lo, oos_hi = pd.Timestamp("2026-01-01"), pd.Timestamp("2026-04-01")
        wk_lo, wk_hi = densest_pivot_week(ts, pivots, oos_lo, oos_hi)
        wmask = ((ts >= wk_lo) & (ts < wk_hi)).to_numpy()
        widx = np.flatnonzero(wmask)
        wpivots = [p for p in pivots if wk_lo <= ts.iloc[p] < wk_hi]
        print(f"{asset}: week {wk_lo:%Y-%m-%d} .. {wk_hi:%Y-%m-%d}, {len(wpivots)} oracle pivots")

        sel_pred, cols = selected_predictions(asset)
        panels = [(cfg["title"], sel_pred)]
        for name, tmpl in cfg["incumbents"]:
            p = csv_predictions(tmpl, frame)
            if p is not None:
                panels.append((name, p))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(15, 4.0 * n), sharex=True)
        axes = np.atleast_1d(axes)
        wts = ts.to_numpy()[wmask]
        wclose = close[wmask]
        for ax, (title, pred) in zip(axes, panels):
            runs = shade(ax, wts, pred[wmask])
            ax.plot(wts, wclose, color="black", linewidth=1.1)
            for p in wpivots:
                ax.axvline(ts.iloc[p], color="#111", linestyle="--", linewidth=1.2, alpha=0.75)
            d = to_direction(pred)
            d[~wmask] = 0
            dl = detection_lag(d, oracle, pivots, int(widx[0]), int(widx[-1]), horizon=576)
            lag_txt = ("no pivot with enough lead-in inside this week"
                       if dl["median_bars"] is None
                       else f"detection lag in-week: median {dl['median_bars']:.0f} bars "
                            f"({dl['median_bars'] * 5:.0f} min) over {dl['n_pivots']} pivots")
            ax.set_title(title, fontsize=11, loc="left")
            ax.set_ylabel(f"{asset.upper()} close")
            ax.grid(alpha=0.15)
            ax.text(0.995, 0.05, f"{runs} regime runs   |   {lag_txt}",
                    transform=ax.transAxes, ha="right", fontsize=9, color="#333")

        axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %Hh"))
        handles = [Patch(facecolor=COLORS[c], alpha=0.45, label=c) for c in CLASSES3]
        handles.append(Line2D([0], [0], color="#111", linestyle="--",
                              label=f"zigzag oracle wave turn ({ORACLE_THETA:.0%})"))
        axes[0].legend(handles=handles, loc="upper left", ncol=4, fontsize=9, framealpha=0.9)
        fig.suptitle(f"{asset.upper()} 5m regime, OOS week {wk_lo:%Y-%m-%d} to {wk_hi:%Y-%m-%d}  "
                     f"(fit on 2024 only, causal forward decode)", fontsize=13)
        fig.text(0.01, 0.005, f"selected features (m={len(cols)}): " + ", ".join(cols),
                 fontsize=8, color="#555")
        fig.tight_layout(rect=(0, 0.02, 1, 0.97))
        out = OUT_DIR / f"{asset}_jm_selected_vs_incumbent_oos_week.png"
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  -> {out}")


if __name__ == "__main__":
    main()
