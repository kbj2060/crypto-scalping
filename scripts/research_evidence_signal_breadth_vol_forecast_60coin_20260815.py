#!/usr/bin/env python3
"""RESEARCH -- Does breadth add anything to a FREE volatility forecast? (2026-08-15)

Follow-up to research_evidence_signal_breadth_risk_gate_60coin_20260815.py, which found:
  - RETURN hypothesis FAILED (bottom-extreme forward return direction was right at all 4 horizons
    but p = 0.18-0.69, period consistency only 2-3/4; top-extreme direction was wrong at 3 of 4).
  - VOLATILITY hypothesis PASSED decisively (bottom-extreme forward vol +38% vs the random-episode
    null, p=0.000, 4/4 periods, at h=48/144/288; left tail fatter at h=48, p=0.000).

Two problems with that script's economic test, both consequences of it having been designed around
the hypothesis that FAILED, and both fixed here:

  (1) WRONG BENEFIT METRIC. Its breakeven was computed from the ARITHMETIC mean return difference,
      but the benefit of cutting exposure in a high-volatility regime is mostly reduced volatility
      drag, which only shows up in the COMPOUNDED path. That is not a rounding difference: at
      h=288 the gate turned -41.3% into +18.5% compounded while its arithmetic-mean difference was
      negative, so the old metric reported "loses money before costs" about a series that had in
      fact improved. Breakeven is recomputed here by solving for the round-trip cost at which the
      gated COMPOUNDED total equals the ungated one.
  (2) WRONG CONSTRUCTION. A binary on/off gate keyed to direction is the natural use of a direction
      signal. What survived is a VOLATILITY signal, whose textbook use is volatility-targeted
      sizing (exposure ~ target / forecast vol), tested here instead.

=== The control that decides this, pre-registered ===
Forward volatility is famously predictable from TRAILING realized volatility, which is free. So the
question is NOT "does breadth predict forward vol" (already answered yes) but "does breadth predict
forward vol BEYOND trailing realized vol". This is the exact analogue of the short-term-reversal
benchmark that dissolved the cross-sectional IC result, and it is the primary test here:

  y  = log forward realized vol over the next h bars
  x1 = log trailing realized vol over the past h bars      (the free benchmark)
  x2 = causal rolling percentile of breadth_bottom_K2      (the candidate increment)
  x3 = causal rolling percentile of breadth_top_K2

  FIT ON 2024 ONLY, evaluate out-of-sample on 2025H1 / 2025H2 / 2026 separately. No full-sample
  fit anywhere. Samples taken every h bars so forward windows never overlap.

  PRIMARY METRIC: out-of-sample Delta R-squared of (x1,x2,x3) over (x1) alone.
  KILL CRITERION (fixed before running): breadth is closed as a vol-forecast input unless Delta R2
  is positive in >= 3 of the 3 out-of-sample periods AND the pooled OOS Delta R2 exceeds +0.005.
  A positive-but-negligible increment counts as a failure, not a partial success.

Economic test: volatility-targeted exposure on the passive EW panel, exposure_t =
clip(target_vol / forecast_vol_t, 0, MAX_EXPOSURE), forecast from (a) trailing vol alone and
(b) trailing vol + breadth, compared against constant exposure. Costs charged on |exposure change|
per bar; breakeven round-trip cost solved on the compounded path.

Same universe caveats as the companion studies (liquidity lookahead, survivorship) -- only the
DIFFERENCE between the three exposure rules should be read, never the absolute level. Causal
throughout: every regressor is trailing/rolling; forward windows are outcomes only.
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training beyond a 3-variable OLS, no GPU, no live files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import research_evidence_signal_breadth_risk_gate_60coin_20260815 as bg  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/evidence_signal_breadth_vol_forecast_60coin_20260815"
HORIZONS = (48, 144, 288)
PCT_WINDOW = bg.PCT_WINDOW
TRAIN_PERIOD = ("2024-01-01", "2024-12-31 23:59:59")
OOS_PERIODS = {k: v for k, v in bg.PERIODS.items() if k != "2024"}
DELTA_R2_MIN = 0.005
TARGET_VOL_QUANTILE = 0.50       # target = median forecast vol in the TRAIN period only
MAX_EXPOSURE = 2.0
COST_GRID_BPS = (0.0, 1.0, 2.0, 4.0, 10.0, 20.0, 50.0)


def log(msg: str) -> None:
    print(f"[breadth_vol] {msg}", flush=True)


def _ols(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    X1 = np.column_stack([np.ones(len(X)), X])
    return np.linalg.lstsq(X1, y, rcond=None)[0]


def _pred(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(X)), X]) @ beta


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(((y - yhat) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def _compounded(x: np.ndarray) -> dict[str, Any]:
    cum = np.cumprod(1.0 + x)
    peak = np.maximum.accumulate(cum)
    return {"total_return_pct": float((cum[-1] - 1) * 100),
            "ann_sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(288 * 365)) if x.std(ddof=1) > 0 else float("nan"),
            "mdd_pct": float((cum / peak - 1).min() * 100),
            "realized_vol_ann": float(x.std(ddof=1) * np.sqrt(288 * 365))}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=build_breadth (reusing companion module) ===")
    breadth, ew_ret, _eth = bg.build_breadth()
    ts = breadth.index
    r = np.nan_to_num(ew_ret.to_numpy(dtype=float), nan=0.0)
    logret = np.log1p(r)
    n = len(r)

    pct_bottom = breadth["bottom_K2"].rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True).to_numpy()
    pct_top = breadth["top_K2"].rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True).to_numpy()

    report: dict[str, Any] = {
        "design": "Does breadth add to a FREE trailing-realized-vol forecast? Train on 2024 only, evaluate OOS on "
                  "2025H1/2025H2/2026. Primary metric: OOS Delta R2 of log forward vol. Plus vol-targeted sizing "
                  "economics with breakeven solved on the compounded path.",
        "pre_registered": {"train_period": list(TRAIN_PERIOD), "oos_periods": {k: list(v) for k, v in OOS_PERIODS.items()},
                           "horizons": list(HORIZONS), "delta_r2_min": DELTA_R2_MIN,
                           "kill_criterion": "Delta R2 > 0 in >=3/3 OOS periods AND pooled OOS Delta R2 > 0.005",
                           "max_exposure": MAX_EXPOSURE, "cost_grid_bps": list(COST_GRID_BPS)},
        "look_ahead_fix": "Vol-targeted exposure is lagged one bar (exposure at t is computed at t-1). Breadth fires ON "
                          "the extreme bar, so the unlagged version sizes the crash bar with knowledge of it and reports "
                          "a spurious Sharpe ~6.2; the lagged figures below are the only valid ones.",
        "fixes_vs_companion_script": [
            "breakeven recomputed on the COMPOUNDED path (the arithmetic-mean version mis-signed the h=288 case)",
            "vol-targeted sizing replaces the binary direction-keyed gate, matching the hypothesis that actually survived",
        ],
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "by_horizon": {},
    }

    for h in HORIZONS:
        # trailing and forward realized vol of the EW panel return, both from the same estimator
        s = pd.Series(logret)
        trail_vol = s.rolling(h, min_periods=h).std().to_numpy()
        fwd_vol = np.full(n, np.nan)
        fwd_vol[:n - h] = trail_vol[h:]
        ok = np.isfinite(trail_vol) & np.isfinite(fwd_vol) & np.isfinite(pct_bottom) & np.isfinite(pct_top)
        ok &= (trail_vol > 0) & (fwd_vol > 0)
        idx = np.flatnonzero(ok)
        idx = idx[idx % h == 0]                      # non-overlapping forward windows
        y = np.log(fwd_vol[idx])
        X_base = np.log(trail_vol[idx]).reshape(-1, 1)
        X_full = np.column_stack([np.log(trail_vol[idx]), pct_bottom[idx], pct_top[idx]])
        t_idx = ts[idx]

        tr = (t_idx >= pd.Timestamp(TRAIN_PERIOD[0])) & (t_idx <= pd.Timestamp(TRAIN_PERIOD[1]))
        b_base = _ols(X_base[tr], y[tr])
        b_full = _ols(X_full[tr], y[tr])

        per_period, deltas = {}, []
        for pname, (a, bnd) in OOS_PERIODS.items():
            sel = (t_idx >= pd.Timestamp(a)) & (t_idx <= pd.Timestamp(bnd))
            if sel.sum() < 20:
                per_period[pname] = {"n": int(sel.sum()), "underpowered": True}
                continue
            r2b, r2f = _r2(y[sel], _pred(b_base, X_base[sel])), _r2(y[sel], _pred(b_full, X_full[sel]))
            per_period[pname] = {"n": int(sel.sum()), "r2_trailing_only": r2b, "r2_with_breadth": r2f,
                                 "delta_r2": r2f - r2b}
            deltas.append(r2f - r2b)
        pooled = (t_idx >= pd.Timestamp(OOS_PERIODS["2025H1"][0]))
        r2b_p, r2f_p = _r2(y[pooled], _pred(b_base, X_base[pooled])), _r2(y[pooled], _pred(b_full, X_full[pooled]))
        pooled_delta = r2f_p - r2b_p
        n_pos = int(sum(1 for d in deltas if d > 0))
        passes = bool(n_pos >= 3 and pooled_delta > DELTA_R2_MIN)

        log(f"  h={h:3d}: train_n={int(tr.sum())} | OOS R2 trailing={r2b_p:.4f} with_breadth={r2f_p:.4f} "
            f"deltaR2={pooled_delta:+.5f} | per-period delta={[round(d, 5) for d in deltas]} "
            f"positive={n_pos}/3 -> pass={passes}")
        log(f"        breadth coefficients (train): bottom={b_full[2]:+.4f} top={b_full[3]:+.4f} "
            f"(trailing log-vol coef {b_full[1]:+.4f})")

        # ---- economics: volatility-targeted exposure ----
        fc_base = np.exp(_pred(b_base, np.log(trail_vol[ok]).reshape(-1, 1)))
        fc_full = np.exp(_pred(b_full, np.column_stack([np.log(trail_vol[ok]), pct_bottom[ok], pct_top[ok]])))
        rr = r[ok]
        t_ok = ts[ok]
        train_mask = (t_ok >= pd.Timestamp(TRAIN_PERIOD[0])) & (t_ok <= pd.Timestamp(TRAIN_PERIOD[1]))
        oos_mask = t_ok >= pd.Timestamp(OOS_PERIODS["2025H1"][0])
        target = float(np.quantile(fc_base[train_mask], TARGET_VOL_QUANTILE))  # target set on TRAIN only

        econ: dict[str, Any] = {"target_vol": target}
        for label, fc in (("trailing_only", fc_base), ("with_breadth", fc_full)):
            expo = np.clip(target / np.maximum(fc, 1e-12), 0.0, MAX_EXPOSURE)
            # MANDATORY one-bar shift: the forecast at bar t uses bar t's own OHLC (breadth fires ON
            # the extreme bar), so sizing bar t's return with it is look-ahead. Exposure applied to
            # bar t is therefore the value computed at t-1. Without this the breadth arm reports an
            # absurd Sharpe ~6 purely from being small on the very bar of each crash.
            expo = np.concatenate([[1.0], expo[:-1]])
            gross = expo[oos_mask] * rr[oos_mask]
            dexp = np.abs(np.diff(np.concatenate([[1.0], expo[oos_mask]])))
            stats_by_cost = {}
            for c_bps in COST_GRID_BPS:
                net = gross - dexp * (c_bps / 10000.0) / 2.0
                stats_by_cost[f"{c_bps:g}bps"] = _compounded(net)
            econ[label] = {"mean_exposure": float(expo[oos_mask].mean()),
                           "turnover_sum": float(dexp.sum()), "by_cost": stats_by_cost}
        econ["constant_exposure"] = {"mean_exposure": 1.0, "turnover_sum": 0.0,
                                     "by_cost": {"0bps": _compounded(rr[oos_mask])}}
        for label in ("trailing_only", "with_breadth"):
            log(f"        vol-target [{label:13s}] OOS: exposure={econ[label]['mean_exposure']:.3f} "
                f"@0bps ret={econ[label]['by_cost']['0bps']['total_return_pct']:+.1f}% "
                f"sharpe={econ[label]['by_cost']['0bps']['ann_sharpe']:+.2f} mdd={econ[label]['by_cost']['0bps']['mdd_pct']:.1f}% "
                f"| @4bps ret={econ[label]['by_cost']['4bps']['total_return_pct']:+.1f}% "
                f"| @10bps ret={econ[label]['by_cost']['10bps']['total_return_pct']:+.1f}%")
        c = econ["constant_exposure"]["by_cost"]["0bps"]
        log(f"        vol-target [{'constant':13s}] OOS: ret={c['total_return_pct']:+.1f}% sharpe={c['ann_sharpe']:+.2f} mdd={c['mdd_pct']:.1f}%")

        report["by_horizon"][str(h)] = {
            "oos_r2_trailing_only": r2b_p, "oos_r2_with_breadth": r2f_p, "oos_delta_r2": pooled_delta,
            "per_period": per_period, "periods_positive": n_pos, "passes_kill_criterion": passes,
            "train_coefficients": {"intercept": float(b_full[0]), "log_trailing_vol": float(b_full[1]),
                                   "pct_breadth_bottom": float(b_full[2]), "pct_breadth_top": float(b_full[3])},
            "economics": econ,
        }

    any_pass = any(v["passes_kill_criterion"] for v in report["by_horizon"].values())
    report["verdict"] = "BREADTH_ADDS_TO_VOL_FORECAST" if any_pass else "CLOSED_NO_INCREMENT_OVER_TRAILING_VOL"
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done VERDICT={report['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
