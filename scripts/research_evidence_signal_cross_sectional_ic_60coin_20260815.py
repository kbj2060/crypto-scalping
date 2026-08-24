#!/usr/bin/env python3
"""RESEARCH -- Evidence signals as a CROSS-SECTIONAL factor on the 60-perp panel (2026-08-15).

=== Why this question, and why it is not a repeat ===
Every prior test of the ETH reversal evidence signals was single-asset time-series: as an entry
trigger (docs/experiments/eth_evidence_signal_top6_confluence_standalone_backtest_20260814.md --
0/36 window x K cells vs always_long/always_short) or injected into the live Omega4.6.1 model
(contract Odyssey2 #18-#22 -- all rejected; #22 showed fire bars are indistinguishable from matched
random bars for the exit decision). In BOTH settings the benchmark was directional drift: a
single-asset strategy must out-earn buy-and-hold, and the injection host earns most of its PnL from
a short-drift regime. A dollar-neutral CROSS-SECTIONAL portfolio removes that benchmark entirely --
it asks only "among 60 coins at the same instant, does a higher evidence score predict a higher
forward return?" That question has never been asked in this repo for these signals, and it is the
setting where reversal/exhaustion factors are conventionally strongest.

=== What is measured (all pre-registered before looking at any result) ===
  Stage A -- PROXY VALIDATION (gate). The panel feature store carries no raw `volume`/
    `taker_buy_base`, so the order-flow term delta_z = z288(2*taker_buy_base - volume) cannot be
    reproduced exactly; the panel-computable proxy is z288(rvol_48 * (2*taker_buy_ratio - 1)),
    i.e. the same net-aggressive-flow quantity divided by a rolling 48-bar mean volume (and with
    quote- rather than base-denominated taker share, plus rvol_48's clip(0,20)). ETH and BTC have
    BOTH the raw klines and a panel parquet, so the proxy is checked against ground truth on those
    two symbols BEFORE any panel-wide result is trusted. Pre-registered gate: Spearman(proxy, true)
    >= 0.80 AND event-level Jaccard(proxy<=-2, true<=-2) >= 0.30 on both symbols. Failing the gate
    does not silently degrade the study -- it aborts it and reports that raw panel klines must be
    re-downloaded.

  Stage B -- CROSS-SECTIONAL RANK IC of the evidence score against forward returns at horizons
    h in {12, 48, 144, 288} bars (1h/4h/12h/24h), sampled every h bars so IC observations do not
    overlap. Two score variants, both fixed in advance:
      `votes`      -- bottom_votes - top_votes over the same 6 signals as the standalone backtest
                      (orthogonal_combo, liquidity_sweep, volume_wick_climax, short_term_return_z,
                      taker_delta_z_climax, bollinger_pctb_extreme), definitions reused unmodified.
                      Faithful to the validated construction but coarse (mostly 0 -> heavy ties).
      `continuous` -- sum of the same components in standardized continuous form, for ranking
                      resolution. Reported side by side; neither is selected after the fact.

  Three mandatory controls, because a nonzero IC alone proves nothing here:
      (1) PERMUTATION -- scores shuffled across symbols within each bar (destroys cross-sectional
          information, preserves everything else). 20 replicates, fixed seed.
      (2) SHORT-TERM REVERSAL BENCHMARK -- IC of the plain factor -ret_{past h}. Cross-sectional
          short-horizon reversal is a well-known crypto factor; if the evidence score merely
          reproduces it, it adds nothing. Also reported: IC of the evidence score after
          cross-sectionally neutralizing (regressing out) that reversal factor.
      (3) PERIOD SPLIT -- 2024 / 2025H1 / 2025H2 / 2026, so a single-period result cannot be
          reported as "confirmed" (repo rule: >=4 independent sign-consistent windows).

  Stage C -- dollar-neutral quintile long-short portfolio at each horizon, gross and net of a
    0.1% round-trip cost charged on turnover, as the economic (not statistical) reading of the IC.

=== Known limitations, stated up front ===
The panel universe is the top-60 USDT perps by 24h volume AT SELECTION TIME (2026-08-04) among
symbols still trading and onboarded before 2024-01-01 -- it carries the liquidity-lookahead and
survivorship caveats its own manifest documents (data/splits/panel_universe_symbols_20260804.json).
Those biases inflate a long-short factor study of this kind and are NOT corrected here; any
positive result must be read as an upper bound pending a point-in-time universe. Costs are modelled
as a flat 0.1% round trip (the same constant this lineage already uses for ETH) applied to
portfolio turnover; no borrow/funding/impact model, and small-cap perps are exactly where that
assumption is weakest.

fresh_forward_bar_by_bar: not applicable (this is a factor IC study, not a trade replay) -- but
every input is causal (rolling/shift only, no negative shift) and forward returns are used ONLY as
the prediction target, never as an input. trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. No training, no GPU.
Does NOT touch live files. Reads data/panel/features/*.parquet (read-only) and, for Stage A only,
the raw ETH/BTC kline CSVs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

PANEL_DIR = ROOT / "data/panel/features"
UNIVERSE_JSON = ROOT / "data/splits/panel_universe_symbols_20260804.json"
ETH_RAW = ROOT / "data/splits/year_oos/training_features_2025.csv"
BTC_RAW = ROOT / "data/btc_5m_1year.csv"
OUT_DIR = ROOT / "tmp/causal_regen_20260516/evidence_signal_cross_sectional_ic_60coin_20260815"

# --- signal constants: reused unmodified from the ETH evidence lineage, nothing re-tuned here ---
OSC_WINDOW, OSC_DECILE = 864, 0.10
DZ_WINDOW, DZ_ABS = 288, 2.0
RET_Z_BARS, RET_Z_WINDOW, RET_Z_ABS = 3, 288, 2.5
SWEEP_LOOKBACK = 48
VOL_Z_MIN, WICK_MIN = 2.0, 0.50
BB_WINDOW, BB_LOW, BB_HIGH = 20, 0.05, 0.95

HORIZONS = (12, 48, 144, 288)
PERM_REPS, PERM_SEED = 20, 20260815
MIN_SYMBOLS_PER_BAR = 20
COST_ROUNDTRIP = 0.001
PROXY_GATE_SPEARMAN, PROXY_GATE_JACCARD = 0.80, 0.30
PERIODS = {"2024": ("2024-01-01", "2024-12-31 23:59:59"), "2025H1": ("2025-01-01", "2025-06-30 23:59:59"),
           "2025H2": ("2025-07-01", "2025-12-31 23:59:59"), "2026": ("2026-01-01", "2026-12-31 23:59:59")}


def log(msg: str) -> None:
    print(f"[xsect_ic] {msg}", flush=True)


# =====================================================================================================
# Signal construction. Everything below is rolling/shift only.
# =====================================================================================================
def _oscillator_percentiles(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[pd.Series, pd.Series]:
    hh, ll = high.rolling(14, min_periods=14).max(), low.rolling(14, min_periods=14).min()
    rng = (hh - ll).replace(0.0, np.nan)
    fast_k = 100.0 + (-100.0 * (hh - close) / rng)
    slow_k = fast_k.rolling(3, min_periods=3).mean()
    return (fast_k.rolling(OSC_WINDOW, min_periods=OSC_WINDOW).rank(pct=True),
            slow_k.rolling(OSC_WINDOW, min_periods=OSC_WINDOW).rank(pct=True))


def _z(s: pd.Series, w: int) -> pd.Series:
    return (s - s.rolling(w, min_periods=w).mean()) / s.rolling(w, min_periods=w).std().replace(0.0, np.nan)


def build_signals(df: pd.DataFrame, *, delta_series: pd.Series, volume_series: pd.Series) -> pd.DataFrame:
    """`delta_series` is net aggressive buy flow (raw or proxy), `volume_series` is volume (raw or
    the rvol_48 proxy). Both enter only through 288-bar z-scores, which are scale-invariant."""
    high, low, close, open_ = df["high"], df["low"], df["close"], df["open"]
    p_fast, p_slow = _oscillator_percentiles(high, low, close)
    delta_z = _z(delta_series, DZ_WINDOW)
    ret_n = close / close.shift(RET_Z_BARS) - 1.0
    ret_z = _z(ret_n, RET_Z_WINDOW)
    vol_z = _z(volume_series, DZ_WINDOW)
    rng = (high - low).replace(0.0, np.nan)
    lower_wick = (np.minimum(open_, close) - low) / rng
    upper_wick = (high - np.maximum(open_, close)) / rng
    sma, sd = close.rolling(BB_WINDOW, min_periods=BB_WINDOW).mean(), close.rolling(BB_WINDOW, min_periods=BB_WINDOW).std()
    pctb = (close - (sma - 2 * sd)) / (4 * sd).replace(0.0, np.nan)
    swing_low = low.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).min().shift(1)
    swing_high = high.rolling(SWEEP_LOOKBACK, min_periods=SWEEP_LOOKBACK).max().shift(1)

    bottom = pd.DataFrame({
        "orthogonal_combo": (p_fast <= OSC_DECILE) & (p_slow <= OSC_DECILE) & (delta_z <= -DZ_ABS),
        "liquidity_sweep": (low < swing_low) & (close > swing_low),
        "volume_wick_climax": (vol_z >= VOL_Z_MIN) & (lower_wick >= WICK_MIN),
        "short_term_return_z": ret_z <= -RET_Z_ABS,
        "taker_delta_z_climax": delta_z <= -DZ_ABS,
        "bollinger_pctb_extreme": pctb <= BB_LOW,
    }).fillna(False)
    top = pd.DataFrame({
        "orthogonal_combo": (p_fast >= 1 - OSC_DECILE) & (p_slow >= 1 - OSC_DECILE) & (delta_z >= DZ_ABS),
        "liquidity_sweep": (high > swing_high) & (close < swing_high),
        "volume_wick_climax": (vol_z >= VOL_Z_MIN) & (upper_wick >= WICK_MIN),
        "short_term_return_z": ret_z >= RET_Z_ABS,
        "taker_delta_z_climax": delta_z >= DZ_ABS,
        "bollinger_pctb_extreme": pctb >= BB_HIGH,
    }).fillna(False)

    out = pd.DataFrame({"timestamp": df["timestamp"].to_numpy()})
    out["votes"] = bottom.sum(axis=1).to_numpy() - top.sum(axis=1).to_numpy()
    # Continuous variant: same components, signed so that POSITIVE = bottom/oversold = bullish.
    cont = (-delta_z.fillna(0.0) - ret_z.fillna(0.0)
            + (0.5 - p_fast.fillna(0.5)) * 4.0 + (0.5 - p_slow.fillna(0.5)) * 4.0
            + (0.5 - pctb.clip(0, 1).fillna(0.5)) * 4.0
            + vol_z.fillna(0.0).clip(0, 5) * (lower_wick.fillna(0.0) - upper_wick.fillna(0.0)))
    out["continuous"] = cont.to_numpy()
    out["delta_z"] = delta_z.to_numpy()
    out["close"] = close.to_numpy()
    return out


# =====================================================================================================
# Stage A -- proxy validation on the two symbols that have both raw klines and a panel parquet.
# =====================================================================================================
def _raw_delta_z(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False, usecols=["timestamp", "volume", "taker_buy_base"])
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    delta = 2.0 * df["taker_buy_base"] - df["volume"]
    return pd.DataFrame({"timestamp": df["timestamp"], "delta_z_true": _z(delta, DZ_WINDOW)})


def _panel_proxy_delta_z(sym: str) -> pd.DataFrame:
    p = pd.read_parquet(PANEL_DIR / f"{sym}.parquet", columns=["timestamp", "rvol_48", "taker_buy_ratio"])
    proxy = p["rvol_48"] * (2.0 * p["taker_buy_ratio"] - 1.0)
    return pd.DataFrame({"timestamp": p["timestamp"], "delta_z_proxy": _z(proxy, DZ_WINDOW)})


def stage_a() -> dict[str, Any]:
    res: dict[str, Any] = {}
    for sym, raw in (("ETHUSDT", ETH_RAW), ("BTCUSDT", BTC_RAW)):
        m = _raw_delta_z(raw).merge(_panel_proxy_delta_z(sym), on="timestamp", how="inner").dropna()
        rho = float(spearmanr(m["delta_z_true"], m["delta_z_proxy"]).statistic) if len(m) else float("nan")
        pear = float(np.corrcoef(m["delta_z_true"], m["delta_z_proxy"])[0, 1]) if len(m) else float("nan")
        jac = {}
        for side, cond_t, cond_p in (("bottom", m["delta_z_true"] <= -DZ_ABS, m["delta_z_proxy"] <= -DZ_ABS),
                                     ("top", m["delta_z_true"] >= DZ_ABS, m["delta_z_proxy"] >= DZ_ABS)):
            inter, union = int((cond_t & cond_p).sum()), int((cond_t | cond_p).sum())
            jac[side] = {"true_events": int(cond_t.sum()), "proxy_events": int(cond_p.sum()),
                         "intersection": inter, "jaccard": float(inter / union) if union else 0.0}
        res[sym] = {"overlap_rows": int(len(m)), "spearman": rho, "pearson": pear, "event_agreement": jac,
                    "pass": bool(rho >= PROXY_GATE_SPEARMAN and min(jac["bottom"]["jaccard"], jac["top"]["jaccard"]) >= PROXY_GATE_JACCARD)}
        log(f"  {sym}: rows={len(m)} spearman={rho:.4f} pearson={pear:.4f} "
            f"jaccard_bottom={jac['bottom']['jaccard']:.3f} jaccard_top={jac['top']['jaccard']:.3f} pass={res[sym]['pass']}")
    res["gate_pass"] = bool(all(res[s]["pass"] for s in ("ETHUSDT", "BTCUSDT")))
    return res


# =====================================================================================================
# Stage B/C -- cross-sectional IC and long-short portfolio.
# =====================================================================================================
def _ic_series(score_w: pd.DataFrame, fwd_w: pd.DataFrame, h: int) -> pd.Series:
    """Spearman IC per bar, sampled every h bars so observations do not overlap."""
    idx = np.arange(0, len(score_w), h)
    out = {}
    s_np, f_np = score_w.to_numpy(), fwd_w.to_numpy()
    ts = score_w.index.to_numpy()
    for i in idx:
        s, f = s_np[i], f_np[i]
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() < MIN_SYMBOLS_PER_BAR or np.nanstd(s[ok]) == 0:
            continue
        out[ts[i]] = spearmanr(s[ok], f[ok]).statistic
    return pd.Series(out, dtype=float).dropna()


def _summ(ic: pd.Series) -> dict[str, Any]:
    n = int(len(ic))
    mean = float(ic.mean()) if n else float("nan")
    sd = float(ic.std(ddof=1)) if n > 1 else float("nan")
    return {"n_obs": n, "mean_ic": mean, "std_ic": sd,
            "t_stat": float(mean / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
            "pct_positive": float((ic > 0).mean()) if n else float("nan")}


def _neutralize(score_w: pd.DataFrame, factor_w: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectionally regress score on the benchmark factor bar-by-bar, keep residuals."""
    s, f = score_w.to_numpy(dtype=float), factor_w.to_numpy(dtype=float)
    resid = np.full_like(s, np.nan)
    for i in range(s.shape[0]):
        ok = np.isfinite(s[i]) & np.isfinite(f[i])
        if ok.sum() < MIN_SYMBOLS_PER_BAR:
            continue
        x, y = f[i][ok], s[i][ok]
        x_c = x - x.mean()
        denom = float((x_c ** 2).sum())
        beta = float((x_c * (y - y.mean())).sum() / denom) if denom > 0 else 0.0
        resid[i][ok] = (y - y.mean()) - beta * x_c
    return pd.DataFrame(resid, index=score_w.index, columns=score_w.columns)


def _long_short(score_w: pd.DataFrame, fwd_w: pd.DataFrame, h: int) -> dict[str, Any]:
    """Dollar-neutral top/bottom quintile, rebalanced every h bars. Turnover cost charged as
    COST_ROUNDTRIP on the fraction of the book that changes between consecutive rebalances."""
    idx = np.arange(0, len(score_w), h)
    s_np, f_np = score_w.to_numpy(dtype=float), fwd_w.to_numpy(dtype=float)
    prev_w = None
    gross, net = [], []
    for i in idx:
        s, f = s_np[i], f_np[i]
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() < MIN_SYMBOLS_PER_BAR:
            continue
        k = max(int(ok.sum() // 5), 1)
        # invalid symbols pushed to the far end of each ordering so they can never be selected
        longs = np.argsort(np.where(ok, s, -np.inf))[-k:]
        shorts = np.argsort(np.where(ok, s, np.inf))[:k]
        w = np.zeros_like(s)
        w[longs], w[shorts] = 0.5 / k, -0.5 / k
        g = float(np.nansum(w * np.where(np.isfinite(f), f, 0.0)))
        turn = float(np.abs(w - prev_w).sum()) if prev_w is not None else float(np.abs(w).sum())
        gross.append(g)
        net.append(g - turn * COST_ROUNDTRIP / 2.0)
        prev_w = w
    g_arr, n_arr = np.array(gross), np.array(net)
    per_year = (288 * 365) / h
    def _stats(a: np.ndarray) -> dict[str, Any]:
        if not len(a):
            return {"n": 0}
        return {"n": int(len(a)), "mean_per_period_pct": float(a.mean() * 100), "cum_pct": float((np.prod(1 + a) - 1) * 100),
                "ann_sharpe": float(a.mean() / a.std(ddof=1) * np.sqrt(per_year)) if a.std(ddof=1) > 0 else float("nan"),
                "hit_rate": float((a > 0).mean())}
    return {"gross": _stats(g_arr), "net_of_cost": _stats(n_arr)}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "design": "Evidence signals as a dollar-neutral cross-sectional factor on the 60-perp panel; "
                  "pre-registered controls: permutation, short-term-reversal benchmark + neutralization, period split.",
        "pre_registered": {"horizons": list(HORIZONS), "score_variants": ["votes", "continuous"],
                           "controls": ["permutation", "reversal_benchmark", "reversal_neutralized", "period_split"],
                           "perm_reps": PERM_REPS, "perm_seed": PERM_SEED, "cost_roundtrip": COST_ROUNDTRIP},
        "universe_caveats": ["liquidity_lookahead (top-60 by volume at 2026-08-04)", "survivorship (only currently-trading perps)"],
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
    }

    log("=== stage=A_proxy_validation ===")
    stage_a_res = stage_a()
    report["stage_a_proxy_validation"] = stage_a_res
    if not stage_a_res["gate_pass"]:
        report["gate_pass"] = False
        report["note"] = ("Proxy validation FAILED -- the panel-computable order-flow proxy does not reproduce the raw "
                          "delta_z well enough on ETH/BTC. Panel-wide results are NOT computed; raw klines "
                          "(taker_buy_base/volume) must be re-downloaded for the 60 symbols first.")
        (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        log("stage=ABORT proxy gate failed")
        return 1

    log("=== stage=B_build_panel_signals ===")
    symbols = [s["symbol"] for s in json.loads(UNIVERSE_JSON.read_text())["symbols"]]
    votes, cont, closes = {}, {}, {}
    for i, sym in enumerate(symbols):
        f = PANEL_DIR / f"{sym}.parquet"
        if not f.exists():
            log(f"  SKIP {sym}: no parquet")
            continue
        p = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "rvol_48", "taker_buy_ratio"])
        p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        sig = build_signals(p, delta_series=p["rvol_48"] * (2.0 * p["taker_buy_ratio"] - 1.0), volume_series=p["rvol_48"])
        votes[sym] = sig.set_index("timestamp")["votes"]
        cont[sym] = sig.set_index("timestamp")["continuous"]
        closes[sym] = sig.set_index("timestamp")["close"]
        if (i + 1) % 10 == 0:
            log(f"  built {i + 1}/{len(symbols)}")
    votes_w, cont_w, close_w = pd.DataFrame(votes), pd.DataFrame(cont), pd.DataFrame(closes)
    votes_w, cont_w, close_w = votes_w.sort_index(), cont_w.sort_index(), close_w.sort_index()
    log(f"  panel matrix: {close_w.shape[0]} bars x {close_w.shape[1]} symbols "
        f"[{close_w.index.min()} .. {close_w.index.max()}]")
    report["panel"] = {"symbols": int(close_w.shape[1]), "bars": int(close_w.shape[0]),
                       "start": str(close_w.index.min()), "end": str(close_w.index.max())}

    log("=== stage=B_ic ===")
    rng = np.random.default_rng(PERM_SEED)
    results: dict[str, Any] = {}
    for h in HORIZONS:
        fwd_w = close_w.shift(-h) / close_w - 1.0
        past_w = close_w / close_w.shift(h) - 1.0
        reversal_w = -past_w
        h_res: dict[str, Any] = {}
        for name, score_w in (("votes", votes_w), ("continuous", cont_w)):
            ic = _ic_series(score_w, fwd_w, h)
            neut = _ic_series(_neutralize(score_w, reversal_w), fwd_w, h)
            per_period = {}
            for pname, (a, b) in PERIODS.items():
                sl = ic.loc[(ic.index >= pd.Timestamp(a)) & (ic.index <= pd.Timestamp(b))]
                per_period[pname] = _summ(sl)
            h_res[name] = {"overall": _summ(ic), "reversal_neutralized": _summ(neut),
                           "per_period": per_period, "long_short": _long_short(score_w, fwd_w, h)}
            log(f"  h={h:3d} {name:10s} mean_IC={h_res[name]['overall']['mean_ic']:+.5f} "
                f"t={h_res[name]['overall']['t_stat']:+.2f} n={h_res[name]['overall']['n_obs']:5d} "
                f"| neutralized mean_IC={h_res[name]['reversal_neutralized']['mean_ic']:+.5f} "
                f"t={h_res[name]['reversal_neutralized']['t_stat']:+.2f} "
                f"| LS net cum={h_res[name]['long_short']['net_of_cost'].get('cum_pct', float('nan')):+.1f}% "
                f"sharpe={h_res[name]['long_short']['net_of_cost'].get('ann_sharpe', float('nan')):+.2f}")
        bench = _ic_series(reversal_w, fwd_w, h)
        h_res["reversal_benchmark"] = {"overall": _summ(bench), "long_short": _long_short(reversal_w, fwd_w, h)}
        log(f"  h={h:3d} {'REVERSAL':10s} mean_IC={h_res['reversal_benchmark']['overall']['mean_ic']:+.5f} "
            f"t={h_res['reversal_benchmark']['overall']['t_stat']:+.2f} "
            f"| LS net cum={h_res['reversal_benchmark']['long_short']['net_of_cost'].get('cum_pct', float('nan')):+.1f}%")
        perm_means = []
        for _ in range(PERM_REPS):
            sh = cont_w.to_numpy(dtype=float).copy()
            for i in range(sh.shape[0]):
                rng.shuffle(sh[i])
            perm_means.append(_summ(_ic_series(pd.DataFrame(sh, index=cont_w.index, columns=cont_w.columns), fwd_w, h))["mean_ic"])
        h_res["permutation_control"] = {"reps": PERM_REPS, "mean_of_means": float(np.mean(perm_means)),
                                        "std_of_means": float(np.std(perm_means))}
        log(f"  h={h:3d} {'PERMUTED':10s} mean_IC={np.mean(perm_means):+.5f} (sd {np.std(perm_means):.5f})")
        results[str(h)] = h_res

    report["results_by_horizon"] = results
    report["gate_pass"] = True
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log("stage=done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
