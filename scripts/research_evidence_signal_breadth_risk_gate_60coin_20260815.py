#!/usr/bin/env python3
"""RESEARCH -- Evidence-signal BREADTH as a market-wide risk gate on the 60-perp panel (2026-08-15).

Sub-project candidate #1 (docs/model_contracts/evidence_signal_quant_use_contract_20260815.md).

=== Why breadth, when the per-symbol signal already failed economically ===
The companion studies established two things: the evidence signals carry real cross-sectional
information (rank IC significant in 4/4 periods, permutation control ~0) but the per-symbol edge is
~0.24bp/event against a >=10bp round trip -- one to two orders of magnitude too small to trade.
Breadth is a structurally different object from the per-symbol signal:
  (a) averaging the firing indicator over 60 symbols suppresses exactly the weakness that killed the
      per-symbol use (tiny per-unit effect drowned in idiosyncratic noise);
  (b) it is used to SIZE EXPOSURE, not to pick entries, so it does not pay per-symbol round trips --
      the turnover-cost mechanism that annihilated both the full-panel long-short (17x gross) and
      the event-driven construction is bypassed by design;
  (c) it matches the signals' one validated property -- they fire near genuine pivots -- promoted
      from "this coin is turning" (too weak per unit) to "the whole market is capitulating".

=== PRE-REGISTERED, fixed before any result was inspected ===
Signal:
  breadth_bottom[t] = fraction of valid symbols with net votes >= K
  breadth_top[t]    = fraction of valid symbols with net votes <= -K      (K = 2, same as the
                      event study; K in {1,3} reported as robustness only, never as the verdict)
Extreme:
  breadth in its OWN CAUSAL rolling percentile (window 8064 bars = 28 days, min_periods full) at or
  above P in {0.95, 0.99}. P=0.99 is the primary; P=0.95 is robustness. No full-sample quantile is
  used anywhere.
Episodes:
  consecutive/nearby extreme bars merged into one episode if within h bars of each other, and each
  episode contributes exactly ONE observation. This is the multiplicity control: extremes arrive in
  clusters, and counting every bar would inflate every t-stat in the study.
Horizons: h in {48, 144, 288, 864} bars (4h / 12h / 24h / 3d).
DIRECTIONAL HYPOTHESIS (fixed now so a sign flip counts as failure, not as a finding):
  bottom-breadth extreme (mass capitulation) -> forward panel return HIGHER than unconditional;
  top-breadth extreme (mass euphoria)        -> forward panel return LOWER  than unconditional;
  both extremes                              -> forward realized volatility HIGHER, and the forward
                                                left tail (5th pct) FATTER, than unconditional.
Null:
  random-episode bootstrap -- the same number of episode start bars drawn uniformly from the same
  valid range, 1000 replicates, seed 20260815. This preserves the return series' own autocorrelation
  and the episode count, so the comparison isolates "did breadth pick these moments" rather than
  "are overlapping forward windows autocorrelated". Empirical two-sided p-values.
KILL CRITERION (contract-mandated, fixed now):
  the candidate is CLOSED unless, for the primary P=0.99 and at >=1 horizon, the hypothesised shift
  holds with pooled empirical p < 0.05 AND the same sign appears in >= 3 of the 4 period splits
  (2024 / 2025H1 / 2025H2 / 2026). A shift that is significant pooled but sign-inconsistent across
  periods is a FAILURE, not a partial success.
Economic test (contract requires a breakeven cost in bp, not just a distribution shift):
  gate a passive equal-weight long panel position -- exposure 0 for h bars after each top-breadth
  extreme, exposure 1 otherwise -- and report return / Sharpe / MDD vs ungated, the induced
  turnover, and the round-trip cost at which the gate stops paying. A gate that improves risk but
  costs more than it saves is reported as such.

Limitations restated (they bind here too): the 60-symbol universe carries liquidity-lookahead and
survivorship bias (data/splits/panel_universe_symbols_20260804.json) -- for a TIMING gate this
matters less than for a cross-sectional alpha, but the passive panel return being gated is itself a
survivorship-inflated series, so the economic test's absolute level is optimistic and only the
GATED-vs-UNGATED difference should be read. The order-flow term is the panel proxy validated at
Spearman 0.97/0.96 against raw ETH/BTC klines.

Causal: breadth, its rolling percentile, and every input are rolling/shift only; forward windows are
outcomes exclusively. trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training, no GPU, no live files touched. Imports
research_evidence_signal_cross_sectional_ic_60coin_20260815 read-only and reuses build_signals
verbatim.
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

import research_evidence_signal_cross_sectional_ic_60coin_20260815 as xs  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/evidence_signal_breadth_risk_gate_60coin_20260815"
K_PRIMARY, K_ROBUST = 2, (1, 3)
PCT_WINDOW = 8064               # 28 days of 5m bars, causal rolling
P_PRIMARY, P_ROBUST = 0.99, 0.95
HORIZONS = (48, 144, 288, 864)
BOOT_REPS, BOOT_SEED = 1000, 20260815
COST_ROUNDTRIP = 0.001
PERIODS = xs.PERIODS


def log(msg: str) -> None:
    print(f"[breadth_gate] {msg}", flush=True)


def build_breadth() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Returns (breadth frame indexed by timestamp, panel EW per-bar return, ETH per-bar return)."""
    symbols = [s["symbol"] for s in json.loads(xs.UNIVERSE_JSON.read_text())["symbols"]]
    net, closes = {}, {}
    for i, sym in enumerate(symbols):
        f = xs.PANEL_DIR / f"{sym}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "rvol_48", "taker_buy_ratio"])
        p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        sig = xs.build_signals(p, delta_series=p["rvol_48"] * (2.0 * p["taker_buy_ratio"] - 1.0), volume_series=p["rvol_48"])
        net[sym] = sig.set_index("timestamp")["votes"]
        closes[sym] = sig.set_index("timestamp")["close"]
        if (i + 1) % 20 == 0:
            log(f"  built {i + 1}/{len(symbols)}")
    net_w = pd.DataFrame(net).sort_index()
    close_w = pd.DataFrame(closes).sort_index()
    ret_w = close_w.pct_change()
    ew_ret = ret_w.mean(axis=1, skipna=True)
    eth_ret = ret_w["ETHUSDT"] if "ETHUSDT" in ret_w.columns else pd.Series(index=ret_w.index, dtype=float)

    valid = net_w.notna()
    n_valid = valid.sum(axis=1)
    out = pd.DataFrame(index=net_w.index)
    out["n_valid"] = n_valid
    for K in (K_PRIMARY, *K_ROBUST):
        out[f"bottom_K{K}"] = (net_w >= K).sum(axis=1) / n_valid.replace(0, np.nan)
        out[f"top_K{K}"] = (net_w <= -K).sum(axis=1) / n_valid.replace(0, np.nan)
    return out, ew_ret, eth_ret


def _episodes(mask: np.ndarray, merge_within: int) -> np.ndarray:
    """Indices of episode starts: a True bar begins a new episode only if no True bar occurred in
    the previous `merge_within` bars."""
    idx = np.flatnonzero(mask)
    if not len(idx):
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] > merge_within:
            keep.append(i)
    return np.array(keep, dtype=int)


def _forward_stats(ew_ret: np.ndarray, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Forward compounded return and forward realized vol over (t, t+h], per bar."""
    n = len(ew_ret)
    logret = np.log1p(np.nan_to_num(ew_ret, nan=0.0))
    cum = np.concatenate([[0.0], np.cumsum(logret)])
    fwd = np.full(n, np.nan)
    fwd[:n - h] = np.expm1(cum[h + 1:n + 1] - cum[1:n - h + 1])
    sq = np.concatenate([[0.0], np.cumsum(logret ** 2)])
    vol = np.full(n, np.nan)
    mean_ = (cum[h + 1:n + 1] - cum[1:n - h + 1]) / h
    msq = (sq[h + 1:n + 1] - sq[1:n - h + 1]) / h
    vol[:n - h] = np.sqrt(np.maximum(msq - mean_ ** 2, 0.0))
    return fwd, vol


def _boot_p(observed: float, null_means: np.ndarray) -> float:
    """Two-sided empirical p-value against the random-episode null."""
    centre = float(np.mean(null_means))
    return float((np.abs(null_means - centre) >= abs(observed - centre)).mean())


def analyse(breadth: pd.DataFrame, ew_ret: pd.Series, eth_ret: pd.Series) -> dict[str, Any]:
    ts = breadth.index.to_numpy()
    ew = ew_ret.to_numpy(dtype=float)
    eth = eth_ret.to_numpy(dtype=float)
    rng = np.random.default_rng(BOOT_SEED)
    results: dict[str, Any] = {}

    for K in (K_PRIMARY, *K_ROBUST):
        for P in (P_PRIMARY, P_ROBUST):
            for side in ("bottom", "top"):
                col = f"{side}_K{K}"
                b = breadth[col]
                # causal rolling percentile rank of breadth within its own trailing 28d window
                pct = b.rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True).to_numpy()
                extreme = np.nan_to_num(pct, nan=0.0) >= P
                for h in HORIZONS:
                    fwd, vol = _forward_stats(ew, h)
                    fwd_eth, _ = _forward_stats(eth, h)
                    ok = np.isfinite(fwd) & np.isfinite(vol) & (np.arange(len(ts)) >= PCT_WINDOW)
                    ep = _episodes(extreme & ok, merge_within=h)
                    if len(ep) < 5:
                        results[f"{side}_K{K}_P{P}_h{h}"] = {"episodes": int(len(ep)), "underpowered": True}
                        continue
                    obs_ret, obs_vol, obs_eth = float(np.mean(fwd[ep])), float(np.mean(vol[ep])), float(np.nanmean(fwd_eth[ep]))
                    obs_tail = float(np.percentile(fwd[ep], 5))
                    pool = np.flatnonzero(ok)
                    null_ret, null_vol, null_tail, null_eth = [], [], [], []
                    for _ in range(BOOT_REPS):
                        draw = rng.choice(pool, size=len(ep), replace=False)
                        null_ret.append(float(np.mean(fwd[draw])))
                        null_vol.append(float(np.mean(vol[draw])))
                        null_tail.append(float(np.percentile(fwd[draw], 5)))
                        null_eth.append(float(np.nanmean(fwd_eth[draw])))
                    null_ret, null_vol = np.array(null_ret), np.array(null_vol)
                    null_tail, null_eth = np.array(null_tail), np.array(null_eth)

                    per_period = {}
                    for pname, (a, bnd) in PERIODS.items():
                        sel = ep[(ts[ep] >= np.datetime64(pd.Timestamp(a))) & (ts[ep] <= np.datetime64(pd.Timestamp(bnd)))]
                        base = np.flatnonzero(ok & (ts >= np.datetime64(pd.Timestamp(a))) & (ts <= np.datetime64(pd.Timestamp(bnd))))
                        per_period[pname] = {
                            "episodes": int(len(sel)),
                            "mean_fwd_ret_pct": float(np.mean(fwd[sel]) * 100) if len(sel) else float("nan"),
                            "uncond_fwd_ret_pct": float(np.mean(fwd[base]) * 100) if len(base) else float("nan"),
                            "mean_fwd_vol": float(np.mean(vol[sel])) if len(sel) else float("nan"),
                            "uncond_fwd_vol": float(np.mean(vol[base])) if len(base) else float("nan"),
                        }
                    exp_sign = 1.0 if side == "bottom" else -1.0
                    sign_ok = sum(1 for v in per_period.values()
                                  if np.isfinite(v["mean_fwd_ret_pct"]) and np.isfinite(v["uncond_fwd_ret_pct"])
                                  and exp_sign * (v["mean_fwd_ret_pct"] - v["uncond_fwd_ret_pct"]) > 0)
                    vol_sign_ok = sum(1 for v in per_period.values()
                                      if np.isfinite(v["mean_fwd_vol"]) and np.isfinite(v["uncond_fwd_vol"])
                                      and v["mean_fwd_vol"] > v["uncond_fwd_vol"])
                    key = f"{side}_K{K}_P{P}_h{h}"
                    results[key] = {
                        "episodes": int(len(ep)), "underpowered": False,
                        "fwd_ret_pct": obs_ret * 100, "null_fwd_ret_pct": float(null_ret.mean()) * 100,
                        "fwd_ret_p": _boot_p(obs_ret, null_ret),
                        "fwd_ret_direction_as_hypothesised": bool(exp_sign * (obs_ret - null_ret.mean()) > 0),
                        "fwd_vol": obs_vol, "null_fwd_vol": float(null_vol.mean()), "fwd_vol_p": _boot_p(obs_vol, null_vol),
                        "fwd_vol_higher_as_hypothesised": bool(obs_vol > null_vol.mean()),
                        "fwd_tail5_pct": obs_tail * 100, "null_tail5_pct": float(null_tail.mean()) * 100,
                        "fwd_tail_p": _boot_p(obs_tail, null_tail),
                        "eth_fwd_ret_pct": obs_eth * 100, "null_eth_fwd_ret_pct": float(null_eth.mean()) * 100,
                        "eth_fwd_ret_p": _boot_p(obs_eth, null_eth),
                        "per_period": per_period,
                        "period_sign_consistency_ret": int(sign_ok),
                        "period_sign_consistency_vol": int(vol_sign_ok),
                    }
                    if K == K_PRIMARY and P == P_PRIMARY:
                        log(f"  {side:6s} K={K} P={P} h={h:4d}: ep={len(ep):4d} "
                            f"fwd_ret={obs_ret * 100:+.3f}% (null {null_ret.mean() * 100:+.3f}%, p={results[key]['fwd_ret_p']:.3f}, "
                            f"dir_ok={results[key]['fwd_ret_direction_as_hypothesised']}, periods={sign_ok}/4) | "
                            f"vol={obs_vol:.5f} vs {null_vol.mean():.5f} p={results[key]['fwd_vol_p']:.3f} periods={vol_sign_ok}/4 | "
                            f"tail5={obs_tail * 100:+.2f}% vs {null_tail.mean() * 100:+.2f}% p={results[key]['fwd_tail_p']:.3f}")
    return results


def economic_gate(breadth: pd.DataFrame, ew_ret: pd.Series, h: int, K: int, P: float) -> dict[str, Any]:
    """Passive EW long panel, exposure cut to 0 for h bars after each top-breadth extreme."""
    b = breadth[f"top_K{K}"]
    pct = b.rolling(PCT_WINDOW, min_periods=PCT_WINDOW).rank(pct=True).to_numpy()
    extreme = np.nan_to_num(pct, nan=0.0) >= P
    n = len(b)
    expo = np.ones(n)
    fire = np.flatnonzero(extreme)
    for i in fire:
        expo[i + 1:min(i + 1 + h, n)] = 0.0
    r = np.nan_to_num(ew_ret.to_numpy(dtype=float), nan=0.0)
    warm = np.arange(n) >= PCT_WINDOW
    r, expo = r[warm], expo[warm]
    turns = float(np.abs(np.diff(np.concatenate([[1.0], expo]))).sum())
    gated_gross = expo * r
    cost_per_switch = COST_ROUNDTRIP / 2.0
    cost_vec = np.zeros_like(gated_gross)
    switch_idx = np.flatnonzero(np.abs(np.diff(np.concatenate([[1.0], expo]))) > 0)
    cost_vec[switch_idx] = cost_per_switch
    gated_net = gated_gross - cost_vec

    def _stats(x: np.ndarray) -> dict[str, Any]:
        cum = np.cumprod(1.0 + x)
        peak = np.maximum.accumulate(cum)
        return {"total_return_pct": float((cum[-1] - 1) * 100),
                "ann_sharpe": float(x.mean() / x.std(ddof=1) * np.sqrt(288 * 365)) if x.std(ddof=1) > 0 else float("nan"),
                "mdd_pct": float((cum / peak - 1).min() * 100),
                "exposure_frac": float((x != 0).mean())}
    base, gross, net = _stats(r), _stats(gated_gross), _stats(gated_net)
    ret_diff = float(np.mean(gated_gross - r))
    breakeven_bps = float(ret_diff * len(r) / max(turns, 1e-9) * 10000.0)
    return {"h": h, "K": K, "P": P, "switches": int(turns), "ungated": base, "gated_gross": gross,
            "gated_net_of_cost": net, "exposure_frac": float(expo.mean()),
            "breakeven_roundtrip_bps": breakeven_bps,
            "note": "breakeven_roundtrip_bps = total gross return advantage of gating, divided by the number of "
                    "exposure switches, expressed in bps -- the round-trip cost at which the gate stops paying. "
                    "Negative means gating loses money before any cost is charged."}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=build_breadth ===")
    breadth, ew_ret, eth_ret = build_breadth()
    log(f"  bars={len(breadth)} [{breadth.index.min()} .. {breadth.index.max()}] "
        f"median_valid_symbols={float(breadth['n_valid'].median()):.0f}")
    log(f"  breadth_bottom_K2 mean={breadth['bottom_K2'].mean():.4f} p99={breadth['bottom_K2'].quantile(0.99):.4f} | "
        f"top_K2 mean={breadth['top_K2'].mean():.4f} p99={breadth['top_K2'].quantile(0.99):.4f}")

    log("=== stage=episode_analysis ===")
    results = analyse(breadth, ew_ret, eth_ret)

    log("=== stage=economic_gate ===")
    econ = {}
    for h in HORIZONS:
        econ[f"h{h}"] = economic_gate(breadth, ew_ret, h, K_PRIMARY, P_PRIMARY)
        e = econ[f"h{h}"]
        log(f"  h={h:4d}: ungated ret={e['ungated']['total_return_pct']:+.1f}% sharpe={e['ungated']['ann_sharpe']:+.2f} "
            f"mdd={e['ungated']['mdd_pct']:.1f}% | gated_gross ret={e['gated_gross']['total_return_pct']:+.1f}% "
            f"sharpe={e['gated_gross']['ann_sharpe']:+.2f} mdd={e['gated_gross']['mdd_pct']:.1f}% "
            f"| net ret={e['gated_net_of_cost']['total_return_pct']:+.1f}% | exposure={e['exposure_frac']:.3f} "
            f"switches={e['switches']} breakeven={e['breakeven_roundtrip_bps']:+.2f}bps")

    # ---- pre-registered kill criterion, evaluated mechanically ----
    verdict_rows = []
    for side in ("bottom", "top"):
        for h in HORIZONS:
            r = results.get(f"{side}_K{K_PRIMARY}_P{P_PRIMARY}_h{h}", {})
            if r.get("underpowered", True):
                continue
            verdict_rows.append({
                "key": f"{side}_h{h}",
                "ret_pass": bool(r["fwd_ret_direction_as_hypothesised"] and r["fwd_ret_p"] < 0.05
                                 and r["period_sign_consistency_ret"] >= 3),
                "vol_pass": bool(r["fwd_vol_higher_as_hypothesised"] and r["fwd_vol_p"] < 0.05
                                 and r["period_sign_consistency_vol"] >= 3),
            })
    ret_pass = any(v["ret_pass"] for v in verdict_rows)
    vol_pass = any(v["vol_pass"] for v in verdict_rows)
    verdict = "SURVIVES" if (ret_pass or vol_pass) else "CLOSED_FAILED_KILL_CRITERION"

    report = {
        "design": "Evidence-signal breadth as a market-wide risk gate; pre-registered directional hypothesis, "
                  "episode-level observations, random-episode bootstrap null, 4-period sign consistency, "
                  "and an exposure-gate economic test reporting breakeven round-trip cost in bps.",
        "pre_registered": {"K_primary": K_PRIMARY, "K_robustness": list(K_ROBUST), "pct_window_bars": PCT_WINDOW,
                           "P_primary": P_PRIMARY, "P_robustness": P_ROBUST, "horizons": list(HORIZONS),
                           "bootstrap_reps": BOOT_REPS, "bootstrap_seed": BOOT_SEED,
                           "hypothesis": "bottom-extreme -> higher forward panel return; top-extreme -> lower; "
                                         "both -> higher forward vol and fatter left tail",
                           "kill_criterion": "primary (K=2,P=0.99) must show, at >=1 horizon, the hypothesised "
                                             "direction with pooled p<0.05 AND >=3/4 period sign consistency"},
        "universe_caveats": ["liquidity_lookahead", "survivorship -- the gated passive series is itself inflated; "
                             "read only the gated-vs-ungated difference"],
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "breadth_summary": {"bars": int(len(breadth)), "start": str(breadth.index.min()), "end": str(breadth.index.max()),
                            "bottom_K2_mean": float(breadth["bottom_K2"].mean()), "top_K2_mean": float(breadth["top_K2"].mean())},
        "episode_analysis": results, "economic_gate": econ,
        "kill_criterion_rows": verdict_rows, "return_test_passes": ret_pass, "vol_test_passes": vol_pass,
        "verdict": verdict,
    }
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done VERDICT={verdict} (return_test={ret_pass}, vol_test={vol_pass})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
