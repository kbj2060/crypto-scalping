#!/usr/bin/env python3
"""RESEARCH -- Liquidity-bucket decomposition of IC and cost (2026-08-15).

Sub-project candidate #1 (docs/model_contracts/evidence_signal_quant_use_contract_20260815.md),
derived from the industry survey (docs/experiments/evidence_signal_industry_practice_survey_20260815.md).

=== The question this settles ===
de Groot/Huij/Zhou (JBF 2012) showed short-term reversal survives costs at 30-50bp/week once the
universe is restricted to large caps and turnover is reduced -- the cost problem was attributable to
over-trading small caps. Our own studies charged a FLAT 10bp to all 60 symbols and rebalanced the
whole panel every h bars, i.e. exactly the construction that paper blames. But crypto may invert the
prescription: published work finds daily reversals concentrate in ILLIQUID coins while the largest,
most tradeable coins show daily MOMENTUM instead -- the classic "the alpha is where you cannot trade
it" trap. Both claims are testable on the panel we already have, and they point opposite ways, so
this is decided with numbers rather than by importing an equity prescription.

=== PRE-REGISTERED, fixed before any result was inspected ===
Buckets (static, by the universe manifest's own descending 24h quote-volume ordering):
    T10   = ranks 1-10   (BTC, ETH, SOL, XRP, ZEC, 1000RATS, ADA, BICO, DOGE, BNB)
    T11_30 = ranks 11-30
    T31_60 = ranks 31-60
  KNOWN BIAS, not corrected: that ordering is measured at ONE instant (2026-08-04), so bucket
  membership is forward-looking with respect to 2024-2025. It is used because the panel feature
  store carries no absolute volume (only rvol_48, which is normalised per symbol and therefore
  cannot rank symbols against each other). Mitigation: results are also reported per period, and a
  2024-only sub-period result that agrees with the pooled one is evidence the staleness does not
  drive the finding. This is a diagnostic, never a promotion basis.
Scores: `continuous` and `votes`, both reused verbatim from the companion IC study.
Horizons: h in {12, 48, 144} bars.
Metrics, per bucket x horizon:
  (1) WITHIN-BUCKET cross-sectional rank IC (sampled every h bars, non-overlapping);
  (2) WITHIN-BUCKET dollar-neutral top/bottom-quintile long-short: gross return per rebalance,
      realised turnover, and the BREAKEVEN round-trip cost in bp at which it stops paying;
  (3) EVENT study: mean excess return per event, where excess is measured against that BUCKET's own
      equal-weight return (the correct neutraliser for a within-bucket book), and the breakeven
      round-trip cost in bp.
Reference cost levels (stated, not assumed into the verdict -- breakeven bp is reported so any
other schedule can be applied): Binance USD-M futures taker 5bp/side => 10bp round trip; maker
2bp/side => 4bp round trip; spread and impact NOT modelled, and they are worst exactly in T31_60.
KILL CRITERION (contract-mandated, fixed now):
  path B ("attack turnover/universe") is CLOSED unless at least one bucket x horizon shows a
  breakeven round-trip cost above 4bp (the best realistic maker round trip) AND the same sign in
  >= 3 of the 4 period splits. A breakeven that only clears 0bp is a failure.

Causal: all inputs rolling/shift only; forward windows are outcomes only.
trade_ledgers_used_as_input=false, saved_parent_exit_timestamps_used=false,
future_rows_used_for_entry=false. No training, no GPU, no live files touched. Imports
research_evidence_signal_cross_sectional_ic_60coin_20260815 read-only (build_signals reused verbatim).
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

import research_evidence_signal_cross_sectional_ic_60coin_20260815 as xs  # noqa: E402

OUT_DIR = ROOT / "tmp/causal_regen_20260516/evidence_signal_liquidity_bucket_ic_cost_60coin_20260815"
BUCKETS = {"T10": (0, 10), "T11_30": (10, 30), "T31_60": (30, 60)}
HORIZONS = (12, 48, 144)
K_EVENT = 2
MIN_SYMBOLS = {"T10": 8, "T11_30": 15, "T31_60": 22}
TAKER_ROUNDTRIP_BPS, MAKER_ROUNDTRIP_BPS = 10.0, 4.0
PERIODS = xs.PERIODS


def log(msg: str) -> None:
    print(f"[liq_bucket] {msg}", flush=True)


def build_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    order = [s["symbol"] for s in json.loads(xs.UNIVERSE_JSON.read_text())["symbols"]]
    votes, cont, closes, kept = {}, {}, {}, []
    for i, sym in enumerate(order):
        f = xs.PANEL_DIR / f"{sym}.parquet"
        if not f.exists():
            continue
        p = pd.read_parquet(f, columns=["timestamp", "open", "high", "low", "close", "rvol_48", "taker_buy_ratio"])
        p = p.sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
        sig = xs.build_signals(p, delta_series=p["rvol_48"] * (2.0 * p["taker_buy_ratio"] - 1.0), volume_series=p["rvol_48"])
        votes[sym] = sig.set_index("timestamp")["votes"]
        cont[sym] = sig.set_index("timestamp")["continuous"]
        closes[sym] = sig.set_index("timestamp")["close"]
        kept.append(sym)
        if (i + 1) % 20 == 0:
            log(f"  built {i + 1}/{len(order)}")
    return (pd.DataFrame(votes).sort_index()[kept], pd.DataFrame(cont).sort_index()[kept],
            pd.DataFrame(closes).sort_index()[kept], kept)


def _summ(a: np.ndarray) -> dict[str, Any]:
    n = int(len(a))
    if n == 0:
        return {"n": 0}
    m = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else float("nan")
    return {"n": n, "mean": m, "t_stat": float(m / (sd / np.sqrt(n))) if n > 1 and sd > 0 else float("nan"),
            "pct_positive": float((a > 0).mean())}


def bucket_analysis(score_w: pd.DataFrame, close_w: pd.DataFrame, votes_w: pd.DataFrame,
                    h: int, min_sym: int) -> dict[str, Any]:
    s_np = score_w.to_numpy(dtype=float)
    c_np = close_w.to_numpy(dtype=float)
    v_np = votes_w.to_numpy(dtype=float)
    ts = score_w.index.to_numpy()
    n_bars = len(score_w)

    entry = np.full_like(c_np, np.nan)
    exit_ = np.full_like(c_np, np.nan)
    entry[:n_bars - 1 - h] = c_np[1:n_bars - h]
    exit_[:n_bars - 1 - h] = c_np[1 + h:n_bars]
    fwd_event = exit_ / entry - 1.0                    # signal bar t -> enter t+1, exit t+1+h
    fwd_ic = np.full_like(c_np, np.nan)
    fwd_ic[:n_bars - h] = c_np[h:] / c_np[:n_bars - h] - 1.0   # t -> t+h, for IC/LS

    # ---- (1) within-bucket rank IC, non-overlapping ----
    ic_vals, ic_ts = [], []
    for i in range(0, n_bars, h):
        s, f = s_np[i], fwd_ic[i]
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() < min_sym or np.nanstd(s[ok]) == 0:
            continue
        ic_vals.append(spearmanr(s[ok], f[ok]).statistic)
        ic_ts.append(ts[i])
    ic_vals, ic_ts = np.array(ic_vals, dtype=float), np.array(ic_ts)
    ic_ok = np.isfinite(ic_vals)
    ic_vals, ic_ts = ic_vals[ic_ok], ic_ts[ic_ok]

    # ---- (2) within-bucket quintile long-short + turnover + breakeven ----
    prev_w, gross, turns = None, [], []
    for i in range(0, n_bars, h):
        s, f = s_np[i], fwd_ic[i]
        ok = np.isfinite(s) & np.isfinite(f)
        if ok.sum() < min_sym:
            continue
        k = max(int(ok.sum() // 5), 1)
        longs = np.argsort(np.where(ok, s, -np.inf))[-k:]
        shorts = np.argsort(np.where(ok, s, np.inf))[:k]
        w = np.zeros_like(s)
        w[longs], w[shorts] = 0.5 / k, -0.5 / k
        gross.append(float(np.nansum(w * np.where(np.isfinite(f), f, 0.0))))
        turns.append(float(np.abs(w - prev_w).sum()) if prev_w is not None else float(np.abs(w).sum()))
        prev_w = w
    gross, turns = np.array(gross), np.array(turns)
    ls_breakeven_bps = float(gross.mean() / (turns.mean() / 2.0) * 10000.0) if len(gross) and turns.mean() > 0 else float("nan")

    # ---- (3) event study within bucket, excess vs the bucket's own EW return ----
    bucket_mean = np.nanmean(fwd_event, axis=1)
    valid = np.isfinite(fwd_event) & np.isfinite(bucket_mean)[:, None]
    rl, cl = np.where((v_np >= K_EVENT) & valid)
    rs, cs = np.where((v_np <= -K_EVENT) & valid)
    exc = np.concatenate([fwd_event[rl, cl] - bucket_mean[rl], -(fwd_event[rs, cs] - bucket_mean[rs])])
    ev_ts = np.concatenate([ts[rl], ts[rs]])

    per_period = {}
    for pname, (a, b) in PERIODS.items():
        m_ic = (ic_ts >= np.datetime64(pd.Timestamp(a))) & (ic_ts <= np.datetime64(pd.Timestamp(b))) if len(ic_ts) else np.array([], dtype=bool)
        m_ev = (ev_ts >= np.datetime64(pd.Timestamp(a))) & (ev_ts <= np.datetime64(pd.Timestamp(b))) if len(ev_ts) else np.array([], dtype=bool)
        per_period[pname] = {"ic": _summ(ic_vals[m_ic]) if len(ic_ts) else {"n": 0},
                             "event_excess_bps": float(np.mean(exc[m_ev]) * 10000) if m_ev.sum() else float("nan"),
                             "event_n": int(m_ev.sum())}
    ev_sign_ok = int(sum(1 for v in per_period.values() if np.isfinite(v["event_excess_bps"]) and v["event_excess_bps"] > 0))
    ic_sign_ok = int(sum(1 for v in per_period.values() if v["ic"].get("n", 0) > 0 and v["ic"].get("mean", 0) > 0))

    event_breakeven_bps = float(np.mean(exc) * 10000.0) if len(exc) else float("nan")
    return {
        "ic": _summ(ic_vals),
        "long_short": {"rebalances": int(len(gross)), "gross_mean_per_period_bps": float(gross.mean() * 10000) if len(gross) else float("nan"),
                       "mean_turnover": float(turns.mean()) if len(turns) else float("nan"),
                       "breakeven_roundtrip_bps": ls_breakeven_bps},
        "event": {"events": int(len(exc)), "excess_mean_bps": event_breakeven_bps,
                  "t_stat": float(np.mean(exc) / (np.std(exc, ddof=1) / np.sqrt(len(exc)))) if len(exc) > 1 and np.std(exc, ddof=1) > 0 else float("nan"),
                  "breakeven_roundtrip_bps": event_breakeven_bps},
        "per_period": per_period,
        "period_sign_consistency_event": ev_sign_ok, "period_sign_consistency_ic": ic_sign_ok,
        "clears_maker_roundtrip": bool(np.isfinite(event_breakeven_bps) and event_breakeven_bps > MAKER_ROUNDTRIP_BPS)
                                  or bool(np.isfinite(ls_breakeven_bps) and ls_breakeven_bps > MAKER_ROUNDTRIP_BPS),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("=== stage=build ===")
    votes_w, cont_w, close_w, kept = build_matrices()
    log(f"  {close_w.shape[0]} bars x {close_w.shape[1]} symbols [{close_w.index.min()} .. {close_w.index.max()}]")

    report: dict[str, Any] = {
        "design": "Liquidity-bucket decomposition of IC and cost. Tests whether the de Groot large-cap/low-turnover "
                  "prescription transfers to crypto perps, against the opposite published claim that crypto reversals "
                  "live in illiquid names. Primary output: breakeven round-trip cost in bp per bucket.",
        "pre_registered": {"buckets": {k: list(v) for k, v in BUCKETS.items()}, "horizons": list(HORIZONS),
                           "K_event": K_EVENT, "min_symbols": MIN_SYMBOLS,
                           "reference_costs_bps": {"taker_roundtrip": TAKER_ROUNDTRIP_BPS, "maker_roundtrip": MAKER_ROUNDTRIP_BPS},
                           "kill_criterion": "path closed unless some bucket x horizon has breakeven round-trip > 4bp "
                                             "AND >=3/4 period sign consistency"},
        "known_bias": "Bucket membership uses the manifest's single-instant (2026-08-04) volume ordering, so it is "
                      "forward-looking w.r.t. 2024-2025; the panel carries no absolute volume (rvol_48 is per-symbol "
                      "normalised) so no causal alternative exists without re-downloading raw klines. Per-period "
                      "results are reported so staleness can be judged.",
        "bucket_members": {k: kept[a:b] for k, (a, b) in BUCKETS.items()},
        "trade_ledgers_used_as_input": False, "saved_parent_exit_timestamps_used": False, "future_rows_used_for_entry": False,
        "results": {},
    }

    for bname, (a, b) in BUCKETS.items():
        cols = kept[a:b]
        log(f"=== bucket={bname} ({len(cols)} symbols: {cols[0]}..{cols[-1]}) ===")
        for h in HORIZONS:
            for sname, sw in (("continuous", cont_w[cols]), ("votes", votes_w[cols])):
                res = bucket_analysis(sw, close_w[cols], votes_w[cols], h, MIN_SYMBOLS[bname])
                report["results"][f"{bname}_{sname}_h{h}"] = res
                if sname == "continuous":
                    log(f"  h={h:3d} {sname:10s} IC={res['ic']['mean']:+.5f} (t={res['ic']['t_stat']:+.2f}, n={res['ic']['n']}) "
                        f"| LS gross={res['long_short']['gross_mean_per_period_bps']:+.3f}bp turn={res['long_short']['mean_turnover']:.2f} "
                        f"breakeven={res['long_short']['breakeven_roundtrip_bps']:+.3f}bp "
                        f"| event n={res['event']['events']:6d} excess={res['event']['excess_mean_bps']:+.3f}bp "
                        f"(t={res['event']['t_stat']:+.2f}) periods={res['period_sign_consistency_event']}/4 "
                        f"| clears_4bp={res['clears_maker_roundtrip']}")
                else:
                    log(f"  h={h:3d} {sname:10s} IC={res['ic']['mean']:+.5f} (t={res['ic']['t_stat']:+.2f}) "
                        f"| LS breakeven={res['long_short']['breakeven_roundtrip_bps']:+.3f}bp "
                        f"| event excess={res['event']['excess_mean_bps']:+.3f}bp periods={res['period_sign_consistency_event']}/4")

    survivors = [k for k, v in report["results"].items()
                 if v.get("clears_maker_roundtrip") and v.get("period_sign_consistency_event", 0) >= 3]
    report["survivors_clearing_kill_criterion"] = survivors
    report["verdict"] = "SURVIVES" if survivors else "CLOSED_NO_BUCKET_CLEARS_REALISTIC_COST"
    (OUT_DIR / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    log(f"report={OUT_DIR / 'report.json'}")
    log(f"stage=done VERDICT={report['verdict']} survivors={survivors}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
