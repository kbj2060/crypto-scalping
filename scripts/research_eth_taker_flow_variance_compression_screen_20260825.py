#!/usr/bin/env python3
"""ETH -- taker order-flow VARIANCE-compression precursor screen, 2026-08-25 follow-up to the
liquidation-map literature review (docs/experiments/
eth_liquidation_map_v2_oi_cohort_direction_design_20260825.md and memory
eth_liquidation_map_literature_review_cascade_mechanism_20260825).

Two very recent papers on crypto-perpetual liquidation cascades (arXiv:2607.27070 "Where does the
criticality live?" and arXiv:2608.03616 "Measuring the engine of a liquidation cascade") found that
no price/leverage/order-flow variable gives an event-invariant early-warning signal across 7 major
BTC cascades, EXCEPT one: a compression of taker order-flow VARIANCE survives a placebo test across
all seven (Fisher-combined p~5e-6) -- explicitly flagged by the authors as a population-level
precursor, not a per-event alarm. This script is NOT a replication (the papers' exact window/
statistic for "compression" is not given in their abstracts, which is all this session read; no
methods-section detail was available) -- it is an independent operationalization of the same
DIRECTIONAL hypothesis on this repo's own ETH data: does LOW rolling dispersion of taker order flow
precede an EXPANSION in forward realized range?

This is the OPPOSITE causal shape from research_eth_model_indicator_volatility_screen_20260825.py's
already-tested hypothesis on the same 3 features (taker_buy_ratio/nif_whale/nif_retail): that script
tested whether an EXTREME LEVEL of order flow (a directional imbalance) precedes forward-range
COMPRESSION ("climax -> rest"; nif_retail survived TRAIN, VAL underpowered -- see that memory).
This script tests whether LOW DISPERSION of the flow itself (a quiet, steady state) precedes
forward-range EXPANSION ("coiled spring -> release"). Same 3 features, same data source, same
target definition, deliberately different predictor construction -- not a re-run of the same test.

Methodology (reused wherever possible, not re-derived):
  - Predictor: rolling std of the RAW feature (not z-scored -- dispersion is already scale-free
    under Spearman, and z-scoring here would just rescale by a second, unrelated rolling std).
    Two windows reported side by side, not swept for a winner: W=288 (24h, PRIMARY -- identical
    window to the level-screen's Z_WIN, chosen for direct comparability, not tuned for this test)
    and W=48 (4h, secondary sensitivity check only).
  - Target: identical fwd_range_pct(h=12/48) from research_eth_model_indicator_volatility_screen_
    20260825.py, imported not copied.
  - Primary significance: raw Spearman(rolling_std, fwd_range) + shift_z permutation (imported).
    Expected sign under the hypothesis is NEGATIVE (low std -> high forward range).
  - Robustness (the central check, given how likely a raw hit is to be pure vol-clustering echo --
    order-flow dispersion and price dispersion plausibly share the same regime): partial Spearman
    controlling for BACKWARD realized range over a window matched to the target horizon (12 bars for
    h12, 48 bars for h48 -- the original script's own partial-corr check only ever controlled h12
    with a bwd-12 confound and never ran this control for h48 at all; this script extends it there).
    Same partial-correlation formula and shift-z-on-residuals confirmation as the original.
  - TRAIN (2026-05-03~07-31) screen first; VAL (2026-08-01~08-16) confirmation ONLY for whatever
    passes TRAIN's raw+partial screen, same two-stage discipline as the level-screen script.
  - Not a pre-registered decision gate -- same-session exploratory discovery. Any TRAIN+VAL survivor
    is a candidate needing a proper walk-forward re-check, not a validated signal.
"""
import time

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr

import scripts.research_eth_model_indicator_volatility_screen_20260825 as lvl

# 2026-08-25 data-integrity note: lvl.load_klines() reads binance_data/klines/ETHUSDT/
# ETHUSDT-5m-api.csv, which -- verified on the server just before writing this script -- stops at
# 2026-08-04 03:25, NOT the 2026-08-17 its own filter implies. That file is untracked (not in git,
# no crontab job found writing it), so it looks like a one-off snapshot rather than a maintained
# archive; root cause of the staleness (and whether it was already stale when the level-screen's
# recorded VAL run reported n=3538 for 2026-08-01..08-16, which arithmetically looks too large for
# a ~3-day window) was NOT resolved -- flagged, not silently trusted either way (see
# eth_liquidation_map_literature_review_cascade_mechanism_20260825 follow-up memory). This script
# sidesteps the question entirely by fetching 5m klines fresh from Binance for exactly the needed
# range instead of reading that file, so its own TRAIN/VAL windows are verified-complete regardless.


def fetch_5m_klines(start: str, end: str) -> pd.DataFrame:
    """Paginated /fapi/v1/klines, 5m interval, [start, end] inclusive of end's calendar day."""
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    end_ms = int((pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    end_ms = min(end_ms, now_ms)
    rows, cursor = [], start_ms
    while cursor < end_ms:
        resp = requests.get(
            "https://fapi.binance.com/fapi/v1/klines",
            params={"symbol": "ETHUSDT", "interval": "5m", "startTime": cursor,
                    "endTime": end_ms, "limit": 1500},
            timeout=15,
        )
        resp.raise_for_status()
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_open = int(batch[-1][0])
        if last_open < cursor:
            break
        cursor = last_open + 300_000
        if len(batch) < 1500:
            break
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = df[c].astype("float64")
    df["timestamp"] = pd.to_datetime(df["open_time"].astype("int64"), unit="ms", utc=True).dt.tz_localize(None)
    df["close_time"] = df["close_time"].astype("int64")
    df = df[df["close_time"] < now_ms]  # drop the still-forming bar
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["bar_close_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
    # lvl.load_micro()'s ts comes back at us-resolution (pandas' current duckdb round-trip);
    # merge_asof requires identical key dtype, not just identical semantics.
    df["timestamp"] = df["timestamp"].astype("datetime64[us]")
    df["bar_close_time"] = df["bar_close_time"].astype("datetime64[us]")
    return df[["timestamp", "bar_close_time", "open", "high", "low", "close", "volume"]]

FEATURES = ["taker_buy_ratio", "nif_whale", "nif_retail"]  # the 3 features the level-screen
                                                            # implicated (partial-corr survivors or
                                                            # near-survivors), not the full 9+1 --
                                                            # this is a follow-up on those specific
                                                            # candidates' flow data, not a fresh sweep
STD_WINDOWS = {"w288_24h": 288, "w48_4h": 48}
STD_PRIMARY = "w288_24h"
STD_MINP_FRAC = lvl.Z_MINP / lvl.Z_WIN  # same 90% completeness convention, any window size
LOW_PCTL = 0.20  # bottom-quintile "compressed" reading for the human-readable lift readout


def rolling_std(klines: pd.DataFrame, feat: str, window: int) -> pd.Series:
    minp = max(2, int(round(window * STD_MINP_FRAC)))
    return klines[feat].rolling(window, min_periods=minp).std()


def partial_ic(predictor: pd.Series, fwd: pd.Series, bwd: pd.Series) -> dict:
    """Same partial-correlation formula as research_eth_model_indicator_volatility_screen_
    20260825.py::partial_corr_check, generalized over an arbitrary (predictor, fwd, bwd) triple
    instead of being hardcoded to h12/bwd_h12_1h -- this is the extension to h48 the original
    script never ran. Returns raw IC, r_bf baseline, partial IC, and its own shift-z on residuals."""
    m = predictor.notna() & fwd.notna() & bwd.notna()
    if m.sum() < 40:
        return {"n": int(m.sum()), "raw_ic": float("nan"), "r_bf": float("nan"),
                "partial_ic": float("nan"), "shift_z": float("nan")}
    r_bf = spearmanr(bwd[m], fwd[m]).statistic
    r_zf, _, n = lvl.shift_z(predictor[m], fwd[m])
    r_zb, _, _ = lvl.shift_z(predictor[m], bwd[m])
    rank_b, rank_f = bwd[m].rank(), fwd[m].rank()
    slope = np.polyfit(rank_b, rank_f, 1)
    resid = rank_f - np.polyval(slope, rank_b)
    denom = np.sqrt((1 - r_zb ** 2) * (1 - r_bf ** 2))
    partial = (r_zf - r_zb * r_bf) / denom if denom > 0 else float("nan")
    _, zperm, npar = lvl.shift_z(predictor[m], pd.Series(resid, index=predictor[m].index))
    return {"n": int(npar), "raw_ic": r_zf, "r_bf": r_bf, "partial_ic": partial, "shift_z": zperm}


def verdict(res: dict) -> str:
    if np.isnan(res["shift_z"]):
        return "insufficient_n"
    return "SURVIVES" if (abs(res["shift_z"]) >= 2 and abs(res["partial_ic"]) >= 0.025) else "weak/fails control"


def low_dispersion_lift(predictor: pd.Series, fwd: pd.Series) -> tuple[float, int]:
    m = predictor.notna() & fwd.notna()
    if m.sum() < 40:
        return float("nan"), int(m.sum())
    thresh = predictor[m].quantile(LOW_PCTL)
    cond = m & (predictor <= thresh)
    if cond.sum() < 20:
        return float("nan"), int(cond.sum())
    return float(fwd[cond].mean() / fwd[m].mean()), int(cond.sum())


def decile_table(predictor: pd.Series, fwd: pd.Series) -> None:
    m = predictor.notna() & fwd.notna()
    if m.sum() < 100:
        print("    (decile table skipped, n<100)")
        return
    bins = pd.qcut(predictor[m], 10, labels=False, duplicates="drop")
    g = fwd[m].groupby(bins).agg(["mean", "count"])
    print("    decile(std) -> mean fwd_range%%, n  (decile 0 = lowest dispersion):")
    for i, row in g.iterrows():
        print(f"      d{int(i)}: {row['mean']*100:.4f}%  (n={int(row['count'])})")


def diagnose_confounds(klines: pd.DataFrame, mask: pd.Series, split_name: str, features: list[str]) -> None:
    """2026-08-25: the raw effect sizes this screen found (|IC| up to 0.44) are far larger than
    anything else validated in this repo's history (typically 0.02-0.15) -- big enough to distrust
    by default until checked. Three specific confounds this repo has been burned by before:
      1. Is rolling-std(feature) nearly collinear with backward realized range itself (i.e. just
         re-deriving "we are in a low-vol regime" through a noisy proxy, leaking past the
         partial-corr control because that control is imperfect, not because there's new info)?
      2. Is it nearly collinear with rolling VOLUME (a mundane "thin trading -> quiet now, but
         mean-reverts" story, not an order-flow-specific one)?
      3. Is "low std" secretly just "sustained extreme level" -- i.e. the same predictor the
         LEVEL screen (research_eth_model_indicator_volatility_screen_20260825.py) already tested,
         relabeled, rather than the deliberately different quantity this script's docstring claims?
    """
    close = klines["close"]
    bwd12 = lvl.bwd_range_pct(klines["high"], klines["low"], close, 12)
    vol288 = klines["volume"].rolling(288, min_periods=259).mean()
    print(f"\n--- [{split_name}] confound diagnostics ---")
    for feat in features:
        std288 = rolling_std(klines, feat, 288).where(mask)
        absz288 = lvl.rolling_absz(klines, feat, None).where(mask)
        m = std288.notna() & bwd12.notna()
        r_std_bwd = spearmanr(std288[m], bwd12[m]).statistic if m.sum() >= 40 else float("nan")
        m2 = std288.notna() & vol288.notna()
        r_std_vol = spearmanr(std288[m2], vol288[m2]).statistic if m2.sum() >= 40 else float("nan")
        m3 = std288.notna() & absz288.notna()
        r_std_level = spearmanr(std288[m3], absz288[m3]).statistic if m3.sum() >= 40 else float("nan")
        print(f"  {feat:16s} Spearman(std288,bwd_range12)={r_std_bwd:+.4f}  "
              f"Spearman(std288,volume288)={r_std_vol:+.4f}  "
              f"Spearman(std288,|level_z|)={r_std_level:+.4f}")
        fwd12 = klines["fwd_h12_1h"]
        r2 = partial_ic_2conf(std288, fwd12, bwd12, vol288)
        v2 = "SURVIVES" if (not np.isnan(r2["shift_z"]) and abs(r2["shift_z"]) >= 2) else "fails volume control"
        print(f"    -> h12_1h after controlling for [bwd_range12 AND volume288] jointly: "
              f"resid_ic={r2['ic_after_volume_control']:+.4f} shift_z={r2['shift_z']:+.2f} "
              f"(n={r2['n']})  [{v2}]")


def partial_ic_2conf(predictor: pd.Series, fwd: pd.Series, conf1: pd.Series, conf2: pd.Series) -> dict:
    """2026-08-25 follow-up to diagnose_confounds(): std288(taker flow) turned out to be 0.59-0.83
    Spearman-collinear with rolling mean volume (see that function's docstring) -- strong enough
    that the single-confound partial_ic() above may just be leaking a volume-compression effect
    through. Same semi-partial pattern as the original script's partial_corr_check (only the
    TARGET is residualized, via OLS on RANKS of both confounds jointly; the predictor stays raw),
    extended from 1 confound to 2."""
    m = predictor.notna() & fwd.notna() & conf1.notna() & conf2.notna()
    if m.sum() < 40:
        return {"n": int(m.sum()), "shift_z": float("nan"), "ic_after_volume_control": float("nan")}
    X = np.column_stack([conf1[m].rank().to_numpy(), conf2[m].rank().to_numpy(), np.ones(int(m.sum()))])
    y = fwd[m].rank().to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = pd.Series(y - X @ beta, index=predictor[m].index)
    obs, zperm, n = lvl.shift_z(predictor[m], resid)
    return {"n": n, "shift_z": zperm, "ic_after_volume_control": obs}


def screen(klines: pd.DataFrame, mask: pd.Series, split_name: str, features: list[str],
          decile_for: set[str]) -> dict:
    close = klines["close"]
    bwd_by_h = {h: lvl.bwd_range_pct(klines["high"], klines["low"], close, h) for h in lvl.HORIZONS.values()}
    out = {}
    for feat in features:
        print(f"\n--- [{split_name}] {feat} ---")
        out[feat] = {}
        for wname, w in STD_WINDOWS.items():
            std = rolling_std(klines, feat, w).where(mask)
            for hname, h in lvl.HORIZONS.items():
                fwd = klines[f"fwd_{hname}"]
                bwd = bwd_by_h[h]
                res = partial_ic(std, fwd, bwd)
                v = verdict(res)
                lift, n_low = low_dispersion_lift(std.where(mask), fwd)
                print(f"  [{wname:9s} -> {hname}] raw_ic={res['raw_ic']:+.4f}  "
                      f"partial_ic={res['partial_ic']:+.4f}  shift_z={res['shift_z']:+.2f}  "
                      f"(n={res['n']})  [{v}]  |  bottom-{int(LOW_PCTL*100)}% lift={lift:.3f}x(n={n_low})")
                out[feat][(wname, hname)] = {**res, "verdict": v, "low_lift": lift}
                if v == "SURVIVES" and wname == STD_PRIMARY and feat in decile_for:
                    decile_table(std.where(mask), fwd)
    return out


def main() -> None:
    klines = fetch_5m_klines("2026-04-28", "2026-08-17")
    print(f"fetched fresh: {len(klines)} rows, {klines['timestamp'].min()} .. {klines['timestamp'].max()}")
    klines = lvl.attach(klines, lvl.load_micro(), lvl.CORE_FEATURES + lvl.NEW_FEATURES)
    close = klines["close"]
    for hname, h in lvl.HORIZONS.items():
        klines[f"fwd_{hname}"] = lvl.fwd_range_pct(klines["high"], klines["low"], close, h)

    train_mask = (klines["timestamp"] >= lvl.TRAIN_START) & (klines["timestamp"] <= lvl.TRAIN_END)
    val_mask = (klines["timestamp"] >= lvl.VAL_START) & (klines["timestamp"] <= lvl.VAL_END)
    print(f"klines loaded: {len(klines)} rows, {klines['timestamp'].min()} .. {klines['timestamp'].max()}")
    print(f"TRAIN rows: {int(train_mask.sum())}  VAL rows: {int(val_mask.sum())}")

    print(f"\n{'='*110}\nTAKER-FLOW VARIANCE-COMPRESSION SCREEN -- TRAIN {lvl.TRAIN_START}~{lvl.TRAIN_END}\n"
          f"hypothesis: LOW rolling std(feature) -> HIGH forward realized range (negative IC expected)\n{'='*110}")
    diagnose_confounds(klines, train_mask, "TRAIN", FEATURES)
    train_res = screen(klines, train_mask, "TRAIN", FEATURES, decile_for=set(FEATURES))

    survivors = sorted({feat for feat, cells in train_res.items()
                        if any(c["verdict"] == "SURVIVES" for c in cells.values())})
    print(f"\nTRAIN survivors (>=1 window/horizon cell passes): {survivors or 'NONE'}")

    if survivors:
        print(f"\n{'='*110}\nVAL {lvl.VAL_START}~{lvl.VAL_END} CONFIRMATION -- TRAIN survivors only\n{'='*110}")
        screen(klines, val_mask, "VAL", survivors, decile_for=set(survivors))
    else:
        print("\nNo TRAIN survivors -- VAL left untouched, nothing to confirm.")

    print(f"\n{'='*110}\nNothing above is a promotion/deployment decision.\n{'='*110}")


if __name__ == "__main__":
    main()
