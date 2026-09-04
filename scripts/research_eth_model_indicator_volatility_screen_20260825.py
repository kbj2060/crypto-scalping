#!/usr/bin/env python3
"""ETH -- model-indicator (microstructure_1m/tail_risk_1m) family, VOLATILITY-framing screen.

Context: this exact 9-feature family (obi, taker_buy_ratio, spoofing_score, nif_whale,
nif_retail, shadow_toxicity_score, shadow_queue_collapse, eai, oi_delta_pct) has already
been exhaustively tested for DIRECTION at every horizon from 1 minute to 4 hours:
research_eth_microstructure_panel_1h4h_direction_screen_20260823.py (h=12/48, 3/18 stat ->
1/18 after a stale-dev-copy fix, 0/3 economic) and research_eth_microstructure_scalp_
horizon_screen_20260824.py (h=1/3/5/10/15min, 19/45 stat, 45/45 economic FAIL after 4
rounds of correction including a min_periods completeness-trap bug). Direction is closed
for this data family across its full tested horizon range.

The repo's own established meta-finding is that this exact data family (OI/liquidation/
positioning) shows real, monotonic, direction-SYMMETRIC lift when reframed as a forward-
VOLATILITY precursor instead of a direction predictor -- already confirmed for oi_delta_pct
(1.21-1.45x lift @ |z|>=1/2/3, now live as the "OI 급변" dashboard chip) and for tail_risk
generally. That vol-framing has NEVER been run for the other 8 microstructure_1m raw
features, nor for whale_position_score or tail_risk_1m's shadow_aftershock_prob (both of
which have only ever been tested/displayed, never validated as predictors of anything).
This script runs the same vol-framing test across that full untested set, using oi_delta_pct
itself as an in-script replication check (if it doesn't roughly reproduce its own known
result, something in this script's methodology is wrong and the rest should not be trusted).

Methodology (deliberately reused, not re-derived):
  - Rolling 288-bar (24h) z-score per feature, min_periods=259 (90% completeness) -- NOT
    full-window completeness, which research_eth_microstructure_scalp_horizon_screen_20260824.py's
    own post-mortem found silently subselects results to a handful of abnormally-clean days.
  - Target: forward realized range (max(high)-min(low) over the NEXT h bars, excluding the
    firing bar itself) / close, at h=12 (1h) and h=48 (4h) -- identical target definition to
    the oi_delta_pct-as-vol discovery.
  - Primary significance: Spearman(|z|, fwd_range) + circular-shift permutation (shift_z,
    verbatim from research_eth_weekly_oi_growth_hong_yogo_cheap_gate_20260824.py).
  - Secondary/human-readable: lift = mean(fwd_range | |z|>=2) / mean(fwd_range | all bars),
    plus a direction-symmetry breakdown (z>=+2 vs z<=-2) for any feature clearing the primary
    screen -- replicates exactly how oi_delta_pct's own direction-symmetry was reported.
  - Contamination check: Spearman(feature_raw, close) per feature, flag |r|>=0.5 as a
    price-trend-contaminated candidate (standing repo convention, see
    feedback_raw_feature_price_trend_contamination).

Scope: TRAIN window only (2026-05-03~07-31, matching the two direction screens' own TRAIN
split) -- VAL/OOS for both tables deliberately untouched. whale_position_score is only
populated from 2026-07-18 23:58 UTC on (pre-that is 100% NULL, known data-start artifact,
not a new finding) so its own valid sub-window is shorter and reported separately.
shadow_aftershock_prob (tail_risk_1m) is restricted to the liquidation-stream valid-since
cutoff (2026-07-18 15:03 UTC, forceOrder WS fix) for the same reason -- pre-that its values
are >=72% hard zero, a documented feed defect, not signal.

This is same-session exploratory discovery, not a pre-registered decision gate -- any
survivor should carry the same "candidate, needs a proper walk-forward re-check" caveat the
original oi_delta_pct finding shipped with, not be treated as validated.
"""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

KLINES_PATH = "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
MICRO_DB_PATH = "data/live/microstructure.duckdb"
TAIL_DB_PATH = "data/live/tail_risk.duckdb"

N_PERM = 2000
MIN_SHIFT = 8
Z_WIN, Z_MINP = 288, 259          # 24h of 5m bars, 90% completeness (not 100% -- see docstring)
HORIZONS = {"h12_1h": 12, "h48_4h": 48}
Z_THRESH_REPORT = [1.0, 2.0, 3.0]  # for the oi_delta_pct/eai replication rows only
Z_THRESH_PRIMARY = 2.0             # single pre-registered threshold for all other rows

TRAIN_START, TRAIN_END = "2026-05-03", "2026-07-31"
VAL_START, VAL_END = "2026-08-01", "2026-08-16"  # same split as research_eth_microstructure_panel_1h4h_direction_screen_20260823.py
WHALE_POS_VALID_SINCE = pd.Timestamp("2026-07-18 23:58:00")  # UTC-naive, matches klines convention
LIQ_VALID_SINCE = pd.Timestamp("2026-07-18 15:03:00")

# TRAIN survivors of the raw screen + partial-correlation (autocorrelation-controlled) robustness
# check -- 2026-08-25 same-session follow-up, confirming on the untouched VAL window before any
# deployment consideration. oi_delta_pct rides along as the same sanity anchor it was in the
# TRAIN pass (if it doesn't replicate its own known VAL-era behavior, distrust the rest).
VAL_CONFIRM_FEATURES = ["nif_retail", "taker_buy_ratio", "nif_whale", "oi_delta_pct"]

# 7 raw + 2 z already tested for DIRECTION (now testing VOLATILITY framing instead)
CORE_FEATURES = ["obi", "taker_buy_ratio", "spoofing_score", "nif_whale", "nif_retail",
                  "shadow_toxicity_score", "shadow_queue_collapse", "eai", "oi_delta_pct"]
# genuinely untested in ANY framing before this script
NEW_FEATURES = ["whale_position_score"]
REPLICATION_FEATURES = {"oi_delta_pct", "eai"}  # sanity-check rows, expect known-shape result


def shift_z(x: pd.Series, y: pd.Series, seed: int = 20260825) -> tuple[float, float, int]:
    """Verbatim from research_eth_weekly_oi_growth_hong_yogo_cheap_gate_20260824.py::shift_z."""
    d = pd.concat([x, y], axis=1).dropna().to_numpy()
    n = len(d)
    if n < 40:
        return float("nan"), float("nan"), n
    obs = spearmanr(d[:, 0], d[:, 1]).statistic
    rng = np.random.default_rng(seed)
    shifts = rng.integers(MIN_SHIFT, n - MIN_SHIFT, size=N_PERM)
    null = np.array([spearmanr(np.roll(d[:, 0], s), d[:, 1]).statistic for s in shifts])
    return obs, (obs - null.mean()) / null.std(ddof=1), n


def fwd_range_pct(high: pd.Series, low: pd.Series, close: pd.Series, h: int) -> pd.Series:
    """(max(high)-min(low)) over the NEXT h bars strictly after the current one, / close.
    Verified by hand for h=1 against a toy series before use (see module docstring)."""
    fh = high[::-1].rolling(window=h, min_periods=h).max()[::-1].shift(-1)
    fl = low[::-1].rolling(window=h, min_periods=h).min()[::-1].shift(-1)
    return (fh - fl) / close


def bwd_range_pct(high: pd.Series, low: pd.Series, close: pd.Series, h: int) -> pd.Series:
    """(max(high)-min(low)) over the h bars ending at and including the current one, / close --
    used only as the autocorrelation-control confound in the partial-correlation robustness check,
    never as a target."""
    return (high.rolling(h, min_periods=h).max() - low.rolling(h, min_periods=h).min()) / close


def rolling_absz(klines: pd.DataFrame, feat: str, valid_since: pd.Timestamp | None) -> pd.Series:
    s = klines[feat].copy()
    if valid_since is not None:
        s = s.where(klines["bar_close_time"] >= valid_since)
    mu = s.rolling(Z_WIN, min_periods=Z_MINP).mean()
    sd = s.rolling(Z_WIN, min_periods=Z_MINP).std()
    return ((s - mu) / sd.replace(0.0, np.nan)).abs()


def partial_corr_check(klines: pd.DataFrame, mask: pd.Series, split_name: str, features: list[str]) -> None:
    """Re-run of the 2026-08-25 TRAIN-pass robustness check (autocorrelation-controlled partial
    Spearman IC), now on `mask`'s window, for `features`. Isolates each feature's incremental
    forward-vol information beyond what the CONCURRENT (backward-looking) realized range -- i.e.
    "we are already in a high/low-vol regime" -- already explains. A feature that only passes the
    raw screen but not this control is riding volatility clustering, not offering new information."""
    close = klines["close"]
    bwd = bwd_range_pct(klines["high"], klines["low"], close, 12)
    fwd = klines["fwd_h12_1h"]
    m0 = mask & bwd.notna() & fwd.notna()
    r_bf = spearmanr(klines.loc[m0, "bwd_h12_1h"], fwd[m0]).statistic if m0.sum() >= 40 else float("nan")
    print(f"\n[{split_name} robustness check] Spearman(backward_range, forward_range)={r_bf:+.4f} "
          f"(n={int(m0.sum())}) -- vol-clustering baseline; partial IC below nets this out")
    for feat in features:
        absz = rolling_absz(klines, feat, WHALE_POS_VALID_SINCE if feat == "whale_position_score" else None)
        m = m0 & absz.notna()
        if m.sum() < 40:
            print(f"  {feat:16s} n={int(m.sum())} < 40, skipped")
            continue
        r_zf, _, n = shift_z(absz[m], fwd[m])
        r_zb, _, _ = shift_z(absz[m], klines.loc[m, "bwd_h12_1h"])
        rank_b = klines.loc[m, "bwd_h12_1h"].rank()
        rank_f = fwd[m].rank()
        slope = np.polyfit(rank_b, rank_f, 1)
        resid = rank_f - np.polyval(slope, rank_b)
        partial = (r_zf - r_zb * r_bf) / np.sqrt((1 - r_zb ** 2) * (1 - r_bf ** 2))
        _, zperm_partial, npar = shift_z(absz[m], pd.Series(resid, index=absz[m].index))
        verdict = "SURVIVES" if (not np.isnan(zperm_partial) and abs(zperm_partial) >= 2 and abs(partial) >= 0.025) \
            else "weak/fails control"
        print(f"  {feat:16s} raw IC={r_zf:+.4f}  |  partial IC(control bwd_range)={partial:+.4f} "
              f"shift-z={zperm_partial:+.2f} (n={npar})  [{verdict}]")


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df = df[(df["timestamp"] >= "2026-04-28") & (df["timestamp"] <= "2026-08-17")].reset_index(drop=True)
    df["bar_close_time"] = df["timestamp"] + pd.Timedelta(minutes=5)
    return df


def load_micro() -> pd.DataFrame:
    cols = CORE_FEATURES + NEW_FEATURES
    con = duckdb.connect(MICRO_DB_PATH, read_only=True)
    try:
        micro = con.execute(f"SELECT ts, {', '.join(cols)} FROM microstructure_1m ORDER BY ts").fetchdf()
    finally:
        con.close()
    micro["ts"] = pd.to_datetime(micro["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    return micro


def load_tail_risk() -> pd.DataFrame:
    con = duckdb.connect(TAIL_DB_PATH, read_only=True)
    try:
        tr = con.execute(
            "SELECT ts, shadow_aftershock_prob, valid_liq_stream, ws_stale FROM tail_risk_1m ORDER BY ts"
        ).fetchdf()
    finally:
        con.close()
    tr["ts"] = pd.to_datetime(tr["ts"]).dt.tz_convert("UTC").dt.tz_localize(None)
    tr = tr[(tr["valid_liq_stream"] == True) & (tr["ws_stale"] != True)]  # noqa: E712
    return tr[["ts", "shadow_aftershock_prob"]]


def attach(klines: pd.DataFrame, side: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    m = pd.merge_asof(
        klines.sort_values("bar_close_time"), side.sort_values("ts"),
        left_on="bar_close_time", right_on="ts", direction="backward",
        tolerance=pd.Timedelta("5min"), suffixes=("", "_dup"),
    )
    for c in cols:
        klines[c] = m[c].to_numpy()
    return klines


def screen_split(klines: pd.DataFrame, mask: pd.Series, split_name: str, features: list[str]) -> None:
    close = klines["close"]
    for feat in features:
        valid_since = None
        if feat == "whale_position_score":
            valid_since = WHALE_POS_VALID_SINCE
        elif feat == "shadow_aftershock_prob":
            valid_since = LIQ_VALID_SINCE
        s_raw = klines[feat].where(klines["bar_close_time"] >= valid_since) if valid_since is not None else klines[feat]
        mu = s_raw.rolling(Z_WIN, min_periods=Z_MINP).mean()
        sd = s_raw.rolling(Z_WIN, min_periods=Z_MINP).std()
        z = (s_raw - mu) / sd.replace(0.0, np.nan)
        absz = z.abs()

        contam = spearmanr(*pd.concat([s_raw, close], axis=1).dropna().to_numpy().T).statistic \
            if s_raw.notna().sum() >= 40 else float("nan")
        contam_flag = "EXCLUDED-CANDIDATE(|r|>=0.5)" if abs(contam) >= 0.5 else "ok"

        print(f"\n--- [{split_name}] {feat} {'[REPLICATION CHECK]' if feat in REPLICATION_FEATURES else ''}"
              f"{'[NEW]' if feat in NEW_FEATURES + ['shadow_aftershock_prob'] else ''} "
              f"(contamination spearman(feat,close)={contam:+.3f} {contam_flag}) ---")

        for hname in HORIZONS:
            tgt = klines[f"fwd_{hname}"]
            m = mask & absz.notna() & tgt.notna()
            ic, zperm, n = shift_z(absz[m], tgt[m])
            base_mean = tgt[m].mean()

            threshes = Z_THRESH_REPORT if feat in REPLICATION_FEATURES else [Z_THRESH_PRIMARY]
            lift_parts = []
            for th in threshes:
                cond = m & (absz >= th)
                if cond.sum() >= 20:
                    lift = tgt[cond].mean() / base_mean
                    lift_parts.append(f"|z|>={th:.0f}: {lift:.2f}x(n={int(cond.sum())})")
                else:
                    lift_parts.append(f"|z|>={th:.0f}: n/a(n={int(cond.sum())})")

            sig = "PASS" if (not np.isnan(zperm) and abs(zperm) >= 2 and abs(ic) >= 0.025) else "fail"
            print(f"  {hname}: IC(|z|,fwd_range)={ic:+.4f} shift-z={'n/a' if np.isnan(zperm) else f'{zperm:+.2f}'} "
                  f"(n={n})  [{sig}]  |  {' / '.join(lift_parts)}")

            if sig == "PASS":
                up = m & (z >= Z_THRESH_PRIMARY)
                dn = m & (z <= -Z_THRESH_PRIMARY)
                up_lift = tgt[up].mean() / base_mean if up.sum() >= 20 else float("nan")
                dn_lift = tgt[dn].mean() / base_mean if dn.sum() >= 20 else float("nan")
                print(f"           direction-symmetry: z>=+{Z_THRESH_PRIMARY:.0f} lift={up_lift:.2f}x(n={int(up.sum())}) "
                      f"vs z<=-{Z_THRESH_PRIMARY:.0f} lift={dn_lift:.2f}x(n={int(dn.sum())})")


def main() -> None:
    klines = load_klines()
    klines = attach(klines, load_micro(), CORE_FEATURES + NEW_FEATURES)
    klines = attach(klines, load_tail_risk(), ["shadow_aftershock_prob"])

    close = klines["close"]
    for hname, h in HORIZONS.items():
        klines[f"fwd_{hname}"] = fwd_range_pct(klines["high"], klines["low"], close, h)
    klines["bwd_h12_1h"] = bwd_range_pct(klines["high"], klines["low"], close, 12)

    train_mask = (klines["timestamp"] >= TRAIN_START) & (klines["timestamp"] <= TRAIN_END)
    val_mask = (klines["timestamp"] >= VAL_START) & (klines["timestamp"] <= VAL_END)

    all_features = CORE_FEATURES + NEW_FEATURES + ["shadow_aftershock_prob"]
    print(f"{'='*110}\nMODEL-INDICATOR VOLATILITY-FRAMING SCREEN -- TRAIN {TRAIN_START}~{TRAIN_END} (full battery, "
          f"exploratory)\n{'='*110}")
    screen_split(klines, train_mask, "TRAIN", all_features)
    partial_corr_check(klines, train_mask, "TRAIN", [f for f in VAL_CONFIRM_FEATURES if f != "oi_delta_pct"])

    print(f"\n{'='*110}\nVAL {VAL_START}~{VAL_END} CONFIRMATION -- TRAIN survivors only "
          f"({', '.join(VAL_CONFIRM_FEATURES)}), untouched until now\n{'='*110}")
    screen_split(klines, val_mask, "VAL", VAL_CONFIRM_FEATURES)
    partial_corr_check(klines, val_mask, "VAL", [f for f in VAL_CONFIRM_FEATURES if f != "oi_delta_pct"])

    print(f"\n{'='*110}\nNothing above is a promotion/deployment decision.\n{'='*110}")


if __name__ == "__main__":
    main()
