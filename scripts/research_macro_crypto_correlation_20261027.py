"""Macro <-> crypto co-movement battery on the panel from research_macro_crypto_panel_20261027.py.

Sections (all written to <out>/):
  A  correlation table  daily / weekly / monthly x periods, Pearson + Spearman + HAC t
  B  calendar-year weekly correlations (sign stability) + rolling 26w correlation chart
  C  lead-lag cross-correlation (weekly, k=-8..8; k>0 = macro leads crypto)
  D  Granger causality (weekly, 4 lags, both directions)
  E  multivariate weekly regression (standardised, HAC) + partial corr controlling SPX
  F  tail / conditional correlation (SPX bottom decile days, VIX spike days)
  G  liquidity-lead claims: NETLIQ / M2 / stablecoin supply vs forward BTC returns
  H  regime-conditional forward returns (expanding terciles, no look-ahead)
  I  out-of-sample predictability (expanding OLS, Campbell-Thompson R2_OS, Clark-West)
  J  robustness of the strongest regime effects (sub-periods, weekly rule, momentum control, 13w OOS)

Timing convention: Yahoo BTC/ETH daily close = 00:00 UTC of the next day, i.e. 3-4h after the
US cash close of the same calendar date. Daily "same date" correlations therefore let BTC see
3-4h more news than SPX/FRED; section C (lag 0 vs +/-1) shows how much this matters.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests

# name -> (column, transform, label)   logret = 100*dlog, diff = level change (bp for rates)
VARS = {
    "SPX": ("SPX", "logret", "S&P 500"),
    "KOSPI": ("KOSPI", "logret", "KOSPI"),
    "GOLD": ("GOLD", "logret", "Gold (GC=F)"),
    "BRENT": ("DCOILBRENTEU", "logret", "Brent oil"),
    "COPPER": ("PCOPPUSDM", "logret", "Copper (monthly)"),
    "DXY_BROAD": ("DTWEXBGS", "logret", "USD broad index"),
    "USDJPY": ("DEXJPUS", "logret", "USD/JPY"),
    "USDCNY": ("DEXCHUS", "logret", "USD/CNY"),
    "EURUSD": ("DEXUSEU", "logret", "EUR/USD"),
    "USDKRW": ("DEXKOUS", "logret", "USD/KRW"),
    "UST2Y": ("DGS2", "diff_bp", "UST 2Y yield"),
    "UST10Y": ("DGS10", "diff_bp", "UST 10Y yield"),
    "REAL10Y": ("DFII10", "diff_bp", "10Y TIPS real yield"),
    "BE10Y": ("T10YIE", "diff_bp", "10Y breakeven infl."),
    "CURVE_2s10s": ("T10Y2Y", "diff_bp", "2s10s curve"),
    "TERM_PREM": ("THREEFYTP10", "diff_bp", "10Y term premium (KW)"),
    "FEDFUNDS": ("DFF", "diff_bp", "Fed funds eff."),
    "VIX": ("VIXCLS", "logret", "VIX"),
    "BAA_SPREAD": ("BAA10Y", "diff_bp", "Baa-10Y credit spread"),
    "HY_OAS": ("BAMLH0A0HYM2", "diff_bp", "HY OAS (2023+)"),
    "NFCI": ("NFCI", "diff", "Chicago NFCI (weekly)"),
    "NFCI_RISK": ("NFCIRISK", "diff", "NFCI risk sub-index"),
    "NFCI_LEV": ("NFCILEVERAGE", "diff", "NFCI leverage sub-index"),
    "STLFSI": ("STLFSI4", "diff", "St.Louis stress (weekly)"),
    "FED_BS": ("WALCL", "logret", "Fed balance sheet (weekly)"),
    "NETLIQ": ("NETLIQ", "logret", "Net liquidity WALCL-TGA-RRP"),
    "RESERVES": ("WRESBAL", "logret", "Bank reserves (weekly)"),
    "TGA": ("WTREGEN", "logret", "Treasury General Account"),
    "RRP": ("RRPONTSYD", "diff", "ON RRP ($bn)"),
    "M2": ("M2SL", "logret", "US M2 (monthly)"),
    "STABLES": ("STABLE_SUPPLY", "logret", "Stablecoin supply (CM)"),
}
DAILY_OK = {"SPX", "KOSPI", "GOLD", "BRENT", "DXY_BROAD", "USDJPY", "USDCNY", "EURUSD", "USDKRW", "UST2Y", "UST10Y",
            "REAL10Y", "BE10Y", "CURVE_2s10s", "TERM_PREM", "FEDFUNDS", "VIX", "BAA_SPREAD", "HY_OAS", "STABLES"}
WEEKLY_NATIVE = {"NFCI", "NFCI_RISK", "NFCI_LEV", "STLFSI", "FED_BS", "NETLIQ", "RESERVES", "TGA", "RRP"}
MONTHLY_ONLY = {"COPPER", "M2"}

PERIODS = {
    "ALL_2015_2026": ("2015-01-01", "2026-12-31"),
    "P1_2015_2019": ("2015-01-01", "2019-12-31"),
    "P2_2020_2023": ("2020-01-01", "2024-01-10"),
    "P3_ETF_2024_2026": ("2024-01-11", "2026-12-31"),
}


def transform(level: pd.Series, how: str) -> pd.Series:
    if how == "logret":
        return 100 * np.log(level.where(level > 0)).diff()
    if how == "diff_bp":
        return 100 * level.diff()
    return level.diff()


def sample(panel: pd.DataFrame, freq: str) -> pd.DataFrame:
    """Level frame on a frequency grid. D = US business days (SPX or DGS10 printed)."""
    if freq == "D":
        grid = panel.index[panel["SPX"].notna() | panel["DGS10"].notna()]
        lv = panel.ffill(limit=3).reindex(grid)
        # weekly/monthly natives must not be smeared onto daily grid
        for k in WEEKLY_NATIVE | MONTHLY_ONLY:
            lv[VARS[k][0]] = np.nan
        return lv
    rule = {"W": "W-FRI", "M": "ME"}[freq]
    lv = panel.ffill(limit=10 if freq == "W" else 40).resample(rule).last()
    if freq == "W":
        for k in MONTHLY_ONLY:
            lv[VARS[k][0]] = np.nan
    return lv


def returns(lv: pd.DataFrame) -> pd.DataFrame:
    out = {"BTC": transform(lv["BTC"], "logret"), "ETH": transform(lv["ETH"], "logret")}
    for k, (col, how, _) in VARS.items():
        out[k] = transform(lv[col], how)
    return pd.DataFrame(out)


def hac_t(y: pd.Series, x: pd.Series, lags: int) -> tuple[float, float]:
    d = pd.concat([y, x], axis=1).dropna()
    if len(d) < 30 or d.iloc[:, 1].std() == 0:
        return np.nan, np.nan
    z = (d - d.mean()) / d.std()
    m = sm.OLS(z.iloc[:, 0], sm.add_constant(z.iloc[:, 1])).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(m.params.iloc[1]), float(m.tvalues.iloc[1])


def corr_row(y, x, lags):
    d = pd.concat([y, x], axis=1).dropna()
    n = len(d)
    if n < 30:
        return dict(n=n, pearson=np.nan, spearman=np.nan, hac_t=np.nan, ci_lo=np.nan, ci_hi=np.nan)
    p = d.corr().iloc[0, 1]
    s = d.corr(method="spearman").iloc[0, 1]
    _, t = hac_t(d.iloc[:, 0], d.iloc[:, 1], lags)
    z, se = np.arctanh(p), 1 / np.sqrt(max(n - 3, 1))
    return dict(n=n, pearson=p, spearman=s, hac_t=t, ci_lo=np.tanh(z - 1.96 * se), ci_hi=np.tanh(z + 1.96 * se))


def section_a(R: dict, out: Path) -> pd.DataFrame:
    rows = []
    lags = {"D": 5, "W": 4, "M": 3}
    for freq, r in R.items():
        for asset in ("BTC", "ETH"):
            for pname, (a, b) in PERIODS.items():
                rr = r.loc[a:b]
                for k in VARS:
                    if freq == "D" and k not in DAILY_OK:
                        continue
                    if freq == "W" and k in MONTHLY_ONLY:
                        continue
                    row = corr_row(rr[asset], rr[k], lags[freq])
                    rows.append(dict(freq=freq, asset=asset, period=pname, var=k, label=VARS[k][2], **row))
    df = pd.DataFrame(rows)
    df.to_csv(out / "A_correlations.csv", index=False, float_format="%.4f")
    return df


def section_b(R: dict, out: Path) -> pd.DataFrame:
    w = R["W"]
    keys = ["SPX", "KOSPI", "GOLD", "BRENT", "DXY_BROAD", "USDJPY", "EURUSD", "USDKRW", "UST2Y", "UST10Y", "REAL10Y",
            "BE10Y", "VIX", "BAA_SPREAD", "NFCI", "NETLIQ", "FED_BS", "STABLES"]
    rows = []
    for y in range(2015, 2027):
        rr = w.loc[str(y)]
        rows.append({"year": y, **{k: rr["BTC"].corr(rr[k]) for k in keys}})
    yt = pd.DataFrame(rows).set_index("year")
    yt.to_csv(out / "B_yearly_weekly_corr_BTC.csv", float_format="%.3f")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    pairs = [("SPX", "S&P 500"), ("DXY_BROAD", "USD broad"), ("REAL10Y", "10Y real yield"),
             ("GOLD", "Gold"), ("VIX", "VIX"), ("NETLIQ", "Net liquidity (weekly chg)")]
    for ax, (k, lab) in zip(axes.flat, pairs):
        d = w[["BTC", k]].dropna().loc["2015":]
        rc = d["BTC"].rolling(26).corr(d[k])
        ax.plot(rc.index, rc.values, lw=1.2, color="#2b6cb0")
        ax.axhline(0, color="#888", lw=0.8)
        for x in ("2020-03-12", "2024-01-11"):
            ax.axvline(pd.Timestamp(x), color="#c53030", lw=0.8, ls="--")
        ax.set_title(f"BTC vs {lab}: rolling 26w corr (weekly)")
        ax.set_ylim(-0.9, 0.9)
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "B_rolling26w_corr.png", dpi=110)
    plt.close(fig)

    d = R["D"][["BTC", "SPX"]].dropna().loc["2015":]
    fig, ax = plt.subplots(figsize=(13, 3.8))
    for win, c in ((60, "#90cdf4"), (250, "#2c5282")):
        ax.plot(d.index, d["BTC"].rolling(win).corr(d["SPX"]), lw=1.1, color=c, label=f"{win}d")
    ax.axhline(0, color="#888", lw=0.8)
    ax.legend()
    ax.set_title("BTC vs S&P 500 rolling daily correlation")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "B_rolling_daily_btc_spx.png", dpi=110)
    plt.close(fig)
    return yt


def section_c(R: dict, out: Path) -> pd.DataFrame:
    rows = []
    for freq, K in (("W", 8), ("D", 3)):
        r = R[freq]
        for pname in ("ALL_2015_2026", "P2_2020_2023", "P3_ETF_2024_2026"):
            a, b = PERIODS[pname]
            for k in VARS:
                if (freq == "D" and k not in DAILY_OK) or k in MONTHLY_ONLY:
                    continue
                for lag in range(-K, K + 1):
                    x = r[k].shift(lag)  # lag>0: macro at t-lag vs crypto at t (macro leads)
                    d = pd.concat([r["BTC"], x], axis=1).loc[a:b].dropna()
                    if len(d) < 30:
                        continue
                    c = d.corr().iloc[0, 1]
                    rows.append(dict(freq=freq, period=pname, var=k, lag=lag, corr=c, n=len(d), z=c * np.sqrt(len(d))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "C_leadlag.csv", index=False, float_format="%.4f")
    return df


def section_d(R: dict, out: Path) -> pd.DataFrame:
    rows = []
    w = R["W"]
    for pname in ("ALL_2015_2026", "P1_2015_2019", "P2_2020_2023", "P3_ETF_2024_2026"):
        a, b = PERIODS[pname]
        for k in VARS:
            if k in MONTHLY_ONLY:
                continue
            d = w[["BTC", k]].loc[a:b].dropna()
            if len(d) < 60 or d[k].std() == 0:
                continue
            res = {}
            for direction, cols in (("macro->btc", ["BTC", k]), ("btc->macro", [k, "BTC"])):
                g = grangercausalitytests(d[cols].values, maxlag=[4])
                res[direction] = g[4][0]["ssr_ftest"][1]
            rows.append(dict(period=pname, var=k, n=len(d), p_macro_to_btc=res["macro->btc"], p_btc_to_macro=res["btc->macro"]))
    df = pd.DataFrame(rows)
    df.to_csv(out / "D_granger_weekly.csv", index=False, float_format="%.4f")
    return df


def section_e(R: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    w = R["W"]
    X = ["SPX", "DXY_BROAD", "REAL10Y", "BE10Y", "GOLD", "BRENT", "VIX", "BAA_SPREAD", "USDJPY"]
    rows, prow = [], []
    for asset in ("BTC", "ETH"):
        for pname, (a, b) in PERIODS.items():
            d = w[[asset] + X].loc[a:b].dropna()
            if len(d) < 80:
                continue
            z = (d - d.mean()) / d.std()
            m = sm.OLS(z[asset], sm.add_constant(z[X])).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
            m_spx = sm.OLS(z[asset], sm.add_constant(z[["SPX"]])).fit()
            for x in X:
                rows.append(dict(asset=asset, period=pname, var=x, beta_std=m.params[x], hac_t=m.tvalues[x], n=len(d),
                                 r2_full=m.rsquared, r2_spx_only=m_spx.rsquared))
            # partial correlation of each macro var with crypto controlling SPX only
            for x in VARS:
                if x in ("SPX",) or x in MONTHLY_ONLY:
                    continue
                dd = w[[asset, "SPX", x]].loc[a:b].dropna()
                if len(dd) < 60 or dd[x].std() == 0:
                    continue
                ra = sm.OLS(dd[asset], sm.add_constant(dd["SPX"])).fit().resid
                rx = sm.OLS(dd[x], sm.add_constant(dd["SPX"])).fit().resid
                pc = ra.corr(rx)
                prow.append(dict(asset=asset, period=pname, var=x, raw_corr=dd[asset].corr(dd[x]), partial_corr_ex_spx=pc,
                                 corr_x_spx=dd[x].corr(dd["SPX"]), n=len(dd), z=pc * np.sqrt(len(dd))))
    reg, par = pd.DataFrame(rows), pd.DataFrame(prow)
    reg.to_csv(out / "E_multivariate_weekly.csv", index=False, float_format="%.4f")
    par.to_csv(out / "E_partial_corr_ex_spx.csv", index=False, float_format="%.4f")
    return reg, par


def section_f(R: dict, out: Path) -> pd.DataFrame:
    rows = []
    d0 = R["D"]
    for pname in ("ALL_2015_2026", "P2_2020_2023", "P3_ETF_2024_2026"):
        a, b = PERIODS[pname]
        d = d0[["BTC", "ETH", "SPX", "VIX", "DXY_BROAD", "REAL10Y", "GOLD"]].loc[a:b].dropna(subset=["BTC", "SPX", "VIX"])
        q10, q90 = d["SPX"].quantile([0.10, 0.90])
        v90 = d["VIX"].quantile(0.90)
        conds = {
            "all": d.index == d.index,
            "spx_bottom10": d["SPX"] <= q10,
            "spx_middle80": (d["SPX"] > q10) & (d["SPX"] < q90),
            "spx_top10": d["SPX"] >= q90,
            "vix_spike_top10": d["VIX"] >= v90,
        }
        for cname, m in conds.items():
            s = d[m]
            beta = np.polyfit(s["SPX"], s["BTC"], 1)[0] if len(s) > 10 else np.nan
            rows.append(dict(period=pname, cond=cname, n=int(m.sum()), corr_btc_spx=s["BTC"].corr(s["SPX"]),
                             beta_btc_spx=beta, mean_btc=s["BTC"].mean(), mean_spx=s["SPX"].mean(),
                             p_btc_down=(s["BTC"] < 0).mean(), corr_btc_gold=s["BTC"].corr(s["GOLD"]),
                             corr_btc_dxy=s["BTC"].corr(s["DXY_BROAD"])))
        # semi-betas (down vs up market)
        dn, up = d[d["SPX"] < 0], d[d["SPX"] > 0]
        rows.append(dict(period=pname, cond="semibeta_down", n=len(dn), beta_btc_spx=np.polyfit(dn["SPX"], dn["BTC"], 1)[0],
                         corr_btc_spx=dn["BTC"].corr(dn["SPX"])))
        rows.append(dict(period=pname, cond="semibeta_up", n=len(up), beta_btc_spx=np.polyfit(up["SPX"], up["BTC"], 1)[0],
                         corr_btc_spx=up["BTC"].corr(up["SPX"])))
    df = pd.DataFrame(rows)
    df.to_csv(out / "F_tail_conditional_daily.csv", index=False, float_format="%.4f")
    return df


def fwd_sum(r: pd.Series, h: int) -> pd.Series:
    """sum of r[t+1..t+h] (log-return horizon h strictly after t)."""
    return r[::-1].rolling(h).sum()[::-1].shift(-1)


def section_g(R: dict, lv: dict, out: Path) -> pd.DataFrame:
    rows = []
    w, wl = R["W"], lv["W"]
    btc = w["BTC"]
    for var, col in (("NETLIQ", "NETLIQ"), ("FED_BS", "WALCL"), ("RESERVES", "WRESBAL"), ("STABLES", "STABLE_SUPPLY")):
        x13 = 100 * np.log(wl[col]).diff(13)
        for H in (4, 13):
            fb = fwd_sum(btc, H)
            for L in range(0, 27, 1):
                # macro 13w change known at t-L-1 (publish lag 1w) vs BTC return t+1..t+H
                xs = x13.shift(L + 1)
                for pname in ("ALL_2015_2026", "P2_2020_2023", "P3_ETF_2024_2026"):
                    a, b = PERIODS[pname]
                    d = pd.concat([fb, xs], axis=1).loc[a:b].dropna()
                    if len(d) < 40:
                        continue
                    _, t = hac_t(d.iloc[:, 0], d.iloc[:, 1], lags=H + 13)
                    rows.append(dict(var=var, H_weeks=H, lead_weeks=L + 1, period=pname, corr=d.corr().iloc[0, 1], hac_t=t, n=len(d)))
        # reverse direction placebo: past BTC 13w return vs future 13w liquidity change
        fx = x13.shift(-13)
        b13 = 100 * np.log(wl["BTC"]).diff(13)
        d = pd.concat([fx, b13], axis=1).loc["2015":].dropna()
        _, t = hac_t(d.iloc[:, 0], d.iloc[:, 1], lags=26)
        rows.append(dict(var=var, H_weeks=13, lead_weeks=-13, period="REVERSE_btc_leads", corr=d.corr().iloc[0, 1], hac_t=t, n=len(d)))

    # M2 monthly: yoy growth known with ~1m publication lag
    m, ml = R["M"], lv["M"]
    m2y = 100 * np.log(ml["M2SL"]).diff(12)
    for H in (1, 3, 6):
        fb = fwd_sum(m["BTC"], H)
        for L in range(0, 13):
            xs = m2y.shift(L + 1)
            for pname in ("ALL_2015_2026", "P2_2020_2023", "P3_ETF_2024_2026"):
                a, b = PERIODS[pname]
                d = pd.concat([fb, xs], axis=1).loc[a:b].dropna()
                if len(d) < 24:
                    continue
                _, t = hac_t(d.iloc[:, 0], d.iloc[:, 1], lags=H + 12)
                rows.append(dict(var="M2_yoy", H_weeks=H * 4.33, lead_weeks=round((L + 1) * 4.33, 1), period=pname,
                                 corr=d.corr().iloc[0, 1], hac_t=t, n=len(d)))
    df = pd.DataFrame(rows)
    df.to_csv(out / "G_liquidity_lead.csv", index=False, float_format="%.4f")

    # chart: level co-trend vs differenced
    d = pd.concat([np.log(wl["BTC"]), np.log(wl["NETLIQ"]), np.log(ml["M2SL"]).resample("W-FRI").ffill()], axis=1).loc["2015":]
    d.columns = ["log BTC", "log NetLiq", "log M2"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    z = (d - d.mean()) / d.std()
    z.plot(ax=ax[0], lw=1.1)
    ax[0].set_title("Levels (z-scored logs): co-trending")
    ax[0].grid(alpha=0.25)
    sub = df[(df["var"] == "NETLIQ") & (df["H_weeks"] == 13) & (df["period"] == "ALL_2015_2026")]
    ax[1].bar(sub["lead_weeks"], sub["corr"], color=np.where(sub["hac_t"].abs() > 2, "#2b6cb0", "#a0aec0"))
    ax[1].axhline(0, color="#888", lw=0.8)
    ax[1].set_title("NetLiq 13w chg (lead L wks) vs BTC fwd 13w ret\n(blue = |HAC t|>2)")
    ax[1].set_xlabel("lead weeks")
    ax[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out / "G_liquidity_lead.png", dpi=110)
    plt.close(fig)
    return df


def section_h(R: dict, lv: dict, out: Path) -> pd.DataFrame:
    w, wl = R["W"], lv["W"]
    f4 = fwd_sum(w["BTC"], 4)
    states = {
        "USD_13w_chg": 100 * np.log(wl["DTWEXBGS"]).diff(13),
        "REAL10Y_13w_chg_bp": 100 * wl["DFII10"].diff(13),
        "UST2Y_13w_chg_bp": 100 * wl["DGS2"].diff(13),
        "NETLIQ_13w_chg": 100 * np.log(wl["NETLIQ"]).diff(13),
        "VIX_level": wl["VIXCLS"],
        "NFCI_level": wl["NFCI"],
        "BAA_spread_level": wl["BAA10Y"],
        "SPX_13w_ret": 100 * np.log(wl["SPX"]).diff(13),
        "GOLD_13w_ret": 100 * np.log(wl["GOLD"]).diff(13),
        "CURVE_2s10s_level": wl["T10Y2Y"],
    }
    rows = []
    for sname, s in states.items():
        s = s.shift(1)  # known one week before the forward window starts
        lo = s.expanding(104).quantile(1 / 3)
        hi = s.expanding(104).quantile(2 / 3)
        bucket = pd.Series(np.where(s <= lo, "low", np.where(s >= hi, "high", "mid")), index=s.index).where(lo.notna())
        d = pd.concat([f4.rename("f4"), bucket.rename("b")], axis=1).loc["2015":].dropna()
        for bname in ("low", "mid", "high"):
            x = d.loc[d["b"] == bname, "f4"]
            if len(x) < 20:
                continue
            ones = pd.Series(1.0, index=x.index)
            t = sm.OLS(x, ones).fit(cov_type="HAC", cov_kwds={"maxlags": 4}).tvalues.iloc[0]
            rows.append(dict(state=sname, bucket=bname, n_weeks=len(x), mean_fwd4w_btc_pct=x.mean(), hac_t=t, hit_rate=(x > 0).mean()))
        hiv = d.loc[d["b"] == "high", "f4"]
        lov = d.loc[d["b"] == "low", "f4"]
        if len(hiv) > 20 and len(lov) > 20:
            rows.append(dict(state=sname, bucket="high_minus_low", n_weeks=len(hiv) + len(lov),
                             mean_fwd4w_btc_pct=hiv.mean() - lov.mean(),
                             hac_t=(hiv.mean() - lov.mean()) / np.sqrt(hiv.var() * 4 / len(hiv) + lov.var() * 4 / len(lov))))
    df = pd.DataFrame(rows)
    df.to_csv(out / "H_regime_conditional.csv", index=False, float_format="%.3f")
    return df


def section_i(R: dict, out: Path) -> pd.DataFrame:
    w = R["W"]
    preds = ["SPX", "GOLD", "DXY_BROAD", "USDJPY", "UST2Y", "UST10Y", "REAL10Y", "BE10Y", "VIX", "BAA_SPREAD", "NFCI",
             "NFCI_RISK", "NETLIQ", "FED_BS", "RRP", "TGA", "STABLES", "BRENT", "BTC"]
    rows = []
    for H in (1, 4):
        y = fwd_sum(w["BTC"], H).rename("y")
        X = w[preds].rolling(4).sum().shift(1)  # 4w change, known one week before forecast origin
        X.columns = [c + "_4w" for c in preds]
        base = pd.concat([y, X], axis=1).loc["2015":]
        for p in list(X.columns) + ["KITCHEN_SINK", "COMBINATION_MEAN"]:
            if p == "KITCHEN_SINK":
                cols = [c for c in X.columns if c != "STABLES_4w"]
            elif p == "COMBINATION_MEAN":
                cols = None
            else:
                cols = [p]
            d = base.dropna(subset=["y"] + (cols if cols else [c for c in X.columns if c != "STABLES_4w"]))
            dates = d.index
            start = dates.searchsorted(pd.Timestamp("2018-01-05"))
            e_model, e_mean, fc_m, fc_b, realised = [], [], [], [], []
            for i in range(start, len(dates), 1):
                # training sample ends H weeks before origin so that its targets are realised
                tr = d.iloc[: max(i - H + 1, 0)]
                if len(tr) < 100:
                    continue
                te = d.iloc[i]
                mean_fc = tr["y"].mean()
                if p == "COMBINATION_MEAN":
                    fcs = []
                    for c in [c for c in X.columns if c != "STABLES_4w"]:
                        bb = np.polyfit(tr[c], tr["y"], 1)
                        fcs.append(bb[0] * te[c] + bb[1])
                    fc = float(np.mean(fcs))
                else:
                    m = sm.OLS(tr["y"], sm.add_constant(tr[cols], has_constant="add")).fit()
                    fc = float(m.predict(np.r_[1.0, te[cols].values].reshape(1, -1))[0])
                fc_m.append(fc)
                fc_b.append(mean_fc)
                realised.append(te["y"])
            if len(realised) < 50:
                continue
            yv, fm, fb = np.array(realised), np.array(fc_m), np.array(fc_b)
            sse_m, sse_b = ((yv - fm) ** 2).sum(), ((yv - fb) ** 2).sum()
            r2os = 1 - sse_m / sse_b
            f_cw = (yv - fb) ** 2 - ((yv - fm) ** 2 - (fb - fm) ** 2)
            cw = sm.OLS(f_cw, np.ones_like(f_cw)).fit(cov_type="HAC", cov_kwds={"maxlags": max(H, 1)})
            dir_hit = np.mean(np.sign(fm - fb) == np.sign(yv - fb))
            rows.append(dict(H_weeks=H, predictor=p, n_oos=len(yv), r2_os_pct=100 * r2os, clark_west_t=float(cw.tvalues[0]),
                             direction_hit_vs_mean=dir_hit))
    df = pd.DataFrame(rows)
    df.to_csv(out / "I_oos_predictability.csv", index=False, float_format="%.4f")
    return df


def section_j(R: dict, lv: dict, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Robustness of the strongest regime effects: per sub-period buckets, a weekly in/out rule
    (non-overlapping 1w returns), a BTC-momentum control, and 13w-change OOS R2 by sub-period."""
    w, wl = R["W"], lv["W"]
    f1, f4 = w["BTC"].shift(-1), fwd_sum(w["BTC"], 4)
    X = pd.DataFrame({
        "USD13": 100 * np.log(wl["DTWEXBGS"]).diff(13), "EURUSD13": 100 * np.log(wl["DEXUSEU"]).diff(13),
        "REAL13": 100 * wl["DFII10"].diff(13), "NETLIQ13": 100 * np.log(wl["NETLIQ"]).diff(13),
        "BTC13": 100 * np.log(wl["BTC"]).diff(13), "SPX13": 100 * np.log(wl["SPX"]).diff(13),
        "GOLD13": 100 * np.log(wl["GOLD"]).diff(13),
    }).shift(1)
    subs = [("2015", "2019"), ("2020", "2023"), ("2024", "2026"), ("2015", "2026")]
    rows = []
    for name, want in (("USD13", "low"), ("REAL13", "low"), ("NETLIQ13", "high")):
        s = X[name]
        lo, hi = s.expanding(104).quantile(1 / 3), s.expanding(104).quantile(2 / 3)
        b = pd.Series(np.where(s <= lo, "low", np.where(s >= hi, "high", "mid")), index=s.index).where(lo.notna())
        d = pd.concat([f4.rename("f4"), f1.rename("f1"), b.rename("b")], axis=1).loc["2015":].dropna()
        for a, c in subs:
            dd = d.loc[a:c]
            pos = (dd["b"] == want).astype(float)
            strat = pos * dd["f1"]
            sharpe = lambda x: x.mean() / x.std() * np.sqrt(52)
            row = dict(state=name, rule_bucket=want, period=f"{a}-{c}", exposure=pos.mean(),
                       rule_sharpe=sharpe(strat), buyhold_sharpe=sharpe(dd["f1"]),
                       rule_ann_logret_pct=strat.mean() * 52, buyhold_ann_logret_pct=dd["f1"].mean() * 52)
            for bk in ("low", "mid", "high"):
                row[f"fwd4w_mean_{bk}"] = dd.loc[dd["b"] == bk, "f4"].mean()
                row[f"n_{bk}"] = int((dd["b"] == bk).sum())
            rows.append(row)
    rob = pd.DataFrame(rows)

    # momentum control: standardised f4 on USD13 with and without BTC13 (HAC 8)
    d = pd.concat([f4.rename("f4"), X], axis=1).loc["2015":].dropna()
    z = (d - d.mean()) / d.std()
    ctrl = []
    for a, c in subs:
        zz = z.loc[a:c]
        for cols in (["USD13"], ["USD13", "BTC13"], ["USD13", "BTC13", "SPX13", "GOLD13", "REAL13", "NETLIQ13"]):
            m = sm.OLS(zz["f4"], sm.add_constant(zz[cols])).fit(cov_type="HAC", cov_kwds={"maxlags": 8})
            ctrl.append(dict(kind="insample_ctrl", period=f"{a}-{c}", spec="+".join(cols), beta_USD13=m.params["USD13"],
                             t_USD13=m.tvalues["USD13"], beta_BTC13=m.params.get("BTC13", np.nan), t_BTC13=m.tvalues.get("BTC13", np.nan)))
    # 13w-change OOS, expanding univariate OLS from 2018
    for H in (1, 4):
        base = pd.concat([fwd_sum(w["BTC"], H).rename("y"), X], axis=1).loc["2015":].dropna()
        dates = base.index
        st = dates.searchsorted(pd.Timestamp("2018-01-05"))
        for pr in X.columns:
            fm, fb, yy = [], [], []
            for i in range(st, len(dates)):
                tr, te = base.iloc[: i - H + 1], base.iloc[i]
                bb = np.polyfit(tr[pr], tr["y"], 1)
                fm.append(bb[0] * te[pr] + bb[1])
                fb.append(tr["y"].mean())
                yy.append(te["y"])
            yy, fm, fb = map(np.array, (yy, fm, fb))
            fcw = (yy - fb) ** 2 - ((yy - fm) ** 2 - (fb - fm) ** 2)
            row = dict(kind="oos_13w", H_weeks=H, predictor=pr, n_oos=len(yy),
                       r2_os_pct=100 * (1 - ((yy - fm) ** 2).sum() / ((yy - fb) ** 2).sum()),
                       clark_west_t=float(sm.OLS(fcw, np.ones_like(fcw)).fit(cov_type="HAC", cov_kwds={"maxlags": H}).tvalues[0]))
            idx = dates[st:]
            for a, c in (("2018", "2019"), ("2020", "2023"), ("2024", "2026")):
                mk = (idx >= pd.Timestamp(a)) & (idx <= pd.Timestamp(f"{c}-12-31"))
                row[f"r2_os_{a}_{c}"] = 100 * (1 - ((yy[mk] - fm[mk]) ** 2).sum() / ((yy[mk] - fb[mk]) ** 2).sum())
            ctrl.append(row)
    ctl = pd.DataFrame(ctrl)
    rob.to_csv(out / "J_regime_robustness.csv", index=False, float_format="%.3f")
    ctl.to_csv(out / "J_momentum_control_and_oos13w.csv", index=False, float_format="%.4f")
    return rob, ctl


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/experiments/macro_crypto_correlation_20261027")
    a = ap.parse_args()
    out = Path(a.out)
    panel = pd.read_parquet(out / "panel_daily.parquet")

    lv = {f: sample(panel, f) for f in ("D", "W", "M")}
    R = {f: returns(lv[f]) for f in lv}
    # never let weekly-native unchanged prints masquerade as daily zeros; mask where level missing
    for f in R:
        for k, (col, _, _) in VARS.items():
            R[f].loc[lv[f][col].isna(), k] = np.nan

    # sanity: Yahoo SPX vs FRED SP500
    d = pd.concat([transform(lv["D"]["SPX"], "logret"), transform(lv["D"]["SP500"], "logret")], axis=1).dropna()
    sanity = {"spx_yahoo_vs_fred_daily_corr": float(d.corr().iloc[0, 1])}

    A = section_a(R, out)
    B = section_b(R, out)
    C = section_c(R, out)
    D = section_d(R, out)
    E = section_e(R, out)
    F = section_f(R, out)
    G = section_g(R, lv, out)
    H = section_h(R, lv, out)
    I = section_i(R, out)
    section_j(R, lv, out)
    (out / "sanity.json").write_text(json.dumps(sanity, indent=1))
    print(json.dumps(sanity))
    print("done", {k: len(v) if hasattr(v, "__len__") else 0 for k, v in dict(A=A, B=B, C=C, D=D, F=F, G=G, H=H, I=I).items()})


if __name__ == "__main__":
    main()
