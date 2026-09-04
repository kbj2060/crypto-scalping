#!/usr/bin/env python3
"""Does ANY causally-available indicator (microstructure/tail-risk model indicators, regime state,
volatility, session timing) reliably distinguish periods where the dashboard's 8 evidence signals
(scripts/live_evidence_signal_dashboard_20260823.py::compute_signals()) will hit their predicted
direction from periods where they won't? Per the plan at /home/kbj20/.claude/plans/pure-hugging-book.md
(approved 2026-08-28).

Motivated by a same-day false start: comparing 2 hand-picked real windows found a seemingly clean
"toxicity" story that completely failed to replicate across 221 independent 12h windows spanning
the full available history (shadow_toxicity_score rho=-0.01, p=0.87) -- only `eai` (energy activity
index) was statistically significant (rho~-0.2, p<0.05), weakly. This script re-does that screen
properly: BOTH at 6h-window and individual-fire granularity (a feature must agree in sign AND
significance at both to survive -- a strong filter against exactly the false-positive trap that
caught the earlier "toxicity" story), with circular-shift permutation significance (shift_z, not a
naive scipy p-value) and a partial-correlation control against realized_vol_ratio (is a feature's
effect real, or just riding "we're already in a vol-expansion regime", the same failure mode
documented in docs/experiments/evidence_signal_breadth_risk_gate_60coin_20260815.md).

IMPORTANT ceiling on this result's status, stated once here and in the report: this repo's own
promotion bar for evidence-signal work (docs/model_contracts/evidence_signal_quant_use_contract_
20260815.md) requires sign agreement across >=4 INDEPENDENT CHRONOLOGICAL PERIODS. This analysis
has exactly ONE continuous period (2026-05-03~08-19, bounded by microstructure_1m's start and the
offline evidence-signal source's coverage) -- that bar is structurally unmeetable here, not a gap
to argue around. Report status is therefore always "exploratory_single_period_below_promotion_bar",
never "confirmed"/"validated", regardless of how the numbers come out.

Diagnostic/research only. Does NOT touch dashboard/server.py, app.js, trading_bot.py, or
trading_bot_modules/. Report written to tmp/, NOT data/ensemble/reports/ (that directory is read by
trading_bot_modules/runtime_config.py as a de facto live promotion registry).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/kbj20/crypto-scalping")
sys.path.insert(0, "/home/kbj20/crypto-scalping/scripts")

import joblib
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from live_evidence_signal_dashboard_20260823 import compute_signals, SIGNAL_ORDER
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z
from research_eth_model_indicator_volatility_screen_20260825 import shift_z
from train_eth_regime_volexpand_20260827 import _compute_er24
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12

ROOT = Path("/home/kbj20/crypto-scalping")
SCRATCH = Path("/tmp/claude-1000/-home-kbj20-crypto-scalping/3246e659-25f1-4023-91a1-0d925c1b2b1a/scratchpad")
OUT_DIR = ROOT / "tmp/eth_evidence_signal_reliability_gate_20260828"
GBM2_MODEL_PATH = ROOT / "tmp/eth_regime_gbm2_trend_chop_20260827/model.joblib"

ETH_PATH = ROOT / "data/splits/year_oos/training_features_2026_rebuilt.csv"
BTC_PATH = ROOT / "data/splits/year_oos/btc_features_2026.csv"

RANGE_START, RANGE_END = pd.Timestamp("2026-05-01"), pd.Timestamp("2026-08-19 23:59:59")
TRAIN_START, TRAIN_END = pd.Timestamp("2026-05-03"), pd.Timestamp("2026-07-31 23:59:59")  # matches research_eth_whale_position_score_direction_screen_20260825.py
VAL_START, VAL_END = pd.Timestamp("2026-08-01"), pd.Timestamp("2026-08-19 23:59:59")       # extended to offline CSV's own cap (that script's own VAL_END was 08-16)
WHALE_POS_VALID_SINCE = pd.Timestamp("2026-07-18 23:58:00")  # docs/experiments/eth_candidate_evidence_signal_whale_confirmation_combination_20260823.md

HORIZON = 12          # 1h forward, matches every prior screen in this lineage
WINDOW_HOURS = 6       # matches NYSE-open [-60,+60]min effect window better than 12h; still clears >=5 fires/window in practice (checked below)
EAI_LIVE_THRESHOLD = 2.0  # microstructure_scanner.py:107, MS_EAI_THRESHOLD default -- live "squeeze" cutoff, reused not reinvented

MS_COLS = ["obi", "taker_buy_ratio", "spoofing_score", "nif_whale", "nif_retail", "eai",
           "oi_delta_pct", "shadow_toxicity_score", "shadow_queue_collapse",
           "shadow_absorption_score", "whale_position_score"]
TR_COLS = ["long_usd_1m", "short_usd_1m", "shadow_aftershock_prob", "liq_event_count_1m"]
NEW_COLS = ["realized_vol_ratio", "er_24", "gbm2_trend_prob", "nyse_open_flag"]
ALL_FEATURES = MS_COLS + TR_COLS + NEW_COLS


def log(msg: str) -> None:
    print(f"[reliability_gate] {msg}", flush=True)


# =================================================================================================
# Phase A -- evidence signals + forward-1h fire outcomes (offline range, causal, no OI dependency)
# =================================================================================================

def build_fires(range_start: pd.Timestamp, range_end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (sig, fires): sig = full 5m frame with signals + new causal features;
    fires = one row per bottom/top firing with forward-1h hit/pred_dir_ret."""
    raw = pd.read_csv(ETH_PATH, low_memory=False)
    raw["timestamp"] = pd.to_datetime(raw["timestamp"])
    raw = raw[(raw["timestamp"] >= range_start) & (raw["timestamp"] <= range_end)].sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    eth = raw[["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]].copy()
    btc = pd.read_csv(BTC_PATH, parse_dates=["timestamp"], usecols=["timestamp", "high", "low"])
    funding = load_funding_z()
    funding = funding.copy()
    funding["calc_time"] = funding["calc_time"].astype("datetime64[ns]")
    eth["timestamp"] = eth["timestamp"].astype("datetime64[ns]")

    sig = compute_signals(eth, btc_df=btc, funding_df=funding).reset_index(drop=True)
    sig["er_24"] = _compute_er24(sig["close"])
    log_ret = np.log(sig["close"] / sig["close"].shift(1))
    sig["realized_vol_ratio"] = log_ret.rolling(12).std() / log_ret.rolling(288).std()

    # NYSE-open flag: simplified deterministic UTC-hour check (12:30-14:30 UTC == 9:30 ET +-60min,
    # EDT year-round for this May-Aug range) -- no holiday-calendar exclusion, acceptable imprecision
    # for a screening feature (not a promoted signal). See live_session_volatility_alert_20260826.py
    # for the real mcal-based live version this approximates.
    tmin = sig["timestamp"].dt.hour * 60 + sig["timestamp"].dt.minute
    is_weekday = sig["timestamp"].dt.dayofweek < 5
    sig["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)

    # GBM2 model trend_prob (the actual live-serving model's own prediction, not a hand RegimeEngine
    # recompute). training_features_2026_rebuilt.csv already carries FeatureEngineer's full output
    # (confirmed earlier this session, 142 columns incl. realized_vol_ratio) -- only state7/state12
    # columns need deriving on top, not the whole engineered set from scratch.
    feats = _with_raw_state12(raw)
    payload = joblib.load(GBM2_MODEL_PATH)
    cols, med = payload["feature_cols"], pd.Series(payload["feature_medians"])
    for c in cols:
        if c not in feats.columns:
            feats[c] = med.get(c, 0.0)
    x = feats[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    gbm2 = pd.DataFrame({"timestamp": feats["timestamp"].reset_index(drop=True),
                          "gbm2_trend_prob": proba[:, payload["classes"].index("trend")]})
    sig = sig.merge(gbm2, on="timestamp", how="left")
    sig["pos"] = sig.index

    close = sig["close"].to_numpy()
    n = len(sig)
    rows = []
    for name, _desc in SIGNAL_ORDER:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in sig.columns:
                continue
            fire_idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
            fire_idx = fire_idx[fire_idx + HORIZON < n]
            if len(fire_idx) == 0:
                continue
            entry, fwd = close[fire_idx], close[fire_idx + HORIZON]
            ret = (fwd - entry) / entry
            hit = (ret > 0) if side == "bottom" else (ret < 0)
            pred_dir_ret = ret * (1 if side == "bottom" else -1)
            rows.append(pd.DataFrame({"pos": fire_idx, "timestamp": sig["timestamp"].to_numpy()[fire_idx],
                                       "signal": name, "side": side, "hit": hit.astype(float),
                                       "pred_dir_ret": pred_dir_ret}))
    fires = pd.concat(rows, ignore_index=True).merge(sig[["pos"] + NEW_COLS], on="pos", how="left")
    return sig, fires


# =================================================================================================
# Phase B -- merge live microstructure_1m / tail_risk_1m (reuses today's earlier full-range pull)
# =================================================================================================

def merge_microstructure(sig: pd.DataFrame, fires: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ms = pd.read_csv(SCRATCH / "full_range_export/microstructure_1m_full.csv", parse_dates=["ts"])
    tr = pd.read_csv(SCRATCH / "full_range_export/tail_risk_1m_full.csv", parse_dates=["ts"])
    # ts is KST TIMESTAMPTZ -- must go through tz_convert("UTC") first, a naive tz_localize(None)
    # silently keeps KST wall-clock labeled as UTC (a 9h shift that already broke this exact data
    # family once, see docs/experiments/eth_candidate_evidence_signal_whale_confirmation_combination_20260823.md)
    ms["timestamp"] = ms["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    tr["timestamp"] = tr["ts"].dt.tz_convert("UTC").dt.tz_localize(None)
    ms.loc[ms["timestamp"] < WHALE_POS_VALID_SINCE, "whale_position_score"] = np.nan

    ms5 = ms.set_index("timestamp")[MS_COLS].resample("5min", label="right", closed="right").mean().reset_index()
    tr5 = tr.set_index("timestamp")[TR_COLS].resample("5min", label="right", closed="right").agg(
        {"long_usd_1m": "sum", "short_usd_1m": "sum", "shadow_aftershock_prob": "mean", "liq_event_count_1m": "sum"}
    ).reset_index()

    sig = sig.drop(columns=[c for c in MS_COLS + TR_COLS if c in sig.columns], errors="ignore")
    sig = sig.merge(ms5, on="timestamp", how="left").merge(tr5, on="timestamp", how="left")
    fires = fires.drop(columns=[c for c in MS_COLS + TR_COLS if c in fires.columns], errors="ignore")
    fires = fires.merge(sig[["pos"] + MS_COLS + TR_COLS], on="pos", how="left")
    return sig, fires


# =================================================================================================
# Phase C -- dual-granularity screening: window-level (6h) + fire-level, shift_z + partial corr
# =================================================================================================

def build_windows(sig: pd.DataFrame, fires: pd.DataFrame) -> pd.DataFrame:
    sig = sig.copy()
    sig["window_id"] = ((sig["timestamp"] - RANGE_START) / pd.Timedelta(hours=WINDOW_HOURS)).astype(int)
    fires = fires.merge(sig[["pos", "window_id"]], on="pos", how="left")
    rows = []
    for wid, wdf in sig.groupby("window_id"):
        wfires = fires[fires["window_id"] == wid]
        if len(wfires) < 5:
            continue
        row = {"window_id": wid, "timestamp": wdf["timestamp"].iloc[0], "n_fires": len(wfires),
               "hit_rate": wfires["hit"].mean(), "mean_pred_dir_ret": wfires["pred_dir_ret"].mean()}
        row.update(wdf[ALL_FEATURES].mean().to_dict())
        rows.append(row)
    return pd.DataFrame(rows)


def partial_ic(feat: pd.Series, target: pd.Series, confound: pd.Series) -> tuple[float, float, int]:
    """Spearman IC of feat vs target, controlling for confound (realized_vol_ratio) via the same
    residual-rank partial-correlation pattern as research_eth_model_indicator_volatility_screen_
    20260825.py::partial_corr_check -- generalized here to an arbitrary target, not just fwd-vol."""
    d = pd.concat([feat, target, confound], axis=1).dropna()
    d.columns = ["f", "t", "c"]
    if len(d) < 40:
        return float("nan"), float("nan"), len(d)
    r_ft, z_ft, n = shift_z(d["f"], d["t"])
    r_fc, _, _ = shift_z(d["f"], d["c"])
    r_tc = spearmanr(d["t"], d["c"]).statistic
    denom = np.sqrt(max((1 - r_fc ** 2) * (1 - r_tc ** 2), 1e-9))
    partial = (r_ft - r_fc * r_tc) / denom
    return float(r_ft), float(partial), n


def screen(df: pd.DataFrame, target_col: str, label: str) -> pd.DataFrame:
    rows = []
    for feat in ALL_FEATURES:
        if feat not in df.columns or feat in ("realized_vol_ratio",):
            continue
        raw_ic, z, n = shift_z(df[feat], df[target_col]) if feat in df.columns else (np.nan, np.nan, 0)
        partial, partial_val, npar = partial_ic(df[feat], df[target_col], df["realized_vol_ratio"]) if "realized_vol_ratio" in df.columns else (np.nan, np.nan, 0)
        rows.append({"granularity": label, "feature": feat, "n": n, "raw_ic": raw_ic, "shift_z": z,
                     "partial_ic": partial_val, "n_partial": npar})
    return pd.DataFrame(rows)


# =================================================================================================
# Phase D -- escalation ladder: Step0 (naive) / Step1 (eai>=2.0 threshold) / Step2 (small logistic)
# =================================================================================================

def evaluate_ladder(df: pd.DataFrame, target_col: str, survivors: list[str], fit_scaler=None, fit_model=None):
    step0 = df[target_col].mean()
    hi_eai = df["eai"] >= EAI_LIVE_THRESHOLD
    step1_lo = df.loc[hi_eai, target_col].mean() if hi_eai.sum() >= 5 else np.nan   # expect worse (per Round-2 sign)
    step1_hi = df.loc[~hi_eai, target_col].mean() if (~hi_eai).sum() >= 5 else np.nan
    result = {"n": len(df), "step0_naive_mean": step0,
              "step1_low_eai_mean": step1_hi, "step1_high_eai_mean": step1_lo,
              "step1_n_low_eai": int((~hi_eai).sum()), "step1_n_high_eai": int(hi_eai.sum())}
    if fit_model is not None and survivors:
        X = fit_scaler.transform(df[survivors].fillna(df[survivors].median()))
        proba = fit_model.predict_proba(X)[:, 1]
        top_third = proba >= np.quantile(proba, 2 / 3)
        bot_third = proba <= np.quantile(proba, 1 / 3)
        result["step2_top_tercile_mean"] = df.loc[top_third, target_col].mean()
        result["step2_bottom_tercile_mean"] = df.loc[bot_third, target_col].mean()
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"Building fires + causal features, {RANGE_START.date()}~{RANGE_END.date()}...")
    sig, fires = build_fires(RANGE_START, RANGE_END)
    log(f"  total fires: {len(fires)}")

    log("Merging live microstructure_1m / tail_risk_1m (reusing today's full-range export)...")
    sig, fires = merge_microstructure(sig, fires)
    log(f"  microstructure coverage in fires: {fires[MS_COLS[0]].notna().mean():.1%}, "
        f"whale_position_score valid coverage: {fires['whale_position_score'].notna().mean():.1%} "
        f"(expected low pre-07-18, healthy after)")

    log(f"Building {WINDOW_HOURS}h windows...")
    windows = build_windows(sig, fires)
    log(f"  usable windows (>=5 fires): {len(windows)}")

    train_f = fires[(fires["timestamp"] >= TRAIN_START) & (fires["timestamp"] <= TRAIN_END)]
    val_f = fires[(fires["timestamp"] >= VAL_START) & (fires["timestamp"] <= VAL_END)]
    train_w = windows[(windows["timestamp"] >= TRAIN_START) & (windows["timestamp"] <= TRAIN_END)]
    val_w = windows[(windows["timestamp"] >= VAL_START) & (windows["timestamp"] <= VAL_END)]
    log(f"  TRAIN: {len(train_f)} fires / {len(train_w)} windows.  VAL: {len(val_f)} fires / {len(val_w)} windows.")

    log("\n=== Screening (TRAIN), both granularities, shift_z + partial IC vs realized_vol_ratio ===")
    scr_fire = screen(train_f, "pred_dir_ret", "fire")
    scr_win = screen(train_w, "mean_pred_dir_ret", "window")
    scr = pd.concat([scr_fire, scr_win], ignore_index=True)
    pd.set_option("display.width", 200)
    print(scr.round(4).to_string(index=False))

    # a feature "survives" only if BOTH granularities agree in sign and clear |shift_z|>=2 on raw IC
    piv = scr.pivot_table(index="feature", columns="granularity", values=["raw_ic", "shift_z"])
    survivors = []
    for feat in ALL_FEATURES:
        if feat not in piv.index or feat == "realized_vol_ratio":
            continue
        try:
            ic_f, ic_w = piv.loc[feat, ("raw_ic", "fire")], piv.loc[feat, ("raw_ic", "window")]
            z_f, z_w = piv.loc[feat, ("shift_z", "fire")], piv.loc[feat, ("shift_z", "window")]
        except KeyError:
            continue
        if pd.isna(ic_f) or pd.isna(ic_w):
            continue
        same_sign = np.sign(ic_f) == np.sign(ic_w) and ic_f != 0
        both_sig = abs(z_f) >= 2 and abs(z_w) >= 2
        if same_sign and both_sig:
            survivors.append(feat)
    log(f"\nDual-granularity survivors (same sign + |shift_z|>=2 at BOTH fire and window level): {survivors}")

    log("\n=== Escalation ladder, evaluated on VAL (TRAIN-fit model where applicable) ===")
    fit_scaler, fit_model = None, None
    if survivors:
        Xtr = train_f[survivors].fillna(train_f[survivors].median())
        ytr = (train_f["pred_dir_ret"] > 0).astype(int)
        fit_scaler = StandardScaler().fit(Xtr)
        fit_model = LogisticRegression(C=0.5, max_iter=1000).fit(fit_scaler.transform(Xtr), ytr)

    ladder_val_fire = evaluate_ladder(val_f, "pred_dir_ret", survivors, fit_scaler, fit_model)
    ladder_val_win = evaluate_ladder(val_w, "mean_pred_dir_ret", survivors, fit_scaler, fit_model)
    print("VAL, fire-level:", json.dumps(ladder_val_fire, indent=2, default=float))
    print("VAL, window-level:", json.dumps(ladder_val_win, indent=2, default=float))

    log(f"\n=== Small live holdout: 2026-08-20 ~ today (single-touch, evaluated once) ===")
    from live_regime_wide24_signal_20260826 import SYMBOL, BTC_SYMBOL, _fetch_klines
    now = pd.Timestamp.now("UTC").tz_localize(None)
    hstart = pd.Timestamp("2026-08-14")  # extra lead-in for rolling-window feature warmup before 08-20
    start_ms, end_ms = int(hstart.timestamp() * 1000), int(now.timestamp() * 1000)
    heth = _fetch_klines(SYMBOL, start_ms, end_ms)
    hbtc = _fetch_klines(BTC_SYMBOL, start_ms, end_ms)
    hfunding = load_funding_z()
    hfunding = hfunding.copy()
    hfunding["calc_time"] = hfunding["calc_time"].astype("datetime64[ns]")
    heth["timestamp"] = heth["timestamp"].astype("datetime64[ns]")
    hsig = compute_signals(heth, btc_df=hbtc.rename(columns={"high": "high", "low": "low"}), funding_df=hfunding).reset_index(drop=True)
    hsig["er_24"] = _compute_er24(hsig["close"])
    hlog_ret = np.log(hsig["close"] / hsig["close"].shift(1))
    hsig["realized_vol_ratio"] = hlog_ret.rolling(12).std() / hlog_ret.rolling(288).std()
    tmin = hsig["timestamp"].dt.hour * 60 + hsig["timestamp"].dt.minute
    is_wd = hsig["timestamp"].dt.dayofweek < 5
    hsig["nyse_open_flag"] = (is_wd & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)
    hsig["gbm2_trend_prob"] = np.nan  # skipped for the holdout (OI-derived features unavailable this deep live -- see report caveat)
    hsig["pos"] = hsig.index
    hclose = hsig["close"].to_numpy()
    hn = len(hsig)
    hrows = []
    for name, _desc in SIGNAL_ORDER:
        for side in ("bottom", "top"):
            col = f"{side}_{name}"
            if col not in hsig.columns:
                continue
            fi = np.flatnonzero(hsig[col].fillna(False).to_numpy())
            fi = fi[fi + HORIZON < hn]
            if len(fi) == 0:
                continue
            entry, fwd = hclose[fi], hclose[fi + HORIZON]
            ret = (fwd - entry) / entry
            hit = (ret > 0) if side == "bottom" else (ret < 0)
            pdr = ret * (1 if side == "bottom" else -1)
            hrows.append(pd.DataFrame({"pos": fi, "timestamp": hsig["timestamp"].to_numpy()[fi],
                                        "hit": hit.astype(float), "pred_dir_ret": pdr}))
    hfires = pd.concat(hrows, ignore_index=True).merge(hsig[["pos"] + [c for c in ALL_FEATURES if c != "gbm2_trend_prob"]], on="pos", how="left")
    hfires = hfires[hfires["timestamp"] >= pd.Timestamp("2026-08-20")]
    log(f"  holdout fires (08-20~today): {len(hfires)}")
    holdout_result = evaluate_ladder(hfires, "pred_dir_ret", [s for s in survivors if s != "gbm2_trend_prob"], fit_scaler, fit_model) if len(hfires) >= 10 else {"note": "too few holdout fires"}
    print("HOLDOUT (single touch):", json.dumps(holdout_result, indent=2, default=float))

    report = {
        "status": "exploratory_single_period_below_promotion_bar",
        "status_reason": "docs/model_contracts/evidence_signal_quant_use_contract_20260815.md requires "
                          ">=4 independent chronological periods; this study has exactly one continuous "
                          f"period ({RANGE_START.date()}~{RANGE_END.date()}), bounded by microstructure_1m's "
                          "start (2026-05-03) and the offline evidence-signal source's coverage (2026-08-19). "
                          "That bar is structurally unmeetable with current data -- not a gap to argue around.",
        "window_hours": WINDOW_HOURS, "horizon_bars": HORIZON, "eai_live_threshold": EAI_LIVE_THRESHOLD,
        "train_range": [str(TRAIN_START), str(TRAIN_END)], "val_range": [str(VAL_START), str(VAL_END)],
        "n_fires_total": len(fires), "n_windows_total": len(windows),
        "n_fires_train": len(train_f), "n_fires_val": len(val_f),
        "n_windows_train": len(train_w), "n_windows_val": len(val_w),
        "screen_train": scr.to_dict(orient="records"),
        "dual_granularity_survivors": survivors,
        "ladder_val_fire_level": ladder_val_fire, "ladder_val_window_level": ladder_val_win,
        "ladder_holdout_2026_08_20_onward_single_touch": holdout_result,
        "excluded_by_design": ["liquidation_map(compute_spliced_levels) -- repo docstring already says "
                                "never wire into any promotion/backtest path", "GEX(2 days real history only)",
                                "liq_magnet_history.duckdb(file does not exist, zero rows)",
                                "basis_z48 -- dropped this pass, extra spot-klines fetch not done, already "
                                "documented as exploratory/single-month-validated elsewhere"],
        "notes": "Diagnostic only. Not wired into dashboard/server.py, app.js, or trading_bot.py. "
                 "gbm2_trend_prob omitted from the live holdout (OI-derived FeatureEngineer columns "
                 "not available this deep in a live fetch without the 500-row OI-endpoint cap) -- "
                 "holdout ladder therefore excludes it even if it was a TRAIN/VAL survivor.",
    }
    with open(OUT_DIR / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    windows.to_csv(OUT_DIR / "windows.csv", index=False)
    fires.to_csv(OUT_DIR / "fires.csv", index=False)
    log(f"\nSaved report to {OUT_DIR / 'report.json'}")


if __name__ == "__main__":
    main()
