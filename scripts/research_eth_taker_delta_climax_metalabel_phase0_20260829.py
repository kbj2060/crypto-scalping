"""Phase 0 of meta-labeling for the taker_delta_z_climax evidence signal (1st of the 7 remaining
signals -- liquidity_sweep was already done separately as the V-rebound TabPFN model, see
docs/experiments/eth_liquidity_sweep_v_rebound_feature_plan_20260829.md).

Goal: NOT "is taker_delta_z_climax profitable" (already known to fail the economic cost-gate,
see eth_evidence_signal_short_horizon_economic_gate_rejected_20260824). Goal is meta-labeling --
given that this signal already fires (trigger), can a small model predict WHICH fires will hit
their 1h forward direction call, using only information available at fire time?

Design (per user-approved plan/discussion 2026-08-29):
  - Label: hit = (fwd_1h_ret > 0) if side=="bottom" else (fwd_1h_ret < 0). Verbatim reuse of the
    HORIZON=12 (1h) definition from research_eth_evidence_signal_reliability_gate_20260828.py.
  - Features: klines-only (OHLCV + taker_buy_base), so the FULL 2024-01-01~2026-08-28 history is
    usable (microstructure_1m/tail_risk_1m duckdb only goes back to 2026-05-03 -- too short to
    hit the 4-period bar below). All directional features are SIDE-NORMALIZED (see orient()/
    confluence() below) so bottom+top fires can be pooled into one screen/model with consistent
    sign semantics.
  - Screen: per-feature Spearman IC vs pred_dir_ret, independently within each of 4 chronological
    periods (2024, 2025H1, 2025H2, 2026-partial), significance via the shift_z circular-shift
    permutation test (reused verbatim, not reimplemented). This is the
    docs/model_contracts/evidence_signal_quant_use_contract_20260815.md 4-independent-period
    sign-agreement bar -- the FIRST time this signal family's meta-label work can actually attempt
    it (duckdb-feature attempts structurally can't, only one continuous era since 2026-05-03).
  - Model: Step0 (naive baseline) / Step1 (single best-screened-feature threshold, TRAIN-fit) /
    Step2 (logistic regression on all screen survivors, TRAIN-fit) -- TRAIN = 2024+2025H1+2025H2
    pooled, genuine OOS = 2026, evaluated once. This TRAIN/OOS split answers a DIFFERENT question
    than the 4-period screen above (which deliberately includes 2026 to check sign-stability
    across eras, i.e. replication -- not the same thing as "does a model fit on old data
    generalize forward", which is what Step0/1/2's OOS check answers). Both are reported.

Status label: this is exploratory Phase 0 (screening + a linear baseline), not a promoted model.
Report explicitly states pass/fail against the 4-period bar and the OOS generalization check
separately -- never conflated, never called "confirmed".

Report: tmp/eth_taker_delta_climax_metalabel_phase0_20260829/report.json (NOT data/ensemble/reports/
-- that dir is trading_bot_modules/runtime_config.py's live promotion registry, this is diagnostic).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/kbj20/crypto-scalping")
sys.path.insert(0, "/home/kbj20/crypto-scalping/scripts")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from live_evidence_signal_dashboard_20260823 import compute_signals
from research_eth_model_indicator_volatility_screen_20260825 import shift_z

ROOT = Path("/home/kbj20/crypto-scalping")
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "tmp/eth_taker_delta_climax_metalabel_phase0_20260829"

HORIZON = 12  # 1h forward, matches every prior screen in this lineage

PERIODS = {
    "2024": (pd.Timestamp("2024-01-01"), pd.Timestamp("2025-01-01")),
    "2025H1": (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-07-01")),
    "2025H2": (pd.Timestamp("2025-07-01"), pd.Timestamp("2026-01-01")),
    "2026": (pd.Timestamp("2026-01-01"), pd.Timestamp("2026-12-31")),
}
TRAIN_PERIODS = ["2024", "2025H1", "2025H2"]
OOS_PERIOD = "2026"

FEATURES = [
    "delta_z_mag", "vol_z", "atr_pct", "realized_vol_ratio", "er_24",
    "p_fast_confl", "p_slow_confl", "wick_same_side", "ret3z_aligned", "nyse_open_flag",
]


def log(msg: str) -> None:
    print(f"[taker_delta_climax_phase0] {msg}", flush=True)


def load_klines() -> pd.DataFrame:
    df = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    assert df["timestamp"].diff().dropna().eq(pd.Timedelta(minutes=5)).all(), "gap/dup in klines"
    return df


def build_sig(klines: pd.DataFrame) -> pd.DataFrame:
    sig = compute_signals(klines, btc_df=None, funding_df=None).reset_index(drop=True)
    log_ret = np.log(sig["close"] / sig["close"].shift(1))
    sig["realized_vol_ratio"] = log_ret.rolling(12, min_periods=12).std() / log_ret.rolling(288, min_periods=288).std()
    c = sig["close"]
    net_change_24 = c - c.shift(24)
    diff_abs = c.diff().abs()
    sig["er_24"] = (net_change_24.abs() / (diff_abs.rolling(24, min_periods=4).sum() + 1e-12)).fillna(0.0)
    tmin = sig["timestamp"].dt.hour * 60 + sig["timestamp"].dt.minute
    is_weekday = sig["timestamp"].dt.dayofweek < 5
    sig["nyse_open_flag"] = (is_weekday & (tmin >= 12 * 60 + 30) & (tmin <= 14 * 60 + 30)).astype(int)
    return sig


def build_fires(sig: pd.DataFrame) -> pd.DataFrame:
    """One row per taker_delta_z_climax fire (bottom+top pooled), with hit/pred_dir_ret at
    HORIZON=12, plus side-normalized klines-only features attached."""
    close = sig["close"].to_numpy()
    n = len(sig)
    rows = []
    for side, col in [("bottom", "bottom_taker_delta_z_climax"), ("top", "top_taker_delta_z_climax")]:
        idx = np.flatnonzero(sig[col].fillna(False).to_numpy())
        idx = idx[idx < n - HORIZON]
        fwd_ret = close[idx + HORIZON] / close[idx] - 1.0
        hit = (fwd_ret > 0) if side == "bottom" else (fwd_ret < 0)
        pred_dir_ret = fwd_ret * (1 if side == "bottom" else -1)
        sub = sig.iloc[idx]
        sign = 1.0 if side == "bottom" else -1.0  # orient(): "aligned with fire direction" -> positive
        rows.append(pd.DataFrame({
            "pos": idx, "timestamp": sub["timestamp"].to_numpy(), "side": side,
            "hit": hit.astype(float), "pred_dir_ret": pred_dir_ret,
            "delta_z_mag": sub["delta_z"].abs().to_numpy(),
            "vol_z": sub["vol_z"].to_numpy(),
            "atr_pct": sub["atr_pct"].to_numpy(),
            "realized_vol_ratio": sub["realized_vol_ratio"].to_numpy(),
            "er_24": sub["er_24"].to_numpy(),
            "p_fast_confl": np.where(side == "bottom", 1.0 - sub["p_fast"].to_numpy(), sub["p_fast"].to_numpy()),
            "p_slow_confl": np.where(side == "bottom", 1.0 - sub["p_slow"].to_numpy(), sub["p_slow"].to_numpy()),
            "wick_same_side": (sub["lower_wick_ratio"] if side == "bottom" else sub["upper_wick_ratio"]).to_numpy(),
            "ret3z_aligned": sign * sub["ret3_z"].to_numpy(),
            "nyse_open_flag": sub["nyse_open_flag"].to_numpy(),
        }))
    fires = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    for name, (start, end) in PERIODS.items():
        fires.loc[(fires["timestamp"] >= start) & (fires["timestamp"] < end), "period"] = name
    return fires


def screen(fires: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        sub = fires[fires["period"] == period]
        for feat in FEATURES:
            rho, z, n = shift_z(sub[feat], sub["pred_dir_ret"])
            rows.append({"period": period, "feature": feat, "rho": rho, "shift_z": z, "n": n})
    return pd.DataFrame(rows)


def find_survivors(scr: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Returns (confirmed, promising). confirmed = same sign in all 4 periods AND |shift_z|>=2 in
    >=3/4. promising = same sign in >=3/4 periods but short of the significance bar -- reported for
    honesty, not treated as a candidate for Step2."""
    confirmed, promising = [], []
    for feat in FEATURES:
        rows = scr[scr["feature"] == feat].set_index("period")
        signs = np.sign(rows.loc[list(PERIODS), "rho"].to_numpy())
        sig_count = (rows.loc[list(PERIODS), "shift_z"].abs() >= 2.0).sum()
        if len(set(signs)) == 1 and not np.isnan(signs).any():
            if sig_count >= 3:
                confirmed.append(feat)
            else:
                promising.append(feat)
        elif (signs == signs[0]).sum() >= 3:
            promising.append(feat)
    return confirmed, promising


def fit_step1(train: pd.DataFrame, feat: str) -> float:
    return float(train[feat].median())


def eval_ladder(train: pd.DataFrame, test: pd.DataFrame, confirmed: list[str]) -> dict:
    result = {
        "n_train": int(len(train)), "n_test": int(len(test)),
        "step0_naive_hit_rate": float(train["hit"].mean()),
        "step0_naive_hit_rate_test": float(test["hit"].mean()),
    }
    if not confirmed:
        result["step1"] = "skipped: no confirmed survivors"
        result["step2"] = "skipped: no confirmed survivors"
        return result

    # Step1: single strongest confirmed feature (largest |mean shift_z| across TRAIN periods),
    # TRAIN-fit median split.
    best_feat = confirmed[0]
    thresh = fit_step1(train, best_feat)
    hi_test = test[test[best_feat] >= thresh]
    lo_test = test[test[best_feat] < thresh]
    result["step1_feature"] = best_feat
    result["step1_threshold_train_median"] = thresh
    result["step1_hi_hit_rate_test"] = float(hi_test["hit"].mean()) if len(hi_test) else None
    result["step1_lo_hit_rate_test"] = float(lo_test["hit"].mean()) if len(lo_test) else None
    result["step1_n_hi_test"] = int(len(hi_test))
    result["step1_n_lo_test"] = int(len(lo_test))

    # Step2: logistic regression on ALL confirmed features.
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(train[confirmed].to_numpy())
    ytr = train["hit"].to_numpy().astype(int)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(Xtr, ytr)
    Xte = scaler.transform(test[confirmed].to_numpy())
    yte = test["hit"].to_numpy().astype(int)
    proba = model.predict_proba(Xte)[:, 1]
    result["step2_features"] = confirmed
    result["step2_coef"] = dict(zip(confirmed, model.coef_[0].tolist()))
    try:
        result["step2_auc_test"] = float(roc_auc_score(yte, proba))
    except ValueError:
        result["step2_auc_test"] = None
    tercile_edges = np.quantile(proba, [1 / 3, 2 / 3])
    lo_mask = proba <= tercile_edges[0]
    hi_mask = proba >= tercile_edges[1]
    result["step2_top_tercile_hit_rate_test"] = float(yte[hi_mask].mean()) if hi_mask.sum() else None
    result["step2_bottom_tercile_hit_rate_test"] = float(yte[lo_mask].mean()) if lo_mask.sum() else None
    result["step2_n_top_tercile"] = int(hi_mask.sum())
    result["step2_n_bottom_tercile"] = int(lo_mask.sum())
    return result


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log("loading full-history klines (2023-12-31 ~ 2026-08-28)...")
    klines = load_klines()
    log(f"{len(klines)} bars loaded")

    log("computing compute_signals() + causal klines-only extras over full history...")
    sig = build_sig(klines)

    log("building taker_delta_z_climax fires (bottom+top, HORIZON=12)...")
    fires = build_fires(sig)
    fires = fires.dropna(subset=FEATURES + ["pred_dir_ret"]).reset_index(drop=True)
    log(f"{len(fires)} usable fires after dropna "
        f"(bottom={int((fires['side']=='bottom').sum())}, top={int((fires['side']=='top').sum())})")

    desc = fires.groupby(["period", "side"]).agg(n=("hit", "size"), hit_rate=("hit", "mean")).reset_index()
    log("descriptive hit rates by period x side:\n" + desc.to_string(index=False))

    log("running 4-period screen (shift_z permutation test per feature per period)...")
    scr = screen(fires)
    log("screening table:\n" + scr.to_string(index=False))

    confirmed, promising = find_survivors(scr)
    log(f"confirmed (4/4 sign agreement, sig in >=3/4 periods): {confirmed}")
    log(f"promising (>=3/4 sign agreement, short of significance bar): {promising}")

    train = fires[fires["period"].isin(TRAIN_PERIODS)].reset_index(drop=True)
    test = fires[fires["period"] == OOS_PERIOD].reset_index(drop=True)
    log(f"Step0-2 ladder: TRAIN(2024+2025H1+2025H2) n={len(train)}, OOS(2026) n={len(test)}")
    ladder = eval_ladder(train, test, confirmed)
    log("ladder result:\n" + json.dumps(ladder, indent=2, default=str))

    report = {
        "signal": "taker_delta_z_climax",
        "status": "exploratory_single_signal_phase0_below_promotion_bar",
        "methodology_note": (
            "4-period screen (incl. 2026) tests SIGN STABILITY across eras -- a different question "
            "from Step0-2's TRAIN(2024+2025H1+2025H2)->OOS(2026) generalization check. Screen "
            "'confirmed' features are the ones fed into Step2; Step2's OOS(2026) evaluation never "
            "used 2026 labels/features during fitting, evaluated once (single-touch)."
        ),
        "horizon_bars": HORIZON,
        "n_fires_total": int(len(fires)),
        "descriptive_hit_rates": desc.to_dict(orient="records"),
        "screening_table": scr.to_dict(orient="records"),
        "confirmed_survivors_4period_bar": confirmed,
        "promising_not_confirmed": promising,
        "ladder": ladder,
        "four_period_contract_bar_met": len(confirmed) > 0,
    }
    out_path = OUT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")


if __name__ == "__main__":
    main()
