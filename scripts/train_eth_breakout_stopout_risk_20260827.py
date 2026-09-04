#!/usr/bin/env python3
"""Stream 2 of the chop-fade risk-management redesign (2026-08-27): predict, AT THE MOMENT an
evidence-signal fade entry would trigger, whether that trade is headed for a stop-loss within its
48-bar hold window -- i.e. forecast the outcome Stream 1's rule-based regime_stop was only catching
after the fact (see scripts/backtest_eth_evidence_signal_regime_entry_exit_20260827.py and its
persistence follow-up; best rule-based result there was a modest +1.5~3.5pp loss reduction, nowhere
near flipping the strategy to profitable).

Different from the 2026-05-30 regime3_transition_h6_risk_prob attempt (AUC=0.676, deemed unreliable,
see docs/active_live/regime3_policy_20260530.md) in four ways: (1) horizon is tied to the ACTUAL
hold window (up to 48 bars / 4h), not a fixed 30-minute class-change; (2) label is the ECONOMIC
outcome (would this specific trade hit its ATR-based SL) via core.causal_futures_backtest's own
triple-barrier resolution, not an abstract regime-class flip; (3) only trigger bars where GBM2
already reads chop are labeled -- exactly the bars this strategy would actually act on; (4) features
include the live evidence-signal's own continuous inputs (p_fast/p_slow/delta_z/funding_z for
orthogonal_combo, ret3_z for short_term_return_z), unavailable to the May model since these signals
didn't exist yet.

Leakage guard: the label looks 1-48 bars AHEAD of the trigger bar (standard triple-barrier, same
convention as h48qual and every other barrier label in this repo) -- but every FEATURE is read at or
before the trigger bar only. GBM2's already-vetted 136 feature_cols are reused verbatim (no new
circularity), plus a handful of evidence-signal-native columns that are themselves causal (rolling
windows ending at the trigger bar).

Architecture: HistGradientBoostingClassifier, same family as GBM2/GBM3 and as the 2026-05-30 attempt
-- no evidence in this repo's history that architecture (vs. label/horizon/feature choice) was the
weak link for this kind of tabular financial-outcome prediction, so no new architecture is tried
here without a specific reason to expect it would help.

Honesty note: the 2026-05-30 precedent was weak (AUC 0.676). This may be too. bal_acc/AUC are
reported alongside the only claim that actually matters -- whether using this model's probability
as a filter/sizing input on TOP of Stream 1's backtest reduces losses further than the rule-based
regime_stop alone. That comparison is NOT run in this script (needs the trained model first); it is
the explicit next step once this script's own OOS numbers are known.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from core.causal_futures_backtest import purged_decision_mask  # noqa: E402
from eval_omega4_1_atr_safety_sltp_20260622 import _atr_pct  # noqa: E402
from live_evidence_signal_dashboard_20260823 import compute_signals  # noqa: E402
from research_eth_funding_crossasset_combo_signal_20260825 import load_funding_z  # noqa: E402
from retrain_clean_regime_hmm_raw_state12_20260517 import _with_raw_state12  # noqa: E402
from backtest_eth_evidence_signal_regime_entry_exit_20260827 import (  # noqa: E402
    _resolve_trade_regime_stop, TP_ATR_MULT, SL_ATR_MULT, HORIZON_BARS, ATR_N,
)

GBM2_MODEL_PATH = ROOT / "tmp" / "eth_regime_gbm2_trend_chop_20260827" / "model.joblib"
TRAIN_CSVS = [ROOT / f"data/splits/year_oos/training_features_{y}.csv" for y in ("2024", "2025", "2026_rebuilt")]
BTC_PATH = ROOT / "data" / "btc_5m_1year.csv"
MODEL_ID = "eth_breakout_stopout_risk_20260827"
OUT_DIR = ROOT / f"tmp/{MODEL_ID}"
REPORT_PATH = ROOT / f"data/ensemble/reports/{MODEL_ID}_report.json"

TRAIN_START, TRAIN_END = pd.Timestamp("2024-01-01"), pd.Timestamp("2026-06-30 23:55:00")
OOS_START, OOS_END = pd.Timestamp("2026-07-01"), pd.Timestamp("2026-08-19 23:55:00")

SIGNALS = [
    {"name": "orthogonal_combo", "side": "bottom", "extra_cols": ["p_fast", "p_slow", "delta_z", "funding_z"]},
    {"name": "short_term_return_z", "side": "top", "extra_cols": ["ret3_z"]},
]


def log(msg: str) -> None:
    print(f"[breakout_stopout_risk] {msg}", flush=True)


def _gbm2_trend_prob(raw: pd.DataFrame, feats: pd.DataFrame, payload: dict) -> np.ndarray:
    cols = payload["feature_cols"]
    med = pd.Series(payload["feature_medians"])
    x = feats.copy()
    for c in cols:
        if c not in x.columns:
            x[c] = med.get(c, 0.0)
    x = x[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(med).fillna(0.0)
    proba = payload["model"].predict_proba(x)
    return proba[:, list(payload["classes"]).index("trend")]


def build_frame() -> tuple[pd.DataFrame, list[str]]:
    frames = [pd.read_csv(p, parse_dates=["timestamp"]) for p in TRAIN_CSVS]
    raw = pd.concat(frames, ignore_index=True).sort_values("timestamp").drop_duplicates("timestamp", keep="last").reset_index(drop=True)
    raw = raw[raw["timestamp"] <= OOS_END].reset_index(drop=True)

    feats = _with_raw_state12(raw)
    gbm2_payload = joblib.load(GBM2_MODEL_PATH)
    feature_cols = list(gbm2_payload["feature_cols"])
    trend_prob = _gbm2_trend_prob(raw, feats, gbm2_payload)

    btc = pd.read_csv(BTC_PATH, usecols=["timestamp", "high", "low"], parse_dates=["timestamp"])
    funding = load_funding_z()
    base_cols = ["timestamp", "open", "high", "low", "close", "volume", "taker_buy_base"]
    sig = compute_signals(raw[base_cols].copy(), btc_df=btc, funding_df=funding)

    frame = feats.copy()
    for c in sig.columns:
        if c not in frame.columns:
            frame[c] = sig[c].to_numpy()
    frame["trend_prob"] = trend_prob
    frame["trend_prob_slope_6"] = frame["trend_prob"] - frame["trend_prob"].shift(6)
    frame["is_trend_raw"] = (trend_prob >= 0.5).astype(int)
    frame["atr_pct"] = pd.Series(_atr_pct(raw, ATR_N), index=raw.index)
    return frame, feature_cols


def build_label_and_features(frame: pd.DataFrame, feature_cols: list[str], signal: dict) -> pd.DataFrame:
    name, side, extra_cols = signal["name"], signal["side"], signal["extra_cols"]
    trig_col = f"{'bottom' if side == 'bottom' else 'top'}_{name}"
    ts = frame["timestamp"]

    eligible = purged_decision_mask(ts, start=TRAIN_START, end=OOS_END + pd.Timedelta(minutes=5), horizon_bars=HORIZON_BARS)
    triggered = frame[trig_col].fillna(False).to_numpy()
    chop_now = (frame["is_trend_raw"].to_numpy() == 0)
    has_atr = frame["atr_pct"].notna().to_numpy()
    trigger_idx = np.flatnonzero(eligible & triggered & chop_now & has_atr)
    log(f"  {name} ({side}): {len(trigger_idx)} chop-gated trigger bars in [{TRAIN_START.date()}, {OOS_END.date()}]")

    open_v, high_v, low_v, close_v = (frame[c].to_numpy() for c in ("open", "high", "low", "close"))
    trend_prob_v, atr_v = frame["trend_prob"].to_numpy(), frame["atr_pct"].to_numpy()
    side_sign = 1 if side == "bottom" else -1

    rows = []
    for ti in trigger_idx:
        entry_i = int(ti) + 1
        final_i = min(entry_i + HORIZON_BARS - 1, len(ts) - 1)
        if final_i < entry_i or entry_i >= len(ts):
            continue
        entry = float(open_v[entry_i])
        tp_move, sl_move = TP_ATR_MULT * atr_v[ti], SL_ATR_MULT * atr_v[ti]
        if not (np.isfinite(tp_move) and np.isfinite(sl_move)):
            continue
        _move, reason, _off, _mae = _resolve_trade_regime_stop(
            side=side_sign, entry=entry, high=high_v[entry_i:final_i+1], low=low_v[entry_i:final_i+1],
            close=close_v[entry_i:final_i+1], tp_move=tp_move, sl_move=sl_move,
            trend_prob=trend_prob_v[entry_i:final_i+1], theta_exit=None,
        )
        row = {"trigger_i": int(ti), "timestamp": ts.iloc[ti], "label": int(reason == "sl")}
        for c in feature_cols:
            row[c] = frame[c].iloc[ti]
        for c in extra_cols + ["trend_prob", "trend_prob_slope_6"]:
            row[c] = frame[c].iloc[ti]
        rows.append(row)
    return pd.DataFrame(rows)


def train_one(name: str, df: pd.DataFrame, feature_cols: list[str], extra_cols: list[str]) -> dict[str, Any]:
    all_feature_cols = feature_cols + extra_cols + ["trend_prob", "trend_prob_slope_6"]
    x = df[all_feature_cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    medians = x[df["timestamp"] <= TRAIN_END].median(numeric_only=True).fillna(0.0)
    x = x.fillna(medians).fillna(0.0)
    y = df["label"].to_numpy()

    train_mask = (df["timestamp"] <= TRAIN_END).to_numpy()
    oos_mask = ((df["timestamp"] >= OOS_START) & (df["timestamp"] <= OOS_END)).to_numpy()
    log(f"  {name}: train n={train_mask.sum()} (base_rate={y[train_mask].mean():.3f})  "
        f"oos n={oos_mask.sum()} (base_rate={y[oos_mask].mean() if oos_mask.sum() else float('nan'):.3f})")

    if train_mask.sum() < 50 or oos_mask.sum() < 10:
        log(f"  {name}: SKIPPED, insufficient samples")
        return {"skipped": True, "n_train": int(train_mask.sum()), "n_oos": int(oos_mask.sum())}

    # Real OOS (2026-07~08) is tiny (n=39/112) -- too small alone to tell "no signal" from "unlucky
    # sample." Cross-check on a much larger chronological holdout carved out of TRAIN itself (last
    # 20% of TRAIN rows by time, fit on the first 80%) before trusting either number in isolation.
    train_idx = np.flatnonzero(train_mask)
    split_pt = train_idx[int(len(train_idx) * 0.8)]
    fit_mask = train_mask & (df["timestamp"] < df["timestamp"].iloc[split_pt]).to_numpy()
    holdout_mask = train_mask & ~fit_mask
    holdout_metrics = None
    if fit_mask.sum() >= 50 and holdout_mask.sum() >= 30 and len(set(y[holdout_mask])) > 1:
        probe = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=7529)
        probe.fit(x[fit_mask], y[fit_mask])
        proba_h = probe.predict_proba(x[holdout_mask])[:, 1]
        pred_h = (proba_h >= 0.5).astype(int)
        y_h = y[holdout_mask]
        holdout_metrics = {
            "n_fit": int(fit_mask.sum()), "n_holdout": int(holdout_mask.sum()),
            "holdout_base_rate": float(y_h.mean()),
            "holdout_balanced_accuracy": float(balanced_accuracy_score(y_h, pred_h)),
            "holdout_auc": float(roc_auc_score(y_h, proba_h)),
        }
        log(f"  {name} internal holdout (larger sample, chronological, still pre-OOS): {holdout_metrics}")

    model = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.05, max_iter=300, random_state=7529)
    model.fit(x[train_mask], y[train_mask])
    assert (model.classes_ == np.array([0, 1])).all()

    proba_oos = model.predict_proba(x[oos_mask])[:, 1]
    pred_oos = (proba_oos >= 0.5).astype(int)
    y_oos = y[oos_mask]
    metrics = {
        "n_train": int(train_mask.sum()), "n_oos": int(oos_mask.sum()),
        "train_base_rate": float(y[train_mask].mean()), "oos_base_rate": float(y_oos.mean()),
        "oos_balanced_accuracy": float(balanced_accuracy_score(y_oos, pred_oos)) if len(set(y_oos)) > 1 else None,
        "oos_auc": float(roc_auc_score(y_oos, proba_oos)) if len(set(y_oos)) > 1 else None,
        "oos_confusion": confusion_matrix(y_oos, pred_oos, labels=[0, 1]).tolist(),
        "internal_holdout": holdout_metrics,
    }
    log(f"  {name} OOS: bal_acc={metrics['oos_balanced_accuracy']}  auc={metrics['oos_auc']}")

    try:
        from sklearn.inspection import permutation_importance
        perm = permutation_importance(model, x[oos_mask], y_oos, n_repeats=5, random_state=7529, n_jobs=-1)
        top10 = sorted(zip(all_feature_cols, perm.importances_mean), key=lambda t: -t[1])[:10]
        metrics["top10_permutation_importance_oos"] = [(n, float(v)) for n, v in top10]
        log(f"  {name} top10 importance: {top10}")
    except Exception as e:  # noqa: BLE001
        metrics["top10_permutation_importance_oos"] = f"failed: {e}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": f"{MODEL_ID}_{name}", "model": model, "feature_cols": all_feature_cols,
        "feature_medians": medians.to_dict(), "signal": name, "metrics": metrics,
        "train_range": f"{TRAIN_START} ~ {TRAIN_END}", "oos_range": f"{OOS_START} ~ {OOS_END}",
        "label_logic": "1 if a trade entered at this chop-gated trigger bar (TP=1.6xATR/SL=1.0xATR/"
                        "48bar horizon, core.causal_futures_backtest triple-barrier convention) would "
                        "hit SL, else 0 (tp/timeout).",
    }
    joblib.dump(payload, OUT_DIR / f"model_{name}.joblib")
    return metrics


def main() -> int:
    log("Building shared frame (features + GBM2 trend_prob + evidence signals)...")
    frame, feature_cols = build_frame()
    log(f"  frame rows: {len(frame)}, range {frame['timestamp'].min()} ~ {frame['timestamp'].max()}")

    report: dict[str, Any] = {}
    for signal in SIGNALS:
        name = signal["name"]
        log(f"\n=== {name} ({signal['side']}) ===")
        df = build_label_and_features(frame, feature_cols, signal)
        if len(df) < 50:
            log(f"  {name}: too few trigger bars ({len(df)}), skipping training")
            report[name] = {"skipped": True, "n_triggers": len(df)}
            continue
        report[name] = train_one(name, df, feature_cols, signal["extra_cols"])
        report[name]["n_triggers_total"] = len(df)

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))
    log(f"\nWrote {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
