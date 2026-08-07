#!/usr/bin/env python3
"""Stage-2 prototype: direction head conditioned on the stage-1 event gate.

Stage 1 (research_btc_event_gate_prototype_20260804.py) showed the GMM+IF online-
recalibrated gate fires on ~1% of bars with a stable ~3.3x lift on the volatility-
aware extreme-event label across VAL and OOS. That gate says "something big is
about to happen" but not which direction. This script asks a narrower question than
the one already closed in project-btc-cusum-architecture-structural-redesign-closed-
20260804 ("does causalfix_final's 114-col set predict direction on EVERY bar?" -> no,
confirmed 0/9 across 4 classifier families): does a small, causal, gate-conditioned
feature set predict direction ONLY on the ~1% of bars the stage-1 gate already fired
on? This is a materially different conditional distribution, so the prior closure
does not automatically apply here -- but it also has not been tested before.

This is a MoE-style routing prototype, not a promotion claim: single seed, no
seed-averaging, so the Seed-Diversity Ensemble Promotion Gate in CLAUDE.md does not
apply yet. If this direction head shows real out-of-sample lift, a seed-diversity
check becomes required before any live/promotion claim.

Fresh-Forward compliance: fresh_forward_bar_by_bar=true, trade_ledgers_used_as_input=false,
saved_parent_exit_timestamps_used=false, future_rows_used_for_entry=false. The model is
fit ONLY on TRAIN (< 2025-09-01, strictly before VAL) and evaluated causally forward on
VAL/OOS with no refitting. All input features are causal (rolling/shift, no future data);
the direction LABEL is necessarily forward-looking (bars i+1..i+EVENT_HORIZON) but is used
only as a training/evaluation target, never fed back as a feature.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import research_btc_event_gate_prototype_20260804 as gate_mod  # noqa: E402

OUT_DIR = ROOT / "tmp/research_20260804/btc_direction_head_prototype"

TRAIN_END = gate_mod.VAL_START  # strictly before VAL, no overlap
MOM_WINDOWS = {"mom_1h": 12, "mom_4h": 48, "mom_24h": 288}
LOOKBACK_HL = 288  # 24h, for breakout/mean-reversion distance features
CONFIDENCE_MARGIN = 0.15  # |P(long) - 0.5| threshold for "confident" predictions


def _direction_label(frame: pd.DataFrame, horizon: int) -> pd.Series:
    """+1 if the larger realized excursion over i+1..i+horizon is up, -1 if down,
    0 if the window is incomplete (near the end of the frame)."""
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    entry = frame["close"].to_numpy()
    n = len(frame)
    direction = np.zeros(n)
    for i in range(n - horizon - 1):
        e = entry[i]
        if e <= 0:
            continue
        fut_high = high[i + 1 : i + 1 + horizon]
        fut_low = low[i + 1 : i + 1 + horizon]
        up_move = fut_high.max() / e - 1.0
        down_move = 1.0 - fut_low.min() / e
        direction[i] = 1.0 if up_move > down_move else -1.0
    return pd.Series(direction, index=frame.index)


def _causal_features(df: pd.DataFrame, atr: pd.Series) -> pd.DataFrame:
    feats = pd.DataFrame(index=df.index)
    close = df["close"]
    for name, w in MOM_WINDOWS.items():
        feats[name] = close.pct_change(w)
    feats["atr"] = atr
    feats["atr_chg_4h"] = atr.pct_change(48)
    feats["dist_from_high_24h"] = close / df["high"].rolling(LOOKBACK_HL, min_periods=LOOKBACK_HL // 2).max() - 1.0
    feats["dist_from_low_24h"] = close / df["low"].rolling(LOOKBACK_HL, min_periods=LOOKBACK_HL // 2).min() - 1.0
    feats["gmm_cluster_rank"] = df["gmm_cluster_rank"]
    feats["gmm_confidence"] = df["gmm_confidence"]
    feats["if_score"] = df["if_score"]
    feats["raw_score"] = df["raw_score"]
    feats["agreement"] = df["agreement"]
    return feats.replace([np.inf, -np.inf], np.nan)


def _rebuild_gated_frame() -> pd.DataFrame:
    """Re-run the stage-1 gate pipeline in-process (cheap; ~166k rows) so this
    script stays self-contained and always matches the current gate config."""
    ohlc = gate_mod._load_ohlc()
    gmm = pd.read_csv(gate_mod.GMM_SCORES, usecols=["timestamp", "gmm_cluster_rank", "gmm_confidence"])
    ifs = pd.read_csv(gate_mod.IF_SCORES, usecols=["timestamp", "if_score"])
    gmm["timestamp"] = pd.to_datetime(gmm["timestamp"])
    ifs["timestamp"] = pd.to_datetime(ifs["timestamp"])
    df = ohlc.merge(gmm, on="timestamp", how="inner").merge(ifs, on="timestamp", how="inner")
    df = df.sort_values("timestamp").reset_index(drop=True)

    atr = gate_mod._causal_atr(df)
    gate = gate_mod._multi_timescale_gate(df["gmm_cluster_rank"], df["gmm_confidence"], df["if_score"])
    df = pd.concat([df, gate], axis=1)
    df["threshold"] = gate_mod._online_conformal_threshold(df["raw_score"])
    df["gate_fired"] = (df["raw_score"] >= df["threshold"]) & (df["agreement"] >= gate_mod.AGREEMENT_MIN)
    df["direction_label"] = _direction_label(df, gate_mod.EVENT_HORIZON)
    feats = _causal_features(df, atr)
    return pd.concat([df[["timestamp", "gate_fired", "direction_label"]], feats], axis=1)


def _cluster_dedupe(df_fired: pd.DataFrame, gap: str = "4h") -> pd.DataFrame:
    """Keep only the first bar of each firing cluster so evaluation counts
    independent decisions, not autocorrelated bars within one overlapping event."""
    ts = df_fired["timestamp"]
    new_cluster = ts.diff() > pd.Timedelta(gap)
    new_cluster.iloc[0] = True
    return df_fired[new_cluster.to_numpy()]


def _eval_split(model, scaler, feat_cols, df_split: pd.DataFrame, baseline_sign: pd.Series) -> dict:
    fired = df_split[df_split["gate_fired"] & (df_split["direction_label"] != 0.0)].dropna(subset=feat_cols)
    fired = _cluster_dedupe(fired)
    if fired.empty:
        return {"n_events": 0}
    X = scaler.transform(fired[feat_cols])
    proba_long = model.predict_proba(X)[:, 1]
    pred = np.where(proba_long >= 0.5, 1.0, -1.0)
    y = fired["direction_label"].to_numpy()

    acc_all = float((pred == y).mean())
    confident_mask = np.abs(proba_long - 0.5) >= CONFIDENCE_MARGIN
    n_confident = int(confident_mask.sum())
    acc_confident = float((pred[confident_mask] == y[confident_mask]).mean()) if n_confident else float("nan")

    base_pred = baseline_sign.loc[fired.index].to_numpy()
    base_acc = float((base_pred == y).mean())

    return {
        "n_events": int(len(fired)),
        "accuracy_all_fired": acc_all,
        "n_confident": n_confident,
        "accuracy_confident": acc_confident,
        "confident_coverage": float(n_confident / len(fired)),
        "recent_momentum_baseline_accuracy": base_acc,
        "long_rate_in_labels": float((y == 1.0).mean()),
    }


def main() -> int:
    df = _rebuild_gated_frame()
    feat_cols = [c for c in df.columns if c not in ("timestamp", "gate_fired", "direction_label")]

    train = df[(df["timestamp"] < TRAIN_END) & df["gate_fired"] & (df["direction_label"] != 0.0)].dropna(subset=feat_cols)
    train = _cluster_dedupe(train)

    scaler = StandardScaler().fit(train[feat_cols])
    X_train = scaler.transform(train[feat_cols])
    y_train = (train["direction_label"] == 1.0).astype(int)
    model = LogisticRegression(max_iter=1000, C=0.5).fit(X_train, y_train)

    # naive baseline: predict same direction as the most recent 1h momentum sign
    baseline_sign = np.sign(df["mom_1h"]).replace(0.0, 1.0)

    result = {
        "fresh_forward_bar_by_bar": True,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "config": {
            "train_end_exclusive": str(TRAIN_END.date()),
            "confidence_margin": CONFIDENCE_MARGIN,
            "n_train_events_deduped": int(len(train)),
            "train_long_rate": float(y_train.mean()),
            "feature_cols": feat_cols,
        },
        "train_fit_accuracy": float(model.score(X_train, y_train)),
        "validation_2025_09_to_12": _eval_split(model, scaler, feat_cols, df[(df.timestamp >= gate_mod.VAL_START) & (df.timestamp <= gate_mod.VAL_END)], baseline_sign),
        "oos_2026_01_to_03": _eval_split(model, scaler, feat_cols, df[(df.timestamp >= gate_mod.OOS_START) & (df.timestamp <= gate_mod.OOS_END)], baseline_sign),
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "direction_head_eval_result.json", "w") as f:
        json.dump(result, f, indent=2)

    coefs = dict(zip(feat_cols, model.coef_[0].tolist()))
    with open(OUT_DIR / "direction_head_coefs.json", "w") as f:
        json.dump(coefs, f, indent=2)

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
