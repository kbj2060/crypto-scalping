from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_TRAIN_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified.csv"
DEFAULT_TEST_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2026_unified.csv"
DEFAULT_MODEL = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_catboost.pkl"
DEFAULT_RULE = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_rule.json"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_2026_backtest.json"
FEE = 0.0005
SLIP = 0.0002


@dataclass
class BacktestResult:
    pnl_pct: float
    trades: int
    wr_pct: float
    mdd_pct: float
    longs: int
    shorts: int


def _regime_name(df: pd.DataFrame) -> np.ndarray:
    return (
        df[["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"]]
        .idxmax(axis=1)
        .str.replace("regime_", "", regex=False)
        .to_numpy()
    )


def _sup_side(df: pd.DataFrame) -> np.ndarray:
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df["ud_cat_flat_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.where((lp >= sp) & (lp >= fp), 1, np.where((sp > lp) & (sp >= fp), -1, 0)).astype(np.int8)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return df


def _predict_direction_probs(test_df: pd.DataFrame, train_df: pd.DataFrame, model_payload: dict[str, Any]) -> pd.DataFrame:
    feature_cols = list(model_payload["feature_cols"])
    model = model_payload["model"]
    missing = [c for c in feature_cols if c not in test_df.columns or c not in train_df.columns]
    if missing:
        raise ValueError(f"missing features for direction model: {missing}")

    x_train = train_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    x_test = test_df.loc[:, feature_cols].replace([np.inf, -np.inf], np.nan).copy()
    med = x_train.median(numeric_only=True)
    x_train = x_train.fillna(med).fillna(0.0)
    x_test = x_test.fillna(med).fillna(0.0)

    probs = model.predict_proba(x_test)
    out = test_df.copy()
    out["ud_cat_short_prob"] = probs[:, 0]
    out["ud_cat_flat_prob"] = probs[:, 1]
    out["ud_cat_long_prob"] = probs[:, 2]
    out["ud_cat_edge"] = out["ud_cat_long_prob"] - out["ud_cat_short_prob"]
    out["ud_cat_prob_max"] = np.max(probs, axis=1)
    out["ud_cat_pred_class"] = np.argmax(probs, axis=1)
    return out


def _build_candidates(df: pd.DataFrame, p: dict[str, Any]) -> pd.DataFrame:
    out = df.copy()
    raw_side = np.sign(pd.to_numeric(out["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)
    sup_side = _sup_side(out)
    q = pd.to_numeric(out["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_edge = (
        pd.to_numeric(out["m7_prob_up"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["m7_prob_dn"], errors="coerce").fillna(0.0)
    ).abs().to_numpy(np.float64)
    sup_prob_max = (
        out[["ud_cat_long_prob", "ud_cat_short_prob", "ud_cat_flat_prob"]]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .max(axis=1)
        .to_numpy(np.float64)
    )
    hold_pred = (
        pd.to_numeric(out["m7_hold_pred"], errors="coerce")
        .fillna(6.0)
        .clip(4.0, 8.0)
        .round()
        .astype(np.int32)
        .to_numpy()
    )
    raw_change = np.r_[True, raw_side[1:] != raw_side[:-1]]
    agree = (raw_side == sup_side) & (sup_side != 0)
    regimes = _regime_name(out)

    candidate = np.zeros(len(out), dtype=np.int8)
    side = np.zeros(len(out), dtype=np.int8)
    last_idx = -10**9
    for i in range(len(out)):
        if sup_side[i] == 0:
            continue
        if q[i] < p["quality_min"] or raw_edge[i] < p["raw_edge_min"] or sup_prob_max[i] < p["sup_prob_min"]:
            continue
        if p.get("require_agreement", False) and not agree[i]:
            continue
        if p.get("sign_change_only", False) and not raw_change[i]:
            continue
        if i - last_idx < int(p["debounce_bars"]):
            continue
        candidate[i] = 1
        side[i] = sup_side[i]
        last_idx = i

    out["ud_cand_flag"] = candidate
    out["ud_cand_side"] = side
    out["ud_cand_hold"] = hold_pred
    out["ud_cand_quality"] = q
    out["ud_cand_raw_edge"] = raw_edge
    out["ud_cand_sup_prob_max"] = sup_prob_max
    out["ud_cand_regime"] = regimes
    return out


def _run(df: pd.DataFrame, rule: dict[str, str], hold_scale: float, close_on_opp: bool) -> BacktestResult:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    regimes = df["ud_cand_regime"].astype(str).to_numpy()

    pos = 0
    entry = 0.0
    hold = 0
    target_hold = 0
    balance = 1.0
    peak = 1.0
    mdd = 0.0
    trades = 0
    wins = 0
    longs = 0
    shorts = 0
    for i in range(len(df)):
        allowed = False
        if cand[i] == 1 and side[i] != 0:
            mode = rule[regimes[i]]
            allowed = (
                mode == "both"
                or (mode == "long" and side[i] == 1)
                or (mode == "short" and side[i] == -1)
            )
        if pos == 0:
            if allowed:
                pos = int(side[i])
                entry = close[i] * (1.0 + SLIP) if pos == 1 else close[i] * (1.0 - SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
                longs += int(pos == 1)
                shorts += int(pos == -1)
        else:
            hold += 1
            reverse = allowed and side[i] == -pos and close_on_opp
            if reverse or hold >= target_hold:
                fill = close[i] * (1.0 - SLIP) if pos == 1 else close[i] * (1.0 + SLIP)
                pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                pos = 0
                entry = 0.0
                hold = 0
                target_hold = 0
        peak = max(peak, balance)
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)
    if pos != 0:
        fill = close[-1] * (1.0 - SLIP) if pos == 1 else close[-1] * (1.0 + SLIP)
        pnl = ((fill - entry) / entry) if pos == 1 else ((entry - fill) / entry)
        balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
        trades += 1
        wins += int(pnl > 0)
    return BacktestResult(
        pnl_pct=float((balance - 1.0) * 100.0),
        trades=int(trades),
        wr_pct=float(wins / max(trades, 1) * 100.0),
        mdd_pct=float(mdd * 100.0),
        longs=int(longs),
        shorts=int(shorts),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Backtest sparse handcrafted unified shadow rule on 2026 data")
    ap.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--model-path", default=DEFAULT_MODEL)
    ap.add_argument("--rule-path", default=DEFAULT_RULE)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    train_df = _load_frame(args.train_csv)
    test_df = _load_frame(args.test_csv)
    model_payload = _load_pickle(args.model_path)
    with open(args.rule_path, "r", encoding="utf-8") as f:
        rule_cfg = json.load(f)

    scored = _predict_direction_probs(test_df, train_df, model_payload)
    cand = _build_candidates(scored, rule_cfg["candidate_params"])
    res = _run(cand, rule_cfg["veto_rule"], float(rule_cfg["hold_scale"]), bool(rule_cfg["close_on_opp"]))
    out = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "model_path": args.model_path,
        "rule_path": args.rule_path,
        "candidate_params": rule_cfg["candidate_params"],
        "veto_rule": rule_cfg["veto_rule"],
        "hold_scale": rule_cfg["hold_scale"],
        "close_on_opp": rule_cfg["close_on_opp"],
        "candidate_rows": int(pd.to_numeric(cand["ud_cand_flag"], errors="coerce").fillna(0).sum()),
        "take_rows": int(
            (
                (pd.to_numeric(cand["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8) == 1)
                & (
                    ((cand["ud_cand_regime"] == "bull") & False)
                    | ((cand["ud_cand_regime"] == "bear") & (pd.to_numeric(cand["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8) == -1))
                    | ((cand["ud_cand_regime"] == "chop") & False)
                    | ((cand["ud_cand_regime"] == "whipsaw") & (pd.to_numeric(cand["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8) == 1))
                    | ((cand["ud_cand_regime"] == "normal") & (pd.to_numeric(cand["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8) == -1))
                )
            ).sum()
        ),
        "backtest": asdict(res),
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
