from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.build_unified_sparse_candidates_v2 import build_candidates


DEFAULT_TRAIN_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_TEST_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2026_unified.csv"
DEFAULT_DIR_MODEL = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_direction_catboost.pkl"
DEFAULT_SPECIALISTS = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_regime_specialists.json"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_long_only_specialists_eval.json"
FEE = 0.0005
SLIP = 0.0002
ACTIVE_KEYS = ["bull_long", "whipsaw_long", "normal_long"]
CANDIDATE_PARAM_GRID = [
    {"quality_min": 0.005, "raw_edge_min": 0.45, "sup_prob_min": 0.75, "debounce_bars": 6, "sign_change_only": True, "require_agreement": False},
    {"quality_min": 0.005, "raw_edge_min": 0.55, "sup_prob_min": 0.80, "debounce_bars": 8, "sign_change_only": True, "require_agreement": False},
    {"quality_min": 0.006, "raw_edge_min": 0.55, "sup_prob_min": 0.85, "debounce_bars": 8, "sign_change_only": True, "require_agreement": False},
    {"quality_min": 0.007, "raw_edge_min": 0.65, "sup_prob_min": 0.90, "debounce_bars": 12, "sign_change_only": True, "require_agreement": False},
]
THRESH_GRID = {
    "bull_long": [0.50, 0.55, 0.60],
    "whipsaw_long": [0.50, 0.55, 0.60],
    "normal_long": [0.55, 0.60, 0.65],
}
HOLD_GRID = {
    "bull": [0.6, 0.8],
    "whipsaw": [0.6, 0.8, 1.0],
    "normal": [0.6, 0.8, 1.0],
}
EXIT_MIN_HOLD_GRID = [1, 2]
EXIT_PROB_MARGIN_GRID = [0.00, 0.05]
EXIT_QUALITY_GRID = [None, 0.0]


@dataclass
class BacktestResult:
    pnl_pct: float
    trades: int
    wr_pct: float
    mdd_pct: float
    longs: int
    shorts: int


def _safe_fill(df: pd.DataFrame) -> pd.DataFrame:
    out = df.replace([np.inf, -np.inf], np.nan).copy()
    for c in out.columns:
        if out[c].dtype.kind in "biufc":
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _load_frame(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
    return _safe_fill(df)


def _load_pickle(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _predict_direction_probs(test_df: pd.DataFrame, train_df: pd.DataFrame, model_payload: dict[str, Any]) -> pd.DataFrame:
    feature_cols = list(model_payload["feature_cols"])
    model = model_payload["model"]
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


def _apply_specialists(df: pd.DataFrame, spec_meta: dict[str, Any], model_root: str) -> pd.DataFrame:
    out = df.copy()
    prob = np.zeros(len(out), dtype=np.float64)
    key_arr = np.full(len(out), "", dtype=object)
    for key in ACTIVE_KEYS:
        meta = spec_meta["specialists"].get(key, {})
        if not meta.get("trained"):
            continue
        regime, side_name = key.split("_", 1)
        side_val = 1 if side_name == "long" else -1
        mask = (
            (out["ud2_cand_flag"].fillna(0).astype(np.int8) == 1)
            & (out["ud2_cand_regime"].astype(str) == regime)
            & (pd.to_numeric(out["ud2_cand_side"], errors="coerce").fillna(0).astype(np.int8) == side_val)
        )
        if not mask.any():
            continue
        payload = _load_pickle(os.path.join(model_root, meta["model_path"]))
        feats = payload["feature_cols"]
        med = payload["median"]
        x = out.loc[mask, feats].copy()
        x = x.fillna(pd.Series(med)).fillna(0.0)
        p = payload["model"].predict_proba(x)[:, 1]
        prob[mask.to_numpy()] = p
        key_arr[mask.to_numpy()] = key
    out["ud2_spec_key"] = key_arr
    out["ud2_spec_prob"] = prob
    return out


def _run(
    df: pd.DataFrame,
    thresholds: dict[str, float],
    hold_scale_by_regime: dict[str, float],
    exit_min_hold_bars: int,
    exit_prob_flip_margin: float,
    exit_quality_floor: float | None,
) -> BacktestResult:
    close = pd.to_numeric(df["close"], errors="coerce").ffill().bfill().to_numpy(np.float64)
    cand = pd.to_numeric(df["ud2_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    side = pd.to_numeric(df["ud2_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy()
    hold_arr = pd.to_numeric(df["ud2_cand_hold"], errors="coerce").fillna(6).astype(np.int32).to_numpy()
    regime_arr = df["ud2_cand_regime"].astype(str).to_numpy()
    spec_key = df["ud2_spec_key"].astype(str).to_numpy()
    spec_prob = pd.to_numeric(df["ud2_spec_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    q = pd.to_numeric(df["m7_target_quality"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    raw_side = np.sign(pd.to_numeric(df["m7_action"], errors="coerce").fillna(0.0).to_numpy(np.float64)).astype(np.int8)

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
    current_regime = ""

    for i in range(len(df)):
        allowed = False
        if cand[i] == 1 and side[i] == 1 and spec_key[i] in thresholds:
            allowed = spec_prob[i] >= float(thresholds[spec_key[i]])
        if pos == 0:
            if allowed:
                pos = 1
                entry = close[i] * (1.0 + SLIP)
                balance *= (1.0 - FEE)
                hold = 0
                current_regime = regime_arr[i]
                hold_scale = float(hold_scale_by_regime.get(current_regime, 0.8))
                target_hold = max(2, int(round(float(hold_arr[i]) * hold_scale)))
                longs += 1
        else:
            hold += 1
            prob_flip = sp[i] >= (lp[i] + float(exit_prob_flip_margin))
            raw_flip = raw_side[i] < 0
            quality_break = exit_quality_floor is not None and q[i] <= float(exit_quality_floor)
            early_exit = hold >= int(exit_min_hold_bars) and (prob_flip or raw_flip or quality_break)
            if early_exit or hold >= target_hold:
                fill = close[i] * (1.0 - SLIP)
                pnl = (fill - entry) / max(entry, 1e-8)
                balance *= max(1e-8, (1.0 + pnl) * (1.0 - FEE))
                trades += 1
                wins += int(pnl > 0)
                pos = 0
                entry = 0.0
                hold = 0
                target_hold = 0
                current_regime = ""
        peak = max(peak, balance)
        mdd = min(mdd, balance / max(peak, 1e-8) - 1.0)

    if pos != 0:
        fill = close[-1] * (1.0 - SLIP)
        pnl = (fill - entry) / max(entry, 1e-8)
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


def _score(res: BacktestResult) -> float:
    score = float(res.pnl_pct) - 0.35 * abs(min(float(res.mdd_pct), 0.0))
    if int(res.trades) < 12:
        score -= 6.0
    if int(res.trades) > 80:
        score -= 0.06 * (int(res.trades) - 80)
    return float(score)


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate long-only regime specialists with regime-specific hold and deterministic exits")
    ap.add_argument("--train-csv", default=DEFAULT_TRAIN_CSV)
    ap.add_argument("--test-csv", default=DEFAULT_TEST_CSV)
    ap.add_argument("--dir-model", default=DEFAULT_DIR_MODEL)
    ap.add_argument("--specialists", default=DEFAULT_SPECIALISTS)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    train_df = _load_frame(args.train_csv)
    test_df = _load_frame(args.test_csv)
    dir_payload = _load_pickle(args.dir_model)
    with open(args.specialists, "r", encoding="utf-8") as f:
        spec_meta = json.load(f)
    model_root = os.path.dirname(args.specialists)

    calib_windows: list[tuple[int, int, int]] = []
    cache: dict[int, pd.DataFrame] = {}
    best: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []

    for p_idx, params in enumerate(CANDIDATE_PARAM_GRID):
        calib_scored = build_candidates(train_df, params)
        calib_scored = _apply_specialists(calib_scored, spec_meta, model_root)
        calib = calib_scored[calib_scored["ud_cat_is_holdout"] == 0].copy().reset_index(drop=True)
        holdout = calib_scored[calib_scored["ud_cat_is_holdout"] == 1].copy().reset_index(drop=True)
        cache[p_idx] = holdout

        fold_col = pd.to_numeric(calib["ud_cat_oof_fold"], errors="coerce").fillna(-1).astype(np.int32).to_numpy()
        recent_folds = sorted(int(x) for x in np.unique(fold_col) if x >= 0)[-4:]
        windows = []
        for fid in recent_folds:
            idx = np.flatnonzero(fold_col == fid)
            if len(idx):
                windows.append((int(idx[0]), int(idx[-1] + 1), fid))
        if not calib_windows:
            calib_windows = windows

        for bull_thr, whip_thr, norm_thr in itertools.product(
            THRESH_GRID["bull_long"], THRESH_GRID["whipsaw_long"], THRESH_GRID["normal_long"]
        ):
            thresholds = {
                "bull_long": float(bull_thr),
                "whipsaw_long": float(whip_thr),
                "normal_long": float(norm_thr),
            }
            for bull_hold, whip_hold, norm_hold in itertools.product(
                HOLD_GRID["bull"], HOLD_GRID["whipsaw"], HOLD_GRID["normal"]
            ):
                hold_map = {
                    "bull": float(bull_hold),
                    "whipsaw": float(whip_hold),
                    "normal": float(norm_hold),
                }
                for min_hold, prob_margin, quality_floor in itertools.product(
                    EXIT_MIN_HOLD_GRID, EXIT_PROB_MARGIN_GRID, EXIT_QUALITY_GRID
                ):
                    per = []
                    scores = []
                    for ws, we, fid in windows:
                        res = _run(
                            calib.iloc[ws:we].reset_index(drop=True),
                            thresholds=thresholds,
                            hold_scale_by_regime=hold_map,
                            exit_min_hold_bars=int(min_hold),
                            exit_prob_flip_margin=float(prob_margin),
                            exit_quality_floor=quality_floor,
                        )
                        sc = _score(res)
                        per.append({"fold": fid, "start": ws, "end": we, "result": asdict(res), "score": sc})
                        scores.append(sc)
                    row = {
                        "candidate_params": params,
                        "thresholds": thresholds,
                        "hold_scale_by_regime": hold_map,
                        "exit_min_hold_bars": int(min_hold),
                        "exit_prob_flip_margin": float(prob_margin),
                        "exit_quality_floor": quality_floor,
                        "avg_score": float(np.mean(scores)),
                        "avg_pnl": float(np.mean([w["result"]["pnl_pct"] for w in per])),
                        "avg_trades": float(np.mean([w["result"]["trades"] for w in per])),
                        "windows": per,
                    }
                    rows.append(row)
                    if best is None or row["avg_score"] > best["avg_score"]:
                        best = row

    assert best is not None

    best_params = best["candidate_params"]
    holdout = cache[CANDIDATE_PARAM_GRID.index(best_params)]
    holdout_res = _run(
        holdout,
        thresholds=best["thresholds"],
        hold_scale_by_regime=best["hold_scale_by_regime"],
        exit_min_hold_bars=best["exit_min_hold_bars"],
        exit_prob_flip_margin=best["exit_prob_flip_margin"],
        exit_quality_floor=best["exit_quality_floor"],
    )

    test_scored = _predict_direction_probs(test_df, train_df, dir_payload)
    test_scored = build_candidates(test_scored, best_params)
    test_scored = _apply_specialists(test_scored, spec_meta, model_root)
    oos_res = _run(
        test_scored,
        thresholds=best["thresholds"],
        hold_scale_by_regime=best["hold_scale_by_regime"],
        exit_min_hold_bars=best["exit_min_hold_bars"],
        exit_prob_flip_margin=best["exit_prob_flip_margin"],
        exit_quality_floor=best["exit_quality_floor"],
    )

    rows_sorted = sorted(rows, key=lambda x: x["avg_score"], reverse=True)
    out = {
        "train_csv": args.train_csv,
        "test_csv": args.test_csv,
        "dir_model": args.dir_model,
        "specialists": args.specialists,
        "active_keys": ACTIVE_KEYS,
        "best": best,
        "holdout_2025": asdict(holdout_res),
        "oos_2026": asdict(oos_res),
        "top10": rows_sorted[:10],
    }
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
