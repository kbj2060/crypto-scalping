from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_OUT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_sparse_candidates_v2.csv"
DEFAULT_OUT_JSON = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_candidates_v2.json"


def _sup_side(df: pd.DataFrame) -> np.ndarray:
    lp = pd.to_numeric(df["ud_cat_long_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    sp = pd.to_numeric(df["ud_cat_short_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    fp = pd.to_numeric(df["ud_cat_flat_prob"], errors="coerce").fillna(0.0).to_numpy(np.float64)
    return np.where((lp >= sp) & (lp >= fp), 1, np.where((sp > lp) & (sp >= fp), -1, 0)).astype(np.int8)


def _regime_name(df: pd.DataFrame) -> np.ndarray:
    return (
        df[["regime_bull", "regime_bear", "regime_chop", "regime_whipsaw", "regime_normal"]]
        .idxmax(axis=1)
        .str.replace("regime_", "", regex=False)
        .to_numpy()
    )


def build_candidates(df: pd.DataFrame, params: dict[str, Any]) -> pd.DataFrame:
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
    debounce = int(params["debounce_bars"])

    for i in range(len(out)):
        if sup_side[i] == 0:
            continue
        if q[i] < float(params["quality_min"]):
            continue
        if raw_edge[i] < float(params["raw_edge_min"]):
            continue
        if sup_prob_max[i] < float(params["sup_prob_min"]):
            continue
        if bool(params.get("sign_change_only", True)) and not raw_change[i]:
            continue
        if bool(params.get("require_agreement", False)) and not agree[i]:
            continue
        if i - last_idx < debounce:
            continue
        candidate[i] = 1
        side[i] = sup_side[i]
        last_idx = i

    out["ud2_cand_flag"] = candidate
    out["ud2_cand_side"] = side
    out["ud2_cand_hold"] = hold_pred
    out["ud2_cand_quality"] = q
    out["ud2_cand_raw_edge"] = raw_edge
    out["ud2_cand_sup_prob_max"] = sup_prob_max
    out["ud2_cand_agree"] = agree.astype(np.int8)
    out["ud2_cand_raw_change"] = raw_change.astype(np.int8)
    out["ud2_cand_regime"] = regimes
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build wider sparse candidate pool for regime specialists")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--output-csv", default=DEFAULT_OUT_CSV)
    ap.add_argument("--output-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--quality-min", type=float, default=0.006)
    ap.add_argument("--raw-edge-min", type=float, default=0.55)
    ap.add_argument("--sup-prob-min", type=float, default=0.85)
    ap.add_argument("--debounce-bars", type=int, default=8)
    ap.add_argument("--sign-change-only", action="store_true", default=True)
    ap.add_argument("--require-agreement", action="store_true", default=False)
    args = ap.parse_args()

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    params = {
        "quality_min": float(args.quality_min),
        "raw_edge_min": float(args.raw_edge_min),
        "sup_prob_min": float(args.sup_prob_min),
        "debounce_bars": int(args.debounce_bars),
        "sign_change_only": bool(args.sign_change_only),
        "require_agreement": bool(args.require_agreement),
    }
    out = build_candidates(df, params)
    cand = pd.to_numeric(out["ud2_cand_flag"], errors="coerce").fillna(0).astype(np.int8)
    summary = {
        "csv_path": args.csv_path,
        "params": params,
        "rows": int(len(out)),
        "candidate_rows": int(cand.sum()),
        "candidate_rows_calib": int(((cand == 1) & (pd.to_numeric(out["ud_cat_is_holdout"], errors="coerce").fillna(0).astype(np.int8) == 0)).sum()),
        "candidate_rows_holdout": int(((cand == 1) & (pd.to_numeric(out["ud_cat_is_holdout"], errors="coerce").fillna(0).astype(np.int8) == 1)).sum()),
    }
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
