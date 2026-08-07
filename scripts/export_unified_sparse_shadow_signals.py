from __future__ import annotations

import argparse
import json
import os
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CSV = "/home/llewyn/crypto-scalping/data/rl_training_2025_unified_dircat_oof.csv"
DEFAULT_RULE = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_rule.json"
DEFAULT_OUT = "/home/llewyn/crypto-scalping/data/ensemble/supervised/unified_sparse_shadow_signals.csv"


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


def main() -> None:
    ap = argparse.ArgumentParser(description="Export sparse shadow signals from unified direction probabilities")
    ap.add_argument("--csv-path", default=DEFAULT_CSV)
    ap.add_argument("--rule-path", default=DEFAULT_RULE)
    ap.add_argument("--output-path", default=DEFAULT_OUT)
    args = ap.parse_args()

    with open(args.rule_path, "r", encoding="utf-8") as f:
        rule_cfg = json.load(f)

    df = pd.read_csv(args.csv_path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)

    out = _build_candidates(df, rule_cfg["candidate_params"])
    veto = rule_cfg["veto_rule"]
    allowed = []
    signal = []
    for flag, side, regime in zip(
        pd.to_numeric(out["ud_cand_flag"], errors="coerce").fillna(0).astype(np.int8).to_numpy(),
        pd.to_numeric(out["ud_cand_side"], errors="coerce").fillna(0).astype(np.int8).to_numpy(),
        out["ud_cand_regime"].astype(str).to_numpy(),
    ):
        mode = veto.get(regime, "skip")
        ok = bool(
            flag == 1
            and (
                mode == "both"
                or (mode == "long" and side == 1)
                or (mode == "short" and side == -1)
            )
        )
        allowed.append(int(ok))
        signal.append("LONG" if ok and side == 1 else "SHORT" if ok and side == -1 else "FLAT")

    out["ud_shadow_take"] = np.asarray(allowed, dtype=np.int8)
    out["ud_shadow_signal"] = np.asarray(signal, dtype=object)
    out["ud_shadow_hold_scale"] = float(rule_cfg["hold_scale"])
    out["ud_shadow_close_on_opp"] = int(bool(rule_cfg["close_on_opp"]))

    cols = [
        "timestamp",
        "close",
        "ud_cat_long_prob",
        "ud_cat_flat_prob",
        "ud_cat_short_prob",
        "ud_cat_edge",
        "ud_cand_flag",
        "ud_cand_side",
        "ud_cand_hold",
        "ud_cand_quality",
        "ud_cand_raw_edge",
        "ud_cand_sup_prob_max",
        "ud_cand_regime",
        "ud_shadow_take",
        "ud_shadow_signal",
        "ud_shadow_hold_scale",
        "ud_shadow_close_on_opp",
    ]
    cols = [c for c in cols if c in out.columns]
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    out.loc[:, cols].to_csv(args.output_path, index=False)
    summary = {
        "csv_path": args.csv_path,
        "rule_path": args.rule_path,
        "output_path": args.output_path,
        "rows": int(len(out)),
        "candidate_rows": int(pd.to_numeric(out["ud_cand_flag"], errors="coerce").fillna(0).sum()),
        "shadow_take_rows": int(pd.to_numeric(out["ud_shadow_take"], errors="coerce").fillna(0).sum()),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
