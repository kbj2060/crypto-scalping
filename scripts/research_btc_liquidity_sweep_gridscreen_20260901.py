#!/usr/bin/env python3
"""HORIZON x K grid screen + Tier0 feature analysis for BTC's own `liquidity_sweep` evidence
signal (Homer methodology port to BTC, orchestrated 2026-09-01). Scope: grid screen + feature
analysis ONLY -- no TabPFN training, no economic/cost-gate backtest, no HOLDOUT exposure this
round (see docs/experiments/btc_5m_liquidity_sweep_gridscreen_featureanalysis_20260901.md
"다음 단계").

Data: data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_
tier0.csv (built by scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py). Trigger
`bottom_liquidity_sweep`/`top_liquidity_sweep` and all Tier0 features are read as-is, NOT
recomputed here.

Hit definition (touch-based MFE, matches ETH's own liquidity_sweep convention, e.g.
research_eth_liquidity_sweep_topdown_metalabel_gridscreen_20260830.py, but using absolute-price
`atr` per this task's explicit spec -- NOT atr_pct):
  bottom candidate (bottom_liquidity_sweep==True, direction=UP):
      hit=1 if high[i+1 : i+H+1].max() >= close[i] + K*atr[i]
  top candidate (top_liquidity_sweep==True, direction=DOWN):
      hit=1 if low[i+1 : i+H+1].min()  <= close[i] - K*atr[i]

Screening metric: lift = trigger candidates' hit rate / random non-trigger bars' hit rate, same
(H,K), same side, computed on TRAIN only. VAL is used only to confirm the CHOSEN (H,K) point
generalizes (no re-search on VAL, no OOS/HOLDOUT touched).

GAP=12 is carried as a fixed, documented parameter (matches ETH's own liquidity_sweep grid
screen convention) but plays NO operational role in this script -- it is not used for
fire-clustering/dedup (unlike the ETH reference script's CLUSTER_GAP) and not used as a
TRAIN/VAL purge embargo (the date-disjoint split already needs none, per task spec). Every
trigger-fire row is treated as an independent candidate. This is a deliberate simplification
vs the ETH reference script, stated explicitly here and in the output doc -- not silent.

Split (by each candidate's OWN bar timestamp, fresh-forward date convention):
  TRAIN   : timestamp <  2025-09-01
  VAL     : 2025-09-01 <= timestamp < 2026-01-01
  (OOS/HOLDOUT rows are loaded ONLY as forward-looking context so late-VAL candidates' own
   H-bar-ahead outcome windows have data to read -- they are NEVER selected as their own
   candidate/baseline set and NEVER scored/reported. Rows with timestamp >= 2026-04-01
   (HOLDOUT) are dropped immediately after load and never touched again.)

Run with: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_liquidity_sweep_gridscreen_20260901.py
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/liquidity_sweep_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")  # never touched past this point

HORIZON_GRID = [15, 20, 25, 30, 40, 50]
K_GRID = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
GAP = 12  # fixed, documented convention only -- see module docstring, NOT used operationally

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

MIN_TRAIN_CANDIDATES = 300  # "a few hundred+" per task spec
MIN_TRAIN_HITS = 30  # extra stability gate: avoid corners with huge-but-noisy lift from tiny hit counts
RNG_SEED = 20260901


def log(msg: str) -> None:
    print(f"[btc_liq_sweep_gridscreen] {msg}", flush=True)


def load_data() -> pd.DataFrame:
    usecols = sorted(set(
        ["timestamp", "open", "high", "low", "close", "atr",
         "bottom_liquidity_sweep", "top_liquidity_sweep"] + TIER0_FEATURES
    ))
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    assert df["timestamp"].max() < HOLDOUT_START, "HOLDOUT row leaked past truncation"
    return df


def fwd_extreme(pos_idx: np.ndarray, arr: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """For each position i in pos_idx, the extreme of `arr` over (i+1 .. i+horizon]. mode='max'
    or 'min'. NaN where i+horizon is out of bounds (not enough future bars)."""
    n = len(arr)
    out = np.full(len(pos_idx), np.nan)
    for k, i in enumerate(pos_idx):
        j = i + horizon
        if j >= n:
            continue
        window = arr[i + 1:j + 1]
        out[k] = window.max() if mode == "max" else window.min()
    return out


def screen_side(df: pd.DataFrame, side: str, split_mask: np.ndarray, horizon: int,
                 rng: np.random.Generator) -> dict:
    """One (side, horizon) screening pass: candidate hit-rate vs random-non-trigger-bar hit-rate,
    across the full K_GRID, on the rows selected by split_mask. Returns raw arrays needed to
    compute hit-rate/lift per K (candidate & baseline entry price + atr + fwd extreme)."""
    trig_col = f"{side}_liquidity_sweep"
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    n = len(df)

    elig = split_mask & df["atr"].notna().to_numpy() & (atr > 0) & df["close"].notna().to_numpy()
    trig = df[trig_col].fillna(False).to_numpy()
    cand_pool = np.flatnonzero(elig & trig)
    noncand_pool = np.flatnonzero(elig & ~trig)
    noncand_pool = noncand_pool[noncand_pool + horizon < n]  # needs a full forward window too

    mode = "max" if side == "bottom" else "min"
    cand_ext = fwd_extreme(cand_pool, high if side == "bottom" else low, horizon, mode)
    valid = ~np.isnan(cand_ext)
    cand_idx = cand_pool[valid]
    cand_ext = cand_ext[valid]

    n_base = min(len(cand_idx), len(noncand_pool))
    base_idx = rng.choice(noncand_pool, size=n_base, replace=False) if n_base > 0 else np.array([], dtype=int)
    base_ext = fwd_extreme(base_idx, high if side == "bottom" else low, horizon, mode)
    base_valid = ~np.isnan(base_ext)
    base_idx = base_idx[base_valid]
    base_ext = base_ext[base_valid]

    return {
        "cand_idx": cand_idx, "cand_ext": cand_ext, "cand_close": close[cand_idx], "cand_atr": atr[cand_idx],
        "base_idx": base_idx, "base_ext": base_ext, "base_close": close[base_idx], "base_atr": atr[base_idx],
    }


def hit_rate(ext: np.ndarray, entry: np.ndarray, atr: np.ndarray, k: float, side: str) -> np.ndarray:
    if side == "bottom":
        return (ext - entry) >= k * atr
    return (entry - ext) >= k * atr


def build_grid_rows(pack: dict, side: str, horizon: int) -> list[dict]:
    rows = []
    n_cand = len(pack["cand_idx"])
    n_base = len(pack["base_idx"])
    for k in K_GRID:
        cand_hit = hit_rate(pack["cand_ext"], pack["cand_close"], pack["cand_atr"], k, side)
        base_hit = hit_rate(pack["base_ext"], pack["base_close"], pack["base_atr"], k, side)
        cand_rate = float(cand_hit.mean()) if n_cand else float("nan")
        base_rate = float(base_hit.mean()) if n_base else float("nan")
        lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")
        rows.append({
            "side": side, "horizon": horizon, "k": k,
            "n_cand": n_cand, "n_cand_hits": int(cand_hit.sum()),
            "n_base": n_base, "n_base_hits": int(base_hit.sum()),
            "cand_hit_rate": round(cand_rate, 4), "base_hit_rate": round(base_rate, 4),
            "lift": round(lift, 4) if np.isfinite(lift) else None,
        })
    return rows


def main() -> int:
    log("loading BTC Tier0 candidate CSV...")
    df = load_data()
    log(f"{len(df)} rows loaded, {df['timestamp'].min()} -> {df['timestamp'].max()} (HOLDOUT never loaded)")

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    log(f"TRAIN rows={train_mask.sum()} VAL rows={val_mask.sum()}")

    rng = np.random.default_rng(RNG_SEED)

    # ---- TRAIN grid screen (both sides, full HORIZON x K grid) ----
    grid_rows: list[dict] = []
    train_packs: dict[tuple[str, int], dict] = {}
    for side in ("bottom", "top"):
        for horizon in HORIZON_GRID:
            pack = screen_side(df, side, train_mask, horizon, rng)
            train_packs[(side, horizon)] = pack
            rows = build_grid_rows(pack, side, horizon)
            grid_rows.extend(rows)
            best_k_row = max(rows, key=lambda r: (r["lift"] if r["lift"] is not None else -1))
            log(f"  TRAIN side={side:6s} H={horizon:>3d}: n_cand={len(pack['cand_idx']):>5d} "
                f"best_lift={best_k_row['lift']} @K={best_k_row['k']}")

    grid_df = pd.DataFrame(grid_rows)

    # ---- choose (HORIZON, K): joint bottom+top score = min(lift_bottom, lift_top), gated by
    # candidate count AND hit count on both sides (avoid noisy grid corners) ----
    pivot_bottom = grid_df[grid_df["side"] == "bottom"].set_index(["horizon", "k"])
    pivot_top = grid_df[grid_df["side"] == "top"].set_index(["horizon", "k"])
    candidates_for_choice = []
    for horizon in HORIZON_GRID:
        for k in K_GRID:
            b = pivot_bottom.loc[(horizon, k)]
            t = pivot_top.loc[(horizon, k)]
            if b["n_cand"] < MIN_TRAIN_CANDIDATES or t["n_cand"] < MIN_TRAIN_CANDIDATES:
                continue
            if b["n_cand_hits"] < MIN_TRAIN_HITS or t["n_cand_hits"] < MIN_TRAIN_HITS:
                continue
            if b["lift"] is None or t["lift"] is None:
                continue
            joint = min(b["lift"], t["lift"])
            candidates_for_choice.append((joint, horizon, k, b["lift"], t["lift"]))

    assert candidates_for_choice, "no (H,K) combo passed the frequency/stability gates"
    candidates_for_choice.sort(key=lambda x: x[0], reverse=True)
    joint_best, CHOSEN_H, CHOSEN_K, chosen_lift_bottom, chosen_lift_top = candidates_for_choice[0]
    log(f"\n=== CHOSEN (HORIZON={CHOSEN_H}, K={CHOSEN_K}): "
        f"TRAIN lift bottom={chosen_lift_bottom} top={chosen_lift_top} joint(min)={joint_best} ===")

    log("\nTop 8 (H,K) combos by joint(min) TRAIN lift, gated:")
    for joint, h, k, lb, lt in candidates_for_choice[:8]:
        log(f"  H={h:>3d} K={k:.1f}: bottom={lb:.3f} top={lt:.3f} joint={joint:.3f}")

    # ---- VAL confirmation at CHOSEN (H,K) only (no re-search) ----
    val_rows = []
    val_packs = {}
    for side in ("bottom", "top"):
        pack = screen_side(df, side, val_mask, CHOSEN_H, rng)
        val_packs[side] = pack
        rows = build_grid_rows(pack, side, CHOSEN_H)
        row_at_k = next(r for r in rows if r["k"] == CHOSEN_K)
        val_rows.append(row_at_k)
        log(f"  VAL side={side:6s} H={CHOSEN_H} K={CHOSEN_K}: n_cand={row_at_k['n_cand']} "
            f"lift={row_at_k['lift']} cand_hit_rate={row_at_k['cand_hit_rate']} base_hit_rate={row_at_k['base_hit_rate']}")

    train_rows_chosen = [
        r for r in grid_rows if r["horizon"] == CHOSEN_H and r["k"] == CHOSEN_K
    ]

    # ---- feature analysis at CHOSEN (H,K): TRAIN candidates -> hit label ----
    feature_analysis = {}
    for side in ("bottom", "top"):
        pack = train_packs[(side, CHOSEN_H)]
        cand_idx = pack["cand_idx"]
        hit = hit_rate(pack["cand_ext"], pack["cand_close"], pack["cand_atr"], CHOSEN_K, side).astype(int)
        feat_df = df.loc[cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        feat_df["hit"] = hit

        # (a) point-biserial correlation (== Pearson corr of binary hit vs each continuous feature)
        corr = feat_df.corr(numeric_only=True)["hit"].drop("hit").sort_values(key=lambda s: s.abs(), ascending=False)

        # (b) HistGradientBoostingClassifier fit on TRAIN candidates, permutation importance on VAL candidates
        clf = HistGradientBoostingClassifier(random_state=RNG_SEED)
        X_train = feat_df[TIER0_FEATURES]
        y_train = feat_df["hit"].to_numpy()
        clf.fit(X_train, y_train)
        train_auc = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])

        val_pack = val_packs[side]
        val_cand_idx = val_pack["cand_idx"]
        val_hit = hit_rate(val_pack["cand_ext"], val_pack["cand_close"], val_pack["cand_atr"], CHOSEN_K, side).astype(int)
        val_feat_df = df.loc[val_cand_idx, TIER0_FEATURES].reset_index(drop=True).copy()
        X_val = val_feat_df[TIER0_FEATURES]
        y_val = val_hit
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")

        perm = permutation_importance(clf, X_val, y_val, scoring="roc_auc", n_repeats=20,
                                       random_state=RNG_SEED) if len(np.unique(y_val)) > 1 else None
        if perm is not None:
            perm_series = pd.Series(perm.importances_mean, index=TIER0_FEATURES).sort_values(key=np.abs, ascending=False)
            perm_std = pd.Series(perm.importances_std, index=TIER0_FEATURES)
        else:
            perm_series = pd.Series(dtype=float)
            perm_std = pd.Series(dtype=float)

        log(f"\n=== Feature analysis side={side} H={CHOSEN_H} K={CHOSEN_K} "
            f"n_train_cand={len(cand_idx)} n_val_cand={len(val_cand_idx)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} ===")
        log("  top corr (abs, desc): " + ", ".join(f"{f}={corr[f]:+.3f}" for f in corr.index[:8]))
        if len(perm_series):
            log("  top perm-importance (VAL, desc): " + ", ".join(
                f"{f}={perm_series[f]:+.4f}(+-{perm_std[f]:.4f})" for f in perm_series.index[:8]))

        feature_analysis[side] = {
            "n_train_candidates": int(len(cand_idx)),
            "n_val_candidates": int(len(val_cand_idx)),
            "train_hit_rate": round(float(y_train.mean()), 4),
            "val_hit_rate": round(float(np.mean(y_val)), 4) if len(y_val) else None,
            "gbm_train_auc": round(float(train_auc), 4),
            "gbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "point_biserial_corr_train": {f: round(float(corr[f]), 4) for f in corr.index},
            "permutation_importance_val_mean": {f: round(float(perm_series[f]), 5) for f in perm_series.index} if len(perm_series) else {},
            "permutation_importance_val_std": {f: round(float(perm_std[f]), 5) for f in perm_series.index} if len(perm_series) else {},
        }

    report = {
        "asset": "BTCUSDT", "signal": "liquidity_sweep", "bar": "5m",
        "data_path": str(DATA_PATH),
        "rows_loaded": int(len(df)),
        "date_range_used": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "holdout_start_never_touched": str(HOLDOUT_START),
        "split": {"train_end_excl": str(VAL_START), "val_start_incl": str(VAL_START), "val_end_excl": str(OOS_START)},
        "horizon_grid": HORIZON_GRID, "k_grid": K_GRID, "gap_fixed_documented_only": GAP,
        "hit_formula": "bottom: high[i+1:i+H+1].max()>=close[i]+K*atr[i]; top: low[i+1:i+H+1].min()<=close[i]-K*atr[i] (atr=absolute price ATR14, not atr_pct)",
        "selection_rule": "argmax over (H,K) of min(train_lift_bottom, train_lift_top), gated on n_cand>=%d and n_hits>=%d both sides" % (MIN_TRAIN_CANDIDATES, MIN_TRAIN_HITS),
        "chosen_horizon": CHOSEN_H, "chosen_k": CHOSEN_K,
        "chosen_train_lift": {"bottom": chosen_lift_bottom, "top": chosen_lift_top, "joint_min": joint_best},
        "chosen_val_confirmation": val_rows,
        "chosen_train_rows": train_rows_chosen,
        "full_train_grid": grid_rows,
        "top8_train_combos_by_joint_lift": [
            {"horizon": h, "k": k, "lift_bottom": lb, "lift_top": lt, "joint_min": j}
            for j, h, k, lb, lt in candidates_for_choice[:8]
        ],
        "feature_analysis": feature_analysis,
        "tier0_features": TIER0_FEATURES,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_fresh_forward": "This is a grid-screen/feature-analysis pass (label separability check), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar is N/A=False by construction, no trade ledger exists yet.",
        "cross_asset_info_used": False,
        "cross_asset_note": "smt_divergence-style BTC-confirms-ETH cross-asset info was NOT used -- liquidity_sweep is a single-asset (BTC-only OHLC) signal by definition, matches ETH's own liquidity_sweep methodology.",
        "tabpfn_training_done": False,
        "economic_cost_gate_done": False,
        "holdout_exposure_done": False,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    log(f"\nreport saved -> {OUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
