#!/usr/bin/env python3
"""HORIZON x K grid screen + Tier0 feature analysis for BTC's own `fib_extension_exhaustion`
evidence signal (Homer methodology port to BTC, part of the same 2026-09-01 orchestration as
research_btc_liquidity_sweep_gridscreen_20260901.py / research_btc_orthogonal_combo_gridscreen_
20260901.py -- this script mirrors their architecture closely so the 3 sibling reports stay
directly comparable). Scope: grid screen + feature analysis ONLY -- no TabPFN training, no
economic/cost-gate backtest, no HOLDOUT exposure this round (see docs/experiments/
btc_5m_fib_extension_exhaustion_gridscreen_featureanalysis_20260901.md "다음 단계").

Data: data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_
tier0.csv (built by scripts/build_btc_5m_evidence_signal_candidates_tier0_20260901.py). Triggers
`bottom_fib_extension_exhaustion`/`top_fib_extension_exhaustion` (already-computed causal 48-bar
leg-direction + 27.2-61.8% zone-touch beyond the leg's far extreme, see build script for the
exact derivation) and all Tier0 features are read as-is, NOT recomputed here.

Hit definition (touch-based MFE, matches ETH's own fib_extension_exhaustion convention minus the
MAE cap -- see mae_cap_bonus_check() for an optional secondary look at that refinement -- and
using absolute-price `atr` per this task's explicit spec, NOT atr_pct):
  bottom candidate (bottom_fib_extension_exhaustion==True, direction=UP):
      hit=1 if high[i+1 : i+H+1].max() >= close[i] + K*atr[i]
  top candidate (top_fib_extension_exhaustion==True, direction=DOWN):
      hit=1 if low[i+1 : i+H+1].min()  <= close[i] - K*atr[i]

Screening metric: lift = trigger candidates' hit rate / random non-trigger bars' hit rate (same
count, drawn from bars where NEITHER side's trigger fired), same (H,K), same side, computed on
TRAIN only. VAL is used only to confirm the CHOSEN (H,K) point generalizes (no re-search on VAL).

Split scope decision (matches this task's own step 4, which prescribes VAL-only confirmation, and
the 2 sibling BTC screens' resolution of the identical ambiguity -- OOS boundaries are defined
below for documentation completeness per this repo's Fresh-Forward split contract, but OOS itself
is NOT independently scored this round; reserved for a future TabPFN/economic-gate pass alongside
HOLDOUT). HOLDOUT rows are dropped immediately after load and never touched again.

GAP=12 is carried as a fixed, documented parameter (matches ETH's own fib_extension_exhaustion
grid screen, where it is the CLUSTER_GAP_MERGE burst-dedup distance) but, matching the 2 sibling
BTC screens' explicit resolution of the same "is GAP a sweep axis" question, plays NO operational
role here -- no fire-clustering/dedup is applied, every trigger-fire row is an independent
candidate. Stated explicitly, not silent (this can inflate apparent N somewhat vs a deduped count
if this signal bursts consecutively; not corrected for here, see output doc caveats).

Given this signal's low raw fire count (928 bottom / 1009 top over the full 2024-2026 history,
per build_report.json) -- the project's sparsest-firing evidence signal by a wide margin (roughly
5-8x fewer TRAIN fires than liquidity_sweep/orthogonal_combo's own BTC screens) -- every lift
number here should be read with that sample-size caveat in mind; see MIN_TRAIN_CANDIDATES/
MIN_TRAIN_HITS gates and the thin-VAL check below.

Run with: ~/anaconda3/envs/quant_ai/bin/python3 scripts/research_btc_fib_extension_exhaustion_gridscreen_20260901.py
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
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/btc_5m_evidence_signal_candidates_tier0.csv"
OUT_JSON = ROOT / "data/labels/btc_5m_evidence_signal_candidates_20260901/fib_extension_exhaustion_gridscreen_report.json"

VAL_START = pd.Timestamp("2025-09-01", tz="UTC")
OOS_START = pd.Timestamp("2026-01-01", tz="UTC")       # VAL's end boundary; OOS itself not scored this round
HOLDOUT_START = pd.Timestamp("2026-04-01", tz="UTC")    # never touched past this point

HORIZON_GRID = [12, 16, 20, 24, 30]      # 1h - 2.5h @ 5m bars, per task spec, centered on ETH's H=20
K_GRID = [1.5, 2.0, 2.35, 2.75, 3.25]    # per task spec, centered on ETH's K=2.35
GAP = 12                                  # fixed, documented convention only -- NOT used operationally
MAE_K_LOSS_MULT = 2.0                     # ETH's own MAE-cap ratio (K_loss=2.0*K, e.g. 4.70 @ K=2.35) -- bonus check only

TIER0_FEATURES = [
    "atr", "atr_percentile_864", "sweep_level_low", "sweep_level_high",
    "range_width_pct", "hour_utc", "weekday", "delta_z", "p_fast", "p_slow",
    "ret3_z", "vwap_dev_z", "cvd_roll_roc_48", "vol_z", "lower_wick_ratio",
    "upper_wick_ratio", "bb_pctb", "adx14", "pdi", "ndi", "bb_width_pctile", "rsi",
]

MIN_TRAIN_CANDIDATES = 300  # per side; "few hundred+" per task spec
MIN_TRAIN_HITS = 30         # per side; avoid grid corners with huge-but-noisy lift from tiny hit counts
MIN_VAL_FOR_PERMUTATION = 30   # per side, both classes; below this, fall back to TRAIN 5-fold CV importance
MIN_VAL_MINORITY_CLASS = 10
RNG_SEED = 20260901
Z_95 = 1.959963984540054


def log(msg: str) -> None:
    print(f"[btc_fib_ext_gridscreen] {msg}", flush=True)


def wilson_ci(hits: int, n: int, z: float = Z_95) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = hits / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * np.sqrt((p * (1 - p) / n) + (z * z / (4 * n * n)))
    return (max(0.0, center - half), min(1.0, center + half))


def load_data() -> pd.DataFrame:
    usecols = sorted(set(
        ["timestamp", "open", "high", "low", "close", "atr",
         "bottom_fib_extension_exhaustion", "top_fib_extension_exhaustion"] + TIER0_FEATURES
    ))
    df = pd.read_csv(DATA_PATH, usecols=usecols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df = df.loc[df["timestamp"] < HOLDOUT_START].reset_index(drop=True)
    assert df["timestamp"].max() < HOLDOUT_START, "HOLDOUT row leaked past truncation"
    return df


def fwd_extreme(pos_idx: np.ndarray, arr: np.ndarray, horizon: int, mode: str) -> np.ndarray:
    """For each position i in pos_idx, the extreme of `arr` over (i+1 .. i+horizon]. mode='max'
    or 'min'. NaN where i+horizon is out of bounds (not enough future bars in the loaded frame --
    the loaded frame already excludes HOLDOUT, so this also structurally prevents any HOLDOUT row
    from ever entering a computation)."""
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
    across the full K_GRID, on the rows selected by split_mask. Baseline pool excludes bars where
    EITHER side's trigger fired (a genuine "this signal is not present" background rate). Returns
    raw arrays needed to compute hit-rate/lift per K."""
    trig_col = f"{side}_fib_extension_exhaustion"
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    atr = df["atr"].to_numpy()
    n = len(df)

    elig = split_mask & df["atr"].notna().to_numpy() & (atr > 0) & df["close"].notna().to_numpy()
    trig_this_side = df[trig_col].fillna(False).to_numpy()
    any_trig = df["bottom_fib_extension_exhaustion"].fillna(False).to_numpy() | df["top_fib_extension_exhaustion"].fillna(False).to_numpy()
    cand_pool = np.flatnonzero(elig & trig_this_side)
    noncand_pool = np.flatnonzero(elig & ~any_trig)
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
        n_cand_hits = int(cand_hit.sum())
        cand_rate = float(cand_hit.mean()) if n_cand else float("nan")
        base_rate = float(base_hit.mean()) if n_base else float("nan")
        lift = cand_rate / base_rate if base_rate and base_rate > 0 else float("nan")
        ci_lo, ci_hi = wilson_ci(n_cand_hits, n_cand)
        rows.append({
            "side": side, "horizon": horizon, "k": k,
            "n_cand": n_cand, "n_cand_hits": n_cand_hits,
            "n_base": n_base, "n_base_hits": int(base_hit.sum()),
            "cand_hit_rate": round(cand_rate, 4), "cand_hit_rate_ci_lo": round(ci_lo, 4), "cand_hit_rate_ci_hi": round(ci_hi, 4),
            "base_hit_rate": round(base_rate, 4),
            "lift": round(lift, 4) if np.isfinite(lift) else None,
        })
    return rows


def train_cv_permutation_importance(X: pd.DataFrame, y: np.ndarray, feature_cols: list[str],
                                     seed: int, n_splits: int = 5) -> tuple[dict, dict]:
    """Fallback for when VAL is too thin for a direct permutation-importance read: 5-fold CV
    within TRAIN, fit on 4 folds, permutation-importance on the held-out fold, average across
    folds. Used only when flagged by the caller (see MIN_VAL_FOR_PERMUTATION)."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    per_feat = {c: [] for c in feature_cols}
    for tr_idx, te_idx in kf.split(X):
        y_tr, y_te = y[tr_idx], y[te_idx]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            continue
        clf = HistGradientBoostingClassifier(random_state=seed)
        clf.fit(X.iloc[tr_idx], y_tr)
        perm = permutation_importance(clf, X.iloc[te_idx], y_te, scoring="roc_auc", n_repeats=15, random_state=seed)
        for i, c in enumerate(feature_cols):
            per_feat[c].append(perm.importances_mean[i])
    mean_out = {c: float(np.mean(v)) for c, v in per_feat.items() if v}
    std_out = {c: float(np.std(v)) for c, v in per_feat.items() if v}
    return mean_out, std_out


def mae_cap_bonus_check(df: pd.DataFrame, side: str, cand_idx: np.ndarray, horizon: int, k: float,
                         k_loss_mult: float) -> dict:
    """Exploratory bonus only (task step: 'nice-to-have, not mandatory for first pass'). At the
    CHOSEN (H,K), what fraction of plain touch-based 'hits' would be disqualified by a whole-
    window, order-blind MAE cap (MFE>=K AND MAE<k_loss_mult*K, both measured over the same
    [i+1,i+H] window regardless of order) -- matches ETH's fib_extension_exhaustion final joint
    MFE/MAE rule (K_LOSS_MULT=2.0). Recomputes MFE/MAE directly (not reused from the plain
    screen) since MAE requires the OPPOSITE-direction forward extreme, which screen_side() does
    not compute."""
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy(); atr = df["atr"].to_numpy()
    n = len(df)
    entry = close[cand_idx]
    a = atr[cand_idx]
    fut_hi = np.array([high[i + 1:i + horizon + 1].max() if i + horizon < n else np.nan for i in cand_idx])
    fut_lo = np.array([low[i + 1:i + horizon + 1].min() if i + horizon < n else np.nan for i in cand_idx])
    valid = ~np.isnan(fut_hi) & ~np.isnan(fut_lo)
    entry, a, fut_hi, fut_lo = entry[valid], a[valid], fut_hi[valid], fut_lo[valid]
    if side == "bottom":
        mfe = fut_hi - entry
        mae = entry - fut_lo
    else:
        mfe = entry - fut_lo
        mae = fut_hi - entry
    plain_hit = mfe >= k * a
    joint_hit = plain_hit & (mae < k_loss_mult * k * a)
    n_total = int(len(entry))
    n_plain = int(plain_hit.sum())
    n_joint = int(joint_hit.sum())
    return {
        "n_candidates": n_total, "n_plain_hits": n_plain, "n_joint_hits_after_mae_cap": n_joint,
        "disqualified_by_mae_cap": n_plain - n_joint,
        "disqualified_pct_of_plain_hits": round((n_plain - n_joint) / n_plain, 4) if n_plain else None,
        "plain_hit_rate": round(n_plain / n_total, 4) if n_total else None,
        "joint_hit_rate_after_mae_cap": round(n_joint / n_total, 4) if n_total else None,
        "mae_k_loss_mult": k_loss_mult, "mae_k_loss_abs": round(k_loss_mult * k, 4),
    }


def main() -> int:
    log("loading BTC Tier0 candidate CSV...")
    df = load_data()
    log(f"{len(df)} rows loaded, {df['timestamp'].min()} -> {df['timestamp'].max()} (HOLDOUT never loaded)")

    train_mask = (df["timestamp"] < VAL_START).to_numpy()
    val_mask = ((df["timestamp"] >= VAL_START) & (df["timestamp"] < OOS_START)).to_numpy()
    log(f"TRAIN rows={train_mask.sum()} VAL rows={val_mask.sum()}")
    log(f"raw trigger fires (pre-eligibility, whole loaded frame): bottom={int(df['bottom_fib_extension_exhaustion'].sum())} "
        f"top={int(df['top_fib_extension_exhaustion'].sum())}")

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

    gate_relaxed = False
    if not candidates_for_choice:
        gate_relaxed = True
        log(f"WARNING: no (H,K) combo passed gates (n_cand>={MIN_TRAIN_CANDIDATES}, n_hits>={MIN_TRAIN_HITS} both sides); "
            f"relaxing to n_hits gate only")
        for horizon in HORIZON_GRID:
            for k in K_GRID:
                b = pivot_bottom.loc[(horizon, k)]
                t = pivot_top.loc[(horizon, k)]
                if b["lift"] is None or t["lift"] is None:
                    continue
                joint = min(b["lift"], t["lift"])
                candidates_for_choice.append((joint, horizon, k, b["lift"], t["lift"]))

    assert candidates_for_choice, "no (H,K) combo produced a finite joint lift at all"
    candidates_for_choice.sort(key=lambda x: x[0], reverse=True)
    joint_best, CHOSEN_H, CHOSEN_K, chosen_lift_bottom, chosen_lift_top = candidates_for_choice[0]
    log(f"\n=== CHOSEN (HORIZON={CHOSEN_H}, K={CHOSEN_K}): "
        f"TRAIN lift bottom={chosen_lift_bottom} top={chosen_lift_top} joint(min)={joint_best} "
        f"(gate_relaxed={gate_relaxed}) ===")

    log("\nTop 8 (H,K) combos by joint(min) TRAIN lift:")
    for joint, h, k, lb, lt in candidates_for_choice[:8]:
        log(f"  H={h:>3d} K={k:.2f}: bottom={lb:.3f} top={lt:.3f} joint={joint:.3f}")

    # ---- VAL confirmation at CHOSEN (H,K) only (no re-search) ----
    val_rows = []
    val_packs = {}
    val_thin_flags = {}
    for side in ("bottom", "top"):
        pack = screen_side(df, side, val_mask, CHOSEN_H, rng)
        val_packs[side] = pack
        rows = build_grid_rows(pack, side, CHOSEN_H)
        row_at_k = next(r for r in rows if r["k"] == CHOSEN_K)
        val_rows.append(row_at_k)
        val_thin_flags[side] = row_at_k["n_cand"] < 30
        log(f"  VAL side={side:6s} H={CHOSEN_H} K={CHOSEN_K}: n_cand={row_at_k['n_cand']} "
            f"lift={row_at_k['lift']} cand_hit_rate={row_at_k['cand_hit_rate']} base_hit_rate={row_at_k['base_hit_rate']}"
            f"{'  *** THIN VAL (<30) ***' if val_thin_flags[side] else ''}")

    train_rows_chosen = [r for r in grid_rows if r["horizon"] == CHOSEN_H and r["k"] == CHOSEN_K]

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
        mean_hit1 = feat_df.loc[feat_df["hit"] == 1, TIER0_FEATURES].mean()
        mean_hit0 = feat_df.loc[feat_df["hit"] == 0, TIER0_FEATURES].mean()

        # (b) HistGradientBoostingClassifier fit on TRAIN candidates
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
        minority_val = min((y_val == 0).sum(), (y_val == 1).sum()) if len(y_val) else 0
        val_auc = roc_auc_score(y_val, clf.predict_proba(X_val)[:, 1]) if len(np.unique(y_val)) > 1 else float("nan")

        use_val_for_importance = len(y_val) >= MIN_VAL_FOR_PERMUTATION and minority_val >= MIN_VAL_MINORITY_CLASS
        if use_val_for_importance:
            perm = permutation_importance(clf, X_val, y_val, scoring="roc_auc", n_repeats=30, random_state=RNG_SEED)
            perm_mean = dict(zip(TIER0_FEATURES, perm.importances_mean))
            perm_std = dict(zip(TIER0_FEATURES, perm.importances_std))
            importance_method = "VAL"
        else:
            log(f"  side={side}: VAL too thin for permutation importance (n={len(y_val)}, minority={minority_val}) "
                f"-> falling back to TRAIN 5-fold CV permutation importance")
            perm_mean, perm_std = train_cv_permutation_importance(X_train, y_train, TIER0_FEATURES, RNG_SEED)
            importance_method = "TRAIN_5FOLD_CV"

        perm_series = pd.Series(perm_mean).sort_values(key=np.abs, ascending=False)

        log(f"\n=== Feature analysis side={side} H={CHOSEN_H} K={CHOSEN_K} "
            f"n_train_cand={len(cand_idx)} n_val_cand={len(val_cand_idx)} "
            f"train_auc={train_auc:.4f} val_auc={val_auc:.4f} importance_method={importance_method} ===")
        log("  top |corr| (TRAIN, desc): " + ", ".join(f"{f}={corr[f]:+.3f}" for f in corr.index[:8]))
        log("  top |perm-importance| (desc): " + ", ".join(
            f"{f}={perm_series[f]:+.4f}(+-{perm_std.get(f, float('nan')):.4f})" for f in perm_series.index[:8]))

        feature_analysis[side] = {
            "n_train_candidates": int(len(cand_idx)),
            "n_val_candidates": int(len(val_cand_idx)),
            "train_hit_rate": round(float(y_train.mean()), 4),
            "val_hit_rate": round(float(np.mean(y_val)), 4) if len(y_val) else None,
            "val_minority_class_n": int(minority_val),
            "gbm_train_auc": round(float(train_auc), 4),
            "gbm_val_auc": round(float(val_auc), 4) if np.isfinite(val_auc) else None,
            "permutation_importance_method": importance_method,
            "point_biserial_corr_train": {f: round(float(corr[f]), 4) for f in corr.index},
            "mean_feature_hit1_train": {f: round(float(mean_hit1[f]), 4) for f in TIER0_FEATURES},
            "mean_feature_hit0_train": {f: round(float(mean_hit0[f]), 4) for f in TIER0_FEATURES},
            "permutation_importance_mean": {f: round(float(perm_series[f]), 5) for f in perm_series.index},
            "permutation_importance_std": {f: round(float(perm_std.get(f, float("nan"))), 5) for f in perm_series.index},
        }

    # ---- bonus: MAE-cap secondary check at CHOSEN (H,K), TRAIN only, exploratory ----
    mae_bonus = {}
    for side in ("bottom", "top"):
        pack = train_packs[(side, CHOSEN_H)]
        mae_bonus[side] = mae_cap_bonus_check(df, side, pack["cand_idx"], CHOSEN_H, CHOSEN_K, MAE_K_LOSS_MULT)
        log(f"  MAE-cap bonus side={side}: plain_hit_rate={mae_bonus[side]['plain_hit_rate']} "
            f"-> joint_hit_rate={mae_bonus[side]['joint_hit_rate_after_mae_cap']} "
            f"({mae_bonus[side]['disqualified_pct_of_plain_hits']:.1%} of plain hits disqualified)"
            if mae_bonus[side]['disqualified_pct_of_plain_hits'] is not None else "")

    report = {
        "asset": "BTCUSDT", "signal": "fib_extension_exhaustion", "bar": "5m",
        "data_path": str(DATA_PATH),
        "rows_loaded": int(len(df)),
        "date_range_used": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        "holdout_start_never_touched": str(HOLDOUT_START),
        "oos_boundaries_defined_not_scored": {"oos_start_incl": str(OOS_START), "oos_end_excl": str(HOLDOUT_START),
                                               "note": "OOS defined per this repo's Fresh-Forward split contract but NOT independently scored this round -- task step 4 prescribes VAL-only confirmation; reserved for a future TabPFN/economic-gate round"},
        "split": {"train_end_excl": str(VAL_START), "val_start_incl": str(VAL_START), "val_end_excl": str(OOS_START)},
        "horizon_grid": HORIZON_GRID, "k_grid": K_GRID, "gap_fixed_documented_only": GAP,
        "hit_formula": "bottom: high[i+1:i+H+1].max()>=close[i]+K*atr[i]; top: low[i+1:i+H+1].min()<=close[i]-K*atr[i] (atr=absolute price ATR14, not atr_pct)",
        "selection_rule": ("argmax over (H,K) of min(train_lift_bottom, train_lift_top), gated on n_cand>=%d and n_hits>=%d both sides"
                            % (MIN_TRAIN_CANDIDATES, MIN_TRAIN_HITS)),
        "selection_gate_relaxed": gate_relaxed,
        "chosen_horizon": CHOSEN_H, "chosen_k": CHOSEN_K,
        "chosen_train_lift": {"bottom": chosen_lift_bottom, "top": chosen_lift_top, "joint_min": joint_best},
        "chosen_val_confirmation": val_rows,
        "val_thin_sample_flag": val_thin_flags,
        "chosen_train_rows": train_rows_chosen,
        "full_train_grid": grid_rows,
        "top8_train_combos_by_joint_lift": [
            {"horizon": h, "k": k, "lift_bottom": lb, "lift_top": lt, "joint_min": j}
            for j, h, k, lb, lt in candidates_for_choice[:8]
        ],
        "feature_analysis": feature_analysis,
        "mae_cap_bonus_check": mae_bonus,
        "tier0_features": TIER0_FEATURES,
        "fresh_forward_bar_by_bar": False,
        "trade_ledgers_used_as_input": False,
        "saved_parent_exit_timestamps_used": False,
        "future_rows_used_for_entry": False,
        "note_fresh_forward": "This is a grid-screen/feature-analysis pass (label separability check), not a bar-by-bar TP/SL backtest -- fresh_forward_bar_by_bar is N/A=False by construction, no trade ledger exists yet.",
        "cross_asset_info_used": False,
        "cross_asset_note": "fib_extension_exhaustion is a single-asset (BTC-only OHLC) leg/zone signal by definition, no BTC-ETH cross-asset info used, matches ETH's own fib_extension_exhaustion methodology.",
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
