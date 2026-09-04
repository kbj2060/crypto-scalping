#!/usr/bin/env python3
"""Re-label taker_delta_z_climax fires by REALIZED TRAILING-STOP OUTCOME instead of touch-based
MFE, to test a concern the user raised after the trailing-stop cost-gate breakthrough
(memory: eth_taker_delta_climax_trailing_stop_costgate_breakthrough_20260830.md): the v4
touch-based label (hit = MFE_pct >= 2.0xATR within 2h, see
research_eth_taker_delta_climax_metalabel_tabpfn_20260829.py) is NOT the same thing as "this
became a genuinely money-making trade" -- and a direct check already confirmed the gap is real:
corr(v4 model_proba, realized trailing pnl) was ~0 (-0.02 overall) on VAL+OOS. A model whose
confidence doesn't track trade quality can't be used for confidence-based filtering/sizing, even
though the *unfiltered* trailing-stop cost-gate already cleared standard cost (+5.17bp VAL+OOS,
SL=2.0/ARM=1.5/Trail=0.2xATR, dual optimistic/pessimistic intrabar-ordering verified).

This script keeps EVERYTHING else about the v4 pipeline fixed (same fires, same 23 Tier0
features, same TRAIN<2025-09-01/VAL/OOS split, same TabPFN 4-seed panel) and changes ONLY the
label:
    hit = 1 if realized trailing-stop pnl (SL_init=2.0xATR, ARM=1.5xATR, Trail=0.2xATR, HORIZON=24
    bar timeout -- the exact validated config from the breakthrough memo) exceeds the standard
    10bp round-trip cost, else 0.
This directly targets "was this fire, traded with the validated exit, an economically winning
trade" instead of "did price merely touch a level." If the model's proba now correlates with
realized pnl OUT OF SAMPLE (TRAIN-fit predicting VAL/OOS, never seen during training -- not a
tautology, since this is fire-time-features -> future-outcome exactly like v4), that would be real
evidence the relabel closes the gap; if not, the earlier correlation-null finding was about the
outcome-generating mechanism itself (order-flow climax fires may be inherently hard to hand-pick
even by their own trailing pnl), not an artifact of the touch-based label specifically.

HOLDOUT (2026-04-01~) is excluded entirely from this script -- single-touch policy, not yet earned
for this candidate.

Runs on the GPU server (quant_ai env, CUDA required for TabPFN).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

FIRES_CSV = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829/eth_5m_taker_delta_climax_metalabel_features.csv"
KLINES_PATH = ROOT / "binance_data/klines/ETHUSDT/ETHUSDT-5m-api.csv"
OUT_DIR = ROOT / "data/labels/eth_5m_taker_delta_climax_metalabel_20260829"
REPORT_DIR = ROOT / "tmp/eth_taker_delta_climax_trailing_relabel_tabpfn_20260830"

HORIZON = 24  # matches the fires CSV's own v4 outcome window -- trailing sim uses this as its timeout
SL_INIT, ARM, TRAIL = 2.0, 1.5, 0.2  # validated config, see breakthrough memo (dual-verified)
COST_BP = 10.0  # standard round-trip cost (feedback_no_fee_discount_assumptions_...)

VAL_START = pd.Timestamp("2025-09-01")
OOS_START = pd.Timestamp("2026-01-01")
HOLDOUT_START = pd.Timestamp("2026-04-01")  # excluded entirely -- single-touch policy

SEEDS = [20260829, 141592, 271828, 577215]  # same 4 seeds as v4 / V_REBOUND

FEATURE_COLUMNS = [
    "is_bottom", "delta_z", "atr_pct", "atr_percentile_864", "hour_utc", "weekday", "nyse_open_flag",
    "p_fast", "p_slow", "ret3_z", "vwap_dev_z", "cvd_roll_roc_48",
    "vol_z", "lower_wick_ratio", "upper_wick_ratio", "bb_pctb",
    "adx14", "pdi", "ndi", "bb_width_pctile", "er_24", "realized_vol_ratio",
    "rsi",
]


def log(msg: str) -> None:
    print(f"[trailing_relabel] {msg}", flush=True)


def simulate_trailing(idx: int, side: str, entry: float, atr: float,
                       high: np.ndarray, low: np.ndarray, close: np.ndarray,
                       sl_init: float, arm: float, trail: float) -> float:
    """Verbatim from scratchpad research_taker_delta_climax_trailing_stop_20260830.py -- the exact
    simulation that discovered/validated the +5.17bp VAL+OOS breakthrough. Returns realized
    pnl_pct (fraction, favorable-direction-normalized, gross of cost)."""
    is_bottom = side == "bottom"
    stop = entry - sl_init * atr if is_bottom else entry + sl_init * atr
    peak = entry
    armed = False
    for b in range(idx + 1, idx + HORIZON + 1):
        bh, bl = high[b], low[b]
        if is_bottom:
            if bl <= stop:
                return (stop - entry) / entry
            if bh > peak:
                peak = bh
                if not armed and (peak - entry) >= arm * atr:
                    armed = True
                if armed:
                    stop = max(stop, peak - trail * atr)
        else:
            if bh >= stop:
                return (entry - stop) / entry
            if bl < peak:
                peak = bl
                if not armed and (entry - peak) >= arm * atr:
                    armed = True
                if armed:
                    stop = min(stop, peak + trail * atr)
    end_close = close[idx + HORIZON]
    return (end_close - entry) / entry if is_bottom else (entry - end_close) / entry


def evaluate(proba: np.ndarray, y: np.ndarray) -> dict:
    pred = (proba >= 0.5).astype(int)
    naive_pred = np.full_like(y, np.bincount(y).argmax())
    return {
        "auc": round(float(roc_auc_score(y, proba)), 4),
        "accuracy": round(float((pred == y).mean()), 4),
        "balanced_accuracy": round(float(balanced_accuracy_score(y, pred)), 4),
        "naive_majority_accuracy": round(float((naive_pred == y).mean()), 4),
    }


def run_tabpfn_panel(train: pd.DataFrame, eval_df: pd.DataFrame, feature_cols: list[str],
                      label_col: str, tag: str) -> tuple[dict, np.ndarray]:
    from tabpfn import TabPFNClassifier
    seed_rows = []
    proba_by_seed = []
    for seed in SEEDS:
        clf = TabPFNClassifier(device="cuda", random_state=seed)
        clf.fit(train[feature_cols], train[label_col].to_numpy().astype(int))
        proba = clf.predict_proba(eval_df[feature_cols])[:, 1]
        proba_by_seed.append(proba)
        r = evaluate(proba, eval_df[label_col].to_numpy().astype(int))
        r["seed"] = seed
        seed_rows.append(r)
        log(f"  [{tag}] seed={seed}: auc={r['auc']:.4f} acc={r['accuracy']:.4f} "
            f"bal_acc={r['balanced_accuracy']:.4f} (naive={r['naive_majority_accuracy']:.4f})")
    table = pd.DataFrame(seed_rows)
    mean_proba = np.mean(proba_by_seed, axis=0)
    result = {
        "n_train": int(len(train)), "n_eval": int(len(eval_df)),
        "auc_mean": round(float(table["auc"].mean()), 4), "auc_std": round(float(table["auc"].std(ddof=1)), 4),
        "accuracy_mean": round(float(table["accuracy"].mean()), 4),
        "balanced_accuracy_mean": round(float(table["balanced_accuracy"].mean()), 4),
        "naive_majority_accuracy": seed_rows[0]["naive_majority_accuracy"],
        "per_seed": seed_rows,
    }
    return result, mean_proba


def proba_pnl_diagnostics(mean_proba: np.ndarray, pnl_bp: np.ndarray, tag: str) -> dict:
    """The actual point of this experiment: does the NEW label's out-of-sample confidence track
    realized trade quality (unlike v4's ~0 correlation)? Also checks whether filtering the
    unfiltered trailing-stop population by proba improves on its already-positive baseline."""
    corr = float(np.corrcoef(mean_proba, pnl_bp)[0, 1])
    order = np.argsort(-mean_proba)
    sorted_pnl = pnl_bp[order]
    fracs = [1.0, 0.75, 0.5, 0.33, 0.25]
    filtered = {}
    for frac in fracs:
        k = max(1, int(len(sorted_pnl) * frac))
        top = sorted_pnl[:k]
        filtered[f"top_{int(frac * 100)}pct"] = {
            "n": int(k), "gross_bp": round(float(top.mean()), 2),
            "net_bp": round(float(top.mean() - COST_BP), 2),
        }
    log(f"  [{tag}] corr(mean_proba, trailing_pnl_bp) = {corr:+.4f}")
    for k, v in filtered.items():
        log(f"    {k}: n={v['n']} gross={v['gross_bp']:+.2f}bp net={v['net_bp']:+.2f}bp")
    return {"corr_proba_pnl": round(corr, 4), "proba_filtered_expectancy": filtered}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    log("loading fires+features CSV (v4, reused verbatim except label)...")
    fires = pd.read_csv(FIRES_CSV, parse_dates=["timestamp"])
    fires = fires.loc[fires["timestamp"] < HOLDOUT_START].reset_index(drop=True)  # holdout excluded entirely

    log("loading klines...")
    klines = pd.read_csv(KLINES_PATH, parse_dates=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    high = klines["high"].to_numpy()
    low = klines["low"].to_numpy()
    close = klines["close"].to_numpy()

    log(f"simulating trailing stop (SL={SL_INIT} ARM={ARM} Trail={TRAIL}) for {len(fires)} fires...")
    pnl_frac = np.empty(len(fires))
    for i, row in enumerate(fires.itertuples(index=False)):
        entry_px = close[row.pos]
        atr = row.atr_pct * entry_px
        pnl_frac[i] = simulate_trailing(row.pos, row.side, entry_px, atr, high, low, close, SL_INIT, ARM, TRAIL)
    fires["trailing_pnl_bp"] = pnl_frac * 10000
    fires["hit_v4_touch"] = fires["hit"].astype(int)  # keep old label for reference/comparison
    fires["hit"] = (fires["trailing_pnl_bp"] > COST_BP).astype(int)  # NEW label: net-of-cost winner

    gross_pos_rate = float((fires["trailing_pnl_bp"] > 0).mean())
    net_pos_rate = float(fires["hit"].mean())
    log(f"trailing outcome over {len(fires)} fires (TRAIN+VAL+OOS): "
        f"gross>0 rate={gross_pos_rate:.4f}, net>{COST_BP}bp rate={net_pos_rate:.4f}")
    log(f"v4-touch vs new-label agreement: {(fires['hit_v4_touch'] == fires['hit']).mean():.4f} "
        f"(corr={fires['hit_v4_touch'].corr(fires['hit']):.4f})")

    fires = fires.dropna(subset=FEATURE_COLUMNS + ["hit"]).reset_index(drop=True)

    ts = fires["timestamp"]
    train = fires.loc[ts < VAL_START].reset_index(drop=True)
    val = fires.loc[(ts >= VAL_START) & (ts < OOS_START)].reset_index(drop=True)
    oos = fires.loc[(ts >= OOS_START) & (ts < HOLDOUT_START)].reset_index(drop=True)
    log(f"TRAIN(<{VAL_START.date()}) n={len(train)}, VAL n={len(val)}, OOS n={len(oos)} (HOLDOUT excluded)")
    log(f"TRAIN hit rate={train['hit'].mean():.4f}, VAL={val['hit'].mean():.4f}, OOS={oos['hit'].mean():.4f}")

    fires.drop(columns=["hit_v4_touch"]).to_csv(
        OUT_DIR / "eth_5m_taker_delta_climax_trailing_relabel_features_20260830.csv", index=False)

    log("=== VAL evaluation (TRAIN-fit, 4 seeds) ===")
    val_result, val_proba = run_tabpfn_panel(train, val, FEATURE_COLUMNS, "hit", "VAL")
    log(f"VAL -> AUC {val_result['auc_mean']:.4f}+/-{val_result['auc_std']:.4f}")

    log("=== OOS evaluation (TRAIN-fit, 4 seeds) ===")
    oos_result, oos_proba = run_tabpfn_panel(train, oos, FEATURE_COLUMNS, "hit", "OOS")
    log(f"OOS -> AUC {oos_result['auc_mean']:.4f}+/-{oos_result['auc_std']:.4f}")

    log("=== proba vs realized trailing pnl diagnostics (the actual point of this experiment) ===")
    val_diag = proba_pnl_diagnostics(val_proba, val["trailing_pnl_bp"].to_numpy(), "VAL")
    oos_diag = proba_pnl_diagnostics(oos_proba, oos["trailing_pnl_bp"].to_numpy(), "OOS")
    combined_proba = np.concatenate([val_proba, oos_proba])
    combined_pnl = np.concatenate([val["trailing_pnl_bp"].to_numpy(), oos["trailing_pnl_bp"].to_numpy()])
    combined_diag = proba_pnl_diagnostics(combined_proba, combined_pnl, "VAL+OOS")

    baseline_val_net = round(float(val["trailing_pnl_bp"].mean() - COST_BP), 2)
    baseline_oos_net = round(float(oos["trailing_pnl_bp"].mean() - COST_BP), 2)
    baseline_combined_net = round(float(combined_pnl.mean() - COST_BP), 2)
    log(f"unfiltered (all-fires) baseline net-of-cost expectancy: VAL={baseline_val_net:+.2f}bp "
        f"OOS={baseline_oos_net:+.2f}bp VAL+OOS={baseline_combined_net:+.2f}bp "
        f"(reference: breakthrough memo VAL=+4.64 OOS=+5.93 combined=+5.17)")

    report = {
        "signal": "taker_delta_z_climax",
        "experiment": "trailing_stop_realized_outcome_relabel",
        "config": {"sl_init": SL_INIT, "arm": ARM, "trail": TRAIL, "horizon_bars": HORIZON, "cost_bp": COST_BP},
        "label_definition": f"hit = 1 if realized trailing-stop pnl > {COST_BP}bp (net-of-cost winner), else 0",
        "feature_columns": FEATURE_COLUMNS,
        "n_fires_total_excl_holdout": int(len(fires)),
        "gross_positive_rate": round(gross_pos_rate, 4),
        "net_of_cost_positive_rate": round(net_pos_rate, 4),
        "unfiltered_baseline_net_bp": {"val": baseline_val_net, "oos": baseline_oos_net, "combined": baseline_combined_net},
        "val": val_result, "oos": oos_result,
        "proba_pnl_diagnostics": {"val": val_diag, "oos": oos_diag, "combined": combined_diag},
    }
    out_path = REPORT_DIR / "report.json"
    out_path.write_text(json.dumps(report, indent=2, default=str))
    log(f"report saved -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
